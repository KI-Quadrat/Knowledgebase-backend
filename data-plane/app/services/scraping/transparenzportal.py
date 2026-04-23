"""Special-case enrichment for transparenzportal.gv.at.

Transparenzportal pages hide chart data inside `<div class="t-chartDataDiv hidden">`
blocks that Crawl4AI's markdown pipeline drops (the element is hidden and contains
no rendered text). We fetch the raw HTML ourselves, extract the year/value rows,
and inject them into the scraped markdown just before the
"Auszahlungssummen in 100.000" line that the table belongs to.
"""

from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag

from app.utils.logger import get_logger

log = get_logger(__name__)

import re

TRANSPARENZPORTAL_HOST = "transparenzportal.gv.at"
CHART_DIV_CLASS = "t-chartDataDiv"
INJECTION_ANCHOR = "Auszahlungssummen in 100.000"

# Cookie notice that the portal renders into page content; collapse any run of
# whitespace between sentences so we still match if markdown line-breaks land
# in the middle of the block.
COOKIE_NOTICE_PATTERN = re.compile(
    r"Diese\s+Webseite\s+verwendet\s+Cookies\.\s*"
    r"Durch\s+das\s+Nutzen\s+dieser\s+Seite\s+sind\s+Sie\s+mit\s+der\s+Verwendung\s+von\s+Cookies\s+einverstanden\.\s*"
    r"Mehr\s+Informationen",
    re.IGNORECASE,
)


def is_transparenzportal(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
    except Exception:
        return False
    return netloc == TRANSPARENZPORTAL_HOST or netloc.endswith(f".{TRANSPARENZPORTAL_HOST}")


def _extract_chart_rows(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    chart_div = soup.find(
        "div",
        class_=lambda c: bool(c) and CHART_DIV_CLASS in (c if isinstance(c, list) else c.split()),
    )
    if not isinstance(chart_div, Tag):
        return []

    rows: list[tuple[str, str]] = []
    for row in chart_div.find_all("div", attrs={"data-label": True, "data-data": True}):
        label = (row.get("data-label") or "").strip()
        value = (row.get("data-data") or "").strip()
        if label and value:
            rows.append((label, value))
    return rows


def _format_rows(rows: list[tuple[str, str]]) -> str:
    # Sort numerically descending when labels are years; otherwise keep source order.
    try:
        rows = sorted(rows, key=lambda r: int(r[0]), reverse=True)
    except ValueError:
        pass
    return "\n".join(f"{label}: {value}" for label, value in rows)


async def fetch_chart_data(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: int = 15,
) -> str | None:
    """Fetch raw HTML and return chart rows as 'YYYY: X.XX' lines, or None.

    Uses the provided httpx client when given (reuses the Crawl4AIClient pool);
    otherwise creates a short-lived client.
    """
    try:
        if client is not None:
            resp = await client.get(url, timeout=timeout, follow_redirects=True)
            resp.raise_for_status()
            html = resp.text
        else:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as tmp:
                resp = await tmp.get(url)
                resp.raise_for_status()
                html = resp.text
    except Exception as exc:
        log.warning("transparenzportal_fetch_failed", url=url, error=str(exc))
        return None

    rows = _extract_chart_rows(html)
    if not rows:
        return None

    formatted = _format_rows(rows)
    log.info("transparenzportal_chart_extracted", url=url, row_count=len(rows), source="http")
    return formatted


def remove_cookie_notice(markdown: str) -> str:
    """Drop the portal's cookie consent banner text from the scraped markdown."""
    if not markdown:
        return markdown
    cleaned = COOKIE_NOTICE_PATTERN.sub("", markdown)
    # Collapse any blank-line runs left behind by the removal.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + ("\n" if markdown.endswith("\n") else "")


def inject_chart_data(markdown: str, chart_text: str) -> str:
    """Insert the chart block on the line immediately before the anchor phrase.

    If the anchor is not found, append the block at the end so the data isn't lost.
    """
    if not chart_text:
        return markdown

    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if INJECTION_ANCHOR in line:
            block = chart_text.splitlines()
            lines[i:i] = [*block, ""]
            return "\n".join(lines)

    return markdown.rstrip() + "\n\n" + chart_text + "\n"


async def enrich_if_applicable(
    url: str,
    markdown: str,
    *,
    html: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Entry point: no-op unless URL is transparenzportal and chart data is present.

    First tries to extract the chart div from `html` (e.g. Crawl4AI's response HTML)
    to avoid an extra request. Falls back to a fresh GET when the hidden div was
    stripped by Crawl4AI's cleaned_html pipeline or when no HTML was provided.
    """
    if not is_transparenzportal(url):
        return markdown

    markdown = remove_cookie_notice(markdown)

    if html:
        rows = _extract_chart_rows(html)
        if rows:
            formatted = _format_rows(rows)
            log.info(
                "transparenzportal_chart_extracted",
                url=url,
                row_count=len(rows),
                source="crawl_html",
            )
            return inject_chart_data(markdown, formatted)

    chart_text = await fetch_chart_data(url, client=client)
    if not chart_text:
        return markdown
    return inject_chart_data(markdown, chart_text)
