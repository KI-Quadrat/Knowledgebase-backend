import random
import re
import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import ext, settings
from app.models.common import StageUsage
from app.services import cost
from app.utils.content import clean_html, clean_markdown
from app.utils.logger import get_logger

log = get_logger(__name__)

TARGET_NOT_FOUND_TITLE = "404 Target URL not found"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
]

ACCEPT_LANGUAGES = [
    "de-AT,de;q=0.9,en;q=0.8",
    "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
    "de,en-US;q=0.9,en;q=0.8",
    "en-US,en;q=0.9",
    "*",
]


class CrawlResult:
    def __init__(
        self,
        *,
        markdown: str = "",
        html: str = "",
        title: str | None = None,
        links: list[str] | None = None,
        images: list[str] | None = None,
        success: bool = True,
        error: str | None = None,
        duration_ms: int = 0,
        scraper_used: str | None = None,
        usage: StageUsage | None = None,
    ):
        self.markdown = markdown
        self.html = html
        # Title reported directly by the backend (Jina ``data.title`` /
        # Firecrawl ``metadata.title``). The httpx fallback returns rendered
        # HTML, so the router extracts the title from there. Backends that
        # report no title leave this None and the router falls back to whatever
        # ``extract_metadata`` can pull from any available HTML.
        self.title = title
        self.links = links or []
        # Image URLs harvested by the backend (Jina X-With-Images-Summary).
        # Firecrawl and httpx leave this empty — the router runs
        # ``discover_images`` against the rendered HTML those backends return,
        # so per-page image discovery still works there without this list.
        self.images = images or []
        self.success = success
        self.error = error
        self.duration_ms = duration_ms
        # Which backend actually produced the markdown — set per-branch in
        # ``crawl()`` after a backend succeeds. ``None`` on failed results.
        self.scraper_used = scraper_used
        # Per-call billing record. Populated by each backend branch from the
        # provider's own response (Jina ``meta.usage.tokens``, Firecrawl
        # ``creditsUsed``). Self-hosted / httpx leave this as a zero entry so
        # the schema is uniform.
        self.usage = usage


class ScraperClient:
    """HTTP client wrapper for the Jina Reader, Firecrawl, and raw httpx scraping backends."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._jina_url = ext.jina_api_url.rstrip("/")
        self._jina_key = ext.jina_api_key
        self._firecrawl_url = ext.firecrawl_api_url.rstrip("/")
        self._firecrawl_key = ext.firecrawl_api_key

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.default_timeout + 10, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        log.info(
            "scraper_client_started",
            jina=bool(self._jina_key),
            firecrawl=bool(self._firecrawl_key),
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def crawl(
        self,
        url: str,
        *,
        js_render: bool = True,
        wait_for: str | None = None,
        css_selector: str | None = None,
        timeout: int | None = None,
        markdown_type: str = "fit",
        exclude_tags: list[str] | None = None,
        with_links_summary: bool = False,
        inner_img: bool = False,
        scraper: str = "jina",
    ) -> CrawlResult:
        if not self._client:
            raise RuntimeError("Client not started — call start() first")

        req_timeout = timeout or settings.default_timeout
        start = time.monotonic()

        # Build the prioritized backend list from the caller's preference. The
        # non-preferred backend (and raw httpx) remain as automatic fallbacks
        # so requests stay best-effort even when the chosen backend fails.
        # ``scraper="httpx"`` short-circuits the browser backends and goes
        # straight to the raw httpx fallback below — used by /crawl BFS where
        # Chromium-rendered fetches are wasted on per-URL link harvesting.
        # Any unrecognized value (including the legacy ``"crawl4ai"`` alias)
        # falls through to the default Jina-first chain.
        if scraper == "httpx":
            backend_order: tuple[str, ...] = ()
        elif scraper == "firecrawl":
            backend_order = ("firecrawl", "jina")
        else:
            backend_order = ("jina", "firecrawl")

        for backend in backend_order:
            # ``js_render=False`` opts out of every browser-rendered backend.
            # Both Jina's Chromium engine and Firecrawl fetch via a headless
            # browser, so the flag must skip both — otherwise they silently
            # re-introduce the Chromium cost we asked to avoid.
            if not js_render:
                continue
            if backend == "jina":
                if not self._jina_key:
                    continue
                try:
                    result = await self._scrape_with_jina(
                        url,
                        timeout=req_timeout,
                        markdown_type=markdown_type,
                        exclude_tags=exclude_tags,
                        css_selector=css_selector,
                        with_links_summary=with_links_summary,
                        inner_img=inner_img,
                    )
                    result.duration_ms = int((time.monotonic() - start) * 1000)
                    if result.success:
                        result.scraper_used = "jina"
                        log.info("jina_scrape_success", url=url, primary=(scraper == "jina"))
                        return result
                    if _is_target_not_found_error(result.error):
                        return result
                    log.warning("jina_scrape_failed", url=url, error=result.error)
                except Exception as exc:
                    log.warning("jina_scrape_error", url=url, error=str(exc))

            elif backend == "firecrawl":
                if not self._firecrawl_key:
                    continue
                try:
                    result = await self._scrape_with_firecrawl(
                        url,
                        timeout=req_timeout,
                        markdown_type=markdown_type,
                        exclude_tags=exclude_tags,
                        css_selector=css_selector,
                    )
                    result.duration_ms = int((time.monotonic() - start) * 1000)
                    if result.success:
                        result.scraper_used = "firecrawl"
                        log.info("firecrawl_scrape_success", url=url, primary=(scraper == "firecrawl"))
                        return result
                    if _is_target_not_found_error(result.error):
                        return result
                    log.warning("firecrawl_scrape_failed", url=url, error=result.error)
                except Exception as exc:
                    log.warning("firecrawl_scrape_error", url=url, error=str(exc))

        # Final fallback: Raw httpx
        result = await self._scrape_with_httpx(
            url,
            css_selector=css_selector,
            timeout=req_timeout,
            exclude_tags=exclude_tags,
        )
        result.duration_ms = int((time.monotonic() - start) * 1000)
        if result.success:
            result.scraper_used = "httpx"
            result.usage = StageUsage(stage="scraper", provider="httpx", cost_usd=0.0)
        return result

    async def _scrape_with_jina(
        self,
        url: str,
        *,
        timeout: int = 30,
        markdown_type: str = "fit",
        exclude_tags: list[str] | None = None,
        css_selector: str | None = None,
        with_links_summary: bool = False,
        inner_img: bool = False,
    ) -> CrawlResult:
        """Scrape a URL via Jina Reader API — returns Markdown directly.

        Uses the Chromium ``browser`` engine (best fidelity on JS-heavy
        municipality portals). ``X-Retain-Images: none`` keeps inline image
        markdown out of the returned content. Links summary and images summary
        are requested only when the caller actually needs them
        (``links_summary`` / ``inner_img``), since Jina is no-HTML and those
        headers drive the side lists used by the router.
        """
        headers = {
            "Authorization": f"Bearer {self._jina_key}",
            "Accept": "application/json",
            "X-Engine": "browser",
            "X-Return-Format": "markdown",
            "X-Retain-Images": "none",
        }
        if with_links_summary:
            headers["X-With-Links-Summary"] = "true"
        if inner_img:
            headers["X-With-Images-Summary"] = "true"
        if css_selector:
            headers["X-Target-Selector"] = css_selector
        if exclude_tags:
            headers["X-Remove-Selector"] = ",".join(exclude_tags)

        try:
            resp = await self._client.get(  # type: ignore[union-attr]
                f"{self._jina_url}/{url}",
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            return CrawlResult(success=False, error=f"Jina timeout after {timeout}s")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return _target_not_found_result("Jina HTTP 404 Not Found")
            return CrawlResult(success=False, error=f"Jina HTTP {exc.response.status_code}")
        except Exception as exc:
            return CrawlResult(success=False, error=f"Jina error: {exc}")

        data = resp.json()
        data_section = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
        warning = data_section.get("warning") or data.get("warning")
        if _is_target_not_found_error(warning):
            return _target_not_found_result(str(warning))
        content = data_section.get("content", "")
        title_value = data_section.get("title")
        title = title_value.strip() if isinstance(title_value, str) and title_value.strip() else None
        links = _extract_jina_links(data, url)
        images = _extract_jina_images(data, url)

        if not content.strip():
            return CrawlResult(success=False, error="Jina returned empty content")

        markdown = clean_markdown(content)
        # Jina returns ``meta.usage.tokens`` on every successful read. Newer
        # responses surface it at top level; some older / proxy variants
        # nest it under ``data.usage``. Try both before giving up.
        tokens = _extract_jina_tokens(data, data_section)
        usage = StageUsage(
            stage="scraper",
            provider="jina",
            scrape_tokens=tokens,
            cost_usd=cost.jina_cost(tokens),
        )
        return CrawlResult(
            markdown=markdown,
            html="",
            title=title,
            links=links,
            images=images,
            success=True,
            usage=usage,
        )

    async def _scrape_with_firecrawl(
        self,
        url: str,
        *,
        timeout: int = 30,
        markdown_type: str = "fit",
        exclude_tags: list[str] | None = None,
        css_selector: str | None = None,
    ) -> CrawlResult:
        """Scrape a URL via Firecrawl's ``POST /v2/scrape``.

        Requests both ``markdown`` and ``html`` formats so the router's
        ``links_summary`` HTML path can parse the full link graph (Firecrawl
        returns the rendered DOM, not a content-filtered subset). The
        ``onlyMainContent`` flag maps from ``markdown_type``: ``fit`` keeps
        article-only output, ``raw`` / ``citations`` keep the full page.
        """
        if not self._firecrawl_key:
            return CrawlResult(success=False, error="Firecrawl API key not configured")

        payload: dict = {
            "url": url,
            "formats": ["markdown", "html"],
            "onlyMainContent": markdown_type == "fit",
        }
        if exclude_tags:
            payload["excludeTags"] = list(exclude_tags)
        if css_selector:
            payload["includeTags"] = [css_selector]

        headers = {
            "Authorization": f"Bearer {self._firecrawl_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = await self._client.post(  # type: ignore[union-attr]
                f"{self._firecrawl_url}/v2/scrape",
                json=payload,
                headers=headers,
                timeout=timeout + 10,
            )
            resp.raise_for_status()
        except httpx.TimeoutException:
            return CrawlResult(success=False, error=f"Firecrawl timeout after {timeout}s")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                error = _extract_response_error(exc.response) or "Firecrawl HTTP 404 Not Found"
                return _target_not_found_result(error)
            return CrawlResult(success=False, error=f"Firecrawl HTTP {exc.response.status_code}")
        except Exception as exc:
            return CrawlResult(success=False, error=f"Firecrawl error: {exc}")

        data = resp.json()
        if not data.get("success", True):
            error = str(data.get("error") or "Firecrawl returned success=false")
            if _is_target_not_found_error(error):
                return _target_not_found_result(error)
            return CrawlResult(success=False, error=error)

        result_data = data.get("data") if isinstance(data.get("data"), dict) else {}
        markdown_value = result_data.get("markdown")
        html_value = result_data.get("html") or result_data.get("rawHtml") or ""
        metadata = result_data.get("metadata") if isinstance(result_data.get("metadata"), dict) else {}
        title_value = metadata.get("title") or metadata.get("ogTitle")
        title = title_value.strip() if isinstance(title_value, str) and title_value.strip() else None

        markdown = markdown_value if isinstance(markdown_value, str) else ""
        if not markdown.strip() and not html_value.strip():
            return CrawlResult(success=False, error="Firecrawl returned empty content")

        # ``links`` is a flat list of absolute URLs when Firecrawl populates it
        # (the v2 API includes it whenever the rendered HTML is requested).
        links_value = result_data.get("links")
        links: list[str] = []
        seen: set[str] = set()
        if isinstance(links_value, list):
            for entry in links_value:
                if isinstance(entry, str):
                    candidate = urljoin(url, entry.strip())
                    parsed = urlparse(candidate)
                    if parsed.scheme in {"http", "https"} and candidate not in seen:
                        seen.add(candidate)
                        links.append(candidate)

        credits = _extract_firecrawl_credits(data, result_data)
        usage = StageUsage(
            stage="scraper",
            provider="firecrawl",
            credits=credits,
            cost_usd=cost.firecrawl_cost(credits),
        )
        return CrawlResult(
            markdown=clean_markdown(markdown),
            html=html_value,
            title=title,
            links=links,
            success=True,
            usage=usage,
        )

    async def _map_with_firecrawl(
        self,
        url: str,
        *,
        max_pages: int,
        same_domain_only: bool,
        timeout: int,
    ) -> tuple[bool, list[str], list[dict], str | None, StageUsage | None]:
        """Discover URLs via Firecrawl's ``POST /v2/map``.

        Returns ``(success, visited, links, error, usage)``. ``usage``
        captures the credits billed by Firecrawl for the map call (one
        credit per URL returned on standard plans). Firecrawl's map
        endpoint returns a flat URL list in a single round-trip — we treat
        that list as both the visited set and the source of
        ``discovered_links``, so the caller's downstream
        ``split_documents_and_links`` still partitions pages from documents.
        """
        if not self._firecrawl_key:
            return False, [], [], "Firecrawl API key not configured", None

        payload: dict = {
            "url": url,
            "limit": max_pages,
        }
        headers = {
            "Authorization": f"Bearer {self._firecrawl_key}",
            "Content-Type": "application/json",
        }

        try:
            resp = await self._client.post(  # type: ignore[union-attr]
                f"{self._firecrawl_url}/v2/map",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            return False, [], [], f"Firecrawl map timeout after {timeout}s", None
        except httpx.HTTPStatusError as exc:
            return False, [], [], f"Firecrawl map HTTP {exc.response.status_code}", None
        except Exception as exc:
            return False, [], [], f"Firecrawl map error: {exc}", None

        if not data.get("success", True):
            return False, [], [], str(data.get("error") or "Firecrawl map success=false"), None

        raw_links = data.get("links") or []
        root_domain = urlparse(url).netloc.lower()
        visited: list[str] = []
        discovered: list[dict] = []
        seen_visited: set[str] = set()
        seen_links: set[str] = set()
        for entry in raw_links:
            # Firecrawl v2 returns either bare URL strings or
            # ``{"url": ..., "title": ...}`` objects depending on the request
            # mode. Handle both so the caller sees the same shape.
            if isinstance(entry, str):
                href = entry
                text = ""
            elif isinstance(entry, dict):
                href = entry.get("url") or entry.get("link") or ""
                text = entry.get("title") or entry.get("text") or ""
            else:
                continue
            if not isinstance(href, str) or not href.strip():
                continue
            absolute = urljoin(url, href.strip())
            parsed = urlparse(absolute)
            if parsed.scheme not in {"http", "https"}:
                continue
            if same_domain_only and parsed.netloc.lower() != root_domain:
                continue
            if absolute not in seen_visited:
                seen_visited.add(absolute)
                visited.append(absolute)
            if absolute not in seen_links:
                seen_links.add(absolute)
                discovered.append({"href": absolute, "text": text})

        # Firecrawl bills its map endpoint per URL returned. Some plans
        # report ``creditsUsed`` explicitly; fall back to ``len(visited)`` so
        # the count is at least directionally correct when missing.
        credits = _extract_firecrawl_credits(data, None)
        if credits == 0 and visited:
            credits = float(len(visited))
        usage = StageUsage(
            stage="links_map",
            provider="firecrawl",
            credits=credits,
            cost_usd=cost.firecrawl_cost(credits),
        )
        return True, visited, discovered, None, usage

    async def _scrape_with_httpx(
        self,
        url: str,
        *,
        css_selector: str | None = None,
        timeout: int = 30,
        exclude_tags: list[str] | None = None,
    ) -> CrawlResult:
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": random.choice(ACCEPT_LANGUAGES),
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

        try:
            response = await self._client.get(url, headers=headers, timeout=timeout)  # type: ignore[union-attr]
            response.raise_for_status()
        except httpx.TimeoutException:
            return CrawlResult(success=False, error=f"Timeout after {timeout}s")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return _target_not_found_result("HTTP 404 Not Found")
            return CrawlResult(success=False, error=f"HTTP {exc.response.status_code}")
        except Exception as exc:
            return CrawlResult(success=False, error=str(exc))

        content_type = response.headers.get("content-type", "")
        if "charset" not in content_type.lower():
            response.encoding = "utf-8"
        raw_html = response.text

        cleaned = clean_html(raw_html, css_selector)
        soup = BeautifulSoup(cleaned, "lxml")
        if exclude_tags:
            for selector in exclude_tags:
                for el in soup.select(selector):
                    el.decompose()
        markdown = clean_markdown(_html_to_markdown(soup))

        return CrawlResult(markdown=markdown, html=raw_html, success=True)


def _html_to_markdown(soup: BeautifulSoup) -> str:
    lines: list[str] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "td", "th", "blockquote", "pre"]):
        tag = el.name
        text = el.get_text(separator=" ", strip=True)
        if not text:
            continue
        if tag == "h1":
            lines.append(f"# {text}\n")
        elif tag == "h2":
            lines.append(f"## {text}\n")
        elif tag == "h3":
            lines.append(f"### {text}\n")
        elif tag == "h4":
            lines.append(f"#### {text}\n")
        elif tag in ("h5", "h6"):
            lines.append(f"##### {text}\n")
        elif tag == "li":
            lines.append(f"- {text}")
        elif tag == "blockquote":
            lines.append(f"> {text}\n")
        elif tag == "pre":
            lines.append(f"```\n{text}\n```\n")
        else:
            lines.append(f"{text}\n")
    return "\n".join(lines)


def _extract_jina_tokens(data: dict, data_section: dict) -> int:
    """Pull Jina's token usage off the response.

    Jina docs surface it at ``meta.usage.tokens`` (top level); some older
    builds nest it under ``data.usage.tokens``. Try both, default to 0.
    """
    for container in (data.get("meta"), data_section.get("meta"), data, data_section):
        if not isinstance(container, dict):
            continue
        usage = container.get("usage")
        if isinstance(usage, dict):
            tokens = usage.get("tokens")
            if isinstance(tokens, (int, float)) and tokens > 0:
                return int(tokens)
    return 0


def _extract_firecrawl_credits(data: dict, result_data: dict | None) -> float:
    """Pull Firecrawl's ``creditsUsed`` off the response.

    The v2 scrape envelope reports it on the top-level response or under
    ``data.metadata.creditsUsed``; the v2 map endpoint reports it on the
    top level when present. Returns 0.0 when the field is absent — the
    caller can fall back to a heuristic (e.g. ``len(visited)`` for map).
    """
    candidates: list[dict] = []
    if isinstance(data, dict):
        candidates.append(data)
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            candidates.append(metadata)
    if isinstance(result_data, dict):
        candidates.append(result_data)
        result_meta = result_data.get("metadata")
        if isinstance(result_meta, dict):
            candidates.append(result_meta)
    for container in candidates:
        for key in ("creditsUsed", "credits_used", "credits"):
            value = container.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
    return 0.0


def _is_target_not_found_error(value: object) -> bool:
    """True for backend signals that the target URL itself returned 404."""
    if not isinstance(value, str):
        return False
    text = value.lower()
    return (
        "404" in text and "not found" in text
    ) or text.strip() == "not found"


def _target_not_found_result(detail: str) -> CrawlResult:
    return CrawlResult(
        title=TARGET_NOT_FOUND_TITLE,
        success=False,
        error=f"{TARGET_NOT_FOUND_TITLE}: {detail}",
    )


def _extract_response_error(response: httpx.Response) -> str | None:
    try:
        data = response.json()
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("error")
    return value if isinstance(value, str) and value.strip() else None


def _extract_jina_links(data: dict, base_url: str) -> list[str]:
    """Extract Jina links, preferring the returned links_summary.urls payload."""
    data_section = data.get("data", {})
    direct_links_summary = data_section.get("links_summary")
    if not isinstance(direct_links_summary, dict):
        direct_links_summary = data.get("links_summary")

    urls: list[str] = []
    seen: set[str] = set()

    def _add_url(value: str) -> None:
        normalized = urljoin(base_url, value.strip())
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"}:
            return
        if normalized in seen:
            return
        seen.add(normalized)
        urls.append(normalized)

    if isinstance(direct_links_summary, dict):
        raw_urls = direct_links_summary.get("urls")
        if isinstance(raw_urls, list):
            for value in raw_urls:
                if isinstance(value, str) and value.strip():
                    _add_url(value)
            if urls:
                return urls

    candidates = (
        data_section.get("content"),
        data_section.get("links"),
        data_section.get("links_summary"),
        data.get("links"),
        data.get("links_summary"),
    )

    def _extract_urls_from_text(text: str) -> None:
        for match in re.findall(r"https?://[^\s<>)\\]\"']+", text):
            _add_url(match.rstrip(".,;:!?"))

        for match in re.findall(r"\[[^\]]*\]\((https?://[^)\s]+)\)", text):
            _add_url(match.rstrip(".,;:!?"))

    def _walk(value: object) -> None:
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return
            if text.startswith(("http://", "https://")) and " " not in text and "\n" not in text:
                _add_url(text)
                return
            _extract_urls_from_text(text)
            return
        if isinstance(value, dict):
            for key in ("url", "href", "link"):
                nested = value.get(key)
                if isinstance(nested, str) and nested.strip():
                    _add_url(nested)
                    return
            for nested in value.values():
                _walk(nested)
            return
        if isinstance(value, list):
            for nested in value:
                _walk(nested)

    for candidate in candidates:
        _walk(candidate)

    return urls


def _extract_jina_images(data: dict, base_url: str) -> list[str]:
    """Extract image URLs from Jina's ``data.images`` block.

    Jina returns images as a dict (``{"Image 1": "url", ...}`` or
    ``{"<alt>": "url"}``). Older / different responses may use a list of
    dicts or strings — handle both. Returns deduped, absolute http(s) URLs.
    """
    data_section = data.get("data", {})
    raw = data_section.get("images")
    if raw is None:
        raw = data.get("images")
    if raw is None:
        return []

    urls: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        normalized = urljoin(base_url, value.strip())
        parsed = urlparse(normalized)
        if parsed.scheme not in {"http", "https"}:
            return
        if normalized in seen:
            return
        seen.add(normalized)
        urls.append(normalized)

    if isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, str) and value.strip():
                _add(value)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                _add(item)
            elif isinstance(item, dict):
                for key in ("url", "src", "href"):
                    nested = item.get(key)
                    if isinstance(nested, str) and nested.strip():
                        _add(nested)
                        break

    return urls
