import random
import re
import time
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from app.config import ext, settings
from app.services.metrics import mark_crawl4ai
from app.utils.content import clean_html, clean_markdown
from app.utils.logger import get_logger

log = get_logger(__name__)

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
        links: list[str] | None = None,
        success: bool = True,
        error: str | None = None,
        duration_ms: int = 0,
    ):
        self.markdown = markdown
        self.html = html
        self.links = links or []
        self.success = success
        self.error = error
        self.duration_ms = duration_ms


class Crawl4AIClient:
    """HTTP client wrapper for external Crawl4AI with Jina Reader and httpx fallback."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url = ext.crawl4ai_url.rstrip("/")
        self._api_token = ext.crawl4ai_api_token
        self._jina_url = ext.jina_api_url.rstrip("/")
        self._jina_key = ext.jina_api_key

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.default_timeout + 10, connect=10.0),
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        log.info("crawl4ai_client_started", base_url=self._base_url, jina_fallback=bool(self._jina_key))

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def check_health(self) -> bool:
        if not self._client:
            return False
        try:
            resp = await self._client.get(f"{self._base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

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
        scraper: str = "crawl4ai",
    ) -> CrawlResult:
        if not self._client:
            raise RuntimeError("Client not started — call start() first")

        req_timeout = timeout or settings.default_timeout
        start = time.monotonic()

        # Build prioritized backend list based on client preference. The
        # non-preferred backend (and raw httpx) remain as automatic fallbacks
        # so requests stay best-effort even when the chosen backend fails.
        if scraper == "jina":
            backend_order = ("jina", "crawl4ai")
        else:
            backend_order = ("crawl4ai", "jina")

        for backend in backend_order:
            if backend == "crawl4ai":
                if not js_render:
                    continue
                try:
                    result = await self._crawl_via_api(
                        url,
                        wait_for=wait_for,
                        css_selector=css_selector,
                        timeout=req_timeout,
                        markdown_type=markdown_type,
                        exclude_tags=exclude_tags,
                    )
                    result.duration_ms = int((time.monotonic() - start) * 1000)
                    if result.success:
                        mark_crawl4ai("success", time.monotonic() - start)
                        return result
                    log.warning("crawl4ai_failed_falling_back", url=url, error=result.error)
                except Exception as exc:
                    log.warning("crawl4ai_unavailable_falling_back", url=url, error=str(exc))
                mark_crawl4ai("failed", time.monotonic() - start)

            elif backend == "jina":
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
                    )
                    result.duration_ms = int((time.monotonic() - start) * 1000)
                    if result.success:
                        log.info("jina_scrape_success", url=url, primary=(scraper == "jina"))
                        return result
                    log.warning("jina_scrape_failed", url=url, error=result.error)
                except Exception as exc:
                    log.warning("jina_scrape_error", url=url, error=str(exc))

        # Final fallback: Raw httpx
        result = await self._scrape_with_httpx(
            url,
            css_selector=css_selector,
            timeout=req_timeout,
            exclude_tags=exclude_tags,
        )
        result.duration_ms = int((time.monotonic() - start) * 1000)
        return result

    async def _crawl_via_api(
        self,
        url: str,
        *,
        wait_for: str | None = None,
        css_selector: str | None = None,
        timeout: int = 30,
        markdown_type: str = "fit",
        exclude_tags: list[str] | None = None,
    ) -> CrawlResult:
        headers: dict[str, str] = {}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        crawler_params: dict = {
            "scan_full_page": False,
            "wait_until": "domcontentloaded",
            "page_timeout": timeout * 1000,
            "delay_before_return_html": 0.5,
            "magic": True,
            "remove_overlay_elements": True,
            "cache_mode": "bypass",
        }
        if wait_for:
            crawler_params["wait_for"] = f"css:{wait_for}"
        if css_selector:
            crawler_params["css_selector"] = css_selector
        if exclude_tags:
            crawler_params["excluded_tags"] = exclude_tags
        if markdown_type == "fit":
            crawler_params["markdown_generator"] = {
                "type": "DefaultMarkdownGenerator",
                "params": {
                    "content_filter": {
                        "type": "PruningContentFilter",
                        "params": {
                            "threshold": 0.48,
                            "threshold_type": "fixed",
                            "min_word_threshold": 0,
                        },
                    },
                    "options": {"ignore_links": False, "escape_html": True},
                },
            }

        payload: dict = {
            "urls": [url],
            "browser_config": {
                "type": "BrowserConfig",
                "params": {"headless": True},
            },
            "crawler_config": {
                "type": "CrawlerRunConfig",
                "params": crawler_params,
            },
        }

        resp = await self._client.post(  # type: ignore[union-attr]
            f"{self._base_url}/crawl",
            json=payload,
            headers=headers,
            timeout=timeout + 10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results") or []
        if isinstance(results, list) and results:
            result_data = results[0]
        else:
            result_data = data.get("result", data)

        success = bool(result_data.get("success", data.get("success", False)))
        markdown = _extract_markdown(result_data.get("markdown"), preferred=markdown_type)
        html = _extract_html(result_data)
        error = _extract_error(result_data)

        return CrawlResult(markdown=clean_markdown(markdown), html=html, success=success, error=error)

    async def _scrape_with_jina(
        self,
        url: str,
        *,
        timeout: int = 30,
        markdown_type: str = "fit",
        exclude_tags: list[str] | None = None,
        css_selector: str | None = None,
        with_links_summary: bool = False,
    ) -> CrawlResult:
        """Scrape a URL via Jina Reader API — returns Markdown directly."""
        headers = {
            "Authorization": f"Bearer {self._jina_key}",
            "Accept": "application/json",
            "X-Return-Format": "markdown",
        }
        # "fit" → use readerlm-v2 engine for LLM-based main-content extraction
        # (closest Jina analogue to Crawl4AI's PruningContentFilter).
        # "raw" / "citations" → default engine (Jina has no citations mode).
        if markdown_type == "fit":
            headers["X-Engine"] = "readerlm-v2"
        if css_selector:
            headers["X-Target-Selector"] = css_selector
        if exclude_tags:
            headers["X-Remove-Selector"] = ",".join(exclude_tags)
        if with_links_summary:
            headers["X-With-Links-Summary"] = "true"

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
            return CrawlResult(success=False, error=f"Jina HTTP {exc.response.status_code}")
        except Exception as exc:
            return CrawlResult(success=False, error=f"Jina error: {exc}")

        data = resp.json()
        content = data.get("data", {}).get("content", "")
        links = _extract_jina_links(data, url)

        if not content.strip():
            return CrawlResult(success=False, error="Jina returned empty content")

        markdown = clean_markdown(content)
        return CrawlResult(markdown=markdown, html="", links=links, success=True)

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


_MARKDOWN_PRIORITY: dict[str, tuple[str, ...]] = {
    "fit": ("fit_markdown", "markdown_with_citations", "raw_markdown"),
    "citations": ("markdown_with_citations", "raw_markdown", "fit_markdown"),
    "raw": ("raw_markdown", "markdown_with_citations", "fit_markdown"),
}


def _extract_markdown(markdown_value: object, *, preferred: str = "fit") -> str:
    if isinstance(markdown_value, str):
        return markdown_value
    if isinstance(markdown_value, dict):
        keys = _MARKDOWN_PRIORITY.get(preferred, _MARKDOWN_PRIORITY["fit"])
        for key in keys:
            value = markdown_value.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    return ""


def _extract_html(result_data: dict) -> str:
    # Preserve the full rendered DOM when available so downstream link discovery
    # can see links that may be dropped from cleaned/fit variants.
    for key in ("html", "cleaned_html", "fit_html"):
        value = result_data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_error(result_data: dict) -> str | None:
    for key in ("error_message", "error"):
        value = result_data.get(key)
        if isinstance(value, str) and value:
            return value
    return None


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
