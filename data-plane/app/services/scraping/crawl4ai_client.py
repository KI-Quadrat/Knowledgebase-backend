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
        title: str | None = None,
        links: list[str] | None = None,
        images: list[str] | None = None,
        success: bool = True,
        error: str | None = None,
        duration_ms: int = 0,
        scraper_used: str | None = None,
    ):
        self.markdown = markdown
        self.html = html
        # Title reported directly by the backend (Jina ``data.title``). The
        # Crawl4AI ``/md`` endpoint does not return a title — the router falls
        # back to whatever ``extract_metadata`` can pull from cached HTML or
        # leaves it None. The httpx fallback returns rendered HTML, so the
        # router extracts the title from there.
        self.title = title
        self.links = links or []
        # Image URLs harvested by the backend (Jina X-With-Images-Summary).
        # Crawl4AI ``/md`` and httpx leave this empty — the router runs
        # ``discover_images`` against rendered HTML for the httpx backend, and
        # the Crawl4AI branch produces no HTML, so per-page image discovery is
        # only available via the Jina/httpx paths.
        self.images = images or []
        self.success = success
        self.error = error
        self.duration_ms = duration_ms
        # Which backend actually produced the markdown — set per-branch in
        # ``crawl()`` after a backend succeeds. ``None`` on failed results.
        self.scraper_used = scraper_used


class Crawl4AIClient:
    """HTTP client wrapper for external Crawl4AI with Jina Reader and httpx fallback."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url = ext.crawl4ai_url.rstrip("/")
        self._api_token = ext.crawl4ai_api_token
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
        inner_img: bool = False,
        scraper: str = "crawl4ai",
    ) -> CrawlResult:
        if not self._client:
            raise RuntimeError("Client not started — call start() first")

        req_timeout = timeout or settings.default_timeout
        start = time.monotonic()

        # Build prioritized backend list based on client preference. The
        # non-preferred backend (and raw httpx) remain as automatic fallbacks
        # so requests stay best-effort even when the chosen backend fails.
        # ``scraper="httpx"`` short-circuits both browser backends and goes
        # straight to the raw httpx fallback below — used by /crawl BFS where
        # Chromium-rendered fetches are wasted on per-URL link harvesting.
        if scraper == "httpx":
            backend_order: tuple[str, ...] = ()
        elif scraper == "jina":
            backend_order = ("jina", "crawl4ai")
        elif scraper == "firecrawl":
            backend_order = ("firecrawl", "jina", "crawl4ai")
        else:
            backend_order = ("crawl4ai", "jina")

        for backend in backend_order:
            # ``js_render=False`` opts out of every browser-rendered backend.
            # Both Crawl4AI's ``/md`` and Jina's Chromium engine fetch via a
            # headless browser, so the flag must skip both — otherwise Jina
            # silently re-introduces the Chromium cost we asked to avoid.
            if not js_render:
                continue
            if backend == "crawl4ai":
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
                        result.scraper_used = "crawl4ai"
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
                        inner_img=inner_img,
                    )
                    result.duration_ms = int((time.monotonic() - start) * 1000)
                    if result.success:
                        result.scraper_used = "jina"
                        log.info("jina_scrape_success", url=url, primary=(scraper == "jina"))
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
        # Crawl4AI's /md endpoint takes a tiny payload — just url + filter.
        # The legacy /crawl endpoint with browser_config / crawler_config /
        # markdown_generator / magic was tripping the new server with
        # "Invalid expression" on some pages, so we no longer construct that
        # config envelope. Caller-supplied wait_for / css_selector /
        # exclude_tags can't be forwarded here (the server applies its own
        # defaults); they're still honored on the Jina branch where they map
        # to request headers.
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        result = await self._fetch_md(
            url,
            markdown_type=markdown_type,
            timeout=timeout,
            headers=headers,
        )

        # Fit → raw fallback. Crawl4AI's PruningContentFilter (f=fit)
        # occasionally strips pages whose payload is mostly tabular /
        # label-value content (Austrian gov portals are the canonical case),
        # leaving fit-mode markdown nearly empty. The /md branch has no HTML
        # to compare against — only the markdown — so we use a word-count
        # floor. Retry once with f=raw and keep that result if it's a real
        # improvement (more words than the fit result); otherwise fall
        # through to the original so the outer Jina/httpx chain can engage
        # via the normal success=False path. Skipped for raw/citations —
        # both already map to f=raw, so retrying would change nothing.
        if (
            markdown_type == "fit"
            and result.success
            and len(result.markdown.split()) < _FIT_FALLBACK_WORD_THRESHOLD
        ):
            log.info(
                "crawl4ai_fit_thin_retry_raw",
                url=url,
                word_count=len(result.markdown.split()),
            )
            raw_result = await self._fetch_md(
                url,
                markdown_type="raw",
                timeout=timeout,
                headers=headers,
            )
            if (
                raw_result.success
                and raw_result.markdown.strip()
                and len(raw_result.markdown.split()) > len(result.markdown.split())
            ):
                return raw_result

        return result

    async def _fetch_md(
        self,
        url: str,
        *,
        markdown_type: str,
        timeout: int,
        headers: dict[str, str],
    ) -> CrawlResult:
        """POST one /md request and parse the response into a CrawlResult.

        Maps the public ``markdown_type`` (fit / raw / citations) to /md's
        ``f`` filter via ``_FILTER_BY_MARKDOWN_TYPE``. The /md endpoint only
        supports raw | fit | bm25 | llm — there is no citations filter, so
        citations degrades to f=raw (same best-effort the Jina and httpx
        branches already provide). Empty markdown and the literal
        ``Crawl4AI Error`` preamble are surfaced as failures so the upstream
        Jina/httpx fallback chain engages.
        """
        filter_value = _FILTER_BY_MARKDOWN_TYPE.get(markdown_type, "raw")
        payload: dict = {"url": url, "f": filter_value}

        resp = await self._client.post(  # type: ignore[union-attr]
            f"{self._base_url}/md",
            json=payload,
            headers=headers,
            timeout=timeout + 10,
        )
        resp.raise_for_status()
        data = resp.json()

        markdown_value = data.get("markdown")
        if isinstance(markdown_value, dict):
            # /md returns a string today, but the older /crawl envelope used
            # a dict keyed by fit_markdown / markdown_with_citations /
            # raw_markdown. Defend against either shape: prefer the variant
            # that matches the caller's markdown_type, fall through the rest.
            markdown = _extract_markdown(markdown_value, preferred=markdown_type)
        elif isinstance(markdown_value, str):
            markdown = markdown_value
        else:
            markdown = ""

        success = bool(data.get("success", bool(markdown.strip())))
        error = _extract_error(data)

        # New server still occasionally returns success=true with the literal
        # "Crawl4AI Error" preamble baked into the markdown body. Treat that
        # as a failure so the Jina/httpx fallback chain engages.
        if success and markdown.lstrip().startswith("Crawl4AI Error"):
            success = False
            error = markdown.split("\n", 1)[0].strip()
            markdown = ""

        return CrawlResult(markdown=clean_markdown(markdown), html="", success=success, error=error)

    async def deep_crawl(
        self,
        root_url: str,
        *,
        max_depth: int = 3,
        max_pages: int = 50,
        same_domain_only: bool = True,
        timeout: int = 120,
    ) -> tuple[bool, list[str], list[dict], str | None]:
        """Server-side BFS via Crawl4AI ``POST /crawl`` + ``BFSDeepCrawlStrategy``.

        Returns ``(success, visited_urls, discovered_links, error)``.

        - ``visited_urls`` — pages the server actually crawled (one per result).
        - ``discovered_links`` — the union of internal ``links`` reported across
          all crawled pages, as ``[{"href": ..., "text": ...}, ...]``. Used by
          the caller to classify pages vs documents.
        - On failure ``success=False`` with ``error`` set; the caller is
          expected to fall back to its own BFS.

        The endpoint runs synchronously when ``stream=false`` (default) and
        returns ``{"results": [...]}``. Older builds returned ``{"task_id": ...}``
        for async polling — we treat that as unsupported here and surface a
        failure so the caller falls back rather than hanging on a poll loop.
        """
        if not self._client:
            raise RuntimeError("Client not started — call start() first")

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        payload: dict = {
            "urls": [root_url],
            "crawler_config": {
                "deep_crawl_strategy": {
                    "type": "BFSDeepCrawlStrategy",
                    "max_depth": max_depth,
                    "max_pages": max_pages,
                    "include_external": not same_domain_only,
                },
                "stream": False,
            },
        }

        try:
            resp = await self._client.post(
                f"{self._base_url}/crawl",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            return False, [], [], f"Crawl4AI deep_crawl timeout after {timeout}s"
        except httpx.HTTPStatusError as exc:
            return False, [], [], f"Crawl4AI deep_crawl HTTP {exc.response.status_code}"
        except Exception as exc:
            return False, [], [], f"Crawl4AI deep_crawl error: {exc}"

        if "results" not in data:
            # Async-task variant — poll-based. Treat as unsupported.
            return False, [], [], "Crawl4AI returned task_id (async mode unsupported)"

        results = data.get("results") or []
        visited: list[str] = []
        seen_visited: set[str] = set()
        discovered: list[dict] = []
        seen_links: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            url_value = item.get("url")
            if isinstance(url_value, str) and url_value not in seen_visited:
                visited.append(url_value)
                seen_visited.add(url_value)
            links_obj = item.get("links") or {}
            internal = links_obj.get("internal") or [] if isinstance(links_obj, dict) else []
            for link in internal:
                if not isinstance(link, dict):
                    continue
                href = link.get("href")
                if not isinstance(href, str) or href in seen_links:
                    continue
                seen_links.add(href)
                discovered.append({"href": href, "text": link.get("text") or ""})

        return True, visited, discovered, None

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
            return CrawlResult(success=False, error=f"Jina HTTP {exc.response.status_code}")
        except Exception as exc:
            return CrawlResult(success=False, error=f"Jina error: {exc}")

        data = resp.json()
        data_section = data.get("data", {}) if isinstance(data.get("data"), dict) else {}
        content = data_section.get("content", "")
        title_value = data_section.get("title")
        title = title_value.strip() if isinstance(title_value, str) and title_value.strip() else None
        links = _extract_jina_links(data, url)
        images = _extract_jina_images(data, url)

        if not content.strip():
            return CrawlResult(success=False, error="Jina returned empty content")

        markdown = clean_markdown(content)
        return CrawlResult(
            markdown=markdown,
            html="",
            title=title,
            links=links,
            images=images,
            success=True,
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
            return CrawlResult(success=False, error=f"Firecrawl HTTP {exc.response.status_code}")
        except Exception as exc:
            return CrawlResult(success=False, error=f"Firecrawl error: {exc}")

        data = resp.json()
        if not data.get("success", True):
            return CrawlResult(success=False, error=str(data.get("error") or "Firecrawl returned success=false"))

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

        return CrawlResult(
            markdown=clean_markdown(markdown),
            html=html_value,
            title=title,
            links=links,
            success=True,
        )

    async def _map_with_firecrawl(
        self,
        url: str,
        *,
        max_pages: int,
        same_domain_only: bool,
        timeout: int,
    ) -> tuple[bool, list[str], list[dict], str | None]:
        """Discover URLs via Firecrawl's ``POST /v2/map``.

        Returns the same ``(success, visited, links, error)`` tuple shape as
        ``deep_crawl``. Firecrawl's map endpoint returns a flat URL list in a
        single round-trip — we treat that list as both the visited set and the
        source of ``discovered_links``, so the caller's downstream
        ``split_documents_and_links`` still partitions pages from documents.
        """
        if not self._firecrawl_key:
            return False, [], [], "Firecrawl API key not configured"

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
            return False, [], [], f"Firecrawl map timeout after {timeout}s"
        except httpx.HTTPStatusError as exc:
            return False, [], [], f"Firecrawl map HTTP {exc.response.status_code}"
        except Exception as exc:
            return False, [], [], f"Firecrawl map error: {exc}"

        if not data.get("success", True):
            return False, [], [], str(data.get("error") or "Firecrawl map success=false")

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

        return True, visited, discovered, None

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


# Map the public markdown_type (fit / raw / citations) to Crawl4AI's /md
# ``f`` filter. /md supports raw | fit | bm25 | llm only — citations is a
# /crawl-only feature, so it degrades to f=raw here (same best-effort the
# Jina and httpx branches already provide).
_FILTER_BY_MARKDOWN_TYPE: dict[str, str] = {
    "fit": "fit",
    "raw": "raw",
    "citations": "raw",
}

# Word-count floor for the Crawl4AI /md fit→raw fallback. The /md endpoint
# returns markdown only (no HTML), so the router's ratio-based thin check
# (which needs both) is bypassed; this is the in-client backstop. Tuned to
# match the router's _THIN_WORD_THRESHOLD so both heuristics agree on what
# counts as "suspiciously short."
_FIT_FALLBACK_WORD_THRESHOLD = 20


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
