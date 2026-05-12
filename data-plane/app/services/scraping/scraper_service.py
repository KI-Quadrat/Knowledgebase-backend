import threading
import time
from collections import deque
from collections.abc import Callable
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, Field

from app.config import ext
from app.services.audit import AuditLogger
from app.services.cache import ContentCache
from app.services.metrics import mark_cache_hit, mark_cache_miss, set_active_jobs
from app.services.rate_limiter import DomainRateLimiter
from app.services.scraping.crawl4ai_client import Crawl4AIClient
from app.services.scraping.document_discovery import DiscoveredDoc, discover_documents, split_documents_and_links
from app.utils.content import count_words, extract_links, extract_metadata
from app.utils.logger import get_logger

log = get_logger(__name__)


# CSS selectors for the most common cookie/CMP banners. Always merged into
# ``exclude_tags`` so the scraped markdown is not polluted by consent text.
# Conservative on purpose — only specific CMP hooks plus a small set of
# canonical class/id names. Broad attribute regex (``[class*="cookie"]``) is
# avoided since it would also strip legitimate footer cookie-policy blocks.
DEFAULT_COOKIE_BANNER_SELECTORS: tuple[str, ...] = (
    # OneTrust
    "#onetrust-banner-sdk",
    "#onetrust-consent-sdk",
    "#onetrust-pc-sdk",
    # Cookiebot
    "#CybotCookiebotDialog",
    "#CybotCookiebotDialogBodyUnderlay",
    "#CookieDeclaration",
    # Osano
    ".osano-cm-window",
    ".osano-cm-dialog",
    # Quantcast Choice
    "#qc-cmp2-container",
    "#qc-cmp2-ui",
    # TrustArc
    "#truste-consent-track",
    "#consent_blackbar",
    ".trustarc-banner-container",
    # Termly
    ".termly-styles-banner",
    "#termly-code-snippet-support",
    # Cookie Law Info (WP plugin)
    "#cookie-law-info-bar",
    ".cookie-law-info-bar",
    # Insites Cookie Consent
    ".cc-window",
    ".cc-banner",
    # Usercentrics
    "#usercentrics-root",
    "#uc-banner",
    # Klaro
    "#klaro",
    ".klaro",
    # Didomi
    "#didomi-host",
    ".didomi-popup-container",
    # Axeptio
    "#axeptio_overlay",
    "#axeptio_main_button",
    # Sourcepoint
    '[id^="sp_message_container_"]',
    # Generic, canonical names
    "#cookie-banner",
    ".cookie-banner",
    "#cookie-notice",
    ".cookie-notice",
    "#cookie-consent",
    ".cookie-consent",
    "#cookieConsent",
)


def _merge_cookie_selectors(exclude_tags: list[str] | None) -> list[str]:
    """Return ``exclude_tags`` extended with the default cookie selectors, deduped."""
    seen: set[str] = set()
    merged: list[str] = []
    for sel in (*(exclude_tags or ()), *DEFAULT_COOKIE_BANNER_SELECTORS):
        if sel and sel not in seen:
            seen.add(sel)
            merged.append(sel)
    return merged


# ── Internal models used by the scraper service ──────

class ScrapeOptions(BaseModel):
    js_render: bool = True
    wait_for: str | None = None
    extract_links: bool = True
    with_links_summary: bool = False
    inner_img: bool = False
    css_selector: str | None = None
    timeout: int = Field(30, ge=1, le=120)
    markdown_type: str = "fit"
    exclude_tags: list[str] | None = None
    scraper: str = "jina"


class PageMetadata(BaseModel):
    title: str | None = None
    description: str | None = None
    language: str | None = None
    word_count: int = 0


class DiscoveredDocument(BaseModel):
    url: str
    type: str
    link_text: str | None = None
    found_on: str | None = None


class ScrapeStatus:
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


class ScrapeResult(BaseModel):
    url: str
    status: str
    markdown: str | None = None
    html: str | None = None
    metadata: PageMetadata = Field(default_factory=PageMetadata)
    discovered_documents: list[DiscoveredDocument] = Field(default_factory=list)
    discovered_links: list[str] = Field(default_factory=list)
    # Image URLs reported by the backend (only populated when Jina is the
    # active scraper — Crawl4AI / httpx leave this empty because the rendered
    # HTML is available downstream and ``discover_images`` works there).
    discovered_images: list[str] = Field(default_factory=list)
    error: str | None = None
    duration_ms: int | None = None
    # Backend that actually produced the result: ``crawl4ai``, ``jina``, or
    # ``httpx`` (final fallback). ``None`` on failed scrapes.
    scraper_used: str | None = None


# ── Service ──────────────────────────────────────────

class ScraperService:
    """Main scraper service orchestration."""

    def __init__(self) -> None:
        self.cache = ContentCache()
        self.rate_limiter = DomainRateLimiter()
        self.crawl4ai = Crawl4AIClient()
        self.audit = AuditLogger()
        self._active_jobs = 0
        self._jobs_lock = threading.Lock()
        self._jina_domains: set[str] = {
            d.strip().lower().lstrip(".")
            for d in ext.jina_default_domains.split(",")
            if d.strip()
        }

    def _route_scraper(self, url: str, requested: str) -> str:
        """Pick the actual scraper backend for a URL.

        - Caller chose ``"jina"`` (now the default) → respected.
        - Caller explicitly opted into ``"crawl4ai"`` AND the URL's domain
          (or any parent domain) is in ``JINA_DEFAULT_DOMAINS`` → routed back
          to Jina (the override list flags domains where Crawl4AI is known
          to misbehave).
        - Otherwise → caller's value (``"crawl4ai"``).
        """
        if requested != "crawl4ai":
            return requested
        if not self._jina_domains:
            return requested
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        if not domain:
            return requested
        if domain in self._jina_domains:
            return "jina"
        if any(domain.endswith("." + d) for d in self._jina_domains):
            return "jina"
        return requested

    async def startup(self) -> None:
        log.info("scraper_starting")
        await self.crawl4ai.start()
        await self.cache.start()
        await self.rate_limiter.start()
        await self.audit.start()
        log.info("scraper_started")

    async def shutdown(self) -> None:
        await self.crawl4ai.close()
        await self.cache.close()
        await self.rate_limiter.close()
        await self.audit.close()
        log.info("scraper_shutdown")

    @property
    def is_ready(self) -> bool:
        return self.crawl4ai._client is not None

    @property
    def active_jobs(self) -> int:
        return self._active_jobs

    def _inc_jobs(self) -> None:
        with self._jobs_lock:
            self._active_jobs += 1
            set_active_jobs(self._active_jobs)

    def _dec_jobs(self) -> None:
        with self._jobs_lock:
            self._active_jobs = max(0, self._active_jobs - 1)
            set_active_jobs(self._active_jobs)

    async def scrape_url(
        self,
        url: str,
        options: ScrapeOptions,
        *,
        bypass_cache: bool = False,
        action_prefix: str = "scrape",
        request_id: str = "",
        api_key_hash: str = "",
    ) -> ScrapeResult:
        start = time.monotonic()
        self._inc_jobs()

        try:
            if not bypass_cache:
                cached = await self.cache.get(url)
                if cached:
                    mark_cache_hit()
                    log.info("scrape_cache_hit", url=url)
                    cached_markdown = cached.get("markdown") or ""
                    return ScrapeResult(
                        url=url,
                        status=ScrapeStatus.SUCCESS,
                        markdown=cached_markdown,
                        metadata=PageMetadata(
                            title=cached.get("title"),
                            word_count=count_words(cached_markdown),
                        ),
                        duration_ms=int((time.monotonic() - start) * 1000),
                        scraper_used=cached.get("scraper_used"),
                    )
                mark_cache_miss()

            await self.rate_limiter.acquire(url)

            scraper_choice = self._route_scraper(url, options.scraper)
            if scraper_choice != options.scraper:
                log.info(
                    "scraper_routed",
                    url=url,
                    requested=options.scraper,
                    routed_to=scraper_choice,
                )

            exclude_tags = _merge_cookie_selectors(options.exclude_tags)

            crawl_result = await self.crawl4ai.crawl(
                url,
                js_render=options.js_render,
                wait_for=options.wait_for,
                css_selector=options.css_selector,
                timeout=options.timeout,
                markdown_type=options.markdown_type,
                exclude_tags=exclude_tags,
                with_links_summary=options.with_links_summary,
                inner_img=options.inner_img,
                scraper=scraper_choice,
            )

            if not crawl_result.success:
                duration_ms = int((time.monotonic() - start) * 1000)
                await self.audit.log(
                    f"{action_prefix}.failed",
                    actor="system",
                    url=url,
                    status="failed",
                    request_id=request_id,
                    api_key_hash=api_key_hash,
                    error=crawl_result.error or "Unknown",
                    duration_ms=duration_ms,
                )
                return ScrapeResult(
                    url=url,
                    status=ScrapeStatus.FAILED,
                    error=crawl_result.error or "Crawl failed",
                    duration_ms=duration_ms,
                )

            markdown = crawl_result.markdown
            html = crawl_result.html

            meta = extract_metadata(html) if html else {}

            discovered_docs: list[DiscoveredDocument] = []
            if html:
                raw_docs = discover_documents(html, url)
                discovered_docs = [
                    DiscoveredDocument(
                        url=d.url,
                        type=d.type,
                        link_text=d.link_text,
                        found_on=d.found_on,
                    )
                    for d in raw_docs
                ]
            elif crawl_result.links:
                raw_docs, _ = split_documents_and_links(crawl_result.links, found_on=url)
                discovered_docs = [
                    DiscoveredDocument(
                        url=d.url,
                        type=d.type,
                        link_text=d.link_text,
                        found_on=d.found_on,
                    )
                    for d in raw_docs
                ]

            discovered_links: list[str] = []
            if html and options.extract_links:
                discovered_links = extract_links(html, url)
            elif options.extract_links and crawl_result.links:
                _, discovered_links = split_documents_and_links(crawl_result.links, found_on=url)

            # Backend-reported images (Jina path). Empty for crawl4ai/httpx —
            # the router runs ``discover_images`` against the rendered HTML
            # for those backends, so we don't need to populate it here.
            discovered_images: list[str] = list(crawl_result.images)

            metadata = PageMetadata(
                title=meta.get("title") or crawl_result.title,
                description=meta.get("description"),
                language=meta.get("language"),
                word_count=count_words(markdown),
            )

            if markdown is not None:
                await self.cache.set(
                    url,
                    markdown,
                    title=metadata.title,
                    scraper_used=crawl_result.scraper_used,
                )

            duration_ms = int((time.monotonic() - start) * 1000)

            await self.audit.log(
                f"{action_prefix}.completed",
                actor="system",
                url=url,
                status="success",
                request_id=request_id,
                api_key_hash=api_key_hash,
                documents_found=len(discovered_docs),
                word_count=metadata.word_count,
                duration_ms=duration_ms,
            )

            return ScrapeResult(
                url=url,
                status=ScrapeStatus.SUCCESS,
                markdown=markdown,
                html=html,
                metadata=metadata,
                discovered_documents=discovered_docs,
                discovered_links=discovered_links,
                discovered_images=discovered_images,
                duration_ms=duration_ms,
                scraper_used=crawl_result.scraper_used,
            )

        except TimeoutError:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.warning("scrape_timeout", url=url, timeout=options.timeout)
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.TIMEOUT,
                error=f"Request timed out after {options.timeout}s",
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            log.error("scrape_error", url=url, error=str(exc))
            return ScrapeResult(
                url=url,
                status=ScrapeStatus.FAILED,
                error=str(exc),
                duration_ms=duration_ms,
            )
        finally:
            self._dec_jobs()

    async def discover_urls(
        self,
        root_url: str,
        *,
        max_depth: int = 2,
        max_pages: int = 50,
        same_domain_only: bool = True,
        on_progress: Callable[[int, int, str], None] | None = None,
        scraper: str = "httpx",
    ) -> tuple[list[str], list[DiscoveredDocument], str | None]:
        """Breadth-first URL discovery.

        Backend behavior by ``scraper`` value:
        - ``crawl4ai`` — one server-side ``POST /crawl`` with
          ``BFSDeepCrawlStrategy``. The Crawl4AI server runs the BFS itself
          and returns the visited URL set + per-page link map in one round
          trip. Falls back to the Python BFS over httpx if the deep-crawl
          call fails (server unsupported, timeout, etc).
        - ``jina`` — Python BFS, each URL fetched via Jina's Chromium
          engine. Use only when the link graph is JS-injected.
        - ``httpx`` (**default**) — Python BFS with cheap raw HTTP fetches.
          Sufficient for sites with server-rendered nav.

        Returns ``(pages, documents, scraper_used)``. ``scraper_used``
        prefers the first non-None backend actually reported across the
        BFS, and falls back to the caller's requested ``scraper`` when
        every result was a legacy cache entry without a recorded backend.
        """
        if scraper == "crawl4ai":
            ok, pages, docs = await self._deep_crawl_via_crawl4ai(
                root_url,
                max_depth=max_depth,
                max_pages=max_pages,
                same_domain_only=same_domain_only,
                on_progress=on_progress,
            )
            if ok:
                return pages, docs, "crawl4ai"
            log.info("deep_crawl_falling_back_to_python_bfs", url=root_url)

        if scraper == "firecrawl":
            ok, pages, docs = await self._deep_crawl_via_firecrawl(
                root_url,
                max_pages=max_pages,
                same_domain_only=same_domain_only,
                on_progress=on_progress,
            )
            if ok:
                return pages, docs, "firecrawl"
            log.info("firecrawl_map_falling_back_to_python_bfs", url=root_url)

        return await self._discover_urls_python_bfs(
            root_url,
            max_depth=max_depth,
            max_pages=max_pages,
            same_domain_only=same_domain_only,
            on_progress=on_progress,
            scraper="httpx" if scraper in {"crawl4ai", "firecrawl"} else scraper,
        )

    async def _deep_crawl_via_crawl4ai(
        self,
        root_url: str,
        *,
        max_depth: int,
        max_pages: int,
        same_domain_only: bool,
        on_progress: Callable[[int, int, str], None] | None,
    ) -> tuple[bool, list[str], list[DiscoveredDocument]]:
        """Server-side BFS via ``Crawl4AIClient.deep_crawl``."""
        ok, visited, links, error = await self.crawl4ai.deep_crawl(
            root_url,
            max_depth=max_depth,
            max_pages=max_pages,
            same_domain_only=same_domain_only,
        )
        if not ok:
            log.warning("deep_crawl_failed", url=root_url, error=error)
            return False, [], []

        # Resolve relative hrefs against the root, then split into docs vs
        # pages. Client-side same-domain filter as a safety net — the server
        # honors ``include_external`` but builds vary; this guarantees the
        # caller's contract regardless.
        root_domain = urlparse(root_url).netloc.lower()
        absolute_links: list[str] = []
        for link in links:
            href = link.get("href")
            if not href:
                continue
            absolute = urljoin(root_url, href)
            if same_domain_only and urlparse(absolute).netloc.lower() != root_domain:
                continue
            absolute_links.append(absolute)
        raw_docs, _page_links = split_documents_and_links(absolute_links, found_on=root_url)
        discovered_docs = [
            DiscoveredDocument(
                url=d.url,
                type=d.type,
                link_text=d.link_text,
                found_on=d.found_on,
            )
            for d in raw_docs
        ]
        if on_progress:
            on_progress(len(visited), max_pages, root_url)
        return True, visited, discovered_docs

    async def _deep_crawl_via_firecrawl(
        self,
        root_url: str,
        *,
        max_pages: int,
        same_domain_only: bool,
        on_progress: Callable[[int, int, str], None] | None,
    ) -> tuple[bool, list[str], list[DiscoveredDocument]]:
        """URL discovery via ``Crawl4AIClient._map_with_firecrawl``.

        Firecrawl's ``/v2/map`` endpoint is single-shot — no depth control,
        so ``max_depth`` is ignored. The flat URL list is then partitioned
        into pages and documents via ``split_documents_and_links``, matching
        the contract of ``_deep_crawl_via_crawl4ai``.
        """
        ok, visited, links, error = await self.crawl4ai._map_with_firecrawl(
            root_url,
            max_pages=max_pages,
            same_domain_only=same_domain_only,
            timeout=120,
        )
        if not ok:
            log.warning("firecrawl_map_failed", url=root_url, error=error)
            return False, [], []

        root_domain = urlparse(root_url).netloc.lower()
        absolute_links: list[str] = []
        for link in links:
            href = link.get("href")
            if not href:
                continue
            absolute = urljoin(root_url, href)
            if same_domain_only and urlparse(absolute).netloc.lower() != root_domain:
                continue
            absolute_links.append(absolute)
        raw_docs, page_links = split_documents_and_links(absolute_links, found_on=root_url)
        discovered_docs = [
            DiscoveredDocument(
                url=d.url,
                type=d.type,
                link_text=d.link_text,
                found_on=d.found_on,
            )
            for d in raw_docs
        ]
        # ``visited`` from the map endpoint is the combined URL set; strip
        # the document URLs so the page list mirrors what crawl4ai's
        # deep-crawl path returns.
        doc_urls = {d.url for d in raw_docs}
        page_only = [u for u in visited if u not in doc_urls] or page_links
        if on_progress:
            on_progress(len(page_only), max_pages, root_url)
        return True, page_only, discovered_docs

    async def _discover_urls_python_bfs(
        self,
        root_url: str,
        *,
        max_depth: int,
        max_pages: int,
        same_domain_only: bool,
        on_progress: Callable[[int, int, str], None] | None,
        scraper: str,
    ) -> tuple[list[str], list[DiscoveredDocument], str | None]:
        """In-process BFS — fan out per URL through ``scrape_url``.

        ``js_render`` is set based on the scraper choice: ``httpx`` skips
        rendering entirely (cheap fetches), while ``jina`` enables it so
        the Chromium engine actually runs.
        """
        visited: set[str] = set()
        discovered_pages: list[str] = []
        doc_map: dict[str, DiscoveredDocument] = {}
        queue: deque[tuple[str, int]] = deque([(root_url, 0)])
        observed_scraper: str | None = None

        root_domain = urlparse(root_url).netloc.lower()
        discover_options = ScrapeOptions(
            js_render=(scraper != "httpx"),
            extract_links=True,
            timeout=15,
            scraper=scraper,
        )

        while queue and len(discovered_pages) < max_pages:
            current_url, depth = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)

            result = await self.scrape_url(current_url, discover_options)
            discovered_pages.append(current_url)
            if observed_scraper is None and result.scraper_used:
                observed_scraper = result.scraper_used
            if on_progress:
                on_progress(len(discovered_pages), max_pages, current_url)

            for doc in result.discovered_documents:
                if doc.url not in doc_map:
                    doc_map[doc.url] = doc

            if depth >= max_depth:
                continue

            for link in result.discovered_links:
                if link in visited:
                    continue
                if same_domain_only and urlparse(link).netloc.lower() != root_domain:
                    continue
                queue.append((link, depth + 1))

        scraper_used = observed_scraper or (scraper if discovered_pages else None)
        return discovered_pages, list(doc_map.values()), scraper_used
