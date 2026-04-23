import threading
import time
from collections import deque
from collections.abc import Callable
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from app.services.audit import AuditLogger
from app.services.cache import ContentCache
from app.services.metrics import mark_cache_hit, mark_cache_miss, set_active_jobs
from app.services.rate_limiter import DomainRateLimiter
from app.services.scraping.crawl4ai_client import Crawl4AIClient
from app.services.scraping.document_discovery import DiscoveredDoc, discover_documents, split_documents_and_links
from app.utils.content import count_words, extract_links, extract_metadata
from app.utils.logger import get_logger

log = get_logger(__name__)


# ── Internal models used by the scraper service ──────

class ScrapeOptions(BaseModel):
    js_render: bool = True
    wait_for: str | None = None
    extract_links: bool = True
    with_links_summary: bool = False
    css_selector: str | None = None
    timeout: int = Field(30, ge=1, le=120)
    markdown_type: str = "fit"
    exclude_tags: list[str] | None = None
    scraper: str = "crawl4ai"


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
    error: str | None = None
    duration_ms: int | None = None


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
                    return ScrapeResult(
                        url=url,
                        status=ScrapeStatus.SUCCESS,
                        markdown=cached,
                        metadata=PageMetadata(word_count=count_words(cached)),
                        duration_ms=int((time.monotonic() - start) * 1000),
                    )
                mark_cache_miss()

            await self.rate_limiter.acquire(url)

            crawl_result = await self.crawl4ai.crawl(
                url,
                js_render=options.js_render,
                wait_for=options.wait_for,
                css_selector=options.css_selector,
                timeout=options.timeout,
                markdown_type=options.markdown_type,
                exclude_tags=options.exclude_tags,
                with_links_summary=options.with_links_summary,
                scraper=options.scraper,
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

            metadata = PageMetadata(
                title=meta.get("title"),
                description=meta.get("description"),
                language=meta.get("language"),
                word_count=count_words(markdown),
            )

            if markdown is not None:
                await self.cache.set(url, markdown)

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
                duration_ms=duration_ms,
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
        scraper: str = "crawl4ai",
    ) -> tuple[list[str], list[DiscoveredDocument]]:
        """Breadth-first URL discovery using lightweight scraping."""
        visited: set[str] = set()
        discovered_pages: list[str] = []
        doc_map: dict[str, DiscoveredDocument] = {}
        queue: deque[tuple[str, int]] = deque([(root_url, 0)])

        root_domain = urlparse(root_url).netloc.lower()
        discover_options = ScrapeOptions(
            js_render=False, extract_links=True, timeout=15, scraper=scraper
        )

        while queue and len(discovered_pages) < max_pages:
            current_url, depth = queue.popleft()
            if current_url in visited:
                continue
            visited.add(current_url)

            result = await self.scrape_url(current_url, discover_options)
            discovered_pages.append(current_url)
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

        return discovered_pages, list(doc_map.values())
