from typing import Literal

from pydantic import BaseModel, Field

from app.config import ext
from app.models.classify import ExtractedEntities
from app.models.common import UsageSummary


_VALID_SCRAPERS: set[str] = {"crawl4ai", "jina", "firecrawl"}
_VALID_CRAWLERS: set[str] = {"httpx", "crawl4ai", "jina", "firecrawl"}


def _default_scraper() -> Literal["crawl4ai", "jina", "firecrawl"]:
    value = ext.default_scraper
    return value if value in _VALID_SCRAPERS else "jina"  # type: ignore[return-value]


def _default_crawler() -> Literal["httpx", "crawl4ai", "jina", "firecrawl"]:
    value = ext.default_crawler
    return value if value in _VALID_CRAWLERS else "httpx"  # type: ignore[return-value]


class ScrapeRequest(BaseModel):
    """Request to scrape a single webpage and extract its content as Markdown."""

    url: str = Field(..., description="Full URL of the webpage to scrape (must start with http:// or https://)")
    inner_img: bool = Field(False, description="If true, extract and parse images found on the page (returns alt text, URL, and OCR content if available)")
    inner_docs: bool = Field(False, description="If true, extract and parse documents (PDF, DOCX, etc.) linked on the page using the document parsing backend")
    markdown_type: Literal["fit", "raw", "citations"] = Field(
        "fit",
        description=(
            "Which Markdown variant to return. "
            "`fit` = main content only (Jina Chromium browser-engine markdown by default; "
            "Firecrawl `onlyMainContent` on the fallback path). "
            "`raw` = full page including headers/nav/footer. "
            "`citations` = full content with citation links preserved (best-effort; "
            "no backend exposes a dedicated citations filter, so this currently "
            "degrades to `raw`)."
        ),
    )
    exclude_tags: list[str] | None = Field(
        None,
        description=(
            "CSS selectors or tag names to remove before extraction "
            "(e.g. `['nav', 'footer', '.sidebar']`). Applied on all backends."
        ),
    )
    css_selector: str | None = Field(
        None,
        description=(
            "CSS selector to scope extraction to a specific element "
            "(e.g. `'main'` or `'article.content'`). Applied on all backends."
        ),
    )
    scraper: Literal["crawl4ai", "jina", "firecrawl"] = Field(
        default_factory=_default_scraper,
        description=(
            "Preferred scraping backend (default `jina`, configurable server-side). "
            "`jina` uses the Jina Reader API (Chromium engine). `firecrawl` uses "
            "Firecrawl's `POST /v2/scrape` (when configured). The other backend "
            "(and raw httpx) remain as automatic fallbacks if the selected one fails. "
            "`crawl4ai` is a deprecated alias retained for backward compatibility — "
            "it now behaves exactly like `jina`."
        ),
    )
    links_summary: bool = Field(
        False,
        description=(
            "If true, include a `links_summary.urls` list in the response — deduped "
            "http/https page links extracted from the raw page HTML (not the fit-filtered "
            "HTML), so no boilerplate/footer links are missed. "
            "`documents` and `images` sub-lists are populated only when `inner_docs` / "
            "`inner_img` are also true respectively, so the client opts in per kind. "
            "Triggers one extra raw-HTML fetch (~small)."
        ),
    )
    classify: bool = Field(
        True,
        description=(
            "If true, run content classification and entity extraction after scraping. "
            "Set false to return only scraped content and skip the LLM classifier."
        ),
    )
    bypass_cache: bool = Field(
        False,
        description=(
            "If true, skip the Redis cache and force a fresh fetch from the origin "
            "(then update the cache with the new content). Use when stale cached "
            "content must be refreshed — e.g. policy/funding pages that change "
            "between scheduled re-ingests. `links_summary`, `inner_img`, and "
            "`inner_docs` already imply a fresh fetch."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"url": "https://www.example.gv.at/foerderungen"},
                {
                    "url": "https://www.example.gv.at/article",
                    "markdown_type": "fit",
                    "exclude_tags": ["nav", "footer", "aside"],
                    "css_selector": "main",
                },
                {"url": "https://www.example.gv.at/foerderungen", "inner_img": True, "inner_docs": True},
                {"url": "https://www.example.gv.at/foerderungen", "scraper": "firecrawl"},
                {"url": "https://www.example.gv.at/foerderungen", "links_summary": True},
                {"url": "https://www.example.gv.at/foerderungen", "classify": False},
            ]
        }
    }


class InnerImageData(BaseModel):
    """Parsed image found on the scraped page."""

    url: str = Field(..., description="Absolute URL of the image")
    alt: str | None = Field(None, description="Alt text of the image")
    title: str | None = Field(None, description="Title attribute of the image")
    content: str | None = Field(None, description="Extracted text content from the image (OCR via LlamaParse)")
    content_length: int = Field(0, description="Length of extracted content in characters")
    error: str | None = Field(None, description="Error message if image parsing failed")


class InnerDocData(BaseModel):
    """Parsed document found on the scraped page."""

    url: str = Field(..., description="Absolute URL of the document")
    title: str | None = Field(None, description="Link text or document title")
    doc_type: str = Field(..., description="Document type (pdf, docx, xlsx, etc.)")
    content: str | None = Field(None, description="Extracted text content from the document")
    pages: int | None = Field(None, description="Number of pages parsed")
    content_length: int = Field(0, description="Length of extracted content in characters")
    language: str | None = Field(None, description="Detected document language (ISO 639-1)")
    error: str | None = Field(None, description="Error message if parsing failed")


class LinksSummary(BaseModel):
    """Deduped URL lists extracted from the raw page HTML, grouped by kind.

    `urls` is always populated when `links_summary=true`. `documents` and `images`
    are populated only when the client also sets `inner_docs` / `inner_img`.
    """

    urls: list[str] = Field(default_factory=list, description="Unique page links (non-document http/https URLs)")
    documents: list[str] = Field(default_factory=list, description="Unique document URLs (populated only when inner_docs=true)")
    images: list[str] = Field(default_factory=list, description="Unique image URLs (populated only when inner_img=true)")


class ScrapeData(BaseModel):
    """Scraped webpage content and metadata."""

    url: str = Field(..., description="The URL that was scraped")
    title: str | None = Field(None, description="Page title from <title> tag")
    content: str = Field(..., description="Extracted page content as Markdown")
    content_length: int = Field(..., description="Length of the content string in characters")
    language: str | None = Field(None, description="Detected language (ISO 639-1 code, e.g. 'de')")
    links_found: int = Field(0, description="Number of links discovered on the page")
    last_modified: str | None = Field(None, description="Last-Modified header value if present")
    content_type: list[str] = Field(default_factory=list, description="Classifier-derived content categories for the page (e.g. ['funding', 'renewable_energy']). Empty when the request uses classify=false. Pass this verbatim to /online/ingest only when classification was enabled.")
    entities: ExtractedEntities | None = Field(None, description="Structured entities extracted by the classifier (dates, deadlines, amounts, contacts, departments). Null when classification failed or the request uses classify=false.")
    inner_images: list[InnerImageData] | None = Field(None, description="Parsed images found on the page (only when inner_img=true)")
    inner_documents: list[InnerDocData] | None = Field(None, description="Parsed documents linked on the page (only when inner_docs=true)")
    links_summary: LinksSummary | None = Field(None, description="Deduped URL lists grouped by kind (only when links_summary=true)")
    scraper_used: str | None = Field(None, description="Backend that produced the content: 'jina', 'firecrawl', or 'httpx' (final fallback). Null on failure or cache hits with no recorded backend.")
    usage: UsageSummary | None = Field(
        None,
        description=(
            "Per-stage billing for this scrape. ``by_stage`` carries one "
            "entry per external call (`scraper`, `classifier`, optional "
            "`inner_img` / `inner_docs`). For Jina the scraper entry "
            "reports ``scrape_tokens`` from ``meta.usage.tokens``; for "
            "Firecrawl it reports ``credits``; raw httpx is "
            "self-hosted so ``cost_usd`` is 0. Cache hits report a "
            "``cache`` provider with 0 cost."
        ),
    )


class CrawlRequest(BaseModel):
    """Request to discover URLs from a website via sitemap parsing or BFS crawling."""

    url: str = Field(..., description="Base URL or sitemap URL to crawl")
    method: str = Field(..., description="Discovery method: 'sitemap' (parse XML sitemap) or 'crawl' (BFS link following)", pattern=r"^(sitemap|crawl)$")
    max_depth: int = Field(3, ge=1, le=5, description="Maximum link-following depth for crawl method")
    max_urls: int = Field(500, ge=1, le=5000, description="Maximum number of URLs to return")
    scraper: Literal["httpx", "crawl4ai", "jina", "firecrawl"] = Field(
        default_factory=_default_crawler,
        description=(
            "Backend used during BFS `crawl` discovery (ignored when `method='sitemap'`). "
            "Default `httpx`, configurable server-side.\n"
            "- `httpx` — raw HTTP fetches, no JS rendering. Fast and free; "
            "sufficient for sites with server-rendered nav menus (most muni/gov portals).\n"
            "- `jina` — per-URL Python BFS through Jina's hosted Chromium engine. "
            "Expensive — only pick this when the link graph is genuinely JS-injected and "
            "httpx misses links.\n"
            "- `firecrawl` — single-shot URL map via Firecrawl (when configured). Falls "
            "back to the Python BFS over httpx if the map call fails.\n"
            "- `crawl4ai` — deprecated alias retained for backward compatibility; "
            "behaves like `httpx`."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"url": "https://www.example.gv.at/sitemap.xml", "method": "sitemap", "max_urls": 500},
                {"url": "https://www.example.gv.at", "method": "crawl", "max_depth": 3, "max_urls": 100},
                {"url": "https://www.example.gv.at", "method": "crawl", "scraper": "jina"},
                {"url": "https://www.example.gv.at", "method": "crawl", "scraper": "firecrawl"},
            ]
        }
    }


class CrawlUrl(BaseModel):
    """A discovered URL with its type classification."""

    url: str = Field(..., description="Discovered URL")
    type: str = Field(..., description="URL type: 'page' (HTML) or 'document' (PDF, DOCX, etc.)")
    last_modified: str | None = Field(None, description="Last modified date from sitemap")


class CrawlData(BaseModel):
    """Result of URL discovery via sitemap or crawl."""

    base_url: str = Field(..., description="The URL that was crawled")
    method_used: str = Field(..., description="Method that was used: sitemap or crawl")
    urls: list[CrawlUrl] = Field(..., description="List of discovered URLs")
    total_urls: int = Field(..., description="Total number of URLs discovered")
    scraper_used: str | None = Field(
        None,
        description=(
            "Backend that produced the BFS discovery: 'jina' (per-URL Chromium), "
            "'firecrawl' (single-shot `/v2/map`), or 'httpx' (raw HTTP — the "
            "default for /crawl). When the requested 'firecrawl' call fails, the "
            "service falls back to a Python BFS over httpx and reports 'httpx' "
            "here. Falls back to the requested `scraper` value when every BFS hit "
            "was a legacy cache entry without a recorded backend. Null for "
            "`method='sitemap'` (no scraping involved) or when no pages were "
            "discovered at all."
        ),
    )
    usage: UsageSummary | None = Field(
        None,
        description=(
            "Aggregated scrape usage across the BFS. The Python BFS path "
            "sums per-page Jina tokens / Firecrawl credits / $0 entries "
            "into a single ``scraper`` row; the Firecrawl ``/v2/map`` path "
            "reports one row for the single API call. Null for "
            "`method='sitemap'` (no scraping involved)."
        ),
    )
