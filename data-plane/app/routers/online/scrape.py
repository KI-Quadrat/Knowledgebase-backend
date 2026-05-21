"""
POST /api/v1/online/scrape — Scrape a single webpage (Crawl4AI or Jina Reader)
POST /api/v1/online/crawl  — Discover URLs from site/sitemap
"""

import asyncio

import httpx
from fastapi import APIRouter, Request

from app.models.classify import ExtractedEntities as ClassifyEntities
from app.models.common import ErrorCode, ResponseEnvelope
from app.models.online.scrape import (
    CrawlData,
    CrawlRequest,
    CrawlUrl,
    InnerDocData,
    InnerImageData,
    LinksSummary,
    ScrapeData,
    ScrapeRequest,
)
from app.services.parsing.models import ParseStatus
from app.services.scraping.document_discovery import (
    discover_images,
    document_type,
    extract_documents_and_links,
)
from app.services.scraping.scraper_service import ScrapeOptions, ScrapeStatus
from app.services.scraping.transparenzportal import enrich_if_applicable
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/online", tags=["Online - Web Scraping"])


def _validate_url(url: str) -> str | None:
    """Return error message if invalid, None if valid."""
    url = url.strip()
    if not url:
        return "URL is required"
    if not url.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    return None


# Thin-output detection: tuned for the pattern where Crawl4AI's
# PruningContentFilter aggressively removes tabular/label-value content (e.g.
# Austrian government portals). Below either threshold we re-scrape in `raw`
# mode to recover the payload.
_THIN_WORD_THRESHOLD = 20
_THIN_RATIO_THRESHOLD = 0.005  # markdown_len / html_len
_THIN_MIN_HTML_LEN = 1000


def _is_thin_output(markdown: str | None, html: str | None) -> bool:
    """True when fit-mode markdown looks too sparse relative to the raw HTML.

    Requires the raw HTML to compare against — Jina fallback returns no HTML,
    so thin-detection is skipped there.

    Both signals must trip to flag thinness (tightened from OR to AND after
    the original heuristic fired too aggressively on normal short pages and
    doubled scrape time):

    - ``word_count < 20`` on a page with ``len(html) > 1000``
    - ``len(markdown) / len(html) < 0.005`` (markdown is a sliver of the DOM)
    """
    if not html:
        return False
    text = (markdown or "").strip()
    if not text:
        return True
    html_len = len(html)
    if html_len <= _THIN_MIN_HTML_LEN:
        return False
    word_thin = len(text.split()) < _THIN_WORD_THRESHOLD
    ratio_thin = (len(text) / html_len) < _THIN_RATIO_THRESHOLD
    return word_thin and ratio_thin


@router.post(
    "/scrape",
    summary="Scrape a single webpage",
    description=(
        "Scrape a webpage using **Crawl4AI** (with JavaScript rendering) or the **Jina Reader API** "
        "and return the extracted content as clean Markdown. Backend is selectable per request via "
        "the `scraper` field (default `crawl4ai`). Includes title, language detection, and link discovery. "
        "Results are cached in Redis.\n\n"
        "---\n\n"
        "## How content extraction works\n\n"
        "The scraper processes content in multiple stages:\n\n"
        "1. **Crawl4AI fetches the page** — full JavaScript rendering, waits for `networkidle`, "
        "auto-removes cookie banners and overlay popups\n"
        "2. **Markdown extraction** — controlled by `markdown_type`:\n"
        "   - `fit` (default) — Crawl4AI runs `PruningContentFilter` (text/link-density heuristic) "
        "on top of `DefaultMarkdownGenerator` to return main content only. "
        "Jina fallback uses the `readerlm-v2` engine for equivalent LLM-based cleanup.\n"
        "   - `raw` — full page Markdown including headers/nav/footer.\n"
        "   - `citations` — full content with citation links preserved (Crawl4AI only; "
        "Jina/httpx fallbacks return `raw` equivalent).\n"
        "3. **Tag exclusion** — if `exclude_tags` is set, those selectors are removed before extraction "
        "on every backend (Crawl4AI `excluded_tags`, Jina `X-Remove-Selector`, httpx BeautifulSoup decompose).\n"
        "4. **Scoping** — if `css_selector` is set, extraction is scoped to that element "
        "(Crawl4AI `css_selector`, Jina `X-Target-Selector`, httpx pre-filter).\n"
        "5. **HTML noise removal (httpx fallback only)** — additional strip list: "
        "`nav`, `header`, `footer`, `.navbar`, `.sidebar`, `.cookie-banner`, `.ad`, `script`, `style`, "
        "`[role=banner]`, `[role=navigation]`, `[role=contentinfo]`, and more.\n"
        "6. **Markdown cleanup** — collapses excessive newlines, strips JavaScript URLs, "
        "removes empty links, data URIs, zero-width characters, normalizes Unicode spaces.\n\n"
        "---\n\n"
        "## Request fields\n\n"
        "| Field | Type | Required | Default | Description |\n"
        "|-------|------|----------|---------|-------------|\n"
        "| `url` | string | Required | — | Full URL to scrape (must start with `http://` or `https://`) |\n"
        "| `markdown_type` | string | Optional | `fit` | `fit` = main content only (PruningContentFilter on Crawl4AI, "
        "readerlm-v2 engine on Jina fallback). `raw` = full page. `citations` = full content with citation links "
        "(Crawl4AI only; falls back to `raw` on Jina/httpx). |\n"
        "| `exclude_tags` | string[] | Optional | `null` | CSS selectors / tag names to drop before extraction "
        "(e.g. `['nav','footer','.sidebar']`). Applied on all three backends. |\n"
        "| `css_selector` | string | Optional | `null` | CSS selector to scope extraction to a specific element "
        "(e.g. `'main'` or `'article.content'`). Applied on all three backends. |\n"
        "| `inner_img` | boolean | Optional | `false` | Extract and OCR-parse images found on the page "
        "(returns alt text, URL, and extracted text content via LlamaParse) |\n"
        "| `inner_docs` | boolean | Optional | `false` | Extract and parse documents (PDF, DOCX, XLSX, PPTX, etc.) "
        "linked on the page using the document parsing backend |\n"
        "| `scraper` | string | Optional | `crawl4ai` | Preferred scraping backend: `crawl4ai` "
        "(JS rendering, default) or `jina` (Jina Reader API). The non-selected backend and raw httpx "
        "remain as automatic fallbacks if the primary fails. |\n"
        "| `links_summary` | boolean | Optional | `false` | If true, adds a `links_summary.urls` list "
        "to the response — deduped http/https page links extracted from the **raw** page HTML "
        "(so nav/footer links filtered by `markdown_type='fit'` aren't missed). "
        "`links_summary.documents` is populated only when `inner_docs=true`; "
        "`links_summary.images` is populated only when `inner_img=true`. "
        "Triggers one extra lightweight raw-HTML fetch. |\n\n"
        "---\n\n"
        "## Examples\n\n"
        "**Default — clean main content only:**\n"
        "```json\n"
        "{ \"url\": \"https://transparenzportal.gv.at/tdb/tp/leistung/1051580.html\" }\n"
        "```\n\n"
        "**Scope to `<main>` and drop nav/footer/sidebar:**\n"
        "```json\n"
        "{\n"
        "  \"url\": \"https://example.com/article\",\n"
        "  \"markdown_type\": \"fit\",\n"
        "  \"exclude_tags\": [\"nav\", \"footer\", \"aside\", \".sidebar\"],\n"
        "  \"css_selector\": \"main\"\n"
        "}\n"
        "```\n\n"
        "**Full page including all boilerplate:**\n"
        "```json\n"
        "{ \"url\": \"https://example.com\", \"markdown_type\": \"raw\" }\n"
        "```\n\n"
        "**Use Jina Reader as the primary scraper (no JS rendering):**\n"
        "```json\n"
        "{ \"url\": \"https://example.com/article\", \"scraper\": \"jina\" }\n"
        "```\n\n"
        "---\n\n"
        "## Content filtering tips\n\n"
        "- `markdown_type: \"fit\"` (default) usually produces the cleanest content. For pages with good "
        "semantic HTML (`<main>`, `<article>`), this is all you need.\n"
        "- For sites with site-specific noise blocks, add them to `exclude_tags` "
        "(CSS selectors — e.g. `[\".cookie-banner\", \".breadcrumb\", \"#comments\"]`).\n"
        "- Use `css_selector` when the page has one clear main container (e.g. `\"main\"`, `\"article.post\"`, "
        "`\"#content\"`). Everything outside that element is discarded before markdown generation.\n"
        "- If noise still leaks through, `/online/ingest` with `chunking.strategy = \"contextual\"` helps the "
        "retrieval system suppress noisy chunks.\n\n"
        "**Internally configured Crawl4AI options (not user-facing):**\n\n"
        "| Crawl4AI Parameter | Value | Effect |\n"
        "|---|---|---|\n"
        "| `scan_full_page` | `true` | Scrolls and captures the entire page, not just the viewport |\n"
        "| `wait_until` | `networkidle` | Waits for all network requests to finish before capturing |\n"
        "| `delay_before_return_html` | `2.0s` | Extra wait after load for late-rendering JS content |\n"
        "| `magic` | `true` | Heuristic cleanup — auto-detects and extracts main content |\n"
        "| `remove_overlay_elements` | `true` | Automatically removes cookie banners, popups, modals |\n"
        "| `cache_mode` | `bypass` | Always fetches fresh content (API-level caching is in Redis) |\n"
        "| `headless` | `true` | Runs browser in headless mode |\n"
        "| `markdown_generator` | `DefaultMarkdownGenerator` + `PruningContentFilter(threshold=0.48)` | "
        "Attached only when `markdown_type=\"fit\"` — prunes low-density boilerplate nodes |\n\n"
        "---\n\n"
        "## Backend selection & fallback chain\n\n"
        "The `scraper` field selects the **primary** backend. The non-selected backend "
        "(plus raw httpx) remain as automatic fallbacks if the primary fails — so requests "
        "stay best-effort regardless of which backend you choose.\n\n"
        "| `scraper` | Order tried |\n"
        "|---|---|\n"
        "| `crawl4ai` (default) | Crawl4AI → Jina Reader → Raw httpx |\n"
        "| `jina` | Jina Reader → Crawl4AI → Raw httpx |\n\n"
        "Every per-request field (`markdown_type`, `exclude_tags`, `css_selector`) is mapped to "
        "each backend — fallbacks respect your request rather than silently reverting to defaults.\n\n"
        "| Field | Crawl4AI | Jina Reader | Raw httpx |\n"
        "|---|---|---|---|\n"
        "| `markdown_type=\"fit\"` | `PruningContentFilter` on `DefaultMarkdownGenerator` | header `X-Engine: readerlm-v2` | built-in noise strip |\n"
        "| `markdown_type=\"raw\"` / `\"citations\"` | no filter (default generator) | default engine (citations → same as raw) | default |\n"
        "| `exclude_tags` | `excluded_tags` param | header `X-Remove-Selector` | BeautifulSoup `decompose()` |\n"
        "| `css_selector` | `css_selector` param | header `X-Target-Selector` | pre-filter in `clean_html` |\n\n"
        "Backend characteristics:\n"
        "1. **Crawl4AI** — full JS rendering + heuristic/LLM extraction (best quality)\n"
        "2. **Jina Reader API** — Markdown extraction without JS rendering (requires `JINA_API_KEY`)\n"
        "3. **Raw httpx** — basic HTTP fetch, HTML-to-Markdown conversion (no JavaScript)\n\n"
        "---\n\n"
        "## Supported document types (for `inner_docs`)\n\n"
        "PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT, ODT, ODS, RTF, CSV\n\n"
        "## Supported image formats (for `inner_img`)\n\n"
        "JPG, JPEG, PNG, GIF, BMP, WEBP, SVG, TIFF, ICO\n\n"
        "---\n\n"
        "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.\n\n"
        "**Error codes:** `VALIDATION_URL_INVALID`, `SCRAPE_FAILED`, `SCRAPE_BLOCKED`, "
        "`SCRAPE_TIMEOUT`, `SCRAPE_EMPTY`, `SCRAPE_ROBOTS_BLOCKED`"
    ),
    response_description="Scraped page content as Markdown with metadata",
)
async def scrape(body: ScrapeRequest, request: Request) -> ResponseEnvelope[ScrapeData]:
    request_id = request.state.request_id
    scraper = request.app.state.scraping

    validation_error = _validate_url(body.url)
    if validation_error:
        return ResponseEnvelope(
            success=False,
            error=ErrorCode.VALIDATION_URL_INVALID,
            detail=validation_error,
            request_id=request_id,
        )

    options = ScrapeOptions(
        js_render=True,
        extract_links=True,
        with_links_summary=body.links_summary,
        timeout=30,
        markdown_type=body.markdown_type,
        exclude_tags=body.exclude_tags,
        css_selector=body.css_selector,
        scraper=body.scraper,
    )
    needs_fresh_fetch = body.links_summary or body.inner_img or body.inner_docs
    result = await scraper.scrape_url(
        body.url,
        options,
        bypass_cache=needs_fresh_fetch,
        request_id=request_id,
    )

    if result.status != ScrapeStatus.SUCCESS:
        error_code = _map_scrape_error(result.status, result.error)
        return ResponseEnvelope(
            success=False,
            error=error_code,
            detail=result.error,
            request_id=request_id,
        )

    # Crawl4AI's PruningContentFilter (active in `fit` mode) sometimes eats
    # pages whose main payload is short label/value pairs (government portals,
    # tabular data). If the fit-mode markdown looks suspiciously sparse
    # relative to the raw HTML we just fetched, re-run the scrape once in
    # `raw` mode before any downstream enrichment / parsing. Skipped when the
    # caller explicitly opted out of fit (raw / citations already bypass the
    # filter) or when the HTML is missing (no reliable signal for thinness).
    if (
        result.status == ScrapeStatus.SUCCESS
        and options.markdown_type == "fit"
        and _is_thin_output(result.markdown, result.html)
    ):
        log.info(
            "scrape_thin_output_retry_raw",
            url=body.url,
            markdown_len=len(result.markdown or ""),
            html_len=len(result.html or ""),
            word_count=len((result.markdown or "").split()),
        )
        raw_options = options.model_copy(update={"markdown_type": "raw"})
        retry = await scraper.scrape_url(
            body.url,
            raw_options,
            bypass_cache=True,
            request_id=request_id,
        )
        if retry.status == ScrapeStatus.SUCCESS and retry.markdown:
            result = retry

    content = result.markdown or ""
    if not content.strip():
        return ResponseEnvelope(
            success=False,
            error=ErrorCode.SCRAPE_EMPTY,
            detail="Page returned no extractable content",
            request_id=request_id,
        )

    content = await enrich_if_applicable(
        body.url,
        content,
        html=result.html,
        client=scraper.crawl4ai._client,
    )

    # ── Parse inner images if requested ──
    parser = request.app.state.parser
    inner_images: list[InnerImageData] | None = None
    if body.inner_img and result.html:
        discovered = discover_images(result.html, body.url)
        if discovered:
            inner_images = await _parse_inner_images(parser, discovered, request_id)

    # ── Parse inner documents if requested ──
    inner_documents: list[InnerDocData] | None = None
    if body.inner_docs and result.discovered_documents:
        parser = request.app.state.parser
        inner_documents = await _parse_inner_documents(
            parser, result.discovered_documents, request_id
        )

    # ── Build links summary if requested ──
    links_summary: LinksSummary | None = None
    if body.links_summary:
        if result.html:
            links_summary = _build_links_summary(
                result.html,
                result.url,
                include_documents=body.inner_docs,
                include_images=body.inner_img,
            )
        elif result.discovered_links or result.discovered_documents:
            links_summary = LinksSummary(
                urls=result.discovered_links,
                documents=[doc.url for doc in result.discovered_documents] if body.inner_docs else [],
                images=[],
            )
        else:
            raw_html = await _fetch_raw_html(result.url)
            links_summary = _build_links_summary(
                raw_html,
                result.url,
                include_documents=body.inner_docs,
                include_images=body.inner_img,
            )

    # ── Classify scraped content ──
    content_type, entities = await _classify_content(
        request.app.state.classifier,
        content,
        language=result.metadata.language,
        source_url=result.url,
    )

    return ResponseEnvelope(
        success=True,
        data=ScrapeData(
            url=result.url,
            title=result.metadata.title,
            content=content,
            content_length=len(content),
            language=result.metadata.language,
            links_found=len(result.discovered_links),
            last_modified=None,
            content_type=content_type,
            entities=entities,
            inner_images=inner_images,
            inner_documents=inner_documents,
            links_summary=links_summary,
        ),
        request_id=request_id,
    )


async def _fetch_raw_html(url: str) -> str:
    """Lightweight raw-HTML fetch for link discovery. Bypasses scraper pipelines
    so we never extract links from filtered/cleaned HTML. Returns empty on failure."""
    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except Exception as exc:
        log.warning("links_summary_raw_fetch_failed", url=url, error=str(exc))
        return ""


def _build_links_summary(
    html: str,
    base_url: str,
    *,
    include_documents: bool,
    include_images: bool,
) -> LinksSummary:
    if not html:
        return LinksSummary()
    docs, page_links = extract_documents_and_links(html, base_url)
    summary = LinksSummary(urls=page_links)
    if include_documents:
        summary.documents = [d.url for d in docs]
    if include_images:
        summary.images = [img.url for img in discover_images(html, base_url)]
    return summary


@router.post(
    "/crawl",
    summary="Discover URLs from a website",
    description=(
        "Discover all URLs on a website using either sitemap parsing or BFS link crawling.\n\n"
        "Each discovered URL is classified as either `page` (HTML) or `document` (PDF, DOCX, etc.).\n\n"
        "---\n\n"
        "## Discovery methods\n\n"
        "| Method | How it works | Best for |\n"
        "|--------|-------------|----------|\n"
        "| `sitemap` | Parses XML sitemaps (including nested sitemaps and robots.txt sitemap references) | "
        "Sites with well-maintained sitemaps — fast and complete |\n"
        "| `crawl` | Follows links via breadth-first search up to `max_depth` levels | "
        "Sites without sitemaps or when you want to discover linked documents |\n\n"
        "## Request fields\n\n"
        "| Field | Type | Required | Default | Description |\n"
        "|-------|------|----------|---------|-------------|\n"
        "| `url` | string | Required | — | Base URL or sitemap URL to crawl |\n"
        "| `method` | string | Required | — | `sitemap` or `crawl` |\n"
        "| `max_depth` | integer | Optional | `3` | Maximum link-following depth for crawl method (1–5) |\n"
        "| `max_urls` | integer | Optional | `500` | Maximum number of URLs to return (1–5000) |\n"
        "| `scraper` | string | Optional | `crawl4ai` | Preferred scraping backend used during BFS "
        "discovery (`crawl4ai` or `jina`). Ignored when `method=\"sitemap\"`. |\n\n"
        "---\n\n"
        "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.\n\n"
        "**Error codes:** `VALIDATION_URL_INVALID`, `CRAWL_SITEMAP_NOT_FOUND`"
    ),
    response_description="List of discovered URLs with type classification",
)
async def crawl(body: CrawlRequest, request: Request) -> ResponseEnvelope[CrawlData]:
    request_id = request.state.request_id
    scraper = request.app.state.scraping
    sitemap_parser = request.app.state.sitemap_parser

    validation_error = _validate_url(body.url)
    if validation_error:
        return ResponseEnvelope(
            success=False,
            error=ErrorCode.VALIDATION_URL_INVALID,
            detail=validation_error,
            request_id=request_id,
        )

    if body.method == "sitemap":
        urls = await sitemap_parser.parse(body.url, max_urls=body.max_urls)
        if not urls:
            return ResponseEnvelope(
                success=False,
                error=ErrorCode.CRAWL_SITEMAP_NOT_FOUND,
                detail="No URLs found in sitemap",
                request_id=request_id,
            )

        crawl_urls = []
        for u in urls:
            doc_type = document_type(u)
            crawl_urls.append(CrawlUrl(
                url=u,
                type="document" if doc_type else "page",
                last_modified=None,
            ))

        return ResponseEnvelope(
            success=True,
            data=CrawlData(
                base_url=body.url,
                method_used="sitemap",
                urls=crawl_urls,
                total_urls=len(crawl_urls),
            ),
            request_id=request_id,
        )

    # method == "crawl" — BFS discovery
    pages, docs = await scraper.discover_urls(
        body.url,
        max_depth=body.max_depth,
        max_pages=body.max_urls,
        same_domain_only=True,
        scraper=body.scraper,
    )

    crawl_urls = [CrawlUrl(url=u, type="page", last_modified=None) for u in pages]
    crawl_urls += [CrawlUrl(url=d.url, type="document", last_modified=None) for d in docs]

    return ResponseEnvelope(
        success=True,
        data=CrawlData(
            base_url=body.url,
            method_used="crawl",
            urls=crawl_urls,
            total_urls=len(crawl_urls),
        ),
        request_id=request_id,
    )


async def _parse_inner_images(
    parser, images: list, request_id: str
) -> list[InnerImageData]:
    """Parse each discovered image URL via the ParserService (LlamaParse OCR) concurrently."""

    async def _parse_one(img) -> InnerImageData:
        try:
            parse_result = await parser.parse_from_url(img.url)
            if parse_result.status == ParseStatus.SUCCESS and parse_result.text:
                return InnerImageData(
                    url=img.url,
                    alt=img.alt,
                    title=img.title,
                    content=parse_result.text,
                    content_length=len(parse_result.text),
                )
            else:
                return InnerImageData(
                    url=img.url,
                    alt=img.alt,
                    title=img.title,
                    error=parse_result.error or f"Parse failed: {parse_result.status.value}",
                )
        except Exception as exc:
            log.warning("inner_img_parse_failed", url=img.url, error=str(exc))
            return InnerImageData(
                url=img.url,
                alt=img.alt,
                title=img.title,
                error=str(exc),
            )

    results = await asyncio.gather(*[_parse_one(img) for img in images])
    return list(results)


async def _parse_inner_documents(
    parser, documents: list, request_id: str
) -> list[InnerDocData]:
    """Parse each discovered document URL via the ParserService concurrently."""

    async def _parse_one(doc) -> InnerDocData:
        try:
            parse_result = await parser.parse_from_url(doc.url)
            if parse_result.status == ParseStatus.SUCCESS:
                return InnerDocData(
                    url=doc.url,
                    title=doc.link_text or parse_result.metadata.title,
                    doc_type=doc.type,
                    content=parse_result.text,
                    pages=parse_result.pages_parsed,
                    content_length=len(parse_result.text) if parse_result.text else 0,
                    language=parse_result.metadata.language,
                )
            else:
                return InnerDocData(
                    url=doc.url,
                    title=doc.link_text,
                    doc_type=doc.type,
                    error=parse_result.error or f"Parse failed: {parse_result.status.value}",
                )
        except Exception as exc:
            log.warning("inner_doc_parse_failed", url=doc.url, error=str(exc))
            return InnerDocData(
                url=doc.url,
                title=doc.link_text,
                doc_type=doc.type,
                error=str(exc),
            )

    results = await asyncio.gather(*[_parse_one(doc) for doc in documents])
    return list(results)


async def _classify_content(
    classifier, content: str, language: str | None, source_url: str
) -> tuple[list[str], ClassifyEntities | None]:
    """Run the classifier over content and return (content_type, entities).

    Failures are logged and degraded to (['general'], None) — classification
    is informational on scrape/parse, so it should not fail the request.
    """
    try:
        result = await classifier.classify(content, language=language or "de")
    except Exception as exc:
        log.warning("classify_after_scrape_failed", url=source_url, error=str(exc))
        return (["general"], None)

    content_type = [result.category.value] + result.sub_categories
    entities = ClassifyEntities(
        dates=result.entities.dates,
        deadlines=result.entities.deadlines,
        amounts=result.entities.amounts,
        contacts=result.entities.contacts,
        departments=result.entities.departments,
    )
    return (content_type, entities)


def _map_scrape_error(status: str, error_msg: str | None) -> str:
    if status == ScrapeStatus.TIMEOUT:
        return ErrorCode.SCRAPE_TIMEOUT
    if status == ScrapeStatus.BLOCKED:
        error_lower = (error_msg or "").lower()
        if "robot" in error_lower:
            return ErrorCode.SCRAPE_ROBOTS_BLOCKED
        return ErrorCode.SCRAPE_BLOCKED
    return ErrorCode.SCRAPE_FAILED
