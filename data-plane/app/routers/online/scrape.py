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
    DiscoveredImage,
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
# Austrian government portals). When BOTH thresholds trip we re-scrape in
# `raw` mode to recover the payload (see ``_is_thin_output`` for the AND).
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
        "Scrape a webpage using the **Jina Reader API** (default) or **Crawl4AI** "
        "and return the extracted content as clean Markdown. Backend is selectable per request via "
        "the `scraper` field (default `jina`). Results are cached in Redis.\n\n"
        "---\n\n"
        "## How content extraction works\n\n"
        "The scraper processes content in multiple stages:\n\n"
        "1. **Page fetch** — Jina Reader fetches the page through its hosted Chromium engine "
        "(`X-Engine: browser`) with images suppressed (`X-Retain-Images: none`). The Crawl4AI fallback "
        "calls `POST /md`, which runs Crawl4AI's headless browser server-side and returns Markdown directly. "
        "A default noise-strip list covering OneTrust, Cookiebot, Osano, Quantcast, TrustArc, Termly, Klaro, "
        "Usercentrics, Didomi, Axeptio, Sourcepoint, and common cookie-banner classes is merged into "
        "`exclude_tags` automatically — applied on the Jina branch (via `X-Remove-Selector`) and on the "
        "raw httpx fallback (via BeautifulSoup `decompose`). The Crawl4AI `/md` branch does not accept "
        "selector overrides, so the merged list is ignored there.\n"
        "2. **Markdown extraction** — controlled by `markdown_type`:\n"
        "   - `fit` (default) — Jina returns Chromium-engine markdown; Crawl4AI fallback uses `/md` `f=fit`. "
        "On the Crawl4AI branch, if `f=fit` returns suspiciously short markdown (fewer than ~20 words — the "
        "PruningContentFilter sometimes strips tabular/label-value pages too aggressively), the client "
        "auto-retries once with `f=raw` and returns the richer output. This is a per-request retry inside "
        "the Crawl4AI client and is separate from the router-level retry that fires on the httpx fallback "
        "(which compares markdown length against raw HTML).\n"
        "   - `raw` — full page Markdown including headers/nav/footer (Jina default engine; Crawl4AI `/md` `f=raw`).\n"
        "   - `citations` — full content with citation links preserved (best-effort: the Crawl4AI `/md` "
        "endpoint does not expose a citations filter, so this currently degrades to `raw` on every backend).\n"
        "3. **Tag exclusion** — if `exclude_tags` is set, those selectors are honored on the Jina branch "
        "(`X-Remove-Selector`) and the httpx fallback (BeautifulSoup `decompose`). The Crawl4AI `/md` "
        "endpoint does not accept selector overrides, so per-tag exclusion is skipped on that branch.\n"
        "4. **Scoping** — if `css_selector` is set, extraction is scoped on Jina (`X-Target-Selector`) "
        "and httpx (pre-filter). The Crawl4AI `/md` branch ignores it (server-side defaults only).\n"
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
        "| `markdown_type` | string | Optional | `fit` | `fit` = main content only. `raw` = full page. "
        "`citations` = full content with citation links (best-effort; currently degrades to `raw`). |\n"
        "| `exclude_tags` | string[] | Optional | `null` | CSS selectors / tag names to drop before extraction "
        "(e.g. `['nav','footer','.sidebar']`). Honored on Jina + httpx; ignored by Crawl4AI `/md`. |\n"
        "| `css_selector` | string | Optional | `null` | CSS selector to scope extraction to a specific element "
        "(e.g. `'main'` or `'article.content'`). Honored on Jina + httpx; ignored by Crawl4AI `/md`. |\n"
        "| `inner_img` | boolean | Optional | `false` | Extract and OCR-parse images found on the page "
        "(returns alt text, URL, and extracted text content via LlamaParse) |\n"
        "| `inner_docs` | boolean | Optional | `false` | Extract and parse documents (PDF, DOCX, XLSX, PPTX, etc.) "
        "linked on the page using the document parsing backend |\n"
        "| `scraper` | string | Optional | `jina` | Preferred scraping backend: `jina` (default), "
        "`crawl4ai`, or `firecrawl` (when configured). The other backends and raw httpx remain as "
        "automatic fallbacks if the primary fails. The default can be overridden globally by deployment configuration. |\n"
        "| `links_summary` | boolean | Optional | `false` | If true, adds a `links_summary.urls` list "
        "to the response — deduped http/https page links extracted from the **raw** page HTML "
        "(so nav/footer links filtered by `markdown_type='fit'` aren't missed). "
        "`links_summary.documents` is populated only when `inner_docs=true`; "
        "`links_summary.images` is populated only when `inner_img=true`. "
        "Triggers one extra lightweight raw-HTML fetch. |\n"
        "| `bypass_cache` | boolean | Optional | `false` | If true, skip the Redis cache and force a "
        "fresh origin fetch (cache is then updated with the new content). `links_summary`, "
        "`inner_img`, and `inner_docs` already imply a fresh fetch. |\n\n"
        "---\n\n"
        "## Examples\n\n"
        "**Default — clean main content only (Jina):**\n"
        "```json\n"
        "{ \"url\": \"https://transparenzportal.gv.at/tdb/tp/leistung/1051580.html\" }\n"
        "```\n\n"
        "**Scope to `<main>` and drop nav/footer/sidebar (Jina honors both):**\n"
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
        "**Force the Crawl4AI `/md` backend as primary:**\n"
        "```json\n"
        "{ \"url\": \"https://example.com/article\", \"scraper\": \"crawl4ai\" }\n"
        "```\n\n"
        "---\n\n"
        "## Content filtering tips\n\n"
        "- `markdown_type: \"fit\"` (default) usually produces the cleanest content. For pages with good "
        "semantic HTML (`<main>`, `<article>`), this is all you need.\n"
        "- For sites with site-specific noise blocks, add them to `exclude_tags` "
        "(CSS selectors — e.g. `[\".cookie-banner\", \".breadcrumb\", \"#comments\"]`). Note these are "
        "honored on Jina and httpx but not on the Crawl4AI `/md` branch.\n"
        "- Use `css_selector` when the page has one clear main container (e.g. `\"main\"`, `\"article.post\"`, "
        "`\"#content\"`). Same backend caveat applies.\n"
        "- If noise still leaks through, `/online/ingest` with `chunking.strategy = \"contextual\"` helps the "
        "retrieval system suppress noisy chunks.\n\n"
        "---\n\n"
        "## Backend selection & fallback chain\n\n"
        "The `scraper` field selects the **primary** backend. The non-selected backend "
        "(plus raw httpx) remain as automatic fallbacks if the primary fails — so requests "
        "stay best-effort regardless of which backend you choose.\n\n"
        "| `scraper` | Order tried |\n"
        "|---|---|\n"
        "| `jina` (default) | Jina Reader → Crawl4AI `/md` → Raw httpx |\n"
        "| `crawl4ai` | Crawl4AI `/md` → Jina Reader → Raw httpx |\n\n"
        "| Field | Jina Reader | Crawl4AI `/md` | Raw httpx |\n"
        "|---|---|---|---|\n"
        "| `markdown_type=\"fit\"` | `X-Engine: browser` + `X-Retain-Images: none` | `f=fit` filter | built-in noise strip |\n"
        "| `markdown_type=\"raw\"` / `\"citations\"` | default engine (citations → same as raw) | `f=raw` filter | default |\n"
        "| `exclude_tags` | header `X-Remove-Selector` | not supported (server-side defaults) | BeautifulSoup `decompose()` |\n"
        "| `css_selector` | header `X-Target-Selector` | not supported (server-side defaults) | pre-filter in `clean_html` |\n\n"
        "Backend characteristics:\n"
        "1. **Jina Reader API** — Chromium-engine Markdown extraction (default)\n"
        "2. **Crawl4AI `/md`** — server-side rendered Markdown\n"
        "3. **Raw httpx** — basic HTTP fetch, HTML-to-Markdown conversion (no JavaScript)\n\n"
        "---\n\n"
        "## Supported document types (for `inner_docs`)\n\n"
        "PDF, DOCX, DOC, XLSX, XLS, PPTX, PPT, ODT, ODS, RTF, CSV\n\n"
        "## Supported image formats (for `inner_img`)\n\n"
        "JPG, JPEG, PNG, GIF, BMP, WEBP, SVG, TIFF, ICO\n\n"
        "---\n\n"
        "**Optional X-API-Key header** when API-key auth is configured.\n\n"
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
        inner_img=body.inner_img,
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
        bypass_cache=needs_fresh_fetch or body.bypass_cache,
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
    # Status is guaranteed SUCCESS here — the non-success early return above
    # already short-circuits.
    if (
        options.markdown_type == "fit"
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
    if body.inner_img:
        if result.html:
            discovered = discover_images(result.html, body.url)
        elif result.discovered_images:
            # Jina path: no rendered HTML, so use the URLs Jina already
            # surfaced via X-With-Images-Summary. No alt/title from Jina.
            discovered = [DiscoveredImage(url=u) for u in result.discovered_images]
        else:
            discovered = []
        if discovered:
            inner_images = await _parse_inner_images(parser, discovered, request_id)

    # ── Parse inner documents if requested ──
    inner_documents: list[InnerDocData] | None = None
    if body.inner_docs and result.discovered_documents:
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
        elif result.discovered_links or result.discovered_documents or result.discovered_images:
            # Jina path (no HTML) — fall back to what the backend reported
            # directly. ``discovered_images`` is only populated by Jina.
            links_summary = LinksSummary(
                urls=result.discovered_links,
                documents=[doc.url for doc in result.discovered_documents] if body.inner_docs else [],
                images=result.discovered_images if body.inner_img else [],
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
            scraper_used=result.scraper_used,
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
        "| `crawl` | Breadth-first link discovery up to `max_depth` levels. Backend selected by the "
        "`scraper` field below. | Sites without sitemaps or when you want to enumerate linked documents |\n\n"
        "---\n\n"
        "## BFS backends (`method=\"crawl\"` only)\n\n"
        "The `scraper` field controls **how** the BFS runs. All three return the same response shape — "
        "they only differ in cost, speed, and whether JavaScript is rendered.\n\n"
        "| `scraper` | How it works | Cost / speed | When to pick it |\n"
        "|---|---|---|---|\n"
        "| `httpx` (**default**) | In-process Python BFS. Each URL is fetched with raw `httpx` "
        "(no headless browser, no JS). Links extracted from the served HTML. | Cheapest — no third-party "
        "calls, no Chromium, ~10–100× faster than the browser backends for the same crawl. | Default. "
        "Sufficient for sites with **server-rendered nav menus** (most municipality / government portals). |\n"
        "| `crawl4ai` | One round-trip to Crawl4AI's `POST /crawl` with `BFSDeepCrawlStrategy`. The "
        "Crawl4AI server runs the BFS itself, and returns the visited URL set + per-page link map in "
        "a single response. **Falls back to the Python BFS over httpx if the deep-crawl call fails** "
        "(server unsupported, timeout, etc.) — the response then reports `scraper_used: \"httpx\"`. | "
        "Heavier (server-side Chromium per page), but a single HTTP round-trip from the data-plane. | "
        "Deep / large crawls where you want server-side concurrency, filter chains, or relevance pruning. |\n"
        "| `jina` | In-process Python BFS, but each URL is fetched through Jina Reader's hosted Chromium "
        "engine. **One Jina call per discovered URL.** | Most expensive — burns Jina API quota on every "
        "page and pays Chromium latency per fetch. | Niche: only when a site's nav links are JS-injected "
        "and `httpx` genuinely misses links. |\n"
        "| `firecrawl` | One round-trip to Firecrawl's `POST /v2/map`. Firecrawl returns a flat "
        "domain-wide URL list in a single response (no per-page fetching, no depth control — "
        "`max_depth` is ignored). **Falls back to the Python BFS over httpx if the map call fails.** | "
        "Single API call regardless of site size (when configured). | Large sites where "
        "you want one-shot URL enumeration without paying per-page fetch costs. |\n\n"
        "Ignored when `method=\"sitemap\"` (the sitemap branch parses XML — no scraper involved).\n\n"
        "---\n\n"
        "## Request fields\n\n"
        "| Field | Type | Required | Default | Description |\n"
        "|-------|------|----------|---------|-------------|\n"
        "| `url` | string | Required | — | Base URL or sitemap URL to crawl |\n"
        "| `method` | string | Required | — | `sitemap` or `crawl` |\n"
        "| `max_depth` | integer | Optional | `3` | Maximum link-following depth for crawl method (1–5) |\n"
        "| `max_urls` | integer | Optional | `500` | Maximum number of URLs to return (1–5000) |\n"
        "| `scraper` | string | Optional | `httpx` | BFS backend — `httpx` (default), `crawl4ai` "
        "(server-side BFS), `jina` (per-URL Chromium fan-out), or `firecrawl` (single-shot `/v2/map`). "
        "See the *BFS backends* table above. Ignored when `method=\"sitemap\"`. |\n\n"
        "---\n\n"
        "## Response\n\n"
        "Returns `CrawlData` with `urls`, `total_urls`, `method_used`, and `scraper_used`. "
        "`scraper_used` reports the backend that actually produced the BFS results: `\"httpx\"`, "
        "`\"crawl4ai\"`, `\"jina\"`, or `\"firecrawl\"`. It is `null` for `method=\"sitemap\"` "
        "(no scraping involved). When a `crawl4ai` or `firecrawl` request falls back to the Python "
        "BFS, `scraper_used` reports `\"httpx\"` — the actual backend that produced the results, "
        "not the originally requested one.\n\n"
        "---\n\n"
        "**Optional X-API-Key header** when API-key auth is configured.\n\n"
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
                scraper_used=None,
            ),
            request_id=request_id,
        )

    # method == "crawl" — BFS discovery
    pages, docs, scraper_used = await scraper.discover_urls(
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
            scraper_used=scraper_used,
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
