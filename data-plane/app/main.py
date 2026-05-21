import time
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import ext, settings
from app.dependencies.api_key import require_api_key
from app.middleware.hmac_auth import HMACAuthMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.routers.local import discover as local_discover
from app.routers.local import ingest as local_ingest
from app.routers.local import parse as local_parse
from app.routers.local import vectors as local_vectors
from app.routers.online import collections as online_collections
from app.routers.online import ingest as online_ingest
from app.routers.online import ingest_at as online_ingest_at
from app.routers.online import ingest_stream as online_ingest_stream
from app.routers.online import parse as online_parse
from app.routers.online import scrape as online_scrape
from app.routers.online import vectors as online_vectors
from app.routers.online import vectors_at as online_vectors_at
from app.routers.shared import classify, collections, health, metrics, probe, search
from app.services.discovery.discovery_service import DiscoveryService
from app.services.discovery.r2_client import R2Client
from app.services.discovery.smb_client import SMBClient
from app.services.embedding.bge_gemma2_client import BGEGemma2Client
from app.services.embedding.bge_m3_client import BGEM3Client
from app.services.embedding.openai_client import OpenAIEmbedClient
from app.services.embedding.tei_client_at import TEIEmbedClientAT
from app.services.embedding.tei_sparse_client_at import TEISparseClientAT
from app.services.embedding.qdrant_service import QdrantService
from app.services.ingest.ingest_service import IngestService
from app.services.intelligence.chunker import Chunker
from app.services.intelligence.classifier import Classifier
from app.services.intelligence.contextual import ContextualEnricher
from app.services.intelligence.funding_extractor import FundingExtractor
from app.services.intelligence import llm_router
from app.services.parsing.parser_service import ParserService
from app.services.scraping.scraper_service import ScraperService
from app.services.scraping.sitemap import SitemapParser
from app.services.search.search_service import SearchService
from app.utils.logger import get_logger, setup_logging

setup_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.start_time = time.monotonic()

    # Allow tests to inject fake services before TestClient startup
    if getattr(app.state, "_test_mode", False):
        log.info("app_started_test_mode")
        yield
        log.info("app_stopped_test_mode")
        return

    # AsyncExitStack runs cleanup callbacks in LIFO order on normal exit AND
    # on partial-startup failure — so a raise from any startup() below still
    # closes every client/pool that already came up. Without this, a failed
    # qdrant_at.startup() would leak the embedders, scraper, parser, etc.
    async with AsyncExitStack() as stack:
        # ── Scraping ─────────────────────────────────────
        scraping_svc = ScraperService()
        await scraping_svc.startup()
        stack.push_async_callback(scraping_svc.shutdown)
        app.state.scraping = scraping_svc

        sitemap_parser = SitemapParser()
        stack.push_async_callback(sitemap_parser.close)
        app.state.sitemap_parser = sitemap_parser

        # ── Parsing ──────────────────────────────────────
        parser_svc = ParserService()
        await parser_svc.startup()
        stack.push_async_callback(parser_svc.shutdown)
        app.state.parser = parser_svc

        # ── Intelligence ─────────────────────────────────
        # The LLM router owns the per-task AsyncOpenAI client cache. Close
        # it on shutdown so connection pools drain cleanly. Provider config
        # is resolved lazily from env on the first task ``startup()`` call,
        # so there's nothing to do here on the way up.
        stack.push_async_callback(llm_router.close_all)

        classifier = Classifier()
        app.state.classifier = classifier

        contextual_enricher = ContextualEnricher()
        await contextual_enricher.startup()
        stack.push_async_callback(contextual_enricher.shutdown)
        app.state.contextual_enricher = contextual_enricher

        funding_extractor = FundingExtractor()
        funding_extractor.startup()
        app.state.funding_extractor = funding_extractor

        # ── Embedding + Storage ──────────────────────────
        embedder = BGEM3Client()
        await embedder.startup()
        stack.push_async_callback(embedder.shutdown)
        app.state.embedder = embedder

        openai_embedder = OpenAIEmbedClient()
        await openai_embedder.startup()
        stack.push_async_callback(openai_embedder.shutdown)
        app.state.openai_embedder = openai_embedder

        tei_embedder_at = TEIEmbedClientAT()
        await tei_embedder_at.startup()
        stack.push_async_callback(tei_embedder_at.shutdown)
        app.state.tei_embedder_at = tei_embedder_at

        tei_sparse_embedder = TEISparseClientAT()
        await tei_sparse_embedder.startup()
        stack.push_async_callback(tei_sparse_embedder.shutdown)
        app.state.sparse_embedder = tei_sparse_embedder

        qdrant = QdrantService()
        await qdrant.startup()
        stack.push_async_callback(qdrant.shutdown)
        app.state.qdrant = qdrant

        # AT-specific Qdrant instance (used by POST /api/v1/online/ingest/at).
        # URL / port / api-key come from separate env vars — matches the upstream
        # qdrant-client pattern. Port defaults to 443 (standard HTTPS). When the
        # AT URL env var is unset, we fall back to the default qdrant_url and
        # skip the port kwarg so the URL's embedded port wins.
        if ext.qdrant_url_at:
            qdrant_at = QdrantService(
                url=ext.qdrant_url_at,
                port=ext.qdrant_port_at,
                api_key=ext.qdrant_api_key_at or ext.qdrant_api_key,
            )
        else:
            qdrant_at = QdrantService(
                url=ext.qdrant_url,
                api_key=ext.qdrant_api_key,
            )
        await qdrant_at.startup()
        stack.push_async_callback(qdrant_at.shutdown)
        app.state.qdrant_at = qdrant_at

        # ── Discovery ────────────────────────────────────
        smb_client = SMBClient()
        r2_client = R2Client()
        await r2_client.startup()
        stack.push_async_callback(r2_client.shutdown)
        app.state.discovery = DiscoveryService(smb_client, r2_client)
        app.state.r2_client = r2_client

        # ── Ingest + Search ──────────────────────────────
        chunker = Chunker()
        app.state.chunker = chunker
        app.state.ingest = IngestService(
            chunker, classifier, embedder, qdrant, contextual_enricher,
            sparse_embedder=tei_sparse_embedder,
        )
        app.state.online_ingest = IngestService(
            chunker, classifier, openai_embedder, qdrant, contextual_enricher,
            sparse_embedder=tei_sparse_embedder,
        )
        app.state.search = SearchService(
            openai_embedder,
            qdrant,
            sparse_embedder=tei_sparse_embedder,
            bge_m3_embedder=tei_embedder_at,
        )

        log.info("app_started", mode=settings.mode, version=settings.version)
        yield

    log.info("app_stopped")


tags_metadata = [
    {
        "name": "Health",
        "description": (
            "Liveness and readiness probes for container orchestrators and load balancers.\n\n"
            "**`GET /health`** — bare liveness, no auth, no downstream calls.\n\n"
            "**`GET /ready`** — readiness. Returns minimal `{ready: true/false}` without "
            "auth (suitable for an LB), or full per-service status when called with the "
            "internal auth header.\n\n"
            "For richer per-dependency health see the **Diagnostics** tag."
        ),
    },
    {
        "name": "Metrics",
        "description": "Prometheus-compatible metrics endpoint (`dp_` prefix).",
    },
    {
        "name": "Local - File Discovery",
        "description": "Scan SMB file shares or Cloudflare R2 buckets for new/changed documents. "
        "Returns file metadata, SHA-256 hashes, and NTFS ACLs for change detection.",
    },
    {
        "name": "Local - Document Parsing",
        "description": "Extract text, tables, and metadata from documents via file upload, SMB, or object storage.\n\n"
        "**Input methods:**\n"
        "- `POST /local/document-parse` with `source: smb` — parse from mounted file share\n"
        "- `POST /local/document-parse` with `source: r2` — parse from object storage via presigned URL\n"
        "- `POST /local/document-parse/upload` — upload a file directly\n\n"
        "**Supported formats:** PDF, DOCX, DOC, PPTX, ODT, XLSX, XLS, TXT, CSV, HTML, RTF.\n\n"
        "**Parser backends** (auto-selected at startup):\n"
        "- **Cloud parser** (when configured) — high-quality markdown extraction\n"
        "- **Local parsers** — PyMuPDF for PDF, python-docx for DOCX — lightweight, no heavy dependencies\n"
        "- **SpreadsheetParser** — always used for XLSX/XLS (openpyxl)\n"
        "- **TextParser** — always used for TXT, CSV, HTML, RTF",
    },
    {
        "name": "Local - Ingestion Pipeline",
        "description": "Full RAG ingestion pipeline for local documents: chunk → classify → embed (BGE-M3) → store (Qdrant).\n\n"
        "**Key features:**\n"
        "- Caller specifies the target `collection_name` (multi-tenant)\n"
        "- ACL-aware payloads with visibility-based permission filtering\n"
        "- Idempotent: re-ingesting the same `source_id` replaces old vectors automatically",
    },
    {
        "name": "Local - Vector Management",
        "description": "Delete vectors or update ACL permissions on existing vector points.\n\n"
        "- `DELETE /local/vectors/{source_id}?collection_name=...` — remove all vectors for a document\n"
        "- `PUT /local/vectors/update-acl` — update ACL payload on vectors without re-embedding",
    },
    {
        "name": "Online - Collection Management",
        "description": "List and inspect available Qdrant collections.\n\n"
        "**Optional X-API-Key header** when API-key auth is configured.",
    },
    {
        "name": "Online - Web Scraping",
        "description": "Scrape webpages and discover URLs from sitemaps or BFS crawling.\n\n"
        "**Optional X-API-Key header** when API-key auth is configured.",
    },
    {
        "name": "Online - Document Parsing",
        "description": "Parse documents from public URLs or upload document files directly.\n\n"
        "**Input methods:**\n"
        "- `POST /online/document-parse` — parse from a public URL\n"
        "- `POST /online/document-parse/upload` — upload a file directly\n\n"
        "**Optional X-API-Key header** when API-key auth is configured.",
    },
    {
        "name": "Online - Ingestion Pipeline",
        "description": "Full RAG ingestion pipeline for web-scraped content: chunk → classify → embed → store (Qdrant).\n\n"
        "Caller picks one of two embedding models per-request via `embedding_model`:\n"
        "- `openai` (default) — `text-embedding-3-small` (1536-dim, stored as `dense_openai`)\n"
        "- `bge_m3` — BGE-M3 via the configured TEI endpoint (1024-dim, stored as `dense_bge_m3`)\n\n"
        "With `search_mode: hybrid` the point also carries a `sparse` vector produced by the "
        "configured TEI sparse endpoint.\n\n"
        "**Optional X-API-Key header** when API-key auth is configured.",
    },
    {
        "name": "Online - Ingestion Pipeline (AT)",
        "description": "Dedicated ingest for the Austrian funding assistant (`POST /api/v1/online/ingest/at`). "
        "Runs against a separate Qdrant instance.\n\n"
        "**Caller picks the target collection** via `collection_name` — it is auto-created on first use "
        "with the AT legacy schema (single unnamed cosine vector at the embedder's dim, plus keyword "
        "indexes on `metadata.source_id` / `metadata.source_url`). There is **no per-province collection "
        "routing** — `state_or_province` is stored as metadata only (English lowercase) for search-time "
        "filtering, with the request-body value overriding the funding extractor's output when supplied.\n\n"
        "Country (AT) and assistant type (funding) are implicit. The funding extractor always runs and "
        "its output is merged into `metadata.*`.\n\n"
        "**Embedding model** is selectable per request via `embedding_model`: `bge_m3` (default) uses "
        "the configured TEI endpoint (1024-dim); `openai` uses `text-embedding-3-small` (1536-dim). "
        "Switching models requires a new collection name.",
    },
    {
        "name": "Online - Vector Management",
        "description": "Delete or sparse-encode vectors on the default online Qdrant instance.\n\n"
        "- `DELETE /online/vectors/{source_id}?collection_name=...` — remove all vectors for a document\n"
        "- `POST /online/vectors/delete-by-filter` — remove vectors matching metadata filters (AND-combined)\n"
        "- `POST /online/vectors/sparse-encode` — return the configured TEI sparse vector for arbitrary text "
        "(same encoder used by hybrid ingest/search)\n\n"
        "**Optional X-API-Key header** when API-key auth is configured.",
    },
    {
        "name": "Online - Vector Management (AT)",
        "description": "Delete vectors on the AT Qdrant instance.\n\n"
        "- `DELETE /online/vectors/at/{source_id}?collection_name=...` — mirrors the default delete "
        "endpoint, only the target Qdrant instance differs. `collection_name` must be one of the "
        "nine AT province collections (`Burgenland`, `Kärnten`, `Niederösterreich`, `Oberösterreich`, "
        "`Salzburg`, `Steiermark`, `Tirol`, `Vorarlberg`, `Wien`).\n\n"
        "**Optional X-API-Key header** when API-key auth is configured.",
    },
    {
        "name": "Online - Ingestion Pipeline (AT)",
        "description": "Dedicated ingest for the Austrian funding assistant (`POST /api/v1/online/ingest/at`). "
        "Runs against a separate Qdrant instance (configured via `QDRANT_URL_AT` / `QDRANT_PORT_AT` / `QDRANT_API_KEY_AT`) "
        "with per-province collections: `Burgenland`, `Kärnten`, `Niederösterreich`, `Oberösterreich`, "
        "`Salzburg`, `Steiermark`, `Tirol`, `Vorarlberg`, `Wien`.\n\n"
        "Country (AT) and assistant type (funding) are implicit. The funding extractor's "
        "`state_or_province` output selects target collections; an empty list fans out to all nine. "
        "Callers can override by supplying `state_or_province` (German or English lowercase forms).",
    },
    {
        "name": "Content Intelligence",
        "description": "Classify municipality content into 9 categories (funding, event, policy, contact, form, announcement, minutes, report, general) "
        "and extract structured entities (dates, deadlines, monetary amounts, email contacts, departments).",
    },
    {
        "name": "Semantic Search",
        "description": "Permission-aware semantic and hybrid search across Qdrant collections.\n\n"
        "**Search modes:**\n"
        "- `semantic` (default) — dense-only cosine search against whichever dense vector the "
        "collection was ingested with (`dense_openai` or `dense_bge_m3`)\n"
        "- `hybrid` — dense + sparse combined with Reciprocal Rank Fusion (RRF)\n\n"
        "The caller tells search which dense vector to hit via `embedding_model` "
        "(`openai` → `dense_openai`, `bge_m3` → `dense_bge_m3`) — this must match the "
        "model used when the collection was ingested.\n\n"
        "**Key features:**\n"
        "- Caller specifies the target `collection_name` to search in\n"
        "- Mandatory user context for ACL filtering (citizen → public only; employee → public + internal with AD group intersection)\n"
        "- Results include `municipality_id`, `department`, entity data, and content type\n"
        "- Optional filtering by content type (e.g. `funding`, `policy`)",
    },
    {
        "name": "Collection Management",
        "description": "Create and inspect Qdrant vector collections for municipality tenants. "
        "Online collections store a single dense vector named `dense_openai` or `dense_bge_m3` "
        "plus an optional `sparse` vector for hybrid search. "
        "Local collections use BGE-M3.",
    },
]

app = FastAPI(
    title="KI² Data Plane",
    description=(
        "Unified ingestion, embedding, and permission-aware search for municipality RAG pipelines.\n\n"
        "## Two Operational Modes\n\n"
        "### 1. Online Mode — Knowledgebase from Web Content (`/api/v1/online/...`)\n"
        "Update the knowledgebase using online URLs and cloud services. **Requires X-API-Key header.**\n"
        "- **Scrape** web pages, discover URLs from sitemaps\n"
        "- **Parse** documents from any public URL using a cloud parser for high-quality extraction\n"
        "- **Ingest** scraped/parsed content into Qdrant vector collections — caller picks "
        "`text-embedding-3-small` (1536-dim) or BGE-M3 (1024-dim) per request\n"
        "- **AT funding pipeline** — `POST /api/v1/online/ingest/at` is a dedicated endpoint that writes to a "
        "separate Qdrant instance with nine per-province collections.\n\n"
        "### 2. Local Mode — Fully Offline Document Processing (`/api/v1/local/...`)\n"
        "Process documents entirely locally without any third-party APIs. **No API key required.**\n"
        "- **Upload** documents directly via `POST /local/document-parse/upload` or read from **SMB file shares**\n"
        "- **Parse** locally using **PyMuPDF** (PDF) and **python-docx** (DOCX) — lightweight, no GPU or heavy dependencies\n"
        "- **Discover** files from SMB shares with NTFS ACL extraction\n\n"
        "## Authentication\n"
        "- **HMAC auth** (all endpoints except `/health`): enabled when the deployment is configured for it.\n"
        "- **API key auth** (online endpoints only): optional — when enabled clients send `X-API-Key`. "
        "If not configured, online endpoints are open.\n\n"
        "## Pipeline Flow\n"
        "1. **Discover** → Scan file sources for new/changed documents\n"
        "2. **Scrape / Parse** → Extract text from web pages or documents (URL, upload, SMB, object storage)\n"
        "3. **Ingest** → Chunk, classify, embed, and store in Qdrant with metadata\n"
        "4. **Search** → Permission-filtered semantic search across collections\n"
    ),
    version=settings.version,
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

# Middleware (applied in reverse order — last added runs first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)
app.add_middleware(HMACAuthMiddleware)
app.add_middleware(RequestIDMiddleware)

# ── Shared Routers ────────────────────────────────────
app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(classify.router)
app.include_router(collections.router)
app.include_router(search.router)
# Probe router declares its own `Depends(require_api_key)` at the router level,
# matching the `/model-health` pattern. No additional dependency needed here.
app.include_router(probe.router)

# ── Local Routers (no API key required) ───────────────
app.include_router(local_parse.router)
app.include_router(local_ingest.router)
app.include_router(local_discover.router)
app.include_router(local_vectors.router)

# ── Online Routers (API key required) ─────────────────
app.include_router(online_collections.router, dependencies=[Depends(require_api_key)])
app.include_router(online_scrape.router, dependencies=[Depends(require_api_key)])
app.include_router(online_parse.router, dependencies=[Depends(require_api_key)])
app.include_router(online_ingest.router, dependencies=[Depends(require_api_key)])
app.include_router(online_ingest_at.router, dependencies=[Depends(require_api_key)])
app.include_router(online_ingest_stream.router, dependencies=[Depends(require_api_key)])
app.include_router(online_vectors.router, dependencies=[Depends(require_api_key)])
app.include_router(online_vectors_at.router, dependencies=[Depends(require_api_key)])
