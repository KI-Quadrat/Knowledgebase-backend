import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

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
from app.routers.shared import classify, collections, health, metrics, search
from app.services.discovery.discovery_service import DiscoveryService
from app.services.discovery.r2_client import R2Client
from app.services.discovery.smb_client import SMBClient
from app.services.embedding.bge_gemma2_client import BGEGemma2Client
from app.services.embedding.bge_m3_client import BGEM3Client
from app.services.embedding.openai_client import OpenAIEmbedClient
from app.services.embedding.tei_client_at import TEIEmbedClientAT
from app.services.embedding.qdrant_service import QdrantService
from app.services.ingest.ingest_service import IngestService
from app.services.intelligence.chunker import Chunker
from app.services.intelligence.classifier import Classifier
from app.services.intelligence.contextual import ContextualEnricher
from app.services.intelligence.funding_extractor import FundingExtractor
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

    # ── Scraping ─────────────────────────────────────
    scraping_svc = ScraperService()
    await scraping_svc.startup()
    app.state.scraping = scraping_svc

    sitemap_parser = SitemapParser()
    app.state.sitemap_parser = sitemap_parser

    # ── Parsing ──────────────────────────────────────
    parser_svc = ParserService()
    await parser_svc.startup()
    app.state.parser = parser_svc

    # ── Intelligence ─────────────────────────────────
    classifier = Classifier()
    app.state.classifier = classifier

    contextual_enricher = ContextualEnricher()
    await contextual_enricher.startup()
    app.state.contextual_enricher = contextual_enricher

    funding_extractor = FundingExtractor()
    funding_extractor.startup()
    app.state.funding_extractor = funding_extractor

    # ── Embedding + Storage ──────────────────────────
    embedder = BGEM3Client()
    await embedder.startup()
    app.state.embedder = embedder

    openai_embedder = OpenAIEmbedClient()
    await openai_embedder.startup()
    app.state.openai_embedder = openai_embedder

    tei_embedder_at = TEIEmbedClientAT()
    await tei_embedder_at.startup()
    app.state.tei_embedder_at = tei_embedder_at

    bge_gemma2_embedder = BGEGemma2Client()
    await bge_gemma2_embedder.startup()
    app.state.bge_gemma2_embedder = bge_gemma2_embedder

    qdrant = QdrantService()
    await qdrant.startup()
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
    app.state.qdrant_at = qdrant_at

    # ── Discovery ────────────────────────────────────
    smb_client = SMBClient()
    r2_client = R2Client()
    await r2_client.startup()
    app.state.discovery = DiscoveryService(smb_client, r2_client)
    app.state.r2_client = r2_client

    # ── Ingest + Search ──────────────────────────────
    chunker = Chunker()
    app.state.chunker = chunker
    app.state.ingest = IngestService(chunker, classifier, embedder, qdrant, contextual_enricher)
    app.state.online_ingest = IngestService(
        chunker, classifier, openai_embedder, qdrant, contextual_enricher,
        fallback_embedder=bge_gemma2_embedder,
    )
    app.state.search = SearchService(openai_embedder, qdrant, fallback_embedder=bge_gemma2_embedder)

    log.info("app_started", mode=settings.mode, version=settings.version)
    yield

    # ── Shutdown ─────────────────────────────────────
    await scraping_svc.shutdown()
    await parser_svc.shutdown()
    await embedder.shutdown()
    await openai_embedder.shutdown()
    await tei_embedder_at.shutdown()
    await bge_gemma2_embedder.shutdown()
    await contextual_enricher.shutdown()
    await qdrant.shutdown()
    await qdrant_at.shutdown()
    await r2_client.shutdown()
    await sitemap_parser.close()

    log.info("app_stopped")


tags_metadata = [
    {
        "name": "Health",
        "description": "Liveness and readiness probes for container orchestrators and load balancers. "
        "The `/ready` endpoint checks connectivity to Qdrant, BGE-M3, OpenAI, BGE-Gemma2 (LiteLLM), Parser (LlamaParse or local), Crawl4AI, LDAP, and Redis.",
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
        "description": "Extract text, tables, and metadata from documents via file upload, SMB, or R2.\n\n"
        "**Input methods:**\n"
        "- `POST /local/document-parse` with `source: smb` — parse from mounted file share\n"
        "- `POST /local/document-parse` with `source: r2` — parse from Cloudflare R2 via presigned URL\n"
        "- `POST /local/document-parse/upload` — upload a file directly\n\n"
        "**Supported formats:** PDF, DOCX, DOC, PPTX, ODT, XLSX, XLS, TXT, CSV, HTML, RTF.\n\n"
        "**Parser backends** (auto-selected at startup):\n"
        "- **LlamaParse** (cloud) — `LLAMA_CLOUD_API_KEY` set → high-quality markdown extraction via LlamaCloud API\n"
        "- **Local parsers** (no API key) — PyMuPDF for PDF, python-docx for DOCX — lightweight, no heavy dependencies\n"
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
        "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.",
    },
    {
        "name": "Online - Web Scraping",
        "description": "Scrape webpages via Crawl4AI and discover URLs from sitemaps or BFS crawling.\n\n"
        "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.",
    },
    {
        "name": "Online - Document Parsing",
        "description": "Parse documents from public URLs or upload document files directly.\n\n"
        "**Input methods:**\n"
        "- `POST /online/document-parse` — parse from a public URL\n"
        "- `POST /online/document-parse/upload` — upload a file directly\n\n"
        "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.",
    },
    {
        "name": "Online - Ingestion Pipeline",
        "description": "Full RAG ingestion pipeline for web-scraped content: chunk → classify → embed (multi-vector: OpenAI + BGE-Gemma2 via LiteLLM) → store (Qdrant).\n\n"
        "Every point stores two dense vectors: `dense_openai` (primary) and `dense_bge_gemma2` (fallback). "
        "If one embedder is unavailable during ingest, the point is still stored with the other's vector.\n\n"
        "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.",
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
        "- `semantic` (default) — dense-only cosine search via OpenAI `text-embedding-3-small` (fallback to BGE-Gemma2 via LiteLLM)\n"
        "- `hybrid` — dense (OpenAI or BGE-Gemma2 fallback) + sparse (BM25) with Reciprocal Rank Fusion (RRF)\n\n"
        "**Embedding fallback:** OpenAI is the primary embedder. If unavailable, the service automatically "
        "falls back to BGE-Gemma2 via self-hosted LiteLLM and searches the `dense_bge_gemma2` vectors.\n\n"
        "**Key features:**\n"
        "- Caller specifies the target `collection_name` to search in\n"
        "- Mandatory user context for ACL filtering (citizen → public only; employee → public + internal with AD group intersection)\n"
        "- Results include `municipality_id`, `department`, entity data, and content type\n"
        "- Optional filtering by content type (e.g. `funding`, `policy`)",
    },
    {
        "name": "Collection Management",
        "description": "Create and inspect Qdrant vector collections for municipality tenants. "
        "Online collections store multi-vector embeddings: `dense_openai` + `dense_bge_gemma2` (fallback) "
        "and optional BM25 `sparse` vectors for hybrid search. Local collections use BGE-M3.",
    },
]

app = FastAPI(
    title="KI² Data Plane",
    description=(
        "Unified ingestion, embedding, and permission-aware search for municipality RAG pipelines.\n\n"
        "## Two Operational Modes\n\n"
        "### 1. Online Mode — Knowledgebase from Web Content (`/api/v1/online/...`)\n"
        "Update the knowledgebase using online URLs and cloud services. **Requires X-API-Key header.**\n"
        "- **Scrape** web pages via Crawl4AI, discover URLs from sitemaps\n"
        "- **Parse** documents from any public URL — uses **LlamaParse** (cloud) for high-quality extraction\n"
        "- **Ingest** scraped/parsed content into Qdrant vector collections with multi-vector embeddings "
        "(OpenAI primary + BGE-Gemma2 fallback via LiteLLM)\n"
        "- **AT funding pipeline** — `POST /api/v1/online/ingest/at` is a dedicated endpoint that writes to a "
        "separate Qdrant instance (configured via `QDRANT_URL_AT` / `QDRANT_PORT_AT` / `QDRANT_API_KEY_AT`) "
        "with nine per-province collections.\n"
        "- Requires: `CRAWL4AI_URL`, `LLAMA_CLOUD_API_KEY` (optional), `OPENAI_API_KEY` (for classification + primary embedding), "
        "`LITELLM_URL` (for BGE-Gemma2 fallback embedding)\n\n"
        "### 2. Local Mode — Fully Offline Document Processing (`/api/v1/local/...`)\n"
        "Process documents entirely locally without any third-party APIs. **No API key required.**\n"
        "- **Upload** documents directly via `POST /local/document-parse/upload` or read from **SMB file shares**\n"
        "- **Parse** locally using **PyMuPDF** (PDF) and **python-docx** (DOCX) — lightweight, no GPU or heavy dependencies\n"
        "- **Discover** files from SMB shares with NTFS ACL extraction\n"
        "- Requires: No external API keys — only Qdrant and BGE-M3 for embedding/search\n\n"
        "## Authentication\n"
        "- **HMAC auth** (all endpoints except `/health`): Set `DP_HMAC_SECRET` to enable.\n"
        "- **API key auth** (online endpoints only): Optional. Set `DP_ONLINE_API_KEYS` to enable — clients must then send `X-API-Key` header. "
        "If not configured, online endpoints are open.\n\n"
        "## Pipeline Flow\n"
        "1. **Discover** → Scan file sources (SMB shares, Cloudflare R2) for new/changed documents\n"
        "2. **Scrape / Parse** → Extract text from web pages (Crawl4AI) or documents (URL, upload, SMB, R2)\n"
        "3. **Ingest** → Chunk, classify, embed (OpenAI + BGE-Gemma2 fallback for online / BGE-M3 for local), and store in Qdrant with metadata\n"
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
