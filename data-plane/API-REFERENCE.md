# KI² Data Plane — API Reference

Complete endpoint reference for the KI² Data Plane Service.

**Swagger UI:** `http://localhost:8000/docs`
**ReDoc:** `http://localhost:8000/redoc`
**OpenAPI JSON:** `http://localhost:8000/openapi.json`

---

## Base URL

```
http://localhost:8000/api/v1
```

- On-premise: `http://{vm-ip}:8000/api/v1`
- Cloud: `https://your-coolify-domain/api/v1`

---

## Two Operational Modes

### 1. Online Mode — Knowledgebase from Web Content
Update the knowledgebase using online URLs and cloud services.
- Scrape web pages via Crawl4AI (default) or the Jina Reader API — selectable per request
- Parse documents from any public URL — uses LlamaParse (cloud)
- All online endpoints live under `/api/v1/online/`
- **AT funding assistant** has a dedicated endpoint — `POST /api/v1/online/ingest/at` — that writes to a separate Qdrant instance (`QDRANT_URL_AT` + `QDRANT_PORT_AT` + `QDRANT_API_KEY_AT`) using a caller-supplied single collection. Dense embeddings come from a self-hosted TEI server (`TEI_EMBED_URL_AT`) at 1024 dim. See the endpoint docs below.
- Requires: `CRAWL4AI_URL`, `LLAMA_CLOUD_API_KEY`, `OPENAI_API_KEY`; optional `JINA_API_KEY` to enable the Jina backend; `QDRANT_URL_AT` / `QDRANT_PORT_AT` (optional — leave unset when the port is embedded in `QDRANT_URL_AT`) / `QDRANT_API_KEY_AT` / `TEI_EMBED_URL_AT` / `TEI_EMBED_API_KEY_AT` for the AT pipeline

### 2. Local Mode — Fully Offline Document Processing
Process documents entirely locally without any third-party APIs.
- Upload documents via `POST /api/v1/local/document-parse/upload` or read from SMB file shares
- Parse locally with PyMuPDF (PDF) + python-docx (DOCX) — lightweight, no GPU needed
- All local endpoints live under `/api/v1/local/`
- Requires: Only Qdrant + BGE-M3

---

## Authentication

### HMAC Authentication (global)

All endpoints except `/health` and `/ready` require HMAC-SHA256 authentication (when `DP_HMAC_SECRET` is set).

| Header | Description |
|--------|-------------|
| `X-Signature` | HMAC-SHA256 of `{timestamp}.{request_body}` |
| `X-Timestamp` | Unix epoch seconds (must be within ±5 min) |

Leave `DP_HMAC_SECRET` empty to disable HMAC authentication.

### API Key Authentication (online endpoints only)

All `/api/v1/online/*` endpoints require an `X-API-Key` header. Valid keys are configured via the `DP_ONLINE_API_KEYS` environment variable.

| Header | Description |
|--------|-------------|
| `X-API-Key` | API key for online endpoint access (configured via `DP_ONLINE_API_KEYS` env var) |

Local endpoints (`/api/v1/local/*`) do **not** require an API key — they are designed for trusted network access.

> **Note:** If both `DP_HMAC_SECRET` and `DP_ONLINE_API_KEYS` are set, online endpoints require **both** HMAC and API key headers.

---

## Response Envelope

Every API response is wrapped in a standard envelope:

**Success:**
```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "detail": null,
  "request_id": "ca60c30a-6289-4732-9b9f-028d207bb9a1"
}
```

**Error:**
```json
{
  "success": false,
  "data": null,
  "error": "PARSE_FAILED",
  "detail": "Human-readable error message",
  "request_id": "ca60c30a-6289-4732-9b9f-028d207bb9a1"
}
```

All responses include an `X-Request-ID` header. Send `X-Request-ID` in your request to trace it through.

---

# Shared Endpoints

These endpoints are not scoped to online or local mode.

## `GET /api/v1/health`

Liveness check. No authentication required.

**Response:**
```json
{
  "status": "ok",
  "uptime_seconds": 3421.5
}
```

---

## `GET /api/v1/ready`

Readiness check. Returns minimal response without auth, full response with HMAC auth.

**Minimal response (no auth):**
```json
{
  "ready": true,
  "uptime_seconds": 3421.5
}
```

**Full response (with HMAC auth headers):**
```json
{
  "ready": true,
  "services": {
    "qdrant": true,
    "bge_m3": true,
    "openai_embedder": true,
    "bge_gemma2": true,
    "parser": true,
    "crawl4ai": true,
    "ldap": false,
    "redis": true
  },
  "mode": "on-premise",
  "tenant_id": "wiener-neudorf",
  "worker_id": "wn-worker-01",
  "version": "1.0.0",
  "uptime_seconds": 3421.5
}
```

Core services required for `ready: true`: **qdrant**, **bge_m3**, **parser**, **crawl4ai**. The **openai_embedder** and **bge_gemma2** statuses are informational — at least one must be healthy for online ingest/search to work.

---

## `GET /metrics`

Prometheus-compatible metrics with `dp_` prefix. Returns `text/plain`.

---

## `POST /api/v1/classify`

Classify content into 9 categories and extract structured entities. Designed for German-language municipality documents.

**Categories:** `funding`, `event`, `policy`, `contact`, `form`, `announcement`, `minutes`, `report`, `general`

**Request:**
```json
{
  "content": "Das Förderprogramm für erneuerbare Energien gilt ab 01.04.2025. Antragsfrist bis 30.06.2025. Förderhöhe bis EUR 5.000. Kontakt: energie@wiener-neudorf.gv.at, Umweltamt.",
  "language": "de"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "classification": "funding",
    "confidence": 0.95,
    "sub_categories": ["renewable_energy"],
    "entities": {
      "dates": ["01.04.2025", "30.06.2025"],
      "deadlines": ["30.06.2025"],
      "amounts": ["EUR 5.000"],
      "contacts": ["energie@wiener-neudorf.gv.at"],
      "departments": ["Umweltamt"]
    },
    "summary": "Förderung für erneuerbare Energien, Antragsfrist bis 30. Juni 2025"
  },
  "request_id": "..."
}
```

**Response (empty content):**
```json
{
  "success": false,
  "error": "VALIDATION_EMPTY_CONTENT",
  "detail": "Content must not be empty",
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | — | Text to classify (from `/online/document-parse`, `/local/document-parse`, or `/online/scrape`) |
| `language` | string | No | `de` | ISO 639-1 language code |

### Extracted entities

| Entity | Examples |
|--------|----------|
| `dates` | `01.04.2025`, `2025-06-30` |
| `deadlines` | Dates near words like "Frist", "bis", "deadline" |
| `amounts` | `EUR 5.000`, `€ 10.000` |
| `contacts` | Email addresses |
| `departments` | `Umweltamt`, `Bauamt`, `Finanzabteilung` |

### Error codes
`VALIDATION_EMPTY_CONTENT`, `CLASSIFY_FAILED`, `CLASSIFY_LOW_CONFIDENCE`, `ENTITY_EXTRACTION_FAILED`

---

## `POST /api/v1/search`

Permission-aware semantic or hybrid search. **No search is ever unfiltered** — every request requires a user context.

**Search modes:**
- `semantic` (default) — dense-only cosine search via OpenAI `text-embedding-3-small` (1536-dim). Automatically falls back to BGE-Gemma2 via LiteLLM (`dense_bge_gemma2` vector) when OpenAI is unavailable.
- `hybrid` — dense (OpenAI or BGE-Gemma2 fallback) + sparse (BM25) combined with **Reciprocal Rank Fusion (RRF)**. Requires the collection to have been ingested with `search_mode: hybrid`.

**Permission model:**
- `citizen` → sees only `visibility: "public"` documents
- `employee` → sees `public` + `internal`, filtered by AD group membership

### Case 1: Semantic search (default)

**Request:**
```json
{
  "collection_name": "wiener-neudorf",
  "query": "Wann ist die nächste Förderung für Solaranlagen?",
  "user": {
    "type": "employee",
    "user_id": "maria@wiener-neudorf.gv.at",
    "groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Alle-Mitarbeiter"],
    "roles": ["member"],
    "department": "bauamt"
  },
  "filters": {
    "content_type": ["funding"]
  },
  "search_mode": "semantic",
  "top_k": 10,
  "score_threshold": 0.5
}
```

### Case 2: Hybrid search (dense + BM25 with RRF)

**Request:**
```json
{
  "collection_name": "wiener-neudorf",
  "query": "Förderung Photovoltaik Antragsfrist",
  "user": {
    "type": "citizen",
    "user_id": "anonymous"
  },
  "search_mode": "hybrid",
  "top_k": 10
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "chunk_id": "doc_abc123_chunk_0007",
        "source_id": "doc_abc123",
        "chunk_text": "Die Förderung für Solaranlagen beträgt bis zu EUR 5.000...",
        "score": 0.92,
        "source_path": "//server/bauamt/foerderungen/solar_2025.pdf",
        "content_type": ["funding", "renewable_energy"],
        "entities": {
          "amounts": ["EUR 5.000"],
          "deadlines": ["2025-06-30"]
        },
        "metadata": {
          "title": "Solarförderung 2025",
          "municipality_id": "wiener-neudorf",
          "department": ["Bauamt"],
          "source_type": "web"
        }
      }
    ],
    "total_results": 7,
    "query_embedding_ms": 15,
    "search_ms": 22,
    "search_mode": "semantic",
    "permission_filter_applied": {
      "visibility": ["public", "internal"],
      "must_match_groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Alle-Mitarbeiter"],
      "must_not_match_groups": []
    }
  },
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `collection_name` | string | Yes | — | Qdrant collection to search |
| `query` | string | Yes | — | Natural language search query |
| `user` | object | Yes | — | User identity (always required) |
| `user.type` | string | Yes | — | `citizen` or `employee` |
| `user.user_id` | string | Yes | — | Email, AD username, or `anonymous` |
| `user.groups` | array | No | `[]` | AD groups (required for employees) |
| `user.roles` | array | No | `[]` | Portal roles |
| `user.department` | string | No | — | Department for filtering |
| `filters` | object | No | — | Optional content filters |
| `filters.content_type` | array | No | — | Filter by content types (e.g. `["funding", "policy"]`) |
| `search_mode` | string | No | `"semantic"` | `"semantic"` (dense cosine only) or `"hybrid"` (dense + BM25 sparse with RRF) |
| `top_k` | int | No | 10 | Max results (1-100) |
| `score_threshold` | float | No | 0.5 | Min similarity score (0.0-1.0). Used for semantic mode only. |

### Error codes
`VALIDATION_USER_REQUIRED`, `VALIDATION_EMPTY_CONTENT`, `EMBEDDING_MODEL_NOT_LOADED`, `EMBEDDING_FAILED`, `QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_SEARCH_FAILED`

---

## `POST /api/v1/collections/init`

Create a Qdrant vector collection for a municipality tenant. If the collection already exists, returns `created: false` without error.

**Request (default config):**
```json
{
  "collection_name": "wiener-neudorf"
}
```

**Request (custom config):**
```json
{
  "collection_name": "wiener-neudorf",
  "vector_config": {
    "dense_dim": 1024,
    "sparse": true,
    "distance": "cosine"
  }
}
```

**Response (newly created):**
```json
{
  "success": true,
  "data": {
    "collection": "wiener-neudorf",
    "created": true,
    "dense_dim": 1024,
    "sparse_enabled": true
  },
  "request_id": "..."
}
```

**Response (already exists):**
```json
{
  "success": true,
  "data": {
    "collection": "wiener-neudorf",
    "created": false,
    "dense_dim": 1024,
    "sparse_enabled": true
  },
  "request_id": "..."
}
```

### Error codes
`QDRANT_CONNECTION_FAILED`

---

## `GET /api/v1/collections/stats`

Get statistics for a Qdrant collection.

**Request:**
```
GET /api/v1/collections/stats?collection_name=wiener-neudorf
```

**Response:**
```json
{
  "success": true,
  "data": {
    "collection": "wiener-neudorf",
    "total_vectors": 12450,
    "total_documents": 0,
    "disk_usage_mb": 245.5,
    "by_classification": {},
    "by_visibility": {}
  },
  "request_id": "..."
}
```

**Response (collection not found):**
```json
{
  "success": false,
  "error": "QDRANT_COLLECTION_NOT_FOUND",
  "detail": "Collection 'nonexistent' not found",
  "request_id": "..."
}
```

### Error codes
`QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_CONNECTION_FAILED`

---

# Online Endpoints

All online endpoints require the `X-API-Key` header (configured via `DP_ONLINE_API_KEYS` env var). These endpoints handle web content and cloud-based document processing.

## `GET /api/v1/online/available_collections`

List all available Qdrant collections with basic information including vector count, point count, segments, disk usage, and status.

**Request:**
```bash
curl -X GET "https://your-domain/api/v1/online/available_collections" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total": 2,
    "collections": [
      {
        "name": "wiener-neudorf",
        "vectors_count": 12450,
        "points_count": 12450,
        "segments_count": 4,
        "disk_usage_mb": 245.5,
        "status": "green"
      },
      {
        "name": "test-collection",
        "vectors_count": 500,
        "points_count": 500,
        "segments_count": 1,
        "disk_usage_mb": 12.3,
        "status": "green"
      }
    ]
  },
  "request_id": "..."
}
```

### Response fields

| Field | Type | Description |
|-------|------|-------------|
| `total` | int | Total number of collections |
| `collections[].name` | string | Collection name |
| `collections[].vectors_count` | int | Total vectors stored |
| `collections[].points_count` | int | Total points stored |
| `collections[].segments_count` | int | Number of segments |
| `collections[].disk_usage_mb` | float | Disk usage in MB |
| `collections[].status` | string | Collection status (`green`, `yellow`, `red`, `unknown`) |

### Error codes
`QDRANT_CONNECTION_FAILED`

---

## `POST /api/v1/online/scrape`

Scrape a single webpage using **Crawl4AI** (JavaScript rendering) or the **Jina Reader API**.
Backend is selectable per request via the `scraper` field. Results are cached in Redis.

**Request:**
```bash
curl -X POST "https://your-domain/api/v1/online/scrape" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "url": "https://www.wiener-neudorf.gv.at/foerderungen"
  }'
```

**Request body:**
```json
{
  "url": "https://www.wiener-neudorf.gv.at/foerderungen"
}
```

**Request body (use Jina Reader instead of Crawl4AI):**
```json
{
  "url": "https://www.wiener-neudorf.gv.at/foerderungen",
  "scraper": "jina"
}
```

### Request fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | Yes | — | Full URL to scrape (must start with `http://` or `https://`) |
| `scraper` | string | No | `crawl4ai` | Preferred backend: `crawl4ai` or `jina`. The non-selected backend (and raw httpx) remain as automatic fallbacks if the primary fails. |
| `markdown_type` | string | No | `fit` | `fit` (main content only), `raw` (full page), or `citations` (Crawl4AI only) |
| `exclude_tags` | string[] | No | `null` | CSS selectors / tag names to drop before extraction |
| `css_selector` | string | No | `null` | CSS selector to scope extraction to a specific element |
| `inner_img` | bool | No | `false` | Extract and OCR-parse images linked on the page |
| `inner_docs` | bool | No | `false` | Extract and parse documents (PDF, DOCX, …) linked on the page |
| `links_summary` | bool | No | `false` | Return deduped `urls` list extracted from the **raw** page HTML (not the fit-filtered version, so nav/footer links aren't dropped). `documents` / `images` sub-lists are populated only when `inner_docs` / `inner_img` are also true. Triggers one extra raw-HTML fetch. |

**Response (success):**
```json
{
  "success": true,
  "data": {
    "url": "https://www.wiener-neudorf.gv.at/foerderungen",
    "title": "Förderungen - Gemeinde Wiener Neudorf",
    "content": "# Förderungen\n\nDie Gemeinde Wiener Neudorf bietet folgende Förderungen an...",
    "content_length": 5200,
    "language": "de",
    "links_found": 45,
    "last_modified": null
  },
  "request_id": "..."
}
```

**Response with `links_summary: true`** (only `urls` populated):
```json
{
  "success": true,
  "data": {
    "url": "https://www.wiener-neudorf.gv.at/foerderungen",
    "title": "Förderungen - Gemeinde Wiener Neudorf",
    "content": "# Förderungen\n…",
    "content_length": 5200,
    "language": "de",
    "links_found": 45,
    "last_modified": null,
    "links_summary": {
      "urls": [
        "https://www.wiener-neudorf.gv.at/",
        "https://www.wiener-neudorf.gv.at/kontakt",
        "https://www.wiener-neudorf.gv.at/service/"
      ],
      "documents": [],
      "images": []
    }
  },
  "request_id": "..."
}
```

**Response with `links_summary: true` + `inner_docs: true` + `inner_img: true`** (all three lists populated):
```json
{
  "links_summary": {
    "urls": ["https://www.wiener-neudorf.gv.at/", "…"],
    "documents": ["https://www.wiener-neudorf.gv.at/downloads/antrag.pdf"],
    "images": ["https://www.wiener-neudorf.gv.at/images/logo.png"]
  }
}
```

**Response (invalid URL):**
```json
{
  "success": false,
  "error": "VALIDATION_URL_INVALID",
  "detail": "URL must start with http:// or https://",
  "request_id": "..."
}
```

**Response (empty page):**
```json
{
  "success": false,
  "error": "SCRAPE_EMPTY",
  "detail": "Page returned no extractable content",
  "request_id": "..."
}
```

**Response (timeout):**
```json
{
  "success": false,
  "error": "SCRAPE_TIMEOUT",
  "detail": "Request timed out after 30s",
  "request_id": "..."
}
```

### Error codes
`VALIDATION_URL_INVALID`, `SCRAPE_FAILED`, `SCRAPE_BLOCKED`, `SCRAPE_TIMEOUT`, `SCRAPE_EMPTY`, `SCRAPE_ROBOTS_BLOCKED`

---

## `POST /api/v1/online/crawl`

Discover URLs from a website. Returns URLs only — does not scrape content.

### Case 1: Sitemap discovery

**Request:**
```bash
curl -X POST "https://your-domain/api/v1/online/crawl" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "url": "https://www.wiener-neudorf.gv.at/sitemap.xml",
    "method": "sitemap",
    "max_urls": 500
  }'
```

### Case 2: BFS crawl discovery

**Request:**
```json
{
  "url": "https://www.wiener-neudorf.gv.at",
  "method": "crawl",
  "max_depth": 3,
  "max_urls": 100
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "base_url": "https://www.wiener-neudorf.gv.at",
    "method_used": "sitemap",
    "total_urls": 234,
    "urls": [
      {
        "url": "https://www.wiener-neudorf.gv.at/gemeindeamt/kontakt/",
        "type": "page",
        "last_modified": null
      },
      {
        "url": "https://www.wiener-neudorf.gv.at/files/foerderung.pdf",
        "type": "document",
        "last_modified": null
      }
    ]
  },
  "request_id": "..."
}
```

**Response (no sitemap found):**
```json
{
  "success": false,
  "error": "CRAWL_SITEMAP_NOT_FOUND",
  "detail": "No URLs found in sitemap",
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | Yes | — | Base URL or sitemap URL |
| `method` | string | Yes | — | `sitemap` or `crawl` |
| `max_depth` | int | No | 3 | Max link-following depth (1-5) |
| `max_urls` | int | No | 500 | Max URLs to return (1-5000) |
| `scraper` | string | No | `crawl4ai` | Preferred backend during BFS discovery (`crawl4ai` or `jina`). Ignored when `method="sitemap"`. |

### Error codes
`VALIDATION_URL_INVALID`, `CRAWL_SITEMAP_NOT_FOUND`

---

## `POST /api/v1/online/document-parse`

Parse a document from a public URL. Uses LlamaParse (cloud) when `LLAMA_CLOUD_API_KEY` is set, otherwise falls back to local parsers.

**Supported formats:** PDF, DOCX, DOC, PPTX, ODT, XLSX, XLS, TXT, CSV, HTML, RTF

**Request:**
```bash
curl -X POST "https://your-domain/api/v1/online/document-parse" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "url": "https://pdfobject.com/pdf/sample.pdf"
  }'
```

**Request body:**
```json
{
  "url": "https://pdfobject.com/pdf/sample.pdf"
}
```

`mime_type` is optional — auto-detected from the URL.

**Request with explicit MIME type:**
```json
{
  "url": "https://example.com/download?id=123",
  "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "url": "https://pdfobject.com/pdf/sample.pdf",
    "content": "This is a simple PDF file. Fun fun fun...",
    "pages": 2,
    "language": "en",
    "extracted_tables": 0,
    "content_length": 1234
  },
  "request_id": "5786ede5-7631-46f2-8e6b-0c48f8564274"
}
```

### Request fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | Public URL of the document to parse |
| `mime_type` | string | No | MIME type. Auto-detected from URL if omitted. |

### Error codes
`PARSE_FAILED`, `PARSE_ENCRYPTED`, `PARSE_CORRUPTED`, `PARSE_EMPTY`, `PARSE_TIMEOUT`, `PARSE_UNSUPPORTED_FORMAT`

---

## `POST /api/v1/online/document-parse/upload`

Upload a raw document file directly for parsing via `multipart/form-data`.

Send the **original binary file** in the `file` form field — do **not** base64-encode it. The server auto-detects the file type from the filename extension and content type.

**Supported formats:** PDF, DOCX, DOC, PPTX, ODT, XLSX, XLS, TXT, CSV, HTML, RTF

**Request (cURL):**
```bash
curl -X POST "https://your-domain/api/v1/online/document-parse/upload" \
  -H "X-API-Key: your-api-key" \
  -F "file=@/path/to/document.pdf"
```

**Request (Python — requests):**
```python
import requests

with open("report.pdf", "rb") as f:
    response = requests.post(
        "https://your-domain/api/v1/online/document-parse/upload",
        headers={"X-API-Key": "your-api-key"},
        files={"file": ("report.pdf", f, "application/pdf")},
    )
print(response.json())
```

**Request (JavaScript — fetch):**
```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);  // raw File object from <input type="file">

const response = await fetch("/api/v1/online/document-parse/upload", {
  method: "POST",
  headers: { "X-API-Key": "your-api-key" },
  body: formData,
});
```

**Request (Swagger UI):** Click "Try it out", choose a file, and execute.

**Response:**
```json
{
  "success": true,
  "data": {
    "url": "document.pdf",
    "content": "Extracted text content from the uploaded PDF...",
    "pages": 5,
    "language": "de",
    "extracted_tables": 1,
    "content_length": 8500
  },
  "request_id": "..."
}
```

### Error codes
`PARSE_FAILED`, `PARSE_ENCRYPTED`, `PARSE_CORRUPTED`, `PARSE_EMPTY`, `PARSE_TIMEOUT`, `PARSE_UNSUPPORTED_FORMAT`

---

## `POST /api/v1/online/ingest`

The RAG pipeline endpoint for web content. Takes parsed text and runs: **chunk -> classify -> embed (OpenAI + BGE-Gemma2 via LiteLLM) -> store (Qdrant)**.

Uses `url` instead of `file_path` to identify the source. The `url` is automatically stored as `source_url` in every Qdrant point's metadata.

Every point stores **multi-vector** embeddings: `dense_openai` (primary, 1536-dim) and `dense_bge_gemma2` (fallback, configurable dim via `BGE_GEMMA2_DENSE_DIM`). If one embedder is unavailable during ingest, the point is still stored with the other's vector.

The collection is **auto-created** if it does not exist, using the specified `vector_config` settings.

**Content-type gating:** When `assistant_type` is `"funding"`, the content is pre-classified before ingestion. If the detected content type does not include `funding`, the request is rejected with `CONTENT_TYPE_MISMATCH` and nothing is stored. This prevents non-funding content from polluting a funding-specific knowledge base.

**Funding metadata extraction:** When `assistant_type` is `"funding"` and the content passes the content-type gate, an additional OpenAI call extracts structured funding metadata (titel, region, zielgruppe, förderart, status, förderhöhe, etc.). This metadata is stored in every Qdrant point under `metadata.funding_metadata` for rich filtering and display during search.

**Vector modes** (via `vector_config.search_mode`):
- `semantic` (default) — stores `dense_openai` + `dense_bge_gemma2` cosine vectors.
- `hybrid` — stores `dense_openai` + `dense_bge_gemma2` + `sparse` (BM25) vectors. Enables combined semantic + lexical search for higher recall.

**Request (semantic mode — default):**
```bash
curl -X POST "https://your-domain/api/v1/online/ingest" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "collection_name": "wiener-neudorf",
    "source_id": "web_foerderungen_001",
    "url": "https://www.wiener-neudorf.gv.at/foerderungen",
    "content": "Die Gemeinde Wiener Neudorf bietet folgende Förderungen an...",
    "language": "de",
    "metadata": {
      "assistant_id": "asst_wiener_neudorf_01",
      "title": "Förderungen - Gemeinde Wiener Neudorf",
      "source_type": "web",
      "municipality_id": "wiener-neudorf",
      "department": ["Bürgerservice", "Förderungen"]
    }
  }'
```

**Request (hybrid mode with custom vector size):**
```bash
curl -X POST "https://your-domain/api/v1/online/ingest" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "collection_name": "wiener-neudorf",
    "source_id": "web_foerderungen_001",
    "url": "https://www.wiener-neudorf.gv.at/foerderungen",
    "content": "Die Gemeinde Wiener Neudorf bietet folgende Förderungen an...",
    "language": "de",
    "metadata": {
      "assistant_id": "asst_wiener_neudorf_01",
      "title": "Förderungen - Gemeinde Wiener Neudorf",
      "source_type": "web",
      "municipality_id": "wiener-neudorf",
      "department": ["Bürgerservice"]
    },
    "vector_config": {
      "vector_size": 1536,
      "search_mode": "hybrid"
    }
  }'
```

**Response (success):**
```json
{
  "success": true,
  "data": {
    "source_id": "web_foerderungen_001",
    "chunks_created": 4,
    "vectors_stored": 4,
    "collection": "wiener-neudorf",
    "content_type": ["funding", "renewable_energy"],
    "entities_extracted": {
      "dates": 2,
      "contacts": 1,
      "amounts": 1
    },
    "embedding_time_ms": 850,
    "total_time_ms": 2100
  },
  "request_id": "..."
}
```

**Response (content-type mismatch — `assistant_type: "funding"` but content is not funding):**
```json
{
  "success": false,
  "data": null,
  "error": "CONTENT_TYPE_MISMATCH",
  "detail": "Content not ingested: assistant_type is 'funding' but detected content type is ['event', 'community']. Only funding content is accepted for this assistant type.",
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `collection_name` | string | Required | — | Qdrant collection to store in (auto-created if missing) |
| `source_id` | string | Required | — | Unique document ID (for updates/deletes) |
| `url` | string | Required | — | Source URL — stored as `source_url` in Qdrant point metadata |
| `content` | string | Required | — | Parsed text from `/online/scrape` or `/online/document-parse` |
| `language` | string | Optional | auto-detect | ISO 639-1 language code |
| `assistant_type` | string | Optional | `null` | Type of assistant processing this content (e.g. `municipal`, `internal`, `public`). Stored in Qdrant metadata for search filtering. When set to `funding`, the content is pre-classified and **rejected** with `CONTENT_TYPE_MISMATCH` if the detected content type is not funding. |
| `country` | string | Required when `assistant_type` is `funding`, otherwise Optional | `null` | ISO 3166-1 alpha-2 country code (e.g. `AT`, `DE`, `RO`). Used by the funding extractor to constrain `state_or_province` to the official list for that country, preventing hallucinated region names. Supported: `AT`, `DE`, `CH`, `RO`, `IT`, `FR`, `HU`, `CZ`, `SK`, `SI`, `HR`. |
| `metadata` | object | Required | — | Document metadata (see Online Metadata object below) |
| `chunking` | object | Optional | defaults | Chunking configuration (see Chunking config below) |
| `vector_config` | object | Optional | defaults | Vector storage settings (see Vector config below) |

### Online Metadata object

At least one of `assistant_id` or `municipality_id` must be provided. If neither is set, the request will fail with a validation error.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `assistant_id` | string | Optional | `null` | Identifier of the assistant that owns this content. At least one of `assistant_id` or `municipality_id` must be provided. |
| `title` | string | Optional | `null` | Document/page title (shown in search results) |
| `uploaded_by` | string | Optional | `null` | User or service that triggered ingestion |
| `source_type` | string | Optional | `"web"` | Origin type (typically `web` for online content) |
| `mime_type` | string | Optional | `null` | Original content MIME type |
| `municipality_id` | string | Optional | `null` | Municipality/tenant identifier. At least one of `assistant_id` or `municipality_id` must be provided. |
| `department` | array of strings | Optional | `[]` | Departments within the organization |

### Vector config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `vector_size` | int | `1536` | Dimensionality of the OpenAI dense vector (`dense_openai`, 64–4096). The BGE-Gemma2 fallback dimension is configured server-side via `BGE_GEMMA2_DENSE_DIM`. |
| `search_mode` | string | `"semantic"` | `"semantic"` — `dense_openai` + `dense_bge_gemma2` vectors. `"hybrid"` — `dense_openai` + `dense_bge_gemma2` + `sparse` (BM25) vectors. |

### Qdrant point payload structure

The payload has top-level fields for tenant/agent isolation (`municipality_id`, `assistant_id`, `department`), plus `content` and nested `metadata`.

When `assistant_type` is `"funding"`, extracted funding fields are merged flat into `metadata` (not nested). If the request body and the extracted metadata share a field (e.g. `title`), the request body value takes priority.

**Standard payload (non-funding):**
```json
{
  "id": "uuid",
  "vector": {
    "dense_openai": [1536-dim float array],
    "dense_bge_gemma2": [3584-dim float array],
    "sparse": {"indices": [...], "values": [...]}
  },
  "payload": {
    "municipality_id": "wiener-neudorf",
    "assistant_id": "asst_wiener_neudorf_01",
    "department": ["Bürgerservice", "Förderungen"],
    "content": "The chunk text content...",
    "metadata": {
      "chunk_id": "web_foerderungen_001_chunk_0000",
      "source_id": "web_foerderungen_001",
      "chunk_index": 0,
      "source_url": "https://www.wiener-neudorf.gv.at/foerderungen",
      "source_path": "https://www.wiener-neudorf.gv.at/foerderungen",
      "content_type": ["funding", "renewable_energy"],
      "language": "de",
      "title": "Förderungen - Gemeinde Wiener Neudorf",
      "source_type": "web",
      "mime_type": "text/html",
      "uploaded_by": "scraper"
    }
  }
}
```

**Funding payload (`assistant_type: "funding"`) — includes extracted fields in metadata:**
```json
{
  "id": "uuid",
  "vector": { "dense_openai": [...], "dense_bge_gemma2": [...] },
  "payload": {
    "municipality_id": "wiener-neudorf",
    "assistant_id": "asst_wiener_neudorf_01",
    "department": ["Bürgerservice", "Förderungen"],
    "content": "The chunk text content...",
    "metadata": {
      "chunk_id": "web_foerderungen_001_chunk_0000",
      "source_id": "web_foerderungen_001",
      "chunk_index": 0,
      "source_url": "https://www.wiener-neudorf.gv.at/foerderungen",
      "source_path": "https://www.wiener-neudorf.gv.at/foerderungen",
      "content_type": ["funding", "renewable_energy"],
      "language": "de",
      "title": "Förderungen - Gemeinde Wiener Neudorf",
      "source_type": "web",
      "mime_type": "text/html",
      "uploaded_by": "scraper",
      "country_code": "AT",
      "state_or_province": ["carinthia"],
      "city": ["villach"],
      "target_group": ["Vereine"],
      "funding_type": "Direkte Förderungen",
      "status": "active",
      "funding_amount": "",
      "thematic_focus": ["Sport"],
      "eligibility_criteria": "Schriftliche Antragstellung, Vereinssitz in Villach",
      "legal_basis": "Bereichssubventionsordnung Sport, Basis-Subventionsordnung",
      "funding_provider": ["Stadt Villach"],
      "reference_number": 1052992,
      "start_date": "01.01.2020",
      "end_date": "unlimited",
      "scraped_at": "2025-06-25"
    }
  }
}
```

| Field | Location | Description |
|-------|----------|-------------|
| `municipality_id` | `payload` (top-level) | Municipality/tenant boundary |
| `assistant_id` | `payload` (top-level) | Assistant/agent isolation |
| `department` | `payload` (top-level) | Departments (array of strings) |
| `content` | `payload` (top-level) | The text content of this chunk |
| `chunk_id` | `payload.metadata` | `{source_id}_chunk_{index}` |
| `source_id` | `payload.metadata` | Document identifier |
| `chunk_index` | `payload.metadata` | Position of chunk within document |
| `source_url` | `payload.metadata` | Source URL from request |
| `source_path` | `payload.metadata` | Original source path/URL |
| `content_type` | `payload.metadata` | Auto-detected content categories (array of strings) |
| `language` | `payload.metadata` | ISO 639-1 language code |
| `title` | `payload.metadata` | Document title |
| `source_type` | `payload.metadata` | Origin type |
| `mime_type` | `payload.metadata` | MIME type |
| `uploaded_by` | `payload.metadata` | Uploader identity |

### Funding metadata fields (assistant_type: "funding" only)

Extracted automatically via OpenAI when `assistant_type` is `"funding"`. Merged flat into `payload.metadata` alongside the standard fields. If a field name conflicts with the request body (e.g. `title`), the request body value wins.

When `country` is provided in the request body (e.g. `"AT"`), the extractor constrains `state_or_province` to the official administrative divisions for that country. If the LLM returns a value not in the known list, it is reset to empty string. This prevents hallucinated region names and ensures clean filtering.

**Supported countries:** `AT`, `DE`, `CH`, `RO`, `IT`, `FR`, `HU`, `CZ`, `SK`, `SI`, `HR`. Other country codes are accepted — the extractor still works but without province validation.

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Title of the funding program |
| `country_code` | string | ISO 3166-1 alpha-2 (e.g. `AT`, `DE`). From request `country` or inferred from content. |
| `state_or_province` | array of strings | Official first-level admin divisions in **english lowercase** (e.g. `["carinthia"]`, `["bavaria", "tyrol"]`). Each value validated against known list when `country` is provided — invalid entries are dropped. Empty list if unknown. |
| `city` | array of strings | City names in **english lowercase** (e.g. `["villach"]`, `["villach", "klagenfurt"]`). Empty list if unknown. |
| `target_group` | array of strings | Target groups (e.g. Vereine, Privatpersonen) |
| `funding_type` | string | Funding type (e.g. Direkte Förderungen, Zuschuss) |
| `status` | string | `active`, `inactive`, `expiring`, or `unknown` |
| `funding_amount` | string | Funding amount or range (empty if unknown) |
| `thematic_focus` | array of strings | Thematic focus areas (e.g. Sport, Umwelt) |
| `eligibility_criteria` | string | Eligibility criteria and requirements |
| `legal_basis` | string | Legal basis or regulation |
| `funding_provider` | array of strings | Funding provider organizations |
| `reference_number` | string/null | Reference number or ID |
| `start_date` | string | Start date (DD.MM.YYYY) or empty |
| `end_date` | string | End date (DD.MM.YYYY), `unlimited`, or empty |
| `scraped_at` | string | Date of extraction (YYYY-MM-DD) |

### Error codes
`VALIDATION_EMPTY_CONTENT`, `CONTENT_TYPE_MISMATCH`, `EMBEDDING_MODEL_NOT_LOADED`, `EMBEDDING_FAILED`, `EMBEDDING_OOM`, `QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_UPSERT_FAILED`, `QDRANT_DISK_FULL`, `CLASSIFY_FAILED`

---

## `POST /api/v1/online/ingest/at`

Dedicated ingest for the **Austrian funding assistant**. Runs the pipeline (chunk → contextual enrichment → embed → store) against a single caller-supplied collection on a separate Qdrant instance.

The country (`AT`) and assistant type (`funding`) are **implicit** — don't send them in the body.

### How the AT endpoint differs from `/online/ingest`

| Aspect | `/online/ingest` | `/online/ingest/at` |
|---|---|---|
| Qdrant target | `QDRANT_URL` (host:port combined) | `QDRANT_URL_AT` + `QDRANT_PORT_AT` + `QDRANT_API_KEY_AT` (URL and port split, matching the upstream qdrant-client pattern; `QDRANT_PORT_AT` has no default — leave it unset when the port is already embedded in `QDRANT_URL_AT`, including the implicit 443 for `https://` URLs). Falls back to `QDRANT_URL` / `QDRANT_API_KEY` when `QDRANT_URL_AT` is unset. |
| Embedding model | OpenAI (`text-embedding-3-small`, 1536 dim) | Self-hosted TEI at `TEI_EMBED_URL_AT` (OpenAI-compatible `/v1/embeddings`, bearer `TEI_EMBED_API_KEY_AT`), **1024 dim** |
| Collection schema | Named multi-vector (`dense_openai`, `dense_bge_gemma2`, optional `sparse`) | Single unnamed **1024-dim** cosine vector (legacy schema) |
| Collection(s) per request | One (from `collection_name`) | One (from `collection_name`) |
| Point ID | `uuid5` string | Deterministic 64-bit integer |
| `country` / `assistant_type` / `vector_config` | Required | **Not accepted** — implicit |

### Funding-extractor metadata

The OpenAI-based funding extractor runs unconditionally on every request and its output is merged into `metadata.*` on every stored point:

- `title`, `program_name` (falls back to `title` when the extractor doesn't find a distinct program name)
- `processing_office` (list) — a funding doc may route through more than one office
- `contract_email` (list), `contract_phone` (list) — a funding doc may list multiple contacts
- `state_or_province` (list, english lowercase), `city` (list)
- `target_group`, `funding_type`, `status`, `funding_amount`
- `thematic_focus`, `eligibility_criteria`, `legal_basis`, `funding_provider`
- `reference_number`, `start_date`, `end_date`, `scraped_at`, `country_code`

Request-body fields take precedence — values present in `body.metadata` overwrite the extractor's output on the point.

### `state_or_province`

`body.state_or_province` overrides the extractor's value for the stored `metadata.state_or_province` only. There is no collection routing — the target collection always comes from `body.collection_name`. Normalized to lowercase + deduped before storage.

### Idempotency

Before upserting, the endpoint deletes prior points for the same `source_id` via the indexed `metadata.source_id` filter. This fully clears the document's existing chunks before writing the fresh ones — re-ingesting the same `source_id` replaces the stored content, even if the chunk count changes between ingests. Point IDs are deterministic over `(source_id, chunk_index)`.

### Request

```bash
curl -X POST "https://your-domain/api/v1/online/ingest/at" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "source_id": "web_salzburg_sport_001",
    "collection_name": "foerder_at",
    "url": "https://www.salzburg.gv.at/sport-foerderung",
    "content": "Sportförderung Salzburg ...",
    "content_type": ["funding", "sport"],
    "language": "de",
    "metadata": {
      "assistant_id": "asst_foerder_at_01",
      "municipality_id": "land-salzburg",
      "title": "Sportförderung Salzburg",
      "source_type": "web"
    }
  }'
```

### Response

```json
{
  "success": true,
  "data": {
    "source_id": "web_salzburg_sport_001",
    "chunks_created": 12,
    "vectors_stored": 12,
    "collection_name": "foerder_at",
    "content_type": ["funding", "sport"],
    "embedding_time_ms": 480,
    "total_time_ms": 1320
  },
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `source_id` | string | Yes | — | Unique document ID. Keys the `metadata.source_url` delete-before-upsert. |
| `collection_name` | string | Yes | — | Target collection on the AT Qdrant instance. **Auto-created** with the AT legacy schema (single unnamed 1024-dim cosine vector + keyword indexes on `metadata.source_id` / `metadata.source_url`) on first use. If the collection already exists with a different vector dim, the ingest fails with `QDRANT_COLLECTION_NOT_FOUND`. |
| `url` | string | Yes | — | Source URL (stored as `metadata.source_url`). |
| `source_name` | string | No | `null` | Human-readable label for the source (e.g. `"Land Salzburg — Sportförderung"`). Stored verbatim as `metadata.source_name` when supplied; the field is omitted from stored metadata when null/empty. |
| `content` | string | Yes | — | Parsed/scraped text. Must not be whitespace-only. |
| `content_type` | string[] | Yes | — | Categories from upstream `/scrape` or `/document-parse`. |
| `language` | string | No | `"de"` | ISO 639-1 code. |
| `state_or_province` | string[] | No | `null` | Override for stored `metadata.state_or_province` (english lowercase). |
| `entities` | object | No | `null` | Structured entities from upstream scrape/parse. |
| `metadata` | object | Yes | — | Same shape as `/online/ingest` metadata. At least one of `assistant_id` / `municipality_id` required. |
| `chunking` | object | No | `null` | Override chunking strategy / max size / overlap. When omitted, the endpoint uses AT-specific defaults tuned for German funding pages: `max_chunk_size=1000`, `overlap=150`, `strategy="contextual"`. |

No `country`, `assistant_type`, or `vector_config` fields — they're fixed by the endpoint.

### Response fields

| Field | Type | Description |
|---|---|---|
| `source_id` | string | Echoed from the request. |
| `chunks_created` | int | Chunks produced from the content. |
| `vectors_stored` | int | Vectors upserted into the target collection. |
| `collection_name` | string | Echoed from the request. |
| `content_type` | string[] | Echoed from the request. |
| `embedding_time_ms` | int | TEI embedding call duration. |
| `total_time_ms` | int | End-to-end pipeline time. |

### Error codes
`VALIDATION_EMPTY_CONTENT`, `EMBEDDING_MODEL_NOT_LOADED`, `EMBEDDING_FAILED`, `EMBEDDING_OOM`, `QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_UPSERT_FAILED`, `QDRANT_DISK_FULL`

---

## `DELETE /api/v1/online/vectors/{source_id}`

Remove all vector points associated with a `source_id` from the specified Qdrant collection.

**Request:**
```bash
curl -X DELETE "https://your-domain/api/v1/online/vectors/web_foerderungen_001?collection_name=wiener-neudorf" \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "source_id": "web_foerderungen_001",
    "vectors_deleted": 4
  },
  "request_id": "..."
}
```

### Error codes
`QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`

---

## `POST /api/v1/online/vectors/delete-by-filter`

Delete vectors matching metadata filters. All filters are combined with **AND** logic — only points matching every condition are deleted.

**Filterable metadata fields:** `source_id`, `source_url`, `source_type`, `content_type`, `assistant_id`, `municipality_id`, `department`, `language`, `uploaded_by`, `mime_type`, `title`

### Case 1: Delete all vectors for an assistant

**Request:**
```bash
curl -X POST "https://your-domain/api/v1/online/vectors/delete-by-filter" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "collection_name": "wiener-neudorf",
    "filters": [
      {"key": "assistant_id", "value": "asst_wiener_neudorf_01"}
    ]
  }'
```

### Case 2: Delete by content type and organization

**Request:**
```json
{
  "collection_name": "wiener-neudorf",
  "filters": [
    {"key": "content_type", "value": "funding"},
    {"key": "municipality_id", "value": "wiener-neudorf"}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "vectors_deleted": 12,
    "filters_applied": [
      {"key": "content_type", "value": "funding"},
      {"key": "municipality_id", "value": "wiener-neudorf"}
    ]
  },
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `collection_name` | string | Yes | Qdrant collection to delete from |
| `filters` | array | Yes | Metadata conditions (AND logic, min 1) |
| `filters[].key` | string | Yes | Metadata field name |
| `filters[].value` | string | Yes | Exact value to match |

### Error codes
`QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`

---

## `POST /api/v1/online/vectors/sparse-encode`

Encode arbitrary text into a Qdrant-compatible BM25 sparse vector using the same encoder that `POST /online/ingest` runs in `hybrid` mode (and that hybrid search uses for query encoding). Useful when a caller needs to reproduce the exact `sparse` vector that ingest would have stored, without going through the full ingest pipeline.

**Tokenization:** lowercased, split on non-alphanumeric, German + English stopwords removed, single-character tokens dropped. Each surviving token is hashed (MD5 mod 2^31-1) into the sparse index space; the value is the raw term frequency. Qdrant's IDF modifier on the collection handles inverse-document-frequency weighting at query time.

**Request:**
```bash
curl -X POST "https://your-domain/api/v1/online/vectors/sparse-encode" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "content": "Förderungen der Gemeinde Wiener Neudorf für Photovoltaik."
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "indices": [1115783198, 1136366200, 1236662434, 1585432512, 1740055052, 1864074548],
    "values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "term_count": 6
  },
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | Text to encode (must be non-empty after trim) |

### Error codes
`VALIDATION_EMPTY_CONTENT`

---

# Local Endpoints

Local endpoints do **not** require an `X-API-Key` header. They are designed for trusted network environments (on-premise, internal network).

## `POST /api/v1/local/document-parse`

Parse a document from an SMB file share or Cloudflare R2 bucket. Uses local parsers: PyMuPDF for PDF, python-docx for DOCX, SpreadsheetParser for XLSX/XLS, TextParser for TXT/CSV/HTML/RTF.

**Supported formats:** PDF, DOCX, DOC, PPTX, ODT, XLSX, XLS, TXT, CSV, HTML, RTF

### Case 1: Parse from SMB file share

**Request:**
```bash
curl -X POST "https://your-domain/api/v1/local/document-parse" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "//server/bauamt/dokumente/antrag_001.pdf",
    "source": "smb",
    "mime_type": "application/pdf"
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "file_path": "//server/bauamt/dokumente/antrag_001.pdf",
    "content": "Bauantrag Nr. 2024-001\nAntragsteller: Max Mustermann...",
    "pages": 12,
    "language": "de",
    "extracted_tables": 2,
    "content_length": 15420
  },
  "request_id": "..."
}
```

### Case 2: Parse from Cloudflare R2

**Request:**
```json
{
  "file_path": "tenant/wiener-neudorf/uploads/report.docx",
  "source": "r2",
  "r2_presigned_url": "https://r2.example.com/presigned/report.docx?token=abc123",
  "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}
```

### Parse error examples

**Encrypted PDF:**
```json
{
  "success": false,
  "data": null,
  "error": "PARSE_ENCRYPTED",
  "detail": "Parser error: encrypted PDF requires password",
  "request_id": "..."
}
```

**Empty document:**
```json
{
  "success": false,
  "data": null,
  "error": "PARSE_EMPTY",
  "detail": "Document contained no extractable text",
  "request_id": "..."
}
```

**Unsupported format:**
```json
{
  "success": false,
  "data": null,
  "error": "PARSE_UNSUPPORTED_FORMAT",
  "detail": "Unsupported document type: unknown",
  "request_id": "..."
}
```

**R2 missing presigned URL:**
```json
{
  "success": false,
  "data": null,
  "error": "R2_FILE_NOT_FOUND",
  "detail": "r2_presigned_url is required when source is r2",
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file_path` | string | Yes | SMB path or R2 object key |
| `source` | string | Yes | `smb` or `r2` |
| `mime_type` | string | No | MIME type (recommended for SMB/R2) |
| `r2_presigned_url` | string | Conditional | Required when `source` is `r2` |

### Error codes
`PARSE_FAILED`, `PARSE_ENCRYPTED`, `PARSE_CORRUPTED`, `PARSE_EMPTY`, `PARSE_TIMEOUT`, `PARSE_UNSUPPORTED_FORMAT`, `R2_FILE_NOT_FOUND`

---

## `POST /api/v1/local/document-parse/upload`

Upload a raw document file directly for parsing via `multipart/form-data`.

Send the **original binary file** in the `file` form field — do **not** base64-encode it. The server auto-detects the file type from the filename extension and content type.

**Supported formats:** PDF, DOCX, DOC, PPTX, ODT, XLSX, XLS, TXT, CSV, HTML, RTF

**Request (cURL):**
```bash
curl -X POST "https://your-domain/api/v1/local/document-parse/upload" \
  -F "file=@/path/to/document.pdf"
```

**Request (Python — requests):**
```python
import requests

with open("report.pdf", "rb") as f:
    response = requests.post(
        "https://your-domain/api/v1/local/document-parse/upload",
        files={"file": ("report.pdf", f, "application/pdf")},
    )
print(response.json())
```

**Request (Swagger UI):** Click "Try it out", choose a file, and execute.

**Response:**
```json
{
  "success": true,
  "data": {
    "file_path": "document.pdf",
    "content": "Extracted text content from the uploaded PDF...",
    "pages": 5,
    "language": "de",
    "extracted_tables": 1,
    "content_length": 8500
  },
  "request_id": "..."
}
```

---

## `POST /api/v1/local/discover`

Scan SMB file shares or R2 buckets for new/changed documents. First step in every ingestion pipeline — does NOT parse or embed. Returns NTFS ACLs, SHA-256 hashes, and change status.

### Case 1: SMB file share scan

**Request:**
```bash
curl -X POST "https://your-domain/api/v1/local/discover" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "smb",
    "paths": ["//server/abteilung/dokumente", "//server/bauamt"],
    "since_hash_map": {
      "//server/abteilung/dokumente/antrag.pdf": "sha256:abc123def456..."
    }
  }'
```

### Case 2: R2 bucket scan

**Request:**
```json
{
  "source": "r2",
  "paths": ["tenant/wiener-neudorf/uploads/"],
  "since_hash_map": {}
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_files": 523,
    "new_files": 12,
    "changed_files": 3,
    "unchanged_files": 508,
    "files": [
      {
        "path": "//server/bauamt/antrag_001.pdf",
        "file_hash": "sha256:abc123...",
        "size_bytes": 245000,
        "mime_type": "application/pdf",
        "last_modified": "2025-03-01T10:30:00Z",
        "status": "new",
        "acl": {
          "source": "ntfs",
          "allow_groups": ["DOMAIN\\Bauamt-Mitarbeiter"],
          "deny_groups": ["DOMAIN\\Praktikanten"],
          "allow_users": [],
          "inherited": true
        }
      }
    ]
  },
  "request_id": "..."
}
```

**Response (path not found):**
```json
{
  "success": false,
  "error": "SMB_PATH_NOT_FOUND",
  "detail": "Share path //server/invalid not found",
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `source` | string | Yes | `smb`, `r2`, or `url` |
| `paths` | array | Yes | SMB paths, R2 prefixes, or URLs to scan |
| `since_hash_map` | object | No | `{file_path: last_known_hash}` — matching hashes are skipped |

### Error codes
`SMB_CONNECTION_FAILED`, `SMB_AUTH_FAILED`, `SMB_PATH_NOT_FOUND`, `R2_CONNECTION_FAILED`, `R2_FILE_NOT_FOUND`, `LDAP_CONNECTION_FAILED`, `VALIDATION_PATH_OUTSIDE_ROOTS`

---

## `POST /api/v1/local/ingest`

The core RAG pipeline endpoint for local documents. Takes parsed text + ACL and runs: **chunk -> classify -> embed (BGE-M3) -> store (Qdrant)**.

- Multi-tenant: specify `collection_name`
- Idempotent: re-ingesting the same `source_id` replaces old vectors
- Every document MUST have an ACL with `visibility` set

**Request:**
```bash
curl -X POST "https://your-domain/api/v1/local/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "wiener-neudorf",
    "source_id": "doc_abc123",
    "file_path": "//server/bauamt/antrag_001.pdf",
    "content": "Bauantrag Nr. 2024-001\nAntragsteller: Max Mustermann\n\nDer Antrag auf Errichtung eines Einfamilienhauses...",
    "language": "de",
    "acl": {
      "allow_groups": ["DOMAIN\\Bauamt-Mitarbeiter"],
      "deny_groups": ["DOMAIN\\Praktikanten"],
      "allow_roles": [],
      "allow_users": [],
      "department": "bauamt",
      "visibility": "internal"
    },
    "metadata": {
      "title": "Bauantrag 2024-001",
      "uploaded_by": "moderator_01",
      "source_type": "smb",
      "mime_type": "application/pdf",
      "municipality_id": "wiener-neudorf",
      "department": ["Bauamt"]
    },
    "chunking": {
      "strategy": "late_chunking",
      "max_chunk_size": 512,
      "overlap": 50
    }
  }'
```

**Response (success):**
```json
{
  "success": true,
  "data": {
    "source_id": "doc_abc123",
    "chunks_created": 8,
    "vectors_stored": 8,
    "collection": "wiener-neudorf",
    "content_type": ["policy", "housing"],
    "entities_extracted": {
      "dates": 3,
      "contacts": 1,
      "amounts": 0
    },
    "embedding_time_ms": 1250,
    "total_time_ms": 3500
  },
  "request_id": "..."
}
```

**Response (empty content):**
```json
{
  "success": false,
  "error": "VALIDATION_EMPTY_CONTENT",
  "detail": "Content must not be empty",
  "request_id": "..."
}
```

**Response (embedding OOM):**
```json
{
  "success": false,
  "error": "EMBEDDING_OOM",
  "detail": "BGE-M3 out of memory — reduce chunk size or content length",
  "request_id": "..."
}
```

**Response (collection not found):**
```json
{
  "success": false,
  "error": "QDRANT_COLLECTION_NOT_FOUND",
  "detail": "Collection 'wiener-neudorf' does not exist. Create it first via POST /collections/init",
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `collection_name` | string | Yes | — | Qdrant collection to store in |
| `source_id` | string | Yes | — | Unique document ID (for updates/deletes) |
| `file_path` | string | Yes | — | Original file path (SMB path or R2 key) |
| `content` | string | Yes | — | Parsed text from `/local/document-parse` or `/local/document-parse/upload` |
| `language` | string | No | auto-detect | ISO 639-1 language code |
| `acl` | object | Yes | — | Access control list (see ACL object below) |
| `metadata` | object | Yes | — | Document metadata (see Metadata object below) |
| `chunking` | object | No | defaults | Chunking configuration (see Chunking config below) |

### ACL object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `visibility` | string | Yes | `public`, `internal`, or `restricted` |
| `allow_groups` | array | No | AD groups with access |
| `deny_groups` | array | No | AD groups explicitly denied |
| `allow_roles` | array | No | Portal roles with access |
| `allow_users` | array | No | Specific user IDs |
| `department` | string | No | Department tag |

### Metadata object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `title` | string | No | Document title (shown in search results) |
| `uploaded_by` | string | No | User or service that uploaded |
| `source_type` | string | No | `smb`, `r2`, or `web` |
| `mime_type` | string | No | Original file MIME type |
| `municipality_id` | string | No | Municipality/tenant ID (stored at payload root in Qdrant) |
| `department` | array of strings | No | Departments within organization (stored at payload root in Qdrant) |

### Chunking config

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | string | `contextual` | `contextual` (recursive splitter + AI context prepended, default for online), `recursive` (recursive character text splitter), `late_chunking` (paragraph-aware), `sentence`, or `fixed` |
| `max_chunk_size` | int | `1200` (online) / `512` (local) | Max chunk size in chars (64-4096) |
| `overlap` | int | 50 | Overlap between chunks in chars (0-512) |

**Chunking strategies:**

| Strategy | Description |
|----------|-------------|
| `contextual` | **Default for online.** Recursive splitter + AI-generated context prepended to each chunk via OpenAI (`gpt-4o-mini`). Context is generated in the same language as the content. Based on Anthropic's Contextual Retrieval technique. |
| `recursive` | Recursive character text splitter. Tries to split on the most semantic separator first (`\n\n` → `\n` → `. ` → `, ` → ` `), recursively falling back to finer separators. |
| `late_chunking` | Paragraph-aware splitting on double newlines, merging small paragraphs. |
| `sentence` | Splits on sentence boundaries (`.` `!` `?`). |
| `fixed` | Raw character count splitting with overlap. |

**Atomic pattern protection** (recursive and contextual strategies): URLs, email addresses, phone numbers, code blocks, and API keys are never split across chunk boundaries.

### Error codes
`VALIDATION_EMPTY_CONTENT`, `VALIDATION_ACL_REQUIRED`, `EMBEDDING_MODEL_NOT_LOADED`, `EMBEDDING_FAILED`, `EMBEDDING_OOM`, `QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_UPSERT_FAILED`, `QDRANT_DISK_FULL`, `CLASSIFY_FAILED`

---

## `DELETE /api/v1/local/vectors/{source_id}`

Remove all vectors for a document from a Qdrant collection.

**Request:**
```bash
curl -X DELETE "https://your-domain/api/v1/local/vectors/doc_abc123?collection_name=wiener-neudorf"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "source_id": "doc_abc123",
    "vectors_deleted": 8
  },
  "request_id": "..."
}
```

**Response (connection failed):**
```json
{
  "success": false,
  "error": "QDRANT_CONNECTION_FAILED",
  "detail": "Failed to connect to Qdrant",
  "request_id": "..."
}
```

### Error codes
`QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`

---

## `POST /api/v1/local/vectors/delete-by-filter`

Delete vectors matching metadata filters. All filters are combined with **AND** logic — only points matching every condition are deleted.

**Filterable metadata fields:** `source_id`, `source_type`, `content_type`, `acl_visibility`, `acl_department`, `municipality_id`, `department`, `language`, `uploaded_by`, `mime_type`, `title`

### Case 1: Delete all vectors from a department

**Request:**
```bash
curl -X POST "https://your-domain/api/v1/local/vectors/delete-by-filter" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "wiener-neudorf",
    "filters": [
      {"key": "acl_department", "value": "bauamt"}
    ]
  }'
```

### Case 2: Delete by source type and content type

**Request:**
```json
{
  "collection_name": "wiener-neudorf",
  "filters": [
    {"key": "source_type", "value": "smb"},
    {"key": "content_type", "value": "funding"}
  ]
}
```

### Case 3: Delete all vectors for an organization

**Request:**
```json
{
  "collection_name": "wiener-neudorf",
  "filters": [
    {"key": "municipality_id", "value": "wiener-neudorf"}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "vectors_deleted": 42,
    "filters_applied": [
      {"key": "source_type", "value": "smb"},
      {"key": "content_type", "value": "funding"}
    ]
  },
  "request_id": "..."
}
```

**Response (connection failed):**
```json
{
  "success": false,
  "error": "QDRANT_CONNECTION_FAILED",
  "detail": "Failed to connect to Qdrant",
  "request_id": "..."
}
```

### Request fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `collection_name` | string | Yes | Qdrant collection to delete from |
| `filters` | array | Yes | List of metadata conditions (AND logic, min 1) |
| `filters[].key` | string | Yes | Metadata field name |
| `filters[].value` | string | Yes | Exact value to match |

### Error codes
`QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`

---

## `PUT /api/v1/local/vectors/update-acl`

Update ACL permissions on existing vectors without re-embedding. Used when file permissions change on the source system.

**Request:**
```bash
curl -X PUT "https://your-domain/api/v1/local/vectors/update-acl" \
  -H "Content-Type: application/json" \
  -d '{
    "collection_name": "wiener-neudorf",
    "source_id": "doc_abc123",
    "acl": {
      "allow_groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Neue-Gruppe"],
      "deny_groups": [],
      "allow_roles": [],
      "allow_users": [],
      "department": "bauamt",
      "visibility": "internal"
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "source_id": "doc_abc123",
    "vectors_updated": 8
  },
  "request_id": "..."
}
```

### Error codes
`QDRANT_CONNECTION_FAILED`, `QDRANT_UPSERT_FAILED`

---

# Endpoint Summary

## Shared Endpoints (no mode prefix)

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| GET | `/api/v1/health` | Liveness probe | None |
| GET | `/api/v1/ready` | Readiness probe | None / HMAC |
| GET | `/metrics` | Prometheus metrics | None |
| POST | `/api/v1/classify` | Classify + extract entities | HMAC |
| POST | `/api/v1/search` | Permission-aware semantic / hybrid search | HMAC |
| POST | `/api/v1/collections/init` | Create Qdrant collection | HMAC |
| GET | `/api/v1/collections/stats` | Collection statistics | HMAC |

## Online Endpoints (`X-API-Key` required)

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/api/v1/online/scrape` | Scrape webpage (Crawl4AI or Jina) | HMAC + API Key |
| POST | `/api/v1/online/crawl` | Discover URLs from site/sitemap | HMAC + API Key |
| POST | `/api/v1/online/document-parse` | Parse document from URL | HMAC + API Key |
| POST | `/api/v1/online/document-parse/upload` | Parse uploaded file | HMAC + API Key |
| POST | `/api/v1/online/ingest` | Chunk + embed + store web content | HMAC + API Key |
| DELETE | `/api/v1/online/vectors/{source_id}` | Delete document vectors | HMAC + API Key |
| POST | `/api/v1/online/vectors/delete-by-filter` | Delete vectors by metadata filter | HMAC + API Key |
| POST | `/api/v1/online/vectors/sparse-encode` | BM25 sparse-encode arbitrary text | HMAC + API Key |

## Local Endpoints (trusted network)

| Method | Endpoint | Purpose | Auth |
|--------|----------|---------|------|
| POST | `/api/v1/local/document-parse` | Parse document (SMB, R2) | HMAC |
| POST | `/api/v1/local/document-parse/upload` | Parse uploaded file | HMAC |
| POST | `/api/v1/local/discover` | Scan file sources for changes | HMAC |
| POST | `/api/v1/local/ingest` | Chunk + embed + store local docs | HMAC |
| DELETE | `/api/v1/local/vectors/{source_id}` | Delete document vectors | HMAC |
| POST | `/api/v1/local/vectors/delete-by-filter` | Delete vectors by metadata filter | HMAC |
| PUT | `/api/v1/local/vectors/update-acl` | Update ACL without re-embedding | HMAC |

---

# All Error Codes

| Category | Code | Description |
|----------|------|-------------|
| **Validation** | `VALIDATION_URL_INVALID` | URL is empty or doesn't start with http/https |
| | `VALIDATION_PATH_OUTSIDE_ROOTS` | Path not in allowed roots |
| | `VALIDATION_ACL_REQUIRED` | ACL missing from ingest request |
| | `VALIDATION_EMPTY_CONTENT` | Content/query is empty |
| | `VALIDATION_USER_REQUIRED` | User context missing from search |
| **Auth** | `AUTH_MISSING` | X-Signature or X-Timestamp header missing |
| | `AUTH_INVALID` | HMAC signature doesn't match |
| | `AUTH_EXPIRED` | Timestamp outside ±5 min window |
| | `AUTH_API_KEY_MISSING` | X-API-Key header missing on online endpoint |
| | `AUTH_API_KEY_INVALID` | X-API-Key not in `DP_ONLINE_API_KEYS` |
| **SMB** | `SMB_CONNECTION_FAILED` | Cannot connect to SMB share |
| | `SMB_AUTH_FAILED` | SMB credentials rejected |
| | `SMB_PATH_NOT_FOUND` | Share path doesn't exist |
| | `SMB_FILE_NOT_FOUND` | File not found on share |
| | `SMB_FILE_LOCKED` | File is locked by another process |
| **R2** | `R2_CONNECTION_FAILED` | Cannot connect to Cloudflare R2 |
| | `R2_FILE_NOT_FOUND` | Object key not found or presigned URL missing |
| | `R2_PRESIGNED_EXPIRED` | Pre-signed URL has expired |
| **LDAP** | `LDAP_CONNECTION_FAILED` | Cannot connect to LDAP/AD |
| | `LDAP_AUTH_FAILED` | LDAP bind credentials rejected |
| **Parse** | `PARSE_FAILED` | General parsing failure |
| | `PARSE_ENCRYPTED` | Document is password-protected |
| | `PARSE_CORRUPTED` | Document file is corrupted |
| | `PARSE_EMPTY` | Document has no extractable text |
| | `PARSE_TIMEOUT` | Parsing timed out |
| | `PARSE_UNSUPPORTED_FORMAT` | File format not supported |
| **Scrape** | `SCRAPE_FAILED` | General scraping failure |
| | `SCRAPE_BLOCKED` | Website blocked the request |
| | `SCRAPE_TIMEOUT` | Scraping timed out |
| | `SCRAPE_EMPTY` | Page returned no content |
| | `SCRAPE_ROBOTS_BLOCKED` | Blocked by robots.txt |
| **Crawl** | `CRAWL_SITEMAP_NOT_FOUND` | No URLs found in sitemap |
| | `CRAWL_MAX_URLS_EXCEEDED` | URL limit reached |
| **Classify** | `CLASSIFY_FAILED` | Classification failed |
| | `CLASSIFY_LOW_CONFIDENCE` | Confidence below threshold |
| | `ENTITY_EXTRACTION_FAILED` | Entity extraction failed |
| **Embedding** | `EMBEDDING_MODEL_NOT_LOADED` | Embedding model not available |
| | `EMBEDDING_FAILED` | Embedding generation failed |
| | `EMBEDDING_OOM` | Out of memory during embedding |
| **Qdrant** | `QDRANT_CONNECTION_FAILED` | Cannot connect to Qdrant |
| | `QDRANT_COLLECTION_NOT_FOUND` | Collection doesn't exist |
| | `QDRANT_UPSERT_FAILED` | Failed to store vectors |
| | `QDRANT_SEARCH_FAILED` | Search query failed |
| | `QDRANT_DELETE_FAILED` | Failed to delete vectors |
| | `QDRANT_DISK_FULL` | Qdrant disk space exhausted |
