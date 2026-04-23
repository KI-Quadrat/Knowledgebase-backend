# KI² Data Plane — API Reference

0Pure endpoint reference for the KI² Data Plane Service. For architecture, security, configuration, error handling patterns, and deployment details, see: **KI² Data Plane — Architecture & Security**.
* * *

# Base URL

```cpp
http://localhost:8000/api/v1
```

On-premise: `http://{vm-ip}:8000/api/v1`
Cloud: `https://worker.ki-quadrat.at/api/v1`

All endpoints except `/health` and `/ready` require HMAC-SHA256 signed headers. See Architecture doc for auth details.
* * *

# Health & Status

## `GET /api/v1/health`

Liveness check. No auth.

```json
{ "status": "ok" }
```

## `GET /api/v1/ready`

Readiness check. No auth for minimal response. HMAC auth for full response.

**Minimal (unauthenticated):**

```json
{ "ready": true }
```

**Full (authenticated):**

```json
{
  "ready": true,
  "services": {
    "qdrant": true,
    "bge_m3": true,
    "docling": true,
    "crawl4ai": true,
    "ldap": true
  },
  "mode": "on-premise",
  "tenant_id": "wiener-neudorf",
  "worker_id": "wn-worker-01",
  "version": "1.2.0"
}
```

* * *

# File Discovery & ACL

## `POST /api/v1/discover`

Scans file sources, reads permissions, computes hashes, returns what changed. First step in every ingestion. Does not parse or embed.

**Request (on-prem):**

```json
{
  "source": "smb",
  "paths": ["//server/abteilung/dokumente", "//server/bauamt"],
  "since_hash_map": {
    "//server/abteilung/dokumente/antrag.pdf": "sha256:abc123..."
  }
}
```

**Request (cloud):**

```json
{
  "source": "r2",
  "paths": ["tenant/wiener-neudorf/uploads/"],
  "since_hash_map": {}
}
```

| Parameter | Type | Required | Description |
| ---| ---| ---| --- |
| source | string | Yes | `smb`, `r2`, or `url` |
| paths | array | Yes | SMB paths, R2 prefixes, or website URLs |
| since\_hash\_map | object | No | `file_path → last_known_hash`. Same hash = skipped. |

**Errors:** `SMB_CONNECTION_FAILED`, `SMB_AUTH_FAILED`, `SMB_PATH_NOT_FOUND`, `R2_CONNECTION_FAILED`, `LDAP_CONNECTION_FAILED`, `LDAP_AUTH_FAILED`, `VALIDATION_PATH_OUTSIDE_ROOTS`

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
        "path": "//server/bauamt/bauantraege/2024/antrag_001.pdf",
        "file_hash": "sha256:abc123...",
        "size_bytes": 245000,
        "mime_type": "application/pdf",
        "last_modified": "2025-03-01T10:30:00Z",
        "status": "new",
        "acl": {
          "source": "ntfs",
          "allow_groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Bauamt-Leitung"],
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

* * *

# Document Parsing

## `POST /api/v1/parse`

Parses a document and returns extracted text. Uses **Docling**.

**Request (on-prem):**

```json
{
  "file_path": "//server/bauamt/antrag_001.pdf",
  "source": "smb",
  "mime_type": "application/pdf"
}
```

**Request (cloud):**

```json
{
  "file_path": "tenant/wiener-neudorf/uploads/foerderung.pdf",
  "source": "r2",
  "r2_presigned_url": "https://r2.ki-quadrat.at/...",
  "mime_type": "application/pdf"
}
```

| Parameter | Type | Required | Description |
| ---| ---| ---| --- |
| file\_path | string | Yes | SMB path or R2 key |
| source | string | Yes | `smb` or `r2` |
| r2\_presigned\_url | string | Conditional | Pre-signed URL if `r2`. Expires in 15 min. |
| mime\_type | string | Yes | MIME type |

**Supported formats:** PDF, DOC, DOCX, XLSX, PPTX, TXT, HTML, RTF
**Max file size:** 50MB (configurable)

**Errors:** `PARSE_FAILED`, `PARSE_ENCRYPTED`, `PARSE_CORRUPTED`, `PARSE_EMPTY`, `PARSE_TIMEOUT`, `PARSE_UNSUPPORTED_FORMAT`, `SMB_FILE_NOT_FOUND`, `SMB_FILE_LOCKED`, `R2_FILE_NOT_FOUND`, `R2_PRESIGNED_EXPIRED`

**Response:**

```json
{
  "success": true,
  "data": {
    "file_path": "//server/bauamt/antrag_001.pdf",
    "content": "Bauantrag Nr. 2024-001\nAntragsteller: ...",
    "pages": 12,
    "language": "de",
    "extracted_tables": 2,
    "content_length": 15420
  },
  "request_id": "..."
}
```

* * *

# Web Scraping

## `POST /api/v1/scrape`

Scrapes a single webpage. Uses **Crawl4AI**. Optionally parses inner images (OCR via LlamaParse) and inner documents (PDF, DOCX, etc.) found on the page.

**Request:**

```json
{
  "url": "https://www.wiener-neudorf.gv.at/gemeindeamt/kontakt/",
  "inner_img": true,
  "inner_docs": true
}
```

| Parameter | Type | Required | Default | Description |
| ---| ---| ---| ---| --- |
| url | string | Yes | — | Full URL of the webpage to scrape (must start with `http://` or `https://`) |
| inner\_img | bool | No | `false` | If `true`, extract images from the page and parse them via LlamaParse OCR. Returns extracted text content, alt text, and title for each image. |
| inner\_docs | bool | No | `false` | If `true`, extract linked documents (PDF, DOCX, XLSX, etc.) from the page and parse them via the document parsing backend. Returns extracted text content for each document. |

**Errors:** `SCRAPE_FAILED`, `SCRAPE_BLOCKED`, `SCRAPE_TIMEOUT`, `SCRAPE_EMPTY`, `SCRAPE_ROBOTS_BLOCKED`, `VALIDATION_URL_INVALID`

**Response:**

```json
{
  "success": true,
  "data": {
    "url": "https://www.wiener-neudorf.gv.at/gemeindeamt/kontakt/",
    "title": "Kontakt - Gemeinde Wiener Neudorf",
    "content": "Gemeindeamt Wiener Neudorf\nHauptplatz 1...",
    "content_length": 3200,
    "language": "de",
    "links_found": 45,
    "last_modified": "2025-03-01T00:00:00Z",
    "inner_images": [
      {
        "url": "https://www.wiener-neudorf.gv.at/images/rathaus.jpg",
        "alt": "Rathaus Wiener Neudorf",
        "title": "Rathaus",
        "content": "Gemeindeamt Wiener Neudorf — Hauptplatz 1",
        "content_length": 42,
        "error": null
      }
    ],
    "inner_documents": [
      {
        "url": "https://www.wiener-neudorf.gv.at/files/oeffnungszeiten.pdf",
        "title": "Öffnungszeiten (PDF)",
        "doc_type": "pdf",
        "content": "Öffnungszeiten Gemeindeamt\nMontag bis Freitag...",
        "pages": 1,
        "content_length": 520,
        "language": "de",
        "error": null
      }
    ]
  },
  "request_id": "..."
}
```

> **Note:** `inner_images` and `inner_documents` are `null` when the corresponding parameter is `false` (default). Each item may contain an `error` field if parsing failed for that specific image or document — the overall scrape still succeeds.
```

## `POST /api/v1/crawl`

Discovers URLs from a website. Returns URLs only, does not scrape content.

**Request:**

```json
{
  "url": "https://www.wiener-neudorf.gv.at",
  "method": "sitemap",
  "max_depth": 3,
  "max_urls": 500
}
```

| Parameter | Type | Required | Description |
| ---| ---| ---| --- |
| url | string | Yes | Base URL or sitemap URL |
| method | string | Yes | `sitemap` or `crawl` |
| max\_depth | int | No | Max link depth. Default: 3 |
| max\_urls | int | No | Max URLs. Default: 500 |

**Errors:** `CRAWL_SITEMAP_NOT_FOUND`, `CRAWL_MAX_URLS_EXCEEDED`, `SCRAPE_FAILED`, `SCRAPE_TIMEOUT`, `VALIDATION_URL_INVALID`

**Response:**

```json
{
  "success": true,
  "data": {
    "base_url": "https://www.wiener-neudorf.gv.at",
    "method_used": "sitemap",
    "urls": [
      { "url": "https://www.wiener-neudorf.gv.at/gemeindeamt/kontakt/", "type": "page", "last_modified": "2025-03-01T00:00:00Z" },
      { "url": "https://www.wiener-neudorf.gv.at/files/foerderung.pdf", "type": "document", "last_modified": null }
    ],
    "total_urls": 234
  },
  "request_id": "..."
}
```

* * *

# Content Intelligence

## `POST /api/v1/classify`

Classifies content and extracts entities.

**Request:**

```json
{
  "content": "Das Förderprogramm für erneuerbare Energien gilt ab...",
  "language": "de"
}
```

**Errors:** `CLASSIFY_FAILED`, `CLASSIFY_LOW_CONFIDENCE`, `ENTITY_EXTRACTION_FAILED`, `VALIDATION_EMPTY_CONTENT`

**Categories:** `funding`, `event`, `policy`, `contact`, `form`, `announcement`, `minutes`, `report`, `general`

**Response:**

```json
{
  "success": true,
  "data": {
    "classification": "funding",
    "confidence": 0.94,
    "sub_categories": ["renewable_energy", "subsidy"],
    "entities": {
      "dates": ["2025-04-01"],
      "deadlines": ["2025-06-30"],
      "amounts": ["EUR 5.000"],
      "contacts": ["energie@wiener-neudorf.gv.at"],
      "departments": ["Umweltamt"]
    },
    "summary": "Förderung für erneuerbare Energien, Antragsfrist bis 30. Juni 2025"
  },
  "request_id": "..."
}
```

* * *

# Embed & Store

## `POST /api/v1/ingest`

The core endpoint. Takes parsed content + ACL → chunks → classifies → embeds → stores in Qdrant.

**Request:**

```json
{
  "source_id": "doc_abc123",
  "file_path": "//server/bauamt/antrag_001.pdf",
  "content": "Bauantrag Nr. 2024-001\nAntragsteller: ...",
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
    "mime_type": "application/pdf"
  },
  "chunking": {
    "strategy": "late_chunking",
    "max_chunk_size": 512,
    "overlap": 50
  }
}
```

| Parameter | Type | Required | Description |
| ---| ---| ---| --- |
| source\_id | string | Yes | Unique doc ID (for updates/deletes) |
| file\_path | string | Yes | Original file path |
| content | string | Yes | Text from `/parse` or `/scrape` |
| language | string | No | ISO code. Auto-detected if omitted. |
| acl | object | Yes | Every document must have ACL. |
| acl.allow\_groups | array | No | Groups with access |
| acl.deny\_groups | array | No | Groups explicitly denied |
| acl.allow\_roles | array | No | Portal roles with access |
| acl.allow\_users | array | No | Specific user IDs |
| acl.department | string | No | Department |
| acl.visibility | string | Yes | `public`, `internal`, or `restricted` |
| metadata | object | Yes | Additional metadata |
| chunking | object | No | Override chunking defaults |

**Errors:** `VALIDATION_ACL_REQUIRED`, `VALIDATION_EMPTY_CONTENT`, `EMBEDDING_MODEL_NOT_LOADED`, `EMBEDDING_FAILED`, `EMBEDDING_OOM`, `QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_UPSERT_FAILED`, `QDRANT_DISK_FULL`, `CLASSIFY_FAILED`

**Response:**

```json
{
  "success": true,
  "data": {
    "source_id": "doc_abc123",
    "chunks_created": 47,
    "vectors_stored": 47,
    "collection": "wiener-neudorf",
    "classification": "policy",
    "entities_extracted": { "dates": 3, "contacts": 1, "amounts": 0 },
    "embedding_time_ms": 1200,
    "total_time_ms": 1850
  },
  "request_id": "..."
}
```

* * *

# Permission-Aware Search

## `POST /api/v1/search`

Semantic search with **mandatory** permission filtering. No search is ever unfiltered.

**Request (employee):**

```json
{
  "query": "Wann ist die nächste Förderung für Solaranlagen?",
  "user": {
    "type": "employee",
    "user_id": "maria@wiener-neudorf.gv.at",
    "groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Alle-Mitarbeiter"],
    "roles": ["member"],
    "department": "bauamt"
  },
  "filters": { "classification": ["funding"] },
  "top_k": 10,
  "score_threshold": 0.5
}
```

**Request (citizen):**

```json
{
  "query": "Öffnungszeiten Gemeindeamt",
  "user": { "type": "citizen", "user_id": "anonymous" },
  "top_k": 5,
  "score_threshold": 0.5
}
```

| Parameter | Type | Required | Description |
| ---| ---| ---| --- |
| query | string | Yes | Natural language question |
| user | object | Yes | Always required. |
| user.type | string | Yes | `citizen` or `employee` |
| user.groups | array | Conditional | AD groups (employee) |
| user.roles | array | Conditional | Portal roles (employee) |
| user.department | string | No | For boost/filtering |
| filters | object | No | Content filters |
| top\_k | int | No | Default: 10 |
| score\_threshold | float | No | Default: 0.5 |

**Errors:** `VALIDATION_USER_REQUIRED`, `EMBEDDING_MODEL_NOT_LOADED`, `EMBEDDING_FAILED`, `QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_SEARCH_FAILED`

**Response:**

```json
{
  "success": true,
  "data": {
    "results": [
      {
        "chunk_id": "doc_abc123_chunk_07",
        "source_id": "doc_abc123",
        "chunk_text": "Die Förderung für Solaranlagen beträgt bis zu EUR 5.000...",
        "score": 0.92,
        "source_path": "//server/bauamt/foerderungen/solar_2025.pdf",
        "classification": "funding",
        "entities": { "amounts": ["EUR 5.000"], "deadlines": ["2025-06-30"] },
        "metadata": { "title": "Solarförderung 2025", "department": "bauamt", "source_type": "smb" }
      }
    ],
    "total_results": 7,
    "query_embedding_ms": 15,
    "search_ms": 22,
    "permission_filter_applied": {
      "visibility": ["public", "internal"],
      "must_match_groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Alle-Mitarbeiter"],
      "must_not_match_groups": []
    }
  },
  "request_id": "..."
}
```

* * *

# Vector Management

## `DELETE /api/v1/vectors/{source_id}`

Remove all vectors for a document.

**Errors:** `QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`

```json
{
  "success": true,
  "data": { "source_id": "doc_abc123", "vectors_deleted": 47 },
  "request_id": "..."
}
```

## `POST /api/v1/vectors/update-acl`

Update permissions on existing vectors without re-embedding.

**Request:**

```json
{
  "source_id": "doc_abc123",
  "acl": {
    "allow_groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Neue-Gruppe"],
    "deny_groups": [],
    "visibility": "internal"
  }
}
```

**Errors:** `QDRANT_CONNECTION_FAILED`, `QDRANT_UPSERT_FAILED`

```json
{
  "success": true,
  "data": { "source_id": "doc_abc123", "vectors_updated": 47 },
  "request_id": "..."
}
```

* * *

# Collection Management

## `POST /api/v1/collections/init`

Create Qdrant collection for a municipality. Called once during setup.

**Request:**

```json
{
  "collection_name": "wiener-neudorf",
  "vector_config": { "dense_dim": 1024, "sparse": true, "distance": "cosine" }
}
```

**Errors:** `QDRANT_CONNECTION_FAILED`

```json
{
  "success": true,
  "data": { "collection": "wiener-neudorf", "created": true, "dense_dim": 1024, "sparse_enabled": true },
  "request_id": "..."
}
```

## `GET /api/v1/collections/stats`

Collection statistics.

**Errors:** `QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`

```json
{
  "success": true,
  "data": {
    "collection": "wiener-neudorf",
    "total_vectors": 12450,
    "total_documents": 523,
    "disk_usage_mb": 245,
    "by_classification": { "funding": 1200, "event": 3400, "policy": 2100, "contact": 800, "general": 4950 },
    "by_visibility": { "public": 8200, "internal": 3800, "restricted": 450 }
  },
  "request_id": "..."
}
```

* * *

# Endpoint Summary

| Method | Endpoint | Purpose | Auth |
| ---| ---| ---| --- |
| GET | `/api/v1/health` | Liveness | None |
| GET | `/api/v1/ready` | Readiness | None (minimal) / HMAC (full) |
| POST | `/api/v1/discover` | Scan files + ACL + detect changes | HMAC |
| POST | `/api/v1/parse` | Parse document (Docling) | HMAC |
| POST | `/api/v1/scrape` | Scrape webpage (Crawl4AI) | HMAC |
| POST | `/api/v1/crawl` | Discover URLs from site/sitemap | HMAC |
| POST | `/api/v1/classify` | Classify + extract entities | HMAC |
| POST | `/api/v1/ingest` | Chunk + embed + store with ACL | HMAC |
| POST | `/api/v1/search` | Permission-aware semantic search | HMAC |
| DELETE | `/api/v1/vectors/{source_id}` | Delete document vectors | HMAC |
| POST | `/api/v1/vectors/update-acl` | Update ACL without re-embedding | HMAC |
| POST | `/api/v1/collections/init` | Create Qdrant collection | HMAC |
| GET | `/api/v1/collections/stats` | Collection statistics | HMAC |