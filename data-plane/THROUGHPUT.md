# Throughput & RPM Reference — Online Pipeline

Capacity ceilings for the **online endpoints** (`/api/v1/online/*`), end-to-end
scrape → parse → classify → ingest. Numbers are derived from the code as it
stands, not measured. Anything that depends on an external provider's tier
(OpenAI quota, TEI server hardware, Crawl4AI host) is marked with a caveat.

All file:line references are relative to `data-plane/`.

---

## 1. What "per domain" means

The scrape rate limit is **per origin (`netloc`)**, not per request,
not per IP, not per tenant.

- Defined at `app/services/rate_limiter.py:17-83` (`DomainRateLimiter`).
- Domain key: `urlparse(url).netloc.lower()` (line 42).
- Backed by a Redis sliding-window sorted set per domain.

What this implies:

| URLs you send | Same bucket? | Effective cap |
|---|---|---|
| `https://example.com/a` + `https://example.com/b` | Yes | 10 per 60 s combined |
| `https://www.example.com` + `https://api.example.com` | No (different netloc) | 10 per 60 s each |
| `https://example.com` + `https://other-site.com` | No | 10 per 60 s each |
| Repeat scrape of same URL within `cache_ttl` (3600 s default) | Cache hit — skips limiter entirely (`scraper_service.py:240-252`) | unbounded |

So "single domain" means: you're hammering one origin. Spread requests across
N distinct origins and you get **N × 10 RPM** scrape capacity.

The limit is configurable via `DP_RATE_LIMIT_PER_DOMAIN` and
`DP_RATE_LIMIT_WINDOW` (`config.py:101-102`).

---

## 2. Server-level capacity

Single uvicorn process, no worker fan-out, no acceptance limit.

| Setting | Value | File:line |
|---|---|---|
| Workers | 1 (uvicorn default — no `--workers` flag) | `Dockerfile:99` |
| `--limit-concurrency` | not set | `Dockerfile:99` |
| `--backlog` | not set (default 2048) | uvicorn default |
| Rate-limit middleware | none | `app/main.py` |
| HMAC auth on online endpoints | optional (only when `DP_ONLINE_API_KEYS` set) | `config.py:37` |

**Implication:** the server itself isn't the bottleneck — asyncio handles
~1000 concurrent connections fine. The cap is whatever downstream service
gets saturated first.

---

## 3. Per-component RPM ceilings

### 3.1 Scraping backends

Three browser-rendering backends are wired in, plus a raw-httpx final
fallback. The active per-request backend is chosen by the request body's
`scraper` field, defaulting to the `DEFAULT_SCRAPER` env var.

| Component | Concurrency in our code | External cap | File:line |
|---|---|---|---|
| **Per-domain rate limit** | sliding window, Redis-backed, 10 req / 60 s | — | `rate_limiter.py:27-28`; `config.py:101-102` |
| **Jina Reader (default)** | no in-process semaphore | depends on your Jina plan — see 3.1.1 | `config.py:60-61` |
| **Crawl4AI (`/md`)** | no in-process semaphore | depends on your Crawl4AI host's CPU/Chromium pool | `config.py:55-57` |
| **Firecrawl (`/v2/scrape`)** | no in-process semaphore | depends on your Firecrawl plan — see 3.1.1 | `config.py:71-74` |
| **Raw httpx (fallback)** | no semaphore | target site's own limits | `scrape.py:379-399` |
| **Scrape cache (Redis)** | — | cache hit returns before rate limiter | `config.py:23`; `scraper_service.py:240-252` |

**Ceiling per single origin: 10 scrape RPM** (hard, regardless of which
backend you pick — the limiter sits in front of the backend dispatcher).
Multi-origin scales linearly until you hit the backend's own RPM cap (see
next sub-section).

### 3.1.1 Backend provider RPM / batch caps

Numbers below are upstream provider limits, not our code — change with
your plan. Use these to size aggregate concurrency above the per-domain
10-RPM floor.

**Jina Reader** (per `JINA_API_KEY`; IP-wide cap of 10,000 req/60 s
applies on top regardless of tier):

| Tier | RPM | TPM | Concurrent |
|---|---|---|---|
| Free | 100 | 100K | 2 |
| Paid | 500 | 2M | 50 |
| Premium | 5,000 | 50M | 500 |

**No batch endpoint.** `r.jina.ai/{url}` is single-URL only — the only
way to "batch" with Jina is client-side fan-out within the concurrency
quota.

**Firecrawl** (per `FIRECRAWL_API_KEY`; managed cloud only — for EU data
residency point `FIRECRAWL_API_URL` at a self-hosted instance and the
caps below no longer apply):

| Plan | `/scrape` & `/map` RPM | `/crawl` & `/batch/scrape` RPM |
|---|---|---|
| Free | 10 | 1 |
| Hobby | 100 | 15 |
| Standard | 500 | 50 |
| Growth | 5,000 | 250 |
| Scale | 7,500 | 750 |

`/batch/scrape` and `/crawl` share one bucket — burning crawl quota also
limits batch jobs. `/batch/scrape` accepts many URLs per request (no
published hard cap); we don't expose a batch endpoint in our router
today, but the Firecrawl client method exists if added.

**Crawl4AI** — self-hosted in this codebase (`crawl4ai:11235`). No
provider RPM; limits are whatever Chromium pool size + CPU/RAM your
deployment can sustain. `POST /md` is single-URL; `POST /crawl` accepts
a `urls` array (we currently always send one root URL — see
`crawl4ai_client.py:345-356`).

**Practical aggregate ceiling** (above the 10-RPM-per-origin floor):

| Backend | Aggregate scrape cap (default tier assumptions) |
|---|---|
| Jina Paid (default) | min(500 RPM, 50 concurrent in-flight) |
| Firecrawl Standard | 500 RPM for `/scrape`; 50 RPM if you switch to `/batch/scrape` |
| Crawl4AI self-host | whatever your Chromium concurrency permits |

### 3.2 Document parsing

| Component | Config | Concurrency | File:line |
|---|---|---|---|
| **LlamaParse (cloud)** | `LLAMA_CLOUD_API_KEY` set | no in-process semaphore | `config.py:70-71` |
| **Local unstructured** | `LLAMA_CLOUD_API_KEY` empty | no in-process semaphore | `config.py:70` |
| **Max file size** | 50 MB | — | `config.py:26` |

LlamaParse free tier is 1000 pages/day at the time of writing — depends on
your plan. No per-RPM limit in our code.

### 3.3 OpenAI calls (classifier, funding extractor, contextual enricher, OpenAI embedder)

Three pipeline stages call OpenAI Chat (`gpt-4o-mini` by default); one stage
calls OpenAI Embeddings (`text-embedding-3-small`). All four go through the
same `OPENAI_API_KEY` quota.

| Stage | Triggered when | Calls per ingest | Concurrency | File:line |
|---|---|---|---|---|
| **Classifier (on `/scrape` + `/document-parse`)** | always | 1 chat call | no semaphore | `config.py:127` |
| **Funding extractor (`/ingest`)** | `assistant_type=="funding"`; always on `/ingest/at` | 1 chat call | no semaphore; runs in parallel with the rest via `asyncio.create_task` | `routers/online/ingest.py:101-112` |
| **Contextual enricher** | `chunking.strategy=="contextual"` (the default) | 1 batched chat call per 32 chunks (`openai_contextual_max_batch`); falls back to per-chunk on failure | `Semaphore(10)` on per-chunk fallback path | `contextual.py:77, 81`; `config.py:123` |
| **OpenAI embedder (`embedding_model=="openai"`)** | when caller picks openai | 1 embedding call per 256 chunks (`openai_embed_max_batch`) | no semaphore — sequential windows | `config.py:118` |

Char caps on inputs (`config.py:127-129`): 120K for classify, 120K for funding
extract, 60K for contextual. Above these the input is truncated.

**LiteLLM fallback** (`config.py:131-137`): if the OpenAI call fails with a
rate-limit / connection / 5xx, the classifier, contextual enricher, and
funding extractor retry against `LITELLM_URL`. Empty URL or key disables it.

**Practical RPM ceiling: governed by your OpenAI tier.** At Tier 1 (gpt-4o-mini
500 RPM / 200K TPM), classifier + enricher + funding together consume 2–3
calls per ingest, so plan **~150 ingests/min** before hitting OpenAI 429s.
With BGE-M3 instead of OpenAI for embeddings, the embedding step drops off
the OpenAI quota.

### 3.4 TEI dense (BGE-M3) embedding

`embedding_model="bge_m3"` (now the default — see `models/online/ingest.py:91`).

| Component | Setting | Value | File:line |
|---|---|---|---|
| Server URL | `TEI_EMBED_URL_AT` | `https://embed.ki2.at` | `config.py:147` |
| Per-request batch cap | `TEI_EMBED_MAX_BATCH_AT` | 32 (TEI server's `--max-client-batch-size`) | `config.py:154` |
| Concurrency in our code | — | none — sequential windows in `embed_batch` | `services/embedding/bge_m3_client.py` |

**What the 32 means in practice:** the TEI server is started with
`--max-client-batch-size=32`, so **one HTTP request to `/v1/embeddings` may
carry up to 32 input strings**. Our client (`bge_m3_client.py`'s
`embed_batch`) splits anything larger into sequential 32-item POSTs against
the same connection.

So for a 10-chunk document: 1 HTTP request, 10 inputs in the batch payload.
For a 100-chunk document: 4 sequential HTTP requests (32 + 32 + 32 + 4).
The 32 is **not** a per-minute limit — it's a per-call payload cap.

**Real RPM ceiling is set by the TEI server's GPU**, not by the 32. On
typical TEI BGE-M3 hardware: ~5–20 ms per input item, so one client serializes
~500–1500 inputs/s. Multiply by the number of concurrent ingest workers (we
have none above the asyncio default) to get aggregate throughput.

### 3.5 TEI sparse embedding (hybrid mode)

`vector_config.search_mode="hybrid"`.

| Component | Setting | Value | File:line |
|---|---|---|---|
| Server URL | `SPARSE_EMBED_URL_AT` | `https://sparse.ki2.at` | `config.py:161` |
| Per-request batch cap | `SPARSE_EMBED_MAX_BATCH_AT` | 32 | `config.py:169` |
| Concurrency | — | none — sequential windows; runs **in parallel with dense embed** via `asyncio.gather` | `services/ingest/ingest_service.py` |

Same shape as dense: **32 = max inputs in one HTTP POST to
`{SPARSE_EMBED_URL_AT}/embed_sparse`**, not a per-minute cap. Larger inputs
split into sequential 32-item POSTs. Real RPM is GPU-bound on the sparse
server. Dense and sparse fire concurrently via `asyncio.gather`, so total
ingest wall-clock = `max(dense_time, sparse_time)`.

### 3.6 Qdrant write

| Component | Setting | Value | File:line |
|---|---|---|---|
| Default URL | `QDRANT_URL` | `http://qdrant:6333` | `config.py:80` |
| AT instance | `QDRANT_URL_AT` (optional override) | empty by default | `config.py:93-95` |
| Upsert batching | one HTTP PUT per ingest, all chunks in one request | unbounded | `services/embedding/qdrant_service.py` |
| Connection pool | httpx default (10) | — | — |

Single PUT per ingest is not a per-RPM bottleneck. A self-hosted Qdrant on
modest hardware handles thousands of upserts/min easily.

### 3.7 Redis (cache + rate limiter)

| Use | Setting | File:line |
|---|---|---|
| Scrape cache TTL | 3600 s | `config.py:23` |
| Rate limiter window state | 60 s | `config.py:102` |
| Connection URL | `REDIS_URL` | `config.py:98` |

Not a meaningful bottleneck for these workloads.

---

## 4. Per-request cost — one ingest of a 5 KB page (~10 chunks)

Reading the orchestration in `routers/online/ingest.py:85-150` and
`services/ingest/ingest_service.py`:

| Step | Sequential or parallel? | Typical wall-clock | Notes |
|---|---|---|---|
| Funding extraction (if funding) | parallel (`asyncio.create_task` at line 104) | 1–2 s | OpenAI chat |
| Chunking | sequential | ~10 ms | sync |
| Contextual enrichment (if `strategy=contextual`, default) | sequential | **5–30 s** | batched OpenAI chat — dominates |
| Dense embed | parallel with sparse | 100–500 ms | TEI BGE-M3 or OpenAI |
| Sparse embed (if hybrid) | parallel with dense | 100–500 ms | TEI sparse |
| Qdrant upsert | sequential | ~500 ms | one PUT |

**Total wall-clock per ingest:**

| Configuration | Approx duration |
|---|---|
| `strategy=recursive`, semantic, no funding | **~1–2 s** |
| `strategy=recursive`, semantic, funding | **~2–3 s** (funding overlaps with embed) |
| `strategy=contextual` (default), semantic | **~6–15 s** |
| `strategy=contextual`, hybrid, funding | **~8–20 s** |

---

## 5. End-to-end RPM ceiling (scrape → parse → classify → ingest)

This is the answer you actually want. "Scrape → ingest" means the caller does:
1. `POST /online/scrape` (or `/document-parse`) — page fetch + classifier
2. `POST /online/ingest` (or `/ingest/at`) — chunk + enrich + embed + Qdrant

| Scenario | Cap | Bottleneck |
|---|---|---|
| **Single origin** (one website, default settings) | **10 RPM** | `DomainRateLimiter` — 10 req/60 s on the scrape step |
| **N origins**, `strategy=contextual` (default), no funding | **~20–30 RPM total** | contextual enricher's sequential OpenAI windows |
| **N origins**, `strategy=contextual`, `assistant_type=funding` | **~15–25 RPM total** | enricher + funding extractor share the same OpenAI tier |
| **N origins**, `strategy=recursive` (caller skips enrichment) | **~50–100 RPM total** | classifier OpenAI chat (1–2 s per call) |
| **N origins**, `strategy=recursive`, caller pre-classifies and pre-extracts (skip classifier + funding) | **~200–400 RPM total** | TEI embedding throughput / OpenAI embed quota |
| **N origins**, content already in cache (scrape cache hit) | scrape step bypasses limiter; cap shifts to ingest's OpenAI/TEI ceilings | cache hit |

### Quick rules of thumb

- One origin will never exceed **10 RPM** end-to-end. To beat that, spread
  across origins or warm the scrape cache.
- The classifier runs on every `/scrape` and `/document-parse` response —
  there is no "skip classify" toggle. If you need higher throughput, either
  upgrade your OpenAI tier or self-host the classifier behind LiteLLM
  (`LITELLM_URL`).
- Contextual enrichment is the most expensive step. Switch
  `chunking.strategy` to `"recursive"` for a ~5× speedup at the cost of
  retrieval quality.
- Hybrid search (`search_mode=hybrid`) adds the sparse embedder but it runs
  concurrently with the dense embedder, so it doesn't move the RPM cap.

### 5.1 `POST /api/v1/online/batch/ingest` — N items in one call

The batch endpoint runs up to `DP_BATCH_INGEST_CONCURRENCY` items in parallel
(default 10) under one asyncio semaphore. Each item executes the **same**
per-document pipeline as `/ingest` — same OpenAI/TEI/Qdrant cost — just
fanned out. Per-item failures are reported in `results[]` with `success=false`
but **do not abort the batch**. No per-domain scrape limit applies (caller
supplies the content).

Numbers below assume `DP_MAX_BATCH_INGEST_ITEMS=50` and
`DP_BATCH_INGEST_CONCURRENCY=10`. Adjust linearly with concurrency until you
hit the binding upstream cap.

**How two "batches" compose** — there are two batch sizes in play and they
multiply, not collide:

- `DP_BATCH_INGEST_CONCURRENCY=10` — items running in parallel inside the
  endpoint (one asyncio task each).
- TEI `--max-client-batch-size=32` — chunks per single HTTP POST to the
  BGE-M3 embedder (`TEI_EMBED_MAX_BATCH_AT`, §3.4).

For a 10-chunk document each item fires **one** HTTP POST carrying 10
inputs. For a 100-chunk document each item fires **four sequential** POSTs
(32+32+32+4). With concurrency 10, the TEI server sees up to 10 concurrent
client connections each carrying ≤32 inputs per call — so peak in-flight
embed payload is **~320 inputs across 10 sockets**, not 32 total. Same
shape applies to the sparse embedder when `search_mode=hybrid` (`SPARSE_EMBED_MAX_BATCH_AT=32`).

**Best case — `recursive` chunking + BGE-M3, no funding**

No OpenAI in the ingest path. Per item: chunk → TEI dense embed → Qdrant
upsert, ~1–2 s wall-clock. 10 parallel items finish in ~2 s.

| Metric | Value |
|---|---|
| Per-item cost | 1–2 s |
| 50-item batch | **~10–15 s** |
| Sustained RPM (back-to-back batches) | **200–300 items/min** |
| Binding constraint | TEI BGE-M3 server CPU/GPU |

**Average case — defaults (`contextual` + BGE-M3, no funding)**

One OpenAI chat call per item for context enrichment; embeddings stay on
TEI. Tier 1 (gpt-4o-mini 500 RPM / 200K TPM) has ~5–10× headroom.

| Metric | Value |
|---|---|
| Per-item cost | 6–15 s |
| 50-item batch | **~30–80 s** |
| Sustained RPM | **40–100 items/min** |
| OpenAI chat RPM consumed | ~40–100 RPM (well under Tier 1's 500) |
| Binding constraint | OpenAI contextual enricher |

**Heavy case — `contextual` + funding + OpenAI embeddings**

Three OpenAI chat calls per item (classifier ran on `/scrape` already, but
contextual + funding both fire on `/ingest`) plus one embed call. At
concurrency 10 you flood Tier 1's chat RPM/TPM and start seeing per-item
`429`s.

| Metric | Value |
|---|---|
| Per-item cost (no 429s) | 10–25 s |
| 50-item batch on Tier 1, conc=10 | **~3–6 min**, several items 429 |
| 50-item batch on Tier 1, conc=3 | **~5 min**, near-zero 429s |
| 50-item batch on Tier 4+, conc=10 | **~50–120 s**, clean |
| Binding constraint | OpenAI chat + embed RPM/TPM (your tier) |

**Edge cases**

| Scenario | Behavior |
|---|---|
| Single item with `content` > 120K chars (classify cap) / 60K (contextual cap) | Truncated inside the helper; item still succeeds, but enrichment quality drops on the trimmed tail. |
| Mixed batch (some 1-chunk docs, some 200-chunk docs) | Concurrency is per-item, not per-chunk — small items finish fast and pick up new ones from the queue while large ones run. Effective parallelism stays at the semaphore size. |
| OpenAI 429 mid-batch (no LiteLLM) | Failing items return `EMBEDDING_FAILED` (or the closest mapped code) in `results[]`. Batch keeps running. Top-level envelope still `success=true`. |
| OpenAI 429 mid-batch (LiteLLM configured) | Classifier/contextual/funding retry against `LITELLM_URL` transparently — most 429s are absorbed; batch sees few failures. |
| TEI BGE-M3 server unreachable | Every item fails with `EMBEDDING_FAILED`. Detect via `succeeded=0` in the response and back off. |
| Qdrant unreachable | Every item fails with `QDRANT_CONNECTION_FAILED`. Same signal — `succeeded=0`. |
| Two items share the same `source_id` | `IngestService` deletes prior vectors for that `source_id` before upsert (idempotent). Within one batch this means the **second** item wins; the first item's vectors are deleted by the second. Avoid duplicate source_ids in one batch. |
| Item with empty `content` | That item returns `VALIDATION_EMPTY_CONTENT` in `results[]`; others proceed. |
| Batch size > `DP_MAX_BATCH_INGEST_ITEMS` | Whole request rejected with `VALIDATION_BATCH_TOO_LARGE` before any item runs. No partial work. |
| Empty `items: []` | Whole request rejected with `VALIDATION_BATCH_EMPTY`. |
| `assistant_type="funding"` with `country` missing | Funding extractor runs without the country constraint — `state_or_province` results may not match the canonical list, get dropped. Per-item still `success=true`. |

**Recommendations**

| You're running… | Set `DP_BATCH_INGEST_CONCURRENCY` to | Why |
|---|---|---|
| Recursive + BGE-M3 (no OpenAI in ingest) | **20–30** | TEI is the only bottleneck; push it. |
| Contextual + BGE-M3, no funding (default) | **10** | Default — comfortable headroom on OpenAI Tier 1. |
| Contextual + OpenAI embeddings | **5** on Tier 1; **10–15** on Tier 2+ | Each item burns embed RPM too. |
| Contextual + funding + OpenAI embeddings | **3** on Tier 1; **8–10** on Tier 2+; **15+** if LiteLLM configured | 3 chat calls + 1 embed per item; needs the most headroom. |
| Any of the above with LiteLLM fallback wired | **2× the above** | 429s get absorbed by the fallback chain. |

Keep `DP_MAX_BATCH_INGEST_ITEMS` at 50 unless you have a specific reason to
go bigger — single-batch wall-clock above ~2 min hurts client-side timeout
budgets and makes partial-failure diagnosis annoying. For 1000 docs, send
20 batches of 50 sequentially rather than one batch of 1000.

---

## 6. Worked examples

Each example uses the **defaults** unless stated otherwise:
`embedding_model="bge_m3"`, `chunking.strategy="contextual"`,
`vector_config.search_mode="semantic"`, `assistant_type=null`,
`rate_limit_per_domain=10`, `rate_limit_window=60`.

### Example 1 — 100 URLs from a single website

Workload: scrape + ingest 100 pages from `https://www.wiener-neudorf.gv.at/*`.

**What happens, step by step:**

- All 100 URLs share the netloc `www.wiener-neudorf.gv.at` → one rate-limit
  bucket → **10 scrapes per 60 s allowed**.
- First 10 requests fire immediately. Requests 11–100 each block on
  `rate_limiter.acquire()` (`scraper_service.py:254`) and emit
  `rate_limit_waiting` log lines.
- Per-page ingest cost (contextual on, no funding): ~8–10 s wall-clock, but
  most of that overlaps the next scrape's wait — pipelined.

**Math:**
- Scrape-side cap: 100 URLs ÷ 10 RPM = **10 minutes minimum** just for the
  scrape limiter.
- Ingest-side cost runs concurrently with the limiter wait, so it doesn't
  extend the wall-clock unless your OpenAI tier saturates.

**Total:** **~10–12 minutes for 100 URLs**, bottlenecked by the per-domain
rate limit. Result: ~10 effective RPM end-to-end.

**How to go faster:** raise `RATE_LIMIT_PER_DOMAIN` if the target site can
take it, *or* warm the scrape cache for repeat ingests (cache hit bypasses
the limiter, `scraper_service.py:240-252`).

---

### Example 2 — 1000 URLs spread across 50 different websites (~20 URLs each)

**What happens:**

- Each of the 50 netlocs gets its own 10-per-60-s bucket.
- 20 URLs at one netloc finishes in ~2 minutes (20 ÷ 10 RPM).
- Across 50 origins, the scrape phase is no longer the bottleneck —
  the next ceiling is the **contextual enricher** at ~20–30 RPM total.

**Math:**
- Scrape ceiling: 50 × 10 = 500 RPM (theoretical) — not the binding constraint.
- Contextual enricher: ~25 RPM aggregate.
- 1000 URLs ÷ 25 RPM = **~40 minutes**.

**Total:** **~40–50 minutes for 1000 URLs**, bottlenecked by OpenAI contextual
enrichment.

**How to go faster:** switch `chunking.strategy="recursive"` (skip enrichment)
and you jump to ~50–100 RPM → 1000 URLs in **~10–20 minutes**, at the cost of
retrieval quality.

---

### Example 3 — Same 1000 URLs, but caller pre-classifies and uses `recursive` chunking

Caller already has `content_type` from a previous run, calls `/online/scrape`
once, then `/online/ingest` with:

```json
{ "chunking": { "strategy": "recursive" }, "content_type": ["funding"], ... }
```

This skips both the contextual enricher and (since `assistant_type` is null)
the funding extractor. The only remaining OpenAI call is the classifier on
the `/scrape` side, which is unavoidable.

**Math:**
- Per-request cost drops to ~1–2 s (classifier + embed + Qdrant upsert).
- TEI embedding (BGE-M3) at 32-item batches handles ~500–1000 chunks/min.
- For ~10-chunk docs: ~50 ingests/min embedding-side; classifier at
  ~50–100 RPM is comparable.

**Total:** **~10–20 minutes for 1000 URLs**, bottlenecked by classifier
OpenAI calls + TEI server.

---

### Example 4 — 1000 AT funding pages via `/ingest/at`

Workload: scrape transparenzportal.gv.at + 10 provincial portals (~11 origins),
~90 pages each, ingest into the AT Qdrant collections.

**Pipeline:**
1. `POST /online/scrape` per URL → classifier runs (1 OpenAI chat).
2. `POST /online/ingest/at` per URL — `assistant_type` is implicit funding,
   so the funding extractor *always* runs (2nd OpenAI chat).
3. Funding extractor runs in parallel with chunk → embed via
   `asyncio.create_task` (`ingest_at.py:260-264`).

**Math:**
- Scrape cap: 11 origins × 10 RPM = 110 RPM (not binding).
- Per ingest: classifier (1 call) + funding extractor (1 call) + contextual
  enricher (default) — three OpenAI hits per doc.
- Aggregate ceiling: **~15–25 RPM** for the full chain.
- 1000 URLs ÷ 20 RPM = **~50 minutes**.

**Total:** **~50–70 minutes for 1000 AT funding pages.**

**Watch out for:** OpenAI Tier 1 caps at 500 RPM combined; three calls per
doc means you hit the OpenAI ceiling at ~165 docs/min, well above the
contextual enricher's ~25 RPM cap. So contextual is still the bottleneck on
Tier 1; on Tier 5 the bottleneck shifts to your TEI deployment.

---

### Example 5 — Cache-warm re-ingest

You ingested 1000 pages an hour ago. Now you re-run the same ingest (say,
after tuning chunking config) within the 1-hour cache TTL.

**What changes:**
- `/online/scrape` returns from Redis cache before reaching the limiter
  (`scraper_service.py:240-252`). Scrape ceiling is now Redis throughput
  (thousands RPM, irrelevant).
- The bottleneck shifts entirely to the ingest path.

**Math:**
- With `strategy=contextual`: ~25 RPM × 1000 docs = **~40 minutes** —
  same as Example 2 because the limiter wasn't the bottleneck there.
- With `strategy=recursive`: ~80 RPM × 1000 docs = **~13 minutes**.

So scrape cache only helps when the per-domain limiter *was* the bottleneck
(Example 1's single-domain case).

---

### Example 6 — Driver script for measuring real RPM

Minimal asyncio driver to actually measure end-to-end RPM against a deployed
instance. Adjust `BASE_URL`, `API_KEY`, and the URL list.

```python
# scripts/measure_throughput.py
import asyncio
import time
import httpx

BASE_URL = "https://data.ki2.at"
API_KEY = "your-key"  # only if DP_ONLINE_API_KEYS is set
URLS = [
    "https://www.wiener-neudorf.gv.at/foerderungen",
    "https://www.example.gv.at/page1",
    # ... add 50–100 URLs
]
CONCURRENCY = 20  # client-side fanout; backend caps will apply

async def ingest_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> dict:
    async with sem:
        t0 = time.monotonic()
        # 1) scrape
        s = await client.post(
            f"{BASE_URL}/api/v1/online/scrape",
            json={"url": url, "markdown_type": "fit"},
            headers={"X-API-Key": API_KEY},
            timeout=120,
        )
        s.raise_for_status()
        scrape = s.json()["data"]
        # 2) ingest
        i = await client.post(
            f"{BASE_URL}/api/v1/online/ingest",
            json={
                "collection_name": "throughput-test",
                "source_id": f"test_{hash(url) & 0xffffffff}",
                "url": url,
                "content": scrape["content"],
                "content_type": scrape["content_type"],
                "entities": scrape.get("entities"),
                "metadata": {"municipality_id": "throughput-test"},
                "chunking": {"strategy": "contextual"},
            },
            headers={"X-API-Key": API_KEY},
            timeout=300,
        )
        i.raise_for_status()
        return {"url": url, "duration_s": time.monotonic() - t0, "ok": True}

async def main() -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    t0 = time.monotonic()
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(ingest_one(client, sem, u) for u in URLS),
            return_exceptions=True,
        )
    elapsed = time.monotonic() - t0
    ok = sum(1 for r in results if isinstance(r, dict) and r.get("ok"))
    print(f"{ok}/{len(URLS)} succeeded in {elapsed:.1f}s "
          f"-> {ok / (elapsed / 60):.1f} RPM")

if __name__ == "__main__":
    asyncio.run(main())
```

Run with `python scripts/measure_throughput.py`. The reported RPM is the
real-world ceiling for *your* deployment with *your* OpenAI tier and TEI
hardware — the tables above are derived ceilings, this is measured.

---

## 7. How to actually measure this

Estimates are not measurements. To produce real numbers:

1. Pick a workload (e.g. 100 URLs across 10 origins).
2. Run a small `asyncio.gather` driver against a deployed instance with the
   real OpenAI / TEI / Qdrant backing it.
3. Watch `data-plane` logs for `rate_limit_waiting` events
   (`rate_limiter.py:79`) — these reveal which origins are hitting the cap.
4. Watch OpenAI dashboard for 429 frequency.
5. Watch TEI server load.

The bottleneck order in practice is almost always:
**per-domain limit → OpenAI 429 → TEI saturation → Qdrant**.

---

## 8. Tuning knobs

All overridable via env vars (`DP_` prefix for `Settings`, no prefix for
`ExternalSettings`):

| Lever | Env var | Default | Effect |
|---|---|---|---|
| Per-domain rate limit | `RATE_LIMIT_PER_DOMAIN` | 10 | Raise to allow more scrapes per origin |
| Per-domain rate window | `RATE_LIMIT_WINDOW` | 60 s | Shorten for burstier scrape patterns |
| Scrape cache TTL | `DP_CACHE_TTL` | 3600 s | Longer cache = more cache hits = bypass limiter |
| Default scrape backend | `DEFAULT_SCRAPER` | `jina` | Picks which upstream RPM ceiling applies when the request body omits `scraper` (see 3.1.1) |
| OpenAI embed batch | `OPENAI_EMBED_MAX_BATCH` | 256 | Larger = fewer requests, more tokens per call |
| Contextual batch | `OPENAI_CONTEXTUAL_MAX_BATCH` | 32 | Larger = fewer calls, but risks `max_tokens` truncation |
| Contextual concurrency (fallback path) | constructor arg `max_concurrent` | 10 | In-process Semaphore size |
| TEI dense batch | `TEI_EMBED_MAX_BATCH_AT` | 32 | Must match TEI server's `--max-client-batch-size` |
| TEI sparse batch | `SPARSE_EMBED_MAX_BATCH_AT` | 32 | Same |
| Uvicorn workers | (add `--workers N` in `Dockerfile`) | 1 | Multi-process for CPU-bound work (rarely needed here — pipeline is I/O-bound) |
| LiteLLM fallback | `LITELLM_URL` + `LITELLM_API_KEY` | empty | Self-host classifier/enricher/funding when OpenAI throttles |
