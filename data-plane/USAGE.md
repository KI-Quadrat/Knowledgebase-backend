# Usage & Cost Tracking

How the data-plane counts tokens, credits, and pages — and turns them into USD.

Every billable third-party call ships back a `StageUsage` record. Routers
aggregate those records into a `UsageSummary` returned on the response and
persist one row per record into ClickHouse `usage_log` for historical
queries.

---

## 1. What gets counted — per provider

Counts come from the provider's own response, never an estimate.

| Provider | Stage(s) | Field captured | Where parsed |
|---|---|---|---|
| **Jina Reader** | `scraper` | `meta.usage.tokens` (top level) or `data.usage.tokens` | `crawl4ai_client._extract_jina_tokens` |
| **Firecrawl /v2/scrape** | `scraper` | `data.metadata.creditsUsed` / `creditsUsed` / `credits_used` | `crawl4ai_client._extract_firecrawl_credits` |
| **Firecrawl /v2/map** | `links_map` | same as above; falls back to `len(visited)` when absent | `crawl4ai_client._map_with_firecrawl` |
| **Crawl4AI** | `scraper` | none — self-hosted | hard-coded `cost_usd=0.0` |
| **Raw httpx fallback** | `scraper` | none | hard-coded `cost_usd=0.0` |
| **OpenAI chat** (classifier, contextual, funding) | `classifier` / `contextual` / `funding` | `usage.prompt_tokens`, `usage.completion_tokens`, `usage.prompt_tokens_details.cached_tokens` | `llm_classifier`, `contextual`, `funding_extractor` |
| **OpenAI embeddings** | `embedding` | `usage.total_tokens` (falls back to `prompt_tokens`) | `embedding/openai_client.py` |
| **BGE-M3 / TEI dense** | `embedding` | none — self-hosted | `bge_m3_client`, `tei_client_at` |
| **TEI sparse** | `sparse_embedding` | none — self-hosted | `tei_sparse_client_at` |
| **LlamaParse Cloud** | `inner_img` / `inner_docs` / `parse` | `pages_parsed` per parsed file | `routers/online/scrape.py`, `routers/online/parse.py` |
| **Cache hit** | `scraper` | none — no API call this request | hard-coded `cost_usd=0.0`, provider `"cache"` |

OpenAI-compatible alternatives (Nebius, Together, Groq, Fireworks,
DeepInfra) flow through the same `classifier` / `contextual` / `funding`
parsers — they return the same `usage` block, so the count fields fill in
automatically. Only the `cost_usd` computation depends on rates being set
for them (see §4).

---

## 2. The `StageUsage` record

One per external call. Fields:

```python
StageUsage(
    stage,              # "scraper" | "classifier" | "contextual" | "funding"
                        # | "embedding" | "sparse_embedding" | "inner_img"
                        # | "inner_docs" | "parse" | "links_map"
    provider,           # "jina" | "firecrawl" | "crawl4ai" | "openai" | ...
    model,              # "gpt-4o-mini", "text-embedding-3-small", or None
    prompt_tokens,      # chat input
    completion_tokens,  # chat output
    cached_tokens,      # subset of prompt_tokens billed at the cache rate
    embed_tokens,       # /v1/embeddings input
    scrape_tokens,      # Jina meta.usage.tokens
    credits,            # Firecrawl credits
    pages,              # LlamaParse pages
    cost_usd,           # see §4: number | 0.0 | None
)
```

Only the count fields relevant to the provider are populated; the rest stay
at zero.

---

## 3. The `UsageSummary` response field

Every billable endpoint returns a slim `usage` block — self-hosted stages
with zero counts and zero cost are dropped, and every zero-valued count
field is omitted per entry. The Python object retains every stage
internally so aggregation still works; only the JSON representation is
slim. ClickHouse `usage_log` always sees the full per-stage list.

```json
"usage": {
  "total_tokens": 4234,
  "total_cost_usd": 0.00079,
  "by_stage": {
    "scraper":    { "stage": "scraper",    "provider": "jina",   "scrape_tokens": 89,  "cost_usd": 0.000445 },
    "classifier": { "stage": "classifier", "provider": "openai", "model": "gpt-4o-mini",  "prompt_tokens": 3000, "completion_tokens": 200, "cost_usd": 0.000570 },
    "contextual": { "stage": "contextual", "provider": "openai", "model": "gpt-4.1-nano", "prompt_tokens": 945,  "completion_tokens": 200, "cost_usd": 0.0000945 }
  }
}
```

What gets included:
- `stage`, `provider`, and `cost_usd` are **always** present on every entry.
- `model` is included only when the provider has one (LLMs and OpenAI
  embeddings have models; Jina / Firecrawl do not).
- Count fields (`prompt_tokens`, `completion_tokens`, `cached_tokens`,
  `embed_tokens`, `scrape_tokens`, `credits`, `pages`) are included only
  when non-zero.

What gets dropped:
- Entire `by_stage` entries are dropped when `cost_usd == 0` AND all count
  fields are zero. Self-hosted (`bge_m3`, `tei_sparse`, `crawl4ai`,
  `httpx`) and `cache` provider entries land here. Entries with
  `cost_usd is None` (rate unset) are **kept** — that's the signal you're
  paying for something we can't price.
- `total_credits` and `total_pages` are dropped when zero; `total_tokens`
  and `total_cost_usd` are always emitted so consumers can rely on them.

Roll-up math:
- `total_tokens` sums `prompt + completion + embed + scrape` across every
  stage. (Cached tokens are not double-counted — they're already inside
  `prompt_tokens`.)
- `total_cost_usd` is the sum of per-stage `cost_usd` **unless** any stage
  reports `null`, in which case the total is `null` too.

### Where it appears

| Endpoint | Field |
|---|---|
| `POST /api/v1/online/scrape` | `data.usage` |
| `POST /api/v1/online/crawl` | `data.usage` (sums BFS scraper rows) |
| `POST /api/v1/online/document-parse` | `data.usage` |
| `POST /api/v1/online/document-parse/upload` | `data.usage` |
| `POST /api/v1/online/ingest` | `data.usage` |
| `POST /api/v1/online/ingest/at` | `data.usage` |
| `POST /api/v1/online/ingest/stream` | `data.usage` on the `completed` SSE event |
| `POST /api/v1/online/batch/ingest` | `data.results[i].data.usage` per item + `data.total_usage` for the batch |

---

## 4. How dollars are computed

Rates live in `pricing.yaml` at the data-plane root. `services/cost.py`
loads it once at startup and exposes one helper per provider:

```python
chat_cost(provider, model, prompt_tokens, completion_tokens, cached_tokens)
embed_cost(provider, model, tokens)
jina_cost(tokens)
firecrawl_cost(credits)
llamaparse_cost(pages)
```

### Chat (per 1M tokens)

```
uncached_prompt  = prompt_tokens - cached_tokens
cost_usd = (uncached_prompt    × input_per_1m
          + cached_tokens      × cached_input_per_1m
          + completion_tokens  × output_per_1m) / 1_000_000
```

If `cached_input_per_1m` isn't set for the model, the cached portion is
billed at the full `input_per_1m` rate (no discount, no error).

### Embeddings (per 1M tokens)

```
cost_usd = tokens × embed_per_1m / 1_000_000
```

### Jina

```
cost_usd = scrape_tokens × per_token
```

### Firecrawl

```
cost_usd = credits × per_credit
```

### LlamaParse

```
cost_usd = pages × per_page
```

### Three possible return values for `cost_usd`

| Value | Meaning |
|---|---|
| **a number** | Rate found in `pricing.yaml`; this is the computed dollar amount. |
| **`0.0`** | Provider not present in `pricing.yaml` at all — treated as self-hosted (BGE-M3, TEI sparse, Crawl4AI, httpx, cache, Qdrant, Redis). |
| **`null`** | Provider is listed but the rate field is `null` (plan-dependent). Raw count is still recorded. |

The `null` case is the operator alarm — see §6.

---

## 5. Editing `pricing.yaml`

Edit, commit, tag, deploy. The file is read once at container startup.

```yaml
openai:
  gpt-4o-mini:
    input_per_1m: 0.150
    output_per_1m: 0.600
    cached_input_per_1m: 0.075      # 50% off for gpt-4o-mini
  gpt-4.1-nano:
    input_per_1m: 0.100
    output_per_1m: 0.400
    cached_input_per_1m: 0.025      # 75% off for gpt-4.1-nano
  text-embedding-3-small:
    embed_per_1m: 0.020

jina:
  "*":                              # "*" = default for every Jina model
    per_token: 0.000005             # = $0.005 / 1K tokens

firecrawl:
  "*":
    # Standard: $83/month billed yearly, includes 100,000 credits/pages.
    # Effective included rate: 83 / 100000 = $0.00083 per credit.
    # Overage rate: $47 / 35000 = $0.001343 per extra credit.
    per_credit: 0.00083
    notes: "Standard plan effective included rate. Overage is $0.001343/credit."

llamaparse:
  "*":
    per_page: 0.003
```

Rules:
- **Provider absent** → treated as self-hosted → `cost_usd = 0.0`.
- **Provider present, model present** → that model's rate is used.
- **Provider present, model not listed** → falls back to `"*"`.
- **Rate field set to `null`** → `cost_usd = null` (count is still recorded).

Keep `pricing.yaml` in sync with `COST.md §2`.

---

## 6. ClickHouse `usage_log`

Every `StageUsage` produced by a request is also written to ClickHouse so
spend can be queried over time. Schema (DDL in `app/services/audit.py`,
applied via `scripts/migrate_usage_log.py`):

```sql
CREATE TABLE usage_log (
    event_time      DateTime DEFAULT now(),
    endpoint        LowCardinality(String),   -- "scrape" | "ingest" | ...
    stage           LowCardinality(String),   -- "scraper" | "classifier" | ...
    provider        LowCardinality(String),   -- "jina" | "openai" | ...
    model           LowCardinality(String),
    url             String,
    domain          String,
    prompt_tokens     UInt32,
    completion_tokens UInt32,
    cached_tokens     UInt32,
    embed_tokens      UInt32,
    scrape_tokens     UInt32,
    credits           Float64,
    pages             UInt32,
    cost_usd          Nullable(Float64),      -- null = rate unknown
    municipality_id   String,
    assistant_id      String,
    assistant_type    String,
    request_id        String,
    api_key_hash      String,
    status            LowCardinality(String)
) ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, endpoint, provider, model);
```

Writes are best-effort — if ClickHouse is down the request still succeeds
and the response still carries `usage`. The next ClickHouse query simply
won't see those rows.

### Useful queries

Daily spend by provider:

```sql
SELECT toDate(event_time) AS day, provider, sum(cost_usd) AS usd
FROM usage_log
WHERE event_time >= now() - INTERVAL 30 DAY
GROUP BY day, provider
ORDER BY day, provider;
```

Per-tenant cost this month:

```sql
SELECT municipality_id, sum(cost_usd) AS usd
FROM usage_log
WHERE event_time >= toStartOfMonth(now()) AND municipality_id != ''
GROUP BY municipality_id
ORDER BY usd DESC;
```

Most expensive URLs:

```sql
SELECT url, sum(cost_usd) AS usd, count() AS calls
FROM usage_log
WHERE event_time >= now() - INTERVAL 7 DAY AND url != ''
GROUP BY url
ORDER BY usd DESC
LIMIT 50;
```

### Operator alarm — find missing rates

If a provider/model pair starts appearing with `cost_usd IS NULL`, the
rate is missing from `pricing.yaml`. The raw count is still on the row, so
you can still bill, but you'll want to fix `pricing.yaml`:

```sql
SELECT provider, model, count() AS calls,
       sum(prompt_tokens + completion_tokens + embed_tokens + scrape_tokens) AS tokens,
       sum(credits) AS credits, sum(pages) AS pages
FROM usage_log
WHERE event_time >= now() - INTERVAL 1 DAY AND cost_usd IS NULL
GROUP BY provider, model
ORDER BY calls DESC;
```

---

## 7. What `0` actually means

A few rules worth internalizing:

- `cost_usd = 0.0` on a stage with non-zero token / credit / page counts
  means **self-hosted** (or `cache`) — by design. The token count is real
  work that happened on infrastructure you already pay for.
- `cost_usd = null` on a stage with non-zero counts means **the rate was
  not set**. Fix `pricing.yaml`.
- `total_cost_usd = null` on `UsageSummary` means **at least one stage's
  cost was unknown** — never silently treat the unknown stages as zero in
  the rollup.

---

## 8. Adding a new provider

1. Wire the response parsing in the client (`services/scraping/*` or
   `services/intelligence/*` or `services/embedding/*`). Build a
   `StageUsage(stage=..., provider=..., model=..., <counts>=..., cost_usd=cost.<helper>(...))`.
2. Add the provider's rates under a new top-level key in `pricing.yaml`:
   ```yaml
   newprovider:
     "*":
       input_per_1m: 0.50
       output_per_1m: 1.00
   ```
3. If the provider's billing unit isn't tokens/credits/pages, add a new
   helper in `services/cost.py` and a matching field on `StageUsage` in
   `app/models/common.py`. Then add a `<provider>_<unit>` column in
   `usage_log` if you want it queryable.
4. Unit test the parser with a canned response (see `tests/test_usage.py`).

---

## 9. Where it lives in code

| Concern | File |
|---|---|
| Pricing table | `pricing.yaml` |
| Cost math | `app/services/cost.py` |
| `StageUsage`/`UsageSummary` models | `app/models/common.py` |
| Scrape parsers | `app/services/scraping/crawl4ai_client.py` |
| Classifier usage | `app/services/intelligence/llm_classifier.py` |
| Contextual usage (per-window aggregation) | `app/services/intelligence/contextual.py` |
| Funding extractor usage | `app/services/intelligence/funding_extractor.py` |
| Embedding usage | `app/services/embedding/openai_client.py`, `bge_m3_client.py`, `tei_client_at.py`, `tei_sparse_client_at.py` |
| Ingest aggregation | `app/services/ingest/ingest_service.py` |
| Routers (response + ClickHouse) | `app/routers/online/{scrape,parse,ingest,ingest_at,ingest_stream}.py` |
| ClickHouse table + DDL | `app/services/audit.py` (DDL constant + `log_usage` writer) |
| CH migration script | `scripts/migrate_usage_log.py` |
| Tests | `tests/test_cost.py`, `tests/test_usage.py` |
