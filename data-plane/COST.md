# Cost Reference — Online Ingestion Pipeline

Per-URL cost surfaces for the **online ingestion path** (`POST /api/v1/online/scrape`,
`/online/ingest`, `/online/ingest/at`, `/online/batch/ingest`). End-to-end:
**scrape → classify → chunk → contextual enrich → embed → upsert**.

All file:line references are relative to `data-plane/`. Provider list prices
shown are as of **May 2026** and change frequently — always reconcile against
your invoice. Self-hosted services (Qdrant, Redis, the TEI dense/sparse servers)
are **not billed externally** and are therefore
**excluded from the cost tables** — their cost is whatever hardware/colo
already pays for them.

---

## 0. What the project currently runs

The defaults that ship today (post `monolith-dev` branch, commit `59bfbdd`):

| Stage | Currently using | Notes |
|---|---|---|
| **Scrape (primary)** | Jina Reader (`https://eu-r-beta.jina.ai`) | `DEFAULT_SCRAPER=jina` (`config.py:70`); EU endpoint for data residency |
| **Scrape (crawl/BFS)** | httpx | `DEFAULT_CRAWLER=httpx`; free, no JS |
| **Scrape (fallback / opt-in)** | Firecrawl | automatic fallback after Jina; also opt-in primary via `scraper="firecrawl"` |
| **Classifier** | OpenAI **gpt-4o-mini** (`DP_CLASSIFIER_MODEL=gpt-4o-mini`) | always-on after scrape (`config.py:139`) |
| **Chunking strategy** | `contextual` (default) | `routers/online/ingest.py` request schema |
| **Contextual enrichment** | OpenAI **gpt-4.1-nano** (`DP_CONTEXTUAL_MODEL=gpt-4.1-nano`), batched 32 chunks/call, **strict `json_schema`** | `contextual.py:181-209`; see §5.3 |
| **Funding extractor** | OpenAI **gpt-4.1-nano** (`DP_FUNDING_MODEL=gpt-4.1-nano`) | only when `assistant_type="funding"` |
| **Dense embedding (default)** | **BGE-M3** via self-hosted TEI at `embed.ki2.at` | `tei_embed_url_at` (`config.py:172`); made default in commit `7f49e1b` |
| **Dense embedding (opt-in)** | OpenAI `text-embedding-3-small` | only when caller sets `embedding_model="openai"` |
| **Sparse embedding** | self-hosted TEI sparse at `sparse.ki2.at` | only when `search_mode="hybrid"` (`config.py:186`) |
| **Inner-doc / inner-image OCR** | LlamaParse Cloud (if `LLAMA_CLOUD_API_KEY` set) | falls back to local unstructured parser at $0 |
| **Per-task model routing** | `llm_router.py` resolves `<provider>/<model>` per task | OpenAI default; any OpenAI-compatible provider configurable per task — see §5.5 |
| **Vector store** | Qdrant — self-hosted | $0 external |
| **Cache / rate-limit state** | Redis — self-hosted | $0 external; scrape cache TTL 3600s |

The only **always-billed** services on this stack are **Jina** (scrape) and
**OpenAI** chat (classifier on `gpt-4o-mini` + contextual on `gpt-4.1-nano`).
Everything else is either self-hosted or opt-in.

---

## 1. Cost surfaces at a glance

Every line below is touched on the online ingestion path. The right column is
the unit the provider bills on — multiply by your volume to estimate spend.

| # | Service | Provider | Default-on? | Billing unit | Where called |
|---|---|---|---|---|---|
| 1 | Web scrape (primary) | **Jina Reader** | ✅ yes | per request (token-metered) | `config.py`; `services/scraping/scraper_client.py` |
| 2 | Web scrape (fallback / opt-in) | **Firecrawl** | ❌ fallback after Jina; opt-in via `scraper="firecrawl"` | per request (plan tier) | `config.py`; `services/scraping/scraper_client.py` |
| 3 | Content classifier | **OpenAI gpt-4o-mini** | ✅ yes (always runs post-scrape) | per token (in + out) | `services/intelligence/llm_classifier.py`; `routers/online/scrape.py` |
| 4 | Contextual chunk enrich | **OpenAI gpt-4.1-nano** | ✅ yes (default `chunking.strategy="contextual"`) | per token (in + out, cached input 75% off) | `services/intelligence/contextual.py:150-209`; batch cap `config.py:148` |
| 5 | Funding metadata extract | **OpenAI gpt-4.1-nano** | ❌ only when `assistant_type="funding"` | per token (in + out) | `services/intelligence/funding_extractor.py` |
| 6 | Dense embedding (default) | **BGE-M3 via self-hosted TEI** | ✅ yes | self-hosted | `config.py:172`; `services/embedding/tei_client_at.py` |
| 7 | Dense embedding (alt) | **OpenAI text-embedding-3-small** | ❌ opt-in (`embedding_model="openai"`) | per token | `services/embedding/openai_client.py:13-14`; batch cap `config.py:143` |
| 8 | Sparse embedding (hybrid) | **TEI sparse self-hosted** | ❌ opt-in (`search_mode="hybrid"`) | self-hosted | `config.py:186`; `services/embedding/tei_sparse_client_at.py` |
| 9 | Inner-document OCR/parse | **LlamaParse Cloud** | ❌ only when `inner_img=true` or `inner_docs=true` | per page | `config.py:95-96`; `services/parsing/parsers/llama_parser.py` |
| 10 | Vector DB upsert | Qdrant | ✅ yes — **self-hosted** | — | `config.py:105-120` |
| 11 | Cache + rate-limit state | Redis | ✅ yes — **self-hosted** | — | `config.py:123` |

---

## 2. Provider list prices (May 2026 snapshot)

| Provider / model | Input | Output | Notes |
|---|---|---|---|
| **OpenAI gpt-4o-mini** (classifier) | $0.150 / 1M tokens | $0.600 / 1M tokens | Cached input $0.075 / 1M (50% off); Batch API −50% on both |
| **OpenAI gpt-4.1-nano** (contextual + funding) | $0.100 / 1M tokens | $0.400 / 1M tokens | Cached input $0.025 / 1M (**75% off** — better discount than gpt-4o-mini); Batch API −50% on both |
| **OpenAI text-embedding-3-small** | $0.020 / 1M tokens | — | 1536 dims; dense embed only |
| **Jina Reader** | token-metered; ~$0.002–0.01 / page | — | Free key includes 10M tokens; rate-tier caps apply |
| **Firecrawl** | plan-based RPM caps | — | Free 10 RPM → Scale 7500 RPM; price scales with tier |
| **LlamaParse** | per-page | — | Free tier ~1k pages/day; paid plans per-page |
| **Any OpenAI-compatible provider** (Nebius, Together, Groq, Fireworks, DeepInfra, self-hosted vLLM) | provider-specific | provider-specific | Configurable per task — see §5.5 |

#### Automatic prompt caching — when it actually helps

OpenAI auto-caches prompt prefixes ≥**1024 tokens**. When two consecutive
calls (within ~5 minutes) share an identical prefix of that length, the
shared portion is billed at the cached rate above (50% off for gpt-4o-mini,
75% off for gpt-4.1-nano). Caching is automatic, free, and requires no SDK
change — `contextual.py:165-170` already orders the prompt as
`system → <document> → chunks`, which puts the cacheable part first.

**When the cache fires on this pipeline:**
- **Large documents (>32 chunks → 2+ contextual windows).** Window 1 writes
  the doc-prefix; windows 2+ read it at 25% of input cost (gpt-4.1-nano).
  This is the main case. At 10 chunks/URL (the default-path assumption)
  there's only 1 window per URL → no in-URL benefit.
- **Re-ingest of the same URL within 5 min.** Full cache hit on the doc
  portion. Rare in practice — the Redis scrape cache (TTL 3600s) usually
  short-circuits the whole pipeline before contextual runs.
- **Bulk batch where the same SCRAPED CONTENT appears repeatedly.** Possible
  for re-runs of failed batches.

**When it does NOT help:**
- **Cold per-URL ingest** (different doc each call). The cacheable prefix
  between calls is just the system prompt (~150 tokens) — below the
  1024-token threshold. **Auto-caching contributes ~0% in this case** and
  the headline numbers below assume this.
- **Classifier and funding extractor.** Each URL is a different document;
  no shared prefix worth caching.

`contextual.py:_chat_with_fallback` now logs `cached_tokens` on every chat
response (`contextual_batch_usage` in your structured logs). Watch this
field to verify the cache is firing on your specific workload.

Always verify against the provider's current pricing page before budgeting.

---

## 3. Cost flow — how spend accumulates per URL

This is the running-total view of one ingest call on the **default path**
(`bge_m3` embed, contextual chunking, no funding, no inner docs). Same
assumptions as §4: 3 KB scraped markdown ≈ 3K input tokens, 10 chunks/page,
contextual output ~160 tokens/chunk.

```
 URL in
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ STAGE 1 — SCRAPE                                               │
│ Jina Reader  (config.py:79)                                    │
│   • 1 HTTP request, token-metered                              │
│   • Redis cache hit (TTL 3600s) → skip, $0                     │
│ Step cost:    ~$0.003                                          │
│ Running:      $0.003                                           │
└───────────────────────────────────────────────────────────────┘
    │ markdown
    ▼
┌───────────────────────────────────────────────────────────────┐
│ STAGE 2 — CLASSIFY                                             │
│ OpenAI gpt-4o-mini  (llm_classifier.py)                        │
│   • 1 chat call, ~3,000 in / ~200 out tokens                   │
│   • Input truncated at 120K chars (config.py:152)              │
│ Step cost:    ~$0.00057                                        │
│ Running:      $0.00357                                         │
└───────────────────────────────────────────────────────────────┘
    │ + content_type, entities
    ▼
┌───────────────────────────────────────────────────────────────┐
│ STAGE 3 — CHUNK                                                │
│ Recursive splitter, in-process, no API                         │
│   • chunk_size=512, overlap=50 (config.py:37-38)               │
│ Step cost:    $0                                               │
│ Running:      $0.00357                                         │
└───────────────────────────────────────────────────────────────┘
    │ 10 chunks
    ▼
┌───────────────────────────────────────────────────────────────┐
│ STAGE 4 — CONTEXTUAL ENRICH                                    │
│ OpenAI gpt-4.1-nano, batched  (contextual.py:150-209)          │
│   • 10 chunks fit in one 32-chunk batched call                 │
│   • ~3,500 in / ~1,600 out tokens                              │
│   • strict json_schema mode — no per-chunk fallback (§5.3)     │
│   • auto-cache: 0% benefit at 10 chunks/URL (§2 note)          │
│ Step cost:    ~$0.00099                                        │
│ Running:      $0.00456                                         │
└───────────────────────────────────────────────────────────────┘
    │ context-prefixed chunks
    ▼
┌───────────────────────────────────────────────────────────────┐
│ STAGE 5 — EMBED                                                │
│ BGE-M3 via self-hosted TEI  (tei_client_at.py)                 │
│   • 1 HTTP request, ≤32 chunks/batch (config.py:179)           │
│   • optional parallel sparse embed if search_mode="hybrid"     │
│ Step cost:    $0                                               │
│ Running:      $0.00456                                         │
└───────────────────────────────────────────────────────────────┘
    │ 1024-d dense vector (+ optional sparse)
    ▼
┌───────────────────────────────────────────────────────────────┐
│ STAGE 6 — UPSERT                                               │
│ Qdrant — self-hosted                                           │
│   • 1 PUT, all chunks in one request                           │
│ Step cost:    $0                                               │
│ Running:      $0.00456                                         │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
 Ingested. ~$0.0046 per URL  →  ~$4.56 per 1,000 URLs
```

**Reading the totals.** The two OpenAI calls combined (~$0.00156, classifier
on gpt-4o-mini + contextual on gpt-4.1-nano) are smaller than the single
Jina scrape (~$0.003), so on the default stack **scraping is the dominant
cost, not the LLM** — even more so now that contextual runs on the cheaper
nano. That flips as soon as you enable funding extraction.

**Branches that move the running total:**

- **Cache hit on Stage 1** → stages 1+2 both skipped (the classifier reads
  from cached scrape envelope), drops cost to **~$0.001 / URL** (just the
  contextual call survives).
- **`assistant_type="funding"`** → inserts a gpt-4.1-nano call after Stage 2
  (+~$0.00042, +9%).
- **`embedding_model="openai"`** → Stage 5 becomes ~$0.0001 (still
  negligible).
- **`inner_docs=true` w/ LlamaParse** → adds an out-of-band per-page
  charge between Stage 1 and Stage 2.
- **Large doc (>32 chunks)** → contextual splits into multiple windows;
  windows 2+ get auto-cache savings on the doc prefix (75% off cached
  input). At 64 chunks (2 windows) saves ~$0.00023 / URL on the
  contextual line; at 96 chunks (3 windows) saves ~$0.00045 / URL.

---

## 4. What a single URL actually costs

Assumptions used in the worked totals:

- **1 URL = 1 scraped page** (no inner docs/images).
- Page content ≈ 12 KB markdown ≈ **3,000 input tokens** post-scrape.
- Average **10 chunks/page** at the default `chunk_size=512`, `overlap=50`
  (`config.py:37-38`).
- Contextual enrich emits ~**160 output tokens / chunk** budgeted at
  `max_tokens = min(16000, 200 + 160 * len(chunks))` (`contextual.py:174`).
- Classifier truncates input at **120K chars ≈ 30K tokens** worst case
  (`config.py:152`); typical pages are far smaller.
- Embedding inputs ≈ **500 tokens / chunk** (chunk text + context prefix).

### 4.1 Default path — `bge_m3` embed, contextual chunking, no funding

Current model split: classifier on **gpt-4o-mini**, contextual on
**gpt-4.1-nano**. Auto-cache benefit at 10 chunks/URL = 0% (only one
window per URL → no prefix reuse).

| Step | Calls | Tokens (in / out) | Unit cost | Cost / URL |
|---|---|---|---|---|
| Jina scrape | 1 req | — | ~$0.002–0.005 / page | **~$0.002–0.005** |
| OpenAI classify (gpt-4o-mini) | 1 chat | 3,000 in / 200 out | $0.150 / $0.600 per 1M | **~$0.00057** |
| OpenAI contextual enrich (gpt-4.1-nano, batched 32) | 1 batched chat | 3,500 in / 1,600 out | $0.100 / $0.400 per 1M | **~$0.00099** |
| TEI dense embed (BGE-M3) | ~1 req (10 chunks ≤ batch 32) | — | self-hosted | **$0** |
| Qdrant upsert | 1 write | — | self-hosted | **$0** |
| | | | **Per URL total:** | **~$0.0036–0.0066** |
| | | | **Per 1,000 URLs:** | **~$3.60–$6.60** |

Jina still dominates. The two OpenAI calls together add **~$0.00156 / URL
≈ $1.56 / 1k URLs** (down from ~$2.10 / 1k when both were on gpt-4o-mini).

### 4.2 With funding extraction (`assistant_type="funding"`)

Adds one more gpt-4.1-nano call (same 120K-char input ceiling as the
classifier, `config.py:153`).

| Step | Extra calls | Extra tokens | Extra cost / URL |
|---|---|---|---|
| OpenAI funding extract (gpt-4.1-nano) | 1 chat | 3,000 in / 300 out | **~$0.00042** |
| | | **+ per URL:** | **~$0.0004** |
| | | **+ per 1k URLs:** | **~$0.42** |

### 4.3 With OpenAI embeddings (`embedding_model="openai"`)

Replaces self-hosted TEI with the OpenAI embeddings endpoint
(`services/embedding/openai_client.py`).

| Step | Calls | Tokens | Unit cost | Cost / URL |
|---|---|---|---|---|
| OpenAI embed (text-embedding-3-small, batched up to 256 inputs, `config.py:143`) | 1 req for ≤256 chunks | 5,000 in | $0.020 / 1M | **~$0.0001** |
| | | | **+ per 1k URLs:** | **~$0.10** |

Embeddings are **negligible** compared to chat. Switching from BGE-M3 to OpenAI
adds < $0.20 / 1k URLs.

### 4.4 With auto-cache benefit (large docs only)

When a single URL has **>32 chunks**, contextual enrichment splits into
multiple windows. Window 1 writes the doc prefix to OpenAI's cache;
windows 2+ read it back at 25% of input cost (gpt-4.1-nano cached rate).

| Chunks / URL | Windows | Doc tokens cached | Cache savings / URL |
|---|---|---|---|
| ≤ 32 | 1 | 0 | $0 |
| 33–64 | 2 | ~3,000 (window 2) | **~$0.00023** |
| 65–96 | 3 | ~6,000 (windows 2+3) | **~$0.00045** |
| 129–160 | 5 | ~12,000 (windows 2–5) | **~$0.0009** |

Math: each cached window reads ~3,000 doc tokens at `$0.025/1M` instead
of `$0.100/1M` → saves ~$0.000225 per cached window. Net effect on the
contextual line is **−~23% per cached window** on the doc portion only.

### 4.4 With inner-doc / inner-image parsing

When the caller sets `inner_img=true` or `inner_docs=true` on `/scrape`, the
pipeline routes the attached files through LlamaParse if
`LLAMA_CLOUD_API_KEY` is set (`config.py:95`); otherwise it uses the local
unstructured parser at no external cost.

| Asset volume per URL | LlamaParse pages | Cost / URL (paid tier) |
|---|---|---|
| Typical news/blog page | 0 | $0 |
| Funding page w/ 1 PDF guideline (~5 pages) | 5 | per LlamaParse list price |
| Grant portal w/ 3 PDFs (~30 pages) | 30 | per LlamaParse list price |

Free tier covers ~1,000 pages/day — adequate for low-volume but a hard cap for
bulk ingest. Disable by leaving `LLAMA_CLOUD_API_KEY=""` or omitting
`inner_docs=true`.

---

## 5. Cost levers (how to spend less per URL)

### 5.1 Scrape cache

`POST /scrape` caches results in Redis for `cache_ttl = 3600 s` (`config.py:31`).
A re-scrape inside the TTL returns from cache and **skips Jina + classifier
+ funding extract entirely** (`services/scraping/scraper_service.py:240-252`).
Repeat ingests of the same URL within an hour cost **$0** on the API side.

### 5.2 Chunking strategy

`chunking.strategy="recursive"` (or `sentence` / `fixed`) skips contextual
enrich entirely → saves the **largest per-URL line item** (~$1.50 / 1k URLs).
Trade-off: lower retrieval quality.

### 5.3 Contextual-enrichment fix — 10× cost cut on the affected docs

**TL;DR:** `app/services/intelligence/contextual.py` was migrated from
`response_format: "json_object"` to `response_format: "json_schema"` with
`strict: true` and `minItems` / `maxItems` pinned to the chunk count
(`contextual.py:181-209`, commit `59bfbdd`, resolved 2026-05-12). For
documents that previously hit the per-chunk fallback, this is a **~10× cost
reduction on the contextual stage**.

**What was happening before.** The batched contextual call (up to 32 chunks
in one chat completion) ran under `json_object` mode at `temperature=0`.
The model did not reliably honor the "array length must equal N" prompt
instruction — production logs on 2026-05-12 showed mismatch deltas ranging
from **−6 to +3** chunks, with a **~40% fallback rate** across ~9
concurrent documents (≥12 visible mismatches in ~27 windows).

Each fallback re-ran the enrichment **one chat call per chunk**. With our
10-chunks/page average:

| Path | Chat calls / doc | Relative spend |
|---|---|---|
| Batched (happy path) | 1 | 1× |
| Per-chunk fallback | 10 | **10×** |

At a 40% population-wide fallback rate the **average** contextual spend was
`0.6 × 1× + 0.4 × 10× ≈ 4.6×` the batched cost — roughly doubling total
per-URL spend on the default pipeline.

**What changed.** Strict `json_schema` mode pushes the length constraint into
OpenAI's constrained sampler at decode time, not into the prompt. The model
structurally cannot return the wrong number of contexts. Defensive guards
remain: a length-mismatch check still fires if a non-strict provider is
configured via the router (`contextual.py:202-216`), and a `refusal`-field
check returns `None` cleanly when strict mode refuses to generate
(`contextual.py:246-260`), which then triggers the per-chunk fallback as
intended for genuine refusals.

**Post-fix expectation.** `contextual_batch_length_mismatch` warnings
should be essentially zero. If they reappear, check whether the configured
provider (via `DP_CONTEXTUAL_MODEL`) doesn't honor strict `json_schema` —
the warning has a defensive guard there.

**Known similar pattern still using `json_object`:**
`app/services/intelligence/llm_classifier.py` — same code shape as the
pre-fix contextual path. It has produced `Unterminated string` JSON-parse
errors. Not yet migrated; worth doing if classifier reliability becomes a
cost issue (each parse failure currently triggers a retry → 2× classifier
spend on that document).

### 5.4 Skip funding extraction when not needed

Funding extract only runs when the caller passes `assistant_type="funding"`
(`routers/online/ingest.py:128-138`). For non-funding tenants this is **already
$0** — make sure clients don't pass the flag reflexively.

### 5.5 Per-task model routing (any OpenAI-compatible provider)

The three intelligence tasks (classifier / contextual / funding) each pick
their own model via env, in `"<provider>/<model_id>"` format:

- `DP_CLASSIFIER_MODEL`
- `DP_CONTEXTUAL_MODEL`
- `DP_FUNDING_MODEL`

Empty → falls back to `DP_OPENAI_MODEL` (the global default). A bare model
name (no slash) is treated as the `openai` provider for backwards
compatibility with the legacy `OPENAI_MODEL=gpt-4o-mini` shape.

Routing is handled by `app/services/intelligence/llm_router.py`, which
ships with public endpoints for six built-in OpenAI-compatible providers:
`openai`, `nebius`, `together`, `groq`, `fireworks`, `deepinfra`. To use
one, set its API key (e.g. `DP_NEBIUS_API_KEY=...`) and reference it in a
task model spec (e.g. `DP_CONTEXTUAL_MODEL=nebius/Qwen/Qwen2.5-72B-Instruct`).
Override its base URL only when pointing at a self-hosted endpoint (vLLM
behind your VPC, an Azure deployment, etc.).

**Cost lever:** today the contextual enricher runs on **gpt-4.1-nano**
(~$0.99 / 1k URLs at default-path token usage) — already ~33% cheaper
than the prior gpt-4o-mini config (~$1.49 / 1k URLs) plus a deeper
auto-cache discount (75% vs 50% off cached input). To go further, point
`DP_CONTEXTUAL_MODEL` at a hosted Qwen on Nebius/DeepInfra (~3–10×
cheaper). Keep gpt-4o-mini on the classifier (where output quality
matters more and the cost is small) and consider moving funding to a
slightly larger model if structured-extraction quality suffers on nano.

**Caveat:** the contextual fix in §5.3 depends on `response_format:
json_schema` strict mode. OpenAI supports it. Among the alternatives,
support varies by model — verify on a single document before rolling out,
or the §5.3 10× fallback returns.

### 5.6 Pick the cheapest scraper for the workload

| Backend | Cost / page (typical) | Best for |
|---|---|---|
| **Jina Reader** (default) | ~$0.002–0.005 | JS-heavy pages; reliable markdown |
| **Firecrawl** | per-plan tier | bulk discovery (`/v2/map`); automatic fallback after Jina |
| **Raw httpx** (final fallback) | $0 | static HTML; no JS rendering needed |

Setting `DP_DEFAULT_SCRAPER=firecrawl` makes Firecrawl the primary backend;
otherwise it is used automatically when the Jina primary fails.

---

## 6. Volume-based budgeting cheat sheet

Default path (`bge_m3` + contextual + no funding + no inner docs), Jina at
$0.003/page midpoint:

| URLs / month | Jina | OpenAI (classify gpt-4o-mini + enrich gpt-4.1-nano) | Total external |
|---|---|---|---|
| 1,000 | ~$3 | ~$1.60 | **~$4.60** |
| 10,000 | ~$30 | ~$15.60 | **~$45.60** |
| 100,000 | ~$300 | ~$156 | **~$456** |
| 1,000,000 | ~$3,000 | ~$1,560 | **~$4,560** |

Add **~9%** for funding ingests (`assistant_type="funding"` adds one
gpt-4.1-nano call/URL).
Add **LlamaParse separately** based on attached PDF/image volume.
Subtract the **cache hit rate** (Redis TTL 3600s) from the Jina + classifier
columns directly — those calls are skipped entirely on cache hit.
OpenAI auto-cache on contextual: assume **0% benefit** at typical 10
chunks/URL (only fires when a single URL has >32 chunks → see §4.4).

### 6.1 Small-batch estimation — 10 / 100 / 500 URLs

For one-off submits and dev loops. All numbers use the same default-path
assumptions as §4.1 (Jina midpoint $0.003/page, **classify on gpt-4o-mini**,
**contextual on gpt-4.1-nano**, 10 chunks/URL → auto-cache contributes 0%).
The "inner PDF" columns assume **one extra single-page PDF per URL** routed
through LlamaParse — that PDF's text flows downstream through the same
classifier + contextual call as the page itself (no additional OpenAI calls
per attachment), so the only added external charge is the LlamaParse per-page
fee.

**LlamaParse mode reference** (May 2026 list prices, 1,000 credits = $1.25):

| Mode | Credits / page | $ / page |
|---|---|---|
| Fast | 1 | $0.00125 |
| Cost-effective (default) | 3 | $0.00375 |
| Agentic | 10 | $0.0125 |
| Agentic Plus | 45 | $0.05625 |

**Free tier:** 10,000 credits on signup + ~7,000 standard-mode pages/week.
The volumes below sit well within free tier — the **paid** columns only
apply once you exhaust signup credits and the weekly allotment, or if
`LLAMA_CLOUD_API_KEY` points at a paid org.

| URLs | Default (no PDF) | + 1 PDF/URL (Fast) | + 1 PDF/URL (Cost-effective) | + 1 PDF/URL (Agentic) |
|---|---|---|---|---|
| **10** | ~$0.05 | ~$0.06 | ~$0.08 | ~$0.17 |
| **100** | ~$0.46 | ~$0.58 | ~$0.83 | ~$1.71 |
| **500** | ~$2.28 | ~$2.91 | ~$4.16 | ~$8.53 |

**Within free tier** (most realistic case at these volumes — LlamaParse is $0):

| URLs | Default | + 1 PDF/URL (free tier) |
|---|---|---|
| **10** | ~$0.05 | ~$0.05 |
| **100** | ~$0.46 | ~$0.46 |
| **500** | ~$2.28 | ~$2.28 |

**What sits behind each row at 500 URLs (default, no PDF):**

| Line item | Subtotal |
|---|---|
| Jina scrape (500 reqs × ~$0.003) | ~$1.50 |
| OpenAI classify gpt-4o-mini (500 × ~$0.00057) | ~$0.29 |
| OpenAI contextual gpt-4.1-nano (500 × ~$0.00099) | ~$0.50 |
| BGE-M3 / Qdrant / Redis | $0 |
| **Total** | **~$2.29** |

**Caveats:**
- Cache hits (TTL 3600s) drop both Jina and classifier columns for that URL
  to $0 — re-running the same 500 URLs within an hour costs roughly the
  cost of contextual enrich only (~$0.50).
- A page with multiple PDFs or a multi-page PDF scales linearly with
  LlamaParse page count: replace "1 PDF page" with "N pages" and multiply
  the LlamaParse column.
- Auto-cache savings on contextual are ignored in the totals above — they
  only kick in for documents >32 chunks (§4.4).

#### Scraper-backend impact (10 / 100 / 500 URLs)

The "Default" row above assumes Jina at $0.003/page. Swapping the scraper
changes only the scrape line — classifier (gpt-4o-mini) + contextual
(gpt-4.1-nano) stay constant (~$0.00156/URL combined).

| Scraper | $ / page (effective) | 10 URLs | 100 URLs | 500 URLs | Notes |
|---|---|---|---|---|---|
| **Jina Reader** (default) | ~$0.003 (within paid tier) | ~$0.05 | ~$0.46 | ~$2.28 | New keys ship with 10M free tokens → first runs free |
| **Firecrawl** — free tier | $0 (≤500 lifetime, then ≤1,000/mo free) | ~$0.02 | ~$0.16 | ~$0.78 | Only OpenAI stages billed |
| **Firecrawl** — Hobby plan | ~$0.0053/page amortized ($16/mo ÷ 3k credits) | ~$0.07 | ~$0.69 | ~$3.43 | Basic scrape = 1 credit; enabling JSON+enhanced mode jumps to 9 credits/page (~$0.048/page) |
| **Firecrawl** — Standard plan | ~$0.00083/page amortized ($83/mo ÷ 100k credits) | ~$0.02 | ~$0.24 | ~$1.20 | Most cost-effective at this volume if you also use other Firecrawl features |
| **Raw httpx** (final fallback) | $0 | ~$0.02 | ~$0.16 | ~$0.78 | Static HTML only — no JS rendering |

The Firecrawl plan tiers are *monthly subscriptions*, so the "amortized
$/page" only makes sense if you use the full plan allotment. For one-off
500-URL submits you'd want the free tier or pay-as-you-go.

#### Optional add-ons (delta vs. default path)

These stack on top of any scraper choice. Add to the totals above.

| Add-on | $ / URL extra | 10 URLs | 100 URLs | 500 URLs | When it fires |
|---|---|---|---|---|---|
| **Funding extraction** (gpt-4.1-nano) | +$0.00042 | +$0.004 | +$0.04 | +$0.21 | `assistant_type="funding"` |
| **OpenAI dense embed** (text-embedding-3-small) | +$0.0001 | +$0.001 | +$0.01 | +$0.05 | `embedding_model="openai"` |
| **Sparse embed (hybrid)** | $0 (self-hosted TEI) | $0 | $0 | $0 | `search_mode="hybrid"` |
| **LlamaParse — 1 PDF page/URL, Fast** | +$0.00125 | +$0.013 | +$0.13 | +$0.63 | `inner_docs=true`, paid tier |
| **LlamaParse — 1 PDF page/URL, Cost-effective** | +$0.00375 | +$0.038 | +$0.38 | +$1.88 | `inner_docs=true`, paid tier |
| **LlamaParse — 1 PDF page/URL, Agentic** | +$0.0125 | +$0.13 | +$1.25 | +$6.25 | `inner_docs=true`, paid tier |

#### Worst-case all-paid example (500 URLs)

If you stack every paid option — Firecrawl Hobby + funding + OpenAI embed
+ LlamaParse Agentic-Plus on 1 PDF/URL:

| Line | $ |
|---|---|
| Firecrawl Hobby (amortized 500 × $0.0053) | ~$2.65 |
| OpenAI classify gpt-4o-mini | ~$0.29 |
| OpenAI contextual gpt-4.1-nano | ~$0.50 |
| OpenAI funding gpt-4.1-nano | ~$0.21 |
| OpenAI dense embed | ~$0.05 |
| LlamaParse Agentic-Plus (500 pages × $0.05625) | ~$28.13 |
| **Total** | **~$31.83** |

LlamaParse Agentic-Plus is the dominant line here — at high parse modes
the PDF column outweighs everything else combined. Drop to Cost-effective
mode and the total falls to ~$5.58.

---

## 7. What this doc deliberately omits

- **Qdrant, Redis, TEI dense, TEI sparse** — self-hosted, no
  external invoice. Capacity planning for these lives in `THROUGHPUT.md`.
- **Egress / ingress bandwidth** — usually rolled into the host bill, not
  per-URL.
- **Offline / non-`/online` ingestion routes** — the standalone batch and
  back-office tooling have a different cost shape and aren't part of the
  live ingestion path.
- **CPU / memory of the data-plane process itself** — server-side compute is
  effectively free for this workload (one uvicorn worker, asyncio-bound) and
  is covered by `THROUGHPUT.md §2`.
