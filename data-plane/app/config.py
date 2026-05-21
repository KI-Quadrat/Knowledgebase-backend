from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Data Plane internal settings."""

    model_config = {"env_prefix": "DP_"}

    # Auth
    hmac_secret: str = ""  # HMAC-SHA256 shared secret (empty = auth disabled)
    hmac_max_age: int = 300  # Max age of signed requests in seconds

    # CORS
    cors_origins: str = "*"

    # Scraping defaults
    default_timeout: int = 30
    max_concurrent: int = 10
    max_batch_urls: int = 50
    max_sitemap_pages: int = 500

    # Batch ingest limits — caps on POST /api/v1/online/batch/ingest.
    # ``max_batch_ingest_items`` rejects requests larger than this with
    # ``BATCH_TOO_LARGE``. ``batch_ingest_concurrency`` caps how many items
    # run simultaneously inside one batch (asyncio.Semaphore size). Keep
    # the concurrency at or below your OpenAI tier's concurrent quota.
    max_batch_ingest_items: int = 50
    batch_ingest_concurrency: int = 10
    contextual_concurrency: int = 5  # Max contextual-enrichment chat calls per worker
    inner_parse_concurrency: int = 5  # Max inner image/document parses per scrape
    inner_parse_rate_limit_retry_delay: float = 1.0  # Seconds before one 429 retry

    # Cache
    cache_ttl: int = 3600

    # Parsing
    max_file_size_mb: int = 50

    # Ingest
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50

    # Search
    default_top_k: int = 10
    default_score_threshold: float = 0.5

    # Online API key security
    online_api_keys: str = ""  # Comma-separated valid API keys for online endpoints

    # Logging
    log_level: str = "info"
    log_json: bool = True

    # Deployment
    mode: str = "on-premise"  # "on-premise" or "cloud"
    tenant_id: str = ""
    worker_id: str = ""
    version: str = "1.0.0"


class ExternalSettings(BaseSettings):
    """Settings for external services — no env prefix."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Crawl4AI
    crawl4ai_url: str = "http://crawl4ai:11235"
    crawl4ai_api_token: str = ""

    # Default scraping backend used when the request body omits ``scraper``.
    # Must be one of ``"jina"`` or ``"crawl4ai"`` — invalid values fall back to
    # ``"jina"`` at the request-model layer.
    default_scraper: str = "jina"

    # Default crawler backend used when ``/crawl`` requests with
    # ``method="crawl"`` omit ``scraper``. Valid values: ``"httpx"``,
    # ``"crawl4ai"``, ``"jina"``, ``"firecrawl"``. Invalid values fall back
    # to ``"httpx"`` at the request-model layer.
    default_crawler: str = "httpx"

    # Jina Reader (now the default backend; Crawl4AI /md is the fallback).
    jina_api_url: str = "https://eu-r-beta.jina.ai"
    jina_api_key: str = ""
    # Comma-separated list of domains where requests with scraper="crawl4ai"
    # are forced back to Jina (overrides the caller's explicit choice for
    # known-bad domains). Subdomains match by suffix — listing "stadt-wien.at"
    # routes both "stadt-wien.at" and "www.stadt-wien.at". Empty disables the
    # override; the default backend (Jina) is unaffected by this list.
    jina_default_domains: str = ""

    # Firecrawl (optional third backend — opt-in via ``scraper="firecrawl"``).
    # Point ``firecrawl_api_url`` at a self-hosted EU instance for data
    # residency; the managed cloud has no EU region.
    firecrawl_api_url: str = "https://api.firecrawl.dev"
    firecrawl_api_key: str = ""

    # LlamaParse (cloud document parsing)
    llama_cloud_api_key: str = ""  # empty = use local unstructured parser
    llama_cloud_base_url: str = "https://api.cloud.llamaindex.ai/api/v1/parsing"  # EU: https://api.cloud.eu.llamaindex.ai/api/v1/parsing

    # BGE-M3
    bge_m3_url: str = "http://bge-m3:8080"
    # Per-request batch cap for the self-hosted BGE-M3 server. embed_batch
    # splits larger inputs into windows of this size. 0 disables splitting.
    bge_m3_max_batch: int = 32

    # Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = ""  # Default collection name (tenant-based)

    # Qdrant — AT-specific instance used by POST /api/v1/online/ingest/at.
    # Host, port, and api-key are split across three env vars (matches the
    # upstream qdrant-client pattern — QDRANT_URL / QDRANT_PORT / QDRANT_API_KEY).
    # QDRANT_URL_AT may include the port inline (e.g. https://host:6333) or
    # carry only the scheme+host with QDRANT_PORT_AT supplying the port.
    # QDRANT_PORT_AT has no default — leave it unset when the port is already
    # embedded in the URL (including the implicit 443 for https:// URLs).
    # When QDRANT_URL_AT is empty, the service reuses the default QDRANT_URL /
    # QDRANT_API_KEY (port embedded in QDRANT_URL as before).
    qdrant_url_at: str = ""
    qdrant_port_at: int | None = None
    qdrant_api_key_at: str = ""

    # Redis
    redis_url: str = "redis://redis:6379/0"

    # Rate limiting
    rate_limit_per_domain: int = 10
    rate_limit_window: int = 60

    # ClickHouse
    clickhouse_required: bool = False
    clickhouse_host: str = "clickhouse"
    clickhouse_port: int = 9000
    clickhouse_db: str = "ki2_audit"
    clickhouse_user: str = "dataplane"
    clickhouse_password: str = ""

    # LiteLLM (self-hosted proxy for fallback embedding model)
    litellm_url: str = "http://litellm:4000"
    litellm_api_key: str = ""
    bge_gemma2_model: str = "bge-multilingual-gemma2"
    bge_gemma2_dense_dim: int = 3584

    # OpenAI
    # ``openai_model`` is also the *global default* model for any intelligence
    # task whose per-task override (classifier_model / contextual_model /
    # funding_model) is empty. Bare model names (no slash) are interpreted
    # as the ``openai`` provider by the router.
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # Set this only when pointing the ``openai`` provider at a custom
    # endpoint (Azure OpenAI, a self-hosted vLLM, a corporate proxy). Empty
    # uses the OpenAI SDK's built-in https://api.openai.com/v1.
    openai_base_url: str = ""
    # Per-request batch cap for OpenAI /v1/embeddings. The API allows up to
    # 2048 inputs (and ~300K tokens) per request; smaller windows keep
    # individual requests bounded. 0 disables splitting.
    openai_embed_max_batch: int = 256
    # Per-call cap for the contextual enricher's batched chat-completion path
    # (one JSON array of contexts per call). Larger inputs are split into
    # parallel windows of this size. 0 disables splitting (single call for
    # all chunks — risks max_tokens truncation on big docs).
    openai_contextual_max_batch: int = 32
    # Char caps on OpenAI inputs. Sized for gpt-4o-mini's 128K-token window at
    # ~4 chars/token, leaving headroom for system prompt scaffolding + response.
    # If you swap to a smaller-window model, drop these.
    classify_max_input_chars: int = 120_000
    funding_max_input_chars: int = 120_000
    contextual_doc_max_chars: int = 60_000

    # ── Per-task model routing ──────────────────────────────────────────
    # Each intelligence task picks its own model via "<provider>/<model_id>"
    # (e.g. "openai/gpt-4o-mini", "nebius/Qwen/Qwen2.5-72B-Instruct"). When
    # empty, falls back to ``openai_model`` above. A bare model name is
    # interpreted as the ``openai`` provider. See ``llm_router.py`` for the
    # resolution rules.
    classifier_model: str = ""
    contextual_model: str = ""
    funding_model: str = ""

    # ── Additional OpenAI-compatible providers ──────────────────────────
    # Built-in providers are pre-wired with public endpoints — set the
    # corresponding api_key for each one you intend to use. ``base_url`` is
    # only needed when overriding the default (self-hosted, vLLM, etc.).
    # Add a new provider by declaring its pair of fields here and the router
    # will pick them up via getattr.
    nebius_base_url: str = ""       # default: https://api.studio.nebius.ai/v1
    nebius_api_key: str = ""
    together_base_url: str = ""     # default: https://api.together.xyz/v1
    together_api_key: str = ""
    groq_base_url: str = ""         # default: https://api.groq.com/openai/v1
    groq_api_key: str = ""
    fireworks_base_url: str = ""    # default: https://api.fireworks.ai/inference/v1
    fireworks_api_key: str = ""
    deepinfra_base_url: str = ""    # default: https://api.deepinfra.com/v1/openai
    deepinfra_api_key: str = ""

    # TEI — AT-specific embedding endpoint used by POST /api/v1/online/ingest/at.
    # OpenAI-compatible server exposing POST {TEI_EMBED_URL_AT}/v1/embeddings.
    # API key is required. TEI_EMBED_MODEL_AT is optional — many TEI servers
    # ignore the model field since each process serves a single model.
    #
    # When the endpoint is behind Cloudflare Access, also supply a service-token
    # pair — the client sends them as CF-Access-Client-Id / -Client-Secret
    # headers so Cloudflare lets the request through without the login redirect.
    tei_embed_url_at: str = "https://embed.ki2.at"
    tei_embed_api_key_at: str = ""
    tei_embed_model_at: str = "BAAI/bge-m3"
    tei_cf_access_client_id_at: str = ""
    tei_cf_access_client_secret_at: str = ""
    # Per-request cap enforced by the TEI dense server (--max-client-batch-size).
    # embed_batch splits larger inputs into sequential windows. 0 disables split.
    tei_embed_max_batch_at: int = 32

    # Sparse embedding endpoint used for hybrid-search sparse vectors.
    # Dedicated POST {sparse_embed_url_at}/embed_sparse with body {"texts": [...]};
    # bearer ``sparse_embed_api_key_at`` plus optional Cloudflare Access
    # service-token headers. ``sparse_embed_model_at`` is unused by the request
    # (the endpoint serves a fixed model) and is retained only for logging.
    sparse_embed_url_at: str = "https://sparse.ki2.at"
    sparse_embed_api_key_at: str = ""
    sparse_embed_model_at: str = "BAAI/bge-m3"
    sparse_cf_access_client_id_at: str = ""
    sparse_cf_access_client_secret_at: str = ""
    # Per-request cap enforced by the TEI sparse server (max-client-batch-size).
    # Larger inputs to ``encode_batch`` are split into windows of this size and
    # POSTed sequentially. Set to 0 to disable splitting (single request).
    sparse_embed_max_batch_at: int = 32

    # LDAP
    ldap_url: str = ""
    ldap_bind_dn: str = ""
    ldap_bind_password: str = ""
    ldap_base_dn: str = ""

    # SMB
    smb_username: str = ""
    smb_password: str = ""
    smb_domain: str = ""

    # Cloudflare R2
    r2_endpoint_url: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""


settings = Settings()
ext = ExternalSettings()
