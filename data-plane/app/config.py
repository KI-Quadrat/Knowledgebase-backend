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

    # Jina Reader (fallback scraper)
    jina_api_url: str = "https://eu-r-beta.jina.ai"
    jina_api_key: str = ""

    # LlamaParse (cloud document parsing)
    llama_cloud_api_key: str = ""  # empty = use local unstructured parser
    llama_cloud_base_url: str = "https://api.cloud.llamaindex.ai/api/v1/parsing"  # EU: https://api.cloud.eu.llamaindex.ai/api/v1/parsing

    # BGE-M3
    bge_m3_url: str = "http://bge-m3:8080"

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
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

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
