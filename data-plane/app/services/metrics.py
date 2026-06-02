from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# ── Scraping metrics ──────────────────────────────────
SCRAPER_REQUESTS_TOTAL = Counter(
    "dp_scrape_requests_total",
    "Total scrape requests",
    ["endpoint", "status"],
)

SCRAPER_REQUEST_DURATION_SECONDS = Histogram(
    "dp_scrape_request_duration_seconds",
    "Scrape request duration in seconds",
    ["endpoint"],
)

SCRAPER_ACTIVE_JOBS = Gauge(
    "dp_scrape_active_jobs",
    "Current number of active scrape jobs",
)

SCRAPER_CACHE_HITS_TOTAL = Counter(
    "dp_cache_hits_total",
    "Total number of cache hits",
)

SCRAPER_CACHE_MISSES_TOTAL = Counter(
    "dp_cache_misses_total",
    "Total number of cache misses",
)

SCRAPER_RATE_LIMIT_WAITS_TOTAL = Counter(
    "dp_rate_limit_waits_total",
    "Total number of rate limiter waits",
    ["domain"],
)

# ── Parse metrics ─────────────────────────────────────
PARSE_REQUESTS_TOTAL = Counter(
    "dp_parse_requests_total",
    "Total parse requests",
    ["status"],
)

PARSE_DURATION_SECONDS = Histogram(
    "dp_parse_duration_seconds",
    "Parse request duration in seconds",
)

# ── Ingest metrics ────────────────────────────────────
INGEST_REQUESTS_TOTAL = Counter(
    "dp_ingest_requests_total",
    "Total ingest requests",
    ["status"],
)

INGEST_DURATION_SECONDS = Histogram(
    "dp_ingest_duration_seconds",
    "Ingest pipeline duration in seconds",
)

INGEST_CHUNKS_TOTAL = Counter(
    "dp_ingest_chunks_total",
    "Total chunks created by ingest",
)

# ── Search metrics ────────────────────────────────────
SEARCH_REQUESTS_TOTAL = Counter(
    "dp_search_requests_total",
    "Total search requests",
    ["status"],
)

SEARCH_DURATION_SECONDS = Histogram(
    "dp_search_duration_seconds",
    "Search request duration in seconds",
)

# ── Classify metrics ─────────────────────────────
CLASSIFY_REQUESTS_TOTAL = Counter(
    "dp_classify_requests_total",
    "Total classify requests",
    ["status"],
)

CLASSIFY_DURATION_SECONDS = Histogram(
    "dp_classify_duration_seconds",
    "Classify request duration in seconds",
)

# ── Embedding metrics ─────────────────────────────────
EMBEDDING_REQUESTS_TOTAL = Counter(
    "dp_embedding_requests_total",
    "Total embedding requests",
    ["status"],
)

EMBEDDING_DURATION_SECONDS = Histogram(
    "dp_embedding_duration_seconds",
    "Embedding request duration in seconds",
)

# ── Discover metrics ──────────────────────────────────
DISCOVER_REQUESTS_TOTAL = Counter(
    "dp_discover_requests_total",
    "Total file discovery requests",
    ["source", "status"],
)


# ── Helper functions ──────────────────────────────────

def observe_request(endpoint: str, status: str, duration_seconds: float) -> None:
    SCRAPER_REQUESTS_TOTAL.labels(endpoint=endpoint, status=status).inc()
    SCRAPER_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(max(duration_seconds, 0.0))


def set_active_jobs(value: int) -> None:
    SCRAPER_ACTIVE_JOBS.set(max(value, 0))


def mark_cache_hit() -> None:
    SCRAPER_CACHE_HITS_TOTAL.inc()


def mark_cache_miss() -> None:
    SCRAPER_CACHE_MISSES_TOTAL.inc()


def mark_rate_limit_wait(domain: str) -> None:
    SCRAPER_RATE_LIMIT_WAITS_TOTAL.labels(domain=domain or "unknown").inc()


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
