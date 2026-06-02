from enum import Enum

from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Per-component or overall health classification.

    Maps probe outcomes onto actionable categories:

    - ``ok`` — healthy; probe succeeded and any quota signal is above threshold.
    - ``near_quota_limit`` — alive but the rate-limit window is nearly drained
      (provider returned a low ``x-ratelimit-remaining-*`` value).
    - ``rate_limited`` — provider returned 429.
    - ``quota_exhausted`` — credits / subscription exhausted (typically HTTP 402,
      or a provider-specific signal).
    - ``auth_failed`` — invalid / expired key, revoked, or subscription lapsed
      (HTTP 401 / 403).
    - ``unreachable`` — DNS, TCP, TLS, timeout, or 5xx.
    - ``degraded`` — reachable but partially failing (slow, unexpected body, etc.).
    - ``not_configured`` — required env vars are unset; nothing to probe.
    - ``disabled`` — explicitly turned off in this deployment (e.g. parser
      backend is local-only).
    """

    ok = "ok"
    near_quota_limit = "near_quota_limit"
    rate_limited = "rate_limited"
    quota_exhausted = "quota_exhausted"
    auth_failed = "auth_failed"
    unreachable = "unreachable"
    degraded = "degraded"
    not_configured = "not_configured"
    disabled = "disabled"


class QuotaInfo(BaseModel):
    """Provider rate-limit / quota snapshot.

    Populated only for components that expose budget signals through response
    headers (OpenAI-compatible providers, Jina, etc.). Components without
    quota concepts (Qdrant, Redis, TEI) leave this null.
    """

    remaining_tokens: int | None = Field(None, description="Tokens left in the current limit window")
    remaining_requests: int | None = Field(None, description="Requests left in the current limit window")
    reset_at: str | None = Field(None, description="When the limit window resets (provider-formatted string)")
    near_limit: bool = Field(False, description="True if remaining/limit < the configured warning threshold")
    limit_kind: str | None = Field(None, description="Which limit triggered near_limit (e.g. 'openai_tpm', 'openai_rpm', 'jina_requests')")


class ServiceStatus(BaseModel):
    """Status of each external dependency checked during readiness."""

    qdrant: bool = Field(False, description="Qdrant vector database reachable")
    bge_m3: bool = Field(False, description="Local BGE-M3 embedding service reachable")
    openai_embedder: bool = Field(False, description="OpenAI embeddings API reachable")
    tei_embedder_at: bool = Field(False, description="Configured TEI BGE-M3 embedding endpoint reachable")
    sparse_embedder: bool = Field(False, description="Configured TEI sparse embedding endpoint reachable")
    parser: bool = Field(False, description="Document parser available (LlamaParse or Unstructured)")
    scraper: bool = Field(False, description="Scraper client initialized (Jina / Firecrawl / httpx backends)")
    ldap: bool = Field(False, description="LDAP/Active Directory server reachable")
    redis: bool = Field(False, description="Redis cache reachable")


class ModelHealthItem(BaseModel):
    """Health of a single third-party dependency."""

    component: str = Field(..., description="Internal component name")
    task: str = Field(..., description="What this component is used for")
    provider: str = Field(..., description="Provider or serving stack")
    model: str = Field(..., description="Configured model name or backend label")
    healthy: bool = Field(..., description="True iff status == 'ok'. Kept for backward compatibility.")
    configured: bool = Field(..., description="True if the component is configured or enabled")
    required: bool = Field(..., description="True if this configured component is expected to be healthy")
    detail: str | None = Field(None, description="Short explanation for disabled or unhealthy state")
    status: HealthStatus = Field(HealthStatus.ok, description="Granular health classification")
    category: str = Field("llm", description="llm | embedding | scraper | parser | vector_db | cache | audit | directory")
    latency_ms: int | None = Field(None, description="Probe round-trip latency in milliseconds")
    quota: QuotaInfo | None = Field(None, description="Rate-limit / quota snapshot (only providers that expose it)")
    last_checked_at: str = Field(..., description="ISO-8601 UTC timestamp of the probe; reflects cache time when served from cache")


class HealthResponse(BaseModel):
    """Liveness check response. Returns immediately if the process is running."""

    status: str = Field("ok", description="Always 'ok' if the service is alive")
    uptime_seconds: float | None = Field(None, description="Seconds since service started")

    model_config = {
        "json_schema_extra": {
            "examples": [{"status": "ok", "uptime_seconds": 3421.5}]
        }
    }


class ReadyResponse(BaseModel):
    """Readiness check response.

    Without HMAC auth headers: returns minimal `{ready: true}`.
    With HMAC auth headers (X-Signature): returns full dependency status.
    """

    ready: bool = Field(..., description="True if all core services are operational")
    services: ServiceStatus | None = Field(None, description="Per-service health (only with HMAC auth)")
    mode: str | None = Field(None, description="Deployment mode: on-premise or cloud")
    tenant_id: str | None = Field(None, description="Municipality/tenant identifier")
    worker_id: str | None = Field(None, description="Worker instance identifier")
    version: str | None = Field(None, description="Data Plane version")
    uptime_seconds: float | None = Field(None, description="Seconds since service started")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "ready": True,
                    "services": {
                        "qdrant": True,
                        "bge_m3": True,
                        "openai_embedder": True,
                        "tei_embedder_at": True,
                        "sparse_embedder": True,
                        "parser": True,
                        "scraper": True,
                        "ldap": True,
                        "redis": True,
                    },
                    "mode": "on-premise",
                    "tenant_id": "example-tenant",
                    "worker_id": "worker-01",
                    "version": "1.0.0",
                    "uptime_seconds": 3421.5,
                }
            ]
        }
    }


class ModelHealthResponse(BaseModel):
    """Detailed health response for every third-party dependency.

    Each component is classified into a granular HealthStatus and (where the
    provider exposes it) accompanied by a QuotaInfo snapshot. Use ``overall``
    for a single-glance summary; ``any_quota_alerts`` and ``any_auth_failures``
    are convenient booleans for alerting.
    """

    overall: HealthStatus = Field(..., description="Aggregate status (worst-required wins): ok | degraded | auth_failed | quota_exhausted | unreachable")
    any_quota_alerts: bool = Field(False, description="True if any component is near_quota_limit, rate_limited, or quota_exhausted")
    any_auth_failures: bool = Field(False, description="True if any component is auth_failed")
    healthy: bool = Field(..., description="True iff overall == 'ok'. Kept for backward compatibility.")
    models: list[ModelHealthItem] = Field(default_factory=list, description="Per-component health entries")
    mode: str | None = Field(None, description="Deployment mode: on-premise or cloud")
    tenant_id: str | None = Field(None, description="Municipality/tenant identifier")
    worker_id: str | None = Field(None, description="Worker instance identifier")
    version: str | None = Field(None, description="Data Plane version")
    uptime_seconds: float | None = Field(None, description="Seconds since service started")
    cache_ttl_seconds: int = Field(60, description="Probe results are cached server-side for this many seconds; pass ?force=true to bypass.")
    served_from_cache: bool = Field(False, description="True if this response was served from the in-memory cache without re-probing")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Healthy deployment — everything green",
                    "value": {
                        "overall": "ok",
                        "any_quota_alerts": False,
                        "any_auth_failures": False,
                        "healthy": True,
                        "models": [
                            {
                                "component": "openai_chat",
                                "task": "content classification + contextual enrichment + funding extraction",
                                "provider": "openai",
                                "model": "gpt-4o-mini",
                                "healthy": True,
                                "configured": True,
                                "required": True,
                                "detail": None,
                                "status": "ok",
                                "category": "llm",
                                "latency_ms": 412,
                                "quota": {
                                    "remaining_tokens": 199950,
                                    "remaining_requests": 9998,
                                    "reset_at": "8.4s",
                                    "near_limit": False,
                                    "limit_kind": None,
                                },
                                "last_checked_at": "2026-05-14T08:47:08Z",
                            }
                        ],
                        "mode": "on-premise",
                        "tenant_id": "example-tenant",
                        "worker_id": "worker-01",
                        "version": "1.0.0",
                        "uptime_seconds": 3421.5,
                        "cache_ttl_seconds": 60,
                        "served_from_cache": False,
                    },
                },
                {
                    "summary": "Auth failure — OpenAI key rotated, ingest will break",
                    "value": {
                        "overall": "auth_failed",
                        "any_quota_alerts": False,
                        "any_auth_failures": True,
                        "healthy": False,
                        "models": [
                            {
                                "component": "openai_chat",
                                "task": "content classification + contextual enrichment + funding extraction",
                                "provider": "openai",
                                "model": "gpt-4o-mini",
                                "healthy": False,
                                "configured": True,
                                "required": True,
                                "detail": "OpenAI chat HTTP 401",
                                "status": "auth_failed",
                                "category": "llm",
                                "latency_ms": 178,
                                "quota": None,
                                "last_checked_at": "2026-05-14T08:47:08Z",
                            }
                        ],
                        "mode": "on-premise",
                        "tenant_id": "example-tenant",
                        "worker_id": "worker-01",
                        "version": "1.0.0",
                        "uptime_seconds": 3421.5,
                        "cache_ttl_seconds": 60,
                        "served_from_cache": False,
                    },
                },
                {
                    "summary": "Near quota — OpenAI minute window almost drained",
                    "value": {
                        "overall": "degraded",
                        "any_quota_alerts": True,
                        "any_auth_failures": False,
                        "healthy": False,
                        "models": [
                            {
                                "component": "openai_chat",
                                "task": "content classification + contextual enrichment + funding extraction",
                                "provider": "openai",
                                "model": "gpt-4o-mini",
                                "healthy": False,
                                "configured": True,
                                "required": True,
                                "detail": None,
                                "status": "near_quota_limit",
                                "category": "llm",
                                "latency_ms": 502,
                                "quota": {
                                    "remaining_tokens": 5200,
                                    "remaining_requests": 9700,
                                    "reset_at": "42s",
                                    "near_limit": True,
                                    "limit_kind": "openai_tpm",
                                },
                                "last_checked_at": "2026-05-14T08:47:08Z",
                            }
                        ],
                        "mode": "on-premise",
                        "tenant_id": "example-tenant",
                        "worker_id": "worker-01",
                        "version": "1.0.0",
                        "uptime_seconds": 3421.5,
                        "cache_ttl_seconds": 60,
                        "served_from_cache": False,
                    },
                },
            ]
        }
    }
