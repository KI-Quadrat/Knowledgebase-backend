"""Models for the `/api/v1/probe/*` diagnostic endpoints.

These endpoints actively send a single minimal request to a configured
third-party provider (or an allow-listed URL) so an operator can see the raw
error surface — status code, rate-limit headers, error body — without having
to read application logs or burn an ingest job.

The response intentionally **does not include the success body content** —
only error bodies (4xx/5xx) are echoed, truncated to 4 KiB. This matches the
"diagnostic, not content-fetching" intent of the endpoint.
"""

from enum import Enum

from pydantic import BaseModel, Field

from app.models.health import HealthStatus, QuotaInfo


class ProbeProvider(str, Enum):
    """Preset probe targets for `POST /api/v1/probe/provider`.

    Each preset corresponds to a known third-party dependency. The endpoint
    knows the cheapest sensible call to make against each one using the
    credentials already in the data-plane config.
    """

    openai_chat = "openai_chat"
    openai_embeddings = "openai_embeddings"
    jina = "jina"
    firecrawl = "firecrawl"
    llamaparse = "llamaparse"
    nebius = "nebius"
    together = "together"
    groq = "groq"
    fireworks = "fireworks"
    deepinfra = "deepinfra"
    tei_dense = "tei_dense"
    tei_sparse = "tei_sparse"


class UseCredsFrom(str, Enum):
    """Credential source for `POST /api/v1/probe/url`.

    When set, the endpoint auto-attaches the configured bearer token for the
    chosen provider to the outbound request. The actual key is never echoed
    back in the response — only an opaque placeholder.
    """

    openai = "openai"
    jina = "jina"
    firecrawl = "firecrawl"
    llamaparse = "llamaparse"
    nebius = "nebius"
    together = "together"
    groq = "groq"
    fireworks = "fireworks"
    deepinfra = "deepinfra"
    tei_dense = "tei_dense"
    tei_sparse = "tei_sparse"


class ProbeProviderRequest(BaseModel):
    """Request body for `POST /api/v1/probe/provider`."""

    provider: ProbeProvider = Field(..., description="Which preset provider probe to run")
    model_override: str | None = Field(
        None,
        description="Override the model used for chat/embedding probes (defaults to the configured model or per-task spec).",
    )
    input_override: str | None = Field(
        None,
        description="Override the probe input string (defaults to 'ok' for chat/embedding).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"provider": "openai_chat"},
                {"provider": "llamaparse"},
                {"provider": "nebius", "model_override": "Qwen/Qwen2.5-72B-Instruct"},
            ]
        }
    }


class ProbeUrlRequest(BaseModel):
    """Request body for `POST /api/v1/probe/url`.

    Hosts are allow-listed at the endpoint to prevent SSRF. Supported hosts
    are the same set the provider preset endpoint covers.
    """

    url: str = Field(..., description="Fully-qualified URL to probe. Must be on the allow-list.")
    method: str = Field("GET", description="HTTP method (GET, POST, PUT, DELETE, HEAD, PATCH)")
    headers: dict[str, str] | None = Field(
        None,
        description="Additional request headers. Merged with auth headers when use_creds_from is set.",
    )
    body: dict | list | str | None = Field(
        None,
        description="Request body. JSON-encoded if dict/list; sent as-is if string. Ignored for GET/HEAD.",
    )
    use_creds_from: UseCredsFrom | None = Field(
        None,
        description=(
            "When set, the data-plane attaches its configured bearer token for the chosen "
            "provider to the request. The actual key is never echoed in the response — only "
            "an opaque placeholder string."
        ),
    )
    timeout_seconds: float = Field(
        15.0, ge=1.0, le=60.0, description="Per-request timeout in seconds"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://api.openai.com/v1/models",
                    "method": "GET",
                    "use_creds_from": "openai",
                },
                {
                    "url": "https://api.cloud.llamaindex.ai/api/v1/parsing/job/00000000-0000-0000-0000-000000000000",
                    "method": "GET",
                    "use_creds_from": "llamaparse",
                },
            ]
        }
    }


class ProbeRequestSnapshot(BaseModel):
    """Sanitized snapshot of the outbound request — never includes raw keys."""

    method: str = Field(..., description="HTTP method actually sent")
    url: str = Field(..., description="URL actually called")
    auth: str | None = Field(
        None,
        description="Opaque auth indicator — never the literal key",
    )


class ProbeResponseDetails(BaseModel):
    """Sanitized snapshot of the upstream response."""

    content_type: str | None = Field(None, description="Content-Type header from the upstream response")
    content_length: int | None = Field(None, description="Response body size in bytes (from header or measured)")
    rate_limit_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Subset of response headers relevant to rate limits / quotas (x-ratelimit-*, retry-after, etc.). Other headers are intentionally not echoed.",
    )
    error_body: str | None = Field(
        None,
        description=(
            "Upstream response body — populated only when the response is NOT 2xx (so successful "
            "calls do not leak content). Truncated to 4096 bytes; check body_truncated."
        ),
    )
    body_truncated: bool = Field(False, description="True if error_body was clipped at the 4 KiB cap")


class ProbeResponse(BaseModel):
    """Result of a single probe call.

    Designed so callers can answer "is this provider broken, and if so why?"
    from one HTTP response:

    - ``status_code`` + ``status`` classify the outcome.
    - ``quota`` shows remaining budget when the provider exposes it.
    - ``response.error_body`` carries the upstream's actual error JSON
      (4xx/5xx only) so messages like "Incorrect API key provided" or
      "You exceeded your current quota" reach the operator verbatim.
    """

    status_code: int | None = Field(None, description="HTTP status code returned by the upstream (null on network error)")
    status: HealthStatus = Field(..., description="Classification of the outcome (auth_failed, quota_exhausted, ...)")
    latency_ms: int = Field(..., description="Round-trip latency in milliseconds")
    request: ProbeRequestSnapshot = Field(..., description="Snapshot of what was sent")
    response: ProbeResponseDetails | None = Field(None, description="Snapshot of what came back (null on network error)")
    quota: QuotaInfo | None = Field(None, description="Rate-limit / quota snapshot when the provider exposes it")
    provider: str = Field(..., description="Which provider was probed")
    base_url: str | None = Field(None, description="Base URL the request was sent to (for verifying which env var drove the probe)")
    model: str | None = Field(None, description="Model used for chat/embedding probes (null for non-LLM probes)")
    error: str | None = Field(None, description="Network error message when the request could not be made; null otherwise")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "OK — OpenAI chat probe succeeded, quota healthy",
                    "value": {
                        "status_code": 200,
                        "status": "ok",
                        "latency_ms": 412,
                        "request": {
                            "method": "POST",
                            "url": "https://api.openai.com/v1/chat/completions",
                            "auth": "configured_provider_token",
                        },
                        "response": {
                            "content_type": "application/json",
                            "content_length": 358,
                            "rate_limit_headers": {
                                "x-ratelimit-limit-tokens": "200000",
                                "x-ratelimit-remaining-tokens": "199950",
                                "x-ratelimit-reset-tokens": "8ms",
                                "x-ratelimit-limit-requests": "10000",
                                "x-ratelimit-remaining-requests": "9998",
                                "x-ratelimit-reset-requests": "6ms",
                            },
                            "error_body": None,
                            "body_truncated": False,
                        },
                        "quota": {
                            "remaining_tokens": 199950,
                            "remaining_requests": 9998,
                            "reset_at": "8ms",
                            "near_limit": False,
                            "limit_kind": None,
                        },
                        "provider": "openai_chat",
                        "base_url": "https://api.openai.com/v1",
                        "model": "gpt-4o-mini",
                        "error": None,
                    },
                },
                {
                    "summary": "auth_failed — bad / expired / revoked OpenAI key",
                    "value": {
                        "status_code": 401,
                        "status": "auth_failed",
                        "latency_ms": 178,
                        "request": {
                            "method": "POST",
                            "url": "https://api.openai.com/v1/chat/completions",
                            "auth": "configured_provider_token",
                        },
                        "response": {
                            "content_type": "application/json",
                            "content_length": 184,
                            "rate_limit_headers": {},
                            "error_body": "{\"error\":{\"message\":\"Incorrect API key provided.\",\"type\":\"invalid_request_error\",\"param\":null,\"code\":\"invalid_api_key\"}}",
                            "body_truncated": False,
                        },
                        "quota": None,
                        "provider": "openai_chat",
                        "base_url": "https://api.openai.com/v1",
                        "model": "gpt-4o-mini",
                        "error": None,
                    },
                },
                {
                    "summary": "near_quota_limit — OpenAI TPM window almost drained",
                    "value": {
                        "status_code": 200,
                        "status": "near_quota_limit",
                        "latency_ms": 502,
                        "request": {
                            "method": "POST",
                            "url": "https://api.openai.com/v1/chat/completions",
                            "auth": "configured_provider_token",
                        },
                        "response": {
                            "content_type": "application/json",
                            "content_length": 358,
                            "rate_limit_headers": {
                                "x-ratelimit-limit-tokens": "200000",
                                "x-ratelimit-remaining-tokens": "5200",
                                "x-ratelimit-reset-tokens": "42s",
                                "x-ratelimit-limit-requests": "10000",
                                "x-ratelimit-remaining-requests": "9700",
                                "x-ratelimit-reset-requests": "1s",
                            },
                            "error_body": None,
                            "body_truncated": False,
                        },
                        "quota": {
                            "remaining_tokens": 5200,
                            "remaining_requests": 9700,
                            "reset_at": "42s",
                            "near_limit": True,
                            "limit_kind": "openai_tpm",
                        },
                        "provider": "openai_chat",
                        "base_url": "https://api.openai.com/v1",
                        "model": "gpt-4o-mini",
                        "error": None,
                    },
                },
                {
                    "summary": "rate_limited — provider returned 429",
                    "value": {
                        "status_code": 429,
                        "status": "rate_limited",
                        "latency_ms": 95,
                        "request": {
                            "method": "POST",
                            "url": "https://api.openai.com/v1/chat/completions",
                            "auth": "configured_provider_token",
                        },
                        "response": {
                            "content_type": "application/json",
                            "content_length": 256,
                            "rate_limit_headers": {
                                "retry-after": "20",
                                "x-ratelimit-remaining-tokens": "0",
                            },
                            "error_body": "{\"error\":{\"message\":\"Rate limit reached for gpt-4o-mini in organization org-*** on tokens per min (TPM): Limit 200000, Used 200000, Requested 1. Please try again in 20s.\",\"type\":\"tokens\",\"param\":null,\"code\":\"rate_limit_exceeded\"}}",
                            "body_truncated": False,
                        },
                        "quota": None,
                        "provider": "openai_chat",
                        "base_url": "https://api.openai.com/v1",
                        "model": "gpt-4o-mini",
                        "error": None,
                    },
                },
                {
                    "summary": "unreachable — network error / timeout",
                    "value": {
                        "status_code": None,
                        "status": "unreachable",
                        "latency_ms": 15001,
                        "request": {
                            "method": "GET",
                            "url": "https://api.cloud.llamaindex.ai/api/v1/parsing/job/00000000-0000-0000-0000-000000000000",
                            "auth": "configured_provider_token",
                        },
                        "response": None,
                        "quota": None,
                        "provider": "llamaparse",
                        "base_url": "https://api.cloud.llamaindex.ai/api/v1/parsing",
                        "model": None,
                        "error": "timeout",
                    },
                },
                {
                    "summary": "not_configured — env var unset, nothing was sent",
                    "value": {
                        "status_code": None,
                        "status": "not_configured",
                        "latency_ms": 0,
                        "request": {"method": "-", "url": "-", "auth": None},
                        "response": None,
                        "quota": None,
                        "provider": "firecrawl",
                        "base_url": None,
                        "model": None,
                        "error": "provider not configured",
                    },
                },
            ]
        }
    }
