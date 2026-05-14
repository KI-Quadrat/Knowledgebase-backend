"""
POST /api/v1/probe/provider — preset probe per third-party provider
POST /api/v1/probe/url      — allow-listed URL passthrough (optionally with stored creds)

Both endpoints actively fire one minimal HTTP request to a third party and
return the raw outcome (status code, rate-limit headers, error body) so an
operator can diagnose "auth failed vs. quota exhausted vs. rate limited vs.
network down" without grepping application logs.

Auth: `Depends(require_api_key)` — same `X-API-Key` header used by online
endpoints.
"""

import time
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, status as http_status

from app.config import ext
from app.dependencies.api_key import require_api_key
from app.models.health import HealthStatus
from app.models.probe import (
    ProbeProvider,
    ProbeProviderRequest,
    ProbeRequestSnapshot,
    ProbeResponse,
    ProbeResponseDetails,
    ProbeUrlRequest,
    UseCredsFrom,
)
from app.routers.shared.health import (
    _classify_http_status,
    _extract_jina_quota,
    _extract_openai_quota,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/probe",
    tags=["Diagnostics"],
    dependencies=[Depends(require_api_key)],
)


# Hosts the /probe/url passthrough is allowed to hit. Suffixes prefixed with
# "." match any subdomain (".jina.ai" matches "eu-r-beta.jina.ai" but not "evil-jina.ai").
# Exact hostnames must match in full.
_ALLOWED_HOSTS_EXACT: tuple[str, ...] = (
    "api.openai.com",
    "api.firecrawl.dev",
    "api.studio.nebius.ai",
    "api.together.xyz",
    "api.groq.com",
    "api.fireworks.ai",
    "api.deepinfra.com",
    "embed.ki2.at",
    "sparse.ki2.at",
)
_ALLOWED_HOSTS_SUFFIX: tuple[str, ...] = (
    ".jina.ai",
    ".llamaindex.ai",
)

# Maps UseCredsFrom to the DP_<X>_API_KEY env var name on `ext`. The map is
# expressed as the attribute name on the ExternalSettings instance, not the
# env-var spelling, since `ext.<name>` is the source of truth at runtime.
_CREDS_ENV_ATTR: dict[UseCredsFrom, str] = {
    UseCredsFrom.openai: "openai_api_key",
    UseCredsFrom.jina: "jina_api_key",
    UseCredsFrom.firecrawl: "firecrawl_api_key",
    UseCredsFrom.llamaparse: "llama_cloud_api_key",
    UseCredsFrom.nebius: "nebius_api_key",
    UseCredsFrom.together: "together_api_key",
    UseCredsFrom.groq: "groq_api_key",
    UseCredsFrom.fireworks: "fireworks_api_key",
    UseCredsFrom.deepinfra: "deepinfra_api_key",
    UseCredsFrom.tei_dense: "tei_embed_api_key_at",
    UseCredsFrom.tei_sparse: "sparse_embed_api_key_at",
    UseCredsFrom.crawl4ai: "crawl4ai_api_token",
}

_MAX_ERROR_BODY_BYTES = 4096
_ALLOWED_METHODS = {"GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"}


# ─────────────────────────────────────────────────────────────────────────────
# Shared low-level executor
# ─────────────────────────────────────────────────────────────────────────────


async def _execute_http_probe(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    json_body: dict | list | None,
    text_body: str | None,
    timeout: float,
    auth_indicator: str | None,
    provider_label: str,
    model: str | None,
    base_url: str | None,
) -> ProbeResponse:
    """Send one outbound HTTP request and assemble a structured ProbeResponse.

    - Sanitizes the upstream response (only `x-ratelimit-*` and `retry-after`
      headers are echoed; error bodies are truncated to 4 KiB).
    - Never echoes raw API keys — `auth_indicator` carries an opaque
      placeholder instead.
    """
    method_upper = method.upper()
    started = time.monotonic()
    request_snapshot = ProbeRequestSnapshot(method=method_upper, url=url, auth=auth_indicator)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            if json_body is not None and method_upper not in ("GET", "HEAD"):
                resp = await client.request(method_upper, url, headers=headers, json=json_body)
            elif text_body is not None and method_upper not in ("GET", "HEAD"):
                resp = await client.request(method_upper, url, headers=headers, content=text_body)
            else:
                resp = await client.request(method_upper, url, headers=headers)
    except httpx.TimeoutException:
        latency = int((time.monotonic() - started) * 1000)
        return ProbeResponse(
            status_code=None,
            status=HealthStatus.unreachable,
            latency_ms=latency,
            request=request_snapshot,
            response=None,
            quota=None,
            provider=provider_label,
            base_url=base_url,
            model=model,
            error="timeout",
        )
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        return ProbeResponse(
            status_code=None,
            status=HealthStatus.unreachable,
            latency_ms=latency,
            request=request_snapshot,
            response=None,
            quota=None,
            provider=provider_label,
            base_url=base_url,
            model=model,
            error=str(exc),
        )

    latency = int((time.monotonic() - started) * 1000)

    classified = _classify_http_status(resp.status_code)
    quota = _extract_openai_quota(resp.headers) or _extract_jina_quota(resp.headers)
    if classified == HealthStatus.ok and quota and quota.near_limit:
        classified = HealthStatus.near_quota_limit

    rate_limit_headers: dict[str, str] = {}
    for name, value in resp.headers.items():
        lname = name.lower()
        if lname.startswith("x-ratelimit") or lname == "retry-after":
            rate_limit_headers[name] = value

    error_body: str | None = None
    body_truncated = False
    if resp.status_code >= 400:
        try:
            body_text = resp.text
        except Exception:
            body_text = ""
        if body_text:
            if len(body_text.encode("utf-8")) > _MAX_ERROR_BODY_BYTES:
                # Truncate by bytes, not chars, to honour the cap exactly.
                error_body = body_text.encode("utf-8")[:_MAX_ERROR_BODY_BYTES].decode("utf-8", errors="replace")
                body_truncated = True
            else:
                error_body = body_text

    content_type = resp.headers.get("content-type")
    cl_header = resp.headers.get("content-length")
    try:
        content_length = int(cl_header) if cl_header else len(resp.content)
    except Exception:
        content_length = None

    details = ProbeResponseDetails(
        content_type=content_type,
        content_length=content_length,
        rate_limit_headers=rate_limit_headers,
        error_body=error_body,
        body_truncated=body_truncated,
    )

    return ProbeResponse(
        status_code=resp.status_code,
        status=classified,
        latency_ms=latency,
        request=request_snapshot,
        response=details,
        quota=quota,
        provider=provider_label,
        base_url=base_url,
        model=model,
        error=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-provider preset probes (POST /probe/provider)
# ─────────────────────────────────────────────────────────────────────────────


def _bearer_indicator(env_attr: str) -> str:
    # Opaque placeholder — never include the env-var name to avoid leaking
    # which secret is wired up.
    return "configured_provider_token"


def _openai_chat_base_url() -> str:
    return (ext.openai_base_url or "https://api.openai.com/v1").rstrip("/")


async def _preset_openai_chat(body: ProbeProviderRequest) -> ProbeResponse:
    if not ext.openai_api_key:
        return _not_configured("openai_chat", "openai key not configured")
    base = _openai_chat_base_url()
    model = body.model_override or ext.openai_model
    return await _execute_http_probe(
        method="POST",
        url=base + "/chat/completions",
        headers={"Authorization": f"Bearer {ext.openai_api_key}", "Content-Type": "application/json"},
        json_body={
            "model": model,
            "messages": [{"role": "user", "content": body.input_override or "ok"}],
            "temperature": 0.0,
            "max_tokens": 1,
        },
        text_body=None,
        timeout=20.0,
        auth_indicator=_bearer_indicator("openai_api_key"),
        provider_label="openai_chat",
        model=model,
        base_url=base,
    )


async def _preset_openai_embeddings(body: ProbeProviderRequest) -> ProbeResponse:
    if not ext.openai_api_key:
        return _not_configured("openai_embeddings", "openai key not configured")
    base = _openai_chat_base_url()
    model = body.model_override or "text-embedding-3-small"
    return await _execute_http_probe(
        method="POST",
        url=base + "/embeddings",
        headers={"Authorization": f"Bearer {ext.openai_api_key}", "Content-Type": "application/json"},
        json_body={"model": model, "input": body.input_override or "ok"},
        text_body=None,
        timeout=15.0,
        auth_indicator=_bearer_indicator("openai_api_key"),
        provider_label="openai_embeddings",
        model=model,
        base_url=base,
    )


async def _preset_jina(body: ProbeProviderRequest) -> ProbeResponse:
    if not ext.jina_api_key:
        return _not_configured("jina", "jina key not configured")
    base = ext.jina_api_url.rstrip("/")
    target = body.input_override or "https://example.com"
    return await _execute_http_probe(
        method="GET",
        url=f"{base}/{target}",
        headers={
            "Authorization": f"Bearer {ext.jina_api_key}",
            "Accept": "application/json",
            "X-Return-Format": "markdown",
        },
        json_body=None,
        text_body=None,
        timeout=15.0,
        auth_indicator=_bearer_indicator("jina_api_key"),
        provider_label="jina",
        model=None,
        base_url=base,
    )


async def _preset_firecrawl(body: ProbeProviderRequest) -> ProbeResponse:
    if not ext.firecrawl_api_key:
        return _not_configured("firecrawl", "firecrawl key not configured")
    base = ext.firecrawl_api_url.rstrip("/")
    target = body.input_override or "https://example.com"
    return await _execute_http_probe(
        method="POST",
        url=base + "/v1/scrape",
        headers={
            "Authorization": f"Bearer {ext.firecrawl_api_key}",
            "Content-Type": "application/json",
        },
        json_body={"url": target, "formats": ["markdown"]},
        text_body=None,
        timeout=20.0,
        auth_indicator=_bearer_indicator("firecrawl_api_key"),
        provider_label="firecrawl",
        model=None,
        base_url=base,
    )


async def _preset_llamaparse(body: ProbeProviderRequest) -> ProbeResponse:
    if not ext.llama_cloud_api_key:
        return _not_configured("llamaparse", "llamaparse not configured (local parser backend active)")
    base = ext.llama_cloud_base_url.rstrip("/")
    # 404 with valid creds → auth ok, resource missing — cheapest auth probe.
    probe_uuid = body.input_override or "00000000-0000-0000-0000-000000000000"
    resp = await _execute_http_probe(
        method="GET",
        url=f"{base}/job/{probe_uuid}",
        headers={"Authorization": f"Bearer {ext.llama_cloud_api_key}"},
        json_body=None,
        text_body=None,
        timeout=15.0,
        auth_indicator=_bearer_indicator("llama_cloud_api_key"),
        provider_label="llamaparse",
        model=None,
        base_url=base,
    )
    # Re-classify: 404 with this probe is healthy (key works, no such job).
    if resp.status_code == 404 and resp.status == HealthStatus.ok:
        return resp  # already ok
    if resp.status_code == 404:
        return resp.model_copy(update={"status": HealthStatus.ok})
    return resp


async def _preset_extra_chat(
    provider_name: str, env_attr: str, body: ProbeProviderRequest
) -> ProbeResponse:
    """Probe one of nebius/together/groq/fireworks/deepinfra via OpenAI-compatible chat."""
    from app.services.intelligence.llm_router import KNOWN_PROVIDERS, parse_spec

    api_key = getattr(ext, env_attr, "") or ""
    base_url = (
        (getattr(ext, f"{provider_name}_base_url", "") or "").rstrip("/")
        or KNOWN_PROVIDERS.get(provider_name, "")
    )
    if not api_key:
        return _not_configured(provider_name, f"{provider_name} key not configured")
    if not base_url:
        return _not_configured(provider_name, f"no base_url configured for provider {provider_name!r}")

    # If the caller did not override the model, try to find one referenced by a
    # task spec for this provider; fall back to "test" so the upstream can return
    # a model-not-found error which is still useful diagnostic information.
    model = body.model_override
    if not model:
        for spec in (ext.classifier_model, ext.contextual_model, ext.funding_model):
            if not spec:
                continue
            try:
                p, m = parse_spec(spec)
            except Exception:
                continue
            if p == provider_name:
                model = m
                break
    if not model:
        model = "test"

    return await _execute_http_probe(
        method="POST",
        url=base_url + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body={
            "model": model,
            "messages": [{"role": "user", "content": body.input_override or "ok"}],
            "temperature": 0.0,
            "max_tokens": 1,
        },
        text_body=None,
        timeout=20.0,
        auth_indicator=_bearer_indicator(env_attr),
        provider_label=provider_name,
        model=model,
        base_url=base_url,
    )


async def _preset_tei_dense(body: ProbeProviderRequest) -> ProbeResponse:
    if not ext.tei_embed_api_key_at:
        return _not_configured("tei_dense", "tei dense key not configured")
    base = ext.tei_embed_url_at.rstrip("/")
    headers = {
        "Authorization": f"Bearer {ext.tei_embed_api_key_at}",
        "Content-Type": "application/json",
    }
    # Honour optional Cloudflare Access service-token headers when set.
    if ext.tei_cf_access_client_id_at and ext.tei_cf_access_client_secret_at:
        headers["CF-Access-Client-Id"] = ext.tei_cf_access_client_id_at
        headers["CF-Access-Client-Secret"] = ext.tei_cf_access_client_secret_at
    return await _execute_http_probe(
        method="POST",
        url=base + "/v1/embeddings",
        headers=headers,
        json_body={"input": body.input_override or "ok", "model": body.model_override or ext.tei_embed_model_at},
        text_body=None,
        timeout=15.0,
        auth_indicator=_bearer_indicator("tei_embed_api_key_at"),
        provider_label="tei_dense",
        model=body.model_override or ext.tei_embed_model_at,
        base_url=base,
    )


async def _preset_tei_sparse(body: ProbeProviderRequest) -> ProbeResponse:
    if not ext.sparse_embed_api_key_at:
        return _not_configured("tei_sparse", "tei sparse key not configured")
    base = ext.sparse_embed_url_at.rstrip("/")
    headers = {
        "Authorization": f"Bearer {ext.sparse_embed_api_key_at}",
        "Content-Type": "application/json",
    }
    if ext.sparse_cf_access_client_id_at and ext.sparse_cf_access_client_secret_at:
        headers["CF-Access-Client-Id"] = ext.sparse_cf_access_client_id_at
        headers["CF-Access-Client-Secret"] = ext.sparse_cf_access_client_secret_at
    return await _execute_http_probe(
        method="POST",
        url=base + "/embed_sparse",
        headers=headers,
        json_body={"texts": [body.input_override or "ok"]},
        text_body=None,
        timeout=15.0,
        auth_indicator=_bearer_indicator("sparse_embed_api_key_at"),
        provider_label="tei_sparse",
        model=ext.sparse_embed_model_at,
        base_url=base,
    )


async def _preset_crawl4ai(body: ProbeProviderRequest) -> ProbeResponse:
    base = ext.crawl4ai_url.rstrip("/")
    headers: dict[str, str] = {}
    auth_indicator: str | None = None
    if ext.crawl4ai_api_token:
        headers["Authorization"] = f"Bearer {ext.crawl4ai_api_token}"
        auth_indicator = _bearer_indicator("crawl4ai_api_token")
    return await _execute_http_probe(
        method="GET",
        url=base + "/health",
        headers=headers,
        json_body=None,
        text_body=None,
        timeout=10.0,
        auth_indicator=auth_indicator,
        provider_label="crawl4ai",
        model=None,
        base_url=base,
    )


def _not_configured(provider_label: str, detail: str) -> ProbeResponse:
    return ProbeResponse(
        status_code=None,
        status=HealthStatus.not_configured,
        latency_ms=0,
        request=ProbeRequestSnapshot(method="-", url="-", auth=None),
        response=None,
        quota=None,
        provider=provider_label,
        base_url=None,
        model=None,
        error=detail,
    )


_PRESET_HANDLERS = {
    ProbeProvider.openai_chat: _preset_openai_chat,
    ProbeProvider.openai_embeddings: _preset_openai_embeddings,
    ProbeProvider.jina: _preset_jina,
    ProbeProvider.firecrawl: _preset_firecrawl,
    ProbeProvider.llamaparse: _preset_llamaparse,
    ProbeProvider.tei_dense: _preset_tei_dense,
    ProbeProvider.tei_sparse: _preset_tei_sparse,
    ProbeProvider.crawl4ai: _preset_crawl4ai,
}


@router.post(
    "/provider",
    response_model=ProbeResponse,
    summary="Probe one configured third-party provider",
    description=(
        "**Use this when one specific provider seems to be misbehaving** and you want to see "
        "the **exact error the provider returned** — auth message, rate-limit headers, "
        "remaining-token count, etc. — without grepping application logs or shipping a real "
        "ingest job through.\n\n"
        "Sends a single, minimal request to the chosen provider using the credentials "
        "already configured on the data-plane. You never paste a key.\n\n"
        "## When to call\n"
        "- An ingest started failing — \"is the LLM key or the parser subscription the problem?\"\n"
        "- After rotating a key — confirm the new value is wired up.\n"
        "- After hitting 429s in batch ingest — see how much of the rate-limit window is left.\n\n"
        "## Reading the response\n"
        "- `status_code` — raw HTTP status from the upstream.\n"
        "- `status` — classification: `ok` / `auth_failed` / `quota_exhausted` / `rate_limited` / `near_quota_limit` / `unreachable` / ...\n"
        "- `response.error_body` — verbatim upstream error JSON (4xx/5xx only, truncated to 4 KiB).\n"
        "- `response.rate_limit_headers` — `x-ratelimit-*` and `retry-after` from the upstream.\n"
        "- `quota` — parsed view: `remaining_tokens`, `remaining_requests`, `reset_at`, `near_limit`.\n"
        "- `request.auth` — opaque placeholder confirming a configured token was used. **The literal key is never echoed.**\n\n"
        "## Safety\n"
        "- 2xx responses do **not** echo the upstream body — only `content_length`. This is a "
        "diagnostic endpoint, not a content fetcher.\n"
        "- Each call costs a small amount of provider budget (1-token chat, 1 embed, etc.)."
    ),
)
async def probe_provider(body: ProbeProviderRequest) -> ProbeResponse:
    handler = _PRESET_HANDLERS.get(body.provider)
    if handler is not None:
        return await handler(body)
    # Extra LLM providers (per-task) share the same OpenAI-compatible chat probe.
    if body.provider in (
        ProbeProvider.nebius,
        ProbeProvider.together,
        ProbeProvider.groq,
        ProbeProvider.fireworks,
        ProbeProvider.deepinfra,
    ):
        return await _preset_extra_chat(
            provider_name=body.provider.value,
            env_attr=f"{body.provider.value}_api_key",
            body=body,
        )
    # Should be unreachable — every ProbeProvider value is mapped above.
    raise HTTPException(
        status_code=http_status.HTTP_400_BAD_REQUEST,
        detail=f"INVALID_PROVIDER: no handler for {body.provider.value}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Arbitrary URL passthrough (POST /probe/url)
# ─────────────────────────────────────────────────────────────────────────────


def _validate_host_allowed(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"VALIDATION_URL_INVALID: {exc}",
        )
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not host:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail="VALIDATION_URL_INVALID: scheme must be http(s) and host must be set",
        )
    if host in _ALLOWED_HOSTS_EXACT:
        return host
    for suffix in _ALLOWED_HOSTS_SUFFIX:
        if host.endswith(suffix):
            return host
    raise HTTPException(
        status_code=http_status.HTTP_400_BAD_REQUEST,
        detail="URL_NOT_ALLOWED: host is not in the probe allow-list",
    )


@router.post(
    "/url",
    response_model=ProbeResponse,
    summary="Probe an arbitrary (allow-listed) URL",
    description=(
        "**Use this when `/probe/provider` doesn't cover the exact endpoint you need** "
        "(e.g. a sub-route or account-info endpoint a preset doesn't probe). Set "
        "`use_creds_from` and the data-plane attaches its configured bearer token "
        "automatically — you never paste a key into the request body.\n\n"
        "## When to call (vs. `/probe/provider`)\n"
        "- You need a **specific URL** the preset doesn't probe.\n"
        "- You want to test that a key has access to a **specific** endpoint, not just \"the API in general\".\n"
        "- For everything else, prefer `/probe/provider`.\n\n"
        "## SSRF protection\n"
        "Hosts are restricted to a server-side allow-list. Anything else → 400 `URL_NOT_ALLOWED`.\n\n"
        "## `use_creds_from`\n"
        "When set, the data-plane attaches its configured bearer token for the chosen provider "
        "to the request. The literal key is **never** echoed back — the response's "
        "`request.auth` shows an opaque placeholder. If you pass your own `Authorization` "
        "header in `headers`, it wins over `use_creds_from`.\n\n"
        "## Reading the response\n"
        "Identical to `/probe/provider` — `status_code`, `status`, `response.error_body` "
        "(verbatim 4xx/5xx body, truncated to 4 KiB), `response.rate_limit_headers`, `quota`.\n\n"
        "## Safety\n"
        "- 2xx responses do **not** echo the upstream body — only `content_length`.\n"
        "- This endpoint **will** spend provider budget if your URL triggers a billable call."
    ),
)
async def probe_url(body: ProbeUrlRequest) -> ProbeResponse:
    _validate_host_allowed(body.url)

    method = body.method.upper()
    if method not in _ALLOWED_METHODS:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"INVALID_METHOD: {method} (allowed: {', '.join(sorted(_ALLOWED_METHODS))})",
        )

    headers: dict[str, str] = dict(body.headers or {})
    auth_indicator: str | None = None

    if body.use_creds_from is not None:
        env_attr = _CREDS_ENV_ATTR[body.use_creds_from]
        key = getattr(ext, env_attr, "") or ""
        if not key:
            return _not_configured(
                body.use_creds_from.value,
                f"{body.use_creds_from.value} key not configured — cannot attach creds",
            )
        # Don't clobber a caller-supplied Authorization header — let them override
        # if they pass it explicitly (e.g. probing what an alternate key does).
        if "authorization" not in {h.lower() for h in headers}:
            headers["Authorization"] = f"Bearer {key}"
            auth_indicator = _bearer_indicator(env_attr)
        else:
            auth_indicator = "caller-supplied Authorization header"

    json_body: dict | list | None = None
    text_body: str | None = None
    if isinstance(body.body, (dict, list)):
        json_body = body.body
        headers.setdefault("Content-Type", "application/json")
    elif isinstance(body.body, str):
        text_body = body.body

    return await _execute_http_probe(
        method=method,
        url=body.url,
        headers=headers,
        json_body=json_body,
        text_body=text_body,
        timeout=body.timeout_seconds,
        auth_indicator=auth_indicator,
        provider_label=(body.use_creds_from.value if body.use_creds_from else "url"),
        model=None,
        base_url=None,
    )
