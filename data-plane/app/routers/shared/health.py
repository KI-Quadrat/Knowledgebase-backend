"""
GET /api/v1/health        — Liveness check (no auth)
GET /api/v1/ready         — Readiness check (minimal without auth, full with HMAC)
GET /api/v1/model-health  — Detailed per-dependency health (X-API-Key auth, 60s cache, ?force=true)
"""

import time
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, Request

from app.config import ext, settings
from app.dependencies.api_key import require_api_key
from app.models.health import (
    HealthResponse,
    HealthStatus,
    ModelHealthItem,
    ModelHealthResponse,
    QuotaInfo,
    ReadyResponse,
    ServiceStatus,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Health"])
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# ── /model-health cache + thresholds ─────────────────────────
# Probes burn small amounts of LLM tokens (OpenAI / Nebius / etc. chat completions)
# and external rate-limit budget (Jina, LlamaParse). Cache the assembled response
# for `_MODEL_HEALTH_CACHE_TTL_S` seconds. Callers bypass with ?force=true.
_MODEL_HEALTH_CACHE_TTL_S = 60
_NEAR_QUOTA_THRESHOLD_PCT = 5.0
_model_health_cache: dict[str, tuple[float, ModelHealthResponse]] = {}
_MODEL_HEALTH_CACHE_KEY = "default"


def _uptime(request: Request) -> float:
    start = getattr(request.app.state, "start_time", None)
    if start is None:
        return 0.0
    return round(time.monotonic() - start, 1)


async def _probe_component(component: object, method_name: str) -> tuple[bool, str | None]:
    method = getattr(component, method_name, None)
    if method is None:
        return False, f"missing {method_name}()"
    try:
        result = await method("health-check")
        return bool(result), None
    except Exception as exc:
        return False, str(exc)


async def _probe_openai_chat_model() -> tuple[bool, str | None]:
    if not ext.openai_api_key:
        return False, "openai key not configured"

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20)) as client:
            resp = await client.post(
                OPENAI_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {ext.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ext.openai_model,
                    "messages": [{"role": "user", "content": "Reply with: ok"}],
                    "temperature": 0.0,
                    "max_tokens": 5,
                },
            )
            resp.raise_for_status()
        return True, None
    except httpx.HTTPStatusError as exc:
        return False, f"OpenAI HTTP {exc.response.status_code}"
    except Exception as exc:
        return False, str(exc)


async def _probe_jina_reader(scraping_svc: object) -> tuple[bool, str | None]:
    crawl_client = getattr(scraping_svc, "scraper_client", None)
    http_client = getattr(crawl_client, "_client", None)
    jina_key = getattr(crawl_client, "_jina_key", "")
    jina_url = getattr(crawl_client, "_jina_url", "")

    if not crawl_client or not http_client:
        return False, "service not initialized"
    if not jina_key:
        return False, "jina key not configured"

    try:
        resp = await http_client.get(
            f"{jina_url}/https://example.com",
            headers={
                "Authorization": f"Bearer {jina_key}",
                "Accept": "application/json",
                "X-Return-Format": "markdown",
            },
            timeout=10.0,
        )
        resp.raise_for_status()
        return True, None
    except httpx.HTTPStatusError as exc:
        return False, f"Jina HTTP {exc.response.status_code}"
    except Exception as exc:
        return False, str(exc)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Returns `{status: ok}` if the Data Plane process is running. No authentication required. Used by container orchestrators (Docker, Kubernetes) as a liveness probe.",
    response_description="Service is alive",
)
async def health(request: Request) -> HealthResponse:
    return HealthResponse(status="ok", uptime_seconds=_uptime(request))


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness check",
    description="""Check if the Data Plane and all its dependencies are ready to serve requests.

**Without auth headers:** Returns minimal `{ready: true/false}` — suitable for load balancer health checks.

**With internal auth headers:** Returns per-service dependency status (vector DB, embedders, parser, scraper, directory, cache).

Core services that must be healthy for `ready: true`: vector DB, primary embedder, parser, scraper.""",
    response_description="Readiness status with optional service details",
)
async def ready(request: Request) -> ReadyResponse:
    has_auth = bool(request.headers.get("X-Signature"))
    uptime = _uptime(request)

    if not has_auth:
        return ReadyResponse(ready=True, uptime_seconds=uptime)

    services = ServiceStatus()

    # Scraper client (Jina / Firecrawl / httpx)
    scraping_svc = getattr(request.app.state, "scraping", None)
    if scraping_svc:
        services.scraper = getattr(scraping_svc, "is_ready", False)

    # Qdrant
    qdrant = getattr(request.app.state, "qdrant", None)
    if qdrant:
        try:
            services.qdrant = await qdrant.check_health()
        except Exception:
            services.qdrant = False

    # BGE-M3
    embedder = getattr(request.app.state, "embedder", None)
    if embedder:
        try:
            services.bge_m3 = await embedder.check_health()
        except Exception:
            services.bge_m3 = False

    # OpenAI embedder
    openai_embedder = getattr(request.app.state, "openai_embedder", None)
    if openai_embedder:
        try:
            services.openai_embedder = await openai_embedder.check_health()
        except Exception:
            services.openai_embedder = False

    # TEI BGE-M3 (embed.ki2.at)
    tei_embedder_at = getattr(request.app.state, "tei_embedder_at", None)
    if tei_embedder_at:
        try:
            services.tei_embedder_at = await tei_embedder_at.check_health()
        except Exception:
            services.tei_embedder_at = False

    # TEI sparse (sparse.ki2.at)
    sparse_embedder = getattr(request.app.state, "sparse_embedder", None)
    if sparse_embedder:
        try:
            services.sparse_embedder = await sparse_embedder.check_health()
        except Exception:
            services.sparse_embedder = False

    # Parser (LlamaParse or Unstructured)
    parser = getattr(request.app.state, "parser", None)
    if parser:
        try:
            services.parser = await parser.check_health()
        except Exception:
            services.parser = False

    # LDAP
    ldap = getattr(request.app.state, "ldap", None)
    if ldap:
        try:
            services.ldap = await ldap.check_health()
        except Exception:
            services.ldap = False

    # Redis (cache)
    cache = getattr(request.app.state, "cache", None)
    if cache:
        try:
            services.redis = await cache.ping()
        except Exception:
            services.redis = False

    all_ready = all([
        services.scraper,
        services.qdrant,
        services.bge_m3,
        services.parser,
    ])

    return ReadyResponse(
        ready=all_ready,
        services=services,
        mode=settings.mode,
        tenant_id=settings.tenant_id,
        worker_id=settings.worker_id,
        version=settings.version,
        uptime_seconds=uptime,
    )


# ─────────────────────────────────────────────────────────────────────────────
# /model-health — granular per-dependency probes
# ─────────────────────────────────────────────────────────────────────────────


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _classify_http_status(status_code: int) -> HealthStatus:
    """Map an HTTP status code from a third-party probe onto our HealthStatus."""
    if 200 <= status_code < 300:
        return HealthStatus.ok
    if status_code in (401, 403):
        return HealthStatus.auth_failed
    if status_code == 402:
        return HealthStatus.quota_exhausted
    if status_code == 429:
        return HealthStatus.rate_limited
    if status_code == 404:
        # 404 with valid creds means "endpoint reached, resource missing"
        # — useful as an "auth probe" pattern (e.g. LlamaParse job lookup).
        return HealthStatus.ok
    if 400 <= status_code < 500:
        return HealthStatus.degraded
    return HealthStatus.unreachable  # 5xx


def _extract_openai_quota(headers: httpx.Headers | dict) -> QuotaInfo | None:
    """Read OpenAI-compatible x-ratelimit-* headers (also works for Nebius,
    Together, Groq, Fireworks, DeepInfra)."""

    def _int(name: str) -> int | None:
        raw = headers.get(name)
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    rem_tok = _int("x-ratelimit-remaining-tokens")
    rem_req = _int("x-ratelimit-remaining-requests")
    lim_tok = _int("x-ratelimit-limit-tokens")
    lim_req = _int("x-ratelimit-limit-requests")
    reset = headers.get("x-ratelimit-reset-tokens") or headers.get("x-ratelimit-reset-requests")

    if rem_tok is None and rem_req is None:
        return None

    near = False
    limit_kind: str | None = None
    threshold = _NEAR_QUOTA_THRESHOLD_PCT / 100.0
    if rem_tok is not None and lim_tok and lim_tok > 0 and rem_tok / lim_tok < threshold:
        near, limit_kind = True, "openai_tpm"
    if rem_req is not None and lim_req and lim_req > 0 and rem_req / lim_req < threshold:
        near, limit_kind = True, limit_kind or "openai_rpm"

    return QuotaInfo(
        remaining_tokens=rem_tok,
        remaining_requests=rem_req,
        reset_at=reset,
        near_limit=near,
        limit_kind=limit_kind,
    )


def _extract_jina_quota(headers: httpx.Headers | dict) -> QuotaInfo | None:
    rem = headers.get("x-ratelimit-remaining")
    lim = headers.get("x-ratelimit-limit")
    reset = headers.get("x-ratelimit-reset")
    try:
        rem_i = int(rem) if rem is not None else None
        lim_i = int(lim) if lim is not None else None
    except (TypeError, ValueError):
        rem_i = lim_i = None
    if rem_i is None:
        return None
    near = (
        lim_i is not None
        and lim_i > 0
        and (rem_i / lim_i) < (_NEAR_QUOTA_THRESHOLD_PCT / 100.0)
    )
    return QuotaInfo(
        remaining_requests=rem_i,
        reset_at=reset,
        near_limit=near,
        limit_kind="jina_requests" if near else None,
    )


def _item(
    *,
    component: str,
    task: str,
    provider: str,
    model: str,
    configured: bool,
    status: HealthStatus,
    detail: str | None = None,
    category: str = "llm",
    latency_ms: int | None = None,
    quota: QuotaInfo | None = None,
) -> ModelHealthItem:
    required = configured and status not in (HealthStatus.disabled, HealthStatus.not_configured)
    return ModelHealthItem(
        component=component,
        task=task,
        provider=provider,
        model=model,
        configured=configured,
        required=required,
        healthy=(status == HealthStatus.ok),
        status=status,
        detail=detail,
        category=category,
        latency_ms=latency_ms,
        quota=quota,
        last_checked_at=_utc_now_iso(),
    )


async def _probe_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider_label: str,
) -> tuple[HealthStatus, str | None, QuotaInfo | None, int]:
    """Generic OpenAI-compatible chat probe — reused for openai_chat and the
    per-task extra providers (nebius, together, groq, fireworks, deepinfra)."""
    if not api_key:
        return HealthStatus.not_configured, f"{provider_label}: API key not configured", None, 0
    url = base_url.rstrip("/") + "/chat/completions"
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20)) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "temperature": 0.0,
                    "max_tokens": 1,
                },
            )
        latency = int((time.monotonic() - started) * 1000)
        status = _classify_http_status(resp.status_code)
        quota = _extract_openai_quota(resp.headers) if resp.status_code < 500 else None
        if status == HealthStatus.ok and quota and quota.near_limit:
            status = HealthStatus.near_quota_limit
        detail = None if status == HealthStatus.ok else f"{provider_label} HTTP {resp.status_code}"
        return status, detail, quota, latency
    except httpx.TimeoutException:
        return HealthStatus.unreachable, f"{provider_label}: timeout", None, int((time.monotonic() - started) * 1000)
    except Exception as exc:
        return HealthStatus.unreachable, f"{provider_label}: {exc}", None, int((time.monotonic() - started) * 1000)


def _openai_chat_base_url() -> str:
    return (ext.openai_base_url or "https://api.openai.com/v1").rstrip("/")


async def _probe_openai_chat_item() -> ModelHealthItem:
    status, detail, quota, latency = await _probe_chat(
        base_url=_openai_chat_base_url(),
        api_key=ext.openai_api_key,
        model=ext.openai_model,
        provider_label="OpenAI chat",
    )
    return _item(
        component="openai_chat",
        task="content classification + contextual enrichment + funding extraction",
        provider="openai",
        model=ext.openai_model,
        configured=bool(ext.openai_api_key),
        status=status,
        detail=detail,
        category="llm",
        latency_ms=latency,
        quota=quota,
    )


async def _probe_openai_embeddings_item() -> ModelHealthItem:
    model = "text-embedding-3-small"
    if not ext.openai_api_key:
        return _item(
            component="openai_embeddings",
            task="OpenAI online embedding (dense_openai)",
            provider="openai",
            model=model,
            configured=False,
            status=HealthStatus.not_configured,
            detail="openai key not configured",
            category="embedding",
        )
    url = _openai_chat_base_url() + "/embeddings"
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15)) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {ext.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "input": "ok"},
            )
        latency = int((time.monotonic() - started) * 1000)
        status = _classify_http_status(resp.status_code)
        quota = _extract_openai_quota(resp.headers) if resp.status_code < 500 else None
        if status == HealthStatus.ok and quota and quota.near_limit:
            status = HealthStatus.near_quota_limit
        detail = None if status == HealthStatus.ok else f"OpenAI embeddings HTTP {resp.status_code}"
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        status = HealthStatus.unreachable
        detail = f"OpenAI embeddings: {exc}"
        quota = None
    return _item(
        component="openai_embeddings",
        task="OpenAI online embedding (dense_openai)",
        provider="openai",
        model=model,
        configured=True,
        status=status,
        detail=detail,
        category="embedding",
        latency_ms=latency,
        quota=quota,
    )


async def _probe_jina_item(request: Request) -> ModelHealthItem:
    scraping_svc = getattr(request.app.state, "scraping", None)
    crawl_client = getattr(scraping_svc, "scraper_client", None)
    http_client = getattr(crawl_client, "_client", None)
    jina_key = getattr(crawl_client, "_jina_key", "")
    jina_url = getattr(crawl_client, "_jina_url", "")

    if not crawl_client or not http_client:
        return _item(
            component="jina_reader",
            task="reader-based web scraping",
            provider="jina",
            model="jina-reader",
            configured=False,
            status=HealthStatus.not_configured,
            detail="scraping service not initialized",
            category="scraper",
        )
    if not jina_key:
        return _item(
            component="jina_reader",
            task="reader-based web scraping",
            provider="jina",
            model="jina-reader",
            configured=False,
            status=HealthStatus.not_configured,
            detail="jina key not configured",
            category="scraper",
        )
    started = time.monotonic()
    try:
        resp = await http_client.get(
            f"{jina_url}/https://example.com",
            headers={
                "Authorization": f"Bearer {jina_key}",
                "Accept": "application/json",
                "X-Return-Format": "markdown",
            },
            timeout=10.0,
        )
        latency = int((time.monotonic() - started) * 1000)
        status = _classify_http_status(resp.status_code)
        quota = _extract_jina_quota(resp.headers) if resp.status_code < 500 else None
        if status == HealthStatus.ok and quota and quota.near_limit:
            status = HealthStatus.near_quota_limit
        detail = None if status == HealthStatus.ok else f"Jina HTTP {resp.status_code}"
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        status = HealthStatus.unreachable
        detail = f"Jina: {exc}"
        quota = None
    return _item(
        component="jina_reader",
        task="reader-based web scraping",
        provider="jina",
        model="jina-reader",
        configured=True,
        status=status,
        detail=detail,
        category="scraper",
        latency_ms=latency,
        quota=quota,
    )


async def _probe_firecrawl_item() -> ModelHealthItem:
    if not ext.firecrawl_api_key:
        return _item(
            component="firecrawl",
            task="scrape backend (firecrawl)",
            provider="firecrawl",
            model="firecrawl",
            configured=False,
            status=HealthStatus.not_configured,
            detail="firecrawl key not configured",
            category="scraper",
        )
    url = ext.firecrawl_api_url.rstrip("/") + "/v1/scrape"
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20)) as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {ext.firecrawl_api_key}",
                    "Content-Type": "application/json",
                },
                json={"url": "https://example.com", "formats": ["markdown"]},
            )
        latency = int((time.monotonic() - started) * 1000)
        status = _classify_http_status(resp.status_code)
        quota = _extract_jina_quota(resp.headers) if resp.status_code < 500 else None
        if status == HealthStatus.ok and quota and quota.near_limit:
            status = HealthStatus.near_quota_limit
        detail = None if status == HealthStatus.ok else f"Firecrawl HTTP {resp.status_code}"
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        status = HealthStatus.unreachable
        detail = f"Firecrawl: {exc}"
        quota = None
    return _item(
        component="firecrawl",
        task="scrape backend (firecrawl)",
        provider="firecrawl",
        model="firecrawl",
        configured=True,
        status=status,
        detail=detail,
        category="scraper",
        latency_ms=latency,
        quota=quota,
    )


async def _probe_llamaparse_item() -> ModelHealthItem:
    if not ext.llama_cloud_api_key:
        return _item(
            component="llamaparse",
            task="cloud document parsing",
            provider="llamacloud",
            model="llamaparse",
            configured=False,
            status=HealthStatus.disabled,
            detail="local parser backend active",
            category="parser",
        )
    # GET /job/<uuid> returns 404 with a valid key, 401 with a bad key —
    # cheapest way to probe authentication without uploading a document.
    probe_uuid = "00000000-0000-0000-0000-000000000000"
    url = ext.llama_cloud_base_url.rstrip("/") + f"/job/{probe_uuid}"
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15)) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {ext.llama_cloud_api_key}"},
            )
        latency = int((time.monotonic() - started) * 1000)
        # 404 here means "auth OK, resource missing" — that's a healthy signal.
        if resp.status_code == 404:
            status = HealthStatus.ok
            detail = None
        else:
            status = _classify_http_status(resp.status_code)
            detail = None if status == HealthStatus.ok else f"LlamaParse HTTP {resp.status_code}"
    except Exception as exc:
        latency = int((time.monotonic() - started) * 1000)
        status = HealthStatus.unreachable
        detail = f"LlamaParse: {exc}"
    return _item(
        component="llamaparse",
        task="cloud document parsing",
        provider="llamacloud",
        model="llamaparse",
        configured=True,
        status=status,
        detail=detail,
        category="parser",
        latency_ms=latency,
    )


def _extra_providers_in_use() -> dict[str, str]:
    """Return {provider: model} for non-openai providers referenced by a task model.

    Only includes providers that are actually wired up via DP_CLASSIFIER_MODEL,
    DP_CONTEXTUAL_MODEL, or DP_FUNDING_MODEL. We probe each provider once with
    the first model seen for it.
    """
    from app.services.intelligence.llm_router import parse_spec
    specs = [ext.classifier_model, ext.contextual_model, ext.funding_model]
    seen: dict[str, str] = {}
    for spec in specs:
        if not spec:
            continue
        try:
            provider, model = parse_spec(spec)
        except Exception:
            continue
        if provider != "openai" and provider not in seen:
            seen[provider] = model
    return seen


async def _probe_extra_provider_item(provider: str, model: str) -> ModelHealthItem:
    from app.services.intelligence.llm_router import KNOWN_PROVIDERS
    api_key = getattr(ext, f"{provider}_api_key", "") or ""
    base_url = (
        (getattr(ext, f"{provider}_base_url", "") or "").rstrip("/")
        or KNOWN_PROVIDERS.get(provider, "")
    )
    if not api_key:
        return _item(
            component=f"llm_provider_{provider}",
            task="per-task LLM (classify / contextual / funding)",
            provider=provider,
            model=model,
            configured=False,
            status=HealthStatus.not_configured,
            detail=f"{provider} key not configured (referenced by a task model)",
            category="llm",
        )
    if not base_url:
        return _item(
            component=f"llm_provider_{provider}",
            task="per-task LLM (classify / contextual / funding)",
            provider=provider,
            model=model,
            configured=True,
            status=HealthStatus.unreachable,
            detail=f"no base_url configured for provider {provider!r}",
            category="llm",
        )
    status, detail, quota, latency = await _probe_chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        provider_label=provider,
    )
    return _item(
        component=f"llm_provider_{provider}",
        task="per-task LLM (classify / contextual / funding)",
        provider=provider,
        model=model,
        configured=True,
        status=status,
        detail=detail,
        category="llm",
        latency_ms=latency,
        quota=quota,
    )


async def _probe_state_service(
    *,
    component: str,
    task: str,
    provider: str,
    model: str,
    category: str,
    svc: object | None,
    method_name: str = "check_health",
    args: tuple = (),
    not_configured_detail: str = "service not initialized",
) -> ModelHealthItem:
    """Probe a service attached to app.state via its no-arg (or fixed-arg)
    check method (e.g. check_health, ping, is_ready)."""
    if svc is None:
        return _item(
            component=component,
            task=task,
            provider=provider,
            model=model,
            configured=False,
            status=HealthStatus.not_configured,
            detail=not_configured_detail,
            category=category,
        )
    started = time.monotonic()
    detail: str | None = None
    status: HealthStatus
    try:
        method = getattr(svc, method_name, None)
        if method is None:
            status = HealthStatus.not_configured
            detail = f"missing {method_name}()"
        else:
            result = await method(*args)
            status = HealthStatus.ok if bool(result) else HealthStatus.unreachable
            if not bool(result):
                detail = f"{method_name}() returned false"
    except Exception as exc:
        status = HealthStatus.unreachable
        detail = str(exc)
    latency = int((time.monotonic() - started) * 1000)
    return _item(
        component=component,
        task=task,
        provider=provider,
        model=model,
        configured=True,
        status=status,
        detail=detail,
        category=category,
        latency_ms=latency,
    )


async def _probe_tei_embed_at_item(request: Request) -> ModelHealthItem:
    tei = getattr(request.app.state, "tei_embedder_at", None)
    return await _probe_state_service(
        component="tei_embed_at",
        task="BGE-M3 online embedding (dense_bge_m3) via TEI",
        provider="tei",
        model=getattr(tei, "_model", ext.tei_embed_model_at) if tei else ext.tei_embed_model_at,
        category="embedding",
        svc=tei,
    )


async def _probe_tei_sparse_at_item(request: Request) -> ModelHealthItem:
    sparse = getattr(request.app.state, "sparse_embedder", None)
    return await _probe_state_service(
        component="tei_sparse_at",
        task="TEI sparse embedding for hybrid search",
        provider="tei",
        model=getattr(sparse, "_model", ext.sparse_embed_model_at) if sparse else ext.sparse_embed_model_at,
        category="embedding",
        svc=sparse,
    )


async def _probe_qdrant_item(request: Request, *, at: bool) -> ModelHealthItem:
    if at:
        qdrant = getattr(request.app.state, "qdrant_at", None)
        component, task, model = "qdrant_at", "Qdrant (AT funding instance)", "qdrant"
    else:
        qdrant = getattr(request.app.state, "qdrant", None)
        component, task, model = "qdrant", "Qdrant (default instance)", "qdrant"
    return await _probe_state_service(
        component=component,
        task=task,
        provider="qdrant",
        model=model,
        category="vector_db",
        svc=qdrant,
    )


async def _probe_redis_item(request: Request) -> ModelHealthItem:
    cache = getattr(request.app.state, "cache", None)
    return await _probe_state_service(
        component="redis",
        task="cache",
        provider="redis",
        model="redis",
        category="cache",
        svc=cache,
        method_name="ping",
    )


async def _probe_clickhouse_item(request: Request) -> ModelHealthItem:
    audit = getattr(request.app.state, "audit", None)
    return await _probe_state_service(
        component="clickhouse",
        task="audit log",
        provider="clickhouse",
        model="clickhouse",
        category="audit",
        svc=audit,
        not_configured_detail="audit logger not initialized",
    )


async def _probe_ldap_item(request: Request) -> ModelHealthItem:
    ldap = getattr(request.app.state, "ldap", None)
    if ldap is None and not ext.ldap_url:
        return _item(
            component="ldap",
            task="directory auth",
            provider="ldap",
            model="ldap",
            configured=False,
            status=HealthStatus.not_configured,
            detail="ldap not configured",
            category="directory",
        )
    return await _probe_state_service(
        component="ldap",
        task="directory auth",
        provider="ldap",
        model="ldap",
        category="directory",
        svc=ldap,
    )


def _aggregate(items: list[ModelHealthItem]) -> tuple[HealthStatus, bool, bool]:
    """Compute (overall, any_quota_alerts, any_auth_failures) from the per-item statuses."""
    required = [i for i in items if i.required]

    # 'down'-class — surface the single worst one as `overall` for quick scanning.
    for st in (HealthStatus.unreachable, HealthStatus.auth_failed, HealthStatus.quota_exhausted):
        if any(i.status == st for i in required):
            overall = st
            break
    else:
        if any(
            i.status in (HealthStatus.near_quota_limit, HealthStatus.rate_limited, HealthStatus.degraded)
            for i in items
        ):
            overall = HealthStatus.degraded
        else:
            overall = HealthStatus.ok

    any_quota_alerts = any(
        i.status
        in (HealthStatus.near_quota_limit, HealthStatus.rate_limited, HealthStatus.quota_exhausted)
        for i in items
    )
    any_auth_failures = any(i.status == HealthStatus.auth_failed for i in items)
    return overall, any_quota_alerts, any_auth_failures


@router.get(
    "/model-health",
    response_model=ModelHealthResponse,
    dependencies=[Depends(require_api_key)],
    tags=["Diagnostics"],
    summary="Detailed health check for every third-party dependency",
    description=(
        "**Use this when you want a one-glance picture of every third-party dependency** — "
        "API keys, quota state, reachability. Returns a structured per-component report "
        "and a top-level `overall` status.\n\n"
        "## When to call\n"
        "- Operator dashboard / on-call refresh — \"is anything broken right now?\"\n"
        "- Pre-flight before a big batch ingest — confirm dependencies are healthy.\n"
        "- Alerting cron — page when `any_auth_failures: true` or `overall != \"ok\"`.\n"
        "- **Do not** put this on a tight polling loop (every 5s, etc.) — each call burns a "
        "tiny amount of provider budget.\n\n"
        "## Reading the response\n"
        "- `overall` — worst-required status (`ok` / `degraded` / `auth_failed` / `quota_exhausted` / `unreachable`).\n"
        "- `any_quota_alerts` — `true` if any component is `near_quota_limit`, `rate_limited`, or `quota_exhausted`.\n"
        "- `any_auth_failures` — `true` if any component is `auth_failed` (likely a rotated/expired key or lapsed subscription).\n"
        "- `models[i].status` — granular per-component status.\n"
        "- `models[i].quota` — `remaining_tokens` / `remaining_requests` / `reset_at` (only providers that expose them).\n"
        "- `models[i].detail` — short error string.\n\n"
        "## Caching\n"
        "Results are cached in-memory for 60s (see `cache_ttl_seconds`). Pass `?force=true` to "
        "skip the cache and re-probe everything. `served_from_cache: true` in the response "
        "tells you when you got a cached copy.\n\n"
        "**Auth:** requires `X-API-Key` when API-key auth is configured."
    ),
    response_description="Per-dependency health classification with optional quota snapshots",
)
async def model_health(request: Request, force: bool = False) -> ModelHealthResponse:
    uptime = _uptime(request)

    if not force:
        cached = _model_health_cache.get(_MODEL_HEALTH_CACHE_KEY)
        if cached and (time.monotonic() - cached[0]) < _MODEL_HEALTH_CACHE_TTL_S:
            return cached[1].model_copy(update={"uptime_seconds": uptime, "served_from_cache": True})

    items: list[ModelHealthItem] = []

    # ── LLM providers (chat) ─────────────────────────────────────────
    items.append(await _probe_openai_chat_item())
    for provider, model in sorted(_extra_providers_in_use().items()):
        items.append(await _probe_extra_provider_item(provider, model))

    # ── Embedders ────────────────────────────────────────────────────
    items.append(await _probe_openai_embeddings_item())
    items.append(await _probe_tei_embed_at_item(request))
    items.append(await _probe_tei_sparse_at_item(request))

    # ── Scrapers / parser ────────────────────────────────────────────
    items.append(await _probe_jina_item(request))
    if ext.firecrawl_api_key:
        items.append(await _probe_firecrawl_item())
    items.append(await _probe_llamaparse_item())

    # ── Infrastructure ───────────────────────────────────────────────
    items.append(await _probe_qdrant_item(request, at=False))
    items.append(await _probe_qdrant_item(request, at=True))
    items.append(await _probe_redis_item(request))
    items.append(await _probe_clickhouse_item(request))
    items.append(await _probe_ldap_item(request))

    overall, any_quota_alerts, any_auth_failures = _aggregate(items)

    response = ModelHealthResponse(
        overall=overall,
        any_quota_alerts=any_quota_alerts,
        any_auth_failures=any_auth_failures,
        healthy=(overall == HealthStatus.ok),
        models=items,
        mode=settings.mode,
        tenant_id=settings.tenant_id,
        worker_id=settings.worker_id,
        version=settings.version,
        uptime_seconds=uptime,
        cache_ttl_seconds=_MODEL_HEALTH_CACHE_TTL_S,
        served_from_cache=False,
    )
    _model_health_cache[_MODEL_HEALTH_CACHE_KEY] = (time.monotonic(), response)
    return response
