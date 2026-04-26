"""
GET /api/v1/health — Liveness check (no auth)
GET /api/v1/ready  — Readiness check (minimal without auth, full with HMAC)
"""

import time

import httpx
from fastapi import APIRouter, Depends, Request

from app.config import ext, settings
from app.dependencies.api_key import require_api_key
from app.models.health import (
    HealthResponse,
    ModelHealthItem,
    ModelHealthResponse,
    ReadyResponse,
    ServiceStatus,
)
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Health"])
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


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
        return False, "OPENAI_API_KEY not configured"

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
    crawl_client = getattr(scraping_svc, "crawl4ai", None)
    http_client = getattr(crawl_client, "_client", None)
    jina_key = getattr(crawl_client, "_jina_key", "")
    jina_url = getattr(crawl_client, "_jina_url", "")

    if not crawl_client or not http_client:
        return False, "service not initialized"
    if not jina_key:
        return False, "JINA_API_KEY not configured"

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


def _item(
    *,
    component: str,
    task: str,
    provider: str,
    model: str,
    configured: bool,
    healthy: bool,
    detail: str | None = None,
) -> ModelHealthItem:
    return ModelHealthItem(
        component=component,
        task=task,
        provider=provider,
        model=model,
        configured=configured,
        healthy=healthy,
        required=configured,
        detail=detail,
    )


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

**Without HMAC auth headers:** Returns minimal `{ready: true/false}` — suitable for load balancer health checks.

**With HMAC auth headers (X-Signature + X-Timestamp):** Returns full dependency status including Qdrant, BGE-M3, OpenAI embedder, TEI BGE-M3 (embed.ki2.at), TEI sparse (sparse.ki2.at), Parser (LlamaParse/Unstructured), Crawl4AI, LDAP, and Redis.

Core services that must be healthy for `ready: true`: Qdrant, BGE-M3, Parser, Crawl4AI.""",
    response_description="Readiness status with optional service details",
)
async def ready(request: Request) -> ReadyResponse:
    has_auth = bool(request.headers.get("X-Signature"))
    uptime = _uptime(request)

    if not has_auth:
        return ReadyResponse(ready=True, uptime_seconds=uptime)

    services = ServiceStatus()

    # Crawl4AI
    scraping_svc = getattr(request.app.state, "scraping", None)
    if scraping_svc:
        services.crawl4ai = getattr(scraping_svc, "is_ready", False)

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
        services.crawl4ai,
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


@router.get(
    "/model-health",
    response_model=ModelHealthResponse,
    dependencies=[Depends(require_api_key)],
    summary="Model health check",
    description=(
        "Checks every configured model-bearing component used by this project. "
        "Embedding services receive a tiny probe input, the shared OpenAI chat "
        "model is tested with a minimal completion, and LlamaParse is checked "
        "when enabled."
    ),
    response_description="Detailed status for each configured model component",
)
async def model_health(request: Request) -> ModelHealthResponse:
    uptime = _uptime(request)
    models: list[ModelHealthItem] = []

    scraping_svc = getattr(request.app.state, "scraping", None)
    crawl4ai_client = getattr(scraping_svc, "crawl4ai", None)
    crawl4ai_ok, crawl4ai_detail = (
        (await crawl4ai_client.check_health(), None)
        if crawl4ai_client
        else (False, "service not initialized")
    )
    models.append(
        _item(
            component="scraper_crawl4ai",
            task="javascript-rendered web scraping",
            provider="crawl4ai",
            model="crawl4ai",
            configured=crawl4ai_client is not None,
            healthy=crawl4ai_ok,
            detail=crawl4ai_detail,
        )
    )

    jina_configured = bool(crawl4ai_client and getattr(crawl4ai_client, "_jina_key", ""))
    jina_ok, jina_detail = (
        await _probe_jina_reader(scraping_svc)
        if jina_configured
        else (False, "JINA_API_KEY not configured" if crawl4ai_client else "service not initialized")
    )
    models.append(
        _item(
            component="scraper_jina_reader",
            task="reader-based web scraping fallback",
            provider="jina",
            model="jina-reader",
            configured=jina_configured,
            healthy=jina_ok,
            detail=jina_detail,
        )
    )

    embedder = getattr(request.app.state, "embedder", None)
    local_ok, local_detail = await _probe_component(embedder, "embed") if embedder else (False, "service not initialized")
    models.append(
        _item(
            component="local_embedding",
            task="local document embeddings",
            provider="bge-m3 service",
            model="bge-m3",
            configured=embedder is not None,
            healthy=local_ok,
            detail=local_detail,
        )
    )

    openai_embedder = getattr(request.app.state, "openai_embedder", None)
    openai_configured = bool(openai_embedder and getattr(openai_embedder, "_api_key", ""))
    openai_embed_ok, openai_embed_detail = (
        await _probe_component(openai_embedder, "embed")
        if openai_configured
        else (False, "OPENAI_API_KEY not configured" if openai_embedder else "service not initialized")
    )
    models.append(
        _item(
            component="online_embedding_openai",
            task="OpenAI online embedding (dense_openai)",
            provider="openai",
            model=getattr(openai_embedder, "_model", "text-embedding-3-small"),
            configured=openai_configured,
            healthy=openai_embed_ok,
            detail=openai_embed_detail,
        )
    )

    tei_embedder_at = getattr(request.app.state, "tei_embedder_at", None)
    tei_configured = bool(tei_embedder_at and getattr(tei_embedder_at, "_api_key", ""))
    tei_ok, tei_detail = (
        await _probe_component(tei_embedder_at, "embed")
        if tei_configured
        else (False, "TEI_EMBED_API_KEY_AT not configured" if tei_embedder_at else "service not initialized")
    )
    models.append(
        _item(
            component="online_embedding_bge_m3",
            task="BGE-M3 online embedding (dense_bge_m3)",
            provider="tei",
            model=getattr(tei_embedder_at, "_model", ext.tei_embed_model_at),
            configured=tei_configured,
            healthy=tei_ok,
            detail=tei_detail,
        )
    )

    sparse_embedder = getattr(request.app.state, "sparse_embedder", None)
    sparse_configured = bool(sparse_embedder and getattr(sparse_embedder, "_api_key", ""))
    sparse_ok, sparse_detail = (
        await _probe_component(sparse_embedder, "encode")
        if sparse_configured
        else (False, "SPARSE_EMBED_API_KEY_AT not configured" if sparse_embedder else "service not initialized")
    )
    models.append(
        _item(
            component="online_embedding_sparse",
            task="TEI sparse embedding for hybrid search",
            provider="tei",
            model=getattr(sparse_embedder, "_model", ext.sparse_embed_model_at),
            configured=sparse_configured,
            healthy=sparse_ok,
            detail=sparse_detail,
        )
    )

    classifier = getattr(request.app.state, "classifier", None)
    llm_classifier = getattr(classifier, "_llm", None)
    chat_configured = bool(llm_classifier and getattr(llm_classifier, "_client", None))
    chat_ok, chat_detail = await _probe_openai_chat_model() if chat_configured else (False, "OPENAI_API_KEY not configured")

    models.append(
        _item(
            component="content_classifier",
            task="content classification",
            provider="openai",
            model=getattr(llm_classifier, "_model", ext.openai_model),
            configured=chat_configured,
            healthy=chat_ok,
            detail=chat_detail,
        )
    )

    contextual = getattr(request.app.state, "contextual_enricher", None)
    contextual_configured = bool(contextual and getattr(contextual, "_api_key", ""))
    models.append(
        _item(
            component="contextual_enricher",
            task="contextual chunk enrichment",
            provider="openai",
            model=getattr(contextual, "_model", ext.openai_model),
            configured=contextual_configured,
            healthy=chat_ok if contextual_configured else False,
            detail=chat_detail if contextual_configured else "OPENAI_API_KEY not configured",
        )
    )

    funding = getattr(request.app.state, "funding_extractor", None)
    funding_configured = bool(funding and getattr(funding, "_client", None))
    models.append(
        _item(
            component="funding_extractor",
            task="funding metadata extraction",
            provider="openai",
            model=getattr(funding, "_model", ext.openai_model),
            configured=funding_configured,
            healthy=chat_ok if funding_configured else False,
            detail=chat_detail if funding_configured else "OPENAI_API_KEY not configured",
        )
    )

    parser = getattr(request.app.state, "parser", None)
    parser_backend = getattr(parser, "parser_backend", "local") if parser else "local"
    llama_enabled = parser is not None and parser_backend == "llamaparse"
    if llama_enabled:
        try:
            llama_ok = await parser.check_health()
            llama_detail = None
        except Exception as exc:
            llama_ok = False
            llama_detail = str(exc)
    else:
        llama_ok = False
        llama_detail = "local parser backend active"
    models.append(
        _item(
            component="document_parser",
            task="cloud document parsing",
            provider="llamacloud" if llama_enabled else "local",
            model="llamaparse" if llama_enabled else "local-parser",
            configured=llama_enabled,
            healthy=llama_ok if llama_enabled else False,
            detail=llama_detail,
        )
    )

    all_healthy = all(item.healthy for item in models if item.required)

    return ModelHealthResponse(
        healthy=all_healthy,
        models=models,
        mode=settings.mode,
        tenant_id=settings.tenant_id,
        worker_id=settings.worker_id,
        version=settings.version,
        uptime_seconds=uptime,
    )
