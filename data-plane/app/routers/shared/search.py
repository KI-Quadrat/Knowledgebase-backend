"""
POST /api/v1/search — Semantic or hybrid search with mandatory permission filtering.
"""

from fastapi import APIRouter, Request

from app.models.common import ErrorCode, ResponseEnvelope
from app.models.search import (
    PermissionFilterApplied,
    SearchData,
    SearchRequest,
    SearchResult,
    SearchResultEntities,
    SearchResultMetadata,
)
from app.services.search.search_service import SearchError
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Semantic Search"])

_ERROR_CODE_MAP = {
    "VALIDATION_USER_REQUIRED": ErrorCode.VALIDATION_USER_REQUIRED,
    "EMBEDDING_MODEL_NOT_LOADED": ErrorCode.EMBEDDING_MODEL_NOT_LOADED,
    "EMBEDDING_FAILED": ErrorCode.EMBEDDING_FAILED,
    "QDRANT_CONNECTION_FAILED": ErrorCode.QDRANT_CONNECTION_FAILED,
    "QDRANT_COLLECTION_NOT_FOUND": ErrorCode.QDRANT_COLLECTION_NOT_FOUND,
    "QDRANT_SEARCH_FAILED": ErrorCode.QDRANT_SEARCH_FAILED,
}


@router.post(
    "/search",
    summary="Permission-aware semantic or hybrid search",
    description=(
        "Search a specific Qdrant `collection_name` with **mandatory permission filtering** based on user identity.\n\n"
        "**Search modes** (via `search_mode`):\n"
        "- `semantic` (default) — dense-only cosine search via OpenAI `text-embedding-3-small` "
        "(automatic fallback to BGE-Gemma2 via LiteLLM when OpenAI is unavailable)\n"
        "- `hybrid` — dense (OpenAI or BGE-Gemma2 fallback) + sparse (BM25) combined with "
        "**Reciprocal Rank Fusion (RRF)**. "
        "Requires the collection to have been created with `search_mode: hybrid` during ingestion.\n\n"
        "**Embedding fallback:** The search service first attempts to embed the query via OpenAI "
        "(`dense_openai` vector). If OpenAI is down, it automatically falls back to BGE-Gemma2 via "
        "LiteLLM (`dense_bge_gemma2` vector).\n\n"
        "**Multi-tenant:** The caller specifies which `collection_name` to search in.\n\n"
        "**Permission model:**\n"
        "- `citizen` — Can only see documents with `visibility: public`\n"
        "- `employee` — Can see `public` + `internal` documents filtered by AD group membership\n\n"
        "**The search pipeline:**\n"
        "1. Embed the query via OpenAI `text-embedding-3-small` (1536-dim) — falls back to BGE-Gemma2 via LiteLLM if OpenAI fails\n"
        "2. (Hybrid only) Encode query with BM25 for sparse vector\n"
        "3. Build Qdrant filter from user permissions (visibility + group intersection)\n"
        "4. Execute search against `dense_openai` or `dense_bge_gemma2` — semantic (nearest-neighbor) or hybrid (RRF fusion of dense + sparse)\n"
        "5. Return ranked results with transparency on which filters were applied\n\n"
        "**Optional filters:** `content_type` (e.g. `funding`, `event`, `policy`)\n\n"
        "**Error codes:** `VALIDATION_USER_REQUIRED`, `EMBEDDING_MODEL_NOT_LOADED`, `EMBEDDING_FAILED`, "
        "`QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_SEARCH_FAILED`"
    ),
    response_description="Ranked search results with scores, metadata, and permission filter transparency",
)
async def search(body: SearchRequest, request: Request) -> ResponseEnvelope[SearchData]:
    request_id = request.state.request_id
    search_svc = request.app.state.search

    classification_filter = None
    if body.filters and body.filters.content_type:
        classification_filter = body.filters.content_type

    try:
        result = await search_svc.search(
            query=body.query,
            collection_name=body.collection_name,
            user_type=body.user.type,
            user_id=body.user.user_id,
            user_groups=body.user.groups,
            user_roles=body.user.roles,
            user_department=body.user.department,
            classification_filter=classification_filter,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
            search_mode=body.search_mode,
            enable_fallback=body.enable_fallback,
        )
    except SearchError as e:
        error_code = _ERROR_CODE_MAP.get(e.code, ErrorCode.QDRANT_SEARCH_FAILED)
        log.error("search_failed", error=str(e), code=e.code)
        return ResponseEnvelope(
            success=False,
            error=error_code,
            detail=str(e),
            request_id=request_id,
        )

    search_results = [
        SearchResult(
            chunk_id=r.chunk_id,
            source_id=r.source_id,
            chunk_text=r.chunk_text,
            score=r.score,
            source_path=r.source_path,
            content_type=r.classification,
            entities=SearchResultEntities(
                amounts=r.entity_amounts,
                deadlines=r.entity_deadlines,
            ),
            metadata=SearchResultMetadata(
                title=r.title,
                municipality_id=r.municipality_id,
                department=r.department,
                source_type=r.source_type,
            ),
        )
        for r in result.results
    ]

    return ResponseEnvelope(
        success=True,
        data=SearchData(
            results=search_results,
            total_results=result.total_results,
            query_embedding_ms=result.query_embedding_ms,
            search_ms=result.search_ms,
            search_mode=result.search_mode,
            permission_filter_applied=PermissionFilterApplied(
                visibility=result.permission_filter.visibility,
                must_match_groups=result.permission_filter.must_match_groups,
                must_not_match_groups=result.permission_filter.must_not_match_groups,
            ),
        ),
        request_id=request_id,
    )
