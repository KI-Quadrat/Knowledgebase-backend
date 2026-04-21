"""
DELETE /api/v1/online/vectors/{source_id}      — Remove all vectors for a document
POST   /api/v1/online/vectors/delete-by-filter — Remove vectors matching metadata filters
POST   /api/v1/online/vectors/sparse-encode    — BM25 sparse-encode arbitrary text
"""

from fastapi import APIRouter, Query, Request

from app.models.common import ErrorCode, ResponseEnvelope
from app.models.online.vectors import (
    OnlineDeleteByFilterData,
    OnlineDeleteByFilterRequest,
    OnlineDeleteVectorsData,
    OnlineSparseEncodeData,
    OnlineSparseEncodeRequest,
)
from app.services.embedding.bm25_encoder import BM25Encoder
from app.services.embedding.qdrant_service import QdrantError
from app.utils.logger import get_logger

_bm25 = BM25Encoder()

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/online", tags=["Online - Vector Management"])


@router.delete(
    "/vectors/{source_id}",
    summary="Delete all vectors for a document",
    description=(
        "Remove all vector points associated with a `source_id` from the specified Qdrant collection.\n\n"
        "**Required query parameter:** `collection_name` — the Qdrant collection to delete from.\n\n"
        "**Error codes:** `QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`"
    ),
    response_description="Deletion confirmation with count of removed vectors",
)
async def delete_vectors(
    source_id: str,
    request: Request,
    collection_name: str = Query(..., description="Qdrant collection name"),
) -> ResponseEnvelope[OnlineDeleteVectorsData]:
    request_id = request.state.request_id
    qdrant = request.app.state.qdrant

    try:
        deleted = await qdrant.delete_by_source_id(collection_name, source_id)
    except QdrantError as e:
        error_msg = str(e).lower()
        error_code = (
            ErrorCode.QDRANT_CONNECTION_FAILED
            if "connection" in error_msg
            else ErrorCode.QDRANT_DELETE_FAILED
        )
        log.error("online_vectors_delete_failed", source_id=source_id, error=str(e))
        return ResponseEnvelope(
            success=False,
            error=error_code,
            detail=str(e),
            request_id=request_id,
        )

    return ResponseEnvelope(
        success=True,
        data=OnlineDeleteVectorsData(source_id=source_id, vectors_deleted=deleted),
        request_id=request_id,
    )


@router.post(
    "/vectors/delete-by-filter",
    summary="Delete vectors by metadata filter",
    description=(
        "Delete all vector points matching the given metadata filters from the specified Qdrant collection.\n\n"
        "All filters are combined with **AND** logic — only points matching every condition are deleted.\n\n"
        "**Filterable metadata fields:**\n"
        "- `source_id` — Document ID\n"
        "- `source_url` — Source URL of the ingested content\n"
        "- `source_type` — Origin type (`web`)\n"
        "- `content_type` — Content categories (`funding`, `event`, `policy`, etc.)\n"
        "- `assistant_id` — Assistant identifier\n"
        "- `municipality_id` — Municipality/tenant ID\n"
        "- `department` — Department\n"
        "- `language` — Document language\n"
        "- `uploaded_by` — Uploader ID\n"
        "- `mime_type` — File MIME type\n"
        "- `title` — Document title\n\n"
        "**Error codes:** `QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`"
    ),
    response_description="Deletion confirmation with count of removed vectors and filters applied",
)
async def delete_by_filter(
    body: OnlineDeleteByFilterRequest, request: Request,
) -> ResponseEnvelope[OnlineDeleteByFilterData]:
    request_id = request.state.request_id
    qdrant = request.app.state.qdrant

    # Build Qdrant filter — top-level fields stay at root, others are nested under metadata
    top_level_fields = {"municipality_id", "assistant_id", "department"}
    must_conditions = [
        {"key": f.key if f.key in top_level_fields else f"metadata.{f.key}", "match": {"value": f.value}}
        for f in body.filters
    ]
    qdrant_filter = {"must": must_conditions}

    try:
        deleted = await qdrant.delete_by_filter(body.collection_name, qdrant_filter)
    except QdrantError as e:
        error_msg = str(e).lower()
        error_code = (
            ErrorCode.QDRANT_CONNECTION_FAILED
            if "connection" in error_msg
            else ErrorCode.QDRANT_DELETE_FAILED
        )
        log.error("online_vectors_delete_by_filter_failed", error=str(e), filters=[f.model_dump() for f in body.filters])
        return ResponseEnvelope(
            success=False,
            error=error_code,
            detail=str(e),
            request_id=request_id,
        )

    return ResponseEnvelope(
        success=True,
        data=OnlineDeleteByFilterData(
            vectors_deleted=deleted,
            filters_applied=body.filters,
        ),
        request_id=request_id,
    )


@router.post(
    "/vectors/sparse-encode",
    summary="Encode text into a BM25 sparse vector",
    description=(
        "Run the same BM25 encoder used during `POST /online/ingest` (in `hybrid` "
        "search mode) and during hybrid search query encoding. Useful when a caller "
        "needs to reproduce the exact `sparse` vector that ingest would have stored, "
        "without going through the full ingest pipeline.\n\n"
        "**Tokenization:** lowercased, split on non-alphanumeric, German + English "
        "stopwords removed, single-character tokens dropped. Each surviving token is "
        "hashed (MD5 mod 2^31-1) into the sparse index space, and the value is the "
        "raw term frequency. Qdrant's IDF modifier on the collection handles the "
        "inverse-document-frequency weighting at query time.\n\n"
        "**Error codes:** `VALIDATION_EMPTY_CONTENT`"
    ),
    response_description="Sparse vector indices and term-frequency values",
)
async def sparse_encode(
    body: OnlineSparseEncodeRequest, request: Request,
) -> ResponseEnvelope[OnlineSparseEncodeData]:
    request_id = request.state.request_id

    if not body.content.strip():
        return ResponseEnvelope(
            success=False,
            error=ErrorCode.VALIDATION_EMPTY_CONTENT,
            detail="Content must not be empty",
            request_id=request_id,
        )

    sparse = _bm25.encode(body.content)
    return ResponseEnvelope(
        success=True,
        data=OnlineSparseEncodeData(
            indices=sparse["indices"],
            values=sparse["values"],
            term_count=len(sparse["indices"]),
        ),
        request_id=request_id,
    )
