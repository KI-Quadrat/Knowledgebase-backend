"""
DELETE /api/v1/local/vectors/{source_id}      — Remove all vectors for a document
POST   /api/v1/local/vectors/delete-by-filter — Remove vectors matching metadata filters
PUT    /api/v1/local/vectors/update-acl       — Update ACL on existing vectors
"""

from fastapi import APIRouter, Query, Request

from app.models.common import ErrorCode, ResponseEnvelope
from app.models.local.vectors import (
    DeleteByFilterData,
    DeleteByFilterRequest,
    DeleteVectorsData,
    UpdateACLData,
    UpdateACLRequest,
)
from app.services.embedding.qdrant_service import QdrantError
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/local", tags=["Local - Vector Management"])


@router.delete(
    "/vectors/{source_id}",
    summary="Delete all vectors for a document",
    description=(
        "Remove all vector points associated with a `source_id` from the specified Qdrant collection.\n\n"
        "**Required query parameter:** `collection_name` — the Qdrant collection to delete from.\n\n"
        "Used when a document is deleted from the source system or before re-ingesting updated content. "
        "The ingest pipeline calls this automatically during re-ingestion (idempotent upsert).\n\n"
        "**Error codes:** `QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`"
    ),
    response_description="Deletion confirmation with count of removed vectors",
)
async def delete_vectors(
    source_id: str,
    request: Request,
    collection_name: str = Query(..., description="Qdrant collection name"),
) -> ResponseEnvelope[DeleteVectorsData]:
    request_id = request.state.request_id
    qdrant = request.app.state.qdrant
    collection = collection_name

    try:
        deleted = await qdrant.delete_by_source_id(collection, source_id)
    except QdrantError as e:
        error_msg = str(e).lower()
        error_code = (
            ErrorCode.QDRANT_CONNECTION_FAILED
            if "connection" in error_msg
            else ErrorCode.QDRANT_DELETE_FAILED
        )
        log.error("vectors_delete_failed", source_id=source_id, error=str(e))
        return ResponseEnvelope(
            success=False,
            error=error_code,
            detail=str(e),
            request_id=request_id,
        )

    return ResponseEnvelope(
        success=True,
        data=DeleteVectorsData(source_id=source_id, vectors_deleted=deleted),
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
        "- `source_type` — Origin type (`smb`, `r2`, `web`)\n"
        "- `content_type` — Content categories (`funding`, `event`, `policy`, etc.)\n"
        "- `acl_visibility` — Visibility level (`public`, `internal`, `restricted`)\n"
        "- `acl_department` — Department tag\n"
        "- `municipality_id` — Municipality/tenant ID\n"
        "- `department` — Department from metadata\n"
        "- `language` — Document language\n"
        "- `uploaded_by` — Uploader ID\n"
        "- `mime_type` — File MIME type\n"
        "- `title` — Document title\n\n"
        "**Error codes:** `QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`"
    ),
    response_description="Deletion confirmation with count of removed vectors and filters applied",
)
async def delete_by_filter(body: DeleteByFilterRequest, request: Request) -> ResponseEnvelope[DeleteByFilterData]:
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
        log.error("vectors_delete_by_filter_failed", error=str(e), filters=[f.model_dump() for f in body.filters])
        return ResponseEnvelope(
            success=False,
            error=error_code,
            detail=str(e),
            request_id=request_id,
        )

    return ResponseEnvelope(
        success=True,
        data=DeleteByFilterData(
            vectors_deleted=deleted,
            filters_applied=body.filters,
        ),
        request_id=request_id,
    )


@router.put(
    "/vectors/update-acl",
    summary="Update ACL on existing vectors",
    description=(
        "Update the access control list (ACL) payload on all vectors belonging to a `source_id` "
        "in the specified `collection_name`.\n\n"
        "This is used when file permissions change on the source system (e.g. NTFS ACL update) "
        "without re-ingesting the content — avoids the cost of re-embedding.\n\n"
        "**Updated fields:** `acl_allow_groups`, `acl_deny_groups`, `acl_allow_roles`, `acl_allow_users`, "
        "`acl_visibility`, `acl_department`.\n\n"
        "**Error codes:** `QDRANT_CONNECTION_FAILED`, `QDRANT_UPSERT_FAILED`"
    ),
    response_description="Update confirmation with count of modified vectors",
)
async def update_acl(body: UpdateACLRequest, request: Request) -> ResponseEnvelope[UpdateACLData]:
    request_id = request.state.request_id
    qdrant = request.app.state.qdrant
    collection = body.collection_name

    acl_payload = {
        "metadata.acl_allow_groups": body.acl.allow_groups,
        "metadata.acl_deny_groups": body.acl.deny_groups,
        "metadata.acl_allow_roles": body.acl.allow_roles,
        "metadata.acl_allow_users": body.acl.allow_users,
        "metadata.acl_visibility": body.acl.visibility,
    }
    if body.acl.department is not None:
        acl_payload["metadata.acl_department"] = body.acl.department

    try:
        updated = await qdrant.update_payload(collection, body.source_id, acl_payload)
    except QdrantError as e:
        error_msg = str(e).lower()
        error_code = (
            ErrorCode.QDRANT_CONNECTION_FAILED
            if "connection" in error_msg
            else ErrorCode.QDRANT_UPSERT_FAILED
        )
        log.error("vectors_acl_update_failed", source_id=body.source_id, error=str(e))
        return ResponseEnvelope(
            success=False,
            error=error_code,
            detail=str(e),
            request_id=request_id,
        )

    return ResponseEnvelope(
        success=True,
        data=UpdateACLData(source_id=body.source_id, vectors_updated=updated),
        request_id=request_id,
    )
