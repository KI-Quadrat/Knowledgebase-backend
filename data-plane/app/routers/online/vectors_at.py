"""DELETE /api/v1/online/vectors/at/{source_id} — Remove AT-pipeline vectors.

Dedicated delete endpoint for the Austrian funding-assistant pipeline. Same
signature as ``DELETE /api/v1/online/vectors/{source_id}`` — only the target
Qdrant instance differs. Runs against ``app.state.qdrant_at`` (configured via
``QDRANT_URL_AT`` / ``QDRANT_PORT_AT`` / ``QDRANT_API_KEY_AT``).
"""

from fastapi import APIRouter, Query, Request

from app.models.common import ErrorCode, ResponseEnvelope
from app.models.online.vectors import OnlineDeleteVectorsATData
from app.services.embedding.qdrant_service import QdrantError
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/online", tags=["Online - Vector Management (AT)"])


@router.delete(
    "/vectors/at/{source_id}",
    summary="Delete all AT vectors for a document",
    description=(
        "Remove all vector points associated with a `source_id` from the specified "
        "Qdrant collection on the AT Qdrant instance (`QDRANT_URL_AT` / "
        "`QDRANT_PORT_AT` / `QDRANT_API_KEY_AT`).\n\n"
        "Mirrors `DELETE /api/v1/online/vectors/{source_id}` — same signature, "
        "only the target Qdrant instance differs.\n\n"
        "**Required query parameter:** `collection_name` — one of the nine AT "
        "province collections (`Burgenland`, `Kärnten`, `Niederösterreich`, "
        "`Oberösterreich`, `Salzburg`, `Steiermark`, `Tirol`, `Vorarlberg`, `Wien`).\n\n"
        "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.\n\n"
        "**Error codes:** `QDRANT_CONNECTION_FAILED`, `QDRANT_DELETE_FAILED`"
    ),
    response_description="Deletion confirmation with count of removed vectors",
)
async def delete_vectors_at(
    source_id: str,
    request: Request,
    collection_name: str = Query(..., description="AT Qdrant collection name"),
) -> ResponseEnvelope[OnlineDeleteVectorsATData]:
    request_id = request.state.request_id
    qdrant = request.app.state.qdrant_at

    try:
        deleted = await qdrant.delete_by_source_id(collection_name, source_id)
    except QdrantError as e:
        error_msg = str(e).lower()
        error_code = (
            ErrorCode.QDRANT_CONNECTION_FAILED
            if "connection" in error_msg
            else ErrorCode.QDRANT_DELETE_FAILED
        )
        log.error(
            "online_vectors_at_delete_failed",
            source_id=source_id,
            collection=collection_name,
            error=str(e),
        )
        return ResponseEnvelope(
            success=False,
            error=error_code,
            detail=str(e),
            request_id=request_id,
        )

    return ResponseEnvelope(
        success=True,
        data=OnlineDeleteVectorsATData(source_id=source_id, vectors_deleted=deleted),
        request_id=request_id,
    )
