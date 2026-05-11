"""
POST /api/v1/online/ingest — Ingest web-scraped content into the RAG pipeline.
"""

import asyncio

from fastapi import APIRouter, Request

from app.models.common import ErrorCode, ResponseEnvelope
from app.models.online.ingest import EmbeddingModel, OnlineIngestData, OnlineIngestRequest
from app.routers._ingest_utils import INGEST_ERROR_CODE_MAP
from app.services.ingest.ingest_service import IngestError
from app.services.intelligence.funding_extractor import normalize_provinces
from app.utils.logger import get_logger

# Per-model defaults for vector dim + stored vector field name.
#   openai → text-embedding-3-small (1536, "dense_openai")
#   bge_m3 → BGE-M3 via TEI endpoint (1024, "dense_bge_m3")
_EMBEDDING_MODEL_DEFAULTS: dict[EmbeddingModel, tuple[int, str]] = {
    EmbeddingModel.openai: (1536, "dense_openai"),
    EmbeddingModel.bge_m3: (1024, "dense_bge_m3"),
}


def _resolve_primary_embedder(app_state, model: EmbeddingModel):
    """Return (embedder_override, default_dim, vector_name) for the selected model.

    For the default ``openai`` path we return ``None`` so ``IngestService`` uses
    the embedder it was constructed with (the one wired up in lifespan) — this
    keeps the hot path unchanged and preserves backwards compatibility with
    tests that inject a fake IngestService.embedder directly.
    """
    dim, vector_name = _EMBEDDING_MODEL_DEFAULTS[model]
    if model == EmbeddingModel.bge_m3:
        return app_state.tei_embedder_at, dim, vector_name
    return None, dim, vector_name

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/online", tags=["Online - Ingestion Pipeline"])


@router.post(
    "/ingest",
    summary="Ingest web content into the RAG pipeline",
    description=(
        "Takes web-scraped or URL-parsed text content and processes it through the ingestion pipeline:\n\n"
        "1. **Chunk** — Split content using `contextual` (default), `late_chunking`, `sentence`, or `fixed` strategy\n"
        "2. **Contextual Enrichment** — (when using `contextual` strategy) Prepend AI-generated context to each chunk via OpenAI, improving retrieval accuracy\n"
        "3. **Embed** — Generate a dense vector via the model chosen by `embedding_model` (default `bge_m3`): "
        "`bge_m3` → BGE-M3 behind the TEI endpoint at `TEI_EMBED_URL_AT` (1024-dim, stored as `dense_bge_m3`); "
        "`openai` → `text-embedding-3-small` (1536-dim, stored as `dense_openai`).\n"
        "4. **Store** — Upsert vectors into the specified Qdrant `collection_name` with metadata\n\n"
        "**Content type is supplied by the caller.** The `content_type` field is **required** — "
        "obtain it upfront from `/online/scrape` or `/online/document-parse`, which now run the classifier "
        "and return `content_type` on their responses. Classification is no longer performed inside this endpoint. "
        "Content-type gating (e.g. skipping non-funding content when `assistant_type` is `\"funding\"`) "
        "is expected to be done by the caller before invoking ingest.\n\n"
        "**Vector layout:**\n"
        "Each point carries exactly one dense vector — `dense_openai` or `dense_bge_m3` depending on "
        "the chosen `embedding_model`. A collection is pinned to the model it was first ingested with.\n\n"
        "**Vector modes** (via `vector_config.search_mode`):\n"
        "- `semantic` (default) — dense cosine vector only.\n"
        "- `hybrid` — dense + `sparse` vector from the TEI sparse endpoint at `SPARSE_EMBED_URL_AT` "
        "(sparse.ki2.at). Enables combined semantic + lexical search.\n\n"
        "The collection is **auto-created** if it does not exist, using the specified vector size and search mode.\n\n"
        "**Funding metadata extraction:** When `assistant_type` is `\"funding\"`, "
        "an additional OpenAI call extracts structured funding metadata (`country_code`, `state_or_province`, `city`, "
        "`target_group`, `funding_type`, `status`, `funding_amount`, `thematic_focus`, `eligibility_criteria`, "
        "`legal_basis`, `funding_provider`, `application_form`, `reference_number`, `start_date`, `end_date`, `scraped_at`). "
        "`application_form` is a list of URLs to the program's application forms (PDF or online form pages); "
        "the extractor falls back to a verbatim form name when no URL is available. "
        "These fields are merged flat into each Qdrant point's metadata for filtering.\n\n"
        "**Country constraint:** The `country` field (ISO 3166-1 alpha-2) is **required** when `assistant_type` is `\"funding\"`. "
        "It constrains `state_or_province` to the official administrative divisions for that country "
        "(supported: AT, DE, CH, RO, IT, FR, HU, CZ, SK, SI, HR). Values not in the known list are dropped.\n\n"
        "Previous vectors for the same `source_id` are deleted before upserting (idempotent).\n\n"
        "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.\n\n"
        "**Error codes:** `VALIDATION_EMPTY_CONTENT`, `EMBEDDING_MODEL_NOT_LOADED`, "
        "`EMBEDDING_FAILED`, `EMBEDDING_OOM`, `QDRANT_CONNECTION_FAILED`, `QDRANT_COLLECTION_NOT_FOUND`, "
        "`QDRANT_UPSERT_FAILED`, `QDRANT_DISK_FULL`"
    ),
    response_description="Ingestion result with chunk count, vector count, classification, and timing",
)
async def ingest_online(body: OnlineIngestRequest, request: Request) -> ResponseEnvelope[OnlineIngestData]:
    request_id = request.state.request_id
    ingest_svc = request.app.state.online_ingest

    if not body.content.strip():
        return ResponseEnvelope(
            success=False,
            error=ErrorCode.VALIDATION_EMPTY_CONTENT,
            detail="Content must not be empty",
            request_id=request_id,
        )

    # ── Funding metadata extraction (only for funding assistant) ──
    # Launched as a task so it runs concurrently with chunking / contextual
    # enrichment / embedding inside ingest_svc.ingest(). The ingest service
    # awaits this task and merges its result just before building Qdrant points.
    funding_task: asyncio.Task | None = None
    if body.assistant_type == "funding":
        extractor = request.app.state.funding_extractor
        funding_task = asyncio.create_task(
            _safe_extract_funding(
                extractor,
                body.content,
                source_url=body.url,
                country=body.country,
                source_id=body.source_id,
            )
        )

    chunking = body.chunking
    vcfg = body.vector_config
    primary_embedder, default_dim, primary_vector_name = _resolve_primary_embedder(
        request.app.state, body.embedding_model
    )
    vector_size = vcfg.vector_size if (vcfg and vcfg.vector_size is not None) else default_dim
    # Request-supplied metadata wins over anything the funding extractor produces,
    # so build the request-side dict here and let the service merge deferred
    # funding fields under it.
    metadata_dict = body.metadata.model_dump()
    metadata_dict["source_url"] = body.url
    metadata_dict["assistant_type"] = body.assistant_type

    # Explicit state_or_province override from request body — passed through
    # the same normalize_provinces helper the extractor uses, so a caller
    # passing local-language names (e.g. "Bayern") lands in the same canonical
    # English-lowercase form ("bavaria") the extractor would have produced.
    # This guarantees search-time filters stay consistent across ingests
    # regardless of how the field was supplied.
    if body.state_or_province:
        metadata_dict["state_or_province"] = normalize_provinces(
            body.country, body.state_or_province
        )

    try:
        result = await ingest_svc.ingest(
            source_id=body.source_id,
            file_path=body.url,
            content=body.content,
            acl=None,
            metadata=metadata_dict,
            collection_name=body.collection_name,
            language=body.language,
            chunking_strategy=chunking.strategy if chunking else "contextual",
            max_chunk_size=chunking.max_chunk_size if chunking else None,
            chunk_overlap=chunking.overlap if chunking else None,
            vector_size=vector_size,
            search_mode=vcfg.search_mode.value if vcfg else "semantic",
            content_type=body.content_type,
            entities=body.entities.model_dump() if body.entities else None,
            deferred_metadata_task=funding_task,
            primary_embedder=primary_embedder,
            primary_vector_name=primary_vector_name,
        )
    except IngestError as e:
        if funding_task is not None and not funding_task.done():
            funding_task.cancel()
        error_code = INGEST_ERROR_CODE_MAP.get(e.code, ErrorCode.EMBEDDING_FAILED)
        log.error("ingest_online_failed", source_id=body.source_id, error=str(e), code=e.code)
        return ResponseEnvelope(
            success=False,
            error=error_code,
            detail=str(e),
            request_id=request_id,
        )

    return ResponseEnvelope(
        success=True,
        data=OnlineIngestData(
            source_id=result.source_id,
            chunks_created=result.chunks_created,
            vectors_stored=result.vectors_stored,
            collection=result.collection,
            content_type=result.classification,
            embedding_time_ms=result.embedding_time_ms,
            total_time_ms=result.total_time_ms,
        ),
        request_id=request_id,
    )


async def _safe_extract_funding(
    extractor, content: str, *, source_url: str, country: str | None, source_id: str
) -> dict:
    """Run funding extraction; swallow errors so they don't cancel the ingest task."""
    try:
        return await extractor.extract(content, source_url=source_url, country=country)
    except Exception as e:
        log.warning("ingest_online_funding_extract_failed", source_id=source_id, error=str(e))
        return {}
