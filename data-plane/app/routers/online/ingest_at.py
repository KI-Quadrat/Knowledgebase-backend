"""POST /api/v1/online/ingest/at — AT funding-assistant ingest.

Dedicated endpoint for the Austrian funding-assistant pipeline. Runs on a
separate Qdrant instance (``app.state.qdrant_at``) whose target collection is
pre-created with the legacy single-unnamed-vector schema sized for the TEI
embedding model behind ``TEI_EMBED_URL_AT``::

    {"vectors": {"size": 1024, "distance": "Cosine"}}

— no sparse, no multi-vector — with ``metadata.source_url`` indexed as a
keyword field. Because of that schema mismatch with the rest of the platform,
this endpoint drives the ingest directly instead of reusing
:class:`IngestService`.

Behaviour
---------
- Country is implicit — every request is treated as AT.
- Assistant type is implicit — the funding extractor always runs, so callers
  do not supply ``assistant_type``.
- Target collection: ``body.collection_name`` (required). Auto-created with
  the AT legacy schema (single unnamed 1024-dim cosine vector, keyword
  indexes on ``metadata.source_id`` and ``metadata.source_url``) on first
  use. Existing collections are reused; a dim mismatch fails fast with
  ``QDRANT_COLLECTION_NOT_FOUND``.
- ``state_or_province``: override wins over the extractor; both are stored on
  every point as ``metadata.state_or_province`` (english lowercase) for
  search-time filtering. No collection routing.
- Embeddings: TEI OpenAI-compatible endpoint (``TEI_EMBED_URL_AT``) with
  bearer auth. Dense-only, 1024-dim.
- Extra extracted metadata (``program_name``, ``processing_office``,
  ``contract_email``, ``contract_phone``, and the full funding-extractor
  output) lands on every point under ``metadata.*``.
- Idempotency: prior points for the same document are deleted via the
  indexed ``metadata.source_id`` field before the fresh upsert — a repeat
  ingest of the same ``source_id`` fully replaces the stored chunks.
"""

import hashlib
import time

from fastapi import APIRouter, Request

from app.models.common import ErrorCode, ResponseEnvelope
from app.models.online.ingest_at import OnlineIngestATData, OnlineIngestATRequest
from app.services.embedding.bge_m3_client import EmbeddingError
from app.services.embedding.qdrant_service import QdrantError
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/online", tags=["Online - Ingestion Pipeline (AT)"])

_AT_COUNTRY = "AT"
_AT_ASSISTANT_TYPE = "funding"

# Chunking tuned for German-language funding pages (gv.at / municipal sites).
# Those pages organize information into short sections — Fristen,
# Eligibility, Förderhöhe, Kontakt — and ~1000 chars usually captures one
# section whole. German averages ~5 chars/token → ~200 tokens per chunk,
# which sits comfortably inside BGE-M3's high-quality zone (100–512 tokens).
# Contextual enrichment adds ~150–300 chars of prefix, so the final embed
# length lands ~260 tokens — still ideal. Overlap of 150 catches single
# facts (deadlines, amounts) that straddle paragraph breaks.
_AT_DEFAULT_CHUNK_SIZE = 1000
_AT_DEFAULT_CHUNK_OVERLAP = 150


async def _safe_extract_funding(
    extractor, content: str, *, source_url: str, source_id: str
) -> dict:
    try:
        return await extractor.extract(content, source_url=source_url, country=_AT_COUNTRY)
    except Exception as e:
        log.warning("ingest_online_at_funding_extract_failed", source_id=source_id, error=str(e))
        return {}


def _point_id(source_id: str, chunk_index: int) -> int:
    """Deterministic 64-bit unsigned integer ID for a chunk.

    The AT collection uses integer point IDs. We derive ours from the low 64
    bits of ``sha256(source_id|chunk_index)`` so repeat ingests overwrite in
    place and the same chunk always lands on the same ID.
    """
    key = f"{source_id}|{chunk_index}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big")


_KNOWN_METADATA_KEYS = {
    "source_id", "source_url",
    "content_type", "language", "title", "source_type",
    "uploaded_by", "assistant_id", "municipality_id", "department",
    "assistant_type",
}


# German / English → canonical English-lowercase form. The canonical forms
# match funding_extractor.PROVINCES_BY_COUNTRY["AT"] exactly so overrides and
# extractor output produce the same stored metadata.state_or_province value
# regardless of caller input casing or language.
_AT_PROVINCE_ALIASES: dict[str, str] = {
    # English (canonical)
    "burgenland": "burgenland",
    "carinthia": "carinthia",
    "lower austria": "lower austria",
    "upper austria": "upper austria",
    "salzburg": "salzburg",
    "styria": "styria",
    "tyrol": "tyrol",
    "vorarlberg": "vorarlberg",
    "vienna": "vienna",
    # German
    "kärnten": "carinthia",
    "niederösterreich": "lower austria",
    "oberösterreich": "upper austria",
    "steiermark": "styria",
    "tirol": "tyrol",
    "wien": "vienna",
}


def _normalize_provinces(names: list[str] | None) -> list[str]:
    """Canonicalize AT province names to the same English-lowercase form the
    funding extractor emits.

    Accepts German or English input, any casing. Unknown values are dropped so
    the stored ``metadata.state_or_province`` is always a subset of the nine
    official AT provinces — search-time filters stay consistent across ingests.
    """
    if not names:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in names:
        canonical = _AT_PROVINCE_ALIASES.get((raw or "").strip().lower())
        if canonical and canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def _build_point(
    *,
    chunk_text: str,
    chunk_index: int,
    embedding: list[float],
    source_id: str,
    source_url: str,
    content_type: list[str],
    language: str,
    metadata: dict,
    entities: dict | None,
) -> dict:
    """Build a single Qdrant point in the AT collection's unnamed-vector schema."""
    point_metadata: dict = {
        "source_id": source_id,
        "source_url": source_url,
        "content_type": content_type,
        "language": language,
        "title": metadata.get("title", ""),
        "source_type": metadata.get("source_type", ""),
        "uploaded_by": metadata.get("uploaded_by", ""),
    }

    if entities:
        if entities.get("dates"):
            point_metadata["entity_dates"] = entities["dates"][:10]
        if entities.get("deadlines"):
            point_metadata["entity_deadlines"] = entities["deadlines"][:5]
        if entities.get("amounts"):
            point_metadata["entity_amounts"] = entities["amounts"][:10]
        if entities.get("contacts"):
            point_metadata["entity_contacts"] = entities["contacts"][:10]
        if entities.get("departments"):
            point_metadata["entity_departments"] = entities["departments"][:5]

    for key, value in metadata.items():
        if key not in _KNOWN_METADATA_KEYS and value is not None:
            point_metadata[key] = value

    payload = {
        "municipality_id": metadata.get("municipality_id", ""),
        "assistant_id": metadata.get("assistant_id", ""),
        "department": metadata.get("department", []),
        "content": chunk_text,
        "metadata": point_metadata,
    }

    return {
        "id": _point_id(source_id, chunk_index),
        "vector": embedding,
        "payload": payload,
    }


async def _delete_existing_by_source_id(qdrant, collection: str, source_id: str) -> None:
    """Best-effort delete of prior points for this source_id. Swallows errors.

    ``metadata.source_id`` is indexed on the AT collections, so delete-by-filter
    is permitted under strict-mode. This clears every prior chunk for the
    document before the fresh upsert, guaranteeing a repeat ingest fully
    replaces stored content (handles chunk-count changes between ingests).
    """
    try:
        await qdrant.delete_by_filter(
            collection,
            {"must": [{"key": "metadata.source_id", "match": {"value": source_id}}]},
        )
    except QdrantError as e:
        log.warning("ingest_online_at_delete_skipped", collection=collection, error=str(e))


@router.post(
    "/ingest/at",
    summary="Ingest funding content into a single AT Qdrant collection",
    description=(
        "AT funding-assistant ingest. The country (AT) and assistant type "
        "(funding) are implicit — do not pass them in the body.\n\n"
        "**Flow:** chunk → optional contextual enrichment → TEI embed "
        "(1024-dim, cosine) → upsert to `body.collection_name` on the AT "
        "Qdrant instance. The collection is auto-created with the AT legacy "
        "schema (single unnamed 1024-dim cosine vector + keyword indexes on "
        "`metadata.source_id` / `metadata.source_url`) on first use.\n\n"
        "**Metadata:** the funding extractor runs unconditionally and its "
        "output (title, program_name, processing_office, contract_email, "
        "contract_phone, state_or_province, funding_type, status, …) is "
        "merged into `metadata.*` on every point. `state_or_province` in the "
        "request body overrides the extractor's choice for the stored "
        "metadata only — there is no per-province collection routing.\n\n"
        "**Idempotency:** prior points for the same `source_id` are deleted via "
        "the indexed `metadata.source_id` field before upsert — a repeat ingest "
        "fully replaces stored chunks, correctly handling cases where the new "
        "content produces a different chunk count.\n\n"
        "**Embedding:** uses the TEI server at `TEI_EMBED_URL_AT` "
        "(OpenAI-compatible, bearer-auth via `TEI_EMBED_API_KEY_AT`).\n\n"
        "**Qdrant target:** uses `QDRANT_URL_AT` / `QDRANT_PORT_AT` / `QDRANT_API_KEY_AT` "
        "when set, falling back to the default Qdrant endpoint otherwise. "
        "`QDRANT_PORT_AT` has no default — leave it unset when the port is "
        "already embedded in `QDRANT_URL_AT`.\n\n"
        "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.\n\n"
        "**Error codes:** `VALIDATION_EMPTY_CONTENT`, `EMBEDDING_MODEL_NOT_LOADED`, "
        "`EMBEDDING_FAILED`, `EMBEDDING_OOM`, `QDRANT_CONNECTION_FAILED`, "
        "`QDRANT_COLLECTION_NOT_FOUND`, `QDRANT_UPSERT_FAILED`, `QDRANT_DISK_FULL`."
    ),
    response_description="Ingestion result with the collection written to and timing info.",
)
async def ingest_online_at(
    body: OnlineIngestATRequest,
    request: Request,
) -> ResponseEnvelope[OnlineIngestATData]:
    request_id = request.state.request_id
    started = time.monotonic()

    if not body.content.strip():
        return ResponseEnvelope(
            success=False,
            error=ErrorCode.VALIDATION_EMPTY_CONTENT,
            detail="Content must not be empty",
            request_id=request_id,
        )

    log.info(
        "ingest_online_at_received",
        source_id=body.source_id,
        url=body.url,
        collection=body.collection_name,
    )

    chunker = request.app.state.chunker
    contextual_enricher = request.app.state.contextual_enricher
    tei_embedder = request.app.state.tei_embedder_at
    qdrant = request.app.state.qdrant_at
    extractor = request.app.state.funding_extractor

    # ── 1. Funding extraction (once) ──
    extracted = await _safe_extract_funding(
        extractor,
        body.content,
        source_url=body.url,
        source_id=body.source_id,
    )

    # ── 2. Chunk (+ optional contextual enrichment) ──
    chunking = body.chunking
    strategy = chunking.strategy if chunking else "contextual"
    use_contextual = strategy == "contextual"
    base_strategy = "recursive" if use_contextual else strategy

    chunk_result = chunker.chunk(
        text=body.content,
        strategy=base_strategy,
        max_chunk_size=chunking.max_chunk_size if chunking else _AT_DEFAULT_CHUNK_SIZE,
        overlap=(chunking.overlap if chunking and chunking.overlap is not None else _AT_DEFAULT_CHUNK_OVERLAP),
    )
    if not chunk_result.chunks:
        return ResponseEnvelope(
            success=False,
            error=ErrorCode.VALIDATION_EMPTY_CONTENT,
            detail="Content produced no chunks",
            request_id=request_id,
        )

    chunks = chunk_result.chunks
    if use_contextual and contextual_enricher:
        try:
            chunks = await contextual_enricher.enrich_chunks(document=body.content, chunks=chunks)
        except Exception as e:
            log.warning("ingest_online_at_contextual_failed", source_id=body.source_id, error=str(e))

    # ── 3. Embed once via TEI (unnamed 1024-dim cosine vector) ──
    embed_start = time.monotonic()
    try:
        embeddings = await tei_embedder.embed_batch(chunks)
    except EmbeddingError as e:
        msg = str(e).lower()
        if "oom" in msg or "memory" in msg:
            code = ErrorCode.EMBEDDING_OOM
        elif "not initialized" in msg or "not loaded" in msg or "not configured" in msg:
            code = ErrorCode.EMBEDDING_MODEL_NOT_LOADED
        else:
            code = ErrorCode.EMBEDDING_FAILED
        log.error("ingest_online_at_embed_failed", source_id=body.source_id, error=str(e))
        return ResponseEnvelope(
            success=False,
            error=code,
            detail=str(e),
            request_id=request_id,
        )
    embedding_time_ms = int((time.monotonic() - embed_start) * 1000)

    # ── 4. Build metadata ──
    base_metadata = body.metadata.model_dump()
    base_metadata["source_url"] = body.url
    base_metadata["assistant_type"] = _AT_ASSISTANT_TYPE
    if body.source_name:
        base_metadata["source_name"] = body.source_name
    # Funding-extractor output merged under request metadata so request wins.
    merged_metadata = {**extracted, **base_metadata} if extracted else base_metadata

    # state_or_province: request override > extractor output. Lowercase, deduped.
    override_states = _normalize_provinces(body.state_or_province)
    extracted_states = _normalize_provinces(extracted.get("state_or_province") if extracted else None)
    merged_metadata["state_or_province"] = override_states or extracted_states

    # Fall back program_name to title when the extractor didn't find a
    # distinct program name — search-time ranking relies on this field.
    if not merged_metadata.get("program_name"):
        merged_metadata["program_name"] = merged_metadata.get("title", "")

    entities = body.entities.model_dump() if body.entities else None
    language = body.language or "de"

    # ── 5. Build + upsert points (single collection) ──
    collection = body.collection_name
    try:
        await qdrant.ensure_at_collection(collection)
    except QdrantError as e:
        msg = str(e).lower()
        if "connection" in msg:
            code = ErrorCode.QDRANT_CONNECTION_FAILED
        else:
            code = ErrorCode.QDRANT_COLLECTION_NOT_FOUND
        log.error("ingest_online_at_ensure_collection_failed", collection=collection, error=str(e))
        return ResponseEnvelope(
            success=False,
            error=code,
            detail=str(e),
            request_id=request_id,
        )

    await _delete_existing_by_source_id(qdrant, collection, body.source_id)

    points = [
        _build_point(
            chunk_text=chunk_text,
            chunk_index=i,
            embedding=embeddings[i].dense,
            source_id=body.source_id,
            source_url=body.url,
            content_type=body.content_type,
            language=language,
            metadata=merged_metadata,
            entities=entities,
        )
        for i, chunk_text in enumerate(chunks)
    ]

    try:
        vectors_stored = await qdrant.upsert_points(collection, points)
    except QdrantError as e:
        msg = str(e).lower()
        if "disk" in msg or "full" in msg:
            code = ErrorCode.QDRANT_DISK_FULL
        elif "not found" in msg:
            code = ErrorCode.QDRANT_COLLECTION_NOT_FOUND
        elif "connection" in msg:
            code = ErrorCode.QDRANT_CONNECTION_FAILED
        else:
            code = ErrorCode.QDRANT_UPSERT_FAILED
        log.error("ingest_online_at_upsert_failed", source_id=body.source_id, error=str(e))
        return ResponseEnvelope(
            success=False,
            error=code,
            detail=str(e),
            request_id=request_id,
        )

    total_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "ingest_online_at_complete",
        source_id=body.source_id,
        chunks=len(chunks),
        collection=collection,
        vectors_stored=vectors_stored,
        total_ms=total_ms,
    )

    return ResponseEnvelope(
        success=True,
        data=OnlineIngestATData(
            source_id=body.source_id,
            chunks_created=len(chunks),
            vectors_stored=vectors_stored,
            collection_name=collection,
            content_type=body.content_type,
            embedding_time_ms=embedding_time_ms,
            total_time_ms=total_ms,
        ),
        request_id=request_id,
    )
