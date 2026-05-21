"""Ingest pipeline — chunks → classifies → embeds → stores in Qdrant."""

import asyncio
import time
import uuid
from collections.abc import Awaitable

from app.config import settings
from app.models.common import StageUsage, UsageSummary
from app.services.embedding.bge_m3_client import EmbeddingError
from app.services.embedding.qdrant_service import QdrantError, QdrantService
from app.services.intelligence.chunker import Chunker
from app.services.intelligence.classifier import Classifier
from app.utils.logger import get_logger

log = get_logger(__name__)


class IngestError(Exception):
    def __init__(self, message: str, code: str = "EMBEDDING_FAILED"):
        super().__init__(message)
        self.code = code


class IngestResult:
    def __init__(
        self,
        source_id: str,
        chunks_created: int,
        vectors_stored: int,
        collection: str,
        classification: list[str],
        entities_extracted: dict,
        embedding_time_ms: int,
        total_time_ms: int,
        usage: UsageSummary | None = None,
    ):
        self.source_id = source_id
        self.chunks_created = chunks_created
        self.vectors_stored = vectors_stored
        self.collection = collection
        self.classification = classification
        self.entities_extracted = entities_extracted
        self.embedding_time_ms = embedding_time_ms
        self.total_time_ms = total_time_ms
        # Aggregated per-stage billing from this ingest call (classifier,
        # contextual, funding, embedding, sparse). Routers surface it on
        # the response envelope and persist it to ClickHouse usage_log.
        self.usage = usage or UsageSummary()


class IngestService:
    """Orchestrates the full ingest pipeline: chunk → classify → embed → store.

    One dense vector per point, named after the selected embedder
    (``dense_openai`` or ``dense_bge_m3``). When ``search_mode`` is
    ``hybrid`` an additional ``sparse`` vector is produced by the injected
    TEI sparse client (``sparse.ki2.at``).
    """

    def __init__(
        self,
        chunker: Chunker,
        classifier: Classifier,
        embedder,
        qdrant: QdrantService,
        contextual_enricher=None,
        sparse_embedder=None,
    ) -> None:
        self._chunker = chunker
        self._classifier = classifier
        self._embedder = embedder
        self._qdrant = qdrant
        self._contextual_enricher = contextual_enricher
        self._sparse_embedder = sparse_embedder

    async def ingest(
        self,
        source_id: str,
        file_path: str,
        content: str,
        acl: dict | None,
        metadata: dict,
        collection_name: str,
        language: str | None = None,
        chunking_strategy: str = "late_chunking",
        max_chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        vector_size: int = 1536,
        search_mode: str = "semantic",
        content_type: list[str] | None = None,
        entities: dict | None = None,
        deferred_metadata_task: Awaitable[dict] | None = None,
        progress_queue: asyncio.Queue | None = None,
        primary_embedder=None,
        primary_vector_name: str = "dense_openai",
    ) -> IngestResult:
        start = time.monotonic()
        collection = collection_name
        use_sparse = search_mode == "hybrid"
        # Per-request override of the configured primary embedder (e.g. swap
        # OpenAI for the TEI BGE-M3 client). Defaults to the service-wide
        # embedder injected at startup.
        embedder = primary_embedder or self._embedder

        # Per-stage billing records collected as the pipeline runs; rolled
        # up into the ``UsageSummary`` returned on ``IngestResult`` so the
        # router can put it on the response and persist to ClickHouse.
        usage_entries: list[StageUsage] = []

        async def _emit(phase: str, **payload) -> None:
            if progress_queue is not None:
                await progress_queue.put({"phase": phase, **payload})

        await _emit("started", source_id=source_id)

        if not collection:
            raise IngestError("collection_name is required", code="QDRANT_COLLECTION_NOT_FOUND")

        # Ensure collection exists with correct vector config
        try:
            await self._qdrant.create_collection(
                name=collection,
                sparse=use_sparse,
                distance="Cosine",
                multi_vector={primary_vector_name: vector_size},
            )
        except QdrantError as e:
            raise IngestError(str(e), code="QDRANT_CONNECTION_FAILED") from e

        # 1. Chunk
        use_contextual = chunking_strategy == "contextual"
        base_strategy = "recursive" if use_contextual else chunking_strategy

        chunk_result = self._chunker.chunk(
            text=content,
            strategy=base_strategy,
            max_chunk_size=max_chunk_size or settings.default_chunk_size,
            overlap=chunk_overlap if chunk_overlap is not None else settings.default_chunk_overlap,
        )

        if not chunk_result.chunks:
            raise IngestError("Content produced no chunks", code="VALIDATION_EMPTY_CONTENT")

        log.info("ingest_chunked", source_id=source_id, chunks=chunk_result.total_chunks)
        await _emit("chunked", chunks=chunk_result.total_chunks)

        # 1b. Contextual Retrieval — enrich each chunk with document-level context
        if use_contextual and self._contextual_enricher:
            try:
                enriched_chunks, contextual_usage = await self._contextual_enricher.enrich_chunks(
                    document=content,
                    chunks=chunk_result.chunks,
                )
                chunk_result.chunks = enriched_chunks
                if contextual_usage is not None:
                    usage_entries.append(contextual_usage)
                log.info("ingest_contextual_enriched", source_id=source_id, chunks=len(chunk_result.chunks))
                await _emit("enriched", chunks=len(chunk_result.chunks))
            except Exception as e:
                log.warning("ingest_contextual_enrichment_failed", source_id=source_id, error=str(e))
                await _emit("enriched", chunks=len(chunk_result.chunks), error=str(e))

        # 2. Classify (on full content for better accuracy) — skipped when
        # the caller already supplies content_type (e.g. online ingest, where
        # classification happens upstream at scrape/parse time).
        classify_result = None
        if content_type is not None:
            classification = content_type
        else:
            try:
                classify_result = await self._classifier.classify(content, language=language or "de")
            except Exception as e:
                log.warning("ingest_classify_fallback", source_id=source_id, error=str(e))
                classify_result = None

            if classify_result:
                classification = [classify_result.category.value] + classify_result.sub_categories
                if classify_result.usage is not None:
                    usage_entries.append(classify_result.usage)
            else:
                classification = ["general"]
        entities_extracted = {
            "dates": len(classify_result.entities.dates) if classify_result else 0,
            "contacts": len(classify_result.entities.contacts) if classify_result else 0,
            "amounts": len(classify_result.entities.amounts) if classify_result else 0,
        }

        # 3. Embed all chunks. Dense runs in parallel with sparse (when
        # hybrid) so total embed latency is max(dense, sparse).
        embed_start = time.monotonic()

        dense_task = asyncio.create_task(embedder.embed_batch(chunk_result.chunks))
        sparse_task: asyncio.Task | None = None
        if use_sparse:
            if self._sparse_embedder is None:
                raise IngestError(
                    "search_mode='hybrid' requested but no sparse embedder is configured",
                    code="EMBEDDING_MODEL_NOT_LOADED",
                )
            sparse_task = asyncio.create_task(self._sparse_embedder.encode_batch(chunk_result.chunks))

        try:
            dense_embeddings = await dense_task
        except EmbeddingError as e:
            error_msg = str(e).lower()
            if "oom" in error_msg or "memory" in error_msg:
                raise IngestError(str(e), code="EMBEDDING_OOM") from e
            if "not initialized" in error_msg or "not loaded" in error_msg:
                raise IngestError(str(e), code="EMBEDDING_MODEL_NOT_LOADED") from e
            raise IngestError(str(e), code="EMBEDDING_FAILED") from e

        sparse_embeddings: list | None = None
        if sparse_task is not None:
            try:
                sparse_embeddings = await sparse_task
            except EmbeddingError as e:
                raise IngestError(f"Sparse embed failed: {e}", code="EMBEDDING_FAILED") from e

        embedding_time_ms = int((time.monotonic() - embed_start) * 1000)
        # Each embed client stashes its per-call usage on ``last_usage``
        # after embed_batch — pull both (dense + optional sparse) into the
        # aggregate. ``isinstance`` gates out test stubs (MagicMock leaves
        # ``last_usage`` as a non-StageUsage attribute).
        dense_usage = getattr(embedder, "last_usage", None)
        if isinstance(dense_usage, StageUsage):
            usage_entries.append(dense_usage)
        if sparse_embeddings is not None and self._sparse_embedder is not None:
            sparse_usage = getattr(self._sparse_embedder, "last_usage", None)
            if isinstance(sparse_usage, StageUsage):
                usage_entries.append(sparse_usage)
        log.info(
            "ingest_embedded",
            source_id=source_id,
            chunks=len(chunk_result.chunks),
            primary_vector=primary_vector_name,
            has_sparse=sparse_embeddings is not None,
            duration_ms=embedding_time_ms,
        )
        await _emit(
            "embedded",
            chunks=len(chunk_result.chunks),
            primary_vector=primary_vector_name,
            has_sparse=sparse_embeddings is not None,
            duration_ms=embedding_time_ms,
        )

        # 3b. Await deferred metadata (e.g. funding extractor running in
        # parallel with chunking/contextual/embed) and merge it under
        # request-supplied metadata so explicit request fields still win.
        if deferred_metadata_task is not None:
            try:
                deferred_meta = await deferred_metadata_task
            except Exception as e:
                log.warning("ingest_deferred_metadata_failed", source_id=source_id, error=str(e))
                deferred_meta = None
            if deferred_meta:
                # Pull the billing sentinel out before merging — it must not
                # land in the stored Qdrant payload. ``__usage__`` is the
                # constant key the funding extractor stashes its StageUsage
                # under (see funding_extractor._KEY_USAGE).
                funding_usage = deferred_meta.pop("__usage__", None)
                if isinstance(funding_usage, StageUsage):
                    usage_entries.append(funding_usage)
                metadata = {**deferred_meta, **metadata}
                await _emit("funding_extracted", fields=sorted(deferred_meta.keys()))
            else:
                await _emit("funding_extracted", fields=[])

        # 4. Build Qdrant points
        points = []
        for i, chunk_text in enumerate(chunk_result.chunks):
            chunk_id = f"{source_id}_chunk_{i:04d}"
            point_metadata = {
                "chunk_id": chunk_id,
                "source_id": source_id,
                "chunk_index": i,
                "source_url": metadata.get("source_url", ""),
                "content_type": classification,
                "language": language or "de",
                "title": metadata.get("title", ""),
                "source_type": metadata.get("source_type", ""),
                "mime_type": metadata.get("mime_type", ""),
                "uploaded_by": metadata.get("uploaded_by", ""),
            }

            # ACL fields (when provided)
            if acl:
                point_metadata.update({
                    "acl_allow_groups": acl.get("allow_groups", []),
                    "acl_deny_groups": acl.get("deny_groups", []),
                    "acl_allow_roles": acl.get("allow_roles", []),
                    "acl_allow_users": acl.get("allow_users", []),
                    "acl_visibility": acl.get("visibility", "public"),
                    "acl_department": acl.get("department", ""),
                })

            # Add entity data if available. Caller-supplied ``entities`` wins
            # over the classifier (online ingest supplies them from the
            # upstream scrape/parse response; local ingest falls through to
            # the classifier's own extraction).
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
            elif classify_result:
                point_metadata["entity_amounts"] = classify_result.entities.amounts[:5]
                point_metadata["entity_deadlines"] = classify_result.entities.deadlines[:5]

            # Pass through extra metadata fields (e.g. funding extraction fields)
            _known_keys = {
                "chunk_id", "source_id", "chunk_index", "source_url", "source_path",
                "content_type", "language", "title", "source_type", "mime_type",
                "uploaded_by", "assistant_id", "municipality_id", "department",
                "assistant_type",
            }
            for key, value in metadata.items():
                if key not in _known_keys and value is not None:
                    point_metadata[key] = value

            payload = {
                "municipality_id": metadata.get("municipality_id", ""),
                "assistant_id": metadata.get("assistant_id", ""),
                "department": metadata.get("department", []),
                "content": chunk_text,
                "metadata": point_metadata,
            }

            # Build vectors dict — one dense vector named after the selected
            # embedder (dense_openai / dense_bge_m3) plus an optional TEI
            # sparse vector when hybrid mode is on.
            vectors: dict = {primary_vector_name: dense_embeddings[i].dense}
            if use_sparse and sparse_embeddings is not None:
                vectors["sparse"] = sparse_embeddings[i].as_dict()

            point = {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)),
                "vector": vectors,
                "payload": payload,
            }
            points.append(point)

        # 5. Delete old vectors for this source_id, then upsert new ones
        try:
            await self._qdrant.delete_by_source_id(collection, source_id)
        except QdrantError:
            pass  # OK if nothing to delete

        try:
            vectors_stored = await self._qdrant.upsert_points(collection, points)
        except QdrantError as e:
            error_msg = str(e).lower()
            if "disk" in error_msg or "full" in error_msg:
                raise IngestError(str(e), code="QDRANT_DISK_FULL") from e
            if "not found" in error_msg:
                raise IngestError(str(e), code="QDRANT_COLLECTION_NOT_FOUND") from e
            raise IngestError(str(e), code="QDRANT_UPSERT_FAILED") from e

        await _emit("stored", vectors=vectors_stored, collection=collection)

        total_time_ms = int((time.monotonic() - start) * 1000)
        log.info(
            "ingest_complete",
            source_id=source_id,
            chunks=chunk_result.total_chunks,
            vectors=vectors_stored,
            collection=collection,
            total_ms=total_time_ms,
        )

        return IngestResult(
            source_id=source_id,
            chunks_created=chunk_result.total_chunks,
            vectors_stored=vectors_stored,
            collection=collection,
            classification=classification,
            entities_extracted=entities_extracted,
            embedding_time_ms=embedding_time_ms,
            total_time_ms=total_time_ms,
            usage=UsageSummary.from_entries(usage_entries),
        )
