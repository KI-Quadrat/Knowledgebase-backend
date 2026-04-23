"""Ingest pipeline — chunks → classifies → embeds → stores in Qdrant."""

import asyncio
import time
import uuid
from collections.abc import Awaitable

from app.config import settings
from app.services.embedding.bge_m3_client import EmbeddingError
from app.services.embedding.bm25_encoder import BM25Encoder
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
    ):
        self.source_id = source_id
        self.chunks_created = chunks_created
        self.vectors_stored = vectors_stored
        self.collection = collection
        self.classification = classification
        self.entities_extracted = entities_extracted
        self.embedding_time_ms = embedding_time_ms
        self.total_time_ms = total_time_ms


class IngestService:
    """Orchestrates the full ingest pipeline: chunk → classify → embed → store.

    When a ``fallback_embedder`` is provided (BGE-Gemma2 via LiteLLM) and
    ``fallback_dense_dim`` is passed to ``ingest()``, the service stores
    multi-vector points with ``dense_openai`` + ``dense_bge_gemma2`` (and
    optionally ``sparse`` for hybrid mode). If one embedder fails during
    ingest, the point is still stored with the other's vector.
    """

    def __init__(
        self,
        chunker: Chunker,
        classifier: Classifier,
        embedder,
        qdrant: QdrantService,
        contextual_enricher=None,
        fallback_embedder=None,
    ) -> None:
        self._chunker = chunker
        self._classifier = classifier
        self._embedder = embedder
        self._qdrant = qdrant
        self._contextual_enricher = contextual_enricher
        self._fallback_embedder = fallback_embedder
        self._bm25 = BM25Encoder()

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
        fallback_dense_dim: int | None = None,
        content_type: list[str] | None = None,
        entities: dict | None = None,
        deferred_metadata_task: Awaitable[dict] | None = None,
        progress_queue: asyncio.Queue | None = None,
    ) -> IngestResult:
        start = time.monotonic()
        collection = collection_name
        use_sparse = search_mode == "hybrid"
        use_multi_vector = self._fallback_embedder is not None and fallback_dense_dim is not None

        async def _emit(phase: str, **payload) -> None:
            if progress_queue is not None:
                await progress_queue.put({"phase": phase, **payload})

        await _emit("started", source_id=source_id)

        if not collection:
            raise IngestError("collection_name is required", code="QDRANT_COLLECTION_NOT_FOUND")

        # Ensure collection exists with correct vector config
        try:
            multi_vec = {"dense_openai": vector_size}
            if use_multi_vector:
                multi_vec["dense_bge_gemma2"] = fallback_dense_dim
            await self._qdrant.create_collection(
                name=collection,
                sparse=use_sparse,
                distance="Cosine",
                multi_vector=multi_vec,
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
                chunk_result.chunks = await self._contextual_enricher.enrich_chunks(
                    document=content,
                    chunks=chunk_result.chunks,
                )
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
            else:
                classification = ["general"]
        entities_extracted = {
            "dates": len(classify_result.entities.dates) if classify_result else 0,
            "contacts": len(classify_result.entities.contacts) if classify_result else 0,
            "amounts": len(classify_result.entities.amounts) if classify_result else 0,
        }

        # 3. Embed all chunks — primary + fallback run in parallel so
        # total embed latency is max(primary, fallback) instead of the sum.
        embed_start = time.monotonic()

        primary_task = asyncio.create_task(self._embedder.embed_batch(chunk_result.chunks))
        fallback_task: asyncio.Task | None = None
        if use_multi_vector:
            fallback_task = asyncio.create_task(self._fallback_embedder.embed_batch(chunk_result.chunks))

        openai_embeddings = None
        primary_error: EmbeddingError | None = None
        try:
            openai_embeddings = await primary_task
        except EmbeddingError as e:
            primary_error = e
            log.warning("ingest_primary_embed_failed", source_id=source_id, error=str(e))

        fallback_embeddings = None
        if fallback_task is not None:
            try:
                fallback_embeddings = await fallback_task
            except EmbeddingError as e:
                log.warning("ingest_fallback_embed_failed", source_id=source_id, error=str(e))

        # If the only embedder failed and there's no fallback, surface the original error.
        if not use_multi_vector and primary_error is not None:
            error_msg = str(primary_error).lower()
            if "oom" in error_msg or "memory" in error_msg:
                raise IngestError(str(primary_error), code="EMBEDDING_OOM") from primary_error
            if "not initialized" in error_msg or "not loaded" in error_msg:
                raise IngestError(str(primary_error), code="EMBEDDING_MODEL_NOT_LOADED") from primary_error
            raise IngestError(str(primary_error), code="EMBEDDING_FAILED") from primary_error

        # At least one embedding source must succeed
        if openai_embeddings is None and fallback_embeddings is None:
            raise IngestError(
                "Both primary and fallback embedding models failed",
                code="EMBEDDING_FAILED",
            )

        embedding_time_ms = int((time.monotonic() - embed_start) * 1000)
        log.info(
            "ingest_embedded",
            source_id=source_id,
            chunks=len(chunk_result.chunks),
            has_openai=openai_embeddings is not None,
            has_bge_gemma2=fallback_embeddings is not None,
            duration_ms=embedding_time_ms,
        )
        await _emit(
            "embedded",
            chunks=len(chunk_result.chunks),
            has_openai=openai_embeddings is not None,
            has_bge_gemma2=fallback_embeddings is not None,
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

            # Build vectors dict — always use dense_openai as the primary name
            vectors: dict = {}

            if openai_embeddings:
                vectors["dense_openai"] = openai_embeddings[i].dense
            if use_multi_vector and fallback_embeddings:
                vectors["dense_bge_gemma2"] = fallback_embeddings[i].dense

            # Include BM25 sparse vector for hybrid search mode
            if use_sparse:
                vectors["sparse"] = self._bm25.encode(chunk_text)

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
        )
