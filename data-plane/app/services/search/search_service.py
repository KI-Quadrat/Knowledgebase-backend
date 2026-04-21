"""Permission-aware semantic and hybrid search service."""

import time

from app.services.embedding.bge_m3_client import EmbeddingError
from app.services.embedding.bm25_encoder import BM25Encoder
from app.services.embedding.qdrant_service import QdrantError, QdrantService
from app.utils.logger import get_logger

log = get_logger(__name__)


class SearchError(Exception):
    def __init__(self, message: str, code: str = "QDRANT_SEARCH_FAILED"):
        super().__init__(message)
        self.code = code


class PermissionFilter:
    """Builds the permission filter applied to search results."""

    def __init__(self, visibility: list[str], must_match_groups: list[str], must_not_match_groups: list[str]):
        self.visibility = visibility
        self.must_match_groups = must_match_groups
        self.must_not_match_groups = must_not_match_groups


class SearchResultItem:
    def __init__(
        self,
        chunk_id: str,
        source_id: str,
        chunk_text: str,
        score: float,
        source_path: str,
        classification: list[str],
        entity_amounts: list[str],
        entity_deadlines: list[str],
        title: str | None,
        municipality_id: str | None,
        department: list[str] | None,
        source_type: str | None,
    ):
        self.chunk_id = chunk_id
        self.source_id = source_id
        self.chunk_text = chunk_text
        self.score = score
        self.source_path = source_path
        self.classification = classification
        self.entity_amounts = entity_amounts
        self.entity_deadlines = entity_deadlines
        self.title = title
        self.municipality_id = municipality_id
        self.department = department
        self.source_type = source_type


class SearchResult:
    def __init__(
        self,
        results: list[SearchResultItem],
        total_results: int,
        query_embedding_ms: int,
        search_ms: int,
        permission_filter: PermissionFilter,
        search_mode: str,
    ):
        self.results = results
        self.total_results = total_results
        self.query_embedding_ms = query_embedding_ms
        self.search_ms = search_ms
        self.permission_filter = permission_filter
        self.search_mode = search_mode


class SearchService:
    """Semantic and hybrid search with permission filtering.

    - semantic: dense-only cosine search via OpenAI embeddings
    - hybrid: dense (OpenAI) + sparse (BM25) with Reciprocal Rank Fusion (RRF)

    When a ``fallback_embedder`` is provided (BGE-Gemma2 via LiteLLM),
    the service automatically falls back to ``dense_bge_gemma2`` vectors
    if the primary OpenAI embedding call fails.
    """

    def __init__(self, embedder, qdrant: QdrantService, fallback_embedder=None) -> None:
        self._embedder = embedder
        self._qdrant = qdrant
        self._fallback_embedder = fallback_embedder
        self._bm25 = BM25Encoder()

    async def search(
        self,
        query: str,
        collection_name: str,
        user_type: str,
        user_id: str,
        user_groups: list[str] | None = None,
        user_roles: list[str] | None = None,
        user_department: str | None = None,
        classification_filter: list[str] | None = None,
        top_k: int = 10,
        score_threshold: float = 0.5,
        search_mode: str = "semantic",
        enable_fallback: bool = False,
    ) -> SearchResult:
        collection = collection_name
        if not collection:
            raise SearchError("collection_name is required", code="QDRANT_COLLECTION_NOT_FOUND")

        user_groups = user_groups or []

        # 1. Build permission filter
        perm_filter = self._build_permission_filter(user_type, user_groups)

        # 2. Embed the query (dense via OpenAI, optionally fallback to BGE-Gemma2)
        embed_start = time.monotonic()
        use_fallback = enable_fallback and self._fallback_embedder is not None
        dense_vector_name = "dense_openai"
        embedding = None

        try:
            embedding = await self._embedder.embed(query)
        except EmbeddingError as e:
            if not use_fallback:
                error_msg = str(e).lower()
                if "not initialized" in error_msg:
                    raise SearchError(str(e), code="EMBEDDING_MODEL_NOT_LOADED") from e
                raise SearchError(str(e), code="EMBEDDING_FAILED") from e
            log.warning("search_primary_embed_failed_using_fallback", error=str(e))

        if embedding is None and use_fallback:
            try:
                embedding = await self._fallback_embedder.embed(query)
                dense_vector_name = "dense_bge_gemma2"
            except EmbeddingError as e:
                raise SearchError(
                    f"Both primary and fallback embedding failed: {e}",
                    code="EMBEDDING_FAILED",
                ) from e

        query_embedding_ms = int((time.monotonic() - embed_start) * 1000)

        # 3. Build Qdrant filter
        qdrant_filter = self._build_qdrant_filter(perm_filter, classification_filter)

        # 4. Search — semantic or hybrid
        search_start = time.monotonic()
        try:
            if search_mode == "hybrid":
                sparse_vector = self._bm25.encode(query)
                raw_results = await self._qdrant.hybrid_search(
                    collection=collection,
                    dense_vector=embedding.dense,
                    sparse_vector=sparse_vector,
                    filters=qdrant_filter,
                    top_k=top_k,
                    dense_vector_name=dense_vector_name,
                )
            else:
                raw_results = await self._qdrant.search(
                    collection=collection,
                    dense_vector=embedding.dense,
                    filters=qdrant_filter,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    dense_vector_name=dense_vector_name,
                )
        except QdrantError as e:
            error_msg = str(e).lower()
            if "not found" in error_msg:
                raise SearchError(str(e), code="QDRANT_COLLECTION_NOT_FOUND") from e
            if "connection" in error_msg:
                raise SearchError(str(e), code="QDRANT_CONNECTION_FAILED") from e
            raise SearchError(str(e), code="QDRANT_SEARCH_FAILED") from e
        search_ms = int((time.monotonic() - search_start) * 1000)

        # 5. Map results
        items = []
        for hit in raw_results:
            payload = hit.get("payload", {})
            meta = payload.get("metadata", {})
            items.append(SearchResultItem(
                chunk_id=meta.get("chunk_id", ""),
                source_id=meta.get("source_id", ""),
                chunk_text=payload.get("content", ""),
                score=hit.get("score", 0.0),
                source_path=meta.get("source_path", ""),
                classification=meta.get("content_type", ["general"]),
                entity_amounts=meta.get("entity_amounts", []),
                entity_deadlines=meta.get("entity_deadlines", []),
                title=meta.get("title"),
                municipality_id=payload.get("municipality_id"),
                department=payload.get("department") or meta.get("acl_department"),
                source_type=meta.get("source_type"),
            ))

        log.info(
            "search_complete",
            query_len=len(query),
            results=len(items),
            user_type=user_type,
            search_mode=search_mode,
            embedding_ms=query_embedding_ms,
            search_ms=search_ms,
        )

        return SearchResult(
            results=items,
            total_results=len(items),
            query_embedding_ms=query_embedding_ms,
            search_ms=search_ms,
            permission_filter=perm_filter,
            search_mode=search_mode,
        )

    def _build_permission_filter(
        self, user_type: str, user_groups: list[str],
    ) -> PermissionFilter:
        if user_type == "citizen":
            return PermissionFilter(
                visibility=["public"],
                must_match_groups=[],
                must_not_match_groups=[],
            )

        # Employee: can see public + internal
        return PermissionFilter(
            visibility=["public", "internal"],
            must_match_groups=user_groups,
            must_not_match_groups=[],
        )

    def _build_qdrant_filter(
        self,
        perm_filter: PermissionFilter,
        classification_filter: list[str] | None,
    ) -> dict:
        must_conditions = []

        # Visibility filter
        must_conditions.append({
            "key": "metadata.acl_visibility",
            "match": {"any": perm_filter.visibility},
        })

        # Group-based access: user's groups must intersect with allow_groups,
        # OR allow_groups is empty (public doc)
        if perm_filter.must_match_groups:
            must_conditions.append({
                "should": [
                    {
                        "key": "metadata.acl_allow_groups",
                        "match": {"any": perm_filter.must_match_groups},
                    },
                    {
                        "is_empty": {"key": "metadata.acl_allow_groups"},
                    },
                ],
            })

        # Content type filter
        if classification_filter:
            must_conditions.append({
                "key": "metadata.content_type",
                "match": {"any": classification_filter},
            })

        return {"must": must_conditions}
