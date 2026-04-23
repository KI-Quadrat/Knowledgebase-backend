"""Qdrant vector database client — manages collections, upserts, searches, and deletions."""

import asyncio
import time
from urllib.parse import urlparse, urlunparse

import httpx

from app.config import ext
from app.utils.logger import get_logger

log = get_logger(__name__)


class QdrantError(Exception):
    pass


def _compose_base_url(url: str, port: int | None) -> str:
    """Combine a scheme+host URL with an optional port.

    - Mirrors the upstream qdrant-client pattern where URL and port are
      supplied separately.
    - If ``url`` already carries an explicit port (e.g. ``http://qdrant:6333``)
      it's preserved verbatim — the ``port`` arg is ignored in that case.
    - A ``port`` of ``None`` / ``0`` means "leave the URL as-is" so the
      default ``http://qdrant:6333`` config keeps working.
    """
    trimmed = url.rstrip("/")
    if not port:
        return trimmed
    parsed = urlparse(trimmed)
    if parsed.port is not None:
        return trimmed  # explicit port in URL wins over the kwarg
    if not parsed.hostname:
        return trimmed  # malformed; don't try to compose
    netloc = f"{parsed.hostname}:{port}"
    return urlunparse(parsed._replace(netloc=netloc))


class QdrantService:
    """HTTP client for the Qdrant vector database."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        port: int | None = None,
    ) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url = _compose_base_url(url or ext.qdrant_url, port)
        self._api_key = api_key if api_key is not None else ext.qdrant_api_key
        self._validated_collections: set[tuple[str, str]] = set()
        self._collection_lock = asyncio.Lock()

    async def startup(self) -> None:
        headers = {}
        if self._api_key:
            headers["api-key"] = self._api_key
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(30),
            headers=headers,
        )
        log.info("qdrant_service_started", url=self._base_url)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        self._validated_collections.clear()
        log.info("qdrant_service_stopped")

    async def check_health(self) -> bool:
        if not self._client:
            return False
        try:
            resp = await self._client.get("/healthz", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Collection management ────────────────────────────────────────

    async def list_collections(self) -> list[dict]:
        """List all Qdrant collections with basic info (name, vectors_count, points_count, status)."""
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        try:
            resp = await self._client.get("/collections")
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise QdrantError(f"Failed to list collections: {e.response.text}") from e
        except httpx.RequestError as e:
            raise QdrantError(f"Qdrant connection failed: {e}") from e

        collections = resp.json().get("result", {}).get("collections", [])

        result = []
        for col in collections:
            name = col.get("name", "")
            try:
                stats = await self.collection_stats(name)
                vectors_count = stats.get("vectors_count", stats.get("points_count", 0))
                points_count = stats.get("points_count", 0)
                status = stats.get("status", "unknown")
                disk_bytes = stats.get("disk_data_size", 0)
                disk_mb = round(disk_bytes / (1024 * 1024), 1) if disk_bytes else 0.0
                segments_count = stats.get("segments_count", 0)
                result.append({
                    "name": name,
                    "vectors_count": vectors_count,
                    "points_count": points_count,
                    "segments_count": segments_count,
                    "disk_usage_mb": disk_mb,
                    "status": status,
                })
            except QdrantError:
                result.append({
                    "name": name,
                    "vectors_count": 0,
                    "points_count": 0,
                    "segments_count": 0,
                    "disk_usage_mb": 0.0,
                    "status": "unknown",
                })

        return result

    async def create_collection(
        self,
        name: str,
        dense_dim: int = 1024,
        sparse: bool = True,
        distance: str = "Cosine",
        multi_vector: dict[str, int] | None = None,
    ) -> bool:
        """Create or recreate a Qdrant collection.

        Returns True if the collection was (re)created, False if it already
        existed with a compatible vector schema.

        When ``multi_vector`` is provided, the method checks whether an
        existing collection already contains the expected named vectors.
        If the schema doesn't match (e.g. the collection only has a legacy
        ``dense`` vector), the old collection is **deleted and recreated**
        with the new multi-vector config so that upserts succeed.

        Args:
            multi_vector: Optional mapping of vector names to dimensions, e.g.
                ``{"dense_openai": 1536, "dense_bge_gemma2": 3584}``.
                When provided, ``dense_dim`` is ignored and each entry
                becomes a named vector in the collection.
        """
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        # Build target vectors config
        if multi_vector:
            vectors_config = {
                vec_name: {"size": dim, "distance": distance}
                for vec_name, dim in multi_vector.items()
            }
            expected_vector_names = set(multi_vector.keys())
        else:
            vectors_config = {
                "dense": {
                    "size": dense_dim,
                    "distance": distance,
                },
            }
            expected_vector_names = {"dense"}
        schema_signature = self._schema_signature(vectors_config, sparse)

        # Fast path: this worker has already validated the collection schema.
        if (name, schema_signature) in self._validated_collections:
            log.debug("qdrant_collection_validation_cache_hit", collection=name)
            return False

        async with self._collection_lock:
            # Another request may have validated the same collection while we waited.
            if (name, schema_signature) in self._validated_collections:
                log.debug("qdrant_collection_validation_cache_hit_after_lock", collection=name)
                return False

            # Check if collection exists and whether its schema is compatible
            try:
                resp = await self._client.get(f"/collections/{name}")
                if resp.status_code == 200:
                    existing_vectors = resp.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
                    existing_vector_names = set(existing_vectors.keys())

                    if expected_vector_names.issubset(existing_vector_names):
                        self._validated_collections.add((name, schema_signature))
                        log.info("qdrant_collection_exists", collection=name)
                        return False

                    # Schema mismatch — cannot add new vector fields to existing collection
                    raise QdrantError(
                        f"Collection '{name}' has incompatible vector schema: "
                        f"existing={sorted(existing_vector_names)}, "
                        f"expected={sorted(expected_vector_names)}. "
                        f"Delete the collection manually and re-ingest to migrate to multi-vector."
                    )
            except httpx.RequestError as e:
                raise QdrantError(f"Qdrant connection failed: {e}") from e

            # Build sparse config
            if sparse:
                sparse_config = {
                    "sparse": {
                        "modifier": "idf",
                        "index": {"on_disk": False},
                    },
                }
            else:
                sparse_config = {}

            body: dict = {"vectors": vectors_config}
            if sparse_config:
                body["sparse_vectors"] = sparse_config

            try:
                resp = await self._client.put(
                    f"/collections/{name}",
                    json=body,
                )
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise QdrantError(f"Failed to create collection: {e.response.text}") from e
            except httpx.RequestError as e:
                raise QdrantError(f"Qdrant connection failed: {e}") from e

            self._validated_collections.add((name, schema_signature))
            log.info("qdrant_collection_created", collection=name, vectors=list(vectors_config.keys()), sparse=sparse)
            return True

    async def ensure_at_collection(
        self,
        name: str,
        dense_dim: int = 1024,
        distance: str = "Cosine",
    ) -> bool:
        """Ensure an AT-schema Qdrant collection exists and is ingest-ready.

        AT collections use the legacy **single-unnamed-vector** schema
        (``{"vectors": {"size": N, "distance": "..."}}``) — different from the
        named-multi-vector schema used by default platform collections. This
        method is the AT counterpart to :meth:`create_collection`.

        Guarantees on return:
        - Collection exists with a single unnamed vector of the requested dim.
        - Keyword payload indexes are present on ``metadata.source_id`` and
          ``metadata.source_url`` (required for strict-mode delete-by-filter
          used during idempotent re-ingest).

        If the collection already exists with a different vector dim, raises
        :class:`QdrantError` rather than silently proceeding (the upsert would
        fail with a cryptic error otherwise). Successful validations are
        cached per-worker so repeat calls cost nothing.

        Returns True if the collection was created, False if it already
        existed with a compatible schema.
        """
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        signature = f"at-unnamed:{dense_dim}:{distance}"
        cache_key = (name, signature)
        if cache_key in self._validated_collections:
            return False

        async with self._collection_lock:
            if cache_key in self._validated_collections:
                return False

            try:
                get_resp = await self._client.get(f"/collections/{name}")
            except httpx.RequestError as e:
                raise QdrantError(f"Qdrant connection failed: {e}") from e

            created = False
            if get_resp.status_code == 404:
                create_body = {"vectors": {"size": dense_dim, "distance": distance}}
                try:
                    put_resp = await self._client.put(f"/collections/{name}", json=create_body)
                    put_resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise QdrantError(f"Failed to create AT collection '{name}': {e.response.text}") from e
                except httpx.RequestError as e:
                    raise QdrantError(f"Qdrant connection failed: {e}") from e
                created = True
                log.info("qdrant_at_collection_created", collection=name, dim=dense_dim)
            elif get_resp.status_code == 200:
                existing_vectors = (
                    get_resp.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
                )
                # Unnamed schema exposes size/distance at the top level.
                existing_size = existing_vectors.get("size") if isinstance(existing_vectors, dict) else None
                if existing_size is not None and existing_size != dense_dim:
                    raise QdrantError(
                        f"AT collection '{name}' has vector size {existing_size}, expected {dense_dim}. "
                        "Delete the collection manually to recreate with the new dim."
                    )
                log.debug("qdrant_at_collection_exists", collection=name)
            else:
                try:
                    get_resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    raise QdrantError(f"Failed to inspect AT collection '{name}': {e.response.text}") from e

            # Payload indexes are idempotent on Qdrant's side; a repeat PUT on
            # an existing index returns success without side effects.
            for field in ("metadata.source_id", "metadata.source_url"):
                try:
                    idx_resp = await self._client.put(
                        f"/collections/{name}/index",
                        json={"field_name": field, "field_schema": "keyword"},
                    )
                    idx_resp.raise_for_status()
                except httpx.HTTPStatusError as e:
                    log.warning(
                        "qdrant_at_index_skipped",
                        collection=name,
                        field=field,
                        status=e.response.status_code,
                        body=e.response.text,
                    )
                except httpx.RequestError as e:
                    raise QdrantError(f"Qdrant connection failed: {e}") from e

            self._validated_collections.add(cache_key)
            return created

    @staticmethod
    def _schema_signature(vectors_config: dict, sparse: bool) -> str:
        vector_parts = []
        for name, cfg in sorted(vectors_config.items()):
            vector_parts.append(f"{name}:{cfg['size']}:{cfg['distance']}")
        return f"vectors={'|'.join(vector_parts)};sparse={int(sparse)}"

    async def collection_stats(self, name: str) -> dict:
        """Get collection info and stats."""
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        try:
            resp = await self._client.get(f"/collections/{name}")
            if resp.status_code == 404:
                raise QdrantError(f"Collection not found: {name}")
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise QdrantError(f"Collection not found: {name}") from e
            raise QdrantError(f"Qdrant error: {e.response.text}") from e
        except httpx.RequestError as e:
            raise QdrantError(f"Qdrant connection failed: {e}") from e

        return resp.json().get("result", {})

    # ── Vector operations ────────────────────────────────────────────

    async def upsert_points(
        self,
        collection: str,
        points: list[dict],
    ) -> int:
        """Upsert points into a collection. Returns count of upserted points."""
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        try:
            resp = await self._client.put(
                f"/collections/{collection}/points",
                json={"points": points},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise QdrantError(f"Upsert failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise QdrantError(f"Qdrant connection failed: {e}") from e

        log.info("qdrant_upsert", collection=collection, count=len(points))
        return len(points)

    async def delete_by_source_id(self, collection: str, source_id: str) -> int:
        """Delete all points for a source_id. Returns estimated count deleted."""
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        # First count matching points
        count = await self._count_by_source_id(collection, source_id)

        try:
            resp = await self._client.post(
                f"/collections/{collection}/points/delete",
                json={
                    "filter": {
                        "must": [
                            {"key": "metadata.source_id", "match": {"value": source_id}},
                        ],
                    },
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise QdrantError(f"Delete failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise QdrantError(f"Qdrant connection failed: {e}") from e

        log.info("qdrant_delete", collection=collection, source_id=source_id, deleted=count)
        return count

    async def delete_by_filter(
        self,
        collection: str,
        qdrant_filter: dict,
    ) -> int:
        """Delete points matching an arbitrary Qdrant filter. Returns estimated count deleted."""
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        # Count before deleting
        count = 0
        try:
            resp = await self._client.post(
                f"/collections/{collection}/points/count",
                json={"filter": qdrant_filter, "exact": True},
            )
            resp.raise_for_status()
            count = resp.json().get("result", {}).get("count", 0)
        except Exception:
            pass

        try:
            resp = await self._client.post(
                f"/collections/{collection}/points/delete",
                json={"filter": qdrant_filter},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise QdrantError(f"Delete failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise QdrantError(f"Qdrant connection failed: {e}") from e

        log.info("qdrant_delete_by_filter", collection=collection, deleted=count)
        return count

    async def update_payload(
        self,
        collection: str,
        source_id: str,
        payload: dict,
    ) -> int:
        """Update payload fields on all points matching source_id."""
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        count = await self._count_by_source_id(collection, source_id)

        try:
            resp = await self._client.post(
                f"/collections/{collection}/points/payload",
                json={
                    "payload": payload,
                    "filter": {
                        "must": [
                            {"key": "metadata.source_id", "match": {"value": source_id}},
                        ],
                    },
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise QdrantError(f"Payload update failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise QdrantError(f"Qdrant connection failed: {e}") from e

        log.info("qdrant_payload_update", collection=collection, source_id=source_id, updated=count)
        return count

    async def search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        filters: dict | None = None,
        top_k: int = 10,
        score_threshold: float = 0.0,
        dense_vector_name: str = "dense",
    ) -> list[dict]:
        """Search for similar vectors with optional filtering."""
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        start = time.monotonic()
        body: dict = {
            "vector": {"name": dense_vector_name, "vector": dense_vector},
            "limit": top_k,
            "with_payload": True,
            "score_threshold": score_threshold,
        }

        if filters:
            body["filter"] = filters

        try:
            resp = await self._client.post(
                f"/collections/{collection}/points/search",
                json=body,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise QdrantError(f"Search failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise QdrantError(f"Qdrant connection failed: {e}") from e

        duration = int((time.monotonic() - start) * 1000)
        results = resp.json().get("result", [])
        log.info("qdrant_search", collection=collection, results=len(results), duration_ms=duration)
        return results

    async def hybrid_search(
        self,
        collection: str,
        dense_vector: list[float],
        sparse_vector: dict,
        filters: dict | None = None,
        top_k: int = 10,
        prefetch_limit: int = 20,
        dense_vector_name: str = "dense",
    ) -> list[dict]:
        """Hybrid search using RRF (Reciprocal Rank Fusion) over dense + sparse vectors."""
        if not self._client:
            raise QdrantError("Qdrant client not initialized")

        start = time.monotonic()

        prefetch = [
            {
                "query": sparse_vector,
                "using": "sparse",
                "limit": prefetch_limit,
            },
            {
                "query": dense_vector,
                "using": dense_vector_name,
                "limit": prefetch_limit,
            },
        ]

        body: dict = {
            "prefetch": prefetch,
            "query": {"fusion": "rrf"},
            "limit": top_k,
            "with_payload": True,
        }

        if filters:
            body["filter"] = filters

        try:
            resp = await self._client.post(
                f"/collections/{collection}/points/query",
                json=body,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise QdrantError(f"Hybrid search failed: {e.response.text}") from e
        except httpx.RequestError as e:
            raise QdrantError(f"Qdrant connection failed: {e}") from e

        duration = int((time.monotonic() - start) * 1000)
        results = resp.json().get("result", {}).get("points", [])
        log.info("qdrant_hybrid_search", collection=collection, results=len(results), duration_ms=duration)
        return results

    async def _count_by_source_id(self, collection: str, source_id: str) -> int:
        """Count points matching a source_id."""
        try:
            resp = await self._client.post(
                f"/collections/{collection}/points/count",
                json={
                    "filter": {
                        "must": [
                            {"key": "metadata.source_id", "match": {"value": source_id}},
                        ],
                    },
                    "exact": True,
                },
            )
            resp.raise_for_status()
            return resp.json().get("result", {}).get("count", 0)
        except Exception:
            return 0
