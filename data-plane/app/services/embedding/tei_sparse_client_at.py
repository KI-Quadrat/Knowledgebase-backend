"""TEI sparse embedding client for hybrid search.

Talks to the self-hosted sparse-embedding server at ``SPARSE_EMBED_URL_AT``
via its dedicated ``POST /embed_sparse`` endpoint, which accepts a batch of
texts in ``{"texts": [...]}`` form and returns one sparse vector per text.
Auth: bearer ``SPARSE_EMBED_API_KEY_AT`` plus optional Cloudflare Access
service-token headers (``CF-Access-Client-Id`` / ``CF-Access-Client-Secret``)
so the sparse service can live behind the same CF Access perimeter as
``embed.ki2.at``.

Used by the online ingest pipeline (``vector_config.search_mode = "hybrid"``)
and the query side of :class:`SearchService` when the caller asks for hybrid
search.
"""

import time

import httpx

from app.config import ext
from app.services.embedding.bge_m3_client import EmbeddingError
from app.utils.logger import get_logger

log = get_logger(__name__)


class SparseVector:
    """Qdrant-compatible sparse vector: parallel indices + values arrays."""

    def __init__(self, indices: list[int], values: list[float], duration_ms: int = 0):
        self.indices = indices
        self.values = values
        self.duration_ms = duration_ms

    def as_dict(self) -> dict:
        return {"indices": self.indices, "values": self.values}


def _parse_sparse_entry(entry) -> tuple[list[int], list[float]]:
    """Turn a single sparse embedding payload into (indices, values).

    Accepts the response shapes commonly emitted by TEI / SPLADE servers:
    - ``[{"index": i, "value": v}, ...]`` (OpenAI-compatible sparse output)
    - ``{"indices": [...], "values": [...]}`` (Qdrant-native)
    - ``[[i, v], ...]`` (raw pair list)
    """
    if isinstance(entry, dict) and "indices" in entry and "values" in entry:
        return list(entry["indices"]), [float(v) for v in entry["values"]]

    if isinstance(entry, list):
        indices: list[int] = []
        values: list[float] = []
        for item in entry:
            if isinstance(item, dict):
                idx = item.get("index", item.get("token_id"))
                val = item.get("value", item.get("score"))
                if idx is None or val is None:
                    continue
                indices.append(int(idx))
                values.append(float(val))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                indices.append(int(item[0]))
                values.append(float(item[1]))
        return indices, values

    raise EmbeddingError(f"TEI sparse: unrecognized embedding entry shape: {type(entry).__name__}")


class TEISparseClientAT:
    """HTTP client for the sparse embedding server (POST /embed_sparse)."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url = ext.sparse_embed_url_at.rstrip("/")
        self._api_key = ext.sparse_embed_api_key_at
        self._cf_client_id = ext.sparse_cf_access_client_id_at
        self._cf_client_secret = ext.sparse_cf_access_client_secret_at

    async def startup(self) -> None:
        if not self._api_key:
            log.warning("tei_sparse_client_at_no_key")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60),
            follow_redirects=True,
        )
        log.info(
            "tei_sparse_client_at_started",
            url=self._base_url,
            cf_access=bool(self._cf_client_id and self._cf_client_secret),
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        log.info("tei_sparse_client_at_stopped")

    async def check_health(self) -> bool:
        return bool(self._client and self._api_key)

    async def encode(self, text: str) -> SparseVector:
        return (await self.encode_batch([text]))[0]

    async def encode_batch(self, texts: list[str]) -> list[SparseVector]:
        if not self._client:
            raise EmbeddingError("TEI sparse client not initialized")
        if not self._api_key:
            raise EmbeddingError("SPARSE_EMBED_API_KEY_AT not configured")
        if not texts:
            return []

        # The TEI sparse server enforces --max-client-batch-size (default 32).
        # Split larger inputs into sequential windows; window=0 disables split.
        max_batch = ext.sparse_embed_max_batch_at or len(texts)
        start = time.monotonic()
        results: list[SparseVector] = []
        for offset in range(0, len(texts), max_batch):
            window = texts[offset : offset + max_batch]
            results.extend(await self._encode_window(window))
        duration = int((time.monotonic() - start) * 1000)

        log.info(
            "tei_sparse_encode_complete",
            count=len(texts),
            duration_ms=duration,
            windows=(len(texts) + max_batch - 1) // max_batch,
        )
        return results

    async def _encode_window(self, texts: list[str]) -> list[SparseVector]:
        """POST a single ≤max_batch window to /embed_sparse and parse the response."""
        payload: dict = {"texts": texts}

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._cf_client_id and self._cf_client_secret:
            headers["CF-Access-Client-Id"] = self._cf_client_id
            headers["CF-Access-Client-Secret"] = self._cf_client_secret

        start = time.monotonic()
        try:
            resp = await self._client.post(
                f"{self._base_url}/embed_sparse",
                headers=headers,
                json=payload,
            )
        except httpx.RequestError as e:
            raise EmbeddingError(f"TEI sparse connection error: {e}") from e

        if not resp.is_success:
            raise EmbeddingError(f"TEI sparse HTTP {resp.status_code}: {resp.text[:500]}")

        duration = int((time.monotonic() - start) * 1000)
        try:
            data = resp.json()
        except ValueError as e:
            content_type = resp.headers.get("content-type", "<none>")
            preview = (resp.text or "")[:300].replace("\n", "\\n")
            raise EmbeddingError(
                f"TEI sparse returned non-JSON body "
                f"(status={resp.status_code}, content-type={content_type!r}, "
                f"body[:300]={preview!r})"
            ) from e

        # Response shapes covered (one entry per input text, batch-ordered):
        #   list-wrapped     → [<sparse-entry>, ...]
        #   dict-wrapped     → {"embeddings": [<sparse-entry>, ...]},
        #                       {"sparse":     [<sparse-entry>, ...]} (sparse.ki2.at), or
        #                       {"data":       [{"embedding": <sparse-entry>, "index": i}, ...]}
        # A <sparse-entry> itself is one of the shapes _parse_sparse_entry handles.
        entries: list
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            if isinstance(data.get("embeddings"), list):
                entries = data["embeddings"]
            elif isinstance(data.get("sparse"), list):
                entries = data["sparse"]
            elif isinstance(data.get("data"), list):
                items = sorted(data["data"], key=lambda x: x.get("index", 0))
                entries = [item.get("embedding") for item in items]
            else:
                raise EmbeddingError(
                    f"TEI sparse response had no recognizable batch field "
                    f"(got keys: {list(data.keys())[:10]})"
                )
        else:
            raise EmbeddingError(f"TEI sparse: unexpected response type {type(data).__name__}")

        results: list[SparseVector] = []
        for entry in entries:
            indices, values = _parse_sparse_entry(entry)
            # Qdrant wants ascending indices; dedupe on clash by max value.
            pairs: dict[int, float] = {}
            for i, v in zip(indices, values, strict=False):
                if i in pairs:
                    pairs[i] = max(pairs[i], v)
                else:
                    pairs[i] = v
            sorted_indices = sorted(pairs.keys())
            results.append(
                SparseVector(
                    indices=sorted_indices,
                    values=[pairs[i] for i in sorted_indices],
                    duration_ms=duration,
                )
            )

        return results
