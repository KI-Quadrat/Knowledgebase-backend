"""BGE-M3 embedding client — calls the BGE-M3 inference server for dense+sparse embeddings."""

import time

import httpx

from app.config import ext
from app.models.common import StageUsage
from app.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_DENSE_DIM = 1024


class EmbeddingError(Exception):
    pass


class EmbeddingResult:
    def __init__(
        self,
        dense: list[float],
        sparse: dict[int, float] | None = None,
        duration_ms: int = 0,
    ):
        self.dense = dense
        self.sparse = sparse
        self.duration_ms = duration_ms


class BGEM3Client:
    """HTTP client for the BGE-M3 embedding service."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url = ext.bge_m3_url.rstrip("/")
        # ``last_usage`` is set after every ``embed_batch`` call so callers
        # (IngestService) can attribute embedding cost without a method
        # signature change. Self-hosted → always $0; the field is kept for
        # contract parity with ``OpenAIEmbedClient.last_usage``.
        self.last_usage: StageUsage | None = None

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(60),
        )
        log.info("bge_m3_client_started", url=self._base_url)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        log.info("bge_m3_client_stopped")

    async def check_health(self) -> bool:
        if not self._client:
            return False
        try:
            resp = await self._client.get("/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate dense + sparse embeddings for a single text."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for a batch of texts."""
        if not self._client:
            raise EmbeddingError("BGE-M3 client not initialized")
        if not texts:
            return []

        # Self-hosted BGE-M3 has finite GPU memory; split larger inputs into
        # sequential windows of ext.bge_m3_max_batch. 0 disables splitting.
        max_batch = ext.bge_m3_max_batch or len(texts)
        start = time.monotonic()
        results: list[EmbeddingResult] = []
        for offset in range(0, len(texts), max_batch):
            window = texts[offset : offset + max_batch]
            results.extend(await self._embed_window(window))
        duration = int((time.monotonic() - start) * 1000)

        log.info(
            "bge_m3_embed_complete",
            count=len(texts),
            duration_ms=duration,
            windows=(len(texts) + max_batch - 1) // max_batch,
        )
        self.last_usage = StageUsage(
            stage="embedding", provider="bge_m3", model="bge-m3", cost_usd=0.0
        )
        return results

    async def _embed_window(self, texts: list[str]) -> list[EmbeddingResult]:
        """POST a single ≤max_batch window to /embed and parse the response."""
        start = time.monotonic()
        try:
            resp = await self._client.post(
                "/embed",
                json={"texts": texts, "return_sparse": True},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"BGE-M3 HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise EmbeddingError(f"BGE-M3 connection error: {e}") from e

        duration = int((time.monotonic() - start) * 1000)
        data = resp.json()

        dense_embeddings = data.get("dense", [])
        sparse_embeddings = data.get("sparse", [None] * len(texts))

        return [
            EmbeddingResult(
                dense=dense_embeddings[i] if i < len(dense_embeddings) else [],
                sparse=sparse_embeddings[i] if i < len(sparse_embeddings) else None,
                duration_ms=duration,
            )
            for i in range(len(texts))
        ]
