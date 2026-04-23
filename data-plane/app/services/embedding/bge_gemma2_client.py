"""BGE-Gemma2 embedding client — calls a self-hosted LiteLLM proxy for dense embeddings.

LiteLLM exposes an OpenAI-compatible ``POST /v1/embeddings`` endpoint,
so this client uses the same request/response format as the OpenAI API.
"""

import time

import httpx

from app.config import ext
from app.services.embedding.bge_m3_client import EmbeddingError, EmbeddingResult
from app.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_DENSE_DIM = 3584


class BGEGemma2Client:
    """HTTP client for BGE-multilingual-gemma2 via self-hosted LiteLLM proxy.

    Used as a fallback dense embedder when OpenAI is unavailable.
    Communicates with LiteLLM's OpenAI-compatible embeddings endpoint.
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url = ext.litellm_url.rstrip("/")
        self._model = ext.bge_gemma2_model
        self._api_key = ext.litellm_api_key

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120),
        )
        log.info("bge_gemma2_client_started", url=self._base_url, model=self._model)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        log.info("bge_gemma2_client_stopped")

    async def check_health(self) -> bool:
        if not self._client:
            return False
        try:
            headers = {}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            resp = await self._client.get(f"{self._base_url}/health", headers=headers, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate dense embeddings for a single text."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate dense embeddings for a batch of texts via LiteLLM proxy."""
        if not self._client:
            raise EmbeddingError("BGE-Gemma2 client not initialized")

        start = time.monotonic()
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            resp = await self._client.post(
                f"{self._base_url}/v1/embeddings",
                headers=headers,
                json={
                    "input": texts,
                    "model": self._model,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"BGE-Gemma2 HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise EmbeddingError(f"BGE-Gemma2 connection error: {e}") from e

        duration = int((time.monotonic() - start) * 1000)
        data = resp.json()

        embeddings = sorted(data.get("data", []), key=lambda x: x["index"])

        results = []
        for item in embeddings:
            results.append(EmbeddingResult(
                dense=item["embedding"],
                sparse=None,
                duration_ms=duration,
            ))

        log.info("bge_gemma2_embed_complete", model=self._model, count=len(texts), duration_ms=duration)
        return results
