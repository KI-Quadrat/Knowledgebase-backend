"""OpenAI embedding client — calls the OpenAI API for dense embeddings (text-embedding-3-small)."""

import time

import httpx

from app.config import ext
from app.models.common import StageUsage
from app.services import cost
from app.services.embedding.bge_m3_client import EmbeddingError, EmbeddingResult
from app.utils.logger import get_logger

log = get_logger(__name__)

OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedClient:
    """HTTP client for OpenAI embeddings API. Drop-in replacement for BGEM3Client."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._client: httpx.AsyncClient | None = None
        self._model = model
        self._api_key = ext.openai_api_key
        # Per-call usage record set after every ``embed_batch`` so callers
        # (IngestService) can read it without changing return signatures.
        # Aggregated across all windows of a single embed_batch invocation.
        self.last_usage: StageUsage | None = None

    async def startup(self) -> None:
        if not self._api_key:
            log.warning("openai_embed_client_no_key")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60))
        log.info("openai_embed_client_started", model=self._model)

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        log.info("openai_embed_client_stopped")

    async def check_health(self) -> bool:
        return bool(self._api_key)

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate dense embeddings for a single text."""
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for a batch of texts via OpenAI API."""
        if not self._client:
            raise EmbeddingError("OpenAI embed client not initialized")
        if not self._api_key:
            raise EmbeddingError("OPENAI_API_KEY not configured")
        if not texts:
            return []

        # OpenAI accepts up to 2048 inputs (~300K tokens) per request; keep
        # individual requests bounded by splitting into windows. 0 disables.
        max_batch = ext.openai_embed_max_batch or len(texts)
        start = time.monotonic()
        results: list[EmbeddingResult] = []
        total_tokens = 0
        for offset in range(0, len(texts), max_batch):
            window = texts[offset : offset + max_batch]
            window_results, window_tokens = await self._embed_window(window)
            results.extend(window_results)
            total_tokens += window_tokens
        duration = int((time.monotonic() - start) * 1000)

        log.info(
            "openai_embed_complete",
            model=self._model,
            count=len(texts),
            duration_ms=duration,
            windows=(len(texts) + max_batch - 1) // max_batch,
            tokens=total_tokens,
        )
        self.last_usage = StageUsage(
            stage="embedding",
            provider="openai",
            model=self._model,
            embed_tokens=total_tokens,
            cost_usd=cost.embed_cost("openai", self._model, tokens=total_tokens),
        )
        return results

    async def _embed_window(self, texts: list[str]) -> tuple[list[EmbeddingResult], int]:
        """POST a single ≤max_batch window to OpenAI and parse the response.

        Returns ``(results, tokens)`` so the caller can aggregate token
        usage across windows for billing.
        """
        start = time.monotonic()
        try:
            resp = await self._client.post(
                OPENAI_EMBED_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": texts,
                    "model": self._model,
                },
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"OpenAI HTTP {e.response.status_code}: {e.response.text}") from e
        except httpx.RequestError as e:
            raise EmbeddingError(f"OpenAI connection error: {e}") from e

        duration = int((time.monotonic() - start) * 1000)
        data = resp.json()

        embeddings = sorted(data.get("data", []), key=lambda x: x["index"])
        # OpenAI returns ``usage.{prompt_tokens, total_tokens}`` for embed
        # responses. ``total_tokens`` is what gets billed.
        usage_block = data.get("usage") or {}
        tokens = int(usage_block.get("total_tokens") or usage_block.get("prompt_tokens") or 0)

        return [
            EmbeddingResult(dense=item["embedding"], sparse=None, duration_ms=duration)
            for item in embeddings
        ], tokens
