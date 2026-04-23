"""TEI embedding client for the AT funding-assistant ingest pipeline.

Talks to a self-hosted Text Embeddings Inference server exposing an
OpenAI-compatible ``POST /v1/embeddings`` endpoint. Bearer auth is required.

Used only by ``POST /api/v1/online/ingest/at`` — the AT funding collection is
pre-created with a 1024-dim vector, so the TEI model behind ``TEI_EMBED_URL_AT``
must produce 1024-dim outputs.
"""

import time

import httpx

from app.config import ext
from app.services.embedding.bge_m3_client import EmbeddingError, EmbeddingResult
from app.utils.logger import get_logger

log = get_logger(__name__)


class TEIEmbedClientAT:
    """HTTP client for the AT TEI embedding server (OpenAI-compatible)."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._base_url = ext.tei_embed_url_at.rstrip("/")
        self._api_key = ext.tei_embed_api_key_at
        self._model = ext.tei_embed_model_at
        self._cf_client_id = ext.tei_cf_access_client_id_at
        self._cf_client_secret = ext.tei_cf_access_client_secret_at

    async def startup(self) -> None:
        if not self._api_key:
            log.warning("tei_embed_client_at_no_key")
        # follow_redirects=True handles reverse proxies that 302/307 POSTs to
        # a canonical URL (trailing-slash normalization, http→https, etc.).
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60),
            follow_redirects=True,
        )
        log.info(
            "tei_embed_client_at_started",
            url=self._base_url,
            model=self._model or "<server-default>",
            cf_access=bool(self._cf_client_id and self._cf_client_secret),
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        log.info("tei_embed_client_at_stopped")

    async def check_health(self) -> bool:
        return bool(self._client and self._api_key)

    async def embed(self, text: str) -> EmbeddingResult:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        if not self._client:
            raise EmbeddingError("TEI AT embed client not initialized")
        if not self._api_key:
            raise EmbeddingError("TEI_EMBED_API_KEY_AT not configured")

        # The embed.ki2.at deployment returns an empty 200 body when ``model``
        # is missing, so always send it — default comes from TEI_EMBED_MODEL_AT.
        payload: dict = {"input": texts, "model": self._model}

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
                f"{self._base_url}/v1/embeddings",
                headers=headers,
                json=payload,
            )
        except httpx.RequestError as e:
            raise EmbeddingError(f"TEI AT connection error: {e}") from e

        if resp.is_redirect:
            # Shouldn't happen with follow_redirects=True, but surface the
            # Location header if it does so the misconfiguration is obvious.
            location = resp.headers.get("location", "<none>")
            raise EmbeddingError(
                f"TEI AT HTTP {resp.status_code} redirect to {location} (not followed)"
            )
        if not resp.is_success:
            raise EmbeddingError(f"TEI AT HTTP {resp.status_code}: {resp.text[:500]}")

        duration = int((time.monotonic() - start) * 1000)
        try:
            data = resp.json()
        except ValueError as e:
            # A 2xx with a non-JSON body almost always means:
            #   - CF Access interstitial (HTML login page) leaked through
            #   - wrong endpoint path — proxy returned a default page
            #   - TEI server bug (empty body on timeout / model still loading)
            content_type = resp.headers.get("content-type", "<none>")
            preview = (resp.text or "")[:300].replace("\n", "\\n")
            raise EmbeddingError(
                f"TEI AT returned non-JSON body "
                f"(status={resp.status_code}, content-type={content_type!r}, "
                f"body[:300]={preview!r})"
            ) from e

        items = sorted(data.get("data", []), key=lambda x: x["index"])
        if not items:
            preview = str(data)[:300]
            raise EmbeddingError(
                f"TEI AT response had no 'data' entries (got: {preview})"
            )

        results = [
            EmbeddingResult(dense=item["embedding"], sparse=None, duration_ms=duration)
            for item in items
        ]
        log.info("tei_embed_at_complete", count=len(texts), duration_ms=duration)
        return results
