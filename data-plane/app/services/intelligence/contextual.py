"""Contextual Retrieval — generates short context for each chunk using OpenAI.

Based on Anthropic's Contextual Retrieval technique: prepends a concise context
to each chunk explaining how it fits within the whole document, improving
retrieval accuracy.

The context is generated in the same language as the content.
"""

import asyncio
import json
import time

import httpx

from app.config import ext
from app.services.intelligence import llm_router
from app.utils.logger import get_logger

log = get_logger(__name__)

# Default chat-completions URL used when the enricher is constructed before
# ``startup()`` resolves a provider — kept so unit tests that build a bare
# ``ContextualEnricher()`` (and inject their own mock client) still have a
# valid URL on the instance.
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

CONTEXT_PROMPT = """\
<document>
{document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short succinct context (2-3 sentences) to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Respond ONLY with the context, nothing else. Write the context in the same language as the content."""


BATCH_CONTEXT_SYSTEM_PROMPT = """\
You produce short situating contexts for chunks of a document, used to improve
retrieval. For each chunk, write 2-3 concise sentences explaining how that
chunk fits within the overall document. Write each context in the same language
as the source content. Return one context per chunk, in the same order as the
chunks. The response is structured JSON enforced by schema."""


class ContextualEnricher:
    """Enriches chunks with document-level context via OpenAI.

    Prefers a single batched OpenAI call per document (one context array in
    one JSON response). Falls back to the per-chunk concurrent path if the
    batched call fails, returns malformed JSON, or returns a mismatched
    number of contexts.
    """

    def __init__(self, model: str | None = None, max_concurrent: int = 10) -> None:
        self._client: httpx.AsyncClient | None = None
        self._model = model or ext.openai_model
        self._api_key = ext.openai_api_key
        self._chat_url = OPENAI_CHAT_URL
        self._provider = "openai"
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def startup(self) -> None:
        try:
            resolved = llm_router.for_contextual()
        except llm_router.LLMRouterError as exc:
            # No usable provider — leave _api_key empty so enrich_chunks
            # short-circuits to a no-op and chunks pass through unenriched.
            log.info("contextual_enricher_disabled", reason=str(exc))
            self._api_key = ""
            return
        self._model = resolved.model
        self._api_key = resolved.api_key
        self._provider = resolved.provider
        self._chat_url = llm_router.chat_completions_url(resolved)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120))
        log.info(
            "contextual_enricher_started",
            provider=resolved.provider,
            model=self._model,
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.aclose()
        log.info("contextual_enricher_stopped")

    async def enrich_chunks(
        self,
        document: str,
        chunks: list[str],
    ) -> list[str]:
        """Prepend contextual descriptions to each chunk.

        Returns a list of enriched chunks: "{context}\n\n{original_chunk}".

        Splits chunks into windows of ext.openai_contextual_max_batch and runs
        one batched OpenAI call per window in parallel. Each window falls back
        to per-chunk independently if its batched call fails (so successful
        windows aren't discarded alongside failed ones).
        """
        if not chunks:
            return chunks

        start = time.monotonic()

        # Truncate document for the prompt to stay within token limits.
        cap = ext.contextual_doc_max_chars
        if len(document) > cap:
            log.info("contextual_truncated", chars_in=len(document), chars_kept=cap)
        doc_summary = document[:cap]

        max_batch = ext.openai_contextual_max_batch or len(chunks)
        windows = [
            chunks[i : i + max_batch] for i in range(0, len(chunks), max_batch)
        ]

        window_results = await asyncio.gather(
            *(self._enrich_window(doc_summary, w) for w in windows)
        )
        enriched = [item for window_enriched, _ in window_results for item in window_enriched]
        fallbacks = sum(1 for _, fell_back in window_results if fell_back)

        duration = int((time.monotonic() - start) * 1000)
        log.info(
            "contextual_enrichment_complete",
            chunks=len(chunks),
            duration_ms=duration,
            windows=len(windows),
            fallback_windows=fallbacks,
        )
        return enriched

    async def _enrich_window(
        self, document: str, chunks: list[str]
    ) -> tuple[list[str], bool]:
        """Enrich one window of chunks. Returns (enriched, fell_back_to_per_chunk)."""
        contexts = await self._enrich_batch_single_call(document, chunks)
        if contexts is not None:
            return [
                f"{ctx}\n\n{chunk}" if ctx else chunk
                for ctx, chunk in zip(contexts, chunks)
            ], False

        # Per-chunk fallback for this window only.
        enriched = await asyncio.gather(
            *(self._enrich_single(document, chunk) for chunk in chunks)
        )
        return list(enriched), True

    async def _enrich_batch_single_call(
        self, document: str, chunks: list[str]
    ) -> list[str] | None:
        """Request all contexts in one OpenAI call.

        Returns the list of contexts on success, or None on any failure
        (network error, malformed JSON, length mismatch) so the caller can
        fall back to the per-chunk path.
        """
        if not self._api_key or not self._client:
            return None

        numbered = "\n\n".join(
            f"[chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
        )
        user_msg = (
            f"<document>\n{document}\n</document>\n\n"
            f"There are {len(chunks)} chunks below. Return one context per chunk, "
            f"in order, as a JSON object with key 'contexts'.\n\n"
            f"{numbered}"
        )

        # Budget ~160 tokens per context + overhead, capped so large docs
        # don't blow through the response limit.
        max_tokens = min(16000, 200 + 160 * len(chunks))

        # ``json_schema`` strict mode forces OpenAI's constrained sampler to
        # produce an array of exactly ``len(chunks)`` strings — schema-level
        # enforcement, not a prompt instruction. Eliminates the prior
        # ``contextual_batch_length_mismatch`` failure mode where the model
        # returned ±N too many/few entries under ``json_object`` mode.
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": BATCH_CONTEXT_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "contexts",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "contexts": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": len(chunks),
                                "maxItems": len(chunks),
                            },
                        },
                        "required": ["contexts"],
                        "additionalProperties": False,
                    },
                },
            },
        }

        raw = await self._chat_with_fallback(body, label="contextual_batch")
        if raw is None:
            return None

        try:
            data = json.loads(raw)
        except Exception as e:
            log.warning("contextual_batch_parse_failed", error=str(e))
            return None

        contexts = data.get("contexts")
        # Length mismatch is structurally impossible under strict json_schema
        # mode (the schema pins ``minItems``/``maxItems`` to ``len(chunks)``),
        # but kept as a defensive guard in case the configured provider
        # doesn't honor the schema, or strict mode is degraded in some
        # future revision. Either way we fall back to per-chunk enrichment
        # rather than mis-aligning contexts.
        if not isinstance(contexts, list) or len(contexts) != len(chunks):
            log.warning(
                "contextual_batch_length_mismatch",
                expected=len(chunks),
                got=len(contexts) if isinstance(contexts, list) else None,
            )
            return None

        return [str(c).strip() for c in contexts]

    async def _enrich_single(self, document: str, chunk: str) -> str:
        """Generate context for a single chunk and prepend it."""
        if not self._api_key or not self._client:
            return chunk

        async with self._semaphore:
            body = {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": CONTEXT_PROMPT.format(
                            document=document,
                            chunk=chunk,
                        ),
                    }
                ],
                "max_tokens": 200,
                "temperature": 0.0,
            }
            raw = await self._chat_with_fallback(body, label="contextual_enrichment")
            if raw is None:
                return chunk
            return f"{raw.strip()}\n\n{chunk}"

    async def _chat_with_fallback(self, body: dict, label: str) -> str | None:
        """POST chat completion to the resolved provider's chat endpoint.

        Returns the assistant message content on success, or ``None`` when
        the call fails for any reason. Callers handle ``None`` as
        graceful-degradation (per-chunk fallback or unenriched chunk).
        """
        try:
            resp = await self._post_with_rate_limit_retry(body)
            data = resp.json()
            # Surface token usage so we can verify OpenAI's automatic prompt
            # caching is firing on the document prefix. ``cached_tokens > 0``
            # means windows 2+ on the same doc (or a re-ingest within the
            # ~5-min cache window) reused the cached prefix at 50% input
            # cost. Field is absent on non-OpenAI providers — treat as 0.
            usage = data.get("usage") or {}
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            log.info(
                f"{label}_usage",
                prompt_tokens=usage.get("prompt_tokens"),
                cached_tokens=cached,
                completion_tokens=usage.get("completion_tokens"),
            )
            message = data["choices"][0]["message"]
            # Strict structured-outputs surface a non-null ``refusal`` field
            # when the model declines instead of returning ``content``. Treat
            # it as a soft failure so the caller falls back per-chunk.
            refusal = message.get("refusal")
            if refusal:
                log.warning(f"{label}_refused", reason=str(refusal)[:200])
                return None
            return message.get("content")
        except Exception as e:
            log.warning(f"{label}_call_failed", error=str(e), error_type=type(e).__name__)
            return None

    async def _post_with_rate_limit_retry(self, body: dict) -> httpx.Response:
        """POST to the provider's chat-completions URL, retrying up to 3x on 429.

        Raises ``httpx.HTTPStatusError`` for non-429 4xx/5xx and for the final
        429 after all retries are exhausted. Callers wrap in their own
        try/except for soft fallback behavior.
        """
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(3):
            resp = await self._client.post(
                self._chat_url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp
            last_exc = httpx.HTTPStatusError(
                f"{self._provider} rate limit (429): {resp.text[:200]}",
                request=resp.request,
                response=resp,
            )
            if attempt < 2:
                wait = 2 ** attempt
                log.warning("contextual_rate_limit_retry", attempt=attempt + 1, wait_s=wait)
                await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc
