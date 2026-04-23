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
from app.utils.logger import get_logger

log = get_logger(__name__)

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
as the source content.

Respond ONLY with valid JSON of the exact shape:
{"contexts": ["<context for chunk 1>", "<context for chunk 2>", ...]}

The array length MUST equal the number of chunks, and contexts MUST be in the
same order as the chunks."""


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
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def startup(self) -> None:
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120))
        log.info("contextual_enricher_started", model=self._model)

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

        Returns a list of enriched chunks: "{context}\n\n{original_chunk}"
        """
        if not chunks:
            return chunks

        start = time.monotonic()

        # Truncate document to ~6000 chars for the prompt to stay within token limits
        doc_summary = document[:6000]

        # Try the batched single-call path first.
        contexts = await self._enrich_batch_single_call(doc_summary, chunks)
        if contexts is not None:
            enriched = [
                f"{ctx}\n\n{chunk}" if ctx else chunk
                for ctx, chunk in zip(contexts, chunks)
            ]
            duration = int((time.monotonic() - start) * 1000)
            log.info(
                "contextual_enrichment_complete",
                chunks=len(chunks),
                duration_ms=duration,
                mode="batch",
            )
            return enriched

        # Fallback: per-chunk parallel calls (higher cost, used only when batch fails).
        tasks = [
            self._enrich_single(doc_summary, chunk)
            for chunk in chunks
        ]
        enriched = await asyncio.gather(*tasks)

        duration = int((time.monotonic() - start) * 1000)
        log.info(
            "contextual_enrichment_complete",
            chunks=len(chunks),
            duration_ms=duration,
            mode="per_chunk_fallback",
        )
        return enriched

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

        try:
            resp = await self._client.post(
                OPENAI_CHAT_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": BATCH_CONTEXT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(raw)
        except Exception as e:
            log.warning("contextual_batch_call_failed", error=str(e))
            return None

        contexts = data.get("contexts")
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
            try:
                resp = await self._client.post(
                    OPENAI_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
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
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                context = data["choices"][0]["message"]["content"].strip()
                return f"{context}\n\n{chunk}"
            except Exception as e:
                log.warning("contextual_enrichment_failed", error=str(e))
                return chunk
