"""LiteLLM-backed fallback for OpenAI chat completions.

Used by the classifier, contextual enricher, and funding extractor when the
primary OpenAI call fails with a rate-limit / connection / 5xx error. LiteLLM
is OpenAI-compatible, so we reuse the AsyncOpenAI client with a different
``base_url`` and ``api_key``. The fallback is disabled (returns ``None``) when
``LITELLM_URL`` or ``LITELLM_API_KEY`` is unset.
"""

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from app.config import ext
from app.utils.logger import get_logger

log = get_logger(__name__)


# Errors that should trigger a fallback to LiteLLM. These are all the
# "OpenAI is unavailable" cases — rate limit (after the in-call 429 retries
# are exhausted), network/connection, timeout, and 5xx server errors.
# BadRequestError / AuthenticationError / NotFoundError etc. are NOT in this
# set: they signal an input or config problem that the fallback can't fix.
def _outage_error_classes() -> tuple[type[Exception], ...]:
    from openai import (
        APIConnectionError,
        APITimeoutError,
        InternalServerError,
        RateLimitError,
    )
    return (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


OUTAGE_ERRORS = _outage_error_classes()


class _LiteLLMFallback:
    """Module-level singleton wrapping AsyncOpenAI pointed at LiteLLM."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None
        self._model: str = ""

    def is_enabled(self) -> bool:
        return self._client is not None

    @property
    def model(self) -> str:
        return self._model

    async def startup(self) -> None:
        if not ext.litellm_url or not ext.litellm_api_key:
            log.info(
                "litellm_fallback_disabled",
                reason="LITELLM_URL or LITELLM_API_KEY is unset",
            )
            return
        self._client = AsyncOpenAI(
            base_url=ext.litellm_url,
            api_key=ext.litellm_api_key,
        )
        self._model = ext.litellm_fallback_model
        log.info(
            "litellm_fallback_started",
            url=ext.litellm_url,
            model=self._model,
        )

    async def shutdown(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None
            log.info("litellm_fallback_stopped")

    async def chat_completion(
        self,
        *,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> ChatCompletion | None:
        """Run a chat completion against the LiteLLM fallback model.

        Returns the ``ChatCompletion`` on success, or ``None`` if the fallback
        is disabled. Raises any error from the fallback call so the caller can
        decide whether to bubble it up or swallow (e.g., the contextual
        enricher's per-chunk graceful-degradation path).
        """
        if self._client is None:
            return None
        return await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )


_fallback = _LiteLLMFallback()


async def startup() -> None:
    await _fallback.startup()


async def shutdown() -> None:
    await _fallback.shutdown()


def is_enabled() -> bool:
    return _fallback.is_enabled()


def model() -> str:
    return _fallback.model


async def chat_completion(
    *,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> ChatCompletion | None:
    return await _fallback.chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
