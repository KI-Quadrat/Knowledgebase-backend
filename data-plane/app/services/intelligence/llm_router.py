"""Per-task LLM model router across OpenAI-compatible providers.

Model spec format: ``"<provider>/<model_id>"`` — e.g. ``"openai/gpt-4o-mini"``,
``"nebius/Qwen/Qwen2.5-72B-Instruct"``, ``"together/meta-llama/Llama-3.1-70B"``.
A bare model name with no slash is interpreted as the ``openai`` provider
(backwards-compat with the previous ``OPENAI_MODEL`` env).

Each provider is configured via two env vars:

* ``DP_<PROVIDER>_BASE_URL`` — OpenAI-compatible base URL (optional for
  built-in providers — the default is used if unset)
* ``DP_<PROVIDER>_API_KEY`` — bearer token (required)

A small registry ships with the public endpoints for common providers so you
only have to set the API key. Unknown providers require an explicit base URL.

Per-task selection lives in env:

* ``DP_CLASSIFIER_MODEL``
* ``DP_CONTEXTUAL_MODEL``
* ``DP_FUNDING_MODEL``

Each falls back to ``DP_OPENAI_MODEL`` (the global default) when unset.

The router replaces the per-call ``AsyncOpenAI(api_key=ext.openai_api_key)``
construction in classifier.py / contextual.py / funding_extractor.py, and
supersedes the prior LiteLLM-fallback module — any provider is now a
primary, not a fallback.
"""

from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config import ext
from app.utils.logger import get_logger

log = get_logger(__name__)


# Public endpoints for known OpenAI-compatible providers. Env-var override
# (``DP_<PROVIDER>_BASE_URL``) always wins over the entry here. Add a
# provider by appending here AND declaring its base_url/api_key fields on
# ``ExternalSettings`` in app/config.py.
KNOWN_PROVIDERS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "nebius": "https://api.studio.nebius.ai/v1",
    "together": "https://api.together.xyz/v1",
    "groq": "https://api.groq.com/openai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
}

_OPENAI_DEFAULT_URL = KNOWN_PROVIDERS["openai"]


class LLMRouterError(Exception):
    """Raised when a model spec can't be resolved to a usable provider."""


@dataclass(frozen=True)
class Resolved:
    """Outcome of resolving a model spec to a concrete provider config.

    ``base_url`` is ``None`` when the resolved endpoint is OpenAI's default
    — both the AsyncOpenAI SDK and raw httpx callers can leave it unset in
    that case (the SDK has its own default).
    """
    provider: str
    model: str
    base_url: str | None
    api_key: str


def parse_spec(spec: str) -> tuple[str, str]:
    """Split a ``"<provider>/<model_id>"`` spec into its parts.

    A bare model name (no slash) is interpreted as the ``openai`` provider
    — this keeps the legacy ``OPENAI_MODEL=gpt-4o-mini`` shape working.
    """
    if not spec:
        raise LLMRouterError("Empty model spec")
    if "/" not in spec:
        return "openai", spec
    provider, _, model_id = spec.partition("/")
    if not provider or not model_id:
        raise LLMRouterError(f"Malformed model spec: {spec!r}")
    return provider, model_id


def _provider_base_url(provider: str) -> str | None:
    override = getattr(ext, f"{provider}_base_url", "") or ""
    if override:
        return None if override.rstrip("/") == _OPENAI_DEFAULT_URL else override.rstrip("/")
    if provider in KNOWN_PROVIDERS:
        url = KNOWN_PROVIDERS[provider]
        return None if url == _OPENAI_DEFAULT_URL else url
    raise LLMRouterError(
        f"Unknown provider {provider!r}. Set DP_{provider.upper()}_BASE_URL "
        f"and DP_{provider.upper()}_API_KEY, or use a built-in: "
        f"{', '.join(sorted(KNOWN_PROVIDERS))}."
    )


def _provider_api_key(provider: str) -> str:
    return getattr(ext, f"{provider}_api_key", "") or ""


def resolve(spec: str) -> Resolved:
    """Resolve a ``"<provider>/<model_id>"`` spec to provider config + key."""
    provider, model = parse_spec(spec)
    api_key = _provider_api_key(provider)
    if not api_key:
        raise LLMRouterError(
            f"Missing DP_{provider.upper()}_API_KEY for provider {provider!r} "
            f"(model spec: {spec!r})"
        )
    return Resolved(
        provider=provider,
        model=model,
        base_url=_provider_base_url(provider),
        api_key=api_key,
    )


def _resolve_task(task_spec: str, default_spec: str) -> Resolved:
    return resolve(task_spec or default_spec)


def for_classifier() -> Resolved:
    return _resolve_task(ext.classifier_model, ext.openai_model)


def for_contextual() -> Resolved:
    return _resolve_task(ext.contextual_model, ext.openai_model)


def for_funding() -> Resolved:
    return _resolve_task(ext.funding_model, ext.openai_model)


# Shared client cache keyed by (provider, base_url, api_key). Reusing the
# AsyncOpenAI client matters because each instance owns an httpx connection
# pool — constructing one per call leaks sockets.
_clients: dict[tuple[str, str | None, str], AsyncOpenAI] = {}


def get_client(resolved: Resolved) -> AsyncOpenAI:
    """Return a shared AsyncOpenAI client for the resolved provider."""
    key = (resolved.provider, resolved.base_url, resolved.api_key)
    client = _clients.get(key)
    if client is None:
        client = AsyncOpenAI(
            base_url=resolved.base_url,
            api_key=resolved.api_key,
        )
        _clients[key] = client
        log.info(
            "llm_router_client_created",
            provider=resolved.provider,
            base_url=resolved.base_url or _OPENAI_DEFAULT_URL,
        )
    return client


def chat_completions_url(resolved: Resolved) -> str:
    """Build the raw chat-completions URL for callers using httpx directly."""
    base = (resolved.base_url or _OPENAI_DEFAULT_URL).rstrip("/")
    return f"{base}/chat/completions"


async def close_all() -> None:
    """Close every cached AsyncOpenAI client. Call from lifespan shutdown."""
    for client in list(_clients.values()):
        try:
            await client.close()
        except Exception as exc:
            log.warning("llm_router_close_failed", error=str(exc))
    _clients.clear()
    log.info("llm_router_stopped")
