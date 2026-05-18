"""Cost computation from ``pricing.yaml``.

Loaded once at import time — tag-based deploys restart the container, so
hot reload isn't needed. Edit ``pricing.yaml``, tag, deploy.

A provider absent from the YAML returns ``cost_usd = 0.0`` (treated as
self-hosted). A provider present with a ``null`` rate returns ``None``
(unknown / plan-dependent) — the API response still carries the raw
token / credit / page count, so accounting works even without a rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from app.utils.logger import get_logger

log = get_logger(__name__)

_PRICING_PATH = Path(__file__).resolve().parents[2] / "pricing.yaml"


def _load_pricing() -> dict[str, dict[str, dict[str, Any]]]:
    try:
        with _PRICING_PATH.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        log.warning("pricing_file_missing", path=str(_PRICING_PATH))
        return {}
    except Exception as exc:
        log.error("pricing_file_load_failed", path=str(_PRICING_PATH), error=str(exc))
        return {}
    if not isinstance(data, dict):
        log.warning("pricing_file_malformed", path=str(_PRICING_PATH))
        return {}
    return data


_PRICING: dict[str, dict[str, dict[str, Any]]] = _load_pricing()


def _rate(provider: str, model: str | None = None) -> dict[str, Any] | None:
    """Return the rate dict for (provider, model), or None if not in the table.

    Lookup order:
      1. exact ``provider.model`` entry
      2. ``provider."*"`` wildcard entry
      3. ``None`` — provider missing from YAML → caller treats as self-hosted ($0).
    """
    table = _PRICING.get(provider)
    if not isinstance(table, dict):
        return None
    if model and model in table and isinstance(table[model], dict):
        return table[model]
    star = table.get("*")
    return star if isinstance(star, dict) else None


def _per_million(tokens: int, rate: float | None) -> float | None:
    if rate is None or tokens <= 0:
        return 0.0 if tokens <= 0 else None
    return tokens * rate / 1_000_000


def chat_cost(
    provider: str,
    model: str | None,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cached_tokens: int = 0,
) -> float | None:
    """Compute USD cost for one chat completion.

    ``cached_tokens`` is the OpenAI auto-cache hit count (subset of
    ``prompt_tokens``). The uncached portion is billed at ``input_per_1m``
    and the cached portion at ``cached_input_per_1m`` (falls back to
    ``input_per_1m`` when the cached rate isn't set).

    Returns ``None`` when the rate is unknown (provider/model absent or
    pinned to ``null``); returns ``0.0`` for self-hosted (provider not in
    the YAML at all).
    """
    rate = _rate(provider, model)
    if rate is None:
        return 0.0   # self-hosted / unbilled
    input_rate = rate.get("input_per_1m")
    output_rate = rate.get("output_per_1m")
    if input_rate is None or output_rate is None:
        return None
    cached_rate = rate.get("cached_input_per_1m") or input_rate
    uncached_in = max(prompt_tokens - cached_tokens, 0)
    return (
        uncached_in * input_rate
        + cached_tokens * cached_rate
        + completion_tokens * output_rate
    ) / 1_000_000


def embed_cost(provider: str, model: str | None, *, tokens: int) -> float | None:
    """USD cost for an embedding call. ``tokens`` is total input tokens."""
    rate = _rate(provider, model)
    if rate is None:
        return 0.0
    embed_rate = rate.get("embed_per_1m")
    if embed_rate is None:
        return None
    return _per_million(tokens, embed_rate)


def jina_cost(tokens: int) -> float | None:
    """USD cost for a Jina Reader call (``meta.usage.tokens``)."""
    rate = _rate("jina")
    if rate is None:
        return 0.0
    per_token = rate.get("per_token")
    if per_token is None:
        return None
    return tokens * per_token if tokens > 0 else 0.0


def firecrawl_cost(credits: float) -> float | None:
    """USD cost for a Firecrawl call (credits reported by the API)."""
    rate = _rate("firecrawl")
    if rate is None:
        return 0.0
    per_credit = rate.get("per_credit")
    if per_credit is None:
        return None
    return credits * per_credit if credits > 0 else 0.0


def llamaparse_cost(pages: int) -> float | None:
    """USD cost for a LlamaParse call (``pages_parsed``)."""
    rate = _rate("llamaparse")
    if rate is None:
        return 0.0
    per_page = rate.get("per_page")
    if per_page is None:
        return None
    return pages * per_page if pages > 0 else 0.0
