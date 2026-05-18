"""Unit tests for the cost layer (`pricing.yaml` + `services/cost.py`).

The pricing table is loaded once at import time from the YAML file shipping
in the repo, so these tests run against the real rates we'll bill at — the
goal is to lock down the *math* and the *fallback semantics* (rate unknown
→ ``None``, provider absent → ``0.0``).
"""

import pytest

from app.services import cost


def test_chat_cost_known_model_openai_gpt_4o_mini():
    # 1,000 prompt tokens @ $0.150/1M + 200 completion tokens @ $0.600/1M
    # = 0.000150 + 0.000120 = 0.000270
    usd = cost.chat_cost(
        "openai", "gpt-4o-mini",
        prompt_tokens=1000, completion_tokens=200, cached_tokens=0,
    )
    assert usd == pytest.approx(0.000270, rel=1e-6)


def test_chat_cost_applies_cached_input_discount():
    # 500 of the 1000 prompt tokens hit the cache at $0.025/1M (gpt-4.1-nano)
    # vs $0.100/1M for the uncached half. Output = 100 @ $0.400/1M.
    # Expected: 500*0.025/1M + 500*0.100/1M + 100*0.400/1M
    #         = 0.0000125  + 0.000050  + 0.00004 = 0.0001025
    usd = cost.chat_cost(
        "openai", "gpt-4.1-nano",
        prompt_tokens=1000, completion_tokens=100, cached_tokens=500,
    )
    assert usd == pytest.approx(0.0001025, rel=1e-6)


def test_chat_cost_unknown_model_falls_to_wildcard():
    # nebius.* has input/output set to null → cost is None (unknown rate).
    usd = cost.chat_cost("nebius", "any-llama", prompt_tokens=1000, completion_tokens=100)
    assert usd is None


def test_chat_cost_unknown_provider_is_zero():
    # Self-hosted (provider absent from pricing.yaml) → $0.
    usd = cost.chat_cost("local-vllm", "anything", prompt_tokens=1000, completion_tokens=100)
    assert usd == 0.0


def test_embed_cost_openai_small():
    # 1M tokens @ $0.020/1M = $0.020
    usd = cost.embed_cost("openai", "text-embedding-3-small", tokens=1_000_000)
    assert usd == pytest.approx(0.020, rel=1e-6)


def test_embed_cost_self_hosted_is_zero():
    usd = cost.embed_cost("bge_m3", "bge-m3", tokens=10_000)
    assert usd == 0.0


def test_jina_cost_known_rate():
    # pricing.yaml currently sets jina.*.per_token to 0.000005.
    rate = cost._rate("jina")
    assert rate is not None
    per_token = rate.get("per_token")
    if per_token is None:
        pytest.skip("Jina per_token unset in pricing.yaml — rate-unknown branch covered elsewhere")
    usd = cost.jina_cost(1000)
    assert usd == pytest.approx(1000 * per_token, rel=1e-6)


def test_jina_cost_zero_tokens_is_zero():
    assert cost.jina_cost(0) == 0.0


def test_firecrawl_cost_null_rate_returns_none():
    # pricing.yaml ships with firecrawl.*.per_credit = null (plan-dependent).
    usd = cost.firecrawl_cost(10.0)
    assert usd is None


def test_llamaparse_cost_known_rate():
    # Default rate is $0.003/page; 100 pages → $0.30.
    rate = cost._rate("llamaparse")
    assert rate is not None
    per_page = rate.get("per_page")
    if per_page is None:
        pytest.skip("llamaparse per_page unset — rate-unknown branch covered elsewhere")
    usd = cost.llamaparse_cost(100)
    assert usd == pytest.approx(100 * per_page, rel=1e-6)
