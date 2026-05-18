"""Unit tests for usage parsing + UsageSummary aggregation.

Covers the parsers we added on top of provider responses:
- Jina ``meta.usage.tokens`` extraction
- Firecrawl ``creditsUsed`` extraction
- UsageSummary.from_entries / merge token & cost rollups
"""

import pytest

from app.models.common import StageUsage, UsageSummary
from app.services.scraping.crawl4ai_client import (
    _extract_firecrawl_credits,
    _extract_jina_tokens,
)


# ── Jina token extractor ────────────────────────────────────────────

def test_jina_tokens_from_top_level_meta():
    data = {"meta": {"usage": {"tokens": 29}}, "data": {"content": "x"}}
    assert _extract_jina_tokens(data, data["data"]) == 29


def test_jina_tokens_from_nested_data_section():
    data = {"data": {"content": "x", "usage": {"tokens": 42}}}
    assert _extract_jina_tokens(data, data["data"]) == 42


def test_jina_tokens_missing_returns_zero():
    data = {"data": {"content": "x"}}
    assert _extract_jina_tokens(data, data["data"]) == 0


def test_jina_tokens_zero_or_negative_treated_as_missing():
    data = {"meta": {"usage": {"tokens": 0}}}
    assert _extract_jina_tokens(data, {}) == 0


# ── Firecrawl credit extractor ──────────────────────────────────────

def test_firecrawl_credits_top_level():
    data = {"success": True, "creditsUsed": 3}
    assert _extract_firecrawl_credits(data, None) == 3.0


def test_firecrawl_credits_nested_under_metadata():
    data = {"success": True}
    result_data = {"metadata": {"creditsUsed": 7.5}}
    assert _extract_firecrawl_credits(data, result_data) == 7.5


def test_firecrawl_credits_alternate_snake_case_key():
    data = {"credits_used": 12}
    assert _extract_firecrawl_credits(data, None) == 12.0


def test_firecrawl_credits_missing_returns_zero():
    assert _extract_firecrawl_credits({"success": True}, None) == 0.0


# ── UsageSummary aggregation ────────────────────────────────────────

def test_summary_sums_tokens_and_costs():
    entries = [
        StageUsage(
            stage="classifier", provider="openai", model="gpt-4o-mini",
            prompt_tokens=1000, completion_tokens=200, cost_usd=0.000270,
        ),
        StageUsage(
            stage="contextual", provider="openai", model="gpt-4.1-nano",
            prompt_tokens=2000, completion_tokens=500, cost_usd=0.000400,
        ),
        StageUsage(
            stage="embedding", provider="bge_m3", cost_usd=0.0,
        ),
    ]
    s = UsageSummary.from_entries(entries)
    # 1000+200 + 2000+500 = 3700
    assert s.total_tokens == 3700
    assert s.total_cost_usd == pytest.approx(0.000670, rel=1e-6)
    assert set(s.by_stage.keys()) == {"classifier", "contextual", "embedding"}


def test_summary_cost_is_none_when_any_entry_unknown():
    entries = [
        StageUsage(stage="scraper", provider="firecrawl", credits=5.0, cost_usd=None),
        StageUsage(stage="classifier", provider="openai", model="gpt-4o-mini",
                   prompt_tokens=100, completion_tokens=10, cost_usd=0.000021),
    ]
    s = UsageSummary.from_entries(entries)
    assert s.total_cost_usd is None
    # Token count still rolls up correctly even when cost is unknown.
    assert s.total_tokens == 110
    # Credits also flow through to the total.
    assert s.total_credits == 5.0


def test_summary_merge_sums_per_stage_across_items():
    """Batch rollup: 3 ingest items each contributing a classifier entry."""
    items = [
        UsageSummary.from_entries([StageUsage(
            stage="classifier", provider="openai", model="gpt-4o-mini",
            prompt_tokens=1000, completion_tokens=100, cost_usd=0.000210,
        )])
        for _ in range(3)
    ]
    merged = UsageSummary.merge(items)
    assert merged.by_stage["classifier"].prompt_tokens == 3000
    assert merged.by_stage["classifier"].completion_tokens == 300
    assert merged.total_cost_usd == pytest.approx(3 * 0.000210, rel=1e-6)


def test_serialized_summary_drops_zero_cost_self_hosted_stages():
    """The slim format strips self-hosted ($0, no counts) stages from
    by_stage on serialization while keeping the Python object intact."""
    s = UsageSummary.from_entries([
        StageUsage(
            stage="contextual", provider="openai", model="gpt-4.1-nano",
            prompt_tokens=181, completion_tokens=55, cost_usd=0.0000401,
        ),
        StageUsage(stage="embedding", provider="bge_m3", model="BAAI/bge-m3", cost_usd=0.0),
        StageUsage(stage="sparse_embedding", provider="tei_sparse", cost_usd=0.0),
    ])

    # Python object retains every stage — aggregation / iteration still works.
    assert set(s.by_stage.keys()) == {"contextual", "embedding", "sparse_embedding"}

    # Serialized form keeps only the billable stage.
    dumped = s.model_dump()
    assert set(dumped["by_stage"].keys()) == {"contextual"}
    # Slim entry drops zero count fields too.
    entry = dumped["by_stage"]["contextual"]
    assert "cached_tokens" not in entry
    assert "embed_tokens" not in entry
    assert "scrape_tokens" not in entry
    assert "credits" not in entry
    assert "pages" not in entry
    assert entry["prompt_tokens"] == 181
    assert entry["completion_tokens"] == 55
    # Zero totals are also dropped.
    assert "total_credits" not in dumped
    assert "total_pages" not in dumped
    # Roll-ups always included.
    assert dumped["total_tokens"] == 236
    # UsageSummary.from_entries rounds total to 6 decimals → 0.0000401 → 0.00004
    assert dumped["total_cost_usd"] == pytest.approx(0.00004, abs=1e-6)


def test_serialized_summary_keeps_entries_with_unknown_rate():
    """Stages with ``cost_usd=None`` (rate unset in pricing.yaml) must
    survive slim filtering — the count is the operator's only signal that
    something billed but wasn't priced."""
    s = UsageSummary.from_entries([
        StageUsage(stage="scraper", provider="firecrawl", credits=5.0, cost_usd=None),
    ])
    dumped = s.model_dump()
    assert "scraper" in dumped["by_stage"]
    assert dumped["by_stage"]["scraper"]["credits"] == 5.0
    assert dumped["by_stage"]["scraper"]["cost_usd"] is None


def test_summary_merge_handles_none_entries():
    """Failed batch items have data=None and contribute nothing."""
    items = [
        UsageSummary.from_entries([StageUsage(
            stage="classifier", provider="openai", model="gpt-4o-mini",
            prompt_tokens=1000, completion_tokens=100, cost_usd=0.000210,
        )]),
        None,  # failed item
    ]
    merged = UsageSummary.merge(items)
    assert merged.by_stage["classifier"].prompt_tokens == 1000
    assert merged.total_cost_usd == pytest.approx(0.000210, rel=1e-6)
