"""Tests for ContextualEnricher's strict JSON schema mode.

Focus is on the request shape sent to OpenAI and how the response (or a
refusal) is consumed. We mock the underlying HTTP layer rather than
hitting OpenAI.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.intelligence.contextual import ContextualEnricher


@pytest.fixture
def enricher():
    e = ContextualEnricher(model="gpt-4o-mini")
    # Provide a non-empty API key so the production branch is exercised; the
    # actual HTTP POST is patched out per-test.
    e._api_key = "test-key"
    e._client = MagicMock()
    return e


def _ok_response(contexts: list[str]) -> MagicMock:
    """Build an httpx.Response-shaped mock with a chat completion body."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"contexts": ' + repr(contexts).replace("'", '"') + "}",
                    "refusal": None,
                }
            }
        ]
    }
    return resp


def _refusal_response(reason: str) -> MagicMock:
    """Build an httpx.Response-shaped mock with a refusal."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "refusal": reason,
                }
            }
        ]
    }
    return resp


@pytest.mark.asyncio
async def test_batch_request_uses_json_schema_strict_mode(enricher):
    """The outgoing OpenAI request must use ``json_schema`` strict mode with
    minItems/maxItems pinned to the chunk count — the whole reason for the
    migration."""
    chunks = ["chunk-a", "chunk-b", "chunk-c"]
    posted_body: dict = {}

    async def _capture_post(*_args, **kwargs):
        posted_body.update(kwargs["json"])
        return _ok_response(["ctx-a", "ctx-b", "ctx-c"])

    enricher._client.post = AsyncMock(side_effect=_capture_post)

    result = await enricher._enrich_batch_single_call("the document", chunks)

    assert result == ["ctx-a", "ctx-b", "ctx-c"]

    rf = posted_body["response_format"]
    assert rf["type"] == "json_schema"
    schema = rf["json_schema"]
    assert schema["strict"] is True
    contexts_field = schema["schema"]["properties"]["contexts"]
    assert contexts_field["minItems"] == len(chunks)
    assert contexts_field["maxItems"] == len(chunks)
    assert schema["schema"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_batch_handles_refusal_by_returning_none(enricher):
    """Strict-mode responses can surface a ``refusal`` field instead of
    ``content``. The caller must return None so the per-window code falls
    back to per-chunk enrichment."""
    enricher._client.post = AsyncMock(return_value=_refusal_response("model declined"))

    result = await enricher._enrich_batch_single_call(
        "the document", ["chunk-a", "chunk-b"]
    )

    assert result is None


@pytest.mark.asyncio
async def test_batch_length_mismatch_guard_still_fires_defensively(enricher):
    """The length-mismatch check is documented as defensive — it shouldn't
    fire under strict mode in practice, but if the API does return the wrong
    length for any reason, we still fall back rather than mis-align contexts
    onto chunks."""
    # Force a mismatched response by hand (simulating a non-strict provider
    # on the LiteLLM fallback path, or a future API quirk).
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"contexts": ["only-one"]}',
                    "refusal": None,
                }
            }
        ]
    }
    enricher._client.post = AsyncMock(return_value=resp)

    result = await enricher._enrich_batch_single_call(
        "the document", ["chunk-a", "chunk-b", "chunk-c"]
    )

    # 3 chunks asked for, 1 returned → fall back.
    assert result is None


@pytest.mark.asyncio
async def test_batch_malformed_json_falls_back(enricher):
    """JSON parsing errors (e.g. truncation) still trigger fallback."""
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "choices": [
            {"message": {"content": '{"contexts": ["incompl', "refusal": None}}
        ]
    }
    enricher._client.post = AsyncMock(return_value=resp)

    result = await enricher._enrich_batch_single_call("doc", ["a", "b"])
    assert result is None


@pytest.mark.asyncio
async def test_enrich_chunks_prepends_context_to_each_chunk(enricher):
    """End-to-end: one window of 2 chunks → 2 enriched outputs of the form
    '{context}\\n\\n{original chunk}'."""
    enricher._client.post = AsyncMock(
        return_value=_ok_response(["context for A", "context for B"])
    )

    enriched = await enricher.enrich_chunks(
        document="the full document",
        chunks=["chunk A content", "chunk B content"],
    )

    assert enriched == [
        "context for A\n\nchunk A content",
        "context for B\n\nchunk B content",
    ]
