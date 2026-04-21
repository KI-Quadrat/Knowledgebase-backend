"""Tests proving the online-ingest speedup changes actually parallelize work.

Covers:
- ``IngestService`` runs primary and fallback embedders concurrently.
- ``IngestService`` awaits ``deferred_metadata_task`` and merges its result
  into each Qdrant point's payload; failures are non-fatal.
- ``ContextualEnricher.enrich_chunks`` prefers a single batched OpenAI call
  and falls back to per-chunk calls on length mismatch / malformed JSON /
  HTTP error.

Uses ``asyncio.run`` inside sync tests so the suite works without
pytest-asyncio being installed.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock

import httpx

from app.services.embedding.bge_m3_client import EmbeddingError, EmbeddingResult
from app.services.ingest.ingest_service import IngestError, IngestService
from app.services.intelligence.chunker import Chunker
from app.services.intelligence.contextual import ContextualEnricher


# ─────────────────────────────── helpers ────────────────────────────────


def _dummy_embeddings(chunks: list[str], dim: int = 4) -> list[EmbeddingResult]:
    """Stable dense vectors so reordering doesn't affect equality checks."""
    return [EmbeddingResult(dense=[float(i)] * dim) for i, _ in enumerate(chunks)]


def _build_ingest_service(
    primary_sleep: float = 0.0,
    fallback_sleep: float = 0.0,
    primary_error: Exception | None = None,
    fallback_error: Exception | None = None,
) -> tuple[IngestService, MagicMock]:
    """Build an IngestService with mocked collaborators. Returns (service, qdrant)."""

    async def primary_embed(chunks):
        if primary_sleep:
            await asyncio.sleep(primary_sleep)
        if primary_error:
            raise primary_error
        return _dummy_embeddings(chunks)

    async def fallback_embed(chunks):
        if fallback_sleep:
            await asyncio.sleep(fallback_sleep)
        if fallback_error:
            raise fallback_error
        return _dummy_embeddings(chunks)

    primary = MagicMock()
    primary.embed_batch = AsyncMock(side_effect=primary_embed)

    fallback = MagicMock()
    fallback.embed_batch = AsyncMock(side_effect=fallback_embed)

    qdrant = MagicMock()
    qdrant.create_collection = AsyncMock()
    qdrant.delete_by_source_id = AsyncMock(return_value=0)
    qdrant.upsert_points = AsyncMock(side_effect=lambda _c, points: len(points))

    classifier = MagicMock()  # unused when content_type is supplied
    chunker = Chunker()
    enricher = None  # not using contextual strategy in these tests

    service = IngestService(
        chunker=chunker,
        classifier=classifier,
        embedder=primary,
        qdrant=qdrant,
        contextual_enricher=enricher,
        fallback_embedder=fallback,
    )
    return service, qdrant


def _base_ingest_kwargs(**overrides):
    base = dict(
        source_id="doc_1",
        file_path="https://example.test/doc",
        content="Some content to embed. " * 20,
        acl=None,
        metadata={"title": "Test"},
        collection_name="test_coll",
        language="de",
        chunking_strategy="fixed",
        max_chunk_size=256,
        chunk_overlap=0,
        vector_size=4,
        search_mode="semantic",
        fallback_dense_dim=4,  # enables multi-vector path
        content_type=["general"],
    )
    base.update(overrides)
    return base


# ──────────────────────── parallel embedders ─────────────────────────────


def test_embedders_run_in_parallel():
    """Primary + fallback should run concurrently — embed phase wall time ≈ max,
    not sum. Using 0.3s each: serial would be ~600ms, parallel ~300ms.

    We assert on ``embedding_time_ms`` specifically so unrelated setup cost
    (mock construction, chunking) doesn't make the test flaky."""
    service, _ = _build_ingest_service(primary_sleep=0.3, fallback_sleep=0.3)

    async def run():
        return await service.ingest(**_base_ingest_kwargs())

    result = asyncio.run(run())

    assert result.vectors_stored > 0
    assert result.embedding_time_ms < 500, (
        f"Expected parallel embed (<500ms), got {result.embedding_time_ms}ms "
        f"(serial baseline would be ≥600ms)"
    )


# ─────────────────────── fallback vector generation ──────────────────────


def test_fallback_vector_written_when_primary_fails():
    """When primary (OpenAI) embed fails but fallback (BGE-Gemma2) succeeds,
    each Qdrant point must still be upserted carrying only the ``dense_bge_gemma2``
    vector — no ``dense_openai`` key."""
    service, qdrant = _build_ingest_service(
        primary_error=EmbeddingError("OpenAI 500"),
    )

    result = asyncio.run(service.ingest(**_base_ingest_kwargs()))

    assert result.vectors_stored > 0
    points = qdrant.upsert_points.await_args.args[1]
    assert len(points) > 0
    for p in points:
        vectors = p["vector"]
        assert "dense_bge_gemma2" in vectors, "Fallback vector must be stored"
        assert vectors["dense_bge_gemma2"], "Fallback vector must be non-empty"
        assert "dense_openai" not in vectors, (
            "Primary vector must be absent when primary embedder failed"
        )


def test_primary_vector_written_when_fallback_fails():
    """Symmetric case: primary succeeds, fallback fails → point stored with only
    ``dense_openai``, no ``dense_bge_gemma2`` key."""
    service, qdrant = _build_ingest_service(
        fallback_error=EmbeddingError("LiteLLM timeout"),
    )

    result = asyncio.run(service.ingest(**_base_ingest_kwargs()))

    assert result.vectors_stored > 0
    points = qdrant.upsert_points.await_args.args[1]
    for p in points:
        vectors = p["vector"]
        assert "dense_openai" in vectors
        assert vectors["dense_openai"]
        assert "dense_bge_gemma2" not in vectors


def test_both_embedders_produce_both_vectors():
    """Happy path: both embedders succeed → each point carries both vectors."""
    service, qdrant = _build_ingest_service()

    asyncio.run(service.ingest(**_base_ingest_kwargs()))

    points = qdrant.upsert_points.await_args.args[1]
    for p in points:
        vectors = p["vector"]
        assert "dense_openai" in vectors and vectors["dense_openai"]
        assert "dense_bge_gemma2" in vectors and vectors["dense_bge_gemma2"]


def test_both_embedders_fail_raises_ingest_error():
    """If both primary and fallback fail, ingest raises and nothing is upserted."""
    service, qdrant = _build_ingest_service(
        primary_error=EmbeddingError("OpenAI 500"),
        fallback_error=EmbeddingError("LiteLLM timeout"),
    )

    try:
        asyncio.run(service.ingest(**_base_ingest_kwargs()))
    except IngestError as e:
        assert e.code == "EMBEDDING_FAILED"
    else:
        raise AssertionError("Expected IngestError when both embedders fail")

    qdrant.upsert_points.assert_not_awaited()


# ─────────────────────── deferred funding metadata ────────────────────────


def test_deferred_metadata_merged_into_points():
    """deferred_metadata_task's result is merged into every Qdrant point payload,
    and request-supplied metadata wins on conflicts."""
    service, qdrant = _build_ingest_service()

    async def run():
        async def _deferred():
            await asyncio.sleep(0.05)
            return {
                "funding_type": "grant",
                "country_code": "AT",
                # Request metadata should overwrite this conflicting value:
                "title": "Funding-extractor title (should NOT win)",
            }

        task = asyncio.create_task(_deferred())
        kwargs = _base_ingest_kwargs(metadata={"title": "Request title (should win)"})
        return await service.ingest(**kwargs, deferred_metadata_task=task)

    result = asyncio.run(run())

    assert result.vectors_stored > 0
    upsert_call = qdrant.upsert_points.await_args
    points = upsert_call.args[1]

    for p in points:
        meta = p["payload"]["metadata"]
        assert meta["funding_type"] == "grant"
        assert meta["country_code"] == "AT"
        assert meta["title"] == "Request title (should win)"


def test_deferred_metadata_runs_concurrently_with_ingest():
    """The deferred task should overlap with chunking/embedding, not serialize after.

    We make the deferred task sleep 0.3s and the embedders also sleep 0.3s
    each (but parallel, so ~0.3s). Total should be ~0.3s, not ~0.6s."""
    service, _ = _build_ingest_service(primary_sleep=0.3, fallback_sleep=0.3)

    async def run():
        async def _slow_deferred():
            await asyncio.sleep(0.3)
            return {"extra": "value"}

        task = asyncio.create_task(_slow_deferred())
        start = time.monotonic()
        await service.ingest(**_base_ingest_kwargs(), deferred_metadata_task=task)
        return time.monotonic() - start

    elapsed = asyncio.run(run())

    assert elapsed < 0.55, (
        f"Deferred task should overlap with embeds (<0.55s), got {elapsed:.3f}s"
    )


def test_entities_stamped_into_point_metadata():
    """Caller-supplied entities (dates, deadlines, amounts, contacts, departments)
    are written as entity_* fields on every Qdrant point and capped per field."""
    service, qdrant = _build_ingest_service()

    supplied_entities = {
        "dates": [f"2026-01-{d:02d}" for d in range(1, 15)],  # 14 → capped at 10
        "deadlines": ["2026-06-30", "2026-07-15", "2026-08-01"],
        "amounts": ["EUR 1.000", "EUR 2.000", "€ 3.500"],
        "contacts": ["a@b.at", "c@d.de"],
        "departments": ["Umweltamt", "Bürgerservice"],
    }

    async def run():
        return await service.ingest(**_base_ingest_kwargs(), entities=supplied_entities)

    result = asyncio.run(run())

    assert result.vectors_stored > 0
    points = qdrant.upsert_points.await_args.args[1]
    meta = points[0]["payload"]["metadata"]

    assert len(meta["entity_dates"]) == 10  # capped at 10
    assert meta["entity_deadlines"] == ["2026-06-30", "2026-07-15", "2026-08-01"]
    assert meta["entity_amounts"] == ["EUR 1.000", "EUR 2.000", "€ 3.500"]
    assert meta["entity_contacts"] == ["a@b.at", "c@d.de"]
    assert meta["entity_departments"] == ["Umweltamt", "Bürgerservice"]


def test_entities_omitted_leaves_no_entity_fields():
    """When entities is None and no classifier result is produced (content_type
    path), point metadata carries no entity_* keys."""
    service, qdrant = _build_ingest_service()

    asyncio.run(service.ingest(**_base_ingest_kwargs(), entities=None))

    points = qdrant.upsert_points.await_args.args[1]
    meta = points[0]["payload"]["metadata"]
    for key in ("entity_dates", "entity_deadlines", "entity_amounts", "entity_contacts", "entity_departments"):
        assert key not in meta, f"{key} should not be present when entities is omitted"


def test_deferred_metadata_failure_is_non_fatal():
    """If the deferred task raises, ingest still succeeds — points just lack the extra fields."""
    service, qdrant = _build_ingest_service()

    async def run():
        async def _broken():
            raise RuntimeError("funding extractor blew up")

        task = asyncio.create_task(_broken())
        return await service.ingest(**_base_ingest_kwargs(), deferred_metadata_task=task)

    result = asyncio.run(run())

    assert result.vectors_stored > 0
    points = qdrant.upsert_points.await_args.args[1]
    # No extra fields leaked in — just the request-supplied title.
    assert points[0]["payload"]["metadata"]["title"] == "Test"


# ────────────────────── batched contextual enrichment ──────────────────────


def _build_enricher_with_mock_client(post_side_effect) -> ContextualEnricher:
    enricher = ContextualEnricher(max_concurrent=4)
    enricher._api_key = "test-key"
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.post = AsyncMock(side_effect=post_side_effect)
    enricher._client = mock_client
    return enricher


def _make_openai_response(content: str) -> httpx.Response:
    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"choices": [{"message": {"content": content}}]}
    return httpx.Response(200, request=req, json=body)


def test_contextual_batch_single_call_is_preferred():
    """One batched call returns N contexts → exactly one HTTP call,
    and each chunk gets its corresponding context prepended."""
    chunks = ["First chunk text.", "Second chunk text.", "Third chunk text."]
    payload = json.dumps({
        "contexts": ["Ctx A", "Ctx B", "Ctx C"],
    })
    enricher = _build_enricher_with_mock_client(
        post_side_effect=[_make_openai_response(payload)]
    )

    enriched = asyncio.run(
        enricher.enrich_chunks(document="Full document text.", chunks=chunks)
    )

    assert enricher._client.post.await_count == 1, "Should use a single batched call"
    assert enriched == [
        "Ctx A\n\nFirst chunk text.",
        "Ctx B\n\nSecond chunk text.",
        "Ctx C\n\nThird chunk text.",
    ]


def test_contextual_falls_back_on_length_mismatch():
    """If the batched call returns a wrong-length context list, fall back to
    per-chunk calls — one HTTP call per chunk + the failed batch call."""
    chunks = ["a", "b", "c"]
    bad_batch = json.dumps({"contexts": ["only one"]})
    post_calls = [
        _make_openai_response(bad_batch),
        _make_openai_response("single ctx A"),
        _make_openai_response("single ctx B"),
        _make_openai_response("single ctx C"),
    ]
    enricher = _build_enricher_with_mock_client(post_side_effect=post_calls)

    enriched = asyncio.run(enricher.enrich_chunks(document="doc", chunks=chunks))

    assert enricher._client.post.await_count == 1 + len(chunks)
    assert enriched == [
        "single ctx A\n\na",
        "single ctx B\n\nb",
        "single ctx C\n\nc",
    ]


def test_contextual_falls_back_on_malformed_json():
    """Malformed JSON from the batch call → fall back to per-chunk."""
    chunks = ["a", "b"]
    post_calls = [
        _make_openai_response("not-json at all"),
        _make_openai_response("ctx a"),
        _make_openai_response("ctx b"),
    ]
    enricher = _build_enricher_with_mock_client(post_side_effect=post_calls)

    enriched = asyncio.run(enricher.enrich_chunks(document="doc", chunks=chunks))

    assert enricher._client.post.await_count == 1 + len(chunks)
    assert enriched == ["ctx a\n\na", "ctx b\n\nb"]


def test_contextual_falls_back_on_http_error():
    """Network/HTTP error on the batch call → fall back to per-chunk."""
    chunks = ["a", "b"]
    post_calls = [
        httpx.ConnectError("boom"),
        _make_openai_response("ctx a"),
        _make_openai_response("ctx b"),
    ]
    enricher = _build_enricher_with_mock_client(post_side_effect=post_calls)

    enriched = asyncio.run(enricher.enrich_chunks(document="doc", chunks=chunks))

    assert enricher._client.post.await_count == 1 + len(chunks)
    assert enriched == ["ctx a\n\na", "ctx b\n\nb"]


def test_contextual_empty_chunks_short_circuits():
    """No chunks → no HTTP calls, return input unchanged."""
    enricher = _build_enricher_with_mock_client(post_side_effect=[])

    result = asyncio.run(enricher.enrich_chunks(document="doc", chunks=[]))

    assert result == []
    assert enricher._client.post.await_count == 0
