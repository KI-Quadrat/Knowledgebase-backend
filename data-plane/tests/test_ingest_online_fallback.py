"""End-to-end test for POST /api/v1/online/ingest proving the BGE-Gemma2
fallback vector is actually generated and stored in Qdrant when
``vector_config.enable_fallback`` is true.

Drives the real router + real ``IngestService`` with mocked embedder /
fallback-embedder / qdrant so we can inspect the exact upsert payload."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import ext
from app.main import app
from app.services.embedding.bge_m3_client import EmbeddingError, EmbeddingResult
from app.services.ingest.ingest_service import IngestService
from app.services.intelligence.chunker import Chunker


USER_PAYLOAD = {
    "collection_name": "wiener-neudorf_test_fallback",
    "content": "Förderungen der Gemeinde Wiener Neudorf\n\nDie Gemeinde bietet verschiedene Förderungen...",
    "content_type": ["funding", "renewable_energy"],
    "language": "de",
    "metadata": {
        "assistant_id": "asst_wiener_neudorf_01",
        "department": ["Bürgerservice", "Förderungen"],
        "municipality_id": "wiener-neudorf",
        "source_type": "web",
        "title": "Förderungen - Gemeinde Wiener Neudorf",
    },
    "source_id": "web_foerderungen_001",
    "url": "https://www.wiener-neudorf.gv.at/foerderungen",
    "vector_config": {
        "enable_fallback": True,
        "search_mode": "semantic",
        "vector_size": 1536,
    },
}


def _dummy(dim: int):
    async def _embed_batch(chunks):
        return [EmbeddingResult(dense=[0.01 * (i + 1)] * dim) for i, _ in enumerate(chunks)]
    return _embed_batch


def _build_service(primary_embed, fallback_embed):
    primary = MagicMock()
    primary.embed_batch = AsyncMock(side_effect=primary_embed)
    fallback = MagicMock()
    fallback.embed_batch = AsyncMock(side_effect=fallback_embed)

    qdrant = MagicMock()
    qdrant.create_collection = AsyncMock()
    qdrant.delete_by_source_id = AsyncMock(return_value=0)
    qdrant.upsert_points = AsyncMock(side_effect=lambda _c, points: len(points))

    svc = IngestService(
        chunker=Chunker(),
        classifier=MagicMock(),
        embedder=primary,
        qdrant=qdrant,
        contextual_enricher=None,
        fallback_embedder=fallback,
    )
    return svc, qdrant


@pytest.fixture
def client():
    app.state._test_mode = True
    yield  # set per-test below


def _install(service):
    app.state._test_mode = True
    app.state.online_ingest = service
    app.state.funding_extractor = MagicMock()


def test_user_payload_stores_both_vectors_when_both_embedders_succeed():
    """Happy path for the user's payload: ``enable_fallback: true`` → every
    Qdrant point carries both ``dense_openai`` (1536) and ``dense_bge_gemma2``
    (server-configured fallback dim)."""
    svc, qdrant = _build_service(
        primary_embed=_dummy(1536),
        fallback_embed=_dummy(ext.bge_gemma2_dense_dim),
    )
    _install(svc)

    with TestClient(app) as c:
        r = c.post("/api/v1/online/ingest", json=USER_PAYLOAD)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True, body
    assert body["data"]["vectors_stored"] > 0
    assert body["data"]["collection"] == "wiener-neudorf_test_fallback"
    assert body["data"]["content_type"] == ["funding", "renewable_energy"]

    # Inspect what actually went into Qdrant
    points = qdrant.upsert_points.await_args.args[1]
    assert len(points) > 0
    for p in points:
        vectors = p["vector"]
        assert "dense_openai" in vectors
        assert len(vectors["dense_openai"]) == 1536
        assert "dense_bge_gemma2" in vectors, "Fallback vector must be stored when enable_fallback=true"
        assert len(vectors["dense_bge_gemma2"]) == ext.bge_gemma2_dense_dim
        # semantic mode → no sparse vector
        assert "sparse" not in vectors


def test_user_payload_fallback_generates_vector_when_primary_fails():
    """Exact scenario the user asked about: primary OpenAI embedder fails,
    the BGE-Gemma2 fallback must still produce the vector stored in Qdrant."""
    async def _primary_fail(_chunks):
        raise EmbeddingError("OpenAI outage")

    svc, qdrant = _build_service(
        primary_embed=_primary_fail,
        fallback_embed=_dummy(ext.bge_gemma2_dense_dim),
    )
    _install(svc)

    with TestClient(app) as c:
        r = c.post("/api/v1/online/ingest", json=USER_PAYLOAD)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True, body
    assert body["data"]["vectors_stored"] > 0

    points = qdrant.upsert_points.await_args.args[1]
    assert len(points) > 0
    for p in points:
        vectors = p["vector"]
        assert "dense_bge_gemma2" in vectors and len(vectors["dense_bge_gemma2"]) == ext.bge_gemma2_dense_dim
        assert "dense_openai" not in vectors, "Primary vector must be absent when primary embedder failed"


def test_user_payload_primary_only_when_fallback_fails():
    """Symmetric: if the fallback embedder fails, the point is still stored —
    now with only ``dense_openai``."""
    async def _fallback_fail(_chunks):
        raise EmbeddingError("LiteLLM timeout")

    svc, qdrant = _build_service(
        primary_embed=_dummy(1536),
        fallback_embed=_fallback_fail,
    )
    _install(svc)

    with TestClient(app) as c:
        r = c.post("/api/v1/online/ingest", json=USER_PAYLOAD)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True

    points = qdrant.upsert_points.await_args.args[1]
    for p in points:
        vectors = p["vector"]
        assert "dense_openai" in vectors and len(vectors["dense_openai"]) == 1536
        assert "dense_bge_gemma2" not in vectors
