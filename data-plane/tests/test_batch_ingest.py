"""Tests for POST /api/v1/online/batch/ingest.

The batch endpoint runs each item through the same per-document pipeline as
``/ingest`` and reports per-item outcomes in ``results[]``. The tests below
exercise:

- Happy path: every item ingests successfully.
- Empty items list → ``VALIDATION_BATCH_EMPTY`` (top-level envelope).
- Oversized batch → ``VALIDATION_BATCH_TOO_LARGE`` (top-level envelope).
- Per-item empty content → that single item fails with
  ``VALIDATION_EMPTY_CONTENT``; other items in the same batch still succeed.
- Per-item ``IngestError`` from the underlying service → that item maps to
  the public error code; batch keeps running.
- Concurrency cap: simultaneous in-flight items never exceed
  ``DP_BATCH_INGEST_CONCURRENCY``.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.ingest.ingest_service import IngestError, IngestResult


@pytest.fixture
def mock_ingest():
    svc = MagicMock()
    svc.ingest = AsyncMock()
    return svc


@pytest.fixture
def client(mock_ingest):
    # Mirrors test_ingest_stream.py: stubs every router-touched piece of
    # app.state so TestClient startup doesn't fail, then wires the ingest
    # service mock specifically.
    app.state._test_mode = True
    app.state.online_ingest = mock_ingest
    app.state.funding_extractor = MagicMock()
    app.state.classifier = MagicMock()
    app.state.tei_embedder_at = MagicMock()  # default embedding_model=bge_m3
    app.state.ingest = MagicMock()
    app.state.search = MagicMock()
    app.state.scraping = MagicMock()
    # Ingest router persists usage to ClickHouse via scraping.audit. The
    # production audit logger is async; MagicMock attributes aren't, so
    # patch the one method we call as AsyncMock to keep the await happy.
    app.state.scraping.audit.log_usage = AsyncMock()
    app.state.sitemap_parser = MagicMock()
    app.state.parser = MagicMock()
    app.state.embedder = MagicMock()
    app.state.qdrant = MagicMock()
    app.state.discovery = MagicMock()
    with TestClient(app) as c:
        yield c


def _make_item(source_id: str = "doc_1", **overrides) -> dict:
    """Build a minimal valid OnlineIngestRequest body."""
    base = {
        "collection_name": "test-coll",
        "source_id": source_id,
        "url": f"https://example.test/{source_id}",
        "content": f"Some scraped content for {source_id}.",
        "content_type": ["general"],
        "metadata": {
            "assistant_id": "asst_01",
            "municipality_id": "test-muni",
        },
    }
    base.update(overrides)
    return base


def _success_result(source_id: str, chunks: int = 3) -> IngestResult:
    return IngestResult(
        source_id=source_id,
        chunks_created=chunks,
        vectors_stored=chunks,
        collection="test-coll",
        classification=["general"],
        entities_extracted={"dates": 0, "contacts": 0, "amounts": 0},
        embedding_time_ms=10,
        total_time_ms=20,
    )


def test_batch_ingest_happy_path(client, mock_ingest):
    """Two items, both succeed → top-level success=true with both per-item
    results carrying ingest data."""
    mock_ingest.ingest.side_effect = [
        _success_result("doc_1", chunks=3),
        _success_result("doc_2", chunks=5),
    ]

    response = client.post(
        "/api/v1/online/batch/ingest",
        json={"items": [_make_item("doc_1"), _make_item("doc_2")]},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["success"] is True
    data = body["data"]
    assert data["total"] == 2
    assert data["succeeded"] == 2
    assert data["failed"] == 0
    assert len(data["results"]) == 2

    # Results preserve request order
    assert data["results"][0]["source_id"] == "doc_1"
    assert data["results"][0]["success"] is True
    assert data["results"][0]["data"]["chunks_created"] == 3
    assert data["results"][0]["error"] is None

    assert data["results"][1]["source_id"] == "doc_2"
    assert data["results"][1]["data"]["chunks_created"] == 5

    assert data["total_time_ms"] >= 0
    assert body["request_id"]
    assert mock_ingest.ingest.await_count == 2


def test_batch_ingest_empty_list_rejected(client, mock_ingest):
    """Empty ``items`` → top-level VALIDATION_BATCH_EMPTY, no ingest calls."""
    response = client.post(
        "/api/v1/online/batch/ingest",
        json={"items": []},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "VALIDATION_BATCH_EMPTY"
    assert body["data"] is None
    assert mock_ingest.ingest.await_count == 0


def test_batch_ingest_too_large_rejected(client, mock_ingest):
    """Batch above ``DP_MAX_BATCH_INGEST_ITEMS`` → top-level
    VALIDATION_BATCH_TOO_LARGE. No partial work."""
    oversized = [_make_item(f"doc_{i}") for i in range(settings.max_batch_ingest_items + 1)]

    response = client.post(
        "/api/v1/online/batch/ingest",
        json={"items": oversized},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "VALIDATION_BATCH_TOO_LARGE"
    assert str(settings.max_batch_ingest_items) in body["detail"]
    assert mock_ingest.ingest.await_count == 0


def test_batch_ingest_per_item_empty_content(client, mock_ingest):
    """An item with empty content fails per-item with
    VALIDATION_EMPTY_CONTENT; other items in the same batch still succeed."""
    # Only the valid item should reach the ingest service.
    mock_ingest.ingest.return_value = _success_result("doc_ok")

    items = [
        _make_item("doc_empty", content="   "),  # whitespace-only → empty
        _make_item("doc_ok"),
    ]
    response = client.post(
        "/api/v1/online/batch/ingest",
        json={"items": items},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["success"] is True
    data = body["data"]
    assert data["total"] == 2
    assert data["succeeded"] == 1
    assert data["failed"] == 1

    results_by_id = {r["source_id"]: r for r in data["results"]}
    assert results_by_id["doc_empty"]["success"] is False
    assert results_by_id["doc_empty"]["error"] == "VALIDATION_EMPTY_CONTENT"
    assert results_by_id["doc_empty"]["data"] is None
    assert results_by_id["doc_ok"]["success"] is True
    assert results_by_id["doc_ok"]["data"]["chunks_created"] == 3

    # Only the valid item should have reached IngestService.
    assert mock_ingest.ingest.await_count == 1


def test_batch_ingest_per_item_ingest_error(client, mock_ingest):
    """An item whose IngestService.ingest raises IngestError maps to the
    matching public error code; siblings still complete."""
    mock_ingest.ingest.side_effect = [
        IngestError("Embedding service offline", code="EMBEDDING_FAILED"),
        _success_result("doc_ok"),
    ]

    items = [_make_item("doc_fail"), _make_item("doc_ok")]
    response = client.post(
        "/api/v1/online/batch/ingest",
        json={"items": items},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["success"] is True
    data = body["data"]
    assert data["total"] == 2
    assert data["succeeded"] == 1
    assert data["failed"] == 1

    by_id = {r["source_id"]: r for r in data["results"]}
    assert by_id["doc_fail"]["success"] is False
    assert by_id["doc_fail"]["error"] == "EMBEDDING_FAILED"
    assert "offline" in by_id["doc_fail"]["detail"].lower()
    assert by_id["doc_ok"]["success"] is True


def test_batch_ingest_respects_concurrency_cap(client, mock_ingest, monkeypatch):
    """The semaphore caps simultaneous in-flight items at
    ``DP_BATCH_INGEST_CONCURRENCY``. We instrument IngestService.ingest with
    a counter that asserts the peak never exceeds that limit."""
    monkeypatch.setattr(settings, "batch_ingest_concurrency", 3)

    in_flight = 0
    peak = 0
    lock = asyncio.Lock()

    async def _slow_ingest(**kwargs):
        nonlocal in_flight, peak
        async with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            # Let the event loop schedule sibling tasks so we actually
            # observe concurrency rather than serial execution.
            await asyncio.sleep(0.02)
        finally:
            async with lock:
                in_flight -= 1
        return _success_result(kwargs["source_id"])

    mock_ingest.ingest.side_effect = _slow_ingest

    items = [_make_item(f"doc_{i}") for i in range(10)]
    response = client.post(
        "/api/v1/online/batch/ingest",
        json={"items": items},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["total"] == 10
    assert body["data"]["succeeded"] == 10
    # Peak in-flight must be ≤ the configured concurrency.
    assert peak <= 3, f"peak={peak} exceeded concurrency cap of 3"
    assert peak >= 2, f"peak={peak} suggests effective serial execution"
