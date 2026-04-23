"""Tests for POST /api/v1/online/ingest/stream (Server-Sent Events)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ingest.ingest_service import IngestError


def _parse_sse(body: str) -> list[dict]:
    """Parse an SSE response body into a list of {event, data} dicts.

    Heartbeat comments (lines starting with ':') are skipped.
    """
    frames: list[dict] = []
    for raw in body.split("\n\n"):
        raw = raw.strip("\n")
        if not raw or raw.startswith(":"):
            continue
        frame: dict = {}
        for line in raw.split("\n"):
            if line.startswith(":"):
                continue
            key, _, value = line.partition(":")
            frame[key.strip()] = value.strip()
        frames.append(frame)
    return frames


def _build_ingest_side_effect(events: list[dict], final_result=None, raise_error=None):
    """Return an AsyncMock side_effect that drains ``events`` into the progress_queue
    then either returns ``final_result`` or raises ``raise_error``."""

    async def _side_effect(**kwargs):
        queue = kwargs.get("progress_queue")
        if queue is not None:
            for ev in events:
                await queue.put(ev)
        if raise_error is not None:
            raise raise_error
        return final_result

    return _side_effect


@pytest.fixture
def mock_ingest():
    svc = MagicMock()
    svc.ingest = AsyncMock()
    return svc


@pytest.fixture
def client(mock_ingest):
    app.state._test_mode = True
    app.state.online_ingest = mock_ingest
    app.state.funding_extractor = MagicMock()
    app.state.classifier = MagicMock()
    # Stubs for unrelated routers the TestClient startup may touch.
    app.state.ingest = MagicMock()
    app.state.search = MagicMock()
    app.state.scraping = MagicMock()
    app.state.sitemap_parser = MagicMock()
    app.state.parser = MagicMock()
    app.state.embedder = MagicMock()
    app.state.qdrant = MagicMock()
    app.state.discovery = MagicMock()
    with TestClient(app) as c:
        yield c


def _make_request(**overrides):
    base = {
        "collection_name": "test-coll",
        "source_id": "doc_stream_1",
        "url": "https://example.test/page",
        "content": "Some scraped page content.",
        "content_type": ["funding"],
        "metadata": {
            "assistant_id": "asst_01",
            "municipality_id": "test-muni",
        },
    }
    base.update(overrides)
    return base


def test_stream_emits_progress_then_completed(client, mock_ingest):
    """Happy path: each queued phase becomes a progress event, then a single
    completed event with the final result."""
    from app.services.ingest.ingest_service import IngestResult

    progress_events = [
        {"phase": "started", "source_id": "doc_stream_1"},
        {"phase": "chunked", "chunks": 3},
        {"phase": "embedded", "chunks": 3, "has_openai": True, "has_bge_gemma2": False, "duration_ms": 42},
        {"phase": "stored", "vectors": 3, "collection": "test-coll"},
    ]
    result = IngestResult(
        source_id="doc_stream_1",
        chunks_created=3,
        vectors_stored=3,
        collection="test-coll",
        classification=["funding"],
        entities_extracted={"dates": 0, "contacts": 0, "amounts": 0},
        embedding_time_ms=42,
        total_time_ms=100,
    )
    mock_ingest.ingest.side_effect = _build_ingest_side_effect(progress_events, final_result=result)

    response = client.post("/api/v1/online/ingest/stream", json=_make_request())
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers.get("cache-control") == "no-cache"

    frames = _parse_sse(response.text)

    progress = [f for f in frames if f["event"] == "progress"]
    assert len(progress) == len(progress_events)

    # Phases preserved in order
    import json as _json
    phases = [_json.loads(f["data"])["phase"] for f in progress]
    assert phases == ["started", "chunked", "embedded", "stored"]

    completed = [f for f in frames if f["event"] == "completed"]
    assert len(completed) == 1
    done = _json.loads(completed[0]["data"])
    assert done["source_id"] == "doc_stream_1"
    assert done["vectors_stored"] == 3
    assert done["content_type"] == ["funding"]

    errors = [f for f in frames if f["event"] == "error"]
    assert errors == []


def test_stream_emits_error_event_on_ingest_failure(client, mock_ingest):
    """When IngestService raises IngestError, the stream emits one error event (no completed)."""
    progress_events = [{"phase": "started", "source_id": "doc_stream_1"}]
    mock_ingest.ingest.side_effect = _build_ingest_side_effect(
        progress_events,
        raise_error=IngestError("Qdrant disk full", code="QDRANT_DISK_FULL"),
    )

    response = client.post("/api/v1/online/ingest/stream", json=_make_request())
    assert response.status_code == 200

    frames = _parse_sse(response.text)
    errors = [f for f in frames if f["event"] == "error"]
    completed = [f for f in frames if f["event"] == "completed"]

    assert len(errors) == 1
    assert len(completed) == 0

    import json as _json
    payload = _json.loads(errors[0]["data"])
    assert payload["code"] == "QDRANT_DISK_FULL"
    assert "disk full" in payload["detail"].lower()


def test_stream_empty_content_fails_fast(client, mock_ingest):
    """Empty content returns an error event without running the pipeline."""
    response = client.post("/api/v1/online/ingest/stream", json=_make_request(content="   "))
    assert response.status_code == 200

    frames = _parse_sse(response.text)
    errors = [f for f in frames if f["event"] == "error"]
    assert len(errors) == 1

    import json as _json
    payload = _json.loads(errors[0]["data"])
    assert payload["code"] == "VALIDATION_EMPTY_CONTENT"

    # Ingest service must not have been called.
    mock_ingest.ingest.assert_not_called()


def test_stream_rejects_missing_content_type(client):
    """content_type is required by the shared OnlineIngestRequest schema."""
    payload = _make_request()
    payload.pop("content_type")
    response = client.post("/api/v1/online/ingest/stream", json=payload)
    assert response.status_code == 422
