"""Tests for POST /api/v1/local/ingest endpoint."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.ingest.ingest_service import IngestError, IngestResult


@pytest.fixture
def mock_ingest():
    svc = MagicMock()
    svc.ingest = AsyncMock()
    return svc


@pytest.fixture
def client(mock_ingest):
    app.state._test_mode = True
    app.state.ingest = mock_ingest
    app.state.search = MagicMock()
    app.state.scraping = MagicMock()
    app.state.sitemap_parser = MagicMock()
    app.state.parser = MagicMock()
    app.state.classifier = MagicMock()
    app.state.embedder = MagicMock()
    app.state.qdrant = MagicMock()
    app.state.discovery = MagicMock()
    with TestClient(app) as c:
        yield c


def _make_request(**overrides):
    base = {
        "collection_name": "wiener-neudorf",
        "source_id": "doc_abc123",
        "file_path": "//server/bauamt/antrag.pdf",
        "content": "Bauantrag Nr. 2024-001. Antragsteller Max Mustermann.",
        "language": "de",
        "acl": {
            "allow_groups": ["DOMAIN\\Bauamt"],
            "deny_groups": [],
            "allow_roles": [],
            "allow_users": [],
            "department": "bauamt",
            "visibility": "internal",
        },
        "metadata": {
            "title": "Bauantrag 2024-001",
            "uploaded_by": "moderator_01",
            "source_type": "smb",
            "mime_type": "application/pdf",
        },
    }
    base.update(overrides)
    return base


def test_ingest_success(client, mock_ingest):
    mock_ingest.ingest.return_value = IngestResult(
        source_id="doc_abc123",
        chunks_created=5,
        vectors_stored=5,
        collection="wiener-neudorf",
        classification=["policy"],
        entities_extracted={"dates": 2, "contacts": 1, "amounts": 0},
        embedding_time_ms=150,
        total_time_ms=300,
    )

    response = client.post("/api/v1/local/ingest", json=_make_request())
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["data"]["source_id"] == "doc_abc123"
    assert data["data"]["chunks_created"] == 5
    assert data["data"]["vectors_stored"] == 5
    assert data["data"]["collection"] == "wiener-neudorf"
    assert data["data"]["content_type"] == ["policy"]
    assert data["data"]["entities_extracted"]["dates"] == 2
    assert data["data"]["embedding_time_ms"] == 150
    assert data["data"]["total_time_ms"] == 300
    assert data["request_id"]


def test_ingest_with_chunking_config(client, mock_ingest):
    mock_ingest.ingest.return_value = IngestResult(
        source_id="doc_xyz",
        chunks_created=10,
        vectors_stored=10,
        collection="test",
        classification=["general"],
        entities_extracted={"dates": 0, "contacts": 0, "amounts": 0},
        embedding_time_ms=200,
        total_time_ms=400,
    )

    response = client.post("/api/v1/local/ingest", json=_make_request(
        source_id="doc_xyz",
        chunking={"strategy": "sentence", "max_chunk_size": 256, "overlap": 25},
    ))
    data = response.json()
    assert data["success"] is True

    # Verify chunking args were passed
    call_kwargs = mock_ingest.ingest.call_args.kwargs
    assert call_kwargs["collection_name"] == "wiener-neudorf"
    assert call_kwargs["chunking_strategy"] == "sentence"
    assert call_kwargs["max_chunk_size"] == 256
    assert call_kwargs["chunk_overlap"] == 25


def test_ingest_empty_content(client):
    response = client.post("/api/v1/local/ingest", json=_make_request(content="   "))
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "VALIDATION_EMPTY_CONTENT"


def test_ingest_empty_content_pydantic(client):
    """min_length=1 in LocalIngestRequest should reject empty string."""
    response = client.post("/api/v1/local/ingest", json=_make_request(content=""))
    assert response.status_code == 422


def test_ingest_embedding_failed(client, mock_ingest):
    mock_ingest.ingest.side_effect = IngestError("BGE-M3 connection error", code="EMBEDDING_FAILED")

    response = client.post("/api/v1/local/ingest", json=_make_request())
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "EMBEDDING_FAILED"


def test_ingest_embedding_oom(client, mock_ingest):
    mock_ingest.ingest.side_effect = IngestError("Out of memory", code="EMBEDDING_OOM")

    response = client.post("/api/v1/local/ingest", json=_make_request())
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "EMBEDDING_OOM"


def test_ingest_qdrant_upsert_failed(client, mock_ingest):
    mock_ingest.ingest.side_effect = IngestError("Upsert failed", code="QDRANT_UPSERT_FAILED")

    response = client.post("/api/v1/local/ingest", json=_make_request())
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "QDRANT_UPSERT_FAILED"


def test_ingest_qdrant_collection_not_found(client, mock_ingest):
    mock_ingest.ingest.side_effect = IngestError("Collection missing", code="QDRANT_COLLECTION_NOT_FOUND")

    response = client.post("/api/v1/local/ingest", json=_make_request())
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "QDRANT_COLLECTION_NOT_FOUND"


def test_ingest_qdrant_disk_full(client, mock_ingest):
    mock_ingest.ingest.side_effect = IngestError("Disk full", code="QDRANT_DISK_FULL")

    response = client.post("/api/v1/local/ingest", json=_make_request())
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "QDRANT_DISK_FULL"


def test_ingest_invalid_visibility(client):
    """Pydantic should reject invalid ACL visibility."""
    response = client.post("/api/v1/local/ingest", json=_make_request(
        acl={"allow_groups": [], "deny_groups": [], "visibility": "secret"},
    ))
    assert response.status_code == 422


def test_ingest_request_id(client, mock_ingest):
    mock_ingest.ingest.return_value = IngestResult(
        source_id="doc_test",
        chunks_created=1,
        vectors_stored=1,
        collection="test",
        classification=["general"],
        entities_extracted={"dates": 0, "contacts": 0, "amounts": 0},
        embedding_time_ms=10,
        total_time_ms=20,
    )

    response = client.post(
        "/api/v1/local/ingest",
        json=_make_request(source_id="doc_test"),
        headers={"X-Request-ID": "ingest-req-999"},
    )
    data = response.json()
    assert data["request_id"] == "ingest-req-999"
    assert response.headers["X-Request-ID"] == "ingest-req-999"
