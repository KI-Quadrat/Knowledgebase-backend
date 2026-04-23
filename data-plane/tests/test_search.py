"""Tests for POST /api/v1/search endpoint."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.search.search_service import (
    PermissionFilter,
    SearchError,
    SearchResult,
    SearchResultItem,
)


@pytest.fixture
def mock_search():
    svc = MagicMock()
    svc.search = AsyncMock()
    return svc


@pytest.fixture
def client(mock_search):
    app.state._test_mode = True
    app.state.search = mock_search
    app.state.ingest = MagicMock()
    app.state.scraping = MagicMock()
    app.state.sitemap_parser = MagicMock()
    app.state.parser = MagicMock()
    app.state.classifier = MagicMock()
    app.state.embedder = MagicMock()
    app.state.qdrant = MagicMock()
    app.state.discovery = MagicMock()
    with TestClient(app) as c:
        yield c


def _employee_request(**overrides):
    base = {
        "collection_name": "wiener-neudorf",
        "query": "Wann ist die nächste Förderung für Solaranlagen?",
        "user": {
            "type": "employee",
            "user_id": "maria@wiener-neudorf.gv.at",
            "groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Alle-Mitarbeiter"],
            "roles": ["member"],
            "department": "bauamt",
        },
        "top_k": 10,
        "score_threshold": 0.5,
    }
    base.update(overrides)
    return base


def _citizen_request(**overrides):
    base = {
        "collection_name": "wiener-neudorf",
        "query": "Öffnungszeiten Gemeindeamt",
        "user": {"type": "citizen", "user_id": "anonymous"},
        "top_k": 5,
        "score_threshold": 0.5,
    }
    base.update(overrides)
    return base


def _make_search_result(items=None, perm_filter=None):
    items = items or []
    perm_filter = perm_filter or PermissionFilter(
        visibility=["public", "internal"],
        must_match_groups=["DOMAIN\\Bauamt-Mitarbeiter"],
        must_not_match_groups=[],
    )
    return SearchResult(
        results=items,
        total_results=len(items),
        query_embedding_ms=15,
        search_ms=22,
        permission_filter=perm_filter,
    )


def test_search_employee_success(client, mock_search):
    items = [
        SearchResultItem(
            chunk_id="doc_abc123_chunk_07",
            source_id="doc_abc123",
            chunk_text="Die Förderung für Solaranlagen beträgt bis zu EUR 5.000...",
            score=0.92,
            source_path="//server/bauamt/foerderungen/solar_2025.pdf",
            classification="funding",
            entity_amounts=["EUR 5.000"],
            entity_deadlines=["2025-06-30"],
            title="Solarförderung 2025",
            municipality_id="wiener-neudorf",
            department="bauamt",
            source_type="smb",
        ),
    ]
    mock_search.search.return_value = _make_search_result(items)

    response = client.post("/api/v1/search", json=_employee_request())
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["data"]["total_results"] == 1
    assert data["data"]["query_embedding_ms"] == 15
    assert data["data"]["search_ms"] == 22

    result = data["data"]["results"][0]
    assert result["chunk_id"] == "doc_abc123_chunk_07"
    assert result["score"] == 0.92
    assert result["classification"] == "funding"
    assert "EUR 5.000" in result["entities"]["amounts"]
    assert result["metadata"]["title"] == "Solarförderung 2025"

    pf = data["data"]["permission_filter_applied"]
    assert "public" in pf["visibility"]
    assert "internal" in pf["visibility"]
    assert "DOMAIN\\Bauamt-Mitarbeiter" in pf["must_match_groups"]
    assert data["request_id"]


def test_search_citizen_success(client, mock_search):
    mock_search.search.return_value = _make_search_result(
        items=[
            SearchResultItem(
                chunk_id="doc_pub_chunk_01",
                source_id="doc_pub",
                chunk_text="Öffnungszeiten: Mo-Fr 8-16 Uhr",
                score=0.85,
                source_path="//server/public/kontakt.txt",
                classification="contact",
                entity_amounts=[],
                entity_deadlines=[],
                title="Kontaktseite",
                municipality_id=None,
                department=None,
                source_type="smb",
            ),
        ],
        perm_filter=PermissionFilter(
            visibility=["public"],
            must_match_groups=[],
            must_not_match_groups=[],
        ),
    )

    response = client.post("/api/v1/search", json=_citizen_request())
    data = response.json()
    assert data["success"] is True
    assert data["data"]["permission_filter_applied"]["visibility"] == ["public"]
    assert data["data"]["permission_filter_applied"]["must_match_groups"] == []


def test_search_with_classification_filter(client, mock_search):
    mock_search.search.return_value = _make_search_result()

    response = client.post("/api/v1/search", json=_employee_request(
        filters={"classification": ["funding", "policy"]},
    ))
    data = response.json()
    assert data["success"] is True

    # Verify filter was passed
    call_kwargs = mock_search.search.call_args.kwargs
    assert call_kwargs["collection_name"] == "wiener-neudorf"
    assert call_kwargs["classification_filter"] == ["funding", "policy"]


def test_search_no_results(client, mock_search):
    mock_search.search.return_value = _make_search_result()

    response = client.post("/api/v1/search", json=_employee_request(
        query="something that matches nothing",
    ))
    data = response.json()
    assert data["success"] is True
    assert data["data"]["total_results"] == 0
    assert data["data"]["results"] == []


def test_search_embedding_failed(client, mock_search):
    mock_search.search.side_effect = SearchError("BGE-M3 error", code="EMBEDDING_FAILED")

    response = client.post("/api/v1/search", json=_employee_request())
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "EMBEDDING_FAILED"


def test_search_qdrant_connection_failed(client, mock_search):
    mock_search.search.side_effect = SearchError("Qdrant down", code="QDRANT_CONNECTION_FAILED")

    response = client.post("/api/v1/search", json=_employee_request())
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "QDRANT_CONNECTION_FAILED"


def test_search_collection_not_found(client, mock_search):
    mock_search.search.side_effect = SearchError("No collection", code="QDRANT_COLLECTION_NOT_FOUND")

    response = client.post("/api/v1/search", json=_employee_request())
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "QDRANT_COLLECTION_NOT_FOUND"


def test_search_invalid_user_type(client):
    """Pydantic should reject invalid user type."""
    response = client.post("/api/v1/search", json={
        "collection_name": "test",
        "query": "test",
        "user": {"type": "admin", "user_id": "test"},
    })
    assert response.status_code == 422


def test_search_empty_query(client):
    """Pydantic should reject empty query (min_length=1)."""
    response = client.post("/api/v1/search", json={
        "collection_name": "test",
        "query": "",
        "user": {"type": "citizen", "user_id": "anon"},
    })
    assert response.status_code == 422


def test_search_request_id(client, mock_search):
    mock_search.search.return_value = _make_search_result()

    response = client.post(
        "/api/v1/search",
        json=_citizen_request(),
        headers={"X-Request-ID": "search-req-777"},
    )
    data = response.json()
    assert data["request_id"] == "search-req-777"
    assert response.headers["X-Request-ID"] == "search-req-777"
