"""Tests for GET /api/v1/health and GET /api/v1/ready."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.shared import health as health_router


@pytest.fixture
def client():
    app.state._test_mode = True
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_online_api_keys():
    previous = health_router.settings.online_api_keys
    yield
    health_router.settings.online_api_keys = previous


def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data


def test_ready_minimal(client):
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data.get("services") is None
    assert "uptime_seconds" in data


def test_ready_full_all_healthy(client):
    """Authenticated ready check with all services healthy."""
    # Set up mock services
    mock_scraping = MagicMock()
    mock_scraping.is_ready = True

    mock_qdrant = MagicMock()
    mock_qdrant.check_health = AsyncMock(return_value=True)

    mock_embedder = MagicMock()
    mock_embedder.check_health = AsyncMock(return_value=True)

    mock_parser = MagicMock()
    mock_parser.check_health = AsyncMock(return_value=True)

    mock_cache = MagicMock()
    mock_cache.ping = AsyncMock(return_value=True)

    app.state.scraping = mock_scraping
    app.state.qdrant = mock_qdrant
    app.state.embedder = mock_embedder
    app.state.parser = mock_parser
    app.state.cache = mock_cache

    response = client.get(
        "/api/v1/ready",
        headers={"X-Signature": "dummy-for-full-check"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
    assert data["services"]["crawl4ai"] is True
    assert data["services"]["qdrant"] is True
    assert data["services"]["bge_m3"] is True
    assert data["services"]["parser"] is True
    assert data["services"]["redis"] is True
    assert data["version"]
    assert data["mode"]


def test_ready_full_service_down(client):
    """Authenticated ready check with one core service down."""
    mock_scraping = MagicMock()
    mock_scraping.is_ready = True

    mock_qdrant = MagicMock()
    mock_qdrant.check_health = AsyncMock(return_value=False)  # Qdrant down

    mock_embedder = MagicMock()
    mock_embedder.check_health = AsyncMock(return_value=True)

    mock_parser = MagicMock()
    mock_parser.check_health = AsyncMock(return_value=True)

    app.state.scraping = mock_scraping
    app.state.qdrant = mock_qdrant
    app.state.embedder = mock_embedder
    app.state.parser = mock_parser

    response = client.get(
        "/api/v1/ready",
        headers={"X-Signature": "dummy"},
    )
    data = response.json()
    assert data["ready"] is False
    assert data["services"]["qdrant"] is False
    assert data["services"]["bge_m3"] is True


def test_ready_full_service_exception(client):
    """Health check that throws should result in service=false, not 500."""
    mock_qdrant = MagicMock()
    mock_qdrant.check_health = AsyncMock(side_effect=Exception("connection refused"))

    app.state.scraping = MagicMock(is_ready=True)
    app.state.qdrant = mock_qdrant
    app.state.embedder = MagicMock(check_health=AsyncMock(return_value=True))
    app.state.parser = MagicMock(check_health=AsyncMock(return_value=True))

    response = client.get(
        "/api/v1/ready",
        headers={"X-Signature": "dummy"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["services"]["qdrant"] is False


def test_request_id_header(client):
    response = client.get("/api/v1/health")
    assert "X-Request-ID" in response.headers


def test_request_id_echo(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "test-123"})
    assert response.headers["X-Request-ID"] == "test-123"


def test_model_health_requires_api_key(client):
    health_router.settings.online_api_keys = "secret-key"

    response = client.get("/api/v1/model-health")

    assert response.status_code == 401
    assert response.json()["detail"] == "X-API-Key header is required"


def test_model_health_uses_shared_env_api_key_behavior(client, monkeypatch):
    health_router.settings.online_api_keys = ""
    monkeypatch.setattr(
        health_router,
        "_probe_openai_chat_model",
        AsyncMock(return_value=(True, None)),
    )
    monkeypatch.setattr(
        health_router,
        "_probe_jina_reader",
        AsyncMock(return_value=(True, None)),
    )

    app.state.scraping = MagicMock(
        crawl4ai=MagicMock(
            _jina_key="jina-key",
            check_health=AsyncMock(return_value=True),
        )
    )
    app.state.embedder = MagicMock(embed=AsyncMock(return_value=object()))
    app.state.openai_embedder = MagicMock(
        _api_key="openai-key",
        _model="text-embedding-3-small",
        embed=AsyncMock(return_value=object()),
    )
    app.state.bge_gemma2_embedder = MagicMock(
        _model="bge-multilingual-gemma2",
        embed=AsyncMock(return_value=object()),
    )
    app.state.classifier = MagicMock(
        _llm=MagicMock(_client=object(), _model="gpt-4o-mini")
    )
    app.state.contextual_enricher = MagicMock(
        _api_key="openai-key",
        _model="gpt-4o-mini",
    )
    app.state.funding_extractor = MagicMock(
        _client=object(),
        _model="gpt-4o-mini",
    )
    app.state.parser = MagicMock(
        parser_backend="local",
    )

    response = client.get("/api/v1/model-health")

    assert response.status_code == 200


def test_model_health_returns_model_statuses(client, monkeypatch):
    health_router.settings.online_api_keys = "secret-key"
    monkeypatch.setattr(
        health_router,
        "_probe_openai_chat_model",
        AsyncMock(return_value=(True, None)),
    )
    monkeypatch.setattr(
        health_router,
        "_probe_jina_reader",
        AsyncMock(return_value=(True, None)),
    )

    app.state.scraping = MagicMock(
        crawl4ai=MagicMock(
            _jina_key="jina-key",
            check_health=AsyncMock(return_value=True),
        )
    )
    app.state.embedder = MagicMock(embed=AsyncMock(return_value=object()))
    app.state.openai_embedder = MagicMock(
        _api_key="openai-key",
        _model="text-embedding-3-small",
        embed=AsyncMock(return_value=object()),
    )
    app.state.bge_gemma2_embedder = MagicMock(
        _model="bge-multilingual-gemma2",
        embed=AsyncMock(return_value=object()),
    )
    app.state.classifier = MagicMock(
        _llm=MagicMock(_client=object(), _model="gpt-4o-mini")
    )
    app.state.contextual_enricher = MagicMock(
        _api_key="openai-key",
        _model="gpt-4o-mini",
    )
    app.state.funding_extractor = MagicMock(
        _client=object(),
        _model="gpt-4o-mini",
    )
    app.state.parser = MagicMock(
        parser_backend="llamaparse",
        check_health=AsyncMock(return_value=True),
    )

    response = client.get(
        "/api/v1/model-health",
        headers={"X-API-Key": "secret-key"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["healthy"] is True
    assert len(data["models"]) == 9
    assert {item["component"] for item in data["models"]} == {
        "scraper_crawl4ai",
        "scraper_jina_reader",
        "local_embedding",
        "online_embedding_primary",
        "online_embedding_fallback",
        "content_classifier",
        "contextual_enricher",
        "funding_extractor",
        "document_parser",
    }


def test_model_health_reports_unhealthy_components(client, monkeypatch):
    health_router.settings.online_api_keys = "secret-key"
    monkeypatch.setattr(
        health_router,
        "_probe_openai_chat_model",
        AsyncMock(return_value=(False, "OpenAI HTTP 500")),
    )
    monkeypatch.setattr(
        health_router,
        "_probe_jina_reader",
        AsyncMock(return_value=(False, "Jina HTTP 502")),
    )

    app.state.scraping = MagicMock(
        crawl4ai=MagicMock(
            _jina_key="jina-key",
            check_health=AsyncMock(return_value=False),
        )
    )
    app.state.embedder = MagicMock(embed=AsyncMock(return_value=object()))
    app.state.openai_embedder = MagicMock(
        _api_key="openai-key",
        _model="text-embedding-3-small",
        embed=AsyncMock(side_effect=RuntimeError("embedding failed")),
    )
    app.state.bge_gemma2_embedder = MagicMock(
        _model="bge-multilingual-gemma2",
        embed=AsyncMock(return_value=object()),
    )
    app.state.classifier = MagicMock(
        _llm=MagicMock(_client=object(), _model="gpt-4o-mini")
    )
    app.state.contextual_enricher = MagicMock(
        _api_key="openai-key",
        _model="gpt-4o-mini",
    )
    app.state.funding_extractor = MagicMock(
        _client=None,
        _model="gpt-4o-mini",
    )
    app.state.parser = MagicMock(
        parser_backend="local",
    )

    response = client.get(
        "/api/v1/model-health",
        headers={"X-API-Key": "secret-key"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["healthy"] is False

    by_component = {item["component"]: item for item in data["models"]}
    assert by_component["scraper_crawl4ai"]["healthy"] is False
    assert by_component["scraper_jina_reader"]["healthy"] is False
    assert by_component["online_embedding_primary"]["healthy"] is False
    assert "embedding failed" in by_component["online_embedding_primary"]["detail"]
    assert by_component["content_classifier"]["healthy"] is False
    assert by_component["contextual_enricher"]["healthy"] is False
    assert by_component["funding_extractor"]["configured"] is False
    assert by_component["document_parser"]["configured"] is False
