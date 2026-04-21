"""Tests for ``DELETE /api/v1/online/vectors/at/{source_id}``.

Mirrors the default vector-delete endpoint with one key difference: the
AT endpoint targets ``app.state.qdrant_at`` instead of ``app.state.qdrant``.
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.main import app
from app.services.embedding.qdrant_service import QdrantError


def _install_qdrant_at(*, delete_return: int = 3):
    """Wire a MagicMock qdrant_at onto app.state."""
    app.state._test_mode = True
    qdrant = MagicMock()
    qdrant.delete_by_source_id = AsyncMock(return_value=delete_return)
    app.state.qdrant_at = qdrant
    return qdrant


def test_delete_success_hits_qdrant_at():
    qdrant = _install_qdrant_at(delete_return=7)
    client = TestClient(app)

    resp = client.delete(
        "/api/v1/online/vectors/at/web_foerderungen_001",
        params={"collection_name": "Wien"},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert body["success"] is True
    assert body["data"]["source_id"] == "web_foerderungen_001"
    assert body["data"]["vectors_deleted"] == 7

    qdrant.delete_by_source_id.assert_awaited_once_with("Wien", "web_foerderungen_001")


def test_delete_uses_qdrant_at_not_default_qdrant():
    """Regression: the AT delete must target app.state.qdrant_at, never the
    default app.state.qdrant — the two point at different Qdrant instances."""
    qdrant_at = _install_qdrant_at(delete_return=1)

    default_qdrant = MagicMock()
    default_qdrant.delete_by_source_id = AsyncMock(return_value=99)
    app.state.qdrant = default_qdrant

    client = TestClient(app)
    client.delete(
        "/api/v1/online/vectors/at/src1",
        params={"collection_name": "Tirol"},
    )

    assert qdrant_at.delete_by_source_id.await_count == 1
    assert default_qdrant.delete_by_source_id.await_count == 0


def test_missing_collection_name_rejected():
    _install_qdrant_at()
    client = TestClient(app)

    resp = client.delete("/api/v1/online/vectors/at/src1")
    # FastAPI Query(..., required) → 422 unprocessable entity.
    assert resp.status_code == 422


def test_qdrant_connection_error_maps_to_connection_code():
    qdrant = _install_qdrant_at()
    qdrant.delete_by_source_id = AsyncMock(
        side_effect=QdrantError("Qdrant connection failed")
    )
    client = TestClient(app)

    resp = client.delete(
        "/api/v1/online/vectors/at/src1",
        params={"collection_name": "Wien"},
    )
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "QDRANT_CONNECTION_FAILED"


def test_qdrant_generic_error_maps_to_delete_failed():
    qdrant = _install_qdrant_at()
    qdrant.delete_by_source_id = AsyncMock(side_effect=QdrantError("index not ready"))
    client = TestClient(app)

    resp = client.delete(
        "/api/v1/online/vectors/at/src1",
        params={"collection_name": "Wien"},
    )
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "QDRANT_DELETE_FAILED"
