"""Tests for ``POST /api/v1/online/ingest/at`` and its pure helpers.

Covers the AT funding-assistant ingest: pure helpers for province
normalization and point-ID derivation, plus end-to-end endpoint cases driven
through the real router + real Chunker with mocked TEI embedder / extractor /
Qdrant so we can inspect the exact upsert payload.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers.online.ingest_at import (
    _AT_DEFAULT_CHUNK_OVERLAP,
    _AT_DEFAULT_CHUNK_SIZE,
    _normalize_provinces,
    _point_id,
)
from app.services.embedding.bge_m3_client import EmbeddingResult
from app.services.intelligence.chunker import Chunker


# ─────────────────────────────────────────────────────────────────────
# Pure helpers — no FastAPI harness needed
# ─────────────────────────────────────────────────────────────────────


class TestNormalizeProvinces:
    def test_dedupes_and_lowercases(self):
        assert _normalize_provinces(["Lower Austria", "lower austria"]) == ["lower austria"]

    def test_preserves_order(self):
        assert _normalize_provinces(["vienna", "tyrol", "salzburg"]) == ["vienna", "tyrol", "salzburg"]

    def test_trims_whitespace(self):
        assert _normalize_provinces(["  vienna  ", "salzburg"]) == ["vienna", "salzburg"]

    def test_drops_empty_entries(self):
        assert _normalize_provinces(["", "  ", "vienna"]) == ["vienna"]

    def test_empty_inputs(self):
        assert _normalize_provinces([]) == []
        assert _normalize_provinces(None) == []

    def test_translates_german_to_english(self):
        assert _normalize_provinces(["Niederösterreich"]) == ["lower austria"]
        assert _normalize_provinces(["Wien", "Kärnten", "Tirol"]) == ["vienna", "carinthia", "tyrol"]
        assert _normalize_provinces(["Steiermark", "Oberösterreich"]) == ["styria", "upper austria"]

    def test_dedupes_across_languages(self):
        assert _normalize_provinces(["Niederösterreich", "lower austria"]) == ["lower austria"]
        assert _normalize_provinces(["wien", "Vienna", "WIEN"]) == ["vienna"]

    def test_drops_unknown_values(self):
        assert _normalize_provinces(["atlantis", "transnistria", "vienna"]) == ["vienna"]
        assert _normalize_provinces(["bavaria"]) == []  # German state, not Austrian

    def test_german_umlauts_case_insensitive(self):
        assert _normalize_provinces(["KÄRNTEN"]) == ["carinthia"]
        assert _normalize_provinces(["NiederÖsterreich"]) == ["lower austria"]


class TestComposeBaseUrl:
    """Ensures QDRANT_URL_AT + QDRANT_PORT_AT compose correctly, matching the
    upstream qdrant-client pattern (URL and port supplied separately)."""

    def test_appends_port_to_hostname_url(self):
        from app.services.embedding.qdrant_service import _compose_base_url
        assert _compose_base_url("https://at-qdrant.example.com", 443) == "https://at-qdrant.example.com:443"

    def test_http_localhost_with_port(self):
        from app.services.embedding.qdrant_service import _compose_base_url
        assert _compose_base_url("http://localhost", 6333) == "http://localhost:6333"

    def test_explicit_port_in_url_wins(self):
        from app.services.embedding.qdrant_service import _compose_base_url
        assert _compose_base_url("http://qdrant:6333", 443) == "http://qdrant:6333"

    def test_none_port_leaves_url_unchanged(self):
        from app.services.embedding.qdrant_service import _compose_base_url
        assert _compose_base_url("http://qdrant:6333", None) == "http://qdrant:6333"
        assert _compose_base_url("https://at-qdrant.example.com", None) == "https://at-qdrant.example.com"

    def test_zero_port_treated_as_no_port(self):
        from app.services.embedding.qdrant_service import _compose_base_url
        assert _compose_base_url("http://qdrant:6333", 0) == "http://qdrant:6333"

    def test_strips_trailing_slash(self):
        from app.services.embedding.qdrant_service import _compose_base_url
        assert _compose_base_url("https://at-qdrant.example.com/", 443) == "https://at-qdrant.example.com:443"


class TestPointId:
    def test_deterministic(self):
        assert _point_id("src1", 0) == _point_id("src1", 0)

    def test_changes_with_chunk_index(self):
        assert _point_id("src1", 0) != _point_id("src1", 1)

    def test_changes_with_source_id(self):
        assert _point_id("src1", 0) != _point_id("src2", 0)

    def test_fits_uint64(self):
        pid = _point_id("src1", 0)
        assert 0 <= pid < (1 << 64)


# ─────────────────────────────────────────────────────────────────────
# Endpoint end-to-end — real router, mocked services
# ─────────────────────────────────────────────────────────────────────


def _dummy_embed(dim: int = 1024):
    async def _embed_batch(chunks):
        return [EmbeddingResult(dense=[0.01 * (i + 1)] * dim) for i, _ in enumerate(chunks)]
    return _embed_batch


def _install(*, extract_return: dict | None = None, embed_dim: int = 1024):
    """Wire minimal mocks onto app.state for the AT endpoint.

    Returns a dict with handles for post-assertion inspection.
    """
    app.state._test_mode = True

    chunker = Chunker()
    app.state.chunker = chunker

    enricher = MagicMock()
    enricher.enrich_chunks = AsyncMock(side_effect=lambda document, chunks: list(chunks))
    app.state.contextual_enricher = enricher

    embedder = MagicMock()
    embedder.embed_batch = AsyncMock(side_effect=_dummy_embed(embed_dim))
    app.state.tei_embedder_at = embedder

    extractor = MagicMock()
    extractor.extract = AsyncMock(return_value=extract_return or {})
    app.state.funding_extractor = extractor

    qdrant = MagicMock()
    qdrant.ensure_at_collection = AsyncMock(return_value=False)
    qdrant.delete_by_filter = AsyncMock(return_value=0)
    qdrant.upsert_points = AsyncMock(side_effect=lambda _col, points: len(points))
    app.state.qdrant_at = qdrant

    return {"chunker": chunker, "embedder": embedder, "extractor": extractor, "qdrant": qdrant}


BASE_PAYLOAD = {
    "source_id": "web_foerderungen_001",
    "collection_name": "foerder_at",
    "url": "https://www.salzburg.gv.at/foerderungen",
    "content": "Sportförderung des Landes Salzburg. Die Förderung unterstützt Vereine.",
    "content_type": ["funding", "sport"],
    "language": "de",
    "metadata": {
        "assistant_id": "asst_foerder_at_01",
        "municipality_id": "land-salzburg",
        "title": "Sportförderung Salzburg",
        "source_type": "web",
    },
}


def _upsert_calls(qdrant_mock) -> list[tuple[str, list[dict]]]:
    return [(call.args[0], call.args[1]) for call in qdrant_mock.upsert_points.await_args_list]


def _all_points(qdrant_mock) -> list[dict]:
    out: list[dict] = []
    for call in qdrant_mock.upsert_points.await_args_list:
        out.extend(call.args[1])
    return out


class TestEndpoint:
    def test_single_collection_upsert(self):
        handles = _install()

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] is True
        assert body["data"]["collection_name"] == "foerder_at"
        assert body["data"]["vectors_stored"] == body["data"]["chunks_created"]

        calls = _upsert_calls(handles["qdrant"])
        assert len(calls) == 1
        assert calls[0][0] == "foerder_at"

    def test_override_state_or_province_stored_lowercase(self):
        handles = _install(extract_return={"state_or_province": ["salzburg"]})
        payload = {**BASE_PAYLOAD, "state_or_province": ["Lower Austria", "VIENNA"]}

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=payload)

        assert r.status_code == 200, r.text
        # Override wins over extractor; normalized to lowercase, dedupe/order preserved.
        for p in _all_points(handles["qdrant"]):
            assert p["payload"]["metadata"]["state_or_province"] == ["lower austria", "vienna"]

    def test_german_override_stored_as_english_canonical(self):
        """Override sent in German must land as the English-lowercase form
        so it matches extractor output for cross-ingest search filtering."""
        handles = _install(extract_return={"state_or_province": ["salzburg"]})
        payload = {**BASE_PAYLOAD, "state_or_province": ["Niederösterreich", "Wien"]}

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=payload)

        assert r.status_code == 200, r.text
        for p in _all_points(handles["qdrant"]):
            assert p["payload"]["metadata"]["state_or_province"] == ["lower austria", "vienna"]

    def test_extractor_value_used_when_no_override(self):
        handles = _install(extract_return={"state_or_province": ["tyrol", "salzburg"]})

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200, r.text
        for p in _all_points(handles["qdrant"]):
            assert p["payload"]["metadata"]["state_or_province"] == ["tyrol", "salzburg"]

    def test_state_or_province_empty_when_both_missing(self):
        handles = _install(extract_return={"state_or_province": []})

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200
        for p in _all_points(handles["qdrant"]):
            assert p["payload"]["metadata"]["state_or_province"] == []

    def test_point_shape_matches_at_schema(self):
        """Points must carry an integer ID and a flat 1024-float vector (not a
        named dict), matching the AT collection schema."""
        handles = _install()

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200, r.text
        points = _all_points(handles["qdrant"])
        assert len(points) > 0
        for p in points:
            assert isinstance(p["id"], int)
            assert 0 <= p["id"] < (1 << 64)
            assert isinstance(p["vector"], list)
            assert len(p["vector"]) == 1024
            assert all(isinstance(v, float) for v in p["vector"])

    def test_no_region_field_on_points(self):
        """region metadata was removed — must not appear on any stored point."""
        handles = _install(extract_return={"state_or_province": ["vienna"]})

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200
        for p in _all_points(handles["qdrant"]):
            assert "region" not in p["payload"]["metadata"]

    def test_source_name_stored_in_metadata_when_provided(self):
        handles = _install()
        payload = {**BASE_PAYLOAD, "source_name": "Land Salzburg — Sportförderung"}

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=payload)

        assert r.status_code == 200, r.text
        points = _all_points(handles["qdrant"])
        assert points
        for p in points:
            assert p["payload"]["metadata"]["source_name"] == "Land Salzburg — Sportförderung"

    def test_source_name_omitted_leaves_metadata_clean(self):
        handles = _install()

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200
        for p in _all_points(handles["qdrant"]):
            assert "source_name" not in p["payload"]["metadata"]

    def test_program_name_falls_back_to_title_when_extractor_missing(self):
        """When the extractor doesn't return a program_name, the stored
        metadata.program_name defaults to the title."""
        handles = _install(extract_return={"program_name": "", "state_or_province": []})
        payload = {**BASE_PAYLOAD}  # metadata.title = "Sportförderung Salzburg"

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=payload)

        assert r.status_code == 200, r.text
        for p in _all_points(handles["qdrant"]):
            assert p["payload"]["metadata"]["program_name"] == "Sportförderung Salzburg"

    def test_program_name_preserved_when_extractor_returns_one(self):
        """A distinct extractor program_name wins over the title fallback."""
        handles = _install(extract_return={
            "program_name": "Sportförderung 2025",
            "state_or_province": [],
        })

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200, r.text
        for p in _all_points(handles["qdrant"]):
            assert p["payload"]["metadata"]["program_name"] == "Sportförderung 2025"

    def test_extracted_contact_fields_land_in_metadata(self):
        """program_name / processing_office / contract_email / contract_phone
        from the funding extractor flow through to point metadata.
        contract_email / contract_phone are arrays — a funding doc can list
        multiple contacts."""
        handles = _install(extract_return={
            "program_name": "Sportförderung 2025",
            "processing_office": ["Abteilung Sport, Land Salzburg", "Magistrat Salzburg"],
            "contract_email": ["sport@salzburg.gv.at", "foerderungen@salzburg.gv.at"],
            "contract_phone": ["+43 662 8042 3333", "+43 662 8042 3334"],
            "state_or_province": ["salzburg"],
        })

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200, r.text
        points = _all_points(handles["qdrant"])
        assert points
        for p in points:
            md = p["payload"]["metadata"]
            assert md["program_name"] == "Sportförderung 2025"
            assert md["processing_office"] == ["Abteilung Sport, Land Salzburg", "Magistrat Salzburg"]
            assert md["contract_email"] == ["sport@salzburg.gv.at", "foerderungen@salzburg.gv.at"]
            assert md["contract_phone"] == ["+43 662 8042 3333", "+43 662 8042 3334"]

    def test_metadata_includes_source_id_and_omits_chunk_fields(self):
        """source_id is written to every point's metadata; chunk_id and
        chunk_index are intentionally not — they're only used to derive
        the deterministic integer point ID."""
        handles = _install()

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200
        points = _all_points(handles["qdrant"])
        assert len(points) > 0
        for p in points:
            md = p["payload"]["metadata"]
            assert md["source_id"] == BASE_PAYLOAD["source_id"]
            assert "chunk_id" not in md
            assert "chunk_index" not in md
            assert md["source_url"] == BASE_PAYLOAD["url"]

    def test_ensures_collection_before_delete_and_upsert(self):
        """Auto-create: ensure_at_collection is called with the body's
        collection_name and runs before delete_by_filter / upsert_points."""
        handles = _install()

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200, r.text
        qdrant = handles["qdrant"]
        qdrant.ensure_at_collection.assert_awaited_once()
        ensure_call = qdrant.ensure_at_collection.await_args
        assert ensure_call.args[0] == BASE_PAYLOAD["collection_name"]

    def test_ensure_collection_failure_maps_to_error_code(self):
        """If the collection can't be created (e.g. dim mismatch), the
        endpoint returns QDRANT_COLLECTION_NOT_FOUND."""
        from app.services.embedding.qdrant_service import QdrantError

        handles = _install()
        handles["qdrant"].ensure_at_collection = AsyncMock(
            side_effect=QdrantError("AT collection 'foerder_at' has vector size 1536, expected 1024.")
        )

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert body["error"] == "QDRANT_COLLECTION_NOT_FOUND"

    def test_deletes_by_source_id_before_upsert(self):
        """Idempotency path: delete by metadata.source_id before upserting
        so a repeat ingest fully replaces prior chunks."""
        handles = _install()

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200
        qdrant = handles["qdrant"]
        qdrant.delete_by_filter.assert_awaited()
        for call in qdrant.delete_by_filter.await_args_list:
            flt = call.args[1]
            assert flt["must"][0]["key"] == "metadata.source_id"
            assert flt["must"][0]["match"]["value"] == BASE_PAYLOAD["source_id"]
            assert call.args[0] == BASE_PAYLOAD["collection_name"]

    def test_missing_collection_name_rejected_by_pydantic(self):
        _install()
        payload = {k: v for k, v in BASE_PAYLOAD.items() if k != "collection_name"}

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=payload)

        assert r.status_code == 422

    def test_empty_collection_name_rejected_by_pydantic(self):
        _install()
        payload = {**BASE_PAYLOAD, "collection_name": ""}

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=payload)

        assert r.status_code == 422

    def test_whitespace_only_content_returns_validation_error_envelope(self):
        _install()
        payload = {**BASE_PAYLOAD, "content": "   "}

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=payload)

        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert body["error"] == "VALIDATION_EMPTY_CONTENT"

    def test_truly_empty_content_rejected_by_pydantic(self):
        _install()
        payload = {**BASE_PAYLOAD, "content": ""}

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=payload)

        assert r.status_code == 422

    def test_funding_extractor_called_with_at_country(self):
        handles = _install()

        with TestClient(app) as c:
            c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        kwargs = handles["extractor"].extract.await_args.kwargs
        assert kwargs["country"] == "AT"

    def test_default_chunking_uses_at_specific_constants(self):
        """When body.chunking is omitted, the router must pass the AT-specific
        defaults (1000 / 150) to the chunker — not the global 512 / 50."""
        handles = _install()

        real_chunk = handles["chunker"].chunk
        captured: dict = {}

        def _spy(*args, **kwargs):
            captured.update(kwargs)
            return real_chunk(*args, **kwargs)

        handles["chunker"].chunk = _spy

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        assert r.status_code == 200, r.text
        assert captured.get("max_chunk_size") == _AT_DEFAULT_CHUNK_SIZE == 1000
        assert captured.get("overlap") == _AT_DEFAULT_CHUNK_OVERLAP == 150
        assert captured.get("strategy") == "recursive"  # contextual uses recursive as base

    def test_body_chunking_overrides_at_defaults(self):
        """Explicit body.chunking still wins over the AT defaults."""
        handles = _install()

        real_chunk = handles["chunker"].chunk
        captured: dict = {}

        def _spy(*args, **kwargs):
            captured.update(kwargs)
            return real_chunk(*args, **kwargs)

        handles["chunker"].chunk = _spy

        payload = {**BASE_PAYLOAD, "chunking": {"strategy": "recursive", "max_chunk_size": 256, "overlap": 32}}
        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=payload)

        assert r.status_code == 200, r.text
        assert captured.get("max_chunk_size") == 256
        assert captured.get("overlap") == 32

    def test_response_metadata_counts(self):
        _install()

        with TestClient(app) as c:
            r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

        data = r.json()["data"]
        assert data["vectors_stored"] == data["chunks_created"]
        assert data["content_type"] == ["funding", "sport"]
        assert data["source_id"] == BASE_PAYLOAD["source_id"]
        assert data["collection_name"] == BASE_PAYLOAD["collection_name"]


# ─────────────────────────────────────────────────────────────────────
# Error-path tests
# ─────────────────────────────────────────────────────────────────────


def test_embedder_failure_returns_error_envelope():
    from app.services.embedding.bge_m3_client import EmbeddingError

    async def _fail(_chunks):
        raise EmbeddingError("TEI outage")

    handles = _install()
    handles["embedder"].embed_batch = AsyncMock(side_effect=_fail)

    with TestClient(app) as c:
        r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"] in {"EMBEDDING_FAILED", "EMBEDDING_OOM", "EMBEDDING_MODEL_NOT_LOADED"}


def test_upsert_disk_full_mapped_to_error_code():
    from app.services.embedding.qdrant_service import QdrantError

    async def _upsert_disk_full(_col, _points):
        raise QdrantError("disk is full")

    handles = _install()
    handles["qdrant"].upsert_points = AsyncMock(side_effect=_upsert_disk_full)

    with TestClient(app) as c:
        r = c.post("/api/v1/online/ingest/at", json=BASE_PAYLOAD)

    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"] == "QDRANT_DISK_FULL"
