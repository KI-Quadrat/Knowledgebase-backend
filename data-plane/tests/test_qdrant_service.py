import httpx
import pytest

from app.services.embedding.qdrant_service import QdrantService


@pytest.mark.asyncio
async def test_create_collection_creates_default_payload_indexes():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path == "/collections/test-collection":
            return httpx.Response(404)
        if request.method == "PUT" and request.url.path == "/collections/test-collection":
            return httpx.Response(200, json={"result": True})
        if request.method == "PUT" and request.url.path == "/collections/test-collection/index":
            return httpx.Response(200, json={"result": True})
        return httpx.Response(500)

    service = QdrantService(url="http://qdrant.test")
    service._client = httpx.AsyncClient(
        base_url="http://qdrant.test",
        transport=httpx.MockTransport(handler),
    )

    try:
        created = await service.create_collection(
            name="test-collection",
            sparse=False,
            multi_vector={"dense_bge_m3": 1024},
        )
    finally:
        await service.shutdown()

    assert created is True

    index_fields = [
        request.read().decode()
        for request in requests
        if request.method == "PUT" and request.url.path == "/collections/test-collection/index"
    ]
    assert index_fields == [
        '{"field_name":"metadata.source_id","field_schema":"keyword"}',
        '{"field_name":"metadata.source_url","field_schema":"keyword"}',
        '{"field_name":"assistant_id","field_schema":"keyword"}',
        '{"field_name":"municipality_id","field_schema":"keyword"}',
    ]
