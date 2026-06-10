import httpx
import pytest

from app.services.scraping.scraper_client import (
    TARGET_NOT_FOUND_TITLE,
    ScraperClient,
)


@pytest.mark.asyncio
async def test_jina_warning_404_is_terminal_not_found():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "data": {
                    "title": "Original title",
                    "content": "Some fallback-looking content",
                    "warning": "Target URL returned error 404: Not Found",
                }
            },
        )

    client = ScraperClient()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._jina_key = "jina-key"
    client._jina_url = "https://jina.test"
    client._firecrawl_key = "firecrawl-key"
    client._firecrawl_url = "https://firecrawl.test"

    try:
        result = await client.crawl("https://missing.test/page", scraper="jina")
    finally:
        await client.close()

    assert result.success is False
    assert result.title == TARGET_NOT_FOUND_TITLE
    assert TARGET_NOT_FOUND_TITLE in (result.error or "")
    assert "Target URL returned error 404: Not Found" in (result.error or "")
    assert len(requests) == 1
    assert requests[0].method == "GET"


@pytest.mark.asyncio
async def test_firecrawl_json_not_found_is_terminal_not_found():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            request=request,
            json={"success": False, "error": "Not Found"},
        )

    client = ScraperClient()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client._jina_key = "jina-key"
    client._jina_url = "https://jina.test"
    client._firecrawl_key = "firecrawl-key"
    client._firecrawl_url = "https://firecrawl.test"

    try:
        result = await client.crawl("https://missing.test/page", scraper="firecrawl")
    finally:
        await client.close()

    assert result.success is False
    assert result.title == TARGET_NOT_FOUND_TITLE
    assert result.error == f"{TARGET_NOT_FOUND_TITLE}: Not Found"
    assert len(requests) == 1
    assert requests[0].method == "POST"


@pytest.mark.asyncio
async def test_httpx_404_returns_standard_not_found_failure():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request, text="Not Found")

    client = ScraperClient()
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        result = await client.crawl("https://missing.test/page", scraper="httpx")
    finally:
        await client.close()

    assert result.success is False
    assert result.title == TARGET_NOT_FOUND_TITLE
    assert result.error == f"{TARGET_NOT_FOUND_TITLE}: HTTP 404 Not Found"
