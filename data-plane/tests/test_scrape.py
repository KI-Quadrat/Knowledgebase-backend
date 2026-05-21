"""Tests for POST /api/v1/online/scrape and POST /api/v1/online/crawl endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.intelligence.models import (
    ClassifyResult,
    ContentCategory,
    ExtractedEntities,
)
from app.services.parsing.models import (
    DocumentMetadata,
    DocumentType,
    ParseResult,
    ParseStatus,
)
from app.routers.online.scrape import _is_thin_output
from app.services.scraping.crawl4ai_client import _extract_jina_links
from app.services.scraping.scraper_service import (
    DiscoveredDocument,
    PageMetadata,
    ScrapeResult,
    ScrapeStatus,
)


@pytest.fixture
def mock_scraper():
    scraper = MagicMock()
    scraper.scrape_url = AsyncMock()
    scraper.discover_urls = AsyncMock()
    scraper.is_ready = True
    return scraper


@pytest.fixture
def mock_sitemap_parser():
    parser = MagicMock()
    parser.parse = AsyncMock(return_value=[])
    return parser


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.parse_from_url = AsyncMock()
    return parser


@pytest.fixture
def mock_classifier():
    classifier = MagicMock()
    classifier.classify = AsyncMock(return_value=ClassifyResult(
        category=ContentCategory.GENERAL,
        confidence=0.5,
        sub_categories=[],
        entities=ExtractedEntities(),
        summary="",
    ))
    return classifier


@pytest.fixture
def client(mock_scraper, mock_sitemap_parser, mock_parser, mock_classifier):
    app.state._test_mode = True
    app.state.scraping = mock_scraper
    app.state.sitemap_parser = mock_sitemap_parser
    app.state.parser = mock_parser
    app.state.classifier = mock_classifier
    with TestClient(app) as c:
        yield c


def test_scrape_success(client, mock_scraper):
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Hello\n\nSome content here.",
        metadata=PageMetadata(title="Hello", language="de", word_count=4),
        discovered_links=["https://example.gv.at/other"],
    )

    response = client.post("/api/v1/online/scrape", json={"url": "https://example.gv.at/page"})
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["data"]["url"] == "https://example.gv.at/page"
    assert data["data"]["title"] == "Hello"
    assert data["data"]["content_length"] > 0
    assert data["data"]["links_found"] == 1
    assert data["request_id"]
    mock_scraper.scrape_url.assert_awaited_once()
    assert mock_scraper.scrape_url.await_args.kwargs["bypass_cache"] is False


def test_scrape_bypasses_cache_for_links_summary(client, mock_scraper):
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Hello\n\nSome content here.",
        metadata=PageMetadata(title="Hello", language="de", word_count=4),
        discovered_links=["https://example.gv.at/other"],
    )

    response = client.post(
        "/api/v1/online/scrape",
        json={"url": "https://example.gv.at/page", "scraper": "jina", "links_summary": True},
    )
    assert response.status_code == 200
    mock_scraper.scrape_url.assert_awaited_once()
    assert mock_scraper.scrape_url.await_args.kwargs["bypass_cache"] is True


def test_scrape_invalid_url(client):
    response = client.post("/api/v1/online/scrape", json={"url": "not-a-url"})
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is False
    assert data["error"] == "VALIDATION_URL_INVALID"


def test_scrape_empty_content(client, mock_scraper):
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/empty",
        status=ScrapeStatus.SUCCESS,
        markdown="   ",
        metadata=PageMetadata(),
    )

    response = client.post("/api/v1/online/scrape", json={"url": "https://example.gv.at/empty"})
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "SCRAPE_EMPTY"


def test_scrape_failed(client, mock_scraper):
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/fail",
        status=ScrapeStatus.FAILED,
        error="Connection refused",
    )

    response = client.post("/api/v1/online/scrape", json={"url": "https://example.gv.at/fail"})
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "SCRAPE_FAILED"


def test_scrape_timeout(client, mock_scraper):
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/slow",
        status=ScrapeStatus.TIMEOUT,
        error="Timed out",
    )

    response = client.post("/api/v1/online/scrape", json={"url": "https://example.gv.at/slow"})
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "SCRAPE_TIMEOUT"


class TestIsThinOutput:
    """Unit coverage for the word-count / ratio heuristic."""

    def test_empty_markdown_is_thin_when_html_present(self):
        assert _is_thin_output("", "<html>" + "x" * 2000 + "</html>") is True
        assert _is_thin_output(None, "<html>" + "x" * 2000 + "</html>") is True
        assert _is_thin_output("   ", "<html>" + "x" * 2000 + "</html>") is True

    def test_short_markdown_is_thin_only_against_nontrivial_html(self):
        # Short markdown + sizeable HTML (>1000 chars) → thin.
        assert _is_thin_output("short note", "<html>" + "x" * 2000 + "</html>") is True
        # Short markdown + trivially small HTML → genuinely short page, not thin.
        assert _is_thin_output("short note", "<html>abc</html>") is False

    def test_long_markdown_not_thin(self):
        md = "Ausführlicher Inhalt. " * 200
        assert _is_thin_output(md, "<html>" + "x" * 2000 + "</html>") is False

    def test_ratio_alone_does_not_trip(self):
        # Ratio-alone no longer trips thinness — both word-count AND ratio
        # signals must fire (the old OR-logic doubled scrape time on normal
        # short pages). 200 words clears the word threshold even against
        # massive HTML.
        md = "Wort " * 200
        html = "<html>" + "x" * 500_000 + "</html>"
        assert _is_thin_output(md, html) is False

    def test_missing_html_suppresses_detection(self):
        # Without HTML we refuse to retry — no reliable signal.
        assert _is_thin_output("", None) is False
        assert _is_thin_output("short", None) is False
        assert _is_thin_output("Wort " * 200, None) is False


def test_thin_fit_output_triggers_raw_retry(client, mock_scraper):
    """When fit-mode markdown is suspiciously short relative to the HTML,
    the router retransparent-retries once in raw mode and returns the
    richer output."""
    thin = ScrapeResult(
        url="https://example.gv.at/labelvalue",
        status=ScrapeStatus.SUCCESS,
        markdown="Nur kurzer Wartungshinweis.",
        html="<html>" + "x" * 20000 + "</html>",
        metadata=PageMetadata(title="x", language="de", word_count=3),
    )
    rich = ScrapeResult(
        url="https://example.gv.at/labelvalue",
        status=ScrapeStatus.SUCCESS,
        markdown="# Förderung\n\n" + ("Ausführlicher Inhalt. " * 200),
        html="<html>" + "x" * 20000 + "</html>",
        metadata=PageMetadata(title="Förderung", language="de", word_count=400),
    )
    mock_scraper.scrape_url.side_effect = [thin, rich]

    response = client.post("/api/v1/online/scrape", json={"url": "https://example.gv.at/labelvalue"})
    assert response.status_code == 200

    # Two scrape calls: first fit, second raw (with bypass_cache=True).
    assert mock_scraper.scrape_url.await_count == 2
    first_options = mock_scraper.scrape_url.await_args_list[0].args[1]
    retry_options = mock_scraper.scrape_url.await_args_list[1].args[1]
    assert first_options.markdown_type == "fit"
    assert retry_options.markdown_type == "raw"
    assert mock_scraper.scrape_url.await_args_list[1].kwargs["bypass_cache"] is True

    # The returned content is the richer raw scrape.
    assert "Ausführlicher Inhalt" in response.json()["data"]["content"]


def test_healthy_fit_output_does_not_retry(client, mock_scraper):
    """A page with plenty of markdown (well over the word threshold) stays in
    fit mode — no retry tax."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/rich",
        status=ScrapeStatus.SUCCESS,
        markdown="Reichhaltiger Seiteninhalt. " * 200,
        html="<html>" + "<p>ok</p>" * 100 + "</html>",
        metadata=PageMetadata(title="Rich", language="de", word_count=400),
    )

    response = client.post("/api/v1/online/scrape", json={"url": "https://example.gv.at/rich"})
    assert response.status_code == 200
    assert mock_scraper.scrape_url.await_count == 1


def test_raw_mode_never_retries(client, mock_scraper):
    """If the caller explicitly picked raw / citations, the fit->raw retry
    must not fire even on thin output (there's nothing to fall back to)."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/short",
        status=ScrapeStatus.SUCCESS,
        markdown="tiny",
        html="<html>" + "x" * 20000 + "</html>",
        metadata=PageMetadata(title="x", language="de", word_count=1),
    )

    client.post(
        "/api/v1/online/scrape",
        json={"url": "https://example.gv.at/short", "markdown_type": "raw"},
    )
    assert mock_scraper.scrape_url.await_count == 1


def test_crawl_sitemap(client, mock_sitemap_parser):
    mock_sitemap_parser.parse.return_value = [
        "https://example.gv.at/page1",
        "https://example.gv.at/page2",
        "https://example.gv.at/files/doc.pdf",
    ]

    response = client.post("/api/v1/online/crawl", json={
        "url": "https://example.gv.at/sitemap.xml",
        "method": "sitemap",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["data"]["method_used"] == "sitemap"
    assert data["data"]["total_urls"] == 3

    # Check type detection
    urls = data["data"]["urls"]
    types = {u["url"]: u["type"] for u in urls}
    assert types["https://example.gv.at/page1"] == "page"
    assert types["https://example.gv.at/files/doc.pdf"] == "document"


def test_crawl_sitemap_not_found(client, mock_sitemap_parser):
    mock_sitemap_parser.parse.return_value = []

    response = client.post("/api/v1/online/crawl", json={
        "url": "https://example.gv.at/sitemap.xml",
        "method": "sitemap",
    })
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "CRAWL_SITEMAP_NOT_FOUND"


def test_crawl_bfs(client, mock_scraper):
    mock_scraper.discover_urls.return_value = (
        ["https://example.gv.at/", "https://example.gv.at/about"],
        [DiscoveredDocument(url="https://example.gv.at/doc.pdf", type="pdf")],
    )

    response = client.post("/api/v1/online/crawl", json={
        "url": "https://example.gv.at",
        "method": "crawl",
        "max_depth": 2,
        "max_urls": 50,
    })
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["data"]["method_used"] == "crawl"
    assert data["data"]["total_urls"] == 3  # 2 pages + 1 doc


def test_crawl_invalid_url(client):
    response = client.post("/api/v1/online/crawl", json={
        "url": "ftp://bad",
        "method": "sitemap",
    })
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "VALIDATION_URL_INVALID"


def test_request_id_in_scrape_response(client, mock_scraper):
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at",
        status=ScrapeStatus.SUCCESS,
        markdown="content",
        metadata=PageMetadata(word_count=1),
    )

    response = client.post(
        "/api/v1/online/scrape",
        json={"url": "https://example.gv.at"},
        headers={"X-Request-ID": "test-req-123"},
    )
    data = response.json()
    assert data["request_id"] == "test-req-123"
    assert response.headers["X-Request-ID"] == "test-req-123"


# ── inner_img tests ──────────────────────────────────────


def test_scrape_inner_img_parses_images(client, mock_scraper, mock_parser):
    """When inner_img=true, images from HTML are parsed via LlamaParse and returned."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Page with images",
        html='<html><body>'
             '<img src="https://example.gv.at/photo.jpg" alt="Town hall" title="Rathaus">'
             '<img src="/images/banner.png" alt="Banner">'
             '<img src="data:image/gif;base64,R0lGODlh" alt="pixel">'
             '</body></html>',
        metadata=PageMetadata(title="Page", word_count=3),
    )

    mock_parser.parse_from_url.side_effect = [
        ParseResult(
            status=ParseStatus.SUCCESS,
            document_type=DocumentType.JPG,
            text="Town hall building with Austrian flag",
            metadata=DocumentMetadata(),
        ),
        ParseResult(
            status=ParseStatus.SUCCESS,
            document_type=DocumentType.PNG,
            text="Welcome to Wiener Neudorf",
            metadata=DocumentMetadata(),
        ),
    ]

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
        "inner_img": True,
    })
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    images = data["data"]["inner_images"]
    assert images is not None
    assert len(images) == 2  # data: URL is skipped

    assert images[0]["url"] == "https://example.gv.at/photo.jpg"
    assert images[0]["alt"] == "Town hall"
    assert images[0]["title"] == "Rathaus"
    assert images[0]["content"] == "Town hall building with Austrian flag"
    assert images[0]["content_length"] == len("Town hall building with Austrian flag")

    assert images[1]["url"] == "https://example.gv.at/images/banner.png"
    assert images[1]["alt"] == "Banner"
    assert images[1]["content"] == "Welcome to Wiener Neudorf"


def test_scrape_inner_img_parse_failure(client, mock_scraper, mock_parser):
    """When image parsing fails, error is returned with alt/title preserved."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Page",
        html='<html><body><img src="/broken.jpg" alt="test"></body></html>',
        metadata=PageMetadata(word_count=1),
    )

    mock_parser.parse_from_url.return_value = ParseResult(
        status=ParseStatus.FAILED,
        document_type=DocumentType.JPG,
        error="Unsupported image format",
    )

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
        "inner_img": True,
    })
    data = response.json()
    assert data["success"] is True
    images = data["data"]["inner_images"]
    assert len(images) == 1
    assert images[0]["content"] is None
    assert images[0]["error"] == "Unsupported image format"
    assert images[0]["alt"] == "test"


def test_scrape_inner_img_false_returns_null(client, mock_scraper):
    """When inner_img=false (default), inner_images is null."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Hello",
        html='<html><body><img src="/photo.jpg" alt="test"></body></html>',
        metadata=PageMetadata(word_count=1),
    )

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
    })
    data = response.json()
    assert data["success"] is True
    assert data["data"]["inner_images"] is None


def test_scrape_inner_img_no_html(client, mock_scraper):
    """When inner_img=true but no HTML is available, inner_images is null."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Hello",
        html=None,
        metadata=PageMetadata(word_count=1),
    )

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
        "inner_img": True,
    })
    data = response.json()
    assert data["success"] is True
    assert data["data"]["inner_images"] is None


# ── inner_docs tests ─────────────────────────────────────


def test_scrape_inner_docs_parses_documents(client, mock_scraper, mock_parser):
    """When inner_docs=true, discovered documents are parsed and returned."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Page with docs",
        metadata=PageMetadata(title="Page", word_count=3),
        discovered_documents=[
            DiscoveredDocument(url="https://example.gv.at/report.pdf", type="pdf", link_text="Annual Report"),
            DiscoveredDocument(url="https://example.gv.at/form.docx", type="docx", link_text="Application Form"),
        ],
    )

    mock_parser.parse_from_url.side_effect = [
        ParseResult(
            status=ParseStatus.SUCCESS,
            document_type=DocumentType.PDF,
            text="Parsed PDF content here.",
            metadata=DocumentMetadata(title="Annual Report 2025", language="de", page_count=5),
            pages_parsed=5,
        ),
        ParseResult(
            status=ParseStatus.SUCCESS,
            document_type=DocumentType.DOCX,
            text="Form fields and instructions.",
            metadata=DocumentMetadata(title="Application Form", language="de", page_count=2),
            pages_parsed=2,
        ),
    ]

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
        "inner_docs": True,
    })
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    docs = data["data"]["inner_documents"]
    assert docs is not None
    assert len(docs) == 2

    assert docs[0]["url"] == "https://example.gv.at/report.pdf"
    assert docs[0]["title"] == "Annual Report"  # link_text takes precedence
    assert docs[0]["doc_type"] == "pdf"
    assert docs[0]["content"] == "Parsed PDF content here."
    assert docs[0]["pages"] == 5
    assert docs[0]["content_length"] == len("Parsed PDF content here.")
    assert docs[0]["error"] is None

    assert docs[1]["url"] == "https://example.gv.at/form.docx"
    assert docs[1]["doc_type"] == "docx"
    assert docs[1]["content"] == "Form fields and instructions."
    assert docs[1]["pages"] == 2


def test_scrape_inner_docs_parse_failure(client, mock_scraper, mock_parser):
    """When a document fails to parse, error is returned instead of content."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Page",
        metadata=PageMetadata(word_count=1),
        discovered_documents=[
            DiscoveredDocument(url="https://example.gv.at/broken.pdf", type="pdf", link_text="Broken PDF"),
        ],
    )

    mock_parser.parse_from_url.return_value = ParseResult(
        status=ParseStatus.FAILED,
        document_type=DocumentType.PDF,
        error="Document is encrypted",
    )

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
        "inner_docs": True,
    })
    data = response.json()
    assert data["success"] is True  # scrape itself succeeded
    docs = data["data"]["inner_documents"]
    assert len(docs) == 1
    assert docs[0]["content"] is None
    assert docs[0]["error"] == "Document is encrypted"


def test_scrape_inner_docs_exception(client, mock_scraper, mock_parser):
    """When parser raises an exception, it's caught and returned as error."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Page",
        metadata=PageMetadata(word_count=1),
        discovered_documents=[
            DiscoveredDocument(url="https://example.gv.at/timeout.pdf", type="pdf", link_text="Slow Doc"),
        ],
    )

    mock_parser.parse_from_url.side_effect = TimeoutError("Download timed out")

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
        "inner_docs": True,
    })
    data = response.json()
    assert data["success"] is True
    docs = data["data"]["inner_documents"]
    assert len(docs) == 1
    assert docs[0]["error"] == "Download timed out"
    assert docs[0]["content"] is None


def test_scrape_inner_docs_false_returns_null(client, mock_scraper):
    """When inner_docs=false (default), inner_documents is null."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Hello",
        metadata=PageMetadata(word_count=1),
        discovered_documents=[
            DiscoveredDocument(url="https://example.gv.at/doc.pdf", type="pdf"),
        ],
    )

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
    })
    data = response.json()
    assert data["success"] is True
    assert data["data"]["inner_documents"] is None


def test_scrape_inner_docs_no_documents(client, mock_scraper):
    """When inner_docs=true but no documents found, inner_documents is null."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Hello",
        metadata=PageMetadata(word_count=1),
        discovered_documents=[],
    )

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
        "inner_docs": True,
    })
    data = response.json()
    assert data["success"] is True
    assert data["data"]["inner_documents"] is None


# ── inner_img + inner_docs combined ──────────────────────


def test_scrape_both_inner_img_and_docs(client, mock_scraper, mock_parser):
    """When both inner_img and inner_docs are true, both are returned."""
    mock_scraper.scrape_url.return_value = ScrapeResult(
        url="https://example.gv.at/page",
        status=ScrapeStatus.SUCCESS,
        markdown="# Full page",
        html='<html><body>'
             '<img src="/logo.png" alt="Logo">'
             '<a href="/report.pdf">Report</a>'
             '</body></html>',
        metadata=PageMetadata(title="Full", word_count=2),
        discovered_documents=[
            DiscoveredDocument(url="https://example.gv.at/report.pdf", type="pdf", link_text="Report"),
        ],
    )

    # parse_from_url is called for both the image and the document
    mock_parser.parse_from_url.side_effect = [
        ParseResult(
            status=ParseStatus.SUCCESS,
            document_type=DocumentType.PNG,
            text="Company logo text",
            metadata=DocumentMetadata(),
        ),
        ParseResult(
            status=ParseStatus.SUCCESS,
            document_type=DocumentType.PDF,
            text="Report content.",
            metadata=DocumentMetadata(language="de"),
            pages_parsed=1,
        ),
    ]

    response = client.post("/api/v1/online/scrape", json={
        "url": "https://example.gv.at/page",
        "inner_img": True,
        "inner_docs": True,
    })
    data = response.json()
    assert data["success"] is True

    assert data["data"]["inner_images"] is not None
    assert len(data["data"]["inner_images"]) == 1
    assert data["data"]["inner_images"][0]["alt"] == "Logo"
    assert data["data"]["inner_images"][0]["content"] == "Company logo text"

    assert data["data"]["inner_documents"] is not None
    assert len(data["data"]["inner_documents"]) == 1
    assert data["data"]["inner_documents"][0]["content"] == "Report content."


def test_extract_jina_links_prefers_direct_links_summary_urls():
    data = {
        "data": {
            "links_summary": {
                "urls": [
                    "/news/article-1",
                    "https://example.test/resources/feed.xml",
                    "/programs/open-call",
                ]
            }
        }
    }

    links = _extract_jina_links(data, "https://example.test/")

    assert links == [
        "https://example.test/news/article-1",
        "https://example.test/resources/feed.xml",
        "https://example.test/programs/open-call",
    ]
