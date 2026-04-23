"""Tests for POST /api/v1/local/document-parse, POST /api/v1/local/document-parse/upload, POST /api/v1/online/document-parse, and POST /api/v1/online/document-parse/upload endpoints."""

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
    TableData,
)


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.parse_from_url = AsyncMock()
    parser.parse_from_file = AsyncMock()
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
def client(mock_parser, mock_classifier):
    app.state._test_mode = True
    app.state.parser = mock_parser
    app.state.classifier = mock_classifier
    # Provide stubs for other services expected by existing routers
    app.state.scraping = MagicMock()
    app.state.sitemap_parser = MagicMock()
    with TestClient(app) as c:
        yield c


def test_parse_url_success(client, mock_parser):
    mock_parser.parse_from_url.return_value = ParseResult(
        status=ParseStatus.SUCCESS,
        document_type=DocumentType.PDF,
        text="Parsed content from URL",
        metadata=DocumentMetadata(page_count=2, word_count=4, language="en"),
        pages_parsed=2,
        pages_failed=0,
    )

    response = client.post("/api/v1/online/document-parse", json={
        "url": "https://example.com/report.pdf",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["data"]["content"] == "Parsed content from URL"
    assert data["data"]["pages"] == 2
    assert data["data"]["content_length"] == 23
    assert data["request_id"]

    mock_parser.parse_from_url.assert_called_once_with(
        url="https://example.com/report.pdf",
        mime_type=None,
    )


def test_parse_url_with_mime_type(client, mock_parser):
    mock_parser.parse_from_url.return_value = ParseResult(
        status=ParseStatus.SUCCESS,
        document_type=DocumentType.DOCX,
        text="DOCX from URL",
        metadata=DocumentMetadata(word_count=3),
        pages_parsed=1,
        pages_failed=0,
    )

    response = client.post("/api/v1/online/document-parse", json={
        "url": "https://example.com/doc",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    })
    assert response.status_code == 200
    assert response.json()["success"] is True

    mock_parser.parse_from_url.assert_called_once_with(
        url="https://example.com/doc",
        mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


def test_parse_smb_success(client, mock_parser):
    mock_parser.parse_from_file.return_value = ParseResult(
        status=ParseStatus.SUCCESS,
        document_type=DocumentType.PDF,
        text="Hello world from PDF",
        metadata=DocumentMetadata(page_count=3, word_count=4, language="de"),
        tables=[TableData(headers=["A", "B"], rows=[["1", "2"]])],
        pages_parsed=3,
        pages_failed=0,
    )

    response = client.post("/api/v1/local/document-parse", json={
        "file_path": "/mnt/share/docs/report.pdf",
        "source": "smb",
        "mime_type": "application/pdf",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["data"]["file_path"] == "/mnt/share/docs/report.pdf"
    assert data["data"]["content"] == "Hello world from PDF"
    assert data["data"]["pages"] == 3
    assert data["data"]["language"] == "de"
    assert data["data"]["extracted_tables"] == 1
    assert data["data"]["content_length"] == 20
    assert data["request_id"]


def test_parse_r2_success(client, mock_parser):
    mock_parser.parse_from_url.return_value = ParseResult(
        status=ParseStatus.SUCCESS,
        document_type=DocumentType.DOCX,
        text="Document content from R2",
        metadata=DocumentMetadata(word_count=4),
        pages_parsed=1,
        pages_failed=0,
    )

    response = client.post("/api/v1/local/document-parse", json={
        "file_path": "tenant/docs/file.docx",
        "source": "r2",
        "r2_presigned_url": "https://r2.example.com/presigned/file.docx?token=abc",
        "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["data"]["content"] == "Document content from R2"


def test_parse_r2_missing_presigned_url(client):
    response = client.post("/api/v1/local/document-parse", json={
        "file_path": "tenant/docs/file.docx",
        "source": "r2",
        "mime_type": "application/pdf",
    })
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is False
    assert data["error"] == "R2_FILE_NOT_FOUND"


def test_parse_unsupported_format(client, mock_parser):
    mock_parser.parse_from_file.return_value = ParseResult(
        status=ParseStatus.UNSUPPORTED,
        document_type=DocumentType.UNKNOWN,
        error="Unsupported document type: unknown",
    )

    response = client.post("/api/v1/local/document-parse", json={
        "file_path": "/mnt/share/docs/file.xyz",
        "source": "smb",
        "mime_type": "application/octet-stream",
    })
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "PARSE_UNSUPPORTED_FORMAT"


def test_parse_failed(client, mock_parser):
    mock_parser.parse_from_file.return_value = ParseResult(
        status=ParseStatus.FAILED,
        document_type=DocumentType.PDF,
        error="Parser error: failed to read file",
    )

    response = client.post("/api/v1/local/document-parse", json={
        "file_path": "/mnt/share/docs/broken.pdf",
        "source": "smb",
        "mime_type": "application/pdf",
    })
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "PARSE_FAILED"


def test_parse_encrypted_pdf(client, mock_parser):
    mock_parser.parse_from_file.return_value = ParseResult(
        status=ParseStatus.FAILED,
        document_type=DocumentType.PDF,
        error="Parser error: encrypted PDF requires password",
    )

    response = client.post("/api/v1/local/document-parse", json={
        "file_path": "/mnt/share/docs/secret.pdf",
        "source": "smb",
        "mime_type": "application/pdf",
    })
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "PARSE_ENCRYPTED"


def test_parse_empty_content(client, mock_parser):
    mock_parser.parse_from_file.return_value = ParseResult(
        status=ParseStatus.SUCCESS,
        document_type=DocumentType.PDF,
        text="   ",
        metadata=DocumentMetadata(),
        pages_parsed=1,
    )

    response = client.post("/api/v1/local/document-parse", json={
        "file_path": "/mnt/share/docs/blank.pdf",
        "source": "smb",
        "mime_type": "application/pdf",
    })
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "PARSE_EMPTY"


def test_parse_partial_success(client, mock_parser):
    mock_parser.parse_from_file.return_value = ParseResult(
        status=ParseStatus.PARTIAL,
        document_type=DocumentType.PDF,
        text="Page 1 content\n\nPage 3 content",
        metadata=DocumentMetadata(page_count=5),
        pages_parsed=3,
        pages_failed=2,
    )

    response = client.post("/api/v1/local/document-parse", json={
        "file_path": "/mnt/share/docs/partial.pdf",
        "source": "smb",
        "mime_type": "application/pdf",
    })
    data = response.json()
    assert data["success"] is True
    assert data["data"]["pages"] == 3
    assert data["data"]["content_length"] > 0


def test_parse_request_id(client, mock_parser):
    mock_parser.parse_from_file.return_value = ParseResult(
        status=ParseStatus.SUCCESS,
        document_type=DocumentType.TXT,
        text="test content",
        metadata=DocumentMetadata(),
        pages_parsed=1,
    )

    response = client.post(
        "/api/v1/local/document-parse",
        json={
            "file_path": "/mnt/share/test.txt",
            "source": "smb",
            "mime_type": "text/plain",
        },
        headers={"X-Request-ID": "parse-req-456"},
    )
    data = response.json()
    assert data["request_id"] == "parse-req-456"
    assert response.headers["X-Request-ID"] == "parse-req-456"


def test_parse_invalid_source(client):
    """Pydantic should reject invalid source values."""
    response = client.post("/api/v1/local/document-parse", json={
        "file_path": "/mnt/share/test.txt",
        "source": "ftp",
        "mime_type": "text/plain",
    })
    assert response.status_code == 422


def test_parse_upload_success(client, mock_parser):
    mock_parser.parse_from_file.return_value = ParseResult(
        status=ParseStatus.SUCCESS,
        document_type=DocumentType.PDF,
        text="Uploaded PDF content",
        metadata=DocumentMetadata(word_count=3, language="en"),
        pages_parsed=1,
        pages_failed=0,
    )

    response = client.post(
        "/api/v1/local/document-parse/upload",
        files={"file": ("test.pdf", b"%PDF-fake-content", "application/pdf")},
    )
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["data"]["content"] == "Uploaded PDF content"
    assert data["data"]["file_path"] == "test.pdf"


def test_parse_upload_empty_content(client, mock_parser):
    mock_parser.parse_from_file.return_value = ParseResult(
        status=ParseStatus.SUCCESS,
        document_type=DocumentType.TXT,
        text="",
        metadata=DocumentMetadata(),
        pages_parsed=1,
    )

    response = client.post(
        "/api/v1/local/document-parse/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    data = response.json()
    assert data["success"] is False
    assert data["error"] == "PARSE_EMPTY"
