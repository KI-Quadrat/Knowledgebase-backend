"""
POST /api/v1/online/document-parse        — Parse a document from a public URL.
POST /api/v1/online/document-parse/upload — Upload and parse a document file.
"""

import os
import tempfile

from fastapi import APIRouter, File, Request, UploadFile

from app.models.classify import ExtractedEntities as ClassifyEntities
from app.models.common import ResponseEnvelope
from app.models.online.parse import OnlineParseData, OnlineParseRequest
from app.routers._parse_utils import check_parse_failure
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/online", tags=["Online - Document Parsing"])


@router.post(
    "/document-parse",
    summary="Parse a document from URL",
    description="Download and extract text, tables, and metadata from a document at a public URL.\n\n"
    "**Supported formats:** PDF, DOCX, DOC, PPTX, ODT, XLSX, XLS, TXT, CSV, HTML, RTF.\n\n"
    "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.\n\n"
    "**Error codes:** `PARSE_FAILED`, `PARSE_ENCRYPTED`, `PARSE_CORRUPTED`, `PARSE_EMPTY`, "
    "`PARSE_TIMEOUT`, `PARSE_UNSUPPORTED_FORMAT`",
    response_description="Extracted text content with page count, language, and table count",
)
async def parse_online(body: OnlineParseRequest, request: Request) -> ResponseEnvelope[OnlineParseData]:
    request_id = request.state.request_id
    parser = request.app.state.parser

    result = await parser.parse_from_url(
        url=body.url,
        mime_type=body.mime_type,
    )

    return await _build_response(result, body.url, request_id, request.app.state.classifier)


@router.post(
    "/document-parse/upload",
    summary="Parse an uploaded document",
    description="Upload a document file directly and extract text, tables, and metadata.\n\n"
    "**Content-Type:** `multipart/form-data` — send the raw binary file in the `file` form field.\n"
    "Do **not** base64-encode the file; send the original binary document as-is.\n\n"
    "**Supported formats:** PDF, DOCX, DOC, PPTX, ODT, XLSX, XLS, TXT, CSV, HTML, RTF.\n\n"
    "**Example (cURL):**\n"
    "```\n"
    "curl -X POST /api/v1/online/document-parse/upload -H \"X-API-Key: your-key\" -F \"file=@report.pdf\"\n"
    "```\n\n"
    "**Optional X-API-Key header** — required only when `DP_ONLINE_API_KEYS` is configured.\n\n"
    "**Error codes:** `PARSE_FAILED`, `PARSE_ENCRYPTED`, `PARSE_CORRUPTED`, `PARSE_EMPTY`, "
    "`PARSE_TIMEOUT`, `PARSE_UNSUPPORTED_FORMAT`",
    response_description="Extracted text content with page count, language, and table count",
)
async def parse_online_upload(request: Request, file: UploadFile = File(...)) -> ResponseEnvelope[OnlineParseData]:
    request_id = request.state.request_id
    parser = request.app.state.parser

    # Save upload to temp file
    suffix = os.path.splitext(file.filename or "")[1] or ""
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        content = await file.read()
        with os.fdopen(fd, "wb") as f:
            f.write(content)

        result = await parser.parse_from_file(
            file_path=temp_path,
            mime_type=file.content_type,
            filename=file.filename,
        )

        return await _build_response(
            result, file.filename or "upload", request_id, request.app.state.classifier
        )

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def _build_response(
    result, url: str, request_id: str, classifier
) -> ResponseEnvelope[OnlineParseData]:
    """Convert a ParseResult into the standard API response."""
    error = check_parse_failure(result, request_id)
    if error:
        return ResponseEnvelope(**error)

    content = result.text or ""
    content_type, entities = await _classify_content(
        classifier, content, language=result.metadata.language, source_url=url
    )
    return ResponseEnvelope(
        success=True,
        data=OnlineParseData(
            url=url,
            content=content,
            pages=result.pages_parsed,
            language=result.metadata.language,
            extracted_tables=len(result.tables),
            content_length=len(content),
            content_type=content_type,
            entities=entities,
        ),
        request_id=request_id,
    )


async def _classify_content(
    classifier, content: str, language: str | None, source_url: str
) -> tuple[list[str], ClassifyEntities | None]:
    """Run the classifier over content and return (content_type, entities).

    Failures are logged and degraded to (['general'], None) — classification
    is informational on parse, so it should not fail the request.
    """
    try:
        result = await classifier.classify(content, language=language or "de")
    except Exception as exc:
        log.warning("classify_after_parse_failed", url=source_url, error=str(exc))
        return (["general"], None)

    content_type = [result.category.value] + result.sub_categories
    entities = ClassifyEntities(
        dates=result.entities.dates,
        deadlines=result.entities.deadlines,
        amounts=result.entities.amounts,
        contacts=result.entities.contacts,
        departments=result.entities.departments,
    )
    return (content_type, entities)
