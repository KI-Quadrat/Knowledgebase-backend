"""
POST /api/v1/online/document-parse        — Parse a document from a public URL.
POST /api/v1/online/document-parse/upload — Upload and parse a document file.
"""

import os
import tempfile

from fastapi import APIRouter, File, Request, UploadFile

from app.models.classify import ExtractedEntities as ClassifyEntities
from app.models.common import ResponseEnvelope, StageUsage, UsageSummary
from app.models.online.parse import OnlineParseData, OnlineParseRequest
from app.routers._parse_utils import check_parse_failure
from app.services import cost
from app.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/online", tags=["Online - Document Parsing"])


@router.post(
    "/document-parse",
    summary="Parse a document from URL",
    description=(
        "Download and extract text, tables, and metadata from a document at a public URL.\n\n"
        "By default, the extracted text is classified after parsing so the response includes "
        "`content_type`, `entities`, and classifier usage. Set request field `classify: false` "
        "to skip that extra LLM call; the response then returns `content_type: []` and "
        "`entities: null`.\n\n"
        "**Request fields:**\n\n"
        "| Field | Type | Required | Default | Description |\n"
        "|-------|------|----------|---------|-------------|\n"
        "| `url` | string | Required | — | Public document URL to download and parse. |\n"
        "| `mime_type` | string | Optional | `null` | MIME type hint. Auto-detected if omitted. |\n"
        "| `classify` | boolean | Optional | `true` | Run classifier after parsing. Set `false` to skip classifier latency/cost. |\n\n"
        "**Supported formats:** PDF, DOCX, DOC, PPTX, ODT, XLSX, XLS, TXT, CSV, HTML, RTF.\n\n"
        "**Example without classification:**\n"
        "```json\n"
        "{ \"url\": \"https://example.com/report.pdf\", \"classify\": false }\n"
        "```\n\n"
        "**Optional X-API-Key header** when API-key auth is configured.\n\n"
        "**Error codes:** `PARSE_FAILED`, `PARSE_ENCRYPTED`, `PARSE_CORRUPTED`, `PARSE_EMPTY`, "
        "`PARSE_TIMEOUT`, `PARSE_UNSUPPORTED_FORMAT`"
    ),
    response_description="Extracted text content with page count, language, and table count",
)
async def parse_online(body: OnlineParseRequest, request: Request) -> ResponseEnvelope[OnlineParseData]:
    request_id = request.state.request_id
    parser = request.app.state.parser

    result = await parser.parse_from_url(
        url=body.url,
        mime_type=body.mime_type,
    )

    return await _build_response(
        result, body.url, request_id, request.app.state.classifier, parser,
        classify=body.classify,
        audit=request.app.state.scraping.audit,
    )


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
    "**Optional X-API-Key header** when API-key auth is configured.\n\n"
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
            result, file.filename or "upload", request_id, request.app.state.classifier, parser,
            audit=request.app.state.scraping.audit,
        )

    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


async def _build_response(
    result, url: str, request_id: str, classifier, parser, *, audit=None,
    classify: bool = True,
) -> ResponseEnvelope[OnlineParseData]:
    """Convert a ParseResult into the standard API response."""
    error = check_parse_failure(result, request_id)
    if error:
        return ResponseEnvelope(**error)

    content = result.text or ""
    content_type: list[str] = []
    entities: ClassifyEntities | None = None
    classify_usage: StageUsage | None = None
    if classify:
        content_type, entities, classify_usage = await _classify_content(
            classifier, content, language=result.metadata.language, source_url=url
        )

    # Build per-stage usage. Parsing itself costs LlamaParse pages when the
    # cloud parser is in use; everything else is self-hosted at $0. The
    # classifier call (always-on after parse) contributes its OpenAI tokens.
    usage_entries: list[StageUsage] = []
    pages = result.pages_parsed or 0
    if getattr(parser, "_use_llama", False) and pages > 0:
        usage_entries.append(StageUsage(
            stage="parse",
            provider="llamaparse",
            pages=pages,
            cost_usd=cost.llamaparse_cost(pages),
        ))
    else:
        usage_entries.append(StageUsage(
            stage="parse",
            provider="local",
            pages=pages,
            cost_usd=0.0,
        ))
    if classify_usage is not None:
        usage_entries.append(classify_usage)
    usage = UsageSummary.from_entries(usage_entries)

    if audit is not None:
        await audit.log_usage(
            usage_entries,
            endpoint="document_parse",
            request_id=request_id,
            url=url,
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
            usage=usage,
        ),
        request_id=request_id,
    )


async def _classify_content(
    classifier, content: str, language: str | None, source_url: str
) -> tuple[list[str], ClassifyEntities | None, StageUsage | None]:
    """Run the classifier over content and return (content_type, entities, usage).

    Failures are logged and degraded to (['general'], None, None) —
    classification is informational on parse, so it should not fail the
    request. The usage record is forwarded so the response surfaces
    classifier token spend alongside parse usage.
    """
    try:
        result = await classifier.classify(content, language=language or "de")
    except Exception as exc:
        log.warning("classify_after_parse_failed", url=source_url, error=str(exc))
        return (["general"], None, None)

    content_type = [result.category.value] + result.sub_categories
    entities = ClassifyEntities(
        dates=result.entities.dates,
        deadlines=result.entities.deadlines,
        amounts=result.entities.amounts,
        contacts=result.entities.contacts,
        departments=result.entities.departments,
    )
    return (content_type, entities, getattr(result, "usage", None))
