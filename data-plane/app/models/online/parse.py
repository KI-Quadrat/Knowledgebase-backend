from pydantic import BaseModel, Field

from app.models.classify import ExtractedEntities
from app.models.common import UsageSummary


class OnlineParseRequest(BaseModel):
    """Request to parse a document from a public URL."""

    url: str = Field(..., description="Public URL pointing to a document (PDF, DOCX, etc.)")
    mime_type: str | None = Field(None, description="MIME type of the file (e.g. application/pdf). Auto-detected if omitted.")
    classify: bool = Field(
        True,
        description=(
            "If true, run content classification and entity extraction after parsing. "
            "Set false to return only parsed content and skip the LLM classifier."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "url": "https://example.com/report.pdf",
                },
                {
                    "url": "https://example.com/document.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                },
                {
                    "url": "https://example.com/report.pdf",
                    "classify": False,
                },
            ]
        }
    }


class OnlineParseData(BaseModel):
    """Extracted content from a URL-parsed document."""

    url: str = Field(..., description="Original URL from the request")
    content: str = Field(..., description="Extracted text content from the document")
    pages: int | None = Field(None, description="Number of pages successfully parsed")
    language: str | None = Field(None, description="Detected document language (ISO 639-1)")
    extracted_tables: int = Field(0, description="Number of tables extracted from the document")
    content_length: int = Field(..., description="Length of extracted content in characters")
    content_type: list[str] = Field(default_factory=list, description="Classifier-derived content categories for the parsed document (e.g. ['funding', 'renewable_energy']). Empty when the request uses classify=false. Pass this verbatim to /online/ingest only when classification was enabled.")
    entities: ExtractedEntities | None = Field(None, description="Structured entities extracted by the classifier (dates, deadlines, amounts, contacts, departments). Null when classification failed or the request uses classify=false.")
    usage: UsageSummary | None = Field(
        None,
        description=(
            "Per-stage billing for this parse. ``parse`` reports "
            "LlamaParse pages when the cloud parser is configured ($0 for "
            "the local parsers). ``classifier`` reports OpenAI tokens "
            "consumed by the always-on post-parse classification."
        ),
    )
