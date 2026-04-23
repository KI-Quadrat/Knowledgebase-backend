from pydantic import BaseModel, Field

from app.models.classify import ExtractedEntities


class OnlineParseRequest(BaseModel):
    """Request to parse a document from a public URL."""

    url: str = Field(..., description="Public URL pointing to a document (PDF, DOCX, etc.)")
    mime_type: str | None = Field(None, description="MIME type of the file (e.g. application/pdf). Auto-detected if omitted.")

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
    content_type: list[str] = Field(default_factory=list, description="Classifier-derived content categories for the parsed document (e.g. ['funding', 'renewable_energy']). Pass this verbatim to /online/ingest.")
    entities: ExtractedEntities | None = Field(None, description="Structured entities extracted by the classifier (dates, deadlines, amounts, contacts, departments). Null when classification failed.")
