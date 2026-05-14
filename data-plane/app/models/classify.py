from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Request to classify content and extract structured entities.

    Designed for German-language municipality documents. Supports 9 content
    categories and extracts dates, deadlines, monetary amounts, email contacts,
    and department references.
    """

    content: str = Field(..., min_length=1, description="Text content to classify (from /parse or /scrape)")
    language: str = Field("de", description="ISO 639-1 language code (default: 'de' for German)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "content": "Das Förderprogramm für erneuerbare Energien gilt ab 01.04.2025. Antragsfrist bis 30.06.2025. Förderhöhe bis EUR 5.000. Kontakt: energie@example.gv.at, Umweltamt.",
                    "language": "de",
                }
            ]
        }
    }


class ExtractedEntities(BaseModel):
    """Structured entities extracted from document content."""

    dates: list[str] = Field(default_factory=list, description="Dates found (e.g. '01.04.2025', '2025-06-30')")
    deadlines: list[str] = Field(default_factory=list, description="Dates identified as deadlines/due dates")
    amounts: list[str] = Field(default_factory=list, description="Monetary amounts (e.g. 'EUR 5.000', '€ 10.000')")
    contacts: list[str] = Field(default_factory=list, description="Email addresses found")
    departments: list[str] = Field(default_factory=list, description="Department/office names (e.g. 'Umweltamt')")


class ClassifyData(BaseModel):
    """Classification result with entities and summary."""

    classification: str = Field(
        ...,
        description="Content category: funding, event, policy, contact, form, announcement, minutes, report, or general",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score (0.0 to 1.0)")
    sub_categories: list[str] = Field(default_factory=list, description="Fine-grained sub-categories (e.g. renewable_energy, housing)")
    entities: ExtractedEntities = Field(..., description="Extracted named entities")
    summary: str = Field(..., description="Auto-generated summary of the content (max ~300 chars)")
