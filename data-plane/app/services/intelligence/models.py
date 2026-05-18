"""Internal models for the intelligence subsystem."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from app.models.common import StageUsage


class ContentCategory(str, Enum):
    FUNDING = "funding"
    EVENT = "event"
    POLICY = "policy"
    CONTACT = "contact"
    FORM = "form"
    ANNOUNCEMENT = "announcement"
    MINUTES = "minutes"
    REPORT = "report"
    GENERAL = "general"


class ExtractedEntities(BaseModel):
    dates: list[str] = Field(default_factory=list)
    deadlines: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    contacts: list[str] = Field(default_factory=list)
    departments: list[str] = Field(default_factory=list)


class ClassifyResult(BaseModel):
    # Allow ``usage`` to be assigned after construction by LLMClassifier —
    # the rule-based fallback in Classifier doesn't populate it.
    model_config = ConfigDict(validate_assignment=False)

    category: ContentCategory
    confidence: float
    sub_categories: list[str] = Field(default_factory=list)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    summary: str = ""
    usage: StageUsage | None = Field(
        None,
        description="Per-call usage record from the LLM classifier (None when the rule-based fallback ran instead).",
    )


class ChunkResult(BaseModel):
    chunks: list[str]
    total_chunks: int
    strategy: str
    avg_chunk_size: int
