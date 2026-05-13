from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.classify import ExtractedEntities


class SearchMode(str, Enum):
    """Vector search strategy for Qdrant storage."""

    semantic = "semantic"
    hybrid = "hybrid"


class EmbeddingModel(str, Enum):
    """Which primary embedder to use for this ingest.

    - ``openai`` — OpenAI ``text-embedding-3-small`` (1536-dim). Stored as
      ``dense_openai`` in Qdrant.
    - ``bge_m3`` — BGE-M3 served behind the TEI OpenAI-compatible endpoint
      at ``TEI_EMBED_URL_AT`` (1024-dim). Stored as ``dense_bge_m3``.
    """

    openai = "openai"
    bge_m3 = "bge_m3"


class OnlineChunkingConfig(BaseModel):
    """Configuration for text chunking during ingestion."""

    strategy: str = Field("contextual", description="Chunking strategy: 'contextual' (recursive splitter + AI context prepended, default), 'recursive' (recursive character text splitter), 'late_chunking' (paragraph-aware), 'sentence' (sentence boundaries), or 'fixed' (character count)")
    max_chunk_size: int = Field(1200, le=4096, description="Maximum chunk size in characters. Values below 1000 are silently clamped to 1000 (the per-chunk minimum produces too many tiny chunks otherwise and inflates the contextual-enrichment bill); the upper bound is whatever the caller provides up to 4096.")
    overlap: int = Field(50, ge=0, le=512, description="Overlap between consecutive chunks in characters")

    @field_validator("max_chunk_size", mode="after")
    @classmethod
    def _enforce_min_chunk_size(cls, v: int) -> int:
        return max(v, 1000)


class OnlineVectorConfig(BaseModel):
    """Configuration for vector storage in Qdrant."""

    vector_size: int | None = Field(None, ge=64, le=4096, description="Dimensionality of the dense embedding vector. When omitted, derived from `embedding_model` (1536 for openai, 1024 for bge_m3).")
    search_mode: SearchMode = Field(SearchMode.semantic, description="'semantic' — dense cosine vector only. 'hybrid' — dense + sparse vector from the TEI sparse endpoint at `SPARSE_EMBED_URL_AT` (sparse.ki2.at) for combined semantic + lexical search.")


class OnlineIngestMetadata(BaseModel):
    """Additional metadata attached to every vector in Qdrant."""

    assistant_id: str | None = Field(None, description="Identifier of the assistant that owns this content. At least one of assistant_id or municipality_id must be provided.")
    title: str | None = Field(None, description="Document/page title (shown in search results)")
    uploaded_by: str | None = Field(None, description="User or service that triggered the ingestion")
    source_type: str | None = Field("web", description="Origin: typically 'web' for online content")
    mime_type: str | None = Field(None, description="Original content MIME type")
    municipality_id: str | None = Field(None, description="Municipality/tenant identifier. At least one of assistant_id or municipality_id must be provided.")
    department: list[str] = Field(default_factory=list, description="Departments within the organization")
    last_modified: str | None = Field(None, description="Last modification date/time of the source content (e.g. ISO 8601 format). Stored in Qdrant point metadata for filtering.")

    @model_validator(mode="after")
    def check_at_least_one_id(self) -> "OnlineIngestMetadata":
        if not self.assistant_id and not self.municipality_id:
            raise ValueError("At least one of 'assistant_id' or 'municipality_id' must be provided")
        return self


class OnlineIngestRequest(BaseModel):
    """Request to ingest web-scraped content into the vector database.

    Takes scraped/parsed text and runs:
    chunks -> classifies -> embeds -> stores in Qdrant.

    One dense vector per point, named after the chosen ``embedding_model``:
    ``dense_openai`` (1536) or ``dense_bge_m3`` (1024). A collection is
    pinned to the model it was first ingested with.

    Existing vectors for the same source_id are automatically replaced (upsert).

    **Vector modes:**
    - `semantic` (default) — dense cosine vector only.
    - `hybrid` — dense + ``sparse`` vector from the TEI sparse endpoint at
      ``SPARSE_EMBED_URL_AT`` for combined semantic + lexical search.
    """

    collection_name: str = Field(..., description="Qdrant collection name to store vectors in")
    source_id: str = Field(..., description="Unique document ID. Used for updates and deletes.")
    url: str = Field(..., description="Source URL (stored as source_url in Qdrant point metadata)")
    content: str = Field(..., min_length=1, description="Parsed/scraped text content (from /online/scrape or /online/document-parse)")
    content_type: list[str] = Field(..., min_length=1, description="Content categories for this document, e.g. ['funding', 'renewable_energy']. Must be obtained upfront from /online/scrape or /online/document-parse (which now return content_type) — classification is no longer performed at ingest time.")
    entities: ExtractedEntities | None = Field(None, description="Optional structured entities (dates, deadlines, amounts, contacts, departments) obtained from /online/scrape or /online/document-parse. When supplied, these are stored as entity_* fields in each Qdrant point's metadata for filtering. Pass null or omit if you do not want entity data stored.")
    language: str | None = Field(None, description="ISO 639-1 code. Auto-detected from content if omitted.")
    assistant_type: str | None = Field(None, description="Type of assistant processing this content (e.g. 'municipal', 'internal', 'public'). Stored in Qdrant point metadata for filtering during search.")
    country: str | None = Field(None, description="ISO 3166-1 alpha-2 country code (e.g. 'AT', 'DE', 'RO'). Required when assistant_type is 'funding'. Used by the funding extractor to constrain state_or_province to the official list for that country, preventing hallucinated region names.")
    state_or_province: list[str] | None = Field(None, description="Optional override for the funding `state_or_province` metadata field. When provided, values pass through the same per-country alias map the extractor uses — local-language names (e.g. 'Bayern', 'Praha', 'Wien') are canonicalized to English lowercase ('bavaria', 'prague', 'vienna'), and values not in the official list for the supplied `country` are dropped. Supported alias maps: AT, DE, CH, RO, IT, FR, HU, CZ, SK, SI, HR. For countries outside that list, values are lowercased and stripped but not validated. When omitted, the funding extractor detects and normalizes the value automatically.")
    embedding_model: EmbeddingModel = Field(
        EmbeddingModel.bge_m3,
        description=(
            "Primary embedder for this ingest. `bge_m3` (default) uses "
            "BGE-M3 via the TEI endpoint at `TEI_EMBED_URL_AT` (1024-dim, "
            "stored as `dense_bge_m3`). `openai` uses `text-embedding-3-small` "
            "(1536-dim, stored as `dense_openai`). A collection is tied to "
            "the model it was first ingested with — mixing models in one "
            "collection is not supported."
        ),
    )
    metadata: OnlineIngestMetadata = Field(..., description="Document metadata stored alongside vectors")

    @model_validator(mode="after")
    def check_country_for_funding(self) -> "OnlineIngestRequest":
        if self.assistant_type == "funding" and not self.country:
            raise ValueError("'country' is required when assistant_type is 'funding'")
        return self
    chunking: OnlineChunkingConfig | None = Field(None, description="Override default chunking settings")
    vector_config: OnlineVectorConfig | None = Field(None, description="Override default vector storage settings (size, search mode)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_name": "wiener-neudorf",
                    "source_id": "web_foerderungen_001",
                    "url": "https://www.wiener-neudorf.gv.at/foerderungen",
                    "content": "Förderungen der Gemeinde Wiener Neudorf\n\nDie Gemeinde bietet verschiedene Förderungen...",
                    "content_type": ["funding", "renewable_energy"],
                    "language": "de",
                    "metadata": {
                        "assistant_id": "asst_wiener_neudorf_01",
                        "title": "Förderungen - Gemeinde Wiener Neudorf",
                        "source_type": "web",
                        "municipality_id": "wiener-neudorf",
                        "department": ["Bürgerservice", "Förderungen"],
                    },
                    "vector_config": {
                        "search_mode": "semantic",
                    },
                }
            ]
        }
    }


class OnlineIngestData(BaseModel):
    """Result of the online ingest pipeline."""

    source_id: str = Field(..., description="Document ID that was ingested")
    chunks_created: int = Field(..., description="Number of text chunks created")
    vectors_stored: int = Field(..., description="Number of vectors stored in Qdrant")
    collection: str = Field(..., description="Qdrant collection name")
    content_type: list[str] = Field(..., description="Content categories stored with the vectors (passed through from the request body)")
    embedding_time_ms: int = Field(..., description="Time spent on embedding (ms)")
    total_time_ms: int = Field(..., description="Total pipeline duration (ms)")


class BatchIngestRequest(BaseModel):
    """Batch wrapper around `OnlineIngestRequest` — same per-item shape as
    `POST /api/v1/online/ingest`, just N at a time."""

    items: list[OnlineIngestRequest] = Field(
        ...,
        description=(
            "List of ingest requests. Each item is identical in shape to the "
            "body of `POST /api/v1/online/ingest`. Capped by the "
            "`DP_MAX_BATCH_INGEST_ITEMS` env var (default 50). Empty list "
            "returns `VALIDATION_BATCH_EMPTY`."
        ),
    )


class BatchIngestItemResult(BaseModel):
    """Per-item outcome inside a batch ingest response."""

    source_id: str = Field(..., description="`source_id` of the item this result corresponds to")
    success: bool = Field(..., description="Whether this item ingested successfully")
    data: OnlineIngestData | None = Field(None, description="Ingest result; null when `success=false`")
    error: str | None = Field(None, description="Error code when `success=false`; null otherwise")
    detail: str | None = Field(None, description="Human-readable error detail when `success=false`")


class BatchIngestData(BaseModel):
    """Aggregate result of a batch ingest call."""

    total: int = Field(..., description="Total items submitted")
    succeeded: int = Field(..., description="Items that completed without error")
    failed: int = Field(..., description="Items that returned an error (still present in `results`)")
    results: list[BatchIngestItemResult] = Field(..., description="Per-item outcomes, same order as the request")
    total_time_ms: int = Field(..., description="Wall-clock duration of the entire batch")
