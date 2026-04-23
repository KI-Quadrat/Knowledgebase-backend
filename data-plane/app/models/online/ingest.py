from enum import Enum

from pydantic import BaseModel, Field, model_validator

from app.models.classify import ExtractedEntities


class SearchMode(str, Enum):
    """Vector search strategy for Qdrant storage."""

    semantic = "semantic"
    hybrid = "hybrid"


class OnlineChunkingConfig(BaseModel):
    """Configuration for text chunking during ingestion."""

    strategy: str = Field("contextual", description="Chunking strategy: 'contextual' (recursive splitter + AI context prepended, default), 'recursive' (recursive character text splitter), 'late_chunking' (paragraph-aware), 'sentence' (sentence boundaries), or 'fixed' (character count)")
    max_chunk_size: int = Field(1200, ge=64, le=4096, description="Maximum chunk size in characters")
    overlap: int = Field(50, ge=0, le=512, description="Overlap between consecutive chunks in characters")


class OnlineVectorConfig(BaseModel):
    """Configuration for vector storage in Qdrant."""

    vector_size: int = Field(1536, ge=64, le=4096, description="Dimensionality of the OpenAI dense embedding vector. Default: 1536.")
    search_mode: SearchMode = Field(SearchMode.semantic, description="'semantic' — dense cosine vectors only. 'hybrid' — dense + sparse (BM25) vectors for combined semantic + lexical search.")
    enable_fallback: bool = Field(False, description="When true, stores an additional dense_bge_gemma2 vector via LiteLLM alongside dense_openai. Enables automatic fallback to BGE-Gemma2 during search when OpenAI is unavailable. The fallback vector dimension is configured server-side via BGE_GEMMA2_DENSE_DIM.")


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
    chunks -> classifies -> embeds (OpenAI, optionally + BGE-Gemma2) -> stores in Qdrant.

    When ``vector_config.enable_fallback`` is true, every point gets **multi-vector**
    embeddings: ``dense_openai`` (primary) and ``dense_bge_gemma2`` (fallback via
    LiteLLM). During search, OpenAI is tried first; if unavailable,
    ``dense_bge_gemma2`` is used automatically.

    When ``enable_fallback`` is false (default), only a single ``dense`` vector
    (OpenAI) is stored — same as the original behavior.

    Existing vectors for the same source_id are automatically replaced (upsert).

    **Vector modes:**
    - `semantic` (default) — stores dense cosine vectors only.
    - `hybrid` — stores dense + ``sparse`` (BM25) vectors for combined semantic + lexical search.
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
    state_or_province: list[str] | None = Field(None, description="Optional override for the funding `state_or_province` metadata field. When omitted, the funding extractor detects and normalizes the value automatically. When provided as a non-empty list, the values are stored in Qdrant verbatim (no LLM modification, no lowercasing, no validation) — overriding anything the extractor produced.")
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
                        "vector_size": 1536,
                        "search_mode": "semantic",
                        "enable_fallback": True,
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
