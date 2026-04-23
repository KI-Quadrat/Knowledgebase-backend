"""Request/response models for ``POST /api/v1/online/ingest/at``.

The AT endpoint serves the Austrian funding assistant on a dedicated Qdrant
instance. The caller supplies ``collection_name`` directly — the country and
assistant type are still implicit (AT / funding). The target collection is
auto-created with a single unnamed 1024-dim cosine vector (no sparse, no
fallback) on first use to match the TEI embedding model behind
``TEI_EMBED_URL_AT``.
"""

from pydantic import BaseModel, Field

from app.models.classify import ExtractedEntities
from app.models.online.ingest import OnlineChunkingConfig, OnlineIngestMetadata


class OnlineIngestATRequest(BaseModel):
    """Request to ingest funding content into a single AT collection.

    The funding extractor runs unconditionally to enrich the stored metadata
    (provinces, contract contacts, program name, etc.). ``state_or_province``
    (below) overrides the extractor's choice for the stored metadata only —
    there is no per-province collection routing; the caller selects the target
    collection via ``collection_name``.
    """

    source_id: str = Field(..., description="Unique document ID. Prior points with the same source_id are deleted before the fresh upsert.")
    collection_name: str = Field(..., min_length=1, description="Target collection on the AT Qdrant instance. Auto-created with the AT legacy schema on first use; reused thereafter.")
    url: str = Field(..., description="Source URL (stored as metadata.source_url).")
    source_name: str | None = Field(None, description="Optional human-readable label for the source (e.g. 'Land Salzburg — Sportförderung'). Stored verbatim as metadata.source_name when supplied.")
    content: str = Field(..., min_length=1, description="Parsed/scraped text content from /online/scrape or /online/document-parse.")
    content_type: list[str] = Field(..., min_length=1, description="Content categories, e.g. ['funding','sport']. Obtained upstream from /online/scrape or /online/document-parse.")
    entities: ExtractedEntities | None = Field(None, description="Optional structured entities (dates, deadlines, amounts, contacts, departments) from the upstream scrape/parse call.")
    language: str | None = Field(None, description="ISO 639-1 code. Defaults to 'de' when omitted.")
    state_or_province: list[str] | None = Field(
        None,
        description=(
            "Optional explicit province override stored on every point as "
            "`metadata.state_or_province`. English lowercase (e.g. "
            "'lower austria', 'vienna'). When omitted, the funding extractor's "
            "output is used."
        ),
    )
    metadata: OnlineIngestMetadata = Field(..., description="Document metadata stored alongside vectors.")
    chunking: OnlineChunkingConfig | None = Field(None, description="Override default chunking settings.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source_id": "web_salzburg_sport_001",
                    "collection_name": "foerder_at",
                    "url": "https://www.salzburg.gv.at/sport-foerderung",
                    "source_name": "Land Salzburg — Sportförderung",
                    "content": "Sportförderung Salzburg\n\n...",
                    "content_type": ["funding", "sport"],
                    "language": "de",
                    "metadata": {
                        "assistant_id": "asst_foerder_at_01",
                        "municipality_id": "land-salzburg",
                        "title": "Sportförderung Salzburg",
                        "source_type": "web",
                    },
                }
            ]
        }
    }


class OnlineIngestATData(BaseModel):
    """Result of the AT single-collection ingest."""

    source_id: str = Field(..., description="Document ID that was ingested.")
    chunks_created: int = Field(..., description="Number of text chunks produced from the content.")
    vectors_stored: int = Field(..., description="Vectors stored in the target collection.")
    collection_name: str = Field(..., description="Collection the ingest was written to.")
    content_type: list[str] = Field(..., description="Content categories passed through from the request.")
    embedding_time_ms: int = Field(..., description="Time spent on embedding (ms).")
    total_time_ms: int = Field(..., description="Total pipeline duration (ms).")
