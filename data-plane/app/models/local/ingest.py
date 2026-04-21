from pydantic import BaseModel, Field

from app.models.common import ACL


class LocalChunkingConfig(BaseModel):
    """Configuration for text chunking during ingestion."""

    strategy: str = Field("late_chunking", description="Chunking strategy: 'late_chunking' (paragraph-aware, default), 'sentence' (sentence boundaries), or 'fixed' (character count)")
    max_chunk_size: int = Field(512, ge=64, le=4096, description="Maximum chunk size in characters")
    overlap: int = Field(50, ge=0, le=512, description="Overlap between consecutive chunks in characters")


class LocalIngestMetadata(BaseModel):
    """Additional metadata attached to every vector in Qdrant."""

    title: str | None = Field(None, description="Document title (shown in search results)")
    uploaded_by: str | None = Field(None, description="User or service that uploaded the document")
    source_type: str | None = Field(None, description="Origin: 'smb' or 'r2'")
    mime_type: str | None = Field(None, description="Original file MIME type")
    municipality_id: str | None = Field(None, description="Municipality/tenant identifier")
    department: str | None = Field(None, description="Department within the organization (e.g. 'bauamt', 'umwelt')")
    last_modified: str | None = Field(None, description="Last modification date/time of the source content (e.g. ISO 8601 format). Stored in Qdrant point metadata for filtering.")


class LocalIngestRequest(BaseModel):
    """Request to ingest locally parsed content into the vector database.

    Takes parsed text + ACL and runs:
    chunks -> classifies -> embeds (BGE-M3) -> stores in Qdrant.

    Every document MUST have an ACL. Existing vectors for the same source_id
    are automatically replaced (upsert).
    """

    collection_name: str = Field(..., description="Qdrant collection name to store vectors in")
    source_id: str = Field(..., description="Unique document ID. Used for updates and deletes.")
    file_path: str = Field(..., description="Original file path (stored as metadata)")
    content: str = Field(..., min_length=1, description="Parsed text content (from /local/document-parse)")
    language: str | None = Field(None, description="ISO 639-1 code. Auto-detected from content if omitted.")
    acl: ACL = Field(..., description="Access control list. Every document must have visibility set.")
    metadata: LocalIngestMetadata = Field(..., description="Document metadata stored alongside vectors")
    chunking: LocalChunkingConfig | None = Field(None, description="Override default chunking settings")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_name": "wiener-neudorf",
                    "source_id": "doc_abc123",
                    "file_path": "//server/bauamt/bauantraege/antrag_001.pdf",
                    "content": "Bauantrag Nr. 2024-001\nAntragsteller: Max Mustermann\n\nDer Antrag auf Errichtung eines Einfamilienhauses...",
                    "language": "de",
                    "acl": {
                        "allow_groups": ["DOMAIN\\Bauamt-Mitarbeiter"],
                        "deny_groups": ["DOMAIN\\Praktikanten"],
                        "allow_roles": [],
                        "allow_users": [],
                        "department": "bauamt",
                        "visibility": "internal",
                    },
                    "metadata": {
                        "title": "Bauantrag 2024-001",
                        "uploaded_by": "moderator_01",
                        "source_type": "smb",
                        "mime_type": "application/pdf",
                        "municipality_id": "wiener-neudorf",
                        "department": "bauamt",
                    },
                    "chunking": {
                        "strategy": "late_chunking",
                        "max_chunk_size": 512,
                        "overlap": 50,
                    },
                }
            ]
        }
    }


class LocalEntityCounts(BaseModel):
    """Count of entities extracted during classification."""

    dates: int = Field(0, description="Number of dates found")
    contacts: int = Field(0, description="Number of email addresses found")
    amounts: int = Field(0, description="Number of monetary amounts found")


class LocalIngestData(BaseModel):
    """Result of the local ingest pipeline."""

    source_id: str = Field(..., description="Document ID that was ingested")
    chunks_created: int = Field(..., description="Number of text chunks created")
    vectors_stored: int = Field(..., description="Number of vectors stored in Qdrant")
    collection: str = Field(..., description="Qdrant collection name")
    content_type: list[str] = Field(..., description="Auto-detected content categories (e.g. ['funding', 'renewable_energy'])")
    entities_extracted: LocalEntityCounts = Field(..., description="Entity extraction counts")
    embedding_time_ms: int = Field(..., description="Time spent on BGE-M3 embedding (ms)")
    total_time_ms: int = Field(..., description="Total pipeline duration (ms)")
