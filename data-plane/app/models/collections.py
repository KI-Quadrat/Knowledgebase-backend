from pydantic import BaseModel, Field


class VectorConfig(BaseModel):
    """Qdrant collection vector configuration."""

    dense_dim: int = Field(1024, description="Dense vector dimensions (1024 for BGE-M3)")
    sparse: bool = Field(True, description="Enable sparse vectors for hybrid search")
    distance: str = Field("cosine", description="Distance metric: cosine, euclid, or dot")


class InitCollectionRequest(BaseModel):
    """Request to create a Qdrant collection for a municipality.

    Called once during tenant setup. If the collection already exists,
    returns `created: false` without error.
    """

    collection_name: str = Field(..., description="Collection name (typically the tenant/municipality slug)")
    vector_config: VectorConfig | None = Field(None, description="Override default vector configuration (1024-dim dense + sparse)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_name": "example-municipality",
                    "vector_config": {"dense_dim": 1024, "sparse": True, "distance": "cosine"},
                }
            ]
        }
    }


class InitCollectionData(BaseModel):
    """Result of collection initialization."""

    collection: str = Field(..., description="Collection name")
    created: bool = Field(..., description="True if newly created, False if already existed")
    dense_dim: int = Field(..., description="Dense vector dimensions configured")
    sparse_enabled: bool = Field(..., description="Whether sparse vectors are enabled")


class CollectionStatsData(BaseModel):
    """Qdrant collection statistics."""

    collection: str = Field(..., description="Collection name")
    total_vectors: int = Field(..., description="Total number of vectors stored")
    total_documents: int = Field(..., description="Number of unique source documents")
    disk_usage_mb: float = Field(..., description="Disk space used in megabytes")
    by_classification: dict[str, int] = Field(..., description="Vector count by content category")
    by_visibility: dict[str, int] = Field(..., description="Vector count by visibility level")


class CollectionInfo(BaseModel):
    """Basic info about a single Qdrant collection."""

    name: str = Field(..., description="Collection name")
    vectors_count: int = Field(..., description="Total number of vectors stored")
    points_count: int = Field(..., description="Total number of points stored")
    segments_count: int = Field(..., description="Number of segments")
    disk_usage_mb: float = Field(..., description="Disk space used in megabytes")
    status: str = Field(..., description="Collection status (green, yellow, red, unknown)")


class AvailableCollectionsData(BaseModel):
    """List of available Qdrant collections."""

    total: int = Field(..., description="Total number of collections")
    collections: list[CollectionInfo] = Field(..., description="List of collections with info")
