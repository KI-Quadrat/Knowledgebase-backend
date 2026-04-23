from pydantic import BaseModel, Field


class OnlineMetadataFilter(BaseModel):
    """A single metadata filter condition."""

    key: str = Field(..., description="Metadata field name (e.g. 'source_type', 'content_type', 'assistant_id', 'municipality_id')")
    value: str = Field(..., description="Exact value to match")


class OnlineDeleteByFilterRequest(BaseModel):
    """Request to delete vectors matching metadata filters.

    All filters are combined with AND logic — only points matching every
    condition are deleted.
    """

    collection_name: str = Field(..., description="Qdrant collection name")
    filters: list[OnlineMetadataFilter] = Field(..., min_length=1, description="Metadata conditions (AND logic). At least one filter is required.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_name": "wiener-neudorf",
                    "filters": [
                        {"key": "assistant_id", "value": "asst_wiener_neudorf_01"},
                        {"key": "content_type", "value": "funding"},
                    ],
                },
            ]
        }
    }


class OnlineDeleteByFilterData(BaseModel):
    """Result of deleting vectors by metadata filter."""

    vectors_deleted: int = Field(..., description="Number of vectors removed from Qdrant")
    filters_applied: list[OnlineMetadataFilter] = Field(..., description="Filters that were used")


class OnlineDeleteVectorsData(BaseModel):
    """Result of deleting all vectors for a document."""

    source_id: str = Field(..., description="Document ID whose vectors were deleted")
    vectors_deleted: int = Field(..., description="Number of vectors removed from Qdrant")


class OnlineDeleteVectorsATData(BaseModel):
    """Result of deleting all vectors for a document on the AT Qdrant instance."""

    source_id: str = Field(..., description="Document ID whose vectors were deleted")
    vectors_deleted: int = Field(..., description="Number of vectors removed from Qdrant")


class OnlineSparseEncodeRequest(BaseModel):
    """Encode arbitrary text into a BM25 sparse vector."""

    content: str = Field(..., description="Text to encode")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"content": "Förderungen der Gemeinde Wiener Neudorf für Photovoltaik."}
            ]
        }
    }


class OnlineSparseEncodeData(BaseModel):
    """Qdrant-compatible sparse vector for the supplied content.

    The encoder is the same BM25 encoder used during ``POST /online/ingest``
    in ``hybrid`` mode and during hybrid search query encoding — so the
    indices/values returned here align with what is stored in Qdrant.
    """

    indices: list[int] = Field(..., description="Sparse vector indices (sorted ascending)")
    values: list[float] = Field(..., description="Sparse vector values (term frequencies, parallel to indices)")
    term_count: int = Field(..., description="Number of unique terms in the encoded vector")
