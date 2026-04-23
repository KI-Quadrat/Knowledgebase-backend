from pydantic import BaseModel, Field


class UserContext(BaseModel):
    """User identity for permission-filtered search.

    Every search requires a user context. Citizens only see 'public' documents.
    Employees see 'public' + 'internal' documents filtered by their AD groups.
    """

    type: str = Field(..., description="User type: 'citizen' (public access only) or 'employee' (group-based access)", pattern=r"^(citizen|employee)$")
    user_id: str = Field(..., description="User identifier (email, AD username, or 'anonymous')")
    groups: list[str] = Field(default_factory=list, description="Active Directory groups (required for employees)")
    roles: list[str] = Field(default_factory=list, description="Portal roles (e.g. member, admin)")
    department: str | None = Field(None, description="Department for result boosting/filtering")


class SearchFilters(BaseModel):
    """Optional filters to narrow search results."""

    content_type: list[str] | None = Field(None, description="Filter by content types (e.g. ['funding', 'policy'])")


class SearchRequest(BaseModel):
    """Semantic or hybrid search request with mandatory permission filtering.

    No search is ever unfiltered. The user context determines which documents
    are visible based on ACL visibility and group membership.

    **Search modes:**
    - `semantic` (default) — dense-only cosine search via OpenAI embeddings
    - `hybrid` — dense (OpenAI) + sparse (BM25) with Reciprocal Rank Fusion (RRF)

    Set ``enable_fallback: true`` if the collection was ingested with fallback enabled.
    This searches ``dense_openai`` and falls back to ``dense_bge_gemma2`` if OpenAI is down.
    """

    collection_name: str = Field(..., description="Qdrant collection name to search in")
    query: str = Field(..., min_length=1, description="Natural language search query")
    user: UserContext = Field(..., description="User identity for permission filtering (always required)")
    filters: SearchFilters | None = Field(None, description="Optional content filters")
    search_mode: str = Field("semantic", description="'semantic' (dense cosine only, default) or 'hybrid' (dense + BM25 sparse with RRF)")
    enable_fallback: bool = Field(False, description="Set to true if the collection was ingested with enable_fallback=true. Searches dense_openai first, falls back to dense_bge_gemma2 via LiteLLM when OpenAI is unavailable.")
    top_k: int = Field(10, ge=1, le=100, description="Maximum number of results to return")
    score_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score (0.0 = all, 1.0 = exact match). Used for semantic mode only.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_name": "wiener-neudorf",
                    "query": "Wann ist die nächste Förderung für Solaranlagen?",
                    "user": {
                        "type": "employee",
                        "user_id": "maria@wiener-neudorf.gv.at",
                        "groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Alle-Mitarbeiter"],
                        "roles": ["member"],
                        "department": "bauamt",
                    },
                    "filters": {"content_type": ["funding"]},
                    "search_mode": "semantic",
                    "top_k": 10,
                    "score_threshold": 0.5,
                },
                {
                    "collection_name": "wiener-neudorf",
                    "query": "Öffnungszeiten Gemeindeamt",
                    "user": {"type": "citizen", "user_id": "anonymous"},
                    "top_k": 5,
                    "score_threshold": 0.5,
                },
            ]
        }
    }


class SearchResultMetadata(BaseModel):
    """Metadata from the source document."""

    title: str | None = Field(None, description="Document title")
    municipality_id: str | None = Field(None, description="Municipality/tenant identifier")
    department: list[str] | None = Field(None, description="Source departments")
    source_type: str | None = Field(None, description="Origin: smb, r2, or web")


class SearchResultEntities(BaseModel):
    """Entities extracted from the matching chunk."""

    amounts: list[str] = Field(default_factory=list, description="Monetary amounts in this chunk")
    deadlines: list[str] = Field(default_factory=list, description="Deadlines mentioned in this chunk")


class SearchResult(BaseModel):
    """A single search result with chunk text, score, and metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier (format: source_id_chunk_NNNN)")
    source_id: str = Field(..., description="Parent document ID")
    chunk_text: str = Field(..., description="The matching text chunk")
    score: float = Field(..., description="Semantic similarity score (0.0 to 1.0)")
    source_path: str = Field(..., description="Original file path or URL")
    content_type: list[str] = Field(..., description="Content categories of the source document")
    entities: SearchResultEntities = Field(..., description="Entities found in this chunk")
    metadata: SearchResultMetadata = Field(..., description="Source document metadata")


class PermissionFilterApplied(BaseModel):
    """Transparency: shows exactly which permission filters were applied to the search."""

    visibility: list[str] = Field(..., description="Visibility levels included (e.g. ['public', 'internal'])")
    must_match_groups: list[str] = Field(..., description="User's AD groups used for filtering")
    must_not_match_groups: list[str] = Field(..., description="Groups in deny lists")


class SearchData(BaseModel):
    """Search results with timing and permission transparency."""

    results: list[SearchResult] = Field(..., description="Matching chunks ranked by similarity")
    total_results: int = Field(..., description="Number of results returned")
    query_embedding_ms: int = Field(..., description="Time to embed the query (ms)")
    search_ms: int = Field(..., description="Time to search Qdrant (ms)")
    search_mode: str = Field(..., description="Search mode used: 'semantic' or 'hybrid'")
    permission_filter_applied: PermissionFilterApplied = Field(..., description="Permission filters that were enforced")
