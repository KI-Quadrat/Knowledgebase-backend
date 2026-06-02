from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field, model_serializer

T = TypeVar("T")

# Count fields on StageUsage that should be omitted when zero. These are
# the per-provider billing units — keeping them around as zeros on stages
# that don't use them just bloats the response.
_STAGE_COUNT_FIELDS: tuple[str, ...] = (
    "prompt_tokens", "completion_tokens", "cached_tokens",
    "embed_tokens", "scrape_tokens", "credits", "pages",
)


class StageUsage(BaseModel):
    """Usage record for one external call within a request.

    A request may produce multiple ``StageUsage`` entries — e.g. an
    ``/online/ingest`` call records one per stage (classifier, contextual,
    funding, embedding). Counts are populated based on the provider's
    billing unit; unrelated fields are zero/None.

    ``cost_usd`` is computed from ``pricing.yaml`` (see ``services/cost.py``):
    - ``0.0`` when the provider is self-hosted (absent from the YAML).
    - ``None`` when the provider is listed but the rate is unset (plan-
      dependent) — the raw count is still recorded so accounting works.
    - Otherwise the computed dollar amount.
    """

    stage: str = Field(..., description="Pipeline stage: 'scraper', 'classifier', 'contextual', 'funding', 'embedding', 'inner_docs', 'inner_img', 'links_map'")
    provider: str = Field(..., description="Billing provider: 'jina', 'firecrawl', 'httpx', 'openai', 'bge_m3', 'tei_sparse', 'llamaparse', or any OpenAI-compatible provider name from llm_router")
    model: str | None = Field(None, description="Model identifier (e.g. 'gpt-4o-mini', 'text-embedding-3-small'). Null for non-LLM providers.")
    prompt_tokens: int = Field(0, description="Chat prompt tokens — OpenAI chat calls.")
    completion_tokens: int = Field(0, description="Chat completion tokens — OpenAI chat calls.")
    cached_tokens: int = Field(0, description="Cached input tokens from OpenAI auto-cache (subset of prompt_tokens billed at the discounted rate).")
    embed_tokens: int = Field(0, description="Embedding input tokens — OpenAI /v1/embeddings calls.")
    scrape_tokens: int = Field(0, description="Tokens reported by token-metered scrapers (Jina's meta.usage.tokens).")
    credits: float = Field(0.0, description="Credits consumed by credit-billed providers (Firecrawl).")
    pages: int = Field(0, description="Pages processed by per-page providers (LlamaParse OCR).")
    cost_usd: float | None = Field(None, description="Computed USD cost from pricing.yaml. 0.0 for self-hosted. Null when the rate is plan-dependent and not set — the raw count above still records what happened.")

    @model_serializer(mode="wrap")
    def _slim_dump(self, handler):
        """Drop zero-valued count fields when serializing.

        Self-hosted stages (BGE-M3, TEI sparse, raw httpx) have every count
        field at zero — rendering them all bloats responses without adding
        information. The Python object keeps the full field set so summing
        / aggregation still works; only the JSON / dict output is slim.

        ``stage``, ``provider``, ``model``, and ``cost_usd`` are always
        included (the slim format guarantees those four fields per entry).
        """
        data = handler(self)
        if data.get("model") is None:
            data.pop("model", None)
        for key in _STAGE_COUNT_FIELDS:
            if data.get(key) in (0, 0.0):
                data.pop(key, None)
        return data


class UsageSummary(BaseModel):
    """Aggregated usage for one endpoint response.

    ``by_stage`` maps stage name → entry; ``total_tokens`` and
    ``total_cost_usd`` are convenience roll-ups across all entries.
    ``total_cost_usd`` is null when any contributing entry's cost is null
    (we don't silently treat unknown rates as zero in the rollup).
    """

    total_tokens: int = Field(0, description="Sum of all token counters across stages (prompt + completion + embed + scrape).")
    total_credits: float = Field(0.0, description="Sum of credit-billed provider usage across stages.")
    total_pages: int = Field(0, description="Sum of per-page provider usage across stages.")
    total_cost_usd: float | None = Field(None, description="Sum of cost_usd across stages, or null if any contributing stage cost is unknown.")
    by_stage: dict[str, StageUsage] = Field(default_factory=dict, description="Per-stage usage entries keyed by stage name. Multiple stages can share a provider (e.g. classifier + contextual both call openai). Self-hosted stages with cost_usd=0 and no counts are omitted from the serialized response — they're recorded in ClickHouse usage_log for full audit but aren't useful on the response itself.")

    @model_serializer(mode="wrap")
    def _slim_dump(self, handler):
        """Drop self-hosted zero-cost entries and zero-valued totals.

        - ``by_stage`` entries are kept only when they billed something:
          ``cost_usd != 0`` OR at least one count field is non-zero.
          ``cost_usd is None`` (rate unset) also counts as "kept" — those
          stages bill something we don't know how to price yet.
        - ``total_credits`` / ``total_pages`` are dropped when zero so
          responses for token-only paths stay tight.
        - ``total_tokens`` and ``total_cost_usd`` are always included so
          consumers can rely on the rollup fields.

        ClickHouse ``usage_log`` writes use the raw ``StageUsage`` list
        directly (not this serialization), so the audit trail keeps every
        stage even after slim filtering trims the response.
        """
        data = handler(self)

        slim_by_stage: dict = {}
        for stage_name, entry in (data.get("by_stage") or {}).items():
            cost = entry.get("cost_usd")
            has_counts = any(entry.get(k) for k in _STAGE_COUNT_FIELDS)
            # Drop only self-hosted ($0 with zero counts). Keep cost==None
            # (rate unknown) and any entry with real usage counts.
            if cost == 0 and not has_counts:
                continue
            slim_by_stage[stage_name] = entry
        data["by_stage"] = slim_by_stage

        if data.get("total_credits") in (0, 0.0):
            data.pop("total_credits", None)
        if data.get("total_pages") in (0, 0.0):
            data.pop("total_pages", None)
        return data

    @classmethod
    def from_entries(cls, entries: list[StageUsage]) -> "UsageSummary":
        """Build a summary from a flat list of per-stage entries."""
        by_stage: dict[str, StageUsage] = {}
        total_tokens = 0
        total_credits = 0.0
        total_pages = 0
        total_cost: float | None = 0.0
        for e in entries:
            by_stage[e.stage] = e
            total_tokens += (
                e.prompt_tokens + e.completion_tokens
                + e.embed_tokens + e.scrape_tokens
            )
            total_credits += e.credits
            total_pages += e.pages
            if total_cost is None or e.cost_usd is None:
                total_cost = None
            else:
                total_cost += e.cost_usd
        return cls(
            total_tokens=total_tokens,
            total_credits=total_credits,
            total_pages=total_pages,
            total_cost_usd=round(total_cost, 6) if total_cost is not None else None,
            by_stage=by_stage,
        )

    @classmethod
    def merge(cls, summaries: list["UsageSummary | None"]) -> "UsageSummary":
        """Combine multiple summaries (e.g. per-item → batch totals).

        Entries with the same ``stage`` name are summed into one merged
        entry so a batch's ``by_stage`` shows "classifier across all items"
        instead of overwriting. Provider/model are preserved from the first
        entry seen; if items use different models for the same stage the
        first one wins (sum is still correct in token/cost terms).
        """
        merged: dict[str, StageUsage] = {}
        for s in summaries:
            if s is None:
                continue
            for stage, entry in s.by_stage.items():
                existing = merged.get(stage)
                if existing is None:
                    merged[stage] = entry.model_copy()
                    continue
                existing.prompt_tokens += entry.prompt_tokens
                existing.completion_tokens += entry.completion_tokens
                existing.cached_tokens += entry.cached_tokens
                existing.embed_tokens += entry.embed_tokens
                existing.scrape_tokens += entry.scrape_tokens
                existing.credits += entry.credits
                existing.pages += entry.pages
                if existing.cost_usd is None or entry.cost_usd is None:
                    existing.cost_usd = None
                else:
                    existing.cost_usd = round(existing.cost_usd + entry.cost_usd, 6)
        return cls.from_entries(list(merged.values()))


class ResponseEnvelope(BaseModel, Generic[T]):
    """Standard response wrapper for all Data Plane endpoints.

    Every API response is wrapped in this envelope. On success, `data` contains
    the result. On failure, `error` contains an error code and `detail` provides
    a human-readable message.
    """

    success: bool = Field(..., description="Whether the request succeeded")
    data: T | None = Field(None, description="Response payload (null on error)")
    error: str | None = Field(None, description="Error code from ErrorCode enum (null on success)")
    detail: str | None = Field(None, description="Human-readable error message (null on success)")
    request_id: str = Field(..., description="Unique request identifier for tracing")


class ACL(BaseModel):
    """Access control list attached to every document.

    Defines who can access a document based on Active Directory groups,
    portal roles, specific users, and visibility level.
    """

    allow_groups: list[str] = Field(default_factory=list, description="AD groups with access (e.g. DOMAIN\\\\Bauamt-Mitarbeiter)")
    deny_groups: list[str] = Field(default_factory=list, description="AD groups explicitly denied access")
    allow_roles: list[str] = Field(default_factory=list, description="Portal roles with access (e.g. member, admin)")
    allow_users: list[str] = Field(default_factory=list, description="Specific user IDs with access")
    department: str | None = Field(None, description="Department tag (e.g. bauamt, umwelt)")
    visibility: str = Field(
        ...,
        description="Access level: public (citizens), internal (employees), restricted (specific groups)",
        pattern=r"^(public|internal|restricted)$",
        json_schema_extra={"examples": ["public", "internal", "restricted"]},
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "allow_groups": ["DOMAIN\\Bauamt-Mitarbeiter", "DOMAIN\\Bauamt-Leitung"],
                    "deny_groups": ["DOMAIN\\Praktikanten"],
                    "allow_roles": [],
                    "allow_users": [],
                    "department": "bauamt",
                    "visibility": "internal",
                }
            ]
        }
    }


class NtfsACL(BaseModel):
    """NTFS permission info returned by file discovery.

    Represents the Windows NTFS ACL read from a file share, including
    inherited permissions from parent folders.
    """

    source: str = Field("ntfs", description="Permission source (ntfs or r2)")
    allow_groups: list[str] = Field(default_factory=list, description="AD groups with read access")
    deny_groups: list[str] = Field(default_factory=list, description="AD groups explicitly denied")
    allow_users: list[str] = Field(default_factory=list, description="Specific user accounts allowed")
    inherited: bool = Field(True, description="Whether permissions are inherited from parent folder")


class ErrorCode(str, Enum):
    # Validation
    VALIDATION_URL_INVALID = "VALIDATION_URL_INVALID"
    VALIDATION_PATH_OUTSIDE_ROOTS = "VALIDATION_PATH_OUTSIDE_ROOTS"
    VALIDATION_ACL_REQUIRED = "VALIDATION_ACL_REQUIRED"
    VALIDATION_EMPTY_CONTENT = "VALIDATION_EMPTY_CONTENT"
    VALIDATION_USER_REQUIRED = "VALIDATION_USER_REQUIRED"
    VALIDATION_BATCH_TOO_LARGE = "VALIDATION_BATCH_TOO_LARGE"
    VALIDATION_BATCH_EMPTY = "VALIDATION_BATCH_EMPTY"

    # Auth
    AUTH_MISSING = "AUTH_MISSING"
    AUTH_INVALID = "AUTH_INVALID"
    AUTH_EXPIRED = "AUTH_EXPIRED"

    # SMB
    SMB_CONNECTION_FAILED = "SMB_CONNECTION_FAILED"
    SMB_AUTH_FAILED = "SMB_AUTH_FAILED"
    SMB_PATH_NOT_FOUND = "SMB_PATH_NOT_FOUND"
    SMB_FILE_NOT_FOUND = "SMB_FILE_NOT_FOUND"
    SMB_FILE_LOCKED = "SMB_FILE_LOCKED"

    # R2
    R2_CONNECTION_FAILED = "R2_CONNECTION_FAILED"
    R2_FILE_NOT_FOUND = "R2_FILE_NOT_FOUND"
    R2_PRESIGNED_EXPIRED = "R2_PRESIGNED_EXPIRED"

    # LDAP
    LDAP_CONNECTION_FAILED = "LDAP_CONNECTION_FAILED"
    LDAP_AUTH_FAILED = "LDAP_AUTH_FAILED"

    # Parse
    PARSE_FAILED = "PARSE_FAILED"
    PARSE_ENCRYPTED = "PARSE_ENCRYPTED"
    PARSE_CORRUPTED = "PARSE_CORRUPTED"
    PARSE_EMPTY = "PARSE_EMPTY"
    PARSE_TIMEOUT = "PARSE_TIMEOUT"
    PARSE_UNSUPPORTED_FORMAT = "PARSE_UNSUPPORTED_FORMAT"

    # Scrape
    SCRAPE_FAILED = "SCRAPE_FAILED"
    SCRAPE_BLOCKED = "SCRAPE_BLOCKED"
    SCRAPE_TIMEOUT = "SCRAPE_TIMEOUT"
    SCRAPE_EMPTY = "SCRAPE_EMPTY"
    SCRAPE_ROBOTS_BLOCKED = "SCRAPE_ROBOTS_BLOCKED"

    # Crawl
    CRAWL_SITEMAP_NOT_FOUND = "CRAWL_SITEMAP_NOT_FOUND"
    CRAWL_MAX_URLS_EXCEEDED = "CRAWL_MAX_URLS_EXCEEDED"

    # Classify
    CONTENT_TYPE_MISMATCH = "CONTENT_TYPE_MISMATCH"
    CLASSIFY_FAILED = "CLASSIFY_FAILED"
    CLASSIFY_LOW_CONFIDENCE = "CLASSIFY_LOW_CONFIDENCE"
    ENTITY_EXTRACTION_FAILED = "ENTITY_EXTRACTION_FAILED"

    # Embedding
    EMBEDDING_MODEL_NOT_LOADED = "EMBEDDING_MODEL_NOT_LOADED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    EMBEDDING_OOM = "EMBEDDING_OOM"

    # Qdrant
    QDRANT_CONNECTION_FAILED = "QDRANT_CONNECTION_FAILED"
    QDRANT_COLLECTION_NOT_FOUND = "QDRANT_COLLECTION_NOT_FOUND"
    QDRANT_UPSERT_FAILED = "QDRANT_UPSERT_FAILED"
    QDRANT_SEARCH_FAILED = "QDRANT_SEARCH_FAILED"
    QDRANT_DELETE_FAILED = "QDRANT_DELETE_FAILED"
    QDRANT_DISK_FULL = "QDRANT_DISK_FULL"
