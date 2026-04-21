from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


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
