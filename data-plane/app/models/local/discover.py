from pydantic import BaseModel, Field

from app.models.common import NtfsACL


class DiscoverRequest(BaseModel):
    """Request to scan file sources for new, changed, or unchanged documents.

    This is the first step in every local ingestion pipeline. It scans SMB shares or
    R2 buckets, reads NTFS permissions, computes SHA-256 hashes, and returns
    what changed since the last scan. Does NOT parse or embed.
    """

    source: str = Field(..., description="Storage source: 'smb' (file share) or 'r2' (Cloudflare R2)", pattern=r"^(smb|r2)$")
    paths: list[str] = Field(..., min_length=1, description="SMB share paths or R2 key prefixes to scan")
    since_hash_map: dict[str, str] = Field(
        default_factory=dict,
        description="Map of file_path to last known SHA-256 hash. Files with matching hashes are marked as 'unchanged'.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "source": "smb",
                    "paths": ["//server/abteilung/dokumente", "//server/bauamt"],
                    "since_hash_map": {"//server/abteilung/dokumente/antrag.pdf": "sha256:abc123def456..."},
                },
                {
                    "source": "r2",
                    "paths": ["tenant/example-municipality/uploads/"],
                    "since_hash_map": {},
                },
            ]
        }
    }


class FileInfo(BaseModel):
    """Metadata for a discovered file including hash, size, and NTFS permissions."""

    path: str = Field(..., description="Full file path (SMB) or object key (R2)")
    file_hash: str = Field(..., description="SHA-256 content hash (format: sha256:hexdigest)")
    size_bytes: int = Field(..., description="File size in bytes")
    mime_type: str = Field(..., description="Detected MIME type (e.g. application/pdf)")
    last_modified: str = Field(..., description="Last modification timestamp (ISO 8601)")
    status: str = Field(..., description="Change status: 'new', 'changed', or 'unchanged'")
    acl: NtfsACL | None = Field(None, description="NTFS/R2 permissions read from the file system")


class DiscoverData(BaseModel):
    """Summary of discovered files with change detection results."""

    total_files: int = Field(..., description="Total number of files found")
    new_files: int = Field(..., description="Files not present in since_hash_map")
    changed_files: int = Field(..., description="Files with a different hash than since_hash_map")
    unchanged_files: int = Field(..., description="Files with matching hash (skipped)")
    files: list[FileInfo] = Field(..., description="Detailed info for each discovered file")
