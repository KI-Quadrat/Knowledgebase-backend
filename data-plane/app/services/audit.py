import asyncio
import functools
from collections.abc import Iterable
from urllib.parse import urlparse

from app.config import ext
from app.models.common import StageUsage
from app.utils.logger import get_logger

log = get_logger(__name__)


class AuditLogger:
    """ClickHouse audit logger for Data Plane events."""

    def __init__(self) -> None:
        self._client = None

    async def start(self) -> None:
        try:
            from clickhouse_driver import Client

            loop = asyncio.get_running_loop()
            self._client = await loop.run_in_executor(
                None,
                functools.partial(
                    Client,
                    host=ext.clickhouse_host,
                    port=ext.clickhouse_port,
                    database=ext.clickhouse_db,
                    user=ext.clickhouse_user,
                    password=ext.clickhouse_password,
                ),
            )
            log.info("audit_logger_started", host=ext.clickhouse_host)
            ok = await self.check_health()
            if not ok and ext.clickhouse_required:
                raise RuntimeError("ClickHouse health check failed")
        except Exception as exc:
            log.warning("audit_logger_init_failed", error=str(exc))
            self._client = None
            if ext.clickhouse_required:
                raise

    async def close(self) -> None:
        if self._client:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._client.disconnect)
            except Exception:
                pass
            self._client = None

    async def check_health(self) -> bool:
        if not self._client:
            return False
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._client.execute, "SELECT 1")
            return bool(result)
        except Exception:
            return False

    async def log(
        self,
        action: str,
        actor: str,
        url: str,
        status: str = "success",
        request_id: str = "",
        api_key_hash: str = "",
        **details: int | str,
    ) -> None:
        if not self._client:
            log.debug("audit_skipped_no_client", action=action, url=url)
            return

        row = {
            "action": action,
            "actor": actor,
            "url": url,
            "domain": urlparse(url).netloc,
            "status": status,
            "documents_found": int(details.get("documents_found", 0)),
            "word_count": int(details.get("word_count", 0)),
            "duration_ms": int(details.get("duration_ms", 0)),
            "error": str(details.get("error", "")),
            "request_id": request_id,
            "api_key_hash": api_key_hash,
        }

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._insert, row)
        except Exception as exc:
            log.warning("audit_log_failed", action=action, url=url, error=str(exc))

    def _insert(self, row: dict) -> None:
        self._client.execute(  # type: ignore[union-attr]
            "INSERT INTO audit_log "
            "(action, actor, url, domain, status, documents_found, "
            "word_count, duration_ms, error, request_id, api_key_hash) VALUES",
            [row],
        )

    async def log_usage(
        self,
        entries: Iterable[StageUsage],
        *,
        endpoint: str,
        request_id: str = "",
        api_key_hash: str = "",
        url: str = "",
        municipality_id: str = "",
        assistant_id: str = "",
        assistant_type: str = "",
        status: str = "success",
    ) -> None:
        """Write per-stage usage rows to ClickHouse ``usage_log``.

        One row per StageUsage so a single ingest call produces ~4 rows
        (classifier + contextual + funding + embedding). Failures are
        swallowed — the audit sink is best-effort and a ClickHouse outage
        must not break the request.

        Schema is documented in ``USAGE_LOG_DDL`` below; the table is
        created out-of-band (migration in ``scripts/`` or via the ClickHouse
        operator) before this code runs.
        """
        if not self._client:
            return
        rows = [_usage_row(
            entry,
            endpoint=endpoint,
            request_id=request_id,
            api_key_hash=api_key_hash,
            url=url,
            municipality_id=municipality_id,
            assistant_id=assistant_id,
            assistant_type=assistant_type,
            status=status,
        ) for entry in entries if entry is not None]
        if not rows:
            return
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._insert_usage, rows)
        except Exception as exc:
            log.warning("usage_log_failed", endpoint=endpoint, error=str(exc))

    def _insert_usage(self, rows: list[dict]) -> None:
        self._client.execute(  # type: ignore[union-attr]
            "INSERT INTO usage_log "
            "(endpoint, stage, provider, model, url, domain, "
            "prompt_tokens, completion_tokens, cached_tokens, "
            "embed_tokens, scrape_tokens, credits, pages, cost_usd, "
            "municipality_id, assistant_id, assistant_type, "
            "request_id, api_key_hash, status) VALUES",
            rows,
        )


def _usage_row(
    entry: StageUsage,
    *,
    endpoint: str,
    request_id: str,
    api_key_hash: str,
    url: str,
    municipality_id: str,
    assistant_id: str,
    assistant_type: str,
    status: str,
) -> dict:
    domain = urlparse(url).netloc if url else ""
    return {
        "endpoint": endpoint,
        "stage": entry.stage,
        "provider": entry.provider,
        "model": entry.model or "",
        "url": url,
        "domain": domain,
        "prompt_tokens": int(entry.prompt_tokens or 0),
        "completion_tokens": int(entry.completion_tokens or 0),
        "cached_tokens": int(entry.cached_tokens or 0),
        "embed_tokens": int(entry.embed_tokens or 0),
        "scrape_tokens": int(entry.scrape_tokens or 0),
        "credits": float(entry.credits or 0.0),
        "pages": int(entry.pages or 0),
        # ``cost_usd`` is Nullable in CH — None preserves the "rate unknown"
        # signal we use to detect a missing entry in pricing.yaml.
        "cost_usd": entry.cost_usd,
        "municipality_id": municipality_id or "",
        "assistant_id": assistant_id or "",
        "assistant_type": assistant_type or "",
        "request_id": request_id or "",
        "api_key_hash": api_key_hash or "",
        "status": status,
    }


# DDL for the ``usage_log`` table. Apply via:
#   clickhouse-client --multiquery --query="$USAGE_LOG_DDL"
# or your migration tool of choice. Kept here so the schema lives next to
# the writer that depends on it.
USAGE_LOG_DDL = """
CREATE TABLE IF NOT EXISTS usage_log (
    event_time          DateTime DEFAULT now(),
    endpoint            LowCardinality(String),
    stage               LowCardinality(String),
    provider            LowCardinality(String),
    model               LowCardinality(String),
    url                 String,
    domain              String,
    prompt_tokens       UInt32,
    completion_tokens   UInt32,
    cached_tokens       UInt32,
    embed_tokens        UInt32,
    scrape_tokens       UInt32,
    credits             Float64,
    pages               UInt32,
    cost_usd            Nullable(Float64),
    municipality_id     String,
    assistant_id        String,
    assistant_type      String,
    request_id          String,
    api_key_hash        String,
    status              LowCardinality(String)
) ENGINE = MergeTree
PARTITION BY toYYYYMM(event_time)
ORDER BY (event_time, endpoint, provider, model);
"""
