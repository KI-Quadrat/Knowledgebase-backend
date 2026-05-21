"""Create the ``usage_log`` table in ClickHouse.

Idempotent: re-running on an existing table is a no-op (the DDL uses
``CREATE TABLE IF NOT EXISTS``). Reads the same ClickHouse connection
settings the data-plane uses at runtime (``DP_CLICKHOUSE_*`` env vars
via ``ExternalSettings``).

Run from the data-plane root:

    uv run python scripts/migrate_usage_log.py
"""

from clickhouse_driver import Client

from app.config import ext
from app.services.audit import USAGE_LOG_DDL


def main() -> None:
    client = Client(
        host=ext.clickhouse_host,
        port=ext.clickhouse_port,
        database=ext.clickhouse_db,
        user=ext.clickhouse_user,
        password=ext.clickhouse_password,
    )
    client.execute(USAGE_LOG_DDL)
    print(f"usage_log table ensured on {ext.clickhouse_host}:{ext.clickhouse_port}/{ext.clickhouse_db}")


if __name__ == "__main__":
    main()
