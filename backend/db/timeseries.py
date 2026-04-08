"""
FlowWatch AI — TimescaleDB Integration
=======================================
Async PostgreSQL connection pool using asyncpg.  Creates hypertables for
telemetry_records and anomaly_events on first startup.

Usage::

    from backend.db.timeseries import init_db, insert_telemetry, close_db

    await init_db(os.environ["DATABASE_URL"])
    await insert_telemetry(processed_record)
    await close_db()
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import asyncpg
from loguru import logger

# ─── Module-level pool (set by init_db) ──────────────────────────────────────

_pool: asyncpg.Pool | None = None

# ─── DDL ──────────────────────────────────────────────────────────────────────

_CREATE_TELEMETRY = """
CREATE TABLE IF NOT EXISTS telemetry_records (
    id              SERIAL,
    host_id         VARCHAR(20)  NOT NULL,
    timestamp       TIMESTAMPTZ  NOT NULL,
    latency_ms      FLOAT,
    packet_loss_pct FLOAT,
    dns_failure_rate FLOAT,
    jitter_ms       FLOAT,
    health_score    FLOAT,
    is_anomaly      BOOLEAN      DEFAULT FALSE,
    anomaly_score   FLOAT,
    severity        VARCHAR(20),
    created_at      TIMESTAMPTZ  DEFAULT NOW()
);
"""

_CREATE_ANOMALY = """
CREATE TABLE IF NOT EXISTS anomaly_events (
    id               SERIAL,
    host_id          VARCHAR(20)  NOT NULL,
    timestamp        TIMESTAMPTZ  NOT NULL,
    combined_score   FLOAT,
    severity         VARCHAR(20),
    worst_feature    VARCHAR(50),
    lstm_score       FLOAT,
    if_score         FLOAT,
    detection_method VARCHAR(20),
    created_at       TIMESTAMPTZ  DEFAULT NOW()
);
"""

_HYPERTABLE_TELEMETRY = """
SELECT create_hypertable('telemetry_records', 'timestamp',
    if_not_exists => TRUE, migrate_data => TRUE);
"""

_HYPERTABLE_ANOMALY = """
SELECT create_hypertable('anomaly_events', 'timestamp',
    if_not_exists => TRUE, migrate_data => TRUE);
"""


# ─── Public API ───────────────────────────────────────────────────────────────


async def init_db(database_url: str) -> asyncpg.Pool:
    """
    Create connection pool, create tables, and convert to hypertables.

    Returns the pool and also stores it at module level for the other helpers.
    Raises on connection failure so the caller can log and fall back.
    """
    global _pool

    _pool = await asyncpg.create_pool(
        dsn=database_url,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )

    async with _pool.acquire() as conn:
        await conn.execute(_CREATE_TELEMETRY)
        await conn.execute(_CREATE_ANOMALY)
        # TimescaleDB extension must be present; ignore if already a hypertable
        try:
            await conn.execute(_HYPERTABLE_TELEMETRY)
            await conn.execute(_HYPERTABLE_ANOMALY)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create hypertable (may already exist): {e}", e=exc)

    logger.info("TimescaleDB tables ready")
    return _pool


async def insert_telemetry(record: Any) -> None:
    """
    Insert a ProcessedRecord into telemetry_records.

    Args:
        record: A ProcessedRecord dataclass/object with the expected fields.
    """
    if _pool is None:
        return

    ts = record.timestamp
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO telemetry_records
                (host_id, timestamp, latency_ms, packet_loss_pct,
                 dns_failure_rate, jitter_ms, health_score)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            record.host_id,
            ts,
            record.latency_ms,
            record.packet_loss_pct,
            record.dns_failure_rate,
            record.jitter_ms,
            record.composite_health_score,
        )


async def insert_anomaly(host_id: str, result: Any) -> None:
    """
    Insert a CombinedAnomalyResult into anomaly_events.

    Args:
        host_id: The host that triggered this anomaly.
        result:  A CombinedAnomalyResult dataclass with the expected fields.
    """
    if _pool is None:
        return

    ts = result.timestamp
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    async with _pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO anomaly_events
                (host_id, timestamp, combined_score, severity,
                 worst_feature, lstm_score, if_score, detection_method)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            host_id,
            ts,
            result.combined_score,
            result.severity,
            result.worst_feature,
            result.lstm_result.anomaly_score,
            result.if_result.anomaly_score,
            result.detection_method,
        )


async def get_recent_telemetry(host_id: str, minutes: int = 5) -> list[dict]:
    """Return telemetry records for a host from the last N minutes."""
    if _pool is None:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT host_id, timestamp, latency_ms, packet_loss_pct,
                   dns_failure_rate, jitter_ms, health_score
            FROM telemetry_records
            WHERE host_id = $1
              AND timestamp >= NOW() - ($2 || ' minutes')::INTERVAL
            ORDER BY timestamp DESC
            LIMIT 1000
            """,
            host_id,
            str(minutes),
        )
    return [dict(r) for r in rows]


async def get_recent_anomalies(host_id: str, limit: int = 50) -> list[dict]:
    """Return the most recent anomaly events for a host."""
    if _pool is None:
        return []

    async with _pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT host_id, timestamp, combined_score, severity,
                   worst_feature, lstm_score, if_score, detection_method
            FROM anomaly_events
            WHERE host_id = $1
            ORDER BY timestamp DESC
            LIMIT $2
            """,
            host_id,
            limit,
        )
    return [dict(r) for r in rows]


async def get_host_stats(host_id: str) -> dict:
    """Return aggregate stats for a host over the last 24 hours."""
    if _pool is None:
        return {}

    async with _pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*)                     AS total_records,
                AVG(latency_ms)              AS avg_latency_ms,
                AVG(packet_loss_pct)         AS avg_packet_loss,
                AVG(health_score)            AS avg_health_score,
                MAX(timestamp)               AS last_seen
            FROM telemetry_records
            WHERE host_id = $1
              AND timestamp >= NOW() - INTERVAL '24 hours'
            """,
            host_id,
        )
    return dict(row) if row else {}


async def close_db() -> None:
    """Close the connection pool gracefully."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("TimescaleDB connection pool closed")
