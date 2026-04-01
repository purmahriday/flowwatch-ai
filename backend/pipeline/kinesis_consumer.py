"""
FlowWatch AI — Telemetry Consumer
===================================
Continuously reads telemetry records from a Kinesis stream, validates them
with a Pydantic schema, logs anomalies, and emits rolling statistics every
30 seconds.

Usage:
    python -m backend.pipeline.kinesis_consumer
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from collections import defaultdict
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, field_validator

load_dotenv()

# ─── Pydantic schema ──────────────────────────────────────────────────────────


class TelemetryRecord(BaseModel):
    """
    Pydantic v2 model that validates a raw telemetry record arriving from
    the Kinesis stream.

    All numeric fields are range-checked at construction time so that
    downstream ML code can assume clean inputs.
    """

    timestamp: str
    """ISO 8601 UTC timestamp produced by the telemetry generator."""

    host_id: str
    """Unique identifier of the originating host (e.g. ``"host-01"``)."""

    latency_ms: float = Field(ge=0.0, description="Round-trip latency in milliseconds.")
    packet_loss_pct: float = Field(ge=0.0, le=100.0, description="Packet loss percentage [0, 100].")
    dns_failure_rate: float = Field(ge=0.0, le=1.0, description="Fraction of DNS queries that failed [0, 1].")
    jitter_ms: float = Field(ge=0.0, description="Jitter in milliseconds.")
    is_anomaly: bool
    """True when the record was injected as an anomaly by the producer."""

    anomaly_type: Optional[str] = None
    """SPIKE | LOSS | DNS | CASCADE — present only when is_anomaly is True."""

    @field_validator("host_id")
    @classmethod
    def host_id_not_empty(cls, v: str) -> str:
        """Reject blank host identifiers."""
        if not v.strip():
            raise ValueError("host_id must not be empty")
        return v

    @field_validator("anomaly_type")
    @classmethod
    def valid_anomaly_type(cls, v: Optional[str]) -> Optional[str]:
        """Allow only the four known anomaly type strings (or None)."""
        allowed = {"SPIKE", "LOSS", "DNS", "CASCADE"}
        if v is not None and v not in allowed:
            raise ValueError(f"anomaly_type must be one of {allowed}, got {v!r}")
        return v


# ─── Kinesis client factory ───────────────────────────────────────────────────


def get_kinesis_client() -> Any:
    """
    Build and return a boto3 Kinesis client.

    Uses LocalStack when ``ENVIRONMENT=development`` (default), falls back
    to the standard AWS credential chain in all other environments.
    """
    environment = os.getenv("ENVIRONMENT", "development")
    region = os.getenv("AWS_REGION", "us-east-1")

    if environment == "development":
        endpoint_url = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
        client = boto3.client(
            "kinesis",
            region_name=region,
            endpoint_url=endpoint_url,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        )
        logger.info("Consumer connected to LocalStack Kinesis at {url}", url=endpoint_url)
    else:
        client = boto3.client("kinesis", region_name=region)
        logger.info("Consumer connected to AWS Kinesis | region={region}", region=region)

    return client


def get_shard_iterators(client: Any, stream_name: str) -> dict[str, Optional[str]]:
    """
    Describe *stream_name* and return a ``{shard_id: iterator}`` mapping
    using the ``LATEST`` iterator type (consume only new records).

    Args:
        client:      A boto3 Kinesis client.
        stream_name: Name of the stream to consume.

    Returns:
        Dict mapping each shard ID to its initial shard iterator string.
    """
    response = client.describe_stream(StreamName=stream_name)
    shards = response["StreamDescription"]["Shards"]
    iterators: dict[str, Optional[str]] = {}

    for shard in shards:
        shard_id: str = shard["ShardId"]
        it_response = client.get_shard_iterator(
            StreamName=stream_name,
            ShardId=shard_id,
            ShardIteratorType="LATEST",
        )
        iterators[shard_id] = it_response["ShardIterator"]
        logger.debug("Shard iterator obtained | shard={shard}", shard=shard_id)

    logger.info("Consuming {n} shard(s) from stream {stream}", n=len(shards), stream=stream_name)
    return iterators


# ─── Rolling statistics tracker ───────────────────────────────────────────────


class StatsTracker:
    """
    Accumulates running counters and emits a human-readable summary log
    every ``report_interval`` seconds.
    """

    def __init__(self, report_interval: float = 30.0) -> None:
        self.report_interval = report_interval
        self.total: int = 0
        self.anomaly_count: int = 0
        self._latency_sum: dict[str, float] = defaultdict(float)
        self._latency_count: dict[str, int] = defaultdict(int)
        self._max_packet_loss: float = 0.0
        self._last_report: float = time.monotonic()

    def record(self, rec: TelemetryRecord) -> None:
        """
        Update counters with a newly received *rec*.

        Args:
            rec: A validated TelemetryRecord.
        """
        self.total += 1
        if rec.is_anomaly:
            self.anomaly_count += 1
        self._latency_sum[rec.host_id] += rec.latency_ms
        self._latency_count[rec.host_id] += 1
        if rec.packet_loss_pct > self._max_packet_loss:
            self._max_packet_loss = rec.packet_loss_pct

    def maybe_report(self) -> None:
        """
        Emit a structured stats summary if ``report_interval`` seconds have
        elapsed since the last report.  No-op otherwise.
        """
        now = time.monotonic()
        if now - self._last_report < self.report_interval:
            return

        self._last_report = now
        anomaly_rate = (self.anomaly_count / self.total * 100.0) if self.total else 0.0
        avg_latency = {
            host: self._latency_sum[host] / self._latency_count[host]
            for host in sorted(self._latency_count)
        }
        avg_latency_str = {h: f"{v:.1f}ms" for h, v in avg_latency.items()}

        logger.info(
            "\n"
            "┌─── STATS SUMMARY ────────────────────────────\n"
            "│  total_records  : {total}\n"
            "│  anomaly_count  : {anomalies}\n"
            "│  anomaly_rate   : {rate:.2f}%\n"
            "│  avg_latency    : {latency}\n"
            "│  max_pkt_loss   : {loss:.4f}%\n"
            "└──────────────────────────────────────────────",
            total=self.total,
            anomalies=self.anomaly_count,
            rate=anomaly_rate,
            latency=avg_latency_str,
            loss=self._max_packet_loss,
        )


# ─── Record processing ────────────────────────────────────────────────────────


def process_records(raw_records: list[dict[str, Any]], stats: StatsTracker) -> None:
    """
    Deserialise, validate, and log a batch of raw Kinesis records.

    Each record's ``Data`` field is Base-64 decoded by boto3 before this
    function is called, so we receive raw bytes.  Malformed records are
    skipped with an error log.

    Args:
        raw_records: List of record dicts returned by ``client.get_records()``.
        stats:       StatsTracker to update with each valid record.
    """
    for raw in raw_records:
        try:
            payload = json.loads(raw["Data"])
            rec = TelemetryRecord.model_validate(payload)
        except json.JSONDecodeError as exc:
            logger.error("JSON decode error — skipping record: {err}", err=exc)
            continue
        except Exception as exc:  # Pydantic ValidationError or unexpected
            logger.error("Validation error — skipping record: {err}", err=exc)
            continue

        stats.record(rec)

        if rec.is_anomaly:
            logger.warning(
                "⚠ ANOMALY | host={host} type={atype} "
                "latency={lat}ms loss={loss}% dns={dns} jitter={jitter}ms",
                host=rec.host_id,
                atype=rec.anomaly_type or "UNKNOWN",
                lat=rec.latency_ms,
                loss=rec.packet_loss_pct,
                dns=rec.dns_failure_rate,
                jitter=rec.jitter_ms,
            )
        else:
            logger.debug(
                "OK | host={host} latency={lat}ms loss={loss}% dns={dns}",
                host=rec.host_id,
                lat=rec.latency_ms,
                loss=rec.packet_loss_pct,
                dns=rec.dns_failure_rate,
            )


# ─── Retry / backoff config ───────────────────────────────────────────────────

_BACKOFF_BASE: float = 0.5   # seconds
_BACKOFF_MAX: float = 30.0   # seconds
_POLL_SLEEP: float = 0.25    # seconds between poll rounds


# ─── Main consumer loop ───────────────────────────────────────────────────────


def run_consumer(stream_name: str) -> None:
    """
    Poll all shards of *stream_name* indefinitely, processing records as
    they arrive.

    Resilience features:
      - Expired shard iterators are transparently re-fetched.
      - ``ProvisionedThroughputExceededException`` triggers exponential
        back-off (starting at 0.5 s, capped at 30 s) per shard.
      - ``KeyboardInterrupt`` / SIGINT / SIGTERM exit cleanly.

    Args:
        stream_name: Kinesis stream to consume.
    """
    client = get_kinesis_client()
    stats = StatsTracker(report_interval=30.0)

    logger.info("Consumer starting | stream={stream}", stream=stream_name)
    iterators = get_shard_iterators(client, stream_name)
    backoff: dict[str, float] = {sid: _BACKOFF_BASE for sid in iterators}

    while True:
        for shard_id in list(iterators):
            iterator = iterators[shard_id]

            # ── Re-fetch expired iterator ──────────────────────────────────
            if iterator is None:
                try:
                    resp = client.get_shard_iterator(
                        StreamName=stream_name,
                        ShardId=shard_id,
                        ShardIteratorType="LATEST",
                    )
                    iterators[shard_id] = resp["ShardIterator"]
                    backoff[shard_id] = _BACKOFF_BASE
                    logger.info("Re-fetched iterator for shard {shard}", shard=shard_id)
                except ClientError as exc:
                    logger.error(
                        "Cannot re-fetch iterator for {shard}: {err}",
                        shard=shard_id,
                        err=exc,
                    )
                continue

            # ── Poll for records ───────────────────────────────────────────
            try:
                response = client.get_records(ShardIterator=iterator, Limit=100)
                # Advance the iterator (may be None when shard is exhausted)
                iterators[shard_id] = response.get("NextShardIterator")
                backoff[shard_id] = _BACKOFF_BASE  # reset after successful call

                if response["Records"]:
                    process_records(response["Records"], stats)

            except client.exceptions.ExpiredIteratorException:
                logger.warning(
                    "Shard iterator expired for {shard} — will re-fetch next round",
                    shard=shard_id,
                )
                iterators[shard_id] = None

            except client.exceptions.ProvisionedThroughputExceededException:
                wait = backoff[shard_id]
                logger.warning(
                    "ProvisionedThroughputExceeded on {shard} — backing off {wait:.1f}s",
                    shard=shard_id,
                    wait=wait,
                )
                time.sleep(wait)
                backoff[shard_id] = min(wait * 2.0, _BACKOFF_MAX)

            except ClientError as exc:
                logger.error(
                    "Kinesis ClientError on {shard}: {err}",
                    shard=shard_id,
                    err=exc,
                )

        stats.maybe_report()
        time.sleep(_POLL_SLEEP)


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point with graceful SIGINT / SIGTERM shutdown."""
    stream_name = os.getenv("KINESIS_STREAM_NAME", "flowwatch-telemetry")

    def _handle_shutdown(sig: int, _frame: Any) -> None:
        logger.info("Received signal {sig} — shutting down gracefully.", sig=sig)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    run_consumer(stream_name)


if __name__ == "__main__":
    main()
