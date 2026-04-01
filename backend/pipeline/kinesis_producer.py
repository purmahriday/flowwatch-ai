"""
FlowWatch AI — Telemetry Producer
==================================
Simulates realistic network telemetry from multiple hosts and streams
records to AWS Kinesis (or LocalStack for local development).

Usage:
    python -m backend.pipeline.kinesis_producer [--hosts N] [--interval S] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import numpy as np
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ─── Constants ────────────────────────────────────────────────────────────────

ANOMALY_RATE: float = 0.05  # 5% chance of anomaly per host per tick

ANOMALY_TYPES: list[str] = ["SPIKE", "LOSS", "DNS", "CASCADE"]

# ─── Kinesis client factory ───────────────────────────────────────────────────


def get_kinesis_client() -> Any:
    """
    Build and return a boto3 Kinesis client.

    In development, connects to LocalStack at LOCALSTACK_ENDPOINT
    (default: http://localhost:4566).  In all other environments, uses the
    standard AWS SDK credential chain (env vars, instance profile, etc.).
    Credentials are never hardcoded.
    """
    environment = os.getenv("ENVIRONMENT", "development")
    region = os.getenv("AWS_REGION", "us-east-1")

    if environment == "development":
        endpoint_url = os.getenv("LOCALSTACK_ENDPOINT", "http://localhost:4566")
        client = boto3.client(
            "kinesis",
            region_name=region,
            endpoint_url=endpoint_url,
            # LocalStack accepts any non-empty credentials
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
        )
        logger.info("Producer connected to LocalStack Kinesis at {url}", url=endpoint_url)
    else:
        # Production: rely on the standard boto3 credential chain
        client = boto3.client("kinesis", region_name=region)
        logger.info("Producer connected to AWS Kinesis | region={region}", region=region)

    return client


# ─── Telemetry generation ─────────────────────────────────────────────────────


def _generate_normal(host_id: str) -> dict[str, Any]:
    """
    Generate a single normal-baseline telemetry sample for *host_id*.

    Distributions:
        latency_ms       — Normal(mean=45, std=10), clipped to [20, 80]
        packet_loss_pct  — Exponential(scale=0.5), clipped to [0, 2]
        dns_failure_rate — Uniform(0, 0.05)
        jitter_ms        — Normal(mean=8, std=3), clipped to [2, 15]
    """
    latency = float(np.clip(np.random.normal(loc=45.0, scale=10.0), 20.0, 80.0))
    packet_loss = float(np.clip(np.random.exponential(scale=0.5), 0.0, 2.0))
    dns_failure = float(np.random.uniform(0.0, 0.05))
    jitter = float(np.clip(np.random.normal(loc=8.0, scale=3.0), 2.0, 15.0))

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host_id": host_id,
        "latency_ms": round(latency, 3),
        "packet_loss_pct": round(packet_loss, 4),
        "dns_failure_rate": round(dns_failure, 4),
        "jitter_ms": round(jitter, 3),
        "is_anomaly": False,
    }


def _inject_anomaly(record: dict[str, Any]) -> dict[str, Any]:
    """
    Mutate a normal telemetry record into an anomalous one.

    Anomaly types:
        SPIKE    — latency spikes to 300–800 ms
        LOSS     — packet_loss jumps to 15–40 %
        DNS      — dns_failure_rate rises to 0.4–0.9
        CASCADE  — all metrics spike simultaneously (worst-case scenario)

    The ``is_anomaly`` flag is set to ``True`` and an ``anomaly_type``
    field is added.
    """
    anomaly_type = random.choice(ANOMALY_TYPES)
    record = record.copy()
    record["is_anomaly"] = True
    record["anomaly_type"] = anomaly_type

    if anomaly_type == "SPIKE":
        record["latency_ms"] = round(random.uniform(300.0, 800.0), 3)

    elif anomaly_type == "LOSS":
        record["packet_loss_pct"] = round(random.uniform(15.0, 40.0), 4)

    elif anomaly_type == "DNS":
        record["dns_failure_rate"] = round(random.uniform(0.4, 0.9), 4)

    elif anomaly_type == "CASCADE":
        record["latency_ms"] = round(random.uniform(300.0, 800.0), 3)
        record["packet_loss_pct"] = round(random.uniform(15.0, 40.0), 4)
        record["dns_failure_rate"] = round(random.uniform(0.4, 0.9), 4)
        record["jitter_ms"] = round(random.uniform(50.0, 150.0), 3)

    return record


def generate_telemetry(host_id: str) -> dict[str, Any]:
    """
    Generate a telemetry record for *host_id*, injecting an anomaly with
    probability ``ANOMALY_RATE`` (default 5 %).

    Args:
        host_id: Identifier for the simulated host (e.g. ``"host-01"``).

    Returns:
        A dict ready for JSON serialisation and Kinesis publication.
    """
    record = _generate_normal(host_id)
    if random.random() < ANOMALY_RATE:
        record = _inject_anomaly(record)
    return record


# ─── Kinesis I/O ──────────────────────────────────────────────────────────────


def send_to_kinesis(client: Any, stream_name: str, record: dict[str, Any]) -> None:
    """
    Serialise *record* as UTF-8 JSON and publish it to *stream_name*.

    Uses ``host_id`` as the Kinesis partition key so that all records for a
    single host land on the same shard (preserving per-host ordering).

    Args:
        client:      A boto3 Kinesis client.
        stream_name: The target Kinesis stream name.
        record:      Telemetry dict to publish.

    Raises:
        botocore.exceptions.ClientError: Propagated to the caller for
            per-record error handling.
    """
    data = json.dumps(record, default=str).encode("utf-8")
    response = client.put_record(
        StreamName=stream_name,
        Data=data,
        PartitionKey=record["host_id"],
    )
    logger.debug(
        "→ Kinesis | host={host} shard={shard} seq={seq} anomaly={anomaly}",
        host=record["host_id"],
        shard=response["ShardId"],
        seq=response["SequenceNumber"][:12] + "…",
        anomaly=record["is_anomaly"],
    )


# ─── Main producer loop ───────────────────────────────────────────────────────


def run_producer(
    hosts: int,
    interval: float,
    dry_run: bool,
    stream_name: str,
) -> None:
    """
    Infinite loop: generate and publish one telemetry record per host per tick.

    Each tick targets a wall-clock duration of *interval* seconds.  If
    generation + publishing finishes faster, the remainder is spent sleeping.
    If it takes longer, the next tick starts immediately (no backlog).

    Args:
        hosts:       Number of simulated hosts (named host-01 … host-N).
        interval:    Target seconds between ticks (default 1.0).
        dry_run:     If True, print records to stdout instead of sending to Kinesis.
        stream_name: Kinesis stream name.
    """
    host_ids = [f"host-{i:02d}" for i in range(1, hosts + 1)]
    client: Optional[Any] = None

    if not dry_run:
        client = get_kinesis_client()

    logger.info(
        "Producer ready | hosts={hosts} interval={interval}s "
        "dry_run={dry_run} stream={stream}",
        hosts=hosts,
        interval=interval,
        dry_run=dry_run,
        stream=stream_name,
    )

    tick = 0
    while True:
        tick += 1
        tick_start = time.monotonic()

        for host_id in host_ids:
            record = generate_telemetry(host_id)

            if dry_run:
                # Pretty-print to stdout for inspection
                logger.info("DRY-RUN | {payload}", payload=json.dumps(record))
            else:
                assert client is not None
                try:
                    send_to_kinesis(client, stream_name, record)
                except ClientError as exc:
                    logger.error(
                        "Kinesis error for {host}: {err}",
                        host=host_id,
                        err=exc,
                    )

            # Emit a prominent log line for every anomaly regardless of dry-run mode
            if record["is_anomaly"]:
                logger.warning(
                    "⚠ ANOMALY | host={host} type={atype} "
                    "latency={lat}ms loss={loss}% dns={dns}",
                    host=record["host_id"],
                    atype=record.get("anomaly_type", "UNKNOWN"),
                    lat=record["latency_ms"],
                    loss=record["packet_loss_pct"],
                    dns=record["dns_failure_rate"],
                )

        elapsed = time.monotonic() - tick_start
        sleep_for = max(0.0, interval - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)


# ─── Entry point ──────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the producer."""
    parser = argparse.ArgumentParser(
        prog="kinesis_producer",
        description="FlowWatch AI — simulated network telemetry producer",
    )
    parser.add_argument(
        "--hosts",
        type=int,
        default=5,
        metavar="N",
        help="Number of hosts to simulate (default: 5)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        metavar="S",
        help="Seconds between ticks (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print records to stdout without sending to Kinesis",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint with graceful SIGINT / SIGTERM shutdown."""
    args = _parse_args()
    stream_name = os.getenv("KINESIS_STREAM_NAME", "flowwatch-telemetry")

    def _handle_shutdown(sig: int, _frame: Any) -> None:
        logger.info("Received signal {sig} — shutting down gracefully.", sig=sig)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)

    run_producer(
        hosts=args.hosts,
        interval=args.interval,
        dry_run=args.dry_run,
        stream_name=stream_name,
    )


if __name__ == "__main__":
    main()
