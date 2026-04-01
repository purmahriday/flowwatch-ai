"""
FlowWatch AI — Telemetry Preprocessor
=======================================
Cleans raw TelemetryRecord instances (outlier clipping), normalises all
numeric fields to [0, 1], derives composite features, and returns a
ProcessedRecord ready for the ML model layer.

Intended to be called by the Kinesis consumer after each validated record.

Example usage::

    from backend.pipeline.kinesis_consumer import TelemetryRecord
    from backend.pipeline.preprocessor import preprocess

    raw = TelemetryRecord(
        timestamp="2026-04-01T12:00:00+00:00",
        host_id="host-01",
        latency_ms=45.0,
        packet_loss_pct=0.5,
        dns_failure_rate=0.01,
        jitter_ms=8.0,
        is_anomaly=False,
    )
    processed = preprocess(raw)
    print(processed.composite_health_score)
"""

from __future__ import annotations

import doctest
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

# Relative import so both files stay in the same pipeline package
from .kinesis_consumer import TelemetryRecord

# ─── Domain bounds for min-max normalisation ──────────────────────────────────

# Latency: physical links rarely exceed 1 000 ms under any real condition.
LATENCY_MIN: float = 0.0
LATENCY_MAX: float = 1_000.0  # hard cap (ms)

# Packet loss is a percentage.
LOSS_MIN: float = 0.0
LOSS_MAX: float = 100.0  # hard cap (%)

# DNS failure rate is already a fraction.
DNS_MIN: float = 0.0
DNS_MAX: float = 1.0

# Jitter: 200 ms is an extreme upper bound for any WAN link.
JITTER_MIN: float = 0.0
JITTER_MAX: float = 200.0  # hard cap (ms)

# ─── Composite health-score weights ──────────────────────────────────────────

WEIGHT_LATENCY: float = 0.40
WEIGHT_LOSS: float = 0.30
WEIGHT_DNS: float = 0.20
WEIGHT_JITTER: float = 0.10

assert abs(WEIGHT_LATENCY + WEIGHT_LOSS + WEIGHT_DNS + WEIGHT_JITTER - 1.0) < 1e-9, (
    "Health-score weights must sum to 1.0"
)

# ─── ProcessedRecord schema ───────────────────────────────────────────────────


class ProcessedRecord(BaseModel):
    """
    Cleaned and feature-enriched telemetry record output by the preprocessor.

    Inherits all raw fields from TelemetryRecord (with clipping applied),
    and adds four normalised scalars, a composite health score, and a
    business-hours context flag.
    """

    # ── Original fields (post-clipping) ──────────────────────────────────────
    timestamp: str
    host_id: str
    latency_ms: float = Field(ge=0.0, description="Latency capped at 1 000 ms.")
    packet_loss_pct: float = Field(ge=0.0, le=100.0, description="Packet loss capped at 100 %.")
    dns_failure_rate: float = Field(ge=0.0, le=1.0)
    jitter_ms: float = Field(ge=0.0)
    is_anomaly: bool
    anomaly_type: Optional[str] = None

    # ── Normalised features [0, 1] ────────────────────────────────────────────
    latency_normalized: float = Field(ge=0.0, le=1.0, description="latency_ms scaled to [0, 1].")
    loss_normalized: float = Field(ge=0.0, le=1.0, description="packet_loss_pct scaled to [0, 1].")
    dns_normalized: float = Field(ge=0.0, le=1.0, description="dns_failure_rate scaled to [0, 1].")
    jitter_normalized: float = Field(ge=0.0, le=1.0, description="jitter_ms scaled to [0, 1].")

    # ── Composite health score [0, 1] ─────────────────────────────────────────
    composite_health_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Weighted average of normalised metrics. "
            "0.0 = perfect health, 1.0 = completely degraded. "
            "Weights: latency 40%, packet_loss 30%, dns 20%, jitter 10%."
        ),
    )

    # ── Context flag ──────────────────────────────────────────────────────────
    is_business_hours: bool = Field(
        description="True when the record's UTC timestamp falls Mon-Fri 08:00-18:00."
    )


# ─── Helper functions ─────────────────────────────────────────────────────────


def _min_max_scale(value: float, lo: float, hi: float) -> float:
    """
    Clip *value* to [*lo*, *hi*] and scale it linearly to [0, 1].

    Args:
        value: Raw metric value.
        lo:    Minimum of the domain range.
        hi:    Maximum of the domain range.

    Returns:
        A float in [0.0, 1.0].

    >>> _min_max_scale(0.0, 0.0, 100.0)
    0.0
    >>> _min_max_scale(100.0, 0.0, 100.0)
    1.0
    >>> _min_max_scale(50.0, 0.0, 100.0)
    0.5
    >>> _min_max_scale(-10.0, 0.0, 100.0)   # clipped to lo
    0.0
    >>> _min_max_scale(150.0, 0.0, 100.0)   # clipped to hi
    1.0
    """
    if hi <= lo:
        return 0.0
    clipped = max(lo, min(hi, value))
    return (clipped - lo) / (hi - lo)


def _compute_health_score(
    latency_norm: float,
    loss_norm: float,
    dns_norm: float,
    jitter_norm: float,
) -> float:
    """
    Compute a composite network health score as a weighted average of the
    four normalised metrics.

    A score of **0.0** means all metrics are at their best (perfect health).
    A score of **1.0** means every metric is at its worst (completely degraded).

    Weights:
        latency  → 40 %
        loss     → 30 %
        dns      → 20 %
        jitter   → 10 %

    Args:
        latency_norm:  Normalised latency [0, 1].
        loss_norm:     Normalised packet-loss [0, 1].
        dns_norm:      Normalised DNS failure rate [0, 1].
        jitter_norm:   Normalised jitter [0, 1].

    Returns:
        Composite health score in [0.0, 1.0].

    >>> round(_compute_health_score(0.0, 0.0, 0.0, 0.0), 6)
    0.0
    >>> round(_compute_health_score(1.0, 1.0, 1.0, 1.0), 6)
    1.0
    >>> round(_compute_health_score(0.5, 0.5, 0.5, 0.5), 6)
    0.5
    >>> round(_compute_health_score(1.0, 0.0, 0.0, 0.0), 6)
    0.4
    >>> round(_compute_health_score(0.0, 1.0, 0.0, 0.0), 6)
    0.3
    """
    return (
        WEIGHT_LATENCY * latency_norm
        + WEIGHT_LOSS * loss_norm
        + WEIGHT_DNS * dns_norm
        + WEIGHT_JITTER * jitter_norm
    )


def _is_business_hours(timestamp_iso: str) -> bool:
    """
    Return True if *timestamp_iso* falls within business hours (UTC).

    Business hours are defined as **Monday through Friday, 08:00–17:59 UTC**.

    Args:
        timestamp_iso: ISO 8601 string (timezone-aware or naive UTC).

    Returns:
        True when the timestamp is within Mon–Fri 08:00–18:00 UTC.

    >>> _is_business_hours("2026-04-01T10:00:00+00:00")  # Wednesday 10:00 UTC
    True
    >>> _is_business_hours("2026-04-01T07:59:00+00:00")  # Wednesday 07:59 UTC
    False
    >>> _is_business_hours("2026-04-01T18:00:00+00:00")  # Wednesday 18:00 UTC (boundary)
    False
    >>> _is_business_hours("2026-04-01T22:30:00+00:00")  # Wednesday 22:30 UTC
    False
    >>> _is_business_hours("2026-04-05T10:00:00+00:00")  # Sunday 10:00 UTC
    False
    >>> _is_business_hours("2026-04-06T10:00:00+00:00")  # Monday 10:00 UTC
    True
    """
    dt = datetime.fromisoformat(timestamp_iso)
    if dt.tzinfo is None:
        # Treat naive timestamps as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    # weekday(): 0 = Monday … 6 = Sunday
    return dt.weekday() < 5 and 8 <= dt.hour < 18


# ─── Main preprocessing entry point ──────────────────────────────────────────


def preprocess(record: TelemetryRecord) -> ProcessedRecord:
    """
    Transform a validated *TelemetryRecord* into a *ProcessedRecord* ready
    for the ML model layer.

    Processing steps:
        1. **Clip** — cap ``latency_ms`` at 1 000 ms and ``packet_loss_pct``
           at 100 % to neutralise impossible sensor values.
        2. **Normalise** — min-max scale all four numeric metrics to [0, 1]
           using fixed domain bounds (not fit on data, so inference is safe).
        3. **Health score** — compute the weighted composite score.
        4. **Business hours** — derive a boolean context flag from the
           record timestamp.

    Args:
        record: A validated :class:`TelemetryRecord` from the Kinesis consumer.

    Returns:
        A :class:`ProcessedRecord` with all original fields plus derived ones.

    Example::

        >>> from backend.pipeline.kinesis_consumer import TelemetryRecord
        >>> r = TelemetryRecord(
        ...     timestamp="2026-04-01T12:00:00+00:00",
        ...     host_id="host-01",
        ...     latency_ms=500.0,   # maps to 0.5 on [0, 1000]
        ...     packet_loss_pct=0.0,
        ...     dns_failure_rate=0.0,
        ...     jitter_ms=0.0,
        ...     is_anomaly=False,
        ... )
        >>> p = preprocess(r)
        >>> p.latency_normalized
        0.5
        >>> p.composite_health_score  # 0.5 * 0.40 = 0.20
        0.2
    """
    # Step 1: Clip outliers
    latency_clipped = min(record.latency_ms, LATENCY_MAX)
    loss_clipped = min(record.packet_loss_pct, LOSS_MAX)

    # Step 2: Normalise
    latency_norm = _min_max_scale(latency_clipped, LATENCY_MIN, LATENCY_MAX)
    loss_norm = _min_max_scale(loss_clipped, LOSS_MIN, LOSS_MAX)
    dns_norm = _min_max_scale(record.dns_failure_rate, DNS_MIN, DNS_MAX)
    jitter_norm = _min_max_scale(record.jitter_ms, JITTER_MIN, JITTER_MAX)

    # Step 3: Composite health score
    health = _compute_health_score(latency_norm, loss_norm, dns_norm, jitter_norm)

    # Step 4: Business hours flag
    is_biz = _is_business_hours(record.timestamp)

    return ProcessedRecord(
        # Original (clipped) fields
        timestamp=record.timestamp,
        host_id=record.host_id,
        latency_ms=latency_clipped,
        packet_loss_pct=loss_clipped,
        dns_failure_rate=record.dns_failure_rate,
        jitter_ms=record.jitter_ms,
        is_anomaly=record.is_anomaly,
        anomaly_type=record.anomaly_type,
        # Derived features
        latency_normalized=round(latency_norm, 6),
        loss_normalized=round(loss_norm, 6),
        dns_normalized=round(dns_norm, 6),
        jitter_normalized=round(jitter_norm, 6),
        composite_health_score=round(health, 6),
        is_business_hours=is_biz,
    )


# ─── Doctest runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = doctest.testmod(verbose=True)
    raise SystemExit(0 if results.failed == 0 else 1)
