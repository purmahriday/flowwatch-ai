"""
FlowWatch AI — Feature Engineering
=====================================
Transforms :class:`ProcessedRecord` instances from the preprocessor into
rich feature vectors that can be fed directly into the LSTM autoencoder
(temporal sliding-window format) and the Isolation Forest (flat vector).

Pipeline position::

    ProcessedRecord  →  WindowBuffer.add()  →  FeatureExtractor.process()
                                                      │
                                    ┌─────────────────┴──────────────────┐
                                    ▼                                    ▼
                          to_lstm_input()                  to_isolation_forest_input()
                          np.ndarray (30, 4)               np.ndarray (19,)

Feature groups (19 total):
    Statistical (11)  — rolling statistics over the 30-record window
    Rate-of-change (3) — first differences of the last two records
    Temporal (5)      — cyclic time-of-day / day-of-week + business-hours flag

Dependencies: numpy, collections, threading, datetime — no additional packages.
"""

from __future__ import annotations

import collections
import doctest
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

# Import the upstream schema so callers don't need to juggle two import paths.
from backend.pipeline.preprocessor import ProcessedRecord

# ─── Module-level constants ───────────────────────────────────────────────────

WINDOW_SIZE: int = 30
"""Number of consecutive records required before features can be computed."""

# Indices into the per-record feature columns stored in the window array
_COL_LATENCY: int = 0
_COL_LOSS: int = 1
_COL_DNS: int = 2
_COL_JITTER: int = 3
_COL_HEALTH: int = 4  # composite_health_score — used for trend only

# Total window array width (4 ML-input columns + 1 auxiliary)
_WINDOW_COLS: int = 5

# ─── Sliding-window buffer ────────────────────────────────────────────────────


class WindowBuffer:
    """
    Thread-safe, per-host rolling buffer of the last ``WINDOW_SIZE`` records.

    Internally each host maps to a :class:`collections.deque` with
    ``maxlen=WINDOW_SIZE``, storing one ``numpy`` row per record.  A single
    :class:`threading.Lock` guards all mutation so the buffer is safe to
    share between the Kinesis consumer thread and any inference threads.

    Row layout (width 5):
        [latency_normalized, loss_normalized, dns_normalized,
         jitter_normalized, composite_health_score]

    Example::

        >>> buf = WindowBuffer()
        >>> buf.is_ready("host-01")
        False
        >>> rec = _make_test_record(latency_norm=0.05, loss_norm=0.01,
        ...                         dns_norm=0.02, jitter_norm=0.04,
        ...                         health=0.035)
        >>> for _ in range(30):
        ...     _ = buf.add(rec)
        >>> buf.is_ready("host-01")
        True
        >>> buf.get_window("host-01").shape
        (30, 5)
    """

    def __init__(self) -> None:
        # host_id → deque of numpy rows
        self._windows: dict[str, collections.deque[np.ndarray]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: ProcessedRecord) -> Optional[np.ndarray]:
        """
        Append *record* to the host's window and return the full window
        array once it reaches ``WINDOW_SIZE`` records.

        Args:
            record: A :class:`ProcessedRecord` from the preprocessor.

        Returns:
            ``numpy`` array of shape ``(WINDOW_SIZE, 5)`` when the window is
            full, or ``None`` if the window is still filling up.
        """
        row = np.array(
            [
                record.latency_normalized,
                record.loss_normalized,
                record.dns_normalized,
                record.jitter_normalized,
                record.composite_health_score,
            ],
            dtype=np.float64,
        )

        with self._lock:
            if record.host_id not in self._windows:
                self._windows[record.host_id] = collections.deque(maxlen=WINDOW_SIZE)
            self._windows[record.host_id].append(row)

            if len(self._windows[record.host_id]) == WINDOW_SIZE:
                return np.array(self._windows[record.host_id], dtype=np.float64)

        return None

    def get_window(self, host_id: str) -> np.ndarray:
        """
        Return the current window for *host_id* as a ``(N, 5)`` array,
        where ``N`` is the number of records accumulated so far (≤ WINDOW_SIZE).

        Args:
            host_id: Host identifier string.

        Returns:
            Array of shape ``(N, 5)``.  Shape is ``(0, 5)`` if no records
            have been received for *host_id* yet.
        """
        with self._lock:
            if host_id not in self._windows:
                return np.empty((0, _WINDOW_COLS), dtype=np.float64)
            return np.array(self._windows[host_id], dtype=np.float64)

    def is_ready(self, host_id: str) -> bool:
        """
        Return ``True`` only when the window for *host_id* contains exactly
        ``WINDOW_SIZE`` records and feature extraction can begin.

        Args:
            host_id: Host identifier string.

        Returns:
            Boolean readiness flag.

        >>> buf = WindowBuffer()
        >>> buf.is_ready("x")
        False
        """
        with self._lock:
            return (
                host_id in self._windows
                and len(self._windows[host_id]) == WINDOW_SIZE
            )

    def fill_level(self, host_id: str) -> int:
        """
        Return how many records have been buffered for *host_id* so far.

        Args:
            host_id: Host identifier string.

        Returns:
            Integer in ``[0, WINDOW_SIZE]``.
        """
        with self._lock:
            if host_id not in self._windows:
                return 0
            return len(self._windows[host_id])


# ─── Statistical helpers ──────────────────────────────────────────────────────


def _linear_slope(values: np.ndarray) -> float:
    """
    Fit a degree-1 polynomial to *values* and return the slope.

    Uses index positions ``[0, 1, …, n-1]`` as the x-axis, so the slope
    reflects change per step.  A positive slope means the metric is getting
    worse; negative means it is recovering.

    Args:
        values: 1-D numpy array of floats.

    Returns:
        Slope coefficient (float).  Returns ``0.0`` for arrays shorter than 2.

    >>> round(_linear_slope(np.array([0.1, 0.2, 0.3, 0.4])), 6)
    0.1
    >>> round(_linear_slope(np.array([0.4, 0.3, 0.2, 0.1])), 6)
    -0.1
    >>> abs(_linear_slope(np.array([0.5, 0.5, 0.5]))) < 1e-10
    True
    >>> _linear_slope(np.array([0.5]))
    0.0
    """
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    coeffs = np.polyfit(x, values, 1)
    return float(coeffs[0])


def _cyclic_encode(value: float, period: float) -> tuple[float, float]:
    """
    Encode *value* cyclically as ``(sin, cos)`` to avoid discontinuities at
    period boundaries (e.g. hour 23 → 0, weekday 6 → 0).

    Args:
        value:  Raw value (e.g. hour 0–23, weekday 0–6).
        period: Full cycle length (24 for hours, 7 for weekdays).

    Returns:
        Tuple ``(sin_encoded, cos_encoded)`` both in ``[-1, 1]``.

    >>> s, c = _cyclic_encode(0.0, 24.0)
    >>> round(s, 6), round(c, 6)
    (0.0, 1.0)
    >>> s, c = _cyclic_encode(6.0, 24.0)   # quarter turn
    >>> round(s, 6), round(c, 6)
    (1.0, 0.0)
    >>> s, c = _cyclic_encode(12.0, 24.0)  # half turn
    >>> round(s, 6)
    0.0
    >>> round(c, 6)
    -1.0
    """
    angle = 2.0 * math.pi * value / period
    return math.sin(angle), math.cos(angle)


def _count_spikes(series: np.ndarray, threshold: float) -> int:
    """
    Count how many values in *series* exceed *threshold*.

    Args:
        series:    1-D numpy array of floats.
        threshold: Scalar boundary (exclusive comparison >).

    Returns:
        Integer spike count.

    >>> _count_spikes(np.array([0.1, 0.8, 0.9, 0.5, 0.71]), 0.7)
    3
    >>> _count_spikes(np.array([0.1, 0.2, 0.3]), 0.7)
    0
    >>> _count_spikes(np.array([0.7, 0.7, 0.7]), 0.7)  # boundary — not >
    0
    """
    return int(np.sum(series > threshold))


# ─── Feature vector ───────────────────────────────────────────────────────────

# Canonical ordering for to_isolation_forest_input() — must stay stable.
_IF_FEATURE_ORDER: tuple[str, ...] = (
    "rolling_mean_latency",
    "rolling_std_latency",
    "rolling_mean_loss",
    "rolling_std_loss",
    "rolling_mean_dns",
    "rolling_mean_jitter",
    "rolling_std_jitter",
    "latency_trend",
    "health_score_trend",
    "spike_count",
    "loss_spike_count",
    "latency_delta",
    "loss_delta",
    "dns_delta",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_business_hours",
)

assert len(_IF_FEATURE_ORDER) == 19, "Expected exactly 19 Isolation Forest features."


@dataclass
class FeatureVector:
    """
    A fully computed feature snapshot for a single host at a single point
    in time.

    Contains all 19 scalar features plus the raw ``(WINDOW_SIZE, 4)`` window
    used as LSTM input.

    Attributes:
        host_id:               Host this vector was computed for.
        timestamp:             ISO 8601 timestamp of the triggering record.

        Statistical features (11):
            rolling_mean_latency:   Mean of latency_normalized over window.
            rolling_std_latency:    Std-dev of latency_normalized over window.
            rolling_mean_loss:      Mean of loss_normalized over window.
            rolling_std_loss:       Std-dev of loss_normalized over window.
            rolling_mean_dns:       Mean of dns_normalized over window.
            rolling_mean_jitter:    Mean of jitter_normalized over window.
            rolling_std_jitter:     Std-dev of jitter_normalized over window.
            latency_trend:          Linear regression slope of latency_normalized.
            health_score_trend:     Linear regression slope of composite_health_score.
            spike_count:            Records where latency_normalized > 0.7.
            loss_spike_count:       Records where loss_normalized > 0.5.

        Rate-of-change features (3):
            latency_delta:          latency_normalized[-1] − latency_normalized[-2].
            loss_delta:             loss_normalized[-1] − loss_normalized[-2].
            dns_delta:              dns_normalized[-1] − dns_normalized[-2].

        Temporal features (5):
            hour_sin:               sin(2π × hour / 24).
            hour_cos:               cos(2π × hour / 24).
            day_sin:                sin(2π × weekday / 7).
            day_cos:                cos(2π × weekday / 7).
            is_business_hours:      Float 1.0 if Mon–Fri 08:00–17:59 UTC, else 0.0.

        _raw_window:               Internal (WINDOW_SIZE, 4) array for LSTM.
    """

    # Context
    host_id: str
    timestamp: str

    # ── Statistical (11) ──────────────────────────────────────────────────────
    rolling_mean_latency: float
    rolling_std_latency: float
    rolling_mean_loss: float
    rolling_std_loss: float
    rolling_mean_dns: float
    rolling_mean_jitter: float
    rolling_std_jitter: float
    latency_trend: float
    health_score_trend: float
    spike_count: float        # kept as float so the IF array is homogeneous
    loss_spike_count: float

    # ── Rate-of-change (3) ────────────────────────────────────────────────────
    latency_delta: float
    loss_delta: float
    dns_delta: float

    # ── Temporal (5) ─────────────────────────────────────────────────────────
    hour_sin: float
    hour_cos: float
    day_sin: float
    day_cos: float
    is_business_hours: float  # 1.0 / 0.0

    # ── Raw window for LSTM (not a scalar feature) ────────────────────────────
    _raw_window: np.ndarray = field(repr=False)

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def to_lstm_input(self) -> np.ndarray:
        """
        Return the raw sliding-window array suitable for LSTM autoencoder input.

        Returns:
            ``numpy`` array of shape ``(WINDOW_SIZE, 4)`` with columns
            ``[latency_normalized, loss_normalized, dns_normalized,
            jitter_normalized]`` in chronological order (oldest first).

        >>> fv = _make_test_feature_vector()
        >>> fv.to_lstm_input().shape
        (30, 4)
        >>> fv.to_lstm_input().dtype
        dtype('float64')
        """
        # _raw_window has 5 columns; drop the auxiliary health column (index 4)
        return self._raw_window[:, :4].astype(np.float64)

    def to_isolation_forest_input(self) -> np.ndarray:
        """
        Return a flat feature vector suitable for the Isolation Forest model.

        Feature order is deterministic and matches ``_IF_FEATURE_ORDER``
        (19 elements).  Boolean ``is_business_hours`` is encoded as ``1.0``
        or ``0.0`` so the array is entirely ``float64``.

        Returns:
            ``numpy`` array of shape ``(19,)`` with ``dtype=float64``.

        >>> fv = _make_test_feature_vector()
        >>> fv.to_isolation_forest_input().shape
        (19,)
        >>> fv.to_isolation_forest_input().dtype
        dtype('float64')
        """
        values = [float(getattr(self, name)) for name in _IF_FEATURE_ORDER]
        return np.array(values, dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        """
        Serialise all scalar features to a plain dict for logging or JSON export.

        The ``_raw_window`` array is excluded.

        Returns:
            Dict with ``host_id``, ``timestamp``, and all 19 feature scalars.
        """
        result: dict[str, object] = {
            "host_id": self.host_id,
            "timestamp": self.timestamp,
        }
        for name in _IF_FEATURE_ORDER:
            result[name] = getattr(self, name)
        return result


# ─── Feature computation ──────────────────────────────────────────────────────


def _extract_time_features(timestamp_iso: str) -> tuple[float, float, float, float, float]:
    """
    Derive the five temporal features from an ISO 8601 timestamp.

    Args:
        timestamp_iso: Timezone-aware (or naive UTC) ISO 8601 string.

    Returns:
        ``(hour_sin, hour_cos, day_sin, day_cos, is_business_hours)``
        where ``is_business_hours`` is ``1.0`` or ``0.0``.

    >>> h_sin, h_cos, d_sin, d_cos, biz = _extract_time_features(
    ...     "2026-04-01T12:00:00+00:00"  # Wednesday noon
    ... )
    >>> round(h_sin, 4)
    0.0
    >>> round(h_cos, 4)
    -1.0
    >>> biz
    1.0
    """
    dt = datetime.fromisoformat(timestamp_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    hour_sin, hour_cos = _cyclic_encode(float(dt.hour), 24.0)
    day_sin, day_cos = _cyclic_encode(float(dt.weekday()), 7.0)
    is_biz = 1.0 if (dt.weekday() < 5 and 8 <= dt.hour < 18) else 0.0

    return hour_sin, hour_cos, day_sin, day_cos, is_biz


def _compute_features(
    host_id: str,
    timestamp: str,
    window: np.ndarray,
) -> FeatureVector:
    """
    Compute all 19 features from a fully-populated window array.

    Args:
        host_id:   Host the window belongs to.
        timestamp: ISO 8601 timestamp of the most recent record.
        window:    ``numpy`` array of shape ``(WINDOW_SIZE, 5)`` with columns
                   ``[latency, loss, dns, jitter, health]``.

    Returns:
        A :class:`FeatureVector` instance.
    """
    lat = window[:, _COL_LATENCY]
    loss = window[:, _COL_LOSS]
    dns = window[:, _COL_DNS]
    jitter = window[:, _COL_JITTER]
    health = window[:, _COL_HEALTH]

    # ── Statistical features ──────────────────────────────────────────────────
    rolling_mean_latency = float(np.mean(lat))
    rolling_std_latency = float(np.std(lat, ddof=1) if len(lat) > 1 else 0.0)
    rolling_mean_loss = float(np.mean(loss))
    rolling_std_loss = float(np.std(loss, ddof=1) if len(loss) > 1 else 0.0)
    rolling_mean_dns = float(np.mean(dns))
    rolling_mean_jitter = float(np.mean(jitter))
    rolling_std_jitter = float(np.std(jitter, ddof=1) if len(jitter) > 1 else 0.0)

    latency_trend = _linear_slope(lat)
    health_score_trend = _linear_slope(health)

    spike_count = float(_count_spikes(lat, threshold=0.7))
    loss_spike_count = float(_count_spikes(loss, threshold=0.5))

    # ── Rate-of-change features ───────────────────────────────────────────────
    latency_delta = float(lat[-1] - lat[-2])
    loss_delta = float(loss[-1] - loss[-2])
    dns_delta = float(dns[-1] - dns[-2])

    # ── Temporal features ─────────────────────────────────────────────────────
    hour_sin, hour_cos, day_sin, day_cos, is_biz = _extract_time_features(timestamp)

    return FeatureVector(
        host_id=host_id,
        timestamp=timestamp,
        rolling_mean_latency=rolling_mean_latency,
        rolling_std_latency=rolling_std_latency,
        rolling_mean_loss=rolling_mean_loss,
        rolling_std_loss=rolling_std_loss,
        rolling_mean_dns=rolling_mean_dns,
        rolling_mean_jitter=rolling_mean_jitter,
        rolling_std_jitter=rolling_std_jitter,
        latency_trend=latency_trend,
        health_score_trend=health_score_trend,
        spike_count=spike_count,
        loss_spike_count=loss_spike_count,
        latency_delta=latency_delta,
        loss_delta=loss_delta,
        dns_delta=dns_delta,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        day_sin=day_sin,
        day_cos=day_cos,
        is_business_hours=is_biz,
        _raw_window=window,
    )


# ─── Feature extractor ────────────────────────────────────────────────────────


class FeatureExtractor:
    """
    Orchestrates :class:`WindowBuffer` management and feature computation.

    Intended to be instantiated once and called for every :class:`ProcessedRecord`
    that exits the preprocessor.  The extractor is thread-safe through the
    lock embedded in :class:`WindowBuffer`.

    Example::

        extractor = FeatureExtractor()
        # Feed records from the Kinesis consumer loop:
        for record in stream:
            fv = extractor.process(record)
            if fv is not None:
                model.predict(fv.to_isolation_forest_input())
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        """
        Initialise the extractor with a fresh :class:`WindowBuffer`.

        Args:
            window_size: Number of records required before features can be
                         computed.  Defaults to ``WINDOW_SIZE`` (30).
        """
        self._window_size = window_size
        self._buffer = WindowBuffer()

    def process(self, record: ProcessedRecord) -> Optional[FeatureVector]:
        """
        Ingest *record*, update the host's window, and return a
        :class:`FeatureVector` once the window is full.

        Args:
            record: A :class:`ProcessedRecord` from the preprocessor.

        Returns:
            A :class:`FeatureVector` when ``WINDOW_SIZE`` records have been
            accumulated for ``record.host_id``, or ``None`` during the
            warm-up period.

        Example::

            >>> extractor = FeatureExtractor()
            >>> rec = _make_test_record(0.05, 0.01, 0.02, 0.04, 0.035)
            >>> results = [extractor.process(rec) for _ in range(30)]
            >>> results[28] is None   # still warming up at record 29
            True
            >>> results[29] is not None  # full at record 30
            True
        """
        window = self._buffer.add(record)
        if window is None:
            return None
        return _compute_features(record.host_id, record.timestamp, window)

    def get_stats(self) -> dict[str, dict[str, object]]:
        """
        Return a snapshot of the window fill status for every known host.

        Returns:
            Dict mapping each ``host_id`` to a sub-dict with:
                ``fill``:    Number of records buffered so far.
                ``ready``:   Whether the window is full.
                ``pct``:     Fill percentage as a float in ``[0.0, 100.0]``.

        Example::

            >>> extractor = FeatureExtractor()
            >>> extractor.get_stats()
            {}
        """
        # Snapshot the known host IDs under the buffer's lock
        with self._buffer._lock:
            host_ids = list(self._buffer._windows.keys())

        stats: dict[str, dict[str, object]] = {}
        for hid in host_ids:
            fill = self._buffer.fill_level(hid)
            stats[hid] = {
                "fill": fill,
                "ready": self._buffer.is_ready(hid),
                "pct": round(fill / self._window_size * 100.0, 1),
            }
        return stats


# ─── Doctest helpers (not part of the public API) ─────────────────────────────


def _make_test_record(
    latency_norm: float = 0.05,
    loss_norm: float = 0.01,
    dns_norm: float = 0.02,
    jitter_norm: float = 0.04,
    health: float = 0.035,
) -> ProcessedRecord:
    """
    Build a minimal :class:`ProcessedRecord` for use in doctests.

    This helper is intentionally not exported — it lives here so doctests
    in this module can reference it without importing test fixtures.
    """
    return ProcessedRecord(
        timestamp="2026-04-01T10:00:00+00:00",
        host_id="host-01",
        latency_ms=50.0,
        packet_loss_pct=1.0,
        dns_failure_rate=0.02,
        jitter_ms=8.0,
        is_anomaly=False,
        anomaly_type=None,
        latency_normalized=latency_norm,
        loss_normalized=loss_norm,
        dns_normalized=dns_norm,
        jitter_normalized=jitter_norm,
        composite_health_score=health,
        is_business_hours=True,
    )


def _make_test_feature_vector() -> FeatureVector:
    """
    Return a :class:`FeatureVector` with a constant synthetic window for use
    in doctests.
    """
    window = np.full((WINDOW_SIZE, _WINDOW_COLS), 0.05, dtype=np.float64)
    return _compute_features(
        host_id="host-01",
        timestamp="2026-04-01T10:00:00+00:00",
        window=window,
    )


# ─── Doctest runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = doctest.testmod(verbose=True)
    raise SystemExit(0 if results.failed == 0 else 1)
