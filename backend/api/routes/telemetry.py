"""
FlowWatch AI — Telemetry Route Handlers
========================================
Handles raw telemetry ingestion and serves recent telemetry data from the
in-memory store.

Endpoints
---------
POST /telemetry/ingest
    Validates a raw telemetry payload, runs it through the preprocessing and
    feature-engineering pipeline, optionally detects anomalies, and persists
    the ProcessedRecord in the in-memory store (deque of last 1 000 records
    per host).

GET /telemetry/recent
    Returns ProcessedRecords from the last N minutes for one or all hosts.

GET /telemetry/hosts
    Returns a health snapshot for every known host.

In-memory store layout (app.state.telemetry_store):
    dict[host_id, deque[ProcessedRecord]]  — maxlen=1 000 per host
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from loguru import logger

from backend.api.dependencies import (
    get_anomaly_detector,
    get_feature_extractor,
    severity_recommendation,
)
from backend.db import timeseries as db
from backend.api.schemas import (
    AnomalyRecord,
    CombinedAnomalyResultSchema,
    HostStatusResponse,
    IFResultSchema,
    LSTMResultSchema,
    TelemetryIngestRequest,
    TelemetryIngestResponse,
    TelemetryRecentResponse,
)
from backend.models.feature_engineering import FeatureExtractor
from backend.models.lstm_model import AnomalyDetector, CombinedAnomalyResult
from backend.pipeline.kinesis_consumer import TelemetryRecord
from backend.pipeline.preprocessor import ProcessedRecord, preprocess

router = APIRouter()

# ─── Store constants ──────────────────────────────────────────────────────────

_TELEMETRY_MAXLEN: int = 1_000
_ANOMALY_MAXLEN: int = 500


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _combined_to_schema(result: CombinedAnomalyResult) -> CombinedAnomalyResultSchema:
    """Convert a CombinedAnomalyResult dataclass to its Pydantic schema."""
    return CombinedAnomalyResultSchema(
        is_anomaly=result.is_anomaly,
        combined_score=result.combined_score,
        severity=result.severity,
        lstm_result=LSTMResultSchema(
            is_anomaly=result.lstm_result.is_anomaly,
            anomaly_score=result.lstm_result.anomaly_score,
            reconstruction_error=result.lstm_result.reconstruction_error,
            threshold_used=result.lstm_result.threshold_used,
            per_feature_errors=result.lstm_result.per_feature_errors,
            worst_feature=result.lstm_result.worst_feature,
            inference_time_ms=result.lstm_result.inference_time_ms,
            model_version=result.lstm_result.model_version,
        ),
        if_result=IFResultSchema(
            is_anomaly=result.if_result.is_anomaly,
            anomaly_score=result.if_result.anomaly_score,
            raw_score=result.if_result.raw_score,
            confidence=result.if_result.confidence,
            top_contributing_features=result.if_result.top_contributing_features,
            host_id=result.if_result.host_id,
            timestamp=result.if_result.timestamp,
            model_version=result.if_result.model_version,
            inference_time_ms=result.if_result.inference_time_ms,
        ),
        detection_method=result.detection_method,
        worst_feature=result.worst_feature,
        top_contributing_features=result.top_contributing_features,
        timestamp=result.timestamp,
    )


def _store_anomaly(
    request: Request,
    combined: CombinedAnomalyResult,
    host_id: str,
) -> None:
    """
    Persist a detected anomaly to the in-memory anomaly store.

    Creates an AnomalyRecord Pydantic model and pushes it to the per-host
    deque (capped at _ANOMALY_MAXLEN).
    """
    store: dict[str, deque] = request.app.state.anomaly_store
    if host_id not in store:
        store[host_id] = deque(maxlen=_ANOMALY_MAXLEN)

    record = AnomalyRecord(
        record_id=str(uuid.uuid4()),
        host_id=host_id,
        detected_at=datetime.now(timezone.utc),
        severity=combined.severity,
        combined_score=combined.combined_score,
        is_anomaly=combined.is_anomaly,
        detection_method=combined.detection_method,
        worst_feature=combined.worst_feature,
        top_contributing_features=combined.top_contributing_features,
        lstm_result=LSTMResultSchema(
            is_anomaly=combined.lstm_result.is_anomaly,
            anomaly_score=combined.lstm_result.anomaly_score,
            reconstruction_error=combined.lstm_result.reconstruction_error,
            threshold_used=combined.lstm_result.threshold_used,
            per_feature_errors=combined.lstm_result.per_feature_errors,
            worst_feature=combined.lstm_result.worst_feature,
            inference_time_ms=combined.lstm_result.inference_time_ms,
            model_version=combined.lstm_result.model_version,
        ),
        if_result=IFResultSchema(
            is_anomaly=combined.if_result.is_anomaly,
            anomaly_score=combined.if_result.anomaly_score,
            raw_score=combined.if_result.raw_score,
            confidence=combined.if_result.confidence,
            top_contributing_features=combined.if_result.top_contributing_features,
            host_id=combined.if_result.host_id,
            timestamp=combined.if_result.timestamp,
            model_version=combined.if_result.model_version,
            inference_time_ms=combined.if_result.inference_time_ms,
        ),
        anomaly_timestamp=combined.timestamp,
        recommendation=severity_recommendation(combined.severity),
    )
    store[host_id].append(record)


# ─── POST /telemetry/ingest ───────────────────────────────────────────────────


@router.post(
    "/ingest",
    response_model=TelemetryIngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest a raw telemetry record",
)
async def ingest_telemetry(
    body: TelemetryIngestRequest,
    request: Request,
    feature_extractor: FeatureExtractor = Depends(get_feature_extractor),
    anomaly_detector: Optional[AnomalyDetector] = Depends(get_anomaly_detector),
) -> TelemetryIngestResponse:
    """
    Accept a raw telemetry payload, preprocess it, and optionally run anomaly
    detection when the per-host sliding window is full.

    Processing pipeline:
        1. Build a ``TelemetryRecord`` from the validated request.
        2. Run ``preprocess()`` → ``ProcessedRecord`` (normalisation + health score).
        3. Store in the in-memory telemetry store (deque, maxlen 1 000 per host).
        4. Run ``FeatureExtractor.process()`` — returns ``FeatureVector`` once
           30 records have been buffered for this host.
        5. If a ``FeatureVector`` is ready and models are loaded, run
           ``AnomalyDetector.detect()`` in a thread pool (CPU-bound).
        6. If an anomaly is detected, persist to the anomaly store.

    Args:
        body: Validated telemetry payload.
        request: Incoming HTTP request (provides app.state access).
        feature_extractor: Shared FeatureExtractor from app.state.
        anomaly_detector: AnomalyDetector from app.state, or None.

    Returns:
        TelemetryIngestResponse with record ID, processing status, and any
        anomaly result.
    """
    record_id = str(uuid.uuid4())

    # ── Resolve timestamp ─────────────────────────────────────────────────────
    ts: datetime = body.timestamp or datetime.now(timezone.utc)
    ts_iso = ts.isoformat() if ts.tzinfo else ts.replace(tzinfo=timezone.utc).isoformat()

    # ── Build TelemetryRecord for the preprocessor ────────────────────────────
    raw_record = TelemetryRecord(
        timestamp=ts_iso,
        host_id=body.host_id,
        latency_ms=body.latency_ms,
        packet_loss_pct=body.packet_loss_pct,
        dns_failure_rate=body.dns_failure_rate,
        jitter_ms=body.jitter_ms,
        is_anomaly=False,
        anomaly_type=None,
    )

    # ── Preprocess ────────────────────────────────────────────────────────────
    processed: ProcessedRecord = preprocess(raw_record)

    # ── Store in telemetry store ──────────────────────────────────────────────
    store: dict[str, deque] = request.app.state.telemetry_store
    if body.host_id not in store:
        store[body.host_id] = deque(maxlen=_TELEMETRY_MAXLEN)
    store[body.host_id].append(processed)
    request.app.state.total_records_processed += 1

    # ── Persist to DB (fire-and-forget, never blocks ingest) ─────────────────
    if request.app.state.db is not None:
        try:
            await db.insert_telemetry(processed)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB telemetry insert failed: {e}", e=exc)

    # ── Feature extraction ────────────────────────────────────────────────────
    feature_vector = await asyncio.to_thread(feature_extractor.process, processed)
    window_ready = feature_vector is not None

    logger.debug(
        "Telemetry ingested | record_id={rid} host={host} health={h:.4f} "
        "window_ready={wr}",
        rid=record_id,
        host=body.host_id,
        h=processed.composite_health_score,
        wr=window_ready,
    )

    # ── Anomaly detection ─────────────────────────────────────────────────────
    anomaly_detected = False
    anomaly_schema: Optional[CombinedAnomalyResultSchema] = None
    alert_id: Optional[str] = None

    if feature_vector is not None and anomaly_detector is not None:
        combined: CombinedAnomalyResult = await asyncio.to_thread(
            anomaly_detector.detect, feature_vector
        )
        anomaly_detected = combined.is_anomaly

        if combined.is_anomaly:
            logger.warning(
                "Anomaly detected via ingest | host={host} severity={sev} "
                "score={score:.4f} method={method}",
                host=body.host_id,
                sev=combined.severity,
                score=combined.combined_score,
                method=combined.detection_method,
            )
            _store_anomaly(request, combined, body.host_id)

            # ── Persist anomaly to DB ─────────────────────────────────────────
            if request.app.state.db is not None:
                try:
                    await db.insert_anomaly(body.host_id, combined)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("DB anomaly insert failed: {e}", e=exc)

            # ── Alert evaluation ──────────────────────────────────────────────
            alert_manager = getattr(request.app.state, "alert_manager", None)
            if alert_manager is not None:
                alert = await asyncio.to_thread(
                    alert_manager.evaluate, body.host_id, combined
                )
                if alert is not None:
                    alert_id = alert.alert_id
                    logger.info(
                        "Alert fired via ingest | host={host} alert_id={aid} "
                        "severity={sev}",
                        host=body.host_id,
                        aid=alert_id,
                        sev=combined.severity,
                    )

        anomaly_schema = _combined_to_schema(combined)

    # ── Build response ────────────────────────────────────────────────────────
    if not window_ready:
        fill = feature_extractor._buffer.fill_level(body.host_id)
        message = (
            f"Record buffered ({fill}/30). Anomaly detection starts at 30 records."
        )
    elif anomaly_detector is None:
        message = "Record processed. Models not loaded — anomaly detection unavailable."
    elif anomaly_detected:
        message = f"Anomaly detected! Severity: {anomaly_schema.severity}."
    else:
        message = "Record processed. No anomaly detected."

    return TelemetryIngestResponse(
        record_id=record_id,
        host_id=body.host_id,
        processed=True,
        anomaly_detected=anomaly_detected,
        anomaly_result=anomaly_schema if anomaly_detected else None,
        window_ready=window_ready,
        message=message,
        alert_id=alert_id,
    )


# ─── GET /telemetry/recent ────────────────────────────────────────────────────


@router.get(
    "/recent",
    response_model=TelemetryRecentResponse,
    summary="Fetch recent telemetry records",
)
async def get_recent_telemetry(
    request: Request,
    host_id: Optional[str] = Query(None, description="Filter to a single host"),
    minutes: int = Query(5, ge=1, le=60, description="Time window in minutes [1, 60]"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return [1, 1000]"),
) -> TelemetryRecentResponse:
    """
    Return ProcessedRecords from the last ``minutes`` minutes for one or all hosts.

    Records are filtered by their ``timestamp`` field against the current UTC
    clock.  Results are returned newest-first, capped at ``limit``.

    Args:
        request: Provides access to app.state.telemetry_store.
        host_id: Optional host filter.  None returns all hosts.
        minutes: Lookback window in minutes (1–60).
        limit:   Maximum number of records in the response (1–1000).

    Returns:
        TelemetryRecentResponse with matching records and metadata.
    """
    # ── Prefer TimescaleDB when available (persists across restarts) ─────────
    if request.app.state.db is not None and host_id is not None:
        try:
            rows = await db.get_recent_telemetry(host_id, minutes)
            rows = rows[:limit]
            return TelemetryRecentResponse(
                records=rows,
                total_count=len(rows),
                hosts_included=[host_id] if rows else [],
                time_range_minutes=minutes,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("DB telemetry read failed, falling back to memory: {e}", e=exc)

    # ── In-memory fallback ────────────────────────────────────────────────────
    store: dict[str, deque] = request.app.state.telemetry_store
    cutoff_iso = datetime.now(timezone.utc)

    if host_id is not None:
        if host_id not in store:
            return TelemetryRecentResponse(
                records=[],
                total_count=0,
                hosts_included=[],
                time_range_minutes=minutes,
            )
        hosts_to_query = [host_id]
    else:
        hosts_to_query = list(store.keys())

    matching: list[dict] = []
    for hid in hosts_to_query:
        for record in store[hid]:
            try:
                rec_dt = datetime.fromisoformat(record.timestamp)
                if rec_dt.tzinfo is None:
                    rec_dt = rec_dt.replace(tzinfo=timezone.utc)
                age_minutes = (cutoff_iso - rec_dt).total_seconds() / 60.0
                if age_minutes <= minutes:
                    matching.append(record.model_dump())
            except (ValueError, AttributeError):
                continue

    matching.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    matching = matching[:limit]
    included_hosts = sorted({r["host_id"] for r in matching})

    return TelemetryRecentResponse(
        records=matching,
        total_count=len(matching),
        hosts_included=included_hosts,
        time_range_minutes=minutes,
    )


# ─── GET /telemetry/hosts ─────────────────────────────────────────────────────


@router.get(
    "/hosts",
    response_model=list[HostStatusResponse],
    summary="List all active hosts with health status",
)
async def get_hosts(request: Request) -> list[HostStatusResponse]:
    """
    Return a health snapshot for every host that has sent at least one record.

    For each host the response includes:
        - ``record_count``: Total records buffered (up to 1 000).
        - ``latest_health_score``: Composite health of the most recent record.
        - ``latest_latency_ms``: Latency of the most recent record.
        - ``window_ready``: True when the 30-record sliding window is full.
        - ``last_seen``: ISO 8601 timestamp of the most recent record.

    Args:
        request: Provides access to app.state.

    Returns:
        List of HostStatusResponse, one per known host.
    """
    store: dict[str, deque] = request.app.state.telemetry_store
    extractor = request.app.state.feature_extractor

    results: list[HostStatusResponse] = []
    for hid, records in store.items():
        if not records:
            continue
        latest: ProcessedRecord = records[-1]
        results.append(
            HostStatusResponse(
                host_id=hid,
                record_count=len(records),
                latest_health_score=latest.composite_health_score,
                latest_latency_ms=latest.latency_ms,
                window_ready=extractor._buffer.is_ready(hid),
                last_seen=latest.timestamp,
            )
        )

    results.sort(key=lambda h: h.host_id)
    return results
