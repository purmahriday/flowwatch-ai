"""
FlowWatch AI — FastAPI Application Entrypoint
=============================================
Creates the FastAPI app, wires middleware, mounts all routers, and manages
the ML model lifecycle via an async lifespan context manager.

Startup sequence
----------------
1. Initialise in-memory telemetry and anomaly stores on ``app.state``.
2. Create a shared ``FeatureExtractor`` instance.
3. Attempt to load both model artifacts (LSTM + Isolation Forest) via
   ``AnomalyDetector``.  If artifacts are missing a warning is logged and
   ``app.state.anomaly_detector`` is set to ``None``; inference endpoints
   will return HTTP 503 until models are trained.
4. Log total startup time and model versions.

All routes except ``GET /health`` require a valid ``X-API-Key`` header.
The key is read from the ``API_KEYS`` environment variable (comma-separated
list).  If ``API_KEYS`` is empty, auth is bypassed (development convenience).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.alerting.alert_manager import AlertManager
from backend.api.dependencies import verify_api_key
from backend.db import timeseries as db
from backend.models.feature_engineering import FeatureExtractor
from backend.models.isolation_forest import DEFAULT_MODEL_PATH as IF_MODEL_PATH
from backend.models.lstm_model import DEFAULT_LSTM_PATH, AnomalyDetector

# ─── Module-level startup timestamp ──────────────────────────────────────────

_START_TIME: float = time.monotonic()

# ─── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async lifespan context manager — runs once at startup and once at shutdown.

    Startup:
        - Allocates in-memory stores (telemetry, anomalies, counters).
        - Instantiates FeatureExtractor (always succeeds).
        - Loads AnomalyDetector via asyncio.to_thread so the event loop is
          not blocked during model deserialisation.
        - Logs versions and elapsed startup time.

    Shutdown:
        - Logs a graceful shutdown message (clean-up is implicit; Python GC
          handles in-memory objects).
    """
    startup_t0 = time.monotonic()
    logger.info("FlowWatch AI starting up…")

    # ── In-memory stores ──────────────────────────────────────────────────────
    # keyed by host_id → deque[ProcessedRecord], capped at 1 000 records/host
    app.state.telemetry_store: dict[str, deque] = {}
    # keyed by host_id → deque[AnomalyRecord], capped at 500 records/host
    app.state.anomaly_store: dict[str, deque] = {}
    # global ingestion counter (mutated only on the event loop thread)
    app.state.total_records_processed: int = 0

    # ── Feature extractor ─────────────────────────────────────────────────────
    app.state.feature_extractor = FeatureExtractor()
    logger.info("FeatureExtractor initialised")

    # ── Alert manager ─────────────────────────────────────────────────────────
    app.state.alert_manager = AlertManager()  # auto-detects CloudWatch from env

    # ── ML models ─────────────────────────────────────────────────────────────
    app.state.anomaly_detector: Any = None
    app.state.lstm_version: str = "not loaded"
    app.state.if_version: str = "not loaded"

    try:
        detector: AnomalyDetector = await asyncio.to_thread(
            _load_anomaly_detector
        )
        app.state.anomaly_detector = detector
        app.state.lstm_version = detector._lstm._model_version
        app.state.if_version = detector._if._metadata.get(
            "model_version", "unknown"
        )
        logger.info(
            "Models loaded | lstm_version={lv} if_version={iv}",
            lv=app.state.lstm_version,
            iv=app.state.if_version,
        )
    except FileNotFoundError as exc:
        logger.warning(
            "Model artifact not found — inference disabled until models are "
            "trained. Missing: {err}",
            err=exc,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Unexpected error loading models — inference disabled. Error: {err}",
            err=exc,
        )

    # ── TimescaleDB (optional) ────────────────────────────────────────────────
    app.state.db = None
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            await db.init_db(database_url)
            app.state.db = True  # sentinel — actual pool lives in db module
            logger.info("TimescaleDB connected")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TimescaleDB unavailable — running in-memory only. Error: {e}", e=exc
            )
    else:
        logger.warning("DATABASE_URL not set — running without database")

    elapsed = time.monotonic() - startup_t0
    logger.info(
        "Startup complete in {elapsed:.3f}s | models_loaded={ml} db={db}",
        elapsed=elapsed,
        ml=app.state.anomaly_detector is not None,
        db=app.state.db is not None,
    )

    yield  # ─── Application serves requests ───────────────────────────────────

    if app.state.db is not None:
        await db.close_db()
    logger.info("FlowWatch AI shutting down gracefully.")


def _load_anomaly_detector() -> AnomalyDetector:
    """
    Synchronous model loader — executed in a thread pool via asyncio.to_thread.

    Raises:
        FileNotFoundError: If either model artifact is absent.
    """
    return AnomalyDetector(
        lstm_path=DEFAULT_LSTM_PATH,
        if_path=IF_MODEL_PATH,
    )


# ─── Application factory ──────────────────────────────────────────────────────

app = FastAPI(
    title="FlowWatch AI",
    version="1.0.0",
    description=(
        "Real-time network monitoring and ML-powered anomaly detection API. "
        "Ingests live telemetry, runs an LSTM + Isolation Forest ensemble, "
        "and provides LLM-assisted root cause analysis via Claude."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS middleware ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Lock down in production; open for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global exception handler ─────────────────────────────────────────────────


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Catch all unhandled exceptions, log them, and return a safe 500 response.

    Never exposes internal stack traces or implementation details to clients.
    """
    logger.exception(
        "Unhandled exception | method={m} path={p} error_type={et}",
        m=request.method,
        p=request.url.path,
        et=type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": (
                "An internal server error occurred. "
                "Please try again later or contact support."
            ),
        },
    )


# ─── Health endpoint ──────────────────────────────────────────────────────────


@app.get(
    "/health",
    tags=["Health"],
    summary="Liveness and readiness probe",
    response_description="Service health snapshot",
)
async def health(request: Request) -> dict:
    """
    Returns service health status.

    No authentication required — safe for load-balancer liveness probes.

    Returns:
        JSON object with:
            - ``status``: Always ``"ok"`` when the process is alive.
            - ``uptime_seconds``: Seconds since the process started.
            - ``models_loaded``: True when both ML models are ready.
            - ``lstm_version``: LSTM artifact version string.
            - ``if_version``: Isolation Forest artifact version string.
            - ``total_records_processed``: Cumulative ingest counter.
    """
    return {
        "status": "ok",
        "uptime_seconds": round(time.monotonic() - _START_TIME, 2),
        "models_loaded": request.app.state.anomaly_detector is not None,
        "lstm_version": request.app.state.lstm_version,
        "if_version": request.app.state.if_version,
        "total_records_processed": request.app.state.total_records_processed,
    }


# ─── Router mounting ──────────────────────────────────────────────────────────

from backend.api.routes import telemetry as _telemetry_routes  # noqa: E402
from backend.api.routes import anomalies as _anomaly_routes  # noqa: E402
from backend.api.routes import assistant as _assistant_routes  # noqa: E402

_auth_dep = [Depends(verify_api_key)]

app.include_router(
    _telemetry_routes.router,
    prefix="/telemetry",
    tags=["Telemetry"],
    dependencies=_auth_dep,
)
app.include_router(
    _anomaly_routes.router,
    prefix="/anomalies",
    tags=["Anomalies"],
    dependencies=_auth_dep,
)
app.include_router(
    _anomaly_routes.alerts_router,
    prefix="/alerts",
    tags=["Alerts"],
    dependencies=_auth_dep,
)
app.include_router(
    _assistant_routes.router,
    prefix="/assistant",
    tags=["Assistant"],
    dependencies=_auth_dep,
)
