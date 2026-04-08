"""
FlowWatch AI — FastAPI Shared Dependencies
==========================================
Reusable FastAPI dependency-injection callables:

    verify_api_key        — Validates X-API-Key header; skips /health.
    get_anomaly_detector  — Returns AnomalyDetector from app.state (or None).
    get_feature_extractor — Returns FeatureExtractor from app.state.

All callables are compatible with FastAPI's ``Depends()`` system and can be
composed freely in router-level or endpoint-level dependencies.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from loguru import logger

from backend.models.feature_engineering import FeatureExtractor
from backend.models.lstm_model import AnomalyDetector

# ─── API key scheme definition ────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# ─── Severity → recommendation mapping ───────────────────────────────────────

SEVERITY_RECOMMENDATIONS: dict[str, str] = {
    "critical": "Immediate investigation required — escalate to on-call engineer.",
    "high": "Schedule investigation within 1 hour — review recent changes and logs.",
    "medium": "Monitor closely for the next 30 minutes — check host-level metrics.",
    "low": "Log and continue monitoring — no immediate action required.",
}


def severity_recommendation(severity: str) -> str:
    """Return a rule-based action recommendation for the given severity level."""
    return SEVERITY_RECOMMENDATIONS.get(severity.lower(), "Continue monitoring.")


# ─── Authentication dependency ────────────────────────────────────────────────


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(_api_key_header),
) -> None:
    """
    Validate the ``X-API-Key`` header against the ``API_KEYS`` env var.

    ``API_KEYS`` is a comma-separated list of valid key strings read at
    call time (hot-reloadable without restart).

    Behaviour:
        - ``/health`` path → always allowed (no credentials required).
        - No keys configured in ``API_KEYS`` → bypass auth (dev convenience).
        - Key missing or not in valid set → HTTP 401 + IP logged.

    Args:
        request: Incoming HTTP request (used for client IP in logs).
        api_key: Value of the ``X-API-Key`` header, or ``None`` if absent.

    Raises:
        HTTPException 401: Key missing or invalid.
    """
    if request.url.path == "/health":
        return

    # Dev mode: bypass auth entirely when ENVIRONMENT=development
    if os.getenv("ENVIRONMENT", "development") == "development":
        return

    raw_keys = os.getenv("API_KEYS", "")
    valid_keys = {k.strip() for k in raw_keys.split(",") if k.strip()}

    if not valid_keys:
        # No keys configured → allow all requests
        return

    client_ip = request.client.host if request.client else "unknown"

    if not api_key or api_key not in valid_keys:
        logger.warning(
            "Unauthorized request | ip={ip} path={path} key_present={kp}",
            ip=client_ip,
            path=request.url.path,
            kp=bool(api_key),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ─── Model / extractor dependencies ──────────────────────────────────────────


def get_anomaly_detector(request: Request) -> Optional[AnomalyDetector]:
    """
    Return the :class:`AnomalyDetector` stored in ``app.state`` at startup.

    Returns ``None`` when model artifacts were not found at startup.
    Route handlers that require a trained model should check for ``None``
    and return HTTP 503 Service Unavailable.

    Args:
        request: Incoming HTTP request (provides access to ``app.state``).

    Returns:
        Loaded :class:`AnomalyDetector` or ``None``.
    """
    return getattr(request.app.state, "anomaly_detector", None)


def get_feature_extractor(request: Request) -> FeatureExtractor:
    """
    Return the :class:`FeatureExtractor` stored in ``app.state`` at startup.

    The extractor is always available (no artifacts required) and is safe
    to share across requests — its internal ``WindowBuffer`` is thread-safe.

    Args:
        request: Incoming HTTP request.

    Returns:
        Shared :class:`FeatureExtractor` instance.
    """
    return request.app.state.feature_extractor
