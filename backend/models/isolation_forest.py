"""
FlowWatch AI — Isolation Forest Anomaly Detector
==================================================
Trains, persists, and serves an Isolation Forest model that detects
point-in-time network anomalies from ``FeatureVector`` snapshots.

Position in the pipeline::

    FeatureVector  →  IsolationForestDetector.predict()  →  AnomalyResult
                              ↑
              IsolationForestDetector.train(list[FeatureVector])

The model complements the LSTM autoencoder: Isolation Forest provides fast,
stateless snapshot-level detection while LSTM captures temporal patterns.

Persistence format: a single ``.joblib`` file that bundles the fitted
``sklearn`` model and all training metadata needed for correct inference.
"""

from __future__ import annotations

import doctest
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from loguru import logger
from sklearn.ensemble import IsolationForest as _SklearnIF

from backend.models.feature_engineering import (
    FeatureVector,
    _IF_FEATURE_ORDER,
    WINDOW_SIZE,
)

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH: Path = Path("backend/models/artifacts/isolation_forest.joblib")
"""Default filesystem path for model persistence."""

N_FEATURES: int = len(_IF_FEATURE_ORDER)  # 19
_MODEL_VERSION_PREFIX: str = "1.0.0"

# ─── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class TrainingResult:
    """
    Summary of a completed training run.

    Attributes:
        n_samples:                  Number of FeatureVectors the model was trained on.
        contamination:              Anomaly fraction used to set the decision threshold.
        training_anomaly_rate:      Fraction of training samples flagged as anomalies
                                    by the freshly fitted model.
        feature_importance:         Per-feature std-dev across the training matrix —
                                    higher variance ≈ more discriminative.
        feature_names:              Ordered list of feature names matching
                                    ``feature_importance``.
        training_duration_seconds:  Wall-clock seconds the fit took.
        model_version:              Version string (e.g. ``"1.0.0-20260401-120000"``).
        training_date:              UTC ISO 8601 timestamp of when training finished.
    """

    n_samples: int
    contamination: float
    training_anomaly_rate: float
    feature_importance: list[float]
    feature_names: list[str]
    training_duration_seconds: float
    model_version: str
    training_date: str


@dataclass
class AnomalyResult:
    """
    Inference result for a single ``FeatureVector``.

    Attributes:
        is_anomaly:                 ``True`` when the model classifies the sample
                                    as anomalous (raw prediction == -1).
        anomaly_score:              Normalised score in ``[0.0, 1.0]``.
                                    0.0 = certainly normal, 1.0 = certainly anomalous.
        raw_score:                  Raw ``decision_function`` output (negative = anomalous).
        confidence:                 Distance from the decision boundary, normalised to
                                    ``[0.0, 1.0]``.  0.0 = right at boundary, 1.0 = far away.
        top_contributing_features:  Top-3 feature names with the largest absolute
                                    deviation from their training-set means.
        host_id:                    Forwarded from the input FeatureVector.
        timestamp:                  Forwarded from the input FeatureVector.
        model_version:              Version string of the model that produced this result.
        inference_time_ms:          Wall-clock milliseconds taken for this inference call.
    """

    is_anomaly: bool
    anomaly_score: float
    raw_score: float
    confidence: float
    top_contributing_features: list[str]
    host_id: str
    timestamp: str
    model_version: str
    inference_time_ms: float


# ─── Score normalisation helpers ──────────────────────────────────────────────


def _normalize_score(raw_score: float, score_min: float, score_max: float) -> float:
    """
    Map a raw ``decision_function`` score to a ``[0, 1]`` anomaly probability.

    Isolation Forest convention: more negative score → more anomalous.
    The mapping is a linear rescaling so that::

        score_min  (most anomalous seen in training)  →  1.0
        score_max  (most normal seen in training)     →  0.0
        mid-point                                     →  0.5

    Values outside the training range are clipped to ``[0.0, 1.0]``.

    Args:
        raw_score:  A single ``decision_function`` output value.
        score_min:  Minimum score observed during training (most anomalous end).
        score_max:  Maximum score observed during training (most normal end).

    Returns:
        Normalised anomaly probability in ``[0.0, 1.0]``.

    >>> _normalize_score(0.0, -0.4, 0.4)       # midpoint → 0.5
    0.5
    >>> _normalize_score(-0.4, -0.4, 0.4)      # most anomalous → 1.0
    1.0
    >>> _normalize_score(0.4, -0.4, 0.4)       # most normal → 0.0
    0.0
    >>> _normalize_score(-1.0, -0.4, 0.4)      # beyond range → clipped to 1.0
    1.0
    >>> _normalize_score(0.0, 0.0, 0.0)        # degenerate equal bounds → 0.5
    0.5
    """
    if score_max <= score_min:
        return 0.5
    normalized = (score_max - raw_score) / (score_max - score_min)
    return float(np.clip(normalized, 0.0, 1.0))


def _compute_confidence(anomaly_score: float) -> float:
    """
    Convert a normalised anomaly score to a confidence value.

    Confidence measures how far the sample is from the decision boundary
    (0.5).  A sample sitting exactly on the boundary has confidence 0.0;
    a clear-cut normal or anomalous sample has confidence approaching 1.0.

    Formula::  ``abs(anomaly_score - 0.5) * 2``

    Args:
        anomaly_score: Normalised anomaly probability in ``[0, 1]``.

    Returns:
        Confidence in ``[0.0, 1.0]``.

    >>> _compute_confidence(0.5)    # on the boundary
    0.0
    >>> _compute_confidence(1.0)    # certain anomaly
    1.0
    >>> _compute_confidence(0.0)    # certain normal
    1.0
    >>> round(_compute_confidence(0.75), 4)
    0.5
    """
    return float(abs(anomaly_score - 0.5) * 2.0)


def _top_deviating_features(
    current: np.ndarray,
    training_mean: np.ndarray,
    n: int = 3,
) -> list[str]:
    """
    Return the names of the *n* features whose current values deviate most
    from their training-set means (by absolute difference).

    Args:
        current:       1-D array of shape ``(N_FEATURES,)`` for one sample.
        training_mean: 1-D array of shape ``(N_FEATURES,)`` — per-feature
                       mean computed over the training matrix.
        n:             Number of top features to return (default 3).

    Returns:
        List of feature name strings, length ``min(n, N_FEATURES)``.
    """
    deviations = np.abs(current - training_mean)
    top_indices = np.argsort(deviations)[::-1][:n]
    return [_IF_FEATURE_ORDER[i] for i in top_indices]


# ─── Isolation Forest detector ────────────────────────────────────────────────


class IsolationForestDetector:
    """
    Production wrapper around ``sklearn.ensemble.IsolationForest``.

    Responsibilities:
        - Train on a list of :class:`FeatureVector` objects.
        - Persist/load the model and all inference metadata.
        - Predict single vectors and batches with rich result objects.
        - Approximate online learning via reservoir sampling.

    Example::

        detector = IsolationForestDetector()
        training_data = generate_training_data(n_samples=5000)
        result = detector.train(training_data)
        print(f"Trained on {result.n_samples} samples")

        fv = ...  # FeatureVector from FeatureExtractor
        anomaly = detector.predict(fv)
        print(anomaly.is_anomaly, anomaly.anomaly_score)
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        max_samples: str = "auto",
        random_state: int = 42,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        """
        Initialise a (not yet trained) detector.

        Args:
            contamination:  Expected fraction of anomalies in training data.
                            Sets the decision threshold for ``predict()``.
            n_estimators:   Number of isolation trees.  More trees = stabler
                            scores at the cost of training/inference time.
            max_samples:    Samples to draw per tree.  ``"auto"`` uses
                            ``min(256, n_training_samples)``.
            random_state:   Seed for reproducibility.
            model_path:     Default filesystem path used by ``save()`` and
                            ``load()``.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.model_path = model_path

        # Populated by train() / load()
        self._model: Optional[_SklearnIF] = None
        self._training_mean: Optional[np.ndarray] = None
        self._score_min: float = 0.0
        self._score_max: float = 0.0
        self._metadata: dict[str, Any] = {}

        # Online-learning state
        self._reservoir: list[FeatureVector] = []
        self._reservoir_total_seen: int = 0
        self._reservoir_max: int = 1000

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, feature_vectors: list[FeatureVector]) -> TrainingResult:
        """
        Fit the Isolation Forest on *feature_vectors* and persist the model.

        Steps:
            1. Extract ``to_isolation_forest_input()`` from every vector →
               build ``(n_samples, 19)`` matrix.
            2. Fit ``IsolationForest``.
            3. Run ``decision_function`` on the training set to calibrate the
               score normalisation bounds.
            4. Record training metrics.
            5. Save to ``model_path``.

        Args:
            feature_vectors: Non-empty list of :class:`FeatureVector` instances.

        Returns:
            A :class:`TrainingResult` with all training metrics.

        Raises:
            ValueError: If ``feature_vectors`` is empty.
        """
        if not feature_vectors:
            raise ValueError("feature_vectors must not be empty")

        t0 = time.perf_counter()
        logger.info(
            "Training Isolation Forest | n_samples={n} contamination={c}",
            n=len(feature_vectors),
            c=self.contamination,
        )

        # ── Build feature matrix ──────────────────────────────────────────────
        X = np.array(
            [fv.to_isolation_forest_input() for fv in feature_vectors],
            dtype=np.float64,
        )  # shape: (n_samples, 19)

        # ── Fit model ────────────────────────────────────────────────────────
        self._model = _SklearnIF(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X)

        # ── Calibrate score normalisation bounds ──────────────────────────────
        train_scores = self._model.decision_function(X)
        self._score_min = float(train_scores.min())
        self._score_max = float(train_scores.max())

        # ── Per-feature training mean (for deviation analysis) ────────────────
        self._training_mean = X.mean(axis=0)

        # ── Training metrics ──────────────────────────────────────────────────
        raw_predictions = self._model.predict(X)
        anomaly_rate = float(np.mean(raw_predictions == -1))
        feature_importance = X.std(axis=0).tolist()

        elapsed = time.perf_counter() - t0
        version = self._build_version()

        self._metadata = {
            "training_date": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(feature_vectors),
            "contamination": self.contamination,
            "feature_names": list(_IF_FEATURE_ORDER),
            "training_anomaly_rate": anomaly_rate,
            "model_version": version,
            "score_min": self._score_min,
            "score_max": self._score_max,
            "training_mean": self._training_mean.tolist(),
        }

        result = TrainingResult(
            n_samples=len(feature_vectors),
            contamination=self.contamination,
            training_anomaly_rate=anomaly_rate,
            feature_importance=feature_importance,
            feature_names=list(_IF_FEATURE_ORDER),
            training_duration_seconds=round(elapsed, 4),
            model_version=version,
            training_date=self._metadata["training_date"],
        )

        logger.info(
            "Training complete | version={v} duration={d:.2f}s "
            "anomaly_rate={ar:.2%} score_range=[{smin:.4f}, {smax:.4f}]",
            v=version,
            d=elapsed,
            ar=anomaly_rate,
            smin=self._score_min,
            smax=self._score_max,
        )

        self.save()
        return result

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feature_vector: FeatureVector) -> AnomalyResult:
        """
        Classify a single :class:`FeatureVector` and return a rich result.

        Args:
            feature_vector: A fully-computed feature snapshot from
                            :class:`FeatureExtractor`.

        Returns:
            :class:`AnomalyResult` with score, confidence, and contributing
            features.

        Raises:
            RuntimeError: If the model has not been trained or loaded yet.
        """
        self._assert_trained()
        t0 = time.perf_counter()

        x = feature_vector.to_isolation_forest_input().reshape(1, -1)

        raw_label: int = int(self._model.predict(x)[0])           # -1 or 1
        raw_score: float = float(self._model.decision_function(x)[0])

        anomaly_score = _normalize_score(raw_score, self._score_min, self._score_max)
        confidence = _compute_confidence(anomaly_score)

        top_features = _top_deviating_features(
            feature_vector.to_isolation_forest_input(),
            self._training_mean,  # type: ignore[arg-type]
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = AnomalyResult(
            is_anomaly=raw_label == -1,
            anomaly_score=round(anomaly_score, 6),
            raw_score=round(raw_score, 6),
            confidence=round(confidence, 6),
            top_contributing_features=top_features,
            host_id=feature_vector.host_id,
            timestamp=feature_vector.timestamp,
            model_version=self._metadata.get("model_version", "unknown"),
            inference_time_ms=round(elapsed_ms, 3),
        )

        if result.is_anomaly:
            logger.warning(
                "⚠ IF ANOMALY | host={host} score={score:.4f} "
                "confidence={conf:.4f} top_features={feats}",
                host=result.host_id,
                score=result.anomaly_score,
                conf=result.confidence,
                feats=result.top_contributing_features,
            )
        else:
            logger.debug(
                "IF normal | host={host} score={score:.4f}",
                host=result.host_id,
                score=result.anomaly_score,
            )

        return result

    def predict_batch(self, feature_vectors: list[FeatureVector]) -> list[AnomalyResult]:
        """
        Classify multiple feature vectors using vectorised numpy operations.

        All score normalisations and deviation analyses are batched for
        efficiency, avoiding a Python loop over ``predict()``.

        Args:
            feature_vectors: List of :class:`FeatureVector` instances to classify.

        Returns:
            List of :class:`AnomalyResult` in the same order as the input.

        Raises:
            RuntimeError: If the model has not been trained or loaded yet.
            ValueError:   If ``feature_vectors`` is empty.
        """
        self._assert_trained()
        if not feature_vectors:
            raise ValueError("feature_vectors must not be empty")

        t0 = time.perf_counter()

        X = np.array(
            [fv.to_isolation_forest_input() for fv in feature_vectors],
            dtype=np.float64,
        )

        raw_labels: np.ndarray = self._model.predict(X)           # shape (n,)
        raw_scores: np.ndarray = self._model.decision_function(X) # shape (n,)

        # Vectorised normalisation
        anomaly_scores = np.clip(
            (self._score_max - raw_scores) / (self._score_max - self._score_min + 1e-12),
            0.0,
            1.0,
        )
        confidences = np.abs(anomaly_scores - 0.5) * 2.0

        # Per-sample top-contributing features
        deviations = np.abs(X - self._training_mean)               # (n, 19)
        top_indices = np.argsort(deviations, axis=1)[:, ::-1][:, :3]

        elapsed_per_sample_ms = (time.perf_counter() - t0) * 1000.0 / len(feature_vectors)

        results: list[AnomalyResult] = []
        for i, fv in enumerate(feature_vectors):
            top_feats = [_IF_FEATURE_ORDER[idx] for idx in top_indices[i]]
            results.append(
                AnomalyResult(
                    is_anomaly=bool(raw_labels[i] == -1),
                    anomaly_score=round(float(anomaly_scores[i]), 6),
                    raw_score=round(float(raw_scores[i]), 6),
                    confidence=round(float(confidences[i]), 6),
                    top_contributing_features=top_feats,
                    host_id=fv.host_id,
                    timestamp=fv.timestamp,
                    model_version=self._metadata.get("model_version", "unknown"),
                    inference_time_ms=round(elapsed_per_sample_ms, 3),
                )
            )

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(
            "Batch inference | n={n} anomalies={a} ({pct:.1%})",
            n=len(results),
            a=anomaly_count,
            pct=anomaly_count / len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> None:
        """
        Serialise the fitted model and all inference metadata to a single
        ``.joblib`` file.

        The saved artifact contains:
            - ``model``: The fitted ``sklearn`` IsolationForest.
            - ``metadata``: Training date, contamination, feature names, etc.
            - ``score_min`` / ``score_max``: For score normalisation.
            - ``training_mean``: For deviation analysis.

        Args:
            path: Override the default ``model_path``.  Creates parent
                  directories if they do not exist.

        Raises:
            RuntimeError: If called before ``train()``.
        """
        self._assert_trained()
        target = Path(path or self.model_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self._model,
            "metadata": self._metadata,
            "score_min": self._score_min,
            "score_max": self._score_max,
            "training_mean": self._training_mean,
        }
        joblib.dump(artifact, target)
        logger.info("Model saved to {path}", path=target)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "IsolationForestDetector":
        """
        Deserialise a previously saved detector from disk.

        Args:
            path: Path to a ``.joblib`` artifact produced by ``save()``.
                  Defaults to ``DEFAULT_MODEL_PATH``.

        Returns:
            A fully initialised :class:`IsolationForestDetector` ready for
            inference.

        Raises:
            FileNotFoundError: If the artifact does not exist at *path*.
        """
        target = Path(path or DEFAULT_MODEL_PATH)
        if not target.exists():
            raise FileNotFoundError(f"Model artifact not found: {target}")

        artifact = joblib.load(target)
        metadata: dict[str, Any] = artifact["metadata"]

        detector = cls(
            contamination=metadata.get("contamination", 0.05),
            model_path=target,
        )
        detector._model = artifact["model"]
        detector._score_min = float(artifact["score_min"])
        detector._score_max = float(artifact["score_max"])
        detector._training_mean = np.array(artifact["training_mean"], dtype=np.float64)
        detector._metadata = metadata

        logger.info(
            "Model loaded from {path} | version={v} trained_on={date}",
            path=target,
            v=metadata.get("model_version", "unknown"),
            date=metadata.get("training_date", "unknown"),
        )
        return detector

    def is_trained(self) -> bool:
        """
        Return ``True`` if the model has been fitted (via ``train()`` or
        ``load()``).

        Returns:
            Boolean readiness flag.

        >>> IsolationForestDetector().is_trained()
        False
        """
        return self._model is not None

    # ------------------------------------------------------------------
    # Online / approximate learning
    # ------------------------------------------------------------------

    def update(
        self,
        new_vectors: list[FeatureVector],
        max_samples: int = 1000,
    ) -> None:
        """
        Approximate online learning via reservoir sampling (Vitter's Algorithm R).

        ``IsolationForest`` does not support incremental fitting, so instead
        we maintain a reservoir buffer of up to *max_samples* FeatureVectors
        using random replacement to ensure the buffer is always a uniform
        random sample of all vectors seen so far.  When the buffer first
        reaches *max_samples*, we retrain the model from scratch on the
        reservoir.  Subsequent calls continue updating the reservoir and
        re-trigger training each time the buffer accumulates another
        *max_samples* new arrivals.

        Args:
            new_vectors: Newly arrived :class:`FeatureVector` instances.
            max_samples: Maximum reservoir size.  Controls the trade-off
                         between training cost and sample diversity.
        """
        self._reservoir_max = max_samples
        retrain_threshold = max_samples  # retrain every max_samples new arrivals

        for vec in new_vectors:
            n = self._reservoir_total_seen
            self._reservoir_total_seen += 1

            if len(self._reservoir) < max_samples:
                # Fill the reservoir first
                self._reservoir.append(vec)
            else:
                # Algorithm R: replace a random position with probability
                # max_samples / (n + 1)
                j = random.randint(0, n)
                if j < max_samples:
                    self._reservoir[j] = vec

        # Retrain when we cross a new multiple of max_samples
        completed_epochs = self._reservoir_total_seen // retrain_threshold
        last_train_epoch = getattr(self, "_last_train_epoch", 0)

        if completed_epochs > last_train_epoch and len(self._reservoir) == max_samples:
            logger.info(
                "Reservoir sampling triggered retraining | "
                "total_seen={n} reservoir_size={r}",
                n=self._reservoir_total_seen,
                r=len(self._reservoir),
            )
            self.train(list(self._reservoir))
            self._last_train_epoch = completed_epochs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_trained(self) -> None:
        """Raise RuntimeError if the model has not been trained or loaded."""
        if self._model is None:
            raise RuntimeError(
                "Model is not trained.  Call train() or load() first."
            )

    @staticmethod
    def _build_version() -> str:
        """Generate a timestamp-based model version string."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{_MODEL_VERSION_PREFIX}-{stamp}"


# ─── Training-data generator ──────────────────────────────────────────────────


def generate_training_data(n_samples: int = 5000) -> list[FeatureVector]:
    """
    Synthesise a list of :class:`FeatureVector` objects for bootstrapping
    the Isolation Forest before real telemetry data is available.

    Distributions mirror those used by ``kinesis_producer.py``:

    *Normal samples (~95%)*:
        - Rolling means and stds matching the normal operating baselines
          (latency ~45 ms, packet-loss ~0.5 %, jitter ~8 ms).
        - Trends near zero (stable network).
        - Spike counts near zero.

    *Anomaly samples (~5%)*:
        - SPIKE: latency mean and spike count elevated.
        - LOSS: packet-loss mean and loss spike count elevated.
        - DNS: DNS failure rate mean elevated.
        - CASCADE: all metrics simultaneously degraded.

    Args:
        n_samples: Total number of synthetic FeatureVectors to generate.
                   Default is 5 000 (enough for a stable IF estimate).

    Returns:
        List of :class:`FeatureVector` instances with synthetic but
        realistic feature values.
    """
    rng = np.random.default_rng(seed=42)
    vectors: list[FeatureVector] = []
    anomaly_rate = 0.05
    anomaly_types = ["SPIKE", "LOSS", "DNS", "CASCADE"]

    # Pre-build a dummy raw window (only needed for FeatureVector construction;
    # to_isolation_forest_input() uses the scalar features, not the window).
    dummy_window = np.full((WINDOW_SIZE, 5), 0.045, dtype=np.float64)

    for i in range(n_samples):
        is_anomaly = rng.random() < anomaly_rate
        atype = random.choice(anomaly_types) if is_anomaly else None

        # ── Normal baseline values ────────────────────────────────────────────
        rm_lat = float(np.clip(rng.normal(0.045, 0.010), 0.0, 1.0))
        rs_lat = float(np.clip(rng.normal(0.010, 0.003), 0.0, 1.0))
        rm_loss = float(np.clip(rng.exponential(0.005), 0.0, 1.0))
        rs_loss = float(np.clip(rng.normal(0.003, 0.001), 0.0, 1.0))
        rm_dns = float(np.clip(rng.uniform(0.0, 0.050), 0.0, 1.0))
        rm_jitter = float(np.clip(rng.normal(0.040, 0.015), 0.0, 1.0))
        rs_jitter = float(np.clip(rng.normal(0.015, 0.005), 0.0, 1.0))
        lat_trend = float(rng.normal(0.0, 0.001))
        health_trend = float(rng.normal(0.0, 0.001))
        spike_count = float(max(0, int(rng.poisson(0.1))))
        loss_spike_count = float(max(0, int(rng.poisson(0.05))))
        lat_delta = float(rng.normal(0.0, 0.005))
        loss_delta = float(rng.normal(0.0, 0.001))
        dns_delta = float(rng.normal(0.0, 0.002))

        # Cyclic time features — uniformly distributed over a day/week
        hour = rng.integers(0, 24)
        weekday = rng.integers(0, 7)
        import math as _math
        hour_sin = _math.sin(2.0 * _math.pi * hour / 24.0)
        hour_cos = _math.cos(2.0 * _math.pi * hour / 24.0)
        day_sin = _math.sin(2.0 * _math.pi * weekday / 7.0)
        day_cos = _math.cos(2.0 * _math.pi * weekday / 7.0)
        is_biz = 1.0 if (weekday < 5 and 8 <= hour < 18) else 0.0

        # ── Anomaly overrides ─────────────────────────────────────────────────
        if atype == "SPIKE":
            rm_lat = float(np.clip(rng.uniform(0.45, 0.80), 0.0, 1.0))
            lat_trend = float(rng.uniform(0.005, 0.020))
            spike_count = float(rng.integers(8, 20))

        elif atype == "LOSS":
            rm_loss = float(np.clip(rng.uniform(0.25, 0.45), 0.0, 1.0))
            loss_spike_count = float(rng.integers(8, 20))

        elif atype == "DNS":
            rm_dns = float(np.clip(rng.uniform(0.55, 0.90), 0.0, 1.0))
            dns_delta = float(rng.uniform(0.05, 0.20))

        elif atype == "CASCADE":
            rm_lat = float(np.clip(rng.uniform(0.45, 0.80), 0.0, 1.0))
            rm_loss = float(np.clip(rng.uniform(0.25, 0.45), 0.0, 1.0))
            rm_dns = float(np.clip(rng.uniform(0.55, 0.90), 0.0, 1.0))
            rm_jitter = float(np.clip(rng.uniform(0.30, 0.70), 0.0, 1.0))
            spike_count = float(rng.integers(8, 20))
            loss_spike_count = float(rng.integers(8, 20))
            lat_trend = float(rng.uniform(0.005, 0.020))
            health_trend = float(rng.uniform(0.005, 0.015))

        ts = f"2026-04-01T{hour:02d}:00:00+00:00"
        host_id = f"host-{(i % 5) + 1:02d}"

        vectors.append(
            FeatureVector(
                host_id=host_id,
                timestamp=ts,
                rolling_mean_latency=rm_lat,
                rolling_std_latency=rs_lat,
                rolling_mean_loss=rm_loss,
                rolling_std_loss=rs_loss,
                rolling_mean_dns=rm_dns,
                rolling_mean_jitter=rm_jitter,
                rolling_std_jitter=rs_jitter,
                latency_trend=lat_trend,
                health_score_trend=health_trend,
                spike_count=spike_count,
                loss_spike_count=loss_spike_count,
                latency_delta=lat_delta,
                loss_delta=loss_delta,
                dns_delta=dns_delta,
                hour_sin=hour_sin,
                hour_cos=hour_cos,
                day_sin=day_sin,
                day_cos=day_cos,
                is_business_hours=is_biz,
                _raw_window=dummy_window.copy(),
            )
        )

    n_anomalies = sum(1 for v in vectors if v.spike_count > 5 or v.loss_spike_count > 5)
    logger.info(
        "Generated {n} synthetic training vectors "
        "(approx {a} anomalies, {pct:.1%} rate)",
        n=len(vectors),
        a=n_anomalies,
        pct=n_anomalies / len(vectors),
    )
    return vectors


# ─── Doctest runner ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = doctest.testmod(verbose=True)
    raise SystemExit(0 if results.failed == 0 else 1)
