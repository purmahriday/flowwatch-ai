"""
FlowWatch AI — LSTM Autoencoder Anomaly Detector
==================================================
Trains, persists, and serves a PyTorch LSTM Autoencoder that detects
temporal anomalies in 30-step network telemetry windows by measuring
reconstruction error.

Design rationale:
    An LSTM autoencoder is trained *only* on normal traffic.  At inference
    time, anomalous windows produce high reconstruction error because the
    model has never learned to compress/reconstruct irregular patterns.
    The 95th-percentile training error is used as the decision threshold.

Architecture (both encoder and decoder are 2-layer LSTMs):

    Input  (B, 30, 4)
    ↓ Encoder LSTM-1   (4  → 64 hidden)
    ↓ Dropout(0.2)
    ↓ Encoder LSTM-2   (64 → 32 hidden)  ← bottleneck hidden state
    ↓ Repeat 30 ×      (B, 30, 32)
    ↓ Decoder LSTM-3   (32 → 32 hidden)
    ↓ Dropout(0.2)
    ↓ Decoder LSTM-4   (32 → 64 hidden)
    ↓ Linear           (64 → 4)
    Output (B, 30, 4)  — reconstructed window

Combined detector (AnomalyDetector) fuses LSTM and Isolation Forest scores
in a 60/40 weighted average to leverage complementary detection strengths.

Persistence:
    LSTMTrainer  → saves TorchScript + metadata to ``lstm_model.pt``
    LSTMDetector → loads and serves from the same artifact
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset, random_split

from backend.models.feature_engineering import FeatureVector, WINDOW_SIZE
from backend.models.isolation_forest import AnomalyResult, IsolationForestDetector

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_LSTM_PATH: Path = Path("backend/models/artifacts/lstm_model.pt")
"""Default filesystem path used by the trainer and detector."""

N_SEQUENCE_FEATURES: int = 4
"""Input feature count per timestep: latency, loss, dns, jitter (normalised)."""

_FEATURE_NAMES: tuple[str, ...] = ("latency", "loss", "dns", "jitter")
"""Human-readable names matching the 4 LSTM input columns."""

_MODEL_VERSION_PREFIX: str = "1.0.0"

# ─── Result dataclasses ───────────────────────────────────────────────────────


@dataclass
class LSTMTrainingResult:
    """
    Summary of a completed LSTM training run.

    Attributes:
        n_samples_train:            Number of windows in the training split.
        n_samples_val:              Number of windows in the validation split.
        best_val_loss:              Lowest validation MSE observed during training.
        epochs_trained:             Actual epochs run before early stopping.
        threshold:                  95th-percentile reconstruction error on the
                                    full normal training set — used as the
                                    anomaly decision boundary.
        error_mean:                 Mean reconstruction error on training data
                                    (used for z-score normalisation).
        error_std:                  Std-dev of reconstruction errors on training
                                    data (used for z-score normalisation).
        training_duration_seconds:  Wall-clock training time.
        model_version:              Timestamp-based version string.
        device_used:                ``"cuda"`` or ``"cpu"``.
        train_losses:               Per-epoch training MSE losses.
        val_losses:                 Per-epoch validation MSE losses.
    """

    n_samples_train: int
    n_samples_val: int
    best_val_loss: float
    epochs_trained: int
    threshold: float
    error_mean: float
    error_std: float
    training_duration_seconds: float
    model_version: str
    device_used: str
    train_losses: list[float]
    val_losses: list[float]


@dataclass
class LSTMResult:
    """
    Inference result for a single :class:`FeatureVector` from the LSTM detector.

    Attributes:
        is_anomaly:             ``True`` when reconstruction error exceeds threshold.
        anomaly_score:          Normalised score in ``[0, 1]`` via z-score + sigmoid.
        reconstruction_error:   Raw mean-squared reconstruction error across all
                                timesteps and features.
        threshold_used:         Decision threshold applied (95th-percentile from training).
        per_feature_errors:     MSE broken down by feature for interpretability.
        worst_feature:          Name of the feature with the highest reconstruction error.
        inference_time_ms:      Wall-clock milliseconds for this call.
        model_version:          Version string of the model that produced this result.
    """

    is_anomaly: bool
    anomaly_score: float
    reconstruction_error: float
    threshold_used: float
    per_feature_errors: dict[str, float]
    worst_feature: str
    inference_time_ms: float
    model_version: str


@dataclass
class CombinedAnomalyResult:
    """
    Fused result from both the LSTM and Isolation Forest detectors.

    Attributes:
        is_anomaly:                 Final anomaly verdict (ensemble decision).
        combined_score:             Weighted score: LSTM×0.6 + IF×0.4.
        severity:                   ``"critical"`` / ``"high"`` / ``"medium"`` / ``"low"``.
        lstm_result:                Full result from the LSTM detector.
        if_result:                  Full result from the Isolation Forest detector.
        detection_method:           Which model(s) raised the alarm.
                                    One of: ``"lstm+if"``, ``"lstm_only"``,
                                    ``"if_only"``, ``"none"``.
        worst_feature:              Most degraded feature (from LSTM per-feature errors).
        top_contributing_features:  Top-3 features from IF deviation analysis.
        timestamp:                  UTC datetime of the triggering record.
    """

    is_anomaly: bool
    combined_score: float
    severity: str
    lstm_result: LSTMResult
    if_result: AnomalyResult
    detection_method: str
    worst_feature: str
    top_contributing_features: list[str]
    timestamp: datetime


# ─── Neural network architecture ──────────────────────────────────────────────


class LSTMAutoencoder(nn.Module):
    """
    Two-layer LSTM encoder–decoder autoencoder for sequence reconstruction.

    The encoder compresses a ``(B, 30, 4)`` window into a single 32-dim
    hidden state (bottleneck).  The decoder expands this bottleneck back
    into a ``(B, 30, 4)`` reconstruction.  Dropout(0.2) is applied between
    the encoder layers and between the decoder layers.

    Training objective: minimise MSE between input and reconstruction.
    Anomaly signal: reconstruction error is low for normal traffic (seen
    during training) and high for novel anomalous patterns.
    """

    def __init__(self, dropout_rate: float = 0.2) -> None:
        """
        Initialise all layers.

        Args:
            dropout_rate: Dropout probability applied between stacked LSTM layers
                          (default 0.2).
        """
        super().__init__()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_lstm1 = nn.LSTM(
            input_size=N_SEQUENCE_FEATURES,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.enc_lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec_lstm3 = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
        )
        self.dec_lstm4 = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.output_linear = nn.Linear(64, N_SEQUENCE_FEATURES)

        # Explicit dropout module applied between stacked layers
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run a full encode–decode pass.

        Args:
            x: Input tensor of shape ``(B, T, 4)`` where ``T=WINDOW_SIZE=30``.

        Returns:
            Reconstructed tensor of shape ``(B, T, 4)`` — same shape as input.
        """
        T = x.size(1)  # sequence length (30)

        # ── Encoder ───────────────────────────────────────────────────────────
        enc_out1, _ = self.enc_lstm1(x)           # (B, T, 64)
        enc_out1 = self.dropout(enc_out1)
        _, (h2, _) = self.enc_lstm2(enc_out1)     # h2: (1, B, 32)

        # Bottleneck: take the final hidden state of the second encoder layer
        bottleneck = h2.squeeze(0)                # (B, 32)

        # ── Decoder ───────────────────────────────────────────────────────────
        # Repeat the bottleneck T times to seed the decoder sequence
        dec_input = bottleneck.unsqueeze(1).repeat(1, T, 1)  # (B, T, 32)

        dec_out1, _ = self.dec_lstm3(dec_input)   # (B, T, 32)
        dec_out1 = self.dropout(dec_out1)
        dec_out2, _ = self.dec_lstm4(dec_out1)    # (B, T, 64)

        output = self.output_linear(dec_out2)     # (B, T, 4)
        return output


# ─── Training ─────────────────────────────────────────────────────────────────


class LSTMTrainer:
    """
    Orchestrates LSTM autoencoder training, threshold calibration, and
    TorchScript export.

    Key design decisions:
        - Only *normal* samples are used for training.  Exposing the model
          to anomalies during training would reduce reconstruction error on
          anomalies and blunt the detection signal.
        - Early stopping (patience=10) prevents overfitting on the relatively
          simple normal-traffic distribution.
        - The anomaly threshold is set to the 95th-percentile reconstruction
          error observed on the full normal training set, ensuring roughly 5 %
          of normal traffic triggers a false positive — consistent with the
          Isolation Forest contamination parameter.
    """

    def __init__(
        self,
        model: Optional[LSTMAutoencoder] = None,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 50,
        patience: int = 10,
        device: Optional[str] = None,
        model_path: Path = DEFAULT_LSTM_PATH,
    ) -> None:
        """
        Initialise the trainer.

        Args:
            model:         Pre-constructed :class:`LSTMAutoencoder`.  A new one
                           is created automatically if ``None``.
            learning_rate: Adam learning rate.
            batch_size:    Mini-batch size for training and validation loaders.
            epochs:        Maximum number of training epochs.
            patience:      Early-stopping patience (epochs without validation
                           loss improvement before halting).
            device:        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
            model_path:    Where to save the trained TorchScript artifact.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model_path = model_path

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: LSTMAutoencoder = model or LSTMAutoencoder()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, feature_vectors: list[FeatureVector]) -> LSTMTrainingResult:
        """
        Fit the autoencoder on the normal-traffic subset of *feature_vectors*.

        Steps:
            1. Filter to normal samples only (``is_anomaly=False`` flag on the
               raw FeatureVector — note: FeatureVector itself doesn't carry this
               flag directly; we rely on the IS_ANOMALY heuristic described below).
            2. Build ``(B, 30, 4)`` window tensors via ``to_lstm_input()``.
            3. Split 80/20 train/val, create shuffled DataLoaders.
            4. Train with Adam + ReduceLROnPlateau + early stopping.
            5. Restore the best checkpoint.
            6. Compute reconstruction errors on the full training set and
               derive threshold (95th percentile), mean, and std.
            7. Trace to TorchScript and save.

        Args:
            feature_vectors: List of :class:`FeatureVector` instances.  The
                             trainer automatically filters to normal samples
                             (``spike_count == 0`` and ``loss_spike_count == 0``
                             as a proxy when the ``is_anomaly`` label is absent).

        Returns:
            :class:`LSTMTrainingResult` with all training metrics.

        Raises:
            ValueError: If no normal samples are found in *feature_vectors*.
        """
        t0 = time.perf_counter()
        version = self._build_version()

        # ── Filter to normal-traffic windows ──────────────────────────────────
        normal_vecs = self._filter_normal(feature_vectors)
        if len(normal_vecs) < 2:
            raise ValueError(
                f"Need at least 2 normal samples for training, got {len(normal_vecs)}."
            )

        logger.info(
            "LSTM training start | device={dev} normal_samples={n} "
            "total_samples={t} epochs={e}",
            dev=self.device,
            n=len(normal_vecs),
            t=len(feature_vectors),
            e=self.epochs,
        )

        # ── Build tensor dataset ──────────────────────────────────────────────
        X = torch.tensor(
            np.array([fv.to_lstm_input() for fv in normal_vecs], dtype=np.float32),
            dtype=torch.float32,
        )  # (N, 30, 4)

        n_val = max(1, int(len(normal_vecs) * 0.2))
        n_train = len(normal_vecs) - n_val
        train_ds, val_ds = random_split(
            TensorDataset(X),
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

        # ── Optimiser, loss, scheduler ────────────────────────────────────────
        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", patience=5, factor=0.5
        )

        # ── Training loop with early stopping ─────────────────────────────────
        best_val_loss = float("inf")
        best_weights: Optional[dict[str, Any]] = None
        epochs_no_improve = 0
        train_losses: list[float] = []
        val_losses: list[float] = []
        actual_epochs = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(train_loader, criterion, optimiser, training=True)
            val_loss = self._run_epoch(val_loader, criterion, None, training=False)

            train_losses.append(round(train_loss, 6))
            val_losses.append(round(val_loss, 6))
            scheduler.step(val_loss)
            actual_epochs = epoch

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "LSTM epoch {ep:03d}/{total} | "
                    "train_loss={tl:.6f} val_loss={vl:.6f}",
                    ep=epoch,
                    total=self.epochs,
                    tl=train_loss,
                    vl=val_loss,
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(
                        "Early stopping at epoch {ep} "
                        "(no val improvement for {p} epochs)",
                        ep=epoch,
                        p=self.patience,
                    )
                    break

        # ── Restore best weights ──────────────────────────────────────────────
        if best_weights is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_weights.items()}
            )

        # ── Threshold calibration ─────────────────────────────────────────────
        errors = self._compute_errors(X)
        threshold = float(np.percentile(errors, 95))
        error_mean = float(np.mean(errors))
        error_std = float(np.std(errors))

        elapsed = time.perf_counter() - t0
        logger.info(
            "LSTM training complete | version={v} epochs={e} "
            "best_val_loss={bvl:.6f} threshold={th:.6f} "
            "error_mean={em:.6f} error_std={es:.6f} duration={d:.1f}s",
            v=version,
            e=actual_epochs,
            bvl=best_val_loss,
            th=threshold,
            em=error_mean,
            es=error_std,
            d=elapsed,
        )

        # ── Save TorchScript artifact ─────────────────────────────────────────
        self._save_artifact(threshold, error_mean, error_std, version)

        return LSTMTrainingResult(
            n_samples_train=n_train,
            n_samples_val=n_val,
            best_val_loss=round(best_val_loss, 6),
            epochs_trained=actual_epochs,
            threshold=round(threshold, 8),
            error_mean=round(error_mean, 8),
            error_std=round(error_std, 8),
            training_duration_seconds=round(elapsed, 3),
            model_version=version,
            device_used=self.device,
            train_losses=train_losses,
            val_losses=val_losses,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.MSELoss,
        optimiser: Optional[torch.optim.Optimizer],
        training: bool,
    ) -> float:
        """
        Run one pass over *loader* in training or evaluation mode.

        Args:
            loader:    DataLoader yielding batches of ``(B, 30, 4)`` tensors.
            criterion: MSE loss function.
            optimiser: Adam optimiser (``None`` for eval pass).
            training:  If ``True``, compute gradients and step the optimiser.

        Returns:
            Mean MSE loss across all batches.
        """
        self.model.train(training)
        total_loss = 0.0
        n_batches = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for (batch,) in loader:
                batch = batch.to(self.device)
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)

                if training and optimiser is not None:
                    optimiser.zero_grad()
                    loss.backward()
                    # Clip gradients to prevent exploding gradients in LSTM
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimiser.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _compute_errors(self, X: torch.Tensor) -> np.ndarray:
        """
        Compute per-sample mean-squared reconstruction error for all windows in X.

        Args:
            X: Tensor of shape ``(N, 30, 4)`` on CPU.

        Returns:
            1-D numpy array of shape ``(N,)`` with per-sample MSE values.
        """
        self.model.eval()
        errors: list[float] = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i : i + self.batch_size].to(self.device)
                recon = self.model(batch)
                # Per-sample MSE: mean over timesteps and features
                mse = ((recon - batch) ** 2).mean(dim=(1, 2))
                errors.extend(mse.cpu().tolist())

        return np.array(errors, dtype=np.float64)

    def _filter_normal(self, vectors: list[FeatureVector]) -> list[FeatureVector]:
        """
        Return only vectors that appear to represent normal network behaviour.

        Since :class:`FeatureVector` doesn't carry a direct ``is_anomaly`` label
        (it's a feature summary, not a raw record), we use ``spike_count == 0``
        and ``loss_spike_count == 0`` as a lightweight proxy for normal samples.
        This ensures the LSTM trains only on clean baseline windows.

        Args:
            vectors: All available FeatureVectors.

        Returns:
            Filtered list containing only normal-looking vectors.
        """
        return [v for v in vectors if v.spike_count == 0 and v.loss_spike_count == 0]

    def _save_artifact(
        self,
        threshold: float,
        error_mean: float,
        error_std: float,
        version: str,
    ) -> None:
        """
        Trace the model to TorchScript and save it alongside metadata.

        The artifact is a single ``.pt`` file containing a dict:
            ``scripted_bytes``:  Raw bytes of the TorchScript model.
            ``threshold``:       95th-percentile reconstruction error.
            ``error_mean``:      Mean reconstruction error (training set).
            ``error_std``:       Std-dev of reconstruction error (training set).
            ``model_version``:   Version string.
            ``training_date``:   UTC ISO 8601 timestamp.

        Args:
            threshold:   Decision boundary for anomaly classification.
            error_mean:  For z-score normalisation.
            error_std:   For z-score normalisation.
            version:     Model version string.
        """
        self.model.eval()
        dummy = torch.zeros(1, WINDOW_SIZE, N_SEQUENCE_FEATURES, dtype=torch.float32)
        dummy = dummy.to(self.device)

        scripted = torch.jit.trace(self.model, dummy)

        buf = io.BytesIO()
        torch.jit.save(scripted, buf)

        artifact: dict[str, Any] = {
            "scripted_bytes": buf.getvalue(),
            "threshold": threshold,
            "error_mean": error_mean,
            "error_std": error_std,
            "model_version": version,
            "training_date": datetime.now(timezone.utc).isoformat(),
        }

        target = Path(self.model_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, target)
        logger.info("LSTM artifact saved to {path}", path=target)

    @staticmethod
    def _build_version() -> str:
        """Generate a timestamp-based version string."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{_MODEL_VERSION_PREFIX}-{stamp}"


# ─── Inference ────────────────────────────────────────────────────────────────


def _error_to_anomaly_score(
    error: float,
    error_mean: float,
    error_std: float,
) -> float:
    """
    Convert a raw reconstruction error to a ``[0, 1]`` anomaly score.

    Uses z-score normalisation followed by clipping to ±3σ and rescaling
    to ``[0, 1]``.

    - Errors near the training mean (z ≈ 0) → score ≈ 0.5
    - Errors 3+ σ above mean (extreme anomaly) → score ≈ 1.0
    - Errors 3+ σ below mean (unusually clean) → score ≈ 0.0

    Args:
        error:       Raw MSE reconstruction error.
        error_mean:  Mean error observed on training data.
        error_std:   Std-dev of errors on training data.

    Returns:
        Anomaly score in ``[0.0, 1.0]``.
    """
    std = error_std if error_std > 1e-10 else 1e-10
    z = (error - error_mean) / std
    # Map z ∈ [-3, 3] → [0, 1]; clip values outside this range
    return float(np.clip((z + 3.0) / 6.0, 0.0, 1.0))


class LSTMDetector:
    """
    Serves inference from a saved LSTM TorchScript artifact.

    Typical usage::

        detector = LSTMDetector.load()
        result = detector.predict(feature_vector)
        if result.is_anomaly:
            alert(result.worst_feature, result.anomaly_score)
    """

    def __init__(self, model_path: Path = DEFAULT_LSTM_PATH) -> None:
        """
        Initialise an (not yet loaded) detector pointing at *model_path*.

        Args:
            model_path: Path to the ``.pt`` artifact produced by
                        :class:`LSTMTrainer`.
        """
        self.model_path = model_path
        self._model: Optional[Any] = None           # torch.jit.ScriptModule
        self._threshold: float = 0.0
        self._error_mean: float = 0.0
        self._error_std: float = 1.0
        self._model_version: str = "unknown"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LSTMDetector":
        """
        Deserialise a previously saved TorchScript artifact and return a
        ready-to-use :class:`LSTMDetector`.

        Args:
            path: Path to the ``.pt`` artifact.  Defaults to
                  ``DEFAULT_LSTM_PATH``.

        Returns:
            Initialised :class:`LSTMDetector`.

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        target = Path(path or DEFAULT_LSTM_PATH)
        if not target.exists():
            raise FileNotFoundError(f"LSTM artifact not found: {target}")

        artifact: dict[str, Any] = torch.load(
            target, map_location="cpu", weights_only=False
        )
        buf = io.BytesIO(artifact["scripted_bytes"])
        scripted_model = torch.jit.load(buf, map_location="cpu")
        scripted_model.eval()

        detector = cls(model_path=target)
        detector._model = scripted_model
        detector._threshold = float(artifact["threshold"])
        detector._error_mean = float(artifact["error_mean"])
        detector._error_std = float(artifact["error_std"])
        detector._model_version = str(artifact.get("model_version", "unknown"))

        logger.info(
            "LSTM model loaded from {path} | version={v} threshold={th:.6f}",
            path=target,
            v=detector._model_version,
            th=detector._threshold,
        )
        return detector

    def is_loaded(self) -> bool:
        """
        Return ``True`` if the TorchScript model has been loaded and is
        ready for inference.

        Returns:
            Boolean readiness flag.
        """
        return self._model is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feature_vector: FeatureVector) -> LSTMResult:
        """
        Run a single-window inference pass.

        Args:
            feature_vector: A :class:`FeatureVector` with a full 30-step window.

        Returns:
            :class:`LSTMResult` with reconstruction error, anomaly flag, and
            per-feature breakdown.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        self._assert_loaded()
        t0 = time.perf_counter()

        # ── Prepare input ─────────────────────────────────────────────────────
        window = feature_vector.to_lstm_input()  # (30, 4) numpy
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 30, 4)

        # ── Forward pass ──────────────────────────────────────────────────────
        with torch.no_grad():
            recon = self._model(x)  # (1, 30, 4)

        x_np = x.squeeze(0).numpy()          # (30, 4)
        recon_np = recon.squeeze(0).numpy()  # (30, 4)

        # ── Error computation ─────────────────────────────────────────────────
        per_step_error = ((recon_np - x_np) ** 2)          # (30, 4)
        per_feature_errors = per_step_error.mean(axis=0)   # (4,)
        overall_error = float(per_feature_errors.mean())

        per_feature_dict = {
            name: round(float(err), 8)
            for name, err in zip(_FEATURE_NAMES, per_feature_errors)
        }
        worst_feature = _FEATURE_NAMES[int(np.argmax(per_feature_errors))]
        anomaly_score = _error_to_anomaly_score(
            overall_error, self._error_mean, self._error_std
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        is_anomaly = overall_error > self._threshold

        result = LSTMResult(
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 6),
            reconstruction_error=round(overall_error, 8),
            threshold_used=round(self._threshold, 8),
            per_feature_errors=per_feature_dict,
            worst_feature=worst_feature,
            inference_time_ms=round(elapsed_ms, 3),
            model_version=self._model_version,
        )

        if is_anomaly:
            logger.warning(
                "⚠ LSTM ANOMALY | host={host} score={score:.4f} "
                "error={err:.6f} threshold={th:.6f} worst={feat}",
                host=feature_vector.host_id,
                score=anomaly_score,
                err=overall_error,
                th=self._threshold,
                feat=worst_feature,
            )
        else:
            logger.debug(
                "LSTM normal | host={host} error={err:.6f}",
                host=feature_vector.host_id,
                err=overall_error,
            )

        return result

    def predict_batch(self, feature_vectors: list[FeatureVector]) -> list[LSTMResult]:
        """
        Classify multiple feature vectors.

        Window tensors are stacked into a single batch for efficient GPU/CPU
        utilisation, then results are decomposed per-sample.

        Args:
            feature_vectors: List of :class:`FeatureVector` instances.

        Returns:
            List of :class:`LSTMResult` in the same order as the input.

        Raises:
            RuntimeError: If the model has not been loaded.
            ValueError:   If *feature_vectors* is empty.
        """
        self._assert_loaded()
        if not feature_vectors:
            raise ValueError("feature_vectors must not be empty")

        t0 = time.perf_counter()

        # Stack all windows into a single tensor: (N, 30, 4)
        windows = np.array(
            [fv.to_lstm_input() for fv in feature_vectors], dtype=np.float32
        )
        X = torch.tensor(windows, dtype=torch.float32)

        with torch.no_grad():
            recon = self._model(X)  # (N, 30, 4)

        X_np = X.numpy()           # (N, 30, 4)
        recon_np = recon.numpy()   # (N, 30, 4)

        per_step_errors = (recon_np - X_np) ** 2    # (N, 30, 4)
        per_feat_errors = per_step_errors.mean(axis=1)  # (N, 4)
        overall_errors = per_feat_errors.mean(axis=1)   # (N,)

        n = len(feature_vectors)
        elapsed_per_ms = (time.perf_counter() - t0) * 1000.0 / n

        results: list[LSTMResult] = []
        for i, fv in enumerate(feature_vectors):
            feat_errs = per_feat_errors[i]
            overall = float(overall_errors[i])
            score = _error_to_anomaly_score(overall, self._error_mean, self._error_std)
            worst = _FEATURE_NAMES[int(np.argmax(feat_errs))]
            results.append(
                LSTMResult(
                    is_anomaly=overall > self._threshold,
                    anomaly_score=round(score, 6),
                    reconstruction_error=round(overall, 8),
                    threshold_used=round(self._threshold, 8),
                    per_feature_errors={
                        name: round(float(err), 8)
                        for name, err in zip(_FEATURE_NAMES, feat_errs)
                    },
                    worst_feature=worst,
                    inference_time_ms=round(elapsed_per_ms, 3),
                    model_version=self._model_version,
                )
            )

        anomaly_count = sum(1 for r in results if r.is_anomaly)
        logger.info(
            "LSTM batch | n={n} anomalies={a} ({pct:.1%})",
            n=n,
            a=anomaly_count,
            pct=anomaly_count / n,
        )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_loaded(self) -> None:
        """Raise RuntimeError if the model artifact has not been loaded."""
        if self._model is None:
            raise RuntimeError(
                "LSTM model is not loaded.  Call LSTMDetector.load() first."
            )


# ─── Combined detector ────────────────────────────────────────────────────────


def _determine_severity(combined_score: float) -> str:
    """
    Map a combined anomaly score to a human-readable severity label.

    Args:
        combined_score: Weighted ensemble score in ``[0, 1]``.

    Returns:
        One of ``"critical"``, ``"high"``, ``"medium"``, ``"low"``.
    """
    if combined_score > 0.8:
        return "critical"
    if combined_score > 0.6:
        return "high"
    if combined_score > 0.4:
        return "medium"
    return "low"


def _detection_method(lstm_anomaly: bool, if_anomaly: bool) -> str:
    """
    Return a string label describing which detector(s) raised the alarm.

    Args:
        lstm_anomaly: Whether the LSTM flagged the sample.
        if_anomaly:   Whether the Isolation Forest flagged the sample.

    Returns:
        ``"lstm+if"``, ``"lstm_only"``, ``"if_only"``, or ``"none"``.
    """
    if lstm_anomaly and if_anomaly:
        return "lstm+if"
    if lstm_anomaly:
        return "lstm_only"
    if if_anomaly:
        return "if_only"
    return "none"


class AnomalyDetector:
    """
    Ensemble detector that fuses the LSTM and Isolation Forest models.

    The two models are run in parallel using daemon threads to minimise
    latency.  Their normalised anomaly scores are combined as a weighted
    average (LSTM 60 %, IF 40 %) to produce a single decision.

    Example::

        detector = AnomalyDetector()
        result = detector.detect(feature_vector)
        if result.is_anomaly:
            print(result.severity, result.detection_method, result.worst_feature)
    """

    def __init__(
        self,
        lstm_path: Path = DEFAULT_LSTM_PATH,
        if_path: Optional[Path] = None,
    ) -> None:
        """
        Load both detector artifacts from disk.

        Args:
            lstm_path: Path to the ``.pt`` LSTM artifact.
            if_path:   Path to the ``.joblib`` Isolation Forest artifact.
                       Defaults to ``IsolationForestDetector``'s default path.
        """
        from backend.models.isolation_forest import DEFAULT_MODEL_PATH as _IF_PATH

        self._lstm = LSTMDetector.load(lstm_path)
        self._if = IsolationForestDetector.load(if_path or _IF_PATH)

        logger.info(
            "AnomalyDetector ready | lstm_version={lv} if_version={iv}",
            lv=self._lstm._model_version,
            iv=self._if._metadata.get("model_version", "unknown"),
        )

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, feature_vector: FeatureVector) -> CombinedAnomalyResult:
        """
        Run both detectors in parallel and return a fused result.

        The two inference calls execute in separate daemon threads to overlap
        PyTorch forward pass and sklearn inference latency.  Thread join
        ensures both results are available before scoring.

        Combined score formula::

            combined = lstm_score × 0.6 + if_score × 0.4

        is_anomaly is raised when::

            combined_score > 0.5  OR  (lstm.is_anomaly AND if.is_anomaly)

        Args:
            feature_vector: A :class:`FeatureVector` from :class:`FeatureExtractor`.

        Returns:
            :class:`CombinedAnomalyResult` with ensemble decision and full
            sub-results from both models.
        """
        # ── Run both detectors in parallel ────────────────────────────────────
        lstm_holder: list[Optional[LSTMResult]] = [None]
        if_holder: list[Optional[AnomalyResult]] = [None]

        def _run_lstm() -> None:
            lstm_holder[0] = self._lstm.predict(feature_vector)

        def _run_if() -> None:
            if_holder[0] = self._if.predict(feature_vector)

        t_lstm = threading.Thread(target=_run_lstm, daemon=True)
        t_if = threading.Thread(target=_run_if, daemon=True)
        t_lstm.start()
        t_if.start()
        t_lstm.join()
        t_if.join()

        lstm_result: LSTMResult = lstm_holder[0]   # type: ignore[assignment]
        if_result: AnomalyResult = if_holder[0]    # type: ignore[assignment]

        # ── Ensemble scoring ──────────────────────────────────────────────────
        combined_score = round(
            lstm_result.anomaly_score * 0.6 + if_result.anomaly_score * 0.4, 6
        )
        both_anomalous = lstm_result.is_anomaly and if_result.is_anomaly
        is_anomaly = combined_score > 0.5 or both_anomalous

        severity = _determine_severity(combined_score)
        method = _detection_method(lstm_result.is_anomaly, if_result.is_anomaly)

        # Parse timestamp from feature_vector
        try:
            ts = datetime.fromisoformat(feature_vector.timestamp)
        except ValueError:
            ts = datetime.now(timezone.utc)

        result = CombinedAnomalyResult(
            is_anomaly=is_anomaly,
            combined_score=combined_score,
            severity=severity,
            lstm_result=lstm_result,
            if_result=if_result,
            detection_method=method,
            worst_feature=lstm_result.worst_feature,
            top_contributing_features=if_result.top_contributing_features,
            timestamp=ts,
        )

        if is_anomaly:
            logger.warning(
                "⚠ ENSEMBLE ANOMALY | host={host} severity={sev} "
                "score={score:.4f} method={method} "
                "worst={worst} top_features={feats}",
                host=feature_vector.host_id,
                sev=severity,
                score=combined_score,
                method=method,
                worst=lstm_result.worst_feature,
                feats=if_result.top_contributing_features,
            )
        else:
            logger.debug(
                "Ensemble normal | host={host} combined_score={score:.4f}",
                host=feature_vector.host_id,
                score=combined_score,
            )

        return result
