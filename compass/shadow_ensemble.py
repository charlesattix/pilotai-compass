"""
ShadowEnsemble — Phase 4A shadow-mode wrapper.

Runs EnsembleSignalModel alongside the primary SignalModel for head-to-head
comparison without affecting live trading decisions.  The primary model controls
all signal output; the ensemble is queried in parallel and its predictions are:

  - Logged to a CSV file (compass/shadow_log.csv by default) for offline analysis
  - Tracked in-memory for agreement rate and mean |Δprobability| monitoring

The wrapper is interface-compatible with SignalModel: any caller that accepts
a SignalModel can be passed a ShadowEnsemble without modification.

Usage::

    from compass.signal_model import SignalModel
    from compass.ensemble_signal_model import EnsembleSignalModel
    from compass.shadow_ensemble import ShadowEnsemble

    primary = SignalModel(model_dir="ml/models")
    primary.load()

    shadow = EnsembleSignalModel(model_dir="ml/models")
    shadow.train(features_df, labels, save_model=False)

    model = ShadowEnsemble(primary, shadow)
    # Drop-in replacement — same interface as SignalModel.

Shadow mode activation checklist:
  - Deploy ShadowEnsemble (this file)
  - Observe SHADOW log lines for 2+ weeks
  - Pre-conditions for Phase 4B hard swap:
      - shadow_fallbacks == 0 across all predictions
      - mean_abs_delta < 0.15
      - Ensemble Brier Score within 0.02 of primary on live data
"""

from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from shared.types import PredictionResult

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("compass/shadow_log.csv")

_CSV_FIELDNAMES = [
    "timestamp",
    "primary_prob",
    "ensemble_prob",
    "delta",
    "primary_signal",
    "ensemble_signal",
    "agreed",
    "primary_fallback",
    "shadow_fallback",
]


class ShadowEnsemble:
    """Primary SignalModel with a shadow EnsembleSignalModel running in parallel.

    Parameters
    ----------
    primary:
        The model that controls live trading decisions. Unchanged by this wrapper.
    shadow:
        The EnsembleSignalModel being evaluated. Never acts on trade signals.
    log_path:
        Path to the CSV comparison log. Created with header if it does not exist.
        Rows are appended on every predict() call.
    """

    def __init__(
        self,
        primary: Any,
        shadow: Any,
        log_path: Optional[Path] = None,
    ) -> None:
        self.primary = primary
        self.shadow = shadow
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH

        # Separate locks: stats counters vs. file I/O
        self._stats_lock = threading.Lock()
        self._csv_lock = threading.Lock()

        # In-memory tracking
        self._total_predictions: int = 0
        self._agreed_predictions: int = 0
        self._shadow_fallback_count: int = 0
        self._sum_abs_delta: float = 0.0

        self._ensure_log_header()

        logger.info(
            "ShadowEnsemble initialized: primary=%s  shadow=%s  log=%s",
            type(primary).__name__,
            type(shadow).__name__,
            self.log_path,
        )

    # ── Primary prediction interface (controls live trading) ──────────────

    def predict(self, features: Dict) -> PredictionResult:
        """Return primary model prediction; shadow-log ensemble result.

        The primary result is always returned unchanged. The ensemble is queried
        concurrently and its output is written to the CSV log plus tracked in
        in-memory statistics. If the shadow model fails, the failure is counted
        and the primary result is still returned normally.
        """
        primary_result = self.primary.predict(features)

        shadow_result: Optional[PredictionResult] = None
        shadow_failed = False
        try:
            shadow_result = self.shadow.predict(features)
        except Exception as exc:
            shadow_failed = True
            logger.warning("ShadowEnsemble: shadow.predict() failed: %s", exc)

        self._record(primary_result, shadow_result, shadow_failed)

        return primary_result  # primary always controls signal

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Batch prediction — primary controls output; shadow runs for comparison.

        Aggregate batch statistics (mean delta, batch agreement) are emitted
        at INFO level but are NOT written to the per-row CSV log.
        """
        primary_probas = self.primary.predict_batch(features_df)

        try:
            shadow_probas = self.shadow.predict_batch(features_df)
            deltas = shadow_probas - primary_probas
            batch_agreement = float(
                np.mean((primary_probas > 0.5) == (shadow_probas > 0.5))
            )
            logger.info(
                "ShadowEnsemble batch n=%d: mean_delta=%+.4f mean_abs_delta=%.4f "
                "batch_agreement=%.3f",
                len(features_df),
                float(np.mean(deltas)),
                float(np.mean(np.abs(deltas))),
                batch_agreement,
            )
        except Exception as exc:
            logger.warning("ShadowEnsemble: shadow.predict_batch() failed: %s", exc)

        return primary_probas

    # ── Persistence / training — forwarded to primary ─────────────────────

    def train(self, *args: Any, **kwargs: Any) -> Dict:
        return self.primary.train(*args, **kwargs)

    def save(self, *args: Any, **kwargs: Any) -> None:
        return self.primary.save(*args, **kwargs)

    def load(self, *args: Any, **kwargs: Any) -> bool:
        return self.primary.load(*args, **kwargs)

    def backtest(self, *args: Any, **kwargs: Any) -> Dict:
        return self.primary.backtest(*args, **kwargs)

    def get_fallback_stats(self) -> Dict[str, int]:
        return self.primary.get_fallback_stats()

    # ── SignalModel attribute proxies (needed by online_retrain.py) ────────

    @property
    def trained(self) -> bool:
        return self.primary.trained

    @property
    def feature_names(self) -> Optional[List[str]]:
        return self.primary.feature_names

    @property
    def training_stats(self) -> Dict:
        return self.primary.training_stats

    @property
    def feature_means(self) -> Optional[np.ndarray]:
        return self.primary.feature_means

    @property
    def feature_stds(self) -> Optional[np.ndarray]:
        return self.primary.feature_stds

    # ── Shadow agreement statistics ────────────────────────────────────────

    @property
    def agreement_rate(self) -> Optional[float]:
        """Fraction of predictions where both models agree on direction (> or ≤ 0.5).

        Returns None until the first prediction has been made.
        Shadow fallbacks count as disagreements.
        """
        with self._stats_lock:
            if self._total_predictions == 0:
                return None
            return self._agreed_predictions / self._total_predictions

    @property
    def mean_abs_delta(self) -> Optional[float]:
        """Mean |ensemble_probability − primary_probability| across all predictions.

        Returns None until the first successful (non-fallback) shadow prediction.
        """
        with self._stats_lock:
            # Only computed over non-fallback predictions
            valid = self._total_predictions - self._shadow_fallback_count
            if valid == 0:
                return None
            return self._sum_abs_delta / valid

    def get_shadow_stats(self) -> Dict[str, Any]:
        """Return a snapshot of in-memory shadow comparison statistics.

        Keys
        ----
        total_predictions   : int   — total predict() calls
        agreed_predictions  : int   — calls where both models agree on direction
        shadow_fallbacks    : int   — calls where shadow model returned fallback/error
        agreement_rate      : float | None
        mean_abs_delta      : float | None
        """
        with self._stats_lock:
            total = self._total_predictions
            agreed = self._agreed_predictions
            fallbacks = self._shadow_fallback_count
            valid = total - fallbacks
            return {
                "total_predictions": total,
                "agreed_predictions": agreed,
                "shadow_fallbacks": fallbacks,
                "agreement_rate": agreed / total if total > 0 else None,
                "mean_abs_delta": (
                    self._sum_abs_delta / valid if valid > 0 else None
                ),
            }

    # ── Internal helpers ───────────────────────────────────────────────────

    def _record(
        self,
        primary_result: PredictionResult,
        shadow_result: Optional[PredictionResult],
        shadow_failed: bool,
    ) -> None:
        """Update stats counters and append one row to the CSV log."""
        primary_prob = float(primary_result.get("probability", 0.5))
        primary_signal = str(primary_result.get("signal", "neutral"))
        primary_fallback = bool(primary_result.get("fallback", False))

        shadow_fallback_flag = shadow_failed

        if shadow_result is not None and not shadow_failed:
            ensemble_prob = float(shadow_result.get("probability", 0.5))
            ensemble_signal = str(shadow_result.get("signal", "neutral"))
            if shadow_result.get("fallback", False):
                shadow_fallback_flag = True
            delta = ensemble_prob - primary_prob
            agreed = (primary_prob > 0.5) == (ensemble_prob > 0.5)
        else:
            ensemble_prob = float("nan")
            ensemble_signal = "error"
            delta = float("nan")
            agreed = False

        # Update in-memory stats
        with self._stats_lock:
            self._total_predictions += 1
            if agreed:
                self._agreed_predictions += 1
            if shadow_fallback_flag:
                self._shadow_fallback_count += 1
            if not (shadow_failed or np.isnan(delta)):
                self._sum_abs_delta += abs(delta)

        logger.info(
            "SHADOW | primary_prob=%.4f  ensemble_prob=%s  delta=%s  agreed=%s  "
            "shadow_fallback=%s",
            primary_prob,
            f"{ensemble_prob:.4f}" if not np.isnan(ensemble_prob) else "ERROR",
            f"{delta:+.4f}" if not np.isnan(delta) else "ERROR",
            agreed,
            shadow_fallback_flag,
        )

        # Write CSV row (separate lock from stats)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "primary_prob": round(primary_prob, 6),
            "ensemble_prob": (
                round(ensemble_prob, 6) if not np.isnan(ensemble_prob) else ""
            ),
            "delta": round(delta, 6) if not np.isnan(delta) else "",
            "primary_signal": primary_signal,
            "ensemble_signal": ensemble_signal,
            "agreed": int(agreed),
            "primary_fallback": int(primary_fallback),
            "shadow_fallback": int(shadow_fallback_flag),
        }
        self._append_csv_row(row)

    def _ensure_log_header(self) -> None:
        """Create the CSV log with a header row if it does not yet exist."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with self._csv_lock:
                # Double-check after acquiring lock (race condition guard)
                if not self.log_path.exists():
                    with open(self.log_path, "w", newline="") as fh:
                        csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES).writeheader()
                    logger.info("ShadowEnsemble: created shadow log at %s", self.log_path)

    def _append_csv_row(self, row: Dict) -> None:
        """Append a single row to the CSV log (thread-safe)."""
        try:
            with self._csv_lock:
                with open(self.log_path, "a", newline="") as fh:
                    csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES).writerow(row)
        except Exception as exc:
            logger.warning("ShadowEnsemble: failed to write CSV row: %s", exc)
