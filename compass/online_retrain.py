"""
Online / Rolling Retraining for Signal Models

Monitors model staleness, feature drift, and out-of-sample performance to
decide when to retrain.  After retraining, compares the new model against
the current model on a held-out set before promoting it to production.

Keeps the last N model versions on disk for rollback.

GAP fixes (phase4_integration_plan.md §3.2 Step 1):
  GAP-1  _check_performance reads "ensemble_test_auc" OR "test_auc".
  GAP-3  model_class parameter: replaces hardcoded SignalModel() construction.
  GAP-4  File glob patterns derived from model class name so ensemble files
         are saved, pruned, listed, and age-checked correctly.
  GAP-5  feature_pipeline parameter: applied before training so training
         features always match inference features.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compass.signal_model import SignalModel
from shared.indicators import sanitize_features

logger = logging.getLogger(__name__)


def _model_file_prefix(model_class: type) -> str:
    """Return the versioned-file prefix for a model class.

    ``EnsembleSignalModel`` → ``"ensemble_model"``
    ``SignalModel`` (default) → ``"signal_model"``

    Derived from ``model_class.__name__`` so subclasses are handled
    automatically without an explicit registry.
    """
    name = getattr(model_class, '__name__', '')
    return 'ensemble_model' if 'Ensemble' in name else 'signal_model'


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrainTrigger:
    """Why a retrain was triggered."""
    model_age_days: Optional[int] = None
    drift_features: List[str] = field(default_factory=list)
    perf_auc_current: Optional[float] = None
    perf_auc_baseline: Optional[float] = None
    triggered: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class ABResult:
    """Side-by-side evaluation of old vs new model on a holdout set."""
    old_auc: float
    new_auc: float
    old_accuracy: float
    new_accuracy: float
    holdout_size: int
    promoted: bool
    reason: str


@dataclass
class RetrainResult:
    """Full result of a check_and_retrain cycle."""
    trigger: RetrainTrigger
    retrained: bool = False
    ab_result: Optional[ABResult] = None
    new_model_path: Optional[str] = None
    training_stats: Optional[Dict] = None
    versions_on_disk: int = 0


# ---------------------------------------------------------------------------
# ModelRetrainer
# ---------------------------------------------------------------------------

class ModelRetrainer:
    """Online / rolling retraining manager for signal models.

    Supports both :class:`~compass.signal_model.SignalModel` and
    :class:`~compass.ensemble_signal_model.EnsembleSignalModel` via the
    ``model_class`` parameter (GAP-3).  File glob patterns are derived
    automatically from the model class name (GAP-4).  An optional
    ``feature_pipeline`` ensures training and inference features stay
    aligned (GAP-5).

    Parameters
    ----------
    model_dir : str
        Directory where versioned model files are stored.
    max_age_days : int
        Retrain if the current model is older than this.
    drift_threshold : float
        Number of standard deviations used to flag a feature as drifted.
    drift_feature_pct : float
        Fraction of features that must be drifted to trigger a retrain.
    perf_auc_drop : float
        Absolute AUC drop from baseline that triggers a retrain.
    rolling_window_months : int
        Training window size in months (most recent trades).
    holdout_fraction : float
        Fraction of training data reserved for A/B holdout comparison.
    keep_versions : int
        Number of old model versions to keep on disk.
    min_promotion_auc_delta : float
        New model must beat old model by at least this much AUC to be
        promoted.  Set to a small negative number (e.g. -0.005) to allow
        promotions that are slightly worse on the holdout but still fresh.
    min_samples : int
        Minimum number of training samples required to attempt retraining.
    model_class : type, optional
        Model class to instantiate when no ``current_model`` is supplied and
        when training the candidate model.  Defaults to
        :class:`~compass.signal_model.SignalModel`.  Pass
        :class:`~compass.ensemble_signal_model.EnsembleSignalModel` to use
        the ensemble instead.  Any class with the same
        ``train`` / ``predict_batch`` / ``save`` / ``load`` interface works.
        **GAP-3 fix.**
    feature_pipeline : object, optional
        A stateless transformer with a ``transform(df) -> DataFrame`` method
        (e.g. :class:`~compass.feature_pipeline.FeaturePipeline`).
        When provided, ``check_and_retrain`` applies it to ``trades_df``
        *before* training so that training features match inference features.
        The rolling-window trim still uses the raw (un-transformed) DataFrame
        to preserve date-column access.  **GAP-5 fix.**
    """

    def __init__(
        self,
        model_dir: str = "ml/models",
        max_age_days: int = 30,
        drift_threshold: float = 3.0,
        drift_feature_pct: float = 0.15,
        perf_auc_drop: float = 0.05,
        rolling_window_months: int = 12,
        holdout_fraction: float = 0.20,
        keep_versions: int = 3,
        min_promotion_auc_delta: float = -0.005,
        min_samples: int = 100,
        model_class: Optional[Type] = None,       # GAP-3
        feature_pipeline: Optional[Any] = None,   # GAP-5
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self.drift_threshold = drift_threshold
        self.drift_feature_pct = drift_feature_pct
        self.perf_auc_drop = perf_auc_drop
        self.rolling_window_months = rolling_window_months
        self.holdout_fraction = holdout_fraction
        self.keep_versions = keep_versions
        self.min_promotion_auc_delta = min_promotion_auc_delta
        self.min_samples = min_samples
        # GAP-3: configurable model class (default: SignalModel for backward compat)
        self.model_class: Type = model_class if model_class is not None else SignalModel
        # GAP-4: file prefix derived from model class name
        self._model_file_prefix: str = _model_file_prefix(self.model_class)
        # GAP-5: optional feature pipeline for training/inference alignment
        self.feature_pipeline: Optional[Any] = feature_pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_retrain(
        self,
        trades_df: pd.DataFrame,
        labels: np.ndarray,
        current_model: Optional[Any] = None,
        force: bool = False,
    ) -> RetrainResult:
        """Evaluate whether the model needs retraining and, if so, retrain.

        Parameters
        ----------
        trades_df : pd.DataFrame
            Feature DataFrame (rows = trades, columns = feature names).
            Should be sorted chronologically (oldest first).
            If a ``feature_pipeline`` is configured, the raw DataFrame is
            expected here — the pipeline is applied internally.
        labels : np.ndarray
            Binary labels aligned with *trades_df* rows.
        current_model : model object, optional
            The currently-deployed model.  If ``None``, a fresh one is loaded
            from ``model_dir`` using ``self.model_class``.
        force : bool
            Skip trigger checks and retrain unconditionally.

        Returns
        -------
        RetrainResult
        """
        # --- load current model if needed ---
        if current_model is None:
            # GAP-3: use self.model_class instead of hardcoded SignalModel
            current_model = self.model_class(model_dir=str(self.model_dir))
            if not current_model.load():
                logger.info("No existing model found — will train from scratch")
                force = True

        # GAP-5: Apply FeaturePipeline to produce the feature representation
        # that the model sees at inference time.  Raw trades_df is preserved
        # for rolling-window trimming (which may need date columns).
        if self.feature_pipeline is not None:
            features_df = self.feature_pipeline.transform(trades_df)
            logger.debug(
                "FeaturePipeline applied: %d raw cols → %d pipeline cols",
                trades_df.shape[1], features_df.shape[1],
            )
        else:
            features_df = trades_df

        # --- check triggers (on pipeline-transformed features if set) ---
        trigger = self._evaluate_triggers(current_model, features_df, labels)

        if not force and not trigger.triggered:
            logger.info("No retrain trigger fired — model is current")
            return RetrainResult(
                trigger=trigger,
                versions_on_disk=self._count_versions(),
            )

        if force:
            trigger.triggered = True
            trigger.reasons.append("forced")

        logger.info("Retrain triggered: %s", ", ".join(trigger.reasons))

        # --- window the raw data (preserves date columns for calendar trim) ---
        trades_windowed_raw, labels_windowed = self._apply_rolling_window(
            trades_df, labels
        )

        # GAP-5: Apply pipeline to the windowed slice used for training
        if self.feature_pipeline is not None:
            trades_windowed = self.feature_pipeline.transform(trades_windowed_raw)
        else:
            trades_windowed = trades_windowed_raw

        if len(trades_windowed) < self.min_samples:
            logger.warning(
                "Only %d samples in rolling window (need %d) — skipping retrain",
                len(trades_windowed),
                self.min_samples,
            )
            return RetrainResult(trigger=trigger)

        # --- split holdout ---
        n_holdout = max(1, int(len(trades_windowed) * self.holdout_fraction))
        train_features = trades_windowed.iloc[:-n_holdout]
        train_labels = labels_windowed[:-n_holdout]
        holdout_features = trades_windowed.iloc[-n_holdout:]
        holdout_labels = labels_windowed[-n_holdout:]

        # --- train new model ---
        # GAP-3: use self.model_class instead of hardcoded SignalModel
        new_model = self.model_class(model_dir=str(self.model_dir))
        stats = new_model.train(
            train_features,
            train_labels,
            calibrate=True,
            save_model=False,
        )

        if not stats or not new_model.trained:
            logger.error("New model training failed — keeping current model")
            return RetrainResult(trigger=trigger)

        # --- A/B comparison on holdout ---
        ab_result = self._compare_models(
            current_model, new_model, holdout_features, holdout_labels
        )

        result = RetrainResult(
            trigger=trigger,
            retrained=True,
            ab_result=ab_result,
            training_stats=stats,
        )

        if ab_result.promoted:
            version_path = self._save_versioned(new_model)
            result.new_model_path = str(version_path)
            self._prune_old_versions()
            logger.info("New model promoted → %s", version_path)
        else:
            logger.info(
                "New model NOT promoted (old AUC=%.4f, new AUC=%.4f): %s",
                ab_result.old_auc,
                ab_result.new_auc,
                ab_result.reason,
            )

        result.versions_on_disk = self._count_versions()
        return result

    # ------------------------------------------------------------------
    # Trigger evaluation
    # ------------------------------------------------------------------

    def _evaluate_triggers(
        self,
        model: Any,
        features_df: pd.DataFrame,
        labels: np.ndarray,
    ) -> RetrainTrigger:
        trigger = RetrainTrigger()

        # 1. Model age
        age = self._get_model_age_days(model)
        trigger.model_age_days = age
        if age is not None and age > self.max_age_days:
            trigger.triggered = True
            trigger.reasons.append(f"model_age={age}d > {self.max_age_days}d")

        # 2. Feature drift
        drifted = self._check_feature_drift(model, features_df)
        trigger.drift_features = drifted
        if model.feature_names and len(drifted) / max(len(model.feature_names), 1) >= self.drift_feature_pct:
            trigger.triggered = True
            trigger.reasons.append(
                f"feature_drift={len(drifted)}/{len(model.feature_names)} "
                f"(>= {self.drift_feature_pct:.0%})"
            )

        # 3. Performance degradation
        perf = self._check_performance(model, features_df, labels)
        if perf is not None:
            trigger.perf_auc_current = perf["current_auc"]
            trigger.perf_auc_baseline = perf["baseline_auc"]
            drop = perf["baseline_auc"] - perf["current_auc"]
            if drop >= self.perf_auc_drop:
                trigger.triggered = True
                trigger.reasons.append(
                    f"auc_drop={drop:.4f} (baseline={perf['baseline_auc']:.4f}, "
                    f"current={perf['current_auc']:.4f})"
                )

        return trigger

    def _get_model_age_days(self, model: Any) -> Optional[int]:
        """Return model age in days, or None if unknown."""
        ts = model.training_stats.get("timestamp") if model.training_stats else None
        if ts is None:
            # GAP-4: fall back using the correct file prefix for this model class
            model_files = sorted(
                self.model_dir.glob(f"{self._model_file_prefix}_*.joblib"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not model_files:
                return None
            # Use file mtime as approximation
            mtime = datetime.fromtimestamp(
                model_files[0].stat().st_mtime, tz=timezone.utc
            )
            return (datetime.now(timezone.utc) - mtime).days

        try:
            trained_at = datetime.fromisoformat(ts)
            if trained_at.tzinfo is None:
                trained_at = trained_at.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - trained_at).days
        except (ValueError, TypeError):
            return None

    def _check_feature_drift(
        self, model: Any, features_df: pd.DataFrame
    ) -> List[str]:
        """Return names of features whose recent mean drifted beyond threshold.

        Integrates with SignalModel's existing ``feature_means`` / ``feature_stds``
        that are computed during training.
        """
        if model.feature_means is None or model.feature_stds is None:
            return []
        if model.feature_names is None:
            return []

        drifted: List[str] = []
        # Align columns to model's expected order
        available = [c for c in model.feature_names if c in features_df.columns]
        if not available:
            return []

        X = features_df[available].values
        X = sanitize_features(X)
        recent_means = np.nanmean(X, axis=0)

        # Map available columns back to index in model.feature_names
        for j, col in enumerate(available):
            idx = model.feature_names.index(col)
            std = model.feature_stds[idx]
            if std == 0 or np.isnan(std):
                continue
            z = abs(recent_means[j] - model.feature_means[idx]) / std
            if z > self.drift_threshold:
                drifted.append(col)

        return drifted

    def _check_performance(
        self,
        model: Any,
        features_df: pd.DataFrame,
        labels: np.ndarray,
    ) -> Optional[Dict]:
        """Evaluate model on recent data and compare to training baseline."""
        if not model.trained:
            return None

        # GAP-1 fix: EnsembleSignalModel uses "ensemble_test_auc"; SignalModel
        # uses "test_auc".  Try both so this works with either class.
        stats = model.training_stats or {}
        baseline_auc = stats.get("ensemble_test_auc") or stats.get("test_auc")
        if baseline_auc is None:
            return None

        try:
            probas = model.predict_batch(features_df)
            if len(np.unique(labels)) < 2:
                return None
            current_auc = float(roc_auc_score(labels, probas))
            return {"baseline_auc": baseline_auc, "current_auc": current_auc}
        except Exception as e:
            logger.warning("Performance check failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Rolling window
    # ------------------------------------------------------------------

    def _apply_rolling_window(
        self, features_df: pd.DataFrame, labels: np.ndarray
    ) -> tuple:
        """Trim to the most recent ``rolling_window_months`` of data.

        If the DataFrame has a DatetimeIndex or a 'date'/'timestamp' column,
        the window is calendar-based.  Otherwise, the last
        ``rolling_window_months * 21`` rows are used (approx trading days).
        """
        n_rows = len(features_df)

        # Try calendar-based window
        dt_index = None
        if isinstance(features_df.index, pd.DatetimeIndex):
            dt_index = features_df.index
        elif "date" in features_df.columns:
            dt_index = pd.to_datetime(features_df["date"], errors="coerce")
        elif "timestamp" in features_df.columns:
            dt_index = pd.to_datetime(features_df["timestamp"], errors="coerce")

        if dt_index is not None and not dt_index.isna().all():
            cutoff = dt_index.max() - pd.DateOffset(months=self.rolling_window_months)
            mask = dt_index >= cutoff
            mask_arr = mask.values if hasattr(mask, 'values') else np.asarray(mask)
            return features_df.loc[mask], labels[mask_arr]

        # Fallback: row-count approximation
        approx_rows = self.rolling_window_months * 21
        if n_rows > approx_rows:
            return features_df.iloc[-approx_rows:], labels[-approx_rows:]
        return features_df, labels

    # ------------------------------------------------------------------
    # A/B comparison
    # ------------------------------------------------------------------

    def _compare_models(
        self,
        old_model: Any,
        new_model: Any,
        holdout_features: pd.DataFrame,
        holdout_labels: np.ndarray,
    ) -> ABResult:
        """Compare old and new model on a holdout set."""
        from sklearn.metrics import accuracy_score

        has_two_classes = len(np.unique(holdout_labels)) >= 2

        # --- old model ---
        if old_model.trained:
            old_probas = old_model.predict_batch(holdout_features)
            old_preds = (old_probas > 0.5).astype(int)
            old_auc = float(roc_auc_score(holdout_labels, old_probas)) if has_two_classes else 0.5
            old_acc = float(accuracy_score(holdout_labels, old_preds))
        else:
            old_auc = 0.5
            old_acc = 0.0

        # --- new model ---
        new_probas = new_model.predict_batch(holdout_features)
        new_preds = (new_probas > 0.5).astype(int)
        new_auc = float(roc_auc_score(holdout_labels, new_probas)) if has_two_classes else 0.5
        new_acc = float(accuracy_score(holdout_labels, new_preds))

        # --- promotion decision ---
        delta = new_auc - old_auc
        if delta >= self.min_promotion_auc_delta:
            promoted = True
            reason = f"new_auc={new_auc:.4f} >= old_auc={old_auc:.4f} + {self.min_promotion_auc_delta}"
        else:
            promoted = False
            reason = (
                f"new_auc={new_auc:.4f} < old_auc={old_auc:.4f} + "
                f"{self.min_promotion_auc_delta} (delta={delta:.4f})"
            )

        return ABResult(
            old_auc=old_auc,
            new_auc=new_auc,
            old_accuracy=old_acc,
            new_accuracy=new_acc,
            holdout_size=len(holdout_labels),
            promoted=promoted,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Versioned save / prune
    # ------------------------------------------------------------------

    def _save_versioned(self, model: Any) -> Path:
        """Save model with a timestamped filename and return the path.

        GAP-4 fix: filename prefix is derived from ``self._model_file_prefix``
        so EnsembleSignalModel files are named ``ensemble_model_*.joblib``
        (which ``EnsembleSignalModel.load()`` expects) rather than the old
        hardcoded ``signal_model_*.joblib``.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{self._model_file_prefix}_{ts}.joblib"
        model.save(filename)
        return self.model_dir / filename

    def _prune_old_versions(self) -> List[Path]:
        """Delete model versions beyond ``keep_versions``, oldest first.

        GAP-4 fix: uses ``self._model_file_prefix`` glob so ensemble files are
        pruned when the retrainer is configured with EnsembleSignalModel.

        Returns list of deleted paths.
        """
        model_files = sorted(
            self.model_dir.glob(f"{self._model_file_prefix}_*.joblib"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        to_delete = model_files[self.keep_versions:]
        deleted: List[Path] = []
        for fp in to_delete:
            try:
                fp.unlink()
                # Also remove companion feature_stats json if present
                stats_path = fp.with_suffix(".feature_stats.json")
                if stats_path.exists():
                    stats_path.unlink()
                deleted.append(fp)
                logger.info("Pruned old model version: %s", fp.name)
            except OSError as e:
                logger.warning("Failed to prune %s: %s", fp, e)

        return deleted

    def _count_versions(self) -> int:
        # GAP-4: count files matching this retrainer's model class prefix
        return len(list(self.model_dir.glob(f"{self._model_file_prefix}_*.joblib")))

    # ------------------------------------------------------------------
    # Convenience: list versions
    # ------------------------------------------------------------------

    def list_versions(self) -> List[Dict]:
        """Return metadata for each model version on disk, newest first.

        GAP-4 fix: lists files matching this retrainer's model class prefix.
        """
        model_files = sorted(
            self.model_dir.glob(f"{self._model_file_prefix}_*.joblib"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        versions = []
        for fp in model_files:
            entry = {
                "filename": fp.name,
                "size_bytes": fp.stat().st_size,
                "modified": datetime.fromtimestamp(
                    fp.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            stats_path = fp.with_suffix(".feature_stats.json")
            if stats_path.exists():
                entry["has_feature_stats"] = True
            versions.append(entry)
        return versions
