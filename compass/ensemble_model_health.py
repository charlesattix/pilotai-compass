"""
Ensemble model health monitor for live paper trading.

Tracks prediction accuracy vs actual outcomes (rolling window),
detects feature drift via KS-test, monitors ensemble disagreement,
computes rolling AUC and Brier score, and triggers retrain
recommendations when metrics degrade.

Usage::

    from compass.ensemble_model_health import ModelHealthMonitor
    monitor = ModelHealthMonitor()
    monitor.record_prediction(prob=0.75, model_probs={"xgb": 0.80, "rf": 0.70})
    monitor.record_outcome(win=True)
    report = monitor.get_report()
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class HealthConfig:
    rolling_window: int = 100         # predictions to keep in rolling buffer
    auc_drop_threshold: float = 0.05  # AUC drop from baseline → retrain
    brier_rise_threshold: float = 0.03
    disagreement_alert: float = 0.20  # model std > this → alert
    drift_ks_pvalue: float = 0.05     # KS p-value below this → drift
    drift_feature_pct: float = 0.15   # >15% features drifted → retrain
    min_samples_for_auc: int = 20     # minimum outcomes before computing AUC
    baseline_auc: Optional[float] = None  # set from training; None = auto


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class PredictionRecord:
    """One recorded prediction."""
    probability: float
    model_probs: Dict[str, float]
    disagreement: float
    features: Optional[Dict[str, float]] = None
    timestamp: str = ""


@dataclass
class OutcomeRecord:
    """One recorded outcome."""
    actual: int                       # 1 = win, 0 = loss
    predicted_prob: float
    timestamp: str = ""


@dataclass
class DriftResult:
    """KS-test result for one feature."""
    feature: str
    ks_stat: float
    p_value: float
    drifted: bool
    live_mean: float
    train_mean: float
    live_std: float
    train_std: float


@dataclass
class DisagreementAlert:
    """Alert when ensemble models disagree heavily."""
    timestamp: str
    disagreement: float
    model_probs: Dict[str, float]
    severity: str                     # "warning" or "critical"


@dataclass
class RetrainRecommendation:
    """Recommendation to retrain the model."""
    reason: str
    severity: str                     # "warning" or "critical"
    metric_name: str
    current_value: float
    threshold: float
    timestamp: str


@dataclass
class HealthReport:
    """Comprehensive health report."""
    rolling_accuracy: float
    rolling_auc: float
    rolling_brier: float
    baseline_auc: float
    auc_drop: float
    n_predictions: int
    n_outcomes: int
    avg_disagreement: float
    max_disagreement: float
    n_disagreement_alerts: int
    n_features_drifted: int
    n_features_tested: int
    drift_pct: float
    retrain_recommended: bool
    retrain_reasons: List[str]
    health_score: float               # 0-100
    grade: str                        # A-F


# ── Monitor ─────────────────────────────────────────────────────────────


class ModelHealthMonitor:
    """Monitor ensemble model health during live trading."""

    def __init__(self, config: Optional[HealthConfig] = None) -> None:
        self.config = config or HealthConfig()

        # Rolling buffers
        self._predictions: Deque[PredictionRecord] = deque(maxlen=self.config.rolling_window)
        self._outcomes: Deque[OutcomeRecord] = deque(maxlen=self.config.rolling_window)

        # Training-time feature distributions (set via set_training_stats)
        self._train_features: Dict[str, np.ndarray] = {}
        self._train_feature_stats: Dict[str, Tuple[float, float]] = {}  # name → (mean, std)

        # Live feature buffer for drift detection
        self._live_features: Dict[str, List[float]] = {}

        # Alerts
        self._disagreement_alerts: List[DisagreementAlert] = []
        self._retrain_recs: List[RetrainRecommendation] = []

        # Baseline AUC (set from training or auto-computed from first window)
        self._baseline_auc = self.config.baseline_auc
        self._baseline_locked = self.config.baseline_auc is not None

    # ── Recording ───────────────────────────────────────────────────────

    def record_prediction(
        self,
        prob: float,
        model_probs: Optional[Dict[str, float]] = None,
        features: Optional[Dict[str, float]] = None,
    ) -> Optional[DisagreementAlert]:
        """Record an ensemble prediction.  Returns alert if disagreement is high."""
        model_probs = model_probs or {}
        if model_probs:
            disagreement = float(np.std(list(model_probs.values())))
        else:
            disagreement = 0.0

        now = datetime.now(timezone.utc).isoformat()
        self._predictions.append(PredictionRecord(
            probability=prob, model_probs=model_probs,
            disagreement=disagreement, features=features, timestamp=now,
        ))

        # Track live features for drift
        if features:
            for name, val in features.items():
                if name not in self._live_features:
                    self._live_features[name] = []
                self._live_features[name].append(val)

        # Disagreement check
        alert = None
        if disagreement > self.config.disagreement_alert:
            severity = "critical" if disagreement > self.config.disagreement_alert * 1.5 else "warning"
            alert = DisagreementAlert(now, disagreement, model_probs, severity)
            self._disagreement_alerts.append(alert)
            logger.warning("Ensemble disagreement %.3f > %.2f (%s)",
                           disagreement, self.config.disagreement_alert, severity)

        return alert

    def record_outcome(self, win: bool, predicted_prob: Optional[float] = None) -> None:
        """Record actual trade outcome (win/loss)."""
        if predicted_prob is None:
            # Use last prediction
            if self._predictions:
                predicted_prob = self._predictions[-1].probability
            else:
                predicted_prob = 0.5

        now = datetime.now(timezone.utc).isoformat()
        self._outcomes.append(OutcomeRecord(
            actual=int(win), predicted_prob=predicted_prob, timestamp=now,
        ))

        # Auto-set baseline AUC once we have enough data
        if not self._baseline_locked and len(self._outcomes) >= self.config.min_samples_for_auc:
            self._baseline_auc = self.rolling_auc()
            self._baseline_locked = True

    # ── Training stats ──────────────────────────────────────────────────

    def set_training_stats(
        self,
        feature_means: Dict[str, float],
        feature_stds: Dict[str, float],
    ) -> None:
        """Set training-time feature statistics for drift detection."""
        self._train_feature_stats = {
            name: (feature_means[name], feature_stds.get(name, 1.0))
            for name in feature_means
        }

    def set_training_distributions(self, feature_arrays: Dict[str, np.ndarray]) -> None:
        """Set full training feature distributions for KS-test."""
        self._train_features = {k: np.asarray(v, dtype=float) for k, v in feature_arrays.items()}
        # Also compute stats
        for name, arr in self._train_features.items():
            self._train_feature_stats[name] = (float(np.mean(arr)), float(np.std(arr)))

    # ── Rolling metrics ─────────────────────────────────────────────────

    def rolling_accuracy(self) -> float:
        """Rolling accuracy: fraction of correct predictions."""
        if not self._outcomes:
            return 0.0
        correct = sum(
            1 for o in self._outcomes
            if (o.predicted_prob >= 0.5) == (o.actual == 1)
        )
        return correct / len(self._outcomes)

    def rolling_auc(self) -> float:
        """Rolling AUC from recorded outcomes."""
        if len(self._outcomes) < self.config.min_samples_for_auc:
            return 0.0
        actuals = np.array([o.actual for o in self._outcomes])
        probs = np.array([o.predicted_prob for o in self._outcomes])
        if len(np.unique(actuals)) < 2:
            return 0.5  # can't compute AUC with single class
        # Manual AUC (avoid sklearn import for speed)
        return self._compute_auc(actuals, probs)

    def rolling_brier(self) -> float:
        """Rolling Brier score: mean((prob - actual)^2).  Lower is better."""
        if not self._outcomes:
            return 0.0
        actuals = np.array([o.actual for o in self._outcomes], dtype=float)
        probs = np.array([o.predicted_prob for o in self._outcomes])
        return float(np.mean((probs - actuals) ** 2))

    def avg_disagreement(self) -> float:
        """Average disagreement across recent predictions."""
        if not self._predictions:
            return 0.0
        return float(np.mean([p.disagreement for p in self._predictions]))

    def max_disagreement(self) -> float:
        """Maximum disagreement in recent predictions."""
        if not self._predictions:
            return 0.0
        return float(np.max([p.disagreement for p in self._predictions]))

    # ── Feature drift (KS-test) ─────────────────────────────────────────

    def detect_drift(self, min_samples: int = 30) -> List[DriftResult]:
        """Run KS-test on each feature comparing live vs training distributions."""
        results: List[DriftResult] = []
        for name, live_vals in self._live_features.items():
            if len(live_vals) < min_samples:
                continue

            live_arr = np.array(live_vals[-self.config.rolling_window:])

            if name in self._train_features and len(self._train_features[name]) >= min_samples:
                train_arr = self._train_features[name]
                ks_stat, p_val = sp_stats.ks_2samp(live_arr, train_arr)
            elif name in self._train_feature_stats:
                # Synthetic: compare to normal with training mean/std
                mean, std = self._train_feature_stats[name]
                if std > 0:
                    train_synthetic = np.random.RandomState(42).normal(mean, std, len(live_arr))
                    ks_stat, p_val = sp_stats.ks_2samp(live_arr, train_synthetic)
                else:
                    ks_stat, p_val = 0.0, 1.0
            else:
                continue

            t_mean, t_std = self._train_feature_stats.get(name, (0.0, 1.0))
            results.append(DriftResult(
                feature=name,
                ks_stat=float(ks_stat),
                p_value=float(p_val),
                drifted=p_val < self.config.drift_ks_pvalue,
                live_mean=float(np.mean(live_arr)),
                train_mean=t_mean,
                live_std=float(np.std(live_arr)),
                train_std=t_std,
            ))

        return sorted(results, key=lambda d: d.p_value)

    # ── Retrain recommendations ─────────────────────────────────────────

    def check_retrain(self) -> List[RetrainRecommendation]:
        """Check if model should be retrained.  Returns list of recommendations."""
        recs: List[RetrainRecommendation] = []
        now = datetime.now(timezone.utc).isoformat()

        # 1. AUC drop
        current_auc = self.rolling_auc()
        baseline = self._baseline_auc or 0.80
        if current_auc > 0 and (baseline - current_auc) > self.config.auc_drop_threshold:
            drop = baseline - current_auc
            rec = RetrainRecommendation(
                reason=f"AUC dropped {drop:.3f} from baseline {baseline:.3f} to {current_auc:.3f}",
                severity="critical" if drop > self.config.auc_drop_threshold * 2 else "warning",
                metric_name="auc_drop", current_value=current_auc,
                threshold=baseline - self.config.auc_drop_threshold, timestamp=now,
            )
            recs.append(rec)

        # 2. Brier score rise
        brier = self.rolling_brier()
        if brier > 0.25 + self.config.brier_rise_threshold:
            rec = RetrainRecommendation(
                reason=f"Brier score {brier:.3f} exceeds threshold {0.25 + self.config.brier_rise_threshold:.3f}",
                severity="warning", metric_name="brier_score",
                current_value=brier, threshold=0.25 + self.config.brier_rise_threshold,
                timestamp=now,
            )
            recs.append(rec)

        # 3. Feature drift
        drift_results = self.detect_drift()
        n_drifted = sum(1 for d in drift_results if d.drifted)
        n_tested = len(drift_results)
        if n_tested > 0:
            drift_pct = n_drifted / n_tested
            if drift_pct > self.config.drift_feature_pct:
                drifted_names = [d.feature for d in drift_results if d.drifted][:5]
                rec = RetrainRecommendation(
                    reason=f"{n_drifted}/{n_tested} features drifted ({drift_pct:.0%}): {', '.join(drifted_names)}",
                    severity="critical" if drift_pct > 0.30 else "warning",
                    metric_name="feature_drift", current_value=drift_pct,
                    threshold=self.config.drift_feature_pct, timestamp=now,
                )
                recs.append(rec)

        # 4. High sustained disagreement
        avg_dis = self.avg_disagreement()
        if avg_dis > self.config.disagreement_alert:
            rec = RetrainRecommendation(
                reason=f"Sustained ensemble disagreement {avg_dis:.3f} > {self.config.disagreement_alert}",
                severity="warning", metric_name="disagreement",
                current_value=avg_dis, threshold=self.config.disagreement_alert,
                timestamp=now,
            )
            recs.append(rec)

        self._retrain_recs = recs
        return recs

    # ── Health report ───────────────────────────────────────────────────

    def get_report(self) -> HealthReport:
        """Generate comprehensive health report."""
        acc = self.rolling_accuracy()
        auc = self.rolling_auc()
        brier = self.rolling_brier()
        baseline = self._baseline_auc or 0.80
        auc_drop = max(baseline - auc, 0) if auc > 0 else 0

        drift_results = self.detect_drift()
        n_drifted = sum(1 for d in drift_results if d.drifted)
        n_tested = len(drift_results)
        drift_pct = n_drifted / n_tested if n_tested > 0 else 0

        recs = self.check_retrain()
        retrain = len(recs) > 0
        reasons = [r.reason for r in recs]

        # Health score: 100 - penalties
        score = 100.0
        if auc_drop > 0.05:
            score -= auc_drop * 200
        if brier > 0.25:
            score -= (brier - 0.25) * 100
        if drift_pct > 0.15:
            score -= drift_pct * 50
        if self.avg_disagreement() > 0.20:
            score -= self.avg_disagreement() * 50
        score = max(0, min(100, score))

        if score >= 85:
            grade = "A"
        elif score >= 70:
            grade = "B"
        elif score >= 55:
            grade = "C"
        elif score >= 40:
            grade = "D"
        else:
            grade = "F"

        return HealthReport(
            rolling_accuracy=acc, rolling_auc=auc, rolling_brier=brier,
            baseline_auc=baseline, auc_drop=auc_drop,
            n_predictions=len(self._predictions),
            n_outcomes=len(self._outcomes),
            avg_disagreement=self.avg_disagreement(),
            max_disagreement=self.max_disagreement(),
            n_disagreement_alerts=len(self._disagreement_alerts),
            n_features_drifted=n_drifted, n_features_tested=n_tested,
            drift_pct=drift_pct,
            retrain_recommended=retrain, retrain_reasons=reasons,
            health_score=score, grade=grade,
        )

    # ── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_auc(actuals: np.ndarray, probs: np.ndarray) -> float:
        """Compute AUC without sklearn (Wilcoxon-Mann-Whitney)."""
        pos = probs[actuals == 1]
        neg = probs[actuals == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        n_pos = len(pos)
        n_neg = len(neg)
        # Count concordant pairs
        concordant = 0
        for p in pos:
            concordant += np.sum(p > neg) + 0.5 * np.sum(p == neg)
        return float(concordant / (n_pos * n_neg))

    def reset(self) -> None:
        """Clear all recorded data."""
        self._predictions.clear()
        self._outcomes.clear()
        self._live_features.clear()
        self._disagreement_alerts.clear()
        self._retrain_recs.clear()
        self._baseline_locked = self.config.baseline_auc is not None
        self._baseline_auc = self.config.baseline_auc
