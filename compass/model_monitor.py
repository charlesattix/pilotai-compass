"""
Model monitoring — drift detection, accuracy tracking, and alerting.

Tracks the deployed signal model's health over time by watching for:
  1. **Prediction accuracy decay** — rolling accuracy on recent trades
  2. **Feature drift** — KS-test comparing recent feature distributions
     against training-time baselines
  3. **Concept drift** — AUC degradation in rolling windows
  4. **Alert generation** — fires when metrics cross configurable thresholds

Integrates with :mod:`compass.online_retrain` by producing
:class:`MonitorSnapshot` objects that the retraining scheduler can
inspect to decide whether to trigger a retrain cycle.

Usage::

    from compass.model_monitor import ModelMonitor
    monitor = ModelMonitor(feature_means=train_means, feature_stds=train_stds)
    monitor.record(features, y_true=1, y_pred_proba=0.72)
    snapshot = monitor.snapshot()
    html = monitor.generate_dashboard()
"""

from __future__ import annotations

import base64
import io
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class PredictionRecord:
    """One model prediction plus its eventual outcome."""
    timestamp: str
    y_true: int                # actual outcome: 1=win, 0=loss
    y_pred_proba: float        # model's predicted probability of win
    features: Optional[Dict[str, float]] = None


@dataclass
class DriftAlert:
    """A single drift alert event."""
    timestamp: str
    alert_type: str            # "feature_drift" | "concept_drift" | "accuracy_decay"
    severity: str              # "warning" | "critical"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorSnapshot:
    """Current monitoring state — consumed by online_retrain triggers."""
    timestamp: str
    n_predictions: int
    rolling_accuracy: Optional[float]
    rolling_auc: Optional[float]
    drifted_features: List[str]
    drift_fraction: float
    n_alerts: int
    should_retrain: bool
    retrain_reasons: List[str]


# ── KS-test (two-sample Kolmogorov-Smirnov) ─────────────────────────────


def ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Two-sample KS statistic (max absolute difference of empirical CDFs).

    Pure-numpy implementation — no scipy dependency required.

    Returns:
        KS statistic in [0, 1].  0 = identical distributions, 1 = no overlap.
    """
    a = np.sort(sample_a)
    b = np.sort(sample_b)
    all_vals = np.concatenate([a, b])
    all_vals = np.unique(all_vals)

    cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)

    return float(np.max(np.abs(cdf_a - cdf_b)))


def ks_critical_value(n1: int, n2: int, alpha: float = 0.05) -> float:
    """Approximate KS critical value for two samples at significance level alpha.

    Uses the asymptotic approximation: c(alpha) * sqrt((n1+n2) / (n1*n2))
    where c(0.05) = 1.36.
    """
    c_alpha = {0.01: 1.63, 0.05: 1.36, 0.10: 1.22}.get(alpha, 1.36)
    return c_alpha * math.sqrt((n1 + n2) / (n1 * n2))


# ── ModelMonitor ─────────────────────────────────────────────────────────


class ModelMonitor:
    """Rolling model health monitor with drift detection and alerting.

    Parameters
    ----------
    feature_means : np.ndarray, optional
        Training-time per-feature means (from SignalModel.feature_means).
    feature_stds : np.ndarray, optional
        Training-time per-feature stds (from SignalModel.feature_stds).
    feature_names : list of str, optional
        Feature names matching the means/stds arrays.
    training_samples : np.ndarray, optional
        Raw training feature matrix (n_samples, n_features) for KS tests.
        If not provided, KS tests use the means/stds z-score fallback.
    accuracy_window : int
        Rolling window size for accuracy tracking (default 50 trades).
    auc_window : int
        Rolling window size for AUC computation (default 100 trades).
    ks_threshold : float
        KS statistic threshold to flag a feature as drifted (default 0.15).
    drift_fraction_threshold : float
        Fraction of features drifted to generate an alert (default 0.20).
    auc_drop_threshold : float
        AUC drop from baseline that triggers an alert (default 0.05).
    accuracy_drop_threshold : float
        Absolute accuracy drop from baseline (default 0.10).
    baseline_auc : float, optional
        Training-time AUC for comparison.
    baseline_accuracy : float, optional
        Training-time accuracy for comparison.
    max_history : int
        Maximum prediction records to retain (default 2000).
    """

    def __init__(
        self,
        feature_means: Optional[np.ndarray] = None,
        feature_stds: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        training_samples: Optional[np.ndarray] = None,
        accuracy_window: int = 50,
        auc_window: int = 100,
        ks_threshold: float = 0.15,
        drift_fraction_threshold: float = 0.20,
        auc_drop_threshold: float = 0.05,
        accuracy_drop_threshold: float = 0.10,
        baseline_auc: Optional[float] = None,
        baseline_accuracy: Optional[float] = None,
        max_history: int = 2000,
    ):
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        self.feature_names = feature_names or []
        self.training_samples = training_samples
        self.accuracy_window = accuracy_window
        self.auc_window = auc_window
        self.ks_threshold = ks_threshold
        self.drift_fraction_threshold = drift_fraction_threshold
        self.auc_drop_threshold = auc_drop_threshold
        self.accuracy_drop_threshold = accuracy_drop_threshold
        self.baseline_auc = baseline_auc
        self.baseline_accuracy = baseline_accuracy
        self.max_history = max_history

        self._history: Deque[PredictionRecord] = deque(maxlen=max_history)
        self._alerts: List[DriftAlert] = []
        self._accuracy_series: List[Tuple[str, float]] = []
        self._auc_series: List[Tuple[str, float]] = []
        self._drift_series: List[Tuple[str, float]] = []

    # ── Recording predictions ────────────────────────────────────────

    def record(
        self,
        features: Optional[Dict[str, float]] = None,
        y_true: int = 0,
        y_pred_proba: float = 0.5,
        timestamp: Optional[str] = None,
    ) -> Optional[DriftAlert]:
        """Record a prediction and its outcome.  Returns an alert if triggered."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        self._history.append(PredictionRecord(
            timestamp=ts, y_true=y_true,
            y_pred_proba=y_pred_proba, features=features,
        ))

        # Update rolling metrics
        self._update_rolling_accuracy(ts)
        self._update_rolling_auc(ts)

        if features is not None:
            self._update_drift(ts, features)

        # Check thresholds and maybe fire alerts
        return self._check_alerts(ts)

    # ── Rolling accuracy ─────────────────────────────────────────────

    def _update_rolling_accuracy(self, ts: str) -> None:
        recent = list(self._history)[-self.accuracy_window:]
        if len(recent) < 10:
            return
        correct = sum(
            1 for r in recent
            if (r.y_pred_proba >= 0.5) == (r.y_true == 1)
        )
        acc = correct / len(recent)
        self._accuracy_series.append((ts, round(acc, 4)))

    def _update_rolling_auc(self, ts: str) -> None:
        recent = list(self._history)[-self.auc_window:]
        if len(recent) < 20:
            return
        y_true = np.array([r.y_true for r in recent])
        y_proba = np.array([r.y_pred_proba for r in recent])
        if len(np.unique(y_true)) < 2:
            return
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(y_true, y_proba))
        self._auc_series.append((ts, round(auc, 4)))

    # ── Feature drift ────────────────────────────────────────────────

    def _update_drift(self, ts: str, features: Dict[str, float]) -> None:
        """Run KS-test on each feature comparing recent vs training."""
        if not self.feature_names:
            return

        recent_records = [
            r for r in self._history
            if r.features is not None
        ][-self.accuracy_window:]

        if len(recent_records) < 20:
            return

        n_drifted = 0
        n_tested = 0

        for i, name in enumerate(self.feature_names):
            recent_vals = np.array([
                r.features.get(name, 0.0) for r in recent_records
            ])

            if self.training_samples is not None and i < self.training_samples.shape[1]:
                train_vals = self.training_samples[:, i]
                ks = ks_statistic(recent_vals, train_vals)
            elif self.feature_means is not None and self.feature_stds is not None:
                if i >= len(self.feature_means):
                    continue
                std = self.feature_stds[i]
                if std <= 0:
                    continue
                z_recent = (recent_vals - self.feature_means[i]) / std
                z_train = np.random.RandomState(42).normal(0, 1, 200)
                ks = ks_statistic(z_recent, z_train)
            else:
                continue

            n_tested += 1
            if ks > self.ks_threshold:
                n_drifted += 1

        if n_tested > 0:
            drift_frac = n_drifted / n_tested
            self._drift_series.append((ts, round(drift_frac, 4)))

    # ── Alert checking ───────────────────────────────────────────────

    def _check_alerts(self, ts: str) -> Optional[DriftAlert]:
        """Check all thresholds and fire an alert if any are breached."""
        # Accuracy decay
        if self._accuracy_series and self.baseline_accuracy is not None:
            current_acc = self._accuracy_series[-1][1]
            drop = self.baseline_accuracy - current_acc
            if drop >= self.accuracy_drop_threshold:
                alert = DriftAlert(
                    timestamp=ts,
                    alert_type="accuracy_decay",
                    severity="critical" if drop >= self.accuracy_drop_threshold * 2 else "warning",
                    message=f"Accuracy dropped {drop:.1%} from baseline ({self.baseline_accuracy:.1%} → {current_acc:.1%})",
                    details={"baseline": self.baseline_accuracy, "current": current_acc, "drop": drop},
                )
                self._alerts.append(alert)
                return alert

        # AUC degradation (concept drift)
        if self._auc_series and self.baseline_auc is not None:
            current_auc = self._auc_series[-1][1]
            drop = self.baseline_auc - current_auc
            if drop >= self.auc_drop_threshold:
                alert = DriftAlert(
                    timestamp=ts,
                    alert_type="concept_drift",
                    severity="critical" if drop >= self.auc_drop_threshold * 2 else "warning",
                    message=f"AUC dropped {drop:.4f} from baseline ({self.baseline_auc:.4f} → {current_auc:.4f})",
                    details={"baseline": self.baseline_auc, "current": current_auc, "drop": drop},
                )
                self._alerts.append(alert)
                return alert

        # Feature drift
        if self._drift_series:
            drift_frac = self._drift_series[-1][1]
            if drift_frac >= self.drift_fraction_threshold:
                alert = DriftAlert(
                    timestamp=ts,
                    alert_type="feature_drift",
                    severity="critical" if drift_frac >= self.drift_fraction_threshold * 2 else "warning",
                    message=f"{drift_frac:.0%} of features drifted (threshold: {self.drift_fraction_threshold:.0%})",
                    details={"drift_fraction": drift_frac},
                )
                self._alerts.append(alert)
                return alert

        return None

    # ── Snapshot for online_retrain integration ──────────────────────

    def snapshot(self) -> MonitorSnapshot:
        """Produce a snapshot of current monitoring state.

        The ``should_retrain`` and ``retrain_reasons`` fields can be used
        by :class:`~compass.online_retrain.ModelRetrainer` as an additional
        trigger source.
        """
        reasons: List[str] = []
        should_retrain = False

        rolling_acc = self._accuracy_series[-1][1] if self._accuracy_series else None
        rolling_auc = self._auc_series[-1][1] if self._auc_series else None
        drift_frac = self._drift_series[-1][1] if self._drift_series else 0.0

        # Check accuracy
        if rolling_acc is not None and self.baseline_accuracy is not None:
            if self.baseline_accuracy - rolling_acc >= self.accuracy_drop_threshold:
                should_retrain = True
                reasons.append(f"accuracy_decay ({rolling_acc:.1%} vs baseline {self.baseline_accuracy:.1%})")

        # Check AUC
        if rolling_auc is not None and self.baseline_auc is not None:
            if self.baseline_auc - rolling_auc >= self.auc_drop_threshold:
                should_retrain = True
                reasons.append(f"concept_drift (AUC {rolling_auc:.4f} vs baseline {self.baseline_auc:.4f})")

        # Check feature drift
        if drift_frac >= self.drift_fraction_threshold:
            should_retrain = True
            reasons.append(f"feature_drift ({drift_frac:.0%} drifted)")

        # Identify drifted features
        drifted = self._get_drifted_features()

        return MonitorSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_predictions=len(self._history),
            rolling_accuracy=rolling_acc,
            rolling_auc=rolling_auc,
            drifted_features=drifted,
            drift_fraction=drift_frac,
            n_alerts=len(self._alerts),
            should_retrain=should_retrain,
            retrain_reasons=reasons,
        )

    def _get_drifted_features(self) -> List[str]:
        """Return names of features currently flagged as drifted."""
        if not self.feature_names:
            return []
        recent = [r for r in self._history if r.features is not None][-self.accuracy_window:]
        if len(recent) < 20:
            return []

        drifted: List[str] = []
        for i, name in enumerate(self.feature_names):
            recent_vals = np.array([r.features.get(name, 0.0) for r in recent])
            if self.training_samples is not None and i < self.training_samples.shape[1]:
                ks = ks_statistic(recent_vals, self.training_samples[:, i])
            elif self.feature_means is not None and self.feature_stds is not None and i < len(self.feature_means):
                std = self.feature_stds[i]
                if std <= 0:
                    continue
                z = (recent_vals - self.feature_means[i]) / std
                ks = ks_statistic(z, np.random.RandomState(42).normal(0, 1, 200))
            else:
                continue
            if ks > self.ks_threshold:
                drifted.append(name)
        return drifted

    # ── Properties ───────────────────────────────────────────────────

    @property
    def alerts(self) -> List[DriftAlert]:
        return list(self._alerts)

    @property
    def prediction_count(self) -> int:
        return len(self._history)

    # ── HTML Dashboard ───────────────────────────────────────────────

    def generate_dashboard(self, title: str = "Model Monitoring Dashboard") -> str:
        """Generate a self-contained HTML monitoring dashboard."""
        charts = self._render_charts()
        snap = self.snapshot()
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Status badge
        if snap.should_retrain:
            status_badge = '<span class="badge critical">RETRAIN RECOMMENDED</span>'
        elif snap.n_alerts > 0:
            status_badge = '<span class="badge warning">ALERTS ACTIVE</span>'
        else:
            status_badge = '<span class="badge healthy">HEALTHY</span>'

        # Alert history table
        alert_rows = ""
        for a in reversed(self._alerts[-20:]):
            sev_cls = "critical" if a.severity == "critical" else "warning"
            alert_rows += (
                f'<tr class="{sev_cls}-row">'
                f'<td>{a.timestamp[:19]}</td>'
                f'<td><span class="badge {sev_cls}">{a.severity.upper()}</span></td>'
                f'<td>{a.alert_type}</td>'
                f'<td>{a.message}</td>'
                f'</tr>\n'
            )
        if not alert_rows:
            alert_rows = '<tr><td colspan="4" class="empty">No alerts generated</td></tr>'

        # Drifted features list
        drifted_html = ""
        if snap.drifted_features:
            drifted_html = "<ul>" + "".join(f"<li><code>{f}</code></li>" for f in snap.drifted_features) + "</ul>"
        else:
            drifted_html = "<p class='muted'>No features currently drifted</p>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .muted {{ color: #94a3b8; }}
  .empty {{ color: #94a3b8; text-align: center; font-style: italic; padding: 1.5em; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 130px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .badge {{ display: inline-block; padding: 3px 12px; border-radius: 12px;
            font-size: 0.78em; font-weight: 600; }}
  .badge.healthy {{ background: #dcfce7; color: #166534; }}
  .badge.warning {{ background: #fef3c7; color: #92400e; }}
  .badge.critical {{ background: #fecaca; color: #991b1b; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .warning-row {{ background: #fffbeb; }}
  .critical-row {{ background: #fef2f2; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  code {{ font-size: 0.85em; background: #f1f5f9; padding: 1px 4px; border-radius: 3px; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>{title} {status_badge}</h1>
<div class="meta">
  {snap.n_predictions} predictions tracked &middot;
  {snap.n_alerts} alerts fired &middot;
  Generated {now}
</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{snap.n_predictions}</div><div class="label">Predictions</div></div>
  <div class="kpi"><div class="value">{f'{snap.rolling_accuracy:.1%}' if snap.rolling_accuracy is not None else '—'}</div><div class="label">Rolling Accuracy ({self.accuracy_window})</div></div>
  <div class="kpi"><div class="value">{f'{snap.rolling_auc:.4f}' if snap.rolling_auc is not None else '—'}</div><div class="label">Rolling AUC ({self.auc_window})</div></div>
  <div class="kpi"><div class="value">{f'{snap.drift_fraction:.0%}'}</div><div class="label">Features Drifted</div></div>
  <div class="kpi"><div class="value">{snap.n_alerts}</div><div class="label">Total Alerts</div></div>
</div>

<h2>1. Accuracy &amp; AUC Over Time</h2>
{f'<div class="chart"><img src="data:image/png;base64,{charts["metrics"]}" alt="Metrics"></div>' if charts.get("metrics") else '<p class="muted">Insufficient data for accuracy/AUC tracking (need ≥20 predictions)</p>'}

<h2>2. Feature Drift Over Time</h2>
{f'<div class="chart"><img src="data:image/png;base64,{charts["drift"]}" alt="Feature Drift"></div>' if charts.get("drift") else '<p class="muted">No feature drift data yet</p>'}

<h2>3. Drifted Features (Current)</h2>
{drifted_html}

<h2>4. Alert History</h2>
<table>
<thead><tr><th>Timestamp</th><th>Severity</th><th>Type</th><th>Message</th></tr></thead>
<tbody>
{alert_rows}
</tbody>
</table>

<footer>Generated by <code>compass/model_monitor.py</code></footer>
</body>
</html>"""

        return html

    def _render_charts(self) -> Dict[str, str]:
        """Render monitoring charts as base64 PNGs."""
        charts: Dict[str, str] = {}

        if len(self._accuracy_series) >= 2 or len(self._auc_series) >= 2:
            charts["metrics"] = self._render_metrics_chart()

        if len(self._drift_series) >= 2:
            charts["drift"] = self._render_drift_chart()

        return charts

    def _render_metrics_chart(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(10, 4))

        if self._accuracy_series:
            indices = list(range(len(self._accuracy_series)))
            vals = [v for _, v in self._accuracy_series]
            ax1.plot(indices, vals, color="#2563eb", lw=1.5, label="Rolling Accuracy")
            if self.baseline_accuracy is not None:
                ax1.axhline(self.baseline_accuracy, color="#2563eb", ls="--", lw=0.8, alpha=0.5)

        ax1.set_ylabel("Accuracy", color="#2563eb")
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        if self._auc_series:
            ax2 = ax1.twinx()
            indices = list(range(len(self._auc_series)))
            vals = [v for _, v in self._auc_series]
            ax2.plot(indices, vals, color="#dc2626", lw=1.5, label="Rolling AUC")
            if self.baseline_auc is not None:
                ax2.axhline(self.baseline_auc, color="#dc2626", ls="--", lw=0.8, alpha=0.5)
            ax2.set_ylabel("AUC", color="#dc2626")
            ax2.set_ylim(0, 1)

        ax1.set_xlabel("Prediction index")
        ax1.set_title(f"Rolling Accuracy ({self.accuracy_window}) & AUC ({self.auc_window})", fontsize=11)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_drift_chart(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3.5))
        indices = list(range(len(self._drift_series)))
        vals = [v for _, v in self._drift_series]
        ax.plot(indices, vals, color="#d97706", lw=1.5)
        ax.axhline(self.drift_fraction_threshold, color="#dc2626", ls="--", lw=1, label=f"Threshold ({self.drift_fraction_threshold:.0%})")
        ax.fill_between(indices, vals, alpha=0.15, color="#d97706")
        ax.set_ylabel("Fraction of Features Drifted")
        ax.set_xlabel("Prediction index")
        ax.set_title("Feature Drift Over Time", fontsize=11)
        ax.set_ylim(0, max(max(vals) * 1.2, self.drift_fraction_threshold * 1.5) if vals else 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
