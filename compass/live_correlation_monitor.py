"""
Live strategy correlation monitor — real-time diversification tracking.

Monitors rolling correlations between experiment return streams and fires
alerts when correlations spike during market stress, reducing effective
diversification below safe thresholds.

Key concepts:
  - **Effective N**: Measures true diversification.  For *k* strategies with
    pairwise correlation *rho*, effective_N = k / (1 + (k-1)*rho_avg).
    When rho_avg → 1, effective_N → 1 (all strategies move together).
    When rho_avg → 0, effective_N → k (maximum diversification).
  - **Correlation regime**: Classifies the correlation environment as
    NORMAL (avg < 0.4), ELEVATED (0.4-0.7), or DANGER (> 0.7).
  - **Allocation adjustment**: When correlation spikes, recommends reducing
    allocation to the most-correlated pair to restore diversification.

Usage::

    from compass.live_correlation_monitor import LiveCorrelationMonitor
    monitor = LiveCorrelationMonitor(["EXP-400", "EXP-401", "EXP-600"])
    monitor.add_returns({"EXP-400": 0.002, "EXP-401": -0.001, "EXP-600": 0.003})
    snapshot = monitor.snapshot()
    alerts = monitor.check_alerts()
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
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "live_correlation.html"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class CorrelationAlert:
    """A correlation-based alert event."""
    timestamp: str
    alert_type: str            # "correlation_spike" | "diversification_low" | "regime_change"
    severity: str              # "warning" | "critical"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AllocationAdjustment:
    """Recommended allocation change for a single experiment."""
    experiment: str
    current_weight: float
    recommended_weight: float
    reason: str


@dataclass
class CorrelationSnapshot:
    """Point-in-time snapshot of the correlation state."""
    timestamp: str
    n_experiments: int
    n_observations: int
    avg_correlation: Optional[float]
    max_correlation: Optional[float]
    max_corr_pair: Optional[Tuple[str, str]]
    min_correlation: Optional[float]
    effective_n: Optional[float]
    correlation_regime: str       # "NORMAL" | "ELEVATED" | "DANGER"
    diversification_score: Optional[float]  # effective_n / n_experiments (0-1)
    adjustments: List[AllocationAdjustment]
    n_alerts: int


# ── Core math ────────────────────────────────────────────────────────────


def compute_effective_n(n_assets: int, avg_correlation: float) -> float:
    """Compute effective number of independent bets.

    From Markowitz diversification theory:
      effective_N = N / (1 + (N-1) * rho_avg)

    Args:
        n_assets: Number of strategies/experiments.
        avg_correlation: Average pairwise correlation.

    Returns:
        Effective N (1.0 to n_assets). Higher = more diversified.
    """
    if n_assets <= 1:
        return float(n_assets)
    denom = 1.0 + (n_assets - 1) * max(0.0, avg_correlation)
    if denom <= 0:
        return float(n_assets)
    return n_assets / denom


def classify_correlation_regime(avg_correlation: float) -> str:
    """Classify the correlation environment."""
    if avg_correlation >= 0.7:
        return "DANGER"
    elif avg_correlation >= 0.4:
        return "ELEVATED"
    return "NORMAL"


def pairwise_correlations(
    return_matrix: np.ndarray,
    labels: List[str],
) -> Dict[Tuple[str, str], float]:
    """Compute all pairwise correlations from a (n_obs, n_assets) matrix.

    Returns dict: {(label_i, label_j): correlation} for all i < j.
    """
    n = return_matrix.shape[1]
    if n < 2 or return_matrix.shape[0] < 3:
        return {}

    corr = np.corrcoef(return_matrix, rowvar=False)
    result: Dict[Tuple[str, str], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            val = corr[i, j]
            if not np.isnan(val):
                result[(labels[i], labels[j])] = round(float(val), 4)
    return result


# ── LiveCorrelationMonitor ───────────────────────────────────────────────


class LiveCorrelationMonitor:
    """Real-time rolling correlation tracker across experiment return streams.

    Args:
        experiments: List of experiment labels to track.
        window: Rolling window size for correlation (default 21 trading days).
        danger_threshold: Average correlation above this → DANGER regime.
        elevated_threshold: Average correlation above this → ELEVATED regime.
        pair_alert_threshold: Pairwise correlation above this fires alert.
        min_effective_n: Alert when effective_N drops below this.
        max_history: Maximum return observations to store.
        send_telegram: Whether to send Telegram alerts (default False).
    """

    def __init__(
        self,
        experiments: List[str],
        window: int = 21,
        danger_threshold: float = 0.70,
        elevated_threshold: float = 0.40,
        pair_alert_threshold: float = 0.70,
        min_effective_n: float = 1.5,
        max_history: int = 500,
        send_telegram: bool = False,
    ):
        self.experiments = list(experiments)
        self.window = window
        self.danger_threshold = danger_threshold
        self.elevated_threshold = elevated_threshold
        self.pair_alert_threshold = pair_alert_threshold
        self.min_effective_n = min_effective_n
        self.max_history = max_history
        self.send_telegram = send_telegram

        self._returns: Deque[Dict[str, float]] = deque(maxlen=max_history)
        self._timestamps: Deque[str] = deque(maxlen=max_history)
        self._alerts: List[CorrelationAlert] = []
        self._snapshot_history: List[CorrelationSnapshot] = []
        self._prev_regime: str = "NORMAL"

    # ── Adding data ──────────────────────────────────────────────────

    def add_returns(
        self,
        returns: Dict[str, float],
        timestamp: Optional[str] = None,
    ) -> Optional[CorrelationAlert]:
        """Record daily returns for tracked experiments.

        Args:
            returns: {experiment_label: daily_return_fraction}
            timestamp: ISO timestamp (defaults to now).

        Returns:
            Alert if a threshold was breached, else None.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        self._returns.append(returns)
        self._timestamps.append(ts)

        snap = self.snapshot()
        self._snapshot_history.append(snap)

        return self._check_alerts(snap, ts)

    # ── Snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> CorrelationSnapshot:
        """Compute current correlation state."""
        n = len(self._returns)
        n_exp = len(self.experiments)

        if n < 3 or n_exp < 2:
            return CorrelationSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                n_experiments=n_exp,
                n_observations=n,
                avg_correlation=None,
                max_correlation=None,
                max_corr_pair=None,
                min_correlation=None,
                effective_n=float(n_exp) if n_exp > 0 else None,
                correlation_regime="NORMAL",
                diversification_score=1.0 if n_exp > 0 else None,
                adjustments=[],
                n_alerts=len(self._alerts),
            )

        # Build return matrix from recent window
        matrix = self._build_return_matrix()
        if matrix is None or matrix.shape[0] < 3:
            return CorrelationSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                n_experiments=n_exp, n_observations=n,
                avg_correlation=None, max_correlation=None,
                max_corr_pair=None, min_correlation=None,
                effective_n=float(n_exp), correlation_regime="NORMAL",
                diversification_score=1.0, adjustments=[],
                n_alerts=len(self._alerts),
            )

        pairs = pairwise_correlations(matrix, self.experiments)
        if not pairs:
            return CorrelationSnapshot(
                timestamp=datetime.now(timezone.utc).isoformat(),
                n_experiments=n_exp, n_observations=n,
                avg_correlation=None, max_correlation=None,
                max_corr_pair=None, min_correlation=None,
                effective_n=float(n_exp), correlation_regime="NORMAL",
                diversification_score=1.0, adjustments=[],
                n_alerts=len(self._alerts),
            )

        corr_vals = list(pairs.values())
        avg_corr = float(np.mean(corr_vals))
        max_pair = max(pairs.items(), key=lambda x: x[1])
        min_pair = min(pairs.items(), key=lambda x: x[1])

        eff_n = compute_effective_n(n_exp, max(0, avg_corr))
        regime = classify_correlation_regime(avg_corr)
        div_score = eff_n / n_exp if n_exp > 0 else None

        adjustments = self._compute_adjustments(pairs, regime)

        return CorrelationSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            n_experiments=n_exp,
            n_observations=n,
            avg_correlation=round(avg_corr, 4),
            max_correlation=round(max_pair[1], 4),
            max_corr_pair=max_pair[0],
            min_correlation=round(min_pair[1], 4),
            effective_n=round(eff_n, 2),
            correlation_regime=regime,
            diversification_score=round(div_score, 4) if div_score is not None else None,
            adjustments=adjustments,
            n_alerts=len(self._alerts),
        )

    def _build_return_matrix(self) -> Optional[np.ndarray]:
        """Build (n_obs, n_experiments) matrix from recent window."""
        recent = list(self._returns)[-self.window:]
        if len(recent) < 3:
            return None

        rows = []
        for r in recent:
            row = [r.get(exp, 0.0) for exp in self.experiments]
            rows.append(row)

        matrix = np.array(rows, dtype=np.float64)

        # Check for constant columns (zero variance)
        stds = np.std(matrix, axis=0)
        if np.any(stds == 0):
            # Replace constant columns with tiny noise to avoid NaN correlations
            for i in range(matrix.shape[1]):
                if stds[i] == 0:
                    matrix[:, i] += np.random.RandomState(42).normal(0, 1e-10, matrix.shape[0])

        return matrix

    # ── Allocation adjustments ───────────────────────────────────────

    def _compute_adjustments(
        self,
        pairs: Dict[Tuple[str, str], float],
        regime: str,
    ) -> List[AllocationAdjustment]:
        """Generate allocation recommendations when correlation is elevated."""
        if regime == "NORMAL":
            return []

        n = len(self.experiments)
        equal_weight = 1.0 / n if n > 0 else 0.0
        adjustments: List[AllocationAdjustment] = []

        # Find the most-correlated pair
        if not pairs:
            return []
        worst_pair, worst_corr = max(pairs.items(), key=lambda x: x[1])

        if worst_corr < self.pair_alert_threshold:
            return []

        # Recommend reducing the experiment with worst individual performance
        # (proxy: higher average absolute return deviation = more volatile)
        recent = list(self._returns)[-self.window:]
        exp_a, exp_b = worst_pair
        vol_a = np.std([r.get(exp_a, 0) for r in recent])
        vol_b = np.std([r.get(exp_b, 0) for r in recent])
        reduce_exp = exp_a if vol_a > vol_b else exp_b
        keep_exp = exp_b if reduce_exp == exp_a else exp_a

        reduction = 0.15 if regime == "ELEVATED" else 0.30
        new_weight = max(0.05, equal_weight - reduction * equal_weight)
        redistribute = (equal_weight - new_weight) / max(n - 1, 1)

        adjustments.append(AllocationAdjustment(
            experiment=reduce_exp,
            current_weight=round(equal_weight, 4),
            recommended_weight=round(new_weight, 4),
            reason=f"High correlation ({worst_corr:.2f}) with {keep_exp} in {regime} regime",
        ))

        for exp in self.experiments:
            if exp != reduce_exp:
                adjustments.append(AllocationAdjustment(
                    experiment=exp,
                    current_weight=round(equal_weight, 4),
                    recommended_weight=round(equal_weight + redistribute, 4),
                    reason="Redistribute from correlated pair",
                ))

        return adjustments

    # ── Alert checking ───────────────────────────────────────────────

    def _check_alerts(
        self, snap: CorrelationSnapshot, ts: str,
    ) -> Optional[CorrelationAlert]:
        """Fire alerts when correlation thresholds are breached."""
        alert = None

        # Pair-level correlation spike
        if snap.max_correlation is not None and snap.max_correlation >= self.pair_alert_threshold:
            pair_str = f"{snap.max_corr_pair[0]} vs {snap.max_corr_pair[1]}" if snap.max_corr_pair else "?"
            alert = CorrelationAlert(
                timestamp=ts,
                alert_type="correlation_spike",
                severity="critical" if snap.max_correlation >= 0.85 else "warning",
                message=f"Correlation spike: {pair_str} at {snap.max_correlation:.2f}",
                details={"pair": snap.max_corr_pair, "correlation": snap.max_correlation},
            )

        # Effective N too low
        elif snap.effective_n is not None and snap.effective_n < self.min_effective_n:
            alert = CorrelationAlert(
                timestamp=ts,
                alert_type="diversification_low",
                severity="critical" if snap.effective_n < 1.2 else "warning",
                message=f"Effective N = {snap.effective_n:.2f} (min: {self.min_effective_n:.1f})",
                details={"effective_n": snap.effective_n},
            )

        # Regime change
        elif snap.correlation_regime != self._prev_regime:
            if snap.correlation_regime in ("ELEVATED", "DANGER"):
                alert = CorrelationAlert(
                    timestamp=ts,
                    alert_type="regime_change",
                    severity="critical" if snap.correlation_regime == "DANGER" else "warning",
                    message=f"Correlation regime: {self._prev_regime} → {snap.correlation_regime}",
                    details={"from": self._prev_regime, "to": snap.correlation_regime},
                )

        if alert:
            self._alerts.append(alert)
            if self.send_telegram:
                self._send_telegram_alert(alert)

        self._prev_regime = snap.correlation_regime
        return alert

    def _send_telegram_alert(self, alert: CorrelationAlert) -> None:
        """Send alert via Telegram (best-effort, never raises)."""
        try:
            from shared.telegram_alerts import send_message
            emoji = "\u26a0\ufe0f" if alert.severity == "warning" else "\U0001f6a8"
            text = f"{emoji} <b>Correlation Monitor</b>\n{alert.message}"
            send_message(text, parse_mode="HTML")
        except Exception as exc:
            logger.warning("Telegram alert failed: %s", exc)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def alerts(self) -> List[CorrelationAlert]:
        return list(self._alerts)

    @property
    def observation_count(self) -> int:
        return len(self._returns)

    # ── HTML Dashboard ───────────────────────────────────────────────

    def generate_dashboard(
        self, output: str = str(DEFAULT_OUTPUT),
    ) -> str:
        """Generate self-contained HTML dashboard."""
        snap = self.snapshot()
        charts = self._render_charts()
        html = self._build_html(snap, charts)

        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Dashboard written to %s", out)
        return str(out.resolve())

    def _render_charts(self) -> Dict[str, str]:
        charts: Dict[str, str] = {}
        if len(self._snapshot_history) >= 3:
            charts["div_score"] = self._chart_diversification_timeline()
            charts["avg_corr"] = self._chart_correlation_timeline()
        if len(self._returns) >= 5 and len(self.experiments) >= 2:
            charts["heatmap"] = self._chart_heatmap()
        return charts

    def _fig_to_b64(self, fig) -> str:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _chart_diversification_timeline(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = [s.diversification_score for s in self._snapshot_history if s.diversification_score is not None]
        if len(scores) < 2:
            return ""

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(range(len(scores)), scores, color="#2563eb", lw=1.5)
        ax.fill_between(range(len(scores)), scores, alpha=0.1, color="#2563eb")
        ax.axhline(1.0 / len(self.experiments) * self.min_effective_n if self.experiments else 0.5,
                    color="#dc2626", ls="--", lw=1, label=f"Min threshold")
        ax.set_ylabel("Diversification Score")
        ax.set_xlabel("Observation")
        ax.set_title("Diversification Score Over Time (1.0 = maximum)", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_correlation_timeline(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        avg_corrs = [s.avg_correlation for s in self._snapshot_history if s.avg_correlation is not None]
        if len(avg_corrs) < 2:
            return ""

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(range(len(avg_corrs)), avg_corrs, color="#d97706", lw=1.5)
        ax.fill_between(range(len(avg_corrs)), avg_corrs, alpha=0.1, color="#d97706")
        ax.axhline(self.danger_threshold, color="#dc2626", ls="--", lw=1, label=f"Danger ({self.danger_threshold})")
        ax.axhline(self.elevated_threshold, color="#d97706", ls="--", lw=0.8, label=f"Elevated ({self.elevated_threshold})")
        ax.set_ylabel("Avg Pairwise Correlation")
        ax.set_xlabel("Observation")
        ax.set_title("Average Correlation Over Time", fontsize=11)
        ax.set_ylim(-0.5, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_heatmap(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        matrix = self._build_return_matrix()
        if matrix is None:
            return ""

        corr = np.corrcoef(matrix, rowvar=False)
        n = len(self.experiments)

        fig, ax = plt.subplots(figsize=(max(4, 1.5 * n), max(3.5, 1.3 * n)))
        im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(self.experiments, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(self.experiments, fontsize=9)
        for i in range(n):
            for j in range(n):
                val = corr[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.6 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, fontweight="bold", color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Current Correlation Heatmap", fontsize=12)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, snap: CorrelationSnapshot, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        regime_cls = {"NORMAL": "healthy", "ELEVATED": "warning", "DANGER": "critical"}.get(snap.correlation_regime, "")
        regime_badge = f'<span class="badge {regime_cls}">{snap.correlation_regime}</span>'

        def _img(key):
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else '<p class="muted">Insufficient data</p>'

        def _fmt(v, f=".4f"):
            return f"{v:{f}}" if v is not None else "—"

        # Adjustment table
        adj_rows = ""
        for a in snap.adjustments:
            delta = a.recommended_weight - a.current_weight
            cls = "bad" if delta < 0 else "good"
            adj_rows += (
                f'<tr><td>{a.experiment}</td><td>{a.current_weight:.1%}</td>'
                f'<td class="{cls}">{a.recommended_weight:.1%}</td>'
                f'<td>{delta:+.1%}</td><td>{a.reason}</td></tr>\n'
            )
        if not adj_rows:
            adj_rows = '<tr><td colspan="5" class="muted">No adjustments needed — correlation is normal</td></tr>'

        # Alert table
        alert_rows = ""
        for a in reversed(self._alerts[-15:]):
            sev_cls = "critical" if a.severity == "critical" else "warning"
            alert_rows += f'<tr class="{sev_cls}-row"><td>{a.timestamp[:19]}</td><td><span class="badge {sev_cls}">{a.severity.upper()}</span></td><td>{a.alert_type}</td><td>{a.message}</td></tr>\n'
        if not alert_rows:
            alert_rows = '<tr><td colspan="4" class="muted">No alerts</td></tr>'

        pair_str = f"{snap.max_corr_pair[0]} / {snap.max_corr_pair[1]}" if snap.max_corr_pair else "—"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Live Correlation Monitor</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .muted {{ color: #94a3b8; font-style: italic; text-align: center; padding: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .badge {{ display: inline-block; padding: 3px 12px; border-radius: 12px; font-size: 0.78em; font-weight: 600; }}
  .badge.healthy {{ background: #dcfce7; color: #166534; }}
  .badge.warning {{ background: #fef3c7; color: #92400e; }}
  .badge.critical {{ background: #fecaca; color: #991b1b; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left; border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .warning-row {{ background: #fffbeb; }}
  .critical-row {{ background: #fef2f2; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0; font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>
<h1>Live Correlation Monitor {regime_badge}</h1>
<div class="meta">{snap.n_experiments} experiments &middot; {snap.n_observations} observations &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{_fmt(snap.avg_correlation)}</div><div class="label">Avg Correlation</div></div>
  <div class="kpi"><div class="value">{_fmt(snap.max_correlation)}</div><div class="label">Max Correlation</div></div>
  <div class="kpi"><div class="value">{_fmt(snap.effective_n, '.2f')}</div><div class="label">Effective N</div></div>
  <div class="kpi"><div class="value">{_fmt(snap.diversification_score)}</div><div class="label">Diversification</div></div>
  <div class="kpi"><div class="value">{pair_str}</div><div class="label">Most Correlated</div></div>
  <div class="kpi"><div class="value">{snap.n_alerts}</div><div class="label">Total Alerts</div></div>
</div>

<h2>1. Correlation Heatmap</h2>
{_img("heatmap")}

<h2>2. Diversification Score Timeline</h2>
{_img("div_score")}

<h2>3. Average Correlation Timeline</h2>
{_img("avg_corr")}

<h2>4. Allocation Adjustments</h2>
<table><thead><tr><th>Experiment</th><th>Current</th><th>Recommended</th><th>Change</th><th>Reason</th></tr></thead>
<tbody>{adj_rows}</tbody></table>

<h2>5. Alert History</h2>
<table><thead><tr><th>Time</th><th>Severity</th><th>Type</th><th>Message</th></tr></thead>
<tbody>{alert_rows}</tbody></table>

<footer>Generated by <code>compass/live_correlation_monitor.py</code></footer>
</body></html>"""
        return html
