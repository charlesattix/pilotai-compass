"""
Real-time production monitoring system.

Tracks:
  - Strategy P&L (per-experiment and portfolio)
  - Position exposure (delta, gamma, vega, theta)
  - Margin utilization
  - Fill rates and slippage
  - Signal freshness
  - Model prediction accuracy
  - Risk limit breaches
  - System latency

Alert engine:
  - Configurable thresholds per metric
  - Severity levels (INFO, WARNING, CRITICAL)
  - Telegram-formatted alert messages
  - Alert cooldowns to prevent spam

State persistence via JSON.
HTML dashboard with metric cards, alert history, health gauges.

Usage::

    from compass.prod_monitor import ProductionMonitor
    mon = ProductionMonitor(config)
    mon.record_trade(trade_data)
    mon.record_fill(fill_data)
    snapshot = mon.snapshot()
    ProductionMonitor.generate_report(snapshot)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "prod_monitor.html"
DEFAULT_STATE_PATH = ROOT / "reports" / "prod_monitor_state.json"


# ── Severity ─────────────────────────────────────────────────────────────


class Severity:
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @staticmethod
    def rank(s: str) -> int:
        return {"INFO": 0, "WARNING": 1, "CRITICAL": 2}.get(s, -1)


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class AlertThreshold:
    """Threshold for a single metric."""

    metric: str
    warning_value: float
    critical_value: float
    direction: str = "above"  # "above" or "below"
    cooldown_seconds: float = 300.0  # 5 min default


@dataclass
class MonitorConfig:
    """Full monitor configuration."""

    # P&L
    max_daily_loss: float = 5_000.0
    max_drawdown_pct: float = 0.10
    # Exposure
    max_delta: float = 500.0
    max_gamma: float = 100.0
    max_vega: float = 5_000.0
    # Margin
    max_margin_utilization: float = 0.80
    # Fill quality
    min_fill_rate: float = 0.80
    max_avg_slippage_bps: float = 10.0
    # Signal/model
    max_signal_age_minutes: float = 60.0
    min_model_accuracy: float = 0.50
    # System
    max_latency_ms: float = 500.0
    # Alert cooldown
    alert_cooldown_seconds: float = 300.0


DEFAULT_THRESHOLDS = [
    AlertThreshold("daily_pnl", -3000, -5000, "below"),
    AlertThreshold("drawdown_pct", 0.07, 0.10, "above"),
    AlertThreshold("abs_delta", 300, 500, "above"),
    AlertThreshold("abs_vega", 3000, 5000, "above"),
    AlertThreshold("margin_util", 0.60, 0.80, "above"),
    AlertThreshold("fill_rate", 0.85, 0.70, "below"),
    AlertThreshold("avg_slippage_bps", 7.0, 12.0, "above"),
    AlertThreshold("signal_age_min", 30.0, 60.0, "above"),
    AlertThreshold("model_accuracy", 0.55, 0.45, "below"),
    AlertThreshold("latency_ms", 200, 500, "above"),
]


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class GreeksExposure:
    """Portfolio Greeks exposure."""

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {"delta": self.delta, "gamma": self.gamma,
                "vega": self.vega, "theta": self.theta}


@dataclass
class PnLState:
    """P&L tracking state."""

    daily_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown_pct: float = 0.0
    n_trades_today: int = 0
    per_experiment: Dict[str, float] = field(default_factory=dict)


@dataclass
class FillMetrics:
    """Fill quality metrics."""

    total_fills: int = 0
    total_orders: int = 0
    fill_rate: float = 1.0
    total_slippage_bps: float = 0.0
    avg_slippage_bps: float = 0.0


@dataclass
class ModelMetrics:
    """Model/signal health metrics."""

    signal_age_minutes: float = 0.0
    model_accuracy: float = 0.0
    n_predictions: int = 0
    n_correct: int = 0
    last_signal_time: Optional[str] = None


@dataclass
class SystemMetrics:
    """System health metrics."""

    latency_ms: float = 0.0
    uptime_seconds: float = 0.0
    errors_last_hour: int = 0


@dataclass
class Alert:
    """A single alert."""

    timestamp: str
    severity: str
    metric: str
    message: str
    value: float
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "severity": self.severity,
            "metric": self.metric,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
        }

    def telegram_format(self) -> str:
        """Format alert for Telegram."""
        emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(self.severity, "📊")
        return (
            f"{emoji} *{self.severity}*: {self.metric}\n"
            f"{self.message}\n"
            f"Value: `{self.value:.4f}` | Threshold: `{self.threshold:.4f}`\n"
            f"Time: {self.timestamp}"
        )


@dataclass
class MonitorSnapshot:
    """Full point-in-time monitor snapshot."""

    timestamp: str
    pnl: PnLState
    greeks: GreeksExposure
    margin_utilization: float
    fills: FillMetrics
    model: ModelMetrics
    system: SystemMetrics
    alerts: List[Alert]
    alert_history: List[Alert]
    n_active_positions: int
    health_score: float  # 0-100


# ── Alert engine ─────────────────────────────────────────────────────────


class AlertEngine:
    """Configurable alert engine with cooldowns."""

    def __init__(
        self,
        thresholds: Optional[List[AlertThreshold]] = None,
    ):
        self.thresholds = thresholds or list(DEFAULT_THRESHOLDS)
        self._last_alert_time: Dict[str, float] = {}

    def check(self, metric: str, value: float) -> Optional[Alert]:
        """Check a metric value against thresholds."""
        for t in self.thresholds:
            if t.metric != metric:
                continue

            # Cooldown check
            now = time.time()
            key = f"{metric}_{t.direction}"
            last = self._last_alert_time.get(key, 0)
            if now - last < t.cooldown_seconds:
                return None

            triggered_severity = None
            triggered_threshold = 0.0

            if t.direction == "above":
                if value >= t.critical_value:
                    triggered_severity = Severity.CRITICAL
                    triggered_threshold = t.critical_value
                elif value >= t.warning_value:
                    triggered_severity = Severity.WARNING
                    triggered_threshold = t.warning_value
            else:  # below
                if value <= t.critical_value:
                    triggered_severity = Severity.CRITICAL
                    triggered_threshold = t.critical_value
                elif value <= t.warning_value:
                    triggered_severity = Severity.WARNING
                    triggered_threshold = t.warning_value

            if triggered_severity:
                self._last_alert_time[key] = now
                return Alert(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    severity=triggered_severity,
                    metric=metric,
                    message=f"{metric} = {value:.4f} breached {triggered_severity.lower()} threshold {triggered_threshold:.4f}",
                    value=value,
                    threshold=triggered_threshold,
                )

        return None

    def check_all(self, metrics: Dict[str, float]) -> List[Alert]:
        """Check all metrics at once."""
        alerts: List[Alert] = []
        for metric, value in metrics.items():
            alert = self.check(metric, value)
            if alert:
                alerts.append(alert)
        return alerts

    def reset_cooldowns(self) -> None:
        self._last_alert_time.clear()


# ── Health score ─────────────────────────────────────────────────────────


def compute_health_score(
    pnl: PnLState,
    greeks: GreeksExposure,
    margin: float,
    fills: FillMetrics,
    model: ModelMetrics,
    system: SystemMetrics,
    config: MonitorConfig,
) -> float:
    """Compute overall system health (0-100).

    Each component contributes equally: P&L, greeks, margin,
    fills, model, system = 6 components × ~16.7 points each.
    """
    scores: List[float] = []

    # P&L health (0-16.7)
    dd_ratio = min(pnl.drawdown_pct / config.max_drawdown_pct, 1.0) if config.max_drawdown_pct > 0 else 0
    scores.append(16.7 * (1 - dd_ratio))

    # Greeks health
    delta_ratio = min(abs(greeks.delta) / config.max_delta, 1.0) if config.max_delta > 0 else 0
    scores.append(16.7 * (1 - delta_ratio))

    # Margin health
    margin_ratio = min(margin / config.max_margin_utilization, 1.0) if config.max_margin_utilization > 0 else 0
    scores.append(16.7 * (1 - margin_ratio))

    # Fill health
    fill_score = fills.fill_rate if fills.fill_rate > 0 else 1.0
    scores.append(16.7 * fill_score)

    # Model health
    model_score = min(model.model_accuracy / 0.7, 1.0) if model.n_predictions > 0 else 0.5
    scores.append(16.7 * model_score)

    # System health
    lat_ratio = min(system.latency_ms / config.max_latency_ms, 1.0) if config.max_latency_ms > 0 else 0
    scores.append(16.7 * (1 - lat_ratio))

    return min(100.0, max(0.0, sum(scores)))


# ── State persistence ────────────────────────────────────────────────────


def save_state(snapshot: MonitorSnapshot, path: Path) -> None:
    """Save monitor state to JSON."""
    data = {
        "timestamp": snapshot.timestamp,
        "pnl": {
            "daily": snapshot.pnl.daily_pnl,
            "cumulative": snapshot.pnl.cumulative_pnl,
            "peak": snapshot.pnl.peak_equity,
            "drawdown": snapshot.pnl.drawdown_pct,
            "trades_today": snapshot.pnl.n_trades_today,
            "per_experiment": snapshot.pnl.per_experiment,
        },
        "greeks": snapshot.greeks.to_dict(),
        "margin": snapshot.margin_utilization,
        "fills": {
            "total": snapshot.fills.total_fills,
            "orders": snapshot.fills.total_orders,
            "rate": snapshot.fills.fill_rate,
            "avg_slippage": snapshot.fills.avg_slippage_bps,
        },
        "model": {
            "signal_age": snapshot.model.signal_age_minutes,
            "accuracy": snapshot.model.model_accuracy,
            "predictions": snapshot.model.n_predictions,
            "correct": snapshot.model.n_correct,
        },
        "system": {
            "latency": snapshot.system.latency_ms,
            "uptime": snapshot.system.uptime_seconds,
            "errors": snapshot.system.errors_last_hour,
        },
        "health_score": snapshot.health_score,
        "n_active": snapshot.n_active_positions,
        "alerts": [a.to_dict() for a in snapshot.alert_history[-50:]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_state(path: Path) -> Optional[Dict[str, Any]]:
    """Load monitor state from JSON."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ── Core monitor ─────────────────────────────────────────────────────────


class ProductionMonitor:
    """Real-time production monitoring system."""

    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        thresholds: Optional[List[AlertThreshold]] = None,
        initial_capital: float = 100_000.0,
    ):
        self.config = config or MonitorConfig()
        self.alert_engine = AlertEngine(thresholds)
        self.initial_capital = initial_capital

        # State
        self._pnl = PnLState(peak_equity=initial_capital)
        self._greeks = GreeksExposure()
        self._margin = 0.0
        self._fills = FillMetrics()
        self._model = ModelMetrics()
        self._system = SystemMetrics()
        self._n_active = 0
        self._alert_history: List[Alert] = []
        self._start_time = time.time()

    # ── Recording methods ────────────────────────────────────────────

    def record_trade(
        self,
        pnl: float,
        experiment: str = "default",
    ) -> List[Alert]:
        """Record a completed trade."""
        self._pnl.daily_pnl += pnl
        self._pnl.cumulative_pnl += pnl
        self._pnl.n_trades_today += 1
        self._pnl.per_experiment[experiment] = (
            self._pnl.per_experiment.get(experiment, 0.0) + pnl
        )

        equity = self.initial_capital + self._pnl.cumulative_pnl
        self._pnl.peak_equity = max(self._pnl.peak_equity, equity)
        if self._pnl.peak_equity > 0:
            self._pnl.drawdown_pct = (self._pnl.peak_equity - equity) / self._pnl.peak_equity

        return self._check_alerts()

    def record_greeks(self, delta: float, gamma: float, vega: float, theta: float) -> List[Alert]:
        """Update portfolio Greeks."""
        self._greeks = GreeksExposure(delta=delta, gamma=gamma, vega=vega, theta=theta)
        return self._check_alerts()

    def record_margin(self, utilization: float) -> List[Alert]:
        """Update margin utilization (0-1)."""
        self._margin = utilization
        return self._check_alerts()

    def record_fill(self, filled: bool, slippage_bps: float = 0.0) -> List[Alert]:
        """Record an order fill attempt."""
        self._fills.total_orders += 1
        if filled:
            self._fills.total_fills += 1
        self._fills.fill_rate = (
            self._fills.total_fills / self._fills.total_orders
            if self._fills.total_orders > 0 else 1.0
        )
        self._fills.total_slippage_bps += slippage_bps
        self._fills.avg_slippage_bps = (
            self._fills.total_slippage_bps / max(self._fills.total_fills, 1)
        )
        return self._check_alerts()

    def record_signal(self, age_minutes: float) -> List[Alert]:
        """Record signal freshness."""
        self._model.signal_age_minutes = age_minutes
        self._model.last_signal_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self._check_alerts()

    def record_prediction(self, predicted: int, actual: int) -> List[Alert]:
        """Record a model prediction outcome."""
        self._model.n_predictions += 1
        if predicted == actual:
            self._model.n_correct += 1
        self._model.model_accuracy = (
            self._model.n_correct / self._model.n_predictions
            if self._model.n_predictions > 0 else 0.0
        )
        return self._check_alerts()

    def record_latency(self, latency_ms: float) -> List[Alert]:
        """Record system latency."""
        self._system.latency_ms = latency_ms
        return self._check_alerts()

    def record_error(self) -> None:
        """Record a system error."""
        self._system.errors_last_hour += 1

    def set_active_positions(self, n: int) -> None:
        """Update active position count."""
        self._n_active = n

    def reset_daily(self) -> None:
        """Reset daily counters."""
        self._pnl.daily_pnl = 0.0
        self._pnl.n_trades_today = 0
        self._fills = FillMetrics()
        self._system.errors_last_hour = 0

    # ── Snapshot ─────────────────────────────────────────────────────

    def snapshot(self) -> MonitorSnapshot:
        """Capture current state."""
        self._system.uptime_seconds = time.time() - self._start_time
        health = compute_health_score(
            self._pnl, self._greeks, self._margin,
            self._fills, self._model, self._system, self.config,
        )
        new_alerts = self._check_alerts()

        return MonitorSnapshot(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            pnl=PnLState(
                daily_pnl=self._pnl.daily_pnl,
                cumulative_pnl=self._pnl.cumulative_pnl,
                peak_equity=self._pnl.peak_equity,
                drawdown_pct=self._pnl.drawdown_pct,
                n_trades_today=self._pnl.n_trades_today,
                per_experiment=dict(self._pnl.per_experiment),
            ),
            greeks=GreeksExposure(
                delta=self._greeks.delta, gamma=self._greeks.gamma,
                vega=self._greeks.vega, theta=self._greeks.theta,
            ),
            margin_utilization=self._margin,
            fills=FillMetrics(
                total_fills=self._fills.total_fills,
                total_orders=self._fills.total_orders,
                fill_rate=self._fills.fill_rate,
                total_slippage_bps=self._fills.total_slippage_bps,
                avg_slippage_bps=self._fills.avg_slippage_bps,
            ),
            model=ModelMetrics(
                signal_age_minutes=self._model.signal_age_minutes,
                model_accuracy=self._model.model_accuracy,
                n_predictions=self._model.n_predictions,
                n_correct=self._model.n_correct,
                last_signal_time=self._model.last_signal_time,
            ),
            system=SystemMetrics(
                latency_ms=self._system.latency_ms,
                uptime_seconds=self._system.uptime_seconds,
                errors_last_hour=self._system.errors_last_hour,
            ),
            alerts=new_alerts,
            alert_history=list(self._alert_history[-100:]),
            n_active_positions=self._n_active,
            health_score=health,
        )

    def save(self, path: Path = DEFAULT_STATE_PATH) -> None:
        """Persist current state."""
        save_state(self.snapshot(), path)

    # ── Internal ─────────────────────────────────────────────────────

    def _check_alerts(self) -> List[Alert]:
        metrics = {
            "daily_pnl": self._pnl.daily_pnl,
            "drawdown_pct": self._pnl.drawdown_pct,
            "abs_delta": abs(self._greeks.delta),
            "abs_vega": abs(self._greeks.vega),
            "margin_util": self._margin,
            "fill_rate": self._fills.fill_rate,
            "avg_slippage_bps": self._fills.avg_slippage_bps,
            "signal_age_min": self._model.signal_age_minutes,
            "model_accuracy": self._model.model_accuracy,
            "latency_ms": self._system.latency_ms,
        }
        new_alerts = self.alert_engine.check_all(metrics)
        self._alert_history.extend(new_alerts)
        return new_alerts

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        snapshot: MonitorSnapshot,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(snapshot)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fd(v: float) -> str:
    return f"${v:,.2f}"


def _fp(v: float) -> str:
    return f"{v:.1%}"


def _fr(v: float) -> str:
    return f"{v:.2f}"


def _health_color(score: float) -> str:
    if score >= 80:
        return "#3fb950"
    if score >= 50:
        return "#d29922"
    return "#f85149"


def _sev_color(sev: str) -> str:
    return {"INFO": "#58a6ff", "WARNING": "#d29922", "CRITICAL": "#f85149"}.get(sev, "#8b949e")


def _gauge_svg(value: float, max_val: float, label: str, w: int = 140, h: int = 80) -> str:
    pct = min(value / max_val, 1.0) if max_val > 0 else 0
    color = "#3fb950" if pct < 0.6 else "#d29922" if pct < 0.8 else "#f85149"
    bar_w = w - 20
    fill_w = bar_w * pct
    return f"""<svg viewBox="0 0 {w} {h}" class="gauge">
      <text x="{w//2}" y="18" text-anchor="middle" font-size="10" fill="#8b949e">{label}</text>
      <rect x="10" y="25" width="{bar_w}" height="14" fill="#21262d" rx="7"/>
      <rect x="10" y="25" width="{fill_w:.0f}" height="14" fill="{color}" rx="7"/>
      <text x="{w//2}" y="55" text-anchor="middle" font-size="14" fill="#f0f6fc" font-weight="bold">{_fr(value)}</text>
      <text x="{w//2}" y="70" text-anchor="middle" font-size="9" fill="#8b949e">/ {_fr(max_val)}</text>
    </svg>"""


def _build_html(snap: MonitorSnapshot) -> str:
    p = snap.pnl
    g = snap.greeks
    f = snap.fills
    m = snap.model
    s = snap.system
    hc = _health_color(snap.health_score)

    # Experiment P&L rows
    exp_rows = ""
    for name, pnl in sorted(p.per_experiment.items()):
        color = "#3fb950" if pnl >= 0 else "#f85149"
        exp_rows += f"<tr><td style='text-align:left'>{name}</td><td style='color:{color}'>{_fd(pnl)}</td></tr>"
    exp_table = f"<table class='dt'><tr><th style='text-align:left'>Experiment</th><th>P&L</th></tr>{exp_rows}</table>" if exp_rows else ""

    # Alert history rows
    alert_rows = ""
    for a in reversed(snap.alert_history[-20:]):
        sc = _sev_color(a.severity)
        alert_rows += f"<tr><td>{a.timestamp}</td><td style='color:{sc}'>{a.severity}</td><td style='text-align:left'>{a.metric}</td><td style='text-align:left'>{a.message}</td><td>{_fr(a.value)}</td></tr>"
    alert_table = f"""<table class='dt'><tr><th>Time</th><th>Severity</th><th style='text-align:left'>Metric</th><th style='text-align:left'>Message</th><th>Value</th></tr>{alert_rows}</table>""" if alert_rows else "<p class='meta'>No alerts.</p>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><title>Production Monitor</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
  h1,h2,h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; }}
  .health {{ background: #161b22; border: 2px solid {hc}; border-radius: 12px;
             padding: 20px; text-align: center; margin: 20px 0; }}
  .health .big {{ font-size: 3em; font-weight: 800; color: {hc}; }}
  .health .label {{ color: #8b949e; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px; margin: 20px 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 14px; text-align: center; }}
  .card .label {{ color: #8b949e; font-size: 0.8em; }}
  .card .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.2em; }}
  .gauges {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
             gap: 8px; margin: 16px 0; }}
  .gauge {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; }}
  table.dt {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }}
  table.dt th, table.dt td {{ padding: 5px 8px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.dt th {{ color: #8b949e; background: #161b22; }}
</style>
</head>
<body>
<h1>Production Monitor</h1>
<p class="meta">{snap.timestamp} &middot; Uptime: {s.uptime_seconds:.0f}s &middot;
   {snap.n_active_positions} active positions &middot;
   {len(snap.alert_history)} total alerts</p>

<div class="health">
  <div class="big">{snap.health_score:.0f}</div>
  <div class="label">System Health Score</div>
</div>

<div class="cards">
  <div class="card"><div class="label">Daily P&L</div><div class="value" style="color:{'#3fb950' if p.daily_pnl >= 0 else '#f85149'}">{_fd(p.daily_pnl)}</div></div>
  <div class="card"><div class="label">Cumulative P&L</div><div class="value">{_fd(p.cumulative_pnl)}</div></div>
  <div class="card"><div class="label">Drawdown</div><div class="value">{_fp(p.drawdown_pct)}</div></div>
  <div class="card"><div class="label">Trades Today</div><div class="value">{p.n_trades_today}</div></div>
  <div class="card"><div class="label">Delta</div><div class="value">{_fr(g.delta)}</div></div>
  <div class="card"><div class="label">Gamma</div><div class="value">{_fr(g.gamma)}</div></div>
  <div class="card"><div class="label">Vega</div><div class="value">{_fr(g.vega)}</div></div>
  <div class="card"><div class="label">Theta</div><div class="value">{_fr(g.theta)}</div></div>
  <div class="card"><div class="label">Margin Util</div><div class="value">{_fp(snap.margin_utilization)}</div></div>
  <div class="card"><div class="label">Fill Rate</div><div class="value">{_fp(f.fill_rate)}</div></div>
  <div class="card"><div class="label">Avg Slippage</div><div class="value">{_fr(f.avg_slippage_bps)} bps</div></div>
  <div class="card"><div class="label">Model Accuracy</div><div class="value">{_fp(m.model_accuracy)}</div></div>
  <div class="card"><div class="label">Signal Age</div><div class="value">{_fr(m.signal_age_minutes)}m</div></div>
  <div class="card"><div class="label">Latency</div><div class="value">{_fr(s.latency_ms)}ms</div></div>
  <div class="card"><div class="label">Errors/hr</div><div class="value">{s.errors_last_hour}</div></div>
</div>

<h2>Per-Experiment P&L</h2>
{exp_table}

<h2>Alert History</h2>
{alert_table}

</body></html>"""
