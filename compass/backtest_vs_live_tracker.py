"""
Backtest vs Live Tracker — compares paper trading results against
backtest expectations in real-time.

Features:
  1. Load backtest baseline metrics (CAGR, DD, Sharpe, WR)
  2. Track live paper trades from SQLite DB
  3. Compute drift metrics (CAGR deviation, Sharpe degradation, WR delta)
  4. Generate alerts when live deviates >30% from backtest
  5. HTML comparison chart (backtest curve vs live curve)

Usage::

    from compass.backtest_vs_live_tracker import BacktestVsLiveTracker
    tracker = BacktestVsLiveTracker(baseline, db_path="data/exp880/pilotai_exp880.db")
    result = tracker.evaluate()
    BacktestVsLiveTracker.generate_report(result)
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "backtest_vs_live.html"
TRADING_DAYS = 252


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class BacktestBaseline:
    """Expected performance from backtesting."""

    experiment_id: str = "EXP-880"
    cagr_pct: float = 76.9
    sharpe: float = 4.97
    max_dd_pct: float = 10.2
    win_rate: float = 0.87
    profit_factor: float = 5.10
    avg_pnl_per_trade: float = 1084.0
    trades_per_year: float = 31.8
    annual_return_pct: float = 15.9
    capital: float = 100_000.0


@dataclass
class LiveTrade:
    """A single live paper trade."""

    trade_id: int
    entry_date: str
    exit_date: str
    pnl: float
    win: bool
    strategy_type: str = ""
    regime: str = ""


@dataclass
class DriftMetric:
    """Deviation between live and backtest for one metric."""

    metric_name: str
    backtest_value: float
    live_value: float
    absolute_diff: float
    relative_diff_pct: float
    within_tolerance: bool
    tolerance_pct: float


@dataclass
class DriftAlert:
    """Alert when live drifts from backtest."""

    metric: str
    severity: str  # "warning", "critical"
    message: str
    backtest_value: float
    live_value: float
    deviation_pct: float


@dataclass
class TrackerResult:
    """Full comparison result."""

    baseline: BacktestBaseline
    live_trades: List[LiveTrade]
    drift_metrics: List[DriftMetric]
    alerts: List[DriftAlert]
    # Live aggregate metrics
    live_total_pnl: float
    live_win_rate: float
    live_sharpe: float
    live_max_dd_pct: float
    live_profit_factor: float
    live_avg_pnl: float
    live_n_trades: int
    live_days_active: int
    live_annualised_return_pct: float
    # Comparison
    overall_health: str  # "healthy", "degraded", "critical"
    n_alerts: int
    n_critical: int
    backtest_equity: List[float]
    live_equity: List[float]


# ── Trade loading from SQLite ────────────────────────────────────────────


def load_trades_from_db(db_path: Path) -> List[LiveTrade]:
    """Load closed trades from paper trading SQLite database."""
    if not db_path.exists():
        return []

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Try common table names
        tables_to_try = [
            "SELECT id, entry_date, exit_date, pnl, strategy_type, regime FROM trades WHERE exit_date IS NOT NULL AND pnl IS NOT NULL ORDER BY exit_date",
            "SELECT rowid as id, entry_date, exit_date, pnl, '' as strategy_type, '' as regime FROM closed_trades ORDER BY exit_date",
            "SELECT rowid as id, date as entry_date, date as exit_date, pnl, type as strategy_type, '' as regime FROM trade_log WHERE pnl IS NOT NULL ORDER BY date",
        ]

        for query in tables_to_try:
            try:
                rows = conn.execute(query).fetchall()
                if rows:
                    trades = []
                    for r in rows:
                        trades.append(LiveTrade(
                            trade_id=int(r["id"]),
                            entry_date=str(r["entry_date"])[:10],
                            exit_date=str(r["exit_date"])[:10],
                            pnl=float(r["pnl"]),
                            win=float(r["pnl"]) > 0,
                            strategy_type=str(r["strategy_type"] or ""),
                            regime=str(r["regime"] or ""),
                        ))
                    conn.close()
                    return trades
            except sqlite3.OperationalError:
                continue

        conn.close()
    except Exception as e:
        logger.warning("Failed to load trades from %s: %s", db_path, e)

    return []


def load_trades_from_dataframe(df: pd.DataFrame) -> List[LiveTrade]:
    """Load trades from a DataFrame (for testing or CSV input)."""
    trades = []
    for idx, row in df.iterrows():
        trades.append(LiveTrade(
            trade_id=int(idx),
            entry_date=str(row.get("entry_date", ""))[:10],
            exit_date=str(row.get("exit_date", ""))[:10],
            pnl=float(row.get("pnl", 0)),
            win=float(row.get("pnl", 0)) > 0,
            strategy_type=str(row.get("strategy_type", "")),
            regime=str(row.get("regime", "")),
        ))
    return trades


# ── Metrics computation ──────────────────────────────────────────────────


def compute_live_metrics(
    trades: List[LiveTrade],
    capital: float,
) -> Dict[str, float]:
    """Compute aggregate metrics from live trades."""
    if not trades:
        return {
            "total_pnl": 0, "win_rate": 0, "sharpe": 0,
            "max_dd_pct": 0, "profit_factor": 0, "avg_pnl": 0,
            "n_trades": 0, "days_active": 0, "annualised_return_pct": 0,
        }

    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    wins = sum(1 for t in trades if t.win)
    total = float(pnls.sum())

    # Sharpe
    mu = pnls.mean()
    std = pnls.std(ddof=1) if n > 1 else 1.0
    sharpe = float(mu / std * math.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0

    # Max DD
    equity = capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(abs(dd.min()) * 100)

    # Profit factor
    gains = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    pf = float(gains / losses) if losses > 1e-12 else (10.0 if gains > 0 else 0.0)

    # Days active
    try:
        dates = [pd.Timestamp(t.exit_date) for t in trades if t.exit_date]
        if len(dates) >= 2:
            days = (max(dates) - min(dates)).days
        else:
            days = 0
    except Exception:
        days = 0

    # Annualised return
    ann_ret = total / capital * (365 / max(days, 1)) * 100 if days > 0 else 0.0

    return {
        "total_pnl": total,
        "win_rate": wins / n if n > 0 else 0.0,
        "sharpe": sharpe,
        "max_dd_pct": max_dd,
        "profit_factor": min(pf, 50.0),
        "avg_pnl": float(pnls.mean()),
        "n_trades": n,
        "days_active": days,
        "annualised_return_pct": ann_ret,
    }


# ── Drift computation ────────────────────────────────────────────────────


def compute_drift(
    baseline: BacktestBaseline,
    live_metrics: Dict[str, float],
    tolerance_pct: float = 30.0,
) -> List[DriftMetric]:
    """Compare live metrics against backtest baseline."""
    comparisons = [
        ("win_rate", baseline.win_rate, live_metrics["win_rate"]),
        ("sharpe", baseline.sharpe, live_metrics["sharpe"]),
        ("max_dd_pct", baseline.max_dd_pct, live_metrics["max_dd_pct"]),
        ("profit_factor", baseline.profit_factor, live_metrics["profit_factor"]),
        ("avg_pnl", baseline.avg_pnl_per_trade, live_metrics["avg_pnl"]),
    ]

    drifts: List[DriftMetric] = []
    for name, bt_val, live_val in comparisons:
        abs_diff = live_val - bt_val
        rel_diff = (abs_diff / abs(bt_val) * 100) if abs(bt_val) > 1e-12 else 0.0

        # For DD, higher live is worse (invert tolerance direction)
        if name == "max_dd_pct":
            within = live_val <= bt_val * (1 + tolerance_pct / 100)
        else:
            # For return metrics, lower live is worse
            within = abs(rel_diff) <= tolerance_pct

        drifts.append(DriftMetric(
            metric_name=name,
            backtest_value=bt_val,
            live_value=live_val,
            absolute_diff=abs_diff,
            relative_diff_pct=rel_diff,
            within_tolerance=within,
            tolerance_pct=tolerance_pct,
        ))

    return drifts


# ── Alert generation ─────────────────────────────────────────────────────


def generate_alerts(
    drifts: List[DriftMetric],
    warning_pct: float = 30.0,
    critical_pct: float = 50.0,
) -> List[DriftAlert]:
    """Generate alerts for significant deviations."""
    alerts: List[DriftAlert] = []

    for d in drifts:
        abs_dev = abs(d.relative_diff_pct)

        # Skip small deviations
        if abs_dev < warning_pct:
            continue

        # Determine if deviation is bad (metric-dependent direction)
        is_bad = False
        if d.metric_name in ("win_rate", "sharpe", "profit_factor", "avg_pnl"):
            is_bad = d.live_value < d.backtest_value  # lower is worse
        elif d.metric_name == "max_dd_pct":
            is_bad = d.live_value > d.backtest_value  # higher DD is worse

        if not is_bad:
            continue  # outperforming backtest is not an alert

        severity = "critical" if abs_dev >= critical_pct else "warning"

        alerts.append(DriftAlert(
            metric=d.metric_name,
            severity=severity,
            message=(
                f"{d.metric_name}: live {d.live_value:.3f} vs backtest {d.backtest_value:.3f} "
                f"({d.relative_diff_pct:+.1f}% deviation)"
            ),
            backtest_value=d.backtest_value,
            live_value=d.live_value,
            deviation_pct=d.relative_diff_pct,
        ))

    return alerts


# ── Equity curve projection ──────────────────────────────────────────────


def project_backtest_equity(
    baseline: BacktestBaseline,
    n_trades: int,
) -> List[float]:
    """Project expected equity curve from backtest stats."""
    if n_trades <= 0:
        return [baseline.capital]
    equity = [baseline.capital]
    for i in range(n_trades):
        # Use avg PnL adjusted for win rate
        if np.random.RandomState(42 + i).random() < baseline.win_rate:
            pnl = abs(baseline.avg_pnl_per_trade)
        else:
            pnl = -abs(baseline.avg_pnl_per_trade) * 0.8  # losses slightly smaller
        equity.append(equity[-1] + pnl)
    return equity


def build_live_equity(
    trades: List[LiveTrade],
    capital: float,
) -> List[float]:
    """Build equity curve from live trades."""
    equity = [capital]
    for t in trades:
        equity.append(equity[-1] + t.pnl)
    return equity


# ── Overall health assessment ────────────────────────────────────────────


def assess_health(alerts: List[DriftAlert]) -> str:
    """Assess overall system health from alerts."""
    n_critical = sum(1 for a in alerts if a.severity == "critical")
    n_warning = sum(1 for a in alerts if a.severity == "warning")

    if n_critical >= 2:
        return "critical"
    if n_critical >= 1 or n_warning >= 3:
        return "degraded"
    return "healthy"


# ── Core tracker ─────────────────────────────────────────────────────────


class BacktestVsLiveTracker:
    """Compares live paper trading against backtest expectations."""

    def __init__(
        self,
        baseline: Optional[BacktestBaseline] = None,
        db_path: Optional[Path] = None,
        trades_df: Optional[pd.DataFrame] = None,
        tolerance_pct: float = 30.0,
        warning_pct: float = 30.0,
        critical_pct: float = 50.0,
    ):
        self.baseline = baseline or BacktestBaseline()
        self.db_path = Path(db_path) if db_path else None
        self.trades_df = trades_df
        self.tolerance_pct = tolerance_pct
        self.warning_pct = warning_pct
        self.critical_pct = critical_pct

    def evaluate(self) -> TrackerResult:
        """Run full comparison."""
        baseline = self.baseline

        # Load live trades
        if self.trades_df is not None:
            live_trades = load_trades_from_dataframe(self.trades_df)
        elif self.db_path and self.db_path.exists():
            live_trades = load_trades_from_db(self.db_path)
        else:
            live_trades = []

        # Compute live metrics
        live = compute_live_metrics(live_trades, baseline.capital)

        # Drift
        drifts = compute_drift(baseline, live, self.tolerance_pct)

        # Alerts
        alerts = generate_alerts(drifts, self.warning_pct, self.critical_pct)

        # Health
        health = assess_health(alerts)

        # Equity curves
        bt_equity = project_backtest_equity(baseline, live["n_trades"])
        live_equity = build_live_equity(live_trades, baseline.capital)

        return TrackerResult(
            baseline=baseline,
            live_trades=live_trades,
            drift_metrics=drifts,
            alerts=alerts,
            live_total_pnl=live["total_pnl"],
            live_win_rate=live["win_rate"],
            live_sharpe=live["sharpe"],
            live_max_dd_pct=live["max_dd_pct"],
            live_profit_factor=live["profit_factor"],
            live_avg_pnl=live["avg_pnl"],
            live_n_trades=live["n_trades"],
            live_days_active=live["days_active"],
            live_annualised_return_pct=live["annualised_return_pct"],
            overall_health=health,
            n_alerts=len(alerts),
            n_critical=sum(1 for a in alerts if a.severity == "critical"),
            backtest_equity=bt_equity,
            live_equity=live_equity,
        )

    @staticmethod
    def generate_report(
        result: TrackerResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fr(v: float) -> str:
    return f"{v:.2f}"


def _fp(v: float) -> str:
    return f"{v:.1%}"


def _fd(v: float) -> str:
    return f"${v:,.2f}"


HEALTH_COLORS = {"healthy": "#3fb950", "degraded": "#d29922", "critical": "#f85149"}


def _dual_line_svg(bt_vals: List[float], live_vals: List[float], title: str) -> str:
    """SVG with two overlaid equity curves."""
    if len(bt_vals) < 2 and len(live_vals) < 2:
        return ""

    w, h = 700, 220
    pad = 55
    all_vals = bt_vals + live_vals
    y_min = min(all_vals)
    y_max = max(all_vals)
    if y_max <= y_min:
        y_max = y_min + 1
    n = max(len(bt_vals), len(live_vals))
    pw, ph = w - 2 * pad, h - 65

    def tx(i):
        return pad + i / max(n - 1, 1) * pw

    def ty(v):
        return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="20" text-anchor="middle" class="st">{title}</text>')

    # Backtest line (dashed blue)
    if len(bt_vals) >= 2:
        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(bt_vals[i]):.1f}" for i in range(len(bt_vals)))
        parts.append(f'<path d="{d}" fill="none" stroke="#58a6ff" stroke-width="2" stroke-dasharray="6,3"/>')

    # Live line (solid green)
    if len(live_vals) >= 2:
        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(live_vals[i]):.1f}" for i in range(len(live_vals)))
        parts.append(f'<path d="{d}" fill="none" stroke="#3fb950" stroke-width="2.5"/>')

    # Legend
    parts.append(f'<rect x="{w - 160}" y="{h - 30}" width="12" height="3" fill="#58a6ff"/>')
    parts.append(f'<text x="{w - 145}" y="{h - 25}" font-size="9" fill="#8b949e">Backtest</text>')
    parts.append(f'<rect x="{w - 90}" y="{h - 30}" width="12" height="3" fill="#3fb950"/>')
    parts.append(f'<text x="{w - 75}" y="{h - 25}" font-size="9" fill="#8b949e">Live</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _build_html(r: TrackerResult) -> str:
    bl = r.baseline
    hc = HEALTH_COLORS.get(r.overall_health, "#8b949e")

    # Drift table
    drift_rows = ""
    for d in r.drift_metrics:
        color = "#3fb950" if d.within_tolerance else "#f85149"
        icon = "&#10003;" if d.within_tolerance else "&#10007;"
        drift_rows += (
            f"<tr><td style='text-align:left'>{d.metric_name}</td>"
            f"<td>{_fr(d.backtest_value)}</td><td>{_fr(d.live_value)}</td>"
            f"<td style='color:{color}'>{d.relative_diff_pct:+.1f}%</td>"
            f"<td style='color:{color}'>{icon}</td></tr>"
        )

    # Alert rows
    alert_rows = ""
    for a in r.alerts:
        ac = "#f85149" if a.severity == "critical" else "#d29922"
        alert_rows += (
            f"<tr><td style='color:{ac}'>{a.severity.upper()}</td>"
            f"<td style='text-align:left'>{a.metric}</td>"
            f"<td style='text-align:left'>{a.message}</td></tr>"
        )
    alert_html = (
        f"<table class='dt'><tr><th>Severity</th><th style='text-align:left'>Metric</th>"
        f"<th style='text-align:left'>Message</th></tr>{alert_rows}</table>"
        if alert_rows else "<p class='meta'>No alerts — all metrics within tolerance.</p>"
    )

    # Equity chart
    eq_svg = _dual_line_svg(r.backtest_equity, r.live_equity, "Backtest vs Live Equity ($)")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Backtest vs Live Tracker</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2{{color:#58a6ff}}.meta{{color:#8b949e}}
.hero{{background:#161b22;border:2px solid {hc};border-radius:12px;padding:24px;text-align:center;margin:20px 0}}
.hero .big{{font-size:2.2em;font-weight:800;color:{hc}}}.hero .sub{{color:#8b949e;margin-top:6px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
table.dt{{width:100%;border-collapse:collapse;margin:12px 0}}
table.dt th,table.dt td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}
table.dt th{{color:#8b949e;background:#161b22}}
.chart{{width:100%;max-width:750px;margin:16px auto;display:block}}.st{{fill:#58a6ff;font-size:13px}}
</style></head><body>
<h1>Backtest vs Live Tracker</h1>
<div class="hero">
<div class="big">{r.overall_health.upper()}</div>
<div class="sub">{bl.experiment_id} &middot; {r.live_n_trades} live trades &middot;
   {r.live_days_active} days active &middot; {r.n_alerts} alerts ({r.n_critical} critical)</div>
</div>

<div class="cards">
<div class="c"><div class="l">Live PnL</div><div class="v">{_fd(r.live_total_pnl)}</div></div>
<div class="c"><div class="l">Live Win Rate</div><div class="v">{_fp(r.live_win_rate)}</div></div>
<div class="c"><div class="l">Live Sharpe</div><div class="v">{_fr(r.live_sharpe)}</div></div>
<div class="c"><div class="l">Live Max DD</div><div class="v">{_fr(r.live_max_dd_pct)}%</div></div>
<div class="c"><div class="l">BT Win Rate</div><div class="v">{_fp(bl.win_rate)}</div></div>
<div class="c"><div class="l">BT Sharpe</div><div class="v">{_fr(bl.sharpe)}</div></div>
<div class="c"><div class="l">BT Max DD</div><div class="v">{_fr(bl.max_dd_pct)}%</div></div>
<div class="c"><div class="l">Live Avg PnL</div><div class="v">{_fd(r.live_avg_pnl)}</div></div>
</div>

<h2>Equity Comparison</h2>
{eq_svg}

<h2>Drift Metrics</h2>
<table class="dt"><tr><th style="text-align:left">Metric</th><th>Backtest</th><th>Live</th><th>Deviation</th><th>OK?</th></tr>{drift_rows}</table>

<h2>Alerts</h2>
{alert_html}

</body></html>"""
