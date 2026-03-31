"""Unified experiment dashboard aggregator – pulls data from all other dashboards
and reports into a single view with traffic-light status per experiment.

Aggregates: experiment registry, backtest results, stress tests, model monitoring,
correlation analysis, signal decay, and execution quality into per-experiment
status cards and portfolio-level overview.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Traffic-light constants ─────────────────────────────────────────────────
GREEN = "GREEN"
YELLOW = "YELLOW"
RED = "RED"

# ── Default targets ─────────────────────────────────────────────────────────
DEFAULT_TARGETS: Dict[str, float] = {
    "sharpe": 1.5,
    "max_dd_pct": 15.0,       # max acceptable drawdown %
    "win_rate": 70.0,          # percentage
    "model_auc": 0.60,
    "max_retrain_days": 30,
    "min_profit_factor": 1.5,
}

YELLOW_TOLERANCE = 0.20  # within 20% of target = YELLOW


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class TrafficLight:
    """Traffic-light status for a single metric."""
    metric: str
    value: float
    target: float
    status: str  # GREEN, YELLOW, RED
    detail: str = ""


@dataclass
class RecentTrade:
    """Summary of a recent trade."""
    date: str
    pnl: float
    ticker: str = ""
    spread_type: str = ""


@dataclass
class ModelHealth:
    """Model health snapshot for an experiment."""
    rolling_auc: float = 0.0
    rolling_accuracy: float = 0.0
    drift_fraction: float = 0.0
    n_alerts: int = 0
    should_retrain: bool = False
    health_status: str = GREEN    # GREEN / YELLOW / RED


@dataclass
class ExperimentStatus:
    """Per-experiment status card."""
    experiment_id: str
    name: str
    ticker: str
    status: str                  # overall GREEN/YELLOW/RED
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    return_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    model_auc: float = 0.0
    days_since_retrain: int = -1
    hedge_status: str = "unknown"
    capacity_estimate: float = 0.0
    signal_half_life: float = 0.0
    execution_quality: float = 0.0
    recent_trades: List[RecentTrade] = field(default_factory=list)
    model_health: Optional[ModelHealth] = None
    lights: List[TrafficLight] = field(default_factory=list)


@dataclass
class PortfolioOverview:
    """Portfolio-level aggregate metrics."""
    blended_sharpe: float = 0.0
    blended_dd_pct: float = 0.0
    blended_return_pct: float = 0.0
    diversification_score: float = 0.0
    total_aum_capacity: float = 0.0
    n_experiments: int = 0
    n_green: int = 0
    n_yellow: int = 0
    n_red: int = 0
    overall_status: str = GREEN


@dataclass
class DashboardResult:
    """Complete dashboard output."""
    experiments: List[ExperimentStatus] = field(default_factory=list)
    portfolio: Optional[PortfolioOverview] = None
    generated_at: str = ""


# ── Core dashboard aggregator ───────────────────────────────────────────────
class ExperimentDashboard:
    """Aggregates all experiment data into a unified dashboard."""

    def __init__(
        self,
        targets: Optional[Dict[str, float]] = None,
        yellow_tolerance: float = YELLOW_TOLERANCE,
    ) -> None:
        self.targets = targets or dict(DEFAULT_TARGETS)
        self.yellow_tolerance = yellow_tolerance

    # ── Public API ──────────────────────────────────────────────────────────
    def build(
        self,
        experiments: List[Dict[str, Any]],
        backtest_results: Optional[Dict[str, Dict[str, Any]]] = None,
        stress_results: Optional[Dict[str, Dict[str, Any]]] = None,
        model_snapshots: Optional[Dict[str, Dict[str, Any]]] = None,
        signal_decay: Optional[Dict[str, Dict[str, Any]]] = None,
        execution_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        recent_trades: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> DashboardResult:
        """Build unified dashboard from all data sources.

        Parameters
        ----------
        experiments : list of dict
            Registry entries, each with at least: experiment_id, name, ticker.
        backtest_results : dict, optional
            experiment_id → {sharpe, max_dd_pct, win_rate, return_pct, total_trades, profit_factor, ...}
        stress_results : dict, optional
            experiment_id → {hedged_dd, unhedged_dd, hedge_status, ...}
        model_snapshots : dict, optional
            experiment_id → {rolling_auc, days_since_retrain, should_retrain,
                             rolling_accuracy, drift_fraction, n_alerts, ...}
        signal_decay : dict, optional
            experiment_id → {half_life_hours, optimal_period, snr, ...}
        execution_stats : dict, optional
            experiment_id → {fill_rate, avg_slippage, execution_score, ...}
        correlation_matrix : pd.DataFrame, optional
            Experiment-by-experiment return correlation matrix.
        recent_trades : dict, optional
            experiment_id → list of {date, pnl, ticker?, spread_type?}
        """
        if not experiments:
            return DashboardResult(generated_at=self._now())

        backtest_results = backtest_results or {}
        stress_results = stress_results or {}
        model_snapshots = model_snapshots or {}
        signal_decay = signal_decay or {}
        execution_stats = execution_stats or {}
        recent_trades = recent_trades or {}

        exp_statuses: List[ExperimentStatus] = []
        for exp in experiments:
            eid = exp.get("experiment_id", "")
            status = self._build_experiment_status(
                exp,
                backtest_results.get(eid, {}),
                stress_results.get(eid, {}),
                model_snapshots.get(eid, {}),
                signal_decay.get(eid, {}),
                execution_stats.get(eid, {}),
                recent_trades.get(eid, []),
            )
            exp_statuses.append(status)

        portfolio = self._build_portfolio_overview(
            exp_statuses, correlation_matrix,
        )

        return DashboardResult(
            experiments=exp_statuses,
            portfolio=portfolio,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: DashboardResult,
        output_path: str | Path = "reports/experiment_dashboard.html",
    ) -> Path:
        """Write self-contained HTML dashboard."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Experiment dashboard written to %s", path)
        return path

    # ── Per-experiment status ───────────────────────────────────────────────
    def _build_experiment_status(
        self,
        exp: Dict[str, Any],
        bt: Dict[str, Any],
        stress: Dict[str, Any],
        model: Dict[str, Any],
        decay: Dict[str, Any],
        exec_stats: Dict[str, Any],
        trades_raw: Optional[List[Dict[str, Any]]] = None,
    ) -> ExperimentStatus:
        sharpe = float(bt.get("sharpe", 0.0))
        max_dd = float(bt.get("max_dd_pct", 0.0))
        win_rate = float(bt.get("win_rate", 0.0))
        return_pct = float(bt.get("return_pct", 0.0))
        total_trades = int(bt.get("total_trades", 0))
        pf = float(bt.get("profit_factor", 0.0))
        auc = float(model.get("rolling_auc", 0.0))
        retrain_days = int(model.get("days_since_retrain", -1))
        hedge_st = str(stress.get("hedge_status", "unknown"))
        capacity = float(bt.get("capacity_estimate", 0.0))
        half_life = float(decay.get("half_life_hours", 0.0))
        exec_score = float(exec_stats.get("execution_score", 0.0))

        # Recent trades
        recent: List[RecentTrade] = []
        for t in (trades_raw or []):
            recent.append(RecentTrade(
                date=str(t.get("date", "")),
                pnl=float(t.get("pnl", 0.0)),
                ticker=str(t.get("ticker", "")),
                spread_type=str(t.get("spread_type", t.get("type", ""))),
            ))

        # Model health
        mh = self._build_model_health(model)

        lights = self._evaluate_lights(
            sharpe=sharpe, max_dd=max_dd, win_rate=win_rate,
            auc=auc, retrain_days=retrain_days, pf=pf,
        )

        overall = self._overall_status(lights)

        return ExperimentStatus(
            experiment_id=exp.get("experiment_id", ""),
            name=exp.get("name", ""),
            ticker=exp.get("ticker", ""),
            status=overall,
            sharpe=sharpe,
            max_dd_pct=max_dd,
            return_pct=return_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            profit_factor=pf,
            model_auc=auc,
            days_since_retrain=retrain_days,
            hedge_status=hedge_st,
            capacity_estimate=capacity,
            signal_half_life=half_life,
            execution_quality=exec_score,
            recent_trades=recent,
            model_health=mh,
            lights=lights,
        )

    @staticmethod
    def _build_model_health(model: Dict[str, Any]) -> ModelHealth:
        """Derive model health status from snapshot data."""
        auc = float(model.get("rolling_auc", 0.0))
        acc = float(model.get("rolling_accuracy", 0.0))
        drift = float(model.get("drift_fraction", 0.0))
        alerts = int(model.get("n_alerts", 0))
        retrain = bool(model.get("should_retrain", False))

        if retrain or alerts >= 3 or drift > 0.5:
            health = RED
        elif alerts >= 1 or drift > 0.2 or auc < 0.55:
            health = YELLOW
        else:
            health = GREEN

        return ModelHealth(
            rolling_auc=auc,
            rolling_accuracy=acc,
            drift_fraction=drift,
            n_alerts=alerts,
            should_retrain=retrain,
            health_status=health,
        )

    def _evaluate_lights(
        self,
        sharpe: float,
        max_dd: float,
        win_rate: float,
        auc: float,
        retrain_days: int,
        pf: float,
    ) -> List[TrafficLight]:
        lights: List[TrafficLight] = []

        # Sharpe (higher is better)
        lights.append(self._light_higher_better("sharpe", sharpe, self.targets["sharpe"]))

        # Max drawdown (lower is better — invert)
        lights.append(self._light_lower_better("max_dd_pct", max_dd, self.targets["max_dd_pct"]))

        # Win rate (higher is better)
        lights.append(self._light_higher_better("win_rate", win_rate, self.targets["win_rate"]))

        # Model AUC (higher is better)
        if auc > 0:
            lights.append(self._light_higher_better("model_auc", auc, self.targets["model_auc"]))

        # Days since retrain (lower is better)
        if retrain_days >= 0:
            lights.append(self._light_lower_better(
                "days_since_retrain", float(retrain_days),
                self.targets["max_retrain_days"],
            ))

        # Profit factor (higher is better)
        if pf > 0:
            lights.append(self._light_higher_better(
                "profit_factor", pf, self.targets["min_profit_factor"],
            ))

        return lights

    def _light_higher_better(
        self, metric: str, value: float, target: float,
    ) -> TrafficLight:
        """GREEN if value >= target, YELLOW if within tolerance, RED otherwise."""
        if target <= 0:
            return TrafficLight(metric, value, target, GREEN)
        ratio = value / target
        if ratio >= 1.0:
            return TrafficLight(metric, value, target, GREEN,
                                f"{value:.2f} >= {target:.2f}")
        if ratio >= (1.0 - self.yellow_tolerance):
            return TrafficLight(metric, value, target, YELLOW,
                                f"{value:.2f} within {self.yellow_tolerance:.0%} of {target:.2f}")
        return TrafficLight(metric, value, target, RED,
                            f"{value:.2f} < {target:.2f}")

    def _light_lower_better(
        self, metric: str, value: float, target: float,
    ) -> TrafficLight:
        """GREEN if value <= target, YELLOW if within tolerance above, RED otherwise."""
        if target <= 0:
            return TrafficLight(metric, value, target, GREEN)
        if value <= target:
            return TrafficLight(metric, value, target, GREEN,
                                f"{value:.2f} <= {target:.2f}")
        ratio = value / target
        if ratio <= (1.0 + self.yellow_tolerance):
            return TrafficLight(metric, value, target, YELLOW,
                                f"{value:.2f} within {self.yellow_tolerance:.0%} of {target:.2f}")
        return TrafficLight(metric, value, target, RED,
                            f"{value:.2f} > {target:.2f}")

    @staticmethod
    def _overall_status(lights: List[TrafficLight]) -> str:
        if not lights:
            return GREEN
        if any(l.status == RED for l in lights):
            return RED
        if any(l.status == YELLOW for l in lights):
            return YELLOW
        return GREEN

    # ── Portfolio overview ──────────────────────────────────────────────────
    def _build_portfolio_overview(
        self,
        experiments: List[ExperimentStatus],
        corr_matrix: Optional[pd.DataFrame],
    ) -> PortfolioOverview:
        n = len(experiments)
        if n == 0:
            return PortfolioOverview()

        sharpes = [e.sharpe for e in experiments if e.sharpe != 0]
        dds = [e.max_dd_pct for e in experiments if e.max_dd_pct != 0]
        rets = [e.return_pct for e in experiments]
        caps = [e.capacity_estimate for e in experiments]

        blended_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
        blended_dd = float(np.max(dds)) if dds else 0.0
        blended_ret = float(np.mean(rets)) if rets else 0.0
        total_cap = float(np.sum(caps))

        div_score = self._diversification_score(corr_matrix, n)

        n_green = sum(1 for e in experiments if e.status == GREEN)
        n_yellow = sum(1 for e in experiments if e.status == YELLOW)
        n_red = sum(1 for e in experiments if e.status == RED)

        if n_red > 0:
            overall = RED
        elif n_yellow > n // 2:
            overall = YELLOW
        else:
            overall = GREEN

        return PortfolioOverview(
            blended_sharpe=blended_sharpe,
            blended_dd_pct=blended_dd,
            blended_return_pct=blended_ret,
            diversification_score=div_score,
            total_aum_capacity=total_cap,
            n_experiments=n,
            n_green=n_green,
            n_yellow=n_yellow,
            n_red=n_red,
            overall_status=overall,
        )

    @staticmethod
    def _diversification_score(
        corr_matrix: Optional[pd.DataFrame], n_exp: int,
    ) -> float:
        """0–100 score: 100 = perfectly uncorrelated, 0 = fully correlated."""
        if corr_matrix is None or corr_matrix.empty or n_exp < 2:
            return 50.0  # neutral default

        # Average off-diagonal absolute correlation
        mask = ~np.eye(len(corr_matrix), dtype=bool)
        off_diag = np.abs(corr_matrix.values[mask])
        if len(off_diag) == 0:
            return 50.0
        avg_corr = float(np.mean(off_diag))
        # Map: 0 corr → 100, 1 corr → 0
        return max(0.0, min(100.0, (1.0 - avg_corr) * 100.0))

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: DashboardResult) -> str:
        portfolio_cards = self._html_portfolio_cards(r.portfolio)
        exp_cards = self._html_experiment_cards(r.experiments)
        detail_table = self._html_detail_table(r.experiments)
        lights_summary = self._html_lights_summary(r.experiments)
        model_health_table = self._html_model_health(r.experiments)
        recent_trades_section = self._html_recent_trades(r.experiments)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Experiment Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.82rem}}
th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.GREEN{{color:#4ade80}}.YELLOW{{color:#fbbf24}}.RED{{color:#f87171}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
.dot.GREEN{{background:#4ade80}}.dot.YELLOW{{background:#fbbf24}}.dot.RED{{background:#f87171}}
.exp-card{{background:#1e293b;border-radius:10px;padding:16px;border-left:4px solid #475569}}
.exp-card.GREEN{{border-color:#4ade80}}.exp-card.YELLOW{{border-color:#fbbf24}}.exp-card.RED{{border-color:#f87171}}
.exp-card h3{{font-size:.95rem;margin-bottom:8px}}
.exp-card .metrics{{display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;font-size:.8rem}}
.exp-card .metrics .k{{color:#94a3b8}}.exp-card .metrics .v{{font-weight:600}}
.exp-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px;margin-bottom:28px}}
.trade-line{{font-size:.75rem;color:#94a3b8;margin-top:2px}}
.trade-line .pnl-pos{{color:#4ade80}}.trade-line .pnl-neg{{color:#f87171}}
</style>
</head>
<body>
<h1>Experiment Dashboard</h1>
<p class="sub">Generated {r.generated_at or 'N/A'}</p>

{portfolio_cards}

<div class="sec">
<h2>Experiment Status Cards</h2>
<div class="exp-grid">{exp_cards}</div>
</div>

{detail_table}
{model_health_table}
{recent_trades_section}
{lights_summary}

</body>
</html>"""

    # ── Portfolio cards ─────────────────────────────────────────────────────
    @staticmethod
    def _html_portfolio_cards(p: Optional[PortfolioOverview]) -> str:
        if not p:
            return ""
        return f"""<div class="grid">
<div class="card"><div class="lbl">Overall Status</div><div class="val {p.overall_status}"><span class="dot {p.overall_status}"></span>{p.overall_status}</div></div>
<div class="card"><div class="lbl">Blended Sharpe</div><div class="val">{p.blended_sharpe:.2f}</div></div>
<div class="card"><div class="lbl">Max Drawdown</div><div class="val">{p.blended_dd_pct:.1f}%</div></div>
<div class="card"><div class="lbl">Avg Return</div><div class="val">{p.blended_return_pct:.1f}%</div></div>
<div class="card"><div class="lbl">Diversification</div><div class="val">{p.diversification_score:.0f}/100</div></div>
<div class="card"><div class="lbl">AUM Capacity</div><div class="val">${p.total_aum_capacity:,.0f}</div></div>
<div class="card"><div class="lbl">Experiments</div><div class="val">{p.n_experiments}</div></div>
<div class="card"><div class="lbl">Traffic Lights</div><div class="val"><span class="GREEN">{p.n_green}G</span> <span class="YELLOW">{p.n_yellow}Y</span> <span class="RED">{p.n_red}R</span></div></div>
</div>"""

    # ── Experiment cards ────────────────────────────────────────────────────
    @staticmethod
    def _html_experiment_cards(exps: List[ExperimentStatus]) -> str:
        if not exps:
            return "<p>No experiments.</p>"
        cards = ""
        for e in exps:
            hedge_display = e.hedge_status or "unknown"
            retrain = f"{e.days_since_retrain}d" if e.days_since_retrain >= 0 else "N/A"

            # Model health badge
            mh = e.model_health
            if mh:
                health_badge = f'<span class="{mh.health_status}">{mh.health_status}</span>'
            else:
                health_badge = "N/A"

            # Recent trades mini-list (last 3)
            trade_lines = ""
            for t in e.recent_trades[:3]:
                cls = "pnl-pos" if t.pnl >= 0 else "pnl-neg"
                trade_lines += (
                    f'<div class="trade-line">{t.date} '
                    f'<span class="{cls}">${t.pnl:+,.0f}</span> {t.spread_type}</div>'
                )

            cards += f"""<div class="exp-card {e.status}">
<h3><span class="dot {e.status}"></span>{e.experiment_id} — {e.name}</h3>
<div class="metrics">
<span class="k">Ticker</span><span class="v">{e.ticker}</span>
<span class="k">Sharpe</span><span class="v">{e.sharpe:.2f}</span>
<span class="k">Max DD</span><span class="v">{e.max_dd_pct:.1f}%</span>
<span class="k">Return</span><span class="v">{e.return_pct:.1f}%</span>
<span class="k">Win Rate</span><span class="v">{e.win_rate:.1f}%</span>
<span class="k">Trades</span><span class="v">{e.total_trades}</span>
<span class="k">AUC</span><span class="v">{e.model_auc:.3f}</span>
<span class="k">Retrain</span><span class="v">{retrain}</span>
<span class="k">Hedge</span><span class="v">{hedge_display}</span>
<span class="k">Capacity</span><span class="v">${e.capacity_estimate:,.0f}</span>
<span class="k">Model Health</span><span class="v">{health_badge}</span>
</div>
{trade_lines}
</div>"""
        return cards

    # ── Detail table ────────────────────────────────────────────────────────
    @staticmethod
    def _html_detail_table(exps: List[ExperimentStatus]) -> str:
        if not exps:
            return ""
        rows = ""
        for e in exps:
            rows += (
                f'<tr><td><span class="dot {e.status}"></span>{e.experiment_id}</td>'
                f"<td>{e.name}</td><td>{e.ticker}</td>"
                f"<td>{e.sharpe:.2f}</td><td>{e.max_dd_pct:.1f}%</td>"
                f"<td>{e.win_rate:.1f}%</td><td>{e.profit_factor:.2f}</td>"
                f"<td>{e.model_auc:.3f}</td>"
                f"<td>{e.signal_half_life:.1f}h</td>"
                f"<td>{e.execution_quality:.1f}</td>"
                f'<td class="{e.status}">{e.status}</td></tr>'
            )
        return f"""<div class="sec">
<h2>Experiment Detail</h2>
<table>
<thead><tr><th>ID</th><th>Name</th><th>Ticker</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>PF</th><th>AUC</th><th>Half-Life</th><th>Exec Q</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    # ── Model health table ─────────────────────────────────────────────────
    @staticmethod
    def _html_model_health(exps: List[ExperimentStatus]) -> str:
        has_health = any(e.model_health is not None for e in exps)
        if not has_health:
            return ""
        rows = ""
        for e in exps:
            mh = e.model_health
            if not mh:
                continue
            rows += (
                f'<tr><td>{e.experiment_id}</td>'
                f"<td>{mh.rolling_auc:.3f}</td>"
                f"<td>{mh.rolling_accuracy:.1%}</td>"
                f"<td>{mh.drift_fraction:.1%}</td>"
                f"<td>{mh.n_alerts}</td>"
                f'<td>{"Yes" if mh.should_retrain else "No"}</td>'
                f'<td class="{mh.health_status}"><span class="dot {mh.health_status}"></span>{mh.health_status}</td></tr>'
            )
        return f"""<div class="sec">
<h2>Model Health</h2>
<table>
<thead><tr><th>Experiment</th><th>AUC</th><th>Accuracy</th><th>Drift</th><th>Alerts</th><th>Retrain?</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    # ── Recent trades section ───────────────────────────────────────────────
    @staticmethod
    def _html_recent_trades(exps: List[ExperimentStatus]) -> str:
        has_trades = any(len(e.recent_trades) > 0 for e in exps)
        if not has_trades:
            return ""
        rows = ""
        for e in exps:
            for t in e.recent_trades[:5]:
                cls = "GREEN" if t.pnl >= 0 else "RED"
                rows += (
                    f"<tr><td>{e.experiment_id}</td>"
                    f"<td>{t.date}</td>"
                    f"<td>{t.ticker}</td>"
                    f"<td>{t.spread_type}</td>"
                    f'<td class="{cls}">${t.pnl:+,.2f}</td></tr>'
                )
        return f"""<div class="sec">
<h2>Recent Trades</h2>
<table>
<thead><tr><th>Experiment</th><th>Date</th><th>Ticker</th><th>Type</th><th>P&L</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    # ── Lights summary ──────────────────────────────────────────────────────
    @staticmethod
    def _html_lights_summary(exps: List[ExperimentStatus]) -> str:
        if not exps:
            return ""
        rows = ""
        for e in exps:
            for l in e.lights:
                rows += (
                    f'<tr><td>{e.experiment_id}</td><td>{l.metric}</td>'
                    f"<td>{l.value:.2f}</td><td>{l.target:.2f}</td>"
                    f'<td class="{l.status}"><span class="dot {l.status}"></span>{l.status}</td>'
                    f"<td>{l.detail}</td></tr>"
                )
        return f"""<div class="sec">
<h2>Traffic Light Breakdown</h2>
<table>
<thead><tr><th>Experiment</th><th>Metric</th><th>Value</th><th>Target</th><th>Status</th><th>Detail</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
