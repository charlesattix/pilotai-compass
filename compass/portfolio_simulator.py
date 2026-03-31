"""
compass/portfolio_simulator.py — Multi-experiment portfolio simulation with
dynamic allocation, regime-adaptive weighting, and event-driven scaling.

Usage::

    from compass.portfolio_simulator import PortfolioSimulator, ExperimentAlloc

    sim = PortfolioSimulator(
        experiments={"EXP-400": df400, "EXP-401": df401},
        allocations={"EXP-400": 0.6, "EXP-401": 0.4},
    )
    sim.run()
    sim.generate_report("reports/portfolio_simulation.html")
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STARTING_CAPITAL = 100_000.0

# ── Regime allocation profiles ───────────────────────────────────────────

# Default regime-adaptive weight adjustments (multiplied onto base weights).
# Values > 1.0 = overweight, < 1.0 = underweight.
DEFAULT_REGIME_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "bull":     {"momentum": 1.3, "defensive": 0.7, "neutral": 1.0},
    "bear":     {"momentum": 0.5, "defensive": 1.4, "neutral": 0.9},
    "high_vol": {"momentum": 0.4, "defensive": 1.3, "neutral": 0.8},
    "low_vol":  {"momentum": 1.1, "defensive": 0.8, "neutral": 1.1},
    "crash":    {"momentum": 0.2, "defensive": 1.5, "neutral": 0.5},
}

EVENT_SCALE_FACTOR = 0.70  # reduce total allocation by 30% around events


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class ExperimentAlloc:
    """Configuration for one experiment in the portfolio."""
    name: str
    weight: float                # base allocation weight (0-1, all should sum to 1)
    style: str = "neutral"       # "momentum", "defensive", or "neutral"


@dataclass
class RebalanceEntry:
    """Log entry for one rebalance event."""
    date: str
    regime: str
    event_scale: float
    weights: Dict[str, float]    # experiment → effective weight after adjustments
    reason: str = ""


@dataclass
class DailySnapshot:
    """One day of portfolio state."""
    date: str
    equity: float
    pnl_day: float
    regime: str = ""
    contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class PortfolioMetrics:
    """Aggregate portfolio metrics."""
    total_return_pct: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_dd_pct: float = 0.0
    annual_return_pct: float = 0.0
    annual_vol_pct: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0


@dataclass
class ExperimentContribution:
    """Per-experiment contribution to portfolio."""
    name: str
    weight: float
    total_pnl: float = 0.0
    pct_of_portfolio_pnl: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    sharpe: float = 0.0


@dataclass
class SimulationResult:
    """Full simulation output."""
    portfolio_metrics: PortfolioMetrics = field(default_factory=PortfolioMetrics)
    per_experiment: List[ExperimentContribution] = field(default_factory=list)
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    rebalance_log: List[RebalanceEntry] = field(default_factory=list)
    comparison: Dict[str, PortfolioMetrics] = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────────


def compute_metrics_from_daily(
    daily_pnl: pd.Series,
    starting_capital: float = STARTING_CAPITAL,
) -> PortfolioMetrics:
    """Compute portfolio metrics from a daily P&L series."""
    if len(daily_pnl) < 2:
        return PortfolioMetrics()

    returns = daily_pnl / starting_capital
    total_pnl = float(daily_pnl.sum())
    equity = starting_capital + daily_pnl.cumsum()

    # Drawdown
    hwm = equity.cummax()
    dd = (equity - hwm) / np.where(hwm > 0, hwm, 1.0)
    max_dd = float(dd.min())

    # Sharpe
    mean_r = float(returns.mean())
    std_r = float(returns.std())
    sharpe = (mean_r / std_r * math.sqrt(252)) if std_r > 0 else 0.0

    # Sortino
    downside = returns[returns < 0]
    down_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = (mean_r / down_std * math.sqrt(252)) if down_std > 0 else 0.0

    # Annual return
    n_days = len(daily_pnl)
    years = n_days / 252
    total_return = total_pnl / starting_capital
    if total_return > -1 and years > 0:
        ann_return = ((1 + total_return) ** (1 / years) - 1) * 100
    else:
        ann_return = -100.0

    # Calmar
    calmar = (ann_return / 100 / abs(max_dd)) if max_dd < 0 else 0.0

    return PortfolioMetrics(
        total_return_pct=round(total_return * 100, 2),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        calmar=round(calmar, 3),
        max_dd_pct=round(max_dd * 100, 1),
        annual_return_pct=round(ann_return, 2),
        annual_vol_pct=round(std_r * math.sqrt(252) * 100, 2),
        n_trades=0,
        win_rate=0.0,
    )


def _build_daily_pnl(trades_df: pd.DataFrame) -> pd.Series:
    """Build daily P&L series from trade data."""
    df = trades_df.copy()
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily = df.groupby("exit_date")["pnl"].sum().sort_index()
    if len(daily) > 1:
        idx = pd.bdate_range(daily.index.min(), daily.index.max())
        daily = daily.reindex(idx, fill_value=0.0)
    return daily


def _get_regime_on_date(
    trades_df: pd.DataFrame, date: pd.Timestamp,
) -> str:
    """Get the regime on or nearest before a date."""
    df = trades_df[trades_df["entry_date"] <= date].sort_values("entry_date")
    if len(df) > 0 and "regime" in df.columns:
        return str(df.iloc[-1]["regime"])
    return "bull"


def _is_event_period(date: pd.Timestamp) -> bool:
    """Heuristic: FOMC/CPI/NFP cluster around specific days of month."""
    day = date.day
    return day <= 3 or 12 <= day <= 15  # NFP first Friday, CPI mid-month


# ── Simulator ────────────────────────────────────────────────────────────


class PortfolioSimulator:
    """Multi-experiment portfolio simulation.

    Args:
        experiments: {name: trades_DataFrame}
        allocations: {name: base_weight} (should sum to ~1.0)
        styles: {name: "momentum"|"defensive"|"neutral"}
        rebalance_freq_weeks: how often to rebalance (default 1 = weekly)
        regime_adaptive: enable regime-based weight shifts
        event_scaling: reduce allocation around macro events
        correlation_threshold: reduce correlated experiments above this
    """

    def __init__(
        self,
        experiments: Dict[str, pd.DataFrame],
        allocations: Optional[Dict[str, float]] = None,
        styles: Optional[Dict[str, str]] = None,
        rebalance_freq_weeks: int = 1,
        regime_adaptive: bool = True,
        event_scaling: bool = True,
        correlation_threshold: float = 0.70,
    ) -> None:
        self.experiments = experiments
        names = sorted(experiments.keys())

        # Default equal-weight if not specified
        if allocations is None:
            w = 1.0 / max(len(names), 1)
            allocations = {n: w for n in names}
        self.base_allocations = allocations

        self.styles = styles or {n: "neutral" for n in names}
        self.rebalance_freq = rebalance_freq_weeks
        self.regime_adaptive = regime_adaptive
        self.event_scaling = event_scaling
        self.corr_threshold = correlation_threshold

        self._result: Optional[SimulationResult] = None
        self._run_complete = False

        # Parse dates
        for name, df in self.experiments.items():
            for col in ["entry_date", "exit_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])

    def run(self) -> SimulationResult:
        """Execute the portfolio simulation."""
        # Build per-experiment daily P&L
        exp_daily: Dict[str, pd.Series] = {}
        for name, df in self.experiments.items():
            if "pnl" in df.columns and "exit_date" in df.columns:
                exp_daily[name] = _build_daily_pnl(df)

        if not exp_daily:
            self._result = SimulationResult()
            self._run_complete = True
            return self._result

        # Align all series on common date range
        all_dates = sorted(set().union(*(s.index for s in exp_daily.values())))
        if not all_dates:
            self._result = SimulationResult()
            self._run_complete = True
            return self._result

        full_idx = pd.bdate_range(min(all_dates), max(all_dates))
        aligned: Dict[str, pd.Series] = {}
        for name, s in exp_daily.items():
            aligned[name] = s.reindex(full_idx, fill_value=0.0)

        # Compute correlation for adjustments
        corr_matrix = pd.DataFrame(aligned).corr()

        # Run day-by-day simulation
        current_weights = dict(self.base_allocations)
        snapshots: List[DailySnapshot] = []
        rebalance_log: List[RebalanceEntry] = []
        equity = STARTING_CAPITAL
        last_rebalance = full_idx[0]
        contributions: Dict[str, float] = defaultdict(float)

        # Merge all trades for regime lookup
        all_trades = pd.concat(self.experiments.values(), ignore_index=True)
        if "entry_date" in all_trades.columns:
            all_trades["entry_date"] = pd.to_datetime(all_trades["entry_date"])

        for date in full_idx:
            # Check if rebalance is due
            days_since = (date - last_rebalance).days
            if days_since >= self.rebalance_freq * 7:
                regime = _get_regime_on_date(all_trades, date)
                event_scale = EVENT_SCALE_FACTOR if (self.event_scaling and _is_event_period(date)) else 1.0

                new_weights = self._compute_weights(
                    regime, event_scale, corr_matrix,
                )
                current_weights = new_weights
                last_rebalance = date
                rebalance_log.append(RebalanceEntry(
                    date=str(date.date()),
                    regime=regime,
                    event_scale=round(event_scale, 2),
                    weights={k: round(v, 4) for k, v in new_weights.items()},
                    reason=f"Weekly rebalance (regime={regime})",
                ))

            # Compute daily P&L
            day_pnl = 0.0
            day_contribs: Dict[str, float] = {}
            for name in aligned:
                w = current_weights.get(name, 0.0)
                raw = float(aligned[name].get(date, 0.0))
                weighted = raw * w
                day_pnl += weighted
                day_contribs[name] = round(weighted, 2)
                contributions[name] += weighted

            equity += day_pnl
            regime = _get_regime_on_date(all_trades, date)
            snapshots.append(DailySnapshot(
                date=str(date.date()),
                equity=round(equity, 2),
                pnl_day=round(day_pnl, 2),
                regime=regime,
                contributions=day_contribs,
            ))

        # Portfolio metrics
        daily_pnl_series = pd.Series(
            [s.pnl_day for s in snapshots],
            index=pd.DatetimeIndex([s.date for s in snapshots]),
        )
        port_metrics = compute_metrics_from_daily(daily_pnl_series)

        # Trade counts and win rates from weighted selection
        total_trades = sum(len(df) for df in self.experiments.values())
        total_wins = sum(
            int(df["win"].sum()) for df in self.experiments.values()
            if "win" in df.columns
        )
        port_metrics.n_trades = total_trades
        port_metrics.win_rate = round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0.0

        # Per-experiment contributions
        per_exp = []
        total_port_pnl = sum(contributions.values())
        for name in sorted(self.experiments.keys()):
            df = self.experiments[name]
            exp_pnl = contributions.get(name, 0.0)
            exp_trades = len(df)
            exp_wins = int(df["win"].sum()) if "win" in df.columns else 0
            exp_daily = aligned.get(name, pd.Series(dtype=float))
            exp_metrics = compute_metrics_from_daily(exp_daily * self.base_allocations.get(name, 1.0))
            per_exp.append(ExperimentContribution(
                name=name,
                weight=round(self.base_allocations.get(name, 0.0), 4),
                total_pnl=round(exp_pnl, 2),
                pct_of_portfolio_pnl=round(exp_pnl / total_port_pnl * 100, 1) if total_port_pnl != 0 else 0.0,
                n_trades=exp_trades,
                win_rate=round(exp_wins / exp_trades * 100, 1) if exp_trades > 0 else 0.0,
                sharpe=exp_metrics.sharpe,
            ))

        # Comparison: portfolio vs each experiment standalone
        comparison: Dict[str, PortfolioMetrics] = {"Portfolio": port_metrics}
        for name in sorted(self.experiments.keys()):
            standalone = aligned.get(name, pd.Series(dtype=float))
            comparison[name] = compute_metrics_from_daily(standalone)

        self._result = SimulationResult(
            portfolio_metrics=port_metrics,
            per_experiment=per_exp,
            daily_snapshots=snapshots,
            rebalance_log=rebalance_log,
            comparison=comparison,
        )
        self._run_complete = True
        return self._result

    def result(self) -> SimulationResult:
        if not self._run_complete:
            return SimulationResult()
        return self._result

    def generate_report(self, path: Optional[str] = None) -> str:
        if not self._run_complete:
            return "<html><body><p>Simulation not run.</p></body></html>"
        html = self._render_html()
        if path:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html)
        return html

    # ── Weight computation ────────────────────────────────────────────

    def _compute_weights(
        self,
        regime: str,
        event_scale: float,
        corr_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """Compute effective weights given regime, events, and correlation."""
        weights = dict(self.base_allocations)

        # Regime-adaptive adjustment
        if self.regime_adaptive and regime in DEFAULT_REGIME_MULTIPLIERS:
            mults = DEFAULT_REGIME_MULTIPLIERS[regime]
            for name in weights:
                style = self.styles.get(name, "neutral")
                mult = mults.get(style, 1.0)
                weights[name] *= mult

        # Event scaling
        if event_scale < 1.0:
            for name in weights:
                weights[name] *= event_scale

        # Correlation penalty: if avg pairwise corr > threshold, reduce total
        if len(corr_matrix) >= 2:
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
            )
            avg_corr = float(upper.stack().mean()) if upper.stack().size > 0 else 0.0
            if avg_corr > self.corr_threshold:
                penalty = max(0.5, 1.0 - (avg_corr - self.corr_threshold))
                for name in weights:
                    weights[name] *= penalty

        # Normalise so weights sum to <= 1.0
        total = sum(weights.values())
        if total > 1.0:
            for name in weights:
                weights[name] /= total

        return weights

    # ── HTML ──────────────────────────────────────────────────────────

    def _render_html(self) -> str:
        r = self._result
        pm = r.portfolio_metrics
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        cards = (
            f'<div class="cards">'
            f'<div class="card"><div class="ct">Return</div><div class="cv">{pm.total_return_pct:+.1f}%</div></div>'
            f'<div class="card"><div class="ct">Sharpe</div><div class="cv">{pm.sharpe:.3f}</div></div>'
            f'<div class="card"><div class="ct">Sortino</div><div class="cv">{pm.sortino:.3f}</div></div>'
            f'<div class="card"><div class="ct">Calmar</div><div class="cv">{pm.calmar:.3f}</div></div>'
            f'<div class="card"><div class="ct">Max DD</div><div class="cv">{pm.max_dd_pct:.1f}%</div></div>'
            f'<div class="card"><div class="ct">Trades</div><div class="cv">{pm.n_trades}</div></div>'
            f'</div>'
        )

        # Per-experiment contribution
        contrib_rows = ""
        for e in r.per_experiment:
            color = "#16a34a" if e.total_pnl > 0 else "#dc2626"
            contrib_rows += (
                f'<tr><td style="font-weight:600">{_esc(e.name)}</td>'
                f'<td>{e.weight:.0%}</td>'
                f'<td style="color:{color}">${e.total_pnl:+,.0f}</td>'
                f'<td>{e.pct_of_portfolio_pnl:+.0f}%</td>'
                f'<td>{e.n_trades}</td><td>{e.win_rate:.0f}%</td>'
                f'<td>{e.sharpe:.3f}</td></tr>'
            )
        contrib_table = (
            f'<table><thead><tr><th>Experiment</th><th>Weight</th><th>PnL</th>'
            f'<th>% of Portfolio</th><th>Trades</th><th>WR</th><th>Sharpe</th></tr></thead>'
            f'<tbody>{contrib_rows}</tbody></table>'
        )

        # Comparison table
        comp_rows = ""
        for name, m in r.comparison.items():
            style = 'font-weight:700' if name == "Portfolio" else ''
            comp_rows += (
                f'<tr style="{style}"><td>{_esc(name)}</td>'
                f'<td>{m.total_return_pct:+.1f}%</td>'
                f'<td>{m.sharpe:.3f}</td><td>{m.sortino:.3f}</td>'
                f'<td>{m.max_dd_pct:.1f}%</td><td>{m.annual_vol_pct:.1f}%</td></tr>'
            )
        comp_table = (
            f'<table><thead><tr><th>Strategy</th><th>Return</th><th>Sharpe</th>'
            f'<th>Sortino</th><th>Max DD</th><th>Vol</th></tr></thead>'
            f'<tbody>{comp_rows}</tbody></table>'
        )

        # Equity curve (sampled)
        eq_rows = ""
        if r.daily_snapshots:
            step = max(1, len(r.daily_snapshots) // 25)
            for i in range(0, len(r.daily_snapshots), step):
                s = r.daily_snapshots[i]
                eq_rows += f'<tr><td>{s.date}</td><td>${s.equity:,.0f}</td><td>{s.regime}</td></tr>'
            last = r.daily_snapshots[-1]
            eq_rows += f'<tr><td>{last.date}</td><td>${last.equity:,.0f}</td><td>{last.regime}</td></tr>'

        eq_table = (
            f'<table><thead><tr><th>Date</th><th>Equity</th><th>Regime</th></tr></thead>'
            f'<tbody>{eq_rows}</tbody></table>'
        ) if eq_rows else ""

        # Rebalance log (last 15)
        reb_rows = ""
        for entry in r.rebalance_log[-15:]:
            w_str = ", ".join(f"{k}={v:.0%}" for k, v in entry.weights.items())
            reb_rows += (
                f'<tr><td>{entry.date}</td><td>{entry.regime}</td>'
                f'<td>{entry.event_scale:.0%}</td>'
                f'<td style="font-size:0.82em">{w_str}</td></tr>'
            )
        reb_table = (
            f'<table><thead><tr><th>Date</th><th>Regime</th><th>Event Scale</th><th>Weights</th></tr></thead>'
            f'<tbody>{reb_rows}</tbody></table>'
        ) if reb_rows else ""

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Simulation Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:#f8fafc;color:#1e293b;line-height:1.5;padding:24px;max-width:1300px;margin:0 auto}}
h1{{font-size:1.6em;font-weight:700;margin-bottom:4px}}
h2{{font-size:1.15em;font-weight:600;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}}
.sub{{color:#64748b;font-size:0.9em;margin-bottom:20px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;min-width:130px;flex:1}}
.ct{{font-size:0.75em;color:#64748b;text-transform:uppercase;letter-spacing:.5px}}
.cv{{font-size:1.4em;font-weight:700}}
table{{border-collapse:collapse;width:100%;font-size:0.85em;margin-bottom:16px}}
th{{background:#f1f5f9;padding:6px 8px;text-align:center;font-weight:600;border-bottom:2px solid #e2e8f0}}
td{{padding:5px 8px;border-bottom:1px solid #f1f5f9;text-align:center}}
hr{{margin:28px 0;border:none;border-top:1px solid #e2e8f0}}
</style></head><body>

<h1>Portfolio Simulation Report</h1>
<p class="sub">{len(r.per_experiment)} experiments &middot; {pm.n_trades:,} trades &middot; {now}</p>
{cards}

<h2>Per-Experiment Contribution</h2>
{contrib_table}

<h2>Portfolio vs Individual Experiments</h2>
{comp_table}

<h2>Equity Curve</h2>
{eq_table}

<h2>Rebalance Log (Last 15)</h2>
{reb_table}

<hr><p style="font-size:0.75em;color:#94a3b8">Generated by <code>compass/portfolio_simulator.py</code></p>
</body></html>"""


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
