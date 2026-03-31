"""
compass/regime_backtest.py — Walk-forward regime-conditioned backtesting.

Splits historical trades by market regime, measures per-regime performance,
analyses post-transition behaviour, selects the best experiment per regime,
and simulates a regime-switching portfolio.

Usage::

    from compass.regime_backtest import RegimeBacktester
    bt = RegimeBacktester()
    bt.fit({"EXP-400": df400, "EXP-401": df401})
    bt.generate_report("reports/regime_backtest.html")
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

REGIMES = ["bull", "bear", "high_vol", "low_vol", "crash"]
STARTING_CAPITAL = 100_000.0


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class RegimePerformance:
    """Performance metrics for one experiment in one regime."""
    experiment: str
    regime: str
    n_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    avg_pnl: float = 0.0
    avg_hold_days: float = 0.0
    max_dd_pct: float = 0.0


@dataclass
class TransitionPerformance:
    """How trades perform in the first N days after a regime change."""
    from_regime: str
    to_regime: str
    experiment: str
    n_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0


@dataclass
class RegimeSelection:
    """Best experiment for each regime."""
    regime: str
    best_experiment: str
    sharpe: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0


@dataclass
class SwitchingResult:
    """Result of the regime-switching portfolio simulation."""
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    sharpe: float = 0.0
    n_trades: int = 0
    win_rate: float = 0.0
    max_dd_pct: float = 0.0
    equity_curve: List[Tuple[str, float]] = field(default_factory=list)


@dataclass
class BacktestSummary:
    """Full regime backtest output."""
    regime_perf: List[RegimePerformance] = field(default_factory=list)
    transitions: List[TransitionPerformance] = field(default_factory=list)
    selections: List[RegimeSelection] = field(default_factory=list)
    switching: Optional[SwitchingResult] = None
    experiments: List[str] = field(default_factory=list)


# ── Metrics helpers ──────────────────────────────────────────────────────


def _compute_sharpe(pnls: pd.Series, annual_factor: float = 252.0) -> float:
    if len(pnls) < 2:
        return 0.0
    daily = pnls.groupby(pnls.index).sum() if hasattr(pnls, 'index') else pnls
    returns = daily / STARTING_CAPITAL
    std = float(returns.std())
    if std == 0:
        return 0.0
    return float(returns.mean() / std * math.sqrt(annual_factor))


def _compute_max_dd(pnls: np.ndarray) -> float:
    equity = STARTING_CAPITAL + np.cumsum(pnls)
    hwm = np.maximum.accumulate(equity)
    dd = (equity - hwm) / np.where(hwm > 0, hwm, 1.0)
    return float(np.min(dd)) * 100 if len(dd) > 0 else 0.0


def _regime_metrics(
    df: pd.DataFrame, experiment: str, regime: str,
) -> RegimePerformance:
    """Compute metrics for one experiment in one regime."""
    n = len(df)
    if n == 0:
        return RegimePerformance(experiment=experiment, regime=regime)

    total_pnl = float(df["pnl"].sum())
    wins = (df["pnl"] > 0).sum()

    # Daily P&L series for Sharpe
    if "exit_date" in df.columns:
        daily = df.set_index("exit_date")["pnl"]
    else:
        daily = df["pnl"]
    sharpe = _compute_sharpe(daily)
    max_dd = _compute_max_dd(df["pnl"].values)

    hold_days = float(df["hold_days"].mean()) if "hold_days" in df.columns else 0.0

    return RegimePerformance(
        experiment=experiment,
        regime=regime,
        n_trades=n,
        total_pnl=round(total_pnl, 2),
        win_rate=round(wins / n * 100, 1) if n > 0 else 0.0,
        sharpe=round(sharpe, 3),
        avg_pnl=round(total_pnl / n, 2) if n > 0 else 0.0,
        avg_hold_days=round(hold_days, 1),
        max_dd_pct=round(max_dd, 1),
    )


# ── Transition analysis ──────────────────────────────────────────────────


def _detect_transitions(
    df: pd.DataFrame, lookback_days: int = 10,
) -> List[Tuple[str, str, pd.Timestamp]]:
    """Detect regime transitions from trade-level data.

    Returns list of (from_regime, to_regime, transition_date).
    """
    if "regime" not in df.columns or "entry_date" not in df.columns:
        return []

    sorted_df = df.sort_values("entry_date")
    transitions = []
    prev_regime = None
    for _, row in sorted_df.iterrows():
        r = str(row["regime"])
        if prev_regime is not None and r != prev_regime:
            transitions.append((prev_regime, r, row["entry_date"]))
        prev_regime = r
    return transitions


def _transition_performance(
    df: pd.DataFrame,
    transitions: List[Tuple[str, str, pd.Timestamp]],
    experiment: str,
    window_days: int = 10,
) -> List[TransitionPerformance]:
    """Measure trade performance in the first N days after each transition."""
    results: Dict[Tuple[str, str], List[float]] = {}

    for from_r, to_r, t_date in transitions:
        t_date_ts = pd.Timestamp(t_date)
        window_end = t_date_ts + pd.Timedelta(days=window_days * 1.5)
        mask = (
            (df["entry_date"] >= t_date_ts) &
            (df["entry_date"] <= window_end) &
            (df["regime"].astype(str) == to_r)
        )
        window_trades = df[mask]
        for pnl in window_trades["pnl"]:
            results.setdefault((from_r, to_r), []).append(float(pnl))

    out = []
    for (from_r, to_r), pnls in results.items():
        if len(pnls) < 2:
            continue
        wins = sum(1 for p in pnls if p > 0)
        out.append(TransitionPerformance(
            from_regime=from_r,
            to_regime=to_r,
            experiment=experiment,
            n_trades=len(pnls),
            win_rate=round(wins / len(pnls) * 100, 1),
            avg_pnl=round(sum(pnls) / len(pnls), 2),
        ))
    return out


# ── Regime-switching simulation ──────────────────────────────────────────


def _simulate_switching(
    all_trades: Dict[str, pd.DataFrame],
    selections: List[RegimeSelection],
) -> SwitchingResult:
    """Simulate a portfolio that switches to the best experiment per regime."""
    regime_map = {s.regime: s.best_experiment for s in selections}

    # Collect trades: for each trade, use it if its experiment is the
    # selected one for that regime
    selected_trades = []
    for exp_name, df in all_trades.items():
        for _, row in df.iterrows():
            r = str(row.get("regime", "bull"))
            if regime_map.get(r) == exp_name:
                selected_trades.append(row)

    if not selected_trades:
        return SwitchingResult()

    combined = pd.DataFrame(selected_trades)
    if "exit_date" not in combined.columns:
        return SwitchingResult()

    combined = combined.sort_values("exit_date")
    pnls = combined["pnl"].values.astype(float)
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = float(np.sum(pnls))

    # Equity curve
    equity = STARTING_CAPITAL + np.cumsum(pnls)
    curve = list(zip(
        combined["exit_date"].astype(str).tolist(),
        [round(float(e), 2) for e in equity],
    ))

    # Sharpe from daily aggregation
    daily = combined.groupby("exit_date")["pnl"].sum()
    if "exit_date" in combined.columns:
        idx = pd.bdate_range(daily.index.min(), daily.index.max())
        daily = daily.reindex(idx, fill_value=0.0)
    returns = daily / STARTING_CAPITAL
    sharpe = float(returns.mean() / returns.std() * math.sqrt(252)) if returns.std() > 0 else 0.0

    max_dd = _compute_max_dd(pnls)

    return SwitchingResult(
        total_pnl=round(total_pnl, 2),
        total_return_pct=round(total_pnl / STARTING_CAPITAL * 100, 1),
        sharpe=round(sharpe, 3),
        n_trades=n,
        win_rate=round(wins / n * 100, 1) if n > 0 else 0.0,
        max_dd_pct=round(max_dd, 1),
        equity_curve=curve,
    )


# ── Backtester ───────────────────────────────────────────────────────────


class RegimeBacktester:
    """Walk-forward regime-conditioned backtester."""

    def __init__(self, transition_window_days: int = 10) -> None:
        self._summary: Optional[BacktestSummary] = None
        self._fitted = False
        self._transition_window = transition_window_days

    def fit(self, all_trades: Dict[str, pd.DataFrame]) -> "RegimeBacktester":
        """Run the regime backtest across multiple experiments.

        Args:
            all_trades: {experiment_name: trades_DataFrame}.
        """
        experiments = sorted(all_trades.keys())

        # Ensure date columns are parsed
        for name in all_trades:
            df = all_trades[name]
            for col in ["entry_date", "exit_date"]:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            if "regime" not in df.columns:
                df["regime"] = "bull"

        # 1. Per-regime performance
        regime_perf: List[RegimePerformance] = []
        for name, df in all_trades.items():
            for regime in REGIMES:
                sub = df[df["regime"].astype(str) == regime]
                if len(sub) > 0:
                    regime_perf.append(_regime_metrics(sub, name, regime))

        # 2. Transition analysis
        all_transitions: List[TransitionPerformance] = []
        for name, df in all_trades.items():
            trans = _detect_transitions(df, self._transition_window)
            all_transitions.extend(
                _transition_performance(df, trans, name, self._transition_window)
            )

        # 3. Best experiment per regime (by Sharpe, fallback win rate)
        selections: List[RegimeSelection] = []
        for regime in REGIMES:
            candidates = [p for p in regime_perf if p.regime == regime and p.n_trades >= 3]
            if not candidates:
                continue
            best = max(candidates, key=lambda p: (p.sharpe, p.win_rate))
            selections.append(RegimeSelection(
                regime=regime,
                best_experiment=best.experiment,
                sharpe=best.sharpe,
                win_rate=best.win_rate,
                n_trades=best.n_trades,
            ))

        # 4. Switching simulation
        switching = _simulate_switching(all_trades, selections)

        self._summary = BacktestSummary(
            regime_perf=regime_perf,
            transitions=all_transitions,
            selections=selections,
            switching=switching,
            experiments=experiments,
        )
        self._fitted = True
        return self

    def summary(self) -> BacktestSummary:
        if not self._fitted:
            return BacktestSummary()
        return self._summary

    def generate_report(self, path: Optional[str] = None) -> str:
        if not self._fitted:
            return "<html><body><p>No data.</p></body></html>"
        html = self._render_html()
        if path:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html)
        return html

    # ── HTML rendering ────────────────────────────────────────────────

    def _render_html(self) -> str:
        s = self._summary
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        sw = s.switching or SwitchingResult()

        # Cards
        cards = (
            f'<div class="cards">'
            f'<div class="card"><div class="ct">Experiments</div><div class="cv">{len(s.experiments)}</div></div>'
            f'<div class="card"><div class="ct">Switching Sharpe</div><div class="cv">{sw.sharpe:.3f}</div></div>'
            f'<div class="card"><div class="ct">Switching Return</div><div class="cv">{sw.total_return_pct:+.1f}%</div></div>'
            f'<div class="card"><div class="ct">Switching Win Rate</div><div class="cv">{sw.win_rate:.0f}%</div></div>'
            f'<div class="card"><div class="ct">Max DD</div><div class="cv">{sw.max_dd_pct:.1f}%</div></div>'
            f'</div>'
        )

        # Per-regime performance table
        perf_header = '<th>Regime</th>'
        for exp in s.experiments:
            perf_header += f'<th colspan="3">{_esc(exp)}</th>'

        perf_subhdr = '<th></th>'
        for _ in s.experiments:
            perf_subhdr += '<th>Trades</th><th>WR</th><th>Sharpe</th>'

        perf_rows = ""
        for regime in REGIMES:
            entries = {p.experiment: p for p in s.regime_perf if p.regime == regime}
            if not entries:
                continue
            row = f'<td style="font-weight:600">{regime}</td>'
            for exp in s.experiments:
                p = entries.get(exp)
                if p and p.n_trades > 0:
                    row += f'<td>{p.n_trades}</td><td>{p.win_rate:.0f}%</td><td>{p.sharpe:.2f}</td>'
                else:
                    row += '<td>-</td><td>-</td><td>-</td>'
            perf_rows += f'<tr>{row}</tr>'

        perf_table = (
            f'<table><thead><tr>{perf_header}</tr><tr>{perf_subhdr}</tr></thead>'
            f'<tbody>{perf_rows}</tbody></table>'
        )

        # Regime selection table
        sel_rows = ""
        for sel in s.selections:
            sel_rows += (
                f'<tr><td style="font-weight:600">{sel.regime}</td>'
                f'<td>{sel.best_experiment}</td>'
                f'<td>{sel.sharpe:.3f}</td>'
                f'<td>{sel.win_rate:.0f}%</td>'
                f'<td>{sel.n_trades}</td></tr>'
            )
        sel_table = (
            f'<table><thead><tr><th>Regime</th><th>Best Experiment</th>'
            f'<th>Sharpe</th><th>Win Rate</th><th>Trades</th></tr></thead>'
            f'<tbody>{sel_rows}</tbody></table>'
        )

        # Transition table
        trans_rows = ""
        for t in sorted(s.transitions, key=lambda x: -abs(x.avg_pnl)):
            color = "#16a34a" if t.avg_pnl > 0 else "#dc2626"
            trans_rows += (
                f'<tr><td>{t.from_regime} &rarr; {t.to_regime}</td>'
                f'<td>{t.experiment}</td>'
                f'<td>{t.n_trades}</td>'
                f'<td>{t.win_rate:.0f}%</td>'
                f'<td style="color:{color}">${t.avg_pnl:+,.0f}</td></tr>'
            )
        trans_table = (
            f'<table><thead><tr><th>Transition</th><th>Experiment</th>'
            f'<th>Trades</th><th>Win Rate</th><th>Avg PnL</th></tr></thead>'
            f'<tbody>{trans_rows}</tbody></table>'
        ) if trans_rows else "<p>Insufficient transitions.</p>"

        # Equity curve (simple text table for the switching portfolio)
        eq_rows = ""
        if sw.equity_curve:
            step = max(1, len(sw.equity_curve) // 20)
            for i in range(0, len(sw.equity_curve), step):
                d, v = sw.equity_curve[i]
                eq_rows += f'<tr><td>{d}</td><td>${v:,.0f}</td></tr>'
            d, v = sw.equity_curve[-1]
            if i != len(sw.equity_curve) - 1:
                eq_rows += f'<tr><td>{d}</td><td>${v:,.0f}</td></tr>'

        eq_table = (
            f'<table><thead><tr><th>Date</th><th>Equity</th></tr></thead>'
            f'<tbody>{eq_rows}</tbody></table>'
        ) if eq_rows else ""

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Regime Backtest Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:#f8fafc;color:#1e293b;line-height:1.5;padding:24px;max-width:1300px;margin:0 auto}}
h1{{font-size:1.6em;font-weight:700;margin-bottom:4px}}
h2{{font-size:1.15em;font-weight:600;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}}
.sub{{color:#64748b;font-size:0.9em;margin-bottom:20px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;min-width:140px;flex:1}}
.ct{{font-size:0.75em;color:#64748b;text-transform:uppercase;letter-spacing:.5px}}
.cv{{font-size:1.4em;font-weight:700}}
table{{border-collapse:collapse;width:100%;font-size:0.85em;margin-bottom:16px}}
th{{background:#f1f5f9;padding:6px 8px;text-align:center;font-weight:600;border-bottom:2px solid #e2e8f0}}
td{{padding:5px 8px;border-bottom:1px solid #f1f5f9;text-align:center}}
hr{{margin:28px 0;border:none;border-top:1px solid #e2e8f0}}
</style></head><body>

<h1>Regime Backtest Report</h1>
<p class="sub">{len(s.experiments)} experiments &middot; {sum(p.n_trades for p in s.regime_perf):,} trades &middot; {now}</p>
{cards}

<h2>Per-Regime Performance</h2>
{perf_table}

<h2>Optimal Strategy per Regime</h2>
<p style="font-size:0.82em;color:#64748b;margin-bottom:8px">
Best experiment selected by Sharpe ratio (minimum 3 trades).</p>
{sel_table}

<h2>Regime Transition Performance</h2>
<p style="font-size:0.82em;color:#64748b;margin-bottom:8px">
Trade performance in the first {self._transition_window} days after a regime change.</p>
{trans_table}

<h2>Regime-Switching Portfolio</h2>
<p style="font-size:0.82em;color:#64748b;margin-bottom:8px">
Simulated portfolio that uses the best experiment for each regime.</p>
{eq_table}

<hr><p style="font-size:0.75em;color:#94a3b8">Generated by <code>compass/regime_backtest.py</code></p>
</body></html>"""


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
