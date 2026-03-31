"""
Regime-aware adaptive stop loss optimizer.

Backtests multiple stop strategies against historical trade data to find
optimal stop levels per regime. Analyzes premature stops (trades that
would have been profitable if held) and generates recommendations.

Stop strategies tested:
  1. **Fixed %**: Stop at N× credit received (current baseline: 3.5×)
  2. **ATR-based**: Stop scaled to realized volatility
  3. **VIX-scaled**: Wider stops in high-vol, tighter in low-vol
  4. **Time-decay**: Stop tightens as expiration approaches
  5. **Trailing**: Trail distance adapts to regime

Usage::

    from compass.adaptive_stops import AdaptiveStopOptimizer
    optimizer = AdaptiveStopOptimizer.from_csv("compass/training_data_combined.csv")
    results = optimizer.optimize()
    optimizer.generate_report("reports/adaptive_stops.html")
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "adaptive_stops.html"


# ── Stop strategy definitions ────────────────────────────────────────────


@dataclass
class StopResult:
    """Result of applying a stop strategy to a single trade."""
    triggered: bool
    stop_level: float       # as multiple of credit
    pnl_at_stop: float      # P&L if stopped
    pnl_if_held: float      # actual P&L (held to exit)
    premature: bool         # stopped but would have been profitable
    hold_days_at_stop: Optional[int] = None


@dataclass
class StrategyResult:
    """Aggregate result of a stop strategy across all trades."""
    name: str
    stop_rate: float        # fraction of trades stopped
    avg_pnl: float
    total_pnl: float
    win_rate: float
    premature_stop_rate: float  # fraction of stops that were premature
    avg_pnl_stopped: float
    avg_pnl_held: float
    n_trades: int
    sharpe: Optional[float] = None


@dataclass
class RegimeOptimal:
    """Optimal stop parameters for a single regime."""
    regime: str
    best_strategy: str
    best_multiplier: float
    best_pnl: float
    best_win_rate: float
    premature_rate: float
    n_trades: int


# ── Stop strategy functions ──────────────────────────────────────────────

# Each function takes a trade Series and returns the stop level as a
# multiplier of credit. The actual dollar stop = credit × multiplier.


def fixed_stop(trade: pd.Series, multiplier: float = 3.5) -> float:
    """Fixed multiple of credit received."""
    return multiplier


def atr_stop(trade: pd.Series, base_mult: float = 2.5, atr_scale: float = 0.1) -> float:
    """ATR-scaled stop: wider when realized vol is high."""
    rv = trade.get("realized_vol_20d", 15) or 15
    # Scale: base + atr_scale × (rv / 15 - 1), clamped to [1.5, 6.0]
    adj = base_mult + atr_scale * (rv / 15.0 - 1.0) * base_mult
    return max(1.5, min(6.0, adj))


def vix_stop(trade: pd.Series, base_mult: float = 2.5, vix_scale: float = 0.05) -> float:
    """VIX-scaled stop: wider in high-vol environments."""
    vix = trade.get("vix", 20) or 20
    adj = base_mult * (1.0 + vix_scale * (vix - 20.0))
    return max(1.5, min(6.0, adj))


def time_decay_stop(
    trade: pd.Series,
    initial_mult: float = 4.0,
    final_mult: float = 2.0,
) -> float:
    """Stop tightens as expiration approaches."""
    dte = trade.get("dte_at_entry", 30) or 30
    hold = trade.get("hold_days", 1) or 1
    fraction_elapsed = min(1.0, hold / max(dte, 1))
    return initial_mult - (initial_mult - final_mult) * fraction_elapsed


def regime_stop(trade: pd.Series, regime_multipliers: Optional[Dict[str, float]] = None) -> float:
    """Regime-specific stop multiplier."""
    defaults = {
        "bull": 3.0,
        "low_vol": 3.5,
        "neutral": 3.0,
        "bear": 2.0,
        "high_vol": 2.0,
        "crash": 1.5,
    }
    mults = regime_multipliers or defaults
    regime = str(trade.get("regime", "bull")).lower()
    return mults.get(regime, 3.0)


def trailing_stop(
    trade: pd.Series,
    trail_mult: float = 2.0,
    regime_adjustments: Optional[Dict[str, float]] = None,
) -> float:
    """Trailing stop with regime-adaptive trail distance.

    Trail from peak unrealized P&L, not from entry.
    In practice, this is equivalent to a tighter stop for trades
    that have already moved favorably.
    """
    defaults = {"bull": 1.0, "bear": 0.7, "high_vol": 0.6, "crash": 0.5, "low_vol": 1.1, "neutral": 0.9}
    adj = regime_adjustments or defaults
    regime = str(trade.get("regime", "bull")).lower()
    scale = adj.get(regime, 1.0)
    return trail_mult * scale


# ── Stop strategy registry ───────────────────────────────────────────────

STOP_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "fixed_2.0x": {"fn": fixed_stop, "params": {"multiplier": 2.0}},
    "fixed_2.5x": {"fn": fixed_stop, "params": {"multiplier": 2.5}},
    "fixed_3.0x": {"fn": fixed_stop, "params": {"multiplier": 3.0}},
    "fixed_3.5x": {"fn": fixed_stop, "params": {"multiplier": 3.5}},
    "fixed_4.0x": {"fn": fixed_stop, "params": {"multiplier": 4.0}},
    "fixed_5.0x": {"fn": fixed_stop, "params": {"multiplier": 5.0}},
    "atr_based": {"fn": atr_stop, "params": {}},
    "vix_scaled": {"fn": vix_stop, "params": {}},
    "time_decay": {"fn": time_decay_stop, "params": {}},
    "regime_specific": {"fn": regime_stop, "params": {}},
    "trailing_regime": {"fn": trailing_stop, "params": {}},
}


# ── Backtesting engine ──────────────────────────────────────────────────


def simulate_stop(
    trade: pd.Series,
    stop_fn,
    stop_params: Dict,
) -> StopResult:
    """Simulate whether a stop would have triggered for a given trade.

    Since we don't have intra-trade P&L paths, we use the actual exit
    reason and return percentage to determine if the stop would fire:
      - If actual exit was stop_loss AND the simulated stop is tighter
        than the actual stop → triggered earlier (worse P&L).
      - If actual exit was stop_loss AND simulated stop is wider →
        not triggered (trade held to next exit).
      - If actual exit was profit/expiry → check if drawdown during
        holding period would have hit the stop (estimated from return).
    """
    credit = abs(trade.get("net_credit", 1) or 1)
    contracts = trade.get("contracts", 1) or 1
    pnl_actual = trade.get("pnl", 0) or 0
    return_pct = trade.get("return_pct", 0) or 0
    exit_reason = str(trade.get("exit_reason", ""))

    stop_mult = stop_fn(trade, **stop_params)
    stop_loss_dollar = credit * stop_mult * contracts * 100

    was_stopped = "stop_loss" in exit_reason.lower()

    # Estimate: did the trade's worst unrealized P&L breach this stop?
    # For stopped trades, the actual loss ≈ actual stop level × credit
    # For non-stopped trades, we estimate max adverse excursion (MAE)
    # as a fraction of actual P&L range

    if was_stopped:
        actual_loss = abs(pnl_actual)
        # Would our stop have been tighter (hit sooner) or wider (not hit)?
        if stop_loss_dollar < actual_loss * 0.95:
            # Tighter stop → triggered earlier, loss = stop level
            triggered = True
            pnl_at_stop = -stop_loss_dollar
        elif stop_loss_dollar > actual_loss * 1.05:
            # Wider stop → not triggered, trade continues to next exit
            # Assume it would have hit profit target or expiry
            triggered = False
            pnl_at_stop = pnl_actual  # held to actual outcome
        else:
            # Similar level → triggered at roughly the same point
            triggered = True
            pnl_at_stop = pnl_actual
    else:
        # Trade was not stopped (profit target or expiry)
        # Estimate MAE from return: losing trades that weren't stopped
        # had max drawdown < stop level
        mae_estimate = abs(credit * 1.5 * contracts * 100)  # rough MAE
        if return_pct < -20:
            mae_estimate = abs(pnl_actual) * 0.8

        triggered = mae_estimate > stop_loss_dollar and pnl_actual < 0
        pnl_at_stop = -stop_loss_dollar if triggered else pnl_actual

    premature = triggered and pnl_actual > 0

    return StopResult(
        triggered=triggered,
        stop_level=round(stop_mult, 3),
        pnl_at_stop=round(pnl_at_stop, 2),
        pnl_if_held=round(pnl_actual, 2),
        premature=premature,
    )


def backtest_strategy(
    trades: pd.DataFrame,
    strategy_name: str,
    stop_fn,
    stop_params: Dict,
) -> StrategyResult:
    """Backtest a single stop strategy across all trades."""
    results = []
    for _, trade in trades.iterrows():
        results.append(simulate_stop(trade, stop_fn, stop_params))

    n = len(results)
    if n == 0:
        return StrategyResult(strategy_name, 0, 0, 0, 0, 0, 0, 0, 0)

    stopped = [r for r in results if r.triggered]
    held = [r for r in results if not r.triggered]
    premature = [r for r in stopped if r.premature]

    pnls = [r.pnl_at_stop for r in results]
    total_pnl = sum(pnls)
    avg_pnl = total_pnl / n
    wins = sum(1 for p in pnls if p > 0)

    sharpe = None
    if len(pnls) > 1:
        std = np.std(pnls, ddof=1)
        if std > 0:
            sharpe = round(avg_pnl / std * math.sqrt(52), 3)  # annualize ~weekly

    return StrategyResult(
        name=strategy_name,
        stop_rate=round(len(stopped) / n, 4),
        avg_pnl=round(avg_pnl, 2),
        total_pnl=round(total_pnl, 2),
        win_rate=round(wins / n, 4),
        premature_stop_rate=round(len(premature) / max(len(stopped), 1), 4),
        avg_pnl_stopped=round(np.mean([r.pnl_at_stop for r in stopped]), 2) if stopped else 0,
        avg_pnl_held=round(np.mean([r.pnl_at_stop for r in held]), 2) if held else 0,
        n_trades=n,
        sharpe=sharpe,
    )


# ── Optimizer ────────────────────────────────────────────────────────────


class AdaptiveStopOptimizer:
    """Regime-aware stop loss optimizer.

    Args:
        trades: DataFrame of closed trades with P&L, regime, and market context.
    """

    def __init__(self, trades: pd.DataFrame):
        self.trades = trades.copy()
        self.strategy_results: Dict[str, StrategyResult] = {}
        self.regime_results: Dict[str, Dict[str, StrategyResult]] = {}
        self.regime_optimals: List[RegimeOptimal] = []

    @classmethod
    def from_csv(cls, csv_path: str) -> "AdaptiveStopOptimizer":
        return cls(pd.read_csv(csv_path))

    def optimize(self) -> Dict[str, Any]:
        """Run full optimization: all strategies × all regimes."""
        # 1. Global strategy comparison
        for name, cfg in STOP_STRATEGIES.items():
            self.strategy_results[name] = backtest_strategy(
                self.trades, name, cfg["fn"], cfg["params"],
            )

        # 2. Per-regime optimization
        for regime in self.trades["regime"].dropna().unique():
            regime_trades = self.trades[self.trades["regime"] == regime]
            if len(regime_trades) < 3:
                continue
            self.regime_results[regime] = {}
            for name, cfg in STOP_STRATEGIES.items():
                self.regime_results[regime][name] = backtest_strategy(
                    regime_trades, name, cfg["fn"], cfg["params"],
                )

        # 3. Find optimal per regime
        self.regime_optimals = self._find_regime_optimals()

        # 4. Premature stop analysis
        premature = self._premature_stop_analysis()

        return {
            "global": self.strategy_results,
            "by_regime": self.regime_results,
            "regime_optimals": self.regime_optimals,
            "premature_analysis": premature,
        }

    def _find_regime_optimals(self) -> List[RegimeOptimal]:
        """Find the best stop strategy per regime (maximize total P&L)."""
        optimals = []
        for regime, strategies in self.regime_results.items():
            if not strategies:
                continue
            best = max(strategies.values(), key=lambda s: s.total_pnl)
            optimals.append(RegimeOptimal(
                regime=regime,
                best_strategy=best.name,
                best_multiplier=0,  # filled below
                best_pnl=best.total_pnl,
                best_win_rate=best.win_rate,
                premature_rate=best.premature_stop_rate,
                n_trades=best.n_trades,
            ))
            # Extract multiplier from strategy name if fixed
            if "fixed_" in best.name:
                try:
                    optimals[-1].best_multiplier = float(best.name.split("_")[1].replace("x", ""))
                except (ValueError, IndexError):
                    pass
        return optimals

    def _premature_stop_analysis(self) -> Dict[str, Any]:
        """Analyze trades stopped prematurely (would have been profitable)."""
        stopped = self.trades[self.trades["exit_reason"] == "close_stop_loss"]
        if len(stopped) == 0:
            return {"n_stopped": 0, "n_premature": 0, "premature_rate": 0}

        # A stop is "premature" if the actual loss was small relative to potential profit
        # Proxy: trades with return_pct > -30% that were stopped (mild stops)
        mild_stops = stopped[stopped["return_pct"] > -30]
        severe_stops = stopped[stopped["return_pct"] <= -30]

        by_regime = {}
        for regime in stopped["regime"].dropna().unique():
            r_stops = stopped[stopped["regime"] == regime]
            by_regime[regime] = {
                "n_stopped": len(r_stops),
                "avg_loss": round(float(r_stops["pnl"].mean()), 2),
                "avg_return_pct": round(float(r_stops["return_pct"].mean()), 2),
            }

        return {
            "n_stopped": len(stopped),
            "n_mild_stops": len(mild_stops),
            "n_severe_stops": len(severe_stops),
            "mild_stop_rate": round(len(mild_stops) / len(stopped), 4) if len(stopped) > 0 else 0,
            "avg_stop_loss": round(float(stopped["pnl"].mean()), 2),
            "avg_stop_return_pct": round(float(stopped["return_pct"].mean()), 2),
            "by_regime": by_regime,
        }

    # ── HTML Report ──────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        """Generate HTML report with stop strategy analysis."""
        if not self.strategy_results:
            self.optimize()

        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Report written to %s", out)
        return str(out.resolve())

    def _render_charts(self) -> Dict[str, str]:
        import matplotlib
        matplotlib.use("Agg")
        charts: Dict[str, str] = {}
        charts["comparison"] = self._chart_strategy_comparison()
        charts["regime"] = self._chart_regime_comparison()
        return charts

    def _fig_to_b64(self, fig) -> str:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _chart_strategy_comparison(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.strategy_results:
            return ""

        sorted_strats = sorted(self.strategy_results.values(), key=lambda s: -s.total_pnl)
        names = [s.name for s in sorted_strats]
        pnls = [s.total_pnl for s in sorted_strats]
        colors = ["#16a34a" if p >= 0 else "#dc2626" for p in pnls]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(names))))
        y = range(len(names))
        ax.barh(y, pnls, color=colors, alpha=0.85, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Total P&L ($)")
        ax.set_title("Stop Strategy Comparison (Total P&L)", fontsize=12)
        ax.axvline(0, color="black", lw=0.5)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_regime_comparison(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.regime_optimals:
            return ""

        regimes = [r.regime for r in self.regime_optimals]
        pnls = [r.best_pnl for r in self.regime_optimals]
        labels = [f"{r.regime}\n({r.best_strategy})" for r in self.regime_optimals]
        colors = ["#16a34a" if p >= 0 else "#dc2626" for p in pnls]

        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(regimes)), 4))
        ax.bar(range(len(regimes)), pnls, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Best Total P&L ($)")
        ax.set_title("Best Stop Strategy Per Regime", fontsize=12)
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        for i, r in enumerate(self.regime_optimals):
            ax.text(i, pnls[i] + abs(max(pnls, default=1)) * 0.02,
                    f"WR: {r.best_win_rate:.0%}", ha="center", fontsize=8)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Global comparison table
        sorted_strats = sorted(self.strategy_results.values(), key=lambda s: -s.total_pnl)
        global_rows = ""
        best_name = sorted_strats[0].name if sorted_strats else ""
        for s in sorted_strats:
            cls = "best-row" if s.name == best_name else ""
            pnl_cls = "good" if s.total_pnl >= 0 else "bad"
            global_rows += (
                f'<tr class="{cls}"><td>{s.name}</td>'
                f'<td>{s.stop_rate:.1%}</td>'
                f'<td class="{pnl_cls}">${s.total_pnl:,.0f}</td>'
                f'<td>${s.avg_pnl:,.0f}</td>'
                f'<td>{s.win_rate:.1%}</td>'
                f'<td>{s.premature_stop_rate:.1%}</td>'
                f'<td>${s.avg_pnl_stopped:,.0f}</td>'
                f'<td>${s.avg_pnl_held:,.0f}</td>'
                f'<td>{s.sharpe or "—"}</td></tr>\n'
            )

        # Regime optimals table
        regime_rows = ""
        for r in sorted(self.regime_optimals, key=lambda x: -x.best_pnl):
            regime_rows += (
                f'<tr><td>{r.regime}</td><td>{r.best_strategy}</td>'
                f'<td>{r.n_trades}</td><td>{r.best_win_rate:.1%}</td>'
                f'<td class="{"good" if r.best_pnl >= 0 else "bad"}">${r.best_pnl:,.0f}</td>'
                f'<td>{r.premature_rate:.1%}</td></tr>\n'
            )

        # Premature stop analysis
        premature = self._premature_stop_analysis()
        premature_rows = ""
        for regime, data in premature.get("by_regime", {}).items():
            premature_rows += (
                f'<tr><td>{regime}</td><td>{data["n_stopped"]}</td>'
                f'<td>${data["avg_loss"]:,.0f}</td>'
                f'<td>{data["avg_return_pct"]:.1f}%</td></tr>\n'
            )

        def _img(key):
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ''

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Adaptive Stop Loss Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; text-align: right; }}
  td:first-child {{ text-align: left; }}
  .best-row {{ background: #f0fdf4; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Adaptive Stop Loss Analysis</h1>
<div class="meta">{len(self.trades)} trades &middot; {len(STOP_STRATEGIES)} strategies tested &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{len(self.trades)}</div><div class="label">Total Trades</div></div>
  <div class="kpi"><div class="value">{premature.get('n_stopped', 0)}</div><div class="label">Stopped Trades</div></div>
  <div class="kpi"><div class="value">{premature.get('mild_stop_rate', 0):.0%}</div><div class="label">Mild Stop Rate</div></div>
  <div class="kpi"><div class="value">${premature.get('avg_stop_loss', 0):,.0f}</div><div class="label">Avg Stop Loss</div></div>
  <div class="kpi"><div class="value">{best_name}</div><div class="label">Best Strategy</div></div>
</div>

<h2>1. Strategy Comparison</h2>
{_img("comparison")}
<table>
<thead><tr><th>Strategy</th><th>Stop Rate</th><th>Total P&L</th><th>Avg P&L</th>
<th>Win Rate</th><th>Premature %</th><th>Avg Stopped</th><th>Avg Held</th><th>Sharpe</th></tr></thead>
<tbody>{global_rows}</tbody>
</table>

<h2>2. Optimal Stop Per Regime</h2>
{_img("regime")}
<table>
<thead><tr><th>Regime</th><th>Best Strategy</th><th>Trades</th><th>Win Rate</th><th>Total P&L</th><th>Premature %</th></tr></thead>
<tbody>{regime_rows}</tbody>
</table>

<h2>3. Premature Stop Analysis</h2>
<p>Trades that were stopped out but would have been profitable if held longer.</p>
<table>
<thead><tr><th>Regime</th><th>Stopped</th><th>Avg Loss</th><th>Avg Return %</th></tr></thead>
<tbody>{premature_rows}</tbody>
</table>

<footer>Generated by <code>compass/adaptive_stops.py</code></footer>
</body></html>"""
        return html
