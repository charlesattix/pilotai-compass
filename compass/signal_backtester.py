"""
Rapid signal backtesting framework — vectorised evaluation of trading signals.

Designed for fast iteration: no event loop, pure numpy/pandas vectorisation.

Components:
  1. Vectorised backtest engine  (signal × returns → equity curve)
  2. Standard metrics            (Sharpe, Sortino, Calmar, max DD, win rate, PF)
  3. Long / short / long-short   (evaluate each direction independently)
  4. Regime-conditional perf     (metrics per market regime)
  5. Signal combination testing  (pairwise & n-way AND/OR/vote)
  6. Walk-forward validation     (expanding & rolling windows)
  7. HTML report                 (equity, drawdown, heatmap, comparison)

All methods work on pre-loaded data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """Standard performance metrics."""
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade: float = 0.0
    volatility: float = 0.0


@dataclass
class DirectionalResult:
    """Metrics for long, short, and combined."""
    long: BacktestMetrics
    short: BacktestMetrics
    combined: BacktestMetrics


@dataclass
class RegimePerformance:
    """Metrics for one regime."""
    regime: str
    n_days: int
    metrics: BacktestMetrics


@dataclass
class SignalCombination:
    """Result of combining two or more signals."""
    signal_names: List[str]
    method: str               # "and" | "or" | "vote"
    metrics: BacktestMetrics


@dataclass
class WalkForwardFold:
    """One fold of walk-forward validation."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    in_sample: BacktestMetrics
    out_of_sample: BacktestMetrics


@dataclass
class WalkForwardResult:
    """Full walk-forward result."""
    folds: List[WalkForwardFold]
    avg_is_sharpe: float = 0.0
    avg_oos_sharpe: float = 0.0
    oos_degradation: float = 0.0  # (IS - OOS) / IS


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class SignalBacktester:
    """Vectorised signal backtesting engine.

    Args:
        cost_per_trade: Round-trip cost in return units (e.g. 0.001 = 10bps).
    """

    def __init__(self, cost_per_trade: float = 0.001) -> None:
        self.cost_per_trade = cost_per_trade

    # ------------------------------------------------------------------
    # 1. Vectorised backtest
    # ------------------------------------------------------------------

    def backtest(
        self,
        signals: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """Apply signal to returns, return strategy daily returns.

        signals: +1 (long), -1 (short), 0 (flat). Shifted by 1 day
                 (signal at t is applied to return at t+1).
        returns: Daily asset returns.
        """
        aligned = pd.DataFrame({"sig": signals, "ret": returns}).dropna()
        if aligned.empty:
            return pd.Series(dtype=float)

        pos = aligned["sig"].shift(1).fillna(0)
        trades = pos.diff().abs().fillna(0)
        costs = trades * self.cost_per_trade
        strat_ret = pos * aligned["ret"] - costs
        return strat_ret

    def equity_curve(self, strategy_returns: pd.Series) -> pd.Series:
        """Cumulative equity from strategy returns."""
        return (1 + strategy_returns).cumprod()

    # ------------------------------------------------------------------
    # 2. Standard metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(
        strategy_returns: pd.Series,
    ) -> BacktestMetrics:
        """Compute all standard metrics from a return series."""
        r = strategy_returns.dropna()
        if r.empty or len(r) < 2:
            return BacktestMetrics()

        total = float((1 + r).prod() - 1)
        n_years = len(r) / TRADING_DAYS
        annual = (1 + total) ** (1 / max(n_years, 0.01)) - 1 if n_years > 0 else 0.0
        vol = float(r.std() * np.sqrt(TRADING_DAYS))

        mu = float(r.mean())
        std = float(r.std())
        sharpe = mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0

        downside = r[r < 0]
        down_std = float(downside.std()) if len(downside) > 1 else 1e-8
        sortino = mu / down_std * np.sqrt(TRADING_DAYS) if down_std > 1e-12 else 0.0

        eq = (1 + r).cumprod()
        hwm = eq.expanding().max()
        dd = 1 - eq / hwm
        max_dd = float(dd.max())
        calmar = annual / max_dd if max_dd > 1e-8 else 0.0

        # Trade-level: a "trade" is a period with non-zero return
        active = r[r != 0]
        n_trades = len(active)
        wins = active[active > 0]
        losses = active[active < 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 1e-8
        pf = gross_profit / gross_loss if gross_loss > 1e-12 else 0.0
        avg_trade = float(active.mean()) if n_trades > 0 else 0.0

        return BacktestMetrics(
            total_return=total, annual_return=annual, sharpe=sharpe,
            sortino=sortino, calmar=calmar, max_drawdown=max_dd,
            win_rate=win_rate, profit_factor=pf, n_trades=n_trades,
            avg_trade=avg_trade, volatility=vol,
        )

    # ------------------------------------------------------------------
    # 3. Directional evaluation
    # ------------------------------------------------------------------

    def evaluate_directional(
        self,
        signals: pd.Series,
        returns: pd.Series,
    ) -> DirectionalResult:
        """Evaluate long-only, short-only, and combined."""
        long_sig = signals.clip(lower=0)
        short_sig = signals.clip(upper=0)

        long_ret = self.backtest(long_sig, returns)
        short_ret = self.backtest(short_sig, returns)
        combined_ret = self.backtest(signals, returns)

        return DirectionalResult(
            long=self.compute_metrics(long_ret),
            short=self.compute_metrics(short_ret),
            combined=self.compute_metrics(combined_ret),
        )

    # ------------------------------------------------------------------
    # 4. Regime-conditional performance
    # ------------------------------------------------------------------

    def regime_performance(
        self,
        signals: pd.Series,
        returns: pd.Series,
        regimes: pd.Series,
    ) -> List[RegimePerformance]:
        """Compute metrics per regime."""
        strat_ret = self.backtest(signals, returns)
        aligned = pd.DataFrame({
            "ret": strat_ret, "reg": regimes,
        }).dropna()

        results: List[RegimePerformance] = []
        for regime, grp in aligned.groupby("reg"):
            m = self.compute_metrics(grp["ret"])
            results.append(RegimePerformance(
                regime=str(regime), n_days=len(grp), metrics=m,
            ))
        return results

    # ------------------------------------------------------------------
    # 5. Signal combination testing
    # ------------------------------------------------------------------

    @staticmethod
    def combine_signals(
        signal_dict: Dict[str, pd.Series],
        method: str = "and",
    ) -> pd.Series:
        """Combine multiple signals.

        Methods:
          and: all must agree on direction (intersection)
          or: any signal fires (union)
          vote: majority vote (>50% agree)
        """
        df = pd.DataFrame(signal_dict).fillna(0)
        if df.empty:
            return pd.Series(dtype=float)

        if method == "and":
            # All positive → +1; all negative → -1; else 0
            all_pos = (df > 0).all(axis=1)
            all_neg = (df < 0).all(axis=1)
            result = pd.Series(0, index=df.index, dtype=float)
            result[all_pos] = 1.0
            result[all_neg] = -1.0
            return result

        if method == "or":
            any_pos = (df > 0).any(axis=1)
            any_neg = (df < 0).any(axis=1)
            result = pd.Series(0, index=df.index, dtype=float)
            result[any_pos] = 1.0
            result[any_neg & ~any_pos] = -1.0
            return result

        if method == "vote":
            avg = df.mean(axis=1)
            return avg.apply(lambda x: 1.0 if x > 0.1 else (-1.0 if x < -0.1 else 0.0))

        return pd.Series(0, index=df.index, dtype=float)

    def test_combinations(
        self,
        signal_dict: Dict[str, pd.Series],
        returns: pd.Series,
        methods: Optional[List[str]] = None,
        max_n: int = 3,
    ) -> List[SignalCombination]:
        """Test pairwise and n-way signal combinations."""
        methods = methods or ["and", "or", "vote"]
        names = list(signal_dict.keys())
        results: List[SignalCombination] = []

        for n in range(2, min(len(names), max_n) + 1):
            for combo in combinations(names, n):
                subset = {k: signal_dict[k] for k in combo}
                for method in methods:
                    combined = self.combine_signals(subset, method)
                    strat_ret = self.backtest(combined, returns)
                    m = self.compute_metrics(strat_ret)
                    results.append(SignalCombination(
                        signal_names=list(combo), method=method, metrics=m,
                    ))

        results.sort(key=lambda x: x.metrics.sharpe, reverse=True)
        return results

    # ------------------------------------------------------------------
    # 6. Walk-forward validation
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        signals: pd.Series,
        returns: pd.Series,
        n_folds: int = 5,
        expanding: bool = False,
    ) -> WalkForwardResult:
        """Walk-forward out-of-sample testing.

        Args:
            expanding: True = expanding window; False = rolling window.
        """
        aligned = pd.DataFrame({"sig": signals, "ret": returns}).dropna()
        n = len(aligned)
        if n < n_folds * 2:
            return WalkForwardResult(folds=[])

        fold_size = n // n_folds
        folds: List[WalkForwardFold] = []

        for i in range(n_folds - 1):
            test_start = (i + 1) * fold_size
            test_end = min((i + 2) * fold_size, n)
            train_start = 0 if expanding else i * fold_size
            train_end = test_start

            train = aligned.iloc[train_start:train_end]
            test = aligned.iloc[test_start:test_end]

            is_ret = self.backtest(train["sig"], train["ret"])
            oos_ret = self.backtest(test["sig"], test["ret"])

            folds.append(WalkForwardFold(
                fold=i + 1,
                train_start=train.index[0],
                train_end=train.index[-1],
                test_start=test.index[0],
                test_end=test.index[-1],
                in_sample=self.compute_metrics(is_ret),
                out_of_sample=self.compute_metrics(oos_ret),
            ))

        avg_is = float(np.mean([f.in_sample.sharpe for f in folds])) if folds else 0.0
        avg_oos = float(np.mean([f.out_of_sample.sharpe for f in folds])) if folds else 0.0
        degradation = (avg_is - avg_oos) / abs(avg_is) if abs(avg_is) > 1e-8 else 0.0

        return WalkForwardResult(
            folds=folds, avg_is_sharpe=avg_is,
            avg_oos_sharpe=avg_oos, oos_degradation=degradation,
        )

    # ------------------------------------------------------------------
    # Monthly returns helper
    # ------------------------------------------------------------------

    @staticmethod
    def monthly_returns(strategy_returns: pd.Series) -> pd.DataFrame:
        """Pivot returns into year × month heatmap format."""
        r = strategy_returns.copy()
        r.index = pd.to_datetime(r.index)
        monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        if df.empty:
            return pd.DataFrame()
        return df.pivot(index="year", columns="month", values="return")

    # ------------------------------------------------------------------
    # 7. HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_line(
        values: List[float], title: str,
        width: int = 720, height: int = 200, color: str = "#2980b9",
    ) -> str:
        if len(values) < 2:
            return ""
        n = len(values)
        vmin, vmax = min(values), max(values)
        if vmax <= vmin:
            vmax = vmin + 0.01
        pad_l, pad_r, pad_t, pad_b = 50, 15, 28, 25
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b

        def tx(i): return pad_l + i / max(n - 1, 1) * pw
        def ty(v): return pad_t + (1 - (v - vmin) / (vmax - vmin)) * ph

        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="background:#fff;border:1px solid #ddd;'
             f'border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')
        d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                      for i, v in enumerate(values))
        p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
        p.append("</svg>")
        return "\n".join(p)

    @staticmethod
    def _svg_heatmap(
        monthly: pd.DataFrame, width: int = 650, height: int = 0,
    ) -> str:
        """Monthly returns heatmap."""
        if monthly.empty:
            return ""
        years = monthly.index.tolist()
        months = sorted(monthly.columns.tolist())
        n_rows = len(years)
        n_cols = len(months)
        cell_w = 45
        cell_h = 28
        pad_l = 50
        pad_t = 30
        if height == 0:
            height = pad_t + n_rows * cell_h + 20

        p = [f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'width="{pad_l + n_cols * cell_w + 10}" height="{height}" '
             f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{(pad_l + n_cols * cell_w) // 2}" y="16" '
                 f'text-anchor="middle" font-size="12" font-weight="bold" '
                 f'fill="#1a1a2e">Monthly Returns</text>')

        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for j, m in enumerate(months):
            x = pad_l + j * cell_w
            p.append(f'<text x="{x + cell_w // 2}" y="{pad_t - 4}" '
                     f'text-anchor="middle" font-size="9" fill="#666">'
                     f'{month_names[m - 1] if 1 <= m <= 12 else m}</text>')

        for i, yr in enumerate(years):
            y = pad_t + i * cell_h
            p.append(f'<text x="{pad_l - 5}" y="{y + cell_h * 0.7:.0f}" '
                     f'text-anchor="end" font-size="10" fill="#333">{yr}</text>')
            for j, m in enumerate(months):
                x = pad_l + j * cell_w
                val = monthly.loc[yr, m] if m in monthly.columns and not pd.isna(monthly.loc[yr].get(m)) else None
                if val is None:
                    p.append(f'<rect x="{x}" y="{y}" width="{cell_w - 1}" '
                             f'height="{cell_h - 1}" fill="#f5f5f5" rx="3"/>')
                else:
                    intensity = min(abs(val) / 0.05, 1.0)
                    if val >= 0:
                        r, g, b = int(39 + (1 - intensity) * 200), int(174), int(96 + (1 - intensity) * 150)
                    else:
                        r, g, b = int(231), int(76 + (1 - intensity) * 170), int(60 + (1 - intensity) * 180)
                    p.append(f'<rect x="{x}" y="{y}" width="{cell_w - 1}" '
                             f'height="{cell_h - 1}" fill="rgb({r},{g},{b})" rx="3"/>')
                    p.append(f'<text x="{x + cell_w // 2}" y="{y + cell_h * 0.7:.0f}" '
                             f'text-anchor="middle" font-size="8" fill="#333">'
                             f'{val:+.1%}</text>')
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        metrics: BacktestMetrics,
        equity: Optional[pd.Series] = None,
        strategy_returns: Optional[pd.Series] = None,
        directional: Optional[DirectionalResult] = None,
        regime_perf: Optional[List[RegimePerformance]] = None,
        combinations: Optional[List[SignalCombination]] = None,
        walk_forward: Optional[WalkForwardResult] = None,
        output_path: str = "reports/signal_backtest.html",
    ) -> str:
        """HTML report: equity, drawdown, heatmap, comparison."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Equity chart
        eq_svg = ""
        dd_svg = ""
        if equity is not None and len(equity) > 2:
            eq_svg = self._svg_line(equity.tolist(), "Equity Curve", color="#2980b9")
            hwm = equity.expanding().max()
            dd = (1 - equity / hwm).tolist()
            dd_svg = self._svg_line(dd, "Drawdown", color="#e74c3c")

        # Monthly heatmap
        hm_svg = ""
        if strategy_returns is not None:
            monthly = self.monthly_returns(strategy_returns)
            hm_svg = self._svg_heatmap(monthly)

        # Metrics table
        m = metrics
        metrics_html = f"""
<h2>Performance Metrics</h2>
<table class="m"><tr><th>Return</th><th>Annual</th><th>Sharpe</th><th>Sortino</th>
<th>Calmar</th><th>Max DD</th><th>Win Rate</th><th>PF</th><th>Trades</th><th>Vol</th></tr>
<tr><td>{m.total_return:.2%}</td><td>{m.annual_return:.2%}</td>
<td>{m.sharpe:.2f}</td><td>{m.sortino:.2f}</td><td>{m.calmar:.2f}</td>
<td>{m.max_drawdown:.2%}</td><td>{m.win_rate:.1%}</td>
<td>{m.profit_factor:.2f}</td><td>{m.n_trades}</td>
<td>{m.volatility:.2%}</td></tr></table>"""

        # Directional
        dir_html = ""
        if directional:
            def _row(label, mm):
                return (f"<tr><td>{label}</td><td>{mm.sharpe:.2f}</td>"
                        f"<td>{mm.annual_return:.2%}</td><td>{mm.max_drawdown:.2%}</td>"
                        f"<td>{mm.win_rate:.1%}</td><td>{mm.n_trades}</td></tr>")
            dir_html = f"""
<h2>Directional Analysis</h2>
<table><tr><th>Direction</th><th>Sharpe</th><th>Return</th><th>Max DD</th>
<th>Win Rate</th><th>Trades</th></tr>
{_row('Long', directional.long)}
{_row('Short', directional.short)}
{_row('Combined', directional.combined)}
</table>"""

        # Regime
        reg_html = ""
        if regime_perf:
            rows = [
                f"<tr><td>{rp.regime}</td><td>{rp.n_days}</td>"
                f"<td>{rp.metrics.sharpe:.2f}</td><td>{rp.metrics.annual_return:.2%}</td>"
                f"<td>{rp.metrics.max_drawdown:.2%}</td></tr>"
                for rp in regime_perf
            ]
            reg_html = f"""
<h2>Regime Performance</h2>
<table><tr><th>Regime</th><th>Days</th><th>Sharpe</th><th>Return</th><th>Max DD</th></tr>
{''.join(rows)}</table>"""

        # Combinations
        combo_html = ""
        if combinations:
            rows = [
                f"<tr><td>{' + '.join(sc.signal_names)}</td><td>{sc.method}</td>"
                f"<td>{sc.metrics.sharpe:.2f}</td><td>{sc.metrics.annual_return:.2%}</td>"
                f"<td>{sc.metrics.max_drawdown:.2%}</td></tr>"
                for sc in combinations[:15]
            ]
            combo_html = f"""
<h2>Signal Combinations (top 15)</h2>
<table><tr><th>Signals</th><th>Method</th><th>Sharpe</th><th>Return</th><th>Max DD</th></tr>
{''.join(rows)}</table>"""

        # Walk-forward
        wf_html = ""
        if walk_forward and walk_forward.folds:
            wf = walk_forward
            rows = [
                f"<tr><td>{f.fold}</td>"
                f"<td>{f.in_sample.sharpe:.2f}</td>"
                f"<td>{f.out_of_sample.sharpe:.2f}</td></tr>"
                for f in wf.folds
            ]
            wf_html = f"""
<h2>Walk-Forward Validation</h2>
<table class="m"><tr><th>Avg IS Sharpe</th><th>Avg OOS Sharpe</th><th>Degradation</th></tr>
<tr><td>{wf.avg_is_sharpe:.2f}</td><td>{wf.avg_oos_sharpe:.2f}</td>
<td>{wf.oos_degradation:.1%}</td></tr></table>
<table><tr><th>Fold</th><th>IS Sharpe</th><th>OOS Sharpe</th></tr>
{''.join(rows)}</table>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Signal Backtest</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
table.m {{ width: auto; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
</style></head><body>
<h1>Signal Backtest Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>

{eq_svg}
{dd_svg}
{metrics_html}
{hm_svg}
{dir_html}
{reg_html}
{combo_html}
{wf_html}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Signal backtest report -> %s", path)
        return str(path)
