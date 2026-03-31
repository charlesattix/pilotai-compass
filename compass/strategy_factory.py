"""
Automated strategy generation factory.

Templates:
  - mean_reversion:  z-score reversal (Bollinger-style)
  - momentum:        trend following (MA crossover / ROC)
  - stat_arb:        pair spread mean-reversion
  - volatility:      vol regime selling (short vol in calm, long in stress)
  - event_driven:    pre/post event drift exploitation

For each template the factory sweeps a parameter grid, backtests
every candidate, filters by Sharpe/drawdown/capacity, ranks survivors,
and outputs a strategy catalog with recommended top-N.

All methods work on pre-loaded data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StrategyTemplate:
    """Defines a strategy archetype."""
    name: str
    category: str             # mean_reversion | momentum | stat_arb | volatility | event_driven
    params: Dict[str, Any] = field(default_factory=dict)
    entry_signal: str = ""
    exit_signal: str = "trailing_stop"
    sizing_rule: str = "fixed_fraction"
    max_risk_pct: float = 0.02


@dataclass
class PerformanceMetrics:
    sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    n_trades: int = 0
    avg_trade_pnl: float = 0.0
    capacity_score: float = 0.0   # 0-1 (higher = scales better)


@dataclass
class StrategyCandidate:
    """A generated strategy with its performance."""
    template: StrategyTemplate
    metrics: PerformanceMetrics
    passed_filter: bool = True
    rank: int = 0
    sensitivity: Dict[str, float] = field(default_factory=dict)


@dataclass
class FilterCriteria:
    min_sharpe: float = 0.5
    max_drawdown: float = 0.30
    min_trades: int = 20
    min_capacity: float = 0.0


@dataclass
class StrategyCatalog:
    """Full output of the factory."""
    total_generated: int
    total_passed: int
    candidates: List[StrategyCandidate]
    top_10: List[StrategyCandidate]
    by_category: Dict[str, List[StrategyCandidate]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Signal generators per template category
# ---------------------------------------------------------------------------

def _signal_mean_reversion(prices: pd.Series, lookback: int = 20,
                            threshold: float = 1.5) -> pd.Series:
    """Z-score mean-reversion signal."""
    ma = prices.rolling(lookback).mean()
    std = prices.rolling(lookback).std().replace(0, 1e-8)
    z = (prices - ma) / std
    return z.apply(lambda x: -1.0 if x > threshold else (1.0 if x < -threshold else 0.0))


def _signal_momentum(prices: pd.Series, fast: int = 10,
                      slow: int = 50) -> pd.Series:
    """Moving-average crossover momentum."""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return (fast_ma - slow_ma).apply(lambda x: 1.0 if x > 0 else -1.0)


def _signal_stat_arb(prices_a: pd.Series, prices_b: pd.Series,
                      lookback: int = 30, threshold: float = 2.0) -> pd.Series:
    """Pair spread z-score."""
    spread = prices_a - prices_b
    ma = spread.rolling(lookback).mean()
    std = spread.rolling(lookback).std().replace(0, 1e-8)
    z = (spread - ma) / std
    return z.apply(lambda x: -1.0 if x > threshold else (1.0 if x < -threshold else 0.0))


def _signal_volatility(vol: pd.Series, vol_ma: pd.Series,
                        threshold: float = 1.2) -> pd.Series:
    """Sell vol when below MA, buy when above (protective)."""
    ratio = vol / vol_ma.replace(0, 1e-8)
    return ratio.apply(lambda x: -1.0 if x < 1.0 / threshold else (1.0 if x > threshold else 0.0))


def _signal_event(returns: pd.Series, lookback: int = 5,
                   drift_threshold: float = 0.01) -> pd.Series:
    """Post-event drift: buy after sharp drop, sell after sharp rally."""
    cum = returns.rolling(lookback).sum()
    return cum.apply(lambda x: 1.0 if x < -drift_threshold else (
        -1.0 if x > drift_threshold else 0.0))


SIGNAL_GENERATORS: Dict[str, Callable] = {
    "mean_reversion": _signal_mean_reversion,
    "momentum": _signal_momentum,
    "stat_arb": _signal_stat_arb,
    "volatility": _signal_volatility,
    "event_driven": _signal_event,
}

DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, List]] = {
    "mean_reversion": {"lookback": [10, 20, 30, 50], "threshold": [1.0, 1.5, 2.0, 2.5]},
    "momentum": {"fast": [5, 10, 20], "slow": [30, 50, 100]},
    "stat_arb": {"lookback": [15, 30, 50], "threshold": [1.5, 2.0, 2.5]},
    "volatility": {"threshold": [1.1, 1.2, 1.5, 2.0]},
    "event_driven": {"lookback": [3, 5, 10], "drift_threshold": [0.005, 0.01, 0.02]},
}


# ---------------------------------------------------------------------------
# Core factory
# ---------------------------------------------------------------------------

class StrategyFactory:
    """Automated strategy generation and screening.

    Args:
        filter_criteria: Minimum requirements for candidates.
        cost_per_trade: Round-trip transaction cost.
    """

    def __init__(
        self,
        filter_criteria: Optional[FilterCriteria] = None,
        cost_per_trade: float = 0.001,
    ) -> None:
        self.criteria = filter_criteria or FilterCriteria()
        self.cost_per_trade = cost_per_trade

    # ------------------------------------------------------------------
    # Grid generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_grid(param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        return [dict(zip(keys, combo)) for combo in product(*values)]

    def generate_templates(
        self,
        categories: Optional[List[str]] = None,
        param_grids: Optional[Dict[str, Dict[str, List]]] = None,
    ) -> List[StrategyTemplate]:
        """Generate all StrategyTemplate candidates across categories."""
        cats = categories or list(DEFAULT_PARAM_GRIDS.keys())
        grids = param_grids or DEFAULT_PARAM_GRIDS
        templates: List[StrategyTemplate] = []
        for cat in cats:
            grid = grids.get(cat, {})
            combos = self.generate_grid(grid) if grid else [{}]
            for i, params in enumerate(combos):
                templates.append(StrategyTemplate(
                    name=f"{cat}_{i:03d}",
                    category=cat,
                    params=params,
                    entry_signal=cat,
                ))
        return templates

    # ------------------------------------------------------------------
    # Signal dispatch
    # ------------------------------------------------------------------

    @staticmethod
    def generate_signal(
        template: StrategyTemplate,
        prices: pd.Series,
        prices_b: Optional[pd.Series] = None,
        vol: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Generate signal from a template."""
        cat = template.category
        p = template.params

        if cat == "mean_reversion":
            return _signal_mean_reversion(prices, p.get("lookback", 20), p.get("threshold", 1.5))
        if cat == "momentum":
            return _signal_momentum(prices, p.get("fast", 10), p.get("slow", 50))
        if cat == "stat_arb" and prices_b is not None:
            return _signal_stat_arb(prices, prices_b, p.get("lookback", 30), p.get("threshold", 2.0))
        if cat == "volatility" and vol is not None:
            vol_ma = vol.rolling(p.get("lookback", 20)).mean().fillna(vol.mean())
            return _signal_volatility(vol, vol_ma, p.get("threshold", 1.2))
        if cat == "event_driven":
            returns = prices.pct_change().fillna(0)
            return _signal_event(returns, p.get("lookback", 5), p.get("drift_threshold", 0.01))

        return pd.Series(0.0, index=prices.index)

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(
        self, signal: pd.Series, returns: pd.Series,
    ) -> PerformanceMetrics:
        """Vectorised backtest → performance metrics."""
        aligned = pd.DataFrame({"sig": signal, "ret": returns}).dropna()
        if len(aligned) < 10:
            return PerformanceMetrics()

        pos = aligned["sig"].shift(1).fillna(0)
        trades = pos.diff().abs().fillna(0)
        strat_ret = pos * aligned["ret"] - trades * self.cost_per_trade

        r = strat_ret
        mu = float(r.mean())
        std = float(r.std())
        sharpe = mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
        total = float((1 + r).prod() - 1)
        n_years = len(r) / TRADING_DAYS
        annual = (1 + total) ** (1 / max(n_years, 0.01)) - 1

        eq = (1 + r).cumprod()
        dd = float((1 - eq / eq.expanding().max()).max())

        active = r[r != 0]
        n_trades = len(active)
        wins = active[active > 0]
        losses = active[active < 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        pf = float(wins.sum()) / abs(float(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 0.0
        avg_trade = float(active.mean()) if n_trades > 0 else 0.0

        # Capacity score: penalise high turnover
        daily_turnover = float(trades.mean())
        capacity = max(0, 1 - daily_turnover * 50)

        return PerformanceMetrics(
            sharpe=sharpe, annual_return=annual, max_drawdown=dd,
            win_rate=win_rate, profit_factor=pf, n_trades=n_trades,
            avg_trade_pnl=avg_trade, capacity_score=capacity,
        )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_candidate(self, metrics: PerformanceMetrics) -> bool:
        c = self.criteria
        return (
            metrics.sharpe >= c.min_sharpe
            and metrics.max_drawdown <= c.max_drawdown
            and metrics.n_trades >= c.min_trades
            and metrics.capacity_score >= c.min_capacity
        )

    # ------------------------------------------------------------------
    # Parameter sensitivity
    # ------------------------------------------------------------------

    def compute_sensitivity(
        self,
        template: StrategyTemplate,
        prices: pd.Series,
        returns: pd.Series,
        prices_b: Optional[pd.Series] = None,
        vol: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Measure Sharpe sensitivity to each parameter (±20% perturbation)."""
        base_sig = self.generate_signal(template, prices, prices_b, vol)
        base_sharpe = self.backtest(base_sig, returns).sharpe

        sens: Dict[str, float] = {}
        for key, value in template.params.items():
            if not isinstance(value, (int, float)) or value == 0:
                continue
            for mult in [0.8, 1.2]:
                perturbed = dict(template.params)
                perturbed[key] = type(value)(value * mult)
                t_copy = StrategyTemplate(
                    name=template.name, category=template.category,
                    params=perturbed, entry_signal=template.entry_signal)
                sig = self.generate_signal(t_copy, prices, prices_b, vol)
                s = self.backtest(sig, returns).sharpe
                delta = abs(s - base_sharpe)
                sens[key] = max(sens.get(key, 0), delta)
        return sens

    # ------------------------------------------------------------------
    # Full factory run
    # ------------------------------------------------------------------

    def run(
        self,
        prices: pd.Series,
        returns: Optional[pd.Series] = None,
        prices_b: Optional[pd.Series] = None,
        vol: Optional[pd.Series] = None,
        categories: Optional[List[str]] = None,
        param_grids: Optional[Dict[str, Dict[str, List]]] = None,
        top_n: int = 10,
        compute_sensitivity: bool = False,
    ) -> StrategyCatalog:
        """Generate, backtest, filter, and rank strategies."""
        if returns is None:
            returns = prices.pct_change().dropna()

        templates = self.generate_templates(categories, param_grids)
        candidates: List[StrategyCandidate] = []

        for tmpl in templates:
            sig = self.generate_signal(tmpl, prices, prices_b, vol)
            metrics = self.backtest(sig, returns)
            passed = self.filter_candidate(metrics)

            sens: Dict[str, float] = {}
            if compute_sensitivity and passed:
                sens = self.compute_sensitivity(tmpl, prices, returns, prices_b, vol)

            candidates.append(StrategyCandidate(
                template=tmpl, metrics=metrics,
                passed_filter=passed, sensitivity=sens,
            ))

        survivors = [c for c in candidates if c.passed_filter]
        survivors.sort(key=lambda c: c.metrics.sharpe, reverse=True)
        for i, c in enumerate(survivors):
            c.rank = i + 1

        top = survivors[:top_n]

        by_cat: Dict[str, List[StrategyCandidate]] = {}
        for c in survivors:
            by_cat.setdefault(c.template.category, []).append(c)

        return StrategyCatalog(
            total_generated=len(templates),
            total_passed=len(survivors),
            candidates=survivors,
            top_10=top,
            by_category=by_cat,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_bar(
        labels: List[str], values: List[float], title: str,
        width: int = 700, height: int = 220, color: str = "#2980b9",
    ) -> str:
        if not values:
            return ""
        n = len(values)
        vmax = max(abs(v) for v in values) or 1
        pad_l, pad_b = 100, 40
        pw = width - pad_l - 20
        ph = height - 55 - pad_b
        bw = pw / max(n, 1) * 0.7
        gap = pw / max(n, 1)
        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="background:#fff;border:1px solid #ddd;'
             f'border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="18" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')
        for i in range(n):
            x = pad_l + i * gap + (gap - bw) / 2
            bh = abs(values[i]) / vmax * ph
            y = 30 + ph - bh
            c = "#27ae60" if values[i] >= 0 else "#e74c3c"
            p.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" '
                     f'height="{max(bh, 1):.0f}" fill="{c}" rx="3"/>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{height - 8:.0f}" text-anchor="middle" '
                     f'font-size="8" fill="#666">{labels[i]}</text>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{y - 3:.0f}" text-anchor="middle" '
                     f'font-size="9" fill="#333">{values[i]:.2f}</text>')
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        catalog: StrategyCatalog,
        output_path: str = "reports/strategy_factory.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Top-10 Sharpe chart
        top_labels = [c.template.name for c in catalog.top_10]
        top_sharpes = [c.metrics.sharpe for c in catalog.top_10]
        sharpe_svg = self._svg_bar(top_labels, top_sharpes, "Top 10 Strategies by Sharpe")

        # Top-10 table
        top_rows = []
        for c in catalog.top_10:
            m = c.metrics
            top_rows.append(
                f"<tr><td>#{c.rank}</td>"
                f"<td style='text-align:left'>{c.template.name}</td>"
                f"<td>{c.template.category}</td>"
                f"<td>{m.sharpe:.2f}</td><td>{m.annual_return:.2%}</td>"
                f"<td>{m.max_drawdown:.2%}</td><td>{m.win_rate:.1%}</td>"
                f"<td>{m.profit_factor:.2f}</td><td>{m.n_trades}</td>"
                f"<td>{m.capacity_score:.2f}</td></tr>")

        # Category summary
        cat_rows = []
        for cat, cands in sorted(catalog.by_category.items()):
            avg_sh = float(np.mean([c.metrics.sharpe for c in cands]))
            best = cands[0].metrics.sharpe if cands else 0
            cat_rows.append(
                f"<tr><td style='text-align:left'>{cat}</td><td>{len(cands)}</td>"
                f"<td>{avg_sh:.2f}</td><td>{best:.2f}</td></tr>")

        # Sensitivity table for top strategies
        sens_rows = []
        for c in catalog.top_10[:5]:
            if c.sensitivity:
                for param, delta in c.sensitivity.items():
                    sens_rows.append(
                        f"<tr><td style='text-align:left'>{c.template.name}</td>"
                        f"<td>{param}</td><td>{delta:.3f}</td></tr>")

        sens_html = ""
        if sens_rows:
            sens_html = f"""
<h2>Parameter Sensitivity (top 5)</h2>
<table><tr><th style='text-align:left'>Strategy</th><th>Parameter</th><th>&Delta; Sharpe</th></tr>
{''.join(sens_rows)}</table>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Strategy Factory</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
</style></head><body>
<h1>Strategy Factory Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {catalog.total_generated} |
   <strong>Passed Filter:</strong> {catalog.total_passed} |
   <strong>Filter:</strong> Sharpe &ge; {self.criteria.min_sharpe:.1f},
   DD &le; {self.criteria.max_drawdown:.0%},
   Trades &ge; {self.criteria.min_trades}</p>
</div>

{sharpe_svg}

<h2>Top 10 Strategies</h2>
<table><tr><th>#</th><th style='text-align:left'>Name</th><th>Category</th>
<th>Sharpe</th><th>Return</th><th>Max DD</th><th>Win Rate</th>
<th>PF</th><th>Trades</th><th>Capacity</th></tr>
{''.join(top_rows)}</table>

<h2>Category Summary</h2>
<table><tr><th style='text-align:left'>Category</th><th>Survivors</th>
<th>Avg Sharpe</th><th>Best Sharpe</th></tr>
{''.join(cat_rows)}</table>

{sens_html}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Strategy factory report -> %s", path)
        return str(path)
