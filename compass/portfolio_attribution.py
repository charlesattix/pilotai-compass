"""
Portfolio performance attribution engine.

Decomposes combined portfolio returns into 6 alpha sources:
  1. Strategy selection   — which strategies contributed most
  2. Timing alpha         — did regime switches add value
  3. Sizing alpha         — did dynamic sizing help vs equal-weight
  4. Hedging cost/benefit — crisis hedge cost vs drawdown saved
  5. Execution cost       — slippage and spread costs
  6. Factor exposure      — market beta, vol, momentum contributions

Produces monthly attribution reports.
All methods work on pre-loaded data — no API calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
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
class StrategySelectionAlpha:
    """Which strategies drove returns."""
    strategy: str
    weight: float
    gross_return: float
    contribution: float         # weight × return
    pct_of_total: float


@dataclass
class TimingAlpha:
    """Value added by regime-based switching."""
    timing_return: float        # regime-timed portfolio return
    static_return: float        # equal-weight static return
    timing_alpha: float         # difference
    n_regime_switches: int
    best_regime: str
    worst_regime: str


@dataclass
class SizingAlpha:
    """Value of dynamic vs fixed position sizing."""
    dynamic_return: float
    fixed_return: float
    sizing_alpha: float
    avg_dynamic_size: float
    n_size_changes: int


@dataclass
class HedgeCostBenefit:
    """Crisis hedge cost vs drawdown saved."""
    hedge_cost: float           # total drag from hedging
    drawdown_saved: float       # DD reduction from hedging
    net_benefit: float          # drawdown_saved - hedge_cost
    cost_ratio: float           # cost / gross_premium
    n_hedge_activations: int


@dataclass
class ExecutionCostAttr:
    """Execution costs breakdown."""
    total_slippage: float
    spread_cost: float
    timing_cost: float
    total_execution_cost: float
    cost_as_pct_return: float


@dataclass
class FactorAttribution:
    """Return attributed to factors."""
    market_beta: float
    market_contribution: float
    vol_contribution: float
    momentum_contribution: float
    alpha_residual: float
    r_squared: float


@dataclass
class MonthlyAttribution:
    """One month's full attribution."""
    month: str
    total_return: float
    strategy_alpha: float
    timing_alpha: float
    sizing_alpha: float
    hedge_cost: float
    execution_cost: float
    factor_contribution: float
    residual: float


@dataclass
class FullAttribution:
    """Complete attribution result."""
    strategy_selection: List[StrategySelectionAlpha]
    timing: TimingAlpha
    sizing: SizingAlpha
    hedging: HedgeCostBenefit
    execution: ExecutionCostAttr
    factors: FactorAttribution
    monthly: List[MonthlyAttribution]
    total_return: float
    sharpe: float


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_portfolio_data(
    n_days: int = 756, seed: int = 42,
) -> Dict[str, pd.Series]:
    """Generate multi-strategy portfolio data for attribution."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n_days)

    # Strategy returns
    credit_spread = pd.Series(rng.normal(0.0008, 0.006, n_days), index=idx)
    iron_condor = pd.Series(rng.normal(0.0005, 0.005, n_days), index=idx)
    vol_harvest = pd.Series(rng.normal(0.0006, 0.004, n_days), index=idx)

    # COVID-like crisis at day 100
    if n_days > 130:
        credit_spread.iloc[100:120] = rng.normal(-0.005, 0.012, 20)
        iron_condor.iloc[100:120] = rng.normal(-0.003, 0.008, 20)

    # Regime series
    regimes = pd.Series("bull", index=idx)
    if n_days > 130:
        regimes.iloc[95:130] = "bear"
    if n_days > 500:
        regimes.iloc[400:450] = "high_vol"

    # Dynamic weights (regime-adaptive)
    dynamic_weights = pd.DataFrame({
        "credit_spread": np.where(regimes == "bull", 0.50, 0.20),
        "iron_condor": np.where(regimes == "high_vol", 0.50, 0.25),
        "vol_harvest": np.where(regimes == "bear", 0.10, 0.30),
    }, index=idx)
    # Normalise
    dynamic_weights = dynamic_weights.div(dynamic_weights.sum(axis=1), axis=0)

    # Market factor
    market = pd.Series(rng.normal(0.0004, 0.01, n_days), index=idx)
    if n_days > 120:
        market.iloc[100:120] = rng.normal(-0.010, 0.020, 20)

    # Hedge indicator
    hedge_active = pd.Series(0.0, index=idx)
    if n_days > 130:
        hedge_active.iloc[105:130] = 1.0

    return {
        "credit_spread": credit_spread,
        "iron_condor": iron_condor,
        "vol_harvest": vol_harvest,
        "regimes": regimes,
        "dynamic_weights": dynamic_weights,
        "market": market,
        "hedge_active": hedge_active,
    }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PortfolioAttributionEngine:
    """Full portfolio performance attribution.

    Args:
        hedge_cost_annual: Annual cost of hedge overlay.
        slippage_per_trade: Slippage per trade (fraction).
    """

    def __init__(
        self,
        hedge_cost_annual: float = 0.003,
        slippage_per_trade: float = 0.001,
    ) -> None:
        self.hedge_cost_annual = hedge_cost_annual
        self.slippage_per_trade = slippage_per_trade

    # ------------------------------------------------------------------
    # 1. Strategy selection alpha
    # ------------------------------------------------------------------

    @staticmethod
    def strategy_selection(
        strategy_returns: Dict[str, pd.Series],
        weights: pd.DataFrame,
    ) -> List[StrategySelectionAlpha]:
        """Decompose return by strategy contribution."""
        total_port = pd.Series(0.0, index=weights.index)
        contributions: Dict[str, float] = {}
        gross: Dict[str, float] = {}

        for strat in weights.columns:
            if strat in strategy_returns:
                contrib = weights[strat] * strategy_returns[strat]
                contributions[strat] = float(contrib.sum())
                gross[strat] = float(strategy_returns[strat].sum())
                total_port += contrib

        total = float(total_port.sum())
        results = []
        for strat in weights.columns:
            if strat in contributions:
                avg_w = float(weights[strat].mean())
                pct = contributions[strat] / total if abs(total) > 1e-8 else 0
                results.append(StrategySelectionAlpha(
                    strat, avg_w, gross[strat], contributions[strat], pct))

        results.sort(key=lambda s: s.contribution, reverse=True)
        return results

    # ------------------------------------------------------------------
    # 2. Timing alpha (regime switching value)
    # ------------------------------------------------------------------

    @staticmethod
    def timing_alpha(
        strategy_returns: Dict[str, pd.Series],
        dynamic_weights: pd.DataFrame,
        regimes: pd.Series,
    ) -> TimingAlpha:
        """Compare regime-timed vs equal-weight allocation."""
        strats = [s for s in dynamic_weights.columns if s in strategy_returns]
        n = len(strats)
        if n == 0:
            return TimingAlpha(0, 0, 0, 0, "", "")

        # Dynamic portfolio
        dynamic_ret = pd.Series(0.0, index=dynamic_weights.index)
        for s in strats:
            dynamic_ret += dynamic_weights[s] * strategy_returns[s]

        # Static equal-weight
        static_ret = pd.Series(0.0, index=dynamic_weights.index)
        for s in strats:
            static_ret += (1.0 / n) * strategy_returns[s]

        timing = float(dynamic_ret.sum()) - float(static_ret.sum())
        n_switches = int((regimes != regimes.shift(1)).sum())

        # Best/worst regime
        by_regime: Dict[str, float] = {}
        aligned = pd.DataFrame({"ret": dynamic_ret, "reg": regimes})
        for regime, grp in aligned.groupby("reg"):
            by_regime[str(regime)] = float(grp["ret"].sum())
        best = max(by_regime, key=by_regime.get) if by_regime else ""
        worst = min(by_regime, key=by_regime.get) if by_regime else ""

        return TimingAlpha(
            float(dynamic_ret.sum()), float(static_ret.sum()),
            timing, n_switches, best, worst)

    # ------------------------------------------------------------------
    # 3. Sizing alpha
    # ------------------------------------------------------------------

    @staticmethod
    def sizing_alpha(
        portfolio_returns: pd.Series,
        dynamic_sizes: pd.Series,
    ) -> SizingAlpha:
        """Compare dynamic sizing vs fixed size."""
        fixed_ret = portfolio_returns  # base returns at 1x
        dynamic_ret = portfolio_returns * dynamic_sizes  # scaled

        n_changes = int((dynamic_sizes.diff().abs() > 0.01).sum())
        avg_size = float(dynamic_sizes.mean())

        return SizingAlpha(
            float(dynamic_ret.sum()), float(fixed_ret.sum()),
            float(dynamic_ret.sum() - fixed_ret.sum()),
            avg_size, n_changes)

    # ------------------------------------------------------------------
    # 4. Hedging cost/benefit
    # ------------------------------------------------------------------

    def hedge_attribution(
        self,
        portfolio_returns: pd.Series,
        hedge_active: pd.Series,
        unhedged_dd: float,
        hedged_dd: float,
    ) -> HedgeCostBenefit:
        """Compute hedge cost vs drawdown saved."""
        n_active = int(hedge_active.sum())
        total_cost = n_active * self.hedge_cost_annual / TRADING_DAYS
        dd_saved = unhedged_dd - hedged_dd
        gross = float(portfolio_returns.abs().sum())

        return HedgeCostBenefit(
            hedge_cost=total_cost,
            drawdown_saved=dd_saved,
            net_benefit=dd_saved - total_cost,
            cost_ratio=total_cost / gross if gross > 0 else 0,
            n_hedge_activations=n_active,
        )

    # ------------------------------------------------------------------
    # 5. Execution cost
    # ------------------------------------------------------------------

    def execution_attribution(
        self,
        n_trades: int,
        avg_premium: float,
        total_return: float,
    ) -> ExecutionCostAttr:
        """Estimate execution costs."""
        slippage = n_trades * self.slippage_per_trade * avg_premium
        spread_cost = slippage * 0.6  # 60% is spread
        timing_cost = slippage * 0.4  # 40% is timing
        total = slippage
        cost_pct = total / abs(total_return) if abs(total_return) > 1e-8 else 0

        return ExecutionCostAttr(slippage, spread_cost, timing_cost, total, cost_pct)

    # ------------------------------------------------------------------
    # 6. Factor attribution
    # ------------------------------------------------------------------

    @staticmethod
    def factor_attribution(
        portfolio_returns: pd.Series,
        market_returns: pd.Series,
    ) -> FactorAttribution:
        """OLS regression against market factor."""
        aligned = pd.DataFrame({"port": portfolio_returns, "mkt": market_returns}).dropna()
        if len(aligned) < 20:
            return FactorAttribution(0, 0, 0, 0, 0, 0)

        y = aligned["port"].values
        x = np.column_stack([np.ones(len(y)), aligned["mkt"].values])
        try:
            b, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        except np.linalg.LinAlgError:
            return FactorAttribution(0, 0, 0, 0, 0, 0)

        alpha = float(b[0])
        beta = float(b[1])
        mkt_contrib = beta * float(aligned["mkt"].sum())
        alpha_total = alpha * len(aligned)

        pred = x @ b
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0

        # Vol and momentum proxies from market
        vol_proxy = aligned["mkt"].rolling(21).std().fillna(0)
        mom_proxy = aligned["mkt"].rolling(20).mean().fillna(0)
        vol_contrib = float((vol_proxy * y).sum()) * 0.1
        mom_contrib = float((mom_proxy * y).sum()) * 0.1

        return FactorAttribution(beta, mkt_contrib, vol_contrib, mom_contrib, alpha_total, r2)

    # ------------------------------------------------------------------
    # Monthly attribution
    # ------------------------------------------------------------------

    def monthly_attribution(
        self,
        strategy_returns: Dict[str, pd.Series],
        weights: pd.DataFrame,
        market: pd.Series,
        hedge_active: pd.Series,
        regimes: pd.Series,
    ) -> List[MonthlyAttribution]:
        """Compute attribution per calendar month."""
        strats = [s for s in weights.columns if s in strategy_returns]
        if not strats:
            return []

        # Build daily portfolio return
        port_ret = pd.Series(0.0, index=weights.index)
        for s in strats:
            port_ret += weights[s] * strategy_returns[s]

        # Static equal-weight
        static_ret = pd.Series(0.0, index=weights.index)
        for s in strats:
            static_ret += (1.0 / len(strats)) * strategy_returns[s]

        aligned = pd.DataFrame({
            "port": port_ret, "static": static_ret,
            "mkt": market, "hedge": hedge_active, "regime": regimes,
        }).dropna()

        months: List[MonthlyAttribution] = []
        for period, grp in aligned.groupby(aligned.index.to_period("M")):
            total = float(grp["port"].sum())
            timing = float(grp["port"].sum() - grp["static"].sum())
            sizing = 0.0  # would need dynamic size series
            hedge = -float(grp["hedge"].sum()) * self.hedge_cost_annual / TRADING_DAYS
            exec_cost = -len(grp) * self.slippage_per_trade * 0.001

            # Factor: simple beta × market
            if len(grp) >= 5:
                cov = np.cov(grp["port"].values, grp["mkt"].values)
                beta = cov[0, 1] / (cov[1, 1] + 1e-12)
                factor = beta * float(grp["mkt"].sum())
            else:
                factor = 0.0

            strategy = total - timing  # strategy selection = total minus timing
            residual = total - strategy - timing - sizing - hedge - exec_cost - factor

            months.append(MonthlyAttribution(
                str(period), total, strategy, timing, sizing,
                hedge, exec_cost, factor, residual))

        return months

    # ------------------------------------------------------------------
    # Full attribution
    # ------------------------------------------------------------------

    def attribute(
        self,
        strategy_returns: Dict[str, pd.Series],
        weights: pd.DataFrame,
        regimes: pd.Series,
        market: pd.Series,
        hedge_active: pd.Series,
        n_trades: int = 100,
        avg_premium: float = 1.50,
        unhedged_dd: float = 0.20,
        hedged_dd: float = 0.10,
    ) -> FullAttribution:
        """Run complete attribution."""
        strats = [s for s in weights.columns if s in strategy_returns]

        # Portfolio return
        port_ret = pd.Series(0.0, index=weights.index)
        for s in strats:
            port_ret += weights[s] * strategy_returns[s]

        selection = self.strategy_selection(strategy_returns, weights)
        timing = self.timing_alpha(strategy_returns, weights, regimes)
        sizing = self.sizing_alpha(port_ret, pd.Series(1.0, index=port_ret.index))
        hedging = self.hedge_attribution(port_ret, hedge_active, unhedged_dd, hedged_dd)
        execution = self.execution_attribution(n_trades, avg_premium, float(port_ret.sum()))
        factors = self.factor_attribution(port_ret, market)
        monthly = self.monthly_attribution(strategy_returns, weights, market, hedge_active, regimes)

        total = float(port_ret.sum())
        mu, std = float(port_ret.mean()), float(port_ret.std())
        sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0

        return FullAttribution(
            selection, timing, sizing, hedging, execution,
            factors, monthly, total, sharpe)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        result: FullAttribution,
        output_path: str = "reports/portfolio_attribution.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Strategy table
        strat_rows = [
            f"<tr><td style='text-align:left'>{s.strategy}</td><td>{s.weight:.1%}</td>"
            f"<td>{s.gross_return * 10000:+.0f}</td><td>{s.contribution * 10000:+.0f}</td>"
            f"<td>{s.pct_of_total:.0%}</td></tr>"
            for s in result.strategy_selection
        ]

        # Waterfall SVG
        items = [
            ("Strategy", result.timing.static_return),
            ("Timing", result.timing.timing_alpha),
            ("Sizing", result.sizing.sizing_alpha),
            ("Hedging", -result.hedging.hedge_cost),
            ("Execution", -result.execution.total_execution_cost),
            ("Factor", result.factors.market_contribution),
        ]
        items = [(l, v) for l, v in items if abs(v) > 1e-8]
        waterfall_svg = ""
        if items:
            n = len(items)
            w, h = 650, 220
            pad_l, pad_b = 80, 40
            pw, ph = w - pad_l - 20, h - 60 - pad_b
            vals = [v for _, v in items]
            vmax = max(abs(v) for v in vals) or 1
            bw = pw / max(n, 1) * 0.7
            gap = pw / max(n, 1)
            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                      f'style="background:#fff;border:1px solid #e2e8f0;border-radius:6px;margin:.5rem 0">']
            parts.append(f'<text x="{w // 2}" y="18" text-anchor="middle" font-size="12" '
                          f'font-weight="bold" fill="#0f172a">Return Attribution Waterfall (bps)</text>')
            base_y = 30 + ph
            for i, (label, val) in enumerate(items):
                x = pad_l + i * gap + (gap - bw) / 2
                bh = abs(val) / vmax * ph * 10000  # scale to bps
                bh = min(bh, ph)
                y = base_y - bh if val >= 0 else base_y
                c = "#059669" if val >= 0 else "#dc2626"
                parts.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" height="{max(bh, 1):.0f}" fill="{c}" rx="3"/>')
                parts.append(f'<text x="{x + bw / 2:.0f}" y="{h - 8:.0f}" text-anchor="middle" font-size="9" fill="#64748b">{label}</text>')
                parts.append(f'<text x="{x + bw / 2:.0f}" y="{y - 3:.0f}" text-anchor="middle" font-size="9" fill="#334155">{val * 10000:+.0f}</text>')
            parts.append("</svg>")
            waterfall_svg = "\n".join(parts)

        # Monthly table
        month_rows = [
            f"<tr><td>{m.month}</td><td>{m.total_return * 10000:+.0f}</td>"
            f"<td>{m.strategy_alpha * 10000:+.0f}</td><td>{m.timing_alpha * 10000:+.0f}</td>"
            f"<td>{m.hedge_cost * 10000:+.0f}</td><td>{m.factor_contribution * 10000:+.0f}</td></tr>"
            for m in result.monthly[-12:]
        ]

        t = result.timing
        s = result.sizing
        hg = result.hedging
        ex = result.execution
        f = result.factors

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Portfolio Attribution</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fff; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; }}
h2 {{ color: #334155; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; border-bottom: 2px solid #e2e8f0; }}
th:first-child {{ text-align: left; }}
td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #f1f5f9; }}
td:first-child {{ text-align: left; font-weight: 500; }}
.card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }}
.green {{ color: #059669; font-weight: 700; }}
.red {{ color: #dc2626; }}
</style></head><body>
<h1>EXP-1510-max: Portfolio Performance Attribution</h1>
<div class="card">
<p><strong>Total Return:</strong> {result.total_return * 10000:+.0f} bps |
<strong>Sharpe:</strong> {result.sharpe:.2f} |
<strong>Timing Alpha:</strong> {t.timing_alpha * 10000:+.0f} bps |
<strong>Hedge Net Benefit:</strong> {hg.net_benefit:.1%}</p>
</div>

<h2>Attribution Waterfall</h2>
{waterfall_svg}

<h2>Strategy Selection (bps)</h2>
<table><tr><th style='text-align:left'>Strategy</th><th>Avg Weight</th><th>Gross</th><th>Contribution</th><th>% of Total</th></tr>
{''.join(strat_rows)}</table>

<h2>Alpha Sources</h2>
<table>
<tr><td>Timing Alpha (regime switching)</td><td>{t.timing_alpha * 10000:+.0f} bps</td><td>{t.n_regime_switches} switches, best: {t.best_regime}</td></tr>
<tr><td>Sizing Alpha (dynamic vs fixed)</td><td>{s.sizing_alpha * 10000:+.0f} bps</td><td>{s.n_size_changes} size changes</td></tr>
<tr><td>Hedge Cost</td><td class="red">{hg.hedge_cost * 10000:+.0f} bps</td><td>{hg.n_hedge_activations} active days, DD saved: {hg.drawdown_saved:.1%}</td></tr>
<tr><td>Execution Cost</td><td class="red">{ex.total_execution_cost * 10000:+.0f} bps</td><td>{ex.cost_as_pct_return:.1%} of returns</td></tr>
<tr><td>Market Beta</td><td>{f.market_contribution * 10000:+.0f} bps</td><td>beta={f.market_beta:.3f}, R²={f.r_squared:.2f}</td></tr>
</table>

<h2>Monthly Attribution (bps, last 12 months)</h2>
<table><tr><th>Month</th><th>Total</th><th>Strategy</th><th>Timing</th><th>Hedge</th><th>Factor</th></tr>
{''.join(month_rows)}</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
