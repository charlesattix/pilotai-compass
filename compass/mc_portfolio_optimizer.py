"""
Monte Carlo portfolio optimizer with efficient frontier,
regime-conditional allocation, and comprehensive risk metrics.

Runs 10K random weight simulations across strategy/experiment return streams
to find optimal allocations under different regimes. Produces a self-contained
HTML report at reports/mc_portfolio.html.

This is READ-ONLY analysis. No broker connections, no trade placement.

Usage::

    from compass.mc_portfolio_optimizer import MCPortfolioOptimizer
    opt = MCPortfolioOptimizer(returns)
    result = opt.optimize()
    opt.generate_report(result)
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "mc_portfolio.html"
TRADING_DAYS = 252


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class PortfolioMetrics:
    """Risk/return metrics for a single portfolio allocation."""

    weights: np.ndarray
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    cvar_95: float
    calmar_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "annual_return": self.annual_return,
            "annual_volatility": self.annual_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "cvar_95": self.cvar_95,
            "calmar_ratio": self.calmar_ratio,
        }


@dataclass
class EfficientFrontier:
    """Collection of portfolios forming the efficient frontier."""

    portfolios: List[PortfolioMetrics]
    returns: np.ndarray
    volatilities: np.ndarray
    sharpes: np.ndarray

    @property
    def max_sharpe_portfolio(self) -> PortfolioMetrics:
        idx = int(np.argmax(self.sharpes))
        return self.portfolios[idx]

    @property
    def min_volatility_portfolio(self) -> PortfolioMetrics:
        idx = int(np.argmin(self.volatilities))
        return self.portfolios[idx]


@dataclass
class RegimeAllocation:
    """Optimal allocation for a specific market regime."""

    regime: str
    optimal_weights: np.ndarray
    metrics: PortfolioMetrics
    n_periods: int


@dataclass
class OptimizationResult:
    """Full result from MC portfolio optimization."""

    asset_names: List[str]
    all_portfolios: List[PortfolioMetrics]
    efficient_frontier: EfficientFrontier
    regime_allocations: Dict[str, RegimeAllocation]
    best_sharpe: PortfolioMetrics
    best_sortino: PortfolioMetrics
    min_cvar: PortfolioMetrics
    min_drawdown: PortfolioMetrics
    n_simulations: int


# ── Core optimizer ───────────────────────────────────────────────────────


class MCPortfolioOptimizer:
    """Monte Carlo portfolio optimizer with regime-conditional allocation."""

    def __init__(
        self,
        returns: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        n_simulations: int = 10_000,
        risk_free_rate: float = 0.05,
        seed: Optional[int] = None,
    ):
        """
        Args:
            returns: DataFrame of daily returns, each column is an asset/strategy.
            regimes: Optional Series aligned with returns index, values like
                     'bull', 'bear', 'sideways'.
            n_simulations: Number of random weight portfolios to simulate.
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            seed: Random seed for reproducibility.
        """
        if returns.empty:
            raise ValueError("returns DataFrame must not be empty")
        if returns.shape[1] < 2:
            raise ValueError("Need at least 2 assets for portfolio optimization")

        self.returns = returns.copy()
        self.regimes = regimes.copy() if regimes is not None else None
        self.n_simulations = n_simulations
        self.risk_free_rate = risk_free_rate
        self.seed = seed
        self.asset_names = list(returns.columns)
        self.n_assets = len(self.asset_names)

    # ── Weight generation ────────────────────────────────────────────

    def _generate_random_weights(self, rng: np.random.RandomState) -> np.ndarray:
        """Generate n_simulations sets of random portfolio weights summing to 1."""
        raw = rng.dirichlet(np.ones(self.n_assets), size=self.n_simulations)
        return raw

    # ── Risk metrics ─────────────────────────────────────────────────

    @staticmethod
    def compute_sharpe(
        returns_series: np.ndarray, risk_free_rate: float = 0.05
    ) -> float:
        """Annualized Sharpe ratio."""
        if len(returns_series) < 2:
            return 0.0
        mean_daily = np.mean(returns_series)
        std_daily = np.std(returns_series, ddof=1)
        if std_daily < 1e-12:
            return 0.0
        daily_rf = risk_free_rate / TRADING_DAYS
        return float((mean_daily - daily_rf) / std_daily * math.sqrt(TRADING_DAYS))

    @staticmethod
    def compute_sortino(
        returns_series: np.ndarray, risk_free_rate: float = 0.05
    ) -> float:
        """Annualized Sortino ratio (downside deviation)."""
        if len(returns_series) < 2:
            return 0.0
        mean_daily = np.mean(returns_series)
        daily_rf = risk_free_rate / TRADING_DAYS
        downside = returns_series[returns_series < daily_rf] - daily_rf
        if len(downside) == 0:
            return float("inf") if mean_daily > daily_rf else 0.0
        downside_std = np.sqrt(np.mean(downside**2))
        if downside_std < 1e-12:
            return 0.0
        return float(
            (mean_daily - daily_rf) / downside_std * math.sqrt(TRADING_DAYS)
        )

    @staticmethod
    def compute_max_drawdown(returns_series: np.ndarray) -> float:
        """Maximum drawdown from cumulative returns."""
        if len(returns_series) == 0:
            return 0.0
        cumulative = np.cumprod(1.0 + returns_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.min(drawdowns))

    @staticmethod
    def compute_cvar(returns_series: np.ndarray, alpha: float = 0.05) -> float:
        """Conditional Value at Risk (Expected Shortfall) at alpha level."""
        if len(returns_series) == 0:
            return 0.0
        sorted_returns = np.sort(returns_series)
        cutoff = max(1, int(len(sorted_returns) * alpha))
        return float(np.mean(sorted_returns[:cutoff]))

    def _compute_portfolio_metrics(
        self, weights: np.ndarray, daily_returns: np.ndarray
    ) -> PortfolioMetrics:
        """Compute full metrics for a single portfolio."""
        port_returns = daily_returns @ weights
        ann_ret = float(np.mean(port_returns) * TRADING_DAYS)
        ann_vol = float(np.std(port_returns, ddof=1) * math.sqrt(TRADING_DAYS))
        sharpe = self.compute_sharpe(port_returns, self.risk_free_rate)
        sortino = self.compute_sortino(port_returns, self.risk_free_rate)
        max_dd = self.compute_max_drawdown(port_returns)
        cvar = self.compute_cvar(port_returns)
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-12 else 0.0

        return PortfolioMetrics(
            weights=weights,
            annual_return=ann_ret,
            annual_volatility=ann_vol,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            cvar_95=cvar,
            calmar_ratio=calmar,
        )

    # ── Simulation ───────────────────────────────────────────────────

    def _simulate(self, returns_data: np.ndarray) -> List[PortfolioMetrics]:
        """Run MC simulation on given returns matrix."""
        rng = np.random.RandomState(self.seed)
        all_weights = self._generate_random_weights(rng)
        portfolios: List[PortfolioMetrics] = []
        for i in range(self.n_simulations):
            pm = self._compute_portfolio_metrics(all_weights[i], returns_data)
            portfolios.append(pm)
        return portfolios

    def _build_efficient_frontier(
        self, portfolios: List[PortfolioMetrics]
    ) -> EfficientFrontier:
        """Extract efficient frontier from simulated portfolios."""
        rets = np.array([p.annual_return for p in portfolios])
        vols = np.array([p.annual_volatility for p in portfolios])
        sharpes = np.array([p.sharpe_ratio for p in portfolios])

        # Bin by volatility and keep the highest return in each bin
        n_bins = min(100, len(portfolios) // 10)
        if n_bins < 2:
            return EfficientFrontier(
                portfolios=portfolios,
                returns=rets,
                volatilities=vols,
                sharpes=sharpes,
            )

        vol_bins = np.linspace(vols.min(), vols.max(), n_bins + 1)
        frontier_indices: List[int] = []
        for b in range(n_bins):
            mask = (vols >= vol_bins[b]) & (vols < vol_bins[b + 1])
            if not np.any(mask):
                continue
            candidates = np.where(mask)[0]
            best = candidates[np.argmax(rets[candidates])]
            frontier_indices.append(int(best))

        frontier_portfolios = [portfolios[i] for i in frontier_indices]
        frontier_rets = rets[frontier_indices]
        frontier_vols = vols[frontier_indices]
        frontier_sharpes = sharpes[frontier_indices]

        # Sort by volatility
        order = np.argsort(frontier_vols)
        frontier_portfolios = [frontier_portfolios[i] for i in order]
        frontier_rets = frontier_rets[order]
        frontier_vols = frontier_vols[order]
        frontier_sharpes = frontier_sharpes[order]

        return EfficientFrontier(
            portfolios=frontier_portfolios,
            returns=frontier_rets,
            volatilities=frontier_vols,
            sharpes=frontier_sharpes,
        )

    # ── Regime allocation ────────────────────────────────────────────

    def _compute_regime_allocations(
        self,
    ) -> Dict[str, RegimeAllocation]:
        """Optimize separately for each market regime."""
        if self.regimes is None:
            return {}

        allocations: Dict[str, RegimeAllocation] = {}
        for regime in self.regimes.unique():
            mask = self.regimes == regime
            regime_returns = self.returns.loc[mask].values
            if len(regime_returns) < 10:
                logger.warning(
                    "Regime '%s' has only %d periods, skipping",
                    regime,
                    len(regime_returns),
                )
                continue

            portfolios = self._simulate(regime_returns)
            best_idx = int(
                np.argmax([p.sharpe_ratio for p in portfolios])
            )
            best = portfolios[best_idx]
            allocations[regime] = RegimeAllocation(
                regime=regime,
                optimal_weights=best.weights,
                metrics=best,
                n_periods=int(mask.sum()),
            )

        return allocations

    # ── Main optimize ────────────────────────────────────────────────

    def optimize(self) -> OptimizationResult:
        """Run full MC optimization and return results."""
        daily_data = self.returns.values
        all_portfolios = self._simulate(daily_data)

        efficient_frontier = self._build_efficient_frontier(all_portfolios)

        sharpes = [p.sharpe_ratio for p in all_portfolios]
        sortinos = [p.sortino_ratio for p in all_portfolios]
        cvars = [p.cvar_95 for p in all_portfolios]
        drawdowns = [p.max_drawdown for p in all_portfolios]

        best_sharpe = all_portfolios[int(np.argmax(sharpes))]
        best_sortino = all_portfolios[int(np.argmax(sortinos))]
        min_cvar = all_portfolios[int(np.argmax(cvars))]  # least negative
        min_drawdown = all_portfolios[int(np.argmax(drawdowns))]  # least negative

        regime_allocations = self._compute_regime_allocations()

        return OptimizationResult(
            asset_names=self.asset_names,
            all_portfolios=all_portfolios,
            efficient_frontier=efficient_frontier,
            regime_allocations=regime_allocations,
            best_sharpe=best_sharpe,
            best_sortino=best_sortino,
            min_cvar=min_cvar,
            min_drawdown=min_drawdown,
            n_simulations=self.n_simulations,
        )

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: OptimizationResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        """Generate self-contained HTML report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation (module-level) ───────────────────────────────────────


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_ratio(v: float) -> str:
    if v == float("inf"):
        return "∞"
    return f"{v:.3f}"


def _weights_table(names: List[str], weights: np.ndarray) -> str:
    rows = "".join(
        f"<tr><td>{n}</td><td>{_fmt_pct(w)}</td></tr>"
        for n, w in zip(names, weights)
    )
    return f"""
    <table class="weights">
      <tr><th>Asset</th><th>Weight</th></tr>
      {rows}
    </table>"""


def _metrics_card(title: str, m: PortfolioMetrics, names: List[str]) -> str:
    return f"""
    <div class="card">
      <h3>{title}</h3>
      <div class="metrics-grid">
        <div><span class="label">Annual Return</span><span class="value">{_fmt_pct(m.annual_return)}</span></div>
        <div><span class="label">Annual Vol</span><span class="value">{_fmt_pct(m.annual_volatility)}</span></div>
        <div><span class="label">Sharpe</span><span class="value">{_fmt_ratio(m.sharpe_ratio)}</span></div>
        <div><span class="label">Sortino</span><span class="value">{_fmt_ratio(m.sortino_ratio)}</span></div>
        <div><span class="label">Max DD</span><span class="value">{_fmt_pct(m.max_drawdown)}</span></div>
        <div><span class="label">CVaR 95</span><span class="value">{_fmt_pct(m.cvar_95)}</span></div>
        <div><span class="label">Calmar</span><span class="value">{_fmt_ratio(m.calmar_ratio)}</span></div>
      </div>
      {_weights_table(names, m.weights)}
    </div>"""


def _scatter_svg(result: OptimizationResult) -> str:
    """Inline SVG scatter plot of risk vs return colored by Sharpe."""
    vols = [p.annual_volatility for p in result.all_portfolios]
    rets = [p.annual_return for p in result.all_portfolios]
    sharpes = [p.sharpe_ratio for p in result.all_portfolios]

    # Sample for SVG performance
    step = max(1, len(vols) // 2000)
    vols_s = vols[::step]
    rets_s = rets[::step]
    sharpes_s = sharpes[::step]

    w, h = 600, 400
    pad = 60
    min_v, max_v = min(vols_s), max(vols_s)
    min_r, max_r = min(rets_s), max(rets_s)
    range_v = max_v - min_v if max_v > min_v else 1e-6
    range_r = max_r - min_r if max_r > min_r else 1e-6
    min_sh = min(sharpes_s)
    range_sh = max(sharpes_s) - min_sh if max(sharpes_s) > min_sh else 1.0

    dots = []
    for v, r, s in zip(vols_s, rets_s, sharpes_s):
        x = pad + (v - min_v) / range_v * (w - 2 * pad)
        y = h - pad - (r - min_r) / range_r * (h - 2 * pad)
        hue = 240 - (s - min_sh) / range_sh * 240  # blue→red
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" '
            f'fill="hsl({hue:.0f},70%,50%)" opacity="0.5"/>'
        )

    # Frontier line
    ef = result.efficient_frontier
    frontier_pts = []
    for v, r in zip(ef.volatilities, ef.returns):
        x = pad + (v - min_v) / range_v * (w - 2 * pad)
        y = h - pad - (r - min_r) / range_r * (h - 2 * pad)
        frontier_pts.append(f"{x:.1f},{y:.1f}")

    frontier_line = ""
    if len(frontier_pts) > 1:
        frontier_line = (
            f'<polyline points="{" ".join(frontier_pts)}" '
            f'fill="none" stroke="#ff0" stroke-width="2"/>'
        )

    return f"""
    <svg viewBox="0 0 {w} {h}" class="scatter">
      <text x="{w//2}" y="20" text-anchor="middle" class="svg-title">
        Risk vs Return (colored by Sharpe)
      </text>
      <text x="{w//2}" y="{h-5}" text-anchor="middle" class="svg-label">
        Volatility
      </text>
      <text x="15" y="{h//2}" text-anchor="middle" class="svg-label"
            transform="rotate(-90,15,{h//2})">Return</text>
      {"".join(dots)}
      {frontier_line}
    </svg>"""


def _regime_section(result: OptimizationResult) -> str:
    if not result.regime_allocations:
        return ""
    cards = ""
    for regime, alloc in sorted(result.regime_allocations.items()):
        cards += f"""
        <div class="regime-card">
          <h4>{regime.upper()} regime ({alloc.n_periods} days)</h4>
          {_metrics_card("Optimal " + regime, alloc.metrics, result.asset_names)}
        </div>"""
    return f"""
    <section>
      <h2>Regime-Conditional Allocations</h2>
      <div class="regime-grid">{cards}</div>
    </section>"""


def _build_html(result: OptimizationResult) -> str:
    names = result.asset_names
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>MC Portfolio Optimization Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3, h4 {{ color: #58a6ff; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
              gap: 16px; margin: 20px 0; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .metrics-grid .label {{ color: #8b949e; font-size: 0.85em; }}
  .metrics-grid .value {{ color: #f0f6fc; font-weight: 600; }}
  table.weights {{ width: 100%; margin-top: 12px; border-collapse: collapse; }}
  table.weights th, table.weights td {{ padding: 4px 8px; text-align: left;
                                         border-bottom: 1px solid #21262d; }}
  table.weights th {{ color: #8b949e; }}
  .scatter {{ width: 100%; max-width: 700px; margin: 20px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 14px; }}
  .svg-label {{ fill: #8b949e; font-size: 11px; }}
  .regime-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                  gap: 16px; }}
  .regime-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                  padding: 16px; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>Monte Carlo Portfolio Optimization</h1>
<p class="meta">{result.n_simulations:,} simulations &middot; {len(names)} assets:
   {', '.join(names)}</p>

{_scatter_svg(result)}

<h2>Optimal Portfolios</h2>
<div class="summary">
  {_metrics_card("Best Sharpe", result.best_sharpe, names)}
  {_metrics_card("Best Sortino", result.best_sortino, names)}
  {_metrics_card("Min CVaR", result.min_cvar, names)}
  {_metrics_card("Min Drawdown", result.min_drawdown, names)}
</div>

<h2>Efficient Frontier</h2>
<div class="card">
  <p>Max-Sharpe on frontier: Sharpe={_fmt_ratio(result.efficient_frontier.max_sharpe_portfolio.sharpe_ratio)},
     Return={_fmt_pct(result.efficient_frontier.max_sharpe_portfolio.annual_return)},
     Vol={_fmt_pct(result.efficient_frontier.max_sharpe_portfolio.annual_volatility)}</p>
  <p>Min-Vol on frontier: Vol={_fmt_pct(result.efficient_frontier.min_volatility_portfolio.annual_volatility)},
     Return={_fmt_pct(result.efficient_frontier.min_volatility_portfolio.annual_return)}</p>
</div>

{_regime_section(result)}

</body>
</html>"""
