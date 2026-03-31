"""
compass/capacity_analyzer.py — Portfolio capacity analysis engine.

Estimates maximum AUM before alpha decay, models market impact, analyzes
option liquidity, and produces capacity curves per experiment.

Provides:
  1. Market impact model — price impact as f(order_size / ADV) using
     square-root model (Kyle 1985, Almgren-Chriss)
  2. Liquidity analysis — bid-ask spread, open interest, volume by
     strike/expiry
  3. Capacity curve — expected Sharpe degradation as AUM increases
  4. Optimal AUM recommendation per experiment and blended portfolio
  5. HTML report with capacity curves, impact chart, liquidity heatmap,
     AUM recommendations table

Usage:
    from compass.capacity_analyzer import CapacityAnalyzer

    analyzer = CapacityAnalyzer(
        experiment_returns={"EXP-400": r400, "EXP-503": r503},
        experiment_metadata={"EXP-400": {"avg_contracts": 5, "avg_spread_width": 5.0}, ...},
        adv_shares=50_000,
    )
    result = analyzer.run_all()
    html = analyzer.generate_html(result)
"""

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PERIODS_PER_YEAR = 252

# Square-root impact model calibration (Kyle 1985).
# impact_bps = IMPACT_COEFF * sqrt(participation_rate)
# Calibrated to SPY options: ~5 bps at 1% participation.
IMPACT_COEFF = 50.0  # bps per sqrt(participation)

# Temporary impact decay factor (fraction that is permanent).
PERMANENT_IMPACT_FRAC = 0.60

# Default bid-ask spread cost in bps for SPY options.
DEFAULT_SPREAD_BPS = 15.0

# AUM grid for capacity curve (in dollars).
DEFAULT_AUM_GRID = [
    50_000, 100_000, 250_000, 500_000, 1_000_000,
    2_500_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000,
]

# Sharpe considered "decayed" when it drops below this fraction of base.
ALPHA_DECAY_THRESHOLD = 0.50


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class MarketImpactEstimate:
    """Market impact for a given order size."""
    order_contracts: int
    adv_contracts: int
    participation_rate: float  # order / ADV
    temporary_impact_bps: float
    permanent_impact_bps: float
    total_impact_bps: float
    spread_cost_bps: float
    total_cost_bps: float  # impact + spread


@dataclass
class LiquidityProfile:
    """Liquidity characteristics for an experiment's typical trades."""
    experiment_id: str
    avg_bid_ask_spread_bps: float
    avg_daily_volume: float  # contracts
    avg_open_interest: float  # contracts
    volume_concentration: float  # fraction of volume in top 3 strikes
    oi_concentration: float  # fraction of OI in top 3 strikes
    liquidity_score: float  # 0-100 composite score


@dataclass
class CapacityCurvePoint:
    """Single point on the capacity curve."""
    aum: float
    contracts_per_trade: float
    participation_rate: float
    total_cost_bps: float
    net_sharpe: float
    sharpe_retention: float  # fraction of base Sharpe retained


@dataclass
class ExperimentCapacity:
    """Capacity analysis for one experiment."""
    experiment_id: str
    base_sharpe: float
    base_annual_return: float
    base_annual_vol: float
    capacity_curve: List[CapacityCurvePoint]
    optimal_aum: float  # AUM where Sharpe is maximized after costs
    max_aum: float  # AUM where Sharpe crosses decay threshold
    liquidity: LiquidityProfile


@dataclass
class CapacityResult:
    """Complete output of capacity analysis."""
    experiments: List[ExperimentCapacity]
    portfolio_optimal_aum: float
    portfolio_max_aum: float
    impact_grid: List[MarketImpactEstimate]
    summary: Dict[str, Any] = field(default_factory=dict)


# ── Core engine ───────────────────────────────────────────────────────────────

class CapacityAnalyzer:
    """Portfolio capacity analysis engine.

    Estimates how much AUM each experiment (and the blended portfolio) can
    manage before transaction costs erode alpha beyond acceptable levels.

    Args:
        experiment_returns: Dict mapping experiment ID to numpy array of daily returns.
        experiment_metadata: Dict mapping experiment ID to metadata dict.
            Expected keys:
              - avg_contracts: Average contracts per trade.
              - avg_spread_width: Average spread width in dollars.
              - trades_per_year: Estimated annual trade count.
              - avg_credit: Average credit received per spread.
        adv_contracts: Average daily volume in option contracts for the
            primary underlying (default: SPY options ~50k per strike).
        avg_open_interest: Average open interest per strike/expiry.
        bid_ask_spread_bps: Typical bid-ask spread cost in basis points.
        risk_free_rate: Annualized risk-free rate.
        aum_grid: List of AUM levels to evaluate.
    """

    def __init__(
        self,
        experiment_returns: Dict[str, np.ndarray],
        experiment_metadata: Dict[str, Dict[str, Any]],
        adv_contracts: int = 50_000,
        avg_open_interest: int = 100_000,
        bid_ask_spread_bps: float = DEFAULT_SPREAD_BPS,
        risk_free_rate: float = 0.045,
        aum_grid: Optional[List[float]] = None,
    ):
        if not experiment_returns:
            raise ValueError("experiment_returns must contain at least one experiment")

        self.experiment_ids = sorted(experiment_returns.keys())
        self.n_experiments = len(self.experiment_ids)

        # Validate return lengths
        lengths = {eid: len(r) for eid, r in experiment_returns.items()}
        unique = set(lengths.values())
        if len(unique) != 1:
            raise ValueError(f"All return arrays must have same length, got {lengths}")

        self.n_periods = list(unique)[0]
        if self.n_periods < 10:
            raise ValueError("Need at least 10 return periods for capacity analysis")

        self.returns = {eid: np.asarray(r, dtype=float) for eid, r in experiment_returns.items()}
        self.metadata = experiment_metadata
        self.adv_contracts = max(adv_contracts, 1)
        self.avg_open_interest = max(avg_open_interest, 1)
        self.bid_ask_spread_bps = bid_ask_spread_bps
        self.risk_free_rate = risk_free_rate
        self.aum_grid = aum_grid or DEFAULT_AUM_GRID

        logger.info(
            "CapacityAnalyzer: %d experiments, %d periods, ADV=%d contracts",
            self.n_experiments, self.n_periods, self.adv_contracts,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Market impact model
    # ─────────────────────────────────────────────────────────────────────────

    def estimate_market_impact(
        self,
        order_contracts: int,
        adv_override: Optional[int] = None,
        spread_bps_override: Optional[float] = None,
    ) -> MarketImpactEstimate:
        """Estimate market impact for a given order size.

        Uses the square-root model (Kyle 1985, Almgren-Chriss):
          impact_bps = IMPACT_COEFF × sqrt(order / ADV)

        Splits into temporary (reverts post-trade) and permanent components.
        Total cost = permanent_impact + spread_cost.

        Args:
            order_contracts: Number of option contracts in the order.
            adv_override: Override for ADV (default: self.adv_contracts).
            spread_bps_override: Override bid-ask spread cost.

        Returns:
            MarketImpactEstimate with cost breakdown.
        """
        adv = adv_override or self.adv_contracts
        spread_bps = spread_bps_override if spread_bps_override is not None else self.bid_ask_spread_bps

        order = max(order_contracts, 0)
        participation = order / max(adv, 1)

        # Square-root impact
        total_impact = IMPACT_COEFF * math.sqrt(participation)
        permanent = total_impact * PERMANENT_IMPACT_FRAC
        temporary = total_impact * (1.0 - PERMANENT_IMPACT_FRAC)

        return MarketImpactEstimate(
            order_contracts=order,
            adv_contracts=adv,
            participation_rate=round(participation, 6),
            temporary_impact_bps=round(temporary, 4),
            permanent_impact_bps=round(permanent, 4),
            total_impact_bps=round(total_impact, 4),
            spread_cost_bps=round(spread_bps, 4),
            total_cost_bps=round(permanent + spread_bps, 4),
        )

    def compute_impact_grid(
        self,
        contract_sizes: Optional[List[int]] = None,
    ) -> List[MarketImpactEstimate]:
        """Compute market impact across a range of order sizes.

        Args:
            contract_sizes: List of order sizes to evaluate.
                Defaults to [1, 5, 10, 25, 50, 100, 250, 500, 1000].

        Returns:
            List of MarketImpactEstimate for each size.
        """
        if contract_sizes is None:
            contract_sizes = [1, 5, 10, 25, 50, 100, 250, 500, 1000]

        return [self.estimate_market_impact(s) for s in contract_sizes]

    # ─────────────────────────────────────────────────────────────────────────
    # Liquidity analysis
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_liquidity(
        self,
        experiment_id: str,
        strike_volumes: Optional[Dict[float, int]] = None,
        strike_oi: Optional[Dict[float, int]] = None,
    ) -> LiquidityProfile:
        """Analyze liquidity for a given experiment.

        If strike-level data is provided, computes concentration metrics.
        Otherwise uses metadata and defaults to estimate liquidity.

        Args:
            experiment_id: Experiment identifier.
            strike_volumes: Optional dict mapping strike → daily volume.
            strike_oi: Optional dict mapping strike → open interest.

        Returns:
            LiquidityProfile with liquidity metrics and composite score.
        """
        meta = self.metadata.get(experiment_id, {})
        avg_contracts = meta.get("avg_contracts", 5)

        # Volume concentration
        if strike_volumes and len(strike_volumes) > 0:
            sorted_vols = sorted(strike_volumes.values(), reverse=True)
            total_vol = sum(sorted_vols)
            top3_vol = sum(sorted_vols[:3])
            vol_concentration = top3_vol / max(total_vol, 1)
            avg_vol = total_vol / len(strike_volumes)
        else:
            vol_concentration = 0.60  # typical for liquid options
            avg_vol = float(self.adv_contracts)

        # OI concentration
        if strike_oi and len(strike_oi) > 0:
            sorted_oi = sorted(strike_oi.values(), reverse=True)
            total_oi = sum(sorted_oi)
            top3_oi = sum(sorted_oi[:3])
            oi_concentration = top3_oi / max(total_oi, 1)
            avg_oi = total_oi / len(strike_oi)
        else:
            oi_concentration = 0.50
            avg_oi = float(self.avg_open_interest)

        # Composite liquidity score (0-100)
        # Higher is better: high volume, low spread, low concentration
        participation = avg_contracts / max(avg_vol, 1)
        vol_score = max(0, min(40, 40 * (1 - participation * 10)))
        spread_score = max(0, min(30, 30 * (1 - self.bid_ask_spread_bps / 50)))
        concentration_penalty = 15 * vol_concentration
        oi_score = max(0, min(15, 15 * min(avg_oi / 50_000, 1.0)))
        liquidity_score = vol_score + spread_score - concentration_penalty + oi_score

        return LiquidityProfile(
            experiment_id=experiment_id,
            avg_bid_ask_spread_bps=round(self.bid_ask_spread_bps, 2),
            avg_daily_volume=round(avg_vol, 0),
            avg_open_interest=round(avg_oi, 0),
            volume_concentration=round(vol_concentration, 4),
            oi_concentration=round(oi_concentration, 4),
            liquidity_score=round(max(0, min(100, liquidity_score)), 1),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Capacity curve
    # ─────────────────────────────────────────────────────────────────────────

    def _contracts_for_aum(
        self,
        aum: float,
        experiment_id: str,
    ) -> float:
        """Estimate contracts per trade at a given AUM level.

        Uses experiment metadata (avg_spread_width, risk_per_trade) to
        convert dollar AUM into expected contract count.
        """
        meta = self.metadata.get(experiment_id, {})
        spread_width = meta.get("avg_spread_width", 5.0)
        risk_per_trade = meta.get("risk_per_trade", 0.02)  # 2% default
        avg_credit = meta.get("avg_credit", 0.65)

        # Max loss per contract = (spread_width - credit) × 100
        max_loss_per_contract = (spread_width - avg_credit) * 100
        if max_loss_per_contract <= 0:
            max_loss_per_contract = spread_width * 100 * 0.5

        # Dollar risk budget at this AUM
        risk_budget = aum * risk_per_trade

        # Contracts = risk_budget / max_loss_per_contract
        contracts = risk_budget / max(max_loss_per_contract, 1)
        return max(contracts, 0.1)

    def compute_capacity_curve(
        self,
        experiment_id: str,
    ) -> Tuple[List[CapacityCurvePoint], float, float]:
        """Compute the capacity curve for one experiment.

        For each AUM level, estimates:
          - Contracts per trade → participation rate → market impact
          - Net return after impact costs → net Sharpe ratio
          - Sharpe retention (fraction of gross Sharpe preserved)

        Returns:
            (curve_points, optimal_aum, max_aum)
            optimal_aum: AUM that maximizes net Sharpe
            max_aum: AUM where Sharpe drops below ALPHA_DECAY_THRESHOLD × base
        """
        r = self.returns[experiment_id]
        meta = self.metadata.get(experiment_id, {})
        trades_per_year = meta.get("trades_per_year", 100)

        # Base performance (gross, before capacity constraints)
        ann_return = float(np.mean(r) * PERIODS_PER_YEAR)
        ann_vol = float(np.std(r) * math.sqrt(PERIODS_PER_YEAR))
        if ann_vol < 1e-10:
            ann_vol = 1e-10
        base_sharpe = (ann_return - self.risk_free_rate) / ann_vol

        curve = []
        best_sharpe = -999.0
        optimal_aum = self.aum_grid[0]
        max_aum = self.aum_grid[-1]
        decay_threshold = base_sharpe * ALPHA_DECAY_THRESHOLD

        for aum in self.aum_grid:
            contracts = self._contracts_for_aum(aum, experiment_id)
            impact = self.estimate_market_impact(int(max(contracts, 1)))

            # Annual cost = per-trade cost × trades per year
            # Convert bps to decimal: bps / 10000
            cost_per_trade = impact.total_cost_bps / 10_000
            annual_cost = cost_per_trade * trades_per_year

            # Net Sharpe = (gross_return - annual_cost - rf) / vol
            net_return = ann_return - annual_cost
            net_sharpe = (net_return - self.risk_free_rate) / ann_vol

            retention = net_sharpe / base_sharpe if abs(base_sharpe) > 1e-10 else 1.0

            point = CapacityCurvePoint(
                aum=aum,
                contracts_per_trade=round(contracts, 1),
                participation_rate=round(contracts / max(self.adv_contracts, 1), 6),
                total_cost_bps=round(impact.total_cost_bps, 2),
                net_sharpe=round(net_sharpe, 4),
                sharpe_retention=round(max(0, min(1, retention)), 4),
            )
            curve.append(point)

            if net_sharpe > best_sharpe:
                best_sharpe = net_sharpe
                optimal_aum = aum

        # Find max AUM (where Sharpe crosses decay threshold)
        for point in curve:
            if base_sharpe > 0 and point.net_sharpe < decay_threshold:
                max_aum = point.aum
                break
            max_aum = point.aum

        return curve, optimal_aum, max_aum

    # ─────────────────────────────────────────────────────────────────────────
    # Full analysis pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def run_all(self) -> CapacityResult:
        """Run the full capacity analysis pipeline.

        Returns:
            CapacityResult with per-experiment analysis, impact grid, and
            portfolio-level AUM recommendations.
        """
        experiments = []

        for eid in self.experiment_ids:
            r = self.returns[eid]
            ann_return = float(np.mean(r) * PERIODS_PER_YEAR)
            ann_vol = float(np.std(r) * math.sqrt(PERIODS_PER_YEAR))
            if ann_vol < 1e-10:
                ann_vol = 1e-10
            base_sharpe = (ann_return - self.risk_free_rate) / ann_vol

            curve, opt_aum, max_aum = self.compute_capacity_curve(eid)
            liquidity = self.analyze_liquidity(eid)

            experiments.append(ExperimentCapacity(
                experiment_id=eid,
                base_sharpe=round(base_sharpe, 4),
                base_annual_return=round(ann_return, 6),
                base_annual_vol=round(ann_vol, 6),
                capacity_curve=curve,
                optimal_aum=opt_aum,
                max_aum=max_aum,
                liquidity=liquidity,
            ))

        impact_grid = self.compute_impact_grid()

        # Portfolio-level: min of experiment max AUMs (bottleneck)
        portfolio_max = min(e.max_aum for e in experiments) if experiments else 0
        # Portfolio optimal: weighted average of experiment optimals
        portfolio_optimal = (
            sum(e.optimal_aum for e in experiments) / max(len(experiments), 1)
        )

        summary = {
            "n_experiments": self.n_experiments,
            "n_periods": self.n_periods,
            "adv_contracts": self.adv_contracts,
            "bid_ask_spread_bps": self.bid_ask_spread_bps,
            "portfolio_optimal_aum": portfolio_optimal,
            "portfolio_max_aum": portfolio_max,
            "bottleneck_experiment": (
                min(experiments, key=lambda e: e.max_aum).experiment_id
                if experiments else "N/A"
            ),
            "avg_liquidity_score": round(
                sum(e.liquidity.liquidity_score for e in experiments) / max(len(experiments), 1), 1
            ),
        }

        logger.info(
            "Capacity analysis complete: optimal=$%s, max=$%s, bottleneck=%s",
            f"{portfolio_optimal:,.0f}", f"{portfolio_max:,.0f}",
            summary["bottleneck_experiment"],
        )

        return CapacityResult(
            experiments=experiments,
            portfolio_optimal_aum=portfolio_optimal,
            portfolio_max_aum=portfolio_max,
            impact_grid=impact_grid,
            summary=summary,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # HTML report generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_html(self, result: CapacityResult) -> str:
        """Generate a self-contained HTML report.

        Includes:
          - Capacity curves per experiment (SVG line chart)
          - Market impact chart (SVG bar chart)
          - Liquidity heatmap (HTML table with color coding)
          - AUM recommendations table
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        s = result.summary

        # ── AUM recommendations table ────────────────────────────────────
        rec_rows = ""
        for exp in result.experiments:
            rec_rows += (
                f"<tr><td>{exp.experiment_id}</td>"
                f"<td>{exp.base_sharpe:.2f}</td>"
                f"<td>${exp.optimal_aum:,.0f}</td>"
                f"<td>${exp.max_aum:,.0f}</td>"
                f"<td>{exp.liquidity.liquidity_score:.0f}/100</td></tr>\n"
            )

        # ── Capacity curves (SVG) ────────────────────────────────────────
        capacity_svg = self._render_capacity_curves(result.experiments)

        # ── Market impact chart (SVG) ────────────────────────────────────
        impact_svg = self._render_impact_chart(result.impact_grid)

        # ── Liquidity heatmap ────────────────────────────────────────────
        liq_rows = ""
        for exp in result.experiments:
            lq = exp.liquidity
            score_cls = "good" if lq.liquidity_score >= 60 else ("warn" if lq.liquidity_score >= 30 else "bad")
            liq_rows += (
                f"<tr><td>{lq.experiment_id}</td>"
                f"<td>{lq.avg_bid_ask_spread_bps:.1f}</td>"
                f"<td>{lq.avg_daily_volume:,.0f}</td>"
                f"<td>{lq.avg_open_interest:,.0f}</td>"
                f"<td>{lq.volume_concentration:.0%}</td>"
                f"<td class='{score_cls}'>{lq.liquidity_score:.0f}</td></tr>\n"
            )

        # ── Capacity curve data table ────────────────────────────────────
        curve_rows = ""
        for exp in result.experiments:
            for pt in exp.capacity_curve:
                ret_cls = "good" if pt.sharpe_retention >= 0.8 else ("warn" if pt.sharpe_retention >= 0.5 else "bad")
                curve_rows += (
                    f"<tr><td>{exp.experiment_id}</td>"
                    f"<td>${pt.aum:,.0f}</td>"
                    f"<td>{pt.contracts_per_trade:.0f}</td>"
                    f"<td>{pt.participation_rate:.4%}</td>"
                    f"<td>{pt.total_cost_bps:.1f}</td>"
                    f"<td>{pt.net_sharpe:.2f}</td>"
                    f"<td class='{ret_cls}'>{pt.sharpe_retention:.0%}</td></tr>\n"
                )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Portfolio Capacity Analysis</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 130px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .good {{ color: #16a34a; }}
  .bad {{ color: #dc2626; }}
  .warn {{ color: #d97706; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .chart-container {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
                      padding: 1.5em; margin: 1.5em 0; overflow-x: auto; }}
  .section {{ margin-bottom: 2.5em; }}
  .legend {{ display: flex; gap: 1.5em; margin-top: 0.5em; font-size: 0.85em; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.3em; }}
  .legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>Portfolio Capacity Analysis</h1>
<div class="meta">Generated: {now} | {s['n_experiments']} experiments | {s['n_periods']} periods | ADV: {s['adv_contracts']:,} contracts</div>

<div class="kpi-row">
  <div class="kpi">
    <div class="value">${s['portfolio_optimal_aum']:,.0f}</div>
    <div class="label">Optimal Portfolio AUM</div>
  </div>
  <div class="kpi">
    <div class="value">${s['portfolio_max_aum']:,.0f}</div>
    <div class="label">Max AUM (50% Decay)</div>
  </div>
  <div class="kpi">
    <div class="value">{s['bottleneck_experiment']}</div>
    <div class="label">Bottleneck</div>
  </div>
  <div class="kpi">
    <div class="value">{s['avg_liquidity_score']:.0f}/100</div>
    <div class="label">Avg Liquidity Score</div>
  </div>
</div>

<div class="section">
<h2>AUM Recommendations</h2>
<table>
<thead><tr><th>Experiment</th><th>Base Sharpe</th><th>Optimal AUM</th><th>Max AUM</th><th>Liquidity</th></tr></thead>
<tbody>{rec_rows}</tbody>
</table>
</div>

<div class="section">
<h2>Capacity Curves</h2>
<div class="chart-container">
{capacity_svg}
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background:#3b82f6"></div>{result.experiments[0].experiment_id if result.experiments else ''}</div>
  {"".join(f'<div class="legend-item"><div class="legend-swatch" style="background:{c}"></div>{e.experiment_id}</div>' for e, c in zip(result.experiments[1:], ["#f59e0b", "#8b5cf6", "#10b981", "#ef4444", "#6366f1"][:len(result.experiments)-1]))}
</div>
</div>
</div>

<div class="section">
<h2>Market Impact</h2>
<div class="chart-container">
{impact_svg}
</div>
</div>

<div class="section">
<h2>Liquidity Profile</h2>
<table>
<thead><tr><th>Experiment</th><th>Spread (bps)</th><th>Avg Daily Vol</th><th>Avg OI</th><th>Vol Concentration</th><th>Score</th></tr></thead>
<tbody>{liq_rows}</tbody>
</table>
</div>

<div class="section">
<h2>Capacity Curve Detail</h2>
<table>
<thead><tr><th>Experiment</th><th>AUM</th><th>Contracts</th><th>Participation</th><th>Cost (bps)</th><th>Net Sharpe</th><th>Retention</th></tr></thead>
<tbody>{curve_rows}</tbody>
</table>
</div>

</body>
</html>"""
        return html

    # ─────────────────────────────────────────────────────────────────────────
    # SVG chart helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _render_capacity_curves(experiments: List[ExperimentCapacity]) -> str:
        """Render SVG line chart of Sharpe retention vs AUM."""
        if not experiments:
            return "<p>No capacity data</p>"

        chart_w, chart_h = 600, 300
        pad_l, pad_b, pad_t, pad_r = 60, 40, 20, 20
        plot_w = chart_w - pad_l - pad_r
        plot_h = chart_h - pad_t - pad_b

        # Collect all AUM values for x-axis
        all_aum = set()
        for exp in experiments:
            for pt in exp.capacity_curve:
                all_aum.add(pt.aum)
        aum_sorted = sorted(all_aum)
        if len(aum_sorted) < 2:
            return "<p>Insufficient data for chart</p>"

        min_aum = math.log10(max(aum_sorted[0], 1))
        max_aum = math.log10(max(aum_sorted[-1], 2))
        aum_range = max(max_aum - min_aum, 1e-6)

        colors = ["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981", "#ef4444", "#6366f1"]

        lines_svg = ""
        for idx, exp in enumerate(experiments):
            color = colors[idx % len(colors)]
            points = []
            for pt in exp.capacity_curve:
                x = pad_l + ((math.log10(max(pt.aum, 1)) - min_aum) / aum_range) * plot_w
                y = pad_t + (1.0 - pt.sharpe_retention) * plot_h
                points.append(f"{x:.1f},{y:.1f}")

            if points:
                lines_svg += (
                    f'<polyline points="{" ".join(points)}" '
                    f'fill="none" stroke="{color}" stroke-width="2.5" '
                    f'stroke-linejoin="round"/>\n'
                )

        # Decay threshold line
        y_thresh = pad_t + (1.0 - ALPHA_DECAY_THRESHOLD) * plot_h
        lines_svg += (
            f'<line x1="{pad_l}" y1="{y_thresh:.1f}" x2="{chart_w - pad_r}" '
            f'y2="{y_thresh:.1f}" stroke="#dc2626" stroke-width="1" '
            f'stroke-dasharray="5,3"/>\n'
            f'<text x="{chart_w - pad_r + 2}" y="{y_thresh + 4}" '
            f'font-size="10" fill="#dc2626">50% decay</text>\n'
        )

        # Y-axis labels
        y_labels = ""
        for pct in [0, 25, 50, 75, 100]:
            y = pad_t + (1.0 - pct / 100) * plot_h
            y_labels += (
                f'<text x="{pad_l - 8}" y="{y + 4}" text-anchor="end" '
                f'font-size="10" fill="#64748b">{pct}%</text>\n'
                f'<line x1="{pad_l}" y1="{y:.1f}" x2="{chart_w - pad_r}" '
                f'y2="{y:.1f}" stroke="#e2e8f0" stroke-width="0.5"/>\n'
            )

        # X-axis labels (log-scale AUM)
        x_labels = ""
        for aum in aum_sorted[::max(len(aum_sorted) // 5, 1)]:
            x = pad_l + ((math.log10(max(aum, 1)) - min_aum) / aum_range) * plot_w
            label = f"${aum / 1e6:.1f}M" if aum >= 1e6 else f"${aum / 1e3:.0f}K"
            x_labels += (
                f'<text x="{x:.1f}" y="{chart_h - 5}" text-anchor="middle" '
                f'font-size="10" fill="#64748b">{label}</text>\n'
            )

        # Axis labels
        axis_labels = (
            f'<text x="{pad_l + plot_w / 2}" y="{chart_h + 15}" text-anchor="middle" '
            f'font-size="11" fill="#334155">AUM</text>\n'
            f'<text x="12" y="{pad_t + plot_h / 2}" text-anchor="middle" '
            f'font-size="11" fill="#334155" transform="rotate(-90,12,{pad_t + plot_h / 2})">'
            f'Sharpe Retention</text>\n'
        )

        return (
            f'<svg width="{chart_w + 60}" height="{chart_h + 20}" '
            f'xmlns="http://www.w3.org/2000/svg">\n'
            f'{y_labels}{x_labels}{axis_labels}{lines_svg}</svg>'
        )

    @staticmethod
    def _render_impact_chart(grid: List[MarketImpactEstimate]) -> str:
        """Render SVG bar chart of market impact vs order size."""
        if not grid:
            return "<p>No impact data</p>"

        n = len(grid)
        bar_w = 50
        gap = 15
        chart_h = 220
        label_h = 50
        pad_l = 50
        total_w = pad_l + n * (bar_w + gap) + gap

        max_bps = max(g.total_cost_bps for g in grid) if grid else 1.0
        max_bps = max(max_bps, 1.0)

        bars = ""
        for i, g in enumerate(grid):
            x = pad_l + gap + i * (bar_w + gap)

            # Stacked: spread (blue) + impact (orange)
            spread_h = (g.spread_cost_bps / max_bps) * (chart_h - 20)
            impact_h = (g.permanent_impact_bps / max_bps) * (chart_h - 20)

            # Spread bar (bottom)
            y_spread = chart_h - spread_h
            bars += (
                f'<rect x="{x}" y="{y_spread:.1f}" width="{bar_w}" '
                f'height="{spread_h:.1f}" fill="#3b82f6" rx="2"/>\n'
            )
            # Impact bar (on top of spread)
            y_impact = y_spread - impact_h
            bars += (
                f'<rect x="{x}" y="{y_impact:.1f}" width="{bar_w}" '
                f'height="{impact_h:.1f}" fill="#f59e0b" rx="2"/>\n'
            )

            # Value label
            bars += (
                f'<text x="{x + bar_w / 2}" y="{y_impact - 5:.1f}" text-anchor="middle" '
                f'font-size="10" fill="#334155">{g.total_cost_bps:.0f}</text>\n'
            )
            # X-axis label
            bars += (
                f'<text x="{x + bar_w / 2}" y="{chart_h + 16}" text-anchor="middle" '
                f'font-size="10" fill="#64748b">{g.order_contracts}</text>\n'
            )

        # X-axis title
        bars += (
            f'<text x="{total_w / 2}" y="{chart_h + 38}" text-anchor="middle" '
            f'font-size="11" fill="#334155">Order Size (contracts)</text>\n'
        )

        # Legend
        bars += (
            f'<rect x="{total_w - 180}" y="5" width="12" height="12" fill="#3b82f6" rx="2"/>'
            f'<text x="{total_w - 164}" y="15" font-size="10" fill="#334155">Spread</text>'
            f'<rect x="{total_w - 110}" y="5" width="12" height="12" fill="#f59e0b" rx="2"/>'
            f'<text x="{total_w - 94}" y="15" font-size="10" fill="#334155">Impact</text>\n'
        )

        return (
            f'<svg width="{total_w}" height="{chart_h + label_h}" '
            f'xmlns="http://www.w3.org/2000/svg">\n'
            f'<line x1="{pad_l}" y1="0" x2="{pad_l}" y2="{chart_h}" '
            f'stroke="#cbd5e1" stroke-width="1"/>\n'
            f'<line x1="{pad_l}" y1="{chart_h}" x2="{total_w}" y2="{chart_h}" '
            f'stroke="#cbd5e1" stroke-width="1"/>\n'
            f'{bars}</svg>'
        )


# ── Convenience: generate and save report ─────────────────────────────────────

def generate_report(
    experiment_returns: Dict[str, np.ndarray],
    experiment_metadata: Dict[str, Dict[str, Any]],
    output_path: str = "reports/capacity.html",
    adv_contracts: int = 50_000,
    **kwargs,
) -> CapacityResult:
    """One-call convenience: analyze capacity and write HTML report.

    Args:
        experiment_returns: Experiment returns dict.
        experiment_metadata: Per-experiment metadata dict.
        output_path: Where to write the HTML report.
        adv_contracts: Average daily volume in contracts.
        **kwargs: Passed to CapacityAnalyzer constructor.

    Returns:
        CapacityResult (report is also written to disk).
    """
    analyzer = CapacityAnalyzer(
        experiment_returns=experiment_returns,
        experiment_metadata=experiment_metadata,
        adv_contracts=adv_contracts,
        **kwargs,
    )
    result = analyzer.run_all()
    html = analyzer.generate_html(result)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    logger.info("Capacity report written to %s", output_path)
    return result
