"""Risk stress testing engine for credit spread portfolios.

Provides historical scenario replay, hypothetical stress tests,
reverse stress testing, liquidity and correlation stress analysis.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StressScenario:
    """A configurable stress scenario."""

    name: str
    equity_shock: float  # e.g. -0.34 for -34%
    vol_shock: float  # absolute VIX point change
    rate_shock: float  # absolute rate change in bps
    correlation_shock: float  # shift toward 1.0 (0-1 scale)
    duration: int  # calendar days


@dataclass
class GreeksExposure:
    """Portfolio Greeks snapshot."""

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


@dataclass
class PortfolioSnapshot:
    """Minimal portfolio state for stress testing."""

    market_value: float
    maintenance_margin_pct: float = 0.20
    greeks: GreeksExposure = field(default_factory=GreeksExposure)
    position_notionals: Optional[List[float]] = None
    position_vols: Optional[List[float]] = None
    bid_ask_spread: float = 0.05  # as fraction of notional


@dataclass
class ScenarioResult:
    """Result for a single stress scenario."""

    scenario_name: str
    pnl: float
    pnl_pct: float
    margin_impact: float
    greeks_pnl: float
    max_drawdown: float
    daily_pnls: List[float] = field(default_factory=list)


@dataclass
class ReverseStressResult:
    """Result of reverse stress test."""

    target_loss_pct: float
    required_equity_shock: float
    achieved_loss_pct: float
    iterations: int


@dataclass
class LiquidityStressResult:
    """Result of liquidity stress test."""

    spread_multiplier: float
    unwind_cost: float
    unwind_cost_pct: float


@dataclass
class CorrelationStressResult:
    """Result of correlation stress test."""

    normal_portfolio_vol: float
    stressed_portfolio_vol: float
    vol_ratio: float


@dataclass
class RiskStressResult:
    """Aggregate stress test results."""

    scenario_results: List[ScenarioResult]
    reverse_stress: Optional[ReverseStressResult]
    liquidity_stress: Optional[LiquidityStressResult]
    correlation_stress: Optional[CorrelationStressResult]
    worst_case: Optional[ScenarioResult]
    generated_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )


# ---------------------------------------------------------------------------
# Historical scenario definitions
# ---------------------------------------------------------------------------

@dataclass
class HistoricalScenario:
    """A historical market crisis scenario."""

    name: str
    total_shock: float  # e.g. -0.57
    duration_days: int
    vix_start: float
    vix_peak: float
    daily_shocks: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.daily_shocks:
            self.daily_shocks = self._generate_daily_shocks()

    def _generate_daily_shocks(self) -> List[float]:
        """Generate realistic daily shocks that compound to total_shock."""
        if self.duration_days <= 0:
            return [self.total_shock]
        n = self.duration_days
        # Use a skewed distribution: front-load shocks for crashes
        rng = np.random.RandomState(hash(self.name) % (2**31))
        weights = np.exp(-np.linspace(0, 2, n))
        weights /= weights.sum()
        # Target: product of (1 + r_i) = 1 + total_shock
        target_product = 1.0 + self.total_shock
        # Approximate: daily return such that compound = target
        avg_daily = target_product ** (1.0 / n) - 1.0
        # Distribute with some noise
        raw = avg_daily * (weights * n)
        noise = rng.normal(0, abs(avg_daily) * 0.3, n)
        raw = raw + noise
        # Rescale so compound product matches target
        current_product = np.prod(1.0 + raw)
        if current_product != 0 and not np.isnan(current_product):
            adjustment = (target_product / current_product) ** (1.0 / n)
            daily = [(1.0 + r) * adjustment - 1.0 for r in raw]
        else:
            daily = [avg_daily] * n
        return daily


def _build_historical_scenarios() -> Dict[str, HistoricalScenario]:
    """Create the four canonical historical scenarios."""
    return {
        "2008_GFC": HistoricalScenario(
            name="2008_GFC",
            total_shock=-0.57,
            duration_days=350,
            vix_start=22.0,
            vix_peak=80.0,
        ),
        "COVID_2020": HistoricalScenario(
            name="COVID_2020",
            total_shock=-0.34,
            duration_days=23,
            vix_start=14.0,
            vix_peak=82.0,
        ),
        "2022_RATE_HIKES": HistoricalScenario(
            name="2022_RATE_HIKES",
            total_shock=-0.25,
            duration_days=190,
            vix_start=17.0,
            vix_peak=36.0,
        ),
        "FLASH_CRASH": HistoricalScenario(
            name="FLASH_CRASH",
            total_shock=-0.10,
            duration_days=1,
            vix_start=16.0,
            vix_peak=40.0,
        ),
    }


HISTORICAL_SCENARIOS = _build_historical_scenarios()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RiskStressEngine:
    """Comprehensive stress testing engine for credit spread portfolios."""

    def __init__(
        self,
        portfolio: PortfolioSnapshot,
        custom_scenarios: Optional[List[StressScenario]] = None,
    ) -> None:
        self.portfolio = portfolio
        self.custom_scenarios = custom_scenarios or []
        self.historical_scenarios = HISTORICAL_SCENARIOS

    # ------------------------------------------------------------------
    # Per-scenario compute
    # ------------------------------------------------------------------

    def compute_scenario_pnl(
        self, equity_shock: float, vol_shock: float, duration: int
    ) -> ScenarioResult:
        """Compute P&L impact from a simple shock to the portfolio."""
        mv = self.portfolio.market_value
        pnl = mv * equity_shock
        pnl_pct = equity_shock
        stress_value = abs(pnl)
        margin_impact = stress_value * self.portfolio.maintenance_margin_pct
        greeks_pnl = self._compute_greeks_impact(
            equity_shock, vol_shock, duration
        )
        return ScenarioResult(
            scenario_name="",
            pnl=pnl,
            pnl_pct=pnl_pct,
            margin_impact=margin_impact,
            greeks_pnl=greeks_pnl,
            max_drawdown=abs(pnl_pct),
        )

    def _compute_greeks_impact(
        self, equity_shock: float, vol_shock: float, days: int
    ) -> float:
        """delta*move + 0.5*gamma*move^2 + vega*vol_change + theta*days."""
        g = self.portfolio.greeks
        move = equity_shock * self.portfolio.market_value
        result = (
            g.delta * move
            + 0.5 * g.gamma * move ** 2
            + g.vega * vol_shock
            + g.theta * days
        )
        return result

    def run_historical_scenario(
        self, scenario: HistoricalScenario
    ) -> ScenarioResult:
        """Replay a historical scenario through the portfolio."""
        mv = self.portfolio.market_value
        cumulative_value = mv
        peak_value = mv
        max_dd = 0.0
        daily_pnls: List[float] = []

        vol_change = scenario.vix_peak - scenario.vix_start

        for daily_ret in scenario.daily_shocks:
            day_pnl = cumulative_value * daily_ret
            daily_pnls.append(day_pnl)
            cumulative_value += day_pnl
            peak_value = max(peak_value, cumulative_value)
            dd = (peak_value - cumulative_value) / peak_value if peak_value > 0 else 0.0
            max_dd = max(max_dd, dd)

        total_pnl = cumulative_value - mv
        pnl_pct = total_pnl / mv if mv != 0 else 0.0

        stress_value = abs(total_pnl)
        margin_impact = stress_value * self.portfolio.maintenance_margin_pct

        greeks_pnl = self._compute_greeks_impact(
            pnl_pct, vol_change, scenario.duration_days
        )

        return ScenarioResult(
            scenario_name=scenario.name,
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            margin_impact=margin_impact,
            greeks_pnl=greeks_pnl,
            max_drawdown=max_dd,
            daily_pnls=daily_pnls,
        )

    def run_custom_scenario(self, scenario: StressScenario) -> ScenarioResult:
        """Run a hypothetical / configurable stress scenario."""
        mv = self.portfolio.market_value
        pnl = mv * scenario.equity_shock
        pnl_pct = scenario.equity_shock

        stress_value = abs(pnl)
        margin_impact = stress_value * self.portfolio.maintenance_margin_pct

        greeks_pnl = self._compute_greeks_impact(
            scenario.equity_shock, scenario.vol_shock, scenario.duration
        )

        return ScenarioResult(
            scenario_name=scenario.name,
            pnl=pnl,
            pnl_pct=pnl_pct,
            margin_impact=margin_impact,
            greeks_pnl=greeks_pnl,
            max_drawdown=abs(pnl_pct),
        )

    # ------------------------------------------------------------------
    # Reverse stress test
    # ------------------------------------------------------------------

    def reverse_stress_test(
        self,
        target_loss_pct: float = 0.30,
        tolerance: float = 0.001,
        max_iterations: int = 100,
    ) -> ReverseStressResult:
        """Binary search for the equity shock that causes *target_loss_pct*."""
        lo, hi = 0.0, -1.0  # search in negative shock space
        iterations = 0

        for _ in range(max_iterations):
            iterations += 1
            mid = (lo + hi) / 2.0
            result = self.compute_scenario_pnl(
                equity_shock=mid, vol_shock=0.0, duration=1
            )
            achieved = abs(result.pnl_pct)
            if abs(achieved - target_loss_pct) < tolerance:
                break
            if achieved < target_loss_pct:
                lo = mid  # need a bigger (more negative) shock
            else:
                hi = mid  # overshot

        return ReverseStressResult(
            target_loss_pct=target_loss_pct,
            required_equity_shock=mid,
            achieved_loss_pct=abs(result.pnl_pct),
            iterations=iterations,
        )

    # ------------------------------------------------------------------
    # Liquidity stress
    # ------------------------------------------------------------------

    def compute_liquidity_stress(
        self, spread_multiplier: float = 5.0
    ) -> LiquidityStressResult:
        """Cost of emergency unwind if bid-ask spreads widen by *spread_multiplier*."""
        mv = self.portfolio.market_value
        base_spread = self.portfolio.bid_ask_spread
        stressed_spread = base_spread * spread_multiplier
        # Unwind cost = half the spread (crossing the spread) * notional
        unwind_cost = abs(mv) * stressed_spread * 0.5
        unwind_pct = unwind_cost / abs(mv) if mv != 0 else 0.0
        return LiquidityStressResult(
            spread_multiplier=spread_multiplier,
            unwind_cost=unwind_cost,
            unwind_cost_pct=unwind_pct,
        )

    # ------------------------------------------------------------------
    # Correlation stress
    # ------------------------------------------------------------------

    def compute_correlation_stress(self) -> CorrelationStressResult:
        """Portfolio vol when all pairwise correlations go to 1.0.

        Normal case: sqrt(sum of (w_i * sigma_i)^2 + cross terms)
        Stressed case (corr=1): sum of |w_i * sigma_i|  (linear sum)

        With equal weights and given individual vols, the ratio shows
        diversification benefit lost.
        """
        vols = self.portfolio.position_vols
        if not vols or len(vols) == 0:
            return CorrelationStressResult(
                normal_portfolio_vol=0.0,
                stressed_portfolio_vol=0.0,
                vol_ratio=1.0,
            )

        vols_arr = np.array(vols, dtype=float)
        n = len(vols_arr)
        weights = np.ones(n) / n

        # Stressed (corr = 1): portfolio vol = sum of |w_i * sigma_i|
        weighted_vols = np.abs(weights * vols_arr)
        stressed_vol = float(np.sum(weighted_vols))

        # Normal (assume zero correlation for max diversification benefit):
        # portfolio vol = sqrt(sum( (w_i * sigma_i)^2 ))
        normal_vol = float(np.sqrt(np.sum(weighted_vols ** 2)))

        ratio = stressed_vol / normal_vol if normal_vol > 0 else 1.0

        return CorrelationStressResult(
            normal_portfolio_vol=normal_vol,
            stressed_portfolio_vol=stressed_vol,
            vol_ratio=ratio,
        )

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run_all(
        self,
        target_loss_pct: float = 0.30,
        spread_multiplier: float = 5.0,
    ) -> RiskStressResult:
        """Execute all stress tests and return aggregate result."""
        scenario_results: List[ScenarioResult] = []

        # Historical scenarios
        for scenario in self.historical_scenarios.values():
            result = self.run_historical_scenario(scenario)
            scenario_results.append(result)

        # Custom scenarios
        for scenario in self.custom_scenarios:
            result = self.run_custom_scenario(scenario)
            scenario_results.append(result)

        # Reverse stress
        reverse = self.reverse_stress_test(target_loss_pct=target_loss_pct)

        # Liquidity stress
        liquidity = self.compute_liquidity_stress(
            spread_multiplier=spread_multiplier
        )

        # Correlation stress
        correlation = self.compute_correlation_stress()

        # Worst case
        worst = None
        if scenario_results:
            worst = min(scenario_results, key=lambda r: r.pnl)

        return RiskStressResult(
            scenario_results=scenario_results,
            reverse_stress=reverse,
            liquidity_stress=liquidity,
            correlation_stress=correlation,
            worst_case=worst,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(self, result: Optional[RiskStressResult] = None) -> str:
        """Generate an HTML report with scenario table, P&L waterfall SVG, and worst-case dashboard."""
        if result is None:
            result = self.run_all()

        html_parts: List[str] = []
        html_parts.append("<!DOCTYPE html><html><head>")
        html_parts.append("<meta charset='utf-8'>")
        html_parts.append("<title>Risk Stress Report</title>")
        html_parts.append("<style>")
        html_parts.append(
            "body{font-family:Arial,sans-serif;margin:20px;}"
            "table{border-collapse:collapse;width:100%;margin:16px 0;}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:right;}"
            "th{background:#f4f4f4;}"
            ".negative{color:#c0392b;}"
            ".positive{color:#27ae60;}"
            ".dashboard{background:#fdf2f2;border:2px solid #c0392b;"
            "padding:16px;border-radius:8px;margin:16px 0;}"
        )
        html_parts.append("</style></head><body>")
        html_parts.append(f"<h1>Risk Stress Test Report</h1>")
        html_parts.append(
            f"<p>Generated: {result.generated_at}</p>"
        )

        # --- Scenario comparison table ---
        html_parts.append("<h2>Scenario Comparison</h2>")
        html_parts.append("<table>")
        html_parts.append(
            "<tr><th>Scenario</th><th>P&amp;L</th><th>P&amp;L %</th>"
            "<th>Max Drawdown</th><th>Margin Impact</th>"
            "<th>Greeks P&amp;L</th></tr>"
        )
        for sr in result.scenario_results:
            css = "negative" if sr.pnl < 0 else "positive"
            html_parts.append(
                f"<tr>"
                f"<td style='text-align:left'>{sr.scenario_name}</td>"
                f"<td class='{css}'>{sr.pnl:,.0f}</td>"
                f"<td class='{css}'>{sr.pnl_pct:.1%}</td>"
                f"<td>{sr.max_drawdown:.1%}</td>"
                f"<td>{sr.margin_impact:,.0f}</td>"
                f"<td>{sr.greeks_pnl:,.0f}</td>"
                f"</tr>"
            )
        html_parts.append("</table>")

        # --- P&L waterfall (SVG bars) ---
        html_parts.append("<h2>P&amp;L Waterfall</h2>")
        if result.scenario_results:
            svg_width = 600
            svg_height = 300
            n = len(result.scenario_results)
            bar_width = max(30, (svg_width - 40) // n - 10)
            max_abs_pnl = max(abs(sr.pnl) for sr in result.scenario_results)
            if max_abs_pnl == 0:
                max_abs_pnl = 1.0
            scale = (svg_height - 60) / (2 * max_abs_pnl)
            mid_y = svg_height / 2

            html_parts.append(
                f"<svg width='{svg_width}' height='{svg_height}' "
                f"xmlns='http://www.w3.org/2000/svg'>"
            )
            # baseline
            html_parts.append(
                f"<line x1='20' y1='{mid_y}' x2='{svg_width - 20}' "
                f"y2='{mid_y}' stroke='#333' stroke-width='1'/>"
            )
            for i, sr in enumerate(result.scenario_results):
                x = 30 + i * (bar_width + 10)
                bar_h = abs(sr.pnl) * scale
                color = "#c0392b" if sr.pnl < 0 else "#27ae60"
                if sr.pnl < 0:
                    y = mid_y
                else:
                    y = mid_y - bar_h
                html_parts.append(
                    f"<rect x='{x}' y='{y}' width='{bar_width}' "
                    f"height='{bar_h}' fill='{color}'/>"
                )
                # Label
                label_y = mid_y + bar_h + 14 if sr.pnl < 0 else mid_y - bar_h - 4
                short_name = sr.scenario_name[:12]
                html_parts.append(
                    f"<text x='{x + bar_width / 2}' y='{label_y}' "
                    f"font-size='10' text-anchor='middle'>{short_name}</text>"
                )
            html_parts.append("</svg>")

        # --- Worst-case dashboard ---
        html_parts.append("<h2>Worst-Case Dashboard</h2>")
        if result.worst_case:
            wc = result.worst_case
            html_parts.append("<div class='dashboard'>")
            html_parts.append(
                f"<h3>Worst scenario: {wc.scenario_name}</h3>"
                f"<p>P&amp;L: {wc.pnl:,.0f} ({wc.pnl_pct:.1%})</p>"
                f"<p>Max Drawdown: {wc.max_drawdown:.1%}</p>"
                f"<p>Margin Impact: {wc.margin_impact:,.0f}</p>"
            )
            html_parts.append("</div>")

        # --- Reverse stress ---
        if result.reverse_stress:
            rs = result.reverse_stress
            html_parts.append("<h2>Reverse Stress Test</h2>")
            html_parts.append(
                f"<p>Target loss: {rs.target_loss_pct:.1%}<br>"
                f"Required equity shock: {rs.required_equity_shock:.2%}<br>"
                f"Achieved loss: {rs.achieved_loss_pct:.2%}<br>"
                f"Iterations: {rs.iterations}</p>"
            )

        # --- Liquidity stress ---
        if result.liquidity_stress:
            ls = result.liquidity_stress
            html_parts.append("<h2>Liquidity Stress</h2>")
            html_parts.append(
                f"<p>Spread multiplier: {ls.spread_multiplier:.0f}x<br>"
                f"Unwind cost: {ls.unwind_cost:,.0f} "
                f"({ls.unwind_cost_pct:.2%})</p>"
            )

        # --- Correlation stress ---
        if result.correlation_stress:
            cs = result.correlation_stress
            html_parts.append("<h2>Correlation Stress</h2>")
            html_parts.append(
                f"<p>Normal portfolio vol: {cs.normal_portfolio_vol:.4f}<br>"
                f"Stressed portfolio vol (corr=1): "
                f"{cs.stressed_portfolio_vol:.4f}<br>"
                f"Vol ratio: {cs.vol_ratio:.2f}x</p>"
            )

        html_parts.append("</body></html>")
        return "\n".join(html_parts)
