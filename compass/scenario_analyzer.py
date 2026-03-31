"""What-if scenario analysis engine – historical and custom scenario replay with
portfolio impact estimation, stress P&L with Greeks, tail risk contribution,
probability weighting, and recovery time estimation.

Provides:
  1. Historical scenario replay (2008 GFC, 2020 COVID, 2022 rate hikes, flash crashes)
  2. Custom scenario definition (shock size, duration, correlation)
  3. Portfolio impact estimation under each scenario
  4. Stress P&L with delta/vega/theta decomposition
  5. Tail risk contribution per position
  6. Scenario probability weighting
  7. Recovery time estimation
  8. HTML report with comparison table, P&L chart, worst-case dashboard
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

# ── Credit spread beta to market ────────────────────────────────────────────
SPREAD_BETA = 1.5


# ── Built-in historical scenarios ───────────────────────────────────────────
def _build_shock_path(total_return: float, n_days: int, seed: int = 42) -> List[float]:
    """Generate a realistic crash path with front-loaded losses."""
    if n_days <= 0:
        return []
    if n_days == 1:
        return [total_return]

    rng = np.random.RandomState(seed)
    log_ret = math.log(1 + total_return)
    weights = np.exp(-np.arange(n_days) / max(n_days / 3, 1))
    weights /= weights.sum()
    raw = weights * log_ret
    noise = rng.randn(n_days) * abs(log_ret) * 0.05
    raw += noise
    raw *= log_ret / raw.sum() if abs(raw.sum()) > 1e-12 else 1.0
    return [float(math.exp(r) - 1) for r in raw]


HISTORICAL_SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "2008 GFC",
        "description": "S&P 500 fell ~57% over ~17 months",
        "daily_shocks": _build_shock_path(-0.57, 350, seed=1),
        "vix_start": 20.0,
        "vix_peak": 80.0,
        "probability": 0.02,
    },
    {
        "name": "COVID Crash (2020)",
        "description": "S&P 500 fell ~34% in 23 trading days",
        "daily_shocks": _build_shock_path(-0.34, 23, seed=2),
        "vix_start": 15.0,
        "vix_peak": 82.0,
        "probability": 0.03,
    },
    {
        "name": "2022 Rate Hikes",
        "description": "S&P 500 fell ~25% over ~9 months",
        "daily_shocks": _build_shock_path(-0.25, 190, seed=3),
        "vix_start": 17.0,
        "vix_peak": 36.0,
        "probability": 0.08,
    },
    {
        "name": "Flash Crash",
        "description": "Sudden 10% drawdown in a single session",
        "daily_shocks": _build_shock_path(-0.10, 1, seed=4),
        "vix_start": 15.0,
        "vix_peak": 65.0,
        "probability": 0.05,
    },
    {
        "name": "VIX Spike",
        "description": "VIX quadruples over 5 days",
        "daily_shocks": _build_shock_path(-0.15, 5, seed=5),
        "vix_start": 15.0,
        "vix_peak": 65.0,
        "probability": 0.06,
    },
]


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class ScenarioDef:
    """Definition of a scenario (historical or custom)."""
    name: str
    description: str
    daily_shocks: List[float]
    vix_start: float = 15.0
    vix_peak: float = 40.0
    probability: float = 0.05
    correlation_shock: float = 0.0  # increase in cross-asset correlation


@dataclass
class GreeksPnL:
    """Stress P&L decomposed by Greeks."""
    delta_pnl: float = 0.0
    vega_pnl: float = 0.0
    theta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    residual_pnl: float = 0.0
    total_pnl: float = 0.0


@dataclass
class PositionContribution:
    """Tail risk contribution of one position."""
    position_id: str
    weight: float
    standalone_loss: float
    marginal_contribution: float
    pct_of_total: float


@dataclass
class ScenarioResult:
    """Impact assessment for a single scenario."""
    scenario: ScenarioDef
    portfolio_drawdown_pct: float
    adjusted_drawdown_pct: float   # beta-adjusted for credit spreads
    trough_value: float
    peak_to_trough_days: int
    estimated_recovery_days: int
    vix_multiplier: float
    greeks_pnl: Optional[GreeksPnL] = None
    position_contributions: List[PositionContribution] = field(default_factory=list)
    equity_path: List[float] = field(default_factory=list)
    probability_weighted_loss: float = 0.0


@dataclass
class AnalysisResult:
    """Complete scenario analysis output."""
    scenarios: List[ScenarioResult] = field(default_factory=list)
    worst_case: Optional[ScenarioResult] = None
    probability_weighted_loss: float = 0.0
    expected_shortfall: float = 0.0
    starting_capital: float = 0.0
    generated_at: str = ""


# ── Core analyzer ───────────────────────────────────────────────────────────
class ScenarioAnalyzer:
    """What-if scenario analysis engine for credit spread portfolios."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        spread_beta: float = SPREAD_BETA,
        historical_mean_return: float = 0.0004,
    ) -> None:
        self.starting_capital = starting_capital
        self.spread_beta = spread_beta
        self.hist_mean = historical_mean_return

    # ── Public API ──────────────────────────────────────────────────────────
    def analyze(
        self,
        scenarios: Optional[List[ScenarioDef]] = None,
        include_historical: bool = True,
        positions: Optional[Dict[str, float]] = None,
        portfolio_delta: float = 0.0,
        portfolio_vega: float = 0.0,
        portfolio_theta: float = 0.0,
        portfolio_gamma: float = 0.0,
    ) -> AnalysisResult:
        """Run scenario analysis.

        Parameters
        ----------
        scenarios : list of ScenarioDef, optional
            Custom scenarios.  Historical added if include_historical=True.
        include_historical : bool
            Whether to include built-in historical scenarios.
        positions : dict, optional
            position_id → weight (for tail risk contribution).
        portfolio_delta/vega/theta/gamma : float
            Portfolio Greeks for stress P&L decomposition.
        """
        all_scenarios: List[ScenarioDef] = []

        if include_historical:
            for h in HISTORICAL_SCENARIOS:
                all_scenarios.append(ScenarioDef(**h))

        if scenarios:
            all_scenarios.extend(scenarios)

        if not all_scenarios:
            return AnalysisResult(
                starting_capital=self.starting_capital,
                generated_at=self._now(),
            )

        results: List[ScenarioResult] = []
        for s in all_scenarios:
            sr = self._evaluate_scenario(
                s, positions,
                portfolio_delta, portfolio_vega,
                portfolio_theta, portfolio_gamma,
            )
            results.append(sr)

        worst = min(results, key=lambda r: r.adjusted_drawdown_pct)
        pw_loss = sum(r.probability_weighted_loss for r in results)
        es = self._expected_shortfall(results)

        return AnalysisResult(
            scenarios=results,
            worst_case=worst,
            probability_weighted_loss=pw_loss,
            expected_shortfall=es,
            starting_capital=self.starting_capital,
            generated_at=self._now(),
        )

    def create_custom_scenario(
        self,
        name: str,
        shock_pct: float,
        duration_days: int,
        vix_start: float = 15.0,
        vix_peak: float = 40.0,
        probability: float = 0.05,
        correlation_shock: float = 0.0,
        seed: int = 42,
    ) -> ScenarioDef:
        """Create a custom scenario definition."""
        shocks = _build_shock_path(shock_pct, duration_days, seed=seed)
        return ScenarioDef(
            name=name,
            description=f"Custom: {shock_pct:.0%} over {duration_days}d",
            daily_shocks=shocks,
            vix_start=vix_start,
            vix_peak=vix_peak,
            probability=probability,
            correlation_shock=correlation_shock,
        )

    def generate_report(
        self,
        result: AnalysisResult,
        output_path: str | Path = "reports/scenario_analysis.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Scenario analysis report written to %s", path)
        return path

    # ── Scenario evaluation ─────────────────────────────────────────────────
    def _evaluate_scenario(
        self,
        scenario: ScenarioDef,
        positions: Optional[Dict[str, float]],
        delta: float,
        vega: float,
        theta: float,
        gamma: float,
    ) -> ScenarioResult:
        shocks = scenario.daily_shocks
        n = len(shocks)
        if n == 0:
            return ScenarioResult(
                scenario=scenario, portfolio_drawdown_pct=0.0,
                adjusted_drawdown_pct=0.0, trough_value=self.starting_capital,
                peak_to_trough_days=0, estimated_recovery_days=0,
                vix_multiplier=1.0,
            )

        # Build equity path
        equity = [self.starting_capital]
        for s in shocks:
            equity.append(equity[-1] * (1 + s))
        equity_arr = np.array(equity)

        trough_idx = int(np.argmin(equity_arr))
        trough = float(equity_arr[trough_idx])
        raw_dd = (trough - self.starting_capital) / self.starting_capital
        adj_dd = raw_dd * self.spread_beta

        vix_mult = scenario.vix_peak / max(scenario.vix_start, 1.0)

        # Recovery estimation
        recovery_needed = (self.starting_capital / max(trough * abs(self.spread_beta), 1)) - 1
        if self.hist_mean > 1e-9 and recovery_needed > 0:
            recovery_days = int(math.ceil(
                math.log(1 + recovery_needed) / math.log(1 + self.hist_mean)
            ))
        else:
            recovery_days = 0

        # Greeks P&L
        total_underlying_move = float(np.prod(1 + np.array(shocks)) - 1)
        vix_change = scenario.vix_peak - scenario.vix_start
        greeks = self._greeks_pnl(
            delta, vega, theta, gamma,
            total_underlying_move, vix_change, n,
        )

        # Position contributions
        contribs: List[PositionContribution] = []
        if positions:
            contribs = self._position_contributions(positions, adj_dd)

        pw_loss = abs(adj_dd) * scenario.probability * self.starting_capital

        return ScenarioResult(
            scenario=scenario,
            portfolio_drawdown_pct=raw_dd,
            adjusted_drawdown_pct=adj_dd,
            trough_value=self.starting_capital * (1 + adj_dd),
            peak_to_trough_days=trough_idx,
            estimated_recovery_days=recovery_days,
            vix_multiplier=vix_mult,
            greeks_pnl=greeks,
            position_contributions=contribs,
            equity_path=equity,
            probability_weighted_loss=pw_loss,
        )

    @staticmethod
    def _greeks_pnl(
        delta: float, vega: float, theta: float, gamma: float,
        underlying_move: float, vix_change: float, n_days: int,
    ) -> GreeksPnL:
        delta_pnl = delta * underlying_move * 100
        gamma_pnl = 0.5 * gamma * (underlying_move * 100) ** 2
        vega_pnl = vega * vix_change
        theta_pnl = theta * n_days
        total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
        return GreeksPnL(
            delta_pnl=delta_pnl,
            vega_pnl=vega_pnl,
            theta_pnl=theta_pnl,
            gamma_pnl=gamma_pnl,
            residual_pnl=0.0,
            total_pnl=total,
        )

    @staticmethod
    def _position_contributions(
        positions: Dict[str, float], total_dd: float,
    ) -> List[PositionContribution]:
        total_loss = abs(total_dd)
        contribs: List[PositionContribution] = []
        for pid, weight in positions.items():
            standalone = total_loss * weight
            marginal = standalone  # simplified: proportional to weight
            pct = weight  # proportional contribution
            contribs.append(PositionContribution(
                position_id=pid,
                weight=weight,
                standalone_loss=standalone,
                marginal_contribution=marginal,
                pct_of_total=pct,
            ))
        return contribs

    @staticmethod
    def _expected_shortfall(results: List[ScenarioResult]) -> float:
        """Probability-weighted expected shortfall across scenarios."""
        if not results:
            return 0.0
        total_prob = sum(r.scenario.probability for r in results)
        if total_prob < 1e-12:
            return 0.0
        weighted = sum(
            abs(r.adjusted_drawdown_pct) * r.scenario.probability
            for r in results
        )
        return weighted / total_prob

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: AnalysisResult) -> str:
        cards = self._html_cards(r)
        comp_table = self._html_comparison_table(r.scenarios)
        impact_chart = self._svg_impact_bars(r.scenarios)
        worst_section = self._html_worst_case(r.worst_case)
        greeks_tbl = self._html_greeks(r.scenarios)
        contrib_tbl = self._html_contributions(r.worst_case)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Scenario Analysis</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}.warn{{color:#fbbf24}}
svg{{display:block;margin:0 auto}}
</style>
</head>
<body>
<h1>Scenario Analysis</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {len(r.scenarios)} scenarios &middot; Capital: ${r.starting_capital:,.0f}</p>

{cards}

<div class="sec">
<h2>P&L Impact by Scenario</h2>
{impact_chart}
</div>

{comp_table}
{worst_section}
{greeks_tbl}
{contrib_tbl}

</body>
</html>"""

    @staticmethod
    def _html_cards(r: AnalysisResult) -> str:
        wc = r.worst_case
        wc_name = wc.scenario.name if wc else "N/A"
        wc_dd = f"{wc.adjusted_drawdown_pct:.1%}" if wc else "N/A"
        wc_rec = f"{wc.estimated_recovery_days}d" if wc else "N/A"
        return f"""<div class="grid">
<div class="card"><div class="lbl">Worst Case</div><div class="val neg">{wc_name}</div></div>
<div class="card"><div class="lbl">Worst DD</div><div class="val neg">{wc_dd}</div></div>
<div class="card"><div class="lbl">Recovery</div><div class="val">{wc_rec}</div></div>
<div class="card"><div class="lbl">PW Loss</div><div class="val">${r.probability_weighted_loss:,.0f}</div></div>
<div class="card"><div class="lbl">Expected Shortfall</div><div class="val">{r.expected_shortfall:.2%}</div></div>
<div class="card"><div class="lbl">Scenarios</div><div class="val">{len(r.scenarios)}</div></div>
</div>"""

    @staticmethod
    def _html_comparison_table(results: List[ScenarioResult]) -> str:
        if not results:
            return ""
        rows = ""
        for sr in sorted(results, key=lambda r: r.adjusted_drawdown_pct):
            rows += (
                f"<tr><td>{sr.scenario.name}</td>"
                f"<td class='neg'>{sr.adjusted_drawdown_pct:.1%}</td>"
                f"<td>{sr.peak_to_trough_days}d</td>"
                f"<td>{sr.estimated_recovery_days}d</td>"
                f"<td>{sr.vix_multiplier:.1f}x</td>"
                f"<td>{sr.scenario.probability:.0%}</td>"
                f"<td>${sr.probability_weighted_loss:,.0f}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Scenario Comparison</h2>
<table>
<thead><tr><th>Scenario</th><th>Adj DD</th><th>Peak→Trough</th><th>Recovery</th><th>VIX Mult</th><th>Prob</th><th>PW Loss</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _svg_impact_bars(results: List[ScenarioResult]) -> str:
        if not results:
            return "<p>No scenarios.</p>"
        w, h = 580, 40 * len(results) + 40
        pl = 160
        max_dd = max(abs(r.adjusted_drawdown_pct) for r in results) or 0.01
        bars = ""
        sorted_r = sorted(results, key=lambda r: r.adjusted_drawdown_pct)
        for i, sr in enumerate(sorted_r):
            y = 20 + i * 40
            bw = abs(sr.adjusted_drawdown_pct) / max_dd * (w - pl - 60)
            bars += (
                f'<text x="{pl - 5}" y="{y + 14}" text-anchor="end" font-size="11" fill="#e2e8f0">{sr.scenario.name}</text>'
                f'<rect x="{pl}" y="{y}" width="{bw:.0f}" height="22" rx="3" fill="#f87171" opacity="0.8"/>'
                f'<text x="{pl + bw + 5}" y="{y + 14}" font-size="10" fill="#94a3b8">{sr.adjusted_drawdown_pct:.1%}</text>'
            )
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'

    @staticmethod
    def _html_worst_case(wc: Optional[ScenarioResult]) -> str:
        if not wc:
            return ""
        return f"""<div class="sec">
<h2>Worst-Case Dashboard — {wc.scenario.name}</h2>
<table>
<thead><tr><th>Metric</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Raw Drawdown</td><td class="neg">{wc.portfolio_drawdown_pct:.2%}</td></tr>
<tr><td>Beta-Adjusted DD</td><td class="neg">{wc.adjusted_drawdown_pct:.2%}</td></tr>
<tr><td>Trough Value</td><td>${wc.trough_value:,.0f}</td></tr>
<tr><td>Peak → Trough</td><td>{wc.peak_to_trough_days} days</td></tr>
<tr><td>Est. Recovery</td><td>{wc.estimated_recovery_days} days</td></tr>
<tr><td>VIX Multiplier</td><td>{wc.vix_multiplier:.1f}x</td></tr>
<tr><td>Scenario Probability</td><td>{wc.scenario.probability:.0%}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_greeks(results: List[ScenarioResult]) -> str:
        has_greeks = any(r.greeks_pnl and r.greeks_pnl.total_pnl != 0 for r in results)
        if not has_greeks:
            return ""
        rows = ""
        for sr in results:
            g = sr.greeks_pnl
            if not g:
                continue
            rows += (
                f"<tr><td>{sr.scenario.name}</td>"
                f"<td>${g.delta_pnl:,.0f}</td>"
                f"<td>${g.gamma_pnl:,.0f}</td>"
                f"<td>${g.vega_pnl:,.0f}</td>"
                f"<td>${g.theta_pnl:,.0f}</td>"
                f"<td><strong>${g.total_pnl:,.0f}</strong></td></tr>"
            )
        return f"""<div class="sec">
<h2>Stress P&L — Greeks Decomposition</h2>
<table>
<thead><tr><th>Scenario</th><th>Delta</th><th>Gamma</th><th>Vega</th><th>Theta</th><th>Total</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_contributions(wc: Optional[ScenarioResult]) -> str:
        if not wc or not wc.position_contributions:
            return ""
        rows = ""
        for c in sorted(wc.position_contributions, key=lambda x: -x.pct_of_total):
            rows += (
                f"<tr><td>{c.position_id}</td>"
                f"<td>{c.weight:.1%}</td>"
                f"<td>{c.standalone_loss:.2%}</td>"
                f"<td>{c.marginal_contribution:.2%}</td>"
                f"<td>{c.pct_of_total:.1%}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Position Tail Risk Contribution (Worst Case)</h2>
<table>
<thead><tr><th>Position</th><th>Weight</th><th>Standalone Loss</th><th>Marginal</th><th>% of Total</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
