"""Advanced stress scenario engine — predefined and custom scenarios with
correlated multi-asset stress, Monte-Carlo recovery simulation, and
portfolio-level Greeks P&L decomposition.

Extends scenario_analyzer.py by adding:
  1. Brexit + Volmageddon predefined scenarios
  2. Correlated stress moves across multiple assets (correlation matrix)
  3. Monte-Carlo recovery path simulation from trough
  4. Per-asset stress P&L decomposition
  5. Conditional VaR under each scenario
  6. HTML report with comparison, waterfall, and recovery curves
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

SPREAD_BETA = 1.5
PERIODS_PER_YEAR = 252


# ── Shock path builder ──────────────────────────────────────────────────────
def _build_shock_path(total_return: float, n_days: int, seed: int = 42) -> List[float]:
    if n_days <= 0:
        return []
    if n_days == 1:
        return [total_return]
    rng = np.random.RandomState(seed)
    log_ret = math.log(1 + total_return)
    weights = np.exp(-np.arange(n_days) / max(n_days / 3, 1))
    weights /= weights.sum()
    raw = weights * log_ret
    raw += rng.randn(n_days) * abs(log_ret) * 0.05
    raw *= log_ret / raw.sum() if abs(raw.sum()) > 1e-12 else 1.0
    return [float(math.exp(r) - 1) for r in raw]


# ── Predefined scenarios ────────────────────────────────────────────────────
@dataclass
class ScenarioDef:
    """Scenario definition."""
    name: str
    description: str
    daily_shocks: List[float]
    vix_start: float = 15.0
    vix_peak: float = 40.0
    probability: float = 0.05
    asset_correlations: Optional[Dict[str, float]] = None  # asset → corr with driver


PREDEFINED_SCENARIOS: List[Dict[str, Any]] = [
    {
        "name": "2008 GFC",
        "description": "S&P 500 fell ~57% over ~17 months",
        "daily_shocks": _build_shock_path(-0.57, 350, seed=1),
        "vix_start": 20.0, "vix_peak": 80.0, "probability": 0.02,
        "asset_correlations": {"SPY": 1.0, "QQQ": 0.95, "IWM": 0.92, "HYG": 0.80},
    },
    {
        "name": "COVID Crash (2020)",
        "description": "S&P 500 fell ~34% in 23 trading days",
        "daily_shocks": _build_shock_path(-0.34, 23, seed=2),
        "vix_start": 15.0, "vix_peak": 82.0, "probability": 0.03,
        "asset_correlations": {"SPY": 1.0, "QQQ": 0.97, "IWM": 0.90, "HYG": 0.85},
    },
    {
        "name": "2022 Rate Hikes",
        "description": "S&P 500 fell ~25% over ~9 months",
        "daily_shocks": _build_shock_path(-0.25, 190, seed=3),
        "vix_start": 17.0, "vix_peak": 36.0, "probability": 0.08,
        "asset_correlations": {"SPY": 1.0, "QQQ": 0.88, "IWM": 0.82, "HYG": 0.65},
    },
    {
        "name": "Flash Crash",
        "description": "Sudden 10% drawdown in a single session",
        "daily_shocks": _build_shock_path(-0.10, 1, seed=4),
        "vix_start": 15.0, "vix_peak": 65.0, "probability": 0.05,
        "asset_correlations": {"SPY": 1.0, "QQQ": 0.98, "IWM": 0.95, "HYG": 0.70},
    },
    {
        "name": "Brexit (2016)",
        "description": "GBP flash crash, global equities fell ~5-8% in 2 days",
        "daily_shocks": _build_shock_path(-0.07, 3, seed=6),
        "vix_start": 14.0, "vix_peak": 26.0, "probability": 0.06,
        "asset_correlations": {"SPY": 1.0, "QQQ": 0.90, "IWM": 0.85, "HYG": 0.50},
    },
    {
        "name": "Volmageddon (2018)",
        "description": "XIV collapse, VIX doubled overnight, S&P ~10% correction",
        "daily_shocks": _build_shock_path(-0.12, 8, seed=7),
        "vix_start": 11.0, "vix_peak": 50.0, "probability": 0.04,
        "asset_correlations": {"SPY": 1.0, "QQQ": 0.93, "IWM": 0.88, "HYG": 0.60},
    },
]


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class AssetStress:
    """Stress result for a single asset."""
    asset: str
    correlation: float
    stress_return: float
    stress_pnl: float


@dataclass
class GreeksStressPnL:
    """P&L decomposition under stress."""
    delta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    vega_pnl: float = 0.0
    theta_pnl: float = 0.0
    total_pnl: float = 0.0


@dataclass
class RecoveryPath:
    """Monte-Carlo recovery simulation from trough."""
    median_days: int
    p10_days: int               # fast recovery (10th percentile)
    p90_days: int               # slow recovery (90th percentile)
    recovery_probability: float  # P(recover within 2 years)
    sample_paths: List[List[float]] = field(default_factory=list)  # first 5 paths


@dataclass
class ScenarioOutcome:
    """Full outcome for one scenario."""
    scenario: ScenarioDef
    raw_drawdown: float
    adjusted_drawdown: float       # beta-adjusted
    trough_value: float
    peak_to_trough_days: int
    vix_multiplier: float
    greeks_pnl: Optional[GreeksStressPnL] = None
    asset_stress: List[AssetStress] = field(default_factory=list)
    recovery: Optional[RecoveryPath] = None
    conditional_var_95: float = 0.0
    probability_weighted_loss: float = 0.0
    equity_path: List[float] = field(default_factory=list)


@dataclass
class StressResult:
    """Complete stress analysis output."""
    outcomes: List[ScenarioOutcome] = field(default_factory=list)
    worst_case: Optional[ScenarioOutcome] = None
    total_pw_loss: float = 0.0
    expected_shortfall: float = 0.0
    starting_capital: float = 0.0
    n_scenarios: int = 0
    generated_at: str = ""


# ── Core engine ─────────────────────────────────────────────────────────────
class StressScenarioEngine:
    """Advanced stress scenario engine with correlated moves and recovery sim."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        spread_beta: float = SPREAD_BETA,
        hist_daily_mean: float = 0.0004,
        hist_daily_vol: float = 0.012,
        recovery_simulations: int = 500,
        random_state: int = 42,
    ) -> None:
        self.starting_capital = starting_capital
        self.spread_beta = spread_beta
        self.hist_mean = hist_daily_mean
        self.hist_vol = hist_daily_vol
        self.n_sims = recovery_simulations
        self.rng = np.random.RandomState(random_state)

    # ── Public API ──────────────────────────────────────────────────────────
    def run(
        self,
        scenarios: Optional[List[ScenarioDef]] = None,
        include_predefined: bool = True,
        asset_weights: Optional[Dict[str, float]] = None,
        portfolio_delta: float = 0.0,
        portfolio_gamma: float = 0.0,
        portfolio_vega: float = 0.0,
        portfolio_theta: float = 0.0,
    ) -> StressResult:
        """Run stress analysis across all scenarios.

        Parameters
        ----------
        scenarios : list of ScenarioDef, optional
            Custom scenarios.
        include_predefined : bool
            Include built-in historical scenarios.
        asset_weights : dict, optional
            asset → portfolio weight for correlated stress.
        portfolio_delta/gamma/vega/theta : float
            Portfolio Greeks for stress P&L.
        """
        all_scenarios: List[ScenarioDef] = []
        if include_predefined:
            for s in PREDEFINED_SCENARIOS:
                all_scenarios.append(ScenarioDef(**s))
        if scenarios:
            all_scenarios.extend(scenarios)

        if not all_scenarios:
            return StressResult(starting_capital=self.starting_capital, generated_at=self._now())

        outcomes: List[ScenarioOutcome] = []
        for s in all_scenarios:
            outcome = self._evaluate(
                s, asset_weights,
                portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta,
            )
            outcomes.append(outcome)

        worst = min(outcomes, key=lambda o: o.adjusted_drawdown)
        pw_loss = sum(o.probability_weighted_loss for o in outcomes)

        total_prob = sum(o.scenario.probability for o in outcomes)
        es = (sum(abs(o.adjusted_drawdown) * o.scenario.probability for o in outcomes) / total_prob
              if total_prob > 1e-12 else 0.0)

        return StressResult(
            outcomes=outcomes,
            worst_case=worst,
            total_pw_loss=pw_loss,
            expected_shortfall=es,
            starting_capital=self.starting_capital,
            n_scenarios=len(outcomes),
            generated_at=self._now(),
        )

    def create_scenario(
        self,
        name: str,
        shock_pct: float,
        duration_days: int,
        vix_start: float = 15.0,
        vix_peak: float = 40.0,
        probability: float = 0.05,
        asset_correlations: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ) -> ScenarioDef:
        """Build a custom scenario."""
        return ScenarioDef(
            name=name,
            description=f"Custom: {shock_pct:.0%} over {duration_days}d",
            daily_shocks=_build_shock_path(shock_pct, duration_days, seed=seed),
            vix_start=vix_start,
            vix_peak=vix_peak,
            probability=probability,
            asset_correlations=asset_correlations,
        )

    def generate_report(
        self,
        result: StressResult,
        output_path: str | Path = "reports/stress_scenario.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Stress scenario report written to %s", path)
        return path

    # ── Evaluation ──────────────────────────────────────────────────────────
    def _evaluate(
        self,
        scenario: ScenarioDef,
        asset_weights: Optional[Dict[str, float]],
        delta: float, gamma: float, vega: float, theta: float,
    ) -> ScenarioOutcome:
        shocks = scenario.daily_shocks
        n = len(shocks)
        if n == 0:
            return ScenarioOutcome(
                scenario=scenario, raw_drawdown=0.0, adjusted_drawdown=0.0,
                trough_value=self.starting_capital, peak_to_trough_days=0,
                vix_multiplier=1.0,
            )

        # Equity path
        equity = [self.starting_capital]
        for s in shocks:
            equity.append(equity[-1] * (1 + s))
        eq_arr = np.array(equity)
        trough_idx = int(np.argmin(eq_arr))
        trough = float(eq_arr[trough_idx])
        raw_dd = (trough - self.starting_capital) / self.starting_capital
        adj_dd = raw_dd * self.spread_beta

        vix_mult = scenario.vix_peak / max(scenario.vix_start, 1.0)

        # Greeks P&L
        total_move = float(np.prod(1 + np.array(shocks)) - 1)
        vix_change = scenario.vix_peak - scenario.vix_start
        greeks = self._greeks_pnl(delta, gamma, vega, theta, total_move, vix_change, n)

        # Correlated asset stress
        asset_stress = self._correlated_stress(scenario, asset_weights, total_move)

        # Recovery simulation
        recovery = self._simulate_recovery(adj_dd)

        # Conditional VaR: 95th percentile of losses during scenario
        losses = -np.diff(eq_arr) / eq_arr[:-1]  # daily losses (positive = loss)
        cvar_95 = float(np.percentile(losses, 95)) if len(losses) > 0 else 0.0

        pw_loss = abs(adj_dd) * scenario.probability * self.starting_capital

        return ScenarioOutcome(
            scenario=scenario,
            raw_drawdown=raw_dd,
            adjusted_drawdown=adj_dd,
            trough_value=self.starting_capital * (1 + adj_dd),
            peak_to_trough_days=trough_idx,
            vix_multiplier=vix_mult,
            greeks_pnl=greeks,
            asset_stress=asset_stress,
            recovery=recovery,
            conditional_var_95=cvar_95,
            probability_weighted_loss=pw_loss,
            equity_path=equity,
        )

    @staticmethod
    def _greeks_pnl(
        delta: float, gamma: float, vega: float, theta: float,
        underlying_move: float, vix_change: float, n_days: int,
    ) -> GreeksStressPnL:
        d = delta * underlying_move * 100
        g = 0.5 * gamma * (underlying_move * 100) ** 2
        v = vega * vix_change
        t = theta * n_days
        return GreeksStressPnL(delta_pnl=d, gamma_pnl=g, vega_pnl=v, theta_pnl=t, total_pnl=d + g + v + t)

    @staticmethod
    def _correlated_stress(
        scenario: ScenarioDef,
        asset_weights: Optional[Dict[str, float]],
        driver_return: float,
    ) -> List[AssetStress]:
        if not asset_weights or not scenario.asset_correlations:
            return []
        results: List[AssetStress] = []
        for asset, weight in asset_weights.items():
            corr = scenario.asset_correlations.get(asset, 0.5)
            stress_ret = driver_return * corr
            stress_pnl = stress_ret * weight
            results.append(AssetStress(
                asset=asset, correlation=corr,
                stress_return=stress_ret, stress_pnl=stress_pnl,
            ))
        return results

    def _simulate_recovery(self, adj_dd: float) -> RecoveryPath:
        """Monte-Carlo recovery path simulation from trough."""
        if adj_dd >= 0:
            return RecoveryPath(0, 0, 0, 1.0)

        target = 1.0  # recover to 100% of starting
        current = 1.0 + adj_dd  # trough level as fraction
        max_days = PERIODS_PER_YEAR * 2  # 2-year horizon

        recovery_times: List[int] = []
        paths: List[List[float]] = []

        for sim in range(self.n_sims):
            equity = current
            path = [equity]
            recovered = False
            for day in range(1, max_days + 1):
                ret = self.hist_mean + self.rng.randn() * self.hist_vol
                equity *= (1 + ret)
                path.append(equity)
                if equity >= target:
                    recovery_times.append(day)
                    recovered = True
                    break
            if not recovered:
                recovery_times.append(max_days)
            if sim < 5:
                paths.append(path[:min(len(path), 200)])

        rt = np.array(recovery_times)
        return RecoveryPath(
            median_days=int(np.median(rt)),
            p10_days=int(np.percentile(rt, 10)),
            p90_days=int(np.percentile(rt, 90)),
            recovery_probability=float(np.mean(rt < max_days)),
            sample_paths=paths,
        )

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: StressResult) -> str:
        cards = self._html_cards(r)
        comp_tbl = self._html_comparison(r.outcomes)
        waterfall = self._svg_waterfall(r.outcomes)
        worst_sec = self._html_worst_case(r.worst_case)
        greeks_tbl = self._html_greeks(r.outcomes)
        asset_tbl = self._html_assets(r.worst_case)
        recovery_sec = self._html_recovery(r.outcomes)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Stress Scenario Analysis</title>
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
<h1>Stress Scenario Analysis</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_scenarios} scenarios &middot; Capital: ${r.starting_capital:,.0f}</p>

{cards}

<div class="sec"><h2>P&L Waterfall</h2>{waterfall}</div>

{comp_tbl}
{worst_sec}
{greeks_tbl}
{asset_tbl}
{recovery_sec}

</body>
</html>"""

    @staticmethod
    def _html_cards(r: StressResult) -> str:
        wc = r.worst_case
        return f"""<div class="grid">
<div class="card"><div class="lbl">Worst Scenario</div><div class="val neg">{wc.scenario.name if wc else 'N/A'}</div></div>
<div class="card"><div class="lbl">Worst DD</div><div class="val neg">{f'{wc.adjusted_drawdown:.1%}' if wc else 'N/A'}</div></div>
<div class="card"><div class="lbl">Recovery (med)</div><div class="val">{wc.recovery.median_days if wc and wc.recovery else 0}d</div></div>
<div class="card"><div class="lbl">PW Loss</div><div class="val">${r.total_pw_loss:,.0f}</div></div>
<div class="card"><div class="lbl">Exp Shortfall</div><div class="val">{r.expected_shortfall:.1%}</div></div>
</div>"""

    @staticmethod
    def _html_comparison(outcomes: List[ScenarioOutcome]) -> str:
        if not outcomes:
            return ""
        rows = ""
        for o in sorted(outcomes, key=lambda x: x.adjusted_drawdown):
            rec = f"{o.recovery.median_days}d" if o.recovery else "N/A"
            rows += (
                f"<tr><td>{o.scenario.name}</td>"
                f"<td class='neg'>{o.adjusted_drawdown:.1%}</td>"
                f"<td>{o.peak_to_trough_days}d</td>"
                f"<td>{rec}</td>"
                f"<td>{o.vix_multiplier:.1f}x</td>"
                f"<td>{o.conditional_var_95:.2%}</td>"
                f"<td>{o.scenario.probability:.0%}</td>"
                f"<td>${o.probability_weighted_loss:,.0f}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Scenario Comparison</h2>
<table>
<thead><tr><th>Scenario</th><th>Adj DD</th><th>Peak→Trough</th><th>Recovery</th><th>VIX</th><th>CVaR 95</th><th>Prob</th><th>PW Loss</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _svg_waterfall(outcomes: List[ScenarioOutcome]) -> str:
        if not outcomes:
            return "<p>No data.</p>"
        w, h = 580, 36 * len(outcomes) + 30
        pl = 160
        max_dd = max(abs(o.adjusted_drawdown) for o in outcomes) or 0.01
        bars = ""
        for i, o in enumerate(sorted(outcomes, key=lambda x: x.adjusted_drawdown)):
            y = 10 + i * 36
            bw = abs(o.adjusted_drawdown) / max_dd * (w - pl - 70)
            bars += (
                f'<text x="{pl - 5}" y="{y + 14}" text-anchor="end" font-size="11" fill="#e2e8f0">{o.scenario.name}</text>'
                f'<rect x="{pl}" y="{y}" width="{bw:.0f}" height="24" rx="3" fill="#f87171" opacity="0.8"/>'
                f'<text x="{pl + bw + 5}" y="{y + 14}" font-size="10" fill="#94a3b8">{o.adjusted_drawdown:.1%}</text>'
            )
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'

    @staticmethod
    def _html_worst_case(wc: Optional[ScenarioOutcome]) -> str:
        if not wc:
            return ""
        return f"""<div class="sec">
<h2>Worst Case — {wc.scenario.name}</h2>
<table>
<tbody>
<tr><td>Raw DD</td><td class="neg">{wc.raw_drawdown:.2%}</td></tr>
<tr><td>Adjusted DD</td><td class="neg">{wc.adjusted_drawdown:.2%}</td></tr>
<tr><td>Trough Value</td><td>${wc.trough_value:,.0f}</td></tr>
<tr><td>Peak→Trough</td><td>{wc.peak_to_trough_days}d</td></tr>
<tr><td>VIX Multiplier</td><td>{wc.vix_multiplier:.1f}x</td></tr>
<tr><td>Recovery (median)</td><td>{wc.recovery.median_days if wc.recovery else 0}d</td></tr>
<tr><td>Recovery P10/P90</td><td>{wc.recovery.p10_days if wc.recovery else 0}d / {wc.recovery.p90_days if wc.recovery else 0}d</td></tr>
<tr><td>Recovery Prob (2yr)</td><td>{f'{wc.recovery.recovery_probability:.0%}' if wc.recovery else 'N/A'}</td></tr>
</tbody>
</table>
</div>"""

    @staticmethod
    def _html_greeks(outcomes: List[ScenarioOutcome]) -> str:
        has = any(o.greeks_pnl and o.greeks_pnl.total_pnl != 0 for o in outcomes)
        if not has:
            return ""
        rows = ""
        for o in outcomes:
            g = o.greeks_pnl
            if not g:
                continue
            rows += (
                f"<tr><td>{o.scenario.name}</td>"
                f"<td>${g.delta_pnl:,.0f}</td><td>${g.gamma_pnl:,.0f}</td>"
                f"<td>${g.vega_pnl:,.0f}</td><td>${g.theta_pnl:,.0f}</td>"
                f"<td><strong>${g.total_pnl:,.0f}</strong></td></tr>"
            )
        return f"""<div class="sec">
<h2>Greeks Stress P&L</h2>
<table>
<thead><tr><th>Scenario</th><th>Delta</th><th>Gamma</th><th>Vega</th><th>Theta</th><th>Total</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_assets(wc: Optional[ScenarioOutcome]) -> str:
        if not wc or not wc.asset_stress:
            return ""
        rows = ""
        for a in sorted(wc.asset_stress, key=lambda x: x.stress_pnl):
            rows += (
                f"<tr><td>{a.asset}</td><td>{a.correlation:.2f}</td>"
                f"<td class='neg'>{a.stress_return:.2%}</td>"
                f"<td class='neg'>{a.stress_pnl:.4f}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Correlated Asset Stress — {wc.scenario.name}</h2>
<table>
<thead><tr><th>Asset</th><th>Correlation</th><th>Stress Return</th><th>Stress P&L</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _html_recovery(outcomes: List[ScenarioOutcome]) -> str:
        has = any(o.recovery and o.recovery.median_days > 0 for o in outcomes)
        if not has:
            return ""
        rows = ""
        for o in sorted(outcomes, key=lambda x: -(x.recovery.median_days if x.recovery else 0)):
            r = o.recovery
            if not r:
                continue
            rows += (
                f"<tr><td>{o.scenario.name}</td>"
                f"<td>{r.median_days}d</td>"
                f"<td>{r.p10_days}d</td><td>{r.p90_days}d</td>"
                f"<td>{r.recovery_probability:.0%}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Recovery Simulation</h2>
<table>
<thead><tr><th>Scenario</th><th>Median</th><th>P10 (fast)</th><th>P90 (slow)</th><th>Prob (2yr)</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
