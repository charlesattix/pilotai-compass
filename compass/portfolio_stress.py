"""Advanced portfolio stress testing — historical and synthetic scenarios,
reverse stress testing, P&L attribution under stress, margin adequacy,
and recovery time estimation.

Provides:
  1. Historical scenario replay (2008 GFC, 2020 COVID, 2022 rate hikes)
  2. Synthetic stress (parallel yield shift, vol spike, correlation breakdown)
  3. Reverse stress testing (find scenario causing target loss)
  4. P&L attribution under stress (delta, vega, theta, gamma)
  5. Margin adequacy under stress
  6. Recovery time estimation
  7. HTML report with comparison, waterfall, margin analysis
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


def _shock_path(total: float, days: int, seed: int = 42) -> List[float]:
    if days <= 0:
        return []
    if days == 1:
        return [total]
    rng = np.random.RandomState(seed)
    lr = math.log(1 + total)
    w = np.exp(-np.arange(days) / max(days / 3, 1))
    w /= w.sum()
    raw = w * lr + rng.randn(days) * abs(lr) * 0.05
    raw *= lr / raw.sum() if abs(raw.sum()) > 1e-12 else 1.0
    return [float(math.exp(r) - 1) for r in raw]


# ── Built-in scenarios ──────────────────────────────────────────────────────
@dataclass
class StressScenario:
    name: str
    description: str
    equity_shock: float          # total equity move
    vol_shock: float             # VIX change (absolute points)
    yield_shift_bps: float       # parallel yield curve shift
    correlation_shock: float     # increase in cross-asset correlation
    duration_days: int
    daily_shocks: List[float] = field(default_factory=list)
    probability: float = 0.05


HISTORICAL_SCENARIOS: List[StressScenario] = [
    StressScenario("2008 GFC", "Global financial crisis", -0.57, 60.0, -200, 0.30, 350, _shock_path(-0.57, 350, 1), 0.02),
    StressScenario("COVID 2020", "Pandemic crash", -0.34, 67.0, -150, 0.25, 23, _shock_path(-0.34, 23, 2), 0.03),
    StressScenario("2022 Rate Hikes", "Fed tightening cycle", -0.25, 19.0, 300, 0.15, 190, _shock_path(-0.25, 190, 3), 0.08),
]

SYNTHETIC_SCENARIOS: List[StressScenario] = [
    StressScenario("Vol Spike (+40)", "VIX jumps 40 points overnight", -0.08, 40.0, 0, 0.20, 3, _shock_path(-0.08, 3, 10), 0.05),
    StressScenario("Yield +200bps", "Parallel yield curve shift up", -0.12, 10.0, 200, 0.10, 60, _shock_path(-0.12, 60, 11), 0.06),
    StressScenario("Corr Breakdown", "Cross-asset correlations spike to 1", -0.15, 25.0, 0, 0.50, 10, _shock_path(-0.15, 10, 12), 0.04),
]


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class StressPnL:
    delta_pnl: float = 0.0
    gamma_pnl: float = 0.0
    vega_pnl: float = 0.0
    theta_pnl: float = 0.0
    total_pnl: float = 0.0


@dataclass
class MarginAnalysis:
    current_margin: float
    stress_margin: float
    margin_call_triggered: bool
    excess_margin: float          # positive = safe, negative = call
    margin_utilisation: float     # 0-1


@dataclass
class ScenarioOutcome:
    scenario: StressScenario
    portfolio_dd: float           # raw
    adjusted_dd: float            # beta-adjusted
    trough_value: float
    peak_to_trough_days: int
    recovery_days: int
    pnl: Optional[StressPnL] = None
    margin: Optional[MarginAnalysis] = None
    equity_path: List[float] = field(default_factory=list)
    pw_loss: float = 0.0


@dataclass
class ReverseStressResult:
    target_loss_pct: float
    required_equity_shock: float
    required_vol_shock: float
    scenario_name: str


@dataclass
class PortfolioStressResult:
    outcomes: List[ScenarioOutcome] = field(default_factory=list)
    worst_case: Optional[ScenarioOutcome] = None
    reverse_stress: Optional[ReverseStressResult] = None
    total_pw_loss: float = 0.0
    starting_capital: float = 0.0
    n_scenarios: int = 0
    generated_at: str = ""


# ── Core engine ─────────────────────────────────────────────────────────────
class PortfolioStressEngine:
    """Advanced portfolio stress testing engine."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        spread_beta: float = SPREAD_BETA,
        hist_daily_mean: float = 0.0004,
        maintenance_margin_pct: float = 0.25,
    ) -> None:
        self.starting_capital = starting_capital
        self.spread_beta = spread_beta
        self.hist_mean = hist_daily_mean
        self.maint_margin = maintenance_margin_pct

    def run(
        self,
        include_historical: bool = True,
        include_synthetic: bool = True,
        custom_scenarios: Optional[List[StressScenario]] = None,
        portfolio_delta: float = 0.0,
        portfolio_gamma: float = 0.0,
        portfolio_vega: float = 0.0,
        portfolio_theta: float = 0.0,
        current_margin: float = 0.0,
        reverse_target_pct: Optional[float] = None,
    ) -> PortfolioStressResult:
        scenarios: List[StressScenario] = []
        if include_historical:
            scenarios.extend(HISTORICAL_SCENARIOS)
        if include_synthetic:
            scenarios.extend(SYNTHETIC_SCENARIOS)
        if custom_scenarios:
            scenarios.extend(custom_scenarios)

        if not scenarios:
            return PortfolioStressResult(starting_capital=self.starting_capital, generated_at=self._now())

        outcomes: List[ScenarioOutcome] = []
        for s in scenarios:
            outcomes.append(self._evaluate(s, portfolio_delta, portfolio_gamma, portfolio_vega, portfolio_theta, current_margin))

        worst = min(outcomes, key=lambda o: o.adjusted_dd)
        pw = sum(o.pw_loss for o in outcomes)

        reverse = None
        if reverse_target_pct is not None:
            reverse = self._reverse_stress(reverse_target_pct, portfolio_delta, portfolio_vega)

        return PortfolioStressResult(
            outcomes=outcomes,
            worst_case=worst,
            reverse_stress=reverse,
            total_pw_loss=pw,
            starting_capital=self.starting_capital,
            n_scenarios=len(outcomes),
            generated_at=self._now(),
        )

    def create_scenario(
        self, name: str, equity_shock: float, vol_shock: float = 0,
        yield_shift: float = 0, corr_shock: float = 0,
        duration: int = 10, probability: float = 0.05, seed: int = 42,
    ) -> StressScenario:
        return StressScenario(
            name=name, description=f"Custom: {equity_shock:.0%}",
            equity_shock=equity_shock, vol_shock=vol_shock,
            yield_shift_bps=yield_shift, correlation_shock=corr_shock,
            duration_days=duration,
            daily_shocks=_shock_path(equity_shock, duration, seed),
            probability=probability,
        )

    def generate_report(
        self, result: PortfolioStressResult,
        output_path: str | Path = "reports/portfolio_stress.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Portfolio stress report written to %s", path)
        return path

    # ── Evaluation ──────────────────────────────────────────────────────────
    def _evaluate(
        self, s: StressScenario,
        delta: float, gamma: float, vega: float, theta: float,
        current_margin: float,
    ) -> ScenarioOutcome:
        shocks = s.daily_shocks or _shock_path(s.equity_shock, s.duration_days)
        n = len(shocks)

        eq = [self.starting_capital]
        for sh in shocks:
            eq.append(eq[-1] * (1 + sh))
        eq_arr = np.array(eq)
        trough_idx = int(np.argmin(eq_arr))
        trough = float(eq_arr[trough_idx])
        raw_dd = (trough - self.starting_capital) / self.starting_capital
        adj_dd = raw_dd * self.spread_beta

        # Recovery
        rec_needed = (self.starting_capital / max(abs(trough * self.spread_beta), 1)) - 1
        rec_days = int(math.ceil(math.log(1 + rec_needed) / math.log(1 + self.hist_mean))) if self.hist_mean > 1e-9 and rec_needed > 0 else 0

        # Greeks P&L
        total_move = float(np.prod(1 + np.array(shocks)) - 1)
        d_pnl = delta * total_move * 100
        g_pnl = 0.5 * gamma * (total_move * 100) ** 2
        v_pnl = vega * s.vol_shock
        t_pnl = theta * n
        pnl = StressPnL(d_pnl, g_pnl, v_pnl, t_pnl, d_pnl + g_pnl + v_pnl + t_pnl)

        # Margin analysis
        margin = None
        if current_margin > 0:
            stress_val = max(0, self.starting_capital * (1 + adj_dd))
            req = stress_val * self.maint_margin
            # Even when portfolio value drops below zero, margin required is at least the loss
            req = max(req, abs(adj_dd) * self.starting_capital * self.maint_margin)
            excess = current_margin - req
            util = req / current_margin if current_margin > 0 else 1.0
            margin = MarginAnalysis(
                current_margin=current_margin,
                stress_margin=req,
                margin_call_triggered=excess < 0,
                excess_margin=excess,
                margin_utilisation=min(util, 1.0),
            )

        pw = abs(adj_dd) * s.probability * self.starting_capital

        return ScenarioOutcome(
            scenario=s, portfolio_dd=raw_dd, adjusted_dd=adj_dd,
            trough_value=self.starting_capital * (1 + adj_dd),
            peak_to_trough_days=trough_idx, recovery_days=rec_days,
            pnl=pnl, margin=margin, equity_path=eq, pw_loss=pw,
        )

    # ── Reverse stress ──────────────────────────────────────────────────────
    def _reverse_stress(
        self, target_pct: float, delta: float, vega: float,
    ) -> ReverseStressResult:
        """Find equity and vol shocks that produce target_pct loss."""
        # target_pct should be positive (e.g. 0.30 for 30% loss)
        # Adjusted DD = raw_dd * beta → raw needed = target / beta
        raw_needed = -target_pct / self.spread_beta
        # Vol shock estimate: use vega sensitivity
        vol_needed = 0.0
        if abs(vega) > 1e-9:
            vol_needed = abs(target_pct * self.starting_capital / vega) * 0.5

        return ReverseStressResult(
            target_loss_pct=target_pct,
            required_equity_shock=raw_needed,
            required_vol_shock=vol_needed,
            scenario_name=f"Reverse: {target_pct:.0%} loss",
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML ────────────────────────────────────────────────────────────────
    def _build_html(self, r: PortfolioStressResult) -> str:
        cards = self._html_cards(r)
        comp = self._html_comparison(r.outcomes)
        waterfall = self._svg_waterfall(r.outcomes)
        pnl_tbl = self._html_pnl(r.outcomes)
        margin_tbl = self._html_margin(r.outcomes)
        reverse_sec = self._html_reverse(r.reverse_stress)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Portfolio Stress Test</title>
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
<h1>Portfolio Stress Test</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_scenarios} scenarios &middot; Capital: ${r.starting_capital:,.0f}</p>
{cards}
<div class="sec"><h2>P&L Waterfall</h2>{waterfall}</div>
{comp}
{pnl_tbl}
{margin_tbl}
{reverse_sec}
</body>
</html>"""

    @staticmethod
    def _html_cards(r: PortfolioStressResult) -> str:
        wc = r.worst_case
        margin_calls = sum(1 for o in r.outcomes if o.margin and o.margin.margin_call_triggered)
        return f"""<div class="grid">
<div class="card"><div class="lbl">Worst Case</div><div class="val neg">{wc.scenario.name if wc else 'N/A'}</div></div>
<div class="card"><div class="lbl">Worst DD</div><div class="val neg">{f'{wc.adjusted_dd:.1%}' if wc else 'N/A'}</div></div>
<div class="card"><div class="lbl">Recovery</div><div class="val">{f'{wc.recovery_days}d' if wc else 'N/A'}</div></div>
<div class="card"><div class="lbl">PW Loss</div><div class="val">${r.total_pw_loss:,.0f}</div></div>
<div class="card"><div class="lbl">Margin Calls</div><div class="val {'neg' if margin_calls else ''}">{margin_calls}</div></div>
</div>"""

    @staticmethod
    def _html_comparison(outcomes: List[ScenarioOutcome]) -> str:
        if not outcomes:
            return ""
        rows = ""
        for o in sorted(outcomes, key=lambda x: x.adjusted_dd):
            rows += (f"<tr><td>{o.scenario.name}</td><td class='neg'>{o.adjusted_dd:.1%}</td>"
                     f"<td>{o.peak_to_trough_days}d</td><td>{o.recovery_days}d</td>"
                     f"<td>{o.scenario.vol_shock:+.0f}</td><td>{o.scenario.yield_shift_bps:+.0f}</td>"
                     f"<td>{o.scenario.probability:.0%}</td></tr>")
        return f"""<div class="sec"><h2>Scenario Comparison</h2>
<table><thead><tr><th>Scenario</th><th>Adj DD</th><th>Peak→Trough</th><th>Recovery</th><th>Vol Shock</th><th>Yield Shift</th><th>Prob</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _svg_waterfall(outcomes: List[ScenarioOutcome]) -> str:
        if not outcomes:
            return "<p>No data.</p>"
        w, h = 580, 36 * len(outcomes) + 30
        pl = 160
        mx = max(abs(o.adjusted_dd) for o in outcomes) or 0.01
        bars = ""
        for i, o in enumerate(sorted(outcomes, key=lambda x: x.adjusted_dd)):
            y = 10 + i * 36
            bw = abs(o.adjusted_dd) / mx * (w - pl - 70)
            bars += (f'<text x="{pl - 5}" y="{y + 14}" text-anchor="end" font-size="11" fill="#e2e8f0">{o.scenario.name}</text>'
                     f'<rect x="{pl}" y="{y}" width="{bw:.0f}" height="24" rx="3" fill="#f87171" opacity="0.8"/>'
                     f'<text x="{pl + bw + 5}" y="{y + 14}" font-size="10" fill="#94a3b8">{o.adjusted_dd:.1%}</text>')
        return f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'

    @staticmethod
    def _html_pnl(outcomes: List[ScenarioOutcome]) -> str:
        has = any(o.pnl and o.pnl.total_pnl != 0 for o in outcomes)
        if not has:
            return ""
        rows = ""
        for o in outcomes:
            p = o.pnl
            if not p:
                continue
            rows += (f"<tr><td>{o.scenario.name}</td><td>${p.delta_pnl:,.0f}</td><td>${p.gamma_pnl:,.0f}</td>"
                     f"<td>${p.vega_pnl:,.0f}</td><td>${p.theta_pnl:,.0f}</td><td><strong>${p.total_pnl:,.0f}</strong></td></tr>")
        return f"""<div class="sec"><h2>Stress P&L Attribution</h2>
<table><thead><tr><th>Scenario</th><th>Delta</th><th>Gamma</th><th>Vega</th><th>Theta</th><th>Total</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_margin(outcomes: List[ScenarioOutcome]) -> str:
        has = any(o.margin for o in outcomes)
        if not has:
            return ""
        rows = ""
        for o in outcomes:
            m = o.margin
            if not m:
                continue
            cls = "neg" if m.margin_call_triggered else "pos"
            rows += (f"<tr><td>{o.scenario.name}</td><td>${m.current_margin:,.0f}</td>"
                     f"<td>${m.stress_margin:,.0f}</td><td class='{cls}'>${m.excess_margin:,.0f}</td>"
                     f"<td>{m.margin_utilisation:.0%}</td><td class='{cls}'>{'YES' if m.margin_call_triggered else 'No'}</td></tr>")
        return f"""<div class="sec"><h2>Margin Adequacy</h2>
<table><thead><tr><th>Scenario</th><th>Current</th><th>Stress Req</th><th>Excess</th><th>Utilisation</th><th>Margin Call</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""

    @staticmethod
    def _html_reverse(rs: Optional[ReverseStressResult]) -> str:
        if not rs:
            return ""
        return f"""<div class="sec"><h2>Reverse Stress Test</h2>
<table><tbody>
<tr><td>Target Loss</td><td class="neg">{rs.target_loss_pct:.0%}</td></tr>
<tr><td>Required Equity Shock</td><td class="neg">{rs.required_equity_shock:.1%}</td></tr>
<tr><td>Required Vol Shock</td><td>+{rs.required_vol_shock:.0f} VIX pts</td></tr>
</tbody></table></div>"""
