"""Real-time portfolio risk dashboard — VaR, CVaR, stress tests, Greeks,
concentration, margin, and correlation monitoring in a single view.

Provides:
  1. VaR at 95%/99% via historical, parametric, and Monte Carlo methods
  2. Expected Shortfall (CVaR)
  3. Stress test scenarios (COVID, 2022 bear, flash crash, VIX spike)
  4. Portfolio Greeks summary (delta, gamma, theta, vega)
  5. Concentration risk (max position, sector, Herfindahl)
  6. Margin utilisation tracking
  7. Strategy correlation matrix
  8. HTML dashboard with all sections
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Constants ───────────────────────────────────────────────────────────────
CONFIDENCE_LEVELS = (0.95, 0.99)
TRADING_DAYS = 252

STRESS_SCENARIOS = {
    "COVID_2020": {"equity_shock": -0.34, "vix_peak": 82, "duration": 23},
    "2022_BEAR": {"equity_shock": -0.25, "vix_peak": 36, "duration": 190},
    "FLASH_CRASH": {"equity_shock": -0.10, "vix_peak": 65, "duration": 1},
    "VIX_SPIKE": {"equity_shock": -0.15, "vix_peak": 65, "duration": 5},
}
SPREAD_BETA = 1.5


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class VaREstimate:
    """Value-at-Risk estimate from one method."""
    method: str               # historical, parametric, monte_carlo
    confidence: float
    var: float                # positive = loss magnitude
    cvar: float               # expected shortfall beyond VaR


@dataclass
class StressResult:
    """Impact of one stress scenario."""
    scenario: str
    equity_shock: float
    portfolio_loss_pct: float  # beta-adjusted
    portfolio_loss_dollar: float
    recovery_days: int
    vix_peak: float


@dataclass
class GreeksExposure:
    """Portfolio-level Greeks summary."""
    delta: float
    gamma: float
    theta: float
    vega: float
    delta_dollars: float       # delta × SPY price × 100
    theta_daily: float         # daily theta decay in dollars


@dataclass
class ConcentrationRisk:
    """Concentration analysis."""
    max_position_pct: float
    max_position_name: str
    herfindahl: float          # 0-1 (0=perfectly diversified, 1=single position)
    top3_pct: float            # top-3 positions as % of portfolio
    sector_exposures: Dict[str, float]
    is_concentrated: bool      # True if HHI > 0.25 or max > 20%


@dataclass
class MarginState:
    """Margin utilisation snapshot."""
    total_margin_required: float
    margin_available: float
    utilisation_pct: float
    excess_margin: float
    margin_call_distance_pct: float  # how far from margin call


@dataclass
class RiskDashboardResult:
    """Complete risk dashboard output."""
    var_estimates: List[VaREstimate] = field(default_factory=list)
    stress_results: List[StressResult] = field(default_factory=list)
    greeks: Optional[GreeksExposure] = None
    concentration: Optional[ConcentrationRisk] = None
    margin: Optional[MarginState] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    portfolio_value: float = 0.0
    generated_at: str = ""


# ── Core dashboard ──────────────────────────────────────────────────────────
class RiskDashboard:
    """Portfolio risk monitoring dashboard."""

    def __init__(
        self,
        confidence_levels: Tuple[float, ...] = CONFIDENCE_LEVELS,
        mc_simulations: int = 5000,
        mc_horizon: int = 1,
        seed: int = 42,
    ) -> None:
        self.confidence_levels = confidence_levels
        self.mc_sims = mc_simulations
        self.mc_horizon = mc_horizon
        self.rng = np.random.RandomState(seed)

    def compute(
        self,
        strategy_returns: Dict[str, pd.Series],
        weights: Dict[str, float],
        portfolio_value: float = 100_000.0,
        greeks: Optional[Dict[str, float]] = None,
        positions: Optional[Dict[str, float]] = None,
        sectors: Optional[Dict[str, str]] = None,
        margin_required: float = 0.0,
        margin_available: float = 0.0,
        spy_price: float = 450.0,
    ) -> RiskDashboardResult:
        """Compute all risk metrics.

        Parameters
        ----------
        strategy_returns : dict of strategy_id → daily return Series
        weights : dict of strategy_id → portfolio weight
        portfolio_value : total portfolio value
        greeks : dict with keys delta, gamma, theta, vega
        positions : dict of position_name → weight (for concentration)
        sectors : dict of position_name → sector label
        margin_required/available : for margin tracking
        spy_price : for Greeks dollar conversion
        """
        # Align and build portfolio returns
        port_ret = self._portfolio_returns(strategy_returns, weights)

        # VaR / CVaR
        var_estimates = self._compute_var(port_ret, portfolio_value)

        # Stress tests
        stress = self._stress_test(portfolio_value)

        # Greeks
        greeks_exp = None
        if greeks:
            greeks_exp = self._compute_greeks(greeks, spy_price)

        # Concentration
        concentration = None
        if positions:
            concentration = self._compute_concentration(positions, sectors)

        # Margin
        margin = None
        if margin_available > 0:
            margin = self._compute_margin(margin_required, margin_available)

        # Correlation matrix
        corr = self._correlation_matrix(strategy_returns)

        return RiskDashboardResult(
            var_estimates=var_estimates,
            stress_results=stress,
            greeks=greeks_exp,
            concentration=concentration,
            margin=margin,
            correlation_matrix=corr,
            portfolio_value=portfolio_value,
            generated_at=_now(),
        )

    # ── VaR / CVaR ─────────────────────────────────────────────────────────
    def _compute_var(
        self, returns: np.ndarray, portfolio_value: float,
    ) -> List[VaREstimate]:
        results: List[VaREstimate] = []
        if len(returns) < 20:
            return results

        for conf in self.confidence_levels:
            alpha = 1 - conf

            # Historical
            losses = -returns
            var_h = float(np.percentile(losses, conf * 100))
            tail_h = losses[losses >= var_h]
            cvar_h = float(tail_h.mean()) if len(tail_h) > 0 else var_h
            results.append(VaREstimate("historical", conf,
                                        round(var_h * portfolio_value, 2),
                                        round(cvar_h * portfolio_value, 2)))

            # Parametric (Gaussian)
            mu = float(returns.mean())
            sigma = float(returns.std())
            z = _z_score(conf)
            var_p = (-mu + z * sigma) * portfolio_value
            # CVaR parametric: mu - sigma × φ(z) / α
            phi_z = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
            cvar_p = (-mu + sigma * phi_z / alpha) * portfolio_value
            results.append(VaREstimate("parametric", conf,
                                        round(var_p, 2), round(cvar_p, 2)))

            # Monte Carlo
            mc_returns = self.rng.normal(mu, sigma, (self.mc_sims, self.mc_horizon))
            mc_portfolio = mc_returns.sum(axis=1)
            mc_losses = -mc_portfolio
            var_mc = float(np.percentile(mc_losses, conf * 100))
            tail_mc = mc_losses[mc_losses >= var_mc]
            cvar_mc = float(tail_mc.mean()) if len(tail_mc) > 0 else var_mc
            results.append(VaREstimate("monte_carlo", conf,
                                        round(var_mc * portfolio_value, 2),
                                        round(cvar_mc * portfolio_value, 2)))

        return results

    # ── Stress tests ────────────────────────────────────────────────────────
    @staticmethod
    def _stress_test(portfolio_value: float) -> List[StressResult]:
        results: List[StressResult] = []
        for name, params in STRESS_SCENARIOS.items():
            shock = params["equity_shock"]
            adj_loss = shock * SPREAD_BETA
            loss_dollar = abs(adj_loss) * portfolio_value
            # Recovery estimation
            daily_mean = 0.0004
            recovery_needed = abs(adj_loss)
            rec_days = int(math.ceil(
                math.log(1 + recovery_needed) / math.log(1 + daily_mean)
            )) if daily_mean > 0 and recovery_needed > 0 else 0

            results.append(StressResult(
                scenario=name,
                equity_shock=shock,
                portfolio_loss_pct=round(adj_loss * 100, 1),
                portfolio_loss_dollar=round(loss_dollar, 2),
                recovery_days=rec_days,
                vix_peak=params["vix_peak"],
            ))
        return results

    # ── Greeks ──────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_greeks(
        greeks: Dict[str, float], spy_price: float,
    ) -> GreeksExposure:
        d = greeks.get("delta", 0)
        g = greeks.get("gamma", 0)
        t = greeks.get("theta", 0)
        v = greeks.get("vega", 0)
        return GreeksExposure(
            delta=d, gamma=g, theta=t, vega=v,
            delta_dollars=round(d * spy_price * 100, 2),
            theta_daily=round(t * 100, 2),  # theta per day in dollars
        )

    # ── Concentration ───────────────────────────────────────────────────────
    @staticmethod
    def _compute_concentration(
        positions: Dict[str, float],
        sectors: Optional[Dict[str, str]] = None,
    ) -> ConcentrationRisk:
        if not positions:
            return ConcentrationRisk(0, "", 0, 0, {}, False)

        weights = np.array(list(positions.values()))
        abs_w = np.abs(weights)
        total = abs_w.sum()
        if total < 1e-9:
            return ConcentrationRisk(0, "", 0, 0, {}, False)

        fracs = abs_w / total
        max_pct = float(fracs.max()) * 100
        max_name = list(positions.keys())[int(np.argmax(fracs))]
        hhi = float(np.sum(fracs ** 2))
        top3 = float(np.sort(fracs)[-3:].sum()) * 100 if len(fracs) >= 3 else 100.0

        sector_exp: Dict[str, float] = {}
        if sectors:
            for name, w in positions.items():
                sec = sectors.get(name, "other")
                sector_exp[sec] = sector_exp.get(sec, 0) + abs(w)
            # Normalise
            sec_total = sum(sector_exp.values()) or 1
            sector_exp = {k: round(v / sec_total * 100, 1) for k, v in sector_exp.items()}

        is_conc = hhi > 0.25 or max_pct > 20

        return ConcentrationRisk(
            max_position_pct=round(max_pct, 1),
            max_position_name=max_name,
            herfindahl=round(hhi, 4),
            top3_pct=round(top3, 1),
            sector_exposures=sector_exp,
            is_concentrated=is_conc,
        )

    # ── Margin ──────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_margin(required: float, available: float) -> MarginState:
        util = required / available * 100 if available > 0 else 0
        excess = available - required
        call_dist = excess / available * 100 if available > 0 else 0
        return MarginState(
            total_margin_required=round(required, 2),
            margin_available=round(available, 2),
            utilisation_pct=round(util, 1),
            excess_margin=round(excess, 2),
            margin_call_distance_pct=round(call_dist, 1),
        )

    # ── Correlation matrix ──────────────────────────────────────────────────
    @staticmethod
    def _correlation_matrix(returns: Dict[str, pd.Series]) -> Optional[pd.DataFrame]:
        if len(returns) < 2:
            return None
        df = pd.DataFrame(returns)
        common = df.dropna()
        if len(common) < 20:
            return None
        return common.corr().round(4)

    # ── Portfolio returns ───────────────────────────────────────────────────
    @staticmethod
    def _portfolio_returns(
        returns: Dict[str, pd.Series], weights: Dict[str, float],
    ) -> np.ndarray:
        if not returns:
            return np.array([])
        df = pd.DataFrame(returns).dropna()
        if df.empty:
            return np.array([])
        w = np.array([weights.get(c, 0) for c in df.columns])
        return (df.values @ w).astype(float)

    # ── HTML report ─────────────────────────────────────────────────────────
    def generate_report(
        self,
        result: RiskDashboardResult,
        output_path: str = "reports/risk_dashboard.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        return path

    def _build_html(self, r: RiskDashboardResult) -> str:
        cards = self._html_cards(r)
        var_tbl = self._html_var(r.var_estimates)
        stress_tbl = self._html_stress(r.stress_results)
        greeks_tbl = self._html_greeks(r.greeks)
        conc_tbl = self._html_concentration(r.concentration)
        margin_tbl = self._html_margin(r.margin)
        corr_tbl = self._html_correlation(r.correlation_matrix)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Portfolio Risk Dashboard</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.5rem;margin-bottom:4px}}
h2{{font-size:1rem;color:#38bdf8;border-bottom:1px solid #334155;padding-bottom:4px;margin:20px 0 10px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin-bottom:20px}}
.card{{background:#1e293b;border-radius:8px;padding:14px}}
.card .lbl{{font-size:.7rem;color:#94a3b8;text-transform:uppercase}}
.card .val{{font-size:1.2rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:.83rem;margin-bottom:14px}}
th,td{{padding:6px 10px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;background:#1e293b}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}.warn{{color:#fbbf24}}
</style>
</head>
<body>
<h1>Portfolio Risk Dashboard</h1>
<p class="sub">{r.generated_at} &middot; Portfolio: ${r.portfolio_value:,.0f}</p>
{cards}{var_tbl}{stress_tbl}{greeks_tbl}{conc_tbl}{margin_tbl}{corr_tbl}
</body></html>"""

    @staticmethod
    def _html_cards(r: RiskDashboardResult) -> str:
        # Headline VaR (historical 95%)
        var95 = next((v for v in r.var_estimates if v.method == "historical" and v.confidence == 0.95), None)
        var99 = next((v for v in r.var_estimates if v.method == "historical" and v.confidence == 0.99), None)
        worst_stress = max(r.stress_results, key=lambda s: abs(s.portfolio_loss_pct)) if r.stress_results else None
        conc = r.concentration
        margin = r.margin
        return f"""<div class="grid">
<div class="card"><div class="lbl">VaR 95%</div><div class="val neg">${var95.var:,.0f}</div></div>
<div class="card"><div class="lbl">CVaR 95%</div><div class="val neg">${var95.cvar:,.0f}</div></div>
<div class="card"><div class="lbl">VaR 99%</div><div class="val neg">${var99.var:,.0f}</div></div>
<div class="card"><div class="lbl">Worst Stress</div><div class="val neg">{worst_stress.scenario if worst_stress else 'N/A'}</div></div>
<div class="card"><div class="lbl">Max Position</div><div class="val">{f'{conc.max_position_pct:.0f}%' if conc else 'N/A'}</div></div>
<div class="card"><div class="lbl">Margin Used</div><div class="val {'warn' if margin and margin.utilisation_pct > 70 else ''}">{f'{margin.utilisation_pct:.0f}%' if margin else 'N/A'}</div></div>
</div>""" if var95 and var99 else ""

    @staticmethod
    def _html_var(estimates: List[VaREstimate]) -> str:
        if not estimates:
            return ""
        rows = ""
        for v in estimates:
            rows += f"<tr><td>{v.method}</td><td>{v.confidence:.0%}</td><td class='neg'>${v.var:,.0f}</td><td class='neg'>${v.cvar:,.0f}</td></tr>"
        return f"<h2>Value-at-Risk / CVaR</h2><table><thead><tr><th>Method</th><th>Conf</th><th>VaR</th><th>CVaR</th></tr></thead><tbody>{rows}</tbody></table>"

    @staticmethod
    def _html_stress(results: List[StressResult]) -> str:
        if not results:
            return ""
        rows = ""
        for s in sorted(results, key=lambda x: x.portfolio_loss_pct):
            rows += (f"<tr><td>{s.scenario}</td><td class='neg'>{s.equity_shock:.0%}</td>"
                     f"<td class='neg'>{s.portfolio_loss_pct:.1f}%</td>"
                     f"<td class='neg'>${s.portfolio_loss_dollar:,.0f}</td>"
                     f"<td>{s.recovery_days}d</td><td>{s.vix_peak}</td></tr>")
        return f"<h2>Stress Scenarios</h2><table><thead><tr><th>Scenario</th><th>Shock</th><th>Portfolio DD</th><th>Loss $</th><th>Recovery</th><th>VIX Peak</th></tr></thead><tbody>{rows}</tbody></table>"

    @staticmethod
    def _html_greeks(g: Optional[GreeksExposure]) -> str:
        if not g:
            return ""
        return f"""<h2>Greeks Exposure</h2><table>
<tbody><tr><td>Delta</td><td>{g.delta:.2f}</td><td>${g.delta_dollars:,.0f}</td></tr>
<tr><td>Gamma</td><td>{g.gamma:.4f}</td><td></td></tr>
<tr><td>Theta</td><td>{g.theta:.2f}</td><td>${g.theta_daily:,.0f}/day</td></tr>
<tr><td>Vega</td><td>{g.vega:.2f}</td><td></td></tr></tbody></table>"""

    @staticmethod
    def _html_concentration(c: Optional[ConcentrationRisk]) -> str:
        if not c:
            return ""
        cls = "neg" if c.is_concentrated else "pos"
        sec_rows = "".join(f"<tr><td>{s}</td><td>{p:.1f}%</td></tr>" for s, p in sorted(c.sector_exposures.items()))
        return f"""<h2>Concentration Risk</h2><table><tbody>
<tr><td>Max Position</td><td>{c.max_position_name}</td><td>{c.max_position_pct:.1f}%</td></tr>
<tr><td>Herfindahl</td><td class="{cls}">{c.herfindahl:.4f}</td><td>{'Concentrated' if c.is_concentrated else 'Diversified'}</td></tr>
<tr><td>Top 3</td><td colspan="2">{c.top3_pct:.1f}%</td></tr>
</tbody></table>{'<h2>Sector Exposure</h2><table><thead><tr><th>Sector</th><th>%</th></tr></thead><tbody>' + sec_rows + '</tbody></table>' if sec_rows else ''}"""

    @staticmethod
    def _html_margin(m: Optional[MarginState]) -> str:
        if not m:
            return ""
        cls = "neg" if m.utilisation_pct > 80 else "warn" if m.utilisation_pct > 60 else "pos"
        return f"""<h2>Margin</h2><table><tbody>
<tr><td>Required</td><td>${m.total_margin_required:,.0f}</td></tr>
<tr><td>Available</td><td>${m.margin_available:,.0f}</td></tr>
<tr><td>Utilisation</td><td class="{cls}">{m.utilisation_pct:.1f}%</td></tr>
<tr><td>Excess</td><td>${m.excess_margin:,.0f}</td></tr>
<tr><td>Margin Call Distance</td><td>{m.margin_call_distance_pct:.1f}%</td></tr></tbody></table>"""

    @staticmethod
    def _html_correlation(corr: Optional[pd.DataFrame]) -> str:
        if corr is None or corr.empty:
            return ""
        cols = list(corr.columns)
        hdr = "".join(f"<th>{c}</th>" for c in cols)
        rows = ""
        for idx in cols:
            cells = ""
            for c in cols:
                v = corr.loc[idx, c]
                cls = "pos" if v > 0.7 and idx != c else "neg" if v < -0.3 else ""
                cells += f'<td class="{cls}">{v:.3f}</td>'
            rows += f"<tr><td><strong>{idx}</strong></td>{cells}</tr>"
        return f"<h2>Strategy Correlations</h2><table><thead><tr><th></th>{hdr}</tr></thead><tbody>{rows}</tbody></table>"


def _z_score(confidence: float) -> float:
    table = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
    return table.get(confidence, 1.645)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
