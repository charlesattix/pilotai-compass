"""
compass/position_risk.py — Position-level risk attribution and margin estimation.

Computes per-position Greeks, aggregates portfolio-level exposure with
netting, estimates margin requirements under Reg-T and portfolio margin,
and checks against configurable risk limits.

Usage::

    from compass.position_risk import PositionRiskAnalyzer, Position

    positions = [
        Position(ticker="SPY", short_strike=445, long_strike=440,
                 spread_type="bull_put", contracts=2, credit=0.65,
                 dte=21, underlying_price=450, iv=0.20),
    ]
    analyzer = PositionRiskAnalyzer()
    report = analyzer.analyze(positions)
    analyzer.generate_report(positions, "reports/position_risk.html")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from compass.greeks_sensitivity import (
    bs_put_price,
    bs_call_price,
    compute_greeks,
    put_spread_value,
    call_spread_value,
)

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class Position:
    """One open spread position."""
    ticker: str
    short_strike: float
    long_strike: float
    spread_type: str = "bull_put"   # "bull_put" or "bear_call"
    contracts: int = 1
    credit: float = 0.65           # credit received per spread
    dte: int = 21
    underlying_price: float = 450.0
    iv: float = 0.20               # implied volatility (annualised decimal)
    regime: str = "bull"


@dataclass
class PositionGreeks:
    """Greeks for a single position (per-contract, then scaled)."""
    ticker: str
    contracts: int
    # Per-contract
    delta: float = 0.0
    gamma: float = 0.0
    theta_day: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    # Scaled by contracts * 100 (notional multiplier)
    delta_dollars: float = 0.0
    gamma_dollars: float = 0.0
    theta_dollars_day: float = 0.0
    vega_dollars: float = 0.0
    # Risk
    max_loss: float = 0.0
    current_value: float = 0.0
    pnl: float = 0.0


@dataclass
class PortfolioGreeks:
    """Aggregate portfolio-level Greeks with netting."""
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta_day: float = 0.0
    net_vega: float = 0.0
    net_rho: float = 0.0
    # Dollar-weighted
    delta_dollars: float = 0.0
    gamma_dollars: float = 0.0
    theta_dollars_day: float = 0.0
    vega_dollars: float = 0.0
    # Summary
    total_max_loss: float = 0.0
    total_pnl: float = 0.0
    n_positions: int = 0


@dataclass
class MarginEstimate:
    """Margin requirements under Reg-T and portfolio margin."""
    reg_t_margin: float = 0.0
    portfolio_margin: float = 0.0
    margin_utilization_pct: float = 0.0   # as % of account
    account_value: float = 100_000.0


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_net_delta: float = 50.0           # max absolute net delta
    max_net_gamma: float = 5.0            # max absolute net gamma
    max_net_vega: float = 5000.0          # max absolute net vega ($)
    max_margin_pct: float = 60.0          # max margin utilization %
    max_total_loss_pct: float = 30.0      # max loss as % of account


@dataclass
class LimitCheck:
    """Result of a risk limit check."""
    name: str
    current: float
    limit: float
    breached: bool
    utilization_pct: float


@dataclass
class RiskReport:
    """Complete risk analysis output."""
    position_greeks: List[PositionGreeks] = field(default_factory=list)
    portfolio_greeks: PortfolioGreeks = field(default_factory=PortfolioGreeks)
    margin: MarginEstimate = field(default_factory=MarginEstimate)
    limit_checks: List[LimitCheck] = field(default_factory=list)
    any_breach: bool = False


# ── Greeks computation ───────────────────────────────────────────────────


def compute_position_greeks(pos: Position) -> PositionGreeks:
    """Compute Greeks for one position."""
    S = pos.underlying_price
    K_short = pos.short_strike
    K_long = pos.long_strike
    T = max(pos.dte, 1) / 252.0
    sigma = max(pos.iv, 0.01)
    n = pos.contracts
    multiplier = n * 100  # options multiplier

    g = compute_greeks(S, K_short, K_long, T, sigma, spread_type=pos.spread_type)

    # Current spread value
    if "put" in pos.spread_type.lower():
        current_value = put_spread_value(S, K_short, K_long, T, sigma)
    else:
        current_value = call_spread_value(S, K_short, K_long, T, sigma)

    # Max loss = (spread_width - credit) * 100 * contracts
    width = abs(K_short - K_long)
    max_loss = (width - pos.credit) * 100 * n
    pnl = (pos.credit - current_value) * 100 * n  # credit seller's P&L

    return PositionGreeks(
        ticker=pos.ticker,
        contracts=n,
        delta=round(g.delta, 6),
        gamma=round(g.gamma, 6),
        theta_day=round(g.theta, 6),
        vega=round(g.vega, 4),
        rho=round(g.rho, 4),
        delta_dollars=round(g.delta * multiplier * S, 2),
        gamma_dollars=round(g.gamma * multiplier * S * S * 0.01, 2),
        theta_dollars_day=round(g.theta * multiplier, 2),
        vega_dollars=round(g.vega * multiplier, 2),
        max_loss=round(max_loss, 2),
        current_value=round(current_value, 4),
        pnl=round(pnl, 2),
    )


def aggregate_greeks(position_greeks: List[PositionGreeks]) -> PortfolioGreeks:
    """Aggregate position Greeks into portfolio-level with netting."""
    if not position_greeks:
        return PortfolioGreeks()

    return PortfolioGreeks(
        net_delta=round(sum(pg.delta * pg.contracts for pg in position_greeks), 4),
        net_gamma=round(sum(pg.gamma * pg.contracts for pg in position_greeks), 6),
        net_theta_day=round(sum(pg.theta_day * pg.contracts for pg in position_greeks), 4),
        net_vega=round(sum(pg.vega * pg.contracts for pg in position_greeks), 4),
        net_rho=round(sum(pg.rho * pg.contracts for pg in position_greeks), 4),
        delta_dollars=round(sum(pg.delta_dollars for pg in position_greeks), 2),
        gamma_dollars=round(sum(pg.gamma_dollars for pg in position_greeks), 2),
        theta_dollars_day=round(sum(pg.theta_dollars_day for pg in position_greeks), 2),
        vega_dollars=round(sum(pg.vega_dollars for pg in position_greeks), 2),
        total_max_loss=round(sum(pg.max_loss for pg in position_greeks), 2),
        total_pnl=round(sum(pg.pnl for pg in position_greeks), 2),
        n_positions=len(position_greeks),
    )


# ── Margin estimation ────────────────────────────────────────────────────


def estimate_margin(
    positions: List[Position],
    account_value: float = 100_000.0,
) -> MarginEstimate:
    """Estimate margin requirements under Reg-T and portfolio margin.

    Reg-T for credit spreads: margin = max_loss = (width - credit) * 100 * contracts
    Portfolio margin: ~50-70% of Reg-T for defined-risk spreads (uses a
    stress-test-based model; we approximate at 60%).
    """
    reg_t = 0.0
    for pos in positions:
        width = abs(pos.short_strike - pos.long_strike)
        max_loss_per = (width - pos.credit) * 100
        reg_t += max(max_loss_per, 0) * pos.contracts

    portfolio_margin = reg_t * 0.60  # portfolio margin approximation

    utilization = (reg_t / account_value * 100) if account_value > 0 else 0.0

    return MarginEstimate(
        reg_t_margin=round(reg_t, 2),
        portfolio_margin=round(portfolio_margin, 2),
        margin_utilization_pct=round(utilization, 1),
        account_value=account_value,
    )


# ── Risk limit checker ───────────────────────────────────────────────────


def check_limits(
    portfolio: PortfolioGreeks,
    margin: MarginEstimate,
    limits: RiskLimits,
) -> List[LimitCheck]:
    """Check portfolio Greeks against risk limits."""
    checks = []

    def _check(name: str, current: float, limit: float) -> LimitCheck:
        breached = abs(current) > limit
        util = abs(current) / limit * 100 if limit > 0 else 0.0
        return LimitCheck(
            name=name, current=round(current, 2), limit=limit,
            breached=breached, utilization_pct=round(util, 1),
        )

    checks.append(_check("Net Delta", portfolio.net_delta, limits.max_net_delta))
    checks.append(_check("Net Gamma", portfolio.net_gamma, limits.max_net_gamma))
    checks.append(_check("Net Vega ($)", portfolio.vega_dollars, limits.max_net_vega))
    checks.append(_check(
        "Margin Utilization (%)",
        margin.margin_utilization_pct,
        limits.max_margin_pct,
    ))
    checks.append(_check(
        "Total Max Loss (%)",
        portfolio.total_max_loss / max(margin.account_value, 1) * 100,
        limits.max_total_loss_pct,
    ))

    return checks


# ── Analyzer ─────────────────────────────────────────────────────────────


class PositionRiskAnalyzer:
    """Analyse position-level risk, aggregate portfolio Greeks, estimate margin."""

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        account_value: float = 100_000.0,
    ) -> None:
        self.limits = limits or RiskLimits()
        self.account_value = account_value

    def analyze(self, positions: List[Position]) -> RiskReport:
        """Run full risk analysis on a set of positions."""
        if not positions:
            return RiskReport()

        pos_greeks = [compute_position_greeks(p) for p in positions]
        port_greeks = aggregate_greeks(pos_greeks)
        margin = estimate_margin(positions, self.account_value)
        limit_checks = check_limits(port_greeks, margin, self.limits)
        any_breach = any(c.breached for c in limit_checks)

        return RiskReport(
            position_greeks=pos_greeks,
            portfolio_greeks=port_greeks,
            margin=margin,
            limit_checks=limit_checks,
            any_breach=any_breach,
        )

    def generate_report(
        self,
        positions: List[Position],
        path: Optional[str] = None,
    ) -> str:
        """Generate HTML risk report."""
        report = self.analyze(positions)
        html = self._render_html(report, positions)
        if path:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html)
        return html

    def _render_html(self, r: RiskReport, positions: List[Position]) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        pg = r.portfolio_greeks
        mg = r.margin

        # Cards
        breach_color = "#dc2626" if r.any_breach else "#16a34a"
        breach_label = "BREACH" if r.any_breach else "OK"
        cards = (
            f'<div class="cards">'
            f'<div class="card"><div class="ct">Positions</div><div class="cv">{pg.n_positions}</div></div>'
            f'<div class="card"><div class="ct">Net Delta</div><div class="cv">{pg.net_delta:+.2f}</div></div>'
            f'<div class="card"><div class="ct">Theta/Day</div><div class="cv">${pg.theta_dollars_day:+,.0f}</div></div>'
            f'<div class="card"><div class="ct">Margin Used</div><div class="cv">{mg.margin_utilization_pct:.0f}%</div></div>'
            f'<div class="card"><div class="ct">Max Loss</div><div class="cv">${pg.total_max_loss:,.0f}</div></div>'
            f'<div class="card"><div class="ct">Limits</div>'
            f'<div class="cv" style="color:{breach_color}">{breach_label}</div></div>'
            f'</div>'
        )

        # Position Greeks table
        pos_rows = ""
        for p, g in zip(positions, r.position_greeks):
            pos_rows += (
                f'<tr><td style="font-weight:600">{_esc(p.ticker)}</td>'
                f'<td>{p.spread_type}</td><td>{g.contracts}</td>'
                f'<td>{g.delta:+.4f}</td><td>{g.gamma:.6f}</td>'
                f'<td>${g.theta_dollars_day:+.2f}</td>'
                f'<td>${g.vega_dollars:+.0f}</td>'
                f'<td>${g.max_loss:,.0f}</td>'
                f'<td>${g.pnl:+,.0f}</td></tr>'
            )
        pos_table = (
            f'<table><thead><tr><th>Ticker</th><th>Type</th><th>Qty</th>'
            f'<th>Delta</th><th>Gamma</th><th>Theta/d</th>'
            f'<th>Vega($)</th><th>Max Loss</th><th>P&L</th></tr></thead>'
            f'<tbody>{pos_rows}</tbody></table>'
        )

        # Portfolio aggregate
        agg_table = (
            f'<table><thead><tr><th>Greek</th><th>Net Value</th><th>Dollar Value</th></tr></thead>'
            f'<tbody>'
            f'<tr><td>Delta</td><td>{pg.net_delta:+.4f}</td><td>${pg.delta_dollars:+,.0f}</td></tr>'
            f'<tr><td>Gamma</td><td>{pg.net_gamma:+.6f}</td><td>${pg.gamma_dollars:+,.0f}</td></tr>'
            f'<tr><td>Theta/day</td><td>{pg.net_theta_day:+.4f}</td><td>${pg.theta_dollars_day:+,.0f}</td></tr>'
            f'<tr><td>Vega</td><td>{pg.net_vega:+.4f}</td><td>${pg.vega_dollars:+,.0f}</td></tr>'
            f'</tbody></table>'
        )

        # Margin table
        margin_table = (
            f'<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
            f'<tr><td>Reg-T Margin Required</td><td>${mg.reg_t_margin:,.0f}</td></tr>'
            f'<tr><td>Portfolio Margin (est.)</td><td>${mg.portfolio_margin:,.0f}</td></tr>'
            f'<tr><td>Account Value</td><td>${mg.account_value:,.0f}</td></tr>'
            f'<tr><td>Utilization</td><td>{mg.margin_utilization_pct:.1f}%</td></tr>'
            f'</tbody></table>'
        )

        # Limit checks
        limit_rows = ""
        for c in r.limit_checks:
            color = "#dc2626" if c.breached else "#16a34a"
            icon = "BREACH" if c.breached else "OK"
            limit_rows += (
                f'<tr><td>{c.name}</td><td>{c.current:.2f}</td><td>{c.limit:.1f}</td>'
                f'<td>{c.utilization_pct:.0f}%</td>'
                f'<td style="color:{color};font-weight:600">{icon}</td></tr>'
            )
        limit_table = (
            f'<table><thead><tr><th>Limit</th><th>Current</th><th>Max</th>'
            f'<th>Util%</th><th>Status</th></tr></thead>'
            f'<tbody>{limit_rows}</tbody></table>'
        )

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Position Risk Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:#f8fafc;color:#1e293b;line-height:1.5;padding:24px;max-width:1200px;margin:0 auto}}
h1{{font-size:1.6em;font-weight:700;margin-bottom:4px}}
h2{{font-size:1.15em;font-weight:600;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}}
.sub{{color:#64748b;font-size:0.9em;margin-bottom:20px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;min-width:130px;flex:1}}
.ct{{font-size:0.75em;color:#64748b;text-transform:uppercase;letter-spacing:.5px}}
.cv{{font-size:1.4em;font-weight:700}}
table{{border-collapse:collapse;width:100%;font-size:0.85em;margin-bottom:16px}}
th{{background:#f1f5f9;padding:6px 8px;text-align:center;font-weight:600;border-bottom:2px solid #e2e8f0}}
td{{padding:5px 8px;border-bottom:1px solid #f1f5f9;text-align:center}}
hr{{margin:28px 0;border:none;border-top:1px solid #e2e8f0}}
</style></head><body>

<h1>Position Risk Report</h1>
<p class="sub">{pg.n_positions} positions &middot; {now}</p>
{cards}

<h2>Position-Level Greeks</h2>
{pos_table}

<h2>Portfolio Aggregate Greeks</h2>
{agg_table}

<h2>Margin Estimation</h2>
{margin_table}

<h2>Risk Limit Checks</h2>
{limit_table}

<hr><p style="font-size:0.75em;color:#94a3b8">Generated by <code>compass/position_risk.py</code></p>
</body></html>"""


def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
