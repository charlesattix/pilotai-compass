"""
Real-time P&L estimator with Black-Scholes Greek attribution.

Estimates current and projected P&L for open credit spread positions by
decomposing value changes into theta decay, delta, vega, and gamma
contributions using the Black-Scholes model.

Usage::

    from compass.realtime_pnl import RealtimePnLEstimator, SpreadPosition
    pos = SpreadPosition(
        short_strike=550, long_strike=545, expiration_days=21,
        entry_credit=1.50, contracts=2, underlying_price=560,
        iv=0.18, direction="bull_put",
    )
    estimator = RealtimePnLEstimator()
    estimator.add_position(pos)
    snapshot = estimator.portfolio_snapshot(current_price=558, current_iv=0.20)
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "realtime_pnl.html"

RISK_FREE_RATE = 0.05
TRADING_DAYS_YEAR = 252


# ── Black-Scholes primitives ────────────────────────────────────────────


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put",
) -> float:
    """Black-Scholes option price. T in years, sigma annualized."""
    if T <= 0 or sigma <= 0 or S <= 0:
        intrinsic = max(K - S, 0) if option_type == "put" else max(S - K, 0)
        return intrinsic
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put",
) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        if option_type == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if option_type == "put":
        return norm.cdf(d1) - 1.0
    return norm.cdf(d1)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put",
) -> float:
    """Theta in $/day (negative = time decay costs the holder)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    common = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type == "put":
        return (common + r * K * math.exp(-r * T) * norm.cdf(-d2)) / TRADING_DAYS_YEAR
    return (common - r * K * math.exp(-r * T) * norm.cdf(d2)) / TRADING_DAYS_YEAR


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega: price change per 1% (0.01) change in IV."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * norm.pdf(d1) * math.sqrt(T) * 0.01


# ── Spread pricing ──────────────────────────────────────────────────────


def spread_value(
    S: float, short_K: float, long_K: float, T: float, r: float, sigma: float,
    direction: str = "bull_put",
) -> float:
    """Current value of a vertical spread (positive = credit seller owes)."""
    opt_type = "put" if "put" in direction else "call"
    short_val = bs_price(S, short_K, T, r, sigma, opt_type)
    long_val = bs_price(S, long_K, T, r, sigma, opt_type)
    return short_val - long_val  # net value of spread


def spread_greeks(
    S: float, short_K: float, long_K: float, T: float, r: float, sigma: float,
    direction: str = "bull_put",
) -> Dict[str, float]:
    """Net Greeks for a vertical spread (short - long)."""
    opt_type = "put" if "put" in direction else "call"
    return {
        "delta": bs_delta(S, short_K, T, r, sigma, opt_type) - bs_delta(S, long_K, T, r, sigma, opt_type),
        "gamma": bs_gamma(S, short_K, T, r, sigma) - bs_gamma(S, long_K, T, r, sigma),
        "theta": bs_theta(S, short_K, T, r, sigma, opt_type) - bs_theta(S, long_K, T, r, sigma, opt_type),
        "vega": bs_vega(S, short_K, T, r, sigma) - bs_vega(S, long_K, T, r, sigma),
    }


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class SpreadPosition:
    """An open credit spread position."""
    short_strike: float
    long_strike: float
    expiration_days: int        # days to expiration
    entry_credit: float         # credit received per contract
    contracts: int = 1
    underlying_price: float = 0.0  # price at entry
    iv: float = 0.20           # IV at entry
    direction: str = "bull_put"  # "bull_put" or "bear_call"
    label: str = ""
    stop_loss_mult: float = 3.5
    profit_target_pct: float = 0.50


@dataclass
class PositionPnL:
    """Estimated P&L for one position."""
    label: str
    current_value: float        # current spread value (what it costs to close)
    entry_credit: float
    unrealized_pnl: float       # entry_credit - current_value (per contract ×100)
    unrealized_pnl_pct: float   # as % of max risk
    # Greek attribution
    theta_pnl: float
    delta_pnl: float
    vega_pnl: float
    gamma_pnl: float
    residual_pnl: float
    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    # Alerts
    near_stop: bool
    near_target: bool
    days_to_expiry: int
    # Projection
    projected_pnl_at_expiry: float


@dataclass
class PortfolioSnapshot:
    """Aggregate portfolio P&L snapshot."""
    timestamp: str
    positions: List[PositionPnL]
    total_unrealized_pnl: float
    total_theta_pnl: float
    total_delta_pnl: float
    total_vega_pnl: float
    total_gamma_pnl: float
    portfolio_delta: float
    portfolio_theta: float
    portfolio_vega: float
    n_near_stop: int
    n_near_target: int


# ── RealtimePnLEstimator ────────────────────────────────────────────────


class RealtimePnLEstimator:
    """Estimates current P&L using Black-Scholes Greek attribution.

    Args:
        risk_free_rate: Annual risk-free rate.
        multiplier: Option contract multiplier (100 for equity options).
    """

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE, multiplier: int = 100):
        self.r = risk_free_rate
        self.multiplier = multiplier
        self.positions: List[SpreadPosition] = []

    def add_position(self, position: SpreadPosition) -> None:
        self.positions.append(position)

    def clear_positions(self) -> None:
        self.positions.clear()

    def estimate_position(
        self,
        pos: SpreadPosition,
        current_price: float,
        current_iv: float,
        days_elapsed: int = 0,
    ) -> PositionPnL:
        """Estimate P&L and Greek attribution for a single position."""
        T_entry = pos.expiration_days / TRADING_DAYS_YEAR
        T_now = max(0, (pos.expiration_days - days_elapsed)) / TRADING_DAYS_YEAR
        mult = self.multiplier * pos.contracts

        # Current spread value
        cv = spread_value(current_price, pos.short_strike, pos.long_strike,
                          T_now, self.r, current_iv, pos.direction)
        # Entry spread value
        ev = spread_value(pos.underlying_price, pos.short_strike, pos.long_strike,
                          T_entry, self.r, pos.iv, pos.direction)

        # Unrealized P&L: credit received minus cost to close
        # For credit spreads, entry_credit > 0, and current_value increases when spread moves against us
        unrealized = (pos.entry_credit - cv) * mult
        spread_width = abs(pos.short_strike - pos.long_strike)
        max_risk = (spread_width - pos.entry_credit) * mult
        unrealized_pct = (unrealized / max_risk * 100) if max_risk > 0 else 0.0

        # Greeks at current state
        greeks = spread_greeks(current_price, pos.short_strike, pos.long_strike,
                               T_now, self.r, current_iv, pos.direction)

        # Attribution: decompose P&L into Greek components
        price_change = current_price - pos.underlying_price
        iv_change = current_iv - pos.iv
        days_remaining = pos.expiration_days - days_elapsed

        # Theta P&L: cumulative theta decay over days_elapsed
        # Approximate as average theta × days
        theta_mid = spread_greeks(
            (current_price + pos.underlying_price) / 2,
            pos.short_strike, pos.long_strike,
            (T_entry + T_now) / 2, self.r, (pos.iv + current_iv) / 2, pos.direction,
        )["theta"]
        theta_pnl = -theta_mid * days_elapsed * mult  # negative theta = we collect as sellers

        # Delta P&L
        delta_entry = spread_greeks(
            pos.underlying_price, pos.short_strike, pos.long_strike,
            T_entry, self.r, pos.iv, pos.direction,
        )["delta"]
        delta_pnl = delta_entry * price_change * mult

        # Vega P&L
        vega_entry = spread_greeks(
            pos.underlying_price, pos.short_strike, pos.long_strike,
            T_entry, self.r, pos.iv, pos.direction,
        )["vega"]
        vega_pnl = vega_entry * (iv_change * 100) * mult  # vega is per 1% IV change

        # Gamma P&L (second-order delta correction)
        gamma_entry = spread_greeks(
            pos.underlying_price, pos.short_strike, pos.long_strike,
            T_entry, self.r, pos.iv, pos.direction,
        )["gamma"]
        gamma_pnl = 0.5 * gamma_entry * price_change**2 * mult

        residual = unrealized - theta_pnl - delta_pnl - vega_pnl - gamma_pnl

        # Alerts
        stop_level = -pos.entry_credit * pos.stop_loss_mult * mult
        target_level = pos.entry_credit * pos.profit_target_pct * mult
        near_stop = bool(unrealized <= stop_level * 0.8)  # within 80% of stop
        near_target = bool(unrealized >= target_level * 0.8)

        # Projected P&L at expiry (assume price stays, IV decays to 0)
        ev_at_expiry = spread_value(current_price, pos.short_strike, pos.long_strike,
                                     0.001, self.r, current_iv, pos.direction)
        projected = (pos.entry_credit - ev_at_expiry) * mult

        return PositionPnL(
            label=pos.label or f"{pos.direction} {pos.short_strike}/{pos.long_strike}",
            current_value=round(cv, 4),
            entry_credit=pos.entry_credit,
            unrealized_pnl=round(unrealized, 2),
            unrealized_pnl_pct=round(unrealized_pct, 2),
            theta_pnl=round(theta_pnl, 2),
            delta_pnl=round(delta_pnl, 2),
            vega_pnl=round(vega_pnl, 2),
            gamma_pnl=round(gamma_pnl, 2),
            residual_pnl=round(residual, 2),
            delta=round(greeks["delta"], 4),
            gamma=round(greeks["gamma"], 6),
            theta=round(greeks["theta"], 4),
            vega=round(greeks["vega"], 4),
            near_stop=near_stop,
            near_target=near_target,
            days_to_expiry=pos.expiration_days - days_elapsed,
            projected_pnl_at_expiry=round(projected, 2),
        )

    def portfolio_snapshot(
        self,
        current_price: float,
        current_iv: float,
        days_elapsed: int = 0,
    ) -> PortfolioSnapshot:
        """Estimate P&L for all positions."""
        results = [
            self.estimate_position(p, current_price, current_iv, days_elapsed)
            for p in self.positions
        ]

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            positions=results,
            total_unrealized_pnl=round(sum(r.unrealized_pnl for r in results), 2),
            total_theta_pnl=round(sum(r.theta_pnl for r in results), 2),
            total_delta_pnl=round(sum(r.delta_pnl for r in results), 2),
            total_vega_pnl=round(sum(r.vega_pnl for r in results), 2),
            total_gamma_pnl=round(sum(r.gamma_pnl for r in results), 2),
            portfolio_delta=round(sum(r.delta * p.contracts for r, p in zip(results, self.positions)), 4),
            portfolio_theta=round(sum(r.theta * p.contracts for r, p in zip(results, self.positions)), 4),
            portfolio_vega=round(sum(r.vega * p.contracts for r, p in zip(results, self.positions)), 4),
            n_near_stop=sum(1 for r in results if r.near_stop),
            n_near_target=sum(1 for r in results if r.near_target),
        )

    # ── HTML Dashboard ───────────────────────────────────────────────

    def generate_dashboard(
        self,
        current_price: float,
        current_iv: float,
        days_elapsed: int = 0,
        output: str = str(DEFAULT_OUTPUT),
    ) -> str:
        snap = self.portfolio_snapshot(current_price, current_iv, days_elapsed)
        charts = self._render_charts(snap)
        html = self._build_html(snap, charts, current_price, current_iv)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out.resolve())

    def _render_charts(self, snap: PortfolioSnapshot) -> Dict[str, str]:
        import matplotlib
        matplotlib.use("Agg")
        charts: Dict[str, str] = {}
        if snap.positions:
            charts["attribution"] = self._chart_attribution(snap)
            charts["positions"] = self._chart_positions(snap)
        return charts

    def _fig_to_b64(self, fig) -> str:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _chart_attribution(self, snap: PortfolioSnapshot) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        components = {
            "Theta": snap.total_theta_pnl,
            "Delta": snap.total_delta_pnl,
            "Vega": snap.total_vega_pnl,
            "Gamma": snap.total_gamma_pnl,
        }
        names = list(components.keys())
        vals = list(components.values())
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in vals]

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="white")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("P&L ($)")
        ax.set_title("Portfolio P&L Attribution by Greek", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(i, v + abs(v) * 0.05 * (1 if v >= 0 else -1.5),
                    f"${v:,.0f}", ha="center", fontsize=9, fontweight="bold")
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_positions(self, snap: PortfolioSnapshot) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not snap.positions:
            return ""
        names = [p.label[:25] for p in snap.positions]
        pnls = [p.unrealized_pnl for p in snap.positions]
        colors = ["#16a34a" if v >= 0 else "#dc2626" for v in pnls]

        fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.5))
        ax.bar(range(len(names)), pnls, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel("Unrealized P&L ($)")
        ax.set_title("Position P&L", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(
        self, snap: PortfolioSnapshot, charts: Dict[str, str],
        current_price: float, current_iv: float,
    ) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        pnl_cls = "good" if snap.total_unrealized_pnl >= 0 else "bad"

        def _img(key):
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ''

        # Position cards
        pos_cards = ""
        for p in snap.positions:
            cls = "good" if p.unrealized_pnl >= 0 else "bad"
            alert = ""
            if p.near_stop:
                alert = '<span class="alert stop">NEAR STOP</span>'
            elif p.near_target:
                alert = '<span class="alert target">NEAR TARGET</span>'
            pos_cards += f"""
            <div class="pos-card">
              <div class="pos-header"><strong>{p.label}</strong> {alert}</div>
              <div class="pos-kpis">
                <span class="{cls}">${p.unrealized_pnl:,.0f} ({p.unrealized_pnl_pct:+.1f}%)</span>
                · DTE: {p.days_to_expiry}d · Proj: ${p.projected_pnl_at_expiry:,.0f}
              </div>
              <div class="greek-bars">
                <div>θ: ${p.theta_pnl:,.0f}</div>
                <div>Δ: ${p.delta_pnl:,.0f}</div>
                <div>ν: ${p.vega_pnl:,.0f}</div>
                <div>γ: ${p.gamma_pnl:,.0f}</div>
              </div>
              <div class="pos-greeks">δ={p.delta:.3f} γ={p.gamma:.5f} θ={p.theta:.3f}/d ν={p.vega:.3f}</div>
            </div>"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Real-Time P&L</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  .pos-card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
               padding: 1em; margin: 0.8em 0; }}
  .pos-header {{ font-size: 1em; margin-bottom: 0.3em; }}
  .pos-kpis {{ font-size: 0.9em; margin: 0.2em 0; }}
  .greek-bars {{ display: flex; gap: 1em; font-size: 0.85em; color: #475569; margin: 0.3em 0; }}
  .pos-greeks {{ font-size: 0.78em; color: #94a3b8; }}
  .alert {{ padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 600; }}
  .alert.stop {{ background: #fecaca; color: #991b1b; }}
  .alert.target {{ background: #dcfce7; color: #166534; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Real-Time P&L Estimator</h1>
<div class="meta">{len(snap.positions)} positions · SPY={current_price:.2f} · IV={current_iv:.1%} · {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value {pnl_cls}">${snap.total_unrealized_pnl:,.0f}</div><div class="label">Total P&L</div></div>
  <div class="kpi"><div class="value">${snap.total_theta_pnl:,.0f}</div><div class="label">Theta P&L</div></div>
  <div class="kpi"><div class="value">${snap.total_delta_pnl:,.0f}</div><div class="label">Delta P&L</div></div>
  <div class="kpi"><div class="value">${snap.total_vega_pnl:,.0f}</div><div class="label">Vega P&L</div></div>
  <div class="kpi"><div class="value">{snap.portfolio_delta:.3f}</div><div class="label">Net Delta</div></div>
  <div class="kpi"><div class="value">{snap.n_near_stop}</div><div class="label">Near Stop</div></div>
  <div class="kpi"><div class="value">{snap.n_near_target}</div><div class="label">Near Target</div></div>
</div>

<h2>1. Greek Attribution</h2>
{_img("attribution")}

<h2>2. Position P&L</h2>
{_img("positions")}

<h2>3. Position Details</h2>
{pos_cards if pos_cards else '<p class="meta">No open positions</p>'}

<footer>Generated by <code>compass/realtime_pnl.py</code></footer>
</body></html>"""
        return html
