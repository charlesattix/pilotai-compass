"""
Real-time Greeks risk engine for the combined portfolio.

Portfolio-level delta/gamma/theta/vega exposure, delta-neutral hedging
suggestions, theta decay P&L attribution, gamma scalping opportunity
detection, and regime-conditional vega limits.

Builds on compass.greeks_calculator for per-position Greeks.

Usage::

    from compass.greeks_risk_engine import GreeksRiskEngine, LivePosition
    engine = GreeksRiskEngine()
    engine.add_position(LivePosition(...))
    snapshot = engine.snapshot()
    hedges = engine.hedging_suggestions()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from compass.greeks_calculator import (
    OptionGreeks,
    compute_option_greeks,
)


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class RiskConfig:
    # Delta limits
    max_portfolio_delta: float = 50.0
    delta_neutral_threshold: float = 10.0  # suggest hedge when |delta| > this
    # Gamma
    gamma_scalp_threshold: float = 5.0     # gamma > this → scalping opportunity
    max_portfolio_gamma: float = 100.0
    # Vega
    max_portfolio_vega: float = 500.0
    # Regime-conditional vega limits
    vega_limits_by_regime: Dict[str, float] = field(default_factory=lambda: {
        "bull": 500.0, "neutral": 400.0, "bear": 200.0,
        "high_vol": 100.0, "crash": 50.0,
    })
    # Theta
    min_daily_theta: float = -200.0        # alert if theta income below this


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class LivePosition:
    """An open position for real-time Greek tracking."""
    position_id: str
    strategy: str
    ticker: str
    option_type: str          # "call" or "put"
    direction: str            # "long" or "short"
    strike: float
    spread_strike: Optional[float]   # None = naked, else spread long leg
    underlying_price: float
    iv: float
    dte: float
    contracts: int
    rate: float = 0.045
    entry_credit: float = 0.0


@dataclass
class PositionGreeks:
    """Greeks for one live position."""
    position_id: str
    strategy: str
    ticker: str
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    net_value: float


@dataclass
class PortfolioSnapshot:
    """Portfolio-level Greeks snapshot."""
    timestamp: str
    total_delta: float
    total_gamma: float
    total_theta: float       # daily theta income (positive = earning)
    total_vega: float
    total_rho: float
    n_positions: int
    by_strategy: Dict[str, Dict[str, float]]
    by_ticker: Dict[str, Dict[str, float]]
    delta_neutral: bool      # |delta| < threshold
    theta_daily_pnl: float   # estimated daily theta P&L ($)


@dataclass
class HedgeSuggestion:
    """Delta-neutral hedging suggestion."""
    action: str              # "buy_shares", "sell_shares", "buy_put", "sell_call"
    ticker: str
    quantity: int            # shares or contracts
    estimated_delta: float   # delta of the hedge
    residual_delta: float    # portfolio delta after hedge
    reason: str
    urgency: str             # "low", "medium", "high"


@dataclass
class ThetaAttribution:
    """Theta decay P&L attribution per position."""
    position_id: str
    strategy: str
    daily_theta: float       # $ earned/lost per day
    pct_of_total: float      # fraction of portfolio theta
    days_remaining: float
    projected_total: float   # theta × days remaining


@dataclass
class GammaScalpOpportunity:
    """Gamma scalping opportunity."""
    position_id: str
    ticker: str
    gamma: float
    delta: float
    suggested_action: str    # "sell_delta" or "buy_delta"
    shares_to_trade: int
    expected_pnl: float      # from gamma × move²
    reason: str


@dataclass
class VegaAlert:
    """Vega exposure limit breach."""
    current_vega: float
    limit: float
    regime: str
    breach_pct: float
    severity: str            # "warning" or "critical"


# ── Engine ──────────────────────────────────────────────────────────────


class GreeksRiskEngine:
    """Real-time portfolio Greeks calculator and risk monitor."""

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()
        self.positions: List[LivePosition] = []
        self._greeks_cache: Dict[str, PositionGreeks] = {}

    # ── Position management ─────────────────────────────────────────────

    def add_position(self, pos: LivePosition) -> PositionGreeks:
        """Add a position and compute its Greeks."""
        self.positions.append(pos)
        pg = self._compute_position_greeks(pos)
        self._greeks_cache[pos.position_id] = pg
        return pg

    def remove_position(self, position_id: str) -> None:
        self.positions = [p for p in self.positions if p.position_id != position_id]
        self._greeks_cache.pop(position_id, None)

    def update_market(
        self, ticker: str, price: float, iv: Optional[float] = None,
    ) -> None:
        """Update underlying price/IV and recompute Greeks for affected positions."""
        for pos in self.positions:
            if pos.ticker == ticker:
                pos.underlying_price = price
                if iv is not None:
                    pos.iv = iv
                self._greeks_cache[pos.position_id] = self._compute_position_greeks(pos)

    def clear(self) -> None:
        self.positions.clear()
        self._greeks_cache.clear()

    # ── Greeks computation ──────────────────────────────────────────────

    def _compute_position_greeks(self, pos: LivePosition) -> PositionGreeks:
        T = max(pos.dte / 365.0, 1e-10)
        sign = -1.0 if pos.direction == "short" else 1.0
        mult = pos.contracts * 100.0

        main = compute_option_greeks(
            pos.underlying_price, pos.strike, T, pos.iv, pos.rate, pos.option_type,
        )
        d = sign * main.delta * mult
        g = sign * main.gamma * mult
        t = sign * main.theta * mult
        v = sign * main.vega * mult
        r = sign * main.rho * mult
        val = sign * main.price * mult

        if pos.spread_strike is not None:
            hedge = compute_option_greeks(
                pos.underlying_price, pos.spread_strike, T, pos.iv, pos.rate, pos.option_type,
            )
            hs = -sign
            d += hs * hedge.delta * mult
            g += hs * hedge.gamma * mult
            t += hs * hedge.theta * mult
            v += hs * hedge.vega * mult
            r += hs * hedge.rho * mult
            val += hs * hedge.price * mult

        return PositionGreeks(
            pos.position_id, pos.strategy, pos.ticker,
            d, g, t, v, r, val,
        )

    # ── Portfolio snapshot ──────────────────────────────────────────────

    def snapshot(self) -> PortfolioSnapshot:
        """Compute full portfolio Greeks snapshot."""
        total = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        by_strat: Dict[str, Dict[str, float]] = {}
        by_ticker: Dict[str, Dict[str, float]] = {}

        for pg in self._greeks_cache.values():
            total["delta"] += pg.delta
            total["gamma"] += pg.gamma
            total["theta"] += pg.theta
            total["vega"] += pg.vega
            total["rho"] += pg.rho

            for key, group in [(pg.strategy, by_strat), (pg.ticker, by_ticker)]:
                if key not in group:
                    group[key] = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
                group[key]["delta"] += pg.delta
                group[key]["gamma"] += pg.gamma
                group[key]["theta"] += pg.theta
                group[key]["vega"] += pg.vega

        delta_neutral = abs(total["delta"]) < self.config.delta_neutral_threshold
        theta_pnl = total["theta"]  # daily theta income

        return PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_delta=total["delta"],
            total_gamma=total["gamma"],
            total_theta=total["theta"],
            total_vega=total["vega"],
            total_rho=total["rho"],
            n_positions=len(self.positions),
            by_strategy=by_strat,
            by_ticker=by_ticker,
            delta_neutral=delta_neutral,
            theta_daily_pnl=theta_pnl,
        )

    # ── Delta-neutral hedging ───────────────────────────────────────────

    def hedging_suggestions(self) -> List[HedgeSuggestion]:
        """Generate delta-neutral hedging suggestions."""
        snap = self.snapshot()
        suggestions: List[HedgeSuggestion] = []
        port_delta = snap.total_delta

        if abs(port_delta) <= self.config.delta_neutral_threshold:
            return suggestions

        # Urgency
        ratio = abs(port_delta) / self.config.max_portfolio_delta
        if ratio > 1.0:
            urgency = "high"
        elif ratio > 0.6:
            urgency = "medium"
        else:
            urgency = "low"

        # Share hedge: buy/sell underlying shares
        shares_needed = -int(round(port_delta))
        if shares_needed > 0:
            action = "buy_shares"
        else:
            action = "sell_shares"

        # Pick the most exposed ticker
        ticker = "SPY"
        if snap.by_ticker:
            ticker = max(snap.by_ticker, key=lambda t: abs(snap.by_ticker[t]["delta"]))

        suggestions.append(HedgeSuggestion(
            action=action, ticker=ticker,
            quantity=abs(shares_needed),
            estimated_delta=float(-port_delta),
            residual_delta=0.0,
            reason=f"Portfolio delta {port_delta:+.1f} exceeds threshold {self.config.delta_neutral_threshold}",
            urgency=urgency,
        ))

        # Option hedge alternative: buy puts if long delta, sell calls if short delta
        if port_delta > self.config.delta_neutral_threshold:
            contracts = max(1, int(abs(port_delta) / 50))  # ~50 delta per ATM put
            suggestions.append(HedgeSuggestion(
                action="buy_put", ticker=ticker,
                quantity=contracts,
                estimated_delta=float(-contracts * 50),
                residual_delta=port_delta - contracts * 50,
                reason="Option-based delta hedge (ATM puts)",
                urgency=urgency,
            ))
        elif port_delta < -self.config.delta_neutral_threshold:
            contracts = max(1, int(abs(port_delta) / 50))
            suggestions.append(HedgeSuggestion(
                action="buy_call", ticker=ticker,
                quantity=contracts,
                estimated_delta=float(contracts * 50),
                residual_delta=port_delta + contracts * 50,
                reason="Option-based delta hedge (ATM calls)",
                urgency=urgency,
            ))

        return suggestions

    # ── Theta decay attribution ─────────────────────────────────────────

    def theta_attribution(self) -> List[ThetaAttribution]:
        """Attribute daily theta P&L to each position."""
        total_theta = sum(pg.theta for pg in self._greeks_cache.values())
        results: List[ThetaAttribution] = []

        for pos in self.positions:
            pg = self._greeks_cache.get(pos.position_id)
            if pg is None:
                continue
            pct = pg.theta / total_theta if abs(total_theta) > 1e-10 else 0
            projected = pg.theta * pos.dte
            results.append(ThetaAttribution(
                position_id=pos.position_id, strategy=pos.strategy,
                daily_theta=pg.theta, pct_of_total=pct,
                days_remaining=pos.dte, projected_total=projected,
            ))

        return sorted(results, key=lambda a: -abs(a.daily_theta))

    # ── Gamma scalping detector ─────────────────────────────────────────

    def gamma_scalp_opportunities(self, expected_move_pct: float = 0.01) -> List[GammaScalpOpportunity]:
        """Detect positions where gamma is high enough for scalping.

        Gamma scalping: when gamma is large, delta changes fast with price
        moves, allowing profitable rebalancing.
        """
        results: List[GammaScalpOpportunity] = []

        for pos in self.positions:
            pg = self._greeks_cache.get(pos.position_id)
            if pg is None:
                continue

            abs_gamma = abs(pg.gamma)
            if abs_gamma < self.config.gamma_scalp_threshold:
                continue

            # Expected P&L from gamma: 0.5 × gamma × (move)²
            dollar_move = pos.underlying_price * expected_move_pct
            expected_pnl = 0.5 * abs_gamma * dollar_move ** 2

            # Shares to trade for delta rebalance
            shares = int(round(abs(pg.delta)))
            action = "sell_delta" if pg.delta > 0 else "buy_delta"

            results.append(GammaScalpOpportunity(
                position_id=pos.position_id, ticker=pos.ticker,
                gamma=pg.gamma, delta=pg.delta,
                suggested_action=action, shares_to_trade=shares,
                expected_pnl=expected_pnl,
                reason=f"Gamma {abs_gamma:.1f} > threshold {self.config.gamma_scalp_threshold}; "
                       f"est ${expected_pnl:.0f} on {expected_move_pct:.0%} move",
            ))

        return sorted(results, key=lambda o: -o.expected_pnl)

    # ── Vega exposure limits ────────────────────────────────────────────

    def check_vega_limits(self, regime: str = "neutral") -> Optional[VegaAlert]:
        """Check vega exposure against regime-conditional limits."""
        snap = self.snapshot()
        abs_vega = abs(snap.total_vega)

        limit = self.config.vega_limits_by_regime.get(
            regime, self.config.max_portfolio_vega,
        )

        if abs_vega <= limit:
            return None

        breach_pct = (abs_vega - limit) / limit * 100
        severity = "critical" if breach_pct > 50 else "warning"

        return VegaAlert(
            current_vega=abs_vega, limit=limit, regime=regime,
            breach_pct=breach_pct, severity=severity,
        )

    # ── Risk summary ───────────────────────────────────────────────────

    def risk_summary(self, regime: str = "neutral") -> Dict[str, Any]:
        """Comprehensive risk summary."""
        snap = self.snapshot()
        hedges = self.hedging_suggestions()
        theta = self.theta_attribution()
        gamma_ops = self.gamma_scalp_opportunities()
        vega_alert = self.check_vega_limits(regime)

        breaches = []
        if abs(snap.total_delta) > self.config.max_portfolio_delta:
            breaches.append(f"Delta {snap.total_delta:+.1f} > {self.config.max_portfolio_delta}")
        if abs(snap.total_gamma) > self.config.max_portfolio_gamma:
            breaches.append(f"Gamma {snap.total_gamma:+.1f} > {self.config.max_portfolio_gamma}")
        if vega_alert:
            breaches.append(f"Vega {vega_alert.current_vega:.1f} > {vega_alert.limit:.0f} ({regime})")

        return {
            "snapshot": snap,
            "n_hedges_suggested": len(hedges),
            "hedges": hedges,
            "theta_daily_pnl": snap.theta_daily_pnl,
            "theta_attributions": len(theta),
            "gamma_opportunities": len(gamma_ops),
            "vega_alert": vega_alert,
            "n_breaches": len(breaches),
            "breaches": breaches,
            "risk_ok": len(breaches) == 0,
        }
