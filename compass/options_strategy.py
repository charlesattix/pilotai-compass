"""
Options strategy construction engine.

Builds credit spreads, iron condors, butterflies, and calendars
with regime/IV/skew-aware selection, Greeks-based sizing,
roll logic, margin scoring, and P&L scenario analysis.

All methods work on pre-loaded data — no broker connections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class StrategyType(str, Enum):
    VERTICAL = "vertical"
    IRON_CONDOR = "iron_condor"
    BUTTERFLY = "butterfly"
    CALENDAR = "calendar"


class Direction(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


@dataclass
class OptionLeg:
    strike: float
    expiry_days: int
    option_type: str          # "call" or "put"
    direction: str            # "long" or "short"
    iv: float = 0.20
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    premium: float = 0.0


@dataclass
class Strategy:
    name: str
    strategy_type: StrategyType
    direction: Direction
    legs: List[OptionLeg]
    max_loss: float = 0.0
    max_profit: float = 0.0
    breakevens: List[float] = field(default_factory=list)
    margin_required: float = 0.0
    net_premium: float = 0.0
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0


@dataclass
class RollSignal:
    trigger_type: str         # "time" | "delta"
    current_dte: int
    current_delta: float
    should_roll: bool
    reason: str


@dataclass
class ScenarioResult:
    underlying_price: float
    days_forward: int
    pnl: float
    return_pct: float


@dataclass
class StrategyRecommendation:
    strategy: Strategy
    regime: str
    iv_rank: float
    skew: float
    score: float


# ---------------------------------------------------------------------------
# Greeks helpers (Black-Scholes)
# ---------------------------------------------------------------------------

def _bs_price(S, K, T, r, sigma, is_call=True):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if is_call else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_greeks(S, K, T, r, sigma, is_call=True):
    if T <= 0 or sigma <= 0:
        return {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    delta = norm.cdf(d1) if is_call else norm.cdf(d1) - 1
    gamma = nd1 / (S * sigma * np.sqrt(T))
    theta = (-S * nd1 * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * (norm.cdf(d2) if is_call else norm.cdf(-d2))) / 365
    vega = S * nd1 * np.sqrt(T) / 100
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class OptionsStrategyEngine:
    """Options strategy construction and analysis.

    Args:
        risk_free_rate: Annualised risk-free rate.
        max_loss_pct: Max loss as fraction of account for sizing.
    """

    def __init__(self, risk_free_rate: float = 0.045, max_loss_pct: float = 0.02) -> None:
        self.risk_free_rate = risk_free_rate
        self.max_loss_pct = max_loss_pct

    # ------------------------------------------------------------------
    # Spread builders
    # ------------------------------------------------------------------

    def build_vertical(
        self, underlying: float, short_strike: float, long_strike: float,
        expiry_days: int, iv_short: float, iv_long: float,
        is_put: bool = True,
    ) -> Strategy:
        """Build a vertical spread (bull put or bear call)."""
        T = expiry_days / 365
        otype = "put" if is_put else "call"
        r = self.risk_free_rate

        short_prem = _bs_price(underlying, short_strike, T, r, iv_short, not is_put)
        long_prem = _bs_price(underlying, long_strike, T, r, iv_long, not is_put)
        short_greeks = _bs_greeks(underlying, short_strike, T, r, iv_short, not is_put)
        long_greeks = _bs_greeks(underlying, long_strike, T, r, iv_long, not is_put)

        net = short_prem - long_prem
        width = abs(short_strike - long_strike)
        max_loss = width - net if net > 0 else width
        direction = Direction.BULL if is_put else Direction.BEAR

        legs = [
            OptionLeg(short_strike, expiry_days, otype, "short", iv_short,
                      premium=short_prem, **short_greeks),
            OptionLeg(long_strike, expiry_days, otype, "long", iv_long,
                      premium=long_prem, **long_greeks),
        ]

        return Strategy(
            name=f"{'Bull Put' if is_put else 'Bear Call'} {short_strike}/{long_strike}",
            strategy_type=StrategyType.VERTICAL, direction=direction,
            legs=legs, max_loss=max_loss, max_profit=max(net, 0),
            net_premium=net, margin_required=max_loss,
            net_delta=short_greeks["delta"] - long_greeks["delta"],
            net_gamma=-(short_greeks["gamma"] - long_greeks["gamma"]),
            net_theta=-(short_greeks["theta"] - long_greeks["theta"]),
            net_vega=-(short_greeks["vega"] - long_greeks["vega"]),
        )

    def build_iron_condor(
        self, underlying: float,
        put_short: float, put_long: float,
        call_short: float, call_long: float,
        expiry_days: int, iv: float,
    ) -> Strategy:
        """Build an iron condor."""
        put_spread = self.build_vertical(underlying, put_short, put_long, expiry_days, iv, iv, True)
        call_spread = self.build_vertical(underlying, call_short, call_long, expiry_days, iv, iv, False)

        legs = put_spread.legs + call_spread.legs
        net_prem = put_spread.net_premium + call_spread.net_premium
        max_loss = max(put_spread.max_loss, call_spread.max_loss)

        return Strategy(
            name=f"IC {put_long}/{put_short}/{call_short}/{call_long}",
            strategy_type=StrategyType.IRON_CONDOR, direction=Direction.NEUTRAL,
            legs=legs, max_loss=max_loss, max_profit=max(net_prem, 0),
            net_premium=net_prem, margin_required=max_loss,
            net_delta=put_spread.net_delta + call_spread.net_delta,
            net_gamma=put_spread.net_gamma + call_spread.net_gamma,
            net_theta=put_spread.net_theta + call_spread.net_theta,
            net_vega=put_spread.net_vega + call_spread.net_vega,
        )

    def build_butterfly(
        self, underlying: float, lower: float, middle: float, upper: float,
        expiry_days: int, iv: float, is_call: bool = True,
    ) -> Strategy:
        """Build a butterfly spread."""
        T = expiry_days / 365
        r = self.risk_free_rate
        otype = "call" if is_call else "put"

        p_low = _bs_price(underlying, lower, T, r, iv, is_call)
        p_mid = _bs_price(underlying, middle, T, r, iv, is_call)
        p_up = _bs_price(underlying, upper, T, r, iv, is_call)

        net_debit = p_low - 2 * p_mid + p_up
        max_profit = (middle - lower) - abs(net_debit)
        max_loss = abs(net_debit)

        legs = [
            OptionLeg(lower, expiry_days, otype, "long", iv, premium=p_low),
            OptionLeg(middle, expiry_days, otype, "short", iv, premium=p_mid),
            OptionLeg(middle, expiry_days, otype, "short", iv, premium=p_mid),
            OptionLeg(upper, expiry_days, otype, "long", iv, premium=p_up),
        ]

        return Strategy(
            name=f"Butterfly {lower}/{middle}/{upper}",
            strategy_type=StrategyType.BUTTERFLY, direction=Direction.NEUTRAL,
            legs=legs, max_loss=max_loss, max_profit=max(max_profit, 0),
            net_premium=-abs(net_debit), margin_required=max_loss,
        )

    @staticmethod
    def build_calendar(
        underlying: float, strike: float,
        near_expiry: int, far_expiry: int,
        iv_near: float, iv_far: float,
        is_call: bool = True,
    ) -> Strategy:
        """Build a calendar spread (sell near, buy far)."""
        legs = [
            OptionLeg(strike, near_expiry, "call" if is_call else "put", "short", iv_near),
            OptionLeg(strike, far_expiry, "call" if is_call else "put", "long", iv_far),
        ]
        return Strategy(
            name=f"Calendar {strike} {near_expiry}/{far_expiry}DTE",
            strategy_type=StrategyType.CALENDAR, direction=Direction.NEUTRAL,
            legs=legs,
        )

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    @staticmethod
    def select_strategy(
        regime: str, iv_rank: float, skew: float,
    ) -> Tuple[StrategyType, Direction]:
        """Select best strategy type based on market conditions."""
        if regime in ("crash",):
            return StrategyType.VERTICAL, Direction.BEAR
        if regime in ("high_vol",) and iv_rank > 50:
            return StrategyType.IRON_CONDOR, Direction.NEUTRAL
        if regime in ("low_vol",) and iv_rank < 25:
            return StrategyType.CALENDAR, Direction.NEUTRAL
        if regime == "bull":
            if skew > 0.05:
                return StrategyType.VERTICAL, Direction.BULL
            return StrategyType.IRON_CONDOR, Direction.NEUTRAL
        if regime == "bear":
            return StrategyType.VERTICAL, Direction.BEAR
        return StrategyType.IRON_CONDOR, Direction.NEUTRAL

    def recommend(
        self, underlying: float, regime: str, iv_rank: float, skew: float,
        expiry_days: int = 45, iv: float = 0.20,
    ) -> StrategyRecommendation:
        """Generate a strategy recommendation."""
        stype, direction = self.select_strategy(regime, iv_rank, skew)
        width = underlying * 0.05

        if stype == StrategyType.VERTICAL:
            if direction == Direction.BULL:
                strat = self.build_vertical(underlying, underlying * 0.95, underlying * 0.90, expiry_days, iv, iv, True)
            else:
                strat = self.build_vertical(underlying, underlying * 1.05, underlying * 1.10, expiry_days, iv, iv, False)
        elif stype == StrategyType.IRON_CONDOR:
            strat = self.build_iron_condor(
                underlying, underlying * 0.93, underlying * 0.88,
                underlying * 1.07, underlying * 1.12, expiry_days, iv)
        elif stype == StrategyType.BUTTERFLY:
            strat = self.build_butterfly(underlying, underlying * 0.95, underlying, underlying * 1.05, expiry_days, iv)
        else:
            strat = self.build_calendar(underlying, underlying, expiry_days, expiry_days + 30, iv, iv * 1.05)

        score = 50 + iv_rank * 0.3 - abs(skew) * 100
        return StrategyRecommendation(strategy=strat, regime=regime, iv_rank=iv_rank, skew=skew, score=score)

    # ------------------------------------------------------------------
    # Greeks-based sizing
    # ------------------------------------------------------------------

    def size_by_max_loss(self, strategy: Strategy, account_size: float) -> int:
        """Number of contracts limited by max loss."""
        if strategy.max_loss <= 0:
            return 0
        max_risk = account_size * self.max_loss_pct
        return max(1, int(max_risk / (strategy.max_loss * 100)))

    # ------------------------------------------------------------------
    # Roll logic
    # ------------------------------------------------------------------

    @staticmethod
    def check_roll(
        current_dte: int, current_delta: float,
        dte_trigger: int = 14, delta_trigger: float = 0.30,
    ) -> RollSignal:
        """Check whether position should be rolled."""
        if current_dte <= dte_trigger:
            return RollSignal("time", current_dte, current_delta, True,
                               f"DTE {current_dte} <= {dte_trigger}")
        if abs(current_delta) >= delta_trigger:
            return RollSignal("delta", current_dte, current_delta, True,
                               f"|delta| {abs(current_delta):.2f} >= {delta_trigger}")
        return RollSignal("none", current_dte, current_delta, False, "no trigger")

    # ------------------------------------------------------------------
    # Margin efficiency
    # ------------------------------------------------------------------

    @staticmethod
    def margin_efficiency(strategy: Strategy) -> float:
        """Max profit / margin required ratio."""
        if strategy.margin_required <= 0:
            return 0.0
        return strategy.max_profit / strategy.margin_required

    # ------------------------------------------------------------------
    # P&L scenario analysis
    # ------------------------------------------------------------------

    def scenario_analysis(
        self, strategy: Strategy, underlying: float,
        price_range_pct: float = 0.10, n_prices: int = 11,
        horizons: Optional[List[int]] = None,
    ) -> List[ScenarioResult]:
        """P&L at multiple price/time scenarios."""
        horizons = horizons or [0, 7, 14, 30]
        prices = np.linspace(underlying * (1 - price_range_pct),
                              underlying * (1 + price_range_pct), n_prices)
        results: List[ScenarioResult] = []
        for days_fwd in horizons:
            for px in prices:
                pnl = 0.0
                for leg in strategy.legs:
                    remaining = max(leg.expiry_days - days_fwd, 0) / 365
                    is_call = leg.option_type == "call"
                    price = _bs_price(px, leg.strike, remaining, self.risk_free_rate, leg.iv, is_call)
                    entry = leg.premium
                    if leg.direction == "short":
                        pnl += entry - price
                    else:
                        pnl += price - entry
                ref = abs(strategy.net_premium) if strategy.net_premium != 0 else 1.0
                results.append(ScenarioResult(px, days_fwd, pnl, pnl / ref if ref > 0 else 0))
        return results

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        strategies: List[Strategy],
        scenarios: Optional[Dict[str, List[ScenarioResult]]] = None,
        output_path: str = "reports/options_strategy.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for s in strategies:
            rows.append(
                f"<tr><td style='text-align:left'>{s.name}</td>"
                f"<td>{s.strategy_type.value}</td><td>{s.direction.value}</td>"
                f"<td>{s.max_profit:.2f}</td><td>{s.max_loss:.2f}</td>"
                f"<td>{s.net_premium:.2f}</td><td>{s.net_delta:.3f}</td>"
                f"<td>{s.net_theta:.4f}</td><td>{self.margin_efficiency(s):.2f}</td></tr>")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Options Strategy</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #f5f5f5; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
</style></head><body>
<h1>Options Strategy Report</h1>
<h2>Strategy Comparison</h2>
<table><tr><th style='text-align:left'>Name</th><th>Type</th><th>Dir</th>
<th>Max Profit</th><th>Max Loss</th><th>Premium</th>
<th>Delta</th><th>Theta</th><th>Margin Eff.</th></tr>
{''.join(rows)}</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
