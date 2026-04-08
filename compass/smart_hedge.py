"""
Smart Hedge — Cost-Efficient Tail Risk Protection
====================================================
Real SPY puts cost 4.36%/yr (2.2x the 2% assumption). This module
implements 5 cost-efficient alternatives using real IronVault pricing.

Variants:
  A) VIX<15 puts only — buy only when cheapest (2.0-2.5%/yr)
  B) Put spreads 5% wide — cap cost via long put wing
  C) Dynamic budget — 0.5-3% allocation based on VIX
  D) Collar — sell OTM calls to fund puts
  E) Selective quarterly — hedge before known risk events only

All costs calibrated from 69 months of real IronVault SPY put data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Real cost calibration (from IronVault 2020-2025)
# ═══════════════════════════════════════════════════════════════════════════

# Annualised cost of 5% OTM put by VIX level (from hedge_cost_reality.py)
VIX_TO_PUT_COST = {
    12: 0.018,  # VIX < 15: cheapest puts (~1.8%/yr)
    15: 0.024,  # Calm
    18: 0.030,
    20: 0.036,  # Normal: 3.6%/yr
    25: 0.048,  # Elevated: 4.8%/yr
    30: 0.058,  # High: 5.8%/yr
    35: 0.065,
    40: 0.072,  # Crisis: 7.2%/yr
}

# Put spread cost ratio (5% wide spread costs ~55% of naked put)
PUT_SPREAD_DISCOUNT = 0.55

# Collar: OTM call premium offsets put cost
# Selling 3% OTM calls generates ~60-80% of put cost
COLLAR_OFFSET_RATIO = 0.70

TRADING_DAYS = 252


def _interp_put_cost(vix: float) -> float:
    """Interpolate annualized put cost from VIX level."""
    levels = sorted(VIX_TO_PUT_COST.keys())
    if vix <= levels[0]:
        return VIX_TO_PUT_COST[levels[0]]
    if vix >= levels[-1]:
        return VIX_TO_PUT_COST[levels[-1]]
    for i in range(len(levels) - 1):
        if levels[i] <= vix <= levels[i+1]:
            frac = (vix - levels[i]) / (levels[i+1] - levels[i])
            return VIX_TO_PUT_COST[levels[i]] * (1-frac) + VIX_TO_PUT_COST[levels[i+1]] * frac
    return 0.04


# ═══════════════════════════════════════════════════════════════════════════
# Hedge variant implementations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HedgeDay:
    """Single day's hedge state."""
    cost: float         # $ cost today
    payoff: float       # $ payoff today (if crash)
    hedge_active: bool
    leverage_adj: float # leverage multiplier (1.0 = no change)
    vix: float
    regime: str


def _put_payoff(spy_ret: float, hedge_notional: float, payoff_mult: float = 10.0) -> float:
    """Compute put payoff on down days."""
    if spy_ret >= -0.005:
        return 0.0
    severity = abs(spy_ret) / 0.01
    payoff = hedge_notional * payoff_mult * severity
    # Convexity: big crashes pay exponentially more
    if abs(spy_ret) > 0.03:
        payoff *= (1 + (abs(spy_ret) - 0.03) * 8)
    return min(payoff, hedge_notional * 50)  # cap at 50x daily cost


class HedgeVariantA:
    """VIX<15 puts only — buy exclusively when cheapest."""

    def __init__(self, max_annual_pct: float = 0.025):
        self.max_annual = max_annual_pct
        self.name = "A: VIX<15 Puts Only"

    def daily(self, portfolio_val: float, spy_ret: float, vix: float, **kw) -> HedgeDay:
        if vix < 15:
            # Buy puts at cheap levels
            cost_rate = _interp_put_cost(vix)
            daily_cost = portfolio_val * min(cost_rate, self.max_annual) / TRADING_DAYS
            payoff = _put_payoff(spy_ret, daily_cost, payoff_mult=12.0)
            return HedgeDay(daily_cost, payoff, True, 1.0, vix, "hedged")
        else:
            # No hedge — reduce leverage instead
            lev_adj = max(0.6, 1.0 - (vix - 15) / 40)
            return HedgeDay(0, 0, False, lev_adj, vix, "unhedged")


class HedgeVariantB:
    """Put spreads (5% wide) — cap cost via long put wing."""

    def __init__(self, annual_budget_pct: float = 0.025):
        self.budget = annual_budget_pct
        self.name = "B: Put Spreads (5% wide)"

    def daily(self, portfolio_val: float, spy_ret: float, vix: float, **kw) -> HedgeDay:
        naked_cost = _interp_put_cost(vix)
        # Put spread costs ~55% of naked put
        spread_cost = naked_cost * PUT_SPREAD_DISCOUNT
        actual_cost = min(spread_cost, self.budget)
        daily_cost = portfolio_val * actual_cost / TRADING_DAYS
        # Spread payoff is capped (can't profit below long wing)
        payoff = _put_payoff(spy_ret, daily_cost, payoff_mult=8.0)  # lower mult (capped)
        return HedgeDay(daily_cost, payoff, True, 1.0, vix, "spread_hedged")


class HedgeVariantC:
    """Dynamic budget — 0.5-3% based on VIX regime."""

    def __init__(self):
        self.name = "C: Dynamic Budget (0.5-3%)"

    def daily(self, portfolio_val: float, spy_ret: float, vix: float, **kw) -> HedgeDay:
        # Budget scales with VIX: cheap when calm, expensive when needed
        if vix < 15:
            budget = 0.005  # 0.5%/yr — minimal maintenance hedge
        elif vix < 20:
            budget = 0.015  # 1.5%/yr — normal
        elif vix < 25:
            budget = 0.025  # 2.5%/yr — elevated, need protection
        else:
            budget = 0.030  # 3.0%/yr — crisis, max protection

        actual_cost = min(_interp_put_cost(vix), budget)
        daily_cost = portfolio_val * actual_cost / TRADING_DAYS
        # Higher payoff when we've spent more
        payoff_mult = 8.0 + (budget - 0.005) / 0.025 * 6  # 8-14x
        payoff = _put_payoff(spy_ret, daily_cost, payoff_mult)
        return HedgeDay(daily_cost, payoff, True, 1.0, vix, "dynamic")


class HedgeVariantD:
    """Collar — sell OTM calls to fund puts."""

    def __init__(self, net_cost_target: float = 0.010):
        self.net_target = net_cost_target
        self.name = "D: Collar (sell calls to fund puts)"

    def daily(self, portfolio_val: float, spy_ret: float, vix: float, **kw) -> HedgeDay:
        put_cost_rate = _interp_put_cost(vix)
        # Selling 3% OTM calls generates ~70% of put cost
        call_income = put_cost_rate * COLLAR_OFFSET_RATIO
        net_cost_rate = max(0.002, put_cost_rate - call_income)  # floor at 0.2%
        daily_cost = portfolio_val * net_cost_rate / TRADING_DAYS

        # Put payoff (full protection on downside)
        payoff = _put_payoff(spy_ret, portfolio_val * put_cost_rate / TRADING_DAYS, 10.0)

        # Call cap: lose upside above cap level (~3% OTM)
        # If spy_ret > 0.03, we lose the excess (call was sold)
        call_loss = 0.0
        if spy_ret > 0.025:
            call_loss = portfolio_val * (spy_ret - 0.025) * 0.5  # partial loss (spread)

        net_payoff = payoff - call_loss
        return HedgeDay(daily_cost, net_payoff, True, 1.0, vix, "collared")


class HedgeVariantE:
    """Selective quarterly — hedge before known risk periods only."""

    def __init__(self, annual_budget_pct: float = 0.015):
        self.budget = annual_budget_pct
        self.name = "E: Selective Quarterly"

    def daily(self, portfolio_val: float, spy_ret: float, vix: float,
              day_of_year: int = 0, **kw) -> HedgeDay:
        # Hedge only during high-risk periods:
        # Q1 (Jan-Feb): post-year-end rebalancing
        # Late Q3 (Aug-Sep): historically worst months
        # Q4 pre-election/FOMC windows
        hedge_months = {1, 2, 8, 9, 10}  # 5 months out of 12
        month = (day_of_year // 21) % 12 + 1  # approximate

        if month in hedge_months:
            cost_rate = min(_interp_put_cost(vix) * PUT_SPREAD_DISCOUNT, self.budget * 12/5)
            daily_cost = portfolio_val * cost_rate / TRADING_DAYS
            payoff = _put_payoff(spy_ret, daily_cost, 10.0)
            return HedgeDay(daily_cost, payoff, True, 1.0, vix, "seasonal_hedge")
        else:
            # No hedge in calm months — save cost
            lev_adj = max(0.8, 1.0 - max(0, vix - 20) / 30)
            return HedgeDay(0, 0, False, lev_adj, vix, "no_hedge")


# ═══════════════════════════════════════════════════════════════════════════
# Backtest engine
# ═══════════════════════════════════════════════════════════════════════════

VARIANTS = {
    "A": HedgeVariantA,
    "B": HedgeVariantB,
    "C": HedgeVariantC,
    "D": HedgeVariantD,
    "E": HedgeVariantE,
}

# EXP-1220 yearly returns (real data)
EXP1220_YEARLY = {
    2020: {"ret": 0.5297, "dd": 0.0388},
    2021: {"ret": 0.4913, "dd": 0.0152},
    2022: {"ret": 0.1482, "dd": 0.0657},
    2023: {"ret": 0.4010, "dd": 0.0337},
    2024: {"ret": 0.3151, "dd": 0.0125},
    2025: {"ret": 0.3724, "dd": 0.0167},
}

# VIX monthly profiles (from real Yahoo data)
VIX_MONTHLY = {
    2020: [14, 15, 58, 40, 30, 28, 26, 22, 27, 28, 22, 20],
    2021: [30, 22, 20, 18, 19, 16, 19, 16, 21, 17, 22, 18],
    2022: [24, 28, 26, 30, 28, 28, 24, 22, 28, 30, 22, 22],
    2023: [20, 20, 22, 17, 16, 14, 14, 16, 16, 19, 14, 13],
    2024: [14, 14, 13, 16, 13, 13, 16, 22, 17, 20, 15, 16],
    2025: [16, 16, 24, 30, 26, 22, 18, 20, 22, 24, 20, 18],
}


def build_daily_data(seed=9000):
    """Build daily portfolio returns, SPY returns, and VIX."""
    rng = np.random.RandomState(seed)
    port_ret, spy_ret, vix_arr = [], [], []

    for yr in range(2020, 2026):
        n = 252 if yr != 2025 else 249
        yr_ret = EXP1220_YEARLY[yr]["ret"]
        dd = EXP1220_YEARLY[yr]["dd"]
        vol = max(dd * 2.0, 0.005)

        daily_mean = yr_ret / n
        daily_vol = vol / math.sqrt(252)
        days = rng.normal(daily_mean, daily_vol, n)

        # SPY returns (correlated but noisier)
        spy_annual = {2020: 0.18, 2021: 0.29, 2022: -0.18, 2023: 0.26, 2024: 0.25, 2025: 0.19}
        spy_mean = spy_annual.get(yr, 0.10) / n
        spy_days = rng.normal(spy_mean, 0.012, n)

        # Inject COVID crash in 2020
        if yr == 2020:
            for i in range(20, 43):
                spy_days[i] = rng.normal(-0.03, 0.02)
                days[i] = rng.normal(-0.005, 0.008)  # EXP-1220 cushions

        # VIX from monthly profiles
        monthly = VIX_MONTHLY[yr]
        for i in range(n):
            m = min(i // 21, 11)
            vix_arr.append(max(10, monthly[m] + rng.normal(0, 1.5)))

        port_ret.extend(days)
        spy_ret.extend(spy_days)

    return np.array(port_ret), np.array(spy_ret), np.array(vix_arr)


def backtest_variant(
    variant_cls,
    port_ret: np.ndarray,
    spy_ret: np.ndarray,
    vix: np.ndarray,
    base_leverage: float = 1.6,
    capital: float = 100_000,
) -> Dict:
    """Run full backtest of a hedge variant."""
    hedge = variant_cls()
    n = len(port_ret)
    equity = capital
    peak = equity
    daily_returns = []
    total_cost = 0.0
    total_payoff = 0.0
    crisis_dd = 0.0  # track COVID-period DD

    for i in range(n):
        h = hedge.daily(equity, spy_ret[i], vix[i], day_of_year=i % 252)
        lev = base_leverage * h.leverage_adj

        # Portfolio return with hedge
        base_r = port_ret[i] * lev
        hedge_net = (h.payoff - h.cost) / max(equity, 1)
        daily_r = base_r + hedge_net

        equity *= (1 + daily_r)
        peak = max(peak, equity)
        daily_returns.append(daily_r)
        total_cost += h.cost
        total_payoff += h.payoff

        # Track COVID DD (days 20-60 of first year)
        if 20 <= i <= 80:
            dd = (equity - peak) / peak
            crisis_dd = min(crisis_dd, dd)

    daily_arr = np.array(daily_returns)
    cum = np.cumprod(1 + daily_arr)
    n_years = n / TRADING_DAYS
    cagr = cum[-1] ** (1/n_years) - 1 if cum[-1] > 0 else -1
    vol = np.std(daily_arr) * math.sqrt(TRADING_DAYS)
    _rf_daily = 0.045 / 252
    sharpe = (float(np.mean(daily_arr)) - _rf_daily) / float(np.std(daily_arr)) * math.sqrt(TRADING_DAYS) if float(np.std(daily_arr)) > 1e-12 else 0
    pk = np.maximum.accumulate(cum)
    max_dd = ((cum - pk) / pk).min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-8 else float("inf")

    annual_cost_pct = total_cost / n_years / capital
    annual_payoff_pct = total_payoff / n_years / capital
    net_cost_pct = (total_cost - total_payoff) / n_years / capital

    # Per-year
    per_year = {}
    idx = 0
    for yr in range(2020, 2026):
        n_yr = 252 if yr != 2025 else 249
        if idx + n_yr > n:
            break
        yr_r = daily_arr[idx:idx+n_yr]
        yr_cum = np.prod(1 + yr_r) - 1
        yr_eq = np.cumprod(1 + yr_r)
        yr_pk = np.maximum.accumulate(yr_eq)
        yr_dd = ((yr_eq - yr_pk) / yr_pk).min()
        per_year[yr] = {"return": float(yr_cum), "dd": float(yr_dd)}
        idx += n_yr

    return {
        "name": hedge.name,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "vol": float(vol),
        "covid_dd": float(crisis_dd),
        "annual_cost_pct": float(annual_cost_pct),
        "annual_payoff_pct": float(annual_payoff_pct),
        "net_cost_pct": float(net_cost_pct),
        "per_year": per_year,
    }


def find_pareto_optimal(results: List[Dict], covid_dd_limit: float = -0.15) -> Dict:
    """Find hedge that minimizes cost while keeping COVID DD < limit."""
    eligible = [r for r in results if r["covid_dd"] > covid_dd_limit]
    if not eligible:
        # None meet limit — return lowest DD
        return min(results, key=lambda r: r["covid_dd"])
    # Among eligible, minimize net cost
    return min(eligible, key=lambda r: r["net_cost_pct"])
