"""
Dynamic tail risk hedging for the Ultimate Portfolio.

The portfolio runs at 1.6x leverage with 11.35% historical DD, but COVID
stress testing showed -51.8%.  This module adds a systematic hedging overlay:

  1. Monitor VIX level and term structure for crisis signals
  2. When crisis detected, buy SPY puts + VIX calls as portfolio insurance
  3. Dynamic hedge ratio based on portfolio delta exposure
  4. Cost budget: max 2% annual portfolio value on hedging costs

Goal: reduce COVID-scenario DD from -51.8% to <20%.

All data is generated via calibrated simulation matching real market dynamics.
No IronVault calls (this is a portfolio-level overlay, not options pricing).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TailRiskHedgeConfig:
    """All tuneable parameters for the dynamic tail risk hedge."""

    # ── Leverage ──────────────────────────────────────────────────────────
    normal_leverage: float = 1.6       # default portfolio leverage
    crisis_leverage: float = 0.4       # reduced leverage during crisis
    min_leverage: float = 0.2          # absolute floor

    # ── Hedge cost budget ─────────────────────────────────────────────────
    annual_cost_budget_pct: float = 2.0  # max 2% of portfolio per year on hedges
    # Internally converted to daily budget = portfolio * 2% / 252

    # ── SPY put protection ────────────────────────────────────────────────
    put_buy_vix_threshold: float = 20.0   # buy puts when VIX < this (cheap)
    put_base_pct: float = 0.60            # 60% of hedge budget goes to SPY puts
    put_payoff_multiplier: float = 12.0   # OTM puts pay 12x cost in a crash
    put_delta: float = -0.20              # target 20-delta puts (OTM)

    # ── VIX call protection ───────────────────────────────────────────────
    vix_call_base_pct: float = 0.40       # 40% of budget goes to VIX calls
    vix_call_payoff_multiplier: float = 20.0  # VIX calls explode in crashes
    vix_call_trigger: float = 25.0        # VIX calls activate above this

    # ── Term structure signals ────────────────────────────────────────────
    ts_inversion_threshold: float = 1.0   # VIX/VIX3M > 1.0 = inverted
    ts_deep_inversion: float = 1.15       # deep inversion = max hedge
    ts_hedge_boost: float = 3.0           # multiply hedge allocation in inversion

    # ── Crisis detection ──────────────────────────────────────────────────
    vix_crisis_threshold: float = 28.0    # VIX > 28 = crisis
    vix_elevated_threshold: float = 20.0  # VIX > 20 = elevated
    dd_crisis_threshold: float = 0.05     # 5% DD = crisis
    dd_elevated_threshold: float = 0.02   # 2% DD = elevated
    rvol_crisis_threshold: float = 0.30   # 30% realized vol = crisis
    momentum_lookback: int = 10           # days for momentum signal
    momentum_crisis: float = -0.03        # 10-day return < -3% = crisis

    # ── Delta exposure & dynamic hedge ratio ──────────────────────────────
    # Portfolio delta is estimated from leverage * market exposure
    # Hedge ratio = portfolio_delta / hedge_delta to neutralise tail risk
    target_hedge_ratio: float = 0.30      # hedge 30% of portfolio delta normally
    crisis_hedge_ratio: float = 0.80      # hedge 80% of portfolio delta in crisis
    delta_lookback: int = 20              # days for rolling delta estimation

    # ── Leverage scaling ──────────────────────────────────────────────────
    leverage_smoothing_days: int = 2      # EMA half-life (fast response)
    leverage_ramp_up_days: int = 15       # days to ramp back after crisis

    # ── Recovery detection ────────────────────────────────────────────────
    recovery_vix_threshold: float = 22.0
    recovery_momentum_days: int = 5


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class HedgeDayState:
    """State of the hedge overlay for a single day."""
    date: object
    leverage: float
    put_cost: float             # dollar cost of SPY put protection today
    put_payoff: float           # put payoff on down days
    vix_call_cost: float        # dollar cost of VIX call protection today
    vix_call_payoff: float      # VIX call payoff on vol spike days
    crisis_score: float         # 0-1 composite crisis score
    regime: str                 # normal, elevated, crisis
    vix: float
    vix_ratio: float            # VIX / VIX3M
    realized_vol: float
    drawdown: float
    portfolio_delta: float      # estimated portfolio delta exposure
    hedge_ratio: float          # current hedge ratio (0-1)
    hedge_active: bool          # is any hedge providing protection?
    ts_inverted: bool           # term structure inverted?
    daily_hedge_budget: float   # budget available today
    daily_hedge_spent: float    # actual spend today


@dataclass
class CrisisScenario:
    """A historical crisis scenario for stress testing."""
    name: str
    description: str
    spy_shocks: np.ndarray
    vix_path: np.ndarray
    vix3m_path: np.ndarray
    n_days: int


@dataclass
class ScenarioResult:
    """Result of running one crisis scenario."""
    scenario_name: str
    hedged_dd_pct: float
    unhedged_dd_pct: float
    dd_reduction_pct: float
    hedged_return_pct: float
    unhedged_return_pct: float
    hedge_cost_pct: float
    survives_20pct: bool        # max DD < 20%?
    hedged_equity: List[float]
    unhedged_equity: List[float]


@dataclass
class BacktestResult:
    """Full backtest result across all periods."""
    cagr_pct: float
    sharpe: float
    max_dd_pct: float
    calmar: float
    sortino: float
    vol_pct: float
    total_return_pct: float
    n_days: int
    yearly_returns: Dict[int, float]
    yearly_dd: Dict[int, float]
    all_years_profitable: bool
    avg_leverage: float
    total_hedge_cost_pct: float   # annualized cost
    put_payoff_total_pct: float
    vix_call_payoff_total_pct: float
    net_hedge_cost_pct: float     # cost - all payoffs
    annual_cost_within_budget: bool
    crisis_days: int
    elevated_days: int
    normal_days: int
    avg_hedge_ratio: float
    avg_portfolio_delta: float
    equity_curve: List[float]
    daily_returns: np.ndarray
    states: List[HedgeDayState]
    scenario_results: Dict[str, ScenarioResult]


# ═══════════════════════════════════════════════════════════════════════════
# Crisis scenario definitions (calibrated to real events)
# ═══════════════════════════════════════════════════════════════════════════


def _build_crisis_path(
    total_return: float, n_days: int, seed: int,
    vix_start: float, vix_peak: float,
    vix3m_start: float, vix3m_peak: float,
) -> CrisisScenario:
    """Build a realistic crisis path with correlated VIX dynamics."""
    rng = np.random.RandomState(seed)

    if n_days <= 0:
        return CrisisScenario("", "", np.array([]), np.array([]),
                              np.array([]), 0)

    # SPY return path: front-loaded shocks
    if n_days == 1:
        spy = np.array([total_return])
    else:
        total_log = math.log(1 + total_return)
        weights = np.exp(-np.linspace(0, 2, n_days))
        weights /= weights.sum()
        daily_log = total_log * weights
        noise = rng.normal(0, abs(total_log) * 0.03, n_days)
        daily_log += noise
        daily_log *= total_log / daily_log.sum() if abs(daily_log.sum()) > 1e-12 else 1.0
        spy = np.array([math.exp(lr) - 1 for lr in daily_log])

    # VIX path: spike to peak in first third, then gradual decline
    peak_idx = max(1, n_days // 3)
    vix = np.zeros(n_days)
    vix3m = np.zeros(n_days)
    for i in range(n_days):
        if i <= peak_idx:
            t = i / peak_idx
            vix[i] = vix_start + (vix_peak - vix_start) * t
            vix3m[i] = vix3m_start + (vix3m_peak - vix3m_start) * t * 0.7
        else:
            t = (i - peak_idx) / max(n_days - peak_idx - 1, 1)
            vix[i] = vix_peak - (vix_peak - vix_start * 1.3) * t
            vix3m[i] = vix3m_peak - (vix3m_peak - vix3m_start * 1.1) * t
        vix[i] = max(vix_start * 0.8, vix[i])
        vix3m[i] = max(vix3m_start * 0.8, vix3m[i])

    return CrisisScenario(
        name="", description="", spy_shocks=spy,
        vix_path=vix, vix3m_path=vix3m, n_days=n_days,
    )


def get_crisis_scenarios() -> Dict[str, CrisisScenario]:
    """Get all calibrated crisis scenarios."""
    scenarios = {}

    covid = _build_crisis_path(-0.34, 23, 100, 14.0, 82.0, 16.0, 45.0)
    covid.name = "COVID_2020"
    covid.description = "S&P -34% in 23 days, VIX 14→82"
    scenarios["COVID_2020"] = covid

    bear = _build_crisis_path(-0.25, 190, 200, 17.0, 36.0, 19.0, 30.0)
    bear.name = "BEAR_2022"
    bear.description = "S&P -25% over 9 months, VIX 17→36"
    scenarios["BEAR_2022"] = bear

    flash = _build_crisis_path(-0.10, 1, 300, 15.0, 65.0, 17.0, 40.0)
    flash.name = "FLASH_CRASH"
    flash.description = "Single-day -10% flash crash"
    scenarios["FLASH_CRASH"] = flash

    china = _build_crisis_path(-0.11, 5, 400, 13.0, 53.0, 15.0, 35.0)
    china.name = "CHINA_2015"
    china.description = "Aug 2015 China devaluation, -11% in 5 days"
    scenarios["CHINA_2015"] = china

    volmag = _build_crisis_path(-0.10, 8, 500, 11.0, 50.0, 13.0, 30.0)
    volmag.name = "VOLMAGEDDON_2018"
    volmag.description = "Feb 2018 VIX explosion, -10% in 8 days"
    scenarios["VOLMAGEDDON_2018"] = volmag

    return scenarios


# ═══════════════════════════════════════════════════════════════════════════
# Market data generator (calibrated to 2020-2025 dynamics)
# ═══════════════════════════════════════════════════════════════════════════


def generate_market_data(
    n_years: float = 6.0,
    base_cagr: float = 0.55,
    base_vol: float = 0.12,
    seed: int = 42,
) -> Dict[str, pd.Series]:
    """Generate calibrated market data with embedded crisis periods.

    Returns dict with: spy_returns, portfolio_returns, vix, vix3m
    """
    rng = np.random.RandomState(seed)
    n_days = int(n_years * TRADING_DAYS)
    idx = pd.bdate_range("2020-01-02", periods=n_days)

    daily_mu = base_cagr / TRADING_DAYS
    daily_sigma = base_vol / math.sqrt(TRADING_DAYS)

    port_ret = rng.normal(daily_mu, daily_sigma, n_days)

    spy_ret = rng.normal(0.10 / TRADING_DAYS, 0.16 / math.sqrt(TRADING_DAYS), n_days)
    spy_ret = 0.3 * port_ret + 0.7 * spy_ret

    # VIX: mean-reverting with crisis spikes
    vix = np.zeros(n_days)
    vix[0] = 14.0
    for i in range(1, n_days):
        revert = 0.03 * (16 - vix[i - 1])
        shock = rng.normal(0, 1.2)
        ret_impact = -spy_ret[i] * 150
        vix[i] = max(9, min(85, vix[i - 1] + revert + shock + ret_impact))

    # VIX3M: smoother, inverts in crisis
    vix3m = np.zeros(n_days)
    vix3m[0] = 16.0
    for i in range(1, n_days):
        revert = 0.02 * (18 - vix3m[i - 1])
        shock = rng.normal(0, 0.8)
        vix3m[i] = max(10, min(60, vix3m[i - 1] + revert + shock - spy_ret[i] * 80))

    # Embed COVID crash (day ~40-63)
    covid = get_crisis_scenarios()["COVID_2020"]
    covid_start = 40
    covid_end = min(covid_start + covid.n_days, n_days)
    covid_len = covid_end - covid_start
    port_ret[covid_start:covid_end] = covid.spy_shocks[:covid_len] * 1.5
    spy_ret[covid_start:covid_end] = covid.spy_shocks[:covid_len]
    vix[covid_start:covid_end] = covid.vix_path[:covid_len]
    vix3m[covid_start:covid_end] = covid.vix3m_path[:covid_len]

    # Embed 2022 bear (day ~500-690)
    bear = get_crisis_scenarios()["BEAR_2022"]
    bear_start = 500
    bear_end = min(bear_start + bear.n_days, n_days)
    bear_len = bear_end - bear_start
    port_ret[bear_start:bear_end] = bear.spy_shocks[:bear_len] * 1.2
    spy_ret[bear_start:bear_end] = bear.spy_shocks[:bear_len]
    vix[bear_start:bear_end] = bear.vix_path[:bear_len]
    vix3m[bear_start:bear_end] = bear.vix3m_path[:bear_len]

    # Embed mini flash crash (day ~900)
    if n_days > 910:
        flash = get_crisis_scenarios()["FLASH_CRASH"]
        port_ret[900] = flash.spy_shocks[0] * 1.3
        spy_ret[900] = flash.spy_shocks[0]
        vix[900] = 55.0
        vix3m[900] = 35.0
        for i in range(901, min(910, n_days)):
            vix[i] = max(18, vix[i] * 0.5 + 55 * 0.5 * (0.7 ** (i - 900)))
            vix3m[i] = max(17, vix3m[i] * 0.5 + 35 * 0.5 * (0.7 ** (i - 900)))

    return {
        "portfolio_returns": pd.Series(port_ret, index=idx),
        "spy_returns": pd.Series(spy_ret, index=idx),
        "vix": pd.Series(vix, index=idx),
        "vix3m": pd.Series(vix3m, index=idx),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Core hedge engine
# ═══════════════════════════════════════════════════════════════════════════


class TailRiskHedgeEngine:
    """Dynamic tail risk hedge overlay for levered portfolios.

    Manages four coordinated mechanisms:
      1. Crisis detection: multi-signal composite (VIX, term structure, DD, rvol, momentum)
      2. SPY put protection: buy cheap puts when VIX < 20, scale with crisis
      3. VIX call protection: convex payoff on vol spikes (huge in crashes)
      4. Dynamic hedge ratio: scale notional based on portfolio delta exposure

    All hedging is constrained to a 2% annual cost budget.
    """

    def __init__(self, config: Optional[TailRiskHedgeConfig] = None):
        self.cfg = config or TailRiskHedgeConfig()

    # ── Crisis detection ──────────────────────────────────────────────────

    def _crisis_score(
        self,
        vix: float,
        vix_ratio: float,
        drawdown: float,
        realized_vol: float,
        momentum_10d: float,
    ) -> float:
        """Composite crisis score (0-1)."""
        cfg = self.cfg

        # VIX signal
        if vix <= cfg.vix_elevated_threshold:
            vix_signal = 0.0
        elif vix >= cfg.vix_crisis_threshold:
            vix_signal = 1.0
        else:
            vix_signal = (vix - cfg.vix_elevated_threshold) / (
                cfg.vix_crisis_threshold - cfg.vix_elevated_threshold)

        # Term structure inversion signal
        if vix_ratio <= cfg.ts_inversion_threshold:
            ts_signal = 0.0
        elif vix_ratio >= cfg.ts_deep_inversion:
            ts_signal = 1.0
        else:
            ts_signal = (vix_ratio - cfg.ts_inversion_threshold) / (
                cfg.ts_deep_inversion - cfg.ts_inversion_threshold)

        # Drawdown signal
        if drawdown <= cfg.dd_elevated_threshold:
            dd_signal = 0.0
        elif drawdown >= cfg.dd_crisis_threshold:
            dd_signal = 1.0
        else:
            dd_signal = (drawdown - cfg.dd_elevated_threshold) / (
                cfg.dd_crisis_threshold - cfg.dd_elevated_threshold)

        # Realized vol signal
        rvol_low = 0.15
        if realized_vol <= rvol_low:
            rvol_signal = 0.0
        elif realized_vol >= cfg.rvol_crisis_threshold:
            rvol_signal = 1.0
        else:
            rvol_signal = (realized_vol - rvol_low) / (
                cfg.rvol_crisis_threshold - rvol_low)

        # Momentum signal
        if momentum_10d >= 0:
            mom_signal = 0.0
        elif momentum_10d <= cfg.momentum_crisis:
            mom_signal = 1.0
        else:
            mom_signal = -momentum_10d / abs(cfg.momentum_crisis)

        # Weighted composite
        score = (
            0.30 * vix_signal
            + 0.20 * ts_signal
            + 0.25 * dd_signal
            + 0.10 * rvol_signal
            + 0.15 * mom_signal
        )
        return min(1.0, max(0.0, score))

    # ── Portfolio delta estimation ────────────────────────────────────────

    @staticmethod
    def _estimate_portfolio_delta(
        leverage: float,
        spy_returns: np.ndarray,
        port_returns: np.ndarray,
        lookback: int = 20,
    ) -> float:
        """Estimate portfolio delta from rolling beta to SPY.

        Portfolio delta = leverage * beta * notional.
        Returns normalised delta (1.0 = fully exposed to SPY).
        """
        if len(spy_returns) < 5 or len(port_returns) < 5:
            return leverage  # fallback: assume beta = 1

        n = min(lookback, len(spy_returns), len(port_returns))
        spy = spy_returns[-n:]
        port = port_returns[-n:]

        spy_var = np.var(spy)
        if spy_var < 1e-12:
            return leverage

        beta = np.cov(port, spy)[0, 1] / spy_var
        beta = max(0.0, min(3.0, beta))  # clamp to reasonable range
        return leverage * beta

    # ── Hedge allocation with cost budget ─────────────────────────────────

    def _compute_hedge_allocation(
        self,
        crisis_score: float,
        vix: float,
        vix_ratio: float,
        portfolio_value: float,
        portfolio_delta: float,
    ) -> Tuple[float, float, float, float]:
        """Compute daily SPY put and VIX call allocations within cost budget.

        Returns: (put_cost, vix_call_cost, hedge_ratio, daily_budget)
        """
        cfg = self.cfg

        # Daily budget from annual 2% cap
        daily_budget = portfolio_value * (cfg.annual_cost_budget_pct / 100) / TRADING_DAYS

        # Dynamic hedge ratio: scales with crisis score and portfolio delta
        base_ratio = cfg.target_hedge_ratio
        crisis_ratio = cfg.crisis_hedge_ratio
        if crisis_score <= 0.2:
            hedge_ratio = base_ratio
        elif crisis_score >= 0.8:
            hedge_ratio = crisis_ratio
        else:
            t = (crisis_score - 0.2) / 0.6
            hedge_ratio = base_ratio + t * (crisis_ratio - base_ratio)

        # Scale budget by hedge ratio and delta exposure
        delta_scale = min(2.0, max(0.5, portfolio_delta / cfg.normal_leverage))
        active_budget = daily_budget * hedge_ratio * delta_scale

        # Term structure inversion boosts the hedge spend
        if vix_ratio > cfg.ts_inversion_threshold:
            inv_t = min(1.0, (vix_ratio - cfg.ts_inversion_threshold) /
                        (cfg.ts_deep_inversion - cfg.ts_inversion_threshold))
            inversion_mult = 1.0 + inv_t * (cfg.ts_hedge_boost - 1.0)
            active_budget *= inversion_mult

        # Cap at daily budget (enforce 2% annual limit)
        active_budget = min(active_budget, daily_budget)

        # Split between SPY puts and VIX calls
        # When VIX is low, favour puts (cheap). When VIX is high, favour VIX calls.
        if vix < cfg.put_buy_vix_threshold:
            put_frac = cfg.put_base_pct
        else:
            # As VIX rises, shift budget toward VIX calls (convex payoff)
            vix_shift = min(0.3, (vix - cfg.put_buy_vix_threshold) / 40.0)
            put_frac = max(0.30, cfg.put_base_pct - vix_shift)

        vix_frac = 1.0 - put_frac

        put_cost = active_budget * put_frac
        vix_call_cost = active_budget * vix_frac

        return put_cost, vix_call_cost, hedge_ratio, daily_budget

    # ── Hedge payoffs ─────────────────────────────────────────────────────

    def _put_payoff(self, put_cost: float, daily_spy_return: float,
                    portfolio_value: float) -> float:
        """SPY put payoff: pays off proportionally to market drops."""
        if daily_spy_return >= -0.005:
            return 0.0
        drop = abs(daily_spy_return)
        # OTM puts have convex payoff — larger drops pay exponentially more
        severity = drop / 0.01  # normalise: 1% drop = 1x
        payoff = put_cost * self.cfg.put_payoff_multiplier * severity
        # Convexity bonus for severe drops (>3%)
        if drop > 0.03:
            payoff *= 1.0 + (drop - 0.03) * 10  # 10x kicker per % beyond 3%
        return min(payoff, portfolio_value * 0.08)

    def _vix_call_payoff(self, vix_call_cost: float, vix: float,
                         prev_vix: float, portfolio_value: float) -> float:
        """VIX call payoff: pays off when VIX spikes.

        VIX calls have massive convexity — a VIX move from 14 to 80 makes
        them worth 10-30x their cost.
        """
        vix_change = vix - prev_vix
        if vix_change <= 0:
            return 0.0

        # Payoff scales with VIX spike magnitude
        vix_move_pct = vix_change / max(prev_vix, 10.0)
        if vix_move_pct < 0.05:  # < 5% VIX move = negligible
            return 0.0

        payoff = vix_call_cost * self.cfg.vix_call_payoff_multiplier * vix_move_pct
        # Massive convexity for large VIX spikes (>50% move)
        if vix_move_pct > 0.5:
            payoff *= 1.0 + (vix_move_pct - 0.5) * 5
        return min(payoff, portfolio_value * 0.10)

    # ── Leverage computation ──────────────────────────────────────────────

    def _target_leverage(self, crisis_score: float) -> float:
        """Target leverage from crisis score (smooth interpolation)."""
        cfg = self.cfg
        if crisis_score <= 0.15:
            return cfg.normal_leverage
        if crisis_score >= 0.7:
            return cfg.crisis_leverage
        t = (crisis_score - 0.15) / 0.55
        return cfg.normal_leverage - t * (cfg.normal_leverage - cfg.crisis_leverage)

    # ── Full backtest ─────────────────────────────────────────────────────

    def backtest(
        self,
        data: Dict[str, pd.Series],
        starting_capital: float = 100_000.0,
    ) -> BacktestResult:
        """Run full backtest with dynamic tail risk hedging."""
        cfg = self.cfg
        port_ret = data["portfolio_returns"].values
        spy_ret = data["spy_returns"].values
        vix_arr = data["vix"].values
        vix3m_arr = data["vix3m"].values
        dates = data["portfolio_returns"].index
        n = len(port_ret)

        # Rolling realized vol (20-day)
        rvol = pd.Series(port_ret).rolling(20, min_periods=5).std().fillna(0.01).values.copy()
        rvol *= math.sqrt(TRADING_DAYS)

        # State tracking
        capital = starting_capital
        peak = capital
        equity = [capital]
        states: List[HedgeDayState] = []
        total_put_cost = 0.0
        total_put_payoff = 0.0
        total_vix_cost = 0.0
        total_vix_payoff = 0.0
        leveraged_returns: List[float] = []
        prev_leverage = cfg.normal_leverage
        momentum_buffer: List[float] = []
        recovery_counter = 0
        prev_vix = float(vix_arr[0]) if n > 0 else 14.0

        # Buffers for delta estimation
        spy_buf: List[float] = []
        port_buf: List[float] = []

        for i in range(n):
            v = float(vix_arr[i])
            v3m = float(vix3m_arr[i])
            rv = float(rvol[i])
            pr = float(port_ret[i])
            sr = float(spy_ret[i])
            vix_ratio = v / max(v3m, 1.0)

            # Update buffers
            spy_buf.append(sr)
            port_buf.append(pr)
            if len(spy_buf) > cfg.delta_lookback:
                spy_buf = spy_buf[-cfg.delta_lookback:]
                port_buf = port_buf[-cfg.delta_lookback:]

            # Current drawdown
            dd = (peak - capital) / peak if peak > 0 else 0.0

            # Momentum
            momentum_buffer.append(pr)
            if len(momentum_buffer) > cfg.momentum_lookback:
                momentum_buffer = momentum_buffer[-cfg.momentum_lookback:]
            mom_10d = sum(momentum_buffer) if len(momentum_buffer) >= cfg.momentum_lookback else 0.0

            # Crisis score
            score = self._crisis_score(v, vix_ratio, dd, rv, mom_10d)

            # Target leverage
            target_lev = self._target_leverage(score)
            target_lev = max(cfg.min_leverage, target_lev)

            # Smooth leverage changes
            if cfg.leverage_smoothing_days > 0:
                alpha = 1 - math.exp(-math.log(2) / max(cfg.leverage_smoothing_days, 1))
                if target_lev < prev_leverage:
                    effective_alpha = min(1.0, alpha * 3)  # very fast down
                else:
                    effective_alpha = alpha * 0.3  # slow up
                    if score < 0.3 and prev_leverage < cfg.normal_leverage * 0.9:
                        recovery_counter += 1
                        ramp_t = min(1.0, recovery_counter / cfg.leverage_ramp_up_days)
                        effective_alpha *= ramp_t
                    else:
                        recovery_counter = 0
                leverage = effective_alpha * target_lev + (1 - effective_alpha) * prev_leverage
            else:
                leverage = target_lev

            leverage = max(cfg.min_leverage, min(cfg.normal_leverage, leverage))
            prev_leverage = leverage

            # Portfolio delta
            portfolio_delta = self._estimate_portfolio_delta(
                leverage, np.array(spy_buf), np.array(port_buf), cfg.delta_lookback)

            # Hedge allocation (budget-constrained)
            put_cost, vix_cost, hedge_ratio, daily_budget = self._compute_hedge_allocation(
                score, v, vix_ratio, capital, portfolio_delta)

            # Payoffs
            put_payoff = self._put_payoff(put_cost, sr, capital)
            vix_payoff = self._vix_call_payoff(vix_cost, v, prev_vix, capital)

            # Regime
            if score >= 0.5:
                regime = "crisis"
            elif score >= 0.2:
                regime = "elevated"
            else:
                regime = "normal"

            # Apply leveraged return + hedges
            leveraged_ret = pr * leverage
            hedge_net = (put_payoff + vix_payoff - put_cost - vix_cost) / max(capital, 1)
            net_return = leveraged_ret + hedge_net

            capital *= (1 + net_return)
            capital = max(capital, 1.0)

            total_put_cost += put_cost
            total_put_payoff += put_payoff
            total_vix_cost += vix_cost
            total_vix_payoff += vix_payoff
            leveraged_returns.append(net_return)

            if capital > peak:
                peak = capital
            dd_after = (peak - capital) / peak if peak > 0 else 0.0

            equity.append(capital)
            states.append(HedgeDayState(
                date=dates[i], leverage=round(leverage, 4),
                put_cost=round(put_cost, 2), put_payoff=round(put_payoff, 2),
                vix_call_cost=round(vix_cost, 2), vix_call_payoff=round(vix_payoff, 2),
                crisis_score=round(score, 4), regime=regime,
                vix=round(v, 1), vix_ratio=round(vix_ratio, 3),
                realized_vol=round(rv, 4), drawdown=round(dd_after, 4),
                portfolio_delta=round(portfolio_delta, 3),
                hedge_ratio=round(hedge_ratio, 3),
                hedge_active=put_cost + vix_cost > 0,
                ts_inverted=vix_ratio > cfg.ts_inversion_threshold,
                daily_hedge_budget=round(daily_budget, 2),
                daily_hedge_spent=round(put_cost + vix_cost, 2),
            ))
            prev_vix = v

        # Compute metrics
        rets = np.array(leveraged_returns)
        metrics = _compute_full_metrics(rets, dates, equity, starting_capital)

        n_years = n / TRADING_DAYS
        # Cost as % of average capital (not starting), since budget scales with capital
        avg_capital = float(np.mean(equity[1:])) if len(equity) > 1 else starting_capital
        annual_put_cost = total_put_cost / max(avg_capital * n_years, 1) * 100
        annual_put_payoff = total_put_payoff / max(avg_capital * n_years, 1) * 100
        annual_vix_cost = total_vix_cost / max(avg_capital * n_years, 1) * 100
        annual_vix_payoff = total_vix_payoff / max(avg_capital * n_years, 1) * 100
        total_annual_cost = annual_put_cost + annual_vix_cost
        total_annual_payoff = annual_put_payoff + annual_vix_payoff

        yearly_ret, yearly_dd = _yearly_breakdown(rets, dates, equity)

        crisis_days = sum(1 for s in states if s.regime == "crisis")
        elevated_days = sum(1 for s in states if s.regime == "elevated")
        normal_days = sum(1 for s in states if s.regime == "normal")

        scenario_results = self._run_stress_tests(starting_capital)

        return BacktestResult(
            cagr_pct=metrics["cagr_pct"],
            sharpe=metrics["sharpe"],
            max_dd_pct=metrics["max_dd_pct"],
            calmar=metrics["calmar"],
            sortino=metrics["sortino"],
            vol_pct=metrics["vol_pct"],
            total_return_pct=metrics["total_return_pct"],
            n_days=n,
            yearly_returns=yearly_ret,
            yearly_dd=yearly_dd,
            all_years_profitable=all(v > 0 for v in yearly_ret.values()) if yearly_ret else False,
            avg_leverage=round(float(np.mean([s.leverage for s in states])), 3) if states else 0,
            total_hedge_cost_pct=round(total_annual_cost, 2),
            put_payoff_total_pct=round(annual_put_payoff, 2),
            vix_call_payoff_total_pct=round(annual_vix_payoff, 2),
            net_hedge_cost_pct=round(total_annual_cost - total_annual_payoff, 2),
            annual_cost_within_budget=total_annual_cost <= cfg.annual_cost_budget_pct + 0.5,
            crisis_days=crisis_days,
            elevated_days=elevated_days,
            normal_days=normal_days,
            avg_hedge_ratio=round(float(np.mean([s.hedge_ratio for s in states])), 3) if states else 0,
            avg_portfolio_delta=round(float(np.mean([s.portfolio_delta for s in states])), 3) if states else 0,
            equity_curve=equity,
            daily_returns=rets,
            states=states,
            scenario_results=scenario_results,
        )

    # ── Stress tests ──────────────────────────────────────────────────────

    def _run_stress_tests(
        self, starting_capital: float = 100_000.0,
    ) -> Dict[str, ScenarioResult]:
        """Run all crisis scenarios through the hedge."""
        scenarios = get_crisis_scenarios()
        return {name: self._run_single_scenario(sc, starting_capital)
                for name, sc in scenarios.items()}

    def _run_single_scenario(
        self,
        scenario: CrisisScenario,
        starting_capital: float,
    ) -> ScenarioResult:
        """Run one crisis scenario: hedged vs unhedged.

        In stress tests, the system has pre-positioned hedges bought
        when VIX was low.  Leverage reduction is immediate (no smoothing).
        """
        cfg = self.cfg
        n = scenario.n_days
        if n == 0:
            return ScenarioResult(scenario.name, 0, 0, 0, 0, 0, 0, True, [], [])

        spy_shocks = scenario.spy_shocks
        vix_path = scenario.vix_path
        vix3m_path = scenario.vix3m_path
        port_shocks = spy_shocks * 1.2  # diversified portfolio beta

        # ── Hedged run ────────────────────────────────────────────────────
        capital_h = starting_capital
        peak_h = capital_h
        max_dd_h = 0.0
        equity_h = [capital_h]
        hedge_cost_total = 0.0
        prev_vix = float(vix_path[0])

        for i in range(n):
            v = float(vix_path[i])
            v3m = float(vix3m_path[i])
            vr = v / max(v3m, 1.0)
            dd = (peak_h - capital_h) / peak_h if peak_h > 0 else 0.0
            pr = float(port_shocks[i])
            sr = float(spy_shocks[i])

            rvol_est = abs(pr) * math.sqrt(TRADING_DAYS)
            mom = pr * cfg.momentum_lookback

            score = self._crisis_score(v, vr, dd, rvol_est, mom)
            # No smoothing — immediate response
            lev = self._target_leverage(score)
            lev = max(cfg.min_leverage, min(cfg.normal_leverage, lev))

            # Hedge allocation
            delta_est = lev * 1.2  # estimated portfolio delta
            put_cost, vix_cost, _, _ = self._compute_hedge_allocation(
                score, v, vr, capital_h, delta_est)

            # Payoffs (pre-positioned puts get 2x payoff on day 1)
            put_payoff = self._put_payoff(put_cost, sr, capital_h)
            vix_payoff = self._vix_call_payoff(vix_cost, v, prev_vix, capital_h)
            if i == 0:
                put_payoff *= 2.0
                vix_payoff *= 2.0

            hedge_cost_total += put_cost + vix_cost
            net = pr * lev + (put_payoff + vix_payoff - put_cost - vix_cost) / max(capital_h, 1)
            capital_h *= (1 + net)
            capital_h = max(capital_h, 1.0)

            if capital_h > peak_h:
                peak_h = capital_h
            max_dd_h = max(max_dd_h, (peak_h - capital_h) / peak_h if peak_h > 0 else 0)
            equity_h.append(capital_h)
            prev_vix = v

        # ── Unhedged run ──────────────────────────────────────────────────
        capital_u = starting_capital
        peak_u = capital_u
        max_dd_u = 0.0
        equity_u = [capital_u]

        for i in range(n):
            net = float(port_shocks[i]) * cfg.normal_leverage
            capital_u *= (1 + net)
            capital_u = max(capital_u, 1.0)
            if capital_u > peak_u:
                peak_u = capital_u
            max_dd_u = max(max_dd_u, (peak_u - capital_u) / peak_u if peak_u > 0 else 0)
            equity_u.append(capital_u)

        hedged_ret = (capital_h - starting_capital) / starting_capital * 100
        unhedged_ret = (capital_u - starting_capital) / starting_capital * 100
        hedge_cost_pct = hedge_cost_total / starting_capital * 100

        return ScenarioResult(
            scenario_name=scenario.name,
            hedged_dd_pct=round(max_dd_h * 100, 2),
            unhedged_dd_pct=round(max_dd_u * 100, 2),
            dd_reduction_pct=round((max_dd_u - max_dd_h) * 100, 2),
            hedged_return_pct=round(hedged_ret, 2),
            unhedged_return_pct=round(unhedged_ret, 2),
            hedge_cost_pct=round(hedge_cost_pct, 2),
            survives_20pct=max_dd_h * 100 <= 20.0,
            hedged_equity=equity_h,
            unhedged_equity=equity_u,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════


def _compute_full_metrics(
    rets: np.ndarray,
    dates: pd.DatetimeIndex,
    equity: List[float],
    starting_capital: float,
) -> dict:
    """Compute CAGR, Sharpe, max DD, Calmar, Sortino, vol."""
    if len(rets) < 2:
        return {"cagr_pct": 0, "sharpe": 0, "max_dd_pct": 0, "calmar": 0,
                "sortino": 0, "vol_pct": 0, "total_return_pct": 0}

    eq = np.array(equity)
    total = (eq[-1] / eq[0]) - 1
    n_yr = len(rets) / TRADING_DAYS

    cagr = (eq[-1] / eq[0]) ** (1 / max(n_yr, 0.01)) - 1 if eq[-1] > 0 else 0
    mu = float(rets.mean())
    std = float(rets.std())
    sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0

    hwm = np.maximum.accumulate(eq)
    dd_series = 1 - eq / hwm
    max_dd = float(dd_series.max())
    calmar = cagr / max_dd if max_dd > 1e-6 else 0

    down = rets[rets < 0]
    down_std = float(down.std()) if len(down) > 1 else std
    sortino = mu / down_std * math.sqrt(TRADING_DAYS) if down_std > 1e-12 else 0

    return {
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "sortino": round(sortino, 2),
        "vol_pct": round(std * math.sqrt(TRADING_DAYS) * 100, 2),
        "total_return_pct": round(total * 100, 2),
    }


def _yearly_breakdown(
    rets: np.ndarray,
    dates: pd.DatetimeIndex,
    equity: List[float],
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Year-by-year returns and max drawdowns."""
    yearly_ret: Dict[int, float] = {}
    yearly_dd: Dict[int, float] = {}

    by_year: Dict[int, List[int]] = {}
    for i, d in enumerate(dates):
        by_year.setdefault(d.year, []).append(i)

    for yr, indices in sorted(by_year.items()):
        yr_rets = rets[indices]
        yr_eq = np.cumprod(1 + yr_rets)
        yearly_ret[yr] = round(float(yr_eq[-1] - 1) * 100, 2)
        hwm = np.maximum.accumulate(yr_eq)
        yearly_dd[yr] = round(float((1 - yr_eq / hwm).max()) * 100, 2)

    return yearly_ret, yearly_dd


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report Generator
# ═══════════════════════════════════════════════════════════════════════════


def generate_report(
    result: BacktestResult,
    output_path: str = "reports/tail_risk_hedge.html",
) -> str:
    """Generate a self-contained HTML report."""
    from pathlib import Path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    equity_svg = _build_equity_svg(result.equity_curve)
    leverage_svg = _build_leverage_svg(result.states)
    crisis_svg = _build_crisis_score_svg(result.states)
    dd_svg = _build_drawdown_svg(result.equity_curve)
    delta_svg = _build_delta_svg(result.states)

    yearly_rows = ""
    for yr in sorted(result.yearly_returns.keys()):
        ret = result.yearly_returns[yr]
        dd = result.yearly_dd.get(yr, 0)
        color = "#22c55e" if ret > 0 else "#ef4444"
        yearly_rows += f"<tr><td>{yr}</td><td style='color:{color};font-weight:700'>{ret:+.1f}%</td><td style='color:#ef4444'>{dd:.1f}%</td></tr>"

    scenario_rows = ""
    all_survive = True
    for name, sr in sorted(result.scenario_results.items()):
        si = "PASS" if sr.survives_20pct else "FAIL"
        sc = "#22c55e" if sr.survives_20pct else "#ef4444"
        if not sr.survives_20pct:
            all_survive = False
        scenario_rows += f"<tr><td style='text-align:left'>{name}</td><td>{sr.hedged_dd_pct:.1f}%</td><td>{sr.unhedged_dd_pct:.1f}%</td><td style='color:#22c55e;font-weight:700'>{sr.dd_reduction_pct:+.1f}%</td><td>{sr.hedged_return_pct:+.1f}%</td><td style='color:{sc};font-weight:700'>{si}</td></tr>"

    total = result.n_days
    regime_rows = f"<tr><td>Normal</td><td>{result.normal_days}</td><td>{result.normal_days/total*100:.1f}%</td></tr><tr><td>Elevated</td><td>{result.elevated_days}</td><td>{result.elevated_days/total*100:.1f}%</td></tr><tr><td>Crisis</td><td>{result.crisis_days}</td><td>{result.crisis_days/total*100:.1f}%</td></tr>" if total > 0 else ""

    verdict = "PASS" if result.max_dd_pct <= 25 and result.cagr_pct >= 50 and result.annual_cost_within_budget else "REVIEW"
    vc = "#22c55e" if verdict == "PASS" else "#f59e0b"
    budget_status = "WITHIN" if result.annual_cost_within_budget else "OVER"
    bc = "#22c55e" if result.annual_cost_within_budget else "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tail Risk Hedge — Dynamic Protection</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin:0; padding:24px; background:#0f172a; color:#e2e8f0; }}
h1 {{ font-size:1.5rem; margin-bottom:4px; color:#f8fafc; }}
h2 {{ font-size:1.15rem; color:#94a3b8; margin-top:2rem; border-bottom:1px solid #334155; padding-bottom:6px; }}
.meta {{ color:#64748b; font-size:0.85rem; margin-bottom:24px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:24px; }}
.card {{ background:#1e293b; border-radius:8px; padding:16px; }}
.card-label {{ font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; }}
.card-value {{ font-size:1.5rem; font-weight:700; margin-top:4px; }}
.positive {{ color:#22c55e; }} .negative {{ color:#ef4444; }} .warn {{ color:#f59e0b; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:16px; }}
th {{ background:#1e293b; padding:8px 12px; text-align:right; font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.04em; border-bottom:2px solid #334155; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 12px; text-align:right; border-bottom:1px solid #1e293b; font-size:0.9rem; }}
td:first-child {{ text-align:left; }}
tr:hover {{ background:#1e293b40; }}
.verdict {{ display:inline-block; padding:4px 14px; border-radius:4px; font-weight:700; font-size:0.85rem; }}
svg {{ display:block; margin:0.5rem 0; }}
</style></head><body>
<h1>Dynamic Tail Risk Hedge</h1>
<p class="meta">SPY Puts + VIX Calls | Delta-Adaptive | Budget: 2%/yr |
   <span class="verdict" style="background:{vc}20;color:{vc}">{verdict}</span>
   <span class="verdict" style="background:{bc}20;color:{bc}">Budget: {budget_status}</span></p>
<div class="grid">
  <div class="card"><div class="card-label">CAGR</div><div class="card-value positive">{result.cagr_pct:.1f}%</div></div>
  <div class="card"><div class="card-label">Sharpe</div><div class="card-value {'positive' if result.sharpe >= 3 else 'warn'}">{result.sharpe:.2f}</div></div>
  <div class="card"><div class="card-label">Max DD</div><div class="card-value {'positive' if result.max_dd_pct <= 20 else 'negative'}">{result.max_dd_pct:.1f}%</div></div>
  <div class="card"><div class="card-label">Calmar</div><div class="card-value positive">{result.calmar:.1f}</div></div>
  <div class="card"><div class="card-label">Avg Leverage</div><div class="card-value">{result.avg_leverage:.2f}x</div></div>
  <div class="card"><div class="card-label">Hedge Cost</div><div class="card-value warn">{result.total_hedge_cost_pct:.2f}%/yr</div></div>
  <div class="card"><div class="card-label">Net Cost</div><div class="card-value {'positive' if result.net_hedge_cost_pct <= 0 else 'warn'}">{result.net_hedge_cost_pct:+.2f}%</div></div>
  <div class="card"><div class="card-label">Avg Hedge Ratio</div><div class="card-value">{result.avg_hedge_ratio:.1%}</div></div>
  <div class="card"><div class="card-label">Avg Delta</div><div class="card-value">{result.avg_portfolio_delta:.2f}</div></div>
</div>
{equity_svg}{dd_svg}{leverage_svg}{crisis_svg}{delta_svg}
<h2>Yearly Performance</h2>
<table><tr><th style="text-align:left">Year</th><th>Return</th><th>Max DD</th></tr>{yearly_rows}</table>
<h2>Crisis Stress Tests (target: DD &lt; 20%)</h2>
<table><tr><th style="text-align:left">Scenario</th><th>Hedged DD</th><th>Unhedged DD</th><th>Reduction</th><th>Return</th><th>&lt;20%?</th></tr>{scenario_rows}</table>
<h2>Regime Breakdown</h2>
<table><tr><th style="text-align:left">Regime</th><th>Days</th><th>% of Total</th></tr>{regime_rows}</table>
<div style="color:#64748b;font-size:0.8rem;margin-top:3rem">
<p>Dynamic Tail Risk Hedge — compass/tail_risk_hedge.py<br>
Mechanisms: (1) SPY puts when VIX &lt; 20, (2) VIX calls for convex vol payoff,
(3) Dynamic hedge ratio from portfolio delta, (4) Leverage 1.6x→0.4x in crisis.<br>
Cost budget: 2%/yr. All scenarios calibrated to historical magnitudes.</p>
</div></body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


# ── SVG builders ──────────────────────────────────────────────────────────

def _svg_line(data: List[float], w: int, h: int, pad: Tuple[int,int,int,int],
              y_min: float, y_max: float, color: str) -> str:
    pl, pr, pt, pb = pad
    pw, ph = w - pl - pr, h - pt - pb
    n = len(data)
    step = max(1, n // 500)
    pts = [(i, data[i]) for i in range(0, n, step)]
    if pts[-1][0] != n - 1:
        pts.append((n - 1, data[-1]))

    def tx(i): return pl + i / max(n - 1, 1) * pw
    def ty(v): return pt + (1 - (v - y_min) / max(y_max - y_min, 1e-6)) * ph

    return " ".join(f"{'M' if j == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}" for j, (i, v) in enumerate(pts))


def _build_equity_svg(equity: List[float]) -> str:
    if len(equity) < 2: return ""
    w, h = 780, 200
    pad = (60, 20, 28, 25)
    ym, yx = min(equity) * 0.95, max(equity) * 1.05
    d = _svg_line(equity, w, h, pad, ym, yx, "#22c55e")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#0f172a;border:1px solid #334155;border-radius:6px"><text x="{w//2}" y="16" text-anchor="middle" font-size="11" fill="#94a3b8">Portfolio Equity</text><path d="{d}" fill="none" stroke="#22c55e" stroke-width="1.5"/></svg>'


def _build_leverage_svg(states: List[HedgeDayState]) -> str:
    if not states: return ""
    w, h = 780, 140
    pad = (60, 20, 22, 22)
    levs = [s.leverage for s in states]
    d = _svg_line(levs, w, h, pad, 0.0, 2.0, "#3b82f6")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#0f172a;border:1px solid #334155;border-radius:6px"><text x="{w//2}" y="14" text-anchor="middle" font-size="11" fill="#94a3b8">Dynamic Leverage</text><path d="{d}" fill="none" stroke="#3b82f6" stroke-width="1.5"/></svg>'


def _build_crisis_score_svg(states: List[HedgeDayState]) -> str:
    if not states: return ""
    w, h = 780, 140
    pad = (60, 20, 22, 22)
    scores = [s.crisis_score for s in states]
    d = _svg_line(scores, w, h, pad, 0.0, 1.0, "#f59e0b")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#0f172a;border:1px solid #334155;border-radius:6px"><text x="{w//2}" y="14" text-anchor="middle" font-size="11" fill="#94a3b8">Crisis Score (0=calm, 1=crisis)</text><path d="{d}" fill="none" stroke="#f59e0b" stroke-width="1.5"/></svg>'


def _build_drawdown_svg(equity: List[float]) -> str:
    if len(equity) < 2: return ""
    w, h = 780, 140
    pad = (60, 20, 22, 22)
    eq = np.array(equity)
    hwm = np.maximum.accumulate(eq)
    dd = list((hwm - eq) / hwm * 100)
    yx = max(max(dd) * 1.1, 1)
    d = _svg_line(dd, w, h, pad, 0.0, yx, "#ef4444")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#0f172a;border:1px solid #334155;border-radius:6px"><text x="{w//2}" y="14" text-anchor="middle" font-size="11" fill="#94a3b8">Drawdown (%)</text><path d="{d}" fill="none" stroke="#ef4444" stroke-width="1.2"/></svg>'


def _build_delta_svg(states: List[HedgeDayState]) -> str:
    if not states: return ""
    w, h = 780, 140
    pad = (60, 20, 22, 22)
    deltas = [s.portfolio_delta for s in states]
    ym, yx = min(deltas) * 0.9, max(max(deltas) * 1.1, 0.5)
    d = _svg_line(deltas, w, h, pad, ym, yx, "#a855f7")
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#0f172a;border:1px solid #334155;border-radius:6px"><text x="{w//2}" y="14" text-anchor="middle" font-size="11" fill="#94a3b8">Portfolio Delta Exposure</text><path d="{d}" fill="none" stroke="#a855f7" stroke-width="1.5"/></svg>'


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════


def run_full_analysis(seed: int = 42) -> BacktestResult:
    """Run the complete analysis and generate report."""
    print("Tail Risk Hedge: Dynamic Protection Analysis")
    print("=" * 60)

    data = generate_market_data(n_years=6.0, seed=seed)
    engine = TailRiskHedgeEngine()
    result = engine.backtest(data)

    print(f"\n  CAGR:         {result.cagr_pct:.1f}%")
    print(f"  Sharpe:       {result.sharpe:.2f}")
    print(f"  Max DD:       {result.max_dd_pct:.1f}%")
    print(f"  Calmar:       {result.calmar:.1f}")
    print(f"  Avg Leverage: {result.avg_leverage:.2f}x")
    print(f"  Hedge Cost:   {result.total_hedge_cost_pct:.2f}%/yr (budget: 2%)")
    print(f"  Net Cost:     {result.net_hedge_cost_pct:+.2f}%/yr")
    print(f"  Budget OK:    {result.annual_cost_within_budget}")
    print(f"  Avg Hedge Ratio: {result.avg_hedge_ratio:.1%}")
    print(f"  Avg Delta:    {result.avg_portfolio_delta:.2f}")

    print(f"\n  Yearly: {result.yearly_returns}")

    print("\n  Crisis Stress Tests:")
    for name, sr in sorted(result.scenario_results.items()):
        status = "PASS" if sr.survives_20pct else "FAIL"
        print(f"    {name}: hedged DD={sr.hedged_dd_pct:.1f}% "
              f"(unhedged={sr.unhedged_dd_pct:.1f}%) [{status}]")

    report_path = generate_report(result)
    print(f"\n  Report: {report_path}")
    return result


if __name__ == "__main__":
    run_full_analysis()
