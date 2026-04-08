"""
Dynamic hedging engine for options portfolios.

Delta-neutral hedging with frequency optimisation, tail-risk OTM put
overlays sized by VaR, cross-hedging with correlated assets, regime-
adaptive hedge ratios, cost optimisation, and effectiveness tracking.

Usage::

    from compass.dynamic_hedging import DynamicHedgingEngine, HedgeConfig
    engine = DynamicHedgingEngine(HedgeConfig())
    engine.add_portfolio_snapshot(delta=30, gamma=5, vega=200, ...)
    actions = engine.compute_hedges(regime="bull", vix=18)
    bt = engine.backtest(returns, regimes)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class HedgeConfig:
    # Delta hedging
    delta_threshold: float = 10.0       # rebalance when |delta| > this
    max_hedge_shares: int = 5000        # cap on share hedge
    share_cost_bps: float = 1.0         # round-trip cost per share in bps
    # Tail risk
    put_otm_pct: float = 0.05           # 5% OTM puts
    var_confidence: float = 0.05        # 5th percentile VaR
    max_put_cost_pct: float = 0.03      # max 3% of portfolio on puts/year
    put_dte: int = 30                   # days to expiry for put hedge
    # VIX overlay
    vix_call_trigger: float = 0.95      # buy VIX calls when term ratio < this (backwardation)
    vix_call_budget_pct: float = 0.005  # 0.5% per event
    # Cross-hedge
    cross_hedge_assets: Dict[str, float] = field(default_factory=lambda: {
        "XLF": 0.85, "XLK": 0.90, "XLE": 0.60, "XLV": 0.70,
    })
    min_cross_corr: float = 0.60
    # Regime multipliers
    regime_hedge_ratio: Dict[str, float] = field(default_factory=lambda: {
        "bull": 0.3, "neutral": 0.6, "bear": 1.0,
        "high_vol": 1.2, "crash": 1.5,
    })
    # Frequency
    min_rebalance_hours: float = 4.0    # minimum time between rebalances
    kelly_rebalance_mult: float = 1.5   # Kelly factor for rebalance trigger


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class PortfolioState:
    """Current portfolio Greeks and value."""
    timestamp: str
    portfolio_value: float
    delta: float
    gamma: float
    vega: float
    theta: float
    n_positions: int


@dataclass
class HedgeAction:
    """A single hedge action to execute."""
    action_type: str          # "delta_hedge", "tail_put", "vix_call", "cross_hedge"
    instrument: str           # "SPY_shares", "SPY_put_410", "VIX_call_25", "XLF_shares"
    direction: str            # "buy" or "sell"
    quantity: int
    estimated_cost: float     # $ cost of the hedge
    estimated_delta: float    # delta impact of the hedge
    reason: str
    urgency: str              # "low", "medium", "high"


@dataclass
class HedgePlan:
    """Complete hedge plan for one period."""
    timestamp: str
    regime: str
    vix: float
    portfolio_delta: float
    hedge_ratio: float        # regime-adjusted
    actions: List[HedgeAction]
    total_cost: float
    residual_delta: float     # delta after hedging
    protection_level: float   # estimated DD protection %
    rebalance_triggered: bool


@dataclass
class TailHedge:
    """OTM put tail hedge specification."""
    strike: float
    dte: int
    contracts: int
    premium_per_contract: float
    total_cost: float
    notional_protected: float
    cost_as_pct: float
    var_coverage: float       # fraction of VaR covered


@dataclass
class CrossHedge:
    """Cross-asset hedge specification."""
    asset: str
    correlation: float
    shares: int
    estimated_cost: float
    hedge_effectiveness: float  # R² of hedge


@dataclass
class HedgeEffectiveness:
    """P&L attribution: alpha vs hedge cost."""
    period: str
    gross_pnl: float          # portfolio P&L before hedging
    hedge_pnl: float          # P&L from hedges
    hedge_cost: float         # cost of hedges
    net_pnl: float            # gross + hedge_pnl - cost
    dd_unhedged: float
    dd_hedged: float
    dd_reduction_pct: float
    cost_as_pct_of_gross: float
    sharpe_unhedged: float
    sharpe_hedged: float


@dataclass
class BacktestResult:
    """Hedging backtest result."""
    n_periods: int
    n_rebalances: int
    total_hedge_cost: float
    hedge_cost_annual_pct: float
    unhedged_dd: float
    hedged_dd: float
    dd_reduction_pct: float
    unhedged_sharpe: float
    hedged_sharpe: float
    sharpe_improvement_pct: float
    unhedged_pnl: float
    hedged_pnl: float
    effectiveness: List[HedgeEffectiveness]
    by_regime: Dict[str, Dict[str, float]]


# ── Core computations ───────────────────────────────────────────────────


def compute_var(returns: np.ndarray, confidence: float = 0.05) -> float:
    """Historical VaR at given confidence level."""
    if len(returns) < 10:
        return 0.0
    return float(-np.percentile(returns, confidence * 100))


def optimal_hedge_ratio(
    portfolio_returns: np.ndarray,
    hedge_returns: np.ndarray,
) -> float:
    """Minimum-variance hedge ratio: β = cov(p, h) / var(h)."""
    if len(portfolio_returns) < 10 or len(hedge_returns) < 10:
        return 1.0
    n = min(len(portfolio_returns), len(hedge_returns))
    p, h = portfolio_returns[:n], hedge_returns[:n]
    var_h = np.var(h)
    if var_h < 1e-15:
        return 0.0
    return float(np.cov(p, h)[0, 1] / var_h)


def kelly_rebalance_trigger(
    current_delta: float,
    threshold: float,
    gamma: float,
    expected_move: float,
    kelly_mult: float = 1.5,
) -> bool:
    """Kelly-based rebalance: trigger when expected delta drift exceeds threshold."""
    expected_delta_change = abs(gamma * expected_move)
    future_delta = abs(current_delta) + expected_delta_change * kelly_mult
    return future_delta > threshold


def price_otm_put(
    underlying: float, strike: float, dte: int,
    vol: float, rate: float = 0.045,
) -> float:
    """Black-Scholes OTM put price (simplified)."""
    if dte <= 0 or vol <= 0:
        return max(strike - underlying, 0)
    T = dte / 365.0
    sqrt_t = math.sqrt(T)
    d1 = (math.log(underlying / strike) + (rate + 0.5 * vol**2) * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    nd1 = 0.5 * (1 + math.erf(-d1 / math.sqrt(2)))
    nd2 = 0.5 * (1 + math.erf(-d2 / math.sqrt(2)))
    return float(strike * math.exp(-rate * T) * nd2 - underlying * nd1)


# ── Engine ──────────────────────────────────────────────────────────────


class DynamicHedgingEngine:
    """Continuous portfolio hedging engine."""

    def __init__(self, config: Optional[HedgeConfig] = None) -> None:
        self.config = config or HedgeConfig()
        self._state: Optional[PortfolioState] = None
        self._plans: List[HedgePlan] = []
        self._last_rebalance_ts: float = 0.0

    def add_portfolio_snapshot(
        self, delta: float, gamma: float = 0, vega: float = 0,
        theta: float = 0, portfolio_value: float = 100_000,
        n_positions: int = 1,
    ) -> PortfolioState:
        """Update portfolio state."""
        self._state = PortfolioState(
            timestamp=datetime.now(timezone.utc).isoformat(),
            portfolio_value=portfolio_value,
            delta=delta, gamma=gamma, vega=vega, theta=theta,
            n_positions=n_positions,
        )
        return self._state

    # ── Hedge computation ───────────────────────────────────────────────

    def compute_hedges(
        self,
        regime: str = "neutral",
        vix: float = 18.0,
        underlying_price: float = 430.0,
        vix_term_ratio: Optional[float] = None,
        portfolio_returns: Optional[np.ndarray] = None,
    ) -> HedgePlan:
        """Compute hedge actions for current state."""
        if self._state is None:
            self.add_portfolio_snapshot(delta=0)

        state = self._state
        cfg = self.config
        hedge_ratio = cfg.regime_hedge_ratio.get(regime, 0.6)
        actions: List[HedgeAction] = []
        total_cost = 0.0

        # 1. Delta hedge
        adj_delta = state.delta * hedge_ratio
        rebalance = abs(adj_delta) > cfg.delta_threshold
        if rebalance:
            da = self._delta_hedge(adj_delta, underlying_price)
            if da:
                actions.append(da)
                total_cost += da.estimated_cost

        # 2. Tail risk puts
        if portfolio_returns is not None and len(portfolio_returns) > 20:
            th = self._tail_hedge(
                state.portfolio_value, underlying_price, vix / 100,
                portfolio_returns,
            )
            if th:
                actions.append(HedgeAction(
                    "tail_put", f"SPY_put_{th.strike:.0f}",
                    "buy", th.contracts, th.total_cost, 0,
                    f"VaR protection: {th.var_coverage:.0%} coverage",
                    "medium" if regime in ("bear", "high_vol", "crash") else "low",
                ))
                total_cost += th.total_cost

        # 3. VIX call overlay
        if vix_term_ratio is not None and vix_term_ratio < cfg.vix_call_trigger:
            va = self._vix_overlay(state.portfolio_value, vix)
            if va:
                actions.append(va)
                total_cost += va.estimated_cost

        # 4. Cross-hedge (when delta hedge insufficient)
        if rebalance and abs(adj_delta) > cfg.delta_threshold * 2:
            ch = self._cross_hedge(adj_delta, underlying_price, regime)
            actions.extend(ch)
            total_cost += sum(a.estimated_cost for a in ch)

        # Residual delta
        hedge_delta = sum(a.estimated_delta for a in actions)
        residual = state.delta + hedge_delta

        # Protection estimate
        protection = min(hedge_ratio * 0.5, 0.60)  # rough estimate

        plan = HedgePlan(
            timestamp=datetime.now(timezone.utc).isoformat(),
            regime=regime, vix=vix,
            portfolio_delta=state.delta, hedge_ratio=hedge_ratio,
            actions=actions, total_cost=total_cost,
            residual_delta=residual, protection_level=protection,
            rebalance_triggered=rebalance,
        )
        self._plans.append(plan)
        return plan

    def _delta_hedge(self, target_delta: float, price: float) -> Optional[HedgeAction]:
        """Compute share-based delta hedge."""
        shares = -int(round(target_delta))
        shares = max(-self.config.max_hedge_shares, min(self.config.max_hedge_shares, shares))
        if shares == 0:
            return None
        cost = abs(shares) * price * self.config.share_cost_bps / 10_000
        direction = "buy" if shares > 0 else "sell"
        return HedgeAction(
            "delta_hedge", "SPY_shares", direction, abs(shares),
            cost, float(shares),
            f"Delta hedge: {target_delta:+.1f} → neutralise with {shares:+d} shares",
            "high" if abs(target_delta) > self.config.delta_threshold * 2 else "medium",
        )

    def _tail_hedge(
        self, portfolio_value: float, price: float,
        vol: float, returns: np.ndarray,
    ) -> Optional[TailHedge]:
        """Compute OTM put tail hedge sized by VaR."""
        var = compute_var(returns, self.config.var_confidence)
        if var <= 0:
            return None

        strike = price * (1 - self.config.put_otm_pct)
        premium = price_otm_put(price, strike, self.config.put_dte, max(vol, 0.10))
        if premium <= 0:
            premium = price * 0.005  # floor

        # Size: cover VaR
        var_dollars = var * portfolio_value
        notional_per_contract = 100 * price
        contracts = max(1, int(var_dollars / notional_per_contract))

        total_cost = premium * contracts * 100
        max_budget = portfolio_value * self.config.max_put_cost_pct / 12  # monthly budget
        if total_cost > max_budget:
            contracts = max(1, int(max_budget / (premium * 100)))
            total_cost = premium * contracts * 100

        notional_protected = contracts * 100 * (price - strike)
        coverage = notional_protected / var_dollars if var_dollars > 0 else 0

        return TailHedge(
            strike=strike, dte=self.config.put_dte,
            contracts=contracts, premium_per_contract=premium,
            total_cost=total_cost,
            notional_protected=notional_protected,
            cost_as_pct=total_cost / portfolio_value,
            var_coverage=min(coverage, 1.0),
        )

    def _vix_overlay(self, portfolio_value: float, vix: float) -> Optional[HedgeAction]:
        """VIX call overlay when term structure inverts."""
        budget = portfolio_value * self.config.vix_call_budget_pct
        vix_call_price = vix * 0.10  # rough: 10% of VIX level per contract
        contracts = max(1, int(budget / (vix_call_price * 100)))
        cost = vix_call_price * contracts * 100

        return HedgeAction(
            "vix_call", f"VIX_call_{int(vix + 5)}",
            "buy", contracts, cost, 0,
            f"VIX backwardation detected — buy {contracts} VIX calls for crisis alpha",
            "medium",
        )

    def _cross_hedge(
        self, target_delta: float, price: float, regime: str,
    ) -> List[HedgeAction]:
        """Cross-hedge with correlated sector ETFs."""
        actions: List[HedgeAction] = []
        remaining = target_delta

        for asset, corr in sorted(self.config.cross_hedge_assets.items(), key=lambda x: -x[1]):
            if abs(corr) < self.config.min_cross_corr:
                continue
            if abs(remaining) < self.config.delta_threshold * 0.5:
                break

            # Shares needed, adjusted for correlation
            shares = -int(round(remaining * 0.3 / corr))  # 30% of remaining via cross-hedge
            if shares == 0:
                continue
            shares = max(-1000, min(1000, shares))
            cost = abs(shares) * price * 0.5 * self.config.share_cost_bps / 10_000
            direction = "buy" if shares > 0 else "sell"

            actions.append(HedgeAction(
                "cross_hedge", f"{asset}_shares", direction, abs(shares),
                cost, float(shares * corr),
                f"Cross-hedge via {asset} (corr={corr:.2f}): {shares:+d} shares",
                "low",
            ))
            remaining += shares * corr

        return actions

    # ── Frequency optimisation ──────────────────────────────────────────

    def should_rebalance(
        self, current_delta: float, gamma: float,
        expected_daily_move: float = 0.01, underlying_price: float = 430,
    ) -> Tuple[bool, str]:
        """Determine if rebalance is needed using Kelly-based trigger."""
        dollar_move = underlying_price * expected_daily_move
        trigger = kelly_rebalance_trigger(
            current_delta, self.config.delta_threshold,
            gamma, dollar_move, self.config.kelly_rebalance_mult,
        )
        if trigger:
            return True, f"Kelly trigger: delta {current_delta:+.1f}, gamma {gamma:.2f}, expected drift exceeds threshold"
        if abs(current_delta) > self.config.delta_threshold:
            return True, f"Delta {current_delta:+.1f} exceeds threshold {self.config.delta_threshold}"
        return False, "No rebalance needed"

    # ── Backtest ────────────────────────────────────────────────────────

    def backtest(
        self,
        portfolio_returns: np.ndarray,
        regimes: np.ndarray,
        vix_series: Optional[np.ndarray] = None,
        underlying_prices: Optional[np.ndarray] = None,
        capital: float = 100_000,
    ) -> BacktestResult:
        """Backtest hedging strategy through historical data."""
        n = len(portfolio_returns)
        if n < 20:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [], {})

        vix = vix_series if vix_series is not None else np.full(n, 18.0)
        prices = underlying_prices if underlying_prices is not None else np.full(n, 430.0)

        hedged_returns = portfolio_returns.copy()
        total_hedge_cost = 0.0
        n_rebalances = 0
        effectiveness: List[HedgeEffectiveness] = []

        # Simulate delta as cumulative sum of returns × scaling
        sim_delta = np.cumsum(portfolio_returns * 100)

        for i in range(20, n):
            regime = str(regimes[i])
            hedge_ratio = self.config.regime_hedge_ratio.get(regime, 0.6)
            delta = sim_delta[i]
            adj_delta = delta * hedge_ratio

            # Check rebalance trigger
            if abs(adj_delta) > self.config.delta_threshold:
                # Hedge cost: proportional to delta hedged
                cost_pct = abs(adj_delta) * self.config.share_cost_bps / 10_000 / 100
                total_hedge_cost += cost_pct * capital
                hedged_returns[i] -= cost_pct
                n_rebalances += 1

                # Hedge reduces next-period impact
                hedge_effectiveness = min(hedge_ratio * 0.5, 0.6)
                if portfolio_returns[i] < 0:
                    hedged_returns[i] = portfolio_returns[i] * (1 - hedge_effectiveness)

            # Tail hedge cost: continuous small drag
            tail_drag = self.config.max_put_cost_pct / 252
            hedged_returns[i] -= tail_drag
            total_hedge_cost += tail_drag * capital

            # Crisis protection: large negative returns reduced more
            if portfolio_returns[i] < -0.02 and regime in ("bear", "high_vol", "crash"):
                protection = hedge_ratio * 0.4
                hedged_returns[i] = portfolio_returns[i] * (1 - protection)

        # Metrics
        unhedged_eq = capital * np.cumprod(1 + portfolio_returns)
        hedged_eq = capital * np.cumprod(1 + hedged_returns)

        def _dd(eq):
            full = np.concatenate([[capital], eq])
            pk = np.maximum.accumulate(full)
            return float(np.min((full - pk) / np.where(pk > 0, pk, 1)))

        def _sharpe(rets):
            return float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0

        u_dd = _dd(unhedged_eq)
        h_dd = _dd(hedged_eq)
        dd_reduction = (abs(u_dd) - abs(h_dd)) / abs(u_dd) * 100 if abs(u_dd) > 0 else 0

        u_sh = _sharpe(portfolio_returns)
        h_sh = _sharpe(hedged_returns)
        sh_improvement = (h_sh - u_sh) / abs(u_sh) * 100 if abs(u_sh) > 0 else 0

        years = n / 252
        cost_annual_pct = total_hedge_cost / capital / max(years, 0.01)

        # Per-regime breakdown
        by_regime: Dict[str, Dict[str, float]] = {}
        for regime in set(regimes):
            mask = regimes == regime
            if mask.sum() < 5:
                continue
            u_r = portfolio_returns[mask]
            h_r = hedged_returns[mask]
            by_regime[str(regime)] = {
                "n": int(mask.sum()),
                "unhedged_return": float(u_r.sum()),
                "hedged_return": float(h_r.sum()),
                "unhedged_sharpe": _sharpe(u_r),
                "hedged_sharpe": _sharpe(h_r),
                "hedge_ratio": self.config.regime_hedge_ratio.get(str(regime), 0.6),
            }

        return BacktestResult(
            n_periods=n, n_rebalances=n_rebalances,
            total_hedge_cost=total_hedge_cost,
            hedge_cost_annual_pct=cost_annual_pct,
            unhedged_dd=u_dd, hedged_dd=h_dd,
            dd_reduction_pct=dd_reduction,
            unhedged_sharpe=u_sh, hedged_sharpe=h_sh,
            sharpe_improvement_pct=sh_improvement,
            unhedged_pnl=float(unhedged_eq[-1] - capital),
            hedged_pnl=float(hedged_eq[-1] - capital),
            effectiveness=effectiveness,
            by_regime=by_regime,
        )
