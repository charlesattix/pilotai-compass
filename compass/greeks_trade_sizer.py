"""
Adaptive Greeks-based trade sizing.

Size each trade to hit a portfolio theta target while respecting gamma
and vega caps.  Dynamic delta budget per regime.  Compares against
fixed-size and Kelly sizing baselines.

Usage::

    from compass.greeks_trade_sizer import GreeksTradeSizer, SizerConfig
    sizer = GreeksTradeSizer(SizerConfig())
    n = sizer.size_trade(trade_greeks, portfolio_state)
    bt = sizer.backtest(trades_df)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.greeks_calculator import compute_option_greeks


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class SizerConfig:
    # Theta target
    target_theta_daily: float = 200.0     # $ theta income/day target
    theta_tolerance: float = 0.20          # ±20% of target is acceptable
    # Gamma/vega caps (portfolio-level)
    max_gamma: float = 50.0
    max_vega: float = 300.0
    # Delta budget per regime
    delta_budget: Dict[str, float] = field(default_factory=lambda: {
        "bull": 30.0, "neutral": 15.0, "bear": 10.0,
        "high_vol": 5.0, "crash": 0.0,
    })
    # Position limits
    min_contracts: int = 1
    max_contracts: int = 20
    # Capital
    capital: float = 100_000.0
    max_capital_per_trade: float = 0.10   # 10% max
    margin_per_contract: float = 500.0
    # Kelly comparison
    kelly_edge: float = 0.10              # assumed edge for Kelly
    kelly_fraction: float = 0.25          # quarter-Kelly


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class TradeGreeks:
    """Per-contract Greeks for a proposed trade."""
    delta: float        # per contract × 100
    gamma: float
    theta: float        # daily, per contract × 100
    vega: float
    premium: float      # credit received per contract × 100
    dte: int
    iv: float
    underlying_price: float
    strike: float
    spread_strike: Optional[float] = None
    option_type: str = "put"
    direction: str = "short"


@dataclass
class PortfolioState:
    """Current portfolio Greeks state."""
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0
    n_positions: int = 0
    margin_used: float = 0.0
    capital: float = 100_000.0
    regime: str = "neutral"


@dataclass
class SizingResult:
    """Result of sizing a single trade."""
    contracts: int
    method: str                 # "greeks", "fixed", "kelly"
    theta_contribution: float   # daily theta from this trade
    gamma_contribution: float
    vega_contribution: float
    delta_contribution: float
    margin_required: float
    capped_by: str              # "" or "gamma", "vega", "delta", "margin", "max_contracts"
    pct_of_theta_target: float  # this trade's theta / target


@dataclass
class BacktestTrade:
    """One trade in the backtest."""
    idx: int
    contracts_greeks: int
    contracts_fixed: int
    contracts_kelly: int
    pnl_greeks: float
    pnl_fixed: float
    pnl_kelly: float
    theta_at_entry: float
    gamma_at_entry: float
    regime: str


@dataclass
class BacktestResult:
    """Comparison of three sizing methods."""
    n_trades: int
    # Greeks sizing
    greeks_pnl: float
    greeks_sharpe: float
    greeks_dd: float
    greeks_win_rate: float
    greeks_avg_theta: float
    greeks_theta_std: float
    # Fixed sizing
    fixed_pnl: float
    fixed_sharpe: float
    fixed_dd: float
    fixed_win_rate: float
    # Kelly sizing
    kelly_pnl: float
    kelly_sharpe: float
    kelly_dd: float
    kelly_win_rate: float
    # Comparisons
    sharpe_improvement_vs_fixed: float
    sharpe_improvement_vs_kelly: float
    dd_improvement_vs_fixed: float
    gamma_breaches_greeks: int
    gamma_breaches_fixed: int
    vega_breaches_greeks: int
    vega_breaches_fixed: int
    trades: List[BacktestTrade]


# ── Per-trade Greeks estimation ─────────────────────────────────────────


def estimate_trade_greeks(
    underlying: float, strike: float, dte: int, iv: float,
    spread_strike: Optional[float] = None,
    option_type: str = "put", direction: str = "short",
    rate: float = 0.045,
) -> TradeGreeks:
    """Estimate Greeks for a proposed credit spread trade (1 contract)."""
    T = max(dte / 365.0, 1e-10)
    sign = -1.0 if direction == "short" else 1.0
    mult = 100.0  # 1 contract = 100 shares

    main = compute_option_greeks(underlying, strike, T, iv, rate, option_type)
    d = sign * main.delta * mult
    g = sign * main.gamma * mult
    t = sign * main.theta * mult
    v = sign * main.vega * mult
    p = sign * main.price * mult

    if spread_strike is not None:
        hedge = compute_option_greeks(underlying, spread_strike, T, iv, rate, option_type)
        hs = -sign
        d += hs * hedge.delta * mult
        g += hs * hedge.gamma * mult
        t += hs * hedge.theta * mult
        v += hs * hedge.vega * mult
        p += hs * hedge.price * mult

    return TradeGreeks(d, g, t, v, abs(p), dte, iv, underlying, strike,
                       spread_strike, option_type, direction)


# ── Sizer ───────────────────────────────────────────────────────────────


class GreeksTradeSizer:
    """Adaptive Greeks-based trade sizing engine."""

    def __init__(self, config: Optional[SizerConfig] = None) -> None:
        self.config = config or SizerConfig()

    def size_trade(
        self,
        trade: TradeGreeks,
        portfolio: PortfolioState,
    ) -> SizingResult:
        """Size a trade to target theta while respecting gamma/vega/delta caps."""
        cfg = self.config

        # Theta-based sizing: how many contracts to hit target?
        remaining_theta = cfg.target_theta_daily - portfolio.total_theta
        if trade.theta <= 0 or remaining_theta <= 0:
            return SizingResult(0, "greeks", 0, 0, 0, 0, 0, "no_theta_needed", 0)

        theta_contracts = remaining_theta / trade.theta
        n = max(cfg.min_contracts, int(round(theta_contracts)))
        capped_by = ""

        # Gamma cap
        gamma_room = cfg.max_gamma - abs(portfolio.total_gamma)
        if gamma_room > 0 and abs(trade.gamma) > 0:
            gamma_max = int(gamma_room / abs(trade.gamma))
            if gamma_max < n:
                n = max(cfg.min_contracts, gamma_max)
                capped_by = "gamma"

        # Vega cap
        vega_room = cfg.max_vega - abs(portfolio.total_vega)
        if vega_room > 0 and abs(trade.vega) > 0:
            vega_max = int(vega_room / abs(trade.vega))
            if vega_max < n:
                n = max(cfg.min_contracts, vega_max)
                capped_by = "vega"

        # Delta budget
        delta_budget = cfg.delta_budget.get(portfolio.regime, 15.0)
        delta_room = delta_budget - abs(portfolio.total_delta)
        if delta_room > 0 and abs(trade.delta) > 0:
            delta_max = int(delta_room / abs(trade.delta))
            if delta_max < n:
                n = max(cfg.min_contracts, delta_max)
                capped_by = "delta"

        # Margin cap
        margin_room = portfolio.capital * cfg.max_capital_per_trade - portfolio.margin_used * 0.1
        if cfg.margin_per_contract > 0:
            margin_max = int(margin_room / cfg.margin_per_contract)
            if margin_max < n:
                n = max(cfg.min_contracts, margin_max)
                capped_by = "margin"

        # Hard cap
        if n > cfg.max_contracts:
            n = cfg.max_contracts
            capped_by = "max_contracts"

        n = max(0, min(n, cfg.max_contracts))

        theta_contrib = trade.theta * n
        pct_target = theta_contrib / cfg.target_theta_daily if cfg.target_theta_daily > 0 else 0

        return SizingResult(
            n, "greeks", theta_contrib, trade.gamma * n,
            trade.vega * n, trade.delta * n,
            n * cfg.margin_per_contract, capped_by, pct_target,
        )

    def size_fixed(self, base_contracts: int = 2) -> int:
        """Fixed sizing baseline."""
        return base_contracts

    def size_kelly(
        self, win_rate: float, avg_win: float, avg_loss: float,
    ) -> int:
        """Kelly criterion sizing."""
        cfg = self.config
        if avg_loss == 0:
            return cfg.min_contracts
        b = avg_win / abs(avg_loss)  # odds ratio
        p = win_rate
        kelly_f = (p * b - (1 - p)) / b if b > 0 else 0
        kelly_f = max(0, kelly_f) * cfg.kelly_fraction
        # Convert fraction to contracts
        notional = cfg.capital * kelly_f
        contracts = int(notional / max(cfg.margin_per_contract, 1))
        return max(cfg.min_contracts, min(contracts, cfg.max_contracts))

    # ── Backtest ────────────────────────────────────────────────────────

    def backtest(
        self,
        trades_df: pd.DataFrame,
        fixed_contracts: int = 2,
    ) -> BacktestResult:
        """Backtest Greeks sizing vs fixed vs Kelly on historical trades."""
        n = len(trades_df)
        if n == 0:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

        cfg = self.config
        bt_trades: List[BacktestTrade] = []

        # Running portfolio state
        port = PortfolioState(capital=cfg.capital)
        gamma_breaches_g = 0
        gamma_breaches_f = 0
        vega_breaches_g = 0
        vega_breaches_f = 0

        for i, row in trades_df.iterrows():
            # Extract trade parameters
            underlying = float(row.get("spy_price", row.get("underlying_price", 430)))
            strike = float(row.get("short_strike", underlying * 0.95)) if pd.notna(row.get("short_strike")) else underlying * 0.95
            spread_strike = strike - float(row.get("spread_width", 5))
            dte = int(row.get("dte_at_entry", 30))
            iv = float(row.get("iv_rank", 30)) / 100 + 0.10  # rough IV from rank
            regime = str(row.get("regime", "neutral"))
            base_pnl = float(row.get("pnl", 0))
            base_contracts = int(row.get("contracts", 2))

            # Estimate per-contract Greeks
            tg = estimate_trade_greeks(underlying, strike, dte, iv, spread_strike)

            # Greeks sizing
            port.regime = regime
            sr = self.size_trade(tg, port)
            n_greeks = sr.contracts

            # Fixed sizing
            n_fixed = fixed_contracts

            # Kelly sizing
            win_rate = float(row.get("win", 0.5))
            avg_win = abs(base_pnl) * 1.2 if base_pnl > 0 else 100
            avg_loss = abs(base_pnl) if base_pnl < 0 else 80
            n_kelly = self.size_kelly(0.65, avg_win, avg_loss)

            # Scale PnL by contract ratio
            if base_contracts > 0:
                pnl_per_contract = base_pnl / base_contracts
            else:
                pnl_per_contract = base_pnl

            pnl_g = pnl_per_contract * n_greeks
            pnl_f = pnl_per_contract * n_fixed
            pnl_k = pnl_per_contract * n_kelly

            # Track breaches
            cum_gamma_g = port.total_gamma + tg.gamma * n_greeks
            cum_gamma_f = port.total_gamma + tg.gamma * n_fixed
            cum_vega_g = port.total_vega + tg.vega * n_greeks
            cum_vega_f = port.total_vega + tg.vega * n_fixed

            if abs(cum_gamma_g) > cfg.max_gamma:
                gamma_breaches_g += 1
            if abs(cum_gamma_f) > cfg.max_gamma:
                gamma_breaches_f += 1
            if abs(cum_vega_g) > cfg.max_vega:
                vega_breaches_g += 1
            if abs(cum_vega_f) > cfg.max_vega:
                vega_breaches_f += 1

            # Update portfolio (simple: reset each trade for independence)
            port.total_theta += tg.theta * n_greeks
            # Reset after exit (assume trades don't overlap heavily)
            if np.random.random() > 0.7:
                port.total_delta = 0
                port.total_gamma = 0
                port.total_vega = 0
                port.total_theta = 0

            bt_trades.append(BacktestTrade(
                int(i), n_greeks, n_fixed, n_kelly,
                pnl_g, pnl_f, pnl_k,
                tg.theta * n_greeks, tg.gamma * n_greeks, regime,
            ))

        # Aggregate metrics
        pnls_g = np.array([t.pnl_greeks for t in bt_trades])
        pnls_f = np.array([t.pnl_fixed for t in bt_trades])
        pnls_k = np.array([t.pnl_kelly for t in bt_trades])

        def _metrics(pnls, cap):
            eq = cap + np.cumsum(pnls)
            eq_f = np.concatenate([[cap], eq])
            rets = pnls / cap
            sh = float(np.mean(rets) / np.std(rets) * np.sqrt(252)) if np.std(rets) > 0 else 0
            pk = np.maximum.accumulate(eq_f)
            dd = float(np.min((eq_f - pk) / np.where(pk > 0, pk, 1)))
            wr = float((pnls > 0).mean())
            return float(pnls.sum()), sh, dd, wr

        g_pnl, g_sh, g_dd, g_wr = _metrics(pnls_g, cfg.capital)
        f_pnl, f_sh, f_dd, f_wr = _metrics(pnls_f, cfg.capital)
        k_pnl, k_sh, k_dd, k_wr = _metrics(pnls_k, cfg.capital)

        thetas = [t.theta_at_entry for t in bt_trades]
        avg_theta = float(np.mean(thetas)) if thetas else 0
        theta_std = float(np.std(thetas)) if thetas else 0

        sh_imp_f = (g_sh - f_sh) / abs(f_sh) * 100 if abs(f_sh) > 0 else 0
        sh_imp_k = (g_sh - k_sh) / abs(k_sh) * 100 if abs(k_sh) > 0 else 0
        dd_imp_f = (abs(f_dd) - abs(g_dd)) / abs(f_dd) * 100 if abs(f_dd) > 0 else 0

        return BacktestResult(
            n, g_pnl, g_sh, g_dd, g_wr, avg_theta, theta_std,
            f_pnl, f_sh, f_dd, f_wr,
            k_pnl, k_sh, k_dd, k_wr,
            sh_imp_f, sh_imp_k, dd_imp_f,
            gamma_breaches_g, gamma_breaches_f,
            vega_breaches_g, vega_breaches_f,
            bt_trades,
        )
