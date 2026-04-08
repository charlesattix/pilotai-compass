"""
Volatility risk premium harvester — multi-tenor VRP with regime sizing.

Computes VRP at 1W/2W/1M/2M tenors, selects optimal tenor from term
structure, applies regime-conditional sizing, and adds gamma scalping
overlay for tail protection.  Backtests 2020-2025.

Usage::

    from compass.vrp_harvester import VRPHarvester, VRPConfig
    harvester = VRPHarvester(market_df, VRPConfig())
    results = harvester.analyze()
    bt = harvester.backtest()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Configuration ───────────────────────────────────────────────────────


TENORS = {"1W": 5, "2W": 10, "1M": 21, "2M": 42}

REGIME_SIZING = {
    "bull": 1.0, "neutral": 0.8, "bear": 0.3,
    "high_vol": 0.15, "crash": 0.0,
}


@dataclass
class VRPConfig:
    tenors: Dict[str, int] = field(default_factory=lambda: dict(TENORS))
    regime_sizing: Dict[str, float] = field(default_factory=lambda: dict(REGIME_SIZING))
    gamma_scalp_freq: int = 1           # rebalance delta every N days
    gamma_scalp_cost_bps: float = 2.0   # round-trip cost per rebalance
    min_vrp_threshold: float = 0.01     # minimum VRP to harvest (1pp)
    max_position_pct: float = 0.10      # max 10% of capital per trade
    starting_capital: float = 100_000.0


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class TenorVRP:
    """VRP measurement at a single tenor."""
    tenor: str
    days: int
    implied_vol: float          # annualised
    realised_vol: float         # annualised
    vrp: float                  # IV − RV (positive = premium exists)
    vrp_ratio: float            # IV / RV
    z_score: float              # vs rolling history
    harvesting_signal: str      # "sell_vol", "buy_vol", "neutral"


@dataclass
class TermStructure:
    """VRP term structure snapshot."""
    tenors: List[TenorVRP]
    optimal_tenor: str
    steepest_vrp: float
    curve_shape: str            # "contango", "flat", "backwardation"
    overall_signal: str


@dataclass
class GammaScalpResult:
    """Gamma scalping P&L for one period."""
    n_rebalances: int
    gamma_pnl: float            # P&L from gamma (0.5 × γ × move²)
    rebalance_cost: float
    net_scalp_pnl: float
    pct_of_premium: float       # scalp P&L as % of premium collected


@dataclass
class HarvestTrade:
    """One VRP harvest trade."""
    entry_date: str
    exit_date: str
    tenor: str
    regime: str
    iv_at_entry: float
    rv_at_exit: float
    vrp_captured: float         # IV − RV actually realised
    position_size: float        # fraction of capital
    gross_pnl: float
    scalp_pnl: float
    cost: float
    net_pnl: float
    win: bool


@dataclass
class BacktestResult:
    """VRP harvester backtest result."""
    n_trades: int
    n_wins: int
    win_rate: float
    total_pnl: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    annual_return: float
    avg_vrp: float
    vrp_positive_pct: float     # fraction of periods with VRP > 0
    total_scalp_pnl: float
    total_cost: float
    by_tenor: Dict[str, Dict[str, float]]
    by_regime: Dict[str, Dict[str, float]]
    correlation_with_spy: float
    trades: List[HarvestTrade]


# ── Core computations ───────────────────────────────────────────────────


def realised_vol(returns: np.ndarray, window: int, annualise: float = 252) -> np.ndarray:
    """Rolling realised volatility (annualised)."""
    n = len(returns)
    rv = np.full(n, np.nan)
    for i in range(window, n):
        rv[i] = float(np.std(returns[i - window:i]) * np.sqrt(annualise))
    return rv


def implied_vol_proxy(
    vix: np.ndarray, tenor_days: int, base_tenor: int = 30,
) -> np.ndarray:
    """Estimate tenor-specific IV from VIX using sqrt-time scaling.

    IV(T) ≈ VIX × √(base_tenor / T) for short tenors,
    IV(T) ≈ VIX × √(T / base_tenor) dampened for long tenors.
    """
    ratio = tenor_days / base_tenor
    if ratio <= 1:
        # Short tenor: IV is higher (front-loaded fear)
        scale = math.sqrt(1 / max(ratio, 0.1)) * 0.85  # dampen
    else:
        # Long tenor: IV is lower (mean-reversion)
        scale = math.sqrt(ratio) * 0.75
    return vix / 100.0 * scale


def compute_vrp(iv: np.ndarray, rv: np.ndarray) -> np.ndarray:
    """VRP = IV − RV. Positive means premium exists."""
    return iv - rv


def gamma_scalp_pnl(
    price_changes: np.ndarray,
    gamma: float,
    cost_per_rebalance: float,
    freq: int = 1,
) -> Tuple[float, float, int]:
    """Compute gamma scalping P&L.

    Gamma profit = 0.5 × γ × Σ(ΔS²) over the period.
    Returns (gamma_pnl, total_cost, n_rebalances).
    """
    n = len(price_changes)
    n_rebal = max(1, n // freq)
    gamma_pnl = 0.5 * gamma * float(np.sum(price_changes ** 2))
    total_cost = n_rebal * cost_per_rebalance
    return gamma_pnl, total_cost, n_rebal


def classify_regime_from_vix(vix: float) -> str:
    """Simple regime from VIX level."""
    if vix >= 35:
        return "crash"
    if vix >= 28:
        return "high_vol"
    if vix >= 20:
        return "neutral"
    if vix >= 14:
        return "bull"
    return "bull"


# ── Harvester ───────────────────────────────────────────────────────────


class VRPHarvester:
    """Multi-tenor volatility risk premium harvester."""

    def __init__(
        self,
        market_data: pd.DataFrame,
        config: Optional[VRPConfig] = None,
    ) -> None:
        self.config = config or VRPConfig()
        self.data = market_data.copy()
        self.n = len(market_data)

        # Extract arrays
        self.close = market_data["close"].values.astype(float)
        self.returns = np.diff(self.close) / self.close[:-1]
        self.returns = np.concatenate([[0], self.returns])
        self.vix = market_data["vix"].values.astype(float) if "vix" in market_data.columns else np.full(self.n, 18.0)

        # Regimes
        if "regime" in market_data.columns:
            self.regimes = market_data["regime"].values
        else:
            self.regimes = np.array([classify_regime_from_vix(v) for v in self.vix])

        # Results
        self.term_structures: List[TermStructure] = []
        self.backtest_result: Optional[BacktestResult] = None

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "VRPHarvester":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls(df, **kwargs)

    # ── Analysis ────────────────────────────────────────────────────────

    def analyze(self) -> List[TermStructure]:
        """Compute VRP term structure at each point in time."""
        self.term_structures = []
        cfg = self.config

        for i in range(max(cfg.tenors.values()), self.n):
            tenors_vrp: List[TenorVRP] = []
            for name, days in cfg.tenors.items():
                iv = float(implied_vol_proxy(np.array([self.vix[i]]), days)[0])
                rv_arr = realised_vol(self.returns[:i + 1], days)
                rv_val = float(rv_arr[i]) if i < len(rv_arr) and not np.isnan(rv_arr[i]) else iv * 0.8

                vrp = iv - rv_val
                ratio = iv / rv_val if rv_val > 0 else 1.0

                # Z-score vs rolling 60-day history
                hist_vrps = []
                for j in range(max(0, i - 60), i):
                    h_iv = float(implied_vol_proxy(np.array([self.vix[j]]), days)[0])
                    h_rv = float(rv_arr[j]) if j < len(rv_arr) and not np.isnan(rv_arr[j]) else h_iv * 0.8
                    hist_vrps.append(h_iv - h_rv)
                if len(hist_vrps) > 5:
                    z = (vrp - np.mean(hist_vrps)) / max(np.std(hist_vrps), 1e-6)
                else:
                    z = 0.0

                if vrp > cfg.min_vrp_threshold:
                    signal = "sell_vol"
                elif vrp < -cfg.min_vrp_threshold:
                    signal = "buy_vol"
                else:
                    signal = "neutral"

                tenors_vrp.append(TenorVRP(
                    name, days, iv, rv_val, vrp, ratio, float(z), signal,
                ))

            # Optimal tenor: highest VRP
            optimal = max(tenors_vrp, key=lambda t: t.vrp)
            steepest = optimal.vrp

            # Curve shape
            front_vrp = tenors_vrp[0].vrp if tenors_vrp else 0
            back_vrp = tenors_vrp[-1].vrp if tenors_vrp else 0
            if back_vrp > front_vrp + 0.005:
                shape = "contango"
            elif front_vrp > back_vrp + 0.005:
                shape = "backwardation"
            else:
                shape = "flat"

            overall = "sell_vol" if steepest > cfg.min_vrp_threshold else "neutral"

            self.term_structures.append(TermStructure(
                tenors_vrp, optimal.tenor, steepest, shape, overall,
            ))

        return self.term_structures

    # ── Backtest ────────────────────────────────────────────────────────

    def backtest(self) -> BacktestResult:
        """Backtest VRP harvesting strategy 2020-2025."""
        if not self.term_structures:
            self.analyze()

        cfg = self.config
        cap = cfg.starting_capital
        trades: List[HarvestTrade] = []
        equity_curve = [cap]

        offset = self.n - len(self.term_structures)

        i = 0
        while i < len(self.term_structures):
            ts = self.term_structures[i]
            data_idx = i + offset

            if ts.overall_signal != "sell_vol":
                equity_curve.append(equity_curve[-1])
                i += 1
                continue

            # Regime sizing
            regime = str(self.regimes[data_idx]) if data_idx < len(self.regimes) else "neutral"
            size_mult = cfg.regime_sizing.get(regime, 0.5)
            if size_mult <= 0:
                equity_curve.append(equity_curve[-1])
                i += 1
                continue

            # Select tenor
            tenor_name = ts.optimal_tenor
            tenor_days = cfg.tenors[tenor_name]

            # Entry
            entry_idx = data_idx
            exit_idx = min(entry_idx + tenor_days, self.n - 1)
            if exit_idx <= entry_idx:
                i += 1
                continue

            iv_entry = ts.steepest_vrp + ts.tenors[0].implied_vol  # use optimal tenor's IV
            for t in ts.tenors:
                if t.tenor == tenor_name:
                    iv_entry = t.implied_vol
                    break

            # Realised vol over holding period
            period_returns = self.returns[entry_idx:exit_idx]
            if len(period_returns) < 3:
                i += 1
                continue
            rv_exit = float(np.std(period_returns) * np.sqrt(252))
            vrp_captured = iv_entry - rv_exit

            # Position size
            position_pct = cfg.max_position_pct * size_mult
            notional = equity_curve[-1] * position_pct

            # Gross P&L: VRP × notional × (tenor / 252)
            time_frac = tenor_days / 252
            gross_pnl = vrp_captured * notional * time_frac

            # Gamma scalp
            price_changes = np.diff(self.close[entry_idx:exit_idx + 1])
            gamma_est = 0.01 * notional / (self.close[entry_idx] ** 2)  # rough gamma
            scalp_cost_per = equity_curve[-1] * cfg.gamma_scalp_cost_bps / 10_000
            g_pnl, g_cost, n_rebal = gamma_scalp_pnl(
                price_changes, gamma_est, scalp_cost_per, cfg.gamma_scalp_freq,
            )
            net_scalp = g_pnl - g_cost

            # Transaction cost
            cost = notional * 0.001  # 10bps entry + exit

            net_pnl = gross_pnl + net_scalp - cost
            win = net_pnl > 0

            entry_date = str(self.data.index[entry_idx]) if hasattr(self.data.index, '__getitem__') else str(entry_idx)
            exit_date = str(self.data.index[exit_idx]) if hasattr(self.data.index, '__getitem__') else str(exit_idx)

            trades.append(HarvestTrade(
                entry_date, exit_date, tenor_name, regime,
                iv_entry, rv_exit, vrp_captured, position_pct,
                gross_pnl, net_scalp, cost, net_pnl, win,
            ))

            new_equity = equity_curve[-1] + net_pnl
            # Fill equity for holding period
            for _ in range(tenor_days):
                equity_curve.append(new_equity)

            i += tenor_days  # skip to after exit

        # Pad equity curve
        while len(equity_curve) <= len(self.term_structures):
            equity_curve.append(equity_curve[-1])

        # Metrics
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, {}, {}, 0, [])

        pnls = np.array([t.net_pnl for t in trades])
        n_trades = len(trades)
        n_wins = sum(1 for t in trades if t.win)
        wr = n_wins / n_trades
        total_pnl = float(pnls.sum())

        eq = np.array(equity_curve)
        eq_returns = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
        eq_returns = eq_returns[eq_returns != 0]  # remove flat periods

        sh = float(np.mean(eq_returns) / np.std(eq_returns) * np.sqrt(252)) if len(eq_returns) > 1 and np.std(eq_returns) > 0 else 0
        down = eq_returns[eq_returns < 0]
        down_std = float(np.std(down)) if len(down) > 0 else 0.001
        sortino = float(np.mean(eq_returns) / down_std * np.sqrt(252)) if down_std > 0 else 0

        pk = np.maximum.accumulate(eq)
        dd = float(np.min((eq - pk) / np.where(pk > 0, pk, 1)))

        years = self.n / 252
        ann_ret = total_pnl / cap / max(years, 0.1)
        calmar = ann_ret / abs(dd) if dd != 0 else 0

        avg_vrp = float(np.mean([t.vrp_captured for t in trades]))
        vrp_pos_pct = float(np.mean([t.vrp_captured > 0 for t in trades]))
        total_scalp = float(sum(t.scalp_pnl for t in trades))
        total_cost = float(sum(t.cost for t in trades))

        # By tenor
        by_tenor: Dict[str, Dict[str, float]] = {}
        for t in trades:
            if t.tenor not in by_tenor:
                by_tenor[t.tenor] = {"n": 0, "pnl": 0, "wins": 0}
            by_tenor[t.tenor]["n"] += 1
            by_tenor[t.tenor]["pnl"] += t.net_pnl
            by_tenor[t.tenor]["wins"] += int(t.win)
        for k in by_tenor:
            by_tenor[k]["win_rate"] = by_tenor[k]["wins"] / by_tenor[k]["n"] if by_tenor[k]["n"] > 0 else 0

        # By regime
        by_regime: Dict[str, Dict[str, float]] = {}
        for t in trades:
            if t.regime not in by_regime:
                by_regime[t.regime] = {"n": 0, "pnl": 0, "wins": 0}
            by_regime[t.regime]["n"] += 1
            by_regime[t.regime]["pnl"] += t.net_pnl
            by_regime[t.regime]["wins"] += int(t.win)
        for k in by_regime:
            by_regime[k]["win_rate"] = by_regime[k]["wins"] / by_regime[k]["n"] if by_regime[k]["n"] > 0 else 0

        # Correlation with SPY
        spy_rets = self.returns[offset:]
        if len(spy_rets) > len(eq_returns):
            spy_rets = spy_rets[:len(eq_returns)]
        elif len(eq_returns) > len(spy_rets):
            eq_returns = eq_returns[:len(spy_rets)]
        if len(spy_rets) > 10 and np.std(eq_returns) > 0 and np.std(spy_rets) > 0:
            corr = float(np.corrcoef(eq_returns, spy_rets)[0, 1])
        else:
            corr = 0.0

        self.backtest_result = BacktestResult(
            n_trades, n_wins, wr, total_pnl, sh, sortino, dd, calmar,
            ann_ret, avg_vrp, vrp_pos_pct, total_scalp, total_cost,
            by_tenor, by_regime, corr, trades,
        )
        return self.backtest_result
