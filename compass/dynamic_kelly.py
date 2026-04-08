"""Dynamic Kelly Criterion — adaptive position sizing using rolling win rate
and payoff ratio with regime-conditional fractional Kelly.

Provides:
  1. Rolling Kelly fraction at 20/60/120 day windows
  2. Fractional Kelly (0.25-0.50×) with regime modulation
  3. Crisis damping (0.15× Kelly in crash regime)
  4. Comparison vs fixed fraction, risk parity, equal weight
  5. Portfolio growth simulation with transaction costs
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# Regime → Kelly fraction multiplier
REGIME_KELLY_MULT = {
    "bull": 0.50,
    "low_vol": 0.45,
    "bear": 0.30,
    "high_vol": 0.20,
    "crash": 0.15,
}


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class KellyEstimate:
    """Kelly fraction estimate at one point in time."""
    date: str
    win_rate: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float    # avg_win / avg_loss
    full_kelly: float      # raw Kelly fraction
    fractional_kelly: float # after fraction × regime
    regime: str
    window: int


@dataclass
class SizingComparison:
    """Performance of one sizing method."""
    method: str
    total_return_pct: float
    cagr_pct: float
    sharpe: float
    max_dd_pct: float
    avg_position_size: float
    total_cost: float


@dataclass
class DynamicKellyResult:
    """Complete experiment output."""
    kelly_history: List[KellyEstimate] = field(default_factory=list)
    comparisons: List[SizingComparison] = field(default_factory=list)
    best_method: str = ""
    best_sharpe: float = 0.0
    generated_at: str = ""


# ── Core Kelly computation ──────────────────────────────────────────────────
def kelly_fraction(win_rate: float, payoff_ratio: float) -> float:
    """Full Kelly: f* = p - q/b where p=win_rate, q=1-p, b=payoff_ratio."""
    if payoff_ratio <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    q = 1 - win_rate
    f = win_rate - q / payoff_ratio
    return max(0.0, f)


def rolling_kelly(
    returns: np.ndarray, window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling Kelly fraction, win rate, and payoff ratio.

    Returns (kelly_arr, win_rate_arr, payoff_ratio_arr).
    """
    n = len(returns)
    kelly = np.zeros(n)
    wr = np.zeros(n)
    pr = np.zeros(n)

    for i in range(window, n):
        w = returns[i - window:i]
        wins = w[w > 0]
        losses = w[w < 0]

        win_rate = len(wins) / window if window > 0 else 0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0
        avg_loss = float(-losses.mean()) if len(losses) > 0 else 1e-8
        payoff = avg_win / avg_loss if avg_loss > 1e-8 else 0

        wr[i] = win_rate
        pr[i] = payoff
        kelly[i] = kelly_fraction(win_rate, payoff)

    return kelly, wr, pr


def classify_regime(vix: float, momentum_20d: float) -> str:
    """Simple regime classifier from VIX and momentum."""
    if vix > 35:
        return "crash"
    if vix > 25:
        return "high_vol"
    if momentum_20d < -0.05:
        return "bear"
    if vix < 14:
        return "low_vol"
    return "bull"


def apply_regime_fraction(full_kelly: float, regime: str) -> float:
    """Apply regime-dependent fractional Kelly."""
    mult = REGIME_KELLY_MULT.get(regime, 0.35)
    return max(0.0, min(1.0, full_kelly * mult))


# ── Multi-window Kelly tracker ─────────────────────────────────────────────
class DynamicKellyTracker:
    """Tracks rolling Kelly at multiple windows and blends them."""

    def __init__(
        self,
        windows: Tuple[int, ...] = (20, 60, 120),
        blend_weights: Optional[Tuple[float, ...]] = None,
    ) -> None:
        self.windows = windows
        self.blend_weights = blend_weights or (0.40, 0.35, 0.25)

    def compute(
        self,
        returns: np.ndarray,
        vix_series: np.ndarray,
        dates: Optional[List[str]] = None,
    ) -> List[KellyEstimate]:
        """Compute blended Kelly estimates over time."""
        n = len(returns)
        max_w = max(self.windows)
        if n < max_w + 10:
            return []

        # Compute per-window
        kellys = {}
        wrs = {}
        prs = {}
        for w in self.windows:
            k, wr, pr = rolling_kelly(returns, w)
            kellys[w] = k
            wrs[w] = wr
            prs[w] = pr

        # Momentum for regime
        cum_ret = np.cumsum(returns)
        mom_20 = np.zeros(n)
        for i in range(20, n):
            mom_20[i] = cum_ret[i] - cum_ret[i - 20]

        estimates: List[KellyEstimate] = []
        for i in range(max_w, n):
            # Blend across windows
            blended_kelly = sum(
                kellys[w][i] * bw
                for w, bw in zip(self.windows, self.blend_weights)
            )
            blended_wr = sum(wrs[w][i] * bw for w, bw in zip(self.windows, self.blend_weights))
            blended_pr = sum(prs[w][i] * bw for w, bw in zip(self.windows, self.blend_weights))

            vix = float(vix_series[i]) if i < len(vix_series) else 20.0
            regime = classify_regime(vix, mom_20[i])
            frac_kelly = apply_regime_fraction(blended_kelly, regime)

            d = dates[i] if dates and i < len(dates) else str(i)

            # Avg win/loss from shortest window for display
            sw = self.windows[0]
            w_ret = returns[max(0, i - sw):i]
            wins_r = w_ret[w_ret > 0]
            losses_r = w_ret[w_ret < 0]
            avg_w = float(wins_r.mean()) if len(wins_r) > 0 else 0
            avg_l = float(-losses_r.mean()) if len(losses_r) > 0 else 0

            estimates.append(KellyEstimate(
                date=d,
                win_rate=round(blended_wr, 4),
                avg_win=round(avg_w, 6),
                avg_loss=round(avg_l, 6),
                payoff_ratio=round(blended_pr, 4),
                full_kelly=round(blended_kelly, 4),
                fractional_kelly=round(frac_kelly, 4),
                regime=regime,
                window=sw,
            ))

        return estimates


# ── Backtest comparison ─────────────────────────────────────────────────────
class DynamicKellyBacktest:
    """Compare dynamic Kelly vs fixed fraction, risk parity, equal weight."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        cost_bps: float = 5.0,
        min_size: float = 0.02,
        max_size: float = 0.30,
        seed: int = 42,
    ) -> None:
        self.starting_capital = starting_capital
        self.cost_bps = cost_bps
        self.min_size = min_size
        self.max_size = max_size
        self.rng = np.random.RandomState(seed)

    def run(
        self,
        returns: np.ndarray,
        vix_series: np.ndarray,
        dates: Optional[List[str]] = None,
    ) -> DynamicKellyResult:
        n = len(returns)
        if n < 150:
            return DynamicKellyResult(generated_at=_now())

        tracker = DynamicKellyTracker()
        kelly_hist = tracker.compute(returns, vix_series, dates)
        if not kelly_hist:
            return DynamicKellyResult(generated_at=_now())

        warmup = max(tracker.windows) + 10
        active_returns = returns[warmup:]
        active_n = len(active_returns)

        # Build sizing arrays per method
        methods: Dict[str, np.ndarray] = {}

        # Dynamic Kelly
        dk_sizes = np.array([
            np.clip(ke.fractional_kelly, self.min_size, self.max_size)
            for ke in kelly_hist[:active_n]
        ])
        methods["dynamic_kelly"] = dk_sizes

        # Fixed half-Kelly (median of dynamic)
        median_kelly = float(np.median(dk_sizes)) if len(dk_sizes) > 0 else 0.05
        methods["fixed_kelly"] = np.full(active_n, median_kelly)

        # Fixed conservative
        methods["fixed_5pct"] = np.full(active_n, 0.05)

        # Risk parity (inverse vol, rolling 60d)
        rp_sizes = np.zeros(active_n)
        for i in range(60, active_n):
            vol = np.std(active_returns[i - 60:i]) * np.sqrt(TRADING_DAYS)
            target_vol = 0.15  # target 15% portfolio vol
            rp_sizes[i] = np.clip(target_vol / max(vol, 0.01), self.min_size, self.max_size)
        rp_sizes[:60] = 0.05
        methods["risk_parity"] = rp_sizes

        # Equal weight
        methods["equal_weight"] = np.full(active_n, 1.0 / 5)  # assume 5 strategies

        # Simulate each
        comparisons: List[SizingComparison] = []
        for name, sizes in methods.items():
            comp = self._simulate(name, active_returns, sizes)
            comparisons.append(comp)

        best = max(comparisons, key=lambda c: c.sharpe)

        return DynamicKellyResult(
            kelly_history=kelly_hist,
            comparisons=comparisons,
            best_method=best.method,
            best_sharpe=best.sharpe,
            generated_at=_now(),
        )

    def _simulate(
        self, name: str, returns: np.ndarray, sizes: np.ndarray,
    ) -> SizingComparison:
        n = min(len(returns), len(sizes))
        capital = self.starting_capital
        peak = capital
        max_dd = 0.0
        daily_pnl: List[float] = []
        total_cost = 0.0
        prev_size = 0.0

        for i in range(n):
            size = float(sizes[i])
            # Transaction cost on rebalance
            turnover = abs(size - prev_size)
            cost = turnover * capital * self.cost_bps / 10_000
            total_cost += cost
            capital -= cost
            prev_size = size

            pnl = float(returns[i]) * size * capital
            capital += pnl
            daily_pnl.append(pnl)

            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        total_ret = (capital - self.starting_capital) / self.starting_capital * 100
        years = n / TRADING_DAYS
        cagr = ((capital / self.starting_capital) ** (1 / years) - 1) * 100 if years > 0 and capital > 0 else 0

        dr = np.array(daily_pnl) if daily_pnl else np.array([0.0])
        sharpe = float(dr.mean() / dr.std() * np.sqrt(TRADING_DAYS)) if dr.std() > 0 else 0

        return SizingComparison(
            method=name,
            total_return_pct=round(total_ret, 2),
            cagr_pct=round(cagr, 2),
            sharpe=round(sharpe, 2),
            max_dd_pct=round(max_dd * 100, 2),
            avg_position_size=round(float(sizes.mean()), 4),
            total_cost=round(total_cost, 2),
        )


# ── Synthetic data ──────────────────────────────────────────────────────────
def generate_strategy_data(
    n: int = 1200, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic strategy returns + VIX."""
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    dates = [str(d.date()) for d in idx]

    returns = rng.randn(n) * 0.008 + 0.0004  # slight positive drift

    # Inject regime changes (guard against short arrays)
    if n > 240:
        returns[200:240] = rng.randn(40) * 0.02 - 0.005
    if n > 520:
        returns[500:520] = rng.randn(20) * 0.03 - 0.01
    if n > 750:
        returns[700:750] = rng.randn(50) * 0.005 + 0.001

    vix = np.zeros(n)
    vix[0] = 15.0
    for i in range(1, n):
        vix[i] = max(9, min(70, vix[i - 1] + 0.03 * (18 - vix[i - 1]) - returns[i] * 150 + rng.randn() * 1.2))

    return returns, vix, dates


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
