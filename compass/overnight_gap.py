"""Overnight gap strategy — exploits the overnight risk premium in SPY options
by selling straddles at close and buying back at open.

Provides:
  1. Overnight premium calculator (theta captured during ~18hr overnight)
  2. Gap risk model (historical gap distribution, percentile-based sizing)
  3. Position sizing based on gap risk and portfolio heat
  4. Backtest: sell ATM straddle 3:55 PM, buy back 9:35 AM
  5. Regime filter (skip high-VIX nights)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ───────────────────────────────────────────────────────────────
HOURS_OVERNIGHT = 17.5         # 3:55 PM to 9:25 AM next day
HOURS_IN_DAY = 24.0
HOURS_MARKET = 6.5
TRADING_DAYS_YEAR = 252

DEFAULT_VIX_SKIP = 30.0       # skip if VIX > this
DEFAULT_MAX_RISK_PCT = 0.02   # 2% portfolio risk per night
DEFAULT_SLIPPAGE_PER_LEG = 0.05


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class GapDistribution:
    """Historical overnight gap statistics."""
    mean_gap_pct: float
    std_gap_pct: float
    p95_gap_pct: float         # 95th percentile absolute gap
    p99_gap_pct: float         # 99th percentile
    max_gap_pct: float
    n_observations: int
    positive_gap_rate: float   # fraction of gaps that are positive


@dataclass
class OvernightPremium:
    """Estimated overnight premium capture."""
    straddle_price: float      # total straddle premium at close
    theta_overnight: float     # theta captured overnight (dollars)
    theta_pct: float           # theta as % of straddle price
    gamma_risk: float          # overnight gamma exposure (dollars per 1% move)
    breakeven_gap_pct: float   # gap size that wipes out theta gain


@dataclass
class OvernightTrade:
    """Single overnight trade result."""
    trade_date: str
    spy_close: float
    spy_open_next: float
    gap_pct: float
    straddle_sold: float       # premium collected
    straddle_bought: float     # premium paid to close
    pnl: float                 # net after slippage
    contracts: int
    vix: float
    regime: str
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class BacktestResult:
    """Complete overnight gap backtest output."""
    trades: List[OvernightTrade] = field(default_factory=list)
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    skipped_nights: int = 0
    avg_pnl: float = 0.0
    total_pnl: float = 0.0
    ending_capital: float = 0.0
    yearly: Dict[str, Dict] = field(default_factory=dict)
    gap_distribution: Optional[GapDistribution] = None
    generated_at: str = ""


# ── Gap risk model ──────────────────────────────────────────────────────────
class GapRiskModel:
    """Models overnight gap risk from historical close-to-open returns."""

    def __init__(self, gap_history: Optional[np.ndarray] = None) -> None:
        self._gaps: np.ndarray = gap_history if gap_history is not None else np.array([])

    def fit(self, close_prices: np.ndarray, open_prices: np.ndarray) -> GapDistribution:
        """Fit gap distribution from close/open price pairs."""
        if len(close_prices) < 10 or len(close_prices) != len(open_prices):
            return GapDistribution(0, 0, 0, 0, 0, 0, 0.5)

        gaps = (open_prices - close_prices) / close_prices * 100  # percentage
        self._gaps = gaps
        abs_gaps = np.abs(gaps)

        return GapDistribution(
            mean_gap_pct=float(np.mean(gaps)),
            std_gap_pct=float(np.std(gaps)),
            p95_gap_pct=float(np.percentile(abs_gaps, 95)),
            p99_gap_pct=float(np.percentile(abs_gaps, 99)),
            max_gap_pct=float(abs_gaps.max()),
            n_observations=len(gaps),
            positive_gap_rate=float(np.mean(gaps > 0)),
        )

    def gap_var(self, confidence: float = 0.99) -> float:
        """Value-at-Risk for overnight gap (percentage)."""
        if len(self._gaps) < 10:
            return 2.0  # conservative default
        return float(np.percentile(np.abs(self._gaps), confidence * 100))


# ── Overnight premium calculator ────────────────────────────────────────────
class OvernightPremiumCalculator:
    """Estimates overnight theta capture and gamma risk."""

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self.rf = risk_free_rate

    def estimate(
        self,
        spy_price: float,
        vix: float,
        dte: float = 1.0,
    ) -> OvernightPremium:
        """Estimate overnight straddle premium and risks.

        Uses Black-Scholes approximation for ATM straddle pricing.
        """
        sigma = vix / 100.0  # annualised vol
        t = max(dte / 365.0, 1e-6)
        sqrt_t = math.sqrt(t)

        # ATM straddle ≈ 2 × S × σ × √T × 0.80 (approximation)
        straddle = 2 * spy_price * sigma * sqrt_t * 0.80

        # Theta per day ≈ -S × σ / (2 × √(2πT))
        theta_daily = spy_price * sigma / (2 * math.sqrt(2 * math.pi * t * 365))

        # Overnight fraction of daily theta (~73% since 17.5/24 hours)
        overnight_frac = HOURS_OVERNIGHT / HOURS_IN_DAY
        theta_overnight = theta_daily * overnight_frac

        theta_pct = theta_overnight / straddle * 100 if straddle > 0 else 0

        # Gamma risk: straddle delta sensitivity to gap
        # ATM gamma ≈ 1 / (S × σ × √T)
        gamma = 1.0 / (spy_price * sigma * sqrt_t) if sigma * sqrt_t > 0 else 0
        # Dollar gamma risk per 1% move
        gamma_risk = 0.5 * gamma * (spy_price * 0.01) ** 2 * 100  # per contract

        # Breakeven: gap that wipes out theta
        breakeven = theta_overnight / (spy_price * 0.01) if spy_price > 0 else 0
        breakeven_pct = breakeven * 0.01 * 100  # approximate

        return OvernightPremium(
            straddle_price=round(straddle, 2),
            theta_overnight=round(theta_overnight, 2),
            theta_pct=round(theta_pct, 2),
            gamma_risk=round(gamma_risk, 2),
            breakeven_gap_pct=round(breakeven_pct, 4),
        )


# ── Position sizer ──────────────────────────────────────────────────────────
def size_overnight_position(
    portfolio_value: float,
    spy_price: float,
    gap_var_pct: float,
    max_risk_pct: float = DEFAULT_MAX_RISK_PCT,
    contract_multiplier: int = 100,
) -> int:
    """Size position based on gap-at-risk.

    Contracts = max_dollar_risk / (gap_var × spy_price × multiplier).
    """
    max_dollar_risk = portfolio_value * max_risk_pct
    loss_per_contract = gap_var_pct / 100 * spy_price * contract_multiplier
    if loss_per_contract <= 0:
        return 0
    contracts = int(max_dollar_risk / loss_per_contract)
    return max(0, min(contracts, 50))  # cap at 50


# ── Backtest engine ─────────────────────────────────────────────────────────
class OvernightGapBacktest:
    """Backtest the overnight straddle strategy."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        vix_skip_threshold: float = DEFAULT_VIX_SKIP,
        max_risk_pct: float = DEFAULT_MAX_RISK_PCT,
        slippage_per_leg: float = DEFAULT_SLIPPAGE_PER_LEG,
        seed: int = 42,
    ) -> None:
        self.starting_capital = starting_capital
        self.vix_skip = vix_skip_threshold
        self.max_risk = max_risk_pct
        self.slippage = slippage_per_leg
        self.rng = np.random.RandomState(seed)

    def run(
        self,
        close_prices: np.ndarray,
        open_prices: np.ndarray,
        vix_series: np.ndarray,
        dates: Optional[List[str]] = None,
    ) -> BacktestResult:
        """Run overnight gap backtest.

        Parameters
        ----------
        close_prices : array of daily close prices
        open_prices : array of NEXT-DAY open prices (shifted by 1)
        vix_series : array of VIX at close
        dates : optional date labels
        """
        n = min(len(close_prices), len(open_prices), len(vix_series))
        if n < 20:
            return BacktestResult(generated_at=_now())

        # Fit gap model on first 60 days (walk-forward)
        gap_model = GapRiskModel()
        calc = OvernightPremiumCalculator()

        capital = self.starting_capital
        peak = capital
        max_dd = 0.0
        trades: List[OvernightTrade] = []
        daily_pnl: List[float] = []
        skipped = 0

        warmup = min(60, n // 4)

        for i in range(warmup, n):
            spy_close = float(close_prices[i])
            spy_open = float(open_prices[i])
            vix = float(vix_series[i])
            d = dates[i] if dates and i < len(dates) else str(i)

            # Update gap model with expanding window
            gap_dist = gap_model.fit(close_prices[:i], open_prices[:i])
            gap_var = gap_model.gap_var(0.99)

            # Regime classification
            if vix > 40:
                regime = "crash"
            elif vix > 25:
                regime = "high_vol"
            elif vix < 14:
                regime = "low_vol"
            else:
                regime = "bull"

            # Skip filter
            if vix > self.vix_skip:
                trades.append(OvernightTrade(
                    trade_date=d, spy_close=spy_close, spy_open_next=spy_open,
                    gap_pct=0, straddle_sold=0, straddle_bought=0, pnl=0,
                    contracts=0, vix=round(vix, 1), regime=regime,
                    skipped=True, skip_reason=f"VIX={vix:.0f}>{self.vix_skip}",
                ))
                skipped += 1
                continue

            # Weekend/holiday filter (gap > 3 days suggests weekend)
            actual_gap_pct = (spy_open - spy_close) / spy_close * 100

            # Position sizing
            contracts = size_overnight_position(
                capital, spy_close, gap_var, self.max_risk,
            )
            if contracts <= 0:
                continue

            # Premium calculation
            prem = calc.estimate(spy_close, vix, dte=1.0)
            straddle_sold = prem.straddle_price

            # Simulate overnight P&L
            # Straddle P&L = premium collected - intrinsic at open - remaining time value
            abs_gap = abs(actual_gap_pct) / 100 * spy_close
            intrinsic = abs_gap  # one leg goes ITM by gap amount

            # Time value remaining at open ≈ straddle * (1 - overnight_fraction × theta_pct/100)
            time_remaining_frac = 1.0 - prem.theta_pct / 100
            time_remaining_frac = max(0.05, min(0.95, time_remaining_frac))
            straddle_at_open = max(intrinsic, straddle_sold * time_remaining_frac)

            # Add noise for realistic pricing
            straddle_at_open *= (1 + self.rng.randn() * 0.03)
            straddle_at_open = max(0.01, straddle_at_open)

            pnl_per_contract = (straddle_sold - straddle_at_open) * 100
            total_slippage = self.slippage * 4 * contracts * 100  # 4 legs (sell 2, buy 2)
            total_pnl = pnl_per_contract * contracts - total_slippage

            trade = OvernightTrade(
                trade_date=d,
                spy_close=round(spy_close, 2),
                spy_open_next=round(spy_open, 2),
                gap_pct=round(actual_gap_pct, 4),
                straddle_sold=round(straddle_sold, 2),
                straddle_bought=round(straddle_at_open, 2),
                pnl=round(total_pnl, 2),
                contracts=contracts,
                vix=round(vix, 1),
                regime=regime,
            )
            trades.append(trade)

            capital += total_pnl
            daily_pnl.append(total_pnl)

            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Metrics
        executed = [t for t in trades if not t.skipped]
        wins = sum(1 for t in executed if t.pnl > 0)
        losses = len(executed) - wins
        total_pnl_val = sum(t.pnl for t in executed)
        win_rate = wins / len(executed) * 100 if executed else 0
        avg_pnl = total_pnl_val / len(executed) if executed else 0

        total_return = (capital - self.starting_capital) / self.starting_capital * 100
        years = (n - warmup) / TRADING_DAYS_YEAR
        cagr = ((capital / self.starting_capital) ** (1 / years) - 1) * 100 if years > 0 and capital > 0 else 0

        dr = np.array(daily_pnl)
        sharpe = float(dr.mean() / dr.std() * np.sqrt(TRADING_DAYS_YEAR)) if len(dr) > 1 and dr.std() > 0 else 0

        win_sum = sum(t.pnl for t in executed if t.pnl > 0)
        loss_sum = abs(sum(t.pnl for t in executed if t.pnl < 0))
        pf = win_sum / loss_sum if loss_sum > 0 else 0

        # Yearly
        yearly: Dict[str, Dict] = {}
        for t in executed:
            y = t.trade_date[:4] if len(t.trade_date) >= 4 else "unknown"
            if y not in yearly:
                yearly[y] = {"trades": 0, "pnl": 0.0, "wins": 0}
            yearly[y]["trades"] += 1
            yearly[y]["pnl"] += t.pnl
            yearly[y]["wins"] += int(t.pnl > 0)
        for y in yearly:
            yearly[y]["win_rate"] = round(yearly[y]["wins"] / yearly[y]["trades"] * 100, 1) if yearly[y]["trades"] > 0 else 0
            yearly[y]["pnl"] = round(yearly[y]["pnl"], 2)

        return BacktestResult(
            trades=trades,
            total_return_pct=round(total_return, 2),
            cagr_pct=round(cagr, 2),
            sharpe=round(sharpe, 2),
            max_dd_pct=round(max_dd * 100, 2),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(pf, 2),
            total_trades=len(executed),
            skipped_nights=skipped,
            avg_pnl=round(avg_pnl, 2),
            total_pnl=round(total_pnl_val, 2),
            ending_capital=round(capital, 2),
            yearly=yearly,
            gap_distribution=gap_model.fit(close_prices[:n], open_prices[:n]),
            generated_at=_now(),
        )


# ── Synthetic data generator ───────────────────────────────────────────────
def generate_spy_data(n: int = 1500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic SPY close/open/VIX data for backtesting."""
    rng = np.random.RandomState(seed)
    dates_idx = pd.bdate_range("2020-01-02", periods=n)
    dates = [str(d.date()) for d in dates_idx]

    # SPY close prices
    close = np.zeros(n)
    close[0] = 320.0
    for i in range(1, n):
        close[i] = close[i - 1] * np.exp(rng.randn() * 0.012 + 0.0003)

    # SPY open (next day) = close + overnight gap
    # Overnight gaps: mean ~+0.02%, std ~0.5%
    gaps = rng.randn(n) * 0.005 + 0.0002
    # Inject a few large gaps
    for idx in rng.choice(n, size=min(10, n), replace=False):
        gaps[idx] = rng.choice([-1, 1]) * rng.uniform(0.02, 0.05)
    open_next = close * (1 + gaps)

    # VIX
    vix = np.zeros(n)
    vix[0] = 15.0
    for i in range(1, n):
        vix[i] = max(9, min(80, vix[i - 1] + 0.03 * (18 - vix[i - 1]) + rng.randn() * 1.5))

    return close, open_next, vix, dates


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
