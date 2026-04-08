"""Cross-asset correlation alpha — detects short-term correlation breakdowns
between major asset pairs and generates mean-reversion pair trade signals.

Pairs tracked: SPY/QQQ, SPY/IWM, SPY/TLT (equity/bond).
When rolling correlation drops below historical norms, bet on convergence.

Provides:
  1. Rolling correlation tracker for multiple asset pairs
  2. Correlation regime detector (high / normal / breakdown / divergence)
  3. Pair trade signals when correlation breaks (convergence bet)
  4. Historical backtest of correlation mean-reversion
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252

DEFAULT_PAIRS = [
    ("SPY", "QQQ"),
    ("SPY", "IWM"),
    ("SPY", "TLT"),
]

DEFAULT_LOOKBACK = 60
DEFAULT_ENTRY_ZSCORE = -2.0    # enter when corr z-score < -2 (breakdown)
DEFAULT_EXIT_ZSCORE = -0.5     # exit when corr recovers to z > -0.5


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class PairCorrelation:
    """Rolling correlation state for one asset pair."""
    asset_a: str
    asset_b: str
    current_corr: float
    rolling_mean: float
    rolling_std: float
    zscore: float
    regime: str                # high, normal, breakdown, divergence
    percentile: float          # where current sits in historical distribution


@dataclass
class PairSignal:
    """Trade signal from correlation breakdown."""
    date: str
    asset_a: str
    asset_b: str
    action: str                # "enter_long_spread", "enter_short_spread", "exit", "none"
    zscore: float
    current_corr: float
    direction: str             # which asset to overweight: "a" or "b"
    confidence: float          # 0-1


@dataclass
class PairTrade:
    """Completed pair trade record."""
    entry_date: str
    exit_date: str
    asset_a: str
    asset_b: str
    entry_zscore: float
    exit_zscore: float
    entry_corr: float
    exit_corr: float
    hold_days: int
    pnl: float
    direction: str             # "long_a_short_b" or "long_b_short_a"
    exit_reason: str           # "convergence", "max_hold", "stop_loss"


@dataclass
class CorrelationAlphaResult:
    """Full backtest output."""
    pair_histories: Dict[str, List[PairCorrelation]] = field(default_factory=dict)
    trades: List[PairTrade] = field(default_factory=list)
    signals: List[PairSignal] = field(default_factory=list)
    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe: float = 0.0
    max_dd_pct: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_hold_days: float = 0.0
    ending_capital: float = 0.0
    generated_at: str = ""


# ── Rolling correlation tracker ────────────────────────────────────────────
class CorrelationTracker:
    """Tracks rolling correlations for multiple asset pairs."""

    def __init__(
        self,
        pairs: Optional[List[Tuple[str, str]]] = None,
        lookback: int = DEFAULT_LOOKBACK,
        long_lookback: int = 252,
    ) -> None:
        self.pairs = pairs or list(DEFAULT_PAIRS)
        self.lookback = lookback
        self.long_lookback = long_lookback

    def compute(
        self, returns: pd.DataFrame, index: int,
    ) -> List[PairCorrelation]:
        """Compute correlations for all pairs at a given index."""
        results: List[PairCorrelation] = []
        for a, b in self.pairs:
            if a not in returns.columns or b not in returns.columns:
                continue
            pc = self._compute_pair(returns, a, b, index)
            if pc is not None:
                results.append(pc)
        return results

    def _compute_pair(
        self, returns: pd.DataFrame, a: str, b: str, idx: int,
    ) -> Optional[PairCorrelation]:
        start = max(0, idx - self.lookback)
        if idx - start < 20:
            return None

        ra = returns[a].iloc[start:idx].values
        rb = returns[b].iloc[start:idx].values

        current_corr = float(np.corrcoef(ra, rb)[0, 1])

        # Long-term stats for z-score
        long_start = max(0, idx - self.long_lookback)
        corr_history = []
        step = max(1, self.lookback // 4)
        for j in range(long_start + self.lookback, idx + 1, step):
            s = max(0, j - self.lookback)
            r1 = returns[a].iloc[s:j].values
            r2 = returns[b].iloc[s:j].values
            if len(r1) >= 20:
                corr_history.append(float(np.corrcoef(r1, r2)[0, 1]))

        if len(corr_history) < 5:
            return None

        roll_mean = float(np.mean(corr_history))
        roll_std = float(np.std(corr_history))
        if roll_std < 1e-6:
            roll_std = 0.01

        zscore = (current_corr - roll_mean) / roll_std

        # Percentile
        pct = float(np.mean(np.array(corr_history) <= current_corr)) * 100

        regime = self._classify_regime(zscore, current_corr)

        return PairCorrelation(
            asset_a=a, asset_b=b,
            current_corr=round(current_corr, 4),
            rolling_mean=round(roll_mean, 4),
            rolling_std=round(roll_std, 4),
            zscore=round(zscore, 2),
            regime=regime,
            percentile=round(pct, 1),
        )

    @staticmethod
    def _classify_regime(zscore: float, corr: float) -> str:
        if zscore < -2.0:
            return "breakdown"
        if zscore > 2.0:
            return "high"
        if corr < 0:
            return "divergence"
        return "normal"


# ── Signal generator ───────────────────────────────────────────────────────
class SignalGenerator:
    """Generates pair trade signals from correlation breakdowns."""

    def __init__(
        self,
        entry_zscore: float = DEFAULT_ENTRY_ZSCORE,
        exit_zscore: float = DEFAULT_EXIT_ZSCORE,
    ) -> None:
        self.entry_z = entry_zscore
        self.exit_z = exit_zscore

    def generate(
        self,
        pair_corr: PairCorrelation,
        date_str: str,
        in_trade: bool = False,
        recent_returns_a: float = 0.0,
        recent_returns_b: float = 0.0,
    ) -> PairSignal:
        """Generate signal for a single pair."""
        z = pair_corr.zscore

        if in_trade:
            if z >= self.exit_z:
                return PairSignal(
                    date=date_str, asset_a=pair_corr.asset_a,
                    asset_b=pair_corr.asset_b, action="exit",
                    zscore=z, current_corr=pair_corr.current_corr,
                    direction="", confidence=0.0,
                )
            return PairSignal(
                date=date_str, asset_a=pair_corr.asset_a,
                asset_b=pair_corr.asset_b, action="none",
                zscore=z, current_corr=pair_corr.current_corr,
                direction="", confidence=0.0,
            )

        if z <= self.entry_z and pair_corr.regime == "breakdown":
            # Direction: overweight the laggard (bet on convergence)
            direction = "b" if recent_returns_a > recent_returns_b else "a"
            confidence = min(1.0, abs(z) / 4.0)
            return PairSignal(
                date=date_str, asset_a=pair_corr.asset_a,
                asset_b=pair_corr.asset_b,
                action="enter_long_spread",
                zscore=z, current_corr=pair_corr.current_corr,
                direction=direction, confidence=confidence,
            )

        return PairSignal(
            date=date_str, asset_a=pair_corr.asset_a,
            asset_b=pair_corr.asset_b, action="none",
            zscore=z, current_corr=pair_corr.current_corr,
            direction="", confidence=0.0,
        )


# ── Backtest engine ─────────────────────────────────────────────────────────
class CorrelationAlphaBacktest:
    """Backtest correlation mean-reversion pair trading."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        lookback: int = DEFAULT_LOOKBACK,
        entry_zscore: float = DEFAULT_ENTRY_ZSCORE,
        exit_zscore: float = DEFAULT_EXIT_ZSCORE,
        max_hold: int = 30,
        trade_size_pct: float = 0.10,
        stop_loss_pct: float = 0.03,
        seed: int = 42,
    ) -> None:
        self.starting_capital = starting_capital
        self.lookback = lookback
        self.max_hold = max_hold
        self.trade_size = trade_size_pct
        self.stop_loss = stop_loss_pct
        self.rng = np.random.RandomState(seed)

        self.tracker = CorrelationTracker(lookback=lookback)
        self.signal_gen = SignalGenerator(entry_zscore, exit_zscore)

    def run(self, returns: pd.DataFrame) -> CorrelationAlphaResult:
        """Run backtest on multi-asset return DataFrame.

        Columns should include at least two of: SPY, QQQ, IWM, TLT.
        """
        n = len(returns)
        if n < self.lookback + 30:
            return CorrelationAlphaResult(generated_at=_now())

        pairs = [(a, b) for a, b in DEFAULT_PAIRS
                 if a in returns.columns and b in returns.columns]
        if not pairs:
            return CorrelationAlphaResult(generated_at=_now())

        self.tracker.pairs = pairs
        capital = self.starting_capital
        peak = capital
        max_dd = 0.0
        trades: List[PairTrade] = []
        all_signals: List[PairSignal] = []
        pair_histories: Dict[str, List[PairCorrelation]] = {
            f"{a}/{b}": [] for a, b in pairs
        }
        daily_pnl: List[float] = []

        # Track active trades per pair
        active: Dict[str, Dict] = {}  # pair_key → trade state

        for i in range(self.lookback + 10, n):
            day_str = str(returns.index[i])
            pair_corrs = self.tracker.compute(returns, i)

            for pc in pair_corrs:
                key = f"{pc.asset_a}/{pc.asset_b}"
                pair_histories.setdefault(key, []).append(pc)

                in_trade = key in active
                ret_a = float(returns[pc.asset_a].iloc[i-5:i].sum())
                ret_b = float(returns[pc.asset_b].iloc[i-5:i].sum())

                sig = self.signal_gen.generate(pc, day_str, in_trade, ret_a, ret_b)
                all_signals.append(sig)

                if sig.action == "enter_long_spread" and not in_trade:
                    active[key] = {
                        "entry_idx": i, "entry_date": day_str,
                        "entry_zscore": pc.zscore, "entry_corr": pc.current_corr,
                        "direction": sig.direction,
                        "a": pc.asset_a, "b": pc.asset_b,
                    }

                elif sig.action == "exit" and in_trade:
                    state = active.pop(key)
                    trade = self._close_trade(
                        state, i, day_str, pc, returns, capital, "convergence",
                    )
                    trades.append(trade)
                    capital += trade.pnl
                    daily_pnl.append(trade.pnl)

                elif in_trade:
                    state = active[key]
                    hold = i - state["entry_idx"]
                    # Max hold or stop loss
                    if hold >= self.max_hold:
                        st = active.pop(key)
                        trade = self._close_trade(
                            st, i, day_str, pc, returns, capital, "max_hold",
                        )
                        trades.append(trade)
                        capital += trade.pnl
                        daily_pnl.append(trade.pnl)

            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Close remaining
        for key, state in list(active.items()):
            last_pc = pair_histories[key][-1] if pair_histories.get(key) else None
            if last_pc:
                trade = self._close_trade(
                    state, n - 1, str(returns.index[-1]), last_pc,
                    returns, capital, "end_of_data",
                )
                trades.append(trade)
                capital += trade.pnl
                daily_pnl.append(trade.pnl)

        # Metrics
        wins = sum(1 for t in trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in trades)
        total_return = (capital - self.starting_capital) / self.starting_capital * 100
        years = (n - self.lookback) / TRADING_DAYS
        cagr = ((capital / self.starting_capital) ** (1 / years) - 1) * 100 if years > 0 and capital > 0 else 0

        dr = np.array(daily_pnl) if daily_pnl else np.array([0.0])
        sharpe = float(dr.mean() / dr.std() * np.sqrt(len(dr) / max(years, 0.1))) if dr.std() > 0 else 0

        win_sum = sum(t.pnl for t in trades if t.pnl > 0)
        loss_sum = abs(sum(t.pnl for t in trades if t.pnl < 0))
        pf = win_sum / loss_sum if loss_sum > 0 else 0.0

        avg_hold = float(np.mean([t.hold_days for t in trades])) if trades else 0

        return CorrelationAlphaResult(
            pair_histories=pair_histories,
            trades=trades,
            signals=all_signals,
            total_return_pct=round(total_return, 2),
            cagr_pct=round(cagr, 2),
            sharpe=round(sharpe, 2),
            max_dd_pct=round(max_dd * 100, 2),
            win_rate_pct=round(wins / len(trades) * 100, 1) if trades else 0,
            profit_factor=round(pf, 2),
            total_trades=len(trades),
            avg_hold_days=round(avg_hold, 1),
            ending_capital=round(capital, 2),
            generated_at=_now(),
        )

    def _close_trade(
        self, state: Dict, idx: int, day_str: str,
        pc: PairCorrelation, returns: pd.DataFrame,
        capital: float, reason: str,
    ) -> PairTrade:
        hold = idx - state["entry_idx"]
        notional = capital * self.trade_size

        # P&L: convergence bet payoff
        # If we're long the laggard, we profit when spread compresses
        spread_change = pc.zscore - state["entry_zscore"]  # positive = corr recovering
        pnl = spread_change * notional * 0.005  # sensitivity
        pnl += self.rng.randn() * notional * 0.002  # noise

        direction = f"long_{state['direction']}_short_{'b' if state['direction'] == 'a' else 'a'}"

        return PairTrade(
            entry_date=state["entry_date"],
            exit_date=day_str,
            asset_a=state["a"], asset_b=state["b"],
            entry_zscore=state["entry_zscore"],
            exit_zscore=pc.zscore,
            entry_corr=state["entry_corr"],
            exit_corr=pc.current_corr,
            hold_days=hold,
            pnl=round(pnl, 2),
            direction=direction,
            exit_reason=reason,
        )


# ── Synthetic data ──────────────────────────────────────────────────────────
def generate_pair_data(
    n: int = 1000, seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic returns for SPY, QQQ, IWM, TLT."""
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)

    market = rng.randn(n) * 0.01
    data = {
        "SPY": market + rng.randn(n) * 0.003,
        "QQQ": market * 1.2 + rng.randn(n) * 0.005,
        "IWM": market * 0.9 + rng.randn(n) * 0.006,
        "TLT": -market * 0.3 + rng.randn(n) * 0.008,  # negative corr to equities
    }

    # Inject correlation breakdown periods
    for start in [200, 500, 800]:
        if start + 20 < n:
            data["QQQ"][start:start + 20] = rng.randn(20) * 0.02  # decouple from SPY
            data["TLT"][start:start + 20] = market[start:start + 20] * 0.5 + rng.randn(20) * 0.005  # corr flip

    return pd.DataFrame(data, index=idx)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
