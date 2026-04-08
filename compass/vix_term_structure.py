"""VIX term structure trading engine.

Analyses VIX futures contango/backwardation to predict SPY options
premium direction.  Generates signals for selling premium in contango
and buying protection in backwardation.

Pure-Python — no external dependencies.

Typical usage::

    from compass.vix_term_structure import VIXTermEngine, VIXCurvePoint
    curve = [VIXCurvePoint(month=1, vix_future=18.5), ...]
    engine = VIXTermEngine(history)
    result = engine.analyse()
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _std(xs: List[float]) -> float:
    if len(xs) < 2: return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def _percentile(xs: List[float], pct: float) -> float:
    if not xs: return 0.0
    s = sorted(xs); idx = pct / 100 * (len(s) - 1)
    lo = int(idx); hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VIXCurvePoint:
    """One VIX futures contract."""
    month: int          # months to expiry (1=front, 2=second, etc.)
    vix_future: float   # futures price


@dataclass
class VIXCurveSnapshot:
    """Full VIX term structure for one day."""
    date: str
    spot_vix: float
    futures: List[VIXCurvePoint]
    spy_return_5d: float = 0.0   # for backtest labelling


# ---------------------------------------------------------------------------
# Term structure calculator
# ---------------------------------------------------------------------------

@dataclass
class TermStructureMetrics:
    """Computed metrics from a VIX curve snapshot."""
    date: str
    spot_vix: float
    front_future: float
    second_future: float
    slope: float                # (M2 - M1) / M1
    spot_to_front: float        # (front - spot) / spot
    contango_pct: float         # positive = contango
    regime: str                 # "contango", "backwardation", "flat"
    slope_zscore: float         # slope relative to history
    is_extreme: bool            # |zscore| > 1.5


def compute_term_structure(
    snapshot: VIXCurveSnapshot,
    historical_slopes: Optional[List[float]] = None,
) -> TermStructureMetrics:
    """Compute term structure metrics from a curve snapshot."""
    futures = sorted(snapshot.futures, key=lambda f: f.month)
    front = futures[0].vix_future if futures else snapshot.spot_vix
    second = futures[1].vix_future if len(futures) > 1 else front

    slope = (second - front) / front if front > 0 else 0.0
    spot_to_front = (front - snapshot.spot_vix) / snapshot.spot_vix if snapshot.spot_vix > 0 else 0.0
    contango_pct = slope * 100

    if slope > 0.02:
        regime = "contango"
    elif slope < -0.02:
        regime = "backwardation"
    else:
        regime = "flat"

    # Z-score relative to history
    if historical_slopes and len(historical_slopes) >= 10:
        h_mean = _mean(historical_slopes)
        h_std = _std(historical_slopes)
        zscore = (slope - h_mean) / h_std if h_std > 0 else 0.0
    else:
        zscore = 0.0

    return TermStructureMetrics(
        date=snapshot.date, spot_vix=snapshot.spot_vix,
        front_future=round(front, 2), second_future=round(second, 2),
        slope=round(slope, 4), spot_to_front=round(spot_to_front, 4),
        contango_pct=round(contango_pct, 2),
        regime=regime, slope_zscore=round(zscore, 2),
        is_extreme=abs(zscore) > 1.5,
    )


# ---------------------------------------------------------------------------
# Contango/backwardation regime detector
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    """Statistics for one term structure regime."""
    regime: str
    n_days: int
    pct_of_total: float
    avg_slope: float
    avg_spot_vix: float
    avg_subsequent_return: float  # avg 5d SPY return when in this regime


def compute_regime_stats(
    metrics: List[TermStructureMetrics],
    snapshots: List[VIXCurveSnapshot],
) -> List[RegimeStats]:
    """Aggregate stats by term structure regime."""
    by_regime: Dict[str, Tuple[List[float], List[float], List[float]]] = defaultdict(
        lambda: ([], [], []))

    for m, s in zip(metrics, snapshots):
        slopes, vixs, rets = by_regime[m.regime]
        slopes.append(m.slope)
        vixs.append(m.spot_vix)
        rets.append(s.spy_return_5d)

    total = len(metrics)
    results: List[RegimeStats] = []
    for regime in ["contango", "flat", "backwardation"]:
        slopes, vixs, rets = by_regime.get(regime, ([], [], []))
        n = len(slopes)
        results.append(RegimeStats(
            regime=regime, n_days=n,
            pct_of_total=round(n / total * 100, 1) if total > 0 else 0,
            avg_slope=round(_mean(slopes), 4),
            avg_spot_vix=round(_mean(vixs), 1),
            avg_subsequent_return=round(_mean(rets), 4),
        ))
    return results


# ---------------------------------------------------------------------------
# Mean-reversion signals
# ---------------------------------------------------------------------------

@dataclass
class MeanReversionSignal:
    """Signal when term structure is at extremes."""
    date: str
    direction: str      # "sell_premium" (extreme contango) or "buy_protection" (extreme backwardation)
    strength: float     # 0 to 1
    slope: float
    zscore: float
    reasoning: str


def generate_mean_reversion_signals(
    metrics: List[TermStructureMetrics],
    zscore_threshold: float = 1.5,
) -> List[MeanReversionSignal]:
    """Generate signals at extreme term structure levels."""
    signals: List[MeanReversionSignal] = []
    for m in metrics:
        if not m.is_extreme:
            continue
        if m.slope_zscore > zscore_threshold:
            signals.append(MeanReversionSignal(
                date=m.date, direction="sell_premium",
                strength=min(1.0, abs(m.slope_zscore) / 3.0),
                slope=m.slope, zscore=m.slope_zscore,
                reasoning=f"Extreme contango (z={m.slope_zscore:.1f}): curve will flatten, sell vol",
            ))
        elif m.slope_zscore < -zscore_threshold:
            signals.append(MeanReversionSignal(
                date=m.date, direction="buy_protection",
                strength=min(1.0, abs(m.slope_zscore) / 3.0),
                slope=m.slope, zscore=m.slope_zscore,
                reasoning=f"Extreme backwardation (z={m.slope_zscore:.1f}): fear elevated, buy protection",
            ))
    return signals


# ---------------------------------------------------------------------------
# Position sizing from term structure
# ---------------------------------------------------------------------------

@dataclass
class SizingRecommendation:
    """Position size based on term structure."""
    date: str
    regime: str
    base_size_pct: float    # % of portfolio to deploy
    slope_adjustment: float # multiplier from slope magnitude
    final_size_pct: float


def compute_sizing(
    metrics: List[TermStructureMetrics],
    base_size: float = 0.10,
) -> List[SizingRecommendation]:
    """Size positions based on term structure regime and slope."""
    results: List[SizingRecommendation] = []
    for m in metrics:
        if m.regime == "contango":
            # Contango = sell premium, scale with slope
            adj = 1.0 + min(1.0, m.slope * 5)  # up to 2x in steep contango
        elif m.regime == "backwardation":
            # Backwardation = reduce exposure
            adj = max(0.2, 1.0 + m.slope * 3)  # slope is negative → reduces
        else:
            adj = 0.8  # flat = slightly reduced

        final = round(base_size * adj, 4)
        results.append(SizingRecommendation(
            date=m.date, regime=m.regime,
            base_size_pct=base_size, slope_adjustment=round(adj, 3),
            final_size_pct=final,
        ))
    return results


# ---------------------------------------------------------------------------
# Backtest: sell premium in contango, buy in backwardation
# ---------------------------------------------------------------------------

@dataclass
class BacktestTrade:
    date: str
    regime: str
    action: str     # "sell_premium" or "buy_protection" or "skip"
    size: float
    pnl: float
    is_winner: bool


@dataclass
class BacktestResult:
    n_trades: int
    n_winners: int
    win_rate: float
    total_pnl: float
    avg_pnl: float
    sharpe: float
    max_dd: float
    contango_pnl: float
    backwardation_pnl: float
    flat_pnl: float
    trades: List[BacktestTrade]


def run_backtest(
    metrics: List[TermStructureMetrics],
    snapshots: List[VIXCurveSnapshot],
    base_premium: float = 0.015,  # base premium per trade as fraction
) -> BacktestResult:
    """Backtest selling premium in contango, buying in backwardation."""
    trades: List[BacktestTrade] = []
    pnls: List[float] = []
    regime_pnls: Dict[str, float] = {"contango": 0, "backwardation": 0, "flat": 0}

    for m, s in zip(metrics, snapshots):
        if m.regime == "contango":
            # Sell premium: profit from theta if SPY doesn't crash
            edge = base_premium * (1 + m.slope * 3)
            noise_factor = 0.5 + abs(m.slope) * 2
            pnl = edge + s.spy_return_5d * 0.3  # partial market exposure
            action = "sell_premium"
        elif m.regime == "backwardation":
            # Buy protection: profit if market drops
            edge = -base_premium * 0.5  # pay some premium
            pnl = edge - s.spy_return_5d * 0.5  # benefit from drops
            action = "buy_protection"
        else:
            pnl = base_premium * 0.3  # small theta in flat
            action = "skip"

        trades.append(BacktestTrade(
            date=m.date, regime=m.regime, action=action,
            size=1.0, pnl=round(pnl, 6), is_winner=pnl > 0,
        ))
        pnls.append(pnl)
        regime_pnls[m.regime] += pnl

    if not trades:
        return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, [])

    n = len(trades)
    winners = sum(1 for t in trades if t.is_winner)
    total = sum(pnls)
    avg = _mean(pnls)
    std = _std(pnls)
    sharpe = avg / std * math.sqrt(52) if std > 0 else 0  # weekly-ish

    cum = 0.0; peak = 0.0; worst_dd = 0.0
    for p in pnls:
        cum += p
        if cum > peak: peak = cum
        dd = (peak - cum) / max(abs(peak), 0.001)
        if dd > worst_dd: worst_dd = dd

    return BacktestResult(
        n_trades=n, n_winners=winners,
        win_rate=round(winners / n, 4),
        total_pnl=round(total, 4), avg_pnl=round(avg, 6),
        sharpe=round(sharpe, 2), max_dd=round(worst_dd, 4),
        contango_pnl=round(regime_pnls["contango"], 4),
        backwardation_pnl=round(regime_pnls["backwardation"], 4),
        flat_pnl=round(regime_pnls["flat"], 4),
        trades=trades,
    )


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

@dataclass
class VIXTermResult:
    n_days: int
    metrics: List[TermStructureMetrics]
    regime_stats: List[RegimeStats]
    signals: List[MeanReversionSignal]
    sizing: List[SizingRecommendation]
    backtest: BacktestResult
    avg_contango: float
    pct_contango: float
    pct_backwardation: float


class VIXTermEngine:
    """VIX term structure analysis engine."""

    def __init__(self, snapshots: List[VIXCurveSnapshot]) -> None:
        self.snapshots = snapshots

    def analyse(self) -> VIXTermResult:
        # Compute metrics with rolling history
        metrics: List[TermStructureMetrics] = []
        historical_slopes: List[float] = []
        for s in self.snapshots:
            m = compute_term_structure(s, historical_slopes if len(historical_slopes) >= 10 else None)
            metrics.append(m)
            historical_slopes.append(m.slope)

        regime_stats = compute_regime_stats(metrics, self.snapshots)
        signals = generate_mean_reversion_signals(metrics)
        sizing = compute_sizing(metrics)
        bt = run_backtest(metrics, self.snapshots)

        slopes = [m.slope for m in metrics]
        contango_days = sum(1 for m in metrics if m.regime == "contango")
        backw_days = sum(1 for m in metrics if m.regime == "backwardation")
        n = len(metrics)

        return VIXTermResult(
            n_days=n, metrics=metrics, regime_stats=regime_stats,
            signals=signals, sizing=sizing, backtest=bt,
            avg_contango=round(_mean(slopes) * 100, 2),
            pct_contango=round(contango_days / n * 100, 1) if n > 0 else 0,
            pct_backwardation=round(backw_days / n * 100, 1) if n > 0 else 0,
        )


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

def generate_sample_data(n_days: int = 500, seed: int = 1080) -> List[VIXCurveSnapshot]:
    """Generate synthetic VIX term structure history."""
    rng = random.Random(seed)
    snapshots: List[VIXCurveSnapshot] = []
    vix = 18.0

    phases = [
        (100, 16.0, 0.04),   # calm contango
        (50, 28.0, -0.02),   # stress backwardation
        (80, 20.0, 0.03),    # recovery
        (120, 14.0, 0.05),   # low vol steep contango
        (50, 35.0, -0.04),   # crisis backwardation
        (100, 18.0, 0.02),   # normalisation
    ]

    day = 0
    for n, vix_target, slope_base in phases:
        for _ in range(n):
            if day >= n_days:
                break
            vix += 0.1 * (vix_target - vix) + rng.gauss(0, 1.5)
            vix = max(9, vix)

            slope = slope_base + rng.gauss(0, 0.015)
            front = vix * (1 + slope * 0.5 + rng.gauss(0, 0.01))
            second = front * (1 + slope + rng.gauss(0, 0.005))
            third = second * (1 + slope * 0.8 + rng.gauss(0, 0.005))

            spy_ret = rng.gauss(0.001, 0.015) - (vix - 18) * 0.0005

            snapshots.append(VIXCurveSnapshot(
                date=f"day-{day}", spot_vix=round(vix, 2),
                futures=[
                    VIXCurvePoint(1, round(max(9, front), 2)),
                    VIXCurvePoint(2, round(max(9, second), 2)),
                    VIXCurvePoint(3, round(max(9, third), 2)),
                ],
                spy_return_5d=round(spy_ret, 4),
            ))
            day += 1

    return snapshots[:n_days]
