"""
Systematic volatility surface trading.

IV surface builder, skew normality scoring, term structure slope
strategy, butterfly spread generator, and backtest framework for
vol surface strategies.  Integrates with compass.iv_surface.

Usage::

    from compass.vol_surface_trader import VolSurfaceTrader
    trader = VolSurfaceTrader(options_chain)
    signals = trader.analyze()
    bt = trader.backtest(trades_df)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class IVPoint:
    """Single point on the IV surface."""
    strike: float
    expiry_days: int
    iv: float
    delta: float
    option_type: str       # "call" or "put"
    moneyness: float       # strike / underlying


@dataclass
class IVSurface:
    """Constructed IV surface."""
    underlying_price: float
    timestamp: str
    points: List[IVPoint]
    expirations: List[int]     # unique DTE values
    strikes: List[float]       # unique strikes
    atm_iv: float
    skew_25d: float            # 25-delta put IV − 25-delta call IV
    skew_10d: float            # 10-delta put IV − 10-delta call IV
    term_slope: float          # back-month IV − front-month IV
    smile_curvature: float     # butterfly = wing avg IV − ATM IV


@dataclass
class SkewScore:
    """Skew normality assessment."""
    score: float               # -1 (puts cheap) to +1 (puts expensive)
    percentile: float          # vs historical
    put_wing_iv: float
    call_wing_iv: float
    atm_iv: float
    signal: str                # "sell_puts", "sell_calls", "neutral"
    confidence: float


@dataclass
class TermStructureSignal:
    """Term structure trading signal."""
    slope: float               # back − front (positive = contango)
    slope_pct: float           # as % of front IV
    regime: str                # "steep_contango", "mild_contango", "flat", "backwardation"
    signal: str                # "sell_front", "reduce_exposure", "hedge", "neutral"
    confidence: float
    front_iv: float
    back_iv: float


@dataclass
class ButterflySpread:
    """Generated butterfly spread for smile arbitrage."""
    center_strike: float
    wing_width: float
    lower_strike: float
    upper_strike: float
    net_credit: float          # estimated
    max_loss: float
    expected_pnl: float
    iv_edge: float             # excess IV being sold
    direction: str             # "sell_butterfly" or "buy_butterfly"


@dataclass
class SurfaceSignal:
    """Combined vol surface signal."""
    action: str                # "aggressive_sell", "normal_sell", "reduce", "hedge", "flat"
    skew: SkewScore
    term: TermStructureSignal
    butterflies: List[ButterflySpread]
    regime: str
    composite_score: float     # -1 (max bearish vol) to +1 (max bullish vol)
    timestamp: str


@dataclass
class BacktestResult:
    """Vol surface strategy backtest result."""
    strategy: str
    n_trades: int
    n_signal_trades: int       # trades taken on surface signal
    win_rate_all: float
    win_rate_signal: float
    pnl_all: float
    pnl_signal: float
    sharpe_all: float
    sharpe_signal: float
    dd_all: float
    dd_signal: float
    improvement_pct: float     # signal P&L vs all P&L


# ── IV surface builder ──────────────────────────────────────────────────


def build_surface(
    chain: pd.DataFrame,
    underlying_price: float,
    iv_col: str = "iv",
    strike_col: str = "strike",
    dte_col: str = "dte",
    type_col: str = "option_type",
    delta_col: str = "delta",
) -> IVSurface:
    """Build an IV surface from an options chain DataFrame."""
    if chain.empty:
        return IVSurface(underlying_price, "", [], [], [], 0, 0, 0, 0, 0)

    chain = chain.copy()
    chain["moneyness"] = chain[strike_col] / underlying_price

    points = []
    for _, row in chain.iterrows():
        points.append(IVPoint(
            strike=float(row[strike_col]),
            expiry_days=int(row[dte_col]),
            iv=float(row[iv_col]),
            delta=float(row.get(delta_col, 0)),
            option_type=str(row.get(type_col, "put")),
            moneyness=float(row["moneyness"]),
        ))

    expirations = sorted(chain[dte_col].unique().tolist())
    strikes = sorted(chain[strike_col].unique().tolist())

    # ATM IV: closest to moneyness=1.0
    chain["abs_money"] = abs(chain["moneyness"] - 1.0)
    atm_row = chain.loc[chain["abs_money"].idxmin()]
    atm_iv = float(atm_row[iv_col])

    # Skew: OTM put IV - OTM call IV at 25-delta and 10-delta
    skew_25d = _compute_skew(chain, underlying_price, iv_col, strike_col, type_col, 0.25)
    skew_10d = _compute_skew(chain, underlying_price, iv_col, strike_col, type_col, 0.10)

    # Term structure: back-month minus front-month ATM IV
    term_slope = _compute_term_slope(chain, underlying_price, iv_col, strike_col, dte_col)

    # Smile curvature: avg wing IV - ATM IV
    curvature = _compute_curvature(chain, underlying_price, iv_col, strike_col, atm_iv)

    return IVSurface(
        underlying_price=underlying_price,
        timestamp=datetime.now(timezone.utc).isoformat(),
        points=points, expirations=expirations, strikes=strikes,
        atm_iv=atm_iv, skew_25d=skew_25d, skew_10d=skew_10d,
        term_slope=term_slope, smile_curvature=curvature,
    )


def _compute_skew(
    chain: pd.DataFrame, price: float, iv_col: str,
    strike_col: str, type_col: str, target_otm_pct: float,
) -> float:
    """Skew = OTM put IV - OTM call IV at given OTM %."""
    put_strike = price * (1 - target_otm_pct)
    call_strike = price * (1 + target_otm_pct)

    puts = chain[chain.get(type_col, pd.Series(dtype=str)).str.lower() == "put"]
    calls = chain[chain.get(type_col, pd.Series(dtype=str)).str.lower() == "call"]

    put_iv = _nearest_iv(puts, strike_col, iv_col, put_strike)
    call_iv = _nearest_iv(calls, strike_col, iv_col, call_strike)

    if put_iv is not None and call_iv is not None:
        return put_iv - call_iv
    return 0.0


def _compute_term_slope(
    chain: pd.DataFrame, price: float, iv_col: str,
    strike_col: str, dte_col: str,
) -> float:
    """Back-month ATM IV minus front-month ATM IV."""
    dtes = sorted(chain[dte_col].unique())
    if len(dtes) < 2:
        return 0.0

    front_dte = dtes[0]
    back_dte = dtes[-1]

    front = chain[chain[dte_col] == front_dte]
    back = chain[chain[dte_col] == back_dte]

    front_iv = _nearest_iv(front, strike_col, iv_col, price)
    back_iv = _nearest_iv(back, strike_col, iv_col, price)

    if front_iv is not None and back_iv is not None:
        return back_iv - front_iv
    return 0.0


def _compute_curvature(
    chain: pd.DataFrame, price: float, iv_col: str,
    strike_col: str, atm_iv: float,
) -> float:
    """Smile curvature: average wing IV - ATM IV."""
    low_wing = price * 0.90
    high_wing = price * 1.10

    low_iv = _nearest_iv(chain, strike_col, iv_col, low_wing)
    high_iv = _nearest_iv(chain, strike_col, iv_col, high_wing)

    if low_iv is not None and high_iv is not None:
        return (low_iv + high_iv) / 2 - atm_iv
    return 0.0


def _nearest_iv(
    df: pd.DataFrame, strike_col: str, iv_col: str, target: float,
) -> Optional[float]:
    if df.empty:
        return None
    idx = (df[strike_col] - target).abs().idxmin()
    return float(df.loc[idx, iv_col])


# ── Skew normality scorer ──────────────────────────────────────────────


def score_skew(
    surface: IVSurface,
    historical_skews: Optional[np.ndarray] = None,
) -> SkewScore:
    """Score the current skew relative to normal."""
    skew = surface.skew_25d
    atm = surface.atm_iv if surface.atm_iv > 0 else 0.20

    # Normalise: skew as fraction of ATM IV
    norm_skew = skew / atm if atm > 0 else 0

    # Historical percentile
    if historical_skews is not None and len(historical_skews) > 10:
        pct = float((historical_skews < skew).mean())
    else:
        # Heuristic: typical SPY 25d skew is 3-8% of ATM
        typical_center = 0.05
        typical_std = 0.03
        z = (norm_skew - typical_center) / typical_std if typical_std > 0 else 0
        pct = float(np.clip(0.5 + z * 0.2, 0, 1))

    # Score: -1 (puts cheap) to +1 (puts expensive)
    score = float(np.clip((pct - 0.5) * 2, -1, 1))

    # Signal
    if score > 0.3:
        signal = "sell_puts"
        confidence = min(abs(score), 1.0)
    elif score < -0.3:
        signal = "sell_calls"
        confidence = min(abs(score), 1.0)
    else:
        signal = "neutral"
        confidence = 0.0

    put_wing = atm + skew * 0.5 if skew > 0 else atm
    call_wing = atm - skew * 0.5 if skew > 0 else atm

    return SkewScore(score, pct, put_wing, call_wing, atm, signal, confidence)


# ── Term structure signal ───────────────────────────────────────────────


def term_structure_signal(
    surface: IVSurface,
    contango_threshold: float = 0.10,
) -> TermStructureSignal:
    """Generate signal from term structure slope."""
    slope = surface.term_slope
    front_iv = surface.atm_iv
    back_iv = front_iv + slope

    if front_iv > 0:
        slope_pct = slope / front_iv
    else:
        slope_pct = 0

    # Classify regime
    if slope_pct > contango_threshold:
        regime = "steep_contango"
        signal = "sell_front"
        confidence = min(slope_pct / 0.20, 1.0)
    elif slope_pct > 0.03:
        regime = "mild_contango"
        signal = "sell_front"
        confidence = slope_pct / contango_threshold
    elif slope_pct > -0.03:
        regime = "flat"
        signal = "neutral"
        confidence = 0.0
    else:
        regime = "backwardation"
        signal = "hedge"
        confidence = min(abs(slope_pct) / 0.10, 1.0)

    return TermStructureSignal(
        slope, slope_pct, regime, signal, confidence, front_iv, back_iv,
    )


# ── Butterfly generator ────────────────────────────────────────────────


def generate_butterflies(
    surface: IVSurface,
    wing_widths: Optional[List[float]] = None,
    min_edge: float = 0.01,
) -> List[ButterflySpread]:
    """Generate butterfly spreads where smile kink creates edge."""
    if not surface.points or surface.atm_iv <= 0:
        return []

    wing_widths = wing_widths or [5.0, 10.0]
    price = surface.underlying_price
    atm_iv = surface.atm_iv
    results: List[ButterflySpread] = []

    for width in wing_widths:
        center = round(price / width) * width  # nearest round strike
        lower = center - width
        upper = center + width

        # Get IVs at each strike
        pts_by_strike: Dict[float, float] = {}
        for p in surface.points:
            if p.strike not in pts_by_strike:
                pts_by_strike[p.strike] = p.iv

        # Find nearest strikes
        c_iv = _find_nearest(pts_by_strike, center)
        l_iv = _find_nearest(pts_by_strike, lower)
        u_iv = _find_nearest(pts_by_strike, upper)

        if c_iv is None or l_iv is None or u_iv is None:
            continue

        # Butterfly edge: center IV vs wing average
        wing_avg = (l_iv + u_iv) / 2
        iv_edge = c_iv - wing_avg  # positive = center overpriced → sell butterfly

        if abs(iv_edge) < min_edge:
            continue

        # Estimate credit/debit (simplified: proportional to IV edge)
        est_credit = iv_edge * price * 0.01 * math.sqrt(30 / 365)

        if iv_edge > 0:
            direction = "sell_butterfly"
            expected = est_credit * 0.4  # 40% of max credit as expected PnL
        else:
            direction = "buy_butterfly"
            expected = abs(est_credit) * 0.3

        results.append(ButterflySpread(
            center_strike=center, wing_width=width,
            lower_strike=lower, upper_strike=upper,
            net_credit=est_credit, max_loss=width - abs(est_credit),
            expected_pnl=expected, iv_edge=iv_edge,
            direction=direction,
        ))

    return sorted(results, key=lambda b: -abs(b.iv_edge))


def _find_nearest(pts: Dict[float, float], target: float) -> Optional[float]:
    if not pts:
        return None
    nearest = min(pts.keys(), key=lambda k: abs(k - target))
    if abs(nearest - target) > 20:  # too far
        return None
    return pts[nearest]


# ── Vol surface trader ──────────────────────────────────────────────────


class VolSurfaceTrader:
    """Systematic vol surface analysis and trading."""

    def __init__(
        self,
        chain: pd.DataFrame,
        underlying_price: float,
        regime: str = "neutral",
        historical_skews: Optional[np.ndarray] = None,
        contango_threshold: float = 0.10,
    ) -> None:
        self.chain = chain
        self.underlying_price = underlying_price
        self.regime = regime
        self.historical_skews = historical_skews
        self.contango_threshold = contango_threshold

        self.surface: Optional[IVSurface] = None
        self.signal: Optional[SurfaceSignal] = None

    def analyze(self) -> SurfaceSignal:
        """Build surface and generate combined signal."""
        self.surface = build_surface(self.chain, self.underlying_price)

        skew = score_skew(self.surface, self.historical_skews)
        term = term_structure_signal(self.surface, self.contango_threshold)
        butterflies = generate_butterflies(self.surface)

        # Composite score
        composite = 0.0
        composite += skew.score * 0.4
        if term.regime == "steep_contango":
            composite += 0.3
        elif term.regime == "backwardation":
            composite -= 0.4
        composite += self.surface.smile_curvature * 2  # curvature edge

        # Regime overlay
        if self.regime in ("crash", "high_vol"):
            composite -= 0.3
        elif self.regime == "bull":
            composite += 0.1

        composite = float(np.clip(composite, -1, 1))

        # Action
        if composite > 0.4:
            action = "aggressive_sell"
        elif composite > 0.1:
            action = "normal_sell"
        elif composite > -0.2:
            action = "flat"
        elif composite > -0.5:
            action = "reduce"
        else:
            action = "hedge"

        self.signal = SurfaceSignal(
            action=action, skew=skew, term=term,
            butterflies=butterflies, regime=self.regime,
            composite_score=composite,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return self.signal

    def backtest(
        self,
        trades_df: pd.DataFrame,
        pnl_col: str = "pnl",
        vix_col: str = "vix",
        iv_rank_col: str = "iv_rank",
        regime_col: str = "regime",
    ) -> BacktestResult:
        """Backtest surface-timed vs always-on entries.

        Simulates: only take trades when surface composite > 0 (favorable).
        Uses VIX and IV rank as proxy for surface state.
        """
        df = trades_df.copy()
        n = len(df)
        if n == 0:
            return BacktestResult("vol_surface", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        pnls = df[pnl_col].values.astype(float)

        # Surface proxy signal: high IV rank + contango-like conditions
        iv_ranks = df[iv_rank_col].values if iv_rank_col in df.columns else np.full(n, 50.0)
        vix = df[vix_col].values if vix_col in df.columns else np.full(n, 20.0)
        regimes = df[regime_col].values if regime_col in df.columns else np.full(n, "neutral")

        # Signal: trade when IV rank > 40 AND VIX < 30 AND not crash
        signal_mask = (iv_ranks > 40) & (vix < 30)
        for i, r in enumerate(regimes):
            if str(r).lower() in ("crash", "high_vol"):
                signal_mask[i] = False

        signal_pnls = pnls[signal_mask]

        # Metrics
        wr_all = float((pnls > 0).mean()) if n > 0 else 0
        wr_sig = float((signal_pnls > 0).mean()) if len(signal_pnls) > 0 else 0
        pnl_all = float(pnls.sum())
        pnl_sig = float(signal_pnls.sum())

        cap = 100_000
        eq_all = cap + np.cumsum(pnls)
        eq_sig = cap + np.cumsum(signal_pnls) if len(signal_pnls) > 0 else np.array([cap])

        def _sharpe(rets):
            if len(rets) < 2 or np.std(rets) == 0:
                return 0
            return float(np.mean(rets) / np.std(rets) * np.sqrt(252))

        def _dd(eq):
            if len(eq) < 2:
                return 0
            pk = np.maximum.accumulate(eq)
            return float(np.min((eq - pk) / np.where(pk > 0, pk, 1)))

        sh_all = _sharpe(pnls / cap)
        sh_sig = _sharpe(signal_pnls / cap) if len(signal_pnls) > 0 else 0
        dd_all = _dd(np.concatenate([[cap], eq_all]))
        dd_sig = _dd(np.concatenate([[cap], eq_sig]))

        improvement = (pnl_sig - pnl_all) / abs(pnl_all) * 100 if abs(pnl_all) > 0 else 0

        return BacktestResult(
            strategy="vol_surface_timing",
            n_trades=n, n_signal_trades=int(signal_mask.sum()),
            win_rate_all=wr_all, win_rate_signal=wr_sig,
            pnl_all=pnl_all, pnl_signal=pnl_sig,
            sharpe_all=sh_all, sharpe_signal=sh_sig,
            dd_all=dd_all, dd_signal=dd_sig,
            improvement_pct=improvement,
        )
