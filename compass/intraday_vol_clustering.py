"""
Intraday volatility clustering — detect vol expansion/contraction
transitions within the trading session.

Features:
  - 5-min realised vol blocks
  - EWMA vol smoother (λ=0.94, RiskMetrics-style)
  - Vol clustering autocorrelation
  - Expansion / contraction regime detection
  - Same-day positioning signals

Usage::

    from compass.intraday_vol_clustering import VolClusterEngine
    engine = VolClusterEngine()
    result = engine.analyze(bars_5min)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "intraday_vol_clustering.html"

EWMA_LAMBDA = 0.94  # RiskMetrics decay factor
EXPANSION_THRESHOLD = 1.5  # σ above session mean → expansion
CONTRACTION_THRESHOLD = -0.5  # σ below → contraction


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class VolBlock:
    """One 5-minute vol measurement."""

    block_idx: int
    timestamp: Any
    realised_vol: float  # annualised from 5-min returns
    ewma_vol: float
    zscore: float  # vs session running mean
    regime: str  # "expansion", "contraction", "normal"


@dataclass
class SessionProfile:
    """Full session volatility profile."""

    date: Any
    blocks: List[VolBlock]
    session_vol: float
    peak_vol: float
    trough_vol: float
    vol_autocorrelation: float  # lag-1 autocorrelation of block vols
    n_expansions: int
    n_contractions: int
    expansion_pct: float  # fraction of session in expansion
    eod_regime: str  # regime at 3:30 PM


@dataclass
class ClusterSignal:
    """Trading signal from vol clustering."""

    date: Any
    signal: str  # "sell_premium" (contraction), "avoid" (expansion), "neutral"
    confidence: float
    session_vol: float
    current_regime: str
    autocorrelation: float


@dataclass
class OverlayResult:
    """EXP-880 overlay result."""

    total_trades: int
    sell_prem_trades: int
    avoid_trades: int
    sell_prem_wr: float
    avoid_wr: float
    improvement_pp: float


@dataclass
class ClusterResult:
    """Full analysis result."""

    sessions: List[SessionProfile]
    signals: List[ClusterSignal]
    overlay: Optional[OverlayResult]
    avg_autocorrelation: float
    expansion_predicts_eod: float  # AUC
    standalone_sharpe: float
    n_sessions: int


# ── 5-min realised vol ───────────────────────────────────────────────────


def compute_block_vol(returns_5min: np.ndarray) -> float:
    """Annualised vol from a block of 5-min returns.

    252 trading days × 78 five-min bars per day = 19,656 annual bars.
    """
    if len(returns_5min) < 2:
        return 0.0
    return float(np.std(returns_5min, ddof=1) * math.sqrt(19_656))


def compute_ewma_vol(
    returns: np.ndarray,
    lam: float = EWMA_LAMBDA,
) -> np.ndarray:
    """EWMA variance (RiskMetrics).

    σ²_t = λ σ²_{t-1} + (1-λ) r²_t
    Returns annualised vol series.
    """
    n = len(returns)
    var = np.zeros(n)
    var[0] = returns[0] ** 2
    for i in range(1, n):
        var[i] = lam * var[i - 1] + (1 - lam) * returns[i] ** 2
    return np.sqrt(var) * math.sqrt(19_656)


# ── Autocorrelation ──────────────────────────────────────────────────────


def vol_autocorrelation(vol_series: np.ndarray, lag: int = 1) -> float:
    """Lag-N autocorrelation of vol blocks."""
    if len(vol_series) < lag + 5:
        return 0.0
    x = vol_series[:-lag]
    y = vol_series[lag:]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


# ── Regime detection ─────────────────────────────────────────────────────


def detect_block_regime(
    current_vol: float,
    session_mean: float,
    session_std: float,
) -> str:
    """Classify a vol block as expansion, contraction, or normal."""
    if session_std < 1e-12:
        return "normal"
    z = (current_vol - session_mean) / session_std
    if z > EXPANSION_THRESHOLD:
        return "expansion"
    if z < CONTRACTION_THRESHOLD:
        return "contraction"
    return "normal"


# ── Session analysis ─────────────────────────────────────────────────────


def analyze_session(
    prices_5min: np.ndarray,
    date: Any = None,
) -> SessionProfile:
    """Analyze one trading session of 5-min prices."""
    if len(prices_5min) < 10:
        return SessionProfile(
            date=date, blocks=[], session_vol=0, peak_vol=0, trough_vol=0,
            vol_autocorrelation=0, n_expansions=0, n_contractions=0,
            expansion_pct=0, eod_regime="normal",
        )

    returns = np.diff(prices_5min) / prices_5min[:-1]
    ewma = compute_ewma_vol(returns)

    # Block vols (5 returns per block = 25 min blocks)
    block_size = 5
    block_vols = []
    for i in range(0, len(returns) - block_size + 1, block_size):
        bv = compute_block_vol(returns[i:i + block_size])
        block_vols.append(bv)

    block_vols_arr = np.array(block_vols)
    if len(block_vols_arr) == 0:
        return SessionProfile(
            date=date, blocks=[], session_vol=0, peak_vol=0, trough_vol=0,
            vol_autocorrelation=0, n_expansions=0, n_contractions=0,
            expansion_pct=0, eod_regime="normal",
        )

    session_mean = block_vols_arr.mean()
    session_std = block_vols_arr.std()

    # Build blocks
    blocks: List[VolBlock] = []
    n_exp, n_con = 0, 0
    for i, bv in enumerate(block_vols):
        z = (bv - session_mean) / session_std if session_std > 1e-12 else 0.0
        regime = detect_block_regime(bv, session_mean, session_std)
        if regime == "expansion":
            n_exp += 1
        elif regime == "contraction":
            n_con += 1
        blocks.append(VolBlock(
            block_idx=i, timestamp=i, realised_vol=bv,
            ewma_vol=float(ewma[min(i * block_size, len(ewma) - 1)]),
            zscore=z, regime=regime,
        ))

    acorr = vol_autocorrelation(block_vols_arr)
    total = len(blocks)
    eod_regime = blocks[-1].regime if blocks else "normal"

    return SessionProfile(
        date=date, blocks=blocks,
        session_vol=float(session_mean),
        peak_vol=float(block_vols_arr.max()),
        trough_vol=float(block_vols_arr.min()),
        vol_autocorrelation=acorr,
        n_expansions=n_exp, n_contractions=n_con,
        expansion_pct=n_exp / total if total > 0 else 0,
        eod_regime=eod_regime,
    )


# ── Signal generation ────────────────────────────────────────────────────


def generate_session_signal(session: SessionProfile) -> ClusterSignal:
    """Generate signal from session vol profile."""
    if not session.blocks:
        return ClusterSignal(session.date, "neutral", 0.0, 0.0, "normal", 0.0)

    # Use mid-session regime as signal (blocks at ~50% through session)
    mid_idx = len(session.blocks) // 2
    current = session.blocks[mid_idx].regime

    acorr = session.vol_autocorrelation
    confidence = min(abs(acorr), 1.0) * 0.7  # higher autocorr → more confident

    if current == "contraction" and acorr > 0.3:
        signal = "sell_premium"
        confidence += 0.2
    elif current == "expansion" and acorr > 0.3:
        signal = "avoid"
        confidence += 0.15
    else:
        signal = "neutral"

    return ClusterSignal(
        date=session.date, signal=signal, confidence=min(confidence, 1.0),
        session_vol=session.session_vol, current_regime=current,
        autocorrelation=acorr,
    )


# ── Multi-session from daily data ────────────────────────────────────────


def simulate_sessions_from_daily(
    daily_df: pd.DataFrame,
    n_bars_per_session: int = 78,
    seed: int = 1320,
) -> List[SessionProfile]:
    """Generate synthetic 5-min sessions from daily OHLCV.

    Uses daily range to calibrate intraday vol, then simulates
    5-min price paths with vol clustering built in.
    """
    rng = np.random.RandomState(seed)
    sessions: List[SessionProfile] = []

    for i, (idx, row) in enumerate(daily_df.iterrows()):
        o = float(row.get("open", row.get("close", 100)))
        h = float(row.get("high", o * 1.01))
        l = float(row.get("low", o * 0.99))
        c = float(row.get("close", o))

        daily_range = (h - l) / o
        intraday_vol = daily_range / math.sqrt(n_bars_per_session) * 3

        # Generate clustered 5-min returns
        # Cluster: vol state persists with 0.7 probability
        vol_state = "normal"
        prices = [o]
        for j in range(n_bars_per_session):
            # Regime transition
            roll = rng.random()
            if vol_state == "normal":
                if roll < 0.05:
                    vol_state = "high"
                elif roll < 0.08:
                    vol_state = "low"
            elif vol_state == "high":
                if roll < 0.2:
                    vol_state = "normal"
            elif vol_state == "low":
                if roll < 0.15:
                    vol_state = "normal"

            mult = {"normal": 1.0, "high": 2.5, "low": 0.4}[vol_state]
            ret = rng.normal(0, intraday_vol * mult)
            prices.append(prices[-1] * (1 + ret))

        # Drift toward close
        drift = (c - prices[-1]) / prices[-1]
        prices = np.array(prices) * (1 + np.linspace(0, drift, len(prices)))

        session = analyze_session(prices, date=idx)
        sessions.append(session)

    return sessions


# ── Expansion → EOD vol prediction ───────────────────────────────────────


def expansion_predicts_eod_auc(sessions: List[SessionProfile]) -> float:
    """Does mid-session expansion predict higher-than-average EOD vol?"""
    if len(sessions) < 20:
        return 0.5

    median_vol = np.median([s.session_vol for s in sessions])
    predictor = []
    actual = []
    for s in sessions:
        predictor.append(s.expansion_pct)
        actual.append(1 if s.session_vol > median_vol else 0)

    pred = np.array(predictor)
    act = np.array(actual)
    pos = pred[act == 1]
    neg = pred[act == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    concordant = sum((neg < p).sum() for p in pos)
    return float(concordant / (len(pos) * len(neg)))


# ── Overlay on trades ────────────────────────────────────────────────────


def overlay_on_trades(
    signals: List[ClusterSignal],
    trades: pd.DataFrame,
) -> OverlayResult:
    if not signals or trades.empty:
        return OverlayResult(0, 0, 0, 0, 0, 0)

    sig_map = {str(s.date)[:10]: s for s in signals if s.date is not None}
    sp_wins, sp_total = 0, 0
    av_wins, av_total = 0, 0

    for _, row in trades.iterrows():
        dt = str(row.get("entry_date", ""))[:10]
        win = int(row.get("win", 0))
        sig = sig_map.get(dt)

        if sig and sig.signal == "sell_premium":
            sp_total += 1
            sp_wins += win
        else:
            av_total += 1
            av_wins += win

    sp_wr = sp_wins / sp_total if sp_total > 0 else 0
    av_wr = av_wins / av_total if av_total > 0 else 0

    return OverlayResult(
        total_trades=sp_total + av_total,
        sell_prem_trades=sp_total, avoid_trades=av_total,
        sell_prem_wr=sp_wr, avoid_wr=av_wr,
        improvement_pp=(sp_wr - av_wr) * 100,
    )


# ── Standalone Sharpe ────────────────────────────────────────────────────


def standalone_sharpe(
    signals: List[ClusterSignal],
    daily_returns: pd.Series,
) -> float:
    if not signals or daily_returns.empty:
        return 0.0

    pnls = []
    for sig in signals:
        dt = str(sig.date)[:10] if sig.date is not None else None
        if dt is None:
            continue
        try:
            idx = daily_returns.index.get_loc(pd.Timestamp(dt))
            if idx + 1 >= len(daily_returns):
                continue
            nxt = float(daily_returns.iloc[idx + 1])
        except (KeyError, TypeError):
            continue

        if sig.signal == "sell_premium":
            pnls.append(nxt)  # premium seller benefits from calm
        elif sig.signal == "avoid":
            pnls.append(-nxt * 0.5)

    if len(pnls) < 10:
        return 0.0
    arr = np.array(pnls)
    mu, std = arr.mean(), arr.std(ddof=1)
    return float(mu / std * math.sqrt(252)) if std > 1e-12 else 0.0


# ── Core engine ──────────────────────────────────────────────────────────


class VolClusterEngine:
    """Intraday volatility clustering engine."""

    def __init__(self, ewma_lambda: float = EWMA_LAMBDA):
        self.ewma_lambda = ewma_lambda

    def analyze_bars(self, prices_5min: np.ndarray, date: Any = None) -> SessionProfile:
        return analyze_session(prices_5min, date)

    def analyze_daily(
        self,
        daily_df: pd.DataFrame,
        trades: Optional[pd.DataFrame] = None,
    ) -> ClusterResult:
        """Analyze multiple sessions from daily OHLCV data."""
        sessions = simulate_sessions_from_daily(daily_df)
        signals = [generate_session_signal(s) for s in sessions]

        avg_ac = float(np.mean([s.vol_autocorrelation for s in sessions])) if sessions else 0
        auc = expansion_predicts_eod_auc(sessions)

        daily_ret = daily_df["close"].pct_change().fillna(0) if "close" in daily_df.columns else pd.Series(dtype=float)
        sh = standalone_sharpe(signals, daily_ret)

        overlay = None
        if trades is not None and not trades.empty:
            overlay = overlay_on_trades(signals, trades)

        return ClusterResult(
            sessions=sessions, signals=signals, overlay=overlay,
            avg_autocorrelation=avg_ac,
            expansion_predicts_eod=auc,
            standalone_sharpe=sh,
            n_sessions=len(sessions),
        )

    @staticmethod
    def generate_report(result: ClusterResult, output_path: Path = DEFAULT_OUTPUT) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path


def _fr(v): return f"{v:.3f}"
def _fp(v): return f"{v:.1f}%"


def _build_html(r: ClusterResult) -> str:
    ov = ""
    if r.overlay:
        o = r.overlay
        ov = f"<h2>EXP-880 Overlay</h2><p>Sell premium: {o.sell_prem_wr:.1%} WR ({o.sell_prem_trades} trades) vs Avoid: {o.avoid_wr:.1%} → {o.improvement_pp:+.1f}pp</p>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Vol Clustering</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}h1,h2{{color:#58a6ff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:20px 0}}.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}</style></head><body>
<h1>Intraday Vol Clustering</h1>
<div class="cards">
<div class="c"><div class="l">Avg Autocorrelation</div><div class="v">{_fr(r.avg_autocorrelation)}</div></div>
<div class="c"><div class="l">Expansion→EOD AUC</div><div class="v">{_fr(r.expansion_predicts_eod)}</div></div>
<div class="c"><div class="l">Standalone Sharpe</div><div class="v">{_fr(r.standalone_sharpe)}</div></div>
<div class="c"><div class="l">Sessions</div><div class="v">{r.n_sessions}</div></div>
</div>{ov}</body></html>"""
