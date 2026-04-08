"""
Microstructure alpha scanner — liquidity regime detection and signals.

Computes 8 microstructure metrics from daily OHLCV, detects liquidity
regime shifts, generates signals for credit spread timing.

Metrics:
  1. Amihud illiquidity
  2. Roll spread estimator
  3. Kyle lambda proxy
  4. Corwin-Schultz high-low spread
  5. Volume-return correlation (toxicity)
  6. Liquidity ratio
  7. Spread z-score
  8. Liquidity regime classification

Usage::

    from compass.microstructure_alpha import MicrostructureScanner
    scanner = MicrostructureScanner(ohlcv_df)
    result = scanner.analyze()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "microstructure_alpha.html"

REGIME_LABELS = ["tight", "normal", "wide", "crisis"]


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class MicroFeatures:
    """Daily microstructure features for one observation."""

    date: Any
    amihud: float = 0.0
    roll_spread: float = 0.0
    kyle_lambda: float = 0.0
    corwin_schultz: float = 0.0
    volume_return_corr: float = 0.0
    liquidity_ratio: float = 0.0
    spread_zscore: float = 0.0
    regime: str = "normal"

    def to_dict(self) -> Dict[str, float]:
        return {
            "amihud": self.amihud, "roll_spread": self.roll_spread,
            "kyle_lambda": self.kyle_lambda, "corwin_schultz": self.corwin_schultz,
            "volume_return_corr": self.volume_return_corr,
            "liquidity_ratio": self.liquidity_ratio,
            "spread_zscore": self.spread_zscore,
        }


@dataclass
class LiquiditySignal:
    """Trading signal from liquidity regime."""

    date: Any
    regime: str
    signal: str  # "enter" (tight/normal), "avoid" (wide), "exit_all" (crisis)
    confidence: float
    amihud_z: float
    spread_z: float


@dataclass
class OverlayResult:
    """Result of filtering EXP-880 trades by liquidity regime."""

    total_trades: int
    enter_trades: int
    avoid_trades: int
    enter_win_rate: float
    avoid_win_rate: float
    improvement_pp: float


@dataclass
class ScannerResult:
    """Full microstructure analysis result."""

    features: pd.DataFrame
    signals: List[LiquiditySignal]
    regime_counts: Dict[str, int]
    overlay: Optional[OverlayResult]
    standalone_sharpe: float
    volatility_prediction_auc: float
    n_observations: int


# ── Metric computations ──────────────────────────────────────────────────


def compute_amihud(
    returns: pd.Series,
    dollar_volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Amihud illiquidity: rolling mean of |return| / dollar_volume."""
    ratio = returns.abs() / dollar_volume.replace(0, np.nan)
    return ratio.rolling(window, min_periods=5).mean().fillna(0) * 1e6


def compute_roll_spread(prices: pd.Series, window: int = 20) -> pd.Series:
    """Roll (1984) effective spread estimator.

    spread = 2 * sqrt(-cov(Δp_t, Δp_{t-1})) when cov < 0, else 0.
    """
    dp = prices.diff()
    result = pd.Series(0.0, index=prices.index)
    for i in range(window, len(dp)):
        chunk = dp.iloc[i - window:i]
        chunk_lag = dp.iloc[i - window - 1:i - 1]
        if len(chunk) < 5 or len(chunk_lag) < 5:
            continue
        # Align lengths
        n = min(len(chunk), len(chunk_lag))
        cov = np.cov(chunk.values[-n:], chunk_lag.values[-n:])[0, 1]
        result.iloc[i] = 2 * math.sqrt(-cov) if cov < 0 else 0.0
    return result


def compute_kyle_lambda(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Kyle lambda proxy: regression slope of |Δprice| on volume."""
    abs_ret = returns.abs()
    result = pd.Series(0.0, index=returns.index)
    for i in range(window, len(returns)):
        y = abs_ret.iloc[i - window:i].values
        x = volume.iloc[i - window:i].values
        if x.std() < 1e-12:
            continue
        # Simple regression slope
        slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0
        result.iloc[i] = abs(slope) * 1e6
    return result


def compute_corwin_schultz(
    high: pd.Series,
    low: pd.Series,
    window: int = 1,
) -> pd.Series:
    """Corwin-Schultz (2012) high-low spread estimator.

    Uses ratio of single-day and two-day high-low ranges.
    """
    log_hl = np.log(high / low.replace(0, np.nan))
    log_hl_sq = log_hl ** 2

    # Two-day range
    high_2d = high.rolling(2).max()
    low_2d = low.rolling(2).min()
    log_hl_2d_sq = np.log(high_2d / low_2d.replace(0, np.nan)) ** 2

    beta = log_hl_sq.rolling(window).mean()
    gamma = log_hl_2d_sq.rolling(window).mean()

    # alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2)) - sqrt(gamma / (3 - 2*sqrt(2)))
    k = 3 - 2 * math.sqrt(2)
    alpha_num = np.sqrt(2 * beta.clip(lower=0)) - np.sqrt(beta.clip(lower=0))
    alpha = alpha_num / k - np.sqrt(gamma.clip(lower=0) / k)

    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0).fillna(0)


def compute_volume_return_correlation(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling correlation of signed volume × return (order flow toxicity)."""
    signed_vol = volume * np.sign(returns)
    return signed_vol.rolling(window, min_periods=5).corr(returns).fillna(0)


def compute_liquidity_ratio(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Volume / |return| — inverse Amihud, higher = more liquid."""
    ratio = volume / returns.abs().replace(0, np.nan)
    return ratio.rolling(window, min_periods=5).mean().fillna(0) / 1e6


def compute_spread_zscore(
    spread: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Z-score of current spread vs rolling average."""
    mu = spread.rolling(window, min_periods=5).mean()
    sigma = spread.rolling(window, min_periods=5).std().replace(0, 1)
    return ((spread - mu) / sigma).fillna(0)


# ── Regime classification ────────────────────────────────────────────────


def classify_regime(amihud_z: float, spread_z: float) -> str:
    """Classify liquidity regime from z-scores."""
    composite = (amihud_z + spread_z) / 2
    if composite > 2.0:
        return "crisis"
    if composite > 1.0:
        return "wide"
    if composite < -0.5:
        return "tight"
    return "normal"


def regime_to_signal(regime: str) -> str:
    """Map regime to trading signal."""
    if regime in ("tight", "normal"):
        return "enter"
    if regime == "wide":
        return "avoid"
    return "exit_all"


# ── Feature engine ───────────────────────────────────────────────────────


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 8 microstructure features from OHLCV DataFrame.

    Expected columns: open, high, low, close, volume.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].astype(float)
    returns = close.pct_change().fillna(0)
    dollar_vol = close * volume

    amihud = compute_amihud(returns, dollar_vol)
    roll = compute_roll_spread(close)
    kyle = compute_kyle_lambda(returns, volume)
    cs = compute_corwin_schultz(high, low)
    vrc = compute_volume_return_correlation(returns, volume)
    liq = compute_liquidity_ratio(returns, volume)

    # Composite spread: average of Roll and Corwin-Schultz (both estimate bid-ask)
    composite_spread = (roll / close.replace(0, 1) * 10_000 + cs * 10_000) / 2
    spread_z = compute_spread_zscore(composite_spread)
    amihud_z = compute_spread_zscore(amihud)  # z-score of Amihud

    # Regime
    regimes = pd.Series("normal", index=df.index)
    for i in range(len(df)):
        regimes.iloc[i] = classify_regime(
            float(amihud_z.iloc[i]),
            float(spread_z.iloc[i]),
        )

    return pd.DataFrame({
        "amihud": amihud,
        "roll_spread": roll,
        "kyle_lambda": kyle,
        "corwin_schultz": cs,
        "volume_return_corr": vrc,
        "liquidity_ratio": liq,
        "spread_zscore": spread_z,
        "amihud_zscore": amihud_z,
        "regime": regimes,
    }, index=df.index)


# ── Signal generation ────────────────────────────────────────────────────


def generate_signals(features: pd.DataFrame) -> List[LiquiditySignal]:
    """Generate trading signals from feature DataFrame."""
    signals = []
    for i, (idx, row) in enumerate(features.iterrows()):
        regime = str(row.get("regime", "normal"))
        sig = regime_to_signal(regime)
        az = float(row.get("amihud_zscore", 0))
        sz = float(row.get("spread_zscore", 0))
        conf = 1.0 - min(abs(az + sz) / 4, 1.0) if sig == "enter" else min((abs(az) + abs(sz)) / 4, 1.0)
        signals.append(LiquiditySignal(
            date=idx, regime=regime, signal=sig,
            confidence=conf, amihud_z=az, spread_z=sz,
        ))
    return signals


# ── Standalone backtest ──────────────────────────────────────────────────


def standalone_sharpe(
    signals: List[LiquiditySignal],
    returns: pd.Series,
) -> float:
    """Trade SPY based on liquidity signal: enter=long, avoid=flat."""
    pnls = []
    for sig in signals:
        dt = sig.date
        try:
            idx = returns.index.get_loc(dt)
            if idx + 1 >= len(returns):
                continue
            next_ret = float(returns.iloc[idx + 1])
        except (KeyError, TypeError):
            continue

        if sig.signal == "enter":
            pnls.append(next_ret)
        elif sig.signal == "avoid":
            pnls.append(-next_ret * 0.3)  # small contrarian bet
        # exit_all → flat

    if len(pnls) < 10:
        return 0.0
    arr = np.array(pnls)
    mu, std = arr.mean(), arr.std(ddof=1)
    return float(mu / std * math.sqrt(252)) if std > 1e-12 else 0.0


# ── Volatility prediction AUC ────────────────────────────────────────────


def volatility_prediction_auc(
    features: pd.DataFrame,
    returns: pd.Series,
    forward_window: int = 5,
) -> float:
    """AUC of predicting next-5-day vol regime from microstructure."""
    if len(features) < forward_window + 20:
        return 0.5

    # Forward volatility: realized vol over next 5 days
    fwd_vol = returns.rolling(forward_window).std().shift(-forward_window)
    median_vol = fwd_vol.median()

    # Predictor: current spread z-score (higher → more vol expected)
    predictor = features["spread_zscore"].values
    actual = (fwd_vol > median_vol).astype(int).values

    # Drop NaN
    mask = ~(np.isnan(predictor) | np.isnan(actual))
    pred = predictor[mask]
    act = actual[mask]

    if len(pred) < 20 or act.sum() == 0 or (1 - act).sum() == 0:
        return 0.5

    # Concordance AUC
    pos = pred[act == 1]
    neg = pred[act == 0]
    concordant = sum((neg < p).sum() for p in pos)
    return float(concordant / (len(pos) * len(neg)))


# ── Overlay on EXP-880 trades ────────────────────────────────────────────


def overlay_on_trades(
    signals: List[LiquiditySignal],
    trades: pd.DataFrame,
) -> OverlayResult:
    """Test if liquidity-filtered entries improve EXP-880 WR."""
    if not signals or trades.empty:
        return OverlayResult(0, 0, 0, 0, 0, 0)

    sig_map = {str(s.date)[:10]: s for s in signals}
    enter_wins, enter_total = 0, 0
    avoid_wins, avoid_total = 0, 0

    for _, row in trades.iterrows():
        dt = str(row.get("entry_date", ""))[:10]
        win = int(row.get("win", 0))
        sig = sig_map.get(dt)

        if sig and sig.signal == "enter":
            enter_total += 1
            enter_wins += win
        else:
            avoid_total += 1
            avoid_wins += win

    ewr = enter_wins / enter_total if enter_total > 0 else 0
    awr = avoid_wins / avoid_total if avoid_total > 0 else 0

    return OverlayResult(
        total_trades=enter_total + avoid_total,
        enter_trades=enter_total, avoid_trades=avoid_total,
        enter_win_rate=ewr, avoid_win_rate=awr,
        improvement_pp=(ewr - awr) * 100,
    )


# ── Core scanner ─────────────────────────────────────────────────────────


class MicrostructureScanner:
    """Microstructure alpha scanner."""

    def __init__(self, ohlcv: pd.DataFrame):
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(c.lower() for c in ohlcv.columns)):
            raise ValueError(f"Need columns: {required}")
        self.df = ohlcv.copy()
        # Normalize column names
        self.df.columns = [c.lower() for c in self.df.columns]

    def analyze(self, trades: Optional[pd.DataFrame] = None) -> ScannerResult:
        features = compute_all_features(self.df)
        signals = generate_signals(features)

        regime_counts = features["regime"].value_counts().to_dict()
        returns = self.df["close"].pct_change().fillna(0)
        sh = standalone_sharpe(signals, returns)
        auc = volatility_prediction_auc(features, returns)

        overlay = None
        if trades is not None and not trades.empty:
            overlay = overlay_on_trades(signals, trades)

        return ScannerResult(
            features=features, signals=signals,
            regime_counts=regime_counts, overlay=overlay,
            standalone_sharpe=sh,
            volatility_prediction_auc=auc,
            n_observations=len(self.df),
        )

    @staticmethod
    def generate_report(result: ScannerResult, output_path: Path = DEFAULT_OUTPUT) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path


# ── HTML ─────────────────────────────────────────────────────────────────


def _fr(v): return f"{v:.3f}"


def _build_html(r: ScannerResult) -> str:
    regime_rows = "".join(
        f"<tr><td style='text-align:left'>{reg}</td><td>{cnt}</td>"
        f"<td>{cnt / r.n_observations:.1%}</td></tr>"
        for reg, cnt in sorted(r.regime_counts.items())
    )

    overlay_html = ""
    if r.overlay:
        o = r.overlay
        overlay_html = f"""<h2>EXP-880 Overlay</h2>
        <p>Enter (tight/normal): {o.enter_win_rate:.1%} WR ({o.enter_trades} trades)<br>
        Avoid (wide/crisis): {o.avoid_win_rate:.1%} WR ({o.avoid_trades} trades)<br>
        Improvement: {o.improvement_pp:+.1f}pp</p>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Microstructure Alpha</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:900px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2{{color:#58a6ff}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}th{{color:#8b949e;background:#161b22}}
</style></head><body>
<h1>Microstructure Alpha Scanner</h1>
<div class="cards">
<div class="c"><div class="l">Standalone Sharpe</div><div class="v">{_fr(r.standalone_sharpe)}</div></div>
<div class="c"><div class="l">Vol Prediction AUC</div><div class="v">{_fr(r.volatility_prediction_auc)}</div></div>
<div class="c"><div class="l">Observations</div><div class="v">{r.n_observations}</div></div>
</div>
<h2>Liquidity Regimes</h2>
<table><tr><th style="text-align:left">Regime</th><th>Count</th><th>Pct</th></tr>{regime_rows}</table>
{overlay_html}
</body></html>"""
