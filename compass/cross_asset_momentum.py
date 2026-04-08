"""
Cross-asset momentum signal engine.

Extracts momentum features from gold, oil, copper, TLT, HYG, DXY.
Tests lead-lag relationships with SPY. Generates composite positioning
signals (bullish/bearish/neutral) with confidence scores.

Can overlay on EXP-880 credit spread entry timing.

Usage::

    from compass.cross_asset_momentum import CrossAssetMomentum
    cam = CrossAssetMomentum(prices_df)
    result = cam.analyze()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "cross_asset_momentum.html"
TRADING_DAYS = 252

ASSETS = {
    "GLD": {"name": "Gold", "spy_relation": "inverse", "expected_lead": 1},
    "USO": {"name": "Oil", "spy_relation": "positive", "expected_lead": 1},
    "CPER": {"name": "Copper", "spy_relation": "positive", "expected_lead": 2},
    "TLT": {"name": "Treasuries", "spy_relation": "inverse", "expected_lead": 1},
    "HYG": {"name": "High Yield", "spy_relation": "positive", "expected_lead": 1},
    "UUP": {"name": "Dollar", "spy_relation": "inverse", "expected_lead": 1},
}


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class MomentumFeature:
    """Momentum feature for one asset."""

    asset: str
    momentum_1d: float = 0.0
    momentum_3d: float = 0.0
    momentum_5d: float = 0.0
    momentum_10d: float = 0.0
    zscore_20d: float = 0.0
    trend_strength: float = 0.0  # slope of 20d regression, annualised


@dataclass
class LeadLagResult:
    """Lead-lag relationship between an asset and SPY."""

    asset: str
    optimal_lag: int  # positive = asset leads SPY
    correlation_at_lag: float
    is_significant: bool
    direction: str  # "positive" or "inverse"


@dataclass
class PositioningSignal:
    """Composite positioning signal."""

    timestamp: Any
    signal: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0-1
    n_bullish: int
    n_bearish: int
    n_neutral: int
    contributing_assets: Dict[str, str]  # asset → its signal


@dataclass
class OverlayResult:
    """Result of overlaying signal on EXP-880 trades."""

    total_trades: int
    signal_confirmed_trades: int
    confirmed_win_rate: float
    unconfirmed_win_rate: float
    improvement_pp: float  # percentage point improvement
    signal_hit_rate: float


@dataclass
class AnalysisResult:
    """Full cross-asset momentum analysis."""

    features: Dict[str, MomentumFeature]
    lead_lags: List[LeadLagResult]
    signals: List[PositioningSignal]
    overlay: Optional[OverlayResult]
    n_significant_leads: int
    composite_sharpe: float
    n_observations: int


# ── Momentum feature computation ─────────────────────────────────────────


def compute_returns(prices: pd.Series) -> pd.Series:
    """Daily percentage returns."""
    return prices.pct_change().fillna(0)


def compute_momentum(returns: pd.Series, window: int) -> pd.Series:
    """Cumulative return over window."""
    return returns.rolling(window).sum()


def compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return ((series - mu) / sigma.replace(0, 1)).fillna(0)


def compute_trend_strength(prices: pd.Series, window: int = 20) -> pd.Series:
    """Rolling linear regression slope, annualised."""
    result = pd.Series(0.0, index=prices.index)
    vals = prices.values
    for i in range(window, len(vals)):
        chunk = vals[i - window:i]
        x = np.arange(window, dtype=float)
        if np.std(chunk) < 1e-12:
            continue
        slope = np.polyfit(x, chunk, 1)[0]
        result.iloc[i] = slope / max(abs(chunk.mean()), 1e-6) * TRADING_DAYS * 100
    return result


def extract_features(
    prices: pd.DataFrame,
    asset: str,
) -> pd.DataFrame:
    """Extract all momentum features for one asset."""
    if asset not in prices.columns:
        return pd.DataFrame()
    p = prices[asset].dropna()
    ret = compute_returns(p)
    return pd.DataFrame({
        f"{asset}_mom_1d": compute_momentum(ret, 1),
        f"{asset}_mom_3d": compute_momentum(ret, 3),
        f"{asset}_mom_5d": compute_momentum(ret, 5),
        f"{asset}_mom_10d": compute_momentum(ret, 10),
        f"{asset}_zscore_20d": compute_zscore(p),
        f"{asset}_trend": compute_trend_strength(p),
    }, index=p.index)


def extract_all_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Extract features for all available assets."""
    frames = []
    for asset in ASSETS:
        if asset in prices.columns:
            frames.append(extract_features(prices, asset))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).fillna(0)


def latest_features(prices: pd.DataFrame) -> Dict[str, MomentumFeature]:
    """Get latest momentum features for each asset."""
    features = {}
    for asset in ASSETS:
        if asset not in prices.columns:
            continue
        p = prices[asset].dropna()
        if len(p) < 20:
            continue
        ret = compute_returns(p)
        features[asset] = MomentumFeature(
            asset=asset,
            momentum_1d=float(ret.iloc[-1]) if len(ret) > 0 else 0.0,
            momentum_3d=float(ret.iloc[-3:].sum()) if len(ret) >= 3 else 0.0,
            momentum_5d=float(ret.iloc[-5:].sum()) if len(ret) >= 5 else 0.0,
            momentum_10d=float(ret.iloc[-10:].sum()) if len(ret) >= 10 else 0.0,
            zscore_20d=float(compute_zscore(p).iloc[-1]),
            trend_strength=float(compute_trend_strength(p).iloc[-1]),
        )
    return features


# ── Lead-lag detection ───────────────────────────────────────────────────


def detect_lead_lag(
    asset_returns: pd.Series,
    spy_returns: pd.Series,
    max_lag: int = 5,
) -> LeadLagResult:
    """Find optimal lead-lag between asset and SPY."""
    aligned = pd.concat([asset_returns.rename("asset"), spy_returns.rename("spy")], axis=1).dropna()
    if len(aligned) < 30:
        return LeadLagResult(asset_returns.name or "", 0, 0.0, False, "positive")

    best_lag = 0
    best_corr = 0.0
    asset_name = str(asset_returns.name or "")

    for lag in range(0, max_lag + 1):
        if lag == 0:
            x, y = aligned["asset"].values, aligned["spy"].values
        else:
            x = aligned["asset"].values[:-lag]
            y = aligned["spy"].values[lag:]

        if len(x) < 20:
            continue
        c = np.corrcoef(x, y)[0, 1]
        if not np.isnan(c) and abs(c) > abs(best_corr):
            best_corr = float(c)
            best_lag = lag

    # Significance: |corr| > 2/sqrt(n)
    n = len(aligned) - best_lag
    threshold = 2.0 / math.sqrt(max(n, 1))
    is_sig = abs(best_corr) > threshold

    meta = ASSETS.get(asset_name, {})
    expected_dir = meta.get("spy_relation", "positive")

    return LeadLagResult(
        asset=asset_name,
        optimal_lag=best_lag,
        correlation_at_lag=best_corr,
        is_significant=is_sig,
        direction="positive" if best_corr > 0 else "inverse",
    )


# ── Positioning signal ───────────────────────────────────────────────────


def generate_signal(
    features: Dict[str, MomentumFeature],
    lead_lags: List[LeadLagResult],
) -> PositioningSignal:
    """Generate composite positioning signal from cross-asset features."""
    asset_signals: Dict[str, str] = {}
    n_bull, n_bear, n_neut = 0, 0, 0

    for ll in lead_lags:
        if not ll.is_significant:
            continue
        feat = features.get(ll.asset)
        if feat is None:
            continue

        # Use 3d momentum as primary signal
        mom = feat.momentum_3d
        meta = ASSETS.get(ll.asset, {})
        relation = meta.get("spy_relation", "positive")

        # Positive relation: asset up → SPY up (bullish for put credit spreads)
        # Inverse relation: asset up → SPY down (bearish for put credit spreads)
        if relation == "positive":
            if mom > 0.005:
                asset_signals[ll.asset] = "bullish"
                n_bull += 1
            elif mom < -0.005:
                asset_signals[ll.asset] = "bearish"
                n_bear += 1
            else:
                asset_signals[ll.asset] = "neutral"
                n_neut += 1
        else:  # inverse
            if mom > 0.005:
                asset_signals[ll.asset] = "bearish"
                n_bear += 1
            elif mom < -0.005:
                asset_signals[ll.asset] = "bullish"
                n_bull += 1
            else:
                asset_signals[ll.asset] = "neutral"
                n_neut += 1

    total = n_bull + n_bear + n_neut
    if total == 0:
        return PositioningSignal(None, "neutral", 0.0, 0, 0, 0, {})

    if n_bull > n_bear:
        signal = "bullish"
        confidence = n_bull / total
    elif n_bear > n_bull:
        signal = "bearish"
        confidence = n_bear / total
    else:
        signal = "neutral"
        confidence = 0.5

    return PositioningSignal(
        timestamp=None, signal=signal, confidence=confidence,
        n_bullish=n_bull, n_bearish=n_bear, n_neutral=n_neut,
        contributing_assets=asset_signals,
    )


def generate_signal_series(
    prices: pd.DataFrame,
    spy_col: str = "SPY",
    lookback: int = 60,
) -> List[PositioningSignal]:
    """Generate signal at each date in the price history."""
    if spy_col not in prices.columns:
        return []

    spy_ret = compute_returns(prices[spy_col])
    asset_rets = {a: compute_returns(prices[a]) for a in ASSETS if a in prices.columns}

    # Compute lead-lags on training window
    lead_lags = []
    for asset, aret in asset_rets.items():
        ll = detect_lead_lag(aret, spy_ret)
        lead_lags.append(ll)

    # Generate signals at each point
    signals: List[PositioningSignal] = []
    dates = prices.index[lookback:]
    for i, dt in enumerate(dates):
        idx = lookback + i
        window = prices.iloc[max(0, idx - 20):idx + 1]
        feats = latest_features(window)
        sig = generate_signal(feats, lead_lags)
        sig.timestamp = dt
        signals.append(sig)

    return signals


# ── Overlay on EXP-880 trades ────────────────────────────────────────────


def overlay_on_trades(
    signals: List[PositioningSignal],
    trades: pd.DataFrame,
) -> OverlayResult:
    """Test if signal-confirmed trades have higher win rate."""
    if not signals or trades.empty:
        return OverlayResult(0, 0, 0, 0, 0, 0)

    sig_by_date = {}
    for s in signals:
        if s.timestamp is not None:
            sig_by_date[str(s.timestamp)[:10]] = s

    confirmed_wins, confirmed_total = 0, 0
    unconfirmed_wins, unconfirmed_total = 0, 0

    for _, row in trades.iterrows():
        dt = str(row.get("entry_date", ""))[:10]
        win = int(row.get("win", 0))
        sig = sig_by_date.get(dt)

        if sig and sig.signal == "bullish" and sig.confidence > 0.5:
            confirmed_total += 1
            confirmed_wins += win
        else:
            unconfirmed_total += 1
            unconfirmed_wins += win

    c_wr = confirmed_wins / confirmed_total if confirmed_total > 0 else 0.0
    u_wr = unconfirmed_wins / unconfirmed_total if unconfirmed_total > 0 else 0.0

    return OverlayResult(
        total_trades=confirmed_total + unconfirmed_total,
        signal_confirmed_trades=confirmed_total,
        confirmed_win_rate=c_wr,
        unconfirmed_win_rate=u_wr,
        improvement_pp=(c_wr - u_wr) * 100,
        signal_hit_rate=confirmed_total / max(confirmed_total + unconfirmed_total, 1),
    )


# ── Composite signal Sharpe ──────────────────────────────────────────────


def composite_signal_sharpe(signals: List[PositioningSignal], spy_returns: pd.Series) -> float:
    """Sharpe of trading SPY based on composite signal."""
    if not signals or spy_returns.empty:
        return 0.0

    pnls = []
    for sig in signals:
        dt = str(sig.timestamp)[:10] if sig.timestamp else None
        if dt is None:
            continue
        # Find next-day SPY return
        try:
            idx = spy_returns.index.get_loc(pd.Timestamp(dt))
            if idx + 1 < len(spy_returns):
                next_ret = float(spy_returns.iloc[idx + 1])
            else:
                continue
        except (KeyError, TypeError):
            continue

        if sig.signal == "bullish":
            pnls.append(next_ret)
        elif sig.signal == "bearish":
            pnls.append(-next_ret)
        # neutral → skip

    if len(pnls) < 10:
        return 0.0
    arr = np.array(pnls)
    mu = arr.mean()
    std = arr.std(ddof=1)
    return float(mu / std * math.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0


# ── Core engine ──────────────────────────────────────────────────────────


class CrossAssetMomentum:
    """Cross-asset momentum signal engine."""

    def __init__(self, prices: pd.DataFrame, spy_col: str = "SPY"):
        if prices.empty:
            raise ValueError("prices must not be empty")
        self.prices = prices
        self.spy_col = spy_col

    def analyze(self, trades: Optional[pd.DataFrame] = None) -> AnalysisResult:
        """Run full cross-asset momentum analysis."""
        spy_ret = compute_returns(self.prices[self.spy_col]) if self.spy_col in self.prices.columns else pd.Series(dtype=float)

        # Features
        features = latest_features(self.prices)

        # Lead-lag
        lead_lags = []
        for asset in ASSETS:
            if asset not in self.prices.columns:
                continue
            aret = compute_returns(self.prices[asset])
            ll = detect_lead_lag(aret, spy_ret)
            lead_lags.append(ll)

        n_sig = sum(1 for ll in lead_lags if ll.is_significant)

        # Signals
        signals = generate_signal_series(self.prices, self.spy_col)

        # Composite Sharpe
        comp_sharpe = composite_signal_sharpe(signals, spy_ret)

        # Overlay
        overlay = None
        if trades is not None and not trades.empty:
            overlay = overlay_on_trades(signals, trades)

        return AnalysisResult(
            features=features,
            lead_lags=lead_lags,
            signals=signals,
            overlay=overlay,
            n_significant_leads=n_sig,
            composite_sharpe=comp_sharpe,
            n_observations=len(self.prices),
        )

    @staticmethod
    def generate_report(result: AnalysisResult, output_path: Path = DEFAULT_OUTPUT) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_build_html(result), encoding="utf-8")
        return output_path


# ── HTML ─────────────────────────────────────────────────────────────────


def _fr(v): return f"{v:.3f}"
def _fp(v): return f"{v:.1f}%"


def _build_html(r: AnalysisResult) -> str:
    ll_rows = "".join(
        f"<tr><td style='text-align:left'>{ll.asset}</td><td>{ll.optimal_lag}d</td>"
        f"<td>{_fr(ll.correlation_at_lag)}</td><td>{ll.direction}</td>"
        f"<td style='color:{('#3fb950' if ll.is_significant else '#8b949e')}'>{'YES' if ll.is_significant else 'no'}</td></tr>"
        for ll in r.lead_lags
    )

    feat_rows = "".join(
        f"<tr><td style='text-align:left'>{f.asset}</td><td>{_fp(f.momentum_1d*100)}</td>"
        f"<td>{_fp(f.momentum_3d*100)}</td><td>{_fp(f.momentum_5d*100)}</td>"
        f"<td>{_fr(f.zscore_20d)}</td><td>{_fr(f.trend_strength)}</td></tr>"
        for f in r.features.values()
    )

    overlay_html = ""
    if r.overlay:
        o = r.overlay
        overlay_html = f"""<h2>EXP-880 Overlay</h2>
        <p>Confirmed: {o.confirmed_win_rate:.1%} WR ({o.signal_confirmed_trades} trades) vs
        Unconfirmed: {o.unconfirmed_win_rate:.1%} WR → {o.improvement_pp:+.1f}pp improvement</p>"""

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Cross-Asset Momentum</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1000px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2{{color:#58a6ff}}.meta{{color:#8b949e}}
table{{width:100%;border-collapse:collapse;margin:12px 0}}th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}th{{color:#8b949e;background:#161b22}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
</style></head><body>
<h1>Cross-Asset Momentum Signals</h1>
<div class="cards">
<div class="c"><div class="l">Significant Leads</div><div class="v">{r.n_significant_leads}/{len(r.lead_lags)}</div></div>
<div class="c"><div class="l">Composite Sharpe</div><div class="v">{_fr(r.composite_sharpe)}</div></div>
<div class="c"><div class="l">Observations</div><div class="v">{r.n_observations}</div></div>
</div>
<h2>Lead-Lag Relationships</h2>
<table><tr><th style="text-align:left">Asset</th><th>Lead (days)</th><th>Correlation</th><th>Direction</th><th>Significant</th></tr>{ll_rows}</table>
<h2>Current Features</h2>
<table><tr><th style="text-align:left">Asset</th><th>Mom 1d</th><th>Mom 3d</th><th>Mom 5d</th><th>Z-Score</th><th>Trend</th></tr>{feat_rows}</table>
{overlay_html}
</body></html>"""
