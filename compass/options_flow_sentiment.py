"""
Options flow sentiment engine — aggregates flow metrics to predict
short-term SPY direction.

Signals:
  1. Put/call volume ratio (ATM, OTM puts, OTM calls)
  2. Dealer gamma exposure (GEX) estimate from volume × OI × gamma
  3. Unusual activity detection (OI changes > 3x avg)
  4. Composite flow score (-1 bearish to +1 bullish)

Use cases:
  - Standalone contrarian/confirming signal
  - EXP-880 timing overlay (block trades in adverse flow)

All methods work on pre-loaded data — no API calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FlowReading:
    """Daily options flow snapshot."""
    date: datetime
    pc_ratio_total: float        # total put/call volume ratio
    pc_ratio_atm: float          # ATM put/call
    pc_ratio_otm_put: float      # far OTM put volume / total
    otm_call_ratio: float        # far OTM call volume / total
    gex_estimate: float          # dealer gamma exposure (notional $)
    unusual_activity_score: float  # 0-1 (1 = very unusual)
    composite_score: float       # -1 (bearish) to +1 (bullish)


@dataclass
class UnusualActivity:
    """A detected unusual options activity event."""
    date: datetime
    strike: float
    option_type: str             # "call" | "put"
    volume: int
    avg_volume: float
    multiple: float              # volume / avg_volume
    oi_change: int


@dataclass
class GEXEstimate:
    """Dealer gamma exposure estimate."""
    date: datetime
    net_gex: float               # positive = dealers long gamma
    call_gex: float
    put_gex: float
    flip_level: float            # price where GEX flips sign


@dataclass
class FlowBacktestResult:
    """Backtest of flow-based timing."""
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    n_signals: int
    win_rate: float
    avg_return_bullish: float
    avg_return_bearish: float
    signal_accuracy: float


# ---------------------------------------------------------------------------
# Synthetic flow data generator
# ---------------------------------------------------------------------------

def generate_flow_data(
    n_days: int = 1512, underlying: float = 450.0, seed: int = 42,
) -> Dict[str, pd.Series]:
    """Generate realistic options flow data calibrated to SPY.

    Returns dict: put_volume, call_volume, atm_put_vol, atm_call_vol,
    otm_put_vol, otm_call_vol, total_oi, oi_change, gamma_weighted_vol,
    spy_returns.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_days)

    # Base volumes (SPY options: ~5M contracts/day)
    base_call = 2_500_000
    base_put = 2_200_000

    # SPY returns with embedded regimes
    spy_ret = rng.normal(0.0004, 0.01, n_days)
    if n_days > 80:
        spy_ret[50:70] = rng.normal(-0.015, 0.025, 20)  # COVID
    if n_days > 650:
        spy_ret[530:620] = rng.normal(-0.003, 0.015, 90)  # 2022

    # Volumes: correlated with fear (put volume spikes before drops)
    fear = np.zeros(n_days)
    for i in range(1, n_days):
        fear[i] = 0.95 * fear[i-1] + (rng.normal(0, 0.1) - spy_ret[i] * 5)
    fear = np.clip(fear, -2, 2)

    put_vol = (base_put * (1 + fear * 0.3) + rng.normal(0, 200000, n_days)).astype(int)
    call_vol = (base_call * (1 - fear * 0.2) + rng.normal(0, 200000, n_days)).astype(int)
    put_vol = np.clip(put_vol, 500000, 8000000)
    call_vol = np.clip(call_vol, 500000, 8000000)

    # ATM vs OTM split
    atm_frac = 0.35
    otm_put_frac = 0.40   # heavy OTM put activity = fear
    atm_put = (put_vol * atm_frac).astype(int)
    atm_call = (call_vol * atm_frac).astype(int)
    otm_put = (put_vol * (otm_put_frac + fear * 0.1)).astype(int)
    otm_call = (call_vol * (0.40 - fear * 0.05)).astype(int)

    # OI and OI changes
    base_oi = 15_000_000
    oi = base_oi + np.cumsum(rng.normal(0, 50000, n_days)).astype(int)
    oi = np.clip(oi, 10_000_000, 25_000_000)
    oi_change = np.diff(oi, prepend=oi[0])

    # Gamma-weighted volume (proxy for GEX)
    # Calls at strikes above spot → dealers long gamma
    # Puts at strikes below spot → dealers short gamma
    gamma_wt = (call_vol * 0.02 - put_vol * 0.015) * underlying * 100
    gamma_wt += rng.normal(0, 1e8, n_days)

    return {
        "put_volume": pd.Series(put_vol, index=idx),
        "call_volume": pd.Series(call_vol, index=idx),
        "atm_put_vol": pd.Series(atm_put, index=idx),
        "atm_call_vol": pd.Series(atm_call, index=idx),
        "otm_put_vol": pd.Series(otm_put, index=idx),
        "otm_call_vol": pd.Series(otm_call, index=idx),
        "total_oi": pd.Series(oi, index=idx),
        "oi_change": pd.Series(oi_change, index=idx),
        "gamma_weighted_vol": pd.Series(gamma_wt, index=idx),
        "spy_returns": pd.Series(spy_ret, index=idx),
    }


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class OptionsFlowSentiment:
    """Options flow sentiment analysis.

    Args:
        lookback: Rolling window for percentile/avg calculations.
        unusual_threshold: OI change multiple to flag as unusual.
        weights: Component weights for composite score.
    """

    def __init__(
        self,
        lookback: int = 21,
        unusual_threshold: float = 3.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.lookback = lookback
        self.unusual_threshold = unusual_threshold
        self.weights = weights or {
            "pc_ratio": 0.30, "gex": 0.30,
            "unusual": 0.20, "otm_skew": 0.20,
        }

    # ------------------------------------------------------------------
    # 1. Put/call ratios
    # ------------------------------------------------------------------

    @staticmethod
    def put_call_ratio(put_vol: pd.Series, call_vol: pd.Series) -> pd.Series:
        """Total put/call volume ratio. >1 = bearish flow."""
        return put_vol / call_vol.replace(0, 1)

    @staticmethod
    def atm_put_call(atm_put: pd.Series, atm_call: pd.Series) -> pd.Series:
        return atm_put / atm_call.replace(0, 1)

    @staticmethod
    def otm_put_fraction(otm_put: pd.Series, total_vol: pd.Series) -> pd.Series:
        """Fraction of total volume that is far OTM puts (fear indicator)."""
        return otm_put / total_vol.replace(0, 1)

    @staticmethod
    def otm_call_fraction(otm_call: pd.Series, total_vol: pd.Series) -> pd.Series:
        return otm_call / total_vol.replace(0, 1)

    # ------------------------------------------------------------------
    # 2. Dealer gamma exposure (GEX)
    # ------------------------------------------------------------------

    def estimate_gex(self, gamma_wt_vol: pd.Series) -> pd.Series:
        """Normalised GEX: positive = dealers long gamma (pin risk)."""
        rolling_std = gamma_wt_vol.rolling(self.lookback).std().replace(0, 1)
        return gamma_wt_vol / rolling_std

    @staticmethod
    def gex_snapshot(
        call_vol: pd.Series, put_vol: pd.Series,
        underlying: float = 450.0,
    ) -> pd.Series:
        """Simple GEX proxy: (call_gamma_vol - put_gamma_vol) × 100 × S."""
        return (call_vol * 0.02 - put_vol * 0.015) * underlying * 100

    # ------------------------------------------------------------------
    # 3. Unusual activity detection
    # ------------------------------------------------------------------

    def detect_unusual(
        self, oi_change: pd.Series,
    ) -> pd.Series:
        """Flag days where OI change > threshold × rolling average."""
        avg = oi_change.abs().rolling(self.lookback, min_periods=5).mean()
        multiple = oi_change.abs() / avg.replace(0, 1)
        return (multiple > self.unusual_threshold).astype(float)

    def unusual_score(self, oi_change: pd.Series) -> pd.Series:
        """Continuous unusual activity score (0-1)."""
        avg = oi_change.abs().rolling(self.lookback, min_periods=5).mean()
        multiple = oi_change.abs() / avg.replace(0, 1)
        return (multiple / (self.unusual_threshold * 2)).clip(0, 1)

    # ------------------------------------------------------------------
    # 4. Composite flow score
    # ------------------------------------------------------------------

    def composite_score(self, data: Dict[str, pd.Series]) -> pd.Series:
        """Composite score from -1 (bearish) to +1 (bullish).

        Components:
        - PC ratio: high = bearish → contrarian bullish (inverted)
        - GEX: positive = dealer long gamma → stabilising (bullish)
        - Unusual: high = uncertainty → slight bearish
        - OTM skew: heavy OTM puts = fear → contrarian bullish
        """
        put_vol = data["put_volume"]
        call_vol = data["call_volume"]
        total = put_vol + call_vol

        # PC ratio percentile (inverted: high PC = contrarian bullish)
        pc = self.put_call_ratio(put_vol, call_vol)
        pc_pctile = pc.rolling(self.lookback * 5, min_periods=20).apply(
            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100, raw=False)
        pc_signal = (50 - pc_pctile) / 50  # invert: high PC → positive signal

        # GEX signal
        gex_raw = data.get("gamma_weighted_vol", pd.Series(0, index=put_vol.index))
        gex_norm = self.estimate_gex(gex_raw)
        gex_signal = gex_norm.clip(-2, 2) / 2  # normalise to -1..1

        # Unusual activity
        oi_change = data.get("oi_change", pd.Series(0, index=put_vol.index))
        unusual = self.unusual_score(oi_change)
        unusual_signal = -unusual * 0.5  # unusual = uncertainty = slight bearish

        # OTM put skew (contrarian: heavy OTM puts = buy signal)
        otm_put = data.get("otm_put_vol", put_vol * 0.4)
        otm_frac = self.otm_put_fraction(otm_put, total)
        otm_pctile = otm_frac.rolling(self.lookback * 5, min_periods=20).apply(
            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100, raw=False)
        otm_signal = (50 - otm_pctile) / 50  # heavy OTM puts → contrarian bullish

        w = self.weights
        composite = (
            w.get("pc_ratio", 0.3) * pc_signal
            + w.get("gex", 0.3) * gex_signal
            + w.get("unusual", 0.2) * unusual_signal
            + w.get("otm_skew", 0.2) * otm_signal
        )
        return composite.clip(-1, 1).fillna(0)

    # ------------------------------------------------------------------
    # Full readings
    # ------------------------------------------------------------------

    def compute_readings(self, data: Dict[str, pd.Series]) -> List[FlowReading]:
        """Compute daily flow readings."""
        composite = self.composite_score(data)
        put_vol = data["put_volume"]
        call_vol = data["call_volume"]
        total = put_vol + call_vol
        pc = self.put_call_ratio(put_vol, call_vol)
        atm_pc = self.atm_put_call(
            data.get("atm_put_vol", put_vol * 0.35),
            data.get("atm_call_vol", call_vol * 0.35))
        otm_put_f = self.otm_put_fraction(
            data.get("otm_put_vol", put_vol * 0.4), total)
        otm_call_f = self.otm_call_fraction(
            data.get("otm_call_vol", call_vol * 0.4), total)
        gex = self.estimate_gex(
            data.get("gamma_weighted_vol", pd.Series(0, index=put_vol.index)))
        unusual = self.unusual_score(
            data.get("oi_change", pd.Series(0, index=put_vol.index)))

        aligned = pd.DataFrame({
            "pc": pc, "atm_pc": atm_pc, "otm_put": otm_put_f,
            "otm_call": otm_call_f, "gex": gex, "unusual": unusual,
            "composite": composite,
        }).dropna()

        readings: List[FlowReading] = []
        for dt, row in aligned.iterrows():
            readings.append(FlowReading(
                date=dt, pc_ratio_total=float(row["pc"]),
                pc_ratio_atm=float(row["atm_pc"]),
                pc_ratio_otm_put=float(row["otm_put"]),
                otm_call_ratio=float(row["otm_call"]),
                gex_estimate=float(row["gex"]),
                unusual_activity_score=float(row["unusual"]),
                composite_score=float(row["composite"]),
            ))
        return readings

    # ------------------------------------------------------------------
    # Signal series
    # ------------------------------------------------------------------

    def signal_series(
        self, data: Dict[str, pd.Series], threshold: float = 0.3,
    ) -> pd.Series:
        """Generate +1/-1/0 signal from composite score."""
        composite = self.composite_score(data)
        return composite.apply(
            lambda x: 1.0 if x > threshold else (-1.0 if x < -threshold else 0.0))

    # ------------------------------------------------------------------
    # EXP-880 overlay
    # ------------------------------------------------------------------

    def overlay_filter(
        self,
        base_signal: pd.Series,
        data: Dict[str, pd.Series],
        block_threshold: float = -0.4,
    ) -> pd.Series:
        """Block base trades when flow is strongly bearish."""
        composite = self.composite_score(data)
        aligned = composite.reindex(base_signal.index).fillna(0)
        filtered = base_signal.copy()
        filtered[aligned < block_threshold] = 0
        return filtered

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(
        self, data: Dict[str, pd.Series],
        threshold: float = 0.3, holding_days: int = 3,
        cost: float = 0.001,
    ) -> FlowBacktestResult:
        """Backtest flow signal as standalone timing strategy."""
        sig = self.signal_series(data, threshold)
        spy_ret = data.get("spy_returns", pd.Series(dtype=float))
        aligned = pd.DataFrame({"sig": sig, "ret": spy_ret}).dropna()

        if len(aligned) < 20:
            return FlowBacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Forward returns
        fwd = aligned["ret"].rolling(holding_days).sum().shift(-holding_days)
        aligned["fwd"] = fwd

        # Strategy
        pos = aligned["sig"].shift(1).fillna(0)
        trades = pos.diff().abs().fillna(0)
        strat_ret = pos * aligned["ret"] - trades * cost

        r = strat_ret.dropna()
        if len(r) < 10:
            return FlowBacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0)

        total = float((1 + r).prod() - 1)
        n_yr = len(r) / TRADING_DAYS
        annual = (1 + total) ** (1 / max(n_yr, 0.01)) - 1
        mu, std = float(r.mean()), float(r.std())
        sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
        eq = (1 + r).cumprod()
        dd = float((1 - eq / eq.expanding().max()).max())

        active = aligned[aligned["sig"] != 0].dropna(subset=["fwd"])
        n_sig = len(active)
        wins = int((active["sig"] * active["fwd"] > 0).sum()) if n_sig > 0 else 0
        wr = wins / n_sig if n_sig > 0 else 0
        bull = active.loc[active["sig"] > 0, "fwd"]
        bear = active.loc[active["sig"] < 0, "fwd"]
        avg_bull = float(bull.mean()) if len(bull) > 0 else 0
        avg_bear = float(bear.mean()) if len(bear) > 0 else 0

        correct = int(((active["sig"] > 0) & (active["fwd"] > 0)).sum()
                       + ((active["sig"] < 0) & (active["fwd"] < 0)).sum()) if n_sig > 0 else 0
        accuracy = correct / n_sig if n_sig > 0 else 0

        return FlowBacktestResult(
            total, annual, sharpe, dd, n_sig, wr,
            avg_bull, avg_bear, accuracy,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, result: FlowBacktestResult,
        readings: Optional[List[FlowReading]] = None,
        output_path: str = "reports/options_flow_sentiment.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Score SVG
        score_svg = ""
        if readings and len(readings) > 10:
            scores = [r.composite_score for r in readings]
            n = len(scores)
            w, h = 750, 180
            pad = 50
            pw, ph = w - 2 * pad, h - 55
            def tx(i): return pad + i / max(n - 1, 1) * pw
            def ty(v): return 28 + (1 - (v + 1) / 2) * ph
            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                      f'style="background:#fff;border:1px solid #e2e8f0;border-radius:6px;margin:.5rem 0">']
            parts.append(f'<text x="{w // 2}" y="16" text-anchor="middle" font-size="12" '
                          f'font-weight="bold" fill="#0f172a">Options Flow Composite Score</text>')
            zy = ty(0)
            parts.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" stroke="#e2e8f0"/>')
            d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(scores[i]):.1f}" for i in range(n))
            parts.append(f'<path d="{d}" fill="none" stroke="#2563eb" stroke-width="1.5"/>')
            parts.append("</svg>")
            score_svg = "\n".join(parts)

        r = result
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Options Flow Sentiment</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fff; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; }}
h2 {{ color: #334155; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; border-bottom: 2px solid #e2e8f0; }}
th:first-child {{ text-align: left; }}
td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #f1f5f9; }}
td:first-child {{ text-align: left; }}
.card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }}
</style></head><body>
<h1>EXP-1310-max: Options Flow Sentiment</h1>
<div class="card">
<p><strong>Sharpe:</strong> {r.sharpe:.2f} |
<strong>Annual Return:</strong> {r.annual_return:.1%} |
<strong>Max DD:</strong> {r.max_drawdown:.1%} |
<strong>Signals:</strong> {r.n_signals} |
<strong>Accuracy:</strong> {r.signal_accuracy:.0%}</p>
</div>

{score_svg}

<h2>Backtest Results</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Return</td><td>{r.total_return:.1%}</td></tr>
<tr><td>Annual Return</td><td>{r.annual_return:.1%}</td></tr>
<tr><td>Sharpe</td><td>{r.sharpe:.2f}</td></tr>
<tr><td>Max Drawdown</td><td>{r.max_drawdown:.1%}</td></tr>
<tr><td>Signals</td><td>{r.n_signals}</td></tr>
<tr><td>Win Rate</td><td>{r.win_rate:.0%}</td></tr>
<tr><td>Accuracy</td><td>{r.signal_accuracy:.0%}</td></tr>
<tr><td>Avg Return (Bullish)</td><td>{r.avg_return_bullish:+.2%}</td></tr>
<tr><td>Avg Return (Bearish)</td><td>{r.avg_return_bearish:+.2%}</td></tr>
</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
