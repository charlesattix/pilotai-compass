"""
Sentiment-driven alpha engine.

Multi-source sentiment aggregation → composite fear/greed score →
contrarian signals at extremes → credit spread timing.

Sources:
  - VIX percentile (fear gauge)
  - Put-call ratio (options flow)
  - AAII bull-bear spread (retail sentiment)
  - Composite fear/greed score (0-100)

Contrarian thesis: extreme fear (score < 10th pctile) → buy signal,
extreme greed (score > 90th pctile) → sell signal.

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
class SentimentReading:
    """Single-day sentiment snapshot."""
    date: datetime
    vix_percentile: float         # 0-100
    put_call_ratio: float         # raw ratio (>1 = bearish)
    put_call_percentile: float    # 0-100
    aaii_spread: float            # bull% - bear% (-100 to +100)
    aaii_percentile: float        # 0-100
    composite_score: float        # 0-100 (0=max fear, 100=max greed)


@dataclass
class ContrarianSignal:
    """Contrarian trade signal from extreme sentiment."""
    date: datetime
    score: float
    signal: int                   # +1 buy (extreme fear), -1 sell (extreme greed), 0 neutral
    strength: float               # 0-1 (how extreme)
    drivers: List[str]            # which sources triggered


@dataclass
class SentimentBacktestResult:
    """Backtest of contrarian sentiment timing."""
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    n_signals: int
    n_buys: int
    n_sells: int
    win_rate: float
    avg_return_after_fear: float
    avg_return_after_greed: float
    signal_accuracy: float        # % of signals that were correct direction


# ---------------------------------------------------------------------------
# Sentiment aggregator
# ---------------------------------------------------------------------------

class SentimentAlpha:
    """Multi-source sentiment aggregation and contrarian signal engine.

    Args:
        lookback: Rolling window for percentile computation.
        fear_threshold: Percentile below which = extreme fear (buy signal).
        greed_threshold: Percentile above which = extreme greed (sell signal).
        weights: Source weights for composite score.
    """

    def __init__(
        self,
        lookback: int = 252,
        fear_threshold: float = 10.0,
        greed_threshold: float = 90.0,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.lookback = lookback
        self.fear_threshold = fear_threshold
        self.greed_threshold = greed_threshold
        self.weights = weights or {"vix": 0.40, "put_call": 0.30, "aaii": 0.30}

    # ------------------------------------------------------------------
    # Percentile helpers
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
        """Compute rolling percentile rank (0-100)."""
        def _pctile(x):
            if len(x) < 2:
                return 50.0
            val = x.iloc[-1]
            return float((x < val).sum() / len(x) * 100)
        return series.rolling(window, min_periods=max(20, window // 4)).apply(_pctile, raw=False)

    # ------------------------------------------------------------------
    # VIX sentiment
    # ------------------------------------------------------------------

    def vix_percentile(self, vix: pd.Series) -> pd.Series:
        """VIX percentile — higher = more fear."""
        return self.rolling_percentile(vix, self.lookback)

    # ------------------------------------------------------------------
    # Put-call ratio
    # ------------------------------------------------------------------

    def put_call_percentile(self, put_call: pd.Series) -> pd.Series:
        """Put-call ratio percentile — higher = more bearish."""
        return self.rolling_percentile(put_call, self.lookback)

    # ------------------------------------------------------------------
    # AAII bull-bear spread
    # ------------------------------------------------------------------

    def aaii_percentile(self, aaii_spread: pd.Series) -> pd.Series:
        """AAII spread percentile — higher = more bullish (greed)."""
        return self.rolling_percentile(aaii_spread, self.lookback)

    # ------------------------------------------------------------------
    # Composite fear/greed score
    # ------------------------------------------------------------------

    def composite_score(
        self,
        vix_pctile: pd.Series,
        pc_pctile: pd.Series,
        aaii_pctile: pd.Series,
    ) -> pd.Series:
        """Composite score: 0 = max fear, 100 = max greed.

        VIX and put-call are inverted (high VIX = fear = low score).
        AAII is direct (high bullish = greed = high score).
        """
        w = self.weights
        # Invert VIX and put-call so high values = greed
        vix_greed = 100 - vix_pctile
        pc_greed = 100 - pc_pctile
        aaii_greed = aaii_pctile

        composite = (
            w.get("vix", 0.4) * vix_greed
            + w.get("put_call", 0.3) * pc_greed
            + w.get("aaii", 0.3) * aaii_greed
        )
        return composite.clip(0, 100)

    # ------------------------------------------------------------------
    # Full sentiment readings
    # ------------------------------------------------------------------

    def compute_readings(
        self,
        vix: pd.Series,
        put_call: pd.Series,
        aaii_spread: pd.Series,
    ) -> List[SentimentReading]:
        """Compute sentiment readings for all dates."""
        vix_p = self.vix_percentile(vix)
        pc_p = self.put_call_percentile(put_call)
        aaii_p = self.aaii_percentile(aaii_spread)
        composite = self.composite_score(vix_p, pc_p, aaii_p)

        aligned = pd.DataFrame({
            "vix_p": vix_p, "pc_raw": put_call, "pc_p": pc_p,
            "aaii_raw": aaii_spread, "aaii_p": aaii_p, "composite": composite,
        }).dropna()

        readings: List[SentimentReading] = []
        for dt, row in aligned.iterrows():
            readings.append(SentimentReading(
                date=dt,
                vix_percentile=float(row["vix_p"]),
                put_call_ratio=float(row["pc_raw"]),
                put_call_percentile=float(row["pc_p"]),
                aaii_spread=float(row["aaii_raw"]),
                aaii_percentile=float(row["aaii_p"]),
                composite_score=float(row["composite"]),
            ))
        return readings

    # ------------------------------------------------------------------
    # Contrarian signals
    # ------------------------------------------------------------------

    def generate_signals(
        self, readings: List[SentimentReading],
    ) -> List[ContrarianSignal]:
        """Generate contrarian signals at sentiment extremes."""
        signals: List[ContrarianSignal] = []
        for r in readings:
            drivers: List[str] = []
            signal = 0
            strength = 0.0

            if r.composite_score <= self.fear_threshold:
                signal = 1  # buy (contrarian: fear = opportunity)
                strength = (self.fear_threshold - r.composite_score) / self.fear_threshold
                if r.vix_percentile > 90:
                    drivers.append("vix_extreme_fear")
                if r.put_call_percentile > 90:
                    drivers.append("put_call_extreme_bearish")
                if r.aaii_percentile < 10:
                    drivers.append("aaii_extreme_bearish")
                if not drivers:
                    drivers.append("composite_fear")

            elif r.composite_score >= self.greed_threshold:
                signal = -1  # sell (contrarian: greed = danger)
                strength = (r.composite_score - self.greed_threshold) / (100 - self.greed_threshold)
                if r.vix_percentile < 10:
                    drivers.append("vix_extreme_complacency")
                if r.put_call_percentile < 10:
                    drivers.append("put_call_extreme_bullish")
                if r.aaii_percentile > 90:
                    drivers.append("aaii_extreme_bullish")
                if not drivers:
                    drivers.append("composite_greed")

            signals.append(ContrarianSignal(
                date=r.date, score=r.composite_score,
                signal=signal, strength=min(strength, 1.0),
                drivers=drivers,
            ))
        return signals

    # ------------------------------------------------------------------
    # Signal series (for backtesting)
    # ------------------------------------------------------------------

    def signal_series(
        self,
        vix: pd.Series,
        put_call: pd.Series,
        aaii_spread: pd.Series,
    ) -> pd.Series:
        """Generate a signal series (+1/-1/0) aligned to dates."""
        readings = self.compute_readings(vix, put_call, aaii_spread)
        signals = self.generate_signals(readings)
        dates = [s.date for s in signals]
        vals = [s.signal for s in signals]
        return pd.Series(vals, index=dates, dtype=float, name="sentiment_signal")

    # ------------------------------------------------------------------
    # Backtest contrarian timing
    # ------------------------------------------------------------------

    def backtest(
        self,
        vix: pd.Series,
        put_call: pd.Series,
        aaii_spread: pd.Series,
        returns: pd.Series,
        holding_period: int = 5,
        cost: float = 0.001,
    ) -> SentimentBacktestResult:
        """Backtest contrarian credit spread timing.

        At extreme fear → enter bull put spread (long signal).
        At extreme greed → enter bear call spread (short signal).
        Hold for `holding_period` days, then exit.
        """
        sig = self.signal_series(vix, put_call, aaii_spread)
        aligned = pd.DataFrame({"sig": sig, "ret": returns}).dropna()

        if len(aligned) < 20:
            return SentimentBacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Forward returns for holding period
        fwd = aligned["ret"].rolling(holding_period).sum().shift(-holding_period)
        aligned["fwd"] = fwd

        # Strategy returns: signal × forward return - cost
        active = aligned[aligned["sig"] != 0].copy()
        active["strat_ret"] = active["sig"] * active["fwd"] - cost

        n_signals = len(active)
        if n_signals == 0:
            return SentimentBacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        n_buys = int((active["sig"] > 0).sum())
        n_sells = int((active["sig"] < 0).sum())
        wins = int((active["strat_ret"] > 0).sum())
        win_rate = wins / n_signals

        # Average return after fear vs greed
        fear_rets = active.loc[active["sig"] > 0, "fwd"]
        greed_rets = active.loc[active["sig"] < 0, "fwd"]
        avg_fear = float(fear_rets.mean()) if len(fear_rets) > 0 else 0.0
        avg_greed = float(greed_rets.mean()) if len(greed_rets) > 0 else 0.0

        # Signal accuracy: fear signals followed by positive returns, greed by negative
        correct = int(((active["sig"] > 0) & (active["fwd"] > 0)).sum()
                       + ((active["sig"] < 0) & (active["fwd"] < 0)).sum())
        accuracy = correct / n_signals if n_signals > 0 else 0

        # Full equity curve (sparse: only on signal days, cash otherwise)
        daily_ret = aligned["sig"].shift(1).fillna(0) * aligned["ret"] - (
            aligned["sig"].diff().abs().fillna(0) * cost)
        total = float((1 + daily_ret).prod() - 1)
        n_years = len(daily_ret) / TRADING_DAYS
        annual = (1 + total) ** (1 / max(n_years, 0.01)) - 1

        mu = float(daily_ret.mean())
        std = float(daily_ret.std())
        sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0

        eq = (1 + daily_ret).cumprod()
        dd = float((1 - eq / eq.expanding().max()).max())

        return SentimentBacktestResult(
            total_return=total, annual_return=annual,
            sharpe=sharpe, max_drawdown=dd,
            n_signals=n_signals, n_buys=n_buys, n_sells=n_sells,
            win_rate=win_rate,
            avg_return_after_fear=avg_fear,
            avg_return_after_greed=avg_greed,
            signal_accuracy=accuracy,
        )

    # ------------------------------------------------------------------
    # Synthetic data generator
    # ------------------------------------------------------------------

    @staticmethod
    def generate_synthetic_data(
        n_days: int = 1512, seed: int = 42,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Generate VIX, put-call, AAII, and SPY returns for testing.

        Calibrated to real statistical properties:
        - VIX: mean 20, range 10-80, mean-reverting, spikes in crashes
        - Put-call: mean 0.85, range 0.5-1.5, correlated with VIX
        - AAII spread: mean +5%, range -40% to +40%, contrarian indicator
        - Returns: ~12% annual, 16% vol, correlated with sentiment extremes
        """
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2020-01-02", periods=n_days)

        # VIX
        vix = np.zeros(n_days)
        vix[0] = 15.0
        for i in range(1, n_days):
            revert = 0.03 * (20 - vix[i - 1])
            shock = rng.normal(0, 1.2)
            vix[i] = max(10, min(80, vix[i - 1] + revert + shock))
        # COVID spike
        vix[45:60] = np.clip(np.linspace(35, 75, 15) + rng.normal(0, 3, 15), 30, 80)
        vix[60:85] = np.clip(np.linspace(70, 25, 25) + rng.normal(0, 2, 25), 15, 80)

        # Put-call ratio (correlated with VIX)
        pc = 0.70 + (vix - 20) * 0.015 + rng.normal(0, 0.08, n_days)
        pc = np.clip(pc, 0.5, 1.8)

        # AAII spread (inversely correlated with VIX)
        aaii = 10 - (vix - 20) * 0.8 + rng.normal(0, 8, n_days)
        aaii = np.clip(aaii, -40, 40)

        # SPY returns (negatively correlated with VIX spikes)
        base_ret = rng.normal(0.0005, 0.01, n_days)
        vix_impact = -(np.diff(vix, prepend=vix[0])) * 0.003
        returns = base_ret + vix_impact
        # COVID crash
        returns[45:65] = rng.normal(-0.015, 0.03, 20)
        returns[65:90] = rng.normal(0.008, 0.015, 25)

        return (
            pd.Series(vix, index=idx, name="vix"),
            pd.Series(pc, index=idx, name="put_call"),
            pd.Series(aaii, index=idx, name="aaii_spread"),
            pd.Series(returns, index=idx, name="returns"),
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        result: SentimentBacktestResult,
        readings: Optional[List[SentimentReading]] = None,
        output_path: str = "reports/sentiment_alpha.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Score timeline SVG
        score_svg = ""
        if readings and len(readings) > 10:
            scores = [r.composite_score for r in readings]
            n = len(scores)
            w, h = 750, 200
            pad = 50
            pw, ph = w - 2 * pad, h - 60
            def tx(i): return pad + i / max(n - 1, 1) * pw
            def ty(v): return 30 + (1 - v / 100) * ph

            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                      f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:.5rem 0">']
            parts.append(f'<text x="{w // 2}" y="16" text-anchor="middle" font-size="12" '
                          f'font-weight="bold" fill="#1a1a2e">Composite Fear/Greed Score</text>')
            # Fear zone
            parts.append(f'<rect x="{pad}" y="{ty(self.fear_threshold):.0f}" '
                          f'width="{pw}" height="{ty(0) - ty(self.fear_threshold):.0f}" '
                          f'fill="#dcfce7" opacity="0.5"/>')
            # Greed zone
            parts.append(f'<rect x="{pad}" y="{ty(100):.0f}" '
                          f'width="{pw}" height="{ty(self.greed_threshold) - ty(100):.0f}" '
                          f'fill="#fef2f2" opacity="0.5"/>')
            d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(scores[i]):.1f}" for i in range(n))
            parts.append(f'<path d="{d}" fill="none" stroke="#2563eb" stroke-width="1.5"/>')
            parts.append(f'<text x="{w - pad + 3}" y="{ty(self.fear_threshold) + 4:.0f}" '
                          f'font-size="9" fill="#059669">BUY</text>')
            parts.append(f'<text x="{w - pad + 3}" y="{ty(self.greed_threshold) + 4:.0f}" '
                          f'font-size="9" fill="#dc2626">SELL</text>')
            parts.append("</svg>")
            score_svg = "\n".join(parts)

        r = result
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sentiment Alpha</title>
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
<h1>EXP-1100-max: Sentiment-Driven Alpha</h1>
<div class="card">
<p><strong>Thesis:</strong> Extreme sentiment readings predict short-term mean reversion.
Buy on extreme fear, sell on extreme greed.</p>
</div>

{score_svg}

<h2>Backtest Results</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total Return</td><td>{r.total_return:.1%}</td></tr>
<tr><td>Annual Return</td><td>{r.annual_return:.1%}</td></tr>
<tr><td>Sharpe Ratio</td><td>{r.sharpe:.2f}</td></tr>
<tr><td>Max Drawdown</td><td>{r.max_drawdown:.1%}</td></tr>
<tr><td>Total Signals</td><td>{r.n_signals}</td></tr>
<tr><td>Buy (Fear) Signals</td><td>{r.n_buys}</td></tr>
<tr><td>Sell (Greed) Signals</td><td>{r.n_sells}</td></tr>
<tr><td>Win Rate</td><td>{r.win_rate:.1%}</td></tr>
<tr><td>Signal Accuracy</td><td>{r.signal_accuracy:.1%}</td></tr>
<tr><td>Avg Return After Fear</td><td>{r.avg_return_after_fear:+.2%}</td></tr>
<tr><td>Avg Return After Greed</td><td>{r.avg_return_after_greed:+.2%}</td></tr>
</table>

<h2>Methodology</h2>
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Fear Threshold</td><td>&le; {self.fear_threshold:.0f}th percentile</td></tr>
<tr><td>Greed Threshold</td><td>&ge; {self.greed_threshold:.0f}th percentile</td></tr>
<tr><td>Lookback</td><td>{self.lookback} days</td></tr>
<tr><td>Weights</td><td>VIX {self.weights.get('vix', 0):.0%}, PC {self.weights.get('put_call', 0):.0%}, AAII {self.weights.get('aaii', 0):.0%}</td></tr>
</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
