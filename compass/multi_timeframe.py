"""
Multi-timeframe signal aggregator for credit spread portfolios.

Computes signals at multiple timeframes (1min to daily), aligns and
resamples data, scores cross-timeframe confirmation, detects divergence
between timeframes, finds optimal timeframe weighting via walk-forward
validation, and selects regime-dependent timeframes.

Generates an HTML report at reports/multi_timeframe.html with per-timeframe
signals, confirmation matrix, and divergence alerts.

Usage::

    from compass.multi_timeframe import MultiTimeframeAggregator
    agg = MultiTimeframeAggregator(ohlcv_1min)
    results = agg.analyze()
    agg.generate_report()
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "multi_timeframe.html"

TIMEFRAMES = ("1min", "5min", "15min", "1h", "4h", "1D")

# pandas resample rules
_RESAMPLE_MAP = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "1h": "1h",
    "4h": "4h",
    "1D": "1D",
}

REGIMES = ("bull", "bear", "high_vol", "neutral")


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class TimeframeSignal:
    """Signal computed at a single timeframe."""
    timeframe: str
    signal: float          # -1 to +1
    strength: float        # 0-1 absolute confidence
    trend: str             # "bullish", "bearish", "neutral"
    rsi: float
    momentum: float
    ma_cross: float        # fast MA − slow MA, normalised
    n_bars: int


@dataclass
class ConfirmationScore:
    """Cross-timeframe confirmation assessment."""
    score: float           # 0-1, 1 = all timeframes agree
    n_bullish: int
    n_bearish: int
    n_neutral: int
    dominant_direction: str
    aligned_timeframes: List[str]
    conflicting_timeframes: List[str]


@dataclass
class DivergenceAlert:
    """Divergence between two timeframes."""
    tf_short: str
    tf_long: str
    short_signal: float
    long_signal: float
    divergence_type: str   # "bullish_divergence", "bearish_divergence"
    severity: str          # "low", "medium", "high"
    description: str


@dataclass
class TimeframeWeight:
    """Optimal weight for a single timeframe."""
    timeframe: str
    weight: float
    sharpe: float
    hit_rate: float


@dataclass
class RegimeTimeframeSelection:
    """Which timeframes to emphasise per regime."""
    regime: str
    weights: Dict[str, float]
    best_timeframe: str
    ensemble_sharpe: float
    n_obs: int


@dataclass
class AggregatedSignal:
    """Final weighted signal across all timeframes."""
    signal: float
    confidence: float
    confirmation: ConfirmationScore
    regime: str
    weights_used: Dict[str, float]


# ── Signal computation helpers ──────────────────────────────────────────


def _rsi(series: pd.Series, period: int = 14) -> float:
    """Compute RSI on the last *period* bars."""
    if len(series) < period + 1:
        return 50.0
    delta = series.diff().iloc[-period:]
    gain = delta.clip(lower=0).mean()
    loss = -delta.clip(upper=0).mean()
    if loss == 0:
        return 100.0
    rs = gain / loss
    return float(100 - 100 / (1 + rs))


def _momentum(series: pd.Series, period: int = 10) -> float:
    """Simple momentum: current close / close N bars ago − 1."""
    if len(series) <= period:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-period - 1] - 1)


def _ma_cross(series: pd.Series, fast: int = 10, slow: int = 30) -> float:
    """Normalised MA cross: (fast_ma − slow_ma) / slow_ma."""
    if len(series) < slow:
        return 0.0
    fast_ma = float(series.iloc[-fast:].mean())
    slow_ma = float(series.iloc[-slow:].mean())
    if slow_ma == 0:
        return 0.0
    return (fast_ma - slow_ma) / abs(slow_ma)


def compute_signal(closes: pd.Series, timeframe: str) -> TimeframeSignal:
    """Compute a composite signal for one timeframe's close prices."""
    r = _rsi(closes)
    m = _momentum(closes)
    ma = _ma_cross(closes)

    # Composite signal: weighted sum, clamped to [-1, 1]
    raw = 0.4 * (r - 50) / 50 + 0.3 * np.clip(m * 20, -1, 1) + 0.3 * np.clip(ma * 50, -1, 1)
    signal = float(np.clip(raw, -1, 1))
    strength = float(min(abs(signal), 1.0))

    if signal > 0.15:
        trend = "bullish"
    elif signal < -0.15:
        trend = "bearish"
    else:
        trend = "neutral"

    return TimeframeSignal(
        timeframe=timeframe, signal=signal, strength=strength,
        trend=trend, rsi=r, momentum=m, ma_cross=ma, n_bars=len(closes),
    )


# ── Aggregator ──────────────────────────────────────────────────────────


class MultiTimeframeAggregator:
    """Aggregate signals across multiple timeframes."""

    def __init__(
        self,
        data: pd.DataFrame,
        timeframes: Optional[Sequence[str]] = None,
        regimes: Optional[pd.Series] = None,
        returns: Optional[pd.Series] = None,
        lookback: int = 60,
        walk_forward_folds: int = 5,
    ) -> None:
        self.data = data.copy()
        self.timeframes = list(timeframes or TIMEFRAMES)
        self.regimes = regimes
        self.returns = returns
        self.lookback = lookback
        self.walk_forward_folds = walk_forward_folds

        # Ensure 'close' column exists
        if "close" not in self.data.columns:
            if "Close" in self.data.columns:
                self.data["close"] = self.data["Close"]
            elif len(self.data.columns) == 1:
                self.data.columns = ["close"]

        # Results
        self.resampled: Dict[str, pd.DataFrame] = {}
        self.tf_signals: Dict[str, TimeframeSignal] = {}
        self.confirmation: Optional[ConfirmationScore] = None
        self.divergences: List[DivergenceAlert] = []
        self.optimal_weights: Dict[str, TimeframeWeight] = {}
        self.regime_selections: Dict[str, RegimeTimeframeSelection] = {}
        self.aggregated: Optional[AggregatedSignal] = None

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> "MultiTimeframeAggregator":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls(df, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        self.resampled = self._resample_all()
        self.tf_signals = self._compute_all_signals()
        self.confirmation = self._confirmation_score()
        self.divergences = self._detect_divergences()
        self.optimal_weights = self._walk_forward_weights()
        self.regime_selections = self._regime_dependent_selection()
        self.aggregated = self._aggregate_signal()
        return {
            "tf_signals": self.tf_signals,
            "confirmation": self.confirmation,
            "divergences": self.divergences,
            "optimal_weights": self.optimal_weights,
            "regime_selections": self.regime_selections,
            "aggregated": self.aggregated,
        }

    # ── Resampling ──────────────────────────────────────────────────────

    def _resample_all(self) -> Dict[str, pd.DataFrame]:
        """Resample the base data to each timeframe."""
        result: Dict[str, pd.DataFrame] = {}
        for tf in self.timeframes:
            rule = _RESAMPLE_MAP.get(tf, tf)
            try:
                ohlc = self.data["close"].resample(rule).ohlc()
                ohlc.columns = ["open", "high", "low", "close"]
                ohlc = ohlc.dropna()
                if len(ohlc) >= 5:
                    result[tf] = ohlc
            except (ValueError, TypeError):
                # If data frequency doesn't support this resample, skip
                continue
        return result

    # ── Signal computation ──────────────────────────────────────────────

    def _compute_all_signals(self) -> Dict[str, TimeframeSignal]:
        signals: Dict[str, TimeframeSignal] = {}
        for tf, ohlc in self.resampled.items():
            signals[tf] = compute_signal(ohlc["close"], tf)
        return signals

    # ── Confirmation scoring ────────────────────────────────────────────

    def _confirmation_score(self) -> ConfirmationScore:
        if not self.tf_signals:
            return ConfirmationScore(0, 0, 0, 0, "neutral", [], [])

        n_bull = n_bear = n_neut = 0
        aligned: List[str] = []
        conflicting: List[str] = []

        for tf, sig in self.tf_signals.items():
            if sig.trend == "bullish":
                n_bull += 1
            elif sig.trend == "bearish":
                n_bear += 1
            else:
                n_neut += 1

        total = len(self.tf_signals)
        dominant = "bullish" if n_bull > n_bear else "bearish" if n_bear > n_bull else "neutral"

        for tf, sig in self.tf_signals.items():
            if sig.trend == dominant:
                aligned.append(tf)
            elif sig.trend != "neutral":
                conflicting.append(tf)

        score = max(n_bull, n_bear) / total if total > 0 else 0.0

        return ConfirmationScore(
            score=score, n_bullish=n_bull, n_bearish=n_bear,
            n_neutral=n_neut, dominant_direction=dominant,
            aligned_timeframes=aligned, conflicting_timeframes=conflicting,
        )

    # ── Divergence detection ────────────────────────────────────────────

    def _detect_divergences(self) -> List[DivergenceAlert]:
        alerts: List[DivergenceAlert] = []
        tfs = list(self.tf_signals.keys())

        for i in range(len(tfs)):
            for j in range(i + 1, len(tfs)):
                short_tf, long_tf = tfs[i], tfs[j]
                s_sig = self.tf_signals[short_tf]
                l_sig = self.tf_signals[long_tf]

                # Divergence: short and long timeframes disagree
                if s_sig.signal * l_sig.signal < -0.05:  # opposite signs
                    delta = abs(s_sig.signal - l_sig.signal)
                    if s_sig.signal > 0 and l_sig.signal < 0:
                        div_type = "bullish_divergence"
                    else:
                        div_type = "bearish_divergence"

                    severity = "high" if delta > 1.0 else "medium" if delta > 0.5 else "low"
                    alerts.append(DivergenceAlert(
                        tf_short=short_tf, tf_long=long_tf,
                        short_signal=s_sig.signal, long_signal=l_sig.signal,
                        divergence_type=div_type, severity=severity,
                        description=f"{short_tf} {s_sig.trend} vs {long_tf} {l_sig.trend}",
                    ))

        return sorted(alerts, key=lambda a: -abs(a.short_signal - a.long_signal))

    # ── Walk-forward optimal weights ────────────────────────────────────

    def _walk_forward_weights(self) -> Dict[str, TimeframeWeight]:
        """Compute optimal timeframe weights via walk-forward validation."""
        if self.returns is None or len(self.returns) < 30:
            # Equal weight fallback
            n = len(self.tf_signals) or 1
            return {
                tf: TimeframeWeight(timeframe=tf, weight=1.0 / n, sharpe=0.0, hit_rate=0.5)
                for tf in self.tf_signals
            }

        results: Dict[str, TimeframeWeight] = {}
        raw_scores: Dict[str, float] = {}

        for tf, ohlc in self.resampled.items():
            closes = ohlc["close"]
            # Generate rolling signal series aligned to returns
            sig_series = self._rolling_signal_series(closes, tf)
            common = sig_series.index.intersection(self.returns.index)
            if len(common) < 20:
                continue
            sig_aligned = sig_series.loc[common]
            ret_aligned = self.returns.loc[common]

            # Walk-forward: split into folds
            fold_size = len(common) // max(self.walk_forward_folds, 1)
            if fold_size < 5:
                continue

            fold_sharpes: List[float] = []
            fold_hits: List[float] = []

            for k in range(1, self.walk_forward_folds):
                test_start = k * fold_size
                test_end = min(test_start + fold_size, len(common))
                test_sig = sig_aligned.iloc[test_start:test_end]
                test_ret = ret_aligned.iloc[test_start:test_end]
                strat_ret = test_sig * test_ret
                if len(strat_ret) < 5 or strat_ret.std() == 0:
                    continue
                sh = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252))
                hr = float((strat_ret > 0).mean())
                fold_sharpes.append(sh)
                fold_hits.append(hr)

            if not fold_sharpes:
                continue

            avg_sharpe = float(np.mean(fold_sharpes))
            avg_hit = float(np.mean(fold_hits))
            score = max(avg_sharpe, 0.0) + 0.5 * avg_hit
            raw_scores[tf] = score
            results[tf] = TimeframeWeight(
                timeframe=tf, weight=0.0,
                sharpe=avg_sharpe, hit_rate=avg_hit,
            )

        total = sum(raw_scores.values())
        if total > 0:
            for tf in results:
                results[tf].weight = raw_scores[tf] / total
        elif results:
            n = len(results)
            for tf in results:
                results[tf].weight = 1.0 / n

        return results

    def _rolling_signal_series(self, closes: pd.Series, tf: str) -> pd.Series:
        """Compute rolling signals for a close price series."""
        min_bars = 30
        signals = []
        idx = []
        for i in range(min_bars, len(closes)):
            window = closes.iloc[max(0, i - 100):i + 1]
            sig = compute_signal(window, tf)
            signals.append(sig.signal)
            idx.append(closes.index[i])
        return pd.Series(signals, index=idx, dtype=float)

    # ── Regime-dependent selection ──────────────────────────────────────

    def _regime_dependent_selection(self) -> Dict[str, RegimeTimeframeSelection]:
        if self.regimes is None or self.returns is None:
            return {}
        results: Dict[str, RegimeTimeframeSelection] = {}

        for regime in REGIMES:
            mask = self.regimes == regime
            if mask.sum() < 10:
                continue
            regime_ret = self.returns.loc[mask]
            raw: Dict[str, float] = {}

            for tf, ohlc in self.resampled.items():
                closes = ohlc["close"]
                sig_series = self._rolling_signal_series(closes, tf)
                common = sig_series.index.intersection(regime_ret.index)
                if len(common) < 5:
                    continue
                strat_ret = sig_series.loc[common] * regime_ret.loc[common]
                sh = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0.0
                raw[tf] = max(sh, 0.01)

            if not raw:
                continue
            total = sum(raw.values())
            weights = {tf: raw[tf] / total for tf in raw}
            best = max(raw, key=raw.get)

            # Ensemble Sharpe
            ens_ret_parts = []
            for tf in raw:
                sig_series = self._rolling_signal_series(self.resampled[tf]["close"], tf)
                common = sig_series.index.intersection(regime_ret.index)
                if len(common) > 0:
                    ens_ret_parts.append(weights[tf] * sig_series.loc[common] * regime_ret.loc[common])
            if ens_ret_parts:
                ens_ret = sum(ens_ret_parts)
                ens_sh = float(ens_ret.mean() / ens_ret.std() * np.sqrt(252)) if ens_ret.std() > 0 else 0.0
            else:
                ens_sh = 0.0

            results[regime] = RegimeTimeframeSelection(
                regime=regime, weights=weights, best_timeframe=best,
                ensemble_sharpe=ens_sh, n_obs=int(mask.sum()),
            )
        return results

    # ── Aggregated signal ───────────────────────────────────────────────

    def _aggregate_signal(self) -> AggregatedSignal:
        """Combine signals across timeframes using optimal weights."""
        if not self.tf_signals:
            return AggregatedSignal(0, 0, self.confirmation or ConfirmationScore(0, 0, 0, 0, "neutral", [], []),
                                    "neutral", {})

        # Determine current regime
        regime = "neutral"
        if self.regimes is not None and len(self.regimes) > 0:
            regime = str(self.regimes.iloc[-1])

        # Pick weights: regime-specific if available, else walk-forward
        rs = self.regime_selections.get(regime)
        if rs:
            weights = rs.weights
        elif self.optimal_weights:
            weights = {tf: w.weight for tf, w in self.optimal_weights.items()}
        else:
            n = len(self.tf_signals)
            weights = {tf: 1.0 / n for tf in self.tf_signals}

        signal = 0.0
        for tf, sig in self.tf_signals.items():
            w = weights.get(tf, 0.0)
            signal += w * sig.signal

        signal = float(np.clip(signal, -1, 1))
        conf = self.confirmation
        confidence = conf.score if conf else 0.0

        return AggregatedSignal(
            signal=signal, confidence=confidence,
            confirmation=conf or ConfirmationScore(0, 0, 0, 0, "neutral", [], []),
            regime=regime, weights_used=weights,
        )

    # ── Report generation ───────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.aggregated is None:
            self.analyze()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Report written to %s", out)
        return str(out.resolve())

    # ── Charts ──────────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        charts: Dict[str, str] = {}
        charts["signal_bars"] = self._chart_signal_bars()
        charts["confirmation"] = self._chart_confirmation_matrix()
        charts["weights"] = self._chart_weights()
        return charts

    def _chart_signal_bars(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.tf_signals:
            return ""
        tfs = list(self.tf_signals.keys())
        sigs = [self.tf_signals[tf].signal for tf in tfs]
        colors = ["#16a34a" if s > 0.15 else "#dc2626" if s < -0.15 else "#64748b" for s in sigs]

        fig, ax = plt.subplots(figsize=(8, max(3, len(tfs) * 0.5)))
        ax.barh(tfs, sigs, color=colors, alpha=0.85, edgecolor="white")
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlim(-1.1, 1.1)
        ax.set_xlabel("Signal (-1 to +1)")
        ax.set_title("Per-Timeframe Signals", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_confirmation_matrix(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.tf_signals:
            return ""
        tfs = list(self.tf_signals.keys())
        n = len(tfs)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                si = self.tf_signals[tfs[i]].signal
                sj = self.tf_signals[tfs[j]].signal
                matrix[i, j] = si * sj  # agreement: positive = same direction

        fig, ax = plt.subplots(figsize=(max(4, n * 0.9), max(4, n * 0.8)))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(tfs, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(tfs, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(matrix[i, j]) > 0.5 else "black")
        fig.colorbar(im, shrink=0.8)
        ax.set_title("Cross-Timeframe Confirmation Matrix", fontsize=11)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_weights(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.optimal_weights:
            return ""
        tfs = list(self.optimal_weights.keys())
        ws = [self.optimal_weights[tf].weight for tf in tfs]
        fig, ax = plt.subplots(figsize=(6, max(3, len(tfs) * 0.5)))
        ax.barh(tfs, ws, color="#3b82f6", alpha=0.85)
        ax.set_xlabel("Weight")
        ax.set_title("Optimal Timeframe Weights", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── HTML builder ────────────────────────────────────────────────────

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        agg = self.aggregated or AggregatedSignal(0, 0, ConfirmationScore(0, 0, 0, 0, "neutral", [], []), "neutral", {})
        conf = agg.confirmation

        sig_color = "#16a34a" if agg.signal > 0.15 else "#dc2626" if agg.signal < -0.15 else "#64748b"

        # Per-TF signal table
        tf_rows = ""
        for tf in self.tf_signals.values():
            cls = "good" if tf.trend == "bullish" else "bad" if tf.trend == "bearish" else ""
            tf_rows += (
                f'<tr><td>{tf.timeframe}</td>'
                f'<td class="{cls}">{tf.signal:+.3f}</td>'
                f'<td>{tf.strength:.2f}</td>'
                f'<td class="{cls}">{tf.trend}</td>'
                f'<td>{tf.rsi:.1f}</td>'
                f'<td>{tf.momentum:+.4f}</td>'
                f'<td>{tf.ma_cross:+.4f}</td>'
                f'<td>{tf.n_bars}</td></tr>\n'
            )

        # Divergence table
        div_rows = ""
        for d in self.divergences:
            sev_cls = {"high": "bad", "medium": "warn", "low": ""}.get(d.severity, "")
            div_rows += (
                f'<tr><td>{d.tf_short}</td><td>{d.tf_long}</td>'
                f'<td>{d.short_signal:+.3f}</td><td>{d.long_signal:+.3f}</td>'
                f'<td>{d.divergence_type}</td>'
                f'<td class="{sev_cls}">{d.severity}</td>'
                f'<td>{d.description}</td></tr>\n'
            )
        if not div_rows:
            div_rows = '<tr><td colspan="7" style="text-align:center;color:#64748b">No divergences</td></tr>'

        # Weights table
        wt_rows = ""
        for tw in sorted(self.optimal_weights.values(), key=lambda w: -w.weight):
            wt_rows += (
                f'<tr><td>{tw.timeframe}</td><td>{tw.weight:.1%}</td>'
                f'<td>{tw.sharpe:.2f}</td><td>{tw.hit_rate:.1%}</td></tr>\n'
            )

        # Regime selections
        regime_rows = ""
        for r in self.regime_selections.values():
            w_str = " / ".join(f"{r.weights.get(tf, 0):.0%}" for tf in self.tf_signals)
            regime_rows += (
                f'<tr><td>{r.regime}</td><td>{r.best_timeframe}</td>'
                f'<td>{w_str}</td><td>{r.ensemble_sharpe:.2f}</td>'
                f'<td>{r.n_obs}</td></tr>\n'
            )

        def _img(key: str) -> str:
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ""

        tf_header = " / ".join(self.tf_signals.keys()) if self.tf_signals else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Multi-Timeframe Signal Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .warn {{ color: #f59e0b; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; text-align: right; }}
  td:first-child {{ text-align: left; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Multi-Timeframe Signal Dashboard</h1>
<div class="meta">{len(self.tf_signals)} timeframes &middot; {len(self.resampled)} resampled &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value" style="color:{sig_color}">{agg.signal:+.3f}</div><div class="label">Aggregate Signal</div></div>
  <div class="kpi"><div class="value">{conf.score:.0%}</div><div class="label">Confirmation</div></div>
  <div class="kpi"><div class="value">{conf.dominant_direction}</div><div class="label">Direction</div></div>
  <div class="kpi"><div class="value">{len(self.divergences)}</div><div class="label">Divergences</div></div>
  <div class="kpi"><div class="value">{agg.regime}</div><div class="label">Regime</div></div>
</div>

<h2>1. Per-Timeframe Signals</h2>
{_img("signal_bars")}
<table>
<thead><tr><th>Timeframe</th><th>Signal</th><th>Strength</th><th>Trend</th><th>RSI</th><th>Momentum</th><th>MA Cross</th><th>Bars</th></tr></thead>
<tbody>{tf_rows}</tbody>
</table>

<h2>2. Confirmation Matrix</h2>
{_img("confirmation")}

<h2>3. Divergence Alerts</h2>
<table>
<thead><tr><th>Short TF</th><th>Long TF</th><th>Short Sig</th><th>Long Sig</th><th>Type</th><th>Severity</th><th>Description</th></tr></thead>
<tbody>{div_rows}</tbody>
</table>

<h2>4. Optimal Timeframe Weights</h2>
{_img("weights")}
<table>
<thead><tr><th>Timeframe</th><th>Weight</th><th>Sharpe</th><th>Hit Rate</th></tr></thead>
<tbody>{wt_rows}</tbody>
</table>

<h2>5. Regime-Dependent Selection</h2>
<table>
<thead><tr><th>Regime</th><th>Best TF</th><th>Weights ({tf_header})</th><th>Ensemble Sharpe</th><th>Obs</th></tr></thead>
<tbody>{regime_rows if regime_rows else '<tr><td colspan="5" style="text-align:center;color:#64748b">No regime data</td></tr>'}</tbody>
</table>

<footer>Generated by <code>compass/multi_timeframe.py</code></footer>
</body></html>"""
        return html
