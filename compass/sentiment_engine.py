"""
NLP sentiment signal engine for trading.

Components:
  - Headline scoring: VADER-style valence + custom financial lexicon
  - Social media sentiment aggregation (bucketed by time)
  - Sentiment momentum (rate of change of composite score)
  - Contrarian signals (extreme readings → fade the crowd)
  - Regime-conditional sentiment (bull sentiment in bear regime = warning)
  - Alpha-combiner integration (outputs signal columns)

HTML report at reports/sentiment_engine.html with sentiment timeline,
signal performance, regime overlay.

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.sentiment_engine import SentimentEngine
    engine = SentimentEngine()
    result = engine.analyze(headlines_df, regimes=regime_series)
    SentimentEngine.generate_report(result)
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "sentiment_engine.html"


# ── Financial lexicon ────────────────────────────────────────────────────

# Positive financial words with intensity weights
POSITIVE_LEXICON: Dict[str, float] = {
    "bullish": 0.8, "rally": 0.7, "surge": 0.8, "soar": 0.9,
    "uptick": 0.4, "gain": 0.5, "growth": 0.5, "beat": 0.6,
    "outperform": 0.7, "upgrade": 0.7, "breakout": 0.6, "strong": 0.5,
    "recovery": 0.6, "boom": 0.8, "profit": 0.5, "earnings beat": 0.7,
    "buy": 0.4, "accumulate": 0.5, "optimistic": 0.6, "upside": 0.6,
    "robust": 0.5, "accelerate": 0.5, "momentum": 0.4, "record high": 0.7,
    "all-time high": 0.8, "dovish": 0.5, "stimulus": 0.5,
}

# Negative financial words
NEGATIVE_LEXICON: Dict[str, float] = {
    "bearish": -0.8, "crash": -0.9, "plunge": -0.8, "selloff": -0.7,
    "sell-off": -0.7, "decline": -0.5, "loss": -0.5, "miss": -0.6,
    "downgrade": -0.7, "underperform": -0.6, "recession": -0.8,
    "default": -0.8, "bankruptcy": -0.9, "fear": -0.6, "panic": -0.8,
    "risk": -0.3, "volatile": -0.4, "uncertainty": -0.5, "weak": -0.5,
    "slump": -0.7, "bear market": -0.8, "hawkish": -0.5, "inflation": -0.4,
    "sell": -0.4, "overvalued": -0.5, "bubble": -0.7, "correction": -0.5,
    "warning": -0.4, "crisis": -0.8, "contagion": -0.7, "downside": -0.6,
}

# Intensifiers and negators
INTENSIFIERS = {"very": 1.3, "extremely": 1.5, "significantly": 1.3, "sharply": 1.4}
NEGATORS = {"not", "no", "never", "neither", "barely", "hardly", "without"}


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class HeadlineScore:
    """Score for a single headline."""

    text: str
    score: float           # -1 to 1
    positive_words: List[str]
    negative_words: List[str]
    timestamp: Optional[Any] = None


@dataclass
class SentimentSnapshot:
    """Aggregated sentiment for a time bucket."""

    timestamp: Any
    mean_score: float
    median_score: float
    n_headlines: int
    pct_positive: float
    pct_negative: float
    composite: float       # weighted composite


@dataclass
class SentimentMomentum:
    """Rate of change of sentiment."""

    timestamp: Any
    momentum_1d: float
    momentum_3d: float
    momentum_7d: float
    acceleration: float    # change of momentum


@dataclass
class ContrarianSignal:
    """Contrarian signal from extreme readings."""

    timestamp: Any
    sentiment_z: float
    signal: str            # "contrarian_bearish", "contrarian_bullish", "neutral"
    strength: float        # 0-1
    threshold: float


@dataclass
class RegimeConditionedSignal:
    """Sentiment conditioned on market regime."""

    timestamp: Any
    sentiment: float
    regime: str
    signal: str            # "confirming", "divergent", "neutral"
    description: str


@dataclass
class AlphaSignalOutput:
    """Output formatted for alpha_combiner integration."""

    dates: pd.DatetimeIndex
    sentiment_signal: pd.Series
    momentum_signal: pd.Series
    contrarian_signal: pd.Series


@dataclass
class SentimentResult:
    """Full result from sentiment analysis."""

    headline_scores: List[HeadlineScore]
    snapshots: List[SentimentSnapshot]
    momentum: List[SentimentMomentum]
    contrarian_signals: List[ContrarianSignal]
    regime_signals: List[RegimeConditionedSignal]
    alpha_output: Optional[AlphaSignalOutput]
    avg_sentiment: float
    n_headlines: int
    n_positive: int
    n_negative: int
    n_neutral: int


# ── Headline scoring ─────────────────────────────────────────────────────


def score_headline(text: str) -> HeadlineScore:
    """Score a single headline using financial lexicon.

    VADER-inspired approach: tokenize, match lexicon, handle negation
    and intensifiers, normalize to [-1, 1].
    """
    lower = text.lower().strip()
    words = re.findall(r"[a-z'-]+", lower)

    pos_found: List[str] = []
    neg_found: List[str] = []
    total_score = 0.0

    # Check multi-word phrases first
    for phrase, weight in POSITIVE_LEXICON.items():
        if " " in phrase and phrase in lower:
            total_score += weight
            pos_found.append(phrase)
    for phrase, weight in NEGATIVE_LEXICON.items():
        if " " in phrase and phrase in lower:
            total_score += weight
            neg_found.append(phrase)

    # Single-word matching with negation/intensifier handling
    i = 0
    while i < len(words):
        word = words[i]

        # Check for intensifier
        intensifier = 1.0
        if word in INTENSIFIERS:
            intensifier = INTENSIFIERS[word]
            i += 1
            if i >= len(words):
                break
            word = words[i]

        # Check for negation in preceding 3 words
        negated = False
        for j in range(max(0, i - 3), i):
            if words[j] in NEGATORS:
                negated = True
                break

        if word in POSITIVE_LEXICON:
            val = POSITIVE_LEXICON[word] * intensifier
            if negated:
                val = -val * 0.5
                neg_found.append(f"not_{word}")
            else:
                pos_found.append(word)
            total_score += val
        elif word in NEGATIVE_LEXICON:
            val = NEGATIVE_LEXICON[word] * intensifier
            if negated:
                val = -val * 0.5
                pos_found.append(f"not_{word}")
            else:
                neg_found.append(word)
            total_score += val

        i += 1

    # Normalize to [-1, 1] using sigmoid-like function
    normalized = total_score / math.sqrt(total_score**2 + 1)

    return HeadlineScore(
        text=text, score=normalized,
        positive_words=pos_found, negative_words=neg_found,
    )


def score_headlines_batch(
    texts: List[str],
    timestamps: Optional[List[Any]] = None,
) -> List[HeadlineScore]:
    """Score a batch of headlines."""
    results = []
    for i, text in enumerate(texts):
        hs = score_headline(text)
        if timestamps and i < len(timestamps):
            hs.timestamp = timestamps[i]
        results.append(hs)
    return results


# ── Sentiment aggregation ────────────────────────────────────────────────


def aggregate_sentiment(
    scores: List[HeadlineScore],
    freq: str = "D",
) -> List[SentimentSnapshot]:
    """Aggregate headline scores into time-bucketed snapshots."""
    if not scores or not any(s.timestamp is not None for s in scores):
        # No timestamps — single snapshot
        if not scores:
            return []
        vals = [s.score for s in scores]
        n_pos = sum(1 for s in scores if s.score > 0.1)
        n_neg = sum(1 for s in scores if s.score < -0.1)
        n = len(scores)
        return [SentimentSnapshot(
            timestamp=None, mean_score=float(np.mean(vals)),
            median_score=float(np.median(vals)), n_headlines=n,
            pct_positive=n_pos / n, pct_negative=n_neg / n,
            composite=float(np.mean(vals)),
        )]

    df = pd.DataFrame({
        "timestamp": [s.timestamp for s in scores],
        "score": [s.score for s in scores],
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    snapshots: List[SentimentSnapshot] = []
    for period, grp in df.resample(freq):
        if grp.empty:
            continue
        vals = grp["score"].values
        n = len(vals)
        n_pos = int((vals > 0.1).sum())
        n_neg = int((vals < -0.1).sum())
        # Composite: weighted by recency within bucket
        weights = np.linspace(0.5, 1.0, n)
        composite = float(np.average(vals, weights=weights))

        snapshots.append(SentimentSnapshot(
            timestamp=period, mean_score=float(vals.mean()),
            median_score=float(np.median(vals)), n_headlines=n,
            pct_positive=n_pos / n, pct_negative=n_neg / n,
            composite=composite,
        ))
    return snapshots


# ── Sentiment momentum ───────────────────────────────────────────────────


def compute_sentiment_momentum(
    snapshots: List[SentimentSnapshot],
) -> List[SentimentMomentum]:
    """Compute rate of change of sentiment composite."""
    if len(snapshots) < 2:
        return []

    composites = np.array([s.composite for s in snapshots])
    n = len(composites)
    results: List[SentimentMomentum] = []

    for i in range(1, n):
        m1d = composites[i] - composites[i - 1]
        m3d = composites[i] - composites[max(0, i - 3)] if i >= 3 else m1d
        m7d = composites[i] - composites[max(0, i - 7)] if i >= 7 else m3d

        # Acceleration: change of 1d momentum
        prev_m1d = composites[i - 1] - composites[max(0, i - 2)] if i >= 2 else 0.0
        accel = m1d - prev_m1d

        results.append(SentimentMomentum(
            timestamp=snapshots[i].timestamp,
            momentum_1d=m1d, momentum_3d=m3d, momentum_7d=m7d,
            acceleration=accel,
        ))
    return results


# ── Contrarian signals ───────────────────────────────────────────────────


def compute_contrarian_signals(
    snapshots: List[SentimentSnapshot],
    z_threshold: float = 2.0,
    lookback: int = 20,
) -> List[ContrarianSignal]:
    """Generate contrarian signals from extreme sentiment readings."""
    if len(snapshots) < lookback:
        return []

    composites = np.array([s.composite for s in snapshots])
    signals: List[ContrarianSignal] = []

    for i in range(lookback, len(composites)):
        window = composites[i - lookback:i]
        mu = window.mean()
        std = window.std()
        if std < 1e-12:
            continue

        z = (composites[i] - mu) / std

        if z > z_threshold:
            # Extreme bullish → contrarian bearish
            strength = min(1.0, (z - z_threshold) / z_threshold)
            signals.append(ContrarianSignal(
                timestamp=snapshots[i].timestamp, sentiment_z=z,
                signal="contrarian_bearish", strength=strength,
                threshold=z_threshold,
            ))
        elif z < -z_threshold:
            # Extreme bearish → contrarian bullish
            strength = min(1.0, (-z - z_threshold) / z_threshold)
            signals.append(ContrarianSignal(
                timestamp=snapshots[i].timestamp, sentiment_z=z,
                signal="contrarian_bullish", strength=strength,
                threshold=z_threshold,
            ))

    return signals


# ── Regime-conditioned sentiment ─────────────────────────────────────────


def regime_condition_sentiment(
    snapshots: List[SentimentSnapshot],
    regimes: pd.Series,
) -> List[RegimeConditionedSignal]:
    """Assess sentiment relative to market regime.

    Divergence (bullish sentiment in bear market) is a signal.
    """
    if not snapshots:
        return []

    signals: List[RegimeConditionedSignal] = []

    for snap in snapshots:
        if snap.timestamp is None:
            continue

        ts = pd.Timestamp(snap.timestamp)
        # Find nearest regime
        if regimes.empty:
            continue
        idx = regimes.index.get_indexer([ts], method="nearest")
        if idx[0] < 0 or idx[0] >= len(regimes):
            continue
        regime = str(regimes.iloc[idx[0]])

        sentiment = snap.composite

        # Classification
        if regime in ("bear", "crisis") and sentiment > 0.3:
            signal = "divergent"
            desc = f"Bullish sentiment ({sentiment:.2f}) in {regime} regime — potential trap"
        elif regime in ("bull", "recovery") and sentiment < -0.3:
            signal = "divergent"
            desc = f"Bearish sentiment ({sentiment:.2f}) in {regime} regime — potential opportunity"
        elif (regime in ("bull",) and sentiment > 0.2) or (regime in ("bear",) and sentiment < -0.2):
            signal = "confirming"
            desc = f"Sentiment confirms {regime} regime"
        else:
            signal = "neutral"
            desc = f"Neutral sentiment in {regime} regime"

        signals.append(RegimeConditionedSignal(
            timestamp=snap.timestamp, sentiment=sentiment,
            regime=regime, signal=signal, description=desc,
        ))

    return signals


# ── Alpha combiner integration ───────────────────────────────────────────


def build_alpha_signals(
    snapshots: List[SentimentSnapshot],
    momentum: List[SentimentMomentum],
    contrarian: List[ContrarianSignal],
) -> Optional[AlphaSignalOutput]:
    """Format signals for alpha_combiner integration."""
    if not snapshots or snapshots[0].timestamp is None:
        return None

    dates = pd.DatetimeIndex([s.timestamp for s in snapshots if s.timestamp is not None])
    sent_vals = pd.Series(
        [s.composite for s in snapshots if s.timestamp is not None],
        index=dates, name="sentiment",
    )

    # Momentum signal
    mom_vals = pd.Series(0.0, index=dates, name="sentiment_momentum")
    for m in momentum:
        if m.timestamp in dates:
            mom_vals[m.timestamp] = m.momentum_3d

    # Contrarian signal
    con_vals = pd.Series(0.0, index=dates, name="contrarian")
    for c in contrarian:
        if c.timestamp in dates:
            val = c.strength if c.signal == "contrarian_bullish" else -c.strength
            con_vals[c.timestamp] = val

    return AlphaSignalOutput(
        dates=dates,
        sentiment_signal=sent_vals,
        momentum_signal=mom_vals,
        contrarian_signal=con_vals,
    )


# ── Core engine ──────────────────────────────────────────────────────────


class SentimentEngine:
    """NLP sentiment signal engine for trading."""

    def __init__(
        self,
        freq: str = "D",
        contrarian_z: float = 2.0,
        contrarian_lookback: int = 20,
    ):
        self.freq = freq
        self.contrarian_z = contrarian_z
        self.contrarian_lookback = contrarian_lookback

    def analyze(
        self,
        headlines: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        text_col: str = "headline",
        time_col: str = "timestamp",
    ) -> SentimentResult:
        """Run full sentiment analysis.

        Expected columns: headline (text), timestamp (optional).
        """
        if text_col not in headlines.columns:
            raise ValueError(f"Missing required column: {text_col}")

        texts = headlines[text_col].tolist()
        timestamps = headlines[time_col].tolist() if time_col in headlines.columns else None

        # Score
        scored = score_headlines_batch(texts, timestamps)

        # Aggregate
        snapshots = aggregate_sentiment(scored, self.freq)

        # Momentum
        momentum = compute_sentiment_momentum(snapshots)

        # Contrarian
        contrarian = compute_contrarian_signals(
            snapshots, self.contrarian_z, self.contrarian_lookback
        )

        # Regime conditioning
        regime_sigs: List[RegimeConditionedSignal] = []
        if regimes is not None:
            regime_sigs = regime_condition_sentiment(snapshots, regimes)

        # Alpha output
        alpha_out = build_alpha_signals(snapshots, momentum, contrarian)

        # Summary
        n = len(scored)
        n_pos = sum(1 for s in scored if s.score > 0.1)
        n_neg = sum(1 for s in scored if s.score < -0.1)
        n_neutral = n - n_pos - n_neg
        avg = float(np.mean([s.score for s in scored])) if scored else 0.0

        return SentimentResult(
            headline_scores=scored,
            snapshots=snapshots,
            momentum=momentum,
            contrarian_signals=contrarian,
            regime_signals=regime_sigs,
            alpha_output=alpha_out,
            avg_sentiment=avg,
            n_headlines=n,
            n_positive=n_pos,
            n_negative=n_neg,
            n_neutral=n_neutral,
        )

    @staticmethod
    def generate_report(
        result: SentimentResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _f(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}"


def _fp(v: float) -> str:
    return f"{v:.1%}"


def _svg_line(values: List[float], title: str, color: str = "#58a6ff",
              w: int = 700, h: int = 200) -> str:
    if len(values) < 2:
        return ""
    n = len(values)
    pad = 55
    pw = w - 2 * pad
    ph = h - 65
    y_min = min(min(values), -0.1)
    y_max = max(max(values), 0.1)
    if y_max <= y_min:
        y_max = y_min + 0.01

    def tx(i): return pad + i / max(n - 1, 1) * pw
    def ty(v): return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">{title}</text>')
    if y_min < 0 < y_max:
        zy = ty(0)
        parts.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" stroke="#30363d" stroke-dasharray="3,3"/>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(values[i]):.1f}" for i in range(n))
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _contrarian_markers(contrarian: List[ContrarianSignal], n_snapshots: int) -> str:
    if not contrarian or n_snapshots == 0:
        return ""
    items = "".join(
        f"<li><strong>{c.signal}</strong> (z={_f(c.sentiment_z, 2)}, strength={_f(c.strength, 2)})</li>"
        for c in contrarian[-10:]
    )
    return f'<div class="card"><h3>Contrarian Signals (recent)</h3><ul>{items}</ul></div>'


def _regime_table(regime_sigs: List[RegimeConditionedSignal]) -> str:
    if not regime_sigs:
        return ""
    rows = ""
    for r in regime_sigs[-20:]:
        color = "#d29922" if r.signal == "divergent" else "#3fb950" if r.signal == "confirming" else "#8b949e"
        rows += f'<tr><td>{r.regime}</td><td>{_f(r.sentiment, 2)}</td><td style="color:{color}">{r.signal}</td><td style="text-align:left">{r.description}</td></tr>'
    return f"""<h2>Regime-Conditioned Signals</h2>
    <table class="data-table"><tr><th>Regime</th><th>Sentiment</th><th>Signal</th><th style="text-align:left">Description</th></tr>{rows}</table>"""


def _build_html(result: SentimentResult) -> str:
    sentiment_vals = [s.composite for s in result.snapshots]
    momentum_vals = [m.momentum_3d for m in result.momentum]

    sent_color = "#3fb950" if result.avg_sentiment > 0.1 else "#f85149" if result.avg_sentiment < -0.1 else "#8b949e"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Sentiment Engine Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
              gap: 12px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 12px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.8em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.15em; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  ul {{ color: #c9d1d9; }}
</style>
</head>
<body>
<h1>Sentiment Engine</h1>
<p class="meta">{result.n_headlines} headlines analyzed &middot;
   {len(result.snapshots)} time buckets &middot;
   {len(result.contrarian_signals)} contrarian signals</p>

<div class="summary">
  <div class="stat"><div class="label">Avg Sentiment</div>
    <div class="value" style="color:{sent_color}">{_f(result.avg_sentiment, 3)}</div></div>
  <div class="stat"><div class="label">Positive</div>
    <div class="value">{result.n_positive} ({_fp(result.n_positive / max(result.n_headlines, 1))})</div></div>
  <div class="stat"><div class="label">Negative</div>
    <div class="value">{result.n_negative} ({_fp(result.n_negative / max(result.n_headlines, 1))})</div></div>
  <div class="stat"><div class="label">Neutral</div>
    <div class="value">{result.n_neutral}</div></div>
  <div class="stat"><div class="label">Contrarian</div>
    <div class="value">{len(result.contrarian_signals)}</div></div>
</div>

<h2>Sentiment Timeline</h2>
{_svg_line(sentiment_vals, "Composite Sentiment", "#58a6ff")}

<h2>Sentiment Momentum (3d)</h2>
{_svg_line(momentum_vals, "Sentiment Momentum", "#d29922")}

{_contrarian_markers(result.contrarian_signals, len(result.snapshots))}

{_regime_table(result.regime_signals)}

</body>
</html>"""
