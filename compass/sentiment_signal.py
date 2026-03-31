"""
Multi-source sentiment signal aggregator.

Text sentiment (VADER-style + financial lexicon), social media volume/velocity,
news sentiment with exponential decay, put-call ratio, VIX term structure
(contango/backwardation), composite sentiment index (0-100), contrarian
signals at extreme readings, and regime-conditional sentiment alpha.

Generates an HTML report at reports/sentiment_signal.html.

Usage::

    from compass.sentiment_signal import SentimentAggregator
    agg = SentimentAggregator(market_data, texts=texts_df)
    results = agg.analyze()
    agg.generate_report()
"""

from __future__ import annotations

import base64
import io
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "sentiment_signal.html"

# ── Financial lexicon ───────────────────────────────────────────────────

_POSITIVE = {
    "bullish", "upgrade", "beat", "outperform", "rally", "breakout",
    "strong", "surge", "recovery", "growth", "gain", "profit", "buy",
    "positive", "optimistic", "upside", "momentum", "accumulate",
}
_NEGATIVE = {
    "bearish", "downgrade", "miss", "underperform", "crash", "breakdown",
    "weak", "plunge", "recession", "decline", "loss", "sell", "negative",
    "pessimistic", "downside", "risk", "fear", "capitulation", "default",
}

REGIMES = ("bull", "bear", "high_vol", "neutral")


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class TextSentiment:
    """Sentiment score for a single text."""
    text: str
    score: float           # -1 to +1
    positive_words: int
    negative_words: int
    source: str
    timestamp: str


@dataclass
class SourceSentiment:
    """Aggregated sentiment from one source."""
    source: str
    mean_score: float
    n_items: int
    velocity: float        # items per period
    trend: str             # "improving", "deteriorating", "stable"


@dataclass
class PCRSentiment:
    """Put-call ratio sentiment."""
    pcr: float
    percentile: float
    signal: str            # "bullish" (high PCR), "bearish" (low), "neutral"
    z_score: float


@dataclass
class VIXTermSentiment:
    """VIX term structure sentiment."""
    front: float
    back: float
    ratio: float           # front/back
    structure: str          # "contango" or "backwardation"
    signal: str            # "complacent", "fear", "neutral"
    z_score: float


@dataclass
class CompositeSentiment:
    """Composite sentiment index."""
    index: float           # 0-100 (0=extreme fear, 100=extreme greed)
    text_component: float
    pcr_component: float
    vix_component: float
    volume_component: float
    regime: str


@dataclass
class ContrarianSignal:
    """Contrarian trading signal at extreme sentiment."""
    direction: str         # "buy" or "sell"
    strength: float        # 0-1
    trigger: str           # which component triggered
    composite_at_trigger: float
    description: str


@dataclass
class SentimentAlpha:
    """Regime-conditional sentiment alpha."""
    regime: str
    alpha: float           # correlation of sentiment with forward returns
    sharpe: float
    n_obs: int
    signal_weight: float


@dataclass
class ExtremeEvent:
    """Extreme sentiment reading."""
    date: str
    composite: float
    direction: str         # "fear" or "greed"
    description: str


# ── Text scoring ────────────────────────────────────────────────────────


def score_text(text: str) -> Tuple[float, int, int]:
    """Score text using financial lexicon. Returns (score, n_pos, n_neg)."""
    words = set(re.findall(r'[a-z]+', text.lower()))
    n_pos = len(words & _POSITIVE)
    n_neg = len(words & _NEGATIVE)
    total = n_pos + n_neg
    if total == 0:
        return 0.0, 0, 0
    score = (n_pos - n_neg) / total
    return float(score), n_pos, n_neg


# ── Aggregator ──────────────────────────────────────────────────────────


class SentimentAggregator:
    """Aggregate sentiment from multiple sources."""

    def __init__(
        self,
        market_data: Optional[pd.DataFrame] = None,
        texts: Optional[pd.DataFrame] = None,
        pcr_series: Optional[pd.Series] = None,
        vix_front: Optional[pd.Series] = None,
        vix_back: Optional[pd.Series] = None,
        volume: Optional[pd.Series] = None,
        returns: Optional[pd.Series] = None,
        regimes: Optional[pd.Series] = None,
        decay_halflife: int = 5,
        extreme_threshold: float = 15.0,
    ) -> None:
        self.market_data = market_data
        self.texts = texts
        self.pcr_series = pcr_series
        self.vix_front = vix_front
        self.vix_back = vix_back
        self.volume = volume
        self.returns = returns
        self.regimes = regimes
        self.decay_halflife = decay_halflife
        self.extreme_threshold = extreme_threshold

        # Results
        self.text_sentiments: List[TextSentiment] = []
        self.source_sentiments: List[SourceSentiment] = []
        self.pcr_sentiment: Optional[PCRSentiment] = None
        self.vix_sentiment: Optional[VIXTermSentiment] = None
        self.composite: Optional[CompositeSentiment] = None
        self.contrarian_signals: List[ContrarianSignal] = []
        self.sentiment_alpha: List[SentimentAlpha] = []
        self.extreme_events: List[ExtremeEvent] = []
        self.composite_history: List[Tuple[str, float]] = []

    @classmethod
    def from_csv(
        cls, market_path: Optional[str] = None,
        texts_path: Optional[str] = None, **kwargs: Any,
    ) -> "SentimentAggregator":
        market = pd.read_csv(market_path, index_col=0, parse_dates=True) if market_path else None
        texts = pd.read_csv(texts_path, parse_dates=True) if texts_path else None
        return cls(market_data=market, texts=texts, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        self.text_sentiments = self._score_texts()
        self.source_sentiments = self._aggregate_sources()
        self.pcr_sentiment = self._pcr_signal()
        self.vix_sentiment = self._vix_term_signal()
        self.composite, self.composite_history = self._composite_index()
        self.contrarian_signals = self._contrarian_signals()
        self.extreme_events = self._detect_extremes()
        self.sentiment_alpha = self._regime_alpha()
        return {
            "text_sentiments": self.text_sentiments,
            "source_sentiments": self.source_sentiments,
            "pcr_sentiment": self.pcr_sentiment,
            "vix_sentiment": self.vix_sentiment,
            "composite": self.composite,
            "contrarian_signals": self.contrarian_signals,
            "sentiment_alpha": self.sentiment_alpha,
            "extreme_events": self.extreme_events,
        }

    # ── Text sentiment ──────────────────────────────────────────────────

    def _score_texts(self) -> List[TextSentiment]:
        if self.texts is None or self.texts.empty:
            return []
        results: List[TextSentiment] = []
        text_col = "text" if "text" in self.texts.columns else self.texts.columns[0]
        source_col = "source" if "source" in self.texts.columns else None
        time_col = "timestamp" if "timestamp" in self.texts.columns else None

        for _, row in self.texts.iterrows():
            txt = str(row[text_col])
            score, n_pos, n_neg = score_text(txt)
            results.append(TextSentiment(
                text=txt[:200], score=score,
                positive_words=n_pos, negative_words=n_neg,
                source=str(row[source_col]) if source_col else "unknown",
                timestamp=str(row[time_col]) if time_col else "",
            ))
        return results

    def _aggregate_sources(self) -> List[SourceSentiment]:
        if not self.text_sentiments:
            return []
        by_source: Dict[str, List[TextSentiment]] = {}
        for ts in self.text_sentiments:
            by_source.setdefault(ts.source, []).append(ts)

        results: List[SourceSentiment] = []
        for source, items in sorted(by_source.items()):
            scores = [t.score for t in items]
            mean = float(np.mean(scores))
            n = len(items)
            # Velocity: items per unit (assume each is one period)
            velocity = float(n)
            # Trend from first half vs second half
            if n >= 4:
                first = np.mean(scores[: n // 2])
                second = np.mean(scores[n // 2:])
                diff = second - first
                trend = "improving" if diff > 0.1 else "deteriorating" if diff < -0.1 else "stable"
            else:
                trend = "stable"
            results.append(SourceSentiment(
                source=source, mean_score=mean, n_items=n,
                velocity=velocity, trend=trend,
            ))
        return results

    # ── Put-call ratio ──────────────────────────────────────────────────

    def _pcr_signal(self) -> Optional[PCRSentiment]:
        if self.pcr_series is None or len(self.pcr_series) < 5:
            return None
        pcr = self.pcr_series.dropna()
        current = float(pcr.iloc[-1])
        mean = float(pcr.mean())
        std = float(pcr.std())
        z = (current - mean) / std if std > 1e-10 else 0.0
        pct = float((pcr < current).mean())

        # High PCR = lots of puts = fear = contrarian bullish
        if z > 1.0:
            signal = "bullish"
        elif z < -1.0:
            signal = "bearish"
        else:
            signal = "neutral"

        return PCRSentiment(pcr=current, percentile=pct, signal=signal, z_score=z)

    # ── VIX term structure ──────────────────────────────────────────────

    def _vix_term_signal(self) -> Optional[VIXTermSentiment]:
        if self.vix_front is None or self.vix_back is None:
            return None
        if len(self.vix_front) < 5 or len(self.vix_back) < 5:
            return None
        front = float(self.vix_front.iloc[-1])
        back = float(self.vix_back.iloc[-1])
        ratio = front / back if back > 0 else 1.0

        structure = "backwardation" if ratio > 1.0 else "contango"

        # Historical z-score of ratio
        hist_ratio = (self.vix_front / self.vix_back).dropna()
        mean_r = float(hist_ratio.mean())
        std_r = float(hist_ratio.std())
        z = (ratio - mean_r) / std_r if std_r > 1e-10 else 0.0

        if ratio > 1.05:
            signal = "fear"
        elif ratio < 0.95:
            signal = "complacent"
        else:
            signal = "neutral"

        return VIXTermSentiment(
            front=front, back=back, ratio=ratio,
            structure=structure, signal=signal, z_score=z,
        )

    # ── Composite index ─────────────────────────────────────────────────

    def _composite_index(self) -> Tuple[Optional[CompositeSentiment], List[Tuple[str, float]]]:
        # Text component: mean of decayed scores → scale to 0-100
        text_comp = 50.0
        if self.text_sentiments:
            n = len(self.text_sentiments)
            decay = np.exp(-np.log(2) / max(self.decay_halflife, 1) * np.arange(n)[::-1])
            decay /= decay.sum()
            scores = np.array([t.score for t in self.text_sentiments])
            text_comp = float((scores * decay).sum() * 50 + 50)

        # PCR component
        pcr_comp = 50.0
        if self.pcr_sentiment:
            pcr_comp = float(np.clip(self.pcr_sentiment.z_score * -15 + 50, 0, 100))

        # VIX component
        vix_comp = 50.0
        if self.vix_sentiment:
            vix_comp = float(np.clip(self.vix_sentiment.z_score * -15 + 50, 0, 100))

        # Volume component
        vol_comp = 50.0
        if self.volume is not None and len(self.volume) >= 20:
            v = self.volume.dropna()
            z = (v.iloc[-1] - v.mean()) / v.std() if v.std() > 0 else 0
            vol_comp = float(np.clip(z * -10 + 50, 0, 100))

        composite = float(np.clip(
            0.30 * text_comp + 0.25 * pcr_comp + 0.25 * vix_comp + 0.20 * vol_comp,
            0, 100,
        ))

        regime = "neutral"
        if self.regimes is not None and len(self.regimes) > 0:
            regime = str(self.regimes.iloc[-1])

        # Build history
        history: List[Tuple[str, float]] = []
        if self.text_sentiments:
            for i, ts in enumerate(self.text_sentiments):
                # Rough composite using just text for history
                score_01 = ts.score * 50 + 50
                history.append((ts.timestamp or str(i), float(np.clip(score_01, 0, 100))))

        comp = CompositeSentiment(
            index=composite, text_component=text_comp,
            pcr_component=pcr_comp, vix_component=vix_comp,
            volume_component=vol_comp, regime=regime,
        )
        return comp, history

    # ── Contrarian signals ──────────────────────────────────────────────

    def _contrarian_signals(self) -> List[ContrarianSignal]:
        signals: List[ContrarianSignal] = []
        if self.composite is None:
            return signals

        c = self.composite.index
        if c < self.extreme_threshold:
            signals.append(ContrarianSignal(
                direction="buy", strength=min((self.extreme_threshold - c) / self.extreme_threshold, 1.0),
                trigger="composite", composite_at_trigger=c,
                description=f"Extreme fear (composite={c:.0f}): contrarian buy",
            ))
        elif c > (100 - self.extreme_threshold):
            signals.append(ContrarianSignal(
                direction="sell", strength=min((c - (100 - self.extreme_threshold)) / self.extreme_threshold, 1.0),
                trigger="composite", composite_at_trigger=c,
                description=f"Extreme greed (composite={c:.0f}): contrarian sell",
            ))

        # PCR extreme
        if self.pcr_sentiment and abs(self.pcr_sentiment.z_score) > 2.0:
            direction = "buy" if self.pcr_sentiment.z_score > 0 else "sell"
            signals.append(ContrarianSignal(
                direction=direction,
                strength=min(abs(self.pcr_sentiment.z_score) / 4.0, 1.0),
                trigger="pcr", composite_at_trigger=c,
                description=f"PCR z={self.pcr_sentiment.z_score:.1f}: contrarian {direction}",
            ))

        return signals

    # ── Extreme events ──────────────────────────────────────────────────

    def _detect_extremes(self) -> List[ExtremeEvent]:
        events: List[ExtremeEvent] = []
        for date, val in self.composite_history:
            if val < self.extreme_threshold:
                events.append(ExtremeEvent(date=date, composite=val, direction="fear",
                                           description=f"Extreme fear: {val:.0f}"))
            elif val > 100 - self.extreme_threshold:
                events.append(ExtremeEvent(date=date, composite=val, direction="greed",
                                           description=f"Extreme greed: {val:.0f}"))
        return events

    # ── Regime-conditional alpha ────────────────────────────────────────

    def _regime_alpha(self) -> List[SentimentAlpha]:
        if self.returns is None or not self.composite_history:
            return []
        results: List[SentimentAlpha] = []
        # Build sentiment series from history
        sent_vals = [v for _, v in self.composite_history]
        if len(sent_vals) < 10:
            return []

        sent_series = pd.Series(sent_vals[-len(self.returns):])
        if len(sent_series) != len(self.returns):
            min_len = min(len(sent_series), len(self.returns))
            sent_series = sent_series.iloc[-min_len:].reset_index(drop=True)
            ret = self.returns.iloc[-min_len:].reset_index(drop=True)
        else:
            ret = self.returns.reset_index(drop=True)

        # Overall alpha
        if len(sent_series) > 5 and sent_series.std() > 0 and ret.std() > 0:
            corr = float(sent_series.corr(ret))
            strat_ret = np.sign(sent_series - 50) * ret
            sh = float(strat_ret.mean() / strat_ret.std() * np.sqrt(252)) if strat_ret.std() > 0 else 0.0
            results.append(SentimentAlpha(
                regime="overall", alpha=corr, sharpe=sh,
                n_obs=len(ret), signal_weight=1.0,
            ))

        # Per-regime
        if self.regimes is not None:
            for regime in REGIMES:
                mask = self.regimes.iloc[-len(ret):].reset_index(drop=True) == regime
                if mask.sum() < 5:
                    continue
                s_r = sent_series[mask]
                r_r = ret[mask]
                if s_r.std() > 0 and r_r.std() > 0:
                    corr = float(s_r.corr(r_r))
                    sr = np.sign(s_r - 50) * r_r
                    sh = float(sr.mean() / sr.std() * np.sqrt(252)) if sr.std() > 0 else 0.0
                    results.append(SentimentAlpha(
                        regime=regime, alpha=corr, sharpe=sh,
                        n_obs=int(mask.sum()), signal_weight=max(abs(corr), 0.1),
                    ))

        return results

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.composite is None:
            self.analyze()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out.resolve())

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        charts: Dict[str, str] = {}
        charts["timeline"] = self._chart_timeline()
        charts["sources"] = self._chart_sources()
        charts["alpha"] = self._chart_alpha()
        return charts

    def _chart_timeline(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.composite_history:
            return ""
        vals = [v for _, v in self.composite_history]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        xs = range(len(vals))
        for i in range(len(vals) - 1):
            c = "#16a34a" if vals[i] > 60 else "#dc2626" if vals[i] < 40 else "#f59e0b"
            ax.plot([xs[i], xs[i+1]], [vals[i], vals[i+1]], color=c, lw=1.2)
        ax.axhline(50, color="black", lw=0.5, ls="--")
        ax.axhline(self.extreme_threshold, color="#dc2626", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(100 - self.extreme_threshold, color="#16a34a", lw=0.5, ls="--", alpha=0.5)
        ax.set_ylim(0, 100); ax.set_ylabel("Sentiment Index"); ax.set_title("Sentiment Timeline", fontsize=11)
        ax.grid(True, alpha=0.2); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_sources(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.source_sentiments:
            return ""
        names = [s.source for s in self.source_sentiments]
        scores = [s.mean_score for s in self.source_sentiments]
        colors = ["#16a34a" if s > 0 else "#dc2626" for s in scores]
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.4)))
        ax.barh(names, scores, color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Mean Sentiment"); ax.set_title("Source Breakdown", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_alpha(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.sentiment_alpha:
            return ""
        names = [a.regime for a in self.sentiment_alpha]
        sharpes = [a.sharpe for a in self.sentiment_alpha]
        colors = ["#16a34a" if s > 0 else "#dc2626" for s in sharpes]
        fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.4)))
        ax.barh(names, sharpes, color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("Sharpe"); ax.set_title("Sentiment Alpha by Regime", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        c = self.composite or CompositeSentiment(50, 50, 50, 50, 50, "neutral")
        idx_cls = "good" if c.index > 60 else "bad" if c.index < 40 else ""

        src_rows = ""
        for s in self.source_sentiments:
            cls = "good" if s.mean_score > 0.1 else "bad" if s.mean_score < -0.1 else ""
            src_rows += f'<tr><td>{s.source}</td><td class="{cls}">{s.mean_score:+.3f}</td><td>{s.n_items}</td><td>{s.trend}</td></tr>\n'

        ctr_rows = ""
        for cs in self.contrarian_signals:
            cls = "good" if cs.direction == "buy" else "bad"
            ctr_rows += f'<tr><td class="{cls}">{cs.direction.upper()}</td><td>{cs.strength:.0%}</td><td>{cs.trigger}</td><td>{cs.description}</td></tr>\n'
        if not ctr_rows:
            ctr_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">No extreme signals</td></tr>'

        ext_rows = ""
        for e in self.extreme_events[-20:]:
            cls = "bad" if e.direction == "fear" else "good"
            ext_rows += f'<tr><td>{e.date}</td><td class="{cls}">{e.composite:.0f}</td><td>{e.direction}</td><td>{e.description}</td></tr>\n'
        if not ext_rows:
            ext_rows = '<tr><td colspan="4" style="text-align:center;color:#64748b">No extreme events</td></tr>'

        alpha_rows = ""
        for a in self.sentiment_alpha:
            alpha_rows += f'<tr><td>{a.regime}</td><td>{a.alpha:.3f}</td><td>{a.sharpe:.2f}</td><td>{a.n_obs}</td></tr>\n'

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Sentiment Signal Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em 1.5em; min-width:120px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }} .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:8px 10px; text-align:left; border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:right; }} td:first-child {{ text-align:left; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em; margin:1.5em 0; text-align:center; }}
  .chart img {{ max-width:100%; height:auto; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.8em; color:#94a3b8; }}
</style></head><body>
<h1>Sentiment Signal Dashboard</h1>
<div class="meta">{len(self.text_sentiments)} texts &middot; {len(self.source_sentiments)} sources &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value {idx_cls}">{c.index:.0f}</div><div class="label">Composite Index</div></div>
  <div class="kpi"><div class="value">{c.text_component:.0f}</div><div class="label">Text</div></div>
  <div class="kpi"><div class="value">{c.pcr_component:.0f}</div><div class="label">PCR</div></div>
  <div class="kpi"><div class="value">{c.vix_component:.0f}</div><div class="label">VIX Term</div></div>
  <div class="kpi"><div class="value">{len(self.contrarian_signals)}</div><div class="label">Contrarian Signals</div></div>
</div>
<h2>1. Sentiment Timeline</h2>{_img("timeline")}
<h2>2. Source Breakdown</h2>{_img("sources")}
<table><thead><tr><th>Source</th><th>Mean Score</th><th>Items</th><th>Trend</th></tr></thead><tbody>{src_rows}</tbody></table>
<h2>3. Contrarian Signals</h2>
<table><thead><tr><th>Direction</th><th>Strength</th><th>Trigger</th><th>Description</th></tr></thead><tbody>{ctr_rows}</tbody></table>
<h2>4. Extreme Events</h2>
<table><thead><tr><th>Date</th><th>Composite</th><th>Direction</th><th>Description</th></tr></thead><tbody>{ext_rows}</tbody></table>
<h2>5. Sentiment Alpha</h2>{_img("alpha")}
<table><thead><tr><th>Regime</th><th>Alpha</th><th>Sharpe</th><th>Obs</th></tr></thead><tbody>{alpha_rows}</tbody></table>
<footer>Generated by <code>compass/sentiment_signal.py</code></footer>
</body></html>"""
        return html
