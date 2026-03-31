"""Tests for compass/sentiment_signal.py — sentiment signal aggregator."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from compass.sentiment_signal import (
    CompositeSentiment, ContrarianSignal, ExtremeEvent,
    PCRSentiment, SentimentAggregator, SentimentAlpha,
    SourceSentiment, TextSentiment, VIXTermSentiment,
    score_text,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_texts(n=50, seed=42):
    rng = np.random.RandomState(seed)
    sources = rng.choice(["twitter", "news", "reddit"], n)
    words_pos = ["bullish rally breakout strong upgrade", "growth surge positive momentum"]
    words_neg = ["bearish crash plunge weak recession", "decline fear sell capitulation"]
    words_neut = ["market traded volume options expiry", "sector reported quarterly data"]
    pool = words_pos + words_neg + words_neut
    texts = [pool[rng.randint(0, len(pool))] for _ in range(n)]
    return pd.DataFrame({
        "text": texts, "source": sources,
        "timestamp": pd.date_range("2024-06-01", periods=n, freq="1h"),
    })

def _make_pcr(n=100, seed=42):
    rng = np.random.RandomState(seed)
    return pd.Series(rng.uniform(0.6, 1.4, n),
                     index=pd.bdate_range("2024-01-01", periods=n))

def _make_vix(n=100, seed=42):
    rng = np.random.RandomState(seed)
    front = pd.Series(18 + rng.normal(0, 3, n).cumsum() * 0.1,
                      index=pd.bdate_range("2024-01-01", periods=n))
    back = front * rng.uniform(0.9, 1.1, n)
    return front, back

def _make_agg(n=50, seed=42, **kwargs):
    texts = _make_texts(n, seed)
    pcr = _make_pcr(100, seed)
    vf, vb = _make_vix(100, seed)
    vol = pd.Series(np.random.RandomState(seed).uniform(5000, 20000, 100),
                    index=pcr.index)
    ret = pd.Series(np.random.RandomState(seed).normal(0.0003, 0.01, 100),
                    index=pcr.index)
    reg = pd.Series(["bull"]*25 + ["neutral"]*25 + ["bear"]*25 + ["high_vol"]*25,
                    index=pcr.index)
    return SentimentAggregator(texts=texts, pcr_series=pcr, vix_front=vf,
                                vix_back=vb, volume=vol, returns=ret,
                                regimes=reg, **kwargs)

# ── Dataclasses ──────────────────────────────────────────────────────────

class TestDataclasses:
    def test_text_sentiment(self):
        t = TextSentiment("test", 0.5, 3, 1, "news", "2024-01-01")
        assert t.score == pytest.approx(0.5)
    def test_source_sentiment(self):
        s = SourceSentiment("news", 0.3, 20, 20.0, "improving")
        assert s.n_items == 20
    def test_pcr_sentiment(self):
        p = PCRSentiment(1.2, 0.8, "bullish", 1.5)
        assert p.signal == "bullish"
    def test_vix_sentiment(self):
        v = VIXTermSentiment(20, 18, 1.11, "backwardation", "fear", 1.5)
        assert v.structure == "backwardation"
    def test_composite(self):
        c = CompositeSentiment(35, 30, 40, 45, 25, "bear")
        assert c.index == pytest.approx(35)
    def test_contrarian(self):
        cs = ContrarianSignal("buy", 0.8, "composite", 10, "desc")
        assert cs.direction == "buy"
    def test_extreme(self):
        e = ExtremeEvent("2024-01-01", 8, "fear", "desc")
        assert e.direction == "fear"

# ── Text scoring ─────────────────────────────────────────────────────────

class TestTextScoring:
    def test_positive_text(self):
        s, p, n = score_text("bullish rally strong upgrade growth")
        assert s > 0 and p > 0
    def test_negative_text(self):
        s, p, n = score_text("bearish crash plunge fear recession")
        assert s < 0 and n > 0
    def test_neutral_text(self):
        s, p, n = score_text("today market open close data")
        assert s == 0.0 and p == 0 and n == 0
    def test_mixed_text(self):
        s, p, n = score_text("bullish crash")
        assert p == 1 and n == 1 and s == pytest.approx(0.0)
    def test_score_range(self):
        s, _, _ = score_text("bullish upgrade strong growth surge")
        assert -1 <= s <= 1

# ── PCR signal ───────────────────────────────────────────────────────────

class TestPCR:
    def test_pcr_returns_result(self):
        agg = _make_agg()
        agg.analyze()
        assert agg.pcr_sentiment is not None
    def test_pcr_signal_valid(self):
        agg = _make_agg()
        agg.analyze()
        assert agg.pcr_sentiment.signal in ("bullish", "bearish", "neutral")
    def test_no_pcr_returns_none(self):
        agg = SentimentAggregator()
        agg.analyze()
        assert agg.pcr_sentiment is None

# ── VIX term ─────────────────────────────────────────────────────────────

class TestVIXTerm:
    def test_vix_returns_result(self):
        agg = _make_agg()
        agg.analyze()
        assert agg.vix_sentiment is not None
    def test_structure_valid(self):
        agg = _make_agg()
        agg.analyze()
        assert agg.vix_sentiment.structure in ("contango", "backwardation")
    def test_signal_valid(self):
        agg = _make_agg()
        agg.analyze()
        assert agg.vix_sentiment.signal in ("complacent", "fear", "neutral")

# ── Composite ────────────────────────────────────────────────────────────

class TestComposite:
    def test_composite_range(self):
        agg = _make_agg()
        agg.analyze()
        assert 0 <= agg.composite.index <= 100
    def test_composite_components(self):
        agg = _make_agg()
        agg.analyze()
        c = agg.composite
        for comp in [c.text_component, c.pcr_component, c.vix_component, c.volume_component]:
            assert 0 <= comp <= 100
    def test_composite_history_populated(self):
        agg = _make_agg()
        agg.analyze()
        assert len(agg.composite_history) > 0

# ── Contrarian ───────────────────────────────────────────────────────────

class TestContrarian:
    def test_extreme_fear_buy(self):
        agg = SentimentAggregator(extreme_threshold=90)
        agg.analyze()
        # Composite defaults to ~50, no extreme expected
        assert isinstance(agg.contrarian_signals, list)
    def test_contrarian_direction_valid(self):
        agg = _make_agg()
        agg.analyze()
        for cs in agg.contrarian_signals:
            assert cs.direction in ("buy", "sell")

# ── Extreme events ───────────────────────────────────────────────────────

class TestExtremes:
    def test_extremes_list(self):
        agg = _make_agg()
        agg.analyze()
        assert isinstance(agg.extreme_events, list)
    def test_extreme_direction_valid(self):
        agg = _make_agg()
        agg.analyze()
        for e in agg.extreme_events:
            assert e.direction in ("fear", "greed")

# ── Sentiment alpha ──────────────────────────────────────────────────────

class TestAlpha:
    def test_alpha_computed(self):
        agg = _make_agg()
        agg.analyze()
        assert len(agg.sentiment_alpha) > 0
    def test_alpha_has_overall(self):
        agg = _make_agg()
        agg.analyze()
        regimes = [a.regime for a in agg.sentiment_alpha]
        assert "overall" in regimes

# ── Pipeline ─────────────────────────────────────────────────────────────

class TestPipeline:
    def test_analyze_keys(self):
        agg = _make_agg()
        result = agg.analyze()
        expected = {"text_sentiments", "source_sentiments", "pcr_sentiment",
                    "vix_sentiment", "composite", "contrarian_signals",
                    "sentiment_alpha", "extreme_events"}
        assert set(result.keys()) == expected
    def test_empty_aggregator(self):
        agg = SentimentAggregator()
        agg.analyze()
        assert agg.composite is not None

# ── Report ───────────────────────────────────────────────────────────────

class TestReport:
    def test_generates_html(self, tmp_path):
        agg = _make_agg()
        path = agg.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Sentiment" in c
    def test_report_sections(self, tmp_path):
        agg = _make_agg()
        path = agg.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Timeline" in c and "Source" in c and "Contrarian" in c
    def test_report_charts(self, tmp_path):
        agg = _make_agg()
        path = agg.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()
    def test_report_auto_analyzes(self, tmp_path):
        agg = _make_agg()
        assert agg.composite is None
        agg.generate_report(str(tmp_path / "r.html"))
        assert agg.composite is not None
    def test_report_default_path(self):
        agg = _make_agg()
        path = agg.generate_report()
        assert "sentiment_signal.html" in path
