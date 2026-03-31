"""Tests for compass/sentiment_engine.py — NLP sentiment engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.sentiment_engine import (
    ContrarianSignal,
    HeadlineScore,
    RegimeConditionedSignal,
    SentimentEngine,
    SentimentMomentum,
    SentimentResult,
    SentimentSnapshot,
    aggregate_sentiment,
    build_alpha_signals,
    compute_contrarian_signals,
    compute_sentiment_momentum,
    regime_condition_sentiment,
    score_headline,
    score_headlines_batch,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_headlines(n: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    positive = [
        "Markets rally on strong earnings beat",
        "S&P 500 surges to record high on growth optimism",
        "Bullish momentum drives stocks higher",
        "Tech sector outperforms with robust gains",
        "Recovery accelerates as economy shows strength",
    ]
    negative = [
        "Markets crash amid recession fears",
        "Sell-off deepens as panic spreads",
        "Bearish sentiment grips Wall Street",
        "Stocks plunge on weak earnings miss",
        "Crisis fears trigger market decline",
    ]
    neutral = [
        "Federal Reserve meets to discuss policy",
        "Markets close mixed in light trading",
        "Investors await employment data release",
    ]
    all_headlines = positive + negative + neutral
    headlines = [all_headlines[rng.randint(0, len(all_headlines))] for _ in range(n)]
    return pd.DataFrame({"headline": headlines, "timestamp": dates})


def _make_regimes(n: int = 60) -> pd.Series:
    dates = pd.bdate_range("2024-01-02", periods=n)
    labels = ["bull"] * 20 + ["bear"] * 20 + ["sideways"] * 20
    return pd.Series(labels[:n], index=dates)


@pytest.fixture
def headlines():
    return _make_headlines()


@pytest.fixture
def regimes():
    return _make_regimes()


@pytest.fixture
def engine():
    return SentimentEngine(freq="D", contrarian_z=1.5, contrarian_lookback=10)


# ── Headline scoring tests ───────────────────────────────────────────────


class TestHeadlineScoring:
    def test_positive_headline(self):
        hs = score_headline("Markets rally on strong earnings beat")
        assert hs.score > 0
        assert len(hs.positive_words) > 0

    def test_negative_headline(self):
        hs = score_headline("Stocks crash amid recession fears")
        assert hs.score < 0
        assert len(hs.negative_words) > 0

    def test_neutral_headline(self):
        hs = score_headline("Federal Reserve meets to discuss policy")
        assert abs(hs.score) < 0.5

    def test_score_bounded(self):
        hs = score_headline("crash crash crash crash crash crash")
        assert -1.0 <= hs.score <= 1.0

    def test_negation_handling(self):
        hs_neg = score_headline("not bullish on this market")
        hs_pos = score_headline("bullish on this market")
        assert hs_neg.score < hs_pos.score

    def test_intensifier_handling(self):
        hs_normal = score_headline("rally expected")
        hs_intense = score_headline("extremely sharp rally expected")
        # Intensified should have stronger score
        assert abs(hs_intense.score) >= abs(hs_normal.score) - 0.01

    def test_empty_headline(self):
        hs = score_headline("")
        assert hs.score == 0.0

    def test_multi_word_phrases(self):
        hs = score_headline("The index reached an all-time high today")
        assert hs.score > 0

    def test_batch_scoring(self):
        texts = ["rally ahead", "crash imminent", "market steady"]
        results = score_headlines_batch(texts)
        assert len(results) == 3
        assert results[0].score > results[1].score

    def test_batch_with_timestamps(self):
        texts = ["rally", "crash"]
        ts = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
        results = score_headlines_batch(texts, ts)
        assert results[0].timestamp == ts[0]


# ── Aggregation tests ────────────────────────────────────────────────────


class TestAggregation:
    def test_aggregate_daily(self, headlines):
        scored = score_headlines_batch(
            headlines["headline"].tolist(),
            headlines["timestamp"].tolist(),
        )
        snapshots = aggregate_sentiment(scored, "D")
        assert len(snapshots) > 0
        assert all(isinstance(s, SentimentSnapshot) for s in snapshots)

    def test_snapshot_fields(self, headlines):
        scored = score_headlines_batch(
            headlines["headline"].tolist(),
            headlines["timestamp"].tolist(),
        )
        snapshots = aggregate_sentiment(scored, "D")
        for s in snapshots:
            assert 0 <= s.pct_positive <= 1.0
            assert 0 <= s.pct_negative <= 1.0
            assert s.n_headlines > 0

    def test_composite_bounded(self, headlines):
        scored = score_headlines_batch(
            headlines["headline"].tolist(),
            headlines["timestamp"].tolist(),
        )
        snapshots = aggregate_sentiment(scored, "D")
        for s in snapshots:
            assert -1.0 <= s.composite <= 1.0

    def test_no_timestamps(self):
        scored = score_headlines_batch(["rally", "crash", "steady"])
        snapshots = aggregate_sentiment(scored, "D")
        assert len(snapshots) == 1

    def test_empty(self):
        assert aggregate_sentiment([], "D") == []


# ── Momentum tests ───────────────────────────────────────────────────────


class TestMomentum:
    def test_momentum_computed(self, headlines):
        scored = score_headlines_batch(
            headlines["headline"].tolist(),
            headlines["timestamp"].tolist(),
        )
        snapshots = aggregate_sentiment(scored, "D")
        momentum = compute_sentiment_momentum(snapshots)
        assert len(momentum) > 0
        assert all(isinstance(m, SentimentMomentum) for m in momentum)

    def test_momentum_length(self):
        snapshots = [SentimentSnapshot(i, 0.1 * i, 0.0, 5, 0.5, 0.3, 0.1 * i) for i in range(10)]
        momentum = compute_sentiment_momentum(snapshots)
        assert len(momentum) == 9  # n - 1

    def test_increasing_sentiment_positive_momentum(self):
        snapshots = [SentimentSnapshot(i, 0.0, 0.0, 1, 0.5, 0.3, 0.1 * i) for i in range(10)]
        momentum = compute_sentiment_momentum(snapshots)
        assert momentum[-1].momentum_1d > 0

    def test_short_data(self):
        snapshots = [SentimentSnapshot(0, 0.5, 0.5, 1, 0.5, 0.3, 0.5)]
        assert compute_sentiment_momentum(snapshots) == []


# ── Contrarian signal tests ──────────────────────────────────────────────


class TestContrarian:
    def test_extreme_bullish_triggers_bearish(self):
        rng = np.random.RandomState(42)
        # Baseline with small variance, then extreme spike
        composites = rng.normal(0.05, 0.05, 25)
        composites[-1] = 0.9  # extreme bullish spike
        snapshots = [SentimentSnapshot(i, c, c, 5, 0.5, 0.3, c) for i, c in enumerate(composites)]
        signals = compute_contrarian_signals(snapshots, z_threshold=1.5, lookback=20)
        bearish = [s for s in signals if s.signal == "contrarian_bearish"]
        assert len(bearish) > 0

    def test_extreme_bearish_triggers_bullish(self):
        rng = np.random.RandomState(42)
        composites = rng.normal(-0.05, 0.05, 25)
        composites[-1] = -0.9  # extreme bearish spike
        snapshots = [SentimentSnapshot(i, c, c, 5, 0.5, 0.3, c) for i, c in enumerate(composites)]
        signals = compute_contrarian_signals(snapshots, z_threshold=1.5, lookback=20)
        bullish = [s for s in signals if s.signal == "contrarian_bullish"]
        assert len(bullish) > 0

    def test_strength_bounded(self):
        snapshots = [SentimentSnapshot(i, 0.0, 0.0, 5, 0.5, 0.3, 0.0) for i in range(25)]
        snapshots[-1] = SentimentSnapshot(24, 0.99, 0.99, 5, 1.0, 0.0, 0.99)
        signals = compute_contrarian_signals(snapshots, z_threshold=1.0, lookback=20)
        for s in signals:
            assert 0 <= s.strength <= 1.0

    def test_no_signal_normal(self):
        snapshots = [SentimentSnapshot(i, 0.1, 0.1, 5, 0.5, 0.3, 0.1) for i in range(30)]
        signals = compute_contrarian_signals(snapshots, z_threshold=3.0, lookback=20)
        assert len(signals) == 0

    def test_short_data(self):
        snapshots = [SentimentSnapshot(i, 0.5, 0.5, 1, 0.5, 0.3, 0.5) for i in range(5)]
        assert compute_contrarian_signals(snapshots, lookback=20) == []


# ── Regime conditioning tests ────────────────────────────────────────────


class TestRegimeConditioning:
    def test_divergent_detected(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        snapshots = [SentimentSnapshot(dates[i], 0.5, 0.5, 5, 0.8, 0.1, 0.5) for i in range(5)]
        regimes = pd.Series(["bear"] * 5, index=dates)
        signals = regime_condition_sentiment(snapshots, regimes)
        divergent = [s for s in signals if s.signal == "divergent"]
        assert len(divergent) > 0

    def test_confirming_detected(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        snapshots = [SentimentSnapshot(dates[i], 0.5, 0.5, 5, 0.8, 0.1, 0.5) for i in range(5)]
        regimes = pd.Series(["bull"] * 5, index=dates)
        signals = regime_condition_sentiment(snapshots, regimes)
        confirming = [s for s in signals if s.signal == "confirming"]
        assert len(confirming) > 0

    def test_empty_regimes(self):
        snapshots = [SentimentSnapshot(pd.Timestamp("2024-01-02"), 0.5, 0.5, 5, 0.5, 0.3, 0.5)]
        assert regime_condition_sentiment(snapshots, pd.Series(dtype=str)) == []


# ── Alpha integration tests ─────────────────────────────────────────────


class TestAlphaIntegration:
    def test_alpha_output_columns(self, headlines):
        scored = score_headlines_batch(
            headlines["headline"].tolist(),
            headlines["timestamp"].tolist(),
        )
        snapshots = aggregate_sentiment(scored, "D")
        momentum = compute_sentiment_momentum(snapshots)
        contrarian = compute_contrarian_signals(snapshots, 1.5, 10)
        alpha = build_alpha_signals(snapshots, momentum, contrarian)
        assert alpha is not None
        assert len(alpha.sentiment_signal) > 0
        assert len(alpha.momentum_signal) > 0
        assert len(alpha.contrarian_signal) > 0

    def test_alpha_signal_names(self, headlines):
        scored = score_headlines_batch(
            headlines["headline"].tolist(),
            headlines["timestamp"].tolist(),
        )
        snapshots = aggregate_sentiment(scored, "D")
        alpha = build_alpha_signals(snapshots, [], [])
        assert alpha.sentiment_signal.name == "sentiment"

    def test_no_timestamps_returns_none(self):
        scored = score_headlines_batch(["rally", "crash"])
        snapshots = aggregate_sentiment(scored, "D")
        alpha = build_alpha_signals(snapshots, [], [])
        assert alpha is None  # no timestamps


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_missing_column_raises(self):
        engine = SentimentEngine()
        with pytest.raises(ValueError, match="Missing"):
            engine.analyze(pd.DataFrame({"foo": [1]}))


# ── Full analysis tests ──────────────────────────────────────────────────


class TestFullAnalysis:
    def test_returns_result(self, engine, headlines):
        result = engine.analyze(headlines)
        assert isinstance(result, SentimentResult)
        assert result.n_headlines == len(headlines)

    def test_with_regimes(self, engine, headlines, regimes):
        result = engine.analyze(headlines, regimes=regimes)
        assert len(result.regime_signals) > 0

    def test_without_regimes(self, engine, headlines):
        result = engine.analyze(headlines)
        assert result.regime_signals == []

    def test_counts_sum(self, engine, headlines):
        result = engine.analyze(headlines)
        assert result.n_positive + result.n_negative + result.n_neutral == result.n_headlines

    def test_alpha_output_present(self, engine, headlines):
        result = engine.analyze(headlines)
        assert result.alpha_output is not None


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, engine, headlines):
        result = engine.analyze(headlines)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "sent.html"
            path = SentimentEngine.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Sentiment Engine" in content

    def test_contains_charts(self, engine, headlines):
        result = engine.analyze(headlines)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            SentimentEngine.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Sentiment Timeline" in content

    def test_contains_momentum(self, engine, headlines):
        result = engine.analyze(headlines)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            SentimentEngine.generate_report(result, out)
            content = out.read_text()
            assert "Momentum" in content

    def test_contains_regime_with_data(self, engine, headlines, regimes):
        result = engine.analyze(headlines, regimes=regimes)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            SentimentEngine.generate_report(result, out)
            content = out.read_text()
            assert "Regime" in content

    def test_default_path(self, engine, headlines):
        result = engine.analyze(headlines)
        path = SentimentEngine.generate_report(result)
        assert path.exists()
        assert "sentiment_engine.html" in str(path)
        path.unlink(missing_ok=True)
