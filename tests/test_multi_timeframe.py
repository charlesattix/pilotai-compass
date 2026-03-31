"""Tests for compass/multi_timeframe.py — multi-timeframe signal aggregator.

Covers:
  - Dataclass construction
  - Signal computation helpers (RSI, momentum, MA cross)
  - Per-timeframe signal generation
  - Resampling to multiple timeframes
  - Cross-timeframe confirmation scoring
  - Divergence detection
  - Walk-forward optimal weights
  - Regime-dependent selection
  - Aggregated signal
  - from_csv constructor
  - HTML report generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.multi_timeframe import (
    TIMEFRAMES,
    AggregatedSignal,
    ConfirmationScore,
    DivergenceAlert,
    MultiTimeframeAggregator,
    RegimeTimeframeSelection,
    TimeframeSignal,
    TimeframeWeight,
    _ma_cross,
    _momentum,
    _rsi,
    compute_signal,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_minute_data(n=5000, seed=42):
    """Generate 1-min OHLCV data spanning ~3 trading days."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-06-01 09:30", periods=n, freq="1min")
    close = 430.0 + rng.normal(0, 0.3, n).cumsum()
    return pd.DataFrame({"close": close}, index=dates)


def _make_daily_data(n=500, seed=42):
    """Generate daily close data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 430.0 + rng.normal(0, 1.5, n).cumsum()
    return pd.DataFrame({"close": close}, index=dates)


def _make_returns(index, seed=42):
    rng = np.random.RandomState(seed)
    return pd.Series(rng.normal(0.0003, 0.01, len(index)), index=index)


def _make_regimes(index):
    n = len(index)
    regimes = np.array(["neutral"] * n, dtype=object)
    regimes[: n // 4] = "bull"
    regimes[n // 4: n // 2] = "neutral"
    regimes[n // 2: 3 * n // 4] = "bear"
    regimes[3 * n // 4:] = "high_vol"
    return pd.Series(regimes, index=index)


def _make_aggregator_daily(n=500, seed=42, **kwargs):
    data = _make_daily_data(n=n, seed=seed)
    return MultiTimeframeAggregator(
        data, timeframes=["1D"], **kwargs,
    )


def _make_aggregator_minute(n=5000, seed=42, **kwargs):
    data = _make_minute_data(n=n, seed=seed)
    return MultiTimeframeAggregator(data, **kwargs)


# ── Dataclass tests ──────────────────────────────────────────────────────


class TestDataclasses:
    def test_timeframe_signal_fields(self):
        ts = TimeframeSignal(
            timeframe="1h", signal=0.5, strength=0.5,
            trend="bullish", rsi=60, momentum=0.02,
            ma_cross=0.01, n_bars=100,
        )
        assert ts.signal == pytest.approx(0.5)

    def test_confirmation_score_fields(self):
        cs = ConfirmationScore(
            score=0.8, n_bullish=3, n_bearish=1, n_neutral=0,
            dominant_direction="bullish",
            aligned_timeframes=["1h", "4h", "1D"],
            conflicting_timeframes=["5min"],
        )
        assert cs.score == pytest.approx(0.8)

    def test_divergence_alert_fields(self):
        da = DivergenceAlert(
            tf_short="5min", tf_long="1D",
            short_signal=0.7, long_signal=-0.5,
            divergence_type="bullish_divergence",
            severity="high", description="test",
        )
        assert da.severity == "high"

    def test_timeframe_weight_fields(self):
        tw = TimeframeWeight(
            timeframe="1h", weight=0.3, sharpe=1.2, hit_rate=0.6,
        )
        assert tw.weight == pytest.approx(0.3)

    def test_regime_selection_fields(self):
        rs = RegimeTimeframeSelection(
            regime="bull", weights={"1h": 0.5, "1D": 0.5},
            best_timeframe="1h", ensemble_sharpe=1.5, n_obs=100,
        )
        assert rs.best_timeframe == "1h"

    def test_aggregated_signal_fields(self):
        agg = AggregatedSignal(
            signal=0.3, confidence=0.7,
            confirmation=ConfirmationScore(0.7, 2, 1, 0, "bullish", [], []),
            regime="bull", weights_used={"1h": 1.0},
        )
        assert agg.signal == pytest.approx(0.3)


# ── Signal helpers tests ─────────────────────────────────────────────────


class TestSignalHelpers:
    def test_rsi_range(self):
        rng = np.random.RandomState(42)
        series = pd.Series(430 + rng.normal(0, 1, 100).cumsum())
        r = _rsi(series)
        assert 0 <= r <= 100

    def test_rsi_overbought(self):
        """Monotonically rising → RSI near 100."""
        series = pd.Series(np.arange(100, dtype=float))
        r = _rsi(series)
        assert r > 80

    def test_rsi_oversold(self):
        """Monotonically falling → RSI near 0."""
        series = pd.Series(np.arange(100, 0, -1, dtype=float))
        r = _rsi(series)
        assert r < 20

    def test_rsi_short_series(self):
        r = _rsi(pd.Series([1, 2]))
        assert r == 50.0

    def test_momentum_positive(self):
        series = pd.Series(np.arange(50, dtype=float) + 100)
        m = _momentum(series)
        assert m > 0

    def test_momentum_negative(self):
        series = pd.Series(np.arange(50, 0, -1, dtype=float) + 100)
        m = _momentum(series)
        assert m < 0

    def test_ma_cross_positive(self):
        """Rising series → fast MA > slow MA."""
        series = pd.Series(np.arange(50, dtype=float) + 100)
        mc = _ma_cross(series, fast=5, slow=20)
        assert mc > 0

    def test_ma_cross_short_series(self):
        mc = _ma_cross(pd.Series([1, 2, 3]), fast=5, slow=10)
        assert mc == 0.0


# ── compute_signal tests ─────────────────────────────────────────────────


class TestComputeSignal:
    def test_returns_timeframe_signal(self):
        series = pd.Series(np.arange(100, dtype=float) + 430)
        sig = compute_signal(series, "1h")
        assert isinstance(sig, TimeframeSignal)
        assert sig.timeframe == "1h"

    def test_signal_range(self):
        rng = np.random.RandomState(42)
        series = pd.Series(430 + rng.normal(0, 1, 200).cumsum())
        sig = compute_signal(series, "1D")
        assert -1 <= sig.signal <= 1

    def test_rising_series_bullish(self):
        series = pd.Series(np.arange(100, dtype=float) + 430)
        sig = compute_signal(series, "1D")
        assert sig.trend == "bullish"
        assert sig.signal > 0

    def test_falling_series_bearish(self):
        series = pd.Series(np.arange(100, 0, -1, dtype=float) + 430)
        sig = compute_signal(series, "1D")
        assert sig.trend == "bearish"
        assert sig.signal < 0


# ── Resampling tests ────────────────────────────────────────────────────


class TestResampling:
    def test_resamples_minute_data(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        assert len(agg.resampled) > 0

    def test_daily_data_produces_1D(self):
        agg = _make_aggregator_daily()
        agg.analyze()
        assert "1D" in agg.resampled

    def test_resampled_has_ohlc(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        for tf, ohlc in agg.resampled.items():
            assert "close" in ohlc.columns

    def test_higher_tf_fewer_bars(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        # 1min should have more bars than 1h
        if "1min" in agg.resampled and "1h" in agg.resampled:
            assert len(agg.resampled["1min"]) > len(agg.resampled["1h"])


# ── Confirmation tests ───────────────────────────────────────────────────


class TestConfirmation:
    def test_score_range(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        assert 0 <= agg.confirmation.score <= 1

    def test_counts_sum_to_total(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        total = agg.confirmation.n_bullish + agg.confirmation.n_bearish + agg.confirmation.n_neutral
        assert total == len(agg.tf_signals)

    def test_dominant_direction_valid(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        assert agg.confirmation.dominant_direction in ("bullish", "bearish", "neutral")


# ── Divergence tests ────────────────────────────────────────────────────


class TestDivergence:
    def test_divergences_are_list(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        assert isinstance(agg.divergences, list)

    def test_divergence_sorted_by_magnitude(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        if len(agg.divergences) >= 2:
            deltas = [abs(d.short_signal - d.long_signal) for d in agg.divergences]
            assert deltas == sorted(deltas, reverse=True)

    def test_divergence_severity_valid(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        for d in agg.divergences:
            assert d.severity in ("low", "medium", "high")

    def test_forced_divergence(self):
        """Create data where short and long TF must diverge."""
        # Short-term rise, but longer-term decline
        n = 5000
        rng = np.random.RandomState(99)
        close = np.zeros(n)
        close[:4000] = np.arange(4000, dtype=float) * -0.1 + 450
        close[4000:] = np.arange(1000, dtype=float) * 0.3 + close[3999]
        data = pd.DataFrame({"close": close},
                            index=pd.date_range("2024-06-01 09:30", periods=n, freq="1min"))
        agg = MultiTimeframeAggregator(data, timeframes=["5min", "1h"])
        agg.analyze()
        # At least check it doesn't crash
        assert agg.confirmation is not None


# ── Walk-forward weights tests ───────────────────────────────────────────


class TestWalkForward:
    def test_weights_without_returns(self):
        """Without returns, should use equal weights."""
        agg = _make_aggregator_daily()
        agg.analyze()
        for tw in agg.optimal_weights.values():
            assert tw.weight > 0

    def test_weights_with_returns(self):
        data = _make_daily_data(500)
        ret = _make_returns(data.index)
        agg = MultiTimeframeAggregator(data, timeframes=["1D"], returns=ret)
        agg.analyze()
        assert len(agg.optimal_weights) > 0

    def test_weights_sum_to_one(self):
        data = _make_daily_data(500)
        ret = _make_returns(data.index)
        agg = MultiTimeframeAggregator(data, timeframes=["1D"], returns=ret)
        agg.analyze()
        total = sum(tw.weight for tw in agg.optimal_weights.values())
        assert total == pytest.approx(1.0, abs=0.01)


# ── Regime selection tests ───────────────────────────────────────────────


class TestRegimeSelection:
    def test_no_regime_no_selection(self):
        agg = _make_aggregator_daily()
        agg.analyze()
        assert len(agg.regime_selections) == 0

    def test_with_regimes(self):
        data = _make_daily_data(500)
        ret = _make_returns(data.index)
        reg = _make_regimes(data.index)
        agg = MultiTimeframeAggregator(
            data, timeframes=["1D"], returns=ret, regimes=reg,
        )
        agg.analyze()
        assert len(agg.regime_selections) > 0

    def test_regime_weights_sum_to_one(self):
        data = _make_daily_data(500)
        ret = _make_returns(data.index)
        reg = _make_regimes(data.index)
        agg = MultiTimeframeAggregator(
            data, timeframes=["1D"], returns=ret, regimes=reg,
        )
        agg.analyze()
        for rs in agg.regime_selections.values():
            total = sum(rs.weights.values())
            assert total == pytest.approx(1.0, abs=0.01)


# ── Aggregated signal tests ─────────────────────────────────────────────


class TestAggregatedSignal:
    def test_signal_range(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        assert -1 <= agg.aggregated.signal <= 1

    def test_confidence_range(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        assert 0 <= agg.aggregated.confidence <= 1

    def test_weights_used_populated(self):
        agg = _make_aggregator_minute()
        agg.analyze()
        assert len(agg.aggregated.weights_used) > 0


# ── Full pipeline tests ─────────────────────────────────────────────────


class TestPipeline:
    def test_analyze_returns_all_keys(self):
        agg = _make_aggregator_minute()
        result = agg.analyze()
        expected = {
            "tf_signals", "confirmation", "divergences",
            "optimal_weights", "regime_selections", "aggregated",
        }
        assert set(result.keys()) == expected

    def test_from_csv(self, tmp_path):
        data = _make_daily_data()
        csv = tmp_path / "data.csv"
        data.to_csv(csv)
        agg = MultiTimeframeAggregator.from_csv(str(csv), timeframes=["1D"])
        agg.analyze()
        assert agg.aggregated is not None


# ── Report tests ─────────────────────────────────────────────────────────


class TestReport:
    def test_generates_html(self, tmp_path):
        agg = _make_aggregator_minute()
        path = agg.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Multi-Timeframe" in content

    def test_report_sections(self, tmp_path):
        agg = _make_aggregator_minute()
        path = agg.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "Per-Timeframe" in content
        assert "Confirmation" in content
        assert "Divergence" in content
        assert "Optimal" in content

    def test_report_embeds_charts(self, tmp_path):
        agg = _make_aggregator_minute()
        path = agg.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "data:image/png;base64," in content

    def test_report_auto_analyzes(self, tmp_path):
        agg = _make_aggregator_minute()
        assert agg.aggregated is None
        agg.generate_report(str(tmp_path / "report.html"))
        assert agg.aggregated is not None

    def test_report_default_path(self):
        agg = _make_aggregator_minute()
        path = agg.generate_report()
        assert "multi_timeframe.html" in path
        assert open(path).read().startswith("<!DOCTYPE html>")
