"""Tests for compass.intraday_patterns – intraday pattern analyzer."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.intraday_patterns import (
    DOW_LABELS,
    DayOfWeekStats,
    HourStats,
    IntradayPatternAnalyzer,
    IntradayResult,
    SessionComparison,
    TimingRecommendation,
    VolatilityProfile,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_trades(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Deterministic intraday trade data spanning weekdays 9-16h."""
    rng = np.random.RandomState(seed)
    base = pd.date_range("2024-01-02 09:30", periods=n, freq="30min")
    # Filter to weekdays and market hours only
    mask = (base.dayofweek < 5) & (base.hour >= 9) & (base.hour < 16)
    dts = base[mask][:n]
    pnl = rng.randn(len(dts)) * 50 + 5
    return pd.DataFrame({
        "datetime": dts,
        "pnl": pnl,
        "slippage": rng.uniform(0.01, 0.05, len(dts)),
    })


def _make_trades_with_regime(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dts = pd.date_range("2024-01-02 09:30", periods=n, freq="30min")
    mask = (dts.dayofweek < 5) & (dts.hour >= 9) & (dts.hour < 16)
    dts = dts[mask][:n]
    m = len(dts)
    pnl = rng.randn(m) * 50 + 5
    regimes = ["bull"] * (m // 2) + ["bear"] * (m - m // 2)
    return pd.DataFrame({
        "datetime": dts,
        "pnl": pnl,
        "regime": regimes,
    })


def _make_price_series(n: int = 2000, seed: int = 77) -> pd.Series:
    """Intraday price series with hourly frequency."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-02 09:00", periods=n, freq="h")
    prices = 450.0 * np.cumprod(1 + rng.randn(n) * 0.002)
    return pd.Series(prices, index=idx, name="price")


def _make_regime_series(trades: pd.DataFrame) -> pd.Series:
    n = len(trades)
    labels = ["bull"] * (n // 3) + ["bear"] * (n // 3) + ["high_vol"] * (n - 2 * (n // 3))
    return pd.Series(labels, index=trades["datetime"].values)


# ── Constructor ─────────────────────────────────────────────────────────────
class TestIntradayPatternAnalyzerInit:
    def test_defaults(self):
        a = IntradayPatternAnalyzer()
        assert a.market_open == 9
        assert a.market_close == 16

    def test_custom_hours(self):
        a = IntradayPatternAnalyzer(market_open=8, market_close=17)
        assert a.market_open == 8
        assert a.market_close == 17


# ── Basic analysis ──────────────────────────────────────────────────────────
class TestAnalyze:
    def test_returns_result(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        assert isinstance(result, IntradayResult)

    def test_hour_stats_populated(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        assert len(result.hour_stats) > 0

    def test_dow_stats_populated(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        assert len(result.dow_stats) > 0

    def test_session_comparison_present(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        assert result.session_comparison is not None

    def test_n_trades_set(self):
        trades = _make_trades(n=200)
        result = IntradayPatternAnalyzer().analyze(trades)
        assert result.n_trades > 0

    def test_generated_at_set(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        assert len(result.generated_at) > 0

    def test_empty_trades(self):
        result = IntradayPatternAnalyzer().analyze(pd.DataFrame({"pnl": []}))
        assert result.hour_stats == []

    def test_missing_pnl_column(self):
        result = IntradayPatternAnalyzer().analyze(pd.DataFrame({"foo": [1]}))
        assert result.hour_stats == []

    def test_date_column_alias(self):
        trades = _make_trades()
        trades = trades.rename(columns={"datetime": "date"})
        result = IntradayPatternAnalyzer().analyze(trades)
        assert result.n_trades > 0


# ── Hour-of-day analysis ───────────────────────────────────────────────────
class TestHourStats:
    def test_hour_stats_fields(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        for h in result.hour_stats:
            assert 0 <= h.hour <= 23
            assert isinstance(h.avg_pnl, float)
            assert h.n_trades > 0
            assert 0.0 <= h.win_rate <= 1.0

    def test_session_classified(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        sessions = {h.session for h in result.hour_stats}
        # All trades are 9-16h so should be "market"
        assert "market" in sessions

    def test_pre_market_classification(self):
        a = IntradayPatternAnalyzer()
        assert a._classify_session(8) == "pre_market"
        assert a._classify_session(9) == "market"
        assert a._classify_session(15) == "market"
        assert a._classify_session(16) == "post_market"


# ── Day-of-week analysis ───────────────────────────────────────────────────
class TestDayOfWeekStats:
    def test_dow_stats_fields(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        for d in result.dow_stats:
            assert d.day in DOW_LABELS
            assert d.n_trades > 0
            assert 0.0 <= d.win_rate <= 1.0

    def test_weekdays_only(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        days = {d.day for d in result.dow_stats}
        # Generated data is weekdays only
        assert days.issubset({"Monday", "Tuesday", "Wednesday", "Thursday", "Friday"})

    def test_dow_idx_matches_label(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        for d in result.dow_stats:
            assert DOW_LABELS[d.day_idx] == d.day


# ── Session comparison ──────────────────────────────────────────────────────
class TestSessionComparison:
    def test_session_fields(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        s = result.session_comparison
        assert isinstance(s.open_avg_pnl, float)
        assert isinstance(s.close_avg_pnl, float)
        assert s.open_n_trades > 0

    def test_better_session_valid(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        assert result.session_comparison.better_session in ["open", "close"]

    def test_slippage_nonnegative(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        s = result.session_comparison
        assert s.open_avg_slippage >= 0
        assert s.close_avg_slippage >= 0


# ── Volatility profile ─────────────────────────────────────────────────────
class TestVolatilityProfile:
    def test_vol_profile_present(self):
        trades = _make_trades()
        prices = _make_price_series()
        result = IntradayPatternAnalyzer().analyze(trades, price_series=prices)
        assert result.volatility_profile is not None

    def test_vol_profile_fields(self):
        prices = _make_price_series()
        result = IntradayPatternAnalyzer().analyze(_make_trades(), price_series=prices)
        vp = result.volatility_profile
        assert isinstance(vp.hourly_vol, dict)
        assert 0 <= vp.peak_hour <= 23
        assert 0 <= vp.trough_hour <= 23
        assert vp.open_close_ratio >= 0

    def test_no_prices_no_profile(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        assert result.volatility_profile is None


# ── Timing recommendations ──────────────────────────────────────────────────
class TestTimingRecommendations:
    def test_recommendations_with_regime_column(self):
        trades = _make_trades_with_regime()
        result = IntradayPatternAnalyzer().analyze(trades)
        assert len(result.timing_recommendations) > 0

    def test_recommendations_with_regime_series(self):
        trades = _make_trades()
        regimes = _make_regime_series(trades)
        result = IntradayPatternAnalyzer().analyze(trades, regimes=regimes)
        assert len(result.timing_recommendations) > 0

    def test_recommendation_fields(self):
        trades = _make_trades_with_regime()
        result = IntradayPatternAnalyzer().analyze(trades)
        for t in result.timing_recommendations:
            assert isinstance(t.regime, str)
            assert 0 <= t.best_entry_hour <= 23
            assert t.best_day in DOW_LABELS
            assert t.worst_day in DOW_LABELS
            assert t.n_obs > 0

    def test_no_regime_no_recommendations(self):
        result = IntradayPatternAnalyzer().analyze(_make_trades())
        assert result.timing_recommendations == []


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = IntradayPatternAnalyzer()
            trades = _make_trades_with_regime()
            prices = _make_price_series()
            result = a.analyze(trades, price_series=prices)
            path = a.generate_report(result, output_path=Path(tmp) / "ip.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = IntradayPatternAnalyzer()
            trades = _make_trades_with_regime()
            prices = _make_price_series()
            result = a.analyze(trades, price_series=prices)
            path = a.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Intraday Pattern" in html
            assert "Hour-of-Day" in html
            assert "Day-of-Week" in html
            assert "Opening vs Closing" in html
            assert "Timing Recommendation" in html
            assert "Volatility" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = IntradayPatternAnalyzer()
            result = a.analyze(_make_trades())
            path = a.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html

    def test_report_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = IntradayPatternAnalyzer()
            result = IntradayResult(generated_at="2024-01-01T00:00:00+00:00")
            path = a.generate_report(result, output_path=Path(tmp) / "e.html")
            assert path.exists()


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_hour_stats(self):
        h = HourStats(10, 25.0, 500.0, 0.65, 20, 30.0, "market")
        assert h.hour == 10
        assert h.session == "market"

    def test_dow_stats(self):
        d = DayOfWeekStats("Monday", 0, 15.0, 300.0, 0.60, 20, 25.0)
        assert d.day == "Monday"

    def test_session_comparison(self):
        s = SessionComparison(10.0, 0.6, 50, 8.0, 0.55, 50, 0.02, 0.03, "open")
        assert s.better_session == "open"

    def test_timing_recommendation(self):
        t = TimingRecommendation("bull", 10, 15, "Tuesday", "Friday", 25.0, 100)
        assert t.regime == "bull"

    def test_volatility_profile(self):
        v = VolatilityProfile({9: 0.20, 15: 0.10}, 9, 15, 2.0)
        assert v.peak_hour == 9

    def test_intraday_result_defaults(self):
        r = IntradayResult()
        assert r.hour_stats == []
        assert r.dow_stats == []
        assert r.n_trades == 0
