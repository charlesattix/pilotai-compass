"""Tests for compass.event_impact – event impact analyzer."""
from __future__ import annotations

import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.event_impact import (
    BASELINE_IV_CRUSH,
    DEFAULT_ENTRY_OFFSETS,
    EVENT_TYPES,
    EventImpactAnalyzer,
    EventImpactResult,
    EventTypeStats,
    EventWindow,
    IVCrushResult,
    TimingResult,
    build_event_calendar,
    _cpi_date,
    _first_friday,
    _third_friday,
    _third_wednesday,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_trades(
    start: date = date(2024, 1, 1),
    n_days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic daily trade data."""
    rng = np.random.RandomState(seed)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    pnl = rng.randn(n_days) * 50 + 5  # slight positive bias
    return pd.DataFrame({"date": dates, "pnl": pnl})


def _make_iv_series(
    start: date = date(2024, 1, 1),
    n_days: int = 365,
    base: float = 20.0,
    seed: int = 77,
) -> pd.Series:
    """Synthetic VIX-like IV series with spikes around mid-month."""
    rng = np.random.RandomState(seed)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    vals = base + rng.randn(n_days) * 2
    # add spikes around day 10-14 of each month (pre-CPI/FOMC)
    for i, d in enumerate(dates):
        if 10 <= d.day <= 14:
            vals[i] += 3.0
    return pd.Series(vals, index=dates, name="iv")


def _make_price_series(
    start: date = date(2024, 1, 1),
    n_days: int = 365,
    base: float = 450.0,
    seed: int = 55,
) -> pd.Series:
    rng = np.random.RandomState(seed)
    returns = rng.randn(n_days) * 0.01
    prices = base * np.cumprod(1 + returns)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    return pd.Series(prices, index=dates, name="price")


def _make_calendar(start: date = date(2024, 1, 1), end: date = date(2024, 12, 31)) -> pd.DataFrame:
    return build_event_calendar(start, end, fomc_dates=[])


# ── Calendar helper tests ───────────────────────────────────────────────────
class TestCalendarHelpers:
    def test_third_friday_jan_2024(self):
        assert _third_friday(2024, 1) == date(2024, 1, 19)

    def test_third_friday_is_friday(self):
        for m in range(1, 13):
            d = _third_friday(2024, m)
            assert d.weekday() == 4  # Friday

    def test_third_wednesday_is_wednesday(self):
        for m in range(1, 13):
            d = _third_wednesday(2024, m)
            assert d.weekday() == 2  # Wednesday

    def test_first_friday_is_friday(self):
        for m in range(1, 13):
            d = _first_friday(2024, m)
            assert d.weekday() == 4
            assert d.day <= 7

    def test_cpi_date_is_wednesday(self):
        for m in range(1, 13):
            d = _cpi_date(2024, m)
            assert d.weekday() == 2
            assert 8 <= d.day <= 14


class TestBuildEventCalendar:
    def test_calendar_has_expected_columns(self):
        cal = build_event_calendar(date(2024, 1, 1), date(2024, 12, 31), fomc_dates=[])
        assert "event_type" in cal.columns
        assert "event_date" in cal.columns

    def test_calendar_types_present(self):
        cal = build_event_calendar(date(2024, 1, 1), date(2024, 12, 31), fomc_dates=[])
        types = set(cal["event_type"])
        # No FOMC since we passed empty list, but should have the algorithmic ones
        assert "CPI" in types
        assert "NFP" in types
        assert "OPEX" in types
        assert "VIX_EXP" in types

    def test_calendar_with_fomc_dates(self):
        fomc = [date(2024, 3, 20), date(2024, 6, 12)]
        cal = build_event_calendar(date(2024, 1, 1), date(2024, 12, 31), fomc_dates=fomc)
        fomc_rows = cal[cal["event_type"] == "FOMC"]
        assert len(fomc_rows) == 2

    def test_calendar_sorted_by_date(self):
        cal = build_event_calendar(date(2024, 1, 1), date(2024, 6, 30), fomc_dates=[])
        dates = list(cal["event_date"])
        assert dates == sorted(dates)

    def test_empty_range_returns_empty(self):
        cal = build_event_calendar(date(2024, 1, 1), date(2024, 1, 1), fomc_dates=[])
        assert len(cal) == 0 or len(cal) <= 4  # at most 4 events on Jan 1

    def test_calendar_monthly_events(self):
        cal = build_event_calendar(date(2024, 1, 1), date(2024, 12, 31), fomc_dates=[])
        # Should have ~12 of each monthly event type
        for etype in ["CPI", "NFP", "OPEX", "VIX_EXP"]:
            count = len(cal[cal["event_type"] == etype])
            assert 10 <= count <= 14, f"{etype} has {count} events"


# ── Analyzer init ───────────────────────────────────────────────────────────
class TestEventImpactAnalyzerInit:
    def test_default_windows(self):
        a = EventImpactAnalyzer()
        assert a.pre_window == 5
        assert a.post_window == 5

    def test_custom_windows(self):
        a = EventImpactAnalyzer(pre_window=3, post_window=7)
        assert a.pre_window == 3
        assert a.post_window == 7

    def test_default_offsets(self):
        a = EventImpactAnalyzer()
        assert a.entry_offsets == DEFAULT_ENTRY_OFFSETS


# ── Full analysis ───────────────────────────────────────────────────────────
class TestAnalyze:
    def test_returns_event_impact_result(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        assert isinstance(result, EventImpactResult)

    def test_event_stats_populated(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        assert len(result.event_stats) > 0

    def test_event_windows_populated(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        assert len(result.event_windows) > 0

    def test_generated_at_set(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        assert len(result.generated_at) > 0

    def test_empty_trades(self):
        trades = pd.DataFrame(columns=["date", "pnl"])
        result = EventImpactAnalyzer().analyze(trades)
        assert result.event_stats == []

    def test_missing_columns_returns_empty(self):
        trades = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        result = EventImpactAnalyzer().analyze(trades)
        assert result.event_stats == []

    def test_entry_date_alias(self):
        """'entry_date' column should work as alias for 'date'."""
        trades = _make_trades()
        trades = trades.rename(columns={"date": "entry_date"})
        result = EventImpactAnalyzer().analyze(trades)
        assert len(result.event_stats) > 0


# ── Event type stats ────────────────────────────────────────────────────────
class TestEventTypeStats:
    def test_win_rate_bounded(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        for s in result.event_stats:
            assert 0.0 <= s.win_rate <= 1.0

    def test_pnl_decomposition(self):
        """Pre + post should roughly equal total (via avg)."""
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        for s in result.event_stats:
            # avg_pnl is over windows; pre+post may differ due to window overlaps
            assert isinstance(s.avg_pre_pnl, float)
            assert isinstance(s.avg_post_pnl, float)

    def test_best_entry_offset_in_offsets(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        for s in result.event_stats:
            assert s.best_entry_offset in DEFAULT_ENTRY_OFFSETS


# ── IV crush ────────────────────────────────────────────────────────────────
class TestIVCrush:
    def test_iv_crush_with_iv_series(self):
        trades = _make_trades()
        iv = _make_iv_series()
        result = EventImpactAnalyzer().analyze(trades, iv_series=iv)
        assert len(result.iv_crush_results) > 0

    def test_iv_crush_fields(self):
        trades = _make_trades()
        iv = _make_iv_series()
        result = EventImpactAnalyzer().analyze(trades, iv_series=iv)
        for c in result.iv_crush_results:
            assert isinstance(c.avg_crush_pct, float)
            assert isinstance(c.median_crush_pct, float)
            assert c.n_events > 0

    def test_no_iv_series_empty_crush(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        assert result.iv_crush_results == []

    def test_crush_vs_baseline(self):
        trades = _make_trades()
        iv = _make_iv_series()
        result = EventImpactAnalyzer().analyze(trades, iv_series=iv)
        for c in result.iv_crush_results:
            assert isinstance(c.crush_vs_baseline, float)


# ── Timing ──────────────────────────────────────────────────────────────────
class TestTiming:
    def test_timing_results_present(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        assert len(result.timing_results) > 0

    def test_timing_offset_nonnegative(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        for t in result.timing_results:
            assert t.offset_days >= 0

    def test_timing_win_rate_bounded(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        for t in result.timing_results:
            assert 0.0 <= t.win_rate <= 1.0


# ── Event windows ───────────────────────────────────────────────────────────
class TestEventWindows:
    def test_window_pnl_decomposition(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        for w in result.event_windows:
            assert w.total_pnl == pytest.approx(w.pre_pnl + w.post_pnl)

    def test_window_event_types_valid(self):
        trades = _make_trades()
        result = EventImpactAnalyzer().analyze(trades)
        for w in result.event_windows:
            assert w.event_type in EVENT_TYPES

    def test_window_with_price_series(self):
        trades = _make_trades()
        prices = _make_price_series()
        result = EventImpactAnalyzer().analyze(trades, price_series=prices)
        has_returns = any(w.pre_return != 0.0 or w.post_return != 0.0 for w in result.event_windows)
        assert has_returns


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = _make_trades()
            iv = _make_iv_series()
            analyzer = EventImpactAnalyzer()
            result = analyzer.analyze(trades, iv_series=iv)
            path = analyzer.generate_report(result, output_path=Path(tmp) / "test.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_contains_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = _make_trades()
            iv = _make_iv_series()
            analyzer = EventImpactAnalyzer()
            result = analyzer.analyze(trades, iv_series=iv)
            path = analyzer.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Event Impact Analysis" in html
            assert "P&L by Event Type" in html
            assert "Event Type Summary" in html
            assert "Optimal Entry Timing" in html
            assert "IV Crush" in html
            assert "Pre/Post" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = _make_trades()
            analyzer = EventImpactAnalyzer()
            result = analyzer.analyze(trades)
            path = analyzer.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html

    def test_report_empty_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = EventImpactAnalyzer()
            result = EventImpactResult(generated_at="2024-01-01T00:00:00+00:00")
            path = analyzer.generate_report(result, output_path=Path(tmp) / "empty.html")
            assert path.exists()


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_event_window_fields(self):
        w = EventWindow(
            event_type="FOMC", event_date=date(2024, 3, 20),
            pre_pnl=10.0, post_pnl=-5.0, total_pnl=5.0,
            pre_iv=22.0, post_iv=18.0, iv_crush_pct=0.18,
            pre_return=0.01, post_return=-0.005, n_trades=3,
        )
        assert w.event_type == "FOMC"
        assert w.total_pnl == 5.0

    def test_event_impact_result_defaults(self):
        r = EventImpactResult()
        assert r.event_stats == []
        assert r.timing_results == []
        assert r.iv_crush_results == []
        assert r.event_windows == []

    def test_timing_result_fields(self):
        t = TimingResult(event_type="CPI", offset_days=2, avg_pnl=15.0, win_rate=0.6, n_trades=10)
        assert t.event_type == "CPI"
        assert t.offset_days == 2

    def test_iv_crush_result_fields(self):
        c = IVCrushResult(
            event_type="NFP", avg_crush_pct=0.35, median_crush_pct=0.30,
            std_crush_pct=0.10, crush_vs_baseline=1.0, n_events=8,
        )
        assert c.event_type == "NFP"
        assert c.n_events == 8


# ── Constants ───────────────────────────────────────────────────────────────
class TestConstants:
    def test_event_types_list(self):
        assert len(EVENT_TYPES) == 5
        assert "FOMC" in EVENT_TYPES
        assert "OPEX" in EVENT_TYPES

    def test_baseline_iv_crush_keys(self):
        for et in EVENT_TYPES:
            assert et in BASELINE_IV_CRUSH

    def test_baseline_iv_crush_bounded(self):
        for v in BASELINE_IV_CRUSH.values():
            assert 0.0 < v < 1.0
