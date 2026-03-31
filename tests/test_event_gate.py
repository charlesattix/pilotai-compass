"""
Unit tests for the COMPASS event calendar and position-size scaling.

Target module: compass/events.py (post-Phase 2 move)
Pre-move source: shared/macro_event_gate.py

Tests cover:
  - FOMC date completeness (2020-2026 including emergency dates)
  - CPI / NFP date computation helpers
  - _iter_months year boundary handling
  - get_upcoming_events() pre-event scaling and post-event buffers (G5)
  - compute_composite_scaling() per-type minimums (G4)
  - run_daily_event_check() DB persistence integration

Blueprint spec: 10+ tests, all green (Phase 3 exit criteria).
"""

from datetime import date, timedelta

import pytest

from compass.events import (
    ALL_FOMC_DATES,
    FOMC_SCALING,
    CPI_SCALING,
    NFP_SCALING,
    get_upcoming_events,
    compute_composite_scaling,
    run_daily_event_check,
    _first_friday_of_month,
    _cpi_release_date,
    _nfp_release_date,
    _iter_months,
)

from tests.compass_helpers import (
    KNOWN_FOMC_DATES,
    KNOWN_CPI_DATES,
    KNOWN_NFP_DATES,
    POST_FOMC_SCALING,
    POST_CPI_SCALING,
    POST_NFP_SCALING,
    mock_macro_db,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def test_db(tmp_path):
    """Isolated SQLite macro DB for integration tests."""
    return mock_macro_db(tmp_path)


# ══════════════════════════════════════════════════════════════════════════════
# A. FOMC date data integrity
# ══════════════════════════════════════════════════════════════════════════════

class TestFomcDates:

    def test_2026_fomc_dates_present(self):
        """All 8 scheduled 2026 FOMC dates are in ALL_FOMC_DATES."""
        fomc_2026 = [
            KNOWN_FOMC_DATES["2026_jan"],
            KNOWN_FOMC_DATES["2026_mar"],
            KNOWN_FOMC_DATES["2026_may"],
            KNOWN_FOMC_DATES["2026_jun"],
            KNOWN_FOMC_DATES["2026_jul"],
            KNOWN_FOMC_DATES["2026_sep"],
            KNOWN_FOMC_DATES["2026_nov"],
            KNOWN_FOMC_DATES["2026_dec"],
        ]
        for d in fomc_2026:
            assert d in ALL_FOMC_DATES, f"Missing FOMC date: {d}"

    def test_emergency_dates_included(self):
        """COVID emergency dates 2020-03-03 and 2020-03-15 are present."""
        assert KNOWN_FOMC_DATES["2020_emergency_1"] in ALL_FOMC_DATES
        assert KNOWN_FOMC_DATES["2020_emergency_2"] in ALL_FOMC_DATES

    def test_all_fomc_dates_sorted(self):
        """ALL_FOMC_DATES is in chronological order."""
        for i in range(1, len(ALL_FOMC_DATES)):
            assert ALL_FOMC_DATES[i] > ALL_FOMC_DATES[i - 1], (
                f"Not sorted at index {i}: {ALL_FOMC_DATES[i-1]} >= {ALL_FOMC_DATES[i]}"
            )

    def test_fomc_dates_no_duplicates(self):
        """ALL_FOMC_DATES contains no duplicate entries."""
        assert len(ALL_FOMC_DATES) == len(set(ALL_FOMC_DATES))


# ══════════════════════════════════════════════════════════════════════════════
# B. Date computation helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestDateHelpers:

    def test_first_friday_is_always_friday(self):
        """_first_friday_of_month always returns a Friday (weekday=4)."""
        for year in (2024, 2025, 2026):
            for month in range(1, 13):
                result = _first_friday_of_month(year, month)
                assert result.weekday() == 4, (
                    f"Not a Friday: {result} (weekday={result.weekday()})"
                )

    def test_cpi_release_date_is_weekday(self):
        """_cpi_release_date never returns a Saturday or Sunday."""
        for year in (2024, 2025, 2026):
            for month in range(1, 13):
                result = _cpi_release_date(year, month)
                assert result.weekday() < 5, (
                    f"Weekend date: {result} (weekday={result.weekday()})"
                )

    def test_cpi_release_date_year_boundary(self):
        """CPI for December -> releases in January of next year."""
        result = _cpi_release_date(2025, 12)
        # CPI for Dec 2025 releases in Jan 2026
        assert result.year == 2026
        assert result.month == 1

    def test_nfp_release_date_is_friday(self):
        """_nfp_release_date always returns a Friday."""
        for year in (2024, 2025, 2026):
            for month in range(1, 12):  # Avoid Dec overflow tested separately
                result = _nfp_release_date(year, month)
                assert result.weekday() == 4, (
                    f"NFP not a Friday: {result} (weekday={result.weekday()})"
                )

    def test_nfp_release_date_year_boundary(self):
        """NFP for December -> first Friday of January next year."""
        result = _nfp_release_date(2025, 12)
        assert result.year == 2026
        assert result.month == 1
        assert result.weekday() == 4

    def test_iter_months_wraps_year(self):
        """_iter_months correctly handles December -> January year boundary."""
        base = date(2025, 11, 15)
        months = list(_iter_months(base, range(0, 4)))
        assert months[0] == (2025, 11)
        assert months[1] == (2025, 12)
        assert months[2] == (2026, 1)
        assert months[3] == (2026, 2)


# ══════════════════════════════════════════════════════════════════════════════
# C. get_upcoming_events() — pre-event scaling
# ══════════════════════════════════════════════════════════════════════════════

class TestGetUpcomingEvents:

    def test_fomc_day_scaling_0_50(self):
        """On FOMC day itself -> scaling_factor = 0.50."""
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]
        events = get_upcoming_events(as_of=fomc_day, horizon_days=14)
        fomc_events = [e for e in events if e["event_type"] == "FOMC"]
        assert any(
            e["days_out"] == 0 and e["scaling_factor"] == 0.50
            for e in fomc_events
        ), f"Expected FOMC day-of scaling 0.50, got: {fomc_events}"

    def test_fomc_1_day_before_scaling_0_60(self):
        """1 day before FOMC -> scaling_factor = 0.60."""
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]
        day_before = fomc_day - timedelta(days=1)
        events = get_upcoming_events(as_of=day_before, horizon_days=14)
        fomc_events = [e for e in events if e["event_type"] == "FOMC" and e["days_out"] == 1]
        assert len(fomc_events) >= 1
        assert fomc_events[0]["scaling_factor"] == 0.60

    def test_fomc_5_days_before_scaling_1_00(self):
        """5 days before FOMC -> scaling_factor = 1.00 (within window but max)."""
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]
        five_before = fomc_day - timedelta(days=5)
        events = get_upcoming_events(as_of=five_before, horizon_days=14)
        fomc_events = [e for e in events if e["event_type"] == "FOMC" and e["days_out"] == 5]
        assert len(fomc_events) >= 1
        assert fomc_events[0]["scaling_factor"] == 1.00

    def test_fomc_6_plus_days_no_event(self):
        """6+ calendar days before FOMC -> no FOMC event returned (outside horizon)."""
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]
        # Use horizon_days=5 to exclude events more than 5 days away
        six_before = fomc_day - timedelta(days=6)
        events = get_upcoming_events(as_of=six_before, horizon_days=5)
        fomc_events = [e for e in events if e["event_type"] == "FOMC"
                       and e["event_date"] == fomc_day.strftime("%Y-%m-%d")]
        assert len(fomc_events) == 0, f"FOMC found when it should be outside horizon: {fomc_events}"

    def test_no_events_far_from_dates(self):
        """Date far from any event -> no FOMC events within a narrow horizon."""
        # Pick a date far from any FOMC/CPI/NFP (mid-month, not near 12th or 1st Friday)
        # Use a very narrow horizon so we don't pick up nearby events
        test_date = date(2026, 4, 20)
        events = get_upcoming_events(as_of=test_date, horizon_days=2)
        # May still have CPI/NFP but FOMC should be absent for narrow window
        fomc_events = [e for e in events if e["event_type"] == "FOMC"]
        assert len(fomc_events) == 0

    def test_events_are_deduplicated(self):
        """No duplicate (event_date, event_type) pairs in output."""
        events = get_upcoming_events(as_of=date(2026, 1, 20), horizon_days=30)
        seen = set()
        for ev in events:
            key = (ev["event_date"], ev["event_type"])
            assert key not in seen, f"Duplicate event: {key}"
            seen.add(key)


# ══════════════════════════════════════════════════════════════════════════════
# D. Post-event buffers (G5)
# ══════════════════════════════════════════════════════════════════════════════

class TestPostEventBuffers:

    def test_post_fomc_buffer_day_after(self):
        """1 day after FOMC -> FOMC_POST event at 0.70x scaling."""
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]
        day_after = fomc_day + timedelta(days=1)
        # Query from the day after, so FOMC_POST is at days_out=0
        events = get_upcoming_events(as_of=day_after, horizon_days=14)
        post_events = [e for e in events if e["event_type"] == "FOMC_POST"
                       and e["event_date"] == day_after.strftime("%Y-%m-%d")]
        assert len(post_events) >= 1, f"No FOMC_POST event found on {day_after}"
        assert post_events[0]["scaling_factor"] == POST_FOMC_SCALING

    def test_post_cpi_buffer_day_after(self):
        """CPI day query includes CPI_POST buffer at 0.80x for the next day."""
        cpi_date = KNOWN_CPI_DATES["jan_2026"]
        post_date = cpi_date + timedelta(days=1)
        # Query from the CPI date itself; the post-buffer is generated alongside the event
        events = get_upcoming_events(as_of=cpi_date, horizon_days=14)
        post_events = [e for e in events if e["event_type"] == "CPI_POST"
                       and e["event_date"] == post_date.strftime("%Y-%m-%d")]
        assert len(post_events) >= 1, f"No CPI_POST event found for {post_date}"
        assert post_events[0]["scaling_factor"] == POST_CPI_SCALING

    def test_post_nfp_buffer_day_after(self):
        """NFP day query includes NFP_POST buffer at 0.80x for the next day."""
        nfp_date = KNOWN_NFP_DATES["jan_2026"]
        post_date = nfp_date + timedelta(days=1)
        # Query from the NFP date itself; the post-buffer is generated alongside the event
        events = get_upcoming_events(as_of=nfp_date, horizon_days=14)
        post_events = [e for e in events if e["event_type"] == "NFP_POST"
                       and e["event_date"] == post_date.strftime("%Y-%m-%d")]
        assert len(post_events) >= 1, f"No NFP_POST event found for {post_date}"
        assert post_events[0]["scaling_factor"] == POST_NFP_SCALING


# ══════════════════════════════════════════════════════════════════════════════
# E. compute_composite_scaling() — G4 per-type minimums
# ══════════════════════════════════════════════════════════════════════════════

class TestCompositeScaling:

    def test_empty_events_returns_1_0(self):
        """No events -> scaling factor is 1.0 (no restriction)."""
        assert compute_composite_scaling([]) == 1.0

    def test_fomc_only(self):
        """Single FOMC event -> composite equals its scaling factor."""
        events = [{"event_type": "FOMC", "scaling_factor": 0.50}]
        assert compute_composite_scaling(events) == 0.50

    def test_cpi_only(self):
        """Single CPI event -> composite equals its scaling factor."""
        events = [{"event_type": "CPI", "scaling_factor": 0.65}]
        assert compute_composite_scaling(events) == 0.65

    def test_fomc_plus_cpi_concurrent(self):
        """FOMC (0.50) + CPI (0.65) same week -> min(0.50, 0.65) = 0.50."""
        events = [
            {"event_type": "FOMC", "scaling_factor": 0.50},
            {"event_type": "CPI", "scaling_factor": 0.65},
        ]
        assert compute_composite_scaling(events) == 0.50

    def test_separates_fomc_from_data_events(self):
        """FOMC_POST (0.70) + NFP (0.75) -> min(fomc_floor=0.70, data_floor=0.75) = 0.70."""
        events = [
            {"event_type": "FOMC_POST", "scaling_factor": 0.70},
            {"event_type": "NFP", "scaling_factor": 0.75},
        ]
        assert compute_composite_scaling(events) == 0.70


# ══════════════════════════════════════════════════════════════════════════════
# F. run_daily_event_check() — DB persistence integration
# ══════════════════════════════════════════════════════════════════════════════

class TestRunDailyEventCheck:

    def test_persists_scaling_factor_to_db(self, test_db):
        """run_daily_event_check writes event_scaling_factor to macro_state table."""
        import sqlite3
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]
        run_daily_event_check(as_of=fomc_day, db_path=test_db)
        conn = sqlite3.connect(test_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT value FROM macro_state WHERE key = 'event_scaling_factor'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert float(row["value"]) <= 1.0

    def test_persists_last_daily_check_date(self, test_db):
        """run_daily_event_check writes last_daily_check to macro_state table."""
        import sqlite3
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]
        run_daily_event_check(as_of=fomc_day, db_path=test_db)
        conn = sqlite3.connect(test_db)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT value FROM macro_state WHERE key = 'last_daily_check'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["value"] == fomc_day.strftime("%Y-%m-%d")

    def test_returns_scaling_and_events_tuple(self, test_db):
        """Return value is (float, list) tuple."""
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]
        result = run_daily_event_check(as_of=fomc_day, db_path=test_db)
        assert isinstance(result, tuple)
        assert len(result) == 2
        scaling, events = result
        assert isinstance(scaling, float)
        assert isinstance(events, list)
