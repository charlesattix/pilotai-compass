"""Tests for the scan scheduler."""

import threading
from datetime import datetime
from unittest.mock import MagicMock, patch

from shared.scheduler import (
    ET,
    MARKET_SCAN_TIMES,
    SCAN_TIMES,
    SLOT_DAILY_REPORT,
    SLOT_MACRO_WEEKLY,
    SLOT_PRE_MARKET,
    SLOT_RETRAIN,
    SLOT_SCAN,
    ScanScheduler,
    _is_weekday,
    _next_scan_time,
)


class TestIsWeekday:
    def test_monday_is_weekday(self):
        # 2026-02-16 is Monday
        dt = ET.localize(datetime(2026, 2, 16, 10, 0))
        assert _is_weekday(dt)

    def test_friday_is_weekday(self):
        dt = ET.localize(datetime(2026, 2, 20, 10, 0))
        assert _is_weekday(dt)

    def test_saturday_is_not_weekday(self):
        dt = ET.localize(datetime(2026, 2, 21, 10, 0))
        assert not _is_weekday(dt)

    def test_sunday_is_not_weekday(self):
        dt = ET.localize(datetime(2026, 2, 22, 10, 0))
        assert not _is_weekday(dt)


class TestNextScanTime:
    def test_before_first_scan(self):
        """Before 9:00 AM on a weekday → next slot is 9:00 (pre-market)."""
        now = ET.localize(datetime(2026, 2, 16, 8, 0))  # Monday 8:00
        nxt, slot_type = _next_scan_time(now)
        assert nxt.hour == 9
        assert nxt.minute == 0
        assert slot_type == SLOT_PRE_MARKET
        assert nxt.date() == now.date()

    def test_between_scans(self):
        """Between 9:15 and 9:45 → next scan is 9:45."""
        now = ET.localize(datetime(2026, 2, 16, 9, 20))  # Monday 9:20
        nxt, slot_type = _next_scan_time(now)
        assert nxt.hour == 9
        assert nxt.minute == 45
        assert slot_type == SLOT_SCAN

    def test_after_last_scan_before_daily_report(self):
        """After 3:30 PM but before 4:15 PM → next is daily report."""
        now = ET.localize(datetime(2026, 2, 16, 15, 35))
        nxt, slot_type = _next_scan_time(now)
        assert nxt.hour == 16
        assert nxt.minute == 15
        assert slot_type == SLOT_DAILY_REPORT

    def test_after_daily_report_goes_to_next_day(self):
        """After 4:15 PM on weekday → next slot is 9:00 AM next weekday."""
        now = ET.localize(datetime(2026, 2, 16, 16, 30))  # Monday 4:30 PM
        nxt, slot_type = _next_scan_time(now)
        assert nxt.hour == 9
        assert nxt.minute == 0
        assert slot_type == SLOT_PRE_MARKET
        assert nxt.date() == datetime(2026, 2, 17).date()  # Tuesday

    def test_friday_after_macro_weekly_skips_weekend(self):
        """Friday after 17:00 (macro_weekly slot) → next slot is Monday 9:00 AM."""
        now = ET.localize(datetime(2026, 2, 20, 17, 1))  # Friday 5:01 PM (past last slot)
        nxt, slot_type = _next_scan_time(now)
        assert nxt.hour == 9
        assert nxt.minute == 0
        assert nxt.weekday() == 0  # Monday

    def test_friday_before_macro_weekly_returns_macro_slot(self):
        """Friday 4:45 PM → next slot is macro_weekly at 17:00 (not Monday)."""
        now = ET.localize(datetime(2026, 2, 20, 16, 45))  # Friday 4:45 PM
        nxt, slot_type = _next_scan_time(now)
        assert nxt.hour == 17
        assert nxt.minute == 0
        assert slot_type == SLOT_MACRO_WEEKLY
        assert nxt.weekday() == 4  # still Friday

    def test_saturday_skips_to_monday(self):
        """Saturday → next slot is Monday 9:00 AM."""
        now = ET.localize(datetime(2026, 2, 21, 12, 0))  # Saturday noon
        nxt, slot_type = _next_scan_time(now)
        assert nxt.weekday() == 0  # Monday
        assert nxt.hour == 9
        assert nxt.minute == 0

    def test_all_18_scan_times_defined(self):
        assert len(SCAN_TIMES) == 18

    def test_market_scan_times_has_14_entries(self):
        """MARKET_SCAN_TIMES (backtester compat) should have 14 scan-only slots."""
        assert len(MARKET_SCAN_TIMES) == 14

    def test_scan_times_are_sorted(self):
        for i in range(len(SCAN_TIMES) - 1):
            a = SCAN_TIMES[i][0] * 60 + SCAN_TIMES[i][1]
            b = SCAN_TIMES[i + 1][0] * 60 + SCAN_TIMES[i + 1][1]
            assert a < b, f"SCAN_TIMES not sorted: {SCAN_TIMES[i]} >= {SCAN_TIMES[i+1]}"

    def test_slot_types_correct(self):
        """First=pre_market, then scans, then daily_report, retrain, macro_weekly."""
        assert SCAN_TIMES[0][2] == SLOT_PRE_MARKET
        assert SCAN_TIMES[-3][2] == SLOT_DAILY_REPORT
        assert SCAN_TIMES[-2][2] == SLOT_RETRAIN
        assert SCAN_TIMES[-1][2] == SLOT_MACRO_WEEKLY
        for h, m, s in SCAN_TIMES[1:-3]:
            assert s == SLOT_SCAN

    def test_retrain_slot_at_1630(self):
        """SLOT_RETRAIN fires at 16:30 ET."""
        retrain_entries = [(h, m, s) for h, m, s in SCAN_TIMES if s == SLOT_RETRAIN]
        assert len(retrain_entries) == 1
        h, m, _ = retrain_entries[0]
        assert (h, m) == (16, 30)

    def test_retrain_slot_after_daily_report(self):
        """SLOT_RETRAIN (16:30) must come after SLOT_DAILY_REPORT (16:15)."""
        dr_min = next(h * 60 + m for h, m, s in SCAN_TIMES if s == SLOT_DAILY_REPORT)
        rt_min = next(h * 60 + m for h, m, s in SCAN_TIMES if s == SLOT_RETRAIN)
        assert dr_min < rt_min


class TestScanScheduler:
    def test_stop_interrupts_wait(self):
        """Calling stop() should break out of run_forever quickly."""
        scan_fn = MagicMock()
        scheduler = ScanScheduler(scan_fn)

        # Stop after a short delay
        def stop_soon():
            import time
            time.sleep(0.1)
            scheduler.stop()

        threading.Thread(target=stop_soon, daemon=True).start()
        scheduler.run_forever()  # Should return quickly

        # Scan should NOT have been called (we stopped before any scan time)
        scan_fn.assert_not_called()

    def test_scan_runs_when_time_matches(self):
        """Scheduler should call scan_fn when the scheduled time arrives."""
        scan_fn = MagicMock()
        scheduler = ScanScheduler(scan_fn, startup_delay=0)

        with patch("shared.scheduler._next_scan_time") as mock:
            # Return (now, SLOT_SCAN) so wait is 0
            mock.side_effect = lambda now: (now, SLOT_SCAN)

            # Stop after brief delay
            def stop_soon():
                import time
                time.sleep(0.5)
                scheduler.stop()

            threading.Thread(target=stop_soon, daemon=True).start()

            # Patch weekday check to return True
            with patch("shared.scheduler._is_weekday", return_value=True):
                scheduler.run_forever()

        assert scan_fn.call_count >= 1
        # Verify slot_type was passed to scan_fn
        scan_fn.assert_called_with(SLOT_SCAN)
