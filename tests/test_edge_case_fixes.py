"""Tests for Task 3 edge case fixes:
  A) Straddle partial fill recovery — cancel retry with backoff + CRITICAL alert
  B) Stale close order retry — auto cancel-and-resubmit up to 3 times
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from execution.execution_engine import ExecutionEngine
from execution.position_monitor import PositionMonitor, _STALE_CLOSE_MAX_RETRIES
from shared.database import init_db, upsert_trade


# ─── helpers ──────────────────────────────────────────────────────────────────

def _mock_alpaca():
    alpaca = MagicMock()
    alpaca.submit_single_leg = MagicMock(return_value={"status": "submitted", "order_id": "oid-call"})
    alpaca.cancel_order = MagicMock(return_value={"status": "cancelled"})
    alpaca.is_market_open = MagicMock(return_value=True)
    return alpaca


def _engine(alpaca=None):
    return ExecutionEngine(alpaca_provider=alpaca or _mock_alpaca(), config={})


def _straddle_opp(ticker="SPY"):
    return {
        "ticker": ticker,
        "type": "short_straddle",
        "expiration": "2025-06-20",
        "call_strike": 450.0,
        "put_strike": 450.0,
        "credit": 8.0,
    }


def _open_pos(trade_id="T001", ticker="SPY", strategy_type="bull_put_spread",
              submitted_at=None, retry_count=0):
    """Build a minimal pending_close position dict."""
    if submitted_at is None:
        submitted_at = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
    return {
        "id": trade_id,
        "ticker": ticker,
        "strategy_type": strategy_type,
        "status": "pending_close",
        "contracts": 1,
        "expiration": "2025-06-20",
        "short_strike": 450.0,
        "long_strike": 445.0,
        "credit": 1.50,
        "entry_date": "2025-05-01",
        "exit_reason": "profit_target",
        "close_order_id": "oid-close-001",
        "close_order_submitted_at": submitted_at,
        "close_order_retry_count": retry_count,
        "source": "execution",
    }


def _monitor_with_db(db_path, alpaca=None, config=None):
    cfg = config or {"risk": {"profit_target": 50, "stop_loss_multiplier": 2.5}}
    monitor = PositionMonitor(alpaca_provider=alpaca, config=cfg, db_path=db_path)
    return monitor


# ─── Task 3A: Straddle partial fill recovery ──────────────────────────────────

class TestStraddlePartialFillRecovery:
    """_cancel_with_retry: 3 attempts, exponential backoff, CRITICAL alert on failure."""

    def test_cancel_succeeds_on_first_attempt(self):
        """If cancel works immediately, no retry and no alert."""
        alpaca = _mock_alpaca()
        engine = _engine(alpaca)
        result = engine._cancel_with_retry("oid-123", context="test")
        assert result is True
        assert alpaca.cancel_order.call_count == 1

    def test_cancel_succeeds_on_second_attempt(self):
        alpaca = _mock_alpaca()
        alpaca.cancel_order.side_effect = [Exception("timeout"), {"status": "cancelled"}]
        engine = _engine(alpaca)
        with patch("time.sleep"):
            result = engine._cancel_with_retry("oid-123", context="test")
        assert result is True
        assert alpaca.cancel_order.call_count == 2

    def test_cancel_retries_three_times_then_fails(self):
        alpaca = _mock_alpaca()
        alpaca.cancel_order.side_effect = Exception("network error")
        engine = _engine(alpaca)
        with patch("time.sleep"), \
             patch("shared.telegram_alerts.notify_api_failure") as mock_alert:
            result = engine._cancel_with_retry("oid-123", context="rollback", max_attempts=3)
        assert result is False
        assert alpaca.cancel_order.call_count == 3

    def test_critical_telegram_alert_on_cancel_failure(self):
        alpaca = _mock_alpaca()
        alpaca.cancel_order.side_effect = Exception("network error")
        engine = _engine(alpaca)
        with patch("time.sleep"), \
             patch("shared.telegram_alerts.notify_api_failure") as mock_alert:
            engine._cancel_with_retry("oid-xyz", context="straddle_test", max_attempts=3)
        mock_alert.assert_called_once()
        call_kwargs = mock_alert.call_args
        assert "straddle_test" in str(call_kwargs)

    def test_submit_straddle_calls_cancel_on_put_failure(self):
        """If put leg fails, _submit_straddle should attempt to cancel the call leg."""
        alpaca = _mock_alpaca()
        alpaca.submit_single_leg.side_effect = [
            {"status": "submitted", "order_id": "oid-call"},   # call succeeds
            {"status": "rejected", "message": "insufficient buying power"},  # put fails
        ]
        engine = _engine(alpaca)
        with patch("time.sleep"):
            result = engine._submit_straddle(_straddle_opp(), contracts=1, credit=8.0, client_id="T001")
        assert result["status"] == "partial_error"
        alpaca.cancel_order.assert_called()

    def test_submit_straddle_marks_manual_review_after_cancel_failure(self):
        """If cancel fails all retries, the trade is marked partial_fill_manual_review in DB."""
        alpaca = _mock_alpaca()
        alpaca.submit_single_leg.side_effect = [
            {"status": "submitted", "order_id": "oid-call"},
            {"status": "rejected", "message": "error"},
        ]
        alpaca.cancel_order.side_effect = Exception("cancel failed")
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            init_db(db_path)
            # Pre-seed the trade so upsert can find it
            upsert_trade(
                {"id": "T001", "ticker": "SPY", "status": "pending_open", "expiration": "2025-06-20"},
                source="execution", path=db_path,
            )
            engine = ExecutionEngine(alpaca_provider=alpaca, db_path=db_path, config={})
            with patch("time.sleep"), \
                 patch("shared.telegram_alerts.notify_api_failure"):
                engine._submit_straddle(_straddle_opp(), contracts=1, credit=8.0, client_id="T001")
            from shared.database import get_trade_by_id
            trade = get_trade_by_id("T001", path=db_path)
            assert trade is not None
            assert trade["status"] == "partial_fill_manual_review"
        finally:
            os.unlink(db_path)

    def test_backoff_sleep_called_between_retries(self):
        """Exponential backoff (2^attempt) is applied between retries."""
        alpaca = _mock_alpaca()
        alpaca.cancel_order.side_effect = Exception("fail")
        engine = _engine(alpaca)
        with patch("time.sleep") as mock_sleep, \
             patch("shared.telegram_alerts.notify_api_failure"):
            engine._cancel_with_retry("oid", context="test", max_attempts=3, backoff_base=2.0)
        # Should sleep after attempt 1 and 2, but not 3 (last attempt)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(2.0)  # 2^1
        mock_sleep.assert_any_call(4.0)  # 2^2


# ─── Task 3B: Stale close order retry ────────────────────────────────────────

class TestStaleCloseOrderRetry:
    """_check_stale_close: cancel + resubmit up to 3 times, then CRITICAL alert."""

    def _setup_db(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db_path = f.name
        f.close()
        init_db(db_path)
        return db_path

    def test_not_stale_within_threshold_does_nothing(self):
        """If submitted < 10 min ago, no retry triggered."""
        alpaca = MagicMock()
        db_path = self._setup_db()
        try:
            monitor = _monitor_with_db(db_path, alpaca=alpaca)
            fresh_at = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
            pos = _open_pos(submitted_at=fresh_at)
            monitor._check_stale_close(pos, "oid-001", "new")
            alpaca.cancel_order.assert_not_called()
        finally:
            os.unlink(db_path)

    def test_stale_order_triggers_cancel_and_resubmit(self):
        """A stale order (15 min old) should be cancelled and resubmitted."""
        alpaca = MagicMock()
        alpaca.cancel_order = MagicMock()
        alpaca.close_spread = MagicMock(return_value={"status": "submitted", "order_id": "new-oid"})
        db_path = self._setup_db()
        try:
            pos = _open_pos(retry_count=0)
            upsert_trade(pos, source="execution", path=db_path)
            monitor = _monitor_with_db(db_path, alpaca=alpaca)
            monitor._check_stale_close(pos, "oid-close-001", "new")
            alpaca.cancel_order.assert_called_once_with("oid-close-001")
        finally:
            os.unlink(db_path)

    def test_retry_count_incremented_after_stale_retry(self):
        """close_order_retry_count should be incremented in the position dict."""
        alpaca = MagicMock()
        alpaca.cancel_order = MagicMock()
        alpaca.close_spread = MagicMock(return_value={"status": "submitted", "order_id": "new-oid"})
        db_path = self._setup_db()
        try:
            pos = _open_pos(retry_count=0)
            upsert_trade(pos, source="execution", path=db_path)
            monitor = _monitor_with_db(db_path, alpaca=alpaca)
            monitor._check_stale_close(pos, "oid-close-001", "new")
            assert pos["close_order_retry_count"] == 1
        finally:
            os.unlink(db_path)

    def test_second_retry_increments_to_two(self):
        alpaca = MagicMock()
        alpaca.cancel_order = MagicMock()
        alpaca.close_spread = MagicMock(return_value={"status": "submitted", "order_id": "new-oid"})
        db_path = self._setup_db()
        try:
            pos = _open_pos(retry_count=1)
            upsert_trade(pos, source="execution", path=db_path)
            monitor = _monitor_with_db(db_path, alpaca=alpaca)
            monitor._check_stale_close(pos, "oid-close-001", "new")
            assert pos["close_order_retry_count"] == 2
        finally:
            os.unlink(db_path)

    def test_max_retries_sends_telegram_alert(self):
        """When retry_count >= max, sends Telegram alert and stops retrying."""
        alpaca = MagicMock()
        db_path = self._setup_db()
        try:
            pos = _open_pos(retry_count=_STALE_CLOSE_MAX_RETRIES)
            monitor = _monitor_with_db(db_path, alpaca=alpaca)
            with patch("execution.position_monitor.notify_api_failure") as mock_alert:
                monitor._check_stale_close(pos, "oid-close-001", "new")
            mock_alert.assert_called_once()
            # No more cancel attempts after exhausting retries
            alpaca.cancel_order.assert_not_called()
        finally:
            os.unlink(db_path)

    def test_max_retries_no_further_resubmit(self):
        """After max retries, _close_position is NOT called again."""
        alpaca = MagicMock()
        db_path = self._setup_db()
        try:
            pos = _open_pos(retry_count=_STALE_CLOSE_MAX_RETRIES)
            monitor = _monitor_with_db(db_path, alpaca=alpaca)
            with patch.object(monitor, "_close_position") as mock_close, \
                 patch("execution.position_monitor.notify_api_failure"):
                monitor._check_stale_close(pos, "oid-close-001", "new")
            mock_close.assert_not_called()
        finally:
            os.unlink(db_path)

    def test_stale_dual_leg_close_cancels_put_order_too(self):
        """For straddle (dual-leg) stale closes, both orders are cancelled."""
        alpaca = MagicMock()
        alpaca.cancel_order = MagicMock()
        alpaca.submit_single_leg = MagicMock(return_value={"status": "submitted", "order_id": "x"})
        db_path = self._setup_db()
        try:
            pos = _open_pos(strategy_type="short_straddle", retry_count=0)
            pos["close_put_order_id"] = "oid-put-001"
            upsert_trade(pos, source="execution", path=db_path)
            monitor = _monitor_with_db(db_path, alpaca=alpaca)
            monitor._check_stale_close(pos, "oid-close-001", "new")
            cancelled_orders = [c.args[0] for c in alpaca.cancel_order.call_args_list]
            assert "oid-close-001" in cancelled_orders
            assert "oid-put-001" in cancelled_orders
        finally:
            os.unlink(db_path)

    def test_cancel_failure_does_not_prevent_resubmit(self):
        """Even if cancel raises, we still attempt to resubmit the close."""
        alpaca = MagicMock()
        alpaca.cancel_order.side_effect = Exception("network error")
        alpaca.close_spread = MagicMock(return_value={"status": "submitted", "order_id": "new-oid"})
        db_path = self._setup_db()
        try:
            pos = _open_pos(retry_count=0)
            upsert_trade(pos, source="execution", path=db_path)
            monitor = _monitor_with_db(db_path, alpaca=alpaca)
            with patch.object(monitor, "_close_position") as mock_close:
                monitor._check_stale_close(pos, "oid-close-001", "new")
            mock_close.assert_called_once()
        finally:
            os.unlink(db_path)
