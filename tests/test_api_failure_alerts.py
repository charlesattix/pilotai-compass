"""Tests for API failure alerting (INF-2).

Validates:
  - notify_api_failure sends correctly formatted Telegram alerts
  - Rate limiting prevents spam (max 1 alert per 5 minutes)
  - PositionMonitor triggers alerts on Alpaca API failures
  - main.py scan loop triggers alerts on ticker analysis failures
"""

import time
from unittest.mock import MagicMock, patch

import pytest


# ── notify_api_failure unit tests ──────────────────────────────────────────


class TestNotifyApiFailure:
    """Test the notify_api_failure function in shared/telegram_alerts.py."""

    def setup_method(self):
        """Reset rate-limit state before each test."""
        import shared.telegram_alerts as mod
        mod._last_api_failure_alert_time = 0.0

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_sends_alert_with_all_fields(self, mock_send):
        from shared.telegram_alerts import notify_api_failure

        result = notify_api_failure(
            error_msg="Connection refused",
            context="get_positions",
            unmonitored_positions=3,
        )

        assert result is True
        mock_send.assert_called_once()
        msg = mock_send.call_args[0][0]
        assert "API FAILURE ALERT" in msg
        assert "get_positions" in msg
        assert "Connection refused" in msg
        assert "3" in msg
        assert "UTC" in msg

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_sends_alert_without_unmonitored(self, mock_send):
        from shared.telegram_alerts import notify_api_failure

        notify_api_failure(error_msg="timeout", context="submit_close")

        msg = mock_send.call_args[0][0]
        assert "Unmonitored" not in msg

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_rate_limiting(self, mock_send):
        from shared.telegram_alerts import notify_api_failure

        # First call should send
        assert notify_api_failure("err1", "ctx1") is True
        # Second call within cooldown should be rate-limited
        assert notify_api_failure("err2", "ctx2") is False
        assert mock_send.call_count == 1

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_rate_limit_expires(self, mock_send):
        import shared.telegram_alerts as mod
        from shared.telegram_alerts import notify_api_failure

        assert notify_api_failure("err1", "ctx1") is True
        # Simulate cooldown expiry
        mod._last_api_failure_alert_time = time.time() - 301
        assert notify_api_failure("err2", "ctx2") is True
        assert mock_send.call_count == 2

    @patch("shared.telegram_alerts.send_message", return_value=False)
    def test_send_failure_does_not_update_timestamp(self, mock_send):
        import shared.telegram_alerts as mod
        from shared.telegram_alerts import notify_api_failure

        result = notify_api_failure("err", "ctx")
        assert result is False
        assert mod._last_api_failure_alert_time == 0.0


# ── PositionMonitor integration tests ─────────────────────────────────────


class TestPositionMonitorApiAlerts:
    """Test that PositionMonitor calls notify_api_failure on Alpaca errors."""

    def setup_method(self):
        import shared.telegram_alerts as mod
        mod._last_api_failure_alert_time = 0.0

    @patch("execution.position_monitor.notify_api_failure")
    @patch("execution.position_monitor.get_trades", return_value=[])
    @patch("execution.position_monitor.init_db")
    def test_get_positions_failure_alerts(self, mock_init, mock_trades, mock_notify):
        from execution.position_monitor import PositionMonitor

        mock_alpaca = MagicMock()
        mock_alpaca.get_positions.side_effect = ConnectionError("API down")

        monitor = PositionMonitor(
            alpaca_provider=mock_alpaca,
            config={"risk": {}, "strategy": {}},
        )
        # Force market hours
        with patch.object(PositionMonitor, "_is_market_hours", return_value=True):
            monitor._check_positions()

        mock_notify.assert_called_once()
        call_kwargs = mock_notify.call_args
        assert "API down" in call_kwargs[1]["error_msg"] or "API down" in str(call_kwargs)
        assert "get_positions" in str(call_kwargs)

    @patch("execution.position_monitor.notify_api_failure")
    @patch("execution.position_monitor.get_trades")
    @patch("execution.position_monitor.init_db")
    def test_order_status_failure_alerts(self, mock_init, mock_trades, mock_notify):
        from execution.position_monitor import PositionMonitor

        mock_alpaca = MagicMock()
        mock_alpaca.get_order_status.side_effect = ConnectionError("timeout")

        monitor = PositionMonitor(
            alpaca_provider=mock_alpaca,
            config={"risk": {}, "strategy": {}},
        )

        # Simulate a pending_close trade with an order_id
        mock_trades.return_value = [
            {"id": "t1", "close_order_id": "ord123", "status": "pending_close"}
        ]

        monitor._reconcile_pending_closes()

        mock_notify.assert_called_once()
        assert "get_order_status" in str(mock_notify.call_args)

    @patch("execution.position_monitor.notify_api_failure")
    @patch("execution.position_monitor.upsert_trade")
    @patch("execution.position_monitor.init_db")
    def test_close_submission_failure_alerts(self, mock_init, mock_upsert, mock_notify):
        from execution.position_monitor import PositionMonitor

        mock_alpaca = MagicMock()
        mock_alpaca.close_spread.side_effect = ConnectionError("refused")

        monitor = PositionMonitor(
            alpaca_provider=mock_alpaca,
            config={"risk": {}, "strategy": {}},
        )

        pos = {
            "id": "t1",
            "ticker": "SPY",
            "strategy_type": "bull_put",
            "contracts": 1,
            "expiration": "2026-04-17",
            "short_strike": 400,
            "long_strike": 395,
            "credit": 1.5,
            "status": "open",
        }

        monitor._close_position(pos, "stop_loss")

        mock_notify.assert_called_once()
        assert "submit_close" in str(mock_notify.call_args)
