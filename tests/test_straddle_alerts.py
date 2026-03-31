"""Tests for straddle Telegram alert formatting and notifications.

Covers:
- TelegramAlertFormatter straddle entry alerts (debit/credit, breakevens, event)
- format_straddle_open() trade notification
- format_event_warning() pre-event alert
- notify_trade_open() routing for straddles
- notify_trade_close() with straddle close reasons
- notify_upcoming_events() integration
- Preflight check straddle validation
"""

import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from alerts.alert_schema import (
    Alert, AlertType, Confidence, Direction, Leg,
    TimeSensitivity,
)
from alerts.formatters.telegram import TelegramAlertFormatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_straddle_alert(**overrides):
    """Build a straddle Alert object for testing."""
    defaults = {
        "type": AlertType.straddle_strangle,
        "ticker": "SPY",
        "direction": Direction.neutral,
        "legs": [
            Leg(strike=500.0, option_type="call", action="sell", expiration="2026-04-17"),
            Leg(strike=500.0, option_type="put", action="sell", expiration="2026-04-17"),
        ],
        "entry_price": 8.50,
        "stop_loss": 12.75,
        "profit_target": 4.25,
        "risk_pct": 0.03,
        "confidence": Confidence.HIGH,
        "thesis": "Post-FOMC IV crush expected",
        "time_sensitivity": TimeSensitivity.IMMEDIATE,
        "management_instructions": "Close at 55% profit or 45% loss",
        "score": 72.0,
    }
    defaults.update(overrides)
    return Alert(**defaults)


def _make_long_straddle_alert(**overrides):
    """Build a long (debit) straddle Alert.

    Note: Alert.entry_price must be > 0 per schema validation.
    For debit positions, entry_price is the debit amount (positive).
    The formatter detects debit from leg actions (buy legs).
    """
    defaults = {
        "type": AlertType.straddle_strangle,
        "ticker": "SPY",
        "direction": Direction.neutral,
        "legs": [
            Leg(strike=500.0, option_type="call", action="buy", expiration="2026-04-17"),
            Leg(strike=500.0, option_type="put", action="buy", expiration="2026-04-17"),
        ],
        "entry_price": 6.00,  # debit amount (positive per schema)
        "stop_loss": 6.00,
        "profit_target": 9.00,
        "risk_pct": 0.03,
        "confidence": Confidence.MEDIUM,
        "thesis": "Pre-CPI vol expansion play",
        "time_sensitivity": TimeSensitivity.WITHIN_1HR,
        "management_instructions": "Exit before event if IV spikes 30%+",
        "score": 65.0,
    }
    defaults.update(overrides)
    return Alert(**defaults)


# ===========================================================================
# Formatter tests
# ===========================================================================

class TestStraddleEntryAlert(unittest.TestCase):
    """Test format_entry_alert for straddle alerts."""

    def setUp(self):
        self.fmt = TelegramAlertFormatter()

    def test_short_straddle_shows_credit(self):
        """Short straddle shows credit, not debit."""
        alert = _make_straddle_alert()
        msg = self.fmt.format_entry_alert(alert)

        assert "STRADDLE/STRANGLE" in msg
        assert "Credit: $8.50" in msg
        assert "Debit" not in msg.split("Credit")[0]  # no "Debit" before Credit

    def test_long_straddle_shows_debit(self):
        """Long straddle shows debit amount."""
        alert = _make_long_straddle_alert()
        msg = self.fmt.format_entry_alert(alert)

        assert "STRADDLE/STRANGLE" in msg
        assert "Debit: $6.00" in msg

    def test_straddle_shows_breakevens(self):
        """Straddle alert includes upper/lower breakeven prices."""
        alert = _make_straddle_alert(entry_price=8.50)
        msg = self.fmt.format_entry_alert(alert)

        # Short straddle: BE = strike ± premium
        assert "Upper BE: $508.50" in msg
        assert "Lower BE: $491.50" in msg

    def test_long_straddle_breakevens(self):
        """Long straddle breakevens use debit as distance."""
        alert = _make_long_straddle_alert(entry_price=6.00)
        msg = self.fmt.format_entry_alert(alert)

        # Long straddle: BE = strike ± total premium
        assert "Upper BE: $506.00" in msg
        assert "Lower BE: $494.00" in msg

    def test_straddle_shows_purple_emoji(self):
        """Straddle uses purple circle emoji."""
        alert = _make_straddle_alert()
        msg = self.fmt.format_entry_alert(alert)

        assert "\U0001f7e3" in msg  # purple circle

    def test_straddle_shows_score(self):
        """Alert includes score."""
        alert = _make_straddle_alert(score=72.0)
        msg = self.fmt.format_entry_alert(alert)

        assert "Score: 72/100" in msg

    def test_straddle_shows_risk_pct(self):
        """Alert includes risk percentage."""
        alert = _make_straddle_alert(risk_pct=0.03)
        msg = self.fmt.format_entry_alert(alert)

        assert "3.0%" in msg

    def test_straddle_event_type_in_metadata(self):
        """If alert has metadata with event_type, it's shown."""
        alert = _make_straddle_alert()
        alert.metadata = {"event_type": "fomc", "regime": "high_vol"}
        msg = self.fmt.format_entry_alert(alert)

        assert "Event: FOMC" in msg
        assert "Regime: high_vol" in msg


# ===========================================================================
# Trade open notification
# ===========================================================================

class TestFormatStraddleOpen(unittest.TestCase):
    """Test format_straddle_open for trade notifications."""

    def setUp(self):
        self.fmt = TelegramAlertFormatter()

    def test_short_straddle_open(self):
        """Short straddle open shows credit amount."""
        trade = {
            "ticker": "SPY",
            "type": "short_straddle",
            "call_strike": 500.0,
            "put_strike": 500.0,
            "contracts": 2,
            "credit": 8.50,
            "is_debit": False,
            "dte_at_entry": 33,
        }
        msg = self.fmt.format_straddle_open(trade)

        assert "SPY" in msg
        assert "Short Straddle" in msg
        assert "SHORT (credit)" in msg
        assert "Call: $500.0" in msg
        assert "Put: $500.0" in msg
        assert "Credit: $1700.00" in msg  # 8.50 * 2 * 100

    def test_long_straddle_open(self):
        """Long straddle open shows debit amount."""
        trade = {
            "ticker": "SPY",
            "type": "long_straddle",
            "call_strike": 500.0,
            "put_strike": 500.0,
            "contracts": 1,
            "credit": -6.00,
            "is_debit": True,
            "dte_at_entry": 5,
            "event_type": "cpi",
        }
        msg = self.fmt.format_straddle_open(trade)

        assert "LONG (debit)" in msg
        assert "Debit: $600.00" in msg  # abs(-6.00) * 1 * 100
        assert "Event: CPI" in msg

    def test_strangle_shows_different_strikes(self):
        """Strangle with different call/put strikes."""
        trade = {
            "ticker": "SPY",
            "type": "short_strangle",
            "call_strike": 510.0,
            "put_strike": 490.0,
            "contracts": 1,
            "credit": 5.00,
            "is_debit": False,
        }
        msg = self.fmt.format_straddle_open(trade)

        assert "Call: $510.0" in msg
        assert "Put: $490.0" in msg


# ===========================================================================
# Pre-event warning
# ===========================================================================

class TestFormatEventWarning(unittest.TestCase):
    """Test format_event_warning for pre-market alerts."""

    def setUp(self):
        self.fmt = TelegramAlertFormatter()

    def test_single_event(self):
        """Formats a single upcoming event."""
        events = [{
            "event_type": "fomc",
            "date": datetime(2026, 3, 18, 14, 0, tzinfo=timezone.utc),
            "description": "FOMC rate decision",
            "importance": 1.0,
        }]
        msg = self.fmt.format_event_warning(events)

        assert "UPCOMING ECONOMIC EVENTS" in msg
        assert "FOMC" in msg
        assert "2026-03-18" in msg
        assert "HIGH impact" in msg

    def test_multiple_events(self):
        """Formats multiple upcoming events."""
        events = [
            {
                "event_type": "cpi",
                "date": datetime(2026, 3, 11, 8, 30, tzinfo=timezone.utc),
                "description": "CPI inflation report",
                "importance": 0.85,
            },
            {
                "event_type": "ppi",
                "date": datetime(2026, 3, 12, 8, 30, tzinfo=timezone.utc),
                "description": "PPI producer prices",
                "importance": 0.70,
            },
        ]
        msg = self.fmt.format_event_warning(events)

        assert "CPI" in msg
        assert "PPI" in msg
        assert "HIGH impact" in msg  # CPI importance 0.85
        assert "MEDIUM impact" in msg  # PPI importance 0.70

    def test_low_importance_event(self):
        """Low importance events show LOW impact."""
        events = [{
            "event_type": "gdp",
            "date": datetime(2026, 4, 30, 8, 30, tzinfo=timezone.utc),
            "description": "GDP report",
            "importance": 0.65,
        }]
        msg = self.fmt.format_event_warning(events)

        assert "LOW impact" in msg


# ===========================================================================
# notify_trade_open routing
# ===========================================================================

class TestNotifyTradeOpen(unittest.TestCase):
    """Test that notify_trade_open routes straddles to dedicated formatter."""

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_straddle_uses_formatter(self, mock_send):
        """Straddle trade open uses TelegramAlertFormatter.format_straddle_open."""
        from shared.telegram_alerts import notify_trade_open

        trade = {
            "ticker": "SPY",
            "type": "short_straddle",
            "call_strike": 500.0,
            "put_strike": 500.0,
            "contracts": 1,
            "credit": 8.50,
            "is_debit": False,
        }
        result = notify_trade_open(trade)

        assert result is True
        msg = mock_send.call_args[0][0]
        assert "SHORT (credit)" in msg

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_strangle_uses_formatter(self, mock_send):
        """Strangle trade open also uses dedicated formatter."""
        from shared.telegram_alerts import notify_trade_open

        trade = {
            "ticker": "SPY",
            "type": "long_strangle",
            "call_strike": 510.0,
            "put_strike": 490.0,
            "contracts": 1,
            "credit": -4.00,
            "is_debit": True,
        }
        result = notify_trade_open(trade)

        assert result is True
        msg = mock_send.call_args[0][0]
        assert "LONG (debit)" in msg

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_credit_spread_uses_legacy_format(self, mock_send):
        """Credit spread still uses the original format."""
        from shared.telegram_alerts import notify_trade_open

        trade = {
            "ticker": "SPY",
            "type": "bull_put",
            "short_strike": 490.0,
            "long_strike": 480.0,
            "contracts": 1,
            "total_credit": 150.0,
            "total_max_loss": 850.0,
            "dte_at_entry": 30,
        }
        result = notify_trade_open(trade)

        assert result is True
        msg = mock_send.call_args[0][0]
        assert "NEW TRADE: SPY Bull Put" in msg
        assert "Strikes: $490.0/480.0" in msg


# ===========================================================================
# notify_trade_close
# ===========================================================================

class TestNotifyTradeClose(unittest.TestCase):
    """Test notify_trade_close with straddle-related close reasons."""

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_straddle_close_profit(self, mock_send):
        from shared.telegram_alerts import notify_trade_close

        trade = {"ticker": "SPY", "type": "short_straddle"}
        notify_trade_close(trade, pnl=500.0, reason="profit_target", balance=100500.0)

        msg = mock_send.call_args[0][0]
        assert "CLOSED: SPY Short Straddle" in msg
        assert "+$500.00" in msg
        assert "Profit Target Hit" in msg

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_expiration_today_reason(self, mock_send):
        from shared.telegram_alerts import notify_trade_close

        trade = {"ticker": "SPY", "type": "long_straddle"}
        notify_trade_close(trade, pnl=-300.0, reason="expiration_today", balance=99700.0)

        msg = mock_send.call_args[0][0]
        assert "Expiring Today" in msg

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_closed_external_reason(self, mock_send):
        from shared.telegram_alerts import notify_trade_close

        trade = {"ticker": "SPY", "type": "short_strangle"}
        notify_trade_close(trade, pnl=0.0, reason="closed_external", balance=100000.0)

        msg = mock_send.call_args[0][0]
        assert "Closed Externally" in msg


# ===========================================================================
# notify_upcoming_events
# ===========================================================================

class TestNotifyUpcomingEvents(unittest.TestCase):
    """Test pre-event warning notification."""

    @patch("shared.telegram_alerts.send_message", return_value=True)
    def test_sends_alert_when_events_upcoming(self, mock_send):
        from shared.telegram_alerts import notify_upcoming_events

        # Use a date just before a known FOMC date
        with patch("shared.economic_calendar.EconomicCalendar.get_upcoming_events") as mock_events:
            mock_events.return_value = [{
                "event_type": "fomc",
                "date": datetime(2026, 3, 18, 14, 0, tzinfo=timezone.utc),
                "description": "FOMC rate decision",
                "importance": 1.0,
            }]
            result = notify_upcoming_events(days_ahead=2)

        assert result is True
        msg = mock_send.call_args[0][0]
        assert "UPCOMING ECONOMIC EVENTS" in msg
        assert "FOMC" in msg

    @patch("shared.telegram_alerts.send_message")
    def test_no_alert_when_no_events(self, mock_send):
        from shared.telegram_alerts import notify_upcoming_events

        with patch("shared.economic_calendar.EconomicCalendar.get_upcoming_events") as mock_events:
            mock_events.return_value = []
            result = notify_upcoming_events(days_ahead=2)

        assert result is False
        mock_send.assert_not_called()


# ===========================================================================
# Preflight straddle validation
# ===========================================================================

class TestPreflightStraddleValidation(unittest.TestCase):
    """Test preflight_check validates straddle config."""

    def test_valid_straddle_config_passes(self):
        from scripts.preflight_check import validate

        config = {
            "db_path": "data/test.db",
            "experiment_id": "EXP-TEST",
            "paper_mode": True,
            "logging": {"level": "INFO", "file": "test.log"},
            "strategy": {
                "min_delta": 0.08,
                "max_delta": 0.18,
                "straddle_strangle": {
                    "enabled": True,
                    "profit_target_pct": 0.55,
                    "stop_loss_pct": 0.45,
                    "max_risk_pct": 3.0,
                },
            },
            "risk": {
                "straddle_strangle_risk_pct": 3.0,
                "regime_scale_crash": 0,
            },
        }
        errors = validate(config)
        assert errors == []

    def test_missing_straddle_fields_errors(self):
        from scripts.preflight_check import validate

        config = {
            "db_path": "data/test.db",
            "experiment_id": "EXP-TEST",
            "paper_mode": True,
            "logging": {"level": "INFO", "file": "test.log"},
            "strategy": {
                "min_delta": 0.08,
                "max_delta": 0.18,
                "straddle_strangle": {
                    "enabled": True,
                    # Missing profit_target_pct, stop_loss_pct, max_risk_pct
                },
            },
            "risk": {},  # Missing straddle_strangle_risk_pct
        }
        errors = validate(config)

        assert any("profit_target_pct" in e for e in errors)
        assert any("stop_loss_pct" in e for e in errors)
        assert any("max_risk_pct" in e for e in errors)
        assert any("straddle_strangle_risk_pct" in e for e in errors)

    def test_disabled_straddle_no_errors(self):
        """When straddle_strangle is disabled, no validation errors."""
        from scripts.preflight_check import validate

        config = {
            "db_path": "data/test.db",
            "experiment_id": "EXP-TEST",
            "paper_mode": True,
            "logging": {"level": "INFO", "file": "test.log"},
            "strategy": {
                "min_delta": 0.08,
                "max_delta": 0.18,
                "straddle_strangle": {"enabled": False},
            },
            "risk": {},
        }
        errors = validate(config)
        assert errors == []

    def test_crash_regime_scale_nonzero_warns(self):
        """regime_scale_crash != 0 produces an error."""
        from scripts.preflight_check import validate

        config = {
            "db_path": "data/test.db",
            "experiment_id": "EXP-TEST",
            "paper_mode": True,
            "logging": {"level": "INFO", "file": "test.log"},
            "strategy": {"min_delta": 0.08, "max_delta": 0.18},
            "risk": {"regime_scale_crash": 0.5},
        }
        errors = validate(config)
        assert any("regime_scale_crash" in e for e in errors)


if __name__ == "__main__":
    unittest.main()
