"""Tests for alerts.formatters.telegram — format output verification."""


from alerts.alert_schema import (
    Alert,
    AlertType,
    Confidence,
    Direction,
    Leg,
    SizeResult,
    TimeSensitivity,
)
from alerts.formatters.telegram import TelegramAlertFormatter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert(**overrides):
    defaults = dict(
        type=AlertType.credit_spread,
        ticker="SPY",
        direction=Direction.bullish,
        legs=[
            Leg(strike=540.0, option_type="put", action="sell", expiration="2025-06-20"),
            Leg(strike=535.0, option_type="put", action="buy", expiration="2025-06-20"),
        ],
        entry_price=1.50,
        stop_loss=3.00,
        profit_target=0.75,
        risk_pct=0.02,
        confidence=Confidence.HIGH,
        thesis="SPY bullish above support",
        management_instructions="Close at 50% profit",
        time_sensitivity=TimeSensitivity.TODAY,
        score=78,
    )
    defaults.update(overrides)
    return Alert(**defaults)


# ---------------------------------------------------------------------------
# Entry alert tests
# ---------------------------------------------------------------------------

class TestFormatEntryAlert:
    def test_contains_ticker(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "SPY" in msg

    def test_contains_type_label(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "CREDIT SPREAD" in msg

    def test_contains_type_emoji(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert(type=AlertType.credit_spread))
        assert "\U0001f7e2" in msg  # green circle

    def test_all_type_emojis(self):
        fmt = TelegramAlertFormatter()
        expected = {
            AlertType.credit_spread: "\U0001f7e2",
            AlertType.momentum_swing: "\U0001f535",
            AlertType.iron_condor: "\U0001f7e1",
            AlertType.earnings_play: "\U0001f7e0",
            AlertType.gamma_lotto: "\U0001f534",
        }
        for atype, emoji in expected.items():
            msg = fmt.format_entry_alert(_make_alert(type=atype))
            assert emoji in msg, f"Missing emoji for {atype.value}"

    def test_contains_direction(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "BULLISH" in msg

    def test_contains_legs(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "SELL $540.00 PUT" in msg
        assert "BUY $535.00 PUT" in msg

    def test_contains_entry_price(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "$1.50" in msg

    def test_contains_stop_loss(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "$3.00" in msg

    def test_contains_profit_target(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "$0.75" in msg

    def test_contains_risk_pct(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "2.0%" in msg

    def test_contains_thesis(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "SPY bullish above support" in msg

    def test_contains_management(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "Close at 50% profit" in msg

    def test_contains_time_sensitivity(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "TODAY" in msg

    def test_contains_confidence(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "HIGH" in msg

    def test_contains_score(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "78/100" in msg

    def test_sizing_shown_when_present(self):
        fmt = TelegramAlertFormatter()
        alert = _make_alert()
        alert.sizing = SizeResult(risk_pct=0.02, contracts=3, dollar_risk=2000, max_loss=1050)
        msg = fmt.format_entry_alert(alert)
        assert "Contracts: 3" in msg
        assert "$1050.00" in msg

    def test_html_bold_tags(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_entry_alert(_make_alert())
        assert "<b>" in msg
        assert "</b>" in msg


# ---------------------------------------------------------------------------
# Exit alert tests
# ---------------------------------------------------------------------------

class TestFormatExitAlert:
    def test_contains_ticker_and_action(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_exit_alert(
            ticker="AAPL",
            action="CLOSE",
            current_pnl=150.0,
            pnl_pct=42.8,
            reason="Hit 50% profit target",
            instructions="Take profit, no action needed",
        )
        assert "AAPL" in msg
        assert "CLOSE" in msg

    def test_positive_pnl_format(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_exit_alert("SPY", "CLOSE", 200.0, 50.0, "Target", "None")
        assert "+$200.00" in msg
        assert "+50.0%" in msg

    def test_negative_pnl_format(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_exit_alert("SPY", "STOP", -300.0, -75.0, "Stop loss", "Review")
        assert "-$300.00" in msg
        assert "-75.0%" in msg


# ---------------------------------------------------------------------------
# Daily summary tests
# ---------------------------------------------------------------------------

class TestFormatDailySummary:
    def test_contains_all_fields(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_daily_summary(
            date="2025-06-20",
            alerts_fired=8,
            closed_today=3,
            wins=2,
            losses=1,
            day_pnl=450.0,
            day_pnl_pct=0.9,
            open_positions=5,
            total_risk_pct=8.5,
            account_balance=50_450.0,
            pct_from_start=0.9,
            best="SPY +$200",
            worst="AAPL -$50",
        )

        assert "2025-06-20" in msg
        assert "8" in msg        # alerts fired
        assert "W:2" in msg
        assert "L:1" in msg
        assert "$450.00" in msg
        assert "5" in msg        # open positions
        assert "8.5%" in msg     # total risk
        assert "$50,450.00" in msg
        assert "SPY +$200" in msg
        assert "AAPL -$50" in msg

    def test_negative_day(self):
        fmt = TelegramAlertFormatter()
        msg = fmt.format_daily_summary(
            date="2025-06-20",
            alerts_fired=2,
            closed_today=1,
            wins=0,
            losses=1,
            day_pnl=-200.0,
            day_pnl_pct=-0.4,
            open_positions=3,
            total_risk_pct=6.0,
            account_balance=49_800.0,
            pct_from_start=-0.4,
            best="N/A",
            worst="SPY -$200",
        )

        assert "-$200.00" in msg
        assert "\U0001f4c9" in msg  # down-trend emoji
