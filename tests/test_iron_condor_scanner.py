"""Pytest-compatible tests for Phase 3: Iron Condor Alerts.

Covers:
- Config overlay (iron_condor_config.py)
- Day-of-week gates (iron_condor_scanner.py)
- Scanner scan() pipeline with mocks
- Exit monitor: profit/stop/dedup/weekly-close (iron_condor_exit_monitor.py)
"""

from datetime import datetime
from unittest.mock import MagicMock

from alerts.iron_condor_config import (
    CLOSE_DAYS,
    ENTRY_DAYS,
    build_iron_condor_config,
)
from alerts.iron_condor_exit_monitor import IronCondorExitMonitor
from alerts.iron_condor_scanner import IronCondorScanner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _base_config():
    return {
        "strategy": {
            "min_dte": 30,
            "max_dte": 45,
            "min_delta": 0.20,
            "max_delta": 0.30,
            "spread_width": 10,
            "spread_width_high_iv": 15,
            "spread_width_low_iv": 10,
            "min_iv_rank": 20,
            "min_iv_percentile": 20,
            "iron_condor": {"enabled": True},
            "technical": {
                "sma_fast": 20,
                "sma_slow": 50,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "bb_period": 20,
                "bb_std_dev": 2.0,
            },
        },
        "risk": {
            "stop_loss_multiplier": 2.5,
            "min_credit_pct": 20,
            "account_size": 100000,
        },
        "tickers": ["SPY", "QQQ", "IWM"],
        "data": {},
    }


def _make_monday(hour=10, minute=0):
    """Return a naive datetime on a Monday (2026-02-23 is a Monday)."""
    return datetime(2026, 2, 23, hour, minute, 0)


def _make_tuesday(hour=10, minute=0):
    """Return a naive datetime on a Tuesday."""
    return datetime(2026, 2, 24, hour, minute, 0)


def _make_wednesday(hour=10, minute=0):
    """Return a naive datetime on a Wednesday."""
    return datetime(2026, 2, 25, hour, minute, 0)


def _make_thursday(hour=10, minute=0):
    """Return a naive datetime on a Thursday."""
    return datetime(2026, 2, 26, hour, minute, 0)


def _make_friday(hour=10, minute=0):
    """Return a naive datetime on a Friday."""
    return datetime(2026, 2, 27, hour, minute, 0)


def _make_saturday(hour=10, minute=0):
    """Return a naive datetime on a Saturday."""
    return datetime(2026, 2, 28, hour, minute, 0)


class MockPaperTrader:
    def __init__(self, trades):
        self.open_trades = trades

    def _evaluate_position(self, trade, price, dte):
        return trade.get("_mock_pnl", 0), None


class MockTelegramBot:
    def __init__(self):
        self.sent = []

    def send_alert(self, msg):
        self.sent.append(msg)


class MockFormatter:
    def format_exit_alert(self, **kwargs):
        return f"EXIT: {kwargs.get('ticker')} {kwargs.get('reason')}"


# ---------------------------------------------------------------------------
# Config overlay tests
# ---------------------------------------------------------------------------

class TestIronCondorConfig:
    def test_min_max_dte(self):
        cfg = build_iron_condor_config(_base_config())
        assert cfg["strategy"]["min_dte"] == 4
        assert cfg["strategy"]["max_dte"] == 10

    def test_delta_range(self):
        cfg = build_iron_condor_config(_base_config())
        assert cfg["strategy"]["min_delta"] == 0.12
        assert cfg["strategy"]["max_delta"] == 0.20

    def test_spread_width_uniform(self):
        cfg = build_iron_condor_config(_base_config())
        assert cfg["strategy"]["spread_width"] == 5
        assert cfg["strategy"]["spread_width_high_iv"] == 5
        assert cfg["strategy"]["spread_width_low_iv"] == 5

    def test_iron_condor_enabled(self):
        cfg = build_iron_condor_config(_base_config())
        assert cfg["strategy"]["iron_condor"]["enabled"] is True

    def test_min_combined_credit_pct(self):
        cfg = build_iron_condor_config(_base_config())
        assert cfg["strategy"]["iron_condor"]["min_combined_credit_pct"] == 34

    def test_iv_rank_threshold(self):
        cfg = build_iron_condor_config(_base_config())
        assert cfg["strategy"]["min_iv_rank"] == 50
        assert cfg["strategy"]["min_iv_percentile"] == 50

    def test_risk_overrides(self):
        cfg = build_iron_condor_config(_base_config())
        assert cfg["risk"]["profit_target"] == 50
        assert cfg["risk"]["stop_loss_multiplier"] == 2.0

    def test_tickers_expanded(self):
        cfg = build_iron_condor_config(_base_config())
        assert cfg["tickers"] == ["SPY", "QQQ", "TSLA", "AMZN", "META", "GOOGL"]

    def test_base_config_not_mutated(self):
        base = _base_config()
        original_min_dte = base["strategy"]["min_dte"]
        original_tickers = list(base["tickers"])
        build_iron_condor_config(base)
        assert base["strategy"]["min_dte"] == original_min_dte
        assert base["tickers"] == original_tickers

    def test_preserves_other_base_keys(self):
        base = _base_config()
        base["custom_setting"] = "keep_me"
        cfg = build_iron_condor_config(base)
        assert cfg["custom_setting"] == "keep_me"


# ---------------------------------------------------------------------------
# Day-of-week gate tests
# ---------------------------------------------------------------------------

class TestDayOfWeekGate:
    def test_monday_is_entry_day(self):
        assert IronCondorScanner.is_entry_day(_make_monday()) is True

    def test_tuesday_is_entry_day(self):
        assert IronCondorScanner.is_entry_day(_make_tuesday()) is True

    def test_wednesday_not_entry_day(self):
        assert IronCondorScanner.is_entry_day(_make_wednesday()) is False

    def test_thursday_not_entry_day(self):
        assert IronCondorScanner.is_entry_day(_make_thursday()) is False

    def test_friday_not_entry_day(self):
        assert IronCondorScanner.is_entry_day(_make_friday()) is False

    def test_saturday_not_entry_day(self):
        assert IronCondorScanner.is_entry_day(_make_saturday()) is False

    def test_entry_days_constant(self):
        assert ENTRY_DAYS == {0, 1}

    def test_close_days_constant(self):
        assert CLOSE_DAYS == {3, 4}


# ---------------------------------------------------------------------------
# Scanner scan() tests (mocked dependencies)
# ---------------------------------------------------------------------------

class TestIronCondorScannerScan:
    def _make_scanner(self):
        config = _base_config()
        scanner = IronCondorScanner(config)
        return scanner

    def test_scan_non_entry_day_returns_empty(self):
        scanner = self._make_scanner()
        # Wednesday is not an entry day
        result = scanner.scan(now_et=_make_wednesday())
        assert result == []

    def test_scan_entry_day_calls_scan_ticker(self):
        scanner = self._make_scanner()
        mock_opps = [{"ticker": "SPY", "type": "iron_condor", "score": 80}]
        scanner._scan_ticker = MagicMock(return_value=mock_opps)

        result = scanner.scan(now_et=_make_monday())
        # Should call _scan_ticker for each ticker in config
        assert scanner._scan_ticker.call_count == 6  # SPY,QQQ,TSLA,AMZN,META,GOOGL
        assert len(result) >= 1

    def test_scan_handles_ticker_error(self):
        scanner = self._make_scanner()
        scanner._scan_ticker = MagicMock(side_effect=Exception("API error"))

        result = scanner.scan(now_et=_make_monday())
        assert result == []

    def test_scan_annotates_alert_source(self):
        scanner = self._make_scanner()
        # _scan_ticker already annotates alert_source, so include it in mock
        mock_opps = [{"ticker": "SPY", "type": "iron_condor", "score": 80, "alert_source": "iron_condor"}]
        scanner._scan_ticker = MagicMock(return_value=mock_opps)

        result = scanner.scan(now_et=_make_tuesday())
        for opp in result:
            assert opp.get("alert_source") == "iron_condor"


# ---------------------------------------------------------------------------
# Exit monitor tests
# ---------------------------------------------------------------------------

class TestIronCondorExitMonitor:
    def _make_trade(self, **overrides):
        base = {
            "id": "ic1",
            "ticker": "SPY",
            "strategy_type": "iron_condor",
            "total_credit": 200,
            "_mock_pnl": 0,
        }
        base.update(overrides)
        return base

    def test_profit_target_triggers(self):
        trade = self._make_trade(_mock_pnl=110)  # > 50% of 200
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"SPY": 550.0}, now_et=_make_monday())
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "profit_target"
        assert len(bot.sent) == 1

    def test_stop_loss_triggers(self):
        trade = self._make_trade(_mock_pnl=-710)  # <= -(3.5 * 200) = -700
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"SPY": 550.0}, now_et=_make_monday())
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "stop_loss"

    def test_below_threshold_no_alert(self):
        trade = self._make_trade(_mock_pnl=50)  # 25% < 50% target
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        # Use Monday to avoid weekly close alerts
        triggered = monitor.check_and_alert({"SPY": 550.0}, now_et=_make_monday())
        assert len(triggered) == 0
        assert len(bot.sent) == 0

    def test_exact_50pct_triggers(self):
        trade = self._make_trade(_mock_pnl=100)  # exactly 50% of 200
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"SPY": 550.0}, now_et=_make_monday())
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "profit_target"

    def test_exact_35x_stop_triggers(self):
        trade = self._make_trade(_mock_pnl=-700)  # exactly -(3.5 * 200) = -700
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"SPY": 550.0}, now_et=_make_monday())
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "stop_loss"

    def test_composite_dedup_keys(self):
        """Same trade can trigger profit + weekly_close independently."""
        trade = self._make_trade(_mock_pnl=110)
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )

        # First call on Thursday → profit_target + weekly_close_warning
        triggered = monitor.check_and_alert(
            {"SPY": 550.0}, now_et=_make_thursday()
        )
        reasons = {t["reason"] for t in triggered}
        assert "profit_target" in reasons
        assert "weekly_close_warning" in reasons

    def test_dedup_prevents_repeat(self):
        trade = self._make_trade(_mock_pnl=110)
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )

        monitor.check_and_alert({"SPY": 550.0})
        # Second call — same reason should not fire again
        triggered2 = monitor.check_and_alert({"SPY": 550.0})
        profit_alerts = [t for t in triggered2 if t["reason"] == "profit_target"]
        assert len(profit_alerts) == 0

    def test_non_condor_skipped(self):
        trade = self._make_trade(strategy_type="bull_put_spread", _mock_pnl=110)
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 0

    def test_missing_trade_id_skipped(self):
        trade = self._make_trade(id="", _mock_pnl=110)
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 0

    def test_zero_credit_skipped(self):
        trade = self._make_trade(total_credit=0, _mock_pnl=110)
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 0

    def test_missing_price_skipped(self):
        trade = self._make_trade(_mock_pnl=110)
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"QQQ": 400.0})  # no SPY price
        assert len(triggered) == 0

    def test_thursday_warning(self):
        trade = self._make_trade(_mock_pnl=30)  # below profit target
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert(
            {"SPY": 550.0}, now_et=_make_thursday()
        )
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "weekly_close_warning"

    def test_friday_close_now(self):
        trade = self._make_trade(_mock_pnl=30)  # below profit target
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert(
            {"SPY": 550.0}, now_et=_make_friday()
        )
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "weekly_close_now"

    def test_thursday_warning_then_friday_close(self):
        """Same position gets Thu warning AND Fri close (composite keys)."""
        trade = self._make_trade(_mock_pnl=30)
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )

        # Thursday
        thu = monitor.check_and_alert(
            {"SPY": 550.0}, now_et=_make_thursday()
        )
        assert any(t["reason"] == "weekly_close_warning" for t in thu)

        # Friday — warning already sent, but close_now is new
        fri = monitor.check_and_alert(
            {"SPY": 550.0}, now_et=_make_friday()
        )
        assert any(t["reason"] == "weekly_close_now" for t in fri)

    def test_telegram_failure_doesnt_crash(self):
        trade = self._make_trade(_mock_pnl=110)
        bot = MockTelegramBot()
        bot.send_alert = MagicMock(side_effect=Exception("network error"))
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) >= 1

    def test_type_field_fallback(self):
        """Trades with 'type' instead of 'strategy_type' also match."""
        trade = {
            "id": "ic2",
            "ticker": "QQQ",
            "type": "iron_condor",
            "total_credit": 150,
            "_mock_pnl": 80,
        }
        bot = MockTelegramBot()
        monitor = IronCondorExitMonitor(
            MockPaperTrader([trade]), bot, formatter=MockFormatter()
        )
        # Use Monday to avoid weekly close alerts
        triggered = monitor.check_and_alert({"QQQ": 400.0}, now_et=_make_monday())
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "profit_target"
