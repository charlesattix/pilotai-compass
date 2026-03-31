"""Pytest-compatible tests for Phase 2: 0DTE Credit Spread Alerts.

Covers:
- Config overlay (zero_dte_config.py)
- Timing windows (zero_dte_scanner.py)
- Scanner scan() pipeline with mocks
- Exit monitor (zero_dte_exit_monitor.py)
- from_opportunity 0DTE-aware behavior (alert_schema.py)
- Backtest validator (zero_dte_backtest.py)
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from alerts.alert_schema import (
    Alert,
    AlertType,
    Confidence,
    Direction,
    TimeSensitivity,
)
from alerts.zero_dte_backtest import ZeroDTEBacktestValidator
from alerts.zero_dte_config import SPX_PROPERTIES, build_zero_dte_config
from alerts.zero_dte_exit_monitor import ZeroDTEExitMonitor
from alerts.zero_dte_scanner import ZeroDTEScanner

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
                "use_trend_filter": True,
                "use_rsi_filter": False,
                "use_support_resistance": False,
                "fast_ma": 20,
                "slow_ma": 50,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
            },
        },
        "risk": {
            "stop_loss_multiplier": 2.5,
            "min_credit_pct": 20,
            "account_size": 100000,
        },
        "tickers": ["SPY", "QQQ", "IWM"],
        "data": {},
        "backtest": {
            "starting_capital": 100000,
            "commission_per_contract": 0.65,
            "slippage": 0.05,
            "generate_reports": False,
            "report_dir": "/tmp/backtest_reports",
        },
    }


def _make_et_time(hour, minute):
    return datetime(2026, 2, 26, hour, minute, 0)


def _make_opp(**overrides):
    base = {
        "ticker": "SPY", "type": "bull_put_spread", "expiration": "2026-02-26",
        "short_strike": 540.0, "long_strike": 535.0, "credit": 1.20,
        "stop_loss": 2.40, "profit_target": 0.60, "score": 72,
    }
    base.update(overrides)
    return base


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

class TestZeroDTEConfig:
    def test_min_max_dte(self):
        cfg = build_zero_dte_config(_base_config())
        assert cfg["strategy"]["min_dte"] == 0
        assert cfg["strategy"]["max_dte"] == 1

    def test_delta_range(self):
        cfg = build_zero_dte_config(_base_config())
        assert cfg["strategy"]["min_delta"] == 0.08
        assert cfg["strategy"]["max_delta"] == 0.16

    def test_spread_width(self):
        cfg = build_zero_dte_config(_base_config())
        assert cfg["strategy"]["spread_width"] == 5
        assert cfg["strategy"]["spread_width_high_iv"] == 5
        assert cfg["strategy"]["spread_width_low_iv"] == 3

    def test_iron_condor_disabled(self):
        cfg = build_zero_dte_config(_base_config())
        assert cfg["strategy"]["iron_condor"]["enabled"] is False

    def test_risk_overrides(self):
        cfg = build_zero_dte_config(_base_config())
        assert cfg["risk"]["stop_loss_multiplier"] == 2.0
        assert cfg["risk"]["min_credit_pct"] == 10

    def test_tickers(self):
        cfg = build_zero_dte_config(_base_config())
        assert cfg["tickers"] == ["SPY", "SPX"]

    def test_base_config_not_mutated(self):
        base = _base_config()
        original_min_dte = base["strategy"]["min_dte"]
        original_tickers = list(base["tickers"])
        build_zero_dte_config(base)
        assert base["strategy"]["min_dte"] == original_min_dte
        assert base["tickers"] == original_tickers

    def test_spx_properties(self):
        for key in ("settlement", "exercise_style", "tax_treatment", "price_ticker"):
            assert key in SPX_PROPERTIES
        assert SPX_PROPERTIES["settlement"] == "cash"
        assert SPX_PROPERTIES["price_ticker"] == "^GSPC"

    def test_iv_thresholds_lowered(self):
        cfg = build_zero_dte_config(_base_config())
        assert cfg["strategy"]["min_iv_rank"] == 8
        assert cfg["strategy"]["min_iv_percentile"] == 8

    def test_preserves_other_base_keys(self):
        base = _base_config()
        base["custom_setting"] = "keep_me"
        cfg = build_zero_dte_config(base)
        assert cfg["custom_setting"] == "keep_me"


# ---------------------------------------------------------------------------
# Timing window tests
# ---------------------------------------------------------------------------

class TestTimingWindows:
    @pytest.mark.parametrize("hour,minute", [(9, 40), (11, 30), (14, 15)])
    def test_in_window(self, hour, minute):
        assert ZeroDTEScanner.is_in_entry_window(_make_et_time(hour, minute)) is True

    @pytest.mark.parametrize("hour,minute", [(8, 0), (10, 30), (13, 0), (15, 0)])
    def test_outside_window(self, hour, minute):
        assert ZeroDTEScanner.is_in_entry_window(_make_et_time(hour, minute)) is False

    def test_boundary_start_inclusive(self):
        assert ZeroDTEScanner.is_in_entry_window(_make_et_time(9, 35)) is True

    def test_boundary_end_exclusive(self):
        assert ZeroDTEScanner.is_in_entry_window(_make_et_time(10, 0)) is False

    def test_boundary_midday_start(self):
        assert ZeroDTEScanner.is_in_entry_window(_make_et_time(11, 0)) is True

    def test_boundary_midday_end(self):
        assert ZeroDTEScanner.is_in_entry_window(_make_et_time(12, 0)) is False

    def test_boundary_afternoon_start(self):
        assert ZeroDTEScanner.is_in_entry_window(_make_et_time(14, 0)) is True

    def test_boundary_afternoon_end(self):
        assert ZeroDTEScanner.is_in_entry_window(_make_et_time(14, 30)) is False


class TestWindowNames:
    def test_post_open(self):
        assert ZeroDTEScanner.active_window_name(_make_et_time(9, 40)) == "post_open"

    def test_midday(self):
        assert ZeroDTEScanner.active_window_name(_make_et_time(11, 30)) == "midday"

    def test_afternoon(self):
        assert ZeroDTEScanner.active_window_name(_make_et_time(14, 15)) == "afternoon"

    def test_none(self):
        assert ZeroDTEScanner.active_window_name(_make_et_time(13, 0)) == "none"


# ---------------------------------------------------------------------------
# Scanner scan() tests (mocked dependencies)
# ---------------------------------------------------------------------------

class TestZeroDTEScannerScan:
    def _make_scanner(self):
        config = _base_config()
        scanner = ZeroDTEScanner(config)
        return scanner

    def test_scan_outside_window_returns_empty(self):
        scanner = self._make_scanner()
        # 8 AM is outside all windows
        result = scanner.scan(now_et=_make_et_time(8, 0))
        assert result == []

    def test_scan_inside_window_calls_scan_ticker(self):
        scanner = self._make_scanner()
        mock_opps = [{"ticker": "SPY", "type": "bull_put_spread", "score": 80}]
        scanner._scan_ticker = MagicMock(return_value=mock_opps)

        result = scanner.scan(now_et=_make_et_time(9, 40))
        # Should call _scan_ticker for each ticker (SPY, SPX)
        assert scanner._scan_ticker.call_count == 2
        assert len(result) >= 1

    def test_scan_handles_ticker_error(self):
        scanner = self._make_scanner()
        scanner._scan_ticker = MagicMock(side_effect=Exception("API error"))

        # Should not raise, just return empty
        result = scanner.scan(now_et=_make_et_time(9, 40))
        assert result == []

    def test_spx_provider_check(self):
        scanner = self._make_scanner()
        # Without polygon/tradier, _has_spx_provider should be False
        assert scanner._has_spx_provider() is False


# ---------------------------------------------------------------------------
# Exit monitor tests
# ---------------------------------------------------------------------------

class TestExitMonitor:
    def test_profit_target_triggers(self):
        trade = {
            "id": "t1", "ticker": "SPY", "dte_at_entry": 0,
            "total_credit": 100, "_mock_pnl": 55,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "profit_target"
        assert len(bot.sent) == 1

    def test_stop_loss_triggers(self):
        trade = {
            "id": "t2", "ticker": "SPY", "dte_at_entry": 1,
            "total_credit": 100, "_mock_pnl": -210,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "stop_loss"

    def test_below_threshold_no_alert(self):
        trade = {
            "id": "t3", "ticker": "SPY", "dte_at_entry": 0,
            "total_credit": 100, "_mock_pnl": 30,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 0
        assert len(bot.sent) == 0

    def test_duplicate_suppression(self):
        trade = {
            "id": "t4", "ticker": "SPY", "dte_at_entry": 0,
            "total_credit": 100, "_mock_pnl": 60,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        monitor.check_and_alert({"SPY": 550.0})
        triggered2 = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered2) == 0
        assert len(bot.sent) == 1

    def test_non_zero_dte_skipped(self):
        trade = {
            "id": "t5", "ticker": "SPY", "dte_at_entry": 30,
            "total_credit": 100, "_mock_pnl": 60,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 0

    def test_missing_trade_id_skipped(self):
        trade = {
            "id": "", "ticker": "SPY", "dte_at_entry": 0,
            "total_credit": 100, "_mock_pnl": 60,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 0

    def test_zero_credit_skipped(self):
        trade = {
            "id": "t6", "ticker": "SPY", "dte_at_entry": 0,
            "total_credit": 0, "_mock_pnl": 60,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 0

    def test_missing_price_skipped(self):
        trade = {
            "id": "t7", "ticker": "SPY", "dte_at_entry": 0,
            "total_credit": 100, "_mock_pnl": 60,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        # No SPY in current_prices
        triggered = monitor.check_and_alert({"QQQ": 400.0})
        assert len(triggered) == 0

    def test_multiple_trades_independent(self):
        trades = [
            {"id": "t8", "ticker": "SPY", "dte_at_entry": 0,
             "total_credit": 100, "_mock_pnl": 55},
            {"id": "t9", "ticker": "SPX", "dte_at_entry": 0,
             "total_credit": 200, "_mock_pnl": -450},
        ]
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(
            MockPaperTrader(trades), bot, formatter=MockFormatter()
        )
        triggered = monitor.check_and_alert({"SPY": 550.0, "SPX": 5500.0})
        assert len(triggered) == 2
        reasons = {t["reason"] for t in triggered}
        assert reasons == {"profit_target", "stop_loss"}

    def test_telegram_failure_doesnt_crash(self):
        trade = {
            "id": "t10", "ticker": "SPY", "dte_at_entry": 0,
            "total_credit": 100, "_mock_pnl": 55,
        }
        bot = MockTelegramBot()
        bot.send_alert = MagicMock(side_effect=Exception("network error"))
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        # Should not raise
        triggered = monitor.check_and_alert({"SPY": 550.0})
        # Alert still counts as triggered (just send failed)
        assert len(triggered) == 1

    def test_exact_50pct_triggers(self):
        """Exactly 50% of credit should trigger profit target."""
        trade = {
            "id": "t11", "ticker": "SPY", "dte_at_entry": 0,
            "total_credit": 100, "_mock_pnl": 50,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "profit_target"

    def test_exact_2x_stop_triggers(self):
        """Exactly -2x credit should trigger stop loss."""
        trade = {
            "id": "t12", "ticker": "SPY", "dte_at_entry": 0,
            "total_credit": 100, "_mock_pnl": -200,
        }
        bot = MockTelegramBot()
        monitor = ZeroDTEExitMonitor(MockPaperTrader([trade]), bot, formatter=MockFormatter())
        triggered = monitor.check_and_alert({"SPY": 550.0})
        assert len(triggered) == 1
        assert triggered[0]["reason"] == "stop_loss"


# ---------------------------------------------------------------------------
# from_opportunity 0DTE tests
# ---------------------------------------------------------------------------

class TestFromOpportunityZeroDTE:
    def test_dte_0_immediate(self):
        alert = Alert.from_opportunity(_make_opp(dte=0))
        assert alert.time_sensitivity == TimeSensitivity.IMMEDIATE

    def test_dte_1_immediate(self):
        alert = Alert.from_opportunity(_make_opp(dte=1))
        assert alert.time_sensitivity == TimeSensitivity.IMMEDIATE

    def test_dte_30_today(self):
        alert = Alert.from_opportunity(_make_opp(dte=30))
        assert alert.time_sensitivity == TimeSensitivity.TODAY

    def test_spx_cash_settled_thesis(self):
        alert = Alert.from_opportunity(_make_opp(ticker="SPX", settlement="cash"))
        assert "cash-settled" in alert.thesis
        assert "Section 1256" in alert.thesis

    def test_dte_0_expires_1hr(self):
        alert = Alert.from_opportunity(_make_opp(dte=0))
        diff = alert.expires_at - datetime.now(timezone.utc)
        assert timedelta(minutes=59) <= diff <= timedelta(hours=1, seconds=5)

    def test_regular_dte_expires_4hr(self):
        alert = Alert.from_opportunity(_make_opp(dte=30))
        diff = alert.expires_at - datetime.now(timezone.utc)
        assert timedelta(hours=3, minutes=59) <= diff <= timedelta(hours=4, seconds=5)

    def test_zero_dte_source_custom_instructions(self):
        alert = Alert.from_opportunity(_make_opp(
            alert_source="zero_dte",
            management_instructions="Custom 0DTE instructions.",
        ))
        assert alert.management_instructions == "Custom 0DTE instructions."

    def test_high_score_high_confidence(self):
        alert = Alert.from_opportunity(_make_opp(score=85))
        assert alert.confidence == Confidence.HIGH

    def test_medium_score_medium_confidence(self):
        alert = Alert.from_opportunity(_make_opp(score=65))
        assert alert.confidence == Confidence.MEDIUM

    def test_low_score_speculative_confidence(self):
        alert = Alert.from_opportunity(_make_opp(score=40))
        assert alert.confidence == Confidence.SPECULATIVE

    def test_bull_put_spread_direction(self):
        alert = Alert.from_opportunity(_make_opp(type="bull_put_spread"))
        assert alert.direction == Direction.bullish
        assert alert.type == AlertType.credit_spread

    def test_bear_call_spread_direction(self):
        alert = Alert.from_opportunity(_make_opp(type="bear_call_spread"))
        assert alert.direction == Direction.bearish
        assert alert.type == AlertType.credit_spread

    def test_legs_count(self):
        alert = Alert.from_opportunity(_make_opp())
        assert len(alert.legs) == 2
        actions = {leg.action for leg in alert.legs}
        assert actions == {"sell", "buy"}

    def test_spx_management_instructions_propagated(self):
        opp = _make_opp(
            ticker="SPX",
            alert_source="zero_dte",
            management_instructions="Cash-settled, no assignment risk.",
        )
        alert = Alert.from_opportunity(opp)
        assert "Cash-settled" in alert.management_instructions


# ---------------------------------------------------------------------------
# Backtest validator tests
# ---------------------------------------------------------------------------

class TestZeroDTEBacktestValidator:
    def test_validation_passes_high_win_rate(self):
        results = {
            "total_trades": 100,
            "win_rate": 82.0,
            "profit_factor": 2.5,
            "total_pnl": 5000,
        }
        validation = ZeroDTEBacktestValidator._validate(results)
        assert validation["passed"] is True

    def test_validation_fails_low_win_rate(self):
        results = {
            "total_trades": 100,
            "win_rate": 65.0,
            "profit_factor": 2.0,
        }
        validation = ZeroDTEBacktestValidator._validate(results)
        assert validation["passed"] is False
        assert "Win rate" in validation["reason"]

    def test_validation_fails_low_profit_factor(self):
        results = {
            "total_trades": 100,
            "win_rate": 80.0,
            "profit_factor": 1.0,
        }
        validation = ZeroDTEBacktestValidator._validate(results)
        assert validation["passed"] is False
        assert "Profit factor" in validation["reason"]

    def test_validation_fails_insufficient_trades(self):
        results = {
            "total_trades": 5,
            "win_rate": 100.0,
            "profit_factor": 5.0,
        }
        validation = ZeroDTEBacktestValidator._validate(results)
        assert validation["passed"] is False
        assert "Insufficient" in validation["reason"]

    def test_validation_boundary_exactly_78_passes(self):
        results = {
            "total_trades": 50,
            "win_rate": 78.0,
            "profit_factor": 1.5,
        }
        validation = ZeroDTEBacktestValidator._validate(results)
        assert validation["passed"] is True

    def test_validation_boundary_exactly_10_trades_passes(self):
        results = {
            "total_trades": 10,
            "win_rate": 80.0,
            "profit_factor": 2.0,
        }
        validation = ZeroDTEBacktestValidator._validate(results)
        assert validation["passed"] is True

    def test_run_empty_results(self):
        validator = ZeroDTEBacktestValidator(_base_config())
        mock_backtester = MagicMock()
        mock_backtester.run_backtest.return_value = None

        with patch("backtest.Backtester", return_value=mock_backtester):
            results = validator.run(ticker="SPY", lookback_days=30)

        assert results["validation"]["passed"] is False
        assert "no results" in results["validation"]["reason"].lower()

    def test_run_with_passing_results(self):
        validator = ZeroDTEBacktestValidator(_base_config())
        mock_backtester = MagicMock()
        mock_backtester.run_backtest.return_value = {
            "total_trades": 50,
            "win_rate": 82.0,
            "profit_factor": 2.1,
            "total_pnl": 3000,
        }

        with patch("backtest.Backtester", return_value=mock_backtester):
            results = validator.run(ticker="SPY", lookback_days=30)

        assert results["validation"]["passed"] is True

    def test_config_uses_zero_dte_overlay(self):
        validator = ZeroDTEBacktestValidator(_base_config())
        # The internal config should have 0DTE overrides
        assert validator._zero_dte_config["strategy"]["min_dte"] == 0
        assert validator._zero_dte_config["strategy"]["max_dte"] == 1
        assert validator._zero_dte_config["tickers"] == ["SPY", "SPX"]
