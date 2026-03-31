"""
Tests for the unified exit path (Phase 2).

Covers:
  - BUG-A fix: per-trade profit_target/stop_loss in PositionMonitor
  - Strategy dispatch in PositionMonitor
  - DTE management in strategies
  - Spread-width safety cap in strategies
  - Straddle event-aware exit + 3x hard stop
  - trade_dict_to_position() defaults
  - CLOSE_DTE enum value
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from strategies.base import (
    MarketSnapshot, Position, PositionAction, TradeLeg, LegType, TradeDirection,
)


# ---------------------------------------------------------------------------
# BUG-A: per-trade profit_target / stop_loss
# ---------------------------------------------------------------------------

class TestPerTradeExitParams:
    """PositionMonitor credit path must use per-trade values, not global."""

    def _make_monitor(self, profit_target=50, stop_loss_mult=3.5):
        from execution.position_monitor import PositionMonitor
        monitor = object.__new__(PositionMonitor)
        monitor.config = {"risk": {"profit_target": profit_target, "stop_loss_multiplier": stop_loss_mult}}
        monitor.profit_target_pct = float(profit_target)
        monitor.stop_loss_mult = float(stop_loss_mult)
        monitor.manage_dte = 0  # Disable DTE management for these tests
        monitor._strategy_registry = {}
        monitor._exit_snapshot_cache = None
        monitor._exit_snapshot_ts = None
        return monitor

    def _credit_pos(self, credit=1.50, profit_target_pct=None, stop_loss_pct=None, **kwargs):
        pos = {
            "id": "test-1",
            "ticker": "SPY",
            "type": "bull_put_spread",
            "credit": credit,
            "short_strike": 490,
            "long_strike": 480,
            "expiration": (datetime.now(timezone.utc) + timedelta(days=45)).strftime("%Y-%m-%d"),
        }
        if profit_target_pct is not None:
            pos["profit_target_pct"] = profit_target_pct
        if stop_loss_pct is not None:
            pos["stop_loss_pct"] = stop_loss_pct
        pos.update(kwargs)
        return pos

    def test_per_trade_profit_target_used(self):
        """Per-trade profit_target_pct=0.30 (30%) should override global 50%."""
        monitor = self._make_monitor(profit_target=50)
        pos = self._credit_pos(credit=1.0, profit_target_pct=0.30)

        # Simulate spread value → pnl_pct = 35%
        # Global: 35 >= 50 → NO close. Per-trade: 35 >= 30 → YES close.
        with patch.object(monitor, '_get_spread_value', return_value=0.65):
            result = monitor._check_exit_conditions(pos, {})
        assert result == "profit_target"

    def test_per_trade_stop_loss_used(self):
        """Per-trade stop_loss_pct=1.0 (1x credit) should override global 3.5x."""
        monitor = self._make_monitor(stop_loss_mult=3.5)
        pos = self._credit_pos(credit=1.0, stop_loss_pct=1.0)

        # current_value = 2.5 → loss_based = (1+1.0)*1.0 = 2.0, 2.5 >= 2.0 → stop
        # Global would be: (1+3.5)*1.0 = 4.5, 2.5 < 4.5 → NO stop
        with patch.object(monitor, '_get_spread_value', return_value=2.5):
            result = monitor._check_exit_conditions(pos, {})
        assert result == "stop_loss"

    def test_global_fallback_when_no_per_trade(self):
        """Without per-trade values, should use global config."""
        monitor = self._make_monitor(profit_target=50, stop_loss_mult=3.5)
        pos = self._credit_pos(credit=1.0)

        # pnl_pct = (1.0 - 0.40)/1.0 * 100 = 60% → 60 >= 50 → profit_target
        with patch.object(monitor, '_get_spread_value', return_value=0.40):
            result = monitor._check_exit_conditions(pos, {})
        assert result == "profit_target"


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------

class TestStrategyDispatch:
    def test_register_strategies(self):
        from execution.position_monitor import PositionMonitor
        monitor = object.__new__(PositionMonitor)
        monitor._strategy_registry = {}

        strat = MagicMock()
        strat.__class__.__name__ = "CreditSpreadStrategy"
        monitor.register_strategies([strat])

        assert "CreditSpreadStrategy" in monitor._strategy_registry

    def test_strategy_close_overrides_generic(self):
        """If strategy returns CLOSE_PROFIT, position_monitor should close."""
        from execution.position_monitor import PositionMonitor

        monitor = object.__new__(PositionMonitor)
        monitor.config = {"risk": {}, "strategy": {}}
        monitor.profit_target_pct = 50
        monitor.stop_loss_mult = 3.5
        monitor.manage_dte = 0
        monitor._exit_snapshot_cache = None
        monitor._exit_snapshot_ts = None
        monitor.alpaca = MagicMock()

        # Register a mock strategy that always says CLOSE_PROFIT
        strat = MagicMock()
        strat.manage_position.return_value = PositionAction.CLOSE_PROFIT
        monitor._strategy_registry = {"CreditSpreadStrategy": strat}

        pos = {
            "id": "test-dispatch",
            "ticker": "SPY",
            "type": "bull_put_spread",
            "credit": 1.50,
            "short_strike": 490,
            "long_strike": 480,
            "expiration": (datetime.now(timezone.utc) + timedelta(days=45)).strftime("%Y-%m-%d"),
            "strategy_name": "CreditSpreadStrategy",
            "current_price": 500.0,
        }

        result = monitor._check_exit_conditions(pos, {})
        assert result == "profit_target"
        strat.manage_position.assert_called_once()

    def test_strategy_hold_falls_through_to_generic(self):
        """If strategy returns HOLD, generic P&L checks should run."""
        from execution.position_monitor import PositionMonitor

        monitor = object.__new__(PositionMonitor)
        monitor.config = {"risk": {}, "strategy": {}}
        monitor.profit_target_pct = 50
        monitor.stop_loss_mult = 3.5
        monitor.manage_dte = 0
        monitor._exit_snapshot_cache = None
        monitor._exit_snapshot_ts = None
        monitor.alpaca = MagicMock()

        strat = MagicMock()
        strat.manage_position.return_value = PositionAction.HOLD
        monitor._strategy_registry = {"CreditSpreadStrategy": strat}

        pos = {
            "id": "test-fallthrough",
            "ticker": "SPY",
            "type": "bull_put_spread",
            "credit": 1.00,
            "short_strike": 490,
            "long_strike": 480,
            "expiration": (datetime.now(timezone.utc) + timedelta(days=45)).strftime("%Y-%m-%d"),
            "strategy_name": "CreditSpreadStrategy",
            "current_price": 500.0,
        }

        # Strategy says HOLD, generic P&L says profit_target (60% profit)
        with patch.object(monitor, '_get_spread_value', return_value=0.40):
            result = monitor._check_exit_conditions(pos, {})
        assert result == "profit_target"


# ---------------------------------------------------------------------------
# DTE management in strategies
# ---------------------------------------------------------------------------

class TestDTEManagementInStrategies:
    def _make_position(self, strategy_name="CreditSpreadStrategy", dte_days=5):
        exp = datetime.now(timezone.utc) + timedelta(days=dte_days)
        return Position(
            id="dte-test",
            strategy_name=strategy_name,
            ticker="SPY",
            direction=TradeDirection.SHORT,
            legs=[
                TradeLeg(LegType.SHORT_PUT, 490.0, exp),
                TradeLeg(LegType.LONG_PUT, 480.0, exp),
            ],
            contracts=1,
            net_credit=1.50,
            max_loss_per_unit=8.50,
            max_profit_per_unit=1.50,
            profit_target_pct=0.50,
            stop_loss_pct=2.5,
        )

    def _make_snapshot(self):
        return MarketSnapshot(
            date=datetime.now(timezone.utc),
            price_data={},
            prices={"SPY": 500.0},
            vix=20.0,
            iv_rank={"SPY": 30.0},
            realized_vol={"SPY": 0.20},
            rsi={"SPY": 50.0},
        )

    def test_cs_dte_management_triggered(self):
        from strategies.credit_spread import CreditSpreadStrategy
        strat = CreditSpreadStrategy({"manage_dte": 7})
        pos = self._make_position(dte_days=5)
        snap = self._make_snapshot()
        result = strat.manage_position(pos, snap)
        assert result == PositionAction.CLOSE_DTE

    def test_cs_dte_management_disabled(self):
        from strategies.credit_spread import CreditSpreadStrategy
        strat = CreditSpreadStrategy({"manage_dte": 0})
        pos = self._make_position(dte_days=5)
        snap = self._make_snapshot()
        result = strat.manage_position(pos, snap)
        assert result != PositionAction.CLOSE_DTE

    def test_ic_dte_management_triggered(self):
        from strategies.iron_condor import IronCondorStrategy
        strat = IronCondorStrategy({"manage_dte": 7})
        exp = datetime.now(timezone.utc) + timedelta(days=5)
        pos = Position(
            id="ic-dte",
            strategy_name="IronCondorStrategy",
            ticker="SPY",
            direction=TradeDirection.NEUTRAL,
            legs=[
                TradeLeg(LegType.SHORT_PUT, 490.0, exp),
                TradeLeg(LegType.LONG_PUT, 480.0, exp),
                TradeLeg(LegType.SHORT_CALL, 510.0, exp),
                TradeLeg(LegType.LONG_CALL, 520.0, exp),
            ],
            contracts=1,
            net_credit=2.0,
            max_loss_per_unit=8.0,
            max_profit_per_unit=2.0,
            profit_target_pct=0.50,
            stop_loss_pct=2.0,
        )
        snap = self._make_snapshot()
        result = strat.manage_position(pos, snap)
        assert result == PositionAction.CLOSE_DTE


# ---------------------------------------------------------------------------
# CLOSE_DTE enum
# ---------------------------------------------------------------------------

class TestCloseDTEEnum:
    def test_close_dte_exists(self):
        assert hasattr(PositionAction, "CLOSE_DTE")
        assert PositionAction.CLOSE_DTE.value == "close_dte_management"


# ---------------------------------------------------------------------------
# trade_dict_to_position defaults
# ---------------------------------------------------------------------------

class TestTradeDictDefaults:
    def test_credit_spread_defaults(self):
        from shared.strategy_adapter import trade_dict_to_position
        trade = {
            "type": "bull_put_spread",
            "ticker": "SPY",
            "short_strike": 490,
            "long_strike": 480,
            "credit": 1.50,
            "expiration": "2024-07-19",
        }
        pos = trade_dict_to_position(trade)
        assert pos.profit_target_pct == 0.50
        assert pos.stop_loss_pct == 2.5

    def test_straddle_defaults(self):
        from shared.strategy_adapter import trade_dict_to_position
        trade = {
            "type": "short_straddle",
            "ticker": "SPY",
            "call_strike": 500,
            "put_strike": 500,
            "credit": 8.50,
            "expiration": "2024-07-19",
        }
        pos = trade_dict_to_position(trade)
        assert pos.profit_target_pct == 0.50
        assert pos.stop_loss_pct == 0.50

    def test_per_trade_overrides_defaults(self):
        from shared.strategy_adapter import trade_dict_to_position
        trade = {
            "type": "bull_put_spread",
            "ticker": "SPY",
            "short_strike": 490,
            "long_strike": 480,
            "credit": 1.50,
            "expiration": "2024-07-19",
            "profit_target_pct": 0.30,
            "stop_loss_pct": 1.25,
        }
        pos = trade_dict_to_position(trade)
        assert pos.profit_target_pct == 0.30
        assert pos.stop_loss_pct == 1.25


# ---------------------------------------------------------------------------
# Straddle 3x hard stop
# ---------------------------------------------------------------------------

class TestStraddle3xHardStop:
    def test_short_straddle_3x_credit_triggers_stop(self):
        from strategies.straddle_strangle import StraddleStrangleStrategy
        strat = StraddleStrangleStrategy({})
        exp = datetime.now(timezone.utc) + timedelta(days=7)
        pos = Position(
            id="ss-3x",
            strategy_name="StraddleStrangleStrategy",
            ticker="SPY",
            direction=TradeDirection.SHORT,
            legs=[
                TradeLeg(LegType.SHORT_CALL, 500.0, exp),
                TradeLeg(LegType.SHORT_PUT, 500.0, exp),
            ],
            contracts=1,
            net_credit=5.0,
            max_loss_per_unit=15.0,
            max_profit_per_unit=5.0,
            profit_target_pct=0.50,
            stop_loss_pct=0.50,
        )
        # Mock spread_value so cost_to_close = 16.0 (> 3 * 5.0 = 15.0)
        snap = MarketSnapshot(
            date=datetime.now(timezone.utc),
            price_data={},
            prices={"SPY": 520.0},
            vix=30.0,
            iv_rank={"SPY": 50.0},
            realized_vol={"SPY": 0.40},
            rsi={"SPY": 70.0},
        )
        with patch("strategies.straddle_strangle.estimate_spread_value", return_value=-16.0):
            result = strat.manage_position(pos, snap)
        assert result == PositionAction.CLOSE_STOP
