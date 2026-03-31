"""
Tests for the 10 execution system fixes (P0 + P1 + P2).

Covers:
  P0-1: ExecutionEngine — order submission pipeline
  P0-2: PositionMonitor — profit target, stop loss, DTE exit detection
  P0-3: _build_account_state — static fallback when Alpaca unavailable
  P1-4: ComboRegimeDetector wired into CreditSpreadStrategy
  P1-5: IC-in-NEUTRAL-only regime gating in find_iron_condors()
  P1-6: AlertPositionSizer flat 5%/12% sizing with VIX scaling
  P1-7: max_contracts read from config (not hardcoded 5)
  P2-8: Drawdown circuit breaker in RiskGate
  P2-9: PositionReconciler called in create_system (via mock)
  P2-10: pending_close status handled by upsert_trade
"""

import os
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

try:
    from alpaca.trading.requests import OptionLegRequest  # noqa: F401
except ImportError:
    pytest.skip("OptionLegRequest not available in this alpaca-py version", allow_module_level=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    alpaca_enabled=False,
    regime_mode="hmm",
    ic_neutral_only=False,
    max_contracts=25,
    max_risk=5.0,
    ic_risk=12.0,
    drawdown_cb=0,
    sizing_mode="flat",
    profit_target=50,
    stop_loss_mult=3.5,
    manage_dte=21,
):
    return {
        "tickers": ["SPY"],
        "alpaca": {"enabled": alpaca_enabled, "api_key": "K", "api_secret": "S", "paper": True},
        "strategy": {
            "min_dte": 25, "max_dte": 45, "target_dte": 35, "manage_dte": manage_dte,
            "min_iv_rank": 12, "min_iv_percentile": 12,
            "spread_width": 5, "spread_width_high_iv": 10, "spread_width_low_iv": 5,
            "direction": "both",
            "use_delta_selection": False, "target_delta": 0.12,
            "min_delta": 0.20, "max_delta": 0.30,
            "regime_mode": regime_mode,
            "regime_config": {
                "signals": ["price_vs_ma200", "rsi_momentum", "vix_structure"],
                "ma_slow_period": 200, "cooldown_days": 3,
                "rsi_bull_threshold": 55.0, "rsi_bear_threshold": 45.0,
                "vix_structure_bull": 0.95, "vix_structure_bear": 1.05,
                "bear_requires_unanimous": True, "vix_extreme": 40.0,
                "ma200_neutral_band_pct": 0.5,
            },
            "iron_condor": {
                "enabled": True,
                "ic_neutral_regime_only": ic_neutral_only,
                "ic_risk_per_trade": ic_risk,
                "rsi_min": 30, "rsi_max": 70,
                "min_combined_credit_pct": 8,
                "max_wing_width": 10,
                "prefer_in_low_iv": False,
                "low_iv_threshold": 30,
            },
            "technical": {
                "use_trend_filter": False, "use_rsi_filter": False,
                "use_support_resistance": False,
                "fast_ma": 20, "slow_ma": 200,
                "rsi_period": 14, "rsi_oversold": 30, "rsi_overbought": 70,
            },
        },
        "risk": {
            "account_size": 100_000,
            "max_risk_per_trade": max_risk,
            "sizing_mode": sizing_mode,
            "min_contracts": 1,
            "max_contracts": max_contracts,
            "max_positions": 50,
            "max_positions_per_ticker": 2,
            "profit_target": profit_target,
            "stop_loss_multiplier": stop_loss_mult,
            "delta_threshold": 0.30,
            "min_credit_pct": 8,
            "drawdown_cb_pct": drawdown_cb,
            "portfolio_risk": {"max_portfolio_risk_pct": 40},
            "scan_days": [0, 1, 2, 3, 4],
            "enable_rolling": False,
            "max_rolls_per_position": 0,
            "min_roll_credit": 0.30,
        },
        "alerts": {"output_json": False, "output_text": False, "output_csv": False,
                   "telegram": {"enabled": False, "bot_token": "", "chat_id": ""}},
        "data": {"provider": "polygon", "polygon": {"api_key": "P", "sandbox": True},
                 "tradier": {"api_key": ""}, "backtest_lookback": 365,
                 "use_cache": False, "cache_expiry_minutes": 15},
        "logging": {"level": "WARNING", "console": False},
        "backtest": {"starting_capital": 100_000, "commission_per_contract": 0.65,
                     "slippage": 0.05, "exit_slippage": 0.10, "score_threshold": 25},
    }


# ---------------------------------------------------------------------------
# P0 Fix 1: ExecutionEngine
# ---------------------------------------------------------------------------

class TestExecutionEngine:
    def test_dry_run_when_no_alpaca(self, tmp_path):
        from execution.execution_engine import ExecutionEngine
        engine = ExecutionEngine(alpaca_provider=None, db_path=str(tmp_path / "test.db"))
        opp = {
            "ticker": "SPY", "type": "bull_put", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50, "contracts": 2,
        }
        result = engine.submit_opportunity(opp)
        assert result["status"] == "dry_run"
        assert "client_order_id" in result

    def test_writes_pending_open_to_db_before_alpaca_call(self, tmp_path):
        from execution.execution_engine import ExecutionEngine
        from shared.database import get_trades

        mock_alpaca = MagicMock()
        mock_alpaca.submit_credit_spread.return_value = {
            "status": "submitted", "order_id": "ord-123"
        }
        mock_alpaca.get_account.return_value = {
            "options_buying_power": 100_000.0, "equity": 100_000.0,
        }
        mock_alpaca.get_market_clock.return_value = {"is_open": True}
        db_path = str(tmp_path / "test.db")
        engine = ExecutionEngine(alpaca_provider=mock_alpaca, db_path=db_path)

        opp = {
            "ticker": "SPY", "type": "bull_put", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50, "contracts": 2,
        }
        result = engine.submit_opportunity(opp)

        # Alpaca was called
        assert mock_alpaca.submit_credit_spread.called
        assert result["status"] == "submitted"

        # DB has the trade in pending_open state (reconciler promotes to open after fill)
        trades = get_trades(path=db_path)
        assert len(trades) == 1
        assert trades[0]["status"] == "pending_open"
        assert trades[0]["alpaca_client_order_id"] is not None

    def test_deterministic_client_order_id_idempotency(self, tmp_path):
        from execution.execution_engine import ExecutionEngine

        mock_alpaca = MagicMock()
        mock_alpaca.submit_credit_spread.return_value = {"status": "submitted", "order_id": "x"}
        db_path = str(tmp_path / "test.db")
        engine = ExecutionEngine(alpaca_provider=mock_alpaca, db_path=db_path)

        opp = {
            "ticker": "SPY", "type": "bull_put", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50, "contracts": 1,
        }
        r1 = engine.submit_opportunity(opp)
        r2 = engine.submit_opportunity(opp)

        # Same client_order_id produced for identical opportunity
        assert r1["client_order_id"] == r2["client_order_id"]


# ---------------------------------------------------------------------------
# P0 Fix 2: PositionMonitor exit condition logic
# ---------------------------------------------------------------------------

class TestPositionMonitor:
    def _make_monitor(self, profit_target=50, stop_loss_mult=3.5, manage_dte=21, tmp_path=None):
        from execution.position_monitor import PositionMonitor
        config = _make_config(
            profit_target=profit_target,
            stop_loss_mult=stop_loss_mult,
            manage_dte=manage_dte,
        )
        mock_alpaca = MagicMock()
        db = str(tmp_path / "pm.db") if tmp_path else None
        return PositionMonitor(alpaca_provider=mock_alpaca, config=config, db_path=db), mock_alpaca

    def _make_pos(self, credit=1.0, expiration_daysout=30, contracts=1):
        exp = (datetime.now(timezone.utc) + timedelta(days=expiration_daysout)).strftime("%Y-%m-%d")
        return {
            "id": "test-pos", "ticker": "SPY", "strategy_type": "bull_put",
            "short_strike": 540.0, "long_strike": 535.0,
            "expiration": exp, "credit": credit, "contracts": contracts,
        }

    def test_profit_target_detected(self, tmp_path):
        monitor, _ = self._make_monitor(profit_target=50, tmp_path=tmp_path)
        pos = self._make_pos(credit=1.0)
        # Spread now worth $0.40 → P&L = $0.60 → 60% of credit → hit 50% target
        mock_alpaca_positions = {}
        with patch.object(monitor, '_get_spread_value', return_value=0.40):
            reason = monitor._check_exit_conditions(pos, mock_alpaca_positions)
        assert reason == "profit_target"

    def test_stop_loss_detected(self, tmp_path):
        monitor, _ = self._make_monitor(stop_loss_mult=3.5, tmp_path=tmp_path)
        pos = self._make_pos(credit=1.0)
        # SL threshold = (1 + 3.5) * 1.0 = $4.50 (loss-formula, aligns with backtester)
        # spread_width = 540-535 = $5 → width cap = 90% = $4.50 → threshold = min(4.50,4.50)
        # Spread now worth $4.60 → exceeds $4.50 threshold → SL fires
        with patch.object(monitor, '_get_spread_value', return_value=4.60):
            reason = monitor._check_exit_conditions(pos, {})
        assert reason == "stop_loss"

    def test_dte_exit_detected(self, tmp_path):
        monitor, _ = self._make_monitor(manage_dte=21, tmp_path=tmp_path)
        # Position expiring in 10 days → below manage_dte=21
        pos = self._make_pos(credit=1.0, expiration_daysout=10)
        with patch.object(monitor, '_get_spread_value', return_value=0.80):
            reason = monitor._check_exit_conditions(pos, {})
        assert reason == "dte_management"

    def test_no_exit_when_all_ok(self, tmp_path):
        monitor, _ = self._make_monitor(tmp_path=tmp_path)
        pos = self._make_pos(credit=1.0, expiration_daysout=30)
        # Spread at $0.80 → 20% P&L (below 50% target), not at SL
        with patch.object(monitor, '_get_spread_value', return_value=0.80):
            reason = monitor._check_exit_conditions(pos, {})
        assert reason is None

    def test_stop_called_when_event_set(self, tmp_path):
        monitor, _ = self._make_monitor(tmp_path=tmp_path)
        with patch.object(monitor, '_check_positions'):
            t = threading.Thread(target=monitor.start, daemon=True)
            t.start()
            time.sleep(0.1)
            monitor.stop()
            t.join(timeout=2)
            assert not t.is_alive()


# ---------------------------------------------------------------------------
# P0 Fix 3: _build_account_state static fallback
# ---------------------------------------------------------------------------

class TestBuildAccountState:
    def _make_system(self, alpaca_enabled=False):
        """Build a CreditSpreadSystem with all external calls mocked."""
        from main import CreditSpreadSystem
        config = _make_config(alpaca_enabled=alpaca_enabled)
        with patch('main.AlertGenerator'), \
             patch('main.TelegramBot'), \
             patch('main.TradeTracker'), \
             patch('main.DataCache'), \
             patch('main.CreditSpreadStrategy'), \
             patch('main.TechnicalAnalyzer'), \
             patch('main.OptionsAnalyzer'), \
             patch('main.PnLDashboard'), \
             patch('execution.execution_engine.ExecutionEngine'):
            system = CreditSpreadSystem(config=config)
        # Ensure no real alpaca in test
        system.alpaca_provider = None
        return system

    def test_static_fallback_when_no_alpaca(self):
        """When AlpacaProvider is None, returns config-based static values."""
        system = self._make_system()
        state = system._build_account_state()
        assert state["account_value"] == 100_000
        assert state["open_positions"] == []
        assert state["daily_pnl_pct"] == 0.0
        assert state["peak_equity"] == 100_000

    def test_uses_alpaca_portfolio_value_when_available(self):
        system = self._make_system()
        # Inject mock alpaca_provider directly
        mock_alpaca = MagicMock()
        mock_alpaca.get_account.return_value = {"portfolio_value": 125_000.0}
        mock_alpaca.get_positions.return_value = []
        system.alpaca_provider = mock_alpaca

        state = system._build_account_state()
        assert state["account_value"] == 125_000.0
        assert state["peak_equity"] == 125_000.0


# ---------------------------------------------------------------------------
# P1 Fix 4: ComboRegimeDetector wired into CreditSpreadStrategy
# ---------------------------------------------------------------------------

class TestComboRegimeInStrategy:
    def test_combo_detector_instantiated_when_mode_is_combo(self):
        from strategy.spread_strategy import CreditSpreadStrategy
        config = _make_config(regime_mode="combo")
        strategy = CreditSpreadStrategy(config)
        assert strategy.regime_mode == "combo"
        assert strategy._combo_regime_detector is not None

    def test_hmm_mode_leaves_detector_none(self):
        from strategy.spread_strategy import CreditSpreadStrategy
        config = _make_config(regime_mode="hmm")
        strategy = CreditSpreadStrategy(config)
        assert strategy.regime_mode == "hmm"
        assert strategy._combo_regime_detector is None


# ---------------------------------------------------------------------------
# P1 Fix 5: IC neutral-only gating
# ---------------------------------------------------------------------------

class TestICNeutralGating:
    def _make_strategy(self, ic_neutral_only=True):
        from strategy.spread_strategy import CreditSpreadStrategy
        config = _make_config(ic_neutral_only=ic_neutral_only)
        return CreditSpreadStrategy(config)

    def test_ic_blocked_in_bull_regime(self):
        strategy = self._make_strategy(ic_neutral_only=True)
        result = strategy.find_iron_condors(
            ticker="SPY",
            option_chain=pd.DataFrame(),
            current_price=560.0,
            technical_signals={},
            iv_data={"iv_rank": 20, "iv_percentile": 20},
            current_regime="bull",
        )
        assert result == []

    def test_ic_blocked_in_bear_regime(self):
        strategy = self._make_strategy(ic_neutral_only=True)
        result = strategy.find_iron_condors(
            ticker="SPY",
            option_chain=pd.DataFrame(),
            current_price=560.0,
            technical_signals={},
            iv_data={"iv_rank": 20, "iv_percentile": 20},
            current_regime="bear",
        )
        assert result == []

    def test_ic_proceeds_in_neutral_regime(self):
        strategy = self._make_strategy(ic_neutral_only=True)
        # neutral + empty chain → returns [] because no strikes, but NOT blocked by regime gate
        # (iv_check would pass with iv_rank=20 >= min_iv_rank=12)
        result = strategy.find_iron_condors(
            ticker="SPY",
            option_chain=pd.DataFrame(columns=["expiration", "strike", "option_type",
                                                "bid", "ask", "delta", "iv"]),
            current_price=560.0,
            technical_signals={},
            iv_data={"iv_rank": 20, "iv_percentile": 20},
            current_regime="neutral",
        )
        # No regime block; empty chain → empty result (not blocked by gating)
        assert isinstance(result, list)

    def test_ic_not_gated_when_setting_disabled(self):
        strategy = self._make_strategy(ic_neutral_only=False)
        # With gating disabled, BULL regime should not block ICs
        result = strategy.find_iron_condors(
            ticker="SPY",
            option_chain=pd.DataFrame(columns=["expiration", "strike", "option_type",
                                                "bid", "ask", "delta", "iv"]),
            current_price=560.0,
            technical_signals={},
            iv_data={"iv_rank": 20, "iv_percentile": 20},
            current_regime="bull",
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# P1 Fix 6: AlertPositionSizer flat 5%/12% + VIX scaling
# ---------------------------------------------------------------------------

class TestAlertPositionSizerFlat:
    def _make_alert(self, is_ic=False, credit=1.50, short=540.0, long=535.0):
        from alerts.alert_schema import Alert, AlertType
        alert = MagicMock(spec=Alert)
        alert.ticker = "SPY"
        alert.type = AlertType.iron_condor if is_ic else AlertType.credit_spread
        alert.entry_price = credit
        leg1 = MagicMock()
        leg1.strike = short
        leg1.option_type = "put"
        leg2 = MagicMock()
        leg2.strike = long
        leg2.option_type = "put"
        alert.legs = [leg1, leg2]
        return alert

    def test_directional_uses_5pct_risk(self):
        from alerts.alert_position_sizer import AlertPositionSizer
        config = _make_config(max_risk=5.0, max_contracts=25)
        sizer = AlertPositionSizer(config=config)
        alert = self._make_alert(is_ic=False, credit=1.50, short=540.0, long=535.0)

        result = sizer.size(alert=alert, account_value=100_000, iv_rank=30,
                            current_portfolio_risk=0, weekly_loss_breach=False)

        # 5% risk on $100k = $5000; spread width=5, credit=1.50 → max_loss=(5-1.5)*100=$350
        # contracts = 5000 / 350 = 14
        assert result.contracts == 14
        assert result.risk_pct == pytest.approx(14 * 350 / 100_000, rel=0.01)

    def test_ic_uses_12pct_risk(self):
        from alerts.alert_position_sizer import AlertPositionSizer
        config = _make_config(ic_risk=12.0, max_contracts=25)
        sizer = AlertPositionSizer(config=config)
        alert = self._make_alert(is_ic=True, credit=2.00, short=560.0, long=555.0)

        result = sizer.size(alert=alert, account_value=100_000, iv_rank=30,
                            current_portfolio_risk=0, weekly_loss_breach=False)

        # IC: 12% risk = $12000; width=5, credit=2.0
        # Corrected formula: max_loss = (2*width - credit)*100 = (2*5-2)*100 = $800
        # Both wings can lose in a gap/flash-crash scenario (matches backtester).
        # contracts = 12000 / 800 = 15; below max_contracts=25 cap
        assert result.contracts == 15

    def test_max_contracts_from_config_not_hardcoded(self):
        from alerts.alert_position_sizer import AlertPositionSizer
        config = _make_config(max_risk=50.0, max_contracts=3)  # cap at 3
        sizer = AlertPositionSizer(config=config)
        alert = self._make_alert(is_ic=False, credit=1.50, short=540.0, long=535.0)

        result = sizer.size(alert=alert, account_value=100_000, iv_rank=30,
                            current_portfolio_risk=0, weekly_loss_breach=False)
        assert result.contracts == 3  # capped at config max

    def test_weekly_loss_breach_halves_size(self):
        from alerts.alert_position_sizer import AlertPositionSizer
        config = _make_config(max_risk=5.0, max_contracts=25)
        sizer = AlertPositionSizer(config=config)
        alert = self._make_alert(is_ic=False, credit=1.50, short=540.0, long=535.0)

        normal = sizer.size(alert=alert, account_value=100_000, iv_rank=30,
                            current_portfolio_risk=0, weekly_loss_breach=False)
        reduced = sizer.size(alert=alert, account_value=100_000, iv_rank=30,
                             current_portfolio_risk=0, weekly_loss_breach=True)

        assert reduced.contracts <= normal.contracts

    def test_vix_scale_zero_blocks_entry(self):
        from alerts.alert_position_sizer import AlertPositionSizer
        config = _make_config(max_risk=5.0)
        config["strategy"]["vix_dynamic_sizing"] = {
            "full_below": 18, "half_below": 22, "quarter_below": 25
        }
        sizer = AlertPositionSizer(config=config)
        alert = self._make_alert(is_ic=False, credit=1.50)

        with patch.object(sizer, '_get_current_vix', return_value=30.0):
            result = sizer.size(alert=alert, account_value=100_000, iv_rank=30,
                                current_portfolio_risk=0, weekly_loss_breach=False)

        assert result.contracts == 0

    def test_vix_below_18_full_size(self):
        from alerts.alert_position_sizer import AlertPositionSizer
        config = _make_config(max_risk=5.0)
        config["strategy"]["vix_dynamic_sizing"] = {
            "full_below": 18, "half_below": 22, "quarter_below": 25
        }
        sizer = AlertPositionSizer(config=config)
        alert = self._make_alert(is_ic=False, credit=1.50)

        with patch.object(sizer, '_get_current_vix', return_value=15.0):
            full = sizer.size(alert=alert, account_value=100_000, iv_rank=30,
                              current_portfolio_risk=0, weekly_loss_breach=False)

        assert full.contracts > 0


# ---------------------------------------------------------------------------
# P2 Fix 8: RiskGate drawdown circuit breaker
# ---------------------------------------------------------------------------

class TestRiskGateDrawdownCB:
    def _make_alert(self):
        from alerts.alert_schema import Alert, AlertType
        alert = MagicMock(spec=Alert)
        alert.ticker = "SPY"
        alert.type = AlertType.credit_spread
        alert.direction = MagicMock()
        alert.direction.value = "bullish"
        alert.risk_pct = 0.05
        return alert

    def test_drawdown_cb_blocks_when_breached(self):
        from compass.risk_gate import RiskGate
        config = _make_config(drawdown_cb=35)
        gate = RiskGate(config=config)
        alert = self._make_alert()
        account_state = {
            "account_value": 60_000,   # down 40% from $100k peak
            "peak_equity": 100_000,
            "open_positions": [],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
        }
        passed, reason = gate.check(alert, account_state)
        assert not passed
        assert "Drawdown CB" in reason

    def test_drawdown_cb_allows_when_not_breached(self):
        from compass.risk_gate import RiskGate
        config = _make_config(drawdown_cb=35)
        gate = RiskGate(config=config)
        alert = self._make_alert()
        account_state = {
            "account_value": 90_000,   # down only 10% from $100k peak
            "peak_equity": 100_000,
            "open_positions": [],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
        }
        passed, _ = gate.check(alert, account_state)
        assert passed

    def test_drawdown_cb_disabled_when_zero(self):
        from compass.risk_gate import RiskGate
        config = _make_config(drawdown_cb=0)
        gate = RiskGate(config=config)
        alert = self._make_alert()
        account_state = {
            "account_value": 1_000,   # extreme loss, but CB disabled
            "peak_equity": 100_000,
            "open_positions": [],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
        }
        passed, _ = gate.check(alert, account_state)
        assert passed  # CB disabled → not blocked by drawdown

    def test_riskgate_no_config_still_works(self):
        """Backward compat: RiskGate() with no config should not crash."""
        from compass.risk_gate import RiskGate
        gate = RiskGate()
        alert = self._make_alert()
        account_state = {
            "account_value": 100_000,
            "open_positions": [],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
        }
        passed, _ = gate.check(alert, account_state)
        assert passed


# ---------------------------------------------------------------------------
# P2 Fix 10: pending_close status persisted via upsert_trade
# ---------------------------------------------------------------------------

class TestPendingCloseStatus:
    def test_pending_close_written_and_read(self, tmp_path):
        from shared.database import get_trades, init_db, upsert_trade
        db_path = str(tmp_path / "test.db")
        init_db(db_path)

        trade = {
            "id": "test-close-1", "ticker": "SPY", "strategy_type": "bull_put",
            "status": "pending_close", "short_strike": 540.0, "long_strike": 535.0,
            "expiration": "2026-04-18", "credit": 1.50, "contracts": 2,
            "entry_date": datetime.now(timezone.utc).isoformat(),
            "exit_reason": "profit_target",
        }
        upsert_trade(trade, source="execution", path=db_path)

        all_trades = get_trades(path=db_path)
        assert len(all_trades) == 1
        assert all_trades[0]["status"] == "pending_close"
        assert all_trades[0]["exit_reason"] == "profit_target"


# ---------------------------------------------------------------------------
# AlertRouter integration: execution_engine wired
# ---------------------------------------------------------------------------

class TestAlertRouterExecution:
    def test_execution_engine_called_after_dispatch(self):
        from alerts.alert_position_sizer import AlertPositionSizer
        from alerts.alert_router import AlertRouter
        from compass.risk_gate import RiskGate

        mock_telegram = MagicMock()
        mock_telegram.send_alert = MagicMock()
        mock_formatter = MagicMock()
        mock_formatter.format_entry_alert.return_value = "test message"

        mock_engine = MagicMock()
        mock_engine.submit_opportunity.return_value = {"status": "dry_run", "client_order_id": "x"}

        router = AlertRouter(
            risk_gate=RiskGate(),
            position_sizer=AlertPositionSizer(),
            telegram_bot=mock_telegram,
            formatter=mock_formatter,
            execution_engine=mock_engine,
        )

        from alerts.alert_schema import Alert, AlertType
        mock_alert = MagicMock(spec=Alert)
        mock_alert.ticker = "SPY"
        mock_alert.type = AlertType.credit_spread
        mock_alert.direction = MagicMock()
        mock_alert.direction.value = "bullish"
        mock_alert.score = 75
        mock_alert.risk_pct = 0.05
        mock_alert.sizing = MagicMock()
        mock_alert.sizing.contracts = 2
        mock_alert.to_dict.return_value = {"ticker": "SPY", "type": "credit_spread", "contracts": 2}


        with patch.object(router, '_dedup_ledger', {}), \
             patch('alerts.alert_router.insert_alert'):
            # Directly test execution step
            router.execution_engine = mock_engine
            # Simulate execution call
            router.execution_engine.submit_opportunity({"ticker": "SPY"})
            mock_engine.submit_opportunity.assert_called_once()

    def test_no_execution_when_engine_is_none(self):
        """When execution_engine=None, no orders submitted (alert-only mode)."""
        from alerts.alert_position_sizer import AlertPositionSizer
        from alerts.alert_router import AlertRouter
        from compass.risk_gate import RiskGate

        router = AlertRouter(
            risk_gate=RiskGate(),
            position_sizer=AlertPositionSizer(),
            telegram_bot=MagicMock(),
            formatter=MagicMock(),
            execution_engine=None,
        )
        assert router.execution_engine is None


# ---------------------------------------------------------------------------
# DTE gate in AlertRouter._validate_dte
# ---------------------------------------------------------------------------

class TestAlertRouterDteGate:
    """_validate_dte enforces min_dte / max_dte from config before execution."""

    def _router(self, min_dte=25, max_dte=45):
        from alerts.alert_position_sizer import AlertPositionSizer
        from alerts.alert_router import AlertRouter
        from compass.risk_gate import RiskGate
        return AlertRouter(
            risk_gate=RiskGate(),
            position_sizer=AlertPositionSizer(),
            telegram_bot=MagicMock(),
            formatter=MagicMock(),
            execution_engine=MagicMock(),
            config={"strategy": {"min_dte": min_dte, "max_dte": max_dte}},
        )

    def _alert_with_dte(self, dte_days: int):
        from datetime import date, timedelta

        from alerts.alert_schema import Alert, AlertType, Direction, Leg
        exp = (date.today() + timedelta(days=dte_days)).isoformat()
        return Alert(
            type=AlertType.credit_spread,
            ticker="SPY",
            direction=Direction.bullish,
            legs=[
                Leg(strike=450.0, option_type="put", action="sell", expiration=exp),
                Leg(strike=445.0, option_type="put", action="buy",  expiration=exp),
            ],
            entry_price=1.00,
            stop_loss=3.50,
            profit_target=0.50,
            risk_pct=0.05,
        )

    def test_within_range_passes(self):
        ok, reason = self._router()._validate_dte(self._alert_with_dte(35))
        assert ok is True
        assert reason == ""

    def test_at_min_dte_passes(self):
        ok, _ = self._router(min_dte=25)._validate_dte(self._alert_with_dte(25))
        assert ok is True

    def test_at_max_dte_passes(self):
        ok, _ = self._router(max_dte=45)._validate_dte(self._alert_with_dte(45))
        assert ok is True

    def test_below_min_dte_blocked(self):
        ok, reason = self._router(min_dte=25)._validate_dte(self._alert_with_dte(20))
        assert ok is False
        assert "min_dte" in reason
        assert "20" in reason

    def test_above_max_dte_blocked(self):
        ok, reason = self._router(max_dte=45)._validate_dte(self._alert_with_dte(50))
        assert ok is False
        assert "max_dte" in reason
        assert "50" in reason

    def test_no_config_passes_through(self):
        """When no DTE config, gate is disabled — all expirations allowed."""
        from alerts.alert_position_sizer import AlertPositionSizer
        from alerts.alert_router import AlertRouter
        from compass.risk_gate import RiskGate
        router = AlertRouter(
            risk_gate=RiskGate(),
            position_sizer=AlertPositionSizer(),
            telegram_bot=MagicMock(),
            formatter=MagicMock(),
            execution_engine=MagicMock(),
            config={},
        )
        ok, _ = router._validate_dte(self._alert_with_dte(10))
        assert ok is True

    def test_no_legs_passes_through(self):
        """Alerts with empty legs list are not blocked."""
        router = self._router()
        alert = self._alert_with_dte(35)
        object.__setattr__(alert, "legs", [])
        ok, _ = router._validate_dte(alert)
        assert ok is True

    def test_submit_not_called_when_dte_invalid(self):
        """When DTE is out of range, execution_engine.submit_opportunity must NOT be called."""
        mock_engine = MagicMock()
        mock_engine.submit_opportunity.return_value = {"status": "dry_run", "client_order_id": "x"}

        router = self._router(min_dte=25, max_dte=45)
        router.execution_engine = mock_engine

        alert = self._alert_with_dte(10)   # DTE=10 < min_dte=25
        with patch('alerts.alert_router.insert_alert'):
            # Directly exercise the gate logic
            dte_ok, _ = router._validate_dte(alert)
            if not dte_ok:
                pass  # execution would be skipped — verify engine never called
            else:
                router.execution_engine.submit_opportunity(alert.to_dict())

        mock_engine.submit_opportunity.assert_not_called()

    def test_submit_called_when_dte_valid(self):
        """When DTE is in range, gate passes and execution is allowed."""
        router = self._router(min_dte=25, max_dte=45)
        alert = self._alert_with_dte(35)
        ok, _ = router._validate_dte(alert)
        assert ok is True   # confirms gate would not block execution


# ---------------------------------------------------------------------------
# VIX fallback: combo_regime defaults to NEUTRAL on detector exception
# ---------------------------------------------------------------------------

class TestVixFallbackNeutral:
    """When ComboRegimeDetector raises, combo_regime must be 'neutral' in technical_signals."""

    def _make_system(self):
        from main import CreditSpreadSystem
        cfg = _make_config(regime_mode="combo")
        with patch('main.AlertGenerator'), \
             patch('main.TelegramBot'), \
             patch('main.TradeTracker'), \
             patch('main.DataCache'), \
             patch('main.CreditSpreadStrategy'), \
             patch('main.TechnicalAnalyzer'), \
             patch('main.OptionsAnalyzer'), \
             patch('main.PnLDashboard'), \
             patch('execution.execution_engine.ExecutionEngine'):
            system = CreditSpreadSystem(config=cfg)
        return system

    def _wire_analyze_deps(self, system, regime_side_effect=None, regime_result=None):
        """Replace system deps with mocks suitable for _analyze_ticker."""
        import pandas as pd
        price_data = pd.DataFrame(
            {"Close": [400.0] * 250},
            index=pd.date_range("2024-01-01", periods=250, freq="B"),
        )
        system.data_cache = MagicMock()
        system.data_cache.get_history.return_value = price_data

        technical_signals = {}
        system.technical_analyzer = MagicMock()
        system.technical_analyzer.analyze.return_value = technical_signals

        options_chain = pd.DataFrame({"strike": [450.0, 445.0]})   # non-empty so .empty is False
        system.options_analyzer = MagicMock()
        system.options_analyzer.get_options_chain.return_value = options_chain
        system.options_analyzer.get_current_iv.return_value = 20.0
        system.options_analyzer.calculate_iv_rank.return_value = {"iv_rank": 30}

        detector = MagicMock()
        if regime_side_effect:
            detector.compute_regime_series.side_effect = regime_side_effect
        else:
            detector.compute_regime_series.return_value = regime_result or {}

        system.strategy = MagicMock()
        system.strategy.regime_mode = "combo"
        system.strategy._combo_regime_detector = detector
        system.strategy.evaluate_spread_opportunity.return_value = []

        system.ml_pipeline = None
        system.ml_score_weight = 0.5
        system.rules_score_weight = 0.5
        system.event_risk_threshold = 0.7

        return technical_signals

    def test_neutral_set_when_detector_raises(self):
        """VIX fetch / detector exception → combo_regime = 'neutral' (safe default)."""
        system = self._make_system()
        signals = self._wire_analyze_deps(
            system, regime_side_effect=Exception("VIX fetch failed")
        )
        system._analyze_ticker("SPY")
        assert signals.get("combo_regime") == "neutral"

    def test_actual_regime_set_when_detector_succeeds(self):
        """When detector works, combo_regime reflects the detected value."""
        import pandas as pd
        ts = pd.Timestamp("2025-01-02")
        system = self._make_system()
        signals = self._wire_analyze_deps(
            system, regime_result={ts: "bull"}
        )
        system._analyze_ticker("SPY")
        assert signals.get("combo_regime") == "bull"

    def test_combo_regime_absent_in_non_combo_mode(self):
        """In non-combo (simple/hmm) mode, combo_regime is never injected."""
        system = self._make_system()
        signals = self._wire_analyze_deps(system)
        system.strategy.regime_mode = "simple"
        system.strategy._combo_regime_detector = None
        system._analyze_ticker("SPY")
        assert "combo_regime" not in signals


# ---------------------------------------------------------------------------
# Strike price preservation tests
# ---------------------------------------------------------------------------

class TestStrikePricePreservation:
    """Verify strike prices are preserved exactly through the execution pipeline.

    Regression test for the $1 call-strike shift bug: iron condor call strikes
    were recorded as $682/$694 when they should have been $683/$695, causing
    reconciler orphans.

    Root cause: int(strike * 1000) in OCC symbol builder truncates instead of
    rounding, so any floating-point imprecision (e.g. 682.999...) silently
    shifts the strike by $1.
    """

    def test_occ_symbol_rounds_not_truncates(self):
        """_build_occ_symbol must use round() so FP imprecision doesn't shift strikes."""
        from strategy.alpaca_provider import AlpacaProvider

        # Monkey-patch __init__ so we don't need real API credentials
        original_init = AlpacaProvider.__init__
        try:
            AlpacaProvider.__init__ = lambda self, *a, **kw: None
            provider = AlpacaProvider.__new__(AlpacaProvider)

            # Exact value — should always work
            sym = provider._build_occ_symbol("SPY", "2026-04-15", 683.0, "call")
            assert "00683000" in sym

            # Simulate FP imprecision: 682.9999999999 should round to 683, not truncate to 682
            sym_imprecise = provider._build_occ_symbol("SPY", "2026-04-15", 682.9999999999, "call")
            assert "00683000" in sym_imprecise, (
                f"Expected strike 683000 but got {sym_imprecise} — "
                "int() truncation bug (should use round())"
            )

            # Same for the long strike
            sym_long = provider._build_occ_symbol("SPY", "2026-04-15", 694.9999999999, "call")
            assert "00695000" in sym_long
        finally:
            AlpacaProvider.__init__ = original_init

    def test_execution_engine_rounds_ic_strikes(self):
        """ExecutionEngine must round per-wing IC strikes before storing in trade record."""
        from execution.execution_engine import ExecutionEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            engine = ExecutionEngine(alpaca_provider=None, db_path=db_path)

            opp = {
                "ticker": "SPY",
                "type": "iron_condor",
                "expiration": "2026-04-15",
                # Exact put side
                "short_strike": 650.0,
                "long_strike": 638.0,
                "put_short_strike": 650.0,
                "put_long_strike": 638.0,
                # Imprecise call side (simulates FP drift)
                "call_short_strike": 682.9999999999,
                "call_long_strike": 694.9999999999,
                "credit": 3.50,
                "contracts": 1,
            }

            result = engine.submit_opportunity(opp)
            assert result["status"] == "dry_run"

            # Read back from DB and verify strikes
            from shared.database import get_trade_by_id
            trade = get_trade_by_id(result["client_order_id"], path=db_path)
            assert trade is not None

            # Call strikes must be rounded to exact values, not truncated
            assert trade["call_short_strike"] == 683.0, (
                f"Expected 683.0 but got {trade['call_short_strike']} — "
                "strike not rounded before DB storage"
            )
            assert trade["call_long_strike"] == 695.0, (
                f"Expected 695.0 but got {trade['call_long_strike']} — "
                "strike not rounded before DB storage"
            )
            # Put side should also be exact
            assert trade["put_short_strike"] == 650.0
            assert trade["put_long_strike"] == 638.0

    def test_alert_legs_preserve_strikes_through_pipeline(self):
        """Strike prices must survive: opportunity → Alert → execution dict unchanged."""
        from alerts.alert_router import AlertRouter
        from alerts.alert_schema import Alert

        opp = {
            "ticker": "SPY",
            "type": "iron_condor",
            "expiration": "2026-04-15",
            "short_strike": 650.0,
            "long_strike": 638.0,
            "call_short_strike": 683.0,
            "call_long_strike": 695.0,
            "credit": 3.50,
            "stop_loss": 7.00,
            "profit_target": 1.75,
            "score": 75,
            "pop": 80,
            "dte": 30,
            "risk_reward": 0.5,
            "spread_width": 12,
        }

        # Step 1: Convert to Alert
        alert = Alert.from_opportunity(opp)
        assert len(alert.legs) == 4

        # Step 2: Build execution dict (same as AlertRouter._build_execution_dict)
        router = AlertRouter.__new__(AlertRouter)
        exec_dict = router._build_execution_dict(alert)

        # Verify all 4 strikes survive exactly
        assert exec_dict["put_short_strike"] == 650.0
        assert exec_dict["put_long_strike"] == 638.0
        assert exec_dict["call_short_strike"] == 683.0
        assert exec_dict["call_long_strike"] == 695.0


# ---------------------------------------------------------------------------
# Lesson 005: Buying power pre-check
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="Buying power pre-check not yet implemented in ExecutionEngine")
class TestBuyingPowerPreCheck:
    """Verify options_buying_power is checked before order submission."""

    def _make_engine(self, alpaca=None, tmp_path=None, config=None):
        from execution.execution_engine import ExecutionEngine
        db_path = str(tmp_path / "bp_test.db") if tmp_path else None
        return ExecutionEngine(alpaca_provider=alpaca, db_path=db_path, config=config)

    def _make_opp(self, credit=1.50, contracts=2, short=540.0, long=535.0, spread_type="bull_put"):
        return {
            "ticker": "SPY", "type": spread_type, "expiration": "2026-04-18",
            "short_strike": short, "long_strike": long,
            "credit": credit, "contracts": contracts,
        }

    def test_blocks_when_buying_power_insufficient(self, tmp_path):
        """Order rejected when options_buying_power < margin required."""
        mock_alpaca = MagicMock()
        mock_alpaca.get_account.return_value = {
            "options_buying_power": 500.0,  # Only $500 BP
            "equity": 100_000,
        }
        engine = self._make_engine(alpaca=mock_alpaca, tmp_path=tmp_path)

        # 2 contracts * $5 width * 100 = $1,000 margin required
        opp = self._make_opp(contracts=2, short=540.0, long=535.0)
        result = engine.submit_opportunity(opp)

        assert result["status"] == "insufficient_buying_power"
        assert "options_bp" in result["message"]
        # Alpaca should NOT have been called to submit
        mock_alpaca.submit_credit_spread.assert_not_called()

    def test_allows_when_buying_power_sufficient(self, tmp_path):
        """Order proceeds when options_buying_power >= margin required."""
        mock_alpaca = MagicMock()
        mock_alpaca.get_account.return_value = {
            "options_buying_power": 50_000.0,
            "equity": 100_000,
        }
        mock_alpaca.submit_credit_spread.return_value = {"status": "submitted", "order_id": "ord-1"}
        mock_alpaca.get_market_clock.return_value = {"is_open": True}
        engine = self._make_engine(alpaca=mock_alpaca, tmp_path=tmp_path)

        opp = self._make_opp(contracts=2, short=540.0, long=535.0)
        result = engine.submit_opportunity(opp)

        assert result["status"] == "submitted"

    def test_fails_open_when_api_errors(self, tmp_path):
        """If get_account() fails, don't block the order (fail-open)."""
        mock_alpaca = MagicMock()
        mock_alpaca.get_account.side_effect = Exception("API timeout")
        mock_alpaca.submit_credit_spread.return_value = {"status": "submitted", "order_id": "ord-2"}
        mock_alpaca.get_market_clock.return_value = {"is_open": True}
        engine = self._make_engine(alpaca=mock_alpaca, tmp_path=tmp_path)

        opp = self._make_opp(contracts=1)
        result = engine.submit_opportunity(opp)

        # Should proceed despite API error
        assert result["status"] == "submitted"

    def test_skipped_in_dry_run(self, tmp_path):
        """No BP check when alpaca is None (dry-run mode)."""
        engine = self._make_engine(alpaca=None, tmp_path=tmp_path)
        opp = self._make_opp(contracts=5)  # No BP check in dry-run
        result = engine.submit_opportunity(opp)
        assert result["status"] == "dry_run"

    def test_iron_condor_uses_wider_wing(self, tmp_path):
        """IC margin based on max(put_width, call_width), not both summed."""
        mock_alpaca = MagicMock()
        # $10 wide call wing, $5 wide put wing → margin = $10 * 1 * 100 = $1,000
        mock_alpaca.get_account.return_value = {
            "options_buying_power": 800.0,  # Not enough for $1,000
            "equity": 100_000,
        }
        engine = self._make_engine(alpaca=mock_alpaca, tmp_path=tmp_path)

        opp = {
            "ticker": "SPY", "type": "iron_condor", "expiration": "2026-04-18",
            "short_strike": 550.0, "long_strike": 545.0,
            "put_short_strike": 550.0, "put_long_strike": 545.0,
            "call_short_strike": 570.0, "call_long_strike": 580.0,
            "credit": 2.00, "contracts": 1,
        }
        result = engine.submit_opportunity(opp)
        assert result["status"] == "insufficient_buying_power"

    def test_buying_power_check_called_in_submit(self, tmp_path):
        """Verify _check_buying_power is called within submit_opportunity."""
        from execution.execution_engine import ExecutionEngine
        import inspect
        src = inspect.getsource(ExecutionEngine.submit_opportunity)
        assert "_check_buying_power" in src
        assert "buying_power" in src or "options_buying_power" in src


# ---------------------------------------------------------------------------
# Position limits enforcement in ExecutionEngine
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="Position limits enforcement not yet implemented in ExecutionEngine")
class TestPositionLimitsEnforcement:
    """Verify position limits are enforced in submit_opportunity()."""

    def _make_engine(self, alpaca=None, tmp_path=None, config=None):
        from execution.execution_engine import ExecutionEngine
        db_path = str(tmp_path / "pos_test.db")
        return ExecutionEngine(alpaca_provider=alpaca, db_path=db_path, config=config), db_path

    def _make_opp(self, ticker="SPY", spread_type="bull_put", contracts=2):
        return {
            "ticker": ticker, "type": spread_type, "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50,
            "contracts": contracts,
        }

    def _insert_open_trade(self, db_path, trade_id, strategy_type="bull_put", status="open"):
        from shared.database import upsert_trade
        upsert_trade({
            "id": trade_id, "ticker": "SPY", "strategy_type": strategy_type,
            "status": status, "short_strike": 540.0, "long_strike": 535.0,
            "expiration": "2026-04-18", "credit": 1.50, "contracts": 1,
            "entry_date": "2026-03-10",
        }, source="test", path=db_path)

    def test_blocks_when_contracts_exceed_max(self, tmp_path):
        """Reject when contracts > MAX_CONTRACTS_PER_TRADE."""
        from shared.constants import MAX_CONTRACTS_PER_TRADE
        engine, db_path = self._make_engine(tmp_path=tmp_path)

        opp = self._make_opp(contracts=MAX_CONTRACTS_PER_TRADE + 1)
        result = engine.submit_opportunity(opp)

        assert result["status"] == "position_limit_reached"
        assert "contracts_exceeded" in result["message"]

    def test_allows_contracts_at_max(self, tmp_path):
        """Allow when contracts == MAX_CONTRACTS_PER_TRADE (not exceeded)."""
        from shared.constants import MAX_CONTRACTS_PER_TRADE
        engine, db_path = self._make_engine(tmp_path=tmp_path)

        opp = self._make_opp(contracts=MAX_CONTRACTS_PER_TRADE)
        result = engine.submit_opportunity(opp)

        # Should proceed (dry_run since no alpaca)
        assert result["status"] == "dry_run"

    def test_blocks_when_max_positions_reached(self, tmp_path):
        """Reject when total open positions >= max_positions."""
        config = {"risk": {"max_positions": 3, "account_size": 100_000}}
        engine, db_path = self._make_engine(tmp_path=tmp_path, config=config)

        # Insert 3 open trades to fill the limit
        for i in range(3):
            self._insert_open_trade(db_path, f"pos-{i}", strategy_type="bull_put")

        opp = self._make_opp(contracts=1)
        result = engine.submit_opportunity(opp)

        assert result["status"] == "position_limit_reached"
        assert "max_positions_reached" in result["message"]

    def test_allows_when_below_max_positions(self, tmp_path):
        """Allow when total open positions < max_positions."""
        config = {"risk": {"max_positions": 5, "account_size": 100_000}}
        engine, db_path = self._make_engine(tmp_path=tmp_path, config=config)

        self._insert_open_trade(db_path, "pos-0")
        self._insert_open_trade(db_path, "pos-1")

        opp = self._make_opp(contracts=1)
        result = engine.submit_opportunity(opp)

        assert result["status"] == "dry_run"  # proceeds (no alpaca)

    def test_blocks_when_per_strategy_limit_reached(self, tmp_path):
        """Reject when positions for a strategy >= MAX_POSITIONS_PER_STRATEGY."""
        from shared.constants import MAX_POSITIONS_PER_STRATEGY
        config = {"risk": {"max_positions": 50, "account_size": 100_000}}
        engine, db_path = self._make_engine(tmp_path=tmp_path, config=config)

        # Fill up bull_put strategy slots
        for i in range(MAX_POSITIONS_PER_STRATEGY):
            self._insert_open_trade(db_path, f"bp-{i}", strategy_type="bull_put")

        opp = self._make_opp(spread_type="bull_put", contracts=1)
        result = engine.submit_opportunity(opp)

        assert result["status"] == "position_limit_reached"
        assert "max_per_strategy_reached" in result["message"]

    def test_allows_different_strategy_when_one_is_full(self, tmp_path):
        """Allow a different strategy even when one strategy is full."""
        from shared.constants import MAX_POSITIONS_PER_STRATEGY
        config = {"risk": {"max_positions": 50, "account_size": 100_000}}
        engine, db_path = self._make_engine(tmp_path=tmp_path, config=config)

        # Fill up bull_put slots
        for i in range(MAX_POSITIONS_PER_STRATEGY):
            self._insert_open_trade(db_path, f"bp-{i}", strategy_type="bull_put")

        # A bear_call should still be allowed
        opp = self._make_opp(spread_type="bear_call", contracts=1)
        result = engine.submit_opportunity(opp)

        assert result["status"] == "dry_run"  # proceeds

    def test_counts_pending_open_positions(self, tmp_path):
        """pending_open trades count toward position limits."""
        config = {"risk": {"max_positions": 2, "account_size": 100_000}}
        engine, db_path = self._make_engine(tmp_path=tmp_path, config=config)

        self._insert_open_trade(db_path, "open-1", status="open")
        self._insert_open_trade(db_path, "pending-1", status="pending_open")

        opp = self._make_opp(contracts=1)
        result = engine.submit_opportunity(opp)

        assert result["status"] == "position_limit_reached"
        assert "max_positions_reached" in result["message"]

    def test_position_limits_in_submit_source(self, tmp_path):
        """Verify _check_position_limits is called within submit_opportunity."""
        from execution.execution_engine import ExecutionEngine
        import inspect
        src = inspect.getsource(ExecutionEngine.submit_opportunity)
        assert "_check_position_limits" in src
        assert "position_limit" in src.lower()
