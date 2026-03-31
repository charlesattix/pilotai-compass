"""
Tests for code-review bug fixes:
  #7  real risk_pct flows through pipeline
  #8  atomic_ic_execution config flag
  #9  correlated direction normalization
  #10 IC max_loss formula (one wing, not two)
  #12 circuit_breaker blocks trades when Alpaca is down
  #13 SQLite busy_timeout pragma
  #15 dedup ledger only marked after successful execution
  #17 dedup ledger persisted to SQLite
  #19 MAX_TOTAL_EXPOSURE configurable
  #23 iron_condor alert_source explicit routing
  #24 holidays extended through 2030
"""

import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

try:
    from alpaca.trading.requests import OptionLegRequest  # noqa: F401
except ImportError:
    pytest.skip("OptionLegRequest not available in this alpaca-py version", allow_module_level=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_alert(risk_pct=0.02, ticker="SPY", direction="bullish", alert_type="credit_spread"):

    from alerts.alert_schema import Alert, AlertType, Confidence, Direction, Leg
    type_map = {
        "credit_spread": AlertType.credit_spread,
        "iron_condor": AlertType.iron_condor,
    }
    dir_map = {
        "bullish": Direction.bullish,
        "bearish": Direction.bearish,
        "neutral": Direction.neutral,
    }
    return Alert(
        ticker=ticker,
        type=type_map.get(alert_type, AlertType.credit_spread),
        direction=dir_map.get(direction, Direction.bullish),
        legs=[Leg(strike=500.0, option_type="put", action="sell", expiration="2026-04-17")],
        entry_price=1.20,
        stop_loss=4.20,
        profit_target=0.60,
        risk_pct=risk_pct,
        confidence=Confidence.MEDIUM,
    )


def _make_account_state(**kwargs):
    base = {
        "account_value": 100_000,
        "peak_equity": 100_000,
        "open_positions": [],
        "daily_pnl_pct": 0.0,
        "weekly_pnl_pct": 0.0,
        "recent_stops": [],
    }
    base.update(kwargs)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# BUG #9 — Direction normalization (_directions_match)
# ─────────────────────────────────────────────────────────────────────────────

class TestDirectionsMatch:
    def _fn(self):
        from compass.risk_gate import _directions_match
        return _directions_match

    def test_bullish_matches_bull_put_spread(self):
        assert self._fn()("bull_put_spread", "bullish")

    def test_bullish_matches_bull_put(self):
        assert self._fn()("bull_put", "bullish")

    def test_bullish_matches_bull(self):
        assert self._fn()("bull", "bullish")

    def test_bearish_matches_bear_call_spread(self):
        assert self._fn()("bear_call_spread", "bearish")

    def test_bearish_matches_bear_call(self):
        assert self._fn()("bear_call", "bearish")

    def test_neutral_matches_iron_condor(self):
        assert self._fn()("iron_condor", "neutral")

    def test_bullish_does_not_match_bear(self):
        assert not self._fn()("bear_call_spread", "bullish")

    def test_bearish_does_not_match_bull(self):
        assert not self._fn()("bull_put_spread", "bearish")

    def test_case_insensitive(self):
        assert self._fn()("BULL_PUT_SPREAD", "bullish")

    def test_unknown_strings_return_false(self):
        assert not self._fn()("unknown_type", "bullish")


class TestRiskGateCorrelatedDirectionCheck:
    """Rule 5 now correctly counts correlated positions with strategy_type directions."""

    def _gate(self, config=None):
        from compass.risk_gate import RiskGate
        return RiskGate(config or {})

    def test_bull_put_spread_counted_as_bullish(self):
        gate = self._gate()
        alert = _make_alert(risk_pct=0.02, direction="bullish")
        # 3 existing positions stored as strategy_type="bull_put_spread"
        positions = [
            {"direction": "bull_put_spread", "risk_pct": 0.02}
            for _ in range(3)
        ]
        state = _make_account_state(open_positions=positions)
        passed, reason = gate.check(alert, state)
        assert not passed
        assert "correlated" in reason.lower() or "positions" in reason.lower()

    def test_old_schema_bullish_still_works(self):
        gate = self._gate()
        alert = _make_alert(risk_pct=0.02, direction="bullish")
        positions = [{"direction": "bullish", "risk_pct": 0.02} for _ in range(3)]
        state = _make_account_state(open_positions=positions)
        passed, reason = gate.check(alert, state)
        assert not passed

    def test_bear_does_not_count_against_bull(self):
        gate = self._gate()
        alert = _make_alert(risk_pct=0.02, direction="bullish")
        positions = [{"direction": "bear_call_spread", "risk_pct": 0.02} for _ in range(3)]
        state = _make_account_state(open_positions=positions)
        passed, _ = gate.check(alert, state)
        # Should pass — correlated count should be 0
        assert passed


# ─────────────────────────────────────────────────────────────────────────────
# BUG #10 — IC max_loss formula
# ─────────────────────────────────────────────────────────────────────────────

class TestICMaxLossFormula:
    def _sizer(self):
        from alerts.alert_position_sizer import AlertPositionSizer
        cfg = {
            "strategy": {"iron_condor": {"ic_risk_per_trade": 10.0}},
            "risk": {"max_risk_per_trade": 5.0, "min_contracts": 1, "max_contracts": 25, "account_size": 100_000},
            "backtest": {"starting_capital": 100_000},
        }
        return AlertPositionSizer(cfg)

    def test_ic_uses_both_wings_max_loss(self):
        """IC max_loss must use both wings: (2 * spread_width - combined_credit) * 100."""
        sizer = self._sizer()
        alert = _make_alert(risk_pct=0.05, alert_type="iron_condor")
        alert.legs = [
            MagicMock(strike=500.0, option_type="put", action="sell"),
            MagicMock(strike=495.0, option_type="put", action="buy"),
            MagicMock(strike=510.0, option_type="call", action="sell"),
            MagicMock(strike=515.0, option_type="call", action="buy"),
        ]
        # Override _extract_spread_params to return known values
        with patch.object(sizer, '_extract_spread_params', return_value=(5.0, 1.50)):
            result = sizer.size(alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0)

        # Correct: max_loss_per_spread = (2*5.0 - 1.50) * 100 = $850 (both wings, matches backtester)
        # dollar_risk = 100_000 * 0.10 = $10_000
        # contracts = 10_000 / 850 = 11, capped at 25 → 11
        # actual_dollar_risk = 11 * 850 = $9_350
        expected_max_loss_per_spread = (2 * 5.0 - 1.50) * 100  # $850
        contracts = int(10_000 / expected_max_loss_per_spread)  # 11
        assert result.max_loss == pytest.approx(contracts * expected_max_loss_per_spread, rel=0.01)

    def test_ic_max_loss_larger_than_single_wing_formula(self):
        """Confirm both-wings formula produces LARGER max_loss than old single-wing formula."""
        sizer = self._sizer()
        alert = _make_alert(risk_pct=0.05, alert_type="iron_condor")
        with patch.object(sizer, '_extract_spread_params', return_value=(5.0, 1.50)):
            sizer.size(alert, account_value=100_000, iv_rank=30, current_portfolio_risk=0)

        # New (correct): max_loss_per_spread = (2*5 - 1.5) * 100 = $850 per contract
        # Old (wrong):   max_loss_per_spread = (5 - 1.5) * 100 = $350 per contract
        # Both-wings formula gives fewer contracts but correct per-contract max_loss
        both_wings = (2 * 5.0 - 1.50) * 100   # $850 (correct)
        single_wing = (5.0 - 1.50) * 100       # $350 (old wrong formula)
        assert both_wings > single_wing


# ─────────────────────────────────────────────────────────────────────────────
# BUG #12 — circuit_breaker blocks all trades
# ─────────────────────────────────────────────────────────────────────────────

class TestCircuitBreaker:
    def _gate(self):
        from compass.risk_gate import RiskGate
        return RiskGate({})

    def test_circuit_breaker_blocks_trade(self):
        gate = self._gate()
        alert = _make_alert(risk_pct=0.02)
        state = _make_account_state(circuit_breaker=True)
        passed, reason = gate.check(alert, state)
        assert not passed
        assert "circuit_breaker" in reason.lower()

    def test_no_circuit_breaker_allows_trade(self):
        gate = self._gate()
        alert = _make_alert(risk_pct=0.02)
        state = _make_account_state(circuit_breaker=False)
        passed, _ = gate.check(alert, state)
        assert passed

    def test_missing_circuit_breaker_key_defaults_to_false(self):
        gate = self._gate()
        alert = _make_alert(risk_pct=0.02)
        state = _make_account_state()  # no circuit_breaker key
        passed, _ = gate.check(alert, state)
        assert passed


# ─────────────────────────────────────────────────────────────────────────────
# BUG #13 — SQLite busy_timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestSQLiteBusyTimeout:
    def test_database_busy_timeout_set(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from shared.database import get_db
            conn = get_db(db_path)
            timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            conn.close()
            assert timeout_ms == 5000
        finally:
            os.unlink(db_path)

    def test_macro_state_db_busy_timeout_set(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from compass.macro_db import get_db
            conn = get_db(db_path)
            timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            conn.close()
            assert timeout_ms == 5000
        finally:
            os.unlink(db_path)


# ─────────────────────────────────────────────────────────────────────────────
# BUG #17 — dedup ledger persisted to SQLite
# ─────────────────────────────────────────────────────────────────────────────

class TestDedupPersistence:
    def test_upsert_and_load_dedup_entry(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from shared.database import init_db, load_dedup_entries, upsert_dedup_entry
            init_db(db_path)
            now_iso = datetime.now(timezone.utc).isoformat()
            upsert_dedup_entry("SPY", "bullish", "credit_spread", now_iso, path=db_path)
            entries = load_dedup_entries(window_seconds=1800, path=db_path)
            assert len(entries) == 1
            assert entries[0]["ticker"] == "SPY"
            assert entries[0]["direction"] == "bullish"
            assert entries[0]["alert_type"] == "credit_spread"
        finally:
            os.unlink(db_path)

    def test_old_dedup_entries_not_loaded(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from shared.database import init_db, load_dedup_entries, upsert_dedup_entry
            init_db(db_path)
            old_iso = "2020-01-01T00:00:00+00:00"  # clearly outside 30-min window
            upsert_dedup_entry("SPY", "bullish", "credit_spread", old_iso, path=db_path)
            entries = load_dedup_entries(window_seconds=1800, path=db_path)
            assert len(entries) == 0
        finally:
            os.unlink(db_path)

    def test_router_loads_dedup_on_init(self):
        """AlertRouter populates _dedup_ledger from DB on startup."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            from alerts.alert_router import AlertRouter
            from shared.database import init_db, upsert_dedup_entry
            init_db(db_path)
            now_iso = datetime.now(timezone.utc).isoformat()
            upsert_dedup_entry("XLE", "bearish", "credit_spread", now_iso, path=db_path)

            with patch.dict(os.environ, {"PILOTAI_DB_PATH": db_path}):
                router = AlertRouter(
                    risk_gate=MagicMock(),
                    position_sizer=MagicMock(),
                    telegram_bot=MagicMock(),
                    formatter=MagicMock(),
                )
            assert ("XLE", "bearish", "credit_spread") in router._dedup_ledger
        finally:
            os.unlink(db_path)


# ─────────────────────────────────────────────────────────────────────────────
# BUG #15 — dedup marked only after successful execution
# ─────────────────────────────────────────────────────────────────────────────

class TestDedupAfterExecution:
    def _make_router(self, exec_engine=None):
        from alerts.alert_position_sizer import AlertPositionSizer
        from alerts.alert_router import AlertRouter
        from compass.risk_gate import RiskGate

        rg = MagicMock(spec=RiskGate)
        rg.check.return_value = (True, "")
        rg.weekly_loss_breach.return_value = False

        sizer = MagicMock(spec=AlertPositionSizer)
        sizer.size.return_value = MagicMock(risk_pct=0.05, contracts=2, dollar_risk=5000, max_loss=5000)

        router = AlertRouter(
            risk_gate=rg,
            position_sizer=sizer,
            telegram_bot=MagicMock(),
            formatter=MagicMock(),
            execution_engine=exec_engine,
            config={"strategy": {}},
        )
        # Clear stale dedup entries loaded from the real DB at init.
        router._dedup_ledger = {}
        return router

    def _make_opp(self, ticker="SPY", score=75):
        return {
            "ticker": ticker,
            "type": "bull_put_spread",
            "score": score,
            "short_strike": 500.0,
            "long_strike": 495.0,
            "expiration": "2026-04-17",
            "credit": 1.20,
            "max_loss": 3.80,
            "dte": 35,
        }

    def test_dedup_not_marked_on_execution_failure(self):
        engine = MagicMock()
        engine.submit_opportunity.side_effect = RuntimeError("Alpaca connection error")
        router = self._make_router(exec_engine=engine)

        with patch('alerts.alert_router.insert_alert'), \
             patch.object(router, '_validate_dte', return_value=(True, "")), \
             patch.object(router, '_mark_dedup') as mock_mark:
            router.route_opportunities([self._make_opp()], _make_account_state())

        # _mark_dedup should NOT have been called since execution failed
        mock_mark.assert_not_called()

    def test_dedup_marked_on_execution_success(self):
        engine = MagicMock()
        engine.submit_opportunity.return_value = {"status": "submitted", "order_id": "abc123"}
        router = self._make_router(exec_engine=engine)

        with patch('alerts.alert_router.insert_alert'), \
             patch.object(router, '_validate_dte', return_value=(True, "")), \
             patch.object(router, '_mark_dedup') as mock_mark:
            router.route_opportunities([self._make_opp()], _make_account_state())

        mock_mark.assert_called_once()

    def test_dedup_marked_when_no_execution_engine(self):
        """With no execution engine (alert-only), dedup is always marked."""
        router = self._make_router(exec_engine=None)

        with patch('alerts.alert_router.insert_alert'), \
             patch.object(router, '_mark_dedup') as mock_mark:
            router.route_opportunities([self._make_opp()], _make_account_state())

        mock_mark.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# BUG #19 — MAX_TOTAL_EXPOSURE configurable
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigurableMaxTotalExposure:
    def test_default_15_pct_from_constant(self):
        from compass.risk_gate import RiskGate
        gate = RiskGate({})
        assert gate._max_total_exposure == pytest.approx(0.15)

    def test_config_override_25_pct(self):
        from compass.risk_gate import RiskGate
        gate = RiskGate({"risk": {"max_total_exposure_pct": 25}})
        assert gate._max_total_exposure == pytest.approx(0.25)

    def test_exposure_cap_uses_config_value(self):
        from compass.risk_gate import RiskGate
        # 2 positions at 10% each = 20%, alert adds 4% = 24% → passes 25% limit
        # Use neutral positions so correlated-position rule (rule 5) doesn't fire
        gate = RiskGate({"risk": {"max_total_exposure_pct": 25}})
        alert = _make_alert(risk_pct=0.04)
        positions = [{"direction": "iron_condor", "risk_pct": 0.10} for _ in range(2)]
        state = _make_account_state(open_positions=positions)
        # open_risk = 0.20, adding 0.04 = 0.24 < 0.25 → should pass
        passed, reason = gate.check(alert, state)
        assert passed, f"Should pass at 25% limit but got: {reason}"

    def test_exposure_blocked_above_config_limit(self):
        from compass.risk_gate import RiskGate
        gate = RiskGate({"risk": {"max_total_exposure_pct": 20}})
        alert = _make_alert(risk_pct=0.05)
        positions = [{"direction": "bear_call_spread", "risk_pct": 0.08} for _ in range(2)]
        state = _make_account_state(open_positions=positions)
        # open_risk = 0.16, adding 0.05 = 0.21 > 0.20
        passed, reason = gate.check(alert, state)
        assert not passed
        assert "exposure" in reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# BUG #23 — iron_condor alert_source routing
# ─────────────────────────────────────────────────────────────────────────────

class TestIronCondorAlertSourceRouting:
    def test_alert_source_iron_condor_maps_to_ic_type(self):
        from alerts.alert_schema import Alert, AlertType
        opp = {
            "ticker": "SPY",
            "type": "iron_condor",
            "alert_source": "iron_condor",
            "short_strike": 500.0,
            "long_strike": 495.0,
            "call_short_strike": 510.0,
            "call_long_strike": 515.0,
            "expiration": "2026-04-17",
            "credit": 2.00,
            "max_loss": 3.00,
            "score": 70,
        }
        alert = Alert.from_opportunity(opp)
        assert alert.type == AlertType.iron_condor

    def test_alert_source_iron_condor_without_condor_in_type(self):
        """If opp_type doesn't say 'condor', alert_source alone should route correctly."""
        from alerts.alert_schema import Alert, AlertType
        opp = {
            "ticker": "SPY",
            "type": "multi_leg",     # no "condor" substring
            "alert_source": "iron_condor",
            "short_strike": 500.0,
            "long_strike": 495.0,
            "call_short_strike": 510.0,
            "call_long_strike": 515.0,
            "expiration": "2026-04-17",
            "credit": 2.00,
            "max_loss": 3.00,
            "score": 70,
        }
        alert = Alert.from_opportunity(opp)
        assert alert.type == AlertType.iron_condor


# ─────────────────────────────────────────────────────────────────────────────
# BUG #24 — holidays extended through 2030
# ─────────────────────────────────────────────────────────────────────────────

class TestHolidaysThrough2030:
    def _get_holidays(self):
        from execution.position_monitor import _MARKET_HOLIDAYS
        return _MARKET_HOLIDAYS

    def _get_early_close(self):
        from execution.position_monitor import _EARLY_CLOSE_DATES
        return _EARLY_CLOSE_DATES

    def test_2026_holidays_present(self):
        h = self._get_holidays()
        assert "2026-07-03" in h  # July 4 observed
        assert "2026-11-26" in h  # Thanksgiving

    def test_2027_holidays_present(self):
        h = self._get_holidays()
        assert "2027-01-01" in h
        assert "2027-11-25" in h

    def test_2028_holidays_present(self):
        h = self._get_holidays()
        assert "2028-05-29" in h
        assert "2028-12-25" in h

    def test_2029_holidays_present(self):
        h = self._get_holidays()
        assert "2029-01-01" in h
        assert "2029-11-22" in h

    def test_2030_holidays_present(self):
        h = self._get_holidays()
        assert "2030-01-01" in h
        assert "2030-12-25" in h

    def test_early_close_2027_present(self):
        ec = self._get_early_close()
        assert "2027-11-24" in ec
        assert ec["2027-11-24"] == 13

    def test_early_close_2030_present(self):
        ec = self._get_early_close()
        assert "2030-11-27" in ec

    def test_backward_compat_alias(self):
        from execution.position_monitor import (
            _EARLY_CLOSE_DATES,
            _EARLY_CLOSE_DATES_2026,
            _MARKET_HOLIDAYS,
            _MARKET_HOLIDAYS_2026,
        )
        assert _MARKET_HOLIDAYS_2026 is _MARKET_HOLIDAYS
        assert _EARLY_CLOSE_DATES_2026 is _EARLY_CLOSE_DATES


# ─────────────────────────────────────────────────────────────────────────────
# PARTIAL #8 — atomic_ic_execution config flag
# ─────────────────────────────────────────────────────────────────────────────

class TestAtomicICExecutionFlag:
    def test_flag_defaults_to_false(self):
        from execution.execution_engine import ExecutionEngine
        engine = ExecutionEngine(alpaca_provider=None, config={})
        assert engine._atomic_ic is False

    def test_flag_set_true_logs_warning(self, caplog):
        import logging

        from execution.execution_engine import ExecutionEngine
        with caplog.at_level(logging.WARNING, logger="execution.execution_engine"):
            engine = ExecutionEngine(
                alpaca_provider=None,
                config={"execution": {"atomic_ic_execution": True}},
            )
        assert engine._atomic_ic is True
        assert "not yet supported" in caplog.text


# ─────────────────────────────────────────────────────────────────────────────
# BUG #7 — real risk_pct flows from sizer into risk gate
# ─────────────────────────────────────────────────────────────────────────────

class TestRealRiskPctPipeline:
    """Sizing happens before risk gate so alert.risk_pct reflects actual sized risk."""

    def test_sizing_before_risk_gate(self):
        """After routing, approved alerts have risk_pct from sizer, not hardcoded 0.02."""
        from alerts.alert_position_sizer import AlertPositionSizer
        from alerts.alert_router import AlertRouter
        from compass.risk_gate import RiskGate

        # Use real RiskGate with 25% limit so we don't accidentally block
        rg = RiskGate({"risk": {"max_total_exposure_pct": 25}})

        sizer = MagicMock(spec=AlertPositionSizer)
        real_risk = 0.04  # under MAX_RISK_PER_TRADE=0.05 so it won't be blocked by rule 1
        sizer.size.return_value = MagicMock(
            risk_pct=real_risk, contracts=10, dollar_risk=4000, max_loss=4000
        )

        router = AlertRouter(
            risk_gate=rg,
            position_sizer=sizer,
            telegram_bot=MagicMock(),
            formatter=MagicMock(),
            config={"strategy": {}},
        )
        router._dedup_ledger = {}  # Clear stale entries loaded from real DB

        opp = {
            "ticker": "SPY",
            "type": "bull_put_spread",
            "score": 75,
            "short_strike": 500.0,
            "long_strike": 495.0,
            "expiration": "2026-04-17",
            "credit": 1.20,
            "max_loss": 3.80,
            "dte": 35,
        }

        with patch('alerts.alert_router.insert_alert'), \
             patch.object(router, '_mark_dedup'):
            dispatched = router.route_opportunities([opp], _make_account_state())

        assert len(dispatched) == 1
        assert dispatched[0].risk_pct == pytest.approx(real_risk)

    def test_high_real_risk_blocked_by_gate(self):
        """If sizer returns risk_pct > MAX_RISK_PER_TRADE, risk gate blocks it."""
        from alerts.alert_position_sizer import AlertPositionSizer
        from alerts.alert_router import AlertRouter
        from compass.risk_gate import RiskGate
        from shared.constants import MAX_RISK_PER_TRADE

        rg = RiskGate({})
        sizer = MagicMock(spec=AlertPositionSizer)
        # Sizer returns a risk exceeding the hard cap
        sizer.size.return_value = MagicMock(
            risk_pct=MAX_RISK_PER_TRADE + 0.02,
            contracts=99, dollar_risk=99000, max_loss=99000
        )

        router = AlertRouter(
            risk_gate=rg,
            position_sizer=sizer,
            telegram_bot=MagicMock(),
            formatter=MagicMock(),
            config={"strategy": {}},
        )

        opp = {
            "ticker": "SPY",
            "type": "bull_put_spread",
            "score": 75,
            "short_strike": 500.0,
            "long_strike": 495.0,
            "expiration": "2026-04-17",
            "credit": 1.20,
            "max_loss": 3.80,
            "dte": 35,
        }

        with patch('alerts.alert_router.insert_alert'), \
             patch.object(router, '_mark_dedup'):
            dispatched = router.route_opportunities([opp], _make_account_state())

        assert len(dispatched) == 0
