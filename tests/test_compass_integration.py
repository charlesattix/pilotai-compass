"""
Integration tests for the COMPASS package.

Tests cover:
  - compass/__init__.py exports all expected symbols
  - Backward compatibility shims (old import paths still work)
  - RegimeClassifier → sizing pipeline (end-to-end flow)
  - RegimeClassifier → RiskGate → sizing (full pipeline)
  - Event gate → composite scaling → DB persistence

These tests verify that the refactored compass/ package works as an
integrated whole, not just individual unit tests.
"""

import importlib
import warnings
from datetime import date, timedelta

import pandas as pd
import pytest

import compass
from compass.regime import Regime, RegimeClassifier, REGIME_INFO
from compass.events import (
    get_upcoming_events,
    compute_composite_scaling,
    run_daily_event_check,
    ALL_FOMC_DATES,
)
from compass.sizing import calculate_dynamic_risk, get_contract_size
from compass.risk_gate import RiskGate

from tests.compass_helpers import (
    mock_spy_prices,
    mock_vix_series,
    mock_spy_dataframe,
    mock_macro_db,
    KNOWN_FOMC_DATES,
    ACCOUNT_100K,
    SPREAD_WIDTH_5,
    CREDIT_065,
)


# ══════════════════════════════════════════════════════════════════════════════
# A. compass/__init__.py exports
# ══════════════════════════════════════════════════════════════════════════════

class TestCompassExports:
    """Verify that compass/__init__.py exports all expected symbols."""

    EXPECTED_SYMBOLS = [
        # regime
        "Regime", "RegimeClassifier", "REGIME_INFO", "ComboRegimeDetector",
        # macro
        "MacroSnapshotEngine",
        # macro_db
        "init_db", "get_db", "get_current_macro_score", "get_sector_rankings",
        "get_event_scaling_factor", "get_eligible_underlyings", "save_snapshot",
        "MACRO_DB_PATH", "LIQUID_SECTOR_ETFS",
        # events
        "get_upcoming_events", "compute_composite_scaling",
        "run_daily_event_check", "ALL_FOMC_DATES",
        # risk
        "RiskGate",
        # sizing
        "calculate_dynamic_risk", "get_contract_size", "PositionSizer",
        # ML
        "SignalModel", "EnsembleSignalModel", "FeatureEngine", "IVAnalyzer",
        "MLEnhancedStrategy", "confidence_to_size_multiplier", "RegimeModelRouter",
        # Stress testing
        "StressTester", "CRISIS_SCENARIOS",
    ]

    def test_all_expected_symbols_in_all(self):
        """Every expected symbol is in compass.__all__."""
        for sym in self.EXPECTED_SYMBOLS:
            assert sym in compass.__all__, f"Missing from __all__: {sym}"

    def test_all_expected_symbols_accessible(self):
        """Every expected symbol is accessible as compass.Name."""
        for sym in self.EXPECTED_SYMBOLS:
            assert hasattr(compass, sym), f"Not accessible: compass.{sym}"

    def test_no_unexpected_symbols_in_all(self):
        """__all__ doesn't contain symbols we haven't accounted for."""
        for sym in compass.__all__:
            assert sym in self.EXPECTED_SYMBOLS, (
                f"Unexpected symbol in __all__: {sym} "
                f"— add to EXPECTED_SYMBOLS if intentional"
            )


# ══════════════════════════════════════════════════════════════════════════════
# B. Backward compatibility shims
# ══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatShims:
    """Remaining shims that have real consumers (deploy/ scripts)."""

    def test_shared_macro_event_gate_shim(self):
        """shared.macro_event_gate exports ALL_FOMC_DATES and get_upcoming_events."""
        mod = importlib.import_module("shared.macro_event_gate")
        assert hasattr(mod, "ALL_FOMC_DATES")
        assert hasattr(mod, "get_upcoming_events")
        assert mod.ALL_FOMC_DATES is ALL_FOMC_DATES

    def test_shared_macro_state_db_shim(self):
        """shared.macro_state_db exports core DB functions."""
        mod = importlib.import_module("shared.macro_state_db")
        assert hasattr(mod, "init_db")
        assert hasattr(mod, "get_current_macro_score")

    def test_ml_package_reexports(self):
        """ml package re-exports compass symbols without deprecation warnings."""
        import ml
        assert hasattr(ml, "SignalModel")
        assert hasattr(ml, "FeatureEngine")
        assert hasattr(ml, "IVAnalyzer")
        assert hasattr(ml, "PositionSizer")


# ══════════════════════════════════════════════════════════════════════════════
# C. RegimeClassifier → Sizing pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestRegimeToSizingPipeline:
    """End-to-end: classify regime → adjust sizing based on regime."""

    # Regime → risk_pct scaling (from champion config pattern)
    REGIME_RISK_SCALE = {
        Regime.BULL: 1.0,
        Regime.BEAR: 0.3,
        Regime.HIGH_VOL: 0.3,
        Regime.LOW_VOL: 0.8,
        Regime.CRASH: 0.0,
    }

    def test_bull_regime_full_sizing(self):
        """Bull regime → full position size."""
        clf = RegimeClassifier()
        prices = mock_spy_prices(days=100, trend=25.0, base=450.0)
        regime = clf.classify(vix=16.0, spy_prices=prices, date=prices.index[-1])
        assert regime == Regime.BULL

        scale = self.REGIME_RISK_SCALE[regime]
        base_risk = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=0.0
        )
        scaled_risk = base_risk * scale
        assert scaled_risk == pytest.approx(2000.0)  # Full size

        contracts = get_contract_size(
            trade_dollar_risk=scaled_risk,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
        )
        assert contracts > 0

    def test_crash_regime_zero_sizing(self):
        """Crash regime → zero position size (no trades)."""
        from tests.compass_helpers import REGIME_SCENARIOS
        scenario = REGIME_SCENARIOS["crash"]
        clf = RegimeClassifier()
        prices = scenario.build_prices()
        regime = clf.classify(vix=45.0, spy_prices=prices, date=prices.index[-1])
        assert regime == Regime.CRASH

        scale = self.REGIME_RISK_SCALE[regime]
        assert scale == 0.0
        base_risk = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=0.0
        )
        scaled_risk = base_risk * scale
        assert scaled_risk == 0.0

        contracts = get_contract_size(
            trade_dollar_risk=scaled_risk,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
        )
        assert contracts == 0

    def test_bear_regime_reduced_sizing(self):
        """Bear regime → 30% of normal position size."""
        clf = RegimeClassifier()
        prices = mock_spy_prices(days=100, trend=-25.0, base=450.0)
        regime = clf.classify(vix=27.0, spy_prices=prices, date=prices.index[-1])
        assert regime == Regime.BEAR

        scale = self.REGIME_RISK_SCALE[regime]
        assert scale == 0.3
        base_risk = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=0.0
        )
        scaled_risk = base_risk * scale
        assert scaled_risk == pytest.approx(600.0)

        contracts = get_contract_size(
            trade_dollar_risk=scaled_risk,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
        )
        assert contracts == 1  # floor(600 / 435) = 1


# ══════════════════════════════════════════════════════════════════════════════
# D. RiskGate integration
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskGateIntegration:
    """RiskGate works with Alert objects from alerts.alert_schema."""

    @staticmethod
    def _make_alert(risk_pct=0.02, direction="bullish", ticker="SPY"):
        """Create a minimal Alert for risk gate testing."""
        from alerts.alert_schema import Alert, AlertType, Direction, Leg
        return Alert(
            type=AlertType.credit_spread,
            ticker=ticker,
            direction=Direction(direction),
            legs=[Leg(strike=440.0, option_type="put", action="sell", expiration="2026-04-17")],
            entry_price=0.65,
            stop_loss=2.50,
            profit_target=0.10,
            risk_pct=risk_pct,
        )

    def test_risk_gate_approves_small_trade(self):
        """A standard 2% trade with clean account state passes."""
        gate = RiskGate()
        alert = self._make_alert(risk_pct=0.02)
        account_state = {
            "account_value": ACCOUNT_100K,
            "open_positions": [],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
            "circuit_breaker": False,
        }
        approved, reason = gate.check(alert, account_state)
        assert approved is True
        assert reason == ""

    def test_risk_gate_blocks_circuit_breaker(self):
        """Circuit breaker active → blocks all trades."""
        gate = RiskGate()
        alert = self._make_alert(risk_pct=0.02)
        account_state = {
            "account_value": ACCOUNT_100K,
            "open_positions": [],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
            "circuit_breaker": True,
        }
        approved, reason = gate.check(alert, account_state)
        assert approved is False
        assert "circuit_breaker" in reason

    def test_risk_gate_blocks_daily_loss_limit(self):
        """Daily P&L below -8% (DAILY_LOSS_LIMIT) → blocks new trades."""
        gate = RiskGate()
        alert = self._make_alert(risk_pct=0.02)
        account_state = {
            "account_value": ACCOUNT_100K,
            "open_positions": [],
            "daily_pnl_pct": -0.09,  # -9%, below -8% limit
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
            "circuit_breaker": False,
        }
        approved, reason = gate.check(alert, account_state)
        assert approved is False
        assert "Daily" in reason or "daily" in reason


# ══════════════════════════════════════════════════════════════════════════════
# E. Event gate → composite scaling → DB persistence pipeline
# ══════════════════════════════════════════════════════════════════════════════

class TestEventGatePipeline:
    """End-to-end: events → composite scaling → DB write."""

    def test_fomc_day_pipeline(self, tmp_path):
        """On FOMC day: get events → compute scaling → persist to DB."""
        db_path = mock_macro_db(tmp_path)
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]

        # Step 1: get events
        events = get_upcoming_events(as_of=fomc_day, horizon_days=5)
        fomc_events = [e for e in events if e["event_type"] == "FOMC"]
        assert len(fomc_events) >= 1

        # Step 2: compute composite scaling
        scaling = compute_composite_scaling(events)
        assert scaling <= 1.0
        assert scaling > 0.0

        # Step 3: run daily check (persists to DB)
        result_scaling, result_events = run_daily_event_check(
            as_of=fomc_day, db_path=db_path
        )
        assert result_scaling == scaling
        assert len(result_events) > 0

        # Step 4: verify DB state
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT value FROM macro_state WHERE key = 'event_scaling_factor'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert float(row["value"]) == scaling

    def test_scaling_applied_to_sizing(self):
        """Event scaling factor reduces position sizing."""
        base_risk = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=0.0
        )
        assert base_risk == pytest.approx(2000.0)

        # FOMC day scaling = 0.50
        event_scaling = 0.50
        adjusted_risk = base_risk * event_scaling
        assert adjusted_risk == pytest.approx(1000.0)

        contracts_base = get_contract_size(
            trade_dollar_risk=base_risk,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
        )
        contracts_scaled = get_contract_size(
            trade_dollar_risk=adjusted_risk,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
        )
        assert contracts_scaled < contracts_base


# ══════════════════════════════════════════════════════════════════════════════
# F. Full pipeline: Regime → Event → Sizing → RiskGate
# ══════════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end simulation of the full COMPASS decision pipeline."""

    def test_full_flow_bull_market_fomc_day(self, tmp_path):
        """
        Scenario: Bull market on FOMC day
        - RegimeClassifier → BULL (VIX=16, uptrend)
        - Event gate → 0.50 scaling (FOMC day)
        - Sizing → base $2K × regime 1.0 × event 0.50 = $1K
        - RiskGate → approved (clean account)
        """
        # 1. Classify regime
        clf = RegimeClassifier()
        prices = mock_spy_prices(days=100, trend=25.0, base=450.0)
        regime = clf.classify(vix=16.0, spy_prices=prices, date=prices.index[-1])
        assert regime == Regime.BULL

        # 2. Get event scaling
        fomc_day = KNOWN_FOMC_DATES["2026_jan"]
        events = get_upcoming_events(as_of=fomc_day, horizon_days=5)
        event_scaling = compute_composite_scaling(events)
        assert event_scaling <= 0.50 + 0.01  # FOMC day

        # 3. Calculate sizing
        regime_scale = 1.0  # BULL
        base_risk = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=0.0
        )
        final_risk = base_risk * regime_scale * event_scaling
        contracts = get_contract_size(
            trade_dollar_risk=final_risk,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
        )
        assert contracts >= 1

        # 4. RiskGate check
        from alerts.alert_schema import Alert, AlertType, Direction, Leg
        risk_pct = final_risk / ACCOUNT_100K
        alert = Alert(
            type=AlertType.credit_spread,
            ticker="SPY",
            direction=Direction.bullish,
            legs=[Leg(strike=440.0, option_type="put", action="sell",
                      expiration="2026-04-17")],
            entry_price=0.65,
            stop_loss=2.50,
            profit_target=0.10,
            risk_pct=risk_pct,
        )
        gate = RiskGate()
        approved, reason = gate.check(alert, {
            "account_value": ACCOUNT_100K,
            "open_positions": [],
            "daily_pnl_pct": 0.0,
            "weekly_pnl_pct": 0.0,
            "recent_stops": [],
            "circuit_breaker": False,
        })
        assert approved is True
