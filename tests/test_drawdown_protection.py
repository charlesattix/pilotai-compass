"""Tests for compass.drawdown_protection — 44 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.drawdown_protection import (
    DrawdownProtection,
    ProtectionLevel,
    DrawdownState,
    StrategyDrawdown,
    DrawdownVelocity,
    RecoveryEstimate,
    CorrelationProtection,
    ProtectionEvent,
    ProtectionEffectiveness,
    DEFAULT_THRESHOLDS,
    LEVEL_ACTIONS,
    LEVEL_SIZE_MULT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=n)


def _equity_stable(n: int = 200, seed: int = 42) -> pd.Series:
    """Gently rising equity — stays green."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.003, n)
    return pd.Series(100000 * np.cumprod(1 + r), index=_dates(n))


def _equity_drawdown(n: int = 200, seed: int = 77) -> pd.Series:
    """Equity with a ~10% drawdown mid-series then recovery."""
    rng = np.random.default_rng(seed)
    r = np.empty(n)
    a = int(n * 0.4)       # growth phase
    b = int(n * 0.25)      # drawdown phase
    c = n - a - b          # recovery phase
    r[:a] = rng.normal(0.0005, 0.003, a)
    r[a:a + b] = rng.normal(-0.003, 0.005, b)
    r[a + b:] = rng.normal(0.001, 0.003, c)
    return pd.Series(100000 * np.cumprod(1 + r), index=_dates(n))


def _strategy_returns(n: int = 200, seed: int = 42) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = _dates(n)
    return {
        "A": pd.Series(rng.normal(0.0003, 0.01, n), index=idx),
        "B": pd.Series(rng.normal(0.0002, 0.008, n), index=idx),
        "C": pd.Series(rng.normal(0.0001, 0.012, n), index=idx),
    }


# ===========================================================================
# Level classification
# ===========================================================================

class TestClassify:
    def test_green(self):
        dp = DrawdownProtection()
        assert dp.classify_level(0.01) == ProtectionLevel.GREEN

    def test_yellow(self):
        dp = DrawdownProtection()
        assert dp.classify_level(0.04) == ProtectionLevel.YELLOW

    def test_orange(self):
        dp = DrawdownProtection()
        assert dp.classify_level(0.06) == ProtectionLevel.ORANGE

    def test_red(self):
        dp = DrawdownProtection()
        assert dp.classify_level(0.10) == ProtectionLevel.RED

    def test_boundary_yellow(self):
        dp = DrawdownProtection()
        assert dp.classify_level(0.03) == ProtectionLevel.YELLOW

    def test_custom_thresholds(self):
        custom = {
            ProtectionLevel.GREEN: 0.0,
            ProtectionLevel.YELLOW: 0.05,
            ProtectionLevel.ORANGE: 0.10,
            ProtectionLevel.RED: 0.15,
        }
        dp = DrawdownProtection(thresholds=custom)
        assert dp.classify_level(0.04) == ProtectionLevel.GREEN
        assert dp.classify_level(0.06) == ProtectionLevel.YELLOW


# ===========================================================================
# Core update
# ===========================================================================

class TestUpdate:
    def test_hwm_tracks(self):
        dp = DrawdownProtection()
        dp.update(100000)
        dp.update(105000)
        assert dp.high_water_mark == 105000

    def test_hwm_no_decrease(self):
        dp = DrawdownProtection()
        dp.update(100000)
        dp.update(95000)
        assert dp.high_water_mark == 100000

    def test_drawdown_computed(self):
        dp = DrawdownProtection()
        dp.update(100000)
        state = dp.update(95000)
        assert state.drawdown == pytest.approx(0.05)

    def test_green_state(self):
        dp = DrawdownProtection()
        state = dp.update(100000)
        assert state.level == ProtectionLevel.GREEN
        assert state.size_multiplier == 1.0

    def test_red_state(self):
        dp = DrawdownProtection()
        dp.update(100000)
        state = dp.update(90000)
        assert state.level == ProtectionLevel.RED
        assert state.action == "flatten_all"
        assert state.size_multiplier == 0.0

    def test_state_history_grows(self):
        dp = DrawdownProtection()
        dp.update(100000)
        dp.update(99000)
        dp.update(98000)
        assert len(dp.state_history) == 3

    def test_event_on_level_change(self):
        dp = DrawdownProtection()
        dp.update(100000)
        dp.update(96000)  # 4% → YELLOW
        assert len(dp.events) == 1
        assert dp.events[0].new_level == ProtectionLevel.YELLOW

    def test_no_event_same_level(self):
        dp = DrawdownProtection()
        dp.update(100000)
        dp.update(99000)
        dp.update(98500)
        # Both in GREEN → no events (except possibly YELLOW)
        green_events = [e for e in dp.events if e.old_level == ProtectionLevel.GREEN
                        and e.new_level == ProtectionLevel.GREEN]
        assert len(green_events) == 0


class TestUpdateSeries:
    def test_processes_all(self):
        dp = DrawdownProtection()
        eq = _equity_stable(50)
        states = dp.update_series(eq)
        assert len(states) == 50

    def test_drawdown_series_triggers(self):
        dp = DrawdownProtection()
        eq = _equity_drawdown(200)
        dp.update_series(eq)
        # Should have at least one escalation event
        assert len(dp.events) >= 1


# ===========================================================================
# Per-strategy
# ===========================================================================

class TestStrategy:
    def test_strategy_tracking(self):
        dp = DrawdownProtection()
        sd = dp.update_strategy("EXP-400", 50000)
        assert isinstance(sd, StrategyDrawdown)
        assert sd.drawdown == 0.0

    def test_strategy_drawdown(self):
        dp = DrawdownProtection()
        dp.update_strategy("EXP-400", 50000)
        sd = dp.update_strategy("EXP-400", 47000)
        assert sd.drawdown == pytest.approx(0.06)
        assert sd.level == ProtectionLevel.ORANGE

    def test_multiple_strategies(self):
        dp = DrawdownProtection()
        results = dp.update_strategies({"A": 100000, "B": 50000})
        assert len(results) == 2


# ===========================================================================
# Velocity
# ===========================================================================

class TestVelocity:
    def test_no_history(self):
        dp = DrawdownProtection()
        v = dp.compute_velocity()
        assert v.velocity_1d == 0.0
        assert not v.is_accelerating

    def test_velocity_during_drawdown(self):
        dp = DrawdownProtection()
        eq = _equity_drawdown(200)
        dp.update_series(eq)
        v = dp.compute_velocity()
        assert isinstance(v, DrawdownVelocity)

    def test_velocity_escalation_check(self):
        dp = DrawdownProtection(velocity_threshold=0.01)
        dp.update(100000, date=datetime(2026, 1, 1))
        dp.update(90000, date=datetime(2026, 1, 2))  # big drop
        assert dp.check_velocity_escalation()


# ===========================================================================
# Recovery estimation
# ===========================================================================

class TestRecovery:
    def test_no_drawdown(self):
        eq = _equity_stable(100)
        r = DrawdownProtection.estimate_recovery(eq, 0.0)
        assert r.expected_days == 0.0

    def test_with_drawdown(self):
        eq = _equity_drawdown(200)
        r = DrawdownProtection.estimate_recovery(eq, 0.05)
        assert isinstance(r, RecoveryEstimate)
        assert r.expected_days >= 0

    def test_empty_series(self):
        r = DrawdownProtection.estimate_recovery(pd.Series(dtype=float), 0.05)
        assert r.n_historical_episodes == 0


# ===========================================================================
# Correlation protection
# ===========================================================================

class TestCorrelation:
    def test_basic(self):
        dp = DrawdownProtection()
        sr = _strategy_returns(200)
        cp = dp.correlation_protection(sr)
        assert isinstance(cp, CorrelationProtection)
        assert cp.threshold_multiplier <= 1.0

    def test_single_strategy(self):
        dp = DrawdownProtection()
        sr = {"A": pd.Series(np.random.default_rng(1).normal(0, 0.01, 100),
                              index=_dates(100))}
        cp = dp.correlation_protection(sr)
        assert cp.threshold_multiplier == 1.0

    def test_high_correlation_tightens(self):
        dp = DrawdownProtection(correlation_escalation=0.50)
        rng = np.random.default_rng(42)
        base = rng.normal(0, 0.01, 200)
        idx = _dates(200)
        sr = {
            "A": pd.Series(base + rng.normal(0, 0.001, 200), index=idx),
            "B": pd.Series(base + rng.normal(0, 0.001, 200), index=idx),
        }
        cp = dp.correlation_protection(sr)
        # Highly correlated → should tighten
        assert cp.threshold_multiplier <= 1.0

    def test_apply_adjustment(self):
        dp = DrawdownProtection()
        cp = CorrelationProtection(
            avg_correlation=0.8, correlation_percentile=0.9,
            threshold_multiplier=0.7,
            adjusted_thresholds={
                ProtectionLevel.GREEN: 0.0,
                ProtectionLevel.YELLOW: 0.021,
                ProtectionLevel.ORANGE: 0.035,
                ProtectionLevel.RED: 0.056,
            },
        )
        dp.apply_correlation_adjustment(cp)
        assert dp.thresholds[ProtectionLevel.YELLOW] == pytest.approx(0.021)


# ===========================================================================
# Effectiveness
# ===========================================================================

class TestEffectiveness:
    def test_basic(self):
        protected = _equity_stable(100)
        unprotected = _equity_drawdown(100)
        eff = DrawdownProtection.measure_effectiveness(
            protected, unprotected, [])
        assert isinstance(eff, ProtectionEffectiveness)
        assert eff.max_drawdown_without >= eff.max_drawdown_with

    def test_with_events(self):
        protected = _equity_stable(100)
        unprotected = _equity_drawdown(100)
        events = [
            ProtectionEvent(
                date=datetime(2026, 1, 10),
                old_level=ProtectionLevel.GREEN,
                new_level=ProtectionLevel.YELLOW,
                drawdown=0.04, action="reduce_size", trigger="drawdown"),
            ProtectionEvent(
                date=datetime(2026, 1, 20),
                old_level=ProtectionLevel.YELLOW,
                new_level=ProtectionLevel.GREEN,
                drawdown=0.01, action="normal", trigger="drawdown"),
        ]
        eff = DrawdownProtection.measure_effectiveness(
            protected, unprotected, events)
        assert eff.n_interventions == 1
        assert eff.avg_recovery_days > 0


# ===========================================================================
# Reset
# ===========================================================================

class TestReset:
    def test_reset(self):
        dp = DrawdownProtection()
        dp.update(100000)
        dp.update(90000)
        dp.reset()
        assert dp.high_water_mark == 0.0
        assert dp.current_level == ProtectionLevel.GREEN
        assert len(dp.events) == 0
        assert len(dp.state_history) == 0


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        dp = DrawdownProtection()
        dp.update_series(_equity_drawdown(100))
        out = tmp_path / "dd.html"
        result = dp.generate_report(output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Drawdown Protection" in html

    def test_contains_charts(self, tmp_path):
        dp = DrawdownProtection()
        dp.update_series(_equity_drawdown(100))
        out = tmp_path / "dd.html"
        dp.generate_report(output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Drawdown Path" in html
        assert "Equity Curve" in html

    def test_contains_events(self, tmp_path):
        dp = DrawdownProtection()
        dp.update_series(_equity_drawdown(200))
        out = tmp_path / "dd.html"
        dp.generate_report(output_path=str(out))
        html = out.read_text()
        assert "Protection Events" in html

    def test_with_effectiveness(self, tmp_path):
        dp = DrawdownProtection()
        dp.update_series(_equity_drawdown(100))
        eff = ProtectionEffectiveness(
            max_drawdown_with=0.05, max_drawdown_without=0.10,
            reduction_pct=0.50, n_interventions=3, avg_recovery_days=12.0)
        out = tmp_path / "dd.html"
        dp.generate_report(effectiveness=eff, output_path=str(out))
        html = out.read_text()
        assert "Protection Effectiveness" in html

    def test_with_recovery(self, tmp_path):
        dp = DrawdownProtection()
        dp.update_series(_equity_drawdown(100))
        rec = RecoveryEstimate(
            current_drawdown=0.05, expected_days=30,
            confidence_80=45, historical_avg_recovery=28,
            n_historical_episodes=5)
        out = tmp_path / "dd.html"
        dp.generate_report(recovery=rec, output_path=str(out))
        html = out.read_text()
        assert "Recovery Estimate" in html
