"""Tests for compass.regime_hedge — 38 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.regime_hedge import (
    RegimeHedgeEngine,
    HedgeParams,
    HedgeState,
    RegimeTransition,
    HedgeCostSummary,
    BacktestComparison,
    GridSweepResult,
    DEFAULT_PROFILES,
)
from compass.regime import Regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int = 300) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=n)


def _regimes(n: int = 300) -> pd.Series:
    labels = []
    for i in range(n):
        if i < n * 0.35:
            labels.append(Regime.BULL)
        elif i < n * 0.55:
            labels.append(Regime.HIGH_VOL)
        elif i < n * 0.75:
            labels.append(Regime.BEAR)
        else:
            labels.append(Regime.BULL)
    return pd.Series(labels, index=_dates(n))


def _returns(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.01, n), index=_dates(n))


# ===========================================================================
# Hedge parameter profiles
# ===========================================================================

class TestProfiles:
    def test_all_regimes_have_profile(self):
        for r in Regime:
            assert r in DEFAULT_PROFILES

    def test_crash_highest_hedge(self):
        assert DEFAULT_PROFILES[Regime.CRASH].hedge_ratio > DEFAULT_PROFILES[Regime.BULL].hedge_ratio

    def test_bull_lowest_cost(self):
        assert DEFAULT_PROFILES[Regime.BULL].cost_budget <= DEFAULT_PROFILES[Regime.CRASH].cost_budget

    def test_stop_mult_decreases_with_risk(self):
        assert DEFAULT_PROFILES[Regime.BULL].stop_multiplier > DEFAULT_PROFILES[Regime.CRASH].stop_multiplier


# ===========================================================================
# Blending
# ===========================================================================

class TestBlending:
    def test_alpha_zero(self):
        old = HedgeParams(hedge_ratio=0.10)
        new = HedgeParams(hedge_ratio=0.40)
        blended = RegimeHedgeEngine.blend_params(old, new, 0.0)
        assert blended.hedge_ratio == pytest.approx(0.10)

    def test_alpha_one(self):
        old = HedgeParams(hedge_ratio=0.10)
        new = HedgeParams(hedge_ratio=0.40)
        blended = RegimeHedgeEngine.blend_params(old, new, 1.0)
        assert blended.hedge_ratio == pytest.approx(0.40)

    def test_alpha_half(self):
        old = HedgeParams(hedge_ratio=0.10, stop_multiplier=3.0)
        new = HedgeParams(hedge_ratio=0.40, stop_multiplier=1.0)
        blended = RegimeHedgeEngine.blend_params(old, new, 0.5)
        assert blended.hedge_ratio == pytest.approx(0.25)
        assert blended.stop_multiplier == pytest.approx(2.0)

    def test_exponential_alpha(self):
        eng = RegimeHedgeEngine(transition_halflife=5)
        a0 = eng._compute_alpha(0)
        a5 = eng._compute_alpha(5)
        assert a0 == pytest.approx(0.0)
        assert a5 == pytest.approx(0.5, abs=0.05)

    def test_alpha_converges(self):
        eng = RegimeHedgeEngine(transition_halflife=5)
        a50 = eng._compute_alpha(50)
        assert a50 > 0.99


# ===========================================================================
# Update
# ===========================================================================

class TestUpdate:
    def test_first_update(self):
        eng = RegimeHedgeEngine()
        state = eng.update(Regime.BULL, date=datetime(2026, 1, 1))
        assert isinstance(state, HedgeState)
        assert state.regime == Regime.BULL
        assert state.params.hedge_ratio == DEFAULT_PROFILES[Regime.BULL].hedge_ratio

    def test_transition_recorded(self):
        eng = RegimeHedgeEngine()
        eng.update(Regime.BULL, date=datetime(2026, 1, 1))
        eng.update(Regime.BEAR, date=datetime(2026, 1, 15))
        assert len(eng.transitions) == 1
        assert eng.transitions[0].from_regime == Regime.BULL
        assert eng.transitions[0].to_regime == Regime.BEAR

    def test_blending_during_transition(self):
        eng = RegimeHedgeEngine(transition_halflife=10)
        eng.update(Regime.BULL, date=datetime(2026, 1, 1))
        state = eng.update(Regime.CRASH, date=datetime(2026, 1, 2))
        # Day 0 of transition: should be blended
        assert state.blended
        assert state.blend_alpha < 0.5

    def test_no_blend_same_regime(self):
        eng = RegimeHedgeEngine()
        eng.update(Regime.BULL, date=datetime(2026, 1, 1))
        state = eng.update(Regime.BULL, date=datetime(2026, 1, 2))
        assert state.regime == Regime.BULL

    def test_daily_cost_positive(self):
        eng = RegimeHedgeEngine()
        state = eng.update(Regime.HIGH_VOL)
        assert state.daily_cost > 0

    def test_state_history_grows(self):
        eng = RegimeHedgeEngine()
        eng.update(Regime.BULL)
        eng.update(Regime.BULL)
        eng.update(Regime.BEAR)
        assert len(eng.state_history) == 3


class TestUpdateSeries:
    def test_processes_all(self):
        eng = RegimeHedgeEngine()
        states = eng.update_series(_regimes(100))
        assert len(states) == 100

    def test_transitions_detected(self):
        eng = RegimeHedgeEngine()
        eng.update_series(_regimes(300))
        assert len(eng.transitions) >= 2  # at least BULL→HIGH_VOL→BEAR


# ===========================================================================
# Cost tracking
# ===========================================================================

class TestCostTracking:
    def test_basic(self):
        eng = RegimeHedgeEngine()
        eng.update_series(_regimes(200), _returns(200))
        costs = eng.cost_by_regime(_returns(200))
        assert len(costs) >= 2
        assert all(isinstance(c, HedgeCostSummary) for c in costs)

    def test_total_cost_positive(self):
        eng = RegimeHedgeEngine()
        eng.update_series(_regimes(100))
        costs = eng.cost_by_regime()
        for c in costs:
            assert c.total_cost >= 0

    def test_crash_costlier_than_bull(self):
        eng = RegimeHedgeEngine()
        # All crash vs all bull
        n = 50
        crash_reg = pd.Series([Regime.CRASH] * n, index=_dates(n))
        bull_reg = pd.Series([Regime.BULL] * n, index=_dates(n))

        eng.update_series(crash_reg)
        crash_costs = eng.cost_by_regime()
        eng.reset()
        eng.update_series(bull_reg)
        bull_costs = eng.cost_by_regime()

        c_cost = crash_costs[0].total_cost if crash_costs else 0
        b_cost = bull_costs[0].total_cost if bull_costs else 0
        assert c_cost > b_cost


# ===========================================================================
# Backtest comparison
# ===========================================================================

class TestBacktest:
    def test_basic(self):
        eng = RegimeHedgeEngine()
        comp = eng.backtest(_regimes(200), _returns(200))
        assert isinstance(comp, BacktestComparison)

    def test_sharpe_computed(self):
        eng = RegimeHedgeEngine()
        comp = eng.backtest(_regimes(200), _returns(200))
        assert isinstance(comp.adaptive_sharpe, float)
        assert isinstance(comp.static_sharpe, float)

    def test_cost_tracked(self):
        eng = RegimeHedgeEngine()
        comp = eng.backtest(_regimes(200), _returns(200))
        assert comp.adaptive_cost > 0
        assert comp.static_cost > 0

    def test_improvement_computed(self):
        eng = RegimeHedgeEngine()
        comp = eng.backtest(_regimes(200), _returns(200))
        assert comp.improvement_pnl == pytest.approx(
            comp.adaptive_pnl - comp.static_pnl)


# ===========================================================================
# Grid sweep
# ===========================================================================

class TestGridSweep:
    def test_basic(self):
        eng = RegimeHedgeEngine()
        sweep = eng.grid_sweep(
            Regime.BEAR, _regimes(200), _returns(200),
            hedge_ratios=[0.10, 0.20], stop_mults=[1.5, 2.0])
        assert isinstance(sweep, GridSweepResult)
        assert sweep.regime == "bear"
        assert len(sweep.all_results) == 4  # 2×2

    def test_best_sharpe_populated(self):
        eng = RegimeHedgeEngine()
        sweep = eng.grid_sweep(
            Regime.HIGH_VOL, _regimes(200), _returns(200),
            hedge_ratios=[0.10, 0.30], stop_mults=[1.0, 2.0])
        assert isinstance(sweep.best_sharpe, float)
        assert sweep.best_hedge_ratio in [0.10, 0.30]


# ===========================================================================
# Reset
# ===========================================================================

class TestReset:
    def test_clears_state(self):
        eng = RegimeHedgeEngine()
        eng.update_series(_regimes(50))
        eng.reset()
        assert len(eng.state_history) == 0
        assert len(eng.transitions) == 0


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        eng = RegimeHedgeEngine()
        comp = eng.backtest(_regimes(200), _returns(200))
        out = tmp_path / "hedge.html"
        result = eng.generate_report(comparison=comp, output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Regime-Adaptive Hedge" in html

    def test_contains_charts(self, tmp_path):
        eng = RegimeHedgeEngine()
        eng.update_series(_regimes(200), _returns(200))
        out = tmp_path / "h.html"
        eng.generate_report(output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Hedge Ratio" in html

    def test_contains_regime_bar(self, tmp_path):
        eng = RegimeHedgeEngine()
        eng.update_series(_regimes(100))
        out = tmp_path / "h.html"
        eng.generate_report(output_path=str(out))
        html = out.read_text()
        assert "Regime Timeline" in html

    def test_contains_cost_table(self, tmp_path):
        eng = RegimeHedgeEngine()
        eng.update_series(_regimes(200), _returns(200))
        out = tmp_path / "h.html"
        eng.generate_report(returns=_returns(200), output_path=str(out))
        html = out.read_text()
        assert "Cost by Regime" in html

    def test_contains_comparison(self, tmp_path):
        eng = RegimeHedgeEngine()
        comp = eng.backtest(_regimes(200), _returns(200))
        out = tmp_path / "h.html"
        eng.generate_report(comparison=comp, output_path=str(out))
        html = out.read_text()
        assert "Adaptive vs Static" in html

    def test_with_sweep(self, tmp_path):
        eng = RegimeHedgeEngine()
        eng.update_series(_regimes(200), _returns(200))
        sweep = eng.grid_sweep(Regime.BEAR, _regimes(200), _returns(200),
                                hedge_ratios=[0.10, 0.20], stop_mults=[1.5])
        out = tmp_path / "h.html"
        eng.generate_report(sweep=sweep, output_path=str(out))
        html = out.read_text()
        assert "Grid Sweep" in html
