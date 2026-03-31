"""Tests for compass.stress_scenario – advanced stress scenario engine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.stress_scenario import (
    PREDEFINED_SCENARIOS,
    SPREAD_BETA,
    AssetStress,
    GreeksStressPnL,
    RecoveryPath,
    ScenarioDef,
    ScenarioOutcome,
    StressResult,
    StressScenarioEngine,
    _build_shock_path,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _engine() -> StressScenarioEngine:
    return StressScenarioEngine(recovery_simulations=50, random_state=42)


def _weights() -> dict:
    return {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2}


def _custom() -> ScenarioDef:
    return ScenarioDef(
        name="Custom Shock", description="Test -20% 10d",
        daily_shocks=_build_shock_path(-0.20, 10, seed=99),
        vix_start=18.0, vix_peak=50.0, probability=0.10,
        asset_correlations={"SPY": 1.0, "QQQ": 0.9, "IWM": 0.8},
    )


# ── Shock path ──────────────────────────────────────────────────────────────
class TestShockPath:
    def test_length(self):
        assert len(_build_shock_path(-0.20, 10)) == 10

    def test_single_day(self):
        p = _build_shock_path(-0.10, 1)
        assert p[0] == pytest.approx(-0.10, abs=0.001)

    def test_empty(self):
        assert _build_shock_path(-0.10, 0) == []

    def test_total_return(self):
        p = _build_shock_path(-0.30, 20)
        total = float(np.prod([1 + s for s in p]) - 1)
        assert total == pytest.approx(-0.30, abs=0.02)

    def test_front_loaded(self):
        p = _build_shock_path(-0.30, 20)
        assert sum(p[:10]) < sum(p[10:])


# ── Predefined scenarios ───────────────────────────────────────────────────
class TestPredefined:
    def test_six_predefined(self):
        assert len(PREDEFINED_SCENARIOS) == 6

    def test_names(self):
        names = {s["name"] for s in PREDEFINED_SCENARIOS}
        assert "2008 GFC" in names
        assert "COVID Crash (2020)" in names
        assert "2022 Rate Hikes" in names
        assert "Flash Crash" in names
        assert "Brexit (2016)" in names
        assert "Volmageddon (2018)" in names

    def test_all_have_correlations(self):
        for s in PREDEFINED_SCENARIOS:
            assert "asset_correlations" in s
            assert "SPY" in s["asset_correlations"]

    def test_all_negative_shocks(self):
        for s in PREDEFINED_SCENARIOS:
            total = float(np.prod([1 + x for x in s["daily_shocks"]]) - 1)
            assert total < 0


# ── Constructor ─────────────────────────────────────────────────────────────
class TestInit:
    def test_defaults(self):
        e = StressScenarioEngine()
        assert e.starting_capital == 100_000
        assert e.spread_beta == SPREAD_BETA

    def test_custom(self):
        e = StressScenarioEngine(starting_capital=500_000, spread_beta=2.0)
        assert e.starting_capital == 500_000
        assert e.spread_beta == 2.0


# ── Core run ────────────────────────────────────────────────────────────────
class TestRun:
    def test_returns_result(self):
        result = _engine().run()
        assert isinstance(result, StressResult)

    def test_predefined_included(self):
        result = _engine().run(include_predefined=True)
        assert result.n_scenarios == 6

    def test_predefined_excluded(self):
        result = _engine().run(scenarios=[_custom()], include_predefined=False)
        assert result.n_scenarios == 1

    def test_custom_plus_predefined(self):
        result = _engine().run(scenarios=[_custom()])
        assert result.n_scenarios == 7

    def test_worst_case_identified(self):
        result = _engine().run()
        assert result.worst_case is not None
        assert result.worst_case.scenario.name == "2008 GFC"

    def test_empty(self):
        result = _engine().run(scenarios=[], include_predefined=False)
        assert result.n_scenarios == 0

    def test_generated_at(self):
        result = _engine().run()
        assert len(result.generated_at) > 0


# ── Outcomes ────────────────────────────────────────────────────────────────
class TestOutcomes:
    def test_drawdowns_negative(self):
        result = _engine().run()
        for o in result.outcomes:
            assert o.raw_drawdown < 0
            assert o.adjusted_drawdown < 0

    def test_adjusted_more_severe(self):
        result = _engine().run()
        for o in result.outcomes:
            assert abs(o.adjusted_drawdown) >= abs(o.raw_drawdown)

    def test_trough_below_start(self):
        result = _engine().run()
        for o in result.outcomes:
            assert o.trough_value < 100_000

    def test_equity_path_starts_at_capital(self):
        result = _engine().run()
        for o in result.outcomes:
            assert o.equity_path[0] == pytest.approx(100_000)

    def test_vix_multiplier_positive(self):
        result = _engine().run()
        for o in result.outcomes:
            assert o.vix_multiplier > 0

    def test_conditional_var_positive(self):
        result = _engine().run()
        for o in result.outcomes:
            assert o.conditional_var_95 >= 0


# ── Greeks P&L ──────────────────────────────────────────────────────────────
class TestGreeksPnL:
    def test_greeks_computed(self):
        result = _engine().run(portfolio_delta=-30, portfolio_vega=100)
        for o in result.outcomes:
            assert o.greeks_pnl is not None

    def test_total_is_sum(self):
        result = _engine().run(
            portfolio_delta=-20, portfolio_gamma=-3,
            portfolio_vega=80, portfolio_theta=10,
        )
        for o in result.outcomes:
            g = o.greeks_pnl
            assert g.total_pnl == pytest.approx(g.delta_pnl + g.gamma_pnl + g.vega_pnl + g.theta_pnl, abs=0.01)

    def test_short_delta_profits_on_crash(self):
        result = _engine().run(portfolio_delta=-50)
        gfc = next(o for o in result.outcomes if o.scenario.name == "2008 GFC")
        assert gfc.greeks_pnl.delta_pnl > 0


# ── Correlated asset stress ────────────────────────────────────────────────
class TestAssetStress:
    def test_asset_stress_present(self):
        result = _engine().run(asset_weights=_weights())
        for o in result.outcomes:
            assert len(o.asset_stress) > 0

    def test_spy_correlation_one(self):
        result = _engine().run(asset_weights=_weights())
        gfc = next(o for o in result.outcomes if o.scenario.name == "2008 GFC")
        spy = next(a for a in gfc.asset_stress if a.asset == "SPY")
        assert spy.correlation == 1.0

    def test_stress_return_proportional(self):
        result = _engine().run(asset_weights=_weights())
        gfc = next(o for o in result.outcomes if o.scenario.name == "2008 GFC")
        spy = next(a for a in gfc.asset_stress if a.asset == "SPY")
        qqq = next(a for a in gfc.asset_stress if a.asset == "QQQ")
        # QQQ corr 0.95 of SPY → stress_return should be ~0.95× SPY's
        assert abs(qqq.stress_return) < abs(spy.stress_return)

    def test_no_weights_no_stress(self):
        result = _engine().run()
        for o in result.outcomes:
            assert o.asset_stress == []


# ── Recovery simulation ─────────────────────────────────────────────────────
class TestRecovery:
    def test_recovery_present(self):
        result = _engine().run()
        for o in result.outcomes:
            assert o.recovery is not None

    def test_recovery_days_positive(self):
        result = _engine().run()
        for o in result.outcomes:
            assert o.recovery.median_days > 0

    def test_p10_leq_median_leq_p90(self):
        result = _engine().run()
        for o in result.outcomes:
            r = o.recovery
            assert r.p10_days <= r.median_days
            assert r.median_days <= r.p90_days

    def test_recovery_probability_bounded(self):
        result = _engine().run()
        for o in result.outcomes:
            assert 0.0 <= o.recovery.recovery_probability <= 1.0

    def test_deeper_dd_longer_recovery(self):
        result = _engine().run()
        gfc = next(o for o in result.outcomes if o.scenario.name == "2008 GFC")
        brexit = next(o for o in result.outcomes if o.scenario.name == "Brexit (2016)")
        assert gfc.recovery.median_days > brexit.recovery.median_days

    def test_sample_paths_present(self):
        result = _engine().run()
        gfc = next(o for o in result.outcomes if o.scenario.name == "2008 GFC")
        assert len(gfc.recovery.sample_paths) > 0


# ── Probability weighting ──────────────────────────────────────────────────
class TestProbability:
    def test_pw_loss_positive(self):
        result = _engine().run()
        assert result.total_pw_loss > 0

    def test_expected_shortfall_positive(self):
        result = _engine().run()
        assert result.expected_shortfall > 0


# ── Custom scenario ────────────────────────────────────────────────────────
class TestCustomScenario:
    def test_create(self):
        s = _engine().create_scenario("Test", -0.15, 5)
        assert s.name == "Test"
        assert len(s.daily_shocks) == 5

    def test_with_correlations(self):
        s = _engine().create_scenario("C", -0.10, 3, asset_correlations={"SPY": 1.0})
        assert s.asset_correlations["SPY"] == 1.0


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = _engine()
            result = e.run(asset_weights=_weights(), portfolio_delta=-20, portfolio_vega=50)
            path = e.generate_report(result, output_path=Path(tmp) / "ss.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = _engine()
            result = e.run(asset_weights=_weights(), portfolio_delta=-20)
            path = e.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Stress Scenario" in html
            assert "Waterfall" in html
            assert "Comparison" in html
            assert "Worst Case" in html
            assert "Greeks" in html
            assert "Correlated" in html
            assert "Recovery" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = _engine()
            result = e.run()
            path = e.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_scenario_def(self):
        s = ScenarioDef("X", "d", [-0.01])
        assert s.probability == 0.05

    def test_asset_stress(self):
        a = AssetStress("SPY", 1.0, -0.20, -0.10)
        assert a.stress_return == -0.20

    def test_greeks(self):
        g = GreeksStressPnL(delta_pnl=-500, total_pnl=-500)
        assert g.delta_pnl == -500

    def test_recovery_path(self):
        r = RecoveryPath(100, 50, 200, 0.85)
        assert r.median_days == 100

    def test_stress_result_defaults(self):
        r = StressResult()
        assert r.outcomes == []
        assert r.worst_case is None
