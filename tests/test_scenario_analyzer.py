"""Tests for compass.scenario_analyzer – what-if scenario analysis engine."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.scenario_analyzer import (
    HISTORICAL_SCENARIOS,
    SPREAD_BETA,
    AnalysisResult,
    GreeksPnL,
    PositionContribution,
    ScenarioAnalyzer,
    ScenarioDef,
    ScenarioResult,
    _build_shock_path,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _custom_scenario(shock: float = -0.20, days: int = 10) -> ScenarioDef:
    return ScenarioDef(
        name="Custom",
        description=f"Custom {shock:.0%} over {days}d",
        daily_shocks=_build_shock_path(shock, days, seed=99),
        vix_start=18.0,
        vix_peak=45.0,
        probability=0.10,
    )


def _positions() -> dict:
    return {"SPY_PUT": 0.6, "QQQ_PUT": 0.3, "IWM_PUT": 0.1}


# ── Shock path generation ──────────────────────────────────────────────────
class TestBuildShockPath:
    def test_length_matches_days(self):
        path = _build_shock_path(-0.20, 10)
        assert len(path) == 10

    def test_single_day(self):
        path = _build_shock_path(-0.10, 1)
        assert len(path) == 1
        assert path[0] == pytest.approx(-0.10, abs=0.001)

    def test_zero_days_empty(self):
        assert _build_shock_path(-0.10, 0) == []

    def test_total_return_approximately_matches(self):
        total = -0.30
        path = _build_shock_path(total, 20)
        realised = float(np.prod([1 + s for s in path]) - 1)
        assert realised == pytest.approx(total, abs=0.02)

    def test_front_loaded_losses(self):
        """First half should have larger losses than second half."""
        path = _build_shock_path(-0.30, 20)
        first_half = sum(path[:10])
        second_half = sum(path[10:])
        assert first_half < second_half  # more negative = bigger loss


# ── Historical scenarios ────────────────────────────────────────────────────
class TestHistoricalScenarios:
    def test_five_built_in(self):
        assert len(HISTORICAL_SCENARIOS) == 5

    def test_all_have_required_keys(self):
        for s in HISTORICAL_SCENARIOS:
            assert "name" in s
            assert "daily_shocks" in s
            assert "vix_start" in s
            assert "vix_peak" in s
            assert "probability" in s

    def test_shocks_are_negative(self):
        for s in HISTORICAL_SCENARIOS:
            total = float(np.prod([1 + x for x in s["daily_shocks"]]) - 1)
            assert total < 0


# ── Constructor ─────────────────────────────────────────────────────────────
class TestScenarioAnalyzerInit:
    def test_defaults(self):
        sa = ScenarioAnalyzer()
        assert sa.starting_capital == 100_000
        assert sa.spread_beta == SPREAD_BETA

    def test_custom_capital(self):
        sa = ScenarioAnalyzer(starting_capital=500_000)
        assert sa.starting_capital == 500_000


# ── Core analysis ───────────────────────────────────────────────────────────
class TestAnalyze:
    def test_returns_result(self):
        result = ScenarioAnalyzer().analyze()
        assert isinstance(result, AnalysisResult)

    def test_historical_scenarios_included(self):
        result = ScenarioAnalyzer().analyze(include_historical=True)
        assert len(result.scenarios) == 5

    def test_historical_excluded(self):
        result = ScenarioAnalyzer().analyze(
            scenarios=[_custom_scenario()], include_historical=False,
        )
        assert len(result.scenarios) == 1

    def test_custom_plus_historical(self):
        result = ScenarioAnalyzer().analyze(scenarios=[_custom_scenario()])
        assert len(result.scenarios) == 6

    def test_worst_case_identified(self):
        result = ScenarioAnalyzer().analyze()
        assert result.worst_case is not None
        # 2008 GFC should be worst (-57% * 1.5 beta)
        assert result.worst_case.scenario.name == "2008 GFC"

    def test_generated_at_set(self):
        result = ScenarioAnalyzer().analyze()
        assert len(result.generated_at) > 0

    def test_empty_scenarios(self):
        result = ScenarioAnalyzer().analyze(scenarios=[], include_historical=False)
        assert result.scenarios == []


# ── Scenario results ────────────────────────────────────────────────────────
class TestScenarioResult:
    def test_drawdown_negative(self):
        result = ScenarioAnalyzer().analyze()
        for sr in result.scenarios:
            assert sr.portfolio_drawdown_pct < 0
            assert sr.adjusted_drawdown_pct < 0

    def test_adjusted_dd_larger_than_raw(self):
        """Beta-adjusted DD should be more severe for beta > 1."""
        result = ScenarioAnalyzer().analyze()
        for sr in result.scenarios:
            assert abs(sr.adjusted_drawdown_pct) >= abs(sr.portfolio_drawdown_pct)

    def test_trough_value_below_starting(self):
        result = ScenarioAnalyzer().analyze()
        for sr in result.scenarios:
            assert sr.trough_value < 100_000

    def test_recovery_days_positive(self):
        result = ScenarioAnalyzer().analyze()
        for sr in result.scenarios:
            assert sr.estimated_recovery_days >= 0

    def test_vix_multiplier_positive(self):
        result = ScenarioAnalyzer().analyze()
        for sr in result.scenarios:
            assert sr.vix_multiplier > 0

    def test_equity_path_starts_at_capital(self):
        result = ScenarioAnalyzer().analyze()
        for sr in result.scenarios:
            if sr.equity_path:
                assert sr.equity_path[0] == pytest.approx(100_000)

    def test_probability_weighted_loss(self):
        result = ScenarioAnalyzer().analyze()
        for sr in result.scenarios:
            assert sr.probability_weighted_loss >= 0


# ── Greeks P&L ──────────────────────────────────────────────────────────────
class TestGreeksPnL:
    def test_greeks_computed_when_provided(self):
        result = ScenarioAnalyzer().analyze(
            portfolio_delta=-50, portfolio_vega=200,
            portfolio_theta=30, portfolio_gamma=-5,
        )
        for sr in result.scenarios:
            assert sr.greeks_pnl is not None
            assert sr.greeks_pnl.total_pnl != 0

    def test_delta_pnl_direction(self):
        """Short delta + market drop → positive delta P&L."""
        result = ScenarioAnalyzer().analyze(portfolio_delta=-50)
        sr = result.scenarios[0]  # 2008 GFC (large negative move)
        assert sr.greeks_pnl.delta_pnl > 0  # short delta profits from drop

    def test_vega_pnl_on_vix_spike(self):
        """Long vega + VIX spike → positive vega P&L."""
        result = ScenarioAnalyzer().analyze(portfolio_vega=100)
        sr = next(s for s in result.scenarios if s.scenario.name == "VIX Spike")
        assert sr.greeks_pnl.vega_pnl > 0

    def test_greeks_total_is_sum(self):
        result = ScenarioAnalyzer().analyze(
            portfolio_delta=-10, portfolio_vega=50,
            portfolio_theta=5, portfolio_gamma=-2,
        )
        for sr in result.scenarios:
            g = sr.greeks_pnl
            expected = g.delta_pnl + g.gamma_pnl + g.vega_pnl + g.theta_pnl
            assert g.total_pnl == pytest.approx(expected, abs=0.01)


# ── Position contributions ──────────────────────────────────────────────────
class TestPositionContributions:
    def test_contributions_present(self):
        result = ScenarioAnalyzer().analyze(positions=_positions())
        for sr in result.scenarios:
            assert len(sr.position_contributions) == 3

    def test_contributions_sum_weights(self):
        result = ScenarioAnalyzer().analyze(positions=_positions())
        for sr in result.scenarios:
            total_w = sum(c.weight for c in sr.position_contributions)
            assert total_w == pytest.approx(1.0)

    def test_largest_weight_largest_contribution(self):
        result = ScenarioAnalyzer().analyze(positions=_positions())
        sr = result.scenarios[0]
        by_pct = sorted(sr.position_contributions, key=lambda c: -c.pct_of_total)
        assert by_pct[0].position_id == "SPY_PUT"

    def test_no_positions_no_contribs(self):
        result = ScenarioAnalyzer().analyze()
        for sr in result.scenarios:
            assert sr.position_contributions == []


# ── Probability weighting ──────────────────────────────────────────────────
class TestProbabilityWeighting:
    def test_pw_loss_positive(self):
        result = ScenarioAnalyzer().analyze()
        assert result.probability_weighted_loss > 0

    def test_expected_shortfall_positive(self):
        result = ScenarioAnalyzer().analyze()
        assert result.expected_shortfall > 0

    def test_higher_prob_higher_pw_loss(self):
        low_prob = ScenarioDef("Low", "low", _build_shock_path(-0.20, 10), probability=0.01)
        high_prob = ScenarioDef("High", "high", _build_shock_path(-0.20, 10), probability=0.50)
        result = ScenarioAnalyzer().analyze(
            scenarios=[low_prob, high_prob], include_historical=False,
        )
        by_name = {s.scenario.name: s for s in result.scenarios}
        assert by_name["High"].probability_weighted_loss > by_name["Low"].probability_weighted_loss


# ── Custom scenarios ────────────────────────────────────────────────────────
class TestCustomScenario:
    def test_create_custom(self):
        sa = ScenarioAnalyzer()
        s = sa.create_custom_scenario("Test", -0.25, 15)
        assert s.name == "Test"
        assert len(s.daily_shocks) == 15

    def test_custom_with_correlation(self):
        sa = ScenarioAnalyzer()
        s = sa.create_custom_scenario("Corr", -0.20, 10, correlation_shock=0.3)
        assert s.correlation_shock == 0.3


# ── Recovery estimation ─────────────────────────────────────────────────────
class TestRecoveryEstimation:
    def test_deeper_dd_longer_recovery(self):
        mild = ScenarioDef("Mild", "5%", _build_shock_path(-0.05, 5), probability=0.1)
        severe = ScenarioDef("Severe", "40%", _build_shock_path(-0.40, 30), probability=0.1)
        result = ScenarioAnalyzer().analyze(
            scenarios=[mild, severe], include_historical=False,
        )
        by_name = {s.scenario.name: s for s in result.scenarios}
        assert by_name["Severe"].estimated_recovery_days > by_name["Mild"].estimated_recovery_days


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            sa = ScenarioAnalyzer()
            result = sa.analyze(
                positions=_positions(),
                portfolio_delta=-20, portfolio_vega=100,
            )
            path = sa.generate_report(result, output_path=Path(tmp) / "sa.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            sa = ScenarioAnalyzer()
            result = sa.analyze(
                positions=_positions(),
                portfolio_delta=-20, portfolio_vega=100,
            )
            path = sa.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Scenario Analysis" in html
            assert "Scenario Comparison" in html
            assert "Worst-Case" in html
            assert "Greeks" in html
            assert "P&L Impact" in html
            assert "Position" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            sa = ScenarioAnalyzer()
            result = sa.analyze()
            path = sa.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html

    def test_report_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            sa = ScenarioAnalyzer()
            result = AnalysisResult(generated_at="2024-01-01T00:00:00+00:00")
            path = sa.generate_report(result, output_path=Path(tmp) / "e.html")
            assert path.exists()


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_scenario_def(self):
        s = ScenarioDef("X", "desc", [-0.01, -0.02], probability=0.05)
        assert s.name == "X"
        assert len(s.daily_shocks) == 2

    def test_greeks_pnl(self):
        g = GreeksPnL(delta_pnl=-500, vega_pnl=200, theta_pnl=50,
                      gamma_pnl=-100, total_pnl=-350)
        assert g.total_pnl == -350

    def test_position_contribution(self):
        pc = PositionContribution("SPY", 0.5, 0.10, 0.08, 0.5)
        assert pc.weight == 0.5

    def test_scenario_result_fields(self):
        s = ScenarioDef("T", "t", [-0.01])
        sr = ScenarioResult(
            scenario=s, portfolio_drawdown_pct=-0.01,
            adjusted_drawdown_pct=-0.015, trough_value=98500,
            peak_to_trough_days=1, estimated_recovery_days=10,
            vix_multiplier=2.0,
        )
        assert sr.adjusted_drawdown_pct == -0.015

    def test_analysis_result_defaults(self):
        r = AnalysisResult()
        assert r.scenarios == []
        assert r.worst_case is None
