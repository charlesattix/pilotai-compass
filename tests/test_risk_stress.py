"""Tests for compass.risk_stress module (35+ tests)."""
from __future__ import annotations

import datetime
import math

import numpy as np
import pytest

from compass.risk_stress import (
    CorrelationStressResult,
    GreeksExposure,
    HistoricalScenario,
    LiquidityStressResult,
    PortfolioSnapshot,
    ReverseStressResult,
    RiskStressEngine,
    RiskStressResult,
    ScenarioResult,
    StressScenario,
    HISTORICAL_SCENARIOS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        market_value=1_000_000,
        maintenance_margin_pct=0.20,
        greeks=GreeksExposure(delta=-0.3, gamma=0.01, vega=500, theta=-50),
        position_notionals=[200_000, 300_000, 500_000],
        position_vols=[0.20, 0.25, 0.30],
        bid_ask_spread=0.05,
    )


@pytest.fixture
def engine(basic_portfolio: PortfolioSnapshot) -> RiskStressEngine:
    return RiskStressEngine(portfolio=basic_portfolio)


@pytest.fixture
def engine_with_custom(basic_portfolio: PortfolioSnapshot) -> RiskStressEngine:
    custom = [
        StressScenario(
            name="Mild_Selloff",
            equity_shock=-0.05,
            vol_shock=5.0,
            rate_shock=25,
            correlation_shock=0.3,
            duration=5,
        ),
        StressScenario(
            name="Severe_Crash",
            equity_shock=-0.40,
            vol_shock=40.0,
            rate_shock=-100,
            correlation_shock=0.9,
            duration=30,
        ),
    ]
    return RiskStressEngine(portfolio=basic_portfolio, custom_scenarios=custom)


# ===========================================================================
# Historical scenario tests
# ===========================================================================

class TestHistoricalScenarios:
    """Tests for historical scenario definitions and replays."""

    def test_gfc_exists(self):
        assert "2008_GFC" in HISTORICAL_SCENARIOS

    def test_covid_exists(self):
        assert "COVID_2020" in HISTORICAL_SCENARIOS

    def test_rate_hikes_exists(self):
        assert "2022_RATE_HIKES" in HISTORICAL_SCENARIOS

    def test_flash_crash_exists(self):
        assert "FLASH_CRASH" in HISTORICAL_SCENARIOS

    def test_gfc_params(self):
        s = HISTORICAL_SCENARIOS["2008_GFC"]
        assert s.total_shock == -0.57
        assert s.duration_days == 350
        assert s.vix_start == 22.0
        assert s.vix_peak == 80.0

    def test_covid_params(self):
        s = HISTORICAL_SCENARIOS["COVID_2020"]
        assert s.total_shock == -0.34
        assert s.duration_days == 23

    def test_rate_hikes_params(self):
        s = HISTORICAL_SCENARIOS["2022_RATE_HIKES"]
        assert s.total_shock == -0.25
        assert s.duration_days == 190

    def test_flash_crash_params(self):
        s = HISTORICAL_SCENARIOS["FLASH_CRASH"]
        assert s.total_shock == -0.10
        assert s.duration_days == 1

    def test_daily_shocks_length_matches_duration(self):
        for name, s in HISTORICAL_SCENARIOS.items():
            assert len(s.daily_shocks) == s.duration_days, (
                f"{name}: expected {s.duration_days} daily shocks, "
                f"got {len(s.daily_shocks)}"
            )

    def test_all_historical_produce_negative_drawdown(self, engine):
        for name, scenario in HISTORICAL_SCENARIOS.items():
            result = engine.run_historical_scenario(scenario)
            assert result.max_drawdown > 0, (
                f"{name} should have positive max drawdown"
            )
            assert result.pnl < 0, f"{name} should produce negative P&L"

    def test_gfc_worst_drawdown(self, engine):
        result = engine.run_historical_scenario(HISTORICAL_SCENARIOS["2008_GFC"])
        # GFC: ~57% total shock => drawdown should be substantial
        assert result.max_drawdown > 0.30

    def test_flash_crash_single_day(self, engine):
        result = engine.run_historical_scenario(HISTORICAL_SCENARIOS["FLASH_CRASH"])
        assert len(result.daily_pnls) == 1


# ===========================================================================
# Custom scenario tests
# ===========================================================================

class TestCustomScenarios:
    def test_mild_selloff(self, engine_with_custom):
        results = engine_with_custom.run_all()
        names = [r.scenario_name for r in results.scenario_results]
        assert "Mild_Selloff" in names

    def test_severe_crash(self, engine_with_custom):
        results = engine_with_custom.run_all()
        severe = [r for r in results.scenario_results if r.scenario_name == "Severe_Crash"][0]
        assert severe.pnl < -300_000

    def test_custom_pnl_pct(self, engine_with_custom):
        scenario = StressScenario(
            name="test", equity_shock=-0.20, vol_shock=10.0,
            rate_shock=0, correlation_shock=0, duration=10,
        )
        result = engine_with_custom.run_custom_scenario(scenario)
        assert abs(result.pnl_pct - (-0.20)) < 1e-9

    def test_custom_positive_shock(self, basic_portfolio):
        scenario = StressScenario(
            name="Rally", equity_shock=0.10, vol_shock=-5.0,
            rate_shock=0, correlation_shock=0, duration=5,
        )
        eng = RiskStressEngine(portfolio=basic_portfolio)
        result = eng.run_custom_scenario(scenario)
        assert result.pnl > 0


# ===========================================================================
# Reverse stress test
# ===========================================================================

class TestReverseStress:
    def test_finds_30pct_loss(self, engine):
        rs = engine.reverse_stress_test(target_loss_pct=0.30)
        assert abs(rs.achieved_loss_pct - 0.30) < 0.01

    def test_required_shock_is_negative(self, engine):
        rs = engine.reverse_stress_test(target_loss_pct=0.30)
        assert rs.required_equity_shock < 0

    def test_finds_50pct_loss(self, engine):
        rs = engine.reverse_stress_test(target_loss_pct=0.50)
        assert abs(rs.achieved_loss_pct - 0.50) < 0.01

    def test_small_target(self, engine):
        rs = engine.reverse_stress_test(target_loss_pct=0.05)
        assert abs(rs.achieved_loss_pct - 0.05) < 0.01
        assert rs.required_equity_shock > -0.10  # small shock needed

    def test_iterations_reasonable(self, engine):
        rs = engine.reverse_stress_test(target_loss_pct=0.30)
        assert rs.iterations < 50


# ===========================================================================
# Greeks impact tests
# ===========================================================================

class TestGreeksImpact:
    def test_delta_profits_on_short_delta_crash(self):
        """Short delta position should profit when market crashes."""
        portfolio = PortfolioSnapshot(
            market_value=1_000_000,
            greeks=GreeksExposure(delta=-0.5, gamma=0.0, vega=0.0, theta=0.0),
        )
        eng = RiskStressEngine(portfolio=portfolio)
        greeks_pnl = eng._compute_greeks_impact(
            equity_shock=-0.20, vol_shock=0.0, days=0
        )
        # delta=-0.5, move = -0.20 * 1M = -200k, pnl = -0.5 * -200k = +100k
        assert greeks_pnl > 0

    def test_gamma_contribution(self):
        portfolio = PortfolioSnapshot(
            market_value=1_000_000,
            greeks=GreeksExposure(delta=0.0, gamma=0.01, vega=0.0, theta=0.0),
        )
        eng = RiskStressEngine(portfolio=portfolio)
        move = -0.10 * 1_000_000  # -100k
        expected = 0.5 * 0.01 * move ** 2
        actual = eng._compute_greeks_impact(-0.10, 0.0, 0)
        assert abs(actual - expected) < 1.0

    def test_vega_contribution(self):
        portfolio = PortfolioSnapshot(
            market_value=1_000_000,
            greeks=GreeksExposure(delta=0.0, gamma=0.0, vega=1000, theta=0.0),
        )
        eng = RiskStressEngine(portfolio=portfolio)
        result = eng._compute_greeks_impact(0.0, 20.0, 0)
        assert abs(result - 20_000) < 1.0

    def test_theta_contribution(self):
        portfolio = PortfolioSnapshot(
            market_value=1_000_000,
            greeks=GreeksExposure(delta=0.0, gamma=0.0, vega=0.0, theta=-100),
        )
        eng = RiskStressEngine(portfolio=portfolio)
        result = eng._compute_greeks_impact(0.0, 0.0, 30)
        assert abs(result - (-3000)) < 1.0

    def test_combined_greeks(self, basic_portfolio):
        eng = RiskStressEngine(portfolio=basic_portfolio)
        result = eng._compute_greeks_impact(-0.10, 15.0, 5)
        # Should be a real number, not nan
        assert math.isfinite(result)


# ===========================================================================
# Liquidity stress tests
# ===========================================================================

class TestLiquidityStress:
    def test_basic_liquidity(self, engine):
        ls = engine.compute_liquidity_stress(spread_multiplier=5.0)
        assert ls.unwind_cost > 0
        assert ls.spread_multiplier == 5.0

    def test_scales_with_multiplier(self, engine):
        ls5 = engine.compute_liquidity_stress(spread_multiplier=5.0)
        ls10 = engine.compute_liquidity_stress(spread_multiplier=10.0)
        assert ls10.unwind_cost == pytest.approx(ls5.unwind_cost * 2.0, rel=1e-9)

    def test_unwind_cost_pct(self, engine):
        ls = engine.compute_liquidity_stress(spread_multiplier=5.0)
        expected_pct = (1_000_000 * 0.05 * 5.0 * 0.5) / 1_000_000
        assert ls.unwind_cost_pct == pytest.approx(expected_pct, rel=1e-9)

    def test_multiplier_1x(self, engine):
        ls = engine.compute_liquidity_stress(spread_multiplier=1.0)
        expected = 1_000_000 * 0.05 * 1.0 * 0.5
        assert ls.unwind_cost == pytest.approx(expected, rel=1e-9)


# ===========================================================================
# Correlation stress tests
# ===========================================================================

class TestCorrelationStress:
    def test_corr1_is_worst_case(self, engine):
        cs = engine.compute_correlation_stress()
        assert cs.stressed_portfolio_vol >= cs.normal_portfolio_vol

    def test_vol_ratio_greater_than_1(self, engine):
        cs = engine.compute_correlation_stress()
        assert cs.vol_ratio >= 1.0

    def test_vol_ratio_equals_sqrt_n_for_equal_vols(self):
        """With n equal vols, corr=1 vol / corr=0 vol = sqrt(n)."""
        n = 4
        portfolio = PortfolioSnapshot(
            market_value=1_000_000,
            position_vols=[0.20] * n,
        )
        eng = RiskStressEngine(portfolio=portfolio)
        cs = eng.compute_correlation_stress()
        assert cs.vol_ratio == pytest.approx(math.sqrt(n), rel=1e-6)

    def test_single_position_ratio_is_1(self):
        portfolio = PortfolioSnapshot(
            market_value=500_000,
            position_vols=[0.25],
        )
        eng = RiskStressEngine(portfolio=portfolio)
        cs = eng.compute_correlation_stress()
        assert cs.vol_ratio == pytest.approx(1.0, rel=1e-9)

    def test_empty_vols(self):
        portfolio = PortfolioSnapshot(market_value=500_000, position_vols=[])
        eng = RiskStressEngine(portfolio=portfolio)
        cs = eng.compute_correlation_stress()
        assert cs.vol_ratio == 1.0
        assert cs.normal_portfolio_vol == 0.0


# ===========================================================================
# Margin impact tests
# ===========================================================================

class TestMarginImpact:
    def test_margin_positive(self, engine):
        result = engine.run_historical_scenario(HISTORICAL_SCENARIOS["COVID_2020"])
        assert result.margin_impact > 0

    def test_margin_scales_with_loss(self, basic_portfolio):
        eng = RiskStressEngine(portfolio=basic_portfolio)
        r1 = eng.compute_scenario_pnl(-0.10, 0, 1)
        r2 = eng.compute_scenario_pnl(-0.20, 0, 1)
        assert r2.margin_impact == pytest.approx(r1.margin_impact * 2.0, rel=1e-9)

    def test_margin_pct(self, engine):
        result = engine.compute_scenario_pnl(-0.10, 0, 1)
        expected = abs(-0.10 * 1_000_000) * 0.20
        assert result.margin_impact == pytest.approx(expected, rel=1e-9)


# ===========================================================================
# Worst case identification
# ===========================================================================

class TestWorstCase:
    def test_worst_case_is_gfc(self, engine):
        result = engine.run_all()
        assert result.worst_case is not None
        assert result.worst_case.scenario_name == "2008_GFC"

    def test_worst_case_with_custom(self, engine_with_custom):
        result = engine_with_custom.run_all()
        # GFC is -57%, Severe_Crash is -40%, so GFC should still be worst
        assert result.worst_case is not None
        assert result.worst_case.pnl < 0

    def test_worst_has_most_negative_pnl(self, engine_with_custom):
        result = engine_with_custom.run_all()
        all_pnls = [r.pnl for r in result.scenario_results]
        assert result.worst_case.pnl == min(all_pnls)


# ===========================================================================
# HTML report tests
# ===========================================================================

class TestHTMLReport:
    def test_report_is_html(self, engine):
        html = engine.generate_report()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_report_has_scenario_table(self, engine):
        html = engine.generate_report()
        assert "<table>" in html
        assert "2008_GFC" in html

    def test_report_has_svg_waterfall(self, engine):
        html = engine.generate_report()
        assert "<svg" in html
        assert "</svg>" in html

    def test_report_has_worst_case_dashboard(self, engine):
        html = engine.generate_report()
        assert "Worst-Case Dashboard" in html
        assert "dashboard" in html

    def test_report_has_reverse_stress(self, engine):
        html = engine.generate_report()
        assert "Reverse Stress Test" in html

    def test_report_has_liquidity(self, engine):
        html = engine.generate_report()
        assert "Liquidity Stress" in html

    def test_report_has_correlation(self, engine):
        html = engine.generate_report()
        assert "Correlation Stress" in html

    def test_report_with_custom_result(self, engine):
        result = engine.run_all()
        html = engine.generate_report(result=result)
        assert "Generated:" in html


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    def test_zero_market_value(self):
        portfolio = PortfolioSnapshot(market_value=0.0)
        eng = RiskStressEngine(portfolio=portfolio)
        result = eng.run_all()
        assert result.worst_case is not None

    def test_very_small_portfolio(self):
        portfolio = PortfolioSnapshot(market_value=100.0)
        eng = RiskStressEngine(portfolio=portfolio)
        result = eng.run_all()
        assert len(result.scenario_results) >= 4

    def test_no_custom_scenarios(self, basic_portfolio):
        eng = RiskStressEngine(portfolio=basic_portfolio, custom_scenarios=[])
        result = eng.run_all()
        assert len(result.scenario_results) == 4  # only historical

    def test_no_greeks(self):
        portfolio = PortfolioSnapshot(market_value=1_000_000)
        eng = RiskStressEngine(portfolio=portfolio)
        result = eng.run_all()
        # Should still work with default zero greeks
        assert result.scenario_results[0].greeks_pnl is not None


# ===========================================================================
# Dataclass tests
# ===========================================================================

class TestDataclasses:
    def test_stress_scenario_fields(self):
        s = StressScenario(
            name="test", equity_shock=-0.1, vol_shock=10.0,
            rate_shock=50, correlation_shock=0.5, duration=10,
        )
        assert s.name == "test"
        assert s.equity_shock == -0.1

    def test_greeks_exposure_defaults(self):
        g = GreeksExposure()
        assert g.delta == 0.0
        assert g.gamma == 0.0
        assert g.vega == 0.0
        assert g.theta == 0.0

    def test_portfolio_snapshot_defaults(self):
        p = PortfolioSnapshot(market_value=100_000)
        assert p.maintenance_margin_pct == 0.20
        assert p.bid_ask_spread == 0.05

    def test_scenario_result_daily_pnls_default(self):
        sr = ScenarioResult(
            scenario_name="x", pnl=-100, pnl_pct=-0.01,
            margin_impact=20, greeks_pnl=-50, max_drawdown=0.01,
        )
        assert sr.daily_pnls == []

    def test_risk_stress_result_generated_at(self):
        r = RiskStressResult(
            scenario_results=[], reverse_stress=None,
            liquidity_stress=None, correlation_stress=None,
            worst_case=None,
        )
        # Should be a valid ISO timestamp
        dt = datetime.datetime.fromisoformat(r.generated_at)
        assert dt.year >= 2024

    def test_historical_scenario_custom_daily_shocks(self):
        s = HistoricalScenario(
            name="custom", total_shock=-0.20, duration_days=3,
            vix_start=15.0, vix_peak=30.0,
            daily_shocks=[-0.08, -0.07, -0.06],
        )
        assert len(s.daily_shocks) == 3
        assert s.daily_shocks[0] == -0.08

    def test_reverse_stress_result_fields(self):
        r = ReverseStressResult(
            target_loss_pct=0.30,
            required_equity_shock=-0.30,
            achieved_loss_pct=0.30,
            iterations=15,
        )
        assert r.iterations == 15

    def test_liquidity_stress_result_fields(self):
        ls = LiquidityStressResult(
            spread_multiplier=5.0,
            unwind_cost=12500.0,
            unwind_cost_pct=0.0125,
        )
        assert ls.spread_multiplier == 5.0

    def test_correlation_stress_result_fields(self):
        cs = CorrelationStressResult(
            normal_portfolio_vol=0.10,
            stressed_portfolio_vol=0.20,
            vol_ratio=2.0,
        )
        assert cs.vol_ratio == 2.0
