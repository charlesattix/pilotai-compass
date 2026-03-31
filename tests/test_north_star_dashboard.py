from __future__ import annotations

import pytest
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compass.north_star_dashboard import (
    GREEN,
    YELLOW,
    RED,
    NorthStarDashboard,
    NorthStarTargets,
    NorthStarResult,
    ExperimentStatus,
    GapItem,
    PortfolioMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_experiment(
    sharpe: float = 7.0,
    annual_return: float = 0.60,
    max_dd: float = 0.20,
    win_rate: float = 0.85,
    profit_factor: float = 2.5,
    yearly_returns: list[float] | None = None,
    robustness_score: float = 0.80,
) -> dict:
    return {
        "sharpe": sharpe,
        "annual_return": annual_return,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "yearly_returns": yearly_returns if yearly_returns is not None else [0.10, 0.15, 0.20],
        "robustness_score": robustness_score,
    }


@pytest.fixture
def dashboard() -> NorthStarDashboard:
    return NorthStarDashboard()


@pytest.fixture
def good_experiments() -> dict[str, dict]:
    return {
        "exp_a": _make_experiment(),
        "exp_b": _make_experiment(sharpe=8.0, annual_return=0.70),
    }


# ===========================================================================
# 1. Dataclass defaults
# ===========================================================================
class TestDataclassDefaults:
    def test_targets_defaults(self):
        t = NorthStarTargets()
        assert t.annual_return_target == 0.55
        assert t.sharpe_target == 6.0
        assert t.max_dd_target == 0.30
        assert t.all_years_profitable is True
        assert t.robustness_target == 0.70

    def test_targets_custom(self):
        t = NorthStarTargets(annual_return_target=0.40, sharpe_target=3.0)
        assert t.annual_return_target == 0.40
        assert t.sharpe_target == 3.0

    def test_north_star_result_fields(self):
        r = NorthStarResult(
            experiment_statuses=[],
            portfolio_metrics=PortfolioMetrics(0, 0, 0, 0, 0, 0, False),
            gap_analysis=[],
            overall_status=RED,
        )
        assert r.overall_status == RED

    def test_gap_item_fields(self):
        g = GapItem(metric="x", target=1.0, current=0.5, gap=0.5, message="hi", status=RED)
        assert g.gap == 0.5


# ===========================================================================
# 2. Traffic-light logic
# ===========================================================================
class TestTrafficLight:
    def test_green_higher_is_better(self, dashboard: NorthStarDashboard):
        assert dashboard._traffic_light(6.0, 6.0) == GREEN

    def test_green_above_target(self, dashboard: NorthStarDashboard):
        assert dashboard._traffic_light(7.0, 6.0) == GREEN

    def test_yellow_higher_is_better(self, dashboard: NorthStarDashboard):
        # 80% of 6.0 = 4.8  -> 5.0 is in yellow zone
        assert dashboard._traffic_light(5.0, 6.0) == YELLOW

    def test_red_higher_is_better(self, dashboard: NorthStarDashboard):
        # below 80% of 6.0 = 4.8
        assert dashboard._traffic_light(4.0, 6.0) == RED

    def test_green_lower_is_better(self, dashboard: NorthStarDashboard):
        # drawdown: 0.20 <= 0.30 target => GREEN
        assert dashboard._traffic_light(0.20, 0.30, higher_is_better=False) == GREEN

    def test_yellow_lower_is_better(self, dashboard: NorthStarDashboard):
        # 120% of 0.30 = 0.36 -> 0.35 is in yellow zone
        assert dashboard._traffic_light(0.35, 0.30, higher_is_better=False) == YELLOW

    def test_red_lower_is_better(self, dashboard: NorthStarDashboard):
        # 0.40 > 0.36 => RED
        assert dashboard._traffic_light(0.40, 0.30, higher_is_better=False) == RED

    def test_exact_80_boundary_is_yellow(self, dashboard: NorthStarDashboard):
        # 80% of 10.0 = 8.0 exactly (no float rounding issue)
        assert dashboard._traffic_light(8.0, 10.0) == YELLOW

    def test_just_below_80_is_red(self, dashboard: NorthStarDashboard):
        assert dashboard._traffic_light(4.79, 6.0) == RED


# ===========================================================================
# 3. Per-experiment status
# ===========================================================================
class TestExperimentStatus:
    def test_all_green_experiment(self, dashboard: NorthStarDashboard):
        exp = _make_experiment()
        status = dashboard._build_experiment_status("test", exp)
        assert status.name == "test"
        assert status.traffic_lights["sharpe"] == GREEN
        assert status.traffic_lights["annual_return"] == GREEN
        assert status.traffic_lights["max_dd"] == GREEN
        assert status.traffic_lights["robustness"] == GREEN
        assert status.traffic_lights["all_years_profitable"] == GREEN

    def test_negative_year_makes_red(self, dashboard: NorthStarDashboard):
        exp = _make_experiment(yearly_returns=[0.10, -0.05, 0.20])
        status = dashboard._build_experiment_status("neg", exp)
        assert status.all_years_profitable is False
        assert status.traffic_lights["all_years_profitable"] == RED

    def test_empty_yearly_returns(self, dashboard: NorthStarDashboard):
        exp = _make_experiment(yearly_returns=[])
        status = dashboard._build_experiment_status("empty_yr", exp)
        assert status.all_years_profitable is False


# ===========================================================================
# 4. Portfolio metrics
# ===========================================================================
class TestPortfolioMetrics:
    def test_blended_sharpe(self, dashboard: NorthStarDashboard, good_experiments):
        pm = dashboard._compute_portfolio_metrics(good_experiments)
        assert pm.blended_sharpe == pytest.approx((7.0 + 8.0) / 2)

    def test_blended_return(self, dashboard: NorthStarDashboard, good_experiments):
        pm = dashboard._compute_portfolio_metrics(good_experiments)
        assert pm.blended_return == pytest.approx((0.60 + 0.70) / 2)

    def test_worst_max_dd(self, dashboard: NorthStarDashboard):
        exps = {
            "a": _make_experiment(max_dd=0.10),
            "b": _make_experiment(max_dd=0.25),
        }
        pm = dashboard._compute_portfolio_metrics(exps)
        assert pm.worst_max_dd == pytest.approx(0.25)

    def test_all_years_profitable_combined(self, dashboard: NorthStarDashboard):
        exps = {
            "a": _make_experiment(yearly_returns=[0.1, 0.2]),
            "b": _make_experiment(yearly_returns=[0.1, -0.05]),
        }
        pm = dashboard._compute_portfolio_metrics(exps)
        assert pm.all_years_profitable is False

    def test_single_experiment_portfolio(self, dashboard: NorthStarDashboard):
        exps = {"only": _make_experiment(sharpe=5.0)}
        pm = dashboard._compute_portfolio_metrics(exps)
        assert pm.blended_sharpe == pytest.approx(5.0)


# ===========================================================================
# 5. Gap analysis
# ===========================================================================
class TestGapAnalysis:
    def test_no_gap_when_targets_met(self, dashboard: NorthStarDashboard):
        pm = PortfolioMetrics(
            blended_sharpe=7.0, blended_return=0.60, worst_max_dd=0.20,
            avg_win_rate=0.85, avg_profit_factor=2.5, avg_robustness=0.80,
            all_years_profitable=True,
        )
        gaps = dashboard._gap_analysis(pm)
        for g in gaps:
            assert g.status == GREEN

    def test_return_gap_message(self, dashboard: NorthStarDashboard):
        pm = PortfolioMetrics(
            blended_sharpe=7.0, blended_return=0.43, worst_max_dd=0.20,
            avg_win_rate=0.85, avg_profit_factor=2.5, avg_robustness=0.80,
            all_years_profitable=True,
        )
        gaps = dashboard._gap_analysis(pm)
        ret_gap = [g for g in gaps if g.metric == "Annual Return"][0]
        assert ret_gap.gap == pytest.approx(0.12, abs=0.01)
        assert "Need" in ret_gap.message and "+12" in ret_gap.message

    def test_drawdown_gap(self, dashboard: NorthStarDashboard):
        pm = PortfolioMetrics(
            blended_sharpe=7.0, blended_return=0.60, worst_max_dd=0.40,
            avg_win_rate=0.85, avg_profit_factor=2.5, avg_robustness=0.80,
            all_years_profitable=True,
        )
        gaps = dashboard._gap_analysis(pm)
        dd_gap = [g for g in gaps if g.metric == "Max Drawdown"][0]
        assert dd_gap.gap == pytest.approx(0.10, abs=0.01)
        assert dd_gap.status == RED

    def test_all_years_profitable_gap(self, dashboard: NorthStarDashboard):
        pm = PortfolioMetrics(
            blended_sharpe=7.0, blended_return=0.60, worst_max_dd=0.20,
            avg_win_rate=0.85, avg_profit_factor=2.5, avg_robustness=0.80,
            all_years_profitable=False,
        )
        gaps = dashboard._gap_analysis(pm)
        ayp = [g for g in gaps if g.metric == "All Years Profitable"][0]
        assert ayp.status == RED
        assert ayp.gap is None


# ===========================================================================
# 6. Overall status
# ===========================================================================
class TestOverallStatus:
    def test_all_green(self, dashboard: NorthStarDashboard):
        gaps = [GapItem("x", 1.0, 1.0, 0.0, "", GREEN), GapItem("y", 1.0, 1.0, 0.0, "", GREEN)]
        assert dashboard._overall_status(gaps) == GREEN

    def test_any_red_makes_red(self, dashboard: NorthStarDashboard):
        gaps = [GapItem("x", 1.0, 1.0, 0.0, "", GREEN), GapItem("y", 1.0, 0.5, 0.5, "", RED)]
        assert dashboard._overall_status(gaps) == RED

    def test_yellow_without_red(self, dashboard: NorthStarDashboard):
        gaps = [GapItem("x", 1.0, 1.0, 0.0, "", GREEN), GapItem("y", 1.0, 0.9, 0.1, "", YELLOW)]
        assert dashboard._overall_status(gaps) == YELLOW


# ===========================================================================
# 7. Evaluate (full pipeline)
# ===========================================================================
class TestEvaluate:
    def test_evaluate_returns_result(self, dashboard: NorthStarDashboard, good_experiments):
        result = dashboard.evaluate(good_experiments)
        assert isinstance(result, NorthStarResult)
        assert len(result.experiment_statuses) == 2
        assert isinstance(result.portfolio_metrics, PortfolioMetrics)
        assert len(result.gap_analysis) > 0
        assert result.overall_status in (GREEN, YELLOW, RED)

    def test_evaluate_good_experiments_green(self, dashboard: NorthStarDashboard, good_experiments):
        result = dashboard.evaluate(good_experiments)
        assert result.overall_status == GREEN


# ===========================================================================
# 8. HTML report generation
# ===========================================================================
class TestHTMLReport:
    def test_report_is_string(self, dashboard: NorthStarDashboard, good_experiments):
        report = dashboard.generate_report(good_experiments)
        assert isinstance(report, str)

    def test_report_contains_html_structure(self, dashboard: NorthStarDashboard, good_experiments):
        report = dashboard.generate_report(good_experiments)
        assert "<!DOCTYPE html>" in report
        assert "<title>North Star Dashboard</title>" in report
        assert "</html>" in report

    def test_report_dark_theme(self, dashboard: NorthStarDashboard, good_experiments):
        report = dashboard.generate_report(good_experiments)
        assert "background: #121212" in report

    def test_report_contains_experiment_names(self, dashboard: NorthStarDashboard, good_experiments):
        report = dashboard.generate_report(good_experiments)
        assert "exp_a" in report
        assert "exp_b" in report

    def test_report_contains_gap_analysis(self, dashboard: NorthStarDashboard, good_experiments):
        report = dashboard.generate_report(good_experiments)
        assert "Gap Analysis" in report

    def test_report_contains_traffic_light_colors(self, dashboard: NorthStarDashboard, good_experiments):
        report = dashboard.generate_report(good_experiments)
        assert "#00e676" in report  # GREEN color


# ===========================================================================
# 9. Edge cases
# ===========================================================================
class TestEdgeCases:
    def test_no_experiments(self, dashboard: NorthStarDashboard):
        result = dashboard.evaluate({})
        assert result.experiment_statuses == []
        assert result.portfolio_metrics.blended_sharpe == 0.0
        assert result.overall_status == RED  # nothing meets targets

    def test_no_experiments_html(self, dashboard: NorthStarDashboard):
        report = dashboard.generate_report({})
        assert "<!DOCTYPE html>" in report

    def test_single_experiment(self, dashboard: NorthStarDashboard):
        exps = {"solo": _make_experiment()}
        result = dashboard.evaluate(exps)
        assert len(result.experiment_statuses) == 1
        assert result.portfolio_metrics.blended_sharpe == pytest.approx(7.0)

    def test_custom_targets(self):
        targets = NorthStarTargets(annual_return_target=0.10, sharpe_target=1.0,
                                   max_dd_target=0.50, robustness_target=0.30)
        dash = NorthStarDashboard(targets)
        exps = {"easy": _make_experiment(sharpe=1.5, annual_return=0.15, max_dd=0.10, robustness_score=0.40)}
        result = dash.evaluate(exps)
        assert result.overall_status == GREEN

    def test_zero_values(self, dashboard: NorthStarDashboard):
        exp = _make_experiment(sharpe=0.0, annual_return=0.0, max_dd=0.0,
                               win_rate=0.0, profit_factor=0.0, robustness_score=0.0,
                               yearly_returns=[])
        result = dashboard.evaluate({"zero": exp})
        assert result.overall_status == RED
