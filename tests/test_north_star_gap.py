"""Tests for compass/north_star_gap.py"""
from __future__ import annotations
import numpy as np
import pytest
from compass.north_star_gap import (
    CapacityPoint, ExperimentResult, GapSummary, MCProjection,
    NorthStarGapAnalyzer, StrategyCombo, TargetGap, TARGETS,
)

def _make_returns(n=500, mu=0.002, seed=42):
    return np.random.RandomState(seed).normal(mu, 0.01, n)

def _make_experiments():
    return [
        ExperimentResult("EXP-400", _make_returns(seed=42), 0.45, 0.08, 3.2, 5e8),
        ExperimentResult("EXP-401", _make_returns(seed=43), 0.55, 0.10, 3.8, 3e8),
        ExperimentResult("EXP-503", _make_returns(seed=44), 0.30, 0.15, 2.5, 2e8),
    ]

def _make_analyzer(**kw):
    return NorthStarGapAnalyzer(_make_experiments(), mc_sims=200, **kw)

class TestDataclasses:
    def test_experiment(self):
        e = ExperimentResult("A", np.array([0.01]), 0.5, 0.1, 3.0, 1e8)
        assert e.sharpe == pytest.approx(3.0)
    def test_target_gap(self):
        g = TargetGap("sharpe", 6.0, 3.0, 3.0, 0.5, 50, "moderate", "A", [], [])
        assert g.gap == pytest.approx(3.0)
    def test_strategy_combo(self):
        c = StrategyCombo(["A", "B"], 0.5, 0.1, 4.0, 0.1, True)
        assert c.combined_sharpe == pytest.approx(4.0)
    def test_capacity_point(self):
        c = CapacityPoint(1e9, 0.3, 2.0, 0.15, True)
        assert c.feasible
    def test_mc_projection(self):
        p = MCProjection("sharpe", 3.0, 6.0, 24, 12, 48, 0.3)
        assert p.probability_in_12m == pytest.approx(0.3)
    def test_gap_summary(self):
        s = GapSummary(50, 1, 1, 2, "sharpe", ["action"])
        assert s.overall_score == pytest.approx(50)

class TestGapComputation:
    def test_four_gaps(self):
        a = _make_analyzer(); a.analyze()
        assert len(a.gaps) == 4
    def test_metrics_covered(self):
        a = _make_analyzer(); a.analyze()
        metrics = {g.metric for g in a.gaps}
        assert metrics == {"annual_return", "max_drawdown", "sharpe", "capacity_aum"}
    def test_scores_in_range(self):
        a = _make_analyzer(); a.analyze()
        for g in a.gaps:
            assert 0 <= g.score <= 100
    def test_status_valid(self):
        a = _make_analyzer(); a.analyze()
        for g in a.gaps:
            assert g.status in ("achieved", "close", "moderate", "far")
    def test_bottlenecks_present(self):
        a = _make_analyzer(); a.analyze()
        for g in a.gaps:
            assert len(g.bottlenecks) > 0
    def test_recommendations_present(self):
        a = _make_analyzer(); a.analyze()
        for g in a.gaps:
            assert len(g.recommendations) > 0
    def test_best_experiment_valid(self):
        a = _make_analyzer(); a.analyze()
        names = {e.name for e in a.experiments}
        for g in a.gaps:
            assert g.best_experiment in names

class TestCombinations:
    def test_combos_generated(self):
        a = _make_analyzer(); a.analyze()
        assert len(a.combos) > 0
    def test_combo_has_experiments(self):
        a = _make_analyzer(); a.analyze()
        for c in a.combos:
            assert len(c.experiments) >= 2
    def test_sorted_by_sharpe(self):
        a = _make_analyzer(); a.analyze()
        sharpes = [c.combined_sharpe for c in a.combos]
        assert sharpes == sorted(sharpes, reverse=True)
    def test_single_experiment_no_combos(self):
        a = NorthStarGapAnalyzer([_make_experiments()[0]])
        a.analyze()
        assert len(a.combos) == 0

class TestCapacity:
    def test_capacity_points(self):
        a = _make_analyzer(); a.analyze()
        assert len(a.capacity) == 5
    def test_sharpe_decreases_with_aum(self):
        a = _make_analyzer(); a.analyze()
        sharpes = [c.estimated_sharpe for c in a.capacity]
        assert sharpes[0] >= sharpes[-1]
    def test_aum_levels(self):
        a = _make_analyzer(); a.analyze()
        aums = [c.aum for c in a.capacity]
        assert aums == sorted(aums)

class TestProjections:
    def test_projections_for_each_gap(self):
        a = _make_analyzer(); a.analyze()
        assert len(a.projections) == len(a.gaps)
    def test_probability_range(self):
        a = _make_analyzer(); a.analyze()
        for p in a.projections:
            assert 0 <= p.probability_in_12m <= 1
    def test_p10_le_p90(self):
        a = _make_analyzer(); a.analyze()
        for p in a.projections:
            assert p.p10_months <= p.p90_months

class TestSummary:
    def test_summary_populated(self):
        a = _make_analyzer(); a.analyze()
        assert a.summary is not None
    def test_overall_score_range(self):
        a = _make_analyzer(); a.analyze()
        assert 0 <= a.summary.overall_score <= 100
    def test_priority_actions(self):
        a = _make_analyzer(); a.analyze()
        assert len(a.summary.priority_actions) > 0
    def test_from_returns(self):
        a = NorthStarGapAnalyzer.from_returns({
            "A": _make_returns(seed=1), "B": _make_returns(seed=2),
        }, mc_sims=100)
        a.analyze()
        assert a.summary is not None

class TestPipeline:
    def test_keys(self):
        a = _make_analyzer()
        result = a.analyze()
        assert {"gaps", "combos", "capacity", "projections", "summary"} == set(result.keys())

class TestReport:
    def test_html(self, tmp_path):
        a = _make_analyzer()
        path = a.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "North Star" in c
    def test_sections(self, tmp_path):
        a = _make_analyzer()
        path = a.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Scorecard" in c and "Capacity" in c and "Priority" in c
    def test_charts(self, tmp_path):
        a = _make_analyzer()
        path = a.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()
    def test_auto_analyze(self, tmp_path):
        a = _make_analyzer()
        assert a.summary is None
        a.generate_report(str(tmp_path / "r.html"))
        assert a.summary is not None
    def test_default_path(self):
        a = _make_analyzer()
        path = a.generate_report()
        assert "north_star_gap.html" in path
