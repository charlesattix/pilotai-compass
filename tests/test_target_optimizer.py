"""Tests for compass.target_optimizer — 30 tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from compass.target_optimizer import (
    TargetOptimizer, PerformanceGap, SweepResult, CombinationResult,
    ImprovementOpportunity, TargetAnalysis, NORTH_STAR,
)

def _strat_returns(n=300, k=3, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return {f"s{i}": pd.Series(rng.normal(0.0003 + i * 0.0001, 0.01, n), index=idx) for i in range(k)}


class TestGaps:
    def test_all_met(self):
        to = TargetOptimizer()
        gaps = to.compute_gaps({"annual_return": 0.60, "sharpe": 7.0, "max_drawdown": 0.20})
        assert all(g.is_met for g in gaps)

    def test_all_gaps(self):
        to = TargetOptimizer()
        gaps = to.compute_gaps({"annual_return": 0.10, "sharpe": 1.0, "max_drawdown": 0.50})
        assert not any(g.is_met for g in gaps)

    def test_partial(self):
        to = TargetOptimizer()
        gaps = to.compute_gaps({"annual_return": 0.60, "sharpe": 2.0, "max_drawdown": 0.25})
        met = [g for g in gaps if g.is_met]
        assert 1 <= len(met) < 3

    def test_pct_achieved(self):
        to = TargetOptimizer()
        gaps = to.compute_gaps({"annual_return": 0.275, "sharpe": 3.0, "max_drawdown": 0.30})
        ret_gap = [g for g in gaps if g.metric == "annual_return"][0]
        assert ret_gap.pct_achieved == pytest.approx(0.5)

    def test_gap_positive_for_shortfall(self):
        to = TargetOptimizer()
        gaps = to.compute_gaps({"annual_return": 0.20, "sharpe": 2.0, "max_drawdown": 0.10})
        ret_gap = [g for g in gaps if g.metric == "annual_return"][0]
        assert ret_gap.gap > 0


class TestSweep:
    def test_basic(self):
        def eval_fn(val):
            return {"sharpe": 2.0 + val * 0.1, "annual_return": 0.20, "max_drawdown": 0.15}
        results = TargetOptimizer.parameter_sweep(eval_fn, "spread_width", [1, 3, 5, 7])
        assert len(results) == 4
        assert results[0].sharpe >= results[-1].sharpe  # sorted desc

    def test_with_baseline(self):
        baseline = {"sharpe": 2.0}
        def eval_fn(val): return {"sharpe": 3.0}
        results = TargetOptimizer.parameter_sweep(eval_fn, "p", [1], baseline)
        assert results[0].improvement_vs_baseline == pytest.approx(1.0)

    def test_exception_skipped(self):
        def eval_fn(val):
            if val == 2: raise ValueError("bad")
            return {"sharpe": 1.0}
        results = TargetOptimizer.parameter_sweep(eval_fn, "p", [1, 2, 3])
        assert len(results) == 2


class TestCombinations:
    def test_basic(self):
        sr = _strat_returns(200, 3)
        combos = TargetOptimizer.evaluate_combinations(sr)
        assert len(combos) > 0
        assert all(isinstance(c, CombinationResult) for c in combos)

    def test_sorted_by_sharpe(self):
        combos = TargetOptimizer.evaluate_combinations(_strat_returns(200, 3))
        sharpes = [c.sharpe for c in combos]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_correlation_computed(self):
        combos = TargetOptimizer.evaluate_combinations(_strat_returns(200, 3))
        assert all(isinstance(c.correlation, float) for c in combos)

    def test_weights_sum_one(self):
        combos = TargetOptimizer.evaluate_combinations(_strat_returns(200, 3))
        for c in combos:
            assert sum(c.weights.values()) == pytest.approx(1.0, abs=0.01)


class TestRequiredImprovement:
    def test_basic(self):
        to = TargetOptimizer()
        gaps = to.compute_gaps({"annual_return": 0.20, "sharpe": 2.0, "max_drawdown": 0.10})
        req = to.required_improvements(gaps)
        assert "annual_return" in req
        assert "sharpe" in req

    def test_all_met_empty(self):
        to = TargetOptimizer()
        gaps = to.compute_gaps({"annual_return": 0.60, "sharpe": 7.0, "max_drawdown": 0.20})
        assert TargetOptimizer.required_improvements(gaps) == {}


class TestOpportunities:
    def test_from_sweeps(self):
        sweeps = [SweepResult("p", 5, 4.0, 0.3, 0.1, 2.0)]
        opps = TargetOptimizer.rank_opportunities(sweeps, [])
        assert len(opps) >= 1
        assert opps[0].category == "parameter"

    def test_from_combos(self):
        combos = [CombinationResult(["a", "b"], {"a": 0.5, "b": 0.5}, 3.0, 0.3, 0.1, 0.2)]
        opps = TargetOptimizer.rank_opportunities([], combos)
        assert len(opps) >= 1

    def test_sorted_by_priority(self):
        sweeps = [SweepResult("p", 5, 4.0, 0.3, 0.1, 2.0)]
        combos = [CombinationResult(["a", "b"], {"a": 0.5, "b": 0.5}, 3.0, 0.3, 0.1, 0.2)]
        opps = TargetOptimizer.rank_opportunities(sweeps, combos)
        scores = [o.priority_score for o in opps]
        assert scores == sorted(scores, reverse=True)


class TestAnalyze:
    def test_basic(self):
        to = TargetOptimizer()
        result = to.analyze({"annual_return": 0.20, "sharpe": 2.0, "max_drawdown": 0.25},
                              strategy_returns=_strat_returns(200, 3))
        assert isinstance(result, TargetAnalysis)
        assert len(result.gaps) == 3
        assert len(result.combinations) > 0

    def test_score(self):
        to = TargetOptimizer()
        result = to.analyze({"annual_return": 0.60, "sharpe": 7.0, "max_drawdown": 0.20})
        assert result.overall_score == pytest.approx(1.0)


class TestReport:
    def test_creates_file(self, tmp_path):
        to = TargetOptimizer()
        result = to.analyze({"annual_return": 0.20, "sharpe": 2.0, "max_drawdown": 0.25},
                              strategy_returns=_strat_returns(200, 3))
        out = tmp_path / "target.html"
        path = to.generate_report(result, output_path=str(out))
        assert Path(path).exists()
        assert "North Star" in out.read_text()

    def test_contains_gaps(self, tmp_path):
        to = TargetOptimizer()
        result = to.analyze({"annual_return": 0.20, "sharpe": 2.0, "max_drawdown": 0.25})
        out = tmp_path / "t.html"
        to.generate_report(result, output_path=str(out))
        assert "Gap Analysis" in out.read_text()
