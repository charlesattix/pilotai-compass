"""Tests for compass.strategy_generator — 32 tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from compass.strategy_generator import (
    StrategyGenerator, StrategyTemplate, FitnessScore, StrategyResult, GenerationSummary,
)

def _prices(n=500, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)),
                      index=pd.bdate_range("2023-01-02", periods=n))

def _returns(n=500, seed=42):
    return _prices(n, seed).pct_change().dropna()


class TestGrid:
    def test_generate(self):
        grid = StrategyGenerator.generate_grid({"a": [1, 2], "b": [3, 4]})
        assert len(grid) == 4

    def test_single_param(self):
        grid = StrategyGenerator.generate_grid({"x": [1, 2, 3]})
        assert len(grid) == 3

    def test_build_templates(self):
        grid = [{"a": 1}, {"a": 2}]
        templates = StrategyGenerator.build_templates(grid)
        assert len(templates) == 2
        assert all(isinstance(t, StrategyTemplate) for t in templates)


class TestSignalGeneration:
    def test_momentum(self):
        t = StrategyTemplate("m", {"lookback": 20, "threshold": 0.0}, entry_signal="momentum")
        sig = StrategyGenerator.generate_signal(t, _prices())
        assert not sig.dropna().empty
        assert set(sig.unique()).issubset({-1.0, 0.0, 1.0})

    def test_mean_reversion(self):
        t = StrategyTemplate("mr", {"lookback": 20, "threshold": 1.5}, entry_signal="mean_reversion")
        sig = StrategyGenerator.generate_signal(t, _prices())
        assert not sig.dropna().empty

    def test_breakout(self):
        t = StrategyTemplate("b", {"lookback": 20}, entry_signal="breakout")
        sig = StrategyGenerator.generate_signal(t, _prices())
        assert not sig.dropna().empty

    def test_unknown(self):
        t = StrategyTemplate("u", {}, entry_signal="unknown")
        sig = StrategyGenerator.generate_signal(t, _prices(50))
        assert (sig == 0).all()


class TestBacktest:
    def test_basic(self):
        sig = pd.Series([1.0] * 100, index=pd.bdate_range("2024-01-02", periods=100))
        ret = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 100), index=sig.index)
        result = StrategyGenerator.quick_backtest(sig, ret)
        assert len(result) == 100

    def test_empty(self):
        assert StrategyGenerator.quick_backtest(pd.Series(dtype=float), pd.Series(dtype=float)).empty


class TestFitness:
    def test_basic(self):
        sg = StrategyGenerator()
        ret = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 300),
                         index=pd.bdate_range("2024-01-02", periods=300))
        f = sg.compute_fitness(ret)
        assert isinstance(f, FitnessScore)
        assert f.composite > 0

    def test_composite_bounded(self):
        sg = StrategyGenerator()
        ret = pd.Series(np.random.default_rng(42).normal(0.001, 0.01, 300),
                         index=pd.bdate_range("2024-01-02", periods=300))
        f = sg.compute_fitness(ret)
        assert 0 <= f.composite <= 1.5

    def test_empty(self):
        sg = StrategyGenerator()
        f = sg.compute_fitness(pd.Series(dtype=float))
        assert f.sharpe == 0.0


class TestWalkForward:
    def test_basic(self):
        sg = StrategyGenerator(n_wf_folds=3)
        sig = pd.Series(np.random.default_rng(42).choice([-1, 0, 1], 300),
                         index=pd.bdate_range("2024-01-02", periods=300), dtype=float)
        ret = _returns(300)
        is_sh, oos_sh, deg = sg.walk_forward_test(sig, ret)
        assert isinstance(deg, float)

    def test_short_data(self):
        sg = StrategyGenerator()
        _, _, deg = sg.walk_forward_test(pd.Series([1.0] * 10), pd.Series([0.01] * 10))
        assert deg == 1.0


class TestFilter:
    def test_rejects_overfit(self):
        sg = StrategyGenerator(max_wf_degradation=0.10)
        results = [
            StrategyResult(StrategyTemplate("a", {}), FitnessScore(), wf_degradation=0.50),
            StrategyResult(StrategyTemplate("b", {}), FitnessScore(), wf_degradation=0.05),
        ]
        passed = sg.filter_strategies(results)
        assert len(passed) == 1
        assert passed[0].template.name == "b"


class TestMutation:
    def test_mutate(self):
        sg = StrategyGenerator(mutation_rate=1.0)
        t = StrategyTemplate("orig", {"lookback": 20, "threshold": 0.0})
        ranges = {"lookback": [5, 10, 20, 50], "threshold": [0.0, 0.5, 1.0]}
        mut = sg.mutate(t, ranges, seed=42)
        assert mut.name.endswith("_mut")

    def test_evolve(self):
        sg = StrategyGenerator()
        results = [StrategyResult(StrategyTemplate(f"s{i}", {"lookback": 20}), FitnessScore(composite=i))
                    for i in range(3)]
        offspring = sg.evolve(results, {"lookback": [5, 10, 20, 50]}, n_offspring=6)
        assert len(offspring) <= 6


class TestFullPipeline:
    def test_run(self):
        sg = StrategyGenerator(max_wf_degradation=0.50, n_wf_folds=3)
        prices = _prices(300)
        rets = prices.pct_change().dropna()
        results, summaries = sg.run(
            prices, rets, {"lookback": [10, 20], "threshold": [0.0, 0.01]},
            n_generations=1, top_n=3)
        assert len(results) > 0
        assert len(summaries) == 1

    def test_multi_gen(self):
        sg = StrategyGenerator(max_wf_degradation=0.80, n_wf_folds=3)
        prices = _prices(200)
        rets = prices.pct_change().dropna()
        results, summaries = sg.run(
            prices, rets, {"lookback": [10, 20], "threshold": [0.0]},
            n_generations=2, top_n=2)
        assert len(summaries) == 2


class TestReport:
    def test_creates_file(self, tmp_path):
        sg = StrategyGenerator(max_wf_degradation=0.80, n_wf_folds=3)
        prices = _prices(200)
        rets = prices.pct_change().dropna()
        results, sums = sg.run(prices, rets, {"lookback": [10, 20], "threshold": [0.0]}, top_n=2)
        out = tmp_path / "sg.html"
        path = sg.generate_report(results, sums, output_path=str(out))
        assert Path(path).exists()
        assert "Strategy Generator" in out.read_text()
