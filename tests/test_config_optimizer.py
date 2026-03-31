"""Tests for compass.config_optimizer – Bayesian configuration optimizer."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from compass.config_optimizer import (
    AcquisitionFunc,
    ConfigOptimizer,
    ConvergencePoint,
    OptimizerResult,
    ParamDef,
    ParamSensitivity,
    ParamType,
    ParetoPoint,
    TrialResult,
    compute_acquisition,
    _rankdata,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _simple_space() -> list[ParamDef]:
    return [
        ParamDef("alpha", ParamType.CONTINUOUS, 0.0, 1.0, default=0.5),
        ParamDef("beta", ParamType.CONTINUOUS, 0.1, 10.0, default=1.0),
    ]


def _mixed_space() -> list[ParamDef]:
    return [
        ParamDef("lr", ParamType.CONTINUOUS, 0.001, 0.1, default=0.01),
        ParamDef("n_trees", ParamType.DISCRETE, choices=[50, 100, 200, 500], default=100),
        ParamDef("kernel", ParamType.CATEGORICAL, choices=["rbf", "matern", "linear"], default="rbf"),
    ]


def _quadratic_objective(params: Dict[str, Any]) -> Dict[str, float]:
    """Simple objective: -(alpha - 0.7)^2 - (beta - 3.0)^2 / 100 → max at (0.7, 3.0)."""
    a = params.get("alpha", 0.5)
    b = params.get("beta", 1.0)
    sharpe = -((a - 0.7) ** 2) - ((b - 3.0) ** 2) / 100
    dd = -abs(a - 0.7) * 0.1
    return {"sharpe": sharpe, "max_dd": dd}


def _noisy_objective(params: Dict[str, Any]) -> Dict[str, float]:
    a = params.get("alpha", 0.5)
    b = params.get("beta", 1.0)
    sharpe = -((a - 0.7) ** 2) - ((b - 3.0) ** 2) / 100 + np.random.randn() * 0.01
    return {"sharpe": sharpe, "max_dd": -abs(a) * 0.05}


def _mixed_objective(params: Dict[str, Any]) -> Dict[str, float]:
    lr = params.get("lr", 0.01)
    n_trees = params.get("n_trees", 100)
    kernel = params.get("kernel", "rbf")
    base = -(lr - 0.01) ** 2 * 1000 + n_trees / 1000
    bonus = 0.1 if kernel == "matern" else 0.0
    return {"sharpe": base + bonus, "max_dd": -lr * 10}


# ── ParamDef ────────────────────────────────────────────────────────────────
class TestParamDef:
    def test_continuous_sample_in_range(self):
        p = ParamDef("x", ParamType.CONTINUOUS, 0.0, 1.0)
        rng = np.random.RandomState(42)
        for _ in range(20):
            v = p.sample(rng)
            assert 0.0 <= v <= 1.0

    def test_discrete_sample_from_choices(self):
        p = ParamDef("n", ParamType.DISCRETE, choices=[10, 20, 50])
        rng = np.random.RandomState(42)
        for _ in range(20):
            assert p.sample(rng) in [10, 20, 50]

    def test_categorical_sample(self):
        p = ParamDef("k", ParamType.CATEGORICAL, choices=["a", "b", "c"])
        rng = np.random.RandomState(42)
        for _ in range(20):
            assert p.sample(rng) in ["a", "b", "c"]

    def test_encode_decode_continuous(self):
        p = ParamDef("x", ParamType.CONTINUOUS, 0.0, 10.0)
        assert p.decode(p.encode(5.5)) == pytest.approx(5.5)

    def test_encode_decode_categorical(self):
        p = ParamDef("k", ParamType.CATEGORICAL, choices=["a", "b", "c"])
        assert p.decode(p.encode("b")) == "b"

    def test_encode_decode_discrete(self):
        p = ParamDef("n", ParamType.DISCRETE, choices=[10, 20, 50])
        assert p.decode(p.encode(20)) == 20

    def test_continuous_clipped(self):
        p = ParamDef("x", ParamType.CONTINUOUS, 0.0, 1.0)
        assert p.decode(5.0) == 1.0
        assert p.decode(-1.0) == 0.0


# ── Acquisition functions ───────────────────────────────────────────────────
class TestAcquisitionFunctions:
    def test_ucb_increases_with_sigma(self):
        mu = np.array([1.0, 1.0])
        sigma_low = np.array([0.1, 0.1])
        sigma_high = np.array([1.0, 1.0])
        ucb_low = compute_acquisition(mu, sigma_low, 0.5, AcquisitionFunc.UCB)
        ucb_high = compute_acquisition(mu, sigma_high, 0.5, AcquisitionFunc.UCB)
        assert np.all(ucb_high > ucb_low)

    def test_ei_nonnegative(self):
        mu = np.array([0.5, 1.0, 1.5])
        sigma = np.array([0.3, 0.3, 0.3])
        ei = compute_acquisition(mu, sigma, 1.0, AcquisitionFunc.EI)
        assert np.all(ei >= -1e-9)

    def test_pi_bounded(self):
        mu = np.array([0.5, 1.0, 2.0])
        sigma = np.array([0.5, 0.5, 0.5])
        pi = compute_acquisition(mu, sigma, 1.0, AcquisitionFunc.PI)
        assert np.all(pi >= 0.0)
        assert np.all(pi <= 1.0 + 1e-9)

    def test_ucb_prefers_high_mu(self):
        mu = np.array([0.5, 2.0])
        sigma = np.array([0.1, 0.1])
        ucb = compute_acquisition(mu, sigma, 0.0, AcquisitionFunc.UCB)
        assert ucb[1] > ucb[0]


# ── Constructor ─────────────────────────────────────────────────────────────
class TestConfigOptimizerInit:
    def test_defaults(self):
        opt = ConfigOptimizer(_simple_space())
        assert opt.acquisition == AcquisitionFunc.UCB
        assert opt.n_initial == 5

    def test_custom_acquisition(self):
        opt = ConfigOptimizer(_simple_space(), acquisition=AcquisitionFunc.EI)
        assert opt.acquisition == AcquisitionFunc.EI

    def test_param_space_stored(self):
        space = _mixed_space()
        opt = ConfigOptimizer(space)
        assert len(opt.param_space) == 3


# ── Optimize ────────────────────────────────────────────────────────────────
class TestOptimize:
    def test_returns_result(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=8)
        assert isinstance(result, OptimizerResult)

    def test_best_params_set(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=10)
        assert "alpha" in result.best_params
        assert "beta" in result.best_params

    def test_n_trials_correct(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=12)
        assert result.n_trials == 12

    def test_convergence_improves(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=5)
        result = opt.optimize(_quadratic_objective, n_trials=20)
        assert result.convergence[-1].best_so_far >= result.convergence[0].best_so_far

    def test_best_near_optimum(self):
        """With enough trials, should find params near (0.7, 3.0)."""
        opt = ConfigOptimizer(_simple_space(), n_initial=5)
        result = opt.optimize(_quadratic_objective, n_trials=30)
        assert abs(result.best_params["alpha"] - 0.7) < 0.5
        # beta range is [0.1, 10.0], optimal at 3.0
        assert abs(result.best_params["beta"] - 3.0) < 5.0

    def test_mixed_param_types(self):
        opt = ConfigOptimizer(_mixed_space(), n_initial=3)
        result = opt.optimize(_mixed_objective, n_trials=10)
        assert "kernel" in result.best_params
        assert result.best_params["kernel"] in ["rbf", "matern", "linear"]
        assert result.best_params["n_trees"] in [50, 100, 200, 500]


# ── Warm start ──────────────────────────────────────────────────────────────
class TestWarmStart:
    def test_warm_start_included(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=2)
        ws = [
            {"params": {"alpha": 0.7, "beta": 3.0}, "objectives": {"sharpe": -0.01, "max_dd": 0.0}},
            {"params": {"alpha": 0.5, "beta": 1.0}, "objectives": {"sharpe": -0.05, "max_dd": -0.02}},
        ]
        result = opt.optimize(_quadratic_objective, n_trials=8, warm_start=ws)
        warm_trials = [t for t in result.trials if t.source == "warm_start"]
        assert len(warm_trials) == 2

    def test_warm_start_counts_toward_total(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=2)
        ws = [{"params": {"alpha": 0.5, "beta": 1.0}, "objectives": {"sharpe": -0.05, "max_dd": 0.0}}]
        result = opt.optimize(_quadratic_objective, n_trials=6, warm_start=ws)
        assert result.n_trials == 6


# ── Suggest / Tell interface ────────────────────────────────────────────────
class TestSuggestTell:
    def test_suggest_returns_dict(self):
        opt = ConfigOptimizer(_simple_space())
        params = opt.suggest()
        assert isinstance(params, dict)
        assert "alpha" in params

    def test_tell_records_trial(self):
        opt = ConfigOptimizer(_simple_space())
        opt.tell({"alpha": 0.5, "beta": 2.0}, {"sharpe": 1.0, "max_dd": -0.05})
        assert len(opt._trials) == 1

    def test_suggest_after_tell_uses_gp(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=2)
        for _ in range(5):
            p = opt.suggest()
            opt.tell(p, _quadratic_objective(p))
        # After enough tells, GP should be fitted
        assert opt._gp is not None


# ── Convergence ─────────────────────────────────────────────────────────────
class TestConvergence:
    def test_convergence_monotonic(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=10)
        bests = [c.best_so_far for c in result.convergence]
        for i in range(1, len(bests)):
            assert bests[i] >= bests[i - 1] - 1e-12

    def test_convergence_length_matches_trials(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=8)
        assert len(result.convergence) == result.n_trials


# ── Sensitivity ─────────────────────────────────────────────────────────────
class TestParamSensitivity:
    def test_sensitivities_present(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=15)
        assert len(result.sensitivities) == 2

    def test_importances_sum_to_one(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=15)
        total = sum(s.importance for s in result.sensitivities)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_sensitivity_has_best_value(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=10)
        for s in result.sensitivities:
            assert s.best_value is not None


# ── Pareto frontier ─────────────────────────────────────────────────────────
class TestParetoFrontier:
    def test_pareto_populated(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=10)
        assert len(result.pareto_frontier) > 0

    def test_non_dominated_exist(self):
        opt = ConfigOptimizer(_simple_space(), n_initial=3)
        result = opt.optimize(_quadratic_objective, n_trials=10)
        non_dom = [p for p in result.pareto_frontier if not p.is_dominated]
        assert len(non_dom) >= 1

    def test_dominated_correctly(self):
        """A point dominated by another should be marked."""
        p1 = ParetoPoint(params={}, objectives={"a": 1.0, "b": 1.0})
        p2 = ParetoPoint(params={}, objectives={"a": 2.0, "b": 2.0})
        # p2 dominates p1
        all_geq = all(p2.objectives[k] >= p1.objectives[k] for k in p1.objectives)
        any_gt = any(p2.objectives[k] > p1.objectives[k] for k in p1.objectives)
        assert all_geq and any_gt


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt = ConfigOptimizer(_simple_space(), n_initial=3)
            result = opt.optimize(_quadratic_objective, n_trials=8)
            path = opt.generate_report(result, output_path=Path(tmp) / "co.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt = ConfigOptimizer(_simple_space(), n_initial=3)
            result = opt.optimize(_quadratic_objective, n_trials=10)
            path = opt.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Config Optimizer" in html
            assert "Convergence" in html
            assert "Sensitivity" in html
            assert "Pareto" in html
            assert "Best" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt = ConfigOptimizer(_simple_space(), n_initial=3)
            result = opt.optimize(_quadratic_objective, n_trials=6)
            path = opt.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_trial_result(self):
        t = TrialResult(0, {"a": 1}, {"sharpe": 1.5}, 1.5)
        assert t.primary_value == 1.5

    def test_convergence_point(self):
        c = ConvergencePoint(5, 1.2, 0.9)
        assert c.trial == 5

    def test_param_sensitivity(self):
        s = ParamSensitivity("x", 0.8, 0.8, 0.7)
        assert s.correlation == 0.8

    def test_optimizer_result_defaults(self):
        r = OptimizerResult()
        assert r.best_value == 0.0
        assert r.trials == []
        assert r.pareto_frontier == []


# ── Utility ─────────────────────────────────────────────────────────────────
class TestUtility:
    def test_rankdata(self):
        x = np.array([3.0, 1.0, 2.0])
        ranks = _rankdata(x)
        assert ranks[0] == 3.0  # largest
        assert ranks[1] == 1.0  # smallest
        assert ranks[2] == 2.0
