"""Tests for scripts/optimize_portfolio.py — experiment-to-portfolio pipeline."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import the module under test (it's a script, so add to path)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from optimize_portfolio import (
    compute_combined_backtest,
    experiments_to_monthly_returns,
    select_experiments,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic leaderboard entries
# ---------------------------------------------------------------------------

def _make_experiment(
    run_id: str,
    avg_return: float = 20.0,
    worst_dd: float = -10.0,
    years: list = None,
    monthly_pnl_per_year: int = 6,
    verdict: str = "ROBUST",
    seed: int = 42,
) -> dict:
    """Build a synthetic leaderboard entry with monthly P&L."""
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024, 2025]

    rng = np.random.RandomState(seed)
    results = {}
    for yr in years:
        monthly_pnl = {}
        for m in range(1, monthly_pnl_per_year + 1):
            key = f"{yr}-{m:02d}"
            monthly_pnl[key] = {"pnl": rng.normal(1500, 2000), "trades": 3, "wins": 2}
        ret_pct = avg_return + rng.normal(0, 5)
        results[str(yr)] = {
            "return_pct": round(ret_pct, 2),
            "total_trades": 40,
            "win_rate": 75.0,
            "max_drawdown": worst_dd + rng.normal(0, 2),
            "sharpe_ratio": 0.8 + rng.uniform(-0.3, 0.3),
            "starting_capital": 100000,
            "ending_capital": 100000 * (1 + ret_pct / 100),
            "monthly_pnl": monthly_pnl,
        }

    return {
        "run_id": run_id,
        "timestamp": "2026-03-12T00:00:00",
        "params": {},
        "ticker": "SPY",
        "strategies": ["credit_spread"],
        "years_run": years,
        "results": results,
        "summary": {
            "avg_return": avg_return,
            "worst_dd": worst_dd,
            "years_profitable": len(years),
        },
        "validation": {"verdict": verdict},
    }


def _make_leaderboard(n=5):
    """Build a small synthetic leaderboard."""
    return [
        _make_experiment("exp_alpha",  avg_return=30.0, worst_dd=-8.0,  seed=10),
        _make_experiment("exp_beta",   avg_return=25.0, worst_dd=-12.0, seed=20),
        _make_experiment("exp_gamma",  avg_return=20.0, worst_dd=-15.0, seed=30),
        _make_experiment("exp_delta",  avg_return=15.0, worst_dd=-10.0, seed=40),
        _make_experiment("exp_epsilon",avg_return=10.0, worst_dd=-5.0,  seed=50),
    ]


# ---------------------------------------------------------------------------
# select_experiments
# ---------------------------------------------------------------------------

class TestSelectExperiments:

    def test_select_by_run_ids(self):
        lb = _make_leaderboard()
        selected = select_experiments(lb, run_ids=["exp_alpha", "exp_gamma"])
        assert len(selected) == 2
        assert {e["run_id"] for e in selected} == {"exp_alpha", "exp_gamma"}

    def test_select_top_n(self):
        lb = _make_leaderboard()
        selected = select_experiments(lb, top_n=3)
        assert len(selected) == 3
        # Top 3 by avg_return: alpha(30), beta(25), gamma(20)
        ids = [e["run_id"] for e in selected]
        assert ids[0] == "exp_alpha"

    def test_filters_by_max_drawdown(self):
        lb = _make_leaderboard()
        selected = select_experiments(lb, top_n=10, max_dd=-11.0)
        # Exclude exp_beta(-12) and exp_gamma(-15)
        ids = {e["run_id"] for e in selected}
        assert "exp_beta" not in ids
        assert "exp_gamma" not in ids

    def test_filters_by_min_years(self):
        lb = _make_leaderboard()
        lb.append(_make_experiment("exp_short", years=[2024, 2025], seed=99))
        selected = select_experiments(lb, top_n=10, min_years=4)
        ids = {e["run_id"] for e in selected}
        assert "exp_short" not in ids

    def test_missing_run_ids_warns(self):
        lb = _make_leaderboard()
        selected = select_experiments(lb, run_ids=["exp_alpha", "nonexistent"])
        assert len(selected) == 1


# ---------------------------------------------------------------------------
# experiments_to_monthly_returns
# ---------------------------------------------------------------------------

class TestExperimentsToMonthlyReturns:

    def test_returns_correct_shape(self):
        lb = _make_leaderboard()
        experiments = lb[:3]
        returns, months = experiments_to_monthly_returns(experiments)
        assert len(returns) == 3
        # All arrays same length (common months)
        lengths = {len(v) for v in returns.values()}
        assert len(lengths) == 1

    def test_returns_are_fractions(self):
        """Monthly returns should be fractions (not percentages)."""
        lb = _make_leaderboard()
        returns, months = experiments_to_monthly_returns(lb[:2])
        for run_id, arr in returns.items():
            # Typical monthly P&L of ~$1500 on $100k = 0.015
            assert np.abs(arr).max() < 1.0, f"Returns look like percentages, not fractions"

    def test_months_sorted(self):
        lb = _make_leaderboard()
        returns, months = experiments_to_monthly_returns(lb[:2])
        assert months == sorted(months)

    def test_single_experiment(self):
        lb = _make_leaderboard()
        returns, months = experiments_to_monthly_returns(lb[:1])
        assert len(returns) == 1
        assert len(months) > 0

    def test_empty_input(self):
        returns, months = experiments_to_monthly_returns([])
        assert returns == {}
        assert months == []


# ---------------------------------------------------------------------------
# PortfolioOptimizer integration via monthly returns
# ---------------------------------------------------------------------------

class TestPortfolioOptimization:

    def test_optimizer_runs_on_monthly_returns(self):
        """PortfolioOptimizer should work with 12 periods_per_year."""
        from compass.portfolio_optimizer import PortfolioOptimizer

        lb = _make_leaderboard()
        returns, months = experiments_to_monthly_returns(lb[:4])
        opt = PortfolioOptimizer(returns, periods_per_year=12)

        w = opt.max_sharpe()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= 0).all()

    def test_all_methods_produce_valid_weights(self):
        from compass.portfolio_optimizer import PortfolioOptimizer

        lb = _make_leaderboard()
        returns, months = experiments_to_monthly_returns(lb[:4])
        opt = PortfolioOptimizer(returns, periods_per_year=12)

        for method_name in ["max_sharpe", "risk_parity", "equal_risk_contribution", "min_variance"]:
            w = getattr(opt, method_name)()
            assert w.sum() == pytest.approx(1.0, abs=1e-6), f"{method_name} weights don't sum to 1"
            assert (w >= 0).all(), f"{method_name} has negative weights"

    def test_regime_tilt_changes_weights(self):
        from compass.portfolio_optimizer import PortfolioOptimizer, EXPERIMENT_PROFILES

        # Use experiment IDs that match EXPERIMENT_PROFILES
        returns = {
            "EXP-400": np.random.RandomState(1).randn(36) * 0.02,
            "EXP-401": np.random.RandomState(2).randn(36) * 0.015,
            "EXP-503": np.random.RandomState(3).randn(36) * 0.03,
            "EXP-600": np.random.RandomState(4).randn(36) * 0.01,
        }
        opt = PortfolioOptimizer(returns, periods_per_year=12)
        w_neutral = opt.risk_parity()
        w_bull = opt.apply_regime_tilt(w_neutral, "BULL_MACRO")
        w_bear = opt.apply_regime_tilt(w_neutral, "BEAR_MACRO")

        # Bull should differ from bear
        assert not np.allclose(w_bull, w_bear, atol=1e-6)
        # Both sum to 1
        assert w_bull.sum() == pytest.approx(1.0, abs=1e-6)
        assert w_bear.sum() == pytest.approx(1.0, abs=1e-6)

    def test_optimize_full_pipeline(self):
        """Full optimize() with mocked event scaling."""
        from compass.portfolio_optimizer import PortfolioOptimizer
        from unittest.mock import patch

        lb = _make_leaderboard()
        returns, months = experiments_to_monthly_returns(lb[:3])
        opt = PortfolioOptimizer(returns, periods_per_year=12)

        with patch.object(PortfolioOptimizer, "get_event_scaling", return_value=0.90):
            result = opt.optimize(method="risk_parity", regime="NEUTRAL_MACRO")

        assert result.method == "risk_parity"
        assert result.event_scaling == 0.90
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-5)
        # Scaled weights should be 90% of raw weights
        for eid in result.weights:
            assert result.scaled_weights[eid] == pytest.approx(
                result.weights[eid] * 0.90, abs=1e-5
            )
        assert "sharpe_ratio" in result.metrics


# ---------------------------------------------------------------------------
# compute_combined_backtest
# ---------------------------------------------------------------------------

class TestCombinedBacktest:

    def test_blended_return_weighted(self):
        """Blended return should be weight-adjusted sum of individual returns."""
        lb = _make_leaderboard()
        exp1, exp2 = lb[0], lb[1]
        weights = {exp1["run_id"]: 0.60, exp2["run_id"]: 0.40}

        combined = compute_combined_backtest([exp1, exp2], weights)

        # Check a specific year
        for yr in exp1["results"]:
            r1 = exp1["results"][yr]["return_pct"]
            r2 = exp2["results"].get(yr, {}).get("return_pct", 0)
            expected = 0.60 * r1 + 0.40 * r2
            assert combined["yearly"][yr]["return_pct"] == pytest.approx(expected, abs=0.1)

    def test_combined_has_summary(self):
        lb = _make_leaderboard()
        weights = {e["run_id"]: 1.0 / len(lb) for e in lb}
        combined = compute_combined_backtest(lb, weights)

        assert "avg_return" in combined
        assert "worst_dd" in combined
        assert "years_profitable" in combined
        assert "years_total" in combined
        assert combined["years_total"] == 6

    def test_equal_weight_returns_average(self):
        """Equal-weight blend should produce roughly average returns."""
        lb = _make_leaderboard()[:3]
        weights = {e["run_id"]: 1.0 / 3 for e in lb}
        combined = compute_combined_backtest(lb, weights)

        # Avg return should be between min and max individual returns
        individual_avgs = [e["summary"]["avg_return"] for e in lb]
        assert combined["avg_return"] >= min(individual_avgs) - 5
        assert combined["avg_return"] <= max(individual_avgs) + 5

    def test_missing_experiment_handled(self):
        """Weights for non-existent experiments shouldn't crash."""
        lb = _make_leaderboard()[:2]
        weights = {lb[0]["run_id"]: 0.5, lb[1]["run_id"]: 0.3, "nonexistent": 0.2}
        combined = compute_combined_backtest(lb, weights)
        assert combined["years_total"] > 0


# ---------------------------------------------------------------------------
# End-to-end with real leaderboard (integration)
# ---------------------------------------------------------------------------

class TestIntegrationWithRealData:

    @pytest.fixture
    def real_leaderboard(self):
        lb_path = Path(__file__).resolve().parent.parent / "output" / "leaderboard.json"
        if not lb_path.exists():
            pytest.skip("leaderboard.json not found")
        with open(lb_path) as f:
            return json.load(f)

    def test_can_select_and_optimize_real_data(self, real_leaderboard):
        """Select top experiments from real leaderboard and run optimizer."""
        from compass.portfolio_optimizer import PortfolioOptimizer
        from unittest.mock import patch

        experiments = select_experiments(real_leaderboard, top_n=4, max_dd=-25.0)
        if len(experiments) < 2:
            pytest.skip("Not enough qualifying experiments")

        returns, months = experiments_to_monthly_returns(experiments)
        if not returns or len(months) < 6:
            pytest.skip("Insufficient monthly data")

        opt = PortfolioOptimizer(returns, periods_per_year=12)
        with patch.object(PortfolioOptimizer, "get_event_scaling", return_value=1.0):
            result = opt.optimize(method="risk_parity", regime="NEUTRAL_MACRO")

        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-5)
        assert result.metrics["sharpe_ratio"] != 0

    def test_combined_backtest_on_real_data(self, real_leaderboard):
        """Combined backtest metrics should be computable on real data."""
        from compass.portfolio_optimizer import PortfolioOptimizer
        from unittest.mock import patch

        experiments = select_experiments(real_leaderboard, top_n=3, max_dd=-25.0)
        if len(experiments) < 2:
            pytest.skip("Not enough qualifying experiments")

        returns, months = experiments_to_monthly_returns(experiments)
        if not returns:
            pytest.skip("No monthly returns")

        opt = PortfolioOptimizer(returns, periods_per_year=12)
        with patch.object(PortfolioOptimizer, "get_event_scaling", return_value=1.0):
            result = opt.optimize(method="max_sharpe", regime="NEUTRAL_MACRO")

        combined = compute_combined_backtest(experiments, result.weights)
        assert combined["years_total"] >= 1
        assert combined["avg_return"] != 0
