"""Tests for compass.experiment_manager — 42 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.experiment_manager import (
    ExperimentManager,
    ExperimentStatus,
    Experiment,
    BacktestResult,
    SuccessCriteria,
    NorthStarComparison,
    ABTestResult,
    LeaderboardEntry,
    ExperimentVersion,
    NORTH_STAR,
    VALID_TRANSITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(spread_width: int = 5, dte: int = 45) -> dict:
    return {"strategy": {"spread_width": spread_width, "dte_target": dte},
            "risk": {"max_risk_per_trade": 2.0}}


def _result(sharpe: float = 2.0, ret: float = 0.20, dd: float = 0.10,
            wr: float = 0.55, n: int = 100) -> BacktestResult:
    rng = np.random.default_rng(42)
    dr = pd.Series(rng.normal(ret / 252, 0.01, 252),
                    index=pd.bdate_range("2024-01-02", periods=252))
    return BacktestResult(
        annual_return=ret, sharpe=sharpe, max_drawdown=dd,
        win_rate=wr, n_trades=n, total_pnl=ret * 100000,
        daily_returns=dr,
    )


def _north_star_result() -> BacktestResult:
    return _result(sharpe=7.0, ret=0.60, dd=0.20, wr=0.60, n=200)


def _mock_backtest(config: dict) -> BacktestResult:
    return _result(sharpe=2.5, ret=0.25)


def _mgr_with_experiments() -> ExperimentManager:
    mgr = ExperimentManager()
    mgr.register("EXP-1", "Strategy A", "Test A", _config())
    mgr.register("EXP-2", "Strategy B", "Test B", _config(spread_width=10))
    mgr.register("EXP-3", "Strategy C", "Test C", _config(dte=30))
    mgr._experiments["EXP-1"].result = _result(sharpe=3.0, ret=0.30)
    mgr._experiments["EXP-1"].status = ExperimentStatus.COMPLETED
    mgr._experiments["EXP-2"].result = _result(sharpe=1.5, ret=0.12)
    mgr._experiments["EXP-2"].status = ExperimentStatus.COMPLETED
    mgr._experiments["EXP-3"].result = _result(sharpe=5.0, ret=0.45)
    mgr._experiments["EXP-3"].status = ExperimentStatus.COMPLETED
    return mgr


# ===========================================================================
# Registration
# ===========================================================================

class TestRegistration:
    def test_register(self):
        mgr = ExperimentManager()
        exp = mgr.register("EXP-1", "Test", "Hypothesis", _config())
        assert isinstance(exp, Experiment)
        assert exp.id == "EXP-1"
        assert exp.status == ExperimentStatus.PROPOSED

    def test_duplicate_raises(self):
        mgr = ExperimentManager()
        mgr.register("EXP-1", "A", "H", _config())
        with pytest.raises(ValueError):
            mgr.register("EXP-1", "B", "H2", _config())

    def test_version_computed(self):
        mgr = ExperimentManager()
        exp = mgr.register("EXP-1", "A", "H", _config())
        assert exp.version is not None
        assert len(exp.version.config_hash) == 16

    def test_tags(self):
        mgr = ExperimentManager()
        exp = mgr.register("EXP-1", "A", "H", _config(), tags=["credit_spread"])
        assert "credit_spread" in exp.tags

    def test_custom_criteria(self):
        mgr = ExperimentManager()
        c = SuccessCriteria(min_sharpe=3.0, min_annual_return=0.30)
        exp = mgr.register("EXP-1", "A", "H", _config(), criteria=c)
        assert exp.criteria.min_sharpe == 3.0


# ===========================================================================
# Status transitions
# ===========================================================================

class TestTransitions:
    def test_proposed_to_running(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config())
        exp = mgr.transition("E1", ExperimentStatus.RUNNING)
        assert exp.status == ExperimentStatus.RUNNING

    def test_running_to_completed(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config())
        mgr.transition("E1", ExperimentStatus.RUNNING)
        exp = mgr.transition("E1", ExperimentStatus.COMPLETED)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.completed_at is not None

    def test_completed_to_promoted(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config())
        mgr.transition("E1", ExperimentStatus.RUNNING)
        mgr.transition("E1", ExperimentStatus.COMPLETED)
        exp = mgr.transition("E1", ExperimentStatus.PROMOTED)
        assert exp.status == ExperimentStatus.PROMOTED

    def test_invalid_transition_raises(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config())
        with pytest.raises(ValueError):
            mgr.transition("E1", ExperimentStatus.COMPLETED)  # skip RUNNING

    def test_killed_is_terminal(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config())
        mgr.transition("E1", ExperimentStatus.KILLED)
        with pytest.raises(ValueError):
            mgr.transition("E1", ExperimentStatus.RUNNING)

    def test_not_found_raises(self):
        mgr = ExperimentManager()
        with pytest.raises(KeyError):
            mgr.transition("NOPE", ExperimentStatus.RUNNING)


# ===========================================================================
# Backtest execution
# ===========================================================================

class TestBacktest:
    def test_run_backtest(self):
        mgr = ExperimentManager(backtest_fn=_mock_backtest)
        mgr.register("E1", "A", "H", _config())
        result = mgr.run_backtest("E1")
        assert isinstance(result, BacktestResult)
        assert mgr.get("E1").status == ExperimentStatus.COMPLETED

    def test_no_fn_raises(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config())
        with pytest.raises(RuntimeError):
            mgr.run_backtest("E1")

    def test_set_result(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config())
        mgr.transition("E1", ExperimentStatus.RUNNING)
        mgr.set_result("E1", _result())
        assert mgr.get("E1").result is not None
        assert mgr.get("E1").status == ExperimentStatus.COMPLETED


# ===========================================================================
# North Star comparison
# ===========================================================================

class TestNorthStar:
    def test_meets(self):
        ns = ExperimentManager.compare_north_star("E1", _north_star_result())
        assert ns.meets_north_star

    def test_fails_sharpe(self):
        r = _result(sharpe=2.0, ret=0.60, dd=0.20)
        ns = ExperimentManager.compare_north_star("E1", r)
        assert not ns.meets_north_star

    def test_fails_drawdown(self):
        r = _result(sharpe=7.0, ret=0.60, dd=0.40)
        ns = ExperimentManager.compare_north_star("E1", r)
        assert not ns.drawdown_ok

    def test_pct_computed(self):
        r = _result(sharpe=3.0, ret=0.275)
        ns = ExperimentManager.compare_north_star("E1", r)
        assert ns.sharpe_pct == pytest.approx(3.0 / 6.0)
        assert ns.return_pct == pytest.approx(0.275 / 0.55)


class TestCriteria:
    def test_meets(self):
        mgr = _mgr_with_experiments()
        assert mgr.check_criteria("EXP-1")

    def test_fails_no_result(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config())
        assert not mgr.check_criteria("E1")

    def test_fails_sharpe(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config(),
                       criteria=SuccessCriteria(min_sharpe=10.0))
        mgr._experiments["E1"].result = _result(sharpe=2.0)
        assert not mgr.check_criteria("E1")


# ===========================================================================
# A/B testing
# ===========================================================================

class TestABTest:
    def test_significant(self):
        mgr = ExperimentManager()
        mgr.register("C", "Control", "H", _config())
        mgr.register("V", "Variant", "H", _config())
        # Make variant clearly better
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2024-01-02", periods=252)
        mgr._experiments["C"].result = BacktestResult(
            sharpe=1.0, annual_return=0.10, daily_returns=pd.Series(rng.normal(0.0004, 0.01, 252), index=idx))
        mgr._experiments["V"].result = BacktestResult(
            sharpe=3.0, annual_return=0.30, daily_returns=pd.Series(rng.normal(0.002, 0.01, 252), index=idx))
        ab = mgr.ab_test("C", "V")
        assert isinstance(ab, ABTestResult)
        assert ab.winner == "V"
        assert ab.is_significant

    def test_tie(self):
        mgr = ExperimentManager()
        mgr.register("C", "Control", "H", _config())
        mgr.register("V", "Variant", "H", _config())
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2024-01-02", periods=252)
        same = pd.Series(rng.normal(0.001, 0.01, 252), index=idx)
        mgr._experiments["C"].result = BacktestResult(sharpe=2.0, daily_returns=same)
        mgr._experiments["V"].result = BacktestResult(sharpe=2.0, daily_returns=same)
        ab = mgr.ab_test("C", "V")
        assert ab.winner == "tie"

    def test_no_result_raises(self):
        mgr = ExperimentManager()
        mgr.register("C", "A", "H", _config())
        mgr.register("V", "B", "H", _config())
        with pytest.raises(ValueError):
            mgr.ab_test("C", "V")


# ===========================================================================
# Versioning
# ===========================================================================

class TestVersioning:
    def test_config_hash_deterministic(self):
        mgr = ExperimentManager()
        v1 = mgr._compute_version(_config())
        v2 = mgr._compute_version(_config())
        assert v1.config_hash == v2.config_hash

    def test_different_config_different_hash(self):
        mgr = ExperimentManager()
        v1 = mgr._compute_version(_config(spread_width=5))
        v2 = mgr._compute_version(_config(spread_width=10))
        assert v1.config_hash != v2.config_hash

    def test_update_version(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config())
        v = mgr.update_version("E1", data_hash="abc123", code_hash="def456")
        assert v.data_hash == "abc123"
        assert v.code_hash == "def456"


# ===========================================================================
# Leaderboard
# ===========================================================================

class TestLeaderboard:
    def test_ranked_by_sharpe(self):
        mgr = _mgr_with_experiments()
        lb = mgr.leaderboard(sort_by="sharpe")
        assert len(lb) == 3
        assert lb[0].rank == 1
        assert lb[0].sharpe >= lb[1].sharpe >= lb[2].sharpe

    def test_ranked_by_return(self):
        mgr = _mgr_with_experiments()
        lb = mgr.leaderboard(sort_by="annual_return")
        rets = [e.annual_return for e in lb]
        assert rets == sorted(rets, reverse=True)

    def test_exclude_killed(self):
        mgr = _mgr_with_experiments()
        mgr.transition("EXP-2", ExperimentStatus.KILLED)
        lb = mgr.leaderboard(include_killed=False)
        ids = {e.experiment_id for e in lb}
        assert "EXP-2" not in ids

    def test_include_killed(self):
        mgr = _mgr_with_experiments()
        mgr.transition("EXP-2", ExperimentStatus.KILLED)
        lb = mgr.leaderboard(include_killed=True)
        ids = {e.experiment_id for e in lb}
        assert "EXP-2" in ids


# ===========================================================================
# Queries
# ===========================================================================

class TestQueries:
    def test_by_status(self):
        mgr = _mgr_with_experiments()
        completed = mgr.by_status(ExperimentStatus.COMPLETED)
        assert len(completed) == 3

    def test_by_tag(self):
        mgr = ExperimentManager()
        mgr.register("E1", "A", "H", _config(), tags=["cs"])
        mgr.register("E2", "B", "H", _config(), tags=["ic"])
        assert len(mgr.by_tag("cs")) == 1

    def test_experiments_dict(self):
        mgr = _mgr_with_experiments()
        assert len(mgr.experiments) == 3


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        mgr = _mgr_with_experiments()
        out = tmp_path / "exp.html"
        result = mgr.generate_report(output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Experiment Dashboard" in html

    def test_contains_leaderboard(self, tmp_path):
        mgr = _mgr_with_experiments()
        out = tmp_path / "exp.html"
        mgr.generate_report(output_path=str(out))
        html = out.read_text()
        assert "Leaderboard" in html
        assert "EXP-1" in html

    def test_contains_chart(self, tmp_path):
        mgr = _mgr_with_experiments()
        out = tmp_path / "exp.html"
        mgr.generate_report(output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Sharpe" in html

    def test_contains_north_star(self, tmp_path):
        mgr = _mgr_with_experiments()
        out = tmp_path / "exp.html"
        mgr.generate_report(output_path=str(out))
        html = out.read_text()
        assert "North Star" in html

    def test_empty_manager(self, tmp_path):
        mgr = ExperimentManager()
        out = tmp_path / "exp.html"
        result = mgr.generate_report(output_path=str(out))
        assert Path(result).exists()
