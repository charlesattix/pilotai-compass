"""Tests for compass.backtest_validator – backtest validation suite."""
from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.backtest_validator import (
    FAIL,
    PASS,
    WARN,
    BacktestValidator,
    CheckResult,
    OverfitMetrics,
    StatTestResult,
    ValidationResult,
    _ks_test_normal,
    _ljung_box,
    _runs_test,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_trades(
    n: int = 200,
    seed: int = 42,
    start: date = date(2023, 1, 1),
    win_rate: float = 0.70,
) -> pd.DataFrame:
    """Deterministic trade data."""
    rng = np.random.RandomState(seed)
    dates = [start + timedelta(days=i * 2) for i in range(n)]
    exit_dates = [d + timedelta(days=5) for d in dates]
    wins = rng.rand(n) < win_rate
    pnl = np.where(wins, rng.uniform(20, 100, n), rng.uniform(-200, -20, n))
    return pd.DataFrame({
        "date": dates,
        "exit_date": exit_dates,
        "pnl": pnl,
        "ticker": "SPY",
        "slippage_applied": rng.uniform(0.01, 0.05, n),
        "commission": rng.uniform(0.5, 2.0, n),
    })


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(rng.randn(n) * 0.01 + 0.0002, index=idx)


def _make_perfect_trades(n: int = 100) -> pd.DataFrame:
    """All-winning trades (unrealistic)."""
    dates = [date(2023, 1, 1) + timedelta(days=i * 2) for i in range(n)]
    exit_dates = [d + timedelta(days=5) for d in dates]
    return pd.DataFrame({
        "date": dates,
        "exit_date": exit_dates,
        "pnl": [50.0] * n,
        "ticker": "SPY",
    })


def _make_short_trades(n: int = 10) -> pd.DataFrame:
    """Too few trades over too short a period."""
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "pnl": [10.0] * n,
    })


def _make_lookahead_trades() -> pd.DataFrame:
    """Trades with exit before entry (look-ahead bias)."""
    return pd.DataFrame({
        "date": [date(2023, 6, 15), date(2023, 7, 1)],
        "exit_date": [date(2023, 6, 10), date(2023, 7, 5)],
        "pnl": [50.0, 30.0],
    })


# ── Constructor ─────────────────────────────────────────────────────────────
class TestBacktestValidatorInit:
    def test_defaults(self):
        v = BacktestValidator()
        assert v.min_days == 252
        assert v.min_trades == 30

    def test_custom(self):
        v = BacktestValidator(min_days=100, min_trades=10, cost_bps=1.0)
        assert v.min_days == 100
        assert v.cost_bps == 1.0


# ── Full validation ─────────────────────────────────────────────────────────
class TestValidate:
    def test_returns_validation_result(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        assert isinstance(result, ValidationResult)

    def test_score_bounded(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        assert 0 <= result.score <= 100

    def test_grade_set(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        assert result.grade in [PASS, WARN, FAIL]

    def test_checks_populated(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        assert len(result.checks) >= 9  # 9 base checks

    def test_n_trades_set(self):
        trades = _make_trades(n=150)
        result = BacktestValidator().validate(trades)
        assert result.n_trades == 150

    def test_n_days_set(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        assert result.n_days > 0

    def test_generated_at_set(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        assert len(result.generated_at) > 0

    def test_empty_trades(self):
        result = BacktestValidator().validate(pd.DataFrame(columns=["date", "pnl"]))
        assert result.grade == FAIL
        assert result.score == 0

    def test_missing_columns(self):
        result = BacktestValidator().validate(pd.DataFrame({"foo": [1]}))
        assert result.grade == FAIL

    def test_entry_date_alias(self):
        trades = _make_trades()
        trades = trades.rename(columns={"date": "entry_date"})
        result = BacktestValidator().validate(trades)
        assert result.n_trades > 0


# ── Look-ahead bias ────────────────────────────────────────────────────────
class TestLookAheadBias:
    def test_exit_before_entry_detected(self):
        trades = _make_lookahead_trades()
        result = BacktestValidator(min_trades=1, min_days=1).validate(trades)
        la_check = next(c for c in result.checks if c.name == "Look-Ahead Bias")
        assert la_check.grade == FAIL

    def test_normal_trades_pass(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        la_check = next(c for c in result.checks if c.name == "Look-Ahead Bias")
        assert la_check.grade == PASS

    def test_perfect_start_warns(self):
        """All first 20 trades winning should warn."""
        trades = _make_perfect_trades(n=50)
        result = BacktestValidator(min_days=1, min_trades=1).validate(trades)
        la_check = next(c for c in result.checks if c.name == "Look-Ahead Bias")
        assert la_check.grade == WARN


# ── Survivorship bias ──────────────────────────────────────────────────────
class TestSurvivorshipBias:
    def test_single_ticker_warns(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        sb = next(c for c in result.checks if c.name == "Survivorship Bias")
        assert sb.grade == WARN

    def test_multi_ticker_passes(self):
        trades = _make_trades()
        trades.loc[::2, "ticker"] = "QQQ"
        result = BacktestValidator().validate(trades)
        sb = next(c for c in result.checks if c.name == "Survivorship Bias")
        # Multi-ticker, might still warn for gaps, but not for single-ticker
        assert sb.grade in [PASS, WARN]


# ── Data snooping ──────────────────────────────────────────────────────────
class TestDataSnooping:
    def test_single_param_passes(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades, n_params_tested=1)
        ds = next(c for c in result.checks if c.name == "Data Snooping")
        assert ds.grade == PASS

    def test_many_params_warns(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades, n_params_tested=50)
        ds = next(c for c in result.checks if c.name == "Data Snooping")
        assert ds.grade == WARN

    def test_extreme_params_fails(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades, n_params_tested=200)
        ds = next(c for c in result.checks if c.name == "Data Snooping")
        assert ds.grade == FAIL


# ── Fill realism ────────────────────────────────────────────────────────────
class TestFillRealism:
    def test_with_slippage_passes(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        fr = next(c for c in result.checks if c.name == "Fill Realism")
        assert fr.grade == PASS

    def test_zero_slippage_warns(self):
        trades = _make_trades()
        trades["slippage_applied"] = 0.0
        result = BacktestValidator().validate(trades)
        fr = next(c for c in result.checks if c.name == "Fill Realism")
        assert fr.grade == WARN

    def test_no_slippage_col_warns(self):
        trades = _make_trades().drop(columns=["slippage_applied"])
        result = BacktestValidator().validate(trades)
        fr = next(c for c in result.checks if c.name == "Fill Realism")
        assert fr.grade == WARN


# ── Win rate realism ────────────────────────────────────────────────────────
class TestWinRateRealism:
    def test_normal_win_rate_passes(self):
        trades = _make_trades(win_rate=0.70)
        result = BacktestValidator().validate(trades)
        wr = next(c for c in result.checks if c.name == "Win Rate Realism")
        assert wr.grade == PASS

    def test_high_win_rate_warns(self):
        trades = _make_trades(win_rate=0.92)
        result = BacktestValidator().validate(trades)
        wr = next(c for c in result.checks if c.name == "Win Rate Realism")
        assert wr.grade == WARN

    def test_perfect_win_rate_fails(self):
        trades = _make_perfect_trades()
        result = BacktestValidator(min_days=1, min_trades=1).validate(trades)
        wr = next(c for c in result.checks if c.name == "Win Rate Realism")
        assert wr.grade == FAIL


# ── Minimum length and trades ──────────────────────────────────────────────
class TestMinimums:
    def test_short_backtest_fails(self):
        trades = _make_short_trades()
        result = BacktestValidator().validate(trades)
        ml = next(c for c in result.checks if c.name == "Minimum Length")
        assert ml.grade == FAIL

    def test_few_trades_fails(self):
        trades = _make_short_trades(n=5)
        result = BacktestValidator().validate(trades)
        mt = next(c for c in result.checks if c.name == "Minimum Trades")
        assert mt.grade == FAIL

    def test_sufficient_passes(self):
        trades = _make_trades(n=200)
        result = BacktestValidator().validate(trades)
        mt = next(c for c in result.checks if c.name == "Minimum Trades")
        assert mt.grade == PASS


# ── Statistical tests ───────────────────────────────────────────────────────
class TestStatisticalTests:
    def test_stat_tests_with_returns(self):
        trades = _make_trades()
        returns = _make_returns()
        result = BacktestValidator().validate(trades, returns=returns)
        assert len(result.stat_tests) == 3

    def test_stat_tests_names(self):
        trades = _make_trades()
        returns = _make_returns()
        result = BacktestValidator().validate(trades, returns=returns)
        names = {t.test_name for t in result.stat_tests}
        assert "Runs Test (Randomness)" in names
        assert "Ljung-Box (Autocorrelation)" in names
        assert "KS Test (Normality)" in names

    def test_no_returns_no_stat_tests(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        assert len(result.stat_tests) == 0


# ── Statistical helper functions ────────────────────────────────────────────
class TestStatHelpers:
    def test_runs_test_random(self):
        rng = np.random.RandomState(42)
        z, p = _runs_test(rng.randn(500))
        assert 0.0 <= p <= 1.0

    def test_runs_test_nonrandom(self):
        """Sorted data should fail runs test."""
        z, p = _runs_test(np.arange(100, dtype=float))
        assert p < 0.05  # should reject randomness

    def test_ljung_box_iid(self):
        rng = np.random.RandomState(99)
        q, p = _ljung_box(rng.randn(500))
        # iid should not reject
        assert p > 0.01

    def test_ljung_box_autocorrelated(self):
        """AR(1) process should show autocorrelation."""
        rng = np.random.RandomState(11)
        x = np.zeros(500)
        for i in range(1, 500):
            x[i] = 0.8 * x[i - 1] + rng.randn()
        q, p = _ljung_box(x)
        assert p < 0.05

    def test_ks_normal(self):
        rng = np.random.RandomState(77)
        d, p = _ks_test_normal(rng.randn(500))
        assert p > 0.01  # should not reject normality

    def test_ks_nonnormal(self):
        rng = np.random.RandomState(33)
        # Uniform distribution
        d, p = _ks_test_normal(rng.uniform(0, 1, 500))
        assert p < 0.05


# ── Overfitting detection ──────────────────────────────────────────────────
class TestOverfitting:
    def test_oos_degradation_detected(self):
        is_trades = _make_trades(n=200, seed=42, win_rate=0.80)
        oos_trades = _make_trades(n=100, seed=99, win_rate=0.50)
        is_ret = _make_returns(n=200, seed=42)
        # Make OOS returns much worse
        oos_ret = _make_returns(n=100, seed=99) * 0.1
        result = BacktestValidator().validate(
            is_trades, returns=is_ret,
            oos_trades=oos_trades, oos_returns=oos_ret,
        )
        assert result.overfit is not None
        assert result.overfit.sharpe_degradation > 0

    def test_param_cliff_detected(self):
        trades = _make_trades()
        sweep = {"spread_width": [1.0, 2.0, 10.0, 2.0, 1.5]}
        result = BacktestValidator().validate(trades, param_sweep=sweep)
        pc = next((c for c in result.checks if c.name == "Parameter Sensitivity"), None)
        assert pc is not None
        assert pc.grade == FAIL  # 10 → 2 is 80% drop

    def test_smooth_params_pass(self):
        trades = _make_trades()
        sweep = {"spread_width": [1.0, 1.5, 2.0, 1.8, 1.5]}
        result = BacktestValidator().validate(trades, param_sweep=sweep)
        pc = next((c for c in result.checks if c.name == "Parameter Sensitivity"), None)
        assert pc is not None
        assert pc.grade == PASS


# ── Scoring ─────────────────────────────────────────────────────────────────
class TestScoring:
    def test_good_backtest_high_score(self):
        trades = _make_trades(n=200, win_rate=0.70)
        returns = _make_returns()
        result = BacktestValidator().validate(trades, returns=returns)
        assert result.score >= 50

    def test_bad_backtest_lower_score(self):
        """Perfect trades should score lower than realistic ones."""
        good_trades = _make_trades(n=200, win_rate=0.70)
        good_result = BacktestValidator().validate(good_trades, returns=_make_returns())
        bad_trades = _make_perfect_trades(n=50)
        bad_result = BacktestValidator(min_days=1, min_trades=1).validate(bad_trades)
        assert bad_result.score < good_result.score

    def test_score_grade_consistency(self):
        trades = _make_trades()
        result = BacktestValidator().validate(trades)
        if result.score >= 70:
            assert result.grade == PASS
        elif result.score >= 40:
            assert result.grade == WARN
        else:
            assert result.grade == FAIL


# ── Recommendations ─────────────────────────────────────────────────────────
class TestRecommendations:
    def test_fails_generate_recommendations(self):
        trades = _make_short_trades(n=5)
        result = BacktestValidator().validate(trades)
        assert len(result.recommendations) > 0

    def test_clean_backtest_no_critical(self):
        trades = _make_trades(n=200, win_rate=0.65)
        result = BacktestValidator().validate(trades)
        critical = [r for r in result.recommendations if "[CRITICAL]" in r]
        # Should have few/no critical issues for a well-formed backtest
        assert len(critical) <= 1  # at most the survivorship single-ticker warn


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = _make_trades()
            returns = _make_returns()
            v = BacktestValidator()
            result = v.validate(trades, returns=returns)
            path = v.generate_report(result, output_path=Path(tmp) / "t.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = _make_trades()
            returns = _make_returns()
            oos_trades = _make_trades(n=100, seed=99)
            oos_returns = _make_returns(n=100, seed=99)
            v = BacktestValidator()
            result = v.validate(
                trades, returns=returns,
                oos_trades=oos_trades, oos_returns=oos_returns,
            )
            path = v.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Backtest Validation" in html
            assert "Validation Checklist" in html
            assert "Statistical Tests" in html
            assert "Overfitting" in html
            assert "Recommendations" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            trades = _make_trades()
            v = BacktestValidator()
            result = v.validate(trades)
            path = v.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html

    def test_report_empty_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = BacktestValidator()
            result = ValidationResult(generated_at="2024-01-01T00:00:00+00:00")
            path = v.generate_report(result, output_path=Path(tmp) / "e.html")
            assert path.exists()


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_check_result(self):
        c = CheckResult(name="Test", grade=PASS, message="ok", score=100, category="bias")
        assert c.name == "Test"
        assert c.grade == PASS

    def test_stat_test_result(self):
        s = StatTestResult(test_name="Runs", statistic=1.5, pvalue=0.13, passed=True, interpretation="ok")
        assert s.passed is True

    def test_validation_result_defaults(self):
        r = ValidationResult()
        assert r.score == 0.0
        assert r.checks == []
        assert r.stat_tests == []
        assert r.overfit is None

    def test_overfit_metrics(self):
        o = OverfitMetrics(
            is_sharpe=2.0, oos_sharpe=1.0, sharpe_degradation=0.5,
            is_win_rate=0.8, oos_win_rate=0.6,
            is_avg_pnl=50, oos_avg_pnl=20, pnl_degradation=0.6,
            parameter_cliff=False, min_length_met=True,
        )
        assert o.sharpe_degradation == 0.5
