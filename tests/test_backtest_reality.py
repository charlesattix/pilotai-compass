"""Tests for compass/backtest_reality.py -- backtest reality checker."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.backtest_reality import (
    BacktestRealityChecker,
    BacktestRealityResult,
    CheckResult,
    _CHECK_WEIGHTS,
    _grade_from_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trades(
    n: int = 100,
    *,
    seed: int = 42,
    add_look_ahead: bool = False,
    add_ticker: bool = False,
    big_quantity: bool = False,
) -> pd.DataFrame:
    """Build a synthetic trades DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
    entry_prices = 100 + rng.normal(0, 2, n)
    exit_prices = entry_prices + rng.normal(0.2, 1, n)
    qty = rng.randint(1, 20, n).astype(float)
    if big_quantity:
        qty = qty * 10_000  # huge sizes
    pnl = (exit_prices - entry_prices) * qty

    df = pd.DataFrame(
        {
            "date": dates,
            "entry_date": dates,
            "exit_date": dates + pd.Timedelta(days=1),
            "pnl": pnl,
            "entry_price": entry_prices,
            "exit_price": exit_prices,
            "quantity": qty,
        }
    )
    if add_look_ahead:
        # Make first 3 trades have exit before entry
        df.loc[:2, "exit_date"] = df.loc[:2, "entry_date"] - pd.Timedelta(days=2)

    if add_ticker:
        tickers = rng.choice(["AAPL", "MSFT", "GOOG"], n)
        df["ticker"] = tickers

    return df


def _make_returns(n: int = 500, seed: int = 42, gap: bool = False) -> pd.Series:
    """Build a synthetic daily returns Series."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n, freq="B")
    if gap:
        # Insert a 40-day gap
        dates = dates[:100].append(dates[100:] + pd.Timedelta(days=40))
    vals = rng.normal(0.0005, 0.01, n)
    return pd.Series(vals, index=dates)


def _make_checker(**kwargs) -> BacktestRealityChecker:
    """Convenience factory with sensible defaults."""
    defaults = dict(
        trades=_make_trades(),
        returns=_make_returns(),
        n_params_tested=3,
        adv=100_000.0,
        assumed_spread_bps=5.0,
        assumed_commission=1.0,
    )
    defaults.update(kwargs)
    return BacktestRealityChecker(**defaults)


# ===========================================================================
# Dataclass tests
# ===========================================================================


class TestCheckResult:
    def test_fields(self):
        cr = CheckResult("test", True, 85.0, "ok")
        assert cr.name == "test"
        assert cr.passed is True
        assert cr.score == pytest.approx(85.0)
        assert cr.detail == "ok"

    def test_failing_result(self):
        cr = CheckResult("bad", False, 10.0, "nope")
        assert not cr.passed
        assert cr.score < 50


class TestBacktestRealityResult:
    def test_fields(self):
        r = BacktestRealityResult(
            checks=[CheckResult("a", True, 100, "ok")],
            credibility_score=90.0,
            grade="A",
            recommendations=[],
        )
        assert r.grade == "A"
        assert r.credibility_score == pytest.approx(90.0)

    def test_with_recommendations(self):
        r = BacktestRealityResult(
            checks=[], credibility_score=30.0, grade="F", recommendations=["fix it"]
        )
        assert len(r.recommendations) == 1


# ===========================================================================
# Look-ahead bias
# ===========================================================================


class TestLookAheadBias:
    def test_no_look_ahead_passes(self):
        ch = _make_checker()
        cr = ch.check_look_ahead_bias()
        assert cr.passed
        assert cr.score == pytest.approx(100.0)

    def test_look_ahead_detected(self):
        trades = _make_trades(add_look_ahead=True)
        ch = _make_checker(trades=trades)
        cr = ch.check_look_ahead_bias()
        assert not cr.passed
        assert cr.score == pytest.approx(0.0)
        assert "exit_date before entry_date" in cr.detail

    def test_no_entry_exit_columns(self):
        trades = _make_trades().drop(columns=["entry_date", "exit_date"])
        ch = _make_checker(trades=trades)
        cr = ch.check_look_ahead_bias()
        assert cr.passed  # no columns to check


# ===========================================================================
# Survivorship bias
# ===========================================================================


class TestSurvivorshipBias:
    def test_single_ticker_warns(self):
        ch = _make_checker()
        cr = ch.check_survivorship_bias()
        assert not cr.passed  # single ticker assumed
        assert "single ticker" in cr.detail

    def test_multi_ticker_ok(self):
        trades = _make_trades(add_ticker=True)
        ch = _make_checker(trades=trades)
        cr = ch.check_survivorship_bias()
        # multi-ticker should not flag single-ticker warning
        assert "single ticker" not in cr.detail

    def test_data_gap_warns(self):
        returns = _make_returns(gap=True)
        ch = _make_checker(returns=returns)
        cr = ch.check_survivorship_bias()
        assert "gap" in cr.detail.lower()

    def test_no_gap_ok(self):
        returns = _make_returns(gap=False)
        ch = _make_checker(returns=returns)
        cr = ch.check_survivorship_bias()
        assert "gap" not in cr.detail.lower() or cr.passed


# ===========================================================================
# Transaction cost realism
# ===========================================================================


class TestTransactionCostRealism:
    def test_realistic_costs_pass(self):
        ch = _make_checker(assumed_spread_bps=5.0, assumed_commission=1.0)
        cr = ch.check_transaction_cost_realism()
        assert cr.passed
        assert cr.score >= 70

    def test_low_spread_fails(self):
        ch = _make_checker(assumed_spread_bps=0.5)
        cr = ch.check_transaction_cost_realism()
        assert not cr.passed
        assert "spread" in cr.detail.lower()

    def test_low_commission_fails(self):
        ch = _make_checker(assumed_commission=0.1)
        cr = ch.check_transaction_cost_realism()
        assert "commission" in cr.detail.lower()


# ===========================================================================
# Fill realism
# ===========================================================================


class TestFillRealism:
    def test_normal_size_passes(self):
        ch = _make_checker(adv=100_000.0)
        cr = ch.check_fill_realism()
        assert cr.passed
        assert cr.score == pytest.approx(100.0)

    def test_big_size_fails(self):
        trades = _make_trades(big_quantity=True)
        ch = _make_checker(trades=trades, adv=1_000.0)
        cr = ch.check_fill_realism()
        assert not cr.passed
        assert "exceed 10% ADV" in cr.detail

    def test_zero_adv(self):
        ch = _make_checker(adv=0.0)
        cr = ch.check_fill_realism()
        assert not cr.passed


# ===========================================================================
# Capacity check
# ===========================================================================


class TestCapacityCheck:
    def test_normal_capacity(self):
        ch = _make_checker(adv=1_000_000.0)
        cr = ch.check_capacity()
        assert cr.passed

    def test_low_adv_fails(self):
        ch = _make_checker(adv=1.0)
        cr = ch.check_capacity()
        assert not cr.passed

    def test_zero_adv(self):
        ch = _make_checker(adv=0.0)
        cr = ch.check_capacity()
        assert not cr.passed
        assert cr.score == pytest.approx(0.0)


# ===========================================================================
# Parameter sensitivity / cliff detection
# ===========================================================================


class TestParameterSensitivity:
    def test_no_sweep_passes(self):
        ch = _make_checker()
        cr = ch.check_parameter_sensitivity()
        assert cr.passed

    def test_smooth_params_pass(self):
        sweep = {"alpha": [1.5, 1.4, 1.3, 1.2, 1.1]}
        ch = _make_checker(param_sweep=sweep)
        cr = ch.check_parameter_sensitivity()
        assert cr.passed

    def test_cliff_detected(self):
        # 2.0 -> 0.5 is a 75% drop
        sweep = {"alpha": [2.0, 0.5, 0.4]}
        ch = _make_checker(param_sweep=sweep)
        cr = ch.check_parameter_sensitivity()
        assert not cr.passed
        assert "alpha" in cr.detail

    def test_multiple_cliffs(self):
        sweep = {
            "alpha": [2.0, 0.3],
            "beta": [1.0, 0.1],
        }
        ch = _make_checker(param_sweep=sweep)
        cr = ch.check_parameter_sensitivity()
        assert not cr.passed
        assert "alpha" in cr.detail
        assert "beta" in cr.detail

    def test_cliff_exactly_50_pct_no_flag(self):
        # 2.0 -> 1.0 is exactly 50%, threshold is > 50%
        sweep = {"alpha": [2.0, 1.0]}
        ch = _make_checker(param_sweep=sweep)
        cr = ch.check_parameter_sensitivity()
        assert cr.passed

    def test_cliff_just_over_50_pct(self):
        sweep = {"alpha": [2.0, 0.99]}
        ch = _make_checker(param_sweep=sweep)
        cr = ch.check_parameter_sensitivity()
        assert not cr.passed

    def test_single_value_param(self):
        sweep = {"alpha": [1.5]}
        ch = _make_checker(param_sweep=sweep)
        cr = ch.check_parameter_sensitivity()
        assert cr.passed


# ===========================================================================
# OOS degradation
# ===========================================================================


class TestOOSDegradation:
    def test_no_sharpe_skipped(self):
        ch = _make_checker()
        cr = ch.check_oos_degradation()
        assert cr.passed  # skipped

    def test_small_degradation_passes(self):
        ch = _make_checker(is_sharpe=2.0, oos_sharpe=1.8)
        cr = ch.check_oos_degradation()
        assert cr.passed  # 10% degradation

    def test_large_degradation_fails(self):
        ch = _make_checker(is_sharpe=2.0, oos_sharpe=1.0)
        cr = ch.check_oos_degradation()
        assert not cr.passed
        assert "50%" in cr.detail

    def test_exactly_30_pct_passes(self):
        ch = _make_checker(is_sharpe=2.0, oos_sharpe=1.4)
        cr = ch.check_oos_degradation()
        assert cr.passed

    def test_zero_is_sharpe(self):
        ch = _make_checker(is_sharpe=0.0, oos_sharpe=1.0)
        cr = ch.check_oos_degradation()
        assert cr.passed  # near zero IS


# ===========================================================================
# Overfitting ratio
# ===========================================================================


class TestOverfittingRatio:
    def test_low_ratio_passes(self):
        trades = _make_trades(n=200)
        ch = _make_checker(trades=trades, n_params_tested=5)
        cr = ch.check_overfitting_ratio()
        assert cr.passed

    def test_high_ratio_fails(self):
        trades = _make_trades(n=10)
        ch = _make_checker(trades=trades, n_params_tested=5)
        cr = ch.check_overfitting_ratio()
        assert not cr.passed

    def test_zero_trades(self):
        empty = pd.DataFrame(columns=["date", "pnl", "entry_price", "exit_price", "quantity"])
        ch = _make_checker(trades=empty, n_params_tested=5)
        cr = ch.check_overfitting_ratio()
        assert not cr.passed
        assert cr.score == pytest.approx(0.0)


# ===========================================================================
# Complexity penalty
# ===========================================================================


class TestComplexityPenalty:
    def test_few_params_passes(self):
        ch = _make_checker(n_params_tested=2)
        cr = ch.check_complexity_penalty()
        assert cr.passed
        assert cr.score == pytest.approx(100.0)

    def test_many_params_fails(self):
        ch = _make_checker(n_params_tested=50)
        cr = ch.check_complexity_penalty()
        assert not cr.passed

    def test_boundary_3(self):
        ch = _make_checker(n_params_tested=3)
        cr = ch.check_complexity_penalty()
        assert cr.score == pytest.approx(100.0)

    def test_moderate_params(self):
        ch = _make_checker(n_params_tested=10)
        cr = ch.check_complexity_penalty()
        assert cr.passed
        assert 60 <= cr.score <= 70


# ===========================================================================
# Credibility score and grade
# ===========================================================================


class TestCredibilityScore:
    def test_good_backtest_high_score(self):
        ch = _make_checker(
            n_params_tested=2,
            adv=1_000_000.0,
            assumed_spread_bps=5.0,
            assumed_commission=1.0,
            is_sharpe=2.0,
            oos_sharpe=1.8,
            trades=_make_trades(n=200, add_ticker=True),
            returns=_make_returns(gap=False),
        )
        res = ch.run_all()
        assert res.credibility_score >= 70

    def test_bad_backtest_low_score(self):
        ch = _make_checker(
            trades=_make_trades(n=10, add_look_ahead=True, big_quantity=True),
            returns=_make_returns(gap=True),
            n_params_tested=50,
            adv=1.0,
            assumed_spread_bps=0.1,
            assumed_commission=0.01,
            is_sharpe=3.0,
            oos_sharpe=0.5,
            param_sweep={"x": [3.0, 0.1]},
        )
        res = ch.run_all()
        assert res.credibility_score < 40


class TestGradeAssignment:
    def test_grade_a(self):
        assert _grade_from_score(90) == "A"

    def test_grade_b(self):
        assert _grade_from_score(75) == "B"

    def test_grade_c(self):
        assert _grade_from_score(60) == "C"

    def test_grade_d(self):
        assert _grade_from_score(45) == "D"

    def test_grade_f(self):
        assert _grade_from_score(30) == "F"

    def test_boundary_85(self):
        assert _grade_from_score(85) == "A"

    def test_boundary_70(self):
        assert _grade_from_score(70) == "B"

    def test_run_all_sets_grade(self):
        ch = _make_checker()
        res = ch.run_all()
        assert res.grade in ("A", "B", "C", "D", "F")


# ===========================================================================
# HTML report
# ===========================================================================


class TestHTMLReport:
    def test_report_contains_html(self):
        ch = _make_checker()
        html = ch.generate_report()
        assert "<!DOCTYPE html>" in html

    def test_report_has_gauge(self):
        ch = _make_checker()
        html = ch.generate_report()
        assert "<svg" in html
        assert "Grade" in html

    def test_report_has_checklist(self):
        ch = _make_checker()
        ch.run_all()
        html = ch.generate_report()
        assert "look_ahead_bias" in html
        assert "survivorship_bias" in html

    def test_report_has_tornado(self):
        ch = _make_checker(param_sweep={"alpha": [2.0, 1.5, 1.0]})
        html = ch.generate_report()
        assert "Parameter Sensitivity" in html
        assert "alpha" in html

    def test_report_has_degradation(self):
        ch = _make_checker(is_sharpe=2.0, oos_sharpe=1.5)
        html = ch.generate_report()
        assert "In-Sample" in html
        assert "OOS" in html

    def test_report_writes_file(self, tmp_path):
        ch = _make_checker()
        out = tmp_path / "report.html"
        html = ch.generate_report(output=str(out))
        assert out.exists()
        assert "<!DOCTYPE html>" in out.read_text()

    def test_report_auto_runs(self):
        ch = _make_checker()
        assert ch._result is None
        ch.generate_report()
        assert ch._result is not None

    def test_report_no_sweep_no_crash(self):
        ch = _make_checker()
        html = ch.generate_report()
        assert "No parameter sweep" in html

    def test_report_no_oos_no_crash(self):
        ch = _make_checker()
        html = ch.generate_report()
        assert "IS/OOS Sharpe not provided" in html


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_empty_trades(self):
        empty = pd.DataFrame(columns=["date", "pnl", "entry_price", "exit_price", "quantity"])
        ch = _make_checker(trades=empty)
        res = ch.run_all()
        assert isinstance(res, BacktestRealityResult)

    def test_single_return(self):
        returns = pd.Series([0.01], index=pd.to_datetime(["2024-01-02"]))
        ch = _make_checker(returns=returns)
        res = ch.run_all()
        assert res.credibility_score >= 0

    def test_all_zero_returns(self):
        returns = pd.Series(np.zeros(100), index=pd.bdate_range("2024-01-02", periods=100))
        ch = _make_checker(returns=returns)
        res = ch.run_all()
        assert isinstance(res.grade, str)

    def test_negative_adv(self):
        ch = _make_checker(adv=-100)
        cr = ch.check_fill_realism()
        assert not cr.passed

    def test_weights_sum_to_one(self):
        assert pytest.approx(sum(_CHECK_WEIGHTS.values()), abs=1e-9) == 1.0

    def test_run_all_returns_nine_checks(self):
        ch = _make_checker()
        res = ch.run_all()
        assert len(res.checks) == 9

    def test_recommendations_populated_for_failures(self):
        ch = _make_checker(
            trades=_make_trades(add_look_ahead=True),
            assumed_spread_bps=0.1,
        )
        res = ch.run_all()
        assert len(res.recommendations) > 0
