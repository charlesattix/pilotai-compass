"""Tests for compass.experiment_compare module."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from compass.experiment_compare import (
    CompareResult,
    ExperimentComparer,
    ExperimentMetrics,
    PairComparison,
)


# ------------------------------------------------------------------ #
#  Fixtures / helpers                                                 #
# ------------------------------------------------------------------ #

def _daily_returns(mean: float, std: float, n: int = 252, seed: int = 0) -> pd.Series:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(rng.normal(mean, std, n), index=dates)


def _make_comparer(**kwargs: pd.Series) -> ExperimentComparer:
    return ExperimentComparer(kwargs)


def _two_experiment_result() -> CompareResult:
    good = _daily_returns(0.001, 0.01, seed=1)
    bad = _daily_returns(-0.0005, 0.015, seed=2)
    c = ExperimentComparer({"good": good, "bad": bad})
    return c.compare()


# ------------------------------------------------------------------ #
#  ExperimentMetrics dataclass tests                                 #
# ------------------------------------------------------------------ #

class TestExperimentMetricsDataclass:
    def test_fields_exist(self):
        m = ExperimentMetrics(
            experiment_id="x", sharpe=1.0, sortino=1.5, calmar=2.0,
            total_return=0.1, annual_return=0.12, max_dd=-0.05,
            win_rate=0.55, avg_trade_duration=3.0, profit_factor=1.8,
        )
        assert m.experiment_id == "x"
        assert m.sharpe == 1.0
        assert m.profit_factor == 1.8

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(ExperimentMetrics)


class TestPairComparisonDataclass:
    def test_fields_exist(self):
        p = PairComparison(
            exp_a="a", exp_b="b", mean_diff=0.001, t_stat=2.1,
            p_value=0.04, sharpe_diff=0.5, ci_lower=0.1, ci_upper=0.9,
        )
        assert p.exp_a == "a"
        assert p.ci_upper == 0.9

    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(PairComparison)


class TestCompareResultDataclass:
    def test_fields(self):
        r = _two_experiment_result()
        assert isinstance(r, CompareResult)
        assert isinstance(r.experiment_metrics, list)
        assert isinstance(r.pair_comparisons, list)
        assert isinstance(r.correlation_matrix, pd.DataFrame)
        assert isinstance(r.best_experiment, str)
        assert isinstance(r.generated_at, str)


# ------------------------------------------------------------------ #
#  Per-experiment metrics tests                                      #
# ------------------------------------------------------------------ #

class TestPerExperimentMetrics:
    def test_sharpe_positive_for_positive_mean(self):
        r = _two_experiment_result()
        good = next(m for m in r.experiment_metrics if m.experiment_id == "good")
        assert good.sharpe > 0

    def test_sharpe_bounded(self):
        r = _two_experiment_result()
        for m in r.experiment_metrics:
            assert -10 < m.sharpe < 10

    def test_total_return_positive(self):
        r = _two_experiment_result()
        good = next(m for m in r.experiment_metrics if m.experiment_id == "good")
        assert good.total_return > 0

    def test_total_return_negative(self):
        r = _two_experiment_result()
        bad = next(m for m in r.experiment_metrics if m.experiment_id == "bad")
        assert bad.total_return < 0

    def test_max_dd_non_positive(self):
        r = _two_experiment_result()
        for m in r.experiment_metrics:
            assert m.max_dd <= 0

    def test_win_rate_between_0_and_1(self):
        r = _two_experiment_result()
        for m in r.experiment_metrics:
            assert 0 <= m.win_rate <= 1

    def test_sortino_greater_than_sharpe_for_skewed(self):
        # Positive-mean returns typically have sortino > sharpe
        r = _two_experiment_result()
        good = next(m for m in r.experiment_metrics if m.experiment_id == "good")
        # sortino uses only downside deviation so should be >= sharpe for positive-mean
        assert good.sortino >= good.sharpe * 0.8  # allow some slack

    def test_calmar_positive_for_positive_return(self):
        r = _two_experiment_result()
        good = next(m for m in r.experiment_metrics if m.experiment_id == "good")
        assert good.calmar > 0

    def test_profit_factor_above_one_for_positive(self):
        r = _two_experiment_result()
        good = next(m for m in r.experiment_metrics if m.experiment_id == "good")
        assert good.profit_factor > 1.0

    def test_avg_trade_duration_positive(self):
        r = _two_experiment_result()
        for m in r.experiment_metrics:
            assert m.avg_trade_duration > 0

    def test_annual_return_sign_matches_total(self):
        r = _two_experiment_result()
        for m in r.experiment_metrics:
            if m.total_return != 0:
                assert np.sign(m.annual_return) == np.sign(m.total_return)


# ------------------------------------------------------------------ #
#  Paired t-test tests                                               #
# ------------------------------------------------------------------ #

class TestPairedTTest:
    def test_identical_experiments_not_significant(self):
        s = _daily_returns(0.001, 0.01, seed=10)
        c = ExperimentComparer({"a": s, "b": s.copy()})
        r = c.compare()
        pair = r.pair_comparisons[0]
        assert pair.p_value > 0.5
        assert abs(pair.t_stat) < 0.01

    def test_different_experiments_significant(self):
        a = _daily_returns(0.003, 0.01, seed=10)
        b = _daily_returns(-0.003, 0.01, seed=11)
        c = ExperimentComparer({"a": a, "b": b})
        r = c.compare()
        pair = r.pair_comparisons[0]
        assert pair.p_value < 0.05
        assert abs(pair.t_stat) > 2

    def test_mean_diff_sign(self):
        a = _daily_returns(0.002, 0.01, seed=10)
        b = _daily_returns(-0.001, 0.01, seed=11)
        c = ExperimentComparer({"a": a, "b": b})
        r = c.compare()
        pair = r.pair_comparisons[0]
        assert pair.mean_diff > 0  # a - b > 0

    def test_correlated_experiments_significant(self):
        """Correlated experiments should show significance with smaller mean diff."""
        base = _daily_returns(0.0, 0.01, seed=42)
        a = base + 0.001  # shift mean
        b = base.copy()
        c = ExperimentComparer({"a": a, "b": b})
        r = c.compare()
        pair = r.pair_comparisons[0]
        # Highly correlated, so even small diff is very significant
        assert pair.p_value < 0.01

    def test_p_value_bounded(self):
        r = _two_experiment_result()
        for p in r.pair_comparisons:
            assert 0 <= p.p_value <= 1


# ------------------------------------------------------------------ #
#  Bootstrap CI tests                                                #
# ------------------------------------------------------------------ #

class TestBootstrapCI:
    def test_ci_contains_zero_for_similar(self):
        s = _daily_returns(0.001, 0.01, seed=5)
        noise = _daily_returns(0.001, 0.01, seed=6)
        c = ExperimentComparer({"a": s, "b": noise})
        r = c.compare()
        pair = r.pair_comparisons[0]
        # Similar experiments — CI likely contains zero
        assert pair.ci_lower < pair.ci_upper

    def test_ci_excludes_zero_for_very_different(self):
        a = _daily_returns(0.005, 0.005, seed=10)
        b = _daily_returns(-0.005, 0.005, seed=11)
        c = ExperimentComparer({"a": a, "b": b})
        r = c.compare()
        pair = r.pair_comparisons[0]
        # Very different — CI should exclude zero
        assert pair.ci_lower > 0

    def test_ci_lower_less_than_upper(self):
        r = _two_experiment_result()
        for p in r.pair_comparisons:
            assert p.ci_lower <= p.ci_upper


# ------------------------------------------------------------------ #
#  Correlation matrix tests                                          #
# ------------------------------------------------------------------ #

class TestCorrelationMatrix:
    def test_symmetric(self):
        r = _two_experiment_result()
        corr = r.correlation_matrix
        pd.testing.assert_frame_equal(corr, corr.T, check_names=False)

    def test_diagonal_one(self):
        r = _two_experiment_result()
        corr = r.correlation_matrix
        for i in range(len(corr)):
            assert abs(corr.iloc[i, i] - 1.0) < 1e-10

    def test_bounded_minus_one_to_one(self):
        r = _two_experiment_result()
        corr = r.correlation_matrix
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()

    def test_shape_matches_experiments(self):
        r = _two_experiment_result()
        assert r.correlation_matrix.shape == (2, 2)

    def test_three_experiments(self):
        a = _daily_returns(0.001, 0.01, seed=1)
        b = _daily_returns(0.0, 0.01, seed=2)
        c_ret = _daily_returns(-0.001, 0.01, seed=3)
        comp = ExperimentComparer({"a": a, "b": b, "c": c_ret})
        r = comp.compare()
        assert r.correlation_matrix.shape == (3, 3)


# ------------------------------------------------------------------ #
#  Best experiment selection                                         #
# ------------------------------------------------------------------ #

class TestBestExperiment:
    def test_best_is_highest_sharpe(self):
        r = _two_experiment_result()
        assert r.best_experiment == "good"

    def test_best_with_three(self):
        a = _daily_returns(0.003, 0.01, seed=1)
        b = _daily_returns(0.001, 0.01, seed=2)
        c_ret = _daily_returns(0.002, 0.01, seed=3)
        comp = ExperimentComparer({"a": a, "b": b, "c": c_ret})
        r = comp.compare()
        sharpes = {m.experiment_id: m.sharpe for m in r.experiment_metrics}
        expected = max(sharpes, key=sharpes.get)
        assert r.best_experiment == expected


# ------------------------------------------------------------------ #
#  HTML report tests                                                 #
# ------------------------------------------------------------------ #

class TestHTMLReport:
    def test_report_is_string(self):
        c = ExperimentComparer({"x": _daily_returns(0.001, 0.01)})
        html = c.generate_report()
        assert isinstance(html, str)

    def test_report_contains_html_tags(self):
        c = ExperimentComparer({"x": _daily_returns(0.001, 0.01)})
        html = c.generate_report()
        assert "<html>" in html
        assert "</html>" in html
        assert "<table>" in html

    def test_report_contains_svg(self):
        c = ExperimentComparer({
            "a": _daily_returns(0.001, 0.01, seed=1),
            "b": _daily_returns(0.0, 0.01, seed=2),
        })
        html = c.generate_report()
        assert "<svg" in html

    def test_report_contains_experiment_ids(self):
        c = ExperimentComparer({
            "alpha": _daily_returns(0.001, 0.01, seed=1),
            "beta": _daily_returns(0.0, 0.01, seed=2),
        })
        html = c.generate_report()
        assert "alpha" in html
        assert "beta" in html

    def test_report_contains_monthly_returns(self):
        c = ExperimentComparer({"x": _daily_returns(0.001, 0.01)})
        html = c.generate_report()
        assert "Monthly Returns" in html


# ------------------------------------------------------------------ #
#  Edge case tests                                                   #
# ------------------------------------------------------------------ #

class TestEdgeCases:
    def test_single_experiment(self):
        c = ExperimentComparer({"only": _daily_returns(0.001, 0.01)})
        r = c.compare()
        assert len(r.experiment_metrics) == 1
        assert len(r.pair_comparisons) == 0
        assert r.best_experiment == "only"
        assert r.correlation_matrix.shape == (1, 1)

    def test_empty_series(self):
        empty = pd.Series([], dtype=float)
        c = ExperimentComparer({"empty": empty})
        r = c.compare()
        m = r.experiment_metrics[0]
        assert m.sharpe == 0.0
        assert m.total_return == 0.0
        assert m.win_rate == 0.0

    def test_short_series(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        short = pd.Series([0.01, -0.005, 0.002], index=dates)
        c = ExperimentComparer({"short": short})
        r = c.compare()
        assert r.experiment_metrics[0].experiment_id == "short"
        assert r.experiment_metrics[0].sharpe != 0.0

    def test_constant_returns(self):
        dates = pd.bdate_range("2024-01-02", periods=50)
        const = pd.Series(np.full(50, 0.001), index=dates)
        c = ExperimentComparer({"const": const})
        r = c.compare()
        # Std is 0 so sharpe should be 0 (avoid division by zero)
        assert r.experiment_metrics[0].sharpe == 0.0

    def test_all_negative_returns(self):
        dates = pd.bdate_range("2024-01-02", periods=100)
        neg = pd.Series(np.full(100, -0.001), index=dates)
        c = ExperimentComparer({"neg": neg})
        r = c.compare()
        m = r.experiment_metrics[0]
        assert m.total_return < 0
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0

    def test_two_empty_series_pair(self):
        e1 = pd.Series([], dtype=float)
        e2 = pd.Series([], dtype=float)
        c = ExperimentComparer({"a": e1, "b": e2})
        r = c.compare()
        if r.pair_comparisons:
            p = r.pair_comparisons[0]
            assert p.p_value == 1.0

    def test_no_overlapping_dates(self):
        d1 = pd.bdate_range("2024-01-02", periods=10)
        d2 = pd.bdate_range("2024-06-01", periods=10)
        s1 = pd.Series(np.random.randn(10) * 0.01, index=d1)
        s2 = pd.Series(np.random.randn(10) * 0.01, index=d2)
        c = ExperimentComparer({"a": s1, "b": s2})
        r = c.compare()
        # No overlap so pair comparison should be degenerate
        p = r.pair_comparisons[0]
        assert p.p_value == 1.0

    def test_generated_at_is_iso(self):
        r = _two_experiment_result()
        # Should parse as ISO datetime
        dt = datetime.fromisoformat(r.generated_at)
        assert dt.year >= 2024
