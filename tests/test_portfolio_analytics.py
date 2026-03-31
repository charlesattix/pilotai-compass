"""Tests for compass.portfolio_analytics module.

Uses deterministic data via np.random.RandomState(seed) for reproducibility.
30+ tests covering all ratio computations, rolling analytics, benchmark
comparison, risk contribution, return tables, drawdown detection, HTML report
generation, edge cases, and dataclasses.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.portfolio_analytics import (
    BenchmarkComparison,
    DrawdownPeriod,
    PortfolioAnalytics,
    PortfolioMetrics,
    ReturnTable,
    RiskContribution,
    RollingResult,
    build_portfolio_metrics,
    build_return_tables,
    compare_benchmark,
    compute_avg_drawdown,
    compute_avg_recovery_days,
    compute_calmar,
    compute_cagr,
    compute_drawdown_series,
    compute_kurtosis,
    compute_max_drawdown,
    compute_omega,
    compute_risk_contributions,
    compute_rolling,
    compute_sharpe,
    compute_skewness,
    compute_sortino,
    compute_total_return,
    compute_volatility,
    detect_drawdowns,
    rolling_correlation,
    rolling_sharpe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_returns() -> pd.Series:
    """252 days of deterministic daily returns."""
    rs = np.random.RandomState(42)
    dates = pd.bdate_range("2024-01-02", periods=252)
    returns = rs.normal(0.0004, 0.012, size=252)
    return pd.Series(returns, index=dates, name="portfolio")


@pytest.fixture
def benchmark_returns() -> pd.Series:
    rs = np.random.RandomState(99)
    dates = pd.bdate_range("2024-01-02", periods=252)
    returns = rs.normal(0.0003, 0.010, size=252)
    return pd.Series(returns, index=dates, name="spy")


@pytest.fixture
def strategy_dict() -> dict[str, pd.Series]:
    dates = pd.bdate_range("2024-01-02", periods=252)
    rs1 = np.random.RandomState(10)
    rs2 = np.random.RandomState(20)
    rs3 = np.random.RandomState(30)
    return {
        "iron_condor": pd.Series(rs1.normal(0.0003, 0.008, 252), index=dates),
        "put_spread": pd.Series(rs2.normal(0.0002, 0.010, 252), index=dates),
        "call_spread": pd.Series(rs3.normal(0.0001, 0.006, 252), index=dates),
    }


@pytest.fixture
def analytics(daily_returns, strategy_dict) -> PortfolioAnalytics:
    return PortfolioAnalytics(
        returns=daily_returns,
        strategy_returns=strategy_dict,
        risk_free_rate=0.05,
    )


# ---------------------------------------------------------------------------
# 1. Total return
# ---------------------------------------------------------------------------

class TestTotalReturn:
    def test_positive_returns(self, daily_returns):
        tr = compute_total_return(daily_returns)
        expected = float((1 + daily_returns).prod() - 1)
        assert abs(tr - expected) < 1e-12

    def test_zero_returns(self):
        s = pd.Series([0.0] * 10, index=pd.bdate_range("2024-01-01", periods=10))
        assert compute_total_return(s) == 0.0


# ---------------------------------------------------------------------------
# 2. CAGR
# ---------------------------------------------------------------------------

class TestCAGR:
    def test_positive(self, daily_returns):
        cagr = compute_cagr(daily_returns)
        # Should be roughly annualised mean * 252
        assert isinstance(cagr, float)
        assert -1 < cagr < 5  # sanity bound

    def test_one_year(self):
        dates = pd.bdate_range("2024-01-02", periods=252)
        ret = pd.Series([0.001] * 252, index=dates)
        cagr = compute_cagr(ret)
        total = (1.001) ** 252
        years = (dates[-1] - dates[0]).days / 365.25
        expected = total ** (1 / years) - 1
        assert abs(cagr - expected) < 1e-6


# ---------------------------------------------------------------------------
# 3. Volatility
# ---------------------------------------------------------------------------

class TestVolatility:
    def test_value(self, daily_returns):
        vol = compute_volatility(daily_returns)
        assert vol > 0
        manual = float(daily_returns.std(ddof=1) * np.sqrt(252))
        assert abs(vol - manual) < 0.02  # close given ann factor estimation

    def test_single_point(self):
        s = pd.Series([0.01], index=pd.to_datetime(["2024-01-01"]))
        assert compute_volatility(s) == 0.0


# ---------------------------------------------------------------------------
# 4. Sharpe
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_deterministic(self, daily_returns):
        sharpe = compute_sharpe(daily_returns, risk_free_rate=0.05)
        assert isinstance(sharpe, float)

    def test_zero_vol(self):
        s = pd.Series([0.01] * 50, index=pd.bdate_range("2024-01-01", periods=50))
        assert compute_sharpe(s) == 0.0

    def test_positive_mean_positive_sharpe(self):
        rs = np.random.RandomState(7)
        s = pd.Series(
            rs.normal(0.002, 0.005, 200),
            index=pd.bdate_range("2024-01-01", periods=200),
        )
        assert compute_sharpe(s) > 0


# ---------------------------------------------------------------------------
# 5. Sortino
# ---------------------------------------------------------------------------

class TestSortino:
    def test_higher_than_sharpe_for_right_skewed(self):
        # All positive returns → downside deviation is zero → inf
        s = pd.Series(
            [0.01] * 5 + [0.02] * 5 + [0.005] * 5,
            index=pd.bdate_range("2024-01-01", periods=15),
        )
        sortino = compute_sortino(s)
        assert sortino == float("inf")

    def test_basic(self, daily_returns):
        sortino = compute_sortino(daily_returns, risk_free_rate=0.05)
        assert isinstance(sortino, float)


# ---------------------------------------------------------------------------
# 6. Calmar
# ---------------------------------------------------------------------------

class TestCalmar:
    def test_basic(self, daily_returns):
        calmar = compute_calmar(daily_returns)
        cagr = compute_cagr(daily_returns)
        mdd = compute_max_drawdown(daily_returns)
        if mdd != 0:
            assert abs(calmar - cagr / abs(mdd)) < 1e-10

    def test_no_drawdown(self):
        s = pd.Series([0.01] * 20, index=pd.bdate_range("2024-01-01", periods=20))
        calmar = compute_calmar(s)
        assert calmar == float("inf")


# ---------------------------------------------------------------------------
# 7. Omega
# ---------------------------------------------------------------------------

class TestOmega:
    def test_positive_mean(self, daily_returns):
        omega = compute_omega(daily_returns)
        assert omega > 0

    def test_all_positive(self):
        s = pd.Series([0.01] * 10, index=pd.bdate_range("2024-01-01", periods=10))
        assert compute_omega(s) == float("inf")

    def test_threshold(self, daily_returns):
        o1 = compute_omega(daily_returns, threshold=0.0)
        o2 = compute_omega(daily_returns, threshold=0.01)
        assert o1 > o2  # higher threshold → lower omega


# ---------------------------------------------------------------------------
# 8. Max Drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_negative(self, daily_returns):
        mdd = compute_max_drawdown(daily_returns)
        assert mdd <= 0

    def test_no_drawdown(self):
        s = pd.Series([0.01] * 10, index=pd.bdate_range("2024-01-01", periods=10))
        assert compute_max_drawdown(s) == 0.0

    def test_known_sequence(self):
        # 1.0 -> 1.1 -> 0.88 -> 0.968 (dd from 1.1 to 0.88 = -20%)
        s = pd.Series(
            [0.10, -0.20, 0.10],
            index=pd.bdate_range("2024-01-01", periods=3),
        )
        mdd = compute_max_drawdown(s)
        assert abs(mdd - (-0.20)) < 1e-10


# ---------------------------------------------------------------------------
# 9. Drawdown detection
# ---------------------------------------------------------------------------

class TestDrawdownDetection:
    def test_returns_list(self, daily_returns):
        periods = detect_drawdowns(daily_returns)
        assert isinstance(periods, list)
        for p in periods:
            assert isinstance(p, DrawdownPeriod)
            assert p.depth < 0

    def test_no_drawdown(self):
        s = pd.Series([0.01] * 10, index=pd.bdate_range("2024-01-01", periods=10))
        assert detect_drawdowns(s) == []

    def test_recovery_days_populated(self, daily_returns):
        periods = detect_drawdowns(daily_returns)
        recovered = [p for p in periods if p.end is not None]
        for p in recovered:
            assert p.recovery_days is not None
            assert p.recovery_days >= 0


# ---------------------------------------------------------------------------
# 10. Average drawdown / recovery
# ---------------------------------------------------------------------------

class TestAvgDrawdownRecovery:
    def test_avg_drawdown(self, daily_returns):
        avg = compute_avg_drawdown(daily_returns)
        assert avg <= 0

    def test_avg_recovery(self, daily_returns):
        avg_rec = compute_avg_recovery_days(daily_returns)
        # May be nan if nothing recovered, but with 252 days we expect recoveries
        assert isinstance(avg_rec, float)


# ---------------------------------------------------------------------------
# 11. Skewness / Kurtosis
# ---------------------------------------------------------------------------

class TestHigherMoments:
    def test_skewness(self, daily_returns):
        sk = compute_skewness(daily_returns)
        assert isinstance(sk, float)
        assert -3 < sk < 3

    def test_kurtosis(self, daily_returns):
        ku = compute_kurtosis(daily_returns)
        assert isinstance(ku, float)

    def test_small_series(self):
        s = pd.Series([0.01, 0.02], index=pd.bdate_range("2024-01-01", periods=2))
        assert compute_skewness(s) == 0.0
        assert compute_kurtosis(s) == 0.0


# ---------------------------------------------------------------------------
# 12. build_portfolio_metrics
# ---------------------------------------------------------------------------

class TestPortfolioMetrics:
    def test_all_fields_set(self, daily_returns):
        m = build_portfolio_metrics(daily_returns, risk_free_rate=0.05)
        assert isinstance(m, PortfolioMetrics)
        for f in [
            "total_return", "cagr", "sharpe", "sortino", "calmar",
            "omega", "max_drawdown", "avg_drawdown", "avg_recovery_days",
            "volatility", "skewness", "kurtosis",
        ]:
            assert hasattr(m, f)


# ---------------------------------------------------------------------------
# 13. Rolling Sharpe
# ---------------------------------------------------------------------------

class TestRollingSharpe:
    def test_length(self, daily_returns):
        rs = rolling_sharpe(daily_returns, 30)
        assert len(rs) == len(daily_returns) - 30 + 1

    def test_windows(self, daily_returns):
        rs30 = rolling_sharpe(daily_returns, 30)
        rs60 = rolling_sharpe(daily_returns, 60)
        assert len(rs30) > len(rs60)


# ---------------------------------------------------------------------------
# 14. Rolling Correlation
# ---------------------------------------------------------------------------

class TestRollingCorrelation:
    def test_length(self, daily_returns, benchmark_returns):
        rc = rolling_correlation(daily_returns, benchmark_returns, 30)
        assert len(rc) > 0
        assert all(-1 <= v <= 1 for v in rc.values)


# ---------------------------------------------------------------------------
# 15. compute_rolling (combined)
# ---------------------------------------------------------------------------

class TestComputeRolling:
    def test_with_benchmark(self, daily_returns, benchmark_returns):
        rr = compute_rolling(daily_returns, benchmark_returns, 0.05)
        assert isinstance(rr, RollingResult)
        assert rr.rolling_corr_30 is not None
        assert rr.rolling_corr_60 is not None
        assert rr.rolling_corr_90 is not None

    def test_without_benchmark(self, daily_returns):
        rr = compute_rolling(daily_returns, None, 0.0)
        assert rr.rolling_corr_30 is None


# ---------------------------------------------------------------------------
# 16. Benchmark comparison
# ---------------------------------------------------------------------------

class TestBenchmarkComparison:
    def test_fields(self, daily_returns, benchmark_returns):
        bc = compare_benchmark(daily_returns, benchmark_returns, "SPY", 0.05)
        assert isinstance(bc, BenchmarkComparison)
        assert bc.name == "SPY"
        assert isinstance(bc.alpha, float)
        assert isinstance(bc.beta, float)
        assert -1 <= bc.correlation <= 1

    def test_tracking_error_positive(self, daily_returns, benchmark_returns):
        bc = compare_benchmark(daily_returns, benchmark_returns)
        assert bc.tracking_error >= 0

    def test_same_series(self, daily_returns):
        bc = compare_benchmark(daily_returns, daily_returns, "Self")
        assert abs(bc.correlation - 1.0) < 1e-10
        assert abs(bc.beta - 1.0) < 1e-10
        assert abs(bc.excess_return) < 1e-10


# ---------------------------------------------------------------------------
# 17. Risk contribution
# ---------------------------------------------------------------------------

class TestRiskContribution:
    def test_output(self, strategy_dict):
        contribs = compute_risk_contributions(strategy_dict)
        assert len(contribs) == 3
        for rc in contribs:
            assert isinstance(rc, RiskContribution)
            assert rc.standalone_vol > 0

    def test_percent_sums_to_one(self, strategy_dict):
        contribs = compute_risk_contributions(strategy_dict)
        total = sum(rc.percent_contribution for rc in contribs)
        assert abs(total - 1.0) < 1e-10

    def test_empty(self):
        assert compute_risk_contributions({}) == []


# ---------------------------------------------------------------------------
# 18. Return tables
# ---------------------------------------------------------------------------

class TestReturnTables:
    def test_structure(self, daily_returns):
        rt = build_return_tables(daily_returns)
        assert isinstance(rt, ReturnTable)
        assert isinstance(rt.monthly, pd.DataFrame)
        assert isinstance(rt.quarterly, pd.DataFrame)
        assert isinstance(rt.annual, pd.Series)

    def test_monthly_columns(self, daily_returns):
        rt = build_return_tables(daily_returns)
        # Should have month abbreviation columns
        assert "Jan" in rt.monthly.columns or "Feb" in rt.monthly.columns

    def test_annual_value(self):
        dates = pd.bdate_range("2024-01-02", periods=252)
        ret = pd.Series([0.001] * 252, index=dates)
        rt = build_return_tables(ret)
        annual_ret = rt.annual.iloc[0]
        expected = (1.001) ** 252 - 1
        assert abs(annual_ret - expected) < 1e-6


# ---------------------------------------------------------------------------
# 19. Drawdown series
# ---------------------------------------------------------------------------

class TestDrawdownSeries:
    def test_max_matches(self, daily_returns):
        dd_series = compute_drawdown_series(daily_returns)
        mdd = compute_max_drawdown(daily_returns)
        assert abs(dd_series.min() - mdd) < 1e-12

    def test_always_leq_zero(self, daily_returns):
        dd = compute_drawdown_series(daily_returns)
        assert (dd <= 1e-14).all()


# ---------------------------------------------------------------------------
# 20. PortfolioAnalytics class
# ---------------------------------------------------------------------------

class TestPortfolioAnalyticsClass:
    def test_metrics(self, analytics):
        m = analytics.metrics()
        assert isinstance(m, PortfolioMetrics)

    def test_rolling(self, analytics, benchmark_returns):
        rr = analytics.rolling(benchmark=benchmark_returns)
        assert isinstance(rr, RollingResult)

    def test_compare(self, analytics, benchmark_returns):
        comps = analytics.compare({"SPY": benchmark_returns})
        assert len(comps) == 1
        assert comps[0].name == "SPY"

    def test_risk_contributions(self, analytics):
        rc = analytics.risk_contributions()
        assert len(rc) == 3

    def test_return_tables(self, analytics):
        rt = analytics.return_tables()
        assert isinstance(rt, ReturnTable)

    def test_drawdowns(self, analytics):
        dd = analytics.drawdowns()
        assert isinstance(dd, list)

    def test_equity_curve(self, analytics):
        ec = analytics.equity_curve()
        assert len(ec) == 252
        assert ec.iloc[0] == 1 + analytics.returns.iloc[0]


# ---------------------------------------------------------------------------
# 21. HTML report generation
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_generates_file(self, analytics, benchmark_returns):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = analytics.generate_report(
                output_path=Path(tmpdir) / "report.html",
                benchmarks={"SPY": benchmark_returns},
            )
            assert path.exists()
            assert path.suffix == ".html"

    def test_self_contained(self, analytics):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = analytics.generate_report(
                output_path=Path(tmpdir) / "report.html",
            )
            content = path.read_text(encoding="utf-8")
            assert "<style>" in content
            assert "<svg" in content
            assert "<!DOCTYPE html>" in content

    def test_dark_theme(self, analytics):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = analytics.generate_report(
                output_path=Path(tmpdir) / "report.html",
            )
            content = path.read_text(encoding="utf-8")
            assert "#1a1a2e" in content  # dark background

    def test_all_sections_present(self, analytics, benchmark_returns):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = analytics.generate_report(
                output_path=Path(tmpdir) / "report.html",
                benchmarks={"SPY": benchmark_returns},
            )
            content = path.read_text(encoding="utf-8")
            assert "Portfolio Metrics" in content
            assert "Equity Curve" in content
            assert "Drawdown" in content
            assert "Benchmark Comparison" in content
            assert "Risk Contribution" in content
            assert "Monthly Returns" in content
            assert "Quarterly Returns" in content
            assert "Annual Returns" in content

    def test_default_path(self, analytics):
        # Uses cwd default
        path = analytics.generate_report()
        try:
            assert path.exists()
            assert path.name == "portfolio_report.html"
        finally:
            path.unlink(missing_ok=True)

    def test_creates_parent_dirs(self, analytics):
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "report.html"
            path = analytics.generate_report(output_path=nested)
            assert path.exists()

    def test_returns_path(self, analytics):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = analytics.generate_report(
                output_path=Path(tmpdir) / "report.html",
            )
            assert isinstance(path, Path)

    def test_svg_charts(self, analytics):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = analytics.generate_report(
                output_path=Path(tmpdir) / "report.html",
            )
            content = path.read_text(encoding="utf-8")
            assert content.count("<svg") >= 2  # equity + drawdown


# ---------------------------------------------------------------------------
# 22. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_constant_returns(self):
        s = pd.Series([0.001] * 100, index=pd.bdate_range("2024-01-01", periods=100))
        m = build_portfolio_metrics(s)
        assert m.max_drawdown == 0.0
        assert m.sharpe == 0.0  # zero std → guard returns 0

    def test_single_large_loss(self):
        rs = np.random.RandomState(55)
        ret = list(rs.normal(0.001, 0.005, 99)) + [-0.15]
        dates = pd.bdate_range("2024-01-01", periods=100)
        s = pd.Series(ret, index=dates)
        mdd = compute_max_drawdown(s)
        assert mdd < -0.10

    def test_all_negative_returns(self):
        s = pd.Series(
            [-0.005] * 50,
            index=pd.bdate_range("2024-01-01", periods=50),
        )
        m = build_portfolio_metrics(s)
        assert m.total_return < 0
        assert m.cagr < 0
        assert m.max_drawdown < 0

    def test_very_short_series(self):
        s = pd.Series([0.01, -0.01], index=pd.bdate_range("2024-01-01", periods=2))
        m = build_portfolio_metrics(s)
        assert isinstance(m, PortfolioMetrics)


# ---------------------------------------------------------------------------
# 23. Dataclass immutability / field access
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_drawdown_period_fields(self):
        dp = DrawdownPeriod(
            start=pd.Timestamp("2024-01-01"),
            trough=pd.Timestamp("2024-01-15"),
            end=pd.Timestamp("2024-02-01"),
            depth=-0.05,
            recovery_days=17,
        )
        assert dp.depth == -0.05
        assert dp.recovery_days == 17

    def test_portfolio_metrics_fields(self):
        m = PortfolioMetrics(
            total_return=0.1, cagr=0.1, sharpe=1.0, sortino=1.5,
            calmar=2.0, omega=1.3, max_drawdown=-0.05,
            avg_drawdown=-0.02, avg_recovery_days=10.0,
            volatility=0.15, skewness=0.1, kurtosis=0.5,
        )
        assert m.sharpe == 1.0
        assert m.calmar == 2.0

    def test_risk_contribution_fields(self):
        rc = RiskContribution(
            strategy="test", weight=0.5, marginal_contribution=0.01,
            percent_contribution=0.3, standalone_vol=0.15,
        )
        assert rc.strategy == "test"


# ---------------------------------------------------------------------------
# 24. Multiple benchmarks
# ---------------------------------------------------------------------------

class TestMultipleBenchmarks:
    def test_three_benchmarks(self, daily_returns):
        rs1 = np.random.RandomState(100)
        rs2 = np.random.RandomState(200)
        rs3 = np.random.RandomState(300)
        dates = daily_returns.index
        benchmarks = {
            "SPY": pd.Series(rs1.normal(0.0003, 0.01, len(dates)), index=dates),
            "60/40": pd.Series(rs2.normal(0.0002, 0.007, len(dates)), index=dates),
            "Risk-Free": pd.Series([0.05 / 252] * len(dates), index=dates),
        }
        pa = PortfolioAnalytics(daily_returns)
        comps = pa.compare(benchmarks)
        assert len(comps) == 3
        names = {c.name for c in comps}
        assert names == {"SPY", "60/40", "Risk-Free"}
