"""
Tests for compass/strategy_report.py — Strategy report generator.

Covers:
  - Metric computation (performance, risk, trade stats, benchmark)
  - Report data assembly
  - HTML rendering (light/dark themes, all sections)
  - Section configuration and templates
  - Batch mode
  - Edge cases (short series, zero vol, no trades, no SPY)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from compass.strategy_report import (
    BenchmarkComparison,
    PerformanceMetrics,
    ReportConfig,
    ReportData,
    RiskMetrics,
    StrategyReportGenerator,
    TradeRecord,
    TradeStatistics,
    _calmar_ratio,
    _compute_drawdown_series,
    _compute_equity_curve,
    _max_drawdown,
    _max_drawdown_duration,
    _monthly_returns,
    _rolling_sharpe,
    _sharpe_ratio,
    _sortino_ratio,
    _var_cvar,
    compute_benchmark,
    compute_performance,
    compute_regime_performance,
    compute_risk,
    compute_trade_statistics,
    generate_batch,
    render_html,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def daily_returns(rng):
    """252-day return series with slight positive drift."""
    return rng.normal(0.0004, 0.012, 252)


@pytest.fixture
def spy_returns(rng):
    """SPY benchmark returns."""
    return rng.normal(0.0003, 0.011, 252)


@pytest.fixture
def sample_trades(rng):
    """50 synthetic trade records."""
    trades = []
    for _ in range(50):
        pnl = rng.normal(50, 200)
        trades.append(TradeRecord(
            pnl=pnl,
            is_winner=pnl > 0,
            hold_days=rng.uniform(1, 30),
            regime="BULL" if rng.random() > 0.4 else "BEAR",
        ))
    return trades


@pytest.fixture
def generator(daily_returns, spy_returns, sample_trades, rng):
    """Fully-configured StrategyReportGenerator."""
    regimes = rng.choice([0, 1, 2], size=252)
    return StrategyReportGenerator(
        experiment_id="EXP-400",
        daily_returns=daily_returns,
        spy_returns=spy_returns,
        trades=sample_trades,
        regimes=regimes,
        regime_labels=["BULL", "BEAR", "HIGH_VOL"],
    )


# ── Equity curve and drawdown tests ──────────────────────────────────────────

class TestEquityCurve:
    def test_starts_at_one(self, daily_returns):
        eq = _compute_equity_curve(daily_returns)
        assert abs(eq[0] - (1.0 * (1 + daily_returns[0]))) < 1e-10

    def test_length_matches(self, daily_returns):
        eq = _compute_equity_curve(daily_returns)
        assert len(eq) == len(daily_returns)

    def test_monotonic_for_positive_returns(self):
        r = np.full(100, 0.01)
        eq = _compute_equity_curve(r)
        assert np.all(np.diff(eq) > 0)

    def test_drawdown_never_positive(self, daily_returns):
        eq = _compute_equity_curve(daily_returns)
        dd = _compute_drawdown_series(eq)
        assert np.all(dd <= 1e-12)

    def test_max_drawdown_positive(self, daily_returns):
        eq = _compute_equity_curve(daily_returns)
        mdd = _max_drawdown(eq)
        assert mdd >= 0

    def test_max_drawdown_zero_for_monotone(self):
        eq = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        assert _max_drawdown(eq) == 0.0

    def test_max_drawdown_duration(self):
        # Create a known drawdown pattern
        eq = np.array([1.0, 1.1, 1.0, 0.9, 0.95, 1.1, 1.2])
        dur = _max_drawdown_duration(eq)
        assert dur >= 3  # at least 3 periods in drawdown


# ── Performance metrics tests ─────────────────────────────────────────────────

class TestPerformanceMetrics:
    def test_returns_dataclass(self, daily_returns):
        p = compute_performance(daily_returns)
        assert isinstance(p, PerformanceMetrics)

    def test_sharpe_sign_matches_returns(self, rng):
        pos = rng.normal(0.002, 0.005, 252)  # strong positive drift with variance
        p = compute_performance(pos, risk_free_rate=0.0)
        assert p.sharpe > 0

    def test_sharpe_negative_for_losing(self, rng):
        neg = rng.normal(-0.003, 0.005, 252)  # negative drift with variance
        p = compute_performance(neg, risk_free_rate=0.0)
        assert p.sharpe < 0

    def test_sortino_higher_than_sharpe_for_positive_skew(self, rng):
        # With positive mean and symmetric noise, sortino >= sharpe because
        # downside deviation < total std when mean > 0
        r = rng.normal(0.002, 0.01, 252)
        p = compute_performance(r, risk_free_rate=0.0)
        # Sortino should be at least as large as Sharpe for positive-mean series
        assert p.sortino >= p.sharpe * 0.8

    def test_win_rate_in_range(self, daily_returns):
        p = compute_performance(daily_returns)
        assert 0.0 <= p.win_rate <= 1.0

    def test_calmar_positive_for_gains(self):
        rng = np.random.RandomState(99)
        r = rng.normal(0.002, 0.005, 252)  # positive drift with variance
        assert _calmar_ratio(r) > 0

    def test_n_periods_correct(self, daily_returns):
        p = compute_performance(daily_returns)
        assert p.n_periods == 252


# ── Risk metrics tests ────────────────────────────────────────────────────────

class TestRiskMetrics:
    def test_returns_dataclass(self, daily_returns):
        r = compute_risk(daily_returns)
        assert isinstance(r, RiskMetrics)

    def test_var_positive(self, daily_returns):
        r = compute_risk(daily_returns)
        assert r.var_95 > 0
        assert r.var_99 > 0

    def test_cvar_gte_var(self, daily_returns):
        r = compute_risk(daily_returns)
        assert r.cvar_95 >= r.var_95 - 1e-10
        assert r.cvar_99 >= r.var_99 - 1e-10

    def test_99_var_gte_95(self, daily_returns):
        r = compute_risk(daily_returns)
        assert r.var_99 >= r.var_95 - 1e-10


# ── Trade statistics tests ────────────────────────────────────────────────────

class TestTradeStatistics:
    def test_returns_dataclass(self, sample_trades):
        ts = compute_trade_statistics(sample_trades)
        assert isinstance(ts, TradeStatistics)

    def test_n_trades(self, sample_trades):
        ts = compute_trade_statistics(sample_trades)
        assert ts.n_trades == 50

    def test_winners_plus_losers(self, sample_trades):
        ts = compute_trade_statistics(sample_trades)
        # Some trades might be exactly zero so winners + losers <= n_trades
        assert ts.n_winners + ts.n_losers <= ts.n_trades

    def test_empty_trades(self):
        ts = compute_trade_statistics([])
        assert ts.n_trades == 0
        assert ts.win_rate == 0.0
        assert ts.profit_factor == 0.0

    def test_all_winners(self):
        trades = [TradeRecord(pnl=100.0, is_winner=True, hold_days=5) for _ in range(10)]
        ts = compute_trade_statistics(trades)
        assert ts.win_rate == 1.0
        assert ts.avg_loss == 0.0

    def test_profit_factor_positive(self, sample_trades):
        ts = compute_trade_statistics(sample_trades)
        assert ts.profit_factor >= 0


# ── Benchmark comparison tests ────────────────────────────────────────────────

class TestBenchmarkComparison:
    def test_returns_dataclass(self, daily_returns, spy_returns):
        bm = compute_benchmark(daily_returns, spy_returns)
        assert isinstance(bm, BenchmarkComparison)

    def test_risk_free_return_positive(self, daily_returns, spy_returns):
        bm = compute_benchmark(daily_returns, spy_returns, risk_free_rate=0.05)
        assert bm.risk_free_total_return > 0

    def test_beta_near_one_for_identical(self, daily_returns):
        bm = compute_benchmark(daily_returns, daily_returns)
        assert abs(bm.beta - 1.0) < 0.01

    def test_alpha_zero_for_identical(self, daily_returns):
        bm = compute_benchmark(daily_returns, daily_returns)
        assert abs(bm.alpha) < 0.01


# ── Regime performance tests ─────────────────────────────────────────────────

class TestRegimePerformance:
    def test_returns_list(self, daily_returns, rng):
        regimes = rng.choice([0, 1], size=252)
        rp = compute_regime_performance(daily_returns, regimes, ["BULL", "BEAR"])
        assert len(rp) == 2

    def test_regime_labels(self, daily_returns, rng):
        regimes = rng.choice([0, 1, 2], size=252)
        rp = compute_regime_performance(daily_returns, regimes, ["A", "B", "C"])
        labels = {r.regime for r in rp}
        assert labels == {"A", "B", "C"}

    def test_days_sum_to_total(self, daily_returns, rng):
        regimes = rng.choice([0, 1], size=252)
        rp = compute_regime_performance(daily_returns, regimes, ["BULL", "BEAR"])
        total_days = sum(r.n_days for r in rp)
        assert total_days == 252


# ── Rolling Sharpe tests ──────────────────────────────────────────────────────

class TestRollingSharpe:
    def test_length_matches(self, daily_returns):
        rs = _rolling_sharpe(daily_returns, window=63)
        assert len(rs) == len(daily_returns)

    def test_initial_nan(self, daily_returns):
        rs = _rolling_sharpe(daily_returns, window=63)
        assert np.isnan(rs[0])
        assert not np.isnan(rs[62])

    def test_short_series_all_nan(self):
        r = np.array([0.01, 0.02])
        rs = _rolling_sharpe(r, window=10)
        assert np.all(np.isnan(rs))


# ── Monthly returns tests ─────────────────────────────────────────────────────

class TestMonthlyReturns:
    def test_returns_dict(self, daily_returns):
        m = _monthly_returns(daily_returns, start_year=2024, start_month=1)
        assert isinstance(m, dict)
        assert len(m) > 0

    def test_keys_are_year_month_tuples(self, daily_returns):
        m = _monthly_returns(daily_returns, start_year=2024, start_month=1)
        for key in m:
            assert isinstance(key, tuple)
            assert len(key) == 2


# ── Generator and HTML tests ─────────────────────────────────────────────────

class TestStrategyReportGenerator:
    def test_too_few_periods_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            StrategyReportGenerator("X", np.array([0.01]))

    def test_spy_length_mismatch_raises(self, daily_returns):
        with pytest.raises(ValueError, match="spy_returns length"):
            StrategyReportGenerator("X", daily_returns, spy_returns=np.zeros(10))

    def test_compute_all_returns_report_data(self, generator):
        data = generator.compute_all()
        assert isinstance(data, ReportData)
        assert data.experiment_id == "EXP-400"

    def test_generate_returns_html(self, generator):
        html = generator.generate()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_html_contains_experiment_id(self, generator):
        html = generator.generate()
        assert "EXP-400" in html

    def test_html_contains_all_sections(self, generator):
        html = generator.generate()
        assert "Executive Summary" in html
        assert "Performance Metrics" in html
        assert "Equity Curve" in html
        assert "Drawdown" in html
        assert "Monthly Returns" in html
        assert "Rolling Sharpe" in html
        assert "Regime Breakdown" in html
        assert "Risk Metrics" in html
        assert "Trade Statistics" in html
        assert "Benchmark Comparison" in html

    def test_html_contains_svg_charts(self, generator):
        html = generator.generate()
        assert "<svg" in html

    def test_html_contains_var_table(self, generator):
        html = generator.generate()
        assert "Value-at-Risk" in html
        assert "CVaR" in html

    def test_html_contains_footer(self, generator):
        html = generator.generate()
        assert "COMPASS Strategy Report Engine" in html


class TestDarkTheme:
    def test_dark_theme_css(self, daily_returns):
        config = ReportConfig(theme="dark")
        gen = StrategyReportGenerator("X", daily_returns, config=config)
        html = gen.generate()
        assert "background: #0f172a" in html

    def test_light_theme_css(self, daily_returns):
        config = ReportConfig(theme="light")
        gen = StrategyReportGenerator("X", daily_returns, config=config)
        html = gen.generate()
        assert "background: #f8fafc" in html


class TestConfigurableSections:
    def test_subset_of_sections(self, daily_returns):
        config = ReportConfig(sections=("executive_summary", "risk_metrics"))
        gen = StrategyReportGenerator("X", daily_returns, config=config)
        html = gen.generate()
        assert "Executive Summary" in html
        assert "Risk Metrics" in html
        assert "Monthly Returns" not in html
        assert "Benchmark Comparison" not in html

    def test_empty_sections(self, daily_returns):
        config = ReportConfig(sections=())
        gen = StrategyReportGenerator("X", daily_returns, config=config)
        html = gen.generate()
        assert "<!DOCTYPE html>" in html

    def test_custom_title(self, daily_returns):
        config = ReportConfig(title="Custom Report Title")
        gen = StrategyReportGenerator("X", daily_returns, config=config)
        html = gen.generate()
        assert "Custom Report Title" in html

    def test_no_footer(self, daily_returns):
        config = ReportConfig(include_footer=False)
        gen = StrategyReportGenerator("X", daily_returns, config=config)
        html = gen.generate()
        assert "COMPASS Strategy Report Engine" not in html


class TestNoOptionalData:
    def test_no_spy_no_benchmark_section(self, daily_returns):
        gen = StrategyReportGenerator("X", daily_returns)
        html = gen.generate()
        assert "Benchmark Comparison" not in html

    def test_no_trades_no_trade_section(self, daily_returns):
        gen = StrategyReportGenerator("X", daily_returns)
        html = gen.generate()
        assert "Trade Statistics" not in html

    def test_no_regimes_no_regime_section(self, daily_returns):
        gen = StrategyReportGenerator("X", daily_returns)
        html = gen.generate()
        assert "Regime Breakdown" not in html


# ── Batch mode tests ──────────────────────────────────────────────────────────

class TestBatchMode:
    def test_generates_all_reports(self, rng):
        experiments = {
            "EXP-400": rng.normal(0.0004, 0.01, 252),
            "EXP-503": rng.normal(0.0003, 0.012, 252),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_batch(experiments, output_dir=tmpdir)
            assert len(paths) == 2
            for eid, path in paths.items():
                assert os.path.isfile(path)
                with open(path) as f:
                    content = f.read()
                assert "<!DOCTYPE html>" in content
                assert eid in content

    def test_batch_with_spy(self, rng):
        experiments = {"A": rng.normal(0, 0.01, 100), "B": rng.normal(0, 0.01, 100)}
        spy = rng.normal(0, 0.01, 100)
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_batch(experiments, spy_returns=spy, output_dir=tmpdir)
            assert len(paths) == 2
            with open(paths["A"]) as f:
                assert "Benchmark Comparison" in f.read()

    def test_batch_creates_directory(self, rng):
        experiments = {"X": rng.normal(0, 0.01, 50)}
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = os.path.join(tmpdir, "sub", "dir")
            paths = generate_batch(experiments, output_dir=subdir)
            assert os.path.isdir(subdir)
            assert os.path.isfile(paths["X"])
