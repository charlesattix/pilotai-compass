"""Tests for compass/mc_portfolio_optimizer.py — MC portfolio optimization."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.mc_portfolio_optimizer import (
    EfficientFrontier,
    MCPortfolioOptimizer,
    OptimizationResult,
    PortfolioMetrics,
    RegimeAllocation,
    _build_html,
    _fmt_pct,
    _fmt_ratio,
    _weights_table,
)

ROOT = Path(__file__).resolve().parent.parent


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_returns(
    n_days: int = 500, n_assets: int = 3, seed: int = 42
) -> pd.DataFrame:
    """Generate synthetic daily returns."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    data = rng.normal(0.0003, 0.01, (n_days, n_assets))
    cols = [f"Strategy_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=cols)


def _make_regimes(returns: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Generate synthetic regime labels aligned with returns."""
    rng = np.random.RandomState(seed)
    labels = rng.choice(["bull", "bear", "sideways"], size=len(returns))
    return pd.Series(labels, index=returns.index, name="regime")


@pytest.fixture
def returns_3a():
    return _make_returns(500, 3, seed=42)


@pytest.fixture
def returns_5a():
    return _make_returns(600, 5, seed=99)


@pytest.fixture
def regimes(returns_3a):
    return _make_regimes(returns_3a)


@pytest.fixture
def optimizer(returns_3a):
    return MCPortfolioOptimizer(returns_3a, n_simulations=500, seed=42)


@pytest.fixture
def optimizer_with_regimes(returns_3a, regimes):
    return MCPortfolioOptimizer(
        returns_3a, regimes=regimes, n_simulations=500, seed=42
    )


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_basic_init(self, returns_3a):
        opt = MCPortfolioOptimizer(returns_3a, seed=1)
        assert opt.n_assets == 3
        assert opt.n_simulations == 10_000
        assert opt.asset_names == ["Strategy_0", "Strategy_1", "Strategy_2"]

    def test_empty_returns_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            MCPortfolioOptimizer(pd.DataFrame())

    def test_single_asset_raises(self):
        df = pd.DataFrame({"A": [0.01, 0.02, -0.01]})
        with pytest.raises(ValueError, match="at least 2 assets"):
            MCPortfolioOptimizer(df)

    def test_custom_simulations(self, returns_3a):
        opt = MCPortfolioOptimizer(returns_3a, n_simulations=100, seed=1)
        assert opt.n_simulations == 100

    def test_custom_risk_free_rate(self, returns_3a):
        opt = MCPortfolioOptimizer(returns_3a, risk_free_rate=0.03, seed=1)
        assert opt.risk_free_rate == 0.03


# ── Weight generation tests ──────────────────────────────────────────────


class TestWeightGeneration:
    def test_weights_sum_to_one(self, optimizer):
        rng = np.random.RandomState(42)
        weights = optimizer._generate_random_weights(rng)
        assert weights.shape == (500, 3)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10)

    def test_weights_non_negative(self, optimizer):
        rng = np.random.RandomState(42)
        weights = optimizer._generate_random_weights(rng)
        assert (weights >= 0).all()

    def test_weights_reproducible(self, optimizer):
        w1 = optimizer._generate_random_weights(np.random.RandomState(42))
        w2 = optimizer._generate_random_weights(np.random.RandomState(42))
        np.testing.assert_array_equal(w1, w2)


# ── Risk metric tests ───────────────────────────────────────────────────


class TestRiskMetrics:
    def test_sharpe_positive(self):
        rng = np.random.RandomState(42)
        rets = rng.normal(0.002, 0.01, 100)
        sharpe = MCPortfolioOptimizer.compute_sharpe(rets, 0.0)
        assert sharpe > 0

    def test_sharpe_zero_vol(self):
        rets = np.array([0.0005] * 50)
        sharpe = MCPortfolioOptimizer.compute_sharpe(rets)
        # constant returns → zero std (ddof=1 with identical values still 0)
        assert sharpe == 0.0

    def test_sharpe_short_series(self):
        assert MCPortfolioOptimizer.compute_sharpe(np.array([0.01])) == 0.0

    def test_sortino_positive(self):
        rets = np.array([0.01, 0.02, -0.005, 0.015, 0.01] * 20)
        sortino = MCPortfolioOptimizer.compute_sortino(rets, 0.0)
        assert sortino > 0

    def test_sortino_no_downside(self):
        rets = np.array([0.01, 0.02, 0.03, 0.04])
        sortino = MCPortfolioOptimizer.compute_sortino(rets, 0.0)
        assert sortino == float("inf")

    def test_sortino_short_series(self):
        assert MCPortfolioOptimizer.compute_sortino(np.array([0.01])) == 0.0

    def test_max_drawdown_negative(self):
        rets = np.array([0.05, -0.1, -0.1, 0.05, 0.05])
        dd = MCPortfolioOptimizer.compute_max_drawdown(rets)
        assert dd < 0

    def test_max_drawdown_empty(self):
        assert MCPortfolioOptimizer.compute_max_drawdown(np.array([])) == 0.0

    def test_max_drawdown_all_positive(self):
        rets = np.array([0.01, 0.01, 0.01])
        dd = MCPortfolioOptimizer.compute_max_drawdown(rets)
        assert dd == 0.0

    def test_cvar_negative(self):
        rng = np.random.RandomState(42)
        rets = rng.normal(0, 0.02, 1000)
        cvar = MCPortfolioOptimizer.compute_cvar(rets, 0.05)
        assert cvar < 0

    def test_cvar_empty(self):
        assert MCPortfolioOptimizer.compute_cvar(np.array([])) == 0.0

    def test_cvar_worse_than_var(self):
        rng = np.random.RandomState(42)
        rets = rng.normal(0, 0.02, 1000)
        cvar = MCPortfolioOptimizer.compute_cvar(rets, 0.05)
        var = np.percentile(rets, 5)
        assert cvar <= var


# ── Portfolio metrics tests ──────────────────────────────────────────────


class TestPortfolioMetrics:
    def test_metrics_to_dict(self):
        pm = PortfolioMetrics(
            weights=np.array([0.5, 0.5]),
            annual_return=0.10,
            annual_volatility=0.15,
            sharpe_ratio=0.67,
            sortino_ratio=0.9,
            max_drawdown=-0.12,
            cvar_95=-0.025,
            calmar_ratio=0.83,
        )
        d = pm.to_dict()
        assert d["weights"] == [0.5, 0.5]
        assert d["annual_return"] == 0.10
        assert "sharpe_ratio" in d

    def test_compute_portfolio_metrics(self, optimizer, returns_3a):
        weights = np.array([0.4, 0.3, 0.3])
        pm = optimizer._compute_portfolio_metrics(weights, returns_3a.values)
        assert isinstance(pm, PortfolioMetrics)
        assert abs(sum(pm.weights) - 1.0) < 1e-10
        assert pm.annual_volatility > 0


# ── Simulation tests ─────────────────────────────────────────────────────


class TestSimulation:
    def test_simulate_returns_correct_count(self, optimizer, returns_3a):
        portfolios = optimizer._simulate(returns_3a.values)
        assert len(portfolios) == 500

    def test_simulate_reproducible(self, returns_3a):
        opt1 = MCPortfolioOptimizer(returns_3a, n_simulations=100, seed=42)
        opt2 = MCPortfolioOptimizer(returns_3a, n_simulations=100, seed=42)
        p1 = opt1._simulate(returns_3a.values)
        p2 = opt2._simulate(returns_3a.values)
        assert p1[0].sharpe_ratio == p2[0].sharpe_ratio


# ── Efficient frontier tests ─────────────────────────────────────────────


class TestEfficientFrontier:
    def test_frontier_built(self, optimizer):
        result = optimizer.optimize()
        ef = result.efficient_frontier
        assert isinstance(ef, EfficientFrontier)
        assert len(ef.portfolios) > 0

    def test_frontier_sorted_by_vol(self, optimizer):
        result = optimizer.optimize()
        ef = result.efficient_frontier
        vols = list(ef.volatilities)
        assert vols == sorted(vols)

    def test_max_sharpe_portfolio(self, optimizer):
        result = optimizer.optimize()
        msp = result.efficient_frontier.max_sharpe_portfolio
        assert isinstance(msp, PortfolioMetrics)

    def test_min_vol_portfolio(self, optimizer):
        result = optimizer.optimize()
        mvp = result.efficient_frontier.min_volatility_portfolio
        assert isinstance(mvp, PortfolioMetrics)
        assert mvp.annual_volatility <= result.efficient_frontier.max_sharpe_portfolio.annual_volatility + 0.01


# ── Full optimize tests ─────────────────────────────────────────────────


class TestOptimize:
    def test_optimize_returns_result(self, optimizer):
        result = optimizer.optimize()
        assert isinstance(result, OptimizationResult)
        assert result.n_simulations == 500
        assert len(result.asset_names) == 3

    def test_best_sharpe_found(self, optimizer):
        result = optimizer.optimize()
        best = result.best_sharpe
        all_sharpes = [p.sharpe_ratio for p in result.all_portfolios]
        assert best.sharpe_ratio == max(all_sharpes)

    def test_best_sortino_found(self, optimizer):
        result = optimizer.optimize()
        best = result.best_sortino
        all_sortinos = [p.sortino_ratio for p in result.all_portfolios]
        assert best.sortino_ratio == max(all_sortinos)

    def test_five_assets(self, returns_5a):
        opt = MCPortfolioOptimizer(returns_5a, n_simulations=200, seed=1)
        result = opt.optimize()
        assert len(result.asset_names) == 5
        assert len(result.best_sharpe.weights) == 5


# ── Regime allocation tests ──────────────────────────────────────────────


class TestRegimeAllocation:
    def test_regime_allocations_present(self, optimizer_with_regimes):
        result = optimizer_with_regimes.optimize()
        assert len(result.regime_allocations) > 0

    def test_regime_keys(self, optimizer_with_regimes):
        result = optimizer_with_regimes.optimize()
        for regime in result.regime_allocations:
            assert regime in ("bull", "bear", "sideways")

    def test_regime_weights_sum_to_one(self, optimizer_with_regimes):
        result = optimizer_with_regimes.optimize()
        for alloc in result.regime_allocations.values():
            np.testing.assert_allclose(alloc.optimal_weights.sum(), 1.0, atol=1e-10)

    def test_no_regimes_returns_empty(self, optimizer):
        result = optimizer.optimize()
        assert result.regime_allocations == {}

    def test_regime_n_periods(self, optimizer_with_regimes, regimes):
        result = optimizer_with_regimes.optimize()
        total = sum(a.n_periods for a in result.regime_allocations.values())
        assert total == len(regimes)


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generate_report_creates_file(self, optimizer):
        result = optimizer.optimize()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_report.html"
            path = MCPortfolioOptimizer.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Monte Carlo Portfolio Optimization" in content

    def test_report_contains_assets(self, optimizer):
        result = optimizer.optimize()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MCPortfolioOptimizer.generate_report(result, out)
            content = out.read_text()
            for name in result.asset_names:
                assert name in content

    def test_report_contains_metrics(self, optimizer):
        result = optimizer.optimize()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MCPortfolioOptimizer.generate_report(result, out)
            content = out.read_text()
            assert "Sharpe" in content
            assert "Sortino" in content
            assert "CVaR" in content

    def test_report_contains_svg(self, optimizer):
        result = optimizer.optimize()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MCPortfolioOptimizer.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content

    def test_report_with_regimes(self, optimizer_with_regimes):
        result = optimizer_with_regimes.optimize()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MCPortfolioOptimizer.generate_report(result, out)
            content = out.read_text()
            assert "Regime-Conditional" in content

    def test_fmt_pct(self):
        assert _fmt_pct(0.1234) == "12.34%"
        assert _fmt_pct(-0.05) == "-5.00%"

    def test_fmt_ratio(self):
        assert _fmt_ratio(1.234) == "1.234"
        assert _fmt_ratio(float("inf")) == "∞"

    def test_weights_table_html(self):
        html = _weights_table(["A", "B"], np.array([0.6, 0.4]))
        assert "<table" in html
        assert "A" in html
        assert "60.00%" in html

    def test_default_output_path(self, optimizer):
        result = optimizer.optimize()
        path = MCPortfolioOptimizer.generate_report(result)
        assert path.exists()
        assert "mc_portfolio.html" in str(path)
        # Clean up
        path.unlink(missing_ok=True)
