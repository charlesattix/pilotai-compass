"""Tests for compass/north_star_integrator.py — master integration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.north_star_integrator import (
    ExperimentContribution,
    ExperimentData,
    IntegratorConfig,
    IntegratorResult,
    MonteCarloResult,
    NorthStarIntegrator,
    Targets,
    TradeResult,
    WalkForwardFold,
    YearMetrics,
    detect_regime,
    kelly_contracts,
    monte_carlo_stress,
    optimise_weights,
    risk_check,
    score_trade,
    _sharpe,
    _sortino,
    _max_dd_pct,
    _pf,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_exp(name: str, n: int = 80, seed: int = 42, win_rate: float = 0.58) -> ExperimentData:
    rng = np.random.RandomState(seed)
    years = rng.choice([2020, 2021, 2022, 2023, 2024], n)
    wins = rng.random(n) < win_rate
    pnl = np.where(wins, rng.uniform(50, 500, n), rng.uniform(-400, -50, n))
    df = pd.DataFrame({
        "entry_date": pd.bdate_range("2020-01-02", periods=n),
        "exit_date": pd.bdate_range("2020-01-03", periods=n),
        "year": years,
        "pnl": pnl,
        "net_credit": rng.uniform(0.5, 3.0, n),
        "strategy_type": rng.choice(["CS", "IC"], n),
        "regime": rng.choice(["bull", "bear", "sideways"], n, p=[0.5, 0.2, 0.3]),
        "vix": rng.uniform(12, 35, n),
        "vix_percentile_50d": rng.uniform(10, 95, n),
        "iv_rank": rng.uniform(5, 90, n),
        "momentum_5d_pct": rng.normal(0, 2, n),
        "momentum_10d_pct": rng.normal(0, 3, n),
        "dte_at_entry": rng.randint(3, 30, n),
        "contracts": 5,
        "win": wins.astype(int),
    })
    return ExperimentData(name=name, trades=df, n_trades=n, years=sorted(df["year"].unique().tolist()))


def _make_experiments() -> list:
    return [_make_exp("EXP-A", 60, 42), _make_exp("EXP-B", 50, 99)]


@pytest.fixture
def experiments():
    return _make_experiments()


@pytest.fixture
def config():
    return IntegratorConfig(
        signal_threshold=0.3,
        mc_paths=500,
        mc_horizon_days=50,
    )


@pytest.fixture
def integrator(config):
    return NorthStarIntegrator(config)


# ── Targets tests ────────────────────────────────────────────────────────


class TestTargets:
    def test_defaults(self):
        t = Targets()
        assert t.annual_return_pct == 100.0
        assert t.max_drawdown_pct == 12.0
        assert t.sharpe_ratio == 6.0

    def test_custom(self):
        t = Targets(annual_return_pct=55.0, max_drawdown_pct=30.0, sharpe_ratio=3.0)
        assert t.annual_return_pct == 55.0


# ── Regime detection tests ───────────────────────────────────────────────


class TestRegime:
    def test_known_regime(self):
        assert detect_regime(pd.Series({"regime": "bull"})) == "bull"
        assert detect_regime(pd.Series({"regime": "bear"})) == "bear"

    def test_infer_from_vix(self):
        assert detect_regime(pd.Series({"regime": "", "vix": 35, "momentum_10d_pct": 0})) == "bear"

    def test_infer_bull(self):
        assert detect_regime(pd.Series({"regime": "", "vix": 15, "momentum_10d_pct": 2})) == "bull"

    def test_fallback_sideways(self):
        assert detect_regime(pd.Series({"regime": "", "vix": 20, "momentum_10d_pct": 0})) == "sideways"


# ── Signal scoring tests ─────────────────────────────────────────────────


class TestSignal:
    def test_bull_higher(self):
        bull = score_trade(pd.Series({"regime": "bull", "vix_percentile_50d": 50, "iv_rank": 50, "momentum_5d_pct": 0, "dte_at_entry": 10}))
        bear = score_trade(pd.Series({"regime": "bear", "vix_percentile_50d": 50, "iv_rank": 50, "momentum_5d_pct": 0, "dte_at_entry": 10}))
        assert bull > bear

    def test_bounded(self):
        s = score_trade(pd.Series({"regime": "bull", "vix_percentile_50d": 95, "iv_rank": 90, "momentum_5d_pct": 3, "dte_at_entry": 10, "win": 1}))
        assert 0 <= s <= 1

    def test_high_vix_bonus(self):
        low = score_trade(pd.Series({"regime": "sideways", "vix_percentile_50d": 20, "iv_rank": 50, "momentum_5d_pct": 0}))
        high = score_trade(pd.Series({"regime": "sideways", "vix_percentile_50d": 80, "iv_rank": 50, "momentum_5d_pct": 0}))
        assert high > low


# ── Kelly sizing tests ───────────────────────────────────────────────────


class TestKelly:
    def test_basic(self, config):
        c = kelly_contracts(100_000, 0.8, 2.0, config)
        assert c >= 1

    def test_higher_signal_more(self, config):
        c_low = kelly_contracts(100_000, 0.55, 2.0, config)
        c_high = kelly_contracts(100_000, 0.95, 2.0, config)
        assert c_high >= c_low

    def test_zero_price(self, config):
        assert kelly_contracts(100_000, 0.7, 0.0, config) == config.base_contracts


# ── Risk gate tests ──────────────────────────────────────────────────────


class TestRiskGate:
    def test_passes(self, config):
        ok, _ = risk_check("bull", 0.05, 0.3, config)
        assert ok

    def test_dd_halt(self, config):
        ok, reason = risk_check("bull", 0.15, 0.0, config)
        assert not ok and "drawdown" in reason

    def test_regime_block(self, config):
        ok, reason = risk_check("bear", 0.01, 0.0, config)
        assert not ok and "regime" in reason

    def test_exposure_limit(self, config):
        ok, reason = risk_check("bull", 0.01, 0.70, config)
        assert not ok and "exposure" in reason


# ── Portfolio optimisation tests ─────────────────────────────────────────


class TestOptimise:
    def test_max_sharpe_sums_one(self):
        pnls = {"A": np.random.RandomState(1).normal(0.5, 1, 100), "B": np.random.RandomState(2).normal(0.3, 1, 100)}
        w = optimise_weights(pnls, "max_sharpe", n_sims=500)
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_equal(self):
        pnls = {"A": np.ones(50), "B": np.ones(50), "C": np.ones(50)}
        w = optimise_weights(pnls, "equal")
        assert abs(w["A"] - 1/3) < 0.01

    def test_risk_parity_sums_one(self):
        pnls = {"A": np.random.RandomState(1).normal(0, 1, 100), "B": np.random.RandomState(2).normal(0, 2, 100)}
        w = optimise_weights(pnls, "risk_parity")
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_erc_sums_one(self):
        pnls = {"A": np.random.RandomState(1).normal(0, 1, 100), "B": np.random.RandomState(2).normal(0, 2, 100)}
        w = optimise_weights(pnls, "erc")
        assert abs(sum(w.values()) - 1.0) < 0.01

    def test_single_experiment(self):
        w = optimise_weights({"A": np.ones(10)}, "max_sharpe")
        assert w["A"] == 1.0

    def test_empty(self):
        assert optimise_weights({}, "max_sharpe") == {}


# ── Monte Carlo stress tests ─────────────────────────────────────────────


class TestMonteCarlo:
    def test_basic(self):
        pnls = np.random.RandomState(42).normal(100, 200, 100)
        mc = monte_carlo_stress(pnls, 100_000, n_paths=500, horizon=50)
        assert isinstance(mc, MonteCarloResult)
        assert mc.n_paths == 500

    def test_median_positive_for_positive_pnl(self):
        pnls = np.random.RandomState(42).normal(200, 100, 200)
        mc = monte_carlo_stress(pnls, 100_000, n_paths=500, horizon=50)
        assert mc.median_return_pct > 0

    def test_p5_less_than_median(self):
        pnls = np.random.RandomState(42).normal(50, 200, 100)
        mc = monte_carlo_stress(pnls, 100_000, n_paths=500, horizon=50)
        assert mc.p5_return_pct <= mc.median_return_pct

    def test_short_data(self):
        mc = monte_carlo_stress(np.array([1.0]), 100_000)
        assert mc.n_paths == 0


# ── Metrics tests ────────────────────────────────────────────────────────


class TestMetrics:
    def test_sharpe_positive(self):
        assert _sharpe(np.array([100, 50, 80, -20, 60])) > 0

    def test_sharpe_short(self):
        assert _sharpe(np.array([1])) == 0.0

    def test_sortino(self):
        assert _sortino(np.array([100, 50, -20, 80])) > 0

    def test_max_dd(self):
        eq = np.array([100, 110, 95, 105])
        assert _max_dd_pct(eq) > 0

    def test_pf(self):
        assert _pf(np.array([100, -50, 200])) == pytest.approx(300/50)


# ── Full integration tests ───────────────────────────────────────────────


class TestIntegration:
    def test_returns_result(self, integrator, experiments):
        result = integrator.run(experiments)
        assert isinstance(result, IntegratorResult)
        assert result.n_trades > 0

    def test_capital_accounting(self, integrator, experiments):
        result = integrator.run(experiments)
        expected = result.initial_capital + result.total_pnl
        assert abs(result.final_capital - expected) < 1.0

    def test_year_metrics(self, integrator, experiments):
        result = integrator.run(experiments)
        assert len(result.year_metrics) > 0

    def test_experiment_contributions(self, integrator, experiments):
        result = integrator.run(experiments)
        assert len(result.experiment_contributions) == 2
        names = {e.name for e in result.experiment_contributions}
        assert "EXP-A" in names and "EXP-B" in names

    def test_portfolio_weights(self, integrator, experiments):
        result = integrator.run(experiments)
        assert len(result.portfolio_weights) > 0
        total = sum(result.portfolio_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_walk_forward(self, integrator, experiments):
        result = integrator.run(experiments)
        assert len(result.walk_forward) > 0

    def test_monte_carlo(self, integrator, experiments):
        result = integrator.run(experiments)
        assert result.monte_carlo.n_paths > 0

    def test_target_assessment(self, integrator, experiments):
        result = integrator.run(experiments)
        assert isinstance(result.return_met, bool)
        assert isinstance(result.sharpe_met, bool)
        assert isinstance(result.dd_met, bool)

    def test_equity_curve(self, integrator, experiments):
        result = integrator.run(experiments)
        assert len(result.equity_curve) == result.n_trades

    def test_monthly_returns(self, integrator, experiments):
        result = integrator.run(experiments)
        assert isinstance(result.monthly_returns, pd.Series)

    def test_empty_experiments(self):
        it = NorthStarIntegrator()
        result = it.run([])
        assert result.n_trades == 0

    def test_all_methods(self, experiments):
        for method in ["max_sharpe", "risk_parity", "erc", "equal"]:
            cfg = IntegratorConfig(opt_method=method, signal_threshold=0.3, mc_paths=100, mc_horizon_days=20)
            result = NorthStarIntegrator(cfg).run(experiments)
            assert result.n_trades > 0

    def test_regime_gate_blocks_all_bear(self, experiments):
        for exp in experiments:
            exp.trades["regime"] = "bear"
        cfg = IntegratorConfig(signal_threshold=0.0, regime_filter=True, allowed_regimes=["bull"])
        result = NorthStarIntegrator(cfg).run(experiments)
        assert result.n_trades == 0

    def test_costs_reduce_pnl(self, integrator, experiments):
        result = integrator.run(experiments)
        total_gross = sum(t.gross_pnl for t in result.trades)
        assert result.total_pnl <= total_gross + 0.01

    def test_win_rate_bounded(self, integrator, experiments):
        result = integrator.run(experiments)
        assert 0 <= result.win_rate <= 1.0


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, integrator, experiments):
        result = integrator.run(experiments)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ns.html"
            path = NorthStarIntegrator.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "North Star" in content

    def test_contains_scorecard(self, integrator, experiments):
        result = integrator.run(experiments)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            NorthStarIntegrator.generate_report(result, out)
            content = out.read_text()
            assert "Scorecard" in content

    def test_contains_charts(self, integrator, experiments):
        result = integrator.run(experiments)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            NorthStarIntegrator.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Equity" in content

    def test_contains_monte_carlo(self, integrator, experiments):
        result = integrator.run(experiments)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            NorthStarIntegrator.generate_report(result, out)
            content = out.read_text()
            assert "Monte Carlo" in content

    def test_contains_walk_forward(self, integrator, experiments):
        result = integrator.run(experiments)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            NorthStarIntegrator.generate_report(result, out)
            content = out.read_text()
            assert "Walk-Forward" in content

    def test_default_path(self, integrator, experiments):
        result = integrator.run(experiments)
        path = NorthStarIntegrator.generate_report(result)
        assert path.exists()
        assert "north_star_integrator.html" in str(path)
        path.unlink(missing_ok=True)
