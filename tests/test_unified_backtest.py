"""Tests for compass/unified_backtest.py — unified backtesting engine."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.unified_backtest import (
    BacktestConfig,
    BacktestResult,
    ExperimentSummary,
    RegimeBenchmark,
    RollingMetrics,
    TradeResult,
    UnifiedBacktester,
    WalkForwardFold,
    aggregate_experiments,
    apply_signal_filter,
    apply_slippage,
    check_risk_gates,
    compute_features,
    compute_max_drawdown,
    compute_position_size,
    compute_profit_factor,
    compute_regime_benchmarks,
    compute_rolling_metrics,
    compute_sharpe,
    compute_sortino,
    compute_trade_costs,
    load_config,
    walk_forward_validate,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_trades(n: int = 100, seed: int = 42, win_rate: float = 0.55) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n * 2)
    entry_dates = dates[::2][:n]
    exit_dates = dates[1::2][:n]

    wins = rng.random(n) < win_rate
    pnl = np.where(wins, rng.uniform(50, 500, n), rng.uniform(-400, -50, n))
    regimes = rng.choice(["bull", "bear", "sideways"], n)

    return pd.DataFrame({
        "entry_date": entry_dates,
        "exit_date": exit_dates,
        "pnl": pnl,
        "entry_price": rng.uniform(2.0, 8.0, n),
        "net_credit": rng.uniform(0.5, 3.0, n),
        "strategy_type": rng.choice(["CS", "IC", "SS"], n),
        "regime": regimes,
        "vix": rng.uniform(12, 35, n),
        "contracts": 5,
        "win": wins.astype(int),
    })


@pytest.fixture
def trades():
    return _make_trades()


@pytest.fixture
def config():
    return BacktestConfig(
        name="test_exp",
        signal_threshold=0.3,
        slippage_bps=5.0,
        commission_per_contract=1.30,
        max_drawdown_pct=0.20,
        wf_n_folds=2,
    )


@pytest.fixture
def backtester(config):
    return UnifiedBacktester(config)


# ── Config tests ─────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.slippage_bps == 5.0

    def test_load_from_dict(self):
        cfg = load_config({"name": "my_exp", "slippage_bps": 10.0})
        assert cfg.name == "my_exp"
        assert cfg.slippage_bps == 10.0

    def test_load_none(self):
        cfg = load_config(None)
        assert isinstance(cfg, BacktestConfig)

    def test_load_config_object(self):
        original = BacktestConfig(name="orig")
        loaded = load_config(original)
        assert loaded.name == "orig"

    def test_load_ignores_extra_keys(self):
        cfg = load_config({"name": "x", "unknown_key": 42})
        assert cfg.name == "x"


# ── Feature computation tests ────────────────────────────────────────────


class TestFeatures:
    def test_adds_regime(self):
        df = pd.DataFrame({"pnl": [100, -50]})
        result = compute_features(df)
        assert "regime" in result.columns
        assert (result["regime"] == "unknown").all()

    def test_preserves_existing_regime(self, trades):
        result = compute_features(trades)
        assert "regime" in result.columns
        assert not (result["regime"] == "unknown").all()

    def test_adds_signal_score(self, trades):
        result = compute_features(trades)
        assert "signal_score" in result.columns


# ── Signal filter tests ──────────────────────────────────────────────────


class TestSignalFilter:
    def test_filters_below_threshold(self):
        df = pd.DataFrame({"signal_score": [0.1, 0.5, 0.9], "pnl": [1, 2, 3]})
        result = apply_signal_filter(df, 0.5)
        assert len(result) == 2

    def test_no_signal_column_passes_all(self):
        df = pd.DataFrame({"pnl": [1, 2, 3]})
        result = apply_signal_filter(df, 0.5)
        assert len(result) == 3


# ── Position sizing tests ────────────────────────────────────────────────


class TestPositionSizing:
    def test_fixed_frac(self):
        contracts, pct = compute_position_size(100_000, 0.7, "fixed_frac", 0.05, 5, 5.0)
        assert contracts >= 1
        assert pct > 0

    def test_kelly_scales_with_signal(self):
        _, pct_low = compute_position_size(100_000, 0.55, "kelly", 0.10, 5, 5.0)
        _, pct_high = compute_position_size(100_000, 0.90, "kelly", 0.10, 5, 5.0)
        assert pct_high >= pct_low

    def test_risk_parity(self):
        contracts, pct = compute_position_size(100_000, 0.7, "risk_parity", 0.05, 5, 5.0)
        assert contracts >= 1

    def test_zero_price(self):
        contracts, _ = compute_position_size(100_000, 0.7, "fixed_frac", 0.05, 5, 0.0)
        assert contracts == 5  # fallback to base


# ── Risk gate tests ──────────────────────────────────────────────────────


class TestRiskGates:
    def test_passes_normal(self, config):
        passed, reason = check_risk_gates(0.05, 0.3, "bull", config)
        assert passed is True
        assert reason == ""

    def test_fails_drawdown(self, config):
        passed, reason = check_risk_gates(0.25, 0.1, "bull", config)
        assert passed is False
        assert "drawdown" in reason

    def test_fails_exposure(self, config):
        passed, reason = check_risk_gates(0.05, 0.6, "bull", config)
        assert passed is False
        assert "exposure" in reason

    def test_fails_regime(self, config):
        passed, reason = check_risk_gates(0.05, 0.1, "bear", config)
        assert passed is False
        assert "regime" in reason

    def test_regime_gate_disabled(self):
        cfg = BacktestConfig(regime_gate_enabled=False)
        passed, _ = check_risk_gates(0.05, 0.1, "bear", cfg)
        assert passed is True


# ── Execution tests ──────────────────────────────────────────────────────


class TestExecution:
    def test_buy_slippage_increases_price(self):
        p = apply_slippage(100.0, "buy", 5.0)
        assert p > 100.0

    def test_sell_slippage_decreases_price(self):
        p = apply_slippage(100.0, "sell", 5.0)
        assert p < 100.0

    def test_trade_costs_positive(self):
        slip, comm = compute_trade_costs(5, 5.0, 5.5, 5.0, 1.30)
        assert slip > 0
        assert comm > 0

    def test_commission_scales_with_contracts(self):
        _, c1 = compute_trade_costs(1, 5.0, 5.5, 5.0, 1.30)
        _, c5 = compute_trade_costs(5, 5.0, 5.5, 5.0, 1.30)
        assert c5 == c1 * 5


# ── Metrics tests ────────────────────────────────────────────────────────


class TestMetrics:
    def test_sharpe_positive(self):
        pnls = np.array([100, 50, 80, -20, 60, 40, 30])
        assert compute_sharpe(pnls) > 0

    def test_sharpe_short(self):
        assert compute_sharpe(np.array([100])) == 0.0

    def test_sortino_positive(self):
        pnls = np.array([100, 50, 80, -20, 60, 40, 30])
        assert compute_sortino(pnls) > 0

    def test_sortino_no_losses(self):
        assert compute_sortino(np.array([100, 50, 80])) == float("inf")

    def test_max_drawdown_negative(self):
        equity = np.array([100, 110, 105, 95, 100])
        dd = compute_max_drawdown(equity)
        assert dd < 0

    def test_max_drawdown_no_dd(self):
        equity = np.array([100, 110, 120, 130])
        assert compute_max_drawdown(equity) == 0.0

    def test_profit_factor(self):
        pnls = np.array([100, -50, 200, -30])
        pf = compute_profit_factor(pnls)
        assert pf == pytest.approx(300.0 / 80.0)

    def test_profit_factor_no_losses(self):
        assert compute_profit_factor(np.array([100, 50])) == float("inf")


# ── Rolling metrics tests ────────────────────────────────────────────────


class TestRollingMetrics:
    def test_length(self, backtester, trades):
        result = backtester.run(trades)
        assert len(result.rolling_metrics) == result.n_trades

    def test_cumulative_return(self, backtester, trades):
        result = backtester.run(trades)
        if result.rolling_metrics:
            last = result.rolling_metrics[-1]
            assert abs(last.cumulative_return - result.total_return) < 0.01


# ── Regime benchmark tests ───────────────────────────────────────────────


class TestRegimeBenchmarks:
    def test_all_regimes_covered(self, backtester, trades):
        result = backtester.run(trades)
        regimes = {r.regime for r in result.regime_benchmarks}
        trade_regimes = {t.regime for t in result.trades}
        assert regimes == trade_regimes

    def test_trade_counts_sum(self, backtester, trades):
        result = backtester.run(trades)
        total = sum(r.n_trades for r in result.regime_benchmarks)
        assert total == result.n_trades


# ── Walk-forward tests ───────────────────────────────────────────────────


class TestWalkForward:
    def test_folds_created(self, backtester, trades):
        result = backtester.run(trades)
        assert len(result.walk_forward_folds) > 0

    def test_fold_structure(self, backtester, trades):
        result = backtester.run(trades)
        for f in result.walk_forward_folds:
            assert isinstance(f, WalkForwardFold)
            assert f.n_train_trades > 0
            assert f.n_test_trades > 0

    def test_expanding_window(self, backtester, trades):
        result = backtester.run(trades)
        if len(result.walk_forward_folds) >= 2:
            assert result.walk_forward_folds[1].n_train_trades > result.walk_forward_folds[0].n_train_trades


# ── Full backtest tests ──────────────────────────────────────────────────


class TestFullBacktest:
    def test_returns_result(self, backtester, trades):
        result = backtester.run(trades)
        assert isinstance(result, BacktestResult)
        assert result.n_trades > 0

    def test_capital_accounting(self, backtester, trades):
        result = backtester.run(trades)
        expected = result.initial_capital + result.total_pnl
        assert abs(result.final_capital - expected) < 1.0

    def test_attribution_sums(self, backtester, trades):
        result = backtester.run(trades)
        for t in result.trades:
            expected_net = t.gross_pnl - t.slippage_cost - t.commission_cost
            assert abs(t.net_pnl - expected_net) < 0.01

    def test_costs_positive(self, backtester, trades):
        result = backtester.run(trades)
        assert result.total_slippage >= 0
        assert result.total_commissions >= 0

    def test_empty_after_filter(self):
        cfg = BacktestConfig(signal_threshold=1.0)  # filter everything
        bt = UnifiedBacktester(cfg)
        df = pd.DataFrame({
            "entry_date": ["2024-01-02"], "exit_date": ["2024-01-05"],
            "pnl": [100], "entry_price": [5.0], "strategy_type": ["CS"],
            "regime": ["bull"], "contracts": [5], "signal_score": [0.3],
        })
        result = bt.run(df)
        assert result.n_trades == 0
        assert result.total_pnl == 0.0

    def test_regime_gate_blocks(self):
        cfg = BacktestConfig(
            signal_threshold=0.0,
            regime_gate_enabled=True,
            allowed_regimes=["bull"],
        )
        bt = UnifiedBacktester(cfg)
        df = _make_trades(20, seed=42)
        df["regime"] = "bear"  # all trades in bear regime
        result = bt.run(df)
        assert result.n_trades == 0

    def test_win_rate_bounded(self, backtester, trades):
        result = backtester.run(trades)
        assert 0 <= result.win_rate <= 1.0

    def test_sharpe_computed(self, backtester, trades):
        result = backtester.run(trades)
        assert isinstance(result.sharpe, float)

    def test_drawdown_non_positive(self, backtester, trades):
        result = backtester.run(trades)
        assert result.max_drawdown <= 0.0


# ── Multi-experiment tests ───────────────────────────────────────────────


class TestMultiExperiment:
    def test_run_multi(self):
        bt = UnifiedBacktester(BacktestConfig(signal_threshold=0.0))
        exps = {
            "EXP-A": _make_trades(50, seed=1),
            "EXP-B": _make_trades(40, seed=2),
        }
        result = bt.run_multi(exps)
        assert isinstance(result, BacktestResult)
        assert result.n_trades > 0

    def test_experiment_summaries(self):
        bt = UnifiedBacktester(BacktestConfig(signal_threshold=0.0))
        exps = {
            "EXP-A": _make_trades(50, seed=1),
            "EXP-B": _make_trades(40, seed=2),
        }
        result = bt.run_multi(exps)
        assert len(result.experiment_summaries) == 2
        names = {e.name for e in result.experiment_summaries}
        assert "EXP-A" in names and "EXP-B" in names

    def test_contribution_sums(self):
        bt = UnifiedBacktester(BacktestConfig(signal_threshold=0.0))
        exps = {"A": _make_trades(30, seed=1), "B": _make_trades(30, seed=2)}
        result = bt.run_multi(exps)
        total_contrib = sum(e.contribution_pct for e in result.experiment_summaries)
        assert abs(total_contrib - 1.0) < 0.01


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, backtester, trades):
        result = backtester.run(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "bt.html"
            path = UnifiedBacktester.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Unified Backtest" in content

    def test_contains_attribution(self, backtester, trades):
        result = backtester.run(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            UnifiedBacktester.generate_report(result, out)
            content = out.read_text()
            assert "PnL Attribution" in content
            assert "Alpha" in content

    def test_contains_charts(self, backtester, trades):
        result = backtester.run(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            UnifiedBacktester.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Equity Curve" in content

    def test_contains_regime(self, backtester, trades):
        result = backtester.run(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            UnifiedBacktester.generate_report(result, out)
            content = out.read_text()
            assert "Regime" in content

    def test_contains_walk_forward(self, backtester, trades):
        result = backtester.run(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            UnifiedBacktester.generate_report(result, out)
            content = out.read_text()
            assert "Walk-Forward" in content

    def test_default_path(self, backtester, trades):
        result = backtester.run(trades)
        path = UnifiedBacktester.generate_report(result)
        assert path.exists()
        assert "unified_backtest.html" in str(path)
        path.unlink(missing_ok=True)
