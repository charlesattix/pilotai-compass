"""Tests for compass/north_star_backtest.py — North Star validation pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.north_star_backtest import (
    BacktestConfig,
    NorthStarBacktest,
    NorthStarResult,
    NorthStarTargets,
    RegimeMetrics,
    TradeResult,
    WalkForwardFold,
    YearMetrics,
    apply_costs,
    compute_contracts,
    compute_max_drawdown_pct,
    compute_profit_factor,
    compute_sharpe,
    compute_signal_score,
    compute_sortino,
    risk_gate,
    walk_forward_by_year,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_trades(n: int = 200, seed: int = 42, win_rate: float = 0.58) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-02", periods=n * 2)
    entry_dates = dates[::2][:n]
    exit_dates = dates[1::2][:n]
    years = pd.to_datetime(entry_dates).year

    wins = rng.random(n) < win_rate
    pnl = np.where(wins, rng.uniform(50, 600, n), rng.uniform(-400, -50, n))
    regimes = rng.choice(["bull", "bear", "sideways"], n, p=[0.5, 0.2, 0.3])

    return pd.DataFrame({
        "entry_date": entry_dates,
        "exit_date": exit_dates,
        "year": years,
        "pnl": pnl,
        "net_credit": rng.uniform(0.5, 3.0, n),
        "strategy_type": rng.choice(["CS", "IC", "SS"], n),
        "regime": regimes,
        "vix": rng.uniform(12, 35, n),
        "vix_percentile_50d": rng.uniform(10, 95, n),
        "iv_rank": rng.uniform(5, 90, n),
        "momentum_5d_pct": rng.normal(0, 2, n),
        "contracts": 5,
        "win": wins.astype(int),
    })


def _make_multi_year(years: list = None) -> pd.DataFrame:
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]
    frames = []
    for i, y in enumerate(years):
        df = _make_trades(40, seed=42 + i)
        df["year"] = y
        df["entry_date"] = pd.bdate_range(f"{y}-01-02", periods=40)
        df["exit_date"] = pd.bdate_range(f"{y}-01-03", periods=40)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def trades():
    return _make_trades()


@pytest.fixture
def multi_year_trades():
    return _make_multi_year()


@pytest.fixture
def config():
    return BacktestConfig(signal_threshold=0.3)


@pytest.fixture
def backtester(config):
    return NorthStarBacktest(config)


# ── Config tests ─────────────────────────────────────────────────────────


class TestConfig:
    def test_default_targets(self):
        t = NorthStarTargets()
        assert t.annual_return_pct == 55.0
        assert t.sharpe_ratio == 6.0
        assert t.max_drawdown_pct == 30.0

    def test_custom_targets(self):
        t = NorthStarTargets(annual_return_pct=100.0, sharpe_ratio=8.0, max_drawdown_pct=12.0)
        assert t.annual_return_pct == 100.0

    def test_default_config(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000.0


# ── Signal scoring tests ─────────────────────────────────────────────────


class TestSignalScoring:
    def test_bull_regime_bonus(self):
        row = pd.Series({"regime": "bull", "vix_percentile_50d": 50, "iv_rank": 50, "momentum_5d_pct": 0})
        score = compute_signal_score(row)
        assert score > 0.5

    def test_bear_regime_penalty(self):
        row_bull = pd.Series({"regime": "bull", "vix_percentile_50d": 50, "iv_rank": 50, "momentum_5d_pct": 0})
        row_bear = pd.Series({"regime": "bear", "vix_percentile_50d": 50, "iv_rank": 50, "momentum_5d_pct": 0})
        assert compute_signal_score(row_bull) > compute_signal_score(row_bear)

    def test_high_vix_bonus(self):
        row_low = pd.Series({"regime": "sideways", "vix_percentile_50d": 20, "iv_rank": 50, "momentum_5d_pct": 0})
        row_high = pd.Series({"regime": "sideways", "vix_percentile_50d": 80, "iv_rank": 50, "momentum_5d_pct": 0})
        assert compute_signal_score(row_high) > compute_signal_score(row_low)

    def test_score_bounded(self, trades):
        for _, row in trades.head(20).iterrows():
            s = compute_signal_score(row)
            assert 0.0 <= s <= 1.0


# ── Sizing tests ─────────────────────────────────────────────────────────


class TestSizing:
    def test_basic_sizing(self, config):
        c = compute_contracts(100_000, 0.8, 2.0, config)
        assert c >= 1

    def test_higher_signal_more_contracts(self, config):
        c_low = compute_contracts(100_000, 0.3, 2.0, config)
        c_high = compute_contracts(100_000, 1.0, 2.0, config)
        assert c_high >= c_low

    def test_zero_price(self, config):
        c = compute_contracts(100_000, 0.7, 0.0, config)
        assert c == config.base_contracts


# ── Risk gate tests ──────────────────────────────────────────────────────


class TestRiskGate:
    def test_passes_normal(self, config):
        passed, _ = risk_gate("bull", 0.05, config)
        assert passed is True

    def test_blocks_drawdown(self, config):
        passed, reason = risk_gate("bull", 0.30, config)
        assert passed is False
        assert "drawdown" in reason

    def test_blocks_regime(self, config):
        passed, reason = risk_gate("bear", 0.05, config)
        assert passed is False
        assert "regime" in reason

    def test_regime_filter_disabled(self):
        cfg = BacktestConfig(regime_filter=False)
        passed, _ = risk_gate("bear", 0.05, cfg)
        assert passed is True


# ── Execution cost tests ─────────────────────────────────────────────────


class TestExecution:
    def test_costs_positive(self, config):
        slip, comm, net = apply_costs(100.0, 5, 2.0, 2.0, config)
        assert slip > 0
        assert comm > 0
        assert net < 100.0

    def test_commission_scales(self, config):
        _, c1, _ = apply_costs(100.0, 1, 2.0, 2.0, config)
        _, c5, _ = apply_costs(100.0, 5, 2.0, 2.0, config)
        assert c5 == c1 * 5


# ── Metrics tests ────────────────────────────────────────────────────────


class TestMetrics:
    def test_sharpe_positive(self):
        pnls = np.array([100, 50, 80, -20, 60, 40, 30, 70])
        assert compute_sharpe(pnls) > 0

    def test_sharpe_short(self):
        assert compute_sharpe(np.array([100])) == 0.0

    def test_sortino(self):
        pnls = np.array([100, 50, -20, 80, 60])
        assert compute_sortino(pnls) > 0

    def test_max_dd(self):
        equity = np.array([100, 110, 95, 105, 100])
        dd = compute_max_drawdown_pct(equity)
        assert dd > 0

    def test_profit_factor(self):
        pnls = np.array([100, -50, 200, -30])
        assert compute_profit_factor(pnls) == pytest.approx(300.0 / 80.0)


# ── Walk-forward tests ───────────────────────────────────────────────────


class TestWalkForward:
    def test_folds_created(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        assert len(result.walk_forward_folds) > 0

    def test_expanding_window(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        folds = result.walk_forward_folds
        if len(folds) >= 2:
            assert folds[1].n_train > folds[0].n_train

    def test_fold_structure(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        for f in result.walk_forward_folds:
            assert isinstance(f, WalkForwardFold)
            assert f.n_test > 0


# ── Full backtest tests ──────────────────────────────────────────────────


class TestFullBacktest:
    def test_returns_result(self, backtester, trades):
        result = backtester.run(trades)
        assert isinstance(result, NorthStarResult)
        assert result.n_trades > 0

    def test_capital_accounting(self, backtester, trades):
        result = backtester.run(trades)
        expected = result.initial_capital + result.total_pnl
        assert abs(result.final_capital - expected) < 1.0

    def test_year_metrics_present(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        assert len(result.year_metrics) > 0
        years = {ym.year for ym in result.year_metrics}
        assert len(years) >= 2

    def test_regime_metrics(self, backtester, trades):
        result = backtester.run(trades)
        assert len(result.regime_metrics) > 0

    def test_target_assessment(self, backtester, trades):
        result = backtester.run(trades)
        assert isinstance(result.annual_return_target_met, bool)
        assert isinstance(result.sharpe_target_met, bool)
        assert isinstance(result.drawdown_target_met, bool)

    def test_equity_curve_length(self, backtester, trades):
        result = backtester.run(trades)
        assert len(result.equity_curve) == result.n_trades

    def test_empty_after_filter(self):
        cfg = BacktestConfig(signal_threshold=1.0)
        bt = NorthStarBacktest(cfg)
        result = bt.run(_make_trades(10))
        assert result.n_trades == 0

    def test_costs_reduce_pnl(self, backtester, trades):
        result = backtester.run(trades)
        total_gross = sum(t.gross_pnl for t in result.trades)
        assert result.total_pnl < total_gross  # costs reduce

    def test_win_rate_bounded(self, backtester, trades):
        result = backtester.run(trades)
        assert 0 <= result.overall_win_rate <= 1.0

    def test_regime_gate_blocks_bear(self):
        cfg = BacktestConfig(signal_threshold=0.0, regime_filter=True, allowed_regimes=["bull"])
        bt = NorthStarBacktest(cfg)
        df = _make_trades(30)
        df["regime"] = "bear"
        result = bt.run(df)
        assert result.n_trades == 0

    def test_multi_year_all_years(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        trade_years = {t.year for t in result.trades}
        metric_years = {ym.year for ym in result.year_metrics}
        assert trade_years == metric_years


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ns.html"
            path = NorthStarBacktest.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "North Star" in content

    def test_contains_scorecard(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            NorthStarBacktest.generate_report(result, out)
            content = out.read_text()
            assert "Scorecard" in content
            assert "Target" in content.lower() or "target" in content

    def test_contains_equity_curve(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            NorthStarBacktest.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Equity" in content

    def test_contains_year_table(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            NorthStarBacktest.generate_report(result, out)
            content = out.read_text()
            assert "Per-Year" in content

    def test_default_path(self, backtester, multi_year_trades):
        result = backtester.run(multi_year_trades)
        path = NorthStarBacktest.generate_report(result)
        assert path.exists()
        assert "north_star_backtest.html" in str(path)
        path.unlink(missing_ok=True)
