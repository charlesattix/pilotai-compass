"""Tests for compass.signal_backtester — 40 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.signal_backtester import (
    SignalBacktester,
    BacktestMetrics,
    DirectionalResult,
    RegimePerformance,
    SignalCombination,
    WalkForwardFold,
    WalkForwardResult,
    TRADING_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int = 500) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2023-01-02", periods=n)


def _returns(n: int = 500, mu: float = 0.0003, sigma: float = 0.01,
             seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n), index=_dates(n))


def _signal_trend(n: int = 500, seed: int = 42) -> pd.Series:
    """Signal that follows recent momentum."""
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0003, 0.01, n)
    ma = pd.Series(ret).rolling(20).mean().fillna(0)
    sig = ma.apply(lambda x: 1.0 if x > 0 else -1.0)
    return pd.Series(sig.values, index=_dates(n))


def _signal_random(n: int = 500, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice([-1, 0, 1], n), index=_dates(n), dtype=float)


def _regimes(n: int = 500) -> pd.Series:
    labels = []
    for i in range(n):
        if i < n * 0.4:
            labels.append("bull")
        elif i < n * 0.7:
            labels.append("bear")
        else:
            labels.append("bull")
    return pd.Series(labels, index=_dates(n))


# ===========================================================================
# 1. Vectorised backtest
# ===========================================================================

class TestBacktest:
    def test_long_only(self):
        sb = SignalBacktester(cost_per_trade=0)
        sig = pd.Series(1.0, index=_dates(100))
        ret = _returns(100)
        strat = sb.backtest(sig, ret)
        # Long-only with no costs should match returns (shifted)
        assert len(strat) == 100
        assert strat.iloc[1:].sum() != 0

    def test_flat_returns_zero(self):
        sb = SignalBacktester(cost_per_trade=0)
        sig = pd.Series(0.0, index=_dates(50))
        ret = _returns(50)
        strat = sb.backtest(sig, ret)
        assert strat.abs().sum() == pytest.approx(0.0, abs=1e-12)

    def test_costs_reduce_returns(self):
        sig = pd.Series([1, -1, 1, -1, 1] * 20, index=_dates(100), dtype=float)
        ret = _returns(100)
        no_cost = SignalBacktester(cost_per_trade=0).backtest(sig, ret)
        with_cost = SignalBacktester(cost_per_trade=0.01).backtest(sig, ret)
        assert with_cost.sum() < no_cost.sum()

    def test_equity_curve(self):
        sb = SignalBacktester()
        sig = pd.Series(1.0, index=_dates(100))
        strat = sb.backtest(sig, _returns(100))
        eq = sb.equity_curve(strat)
        assert eq.iloc[0] > 0
        assert len(eq) == len(strat)

    def test_empty(self):
        sb = SignalBacktester()
        assert sb.backtest(pd.Series(dtype=float), pd.Series(dtype=float)).empty


# ===========================================================================
# 2. Metrics
# ===========================================================================

class TestMetrics:
    def test_basic(self):
        sb = SignalBacktester(cost_per_trade=0)
        strat = sb.backtest(pd.Series(1.0, index=_dates(300)), _returns(300))
        m = sb.compute_metrics(strat)
        assert isinstance(m, BacktestMetrics)
        assert m.sharpe != 0
        assert m.n_trades > 0

    def test_sharpe_positive_for_uptrend(self):
        sb = SignalBacktester(cost_per_trade=0)
        ret = _returns(500, mu=0.001, sigma=0.005)
        strat = sb.backtest(pd.Series(1.0, index=ret.index), ret)
        m = sb.compute_metrics(strat)
        assert m.sharpe > 0

    def test_max_dd_bounded(self):
        sb = SignalBacktester(cost_per_trade=0)
        strat = sb.backtest(pd.Series(1.0, index=_dates(300)), _returns(300))
        m = sb.compute_metrics(strat)
        assert 0 <= m.max_drawdown <= 1.0

    def test_win_rate_bounded(self):
        sb = SignalBacktester(cost_per_trade=0)
        strat = sb.backtest(_signal_random(200), _returns(200))
        m = sb.compute_metrics(strat)
        assert 0 <= m.win_rate <= 1.0

    def test_profit_factor_positive(self):
        sb = SignalBacktester(cost_per_trade=0)
        strat = sb.backtest(pd.Series(1.0, index=_dates(300)), _returns(300, mu=0.001))
        m = sb.compute_metrics(strat)
        assert m.profit_factor > 0

    def test_empty(self):
        m = SignalBacktester.compute_metrics(pd.Series(dtype=float))
        assert m.sharpe == 0.0

    def test_sortino_positive(self):
        sb = SignalBacktester(cost_per_trade=0)
        ret = _returns(300, mu=0.001, sigma=0.005)
        strat = sb.backtest(pd.Series(1.0, index=ret.index), ret)
        m = sb.compute_metrics(strat)
        assert m.sortino > 0

    def test_calmar_positive(self):
        sb = SignalBacktester(cost_per_trade=0)
        ret = _returns(500, mu=0.001, sigma=0.005)
        strat = sb.backtest(pd.Series(1.0, index=ret.index), ret)
        m = sb.compute_metrics(strat)
        assert m.calmar > 0


# ===========================================================================
# 3. Directional
# ===========================================================================

class TestDirectional:
    def test_all_three(self):
        sb = SignalBacktester(cost_per_trade=0)
        dr = sb.evaluate_directional(_signal_trend(300), _returns(300))
        assert isinstance(dr, DirectionalResult)
        assert isinstance(dr.long, BacktestMetrics)
        assert isinstance(dr.short, BacktestMetrics)
        assert isinstance(dr.combined, BacktestMetrics)

    def test_long_only_no_short_trades(self):
        sb = SignalBacktester(cost_per_trade=0)
        sig = pd.Series(1.0, index=_dates(100))
        dr = sb.evaluate_directional(sig, _returns(100))
        # Short component should have zero or near-zero trades
        assert dr.short.n_trades == 0 or dr.short.total_return == pytest.approx(0.0, abs=1e-8)


# ===========================================================================
# 4. Regime performance
# ===========================================================================

class TestRegime:
    def test_basic(self):
        sb = SignalBacktester(cost_per_trade=0)
        rp = sb.regime_performance(_signal_trend(300), _returns(300), _regimes(300))
        assert len(rp) > 0
        assert all(isinstance(r, RegimePerformance) for r in rp)

    def test_all_regimes(self):
        sb = SignalBacktester(cost_per_trade=0)
        rp = sb.regime_performance(_signal_trend(500), _returns(500), _regimes(500))
        regimes = {r.regime for r in rp}
        assert "bull" in regimes
        assert "bear" in regimes

    def test_n_days_sum(self):
        sb = SignalBacktester(cost_per_trade=0)
        rp = sb.regime_performance(_signal_trend(300), _returns(300), _regimes(300))
        # Sum should be close to total (minus 1 for shift)
        total = sum(r.n_days for r in rp)
        assert total > 250


# ===========================================================================
# 5. Signal combinations
# ===========================================================================

class TestCombinations:
    def test_and(self):
        a = pd.Series([1, 1, -1, 0, 1], dtype=float)
        b = pd.Series([1, -1, -1, 1, 1], dtype=float)
        result = SignalBacktester.combine_signals({"a": a, "b": b}, "and")
        assert result.iloc[0] == 1.0   # both +1
        assert result.iloc[1] == 0.0   # disagree
        assert result.iloc[2] == -1.0  # both -1

    def test_or(self):
        a = pd.Series([1, 0, 0, -1], dtype=float)
        b = pd.Series([0, 1, 0, -1], dtype=float)
        result = SignalBacktester.combine_signals({"a": a, "b": b}, "or")
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == 1.0
        assert result.iloc[2] == 0.0

    def test_vote(self):
        a = pd.Series([1, 1, -1], dtype=float)
        b = pd.Series([1, -1, -1], dtype=float)
        c = pd.Series([1, -1, 1], dtype=float)
        result = SignalBacktester.combine_signals({"a": a, "b": b, "c": c}, "vote")
        assert result.iloc[0] == 1.0  # 3/3 positive

    def test_test_combinations(self):
        sb = SignalBacktester(cost_per_trade=0)
        signals = {
            "s1": _signal_trend(200),
            "s2": _signal_random(200, seed=77),
        }
        combos = sb.test_combinations(signals, _returns(200))
        assert len(combos) > 0
        assert all(isinstance(c, SignalCombination) for c in combos)
        # Sorted by Sharpe
        sharpes = [c.metrics.sharpe for c in combos]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_empty_signals(self):
        result = SignalBacktester.combine_signals({}, "and")
        assert result.empty


# ===========================================================================
# 6. Walk-forward
# ===========================================================================

class TestWalkForward:
    def test_basic(self):
        sb = SignalBacktester(cost_per_trade=0)
        wf = sb.walk_forward(_signal_trend(500), _returns(500), n_folds=5)
        assert isinstance(wf, WalkForwardResult)
        assert len(wf.folds) == 4  # n_folds - 1

    def test_expanding(self):
        sb = SignalBacktester(cost_per_trade=0)
        wf = sb.walk_forward(_signal_trend(500), _returns(500), n_folds=5, expanding=True)
        assert len(wf.folds) == 4

    def test_degradation(self):
        sb = SignalBacktester(cost_per_trade=0)
        wf = sb.walk_forward(_signal_trend(500), _returns(500), n_folds=5)
        assert isinstance(wf.oos_degradation, float)

    def test_too_short(self):
        sb = SignalBacktester()
        wf = sb.walk_forward(pd.Series([1.0] * 5, index=_dates(5)),
                              _returns(5), n_folds=5)
        assert wf.folds == []


# ===========================================================================
# Monthly returns
# ===========================================================================

class TestMonthly:
    def test_pivot(self):
        strat = _returns(500)
        monthly = SignalBacktester.monthly_returns(strat)
        assert not monthly.empty
        assert monthly.shape[1] <= 12

    def test_empty(self):
        assert SignalBacktester.monthly_returns(pd.Series(dtype=float)).empty


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        sb = SignalBacktester(cost_per_trade=0)
        ret = _returns(300)
        strat = sb.backtest(pd.Series(1.0, index=ret.index), ret)
        m = sb.compute_metrics(strat)
        eq = sb.equity_curve(strat)
        out = tmp_path / "sig.html"
        result = sb.generate_report(m, equity=eq, strategy_returns=strat,
                                     output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Signal Backtest" in html

    def test_contains_charts(self, tmp_path):
        sb = SignalBacktester(cost_per_trade=0)
        ret = _returns(300)
        strat = sb.backtest(pd.Series(1.0, index=ret.index), ret)
        eq = sb.equity_curve(strat)
        out = tmp_path / "sig.html"
        sb.generate_report(sb.compute_metrics(strat), equity=eq,
                            strategy_returns=strat, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Equity" in html
        assert "Drawdown" in html

    def test_contains_heatmap(self, tmp_path):
        sb = SignalBacktester(cost_per_trade=0)
        ret = _returns(500)
        strat = sb.backtest(pd.Series(1.0, index=ret.index), ret)
        out = tmp_path / "sig.html"
        sb.generate_report(sb.compute_metrics(strat), strategy_returns=strat,
                            output_path=str(out))
        html = out.read_text()
        assert "Monthly Returns" in html

    def test_full_report(self, tmp_path):
        sb = SignalBacktester(cost_per_trade=0)
        ret = _returns(500)
        sig = _signal_trend(500)
        strat = sb.backtest(sig, ret)
        m = sb.compute_metrics(strat)
        eq = sb.equity_curve(strat)
        dr = sb.evaluate_directional(sig, ret)
        rp = sb.regime_performance(sig, ret, _regimes(500))
        wf = sb.walk_forward(sig, ret, n_folds=4)
        out = tmp_path / "full.html"
        result = sb.generate_report(
            m, equity=eq, strategy_returns=strat,
            directional=dr, regime_perf=rp,
            walk_forward=wf, output_path=str(out))
        html = Path(result).read_text()
        for section in ["Equity", "Drawdown", "Metrics",
                         "Directional", "Regime", "Walk-Forward"]:
            assert section in html
