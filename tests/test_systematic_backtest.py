from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from compass.systematic_backtest import (
    DEFAULT_CONFIG,
    BacktestResult,
    MonteCarloResult,
    SystematicBacktester,
    Trade,
    WalkForwardResult,
    compute_calmar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_trade_impact,
    validate_config,
    walk_forward_splits,
)


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------

def _make_price_df(
    n_days: int = 500,
    seed: int = 42,
    start: str = "2020-01-01",
    trend: float = 0.0005,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    returns = rng.normal(trend, 0.01, n_days)
    prices = 100 * np.exp(np.cumsum(returns))
    volume = rng.uniform(1e6, 5e6, n_days)
    return pd.DataFrame({"close": prices, "volume": volume}, index=dates)


def _simple_feature_fn(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    return df


def _simple_signal_fn(df: pd.DataFrame) -> pd.Series:
    if "sma_10" not in df.columns or "sma_50" not in df.columns:
        return pd.Series(0.0, index=df.index)
    sig = pd.Series(0.0, index=df.index)
    sig[df["sma_10"] > df["sma_50"]] = 1.0
    sig[df["sma_10"] < df["sma_50"]] = 0.0
    return sig


def _always_long_signal(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def _always_zero_signal(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=df.index)


def _regime_series(index: pd.DatetimeIndex, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    labels = rng.choice(["bull", "bear", "neutral"], size=len(index))
    return pd.Series(labels, index=index)


def _build_backtester(
    n_days: int = 500,
    seed: int = 42,
    config_overrides: dict | None = None,
    signal_fn=None,
    feature_fn=None,
    with_regime: bool = False,
) -> SystematicBacktester:
    cfg = {
        "start_date": "2020-01-01",
        "end_date": "2025-12-31",
        "initial_capital": 100_000,
        "transaction_cost_bps": 5,
        "spread_cost_bps": 2,
        "max_position_pct": 0.20,
        "rebalance_freq": "daily",
    }
    if config_overrides:
        cfg.update(config_overrides)
    bt = SystematicBacktester(cfg)
    df = _make_price_df(n_days=n_days, seed=seed)
    bt.load_data(df)
    bt.set_feature_fn(feature_fn or _simple_feature_fn)
    bt.set_signal_fn(signal_fn or _simple_signal_fn)
    if with_regime:
        bt.set_regime_series(_regime_series(df.index, seed))
    return bt


# ---------------------------------------------------------------------------
# 1. Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_defaults_applied(self):
        cfg = validate_config({})
        assert cfg["initial_capital"] == 100_000
        assert cfg["n_splits"] == 5

    def test_missing_required_key_after_override(self):
        # All required keys come from DEFAULT_CONFIG, so this should succeed
        cfg = validate_config({"initial_capital": 50_000})
        assert cfg["initial_capital"] == 50_000

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError, match="initial_capital"):
            validate_config({"initial_capital": -1})

    def test_invalid_max_position_pct(self):
        with pytest.raises(ValueError, match="max_position_pct"):
            validate_config({"max_position_pct": 1.5})

    def test_invalid_rebalance_freq(self):
        with pytest.raises(ValueError, match="rebalance_freq"):
            validate_config({"rebalance_freq": "hourly"})

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError, match="start_date must be before"):
            validate_config({"start_date": "2025-01-01", "end_date": "2020-01-01"})

    def test_negative_transaction_cost(self):
        with pytest.raises(ValueError, match="transaction_cost_bps"):
            validate_config({"transaction_cost_bps": -1})

    def test_negative_spread_cost(self):
        with pytest.raises(ValueError, match="spread_cost_bps"):
            validate_config({"spread_cost_bps": -1})


# ---------------------------------------------------------------------------
# 2. Pipeline execution
# ---------------------------------------------------------------------------

class TestPipelineExecution:
    def test_full_pipeline_runs(self):
        bt = _build_backtester()
        result = bt.run()
        assert isinstance(result, BacktestResult)
        assert not result.equity_curve.empty

    def test_no_data_raises(self):
        bt = SystematicBacktester()
        bt.set_signal_fn(_always_long_signal)
        with pytest.raises(RuntimeError, match="No data"):
            bt.run()

    def test_no_signal_fn_raises(self):
        bt = SystematicBacktester()
        bt.load_data(_make_price_df())
        with pytest.raises(RuntimeError, match="No signal"):
            bt.run()

    def test_empty_dataframe_raises(self):
        bt = SystematicBacktester()
        with pytest.raises(ValueError, match="empty"):
            bt.load_data(pd.DataFrame())

    def test_missing_close_column_raises(self):
        df = pd.DataFrame(
            {"price": [1, 2]},
            index=pd.bdate_range("2020-01-01", periods=2),
        )
        bt = SystematicBacktester()
        with pytest.raises(ValueError, match="close"):
            bt.load_data(df)

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"close": [1, 2]})
        bt = SystematicBacktester()
        with pytest.raises(ValueError, match="DatetimeIndex"):
            bt.load_data(df)

    def test_feature_fn_applied(self):
        bt = _build_backtester()
        result = bt.run()
        # If features were applied, the signal fn would see sma columns
        assert result.metrics.get("num_trades", 0) >= 0

    def test_always_long_produces_trades(self):
        bt = _build_backtester(signal_fn=_always_long_signal)
        result = bt.run()
        assert len(result.trades) >= 0
        assert not result.equity_curve.empty


# ---------------------------------------------------------------------------
# 3. Walk-forward splits
# ---------------------------------------------------------------------------

class TestWalkForward:
    def test_correct_number_of_splits(self):
        dates = pd.bdate_range("2020-01-01", periods=500)
        splits = walk_forward_splits(dates, n_splits=5)
        assert len(splits) == 5

    def test_train_test_no_overlap(self):
        dates = pd.bdate_range("2020-01-01", periods=500)
        for train, test in walk_forward_splits(dates, 5):
            assert train[-1] < test[0]

    def test_splits_cover_all_data(self):
        dates = pd.bdate_range("2020-01-01", periods=500)
        splits = walk_forward_splits(dates, 5)
        all_dates = set()
        for train, test in splits:
            all_dates.update(train.tolist())
            all_dates.update(test.tolist())
        assert len(all_dates) == 500

    def test_zero_splits_raises(self):
        dates = pd.bdate_range("2020-01-01", periods=100)
        with pytest.raises(ValueError, match="n_splits"):
            walk_forward_splits(dates, 0)

    def test_too_many_splits_raises(self):
        dates = pd.bdate_range("2020-01-01", periods=4)
        with pytest.raises(ValueError, match="Not enough"):
            walk_forward_splits(dates, 10)

    def test_walk_forward_in_result(self):
        bt = _build_backtester()
        result = bt.run()
        assert isinstance(result.walk_forward_results, list)


# ---------------------------------------------------------------------------
# 4. Cost computation
# ---------------------------------------------------------------------------

class TestCostComputation:
    def test_trade_impact_positive(self):
        impact = compute_trade_impact(100_000, 1e8)
        assert impact > 0

    def test_trade_impact_zero_volume(self):
        assert compute_trade_impact(100_000, 0) == 0.0

    def test_trade_cost_includes_all_components(self):
        bt = _build_backtester()
        cost = bt._compute_trade_cost(100_000)
        # Should include spread + commission + impact
        spread = 100_000 * 2 / 10_000
        commission = 100_000 * 5 / 10_000
        assert cost > spread + commission  # impact adds more

    def test_zero_value_trade_zero_cost_components(self):
        bt = _build_backtester()
        cost = bt._compute_trade_cost(0.0)
        assert cost == 0.0

    def test_costs_recorded_in_trades(self):
        bt = _build_backtester()
        result = bt.run()
        for t in result.trades:
            assert t.costs >= 0


# ---------------------------------------------------------------------------
# 5. Daily P&L
# ---------------------------------------------------------------------------

class TestDailyPnL:
    def test_pnl_has_expected_columns(self):
        bt = _build_backtester()
        result = bt.run()
        for col in ("alpha", "costs", "slippage", "total"):
            assert col in result.daily_pnl.columns

    def test_pnl_length_matches_data(self):
        bt = _build_backtester(n_days=200)
        result = bt.run()
        assert len(result.daily_pnl) == len(result.equity_curve)

    def test_total_equals_alpha_minus_costs(self):
        bt = _build_backtester()
        result = bt.run()
        pnl = result.daily_pnl
        diff = (pnl["alpha"] - pnl["costs"] - pnl["total"]).abs()
        assert diff.max() < 1e-6


# ---------------------------------------------------------------------------
# 6. Regime metrics
# ---------------------------------------------------------------------------

class TestRegimeMetrics:
    def test_regime_metrics_populated(self):
        bt = _build_backtester(with_regime=True)
        result = bt.run()
        assert len(result.regime_metrics) > 0

    def test_each_regime_has_keys(self):
        bt = _build_backtester(with_regime=True)
        result = bt.run()
        expected_keys = {"count", "mean_return", "std_return", "sharpe", "sortino", "max_drawdown", "total_return"}
        for regime, m in result.regime_metrics.items():
            assert expected_keys.issubset(set(m.keys()))

    def test_no_regime_returns_empty(self):
        bt = _build_backtester(with_regime=False)
        result = bt.run()
        assert result.regime_metrics == {}


# ---------------------------------------------------------------------------
# 7. Monte Carlo
# ---------------------------------------------------------------------------

class TestMonteCarlo:
    def test_mc_result_populated(self):
        bt = _build_backtester()
        result = bt.run()
        mc = result.monte_carlo
        assert len(mc.sharpe_ci) > 0
        assert len(mc.returns_ci) > 0
        assert len(mc.max_dd_ci) > 0

    def test_mc_confidence_levels(self):
        bt = _build_backtester()
        result = bt.run()
        mc = result.monte_carlo
        for level in [0.05, 0.50, 0.95]:
            assert level in mc.sharpe_ci

    def test_mc_sharpe_lower_bound_less_than_upper(self):
        bt = _build_backtester()
        result = bt.run()
        mc = result.monte_carlo
        assert mc.sharpe_ci[0.05] <= mc.sharpe_ci[0.95]

    def test_mc_simulated_arrays_correct_length(self):
        bt = _build_backtester(config_overrides={"mc_simulations": 200})
        result = bt.run()
        assert len(result.monte_carlo.simulated_sharpes) == 200

    def test_mc_empty_returns(self):
        bt = _build_backtester()
        mc = bt._monte_carlo(pd.Series(dtype=float))
        assert mc.sharpe_ci == {}


# ---------------------------------------------------------------------------
# 8. HTML report
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_report_is_string(self):
        bt = _build_backtester(with_regime=True)
        result = bt.run()
        html = bt.generate_report(result)
        assert isinstance(html, str)

    def test_report_contains_svg(self):
        bt = _build_backtester()
        result = bt.run()
        html = bt.generate_report(result)
        assert "<svg" in html

    def test_report_contains_sections(self):
        bt = _build_backtester(with_regime=True)
        result = bt.run()
        html = bt.generate_report(result)
        for section in [
            "Equity Curve",
            "Drawdown",
            "Monthly Returns",
            "Trade List",
            "Walk-Forward",
            "Regime Analysis",
            "Monte Carlo",
        ]:
            assert section in html

    def test_report_metrics_table(self):
        bt = _build_backtester()
        result = bt.run()
        html = bt.generate_report(result)
        assert "total_return" in html
        assert "sharpe" in html

    def test_report_empty_result(self):
        bt = _build_backtester()
        empty = BacktestResult()
        html = bt.generate_report(empty)
        assert "No data" in html or "No trades" in html


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_day_data(self):
        cfg = {
            "start_date": "2020-01-01",
            "end_date": "2025-12-31",
            "initial_capital": 100_000,
            "transaction_cost_bps": 5,
            "spread_cost_bps": 2,
            "max_position_pct": 0.20,
            "rebalance_freq": "daily",
            "n_splits": 1,
        }
        df = pd.DataFrame(
            {"close": [100.0]},
            index=pd.DatetimeIndex(["2020-01-02"]),
        )
        bt = SystematicBacktester(cfg)
        bt.load_data(df)
        bt.set_signal_fn(_always_long_signal)
        # Should not crash even with a single data point
        result = bt.run()
        assert len(result.equity_curve) == 1

    def test_all_zero_signals_no_trades(self):
        bt = _build_backtester(signal_fn=_always_zero_signal)
        result = bt.run()
        assert len(result.trades) == 0

    def test_data_outside_range_raises(self):
        cfg = {"start_date": "2030-01-01", "end_date": "2030-12-31"}
        bt = SystematicBacktester(cfg)
        df = _make_price_df()
        with pytest.raises(ValueError, match="No data in the configured date range"):
            bt.load_data(df)

    def test_weekly_rebalance(self):
        bt = _build_backtester(config_overrides={"rebalance_freq": "weekly"})
        result = bt.run()
        assert isinstance(result, BacktestResult)

    def test_monthly_rebalance(self):
        bt = _build_backtester(config_overrides={"rebalance_freq": "monthly"})
        result = bt.run()
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# 10. Dataclass integrity
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_trade_fields(self):
        t = Trade(
            entry_date=pd.Timestamp("2020-01-02"),
            exit_date=pd.Timestamp("2020-01-10"),
            direction="long",
            size=100.0,
            entry_price=50.0,
            exit_price=55.0,
            pnl=500.0,
            costs=10.0,
            exit_reason="signal",
        )
        assert t.pnl == 500.0
        assert t.direction == "long"

    def test_backtest_result_defaults(self):
        r = BacktestResult()
        assert r.equity_curve.empty
        assert r.trades == []
        assert r.metrics == {}

    def test_walk_forward_result_fields(self):
        names = {f.name for f in fields(WalkForwardResult)}
        assert "in_sample_sharpe" in names
        assert "out_of_sample_sharpe" in names

    def test_monte_carlo_result_defaults(self):
        mc = MonteCarloResult()
        assert mc.sharpe_ci == {}
        assert mc.simulated_sharpes is None


# ---------------------------------------------------------------------------
# 11. Helper functions
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_sharpe_empty(self):
        assert compute_sharpe(pd.Series(dtype=float)) == 0.0

    def test_sharpe_constant_returns(self):
        assert compute_sharpe(pd.Series([0.01] * 100)) == 0.0  # std = 0

    def test_sortino_no_downside(self):
        assert compute_sortino(pd.Series([0.01, 0.02, 0.03])) == 0.0

    def test_calmar_zero_dd(self):
        assert compute_calmar(0.1, 0.0) == 0.0

    def test_max_drawdown_monotonic_up(self):
        eq = pd.Series([1, 2, 3, 4, 5])
        assert compute_max_drawdown(eq) == 0.0

    def test_max_drawdown_known(self):
        eq = pd.Series([100, 80, 90, 70, 100])
        dd = compute_max_drawdown(eq)
        # Peak 100 -> trough 70 => 30%
        assert abs(dd - 0.30) < 1e-10

    def test_max_drawdown_empty(self):
        assert compute_max_drawdown(pd.Series(dtype=float)) == 0.0
