"""Tests for compass/slippage_model.py — advanced slippage modeling."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.slippage_model import (
    BudgetAllocation,
    CalibrationResult,
    InstrumentProfile,
    ModelComparison,
    SlippageEngine,
    SlippageResult,
    TimeOfDayPattern,
    allocate_slippage_budget,
    build_instrument_profiles,
    build_time_patterns,
    calibrate_models,
    estimate_bid_ask_cost,
    slippage_fixed_bps,
    slippage_sqrt_impact,
    slippage_volatility_adjusted,
    slippage_volume_dependent,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_trades(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic trade data with observed slippage."""
    rng = np.random.RandomState(seed)
    prices = rng.uniform(2.0, 10.0, n)
    qtys = rng.randint(1, 20, n)
    volumes = rng.randint(500, 5000, n)
    vols = rng.uniform(0.005, 0.03, n)
    spreads = rng.uniform(5.0, 20.0, n)

    # Observed slippage is roughly volume-dependent + noise
    observed = np.array([
        slippage_volume_dependent(p, q, v) + rng.normal(0, 0.005)
        for p, q, v in zip(prices, qtys, volumes)
    ])
    observed = np.abs(observed)  # slippage is always positive

    instruments = rng.choice(["SPY", "QQQ", "IWM", "DIA"], n)
    hours = rng.choice([9, 10, 11, 12, 13, 14, 15], n)
    times = [f"2024-01-15 {h:02d}:{rng.randint(0, 59):02d}:00" for h in hours]

    return pd.DataFrame({
        "price": prices,
        "order_qty": qtys,
        "market_volume": volumes,
        "realised_vol": vols,
        "spread_bps": spreads,
        "observed_slippage": observed,
        "instrument": instruments,
        "trade_time": times,
    })


@pytest.fixture
def trades():
    return _make_trades()


@pytest.fixture
def engine():
    return SlippageEngine(total_budget_bps=20.0)


# ── Slippage model function tests ────────────────────────────────────────


class TestSlippageModels:
    def test_fixed_bps(self):
        cost = slippage_fixed_bps(100.0, 5.0)
        assert cost == pytest.approx(0.05)  # 100 * 5 / 10000

    def test_fixed_bps_zero(self):
        assert slippage_fixed_bps(100.0, 0.0) == 0.0

    def test_volume_dependent_positive(self):
        cost = slippage_volume_dependent(5.0, 10, 1000)
        assert cost > 0

    def test_volume_dependent_scales_with_qty(self):
        small = slippage_volume_dependent(5.0, 1, 1000)
        large = slippage_volume_dependent(5.0, 500, 1000)
        assert large > small

    def test_volume_dependent_zero_volume(self):
        cost = slippage_volume_dependent(5.0, 10, 0)
        assert cost > 0  # fallback

    def test_vol_adjusted_positive(self):
        cost = slippage_volatility_adjusted(5.0, 0.02)
        assert cost > 0

    def test_vol_adjusted_scales_with_vol(self):
        low = slippage_volatility_adjusted(5.0, 0.005)
        high = slippage_volatility_adjusted(5.0, 0.03)
        assert high > low

    def test_sqrt_impact_positive(self):
        cost = slippage_sqrt_impact(5.0, 10, 1000, 0.01)
        assert cost > 0

    def test_sqrt_impact_scales(self):
        small = slippage_sqrt_impact(5.0, 1, 1000, 0.01)
        large = slippage_sqrt_impact(5.0, 500, 1000, 0.01)
        assert large > small

    def test_sqrt_impact_zero_volume(self):
        cost = slippage_sqrt_impact(5.0, 10, 0, 0.01)
        assert cost > 0

    def test_bid_ask_cost(self):
        cost = estimate_bid_ask_cost(100.0, 10.0)
        assert cost == pytest.approx(0.05)  # half spread

    def test_bid_ask_cost_zero_spread(self):
        assert estimate_bid_ask_cost(100.0, 0.0) == 0.0


# ── Calibration tests ────────────────────────────────────────────────────


class TestCalibration:
    def test_calibrate_returns_result(self, trades):
        result = calibrate_models(trades)
        assert isinstance(result, CalibrationResult)
        assert result.n_trades == 100
        assert len(result.model_comparisons) == 4

    def test_best_model_is_first(self, trades):
        result = calibrate_models(trades)
        # First comparison should have lowest RMSE
        rmses = [c.rmse for c in result.model_comparisons]
        assert rmses[0] == min(rmses)
        assert result.best_model == result.model_comparisons[0].model_name

    def test_all_models_present(self, trades):
        result = calibrate_models(trades)
        names = {c.model_name for c in result.model_comparisons}
        assert names == {"fixed_bps", "volume_dependent", "volatility_adjusted", "sqrt_impact"}

    def test_rmse_positive(self, trades):
        result = calibrate_models(trades)
        for c in result.model_comparisons:
            assert c.rmse >= 0

    def test_mae_leq_rmse(self, trades):
        result = calibrate_models(trades)
        for c in result.model_comparisons:
            assert c.mae <= c.rmse + 1e-12

    def test_empty_trades(self):
        result = calibrate_models(pd.DataFrame())
        assert result.n_trades == 0
        assert result.best_model == "fixed_bps"

    def test_calibrated_params(self, trades):
        result = calibrate_models(trades)
        assert "bps" in result.calibrated_params


# ── Instrument profile tests ─────────────────────────────────────────────


class TestInstrumentProfiles:
    def test_profiles_built(self, trades):
        profiles = build_instrument_profiles(trades)
        assert len(profiles) > 0
        assert all(isinstance(p, InstrumentProfile) for p in profiles)

    def test_sorted_by_total(self, trades):
        profiles = build_instrument_profiles(trades)
        totals = [p.total_slippage_dollars for p in profiles]
        assert totals == sorted(totals, reverse=True)

    def test_trade_counts_sum(self, trades):
        profiles = build_instrument_profiles(trades)
        assert sum(p.n_trades for p in profiles) == len(trades)

    def test_p95_geq_median(self, trades):
        profiles = build_instrument_profiles(trades)
        for p in profiles:
            assert p.p95_slippage_bps >= p.median_slippage_bps - 1e-6

    def test_empty_returns_empty(self):
        assert build_instrument_profiles(pd.DataFrame()) == []


# ── Time-of-day pattern tests ────────────────────────────────────────────


class TestTimePatterns:
    def test_patterns_built(self, trades):
        patterns = build_time_patterns(trades)
        assert len(patterns) > 0
        assert all(isinstance(p, TimeOfDayPattern) for p in patterns)

    def test_sorted_by_hour(self, trades):
        patterns = build_time_patterns(trades)
        hours = [int(p.bucket[:2]) for p in patterns]
        assert hours == sorted(hours)

    def test_trade_counts_sum(self, trades):
        patterns = build_time_patterns(trades)
        assert sum(p.n_trades for p in patterns) == len(trades)

    def test_empty_returns_empty(self):
        assert build_time_patterns(pd.DataFrame()) == []


# ── Budget allocation tests ──────────────────────────────────────────────


class TestBudgetAllocation:
    def test_allocations_built(self, trades):
        profiles = build_instrument_profiles(trades)
        allocs = allocate_slippage_budget(profiles, 20.0)
        assert len(allocs) > 0
        assert all(isinstance(a, BudgetAllocation) for a in allocs)

    def test_weights_sum_to_one(self, trades):
        profiles = build_instrument_profiles(trades)
        allocs = allocate_slippage_budget(profiles, 20.0)
        total_weight = sum(a.weight for a in allocs)
        assert total_weight == pytest.approx(1.0)

    def test_budget_sums_correctly(self, trades):
        profiles = build_instrument_profiles(trades)
        allocs = allocate_slippage_budget(profiles, 20.0)
        total_budget = sum(a.allocated_bps for a in allocs)
        assert total_budget == pytest.approx(20.0)

    def test_utilization_positive(self, trades):
        profiles = build_instrument_profiles(trades)
        allocs = allocate_slippage_budget(profiles, 20.0)
        for a in allocs:
            assert a.budget_utilization >= 0

    def test_sorted_by_utilization(self, trades):
        profiles = build_instrument_profiles(trades)
        allocs = allocate_slippage_budget(profiles, 20.0)
        utils = [a.budget_utilization for a in allocs]
        assert utils == sorted(utils, reverse=True)

    def test_empty_profiles(self):
        assert allocate_slippage_budget([]) == []


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_default(self):
        e = SlippageEngine()
        assert e.total_budget_bps == 20.0

    def test_custom_budget(self):
        e = SlippageEngine(total_budget_bps=30.0)
        assert e.total_budget_bps == 30.0

    def test_zero_budget_raises(self):
        with pytest.raises(ValueError, match="positive"):
            SlippageEngine(total_budget_bps=0)

    def test_missing_columns_raises(self):
        e = SlippageEngine()
        with pytest.raises(ValueError, match="Missing"):
            e.analyze(pd.DataFrame({"foo": [1]}))


# ── Full analysis tests ──────────────────────────────────────────────────


class TestFullAnalysis:
    def test_analyze_returns_result(self, engine, trades):
        result = engine.analyze(trades)
        assert isinstance(result, SlippageResult)
        assert result.n_trades == 100

    def test_total_slippage_positive(self, engine, trades):
        result = engine.analyze(trades)
        assert result.total_slippage_dollars > 0

    def test_avg_slippage_positive(self, engine, trades):
        result = engine.analyze(trades)
        assert result.avg_slippage_bps > 0

    def test_all_components_present(self, engine, trades):
        result = engine.analyze(trades)
        assert len(result.calibration.model_comparisons) == 4
        assert len(result.instrument_profiles) > 0
        assert len(result.time_patterns) > 0
        assert len(result.budget_allocations) > 0

    def test_minimal_columns(self):
        """Only required columns — engine fills defaults."""
        df = pd.DataFrame({
            "price": [5.0, 6.0, 4.0],
            "observed_slippage": [0.01, 0.02, 0.015],
        })
        e = SlippageEngine()
        result = e.analyze(df)
        assert result.n_trades == 3


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, engine, trades):
        result = engine.analyze(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "slip.html"
            path = SlippageEngine.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Slippage Model Analysis" in content

    def test_contains_model_comparison(self, engine, trades):
        result = engine.analyze(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            SlippageEngine.generate_report(result, out)
            content = out.read_text()
            assert "Model Comparison" in content
            assert "fixed_bps" in content

    def test_contains_instrument(self, engine, trades):
        result = engine.analyze(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            SlippageEngine.generate_report(result, out)
            content = out.read_text()
            assert "Instrument" in content
            assert "SPY" in content

    def test_contains_time_chart(self, engine, trades):
        result = engine.analyze(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            SlippageEngine.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Time of Day" in content

    def test_contains_budget(self, engine, trades):
        result = engine.analyze(trades)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            SlippageEngine.generate_report(result, out)
            content = out.read_text()
            assert "Budget" in content

    def test_default_path(self, engine, trades):
        result = engine.analyze(trades)
        path = SlippageEngine.generate_report(result)
        assert path.exists()
        assert "slippage_model.html" in str(path)
        path.unlink(missing_ok=True)
