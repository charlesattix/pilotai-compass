"""Tests for compass/execution_simulator.py — trade execution simulation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.execution_simulator import (
    ExecutionSimulator,
    FillResult,
    LatencyConfig,
    MarketImpactConfig,
    OrderRequest,
    QueueModel,
    SimulationResult,
    SlippageConfig,
    SlippageModel,
    apply_impact_decay,
    compute_latency,
    compute_market_impact,
    compute_partial_fill,
    compute_queue_position,
    compute_slippage_bps,
    _fmt_bps,
    _fmt_dollar,
    _fmt_ms,
    _fmt_pct,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_orders(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "order_id": [f"ORD-{i:04d}" for i in range(n)],
        "side": rng.choice(["buy", "sell"], size=n),
        "price": rng.uniform(1.0, 10.0, size=n).round(2),
        "quantity": rng.randint(1, 20, size=n),
        "spread_width": rng.choice([2.0, 5.0, 10.0], size=n),
        "market_volume": rng.randint(100, 5000, size=n),
    })


@pytest.fixture
def orders():
    return _make_orders()


@pytest.fixture
def simulator():
    return ExecutionSimulator(seed=42)


@pytest.fixture
def single_order():
    return OrderRequest(
        order_id="TEST-001", side="buy", price=5.0,
        quantity=10, spread_width=5.0, market_volume=1000,
    )


# ── Slippage model tests ─────────────────────────────────────────────────


class TestSlippage:
    def test_fixed_bps_positive(self, single_order):
        cfg = SlippageConfig(model=SlippageModel.FIXED_BPS, fixed_bps=5.0)
        rng = np.random.RandomState(42)
        slip = compute_slippage_bps(cfg, single_order, rng)
        assert slip >= 0

    def test_fixed_bps_near_target(self, single_order):
        cfg = SlippageConfig(model=SlippageModel.FIXED_BPS, fixed_bps=10.0)
        slips = []
        for seed in range(100):
            rng = np.random.RandomState(seed)
            slips.append(compute_slippage_bps(cfg, single_order, rng))
        assert abs(np.mean(slips) - 10.0) < 2.0  # close to 10 bps on average

    def test_proportional_scales_with_width(self):
        cfg = SlippageConfig(model=SlippageModel.PROPORTIONAL, proportional_factor=0.1)
        rng = np.random.RandomState(42)
        narrow = OrderRequest("A", "buy", 5.0, 10, spread_width=1.0, market_volume=1000)
        wide = OrderRequest("B", "buy", 5.0, 10, spread_width=10.0, market_volume=1000)
        slip_narrow = compute_slippage_bps(cfg, narrow, rng)
        rng2 = np.random.RandomState(42)
        slip_wide = compute_slippage_bps(cfg, wide, rng2)
        # Wide spread should produce more slippage on average (same seed for fair comparison)
        # Due to noise, test with many samples
        slips_n = [compute_slippage_bps(cfg, narrow, np.random.RandomState(s)) for s in range(200)]
        slips_w = [compute_slippage_bps(cfg, wide, np.random.RandomState(s)) for s in range(200)]
        assert np.mean(slips_w) > np.mean(slips_n)

    def test_proportional_zero_price(self):
        cfg = SlippageConfig(model=SlippageModel.PROPORTIONAL)
        rng = np.random.RandomState(42)
        order = OrderRequest("A", "buy", 0.0, 10, spread_width=5.0, market_volume=1000)
        assert compute_slippage_bps(cfg, order, rng) == 0.0

    def test_volume_dependent_scales_with_size(self):
        cfg = SlippageConfig(model=SlippageModel.VOLUME_DEPENDENT, volume_impact_factor=0.5)
        small = OrderRequest("A", "buy", 5.0, 1, spread_width=5.0, market_volume=1000)
        large = OrderRequest("B", "buy", 5.0, 500, spread_width=5.0, market_volume=1000)
        slips_s = [compute_slippage_bps(cfg, small, np.random.RandomState(s)) for s in range(200)]
        slips_l = [compute_slippage_bps(cfg, large, np.random.RandomState(s)) for s in range(200)]
        assert np.mean(slips_l) > np.mean(slips_s)

    def test_volume_dependent_zero_volume(self):
        cfg = SlippageConfig(model=SlippageModel.VOLUME_DEPENDENT)
        rng = np.random.RandomState(42)
        order = OrderRequest("A", "buy", 5.0, 10, market_volume=0)
        slip = compute_slippage_bps(cfg, order, rng)
        assert slip >= 0


# ── Queue position tests ─────────────────────────────────────────────────


class TestQueuePosition:
    def test_time_priority_bounded(self, single_order):
        rng = np.random.RandomState(42)
        pos = compute_queue_position(QueueModel.TIME_PRIORITY, single_order, rng)
        assert 0.0 <= pos <= 1.0

    def test_pro_rata_bounded(self, single_order):
        rng = np.random.RandomState(42)
        pos = compute_queue_position(QueueModel.PRO_RATA, single_order, rng)
        assert 0.0 <= pos <= 1.0

    def test_time_priority_larger_orders_worse(self):
        small = OrderRequest("A", "buy", 5.0, 1, market_volume=1000)
        large = OrderRequest("B", "buy", 5.0, 500, market_volume=1000)
        pos_s = [compute_queue_position(QueueModel.TIME_PRIORITY, small, np.random.RandomState(s)) for s in range(300)]
        pos_l = [compute_queue_position(QueueModel.TIME_PRIORITY, large, np.random.RandomState(s)) for s in range(300)]
        assert np.mean(pos_l) > np.mean(pos_s)  # larger → worse queue position


# ── Partial fill tests ───────────────────────────────────────────────────


class TestPartialFill:
    def test_at_least_one_fill(self):
        rng = np.random.RandomState(42)
        filled = compute_partial_fill(0.9, 10, 1000, rng)
        assert filled >= 1

    def test_zero_quantity(self):
        rng = np.random.RandomState(42)
        assert compute_partial_fill(0.5, 0, 1000, rng) == 0

    def test_front_of_queue_fills_more(self):
        fills_front = [compute_partial_fill(0.1, 20, 1000, np.random.RandomState(s)) for s in range(300)]
        fills_back = [compute_partial_fill(0.9, 20, 1000, np.random.RandomState(s)) for s in range(300)]
        assert np.mean(fills_front) > np.mean(fills_back)

    def test_bounded_by_requested(self):
        for seed in range(50):
            rng = np.random.RandomState(seed)
            filled = compute_partial_fill(0.3, 10, 1000, rng)
            assert 1 <= filled <= 10


# ── Market impact tests ─────────────────────────────────────────────────


class TestMarketImpact:
    def test_impact_non_negative(self, single_order):
        cfg = MarketImpactConfig()
        rng = np.random.RandomState(42)
        temp, perm = compute_market_impact(cfg, single_order, rng)
        assert temp >= 0
        assert perm >= 0

    def test_impact_scales_with_volume(self):
        cfg = MarketImpactConfig(temporary_impact_bps=5.0, permanent_impact_bps=2.0)
        small = OrderRequest("A", "buy", 5.0, 1, market_volume=1000)
        large = OrderRequest("B", "buy", 5.0, 500, market_volume=1000)
        temps_s = [compute_market_impact(cfg, small, np.random.RandomState(s))[0] for s in range(200)]
        temps_l = [compute_market_impact(cfg, large, np.random.RandomState(s))[0] for s in range(200)]
        assert np.mean(temps_l) > np.mean(temps_s)

    def test_decay_at_zero(self):
        assert apply_impact_decay(10.0, 0.0, 30.0) == 10.0

    def test_decay_at_half_life(self):
        result = apply_impact_decay(10.0, 30.0, 30.0)
        assert abs(result - 5.0) < 0.01

    def test_decay_approaches_zero(self):
        result = apply_impact_decay(10.0, 300.0, 30.0)
        assert result < 0.01

    def test_decay_zero_half_life(self):
        assert apply_impact_decay(10.0, 5.0, 0.0) == 10.0


# ── Latency tests ────────────────────────────────────────────────────────


class TestLatency:
    def test_latency_positive(self):
        cfg = LatencyConfig()
        rng = np.random.RandomState(42)
        lat = compute_latency(cfg, rng)
        assert lat >= 1.0

    def test_latency_near_base(self):
        cfg = LatencyConfig(base_latency_ms=50.0, jitter_ms=5.0, network_latency_ms=10.0)
        lats = [compute_latency(cfg, np.random.RandomState(s)) for s in range(200)]
        avg = np.mean(lats)
        assert 55.0 < avg < 70.0  # base + network + abs(jitter)


# ── FillResult tests ─────────────────────────────────────────────────────


class TestFillResult:
    def test_partial_fill_flag(self):
        fr = FillResult(
            order_id="A", side="buy", requested_price=5.0,
            requested_quantity=10, filled_price=5.01, filled_quantity=7,
            slippage_bps=5.0, slippage_dollars=3.5, fill_ratio=0.7,
            latency_ms=55.0, queue_position=0.3,
            temporary_impact_bps=2.0, permanent_impact_bps=1.0,
            total_impact_bps=8.0,
        )
        assert fr.is_partial_fill is True
        assert fr.is_complete_fill is False

    def test_complete_fill_flag(self):
        fr = FillResult(
            order_id="A", side="buy", requested_price=5.0,
            requested_quantity=10, filled_price=5.01, filled_quantity=10,
            slippage_bps=5.0, slippage_dollars=5.0, fill_ratio=1.0,
            latency_ms=55.0, queue_position=0.1,
            temporary_impact_bps=2.0, permanent_impact_bps=1.0,
            total_impact_bps=8.0,
        )
        assert fr.is_complete_fill is True
        assert fr.is_partial_fill is False

    def test_to_dict(self):
        fr = FillResult(
            order_id="X", side="sell", requested_price=3.0,
            requested_quantity=5, filled_price=2.98, filled_quantity=5,
            slippage_bps=6.0, slippage_dollars=1.0, fill_ratio=1.0,
            latency_ms=50.0, queue_position=0.2,
            temporary_impact_bps=1.5, permanent_impact_bps=0.5,
            total_impact_bps=8.0,
        )
        d = fr.to_dict()
        assert d["order_id"] == "X"
        assert "is_partial_fill" in d


# ── Simulator integration tests ──────────────────────────────────────────


class TestSimulatorIntegration:
    def test_simulate_single_buy(self, simulator, single_order):
        rng = np.random.RandomState(42)
        fill = simulator.simulate_single(single_order, rng)
        assert isinstance(fill, FillResult)
        assert fill.filled_price >= single_order.price  # buy → price goes up
        assert fill.filled_quantity >= 1

    def test_simulate_single_sell(self, simulator):
        order = OrderRequest("S1", "sell", 5.0, 10, 5.0, 1000)
        rng = np.random.RandomState(42)
        fill = simulator.simulate_single(order, rng)
        assert fill.filled_price <= order.price  # sell → price goes down

    def test_simulate_orders_batch(self, simulator, orders):
        result = simulator.simulate_orders(orders)
        assert isinstance(result, SimulationResult)
        assert len(result.fills) == 50
        assert result.summary["n_orders"] == 50

    def test_missing_columns_raises(self, simulator):
        bad = pd.DataFrame({"foo": [1]})
        with pytest.raises(ValueError, match="Missing required"):
            simulator.simulate_orders(bad)

    def test_reproducible(self, orders):
        s1 = ExecutionSimulator(seed=42)
        s2 = ExecutionSimulator(seed=42)
        r1 = s1.simulate_orders(orders)
        r2 = s2.simulate_orders(orders)
        assert r1.fills[0].filled_price == r2.fills[0].filled_price
        assert r1.fills[0].slippage_bps == r2.fills[0].slippage_bps

    def test_different_slippage_models(self, orders):
        for model in SlippageModel:
            cfg = SlippageConfig(model=model)
            sim = ExecutionSimulator(slippage_config=cfg, seed=42)
            result = sim.simulate_orders(orders)
            assert result.summary["avg_slippage_bps"] >= 0

    def test_different_queue_models(self, orders):
        for qm in QueueModel:
            sim = ExecutionSimulator(queue_model=qm, seed=42)
            result = sim.simulate_orders(orders)
            assert result.summary["avg_fill_ratio"] > 0


# ── Summary tests ────────────────────────────────────────────────────────


class TestSummary:
    def test_summary_keys(self, simulator, orders):
        result = simulator.simulate_orders(orders)
        s = result.summary
        assert "n_orders" in s
        assert "avg_slippage_bps" in s
        assert "p95_slippage_bps" in s
        assert "total_slippage_dollars" in s
        assert s["n_complete"] + s["n_partial"] == s["n_orders"]

    def test_empty_summary(self):
        s = ExecutionSimulator._compute_summary([])
        assert s["n_orders"] == 0
        assert s["avg_slippage_bps"] == 0.0


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generate_report_creates_file(self, simulator, orders):
        result = simulator.simulate_orders(orders)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_exec.html"
            path = ExecutionSimulator.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Execution Simulation" in content

    def test_report_contains_charts(self, simulator, orders):
        result = simulator.simulate_orders(orders)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            ExecutionSimulator.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Slippage Distribution" in content
            assert "Market Impact Decay" in content

    def test_report_contains_config(self, simulator, orders):
        result = simulator.simulate_orders(orders)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            ExecutionSimulator.generate_report(result, out)
            content = out.read_text()
            assert "Configuration" in content

    def test_formatters(self):
        assert _fmt_bps(5.12) == "5.12 bps"
        assert _fmt_dollar(1234.5) == "$1,234.50"
        assert _fmt_ms(55.3) == "55.3 ms"
        assert _fmt_pct(0.85) == "85.0%"

    def test_report_default_path(self, simulator, orders):
        result = simulator.simulate_orders(orders)
        path = ExecutionSimulator.generate_report(result)
        assert path.exists()
        assert "execution_sim.html" in str(path)
        path.unlink(missing_ok=True)
