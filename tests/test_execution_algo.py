"""Tests for compass.execution_algo – smart execution algorithm engine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.execution_algo import (
    MARKET_MINUTES,
    AlgoComparison,
    AlgoConfig,
    AlgoSchedule,
    AlgoType,
    BenchmarkResult,
    ExecutionAlgoEngine,
    ExecutionResult,
    SliceOrder,
    Urgency,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _engine() -> ExecutionAlgoEngine:
    return ExecutionAlgoEngine()


def _full_result() -> ExecutionResult:
    return ExecutionAlgoEngine().execute(
        total_quantity=100, urgency=Urgency.MEDIUM, adv=5000,
        arrival_price=2.50, vwap_price=2.48, close_price=2.52,
        avg_fill_price=2.49,
    )


# ── Constructor ─────────────────────────────────────────────────────────────
class TestInit:
    def test_defaults(self):
        e = ExecutionAlgoEngine()
        assert e.config.eta == 0.10
        assert e.config.clip_size == 10

    def test_custom_config(self):
        cfg = AlgoConfig(eta=0.20, clip_size=25, dark_pool_threshold=100)
        e = ExecutionAlgoEngine(config=cfg)
        assert e.config.eta == 0.20
        assert e.config.clip_size == 25


# ── Execute ─────────────────────────────────────────────────────────────────
class TestExecute:
    def test_returns_result(self):
        result = _engine().execute(100, Urgency.MEDIUM, adv=5000)
        assert isinstance(result, ExecutionResult)

    def test_selected_algo_set(self):
        result = _engine().execute(100, Urgency.MEDIUM, adv=5000)
        assert result.selected_algo in (AlgoType.TWAP, AlgoType.VWAP, AlgoType.IS, AlgoType.ICEBERG)

    def test_schedule_present(self):
        result = _engine().execute(100, Urgency.MEDIUM, adv=5000)
        assert result.schedule is not None
        assert result.schedule.total_quantity == 100

    def test_comparisons_for_all_algos(self):
        result = _engine().execute(100, Urgency.MEDIUM, adv=5000)
        algo_names = {c.algo for c in result.comparisons}
        assert AlgoType.TWAP in algo_names
        assert AlgoType.VWAP in algo_names
        assert AlgoType.IS in algo_names
        assert AlgoType.ICEBERG in algo_names

    def test_zero_quantity(self):
        result = _engine().execute(0, Urgency.LOW, adv=5000)
        assert result.schedule is None

    def test_generated_at(self):
        result = _engine().execute(50, Urgency.LOW, adv=5000)
        assert len(result.generated_at) > 0


# ── Algorithm selection ─────────────────────────────────────────────────────
class TestAlgoSelection:
    def test_critical_selects_is(self):
        result = _engine().execute(100, Urgency.CRITICAL, adv=5000)
        assert result.selected_algo == AlgoType.IS

    def test_low_urgency_selects_vwap(self):
        result = _engine().execute(100, Urgency.LOW, adv=5000)
        assert result.selected_algo == AlgoType.VWAP

    def test_high_urgency_selects_is(self):
        result = _engine().execute(100, Urgency.HIGH, adv=5000)
        assert result.selected_algo == AlgoType.IS

    def test_medium_selects_twap(self):
        result = _engine().execute(100, Urgency.MEDIUM, adv=5000)
        assert result.selected_algo == AlgoType.TWAP

    def test_large_order_selects_iceberg(self):
        result = _engine().execute(1000, Urgency.MEDIUM, adv=5000)
        assert result.selected_algo == AlgoType.ICEBERG


# ── TWAP ────────────────────────────────────────────────────────────────────
class TestTWAP:
    def test_slices_sum_to_quantity(self):
        sched = _engine()._twap(100, Urgency.MEDIUM, 5000)
        total = sum(s.quantity for s in sched.slices)
        assert total == 100

    def test_even_distribution(self):
        sched = _engine()._twap(100, Urgency.MEDIUM, 5000)
        qtys = [s.quantity for s in sched.slices]
        # TWAP should be roughly equal across slices
        assert max(qtys) - min(qtys) <= 2

    def test_time_increasing(self):
        sched = _engine()._twap(100, Urgency.LOW, 5000)
        times = [s.scheduled_time_min for s in sched.slices]
        assert times == sorted(times)


# ── VWAP ────────────────────────────────────────────────────────────────────
class TestVWAP:
    def test_slices_sum_to_quantity(self):
        sched = _engine()._vwap(100, Urgency.MEDIUM, 5000)
        total = sum(s.quantity for s in sched.slices)
        assert total == 100

    def test_follows_volume_profile(self):
        """First and last slices should be larger (U-shaped volume)."""
        sched = _engine()._vwap(200, Urgency.LOW, 5000)
        if len(sched.slices) >= 3:
            first = sched.slices[0].quantity
            mid = sched.slices[len(sched.slices) // 2].quantity
            assert first >= mid  # opening > midday

    def test_duration_matches_urgency(self):
        low = _engine()._vwap(100, Urgency.LOW, 5000)
        high = _engine()._vwap(100, Urgency.HIGH, 5000)
        assert low.duration_minutes >= high.duration_minutes


# ── Implementation Shortfall ────────────────────────────────────────────────
class TestIS:
    def test_slices_sum_to_quantity(self):
        sched = _engine()._implementation_shortfall(100, Urgency.HIGH, 5000)
        total = sum(s.quantity for s in sched.slices)
        assert total == 100

    def test_front_loaded_high_urgency(self):
        """High urgency should front-load execution."""
        sched = _engine()._implementation_shortfall(100, Urgency.HIGH, 5000)
        if len(sched.slices) >= 2:
            assert sched.slices[0].quantity >= sched.slices[-1].quantity

    def test_critical_fewer_slices(self):
        crit = _engine()._implementation_shortfall(100, Urgency.CRITICAL, 5000)
        low = _engine()._implementation_shortfall(100, Urgency.LOW, 5000)
        assert crit.n_slices <= low.n_slices


# ── Iceberg ─────────────────────────────────────────────────────────────────
class TestIceberg:
    def test_slices_sum_to_quantity(self):
        sched = _engine()._iceberg(100, 5000)
        total = sum(s.quantity for s in sched.slices)
        assert total == 100

    def test_visible_qty_capped(self):
        cfg = AlgoConfig(clip_size=10)
        e = ExecutionAlgoEngine(config=cfg)
        sched = e._iceberg(50, 5000)
        for s in sched.slices:
            if s.is_visible:
                assert s.quantity <= 10

    def test_has_hidden_slices(self):
        cfg = AlgoConfig(clip_size=5)
        e = ExecutionAlgoEngine(config=cfg)
        sched = e._iceberg(20, 5000)
        hidden = [s for s in sched.slices if not s.is_visible]
        assert len(hidden) > 0


# ── Dark pool routing ───────────────────────────────────────────────────────
class TestDarkPool:
    def test_large_order_uses_dark(self):
        cfg = AlgoConfig(dark_pool_threshold=10)
        e = ExecutionAlgoEngine(config=cfg)
        sched = e._twap(50, Urgency.MEDIUM, 5000)
        dark_slices = [s for s in sched.slices if s.is_dark]
        assert len(dark_slices) > 0

    def test_small_order_no_dark(self):
        cfg = AlgoConfig(dark_pool_threshold=1000)
        e = ExecutionAlgoEngine(config=cfg)
        sched = e._twap(50, Urgency.MEDIUM, 5000)
        dark_slices = [s for s in sched.slices if s.is_dark]
        assert len(dark_slices) == 0


# ── Benchmarking ────────────────────────────────────────────────────────────
class TestBenchmark:
    def test_benchmark_present(self):
        result = _full_result()
        assert result.benchmark is not None

    def test_benchmark_fields(self):
        result = _full_result()
        b = result.benchmark
        assert isinstance(b.arrival_cost_bps, float)
        assert isinstance(b.vwap_cost_bps, float)
        assert isinstance(b.close_cost_bps, float)
        assert b.best_benchmark in ("arrival", "vwap", "close")

    def test_no_benchmark_without_fill(self):
        result = _engine().execute(100, Urgency.MEDIUM, adv=5000)
        assert result.benchmark is None

    def test_buy_below_arrival_negative_cost(self):
        result = _engine().execute(
            100, Urgency.MEDIUM, adv=5000,
            arrival_price=2.50, avg_fill_price=2.48, side="buy",
        )
        # Bought below arrival → negative cost (good)
        assert result.benchmark.arrival_cost_bps < 0


# ── Cost savings ────────────────────────────────────────────────────────────
class TestCostSavings:
    def test_savings_nonnegative(self):
        result = _engine().execute(100, Urgency.MEDIUM, adv=5000)
        assert result.cost_savings_bps >= 0


# ── Algo comparison ─────────────────────────────────────────────────────────
class TestComparison:
    def test_comparisons_sorted_by_cost(self):
        result = _engine().execute(100, Urgency.MEDIUM, adv=5000)
        costs = [c.total_cost_bps for c in result.comparisons]
        assert costs == sorted(costs)

    def test_all_costs_positive(self):
        result = _engine().execute(100, Urgency.MEDIUM, adv=5000)
        for c in result.comparisons:
            assert c.expected_cost_bps > 0
            assert c.expected_risk_bps > 0


# ── Order splitting ─────────────────────────────────────────────────────────
class TestSplitting:
    def test_respects_min_slice(self):
        cfg = AlgoConfig(min_slice_qty=5)
        e = ExecutionAlgoEngine(config=cfg)
        slices = e._split_quantity(20, 10)
        assert all(s >= 5 for s in slices)

    def test_split_sums_correctly(self):
        e = _engine()
        for qty in [7, 50, 100, 333]:
            for n in [1, 3, 7, 13]:
                slices = e._split_quantity(qty, n)
                assert sum(slices) == qty


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = _engine()
            result = _full_result()
            path = e.generate_report(result, output_path=Path(tmp) / "ea.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = _engine()
            result = _full_result()
            path = e.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Execution Algorithm" in html
            assert "Comparison" in html
            assert "Execution Quality" in html
            assert "Schedule" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = _engine()
            result = e.execute(50, Urgency.LOW, adv=5000)
            path = e.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_slice_order(self):
        s = SliceOrder(0, 10, 30, 0.05)
        assert s.quantity == 10
        assert s.is_visible is True

    def test_algo_schedule(self):
        a = AlgoSchedule(AlgoType.TWAP, 100, 5, [], 180, 0.02, 0.1)
        assert a.algo == AlgoType.TWAP

    def test_benchmark_result(self):
        b = BenchmarkResult(-5.0, 3.0, 2.0, -5.0, "arrival")
        assert b.best_benchmark == "arrival"

    def test_execution_result_defaults(self):
        r = ExecutionResult()
        assert r.selected_algo == ""
        assert r.schedule is None
