"""Tests for compass.execution_optimizer – 30+ tests covering cost estimation,
algo selection, venue routing, TCA, IS decomposition, time-of-day, monitoring,
HTML report generation, edge cases, and dataclass behaviour."""

from __future__ import annotations

import math

import numpy as np
import pytest

from compass.execution_optimizer import (
    DARK_POOL,
    DEFAULT_VENUES,
    IEX,
    ISDecomposition,
    NASDAQ,
    NYSE,
    AlgoName,
    ExecutionOptimizer,
    ExecutionResult,
    PostTradeTCA,
    PreTradeEstimate,
    TimeRecommendation,
    TradeMonitor,
    VenueAllocation,
    VenueDef,
)


@pytest.fixture
def optimizer() -> ExecutionOptimizer:
    return ExecutionOptimizer()


# ============================================================
# Pre-trade cost estimation
# ============================================================


class TestCostEstimation:
    def test_positive_cost(self, optimizer: ExecutionOptimizer) -> None:
        est = optimizer.estimate_cost(order_qty=10_000, adv=1_000_000, volatility=0.02)
        assert est.total_bps > 0
        assert est.market_impact_bps > 0
        assert est.timing_risk_bps > 0

    def test_cost_scales_with_size(self, optimizer: ExecutionOptimizer) -> None:
        small = optimizer.estimate_cost(order_qty=1_000, adv=1_000_000, volatility=0.02)
        large = optimizer.estimate_cost(order_qty=100_000, adv=1_000_000, volatility=0.02)
        assert large.market_impact_bps > small.market_impact_bps

    def test_cost_scales_with_volatility(self, optimizer: ExecutionOptimizer) -> None:
        low_vol = optimizer.estimate_cost(order_qty=10_000, adv=1_000_000, volatility=0.01)
        high_vol = optimizer.estimate_cost(order_qty=10_000, adv=1_000_000, volatility=0.05)
        assert high_vol.timing_risk_bps > low_vol.timing_risk_bps

    def test_total_equals_sum(self, optimizer: ExecutionOptimizer) -> None:
        est = optimizer.estimate_cost(order_qty=50_000, adv=500_000, volatility=0.03)
        assert math.isclose(est.total_bps, est.market_impact_bps + est.timing_risk_bps, rel_tol=1e-9)

    def test_zero_order_qty(self, optimizer: ExecutionOptimizer) -> None:
        est = optimizer.estimate_cost(order_qty=0, adv=1_000_000, volatility=0.02)
        assert est.total_bps == 0.0

    def test_zero_adv(self, optimizer: ExecutionOptimizer) -> None:
        est = optimizer.estimate_cost(order_qty=10_000, adv=0, volatility=0.02)
        assert est.total_bps == 0.0


# ============================================================
# Algorithm selection
# ============================================================


class TestAlgoSelection:
    def test_critical_always_IS(self, optimizer: ExecutionOptimizer) -> None:
        algo = optimizer.select_algorithm("critical", 1_000, 1_000_000, 0.01)
        assert algo == "IS"

    def test_high_urgency_high_vol(self, optimizer: ExecutionOptimizer) -> None:
        algo = optimizer.select_algorithm("high", 1_000, 1_000_000, 0.03)
        assert algo == "IS"

    def test_high_urgency_low_vol(self, optimizer: ExecutionOptimizer) -> None:
        algo = optimizer.select_algorithm("high", 1_000, 1_000_000, 0.01)
        assert algo == "VWAP"

    def test_medium_large_participation(self, optimizer: ExecutionOptimizer) -> None:
        algo = optimizer.select_algorithm("medium", 100_000, 1_000_000, 0.02)
        assert algo == "Iceberg"

    def test_medium_small_participation(self, optimizer: ExecutionOptimizer) -> None:
        algo = optimizer.select_algorithm("medium", 1_000, 1_000_000, 0.02)
        assert algo == "VWAP"

    def test_low_urgency_default(self, optimizer: ExecutionOptimizer) -> None:
        algo = optimizer.select_algorithm("low", 1_000, 1_000_000, 0.02)
        assert algo == "TWAP"

    def test_low_urgency_large_order(self, optimizer: ExecutionOptimizer) -> None:
        algo = optimizer.select_algorithm("low", 200_000, 1_000_000, 0.02)
        assert algo == "Iceberg"


# ============================================================
# Venue scoring and routing
# ============================================================


class TestVenueRouting:
    def test_weights_sum_to_one(self, optimizer: ExecutionOptimizer) -> None:
        allocations = optimizer.route_order()
        total_weight = sum(a.weight for a in allocations)
        assert math.isclose(total_weight, 1.0, rel_tol=1e-9)

    def test_scores_positive(self, optimizer: ExecutionOptimizer) -> None:
        allocations = optimizer.route_order()
        for a in allocations:
            assert a.score > 0

    def test_all_default_venues_present(self, optimizer: ExecutionOptimizer) -> None:
        allocations = optimizer.route_order()
        names = {a.venue.name for a in allocations}
        assert names == {"NYSE", "NASDAQ", "IEX", "DARK_POOL"}

    def test_single_venue(self) -> None:
        opt = ExecutionOptimizer(venues=[NYSE])
        allocations = opt.route_order()
        assert len(allocations) == 1
        assert math.isclose(allocations[0].weight, 1.0)

    def test_empty_venues(self) -> None:
        opt = ExecutionOptimizer(venues=[])
        allocations = opt.route_order()
        assert allocations == []

    def test_custom_venue(self) -> None:
        v = VenueDef("TEST", spread_bps=0.5, fill_rate=0.95, fee_bps=0.1, rebate_bps=0.05, latency_ms=0.2)
        opt = ExecutionOptimizer(venues=[v])
        alloc = opt.route_order()
        assert len(alloc) == 1
        assert alloc[0].venue.name == "TEST"


# ============================================================
# Post-trade TCA
# ============================================================


class TestTCA:
    def test_arrival_slippage_buy(self, optimizer: ExecutionOptimizer) -> None:
        tca = optimizer.compute_tca(
            avg_fill_price=100.10, arrival_price=100.00, vwap=100.05, close_price=100.20, side="buy"
        )
        assert tca.arrival_slippage_bps > 0  # paid more than arrival

    def test_arrival_slippage_sell(self, optimizer: ExecutionOptimizer) -> None:
        tca = optimizer.compute_tca(
            avg_fill_price=99.90, arrival_price=100.00, vwap=100.05, close_price=100.20, side="sell"
        )
        assert tca.arrival_slippage_bps > 0  # received less than arrival

    def test_vwap_benchmark(self, optimizer: ExecutionOptimizer) -> None:
        tca = optimizer.compute_tca(
            avg_fill_price=100.05, arrival_price=100.00, vwap=100.00, close_price=100.10, side="buy"
        )
        assert tca.vwap_slippage_bps > 0

    def test_close_benchmark(self, optimizer: ExecutionOptimizer) -> None:
        tca = optimizer.compute_tca(
            avg_fill_price=100.00, arrival_price=100.00, vwap=100.00, close_price=99.90, side="buy"
        )
        assert tca.close_slippage_bps > 0  # fill > close for buy

    def test_perfect_fill(self, optimizer: ExecutionOptimizer) -> None:
        tca = optimizer.compute_tca(
            avg_fill_price=100.00, arrival_price=100.00, vwap=100.00, close_price=100.00
        )
        assert tca.arrival_slippage_bps == 0.0
        assert tca.implementation_shortfall_bps == 0.0


# ============================================================
# IS decomposition
# ============================================================


class TestISDecomposition:
    def test_components_sum_to_total(self, optimizer: ExecutionOptimizer) -> None:
        d = optimizer.decompose_is(
            arrival_price=100.05, decision_price=100.00,
            avg_fill_price=100.15, close_price=100.20,
            filled_qty=8_000, order_qty=10_000, side="buy",
        )
        expected = d.timing_cost_bps + d.impact_cost_bps + d.opportunity_cost_bps
        assert math.isclose(d.total_bps, expected, rel_tol=1e-9)

    def test_timing_cost_positive_buy(self, optimizer: ExecutionOptimizer) -> None:
        d = optimizer.decompose_is(
            arrival_price=100.10, decision_price=100.00,
            avg_fill_price=100.15, close_price=100.20,
            filled_qty=10_000, order_qty=10_000, side="buy",
        )
        assert d.timing_cost_bps > 0  # arrival moved up from decision

    def test_opportunity_cost_unfilled(self, optimizer: ExecutionOptimizer) -> None:
        d = optimizer.decompose_is(
            arrival_price=100.00, decision_price=100.00,
            avg_fill_price=100.00, close_price=101.00,
            filled_qty=5_000, order_qty=10_000, side="buy",
        )
        assert d.opportunity_cost_bps > 0  # missed the rally on unfilled shares

    def test_fully_filled_no_opportunity(self, optimizer: ExecutionOptimizer) -> None:
        d = optimizer.decompose_is(
            arrival_price=100.00, decision_price=100.00,
            avg_fill_price=100.05, close_price=101.00,
            filled_qty=10_000, order_qty=10_000, side="buy",
        )
        assert d.opportunity_cost_bps == 0.0

    def test_zero_decision_price(self, optimizer: ExecutionOptimizer) -> None:
        d = optimizer.decompose_is(
            arrival_price=100.00, decision_price=0.0,
            avg_fill_price=100.05, close_price=101.00,
            filled_qty=10_000, order_qty=10_000,
        )
        assert d.total_bps == 0.0


# ============================================================
# Time-of-day recommendations
# ============================================================


class TestTimeOfDay:
    def test_open_high_vol(self, optimizer: ExecutionOptimizer) -> None:
        rec = optimizer.recommend_time(9)
        assert rec.volatility_regime == "high"
        assert rec.recommended_algo == "IS"

    def test_midday_low_vol(self, optimizer: ExecutionOptimizer) -> None:
        rec = optimizer.recommend_time(12)
        assert rec.volatility_regime == "low"
        assert rec.recommended_algo in ("TWAP", "VWAP")

    def test_close_high_vol(self, optimizer: ExecutionOptimizer) -> None:
        rec = optimizer.recommend_time(15)
        assert rec.volatility_regime == "high"
        assert rec.recommended_algo == "IS"

    def test_outside_hours(self, optimizer: ExecutionOptimizer) -> None:
        rec = optimizer.recommend_time(20)
        assert rec.volatility_regime == "unknown"
        assert rec.recommended_algo == "TWAP"


# ============================================================
# Trade monitoring
# ============================================================


class TestTradeMonitor:
    def test_abort_when_cost_exceeds_2x(self, optimizer: ExecutionOptimizer) -> None:
        mon = optimizer.create_monitor(order_qty=1000, arrival_price=100.0, pre_trade_estimate_bps=10.0)
        optimizer.update_monitor(mon, filled_qty=500, avg_price=100.30, elapsed_time=5.0)
        # 30 bps actual vs 10 bps estimate -> should abort
        assert mon.aborted is True
        assert "2x" in mon.abort_reason

    def test_no_abort_within_budget(self, optimizer: ExecutionOptimizer) -> None:
        mon = optimizer.create_monitor(order_qty=1000, arrival_price=100.0, pre_trade_estimate_bps=50.0)
        optimizer.update_monitor(mon, filled_qty=500, avg_price=100.05, elapsed_time=5.0)
        assert mon.aborted is False

    def test_fill_pct(self) -> None:
        mon = TradeMonitor(order_qty=1000, filled_qty=250)
        assert math.isclose(mon.fill_pct, 0.25)

    def test_fill_pct_zero_qty(self) -> None:
        mon = TradeMonitor(order_qty=0)
        assert mon.fill_pct == 0.0

    def test_current_cost_zero_arrival(self) -> None:
        mon = TradeMonitor(order_qty=100, arrival_price=0.0, avg_price=50.0)
        assert mon.current_cost_bps == 0.0


# ============================================================
# HTML report
# ============================================================


class TestHTMLReport:
    def test_report_contains_algo(self, optimizer: ExecutionOptimizer) -> None:
        result = optimizer.optimise(order_qty=10_000, adv=1_000_000, volatility=0.02)
        report = optimizer.generate_report(result)
        assert result.selected_algo in report

    def test_report_contains_venues(self, optimizer: ExecutionOptimizer) -> None:
        result = optimizer.optimise(order_qty=10_000, adv=1_000_000, volatility=0.02)
        report = optimizer.generate_report(result)
        for v in DEFAULT_VENUES:
            assert v.name in report

    def test_report_is_html(self, optimizer: ExecutionOptimizer) -> None:
        result = optimizer.optimise(order_qty=10_000, adv=1_000_000, volatility=0.02)
        report = optimizer.generate_report(result)
        assert report.startswith("<!DOCTYPE html>")
        assert "</html>" in report

    def test_report_with_tca(self, optimizer: ExecutionOptimizer) -> None:
        result = optimizer.optimise(
            order_qty=10_000, adv=1_000_000, volatility=0.02,
            avg_fill_price=100.05, arrival_price=100.00,
            vwap=100.02, close_price=100.10,
            decision_price=99.98, filled_qty=8_000, hour=10,
        )
        report = optimizer.generate_report(result)
        assert "Post-Trade TCA" in report
        assert "IS Decomposition" in report
        assert "Time-of-Day" in report


# ============================================================
# Dataclass basics
# ============================================================


class TestDataclasses:
    def test_venue_def_fields(self) -> None:
        v = VenueDef("X", 1.0, 0.5, 0.1, 0.05, 0.3)
        assert v.name == "X"
        assert v.fill_rate == 0.5

    def test_pre_trade_estimate_fields(self) -> None:
        p = PreTradeEstimate(10.0, 5.0, 15.0)
        assert p.total_bps == 15.0

    def test_execution_result_generated_at(self, optimizer: ExecutionOptimizer) -> None:
        result = optimizer.optimise(order_qty=100, adv=100_000, volatility=0.01)
        assert result.generated_at is not None
        assert len(result.generated_at) > 0

    def test_is_decomposition_fields(self) -> None:
        d = ISDecomposition(1.0, 2.0, 3.0, 6.0)
        assert d.timing_cost_bps == 1.0
        assert d.impact_cost_bps == 2.0
        assert d.opportunity_cost_bps == 3.0
        assert d.total_bps == 6.0


# ============================================================
# Edge cases
# ============================================================


class TestEdgeCases:
    def test_extreme_participation(self, optimizer: ExecutionOptimizer) -> None:
        est = optimizer.estimate_cost(order_qty=1_000_000, adv=1_000_000, volatility=0.05)
        assert est.market_impact_bps > 0
        assert np.isfinite(est.total_bps)

    def test_very_small_volatility(self, optimizer: ExecutionOptimizer) -> None:
        est = optimizer.estimate_cost(order_qty=10_000, adv=1_000_000, volatility=1e-10)
        assert est.timing_risk_bps < 0.01

    def test_optimise_returns_execution_result(self, optimizer: ExecutionOptimizer) -> None:
        result = optimizer.optimise(order_qty=5_000, adv=500_000, volatility=0.02)
        assert isinstance(result, ExecutionResult)
        assert isinstance(result.pre_trade_estimate, PreTradeEstimate)
        assert isinstance(result.venue_allocation, list)
        assert len(result.venue_allocation) == len(DEFAULT_VENUES)
