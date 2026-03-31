from __future__ import annotations

from dataclasses import asdict, fields
from datetime import time

import numpy as np
import pytest

from compass.smart_router import (
    DEFAULT_VENUES,
    AdverseSelectionEstimate,
    CostAttribution,
    QueueEstimate,
    RoutingDecision,
    RoutingResult,
    SmartRouter,
    VenueDef,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lit_venue(**overrides) -> VenueDef:
    defaults = dict(
        name="TEST_LIT",
        venue_type="lit",
        avg_spread_bps=1.0,
        fill_rate=0.80,
        latency_ms=0.5,
        rebate_bps=0.20,
        fee_bps=0.30,
    )
    defaults.update(overrides)
    return VenueDef(**defaults)


def _dark_venue(**overrides) -> VenueDef:
    defaults = dict(
        name="TEST_DARK",
        venue_type="dark",
        avg_spread_bps=0.4,
        fill_rate=0.30,
        latency_ms=2.0,
        rebate_bps=0.0,
        fee_bps=0.15,
    )
    defaults.update(overrides)
    return VenueDef(**defaults)


def _midday() -> time:
    return time(12, 0)


def _open_time() -> time:
    return time(9, 35)


def _close_time() -> time:
    return time(15, 45)


# ---------------------------------------------------------------------------
# 1. Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_venue_def_fields(self):
        names = {f.name for f in fields(VenueDef)}
        assert names == {"name", "venue_type", "avg_spread_bps", "fill_rate",
                         "latency_ms", "rebate_bps", "fee_bps"}

    def test_venue_def_asdict(self):
        v = _lit_venue()
        d = asdict(v)
        assert d["name"] == "TEST_LIT"
        assert d["venue_type"] == "lit"

    def test_routing_decision_fields(self):
        names = {f.name for f in fields(RoutingDecision)}
        assert "venue" in names and "quantity" in names

    def test_cost_attribution_fields(self):
        ca = CostAttribution("X", 100, 1.0, 0.3, 0.2, 0.05, 1.15)
        assert ca.total_cost_bps == 1.15

    def test_routing_result_fields(self):
        names = {f.name for f in fields(RoutingResult)}
        assert "savings_bps" in names


# ---------------------------------------------------------------------------
# 2. Venue scoring
# ---------------------------------------------------------------------------

class TestVenueScoring:
    def test_score_in_range(self):
        router = SmartRouter(current_time=_midday())
        for v in DEFAULT_VENUES:
            s = router.score_venue(v)
            assert 0.0 <= s <= 1.0, f"{v.name} score {s} out of range"

    def test_higher_fill_rate_higher_liquidity(self):
        router = SmartRouter(current_time=_midday())
        v_high = _lit_venue(fill_rate=0.95)
        v_low = _lit_venue(fill_rate=0.50, name="LOW")
        assert router._liquidity_score(v_high) > router._liquidity_score(v_low)

    def test_lower_cost_higher_cost_score(self):
        router = SmartRouter(current_time=_midday())
        cheap = _lit_venue(avg_spread_bps=0.5, fee_bps=0.1, rebate_bps=0.3)
        expensive = _lit_venue(avg_spread_bps=2.0, fee_bps=0.5, rebate_bps=0.0, name="EXP")
        assert router._cost_score(cheap) > router._cost_score(expensive)

    def test_lower_latency_higher_score(self):
        router = SmartRouter(current_time=_midday())
        fast = _lit_venue(latency_ms=0.1)
        slow = _lit_venue(latency_ms=4.0, name="SLOW")
        assert router._latency_score(fast) > router._latency_score(slow)

    def test_score_all_venues_sorted_descending(self):
        router = SmartRouter(current_time=_midday())
        scored = router.score_all_venues()
        scores = [s for _, s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_custom_weights_affect_scoring(self):
        heavy_liq = SmartRouter(
            weights={"liquidity": 0.9, "cost": 0.05, "latency": 0.05},
            current_time=_midday(),
        )
        heavy_cost = SmartRouter(
            weights={"liquidity": 0.05, "cost": 0.9, "latency": 0.05},
            current_time=_midday(),
        )
        v = _lit_venue(fill_rate=0.99, avg_spread_bps=3.0, fee_bps=1.0)
        assert heavy_liq.score_venue(v) > heavy_cost.score_venue(v)


# ---------------------------------------------------------------------------
# 3. Order splitting
# ---------------------------------------------------------------------------

class TestOrderSplitting:
    def test_total_quantity_preserved(self):
        router = SmartRouter(current_time=_midday())
        decisions = router.split_order(1000)
        assert sum(d.quantity for d in decisions) == 1000

    def test_no_zero_quantity_legs(self):
        router = SmartRouter(current_time=_midday())
        decisions = router.split_order(500)
        for d in decisions:
            assert d.quantity > 0

    def test_small_order_excludes_dark(self):
        router = SmartRouter(current_time=_midday(), dark_size_threshold=500)
        decisions = router.split_order(100)
        venue_types = {d.venue.venue_type for d in decisions}
        assert "dark" not in venue_types

    def test_large_order_may_include_dark(self):
        router = SmartRouter(current_time=_midday(), dark_size_threshold=500)
        decisions = router.split_order(5000)
        # should_use_dark may or may not be True depending on spread savings;
        # just verify total qty is preserved
        assert sum(d.quantity for d in decisions) == 5000

    def test_single_venue_gets_all(self):
        v = _lit_venue()
        router = SmartRouter(venues=[v], current_time=_midday())
        decisions = router.split_order(300)
        assert len(decisions) == 1
        assert decisions[0].quantity == 300


# ---------------------------------------------------------------------------
# 4. Dark-pool logic
# ---------------------------------------------------------------------------

class TestDarkPoolLogic:
    def test_small_order_no_dark(self):
        router = SmartRouter(current_time=_midday(), dark_size_threshold=500)
        assert router.should_use_dark(100) is False

    def test_threshold_boundary(self):
        router = SmartRouter(current_time=_midday(), dark_size_threshold=500)
        # Exactly at threshold
        result = router.should_use_dark(500)
        # Result depends on spread saving; just verify it returns bool
        assert isinstance(result, bool)

    def test_large_order_with_big_spread_saving(self):
        lit = _lit_venue(avg_spread_bps=3.0, fee_bps=0.30, rebate_bps=0.20)
        dark = _dark_venue(avg_spread_bps=0.2, fee_bps=0.10)
        router = SmartRouter(
            venues=[lit, dark],
            current_time=_midday(),
            dark_size_threshold=100,
            dark_spread_saving_min_bps=0.1,
        )
        assert router.should_use_dark(500) is True

    def test_no_dark_venues_returns_false(self):
        router = SmartRouter(venues=[_lit_venue()], current_time=_midday())
        assert router.should_use_dark(10000) is False

    def test_no_lit_venues_returns_false(self):
        router = SmartRouter(venues=[_dark_venue()], current_time=_midday())
        assert router.should_use_dark(10000) is False


# ---------------------------------------------------------------------------
# 5. Queue estimation
# ---------------------------------------------------------------------------

class TestQueueEstimation:
    def test_queue_position_nonnegative(self):
        router = SmartRouter(current_time=_midday())
        for v in DEFAULT_VENUES:
            qe = router.estimate_queue_position(v, 100)
            assert qe.queue_position >= 0

    def test_higher_fill_rate_smaller_queue(self):
        router = SmartRouter(current_time=_midday())
        v_high = _lit_venue(fill_rate=0.95)
        v_low = _lit_venue(fill_rate=0.50, name="LOW")
        q_high = router.estimate_queue_position(v_high, 100)
        q_low = router.estimate_queue_position(v_low, 100)
        assert q_high.queue_position < q_low.queue_position

    def test_fill_probability_in_range(self):
        router = SmartRouter(current_time=_midday())
        qe = router.estimate_queue_position(_lit_venue(), 100, depth=10000)
        assert 0.0 <= qe.fill_probability <= 1.0

    def test_expected_wait_positive(self):
        router = SmartRouter(current_time=_midday())
        qe = router.estimate_queue_position(_lit_venue(), 100)
        assert qe.expected_wait_ms > 0


# ---------------------------------------------------------------------------
# 6. Adverse selection
# ---------------------------------------------------------------------------

class TestAdverseSelection:
    def test_toxicity_score_in_range(self):
        router = SmartRouter(current_time=_midday())
        ae = router.estimate_adverse_selection(_lit_venue(), 100)
        assert 0.0 <= ae.toxicity_score <= 1.0

    def test_dark_has_higher_widening(self):
        router = SmartRouter(current_time=_midday())
        lit_v = _lit_venue(avg_spread_bps=1.0)
        dark_v = _dark_venue(avg_spread_bps=1.0, name="DARK_SAME")
        ae_lit = router.estimate_adverse_selection(lit_v, 500, depth=10000)
        ae_dark = router.estimate_adverse_selection(dark_v, 500, depth=10000)
        assert ae_dark.spread_widening_bps > ae_lit.spread_widening_bps

    def test_larger_order_more_toxic(self):
        router = SmartRouter(current_time=_midday())
        v = _lit_venue()
        ae_small = router.estimate_adverse_selection(v, 100, depth=10000)
        ae_large = router.estimate_adverse_selection(v, 5000, depth=10000)
        assert ae_large.toxicity_score >= ae_small.toxicity_score


# ---------------------------------------------------------------------------
# 7. Cost attribution
# ---------------------------------------------------------------------------

class TestCostAttribution:
    def test_total_cost_formula(self):
        router = SmartRouter(current_time=_midday())
        v = _lit_venue()
        ca = router.attribute_cost(v, 100)
        expected = ca.spread_cost_bps + ca.fee_cost_bps - ca.rebate_bps + ca.impact_cost_bps
        assert abs(ca.total_cost_bps - expected) < 1e-4

    def test_rebate_reduces_cost(self):
        router = SmartRouter(current_time=_midday())
        v_rebate = _lit_venue(rebate_bps=0.50)
        v_no_rebate = _lit_venue(rebate_bps=0.0, name="NO_REB")
        ca_r = router.attribute_cost(v_rebate, 100)
        ca_nr = router.attribute_cost(v_no_rebate, 100)
        assert ca_r.total_cost_bps < ca_nr.total_cost_bps


# ---------------------------------------------------------------------------
# 8. Time-of-day adjustments
# ---------------------------------------------------------------------------

class TestTimeOfDay:
    def test_is_open(self):
        router = SmartRouter(current_time=_open_time())
        assert router.is_open_or_close() is True

    def test_is_close(self):
        router = SmartRouter(current_time=_close_time())
        assert router.is_open_or_close() is True

    def test_midday_not_open_close(self):
        router = SmartRouter(current_time=_midday())
        assert router.is_open_or_close() is False

    def test_iex_boosted_at_open(self):
        iex = [v for v in DEFAULT_VENUES if v.name == "IEX"][0]
        router_open = SmartRouter(current_time=_open_time())
        router_mid = SmartRouter(current_time=_midday())
        assert router_open.score_venue(iex) > router_mid.score_venue(iex)

    def test_dark_boosted_at_close(self):
        dark = [v for v in DEFAULT_VENUES if v.venue_type == "dark"][0]
        router_close = SmartRouter(current_time=_close_time())
        router_mid = SmartRouter(current_time=_midday())
        assert router_close.score_venue(dark) > router_mid.score_venue(dark)

    def test_lit_penalized_at_open(self):
        nyse = [v for v in DEFAULT_VENUES if v.name == "NYSE"][0]
        router_open = SmartRouter(current_time=_open_time())
        router_mid = SmartRouter(current_time=_midday())
        assert router_open.score_venue(nyse) < router_mid.score_venue(nyse)


# ---------------------------------------------------------------------------
# 9. HTML report
# ---------------------------------------------------------------------------

class TestHTMLReport:
    def test_report_is_string(self):
        router = SmartRouter(current_time=_midday())
        report = router.generate_report(total_qty=500)
        assert isinstance(report, str)

    def test_report_contains_html_structure(self):
        router = SmartRouter(current_time=_midday())
        report = router.generate_report(total_qty=500)
        assert "<!DOCTYPE html>" in report
        assert "</html>" in report
        assert "<table>" in report

    def test_report_contains_venue_names(self):
        router = SmartRouter(current_time=_midday())
        report = router.generate_report(total_qty=500)
        for v in DEFAULT_VENUES:
            assert v.name in report

    def test_report_contains_cost_summary(self):
        router = SmartRouter(current_time=_midday())
        report = router.generate_report(total_qty=1000)
        assert "Smart routing cost" in report
        assert "Naive routing cost" in report
        assert "Savings" in report

    def test_report_contains_all_sections(self):
        router = SmartRouter(current_time=_midday())
        report = router.generate_report()
        assert "Venue Comparison" in report
        assert "Routing Decisions" in report
        assert "Cost Attribution" in report
        assert "Queue Position Estimates" in report
        assert "Adverse Selection Estimates" in report


# ---------------------------------------------------------------------------
# 10. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_quantity_order(self):
        router = SmartRouter(current_time=_midday())
        decisions = router.split_order(0)
        assert sum(d.quantity for d in decisions) == 0

    def test_single_share(self):
        router = SmartRouter(current_time=_midday())
        decisions = router.split_order(1)
        assert sum(d.quantity for d in decisions) == 1

    def test_very_large_order(self):
        router = SmartRouter(current_time=_midday())
        result = router.route_order(1_000_000)
        assert sum(d.quantity for d in result.decisions) == 1_000_000

    def test_zero_depth_no_crash(self):
        router = SmartRouter(current_time=_midday())
        qe = router.estimate_queue_position(_lit_venue(), 100, depth=0)
        assert qe.queue_position >= 0
        ae = router.estimate_adverse_selection(_lit_venue(), 100, depth=0)
        assert ae.toxicity_score >= 0

    def test_all_zero_scores_still_routes(self):
        """Venues with zero fill rate and high cost should still route."""
        bad = VenueDef("BAD", "lit", avg_spread_bps=10.0, fill_rate=0.0,
                       latency_ms=5.0, rebate_bps=0.0, fee_bps=5.0)
        router = SmartRouter(venues=[bad], current_time=_midday())
        decisions = router.split_order(100)
        assert sum(d.quantity for d in decisions) == 100

    def test_midpoint_venue_treated_as_dark(self):
        mid = VenueDef("MID", "midpoint", avg_spread_bps=0.3, fill_rate=0.25,
                       latency_ms=1.5, rebate_bps=0.0, fee_bps=0.10)
        router = SmartRouter(venues=[_lit_venue(), mid], current_time=_midday(),
                             dark_size_threshold=50)
        # For small orders, midpoint should be excluded like dark
        decisions = router.split_order(10)
        venue_types = {d.venue.venue_type for d in decisions}
        assert "midpoint" not in venue_types
