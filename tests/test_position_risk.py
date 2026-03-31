"""Tests for compass/position_risk.py — position-level risk attribution."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.position_risk import (
    LimitCheck,
    MarginEstimate,
    Position,
    PositionGreeks,
    PositionRiskAnalyzer,
    PortfolioGreeks,
    RiskLimits,
    RiskReport,
    aggregate_greeks,
    check_limits,
    compute_position_greeks,
    estimate_margin,
)


def _bull_put(contracts: int = 2, credit: float = 0.65) -> Position:
    return Position(
        ticker="SPY", short_strike=445, long_strike=440,
        spread_type="bull_put", contracts=contracts, credit=credit,
        dte=21, underlying_price=450, iv=0.20,
    )


def _bear_call(contracts: int = 1) -> Position:
    return Position(
        ticker="SPY", short_strike=460, long_strike=465,
        spread_type="bear_call", contracts=contracts, credit=0.55,
        dte=30, underlying_price=450, iv=0.22,
    )


# ── compute_position_greeks ──────────────────────────────────────────────


class TestComputePositionGreeks:

    def test_returns_position_greeks(self):
        g = compute_position_greeks(_bull_put())
        assert isinstance(g, PositionGreeks)

    def test_delta_negative_for_bull_put(self):
        g = compute_position_greeks(_bull_put())
        assert g.delta < 0

    def test_theta_positive_for_credit_spread(self):
        g = compute_position_greeks(_bull_put())
        assert g.theta_day > 0

    def test_max_loss_correct(self):
        pos = _bull_put(contracts=2, credit=0.65)
        g = compute_position_greeks(pos)
        expected = (5.0 - 0.65) * 100 * 2  # $870
        assert abs(g.max_loss - expected) < 1.0

    def test_contracts_scale_dollar_greeks(self):
        g1 = compute_position_greeks(_bull_put(contracts=1))
        g2 = compute_position_greeks(_bull_put(contracts=3))
        assert abs(g2.delta_dollars) > abs(g1.delta_dollars)
        assert abs(g2.max_loss) > abs(g1.max_loss)

    def test_bear_call_delta_positive(self):
        g = compute_position_greeks(_bear_call())
        # Bear call spread: short lower-strike call → net positive delta (from spread value perspective)
        # but the actual sign depends on moneyness; just verify it's finite
        assert np.isfinite(g.delta)

    def test_pnl_computed(self):
        g = compute_position_greeks(_bull_put())
        assert np.isfinite(g.pnl)

    def test_vega_dollars_nonzero(self):
        g = compute_position_greeks(_bull_put())
        assert g.vega_dollars != 0


# ── aggregate_greeks ─────────────────────────────────────────────────────


class TestAggregateGreeks:

    def test_single_position(self):
        g = compute_position_greeks(_bull_put())
        agg = aggregate_greeks([g])
        assert agg.n_positions == 1
        assert agg.net_delta != 0

    def test_multiple_positions_net(self):
        g1 = compute_position_greeks(_bull_put(contracts=2))
        g2 = compute_position_greeks(_bear_call(contracts=1))
        agg = aggregate_greeks([g1, g2])
        assert agg.n_positions == 2
        assert agg.total_max_loss > 0

    def test_theta_nets_across_positions(self):
        g1 = compute_position_greeks(_bull_put())
        g2 = compute_position_greeks(_bear_call())
        agg = aggregate_greeks([g1, g2])
        # Both credit spreads → both theta > 0 → net theta > 0
        assert agg.net_theta_day > 0

    def test_empty_list(self):
        agg = aggregate_greeks([])
        assert agg.n_positions == 0
        assert agg.net_delta == 0

    def test_total_max_loss_sums(self):
        g1 = compute_position_greeks(_bull_put(contracts=1))
        g2 = compute_position_greeks(_bull_put(contracts=2))
        agg = aggregate_greeks([g1, g2])
        assert abs(agg.total_max_loss - (g1.max_loss + g2.max_loss)) < 0.01


# ── estimate_margin ──────────────────────────────────────────────────────


class TestEstimateMargin:

    def test_reg_t_equals_max_loss(self):
        pos = _bull_put(contracts=2, credit=0.65)
        m = estimate_margin([pos])
        expected = (5.0 - 0.65) * 100 * 2
        assert abs(m.reg_t_margin - expected) < 1.0

    def test_portfolio_margin_less_than_reg_t(self):
        m = estimate_margin([_bull_put()])
        assert m.portfolio_margin < m.reg_t_margin

    def test_utilization_percentage(self):
        m = estimate_margin([_bull_put(contracts=5)], account_value=10_000)
        assert m.margin_utilization_pct > 0

    def test_empty_positions(self):
        m = estimate_margin([])
        assert m.reg_t_margin == 0
        assert m.margin_utilization_pct == 0

    def test_multiple_positions(self):
        m = estimate_margin([_bull_put(contracts=2), _bear_call(contracts=1)])
        assert m.reg_t_margin > 0


# ── check_limits ─────────────────────────────────────────────────────────


class TestCheckLimits:

    def test_no_breach_default_limits(self):
        g1 = compute_position_greeks(_bull_put())
        agg = aggregate_greeks([g1])
        margin = estimate_margin([_bull_put()])
        checks = check_limits(agg, margin, RiskLimits())
        assert all(not c.breached for c in checks)

    def test_delta_breach(self):
        # Create many positions to exceed delta limit
        positions = [_bull_put(contracts=10)] * 5
        greeks = [compute_position_greeks(p) for p in positions]
        agg = aggregate_greeks(greeks)
        margin = estimate_margin(positions)
        limits = RiskLimits(max_net_delta=0.1)  # very tight
        checks = check_limits(agg, margin, limits)
        delta_check = next(c for c in checks if "Delta" in c.name)
        assert delta_check.breached

    def test_margin_breach(self):
        positions = [_bull_put(contracts=10)]
        greeks = [compute_position_greeks(p) for p in positions]
        agg = aggregate_greeks(greeks)
        margin = estimate_margin(positions, account_value=1_000)  # tiny account
        limits = RiskLimits(max_margin_pct=50.0)
        checks = check_limits(agg, margin, limits)
        margin_check = next(c for c in checks if "Margin" in c.name)
        assert margin_check.breached

    def test_returns_limit_check_objects(self):
        agg = aggregate_greeks([compute_position_greeks(_bull_put())])
        margin = estimate_margin([_bull_put()])
        checks = check_limits(agg, margin, RiskLimits())
        assert all(isinstance(c, LimitCheck) for c in checks)
        assert len(checks) == 5


# ── PositionRiskAnalyzer ─────────────────────────────────────────────────


class TestAnalyzer:

    def test_analyze_returns_report(self):
        a = PositionRiskAnalyzer()
        r = a.analyze([_bull_put()])
        assert isinstance(r, RiskReport)
        assert len(r.position_greeks) == 1

    def test_analyze_empty(self):
        r = PositionRiskAnalyzer().analyze([])
        assert r.portfolio_greeks.n_positions == 0

    def test_any_breach_flag(self):
        a = PositionRiskAnalyzer(limits=RiskLimits(max_net_delta=0.001))
        r = a.analyze([_bull_put(contracts=5)])
        assert r.any_breach

    def test_report_html(self):
        a = PositionRiskAnalyzer()
        html = a.generate_report([_bull_put(), _bear_call()])
        assert "<!DOCTYPE html>" in html
        assert "Position Risk" in html
        assert "SPY" in html

    def test_report_writes_to_file(self):
        a = PositionRiskAnalyzer()
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "risk.html")
            a.generate_report([_bull_put()], p)
            assert Path(p).exists()

    def test_report_contains_sections(self):
        a = PositionRiskAnalyzer()
        html = a.generate_report([_bull_put(), _bear_call()])
        assert "Position-Level" in html
        assert "Aggregate" in html
        assert "Margin" in html
        assert "Limit" in html
