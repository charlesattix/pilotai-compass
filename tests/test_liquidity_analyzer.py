"""Tests for compass.liquidity_analyzer – liquidity and capacity analysis."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.liquidity_analyzer import (
    DEFAULT_AUM_GRID,
    CapacityPoint,
    ExperimentCapacity,
    FillQuality,
    LiquidityAnalyzer,
    LiquidityResult,
    MarketParams,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_experiments(n: int = 3) -> dict:
    return {
        f"EXP-{i+1}": {
            "sharpe": 2.0 - i * 0.3,
            "avg_notional": 5_000,
            "trades_per_day": 1.0,
        }
        for i in range(n)
    }


# ── Constructor ─────────────────────────────────────────────────────────────
class TestLiquidityAnalyzerInit:
    def test_default_params(self):
        la = LiquidityAnalyzer()
        assert la.market.adv_contracts == 5_000
        assert la.market.bid_ask_bps == 10.0
        assert la.eta == 0.10
        assert la.alpha == 0.50

    def test_custom_market_params(self):
        mp = MarketParams(adv_contracts=10_000, bid_ask_bps=5.0)
        la = LiquidityAnalyzer(market_params=mp)
        assert la.market.adv_contracts == 10_000

    def test_custom_impact_params(self):
        la = LiquidityAnalyzer(impact_coeff=0.20, impact_exponent=0.60)
        assert la.eta == 0.20
        assert la.alpha == 0.60

    def test_custom_aum_grid(self):
        grid = [1e6, 5e6, 10e6]
        la = LiquidityAnalyzer(aum_grid=grid)
        assert la.aum_grid == grid


# ── Market impact model ─────────────────────────────────────────────────────
class TestMarketImpact:
    def test_zero_order_zero_impact(self):
        la = LiquidityAnalyzer()
        assert la.market_impact_bps(0) == 0.0

    def test_impact_increases_with_size(self):
        la = LiquidityAnalyzer()
        small = la.market_impact_bps(10)
        large = la.market_impact_bps(100)
        assert large > small

    def test_impact_nonnegative(self):
        la = LiquidityAnalyzer()
        for n in [1, 10, 100, 500, 1000]:
            assert la.market_impact_bps(n) >= 0

    def test_square_root_scaling(self):
        """Impact should scale roughly as sqrt(participation)."""
        la = LiquidityAnalyzer()
        i1 = la.market_impact_bps(100)
        i4 = la.market_impact_bps(400)
        # 4× order → 2× impact (sqrt)
        ratio = i4 / i1 if i1 > 0 else 0
        assert 1.8 < ratio < 2.2


# ── Effective spread ────────────────────────────────────────────────────────
class TestEffectiveSpread:
    def test_base_spread_at_tiny_order(self):
        la = LiquidityAnalyzer()
        spread = la.effective_spread_bps(1)
        assert spread >= la.market.bid_ask_bps

    def test_spread_widens_with_size(self):
        la = LiquidityAnalyzer()
        small = la.effective_spread_bps(10)
        large = la.effective_spread_bps(500)
        assert large > small

    def test_zero_order_returns_base(self):
        la = LiquidityAnalyzer()
        assert la.effective_spread_bps(0) == la.market.bid_ask_bps


# ── Fill probability ────────────────────────────────────────────────────────
class TestFillProbability:
    def test_small_order_high_fill(self):
        la = LiquidityAnalyzer()
        assert la.fill_probability(1) > 0.95

    def test_huge_order_low_fill(self):
        la = LiquidityAnalyzer()
        assert la.fill_probability(5000) < 0.5

    def test_fill_bounded(self):
        la = LiquidityAnalyzer()
        for n in [1, 50, 250, 1000, 5000]:
            p = la.fill_probability(n)
            assert 0.0 <= p <= 1.0

    def test_fill_decreases_with_size(self):
        la = LiquidityAnalyzer()
        p_small = la.fill_probability(10)
        p_large = la.fill_probability(1000)
        assert p_small >= p_large


# ── Participation rate ──────────────────────────────────────────────────────
class TestParticipationRate:
    def test_participation_bounded(self):
        la = LiquidityAnalyzer()
        assert la.participation_rate(250) == 250 / 5000

    def test_zero_adv_zero_rate(self):
        la = LiquidityAnalyzer(market_params=MarketParams(adv_contracts=0))
        assert la.participation_rate(100) == 0.0


# ── Total execution cost ───────────────────────────────────────────────────
class TestTotalCost:
    def test_cost_positive_for_nonzero_order(self):
        la = LiquidityAnalyzer()
        assert la.total_execution_cost_bps(50) > 0

    def test_cost_increases_with_size(self):
        la = LiquidityAnalyzer()
        small = la.total_execution_cost_bps(10)
        large = la.total_execution_cost_bps(500)
        assert large > small


# ── Full analysis ───────────────────────────────────────────────────────────
class TestAnalyze:
    def test_returns_liquidity_result(self):
        result = LiquidityAnalyzer().analyze()
        assert isinstance(result, LiquidityResult)

    def test_capacity_curve_populated(self):
        result = LiquidityAnalyzer().analyze()
        assert len(result.capacity_curve) == len(DEFAULT_AUM_GRID)

    def test_fill_quality_populated(self):
        result = LiquidityAnalyzer().analyze()
        assert len(result.fill_quality) > 0

    def test_max_safe_order_positive(self):
        result = LiquidityAnalyzer().analyze()
        assert result.max_safe_order_contracts > 0

    def test_portfolio_max_aum_positive(self):
        result = LiquidityAnalyzer().analyze()
        assert result.portfolio_max_aum > 0

    def test_generated_at_set(self):
        result = LiquidityAnalyzer().analyze()
        assert len(result.generated_at) > 0


# ── Capacity curve ──────────────────────────────────────────────────────────
class TestCapacityCurve:
    def test_sharpe_decreases_with_aum(self):
        result = LiquidityAnalyzer().analyze(base_sharpe=2.0)
        sharpes = [p.expected_sharpe for p in result.capacity_curve]
        # Should be monotonically non-increasing
        for i in range(1, len(sharpes)):
            assert sharpes[i] <= sharpes[i - 1] + 1e-9

    def test_degradation_increases_with_aum(self):
        result = LiquidityAnalyzer().analyze(base_sharpe=2.0)
        degs = [p.sharpe_degradation_pct for p in result.capacity_curve]
        for i in range(1, len(degs)):
            assert degs[i] >= degs[i - 1] - 1e-9

    def test_first_point_low_degradation(self):
        result = LiquidityAnalyzer().analyze(base_sharpe=2.0)
        assert result.capacity_curve[0].sharpe_degradation_pct < 10


# ── Per-experiment capacity ─────────────────────────────────────────────────
class TestExperimentCapacity:
    def test_experiments_populated(self):
        exps = _make_experiments(3)
        result = LiquidityAnalyzer().analyze(experiments=exps)
        assert len(result.experiment_capacities) == 3

    def test_capacity_fields(self):
        exps = _make_experiments(1)
        result = LiquidityAnalyzer().analyze(experiments=exps)
        c = result.experiment_capacities[0]
        assert c.experiment_id == "EXP-1"
        assert c.base_sharpe > 0
        assert c.max_aum > 0
        assert c.recommended_aum > 0
        assert c.recommended_aum < c.max_aum

    def test_ci_bounds(self):
        exps = _make_experiments(1)
        result = LiquidityAnalyzer().analyze(experiments=exps)
        c = result.experiment_capacities[0]
        assert c.ci_lower < c.recommended_aum
        assert c.ci_upper > c.max_aum

    def test_lower_trades_per_day_higher_capacity(self):
        """Fewer trades/day means less market impact → higher capacity."""
        exps = {
            "FAST": {"sharpe": 2.0, "avg_notional": 5000, "trades_per_day": 5.0},
            "SLOW": {"sharpe": 2.0, "avg_notional": 5000, "trades_per_day": 0.5},
        }
        result = LiquidityAnalyzer().analyze(experiments=exps)
        caps = {c.experiment_id: c for c in result.experiment_capacities}
        assert caps["SLOW"].max_aum > caps["FAST"].max_aum

    def test_no_experiments_empty(self):
        result = LiquidityAnalyzer().analyze()
        assert result.experiment_capacities == []


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            la = LiquidityAnalyzer()
            result = la.analyze(experiments=_make_experiments())
            path = la.generate_report(result, output_path=Path(tmp) / "l.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            la = LiquidityAnalyzer()
            result = la.analyze(experiments=_make_experiments())
            path = la.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Liquidity" in html
            assert "Capacity Curve" in html
            assert "Market Impact" in html
            assert "Fill Quality" in html
            assert "Per-Experiment" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            la = LiquidityAnalyzer()
            result = la.analyze()
            path = la.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_market_params_defaults(self):
        mp = MarketParams()
        assert mp.adv_contracts == 5_000
        assert mp.contract_multiplier == 100

    def test_fill_quality(self):
        fq = FillQuality(10, 0.002, 10.5, 3.2, 8.45, 0.99)
        assert fq.order_contracts == 10

    def test_capacity_point(self):
        cp = CapacityPoint(1e6, 10, 0.002, 3.0, 8.0, 1.48, 1.3)
        assert cp.aum == 1e6

    def test_experiment_capacity(self):
        ec = ExperimentCapacity("X", 2.0, 50e6, 30e6, 20e6, 65e6, 250, 0.01)
        assert ec.recommended_aum == 30e6

    def test_liquidity_result_defaults(self):
        lr = LiquidityResult()
        assert lr.capacity_curve == []
        assert lr.portfolio_max_aum == 0.0
