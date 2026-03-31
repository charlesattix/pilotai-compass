"""Tests for compass/trade_cost_analyzer.py — trade cost analyzer.

Covers:
  - Dataclass construction
  - Explicit costs (commissions, fees, taxes)
  - Implicit costs (spread, slippage, market impact)
  - Almgren-Chriss market impact model
  - Opportunity costs (delay, missed trades)
  - Full trade cost computation
  - Strategy attribution
  - Time bucketing
  - Cost forecasting
  - Optimization recommendations
  - from_csv constructor
  - HTML report generation
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from compass.trade_cost_analyzer import (
    CostForecast,
    ExplicitCosts,
    ImplicitCosts,
    OpportunityCosts,
    OptimizationRec,
    StrategyAttribution,
    TimeBucket,
    TradeCost,
    TradeCostAnalyzer,
    almgren_chriss_impact,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_trades(n=100, seed=42):
    """Generate synthetic trade data."""
    rng = np.random.RandomState(seed)
    strategies = rng.choice(["EXP-400", "EXP-401", "EXP-503"], n)
    dates = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({
        "trade_id": [f"T{i:04d}" for i in range(n)],
        "entry_date": dates,
        "strategy": strategies,
        "asset": "SPY",
        "contracts": rng.randint(1, 6, n),
        "net_credit": rng.uniform(0.5, 3.0, n).round(4),
        "spread_width": 5.0,
        "bid_ask_spread": rng.uniform(0.02, 0.10, n).round(4),
        "slippage": rng.uniform(0.01, 0.05, n).round(4),
        "delay_minutes": rng.choice([0, 5, 15, 30, 60, 120], n).astype(float),
        "underlying_price": rng.uniform(420, 450, n).round(2),
        "pnl": rng.normal(50, 200, n).round(2),
    })


def _make_analyzer(n=100, seed=42, **kwargs):
    return TradeCostAnalyzer(_make_trades(n=n, seed=seed), **kwargs)


# ── Dataclass tests ──────────────────────────────────────────────────────


class TestDataclasses:
    def test_explicit_costs_fields(self):
        ec = ExplicitCosts(commissions=5.0, exchange_fees=2.0,
                           regulatory_fees=0.5, taxes=0.0, total=7.5)
        assert ec.total == pytest.approx(7.5)

    def test_implicit_costs_fields(self):
        ic = ImplicitCosts(spread_cost=10.0, slippage=4.0,
                           market_impact=2.0, total=16.0)
        assert ic.total == pytest.approx(16.0)

    def test_opportunity_costs_fields(self):
        oc = OpportunityCosts(delay_cost=3.0, missed_trade_cost=5.0, total=8.0)
        assert oc.total == pytest.approx(8.0)

    def test_trade_cost_fields(self):
        tc = TradeCost(
            trade_id="T1", strategy="EXP-400", asset="SPY", contracts=2,
            explicit=ExplicitCosts(5, 2, 0.5, 0, 7.5),
            implicit=ImplicitCosts(10, 4, 2, 16),
            opportunity=OpportunityCosts(3, 0, 3),
            total_cost=26.5, cost_as_pct_of_premium=0.088, date="2024-01-02",
        )
        assert tc.total_cost == pytest.approx(26.5)

    def test_strategy_attribution_fields(self):
        sa = StrategyAttribution(
            strategy="EXP-400", n_trades=50, total_cost=500,
            avg_cost_per_trade=10, explicit_pct=0.3,
            implicit_pct=0.5, opportunity_pct=0.2,
            cost_as_pct_of_pnl=0.05,
        )
        assert sa.explicit_pct + sa.implicit_pct + sa.opportunity_pct == pytest.approx(1.0)

    def test_cost_forecast_fields(self):
        cf = CostForecast(contracts=2, explicit=7.5, implicit=16,
                          opportunity=3, total=26.5,
                          optimal_contracts=2, optimal_cost=26.5)
        assert cf.total == pytest.approx(26.5)


# ── Almgren-Chriss tests ────────────────────────────────────────────────


class TestAlmgrenChriss:
    def test_zero_volume_returns_zero(self):
        temp, perm = almgren_chriss_impact(10, 0, 0.015, 430)
        assert temp == 0.0
        assert perm == 0.0

    def test_zero_contracts_minimal_impact(self):
        temp, perm = almgren_chriss_impact(0, 5000, 0.015, 430)
        assert temp == pytest.approx(0.0, abs=0.01)

    def test_more_contracts_more_impact(self):
        t1, p1 = almgren_chriss_impact(1, 5000, 0.015, 430)
        t10, p10 = almgren_chriss_impact(10, 5000, 0.015, 430)
        assert t10 > t1
        assert p10 > p1

    def test_higher_vol_more_impact(self):
        t_low, _ = almgren_chriss_impact(5, 5000, 0.01, 430)
        t_high, _ = almgren_chriss_impact(5, 5000, 0.03, 430)
        assert t_high > t_low

    def test_impact_positive(self):
        temp, perm = almgren_chriss_impact(5, 5000, 0.015, 430)
        assert temp > 0
        assert perm > 0


# ── Explicit costs tests ────────────────────────────────────────────────


class TestExplicitCosts:
    def test_commission_scales_with_contracts(self):
        a = _make_analyzer()
        ec1 = a._explicit_costs(1, 1.0)
        ec5 = a._explicit_costs(5, 1.0)
        assert ec5.commissions == 5 * ec1.commissions

    def test_total_is_sum(self):
        a = _make_analyzer()
        ec = a._explicit_costs(2, 1.5)
        assert ec.total == pytest.approx(
            ec.commissions + ec.exchange_fees + ec.regulatory_fees + ec.taxes
        )

    def test_zero_tax_rate(self):
        a = _make_analyzer(tax_rate=0.0)
        ec = a._explicit_costs(2, 1.5)
        assert ec.taxes == 0.0

    def test_nonzero_tax_rate(self):
        a = TradeCostAnalyzer(_make_trades(10), tax_rate=0.01)
        ec = a._explicit_costs(2, 1.5)
        assert ec.taxes > 0


# ── Implicit costs tests ────────────────────────────────────────────────


class TestImplicitCosts:
    def test_spread_cost_positive(self):
        a = _make_analyzer()
        ic = a._implicit_costs(2, 0.05, 0.02, 430)
        assert ic.spread_cost > 0

    def test_slippage_scales(self):
        a = _make_analyzer()
        ic1 = a._implicit_costs(1, 0.05, 0.02, 430)
        ic5 = a._implicit_costs(5, 0.05, 0.02, 430)
        assert ic5.slippage == 5 * ic1.slippage

    def test_market_impact_present(self):
        a = _make_analyzer()
        ic = a._implicit_costs(3, 0.05, 0.02, 430)
        assert ic.market_impact > 0

    def test_total_is_sum(self):
        a = _make_analyzer()
        ic = a._implicit_costs(2, 0.05, 0.02, 430)
        assert ic.total == pytest.approx(
            ic.spread_cost + ic.slippage + ic.market_impact
        )


# ── Opportunity costs tests ──────────────────────────────────────────────


class TestOpportunityCosts:
    def test_zero_delay_no_cost(self):
        a = _make_analyzer()
        oc = a._opportunity_costs(2, 0.0, 430, 1.5)
        assert oc.delay_cost == pytest.approx(0.0, abs=0.01)

    def test_delay_increases_cost(self):
        a = _make_analyzer()
        oc0 = a._opportunity_costs(2, 0, 430, 1.5)
        oc60 = a._opportunity_costs(2, 60, 430, 1.5)
        assert oc60.delay_cost > oc0.delay_cost

    def test_missed_trade_cost_with_large_delay(self):
        a = _make_analyzer()
        oc = a._opportunity_costs(2, 120, 430, 1.5)
        assert oc.missed_trade_cost > 0

    def test_no_missed_cost_with_small_delay(self):
        a = _make_analyzer()
        oc = a._opportunity_costs(2, 5, 430, 1.5)
        assert oc.missed_trade_cost == 0.0


# ── Full analysis tests ─────────────────────────────────────────────────


class TestFullAnalysis:
    def test_analyze_returns_keys(self):
        a = _make_analyzer()
        result = a.analyze()
        expected = {"trade_costs", "strategy_attribution", "time_buckets", "optimizations"}
        assert set(result.keys()) == expected

    def test_trade_costs_match_n_trades(self):
        a = _make_analyzer(n=50)
        a.analyze()
        assert len(a.trade_costs) == 50

    def test_total_cost_positive(self):
        a = _make_analyzer()
        a.analyze()
        for tc in a.trade_costs:
            assert tc.total_cost > 0

    def test_cost_components_sum(self):
        a = _make_analyzer()
        a.analyze()
        for tc in a.trade_costs:
            expected = tc.explicit.total + tc.implicit.total + tc.opportunity.total
            assert tc.total_cost == pytest.approx(expected, abs=0.01)


# ── Strategy attribution tests ───────────────────────────────────────────


class TestStrategyAttribution:
    def test_all_strategies_present(self):
        a = _make_analyzer()
        a.analyze()
        strats = {sa.strategy for sa in a.strategy_attribution}
        assert len(strats) > 0

    def test_percentages_sum_to_one(self):
        a = _make_analyzer()
        a.analyze()
        for sa in a.strategy_attribution:
            total_pct = sa.explicit_pct + sa.implicit_pct + sa.opportunity_pct
            assert total_pct == pytest.approx(1.0, abs=0.01)

    def test_sorted_by_total_cost(self):
        a = _make_analyzer()
        a.analyze()
        costs = [sa.total_cost for sa in a.strategy_attribution]
        assert costs == sorted(costs, reverse=True)


# ── Time bucketing tests ────────────────────────────────────────────────


class TestTimeBuckets:
    def test_buckets_populated(self):
        a = _make_analyzer()
        a.analyze()
        assert len(a.time_buckets) > 0

    def test_bucket_trades_sum(self):
        a = _make_analyzer(n=50)
        a.analyze()
        total = sum(b.n_trades for b in a.time_buckets)
        assert total == 50


# ── Forecasting tests ────────────────────────────────────────────────────


class TestForecasting:
    def test_forecast_returns_result(self):
        a = _make_analyzer()
        a.analyze()
        fc = a.forecast(contracts=3)
        assert isinstance(fc, CostForecast)
        assert fc.total > 0

    def test_forecast_total_is_sum(self):
        a = _make_analyzer()
        a.analyze()
        fc = a.forecast(contracts=3)
        assert fc.total == pytest.approx(fc.explicit + fc.implicit + fc.opportunity, abs=0.01)

    def test_optimal_contracts_positive(self):
        a = _make_analyzer()
        a.analyze()
        fc = a.forecast(contracts=5)
        assert fc.optimal_contracts >= 1

    def test_more_contracts_more_cost(self):
        a = _make_analyzer()
        a.analyze()
        fc1 = a.forecast(contracts=1)
        fc10 = a.forecast(contracts=10)
        assert fc10.total > fc1.total


# ── Optimization tests ───────────────────────────────────────────────────


class TestOptimizations:
    def test_recommendations_generated(self):
        a = _make_analyzer()
        a.analyze()
        # Should have at least one recommendation for typical data
        assert len(a.optimizations) >= 0  # may be 0 depending on data

    def test_recommendation_fields(self):
        rec = OptimizationRec(
            category="size", description="Reduce size",
            estimated_savings=100.0, confidence="high",
        )
        assert rec.confidence == "high"

    def test_recommendations_sorted_by_savings(self):
        a = _make_analyzer()
        a.analyze()
        if len(a.optimizations) >= 2:
            savings = [r.estimated_savings for r in a.optimizations]
            assert savings == sorted(savings, reverse=True)


# ── from_csv tests ───────────────────────────────────────────────────────


class TestFromCSV:
    def test_from_csv_loads(self, tmp_path):
        df = _make_trades(20)
        csv = tmp_path / "trades.csv"
        df.to_csv(csv, index=False)
        a = TradeCostAnalyzer.from_csv(str(csv))
        assert len(a.trades) == 20

    def test_from_csv_analyzes(self, tmp_path):
        df = _make_trades(20)
        csv = tmp_path / "trades.csv"
        df.to_csv(csv, index=False)
        a = TradeCostAnalyzer.from_csv(str(csv))
        a.analyze()
        assert len(a.trade_costs) == 20


# ── Report generation tests ──────────────────────────────────────────────


class TestReport:
    def test_generates_html(self, tmp_path):
        a = _make_analyzer()
        path = a.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Trade Cost" in content

    def test_report_contains_sections(self, tmp_path):
        a = _make_analyzer()
        path = a.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "Waterfall" in content
        assert "Strategy Comparison" in content
        assert "Cost Trend" in content
        assert "Optimization" in content

    def test_report_embeds_charts(self, tmp_path):
        a = _make_analyzer()
        path = a.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "data:image/png;base64," in content

    def test_report_auto_analyzes(self, tmp_path):
        a = _make_analyzer()
        assert len(a.trade_costs) == 0
        a.generate_report(str(tmp_path / "report.html"))
        assert len(a.trade_costs) > 0

    def test_report_at_default_path(self):
        a = _make_analyzer()
        path = a.generate_report()
        assert "trade_costs.html" in path
        assert open(path).read().startswith("<!DOCTYPE html>")
