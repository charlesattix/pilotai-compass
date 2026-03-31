"""Tests for compass.portfolio_stress – advanced portfolio stress testing."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.portfolio_stress import (
    HISTORICAL_SCENARIOS,
    SYNTHETIC_SCENARIOS,
    MarginAnalysis,
    PortfolioStressEngine,
    PortfolioStressResult,
    ReverseStressResult,
    ScenarioOutcome,
    StressPnL,
    StressScenario,
    _shock_path,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _engine() -> PortfolioStressEngine:
    return PortfolioStressEngine()


def _custom() -> StressScenario:
    return _engine().create_scenario("Test", -0.15, vol_shock=20, duration=10)


# ── Shock path ──────────────────────────────────────────────────────────────
class TestShockPath:
    def test_length(self):
        assert len(_shock_path(-0.20, 10)) == 10

    def test_single_day(self):
        assert _shock_path(-0.10, 1)[0] == pytest.approx(-0.10, abs=0.001)

    def test_empty(self):
        assert _shock_path(-0.10, 0) == []

    def test_total(self):
        p = _shock_path(-0.30, 20)
        assert float(np.prod([1 + s for s in p]) - 1) == pytest.approx(-0.30, abs=0.02)


# ── Predefined ──────────────────────────────────────────────────────────────
class TestPredefined:
    def test_historical_count(self):
        assert len(HISTORICAL_SCENARIOS) == 3

    def test_synthetic_count(self):
        assert len(SYNTHETIC_SCENARIOS) == 3

    def test_scenarios_have_shocks(self):
        for s in HISTORICAL_SCENARIOS + SYNTHETIC_SCENARIOS:
            assert len(s.daily_shocks) > 0

    def test_all_negative(self):
        for s in HISTORICAL_SCENARIOS + SYNTHETIC_SCENARIOS:
            total = float(np.prod([1 + x for x in s.daily_shocks]) - 1)
            assert total < 0


# ── Constructor ─────────────────────────────────────────────────────────────
class TestInit:
    def test_defaults(self):
        e = PortfolioStressEngine()
        assert e.starting_capital == 100_000
        assert e.maint_margin == 0.25

    def test_custom(self):
        e = PortfolioStressEngine(starting_capital=500_000, maintenance_margin_pct=0.30)
        assert e.starting_capital == 500_000


# ── Run ─────────────────────────────────────────────────────────────────────
class TestRun:
    def test_returns_result(self):
        r = _engine().run()
        assert isinstance(r, PortfolioStressResult)

    def test_includes_all(self):
        r = _engine().run()
        assert r.n_scenarios == 6  # 3 historical + 3 synthetic

    def test_historical_only(self):
        r = _engine().run(include_synthetic=False)
        assert r.n_scenarios == 3

    def test_synthetic_only(self):
        r = _engine().run(include_historical=False)
        assert r.n_scenarios == 3

    def test_custom_added(self):
        r = _engine().run(custom_scenarios=[_custom()])
        assert r.n_scenarios == 7

    def test_worst_case(self):
        r = _engine().run()
        assert r.worst_case is not None
        assert r.worst_case.scenario.name == "2008 GFC"

    def test_empty(self):
        r = _engine().run(include_historical=False, include_synthetic=False)
        assert r.n_scenarios == 0

    def test_generated_at(self):
        r = _engine().run()
        assert len(r.generated_at) > 0


# ── Outcomes ────────────────────────────────────────────────────────────────
class TestOutcomes:
    def test_drawdowns_negative(self):
        r = _engine().run()
        for o in r.outcomes:
            assert o.adjusted_dd < 0

    def test_adjusted_worse_than_raw(self):
        r = _engine().run()
        for o in r.outcomes:
            assert abs(o.adjusted_dd) >= abs(o.portfolio_dd)

    def test_trough_below_start(self):
        r = _engine().run()
        for o in r.outcomes:
            assert o.trough_value < 100_000

    def test_equity_path(self):
        r = _engine().run()
        for o in r.outcomes:
            assert o.equity_path[0] == pytest.approx(100_000)

    def test_recovery_days_nonneg(self):
        r = _engine().run()
        for o in r.outcomes:
            assert o.recovery_days >= 0


# ── Greeks P&L ──────────────────────────────────────────────────────────────
class TestGreeksPnL:
    def test_computed(self):
        r = _engine().run(portfolio_delta=-30, portfolio_vega=100)
        for o in r.outcomes:
            assert o.pnl is not None

    def test_total_is_sum(self):
        r = _engine().run(portfolio_delta=-20, portfolio_gamma=-3, portfolio_vega=80, portfolio_theta=10)
        for o in r.outcomes:
            p = o.pnl
            assert p.total_pnl == pytest.approx(p.delta_pnl + p.gamma_pnl + p.vega_pnl + p.theta_pnl, abs=0.01)

    def test_short_delta_profits(self):
        r = _engine().run(portfolio_delta=-50)
        gfc = next(o for o in r.outcomes if o.scenario.name == "2008 GFC")
        assert gfc.pnl.delta_pnl > 0

    def test_long_vega_profits_on_vol(self):
        r = _engine().run(portfolio_vega=100)
        vol = next(o for o in r.outcomes if "Vol" in o.scenario.name)
        assert vol.pnl.vega_pnl > 0


# ── Margin analysis ────────────────────────────────────────────────────────
class TestMarginAnalysis:
    def test_margin_present(self):
        r = _engine().run(current_margin=50_000)
        for o in r.outcomes:
            assert o.margin is not None

    def test_no_margin_if_zero(self):
        r = _engine().run(current_margin=0)
        for o in r.outcomes:
            assert o.margin is None

    def test_margin_call_on_severe(self):
        r = PortfolioStressEngine(starting_capital=100_000).run(current_margin=10_000)
        gfc = next(o for o in r.outcomes if o.scenario.name == "2008 GFC")
        assert gfc.margin.margin_call_triggered

    def test_adequate_margin(self):
        r = PortfolioStressEngine(starting_capital=100_000).run(current_margin=500_000)
        # With 500k margin vs 100k capital, should survive most scenarios
        calls = sum(1 for o in r.outcomes if o.margin and o.margin.margin_call_triggered)
        assert calls < len(r.outcomes)

    def test_excess_margin(self):
        r = _engine().run(current_margin=100_000)
        for o in r.outcomes:
            m = o.margin
            assert m.excess_margin == pytest.approx(m.current_margin - m.stress_margin, abs=0.01)


# ── Reverse stress ──────────────────────────────────────────────────────────
class TestReverseStress:
    def test_present(self):
        r = _engine().run(reverse_target_pct=0.30)
        assert r.reverse_stress is not None

    def test_absent_if_not_requested(self):
        r = _engine().run()
        assert r.reverse_stress is None

    def test_target_matches(self):
        r = _engine().run(reverse_target_pct=0.25)
        assert r.reverse_stress.target_loss_pct == 0.25

    def test_equity_shock_negative(self):
        r = _engine().run(reverse_target_pct=0.30)
        assert r.reverse_stress.required_equity_shock < 0


# ── Probability weighting ──────────────────────────────────────────────────
class TestProbability:
    def test_pw_loss_positive(self):
        r = _engine().run()
        assert r.total_pw_loss > 0


# ── Custom scenario ────────────────────────────────────────────────────────
class TestCustomScenario:
    def test_create(self):
        s = _engine().create_scenario("X", -0.20, vol_shock=30, duration=5)
        assert s.name == "X"
        assert len(s.daily_shocks) == 5


# ── HTML ────────────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = _engine()
            r = e.run(portfolio_delta=-20, portfolio_vega=50, current_margin=50_000, reverse_target_pct=0.25)
            path = e.generate_report(r, output_path=Path(tmp) / "ps.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = _engine()
            r = e.run(portfolio_delta=-20, current_margin=50_000, reverse_target_pct=0.25)
            path = e.generate_report(r, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Portfolio Stress" in html
            assert "Comparison" in html
            assert "Waterfall" in html
            assert "Margin" in html
            assert "Reverse" in html

    def test_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = _engine()
            r = e.run()
            path = e.generate_report(r, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_stress_pnl(self):
        p = StressPnL(delta_pnl=-500, total_pnl=-500)
        assert p.delta_pnl == -500

    def test_margin(self):
        m = MarginAnalysis(50000, 30000, False, 20000, 0.6)
        assert not m.margin_call_triggered

    def test_reverse(self):
        r = ReverseStressResult(0.30, -0.20, 40.0, "Test")
        assert r.target_loss_pct == 0.30

    def test_result_defaults(self):
        r = PortfolioStressResult()
        assert r.outcomes == []
        assert r.worst_case is None
