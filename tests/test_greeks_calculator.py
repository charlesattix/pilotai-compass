"""Tests for compass/greeks_calculator.py — options Greeks calculator.

Covers:
  - Black-Scholes core: norm_cdf, norm_pdf, d1d2
  - Individual option Greeks (call + put)
  - Spread Greeks computation
  - Portfolio aggregation by experiment
  - Risk limits checking
  - Scenario analysis
  - Theta decay curves
  - from_dataframe constructor
  - Full analyze() pipeline
  - HTML report generation
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from compass.greeks_calculator import (
    DecayPoint,
    GreeksCalculator,
    OptionGreeks,
    PortfolioGreeks,
    Position,
    RiskLimit,
    ScenarioResult,
    SpreadGreeks,
    compute_option_greeks,
    compute_spread_greeks,
    _norm_cdf,
    _norm_pdf,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _put_position(
    strike=420.0, underlying=430.0, iv=0.22, dte=30,
    contracts=2, experiment="EXP-400", spread_strike=None,
    direction="short",
):
    return Position(
        experiment=experiment, option_type="put", direction=direction,
        strike=strike, underlying_price=underlying, iv=iv, dte=dte,
        contracts=contracts, spread_strike=spread_strike,
    )


def _call_position(
    strike=440.0, underlying=430.0, iv=0.22, dte=30,
    contracts=2, experiment="EXP-401", spread_strike=None,
    direction="short",
):
    return Position(
        experiment=experiment, option_type="call", direction=direction,
        strike=strike, underlying_price=underlying, iv=iv, dte=dte,
        contracts=contracts, spread_strike=spread_strike,
    )


def _sample_portfolio():
    """4 positions across 2 experiments: put spread + call spread."""
    return [
        _put_position(strike=420, underlying=430, spread_strike=410, experiment="EXP-400"),
        _put_position(strike=415, underlying=430, spread_strike=405, experiment="EXP-400"),
        _call_position(strike=440, underlying=430, spread_strike=450, experiment="EXP-401"),
        _put_position(strike=425, underlying=430, spread_strike=415, experiment="EXP-503"),
    ]


# ── Black-Scholes core tests ────────────────────────────────────────────


class TestBlackScholesCore:
    def test_norm_cdf_at_zero(self):
        assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-10)

    def test_norm_cdf_far_positive(self):
        assert _norm_cdf(5.0) == pytest.approx(1.0, abs=1e-6)

    def test_norm_cdf_far_negative(self):
        assert _norm_cdf(-5.0) == pytest.approx(0.0, abs=1e-6)

    def test_norm_pdf_at_zero(self):
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert _norm_pdf(0.0) == pytest.approx(expected, abs=1e-10)

    def test_norm_pdf_symmetric(self):
        assert _norm_pdf(1.0) == pytest.approx(_norm_pdf(-1.0), abs=1e-10)


# ── Individual option Greeks tests ───────────────────────────────────────


class TestOptionGreeks:
    def test_call_delta_positive(self):
        g = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "call")
        assert g.delta > 0

    def test_put_delta_negative(self):
        g = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "put")
        assert g.delta < 0

    def test_atm_call_delta_near_half(self):
        g = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "call")
        assert 0.4 < g.delta < 0.7

    def test_gamma_positive(self):
        g = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "call")
        assert g.gamma > 0

    def test_call_put_gamma_equal(self):
        c = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "call")
        p = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "put")
        assert c.gamma == pytest.approx(p.gamma, abs=1e-10)

    def test_vega_positive(self):
        g = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "call")
        assert g.vega > 0

    def test_call_put_vega_equal(self):
        c = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "call")
        p = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "put")
        assert c.vega == pytest.approx(p.vega, abs=1e-10)

    def test_theta_negative_for_long_option(self):
        """Long options lose value over time."""
        g = compute_option_greeks(100, 100, 0.25, 0.20, 0.05, "call")
        assert g.theta < 0

    def test_put_call_parity_price(self):
        """C - P = S - K*exp(-rT)."""
        S, K, T, sigma, r = 100, 100, 0.25, 0.20, 0.05
        c = compute_option_greeks(S, K, T, sigma, r, "call")
        p = compute_option_greeks(S, K, T, sigma, r, "put")
        parity = S - K * math.exp(-r * T)
        assert c.price - p.price == pytest.approx(parity, abs=0.01)

    def test_deep_itm_call_delta_near_one(self):
        g = compute_option_greeks(200, 100, 0.25, 0.20, 0.05, "call")
        assert g.delta > 0.95

    def test_deep_otm_put_delta_near_zero(self):
        g = compute_option_greeks(200, 100, 0.25, 0.20, 0.05, "put")
        assert abs(g.delta) < 0.05

    def test_expired_option_returns_intrinsic(self):
        g = compute_option_greeks(105, 100, 0.0, 0.20, 0.05, "call")
        assert g.price == pytest.approx(5.0)
        assert g.gamma == 0.0

    def test_higher_vol_higher_price(self):
        low = compute_option_greeks(100, 100, 0.25, 0.10, 0.05, "call")
        high = compute_option_greeks(100, 100, 0.25, 0.40, 0.05, "call")
        assert high.price > low.price


# ── Spread Greeks tests ──────────────────────────────────────────────────


class TestSpreadGreeks:
    def test_short_put_spread_positive_theta(self):
        """Credit spread (short higher, long lower) should earn theta."""
        pos = _put_position(strike=420, spread_strike=410, direction="short")
        sg = compute_spread_greeks(pos)
        assert sg.theta > 0

    def test_spread_delta_less_than_naked(self):
        naked = _put_position(strike=420, direction="short")
        spread = _put_position(strike=420, spread_strike=410, direction="short")
        naked_g = compute_spread_greeks(naked)
        spread_g = compute_spread_greeks(spread)
        assert abs(spread_g.delta) < abs(naked_g.delta)

    def test_spread_gamma_less_than_naked(self):
        naked = _put_position(strike=420, direction="short")
        spread = _put_position(strike=420, spread_strike=410, direction="short")
        naked_g = compute_spread_greeks(naked)
        spread_g = compute_spread_greeks(spread)
        assert abs(spread_g.gamma) < abs(naked_g.gamma)

    def test_contracts_scale_greeks(self):
        one = _put_position(contracts=1)
        two = _put_position(contracts=2)
        g1 = compute_spread_greeks(one)
        g2 = compute_spread_greeks(two)
        assert g2.delta == pytest.approx(2 * g1.delta, abs=0.01)

    def test_long_direction_flips_sign(self):
        short = _put_position(direction="short")
        long = _put_position(direction="long")
        gs = compute_spread_greeks(short)
        gl = compute_spread_greeks(long)
        assert gs.delta == pytest.approx(-gl.delta, abs=0.01)


# ── Portfolio aggregation tests ──────────────────────────────────────────


class TestPortfolioAggregation:
    def test_aggregates_all_positions(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        assert calc.portfolio.n_positions == 4

    def test_by_experiment_keys(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        assert "EXP-400" in calc.portfolio.by_experiment
        assert "EXP-401" in calc.portfolio.by_experiment
        assert "EXP-503" in calc.portfolio.by_experiment

    def test_total_delta_is_sum(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        sum_delta = sum(sg.delta for _, sg in calc.position_greeks)
        assert calc.portfolio.total_delta == pytest.approx(sum_delta, abs=0.01)

    def test_experiment_delta_sums_correctly(self):
        positions = _sample_portfolio()
        calc = GreeksCalculator(positions)
        calc.analyze()
        exp400_delta = sum(
            sg.delta for pos, sg in calc.position_greeks if pos.experiment == "EXP-400"
        )
        assert calc.portfolio.by_experiment["EXP-400"].delta == pytest.approx(exp400_delta, abs=0.01)


# ── Risk limits tests ───────────────────────────────────────────────────


class TestRiskLimits:
    def test_default_limits_present(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        metrics = {rl.metric for rl in calc.risk_limits}
        assert "delta" in metrics
        assert "gamma" in metrics
        assert "vega" in metrics

    def test_utilization_computed(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        for rl in calc.risk_limits:
            assert rl.utilization >= 0

    def test_breach_with_tight_limits(self):
        calc = GreeksCalculator(
            _sample_portfolio(),
            risk_limits={"delta": 0.01, "gamma": 0.0001, "vega": 0.01},
        )
        calc.analyze()
        breached = [rl for rl in calc.risk_limits if rl.breached]
        assert len(breached) > 0

    def test_no_breach_with_loose_limits(self):
        calc = GreeksCalculator(
            _sample_portfolio(),
            risk_limits={"delta": 1e6, "gamma": 1e6, "vega": 1e6},
        )
        calc.analyze()
        breached = [rl for rl in calc.risk_limits if rl.breached]
        assert len(breached) == 0

    def test_custom_limit_metric(self):
        calc = GreeksCalculator(
            _sample_portfolio(),
            risk_limits={"delta": 50.0},
        )
        calc.analyze()
        assert len(calc.risk_limits) == 1
        assert calc.risk_limits[0].metric == "delta"


# ── Scenario analysis tests ─────────────────────────────────────────────


class TestScenarioAnalysis:
    def test_scenarios_populated(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        assert len(calc.scenarios) > 0

    def test_scenario_at_zero_shift_pnl_zero(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        zero = [s for s in calc.scenarios
                if s.underlying_shift == 0 and s.vol_shift == 0 and s.dte_shift == 0]
        assert len(zero) == 1
        assert zero[0].pnl == pytest.approx(0.0, abs=0.01)

    def test_scenario_count(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        # Default: 7 underlying × 5 vol × 4 dte = 140
        assert len(calc.scenarios) == 7 * 5 * 4


# ── Theta decay curves tests ────────────────────────────────────────────


class TestDecayCurves:
    def test_curves_per_experiment(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        assert "EXP-400" in calc.decay_curves
        assert "EXP-401" in calc.decay_curves

    def test_decay_points_sorted_by_dte(self):
        calc = GreeksCalculator(_sample_portfolio())
        calc.analyze()
        for points in calc.decay_curves.values():
            dtes = [p.dte for p in points]
            assert dtes == sorted(dtes, reverse=True)

    def test_theta_accelerates_near_expiry(self):
        """Theta should be larger (more negative for long) near expiry."""
        calc = GreeksCalculator([_put_position(dte=30, direction="long")])
        calc.analyze()
        points = list(calc.decay_curves.values())[0]
        far = [p for p in points if p.dte > 20]
        near = [p for p in points if p.dte < 10]
        if far and near:
            far_theta = abs(np.mean([p.theta for p in far]))
            near_theta = abs(np.mean([p.theta for p in near]))
            assert near_theta > far_theta


# ── from_dataframe tests ─────────────────────────────────────────────────


class TestFromDataframe:
    def test_from_dataframe_constructs(self):
        df = pd.DataFrame([{
            "experiment": "EXP-400", "option_type": "put", "direction": "short",
            "strike": 420, "underlying_price": 430, "iv": 0.22, "dte": 30,
            "contracts": 2, "rate": 0.045, "spread_strike": 410,
        }])
        calc = GreeksCalculator.from_dataframe(df)
        assert len(calc.positions) == 1

    def test_from_dataframe_nan_spread_strike(self):
        df = pd.DataFrame([{
            "experiment": "EXP-400", "option_type": "put", "direction": "short",
            "strike": 420, "underlying_price": 430, "iv": 0.22, "dte": 30,
            "contracts": 2, "rate": 0.045, "spread_strike": np.nan,
        }])
        calc = GreeksCalculator.from_dataframe(df)
        assert calc.positions[0].spread_strike is None


# ── Full pipeline tests ─────────────────────────────────────────────────


class TestAnalyzePipeline:
    def test_analyze_returns_all_keys(self):
        calc = GreeksCalculator(_sample_portfolio())
        result = calc.analyze()
        expected = {"position_greeks", "portfolio", "risk_limits", "scenarios", "decay_curves"}
        assert set(result.keys()) == expected

    def test_empty_portfolio(self):
        calc = GreeksCalculator([])
        calc.analyze()
        assert calc.portfolio.n_positions == 0
        assert calc.portfolio.total_delta == 0.0


# ── Report generation tests ──────────────────────────────────────────────


class TestReport:
    def test_generates_html(self, tmp_path):
        calc = GreeksCalculator(_sample_portfolio())
        path = calc.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Greeks Dashboard" in content

    def test_report_contains_all_sections(self, tmp_path):
        calc = GreeksCalculator(_sample_portfolio())
        path = calc.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "Risk Limit" in content
        assert "Position Greeks" in content
        assert "Decay Curves" in content
        assert "Scenario" in content
        assert "Gamma Exposure" in content

    def test_report_embeds_charts(self, tmp_path):
        calc = GreeksCalculator(_sample_portfolio())
        path = calc.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "data:image/png;base64," in content

    def test_report_auto_runs_analyze(self, tmp_path):
        calc = GreeksCalculator(_sample_portfolio())
        assert calc.portfolio is None
        calc.generate_report(str(tmp_path / "report.html"))
        assert calc.portfolio is not None

    def test_report_at_default_path(self):
        calc = GreeksCalculator(_sample_portfolio())
        path = calc.generate_report()
        assert "greeks_dashboard.html" in path
        assert open(path).read().startswith("<!DOCTYPE html>")
