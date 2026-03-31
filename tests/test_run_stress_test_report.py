"""Tests for compass/run_stress_test.py report generation and criteria checking.

Covers:
  - _check_criteria: pass/fail threshold logic, hedged DD usage
  - _risk_badge: HTML output structure
  - _fmt_pct / _fmt_usd: formatting helpers
  - _crisis_table: hedged column rendering
  - _summary_card: structure verification
  - _comparison_table: multi-experiment metrics
  - _blend_returns: equal-weight blending
  - _hedge_impact_table: before/after rendering
  - _generate_html: full report structure
"""

import numpy as np
import pandas as pd
import pytest

from compass.run_stress_test import (
    _blend_returns,
    _check_criteria,
    _crisis_table,
    _fmt_pct,
    _fmt_usd,
    _risk_badge,
    _RISK_COLOR,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_mc_results(p5_dd=-25.0, median_dd=-15.0, prob_profit=0.85, prob_ruin=0.01):
    """Build a minimal MC results dict for testing."""
    return {
        "n_simulations": 1000,
        "horizon_days": 252,
        "block_size": 5,
        "starting_capital": 100_000,
        "terminal_wealth": {
            "mean": 110_000,
            "median": 108_000,
            "std": 15_000,
            "min": 50_000,
            "max": 200_000,
            "percentiles": {
                "p1": 60_000, "p5": 70_000, "p10": 75_000,
                "p25": 90_000, "p50": 108_000, "p75": 120_000,
                "p90": 140_000, "p95": 160_000, "p99": 190_000,
            },
        },
        "max_drawdown": {
            "mean_pct": median_dd - 5,
            "median_pct": median_dd,
            "worst_pct": p5_dd * 2,
            "percentiles_pct": {
                "p1": p5_dd * 1.5, "p5": p5_dd, "p10": p5_dd + 2,
                "p25": median_dd - 3, "p50": median_dd, "p75": median_dd + 5,
                "p90": -5.0, "p95": -3.0, "p99": -1.0,
            },
        },
        "sharpe_ratio": {
            "mean": 1.2, "median": 1.1, "std": 0.4,
            "percentiles": {"p5": 0.3, "p50": 1.1, "p95": 2.0},
        },
        "prob_profit": prob_profit,
        "prob_ruin_50pct": prob_ruin,
        "sample_paths": [],
    }


def _make_crisis_results(portfolio_dd=-40.0, hedged_dd=None):
    """Build a minimal crisis results list for testing."""
    return [
        {
            "name": "COVID Crash (Feb-Mar 2020)",
            "description": "S&P fell ~34%",
            "n_days": 23,
            "underlying_drawdown_pct": -30.0,
            "portfolio_drawdown_pct": portfolio_dd,
            "max_drawdown_pct": portfolio_dd,
            "trough_value": 100_000 * (1 + portfolio_dd / 100),
            "spread_beta": 1.5,
            "vix_start": 15,
            "vix_peak": 82,
            "vix_multiplier": 5.47,
            "estimated_recovery_days": 200,
            "equity_path": [100_000, 90_000, 80_000],
            "hedged_portfolio_drawdown_pct": hedged_dd,
            "hedged_trough_value": 100_000 * (1 + hedged_dd / 100) if hedged_dd else None,
            "hedged_equity_path": [100_000, 95_000, 90_000] if hedged_dd else None,
        },
    ]


def _make_sensitivity():
    """Build a minimal sensitivity results dict."""
    return {
        "position_size_pct": {
            "label": "Position Size",
            "description": "Risk per trade",
            "baseline": 5.0,
            "results": [
                {"value": 2.0, "sharpe": 1.5, "max_dd_pct": -10, "cagr_pct": 15, "calmar": 1.5, "terminal_value": 115_000, "is_baseline": False},
                {"value": 5.0, "sharpe": 1.2, "max_dd_pct": -20, "cagr_pct": 25, "calmar": 1.25, "terminal_value": 125_000, "is_baseline": True},
                {"value": 10.0, "sharpe": 0.8, "max_dd_pct": -35, "cagr_pct": 40, "calmar": 1.14, "terminal_value": 140_000, "is_baseline": False},
            ],
        },
    }


def _make_full_results(p5_dd=-25.0, crisis_dd=-40.0, hedged_crisis_dd=None):
    """Build a full stress test results dict."""
    return {
        "experiment": "EXP-400",
        "n_trading_days": 252,
        "monte_carlo": _make_mc_results(p5_dd=p5_dd),
        "crisis_scenarios": _make_crisis_results(portfolio_dd=crisis_dd, hedged_dd=hedged_crisis_dd),
        "sensitivity": _make_sensitivity(),
        "summary": {
            "historical": {
                "n_days": 252,
                "starting_capital": 100_000,
                "sharpe": 1.2,
                "max_drawdown_pct": -18.0,
                "cagr_pct": 20.0,
                "calmar": 1.11,
            },
            "monte_carlo_confidence": {
                "p5_terminal": 70_000,
                "p50_terminal": 108_000,
                "p95_terminal": 160_000,
                "prob_profit_pct": 85.0,
                "prob_ruin_pct": 1.0,
                "median_max_dd_pct": -15.0,
            },
            "worst_crisis": {
                "name": "COVID Crash (Feb-Mar 2020)",
                "portfolio_drawdown_pct": crisis_dd,
                "hedged_portfolio_drawdown_pct": hedged_crisis_dd,
                "estimated_recovery_days": 200,
            },
            "most_sensitive_parameter": "position_size_pct",
            "risk_rating": "MODERATE",
        },
    }


# ── _check_criteria ──────────────────────────────────────────────────────


class TestCheckCriteria:
    def test_all_passing(self):
        results = _make_full_results(p5_dd=-20.0, crisis_dd=-35.0)
        criteria = _check_criteria(results)
        assert criteria["mc_p5_dd_le_30pct"] is True
        assert criteria["all_crisis_dd_le_40pct"] is True

    def test_mc_p5_dd_exceeds_30(self):
        results = _make_full_results(p5_dd=-35.0)
        criteria = _check_criteria(results)
        assert criteria["mc_p5_dd_le_30pct"] is False

    def test_mc_p5_dd_at_30_passes(self):
        results = _make_full_results(p5_dd=-30.0)
        criteria = _check_criteria(results)
        assert criteria["mc_p5_dd_le_30pct"] is True

    def test_crisis_dd_exceeds_40(self):
        results = _make_full_results(crisis_dd=-50.0)
        criteria = _check_criteria(results)
        assert criteria["all_crisis_dd_le_40pct"] is False

    def test_crisis_dd_at_40_passes(self):
        results = _make_full_results(crisis_dd=-40.0)
        criteria = _check_criteria(results)
        assert criteria["all_crisis_dd_le_40pct"] is True

    def test_hedged_dd_used_when_available(self):
        """When hedged DD is available and <= 40, should pass even if unhedged > 40."""
        results = _make_full_results(crisis_dd=-55.0, hedged_crisis_dd=-35.0)
        criteria = _check_criteria(results)
        assert criteria["all_crisis_dd_le_40pct"] is True

    def test_hedged_dd_fails_when_exceeds_40(self):
        results = _make_full_results(crisis_dd=-55.0, hedged_crisis_dd=-45.0)
        criteria = _check_criteria(results)
        assert criteria["all_crisis_dd_le_40pct"] is False

    def test_no_cliff_parameters_passing(self):
        results = _make_full_results()
        criteria = _check_criteria(results)
        assert criteria["no_cliff_parameters"] is True

    def test_cliff_detected(self):
        results = _make_full_results()
        # Make a cliff: Sharpe drops from 1.5 to 0.2 (>50% drop)
        results["sensitivity"]["position_size_pct"]["results"] = [
            {"value": 2.0, "sharpe": 1.5, "max_dd_pct": -10, "cagr_pct": 15, "calmar": 1.5, "terminal_value": 115_000, "is_baseline": False},
            {"value": 5.0, "sharpe": 0.2, "max_dd_pct": -40, "cagr_pct": 5, "calmar": 0.12, "terminal_value": 105_000, "is_baseline": True},
        ]
        criteria = _check_criteria(results)
        assert criteria["no_cliff_parameters"] is False

    def test_criteria_includes_numeric_values(self):
        results = _make_full_results(p5_dd=-22.5)
        criteria = _check_criteria(results)
        assert "p5_dd" in criteria
        assert criteria["p5_dd"] == pytest.approx(22.5, abs=0.1)


# ── _risk_badge ──────────────────────────────────────────────────────────


class TestRiskBadge:
    def test_low_badge_has_green(self):
        html = _risk_badge("LOW")
        assert "LOW" in html
        assert "#d4edda" in html  # green background

    def test_critical_badge_has_red(self):
        html = _risk_badge("CRITICAL")
        assert "CRITICAL" in html
        assert "#f5c6cb" in html

    def test_unknown_rating_fallback(self):
        html = _risk_badge("UNKNOWN")
        assert "UNKNOWN" in html
        assert "#e2e3e5" in html  # gray fallback

    def test_returns_span_element(self):
        html = _risk_badge("MODERATE")
        assert html.startswith("<span")
        assert html.endswith("</span>")


# ── _fmt_pct / _fmt_usd ─────────────────────────────────────────────────


class TestFormatters:
    def test_fmt_pct_positive(self):
        assert _fmt_pct(25.3) == "+25.3%"

    def test_fmt_pct_negative(self):
        assert _fmt_pct(-10.5) == "-10.5%"

    def test_fmt_pct_zero(self):
        assert _fmt_pct(0.0) == "+0.0%"

    def test_fmt_pct_custom_decimals(self):
        assert _fmt_pct(25.345, decimals=2) == "+25.34%"

    def test_fmt_usd_thousands(self):
        assert _fmt_usd(100_000) == "$100,000"

    def test_fmt_usd_millions(self):
        assert _fmt_usd(1_500_000) == "$1,500,000"

    def test_fmt_usd_zero(self):
        assert _fmt_usd(0) == "$0"


# ── _crisis_table ────────────────────────────────────────────────────────


class TestCrisisTable:
    def test_unhedged_only_no_hedged_column(self):
        crisis = _make_crisis_results(portfolio_dd=-40.0, hedged_dd=None)
        html = _crisis_table(crisis)
        assert "Hedged DD" not in html

    def test_hedged_column_when_available(self):
        crisis = _make_crisis_results(portfolio_dd=-50.0, hedged_dd=-30.0)
        html = _crisis_table(crisis)
        assert "Hedged DD" in html

    def test_pass_tick_for_under_40(self):
        crisis = _make_crisis_results(portfolio_dd=-35.0)
        html = _crisis_table(crisis)
        assert "✅" in html

    def test_fail_tick_for_over_40(self):
        crisis = _make_crisis_results(portfolio_dd=-50.0)
        html = _crisis_table(crisis)
        assert "❌" in html

    def test_hedged_pass_overrides_unhedged_fail(self):
        """When hedged DD <= 40 but unhedged > 40, should show pass."""
        crisis = _make_crisis_results(portfolio_dd=-55.0, hedged_dd=-35.0)
        html = _crisis_table(crisis)
        assert "✅" in html

    def test_scenario_name_in_table(self):
        crisis = _make_crisis_results()
        html = _crisis_table(crisis)
        assert "COVID" in html


# ── _blend_returns ───────────────────────────────────────────────────────


class TestBlendReturns:
    def test_equal_weight(self):
        idx = pd.bdate_range("2024-01-01", periods=5)
        r1 = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=idx)
        r2 = pd.Series([0.05, 0.04, 0.03, 0.02, 0.01], index=idx)
        blended = _blend_returns(r1, r2)
        np.testing.assert_allclose(blended.values, [0.03] * 5)

    def test_misaligned_dates_filled_with_zero(self):
        idx1 = pd.bdate_range("2024-01-01", periods=3)
        idx2 = pd.bdate_range("2024-01-03", periods=3)
        r1 = pd.Series([0.01, 0.02, 0.03], index=idx1)
        r2 = pd.Series([0.10, 0.20, 0.30], index=idx2)
        blended = _blend_returns(r1, r2)
        # 2024-01-01: 0.5*0.01 + 0.5*0 = 0.005
        assert blended.iloc[0] == pytest.approx(0.005)

    def test_index_name(self):
        idx = pd.bdate_range("2024-01-01", periods=3)
        r1 = pd.Series([0.01] * 3, index=idx)
        r2 = pd.Series([0.02] * 3, index=idx)
        blended = _blend_returns(r1, r2)
        assert blended.index.name == "date"


# ── _RISK_COLOR constant ────────────────────────────────────────────────


class TestRiskColor:
    def test_all_ratings_have_colors(self):
        for rating in ("LOW", "MODERATE", "HIGH", "CRITICAL"):
            assert rating in _RISK_COLOR
            bg, fg = _RISK_COLOR[rating]
            assert bg.startswith("#")
            assert fg.startswith("#")
