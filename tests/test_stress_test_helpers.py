"""Tests for compass/stress_test.py helper functions and StressTester internals.

Covers:
  - _build_crash_path: edge cases and compound-return correctness
  - _sharpe_ratio, _max_drawdown, _cagr, _calmar_ratio: edge cases
  - _returns_to_equity, _percentile_safe: boundary conditions
  - StressTester constructor: min-value enforcement, short-data warning
  - _empty_mc_result: structure verification
  - run_monte_carlo: empty returns, custom horizon
  - run_crisis_scenarios: empty shocks, hedged vs unhedged
  - _heuristic_param_adjustment: each param type
  - _compute_risk_rating: all rating tiers
  - _build_summary: with and without hedged crisis results
"""

import math

import numpy as np
import pytest

from compass.stress_test import (
    CRISIS_SCENARIOS,
    StressTester,
    _build_crash_path,
    _cagr,
    _calmar_ratio,
    _max_drawdown,
    _percentile_safe,
    _returns_to_equity,
    _sharpe_ratio,
)


# ── _build_crash_path ─────────────────────────────────────────────────────


class TestBuildCrashPath:
    def test_zero_days_returns_empty(self):
        assert _build_crash_path(-0.30, 0) == []

    def test_negative_days_returns_empty(self):
        assert _build_crash_path(-0.30, -5) == []

    def test_single_day_returns_total(self):
        path = _build_crash_path(-0.10, 1)
        assert len(path) == 1
        assert path[0] == pytest.approx(-0.10)

    def test_length_matches_n_days(self):
        path = _build_crash_path(-0.34, 23)
        assert len(path) == 23

    def test_compounding_matches_total_return(self):
        """The product of (1 + r_i) should equal (1 + total_return)."""
        total_return = -0.34
        path = _build_crash_path(total_return, 23)
        compound = 1.0
        for r in path:
            compound *= (1 + r)
        assert compound == pytest.approx(1 + total_return, rel=1e-6)

    def test_early_days_absorb_more_shock(self):
        """Early returns should be more negative (concave pattern)."""
        path = _build_crash_path(-0.50, 20)
        early_avg = np.mean(path[:5])
        late_avg = np.mean(path[-5:])
        assert early_avg < late_avg  # early days more negative

    def test_deterministic_with_same_seed(self):
        p1 = _build_crash_path(-0.25, 50)
        p2 = _build_crash_path(-0.25, 50)
        assert p1 == p2

    def test_small_decline(self):
        path = _build_crash_path(-0.01, 5)
        assert len(path) == 5
        compound = math.prod(1 + r for r in path)
        assert compound == pytest.approx(0.99, rel=1e-6)


# ── _sharpe_ratio ─────────────────────────────────────────────────────────


class TestSharpeRatio:
    def test_empty_returns_zero(self):
        assert _sharpe_ratio(np.array([])) == 0.0

    def test_single_return_zero(self):
        assert _sharpe_ratio(np.array([0.01])) == 0.0

    def test_constant_returns_extreme_sharpe(self):
        """Near-zero std with positive mean yields very large Sharpe."""
        returns = np.array([0.01] * 100)
        # std is near-zero due to float noise, so Sharpe is astronomically high
        assert _sharpe_ratio(returns) > 1000

    def test_positive_returns_positive_sharpe(self):
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, 252)
        assert _sharpe_ratio(returns) > 0

    def test_negative_returns_negative_sharpe(self):
        rng = np.random.RandomState(42)
        returns = rng.normal(-0.002, 0.01, 252)
        assert _sharpe_ratio(returns) < 0

    def test_custom_annual_factor(self):
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, 100)
        s252 = _sharpe_ratio(returns, annual_factor=252)
        s52 = _sharpe_ratio(returns, annual_factor=52)
        # sqrt(252) > sqrt(52), so annualized at 252 should be larger in magnitude
        assert abs(s252) > abs(s52)


# ── _max_drawdown ─────────────────────────────────────────────────────────


class TestMaxDrawdown:
    def test_empty_curve_zero(self):
        assert _max_drawdown(np.array([])) == 0.0

    def test_single_point_zero(self):
        assert _max_drawdown(np.array([100.0])) == 0.0

    def test_monotonically_increasing_zero(self):
        curve = np.array([100, 110, 120, 130, 140])
        assert _max_drawdown(curve) == 0.0

    def test_simple_drawdown(self):
        curve = np.array([100, 110, 90, 95])
        dd = _max_drawdown(curve)
        # Peak 110, trough 90 → DD = (90-110)/110 ≈ -18.18%
        assert dd == pytest.approx(-20 / 110, abs=1e-6)

    def test_always_below_start(self):
        curve = np.array([100, 80, 70, 60])
        dd = _max_drawdown(curve)
        assert dd == pytest.approx(-0.40, abs=1e-6)

    def test_recovery_after_drawdown(self):
        curve = np.array([100, 80, 60, 80, 100, 120])
        dd = _max_drawdown(curve)
        # Peak 100, trough 60 → -40%
        assert dd == pytest.approx(-0.40, abs=1e-6)


# ── _cagr ─────────────────────────────────────────────────────────────────


class TestCAGR:
    def test_empty_curve_zero(self):
        assert _cagr(np.array([])) == 0.0

    def test_single_point_zero(self):
        assert _cagr(np.array([100.0])) == 0.0

    def test_zero_starting_value_zero(self):
        assert _cagr(np.array([0, 100])) == 0.0

    def test_doubling_in_one_year(self):
        curve = np.concatenate([[100], np.linspace(100, 200, 252)])
        c = _cagr(curve, trading_days=252)
        # ~ 100% annual return
        assert c == pytest.approx(1.0, abs=0.05)

    def test_loss_gives_negative_cagr(self):
        curve = np.concatenate([[100], np.linspace(100, 50, 252)])
        c = _cagr(curve, trading_days=252)
        assert c < 0

    def test_total_loss_zero(self):
        """If terminal value is 0, CAGR should be 0 (log(0) guard)."""
        curve = np.array([100, 50, 0])
        assert _cagr(curve) == 0.0


# ── _calmar_ratio ─────────────────────────────────────────────────────────


class TestCalmarRatio:
    def test_no_drawdown_returns_zero(self):
        curve = np.array([100, 110, 120, 130])
        assert _calmar_ratio(curve) == 0.0

    def test_positive_cagr_with_drawdown(self):
        # Create curve that goes up, dips, then recovers higher
        curve = np.array([100, 120, 100, 110, 130, 150])
        cr = _calmar_ratio(curve)
        # CAGR > 0 and DD < 0, so calmar should be positive
        assert cr > 0


# ── _returns_to_equity ────────────────────────────────────────────────────


class TestReturnsToEquity:
    def test_empty_returns(self):
        eq = _returns_to_equity(np.array([]), 100000)
        assert len(eq) == 1
        assert eq[0] == 100000

    def test_single_return(self):
        eq = _returns_to_equity(np.array([0.05]), 100000)
        assert len(eq) == 2
        assert eq[0] == 100000
        assert eq[1] == pytest.approx(105000)

    def test_zero_returns_flat(self):
        eq = _returns_to_equity(np.array([0, 0, 0]), 50000)
        np.testing.assert_allclose(eq, [50000, 50000, 50000, 50000])


# ── _percentile_safe ─────────────────────────────────────────────────────


class TestPercentileSafe:
    def test_empty_array_returns_zero(self):
        assert _percentile_safe(np.array([]), 50) == 0.0

    def test_single_value(self):
        assert _percentile_safe(np.array([42.0]), 50) == pytest.approx(42.0)

    def test_median_of_range(self):
        arr = np.arange(101, dtype=float)
        assert _percentile_safe(arr, 50) == pytest.approx(50.0)


# ── StressTester constructor ──────────────────────────────────────────────


class TestStressTesterInit:
    def test_min_simulations_enforced(self):
        """n_simulations < 100 should be clamped to 100."""
        st = StressTester([0.01] * 20, n_simulations=10)
        assert st.n_simulations == 100

    def test_min_block_size_enforced(self):
        """block_size < 1 should be clamped to 1."""
        st = StressTester([0.01] * 20, block_size=0)
        assert st.block_size == 1

    def test_short_data_no_crash(self):
        """5 returns should work without raising (just warns)."""
        st = StressTester([0.01] * 5)
        assert len(st.returns) == 5

    def test_returns_are_float64(self):
        st = StressTester([1, 2, 3])
        assert st.returns.dtype == np.float64


# ── _empty_mc_result ─────────────────────────────────────────────────────


class TestEmptyMCResult:
    def test_structure(self):
        st = StressTester([], starting_capital=50000)
        result = st._empty_mc_result()
        assert result["n_simulations"] == 0
        assert result["terminal_wealth"]["mean"] == 50000
        assert result["prob_profit"] == 0
        assert result["sample_paths"] == []


# ── run_monte_carlo ──────────────────────────────────────────────────────


class TestRunMonteCarlo:
    def test_empty_returns_gives_empty_result(self):
        st = StressTester([])
        mc = st.run_monte_carlo()
        assert mc["n_simulations"] == 0

    def test_custom_horizon(self):
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0005, 0.01, 252).tolist()
        st = StressTester(returns, n_simulations=100, seed=42)
        mc = st.run_monte_carlo(horizon_days=50)
        assert mc["horizon_days"] == 50

    def test_result_keys_present(self):
        returns = [0.001] * 50
        st = StressTester(returns, n_simulations=100, seed=42)
        mc = st.run_monte_carlo()
        assert "terminal_wealth" in mc
        assert "max_drawdown" in mc
        assert "sharpe_ratio" in mc
        assert "prob_profit" in mc
        assert "prob_ruin_50pct" in mc
        assert "sample_paths" in mc

    def test_sample_paths_capped_at_200(self):
        returns = [0.001] * 50
        st = StressTester(returns, n_simulations=500, seed=42)
        mc = st.run_monte_carlo()
        assert len(mc["sample_paths"]) <= 200


# ── run_crisis_scenarios ─────────────────────────────────────────────────


class TestRunCrisisScenarios:
    def test_default_scenarios_all_run(self):
        returns = [0.001] * 100
        st = StressTester(returns)
        results = st.run_crisis_scenarios()
        assert len(results) == len(CRISIS_SCENARIOS)

    def test_empty_shocks_scenario_skipped(self):
        returns = [0.001] * 50
        st = StressTester(returns)
        empty_scenario = {
            "name": "Empty",
            "description": "No shocks",
            "daily_shocks": [],
            "vix_start": 15,
            "vix_peak": 40,
        }
        results = st.run_crisis_scenarios(scenarios=[empty_scenario])
        assert len(results) == 0

    def test_unhedged_results_have_no_hedged_fields(self):
        returns = [0.001] * 50
        st = StressTester(returns)
        results = st.run_crisis_scenarios(crisis_hedge_config=None)
        for r in results:
            assert r["hedged_portfolio_drawdown_pct"] is None
            assert r["hedged_trough_value"] is None
            assert r["hedged_equity_path"] is None

    def test_hedged_results_populated(self):
        from compass.crisis_hedge import CrisisHedgeConfig
        returns = [0.001] * 50
        st = StressTester(returns)
        cfg = CrisisHedgeConfig()
        results = st.run_crisis_scenarios(crisis_hedge_config=cfg)
        for r in results:
            assert r["hedged_portfolio_drawdown_pct"] is not None
            assert r["hedged_trough_value"] is not None
            assert r["hedged_equity_path"] is not None

    def test_hedged_dd_less_severe_than_unhedged(self):
        from compass.crisis_hedge import CrisisHedgeConfig
        returns = [0.001] * 50
        st = StressTester(returns)
        cfg = CrisisHedgeConfig()
        results = st.run_crisis_scenarios(crisis_hedge_config=cfg)
        for r in results:
            # hedged DD should be same or less severe (closer to 0)
            assert r["hedged_portfolio_drawdown_pct"] >= r["portfolio_drawdown_pct"]

    def test_result_fields_present(self):
        returns = [0.001] * 50
        st = StressTester(returns)
        results = st.run_crisis_scenarios()
        for r in results:
            assert "name" in r
            assert "portfolio_drawdown_pct" in r
            assert "trough_value" in r
            assert "vix_start" in r
            assert "vix_peak" in r
            assert "equity_path" in r

    def test_recovery_days_estimated(self):
        """With positive mean returns, recovery days should be an int."""
        returns = [0.002] * 100  # positive mean
        st = StressTester(returns)
        results = st.run_crisis_scenarios()
        for r in results:
            assert r["estimated_recovery_days"] is not None
            assert isinstance(r["estimated_recovery_days"], int)
            assert r["estimated_recovery_days"] > 0

    def test_zero_mean_returns_no_recovery(self):
        """With zero mean returns, recovery is impossible."""
        returns = [0.0] * 50
        st = StressTester(returns)
        results = st.run_crisis_scenarios()
        for r in results:
            assert r["estimated_recovery_days"] is None


# ── _heuristic_param_adjustment ──────────────────────────────────────────


class TestHeuristicParamAdjustment:
    def _make_tester(self):
        rng = np.random.RandomState(42)
        returns = rng.normal(0.001, 0.01, 200).tolist()
        return StressTester(returns, seed=42)

    def test_position_size_scaling(self):
        st = self._make_tester()
        doubled = st._heuristic_param_adjustment("position_size_pct", 10.0, 5.0)
        np.testing.assert_allclose(doubled, st.returns * 2.0)

    def test_stop_loss_tighter_clips_losses(self):
        st = self._make_tester()
        tighter = st._heuristic_param_adjustment("stop_loss_multiplier", 1.5, 3.5)
        # Tighter stops clip losses: min of tighter >= min of original
        assert tighter.min() >= st.returns.min()

    def test_stop_loss_wider_amplifies_losses(self):
        st = self._make_tester()
        wider = st._heuristic_param_adjustment("stop_loss_multiplier", 5.0, 3.5)
        neg_original = st.returns[st.returns < 0].sum()
        neg_wider = wider[wider < 0].sum()
        assert neg_wider <= neg_original  # more negative

    def test_iv_rank_higher_filters_trades(self):
        st = self._make_tester()
        filtered = st._heuristic_param_adjustment("iv_rank_threshold", 50, 12)
        # Some returns should be zeroed out
        assert (filtered == 0).sum() > 0

    def test_profit_target_caps_wins(self):
        st = self._make_tester()
        lower_target = st._heuristic_param_adjustment("profit_target_pct", 25, 50)
        orig_pos_sum = st.returns[st.returns > 0].sum()
        new_pos_sum = lower_target[lower_target > 0].sum()
        assert new_pos_sum <= orig_pos_sum

    def test_spread_width_scales(self):
        st = self._make_tester()
        wider = st._heuristic_param_adjustment("spread_width", 10.0, 5.0)
        # spread width ratio^0.7 scaling
        ratio = 10.0 / 5.0
        expected = st.returns * (ratio ** 0.7)
        np.testing.assert_allclose(wider, expected)

    def test_unknown_param_returns_copy(self):
        st = self._make_tester()
        result = st._heuristic_param_adjustment("unknown_param", 1.0, 1.0)
        np.testing.assert_array_equal(result, st.returns)

    def test_zero_baseline_returns_copy(self):
        st = self._make_tester()
        result = st._heuristic_param_adjustment("position_size_pct", 5.0, 0)
        np.testing.assert_array_equal(result, st.returns)


# ── _compute_risk_rating ─────────────────────────────────────────────────


class TestComputeRiskRating:
    def _make_mc(self, prob_ruin=0.0, median_dd=0, prob_profit=0.9):
        return {
            "prob_ruin_50pct": prob_ruin,
            "max_drawdown": {"median_pct": median_dd},
            "prob_profit": prob_profit,
        }

    def _make_crisis(self, dd_pct=-20, hedged=None):
        return [{"portfolio_drawdown_pct": dd_pct, "hedged_portfolio_drawdown_pct": hedged}]

    def test_low_rating(self):
        st = StressTester([0.001] * 50)
        mc = self._make_mc(prob_ruin=0.0, median_dd=-5, prob_profit=0.95)
        crisis = self._make_crisis(dd_pct=-10)
        assert st._compute_risk_rating(mc, crisis) == "LOW"

    def test_moderate_rating(self):
        st = StressTester([0.001] * 50)
        mc = self._make_mc(prob_ruin=0.02, median_dd=-12, prob_profit=0.80)
        crisis = self._make_crisis(dd_pct=-25)
        assert st._compute_risk_rating(mc, crisis) == "MODERATE"

    def test_high_rating(self):
        st = StressTester([0.001] * 50)
        # prob_ruin 0.02 → +1, median_dd 15 → +1, crisis 45 → +2, prob_profit 0.80 → 0
        # total = 4 → HIGH
        mc = self._make_mc(prob_ruin=0.02, median_dd=-15, prob_profit=0.80)
        crisis = self._make_crisis(dd_pct=-45)
        assert st._compute_risk_rating(mc, crisis) == "HIGH"

    def test_critical_rating(self):
        st = StressTester([0.001] * 50)
        mc = self._make_mc(prob_ruin=0.15, median_dd=-35, prob_profit=0.40)
        crisis = self._make_crisis(dd_pct=-65)
        assert st._compute_risk_rating(mc, crisis) == "CRITICAL"

    def test_empty_crisis_no_crash(self):
        st = StressTester([0.001] * 50)
        mc = self._make_mc()
        assert st._compute_risk_rating(mc, []) in ("LOW", "MODERATE")

    def test_hedged_dd_used_when_available(self):
        """When hedged DD is available, it should be used for rating."""
        st = StressTester([0.001] * 50)
        mc = self._make_mc(prob_ruin=0.0, median_dd=-5, prob_profit=0.9)
        # Unhedged is -70 (would be CRITICAL), but hedged is -15 (mild)
        crisis = self._make_crisis(dd_pct=-70, hedged=-15)
        rating = st._compute_risk_rating(mc, crisis)
        # With hedged DD of -15, crisis score = 0 (under 20), not 3
        assert rating in ("LOW", "MODERATE")


# ── run_all integration ──────────────────────────────────────────────────


class TestRunAll:
    def test_run_all_returns_all_sections(self):
        returns = [0.001] * 50
        st = StressTester(returns, n_simulations=100, seed=42)
        result = st.run_all()
        assert "monte_carlo" in result
        assert "crisis_scenarios" in result
        assert "sensitivity" in result
        assert "summary" in result

    def test_summary_has_risk_rating(self):
        returns = [0.001] * 50
        st = StressTester(returns, n_simulations=100, seed=42)
        result = st.run_all()
        assert result["summary"]["risk_rating"] in ("LOW", "MODERATE", "HIGH", "CRITICAL")

    def test_summary_historical_metrics(self):
        returns = [0.001] * 50
        st = StressTester(returns, n_simulations=100, seed=42)
        result = st.run_all()
        hist = result["summary"]["historical"]
        assert "sharpe" in hist
        assert "max_drawdown_pct" in hist
        assert "cagr_pct" in hist
        assert "n_days" in hist
        assert hist["n_days"] == 50

    def test_run_all_with_hedge_config(self):
        from compass.crisis_hedge import CrisisHedgeConfig
        returns = [0.001] * 50
        st = StressTester(returns, n_simulations=100, seed=42)
        result = st.run_all(crisis_hedge_config=CrisisHedgeConfig())
        for crisis in result["crisis_scenarios"]:
            assert crisis["hedged_portfolio_drawdown_pct"] is not None

    def test_sensitivity_contains_default_params(self):
        returns = [0.001] * 50
        st = StressTester(returns, n_simulations=100, seed=42)
        result = st.run_all()
        sensitivity = result["sensitivity"]
        assert "position_size_pct" in sensitivity
        assert "stop_loss_multiplier" in sensitivity
        assert "iv_rank_threshold" in sensitivity


# ── CRISIS_SCENARIOS constant ────────────────────────────────────────────


class TestCrisisScenarios:
    def test_all_scenarios_have_required_keys(self):
        for s in CRISIS_SCENARIOS:
            assert "name" in s
            assert "description" in s
            assert "daily_shocks" in s
            assert "vix_start" in s
            assert "vix_peak" in s

    def test_all_shocks_are_non_empty(self):
        for s in CRISIS_SCENARIOS:
            assert len(s["daily_shocks"]) > 0

    def test_covid_scenario_has_23_days(self):
        covid = next(s for s in CRISIS_SCENARIOS if "COVID" in s["name"])
        assert len(covid["daily_shocks"]) == 23

    def test_flash_crash_single_day(self):
        flash = next(s for s in CRISIS_SCENARIOS if "Flash" in s["name"])
        assert len(flash["daily_shocks"]) == 1
