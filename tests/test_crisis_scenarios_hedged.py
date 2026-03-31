"""
Tests for CrisisHedgeController integration into StressTester.run_crisis_scenarios().

Verifies that:
  - Without crisis_hedge_config, results have no hedged fields (backward compat)
  - With crisis_hedge_config, hedged_portfolio_drawdown_pct is populated
  - Hedged DD is strictly less severe than unhedged DD for all scenarios
  - COVID scenario hedged DD drops below -40% (from ~-51.8% unhedged)
  - VIX interpolation correctly drives position scaling day-by-day
  - run_all() passes crisis_hedge_config through to crisis scenarios
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from compass.crisis_hedge import CrisisHedgeConfig, CrisisHedgeController
from compass.stress_test import StressTester, CRISIS_SCENARIOS, _build_crash_path


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_returns() -> np.ndarray:
    """252 days of synthetic daily returns (mean ~0.04% / day)."""
    rng = np.random.RandomState(42)
    return rng.normal(0.0004, 0.01, 252)


@pytest.fixture
def tester(sample_returns) -> StressTester:
    return StressTester(sample_returns, starting_capital=100_000, n_simulations=100, seed=42)


@pytest.fixture
def hedge_config() -> CrisisHedgeConfig:
    return CrisisHedgeConfig(log_decisions=False)


# ─── Backward compatibility: no hedge config ─────────────────────────────────

class TestCrisisScenariosUnhedged:

    def test_no_hedged_fields_without_config(self, tester):
        results = tester.run_crisis_scenarios()
        for r in results:
            assert r["hedged_portfolio_drawdown_pct"] is None
            assert r["hedged_trough_value"] is None
            assert r["hedged_equity_path"] is None

    def test_existing_fields_unchanged(self, tester):
        results = tester.run_crisis_scenarios()
        for r in results:
            assert "portfolio_drawdown_pct" in r
            assert "trough_value" in r
            assert "spread_beta" in r
            assert r["spread_beta"] == 1.5

    def test_run_all_without_config_backward_compat(self, tester):
        results = tester.run_all()
        for r in results["crisis_scenarios"]:
            assert r["hedged_portfolio_drawdown_pct"] is None


# ─── Hedged crisis scenarios ─────────────────────────────────────────────────

class TestCrisisScenariosHedged:

    def test_hedged_fields_populated(self, tester, hedge_config):
        results = tester.run_crisis_scenarios(crisis_hedge_config=hedge_config)
        for r in results:
            assert r["hedged_portfolio_drawdown_pct"] is not None
            assert r["hedged_trough_value"] is not None
            assert r["hedged_equity_path"] is not None

    def test_hedged_dd_less_severe_than_unhedged(self, tester, hedge_config):
        """Hedged DD should be less negative (or equal for short/low-VIX scenarios)."""
        results = tester.run_crisis_scenarios(crisis_hedge_config=hedge_config)
        for r in results:
            unhedged = r["portfolio_drawdown_pct"]
            hedged = r["hedged_portfolio_drawdown_pct"]
            assert hedged >= unhedged, (
                f"{r['name']}: hedged DD ({hedged:.1f}%) should be <= "
                f"unhedged ({unhedged:.1f}%)"
            )

    def test_covid_hedged_dd_under_40_pct(self, tester, hedge_config):
        """Primary acceptance criterion: COVID hedged DD < 40%."""
        results = tester.run_crisis_scenarios(crisis_hedge_config=hedge_config)
        covid = next(r for r in results if "COVID" in r["name"])
        assert abs(covid["hedged_portfolio_drawdown_pct"]) < 40.0, (
            f"COVID hedged DD was {covid['hedged_portfolio_drawdown_pct']:.1f}%, "
            f"expected abs value < 40%"
        )

    def test_covid_unhedged_dd_near_51_pct(self, tester):
        """Sanity: unhedged COVID DD should be around -51% (spread_beta=1.5)."""
        results = tester.run_crisis_scenarios()
        covid = next(r for r in results if "COVID" in r["name"])
        # -34% * 1.5 ≈ -51%, allow some tolerance for the noise in _build_crash_path
        assert -60.0 < covid["portfolio_drawdown_pct"] < -45.0, (
            f"Unhedged COVID DD was {covid['portfolio_drawdown_pct']:.1f}%, "
            f"expected approximately -51%"
        )

    def test_all_scenarios_hedged_dd_under_40_pct(self, tester, hedge_config):
        """All scenarios should have hedged DD under 40% with default config."""
        results = tester.run_crisis_scenarios(crisis_hedge_config=hedge_config)
        for r in results:
            assert abs(r["hedged_portfolio_drawdown_pct"]) < 40.0, (
                f"{r['name']}: hedged DD was {r['hedged_portfolio_drawdown_pct']:.1f}%, "
                f"expected abs value < 40%"
            )

    def test_hedged_equity_path_length_matches_shocks(self, tester, hedge_config):
        results = tester.run_crisis_scenarios(crisis_hedge_config=hedge_config)
        for r, s in zip(results, CRISIS_SCENARIOS):
            n_days = len(s["daily_shocks"])
            # equity path has n_days+1 entries (starting value + one per day)
            assert len(r["hedged_equity_path"]) == n_days + 1

    def test_hedged_trough_value_consistent_with_dd(self, tester, hedge_config):
        results = tester.run_crisis_scenarios(crisis_hedge_config=hedge_config)
        for r in results:
            expected_trough = 100_000 * (1 + r["hedged_portfolio_drawdown_pct"] / 100)
            assert abs(r["hedged_trough_value"] - expected_trough) < 10.0, (
                f"{r['name']}: trough value {r['hedged_trough_value']:.2f} "
                f"doesn't match DD {r['hedged_portfolio_drawdown_pct']:.2f}%"
            )

    def test_run_all_passes_hedge_config(self, tester, hedge_config):
        results = tester.run_all(crisis_hedge_config=hedge_config)
        for r in results["crisis_scenarios"]:
            assert r["hedged_portfolio_drawdown_pct"] is not None


# ─── VIX interpolation and scaling ───────────────────────────────────────────

class TestVixInterpolationInCrisis:

    def test_single_day_scenario_uses_start_vix(self):
        """Flash crash (1 day): only day_idx=0 → uses vix_start, not vix_peak."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0004, 0.01, 252)
        tester = StressTester(returns, starting_capital=100_000, n_simulations=100, seed=42)
        cfg = CrisisHedgeConfig(log_decisions=False)

        # Custom 1-day scenario with vix_start below floor (scale=1.0)
        scenario = {
            "name": "Single Day Test",
            "description": "1-day shock",
            "daily_shocks": [-0.10],
            "vix_start": 10.0,
            "vix_peak": 65.0,
        }
        results = tester.run_crisis_scenarios(
            scenarios=[scenario], crisis_hedge_config=cfg
        )
        r = results[0]

        # For 1 day: day_idx=0, t=0/max(0,1)=0, vix = vix_start = 10.0
        # VIX 10 is below floor=12, so scale=1.0 → hedged ≈ unhedged
        assert r["hedged_portfolio_drawdown_pct"] is not None
        assert abs(r["hedged_portfolio_drawdown_pct"] - r["portfolio_drawdown_pct"]) < 1.0

    def test_long_scenario_has_significant_hedge_benefit(self):
        """COVID (23 days, VIX 15→82): hedge should reduce DD substantially."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0004, 0.01, 252)
        tester = StressTester(returns, starting_capital=100_000, n_simulations=100, seed=42)
        cfg = CrisisHedgeConfig(log_decisions=False)

        results = tester.run_crisis_scenarios(crisis_hedge_config=cfg)
        covid = next(r for r in results if "COVID" in r["name"])

        reduction = abs(covid["portfolio_drawdown_pct"]) - abs(covid["hedged_portfolio_drawdown_pct"])
        assert reduction > 10.0, (
            f"Hedge should reduce COVID DD by >10pp, got {reduction:.1f}pp"
        )

    def test_low_vix_scenario_minimal_hedge_impact(self):
        """When VIX stays below scale_floor, hedge has minimal impact."""
        rng = np.random.RandomState(42)
        returns = rng.normal(0.0004, 0.01, 252)
        tester = StressTester(returns, starting_capital=100_000, n_simulations=100, seed=42)
        cfg = CrisisHedgeConfig(log_decisions=False)

        scenario = {
            "name": "Low VIX Selloff",
            "description": "Gentle selloff with low VIX",
            "daily_shocks": _build_crash_path(-0.15, 30),
            "vix_start": 9.0,
            "vix_peak": 11.0,   # stays below default floor of 12
        }
        results = tester.run_crisis_scenarios(
            scenarios=[scenario], crisis_hedge_config=cfg
        )
        r = results[0]

        # Since VIX stays below 12 (floor), scale=1.0 throughout → no hedge benefit
        assert abs(r["hedged_portfolio_drawdown_pct"] - r["portfolio_drawdown_pct"]) < 0.5


# ─── Summary integration ─────────────────────────────────────────────────────

class TestSummaryWithHedge:

    def test_summary_contains_hedged_worst_crisis(self, tester, hedge_config):
        results = tester.run_all(crisis_hedge_config=hedge_config)
        worst = results["summary"]["worst_crisis"]
        assert "hedged_portfolio_drawdown_pct" in worst
        assert worst["hedged_portfolio_drawdown_pct"] is not None

    def test_summary_without_hedge_has_none_hedged_field(self, tester):
        results = tester.run_all()
        worst = results["summary"]["worst_crisis"]
        assert worst["hedged_portfolio_drawdown_pct"] is None
