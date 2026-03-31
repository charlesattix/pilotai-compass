from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.risk_aggregator import (
    BUILTIN_STRESS_SCENARIOS,
    RiskAggregator,
    RiskAggResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def sample_returns(rng):
    n = 500
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return {
        "strat_a": pd.Series(rng.normal(0.0005, 0.01, n), index=dates),
        "strat_b": pd.Series(rng.normal(0.0003, 0.015, n), index=dates),
        "strat_c": pd.Series(rng.normal(0.0002, 0.008, n), index=dates),
    }


@pytest.fixture
def sample_weights():
    return {"strat_a": 0.4, "strat_b": 0.35, "strat_c": 0.25}


@pytest.fixture
def agg(sample_returns, sample_weights):
    return RiskAggregator(sample_returns, sample_weights)


# ---------------------------------------------------------------------------
# VaR / CVaR tests
# ---------------------------------------------------------------------------
class TestVarCvar:
    def test_returns_two_levels(self, agg):
        vc = agg.compute_var_cvar()
        assert len(vc) == 2
        assert vc[0]["confidence"] == 0.95
        assert vc[1]["confidence"] == 0.99

    def test_var_positive(self, agg):
        vc = agg.compute_var_cvar()
        for row in vc:
            assert row["var"] > 0

    def test_cvar_geq_var(self, agg):
        vc = agg.compute_var_cvar()
        for row in vc:
            assert row["cvar"] >= row["var"] - 1e-12

    def test_99_geq_95(self, agg):
        vc = agg.compute_var_cvar()
        assert vc[1]["var"] >= vc[0]["var"] - 1e-12
        assert vc[1]["cvar"] >= vc[0]["cvar"] - 1e-12

    def test_var_reasonable_range(self, agg):
        vc = agg.compute_var_cvar()
        # For daily returns with vol ~1%, VaR should be small
        assert vc[0]["var"] < 0.10
        assert vc[1]["var"] < 0.15


# ---------------------------------------------------------------------------
# Marginal contribution tests
# ---------------------------------------------------------------------------
class TestMarginalContributions:
    def test_returns_all_strategies(self, agg):
        mc = agg.compute_marginal_contributions()
        names = {c["strategy"] for c in mc}
        assert names == {"strat_a", "strat_b", "strat_c"}

    def test_has_portfolio_cvar(self, agg):
        mc = agg.compute_marginal_contributions()
        for c in mc:
            assert "portfolio_cvar" in c
            assert c["portfolio_cvar"] > 0

    def test_marginal_values_are_finite(self, agg):
        mc = agg.compute_marginal_contributions()
        for c in mc:
            assert np.isfinite(c["marginal_cvar"])

    def test_marginal_signs(self, agg):
        """At least one marginal contribution should be non-zero."""
        mc = agg.compute_marginal_contributions()
        vals = [c["marginal_cvar"] for c in mc]
        assert any(v != 0.0 for v in vals)


# ---------------------------------------------------------------------------
# Concentration tests
# ---------------------------------------------------------------------------
class TestConcentration:
    def test_herfindahl_bounds(self, agg):
        conc = agg.compute_concentration()
        hhi = conc["herfindahl_index"]
        # For 3 strategies, min HHI = 1/3 ~ 0.333
        assert 0.0 < hhi <= 1.0

    def test_shares_sum_to_one(self, agg):
        conc = agg.compute_concentration()
        total = sum(conc["strategy_shares"].values())
        assert abs(total - 1.0) < 1e-9

    def test_single_strategy_detection(self):
        rng = np.random.RandomState(99)
        n = 200
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        rets = {
            "big": pd.Series(rng.normal(0, 0.01, n), index=dates),
            "small": pd.Series(rng.normal(0, 0.01, n), index=dates),
        }
        weights = {"big": 0.9, "small": 0.1}
        a = RiskAggregator(rets, weights)
        conc = a.compute_concentration()
        assert "big" in conc["concentrated_strategies"]
        assert conc["is_concentrated"]

    def test_equal_weights_not_concentrated(self):
        rng = np.random.RandomState(7)
        n = 200
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        rets = {f"s{i}": pd.Series(rng.normal(0, 0.01, n), index=dates) for i in range(5)}
        weights = {f"s{i}": 0.2 for i in range(5)}
        a = RiskAggregator(rets, weights)
        conc = a.compute_concentration()
        assert not conc["is_concentrated"]
        assert conc["herfindahl_index"] == pytest.approx(0.2, abs=1e-9)


# ---------------------------------------------------------------------------
# Stress test tests
# ---------------------------------------------------------------------------
class TestStressTest:
    def test_builtin_scenarios(self, agg):
        res = agg.stress_test()
        names = {r["scenario"] for r in res}
        assert names == set(BUILTIN_STRESS_SCENARIOS.keys())

    def test_all_negative_drawdown(self, agg):
        res = agg.stress_test()
        for r in res:
            assert r["stressed_max_drawdown"] < 0

    def test_worst_case_is_gfc(self, agg):
        res = agg.stress_test()
        by_name = {r["scenario"]: r for r in res}
        assert by_name["2008_GFC"]["stressed_max_drawdown"] < by_name["2022_BEAR"]["stressed_max_drawdown"]

    def test_custom_scenario(self, agg):
        res = agg.stress_test({"mild": -0.10})
        assert len(res) == 1
        assert res[0]["scenario"] == "mild"
        assert res[0]["stressed_max_drawdown"] < 0

    def test_final_value_consistent(self, agg):
        res = agg.stress_test()
        for r in res:
            expected = 1.0 + r["shock"]
            assert abs(r["final_value"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Liquidity risk tests
# ---------------------------------------------------------------------------
class TestLiquidityRisk:
    def test_basic_computation(self, agg):
        pos = {"strat_a": 100_000, "strat_b": 50_000, "strat_c": 30_000}
        adv = {"strat_a": 1_000_000, "strat_b": 500_000, "strat_c": 200_000}
        liq = agg.compute_liquidity_risk(pos, adv)
        assert "total_liquidation_cost" in liq
        assert liq["total_liquidation_cost"] > 0

    def test_scales_with_position(self, agg):
        adv = {"strat_a": 1_000_000, "strat_b": 1_000_000, "strat_c": 1_000_000}
        liq_small = agg.compute_liquidity_risk(
            {"strat_a": 10_000, "strat_b": 10_000, "strat_c": 10_000}, adv
        )
        liq_large = agg.compute_liquidity_risk(
            {"strat_a": 100_000, "strat_b": 100_000, "strat_c": 100_000}, adv
        )
        assert liq_large["total_liquidation_cost"] > liq_small["total_liquidation_cost"]

    def test_all_strategies_have_costs(self, agg):
        pos = {"strat_a": 50_000, "strat_b": 50_000, "strat_c": 50_000}
        adv = {"strat_a": 500_000, "strat_b": 500_000, "strat_c": 500_000}
        liq = agg.compute_liquidity_risk(pos, adv)
        assert set(liq["strategy_costs"].keys()) == {"strat_a", "strat_b", "strat_c"}

    def test_unwind_fraction(self, agg):
        pos = {"strat_a": 50_000, "strat_b": 50_000, "strat_c": 50_000}
        adv = {"strat_a": 500_000, "strat_b": 500_000, "strat_c": 500_000}
        liq = agg.compute_liquidity_risk(pos, adv)
        assert liq["unwind_fraction"] == 0.50


# ---------------------------------------------------------------------------
# Tail dependency tests
# ---------------------------------------------------------------------------
class TestTailDependency:
    def test_bounded_zero_one(self, agg):
        td = agg.compute_tail_dependency()
        for val in td.values():
            assert 0.0 <= val <= 1.0

    def test_pairs_present(self, agg):
        td = agg.compute_tail_dependency()
        # 3 strategies -> 3 pairs
        assert len(td) == 3

    def test_single_strategy_empty(self):
        rng = np.random.RandomState(11)
        n = 200
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        rets = {"only": pd.Series(rng.normal(0, 0.01, n), index=dates)}
        a = RiskAggregator(rets, {"only": 1.0})
        td = a.compute_tail_dependency()
        assert td == {}


# ---------------------------------------------------------------------------
# Compliance tests
# ---------------------------------------------------------------------------
class TestCompliance:
    def test_no_breaches(self, agg):
        limits = {
            "max_var_95": 1.0,
            "max_var_99": 1.0,
            "max_cvar_95": 1.0,
            "max_cvar_99": 1.0,
            "max_herfindahl": 1.0,
        }
        breaches = agg.check_compliance(limits)
        assert breaches == []

    def test_var_breach(self, agg):
        limits = {"max_var_95": 0.0001}
        breaches = agg.check_compliance(limits)
        assert len(breaches) >= 1
        assert breaches[0]["metric"] == "max_var_95"
        assert breaches[0]["breach_amount"] > 0

    def test_herfindahl_breach(self, agg):
        limits = {"max_herfindahl": 0.01}
        breaches = agg.check_compliance(limits)
        metrics = [b["metric"] for b in breaches]
        assert "max_herfindahl" in metrics

    def test_single_strategy_share_breach(self):
        rng = np.random.RandomState(55)
        n = 200
        dates = pd.date_range("2021-01-01", periods=n, freq="B")
        rets = {
            "dom": pd.Series(rng.normal(0, 0.01, n), index=dates),
            "tiny": pd.Series(rng.normal(0, 0.01, n), index=dates),
        }
        weights = {"dom": 0.8, "tiny": 0.2}
        a = RiskAggregator(rets, weights)
        breaches = a.check_compliance({"max_single_strategy_share": 0.5})
        assert len(breaches) >= 1
        assert any("dom" in b["metric"] for b in breaches)


# ---------------------------------------------------------------------------
# HTML report tests
# ---------------------------------------------------------------------------
class TestHTMLReport:
    def test_report_is_html(self, agg):
        report = agg.generate_report()
        assert "<!DOCTYPE html>" in report
        assert "</html>" in report

    def test_report_contains_sections(self, agg):
        report = agg.generate_report()
        assert "VaR" in report
        assert "CVaR" in report
        assert "Stress Test" in report
        assert "Concentration" in report
        assert "Compliance" in report

    def test_report_with_result(self, agg):
        result = agg.run()
        report = agg.generate_report(result=result)
        assert "<!DOCTYPE html>" in report

    def test_report_contains_strategies(self, agg):
        report = agg.generate_report()
        assert "strat_a" in report
        assert "strat_b" in report
        assert "strat_c" in report


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_single_strategy(self):
        rng = np.random.RandomState(33)
        n = 300
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        rets = {"solo": pd.Series(rng.normal(0.0003, 0.012, n), index=dates)}
        a = RiskAggregator(rets, {"solo": 1.0})

        vc = a.compute_var_cvar()
        assert len(vc) == 2
        assert vc[0]["var"] > 0

        mc = a.compute_marginal_contributions()
        assert len(mc) == 1

        conc = a.compute_concentration()
        assert conc["herfindahl_index"] == pytest.approx(1.0)
        assert conc["is_concentrated"]

    def test_equal_weights(self):
        rng = np.random.RandomState(77)
        n = 300
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        rets = {f"s{i}": pd.Series(rng.normal(0, 0.01, n), index=dates) for i in range(4)}
        weights = {f"s{i}": 0.25 for i in range(4)}
        a = RiskAggregator(rets, weights)
        conc = a.compute_concentration()
        assert conc["herfindahl_index"] == pytest.approx(0.25, abs=1e-9)
        assert not conc["is_concentrated"]

    def test_empty_returns_raises(self):
        with pytest.raises(ValueError):
            RiskAggregator({}, {})

    def test_zero_weights(self):
        rng = np.random.RandomState(88)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        rets = {
            "a": pd.Series(rng.normal(0, 0.01, n), index=dates),
            "b": pd.Series(rng.normal(0, 0.01, n), index=dates),
        }
        a = RiskAggregator(rets, {"a": 0.0, "b": 0.0})
        # Should not crash
        vc = a.compute_var_cvar()
        assert len(vc) == 2


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestRiskAggResult:
    def test_default_fields(self):
        r = RiskAggResult()
        assert r.var_cvar == []
        assert r.marginal_contributions == []
        assert r.concentration == {}
        assert r.stress_results == []
        assert r.liquidity_risk == {}
        assert r.tail_deps == {}
        assert r.compliance_breaches == []
        assert isinstance(r.generated_at, str)
        assert len(r.generated_at) > 0

    def test_custom_fields(self):
        r = RiskAggResult(
            var_cvar=[{"confidence": 0.95, "var": 0.02, "cvar": 0.03}],
            stress_results=[{"scenario": "test", "shock": -0.1}],
        )
        assert len(r.var_cvar) == 1
        assert len(r.stress_results) == 1

    def test_independent_defaults(self):
        r1 = RiskAggResult()
        r2 = RiskAggResult()
        r1.var_cvar.append({"x": 1})
        assert r2.var_cvar == []


# ---------------------------------------------------------------------------
# Full run integration
# ---------------------------------------------------------------------------
class TestFullRun:
    def test_run_returns_result(self, agg):
        result = agg.run(
            position_sizes={"strat_a": 50_000, "strat_b": 30_000, "strat_c": 20_000},
            adv={"strat_a": 500_000, "strat_b": 300_000, "strat_c": 200_000},
            limits={"max_var_95": 1.0},
        )
        assert isinstance(result, RiskAggResult)
        assert len(result.var_cvar) == 2
        assert len(result.marginal_contributions) == 3
        assert result.concentration
        assert len(result.stress_results) == 3
        assert result.liquidity_risk
        assert result.compliance_breaches == []

    def test_run_without_optional(self, agg):
        result = agg.run()
        assert isinstance(result, RiskAggResult)
        assert result.liquidity_risk == {}
        assert result.compliance_breaches == []
