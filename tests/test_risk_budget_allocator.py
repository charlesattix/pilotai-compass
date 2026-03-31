"""Tests for compass.risk_budget_allocator — 40+ tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.risk_budget_allocator import (
    RiskBudgetAllocator,
    AllocationMethod,
    ExperimentRiskProfile,
    BudgetAllocation,
    BudgetSnapshot,
    RebalanceTrigger,
    RegimeAdjustment,
    MarginalRiskContribution,
    DEFAULT_REGIME_MULTIPLIERS,
    DEFAULT_REBALANCE_THRESHOLD,
)
from compass.vol_forecaster import VolRegime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profiles(n: int = 3, base_vol: float = 0.15) -> list[ExperimentRiskProfile]:
    """Build N experiment profiles with linearly increasing vol."""
    return [
        ExperimentRiskProfile(
            name=f"exp_{i}",
            volatility=base_vol + i * 0.05,
            cvar_95=0.03 + i * 0.02,
        )
        for i in range(n)
    ]


def _profiles_with_usage() -> list[ExperimentRiskProfile]:
    """Profiles where exp_0 is nearly maxed out."""
    return [
        ExperimentRiskProfile(
            name="exp_0", volatility=0.15, cvar_95=0.03,
            current_var=0.045, max_var=0.05),
        ExperimentRiskProfile(
            name="exp_1", volatility=0.20, cvar_95=0.05,
            current_var=0.01, max_var=0.05),
        ExperimentRiskProfile(
            name="exp_2", volatility=0.25, cvar_95=0.07,
            current_var=0.005, max_var=0.05),
    ]


def _profiles_with_returns(n: int = 3, size: int = 200) -> list[ExperimentRiskProfile]:
    """Profiles with synthetic return series for covariance estimation."""
    rng = np.random.default_rng(42)
    profs = []
    for i in range(n):
        r = pd.Series(rng.normal(0, 0.01 + i * 0.005, size),
                       index=pd.bdate_range(start="2024-01-02", periods=size))
        profs.append(ExperimentRiskProfile(
            name=f"exp_{i}", volatility=0.15 + i * 0.05,
            cvar_95=0.03 + i * 0.02, returns=r,
        ))
    return profs


# ===========================================================================
# Allocation weights
# ===========================================================================

class TestRawWeights:
    def test_risk_parity_inverse_vol(self):
        rba = RiskBudgetAllocator(method=AllocationMethod.RISK_PARITY)
        exps = _profiles(3)
        w = rba._raw_weights(exps)
        # Lower vol → higher weight
        assert w[0] > w[1] > w[2]
        assert np.isclose(w.sum(), 1.0)

    def test_cvar_budget_inverse_cvar(self):
        rba = RiskBudgetAllocator(method=AllocationMethod.CVAR_BUDGET)
        exps = _profiles(3)
        w = rba._raw_weights(exps)
        # Lower CVaR → higher weight
        assert w[0] > w[1] > w[2]
        assert np.isclose(w.sum(), 1.0)

    def test_empty_experiments(self):
        rba = RiskBudgetAllocator()
        assert len(rba._raw_weights([])) == 0

    def test_single_experiment(self):
        rba = RiskBudgetAllocator()
        w = rba._raw_weights(_profiles(1))
        assert np.isclose(w[0], 1.0)

    def test_equal_vol_equal_weight(self):
        exps = [
            ExperimentRiskProfile(name=f"e{i}", volatility=0.20, cvar_95=0.05)
            for i in range(4)
        ]
        rba = RiskBudgetAllocator(method=AllocationMethod.RISK_PARITY)
        w = rba._raw_weights(exps)
        assert np.allclose(w, 0.25)


# ===========================================================================
# Regime management
# ===========================================================================

class TestRegime:
    def test_default_regime(self):
        rba = RiskBudgetAllocator()
        assert rba.current_regime == VolRegime.NORMAL

    def test_set_regime_returns_record(self):
        rba = RiskBudgetAllocator(base_budget=0.15)
        rec = rba.set_regime(VolRegime.HIGH, date=datetime(2026, 1, 1))
        assert isinstance(rec, RegimeAdjustment)
        assert rec.old_regime == VolRegime.NORMAL
        assert rec.new_regime == VolRegime.HIGH
        assert rec.new_budget == pytest.approx(0.15 * 0.70)

    def test_same_regime_returns_none(self):
        rba = RiskBudgetAllocator()
        assert rba.set_regime(VolRegime.NORMAL) is None

    def test_effective_budget_low(self):
        rba = RiskBudgetAllocator(base_budget=0.15)
        rba.set_regime(VolRegime.LOW)
        assert rba.effective_budget == pytest.approx(0.15 * 1.20)

    def test_effective_budget_extreme(self):
        rba = RiskBudgetAllocator(base_budget=0.15)
        rba.set_regime(VolRegime.EXTREME)
        assert rba.effective_budget == pytest.approx(0.15 * 0.40)

    def test_regime_history_tracking(self):
        rba = RiskBudgetAllocator()
        rba.set_regime(VolRegime.HIGH)
        rba.set_regime(VolRegime.EXTREME)
        assert len(rba.regime_history) == 2
        assert rba.regime_history[0].new_regime == VolRegime.HIGH
        assert rba.regime_history[1].new_regime == VolRegime.EXTREME

    def test_custom_multipliers(self):
        custom = {
            VolRegime.LOW: 1.50, VolRegime.NORMAL: 1.0,
            VolRegime.HIGH: 0.50, VolRegime.EXTREME: 0.20,
        }
        rba = RiskBudgetAllocator(base_budget=0.10, regime_multipliers=custom)
        rba.set_regime(VolRegime.LOW)
        assert rba.effective_budget == pytest.approx(0.15)


# ===========================================================================
# Limit enforcement
# ===========================================================================

class TestLimits:
    def test_var_cap(self):
        rba = RiskBudgetAllocator(base_budget=0.50)
        exps = [ExperimentRiskProfile(
            name="big", volatility=0.10, cvar_95=0.02, max_var=0.05)]
        snap = rba.allocate(exps)
        assert snap.allocations[0].allocated_var <= 0.05
        assert snap.allocations[0].capped

    def test_correlation_cap(self):
        rba = RiskBudgetAllocator(base_budget=0.15)
        exps = [ExperimentRiskProfile(
            name="corr", volatility=0.15, cvar_95=0.03,
            correlation_to_portfolio=0.90, max_correlation_contribution=0.50)]
        snap = rba.allocate(exps)
        assert snap.allocations[0].capped
        assert "max_corr" in snap.allocations[0].cap_reason

    def test_position_size_flag(self):
        rba = RiskBudgetAllocator()
        exps = [ExperimentRiskProfile(
            name="over", volatility=0.15, cvar_95=0.03,
            current_position_size=0.20, max_position_size=0.10)]
        snap = rba.allocate(exps)
        assert snap.allocations[0].capped
        assert "max_pos" in snap.allocations[0].cap_reason

    def test_no_cap_when_within_limits(self):
        rba = RiskBudgetAllocator(base_budget=0.10)
        exps = [
            ExperimentRiskProfile(name="a", volatility=0.15, cvar_95=0.03, max_var=0.10),
            ExperimentRiskProfile(name="b", volatility=0.20, cvar_95=0.05, max_var=0.10),
        ]
        snap = rba.allocate(exps)
        for a in snap.allocations:
            assert not a.capped


# ===========================================================================
# Allocation + snapshot
# ===========================================================================

class TestAllocate:
    def test_weights_sum_le_one(self):
        rba = RiskBudgetAllocator()
        snap = rba.allocate(_profiles(5))
        assert sum(a.weight for a in snap.allocations) <= 1.0 + 1e-9

    def test_allocated_var_respects_budget(self):
        rba = RiskBudgetAllocator(base_budget=0.10)
        snap = rba.allocate(_profiles(3))
        assert sum(a.allocated_var for a in snap.allocations) <= rba.effective_budget + 1e-9

    def test_snapshot_recorded(self):
        rba = RiskBudgetAllocator()
        rba.allocate(_profiles(2))
        assert len(rba.snapshots) == 1

    def test_record_false_not_stored(self):
        rba = RiskBudgetAllocator()
        rba.allocate(_profiles(2), record=False)
        assert len(rba.snapshots) == 0

    def test_total_utilisation(self):
        rba = RiskBudgetAllocator(base_budget=0.15)
        exps = _profiles_with_usage()
        snap = rba.allocate(exps)
        # total_used = 0.045 + 0.01 + 0.005 = 0.06
        assert snap.total_utilisation > 0

    def test_regime_affects_allocation(self):
        rba = RiskBudgetAllocator(base_budget=0.15)
        snap_normal = rba.allocate(_profiles(2), record=False)
        rba.set_regime(VolRegime.HIGH)
        snap_high = rba.allocate(_profiles(2), record=False)
        assert snap_high.effective_budget < snap_normal.effective_budget


# ===========================================================================
# Rebalance triggers
# ===========================================================================

class TestRebalanceTriggers:
    def test_trigger_fires(self):
        rba = RiskBudgetAllocator(base_budget=0.15, rebalance_threshold=0.80)
        exps = [
            ExperimentRiskProfile(
                name="hot", volatility=0.15, cvar_95=0.03,
                current_var=0.14, max_var=0.15),
        ]
        snap = rba.allocate(exps)
        assert len(snap.triggers) == 1
        assert snap.triggers[0].experiment_name == "hot"

    def test_no_trigger_below_threshold(self):
        rba = RiskBudgetAllocator(base_budget=0.15)
        exps = _profiles(3)  # no current_var
        snap = rba.allocate(exps)
        assert len(snap.triggers) == 0

    def test_all_triggers_accumulated(self):
        rba = RiskBudgetAllocator(base_budget=0.15, rebalance_threshold=0.50)
        hot = [ExperimentRiskProfile(
            name="hot", volatility=0.15, cvar_95=0.03,
            current_var=0.14, max_var=0.15)]
        rba.allocate(hot, date=datetime(2026, 1, 1))
        rba.allocate(hot, date=datetime(2026, 1, 2))
        assert len(rba.all_triggers) == 2

    def test_needs_rebalance(self):
        rba = RiskBudgetAllocator(base_budget=0.15, rebalance_threshold=0.80)
        hot = [ExperimentRiskProfile(
            name="hot", volatility=0.15, cvar_95=0.03,
            current_var=0.14, max_var=0.15)]
        assert rba.needs_rebalance(hot)

    def test_needs_rebalance_false(self):
        rba = RiskBudgetAllocator(base_budget=0.15)
        assert not rba.needs_rebalance(_profiles(3))

    def test_custom_threshold(self):
        rba = RiskBudgetAllocator(base_budget=0.20, rebalance_threshold=0.50)
        exps = [ExperimentRiskProfile(
            name="mid", volatility=0.15, cvar_95=0.03,
            current_var=0.10, max_var=0.20)]
        snap = rba.allocate(exps)
        assert len(snap.triggers) == 1


# ===========================================================================
# Marginal risk contributions
# ===========================================================================

class TestMarginalRisk:
    def test_mrc_diagonal(self):
        exps = _profiles(3)
        mrcs = RiskBudgetAllocator.marginal_risk_contributions(exps)
        assert len(mrcs) == 3
        assert all(isinstance(m, MarginalRiskContribution) for m in mrcs)

    def test_mrc_sums_to_portfolio_vol(self):
        exps = _profiles(3)
        w = np.array([1 / 3, 1 / 3, 1 / 3])
        mrcs = RiskBudgetAllocator.marginal_risk_contributions(exps, w)
        total = sum(m.marginal_risk for m in mrcs)
        # total MRC should equal portfolio vol (Euler decomposition)
        vols = np.array([e.volatility for e in exps])
        corrs = np.array([e.correlation_to_portfolio for e in exps])
        outer = np.outer(corrs, corrs)
        np.fill_diagonal(outer, 1.0)
        cov = np.outer(vols, vols) * outer
        port_vol = np.sqrt(float(w @ cov @ w))
        assert total == pytest.approx(port_vol, rel=1e-6)

    def test_mrc_with_returns(self):
        exps = _profiles_with_returns(3, 200)
        mrcs = RiskBudgetAllocator.marginal_risk_contributions(exps)
        assert len(mrcs) == 3
        # Higher vol experiment should have higher marginal risk (roughly)
        assert mrcs[2].marginal_risk >= mrcs[0].marginal_risk - 0.01

    def test_mrc_empty(self):
        assert RiskBudgetAllocator.marginal_risk_contributions([]) == []

    def test_pct_contribution_sums_near_one(self):
        exps = _profiles(4)
        w = np.ones(4) / 4
        mrcs = RiskBudgetAllocator.marginal_risk_contributions(exps, w)
        total_pct = sum(m.pct_contribution for m in mrcs)
        assert total_pct == pytest.approx(1.0, rel=1e-4)

    def test_risk_contributions_convenience(self):
        rba = RiskBudgetAllocator()
        exps = _profiles(3)
        rc = rba.risk_contributions(exps)
        assert set(rc.keys()) == {"exp_0", "exp_1", "exp_2"}
        assert sum(rc.values()) == pytest.approx(1.0, rel=1e-4)


# ===========================================================================
# Utilisation queries
# ===========================================================================

class TestUtilisation:
    def test_utilisation_by_experiment(self):
        rba = RiskBudgetAllocator(base_budget=0.15)
        exps = _profiles_with_usage()
        u = rba.utilisation_by_experiment(exps)
        assert "exp_0" in u
        assert u["exp_0"] > u["exp_2"]

    def test_zero_usage_zero_util(self):
        rba = RiskBudgetAllocator()
        exps = _profiles(3)
        u = rba.utilisation_by_experiment(exps)
        assert all(v == 0.0 for v in u.values())


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        rba = RiskBudgetAllocator()
        out = tmp_path / "risk_budget.html"
        result = rba.generate_report(_profiles(3), output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Risk Budget Report" in html

    def test_contains_pie_chart(self, tmp_path):
        rba = RiskBudgetAllocator()
        out = tmp_path / "r.html"
        rba.generate_report(_profiles(3), output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Risk Contribution" in html

    def test_contains_utilisation_bars(self, tmp_path):
        rba = RiskBudgetAllocator()
        out = tmp_path / "r.html"
        rba.generate_report(_profiles_with_usage(), output_path=str(out))
        html = out.read_text()
        assert "rebal" in html  # threshold label

    def test_contains_regime_history(self, tmp_path):
        rba = RiskBudgetAllocator()
        rba.set_regime(VolRegime.HIGH, date=datetime(2026, 1, 15))
        out = tmp_path / "r.html"
        rba.generate_report(_profiles(2), output_path=str(out))
        html = out.read_text()
        assert "Regime Adjustment History" in html
        assert "HIGH" in html

    def test_contains_trigger_section(self, tmp_path):
        rba = RiskBudgetAllocator(base_budget=0.15, rebalance_threshold=0.50)
        hot = [ExperimentRiskProfile(
            name="hot", volatility=0.15, cvar_95=0.03,
            current_var=0.14, max_var=0.15)]
        rba.allocate(hot, date=datetime(2026, 1, 1))
        out = tmp_path / "r.html"
        rba.generate_report(hot, output_path=str(out))
        html = out.read_text()
        assert "Rebalance Triggers" in html

    def test_contains_marginal_risk_column(self, tmp_path):
        rba = RiskBudgetAllocator()
        out = tmp_path / "r.html"
        rba.generate_report(_profiles(3), output_path=str(out))
        html = out.read_text()
        assert "Marginal Risk" in html

    def test_timeline_with_multiple_snapshots(self, tmp_path):
        rba = RiskBudgetAllocator(base_budget=0.15)
        exps = _profiles_with_usage()
        for d in range(1, 6):
            rba.allocate(exps, date=datetime(2026, 1, d))
        out = tmp_path / "r.html"
        rba.generate_report(exps, output_path=str(out))
        html = out.read_text()
        assert "Utilisation Over Time" in html
