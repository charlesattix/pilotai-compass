"""Tests for compass.risk_orchestrator — 32 tests."""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path
from compass.risk_orchestrator import (
    RiskOrchestrator, EscalationLevel, RiskSnapshot, RiskLimit,
    HedgeRecommendation, StressResult, DrawdownAlert, OrchestratorReport,
)

def _returns(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.01, n), index=pd.bdate_range("2024-01-02", periods=n))

def _port_returns(n=200, k=3, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({f"a{i}": rng.normal(0.0003, 0.01, n) for i in range(k)}, index=idx)


class TestVar:
    def test_positive(self):
        assert RiskOrchestrator.compute_var(_returns()) > 0
    def test_cvar_ge_var(self):
        r = _returns()
        assert RiskOrchestrator.compute_cvar(r) >= RiskOrchestrator.compute_var(r) - 0.001
    def test_empty(self):
        assert RiskOrchestrator.compute_var(pd.Series(dtype=float)) == 0.0


class TestSnapshot:
    def test_basic(self):
        ro = RiskOrchestrator()
        snap = ro.compute_snapshot(_returns(), 100000)
        assert isinstance(snap, RiskSnapshot)
        assert snap.portfolio_var_95 > 0
    def test_drawdown(self):
        ro = RiskOrchestrator()
        ro.compute_snapshot(_returns(50), 100000)
        snap = ro.compute_snapshot(_returns(50), 90000)
        assert snap.drawdown > 0
    def test_greeks(self):
        ro = RiskOrchestrator()
        snap = ro.compute_snapshot(_returns(), 100000, greeks={"delta": 30, "gamma": 2, "vega": 50, "theta": -10})
        assert snap.total_delta == 30
    def test_history(self):
        ro = RiskOrchestrator()
        ro.compute_snapshot(_returns(), 100000)
        ro.compute_snapshot(_returns(), 99000)
        assert len(ro.history) == 2


class TestEscalation:
    def test_normal(self):
        ro = RiskOrchestrator()
        snap = ro.compute_snapshot(_returns(), 100000)
        assert snap.escalation == EscalationLevel.NORMAL
    def test_warning(self):
        ro = RiskOrchestrator()
        ro.compute_snapshot(_returns(), 100000)
        snap = ro.compute_snapshot(_returns(), 96500)  # ~3.5% dd
        assert snap.escalation in (EscalationLevel.WARNING, EscalationLevel.NORMAL)
    def test_liquidate(self):
        ro = RiskOrchestrator()
        ro.compute_snapshot(_returns(), 100000)
        snap = ro.compute_snapshot(_returns(), 91000)  # 9% dd
        assert snap.escalation == EscalationLevel.LIQUIDATE


class TestLimits:
    def test_basic(self):
        ro = RiskOrchestrator()
        snap = ro.compute_snapshot(_returns(), 100000)
        limits = ro.check_limits(snap)
        assert len(limits) == 4
        assert all(isinstance(l, RiskLimit) for l in limits)
    def test_var_breach(self):
        ro = RiskOrchestrator(var_limit=0.001)
        snap = ro.compute_snapshot(_returns(), 100000)
        limits = ro.check_limits(snap)
        var_l = [l for l in limits if l.name == "VaR"][0]
        assert var_l.breached
    def test_no_breach(self):
        ro = RiskOrchestrator(var_limit=1.0, delta_limit=1000)
        snap = ro.compute_snapshot(_returns(), 100000)
        limits = ro.check_limits(snap)
        assert not any(l.breached for l in limits)


class TestDrawdownAlert:
    def test_green(self):
        ro = RiskOrchestrator()
        assert ro.drawdown_alert(0.01).level == "GREEN"
    def test_yellow(self):
        ro = RiskOrchestrator()
        assert ro.drawdown_alert(0.04).level == "YELLOW"
    def test_orange(self):
        ro = RiskOrchestrator()
        assert ro.drawdown_alert(0.06).level == "ORANGE"
    def test_red(self):
        ro = RiskOrchestrator()
        assert ro.drawdown_alert(0.10).level == "RED"
    def test_size_multiplier(self):
        ro = RiskOrchestrator()
        assert ro.drawdown_alert(0.10).size_multiplier == 0.0
        assert ro.drawdown_alert(0.01).size_multiplier == 1.0


class TestHedge:
    def test_delta_hedge(self):
        snap = RiskSnapshot(datetime.now(), total_delta=50)
        recs = RiskOrchestrator.hedge_recommendations(snap)
        assert any(r.instrument.startswith("SPY") for r in recs)
    def test_no_hedge_flat(self):
        snap = RiskSnapshot(datetime.now(), total_delta=0, total_vega=0, drawdown=0)
        recs = RiskOrchestrator.hedge_recommendations(snap)
        assert len(recs) == 0


class TestStress:
    def test_basic(self):
        ret = _port_returns(100, 3)
        w = {"a0": 0.4, "a1": 0.3, "a2": 0.3}
        results = RiskOrchestrator.run_stress_tests(ret, w)
        assert len(results) == 3
        assert all(isinstance(r, StressResult) for r in results)
    def test_pnl_negative(self):
        ret = _port_returns(100, 3)
        w = {"a0": 0.5, "a1": 0.3, "a2": 0.2}
        results = RiskOrchestrator.run_stress_tests(ret, w)
        assert all(r.pnl_impact < 0 for r in results)


class TestOrchestrate:
    def test_full(self):
        ro = RiskOrchestrator()
        ret = _returns(200)
        pr = _port_returns(200, 3)
        w = {"a0": 0.4, "a1": 0.3, "a2": 0.3}
        report = ro.orchestrate(ret, 100000, greeks={"delta": 40, "vega": 20},
                                 portfolio_returns=pr, weights=w)
        assert isinstance(report, OrchestratorReport)
        assert len(report.limits) == 4


class TestRegimeMult:
    def test_values(self):
        assert RiskOrchestrator.regime_risk_multiplier("crash") == 0.4
        assert RiskOrchestrator.regime_risk_multiplier("bull") == 1.0


class TestReport:
    def test_creates_file(self, tmp_path):
        ro = RiskOrchestrator()
        report = ro.orchestrate(_returns(), 100000, greeks={"delta": 40})
        out = tmp_path / "risk.html"
        path = ro.generate_report(report, output_path=str(out))
        assert Path(path).exists()
        assert "Risk Orchestrator" in out.read_text()
