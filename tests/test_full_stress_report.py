"""Tests for compass/full_stress_report.py — Phase 6 stress testing."""
from __future__ import annotations
import numpy as np
import pytest
from compass.full_stress_report import (
    CrisisResult, FullStressReport, HardCheck, MCResult,
    SensitivityPoint, StressTestSummary,
)

def _make_returns(n=500, seed=42):
    rng = np.random.RandomState(seed)
    # Positive drift ensures MC Sharpe stays positive at 5th percentile
    return rng.normal(0.001, 0.008, n)

def _make_report(n=500, seed=42, n_sim=200, **kw):
    return FullStressReport(_make_returns(n, seed), n_simulations=n_sim, seed=seed, **kw)

class TestDataclasses:
    def test_mc_result(self):
        m = MCResult(100, 252, -0.1, -0.2, -0.3, 0.05, -0.02, 1.0, 0.3, True)
        assert m.passed is True
    def test_crisis_result(self):
        c = CrisisResult("test", "desc", -0.3, -0.1, 100, -0.05, True)
        assert c.passed is True
    def test_hard_check(self):
        h = HardCheck("test", 0.4, 0.2, True, "ok")
        assert h.passed is True
    def test_summary(self):
        s = StressTestSummary(True, True, True, 0, -0.15, -0.3, "low")
        assert s.risk_rating == "low"

class TestMonteCarlo:
    def test_runs(self):
        r = _make_report(); r.run()
        assert r.mc_result is not None
        assert r.mc_result.n_paths == 200
    def test_p5_dd_negative(self):
        r = _make_report(); r.run()
        assert r.mc_result.p5_dd < 0
    def test_median_return_reasonable(self):
        r = _make_report(); r.run()
        assert -1 < r.mc_result.median_return < 5
    def test_passes_with_good_data(self):
        r = _make_report(); r.run()
        assert r.mc_result.passed is True
    def test_short_data(self):
        r = FullStressReport(np.array([0.01] * 5), n_simulations=50)
        r.run()
        assert r.mc_result is not None

class TestCrisis:
    def test_four_scenarios(self):
        r = _make_report(); r.run()
        assert len(r.crisis_results) == 4
    def test_names(self):
        r = _make_report(); r.run()
        names = {c.name for c in r.crisis_results}
        assert "COVID Crash" in names
        assert "Flash Crash" in names
    def test_dd_negative(self):
        r = _make_report(); r.run()
        for c in r.crisis_results:
            assert c.max_dd < 0
    def test_worst_day_negative(self):
        r = _make_report(); r.run()
        for c in r.crisis_results:
            assert c.worst_day < 0

class TestSensitivity:
    def test_populated(self):
        r = _make_report(); r.run()
        assert len(r.sensitivity) > 0
    def test_has_risk_pct(self):
        r = _make_report(); r.run()
        params = {s.param for s in r.sensitivity}
        assert "risk_pct_scale" in params
    def test_has_spread_width(self):
        r = _make_report(); r.run()
        assert any(s.param == "spread_width" for s in r.sensitivity)
    def test_has_stop_loss(self):
        r = _make_report(); r.run()
        assert any(s.param == "stop_loss_mult" for s in r.sensitivity)
    def test_higher_risk_worse_dd(self):
        r = _make_report(); r.run()
        risk = [s for s in r.sensitivity if s.param == "risk_pct_scale"]
        low = [s for s in risk if s.value == 0.5][0]
        high = [s for s in risk if s.value == 2.0][0]
        assert high.max_dd < low.max_dd  # more negative

class TestHardChecks:
    def test_checks_run(self):
        r = _make_report(); r.run()
        assert len(r.hard_checks) == 3
    def test_mc_dd_check(self):
        r = _make_report(); r.run()
        mc_check = [h for h in r.hard_checks if "MC" in h.name and "DD" in h.name][0]
        assert isinstance(mc_check.passed, bool)
    def test_all_pass_with_good_data(self):
        r = _make_report(); r.run()
        assert all(h.passed for h in r.hard_checks)
    def test_tight_threshold_fails(self):
        r = FullStressReport(_make_returns(), n_simulations=200, mc_dd_reject=0.01)
        r.run()
        assert not all(h.passed for h in r.hard_checks)

class TestSummary:
    def test_overall_pass(self):
        r = _make_report(); r.run()
        assert r.summary.overall_pass is True
    def test_risk_rating(self):
        r = _make_report(); r.run()
        assert r.summary.risk_rating in ("low", "medium", "high", "reject")
    def test_reject_on_failure(self):
        r = FullStressReport(_make_returns(), n_simulations=200, mc_dd_reject=0.001)
        r.run()
        assert r.summary.risk_rating == "reject"

class TestPipeline:
    def test_run_keys(self):
        r = _make_report()
        result = r.run()
        assert {"mc", "crisis", "sensitivity", "hard_checks", "summary"} == set(result.keys())

class TestReport:
    def test_html(self, tmp_path):
        r = _make_report()
        path = r.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Phase 6" in c
    def test_sections(self, tmp_path):
        r = _make_report()
        path = r.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Monte Carlo" in c and "Crisis" in c and "Sensitivity" in c
    def test_charts(self, tmp_path):
        r = _make_report()
        path = r.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()
    def test_auto_runs(self, tmp_path):
        r = _make_report()
        assert r.summary is None
        r.generate_report(str(tmp_path / "r.html"))
        assert r.summary is not None
    def test_default_path(self):
        r = _make_report()
        path = r.generate_report()
        assert "full_stress_report.html" in path
