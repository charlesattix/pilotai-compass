"""Tests for compass/liquidity_risk.py — liquidity risk monitor."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from compass.liquidity_risk import (
    AmihudResult, CapacityResult, LiquidityRegime, LiquidityRiskMonitor,
    LiquidityScore, LiquiditySnapshot, SpreadAlert, StressScenario,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_market_data(n=200, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 430 + rng.normal(0, 1.5, n).cumsum()
    spread = np.abs(rng.normal(0.05, 0.02, n))
    # Inject a few stress periods
    stress_start = min(150, n - 10) if n > 10 else 0
    stress_end = min(stress_start + 10, n)
    stress_len = stress_end - stress_start
    if stress_len > 0:
        spread[stress_start:stress_end] = rng.uniform(0.15, 0.30, stress_len)
    volume = rng.uniform(5000, 20000, n)
    if stress_len > 0:
        volume[stress_start:stress_end] = rng.uniform(500, 2000, stress_len)
    return pd.DataFrame({"close": close, "bid_ask_spread": spread, "volume": volume}, index=dates)

def _make_monitor(n=200, seed=42, **kwargs):
    return LiquidityRiskMonitor(_make_market_data(n, seed), **kwargs)

# ── Dataclasses ──────────────────────────────────────────────────────────

class TestDataclasses:
    def test_snapshot(self):
        s = LiquiditySnapshot("2024-01-01", 0.05, 10000, 5000, 0.001, 0.7, "normal")
        assert s.score == pytest.approx(0.7)
    def test_score(self):
        s = LiquidityScore(0.7, 0.8, 0.6, 0.7, "normal")
        assert s.regime == "normal"
    def test_regime(self):
        r = LiquidityRegime("stressed", 0.3, 30, 0.4, 0.12)
        assert r.probability == pytest.approx(0.3)
    def test_capacity(self):
        c = CapacityResult(5, 3, 0.6, 0.5, "reason")
        assert c.adjusted_contracts == 3
    def test_amihud(self):
        a = AmihudResult(0.001, 0.0005, 0.8, 1.5, 7.5)
        assert a.illiquidity_premium_bps == pytest.approx(7.5)
    def test_stress(self):
        s = StressScenario("Flash", 10.0, 0.05, 10.0, 0.1, 0.2)
        assert s.adjusted_score == pytest.approx(0.1)
    def test_alert(self):
        a = SpreadAlert("2024-06-01", 0.25, 3.5, "critical")
        assert a.severity == "critical"

# ── Snapshots ────────────────────────────────────────────────────────────

class TestSnapshots:
    def test_snapshot_count(self):
        m = _make_monitor(n=100)
        m.analyze()
        assert len(m.snapshots) == 100
    def test_score_range(self):
        m = _make_monitor()
        m.analyze()
        for s in m.snapshots:
            assert 0 <= s.score <= 1
    def test_regime_valid(self):
        m = _make_monitor()
        m.analyze()
        for s in m.snapshots:
            assert s.regime in ("normal", "stressed", "crisis")
    def test_depth_positive(self):
        m = _make_monitor()
        m.analyze()
        for s in m.snapshots:
            assert s.depth >= 0

# ── Liquidity score ──────────────────────────────────────────────────────

class TestLiquidityScore:
    def test_current_score_populated(self):
        m = _make_monitor()
        m.analyze()
        assert m.current_score is not None
    def test_components_range(self):
        m = _make_monitor()
        m.analyze()
        cs = m.current_score
        assert 0 <= cs.spread_component <= 1
        assert 0 <= cs.volume_component <= 1
        assert 0 <= cs.impact_component <= 1
    def test_regime_matches_score(self):
        m = _make_monitor()
        m.analyze()
        cs = m.current_score
        if cs.score >= 0.6:
            assert cs.regime == "normal"
        elif cs.score >= 0.3:
            assert cs.regime == "stressed"
        else:
            assert cs.regime == "crisis"

# ── Regime classification ────────────────────────────────────────────────

class TestRegimes:
    def test_regimes_populated(self):
        m = _make_monitor()
        m.analyze()
        assert len(m.regime_history) > 0
    def test_probabilities_sum(self):
        m = _make_monitor()
        m.analyze()
        total = sum(r.probability for r in m.regime_history)
        assert total == pytest.approx(1.0, abs=0.01)
    def test_has_stressed_period(self):
        m = _make_monitor()
        m.analyze()
        regimes = {r.regime for r in m.regime_history}
        # We injected stress, should see stressed or crisis
        assert "stressed" in regimes or "crisis" in regimes

# ── Amihud ───────────────────────────────────────────────────────────────

class TestAmihud:
    def test_amihud_computed(self):
        m = _make_monitor()
        m.analyze()
        assert m.amihud is not None
    def test_percentile_range(self):
        m = _make_monitor()
        m.analyze()
        assert 0 <= m.amihud.percentile <= 1
    def test_premium_non_negative(self):
        m = _make_monitor()
        m.analyze()
        assert m.amihud.illiquidity_premium_bps >= 0

# ── Capacity ─────────────────────────────────────────────────────────────

class TestCapacity:
    def test_capacity_computed(self):
        m = _make_monitor()
        m.analyze()
        assert m.capacity is not None
    def test_adjusted_le_base(self):
        m = _make_monitor()
        m.analyze()
        assert m.capacity.adjusted_contracts <= m.capacity.base_contracts
    def test_factor_range(self):
        m = _make_monitor()
        m.analyze()
        assert 0 < m.capacity.capacity_factor <= 1.0
    def test_custom_base(self):
        m = _make_monitor(base_position=20)
        m.analyze()
        assert m.capacity.base_contracts == 20

# ── Stress scenarios ─────────────────────────────────────────────────────

class TestStress:
    def test_scenarios_generated(self):
        m = _make_monitor()
        m.analyze()
        assert len(m.stress_scenarios) == 5
    def test_normal_scenario_highest_score(self):
        m = _make_monitor()
        m.analyze()
        normal = [s for s in m.stress_scenarios if s.name == "Normal"][0]
        flash = [s for s in m.stress_scenarios if s.name == "Flash Crash"][0]
        assert normal.adjusted_score >= flash.adjusted_score
    def test_capacity_decreases_under_stress(self):
        m = _make_monitor()
        m.analyze()
        normal = [s for s in m.stress_scenarios if s.name == "Normal"][0]
        severe = [s for s in m.stress_scenarios if s.name == "Severe Stress"][0]
        assert severe.capacity_pct <= normal.capacity_pct

# ── Spread alerts ────────────────────────────────────────────────────────

class TestAlerts:
    def test_alerts_detected(self):
        m = _make_monitor()
        m.analyze()
        # Injected stress spreads should trigger alerts
        assert len(m.alerts) > 0
    def test_severity_valid(self):
        m = _make_monitor()
        m.analyze()
        for a in m.alerts:
            assert a.severity in ("warning", "critical")
    def test_z_score_above_threshold(self):
        m = _make_monitor(spread_alert_z=2.0)
        m.analyze()
        for a in m.alerts:
            assert a.z_score > 2.0

# ── Pipeline ─────────────────────────────────────────────────────────────

class TestPipeline:
    def test_analyze_keys(self):
        m = _make_monitor()
        result = m.analyze()
        expected = {"snapshots", "current_score", "regime_history",
                    "amihud", "capacity", "stress_scenarios", "alerts"}
        assert set(result.keys()) == expected
    def test_from_csv(self, tmp_path):
        df = _make_market_data()
        csv = tmp_path / "data.csv"
        df.to_csv(csv)
        m = LiquidityRiskMonitor.from_csv(str(csv))
        m.analyze()
        assert m.current_score is not None

# ── Report ───────────────────────────────────────────────────────────────

class TestReport:
    def test_generates_html(self, tmp_path):
        m = _make_monitor()
        path = m.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Liquidity" in c
    def test_report_sections(self, tmp_path):
        m = _make_monitor()
        path = m.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Timeline" in c and "Regime" in c and "Amihud" in c and "Stress" in c
    def test_report_charts(self, tmp_path):
        m = _make_monitor()
        path = m.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()
    def test_report_auto_analyzes(self, tmp_path):
        m = _make_monitor()
        assert m.current_score is None
        m.generate_report(str(tmp_path / "r.html"))
        assert m.current_score is not None
    def test_report_default_path(self):
        m = _make_monitor()
        path = m.generate_report()
        assert "liquidity_risk.html" in path
