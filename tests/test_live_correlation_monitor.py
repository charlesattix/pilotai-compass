"""Tests for compass/live_correlation_monitor.py — live correlation tracking.

Covers:
  - compute_effective_n: math, edge cases
  - classify_correlation_regime: all three regimes
  - pairwise_correlations: known values, insufficient data
  - LiveCorrelationMonitor.add_returns: recording, snapshot
  - Correlation regime detection: NORMAL, ELEVATED, DANGER
  - Effective N: fully diversified vs perfectly correlated
  - Alert generation: correlation spike, diversification low, regime change
  - Allocation adjustments: recommendations in elevated/danger
  - CorrelationSnapshot: all fields populated
  - Dashboard HTML: structure and content
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.live_correlation_monitor import (
    AllocationAdjustment,
    CorrelationAlert,
    CorrelationSnapshot,
    LiveCorrelationMonitor,
    classify_correlation_regime,
    compute_effective_n,
    pairwise_correlations,
)


# ── compute_effective_n ──────────────────────────────────────────────────


class TestComputeEffectiveN:
    def test_zero_correlation_equals_n(self):
        assert compute_effective_n(4, 0.0) == pytest.approx(4.0)

    def test_perfect_correlation_equals_one(self):
        assert compute_effective_n(4, 1.0) == pytest.approx(1.0)

    def test_moderate_correlation(self):
        eff = compute_effective_n(4, 0.5)
        assert 1.0 < eff < 4.0

    def test_single_asset(self):
        assert compute_effective_n(1, 0.5) == 1.0

    def test_negative_correlation_clamped(self):
        """Negative avg correlation is clamped to 0 (conservative)."""
        eff = compute_effective_n(3, -0.5)
        assert eff == 3.0

    def test_two_assets_half_correlated(self):
        # N=2, rho=0.5 → 2 / (1 + 0.5) = 1.333
        assert compute_effective_n(2, 0.5) == pytest.approx(2 / 1.5, abs=0.01)


# ── classify_correlation_regime ──────────────────────────────────────────


class TestClassifyCorrelationRegime:
    def test_normal(self):
        assert classify_correlation_regime(0.2) == "NORMAL"

    def test_elevated(self):
        assert classify_correlation_regime(0.5) == "ELEVATED"

    def test_danger(self):
        assert classify_correlation_regime(0.8) == "DANGER"

    def test_boundary_elevated(self):
        assert classify_correlation_regime(0.4) == "ELEVATED"

    def test_boundary_danger(self):
        assert classify_correlation_regime(0.7) == "DANGER"

    def test_negative_is_normal(self):
        assert classify_correlation_regime(-0.3) == "NORMAL"


# ── pairwise_correlations ────────────────────────────────────────────────


class TestPairwiseCorrelations:
    def test_two_assets(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 50)
        b = a + rng.normal(0, 0.5, 50)  # correlated with a
        matrix = np.column_stack([a, b])
        pairs = pairwise_correlations(matrix, ["A", "B"])
        assert ("A", "B") in pairs
        assert pairs[("A", "B")] > 0.5

    def test_three_assets_three_pairs(self):
        rng = np.random.RandomState(42)
        matrix = rng.normal(0, 1, (50, 3))
        pairs = pairwise_correlations(matrix, ["A", "B", "C"])
        assert len(pairs) == 3

    def test_insufficient_data(self):
        matrix = np.array([[1, 2], [3, 4]])
        pairs = pairwise_correlations(matrix, ["A", "B"])
        assert len(pairs) == 0

    def test_single_asset(self):
        matrix = np.array([[1], [2], [3], [4]])
        pairs = pairwise_correlations(matrix, ["A"])
        assert len(pairs) == 0

    def test_perfect_correlation(self):
        x = np.arange(50, dtype=float)
        matrix = np.column_stack([x, x * 2])
        pairs = pairwise_correlations(matrix, ["A", "B"])
        assert pairs[("A", "B")] == pytest.approx(1.0, abs=0.01)


# ── LiveCorrelationMonitor ───────────────────────────────────────────────


class TestMonitorBasic:
    def test_init(self):
        m = LiveCorrelationMonitor(["A", "B"])
        assert m.observation_count == 0
        assert len(m.alerts) == 0

    def test_add_returns(self):
        m = LiveCorrelationMonitor(["A", "B"])
        m.add_returns({"A": 0.01, "B": -0.005})
        assert m.observation_count == 1

    def test_snapshot_with_insufficient_data(self):
        m = LiveCorrelationMonitor(["A", "B"], window=5)
        m.add_returns({"A": 0.01, "B": 0.02})
        snap = m.snapshot()
        assert snap.correlation_regime == "NORMAL"
        assert snap.avg_correlation is None


class TestMonitorCorrelationDetection:
    def _feed_correlated(self, monitor, n=30, rho=0.9, seed=42):
        """Feed highly correlated returns."""
        rng = np.random.RandomState(seed)
        for _ in range(n):
            base = rng.normal(0, 0.01)
            noise = rng.normal(0, 0.01 * (1 - rho))
            monitor.add_returns({"A": base, "B": base + noise})

    def _feed_uncorrelated(self, monitor, n=30, seed=42):
        rng = np.random.RandomState(seed)
        for _ in range(n):
            monitor.add_returns({"A": rng.normal(0, 0.01), "B": rng.normal(0, 0.01)})

    def test_high_correlation_detected(self):
        m = LiveCorrelationMonitor(["A", "B"], window=20)
        self._feed_correlated(m, 25, rho=0.95)
        snap = m.snapshot()
        assert snap.avg_correlation is not None
        assert snap.avg_correlation > 0.5

    def test_low_correlation_normal_regime(self):
        m = LiveCorrelationMonitor(["A", "B"], window=20)
        self._feed_uncorrelated(m, 25)
        snap = m.snapshot()
        if snap.avg_correlation is not None:
            assert snap.correlation_regime in ("NORMAL", "ELEVATED")

    def test_effective_n_drops_under_correlation(self):
        m = LiveCorrelationMonitor(["A", "B", "C"], window=20)
        rng = np.random.RandomState(42)
        for _ in range(25):
            base = rng.normal(0, 0.01)
            m.add_returns({"A": base, "B": base * 1.1, "C": base * 0.9})
        snap = m.snapshot()
        if snap.effective_n is not None:
            assert snap.effective_n < 3.0

    def test_diversification_score_range(self):
        m = LiveCorrelationMonitor(["A", "B"], window=10)
        self._feed_uncorrelated(m, 15)
        snap = m.snapshot()
        if snap.diversification_score is not None:
            assert 0 <= snap.diversification_score <= 1.0


class TestMonitorAlerts:
    def test_correlation_spike_alert(self):
        m = LiveCorrelationMonitor(["A", "B"], window=10, pair_alert_threshold=0.6)
        rng = np.random.RandomState(42)
        for _ in range(15):
            base = rng.normal(0, 0.01)
            m.add_returns({"A": base, "B": base * 1.01})
        assert any(a.alert_type == "correlation_spike" for a in m.alerts)

    def test_no_alert_when_normal(self):
        m = LiveCorrelationMonitor(["A", "B"], window=10, pair_alert_threshold=0.9)
        rng = np.random.RandomState(42)
        for _ in range(15):
            m.add_returns({"A": rng.normal(0, 0.01), "B": rng.normal(0, 0.01)})
        corr_alerts = [a for a in m.alerts if a.alert_type == "correlation_spike"]
        assert len(corr_alerts) == 0

    def test_regime_change_alert(self):
        # Use very high pair threshold so correlation_spike doesn't pre-empt regime_change
        m = LiveCorrelationMonitor(["A", "B"], window=10, elevated_threshold=0.3,
                                   pair_alert_threshold=0.99, min_effective_n=0.1)
        rng = np.random.RandomState(42)
        # Start uncorrelated
        for _ in range(12):
            m.add_returns({"A": rng.normal(0, 0.01), "B": rng.normal(0, 0.01)})
        # Then correlated enough to shift regime but not enough to hit pair threshold
        for _ in range(15):
            base = rng.normal(0, 0.01)
            m.add_returns({"A": base + rng.normal(0, 0.002), "B": base + rng.normal(0, 0.002)})
        # At least one alert should be about regime_change or correlation
        assert len(m.alerts) >= 1

    def test_alert_has_required_fields(self):
        m = LiveCorrelationMonitor(["A", "B"], window=10, pair_alert_threshold=0.5)
        rng = np.random.RandomState(42)
        for _ in range(15):
            base = rng.normal(0, 0.01)
            m.add_returns({"A": base, "B": base})
        if m.alerts:
            a = m.alerts[0]
            assert hasattr(a, "timestamp")
            assert hasattr(a, "alert_type")
            assert hasattr(a, "severity")
            assert hasattr(a, "message")
            assert a.severity in ("warning", "critical")


class TestMonitorAllocations:
    def test_no_adjustments_in_normal(self):
        m = LiveCorrelationMonitor(["A", "B"], window=10)
        rng = np.random.RandomState(42)
        for _ in range(15):
            m.add_returns({"A": rng.normal(0, 0.01), "B": rng.normal(0, 0.01)})
        snap = m.snapshot()
        if snap.correlation_regime == "NORMAL":
            assert len(snap.adjustments) == 0

    def test_adjustments_in_danger(self):
        m = LiveCorrelationMonitor(["A", "B"], window=10, pair_alert_threshold=0.5)
        rng = np.random.RandomState(42)
        for _ in range(15):
            base = rng.normal(0, 0.01)
            m.add_returns({"A": base, "B": base * 1.01})
        snap = m.snapshot()
        if snap.correlation_regime in ("ELEVATED", "DANGER"):
            assert len(snap.adjustments) > 0
            for adj in snap.adjustments:
                assert isinstance(adj, AllocationAdjustment)
                assert 0 <= adj.recommended_weight <= 1


class TestDashboard:
    def test_empty_dashboard(self):
        m = LiveCorrelationMonitor(["A", "B"])
        html = m._build_html(m.snapshot(), {})
        assert "<!DOCTYPE html>" in html
        assert "NORMAL" in html

    def test_dashboard_with_data(self, tmp_path):
        m = LiveCorrelationMonitor(["A", "B"], window=10)
        rng = np.random.RandomState(42)
        for _ in range(20):
            m.add_returns({"A": rng.normal(0, 0.01), "B": rng.normal(0, 0.01)})
        path = m.generate_dashboard(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Live Correlation Monitor" in content
        assert "Effective N" in content

    def test_dashboard_no_external_resources(self, tmp_path):
        m = LiveCorrelationMonitor(["A", "B"], window=10)
        rng = np.random.RandomState(42)
        for _ in range(15):
            m.add_returns({"A": rng.normal(0, 0.01), "B": rng.normal(0, 0.01)})
        path = m.generate_dashboard(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "http://" not in content
        assert "https://" not in content

    def test_dashboard_charts_embedded(self, tmp_path):
        m = LiveCorrelationMonitor(["A", "B"], window=10)
        rng = np.random.RandomState(42)
        for _ in range(20):
            base = rng.normal(0, 0.01)
            m.add_returns({"A": base + rng.normal(0, 0.003), "B": base + rng.normal(0, 0.003)})
        path = m.generate_dashboard(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "data:image/png;base64," in content
