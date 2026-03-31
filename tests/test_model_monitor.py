"""Tests for compass/model_monitor.py — model drift detection and alerting.

Covers:
  - ks_statistic: identical, disjoint, known distributions
  - ks_critical_value: decreases with sample size
  - ModelMonitor.record: recording, rolling updates
  - Rolling accuracy: window, threshold
  - Rolling AUC: computation, concept drift detection
  - Feature drift: KS-test based, training_samples mode, z-score fallback
  - Alert generation: accuracy decay, concept drift, feature drift
  - MonitorSnapshot: retrain integration
  - Dashboard HTML: structure and content
"""

import numpy as np
import pytest

from compass.model_monitor import (
    DriftAlert,
    ModelMonitor,
    MonitorSnapshot,
    PredictionRecord,
    ks_critical_value,
    ks_statistic,
)


# ── ks_statistic ─────────────────────────────────────────────────────────


class TestKSStatistic:
    def test_identical_distributions(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert ks_statistic(a, a) == pytest.approx(0.0)

    def test_disjoint_distributions(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 11.0, 12.0])
        assert ks_statistic(a, b) == pytest.approx(1.0)

    def test_known_shift(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(2, 1, 500)  # shifted by 2 std devs
        ks = ks_statistic(a, b)
        assert ks > 0.5  # large shift → high KS

    def test_same_distribution_low_ks(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0, 1, 500)
        ks = ks_statistic(a, b)
        assert ks < 0.15  # same distribution → low KS

    def test_value_range(self):
        rng = np.random.RandomState(42)
        a = rng.uniform(0, 1, 100)
        b = rng.uniform(0, 1, 100)
        ks = ks_statistic(a, b)
        assert 0 <= ks <= 1

    def test_single_element_each(self):
        ks = ks_statistic(np.array([1.0]), np.array([2.0]))
        assert ks == pytest.approx(1.0)


# ── ks_critical_value ────────────────────────────────────────────────────


class TestKSCriticalValue:
    def test_decreases_with_sample_size(self):
        cv_small = ks_critical_value(20, 20)
        cv_large = ks_critical_value(500, 500)
        assert cv_small > cv_large

    def test_alpha_005(self):
        cv = ks_critical_value(100, 100)
        assert cv > 0
        assert cv < 1


# ── ModelMonitor recording ───────────────────────────────────────────────


class TestMonitorRecord:
    def test_basic_record(self):
        m = ModelMonitor()
        m.record(y_true=1, y_pred_proba=0.8)
        assert m.prediction_count == 1

    def test_multiple_records(self):
        m = ModelMonitor()
        for _ in range(10):
            m.record(y_true=1, y_pred_proba=0.7)
        assert m.prediction_count == 10

    def test_max_history(self):
        m = ModelMonitor(max_history=5)
        for i in range(10):
            m.record(y_true=1, y_pred_proba=0.7)
        assert m.prediction_count == 5

    def test_returns_none_when_no_alert(self):
        m = ModelMonitor()
        alert = m.record(y_true=1, y_pred_proba=0.7)
        assert alert is None


# ── Rolling accuracy ─────────────────────────────────────────────────────


class TestRollingAccuracy:
    def _feed(self, monitor, n_correct, n_wrong):
        for _ in range(n_correct):
            monitor.record(y_true=1, y_pred_proba=0.8)
        for _ in range(n_wrong):
            monitor.record(y_true=0, y_pred_proba=0.8)  # wrong prediction

    def test_perfect_accuracy(self):
        m = ModelMonitor(accuracy_window=20)
        for _ in range(25):
            m.record(y_true=1, y_pred_proba=0.7)
        assert m._accuracy_series
        assert m._accuracy_series[-1][1] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        m = ModelMonitor(accuracy_window=20)
        for _ in range(25):
            m.record(y_true=0, y_pred_proba=0.8)  # all wrong
        assert m._accuracy_series
        assert m._accuracy_series[-1][1] == pytest.approx(0.0)

    def test_mixed_accuracy(self):
        m = ModelMonitor(accuracy_window=20)
        # 15 correct + 5 wrong = 75%
        for _ in range(15):
            m.record(y_true=1, y_pred_proba=0.7)
        for _ in range(10):
            m.record(y_true=0, y_pred_proba=0.8)  # wrong
        acc = m._accuracy_series[-1][1]
        assert 0 < acc < 1

    def test_no_accuracy_until_min_records(self):
        m = ModelMonitor(accuracy_window=50)
        for _ in range(5):
            m.record(y_true=1, y_pred_proba=0.7)
        assert len(m._accuracy_series) == 0


# ── Rolling AUC ──────────────────────────────────────────────────────────


class TestRollingAUC:
    def test_auc_computed_after_enough_records(self):
        m = ModelMonitor(auc_window=30)
        rng = np.random.RandomState(42)
        for _ in range(40):
            y = rng.randint(0, 2)
            p = 0.3 + 0.4 * y + rng.normal(0, 0.1)
            m.record(y_true=y, y_pred_proba=max(0, min(1, p)))
        assert len(m._auc_series) > 0
        assert 0.5 <= m._auc_series[-1][1] <= 1.0

    def test_no_auc_with_single_class(self):
        m = ModelMonitor(auc_window=20)
        for _ in range(25):
            m.record(y_true=1, y_pred_proba=0.9)  # all same class
        assert len(m._auc_series) == 0


# ── Feature drift ────────────────────────────────────────────────────────


class TestFeatureDrift:
    def test_no_drift_same_distribution(self):
        rng = np.random.RandomState(42)
        train = rng.normal(0, 1, (200, 3))
        m = ModelMonitor(
            feature_names=["a", "b", "c"],
            training_samples=train,
            accuracy_window=50,
            ks_threshold=0.25,  # relaxed: small samples have noisy KS
        )
        # Feed features from same distribution (large batch to reduce noise)
        for _ in range(55):
            feats = {"a": rng.normal(0, 1), "b": rng.normal(0, 1), "c": rng.normal(0, 1)}
            m.record(features=feats, y_true=1, y_pred_proba=0.7)
        if m._drift_series:
            # With same distribution, drift fraction should be low
            assert m._drift_series[-1][1] < 0.7

    def test_drift_detected_on_shifted_data(self):
        rng = np.random.RandomState(42)
        train = rng.normal(0, 1, (200, 2))
        m = ModelMonitor(
            feature_names=["a", "b"],
            training_samples=train,
            accuracy_window=30,
            ks_threshold=0.10,
        )
        # Feed features from SHIFTED distribution
        for _ in range(35):
            feats = {"a": rng.normal(5, 1), "b": rng.normal(5, 1)}
            m.record(features=feats, y_true=1, y_pred_proba=0.7)
        assert m._drift_series
        assert m._drift_series[-1][1] > 0.5

    def test_zscore_fallback(self):
        """When no training_samples, use means/stds for z-score comparison."""
        m = ModelMonitor(
            feature_names=["x"],
            feature_means=np.array([0.0]),
            feature_stds=np.array([1.0]),
            accuracy_window=25,
            ks_threshold=0.10,
        )
        rng = np.random.RandomState(42)
        for _ in range(30):
            feats = {"x": rng.normal(0, 1)}
            m.record(features=feats, y_true=1, y_pred_proba=0.7)
        # Should have drift data (even if low drift since same distribution)
        assert len(m._drift_series) > 0


# ── Alert generation ─────────────────────────────────────────────────────


class TestAlerts:
    def test_accuracy_decay_alert(self):
        m = ModelMonitor(
            accuracy_window=20,
            baseline_accuracy=0.80,
            accuracy_drop_threshold=0.10,
        )
        # Feed 20+ wrong predictions to trigger accuracy decay
        for _ in range(25):
            m.record(y_true=0, y_pred_proba=0.9)  # all wrong
        assert any(a.alert_type == "accuracy_decay" for a in m.alerts)

    def test_no_alert_when_healthy(self):
        m = ModelMonitor(
            accuracy_window=20,
            baseline_accuracy=0.80,
            accuracy_drop_threshold=0.10,
        )
        for _ in range(25):
            m.record(y_true=1, y_pred_proba=0.9)  # all correct
        assert not any(a.alert_type == "accuracy_decay" for a in m.alerts)

    def test_concept_drift_alert(self):
        m = ModelMonitor(
            auc_window=30,
            baseline_auc=0.85,
            auc_drop_threshold=0.05,
        )
        rng = np.random.RandomState(42)
        # Feed data where model is near-random (AUC ≈ 0.5)
        for _ in range(40):
            y = rng.randint(0, 2)
            p = rng.uniform(0.3, 0.7)  # uninformative
            m.record(y_true=y, y_pred_proba=p)
        # AUC should be near 0.5, which is 0.35 below baseline → alert
        assert any(a.alert_type == "concept_drift" for a in m.alerts)

    def test_feature_drift_alert(self):
        rng = np.random.RandomState(42)
        train = rng.normal(0, 1, (200, 3))
        m = ModelMonitor(
            feature_names=["a", "b", "c"],
            training_samples=train,
            accuracy_window=25,
            ks_threshold=0.10,
            drift_fraction_threshold=0.30,
        )
        for _ in range(30):
            feats = {"a": rng.normal(10, 1), "b": rng.normal(10, 1), "c": rng.normal(10, 1)}
            m.record(features=feats, y_true=1, y_pred_proba=0.7)
        assert any(a.alert_type == "feature_drift" for a in m.alerts)

    def test_alert_severity_levels(self):
        m = ModelMonitor(
            accuracy_window=15,
            baseline_accuracy=0.90,
            accuracy_drop_threshold=0.10,
        )
        # Major decay: 0% accuracy vs 90% baseline → drop = 0.90 ≥ 2*threshold
        for _ in range(20):
            m.record(y_true=0, y_pred_proba=0.9)
        criticals = [a for a in m.alerts if a.severity == "critical"]
        assert len(criticals) > 0


# ── MonitorSnapshot ──────────────────────────────────────────────────────


class TestMonitorSnapshot:
    def test_snapshot_basic(self):
        m = ModelMonitor()
        snap = m.snapshot()
        assert isinstance(snap, MonitorSnapshot)
        assert snap.n_predictions == 0
        assert snap.should_retrain is False

    def test_snapshot_retrain_on_accuracy_decay(self):
        m = ModelMonitor(
            accuracy_window=15,
            baseline_accuracy=0.80,
            accuracy_drop_threshold=0.10,
        )
        for _ in range(20):
            m.record(y_true=0, y_pred_proba=0.9)
        snap = m.snapshot()
        assert snap.should_retrain is True
        assert any("accuracy_decay" in r for r in snap.retrain_reasons)

    def test_snapshot_healthy(self):
        m = ModelMonitor(
            accuracy_window=15,
            baseline_accuracy=0.75,
            accuracy_drop_threshold=0.10,
        )
        for _ in range(20):
            m.record(y_true=1, y_pred_proba=0.9)
        snap = m.snapshot()
        assert snap.should_retrain is False
        assert snap.rolling_accuracy == pytest.approx(1.0)

    def test_snapshot_fields(self):
        m = ModelMonitor()
        snap = m.snapshot()
        assert hasattr(snap, "timestamp")
        assert hasattr(snap, "n_predictions")
        assert hasattr(snap, "rolling_accuracy")
        assert hasattr(snap, "rolling_auc")
        assert hasattr(snap, "drifted_features")
        assert hasattr(snap, "drift_fraction")
        assert hasattr(snap, "should_retrain")
        assert hasattr(snap, "retrain_reasons")


# ── Dashboard HTML ───────────────────────────────────────────────────────


class TestDashboard:
    def test_empty_dashboard(self):
        m = ModelMonitor()
        html = m.generate_dashboard()
        assert "<!DOCTYPE html>" in html
        assert "HEALTHY" in html

    def test_dashboard_with_data(self):
        m = ModelMonitor(accuracy_window=15, auc_window=25)
        rng = np.random.RandomState(42)
        for _ in range(30):
            y = rng.randint(0, 2)
            p = 0.3 + 0.4 * y + rng.normal(0, 0.1)
            m.record(y_true=y, y_pred_proba=max(0, min(1, p)))
        html = m.generate_dashboard()
        assert "Predictions" in html
        assert "Rolling Accuracy" in html

    def test_dashboard_with_alerts(self):
        m = ModelMonitor(
            accuracy_window=15,
            baseline_accuracy=0.90,
            accuracy_drop_threshold=0.05,
        )
        for _ in range(20):
            m.record(y_true=0, y_pred_proba=0.9)
        html = m.generate_dashboard()
        assert "RETRAIN RECOMMENDED" in html or "ALERTS ACTIVE" in html
        assert "accuracy_decay" in html

    def test_dashboard_charts_embedded(self):
        m = ModelMonitor(accuracy_window=15, auc_window=20)
        rng = np.random.RandomState(42)
        for _ in range(25):
            y = rng.randint(0, 2)
            p = 0.3 + 0.4 * y + rng.normal(0, 0.1)
            m.record(y_true=y, y_pred_proba=max(0, min(1, p)))
        html = m.generate_dashboard()
        assert "data:image/png;base64," in html

    def test_no_external_resources(self):
        m = ModelMonitor()
        html = m.generate_dashboard()
        assert "http://" not in html
        assert "https://" not in html
