"""Tests for compass/data_quality.py — data quality monitoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from compass.data_quality import (
    AccuracyCheck, CompletenessCheck, ConsistencyCheck, DataQualityEngine,
    FreshnessCheck, QualityAlert, QualityScore, RepairAction,
)

def _make_clean(n=200, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 430 + rng.normal(0, 1, n).cumsum()
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    vol = rng.uniform(5000, 20000, n)
    return pd.DataFrame({"close": close, "high": high, "low": low, "volume": vol}, index=dates)

def _make_dirty(n=200, seed=42):
    df = _make_clean(n, seed)
    df.loc[df.index[10:15], "close"] = np.nan
    df.loc[df.index[50], "close"] = 9999  # outlier
    df.loc[df.index[100:106], "close"] = df.loc[df.index[99], "close"]  # stale
    df.loc[df.index[80], "high"] = df.loc[df.index[80], "low"] - 1  # high < low
    return df

def _make_engine(dirty=False, **kw):
    df = _make_dirty() if dirty else _make_clean()
    return DataQualityEngine(df, **kw)

class TestDataclasses:
    def test_completeness(self):
        c = CompletenessCheck("close", 200, 5, 0.025, 1, 5, True)
        assert c.missing == 5
    def test_accuracy(self):
        a = AccuracyCheck("close", 3, 1, 0, 4.0, True)
        assert a.outliers == 3
    def test_consistency(self):
        c = ConsistencyCheck("hl", "test", 0, True)
        assert c.passed
    def test_freshness(self):
        f = FreshnessCheck("src", "2024-01-01", 3600, 86400, True)
        assert f.is_fresh
    def test_score(self):
        s = QualityScore("src", 0.95, 0.90, 1.0, 1.0, 0.95, "A")
        assert s.grade == "A"
    def test_repair(self):
        r = RepairAction("close", "interpolate", 5, "desc")
        assert r.rows_affected == 5
    def test_alert(self):
        a = QualityAlert("critical", "comp", "msg", "close")
        assert a.severity == "critical"

class TestCompleteness:
    def test_clean_data_passes(self):
        e = _make_engine(dirty=False); e.analyze()
        for c in e.completeness:
            assert c.passed
    def test_dirty_data_detects_missing(self):
        e = _make_engine(dirty=True); e.analyze()
        close_check = [c for c in e.completeness if c.column == "close"][0]
        assert close_check.missing > 0
    def test_gap_detection(self):
        e = _make_engine(dirty=True); e.analyze()
        close_check = [c for c in e.completeness if c.column == "close"][0]
        assert close_check.gaps >= 1
    def test_longest_gap(self):
        e = _make_engine(dirty=True); e.analyze()
        close_check = [c for c in e.completeness if c.column == "close"][0]
        assert close_check.longest_gap >= 5

class TestAccuracy:
    def test_detects_outliers(self):
        e = _make_engine(dirty=True); e.analyze()
        close_acc = [a for a in e.accuracy if a.column == "close"][0]
        assert close_acc.outliers >= 1
    def test_detects_stale(self):
        e = _make_engine(dirty=True, stale_threshold=5); e.analyze()
        close_acc = [a for a in e.accuracy if a.column == "close"][0]
        assert close_acc.stale_count >= 1
    def test_clean_data_few_outliers(self):
        e = _make_engine(dirty=False); e.analyze()
        for a in e.accuracy:
            assert a.outliers < 5
    def test_z_threshold_configurable(self):
        e = _make_engine(dirty=True, z_threshold=2.0); e.analyze()
        close_acc = [a for a in e.accuracy if a.column == "close"][0]
        # Lower threshold → more outliers
        e2 = _make_engine(dirty=True, z_threshold=10.0); e2.analyze()
        close_acc2 = [a for a in e2.accuracy if a.column == "close"][0]
        assert close_acc.outliers >= close_acc2.outliers

class TestConsistency:
    def test_detects_high_low_violation(self):
        e = _make_engine(dirty=True); e.analyze()
        hl = [c for c in e.consistency if c.name == "high_low"][0]
        assert hl.violations >= 1
    def test_clean_passes_all(self):
        e = _make_engine(dirty=False); e.analyze()
        for c in e.consistency:
            assert c.passed
    def test_volume_positive(self):
        e = _make_engine(dirty=False); e.analyze()
        vol = [c for c in e.consistency if c.name == "volume_positive"]
        if vol:
            assert vol[0].passed
    def test_no_dup_index(self):
        e = _make_engine(dirty=False); e.analyze()
        dup = [c for c in e.consistency if c.name == "no_dup_index"]
        if dup:
            assert dup[0].passed

class TestScoring:
    def test_clean_gets_good_grade(self):
        e = _make_engine(dirty=False); e.analyze()
        assert e.scores[0].grade in ("A", "B")
    def test_dirty_gets_worse_grade(self):
        clean = _make_engine(dirty=False); clean.analyze()
        dirty = _make_engine(dirty=True); dirty.analyze()
        assert dirty.scores[0].overall <= clean.scores[0].overall
    def test_score_range(self):
        e = _make_engine(); e.analyze()
        s = e.scores[0]
        for v in [s.completeness, s.accuracy, s.consistency, s.overall]:
            assert 0 <= v <= 1

class TestAlerts:
    def test_dirty_generates_alerts(self):
        e = _make_engine(dirty=True); e.analyze()
        assert len(e.alerts) > 0
    def test_alert_severity(self):
        e = _make_engine(dirty=True); e.analyze()
        for a in e.alerts:
            assert a.severity in ("info", "warning", "critical")
    def test_clean_fewer_alerts(self):
        clean = _make_engine(dirty=False); clean.analyze()
        dirty = _make_engine(dirty=True); dirty.analyze()
        assert len(clean.alerts) <= len(dirty.alerts)

class TestRepair:
    def test_auto_repair_fills(self):
        e = _make_engine(dirty=True, auto_repair=True); e.analyze()
        assert len(e.repairs) > 0
    def test_repair_reduces_missing(self):
        e = _make_engine(dirty=True, auto_repair=True); e.analyze()
        for r in e.repairs:
            assert r.rows_affected > 0
    def test_no_repair_by_default(self):
        e = _make_engine(dirty=True); e.analyze()
        assert len(e.repairs) == 0

class TestPipeline:
    def test_analyze_keys(self):
        e = _make_engine()
        result = e.analyze()
        expected = {"completeness", "accuracy", "consistency", "freshness", "scores", "alerts", "repairs"}
        assert set(result.keys()) == expected
    def test_from_csv(self, tmp_path):
        df = _make_clean()
        csv = tmp_path / "data.csv"
        df.to_csv(csv)
        e = DataQualityEngine.from_csv(str(csv))
        e.analyze()
        assert len(e.scores) > 0

class TestReport:
    def test_html(self, tmp_path):
        e = _make_engine(dirty=True)
        path = e.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Data Quality" in c
    def test_sections(self, tmp_path):
        e = _make_engine(dirty=True)
        path = e.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Completeness" in c and "Accuracy" in c and "Consistency" in c
    def test_charts(self, tmp_path):
        e = _make_engine()
        path = e.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()
    def test_auto_analyze(self, tmp_path):
        e = _make_engine()
        assert not e.scores
        e.generate_report(str(tmp_path / "r.html"))
        assert len(e.scores) > 0
    def test_default_path(self):
        e = _make_engine()
        path = e.generate_report()
        assert "data_quality.html" in path
