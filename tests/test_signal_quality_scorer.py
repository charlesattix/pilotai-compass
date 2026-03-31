"""Tests for compass/signal_quality_scorer.py"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from compass.signal_quality_scorer import (
    QualitySummary, SignalCorrelation, SignalMetrics, SignalQualityScorer,
)

def _make_data(n=300, signals=5, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    sig = pd.DataFrame({f"sig_{i}": rng.choice([-1, 0, 1], n, p=[0.3, 0.2, 0.5]) for i in range(signals)}, index=dates)
    ret = pd.Series(rng.normal(0.0003, 0.01, n), index=dates)
    return sig, ret

def _make_scorer(n=300, signals=5, seed=42, **kw):
    sig, ret = _make_data(n, signals, seed)
    return SignalQualityScorer(sig, ret, **kw)

class TestDataclasses:
    def test_signal_metrics(self):
        m = SignalMetrics("s", 0.05, 0.5, 0.55, 1.3, 0.01, -0.008, 1.25, 1.5, 0.3, 10, 70, "B", False)
        assert m.grade == "B"
    def test_correlation(self):
        c = SignalCorrelation("a", "b", 0.5)
        assert c.correlation == pytest.approx(0.5)
    def test_summary(self):
        s = QualitySummary(5, 60, 1, {"B": 3}, "best", "worst")
        assert s.avg_score == pytest.approx(60)

class TestMetrics:
    def test_ic_range(self):
        s = _make_scorer(); s.analyze()
        for m in s.metrics:
            assert -1 <= m.ic <= 1
    def test_hit_rate_range(self):
        s = _make_scorer(); s.analyze()
        for m in s.metrics:
            assert 0 <= m.hit_rate <= 1
    def test_quality_score_range(self):
        s = _make_scorer(); s.analyze()
        for m in s.metrics:
            assert 0 <= m.quality_score <= 100
    def test_grade_valid(self):
        s = _make_scorer(); s.analyze()
        for m in s.metrics:
            assert m.grade in ("A", "B", "C", "D", "F")
    def test_turnover_range(self):
        s = _make_scorer(); s.analyze()
        for m in s.metrics:
            assert 0 <= m.turnover <= 1
    def test_all_signals_scored(self):
        s = _make_scorer(signals=4); s.analyze()
        assert len(s.metrics) == 4
    def test_profit_factor_positive(self):
        s = _make_scorer(); s.analyze()
        for m in s.metrics:
            assert m.profit_factor >= 0
    def test_stale_detection(self):
        """Constant signal should be detected as stale."""
        dates = pd.bdate_range("2024-01-01", periods=100)
        sig = pd.DataFrame({"flat": np.ones(100)}, index=dates)
        ret = pd.Series(np.random.randn(100) * 0.01, index=dates)
        s = SignalQualityScorer(sig, ret)
        s.analyze()
        assert s.metrics[0].is_stale is True
        assert s.metrics[0].quality_score == 0

class TestCorrelations:
    def test_correlations_computed(self):
        s = _make_scorer(signals=4); s.analyze()
        expected = 4 * 3 // 2
        assert len(s.correlations) == expected
    def test_sorted_by_abs_corr(self):
        s = _make_scorer(); s.analyze()
        abs_corrs = [abs(c.correlation) for c in s.correlations]
        assert abs_corrs == sorted(abs_corrs, reverse=True)
    def test_correlation_range(self):
        s = _make_scorer(); s.analyze()
        for c in s.correlations:
            assert -1 <= c.correlation <= 1

class TestSummary:
    def test_summary_populated(self):
        s = _make_scorer(); s.analyze()
        assert s.summary is not None
    def test_n_signals(self):
        s = _make_scorer(signals=6); s.analyze()
        assert s.summary.n_signals == 6
    def test_avg_score_range(self):
        s = _make_scorer(); s.analyze()
        assert 0 <= s.summary.avg_score <= 100
    def test_best_worst(self):
        s = _make_scorer(); s.analyze()
        assert len(s.summary.best_signal) > 0
        assert len(s.summary.worst_signal) > 0
    def test_grade_distribution_sums(self):
        s = _make_scorer(); s.analyze()
        total = sum(s.summary.grade_distribution.values())
        assert total == s.summary.n_signals

class TestShortData:
    def test_few_points(self):
        sig, ret = _make_data(n=5, signals=2)
        s = SignalQualityScorer(sig, ret)
        s.analyze()
        assert len(s.metrics) == 2

class TestPipeline:
    def test_keys(self):
        s = _make_scorer()
        result = s.analyze()
        assert {"metrics", "correlations", "summary"} == set(result.keys())
    def test_from_csv(self, tmp_path):
        sig, ret = _make_data()
        sig.to_csv(tmp_path / "s.csv"); ret.to_frame().to_csv(tmp_path / "r.csv")
        s = SignalQualityScorer.from_csv(str(tmp_path / "s.csv"), str(tmp_path / "r.csv"))
        s.analyze()
        assert s.summary is not None

class TestReport:
    def test_html(self, tmp_path):
        s = _make_scorer()
        path = s.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Signal Quality" in c
    def test_sections(self, tmp_path):
        s = _make_scorer()
        path = s.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Scorecard" in c and "Correlation" in c and "Grade" in c
    def test_charts(self, tmp_path):
        s = _make_scorer()
        path = s.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()
    def test_auto_analyze(self, tmp_path):
        s = _make_scorer()
        assert s.summary is None
        s.generate_report(str(tmp_path / "r.html"))
        assert s.summary is not None
    def test_default_path(self):
        s = _make_scorer()
        path = s.generate_report()
        assert "signal_quality.html" in path
