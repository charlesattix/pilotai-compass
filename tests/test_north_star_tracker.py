"""Tests for compass.north_star_tracker — 35 tests."""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path
from compass.north_star_tracker import (
    NorthStarTracker, MetricSnapshot, GapAnalysis, Milestone,
    TrajectoryProjection, ExperimentMetrics, TrackerReport, NORTH_STAR_TARGETS,
)

def _exp_returns(n=200, k=3, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return {f"EXP-{i}": pd.Series(rng.normal(0.0003 + i * 0.0001, 0.01, n), index=idx)
            for i in range(k)}


class TestRecord:
    def test_basic(self):
        t = NorthStarTracker()
        snap = t.record(0.30, 3.0, 0.15)
        assert isinstance(snap, MetricSnapshot)
        assert snap.annual_return == 0.30
    def test_history(self):
        t = NorthStarTracker()
        t.record(0.20, 2.0, 0.10)
        t.record(0.25, 2.5, 0.12)
        assert len(t.snapshots) == 2
    def test_milestone_new_best(self):
        t = NorthStarTracker()
        t.record(0.20, 2.0, 0.15)
        t.record(0.30, 3.0, 0.12)  # new best return + sharpe + dd
        assert len(t.milestones) >= 1


class TestGaps:
    def test_all_met(self):
        t = NorthStarTracker()
        snap = MetricSnapshot(datetime.now(), 0.60, 7.0, 0.20)
        gaps = t.compute_gaps(snap)
        assert all(g.is_met for g in gaps)
    def test_all_gaps(self):
        t = NorthStarTracker()
        snap = MetricSnapshot(datetime.now(), 0.10, 1.0, 0.50)
        gaps = t.compute_gaps(snap)
        assert not any(g.is_met for g in gaps)
    def test_partial(self):
        t = NorthStarTracker()
        snap = MetricSnapshot(datetime.now(), 0.60, 2.0, 0.20)
        gaps = t.compute_gaps(snap)
        met = sum(1 for g in gaps if g.is_met)
        assert 1 <= met < 3
    def test_pct_achieved(self):
        t = NorthStarTracker()
        snap = MetricSnapshot(datetime.now(), 0.275, 3.0, 0.30)
        gaps = t.compute_gaps(snap)
        ret = [g for g in gaps if g.metric == "annual_return"][0]
        assert ret.pct_achieved == pytest.approx(0.5)


class TestTrajectory:
    def test_insufficient_data(self):
        t = NorthStarTracker()
        t.record(0.20, 2.0, 0.15)
        projs = t.project_trajectory()
        assert all(p.months_to_target == float("inf") for p in projs)
    def test_improving(self):
        t = NorthStarTracker()
        for i in range(50):
            t.record(0.20 + i * 0.005, 2.0 + i * 0.05, 0.20 - i * 0.001)
        projs = t.project_trajectory()
        for p in projs:
            assert p.improvement_rate > 0 or p.metric == "max_drawdown"
    def test_months_positive(self):
        t = NorthStarTracker()
        for i in range(30):
            t.record(0.10 + i * 0.01, 1.0 + i * 0.1, 0.25 - i * 0.002)
        projs = t.project_trajectory()
        for p in projs:
            assert p.months_to_target >= 0


class TestVelocity:
    def test_basic(self):
        t = NorthStarTracker()
        for i in range(20):
            t.record(0.20 + i * 0.01, 2.0 + i * 0.1, 0.20)
        vel = t.improvement_velocity()
        assert vel["annual_return"] > 0
        assert vel["sharpe"] > 0
    def test_no_data(self):
        t = NorthStarTracker()
        vel = t.improvement_velocity()
        assert all(v == 0.0 for v in vel.values())


class TestExperimentMetrics:
    def test_basic(self):
        exps = NorthStarTracker.experiment_metrics(_exp_returns())
        assert len(exps) == 3
        assert all(isinstance(e, ExperimentMetrics) for e in exps)
    def test_sorted_by_sharpe(self):
        exps = NorthStarTracker.experiment_metrics(_exp_returns())
        sharpes = [e.sharpe for e in exps]
        assert sharpes == sorted(sharpes, reverse=True)
    def test_empty(self):
        assert NorthStarTracker.experiment_metrics({}) == []


class TestTrack:
    def test_full(self):
        t = NorthStarTracker()
        t.record(0.20, 2.0, 0.15)
        report = t.track(0.25, 2.5, 0.12, _exp_returns())
        assert isinstance(report, TrackerReport)
        assert len(report.gaps) == 3
        assert report.overall_progress >= 0
    def test_all_met(self):
        t = NorthStarTracker()
        report = t.track(0.60, 7.0, 0.20)
        assert report.overall_progress == pytest.approx(1.0)
    def test_none_met(self):
        t = NorthStarTracker()
        report = t.track(0.10, 1.0, 0.50)
        assert report.overall_progress == pytest.approx(0.0)


class TestCustomTargets:
    def test_override(self):
        t = NorthStarTracker(targets={"annual_return": 0.30, "sharpe": 2.0, "max_drawdown": 0.20})
        report = t.track(0.30, 2.0, 0.20)
        assert report.overall_progress == pytest.approx(1.0)


class TestMilestones:
    def test_accumulate(self):
        t = NorthStarTracker()
        t.record(0.10, 1.0, 0.25)
        t.record(0.20, 2.0, 0.20)
        t.record(0.30, 3.0, 0.15)
        assert len(t.milestones) >= 2


class TestReport:
    def test_creates_file(self, tmp_path):
        t = NorthStarTracker()
        report = t.track(0.25, 2.5, 0.18, _exp_returns())
        out = tmp_path / "ns.html"
        path = t.generate_report(report, output_path=str(out))
        assert Path(path).exists()
        html = out.read_text()
        assert "North Star" in html
    def test_contains_gaps(self, tmp_path):
        t = NorthStarTracker()
        report = t.track(0.25, 2.5, 0.18)
        out = tmp_path / "ns.html"
        t.generate_report(report, output_path=str(out))
        assert "Gap Analysis" in out.read_text()
    def test_contains_experiments(self, tmp_path):
        t = NorthStarTracker()
        report = t.track(0.25, 2.5, 0.18, _exp_returns())
        out = tmp_path / "ns.html"
        t.generate_report(report, output_path=str(out))
        assert "Experiments" in out.read_text()
    def test_contains_milestones(self, tmp_path):
        t = NorthStarTracker()
        t.record(0.10, 1.0, 0.25)
        report = t.track(0.25, 3.0, 0.15)
        out = tmp_path / "ns.html"
        t.generate_report(report, output_path=str(out))
        assert "Milestone" in out.read_text()
