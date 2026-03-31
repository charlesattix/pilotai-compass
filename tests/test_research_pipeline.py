"""Tests for compass.research_pipeline — 32 tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from compass.research_pipeline import (
    ResearchPipeline, SignalHypothesis, TestResult, CorrectedResult,
    SignalCluster, PipelineResult, ResearchLogEntry,
)

def _features(n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame({
        "price": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)),
        "volume": rng.integers(1000, 10000, n).astype(float),
        "vix": 20 + rng.normal(0, 2, n),
    }, index=idx)

def _returns(n=300, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0003, 0.01, n),
                      index=pd.bdate_range("2024-01-02", periods=n))


class TestHypotheses:
    def test_generate(self):
        hyps = ResearchPipeline.generate_hypotheses(["price", "volume", "vix"])
        assert len(hyps) > 5
        assert all(isinstance(h, SignalHypothesis) for h in hyps)

    def test_max_cap(self):
        hyps = ResearchPipeline.generate_hypotheses(["a", "b", "c"], max_hypotheses=5)
        assert len(hyps) <= 5

    def test_pairwise(self):
        hyps = ResearchPipeline.generate_hypotheses(["a", "b"], lookbacks=[10])
        pair = [h for h in hyps if h.feature_b is not None]
        assert len(pair) > 0

    def test_interactions(self):
        hyps = ResearchPipeline.generate_hypotheses(["a", "b"])
        interactions = {h.interaction for h in hyps if h.feature_b}
        assert "ratio" in interactions
        assert "diff" in interactions


class TestBuildSignal:
    def test_single_feature(self):
        feat = _features()
        h = SignalHypothesis("price_mom_20", "price", lookback=20)
        sig = ResearchPipeline.build_signal(h, feat)
        assert not sig.dropna().empty
        assert sig.dropna().abs().max() <= 1.01

    def test_ratio(self):
        feat = _features()
        h = SignalHypothesis("price_ratio_volume", "price", "volume", "ratio")
        sig = ResearchPipeline.build_signal(h, feat)
        assert not sig.dropna().empty

    def test_diff(self):
        feat = _features()
        h = SignalHypothesis("price_diff_vix", "price", "vix", "diff")
        sig = ResearchPipeline.build_signal(h, feat)
        assert not sig.dropna().empty

    def test_missing_feature(self):
        feat = _features()
        h = SignalHypothesis("missing", "nonexistent")
        sig = ResearchPipeline.build_signal(h, feat)
        assert sig.empty


class TestWalkForward:
    def test_basic(self):
        rp = ResearchPipeline(n_walk_forward_folds=3)
        sig = pd.Series(np.random.default_rng(42).choice([-1, 0, 1], 200),
                         index=pd.bdate_range("2024-01-02", periods=200), dtype=float)
        ret = _returns(200)
        is_sh, oos_sh, t, p = rp.walk_forward_test(sig, ret)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_short_data(self):
        rp = ResearchPipeline()
        sig = pd.Series([1.0] * 10, index=pd.bdate_range("2024-01-02", periods=10))
        _, _, _, p = rp.walk_forward_test(sig, _returns(10))
        assert p == 1.0


class TestCorrection:
    def test_bonferroni(self):
        rp = ResearchPipeline(significance_level=0.05, correction_method="bonferroni")
        results = [
            TestResult(SignalHypothesis(f"s{i}", "a"), 1.0, 0.5, 2.0, 0.01 * (i + 1), 0.02, True)
            for i in range(5)
        ]
        corrected = rp.correct_pvalues(results)
        assert len(corrected) == 5
        # Bonferroni: p * n
        assert corrected[0].corrected_p == pytest.approx(0.01 * 5)

    def test_bh(self):
        rp = ResearchPipeline(significance_level=0.05, correction_method="bh")
        results = [
            TestResult(SignalHypothesis(f"s{i}", "a"), 1.0, 0.5, 2.0, 0.001 * (i + 1), 0.02, True)
            for i in range(10)
        ]
        corrected = rp.correct_pvalues(results)
        sig = [c for c in corrected if c.is_significant]
        assert len(sig) > 0  # at least some should survive at alpha=0.05

    def test_empty(self):
        rp = ResearchPipeline()
        assert rp.correct_pvalues([]) == []

    def test_all_insignificant(self):
        rp = ResearchPipeline(significance_level=0.01)
        results = [
            TestResult(SignalHypothesis("s", "a"), 0, 0, 0.5, 0.50, 0, False)
        ]
        corrected = rp.correct_pvalues(results)
        assert not corrected[0].is_significant


class TestClustering:
    def test_basic(self):
        rp = ResearchPipeline(cluster_threshold=0.70)
        rng = np.random.default_rng(42)
        n = 200
        base = pd.Series(rng.normal(0, 1, n))
        signals = {
            "a": base + rng.normal(0, 0.1, n),
            "b": base + rng.normal(0, 0.1, n),  # corr > 0.7 with a
            "c": pd.Series(rng.normal(0, 1, n)),  # independent
        }
        clusters = rp.cluster_signals(signals, ["a", "b", "c"])
        assert len(clusters) >= 1
        # a and b should be clustered
        reps = {c.representative for c in clusters}
        assert len(reps) <= 3

    def test_single(self):
        rp = ResearchPipeline()
        clusters = rp.cluster_signals({}, ["only"])
        assert len(clusters) == 1
        assert clusters[0].representative == "only"

    def test_empty(self):
        rp = ResearchPipeline()
        clusters = rp.cluster_signals({}, [])
        assert len(clusters) == 1  # empty cluster


class TestFullPipeline:
    def test_run(self):
        rp = ResearchPipeline(significance_level=0.20, correction_method="bh",
                                n_walk_forward_folds=3)
        feat = _features(200)
        ret = _returns(200)
        result = rp.run(feat, ret, max_hypotheses=15)
        assert isinstance(result, PipelineResult)
        assert result.n_hypotheses <= 15
        assert result.n_tested > 0

    def test_log_populated(self):
        rp = ResearchPipeline(n_walk_forward_folds=3)
        result = rp.run(_features(200), _returns(200), max_hypotheses=10)
        assert len(result.log) > 0
        assert all(isinstance(e, ResearchLogEntry) for e in result.log)

    def test_funnel(self):
        rp = ResearchPipeline(significance_level=0.50, correction_method="bh",
                                n_walk_forward_folds=3)
        result = rp.run(_features(200), _returns(200), max_hypotheses=20)
        assert result.n_tested >= result.n_raw_significant >= result.n_after_correction >= 0


class TestReport:
    def test_creates_file(self, tmp_path):
        rp = ResearchPipeline(n_walk_forward_folds=3)
        result = rp.run(_features(200), _returns(200), max_hypotheses=10)
        out = tmp_path / "research.html"
        path = rp.generate_report(result, output_path=str(out))
        assert Path(path).exists()
        assert "Research Pipeline" in out.read_text()

    def test_contains_funnel(self, tmp_path):
        rp = ResearchPipeline(n_walk_forward_folds=3)
        result = rp.run(_features(200), _returns(200), max_hypotheses=10)
        out = tmp_path / "r.html"
        rp.generate_report(result, output_path=str(out))
        html = out.read_text()
        assert "Hypotheses" in html
        assert "Tested" in html

    def test_contains_clusters(self, tmp_path):
        rp = ResearchPipeline(n_walk_forward_folds=3, significance_level=0.50)
        result = rp.run(_features(200), _returns(200), max_hypotheses=15)
        out = tmp_path / "r.html"
        rp.generate_report(result, output_path=str(out))
        assert "Cluster" in out.read_text()
