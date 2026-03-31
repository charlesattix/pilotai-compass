"""Tests for compass.drawdown_recovery – drawdown recovery prediction engine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.drawdown_recovery import (
    DrawdownEpisode,
    DrawdownRecoveryPredictor,
    EpisodeCluster,
    RecoveryResult,
    RegimeRecovery,
    SizingAdjustment,
    SurvivalPoint,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_returns(n: int = 600, seed: int = 42, vol: float = 0.015) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    rets = rng.randn(n) * vol + 0.0003
    # Inject drawdowns at known positions
    rets[100:115] = -0.02    # ~25% DD
    rets[115:145] = 0.012    # recovery
    rets[250:260] = -0.015   # ~14% DD
    rets[260:285] = 0.008    # recovery
    rets[400:420] = -0.025   # ~40% DD
    rets[420:470] = 0.015    # recovery
    return pd.Series(rets, index=idx)


def _make_regimes(n: int = 600) -> pd.Series:
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    labels = ["bull"] * 200 + ["bear"] * 200 + ["high_vol"] * 200
    return pd.Series(labels, index=idx)


def _make_flat_returns(n: int = 100) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series([0.001] * n, index=idx)


def _make_deep_dd_returns(n: int = 300, seed: int = 11) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rets = rng.randn(n) * 0.005 + 0.001
    # Deep drawdown that doesn't recover
    rets[50:90] = -0.03
    return pd.Series(rets, index=idx)


# ── Constructor ─────────────────────────────────────────────────────────────
class TestInit:
    def test_defaults(self):
        p = DrawdownRecoveryPredictor()
        assert p.min_dd_depth == 0.01
        assert p.n_clusters == 3
        assert p.sizing_floor == 0.25

    def test_custom(self):
        p = DrawdownRecoveryPredictor(min_dd_depth=0.05, n_clusters=5, sizing_floor=0.10)
        assert p.min_dd_depth == 0.05
        assert p.n_clusters == 5


# ── Core analysis ───────────────────────────────────────────────────────────
class TestAnalyze:
    def test_returns_result(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert isinstance(result, RecoveryResult)

    def test_episodes_detected(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert result.n_episodes > 0

    def test_generated_at(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert len(result.generated_at) > 0

    def test_too_few_bars(self):
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        result = DrawdownRecoveryPredictor().analyze(pd.Series([0.01] * 10, index=idx))
        assert result.n_episodes == 0

    def test_flat_returns_no_episodes(self):
        result = DrawdownRecoveryPredictor().analyze(_make_flat_returns())
        assert result.n_episodes == 0


# ── Episodes ────────────────────────────────────────────────────────────────
class TestEpisodes:
    def test_depth_positive(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        for e in result.episodes:
            assert e.depth > 0

    def test_recovery_days_nonneg(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        for e in result.episodes:
            assert e.recovery_days >= 0

    def test_regime_assigned(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns(), regimes=_make_regimes())
        regimes_seen = {e.regime for e in result.episodes if e.regime}
        assert len(regimes_seen) > 0

    def test_min_depth_filters(self):
        low = DrawdownRecoveryPredictor(min_dd_depth=0.001).analyze(_make_returns())
        high = DrawdownRecoveryPredictor(min_dd_depth=0.10).analyze(_make_returns())
        assert high.n_episodes <= low.n_episodes

    def test_overall_recovery_rate_bounded(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert 0.0 <= result.overall_recovery_rate <= 1.0


# ── Kaplan-Meier survival ──────────────────────────────────────────────────
class TestSurvivalCurve:
    def test_curve_present(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert len(result.survival_curve) > 0

    def test_starts_at_one(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert result.survival_curve[0].survival_prob == pytest.approx(1.0)
        assert result.survival_curve[0].time == 0

    def test_monotonically_decreasing(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        probs = [s.survival_prob for s in result.survival_curve]
        for i in range(1, len(probs)):
            assert probs[i] <= probs[i - 1] + 1e-9

    def test_bounded_zero_one(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        for s in result.survival_curve:
            assert 0.0 <= s.survival_prob <= 1.0

    def test_n_at_risk_decreases(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        risks = [s.n_at_risk for s in result.survival_curve]
        for i in range(1, len(risks)):
            assert risks[i] <= risks[i - 1]


# ── Regime-conditional ──────────────────────────────────────────────────────
class TestRegimeRecovery:
    def test_regime_results_present(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns(), regimes=_make_regimes())
        assert len(result.regime_recoveries) > 0

    def test_no_regimes_unknown(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        if result.regime_recoveries:
            assert any(rr.regime == "unknown" for rr in result.regime_recoveries)

    def test_regime_fields(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns(), regimes=_make_regimes())
        for rr in result.regime_recoveries:
            assert rr.n_episodes > 0
            assert rr.avg_depth > 0
            assert 0.0 <= rr.recovery_rate <= 1.0

    def test_recovery_rate_bounded(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns(), regimes=_make_regimes())
        for rr in result.regime_recoveries:
            assert 0.0 <= rr.recovery_rate <= 1.0


# ── Clustering ──────────────────────────────────────────────────────────────
class TestClustering:
    def test_clusters_present(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert len(result.clusters) > 0

    def test_cluster_fields(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        for c in result.clusters:
            assert c.n_episodes > 0
            assert c.avg_depth > 0
            assert c.depth_range[0] <= c.depth_range[1]

    def test_clusters_cover_all_episodes(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        total_in_clusters = sum(c.n_episodes for c in result.clusters)
        assert total_in_clusters == result.n_episodes

    def test_few_episodes_no_clusters(self):
        # Very high min_depth → few episodes → no clusters possible
        result = DrawdownRecoveryPredictor(min_dd_depth=0.50, n_clusters=5).analyze(_make_returns())
        assert result.clusters == []


# ── Conditional recovery ────────────────────────────────────────────────────
class TestConditionalRecovery:
    def test_buckets_present(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert len(result.conditional_recovery) > 0

    def test_expected_keys(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert "0-5%" in result.conditional_recovery
        assert "20%+" in result.conditional_recovery

    def test_values_nonnegative(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        for v in result.conditional_recovery.values():
            assert v >= 0


# ── Sizing adjustment ──────────────────────────────────────────────────────
class TestSizingAdjustment:
    def test_sizing_present_when_depth_given(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns(), current_depth=0.10)
        assert result.sizing_adjustment is not None

    def test_no_sizing_without_depth(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns())
        assert result.sizing_adjustment is None

    def test_full_size_below_threshold(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns(), current_depth=0.01)
        assert result.sizing_adjustment.recommended_scale == 1.0

    def test_floor_at_deep_dd(self):
        p = DrawdownRecoveryPredictor(sizing_floor=0.25, sizing_dd_full=0.25)
        result = p.analyze(_make_returns(), current_depth=0.30)
        assert result.sizing_adjustment.recommended_scale == 0.25

    def test_intermediate_sizing(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns(), current_depth=0.15)
        s = result.sizing_adjustment
        assert 0.25 < s.recommended_scale < 1.0

    def test_reasoning_present(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns(), current_depth=0.10)
        assert len(result.sizing_adjustment.reasoning) > 0

    def test_recovery_probability_bounded(self):
        result = DrawdownRecoveryPredictor().analyze(_make_returns(), current_depth=0.10)
        assert 0.0 <= result.sizing_adjustment.recovery_probability <= 1.0


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = DrawdownRecoveryPredictor()
            result = p.analyze(_make_returns(), regimes=_make_regimes(), current_depth=0.10)
            path = p.generate_report(result, output_path=Path(tmp) / "dr.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = DrawdownRecoveryPredictor()
            result = p.analyze(_make_returns(), regimes=_make_regimes(), current_depth=0.10)
            path = p.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Drawdown Recovery" in html
            assert "Survival" in html
            assert "Waterfall" in html
            assert "Heatmap" in html
            assert "Regime" in html
            assert "Cluster" in html
            assert "Sizing" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = DrawdownRecoveryPredictor()
            result = p.analyze(_make_returns())
            path = p.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html

    def test_report_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = DrawdownRecoveryPredictor()
            result = RecoveryResult(generated_at="2024-01-01T00:00:00+00:00")
            path = p.generate_report(result, output_path=Path(tmp) / "e.html")
            assert path.exists()


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_episode(self):
        e = DrawdownEpisode(0, 5, 10, 20, 0.15, 5, 10, 15, True, "bull")
        assert e.depth == 0.15

    def test_survival_point(self):
        s = SurvivalPoint(10, 0.8, 50, 3)
        assert s.survival_prob == 0.8

    def test_regime_recovery(self):
        rr = RegimeRecovery("bear", 5, 0.12, 30.0, 25.0, 0.8, 0.02)
        assert rr.recovery_rate == 0.8

    def test_cluster(self):
        c = EpisodeCluster(0, 10, 0.08, 20.0, (0.05, 0.10), {"bull": 5, "bear": 5}, [0, 1])
        assert c.n_episodes == 10

    def test_sizing_adjustment(self):
        s = SizingAdjustment(0.10, 0.75, "moderate", 25, 0.85)
        assert s.recommended_scale == 0.75

    def test_recovery_result_defaults(self):
        r = RecoveryResult()
        assert r.episodes == []
        assert r.overall_recovery_rate == 0.0
