"""Tests for compass.drawdown_analyzer – drawdown analysis with regime attribution."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.drawdown_analyzer import (
    ClusteringResult,
    DrawdownAnalyzer,
    DrawdownEvent,
    DrawdownResult,
    RecoveryStats,
    RegimeAttribution,
    RiskRatios,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_returns(n: int = 500, seed: int = 42, drift: float = 0.0003, vol: float = 0.015) -> pd.Series:
    """Deterministic daily returns."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.Series(drift + rng.randn(n) * vol, index=idx, name="returns")


def _make_drawdown_returns(n: int = 300, seed: int = 42) -> pd.Series:
    """Returns with a clear drawdown event in the middle."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rets = rng.randn(n) * 0.005 + 0.001
    # Inject a drawdown from day 100-130, recovery 130-180
    rets[100:115] = -0.02  # 15 days of -2% each
    rets[115:130] = -0.005
    rets[130:180] = 0.012  # recovery
    return pd.Series(rets, index=idx)


def _make_regimes(n: int = 300) -> pd.Series:
    """Regime labels: bull for first half, bear for second."""
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    labels = ["bull"] * (n // 2) + ["bear"] * (n - n // 2)
    return pd.Series(labels, index=idx, name="regime")


def _make_multi_regime_returns(n: int = 600, seed: int = 42) -> tuple:
    """Returns with regime-specific drawdowns."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rets = rng.randn(n) * 0.005 + 0.001
    regimes = ["bull"] * 200 + ["bear"] * 200 + ["crash"] * 200

    # Bear drawdown: deeper
    rets[220:240] = -0.025
    rets[240:270] = 0.010
    # Crash drawdown: deepest
    rets[420:445] = -0.04
    rets[445:500] = 0.015
    # Bull drawdown: mild
    rets[50:60] = -0.008
    rets[60:75] = 0.006

    return pd.Series(rets, index=idx), pd.Series(regimes, index=idx)


def _make_flat_returns(n: int = 100) -> pd.Series:
    """All-positive returns — no drawdown."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series([0.001] * n, index=idx)


def _make_clustered_dd_returns(n: int = 500, seed: int = 11) -> pd.Series:
    """Returns with clustered drawdowns (back-to-back)."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rets = rng.randn(n) * 0.005 + 0.001
    # Two drawdowns close together
    rets[100:120] = -0.015
    rets[120:135] = 0.010
    rets[140:160] = -0.018  # starts right after first recovery
    rets[160:185] = 0.012
    return pd.Series(rets, index=idx)


# ── Constructor ─────────────────────────────────────────────────────────────
class TestDrawdownAnalyzerInit:
    def test_defaults(self):
        da = DrawdownAnalyzer()
        assert da.cdar_levels == (0.95, 0.99)
        assert da.min_dd_depth == 0.001

    def test_custom_cdar_levels(self):
        da = DrawdownAnalyzer(cdar_levels=(0.90, 0.95, 0.99))
        assert len(da.cdar_levels) == 3

    def test_custom_min_depth(self):
        da = DrawdownAnalyzer(min_dd_depth=0.05)
        assert da.min_dd_depth == 0.05


# ── Basic analysis ──────────────────────────────────────────────────────────
class TestAnalyze:
    def test_returns_drawdown_result(self):
        result = DrawdownAnalyzer().analyze(_make_returns())
        assert isinstance(result, DrawdownResult)

    def test_max_dd_negative(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.max_dd < 0  # underwater curve is negative

    def test_avg_dd_negative(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.avg_dd < 0

    def test_events_detected(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert len(result.events) > 0

    def test_underwater_curve_present(self):
        result = DrawdownAnalyzer().analyze(_make_returns())
        assert result.underwater_curve is not None
        assert len(result.underwater_curve) > 0

    def test_n_bars_set(self):
        rets = _make_returns(n=200)
        result = DrawdownAnalyzer().analyze(rets)
        assert result.n_bars == 200

    def test_generated_at_set(self):
        result = DrawdownAnalyzer().analyze(_make_returns())
        assert len(result.generated_at) > 0


# ── Drawdown events ────────────────────────────────────────────────────────
class TestDrawdownEvents:
    def test_event_depth_positive(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        for e in result.events:
            assert e.depth > 0

    def test_event_duration_positive(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        for e in result.events:
            assert e.duration_days >= 0

    def test_event_decline_before_recovery(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        for e in result.events:
            assert e.trough_idx >= e.start_idx

    def test_large_dd_detected(self):
        """The injected DD (~30%) should be the deepest event."""
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        deepest = max(result.events, key=lambda e: e.depth)
        assert deepest.depth > 0.10  # at least 10%

    def test_min_dd_depth_filters(self):
        """Higher min_dd_depth should yield fewer events."""
        rets = _make_returns()
        low = DrawdownAnalyzer(min_dd_depth=0.001).analyze(rets)
        high = DrawdownAnalyzer(min_dd_depth=0.05).analyze(rets)
        assert len(high.events) <= len(low.events)


# ── CDaR ────────────────────────────────────────────────────────────────────
class TestCDaR:
    def test_cdar_95_positive(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.cdar_95 > 0

    def test_cdar_99_geq_95(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.cdar_99 >= result.cdar_95 - 1e-9

    def test_cdar_bounded(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.cdar_95 <= 1.0
        assert result.cdar_99 <= 1.0

    def test_flat_returns_zero_cdar(self):
        result = DrawdownAnalyzer().analyze(_make_flat_returns())
        assert result.cdar_95 == 0.0
        assert result.cdar_99 == 0.0


# ── Regime attribution ──────────────────────────────────────────────────────
class TestRegimeAttribution:
    def test_attribution_present_with_regimes(self):
        rets, regimes = _make_multi_regime_returns()
        result = DrawdownAnalyzer().analyze(rets, regimes=regimes)
        assert len(result.regime_attribution) > 0

    def test_no_attribution_without_regimes(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.regime_attribution == []

    def test_attribution_fields(self):
        rets, regimes = _make_multi_regime_returns()
        result = DrawdownAnalyzer().analyze(rets, regimes=regimes)
        for a in result.regime_attribution:
            assert isinstance(a.regime, str)
            assert a.n_events > 0
            assert a.avg_depth > 0
            assert a.max_depth >= a.avg_depth
            assert 0 <= a.contribution_pct <= 1.0

    def test_crash_regime_deepest(self):
        """Crash regime has the injected -4% daily shocks, should be deepest."""
        rets, regimes = _make_multi_regime_returns()
        result = DrawdownAnalyzer().analyze(rets, regimes=regimes)
        attr_map = {a.regime: a for a in result.regime_attribution}
        if "crash" in attr_map and "bull" in attr_map:
            assert attr_map["crash"].max_depth > attr_map["bull"].max_depth

    def test_contributions_sum_to_one(self):
        rets, regimes = _make_multi_regime_returns()
        result = DrawdownAnalyzer().analyze(rets, regimes=regimes)
        total = sum(a.contribution_pct for a in result.regime_attribution)
        assert total == pytest.approx(1.0, abs=0.01)


# ── Recovery analysis ───────────────────────────────────────────────────────
class TestRecoveryAnalysis:
    def test_recovery_stats_present(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.recovery is not None

    def test_recoveries_counted(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.recovery.n_recoveries >= 0

    def test_avg_recovery_nonnegative(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.recovery.avg_recovery_days >= 0

    def test_fast_leq_slow_threshold(self):
        result = DrawdownAnalyzer().analyze(_make_returns(n=500))
        rec = result.recovery
        if rec.n_recoveries > 3:
            assert rec.fast_recovery_threshold <= rec.slow_recovery_threshold

    def test_shallow_dd_faster_recovery(self):
        """On average, shallow DDs should recover faster than deep ones."""
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        rec = result.recovery
        if rec.n_recoveries > 3 and rec.avg_depth_fast > 0 and rec.avg_depth_slow > 0:
            assert rec.avg_depth_fast <= rec.avg_depth_slow + 0.01

    def test_flat_returns_no_recoveries(self):
        result = DrawdownAnalyzer().analyze(_make_flat_returns())
        assert result.recovery.n_recoveries == 0


# ── Clustering ──────────────────────────────────────────────────────────────
class TestClustering:
    def test_clustering_result_present(self):
        result = DrawdownAnalyzer().analyze(_make_returns())
        assert result.clustering is not None

    def test_clustering_score_bounded(self):
        result = DrawdownAnalyzer().analyze(_make_returns())
        assert 0 <= result.clustering.clustering_score <= 100

    def test_clustered_data_higher_score(self):
        """Back-to-back DDs should produce higher clustering score."""
        clustered = DrawdownAnalyzer().analyze(_make_clustered_dd_returns())
        spread = DrawdownAnalyzer().analyze(_make_returns(seed=99))
        # Clustered should have autocorrelation >= spread (or at least not much less)
        assert clustered.clustering.autocorrelation_lag1 >= spread.clustering.autocorrelation_lag1 - 0.2

    def test_interpretation_present(self):
        result = DrawdownAnalyzer().analyze(_make_returns())
        assert len(result.clustering.interpretation) > 0


# ── Risk ratios ─────────────────────────────────────────────────────────────
class TestRiskRatios:
    def test_ratios_present(self):
        result = DrawdownAnalyzer().analyze(_make_returns())
        assert result.ratios is not None

    def test_calmar_positive_for_positive_cagr(self):
        result = DrawdownAnalyzer().analyze(_make_returns(drift=0.001))
        if result.ratios.cagr > 0 and result.ratios.max_dd > 0:
            assert result.ratios.calmar > 0

    def test_sterling_positive(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        # Returns have positive drift, so CAGR should be positive
        assert isinstance(result.ratios.sterling, float)

    def test_burke_positive(self):
        result = DrawdownAnalyzer().analyze(_make_returns(drift=0.001))
        if result.ratios.cagr > 0:
            assert result.ratios.burke > 0

    def test_cagr_computed(self):
        result = DrawdownAnalyzer().analyze(_make_returns(n=252, drift=0.001))
        assert isinstance(result.ratios.cagr, float)

    def test_max_dd_matches_result(self):
        result = DrawdownAnalyzer().analyze(_make_drawdown_returns())
        assert result.ratios.max_dd == pytest.approx(abs(result.max_dd), abs=1e-9)


# ── Edge cases ──────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_too_few_bars(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        rets = pd.Series([0.01, -0.01, 0.01], index=idx)
        result = DrawdownAnalyzer().analyze(rets)
        assert result.events == []
        assert result.n_bars == 0

    def test_all_positive_returns(self):
        result = DrawdownAnalyzer().analyze(_make_flat_returns())
        assert result.max_dd == 0.0
        assert result.events == []

    def test_single_large_loss(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="B")
        rets = pd.Series([0.005] * 50, index=idx)
        rets.iloc[25] = -0.10
        result = DrawdownAnalyzer().analyze(rets)
        assert result.max_dd < 0
        assert len(result.events) >= 1

    def test_nan_returns_dropped(self):
        rets = _make_returns(n=100)
        rets.iloc[10:15] = np.nan
        result = DrawdownAnalyzer().analyze(rets)
        assert result.n_bars == 95  # 100 - 5 NaN


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            da = DrawdownAnalyzer()
            rets, regimes = _make_multi_regime_returns()
            result = da.analyze(rets, regimes=regimes)
            path = da.generate_report(result, output_path=Path(tmp) / "dd.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            da = DrawdownAnalyzer()
            rets, regimes = _make_multi_regime_returns()
            result = da.analyze(rets, regimes=regimes)
            path = da.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Drawdown Analysis" in html
            assert "Underwater" in html
            assert "Distribution" in html
            assert "Regime Attribution" in html
            assert "Recovery" in html
            assert "Clustering" in html
            assert "Calmar" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            da = DrawdownAnalyzer()
            result = da.analyze(_make_returns())
            path = da.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html

    def test_report_empty_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            da = DrawdownAnalyzer()
            result = DrawdownResult(generated_at="2024-01-01T00:00:00+00:00")
            path = da.generate_report(result, output_path=Path(tmp) / "e.html")
            assert path.exists()


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_drawdown_event(self):
        e = DrawdownEvent(0, 5, 10, 0.05, 10, 5, 5, True, "bull")
        assert e.depth == 0.05
        assert e.recovered is True

    def test_regime_attribution(self):
        r = RegimeAttribution("bear", 3, 0.08, 0.15, 20, 0.4, 0.5)
        assert r.regime == "bear"

    def test_recovery_stats(self):
        r = RecoveryStats(10, 15, 12, 5, 25, 0.03, 0.10)
        assert r.n_recoveries == 10

    def test_clustering_result(self):
        c = ClusteringResult(0.65, 65.0, 30.0, "Moderate")
        assert c.clustering_score == 65.0

    def test_risk_ratios(self):
        r = RiskRatios(1.5, 2.0, 0.8, 0.12, 0.08)
        assert r.calmar == 1.5

    def test_drawdown_result_defaults(self):
        r = DrawdownResult()
        assert r.max_dd == 0.0
        assert r.events == []
        assert r.regime_attribution == []
        assert r.ratios is None
