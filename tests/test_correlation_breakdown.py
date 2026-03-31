"""Tests for compass/correlation_breakdown.py — correlation breakdown analyzer.

Covers:
  - Dataclass construction and field access
  - Conditional correlation matrices by regime
  - Conditional correlation matrices by volatility level
  - Contagion detection and severity classification
  - Contagion risk indicator aggregation
  - Rolling conditional correlations
  - Diversification benefit: normal vs stress
  - Fragility score computation
  - Rolling fragility timeline
  - Full analyze() pipeline
  - from_csv constructor
  - HTML report generation with all sections
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.correlation_breakdown import (
    CALM_REGIMES,
    REGIMES,
    STRESS_REGIMES,
    VOL_BUCKETS,
    ContagionEvent,
    ContagionRiskIndicator,
    CorrelationBreakdownAnalyzer,
    DiversificationBenefit,
    FragilityScore,
    FragilityTimepoint,
    RegimeCorrelation,
    RollingCorrelation,
    VolCorrelation,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_returns(n=200, assets=4, seed=42):
    """Generate synthetic daily returns with regime-dependent correlation."""
    rng = np.random.RandomState(seed)
    names = [f"asset_{i}" for i in range(assets)]
    dates = pd.bdate_range("2024-01-01", periods=n)

    # Base uncorrelated returns
    data = rng.normal(0, 0.01, (n, assets))
    # Inject strong correlation in second half (simulates stress period)
    common = rng.normal(0, 0.01, n)
    for j in range(assets):
        data[n // 2:, j] += common[n // 2:] * 0.8

    return pd.DataFrame(data, index=dates, columns=names)


def _make_regimes(index, seed=42):
    """Generate regime series: bull → neutral → bear → high_vol."""
    n = len(index)
    regimes = np.array(["neutral"] * n, dtype=object)
    regimes[: n // 4] = "bull"
    regimes[n // 4: n // 2] = "neutral"
    regimes[n // 2: 3 * n // 4] = "bear"
    regimes[3 * n // 4:] = "high_vol"
    return pd.Series(regimes, index=index)


def _make_analyzer(n=200, assets=4, seed=42, **kwargs):
    """Create an analyzer with synthetic data."""
    ret = _make_returns(n=n, assets=assets, seed=seed)
    reg = _make_regimes(ret.index, seed=seed)
    return CorrelationBreakdownAnalyzer(ret, regimes=reg, **kwargs)


# ── Dataclass tests ──────────────────────────────────────────────────────


class TestDataclasses:
    def test_contagion_event_fields(self):
        e = ContagionEvent(
            date="2024-06-01", pair=("A", "B"),
            baseline_corr=0.2, stress_corr=0.8, delta=0.6,
            regime="bear", severity="high",
        )
        assert e.pair == ("A", "B")
        assert e.delta == pytest.approx(0.6)
        assert e.severity == "high"

    def test_contagion_risk_indicator_fields(self):
        r = ContagionRiskIndicator(
            level="medium", n_events=3, avg_delta=0.4,
            max_delta=0.6, pct_pairs_affected=0.5, top_pair=("X", "Y"),
        )
        assert r.level == "medium"
        assert r.top_pair == ("X", "Y")

    def test_regime_correlation_fields(self):
        rc = RegimeCorrelation(
            regime="bull", matrix=np.eye(3),
            mean_corr=0.1, max_corr=0.5, min_corr=-0.1,
            n_obs=50, assets=["a", "b", "c"],
        )
        assert rc.min_corr == pytest.approx(-0.1)

    def test_vol_correlation_fields(self):
        vc = VolCorrelation(
            bucket="high", matrix=np.eye(2), mean_corr=0.6,
            max_corr=0.8, n_obs=30, vol_range=(0.02, 0.05),
            assets=["a", "b"],
        )
        assert vc.bucket == "high"
        assert vc.vol_range == (0.02, 0.05)

    def test_fragility_score_fields(self):
        f = FragilityScore(
            score=42.5, eigenvalue_ratio=0.6,
            mean_stress_corr=0.7, mean_calm_corr=0.2,
            correlation_gap=0.5, contagion_count=3,
            diversification_ratio=0.3, tail_concentration=0.8,
        )
        assert f.correlation_gap == pytest.approx(0.5)
        assert f.tail_concentration == pytest.approx(0.8)

    def test_fragility_timepoint_fields(self):
        fp = FragilityTimepoint(
            date="2024-06-01", score=55.0,
            eigenvalue_ratio=0.5, mean_corr=0.3, regime="bear",
        )
        assert fp.regime == "bear"

    def test_diversification_benefit_normal_vs_stress(self):
        d = DiversificationBenefit(
            normal_portfolio_vol=0.01, normal_weighted_avg_vol=0.015,
            normal_benefit=0.33, stress_portfolio_vol=0.025,
            stress_weighted_avg_vol=0.028, stress_benefit=0.10,
            benefit_erosion=0.23,
            marginal_contributions={"A": 0.005},
            stress_marginal_contributions={"A": 0.012},
        )
        assert d.benefit_erosion == pytest.approx(0.23)
        assert d.stress_benefit < d.normal_benefit


# ── Conditional correlation by regime ────────────────────────────────────


class TestConditionalCorrelationsByRegime:
    def test_returns_all_regimes(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for regime in REGIMES:
            assert regime in analyzer.regime_correlations

    def test_matrix_shape(self):
        analyzer = _make_analyzer(assets=5)
        analyzer.analyze()
        for rc in analyzer.regime_correlations.values():
            assert rc.matrix.shape == (5, 5)

    def test_diagonal_is_one(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for rc in analyzer.regime_correlations.values():
            np.testing.assert_allclose(np.diag(rc.matrix), 1.0, atol=1e-10)

    def test_mean_corr_range(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for rc in analyzer.regime_correlations.values():
            assert -1.0 <= rc.mean_corr <= 1.0

    def test_stress_corr_higher_than_calm(self):
        """Bear/high_vol correlation should be higher due to injected common factor."""
        analyzer = _make_analyzer()
        analyzer.analyze()
        calm = analyzer.regime_correlations.get("bull")
        stress = analyzer.regime_correlations.get("bear")
        if calm and stress:
            assert stress.mean_corr > calm.mean_corr

    def test_skips_regime_with_few_obs(self):
        ret = _make_returns(n=20)
        reg = pd.Series(["bull"] * 18 + ["bear"] * 2, index=ret.index)
        analyzer = CorrelationBreakdownAnalyzer(ret, regimes=reg)
        analyzer.analyze()
        assert "bear" not in analyzer.regime_correlations


# ── Conditional correlation by volatility level ──────────────────────────


class TestConditionalCorrelationsByVol:
    def test_produces_vol_buckets(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert len(analyzer.vol_correlations) > 0

    def test_vol_buckets_are_valid(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for bucket in analyzer.vol_correlations:
            assert bucket in VOL_BUCKETS

    def test_vol_buckets_have_obs(self):
        """Each vol bucket should have observations when data is sufficient."""
        analyzer = _make_analyzer()
        analyzer.analyze()
        for vc in analyzer.vol_correlations.values():
            assert vc.n_obs >= 3

    def test_vol_range_tuple(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for vc in analyzer.vol_correlations.values():
            assert len(vc.vol_range) == 2
            assert vc.vol_range[0] <= vc.vol_range[1]

    def test_empty_with_insufficient_data(self):
        ret = _make_returns(n=5)
        reg = _make_regimes(ret.index)
        analyzer = CorrelationBreakdownAnalyzer(ret, regimes=reg)
        analyzer.analyze()
        assert len(analyzer.vol_correlations) == 0


# ── Contagion detection tests ────────────────────────────────────────────


class TestContagionDetection:
    def test_detects_contagion(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert len(analyzer.contagion_events) > 0

    def test_contagion_sorted_by_delta(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        deltas = [e.delta for e in analyzer.contagion_events]
        assert deltas == sorted(deltas, reverse=True)

    def test_contagion_threshold_filters(self):
        a_low = _make_analyzer(contagion_threshold=0.1)
        a_high = _make_analyzer(contagion_threshold=0.9)
        a_low.analyze()
        a_high.analyze()
        assert len(a_low.contagion_events) >= len(a_high.contagion_events)

    def test_contagion_severity_assigned(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for e in analyzer.contagion_events:
            assert e.severity in ("low", "medium", "high")

    def test_contagion_risk_indicator_populated(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.contagion_risk is not None
        assert analyzer.contagion_risk.level in ("low", "medium", "high", "critical")

    def test_contagion_risk_zero_events(self):
        analyzer = _make_analyzer(contagion_threshold=10.0)
        analyzer.analyze()
        assert analyzer.contagion_risk.n_events == 0
        assert analyzer.contagion_risk.level == "low"


# ── Rolling correlation tests ────────────────────────────────────────────


class TestRollingCorrelation:
    def test_produces_pairs(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        n_assets = len(analyzer.assets)
        expected_pairs = n_assets * (n_assets - 1) // 2
        assert len(analyzer.rolling_correlations) == expected_pairs

    def test_values_in_range(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for rc in analyzer.rolling_correlations:
            vals = np.array(rc.values)
            assert np.all(vals >= -1.0 - 1e-10)
            assert np.all(vals <= 1.0 + 1e-10)

    def test_empty_with_short_data(self):
        ret = _make_returns(n=10)
        reg = _make_regimes(ret.index)
        analyzer = CorrelationBreakdownAnalyzer(ret, regimes=reg, window=60)
        analyzer.analyze()
        assert len(analyzer.rolling_correlations) == 0

    def test_configurable_window(self):
        a30 = _make_analyzer(window=30)
        a90 = _make_analyzer(window=90)
        a30.analyze()
        a90.analyze()
        if a30.rolling_correlations and a90.rolling_correlations:
            assert len(a30.rolling_correlations[0].values) >= len(a90.rolling_correlations[0].values)


# ── Diversification benefit tests ────────────────────────────────────────


class TestDiversificationBenefit:
    def test_normal_benefit_positive(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.diversification.normal_benefit > 0

    def test_stress_benefit_computed(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.diversification.stress_portfolio_vol > 0

    def test_benefit_erosion_positive(self):
        """Diversification should erode in stress periods."""
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.diversification.benefit_erosion >= 0

    def test_stress_marginal_contributions_populated(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        d = analyzer.diversification
        assert len(d.stress_marginal_contributions) == len(d.marginal_contributions)

    def test_marginal_contributions_sum_to_pvol(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        d = analyzer.diversification
        total = sum(d.marginal_contributions.values())
        assert total == pytest.approx(d.normal_portfolio_vol, abs=1e-6)

    def test_custom_weights(self):
        ret = _make_returns(assets=3)
        reg = _make_regimes(ret.index)
        weights = {"asset_0": 0.5, "asset_1": 0.3, "asset_2": 0.2}
        analyzer = CorrelationBreakdownAnalyzer(ret, regimes=reg, weights=weights)
        analyzer.analyze()
        assert analyzer.diversification is not None


# ── Fragility score tests ────────────────────────────────────────────────


class TestFragilityScore:
    def test_score_in_range(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert 0 <= analyzer.fragility.score <= 100

    def test_correlation_gap_computed(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.fragility.correlation_gap >= 0

    def test_tail_concentration_positive(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.fragility.tail_concentration > 0

    def test_high_corr_means_higher_fragility(self):
        rng = np.random.RandomState(99)
        n = 200
        base = rng.normal(0, 0.01, n)
        data = {f"a{i}": base + rng.normal(0, 0.0001, n) for i in range(4)}
        ret = pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=n))
        reg = _make_regimes(ret.index)
        high_corr = CorrelationBreakdownAnalyzer(ret, regimes=reg)
        high_corr.analyze()

        low_corr = _make_analyzer()
        low_corr.analyze()

        assert high_corr.fragility.score > low_corr.fragility.score


# ── Fragility timeline tests ────────────────────────────────────────────


class TestFragilityTimeline:
    def test_timeline_populated(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert len(analyzer.fragility_timeline) > 0

    def test_timeline_scores_in_range(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for fp in analyzer.fragility_timeline:
            assert 0 <= fp.score <= 100

    def test_timeline_has_regime_labels(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        regimes_seen = {fp.regime for fp in analyzer.fragility_timeline}
        assert len(regimes_seen) > 0

    def test_timeline_empty_with_short_data(self):
        ret = _make_returns(n=10)
        reg = _make_regimes(ret.index)
        analyzer = CorrelationBreakdownAnalyzer(ret, regimes=reg, window=60)
        analyzer.analyze()
        assert len(analyzer.fragility_timeline) == 0

    def test_timeline_later_points_higher(self):
        """Later points (stress injected in second half) should show higher fragility."""
        analyzer = _make_analyzer()
        analyzer.analyze()
        tl = analyzer.fragility_timeline
        if len(tl) >= 4:
            early_avg = np.mean([fp.score for fp in tl[: len(tl) // 4]])
            late_avg = np.mean([fp.score for fp in tl[3 * len(tl) // 4:]])
            assert late_avg > early_avg


# ── Full pipeline tests ─────────────────────────────────────────────────


class TestAnalyzePipeline:
    def test_analyze_returns_all_keys(self):
        analyzer = _make_analyzer()
        result = analyzer.analyze()
        expected_keys = {
            "contagion_events", "contagion_risk", "regime_correlations",
            "vol_correlations", "fragility", "fragility_timeline",
            "rolling_correlations", "diversification",
        }
        assert set(result.keys()) == expected_keys

    def test_from_csv(self, tmp_path):
        ret = _make_returns()
        csv = tmp_path / "returns.csv"
        ret.to_csv(csv)
        analyzer = CorrelationBreakdownAnalyzer.from_csv(str(csv))
        result = analyzer.analyze()
        assert result["fragility"] is not None

    def test_from_csv_with_regimes(self, tmp_path):
        ret = _make_returns()
        reg = _make_regimes(ret.index)
        ret_csv = tmp_path / "returns.csv"
        reg_csv = tmp_path / "regimes.csv"
        ret.to_csv(ret_csv)
        reg.to_frame("regime").to_csv(reg_csv)
        analyzer = CorrelationBreakdownAnalyzer.from_csv(str(ret_csv), str(reg_csv))
        analyzer.analyze()
        assert len(analyzer.regime_correlations) > 0

    def test_default_neutral_regime(self):
        ret = _make_returns()
        analyzer = CorrelationBreakdownAnalyzer(ret)
        analyzer.analyze()
        assert "neutral" in analyzer.regime_correlations


# ── Report generation tests ──────────────────────────────────────────────


class TestReport:
    def test_generates_html(self, tmp_path):
        analyzer = _make_analyzer()
        path = analyzer.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Correlation Breakdown" in content

    def test_report_contains_all_sections(self, tmp_path):
        analyzer = _make_analyzer()
        path = analyzer.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "Contagion Detection" in content
        assert "Conditional Correlation by Regime" in content
        assert "Volatility Level" in content
        assert "Fragility Timeline" in content
        assert "Diversification Benefit" in content
        assert "Rolling Conditional" in content

    def test_report_embeds_charts(self, tmp_path):
        analyzer = _make_analyzer()
        path = analyzer.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "data:image/png;base64," in content

    def test_report_has_contagion_risk_badge(self, tmp_path):
        analyzer = _make_analyzer()
        path = analyzer.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "risk-badge" in content

    def test_report_has_benefit_erosion(self, tmp_path):
        analyzer = _make_analyzer()
        path = analyzer.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "Benefit Erosion" in content
        assert "Normal" in content
        assert "Stress" in content

    def test_report_auto_runs_analyze(self, tmp_path):
        analyzer = _make_analyzer()
        assert analyzer.fragility is None
        analyzer.generate_report(str(tmp_path / "report.html"))
        assert analyzer.fragility is not None

    def test_report_at_default_path(self):
        analyzer = _make_analyzer()
        path = analyzer.generate_report()
        assert "correlation_breakdown.html" in path
        assert open(path).read().startswith("<!DOCTYPE html>")
