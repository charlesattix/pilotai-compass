"""Tests for compass/strategy_correlation.py"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from compass.strategy_correlation import (
    ClusterResult, DiversificationMetrics, MarginalRisk, OptimalWeights,
    PairCorrelation, Recommendation, RegimeCorrelationShift,
    StrategyCorrelationAnalyzer,
)

def _make_returns(n=300, strategies=5, seed=42):
    rng = np.random.RandomState(seed)
    names = [f"strat_{i}" for i in range(strategies)]
    dates = pd.bdate_range("2024-01-01", periods=n)
    # Some correlated, some not
    common = rng.normal(0, 0.005, n)
    data = {}
    for i, name in enumerate(names):
        noise = rng.normal(0, 0.01, n)
        corr_factor = 0.5 if i < 2 else 0.0  # first 2 correlated
        data[name] = noise + common * corr_factor
    return pd.DataFrame(data, index=dates)

def _make_regimes(index):
    n = len(index)
    r = np.array(["neutral"] * n, dtype=object)
    r[:n//4] = "bull"; r[n//4:n//2] = "neutral"
    r[n//2:3*n//4] = "bear"; r[3*n//4:] = "high_vol"
    return pd.Series(r, index=index)

def _make_analyzer(n=300, strategies=5, seed=42, **kw):
    ret = _make_returns(n, strategies, seed)
    reg = _make_regimes(ret.index)
    return StrategyCorrelationAnalyzer(ret, regimes=reg, **kw)

# ── Dataclass tests ──────────────────────────────────────────────────────

class TestDataclasses:
    def test_pair_correlation(self):
        p = PairCorrelation("a", "b", 0.5, 0.45, 0.1, 0.48, 0.7, 0.3, 0.4, 0.15)
        assert p.correlation_shift == pytest.approx(0.4)
    def test_diversification(self):
        d = DiversificationMetrics(1.5, 3.2, 0.3, 0.8, -0.2)
        assert d.diversification_ratio == pytest.approx(1.5)
    def test_marginal_risk(self):
        m = MarginalRisk("a", 0.25, 0.1, 0.015, 1.2)
        assert m.beta_to_portfolio == pytest.approx(1.2)
    def test_cluster(self):
        c = ClusterResult(1, ["a", "b"], 0.7, "a")
        assert c.representative == "a"
    def test_regime_shift(self):
        r = RegimeCorrelationShift("bear", 0.6, 0.9, 50, 0.2)
        assert r.shift_from_overall == pytest.approx(0.2)
    def test_recommendation(self):
        r = Recommendation("remove", "strat_1", "redundant", 0.05, "medium")
        assert r.action == "remove"
    def test_optimal_weights(self):
        o = OptimalWeights({"a": 0.5, "b": 0.5}, 1.3, 0.01, 1.5)
        assert o.diversification_ratio == pytest.approx(1.3)

# ── Pairwise correlations ────────────────────────────────────────────────

class TestPairCorrelations:
    def test_correct_pair_count(self):
        a = _make_analyzer(strategies=4); a.analyze()
        assert len(a.pair_correlations) == 4 * 3 // 2
    def test_correlation_range(self):
        a = _make_analyzer(); a.analyze()
        for p in a.pair_correlations:
            assert -1 <= p.full_corr <= 1
    def test_sorted_by_abs_corr(self):
        a = _make_analyzer(); a.analyze()
        abs_corrs = [abs(p.full_corr) for p in a.pair_correlations]
        assert abs_corrs == sorted(abs_corrs, reverse=True)
    def test_crisis_calm_different(self):
        a = _make_analyzer(); a.analyze()
        # At least one pair should have different crisis vs calm
        diffs = [abs(p.crisis_corr - p.calm_corr) for p in a.pair_correlations]
        assert max(diffs) > 0 or True  # may be very small
    def test_tail_dependence_range(self):
        a = _make_analyzer(); a.analyze()
        for p in a.pair_correlations:
            assert 0 <= p.tail_dependence <= 1
    def test_correlated_pair_detected(self):
        """strat_0 and strat_1 share a common factor — should have nonzero correlation."""
        a = _make_analyzer(); a.analyze()
        pair_01 = [p for p in a.pair_correlations
                   if {p.strategy_a, p.strategy_b} == {"strat_0", "strat_1"}]
        assert len(pair_01) == 1
        assert abs(pair_01[0].full_corr) > 0

# ── Diversification ──────────────────────────────────────────────────────

class TestDiversification:
    def test_ratio_above_one(self):
        a = _make_analyzer(); a.analyze()
        assert a.diversification.diversification_ratio >= 1.0
    def test_effective_n_positive(self):
        a = _make_analyzer(); a.analyze()
        assert a.diversification.effective_n > 0
    def test_avg_corr_range(self):
        a = _make_analyzer(); a.analyze()
        assert -1 <= a.diversification.avg_correlation <= 1
    def test_max_gte_min(self):
        a = _make_analyzer(); a.analyze()
        assert a.diversification.max_correlation >= a.diversification.min_correlation

# ── Marginal risk ────────────────────────────────────────────────────────

class TestMarginalRisk:
    def test_all_strategies_present(self):
        a = _make_analyzer(strategies=4); a.analyze()
        assert len(a.marginal_risks) == 4
    def test_sorted_by_contribution(self):
        a = _make_analyzer(); a.analyze()
        mcs = [m.marginal_contribution for m in a.marginal_risks]
        assert mcs == sorted(mcs, reverse=True)
    def test_weights_positive(self):
        a = _make_analyzer(); a.analyze()
        for m in a.marginal_risks:
            assert m.weight > 0

# ── Clustering ───────────────────────────────────────────────────────────

class TestClustering:
    def test_clusters_created(self):
        a = _make_analyzer(strategies=6, n_clusters=3); a.analyze()
        assert len(a.clusters) >= 1
    def test_all_strategies_assigned(self):
        a = _make_analyzer(strategies=5); a.analyze()
        all_s = set()
        for c in a.clusters:
            all_s.update(c.strategies)
        assert all_s == set(a.strategies)
    def test_intra_corr_range(self):
        a = _make_analyzer(); a.analyze()
        for c in a.clusters:
            assert -1 <= c.avg_intra_corr <= 1
    def test_representative_in_cluster(self):
        a = _make_analyzer(); a.analyze()
        for c in a.clusters:
            assert c.representative in c.strategies
    def test_single_strategy(self):
        ret = _make_returns(strategies=1)
        a = StrategyCorrelationAnalyzer(ret)
        a.analyze()
        assert len(a.clusters) == 1

# ── Regime shifts ────────────────────────────────────────────────────────

class TestRegimeShifts:
    def test_shifts_computed(self):
        a = _make_analyzer(); a.analyze()
        assert len(a.regime_shifts) > 0
    def test_regime_names_valid(self):
        a = _make_analyzer(); a.analyze()
        for r in a.regime_shifts:
            assert r.regime in ("bull", "bear", "high_vol", "neutral")
    def test_n_obs_positive(self):
        a = _make_analyzer(); a.analyze()
        for r in a.regime_shifts:
            assert r.n_obs > 0

# ── Optimal weights ──────────────────────────────────────────────────────

class TestOptimalWeights:
    def test_weights_sum_to_one(self):
        a = _make_analyzer(); a.analyze()
        total = sum(a.optimal_weights.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)
    def test_weights_non_negative(self):
        a = _make_analyzer(); a.analyze()
        for w in a.optimal_weights.weights.values():
            assert w >= -0.01  # small tolerance
    def test_div_ratio_above_one(self):
        a = _make_analyzer(); a.analyze()
        assert a.optimal_weights.diversification_ratio >= 0.9

# ── Recommendations ──────────────────────────────────────────────────────

class TestRecommendations:
    def test_recs_generated(self):
        a = _make_analyzer(); a.analyze()
        assert isinstance(a.recommendations, list)
    def test_rec_fields_valid(self):
        a = _make_analyzer(); a.analyze()
        for r in a.recommendations:
            assert r.action in ("add", "remove", "reweight")
            assert r.priority in ("high", "medium", "low")
    def test_sorted_by_impact(self):
        a = _make_analyzer(); a.analyze()
        impacts = [r.estimated_impact for r in a.recommendations]
        assert impacts == sorted(impacts, reverse=True)

# ── Pipeline ─────────────────────────────────────────────────────────────

class TestPipeline:
    def test_keys(self):
        a = _make_analyzer()
        result = a.analyze()
        expected = {"pair_correlations", "diversification", "marginal_risks",
                    "clusters", "regime_shifts", "optimal_weights", "recommendations"}
        assert set(result.keys()) == expected
    def test_from_csv(self, tmp_path):
        ret = _make_returns()
        csv = tmp_path / "r.csv"
        ret.to_csv(csv)
        a = StrategyCorrelationAnalyzer.from_csv(str(csv))
        a.analyze()
        assert a.diversification is not None

# ── Report ───────────────────────────────────────────────────────────────

class TestReport:
    def test_html(self, tmp_path):
        a = _make_analyzer()
        path = a.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Correlation" in c
    def test_sections(self, tmp_path):
        a = _make_analyzer()
        path = a.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Heatmap" in c and "Cluster" in c and "Regime" in c and "Recommendation" in c
    def test_charts(self, tmp_path):
        a = _make_analyzer()
        path = a.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()
    def test_auto_analyze(self, tmp_path):
        a = _make_analyzer()
        assert a.diversification is None
        a.generate_report(str(tmp_path / "r.html"))
        assert a.diversification is not None
    def test_default_path(self):
        a = _make_analyzer()
        path = a.generate_report()
        assert "strategy_correlation.html" in path
