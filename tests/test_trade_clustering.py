"""Tests for compass/trade_clustering.py — unsupervised trade clustering.

Covers:
  - prepare_cluster_features: scaling, NaN handling, column selection
  - find_optimal_k: range, edge cases
  - cluster_kmeans: label count, centroid shape
  - cluster_dbscan: noise handling
  - profile_clusters: win rate, label assignment, centroids
  - extract_filters: ranking, thresholds
  - assess_stability: ARI, fold count
  - TradeClusterAnalyzer: fit, from_csv, report generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.trade_clustering import (
    DEFAULT_CLUSTER_FEATURES,
    ActionableFilter,
    ClusterProfile,
    StabilityResult,
    TradeClusterAnalyzer,
    assess_stability,
    cluster_dbscan,
    cluster_kmeans,
    extract_filters,
    find_optimal_k,
    prepare_cluster_features,
    profile_clusters,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_trades(n=80, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        pnl = rng.normal(30, 200)
        rows.append({
            "entry_date": f"2024-{1+i%12:02d}-{1+i%28:02d}",
            "exit_date": f"2024-{1+i%12:02d}-{5+i%24:02d}",
            "pnl": round(pnl, 2),
            "return_pct": round(pnl / 500, 2),
            "win": 1 if pnl > 0 else 0,
            "dte_at_entry": rng.randint(10, 50),
            "hold_days": rng.randint(1, 20),
            "rsi_14": rng.uniform(20, 80),
            "momentum_5d_pct": rng.normal(0, 2),
            "momentum_10d_pct": rng.normal(0, 3),
            "vix": rng.uniform(12, 40),
            "vix_percentile_50d": rng.uniform(0, 100),
            "iv_rank": rng.uniform(0, 100),
            "dist_from_ma20_pct": rng.normal(0, 2),
            "dist_from_ma50_pct": rng.normal(0, 3),
            "dist_from_ma200_pct": rng.normal(0, 5),
            "ma50_slope_ann_pct": rng.normal(5, 15),
            "realized_vol_20d": rng.uniform(8, 40),
            "net_credit": rng.uniform(0.5, 3.0),
            "spread_width": 5.0,
            "max_loss_per_unit": rng.uniform(3, 5),
            "contracts": rng.randint(1, 5),
            "regime": rng.choice(["bull", "bear", "neutral", "high_vol"]),
            "exit_reason": rng.choice(["close_profit_target", "close_stop_loss"]),
        })
    return pd.DataFrame(rows)


# ── prepare_cluster_features ─────────────────────────────────────────────


class TestPrepareFeatures:
    def test_output_shape(self):
        trades = _make_trades(50)
        X, cols, scaler = prepare_cluster_features(trades)
        assert X.shape[0] == 50
        assert X.shape[1] == len(cols)

    def test_scaled_mean_near_zero(self):
        trades = _make_trades(50)
        X, _, _ = prepare_cluster_features(trades)
        assert abs(X.mean()) < 0.1

    def test_nan_handled(self):
        trades = _make_trades(30)
        trades.loc[0, "vix"] = np.nan
        trades.loc[1, "rsi_14"] = np.nan
        X, _, _ = prepare_cluster_features(trades)
        assert not np.isnan(X).any()

    def test_custom_features(self):
        trades = _make_trades(30)
        X, cols, _ = prepare_cluster_features(trades, ["vix", "rsi_14"])
        assert cols == ["vix", "rsi_14"]
        assert X.shape[1] == 2


# ── find_optimal_k ───────────────────────────────────────────────────────


class TestFindOptimalK:
    def test_returns_int(self):
        X, _, _ = prepare_cluster_features(_make_trades(60))
        k = find_optimal_k(X)
        assert isinstance(k, int)
        assert 2 <= k <= 7

    def test_small_dataset(self):
        X, _, _ = prepare_cluster_features(_make_trades(10))
        k = find_optimal_k(X, range(2, 5))
        assert 2 <= k <= 4


# ── cluster_kmeans ───────────────────────────────────────────────────────


class TestClusterKMeans:
    def test_label_count(self):
        X, _, _ = prepare_cluster_features(_make_trades(60))
        labels, centroids = cluster_kmeans(X, 3)
        assert len(labels) == 60
        assert len(set(labels)) <= 3

    def test_centroid_shape(self):
        X, cols, _ = prepare_cluster_features(_make_trades(60))
        _, centroids = cluster_kmeans(X, 4)
        assert centroids.shape == (4, len(cols))


# ── cluster_dbscan ───────────────────────────────────────────────────────


class TestClusterDBSCAN:
    def test_returns_labels(self):
        X, _, _ = prepare_cluster_features(_make_trades(60))
        labels = cluster_dbscan(X, eps=2.0, min_samples=3)
        assert len(labels) == 60

    def test_noise_label_is_minus_one(self):
        X, _, _ = prepare_cluster_features(_make_trades(60))
        labels = cluster_dbscan(X, eps=0.1, min_samples=50)
        assert -1 in labels  # with tight params, most are noise


# ── profile_clusters ─────────────────────────────────────────────────────


class TestProfileClusters:
    def test_profiles_created(self):
        trades = _make_trades(60)
        X, cols, _ = prepare_cluster_features(trades)
        labels, centroids = cluster_kmeans(X, 3)
        profiles = profile_clusters(trades, labels, centroids, cols)
        assert len(profiles) > 0
        assert all(isinstance(p, ClusterProfile) for p in profiles)

    def test_win_rate_range(self):
        trades = _make_trades(60)
        X, cols, _ = prepare_cluster_features(trades)
        labels, centroids = cluster_kmeans(X, 3)
        profiles = profile_clusters(trades, labels, centroids, cols)
        for p in profiles:
            assert 0 <= p.win_rate <= 1

    def test_label_assigned(self):
        trades = _make_trades(60)
        X, cols, _ = prepare_cluster_features(trades)
        labels, centroids = cluster_kmeans(X, 3)
        profiles = profile_clusters(trades, labels, centroids, cols)
        for p in profiles:
            assert p.label in ("High Win", "Moderate", "Risky", "Underperforming")

    def test_centroids_populated(self):
        trades = _make_trades(60)
        X, cols, _ = prepare_cluster_features(trades)
        labels, centroids = cluster_kmeans(X, 3)
        profiles = profile_clusters(trades, labels, centroids, cols)
        for p in profiles:
            assert len(p.centroid) > 0


# ── extract_filters ──────────────────────────────────────────────────────


class TestExtractFilters:
    def test_returns_list(self):
        trades = _make_trades(100)
        X, cols, _ = prepare_cluster_features(trades)
        labels, centroids = cluster_kmeans(X, 4)
        profiles = profile_clusters(trades, labels, centroids, cols)
        filters = extract_filters(profiles, min_trades=10, min_win_rate=0.50)
        assert isinstance(filters, list)

    def test_filter_fields(self):
        trades = _make_trades(100)
        X, cols, _ = prepare_cluster_features(trades)
        labels, centroids = cluster_kmeans(X, 4)
        profiles = profile_clusters(trades, labels, centroids, cols)
        filters = extract_filters(profiles, min_trades=10, min_win_rate=0.50)
        for f in filters:
            assert isinstance(f, ActionableFilter)
            assert f.low <= f.high
            assert f.win_rate >= 0.50

    def test_sorted_by_evidence(self):
        trades = _make_trades(100)
        X, cols, _ = prepare_cluster_features(trades)
        labels, centroids = cluster_kmeans(X, 4)
        profiles = profile_clusters(trades, labels, centroids, cols)
        filters = extract_filters(profiles, min_trades=10, min_win_rate=0.50)
        if len(filters) >= 2:
            scores = [f.win_rate * f.n_trades for f in filters]
            assert scores == sorted(scores, reverse=True)


# ── assess_stability ─────────────────────────────────────────────────────


class TestAssessStability:
    def test_returns_result(self):
        trades = _make_trades(100)
        X, cols, _ = prepare_cluster_features(trades)
        result = assess_stability(trades, cols, k=3, n_folds=2)
        assert isinstance(result, StabilityResult)

    def test_small_dataset(self):
        trades = _make_trades(15)
        X, cols, _ = prepare_cluster_features(trades)
        result = assess_stability(trades, cols, k=3, n_folds=2)
        assert result.n_folds == 0  # too small


# ── TradeClusterAnalyzer ────────────────────────────────────────────────


class TestTradeClusterAnalyzer:
    def test_fit_runs(self):
        trades = _make_trades(80)
        analyzer = TradeClusterAnalyzer(trades)
        results = analyzer.fit()
        assert "optimal_k" in results
        assert "profiles" in results
        assert "filters" in results
        assert "stability" in results

    def test_cluster_column_added(self):
        trades = _make_trades(80)
        analyzer = TradeClusterAnalyzer(trades)
        analyzer.fit()
        assert "cluster" in analyzer.trades.columns

    def test_from_csv(self, tmp_path):
        csv = tmp_path / "trades.csv"
        _make_trades(50).to_csv(csv, index=False)
        analyzer = TradeClusterAnalyzer.from_csv(str(csv))
        assert len(analyzer.trades) == 50

    def test_pca_computed(self):
        trades = _make_trades(80)
        analyzer = TradeClusterAnalyzer(trades)
        analyzer.fit()
        assert analyzer.pca_2d is not None
        assert analyzer.pca_2d.shape == (80, 2)

    def test_generate_report(self, tmp_path):
        trades = _make_trades(80)
        analyzer = TradeClusterAnalyzer(trades)
        path = analyzer.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Cluster" in content
        assert "data:image/png;base64," in content
        assert "Actionable Filters" in content

    def test_report_no_external(self, tmp_path):
        trades = _make_trades(60)
        analyzer = TradeClusterAnalyzer(trades)
        path = analyzer.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "http://" not in content
        assert "https://" not in content
