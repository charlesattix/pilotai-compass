"""Tests for compass/trade_clusters.py — re-export alias module.

Verifies all public names are accessible via the alias and that the
core analyzer works through this import path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_trades(n=60, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        pnl = rng.normal(30, 200)
        rows.append({
            "entry_date": f"2024-{1+i%12:02d}-{1+i%28:02d}",
            "exit_date": f"2024-{1+i%12:02d}-{5+i%24:02d}",
            "pnl": round(pnl, 2), "return_pct": round(pnl / 500, 2),
            "win": 1 if pnl > 0 else 0,
            "dte_at_entry": rng.randint(10, 50), "hold_days": rng.randint(1, 20),
            "rsi_14": rng.uniform(20, 80), "momentum_5d_pct": rng.normal(0, 2),
            "momentum_10d_pct": rng.normal(0, 3), "vix": rng.uniform(12, 40),
            "vix_percentile_50d": rng.uniform(0, 100), "iv_rank": rng.uniform(0, 100),
            "dist_from_ma20_pct": rng.normal(0, 2), "dist_from_ma50_pct": rng.normal(0, 3),
            "dist_from_ma200_pct": rng.normal(0, 5), "ma50_slope_ann_pct": rng.normal(5, 15),
            "realized_vol_20d": rng.uniform(8, 40), "net_credit": rng.uniform(0.5, 3.0),
            "spread_width": 5.0, "max_loss_per_unit": rng.uniform(3, 5),
            "contracts": rng.randint(1, 5),
            "regime": rng.choice(["bull", "bear", "neutral"]),
            "exit_reason": rng.choice(["close_profit_target", "close_stop_loss"]),
        })
    return pd.DataFrame(rows)


class TestAliasImports:
    def test_all_names_importable(self):
        from compass.trade_clusters import (
            TradeClusterAnalyzer, ClusterProfile, ActionableFilter,
            StabilityResult, prepare_cluster_features, find_optimal_k,
            cluster_kmeans, cluster_dbscan, profile_clusters,
            extract_filters, assess_stability, DEFAULT_CLUSTER_FEATURES,
        )
        assert TradeClusterAnalyzer is not None
        assert len(DEFAULT_CLUSTER_FEATURES) > 0

    def test_analyzer_via_alias(self):
        from compass.trade_clusters import TradeClusterAnalyzer
        trades = _make_trades(60)
        analyzer = TradeClusterAnalyzer(trades)
        results = analyzer.fit()
        assert "profiles" in results
        assert "optimal_k" in results

    def test_report_via_alias(self, tmp_path):
        from compass.trade_clusters import TradeClusterAnalyzer
        trades = _make_trades(60)
        analyzer = TradeClusterAnalyzer(trades)
        path = analyzer.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
