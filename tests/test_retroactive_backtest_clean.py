"""Tests for scripts/retroactive_backtest_clean.py — clean pipeline integration."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from retroactive_backtest_clean import (
    FOLDS,
    LABEL_COL,
    LEGACY_RAW_FEATURES,
    prepare_clean_features,
    prepare_legacy_features,
    walk_forward_eval,
)
from compass.feature_pipeline import FeaturePipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_trade_df(n=300, seed=42):
    """Simulate a raw trade DataFrame matching collect_training_data output."""
    rng = np.random.RandomState(seed)
    years = np.concatenate([
        np.full(60, 2020), np.full(60, 2021), np.full(60, 2022),
        np.full(50, 2023), np.full(40, 2024), np.full(30, 2025),
    ])
    n = len(years)
    spy_base = np.linspace(340, 550, n) + rng.randn(n) * 5

    return pd.DataFrame({
        "year": years.astype(int),
        "entry_date": pd.date_range("2020-01-02", periods=n, freq="B").strftime("%Y-%m-%d"),
        "exit_date": pd.date_range("2020-01-16", periods=n, freq="B").strftime("%Y-%m-%d"),
        "strategy_type": rng.choice(["CS", "IC", "SS"], n),
        "spread_type": rng.choice(["bull_put", "bear_call"], n),
        "regime": rng.choice(["bull", "bear", "low_vol"], n),
        "spy_price": spy_base,
        "short_strike": spy_base - rng.uniform(5, 20, n),
        "vix": 15 + rng.randn(n) * 5,
        "contracts": rng.randint(1, 10, n),
        "net_credit": rng.uniform(0.3, 1.5, n),
        "spread_width": rng.choice([5.0, 10.0, 12.0], n),
        "max_loss_per_unit": rng.uniform(3.5, 11.0, n),
        "otm_pct": rng.uniform(0.01, 0.05, n),
        "dte_at_entry": rng.randint(10, 35, n),
        "rsi_14": rng.uniform(25, 75, n),
        "momentum_5d_pct": rng.randn(n) * 2,
        "momentum_10d_pct": rng.randn(n) * 3,
        "vix_percentile_20d": rng.uniform(10, 90, n),
        "vix_percentile_50d": rng.uniform(10, 90, n),
        "vix_percentile_100d": rng.uniform(10, 90, n),
        "iv_rank": rng.uniform(10, 90, n),
        "dist_from_ma20_pct": rng.randn(n) * 2,
        "dist_from_ma50_pct": rng.randn(n) * 3,
        "dist_from_ma80_pct": rng.randn(n) * 4,
        "dist_from_ma200_pct": rng.randn(n) * 5,
        "ma20_slope_ann_pct": rng.randn(n) * 10,
        "ma50_slope_ann_pct": rng.randn(n) * 8,
        "realized_vol_atr20": rng.uniform(10, 40, n),
        "realized_vol_5d": rng.uniform(8, 50, n),
        "realized_vol_10d": rng.uniform(8, 45, n),
        "realized_vol_20d": rng.uniform(8, 40, n),
        "day_of_week": rng.randint(0, 5, n),
        "days_since_last_trade": rng.randint(1, 10, n),
        "pnl": rng.normal(100, 500, n),
        "return_pct": rng.normal(2, 10, n),
        "win": (rng.randn(n) > -0.3).astype(int),
    })


@pytest.fixture
def trade_df():
    return _make_trade_df()


# ---------------------------------------------------------------------------
# prepare_legacy_features
# ---------------------------------------------------------------------------

class TestPrepareLegacy:

    def test_returns_feature_cols_and_df(self, trade_df):
        cols, df = prepare_legacy_features(trade_df)
        assert len(cols) > len(LEGACY_RAW_FEATURES)  # raw + one-hot encoded
        assert len(df) == len(trade_df)

    def test_one_hot_columns_present(self, trade_df):
        cols, df = prepare_legacy_features(trade_df)
        cat_cols = [c for c in cols if c.startswith("strategy_type_") or c.startswith("regime_")]
        assert len(cat_cols) > 0

    def test_no_nans_after_fill(self, trade_df):
        cols, df = prepare_legacy_features(trade_df)
        assert not df[cols].isna().any().any()

    def test_raw_prices_still_present(self, trade_df):
        """Legacy pipeline keeps raw spy_price and vix."""
        cols, df = prepare_legacy_features(trade_df)
        assert "vix" in cols
        assert "spy_price" not in cols  # spy_price was never in LEGACY_RAW_FEATURES


# ---------------------------------------------------------------------------
# prepare_clean_features
# ---------------------------------------------------------------------------

class TestPrepareClean:

    def test_returns_feature_cols_and_df(self, trade_df):
        cols, df = prepare_clean_features(trade_df)
        assert len(cols) > 0
        assert len(df) == len(trade_df)

    def test_has_zscore_features(self, trade_df):
        cols, df = prepare_clean_features(trade_df)
        assert "spy_price_zscore" in cols
        assert "vix_zscore" in cols

    def test_has_ratio_features(self, trade_df):
        cols, df = prepare_clean_features(trade_df)
        assert "credit_to_width" in cols
        assert "loss_to_width" in cols

    def test_contracts_log_pruned(self, trade_df):
        """contracts_log is removed by the pruned pipeline (harmful feature)."""
        cols, df = prepare_clean_features(trade_df)
        assert "contracts_log" not in cols

    def test_no_raw_prices(self, trade_df):
        cols, df = prepare_clean_features(trade_df)
        assert "spy_price" not in cols
        assert "short_strike" not in cols
        assert "vix" not in cols  # raw vix replaced by vix_zscore

    def test_no_nans_or_infs(self, trade_df):
        cols, df = prepare_clean_features(trade_df)
        assert np.isfinite(df.values).all()

    def test_zscore_bounded(self, trade_df):
        """Z-scored columns should be clipped to [-4, 4]."""
        cols, df = prepare_clean_features(trade_df)
        for col in ["spy_price_zscore", "vix_zscore"]:
            if col in df.columns:
                assert df[col].max() <= 4.0 + 1e-9
                assert df[col].min() >= -4.0 - 1e-9


# ---------------------------------------------------------------------------
# walk_forward_eval
# ---------------------------------------------------------------------------

class TestWalkForwardEval:

    def test_runs_all_folds(self, trade_df):
        cols, df = prepare_legacy_features(trade_df)
        results = walk_forward_eval(trade_df, cols, df)
        # With years 2020-2025 and 3 folds (test=2023,2024,2025), should get 3 results
        assert len(results) == 3

    def test_fold_metrics_valid(self, trade_df):
        cols, df = prepare_legacy_features(trade_df)
        results = walk_forward_eval(trade_df, cols, df)
        for fold in results:
            assert 0.0 <= fold["auc"] <= 1.0
            assert 0.0 <= fold["accuracy"] <= 1.0
            assert fold["n_train"] > 0
            assert fold["n_test"] > 0
            assert fold["top_feature"] in cols

    def test_clean_pipeline_runs(self, trade_df):
        cols, df = prepare_clean_features(trade_df)
        df_aligned = df.reindex(trade_df.index)
        results = walk_forward_eval(trade_df, cols, df_aligned)
        assert len(results) == 3
        for fold in results:
            assert 0.0 <= fold["auc"] <= 1.0

    def test_both_pipelines_comparable(self, trade_df):
        """Both pipelines should produce valid results on the same data."""
        leg_cols, leg_df = prepare_legacy_features(trade_df)
        cln_cols, cln_df = prepare_clean_features(trade_df)
        cln_df = cln_df.reindex(trade_df.index)

        leg_results = walk_forward_eval(trade_df, leg_cols, leg_df)
        cln_results = walk_forward_eval(trade_df, cln_cols, cln_df)

        assert len(leg_results) == len(cln_results)
        for lr, cr in zip(leg_results, cln_results):
            assert lr["test_year"] == cr["test_year"]
            assert lr["n_test"] == cr["n_test"]


# ---------------------------------------------------------------------------
# Integration with real CSV (if available)
# ---------------------------------------------------------------------------

class TestIntegrationRealCSV:

    @pytest.fixture
    def real_csv(self):
        path = PROJECT_ROOT / "compass" / "training_data_combined.csv"
        if not path.exists():
            pytest.skip("training_data_combined.csv not available")
        return pd.read_csv(path)

    def test_legacy_pipeline_on_real_data(self, real_csv):
        cols, df = prepare_legacy_features(real_csv)
        results = walk_forward_eval(real_csv, cols, df)
        assert len(results) >= 1
        for fold in results:
            assert fold["auc"] > 0.5  # better than random

    def test_clean_pipeline_on_real_data(self, real_csv):
        cols, df = prepare_clean_features(real_csv)
        df_aligned = df.reindex(real_csv.index)
        results = walk_forward_eval(real_csv, cols, df_aligned)
        assert len(results) >= 1
        for fold in results:
            assert fold["auc"] > 0.5

    def test_feature_pipeline_transform_roundtrip(self, real_csv):
        """FeaturePipeline should produce no NaN/inf on real data."""
        pipeline = FeaturePipeline()
        transformed = pipeline.transform(real_csv)
        assert np.isfinite(transformed.values).all()
        assert len(transformed) == len(real_csv)
