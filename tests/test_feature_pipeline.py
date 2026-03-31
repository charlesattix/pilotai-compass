"""Tests for compass.feature_pipeline — stationary, normalized ML features."""

import numpy as np
import pandas as pd
import pytest

from compass.feature_pipeline import (
    FeaturePipeline,
    _zscore_column,
    _log1p_column,
    _credit_to_width,
    _loss_to_width,
    _IMPUTATION_DEFAULTS,
    _RAW_PRICE_COLUMNS,
    ZSCORE_WINDOW,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw_trade_df(n=100, seed=42):
    """Simulate a raw trade DataFrame as enrich_trades() would produce.

    Includes all the problematic columns (spy_price, vix, contracts, etc.)
    plus the safe ones so we can verify end-to-end pipeline output.
    """
    rng = np.random.RandomState(seed)
    # Simulate SPY price trending upward (the non-stationarity we're fixing)
    spy_base = np.linspace(400, 550, n) + rng.randn(n) * 5

    df = pd.DataFrame({
        # Problematic raw prices
        "spy_price": spy_base,
        "short_strike": spy_base - rng.uniform(5, 20, n),
        "vix": 15 + rng.randn(n) * 5,
        "contracts": rng.randint(1, 10, n),

        # Dollar-denominated trade structure
        "net_credit": rng.uniform(0.3, 1.5, n),
        "spread_width": rng.choice([2.0, 5.0, 10.0], n),
        "max_loss_per_unit": rng.uniform(0.5, 9.5, n),

        # Safe features (should pass through)
        "rsi_14": rng.uniform(20, 80, n),
        "momentum_5d_pct": rng.randn(n) * 2,
        "momentum_10d_pct": rng.randn(n) * 3,
        "iv_rank": rng.uniform(10, 90, n),
        "vix_percentile_50d": rng.uniform(10, 90, n),
        "vix_percentile_100d": rng.uniform(10, 90, n),
        "dist_from_ma20_pct": rng.randn(n) * 2,
        "dist_from_ma50_pct": rng.randn(n) * 3,
        "dist_from_ma80_pct": rng.randn(n) * 4,
        "dist_from_ma200_pct": rng.randn(n) * 5,
        "ma50_slope_ann_pct": rng.randn(n) * 10,
        "realized_vol_atr20": 15 + rng.randn(n) * 5,
        "realized_vol_20d": 15 + rng.randn(n) * 5,
        "days_since_last_trade": rng.randint(1, 30, n),
        "vix_change_5d": rng.randn(n) * 2,

        # Categoricals
        "regime": rng.choice(["bull", "bear", "high_vol"], n),
        "strategy_type": rng.choice(["CS", "IC"], n),
        "spread_type": rng.choice(["bull_put", "bear_call"], n),

        # Target (not used by pipeline, but present in raw data)
        "win": rng.randint(0, 2, n),
        "entry_date": pd.date_range("2020-01-01", periods=n, freq="W"),
    })
    return df


# ---------------------------------------------------------------------------
# Z-score helper
# ---------------------------------------------------------------------------

class TestZscoreColumn:

    def test_output_is_bounded(self):
        """Z-scores should be clipped to [-4, 4]."""
        s = pd.Series(np.random.randn(200) * 100)
        z = _zscore_column(s)
        assert z.min() >= -4.0
        assert z.max() <= 4.0

    def test_output_length_matches_input(self):
        s = pd.Series(np.random.randn(50))
        z = _zscore_column(s)
        assert len(z) == len(s)

    def test_no_nans_in_output(self):
        s = pd.Series(np.random.randn(100))
        z = _zscore_column(s)
        assert not z.isna().any()

    def test_constant_series_gives_zero(self):
        s = pd.Series([5.0] * 50)
        z = _zscore_column(s)
        # Constant series → std=0 → z=0
        np.testing.assert_array_equal(z.values, np.zeros(50))

    def test_mean_near_zero_for_long_series(self):
        """For a stationary series, z-score mean should be near zero."""
        rng = np.random.RandomState(0)
        s = pd.Series(rng.randn(500))
        z = _zscore_column(s, window=60)
        # Skip the first window where expanding is used
        assert abs(z.iloc[100:].mean()) < 0.3

    def test_trending_input_stays_bounded(self):
        """Even on a trending input (like SPY price), z-scores are bounded."""
        s = pd.Series(np.linspace(300, 600, 500))
        z = _zscore_column(s, window=60)
        assert z.min() >= -4.0
        assert z.max() <= 4.0


# ---------------------------------------------------------------------------
# Log transform
# ---------------------------------------------------------------------------

class TestLog1pColumn:

    def test_small_values(self):
        s = pd.Series([0, 1, 2, 3])
        result = _log1p_column(s)
        expected = np.log1p(s)
        np.testing.assert_array_almost_equal(result.values, expected.values)

    def test_zero_stays_zero(self):
        s = pd.Series([0.0])
        assert _log1p_column(s).iloc[0] == 0.0

    def test_large_values_compressed(self):
        s = pd.Series([1, 10, 100, 1000])
        result = _log1p_column(s)
        # Log compresses: result[3] should be much less than 1000
        assert result.iloc[3] < 10


# ---------------------------------------------------------------------------
# Ratio helpers
# ---------------------------------------------------------------------------

class TestCreditToWidth:

    def test_normal_case(self):
        row = pd.Series({"net_credit": 0.50, "spread_width": 5.0})
        assert _credit_to_width(row) == pytest.approx(0.10)

    def test_zero_width_returns_zero(self):
        row = pd.Series({"net_credit": 0.50, "spread_width": 0.0})
        assert _credit_to_width(row) == 0.0

    def test_missing_credit_returns_zero(self):
        row = pd.Series({"net_credit": None, "spread_width": 5.0})
        assert _credit_to_width(row) == 0.0


class TestLossToWidth:

    def test_normal_case(self):
        row = pd.Series({"max_loss_per_unit": 4.50, "spread_width": 5.0})
        assert _loss_to_width(row) == pytest.approx(0.90)

    def test_zero_width_returns_zero(self):
        row = pd.Series({"max_loss_per_unit": 4.50, "spread_width": 0.0})
        assert _loss_to_width(row) == 0.0


# ---------------------------------------------------------------------------
# FeaturePipeline — core transform
# ---------------------------------------------------------------------------

class TestFeaturePipelineTransform:

    @pytest.fixture
    def pipeline(self):
        return FeaturePipeline()

    @pytest.fixture
    def raw_df(self):
        return _make_raw_trade_df(n=100)

    def test_returns_dataframe(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_preserved(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        assert len(result) == len(raw_df)

    def test_no_raw_price_columns(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        for col in _RAW_PRICE_COLUMNS:
            assert col not in result.columns, f"Raw price column {col} should be removed"

    def test_has_zscore_columns(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        assert "spy_price_zscore" in result.columns
        assert "vix_zscore" in result.columns

    def test_zscore_columns_bounded(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        assert result["spy_price_zscore"].min() >= -4.0
        assert result["spy_price_zscore"].max() <= 4.0
        assert result["vix_zscore"].min() >= -4.0
        assert result["vix_zscore"].max() <= 4.0

    def test_contracts_log_pruned_by_default(self, pipeline, raw_df):
        """contracts_log is pruned by the default pipeline (harmful feature)."""
        result = pipeline.transform(raw_df)
        assert "contracts_log" not in result.columns
        # Present in full (unpruned) pipeline
        full = FeaturePipeline(pruned=False).transform(raw_df)
        assert "contracts_log" in full.columns
        assert full["contracts_log"].max() < 5.0

    def test_no_contracts_raw(self, pipeline, raw_df):
        """Raw contracts should not be in output features."""
        result = pipeline.transform(raw_df)
        default_feats = FeaturePipeline.default_numeric_features()
        assert "contracts" not in default_feats

    def test_has_credit_to_width(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        assert "credit_to_width" in result.columns
        # Ratio should be between 0 and ~1
        assert result["credit_to_width"].min() >= 0
        assert result["credit_to_width"].max() <= 1.5

    def test_has_loss_to_width(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        assert "loss_to_width" in result.columns

    def test_has_vix_change_pct(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        assert "vix_change_5d_pct" in result.columns

    def test_no_nans(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        assert not result.isna().any().any(), f"NaN found in: {result.columns[result.isna().any()].tolist()}"

    def test_no_infs(self, pipeline, raw_df):
        result = pipeline.transform(raw_df)
        assert np.isfinite(result.values).all()

    def test_one_hot_encoded_categoricals(self, pipeline, raw_df):
        """Pruned pipeline keeps only signal-carrying dummies."""
        result = pipeline.transform(raw_df)
        # Pruned: only strategy_type_CS and spread_type_bull_put survive
        strategy_cols = [c for c in result.columns if c.startswith("strategy_type_")]
        spread_cols = [c for c in result.columns if c.startswith("spread_type_")]
        assert "strategy_type_CS" in result.columns
        assert "spread_type_bull_put" in result.columns
        # Full pipeline has all dummies
        full = FeaturePipeline(pruned=False).transform(raw_df)
        assert len([c for c in full.columns if c.startswith("regime_")]) >= 2
        assert len([c for c in full.columns if c.startswith("strategy_type_")]) >= 2

    def test_safe_features_pass_through(self, pipeline, raw_df):
        """Features like rsi_14, dist_from_ma* should be in output unchanged."""
        result = pipeline.transform(raw_df)
        assert "rsi_14" in result.columns
        assert "dist_from_ma200_pct" in result.columns
        # Values should match (after NaN fill)
        np.testing.assert_array_almost_equal(
            result["rsi_14"].values,
            raw_df["rsi_14"].fillna(50.0).values,
        )


# ---------------------------------------------------------------------------
# FeaturePipeline — imputation
# ---------------------------------------------------------------------------

class TestFeaturePipelineImputation:

    def test_nan_vix_percentile_filled_with_50(self):
        df = _make_raw_trade_df(n=10)
        df.loc[0, "vix_percentile_50d"] = np.nan
        pipeline = FeaturePipeline()
        result = pipeline.transform(df)
        assert result.loc[0, "vix_percentile_50d"] == 50.0

    def test_nan_rsi_filled_with_50(self):
        df = _make_raw_trade_df(n=10)
        df.loc[0, "rsi_14"] = np.nan
        pipeline = FeaturePipeline()
        result = pipeline.transform(df)
        assert result.loc[0, "rsi_14"] == 50.0

    def test_nan_iv_rank_filled_with_50(self):
        df = _make_raw_trade_df(n=10)
        df.loc[0, "iv_rank"] = np.nan
        pipeline = FeaturePipeline()
        result = pipeline.transform(df)
        assert result.loc[0, "iv_rank"] == 50.0

    def test_nan_realized_vol_filled_with_20(self):
        df = _make_raw_trade_df(n=10)
        df.loc[0, "realized_vol_20d"] = np.nan
        pipeline = FeaturePipeline()
        result = pipeline.transform(df)
        assert result.loc[0, "realized_vol_20d"] == 20.0

    def test_missing_column_gets_default(self):
        """If a numeric feature column is entirely absent, fill with default."""
        df = _make_raw_trade_df(n=10)
        df = df.drop(columns=["iv_rank"])
        pipeline = FeaturePipeline()
        result = pipeline.transform(df)
        assert "iv_rank" in result.columns
        assert (result["iv_rank"] == 50.0).all()


# ---------------------------------------------------------------------------
# FeaturePipeline — sentinel capping
# ---------------------------------------------------------------------------

class TestFeaturePipelineSentinelCapping:

    def test_days_to_earnings_capped(self):
        df = _make_raw_trade_df(n=5)
        df["days_to_earnings"] = [999, 10, 45, 999, 30]
        pipeline = FeaturePipeline(
            numeric_features=["days_to_earnings"],
            categorical_features=[],
        )
        result = pipeline.transform(df)
        assert result["days_to_earnings"].max() <= 60

    def test_days_to_fomc_capped(self):
        df = _make_raw_trade_df(n=5)
        df["days_to_fomc"] = [999, 5, 999, 20, 999]
        pipeline = FeaturePipeline(
            numeric_features=["days_to_fomc"],
            categorical_features=[],
        )
        result = pipeline.transform(df)
        assert result["days_to_fomc"].max() <= 60


# ---------------------------------------------------------------------------
# FeaturePipeline — stationarity verification
# ---------------------------------------------------------------------------

class TestFeaturePipelineStationarity:

    def test_spy_zscore_stationary_on_trending_prices(self):
        """spy_price_zscore should not trend even when spy_price trends."""
        n = 200
        rng = np.random.RandomState(0)
        df = _make_raw_trade_df(n=n, seed=0)
        # Make SPY price trend strongly upward
        df["spy_price"] = np.linspace(350, 600, n)

        pipeline = FeaturePipeline(
            numeric_features=["spy_price_zscore"],
            categorical_features=[],
        )
        result = pipeline.transform(df)

        # The z-score should NOT have a strong trend
        z = result["spy_price_zscore"].values
        # Compare first-half mean vs second-half mean — should be similar
        first_half = z[:n // 2].mean()
        second_half = z[n // 2:].mean()
        assert abs(first_half - second_half) < 2.0, (
            f"Z-score drifted: first_half={first_half:.2f} second_half={second_half:.2f}"
        )

    def test_raw_spy_price_would_drift(self):
        """Sanity check: raw spy_price DOES drift (confirming the problem we fix)."""
        n = 200
        df = _make_raw_trade_df(n=n, seed=0)
        df["spy_price"] = np.linspace(350, 600, n)

        first_half = df["spy_price"].iloc[:n // 2].mean()
        second_half = df["spy_price"].iloc[n // 2:].mean()
        # Raw price has ~125-point drift
        assert abs(second_half - first_half) > 100


# ---------------------------------------------------------------------------
# FeaturePipeline — edge cases
# ---------------------------------------------------------------------------

class TestFeaturePipelineEdgeCases:

    def test_single_row(self):
        df = _make_raw_trade_df(n=1)
        pipeline = FeaturePipeline()
        result = pipeline.transform(df)
        assert len(result) == 1
        assert not result.isna().any().any()

    def test_all_nan_column(self):
        df = _make_raw_trade_df(n=10)
        df["iv_rank"] = np.nan
        pipeline = FeaturePipeline()
        result = pipeline.transform(df)
        # Should fill with imputation default (50.0 for iv_rank)
        assert (result["iv_rank"] == 50.0).all()

    def test_custom_numeric_features(self):
        df = _make_raw_trade_df(n=10)
        pipeline = FeaturePipeline(
            numeric_features=["rsi_14", "vix_zscore"],
            categorical_features=[],
        )
        result = pipeline.transform(df)
        assert list(result.columns) == ["rsi_14", "vix_zscore"]

    def test_empty_categoricals(self):
        df = _make_raw_trade_df(n=10)
        pipeline = FeaturePipeline(categorical_features=[])
        result = pipeline.transform(df)
        cat_cols = [c for c in result.columns
                    if c.startswith("regime_") or c.startswith("strategy_") or c.startswith("spread_")]
        assert len(cat_cols) == 0


# ---------------------------------------------------------------------------
# FeaturePipeline — default feature list
# ---------------------------------------------------------------------------

class TestDefaultFeatures:

    def test_no_raw_prices_in_defaults(self):
        defaults = FeaturePipeline.default_numeric_features()
        for col in _RAW_PRICE_COLUMNS:
            assert col not in defaults, f"{col} should not be in default features"

    def test_no_contracts_raw_in_defaults(self):
        defaults = FeaturePipeline.default_numeric_features()
        assert "contracts" not in defaults
        assert "contracts_log" in defaults

    def test_no_dollar_features_in_defaults(self):
        defaults = FeaturePipeline.default_numeric_features()
        for col in ["net_credit", "spread_width", "max_loss_per_unit"]:
            assert col not in defaults, f"Dollar feature {col} should be ratio-ized"

    def test_has_zscore_features_in_defaults(self):
        defaults = FeaturePipeline.default_numeric_features()
        assert "spy_price_zscore" in defaults
        assert "vix_zscore" in defaults

    def test_defaults_are_all_strings(self):
        defaults = FeaturePipeline.default_numeric_features()
        assert all(isinstance(f, str) for f in defaults)

    def test_defaults_no_duplicates(self):
        defaults = FeaturePipeline.default_numeric_features()
        assert len(defaults) == len(set(defaults))


# ---------------------------------------------------------------------------
# Integration: pipeline produces features suitable for sklearn
# ---------------------------------------------------------------------------

class TestPipelineSklearnIntegration:

    def test_output_works_with_sklearn(self):
        """Pipeline output should be directly usable by sklearn classifiers."""
        from sklearn.tree import DecisionTreeClassifier

        df = _make_raw_trade_df(n=100)
        pipeline = FeaturePipeline()
        features = pipeline.transform(df)
        labels = df["win"].values

        # Should not raise
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(features.values, labels)
        proba = clf.predict_proba(features.values)
        assert proba.shape == (100, 2)

    def test_feature_count_reasonable(self):
        """Output should have ~20 numeric + ~8 one-hot ≈ 28 features."""
        df = _make_raw_trade_df(n=50)
        pipeline = FeaturePipeline()
        features = pipeline.transform(df)
        n_cols = features.shape[1]
        assert 20 <= n_cols <= 40, f"Expected 20-40 features, got {n_cols}"
