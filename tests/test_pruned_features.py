"""Tests for pruned feature set integration.

Verifies that:
  - PRUNED_FEATURES has exactly 21 features
  - PRUNED_REMOVED has exactly 10 features
  - No overlap between PRUNED_FEATURES and PRUNED_REMOVED
  - FeaturePipeline(pruned=True) outputs 21 features
  - FeaturePipeline(pruned=False) outputs 31 features
  - Pruned features are a subset of full features
  - Models train successfully on pruned features
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.feature_pipeline import FeaturePipeline
from compass.features import PRUNED_FEATURES, PRUNED_REMOVED

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "compass" / "training_data_combined.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    pytest.skip("training_data_combined.csv not available")


# ── PRUNED_FEATURES / PRUNED_REMOVED constants ──────────────────────────


class TestPrunedFeatureConstants:

    def test_pruned_features_count(self):
        assert len(PRUNED_FEATURES) == 21

    def test_pruned_removed_count(self):
        assert len(PRUNED_REMOVED) == 10

    def test_no_overlap(self):
        assert not set(PRUNED_FEATURES) & set(PRUNED_REMOVED)

    def test_union_is_31(self):
        assert len(set(PRUNED_FEATURES) | set(PRUNED_REMOVED)) == 31

    def test_no_duplicates_in_pruned(self):
        assert len(PRUNED_FEATURES) == len(set(PRUNED_FEATURES))

    def test_no_duplicates_in_removed(self):
        assert len(PRUNED_REMOVED) == len(set(PRUNED_REMOVED))

    def test_contracts_log_in_removed(self):
        assert "contracts_log" in PRUNED_REMOVED

    def test_contracts_log_not_in_pruned(self):
        assert "contracts_log" not in PRUNED_FEATURES

    def test_signal_features_in_pruned(self):
        """Key signal features must survive pruning."""
        for f in ["credit_to_width", "loss_to_width", "iv_rank",
                  "strategy_type_CS", "spread_type_bull_put", "rsi_14"]:
            assert f in PRUNED_FEATURES, f"{f} should be in PRUNED_FEATURES"

    def test_regime_dummies_removed(self):
        for f in ["regime_bear", "regime_bull", "regime_crash",
                  "regime_high_vol", "regime_low_vol"]:
            assert f in PRUNED_REMOVED

    def test_strategy_type_cs_kept(self):
        assert "strategy_type_CS" in PRUNED_FEATURES

    def test_strategy_type_ic_ss_removed(self):
        assert "strategy_type_IC" in PRUNED_REMOVED
        assert "strategy_type_SS" in PRUNED_REMOVED


# ── FeaturePipeline pruning ──────────────────────────────────────────────


class TestPipelinePruning:

    def test_default_is_pruned(self, raw_df):
        pipeline = FeaturePipeline()
        assert pipeline.pruned is True

    def test_pruned_output_21_features(self, raw_df):
        pipeline = FeaturePipeline(pruned=True)
        features = pipeline.transform(raw_df)
        assert len(features.columns) == 21

    def test_full_output_31_features(self, raw_df):
        pipeline = FeaturePipeline(pruned=False)
        features = pipeline.transform(raw_df)
        assert len(features.columns) == 31

    def test_pruned_matches_constant(self, raw_df):
        pipeline = FeaturePipeline(pruned=True)
        features = pipeline.transform(raw_df)
        assert list(features.columns) == PRUNED_FEATURES

    def test_pruned_subset_of_full(self, raw_df):
        full = FeaturePipeline(pruned=False).transform(raw_df)
        pruned = FeaturePipeline(pruned=True).transform(raw_df)
        assert set(pruned.columns).issubset(set(full.columns))

    def test_removed_not_in_pruned(self, raw_df):
        features = FeaturePipeline(pruned=True).transform(raw_df)
        for f in PRUNED_REMOVED:
            assert f not in features.columns

    def test_no_nans_in_pruned(self, raw_df):
        features = FeaturePipeline(pruned=True).transform(raw_df)
        assert features.isna().sum().sum() == 0

    def test_no_infs_in_pruned(self, raw_df):
        features = FeaturePipeline(pruned=True).transform(raw_df)
        assert np.isfinite(features.values).all()

    def test_row_count_preserved(self, raw_df):
        features = FeaturePipeline(pruned=True).transform(raw_df)
        assert len(features) == len(raw_df)


# ── Model training on pruned features ────────────────────────────────────


class TestModelTrainingPruned:

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="training data not available")
    def test_signal_model_trains_on_pruned(self):
        from compass.signal_model import SignalModel
        df = pd.read_csv(DATA_PATH)
        features = FeaturePipeline(pruned=True).transform(df)
        labels = df["win"].values.astype(int)

        model = SignalModel(model_dir="/tmp/test_pruned_signal")
        model.train(features_df=features, labels=labels, save_model=False)
        assert model.trained
        assert len(model.feature_names) == 21

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="training data not available")
    def test_ensemble_model_trains_on_pruned(self):
        from compass.ensemble_signal_model import EnsembleSignalModel
        df = pd.read_csv(DATA_PATH)
        features = FeaturePipeline(pruned=True).transform(df)
        labels = df["win"].values.astype(int)

        model = EnsembleSignalModel(model_dir="/tmp/test_pruned_ensemble")
        model.train(features_df=features, labels=labels, save_model=False)
        assert model.trained
        assert len(model.feature_names) == 21

    @pytest.mark.skipif(not DATA_PATH.exists(), reason="training data not available")
    def test_ensemble_predict_on_pruned(self):
        from compass.ensemble_signal_model import EnsembleSignalModel
        df = pd.read_csv(DATA_PATH)
        features = FeaturePipeline(pruned=True).transform(df)
        labels = df["win"].values.astype(int)

        model = EnsembleSignalModel(model_dir="/tmp/test_pruned_ensemble_pred")
        model.train(features_df=features, labels=labels, save_model=False)
        probs = model.predict_batch(features)
        assert len(probs) == len(df)
        assert all(0 <= p <= 1 for p in probs)
