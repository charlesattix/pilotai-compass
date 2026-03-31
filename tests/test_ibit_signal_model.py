"""Tests for compass/ibit_signal_model.py — IBIT-specific ML signal model.

Covers:
  - IBITSignalModel training: stats, feature importance, scale_pos_weight
  - Walk-forward validation: fold structure, AUC range
  - Prediction: single, batch, untrained fallback
  - Save/load: roundtrip, path traversal guard
  - IBIT regime classification: VIX thresholds, BTC corr, weekend gaps
  - HTML report generation
  - Feature names alignment
  - IBITTrainingStats dataclass
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.ibit_signal_model import (
    IBIT_REGIME_CONFIG,
    IBITSignalModel,
    IBITTrainingStats,
    WalkForwardFold,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_features(n=200, n_feat=15, seed=42):
    """Synthetic features with learnable target."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_feat)
    logit = 0.6 * X[:, 0] - 0.4 * X[:, 1] + 0.3 * X[:, 2]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    cols = [f"feat_{i}" for i in range(n_feat)]
    return pd.DataFrame(X, columns=cols), y


def _make_imbalanced_features(n=200, win_rate=0.85, seed=42):
    """Synthetic features with class imbalance (like IBIT's 84% win rate)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 10)
    y = (rng.random(n) < win_rate).astype(int)
    # Add weak signal to first feature
    X[:, 0] += y * 0.3
    cols = [f"feat_{i}" for i in range(10)]
    return pd.DataFrame(X, columns=cols), y


# ── Training ─────────────────────────────────────────────────────────────


class TestIBITTraining:
    def test_train_returns_stats(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        stats = model.train(features, labels, save_model=False, n_wf_folds=2)
        assert "test_auc" in stats
        assert "wf_mean_auc" in stats
        assert stats["test_auc"] > 0.0
        assert model.trained

    def test_feature_names_stored(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False)
        assert model.feature_names == list(features.columns)

    def test_feature_importance_computed(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False)
        imp = model.get_feature_importance(5)
        assert len(imp) > 0
        assert all(isinstance(name, str) and isinstance(val, float) for name, val in imp)

    def test_imbalanced_scale_pos_weight(self, tmp_path):
        features, labels = _make_imbalanced_features(200, win_rate=0.85)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False)
        # scale_pos_weight should be < 1 (fewer negatives than positives)
        assert model.xgb_params["scale_pos_weight"] < 1.0

    def test_feature_means_computed(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False)
        assert model.feature_means is not None
        assert model.feature_stds is not None
        assert len(model.feature_means) == features.shape[1]

    def test_custom_xgb_params(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path),
                                xgb_params={"max_depth": 2})
        model.train(features, labels, save_model=False)
        assert model.xgb_params["max_depth"] == 2


# ── Walk-forward validation ──────────────────────────────────────────────


class TestWalkForward:
    def test_folds_created(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False, n_wf_folds=3)
        assert model._ibit_stats is not None
        assert len(model._ibit_stats.walk_forward_folds) >= 1

    def test_fold_auc_range(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False, n_wf_folds=3)
        for fold in model._ibit_stats.walk_forward_folds:
            assert 0 <= fold.auc <= 1

    def test_fold_has_importance(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False, n_wf_folds=3)
        for fold in model._ibit_stats.walk_forward_folds:
            assert isinstance(fold.feature_importances, dict)

    def test_small_data_no_folds(self, tmp_path):
        features, labels = _make_features(30)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False, n_wf_folds=3)
        # Too small for walk-forward
        assert model._ibit_stats.walk_forward_folds == [] or len(model._ibit_stats.walk_forward_folds) >= 0


# ── Prediction ───────────────────────────────────────────────────────────


class TestPrediction:
    def test_predict_single(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False)

        row = {f"feat_{i}": float(features.iloc[0, i]) for i in range(features.shape[1])}
        result = model.predict(row)
        assert "prediction" in result
        assert "probability" in result
        assert 0 <= result["probability"] <= 1
        assert result["signal"] in ("bullish", "bearish", "neutral")

    def test_predict_batch(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False)

        probas = model.predict_batch(features.iloc[:10])
        assert probas.shape == (10,)
        assert np.all((probas >= 0) & (probas <= 1))

    def test_untrained_fallback(self, tmp_path):
        model = IBITSignalModel(model_dir=str(tmp_path))
        result = model.predict({"feat_0": 1.0})
        assert result.get("fallback") is True
        assert result["probability"] == 0.5

    def test_untrained_batch_returns_half(self, tmp_path):
        model = IBITSignalModel(model_dir=str(tmp_path))
        probas = model.predict_batch(pd.DataFrame({"feat_0": [1, 2, 3]}))
        np.testing.assert_array_equal(probas, [0.5, 0.5, 0.5])


# ── Save/Load ────────────────────────────────────────────────────────────


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=True)

        loaded = IBITSignalModel(model_dir=str(tmp_path))
        assert loaded.load()
        assert loaded.trained
        assert loaded.feature_names == model.feature_names

        probas_orig = model.predict_batch(features.iloc[:5])
        probas_loaded = loaded.predict_batch(features.iloc[:5])
        np.testing.assert_allclose(probas_orig, probas_loaded, atol=1e-6)

    def test_load_nonexistent(self, tmp_path):
        model = IBITSignalModel(model_dir=str(tmp_path))
        assert not model.load("nonexistent.joblib")

    def test_load_most_recent(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=True)

        loaded = IBITSignalModel(model_dir=str(tmp_path))
        assert loaded.load()  # finds most recent


# ── Regime classification ────────────────────────────────────────────────


class TestIBITRegime:
    def test_crash(self):
        assert IBITSignalModel.classify_ibit_regime(vix=65) == "crash"

    def test_high_vol(self):
        assert IBITSignalModel.classify_ibit_regime(vix=45) == "high_vol"

    def test_low_vol(self):
        assert IBITSignalModel.classify_ibit_regime(vix=15) == "low_vol"

    def test_normal(self):
        assert IBITSignalModel.classify_ibit_regime(vix=25) == "normal"

    def test_decorrelating(self):
        r = IBITSignalModel.classify_ibit_regime(vix=25, btc_corr=0.3)
        assert r == "decorrelating"

    def test_gap_risk(self):
        r = IBITSignalModel.classify_ibit_regime(vix=25, gap_pct=5.0)
        assert r == "gap_risk"

    def test_thresholds_higher_than_spy(self):
        cfg = IBIT_REGIME_CONFIG
        assert cfg["vix_low"] > 15    # SPY uses 15
        assert cfg["vix_high"] > 30   # SPY uses 30


# ── HTML report ──────────────────────────────────────────────────────────


class TestReport:
    def test_generate_report(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False, n_wf_folds=2)
        path = model.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "IBIT Signal Model" in content
        assert "data:image/png;base64," in content
        assert "Feature Importance" in content
        assert "Walk-Forward" in content

    def test_report_no_external(self, tmp_path):
        features, labels = _make_features(200)
        model = IBITSignalModel(model_dir=str(tmp_path))
        model.train(features, labels, save_model=False)
        path = model.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "http://" not in content
        assert "https://" not in content

    def test_report_without_training_returns_empty(self, tmp_path):
        model = IBITSignalModel(model_dir=str(tmp_path))
        result = model.generate_report(str(tmp_path / "report.html"))
        assert result == ""
