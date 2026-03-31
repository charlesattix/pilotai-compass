"""Tests for SignalModel."""
import tempfile

import numpy as np
import pandas as pd
import pytest

from compass.signal_model import SignalModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_data(n=200, seed=42):
    """Create simple synthetic training data."""
    np.random.seed(seed)
    features = pd.DataFrame({
        'feature_a': np.random.randn(n),
        'feature_b': np.random.randn(n),
        'feature_c': np.random.randn(n),
    })
    # Labels correlated with feature_a
    labels = (features['feature_a'] > 0).astype(int).values
    return features, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSignalModelTrain:

    def test_train_returns_stats(self):
        """train() should return a non-empty stats dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            features, labels = _make_training_data()
            stats = model.train(features, labels, calibrate=False, save_model=False)

            assert isinstance(stats, dict)
            assert 'test_accuracy' in stats
            assert stats['test_accuracy'] > 0.5

    def test_train_sets_trained_flag(self):
        """After training, model.trained should be True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            features, labels = _make_training_data()
            model.train(features, labels, calibrate=False, save_model=False)
            assert model.trained is True

    def test_train_with_calibration(self):
        """Training with calibration should create calibrated_model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            features, labels = _make_training_data()
            stats = model.train(features, labels, calibrate=True, save_model=False)
            assert model.calibrated_model is not None
            assert 'test_auc_calibrated' in stats


class TestSignalModelPredict:

    def test_predict_after_training(self):
        """predict() should return a dict with probability key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            features, labels = _make_training_data()
            model.train(features, labels, calibrate=False, save_model=False)

            test_features = {'feature_a': 1.5, 'feature_b': 0.2, 'feature_c': -0.5}
            result = model.predict(test_features)

            assert isinstance(result, dict)
            assert 'probability' in result
            assert 0.0 <= result['probability'] <= 1.0

    def test_predict_returns_signal(self):
        """predict() result should contain a signal field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            features, labels = _make_training_data()
            model.train(features, labels, calibrate=False, save_model=False)

            test_features = {'feature_a': 1.5, 'feature_b': 0.2, 'feature_c': -0.5}
            result = model.predict(test_features)
            assert result['signal'] in ['bullish', 'bearish', 'neutral']


class TestSignalModelSaveLoad:

    def test_save_and_load_roundtrip(self):
        """A model saved then loaded should produce the same predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            features, labels = _make_training_data()
            model.train(features, labels, calibrate=False, save_model=False)

            # Save
            model.save('test_model.joblib')

            # Predict before load
            test_features = {'feature_a': 1.5, 'feature_b': 0.2, 'feature_c': -0.5}
            pred_before = model.predict(test_features)

            # Load into new model
            model2 = SignalModel(model_dir=tmpdir)
            success = model2.load('test_model.joblib')
            assert success is True

            pred_after = model2.predict(test_features)
            assert pred_before['probability'] == pytest.approx(
                pred_after['probability'], abs=1e-6
            )

    def test_load_nonexistent_returns_false(self):
        """Loading a nonexistent model should return False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            assert model.load('nonexistent.joblib') is False

    def test_load_most_recent(self):
        """load() with no filename should find the most recent .joblib file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            features, labels = _make_training_data()
            model.train(features, labels, calibrate=False, save_model=False)
            model.save('signal_model_20250101.joblib')
            model.save('signal_model_20250601.joblib')

            model2 = SignalModel(model_dir=tmpdir)
            success = model2.load()
            assert success is True


class TestDefaultPrediction:

    def test_default_prediction_structure(self):
        """_get_default_prediction should return a well-formed dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            result = model._get_default_prediction()
            assert result['probability'] == 0.5
            assert result['confidence'] == 0.0
            assert result['signal'] == 'neutral'
            assert result['fallback'] is True

    def test_predict_untrained_returns_default(self):
        """Predicting with untrained model (no saved models) returns default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            result = model.predict({'feature_a': 1.0})
            assert result.get('fallback') is True
            assert result['probability'] == 0.5
