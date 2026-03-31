"""Tests for compass.online_retrain."""
import tempfile
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import pytest

from compass.online_retrain import ModelRetrainer, RetrainTrigger, ABResult, _model_file_prefix
from compass.signal_model import SignalModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(n=300, seed=42):
    """Create synthetic training data with enough samples for train+holdout."""
    rng = np.random.RandomState(seed)
    features = pd.DataFrame({
        'feature_a': rng.randn(n),
        'feature_b': rng.randn(n),
        'feature_c': rng.randn(n),
    })
    labels = (features['feature_a'] + 0.3 * features['feature_b'] > 0).astype(int).values
    return features, labels


def _trained_model(tmpdir, features, labels):
    """Return a SignalModel that has been trained and saved."""
    model = SignalModel(model_dir=tmpdir)
    model.train(features, labels, calibrate=False, save_model=True)
    return model


# ---------------------------------------------------------------------------
# RetrainTrigger checks
# ---------------------------------------------------------------------------

class TestTriggerEvaluation:

    def test_no_trigger_when_model_is_fresh(self):
        """A freshly-trained model should not trigger retraining."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data()
            model = _trained_model(tmpdir, features, labels)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            trigger = retrainer._evaluate_triggers(model, features, labels)

            assert trigger.triggered is False
            assert trigger.reasons == []

    def test_trigger_on_model_age(self):
        """Model older than max_age_days should trigger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data()
            model = _trained_model(tmpdir, features, labels)

            # Fake old timestamp
            old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
            model.training_stats['timestamp'] = old_ts

            retrainer = ModelRetrainer(model_dir=tmpdir, max_age_days=30)
            trigger = retrainer._evaluate_triggers(model, features, labels)

            assert trigger.triggered is True
            assert any("model_age" in r for r in trigger.reasons)

    def test_trigger_on_feature_drift(self):
        """Shifted features should trigger drift detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data()
            model = _trained_model(tmpdir, features, labels)

            # Shift all features by 10 std devs
            shifted = features.copy()
            shifted['feature_a'] += 50
            shifted['feature_b'] += 50
            shifted['feature_c'] += 50

            retrainer = ModelRetrainer(model_dir=tmpdir, drift_feature_pct=0.10)
            trigger = retrainer._evaluate_triggers(model, shifted, labels)

            assert trigger.triggered is True
            assert any("feature_drift" in r for r in trigger.reasons)
            assert len(trigger.drift_features) > 0

    def test_trigger_on_performance_drop(self):
        """AUC drop below threshold should trigger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data()
            model = _trained_model(tmpdir, features, labels)

            # Use random labels to simulate performance drop
            rng = np.random.RandomState(99)
            bad_labels = rng.randint(0, 2, size=len(labels))

            # Force a high baseline so the drop is detectable
            model.training_stats['test_auc'] = 0.95

            retrainer = ModelRetrainer(model_dir=tmpdir, perf_auc_drop=0.05)
            trigger = retrainer._evaluate_triggers(model, features, bad_labels)

            # We can't guarantee the exact AUC on random labels, but the drop
            # from 0.95 should be large enough
            if trigger.perf_auc_current is not None:
                assert trigger.triggered is True
                assert any("auc_drop" in r for r in trigger.reasons)


# ---------------------------------------------------------------------------
# Feature drift
# ---------------------------------------------------------------------------

class TestFeatureDrift:

    def test_no_drift_on_same_data(self):
        """Drift check on the training data itself should return empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data()
            model = _trained_model(tmpdir, features, labels)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            drifted = retrainer._check_feature_drift(model, features)
            assert drifted == []

    def test_drift_detected_on_shifted_data(self):
        """Shifting one feature by many stds should flag it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data()
            model = _trained_model(tmpdir, features, labels)

            shifted = features.copy()
            shifted['feature_a'] += 100  # way beyond 3 stds
            retrainer = ModelRetrainer(model_dir=tmpdir)
            drifted = retrainer._check_feature_drift(model, shifted)
            assert 'feature_a' in drifted

    def test_drift_handles_missing_stats(self):
        """If model has no feature_means, drift check returns empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = SignalModel(model_dir=tmpdir)
            features, _ = _make_data(n=50)
            retrainer = ModelRetrainer(model_dir=tmpdir)
            assert retrainer._check_feature_drift(model, features) == []


# ---------------------------------------------------------------------------
# Rolling window
# ---------------------------------------------------------------------------

class TestRollingWindow:

    def test_window_by_row_count(self):
        """Without date info, should trim by approximate row count."""
        features, labels = _make_data(n=500)
        retrainer = ModelRetrainer(rolling_window_months=6)
        windowed_f, windowed_l = retrainer._apply_rolling_window(features, labels)
        expected_rows = 6 * 21
        assert len(windowed_f) == expected_rows
        assert len(windowed_l) == expected_rows

    def test_window_by_datetime_index(self):
        """With DatetimeIndex, should use calendar-based window."""
        dates = pd.date_range('2024-01-01', periods=400, freq='B')
        features = pd.DataFrame(
            {'feature_a': np.random.randn(400)},
            index=dates,
        )
        labels = np.random.randint(0, 2, 400)

        retrainer = ModelRetrainer(rolling_window_months=6)
        windowed_f, windowed_l = retrainer._apply_rolling_window(features, labels)

        # Should have roughly 6 months of business days
        assert len(windowed_f) < 400
        assert len(windowed_f) > 100

    def test_small_dataset_returned_as_is(self):
        """If dataset smaller than window, return it all."""
        features, labels = _make_data(n=50)
        retrainer = ModelRetrainer(rolling_window_months=12)
        windowed_f, windowed_l = retrainer._apply_rolling_window(features, labels)
        assert len(windowed_f) == 50


# ---------------------------------------------------------------------------
# A/B comparison
# ---------------------------------------------------------------------------

class TestABComparison:

    def test_new_model_promoted_when_better(self):
        """New model trained on same data should be promoted (AUC >= old - delta)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=400)
            old_model = _trained_model(tmpdir, features, labels)
            new_model = _trained_model(tmpdir, features, labels)

            holdout_f = features.iloc[-80:]
            holdout_l = labels[-80:]

            retrainer = ModelRetrainer(model_dir=tmpdir)
            ab = retrainer._compare_models(old_model, new_model, holdout_f, holdout_l)

            assert isinstance(ab, ABResult)
            assert ab.holdout_size == 80
            # Same data → same model → should be promoted
            assert ab.promoted is True

    def test_untrained_old_model_gets_baseline_auc(self):
        """If old model is untrained, it should get AUC=0.5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=300)
            old_model = SignalModel(model_dir=tmpdir)  # untrained
            new_model = _trained_model(tmpdir, features, labels)

            holdout_f = features.iloc[-60:]
            holdout_l = labels[-60:]

            retrainer = ModelRetrainer(model_dir=tmpdir)
            ab = retrainer._compare_models(old_model, new_model, holdout_f, holdout_l)

            assert ab.old_auc == 0.5
            assert ab.promoted is True


# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------

class TestVersioning:

    def test_save_versioned_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data()
            model = _trained_model(tmpdir, features, labels)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            path = retrainer._save_versioned(model)
            assert path.exists()
            assert path.suffix == '.joblib'

    def test_prune_keeps_n_versions(self):
        """After pruning, only keep_versions files should remain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data()
            retrainer = ModelRetrainer(model_dir=tmpdir, keep_versions=2)

            # Train once then save with distinct filenames
            model = SignalModel(model_dir=tmpdir)
            model.train(features, labels, calibrate=False, save_model=False)

            for i in range(5):
                model.save(f'signal_model_2025010{i}_120000.joblib')
                time.sleep(0.02)  # ensure distinct mtime

            before = list(retrainer.model_dir.glob('signal_model_*.joblib'))
            assert len(before) == 5

            deleted = retrainer._prune_old_versions()
            remaining = list(retrainer.model_dir.glob('signal_model_*.joblib'))

            assert len(deleted) == 3  # 5 - keep_versions(2)
            assert len(remaining) == 2

    def test_list_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data()
            model = _trained_model(tmpdir, features, labels)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            retrainer._save_versioned(model)
            versions = retrainer.list_versions()
            assert len(versions) >= 1
            assert 'filename' in versions[0]
            assert 'modified' in versions[0]


# ---------------------------------------------------------------------------
# Full check_and_retrain integration
# ---------------------------------------------------------------------------

class TestCheckAndRetrain:

    def test_no_retrain_when_fresh(self):
        """Fresh model + same data → no retrain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=300)
            model = _trained_model(tmpdir, features, labels)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            result = retrainer.check_and_retrain(features, labels, current_model=model)

            assert result.retrained is False
            assert result.trigger.triggered is False

    def test_force_retrain(self):
        """force=True should always retrain and produce an A/B result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=300)
            model = _trained_model(tmpdir, features, labels)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            result = retrainer.check_and_retrain(
                features, labels, current_model=model, force=True
            )

            assert result.retrained is True
            assert result.ab_result is not None
            assert result.training_stats is not None
            assert 'forced' in result.trigger.reasons

    def test_retrain_from_scratch(self):
        """With no existing model, should train from scratch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=300)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            result = retrainer.check_and_retrain(features, labels)

            assert result.retrained is True
            assert result.ab_result is not None

    def test_retrain_skipped_when_too_few_samples(self):
        """Should skip retraining when data is below min_samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=20)

            retrainer = ModelRetrainer(model_dir=tmpdir, min_samples=100)
            result = retrainer.check_and_retrain(features, labels, force=True)

            assert result.retrained is False

    def test_promoted_model_is_loadable(self):
        """After promotion, the new model should be loadable by SignalModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=300)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            result = retrainer.check_and_retrain(features, labels, force=True)

            assert result.retrained is True
            if result.ab_result and result.ab_result.promoted:
                loader = SignalModel(model_dir=tmpdir)
                assert loader.load() is True
                assert loader.trained is True


# ===========================================================================
# Helpers shared by GAP tests
# ===========================================================================

class _MockModel:
    """Minimal model stub with the same interface as SignalModel.

    Avoids sklearn/calibration dependency issues in GAP tests.
    Used to test ModelRetrainer logic without real model training.
    """

    def __init__(self, model_dir="ml/models"):
        self.model_dir = model_dir
        self.trained = False
        self.feature_names = None
        self.feature_means = None
        self.feature_stds = None
        self.training_stats = {}
        self._save_calls = []
        self._load_returns = True

    def train(self, features_df, labels, calibrate=True, save_model=False, **kw):
        self.trained = True
        self.feature_names = list(features_df.columns)
        X = features_df.values
        self.feature_means = np.mean(X, axis=0)
        self.feature_stds = np.std(X, axis=0)
        self.training_stats = {
            "test_auc": 0.75,
            "n_train": len(features_df),
        }
        return self.training_stats

    def predict_batch(self, features_df):
        n = len(features_df)
        return np.full(n, 0.8)

    def save(self, filename):
        import os
        path = os.path.join(self.model_dir, filename)
        with open(path, 'w') as f:
            f.write('mock')
        self._save_calls.append(filename)

    def load(self):
        self.trained = self._load_returns
        if self.trained:
            self.feature_names = ['f_a', 'f_b']
            self.feature_means = np.array([0.0, 0.0])
            self.feature_stds = np.array([1.0, 1.0])
            self.training_stats = {"test_auc": 0.75}
        return self._load_returns

    def get_fallback_stats(self):
        return {}


class _MockEnsembleModel(_MockModel):
    """Mock that mimics EnsembleSignalModel's training_stats key name."""

    def train(self, features_df, labels, calibrate=True, save_model=False, **kw):
        super().train(features_df, labels, calibrate=calibrate, save_model=save_model)
        # EnsembleSignalModel uses "ensemble_test_auc" not "test_auc"
        self.training_stats = {
            "ensemble_test_auc": 0.72,
            "ensemble_test_accuracy": 0.84,
            "n_train": len(features_df),
        }
        return self.training_stats


class _IdentityPipeline:
    """Pipeline stub that returns the DataFrame unchanged (for base checks)."""

    def transform(self, df):
        self._last_input = df.copy()
        return df


class _PrefixPipeline:
    """Pipeline stub that adds 'pipe_' prefix to all columns."""

    def transform(self, df):
        self._last_input = df.copy()
        result = df.copy()
        result.columns = [f"pipe_{c}" for c in df.columns]
        return result


# ===========================================================================
# GAP-3: model_class parameter
# ===========================================================================

class TestGap3ModelClass:
    """GAP-3: ModelRetrainer.model_class replaces hardcoded SignalModel()."""

    def test_default_model_class_is_signal_model(self):
        retrainer = ModelRetrainer()
        assert retrainer.model_class is SignalModel

    def test_custom_model_class_stored(self):
        retrainer = ModelRetrainer(model_class=_MockModel)
        assert retrainer.model_class is _MockModel

    def test_check_and_retrain_instantiates_model_class(self):
        """When current_model=None, check_and_retrain should instantiate model_class."""
        instantiated = []

        class TrackingModel(_MockModel):
            def __init__(self, model_dir="ml/models"):
                super().__init__(model_dir)
                instantiated.append(self)

        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=200)
            retrainer = ModelRetrainer(
                model_dir=tmpdir,
                model_class=TrackingModel,
                min_samples=10,
            )
            retrainer.check_and_retrain(features, labels, force=True)

        # Should have created at least 2 instances: current_model + new_model
        assert len(instantiated) >= 2
        assert all(isinstance(m, TrackingModel) for m in instantiated)

    def test_check_and_retrain_trains_new_model_class(self):
        """The trained candidate must be an instance of model_class."""
        trained_types = []

        class RecordingModel(_MockModel):
            def train(self, features_df, labels, **kw):
                trained_types.append(type(self).__name__)
                return super().train(features_df, labels, **kw)

        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=200)
            retrainer = ModelRetrainer(
                model_dir=tmpdir,
                model_class=RecordingModel,
                min_samples=10,
            )
            result = retrainer.check_and_retrain(features, labels, force=True)

        assert result.retrained is True
        assert 'RecordingModel' in trained_types

    def test_model_file_prefix_helper_signal_model(self):
        assert _model_file_prefix(SignalModel) == 'signal_model'

    def test_model_file_prefix_helper_ensemble_class(self):
        """Any class with 'Ensemble' in its name gets the ensemble prefix."""

        class EnsembleSignalModel:
            pass

        assert _model_file_prefix(EnsembleSignalModel) == 'ensemble_model'

    def test_model_file_prefix_helper_custom_name(self):
        """Classes without 'Ensemble' in the name get the signal_model prefix."""

        class MyCustomModel:
            pass

        assert _model_file_prefix(MyCustomModel) == 'signal_model'

    def test_check_performance_reads_ensemble_test_auc(self):
        """GAP-1: _check_performance must work when model has 'ensemble_test_auc'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=150)
            model = _MockEnsembleModel(model_dir=tmpdir)
            model.trained = True
            model.training_stats = {"ensemble_test_auc": 0.85}
            model.feature_names = list(features.columns)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            result = retrainer._check_performance(model, features, labels)

            # Should not return None — baseline_auc found via ensemble_test_auc
            assert result is not None
            assert result["baseline_auc"] == 0.85

    def test_check_performance_falls_back_to_test_auc(self):
        """GAP-1: backward compat — 'test_auc' still works for SignalModel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=150)
            model = _MockModel(model_dir=tmpdir)
            model.trained = True
            model.training_stats = {"test_auc": 0.78}
            model.feature_names = list(features.columns)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            result = retrainer._check_performance(model, features, labels)

            assert result is not None
            assert result["baseline_auc"] == 0.78

    def test_check_performance_returns_none_when_no_auc_key(self):
        """If neither AUC key exists, _check_performance returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=150)
            model = _MockModel(model_dir=tmpdir)
            model.trained = True
            model.training_stats = {"other_key": 0.5}
            model.feature_names = list(features.columns)

            retrainer = ModelRetrainer(model_dir=tmpdir)
            assert retrainer._check_performance(model, features, labels) is None


# ===========================================================================
# GAP-4: file glob patterns
# ===========================================================================

class TestGap4FileGlobs:
    """GAP-4: Glob patterns use model_class-derived prefix, not hardcoded strings."""

    def test_default_retrainer_prefix_is_signal_model(self):
        retrainer = ModelRetrainer()
        assert retrainer._model_file_prefix == 'signal_model'

    def test_ensemble_retrainer_prefix_is_ensemble_model(self):
        retrainer = ModelRetrainer(model_class=_MockEnsembleModel)
        assert retrainer._model_file_prefix == 'ensemble_model'

    def test_save_versioned_signal_model_prefix(self):
        """_save_versioned with default (SignalModel) names file signal_model_*."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _MockModel(model_dir=tmpdir)
            model.trained = True

            retrainer = ModelRetrainer(model_dir=tmpdir)
            path = retrainer._save_versioned(model)

            assert path.name.startswith('signal_model_'), \
                f"Expected 'signal_model_*' prefix, got '{path.name}'"
            assert path.exists()

    def test_save_versioned_ensemble_prefix(self):
        """_save_versioned with EnsembleSignalModel names file ensemble_model_*."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _MockEnsembleModel(model_dir=tmpdir)
            model.trained = True

            retrainer = ModelRetrainer(model_dir=tmpdir, model_class=_MockEnsembleModel)
            path = retrainer._save_versioned(model)

            assert path.name.startswith('ensemble_model_'), \
                f"Expected 'ensemble_model_*' prefix, got '{path.name}'"
            assert path.exists()

    def test_prune_ensemble_files_only(self):
        """_prune_old_versions for ensemble retrainer only touches ensemble_model_*."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = __import__('pathlib').Path(tmpdir)
            # Create 4 ensemble files and 2 signal files
            for i in range(4):
                (tmppath / f"ensemble_model_2025010{i}_120000.joblib").write_text('x')
                time.sleep(0.01)
            for i in range(2):
                (tmppath / f"signal_model_2025010{i}_120000.joblib").write_text('x')

            retrainer = ModelRetrainer(
                model_dir=tmpdir,
                model_class=_MockEnsembleModel,
                keep_versions=2,
            )
            deleted = retrainer._prune_old_versions()

            # Should delete 2 ensemble files (4 - keep_versions=2)
            assert len(deleted) == 2
            assert all('ensemble_model' in p.name for p in deleted)

            # Signal model files must be untouched
            signal_files = list(tmppath.glob('signal_model_*.joblib'))
            assert len(signal_files) == 2

    def test_prune_signal_files_only(self):
        """Default retrainer only prunes signal_model_* files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = __import__('pathlib').Path(tmpdir)
            for i in range(4):
                (tmppath / f"signal_model_2025010{i}_120000.joblib").write_text('x')
                time.sleep(0.01)
            for i in range(2):
                (tmppath / f"ensemble_model_2025010{i}_120000.joblib").write_text('x')

            retrainer = ModelRetrainer(model_dir=tmpdir, keep_versions=2)
            deleted = retrainer._prune_old_versions()

            assert len(deleted) == 2
            assert all('signal_model' in p.name for p in deleted)
            # Ensemble files untouched
            ens_files = list(tmppath.glob('ensemble_model_*.joblib'))
            assert len(ens_files) == 2

    def test_count_versions_signal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = __import__('pathlib').Path(tmpdir)
            for i in range(3):
                (tmppath / f"signal_model_2025010{i}.joblib").write_text('x')
            (tmppath / "ensemble_model_20250101.joblib").write_text('x')

            retrainer = ModelRetrainer(model_dir=tmpdir)
            assert retrainer._count_versions() == 3

    def test_count_versions_ensemble(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = __import__('pathlib').Path(tmpdir)
            for i in range(2):
                (tmppath / f"ensemble_model_2025010{i}.joblib").write_text('x')
            (tmppath / "signal_model_20250101.joblib").write_text('x')

            retrainer = ModelRetrainer(model_dir=tmpdir, model_class=_MockEnsembleModel)
            assert retrainer._count_versions() == 2

    def test_list_versions_ensemble_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = __import__('pathlib').Path(tmpdir)
            (tmppath / "ensemble_model_20250101_120000.joblib").write_text('x')
            (tmppath / "signal_model_20250101_120000.joblib").write_text('x')

            retrainer = ModelRetrainer(model_dir=tmpdir, model_class=_MockEnsembleModel)
            versions = retrainer.list_versions()

            assert len(versions) == 1
            assert versions[0]['filename'].startswith('ensemble_model_')

    def test_get_model_age_uses_correct_prefix(self):
        """_get_model_age_days falls back to the right glob when no timestamp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = __import__('pathlib').Path(tmpdir)
            # Only an ensemble file exists
            (tmppath / "ensemble_model_20250101_120000.joblib").write_text('x')

            retrainer = ModelRetrainer(model_dir=tmpdir, model_class=_MockEnsembleModel)
            model = _MockEnsembleModel(model_dir=tmpdir)
            model.trained = True
            model.training_stats = {}  # no timestamp

            age = retrainer._get_model_age_days(model)
            # Should find the ensemble file and return an age (>= 0)
            assert age is not None
            assert age >= 0

    def test_get_model_age_signal_prefix_ignored_for_ensemble(self):
        """Ensemble retrainer ignores signal_model_* files in age fallback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = __import__('pathlib').Path(tmpdir)
            # Only a signal file exists (wrong prefix for ensemble retrainer)
            (tmppath / "signal_model_20250101_120000.joblib").write_text('x')

            retrainer = ModelRetrainer(model_dir=tmpdir, model_class=_MockEnsembleModel)
            model = _MockEnsembleModel(model_dir=tmpdir)
            model.trained = True
            model.training_stats = {}

            age = retrainer._get_model_age_days(model)
            # Should return None — no ensemble files found
            assert age is None


# ===========================================================================
# GAP-5: FeaturePipeline integration
# ===========================================================================

class TestGap5FeaturePipeline:
    """GAP-5: feature_pipeline ensures training features match inference features."""

    def test_no_pipeline_passes_raw_features(self):
        """Without a pipeline, trades_df is used directly for training."""
        trained_cols = []

        class RecordingModel(_MockModel):
            def train(self, features_df, labels, **kw):
                trained_cols.extend(list(features_df.columns))
                return super().train(features_df, labels, **kw)

        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=200)
            retrainer = ModelRetrainer(
                model_dir=tmpdir,
                model_class=RecordingModel,
                min_samples=10,
            )
            retrainer.check_and_retrain(features, labels, force=True)

        # Should see raw feature names (feature_a, feature_b, feature_c)
        assert 'feature_a' in trained_cols

    def test_pipeline_transforms_features_before_training(self):
        """With a pipeline, model trains on pipeline output columns."""
        trained_cols = []

        class RecordingModel(_MockModel):
            def train(self, features_df, labels, **kw):
                trained_cols.extend(list(features_df.columns))
                return super().train(features_df, labels, **kw)

        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=200)
            pipeline = _PrefixPipeline()
            retrainer = ModelRetrainer(
                model_dir=tmpdir,
                model_class=RecordingModel,
                feature_pipeline=pipeline,
                min_samples=10,
            )
            retrainer.check_and_retrain(features, labels, force=True)

        # Pipeline adds 'pipe_' prefix → model should see pipe_feature_a etc.
        assert any(c.startswith('pipe_') for c in trained_cols), \
            f"Expected 'pipe_*' columns, got: {trained_cols}"

    def test_pipeline_applied_to_windowed_slice(self):
        """Pipeline is applied after rolling window, not before — both work correctly."""
        pipeline = _PrefixPipeline()
        trained_shapes = []

        class RecordingModel(_MockModel):
            def train(self, features_df, labels, **kw):
                trained_shapes.append(features_df.shape)
                return super().train(features_df, labels, **kw)

        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=500)
            retrainer = ModelRetrainer(
                model_dir=tmpdir,
                model_class=RecordingModel,
                feature_pipeline=pipeline,
                rolling_window_months=3,  # ~63 rows
                min_samples=10,
                holdout_fraction=0.2,
            )
            retrainer.check_and_retrain(features, labels, force=True)

        # Training shape should reflect windowed + holdout split rows
        assert len(trained_shapes) == 1
        n_train_rows = trained_shapes[0][0]
        assert n_train_rows < 500  # windowing happened

    def test_pipeline_stored_on_retrainer(self):
        pipeline = _IdentityPipeline()
        retrainer = ModelRetrainer(feature_pipeline=pipeline)
        assert retrainer.feature_pipeline is pipeline

    def test_no_pipeline_backward_compat(self):
        """feature_pipeline=None (default) is fully backward-compatible."""
        retrainer = ModelRetrainer()
        assert retrainer.feature_pipeline is None

    def test_pipeline_used_for_trigger_evaluation(self):
        """Drift check uses pipeline-transformed features when pipeline is set."""
        pipeline_calls = []

        class TrackingPipeline:
            def transform(self, df):
                pipeline_calls.append(df.shape)
                return df  # pass through

        with tempfile.TemporaryDirectory() as tmpdir:
            features, labels = _make_data(n=200)
            model = _MockModel(model_dir=tmpdir)
            model.trained = True
            model.feature_names = list(features.columns)
            model.feature_means = np.zeros(len(model.feature_names))
            model.feature_stds = np.ones(len(model.feature_names))
            model.training_stats = {}

            pipeline = TrackingPipeline()
            retrainer = ModelRetrainer(
                model_dir=tmpdir,
                feature_pipeline=pipeline,
                min_samples=10,
            )
            # Fresh model → no trigger → early return, but pipeline was called
            retrainer.check_and_retrain(features, labels, current_model=model)

        # Pipeline must have been called at least once for trigger evaluation
        assert len(pipeline_calls) >= 1

    def test_pipeline_with_feature_pipeline_class(self):
        """Integration: FeaturePipeline from compass works as the pipeline."""
        from compass.feature_pipeline import FeaturePipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            # Build a raw DataFrame similar to what training_data_combined.csv has
            n = 200
            rng = np.random.RandomState(0)
            raw_df = pd.DataFrame({
                'spy_price': 400 + rng.randn(n) * 5,
                'vix': 15 + rng.rand(n) * 10,
                'rsi_14': 40 + rng.rand(n) * 30,
                'momentum_5d_pct': rng.randn(n),
                'momentum_10d_pct': rng.randn(n),
                'iv_rank': rng.rand(n) * 100,
                'net_credit': rng.rand(n) * 5,
                'spread_width': rng.rand(n) * 10 + 5,
                'max_loss_per_unit': rng.rand(n) * 8 + 2,
                'contracts': rng.randint(1, 10, n).astype(float),
            })
            labels = rng.randint(0, 2, n)

            trained_cols = []

            class RecordingModel(_MockModel):
                def train(self, features_df, labels_arr, **kw):
                    trained_cols.extend(list(features_df.columns))
                    return super().train(features_df, labels_arr, **kw)

            pipeline = FeaturePipeline(categorical_features=[])
            retrainer = ModelRetrainer(
                model_dir=tmpdir,
                model_class=RecordingModel,
                feature_pipeline=pipeline,
                min_samples=10,
            )
            retrainer.check_and_retrain(raw_df, labels, force=True)

        # Pipeline should have produced vix_zscore, spy_price_zscore, etc.
        assert 'vix_zscore' in trained_cols, \
            f"Expected pipeline feature 'vix_zscore', got: {trained_cols[:10]}"
        assert 'spy_price_zscore' in trained_cols
        assert 'spy_price' not in trained_cols, \
            "Raw 'spy_price' should not appear — pipeline replaces it with z-score"
