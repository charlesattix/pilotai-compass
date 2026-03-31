"""
Tests for GAP-6/7/8 ensemble model integration.

GAP-6: compass/ml_strategy.py RegimeModelRouter loads EnsembleSignalModel
        when path starts with "ensemble_model_".
GAP-7: ml/regime_model_router.py _load_model() delegates to EnsembleSignalModel
        for ensemble files; _score_features() uses predict() API for our model
        classes and predict_proba() for raw sklearn classifiers.
GAP-8: compass/__init__.py exports EnsembleSignalModel.
"""

import importlib
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# GAP-8: compass.__init__ exports EnsembleSignalModel
# ---------------------------------------------------------------------------

class TestGap8CompassInit:
    def test_ensemble_signal_model_exported(self):
        """compass package must export EnsembleSignalModel."""
        import compass
        assert hasattr(compass, "EnsembleSignalModel"), (
            "EnsembleSignalModel missing from compass.__init__ exports"
        )

    def test_ensemble_signal_model_in_all(self):
        """EnsembleSignalModel must appear in compass.__all__."""
        import compass
        assert "EnsembleSignalModel" in compass.__all__

    def test_ensemble_and_signal_model_same_interface(self):
        """Both SignalModel and EnsembleSignalModel must be importable from compass."""
        import compass
        assert hasattr(compass, "SignalModel")
        assert hasattr(compass, "EnsembleSignalModel")


# ---------------------------------------------------------------------------
# GAP-6: compass/ml_strategy.py RegimeModelRouter ensemble loading
# ---------------------------------------------------------------------------

class TestGap6RegimeModelRouterInMlStrategy:
    """Tests for the RegimeModelRouter defined in compass/ml_strategy.py."""

    def _make_mock_signal_model(self):
        m = MagicMock()
        m.trained = True
        m.predict.return_value = {"probability": 0.7, "confidence": 0.4, "prediction": 1}
        m.load.return_value = True
        return m

    def _make_mock_ensemble_model(self):
        m = MagicMock()
        m.trained = True
        m.predict.return_value = {"probability": 0.8, "confidence": 0.6, "prediction": 1}
        m.load.return_value = True
        return m

    def test_signal_model_path_loads_signal_model(self, tmp_path):
        """Paths named signal_model_* must use SignalModel, not EnsembleSignalModel."""
        from compass.ml_strategy import RegimeModelRouter, _load_model_from_path

        fake_path = str(tmp_path / "signal_model_20260217.joblib")
        Path(fake_path).write_bytes(b"")  # create empty file so basename detection works

        default = self._make_mock_signal_model()

        with patch("compass.ml_strategy.SignalModel") as MockSM, \
             patch("compass.ml_strategy.EnsembleSignalModel") as MockEM:
            mock_instance = MagicMock()
            mock_instance.load.return_value = True
            MockSM.return_value = mock_instance

            result = _load_model_from_path(fake_path)

            MockSM.assert_called_once()
            MockEM.assert_not_called()
            assert result is mock_instance

    def test_ensemble_model_path_loads_ensemble_model(self, tmp_path):
        """Paths named ensemble_model_* must use EnsembleSignalModel."""
        from compass.ml_strategy import _load_model_from_path

        fake_path = str(tmp_path / "ensemble_model_20260217.joblib")
        Path(fake_path).write_bytes(b"")

        with patch("compass.ml_strategy.SignalModel") as MockSM, \
             patch("compass.ml_strategy.EnsembleSignalModel") as MockEM:
            mock_instance = MagicMock()
            mock_instance.load.return_value = True
            MockEM.return_value = mock_instance

            result = _load_model_from_path(fake_path)

            MockEM.assert_called_once()
            MockSM.assert_not_called()
            assert result is mock_instance

    def test_load_failure_returns_none(self, tmp_path):
        """_load_model_from_path returns None when load() returns False."""
        from compass.ml_strategy import _load_model_from_path

        fake_path = str(tmp_path / "signal_model_20260217.joblib")
        Path(fake_path).write_bytes(b"")

        with patch("compass.ml_strategy.SignalModel") as MockSM:
            mock_instance = MagicMock()
            mock_instance.load.return_value = False
            MockSM.return_value = mock_instance

            result = _load_model_from_path(fake_path)
            assert result is None

    def test_regime_router_uses_ensemble_for_ensemble_path(self, tmp_path):
        """RegimeModelRouter.predict() calls ensemble model when loaded for regime."""
        from compass.ml_strategy import RegimeModelRouter

        default = self._make_mock_signal_model()
        ensemble = self._make_mock_ensemble_model()

        ensemble_path = str(tmp_path / "ensemble_model_bull.joblib")

        with patch("compass.ml_strategy._load_model_from_path", return_value=ensemble):
            router = RegimeModelRouter(
                default_model=default,
                regime_model_paths={"bull": ensemble_path},
            )

        result = router.predict({"vix": 15.0}, regime="bull")
        ensemble.predict.assert_called_once_with({"vix": 15.0})
        assert result["probability"] == 0.8

    def test_regime_router_falls_back_to_default(self):
        """RegimeModelRouter falls back to default model for unknown regime."""
        from compass.ml_strategy import RegimeModelRouter

        default = self._make_mock_signal_model()
        router = RegimeModelRouter(default_model=default, regime_model_paths=None)

        result = router.predict({"vix": 18.0}, regime="bear")
        default.predict.assert_called_once_with({"vix": 18.0})
        assert result["probability"] == 0.7

    def test_regime_router_defensive_fallback(self):
        """bear/high_vol regimes use 'defensive' model when present."""
        from compass.ml_strategy import RegimeModelRouter

        default = self._make_mock_signal_model()
        defensive = self._make_mock_ensemble_model()

        with patch("compass.ml_strategy._load_model_from_path", return_value=defensive):
            router = RegimeModelRouter(
                default_model=default,
                regime_model_paths={"defensive": "ensemble_model_defensive.joblib"},
            )

        # Both bear and high_vol should route to the defensive model
        for regime in ("bear", "high_vol"):
            defensive.predict.reset_mock()
            router.predict({"vix": 30.0}, regime=regime)
            defensive.predict.assert_called_once()


# ---------------------------------------------------------------------------
# GAP-7: ml/regime_model_router.py _load_model and _score_features
# ---------------------------------------------------------------------------

class TestGap7MlRegimeModelRouter:
    """Tests for RegimeModelRouter in ml/regime_model_router.py."""

    def _import_router(self):
        """Import via importlib to avoid compass.__init__ side effects."""
        spec = importlib.util.spec_from_file_location(
            "ml.regime_model_router",
            Path(__file__).resolve().parent.parent / "ml" / "regime_model_router.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.RegimeModelRouter

    def test_load_ensemble_model_delegates_to_ensemble_class(self, tmp_path):
        """_load_model() uses EnsembleSignalModel for ensemble_model_* files."""
        RegimeModelRouter = self._import_router()

        ensemble_file = tmp_path / "ensemble_model_20260217.joblib"
        ensemble_file.write_bytes(b"")

        mock_ensemble = MagicMock()
        mock_ensemble.trained = True
        mock_ensemble.load.return_value = True

        with patch("compass.ensemble_signal_model.EnsembleSignalModel", return_value=mock_ensemble):
            router = RegimeModelRouter.__new__(RegimeModelRouter)
            router._use_model = True
            result = router._load_model(ensemble_file)

        # Should have tried to load via EnsembleSignalModel, not raw joblib
        mock_ensemble.load.assert_called_once_with(ensemble_file.name)
        assert result is mock_ensemble

    def test_load_legacy_model_uses_joblib(self, tmp_path):
        """_load_model() uses raw joblib.load for signal_model_* (legacy) files."""
        RegimeModelRouter = self._import_router()

        legacy_file = tmp_path / "signal_model_20260217.joblib"
        legacy_file.write_bytes(b"")

        fake_model = MagicMock(spec=["predict_proba", "predict"])

        with patch("joblib.load", return_value=fake_model) as mock_joblib:
            router = RegimeModelRouter.__new__(RegimeModelRouter)
            router._use_model = True
            result = router._load_model(legacy_file)

        mock_joblib.assert_called_once_with(str(legacy_file))
        assert result is fake_model

    def test_load_missing_file_returns_none(self, tmp_path):
        """_load_model() returns None when file does not exist."""
        RegimeModelRouter = self._import_router()

        missing = tmp_path / "ensemble_model_missing.joblib"

        router = RegimeModelRouter.__new__(RegimeModelRouter)
        router._use_model = True
        result = router._load_model(missing)

        assert result is None

    def test_score_features_uses_predict_api_for_model_classes(self):
        """_score_features() calls model.predict() when model has 'trained' attribute."""
        RegimeModelRouter = self._import_router()

        router = RegimeModelRouter.__new__(RegimeModelRouter)
        router._use_model = False
        router._model = MagicMock()
        router._model.trained = True
        router._model.predict.return_value = {"probability": 0.72, "confidence": 0.44}

        prob = router._score_features({"vix": 18.0, "rsi": 55.0})

        router._model.predict.assert_called_once_with({"vix": 18.0, "rsi": 55.0})
        assert abs(prob - 0.72) < 1e-6

    def test_score_features_uses_predict_proba_for_raw_sklearn(self):
        """_score_features() calls predict_proba for raw sklearn classifiers."""
        import numpy as np
        RegimeModelRouter = self._import_router()

        router = RegimeModelRouter.__new__(RegimeModelRouter)
        router._use_model = False
        # Simulate a raw sklearn model (no 'trained' attribute)
        raw_model = MagicMock(spec=["predict_proba", "predict"])
        raw_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        router._model = raw_model

        prob = router._score_features({"a": 1.0, "b": 2.0})

        raw_model.predict_proba.assert_called_once()
        assert abs(prob - 0.7) < 1e-6

    def test_score_features_fallback_to_predict_when_no_predict_proba(self):
        """_score_features() falls back to predict() for classifiers lacking predict_proba."""
        import numpy as np
        RegimeModelRouter = self._import_router()

        router = RegimeModelRouter.__new__(RegimeModelRouter)
        router._use_model = False
        raw_model = MagicMock(spec=["predict"])  # no predict_proba
        raw_model.predict.return_value = np.array([1])
        router._model = raw_model

        prob = router._score_features({"x": 0.5})

        assert prob == 1.0

    def test_get_multiplier_basic_regimes(self):
        """get_multiplier returns correct values for known regimes."""
        RegimeModelRouter = self._import_router()
        router = RegimeModelRouter(config={"use_signal_model": False})

        assert router.get_multiplier("bull") == pytest.approx(1.50)
        assert router.get_multiplier("neutral") == pytest.approx(1.00)
        assert router.get_multiplier("crash") == pytest.approx(0.00)
        assert router.get_multiplier(None) == pytest.approx(1.00)

    def test_is_defensive(self):
        """is_defensive returns True only for bear/high_vol/crash."""
        RegimeModelRouter = self._import_router()
        router = RegimeModelRouter(config={"use_signal_model": False})

        assert router.is_defensive("bear")
        assert router.is_defensive("high_vol")
        assert router.is_defensive("crash")
        assert not router.is_defensive("bull")
        assert not router.is_defensive("neutral")


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_signal_model_still_importable_directly(self):
        """SignalModel must still be importable from compass.signal_model."""
        from compass.signal_model import SignalModel
        assert SignalModel is not None

    def test_ml_strategy_signal_model_compat(self):
        """RegimeModelRouter in ml_strategy accepts a plain SignalModel as default."""
        from compass.ml_strategy import RegimeModelRouter
        from compass.signal_model import SignalModel

        sm = MagicMock(spec=SignalModel)
        sm.predict.return_value = {"probability": 0.6, "confidence": 0.2, "prediction": 1}

        router = RegimeModelRouter(default_model=sm, regime_model_paths=None)
        result = router.predict({"vix": 20.0})
        sm.predict.assert_called_once()
