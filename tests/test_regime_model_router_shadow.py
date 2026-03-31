"""
Tests for shadow_ensemble integration in RegimeModelRouter.

Covers:
  - shadow_ensemble=false (default): no wrapping, behaviour unchanged
  - shadow_ensemble=true, ensemble model present: primary wrapped in ShadowEnsemble
  - shadow_ensemble=true, no ensemble model: graceful fallback to primary only
  - shadow_ensemble=true, ensemble load fails: graceful fallback to primary only
  - _score_features uses primary result when wrapped in ShadowEnsemble
  - get_multiplier_with_metadata ml_confidence reflects primary model output
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest

from ml.regime_model_router import RegimeModelRouter
from shared.types import PredictionResult


# ─── Stubs ────────────────────────────────────────────────────────────────────


def _make_prediction(prob: float = 0.75, signal: str = "bullish") -> PredictionResult:
    return {
        "prediction": int(prob > 0.5),
        "probability": prob,
        "confidence": prob,
        "signal": signal,
        "signal_strength": prob,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "fallback": False,
    }


class _StubPrimary:
    """Minimal SignalModel-like stub."""

    trained = True
    feature_names = ["a", "b"]
    training_stats: Dict = {"test_auc": 0.82, "timestamp": "2026-01-01T00:00:00+00:00"}
    feature_means: Optional[np.ndarray] = None
    feature_stds: Optional[np.ndarray] = None

    def __init__(self, prob: float = 0.70) -> None:
        self._prob = prob
        self.predict_calls: list = []
        self.predict_batch_calls: int = 0

    def predict(self, features: Dict) -> PredictionResult:
        self.predict_calls.append(features)
        return _make_prediction(self._prob)

    def predict_batch(self, features_df: Any) -> np.ndarray:
        self.predict_batch_calls += 1
        n = len(features_df)
        return np.full(n, self._prob)

    def train(self, *a, **kw):
        return {}

    def save(self, *a, **kw):
        pass

    def load(self, *a, **kw):
        return True

    def backtest(self, *a, **kw):
        return {}

    def get_fallback_stats(self) -> Dict:
        return {}


class _StubShadow(_StubPrimary):
    """Shadow model stub — slightly different probability to verify wrapping."""

    def __init__(self, prob: float = 0.55) -> None:
        super().__init__(prob)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _router_no_model() -> RegimeModelRouter:
    """Router with ML disabled — pure regime table."""
    return RegimeModelRouter({"use_signal_model": False})


def _router_with_primary(primary: Any, shadow_ensemble: bool = False) -> RegimeModelRouter:
    """Create a router whose _model is set directly (avoids disk I/O)."""
    router = _router_no_model()
    router._use_model = True
    router._shadow_ensemble = shadow_ensemble
    router._model = primary
    return router


# ─── Tests: shadow_ensemble=false (default) ───────────────────────────────────


class TestShadowDisabledDefault:
    """shadow_ensemble defaults to false — no wrapping happens."""

    def test_shadow_ensemble_flag_defaults_false(self):
        router = RegimeModelRouter({"use_signal_model": False})
        assert router._shadow_ensemble is False

    def test_model_is_not_shadow_when_disabled(self):
        """Even if a primary model is loaded, it should not be wrapped."""
        primary = _StubPrimary()
        router = _router_with_primary(primary, shadow_ensemble=False)
        # _model should be the raw primary, not a ShadowEnsemble
        from compass.shadow_ensemble import ShadowEnsemble
        assert not isinstance(router._model, ShadowEnsemble)
        assert router._model is primary

    def test_score_features_returns_primary_prob(self):
        primary = _StubPrimary(prob=0.80)
        router = _router_with_primary(primary, shadow_ensemble=False)
        prob = router._score_features({"a": 1.0})
        assert prob == pytest.approx(0.80)

    def test_multiplier_metadata_reflects_primary(self):
        primary = _StubPrimary(prob=0.80)
        router = _router_with_primary(primary, shadow_ensemble=False)
        meta = router.get_multiplier_with_metadata("bull", ml_features={"a": 1.0})
        assert meta["ml_confidence"] is not None
        # 0.80 → adj = 0.25 * (0.80 - 0.5) * 2 = 0.15 → 1.50 * 1.15 = 1.725 → capped at 1.50
        assert meta["multiplier"] == pytest.approx(1.50)


# ─── Tests: shadow_ensemble=true, ensemble model present ─────────────────────


class TestShadowEnabledWithModel:
    """shadow_ensemble=true + ensemble model available → ShadowEnsemble wrapping."""

    def _make_router_with_shadow(self, primary_prob=0.70, shadow_prob=0.55):
        """Build a router whose _model is a real ShadowEnsemble with stubs."""
        from compass.shadow_ensemble import ShadowEnsemble

        primary = _StubPrimary(prob=primary_prob)
        shadow = _StubShadow(prob=shadow_prob)

        with tempfile.TemporaryDirectory() as tmpdir:
            wrapped = ShadowEnsemble(
                primary, shadow,
                log_path=Path(tmpdir) / "shadow_log.csv",
            )
        router = _router_no_model()
        router._use_model = True
        router._shadow_ensemble = True
        router._model = wrapped
        return router, primary, shadow

    def test_model_is_shadow_ensemble(self):
        from compass.shadow_ensemble import ShadowEnsemble
        router, _, _ = self._make_router_with_shadow()
        assert isinstance(router._model, ShadowEnsemble)

    def test_score_features_returns_primary_probability(self):
        """Primary controls the probability even when wrapped in ShadowEnsemble."""
        router, primary, _ = self._make_router_with_shadow(primary_prob=0.72)
        prob = router._score_features({"a": 1.0})
        assert prob == pytest.approx(0.72)

    def test_primary_predict_called_exactly_once(self):
        router, primary, _ = self._make_router_with_shadow()
        router._score_features({"x": 0.5})
        assert len(primary.predict_calls) == 1

    def test_shadow_predict_called_exactly_once(self):
        router, _, shadow = self._make_router_with_shadow()
        router._score_features({"x": 0.5})
        assert len(shadow.predict_calls) == 1

    def test_multiplier_uses_primary_confidence_not_shadow(self):
        """Multiplier is derived from primary probability, not shadow."""
        router, primary, shadow = self._make_router_with_shadow(
            primary_prob=1.0, shadow_prob=0.0
        )
        meta = router.get_multiplier_with_metadata("neutral", ml_features={"a": 1})
        # primary=1.0 → adj = 0.25*(1.0-0.5)*2 = 0.25 → 1.00*1.25 = 1.25
        assert meta["ml_confidence"] == pytest.approx(1.0)
        assert meta["multiplier"] == pytest.approx(1.25)

    def test_csv_row_written_per_predict(self):
        """Verify ShadowEnsemble logs a row to CSV on each _score_features call."""
        from compass.shadow_ensemble import ShadowEnsemble

        primary = _StubPrimary(prob=0.70)
        shadow = _StubShadow(prob=0.55)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "shadow_log.csv"
            wrapped = ShadowEnsemble(primary, shadow, log_path=log_path)
            router = _router_no_model()
            router._use_model = True
            router._model = wrapped

            router._score_features({"a": 1.0})
            router._score_features({"a": 2.0})

            import csv
            with open(log_path) as fh:
                rows = list(csv.DictReader(fh))
            assert len(rows) == 2


# ─── Tests: shadow_ensemble=true, no ensemble model on disk ──────────────────


class TestShadowEnabledNoEnsembleModel:
    """When shadow_ensemble=true but no ensemble_model_*.joblib exists, fall back
    gracefully to primary only — no exception, no wrapping."""

    def test_wrap_shadow_returns_primary_when_no_candidate(self):
        primary = _StubPrimary()
        router = _router_no_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = router._wrap_shadow(primary, Path(tmpdir))
        # Should return primary unchanged
        assert result is primary

    def test_model_is_not_shadow_ensemble_after_fallback(self):
        from compass.shadow_ensemble import ShadowEnsemble
        primary = _StubPrimary()
        router = _router_no_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = router._wrap_shadow(primary, Path(tmpdir))
        assert not isinstance(result, ShadowEnsemble)

    def test_warn_logged_when_no_ensemble_candidate(self, caplog):
        import logging
        primary = _StubPrimary()
        router = _router_no_model()
        with caplog.at_level(logging.WARNING, logger="ml.regime_model_router"):
            with tempfile.TemporaryDirectory() as tmpdir:
                router._wrap_shadow(primary, Path(tmpdir))
        assert any("no ensemble_model_*.joblib found" in r.message for r in caplog.records)

    def test_score_features_still_works_after_fallback(self):
        primary = _StubPrimary(prob=0.65)
        router = _router_no_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            router._model = router._wrap_shadow(primary, Path(tmpdir))
        router._use_model = True
        prob = router._score_features({"a": 1.0})
        assert prob == pytest.approx(0.65)


# ─── Tests: shadow_ensemble=true, ensemble load fails ────────────────────────


class TestShadowEnabledLoadFails:
    """When EnsembleSignalModel.load() returns False, fall back to primary only."""

    def test_wrap_shadow_returns_primary_on_load_failure(self):
        from compass.shadow_ensemble import ShadowEnsemble

        primary = _StubPrimary()
        router = _router_no_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake ensemble_model file so the glob finds it
            fake = Path(tmpdir) / "ensemble_model_20260101_120000.joblib"
            fake.write_bytes(b"not_a_real_model")

            with patch(
                "compass.ensemble_signal_model.EnsembleSignalModel.load",
                return_value=False,
            ):
                result = router._wrap_shadow(primary, Path(tmpdir))

        assert result is primary
        assert not isinstance(result, ShadowEnsemble)

    def test_wrap_shadow_returns_primary_on_exception(self):
        from compass.shadow_ensemble import ShadowEnsemble

        primary = _StubPrimary()
        router = _router_no_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            fake = Path(tmpdir) / "ensemble_model_20260101_120000.joblib"
            fake.write_bytes(b"corrupt")

            with patch(
                "compass.ensemble_signal_model.EnsembleSignalModel.load",
                side_effect=RuntimeError("corrupted"),
            ):
                result = router._wrap_shadow(primary, Path(tmpdir))

        assert result is primary
        assert not isinstance(result, ShadowEnsemble)


# ─── Tests: _wrap_shadow picks latest file when multiple candidates ───────────


class TestWrapShadowPicksLatest:
    """_wrap_shadow should pick the most-recently-modified ensemble_model_*.joblib."""

    def test_latest_file_selected(self):
        from compass.shadow_ensemble import ShadowEnsemble

        primary = _StubPrimary()
        router = _router_no_model()

        loaded_names: list = []

        def fake_load(self_ens, filename: str) -> bool:
            loaded_names.append(filename)
            return False  # prevent full wrapping; we just want to check which was picked

        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir)
            older = p / "ensemble_model_20260101_000000.joblib"
            newer = p / "ensemble_model_20260301_000000.joblib"
            older.write_bytes(b"x")
            newer.write_bytes(b"x")
            # Make newer actually newer
            import os, time
            os.utime(older, (time.time() - 100, time.time() - 100))
            os.utime(newer, (time.time(), time.time()))

            with patch(
                "compass.ensemble_signal_model.EnsembleSignalModel.load",
                new=fake_load,
            ):
                router._wrap_shadow(primary, p)

        assert loaded_names, "load() should have been called"
        assert loaded_names[0] == "ensemble_model_20260301_000000.joblib"


# ─── Tests: __init__ wires shadow mode end-to-end (no disk model) ────────────


class TestInitShadowFlag:
    """Verify __init__ reads shadow_ensemble config key and stores it correctly."""

    def test_shadow_ensemble_false_by_default(self):
        router = RegimeModelRouter({"use_signal_model": False})
        assert router._shadow_ensemble is False

    def test_shadow_ensemble_true_when_configured(self):
        router = RegimeModelRouter(
            {"use_signal_model": False, "shadow_ensemble": True}
        )
        assert router._shadow_ensemble is True

    def test_shadow_ensemble_false_explicit(self):
        router = RegimeModelRouter(
            {"use_signal_model": False, "shadow_ensemble": False}
        )
        assert router._shadow_ensemble is False

    def test_shadow_wrapping_skipped_when_use_signal_model_false(self):
        """shadow_ensemble=True has no effect when use_signal_model=False."""
        from compass.shadow_ensemble import ShadowEnsemble
        router = RegimeModelRouter(
            {"use_signal_model": False, "shadow_ensemble": True}
        )
        assert router._model is None
        # Nothing to wrap — no ShadowEnsemble should appear
        assert not isinstance(router._model, ShadowEnsemble)
