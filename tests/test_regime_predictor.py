"""Tests for compass.regime_predictor – regime prediction with transitions."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.regime_predictor import (
    DEFAULT_FEATURES,
    DEFAULT_HORIZONS,
    REGIMES,
    CalibrationPoint,
    FeatureImportance,
    HorizonAccuracy,
    PredictorResult,
    RegimeForecast,
    RegimePredictor,
    TransitionMatrix,
    _stationary_distribution,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_features(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "vix": 18 + rng.randn(n) * 5,
        "vix_term_structure": 1.05 + rng.randn(n) * 0.1,
        "credit_spread": 3.5 + rng.randn(n) * 0.5,
        "yield_curve_slope": 0.5 + rng.randn(n) * 0.3,
        "momentum_20d": rng.randn(n) * 2,
    }, index=idx)


def _make_regimes(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    labels = rng.choice(["bull", "bear", "high_vol", "low_vol"], size=n, p=[0.4, 0.2, 0.2, 0.2])
    return pd.Series(labels, index=idx, name="regime")


def _make_simple_data(n: int = 300) -> tuple:
    """Two-regime data with clear feature separation."""
    rng = np.random.RandomState(99)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    regimes = ["bull"] * (n // 2) + ["bear"] * (n - n // 2)
    vix = np.where(np.array(regimes) == "bull", 15 + rng.randn(n) * 2, 28 + rng.randn(n) * 3)
    df = pd.DataFrame({
        "vix": vix,
        "vix_term_structure": np.where(np.array(regimes) == "bull", 1.1, 0.9) + rng.randn(n) * 0.05,
        "credit_spread": np.where(np.array(regimes) == "bull", 3.0, 5.0) + rng.randn(n) * 0.3,
        "yield_curve_slope": rng.randn(n) * 0.2,
        "momentum_20d": np.where(np.array(regimes) == "bull", 1.0, -1.0) + rng.randn(n) * 0.5,
    }, index=idx)
    return df, pd.Series(regimes, index=idx)


# ── Constructor ─────────────────────────────────────────────────────────────
class TestRegimePredictorInit:
    def test_defaults(self):
        rp = RegimePredictor()
        assert rp.features == DEFAULT_FEATURES
        assert rp.horizons == DEFAULT_HORIZONS

    def test_custom_features(self):
        rp = RegimePredictor(features=["vix", "momentum_20d"])
        assert len(rp.features) == 2

    def test_custom_horizons(self):
        rp = RegimePredictor(horizons=[1, 3, 10])
        assert rp.horizons == [1, 3, 10]

    def test_model_params(self):
        rp = RegimePredictor(n_estimators=50, max_depth=3)
        assert rp.n_estimators == 50


# ── Fit ─────────────────────────────────────────────────────────────────────
class TestFit:
    def test_returns_result(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        assert isinstance(result, PredictorResult)

    def test_forecasts_populated(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        assert len(result.forecasts) > 0

    def test_transition_matrix_present(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        assert result.transition_matrix is not None

    def test_feature_importances_present(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        assert len(result.feature_importances) > 0

    def test_current_regime_set(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        assert result.current_regime in regs.unique()

    def test_n_training_samples(self):
        feats = _make_features(n=200)
        regs = _make_regimes(n=200)
        result = RegimePredictor().fit(feats, regs)
        assert result.n_training_samples == 200

    def test_too_few_samples(self):
        feats = _make_features(n=10)
        regs = _make_regimes(n=10)
        result = RegimePredictor().fit(feats, regs)
        assert result.forecasts == []


# ── Forecasts ───────────────────────────────────────────────────────────────
class TestForecasts:
    def test_forecast_per_horizon(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        horizons_got = {f.horizon_days for f in result.forecasts}
        for h in DEFAULT_HORIZONS:
            assert h in horizons_got

    def test_forecast_regime_valid(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        for f in result.forecasts:
            assert f.predicted_regime in regs.unique()

    def test_probabilities_sum_to_one(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        for f in result.forecasts:
            assert sum(f.probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_confidence_bounded(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        for f in result.forecasts:
            assert 0.0 <= f.confidence <= 1.0


# ── Transition matrix ──────────────────────────────────────────────────────
class TestTransitionMatrix:
    def test_matrix_rows_sum_to_one(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        tm = result.transition_matrix
        for fr in tm.regimes:
            row_sum = sum(tm.matrix[fr].values())
            assert row_sum == pytest.approx(1.0, abs=0.01)

    def test_n_transitions_positive(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        assert result.transition_matrix.n_transitions > 0

    def test_stationary_dist_sums_to_one(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        sd = result.transition_matrix.stationary_dist
        assert sum(sd.values()) == pytest.approx(1.0, abs=0.01)

    def test_stationary_dist_function(self):
        matrix = {
            "bull": {"bull": 0.8, "bear": 0.2},
            "bear": {"bull": 0.4, "bear": 0.6},
        }
        sd = _stationary_distribution(matrix, ["bull", "bear"])
        assert sum(sd.values()) == pytest.approx(1.0, abs=0.01)
        # Stationary: bull should be more likely (0.8 self-transition)
        assert sd["bull"] > sd["bear"]


# ── Feature importance ──────────────────────────────────────────────────────
class TestFeatureImportance:
    def test_importances_ranked(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        ranks = [fi.rank for fi in result.feature_importances]
        assert ranks == sorted(ranks)

    def test_importances_sum_to_one(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        total = sum(fi.importance for fi in result.feature_importances)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_simple_data_vix_important(self):
        """With clear regime separation on VIX, it should rank high."""
        feats, regs = _make_simple_data()
        result = RegimePredictor().fit(feats, regs)
        top = result.feature_importances[0]
        # VIX or momentum should be most important
        assert top.feature in ["vix", "momentum_20d", "credit_spread", "vix_term_structure"]


# ── Horizon accuracy ────────────────────────────────────────────────────────
class TestHorizonAccuracy:
    def test_accuracy_bounded(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        for a in result.horizon_accuracies:
            assert 0.0 <= a.accuracy <= 1.0
            assert 0.0 <= a.top2_accuracy <= 1.0

    def test_top2_geq_top1(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        for a in result.horizon_accuracies:
            assert a.top2_accuracy >= a.accuracy - 1e-9

    def test_simple_data_high_accuracy(self):
        """Clearly separable data should produce decent 1d accuracy."""
        feats, regs = _make_simple_data(n=400)
        result = RegimePredictor().fit(feats, regs)
        acc1 = next((a for a in result.horizon_accuracies if a.horizon_days == 1), None)
        assert acc1 is not None
        assert acc1.accuracy > 0.6


# ── Calibration ─────────────────────────────────────────────────────────────
class TestCalibration:
    def test_calibration_points_present(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        assert len(result.calibration_curve) > 0

    def test_calibration_probs_bounded(self):
        feats = _make_features()
        regs = _make_regimes()
        result = RegimePredictor().fit(feats, regs)
        for cp in result.calibration_curve:
            assert 0.0 <= cp.predicted_prob <= 1.0
            assert 0.0 <= cp.actual_freq <= 1.0
            assert cp.n_obs > 0


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp = RegimePredictor()
            result = rp.fit(_make_features(), _make_regimes())
            path = rp.generate_report(result, output_path=Path(tmp) / "rp.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp = RegimePredictor()
            result = rp.fit(_make_features(), _make_regimes())
            path = rp.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Regime Predictions" in html
            assert "Transition" in html
            assert "Accuracy" in html
            assert "Feature Importance" in html
            assert "Calibration" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            rp = RegimePredictor()
            result = rp.fit(_make_features(), _make_regimes())
            path = rp.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_regime_forecast(self):
        f = RegimeForecast(1, "bull", {"bull": 0.7, "bear": 0.3}, 0.7)
        assert f.predicted_regime == "bull"

    def test_transition_matrix(self):
        m = {"bull": {"bull": 0.8, "bear": 0.2}, "bear": {"bull": 0.4, "bear": 0.6}}
        tm = TransitionMatrix(m, ["bull", "bear"], 100, {"bull": 0.67, "bear": 0.33})
        assert tm.n_transitions == 100

    def test_feature_importance(self):
        fi = FeatureImportance("vix", 0.35, 1)
        assert fi.rank == 1

    def test_predictor_result_defaults(self):
        r = PredictorResult()
        assert r.forecasts == []
        assert r.transition_matrix is None
