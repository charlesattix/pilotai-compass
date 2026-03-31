"""Tests for compass.pnl_predictor – pre-trade P&L prediction engine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.pnl_predictor import (
    DEFAULT_FEATURES,
    CalibrationBin,
    FeatureImportance,
    ModelMetrics,
    PnLPredictor,
    Prediction,
    PredictionLogEntry,
    PredictorResult,
    _z_score,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_data(n: int = 300, seed: int = 42) -> tuple:
    """Synthetic features with P&L that depends on VIX and momentum."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    vix = 18 + rng.randn(n) * 5
    vts = 1.05 + rng.randn(n) * 0.1
    r1d = rng.randn(n) * 0.01
    r5d = rng.randn(n) * 0.02
    r20d = rng.randn(n) * 0.04
    vol_ratio = 1.0 + rng.randn(n) * 0.3
    iv_rank = rng.rand(n) * 100
    sw = 5.0 + rng.randn(n) * 1.0

    features = pd.DataFrame({
        "vix": vix, "vix_term_structure": vts,
        "return_1d": r1d, "return_5d": r5d, "return_20d": r20d,
        "volume_ratio": vol_ratio, "iv_rank": iv_rank, "spread_width": sw,
    }, index=idx)

    # PnL depends on features + noise
    pnl = -0.5 * (vix - 20) + 2.0 * r5d * 100 + 0.3 * iv_rank + rng.randn(n) * 10
    pnl_series = pd.Series(pnl, index=idx, name="pnl")
    return features, pnl_series


def _fitted_predictor() -> tuple:
    feats, pnl = _make_data()
    pred = PnLPredictor(n_estimators=30, max_depth=3)
    result = pred.fit(feats, pnl)
    return pred, result


# ── Constructor ─────────────────────────────────────────────────────────────
class TestInit:
    def test_defaults(self):
        p = PnLPredictor()
        assert p.features == DEFAULT_FEATURES
        assert p.go_threshold == 0.5
        assert p.confidence_level == 0.80

    def test_custom(self):
        p = PnLPredictor(features=["vix", "return_1d"], go_threshold=1.0, confidence_level=0.95)
        assert len(p.features) == 2
        assert p.go_threshold == 1.0

    def test_model_params(self):
        p = PnLPredictor(n_estimators=50, max_depth=3)
        assert p.n_estimators == 50


# ── Fit ─────────────────────────────────────────────────────────────────────
class TestFit:
    def test_returns_result(self):
        _, result = _fitted_predictor()
        assert isinstance(result, PredictorResult)

    def test_metrics_present(self):
        _, result = _fitted_predictor()
        assert result.metrics is not None

    def test_importances_present(self):
        _, result = _fitted_predictor()
        assert len(result.importances) > 0

    def test_calibration_present(self):
        _, result = _fitted_predictor()
        assert len(result.calibration) > 0

    def test_log_present(self):
        _, result = _fitted_predictor()
        assert len(result.prediction_log) > 0
        assert result.n_predictions > 0

    def test_generated_at(self):
        _, result = _fitted_predictor()
        assert len(result.generated_at) > 0

    def test_too_few_samples(self):
        feats = pd.DataFrame({"vix": [20.0] * 5}, index=pd.date_range("2024-01-01", periods=5))
        pnl = pd.Series([10.0] * 5, index=feats.index)
        result = PnLPredictor().fit(feats, pnl)
        assert result.metrics is None

    def test_missing_features_handled(self):
        feats, pnl = _make_data(n=100)
        feats = feats[["vix", "return_1d"]]  # only 2 of 8 features
        result = PnLPredictor().fit(feats, pnl)
        assert result.metrics is not None


# ── Metrics ─────────────────────────────────────────────────────────────────
class TestMetrics:
    def test_mae_positive(self):
        _, result = _fitted_predictor()
        assert result.metrics.mae > 0

    def test_rmse_geq_mae(self):
        _, result = _fitted_predictor()
        assert result.metrics.rmse >= result.metrics.mae - 1e-9

    def test_direction_accuracy_bounded(self):
        _, result = _fitted_predictor()
        assert 0.0 <= result.metrics.direction_accuracy <= 1.0

    def test_go_precision_bounded(self):
        _, result = _fitted_predictor()
        assert 0.0 <= result.metrics.go_precision <= 1.0

    def test_train_test_counts(self):
        _, result = _fitted_predictor()
        assert result.metrics.n_train > 0
        assert result.metrics.n_test > 0
        assert result.metrics.n_train > result.metrics.n_test


# ── Importances ─────────────────────────────────────────────────────────────
class TestImportances:
    def test_ranked(self):
        _, result = _fitted_predictor()
        ranks = [f.rank for f in result.importances]
        assert ranks == sorted(ranks)

    def test_sum_to_one(self):
        _, result = _fitted_predictor()
        total = sum(f.importance for f in result.importances)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_vix_important(self):
        """VIX is a key driver in our synthetic data."""
        _, result = _fitted_predictor()
        top3 = {f.feature for f in result.importances[:3]}
        # VIX or iv_rank should appear in top 3
        assert len(top3 & {"vix", "iv_rank", "return_5d"}) > 0


# ── Calibration ─────────────────────────────────────────────────────────────
class TestCalibration:
    def test_bins_populated(self):
        _, result = _fitted_predictor()
        assert len(result.calibration) > 0

    def test_bin_fields(self):
        _, result = _fitted_predictor()
        for b in result.calibration:
            assert b.n_obs > 0
            assert isinstance(b.avg_predicted, float)
            assert isinstance(b.avg_actual, float)

    def test_monotonic_predicted(self):
        """Bin midpoints should be monotonically increasing."""
        _, result = _fitted_predictor()
        mids = [b.bin_mid for b in result.calibration]
        assert mids == sorted(mids)


# ── Prediction ──────────────────────────────────────────────────────────────
class TestPredict:
    def test_returns_prediction(self):
        pred, _ = _fitted_predictor()
        p = pred.predict({"vix": 20.0, "return_1d": 0.01, "return_5d": 0.02,
                          "return_20d": 0.03, "volume_ratio": 1.0, "iv_rank": 50.0,
                          "spread_width": 5.0, "vix_term_structure": 1.05})
        assert isinstance(p, Prediction)

    def test_confidence_interval(self):
        pred, _ = _fitted_predictor()
        p = pred.predict({"vix": 20.0, "iv_rank": 50.0, "return_5d": 0.01})
        assert p.confidence_low < p.predicted_pnl
        assert p.confidence_high > p.predicted_pnl

    def test_go_decision_bool(self):
        pred, _ = _fitted_predictor()
        p = pred.predict({"vix": 20.0, "iv_rank": 50.0})
        assert isinstance(p.go_decision, bool)

    def test_recommended_size_bounded(self):
        pred, _ = _fitted_predictor()
        p = pred.predict({"vix": 20.0, "iv_rank": 50.0})
        assert 0.0 <= p.recommended_size <= 1.0

    def test_no_model_returns_zero(self):
        p = PnLPredictor().predict({"vix": 20.0})
        assert p.predicted_pnl == 0.0
        assert not p.go_decision

    def test_features_preserved(self):
        pred, _ = _fitted_predictor()
        feats = {"vix": 25.0, "iv_rank": 60.0}
        p = pred.predict(feats)
        assert p.feature_values == feats

    def test_high_vix_lower_pnl(self):
        """Higher VIX should predict lower P&L (negative coefficient in synthetic data)."""
        pred, _ = _fitted_predictor()
        base = {"return_5d": 0.0, "iv_rank": 50.0, "vix_term_structure": 1.0,
                "return_1d": 0.0, "return_20d": 0.0, "volume_ratio": 1.0, "spread_width": 5.0}
        low_vix = pred.predict({**base, "vix": 12.0})
        high_vix = pred.predict({**base, "vix": 35.0})
        assert low_vix.predicted_pnl > high_vix.predicted_pnl


# ── Position sizing integration ─────────────────────────────────────────────
class TestPositionSizing:
    def test_go_positive_size(self):
        pred, _ = _fitted_predictor()
        # High iv_rank + low vix should produce go
        p = pred.predict({"vix": 12.0, "iv_rank": 80.0, "return_5d": 0.05})
        if p.go_decision:
            assert p.recommended_size > 0

    def test_nogo_zero_size(self):
        pred, _ = _fitted_predictor()
        p = pred.predict({"vix": 50.0, "iv_rank": 5.0, "return_5d": -0.10})
        if not p.go_decision:
            assert p.recommended_size == 0.0


# ── Record actual ───────────────────────────────────────────────────────────
class TestRecordActual:
    def test_record_adds_to_log(self):
        pred, _ = _fitted_predictor()
        n_before = len(pred.get_log())
        pred.record_actual(predicted_pnl=10.0, actual_pnl=12.0, go=True)
        assert len(pred.get_log()) == n_before + 1

    def test_correct_classification(self):
        pred, _ = _fitted_predictor()
        pred.record_actual(10.0, 15.0, go=True)
        assert pred.get_log()[-1].was_correct  # go + profitable

    def test_incorrect_classification(self):
        pred, _ = _fitted_predictor()
        pred.record_actual(10.0, -5.0, go=True)
        assert not pred.get_log()[-1].was_correct  # go + unprofitable


# ── Z-score utility ─────────────────────────────────────────────────────────
class TestZScore:
    def test_known_values(self):
        assert _z_score(0.80) == pytest.approx(1.282)
        assert _z_score(0.95) == pytest.approx(1.960)
        assert _z_score(0.99) == pytest.approx(2.576)

    def test_increases_with_confidence(self):
        assert _z_score(0.90) < _z_score(0.95) < _z_score(0.99)

    def test_positive(self):
        assert _z_score(0.80) > 0


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred, result = _fitted_predictor()
            path = pred.generate_report(result, output_path=Path(tmp) / "pp.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred, result = _fitted_predictor()
            path = pred.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "P&L Predictor" in html
            assert "Feature Importance" in html
            assert "Calibration" in html
            assert "Prediction Log" in html
            assert "Model Performance" in html

    def test_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            pred, result = _fitted_predictor()
            path = pred.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_prediction(self):
        p = Prediction(10.0, 5.0, 15.0, 0.80, True, 1.5, 0.75)
        assert p.go_decision is True

    def test_calibration_bin(self):
        b = CalibrationBin(5.0, 4.8, 5.2, 20)
        assert b.n_obs == 20

    def test_feature_importance(self):
        f = FeatureImportance("vix", 0.35, 1)
        assert f.rank == 1

    def test_model_metrics(self):
        m = ModelMetrics(mae=5.0, rmse=7.0, r_squared=0.5, direction_accuracy=0.65)
        assert m.direction_accuracy == 0.65

    def test_result_defaults(self):
        r = PredictorResult()
        assert r.metrics is None
        assert r.importances == []
        assert r.n_predictions == 0
