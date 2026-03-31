"""Tests for compass/regime_forecast.py — regime forecaster.

Covers:
  - build_transition_matrix: row-stochastic, smoothing, known sequences
  - multi_step_forecast: 1-step, multi-step, sums to 1
  - compute_macro_adjustments: VIX, term structure, yield curve, momentum
  - apply_adjustments: normalization, clamping
  - calibrate: accuracy, Brier score, confusion matrix
  - RegimeForecaster: fit, predict, forecast_all_horizons, calibration
  - RegimeForecast dataclass: fields
  - HTML report generation
"""

from __future__ import annotations

import numpy as np
import pytest

from compass.regime_forecast import (
    REGIME_NAMES,
    REGIME_TO_IDX,
    N_REGIMES,
    CalibrationResult,
    RegimeForecast,
    RegimeForecaster,
    apply_adjustments,
    build_transition_matrix,
    calibrate,
    compute_macro_adjustments,
    multi_step_forecast,
)


# ── build_transition_matrix ──────────────────────────────────────────────


class TestBuildTransitionMatrix:
    def test_row_stochastic(self):
        seq = ["bull"] * 50 + ["bear"] * 20 + ["bull"] * 30
        M = build_transition_matrix(seq)
        for i in range(N_REGIMES):
            assert M[i].sum() == pytest.approx(1.0, abs=1e-6)

    def test_shape(self):
        M = build_transition_matrix(["bull", "bear", "bull"])
        assert M.shape == (N_REGIMES, N_REGIMES)

    def test_dominant_transition(self):
        seq = ["bull", "bull", "bull", "bull", "bear", "bull"]
        M = build_transition_matrix(seq, smoothing=0.1)
        # bull→bull should dominate
        assert M[REGIME_TO_IDX["bull"], REGIME_TO_IDX["bull"]] > M[REGIME_TO_IDX["bull"], REGIME_TO_IDX["bear"]]

    def test_smoothing_prevents_zeros(self):
        seq = ["bull", "bull", "bull"]
        M = build_transition_matrix(seq, smoothing=1.0)
        assert M.min() > 0

    def test_empty_sequence(self):
        M = build_transition_matrix([])
        # With smoothing, should still be row-stochastic
        for i in range(N_REGIMES):
            assert M[i].sum() == pytest.approx(1.0, abs=1e-6)


# ── multi_step_forecast ──────────────────────────────────────────────────


class TestMultiStepForecast:
    def test_sums_to_one(self):
        seq = ["bull"] * 40 + ["bear"] * 10 + ["bull"] * 20
        M = build_transition_matrix(seq)
        probs = multi_step_forecast(M, "bull", 1)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)

    def test_one_step(self):
        seq = ["bull", "bull", "bull", "bear", "bull"]
        M = build_transition_matrix(seq, smoothing=0.1)
        probs = multi_step_forecast(M, "bull", 1)
        # bull→bull should be highest
        assert probs[REGIME_TO_IDX["bull"]] > probs[REGIME_TO_IDX["bear"]]

    def test_multi_step_still_valid(self):
        seq = ["bull"] * 30 + ["bear"] * 10
        M = build_transition_matrix(seq)
        probs = multi_step_forecast(M, "bull", 21)
        assert probs.sum() == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0 for p in probs)

    def test_converges_to_stationary(self):
        seq = ["bull"] * 30 + ["bear"] * 15 + ["high_vol"] * 5
        M = build_transition_matrix(seq)
        p100 = multi_step_forecast(M, "bull", 100)
        p100_bear = multi_step_forecast(M, "bear", 100)
        # After many steps, should converge regardless of start
        np.testing.assert_allclose(p100, p100_bear, atol=0.1)


# ── compute_macro_adjustments ────────────────────────────────────────────


class TestMacroAdjustments:
    def test_high_vix_boosts_high_vol(self):
        adj = compute_macro_adjustments(vix=40)
        assert adj["high_vol"] > 0
        assert adj["bull"] < 0

    def test_low_vix_boosts_low_vol(self):
        adj = compute_macro_adjustments(vix=12)
        assert adj["low_vol"] > 0

    def test_backwardation_signals_stress(self):
        adj = compute_macro_adjustments(vix=30, vix3m=25)
        assert adj["high_vol"] > 0

    def test_contango_signals_calm(self):
        adj = compute_macro_adjustments(vix=18, vix3m=22)
        assert adj["bull"] > 0

    def test_inverted_yield_curve(self):
        adj = compute_macro_adjustments(yield_spread=-1.0)
        assert adj["bear"] > 0

    def test_no_inputs_zero_adjustments(self):
        adj = compute_macro_adjustments()
        assert all(v == 0.0 for v in adj.values())

    def test_adjustments_bounded(self):
        adj = compute_macro_adjustments(vix=80, vix3m=50, yield_spread=-2, credit_spread=10, momentum_10d=-15)
        for v in adj.values():
            assert -0.15 <= v <= 0.15


# ── apply_adjustments ────────────────────────────────────────────────────


class TestApplyAdjustments:
    def test_preserves_sum_to_one(self):
        base = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
        adj = {"bull": 0.1, "bear": -0.05, "high_vol": 0, "low_vol": 0, "crash": 0}
        result = apply_adjustments(base, adj)
        assert result.sum() == pytest.approx(1.0, abs=1e-6)

    def test_no_negatives(self):
        base = np.array([0.01, 0.01, 0.01, 0.01, 0.96])
        adj = {"bull": -0.15, "bear": -0.15, "high_vol": -0.15, "low_vol": -0.15, "crash": 0.15}
        result = apply_adjustments(base, adj)
        assert all(p > 0 for p in result)


# ── calibrate ────────────────────────────────────────────────────────────


class TestCalibrate:
    def test_perfect_persistence(self):
        """If regime never changes, 1-step forecast should be perfect."""
        seq = ["bull"] * 100
        M = build_transition_matrix(seq, smoothing=0.01)
        cal = calibrate(seq, M, 1)
        assert cal.accuracy > 0.9

    def test_short_sequence(self):
        cal = calibrate(["bull", "bear"], build_transition_matrix(["bull", "bear"]), 1)
        assert cal.n_predictions <= 1

    def test_brier_score_range(self):
        seq = ["bull"] * 30 + ["bear"] * 20 + ["bull"] * 30
        M = build_transition_matrix(seq)
        cal = calibrate(seq, M, 1)
        assert 0 <= cal.brier_score <= 1


# ── RegimeForecaster ─────────────────────────────────────────────────────


class TestRegimeForecaster:
    def _make_sequence(self, n=200):
        import numpy as np
        rng = np.random.RandomState(42)
        seq = []
        current = "bull"
        for _ in range(n):
            seq.append(current)
            r = rng.random()
            if current == "bull":
                current = "bear" if r < 0.05 else "high_vol" if r < 0.08 else "bull"
            elif current == "bear":
                current = "bull" if r < 0.1 else "high_vol" if r < 0.15 else "bear"
            elif current == "high_vol":
                current = "bull" if r < 0.2 else "crash" if r < 0.25 else "high_vol"
            elif current == "crash":
                current = "high_vol" if r < 0.3 else "bear" if r < 0.5 else "crash"
            else:
                current = "bull"
        return seq

    def test_fit(self):
        f = RegimeForecaster()
        seq = self._make_sequence()
        M = f.fit(seq)
        assert f.fitted
        assert M.shape == (N_REGIMES, N_REGIMES)

    def test_predict_1d(self):
        f = RegimeForecaster()
        f.fit(self._make_sequence())
        fc = f.predict("bull", "1d")
        assert isinstance(fc, RegimeForecast)
        assert sum(fc.probabilities.values()) == pytest.approx(1.0, abs=0.01)
        assert fc.predicted_regime in REGIME_NAMES
        assert 0 < fc.confidence <= 1

    def test_predict_all_horizons(self):
        f = RegimeForecaster()
        f.fit(self._make_sequence())
        all_fc = f.forecast_all_horizons("bull")
        assert "1d" in all_fc and "1w" in all_fc and "1m" in all_fc

    def test_predict_with_macro(self):
        f = RegimeForecaster()
        f.fit(self._make_sequence())
        fc_no_macro = f.predict("bull", "1d")
        fc_macro = f.predict("bull", "1d", vix=45)
        # High VIX should shift probabilities
        assert fc_macro.probabilities["high_vol"] > fc_no_macro.probabilities["high_vol"]

    def test_calibration_auto_runs(self):
        f = RegimeForecaster()
        f.fit(self._make_sequence(300))
        assert "1d" in f.calibration_results
        assert f.calibration_results["1d"].n_predictions > 0

    def test_unfitted_uniform(self):
        f = RegimeForecaster()
        fc = f.predict("bull")
        # Unfitted → uniform + macro adjustments
        assert fc.predicted_regime in REGIME_NAMES

    def test_report(self, tmp_path):
        f = RegimeForecaster()
        f.fit(self._make_sequence(200))
        path = f.generate_report("bull", str(tmp_path / "report.html"), vix=20)
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Regime Forecast" in content
        assert "Transition Matrix" in content
        assert "data:image/png;base64," in content

    def test_report_no_external(self, tmp_path):
        f = RegimeForecaster()
        f.fit(self._make_sequence())
        path = f.generate_report("bull", str(tmp_path / "r.html"))
        content = open(path).read()
        assert "http://" not in content
        assert "https://" not in content
