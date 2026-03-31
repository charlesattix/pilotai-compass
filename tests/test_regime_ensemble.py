"""Tests for compass.regime_ensemble – ensemble regime detection."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.regime_ensemble import (
    REGIMES,
    EnsembleResult,
    GaussianHMM,
    MethodResult,
    RegimeConsensus,
    RegimeEnsemble,
    TransitionMatrix,
    detect_changepoint,
    detect_hmm,
    detect_macro,
    detect_trend,
    detect_vol_clustering,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    rets = rng.randn(n) * 0.01 + 0.0003
    # Inject regime-like structure
    rets[100:150] = rng.randn(50) * 0.03 - 0.01   # bear/high_vol
    rets[300:320] = rng.randn(20) * 0.04 - 0.02   # crash-like
    return pd.Series(rets, index=idx)


def _make_vix(n: int = 500, seed: int = 77) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    vix = 18 + rng.randn(n) * 3
    vix[100:150] = 32 + rng.randn(50) * 2
    vix[300:320] = 45 + rng.randn(20) * 3
    return pd.Series(np.abs(vix), index=idx)


def _make_yield_curve(n: int = 500, seed: int = 33) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.Series(0.5 + rng.randn(n) * 0.3, index=idx)


# ── HMM ─────────────────────────────────────────────────────────────────────
class TestGaussianHMM:
    def test_returns_states_and_probs(self):
        hmm = GaussianHMM(n_states=3)
        r = _make_returns().values
        states, probs = hmm.fit_predict(r)
        assert len(states) == len(r)
        assert len(probs) == len(r)

    def test_states_in_range(self):
        states, _ = GaussianHMM(n_states=3).fit_predict(_make_returns().values)
        assert set(states).issubset({0, 1, 2})

    def test_probs_bounded(self):
        _, probs = GaussianHMM(n_states=2).fit_predict(_make_returns().values)
        assert np.all(probs >= 0) and np.all(probs <= 1.01)


# ── Individual detectors ───────────────────────────────────────────────────
class TestDetectHMM:
    def test_returns_method_result(self):
        r = detect_hmm(_make_returns())
        assert isinstance(r, MethodResult)
        assert r.method == "hmm"

    def test_regimes_valid(self):
        r = detect_hmm(_make_returns())
        assert all(reg in REGIMES for reg in r.regimes)

    def test_short_series(self):
        r = detect_hmm(pd.Series([0.01] * 5))
        assert r.regimes == []


class TestDetectChangepoint:
    def test_returns_result(self):
        r = detect_changepoint(_make_returns())
        assert r.method == "changepoint"
        assert len(r.regimes) > 0

    def test_regimes_valid(self):
        r = detect_changepoint(_make_returns())
        assert all(reg in REGIMES for reg in r.regimes)


class TestDetectVolClustering:
    def test_returns_result(self):
        r = detect_vol_clustering(_make_returns())
        assert r.method == "vol_cluster"
        assert len(r.regimes) > 0

    def test_crash_detected(self):
        r = detect_vol_clustering(_make_returns())
        assert "crash" in r.regimes or "high_vol" in r.regimes


class TestDetectTrend:
    def test_returns_result(self):
        r = detect_trend(_make_returns())
        assert r.method == "trend"
        assert len(r.regimes) > 0

    def test_has_bull_or_bear(self):
        r = detect_trend(_make_returns())
        assert "bull" in r.regimes or "bear" in r.regimes


class TestDetectMacro:
    def test_returns_result(self):
        r = detect_macro(_make_returns(), _make_vix(), _make_yield_curve())
        assert r.method == "macro"
        assert len(r.regimes) > 0

    def test_high_vix_crash(self):
        r = detect_macro(_make_returns(), _make_vix())
        assert "crash" in r.regimes or "high_vol" in r.regimes

    def test_no_vix_still_works(self):
        r = detect_macro(_make_returns())
        assert len(r.regimes) > 0


# ── Ensemble ────────────────────────────────────────────────────────────────
class TestEnsemble:
    def test_returns_result(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert isinstance(r, EnsembleResult)

    def test_consensus_populated(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert len(r.consensus) > 0

    def test_method_results_populated(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert len(r.method_results) == 5

    def test_current_regime_set(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert r.current_regime in REGIMES

    def test_confidence_bounded(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert 0.0 <= r.current_confidence <= 1.0
        for c in r.consensus:
            assert 0.0 <= c.confidence <= 1.0

    def test_agreement_bounded(self):
        r = RegimeEnsemble().detect(_make_returns())
        for c in r.consensus:
            assert 0.0 <= c.method_agreement <= 1.0

    def test_votes_present(self):
        r = RegimeEnsemble().detect(_make_returns())
        for c in r.consensus:
            assert len(c.votes) > 0

    def test_n_bars(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert r.n_bars > 0

    def test_generated_at(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert len(r.generated_at) > 0

    def test_with_macro_data(self):
        r = RegimeEnsemble().detect(_make_returns(), vix=_make_vix(), yield_curve=_make_yield_curve())
        assert len(r.consensus) > 0

    def test_subset_methods(self):
        r = RegimeEnsemble(methods=["hmm", "vol_cluster"]).detect(_make_returns())
        assert len(r.method_results) == 2

    def test_short_series(self):
        r = RegimeEnsemble().detect(pd.Series([0.01] * 10))
        assert r.consensus == []


# ── Transition matrix ──────────────────────────────────────────────────────
class TestTransitionMatrix:
    def test_present(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert r.transition_matrix is not None

    def test_rows_sum_to_one(self):
        r = RegimeEnsemble().detect(_make_returns())
        tm = r.transition_matrix
        for fr in tm.regimes:
            row_sum = sum(tm.matrix[fr].values())
            assert row_sum == pytest.approx(1.0, abs=0.01)

    def test_stationary_sums_to_one(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert sum(r.transition_matrix.stationary.values()) == pytest.approx(1.0, abs=0.01)

    def test_n_transitions(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert r.transition_matrix.n_transitions > 0


# ── Forecast ────────────────────────────────────────────────────────────────
class TestForecast:
    def test_forecast_populated(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert len(r.forecast) > 0

    def test_forecast_sums_to_one(self):
        r = RegimeEnsemble().detect(_make_returns())
        assert sum(r.forecast.values()) == pytest.approx(1.0, abs=0.01)

    def test_forecast_nonnegative(self):
        r = RegimeEnsemble().detect(_make_returns())
        for p in r.forecast.values():
            assert p >= 0


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = RegimeEnsemble()
            r = e.detect(_make_returns(), vix=_make_vix())
            path = e.generate_report(r, output_path=Path(tmp) / "re.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = RegimeEnsemble()
            r = e.detect(_make_returns(), vix=_make_vix())
            path = e.generate_report(r, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Regime Ensemble" in html
            assert "Timeline" in html
            assert "Agreement" in html
            assert "Transition" in html
            assert "Forecast" in html

    def test_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = RegimeEnsemble()
            r = e.detect(_make_returns())
            path = e.generate_report(r, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_method_result(self):
        m = MethodResult("test", ["bull", "bear"], [0.8, 0.6])
        assert m.method == "test"

    def test_consensus(self):
        c = RegimeConsensus("bull", 0.8, 0.6, {"hmm": "bull"})
        assert c.regime == "bull"

    def test_transition_matrix(self):
        tm = TransitionMatrix({"bull": {"bull": 0.8, "bear": 0.2}}, ["bull", "bear"], 100, {"bull": 0.7, "bear": 0.3})
        assert tm.n_transitions == 100

    def test_result_defaults(self):
        r = EnsembleResult()
        assert r.consensus == []
        assert r.current_regime == ""
