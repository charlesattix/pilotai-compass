"""Tests for compass/vol_surface.py — volatility surface modeler."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.vol_surface import (
    ArbitrageCheck,
    SVIParams,
    SkewKurtosis,
    SurfaceDynamics,
    SurfaceResult,
    TermStructurePoint,
    VolForecast,
    VolSurfaceModeler,
    calibrate_svi,
    check_butterfly,
    check_calendar,
    detect_surface_dynamics,
    extract_skew_kurtosis,
    forecast_iv,
    interpolate_surface,
    svi_implied_vol,
    svi_total_variance,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_chain(
    forward: float = 450.0,
    n_strikes: int = 15,
    expiries: tuple = (30, 60, 90, 180),
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic option chain with vol smile."""
    rng = np.random.RandomState(seed)
    rows = []
    for exp in expiries:
        atm_vol = 0.20 + 0.02 * (exp / 365)
        strikes = np.linspace(forward * 0.85, forward * 1.15, n_strikes)
        for k in strikes:
            log_m = np.log(k / forward)
            skew = -0.15 * log_m
            smile = 0.05 * log_m ** 2
            iv = max(0.05, atm_vol + skew + smile + rng.normal(0, 0.002))
            rows.append({"strike": k, "expiry_days": exp, "iv": iv})
    return pd.DataFrame(rows)


def _make_surface(forward: float = 450.0) -> pd.DataFrame:
    chain = _make_chain(forward)
    return VolSurfaceModeler.build_surface(chain)


@pytest.fixture
def chain():
    return _make_chain()


@pytest.fixture
def surface():
    return _make_surface()


@pytest.fixture
def modeler():
    return VolSurfaceModeler()


# ── SVI model tests ──────────────────────────────────────────────────────


class TestSVI:
    def test_total_variance_positive(self):
        w = svi_total_variance(0.0, 0.04, 0.1, -0.3, 0.0, 0.1)
        assert w > 0

    def test_total_variance_increases_wings(self):
        w_atm = svi_total_variance(0.0, 0.04, 0.1, -0.3, 0.0, 0.1)
        w_otm = svi_total_variance(0.2, 0.04, 0.1, -0.3, 0.0, 0.1)
        assert w_otm > w_atm

    def test_implied_vol_positive(self):
        iv = svi_implied_vol(0.0, 0.25, 0.04, 0.1, -0.3, 0.0, 0.1)
        assert iv > 0

    def test_implied_vol_zero_expiry(self):
        assert svi_implied_vol(0.0, 0.0, 0.04, 0.1, -0.3, 0.0, 0.1) == 0.0

    def test_calibrate_svi_returns_params(self):
        log_k = np.linspace(-0.2, 0.2, 10)
        vols = np.array([0.25 + 0.1 * k ** 2 for k in log_k])
        params = calibrate_svi(log_k, vols, 0.25)
        assert isinstance(params, SVIParams)
        assert params.b > 0

    def test_calibrate_svi_fits_smile(self):
        log_k = np.linspace(-0.15, 0.15, 11)
        true_vols = np.array([0.22 + 0.08 * k ** 2 - 0.1 * k for k in log_k])
        params = calibrate_svi(log_k, true_vols, 0.25)
        fitted = np.array([svi_implied_vol(k, 0.25, params.a, params.b, params.rho, params.m, params.sigma) for k in log_k])
        rmse = np.sqrt(np.mean((fitted - true_vols) ** 2))
        assert rmse < 0.05

    def test_calibrate_svi_short_data(self):
        params = calibrate_svi(np.array([0.0]), np.array([0.2]), 0.25)
        assert isinstance(params, SVIParams)


# ── Surface build tests ──────────────────────────────────────────────────


class TestSurfaceBuild:
    def test_build_surface(self, chain):
        surface = VolSurfaceModeler.build_surface(chain)
        assert not surface.empty
        assert surface.shape[0] > 0
        assert surface.shape[1] == 4

    def test_build_empty_chain(self):
        assert VolSurfaceModeler.build_surface(pd.DataFrame()).empty

    def test_build_missing_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert VolSurfaceModeler.build_surface(df).empty

    def test_strikes_sorted(self, chain):
        surface = VolSurfaceModeler.build_surface(chain)
        strikes = surface.index.values
        assert list(strikes) == sorted(strikes)


# ── Interpolation tests ──────────────────────────────────────────────────


class TestInterpolation:
    def test_on_grid_point(self, surface):
        k = surface.index[5]
        exp = surface.columns[1]
        expected = surface.loc[k, exp]
        result = interpolate_surface(surface, float(k), float(exp))
        assert abs(result - expected) < 0.01

    def test_between_points(self, surface):
        k1, k2 = surface.index[3], surface.index[4]
        mid_k = (k1 + k2) / 2
        exp = surface.columns[0]
        result = interpolate_surface(surface, float(mid_k), float(exp))
        v1 = surface.loc[k1, exp]
        v2 = surface.loc[k2, exp]
        assert min(v1, v2) - 0.01 <= result <= max(v1, v2) + 0.01

    def test_empty_surface(self):
        assert interpolate_surface(pd.DataFrame(), 100.0, 30.0) == 0.0

    def test_clamped_to_boundary(self, surface):
        result = interpolate_surface(surface, 1.0, float(surface.columns[0]))
        assert result > 0


# ── Arbitrage check tests ────────────────────────────────────────────────


class TestArbitrage:
    def test_butterfly_clean_surface(self, surface):
        violations = check_butterfly(surface)
        assert isinstance(violations, list)

    def test_calendar_clean_surface(self, surface):
        violations = check_calendar(surface)
        assert isinstance(violations, list)

    def test_butterfly_violation_detected(self):
        surface = pd.DataFrame(
            {30: [0.20, 0.30, 0.20]},
            index=[90, 100, 110],
        )
        violations = check_butterfly(surface)
        assert len(violations) == 1
        assert violations[0].violation_type == "butterfly"

    def test_calendar_violation_detected(self):
        surface = pd.DataFrame(
            {30: [0.30], 90: [0.10]},
            index=[100],
        )
        violations = check_calendar(surface)
        assert len(violations) >= 1
        assert violations[0].violation_type == "calendar"


# ── Skew/kurtosis tests ─────────────────────────────────────────────────


class TestSkewKurtosis:
    def test_extract(self):
        strikes = np.linspace(400, 500, 11)
        vols = np.array([0.28, 0.26, 0.24, 0.23, 0.22, 0.21, 0.205, 0.21, 0.215, 0.22, 0.23])
        sk = extract_skew_kurtosis(strikes, vols, 450.0, 30)
        assert isinstance(sk, SkewKurtosis)
        assert sk.atm_vol > 0

    def test_positive_skew_for_put_smile(self):
        strikes = np.linspace(400, 500, 11)
        vols = 0.20 + 0.15 * np.exp(-0.01 * (strikes - 400))
        sk = extract_skew_kurtosis(strikes, vols, 450.0, 30)
        assert sk.skew_25d > 0

    def test_butterfly_positive(self):
        strikes = np.linspace(400, 500, 11)
        log_m = np.log(strikes / 450.0)
        vols = 0.20 + 0.3 * log_m ** 2
        sk = extract_skew_kurtosis(strikes, vols, 450.0, 30)
        assert sk.butterfly_25d > 0

    def test_short_data(self):
        sk = extract_skew_kurtosis(np.array([100]), np.array([0.2]), 100.0, 30)
        assert sk.atm_vol == 0.0

    def test_kurtosis_above_three(self):
        strikes = np.linspace(400, 500, 11)
        log_m = np.log(strikes / 450.0)
        vols = 0.20 + 0.3 * log_m ** 2
        sk = extract_skew_kurtosis(strikes, vols, 450.0, 30)
        assert sk.implied_kurtosis > 3.0


# ── Surface dynamics tests ───────────────────────────────────────────────


class TestDynamics:
    def test_detect_regime(self, surface):
        surface2 = surface.copy()
        surface2.index = surface2.index + 5
        dynamics = detect_surface_dynamics(surface, surface2, 450.0, 455.0)
        assert isinstance(dynamics, SurfaceDynamics)
        assert dynamics.regime in ("sticky_strike", "sticky_delta", "mixed")

    def test_no_common_expiries(self):
        s0 = pd.DataFrame({30: [0.2, 0.21]}, index=[100, 110])
        s1 = pd.DataFrame({60: [0.2, 0.21]}, index=[100, 110])
        d = detect_surface_dynamics(s0, s1, 100.0, 100.0)
        assert d.regime == "mixed"

    def test_scores_bounded(self, surface):
        dynamics = detect_surface_dynamics(surface, surface, 450.0, 450.0)
        assert 0 <= dynamics.sticky_strike_score <= 1.0
        assert 0 <= dynamics.sticky_delta_score <= 1.0


# ── Forecast tests ───────────────────────────────────────────────────────


class TestForecast:
    def test_forecast_returns_result(self):
        ts = [TermStructurePoint(30, 0.20, 0.003), TermStructurePoint(90, 0.22, 0.012)]
        sk = [SkewKurtosis(30, 0.20, 0.03, 0.01, -0.5, 4.0, -0.1)]
        fc = forecast_iv(ts, sk)
        assert isinstance(fc, VolForecast)
        assert fc.current_atm_vol == 0.20

    def test_contango_positive_slope(self):
        ts = [TermStructurePoint(30, 0.18, 0.003), TermStructurePoint(90, 0.22, 0.012)]
        fc = forecast_iv(ts, [])
        assert fc.term_structure_slope > 0

    def test_empty_ts(self):
        fc = forecast_iv([], [])
        assert fc.current_atm_vol == 0.0


# ── Full analysis tests ──────────────────────────────────────────────────


class TestFullAnalysis:
    def test_analyze_returns_result(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        assert isinstance(result, SurfaceResult)
        assert result.n_strikes > 0
        assert result.n_expiries > 0

    def test_svi_calibrated(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        assert len(result.svi_params_by_expiry) > 0

    def test_skew_kurtosis_computed(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        assert len(result.skew_kurtosis) > 0

    def test_term_structure(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        assert len(result.term_structure) > 0

    def test_forecast_present(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        assert result.forecast is not None
        assert result.forecast.current_atm_vol > 0

    def test_dynamics_without_t0(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        assert result.dynamics is None

    def test_dynamics_with_t0(self, modeler, chain, surface):
        result = modeler.analyze(chain, forward=450.0, surface_t0=surface, forward_t0=448.0)
        assert result.dynamics is not None

    def test_empty_chain(self, modeler):
        result = modeler.analyze(pd.DataFrame(), forward=450.0)
        assert result.n_strikes == 0


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "vol.html"
            path = VolSurfaceModeler.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Volatility Surface" in content

    def test_contains_svg(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            VolSurfaceModeler.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content

    def test_contains_svi_table(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            VolSurfaceModeler.generate_report(result, out)
            content = out.read_text()
            assert "SVI" in content

    def test_contains_skew(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            VolSurfaceModeler.generate_report(result, out)
            content = out.read_text()
            assert "Skew" in content

    def test_default_path(self, modeler, chain):
        result = modeler.analyze(chain, forward=450.0)
        path = VolSurfaceModeler.generate_report(result)
        assert path.exists()
        assert "vol_surface.html" in str(path)
        path.unlink(missing_ok=True)
