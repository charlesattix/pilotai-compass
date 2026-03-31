"""Tests for compass.vol_forecaster — 30+ tests covering all components."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from compass.vol_forecaster import (
    VolForecaster,
    VolRegime,
    VolForecast,
    GARCHParams,
    ForecastAccuracy,
    IVRVSpread,
    TermStructurePoint,
    TermStructureSnapshot,
    VOL_REGIME_THRESHOLDS,
    TRADING_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 500, mu: float = 0.0, sigma: float = 0.01, seed: int = 42) -> pd.Series:
    """Synthetic daily log-return series."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2024-01-02", periods=n)
    return pd.Series(rng.normal(mu, sigma, n), index=dates, name="returns")


def _make_volatile_returns(n: int = 500, seed: int = 99) -> pd.Series:
    """Returns with a high-vol cluster in the middle."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2024-01-02", periods=n)
    r = rng.normal(0, 0.008, n)
    # inject high-vol cluster
    r[200:260] = rng.normal(0, 0.035, 60)
    return pd.Series(r, index=dates, name="returns")


# ---------------------------------------------------------------------------
# GARCHParams dataclass
# ---------------------------------------------------------------------------

class TestGARCHParams:
    def test_defaults(self):
        p = GARCHParams()
        assert p.omega == 1e-6
        assert p.alpha == 0.10
        assert p.beta == 0.85

    def test_persistence(self):
        p = GARCHParams(alpha=0.08, beta=0.90)
        assert p.persistence == pytest.approx(0.98)

    def test_long_run_var(self):
        p = GARCHParams(omega=1e-6, alpha=0.05, beta=0.90)
        expected = 1e-6 / (1 - 0.95)
        assert p.long_run_var == pytest.approx(expected)

    def test_long_run_var_unit_root(self):
        p = GARCHParams(omega=1e-6, alpha=0.10, beta=0.90)
        assert p.long_run_var == float("inf")


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------

class TestEWMA:
    def test_ewma_vol_shape(self):
        r = _make_returns(100)
        vf = VolForecaster(ewma_span=20)
        s = vf.ewma_vol(r)
        assert len(s) == len(r)

    def test_ewma_vol_positive(self):
        r = _make_returns(100)
        vf = VolForecaster()
        s = vf.ewma_vol(r).dropna()
        assert (s > 0).all()

    def test_ewma_forecast_scalar(self):
        r = _make_returns(100)
        vf = VolForecaster()
        f = vf.ewma_forecast(r)
        assert isinstance(f, float)
        assert f > 0

    def test_ewma_short_series(self):
        r = pd.Series([0.01], index=pd.bdate_range("2026-01-01", periods=1))
        vf = VolForecaster()
        s = vf.ewma_vol(r)
        # Single observation → all NaN (min_periods=2)
        assert s.isna().all()

    def test_ewma_span_effect(self):
        r = _make_volatile_returns(300)
        short = VolForecaster(ewma_span=10).ewma_vol(r)
        long_ = VolForecaster(ewma_span=60).ewma_vol(r)
        # Shorter span reacts more — higher peak vol during cluster
        assert short.max() > long_.max()


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------

class TestGARCH:
    def test_fit_garch_returns_params(self):
        r = _make_returns(300)
        vf = VolForecaster()
        p = vf.fit_garch(r)
        assert isinstance(p, GARCHParams)
        assert 0.01 <= p.alpha <= 0.50
        assert 0.50 <= p.beta <= 0.999
        assert p.persistence < 1.0

    def test_fit_garch_too_few(self):
        r = _make_returns(10)
        vf = VolForecaster()
        p = vf.fit_garch(r)
        # Should return defaults untouched
        assert p.alpha == 0.10
        assert p.beta == 0.85

    def test_garch_variance_series_shape(self):
        r = _make_returns(200)
        vf = VolForecaster()
        v = vf.garch_variance_series(r)
        assert len(v) == len(r.dropna())

    def test_garch_vol_annualised(self):
        r = _make_returns(200)
        vf = VolForecaster()
        s = vf.garch_vol(r).dropna()
        assert (s > 0).all()
        # Should be in a reasonable range for 1% daily vol
        assert s.median() < 1.0  # annualised < 100%

    def test_garch_forecast_scalar(self):
        r = _make_returns(200)
        vf = VolForecaster()
        f = vf.garch_forecast(r)
        assert isinstance(f, float)
        assert f > 0

    def test_garch_multi_step_mean_reversion(self):
        r = _make_returns(300)
        vf = VolForecaster()
        vf.fit_garch(r)
        f1 = vf.garch_forecast(r, horizon=1)
        f60 = vf.garch_forecast(r, horizon=60)
        lr_vol = np.sqrt(vf.garch_params.long_run_var * TRADING_DAYS)
        # Longer horizon should revert toward long-run vol
        assert abs(f60 - lr_vol) <= abs(f1 - lr_vol) + 1e-8


# ---------------------------------------------------------------------------
# Blended forecast
# ---------------------------------------------------------------------------

class TestBlended:
    def test_blended_is_weighted_avg(self):
        r = _make_returns(200)
        vf = VolForecaster(blend_weight=0.6)
        e = vf.ewma_forecast(r)
        g = vf.garch_forecast(r)
        b = vf.blended_forecast(r)
        assert b == pytest.approx(0.6 * e + 0.4 * g)

    def test_blend_weight_one(self):
        r = _make_returns(200)
        vf = VolForecaster(blend_weight=1.0)
        assert vf.blended_forecast(r) == pytest.approx(vf.ewma_forecast(r))

    def test_blend_weight_zero(self):
        r = _make_returns(200)
        vf = VolForecaster(blend_weight=0.0)
        assert vf.blended_forecast(r) == pytest.approx(vf.garch_forecast(r))


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

class TestRegimeClassification:
    def test_low_regime(self):
        vf = VolForecaster()
        assert vf.classify_regime(0.08) == VolRegime.LOW

    def test_normal_regime(self):
        vf = VolForecaster()
        assert vf.classify_regime(0.15) == VolRegime.NORMAL

    def test_high_regime(self):
        vf = VolForecaster()
        assert vf.classify_regime(0.28) == VolRegime.HIGH

    def test_extreme_regime(self):
        vf = VolForecaster()
        assert vf.classify_regime(0.50) == VolRegime.EXTREME

    def test_boundary_low_normal(self):
        vf = VolForecaster()
        assert vf.classify_regime(0.12) == VolRegime.NORMAL

    def test_classify_series(self):
        vf = VolForecaster()
        s = pd.Series([0.08, 0.15, 0.28, 0.50])
        result = vf.classify_series(s)
        assert list(result) == [VolRegime.LOW, VolRegime.NORMAL, VolRegime.HIGH, VolRegime.EXTREME]

    def test_custom_thresholds(self):
        custom = {
            VolRegime.LOW: (0.0, 0.10),
            VolRegime.NORMAL: (0.10, 0.25),
            VolRegime.HIGH: (0.25, 0.40),
            VolRegime.EXTREME: (0.40, float("inf")),
        }
        vf = VolForecaster(regime_thresholds=custom)
        assert vf.classify_regime(0.11) == VolRegime.NORMAL
        assert vf.classify_regime(0.26) == VolRegime.HIGH


# ---------------------------------------------------------------------------
# Realised vol
# ---------------------------------------------------------------------------

class TestRealisedVol:
    def test_rv_shape(self):
        r = _make_returns(100)
        rv = VolForecaster.realised_vol(r, window=21)
        assert len(rv) == len(r)
        assert rv.iloc[:20].isna().all()
        assert rv.iloc[20:].notna().all()

    def test_rv_annualised_magnitude(self):
        r = _make_returns(200, sigma=0.01)
        rv = VolForecaster.realised_vol(r).dropna()
        # 1% daily vol ≈ 15.9% annualised
        assert 0.05 < rv.median() < 0.35


# ---------------------------------------------------------------------------
# IV / RV spread
# ---------------------------------------------------------------------------

class TestIVRVSpread:
    def test_iv_rv_spread_basic(self):
        r = _make_returns(200)
        iv = pd.Series(0.20, index=r.index)
        vf = VolForecaster()
        spreads = vf.iv_rv_spread(iv, r)
        assert len(spreads) > 0
        assert all(isinstance(s, IVRVSpread) for s in spreads)

    def test_iv_rv_spread_percentile_range(self):
        r = _make_returns(300)
        iv = pd.Series(0.18, index=r.index)
        vf = VolForecaster()
        spreads = vf.iv_rv_spread(iv, r)
        for s in spreads:
            assert 0.0 <= s.spread_percentile <= 1.0

    def test_iv_rv_signal_rich(self):
        vf = VolForecaster()
        spreads = [IVRVSpread(date=pd.Timestamp("2026-01-01"), iv=0.25, rv=0.12, spread=0.13, spread_percentile=0.90)]
        assert vf.iv_rv_signal(spreads) == "rich"

    def test_iv_rv_signal_cheap(self):
        vf = VolForecaster()
        spreads = [IVRVSpread(date=pd.Timestamp("2026-01-01"), iv=0.10, rv=0.18, spread=-0.08, spread_percentile=0.10)]
        assert vf.iv_rv_signal(spreads) == "cheap"

    def test_iv_rv_signal_neutral(self):
        vf = VolForecaster()
        spreads = [IVRVSpread(date=pd.Timestamp("2026-01-01"), iv=0.16, rv=0.15, spread=0.01, spread_percentile=0.50)]
        assert vf.iv_rv_signal(spreads) is None

    def test_iv_rv_signal_empty(self):
        vf = VolForecaster()
        assert vf.iv_rv_signal([]) is None


# ---------------------------------------------------------------------------
# Full forecast
# ---------------------------------------------------------------------------

class TestForecast:
    def test_forecast_returns_dataclass(self):
        r = _make_returns(200)
        vf = VolForecaster()
        f = vf.forecast(r)
        assert isinstance(f, VolForecast)
        assert f.ewma_vol > 0
        assert f.garch_vol > 0
        assert isinstance(f.regime, VolRegime)

    def test_forecast_with_iv(self):
        r = _make_returns(200)
        iv = pd.Series(0.22, index=r.index)
        vf = VolForecaster()
        f = vf.forecast(r, iv_series=iv)
        assert f.iv == pytest.approx(0.22)
        assert f.iv_rv_spread is not None

    def test_forecast_with_fit(self):
        r = _make_returns(300)
        vf = VolForecaster()
        f = vf.forecast(r, fit=True)
        assert vf._fitted
        assert isinstance(f, VolForecast)

    def test_forecast_series_length(self):
        r = _make_returns(100)
        vf = VolForecaster()
        fs = vf.forecast_series(r, fit=False)
        # EWMA produces values from min_periods onward
        ewma_len = len(vf.ewma_vol(r))
        assert len(fs) == ewma_len


# ---------------------------------------------------------------------------
# Accuracy tracking
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_log_accuracy(self):
        vf = VolForecaster()
        a = vf.log_accuracy(0.18, 0.15, pd.Timestamp("2026-01-01"))
        assert isinstance(a, ForecastAccuracy)
        assert a.error == pytest.approx(0.03)
        assert a.abs_error == pytest.approx(0.03)
        assert a.squared_error == pytest.approx(0.0009)

    def test_accuracy_stats(self):
        vf = VolForecaster()
        vf.log_accuracy(0.20, 0.15, pd.Timestamp("2026-01-01"))
        vf.log_accuracy(0.10, 0.15, pd.Timestamp("2026-01-02"))
        stats = vf.accuracy_stats()
        assert stats["n"] == 2
        assert stats["mae"] == pytest.approx(0.05)
        assert stats["bias"] == pytest.approx(0.0)
        assert stats["rmse"] == pytest.approx(0.05)

    def test_accuracy_stats_empty(self):
        vf = VolForecaster()
        stats = vf.accuracy_stats()
        assert stats["n"] == 0
        assert stats["mae"] == 0.0

    def test_clear_accuracy(self):
        vf = VolForecaster()
        vf.log_accuracy(0.20, 0.15, pd.Timestamp("2026-01-01"))
        vf.clear_accuracy()
        assert vf.accuracy_stats()["n"] == 0


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

class TestReport:
    def test_generate_report_creates_file(self, tmp_path):
        r = _make_returns(100)
        vf = VolForecaster()
        fs = vf.forecast_series(r, fit=False)
        out = tmp_path / "vol_forecast.html"
        result = vf.generate_report(fs, output_path=str(out))
        assert Path(result).exists()
        content = out.read_text()
        assert "Volatility Forecast Report" in content
        assert "GARCH Params" in content

    def test_report_contains_regime_classes(self, tmp_path):
        vf = VolForecaster()
        forecasts = [
            VolForecast(date=pd.Timestamp("2026-01-01"), ewma_vol=0.08, garch_vol=0.08,
                        blended_vol=0.08, regime=VolRegime.LOW, rv=0.07),
            VolForecast(date=pd.Timestamp("2026-01-02"), ewma_vol=0.40, garch_vol=0.40,
                        blended_vol=0.40, regime=VolRegime.EXTREME, rv=0.38),
        ]
        out = tmp_path / "report.html"
        vf.generate_report(forecasts, output_path=str(out))
        html = out.read_text()
        assert "regime-low" in html
        assert "regime-extreme" in html

    def test_report_with_accuracy(self, tmp_path):
        vf = VolForecaster()
        vf.log_accuracy(0.20, 0.15, pd.Timestamp("2026-01-01"))
        forecasts = [
            VolForecast(date=pd.Timestamp("2026-01-01"), ewma_vol=0.20, garch_vol=0.20,
                        blended_vol=0.20, regime=VolRegime.NORMAL, rv=0.15),
        ]
        out = tmp_path / "report.html"
        vf.generate_report(forecasts, output_path=str(out))
        html = out.read_text()
        assert "Forecast Accuracy" in html
        assert "MAE" in html

    def test_report_contains_iv_rv_chart_with_iv(self, tmp_path):
        vf = VolForecaster()
        forecasts = [
            VolForecast(date=pd.Timestamp(f"2026-01-{d:02d}"), ewma_vol=0.15,
                        garch_vol=0.16, blended_vol=0.155, regime=VolRegime.NORMAL,
                        iv=0.20, rv=0.14, iv_rv_spread=0.06)
            for d in range(1, 11)
        ]
        out = tmp_path / "report.html"
        vf.generate_report(forecasts, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "IV" in html and "RV" in html

    def test_report_contains_vol_chart_no_iv(self, tmp_path):
        vf = VolForecaster()
        forecasts = [
            VolForecast(date=pd.Timestamp(f"2026-01-{d:02d}"), ewma_vol=0.15,
                        garch_vol=0.16, blended_vol=0.155, regime=VolRegime.NORMAL, rv=0.14)
            for d in range(1, 11)
        ]
        out = tmp_path / "report.html"
        vf.generate_report(forecasts, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "EWMA" in html

    def test_report_regime_timeline_svg(self, tmp_path):
        vf = VolForecaster()
        forecasts = [
            VolForecast(date=pd.Timestamp("2026-01-01"), ewma_vol=0.08,
                        garch_vol=0.08, blended_vol=0.08, regime=VolRegime.LOW, rv=0.07),
            VolForecast(date=pd.Timestamp("2026-01-02"), ewma_vol=0.40,
                        garch_vol=0.40, blended_vol=0.40, regime=VolRegime.EXTREME, rv=0.38),
        ]
        out = tmp_path / "report.html"
        vf.generate_report(forecasts, output_path=str(out))
        html = out.read_text()
        assert "Vol Regime Timeline" in html
        assert "#27ae60" in html  # LOW colour
        assert "#e74c3c" in html  # EXTREME colour

    def test_report_with_term_structure(self, tmp_path):
        vf = VolForecaster()
        ts = VolForecaster.build_term_structure({30: 0.18, 60: 0.20, 90: 0.22})
        forecasts = [
            VolForecast(date=pd.Timestamp("2026-01-01"), ewma_vol=0.18,
                        garch_vol=0.18, blended_vol=0.18, regime=VolRegime.NORMAL, rv=0.15),
        ]
        out = tmp_path / "report.html"
        vf.generate_report(forecasts, output_path=str(out), term_structure=ts)
        html = out.read_text()
        assert "Term Structure" in html
        assert "1M" in html


# ---------------------------------------------------------------------------
# Term structure analysis
# ---------------------------------------------------------------------------

class TestTermStructure:
    def test_build_basic(self):
        ts = VolForecaster.build_term_structure({30: 0.18, 60: 0.20, 90: 0.22})
        assert isinstance(ts, TermStructureSnapshot)
        assert len(ts.points) == 3
        assert ts.points[0].tenor_days == 30

    def test_tenor_labels(self):
        ts = VolForecaster.build_term_structure({7: 0.20, 30: 0.18, 90: 0.16})
        labels = [p.tenor_label for p in ts.points]
        assert labels == ["1W", "1M", "3M"]

    def test_custom_tenor_label(self):
        ts = VolForecaster.build_term_structure({42: 0.19})
        assert ts.points[0].tenor_label == "42D"

    def test_slope_contango(self):
        # Normal term structure: short < long → negative slope
        ts = VolForecaster.build_term_structure({30: 0.15, 60: 0.18, 90: 0.22})
        assert ts.slope < 0
        assert not ts.is_inverted

    def test_slope_backwardation(self):
        # Inverted: short >> long
        ts = VolForecaster.build_term_structure({30: 0.30, 60: 0.25, 90: 0.20})
        assert ts.slope > 0
        assert ts.is_inverted

    def test_curvature_humped(self):
        # Belly higher than average of ends → positive curvature
        ts = VolForecaster.build_term_structure({30: 0.15, 60: 0.25, 90: 0.15})
        assert ts.curvature > 0

    def test_curvature_flat(self):
        ts = VolForecaster.build_term_structure({30: 0.20, 60: 0.20, 90: 0.20})
        assert ts.curvature == pytest.approx(0.0)

    def test_empty_input(self):
        ts = VolForecaster.build_term_structure({})
        assert len(ts.points) == 0
        assert ts.slope == 0.0
        assert ts.curvature == 0.0
        assert not ts.is_inverted

    def test_single_tenor(self):
        ts = VolForecaster.build_term_structure({30: 0.20})
        assert len(ts.points) == 1
        assert ts.slope == 0.0  # short == long
        assert ts.curvature == 0.0

    def test_term_structure_series(self):
        data = {
            pd.Timestamp("2026-01-01"): {30: 0.18, 60: 0.20, 90: 0.22},
            pd.Timestamp("2026-01-02"): {30: 0.25, 60: 0.22, 90: 0.19},
        }
        results = VolForecaster.term_structure_series(data)
        assert len(results) == 2
        assert not results[0].is_inverted
        assert results[1].is_inverted

    def test_date_passthrough(self):
        dt = pd.Timestamp("2026-06-15")
        ts = VolForecaster.build_term_structure({30: 0.20}, date=dt)
        assert ts.date == dt


# ---------------------------------------------------------------------------
# SVG chart internals
# ---------------------------------------------------------------------------

class TestSVGCharts:
    def test_line_chart_basic(self):
        svg = VolForecaster._svg_line_chart(
            {"A": [(0, 0.10), (1, 0.15), (2, 0.12)]},
            title="Test Chart",
        )
        assert "<svg" in svg
        assert "Test Chart" in svg
        assert "<path" in svg

    def test_line_chart_empty(self):
        svg = VolForecaster._svg_line_chart({})
        assert svg == ""

    def test_line_chart_multiple_series(self):
        svg = VolForecaster._svg_line_chart({
            "X": [(0, 0.1), (1, 0.2)],
            "Y": [(0, 0.15), (1, 0.25)],
        })
        assert svg.count("<path") == 2

    def test_regime_timeline_colors(self):
        forecasts = [
            VolForecast(date=pd.Timestamp("2026-01-01"), ewma_vol=0.08,
                        garch_vol=0.08, blended_vol=0.08, regime=VolRegime.LOW, rv=0.07),
            VolForecast(date=pd.Timestamp("2026-01-02"), ewma_vol=0.25,
                        garch_vol=0.25, blended_vol=0.25, regime=VolRegime.HIGH, rv=0.22),
        ]
        svg = VolForecaster._svg_regime_timeline(forecasts)
        assert "#27ae60" in svg  # LOW
        assert "#e67e22" in svg  # HIGH

    def test_regime_timeline_empty(self):
        assert VolForecaster._svg_regime_timeline([]) == ""

    def test_term_structure_svg(self):
        ts = VolForecaster.build_term_structure({30: 0.18, 60: 0.20, 90: 0.22})
        svg = VolForecaster._svg_term_structure(ts)
        assert "<svg" in svg
        assert "1M" in svg
        assert "3M" in svg

    def test_term_structure_svg_empty(self):
        ts = VolForecaster.build_term_structure({})
        assert VolForecaster._svg_term_structure(ts) == ""

    def test_term_structure_svg_inverted_label(self):
        ts = VolForecaster.build_term_structure({30: 0.30, 90: 0.18})
        svg = VolForecaster._svg_term_structure(ts)
        assert "INVERTED" in svg
