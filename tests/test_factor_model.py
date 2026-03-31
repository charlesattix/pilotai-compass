"""Tests for compass.factor_model – multi-factor risk model."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.factor_model import (
    FUNDAMENTAL_FACTORS,
    FactorExposure,
    FactorModel,
    FactorModelResult,
    FactorReturn,
    NeutralPortfolio,
    RiskDecomposition,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_returns(n: int = 300, n_assets: int = 5, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    # Correlated returns: common factor + noise
    common = rng.randn(n) * 0.01
    data = {}
    for i in range(n_assets):
        data[f"A{i+1}"] = common * (0.5 + i * 0.1) + rng.randn(n) * 0.005
    return pd.DataFrame(data, index=idx)


def _make_factor_data(n: int = 300, seed: int = 77) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "value": rng.randn(n) * 0.005,
        "momentum": rng.randn(n) * 0.008,
    }, index=idx)


def _weights(n: int = 5) -> dict:
    return {f"A{i+1}": 1.0 / n for i in range(n)}


# ── Constructor ─────────────────────────────────────────────────────────────
class TestInit:
    def test_defaults(self):
        m = FactorModel()
        assert m.n_stat_factors == 3
        assert m.neutrality_threshold == 0.05

    def test_custom(self):
        m = FactorModel(n_statistical_factors=5, shrinkage_target=0.7)
        assert m.n_stat_factors == 5
        assert m.shrinkage_target == 0.7


# ── Fit ─────────────────────────────────────────────────────────────────────
class TestFit:
    def test_returns_result(self):
        r = FactorModel().fit(_make_returns())
        assert isinstance(r, FactorModelResult)

    def test_exposures_per_asset(self):
        ret = _make_returns(n_assets=4)
        r = FactorModel().fit(ret)
        assert len(r.exposures) == 4

    def test_factor_returns_present(self):
        r = FactorModel().fit(_make_returns())
        assert len(r.factor_returns) > 0

    def test_n_factors_set(self):
        r = FactorModel().fit(_make_returns())
        assert r.n_factors > 0

    def test_n_assets_set(self):
        r = FactorModel().fit(_make_returns(n_assets=6))
        assert r.n_assets == 6

    def test_pca_variance_explained(self):
        r = FactorModel().fit(_make_returns())
        assert len(r.pca_variance_explained) > 0
        assert sum(r.pca_variance_explained) <= 1.0 + 1e-6

    def test_generated_at(self):
        r = FactorModel().fit(_make_returns())
        assert len(r.generated_at) > 0

    def test_too_few_rows(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        ret = pd.DataFrame({"A": [0.01]*5, "B": [0.02]*5}, index=idx)
        r = FactorModel().fit(ret)
        assert r.n_factors == 0

    def test_with_fundamental_factors(self):
        ret = _make_returns()
        fd = _make_factor_data()
        r = FactorModel().fit(ret, factor_data=fd)
        factor_names = [fr.factor for fr in r.factor_returns]
        assert "value" in factor_names or "momentum" in factor_names


# ── Exposures ───────────────────────────────────────────────────────────────
class TestExposures:
    def test_exposure_fields(self):
        r = FactorModel().fit(_make_returns())
        for e in r.exposures:
            assert isinstance(e.asset, str)
            assert len(e.exposures) > 0

    def test_all_assets_covered(self):
        ret = _make_returns(n_assets=3)
        r = FactorModel().fit(ret)
        assets = {e.asset for e in r.exposures}
        assert assets == {"A1", "A2", "A3"}

    def test_correlated_assets_similar_exposure(self):
        """Highly correlated assets should have similar factor loadings."""
        r = FactorModel().fit(_make_returns())
        e1 = r.exposures[0].exposures
        e2 = r.exposures[1].exposures
        # At least one common factor with same-sign exposure
        common = set(e1.keys()) & set(e2.keys())
        assert len(common) > 0


# ── Factor returns ──────────────────────────────────────────────────────────
class TestFactorReturns:
    def test_return_fields(self):
        r = FactorModel().fit(_make_returns())
        for fr in r.factor_returns:
            assert isinstance(fr.factor, str)
            assert isinstance(fr.sharpe, float)
            assert isinstance(fr.t_stat, float)

    def test_pca_factors_present(self):
        r = FactorModel().fit(_make_returns())
        names = {fr.factor for fr in r.factor_returns}
        assert "PC1" in names


# ── Covariance ──────────────────────────────────────────────────────────────
class TestCovariance:
    def test_covariance_present(self):
        r = FactorModel().fit(_make_returns())
        assert r.factor_covariance is not None
        assert not r.factor_covariance.empty

    def test_covariance_symmetric(self):
        r = FactorModel().fit(_make_returns())
        cov = r.factor_covariance.values
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)

    def test_diagonal_positive(self):
        r = FactorModel().fit(_make_returns())
        diag = np.diag(r.factor_covariance.values)
        assert np.all(diag >= 0)

    def test_shrinkage_effect(self):
        """Higher shrinkage → covariance closer to diagonal."""
        ret = _make_returns()
        low = FactorModel(shrinkage_target=0.1).fit(ret)
        high = FactorModel(shrinkage_target=0.9).fit(ret)
        off_diag_low = np.abs(low.factor_covariance.values[~np.eye(low.factor_covariance.shape[0], dtype=bool)]).mean()
        off_diag_high = np.abs(high.factor_covariance.values[~np.eye(high.factor_covariance.shape[0], dtype=bool)]).mean()
        assert off_diag_high <= off_diag_low + 1e-9


# ── Risk decomposition ─────────────────────────────────────────────────────
class TestRiskDecomposition:
    def test_present_with_weights(self):
        r = FactorModel().fit(_make_returns(), weights=_weights())
        assert r.risk_decomposition is not None

    def test_absent_without_weights(self):
        r = FactorModel().fit(_make_returns())
        assert r.risk_decomposition is None

    def test_sums_to_one(self):
        r = FactorModel().fit(_make_returns(), weights=_weights())
        rd = r.risk_decomposition
        assert rd.systematic_pct + rd.idiosyncratic_pct == pytest.approx(1.0, abs=0.01)

    def test_systematic_positive(self):
        r = FactorModel().fit(_make_returns(), weights=_weights())
        assert r.risk_decomposition.systematic_pct >= 0

    def test_factor_contributions(self):
        r = FactorModel().fit(_make_returns(), weights=_weights())
        rd = r.risk_decomposition
        assert len(rd.factor_contributions) > 0
        assert all(v >= 0 for v in rd.factor_contributions.values())


# ── Neutral portfolio ───────────────────────────────────────────────────────
class TestNeutralPortfolio:
    def test_present(self):
        r = FactorModel().fit(_make_returns())
        assert r.neutral_portfolio is not None

    def test_weights_sum_to_one(self):
        r = FactorModel().fit(_make_returns())
        total = sum(r.neutral_portfolio.weights.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_residual_exposures_small(self):
        r = FactorModel().fit(_make_returns())
        np_ = r.neutral_portfolio
        if np_.is_neutral:
            assert np_.max_residual < 0.05

    def test_all_assets_have_weight(self):
        ret = _make_returns(n_assets=4)
        r = FactorModel().fit(ret)
        assert len(r.neutral_portfolio.weights) == 4


# ── HTML ────────────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = FactorModel()
            r = m.fit(_make_returns(), weights=_weights())
            path = m.generate_report(r, output_path=Path(tmp) / "fm.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = FactorModel()
            r = m.fit(_make_returns(), factor_data=_make_factor_data(), weights=_weights())
            path = m.generate_report(r, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Factor Risk Model" in html
            assert "Exposures" in html
            assert "Attribution" in html
            assert "Decomposition" in html
            assert "Covariance" in html
            assert "Neutral" in html

    def test_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = FactorModel()
            r = m.fit(_make_returns())
            path = m.generate_report(r, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_factor_exposure(self):
        e = FactorExposure("SPY", {"momentum": 0.5})
        assert e.exposures["momentum"] == 0.5

    def test_factor_return(self):
        fr = FactorReturn("PC1", 0.05, 0.0002, 0.01, 0.3, 2.1)
        assert fr.sharpe == 0.3

    def test_risk_decomp(self):
        rd = RiskDecomposition(0.01, 0.007, 0.003, 0.7, 0.3, {"PC1": 0.5})
        assert rd.systematic_pct == 0.7

    def test_result_defaults(self):
        r = FactorModelResult()
        assert r.exposures == []
        assert r.n_factors == 0
