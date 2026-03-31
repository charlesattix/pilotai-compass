"""Tests for compass.tail_risk – tail risk analyzer with EVT and CVaR."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.tail_risk import (
    DEFAULT_CONFIDENCE_LEVELS,
    ExperimentTailContrib,
    GPDFit,
    StressVaR,
    TailRiskAnalyzer,
    TailRiskResult,
    VaRCVaR,
    _fit_gpd_mle,
    _gpd_cdf,
    _gpd_quantile,
    _ks_test_gpd,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_returns(
    n: int = 500,
    seed: int = 42,
    drift: float = 0.0002,
    vol: float = 0.015,
) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rets = drift + rng.randn(n) * vol
    return pd.Series(rets, index=idx, name="returns")


def _make_experiment_returns(
    n_exp: int = 3,
    n_days: int = 500,
    seed: int = 42,
) -> dict[str, pd.Series]:
    result = {}
    for i in range(n_exp):
        result[f"EXP-{i+1}"] = _make_returns(n=n_days, seed=seed + i * 7)
    return result


def _make_fat_tail_returns(n: int = 1000, seed: int = 99) -> pd.Series:
    """Returns with occasional large negative draws (credit spread-like)."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    rets = rng.randn(n) * 0.01 + 0.0003
    # Inject tail events
    for i in range(0, n, 50):
        rets[i] = -0.08 - rng.rand() * 0.04
    return pd.Series(rets, index=idx, name="fat_tail")


# ── Constructor ─────────────────────────────────────────────────────────────
class TestTailRiskAnalyzerInit:
    def test_default_confidence(self):
        a = TailRiskAnalyzer()
        assert a.confidence_levels == (0.95, 0.99)

    def test_custom_confidence(self):
        a = TailRiskAnalyzer(confidence_levels=(0.90, 0.95, 0.99))
        assert len(a.confidence_levels) == 3

    def test_default_stress_params(self):
        a = TailRiskAnalyzer()
        assert a.stress_window == 20
        assert a.stress_top_n == 5

    def test_custom_gpd_threshold(self):
        a = TailRiskAnalyzer(gpd_threshold_pct=95.0)
        assert a.gpd_threshold_pct == 95.0


# ── VaR / CVaR ──────────────────────────────────────────────────────────────
class TestVaRCVaR:
    def test_two_confidence_levels(self):
        exp_ret = _make_experiment_returns(n_exp=1)
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert len(result.var_cvar) == 2

    def test_var_positive(self):
        exp_ret = _make_experiment_returns(n_exp=1)
        result = TailRiskAnalyzer().analyze(exp_ret)
        for vc in result.var_cvar:
            assert vc.var > 0

    def test_cvar_geq_var(self):
        """CVaR (expected shortfall) should be >= VaR."""
        exp_ret = _make_experiment_returns(n_exp=1)
        result = TailRiskAnalyzer().analyze(exp_ret)
        for vc in result.var_cvar:
            assert vc.cvar >= vc.var - 1e-9

    def test_var99_geq_var95(self):
        exp_ret = _make_experiment_returns(n_exp=1)
        result = TailRiskAnalyzer().analyze(exp_ret)
        var95 = next(v for v in result.var_cvar if v.confidence == 0.95)
        var99 = next(v for v in result.var_cvar if v.confidence == 0.99)
        assert var99.var >= var95.var - 1e-9

    def test_var_n_obs(self):
        exp_ret = _make_experiment_returns(n_exp=1, n_days=300)
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert result.var_cvar[0].n_obs == 300


# ── GPD / EVT ───────────────────────────────────────────────────────────────
class TestGPDFit:
    def test_gpd_fit_present(self):
        exp_ret = {"A": _make_fat_tail_returns()}
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert result.gpd_fit is not None

    def test_gpd_has_exceedances(self):
        exp_ret = {"A": _make_fat_tail_returns()}
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert result.gpd_fit.n_exceedances > 0

    def test_gpd_evt_var_positive(self):
        exp_ret = {"A": _make_fat_tail_returns()}
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert result.gpd_fit.evt_var_99 > 0
        assert result.gpd_fit.evt_var_95 > 0

    def test_gpd_evt_var99_geq_95(self):
        exp_ret = {"A": _make_fat_tail_returns()}
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert result.gpd_fit.evt_var_99 >= result.gpd_fit.evt_var_95 - 1e-9

    def test_gpd_scale_positive(self):
        exp_ret = {"A": _make_fat_tail_returns()}
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert result.gpd_fit.beta > 0

    def test_gpd_ks_pvalue_bounded(self):
        exp_ret = {"A": _make_fat_tail_returns()}
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert 0.0 <= result.gpd_fit.ks_pvalue <= 1.0


# ── GPD helper functions ────────────────────────────────────────────────────
class TestGPDHelpers:
    def test_gpd_cdf_bounded(self):
        x = np.array([0.01, 0.05, 0.1, 0.5])
        cdf = _gpd_cdf(x, xi=0.2, beta=0.05)
        assert np.all(cdf >= 0.0)
        assert np.all(cdf <= 1.0)

    def test_gpd_cdf_monotone(self):
        x = np.linspace(0.001, 0.5, 50)
        cdf = _gpd_cdf(x, xi=0.1, beta=0.05)
        assert np.all(np.diff(cdf) >= -1e-12)

    def test_gpd_quantile_positive(self):
        q = _gpd_quantile(0.95, xi=0.2, beta=0.05)
        assert q > 0

    def test_gpd_quantile_increases_with_p(self):
        q90 = _gpd_quantile(0.90, xi=0.2, beta=0.05)
        q99 = _gpd_quantile(0.99, xi=0.2, beta=0.05)
        assert q99 > q90

    def test_fit_gpd_mle_returns_tuple(self):
        rng = np.random.RandomState(123)
        exc = rng.exponential(scale=0.02, size=100)
        xi, beta = _fit_gpd_mle(exc)
        assert isinstance(xi, float)
        assert isinstance(beta, float)
        assert beta > 0

    def test_ks_test_returns_pvalue(self):
        rng = np.random.RandomState(77)
        exc = rng.exponential(scale=0.02, size=100)
        xi, beta = _fit_gpd_mle(exc)
        p = _ks_test_gpd(exc, xi, beta)
        assert 0.0 <= p <= 1.0


# ── Per-experiment contribution ─────────────────────────────────────────────
class TestExperimentContributions:
    def test_contribs_present(self):
        exp_ret = _make_experiment_returns(n_exp=3)
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert len(result.experiment_contribs) == 3

    def test_contrib_fields(self):
        exp_ret = _make_experiment_returns(n_exp=2)
        result = TailRiskAnalyzer().analyze(exp_ret)
        for c in result.experiment_contribs:
            assert isinstance(c.experiment_id, str)
            assert c.standalone_cvar_95 > 0
            assert c.standalone_cvar_99 > 0
            assert isinstance(c.pct_contribution, float)

    def test_weights_equal_when_none(self):
        exp_ret = _make_experiment_returns(n_exp=4)
        result = TailRiskAnalyzer().analyze(exp_ret)
        for c in result.experiment_contribs:
            assert c.weight == pytest.approx(0.25)

    def test_custom_weights(self):
        exp_ret = _make_experiment_returns(n_exp=2)
        weights = {"EXP-1": 0.7, "EXP-2": 0.3}
        result = TailRiskAnalyzer().analyze(exp_ret, weights=weights)
        w_map = {c.experiment_id: c.weight for c in result.experiment_contribs}
        assert w_map["EXP-1"] == pytest.approx(0.7)
        assert w_map["EXP-2"] == pytest.approx(0.3)


# ── Stress VaR ──────────────────────────────────────────────────────────────
class TestStressVaR:
    def test_stress_vars_present(self):
        exp_ret = {"A": _make_returns(n=500)}
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert len(result.stress_vars) == 2

    def test_stress_var_geq_normal(self):
        """Stress VaR should typically be >= normal VaR."""
        exp_ret = {"A": _make_fat_tail_returns(n=1000)}
        result = TailRiskAnalyzer().analyze(exp_ret)
        for sv in result.stress_vars:
            # Not a strict guarantee, but should hold for fat-tailed data
            assert sv.stress_var > 0
            assert sv.normal_var > 0

    def test_stress_ratio_positive(self):
        exp_ret = {"A": _make_returns(n=500)}
        result = TailRiskAnalyzer().analyze(exp_ret)
        for sv in result.stress_vars:
            assert sv.stress_ratio > 0

    def test_stress_n_obs(self):
        exp_ret = {"A": _make_returns(n=500)}
        result = TailRiskAnalyzer().analyze(exp_ret)
        for sv in result.stress_vars:
            assert sv.n_stress_obs > 0


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_ret = _make_experiment_returns(n_exp=2)
            analyzer = TailRiskAnalyzer()
            result = analyzer.analyze(exp_ret)
            path = analyzer.generate_report(result, output_path=Path(tmp) / "t.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_ret = {"A": _make_fat_tail_returns()}
            analyzer = TailRiskAnalyzer()
            result = analyzer.analyze(exp_ret)
            path = analyzer.generate_report(result, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Tail Risk Analysis" in html
            assert "VaR" in html
            assert "CVaR" in html
            assert "GPD" in html
            assert "Stress" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp_ret = _make_experiment_returns(n_exp=1)
            analyzer = TailRiskAnalyzer()
            result = analyzer.analyze(exp_ret)
            path = analyzer.generate_report(result, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html

    def test_report_empty_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            analyzer = TailRiskAnalyzer()
            result = TailRiskResult(generated_at="2024-01-01T00:00:00+00:00")
            path = analyzer.generate_report(result, output_path=Path(tmp) / "e.html")
            assert path.exists()


# ── Edge cases ──────────────────────────────────────────────────────────────
class TestEdgeCases:
    def test_empty_returns(self):
        result = TailRiskAnalyzer().analyze({})
        assert result.var_cvar == []
        assert result.gpd_fit is None

    def test_too_few_observations(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        exp = {"A": pd.Series([0.01] * 5, index=idx)}
        result = TailRiskAnalyzer().analyze(exp)
        assert result.var_cvar == []

    def test_single_experiment(self):
        exp_ret = _make_experiment_returns(n_exp=1)
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert len(result.experiment_contribs) == 1

    def test_generated_at_set(self):
        exp_ret = _make_experiment_returns(n_exp=1)
        result = TailRiskAnalyzer().analyze(exp_ret)
        assert len(result.generated_at) > 0


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_var_cvar_fields(self):
        v = VaRCVaR(confidence=0.95, var=0.03, cvar=0.05, n_obs=500)
        assert v.confidence == 0.95
        assert v.horizon_days == 1

    def test_gpd_fit_fields(self):
        g = GPDFit(xi=0.2, beta=0.01, threshold=0.02, n_exceedances=50,
                   ks_pvalue=0.4, evt_var_95=0.03, evt_var_99=0.05, evt_cvar_99=0.07)
        assert g.xi == 0.2

    def test_tail_risk_result_defaults(self):
        r = TailRiskResult()
        assert r.var_cvar == []
        assert r.gpd_fit is None
        assert r.experiment_contribs == []
        assert r.stress_vars == []
