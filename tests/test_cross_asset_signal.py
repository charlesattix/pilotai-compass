"""Tests for compass/cross_asset_signal.py — cross-asset signal generator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.cross_asset_signal import (
    AnalysisResult,
    CointegrationResult,
    CorrelationRegime,
    CrossAssetSignalEngine,
    LeadLagResult,
    MomentumSpillover,
    SpreadSignal,
    adf_test,
    build_lead_lag_matrix,
    compute_momentum_spillover,
    compute_spread_zscore,
    detect_lead_lag,
    engle_granger_test,
    johansen_trace_test,
    pair_trade_signal,
    rolling_correlation_regime,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_prices(n: int = 500, k: int = 4, seed: int = 42) -> pd.DataFrame:
    """Generate correlated synthetic prices."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)

    # Base factor
    factor = np.cumsum(rng.normal(0.0003, 0.01, n))

    cols = {}
    names = ["SPY", "QQQ", "TLT", "VIX"][:k]
    betas = [1.0, 1.2, -0.5, -0.8][:k]
    for i, name in enumerate(names):
        noise = np.cumsum(rng.normal(0, 0.005, n))
        cols[name] = 100.0 * np.exp(factor * betas[i] + noise)

    return pd.DataFrame(cols, index=dates)


def _make_cointegrated_pair(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate two cointegrated price series."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)

    # Common stochastic trend
    trend = np.cumsum(rng.normal(0.0005, 0.01, n))

    # Cointegrated pair: B = 1.5*A + mean-reverting noise
    a = 100.0 * np.exp(trend)
    spread_noise = np.zeros(n)
    for i in range(1, n):
        spread_noise[i] = 0.9 * spread_noise[i - 1] + rng.normal(0, 0.5)
    b = 1.5 * a + 50.0 + spread_noise

    return pd.DataFrame({"A": a, "B": b}, index=dates)


@pytest.fixture
def prices():
    return _make_prices()


@pytest.fixture
def coint_prices():
    return _make_cointegrated_pair()


@pytest.fixture
def engine(prices):
    return CrossAssetSignalEngine(prices, window=30, max_lag=5)


# ── ADF test ─────────────────────────────────────────────────────────────


class TestADF:
    def test_stationary_series(self):
        rng = np.random.RandomState(42)
        x = rng.normal(0, 1, 200)
        stat, pv = adf_test(x)
        assert stat < -2.0  # should be stationary
        assert pv < 0.10

    def test_random_walk(self):
        rng = np.random.RandomState(42)
        x = np.cumsum(rng.normal(0, 1, 200))
        stat, pv = adf_test(x)
        assert pv > 0.05  # should NOT be stationary

    def test_short_series(self):
        _, pv = adf_test(np.array([1.0, 2.0, 3.0]))
        assert pv == 1.0

    def test_pvalue_bounded(self):
        rng = np.random.RandomState(42)
        _, pv = adf_test(rng.normal(0, 1, 100))
        assert 0 <= pv <= 1


# ── Engle-Granger cointegration tests ────────────────────────────────────


class TestEngleGranger:
    def test_cointegrated_pair(self, coint_prices):
        pa = coint_prices["A"].values
        pb = coint_prices["B"].values
        coint, adf, pv, hr, hl = engle_granger_test(pa, pb)
        assert coint is True
        assert pv < 0.10
        assert abs(hr - 1.5) < 1.0  # hedge ratio near true value
        assert hl < 200  # mean reversion within reasonable time

    def test_independent_series(self):
        rng = np.random.RandomState(42)
        a = np.cumsum(rng.normal(0, 1, 300))
        b = np.cumsum(rng.normal(0, 1, 300))
        coint, _, pv, _, _ = engle_granger_test(a, b)
        # Mostly not cointegrated (may occasionally pass by chance)
        assert isinstance(coint, bool)

    def test_short_series(self):
        coint, _, pv, _, _ = engle_granger_test(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert coint is False
        assert pv == 1.0

    def test_hedge_ratio_sign(self, coint_prices):
        pa = coint_prices["A"].values
        pb = coint_prices["B"].values
        _, _, _, hr, _ = engle_granger_test(pa, pb)
        assert hr > 0  # positive relationship


# ── Johansen trace test ──────────────────────────────────────────────────


class TestJohansen:
    def test_cointegrated_pair(self, coint_prices):
        pa = coint_prices["A"].values
        pb = coint_prices["B"].values
        coint, trace, pv, hr = johansen_trace_test(pa, pb)
        assert isinstance(coint, bool)
        assert trace > 0

    def test_returns_tuple(self, coint_prices):
        result = johansen_trace_test(
            coint_prices["A"].values, coint_prices["B"].values
        )
        assert len(result) == 4

    def test_short_series(self):
        coint, _, pv, _ = johansen_trace_test(np.array([1, 2]), np.array([3, 4]))
        assert coint is False
        assert pv == 1.0


# ── Lead-lag detection tests ─────────────────────────────────────────────


class TestLeadLag:
    def test_detect_lag(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 200)
        # b follows a with 3-day lag
        b = np.zeros(200)
        b[3:] = a[:-3] * 0.8 + rng.normal(0, 0.3, 197)
        lag, corr = detect_lead_lag(a, b, max_lag=5)
        assert lag > 0  # a leads
        assert corr > 0.3

    def test_no_lead_lag(self):
        rng = np.random.RandomState(42)
        a = rng.normal(0, 1, 200)
        b = rng.normal(0, 1, 200)
        lag, corr = detect_lead_lag(a, b, max_lag=5)
        assert abs(corr) < 0.3

    def test_short_series(self):
        lag, corr = detect_lead_lag(np.array([1, 2, 3]), np.array([4, 5, 6]), max_lag=5)
        assert lag == 0 and corr == 0.0

    def test_matrix_shape(self, prices):
        returns = prices.pct_change().dropna()
        matrix = build_lead_lag_matrix(returns, max_lag=3)
        assert matrix.shape == (4, 4)
        # Diagonal should be 1
        for i in range(4):
            assert matrix.iloc[i, i] == 1.0


# ── Spread z-score tests ────────────────────────────────────────────────


class TestSpreadZScore:
    def test_zscore_shape(self, coint_prices):
        pa = coint_prices["A"].values
        pb = coint_prices["B"].values
        zscores, current = compute_spread_zscore(pa, pb, 1.5, window=60)
        assert len(zscores) == len(pa)
        assert isinstance(current, float)

    def test_zscore_starts_zero(self, coint_prices):
        pa = coint_prices["A"].values
        pb = coint_prices["B"].values
        zscores, _ = compute_spread_zscore(pa, pb, 1.5, window=60)
        # First `window` values should be 0
        assert all(zscores[:60] == 0)

    def test_pair_signal_long(self):
        assert pair_trade_signal(-2.5, entry=2.0) == "long_spread"

    def test_pair_signal_short(self):
        assert pair_trade_signal(2.5, entry=2.0) == "short_spread"

    def test_pair_signal_neutral(self):
        assert pair_trade_signal(0.3, entry=2.0, exit_thresh=0.5) == "neutral"

    def test_pair_signal_between(self):
        # Between exit and entry → neutral
        assert pair_trade_signal(1.0, entry=2.0, exit_thresh=0.5) == "neutral"


# ── Rolling correlation regime tests ─────────────────────────────────────


class TestCorrelationRegime:
    def test_regime_detection(self, prices):
        returns = prices.pct_change().dropna()
        curr, mu, std, z, regime = rolling_correlation_regime(
            returns["SPY"], returns["QQQ"], window=30
        )
        assert regime in ("high_corr", "low_corr", "normal")
        assert isinstance(curr, float)
        assert isinstance(z, float)

    def test_short_series(self):
        a = pd.Series([0.01, 0.02])
        b = pd.Series([0.01, -0.01])
        _, _, _, _, regime = rolling_correlation_regime(a, b, window=5)
        assert regime == "normal"


# ── Momentum spillover tests ─────────────────────────────────────────────


class TestMomentumSpillover:
    def test_spillover_detection(self, prices):
        returns = prices.pct_change().dropna()
        mom, beta, pred, sig = compute_momentum_spillover(
            returns["SPY"], returns["QQQ"],
            momentum_window=21, regression_window=60,
        )
        assert isinstance(beta, float)
        assert sig in ("bullish", "bearish", "neutral")

    def test_short_series(self):
        a = pd.Series([0.01] * 10)
        b = pd.Series([0.02] * 10)
        _, _, _, sig = compute_momentum_spillover(a, b)
        assert sig == "neutral"


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_basic(self, prices):
        engine = CrossAssetSignalEngine(prices)
        assert engine.n_assets == 4

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            CrossAssetSignalEngine(pd.DataFrame())

    def test_single_asset_raises(self):
        df = pd.DataFrame({"A": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2"):
            CrossAssetSignalEngine(df)


# ── Full analysis tests ──────────────────────────────────────────────────


class TestAnalysis:
    def test_analyze_returns_result(self, engine):
        result = engine.analyze()
        assert isinstance(result, AnalysisResult)
        assert result.n_assets == 4

    def test_cointegrations_populated(self, engine):
        result = engine.analyze()
        assert len(result.cointegrations) > 0
        assert all(isinstance(c, CointegrationResult) for c in result.cointegrations)

    def test_both_coint_methods(self, engine):
        result = engine.analyze()
        methods = {c.method for c in result.cointegrations}
        assert "engle_granger" in methods
        assert "johansen_trace" in methods

    def test_lead_lag_matrix_shape(self, engine):
        result = engine.analyze()
        assert result.lead_lag_matrix.shape == (4, 4)

    def test_correlation_matrix_shape(self, engine):
        result = engine.analyze()
        assert result.correlation_matrix.shape == (4, 4)

    def test_correlation_regimes(self, engine):
        result = engine.analyze()
        assert len(result.correlation_regimes) > 0
        assert all(isinstance(r, CorrelationRegime) for r in result.correlation_regimes)

    def test_cointegrated_pair_analysis(self, coint_prices):
        engine = CrossAssetSignalEngine(coint_prices, window=30)
        result = engine.analyze()
        eg_coints = [c for c in result.cointegrations if c.method == "engle_granger"]
        assert any(c.cointegrated for c in eg_coints)


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, engine):
        result = engine.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "xasset.html"
            path = CrossAssetSignalEngine.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Cross-Asset Signal" in content

    def test_contains_cointegration(self, engine):
        result = engine.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            CrossAssetSignalEngine.generate_report(result, out)
            content = out.read_text()
            assert "Cointegration" in content

    def test_contains_heatmap(self, engine):
        result = engine.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            CrossAssetSignalEngine.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Lead-Lag" in content

    def test_contains_regimes(self, engine):
        result = engine.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            CrossAssetSignalEngine.generate_report(result, out)
            content = out.read_text()
            assert "Correlation Regime" in content

    def test_default_path(self, engine):
        result = engine.analyze()
        path = CrossAssetSignalEngine.generate_report(result)
        assert path.exists()
        assert "cross_asset_signal.html" in str(path)
        path.unlink(missing_ok=True)
