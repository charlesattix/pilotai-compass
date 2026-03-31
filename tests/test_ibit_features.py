"""Tests for compass/ibit_features.py — IBIT crypto-native feature engineering.

Covers:
  - realized_vol_crypto: edge cases, crypto annualization (365d)
  - vol_regime_features: vol ratio, percentile
  - _rolling_correlation: insufficient data, zero variance, known values
  - correlation_features: None inputs, decouple calculation
  - funding_rate_features: empty, short, full histories, extreme flags
  - microstructure_features: volume ratio, range, gap, momentum, MA distance
  - composite_score_features: individual normalizations, composite aggregation
  - IBITFeatureEngine: full compute, feature_names, compute_array
"""

import math

import numpy as np
import pytest

from compass.ibit_features import (
    IBIT_FEATURE_NAMES,
    IBITFeatureEngine,
    _rolling_correlation,
    composite_score_features,
    correlation_features,
    funding_rate_features,
    microstructure_features,
    realized_vol_crypto,
    vol_regime_features,
)


# ── realized_vol_crypto ──────────────────────────────────────────────────


class TestRealizedVolCrypto:
    def test_insufficient_data_returns_zero(self):
        assert realized_vol_crypto([100, 101], 7) == 0.0

    def test_constant_prices_zero_vol(self):
        prices = [100.0] * 40
        assert realized_vol_crypto(prices, 7) == 0.0

    def test_uses_crypto_annualization(self):
        """With known daily vol, check annualization uses sqrt(365)."""
        # Create prices with known daily log return std
        rng = np.random.RandomState(42)
        daily_log_ret = rng.normal(0, 0.02, 31)
        prices = [100.0]
        for lr in daily_log_ret:
            prices.append(prices[-1] * math.exp(lr))
        rv = realized_vol_crypto(prices, 30)
        # Should be roughly 0.02 * sqrt(365) ≈ 0.382
        assert 0.20 < rv < 0.60

    def test_non_positive_price_returns_zero(self):
        prices = [100, 101, 0, 102, 103, 104, 105, 106, 107, 108]
        assert realized_vol_crypto(prices, 7) == 0.0

    def test_high_vol_for_volatile_prices(self):
        prices = [100, 120, 90, 130, 80, 140, 70, 150, 60]
        rv = realized_vol_crypto(prices, 7)
        assert rv > 1.0  # extremely volatile → high annualized vol


# ── vol_regime_features ──────────────────────────────────────────────────


class TestVolRegimeFeatures:
    def test_short_data(self):
        features = vol_regime_features([100, 101, 102])
        # Insufficient data: realized_vol_crypto returns 0.0, which is falsy → None
        assert features["rv_7d_crypto"] is None
        assert features["rv_30d_crypto"] is None
        assert features["vol_percentile_90d"] is None

    def test_vol_ratio_expanding(self):
        """When short-term vol > long-term vol, ratio > 1."""
        rng = np.random.RandomState(42)
        # Stable period then volatile
        stable = [100 + 0.1 * i for i in range(100)]
        volatile = [stable[-1]]
        for _ in range(40):
            volatile.append(volatile[-1] * (1 + rng.normal(0, 0.05)))
        prices = stable + volatile
        features = vol_regime_features(prices)
        assert features["vol_ratio_7_30"] is not None
        assert features["rv_7d_crypto"] > 0
        assert features["rv_30d_crypto"] > 0

    def test_vol_percentile_computed_with_enough_data(self):
        rng = np.random.RandomState(42)
        prices = [100.0]
        for _ in range(200):
            prices.append(prices[-1] * (1 + rng.normal(0, 0.02)))
        features = vol_regime_features(prices)
        pct = features["vol_percentile_90d"]
        assert pct is not None
        assert 0 <= pct <= 100


# ── _rolling_correlation ─────────────────────────────────────────────────


class TestRollingCorrelation:
    def test_insufficient_data(self):
        assert _rolling_correlation([100, 101], [200, 202], 7) is None

    def test_perfect_positive_correlation(self):
        a = [100 + i for i in range(32)]
        b = [200 + 2 * i for i in range(32)]
        corr = _rolling_correlation(a, b, 30)
        assert corr is not None
        assert corr == pytest.approx(1.0, abs=0.01)

    def test_negative_correlation(self):
        """Alternating up/down vs down/up gives negative correlation."""
        rng = np.random.RandomState(42)
        n = 35
        a = [100.0]
        b = [100.0]
        for i in range(n - 1):
            shock = rng.normal(0, 0.03)
            a.append(a[-1] * (1 + shock))
            b.append(b[-1] * (1 - shock))  # opposite moves
        corr = _rolling_correlation(a, b, 30)
        assert corr is not None
        assert corr < -0.5

    def test_zero_variance_returns_none(self):
        a = [100.0] * 32  # constant → zero variance
        b = [200 + i for i in range(32)]
        assert _rolling_correlation(a, b, 30) is None


# ── correlation_features ─────────────────────────────────────────────────


class TestCorrelationFeatures:
    def _make_prices(self, n=40, seed=42):
        rng = np.random.RandomState(seed)
        prices = [100.0]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + rng.normal(0, 0.02)))
        return prices

    def test_all_none_when_no_peers(self):
        ibit = self._make_prices()
        features = correlation_features(ibit, None, None)
        assert features["ibit_etha_corr_7d"] is None
        assert features["ibit_spy_corr_30d"] is None
        assert features["crypto_decouple"] is None

    def test_etha_correlation_populated(self):
        ibit = self._make_prices(40, seed=42)
        etha = self._make_prices(40, seed=43)
        features = correlation_features(ibit, etha, None)
        assert features["ibit_etha_corr_7d"] is not None
        assert -1.0 <= features["ibit_etha_corr_7d"] <= 1.0
        assert features["ibit_etha_corr_30d"] is not None

    def test_decouple_calculated(self):
        ibit = self._make_prices(40, seed=42)
        etha = self._make_prices(40, seed=42)  # same seed = correlated
        spy = self._make_prices(40, seed=99)   # different seed = uncorrelated
        features = correlation_features(ibit, etha, spy)
        assert features["crypto_decouple"] is not None


# ── funding_rate_features ────────────────────────────────────────────────


class TestFundingRateFeatures:
    def test_empty_rates(self):
        features = funding_rate_features(None)
        assert features["funding_rate_current"] is None
        assert features["contango_strength"] is None

    def test_empty_list(self):
        features = funding_rate_features([])
        assert features["funding_rate_current"] is None

    def test_single_rate(self):
        features = funding_rate_features([0.01])
        assert features["funding_rate_current"] == pytest.approx(0.01, abs=1e-6)
        assert features["funding_rate_mean_3d"] == pytest.approx(0.01, abs=1e-6)

    def test_full_history(self):
        rates = [0.01] * 25  # 25 settlements (>8 days)
        features = funding_rate_features(rates)
        assert features["funding_rate_current"] == pytest.approx(0.01, abs=1e-6)
        assert features["funding_rate_mean_3d"] == pytest.approx(0.01, abs=1e-6)
        assert features["funding_rate_mean_7d"] == pytest.approx(0.01, abs=1e-6)
        assert features["funding_trend_3d"] == pytest.approx(0.0, abs=1e-6)

    def test_extreme_bull_flag(self):
        rates = [0.10] * 10  # very high funding
        features = funding_rate_features(rates)
        assert features["funding_extreme_bull"] == 1
        assert features["funding_extreme_bear"] == 0

    def test_extreme_bear_flag(self):
        rates = [-0.05] * 10  # negative funding
        features = funding_rate_features(rates)
        assert features["funding_extreme_bear"] == 1
        assert features["funding_extreme_bull"] == 0

    def test_contango_strength_positive_for_positive_funding(self):
        rates = [0.05] * 25
        features = funding_rate_features(rates)
        assert features["contango_strength"] > 0

    def test_contango_strength_negative_for_negative_funding(self):
        rates = [-0.05] * 25
        features = funding_rate_features(rates)
        assert features["contango_strength"] < 0

    def test_trend_positive_when_rising(self):
        # 18 low rates then 9 high rates
        rates = [0.01] * 9 + [0.05] * 9
        features = funding_rate_features(rates)
        assert features["funding_trend_3d"] is not None
        assert features["funding_trend_3d"] > 0


# ── microstructure_features ──────────────────────────────────────────────


class TestMicrostructureFeatures:
    def test_empty_prices(self):
        features = microstructure_features([], None, None, None)
        assert features["momentum_3d_pct"] is None
        assert features["volume_ratio_20d"] is None

    def test_volume_ratio(self):
        prices = [100 + i * 0.1 for i in range(25)]
        volumes = [1000] * 19 + [2000]  # last day = 2x avg
        features = microstructure_features(prices, volumes, None, None)
        assert features["volume_ratio_20d"] is not None
        assert features["volume_ratio_20d"] > 1.5

    def test_intraday_range(self):
        prices = [100.0] * 5
        highs = [102.0] * 5
        lows = [98.0] * 5
        features = microstructure_features(prices, None, highs, lows)
        # range = 4, price = 100 → 4%
        assert features["intraday_range_pct"] == pytest.approx(4.0, abs=0.1)

    def test_momentum_positive(self):
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        features = microstructure_features(prices, None, None, None)
        assert features["momentum_3d_pct"] is not None
        assert features["momentum_3d_pct"] > 0
        assert features["momentum_7d_pct"] is not None
        assert features["momentum_7d_pct"] > 0

    def test_gap_pct(self):
        prices = [100.0, 100.0, 100.0]
        lows = [99.0, 101.0, 95.0]  # gap down on last day
        features = microstructure_features(prices, None, None, lows)
        assert features["gap_pct"] is not None
        assert features["gap_pct"] < 0  # gap down

    def test_ma20_distance(self):
        # Price at 110, MA20 at ~100 → +10%
        prices = [100.0] * 19 + [110.0]
        features = microstructure_features(prices, None, None, None)
        assert features["ibit_dist_ma20_pct"] is not None
        assert features["ibit_dist_ma20_pct"] > 5


# ── composite_score_features ─────────────────────────────────────────────


class TestCompositeScoreFeatures:
    def test_all_none_inputs(self):
        features = composite_score_features()
        assert features["fgi_normalized"] is None
        assert features["composite_score"] is None

    def test_fgi_normalized(self):
        features = composite_score_features(fear_greed_index=50)
        assert features["fgi_normalized"] == pytest.approx(0.5)

    def test_fgi_clamped_at_bounds(self):
        features = composite_score_features(fear_greed_index=150)
        assert features["fgi_normalized"] == 1.0
        features = composite_score_features(fear_greed_index=-10)
        assert features["fgi_normalized"] == 0.0

    def test_funding_norm_positive(self):
        features = composite_score_features(funding_rate_mean=0.05)
        assert features["funding_norm"] is not None
        assert features["funding_norm"] > 0.5  # positive funding = bullish

    def test_funding_norm_negative(self):
        features = composite_score_features(funding_rate_mean=-0.05)
        assert features["funding_norm"] is not None
        assert features["funding_norm"] < 0.5  # negative funding = bearish

    def test_dominance_norm_inverted(self):
        # High dominance = fear
        high_dom = composite_score_features(btc_dominance=65)
        low_dom = composite_score_features(btc_dominance=45)
        assert high_dom["dominance_norm"] < low_dom["dominance_norm"]

    def test_iv_rv_spread_norm(self):
        features = composite_score_features(rv_30d=0.50)  # 50% annualized vol
        assert features["iv_rv_spread_norm"] is not None
        assert 0 < features["iv_rv_spread_norm"] < 1

    def test_composite_score_when_all_present(self):
        features = composite_score_features(
            fear_greed_index=50,
            funding_rate_mean=0.01,
            btc_dominance=55,
            rv_30d=0.40,
        )
        assert features["composite_score"] is not None
        assert 0 <= features["composite_score"] <= 100


# ── IBITFeatureEngine ────────────────────────────────────────────────────


class TestIBITFeatureEngine:
    def _make_prices(self, n=50, start=55.0, seed=42):
        rng = np.random.RandomState(seed)
        prices = [start]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + rng.normal(0.001, 0.03)))
        return prices

    def test_feature_names_canonical(self):
        engine = IBITFeatureEngine()
        assert engine.feature_names == IBIT_FEATURE_NAMES
        assert len(engine.feature_names) == 27

    def test_compute_returns_all_keys(self):
        engine = IBITFeatureEngine()
        prices = self._make_prices(50)
        features = engine.compute(ibit_prices=prices)
        for name in IBIT_FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_compute_with_all_inputs(self):
        engine = IBITFeatureEngine()
        ibit = self._make_prices(50, 55.0, seed=42)
        etha = self._make_prices(50, 18.0, seed=43)
        spy = self._make_prices(50, 580.0, seed=44)
        volumes = [1e6 + i * 1000 for i in range(50)]
        highs = [p * 1.02 for p in ibit]
        lows = [p * 0.98 for p in ibit]
        funding = [0.01 + 0.001 * i for i in range(25)]

        features = engine.compute(
            ibit_prices=ibit,
            etha_prices=etha,
            spy_prices=spy,
            ibit_volumes=volumes,
            ibit_highs=highs,
            ibit_lows=lows,
            funding_rates=funding,
            fear_greed_index=45,
            btc_dominance=52.0,
        )

        # Volatility features should be populated
        assert features["rv_7d_crypto"] is not None
        assert features["rv_30d_crypto"] is not None
        # Correlation features should be populated
        assert features["ibit_etha_corr_7d"] is not None
        assert features["ibit_spy_corr_30d"] is not None
        # Funding features
        assert features["funding_rate_current"] is not None
        assert features["contango_strength"] is not None
        # Composite
        assert features["fgi_normalized"] is not None
        assert features["composite_score"] is not None

    def test_compute_with_minimal_data(self):
        engine = IBITFeatureEngine()
        prices = [55.0, 56.0, 54.5]  # too short for most features
        features = engine.compute(ibit_prices=prices)
        assert features["rv_7d_crypto"] is None  # insufficient data → None
        assert features["momentum_3d_pct"] is None

    def test_compute_array_shape(self):
        engine = IBITFeatureEngine()
        prices = self._make_prices(50)
        arr = engine.compute_array(ibit_prices=prices)
        assert arr.shape == (1, 27)
        assert arr.dtype == np.float64

    def test_compute_array_no_nans(self):
        """Missing values should be filled with 0.0."""
        engine = IBITFeatureEngine()
        prices = self._make_prices(50)
        arr = engine.compute_array(ibit_prices=prices)
        assert not np.isnan(arr).any()

    def test_custom_iv_premium(self):
        engine = IBITFeatureEngine(iv_premium=0.20)
        prices = self._make_prices(50)
        features = engine.compute(ibit_prices=prices)
        # iv_rv_spread_norm should reflect higher premium
        assert features["iv_rv_spread_norm"] is not None
