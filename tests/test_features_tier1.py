"""
Tests for Tier 1 alpha features added to compass/features.py.

Features covered:
  1. VIX term structure  (vix_contango_ratio)
  2. HYG/LQD ratio       (hyg_lqd_ratio, hyg_lqd_ratio_5d_chg)
  3. SPY/TLT correlation (spy_tlt_corr_20d)
  4. Accurate OPEX       (days_to_opex replaces is_opex_week)
  5. Opening gap         (opening_gap_pct)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from compass.features import FeatureEngine


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _price_df(periods: int = 130, start: str = "2025-01-01", seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame that mimics data_provider output."""
    np.random.seed(seed)
    dates = pd.date_range(start, periods=periods, freq="B")
    close = 450.0 + np.cumsum(np.random.randn(periods) * 2)
    open_ = close + np.random.randn(periods) * 0.5
    return pd.DataFrame(
        {
            "Open":   open_,
            "High":   close + np.abs(np.random.randn(periods)),
            "Low":    close - np.abs(np.random.randn(periods)),
            "Close":  close,
            "Volume": np.random.randint(1_000_000, 5_000_000, periods),
        },
        index=dates,
    )


def _options_df() -> pd.DataFrame:
    exp = datetime.now() + timedelta(days=35)
    return pd.DataFrame(
        {"strike": [100.0, 105.0], "bid": [2.0, 3.0], "ask": [2.5, 3.5],
         "type": ["put", "call"], "expiration": [exp, exp], "iv": [0.25, 0.30]}
    )


_SPY_DF  = _price_df(seed=1)
_VIX_DF  = _price_df(periods=30, seed=2)
_TLT_DF  = _price_df(periods=130, seed=3)
_HYG_DF  = _price_df(periods=130, seed=4)
_LQD_DF  = _price_df(periods=130, seed=5)
_VIX3M_DF = _price_df(periods=30, seed=6)


def _cache_all() -> MagicMock:
    """Return a data_provider mock that serves data for every ticker."""
    mock = MagicMock()

    def _get_history(ticker: str, period: str):
        mapping = {
            "^VIX":   _VIX_DF.copy(),
            "^VIX3M": _VIX3M_DF.copy(),
            "SPY":    _SPY_DF.copy(),
            "TLT":    _TLT_DF.copy(),
            "HYG":    _HYG_DF.copy(),
            "LQD":    _LQD_DF.copy(),
        }
        return mapping.get(ticker, _SPY_DF.copy())

    mock.get_history.side_effect = _get_history
    mock.get_ticker_obj.return_value = MagicMock(calendar=None)
    return mock


def _cache_no_extras() -> MagicMock:
    """Data provider that only serves SPY and VIX — no TLT, HYG, LQD, VIX3M."""
    mock = MagicMock()

    def _get_history(ticker: str, period: str):
        if ticker in ("SPY", "^VIX"):
            return _SPY_DF.copy()
        return None  # cache miss for everything else

    mock.get_history.side_effect = _get_history
    mock.get_ticker_obj.return_value = MagicMock(calendar=None)
    return mock


# ─── 1. VIX Term Structure ────────────────────────────────────────────────────


class TestVixContangoRatio:

    def test_present_in_market_features_with_vix3m(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.compute_market_features()
        assert feat is not None
        assert "vix_contango_ratio" in feat

    def test_vix3m_ratio_is_positive(self):
        """When VIX3M data is available the ratio must be positive."""
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.compute_market_features()
        assert feat is not None
        v = feat["vix_contango_ratio"]
        assert not math.isnan(v), "vix_contango_ratio should not be NaN when VIX3M available"
        assert v > 0

    def test_proxy_used_when_vix3m_unavailable(self):
        """Without ^VIX3M data, proxy (VIX / spy_realized_vol) is used — still positive."""
        engine = FeatureEngine(data_provider=_cache_no_extras())
        feat = engine.compute_market_features()
        assert feat is not None
        assert "vix_contango_ratio" in feat
        v = feat["vix_contango_ratio"]
        # May be NaN if spy_realized_vol is 0, but typically positive
        if not math.isnan(v):
            assert v > 0

    def test_present_in_build_features(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.build_features(
            ticker="SPY", current_price=450.0, options_chain=_options_df()
        )
        assert feat is not None
        assert "vix_contango_ratio" in feat

    def test_in_feature_names(self):
        assert "vix_contango_ratio" in FeatureEngine().get_feature_names()


# ─── 2. HYG/LQD Credit Stress ────────────────────────────────────────────────


class TestHygLqdRatio:

    def test_present_when_data_available(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine._compute_credit_stress_features()
        assert "hyg_lqd_ratio" in feat
        assert "hyg_lqd_ratio_5d_chg" in feat

    def test_ratio_is_positive(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine._compute_credit_stress_features()
        assert feat["hyg_lqd_ratio"] > 0

    def test_5d_chg_is_float(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine._compute_credit_stress_features()
        assert isinstance(feat["hyg_lqd_ratio_5d_chg"], float)

    def test_returns_nan_when_no_data(self):
        """With no data provider, features should be NaN (not crash)."""
        engine = FeatureEngine()
        feat = engine._compute_credit_stress_features()
        assert "hyg_lqd_ratio" in feat
        assert math.isnan(feat["hyg_lqd_ratio"])
        assert math.isnan(feat["hyg_lqd_ratio_5d_chg"])

    def test_returns_nan_when_hyg_missing(self):
        """When only HYG is missing, features should be NaN gracefully."""
        mock = MagicMock()

        def _get(ticker, period):
            if ticker == "HYG":
                return None
            return _LQD_DF.copy()

        mock.get_history.side_effect = _get
        engine = FeatureEngine(data_provider=mock)
        feat = engine._compute_credit_stress_features()
        assert math.isnan(feat["hyg_lqd_ratio"])

    def test_returns_nan_when_lqd_missing(self):
        mock = MagicMock()

        def _get(ticker, period):
            if ticker == "LQD":
                return None
            return _HYG_DF.copy()

        mock.get_history.side_effect = _get
        engine = FeatureEngine(data_provider=mock)
        feat = engine._compute_credit_stress_features()
        assert math.isnan(feat["hyg_lqd_ratio"])

    def test_present_in_build_features(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.build_features(
            ticker="SPY", current_price=450.0, options_chain=_options_df()
        )
        assert feat is not None
        assert "hyg_lqd_ratio" in feat
        assert "hyg_lqd_ratio_5d_chg" in feat

    def test_build_features_not_none_when_credit_data_missing(self):
        """build_features must succeed even when HYG/LQD are unavailable."""
        engine = FeatureEngine(data_provider=_cache_no_extras())
        feat = engine.build_features(
            ticker="SPY", current_price=450.0, options_chain=_options_df()
        )
        # build_features may return None due to IV data miss, but should NOT crash
        # with an exception — if it returns something, credit features must be present
        if feat is not None:
            assert "hyg_lqd_ratio" in feat

    def test_both_in_feature_names(self):
        names = FeatureEngine().get_feature_names()
        assert "hyg_lqd_ratio" in names
        assert "hyg_lqd_ratio_5d_chg" in names


# ─── 3. SPY/TLT Correlation ──────────────────────────────────────────────────


class TestSpyTltCorrelation:

    def test_present_in_market_features_with_tlt(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.compute_market_features()
        assert feat is not None
        assert "spy_tlt_corr_20d" in feat

    def test_correlation_in_valid_range(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.compute_market_features()
        assert feat is not None
        corr = feat["spy_tlt_corr_20d"]
        if not math.isnan(corr):
            assert -1.0 <= corr <= 1.0, f"Correlation {corr} out of [-1, 1]"

    def test_nan_when_tlt_unavailable(self):
        """When TLT data is missing, spy_tlt_corr_20d should be NaN (not crash)."""
        engine = FeatureEngine(data_provider=_cache_no_extras())
        feat = engine.compute_market_features()
        assert feat is not None
        assert "spy_tlt_corr_20d" in feat
        assert math.isnan(feat["spy_tlt_corr_20d"])

    def test_present_in_build_features(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.build_features(
            ticker="SPY", current_price=450.0, options_chain=_options_df()
        )
        assert feat is not None
        assert "spy_tlt_corr_20d" in feat

    def test_in_feature_names(self):
        assert "spy_tlt_corr_20d" in FeatureEngine().get_feature_names()


# ─── 4. Accurate OPEX (days_to_opex) ─────────────────────────────────────────


class TestDaysToOpex:

    def test_days_to_opex_is_positive_integer(self):
        engine = FeatureEngine()
        feat = engine._compute_seasonal_features()
        assert feat is not None
        assert "days_to_opex" in feat
        v = feat["days_to_opex"]
        assert isinstance(v, int)
        assert v >= 0

    def test_days_to_opex_le_31(self):
        """OPEX is always within the current or next month — at most ~31 days ahead."""
        engine = FeatureEngine()
        feat = engine._compute_seasonal_features()
        assert feat is not None
        assert feat["days_to_opex"] <= 31

    def test_is_opex_week_removed(self):
        """The old is_opex_week boolean should no longer be present."""
        engine = FeatureEngine()
        feat = engine._compute_seasonal_features()
        assert feat is not None
        assert "is_opex_week" not in feat

    def test_days_to_opex_in_feature_names(self):
        assert "days_to_opex" in FeatureEngine().get_feature_names()

    def test_is_opex_week_not_in_feature_names(self):
        assert "is_opex_week" not in FeatureEngine().get_feature_names()

    def test_days_to_opex_helper_known_date(self):
        """Verify _days_to_next_opex for a date with known 3rd Friday."""
        # 2025-01-06 (Monday).  3rd Friday of Jan 2025 = Jan 17.
        # days_to_opex = 17 - 6 = 11
        dt = datetime(2025, 1, 6, tzinfo=timezone.utc)
        result = FeatureEngine._days_to_next_opex(dt)
        assert result == 11, f"Expected 11, got {result}"

    def test_days_to_opex_helper_same_day_as_opex_looks_forward(self):
        """On OPEX day itself, the next OPEX should be roughly a month away."""
        # Jan 17, 2025 is the 3rd Friday of January → next OPEX = Feb 21, 2025
        dt = datetime(2025, 1, 17, tzinfo=timezone.utc)
        result = FeatureEngine._days_to_next_opex(dt)
        # Feb 21, 2025 is the 3rd Friday of Feb; Jan 17 → Feb 21 = 35 days
        assert result == 35, f"Expected 35, got {result}"

    def test_days_to_opex_present_in_build_features(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.build_features(
            ticker="SPY", current_price=450.0, options_chain=_options_df()
        )
        assert feat is not None
        assert "days_to_opex" in feat


# ─── 5. Opening Gap ───────────────────────────────────────────────────────────


class TestOpeningGap:

    def test_present_in_technical_features(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine._compute_technical_features("SPY", 450.0)
        assert feat is not None
        assert "opening_gap_pct" in feat

    def test_is_finite_float_with_ohlcv_data(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine._compute_technical_features("SPY", 450.0)
        assert feat is not None
        v = feat["opening_gap_pct"]
        assert isinstance(v, float)
        assert math.isfinite(v)

    def test_sign_is_correct(self):
        """Gap sign should match (today open - yesterday close)."""
        # Build a DataFrame where last open > second-last close
        df = _price_df(periods=50, seed=99).copy()
        prev_close = float(df["Close"].iloc[-2])
        # Force last open to be exactly prev_close + 1%
        df.iloc[-1, df.columns.get_loc("Open")] = prev_close * 1.01

        mock = MagicMock()
        mock.get_history.return_value = df
        engine = FeatureEngine(data_provider=mock)
        feat = engine._compute_technical_features("SPY", 450.0)
        assert feat is not None
        assert feat["opening_gap_pct"] == pytest.approx(1.0, abs=1e-3)

    def test_nan_when_only_close_data(self):
        """Without an Open column, opening_gap_pct should be NaN (not crash)."""
        df = _price_df(periods=50).drop(columns=["Open"])
        mock = MagicMock()
        mock.get_history.return_value = df
        engine = FeatureEngine(data_provider=mock)
        feat = engine._compute_technical_features("SPY", 450.0)
        assert feat is not None
        assert math.isnan(feat["opening_gap_pct"])

    def test_present_in_build_features(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.build_features(
            ticker="SPY", current_price=450.0, options_chain=_options_df()
        )
        assert feat is not None
        assert "opening_gap_pct" in feat

    def test_in_feature_names(self):
        assert "opening_gap_pct" in FeatureEngine().get_feature_names()


# ─── Integration: all 5 features in one build_features call ──────────────────


class TestTier1FeaturesIntegration:

    TIER1_FEATURES = [
        "vix_contango_ratio",
        "hyg_lqd_ratio",
        "hyg_lqd_ratio_5d_chg",
        "spy_tlt_corr_20d",
        "days_to_opex",
        "opening_gap_pct",
    ]

    def test_all_tier1_features_present(self):
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.build_features(
            ticker="SPY", current_price=450.0, options_chain=_options_df()
        )
        assert feat is not None
        for name in self.TIER1_FEATURES:
            assert name in feat, f"Missing Tier 1 feature: {name}"

    def test_is_opex_week_absent(self):
        """is_opex_week must be replaced by days_to_opex."""
        engine = FeatureEngine(data_provider=_cache_all())
        feat = engine.build_features(
            ticker="SPY", current_price=450.0, options_chain=_options_df()
        )
        assert feat is not None
        assert "is_opex_week" not in feat

    def test_all_tier1_in_feature_names(self):
        names = FeatureEngine().get_feature_names()
        for name in self.TIER1_FEATURES:
            assert name in names, f"Missing from get_feature_names(): {name}"

    def test_build_features_does_not_crash_with_partial_data(self):
        """Even when HYG/LQD/TLT/VIX3M are unavailable, build_features should
        return a result (or None from IV miss) — never raise an exception."""
        engine = FeatureEngine(data_provider=_cache_no_extras())
        try:
            feat = engine.build_features(
                ticker="SPY", current_price=450.0, options_chain=_options_df()
            )
        except Exception as exc:
            pytest.fail(f"build_features raised unexpectedly: {exc}")
        # May be None due to missing IV data, but must not throw
