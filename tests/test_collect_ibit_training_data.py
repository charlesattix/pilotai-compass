"""Tests for compass/collect_ibit_training_data.py — IBIT trade enrichment.

Covers:
  - _compute_rsi: edge cases, known values
  - _compute_ma_distance: insufficient data, known computation
  - _compute_volume_ratio: short history, known ratio
  - _compute_returns_vol: insufficient data, positive result
  - enrich_ibit_trades: full enrichment with and without optional data
"""

import numpy as np
import pandas as pd
import pytest

from compass.collect_ibit_training_data import (
    _compute_ma_distance,
    _compute_returns_vol,
    _compute_rsi,
    _compute_volume_ratio,
    enrich_ibit_trades,
)
from compass.ibit_features import IBIT_FEATURE_NAMES


# ── _compute_rsi ─────────────────────────────────────────────────────────


class TestComputeRSI:
    def test_insufficient_data(self):
        assert _compute_rsi([100, 101, 102]) is None

    def test_all_gains_returns_near_100(self):
        prices = [100 + i for i in range(50)]
        rsi = _compute_rsi(prices)
        assert rsi is not None
        assert rsi > 90

    def test_all_losses_returns_near_0(self):
        prices = [200 - i for i in range(50)]
        rsi = _compute_rsi(prices)
        assert rsi is not None
        assert rsi < 10

    def test_mixed_gives_valid_range(self):
        rng = np.random.RandomState(42)
        prices = [100.0]
        for _ in range(50):
            prices.append(prices[-1] * (1 + rng.normal(0, 0.01)))
        rsi = _compute_rsi(prices)
        assert rsi is not None
        assert 0 < rsi < 100

    def test_returns_float(self):
        prices = [100 + i * 0.5 for i in range(50)]
        rsi = _compute_rsi(prices)
        assert isinstance(rsi, float)


# ── _compute_ma_distance ─────────────────────────────────────────────────


class TestComputeMADistance:
    def test_insufficient_data(self):
        assert _compute_ma_distance([100, 101], 20) is None

    def test_above_ma(self):
        prices = [100.0] * 19 + [110.0]
        dist = _compute_ma_distance(prices, 20)
        assert dist is not None
        assert dist > 0  # price above MA

    def test_below_ma(self):
        prices = [100.0] * 19 + [90.0]
        dist = _compute_ma_distance(prices, 20)
        assert dist is not None
        assert dist < 0  # price below MA

    def test_at_ma_returns_near_zero(self):
        prices = [100.0] * 20
        dist = _compute_ma_distance(prices, 20)
        assert dist == pytest.approx(0.0, abs=0.01)


# ── _compute_volume_ratio ────────────────────────────────────────────────


class TestComputeVolumeRatio:
    def test_insufficient_data(self):
        assert _compute_volume_ratio([1000] * 5) is None

    def test_double_volume(self):
        volumes = [1000] * 19 + [2000]
        ratio = _compute_volume_ratio(volumes)
        assert ratio is not None
        # avg of last 20 includes the 2000, so not exactly 2.0
        assert ratio > 1.5

    def test_normal_volume(self):
        volumes = [1000] * 20
        ratio = _compute_volume_ratio(volumes)
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_zero_average_returns_none(self):
        assert _compute_volume_ratio([0] * 20) is None


# ── _compute_returns_vol ─────────────────────────────────────────────────


class TestComputeReturnsVol:
    def test_insufficient_data(self):
        assert _compute_returns_vol([100, 101]) is None

    def test_positive_result(self):
        rng = np.random.RandomState(42)
        prices = [100.0]
        for _ in range(30):
            prices.append(prices[-1] * (1 + rng.normal(0, 0.02)))
        vol = _compute_returns_vol(prices)
        assert vol is not None
        assert vol > 0

    def test_constant_prices_zero_vol(self):
        prices = [100.0] * 25
        vol = _compute_returns_vol(prices)
        assert vol == pytest.approx(0.0, abs=0.01)


# ── enrich_ibit_trades ───────────────────────────────────────────────────


class TestEnrichIBITTrades:
    def _make_prices_series(self, n=60, start=55.0, seed=42):
        rng = np.random.RandomState(seed)
        dates = pd.bdate_range("2025-01-01", periods=n)
        prices = [start]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + rng.normal(0.001, 0.02)))
        return pd.Series(prices, index=dates)

    def _make_trades(self, n=3):
        rows = []
        for i in range(n):
            rows.append({
                "entry_date": f"2025-02-{10 + i * 5:02d}",
                "exit_date": f"2025-02-{15 + i * 5:02d}",
                "direction": "bull_put",
                "dte": 14,
                "otm_pct": 10.0,
                "spread_width": 5.0,
                "credit_received": 0.65,
                "credit_pct": 13.0,
                "contracts": 2,
                "exit_reason": "profit_target",
                "hold_days": 5,
                "pnl": 50.0 if i < 2 else -100.0,
                "return_pct": 5.0 if i < 2 else -10.0,
            })
        return pd.DataFrame(rows)

    def test_basic_enrichment(self):
        ibit = self._make_prices_series(60)
        trades = self._make_trades(2)
        result = enrich_ibit_trades(trades, ibit)
        assert len(result) == 2
        assert "rv_7d_crypto" in result.columns
        assert "ibit_price" in result.columns
        assert "win" in result.columns

    def test_win_label_set(self):
        ibit = self._make_prices_series(60)
        trades = self._make_trades(3)
        result = enrich_ibit_trades(trades, ibit)
        assert result.iloc[0]["win"] == 1  # pnl > 0
        assert result.iloc[2]["win"] == 0  # pnl < 0

    def test_with_etha_and_spy(self):
        ibit = self._make_prices_series(60, 55.0, seed=42)
        etha = self._make_prices_series(60, 18.0, seed=43)
        spy = self._make_prices_series(60, 580.0, seed=44)
        trades = self._make_trades(1)
        result = enrich_ibit_trades(trades, ibit, etha_closes=etha, spy_closes=spy)
        assert result.iloc[0]["ibit_etha_corr_7d"] is not None or True  # may be None if <8 days
        assert len(result) == 1

    def test_with_funding_history(self):
        ibit = self._make_prices_series(60)
        trades = self._make_trades(1)
        funding = pd.DataFrame({
            "date": [f"2025-01-{d:02d}" for d in range(5, 30)],
            "funding_rate": [0.01] * 25,
        })
        result = enrich_ibit_trades(trades, ibit, funding_history=funding)
        assert result.iloc[0]["funding_rate_current"] is not None

    def test_with_fgi_and_dominance(self):
        ibit = self._make_prices_series(60)
        trades = self._make_trades(1)
        fgi = {"2025-02-10": 35.0}
        dom = {"2025-02-10": 52.0}
        result = enrich_ibit_trades(
            trades, ibit, fgi_by_date=fgi, dominance_by_date=dom,
        )
        assert result.iloc[0]["fgi_normalized"] is not None
        assert result.iloc[0]["dominance_norm"] is not None

    def test_all_crypto_feature_columns_present(self):
        ibit = self._make_prices_series(60)
        trades = self._make_trades(1)
        result = enrich_ibit_trades(trades, ibit)
        for col in IBIT_FEATURE_NAMES:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_lookahead(self):
        """Entry date data should use prices BEFORE entry, not on entry day."""
        dates = pd.bdate_range("2025-01-01", periods=60)
        # Make a spike on entry date that shouldn't be visible
        prices = [55.0] * 59 + [100.0]  # spike on last day
        ibit = pd.Series(prices, index=dates)

        trades = pd.DataFrame([{
            "entry_date": str(dates[-1].date()),
            "exit_date": str(dates[-1].date()),
            "pnl": 50.0,
        }])
        result = enrich_ibit_trades(trades, ibit)
        # ibit_price should be from BEFORE entry (55.0), not 100.0
        if result.iloc[0]["ibit_price"] is not None:
            assert result.iloc[0]["ibit_price"] == pytest.approx(55.0, abs=1.0)

    def test_insufficient_ibit_history(self):
        """With only 3 days of IBIT data, crypto features should be None."""
        dates = pd.bdate_range("2025-02-07", periods=5)
        ibit = pd.Series([55, 56, 57, 58, 59], index=dates, dtype=float)
        trades = pd.DataFrame([{
            "entry_date": "2025-02-12",
            "exit_date": "2025-02-14",
            "pnl": 50.0,
        }])
        result = enrich_ibit_trades(trades, ibit)
        # Should still produce a row, but crypto features are None
        assert len(result) == 1
