"""
Unit tests for COMPASS ML training data collection utilities.

Target module: compass/collect_training_data.py
Pre-move source: ml/collect_training_data.py (shim)

Tests cover:
  - _compute_vix_percentile() — percentile rank over trailing window
  - _compute_ma() — moving average at a specific date
  - _compute_ma_slope() — annualized slope of MA
  - _compute_returns_vol() — realized volatility
  - _classify_strategy_type() — strategy name mapping
  - _prev_val() — dictionary lookup for most recent prior key
  - DEDUP_KEYS constant — expected dedup columns
  - merge_datasets() — dedup logic with CSV I/O

Blueprint spec: 8+ tests, all green (Phase 3 exit criteria).
"""

import math

import numpy as np
import pandas as pd
import pytest

from compass.collect_training_data import (
    _classify_strategy_type,
    _compute_ma,
    _compute_ma_slope,
    _compute_returns_vol,
    _compute_vix_percentile,
    _prev_val,
    DEDUP_KEYS,
    merge_datasets,
)

from tests.compass_helpers import mock_spy_prices, mock_vix_series


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def spy_100d():
    """100-day SPY close series with mild uptrend."""
    return mock_spy_prices(days=100, trend=15.0, base=450.0)


@pytest.fixture
def vix_100d():
    """100-day constant VIX=20 series."""
    return mock_vix_series(days=100, base_level=20.0)


# ══════════════════════════════════════════════════════════════════════════════
# A. _compute_vix_percentile
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeVixPercentile:

    def test_constant_vix_returns_zero_percentile(self, vix_100d):
        """All values equal → percentile = 0.0 (none are strictly below current)."""
        date_ts = vix_100d.index[50]
        result = _compute_vix_percentile(vix_100d, date_ts, window=20)
        assert result == 0.0

    def test_highest_vix_returns_near_100(self):
        """Current VIX is highest in window → percentile near 100."""
        levels = list(range(10, 60))  # 10, 11, ... 59 (50 values, ascending)
        vix = mock_vix_series(days=50, levels=[float(x) for x in levels])
        last_date = vix.index[-1]
        result = _compute_vix_percentile(vix, last_date, window=50)
        assert result > 90.0

    def test_missing_date_returns_50(self, vix_100d):
        """Date not in VIX index → default 50.0."""
        missing_ts = pd.Timestamp("1999-01-01")
        result = _compute_vix_percentile(vix_100d, missing_ts, window=20)
        assert result == 50.0

    def test_none_series_returns_50(self):
        """None VIX series → default 50.0."""
        result = _compute_vix_percentile(None, pd.Timestamp("2024-01-02"), window=20)
        assert result == 50.0

    def test_insufficient_history_returns_50(self):
        """Fewer than 10 data points in window → default 50.0."""
        vix = mock_vix_series(days=5, base_level=20.0)
        result = _compute_vix_percentile(vix, vix.index[-1], window=5)
        assert result == 50.0


# ══════════════════════════════════════════════════════════════════════════════
# B. _compute_ma
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeMA:

    def test_ma_20_is_close_to_price(self, spy_100d):
        """20-day MA on a smooth series is close to the latest price."""
        date_ts = spy_100d.index[-1]
        ma_20 = _compute_ma(spy_100d, date_ts, period=20)
        assert ma_20 is not None
        # MA should be within a few percent of last price
        assert abs(ma_20 - float(spy_100d.iloc[-1])) / float(spy_100d.iloc[-1]) < 0.05

    def test_insufficient_data_returns_none(self, spy_100d):
        """Fewer data points than period → None."""
        date_ts = spy_100d.index[5]  # only 6 data points available
        result = _compute_ma(spy_100d, date_ts, period=20)
        assert result is None

    def test_ma_200_needs_200_days(self):
        """MA200 returns None when fewer than 200 days available."""
        prices = mock_spy_prices(days=100, trend=10.0, base=450.0)
        result = _compute_ma(prices, prices.index[-1], period=200)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# C. _compute_ma_slope
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeMASlope:

    def test_uptrend_positive_slope(self, spy_100d):
        """MA slope on an uptrending series is positive."""
        date_ts = spy_100d.index[-1]
        slope = _compute_ma_slope(spy_100d, date_ts, ma_period=20, lookback=20)
        assert slope is not None
        assert slope > 0

    def test_flat_series_near_zero_slope(self):
        """Flat price series → slope near zero."""
        flat = mock_spy_prices(days=100, trend=0.0, base=450.0)
        date_ts = flat.index[-1]
        slope = _compute_ma_slope(flat, date_ts, ma_period=20, lookback=20)
        assert slope is not None
        assert abs(slope) < 5.0  # within trend_threshold

    def test_insufficient_data_returns_none(self):
        """Not enough data for MA + lookback → None."""
        short = mock_spy_prices(days=15, trend=20.0, base=450.0)
        result = _compute_ma_slope(short, short.index[-1], ma_period=20, lookback=20)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# D. _compute_returns_vol
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeReturnsVol:

    def test_returns_annualized_vol(self, spy_100d):
        """Returns a non-negative annualized vol percentage."""
        date_ts = spy_100d.index[-1]
        rv = _compute_returns_vol(spy_100d, date_ts, window=20)
        assert rv is not None
        assert rv >= 0

    def test_flat_series_near_zero_vol(self):
        """Perfectly flat series → near-zero realized vol."""
        flat = mock_spy_prices(days=100, trend=0.0, base=450.0)
        rv = _compute_returns_vol(flat, flat.index[-1], window=20)
        assert rv is not None
        assert rv < 5.0  # very low vol

    def test_insufficient_data_returns_none(self):
        """Fewer than window+1 data points → None."""
        short = mock_spy_prices(days=10, trend=10.0, base=450.0)
        result = _compute_returns_vol(short, short.index[-1], window=20)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# E. _classify_strategy_type
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyStrategyType:

    def test_credit_spread_is_CS(self):
        assert _classify_strategy_type("credit_spread") == "CS"

    def test_iron_condor_is_IC(self):
        assert _classify_strategy_type("iron_condor") == "IC"

    def test_straddle_strangle_is_SS(self):
        assert _classify_strategy_type("straddle_strangle") == "SS"

    def test_unknown_defaults_to_CS(self):
        assert _classify_strategy_type("some_new_strategy") == "CS"

    def test_case_insensitive(self):
        assert _classify_strategy_type("Iron_Condor") == "IC"
        assert _classify_strategy_type("STRADDLE_STRANGLE") == "SS"


# ══════════════════════════════════════════════════════════════════════════════
# F. _prev_val
# ══════════════════════════════════════════════════════════════════════════════

class TestPrevVal:

    def test_returns_most_recent_prior(self):
        """Returns the value for the largest key strictly before the given timestamp."""
        d = {
            pd.Timestamp("2024-01-01"): 10,
            pd.Timestamp("2024-01-05"): 20,
            pd.Timestamp("2024-01-10"): 30,
        }
        result = _prev_val(d, pd.Timestamp("2024-01-10"), default=None)
        assert result == 20

    def test_no_prior_keys_returns_default(self):
        """When no keys are before the given timestamp, returns default."""
        d = {pd.Timestamp("2024-01-10"): 100}
        result = _prev_val(d, pd.Timestamp("2024-01-05"), default=-1)
        assert result == -1


# ══════════════════════════════════════════════════════════════════════════════
# G. DEDUP_KEYS constant
# ══════════════════════════════════════════════════════════════════════════════

class TestDedupKeys:

    def test_dedup_keys_contains_expected_columns(self):
        """DEDUP_KEYS has the 4 expected dedup columns."""
        assert "entry_date" in DEDUP_KEYS
        assert "exit_date" in DEDUP_KEYS
        assert "strategy_type" in DEDUP_KEYS
        assert "spread_type" in DEDUP_KEYS
        assert len(DEDUP_KEYS) == 4


# ══════════════════════════════════════════════════════════════════════════════
# H. merge_datasets — dedup logic
# ══════════════════════════════════════════════════════════════════════════════

class TestMergeDatasets:

    @staticmethod
    def _make_trade_row(entry_date, exit_date, strategy_type, spread_type,
                        pnl, win, return_pct, year, regime):
        """Build a trade dict with all columns needed by generate_feature_analysis."""
        return {
            "entry_date": entry_date, "exit_date": exit_date,
            "strategy_type": strategy_type, "spread_type": spread_type,
            "pnl": pnl, "win": win, "return_pct": return_pct,
            "year": year, "regime": regime,
            # Columns needed by generate_feature_analysis
            "dte_at_entry": 30, "hold_days": 8, "vix": 20.0,
            "iv_rank": 40.0, "spy_price": 450.0,
            "dist_from_ma20_pct": 0.5, "dist_from_ma80_pct": 1.0,
            "dist_from_ma200_pct": 2.0, "realized_vol_20d": 15.0,
            "rsi_14": 55.0, "momentum_10d_pct": 1.5,
            "net_credit": 0.65, "spread_width": 5.0, "otm_pct": 3.0,
            "vix_percentile_50d": 50.0, "day_of_week": 2,
        }

    def test_dedup_prefers_exp401_over_exp400(self, tmp_path, monkeypatch):
        """When duplicate rows exist, exp401 wins (keep='last')."""
        # Create mock exp400 CSV
        exp400_data = pd.DataFrame([
            self._make_trade_row("2024-01-02", "2024-01-10", "CS", "bull_put",
                                 100.0, 1, 5.0, 2024, "bull"),
            self._make_trade_row("2024-01-05", "2024-01-15", "CS", "bull_put",
                                 200.0, 1, 10.0, 2024, "bull"),
        ])
        exp401_data = pd.DataFrame([
            self._make_trade_row("2024-01-02", "2024-01-10", "CS", "bull_put",
                                 150.0, 1, 7.5, 2024, "bull"),
            self._make_trade_row("2024-01-20", "2024-01-25", "SS", "long_straddle",
                                 300.0, 1, 15.0, 2024, "bull"),
        ])

        exp400_path = tmp_path / "training_data.csv"
        exp401_path = tmp_path / "training_data_exp401.csv"
        combined_path = tmp_path / "training_data_combined.csv"

        exp400_data.to_csv(exp400_path, index=False)
        exp401_data.to_csv(exp401_path, index=False)

        # Monkeypatch the module-level paths
        import compass.collect_training_data as ctd
        monkeypatch.setattr(ctd, "EXP400_PATH", exp400_path)
        monkeypatch.setattr(ctd, "EXP401_PATH", exp401_path)
        monkeypatch.setattr(ctd, "COMBINED_PATH", combined_path)
        monkeypatch.setattr(ctd, "COMPASS_DIR", tmp_path)

        result = merge_datasets()

        # Should have 3 rows: shared row deduped (exp401 wins), plus 2 unique rows
        assert len(result) == 3
        # The shared row should have exp401's pnl (150.0, not 100.0)
        shared = result[result["entry_date"] == "2024-01-02"]
        assert len(shared) == 1
        assert shared.iloc[0]["pnl"] == 150.0

    def test_single_dataset_no_dedup(self, tmp_path, monkeypatch):
        """Only one dataset present → no dedup needed, all rows preserved."""
        exp401_data = pd.DataFrame([
            self._make_trade_row("2024-01-02", "2024-01-10", "CS", "bull_put",
                                 100.0, 1, 5.0, 2024, "bull"),
            self._make_trade_row("2024-01-05", "2024-01-15", "SS", "long_straddle",
                                 200.0, 1, 10.0, 2024, "bull"),
        ])

        exp401_path = tmp_path / "training_data_exp401.csv"
        combined_path = tmp_path / "training_data_combined.csv"
        exp401_data.to_csv(exp401_path, index=False)

        import compass.collect_training_data as ctd
        monkeypatch.setattr(ctd, "EXP400_PATH", tmp_path / "nonexistent.csv")
        monkeypatch.setattr(ctd, "EXP401_PATH", exp401_path)
        monkeypatch.setattr(ctd, "COMBINED_PATH", combined_path)
        monkeypatch.setattr(ctd, "COMPASS_DIR", tmp_path)

        result = merge_datasets()
        assert len(result) == 2

    def test_no_datasets_raises(self, tmp_path, monkeypatch):
        """Both datasets missing → FileNotFoundError."""
        import compass.collect_training_data as ctd
        monkeypatch.setattr(ctd, "EXP400_PATH", tmp_path / "nope1.csv")
        monkeypatch.setattr(ctd, "EXP401_PATH", tmp_path / "nope2.csv")

        with pytest.raises(FileNotFoundError):
            merge_datasets()
