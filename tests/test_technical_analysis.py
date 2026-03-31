"""Tests for shared RSI calculation."""
import pandas as pd

from shared.indicators import calculate_rsi


class TestCalculateRSI:
    """Tests for the canonical RSI implementation."""

    def test_rsi_returns_series(self, sample_price_data):
        """calculate_rsi should return a pandas Series of the same length."""
        rsi = calculate_rsi(sample_price_data['Close'], period=14)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_price_data)

    def test_rsi_boundaries(self, sample_price_data):
        """RSI values (where not NaN) must always be between 0 and 100."""
        rsi = calculate_rsi(sample_price_data['Close'], period=14)
        valid = rsi.dropna()
        assert (valid >= 0).all(), "RSI dropped below 0"
        assert (valid <= 100).all(), "RSI exceeded 100"

    def test_rsi_all_gains_near_100(self):
        """When prices only go up, RSI should approach 100."""
        prices = pd.Series(range(1, 102), dtype=float)  # 1, 2, ..., 101
        rsi = calculate_rsi(prices, period=14)
        # After the warm-up window the RSI should be very high
        assert rsi.iloc[-1] > 95, f"Expected RSI near 100, got {rsi.iloc[-1]}"

    def test_rsi_all_losses_near_0(self):
        """When prices only go down, RSI should approach 0."""
        prices = pd.Series(range(200, 99, -1), dtype=float)  # 200, 199, ..., 100
        rsi = calculate_rsi(prices, period=14)
        assert rsi.iloc[-1] < 5, f"Expected RSI near 0, got {rsi.iloc[-1]}"
