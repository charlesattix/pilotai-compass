"""Tests for shared IV rank / percentile calculation."""
import numpy as np
import pandas as pd
import pytest

from shared.indicators import calculate_iv_rank


class TestCalculateIVRank:
    """Tests for the canonical IV rank implementation."""

    def _make_hv_series(self, low=10.0, high=30.0, n=252):
        """Helper: create a linearly spaced HV series."""
        return pd.Series(np.linspace(low, high, n))

    def test_iv_rank_at_high(self):
        """When current IV equals the max, IV rank should be 100."""
        hv = self._make_hv_series(low=10, high=30)
        result = calculate_iv_rank(hv, current_iv=30.0)
        assert result['iv_rank'] == pytest.approx(100.0, abs=0.01)

    def test_iv_rank_at_low(self):
        """When current IV equals the min, IV rank should be 0."""
        hv = self._make_hv_series(low=10, high=30)
        result = calculate_iv_rank(hv, current_iv=10.0)
        assert result['iv_rank'] == pytest.approx(0.0, abs=0.01)

    def test_iv_rank_at_midpoint(self):
        """When current IV is exactly the midpoint of min and max, rank should be 50."""
        hv = self._make_hv_series(low=10, high=30)
        mid = (10.0 + 30.0) / 2.0
        result = calculate_iv_rank(hv, current_iv=mid)
        assert result['iv_rank'] == pytest.approx(50.0, abs=0.01)
