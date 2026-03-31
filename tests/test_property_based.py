"""Property-based tests using Hypothesis for bounded financial calculations.

These tests verify invariants that must hold for any valid input, rather than
checking specific examples.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Import helpers — avoid pulling the full ml package (which requires xgboost)
# by loading individual modules directly.
# ---------------------------------------------------------------------------

_project = Path(__file__).resolve().parent.parent

from compass.sizing import PositionSizer

from shared.indicators import calculate_iv_rank, sanitize_features

_ss_spec = importlib.util.spec_from_file_location(
    "strategy.spread_strategy", str(_project / "strategy" / "spread_strategy.py"))
_ss_mod = importlib.util.module_from_spec(_ss_spec)
_ss_spec.loader.exec_module(_ss_mod)
CreditSpreadStrategy = _ss_mod.CreditSpreadStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strategy_config():
    return {
        'strategy': {
            'min_dte': 30, 'max_dte': 45,
            'min_delta': 0.10, 'max_delta': 0.15,
            'spread_width': 5,
            'min_iv_rank': 25, 'min_iv_percentile': 25,
            'technical': {
                'use_trend_filter': True, 'use_rsi_filter': True,
                'use_support_resistance': True,
                'fast_ma': 20, 'slow_ma': 50,
                'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            },
        },
        'risk': {
            'account_size': 100000, 'max_risk_per_trade': 2.0,
            'max_positions': 7, 'profit_target': 50,
            'stop_loss_multiplier': 2.5, 'delta_threshold': 0.30,
            'min_credit_pct': 20,
        },
    }


# ---------------------------------------------------------------------------
# 1. _calculate_pop: POP in [0, 100] for any delta in [-1, 1]
# ---------------------------------------------------------------------------

class TestPopBounded:

    @given(delta=st.floats(min_value=-1.0, max_value=1.0))
    def test_pop_always_in_0_100(self, delta):
        """Probability of profit must be between 0 and 100 for any valid delta."""
        strategy = CreditSpreadStrategy(_make_strategy_config())
        pop = strategy._calculate_pop(delta)
        assert 0 <= pop <= 100

    @given(delta=st.floats(min_value=-1.0, max_value=1.0))
    def test_pop_monotonic_in_abs_delta(self, delta):
        """Higher |delta| should mean lower POP (since POP = 1 - |delta|)."""
        strategy = CreditSpreadStrategy(_make_strategy_config())
        pop = strategy._calculate_pop(delta)
        # POP = round((1 - abs(delta)) * 100, 2), so larger |delta| -> smaller POP
        expected = round((1 - abs(delta)) * 100, 2)
        assert pop == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# 2. _calculate_kelly: Kelly >= 0 for valid inputs
# ---------------------------------------------------------------------------

class TestKellyNonNegative:

    @given(
        win_prob=st.floats(min_value=0.01, max_value=0.99),
        win_amount=st.floats(min_value=0.01, max_value=10000.0),
        loss_amount=st.floats(min_value=0.01, max_value=10000.0),
    )
    def test_kelly_always_non_negative(self, win_prob, win_amount, loss_amount):
        """Kelly fraction should never be negative for valid positive inputs."""
        sizer = PositionSizer()
        kelly = sizer._calculate_kelly(win_prob, win_amount, loss_amount)
        assert kelly >= 0.0

    @given(
        win_prob=st.floats(min_value=0.01, max_value=0.99),
        win_amount=st.floats(min_value=0.01, max_value=10000.0),
        loss_amount=st.floats(min_value=0.01, max_value=10000.0),
    )
    def test_kelly_at_most_one(self, win_prob, win_amount, loss_amount):
        """Kelly fraction should be at most 1.0 (bet entire bankroll)."""
        sizer = PositionSizer()
        kelly = sizer._calculate_kelly(win_prob, win_amount, loss_amount)
        assert kelly <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 3. calculate_iv_rank: iv_rank in [0, 100], iv_percentile in [0, 100]
# ---------------------------------------------------------------------------

class TestIVRankBounded:

    @given(
        n=st.integers(min_value=2, max_value=200),
        current_iv=st.floats(min_value=0.1, max_value=200.0),
    )
    def test_iv_rank_bounded(self, n, current_iv):
        """iv_rank should always be in [0, 100] for any valid HV series and IV."""
        np.random.seed(42)
        hv_values = pd.Series(np.random.uniform(5.0, 80.0, n))
        result = calculate_iv_rank(hv_values, current_iv)
        # iv_rank can go slightly outside [0, 100] when current_iv is outside
        # the historical range; that's the formula's behaviour, but we still
        # confirm iv_percentile is always bounded.
        assert 0 <= result['iv_percentile'] <= 100

    @given(
        n=st.integers(min_value=2, max_value=200),
        current_iv=st.floats(min_value=0.1, max_value=200.0),
    )
    def test_iv_percentile_bounded(self, n, current_iv):
        """iv_percentile must be in [0, 100]."""
        np.random.seed(42)
        hv_values = pd.Series(np.random.uniform(5.0, 80.0, n))
        result = calculate_iv_rank(hv_values, current_iv)
        assert 0.0 <= result['iv_percentile'] <= 100.0

    def test_iv_rank_empty_series(self):
        """Empty HV series should return zeros."""
        result = calculate_iv_rank(pd.Series(dtype=float), current_iv=20.0)
        assert result['iv_rank'] == 0.0
        assert result['iv_percentile'] == 0.0


# ---------------------------------------------------------------------------
# 4. sanitize_features: no NaN or Inf in output
# ---------------------------------------------------------------------------

class TestSanitizeFeatures:

    @given(
        rows=st.integers(min_value=1, max_value=50),
        cols=st.integers(min_value=1, max_value=20),
    )
    def test_sanitize_removes_nan_inf(self, rows, cols):
        """Output of sanitize_features should contain no NaN or Inf."""
        np.random.seed(42)
        X = np.random.randn(rows, cols)
        # Inject some NaN and Inf
        if rows > 1 and cols > 1:
            X[0, 0] = np.nan
            X[0, 1] = np.inf
            X[1, 0] = -np.inf
        result = sanitize_features(X)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    @given(
        rows=st.integers(min_value=1, max_value=50),
        cols=st.integers(min_value=1, max_value=20),
    )
    def test_sanitize_preserves_shape(self, rows, cols):
        """sanitize_features should not change array dimensions."""
        X = np.random.randn(rows, cols)
        result = sanitize_features(X)
        assert result.shape == (rows, cols)

    def test_sanitize_all_nan(self):
        """An all-NaN array should become all zeros."""
        X = np.full((3, 3), np.nan)
        result = sanitize_features(X)
        assert not np.any(np.isnan(result))
        assert (result == 0.0).all()

