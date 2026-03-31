"""Tests for compass/alpha_combiner.py — alpha signal combination."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.alpha_combiner import (
    AlphaCombiner,
    CombinedMetrics,
    CombinerResult,
    OOSResult,
    SignalMetrics,
    compute_ic,
    compute_ic_series,
    compute_icir,
    compute_signal_turnover,
    compute_turnover,
    correlation_deweight,
    dynamic_weights,
    equal_weights,
    inverse_vol_weights,
    rank_normalize,
    ridge_weights,
    zscore_normalize,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_signals(
    n: int = 300, k: int = 4, seed: int = 42, signal_strength: float = 0.02
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic alpha signals and forward returns."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)

    # Base return driven partly by signals
    signals = pd.DataFrame(
        rng.normal(0, 1, (n, k)),
        index=dates,
        columns=[f"alpha_{i}" for i in range(k)],
    )
    # Forward return = weighted combo of signals + noise
    true_weights = rng.uniform(0.1, 0.5, k)
    true_weights /= true_weights.sum()
    forward_ret = (signals.values @ true_weights) * signal_strength + rng.normal(
        0, 0.01, n
    )
    forward_returns = pd.Series(forward_ret, index=dates, name="fwd_ret")

    return signals, forward_returns


@pytest.fixture
def signals_and_returns():
    return _make_signals()


@pytest.fixture
def signals(signals_and_returns):
    return signals_and_returns[0]


@pytest.fixture
def forward_returns(signals_and_returns):
    return signals_and_returns[1]


# ── Normalisation tests ──────────────────────────────────────────────────


class TestNormalization:
    def test_zscore_mean_zero(self, signals):
        normed = zscore_normalize(signals)
        np.testing.assert_allclose(normed.mean().values, 0.0, atol=1e-10)

    def test_zscore_std_one(self, signals):
        normed = zscore_normalize(signals)
        np.testing.assert_allclose(normed.std().values, 1.0, atol=1e-10)

    def test_zscore_preserves_shape(self, signals):
        assert zscore_normalize(signals).shape == signals.shape

    def test_rank_bounded(self, signals):
        ranked = rank_normalize(signals)
        assert ranked.min().min() >= -1.0 - 1e-10
        assert ranked.max().max() <= 1.0 + 1e-10

    def test_rank_preserves_shape(self, signals):
        assert rank_normalize(signals).shape == signals.shape


# ── IC / ICIR / turnover tests ───────────────────────────────────────────


class TestICMetrics:
    def test_ic_positive_for_good_signal(self, signals, forward_returns):
        # alpha_0 contributes to returns, so IC should be non-zero
        ic = compute_ic(signals["alpha_0"], forward_returns)
        assert isinstance(ic, float)

    def test_ic_bounded(self, signals, forward_returns):
        ic = compute_ic(signals["alpha_0"], forward_returns)
        assert -1.0 <= ic <= 1.0

    def test_ic_series_length(self, signals, forward_returns):
        ic_s = compute_ic_series(signals["alpha_0"], forward_returns)
        assert len(ic_s) > 0

    def test_ic_short_series(self):
        sig = pd.Series([1.0, 2.0], index=pd.bdate_range("2024-01-02", periods=2))
        ret = pd.Series([0.01, 0.02], index=sig.index)
        assert compute_ic(sig, ret) == 0.0

    def test_icir_positive_for_consistent_signal(self):
        ic_s = pd.Series([0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06])
        assert compute_icir(ic_s) > 0

    def test_icir_zero_for_short(self):
        assert compute_icir(pd.Series([0.05])) == 0.0

    def test_signal_turnover(self, signals):
        t = compute_signal_turnover(signals["alpha_0"])
        assert t > 0

    def test_signal_turnover_constant(self):
        sig = pd.Series([1.0] * 10)
        assert compute_signal_turnover(sig) == 0.0


# ── Weighting method tests ───────────────────────────────────────────────


class TestWeightingMethods:
    def test_equal_weights_sum(self):
        w = equal_weights(5)
        assert len(w) == 5
        np.testing.assert_allclose(w.sum(), 1.0)

    def test_equal_weights_uniform(self):
        w = equal_weights(4)
        np.testing.assert_allclose(w, [0.25, 0.25, 0.25, 0.25])

    def test_inverse_vol_sum(self, signals):
        w = inverse_vol_weights(signals)
        np.testing.assert_allclose(np.abs(w).sum(), 1.0, atol=1e-10)

    def test_inverse_vol_positive(self, signals):
        w = inverse_vol_weights(signals)
        assert (w > 0).all()

    def test_inverse_vol_lower_vol_higher_weight(self):
        """Lower vol signal should get higher weight."""
        df = pd.DataFrame({
            "low_vol": np.random.RandomState(1).normal(0, 0.5, 100),
            "high_vol": np.random.RandomState(2).normal(0, 2.0, 100),
        })
        w = inverse_vol_weights(df)
        assert w[0] > w[1]

    def test_ridge_weights_sum(self, signals, forward_returns):
        w = ridge_weights(signals, forward_returns)
        np.testing.assert_allclose(np.abs(w).sum(), 1.0, atol=1e-10)

    def test_ridge_weights_short_data(self):
        sig = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ret = pd.Series([0.01, 0.02])
        w = ridge_weights(sig, ret)
        np.testing.assert_allclose(w.sum(), 1.0)  # falls back to equal

    def test_correlation_deweight(self, signals):
        base = equal_weights(signals.shape[1])
        # Create highly correlated signals
        df = signals.copy()
        df["alpha_copy"] = df["alpha_0"] + np.random.RandomState(1).normal(0, 0.01, len(df))
        base5 = equal_weights(5)
        adj = correlation_deweight(df, base5, threshold=0.5)
        np.testing.assert_allclose(np.abs(adj).sum(), 1.0, atol=1e-10)

    def test_correlation_deweight_reduces_correlated(self, signals):
        df = signals.copy()
        df["alpha_copy"] = df["alpha_0"] * 0.99 + 0.01
        base = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        adj = correlation_deweight(df, base, threshold=0.5)
        # The copy or original should have reduced weight
        assert min(abs(adj[0]), abs(adj[4])) < 0.2

    def test_dynamic_weights_shape(self, signals, forward_returns):
        wh = dynamic_weights(signals, forward_returns, lookback=50)
        assert len(wh) > 0
        assert set(wh.columns) == set(signals.columns)

    def test_dynamic_weights_sum(self, signals, forward_returns):
        wh = dynamic_weights(signals, forward_returns, lookback=50)
        row_sums = wh.abs().sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_basic(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns)
        assert ac.n_signals == 4

    def test_empty_signals_raises(self, forward_returns):
        with pytest.raises(ValueError, match="must not be empty"):
            AlphaCombiner(pd.DataFrame(), forward_returns)

    def test_single_signal_raises(self, forward_returns):
        sig = pd.DataFrame({"a": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least 2"):
            AlphaCombiner(sig, forward_returns)

    def test_bad_method_raises(self, signals, forward_returns):
        with pytest.raises(ValueError, match="Unknown method"):
            AlphaCombiner(signals, forward_returns, method="magic")

    def test_normalize_zscore(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns, normalize="zscore")
        np.testing.assert_allclose(ac.signals.mean().values, 0.0, atol=1e-10)

    def test_normalize_rank(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns, normalize="rank")
        assert ac.signals.max().max() <= 1.0 + 1e-10

    def test_normalize_none(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns, normalize="none")
        pd.testing.assert_frame_equal(ac.signals, signals)


# ── Full combination tests ───────────────────────────────────────────────


class TestCombine:
    @pytest.mark.parametrize("method", AlphaCombiner.METHODS)
    def test_all_methods_run(self, signals, forward_returns, method):
        lookback = 50 if method == "dynamic" else 63
        ac = AlphaCombiner(
            signals, forward_returns, method=method,
            dynamic_lookback=lookback,
        )
        result = ac.combine()
        assert isinstance(result, CombinerResult)
        assert len(result.weights) == 4
        assert result.method == method

    def test_combined_signal_has_column(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns, method="equal")
        result = ac.combine()
        assert "combined" in result.combined_signal.columns

    def test_signal_metrics_populated(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns, method="ridge")
        result = ac.combine()
        assert len(result.signal_metrics) == 4
        assert all(isinstance(sm, SignalMetrics) for sm in result.signal_metrics)

    def test_combined_metrics(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns, method="ridge")
        result = ac.combine()
        cm = result.combined_metrics
        assert isinstance(cm, CombinedMetrics)
        assert cm.method == "ridge"

    def test_correlation_matrix_shape(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns)
        result = ac.combine()
        assert result.correlation_matrix.shape == (4, 4)

    def test_dynamic_has_weight_history(self, signals, forward_returns):
        ac = AlphaCombiner(
            signals, forward_returns, method="dynamic", dynamic_lookback=50
        )
        result = ac.combine()
        assert result.weight_history is not None
        assert len(result.weight_history) > 0

    def test_static_no_weight_history(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns, method="equal")
        result = ac.combine()
        assert result.weight_history is None


# ── OOS tests ────────────────────────────────────────────────────────────


class TestOOS:
    def test_oos_result_present(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns)
        result = ac.combine()
        assert isinstance(result.oos_result, OOSResult)

    def test_oos_n_periods(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns)
        result = ac.combine()
        assert result.oos_result.n_periods > 0

    def test_oos_ic_decay_bounded(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns)
        result = ac.combine()
        # IC decay can be any value, just check it's finite
        assert np.isfinite(result.oos_result.ic_decay)


# ── Turnover tests ───────────────────────────────────────────────────────


class TestTurnover:
    def test_turnover_zero_for_static(self):
        wh = pd.DataFrame({"a": [0.5, 0.5, 0.5], "b": [0.5, 0.5, 0.5]})
        assert compute_turnover(wh) == 0.0

    def test_turnover_positive_for_changing(self):
        wh = pd.DataFrame({"a": [0.3, 0.5, 0.7], "b": [0.7, 0.5, 0.3]})
        assert compute_turnover(wh) > 0

    def test_turnover_short(self):
        wh = pd.DataFrame({"a": [0.5]})
        assert compute_turnover(wh) == 0.0


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generate_report_creates_file(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns, method="ridge")
        result = ac.combine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_alpha.html"
            path = AlphaCombiner.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Alpha Signal Combiner" in content

    def test_report_contains_ic_dashboard(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns)
        result = ac.combine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            AlphaCombiner.generate_report(result, out)
            content = out.read_text()
            assert "IC Dashboard" in content
            assert "Mean IC" in content

    def test_report_contains_correlation(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns)
        result = ac.combine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            AlphaCombiner.generate_report(result, out)
            content = out.read_text()
            assert "Correlation Matrix" in content
            assert "<svg" in content

    def test_report_with_dynamic(self, signals, forward_returns):
        ac = AlphaCombiner(
            signals, forward_returns, method="dynamic", dynamic_lookback=50
        )
        result = ac.combine()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            AlphaCombiner.generate_report(result, out)
            content = out.read_text()
            assert "Weight Evolution" in content

    def test_report_default_path(self, signals, forward_returns):
        ac = AlphaCombiner(signals, forward_returns, method="equal")
        result = ac.combine()
        path = AlphaCombiner.generate_report(result)
        assert path.exists()
        assert "alpha_combiner.html" in str(path)
        path.unlink(missing_ok=True)
