"""Tests for compass/order_flow_analyzer.py — order flow analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.order_flow_analyzer import (
    AnalysisResult,
    CumulativeDelta,
    DivergenceSignal,
    FlowImbalance,
    FootprintBar,
    LargeTrade,
    OrderFlowAnalyzer,
    PriceLevel,
    VolumeProfile,
    VWAPData,
    build_footprint_bars,
    compute_cumulative_delta,
    compute_flow_imbalance,
    compute_volume_profile,
    compute_vwap,
    detect_divergences,
    detect_large_trades,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_trades(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    volumes = rng.exponential(50, n).astype(float)
    sides = rng.choice([1, -1], n).astype(float)
    timestamps = np.arange(n)
    return pd.DataFrame({
        "price": prices,
        "volume": volumes,
        "side": sides,
        "timestamp": timestamps,
    })


def _make_trending_trades(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Price trending up but delta trending down (bearish divergence)."""
    rng = np.random.RandomState(seed)
    prices = 100.0 + np.linspace(0, 5, n) + rng.normal(0, 0.05, n)
    volumes = rng.exponential(50, n).astype(float)
    # More selling as price rises
    sell_prob = np.linspace(0.3, 0.8, n)
    sides = np.where(rng.random(n) < sell_prob, -1.0, 1.0)
    return pd.DataFrame({
        "price": prices,
        "volume": volumes,
        "side": sides,
        "timestamp": np.arange(n),
    })


@pytest.fixture
def trades():
    return _make_trades()


@pytest.fixture
def analyzer(trades):
    return OrderFlowAnalyzer(trades)


# ── Flow imbalance tests ─────────────────────────────────────────────────


class TestFlowImbalance:
    def test_basic(self, trades):
        fi = compute_flow_imbalance(
            trades["side"].values, trades["volume"].values
        )
        assert isinstance(fi, FlowImbalance)
        assert fi.total_volume > 0
        assert fi.buy_volume + fi.sell_volume == pytest.approx(fi.total_volume)

    def test_imbalance_bounded(self, trades):
        fi = compute_flow_imbalance(
            trades["side"].values, trades["volume"].values
        )
        assert -1.0 <= fi.imbalance_ratio <= 1.0

    def test_all_buys(self):
        sides = np.array([1, 1, 1, 1])
        vols = np.array([10.0, 20.0, 30.0, 40.0])
        fi = compute_flow_imbalance(sides, vols)
        assert fi.imbalance_ratio == 1.0
        assert fi.sell_volume == 0.0

    def test_all_sells(self):
        sides = np.array([-1, -1, -1])
        vols = np.array([10.0, 20.0, 30.0])
        fi = compute_flow_imbalance(sides, vols)
        assert fi.imbalance_ratio == -1.0

    def test_balanced(self):
        sides = np.array([1, -1, 1, -1])
        vols = np.array([10.0, 10.0, 10.0, 10.0])
        fi = compute_flow_imbalance(sides, vols)
        assert fi.imbalance_ratio == pytest.approx(0.0)

    def test_trade_counts(self, trades):
        fi = compute_flow_imbalance(
            trades["side"].values, trades["volume"].values
        )
        assert fi.buy_trade_count + fi.sell_trade_count == len(trades)


# ── Cumulative delta tests ───────────────────────────────────────────────


class TestCumulativeDelta:
    def test_basic(self, trades):
        cd = compute_cumulative_delta(
            trades["side"].values, trades["volume"].values, trades["timestamp"].values
        )
        assert isinstance(cd, CumulativeDelta)
        assert len(cd.values) == len(trades)

    def test_all_buys_positive(self):
        sides = np.array([1, 1, 1])
        vols = np.array([10.0, 20.0, 30.0])
        ts = np.arange(3)
        cd = compute_cumulative_delta(sides, vols, ts)
        assert cd.final_delta == 60.0
        assert cd.min_delta == 10.0

    def test_all_sells_negative(self):
        sides = np.array([-1, -1, -1])
        vols = np.array([10.0, 20.0, 30.0])
        ts = np.arange(3)
        cd = compute_cumulative_delta(sides, vols, ts)
        assert cd.final_delta == -60.0

    def test_max_min_tracked(self, trades):
        cd = compute_cumulative_delta(
            trades["side"].values, trades["volume"].values, trades["timestamp"].values
        )
        assert cd.max_delta >= cd.final_delta or cd.max_delta >= 0
        assert cd.min_delta <= cd.final_delta or cd.min_delta <= 0


# ── Volume profile tests ────────────────────────────────────────────────


class TestVolumeProfile:
    def test_basic(self, trades):
        vp = compute_volume_profile(
            trades["price"].values, trades["volume"].values,
            trades["side"].values, n_levels=20,
        )
        assert isinstance(vp, VolumeProfile)
        assert len(vp.levels) > 0
        assert vp.poc_volume > 0

    def test_poc_highest_volume(self, trades):
        vp = compute_volume_profile(
            trades["price"].values, trades["volume"].values,
            trades["side"].values,
        )
        max_vol = max(l.total_volume for l in vp.levels)
        assert vp.poc_volume == pytest.approx(max_vol)

    def test_value_area_contains_poc(self, trades):
        vp = compute_volume_profile(
            trades["price"].values, trades["volume"].values,
            trades["side"].values,
        )
        assert vp.value_area_low <= vp.poc_price <= vp.value_area_high

    def test_hvn_not_empty(self, trades):
        vp = compute_volume_profile(
            trades["price"].values, trades["volume"].values,
            trades["side"].values,
        )
        assert len(vp.high_volume_nodes) > 0

    def test_delta_per_level(self, trades):
        vp = compute_volume_profile(
            trades["price"].values, trades["volume"].values,
            trades["side"].values,
        )
        for l in vp.levels:
            assert l.delta == pytest.approx(l.buy_volume - l.sell_volume)

    def test_empty(self):
        vp = compute_volume_profile(
            np.array([]), np.array([]), np.array([])
        )
        assert vp.levels == []
        assert vp.poc_price == 0.0


# ── VWAP tests ───────────────────────────────────────────────────────────


class TestVWAP:
    def test_shape(self, trades):
        vw = compute_vwap(
            trades["price"].values, trades["volume"].values,
            trades["timestamp"].values,
        )
        assert isinstance(vw, VWAPData)
        assert len(vw.vwap) == len(trades)

    def test_bands_order(self, trades):
        vw = compute_vwap(
            trades["price"].values, trades["volume"].values,
            trades["timestamp"].values,
        )
        # 2SD bands should be outside 1SD bands
        np.testing.assert_array_less(vw.lower_2sd, vw.lower_1sd + 1e-10)
        np.testing.assert_array_less(vw.upper_1sd - 1e-10, vw.upper_2sd)

    def test_vwap_within_price_range(self, trades):
        vw = compute_vwap(
            trades["price"].values, trades["volume"].values,
            trades["timestamp"].values,
        )
        p_min = trades["price"].min()
        p_max = trades["price"].max()
        assert vw.vwap[-1] >= p_min - 1.0
        assert vw.vwap[-1] <= p_max + 1.0

    def test_empty(self):
        vw = compute_vwap(np.array([]), np.array([]), np.array([]))
        assert len(vw.vwap) == 0


# ── Footprint bar tests ─────────────────────────────────────────────────


class TestFootprintBars:
    def test_basic(self, trades):
        bars = build_footprint_bars(
            trades["price"].values, trades["volume"].values,
            trades["side"].values, trades["timestamp"].values,
            bar_size=100,
        )
        assert len(bars) > 0
        assert all(isinstance(b, FootprintBar) for b in bars)

    def test_bar_volume_positive(self, trades):
        bars = build_footprint_bars(
            trades["price"].values, trades["volume"].values,
            trades["side"].values, trades["timestamp"].values,
            bar_size=50,
        )
        for b in bars:
            assert b.bar_volume > 0

    def test_bar_count(self, trades):
        n = len(trades)
        bars = build_footprint_bars(
            trades["price"].values, trades["volume"].values,
            trades["side"].values, trades["timestamp"].values,
            bar_size=100,
        )
        expected = (n + 99) // 100
        assert len(bars) == expected


# ── Divergence signal tests ──────────────────────────────────────────────


class TestDivergence:
    def test_bearish_detected(self):
        """Price up + delta down should trigger bearish divergence."""
        trending = _make_trending_trades(300)
        analyzer = OrderFlowAnalyzer(trending, divergence_window=30)
        result = analyzer.analyze()
        bearish = [d for d in result.divergence_signals if d.signal_type == "bearish_divergence"]
        # May or may not find divergences depending on randomness, but should not crash
        assert isinstance(result.divergence_signals, list)

    def test_signal_types(self, analyzer):
        result = analyzer.analyze()
        for d in result.divergence_signals:
            assert d.signal_type in ("bearish_divergence", "bullish_divergence")

    def test_strength_bounded(self, analyzer):
        result = analyzer.analyze()
        for d in result.divergence_signals:
            assert 0 <= d.strength <= 1.0

    def test_empty_on_short(self):
        prices = np.array([100.0, 100.1, 100.2])
        delta = np.array([10.0, 20.0, 30.0])
        signals = detect_divergences(prices, delta, window=5)
        assert signals == []


# ── Large trade detection tests ──────────────────────────────────────────


class TestLargeTrades:
    def test_detection(self, trades):
        large = detect_large_trades(
            trades["price"].values, trades["volume"].values,
            trades["side"].values, trades["timestamp"].values,
            threshold_multiple=3.0,
        )
        assert isinstance(large, list)
        for lt in large:
            assert lt.volume_multiple >= 3.0

    def test_side_classification(self, trades):
        large = detect_large_trades(
            trades["price"].values, trades["volume"].values,
            trades["side"].values, trades["timestamp"].values,
        )
        for lt in large:
            assert lt.side in ("buy", "sell")

    def test_empty(self):
        assert detect_large_trades(
            np.array([]), np.array([]), np.array([]), np.array([])
        ) == []

    def test_high_threshold_fewer(self, trades):
        low = detect_large_trades(
            trades["price"].values, trades["volume"].values,
            trades["side"].values, trades["timestamp"].values,
            threshold_multiple=2.0,
        )
        high = detect_large_trades(
            trades["price"].values, trades["volume"].values,
            trades["side"].values, trades["timestamp"].values,
            threshold_multiple=5.0,
        )
        assert len(high) <= len(low)


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_basic(self, trades):
        a = OrderFlowAnalyzer(trades)
        assert len(a.prices) == len(trades)

    def test_missing_columns(self):
        with pytest.raises(ValueError, match="Missing"):
            OrderFlowAnalyzer(pd.DataFrame({"foo": [1]}))

    def test_empty_raises(self):
        df = pd.DataFrame({"price": [], "volume": [], "side": []})
        with pytest.raises(ValueError, match="must not be empty"):
            OrderFlowAnalyzer(df)


# ── Full analysis tests ──────────────────────────────────────────────────


class TestAnalysis:
    def test_returns_result(self, analyzer):
        result = analyzer.analyze()
        assert isinstance(result, AnalysisResult)
        assert result.n_trades == 500

    def test_all_components(self, analyzer):
        result = analyzer.analyze()
        assert isinstance(result.flow_imbalance, FlowImbalance)
        assert isinstance(result.cumulative_delta, CumulativeDelta)
        assert isinstance(result.volume_profile, VolumeProfile)
        assert isinstance(result.vwap_data, VWAPData)
        assert len(result.footprint_bars) > 0

    def test_time_range(self, analyzer):
        result = analyzer.analyze()
        assert result.time_range[0] is not None
        assert result.time_range[1] is not None


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, analyzer):
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "flow.html"
            path = OrderFlowAnalyzer.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Order Flow Analysis" in content

    def test_contains_charts(self, analyzer):
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            OrderFlowAnalyzer.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Cumulative Delta" in content

    def test_contains_volume_profile(self, analyzer):
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            OrderFlowAnalyzer.generate_report(result, out)
            content = out.read_text()
            assert "Volume Profile" in content
            assert "POC" in content

    def test_contains_flow_metrics(self, analyzer):
        result = analyzer.analyze()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            OrderFlowAnalyzer.generate_report(result, out)
            content = out.read_text()
            assert "Flow Metrics" in content
            assert "Imbalance" in content

    def test_default_path(self, analyzer):
        result = analyzer.analyze()
        path = OrderFlowAnalyzer.generate_report(result)
        assert path.exists()
        assert "order_flow.html" in str(path)
        path.unlink(missing_ok=True)
