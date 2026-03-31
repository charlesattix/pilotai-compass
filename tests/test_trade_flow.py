"""Tests for compass/trade_flow.py — institutional trade flow analyzer."""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from compass.trade_flow import (
    AccumulationSignal, BlockTrade, FlowClassification, FlowImbalance,
    FlowMomentum, FlowSnapshot, TradeFlowAnalyzer, VPINResult,
    classify_smart_money, classify_trades, compute_vpin,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_trades(n=2000, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-06-03 09:30", periods=n, freq="1s")
    base = 430.0 + rng.normal(0, 0.05, n).cumsum()
    volumes = rng.randint(10, 500, n).astype(float)
    # Inject some block trades
    for i in rng.choice(n, 10, replace=False):
        volumes[i] = rng.randint(3000, 8000)
    return pd.DataFrame({"price": base, "volume": volumes}, index=dates)

def _make_analyzer(n=2000, seed=42, **kwargs):
    return TradeFlowAnalyzer(_make_trades(n, seed), **kwargs)

# ── Pure function tests ──────────────────────────────────────────────────

class TestClassifyTrades:
    def test_returns_three_arrays(self):
        prices = np.array([100, 101, 100.5, 102, 101])
        volumes = np.array([100, 200, 150, 300, 100], dtype=float)
        bv, sv, ib = classify_trades(prices, volumes)
        assert len(bv) == 5 and len(sv) == 5 and len(ib) == 5

    def test_uptick_classified_as_buy(self):
        prices = np.array([100, 101, 102], dtype=float)
        volumes = np.array([100, 200, 300], dtype=float)
        bv, sv, _ = classify_trades(prices, volumes)
        assert bv[1] == 200 and sv[1] == 0

    def test_downtick_classified_as_sell(self):
        prices = np.array([102, 101, 100], dtype=float)
        volumes = np.array([100, 200, 300], dtype=float)
        bv, sv, _ = classify_trades(prices, volumes)
        assert sv[1] == 200 and bv[1] == 0

    def test_block_detection(self):
        prices = np.array([100] * 10, dtype=float)
        volumes = np.array([100] * 9 + [2000], dtype=float)
        _, _, ib = classify_trades(prices, volumes, block_threshold=5.0)
        assert ib[-1] is True or ib[-1] == True

    def test_with_mid_prices(self):
        prices = np.array([100.1, 99.9, 100.05], dtype=float)
        volumes = np.array([100, 200, 150], dtype=float)
        mids = np.array([100.0, 100.0, 100.0], dtype=float)
        bv, sv, _ = classify_trades(prices, volumes, mid_prices=mids)
        assert bv[0] == 100  # above mid = buy
        assert sv[1] == 200  # below mid = sell

class TestClassifySmartMoney:
    def test_blocks_are_smart(self):
        volumes = np.array([100] * 9 + [5000], dtype=float)
        is_block = np.array([False] * 9 + [True])
        prices = np.arange(10, dtype=float) + 100
        is_smart = classify_smart_money(volumes, is_block, prices)
        assert is_smart[-1] is True or is_smart[-1] == True

    def test_returns_bool_array(self):
        n = 50
        volumes = np.ones(n) * 100
        is_block = np.zeros(n, dtype=bool)
        prices = np.arange(n, dtype=float) + 100
        result = classify_smart_money(volumes, is_block, prices)
        assert result.dtype == bool

class TestVPIN:
    def test_returns_tuple(self):
        bv = np.random.rand(200) * 100
        sv = np.random.rand(200) * 100
        vpin, series = compute_vpin(bv, sv, bucket_size=20)
        assert isinstance(vpin, float)
        assert len(series) > 0

    def test_vpin_range(self):
        bv = np.random.rand(200) * 100
        sv = np.random.rand(200) * 100
        vpin, _ = compute_vpin(bv, sv, 20)
        assert 0 <= vpin <= 1

    def test_short_data(self):
        vpin, _ = compute_vpin(np.array([1.0]), np.array([1.0]), 50)
        assert vpin == 0.0

    def test_all_buys_high_vpin(self):
        bv = np.ones(100) * 100
        sv = np.zeros(100)
        vpin, _ = compute_vpin(bv, sv, 20)
        assert vpin > 0.8

# ── Block detection ──────────────────────────────────────────────────────

class TestBlockDetection:
    def test_detects_blocks(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert len(analyzer.block_trades) > 0

    def test_block_fields(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for b in analyzer.block_trades:
            assert b.volume > 0
            assert b.side in ("buy", "sell")
            assert b.size_multiple > 1

    def test_sorted_by_size(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        mults = [b.size_multiple for b in analyzer.block_trades]
        assert mults == sorted(mults, reverse=True)

# ── Flow classification ──────────────────────────────────────────────────

class TestFlowClassification:
    def test_buckets_populated(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert len(analyzer.flow_classification) > 0

    def test_smart_ratio_range(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for f in analyzer.flow_classification:
            assert 0 <= f.smart_ratio <= 1

    def test_volumes_non_negative(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        for f in analyzer.flow_classification:
            assert f.smart_volume >= 0
            assert f.retail_volume >= 0

# ── VPIN analysis ────────────────────────────────────────────────────────

class TestVPINAnalysis:
    def test_vpin_computed(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.vpin_result is not None

    def test_toxicity_level_valid(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.vpin_result.toxicity_level in ("low", "medium", "high", "extreme")

    def test_percentile_range(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert 0 <= analyzer.vpin_result.percentile <= 1

# ── Accumulation ─────────────────────────────────────────────────────────

class TestAccumulation:
    def test_phase_valid(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.accumulation.phase in ("accumulation", "distribution", "neutral")

    def test_strength_range(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert 0 <= analyzer.accumulation.strength <= 1

    def test_price_trend_valid(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.accumulation.price_trend in ("up", "down", "flat")

    def test_divergence_is_bool(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert isinstance(analyzer.accumulation.divergence, bool)

# ── Flow imbalance ───────────────────────────────────────────────────────

class TestFlowImbalance:
    def test_imbalance_range(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert -1 <= analyzer.imbalance.imbalance <= 1

    def test_signal_valid(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.imbalance.signal in ("strong_buy", "buy", "neutral", "sell", "strong_sell")

    def test_volumes_positive(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.imbalance.buy_volume >= 0
        assert analyzer.imbalance.sell_volume >= 0

# ── Flow momentum ────────────────────────────────────────────────────────

class TestFlowMomentum:
    def test_signal_valid(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.momentum.signal in (
            "momentum_buy", "momentum_sell",
            "reversal_buy", "reversal_sell", "neutral",
        )

    def test_lookback_positive(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert analyzer.momentum.lookback > 0

# ── Snapshots ────────────────────────────────────────────────────────────

class TestSnapshots:
    def test_snapshots_populated(self):
        analyzer = _make_analyzer()
        analyzer.analyze()
        assert len(analyzer.snapshots) > 0

    def test_cumulative_monotonic_consistency(self):
        """Cumulative flow should match sum of net flows."""
        analyzer = _make_analyzer()
        analyzer.analyze()
        running = 0.0
        for s in analyzer.snapshots:
            running += s.net_flow
            assert s.cumulative_flow == pytest.approx(running, abs=0.01)

# ── Pipeline ─────────────────────────────────────────────────────────────

class TestPipeline:
    def test_analyze_keys(self):
        analyzer = _make_analyzer()
        result = analyzer.analyze()
        expected = {"block_trades", "flow_classification", "vpin",
                    "accumulation", "imbalance", "momentum", "snapshots"}
        assert set(result.keys()) == expected

    def test_from_csv(self, tmp_path):
        df = _make_trades()
        csv = tmp_path / "trades.csv"
        df.to_csv(csv)
        analyzer = TradeFlowAnalyzer.from_csv(str(csv))
        analyzer.analyze()
        assert analyzer.vpin_result is not None

    def test_get_summary(self):
        analyzer = _make_analyzer()
        summary = analyzer.get_summary()
        assert "vpin" in summary
        assert "toxicity" in summary
        assert "imbalance_signal" in summary
        assert summary["n_blocks"] > 0

    def test_missing_price_col_raises(self):
        df = pd.DataFrame({"vol": [100, 200]})
        with pytest.raises(ValueError, match="Price column"):
            TradeFlowAnalyzer(df)

# ── Report ───────────────────────────────────────────────────────────────

class TestReport:
    def test_generates_html(self, tmp_path):
        analyzer = _make_analyzer()
        path = analyzer.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Trade Flow" in c

    def test_report_sections(self, tmp_path):
        analyzer = _make_analyzer()
        path = analyzer.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Cumulative" in c and "VPIN" in c and "Smart Money" in c and "Block" in c

    def test_report_charts(self, tmp_path):
        analyzer = _make_analyzer()
        path = analyzer.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()

    def test_report_auto_analyzes(self, tmp_path):
        analyzer = _make_analyzer()
        assert analyzer.vpin_result is None
        analyzer.generate_report(str(tmp_path / "r.html"))
        assert analyzer.vpin_result is not None

    def test_report_default_path(self):
        analyzer = _make_analyzer()
        path = analyzer.generate_report()
        assert "trade_flow.html" in path
