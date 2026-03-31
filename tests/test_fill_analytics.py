"""Tests for compass.fill_analytics — 42 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.fill_analytics import (
    FillAnalytics,
    ShortfallDecomposition,
    TimingAnalysis,
    VenueStats,
    BenchmarkComparison,
    SlippageBucket,
    ExecutionScorecard,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fills_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "decision_price": [100.0] * n,
        "arrival_price": 100.0 + rng.normal(0, 0.02, n),
        "fill_price": 100.0 + rng.normal(0.03, 0.04, n),
        "end_price": 100.0 + rng.normal(0.01, 0.03, n),
        "fill_qty": rng.integers(10, 100, n).astype(float),
        "ordered_qty": [500.0] * n,
        "side": ["buy"] * n,
    })


def _venue_fills(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    venues = rng.choice(["ARCA", "BATS", "IEX"], n)
    mid = 100 + rng.normal(0, 0.1, n)
    return pd.DataFrame({
        "venue": venues,
        "fill_qty": rng.integers(10, 200, n).astype(float),
        "midprice": mid,
        "fill_price": mid + rng.normal(-0.01, 0.02, n),
        "fill_time_ms": rng.integers(1, 50, n).astype(float),
        "orders_routed": rng.integers(1, 3, n).astype(float),
    })


def _slippage_fills(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2026-01-02 09:30")
    times = [base + pd.Timedelta(minutes=int(rng.integers(0, 390))) for _ in range(n)]
    return pd.DataFrame({
        "fill_time": times,
        "slippage_bps": rng.normal(2, 5, n),
        "fill_qty": rng.integers(10, 500, n).astype(float),
        "volatility": rng.uniform(0.1, 0.4, n),
    })


def _scorecard_fills(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "strategy": rng.choice(["CS", "IC", "Straddle"], n),
        "slippage_bps": rng.normal(3, 5, n),
        "vs_vwap_bps": rng.normal(1, 4, n),
        "shortfall_bps": rng.normal(2, 6, n),
        "fill_rate": rng.uniform(0.8, 1.0, n),
    })


def _market_data(n: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq="1min")
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    volumes = pd.Series(rng.integers(100, 5000, n).astype(float), index=idx)
    return prices, volumes


# ===========================================================================
# 1. Implementation shortfall
# ===========================================================================

class TestShortfall:
    def test_basic_buy(self):
        sf = FillAnalytics.implementation_shortfall(
            decision_price=100.0, arrival_price=100.02,
            avg_fill_price=100.05, end_price=100.10,
            filled_qty=400, ordered_qty=500, side="buy",
        )
        assert isinstance(sf, ShortfallDecomposition)
        assert sf.delay_cost > 0  # price moved up before arrival
        assert sf.trading_cost > 0  # filled above arrival

    def test_sell(self):
        sf = FillAnalytics.implementation_shortfall(
            decision_price=100.0, arrival_price=99.98,
            avg_fill_price=99.95, end_price=99.90,
            filled_qty=400, ordered_qty=500, side="sell",
        )
        assert isinstance(sf, ShortfallDecomposition)

    def test_total_equals_sum(self):
        sf = FillAnalytics.implementation_shortfall(
            decision_price=100.0, arrival_price=100.02,
            avg_fill_price=100.05, end_price=100.10,
            filled_qty=400, ordered_qty=500, side="buy",
        )
        assert sf.total_shortfall == pytest.approx(
            sf.delay_cost + sf.trading_cost + sf.opportunity_cost)

    def test_zero_unfilled(self):
        sf = FillAnalytics.implementation_shortfall(
            decision_price=100.0, arrival_price=100.0,
            avg_fill_price=100.0, end_price=100.0,
            filled_qty=500, ordered_qty=500, side="buy",
        )
        assert sf.opportunity_cost == 0.0
        assert sf.total_shortfall == 0.0

    def test_bps_computed(self):
        sf = FillAnalytics.implementation_shortfall(
            decision_price=100.0, arrival_price=100.01,
            avg_fill_price=100.02, end_price=100.0,
            filled_qty=100, ordered_qty=100, side="buy",
        )
        assert sf.total_bps != 0.0

    def test_from_fills_df(self):
        fills = _fills_df(20)
        sf = FillAnalytics.shortfall_from_fills(fills)
        assert isinstance(sf, ShortfallDecomposition)

    def test_from_empty(self):
        sf = FillAnalytics.shortfall_from_fills(pd.DataFrame())
        assert sf.total_shortfall == 0.0


# ===========================================================================
# 2. Timing analysis
# ===========================================================================

class TestTiming:
    def test_basic(self):
        mp, mv = _market_data(50)
        fp = mp.iloc[[5, 15, 25]]
        fv = pd.Series([100, 200, 150], index=fp.index)
        ta = FillAnalytics.timing_analysis(fp, fv, mp, mv, side="buy")
        assert isinstance(ta, TimingAnalysis)
        assert ta.actual_avg_price > 0
        assert ta.vwap_price > 0
        assert ta.twap_price > 0

    def test_optimal_is_min_for_buy(self):
        mp, mv = _market_data(50)
        fp = mp.iloc[[10]]
        fv = pd.Series([100], index=fp.index)
        ta = FillAnalytics.timing_analysis(fp, fv, mp, mv, side="buy")
        assert ta.optimal_price == pytest.approx(float(mp.min()))

    def test_empty(self):
        ta = FillAnalytics.timing_analysis(
            pd.Series(dtype=float), pd.Series(dtype=float),
            pd.Series(dtype=float), pd.Series(dtype=float),
        )
        assert ta.actual_avg_price == 0.0

    def test_timing_cost_positive_for_bad_buy(self):
        mp = pd.Series([99.0, 100.0, 101.0])
        mv = pd.Series([100, 100, 100])
        fp = pd.Series([101.0])  # bought at worst price
        fv = pd.Series([100])
        ta = FillAnalytics.timing_analysis(fp, fv, mp, mv, side="buy")
        assert ta.timing_cost > 0


# ===========================================================================
# 3. Venue analysis
# ===========================================================================

class TestVenue:
    def test_basic(self):
        fills = _venue_fills(100)
        venues = FillAnalytics.venue_analysis(fills)
        assert len(venues) == 3
        assert all(isinstance(v, VenueStats) for v in venues)

    def test_sorted_by_price_improvement(self):
        fills = _venue_fills(100)
        venues = FillAnalytics.venue_analysis(fills)
        pis = [v.avg_price_improvement for v in venues]
        assert pis == sorted(pis, reverse=True)

    def test_fill_rate(self):
        fills = _venue_fills(50)
        venues = FillAnalytics.venue_analysis(fills)
        for v in venues:
            assert 0 < v.fill_rate <= 1.0

    def test_empty(self):
        assert FillAnalytics.venue_analysis(pd.DataFrame()) == []

    def test_missing_columns(self):
        df = pd.DataFrame({"foo": [1]})
        assert FillAnalytics.venue_analysis(df) == []


# ===========================================================================
# 4. VWAP / TWAP benchmarks
# ===========================================================================

class TestBenchmarks:
    def test_vwap(self):
        prices = pd.Series([100, 101, 102])
        volumes = pd.Series([1000, 2000, 1000])
        vwap = FillAnalytics.compute_vwap(prices, volumes)
        expected = (100 * 1000 + 101 * 2000 + 102 * 1000) / 4000
        assert vwap == pytest.approx(expected)

    def test_twap(self):
        prices = pd.Series([100, 101, 102])
        assert FillAnalytics.compute_twap(prices) == pytest.approx(101.0)

    def test_comparison_buy(self):
        mp = pd.Series([99, 100, 101])
        mv = pd.Series([100, 200, 100])
        bc = FillAnalytics.benchmark_comparison(100.5, mp, mv, side="buy")
        assert isinstance(bc, BenchmarkComparison)
        assert bc.vwap > 0
        assert bc.twap > 0

    def test_comparison_sell(self):
        mp = pd.Series([99, 100, 101])
        mv = pd.Series([100, 200, 100])
        bc = FillAnalytics.benchmark_comparison(99.5, mp, mv, side="sell")
        # Selling below VWAP → negative (good for seller actually means bps < 0)
        assert isinstance(bc, BenchmarkComparison)

    def test_benchmark_fills(self):
        mp, mv = _market_data(50)
        fills = pd.DataFrame({
            "fill_price": [100.01, 100.02, 100.03],
            "side": ["buy", "buy", "sell"],
        })
        bcs = FillAnalytics.benchmark_fills(fills, mp, mv)
        assert len(bcs) == 3

    def test_benchmark_empty(self):
        mp, mv = _market_data(10)
        assert FillAnalytics.benchmark_fills(pd.DataFrame(), mp, mv) == []


# ===========================================================================
# 5. Slippage attribution
# ===========================================================================

class TestSlippage:
    def test_by_time(self):
        fills = _slippage_fills(100)
        buckets = FillAnalytics.slippage_by_time_of_day(fills)
        assert len(buckets) > 0
        assert all(b.bucket_type == "time_of_day" for b in buckets)

    def test_by_volatility(self):
        fills = _slippage_fills(100)
        buckets = FillAnalytics.slippage_by_volatility(fills, n_buckets=3)
        assert len(buckets) > 0
        assert all(b.bucket_type == "volatility" for b in buckets)

    def test_by_size(self):
        fills = _slippage_fills(100)
        buckets = FillAnalytics.slippage_by_size(fills, n_buckets=3)
        assert len(buckets) > 0
        assert all(b.bucket_type == "size" for b in buckets)

    def test_empty(self):
        assert FillAnalytics.slippage_by_time_of_day(pd.DataFrame()) == []
        assert FillAnalytics.slippage_by_volatility(pd.DataFrame()) == []
        assert FillAnalytics.slippage_by_size(pd.DataFrame()) == []


# ===========================================================================
# 6. Execution scorecard
# ===========================================================================

class TestScorecard:
    def test_basic(self):
        fills = _scorecard_fills(100)
        cards = FillAnalytics.execution_scorecard(fills)
        assert len(cards) == 3
        assert all(isinstance(c, ExecutionScorecard) for c in cards)

    def test_sorted_by_score(self):
        fills = _scorecard_fills(100)
        cards = FillAnalytics.execution_scorecard(fills)
        scores = [c.score for c in cards]
        assert scores == sorted(scores, reverse=True)

    def test_score_bounded(self):
        fills = _scorecard_fills(100)
        cards = FillAnalytics.execution_scorecard(fills)
        for c in cards:
            assert 0 <= c.score <= 100

    def test_trade_counts(self):
        fills = _scorecard_fills(80)
        cards = FillAnalytics.execution_scorecard(fills)
        assert sum(c.n_trades for c in cards) == 80

    def test_empty(self):
        assert FillAnalytics.execution_scorecard(pd.DataFrame()) == []


# ===========================================================================
# 7. HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        fa = FillAnalytics()
        sf = ShortfallDecomposition(
            total_shortfall=0.05, delay_cost=0.01,
            trading_cost=0.03, opportunity_cost=0.01, total_bps=5.0)
        out = tmp_path / "fill.html"
        result = fa.generate_report(shortfall=sf, output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Fill Quality Analytics" in html
        assert "Implementation Shortfall" in html

    def test_with_timing(self, tmp_path):
        fa = FillAnalytics()
        ta = TimingAnalysis(
            actual_avg_price=100.05, optimal_price=100.00,
            vwap_price=100.02, twap_price=100.01,
            timing_cost=0.05, timing_cost_bps=5.0)
        out = tmp_path / "fill.html"
        fa.generate_report(timing=ta, output_path=str(out))
        html = out.read_text()
        assert "Timing Analysis" in html

    def test_with_venues(self, tmp_path):
        fa = FillAnalytics()
        venues = [
            VenueStats("ARCA", 50, 10000, 0.95, 0.005, 12),
            VenueStats("IEX", 30, 8000, 0.98, 0.008, 8),
        ]
        out = tmp_path / "fill.html"
        fa.generate_report(venues=venues, output_path=str(out))
        html = out.read_text()
        assert "Venue Analysis" in html

    def test_with_slippage_charts(self, tmp_path):
        fa = FillAnalytics()
        tod = [SlippageBucket("09:00", "time_of_day", 20, 3.5, 70, 50),
               SlippageBucket("12:00", "time_of_day", 30, 1.2, 36, 45)]
        out = tmp_path / "fill.html"
        fa.generate_report(slippage_tod=tod, output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
        assert "Time of Day" in html

    def test_with_scorecards(self, tmp_path):
        fa = FillAnalytics()
        sc = [ExecutionScorecard("CS", 40, 2.5, 1.0, 3.0, 0.95, 85)]
        out = tmp_path / "fill.html"
        fa.generate_report(scorecards=sc, output_path=str(out))
        html = out.read_text()
        assert "Execution Scorecards" in html

    def test_full_report(self, tmp_path):
        fa = FillAnalytics()
        sf = ShortfallDecomposition(0.05, 0.01, 0.03, 0.01, 5.0)
        ta = TimingAnalysis(100.05, 100.0, 100.02, 100.01, 0.05, 5.0)
        venues = [VenueStats("ARCA", 50, 10000, 0.95, 0.005, 12)]
        tod = [SlippageBucket("09:00", "time_of_day", 20, 3.5, 70, 50)]
        sc = [ExecutionScorecard("CS", 40, 2.5, 1.0, 3.0, 0.95, 85)]
        out = tmp_path / "full.html"
        result = fa.generate_report(
            shortfall=sf, timing=ta, venues=venues,
            slippage_tod=tod, scorecards=sc, output_path=str(out))
        html = Path(result).read_text()
        for section in ["Shortfall", "Timing", "Venue", "Slippage", "Scorecard"]:
            assert section in html
