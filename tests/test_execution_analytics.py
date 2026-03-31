"""Tests for compass.execution_analytics — 32 tests."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from compass.execution_analytics import (
    ExecutionAnalytics, ShortfallResult, BenchmarkResult, TimingResult,
    CostAttribution, VenueResult, QualityScore, ExecutionReport,
)

def _market(n=50, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-02 09:30", periods=n, freq="1min")
    prices = pd.Series(100 + np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    volumes = pd.Series(rng.integers(100, 5000, n).astype(float), index=idx)
    return prices, volumes

def _fills(n=50, seed=42):
    rng = np.random.default_rng(seed)
    mid = 100 + rng.normal(0, 0.1, n)
    return pd.DataFrame({
        "venue": rng.choice(["ARCA", "BATS", "IEX"], n),
        "fill_qty": rng.integers(10, 200, n).astype(float),
        "midprice": mid,
        "fill_price": mid + rng.normal(0, 0.02, n),
        "fill_time_ms": rng.integers(1, 50, n).astype(float),
        "orders_routed": rng.integers(1, 3, n).astype(float),
    })


class TestShortfall:
    def test_buy(self):
        sf = ExecutionAnalytics.implementation_shortfall(100, 100.02, 100.05, 100.10, 400, 500, "buy")
        assert isinstance(sf, ShortfallResult)
        assert sf.delay_cost > 0
    def test_sell(self):
        sf = ExecutionAnalytics.implementation_shortfall(100, 99.98, 99.95, 99.90, 400, 500, "sell")
        assert isinstance(sf, ShortfallResult)
    def test_total_is_sum(self):
        sf = ExecutionAnalytics.implementation_shortfall(100, 100.01, 100.03, 100.05, 100, 100, "buy")
        expected = sf.delay_cost + sf.trading_cost + sf.opportunity_cost
        ref = abs(100 * 100)
        bps = expected / ref * 10000
        assert sf.total_bps == pytest.approx(bps)
    def test_zero(self):
        sf = ExecutionAnalytics.implementation_shortfall(100, 100, 100, 100, 100, 100, "buy")
        assert sf.total_bps == 0.0
    def test_from_df(self):
        df = pd.DataFrame({
            "decision_price": [100.0] * 3,
            "arrival_price": [100.01] * 3,
            "fill_price": [100.02, 100.03, 100.01],
            "end_price": [100.05] * 3,
            "fill_qty": [100.0, 150.0, 50.0],
            "ordered_qty": [500.0] * 3,
            "side": ["buy"] * 3,
        })
        sf = ExecutionAnalytics.shortfall_from_df(df)
        assert isinstance(sf, ShortfallResult)
    def test_from_empty(self):
        sf = ExecutionAnalytics.shortfall_from_df(pd.DataFrame())
        assert sf.total_bps == 0.0


class TestBenchmark:
    def test_basic(self):
        mp, mv = _market()
        bm = ExecutionAnalytics.benchmark(100.03, mp, mv, "buy")
        assert isinstance(bm, BenchmarkResult)
        assert bm.vwap > 0
        assert bm.twap > 0
    def test_empty(self):
        bm = ExecutionAnalytics.benchmark(100, pd.Series(dtype=float), pd.Series(dtype=float))
        assert bm.vwap == 0.0


class TestTiming:
    def test_basic(self):
        mp, _ = _market()
        tm = ExecutionAnalytics.timing_analysis(100.05, mp, "buy")
        assert isinstance(tm, TimingResult)
    def test_optimal_for_buy(self):
        mp, _ = _market()
        best = float(mp.min())
        tm = ExecutionAnalytics.timing_analysis(best, mp, "buy")
        assert tm.timing_cost_bps == pytest.approx(0.0, abs=0.1)
        assert tm.was_optimal
    def test_empty(self):
        tm = ExecutionAnalytics.timing_analysis(100, pd.Series(dtype=float))
        assert tm.actual_price == 100


class TestCostAttribution:
    def test_basic(self):
        sf = ShortfallResult(10, 0.01, 0.02, 0.005)
        ca = ExecutionAnalytics.cost_attribution(sf, 2.0)
        assert isinstance(ca, CostAttribution)
        assert ca.total_cost > 0
    def test_zero(self):
        ca = ExecutionAnalytics.cost_attribution(ShortfallResult())
        assert ca.total_cost == 0.0


class TestVenue:
    def test_basic(self):
        venues = ExecutionAnalytics.venue_analysis(_fills())
        assert len(venues) == 3
        assert all(isinstance(v, VenueResult) for v in venues)
    def test_sorted(self):
        venues = ExecutionAnalytics.venue_analysis(_fills())
        pis = [v.avg_price_improvement for v in venues]
        assert pis == sorted(pis, reverse=True)
    def test_empty(self):
        assert ExecutionAnalytics.venue_analysis(pd.DataFrame()) == []


class TestQuality:
    def test_good(self):
        qs = ExecutionAnalytics.quality_score(2.0, 1.0, 1.0)
        assert 70 <= qs.score <= 100
    def test_bad(self):
        qs = ExecutionAnalytics.quality_score(50.0, 30.0, 0.5)
        assert qs.score < 50
    def test_rolling(self):
        df = pd.DataFrame({
            "shortfall_bps": np.random.default_rng(42).normal(3, 2, 50),
            "timing_bps": np.random.default_rng(43).normal(2, 1, 50),
            "fill_rate": np.random.default_rng(44).uniform(0.8, 1.0, 50),
        })
        scores = ExecutionAnalytics.rolling_quality(df, window=10)
        assert len(scores) == 41
    def test_rolling_empty(self):
        assert ExecutionAnalytics.rolling_quality(pd.DataFrame()) == []


class TestRecommendations:
    def test_high_shortfall(self):
        sf = ShortfallResult(total_bps=15)
        tm = TimingResult(was_optimal=True)
        recs = ExecutionAnalytics.recommend(sf, tm, [])
        assert any("shortfall" in r.lower() for r in recs)
    def test_bad_timing(self):
        sf = ShortfallResult()
        tm = TimingResult(was_optimal=False, timing_cost_bps=8)
        recs = ExecutionAnalytics.recommend(sf, tm, [])
        assert any("timing" in r.lower() for r in recs)


class TestFullAnalysis:
    def test_basic(self):
        mp, mv = _market()
        ea = ExecutionAnalytics()
        report = ea.analyze(100, 100.01, 100.03, 100.05, 400, 500, "buy",
                             mp, mv, _fills())
        assert isinstance(report, ExecutionReport)
        assert report.quality.score > 0


class TestHTMLReport:
    def test_creates_file(self, tmp_path):
        mp, mv = _market()
        ea = ExecutionAnalytics()
        report = ea.analyze(100, 100.01, 100.03, 100.05, 400, 500, "buy", mp, mv)
        out = tmp_path / "exec.html"
        path = ea.generate_report(report, output_path=str(out))
        assert Path(path).exists()
        assert "Execution Analytics" in out.read_text()
    def test_contains_cost_chart(self, tmp_path):
        ea = ExecutionAnalytics()
        report = ea.analyze(100, 100.01, 100.03, 100.05, 400, 500, "buy")
        out = tmp_path / "e.html"
        ea.generate_report(report, output_path=str(out))
        assert "<svg" in out.read_text()
