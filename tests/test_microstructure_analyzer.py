"""Tests for compass.microstructure_analyzer — 42 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.microstructure_analyzer import (
    MicrostructureAnalyzer,
    SpreadEstimate,
    PriceImpact,
    ToxicityMetrics,
    IntradayPattern,
    OvernightGap,
    ExecutionWindow,
    MakerTakerStats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=n)


def _prices(n: int = 200, start: float = 100.0, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.01, n)
    prices = start * np.cumprod(1 + returns)
    return pd.Series(prices, index=_dates(n), name="price")


def _bid_ask(n: int = 200, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    mid = _prices(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    half_spread = rng.uniform(0.01, 0.05, n)
    bid = mid - half_spread
    ask = mid + half_spread
    return bid, ask


def _volume(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(100, 10000, n).astype(float), index=_dates(n))


def _signed_volume(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 500, n), index=_dates(n))


def _fills(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mid = 100 + rng.normal(0, 0.5, n)
    spread = rng.uniform(0.02, 0.08, n)
    sides = rng.choice(["maker", "taker"], n)
    price = mid + np.where(sides == "taker", spread / 2, -spread / 4)
    return pd.DataFrame({
        "side": sides, "price": price,
        "midprice": mid, "spread": spread,
    })


# ===========================================================================
# 1. Bid-ask spread estimation
# ===========================================================================

class TestRollSpread:
    def test_positive_or_zero(self):
        p = _prices(200)
        s = MicrostructureAnalyzer.roll_spread(p)
        assert s >= 0.0

    def test_nontrivial(self):
        # Inject negative autocovariance
        rng = np.random.default_rng(7)
        dp = rng.choice([-0.05, 0.05], 200)
        # Alternate signs to create negative autocov
        for i in range(1, len(dp)):
            dp[i] = -dp[i - 1] + rng.normal(0, 0.01)
        p = pd.Series(100 + np.cumsum(dp), index=_dates(200))
        s = MicrostructureAnalyzer.roll_spread(p)
        assert s > 0

    def test_short_series(self):
        p = pd.Series([100.0, 101.0], index=_dates(2))
        assert MicrostructureAnalyzer.roll_spread(p) == 0.0


class TestEffectiveSpread:
    def test_basic(self):
        mid = pd.Series([100.0, 100.0, 100.0])
        trade = pd.Series([100.03, 99.97, 100.02])
        s = MicrostructureAnalyzer.effective_spread(trade, mid)
        assert s > 0

    def test_zero_when_at_mid(self):
        mid = pd.Series([100.0, 100.0])
        s = MicrostructureAnalyzer.effective_spread(mid, mid)
        assert s == pytest.approx(0.0)

    def test_empty(self):
        assert MicrostructureAnalyzer.effective_spread(pd.Series(), pd.Series()) == 0.0


class TestQuotedSpread:
    def test_positive(self):
        bid, ask = _bid_ask(100)
        s = MicrostructureAnalyzer.quoted_spread(bid, ask)
        assert s > 0

    def test_relative_bounded(self):
        bid, ask = _bid_ask(100)
        r = MicrostructureAnalyzer.relative_spread(bid, ask)
        assert 0.0 < r < 0.01  # sub-1%


class TestEstimateSpreads:
    def test_all_fields(self):
        ma = MicrostructureAnalyzer()
        bid, ask = _bid_ask(100)
        mid = (bid + ask) / 2
        se = ma.estimate_spreads(
            mid, bid=bid, ask=ask,
            trade_prices=mid + 0.01, mid_prices=mid,
            date=datetime(2026, 1, 1),
        )
        assert isinstance(se, SpreadEstimate)
        assert se.quoted_spread > 0
        assert se.relative_spread > 0

    def test_rolling(self):
        ma = MicrostructureAnalyzer()
        p = _prices(100)
        rs = ma.rolling_spread(p, window=21)
        assert len(rs) == len(p)
        assert rs.iloc[21:].notna().any()


# ===========================================================================
# 2. Price impact
# ===========================================================================

class TestKyleLambda:
    def test_basic(self):
        n = 200
        rng = np.random.default_rng(42)
        sv = pd.Series(rng.normal(0, 500, n), index=_dates(n))
        dp = 0.0001 * sv + pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
        pi = MicrostructureAnalyzer.kyle_lambda(dp, sv)
        assert isinstance(pi, PriceImpact)
        assert pi.kyle_lambda > 0

    def test_r_squared(self):
        n = 200
        rng = np.random.default_rng(42)
        sv = pd.Series(rng.normal(0, 500, n), index=_dates(n))
        dp = 0.001 * sv  # high R²
        pi = MicrostructureAnalyzer.kyle_lambda(dp, sv)
        assert pi.r_squared > 0.5

    def test_short_data(self):
        dp = pd.Series([0.01, 0.02])
        sv = pd.Series([100, -100])
        pi = MicrostructureAnalyzer.kyle_lambda(dp, sv)
        assert pi.kyle_lambda == 0.0


class TestPermanentTemporary:
    def test_decomposition(self):
        n = 200
        rng = np.random.default_rng(42)
        sv = pd.Series(rng.normal(0, 500, n), index=_dates(n))
        dp = 0.0001 * sv + pd.Series(rng.normal(0, 0.01, n), index=_dates(n))
        pi = MicrostructureAnalyzer.permanent_temporary_impact(dp, sv, lag=5)
        assert isinstance(pi, PriceImpact)
        # permanent + temporary ≈ total
        assert pi.permanent_impact + pi.temporary_impact == pytest.approx(pi.total_impact, abs=1e-6)


# ===========================================================================
# 3. Order flow toxicity
# ===========================================================================

class TestVPIN:
    def test_bounded(self):
        ma = MicrostructureAnalyzer(n_vpin_buckets=20)
        vol = _volume(200)
        dp = _prices(200).pct_change().fillna(0)
        vpin = ma.compute_vpin(vol, dp)
        assert 0.0 <= vpin <= 2.0

    def test_empty(self):
        ma = MicrostructureAnalyzer()
        assert ma.compute_vpin(pd.Series(), pd.Series()) == 0.0


class TestOrderImbalance:
    def test_balanced(self):
        buy = pd.Series([100, 100, 100])
        sell = pd.Series([100, 100, 100])
        oi = MicrostructureAnalyzer.order_imbalance(buy, sell)
        assert oi == pytest.approx(0.0)

    def test_fully_imbalanced(self):
        buy = pd.Series([100, 100])
        sell = pd.Series([0, 0])
        oi = MicrostructureAnalyzer.order_imbalance(buy, sell)
        assert oi == pytest.approx(1.0)


class TestToxicity:
    def test_levels(self):
        ma = MicrostructureAnalyzer(toxicity_threshold=0.5)
        vol = _volume(200)
        dp = _prices(200).pct_change().fillna(0)
        tox = ma.compute_toxicity(vol, dp)
        assert isinstance(tox, ToxicityMetrics)
        assert tox.toxicity_level in ("low", "normal", "elevated", "toxic")

    def test_with_buy_sell(self):
        ma = MicrostructureAnalyzer()
        vol = _volume(100)
        dp = _prices(100).pct_change().fillna(0)
        buy = vol * 0.6
        sell = vol * 0.4
        tox = ma.compute_toxicity(vol, dp, buy, sell)
        assert tox.order_imbalance > 0
        assert tox.buy_ratio > 0.5


# ===========================================================================
# 4. Intraday patterns
# ===========================================================================

class TestIntradayPatterns:
    def test_with_bucket_column(self):
        ma = MicrostructureAnalyzer()
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "time_bucket": np.tile(
                ["09:30", "10:00", "10:30", "11:00", "11:30",
                 "12:00", "12:30", "13:00", "13:30", "14:00",
                 "14:30", "15:00", "15:30"], n // 13 + 1)[:n],
            "return": rng.normal(0, 0.01, n),
        })
        patterns = ma.intraday_patterns(df)
        assert len(patterns) == 13

    def test_with_series(self):
        ma = MicrostructureAnalyzer()
        idx = pd.date_range("2026-01-02 09:30", periods=100, freq="30min")
        ret = pd.Series(np.random.default_rng(42).normal(0, 0.01, 100), index=idx)
        patterns = ma.intraday_patterns(ret)
        assert len(patterns) > 0

    def test_u_shape_detection(self):
        # Construct U-shaped pattern
        patterns = [
            IntradayPattern(f"{h:02d}:00", avg_volatility=v, avg_spread=0.0, avg_volume=100, n_observations=10)
            for h, v in [(9, 0.30), (10, 0.15), (11, 0.12), (12, 0.10),
                          (13, 0.11), (14, 0.14), (15, 0.28)]
        ]
        assert MicrostructureAnalyzer.detect_u_shape(patterns)

    def test_flat_not_u_shape(self):
        patterns = [
            IntradayPattern(f"{h:02d}:00", avg_volatility=0.15, avg_spread=0, avg_volume=100, n_observations=10)
            for h in range(9, 16)
        ]
        assert not MicrostructureAnalyzer.detect_u_shape(patterns)

    def test_too_few_buckets(self):
        assert not MicrostructureAnalyzer.detect_u_shape([])


class TestOvernightGaps:
    def test_basic(self):
        opens = pd.Series([100.5, 101.0, 99.5], index=_dates(3))
        closes = pd.Series([100.0, 101.5, 99.0], index=_dates(3))
        gaps = MicrostructureAnalyzer.overnight_gaps(opens, closes)
        assert len(gaps) == 2
        assert all(isinstance(g, OvernightGap) for g in gaps)

    def test_gap_direction(self):
        opens = pd.Series([100.0, 105.0], index=_dates(2))
        closes = pd.Series([100.0, 105.0], index=_dates(2))
        gaps = MicrostructureAnalyzer.overnight_gaps(opens, closes)
        assert gaps[0].gap_return > 0  # gapped up


# ===========================================================================
# 5. Optimal execution windows
# ===========================================================================

class TestExecutionWindows:
    def test_ranking(self):
        ma = MicrostructureAnalyzer()
        patterns = [
            IntradayPattern("09:30", avg_volatility=0.3, avg_spread=0.05, avg_volume=5000, n_observations=20),
            IntradayPattern("12:00", avg_volatility=0.1, avg_spread=0.02, avg_volume=2000, n_observations=20),
            IntradayPattern("15:00", avg_volatility=0.25, avg_spread=0.04, avg_volume=8000, n_observations=20),
        ]
        windows = ma.optimal_execution_windows(patterns, top_n=2)
        assert len(windows) == 2
        assert windows[0].rank == 1
        assert windows[1].rank == 2

    def test_empty(self):
        ma = MicrostructureAnalyzer()
        assert ma.optimal_execution_windows([], top_n=3) == []

    def test_quality_score(self):
        ma = MicrostructureAnalyzer()
        # High volume + low spread should score best
        patterns = [
            IntradayPattern("A", avg_volatility=0.1, avg_spread=0.01, avg_volume=10000, n_observations=10),
            IntradayPattern("B", avg_volatility=0.2, avg_spread=0.05, avg_volume=1000, n_observations=10),
        ]
        windows = ma.optimal_execution_windows(patterns, top_n=2)
        assert windows[0].bucket_label == "A"


# ===========================================================================
# 6. Maker vs taker
# ===========================================================================

class TestMakerTaker:
    def test_basic(self):
        fills = _fills(100)
        stats = MicrostructureAnalyzer.maker_taker_analysis(fills)
        assert len(stats) == 2
        sides = {s.side for s in stats}
        assert sides == {"maker", "taker"}

    def test_fill_rate_sums_to_one(self):
        fills = _fills(100)
        stats = MicrostructureAnalyzer.maker_taker_analysis(fills)
        total = sum(s.fill_rate for s in stats)
        assert total == pytest.approx(1.0)

    def test_missing_columns(self):
        df = pd.DataFrame({"foo": [1, 2]})
        assert MicrostructureAnalyzer.maker_taker_analysis(df) == []


# ===========================================================================
# 7. HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        ma = MicrostructureAnalyzer()
        se = SpreadEstimate(roll_spread=0.03, effective_spread=0.04,
                            quoted_spread=0.05, relative_spread=0.001)
        out = tmp_path / "micro.html"
        result = ma.generate_report(spread=se, output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Microstructure" in html
        assert "Roll" in html

    def test_with_toxicity(self, tmp_path):
        ma = MicrostructureAnalyzer()
        tox = ToxicityMetrics(vpin=0.65, order_imbalance=0.3,
                               buy_ratio=0.65, toxicity_level="elevated")
        out = tmp_path / "m.html"
        ma.generate_report(toxicity=tox, output_path=str(out))
        html = out.read_text()
        assert "VPIN" in html
        assert "elevated" in html.lower()

    def test_with_heatmap(self, tmp_path):
        ma = MicrostructureAnalyzer()
        patterns = [
            IntradayPattern(f"{h:02d}:00", avg_volatility=0.1 + h * 0.01,
                            avg_spread=0.02, avg_volume=1000, n_observations=10)
            for h in range(9, 16)
        ]
        out = tmp_path / "m.html"
        ma.generate_report(patterns=patterns, output_path=str(out))
        html = out.read_text()
        assert "Execution Quality Heatmap" in html
        assert "<svg" in html

    def test_full_report(self, tmp_path):
        ma = MicrostructureAnalyzer()
        se = SpreadEstimate(roll_spread=0.03, effective_spread=0.04)
        pi = PriceImpact(kyle_lambda=0.001, permanent_impact=0.0008,
                          temporary_impact=0.0002, total_impact=0.001)
        tox = ToxicityMetrics(vpin=0.4, toxicity_level="normal")
        patterns = [
            IntradayPattern(f"{h:02d}:00", 0.15, 0.03, 5000, 20)
            for h in range(9, 16)
        ]
        windows = ma.optimal_execution_windows(patterns, top_n=3)
        mt = [MakerTakerStats("maker", 50, 0.01, 0.03, 0.5),
              MakerTakerStats("taker", 50, 0.02, 0.04, 0.5)]
        out = tmp_path / "full.html"
        result = ma.generate_report(
            spread=se, impact=pi, toxicity=tox,
            patterns=patterns, windows=windows,
            maker_taker=mt, rolling_spreads=[0.03] * 50,
            output_path=str(out),
        )
        html = Path(result).read_text()
        for section in ["Roll", "Kyle", "VPIN", "Execution Quality",
                         "Optimal Execution", "Maker vs Taker"]:
            assert section in html
