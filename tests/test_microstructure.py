"""Tests for compass.microstructure — 42 tests."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from compass.microstructure import (
    MicrostructureEngine,
    SpreadEstimate,
    OrderFlowMetrics,
    PriceImpactEstimate,
    TradeClassification,
    IntradayVolatility,
    OvernightGap,
    InformedTradingEstimate,
    LiquidityMetrics,
    MicrostructureSummary,
    TRADING_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dates(n: int = 200) -> pd.DatetimeIndex:
    return pd.bdate_range(start="2024-01-02", periods=n)


def _prices(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.01, n)
    return pd.Series(100.0 * np.cumprod(1 + r), index=_dates(n))


def _volume(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.integers(1000, 50000, n).astype(float), index=_dates(n))


def _bid_ask(n: int = 200, seed: int = 42):
    mid = _prices(n, seed=seed)
    rng = np.random.default_rng(seed + 1)
    hs = rng.uniform(0.02, 0.06, n)
    return mid - hs, mid + hs


def _signed_volume(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 1000, n), index=_dates(n))


# ===========================================================================
# 1. Spread estimation
# ===========================================================================

class TestRollSpread:
    def test_non_negative(self):
        assert MicrostructureEngine.roll_spread(_prices()) >= 0

    def test_alternating_prices(self):
        rng = np.random.default_rng(7)
        dp = np.empty(200)
        dp[0] = 0.05
        for i in range(1, 200):
            dp[i] = -dp[i - 1] + rng.normal(0, 0.005)
        p = pd.Series(100 + np.cumsum(dp), index=_dates(200))
        assert MicrostructureEngine.roll_spread(p) > 0

    def test_short(self):
        assert MicrostructureEngine.roll_spread(pd.Series([100.0, 101.0])) == 0.0


class TestEffectiveSpread:
    def test_positive(self):
        mid = pd.Series([100.0] * 5)
        tp = pd.Series([100.03, 99.97, 100.02, 99.98, 100.01])
        assert MicrostructureEngine.effective_spread(tp, mid) > 0

    def test_at_mid(self):
        mid = pd.Series([100.0] * 5)
        assert MicrostructureEngine.effective_spread(mid, mid) == pytest.approx(0.0)

    def test_empty(self):
        assert MicrostructureEngine.effective_spread(pd.Series(), pd.Series()) == 0.0


class TestRealisedSpread:
    def test_basic(self):
        tp = pd.Series([100.0] * 10)
        mid = pd.Series([99.97] * 10)
        direction = pd.Series([1] * 10)
        rs = MicrostructureEngine.realised_spread(tp, mid, direction, lag=2)
        assert isinstance(rs, float)

    def test_empty(self):
        assert MicrostructureEngine.realised_spread(
            pd.Series(), pd.Series(), pd.Series()
        ) == 0.0


class TestQuotedRelative:
    def test_quoted_positive(self):
        bid, ask = _bid_ask(100)
        assert MicrostructureEngine.quoted_spread(bid, ask) > 0

    def test_relative_bounded(self):
        bid, ask = _bid_ask(100)
        r = MicrostructureEngine.relative_spread(bid, ask)
        assert 0 < r < 0.01

    def test_empty(self):
        assert MicrostructureEngine.quoted_spread(pd.Series(), pd.Series()) == 0.0


class TestEstimateSpreads:
    def test_all_fields(self):
        me = MicrostructureEngine()
        bid, ask = _bid_ask(100)
        mid = (bid + ask) / 2
        se = me.estimate_spreads(
            mid, bid=bid, ask=ask,
            trade_prices=mid + 0.01, mid_prices=mid,
            direction=pd.Series([1] * 100, index=bid.index),
        )
        assert isinstance(se, SpreadEstimate)
        assert se.quoted_spread > 0

    def test_rolling(self):
        me = MicrostructureEngine()
        rs = me.rolling_roll_spread(_prices(100), window=21)
        assert len(rs) == 100


# ===========================================================================
# 2. Order flow
# ===========================================================================

class TestVPIN:
    def test_bounded(self):
        me = MicrostructureEngine(n_vpin_buckets=20)
        dp = _prices(200).pct_change().fillna(0)
        v = _volume(200)
        assert 0.0 <= me.compute_vpin(v, dp) <= 2.0

    def test_empty(self):
        assert MicrostructureEngine(n_vpin_buckets=10).compute_vpin(
            pd.Series(), pd.Series()) == 0.0


class TestKyleLambda:
    def test_positive_with_impact(self):
        rng = np.random.default_rng(42)
        sv = pd.Series(rng.normal(0, 500, 200), index=_dates(200))
        dp = 0.0001 * sv + pd.Series(rng.normal(0, 0.01, 200), index=_dates(200))
        lam, r2 = MicrostructureEngine.kyle_lambda(dp, sv)
        assert lam > 0

    def test_short(self):
        lam, r2 = MicrostructureEngine.kyle_lambda(pd.Series([1]), pd.Series([1]))
        assert lam == 0.0


class TestOrderFlow:
    def test_full(self):
        me = MicrostructureEngine(n_vpin_buckets=20)
        dp = _prices(200).diff().fillna(0)
        v = _volume(200)
        sv = _signed_volume(200)
        of = me.compute_order_flow(v, dp, sv)
        assert isinstance(of, OrderFlowMetrics)
        assert 0.0 <= of.buy_volume_pct <= 1.0


# ===========================================================================
# 3. Price impact
# ===========================================================================

class TestPriceImpact:
    def test_decomposition(self):
        me = MicrostructureEngine(impact_lag=5)
        rng = np.random.default_rng(42)
        sv = pd.Series(rng.normal(0, 500, 200), index=_dates(200))
        dp = 0.0001 * sv + pd.Series(rng.normal(0, 0.01, 200), index=_dates(200))
        pi = me.estimate_price_impact(dp, sv)
        assert isinstance(pi, PriceImpactEstimate)
        assert pi.permanent_impact + pi.temporary_impact == pytest.approx(
            pi.total_impact, abs=1e-6)

    def test_short_data(self):
        me = MicrostructureEngine()
        pi = me.estimate_price_impact(pd.Series([1.0, 2.0]), pd.Series([10, -10]))
        assert pi.total_impact == 0.0


# ===========================================================================
# 4. Lee-Ready
# ===========================================================================

class TestLeeReady:
    def test_tick_test_only(self):
        tp = pd.Series([100.0, 100.1, 99.9, 100.0, 100.2])
        tc = MicrostructureEngine.lee_ready_classify(tp)
        assert isinstance(tc, TradeClassification)
        assert tc.n_trades == 5
        assert tc.n_buys + tc.n_sells + tc.n_unclassified == 5

    def test_quote_test(self):
        tp = pd.Series([100.03, 99.97, 100.01])
        bid = pd.Series([99.98, 99.95, 99.99])
        ask = pd.Series([100.02, 100.01, 100.03])
        tc = MicrostructureEngine.lee_ready_classify(tp, bid=bid, ask=ask)
        assert tc.n_buys >= 1  # 100.03 > mid(100.00) → buy

    def test_volume_tracked(self):
        tp = pd.Series([100.0, 100.1, 99.9])
        vol = pd.Series([100.0, 200.0, 150.0])
        tc = MicrostructureEngine.lee_ready_classify(tp, volume=vol)
        assert tc.buy_volume + tc.sell_volume > 0

    def test_empty(self):
        tc = MicrostructureEngine.lee_ready_classify(pd.Series(dtype=float))
        assert tc.n_trades == 0

    def test_buy_pct_bounded(self):
        tp = pd.Series([100 + i * 0.01 for i in range(50)])
        tc = MicrostructureEngine.lee_ready_classify(tp)
        assert 0.0 <= tc.buy_pct <= 1.0


# ===========================================================================
# 5. Intraday volatility
# ===========================================================================

class TestIntraday:
    def test_hourly_buckets(self):
        idx = pd.date_range("2026-01-02 09:30", periods=100, freq="30min")
        ret = pd.Series(np.random.default_rng(42).normal(0, 0.01, 100), index=idx)
        patterns = MicrostructureEngine.intraday_volatility(ret)
        assert len(patterns) > 0
        assert all(isinstance(p, IntradayVolatility) for p in patterns)

    def test_no_hour_index(self):
        ret = pd.Series([0.01] * 10, index=range(10))
        assert MicrostructureEngine.intraday_volatility(ret) == []

    def test_u_shape_yes(self):
        patterns = [
            IntradayVolatility(f"{h}:00", v, 0, 0, 10)
            for h, v in [(9, .30), (10, .15), (11, .12), (12, .10),
                          (13, .11), (14, .14), (15, .28)]
        ]
        assert MicrostructureEngine.detect_u_shape(patterns)

    def test_u_shape_flat(self):
        patterns = [
            IntradayVolatility(f"{h}:00", 0.15, 0, 0, 10)
            for h in range(9, 16)
        ]
        assert not MicrostructureEngine.detect_u_shape(patterns)

    def test_u_shape_too_few(self):
        assert not MicrostructureEngine.detect_u_shape([])


class TestOvernightGaps:
    def test_basic(self):
        o = pd.Series([100.5, 101.0, 99.5], index=_dates(3))
        c = pd.Series([100.0, 101.5, 99.0], index=_dates(3))
        gaps = MicrostructureEngine.overnight_gaps(o, c)
        assert len(gaps) == 2
        assert all(isinstance(g, OvernightGap) for g in gaps)

    def test_gap_up(self):
        o = pd.Series([100.0, 105.0], index=_dates(2))
        c = pd.Series([100.0, 105.0], index=_dates(2))
        gaps = MicrostructureEngine.overnight_gaps(o, c)
        assert gaps[0].gap_return > 0


# ===========================================================================
# 6. PIN estimation
# ===========================================================================

class TestPIN:
    def test_basic(self):
        rng = np.random.default_rng(42)
        buys = pd.Series(rng.poisson(100, 200))
        sells = pd.Series(rng.poisson(100, 200))
        # Inject info days
        buys.iloc[10:20] = 300
        pin = MicrostructureEngine.estimate_pin(buys, sells)
        assert isinstance(pin, InformedTradingEstimate)
        assert 0.0 <= pin.pin <= 1.0

    def test_balanced(self):
        buys = pd.Series([100] * 100)
        sells = pd.Series([100] * 100)
        pin = MicrostructureEngine.estimate_pin(buys, sells)
        assert pin.pin < 0.3  # low info

    def test_short_data(self):
        pin = MicrostructureEngine.estimate_pin(pd.Series([10]), pd.Series([10]))
        assert pin.pin == 0.0


# ===========================================================================
# 7. Liquidity
# ===========================================================================

class TestLiquidity:
    def test_amihud(self):
        ret = pd.Series(np.random.default_rng(42).normal(0, 0.01, 200), index=_dates(200))
        dvol = pd.Series(np.random.default_rng(43).uniform(1e6, 1e7, 200), index=_dates(200))
        a = MicrostructureEngine.amihud_illiquidity(ret, dvol)
        assert a > 0

    def test_turnover(self):
        vol = pd.Series([50000.0] * 50)
        assert MicrostructureEngine.turnover_ratio(vol, 1e6) == pytest.approx(0.05)

    def test_turnover_zero_shares(self):
        assert MicrostructureEngine.turnover_ratio(pd.Series([100.0]), 0.0) == 0.0

    def test_full_liquidity(self):
        me = MicrostructureEngine()
        p = _prices(200)
        v = _volume(200)
        r = p.pct_change().dropna()
        lq = me.compute_liquidity(r, v.iloc[1:], p.iloc[1:], shares_outstanding=1e7)
        assert isinstance(lq, LiquidityMetrics)
        assert lq.avg_daily_volume > 0


# ===========================================================================
# Full analyze
# ===========================================================================

class TestAnalyze:
    def test_summary(self):
        me = MicrostructureEngine(n_vpin_buckets=10, impact_lag=3)
        p = _prices(200)
        v = _volume(200)
        summary = me.analyze(p, v)
        assert isinstance(summary, MicrostructureSummary)
        assert summary.spreads.roll_spread >= 0
        assert summary.trade_class.n_trades == 200

    def test_with_quotes(self):
        me = MicrostructureEngine(n_vpin_buckets=10)
        bid, ask = _bid_ask(100)
        mid = (bid + ask) / 2
        v = _volume(100)
        summary = me.analyze(mid, v, bid=bid, ask=ask)
        assert summary.spreads.quoted_spread > 0


# ===========================================================================
# HTML report
# ===========================================================================

class TestReport:
    def test_creates_file(self, tmp_path):
        me = MicrostructureEngine(n_vpin_buckets=10)
        p = _prices(100)
        v = _volume(100)
        summary = me.analyze(p, v)
        out = tmp_path / "micro.html"
        result = me.generate_report(summary, output_path=str(out))
        assert Path(result).exists()
        html = out.read_text()
        assert "Microstructure" in html

    def test_all_sections(self, tmp_path):
        me = MicrostructureEngine(n_vpin_buckets=10)
        p = _prices(100)
        v = _volume(100)
        bid, ask = _bid_ask(100)
        summary = me.analyze(p, v, bid=bid, ask=ask)
        out = tmp_path / "full.html"
        me.generate_report(summary, rolling_spreads=[0.03] * 50,
                            output_path=str(out))
        html = out.read_text()
        for section in ["Spread", "Order Flow", "Price Impact",
                         "Lee-Ready", "PIN", "Liquidity"]:
            assert section in html

    def test_with_charts(self, tmp_path):
        me = MicrostructureEngine(n_vpin_buckets=10)
        p = _prices(100)
        v = _volume(100)
        summary = me.analyze(p, v)
        out = tmp_path / "chart.html"
        me.generate_report(summary, rolling_spreads=[0.03 + i * 0.001 for i in range(30)],
                            output_path=str(out))
        html = out.read_text()
        assert "<svg" in html
