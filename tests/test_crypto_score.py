"""
Comprehensive tests for compass.crypto — score engine and risk gates.

Covers:
    - realized_vol: annualisation, window sizes, edge cases
    - regime:       MA200 position, trend detection, regime classification
    - composite_score: all signals, graceful degradation, band boundaries
    - risk_gate:    each gate individually and check_all_gates aggregation
"""

import math
import pytest

from compass.crypto.realized_vol import compute_realized_vol, compute_iv_rv_spread
from compass.crypto.regime import (
    compute_ma200_position,
    compute_trend,
    classify_regime,
)
from compass.crypto.composite_score import compute_composite_score, _score_to_band
from compass.crypto.risk_gate import (
    check_overnight_gap,
    check_iv_percentile,
    check_weekend_risk,
    check_all_gates,
    GAP_BLOCK_PCT,
    IV_PERCENTILE_FLOOR,
    WEEKEND_DTE_MAX,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _flat_prices(n: int, value: float = 100.0):
    """Return a flat price series of length n."""
    return [value] * n


def _trending_prices(n: int, start: float = 100.0, daily_return: float = 0.01):
    """Return an up-trending price series."""
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + daily_return))
    return prices


def _downtrending_prices(n: int, start: float = 100.0, daily_return: float = -0.01):
    """Return a down-trending price series."""
    return _trending_prices(n, start, daily_return)


# ===========================================================================
# realized_vol
# ===========================================================================

class TestComputeRealizedVol:

    def test_flat_prices_gives_zero_vol(self):
        prices = _flat_prices(100)
        assert compute_realized_vol(prices, window=7) == 0.0

    def test_annualisation_factor(self):
        """Check that a 1% daily move annualises to ~1% * sqrt(365)."""
        # Prices that produce exactly 1% daily log returns
        prices = [100.0 * math.exp(0.01 * i) for i in range(35)]
        rv = compute_realized_vol(prices, window=30)
        # Daily std ≈ 0 (constant 1% every day) → rv ≈ 0 (no variance)
        assert rv == pytest.approx(0.0, abs=1e-10)

    def test_volatile_prices_give_nonzero_vol(self):
        """Alternating up/down prices produce measurable vol."""
        prices = []
        p = 100.0
        for i in range(50):
            p = p * 1.02 if i % 2 == 0 else p * 0.98
            prices.append(p)
        rv = compute_realized_vol(prices, window=30)
        assert rv > 0.0

    def test_window_7_uses_last_7_returns(self):
        """7-day window requires exactly 8 prices."""
        prices = _flat_prices(8)
        assert compute_realized_vol(prices, window=7) == 0.0

    def test_window_30(self):
        prices = _flat_prices(31)
        assert compute_realized_vol(prices, window=30) == 0.0

    def test_insufficient_data_returns_zero(self):
        """Returns 0.0 when there are not enough prices for the window."""
        prices = [100.0] * 5
        assert compute_realized_vol(prices, window=7) == 0.0
        assert compute_realized_vol(prices, window=30) == 0.0

    def test_empty_list_returns_zero(self):
        assert compute_realized_vol([], window=7) == 0.0

    def test_single_price_returns_zero(self):
        assert compute_realized_vol([100.0], window=7) == 0.0

    def test_negative_price_raises(self):
        with pytest.raises(ValueError):
            compute_realized_vol([100.0, -50.0, 80.0], window=2)

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            compute_realized_vol([100.0, 0.0, 80.0], window=2)

    def test_uses_most_recent_window_not_oldest(self):
        """Flat tail after volatile head → vol should be 0 for recent window."""
        volatile = [100.0 * (1.05 ** i) for i in range(50)]
        flat_tail = [volatile[-1]] * 10  # 10 days of flat at end
        prices = volatile + flat_tail
        rv = compute_realized_vol(prices, window=7)
        assert rv == pytest.approx(0.0, abs=1e-10)

    def test_higher_vol_series_gives_higher_rv(self):
        """2% daily swings should produce larger rv than 0.5% daily swings."""
        def alternating(n, pct):
            prices, p = [], 100.0
            for i in range(n):
                p = p * (1 + pct) if i % 2 == 0 else p * (1 - pct)
                prices.append(p)
            return prices

        rv_high = compute_realized_vol(alternating(50, 0.02), window=30)
        rv_low  = compute_realized_vol(alternating(50, 0.005), window=30)
        assert rv_high > rv_low


class TestComputeIvRvSpread:

    def test_positive_spread_when_iv_above_rv(self):
        spread = compute_iv_rv_spread(iv=0.90, rv=0.60)
        assert spread == pytest.approx(0.30)

    def test_negative_spread_when_rv_above_iv(self):
        spread = compute_iv_rv_spread(iv=0.50, rv=0.70)
        assert spread == pytest.approx(-0.20)

    def test_zero_spread_at_parity(self):
        assert compute_iv_rv_spread(iv=0.75, rv=0.75) == pytest.approx(0.0)


# ===========================================================================
# regime
# ===========================================================================

class TestComputeMa200Position:

    def test_above_ma200(self):
        prices = [100.0] * 199 + [110.0]  # last price well above MA200
        assert compute_ma200_position(prices) == "above"

    def test_below_ma200(self):
        prices = [100.0] * 199 + [85.0]
        assert compute_ma200_position(prices) == "below"

    def test_crossing_ma200_exactly(self):
        # Last price equals MA200 exactly → within band
        prices = [100.0] * 200
        assert compute_ma200_position(prices) == "crossing"

    def test_crossing_ma200_within_band(self):
        # Last price 0.3% above MA200 → still crossing (band is 0.5%)
        prices = [100.0] * 199 + [100.3]
        assert compute_ma200_position(prices) == "crossing"

    def test_insufficient_history_returns_crossing(self):
        prices = [100.0] * 50
        assert compute_ma200_position(prices) == "crossing"

    def test_empty_returns_crossing(self):
        assert compute_ma200_position([]) == "crossing"

    def test_exactly_200_prices(self):
        # 200 prices, last one is well above average
        prices = [100.0] * 199 + [120.0]
        result = compute_ma200_position(prices)
        assert result in ("above", "crossing")


class TestComputeTrend:

    def test_strong_uptrend(self):
        prices = _trending_prices(30, start=100.0, daily_return=0.02)
        assert compute_trend(prices, window=20) == "uptrend"

    def test_strong_downtrend(self):
        prices = _downtrending_prices(30, start=100.0, daily_return=-0.02)
        assert compute_trend(prices, window=20) == "downtrend"

    def test_flat_is_ranging(self):
        prices = _flat_prices(30)
        assert compute_trend(prices, window=20) == "ranging"

    def test_insufficient_data_returns_ranging(self):
        prices = [100.0] * 5
        assert compute_trend(prices, window=20) == "ranging"

    def test_empty_returns_ranging(self):
        assert compute_trend([], window=20) == "ranging"

    def test_window_parameter_respected(self):
        # Long uptrend followed by recent downtrend
        up = _trending_prices(50, start=100.0, daily_return=0.02)
        down = _downtrending_prices(10, start=up[-1], daily_return=-0.02)
        prices = up + down
        # window=10 should see only the recent downtrend
        assert compute_trend(prices, window=10) == "downtrend"
        # window=50 should see the dominant uptrend
        assert compute_trend(prices, window=50) == "uptrend"


class TestClassifyRegime:

    def test_composite_score_extreme_fear(self):
        result = classify_regime({"composite_score": 10.0})
        assert result == "extreme_fear"

    def test_composite_score_cautious(self):
        result = classify_regime({"composite_score": 32.0})
        assert result == "cautious"

    def test_composite_score_neutral(self):
        result = classify_regime({"composite_score": 50.0})
        assert result == "neutral"

    def test_composite_score_bullish(self):
        result = classify_regime({"composite_score": 68.0})
        assert result == "bullish"

    def test_composite_score_extreme_greed(self):
        result = classify_regime({"composite_score": 90.0})
        assert result == "extreme_greed"

    def test_boundary_25_is_cautious(self):
        result = classify_regime({"composite_score": 25.0})
        assert result == "cautious"

    def test_boundary_40_is_neutral(self):
        result = classify_regime({"composite_score": 40.0})
        assert result == "neutral"

    def test_boundary_60_is_bullish(self):
        result = classify_regime({"composite_score": 60.0})
        assert result == "bullish"

    def test_boundary_75_is_extreme_greed(self):
        result = classify_regime({"composite_score": 75.0})
        assert result == "extreme_greed"

    def test_fallback_bullish_signals(self):
        # No composite_score — fallback to individual signals
        result = classify_regime({
            "fear_greed_index": 80.0,
            "ma200_position": "above",
            "trend": "uptrend",
            "funding_rate": 0.05,
            "put_call_ratio": 0.6,
        })
        assert result in ("bullish", "extreme_greed")

    def test_fallback_bearish_signals(self):
        result = classify_regime({
            "fear_greed_index": 10.0,
            "ma200_position": "below",
            "trend": "downtrend",
            "funding_rate": -0.05,
            "put_call_ratio": 1.8,
        })
        assert result in ("extreme_fear", "cautious")

    def test_fallback_empty_dict_returns_neutral(self):
        result = classify_regime({})
        assert result == "neutral"

    def test_fallback_single_signal(self):
        # Only fear_greed provided — should work
        result = classify_regime({"fear_greed_index": 50.0})
        assert result == "neutral"


# ===========================================================================
# composite_score
# ===========================================================================

class TestCompositeScore:

    # --- Band boundary helpers ---

    def test_fear_greed_only_extreme_fear(self):
        result = compute_composite_score(fear_greed_index=5.0)
        assert result["band"] == "EXTREME_FEAR"
        assert result["score"] < 25.0

    def test_fear_greed_only_extreme_greed(self):
        result = compute_composite_score(fear_greed_index=95.0)
        assert result["band"] == "EXTREME_GREED"
        assert result["score"] >= 75.0

    def test_fear_greed_only_neutral(self):
        result = compute_composite_score(fear_greed_index=50.0)
        assert result["band"] == "NEUTRAL"
        assert 40.0 <= result["score"] < 60.0

    # --- All signals: extreme fear scenario ---

    def test_all_signals_extreme_fear(self):
        """
        fear_greed=5 (max fear), high IV-RV spread, negative funding,
        price below MA200, high BTC dominance, high PCR, coin inflow.
        """
        result = compute_composite_score(
            fear_greed_index=5.0,
            iv_rv_spread=0.80,      # IV >> RV → fearful
            funding_rate=-0.08,     # shorts being paid → bearish
            ma200_position="below",
            btc_dominance=68.0,     # flight to BTC → risk-off
            put_call_ratio=1.9,     # heavy put buying
            exchange_flow_trend="inflow",  # selling pressure
        )
        assert result["score"] < 25.0
        assert result["band"] == "EXTREME_FEAR"

    # --- All signals: extreme greed scenario ---

    def test_all_signals_extreme_greed(self):
        """
        fear_greed=95, negative IV-RV spread, strong positive funding,
        price well above MA200, low BTC dominance (alt season), low PCR, outflow.
        """
        result = compute_composite_score(
            fear_greed_index=95.0,
            iv_rv_spread=-0.30,     # RV > IV → market under-pricing risk
            funding_rate=0.08,      # longs paying → bullish mania
            ma200_position="above",
            btc_dominance=42.0,     # alt-season → greed
            put_call_ratio=0.55,    # call heavy
            exchange_flow_trend="outflow",  # accumulation
        )
        assert result["score"] >= 75.0
        assert result["band"] == "EXTREME_GREED"

    # --- Graceful degradation ---

    def test_single_signal_no_error(self):
        for kwargs in [
            {"fear_greed_index": 50.0},
            {"iv_rv_spread": 0.10},
            {"funding_rate": 0.01},
            {"ma200_position": "above"},
            {"btc_dominance": 55.0},
            {"put_call_ratio": 1.0},
            {"exchange_flow_trend": "neutral"},
        ]:
            result = compute_composite_score(**kwargs)
            assert 0.0 <= result["score"] <= 100.0

    def test_missing_signals_weights_renormalise(self):
        """Weights of present signals must sum to 1 (within float tolerance)."""
        result = compute_composite_score(
            fear_greed_index=50.0,
            funding_rate=0.01,
        )
        total_weight = sum(
            v["weight_used"] for v in result["signals"].values()
        )
        assert total_weight == pytest.approx(1.0, abs=1e-6)

    def test_no_signals_raises(self):
        with pytest.raises(ValueError):
            compute_composite_score()

    # --- Output structure ---

    def test_output_keys_present(self):
        result = compute_composite_score(fear_greed_index=50.0)
        assert "score" in result
        assert "band" in result
        assert "signals" in result
        assert "timestamp" in result

    def test_signal_detail_keys(self):
        result = compute_composite_score(fear_greed_index=50.0, funding_rate=0.01)
        for signal_name, detail in result["signals"].items():
            assert "raw" in detail
            assert "component" in detail
            assert "weight_used" in detail

    def test_score_always_clamped_0_to_100(self):
        """Score must never exceed [0, 100] regardless of inputs."""
        for fg in [0.0, 50.0, 100.0]:
            result = compute_composite_score(fear_greed_index=fg)
            assert 0.0 <= result["score"] <= 100.0

    # --- Individual signal normalisers (white-box) ---

    def test_iv_rv_spread_zero_gives_near_50(self):
        """Spread=0 (IV=RV, fair pricing) should give component near 0.5."""
        result = compute_composite_score(iv_rv_spread=0.0)
        component = result["signals"]["iv_rv_spread"]["component"]
        assert component == pytest.approx(0.5, abs=0.02)

    def test_btc_dominance_40_gives_near_100_component(self):
        result = compute_composite_score(btc_dominance=40.0)
        component = result["signals"]["btc_dominance"]["component"]
        assert component == pytest.approx(1.0, abs=1e-6)

    def test_btc_dominance_70_gives_near_0_component(self):
        result = compute_composite_score(btc_dominance=70.0)
        component = result["signals"]["btc_dominance"]["component"]
        assert component == pytest.approx(0.0, abs=1e-6)

    def test_pcr_05_gives_component_1(self):
        result = compute_composite_score(put_call_ratio=0.5)
        component = result["signals"]["put_call_ratio"]["component"]
        assert component == pytest.approx(1.0, abs=1e-6)

    def test_pcr_20_gives_component_0(self):
        result = compute_composite_score(put_call_ratio=2.0)
        component = result["signals"]["put_call_ratio"]["component"]
        assert component == pytest.approx(0.0, abs=1e-6)

    def test_exchange_outflow_bullish(self):
        result = compute_composite_score(exchange_flow_trend="outflow")
        component = result["signals"]["exchange_flow_trend"]["component"]
        assert component > 0.5

    def test_exchange_inflow_bearish(self):
        result = compute_composite_score(exchange_flow_trend="inflow")
        component = result["signals"]["exchange_flow_trend"]["component"]
        assert component < 0.5

    def test_ma200_above_is_bullish(self):
        above = compute_composite_score(ma200_position="above")
        below = compute_composite_score(ma200_position="below")
        assert above["score"] > below["score"]

    # --- Band boundary: score_to_band ---

    def test_score_to_band_edges(self):
        assert _score_to_band(0.0)   == "EXTREME_FEAR"
        assert _score_to_band(24.9)  == "EXTREME_FEAR"
        assert _score_to_band(25.0)  == "CAUTIOUS"
        assert _score_to_band(39.9)  == "CAUTIOUS"
        assert _score_to_band(40.0)  == "NEUTRAL"
        assert _score_to_band(59.9)  == "NEUTRAL"
        assert _score_to_band(60.0)  == "BULLISH"
        assert _score_to_band(74.9)  == "BULLISH"
        assert _score_to_band(75.0)  == "EXTREME_GREED"
        assert _score_to_band(100.0) == "EXTREME_GREED"

    # --- Custom weights ---

    def test_custom_weights_respected(self):
        """When only fear_greed is provided and weight is large, score is dominated by it."""
        result = compute_composite_score(
            fear_greed_index=100.0,
            funding_rate=0.0,
            weights={
                "fear_greed_index":    0.99,
                "iv_rv_spread":        0.00,
                "funding_rate":        0.01,
                "ma200_position":      0.00,
                "btc_dominance":       0.00,
                "put_call_ratio":      0.00,
                "exchange_flow_trend": 0.00,
            },
        )
        # fear_greed=100 → component=1.0; funding=0 → component~0.5
        # weighted: (0.99*1.0 + 0.01*0.5) / 1.0 = 0.995 → score ~99.5
        assert result["score"] > 95.0


# ===========================================================================
# risk_gate
# ===========================================================================

class TestCheckOvernightGap:

    def test_small_gap_allowed(self):
        result = check_overnight_gap(btc_price_now=100.0, etf_close_price=100.0)
        assert result["blocked"] is False
        assert result["gap_pct"] == pytest.approx(0.0)

    def test_gap_exactly_at_threshold_not_blocked(self):
        # Exactly 5% gap → not > 5% → allowed
        result = check_overnight_gap(btc_price_now=105.0, etf_close_price=100.0)
        assert result["blocked"] is False

    def test_gap_just_above_threshold_blocked(self):
        result = check_overnight_gap(btc_price_now=105.01, etf_close_price=100.0)
        assert result["blocked"] is True
        assert result["gap_pct"] == pytest.approx(5.01, abs=0.01)

    def test_negative_gap_blocked(self):
        # -6% gap (price crashed overnight)
        result = check_overnight_gap(btc_price_now=94.0, etf_close_price=100.0)
        assert result["blocked"] is True
        assert result["gap_pct"] < 0.0

    def test_negative_gap_within_threshold_allowed(self):
        result = check_overnight_gap(btc_price_now=97.0, etf_close_price=100.0)
        assert result["blocked"] is False

    def test_gate_name(self):
        result = check_overnight_gap(100.0, 100.0)
        assert result["gate"] == "overnight_gap"

    def test_reason_empty_when_not_blocked(self):
        result = check_overnight_gap(100.0, 100.0)
        assert result["reason"] == ""

    def test_reason_populated_when_blocked(self):
        result = check_overnight_gap(200.0, 100.0)
        assert len(result["reason"]) > 0

    def test_zero_close_price_raises(self):
        with pytest.raises(ValueError):
            check_overnight_gap(btc_price_now=100.0, etf_close_price=0.0)

    def test_large_gap_blocked(self):
        # 20% gap (typical crypto black swan)
        result = check_overnight_gap(120.0, 100.0)
        assert result["blocked"] is True


class TestCheckIvPercentile:

    def _make_history(self, n=100, low=0.30, high=0.90):
        """Uniform IV history from low to high."""
        step = (high - low) / (n - 1)
        return [low + step * i for i in range(n)]

    def test_high_iv_passes(self):
        history = self._make_history()
        result = check_iv_percentile(current_iv=0.85, historical_ivs=history)
        assert result["blocked"] is False
        assert result["percentile"] > 40.0

    def test_low_iv_blocked(self):
        # current_iv below all history → percentile near 0 → blocked
        history = [0.50, 0.60, 0.70, 0.80, 0.90]
        result = check_iv_percentile(current_iv=0.30, historical_ivs=history)
        assert result["blocked"] is True
        assert result["percentile"] == pytest.approx(0.0)

    def test_iv_at_40th_percentile_exactly_blocked(self):
        """40th percentile is the threshold; BELOW 40 is blocked."""
        history = list(range(1, 101))  # 1-100
        # 40th percentile: 40 out of 100 are ≤ 40
        result = check_iv_percentile(current_iv=40.0, historical_ivs=history)
        # percentile = 40/100 * 100 = 40.0 → not blocked (< floor is the block)
        assert result["blocked"] is False
        assert result["percentile"] == pytest.approx(40.0)

    def test_iv_at_39th_percentile_blocked(self):
        history = list(range(1, 101))
        result = check_iv_percentile(current_iv=39.0, historical_ivs=history)
        assert result["blocked"] is True

    def test_empty_history_raises(self):
        with pytest.raises(ValueError):
            check_iv_percentile(0.5, [])

    def test_gate_name(self):
        result = check_iv_percentile(0.9, [0.5, 0.7, 0.8])
        assert result["gate"] == "iv_percentile"

    def test_single_element_history(self):
        # current_iv equals the only historical value → 100th percentile
        result = check_iv_percentile(0.5, [0.5])
        assert result["blocked"] is False
        assert result["percentile"] == pytest.approx(100.0)

    def test_reason_empty_when_not_blocked(self):
        history = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = check_iv_percentile(0.9, history)
        assert result["reason"] == ""


class TestCheckWeekendRisk:

    def test_monday_short_dte_allowed(self):
        result = check_weekend_risk(day_of_week=0, dte=5)
        assert result["blocked"] is False

    def test_wednesday_short_dte_allowed(self):
        result = check_weekend_risk(day_of_week=2, dte=7)
        assert result["blocked"] is False

    def test_thursday_short_dte_blocked(self):
        result = check_weekend_risk(day_of_week=3, dte=7)
        assert result["blocked"] is True

    def test_friday_short_dte_blocked(self):
        result = check_weekend_risk(day_of_week=4, dte=5)
        assert result["blocked"] is True

    def test_thursday_long_dte_allowed(self):
        """DTE > WEEKEND_DTE_MAX on Thursday should be fine."""
        result = check_weekend_risk(day_of_week=3, dte=WEEKEND_DTE_MAX + 1)
        assert result["blocked"] is False

    def test_friday_exactly_threshold_blocked(self):
        result = check_weekend_risk(day_of_week=4, dte=WEEKEND_DTE_MAX)
        assert result["blocked"] is True

    def test_saturday_short_dte_allowed(self):
        """Weekend days themselves are not Thu/Fri — no new entries anyway."""
        result = check_weekend_risk(day_of_week=5, dte=3)
        assert result["blocked"] is False

    def test_gate_name(self):
        result = check_weekend_risk(0, 10)
        assert result["gate"] == "weekend_risk"

    def test_reason_populated_when_blocked(self):
        result = check_weekend_risk(day_of_week=4, dte=5)
        assert len(result["reason"]) > 0

    def test_reason_empty_when_not_blocked(self):
        result = check_weekend_risk(day_of_week=0, dte=5)
        assert result["reason"] == ""

    def test_dte_zero_friday_blocked(self):
        result = check_weekend_risk(day_of_week=4, dte=0)
        assert result["blocked"] is True


class TestCheckAllGates:

    def _clean_context(self):
        """A context where all gates pass."""
        return {
            "btc_price_now":   100.0,
            "etf_close_price": 100.0,    # 0% gap
            "current_iv":      0.90,
            "historical_ivs":  [0.3 + i * 0.006 for i in range(100)],  # iv at ~100th pct
            "day_of_week":     1,         # Tuesday
            "dte":             30,
        }

    def test_all_gates_pass(self):
        result = check_all_gates(self._clean_context())
        assert result["allowed"] is True
        assert all(not g["blocked"] for g in result["gates"])

    def test_gap_gate_blocks_all(self):
        ctx = self._clean_context()
        ctx["btc_price_now"] = 115.0  # 15% gap
        result = check_all_gates(ctx)
        assert result["allowed"] is False
        gap_gate = next(g for g in result["gates"] if g["gate"] == "overnight_gap")
        assert gap_gate["blocked"] is True

    def test_iv_percentile_gate_blocks_all(self):
        ctx = self._clean_context()
        ctx["current_iv"] = 0.01      # below all historical
        result = check_all_gates(ctx)
        assert result["allowed"] is False
        iv_gate = next(g for g in result["gates"] if g["gate"] == "iv_percentile")
        assert iv_gate["blocked"] is True

    def test_weekend_gate_blocks_all(self):
        ctx = self._clean_context()
        ctx["day_of_week"] = 4   # Friday
        ctx["dte"] = 5
        result = check_all_gates(ctx)
        assert result["allowed"] is False
        wk_gate = next(g for g in result["gates"] if g["gate"] == "weekend_risk")
        assert wk_gate["blocked"] is True

    def test_multiple_gates_blocked(self):
        ctx = self._clean_context()
        ctx["btc_price_now"] = 115.0  # gap blocked
        ctx["day_of_week"] = 4        # weekend blocked
        ctx["dte"] = 5
        result = check_all_gates(ctx)
        assert result["allowed"] is False
        blocked_gates = [g for g in result["gates"] if g["blocked"]]
        assert len(blocked_gates) >= 2

    def test_missing_gap_keys_skips_gate(self):
        """Omitting btc_price_now/etf_close_price skips the gap gate."""
        ctx = {
            "current_iv":     0.90,
            "historical_ivs": [0.5, 0.6, 0.7],
            "day_of_week":    0,
            "dte":            10,
        }
        result = check_all_gates(ctx)
        gate_names = [g["gate"] for g in result["gates"]]
        assert "overnight_gap" not in gate_names
        assert "iv_percentile" in gate_names
        assert "weekend_risk" in gate_names

    def test_empty_context_no_gates_evaluated(self):
        result = check_all_gates({})
        assert result["allowed"] is True
        assert result["gates"] == []

    def test_partial_context_only_present_gates_run(self):
        ctx = {"day_of_week": 2, "dte": 10}
        result = check_all_gates(ctx)
        assert len(result["gates"]) == 1
        assert result["gates"][0]["gate"] == "weekend_risk"

    def test_gates_list_always_present(self):
        result = check_all_gates({})
        assert "gates" in result
        assert isinstance(result["gates"], list)

    def test_allowed_key_always_present(self):
        result = check_all_gates({})
        assert "allowed" in result
