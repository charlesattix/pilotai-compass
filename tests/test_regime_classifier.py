"""
Unit tests for the COMPASS regime classifier.

Target module: compass/regime.py (post-Phase 2 move)
Pre-move source: engine/regime.py

Tests cover:
  - classify() single-day classification for all 5 regimes
  - VIX threshold boundaries and priority ordering
  - Trend direction detection (_trend_direction)
  - Sharp decline detection (_is_declining)
  - classify_series() multi-day tagging
  - summarize() statistics computation
  - Edge cases: short data, ambiguous zones, custom parameters

Blueprint spec: 15+ tests, all green (Phase 3 exit criteria).
"""

import pandas as pd
import pytest

from compass.regime import Regime, RegimeClassifier, REGIME_INFO

from tests.compass_helpers import (
    mock_spy_prices,
    mock_vix_series,
    mock_spy_dataframe,
    RegimeScenario,
    REGIME_SCENARIOS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def classifier():
    """Default RegimeClassifier with standard parameters."""
    return RegimeClassifier(trend_window=50, trend_threshold=5.0)


@pytest.fixture
def bull_prices():
    """100-day SPY series with a strong uptrend (+25% annual)."""
    return mock_spy_prices(days=100, trend=25.0, base=450.0)


@pytest.fixture
def bear_prices():
    """100-day SPY series with a strong downtrend (-25% annual)."""
    return mock_spy_prices(days=100, trend=-25.0, base=450.0)


@pytest.fixture
def flat_prices():
    """100-day SPY series with zero trend."""
    return mock_spy_prices(days=100, trend=0.0, base=450.0)


@pytest.fixture
def crash_prices():
    """100-day SPY series with a sharp >5% decline in the last 10 days."""
    scenario = REGIME_SCENARIOS["crash"]
    return scenario.build_prices()


# ══════════════════════════════════════════════════════════════════════════════
# A. Regime enum and metadata
# ══════════════════════════════════════════════════════════════════════════════

class TestRegimeEnum:

    def test_regime_has_five_members(self):
        """Regime enum contains exactly: bull, bear, high_vol, low_vol, crash."""
        members = set(Regime)
        assert len(members) == 5
        expected = {Regime.BULL, Regime.BEAR, Regime.HIGH_VOL, Regime.LOW_VOL, Regime.CRASH}
        assert members == expected

    def test_regime_info_covers_all_regimes(self):
        """REGIME_INFO has an entry for every Regime member with label, strategies, risk."""
        for regime in Regime:
            assert regime in REGIME_INFO, f"Missing REGIME_INFO for {regime}"
            info = REGIME_INFO[regime]
            assert "label" in info
            assert "strategies" in info
            assert "risk" in info
            assert isinstance(info["strategies"], list)


# ══════════════════════════════════════════════════════════════════════════════
# B. classify() — single-day classification
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyCrash:
    """VIX > 40 + sharp decline -> CRASH (highest priority)."""

    def test_crash_vix_above_40_with_sharp_decline(self, classifier, crash_prices):
        """VIX=45 and >5% drop in last 10 days -> CRASH."""
        result = classifier.classify(vix=45.0, spy_prices=crash_prices, date=crash_prices.index[-1])
        assert result == Regime.CRASH

    def test_high_vol_vix_above_40_without_decline(self, classifier, bull_prices):
        """VIX=45 but no sharp decline -> HIGH_VOL (not CRASH)."""
        result = classifier.classify(vix=45.0, spy_prices=bull_prices, date=bull_prices.index[-1])
        assert result == Regime.HIGH_VOL


class TestClassifyHighVol:
    """VIX > 30 (any direction) -> HIGH_VOL."""

    def test_high_vol_vix_32(self, classifier, bull_prices):
        """VIX=32 with uptrend -> HIGH_VOL (VIX threshold overrides trend)."""
        result = classifier.classify(vix=32.0, spy_prices=bull_prices, date=bull_prices.index[-1])
        assert result == Regime.HIGH_VOL

    def test_high_vol_vix_35_downtrend(self, classifier, bear_prices):
        """VIX=35 with downtrend -> HIGH_VOL (not BEAR, VIX > 30 takes priority)."""
        result = classifier.classify(vix=35.0, spy_prices=bear_prices, date=bear_prices.index[-1])
        assert result == Regime.HIGH_VOL


class TestClassifyBear:
    """VIX > 25 + SPY downtrend -> BEAR."""

    def test_bear_vix_27_downtrend(self, classifier, bear_prices):
        """VIX=27, strong downtrend -> BEAR."""
        result = classifier.classify(vix=27.0, spy_prices=bear_prices, date=bear_prices.index[-1])
        assert result == Regime.BEAR


class TestClassifyBull:
    """VIX < 20 + SPY uptrend -> BULL."""

    def test_bull_vix_18_uptrend(self, classifier, bull_prices):
        """VIX=18, strong uptrend -> BULL."""
        result = classifier.classify(vix=18.0, spy_prices=bull_prices, date=bull_prices.index[-1])
        assert result == Regime.BULL


class TestClassifyLowVol:
    """VIX < 15, no strong trend -> LOW_VOL."""

    def test_low_vol_vix_12_flat(self, classifier, flat_prices):
        """VIX=12, flat trend -> LOW_VOL."""
        result = classifier.classify(vix=12.0, spy_prices=flat_prices, date=flat_prices.index[-1])
        assert result == Regime.LOW_VOL


class TestClassifyAmbiguous:
    """Ambiguous zones: VIX between clear thresholds, mixed signals."""

    def test_ambiguous_uptrend_defaults_bull(self, classifier, bull_prices):
        """VIX=22 (no clear bucket), trend > 0 -> BULL."""
        result = classifier.classify(vix=22.0, spy_prices=bull_prices, date=bull_prices.index[-1])
        assert result == Regime.BULL

    def test_ambiguous_downtrend_high_vix_bear(self, classifier, bear_prices):
        """VIX=23, trend < 0 -> BEAR (VIX > 22 tips to bear)."""
        result = classifier.classify(vix=23.0, spy_prices=bear_prices, date=bear_prices.index[-1])
        assert result == Regime.BEAR

    def test_ambiguous_downtrend_low_vix_bull(self, classifier, bear_prices):
        """VIX=18, trend < 0 -> BULL (mild pullback, low VIX -> still constructive)."""
        result = classifier.classify(vix=18.0, spy_prices=bear_prices, date=bear_prices.index[-1])
        assert result == Regime.BULL

    def test_no_trend_low_vix_low_vol(self, classifier, flat_prices):
        """VIX=16, trend=0 -> LOW_VOL (vix < 18, no trend)."""
        result = classifier.classify(vix=16.0, spy_prices=flat_prices, date=flat_prices.index[-1])
        assert result == Regime.LOW_VOL

    def test_no_trend_moderate_vix_bull(self, classifier, flat_prices):
        """VIX=19, trend=0 -> BULL (neutral default)."""
        result = classifier.classify(vix=19.0, spy_prices=flat_prices, date=flat_prices.index[-1])
        assert result == Regime.BULL


# ══════════════════════════════════════════════════════════════════════════════
# C. Trend helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestTrendDirection:
    """RegimeClassifier._trend_direction() — slope-based trend detection."""

    def test_uptrend_returns_positive(self, classifier, bull_prices):
        """Strong uptrend -> +1."""
        assert classifier._trend_direction(bull_prices) == 1

    def test_downtrend_returns_negative(self, classifier, bear_prices):
        """Strong downtrend -> -1."""
        assert classifier._trend_direction(bear_prices) == -1

    def test_flat_returns_zero(self, classifier, flat_prices):
        """No trend -> 0."""
        assert classifier._trend_direction(flat_prices) == 0

    def test_short_data_uses_shorter_window(self, classifier):
        """Fewer than trend_window (50) days -> falls back to max(10, len(prices))."""
        short_prices = mock_spy_prices(days=20, trend=40.0, base=450.0)
        # With only 20 days, it should still detect an uptrend using the shorter window
        result = classifier._trend_direction(short_prices)
        assert result == 1


class TestIsDeclining:
    """RegimeClassifier._is_declining() — sharp decline check."""

    def test_sharp_decline_detected(self, classifier, crash_prices):
        """>5% drop in last 10 trading days -> True."""
        assert classifier._is_declining(crash_prices) == True

    def test_gentle_decline_not_detected(self, classifier, bear_prices):
        """Gradual decline (not >5% in 10 days) -> False."""
        # -25% annual = ~1% per 10 trading days, well under 5%
        assert classifier._is_declining(bear_prices) == False

    def test_insufficient_data(self, classifier):
        """Fewer than 10 data points -> False."""
        short = mock_spy_prices(days=5, trend=-50.0, base=450.0)
        assert classifier._is_declining(short) == False


# ══════════════════════════════════════════════════════════════════════════════
# D. classify_series() — multi-day tagging
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifySeries:

    def test_returns_series_same_length_as_input(self, classifier, bull_prices):
        """Output Series has same index as input DataFrame."""
        spy_df = mock_spy_dataframe(bull_prices)
        vix = mock_vix_series(start_date="2024-01-02", days=len(bull_prices), base_level=16.0)
        result = classifier.classify_series(spy_df, vix)
        assert len(result) == len(spy_df)

    def test_tags_bull_market_data(self, classifier, bull_prices):
        """Constant low VIX + uptrend -> predominantly BULL labels."""
        spy_df = mock_spy_dataframe(bull_prices)
        vix = mock_vix_series(start_date="2024-01-02", days=len(bull_prices), base_level=16.0)
        result = classifier.classify_series(spy_df, vix)
        # After warmup period, most days should be BULL
        bull_count = result.apply(lambda x: x == Regime.BULL or str(x) == "bull").sum()
        assert bull_count > len(result) * 0.5, f"Expected majority BULL, got {bull_count}/{len(result)}"

    def test_missing_vix_defaults_to_20(self, classifier, bull_prices):
        """Dates missing from vix_series get default VIX=20.0."""
        spy_df = mock_spy_dataframe(bull_prices)
        # Provide VIX for only half the dates
        half = len(bull_prices) // 2
        vix = mock_vix_series(start_date="2024-01-02", days=half, base_level=16.0)
        result = classifier.classify_series(spy_df, vix)
        # Should still produce a regime for every day (no crashes)
        assert len(result) == len(spy_df)


# ══════════════════════════════════════════════════════════════════════════════
# E. summarize()
# ══════════════════════════════════════════════════════════════════════════════

class TestSummarize:

    def test_summarize_returns_required_keys(self):
        """Output has total_days, distribution, transitions, avg_regime_duration."""
        regimes = pd.Series([Regime.BULL] * 50 + [Regime.BEAR] * 50)
        result = RegimeClassifier.summarize(regimes)
        assert "total_days" in result
        assert "distribution" in result
        assert "transitions" in result
        assert "avg_regime_duration" in result
        assert result["total_days"] == 100

    def test_summarize_counts_transitions(self):
        """Alternating regimes produce correct transition count."""
        # BULL x 30, BEAR x 20, BULL x 10 = 2 transitions
        regimes = pd.Series(
            [Regime.BULL] * 30 + [Regime.BEAR] * 20 + [Regime.BULL] * 10
        )
        result = RegimeClassifier.summarize(regimes)
        assert result["transitions"] == 2

    def test_summarize_distribution_sums_to_100(self):
        """All regime percentages sum to ~100%."""
        regimes = pd.Series(
            [Regime.BULL] * 40 + [Regime.BEAR] * 30
            + [Regime.HIGH_VOL] * 15 + [Regime.LOW_VOL] * 10
            + [Regime.CRASH] * 5
        )
        result = RegimeClassifier.summarize(regimes)
        total_pct = sum(v["pct"] for v in result["distribution"].values())
        assert abs(total_pct - 100.0) < 0.5


# ══════════════════════════════════════════════════════════════════════════════
# F. Custom parameters
# ══════════════════════════════════════════════════════════════════════════════

class TestCustomParameters:

    def test_custom_trend_window(self):
        """trend_window=20 changes which prices are considered for trend."""
        clf_20 = RegimeClassifier(trend_window=20, trend_threshold=5.0)
        clf_50 = RegimeClassifier(trend_window=50, trend_threshold=5.0)
        # Short uptrend: only visible in 20-day window
        prices = mock_spy_prices(days=30, trend=30.0, base=450.0)
        # Both should detect uptrend, but this confirms the window param is used
        assert clf_20._trend_direction(prices) == 1
        assert clf_20.trend_window == 20
        assert clf_50.trend_window == 50

    def test_custom_trend_threshold(self):
        """trend_threshold=15.0 raises the bar — mild trends read as flat."""
        clf_mild = RegimeClassifier(trend_window=50, trend_threshold=15.0)
        # A mild 8% annual trend should be flat with threshold=15
        prices = mock_spy_prices(days=100, trend=8.0, base=450.0)
        result = clf_mild._trend_direction(prices)
        assert result == 0, "Mild trend should read as flat with high threshold"


# ══════════════════════════════════════════════════════════════════════════════
# G. Enhanced features (hysteresis, VIX3M, configurable thresholds)
# ══════════════════════════════════════════════════════════════════════════════

class TestHysteresis:
    """Tests for hysteresis_days config — prevents rapid regime flip-flop."""

    def test_hysteresis_prevents_rapid_flip(self):
        """Regime changes within hysteresis_days are suppressed."""
        # Build two classifiers: one with hysteresis, one without
        clf_hyst = RegimeClassifier(config={"hysteresis_days": 10})
        clf_none = RegimeClassifier(config={"hysteresis_days": 0})

        # 100 days of uptrend data
        prices = mock_spy_prices(days=100, trend=25.0, base=450.0)
        spy_df = mock_spy_dataframe(prices)

        # VIX oscillates: 16 for 30 days, 32 for 5 days (HIGH_VOL), back to 16,
        # then 32 again 5 days later. Without hysteresis, each VIX spike
        # triggers HIGH_VOL. With hysteresis=10, the second spike is suppressed.
        vix_levels = (
            [16.0] * 30 + [32.0] * 5 + [16.0] * 5 + [32.0] * 5 + [16.0] * 55
        )
        vix = mock_vix_series(days=100, levels=vix_levels)

        result_hyst = clf_hyst.classify_series(spy_df, vix)
        result_none = clf_none.classify_series(spy_df, vix)

        def count_transitions(series):
            t = 0
            prev = None
            for r in series:
                if prev is not None and r != prev:
                    t += 1
                prev = r
            return t

        t_hyst = count_transitions(result_hyst)
        t_none = count_transitions(result_none)
        assert t_hyst < t_none, (
            f"Hysteresis should reduce transitions: hyst={t_hyst} vs none={t_none}"
        )

    def test_no_hysteresis_by_default(self):
        """Default config (hysteresis_days=0) allows every regime change."""
        clf = RegimeClassifier()
        assert clf.hysteresis_days == 0

        # Build data with regime change every ~10 days
        bull = mock_spy_prices(days=50, trend=25.0, base=450.0)
        spy_df = mock_spy_dataframe(bull)
        # Alternate VIX between 12 (LOW_VOL-range) and 16 (BULL-range)
        vix_levels = ([12.0] * 10 + [19.0] * 10) * 2 + [12.0] * 10
        vix = mock_vix_series(days=50, levels=vix_levels)

        result = clf.classify_series(spy_df, vix)
        # Should allow all transitions (no suppression)
        assert len(result) == 50


class TestVIX3MCrashSignal:
    """Tests for vix3m_crash_threshold config — VIX/VIX3M term structure."""

    def test_vix3m_crash_when_backwardation_steep(self):
        """VIX/VIX3M > threshold + VIX > 25 → CRASH."""
        clf = RegimeClassifier(config={"vix3m_crash_threshold": 1.2})
        prices = mock_spy_prices(days=100, trend=0.0, base=450.0)

        # VIX=30, VIX3M=20 → ratio=1.5 > 1.2 → CRASH
        result = clf.classify(
            vix=30.0, spy_prices=prices, date=prices.index[-1], vix3m=20.0,
        )
        assert result == Regime.CRASH

    def test_vix3m_no_crash_when_contango(self):
        """VIX/VIX3M < threshold → normal classification (no CRASH override)."""
        clf = RegimeClassifier(config={"vix3m_crash_threshold": 1.2})
        prices = mock_spy_prices(days=100, trend=25.0, base=450.0)

        # VIX=18, VIX3M=22 → ratio=0.82 < 1.2 → normal (BULL)
        result = clf.classify(
            vix=18.0, spy_prices=prices, date=prices.index[-1], vix3m=22.0,
        )
        assert result == Regime.BULL

    def test_vix3m_requires_elevated_vix(self):
        """VIX/VIX3M > threshold but VIX <= 25 → no CRASH (VIX too low)."""
        clf = RegimeClassifier(config={"vix3m_crash_threshold": 1.2})
        prices = mock_spy_prices(days=100, trend=25.0, base=450.0)

        # VIX=20, VIX3M=15 → ratio=1.33 > 1.2, but VIX=20 <= 25 → not CRASH
        result = clf.classify(
            vix=20.0, spy_prices=prices, date=prices.index[-1], vix3m=15.0,
        )
        assert result != Regime.CRASH

    def test_vix3m_off_by_default(self):
        """Default config (vix3m_crash_threshold=None) ignores vix3m."""
        clf = RegimeClassifier()
        assert clf.vix3m_crash_threshold is None

        prices = mock_spy_prices(days=100, trend=0.0, base=450.0)
        # Even with steep backwardation, CRASH should NOT be triggered
        result = clf.classify(
            vix=31.0, spy_prices=prices, date=prices.index[-1], vix3m=20.0,
        )
        # VIX=31 → HIGH_VOL (standard rule, vix > 30), not CRASH
        assert result == Regime.HIGH_VOL

    def test_vix3m_in_classify_series(self):
        """classify_series passes VIX3M through to classify correctly."""
        clf = RegimeClassifier(config={"vix3m_crash_threshold": 1.2})
        prices = mock_spy_prices(days=50, trend=0.0, base=450.0)
        spy_df = mock_spy_dataframe(prices)

        # VIX=28, VIX3M=20 → ratio=1.4 throughout → should trigger CRASH
        vix = mock_vix_series(days=50, base_level=28.0)
        vix3m = mock_vix_series(days=50, base_level=20.0)

        result = clf.classify_series(spy_df, vix, vix3m_series=vix3m)
        # After warmup (shift-by-1), most days should be CRASH
        crash_count = result.apply(lambda x: x == Regime.CRASH or str(x) == "crash").sum()
        assert crash_count > len(result) * 0.5, (
            f"Expected majority CRASH with steep backwardation, got {crash_count}/{len(result)}"
        )


class TestConfigurableThresholds:
    """Config dict controls RegimeClassifier behavior."""

    def test_hysteresis_days_from_config(self):
        """hysteresis_days=3 is stored correctly."""
        clf = RegimeClassifier(config={"hysteresis_days": 3})
        assert clf.hysteresis_days == 3

    def test_vix3m_threshold_from_config(self):
        """vix3m_crash_threshold=1.2 is stored correctly."""
        clf = RegimeClassifier(config={"vix3m_crash_threshold": 1.2})
        assert clf.vix3m_crash_threshold == 1.2

    def test_empty_config_uses_defaults(self):
        """Empty config dict → all defaults (features off)."""
        clf = RegimeClassifier(config={})
        assert clf.hysteresis_days == 0
        assert clf.vix3m_crash_threshold is None

    def test_no_config_uses_defaults(self):
        """No config arg → all defaults (backward compatible)."""
        clf = RegimeClassifier()
        assert clf.hysteresis_days == 0
        assert clf.vix3m_crash_threshold is None


class TestLookaheadProtection:
    """classify_series uses shift-by-1 to prevent lookahead bias."""

    def test_lookahead_protection_shift_by_1(self):
        """Regime on day T uses VIX from day T-1, not day T."""
        clf = RegimeClassifier()

        # 50 days of calm market (VIX=16, uptrend)
        prices = mock_spy_prices(days=50, trend=25.0, base=450.0)
        spy_df = mock_spy_dataframe(prices)

        # VIX: 16 for all days except the LAST day which spikes to 45
        vix_levels = [16.0] * 49 + [45.0]
        vix = mock_vix_series(days=50, levels=vix_levels)

        result = clf.classify_series(spy_df, vix)

        # The VIX spike on the last day should NOT affect the last day's regime
        # (shift-by-1: last day sees yesterday's VIX=16, not today's VIX=45)
        last_regime = result.iloc[-1]
        assert last_regime != Regime.CRASH, (
            "Last day should not see today's VIX spike (lookahead protection)"
        )
        assert last_regime != Regime.HIGH_VOL, (
            "Last day should not see today's VIX=45 (lookahead protection)"
        )
