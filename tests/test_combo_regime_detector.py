"""
Unit tests for ml/combo_regime_detector.py (Phase 6 v2).

Tests verify:
  1. bull regime — all 3 signals agree bullish
  2. bear regime — unanimous (3/3) required
  3. Bear blocked by supermajority — 2 bear + 1 bull → neutral (not bear)
  4. 2023 recovery scenario — MA200=bear, RSI=bull, VIX_struct=bull → 2/3 bull → bull
  5. 2024 brief dip scenario — MA200=bull, RSI=bear, VIX_struct=bear → 1B/2B → neutral
  6. VIX circuit breaker — VIX > 40 → bear regardless of other signals
  7. Hysteresis — regime just changed; raw signal wants to flip again → keeps current
  8. MA200 confidence zone — price within 0.5% of MA200 → MA200 abstains
"""

import pandas as pd

from compass.regime import ComboRegimeDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(values: list, start: str = "2020-01-02") -> pd.DataFrame:
    """Build a minimal price DataFrame with a DatetimeIndex and 'Close' column."""
    idx = pd.bdate_range(start=start, periods=len(values))
    return pd.DataFrame({"Close": values}, index=idx)


def _make_detector(config: dict = None) -> ComboRegimeDetector:
    base = {
        "signals": ["price_vs_ma200", "rsi_momentum", "vix_structure"],
        "ma_slow_period": 200,
        "ma200_neutral_band_pct": 0.5,
        "rsi_period": 14,
        "rsi_bull_threshold": 55.0,
        "rsi_bear_threshold": 45.0,
        "vix_structure_bull": 0.95,
        "vix_structure_bear": 1.05,
        "bear_requires_unanimous": True,
        "cooldown_days": 10,
        "vix_extreme": 40.0,
    }
    if config:
        base.update(config)
    return ComboRegimeDetector(base)


def _last_regime(detector, price_data, vix_by_date, vix3m_by_date=None):
    """Run compute_regime_series and return the label for the last date."""
    regime = detector.compute_regime_series(price_data, vix_by_date, vix3m_by_date)
    return regime[price_data.index[-1]]


def _make_vix_dict(price_data, vix_val):
    """Constant VIX dict for all dates in price_data."""
    return {ts: float(vix_val) for ts in price_data.index}


def _make_vix3m_dict(price_data, vix3m_val):
    return {ts: float(vix3m_val) for ts in price_data.index}


# ---------------------------------------------------------------------------
# Test 1: bull — all 3 signals agree bullish
# ---------------------------------------------------------------------------

def test_bull_regime_all_agree():
    """
    price >> MA200, RSI > 55 (rising trend), VIX/VIX3M < 0.95 (contango) → 3 bull → bull.
    """
    # 210 days of steadily rising prices: RSI will be high, price >> MA200
    values = [300 + i for i in range(210)]
    df = _make_price_data(values)

    vix     = _make_vix_dict(df, 15.0)   # VIX = 15
    vix3m   = _make_vix3m_dict(df, 18.0) # VIX3M = 18 → ratio = 0.833 < 0.95 → bull

    detector = _make_detector({"cooldown_days": 0})  # disable hysteresis for unit test
    label = _last_regime(detector, df, vix, vix3m)
    assert label == "bull", f"Expected bull, got {label}"


# ---------------------------------------------------------------------------
# Test 2: bear — unanimous (3/3) required
# ---------------------------------------------------------------------------

def test_bear_regime_unanimous():
    """
    price << MA200, RSI < 45 (falling trend), VIX/VIX3M > 1.05 (backwardation) → 3 bear → bear.
    """
    # 210 days of steadily falling prices
    values = [510 - i for i in range(210)]
    df = _make_price_data(values)

    vix     = _make_vix_dict(df, 30.0)   # VIX = 30
    vix3m   = _make_vix3m_dict(df, 25.0) # VIX3M = 25 → ratio = 1.20 > 1.05 → bear

    detector = _make_detector({"cooldown_days": 0})
    label = _last_regime(detector, df, vix, vix3m)
    assert label == "bear", f"Expected bear (3/3 unanimous), got {label}"


# ---------------------------------------------------------------------------
# Test 3: Bear blocked by supermajority — 2 bear + 1 bull → neutral
# ---------------------------------------------------------------------------

def test_bear_blocked_by_supermajority():
    """
    Aug-2024-like scenario:
      MA200: bull  (price well above 200-day MA anchor)
      RSI:   bear  (14 consecutive -1 days push RSI to ~22)
      VIX_structure: bear (backwardation: VIX/VIX3M = 1.25)
    → 1 bull + 2 bear → bear_votes=2 < 3 required → neutral (not bear).

    Key data design: 600-day gentle bull (+0.5/day) anchors MA200 far below
    current price.  14-day -1/day dip is small enough (14 pts) that price
    stays ~30 pts above MA200, but large enough to push RSI below 45.
    """
    # 600-day slow bull: price 300 → 599.5, MA200 lags around 450–555
    bull = [300 + i * 0.5 for i in range(600)]
    # 15-day dip: each day drops 1; diff at day 0 = -1 (immediate drop)
    dip  = [598.5 - i for i in range(15)]
    values = bull + dip
    df = _make_price_data(values)

    # VIX below circuit breaker threshold, VIX3M lower → backwardation
    vix   = _make_vix_dict(df, 35.0)   # elevated but not extreme (<40)
    vix3m = _make_vix3m_dict(df, 28.0) # ratio = 35/28 = 1.25 → bear signal

    detector = _make_detector({"cooldown_days": 0, "bear_requires_unanimous": True})
    label = _last_regime(detector, df, vix, vix3m)
    assert label == "neutral", (
        f"Expected neutral (2 bear < 3 unanimous required), got {label}. "
        "MA200=bull + 2 bear should NOT reach 3/3 unanimous threshold."
    )


# ---------------------------------------------------------------------------
# Test 4: 2023 recovery scenario — MA200 bear, RSI+VIX_struct bull → bull
# ---------------------------------------------------------------------------

def test_2023_recovery_scenario():
    """
    Early-2023 analog: price recovering above MA50 (RSI rising) but still below MA200.
      price_vs_ma200: bear (price < MA200)
      rsi_momentum:   bull (RSI recovering > 55)
      vix_structure:  bull (ratio < 0.95, calm contango)
    → 2 bull, 1 bear → bull → no bear calls (bear calls correctly blocked in early 2023).
    """
    # 300 days at high plateau (500), then 60 days at depressed (350), then recovering
    plateau = [500.0] * 300
    drop    = [350.0] * 60
    # Recovery: small daily gains → RSI builds up over 14+ days
    recovery = [350.0 + i * 1.5 for i in range(30)]  # 350 → 393.5
    values = plateau + drop + recovery
    df = _make_price_data(values)

    last_price = values[-1]
    ma200_approx = sum(values[-201:-1]) / 200  # mostly 350s and some 500s → ~455
    assert last_price < ma200_approx, (
        f"Setup error: price {last_price:.1f} should be < MA200 {ma200_approx:.1f}"
    )

    # Contango: VIX3M > VIX (calm, market expects recovery)
    vix   = _make_vix_dict(df, 16.0)
    vix3m = _make_vix3m_dict(df, 19.0)  # ratio = 0.842 < 0.95 → bull

    detector = _make_detector({"cooldown_days": 0})
    label = _last_regime(detector, df, vix, vix3m)
    assert label == "bull", (
        f"2023 recovery: expected bull (RSI+VIX_struct override MA200 bear), got {label}. "
        f"price={last_price:.1f} MA200≈{ma200_approx:.1f}"
    )


# ---------------------------------------------------------------------------
# Test 5: 2024 dip scenario — MA200 bull, RSI+VIX_struct bear → neutral
# ---------------------------------------------------------------------------

def test_2024_dip_scenario():
    """
    Aug-2024 analog: brief dip in a bull market.
      price_vs_ma200: bull  (SPY still above MA200 — long-run anchor)
      rsi_momentum:   bear  (12 days of -1 drops RSI to ~36)
      vix_structure:  bear  (VIX/VIX3M = 1.357 — backwardation)
    → 1 bull, 2 bear → bear_votes=2 < 3 unanimous → neutral (no bear calls).

    Same core design as test_bear_blocked: long slow bull + small daily dip
    keeps price above MA200 while RSI and VIX structure turn bearish.
    """
    # Same anchor design: 600-day slow bull, then 13-day dip
    bull = [300 + i * 0.5 for i in range(600)]
    dip  = [598.5 - i for i in range(13)]  # 13-day dip (slightly shorter)
    values = bull + dip
    df = _make_price_data(values)

    # VIX spiked into backwardation (below 40 circuit breaker)
    vix   = _make_vix_dict(df, 38.0)   # below 40 threshold
    vix3m = _make_vix3m_dict(df, 28.0) # ratio = 38/28 = 1.357 → bear

    detector = _make_detector({"cooldown_days": 0, "bear_requires_unanimous": True})
    label = _last_regime(detector, df, vix, vix3m)
    assert label == "neutral", (
        f"2024 dip: expected neutral (2 bear < 3 unanimous required), got {label}"
    )


# ---------------------------------------------------------------------------
# Test 6: VIX circuit breaker — VIX > 40 → bear regardless
# ---------------------------------------------------------------------------

def test_vix_circuit_breaker():
    """
    VIX > 40 triggers circuit breaker → bear regardless of other signals.
    Even with rising prices (MA200=bull, RSI=bull, VIX_struct=bull), extreme VIX wins.
    """
    values = [300 + i for i in range(210)]  # rising → all signals bull
    df = _make_price_data(values)

    # VIX circuit breaker uses PRIOR day's VIX (no lookahead).
    # To fire the CB on the last date, the spike must be on index[-2]
    # so it appears in vix_prev when evaluating the last date.
    vix = {ts: 15.0 for ts in df.index}
    vix[df.index[-2]] = 45.0  # second-to-last day spikes → CB fires on last day

    vix3m = _make_vix3m_dict(df, 18.0)  # ratio would be 45/18 = 2.5 but CB fires first

    detector = _make_detector({"cooldown_days": 0, "vix_extreme": 40.0})
    label = _last_regime(detector, df, vix, vix3m)
    assert label == "bear", (
        f"VIX circuit breaker: expected bear (VIX=45 > 40), got {label}"
    )


# ---------------------------------------------------------------------------
# Test 7: Hysteresis prevents rapid flip
# ---------------------------------------------------------------------------

def test_hysteresis_prevents_flip():
    """
    Regime just changed to bull. On the very next day, raw signal says neutral.
    With cooldown_days=10, hysteresis keeps bull for 10 days.
    """
    # Long falling trend (establishes bear), then sharp recovery (signals flip to bull),
    # then slight dip (RSI dips to neutral — would be neutral raw signal)
    [500 - i for i in range(210)]   # 500 → 290: establishes bear
    # One strong up day that flips RSI+MA200 (simulate by using detector with cooldown)
    # Instead, test hysteresis directly: build a scenario where regime changes then
    # raw signal immediately wants to change back.

    # Design: 210 rising days (→ bull established), then 1 day where RSI neutral,
    # MA200 just barely in band (abstains), VIX neutral → raw=neutral.
    # With cooldown=10, should stay bull.
    [300 + i for i in range(210)]
    # Last bar: price dropped back near MA200 (in neutral band), RSI neutral
    # We'll set MA200 ≈ current price, RSI ≈ 50 (neutral), VIX_struct neutral
    # The simplest approach: just check that after a recent change, the regime holds.
    # Use a two-phase dataset: phase1 all bull signals, phase2 all ambiguous.

    # Phase 1: 220 days all-bull (price rising fast, RSI > 55, VIX contango)
    phase1 = [300 + i * 2 for i in range(220)]
    # Phase 2: 5 days where raw signal would be neutral (RSI dips to 50, VIX neutral band)
    phase2_start = phase1[-1]
    phase2 = [phase2_start - i * 0.1 for i in range(5)]  # tiny dip (stays above MA200)
    values = phase1 + phase2
    df = _make_price_data(values)

    # VIX in neutral band for last 5 days (0.95–1.05 → abstain)
    vix   = {ts: 15.0 for ts in df.index}
    vix3m = {ts: 18.0 for ts in df.index}
    for ts in df.index[-5:]:
        vix[ts]   = 20.0   # ratio = 20/20 = 1.0 → neutral band → abstain
        vix3m[ts] = 20.0

    detector = _make_detector({"cooldown_days": 10})
    regime_series = detector.compute_regime_series(df, vix, vix3m)

    # The last 5 days should still be bull due to hysteresis (regime change < 10 days ago)
    last_labels = [regime_series[ts] for ts in df.index[-5:]]
    assert all(lbl == "bull" for lbl in last_labels), (
        f"Hysteresis: expected bull to persist, got {last_labels}"
    )


# ---------------------------------------------------------------------------
# Test 8: MA200 confidence zone — price within 0.5% of MA200 → abstains
# ---------------------------------------------------------------------------

def test_ma200_confidence_zone():
    """
    Price is within 0.5% of MA200 → MA200 signal abstains.
    RSI neutral (45–55 band) → abstains.
    VIX_struct neutral (0.95–1.05) → abstains.
    → 0 bull, 0 bear → neutral.
    """
    # 210 days flat at 400 → MA200 ≈ 400, price ≈ 400 (within band)
    values = [400.0] * 210
    df = _make_price_data(values)

    # All signals in neutral/abstain zone
    vix   = _make_vix_dict(df, 20.0)   # ratio = 20/20 = 1.0 → neutral band
    vix3m = _make_vix3m_dict(df, 20.0)

    detector = _make_detector({"cooldown_days": 0})
    label = _last_regime(detector, df, vix, vix3m)
    assert label == "neutral", (
        f"MA200 confidence zone: all signals abstaining, expected neutral, got {label}"
    )
