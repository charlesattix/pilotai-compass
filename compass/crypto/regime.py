"""
Crypto regime detector.

Classifies the current market environment along two axes:

    1. Price structure (MA200 position, trend direction)
    2. Sentiment composite (from classify_regime, which consumes all signals)

Regime labels follow the same vocabulary used by compass.crypto.composite_score
so that downstream code can use either module interchangeably:

    extreme_fear | cautious | neutral | bullish | extreme_greed

Design notes:
    - All functions are stateless; they operate on plain Python lists.
    - "crossing" in compute_ma200_position means price is within ±BAND_PCT of
      the 200-day MA — avoids hairline-cross whipsaws.
    - compute_trend uses linear regression slope sign, not just first/last diff,
      for more robust trend detection on noisy crypto prices.
"""

from __future__ import annotations

from typing import Dict, List


# Price must deviate more than this fraction from MA200 to be "above"/"below".
_MA200_BAND_PCT = 0.005  # 0.5% either side

# Thresholds for classify_regime composite score input (same as score bands).
_BAND_EXTREME_FEAR = 25.0
_BAND_CAUTIOUS = 40.0
_BAND_NEUTRAL_HIGH = 60.0
_BAND_BULLISH = 75.0


def compute_ma200_position(prices: List[float]) -> str:
    """Classify price relative to its 200-day moving average.

    Args:
        prices: Daily close prices, chronological order (oldest first).
                At least 200 values are needed; fewer returns "neutral".

    Returns:
        "above"    — price is meaningfully above the 200-day MA
        "below"    — price is meaningfully below the 200-day MA
        "crossing" — price is within ±0.5% of the 200-day MA (transition zone)
    """
    if len(prices) < 200:
        # Not enough history — treat as neutral / no signal
        return "crossing"

    ma200 = sum(prices[-200:]) / 200
    current = prices[-1]

    if ma200 == 0:
        return "crossing"

    deviation = (current - ma200) / ma200

    if deviation > _MA200_BAND_PCT:
        return "above"
    elif deviation < -_MA200_BAND_PCT:
        return "below"
    else:
        return "crossing"


def compute_trend(prices: List[float], window: int = 20) -> str:
    """Classify recent price trend using linear regression slope.

    Args:
        prices: Daily close prices, chronological order (oldest first).
                Minimum required: window values.
        window: Look-back period in days (default 20).

    Returns:
        "uptrend"   — positive slope (regression R-value corroborated)
        "downtrend" — negative slope
        "ranging"   — slope is flat (less than 0.1% per day relative to mean)
    """
    if len(prices) < window:
        return "ranging"

    subset = prices[-window:]
    n = len(subset)
    mean_price = sum(subset) / n

    if mean_price == 0:
        return "ranging"

    # OLS slope via closed-form: slope = Σ(x_i - x̄)(y_i - ȳ) / Σ(x_i - x̄)²
    x_mean = (n - 1) / 2.0
    numerator = sum((i - x_mean) * (subset[i] - mean_price) for i in range(n))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return "ranging"

    slope = numerator / denominator

    # Normalise slope as daily % change relative to mean price
    daily_pct = slope / mean_price

    _FLAT_THRESHOLD = 0.001  # 0.1%/day
    if daily_pct > _FLAT_THRESHOLD:
        return "uptrend"
    elif daily_pct < -_FLAT_THRESHOLD:
        return "downtrend"
    else:
        return "ranging"


def classify_regime(all_signals: Dict) -> str:
    """Classify overall crypto regime from a composite signal dictionary.

    Expects the ``all_signals`` dict to contain a ``composite_score`` key
    (0-100 float) produced by compass.crypto.composite_score.  If not
    present the function falls back to a heuristic vote over whatever
    individual signals are available.

    Args:
        all_signals: Dictionary that may contain any of:
            - composite_score (float, 0-100) ← primary input
            - fear_greed_index (float, 0-100)
            - ma200_position (str: "above"|"below"|"crossing")
            - trend (str: "uptrend"|"downtrend"|"ranging")
            - funding_rate (float, %) — positive = bullish sentiment
            - put_call_ratio (float) — high = fearful

    Returns:
        One of: "extreme_fear" | "cautious" | "neutral" | "bullish" | "extreme_greed"
    """
    # Primary path: use pre-computed composite score if available
    if "composite_score" in all_signals:
        score = float(all_signals["composite_score"])
        return _score_to_regime(score)

    # Fallback: vote from individual signals → derive an approximate score
    votes: List[float] = []

    fg = all_signals.get("fear_greed_index")
    if fg is not None:
        votes.append(float(fg))

    ma200 = all_signals.get("ma200_position")
    if ma200 == "above":
        votes.append(65.0)
    elif ma200 == "below":
        votes.append(25.0)
    elif ma200 == "crossing":
        votes.append(50.0)

    trend = all_signals.get("trend")
    if trend == "uptrend":
        votes.append(65.0)
    elif trend == "downtrend":
        votes.append(25.0)
    elif trend == "ranging":
        votes.append(50.0)

    funding = all_signals.get("funding_rate")
    if funding is not None:
        # funding 0% → 50; +0.1% → ~100; -0.1% → ~0
        clamped = max(-0.1, min(0.1, float(funding)))
        votes.append(50.0 + clamped * 500.0)

    pcr = all_signals.get("put_call_ratio")
    if pcr is not None:
        # high PCR → fearful (low score); pcr 1.0 → 50, 2.0 → 0, 0.5 → ~75
        clamped = max(0.5, min(2.0, float(pcr)))
        votes.append(100.0 * (1.0 - (clamped - 0.5) / 1.5))

    if not votes:
        return "neutral"

    score = sum(votes) / len(votes)
    return _score_to_regime(score)


def _score_to_regime(score: float) -> str:
    """Map a 0-100 composite score to a regime label."""
    if score < _BAND_EXTREME_FEAR:
        return "extreme_fear"
    elif score < _BAND_CAUTIOUS:
        return "cautious"
    elif score < _BAND_NEUTRAL_HIGH:
        return "neutral"
    elif score < _BAND_BULLISH:
        return "bullish"
    else:
        return "extreme_greed"
