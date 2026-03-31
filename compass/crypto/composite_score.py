"""
Crypto composite score engine — THE CORE.

Combines up to seven market signals into a single 0-100 sentiment score.
Each signal is optional; the engine degrades gracefully if any are missing
by re-weighting the remaining signals proportionally.

Score interpretation
--------------------
  0-25  EXTREME_FEAR  → sell puts aggressively (rich put premiums, fear spike)
 25-40  CAUTIOUS      → small positions only
 40-60  NEUTRAL       → iron condors preferred (low-directional premium selling)
 60-75  BULLISH       → sell puts, avoid calls
 75-100 EXTREME_GREED → sell calls aggressively (rich call premiums, euphoria)

Signal → score contribution mapping
------------------------------------
Each signal is normalised to a [0.0, 1.0] component.  Low component (near 0)
means fear/bearish; high component (near 1) means greed/bullish.

  fear_greed_index   : direct linear — 0 → 0.0, 100 → 1.0
  iv_rv_spread       : inverted — high spread (IV>RV) = rich premium = fear
                       spread=0 → 0.5, spread=+0.50 → ~0.0, spread=-0.50 → ~1.0
  funding_rate       : positive funding = longs paying = bullish sentiment
                       0% → 0.5, +0.10% → ~1.0, -0.10% → ~0.0
  ma200_position     : above → 0.70, crossing → 0.50, below → 0.30
  btc_dominance      : inverted — high dominance = risk-off = fear
                       40% dom → 1.0 (alt-season/greed), 70% → 0.0 (BTC flight)
  put_call_ratio     : inverted — high PCR = put-heavy = fearful
                       0.5 → 1.0, 2.0 → 0.0
  exchange_flow_trend: inflow (selling pressure) → 0.20, neutral → 0.50,
                       outflow (accumulation) → 0.80

Usage
-----
    from compass.crypto.composite_score import compute_composite_score

    result = compute_composite_score(
        fear_greed_index=22,
        iv_rv_spread=0.35,
        funding_rate=-0.005,
    )
    # result["score"]  → e.g. 18.4
    # result["band"]   → "EXTREME_FEAR"
    # result["signals"] → dict of per-signal components and weights used
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------

_BANDS = [
    (0.0,  25.0, "EXTREME_FEAR"),
    (25.0, 40.0, "CAUTIOUS"),
    (40.0, 60.0, "NEUTRAL"),
    (60.0, 75.0, "BULLISH"),
    (75.0, 100.0, "EXTREME_GREED"),
]

# Default signal weights (must sum to 1.0)
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "fear_greed_index":    0.25,
    "iv_rv_spread":        0.20,
    "funding_rate":        0.15,
    "ma200_position":      0.15,
    "btc_dominance":       0.10,
    "put_call_ratio":      0.10,
    "exchange_flow_trend": 0.05,
}


# ---------------------------------------------------------------------------
# Individual signal normalisers → [0.0, 1.0]
# ---------------------------------------------------------------------------

def _norm_fear_greed(value: float) -> float:
    """Linear normalisation of 0-100 Fear & Greed index."""
    return max(0.0, min(1.0, value / 100.0))


def _norm_iv_rv_spread(spread: float) -> float:
    """Inverted sigmoid: high IV-RV spread (fear) → low component.

    Spread is in annualised vol points (e.g. 0.30 = 30 pp IV above RV).
    Centre at 0 (fair pricing). ±0.5 maps to ~0.07 / ~0.93.
    """
    # Negative spread → greed (RV>IV, market underpricing risk)
    # Positive spread → fear (IV>RV, market overpricing risk)
    return 1.0 / (1.0 + math.exp(6.0 * spread))


def _norm_funding_rate(rate_pct: float) -> float:
    """Sigmoid: positive funding (longs pay) = greed; negative = fear.

    rate_pct is in % per 8-hour period (typical Binance convention).
    Typical range: -0.10% to +0.30%.  Centre at 0.
    """
    # Scale so that ±0.10% maps to approximately ±3 steepness units
    return 1.0 / (1.0 + math.exp(-30.0 * rate_pct))


def _norm_ma200_position(position: str) -> float:
    """Discrete mapping for MA200 position string."""
    mapping = {
        "above":    0.70,
        "crossing": 0.50,
        "below":    0.30,
    }
    return mapping.get(str(position).lower(), 0.50)


def _norm_btc_dominance(dominance_pct: float) -> float:
    """Inverted linear: high BTC dominance = risk-off = fear component.

    Maps 40% → 1.0 (alt-season, greed), 70% → 0.0 (BTC flight, fear).
    Values outside [40, 70] are clamped.
    """
    clamped = max(40.0, min(70.0, dominance_pct))
    return 1.0 - (clamped - 40.0) / 30.0


def _norm_put_call_ratio(pcr: float) -> float:
    """Inverted linear: high PCR = put-heavy = fear component.

    Maps 0.5 → 1.0 (greed), 2.0 → 0.0 (extreme fear).
    Values outside [0.5, 2.0] are clamped.
    """
    clamped = max(0.5, min(2.0, pcr))
    return 1.0 - (clamped - 0.5) / 1.5


def _norm_exchange_flow(trend: str) -> float:
    """Discrete mapping for exchange flow direction.

    inflow  — coins moving INTO exchanges → selling pressure → bearish → 0.20
    neutral — no dominant flow → 0.50
    outflow — coins moving OUT of exchanges → accumulation → bullish → 0.80
    """
    mapping = {
        "inflow":  0.20,
        "neutral": 0.50,
        "outflow": 0.80,
    }
    return mapping.get(str(trend).lower(), 0.50)


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

_NORMALISERS = {
    "fear_greed_index":    _norm_fear_greed,
    "iv_rv_spread":        _norm_iv_rv_spread,
    "funding_rate":        _norm_funding_rate,
    "ma200_position":      _norm_ma200_position,
    "btc_dominance":       _norm_btc_dominance,
    "put_call_ratio":      _norm_put_call_ratio,
    "exchange_flow_trend": _norm_exchange_flow,
}


def compute_composite_score(
    fear_greed_index:    Optional[float] = None,
    iv_rv_spread:        Optional[float] = None,
    funding_rate:        Optional[float] = None,
    ma200_position:      Optional[str]   = None,
    btc_dominance:       Optional[float] = None,
    put_call_ratio:      Optional[float] = None,
    exchange_flow_trend: Optional[str]   = None,
    weights:             Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute the weighted composite crypto sentiment score (0-100).

    All inputs are optional. Missing signals are dropped and the remaining
    weights are re-normalised proportionally so the output always spans
    the full 0-100 range regardless of data availability.

    Args:
        fear_greed_index:    Crypto Fear & Greed index (0-100).
        iv_rv_spread:        IV minus RV in annualised vol points (e.g. 0.30).
        funding_rate:        Perpetual funding rate in % per 8h (e.g. 0.01 = 0.01%).
        ma200_position:      Price vs 200-day MA: "above" | "crossing" | "below".
        btc_dominance:       BTC market cap dominance as % (e.g. 52.0).
        put_call_ratio:      Put/call open-interest or volume ratio (e.g. 0.8).
        exchange_flow_trend: Exchange flow direction: "inflow"|"neutral"|"outflow".
        weights:             Override default signal weights (must sum to 1.0).
                             Partial overrides are not supported — supply all keys.

    Returns:
        Dictionary with keys:
            score     (float)  — composite score 0-100
            band      (str)    — "EXTREME_FEAR"|"CAUTIOUS"|"NEUTRAL"|"BULLISH"|"EXTREME_GREED"
            signals   (dict)   — per-signal breakdown:
                                   {signal_name: {raw, component, weight_used}}
            timestamp (datetime) — UTC timestamp of computation

    Raises:
        ValueError: If all inputs are None (no signal available).
    """
    raw_inputs: Dict[str, Any] = {
        "fear_greed_index":    fear_greed_index,
        "iv_rv_spread":        iv_rv_spread,
        "funding_rate":        funding_rate,
        "ma200_position":      ma200_position,
        "btc_dominance":       btc_dominance,
        "put_call_ratio":      put_call_ratio,
        "exchange_flow_trend": exchange_flow_trend,
    }

    base_weights = weights if weights is not None else _DEFAULT_WEIGHTS

    # Filter to signals that are actually present
    present = {k: v for k, v in raw_inputs.items() if v is not None}
    if not present:
        raise ValueError(
            "compute_composite_score: at least one signal must be provided."
        )

    # Re-normalise weights over present signals only
    total_weight = sum(base_weights[k] for k in present)
    effective_weights = {k: base_weights[k] / total_weight for k in present}

    # Compute per-signal components
    signal_details: Dict[str, Any] = {}
    weighted_sum = 0.0

    for name, raw_value in present.items():
        component = _NORMALISERS[name](raw_value)
        w = effective_weights[name]
        weighted_sum += w * component
        signal_details[name] = {
            "raw":         raw_value,
            "component":   round(component, 4),
            "weight_used": round(w, 4),
        }

    score = round(weighted_sum * 100.0, 2)
    score = max(0.0, min(100.0, score))  # safety clamp

    return {
        "score":     score,
        "band":      _score_to_band(score),
        "signals":   signal_details,
        "timestamp": datetime.now(timezone.utc),
    }


def _score_to_band(score: float) -> str:
    """Map a numeric score to its label band."""
    for lo, hi, label in _BANDS:
        if lo <= score < hi:
            return label
    # score == 100.0 exactly
    return "EXTREME_GREED"
