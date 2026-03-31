"""
IBIT-specific feature engineering for EXP-601 ML Signal Filter.

Builds crypto-native features that capture BTC/IBIT market dynamics beyond
the generic SPY-centric features in ``compass.features``.  Designed to work
with both live data (API calls) and backtesting (precomputed price arrays).

Feature categories
------------------
1. **BTC Volatility Regime** — realized vol at 7d/30d horizons (crypto = 365 days),
   vol ratio (7d/30d) for regime detection, vol percentile rank.
2. **Crypto Correlation** — IBIT vs ETHA (ETH ETF) rolling correlation as a
   crypto-beta proxy; divergence from SPY to measure crypto-TradFi decoupling.
3. **Contango/Funding Proxy** — BTC futures contango inferred from funding rate
   history; mean funding rate, funding rate trend, extreme funding flags.
4. **Composite Score Features** — normalized components from the existing
   ``compass.crypto.composite_score`` engine.
5. **IBIT-specific Microstructure** — IBIT premium/discount to NAV proxy (via
   volume-price dynamics), intraday range ratio, gap behavior.

Usage::

    from compass.ibit_features import IBITFeatureEngine

    engine = IBITFeatureEngine()
    features = engine.compute(
        ibit_prices=[55.0, 56.2, ...],   # IBIT daily closes (oldest first)
        etha_prices=[18.0, 18.5, ...],    # ETHA daily closes (oldest first)
        spy_prices=[580.0, 582.0, ...],   # SPY daily closes (oldest first)
        ibit_volumes=[1e6, 1.2e6, ...],   # IBIT daily volumes
        ibit_highs=[56.5, 57.0, ...],     # IBIT daily highs
        ibit_lows=[54.5, 55.8, ...],      # IBIT daily lows
        funding_rates=[0.01, -0.005, ...], # BTC perp funding rates (% per 8h)
        fear_greed_index=25,              # current FGI (0-100)
        btc_dominance=52.0,              # BTC market cap dominance %
    )
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Any

import numpy as np


# ── Constants ────────────────────────────────────────────────────────────

_CRYPTO_DAYS_YEAR = 365  # crypto trades 24/7
_EQUITY_DAYS_YEAR = 252  # traditional markets


# ── BTC Volatility Regime ────────────────────────────────────────────────

def realized_vol_crypto(prices: List[float], window: int) -> float:
    """Annualized realized vol from daily closes using crypto annualization (365d).

    Returns 0.0 if insufficient data.
    """
    if len(prices) < window + 1:
        return 0.0
    recent = prices[-(window + 1):]
    if any(p <= 0 for p in recent):
        return 0.0
    log_rets = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))]
    n = len(log_rets)
    mean = sum(log_rets) / n
    var = sum((r - mean) ** 2 for r in log_rets) / (n - 1) if n > 1 else 0.0
    return math.sqrt(var) * math.sqrt(_CRYPTO_DAYS_YEAR)


def vol_regime_features(prices: List[float]) -> Dict[str, Optional[float]]:
    """Compute BTC/IBIT volatility regime features.

    Returns:
        rv_7d:          7-day realized vol (annualized, crypto)
        rv_30d:         30-day realized vol (annualized, crypto)
        vol_ratio_7_30: Ratio of 7d/30d vol (>1 = vol expanding, <1 = compressing)
        vol_percentile: Where current 30d vol sits in trailing 90d range (0-100)
    """
    rv_7d = realized_vol_crypto(prices, 7)
    rv_30d = realized_vol_crypto(prices, 30)

    vol_ratio = rv_7d / rv_30d if rv_30d > 0 else 1.0

    # Vol percentile: rank current 30d vol against trailing windows
    vol_pct = None
    if len(prices) > 120:  # need 90d of 30d-vol readings
        vols = []
        for i in range(90):
            end = len(prices) - i
            if end > 31:
                v = realized_vol_crypto(prices[:end], 30)
                vols.append(v)
        if vols and rv_30d > 0:
            below = sum(1 for v in vols if v < rv_30d)
            vol_pct = round(below / len(vols) * 100, 2)

    return {
        "rv_7d_crypto": round(rv_7d, 4) if rv_7d else None,
        "rv_30d_crypto": round(rv_30d, 4) if rv_30d else None,
        "vol_ratio_7_30": round(vol_ratio, 4),
        "vol_percentile_90d": vol_pct,
    }


# ── Crypto Correlation Features ─────────────────────────────────────────

def _rolling_correlation(a: List[float], b: List[float], window: int) -> Optional[float]:
    """Pearson correlation of log-returns over trailing window.

    Returns None if insufficient data or zero variance in either series.
    """
    min_len = window + 1
    if len(a) < min_len or len(b) < min_len:
        return None
    ra = [math.log(a[i] / a[i - 1]) for i in range(len(a) - window, len(a))]
    rb = [math.log(b[i] / b[i - 1]) for i in range(len(b) - window, len(b))]
    n = len(ra)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    cov = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n)) / (n - 1)
    var_a = sum((x - mean_a) ** 2 for x in ra) / (n - 1)
    var_b = sum((x - mean_b) ** 2 for x in rb) / (n - 1)
    if var_a <= 0 or var_b <= 0:
        return None
    return cov / (math.sqrt(var_a) * math.sqrt(var_b))


def correlation_features(
    ibit_prices: List[float],
    etha_prices: Optional[List[float]],
    spy_prices: Optional[List[float]],
) -> Dict[str, Optional[float]]:
    """Compute cross-asset correlation features.

    Returns:
        ibit_etha_corr_7d:  7-day IBIT-ETHA correlation (crypto-beta)
        ibit_etha_corr_30d: 30-day IBIT-ETHA correlation
        ibit_spy_corr_30d:  30-day IBIT-SPY correlation (TradFi coupling)
        crypto_decouple:    Difference: ibit_etha_corr - ibit_spy_corr
                            (positive = crypto moves together, away from equities)
    """
    etha_7d = _rolling_correlation(ibit_prices, etha_prices, 7) if etha_prices else None
    etha_30d = _rolling_correlation(ibit_prices, etha_prices, 30) if etha_prices else None
    spy_30d = _rolling_correlation(ibit_prices, spy_prices, 30) if spy_prices else None

    decouple = None
    if etha_30d is not None and spy_30d is not None:
        decouple = round(etha_30d - spy_30d, 4)

    return {
        "ibit_etha_corr_7d": round(etha_7d, 4) if etha_7d is not None else None,
        "ibit_etha_corr_30d": round(etha_30d, 4) if etha_30d is not None else None,
        "ibit_spy_corr_30d": round(spy_30d, 4) if spy_30d is not None else None,
        "crypto_decouple": decouple,
    }


# ── Contango / Funding Rate Features ────────────────────────────────────

def funding_rate_features(rates: Optional[List[float]]) -> Dict[str, Optional[float]]:
    """Derive contango/backwardation signals from BTC perpetual funding rates.

    BTC perps settle funding every 8 hours. Positive rate = longs pay shorts
    (bullish crowding). Negative rate = shorts pay longs (bearish crowding).

    The funding rate is a proxy for the futures basis (contango/backwardation)
    because arb keeps perp price near spot — funding is the mechanism that
    maintains the peg.

    Args:
        rates: List of funding rate observations (% per 8h), most recent last.
               Typical range: -0.10% to +0.30%.

    Returns:
        funding_rate_current:   Most recent settled rate
        funding_rate_mean_3d:   Mean of last 9 settlements (~3 days)
        funding_rate_mean_7d:   Mean of last 21 settlements (~7 days)
        funding_trend_3d:       Slope: mean of last 9 minus mean of prior 9
        funding_extreme_bull:   1 if mean_3d > 0.05%, else 0 (leveraged longs)
        funding_extreme_bear:   1 if mean_3d < -0.02%, else 0 (leveraged shorts)
        contango_strength:      Normalized 7d mean: >0 = contango, <0 = backwardation
    """
    if not rates or len(rates) == 0:
        return {
            "funding_rate_current": None,
            "funding_rate_mean_3d": None,
            "funding_rate_mean_7d": None,
            "funding_trend_3d": None,
            "funding_extreme_bull": None,
            "funding_extreme_bear": None,
            "contango_strength": None,
        }

    current = rates[-1]

    # 3-day mean (9 settlements)
    last_9 = rates[-9:] if len(rates) >= 9 else rates
    mean_3d = sum(last_9) / len(last_9)

    # 7-day mean (21 settlements)
    last_21 = rates[-21:] if len(rates) >= 21 else rates
    mean_7d = sum(last_21) / len(last_21)

    # Trend: compare recent 3d to prior 3d
    trend_3d = None
    if len(rates) >= 18:
        prior_9 = rates[-18:-9]
        prior_mean = sum(prior_9) / len(prior_9)
        trend_3d = round(mean_3d - prior_mean, 6)

    # Extreme flags
    extreme_bull = 1 if mean_3d > 0.05 else 0
    extreme_bear = 1 if mean_3d < -0.02 else 0

    # Contango strength: sigmoid normalization of 7d mean
    # Maps [-0.10, +0.30] range to roughly [-1, +1]
    contango = round(math.tanh(mean_7d * 20), 4)

    return {
        "funding_rate_current": round(current, 6),
        "funding_rate_mean_3d": round(mean_3d, 6),
        "funding_rate_mean_7d": round(mean_7d, 6),
        "funding_trend_3d": trend_3d,
        "funding_extreme_bull": extreme_bull,
        "funding_extreme_bear": extreme_bear,
        "contango_strength": contango,
    }


# ── IBIT Microstructure Features ────────────────────────────────────────

def microstructure_features(
    prices: List[float],
    volumes: Optional[List[float]],
    highs: Optional[List[float]],
    lows: Optional[List[float]],
) -> Dict[str, Optional[float]]:
    """Compute IBIT-specific microstructure features.

    Returns:
        volume_ratio_20d:   Today's volume / 20d average (unusual activity)
        intraday_range_pct: (high - low) / close as % (realized intraday vol proxy)
        range_ratio_5d:     Today's range / 5d average range (expansion/contraction)
        gap_pct:            Overnight gap: (today open proxy - yesterday close) / yest close
                            Approximated as (today low - yesterday close) for daily data.
        momentum_3d_pct:    3-day price momentum (%)
        momentum_7d_pct:    7-day price momentum (%)
        dist_from_ma20_pct: Distance from 20-day MA as % of price
    """
    result: Dict[str, Optional[float]] = {
        "volume_ratio_20d": None,
        "intraday_range_pct": None,
        "range_ratio_5d": None,
        "gap_pct": None,
        "momentum_3d_pct": None,
        "momentum_7d_pct": None,
        "ibit_dist_ma20_pct": None,
    }

    if not prices or len(prices) < 2:
        return result

    current = prices[-1]

    # Volume ratio
    if volumes and len(volumes) >= 20:
        avg_vol = sum(volumes[-20:]) / 20
        if avg_vol > 0:
            result["volume_ratio_20d"] = round(volumes[-1] / avg_vol, 4)

    # Intraday range
    if highs and lows and len(highs) >= 1 and len(lows) >= 1 and current > 0:
        today_range = highs[-1] - lows[-1]
        result["intraday_range_pct"] = round(today_range / current * 100, 4)

        # Range ratio (expansion/contraction)
        if len(highs) >= 5 and len(lows) >= 5:
            ranges_5d = [highs[-i] - lows[-i] for i in range(1, 6)]
            avg_range = sum(ranges_5d) / 5
            if avg_range > 0:
                result["range_ratio_5d"] = round(today_range / avg_range, 4)

    # Gap (proxy: today's low vs yesterday's close)
    if lows and len(lows) >= 1 and len(prices) >= 2 and prices[-2] > 0:
        gap = (lows[-1] - prices[-2]) / prices[-2] * 100
        result["gap_pct"] = round(gap, 4)

    # Momentum
    if len(prices) >= 4 and prices[-4] > 0:
        result["momentum_3d_pct"] = round((current - prices[-4]) / prices[-4] * 100, 4)
    if len(prices) >= 8 and prices[-8] > 0:
        result["momentum_7d_pct"] = round((current - prices[-8]) / prices[-8] * 100, 4)

    # Distance from 20d MA
    if len(prices) >= 20:
        ma20 = sum(prices[-20:]) / 20
        if ma20 > 0:
            result["ibit_dist_ma20_pct"] = round((current - ma20) / ma20 * 100, 4)

    return result


# ── Composite Score Features ─────────────────────────────────────────────

def composite_score_features(
    fear_greed_index: Optional[float] = None,
    funding_rate_mean: Optional[float] = None,
    btc_dominance: Optional[float] = None,
    rv_30d: Optional[float] = None,
    iv_premium: float = 0.10,
) -> Dict[str, Optional[float]]:
    """Extract normalized features from crypto composite score components.

    Instead of using the composite score as a single number, we expose the
    individual normalized components as ML features — letting the model learn
    non-linear interactions between them.

    Args:
        fear_greed_index: Crypto Fear & Greed Index (0-100).
        funding_rate_mean: Mean funding rate (% per 8h) over recent window.
        btc_dominance: BTC market cap dominance as % (e.g. 52.0).
        rv_30d: 30-day realized vol (annualized decimal).
        iv_premium: IV premium above RV for IV proxy (default 10%).

    Returns:
        fgi_normalized:       Fear & Greed scaled to [0, 1]
        iv_rv_spread_norm:    IV-RV spread normalized via sigmoid
        funding_norm:         Funding rate normalized via sigmoid
        dominance_norm:       BTC dominance normalized (inverted, fear scale)
        composite_score:      Full composite score if enough signals present
    """
    result: Dict[str, Optional[float]] = {
        "fgi_normalized": None,
        "iv_rv_spread_norm": None,
        "funding_norm": None,
        "dominance_norm": None,
        "composite_score": None,
    }

    if fear_greed_index is not None:
        result["fgi_normalized"] = round(max(0.0, min(1.0, fear_greed_index / 100.0)), 4)

    if rv_30d is not None and rv_30d > 0:
        iv_proxy = rv_30d * (1.0 + iv_premium)
        spread = iv_proxy - rv_30d  # = rv_30d * iv_premium
        # Sigmoid normalization (inverted: high spread = fear = low value)
        result["iv_rv_spread_norm"] = round(1.0 / (1.0 + math.exp(6.0 * spread)), 4)

    if funding_rate_mean is not None:
        result["funding_norm"] = round(1.0 / (1.0 + math.exp(-30.0 * funding_rate_mean)), 4)

    if btc_dominance is not None:
        clamped = max(40.0, min(70.0, btc_dominance))
        result["dominance_norm"] = round(1.0 - (clamped - 40.0) / 30.0, 4)

    # Composite: simple average of available normalized signals
    normed = [v for v in [
        result["fgi_normalized"],
        result["iv_rv_spread_norm"],
        result["funding_norm"],
        result["dominance_norm"],
    ] if v is not None]
    if normed:
        result["composite_score"] = round(sum(normed) / len(normed) * 100, 2)

    return result


# ── Main Feature Engine ──────────────────────────────────────────────────

# Canonical list of all features produced by this engine, in stable order.
IBIT_FEATURE_NAMES: List[str] = [
    # Volatility regime
    "rv_7d_crypto",
    "rv_30d_crypto",
    "vol_ratio_7_30",
    "vol_percentile_90d",
    # Correlation
    "ibit_etha_corr_7d",
    "ibit_etha_corr_30d",
    "ibit_spy_corr_30d",
    "crypto_decouple",
    # Funding / contango
    "funding_rate_current",
    "funding_rate_mean_3d",
    "funding_rate_mean_7d",
    "funding_trend_3d",
    "funding_extreme_bull",
    "funding_extreme_bear",
    "contango_strength",
    # Microstructure
    "volume_ratio_20d",
    "intraday_range_pct",
    "range_ratio_5d",
    "gap_pct",
    "momentum_3d_pct",
    "momentum_7d_pct",
    "ibit_dist_ma20_pct",
    # Composite score components
    "fgi_normalized",
    "iv_rv_spread_norm",
    "funding_norm",
    "dominance_norm",
    "composite_score",
]


class IBITFeatureEngine:
    """Crypto-native feature engine for IBIT ML signal filtering.

    Computes all IBIT-specific features from raw price/volume/signal data.
    Designed for both live prediction and backtest feature extraction.
    """

    def __init__(self, iv_premium: float = 0.10):
        self.iv_premium = iv_premium

    @property
    def feature_names(self) -> List[str]:
        """Canonical ordered list of feature names."""
        return list(IBIT_FEATURE_NAMES)

    def compute(
        self,
        ibit_prices: List[float],
        etha_prices: Optional[List[float]] = None,
        spy_prices: Optional[List[float]] = None,
        ibit_volumes: Optional[List[float]] = None,
        ibit_highs: Optional[List[float]] = None,
        ibit_lows: Optional[List[float]] = None,
        funding_rates: Optional[List[float]] = None,
        fear_greed_index: Optional[float] = None,
        btc_dominance: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """Compute all IBIT-specific features.

        Args:
            ibit_prices:      IBIT daily closes, oldest first. Min 31 for basic features.
            etha_prices:      ETHA daily closes (same length/alignment as ibit_prices).
            spy_prices:       SPY daily closes (same length/alignment as ibit_prices).
            ibit_volumes:     IBIT daily volumes (same alignment).
            ibit_highs:       IBIT daily highs (same alignment).
            ibit_lows:        IBIT daily lows (same alignment).
            funding_rates:    BTC perp funding rates (% per 8h), most recent last.
            fear_greed_index: Current Fear & Greed Index (0-100).
            btc_dominance:    BTC market cap dominance as %.

        Returns:
            Dict mapping each feature name in IBIT_FEATURE_NAMES to its value
            (float or None if insufficient data).
        """
        features: Dict[str, Optional[float]] = {}

        # 1. Volatility regime
        features.update(vol_regime_features(ibit_prices))

        # 2. Cross-asset correlation
        features.update(correlation_features(ibit_prices, etha_prices, spy_prices))

        # 3. Funding / contango
        features.update(funding_rate_features(funding_rates))

        # 4. Microstructure
        features.update(microstructure_features(
            ibit_prices, ibit_volumes, ibit_highs, ibit_lows,
        ))

        # 5. Composite score components
        rv_30d = features.get("rv_30d_crypto")
        funding_mean = features.get("funding_rate_mean_3d")
        features.update(composite_score_features(
            fear_greed_index=fear_greed_index,
            funding_rate_mean=funding_mean,
            btc_dominance=btc_dominance,
            rv_30d=rv_30d,
            iv_premium=self.iv_premium,
        ))

        return features

    def compute_array(self, **kwargs) -> np.ndarray:
        """Compute features and return as a (1, n_features) numpy array.

        Missing values are filled with 0.0 for ML model compatibility.
        """
        features = self.compute(**kwargs)
        values = [features.get(name, 0.0) or 0.0 for name in IBIT_FEATURE_NAMES]
        return np.array(values, dtype=np.float64).reshape(1, -1)
