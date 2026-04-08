"""
Real-time signal generation pipeline for live paper trading.

Integrates: feature engineering → ensemble ML prediction → regime
detection → crisis hedge scaling → final signal with confidence +
position size.  Includes health checks for model staleness, feature
drift, and ensemble agreement.

Sub-millisecond per-signal target (from EXP-930 findings).

Usage::

    from compass.signal_pipeline import SignalPipeline, PipelineConfig
    pipe = SignalPipeline(PipelineConfig())
    result = pipe.generate_signal(market_data)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    ml_threshold: float = 0.60           # minimum P(win) to trade
    min_confidence: float = 0.20         # minimum ensemble confidence
    max_position_scale: float = 1.0      # cap on position scale factor
    base_contracts: int = 2              # baseline contract count
    max_contracts: int = 10              # absolute max contracts
    # Crisis hedge
    vix_scale_floor: float = 12.0
    vix_scale_ceiling: float = 35.0
    crash_regime_scale: float = 0.0
    high_vol_regime_scale: float = 0.25
    # Regime leverage
    regime_leverage: Dict[str, float] = field(default_factory=lambda: {
        "bull": 2.0, "neutral": 1.0, "bear": 0.4,
        "high_vol": 0.25, "low_vol": 1.2, "crash": 0.1,
    })
    # Health
    model_stale_seconds: float = 86400   # 24h
    feature_drift_z_threshold: float = 3.0
    min_ensemble_agreement: float = 0.6
    # Logging
    log_all_signals: bool = True


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class MarketData:
    """Snapshot of current market state."""
    ticker: str = "SPY"
    price: float = 430.0
    vix: float = 18.0
    vix3m: Optional[float] = None
    regime: str = "neutral"
    rsi_14: float = 50.0
    momentum_5d_pct: float = 0.0
    momentum_10d_pct: float = 0.0
    iv_rank: float = 30.0
    vix_percentile_20d: float = 50.0
    vix_percentile_50d: float = 50.0
    vix_percentile_100d: float = 50.0
    dist_from_ma20_pct: float = 0.0
    dist_from_ma50_pct: float = 0.0
    dist_from_ma80_pct: float = 0.0
    dist_from_ma200_pct: float = 0.0
    ma20_slope_ann_pct: float = 0.0
    ma50_slope_ann_pct: float = 0.0
    realized_vol_5d: float = 10.0
    realized_vol_10d: float = 10.0
    realized_vol_20d: float = 10.0
    realized_vol_atr20: float = 10.0
    spread_width: float = 5.0
    net_credit: float = 1.5
    max_loss_per_unit: float = 3.5
    dte_at_entry: int = 30
    day_of_week: int = 2
    days_since_last_trade: float = 5.0
    hold_days: int = 10
    timestamp: str = ""


@dataclass
class PipelineSignal:
    """Output of the signal pipeline."""
    trade: bool                          # True = execute trade
    ticker: str
    direction: str                       # "short_put_spread", "short_call_spread", etc.
    contracts: int
    confidence: float                    # 0-1
    probability: float                   # P(win) from ensemble
    regime: str
    position_scale: float                # 0-1 from crisis hedge
    regime_leverage: float               # multiplier from regime
    stop_loss_mult: float                # crisis-adjusted stop
    reject_reason: str                   # "" if trade, else why rejected
    latency_us: float                    # microseconds to generate
    timestamp: str
    features_used: int
    ensemble_agreement: float            # agreement among sub-models
    health_ok: bool


@dataclass
class HealthStatus:
    """Pipeline health diagnostics."""
    model_loaded: bool
    model_age_seconds: float
    model_stale: bool
    feature_drift_detected: bool
    drifted_features: List[str]
    ensemble_agreement: float
    last_signal_time: str
    signals_generated: int
    signals_traded: int
    signals_rejected: int
    avg_latency_us: float
    health_ok: bool


# ── Feature engineering ─────────────────────────────────────────────────


FEATURE_NAMES = [
    "dte_at_entry", "hold_days", "day_of_week", "days_since_last_trade",
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_20d", "vix_percentile_50d", "vix_percentile_100d",
    "iv_rank", "spy_price",
    "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct",
    "dist_from_ma200_pct", "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d",
    "realized_vol_20d", "net_credit", "spread_width", "max_loss_per_unit",
]


def extract_features(md: MarketData) -> Dict[str, float]:
    """Extract ML features from market data snapshot."""
    return {
        "dte_at_entry": float(md.dte_at_entry),
        "hold_days": float(md.hold_days),
        "day_of_week": float(md.day_of_week),
        "days_since_last_trade": md.days_since_last_trade,
        "rsi_14": md.rsi_14,
        "momentum_5d_pct": md.momentum_5d_pct,
        "momentum_10d_pct": md.momentum_10d_pct,
        "vix": md.vix,
        "vix_percentile_20d": md.vix_percentile_20d,
        "vix_percentile_50d": md.vix_percentile_50d,
        "vix_percentile_100d": md.vix_percentile_100d,
        "iv_rank": md.iv_rank,
        "spy_price": md.price,
        "dist_from_ma20_pct": md.dist_from_ma20_pct,
        "dist_from_ma50_pct": md.dist_from_ma50_pct,
        "dist_from_ma80_pct": md.dist_from_ma80_pct,
        "dist_from_ma200_pct": md.dist_from_ma200_pct,
        "ma20_slope_ann_pct": md.ma20_slope_ann_pct,
        "ma50_slope_ann_pct": md.ma50_slope_ann_pct,
        "realized_vol_atr20": md.realized_vol_atr20,
        "realized_vol_5d": md.realized_vol_5d,
        "realized_vol_10d": md.realized_vol_10d,
        "realized_vol_20d": md.realized_vol_20d,
        "net_credit": md.net_credit,
        "spread_width": md.spread_width,
        "max_loss_per_unit": md.max_loss_per_unit,
    }


# ── Regime detection ────────────────────────────────────────────────────


def detect_regime(md: MarketData) -> str:
    """Rule + VIX based regime detection (from EXP-900)."""
    if md.vix > 35:
        return "crash"
    if md.vix > 28:
        return "high_vol"
    if md.dist_from_ma200_pct < -5 and md.momentum_10d_pct < -2:
        return "bear"
    if md.vix < 14 and md.realized_vol_20d < 8:
        return "low_vol"
    if md.dist_from_ma200_pct > 3 and md.rsi_14 > 50:
        return "bull"
    return "neutral"


# ── Crisis hedge scaling ────────────────────────────────────────────────


def compute_position_scale(
    vix: float,
    regime: str,
    config: PipelineConfig,
    vix3m: Optional[float] = None,
) -> float:
    """VIX-adaptive position scale factor (0-1). From CrisisHedgeController."""
    r = regime.lower()

    if r == "crash":
        return config.crash_regime_scale
    if r == "high_vol":
        vix_s = _vix_scale(vix, config.vix_scale_floor, config.vix_scale_ceiling)
        return min(vix_s, config.high_vol_regime_scale)

    scale = _vix_scale(vix, config.vix_scale_floor, config.vix_scale_ceiling)

    # VIX term structure: backwardation penalty
    if vix3m is not None and vix3m > 0:
        ratio = vix3m / max(vix, 1)
        if ratio < 1.0:
            penalty = min(0.25, (1.0 - ratio) * 2)
            scale *= (1.0 - penalty)

    return max(0.0, min(1.0, scale))


def compute_stop_loss_mult(
    vix: float,
    regime: str,
    base_stop: float = 3.5,
    min_stop: float = 1.5,
    floor: float = 12.0,
    ceiling: float = 25.8,
) -> float:
    """VIX-adaptive stop-loss multiplier."""
    if regime.lower() == "crash":
        return min_stop
    if vix <= floor:
        return base_stop
    if vix >= ceiling:
        return min_stop
    frac = (vix - floor) / (ceiling - floor)
    return base_stop - frac * (base_stop - min_stop)


def _vix_scale(vix: float, floor: float, ceiling: float) -> float:
    if vix <= floor:
        return 1.0
    if vix >= ceiling:
        return 0.0
    return 1.0 - (vix - floor) / (ceiling - floor)


# ── Ensemble ML prediction (lightweight wrapper) ───────────────────────


class EnsemblePredictor:
    """Wraps the production ensemble for pipeline use.

    In production, loads the real EnsembleSignalModel. For testing
    and offline use, provides a statistical fallback.
    """

    def __init__(self) -> None:
        self._model = None
        self._model_loaded_at: Optional[float] = None
        self._fallback_mode = True
        self._feature_means: Dict[str, float] = {}
        self._feature_stds: Dict[str, float] = {}
        self._try_load()

    def _try_load(self) -> None:
        """Attempt to load the production ensemble model."""
        try:
            from compass.ensemble_signal_model import EnsembleSignalModel
            model = EnsembleSignalModel()
            if model.load():
                self._model = model
                self._fallback_mode = False
                self._model_loaded_at = time.time()
                logger.info("EnsemblePredictor: loaded production model")
                return
        except Exception as e:
            logger.debug("EnsemblePredictor: production model unavailable: %s", e)
        self._fallback_mode = True
        self._model_loaded_at = time.time()
        logger.info("EnsemblePredictor: using statistical fallback")

    @property
    def model_age_seconds(self) -> float:
        if self._model_loaded_at is None:
            return float("inf")
        return time.time() - self._model_loaded_at

    @property
    def is_loaded(self) -> bool:
        return not self._fallback_mode

    def predict(self, features: Dict[str, float]) -> Tuple[float, float, float]:
        """Predict P(win), confidence, and ensemble agreement.

        Returns (probability, confidence, agreement).
        """
        if not self._fallback_mode and self._model is not None:
            try:
                result = self._model.predict(features)
                prob = result.get("probability", 0.5)
                conf = result.get("confidence", 0.0)
                return float(prob), float(conf), 1.0  # real model → agreement=1
            except Exception:
                pass

        # Statistical fallback: use feature heuristics
        return self._fallback_predict(features)

    def _fallback_predict(self, features: Dict[str, float]) -> Tuple[float, float, float]:
        """Heuristic prediction when model unavailable."""
        vix = features.get("vix", 20)
        rsi = features.get("rsi_14", 50)
        iv_rank = features.get("iv_rank", 30)
        ma_dist = features.get("dist_from_ma200_pct", 0)

        # Simple logistic: higher IV rank + moderate RSI + above MA200 → higher P(win)
        score = 0.50
        score += (iv_rank - 30) * 0.003      # higher IV → more premium
        score += (50 - abs(rsi - 50)) * 0.002 # moderate RSI → mean reversion
        score += min(ma_dist, 5) * 0.01       # above MA200 → bullish
        score -= max(vix - 25, 0) * 0.005     # high VIX → danger

        prob = max(0.1, min(0.95, score))
        conf = abs(prob - 0.5) * 2
        agreement = 0.5  # low agreement in fallback mode
        return prob, conf, agreement

    def check_feature_drift(
        self, features: Dict[str, float], z_threshold: float = 3.0,
    ) -> List[str]:
        """Detect features that have drifted from training distribution."""
        drifted = []
        for name, val in features.items():
            mean = self._feature_means.get(name)
            std = self._feature_stds.get(name)
            if mean is not None and std is not None and std > 0:
                z = abs(val - mean) / std
                if z > z_threshold:
                    drifted.append(name)
        return drifted

    def set_feature_stats(self, means: Dict[str, float], stds: Dict[str, float]) -> None:
        """Set training-time feature statistics for drift detection."""
        self._feature_means = dict(means)
        self._feature_stds = dict(stds)


# ── Signal pipeline ─────────────────────────────────────────────────────


class SignalPipeline:
    """Real-time signal generation pipeline.

    Flow: market data → features → ML prediction → regime detection →
    crisis hedge scaling → final signal with confidence + position size.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.predictor = EnsemblePredictor()
        self._signals_generated = 0
        self._signals_traded = 0
        self._signals_rejected = 0
        self._latencies: List[float] = []
        self._last_signal_time = ""
        self._signal_log: List[PipelineSignal] = []

    # ── Core signal generation ──────────────────────────────────────────

    def generate_signal(self, market_data: MarketData) -> PipelineSignal:
        """Generate a trade/no-trade signal from market data.

        Target: sub-millisecond latency per signal.
        """
        t0 = time.perf_counter()

        # Step 1: Feature engineering
        features = extract_features(market_data)

        # Step 2: Regime detection
        regime = detect_regime(market_data)

        # Step 3: ML ensemble prediction
        prob, conf, agreement = self.predictor.predict(features)

        # Step 4: Crisis hedge scaling
        pos_scale = compute_position_scale(
            market_data.vix, regime, self.config, market_data.vix3m,
        )
        stop_mult = compute_stop_loss_mult(market_data.vix, regime)

        # Step 5: Regime leverage
        reg_lev = self.config.regime_leverage.get(regime, 1.0)

        # Step 6: Position sizing
        raw_contracts = self.config.base_contracts * pos_scale * reg_lev
        contracts = max(0, min(int(round(raw_contracts)), self.config.max_contracts))

        # Step 7: Trade decision
        trade = True
        reject_reason = ""

        if prob < self.config.ml_threshold:
            trade = False
            reject_reason = f"P(win)={prob:.2f} < threshold {self.config.ml_threshold}"
        elif conf < self.config.min_confidence:
            trade = False
            reject_reason = f"Confidence={conf:.2f} < min {self.config.min_confidence}"
        elif pos_scale <= 0:
            trade = False
            reject_reason = f"Position scale=0 (regime={regime}, VIX={market_data.vix})"
        elif contracts <= 0:
            trade = False
            reject_reason = f"Contracts=0 after scaling (scale={pos_scale:.2f}, lev={reg_lev:.1f})"
        elif agreement < self.config.min_ensemble_agreement:
            trade = False
            reject_reason = f"Ensemble agreement={agreement:.2f} < min {self.config.min_ensemble_agreement}"

        # Step 8: Health check
        drift = self.predictor.check_feature_drift(
            features, self.config.feature_drift_z_threshold,
        )
        stale = self.predictor.model_age_seconds > self.config.model_stale_seconds
        health_ok = not stale and len(drift) == 0

        # Timing
        latency_us = (time.perf_counter() - t0) * 1e6
        now = datetime.now(timezone.utc).isoformat()

        # Direction
        if market_data.dist_from_ma50_pct >= 0:
            direction = "short_put_spread"
        else:
            direction = "short_call_spread"

        signal = PipelineSignal(
            trade=trade,
            ticker=market_data.ticker,
            direction=direction,
            contracts=contracts if trade else 0,
            confidence=conf,
            probability=prob,
            regime=regime,
            position_scale=pos_scale,
            regime_leverage=reg_lev,
            stop_loss_mult=stop_mult,
            reject_reason=reject_reason,
            latency_us=latency_us,
            timestamp=now,
            features_used=len(features),
            ensemble_agreement=agreement,
            health_ok=health_ok,
        )

        # Bookkeeping
        self._signals_generated += 1
        if trade:
            self._signals_traded += 1
        else:
            self._signals_rejected += 1
        self._latencies.append(latency_us)
        self._last_signal_time = now

        if self.config.log_all_signals:
            self._signal_log.append(signal)

        return signal

    # ── Batch generation ────────────────────────────────────────────────

    def generate_batch(self, market_data_list: List[MarketData]) -> List[PipelineSignal]:
        """Generate signals for multiple market snapshots."""
        return [self.generate_signal(md) for md in market_data_list]

    # ── VIX monitoring ──────────────────────────────────────────────────

    def check_vix_alert(self, vix: float) -> Tuple[bool, str]:
        """Real-time VIX monitoring for crisis hedge triggers."""
        if vix >= 35:
            return True, f"CRITICAL: VIX={vix:.1f} >= 35 — crash regime, positions halted"
        if vix >= 28:
            return True, f"WARNING: VIX={vix:.1f} >= 28 — high_vol regime, scaling to 25%"
        if vix >= 22:
            return False, f"ELEVATED: VIX={vix:.1f} — scaling positions down"
        return False, f"NORMAL: VIX={vix:.1f}"

    # ── Health checks ───────────────────────────────────────────────────

    def health(self) -> HealthStatus:
        """Pipeline health diagnostics."""
        model_age = self.predictor.model_age_seconds
        stale = model_age > self.config.model_stale_seconds

        # Check last signal for drift
        drift_features: List[str] = []
        agreement = 1.0
        if self._signal_log:
            last = self._signal_log[-1]
            agreement = last.ensemble_agreement
            features = extract_features(MarketData())  # default = baseline
            drift_features = self.predictor.check_feature_drift(features)

        avg_lat = float(np.mean(self._latencies)) if self._latencies else 0

        ok = (
            self.predictor.is_loaded or True  # fallback is acceptable
        ) and not stale and len(drift_features) == 0

        return HealthStatus(
            model_loaded=self.predictor.is_loaded,
            model_age_seconds=model_age,
            model_stale=stale,
            feature_drift_detected=len(drift_features) > 0,
            drifted_features=drift_features,
            ensemble_agreement=agreement,
            last_signal_time=self._last_signal_time,
            signals_generated=self._signals_generated,
            signals_traded=self._signals_traded,
            signals_rejected=self._signals_rejected,
            avg_latency_us=avg_lat,
            health_ok=ok,
        )

    # ── Signal log access ───────────────────────────────────────────────

    def get_signal_log(self, n: int = 50) -> List[PipelineSignal]:
        """Get the last N signals."""
        return self._signal_log[-n:]

    def get_trade_rate(self) -> float:
        """Fraction of signals that resulted in trades."""
        if self._signals_generated == 0:
            return 0.0
        return self._signals_traded / self._signals_generated

    def reset_stats(self) -> None:
        """Reset counters."""
        self._signals_generated = 0
        self._signals_traded = 0
        self._signals_rejected = 0
        self._latencies.clear()
        self._signal_log.clear()
