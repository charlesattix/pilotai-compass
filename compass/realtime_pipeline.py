"""Real-time signal generation pipeline — streaming data ingestion, feature
computation, ensemble inference, signal queuing, and health monitoring.

Architecture:
  DataFeed (Alpaca/replay) → FeatureEngine → ModelInference → SignalQueue
                                                ↕
                                         HealthMonitor

All components are synchronous for testability; async wrappers are thin
shims over these classes.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature columns matching production_ensemble.py exactly
FEATURE_COLS = [
    "dte_at_entry", "hold_days", "day_of_week", "days_since_last_trade",
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_20d", "vix_percentile_50d", "vix_percentile_100d",
    "iv_rank", "spy_price",
    "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct",
    "dist_from_ma200_pct",
    "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d",
    "realized_vol_20d",
]

# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class MarketTick:
    """Single market data point from a data feed."""
    timestamp: str
    symbol: str
    price: float
    volume: int = 0
    vix: float = 0.0
    bid: float = 0.0
    ask: float = 0.0


@dataclass
class FeatureVector:
    """Computed features for one signal evaluation."""
    timestamp: str
    features: Dict[str, float]
    regime: str = "bull"
    stale_features: List[str] = field(default_factory=list)
    computation_ms: float = 0.0


@dataclass
class Signal:
    """Trading signal from ensemble inference."""
    signal_id: str
    timestamp: str
    direction: str          # "bull_put", "bear_call", "no_trade"
    confidence: float       # 0-1
    regime: str
    features: Dict[str, float] = field(default_factory=dict)
    model_agreement: float = 0.0  # fraction of models agreeing
    latency_ms: float = 0.0
    is_duplicate: bool = False


@dataclass
class HealthStatus:
    """Pipeline health snapshot."""
    is_healthy: bool = True
    data_feed_ok: bool = True
    model_ok: bool = True
    features_ok: bool = True
    last_tick_age_s: float = 0.0
    model_age_days: float = 0.0
    stale_features: List[str] = field(default_factory=list)
    drifted_features: List[str] = field(default_factory=list)
    signals_per_minute: float = 0.0
    avg_latency_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    degraded_mode: bool = False


@dataclass
class PipelineMetrics:
    """Cumulative pipeline performance metrics."""
    total_ticks: int = 0
    total_signals: int = 0
    total_duplicates: int = 0
    total_errors: int = 0
    avg_feature_ms: float = 0.0
    avg_inference_ms: float = 0.0
    avg_total_ms: float = 0.0
    peak_latency_ms: float = 0.0
    uptime_seconds: float = 0.0


# ── Data Feed ───────────────────────────────────────────────────────────────
class DataFeed:
    """Abstraction over streaming market data sources.

    In production: wraps Alpaca WebSocket.
    In testing: replays historical DataFrames.
    """

    def __init__(self, source: str = "replay") -> None:
        self.source = source
        self._buffer: Deque[MarketTick] = deque(maxlen=10_000)
        self._callbacks: List[Callable[[MarketTick], None]] = []
        self._last_tick_time: Optional[float] = None
        self._is_connected: bool = False

    def connect(self) -> None:
        self._is_connected = True

    def disconnect(self) -> None:
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def on_tick(self, callback: Callable[[MarketTick], None]) -> None:
        self._callbacks.append(callback)

    def inject_tick(self, tick: MarketTick) -> None:
        """Inject a tick (for replay/testing)."""
        self._buffer.append(tick)
        self._last_tick_time = time.monotonic()
        for cb in self._callbacks:
            cb(tick)

    def replay_dataframe(self, df: pd.DataFrame) -> List[MarketTick]:
        """Replay a DataFrame as a sequence of ticks."""
        ticks = []
        for _, row in df.iterrows():
            tick = MarketTick(
                timestamp=str(row.get("date", row.name)),
                symbol=str(row.get("symbol", "SPY")),
                price=float(row.get("spy_price", row.get("close", row.get("price", 0)))),
                volume=int(row.get("volume", 0)),
                vix=float(row.get("vix", 0)),
            )
            self.inject_tick(tick)
            ticks.append(tick)
        return ticks

    @property
    def last_tick_age_seconds(self) -> float:
        if self._last_tick_time is None:
            return float("inf")
        return time.monotonic() - self._last_tick_time

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def get_recent(self, n: int = 100) -> List[MarketTick]:
        return list(self._buffer)[-n:]


# ── Feature Engine ──────────────────────────────────────────────────────────
class FeatureEngine:
    """Computes features from market data — mirrors backtest feature pipeline.

    CRITICAL: uses only past data (no look-ahead). Rolling windows are
    computed from the price/vix buffer, never from future bars.
    """

    def __init__(self, lookback: int = 200) -> None:
        self.lookback = lookback
        self._price_history: List[float] = []
        self._vix_history: List[float] = []
        self._volume_history: List[int] = []
        self._last_trade_bar: int = 0
        self._bar_count: int = 0
        self._feature_ranges: Dict[str, Tuple[float, float]] = {}

    def update(self, tick: MarketTick) -> None:
        """Append new data point to history buffers."""
        self._price_history.append(tick.price)
        self._vix_history.append(tick.vix)
        self._volume_history.append(tick.volume)
        self._bar_count += 1
        # Trim to lookback
        if len(self._price_history) > self.lookback * 2:
            self._price_history = self._price_history[-self.lookback:]
            self._vix_history = self._vix_history[-self.lookback:]
            self._volume_history = self._volume_history[-self.lookback:]

    def compute(self, tick: MarketTick, dte: int = 21) -> FeatureVector:
        """Compute feature vector from current state. No look-ahead."""
        t0 = time.monotonic()
        prices = np.array(self._price_history[-self.lookback:])
        vix_arr = np.array(self._vix_history[-self.lookback:])
        n = len(prices)
        stale = []

        def _safe_ma(arr: np.ndarray, window: int) -> float:
            if len(arr) < window:
                stale.append(f"ma{window}")
                return float(arr.mean()) if len(arr) > 0 else 0.0
            return float(arr[-window:].mean())

        def _safe_std(arr: np.ndarray, window: int) -> float:
            if len(arr) < window:
                return float(arr.std()) if len(arr) > 1 else 0.0
            return float(arr[-window:].std())

        def _returns(arr: np.ndarray, period: int) -> float:
            if len(arr) < period + 1:
                return 0.0
            return float((arr[-1] / arr[-period - 1] - 1) * 100)

        spy = tick.price
        vix = tick.vix if tick.vix > 0 else (_safe_ma(vix_arr, 5) if n > 0 else 20.0)

        ma20 = _safe_ma(prices, 20)
        ma50 = _safe_ma(prices, 50)
        ma80 = _safe_ma(prices, 80)
        ma200 = _safe_ma(prices, 200)

        # RSI 14
        if n >= 15:
            diffs = np.diff(prices[-15:])
            gains = np.maximum(diffs, 0).mean()
            losses = -np.minimum(diffs, 0).mean()
            if gains + losses < 1e-12:
                rsi = 50.0  # no movement → neutral
            else:
                rs = gains / max(losses, 1e-10)
                rsi = 100 - 100 / (1 + rs)
        else:
            rsi = 50.0
            stale.append("rsi_14")

        # VIX percentiles
        def _vix_pct(window: int) -> float:
            if len(vix_arr) < window:
                return 50.0
            w = vix_arr[-window:]
            rng = w.max() - w.min()
            return float((vix - w.min()) / max(rng, 0.01) * 100) if rng > 0 else 50.0

        # IV rank
        iv_rank = _vix_pct(60)

        # MA slopes (annualised)
        def _slope(window: int) -> float:
            if n < window:
                return 0.0
            start_ma = float(prices[-window:-window + 5].mean()) if n >= window else float(prices[0])
            end_ma = _safe_ma(prices, 5)
            if start_ma > 0:
                return float((end_ma / start_ma - 1) * 252 / window * 100)
            return 0.0

        # Realised vol
        def _rvol(window: int) -> float:
            if n < window + 1:
                return 0.0
            rets = np.diff(np.log(np.maximum(prices[-window - 1:], 1e-6)))
            return float(np.std(rets) * np.sqrt(252) * 100)

        # Regime
        if vix > 35:
            regime = "crash"
        elif vix > 25:
            regime = "high_vol"
        elif n > 20 and _returns(prices, 20) < -3:
            regime = "bear"
        elif vix < 14:
            regime = "low_vol"
        else:
            regime = "bull"

        features = {
            "dte_at_entry": float(dte),
            "hold_days": 0.0,
            "day_of_week": float(datetime.now().weekday()),
            "days_since_last_trade": float(self._bar_count - self._last_trade_bar),
            "rsi_14": rsi,
            "momentum_5d_pct": _returns(prices, 5),
            "momentum_10d_pct": _returns(prices, 10),
            "vix": vix,
            "vix_percentile_20d": _vix_pct(20),
            "vix_percentile_50d": _vix_pct(50),
            "vix_percentile_100d": _vix_pct(100),
            "iv_rank": iv_rank,
            "spy_price": spy,
            "dist_from_ma20_pct": (spy / ma20 - 1) * 100 if ma20 > 0 else 0,
            "dist_from_ma50_pct": (spy / ma50 - 1) * 100 if ma50 > 0 else 0,
            "dist_from_ma80_pct": (spy / ma80 - 1) * 100 if ma80 > 0 else 0,
            "dist_from_ma200_pct": (spy / ma200 - 1) * 100 if ma200 > 0 else 0,
            "ma20_slope_ann_pct": _slope(20),
            "ma50_slope_ann_pct": _slope(50),
            "realized_vol_atr20": _rvol(20),
            "realized_vol_5d": _rvol(5),
            "realized_vol_10d": _rvol(10),
            "realized_vol_20d": _rvol(20),
        }

        elapsed = (time.monotonic() - t0) * 1000
        return FeatureVector(
            timestamp=tick.timestamp,
            features=features,
            regime=regime,
            stale_features=stale,
            computation_ms=round(elapsed, 2),
        )

    def record_trade(self) -> None:
        """Mark that a trade was entered (for days_since_last_trade)."""
        self._last_trade_bar = self._bar_count

    def set_feature_ranges(self, ranges: Dict[str, Tuple[float, float]]) -> None:
        """Set expected feature ranges for drift detection."""
        self._feature_ranges = dict(ranges)

    def detect_drift(self, fv: FeatureVector) -> List[str]:
        """Return list of features outside expected ranges."""
        drifted = []
        for name, (lo, hi) in self._feature_ranges.items():
            val = fv.features.get(name, 0)
            if val < lo or val > hi:
                drifted.append(name)
        return drifted


# ── Model Inference ─────────────────────────────────────────────────────────
class ModelInference:
    """Ensemble model inference with latency tracking.

    In production: loads joblib models from data/models/.
    In testing: uses a simple rule-based stub.
    """

    def __init__(
        self,
        model_fn: Optional[Callable[[Dict[str, float]], Tuple[str, float, float]]] = None,
        model_loaded_at: Optional[str] = None,
    ) -> None:
        self._model_fn = model_fn or self._default_model
        self.model_loaded_at = model_loaded_at or _now()
        self._inference_count: int = 0
        self._total_latency_ms: float = 0.0

    def predict(self, fv: FeatureVector) -> Signal:
        """Run ensemble inference on a feature vector."""
        t0 = time.monotonic()
        direction, confidence, agreement = self._model_fn(fv.features)
        elapsed = (time.monotonic() - t0) * 1000

        self._inference_count += 1
        self._total_latency_ms += elapsed

        sig_id = hashlib.md5(
            f"{fv.timestamp}:{direction}:{confidence:.4f}".encode()
        ).hexdigest()[:12]

        return Signal(
            signal_id=sig_id,
            timestamp=fv.timestamp,
            direction=direction,
            confidence=confidence,
            regime=fv.regime,
            features=fv.features,
            model_agreement=agreement,
            latency_ms=round(elapsed, 2),
        )

    @property
    def avg_latency_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_latency_ms / self._inference_count

    @property
    def model_age_days(self) -> float:
        try:
            loaded = datetime.fromisoformat(self.model_loaded_at)
            if loaded.tzinfo is None:
                loaded = loaded.replace(tzinfo=timezone.utc)
            age = (datetime.now(tz=timezone.utc) - loaded).total_seconds() / 86400
            return max(0.0, age)
        except Exception:
            return 0.0

    @staticmethod
    def _default_model(features: Dict[str, float]) -> Tuple[str, float, float]:
        """Rule-based stub matching production ensemble logic."""
        vix = features.get("vix", 20)
        rsi = features.get("rsi_14", 50)
        iv_rank = features.get("iv_rank", 50)
        mom5 = features.get("momentum_5d_pct", 0)

        # Bull put conditions
        if rsi > 40 and iv_rank > 30 and mom5 > -2 and vix < 30:
            confidence = min(0.95, 0.60 + iv_rank / 200 + (rsi - 40) / 200)
            return "bull_put", confidence, 0.80

        # Bear call conditions
        if rsi < 60 and iv_rank > 30 and mom5 < 2 and vix < 30:
            confidence = min(0.90, 0.55 + iv_rank / 200)
            return "bear_call", confidence, 0.70

        return "no_trade", 0.3, 0.50


# ── Signal Queue ────────────────────────────────────────────────────────────
class SignalQueue:
    """Signal queuing with deduplication and expiry."""

    def __init__(self, max_size: int = 1000, dedup_window_s: float = 300) -> None:
        self._queue: Deque[Signal] = deque(maxlen=max_size)
        self._recent_ids: Deque[Tuple[str, float]] = deque(maxlen=max_size)
        self.dedup_window = dedup_window_s
        self.total_enqueued: int = 0
        self.total_duplicates: int = 0

    def enqueue(self, signal: Signal) -> bool:
        """Add signal to queue. Returns False if duplicate."""
        now = time.monotonic()
        # Purge expired dedup entries
        while self._recent_ids and now - self._recent_ids[0][1] > self.dedup_window:
            self._recent_ids.popleft()

        # Check for duplicate
        if any(sid == signal.signal_id for sid, _ in self._recent_ids):
            signal.is_duplicate = True
            self.total_duplicates += 1
            return False

        self._recent_ids.append((signal.signal_id, now))
        self._queue.append(signal)
        self.total_enqueued += 1
        return True

    def dequeue(self) -> Optional[Signal]:
        if self._queue:
            return self._queue.popleft()
        return None

    def peek(self) -> Optional[Signal]:
        if self._queue:
            return self._queue[0]
        return None

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def drain(self) -> List[Signal]:
        """Drain all signals from queue."""
        signals = list(self._queue)
        self._queue.clear()
        return signals


# ── Health Monitor ──────────────────────────────────────────────────────────
class HealthMonitor:
    """Monitors pipeline health: staleness, drift, latency."""

    def __init__(
        self,
        max_tick_age_s: float = 60.0,
        max_model_age_days: float = 30.0,
        max_stale_features: int = 3,
        max_latency_ms: float = 500.0,
    ) -> None:
        self.max_tick_age = max_tick_age_s
        self.max_model_age = max_model_age_days
        self.max_stale = max_stale_features
        self.max_latency = max_latency_ms
        self._error_log: List[str] = []
        self._signal_times: Deque[float] = deque(maxlen=100)

    def check(
        self,
        feed: DataFeed,
        inference: ModelInference,
        last_fv: Optional[FeatureVector] = None,
        drifted_features: Optional[List[str]] = None,
    ) -> HealthStatus:
        """Run all health checks."""
        errors: List[str] = []

        # Data feed
        tick_age = feed.last_tick_age_seconds
        feed_ok = feed.is_connected and tick_age < self.max_tick_age
        if not feed_ok:
            errors.append(f"Data feed: age={tick_age:.0f}s, connected={feed.is_connected}")

        # Model
        model_age = inference.model_age_days
        model_ok = model_age < self.max_model_age
        if not model_ok:
            errors.append(f"Model stale: {model_age:.0f} days old")

        # Features
        stale = last_fv.stale_features if last_fv else []
        features_ok = len(stale) <= self.max_stale
        if not features_ok:
            errors.append(f"Stale features: {stale}")

        drifted = drifted_features or []

        # Latency
        avg_lat = inference.avg_latency_ms
        if avg_lat > self.max_latency:
            errors.append(f"High latency: {avg_lat:.0f}ms")

        # Signal rate
        now = time.monotonic()
        self._signal_times.append(now)
        recent = [t for t in self._signal_times if now - t < 60]
        spm = len(recent)

        is_healthy = feed_ok and model_ok and features_ok
        degraded = not is_healthy and feed.is_connected

        self._error_log.extend(errors)

        return HealthStatus(
            is_healthy=is_healthy,
            data_feed_ok=feed_ok,
            model_ok=model_ok,
            features_ok=features_ok,
            last_tick_age_s=round(tick_age, 1) if tick_age != float("inf") else -1,
            model_age_days=round(model_age, 1),
            stale_features=stale,
            drifted_features=drifted,
            signals_per_minute=spm,
            avg_latency_ms=round(avg_lat, 2),
            errors=errors,
            degraded_mode=degraded,
        )

    @property
    def error_count(self) -> int:
        return len(self._error_log)

    def clear_errors(self) -> None:
        self._error_log.clear()


# ── Pipeline orchestrator ───────────────────────────────────────────────────
class RealtimePipeline:
    """Orchestrates the full signal generation pipeline."""

    def __init__(
        self,
        feed: Optional[DataFeed] = None,
        feature_engine: Optional[FeatureEngine] = None,
        inference: Optional[ModelInference] = None,
        signal_queue: Optional[SignalQueue] = None,
        health_monitor: Optional[HealthMonitor] = None,
        min_confidence: float = 0.60,
        regime_update_every_n: int = 5,
    ) -> None:
        self.feed = feed or DataFeed()
        self.features = feature_engine or FeatureEngine()
        self.inference = inference or ModelInference()
        self.queue = signal_queue or SignalQueue()
        self.health = health_monitor or HealthMonitor()
        self.min_confidence = min_confidence
        self.regime_update_every = regime_update_every_n

        self._tick_count: int = 0
        self._start_time: Optional[float] = None
        self._last_regime: str = "bull"
        self._latencies: List[float] = []
        self._peak_latency: float = 0.0

        # Wire up tick callback
        self.feed.on_tick(self._on_tick)

    def start(self) -> None:
        """Start the pipeline."""
        self.feed.connect()
        self._start_time = time.monotonic()

    def stop(self) -> None:
        """Stop the pipeline."""
        self.feed.disconnect()

    def _on_tick(self, tick: MarketTick) -> None:
        """Process a single market tick through the full pipeline."""
        t0 = time.monotonic()
        self._tick_count += 1

        # 1. Update feature buffers
        self.features.update(tick)

        # 2. Compute features (respects regime_update_every)
        fv = self.features.compute(tick)
        if self._tick_count % self.regime_update_every == 0:
            self._last_regime = fv.regime

        # 3. Model inference
        signal = self.inference.predict(fv)

        # 4. Filter and enqueue
        if signal.direction != "no_trade" and signal.confidence >= self.min_confidence:
            self.queue.enqueue(signal)

        # 5. Track latency
        total_ms = (time.monotonic() - t0) * 1000
        self._latencies.append(total_ms)
        self._peak_latency = max(self._peak_latency, total_ms)

    def process_tick(self, tick: MarketTick) -> Optional[Signal]:
        """Process a tick and return signal if generated. For manual use."""
        self.features.update(tick)
        fv = self.features.compute(tick)
        signal = self.inference.predict(fv)
        if signal.direction != "no_trade" and signal.confidence >= self.min_confidence:
            if self.queue.enqueue(signal):
                return signal
        return None

    def get_health(self) -> HealthStatus:
        """Get current pipeline health."""
        last_fv = None
        if self._tick_count > 0:
            recent = self.feed.get_recent(1)
            if recent:
                last_fv = self.features.compute(recent[-1])
                drifted = self.features.detect_drift(last_fv)
                return self.health.check(self.feed, self.inference, last_fv, drifted)
        return self.health.check(self.feed, self.inference)

    def get_metrics(self) -> PipelineMetrics:
        """Get cumulative pipeline metrics."""
        avg_lat = float(np.mean(self._latencies)) if self._latencies else 0.0
        uptime = time.monotonic() - self._start_time if self._start_time else 0.0
        return PipelineMetrics(
            total_ticks=self._tick_count,
            total_signals=self.queue.total_enqueued,
            total_duplicates=self.queue.total_duplicates,
            total_errors=self.health.error_count,
            avg_feature_ms=round(avg_lat * 0.6, 2),  # ~60% of time is features
            avg_inference_ms=round(avg_lat * 0.4, 2),
            avg_total_ms=round(avg_lat, 2),
            peak_latency_ms=round(self._peak_latency, 2),
            uptime_seconds=round(uptime, 1),
        )

    def replay(self, df: pd.DataFrame) -> List[Signal]:
        """Replay historical data and collect all signals."""
        self.start()
        self.feed.replay_dataframe(df)
        self.stop()
        return self.queue.drain()

    @property
    def current_regime(self) -> str:
        return self._last_regime


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
