"""Multi-timeframe signal fusion — combines intraday (5min), daily (1D), and
weekly (1W) signals via attention-weighted fusion for more robust predictions.

Architecture:
  Raw bars → TimeframeFeatureExtractor (per TF) → Normaliser → AttentionFusion → UnifiedSignal

The attention mechanism learns which timeframe is most informative in the
current market regime, producing a single confidence-weighted signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TIMEFRAMES = ["5min", "1D", "1W"]
TRADING_DAYS = 252


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class TimeframeFeatures:
    """Extracted features for one timeframe."""
    timeframe: str
    momentum: float        # returns over lookback
    mean_reversion: float  # z-score of price vs rolling mean
    volatility: float      # realised vol (annualised)
    trend: float           # slope of rolling regression
    rsi: float             # 0-100
    raw_signal: float      # combined into -1..+1 signal


@dataclass
class NormalisedSignal:
    """Signal after z-score normalisation to common scale."""
    timeframe: str
    signal: float          # -1 (strong bearish) to +1 (strong bullish)
    confidence: float      # 0-1 reliability estimate
    volatility_regime: str # low / normal / high


@dataclass
class AttentionWeights:
    """Learned attention weights across timeframes."""
    weights: Dict[str, float]      # timeframe → weight (sums to 1)
    regime_context: str            # what drove the weighting
    entropy: float                 # higher = more uniform = less certain


@dataclass
class FusedSignal:
    """Final unified signal from attention fusion."""
    signal: float                  # -1..+1
    confidence: float              # 0-1
    direction: str                 # "bullish", "bearish", "neutral"
    attention: AttentionWeights
    components: List[NormalisedSignal]
    agreement: float               # fraction of TFs agreeing on direction


@dataclass
class BacktestComparison:
    """Performance of one signal variant."""
    name: str
    sharpe: float
    cagr_pct: float
    max_dd_pct: float
    hit_rate_pct: float
    n_signals: int
    total_return_pct: float


@dataclass
class FusionResult:
    """Complete multi-timeframe fusion output."""
    fused_signals: List[FusedSignal] = field(default_factory=list)
    comparisons: List[BacktestComparison] = field(default_factory=list)
    best_variant: str = ""
    sharpe_improvement_pct: float = 0.0
    dd_reduction_pct: float = 0.0
    hit_rate_improvement_pct: float = 0.0
    generated_at: str = ""


# ── Feature extraction ──────────────────────────────────────────────────────
class TimeframeFeatureExtractor:
    """Extracts features at a given timeframe resolution."""

    def __init__(self, timeframe: str, lookback: int = 20) -> None:
        self.timeframe = timeframe
        self.lookback = lookback

    def extract(self, prices: np.ndarray) -> TimeframeFeatures:
        """Extract features from a price array at this timeframe's resolution."""
        n = len(prices)
        lb = min(self.lookback, n - 1)
        if n < 5:
            return TimeframeFeatures(self.timeframe, 0, 0, 0, 0, 50, 0)

        # Momentum: return over lookback
        momentum = (prices[-1] / prices[-lb - 1] - 1) if prices[-lb - 1] > 0 else 0

        # Mean reversion: z-score of current price vs rolling mean
        rolling_mean = float(np.mean(prices[-lb:]))
        rolling_std = float(np.std(prices[-lb:]))
        mean_rev = (prices[-1] - rolling_mean) / max(rolling_std, 1e-8)

        # Volatility: annualised from log returns
        if n >= 3:
            log_ret = np.diff(np.log(np.maximum(prices[-lb - 1:], 1e-8)))
            vol = float(np.std(log_ret))
            # Annualise based on timeframe
            if self.timeframe == "5min":
                vol *= np.sqrt(252 * 78)  # 78 five-min bars per day
            elif self.timeframe == "1D":
                vol *= np.sqrt(252)
            elif self.timeframe == "1W":
                vol *= np.sqrt(52)
        else:
            vol = 0.0

        # Trend: linear regression slope
        x = np.arange(lb)
        y = prices[-lb:]
        if len(x) == len(y) and lb > 1:
            slope = float(np.polyfit(x, y, 1)[0])
            trend = slope / max(prices[-1], 1e-8) * lb  # normalised
        else:
            trend = 0.0

        # RSI
        rsi = self._compute_rsi(prices, min(14, lb))

        # Raw signal: combine features
        raw = 0.0
        raw += np.clip(momentum * 10, -1, 1) * 0.30    # momentum contribution
        raw += np.clip(-mean_rev / 3, -1, 1) * 0.20    # mean-rev (contrarian)
        raw += np.clip(trend * 5, -1, 1) * 0.30         # trend contribution
        raw += np.clip((rsi - 50) / 50, -1, 1) * 0.20   # RSI contribution
        raw = float(np.clip(raw, -1, 1))

        return TimeframeFeatures(
            timeframe=self.timeframe,
            momentum=round(momentum, 6),
            mean_reversion=round(float(mean_rev), 4),
            volatility=round(vol, 4),
            trend=round(trend, 6),
            rsi=round(rsi, 1),
            raw_signal=round(raw, 4),
        )

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int) -> float:
        if len(prices) < period + 1:
            return 50.0
        diffs = np.diff(prices[-period - 1:])
        gains = np.maximum(diffs, 0).mean()
        losses = -np.minimum(diffs, 0).mean()
        if gains + losses < 1e-12:
            return 50.0
        rs = gains / max(losses, 1e-10)
        return float(100 - 100 / (1 + rs))


# ── Normalisation ───────────────────────────────────────────────────────────
def normalise_signal(tf_features: TimeframeFeatures) -> NormalisedSignal:
    """Normalise raw signal to common -1..+1 scale with confidence."""
    sig = tf_features.raw_signal  # already -1..+1

    # Confidence from agreement of sub-signals
    mom_dir = np.sign(tf_features.momentum)
    trend_dir = np.sign(tf_features.trend)
    rsi_dir = 1 if tf_features.rsi > 55 else (-1 if tf_features.rsi < 45 else 0)

    signs = [mom_dir, trend_dir, rsi_dir]
    non_zero = [s for s in signs if s != 0]
    if non_zero:
        agreement = abs(sum(non_zero)) / len(non_zero)
    else:
        agreement = 0.0
    confidence = min(1.0, agreement * 0.6 + abs(sig) * 0.4)

    # Vol regime
    vol = tf_features.volatility
    if vol > 0.30:
        vol_regime = "high"
    elif vol < 0.12:
        vol_regime = "low"
    else:
        vol_regime = "normal"

    return NormalisedSignal(
        timeframe=tf_features.timeframe,
        signal=round(sig, 4),
        confidence=round(confidence, 4),
        volatility_regime=vol_regime,
    )


# ── Attention fusion ────────────────────────────────────────────────────────
class AttentionFusion:
    """Attention-weighted fusion of multi-timeframe signals.

    Attention weights are computed from:
      1. Signal confidence (higher confidence → higher weight)
      2. Volatility regime (low-vol → trust longer TFs; high-vol → trust shorter)
      3. Historical accuracy (learnable via exponential moving average)
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.05,
    ) -> None:
        self.base_weights = base_weights or {"5min": 0.20, "1D": 0.50, "1W": 0.30}
        self.lr = learning_rate
        # Accuracy tracking (EMA)
        self._accuracy: Dict[str, float] = {tf: 0.5 for tf in TIMEFRAMES}
        self._n_updates: int = 0

    def fuse(self, signals: List[NormalisedSignal]) -> FusedSignal:
        """Compute attention-weighted fusion of normalised signals."""
        if not signals:
            return FusedSignal(0, 0, "neutral",
                               AttentionWeights({}, "", 0), [], 0)

        # Compute raw attention scores
        scores: Dict[str, float] = {}
        for s in signals:
            base = self.base_weights.get(s.timeframe, 0.33)
            conf_boost = s.confidence * 0.5
            acc_boost = self._accuracy.get(s.timeframe, 0.5) * 0.3

            # Vol-regime adjustment
            if s.volatility_regime == "high":
                # High vol → trust intraday more (faster reaction)
                tf_boost = 0.2 if s.timeframe == "5min" else -0.1
            elif s.volatility_regime == "low":
                # Low vol → trust weekly more (noise in short TFs)
                tf_boost = 0.2 if s.timeframe == "1W" else -0.05
            else:
                tf_boost = 0.0

            scores[s.timeframe] = max(0.01, base + conf_boost + acc_boost + tf_boost)

        # Softmax normalisation
        total = sum(scores.values())
        weights = {tf: s / total for tf, s in scores.items()}

        # Weighted signal
        fused_sig = sum(
            weights.get(s.timeframe, 0) * s.signal for s in signals
        )
        fused_sig = float(np.clip(fused_sig, -1, 1))

        # Fused confidence
        fused_conf = sum(
            weights.get(s.timeframe, 0) * s.confidence for s in signals
        )

        # Direction
        if fused_sig > 0.15:
            direction = "bullish"
        elif fused_sig < -0.15:
            direction = "bearish"
        else:
            direction = "neutral"

        # Agreement
        dirs = [1 if s.signal > 0.1 else (-1 if s.signal < -0.1 else 0) for s in signals]
        non_zero = [d for d in dirs if d != 0]
        if non_zero:
            agreement = abs(sum(non_zero)) / len(non_zero)
        else:
            agreement = 0.0

        # Entropy of weights (0 = one TF dominates, log(n) = uniform)
        w_arr = np.array(list(weights.values()))
        w_arr = w_arr[w_arr > 0]
        entropy = float(-np.sum(w_arr * np.log(w_arr + 1e-10)))

        # Regime context
        dominant = max(weights, key=weights.get)
        context = f"{dominant} dominant ({weights[dominant]:.0%})"

        return FusedSignal(
            signal=round(fused_sig, 4),
            confidence=round(fused_conf, 4),
            direction=direction,
            attention=AttentionWeights(
                weights={k: round(v, 4) for k, v in weights.items()},
                regime_context=context,
                entropy=round(entropy, 4),
            ),
            components=list(signals),
            agreement=round(agreement, 2),
        )

    def update_accuracy(self, timeframe: str, was_correct: bool) -> None:
        """Update EMA accuracy for a timeframe after observing outcome."""
        old = self._accuracy.get(timeframe, 0.5)
        self._accuracy[timeframe] = old * (1 - self.lr) + float(was_correct) * self.lr
        self._n_updates += 1

    @property
    def learned_accuracy(self) -> Dict[str, float]:
        return dict(self._accuracy)


# ── Backtest ────────────────────────────────────────────────────────────────
class MultiTimeframeBacktest:
    """Backtest fused vs individual timeframe signals."""

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        signal_threshold: float = 0.15,
        position_size_pct: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.starting_capital = starting_capital
        self.threshold = signal_threshold
        self.pos_size = position_size_pct
        self.rng = np.random.RandomState(seed)

    def run(self, daily_prices: pd.DataFrame) -> FusionResult:
        """Run comparison backtest.

        daily_prices: DataFrame with 'close' column (daily resolution).
        Intraday and weekly are synthesised from daily for testing.
        """
        n = len(daily_prices)
        if n < 100:
            return FusionResult(generated_at=_now())

        close = daily_prices["close"].values if "close" in daily_prices.columns else daily_prices.iloc[:, 0].values

        # Build multi-TF price arrays
        intraday = close  # proxy: same as daily (in production, actual 5min bars)
        daily = close
        weekly = close[::5] if len(close) > 10 else close  # subsample

        # Extractors
        ext_5m = TimeframeFeatureExtractor("5min", lookback=20)
        ext_1d = TimeframeFeatureExtractor("1D", lookback=20)
        ext_1w = TimeframeFeatureExtractor("1W", lookback=10)

        fusion = AttentionFusion()
        warmup = 30

        # Run each variant
        variants = {
            "5min_only": [],
            "1D_only": [],
            "1W_only": [],
            "fused": [],
        }
        fused_signals: List[FusedSignal] = []

        for i in range(warmup, n):
            # Extract features
            f5 = ext_5m.extract(intraday[:i + 1])
            f1d = ext_1d.extract(daily[:i + 1])
            wi = min(i // 5, len(weekly) - 1)
            f1w = ext_1w.extract(weekly[:wi + 1]) if wi >= 3 else TimeframeFeatures("1W", 0, 0, 0, 0, 50, 0)

            # Normalise
            n5 = normalise_signal(f5)
            n1d = normalise_signal(f1d)
            n1w = normalise_signal(f1w)

            # Fuse
            fs = fusion.fuse([n5, n1d, n1w])
            fused_signals.append(fs)

            # Store signals for each variant
            variants["5min_only"].append(n5.signal)
            variants["1D_only"].append(n1d.signal)
            variants["1W_only"].append(n1w.signal)
            variants["fused"].append(fs.signal)

            # Update accuracy based on next-day return
            if i < n - 1:
                next_ret = (close[i + 1] / close[i]) - 1
                for tf, sig in [("5min", n5), ("1D", n1d), ("1W", n1w)]:
                    correct = (sig.signal > 0 and next_ret > 0) or (sig.signal < 0 and next_ret < 0)
                    fusion.update_accuracy(tf, correct)

        # Compute performance for each variant
        returns = np.diff(close[warmup:]) / close[warmup:-1]
        comparisons: List[BacktestComparison] = []

        for name, signals in variants.items():
            sig_arr = np.array(signals[:len(returns)])
            comp = self._evaluate(name, sig_arr, returns)
            comparisons.append(comp)

        # Compute improvements
        fused_comp = next(c for c in comparisons if c.name == "fused")
        individual_sharpes = [c.sharpe for c in comparisons if c.name != "fused"]
        best_individual = max(individual_sharpes) if individual_sharpes else 0

        sharpe_imp = ((fused_comp.sharpe - best_individual) / max(abs(best_individual), 0.01)) * 100 if best_individual != 0 else 0

        individual_dds = [c.max_dd_pct for c in comparisons if c.name != "fused"]
        worst_individual_dd = max(individual_dds) if individual_dds else 0
        dd_reduction = ((worst_individual_dd - fused_comp.max_dd_pct) / max(worst_individual_dd, 0.01)) * 100

        individual_hrs = [c.hit_rate_pct for c in comparisons if c.name != "fused"]
        best_hr = max(individual_hrs) if individual_hrs else 0
        hr_imp = fused_comp.hit_rate_pct - best_hr

        best_name = max(comparisons, key=lambda c: c.sharpe).name

        return FusionResult(
            fused_signals=fused_signals,
            comparisons=comparisons,
            best_variant=best_name,
            sharpe_improvement_pct=round(sharpe_imp, 1),
            dd_reduction_pct=round(dd_reduction, 1),
            hit_rate_improvement_pct=round(hr_imp, 1),
            generated_at=_now(),
        )

    def _evaluate(
        self, name: str, signals: np.ndarray, returns: np.ndarray,
    ) -> BacktestComparison:
        n = min(len(signals), len(returns))
        signals = signals[:n]
        returns = returns[:n]

        capital = self.starting_capital
        peak = capital
        max_dd = 0.0
        pnl_list = []
        correct = 0
        n_signals = 0

        for i in range(n):
            sig = signals[i]
            if abs(sig) < self.threshold:
                continue

            position = np.sign(sig) * self.pos_size
            day_pnl = position * returns[i] * capital
            capital += day_pnl
            pnl_list.append(day_pnl)
            n_signals += 1

            if (sig > 0 and returns[i] > 0) or (sig < 0 and returns[i] < 0):
                correct += 1

            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        total_return = (capital - self.starting_capital) / self.starting_capital * 100
        years = n / TRADING_DAYS
        cagr = ((capital / self.starting_capital) ** (1 / years) - 1) * 100 if years > 0 and capital > 0 else 0

        dr = np.array(pnl_list) if pnl_list else np.array([0.0])
        sharpe = float(dr.mean() / dr.std() * np.sqrt(TRADING_DAYS)) if dr.std() > 0 else 0
        hit_rate = correct / n_signals * 100 if n_signals > 0 else 0

        return BacktestComparison(
            name=name,
            sharpe=round(sharpe, 2),
            cagr_pct=round(cagr, 2),
            max_dd_pct=round(max_dd * 100, 2),
            hit_rate_pct=round(hit_rate, 1),
            n_signals=n_signals,
            total_return_pct=round(total_return, 2),
        )


# ── Synthetic data ──────────────────────────────────────────────────────────
def generate_price_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range("2020-01-02", periods=n)
    close = np.zeros(n)
    close[0] = 320.0
    for i in range(1, n):
        close[i] = close[i - 1] * np.exp(rng.randn() * 0.012 + 0.0003)
    return pd.DataFrame({"close": close}, index=idx)


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
