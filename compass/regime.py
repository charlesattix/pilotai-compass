"""
Regime classifier — tags each trading day with a market regime.

Uses VIX levels + SPY price trends per MASTERPLAN spec:
  - BULL:         SPY trending up, VIX < 20
  - BEAR:         SPY trending down, VIX > 25
  - HIGH_VOL:     VIX > 30 (any direction)
  - LOW_VOL:      VIX < 15, no strong trend
  - CRASH:        VIX > 40, sharp decline

Enhanced features (all off by default for backward compatibility):
  - Configurable thresholds via config dict
  - hysteresis_days: prevent regime flip-flop within N days (default 0 = off)
  - vix3m_crash_threshold: VIX/VIX3M ratio above this triggers CRASH (default None = off)
  - Shift-by-1 lookahead protection in classify_series()

Also contains ComboRegimeDetector (absorbed from ml/combo_regime_detector.py).
"""

from enum import Enum
from typing import Dict, Optional

import pandas as pd


class Regime(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    CRASH = "crash"


# Strategy recommendations per regime (for logging / allocation hints)
REGIME_INFO = {
    Regime.BULL: {
        "label": "Bull",
        "strategies": ["credit_spread", "momentum_swing", "debit_spread"],
        "risk": "medium",
    },
    Regime.BEAR: {
        "label": "Bear",
        "strategies": ["credit_spread", "gamma_lotto", "debit_spread"],
        "risk": "high",
    },
    Regime.HIGH_VOL: {
        "label": "High Volatility",
        "strategies": ["iron_condor", "straddle_strangle", "credit_spread"],
        "risk": "medium-high",
    },
    Regime.LOW_VOL: {
        "label": "Low Vol Sideways",
        "strategies": ["calendar_spread", "iron_condor", "gamma_lotto"],
        "risk": "low",
    },
    Regime.CRASH: {
        "label": "Crash",
        "strategies": ["gamma_lotto"],
        "risk": "extreme",
    },
}


class RegimeClassifier:
    """Rule-based regime classifier using VIX + price trend.

    All methods work on pre-loaded data (no yfinance calls), so they
    can be called inside the backtester day loop without network overhead.

    Args:
        trend_window: Number of days for the trend moving average.
        trend_threshold: Minimum slope (annualized %) to consider trending.
        config: Optional dict with configurable thresholds:
            - hysteresis_days (int): Min days between regime changes (default 0 = off)
            - vix3m_crash_threshold (float): VIX/VIX3M ratio above this → CRASH
              signal when VIX > 25. (default None = off)
    """

    def __init__(
        self,
        trend_window: int = 50,
        trend_threshold: float = 5.0,
        config: Optional[dict] = None,
    ):
        self.trend_window = trend_window
        self.trend_threshold = trend_threshold  # annualized % slope

        cfg = config or {}

        # Hysteresis: prevent regime flip-flop within N trading days.
        # Default 0 = off (backward compatible with original RegimeClassifier).
        self.hysteresis_days = int(cfg.get('hysteresis_days', 0))

        # VIX term structure crash signal: VIX/VIX3M ratio above this
        # threshold (with VIX > 25) triggers CRASH classification.
        # Default None = off (backward compatible).
        _vix3m = cfg.get('vix3m_crash_threshold', None)
        self.vix3m_crash_threshold: Optional[float] = (
            float(_vix3m) if _vix3m is not None else None
        )

    def classify(
        self,
        vix: float,
        spy_prices: pd.Series,
        date: pd.Timestamp,
        vix3m: Optional[float] = None,
    ) -> Regime:
        """Classify the regime for a single trading day.

        Args:
            vix: Current VIX close value.
            spy_prices: SPY close series up to (and including) this date.
            date: Current date (for debugging; not used in logic).
            vix3m: Optional VIX3M close value. When provided along with
                vix3m_crash_threshold config, enables term structure crash
                detection (VIX/VIX3M > threshold + VIX > 25 → CRASH).

        Returns:
            Regime enum value.
        """
        # VIX/VIX3M term structure crash signal (only when enabled via config)
        if (
            self.vix3m_crash_threshold is not None
            and vix3m is not None
            and vix3m > 0
            and vix > 25
        ):
            ratio = vix / vix3m
            if ratio > self.vix3m_crash_threshold:
                return Regime.CRASH

        # Crash takes highest priority — VIX > 40 + sharp decline
        if vix > 40:
            if self._is_declining(spy_prices):
                return Regime.CRASH
            # VIX > 40 but no sharp decline → still high vol
            return Regime.HIGH_VOL

        # High vol — VIX > 30 (any direction)
        if vix > 30:
            return Regime.HIGH_VOL

        # Determine trend direction
        trend = self._trend_direction(spy_prices)

        # Bear — VIX > 25, SPY trending down
        if vix > 25 and trend < 0:
            return Regime.BEAR

        # Bull — VIX < 20, SPY trending up
        if vix < 20 and trend > 0:
            return Regime.BULL

        # Low vol sideways — VIX < 15, no strong trend
        if vix < 15 and abs(trend) == 0:
            return Regime.LOW_VOL

        # Ambiguous zones: use trend + VIX together
        if trend > 0:
            return Regime.BULL
        elif trend < 0:
            if vix > 22:
                return Regime.BEAR
            return Regime.BULL  # mild pullback, low VIX → still constructive
        else:
            # No trend
            if vix < 18:
                return Regime.LOW_VOL
            return Regime.BULL  # neutral → default to constructive

    def classify_series(
        self,
        spy_data: pd.DataFrame,
        vix_series: pd.Series,
        vix3m_series: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Tag every trading day with a regime.

        Uses shift-by-1 lookahead protection: regime on date T uses VIX
        and prices through T-1 only.

        When hysteresis_days > 0, regime changes are suppressed for that
        many trading days after the last change to prevent rapid flipping.

        Args:
            spy_data: SPY OHLCV DataFrame with DatetimeIndex.
            vix_series: VIX close Series with DatetimeIndex.
            vix3m_series: Optional VIX3M close Series. When provided along
                with vix3m_crash_threshold config, enables term structure
                crash detection.

        Returns:
            pd.Series of Regime values indexed by date.
        """
        close = spy_data["Close"]

        # Shift-by-1 lookahead protection: use yesterday's VIX and prices
        vix_prev = vix_series.shift(1)
        close_prev = close.shift(1)

        # VIX3M: reindex + shift-by-1 if provided and feature enabled
        vix3m_prev = None
        if vix3m_series is not None and self.vix3m_crash_threshold is not None:
            vix3m_prev = vix3m_series.reindex(close.index).shift(1)

        regimes: Dict[pd.Timestamp, Regime] = {}
        current_regime: Optional[Regime] = None
        last_change_idx = -1

        for idx, date in enumerate(close.index):
            vix_val = vix_prev.get(date, 20.0)
            if isinstance(vix_val, pd.Series):
                vix_val = float(vix_val.iloc[0]) if len(vix_val) > 0 else 20.0
            else:
                vix_val = float(vix_val)

            if pd.isna(vix_val):
                vix_val = 20.0

            # VIX3M value for this date (None if not configured)
            vix3m_val = None
            if vix3m_prev is not None:
                v = vix3m_prev.get(date)
                if v is not None and not pd.isna(v):
                    vix3m_val = float(v)

            # Use close_prev for price history (lookahead-safe)
            prices_to_date = close_prev.loc[:date].dropna()
            if len(prices_to_date) < 2:
                raw_regime = Regime.BULL
            else:
                raw_regime = self.classify(
                    vix_val, prices_to_date, date, vix3m=vix3m_val,
                )

            # Hysteresis: suppress regime changes within hysteresis_days
            if self.hysteresis_days > 0:
                if current_regime is not None and raw_regime != current_regime:
                    days_since_change = idx - last_change_idx
                    if (
                        last_change_idx >= 0
                        and days_since_change < self.hysteresis_days
                    ):
                        regimes[date] = current_regime
                        continue
                    current_regime = raw_regime
                    last_change_idx = idx
                elif current_regime is None:
                    current_regime = raw_regime
                    last_change_idx = idx
            else:
                # No hysteresis — raw regime is final
                current_regime = raw_regime

            regimes[date] = current_regime

        return pd.Series(regimes, name="regime")

    # ------------------------------------------------------------------
    # Trend helpers
    # ------------------------------------------------------------------

    def _trend_direction(self, prices: pd.Series) -> int:
        """Determine trend direction: +1 up, -1 down, 0 flat.

        Uses slope of the `trend_window`-day moving average, annualized.
        """
        if len(prices) < self.trend_window:
            # Not enough data — use shorter window
            window = max(10, len(prices))
        else:
            window = self.trend_window

        ma = prices.rolling(window, min_periods=max(5, window // 2)).mean()
        if len(ma.dropna()) < 2:
            return 0

        # Slope over last 20 days (annualized as %)
        lookback = min(20, len(ma.dropna()))
        recent_ma = ma.dropna().iloc[-lookback:]
        if len(recent_ma) < 2 or recent_ma.iloc[0] == 0:
            return 0

        pct_change = (recent_ma.iloc[-1] / recent_ma.iloc[0] - 1) * 100
        annualized = pct_change * (252 / lookback)

        if annualized > self.trend_threshold:
            return 1
        elif annualized < -self.trend_threshold:
            return -1
        return 0

    def _is_declining(self, prices: pd.Series) -> bool:
        """Check for sharp decline (>5% drop over last 10 trading days)."""
        if len(prices) < 10:
            return False
        recent = prices.iloc[-10:]
        pct_change = (recent.iloc[-1] / recent.iloc[0] - 1) * 100
        return pct_change < -5.0

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    @staticmethod
    def summarize(regime_series: pd.Series) -> Dict:
        """Compute regime distribution summary.

        Returns:
            Dict with counts, percentages, and regime transitions.
        """
        counts = regime_series.value_counts()
        total = len(regime_series)

        distribution = {}
        for regime in Regime:
            n = int(counts.get(regime, 0))
            distribution[regime.value] = {
                "days": n,
                "pct": round(n / total * 100, 1) if total else 0,
            }

        # Count transitions
        transitions = 0
        prev = None
        for r in regime_series:
            if prev is not None and r != prev:
                transitions += 1
            prev = r

        return {
            "total_days": total,
            "distribution": distribution,
            "transitions": transitions,
            "avg_regime_duration": round(total / max(1, transitions), 1),
        }


# ---------------------------------------------------------------------------
# ComboRegimeDetector — absorbed from ml/combo_regime_detector.py
# ---------------------------------------------------------------------------

import logging

_logger = logging.getLogger(__name__)


class ComboRegimeDetector:
    """Rule-based multi-signal regime classifier v2.

    Supported signals (VALID_SIGNALS):
      price_vs_ma200 — price above/below 200-day MA with confidence band
      rsi_momentum   — RSI(14) directional momentum
      vix_structure  — VIX/VIX3M term structure ratio (contango=BULL, backwardation=BEAR)
      ma_crossover   — MA50 > MA200 golden/death cross (optional; not in default set)
    """

    VALID_SIGNALS = frozenset(['price_vs_ma200', 'rsi_momentum', 'vix_structure', 'ma_crossover'])

    def __init__(self, config: dict):
        self.signals            = config.get('signals', ['price_vs_ma200', 'rsi_momentum', 'vix_structure'])
        self.ma_slow            = int(config.get('ma_slow_period', 200))
        self.ma200_neutral_pct  = float(config.get('ma200_neutral_band_pct', 0.5)) / 100
        self.rsi_period         = int(config.get('rsi_period', 14))
        self.rsi_bull           = float(config.get('rsi_bull_threshold', 55.0))
        self.rsi_bear           = float(config.get('rsi_bear_threshold', 45.0))
        self.vix_struct_bull    = float(config.get('vix_structure_bull', 0.95))
        self.vix_struct_bear    = float(config.get('vix_structure_bear', 1.05))
        self.bear_unanimous     = bool(config.get('bear_requires_unanimous', True))
        self.cooldown_days      = int(config.get('cooldown_days', 10))
        self.vix_extreme        = float(config.get('vix_extreme', 40.0))

        # ma_crossover needs fast MA period
        self.ma_fast            = int(config.get('ma_fast_period', 50))

        unknown = set(self.signals) - self.VALID_SIGNALS
        if unknown:
            _logger.warning("Unknown combo regime signals (ignored): %s", unknown)
        self.signals = [s for s in self.signals if s in self.VALID_SIGNALS]

    def compute_regime_series(
        self,
        price_data: pd.DataFrame,
        vix_by_date: dict,
        vix3m_by_date: dict = None,
    ) -> Dict[pd.Timestamp, str]:
        """Compute regime label for each date in price_data.

        Args:
            price_data:    DataFrame with 'Close' column and DatetimeIndex.
                           Must include MA warmup prefix (MA_SLOW extra days before
                           the backtest start so early dates have valid MAs).
            vix_by_date:   {pd.Timestamp: float} raw VIX closes.
            vix3m_by_date: {pd.Timestamp: float} raw VIX3M closes (optional).
                           If None or empty, vix_structure signal abstains.

        Returns:
            {pd.Timestamp: 'bull' | 'bear' | 'neutral'} for every date in price_data.

        Lookahead handling:
            All series are shifted by 1 so regime on date T uses data through T-1.
        """
        if vix3m_by_date is None:
            vix3m_by_date = {}

        closes = price_data['Close']

        # --- Pre-compute all signal series (efficient: once, not per date) ---

        # MA200 (and optionally MA50 for ma_crossover)
        ma_slow_series = closes.rolling(
            self.ma_slow, min_periods=max(10, self.ma_slow // 2)
        ).mean()
        ma_slow_prev = ma_slow_series.shift(1)

        ma_fast_prev = None
        if 'ma_crossover' in self.signals:
            ma_fast_series = closes.rolling(
                self.ma_fast, min_periods=max(10, self.ma_fast // 2)
            ).mean()
            ma_fast_prev = ma_fast_series.shift(1)

        # RSI(14) using EMA-smoothed formula (standard Wilder's RSI)
        delta = closes.diff()
        gain = delta.clip(lower=0).ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        loss = (-delta.clip(upper=0)).ewm(com=self.rsi_period - 1, min_periods=self.rsi_period).mean()
        rsi_series = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))
        rsi_prev = rsi_series.shift(1)

        # VIX/VIX3M ratio — convert dicts to Series aligned with price_data index
        vix_series = pd.Series(vix_by_date, dtype=float).reindex(price_data.index)
        vix3m_series = pd.Series(vix3m_by_date, dtype=float).reindex(price_data.index) if vix3m_by_date else pd.Series(index=price_data.index, dtype=float)
        # Forward-fill to handle weekends/gaps, then shift for no-lookahead
        vix_prev = vix_series.ffill().shift(1)
        vix3m_prev = vix3m_series.ffill().shift(1)
        vix_ratio_prev = vix_prev / vix3m_prev  # NaN when either is missing → abstain

        # --- Iterate with hysteresis state ---
        current_regime = 'bull'      # default starting regime (optimistic prior)
        last_change_idx = -1         # index of last regime change (-1 = never changed)
        n_signals = len(self.signals)
        bear_required = n_signals if self.bear_unanimous else 2

        # closes_prev: yesterday's close — used for price_vs_ma200 to keep the comparison
        # look-ahead-free (regime on date T must only use data available through T-1).
        closes_prev = closes.shift(1)

        result: Dict[pd.Timestamp, str] = {}

        for idx, ts in enumerate(price_data.index):
            price = float(closes_prev.loc[ts])  # yesterday's close (no look-ahead)
            ma_s = ma_slow_prev.loc[ts]
            ma_f = ma_fast_prev.loc[ts] if ma_fast_prev is not None else float('nan')
            rsi = rsi_prev.loc[ts]
            vix_val = vix_prev.loc[ts]
            vix_ratio = vix_ratio_prev.loc[ts]

            # --- VIX circuit breaker (overrides hysteresis) ---
            if not pd.isna(vix_val) and vix_val > self.vix_extreme:
                result[ts] = 'bear'
                if current_regime != 'bear':
                    current_regime = 'bear'
                    last_change_idx = idx
                continue

            # --- Vote ---
            bull_votes, bear_votes = self._vote(price, ma_s, ma_f, rsi, vix_ratio)

            # --- Determine raw regime ---
            if bull_votes >= 2:
                raw_regime = 'bull'
            elif bear_votes >= bear_required:
                raw_regime = 'bear'
            else:
                raw_regime = 'neutral'

            # --- Apply hysteresis ---
            if raw_regime != current_regime:
                days_since_change = idx - last_change_idx
                if last_change_idx >= 0 and days_since_change < self.cooldown_days:
                    # Hysteresis active: keep current regime
                    result[ts] = current_regime
                    continue
                # Accept the change
                current_regime = raw_regime
                last_change_idx = idx

            result[ts] = current_regime

        return result

    def _vote(
        self,
        price: float,
        ma_slow: float,
        ma_fast: float,
        rsi: float,
        vix_ratio: float,
    ):
        """Count bull and bear votes from active signals.

        Returns:
            (bull_votes, bear_votes) — int counts; abstains counted in neither.
        """
        bull = 0
        bear = 0

        for signal in self.signals:
            if signal == 'price_vs_ma200':
                if pd.isna(ma_slow):
                    continue  # insufficient warmup — abstain
                band = ma_slow * self.ma200_neutral_pct
                if price > ma_slow + band:
                    bull += 1
                elif price < ma_slow - band:
                    bear += 1
                # else: within confidence band → abstain

            elif signal == 'rsi_momentum':
                if pd.isna(rsi):
                    continue
                if rsi > self.rsi_bull:
                    bull += 1
                elif rsi < self.rsi_bear:
                    bear += 1
                # else: neutral momentum zone (45–55) → abstain

            elif signal == 'vix_structure':
                if pd.isna(vix_ratio):
                    continue  # missing VIX3M data → abstain gracefully
                if vix_ratio < self.vix_struct_bull:
                    bull += 1   # contango: VIX < VIX3M → market expects calm
                elif vix_ratio > self.vix_struct_bear:
                    bear += 1   # backwardation: VIX > VIX3M → market expects sustained stress

            elif signal == 'ma_crossover':
                if pd.isna(ma_fast) or pd.isna(ma_slow):
                    continue
                if ma_fast > ma_slow:
                    bull += 1   # golden cross
                else:
                    bear += 1   # death cross

        return bull, bear
