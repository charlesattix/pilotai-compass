"""
Feature Engineering Module for Credit Spread Trading

Builds comprehensive feature sets combining technical, volatility, market,
event risk, and seasonal factors.

Based on research:
- Fama & French (1993): Multi-factor models
- Jegadeesh & Titman (1993): Momentum factors
- Cooper et al. (2006): Asset growth and returns
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from shared.constants import CPI_RELEASE_DAYS, FOMC_DATES
from shared.indicators import calculate_rsi

logger = logging.getLogger(__name__)


# ── Pruned feature set (post-ablation) ────────────────────────────────────
#
# The 21 features that survived the walk-forward ablation analysis.
# 10 features were removed from the pipeline's original 31:
#   - 1 harmful (contracts_log — hurts AUC by -0.006)
#   - 9 noise (zero gain + zero permutation importance one-hot dummies)
#
# 5 additional harmful/noise features (vix_percentile_20d, otm_pct,
# ma20_slope_ann_pct, spread_width, day_of_week) are already absent from
# the FeaturePipeline output.
#
# Benchmark results (experiments/pruned_features_benchmark.md):
#   XGBoost AUC: 0.8025 (31 feat) → 0.8077 (21 feat)  +0.005
#   Ensemble AUC: 0.8277 (31 feat) → 0.8318 (21 feat)  +0.004
#
# Reference: experiments/feature_importance_report.md, Section 4-5
#            compass/benchmark_pruned_features.py

PRUNED_FEATURES = [
    # Calendar / timing
    "days_since_last_trade",
    # Momentum & trend
    "rsi_14",
    "momentum_5d_pct",
    "momentum_10d_pct",
    # Volatility context
    "vix_zscore",
    "vix_change_5d_pct",
    "vix_percentile_50d",
    "vix_percentile_100d",
    "iv_rank",
    # Price structure
    "spy_price_zscore",
    "dist_from_ma20_pct",
    "dist_from_ma50_pct",
    "dist_from_ma80_pct",
    "dist_from_ma200_pct",
    "ma50_slope_ann_pct",
    # Realized volatility
    "realized_vol_atr20",
    "realized_vol_20d",
    # Trade structure
    "credit_to_width",
    "loss_to_width",
    # Categorical (only signal-carrying dummies)
    "strategy_type_CS",
    "spread_type_bull_put",
]

# Features removed by pruning (present in pipeline, removed by ablation)
PRUNED_REMOVED = [
    "contracts_log",        # harmful: hurts AUC
    "regime_bear",          # noise: zero importance
    "regime_bull",          # noise: zero importance
    "regime_crash",         # noise: zero importance
    "regime_high_vol",      # noise: zero importance
    "regime_low_vol",       # noise: zero importance
    "strategy_type_IC",     # noise: zero importance
    "strategy_type_SS",     # noise: zero importance
    "spread_type_bear_call",  # noise: zero importance
    "spread_type_unknown",  # noise: zero importance
]


class FeatureEngine:
    """
    Comprehensive feature engineering for options trading ML models.

    Feature categories:
    1. Technical: RSI, MACD, Bollinger %B, ATR, volume
    2. Volatility: IV rank, skew, realized vs implied
    3. Market: VIX, put/call ratio, breadth
    4. Event risk: earnings, FOMC, CPI dates
    5. Seasonal: day of week, month, OPEX
    """

    def __init__(self, data_cache=None, data_provider=None):
        """
        Initialize feature engine.

        Args:
            data_cache: Deprecated alias for data_provider.
            data_provider: Data provider (DataCache or IronVault-compatible)
                           with get_history(ticker, period) method.
                           Returns None on cache miss — caller skips trade.
        """
        self.data_provider = data_provider or data_cache

        # Known FOMC meeting dates 2025-2026
        self.fomc_dates = FOMC_DATES

        logger.info("FeatureEngine initialized")

    def _download(self, ticker, period='6mo'):
        """Fetch equity OHLCV via data_provider. Returns None on cache miss."""
        if self.data_provider:
            result = self.data_provider.get_history(ticker, period)
            if result is not None and not (isinstance(result, pd.DataFrame) and result.empty):
                return result
        logger.warning("No data source for %s (data_provider=%s)", ticker, bool(self.data_provider))
        return None

    def build_features(
        self,
        ticker: str,
        current_price: float,
        options_chain: pd.DataFrame,
        regime_data: Optional[Dict] = None,
        iv_analysis: Optional[Dict] = None,
        technical_signals: Optional[Dict] = None,
        market_features: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Build complete feature set for a potential trade.

        Returns None on cache miss (caller should skip trade).

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            options_chain: Options chain data
            regime_data: Regime detection results
            iv_analysis: IV surface analysis
            technical_signals: Technical analysis signals

        Returns:
            Dictionary of features, or None if data unavailable
        """
        try:
            # Initialize feature dict
            features = {
                'ticker': ticker,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'current_price': current_price,
            }

            # 1. Technical features (None = cache miss → skip trade)
            tech_features = self._compute_technical_features(ticker, current_price)
            if tech_features is None:
                logger.warning("Cache miss: no technical data for %s — skipping", ticker)
                return None
            features.update(tech_features)

            # 2. Volatility features (None = cache miss → skip trade)
            vol_features = self._compute_volatility_features(
                ticker, options_chain, iv_analysis
            )
            if vol_features is None:
                logger.warning("Cache miss: no volatility data for %s — skipping", ticker)
                return None
            features.update(vol_features)

            # 3. Market features (use pre-computed if provided)
            if market_features is not None:
                features.update(market_features)
            else:
                computed_market = self.compute_market_features()
                if computed_market is None:
                    logger.warning("Cache miss: no market data — skipping %s", ticker)
                    return None
                features.update(computed_market)

            # 4. Event risk features
            event_features = self._compute_event_risk_features(ticker)
            if event_features is not None:
                features.update(event_features)

            # 5. Seasonal features
            seasonal_features = self._compute_seasonal_features()
            if seasonal_features is not None:
                features.update(seasonal_features)

            # 5b. Credit stress features (HYG/LQD ratio) — always returns dict, never None
            features.update(self._compute_credit_stress_features())

            # 6. Regime features (if available)
            if regime_data:
                features.update(self._extract_regime_features(regime_data))

            # 7. Derived features
            derived_features = self._compute_derived_features(features)
            features.update(derived_features)

            logger.info(f"Built {len(features)} features for {ticker}")

            return features

        except Exception as e:
            logger.error(f"Error building features for {ticker}: {e}", exc_info=True)
            return None

    def _compute_technical_features(self, ticker: str, current_price: float) -> Optional[Dict]:
        """
        Compute technical indicators. Returns None on cache miss.
        """
        try:
            # Fetch price data
            stock = self._download(ticker, period='6mo')

            if stock is None or stock.empty:
                return None

            close = stock['Close']
            high = stock['High']
            low = stock['Low']
            volume = stock['Volume'] if 'Volume' in stock.columns else pd.Series(index=stock.index)

            features = {}

            # RSI (14-day)
            features['rsi_14'] = float(calculate_rsi(close, 14).iloc[-1])

            # MACD
            macd_line, signal_line, macd_hist = self._calculate_macd(close)
            features['macd'] = float(macd_line.iloc[-1])
            features['macd_signal'] = float(signal_line.iloc[-1])
            features['macd_histogram'] = float(macd_hist.iloc[-1])

            # Bollinger Bands %B
            features['bollinger_pct_b'] = float(self._calculate_bollinger_pct_b(close).iloc[-1])

            # ATR (14-day)
            features['atr_14'] = float(self._calculate_atr(high, low, close, 14).iloc[-1])
            features['atr_pct'] = float(features['atr_14'] / current_price * 100)

            # Volume ratio (current vs 20-day average)
            if not volume.empty:
                vol_ma20 = volume.rolling(20).mean().iloc[-1]
                features['volume_ratio'] = float(volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1.0)
            else:
                features['volume_ratio'] = 1.0

            # Price momentum (5, 10, 20 day returns)
            features['return_5d'] = float((close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) > 5 else 0)
            features['return_10d'] = float((close.iloc[-1] / close.iloc[-11] - 1) * 100 if len(close) > 10 else 0)
            features['return_20d'] = float((close.iloc[-1] / close.iloc[-21] - 1) * 100 if len(close) > 20 else 0)

            # Distance from moving averages
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]
            sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50

            features['dist_from_sma20_pct'] = float((current_price - sma_20) / sma_20 * 100)
            features['dist_from_sma50_pct'] = float((current_price - sma_50) / sma_50 * 100)
            features['dist_from_sma200_pct'] = float((current_price - sma_200) / sma_200 * 100)

            # Opening gap: (today's open - yesterday's close) / yesterday's close
            # Captures overnight sentiment and gap-fill dynamics
            if 'Open' in stock.columns and len(stock) >= 2:
                try:
                    prev_close = float(stock['Close'].iloc[-2])
                    today_open = float(stock['Open'].iloc[-1])
                    if prev_close > 0:
                        features['opening_gap_pct'] = (today_open - prev_close) / prev_close * 100
                    else:
                        features['opening_gap_pct'] = float('nan')
                except Exception:
                    features['opening_gap_pct'] = float('nan')
            else:
                features['opening_gap_pct'] = float('nan')

            return features

        except Exception as e:
            logger.error(f"Error computing technical features: {e}", exc_info=True)
            return None

    def _compute_volatility_features(
        self,
        ticker: str,
        options_chain: pd.DataFrame,
        iv_analysis: Optional[Dict]
    ) -> Optional[Dict]:
        """
        Compute volatility-based features. Returns None on cache miss.
        """
        try:
            features = {}

            # Realized volatility (historical)
            stock = self._download(ticker, period='3mo')

            if stock is not None and not stock.empty:
                returns = stock['Close'].pct_change()
                features['realized_vol_10d'] = float(returns.tail(10).std() * np.sqrt(252) * 100)
                features['realized_vol_20d'] = float(returns.tail(20).std() * np.sqrt(252) * 100)
                features['realized_vol_60d'] = float(returns.tail(60).std() * np.sqrt(252) * 100)
            else:
                return None  # No equity data — can't compute vol features

            # IV features from analysis
            if iv_analysis and iv_analysis.get('iv_rank_percentile', {}).get('available'):
                iv_data = iv_analysis['iv_rank_percentile']
                features['iv_rank'] = float(iv_data.get('iv_rank', 50))
                features['iv_percentile'] = float(iv_data.get('iv_percentile', 50))
                features['current_iv'] = float(iv_data.get('current_iv', 20))
            else:
                # Use options chain if available
                if not options_chain.empty and 'iv' in options_chain.columns:
                    features['current_iv'] = float(options_chain['iv'].median() * 100)
                else:
                    return None  # No IV data — can't compute vol features

                features['iv_rank'] = None
                features['iv_percentile'] = None

            # Realized vs Implied spread
            if features.get('current_iv') is not None:
                features['rv_iv_spread'] = features['realized_vol_20d'] - features['current_iv']
            else:
                features['rv_iv_spread'] = None

            # Skew features
            if iv_analysis and iv_analysis.get('skew', {}).get('available'):
                skew_data = iv_analysis['skew']
                features['put_call_skew_ratio'] = float(skew_data.get('put_call_skew_ratio', 1.0))
                features['put_skew_steepness'] = float(skew_data.get('put_skew_steepness', 0))
            else:
                features['put_call_skew_ratio'] = None
                features['put_skew_steepness'] = None

            return features

        except Exception as e:
            logger.error(f"Error computing volatility features: {e}", exc_info=True)
            return None

    def compute_market_features(self) -> Optional[Dict]:
        """
        Compute market-wide features. Returns None on cache miss.
        """
        try:
            features = {}

            # VIX
            vix = self._download('^VIX', period='1mo')
            if vix is not None and not vix.empty:
                features['vix_level'] = float(vix['Close'].iloc[-1])
                features['vix_change_1d'] = float(vix['Close'].pct_change().iloc[-1] * 100)

                # VIX change over last 5 trading days (absolute points)
                if len(vix) >= 6:
                    features['vix_change_5d'] = float(
                        vix['Close'].iloc[-1] - vix['Close'].iloc[-6]
                    )
                else:
                    features['vix_change_5d'] = 0.0
            else:
                return None  # VIX data required for market features

            # Put/Call ratio (approximation using VIX)
            # In production, use actual CBOE put/call data
            features['put_call_ratio'] = 1.0  # Placeholder

            # Market trend (SPY)
            spy = self._download('SPY', period='3mo')
            if spy is not None and not spy.empty:
                spy_returns = spy['Close'].pct_change()
                features['spy_return_5d'] = float((spy['Close'].iloc[-1] / spy['Close'].iloc[-6] - 1) * 100 if len(spy) > 5 else 0)
                features['spy_return_20d'] = float((spy['Close'].iloc[-1] / spy['Close'].iloc[-21] - 1) * 100 if len(spy) > 20 else 0)

                # SPY volatility
                features['spy_realized_vol'] = float(spy_returns.tail(20).std() * np.sqrt(252) * 100)
            else:
                return None  # SPY data required for market features

            # VIX term structure — variance risk premium proxy.
            # Prefer ^VIX3M (9-month VIX); fall back to VIX / spy_realized_vol ratio.
            # Contango (>1): normal term structure, VRP positive → good for premium sellers.
            # Backwardation (<1): inverted, near-term fear spike → reduce exposure.
            try:
                vix3m = self._download('^VIX3M', period='1mo')
                vix_level = features.get('vix_level', 0)
                if vix3m is not None and not vix3m.empty and vix_level > 0:
                    features['vix_contango_ratio'] = float(vix3m['Close'].iloc[-1]) / vix_level
                elif features.get('spy_realized_vol', 0) > 0:
                    # Proxy: implied vol (VIX) / realized vol — high = rich premium environment
                    features['vix_contango_ratio'] = vix_level / features['spy_realized_vol']
                else:
                    features['vix_contango_ratio'] = float('nan')
            except Exception:
                features['vix_contango_ratio'] = float('nan')

            # SPY / TLT 20-day rolling correlation — risk-on/risk-off regime indicator.
            # Negative correlation: classic risk-off flight-to-quality (normal regime).
            # Positive correlation: 2022-style inflation regime; both assets sell off.
            try:
                tlt = self._download('TLT', period='3mo')
                if tlt is not None and not tlt.empty and spy is not None and not spy.empty:
                    spy_rets = spy['Close'].pct_change().dropna()
                    tlt_rets = tlt['Close'].pct_change().dropna()
                    # Align on shared dates
                    aligned = pd.concat(
                        [spy_rets.rename('spy'), tlt_rets.rename('tlt')], axis=1
                    ).dropna()
                    if len(aligned) >= 20:
                        corr = aligned['spy'].tail(20).corr(aligned['tlt'].tail(20))
                        features['spy_tlt_corr_20d'] = float(corr) if not np.isnan(corr) else float('nan')
                    else:
                        features['spy_tlt_corr_20d'] = float('nan')
                else:
                    features['spy_tlt_corr_20d'] = float('nan')
            except Exception:
                features['spy_tlt_corr_20d'] = float('nan')

            return features

        except Exception as e:
            logger.error(f"Error computing market features: {e}", exc_info=True)
            return None

    def _compute_event_risk_features(self, ticker: str) -> Dict:
        """
        Compute event risk features.
        """
        try:
            features = {}
            now = datetime.now(timezone.utc)

            # Days to next earnings (via data_provider only)
            try:
                if self.data_provider and hasattr(self.data_provider, 'get_ticker_obj'):
                    stock = self.data_provider.get_ticker_obj(ticker)
                    calendar = stock.calendar

                    if calendar is not None and 'Earnings Date' in calendar:
                        earnings_date = pd.to_datetime(calendar['Earnings Date'])
                        if isinstance(earnings_date, pd.Series):
                            earnings_date = earnings_date.iloc[0]

                        days_to_earnings = (earnings_date - now).days
                        features['days_to_earnings'] = max(0, days_to_earnings)
                    else:
                        features['days_to_earnings'] = 999  # Unknown
                else:
                    features['days_to_earnings'] = 999  # No data source
            except Exception as e:
                logger.warning(f"Failed to get earnings for {ticker}: {e}")
                features['days_to_earnings'] = 999

            # Days to next FOMC
            upcoming_fomc = [d for d in self.fomc_dates if d > now]
            if upcoming_fomc:
                days_to_fomc = (min(upcoming_fomc) - now).days
                features['days_to_fomc'] = days_to_fomc
                features['fomc_risk'] = 1.0 if days_to_fomc < 7 else 0.0
            else:
                features['days_to_fomc'] = 999
                features['fomc_risk'] = 0.0

            # Days to next CPI (approx 2nd week of each month)
            import calendar
            current_month = now.month
            current_year = now.year
            current_day = now.day
            # Use the median of the shared CPI_RELEASE_DAYS range
            CPI_RELEASE_DAY = CPI_RELEASE_DAYS[len(CPI_RELEASE_DAYS) // 2]

            if current_day < CPI_RELEASE_DAY + 1:
                # CPI this month hasn't passed yet
                days_to_cpi = CPI_RELEASE_DAY - current_day
            else:
                # CPI next month
                days_in_current_month = calendar.monthrange(current_year, current_month)[1]
                days_to_cpi = days_in_current_month - current_day + CPI_RELEASE_DAY

            features['days_to_cpi'] = days_to_cpi
            features['cpi_risk'] = 1.0 if days_to_cpi < 5 else 0.0

            # Overall event risk score
            min_days = min(
                features['days_to_earnings'],
                features['days_to_fomc'],
                features['days_to_cpi']
            )

            if min_days < 7:
                features['event_risk_score'] = 0.8
            elif min_days < 14:
                features['event_risk_score'] = 0.5
            else:
                features['event_risk_score'] = 0.2

            return features

        except Exception as e:
            logger.error(f"Error computing event risk features: {e}", exc_info=True)
            return None

    def _compute_seasonal_features(self) -> Dict:
        """
        Compute seasonal/calendar features.
        """
        try:
            now = datetime.now(timezone.utc)

            features = {
                'day_of_week': now.weekday(),  # 0=Monday, 4=Friday
                'day_of_month': now.day,
                'month': now.month,
                'quarter': (now.month - 1) // 3 + 1,
            }

            # Days to next monthly OPEX (3rd Friday of the month).
            # More granular than the old is_opex_week boolean — allows the model
            # to learn non-linear proximity effects (e.g. gamma ramp in final 5 days).
            features['days_to_opex'] = self._days_to_next_opex(now)

            # Monday effect (higher volatility)
            features['is_monday'] = 1.0 if now.weekday() == 0 else 0.0

            # End of month effect
            features['is_month_end'] = 1.0 if now.day >= 25 else 0.0

            return features

        except Exception as e:
            logger.error(f"Error computing seasonal features: {e}", exc_info=True)
            return None

    @staticmethod
    def _days_to_next_opex(now: datetime) -> int:
        """Return integer days until the next monthly OPEX (3rd Friday).

        Iterates at most two calendar months forward, so it always finds a
        future OPEX even if the current month's has already passed.
        """
        import calendar as _cal
        today = now.date()
        for delta_months in range(3):
            m = today.month + delta_months
            y = today.year + (m - 1) // 12
            m = ((m - 1) % 12) + 1
            # All Fridays (weekday index 4) in the month
            monthly = _cal.monthcalendar(y, m)
            fridays = [week[_cal.FRIDAY] for week in monthly if week[_cal.FRIDAY] != 0]
            if len(fridays) < 3:
                continue
            from datetime import date as _date
            opex_day = _date(y, m, fridays[2])
            if opex_day > today:
                return (opex_day - today).days
        return 0

    def _compute_credit_stress_features(self) -> Dict:
        """Compute HYG/LQD ratio — high-yield vs investment-grade credit stress.

        A falling HYG/LQD ratio signals credit market stress (widening HY spreads
        relative to IG), which tends to precede equity volatility spikes.  For credit
        spread sellers, rising credit stress = reduce short-premium exposure.

        Returns an empty dict (not None) on any data failure so build_features()
        can always call update() on the result without crashing.
        """
        features: Dict = {}
        try:
            hyg = self._download('HYG', period='3mo')
            lqd = self._download('LQD', period='3mo')

            if hyg is None or hyg.empty or lqd is None or lqd.empty:
                features['hyg_lqd_ratio'] = float('nan')
                features['hyg_lqd_ratio_5d_chg'] = float('nan')
                return features

            # Align on common trading dates
            ratio = (
                hyg['Close'].rename('hyg')
                .to_frame()
                .join(lqd['Close'].rename('lqd'), how='inner')
            )
            ratio['ratio'] = ratio['hyg'] / ratio['lqd']

            if len(ratio) < 1:
                features['hyg_lqd_ratio'] = float('nan')
                features['hyg_lqd_ratio_5d_chg'] = float('nan')
                return features

            features['hyg_lqd_ratio'] = float(ratio['ratio'].iloc[-1])

            if len(ratio) >= 6:
                chg = (ratio['ratio'].iloc[-1] / ratio['ratio'].iloc[-6] - 1) * 100
                features['hyg_lqd_ratio_5d_chg'] = float(chg)
            else:
                features['hyg_lqd_ratio_5d_chg'] = float('nan')

        except Exception as exc:
            logger.warning("Failed to compute credit stress features: %s", exc)
            features['hyg_lqd_ratio'] = float('nan')
            features['hyg_lqd_ratio_5d_chg'] = float('nan')

        return features

    def _extract_regime_features(self, regime_data: Dict) -> Dict:
        """
        Extract features from regime detection.
        """
        features = {
            'regime_id': regime_data.get('regime_id', 2),
            'regime_confidence': regime_data.get('confidence', 0.5),
            'regime_duration_days': regime_data.get('regime_duration_days', 0),
        }

        # One-hot encode regime
        regime_id = features['regime_id']
        features['regime_low_vol_trending'] = 1.0 if regime_id == 0 else 0.0
        features['regime_high_vol_trending'] = 1.0 if regime_id == 1 else 0.0
        features['regime_mean_reverting'] = 1.0 if regime_id == 2 else 0.0
        features['regime_crisis'] = 1.0 if regime_id == 3 else 0.0

        return features

    def _compute_derived_features(self, features: Dict) -> Dict:
        """
        Compute derived/interaction features.
        Handles None values gracefully.
        """
        derived = {}

        # RSI extremes
        rsi = features.get('rsi_14')
        if rsi is not None:
            derived['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
            derived['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
        else:
            derived['rsi_oversold'] = None
            derived['rsi_overbought'] = None

        # IV rank extremes
        iv_rank = features.get('iv_rank')
        if iv_rank is not None:
            derived['iv_rank_high'] = 1.0 if iv_rank > 70 else 0.0
            derived['iv_rank_low'] = 1.0 if iv_rank < 30 else 0.0
        else:
            derived['iv_rank_high'] = None
            derived['iv_rank_low'] = None

        # Volatility regime interaction
        # High IV + low realized vol = good for selling premium
        rv = features.get('realized_vol_20d')
        iv = features.get('current_iv')
        if rv is not None and iv is not None:
            derived['vol_premium'] = iv - rv
            derived['vol_premium_pct'] = (iv - rv) / rv * 100 if rv > 0 else 0
        else:
            derived['vol_premium'] = None
            derived['vol_premium_pct'] = None

        # Risk-adjusted momentum
        atr_pct = features.get('atr_pct')
        return_20d = features.get('return_20d')
        if atr_pct is not None and return_20d is not None and atr_pct > 0:
            derived['risk_adjusted_momentum'] = return_20d / atr_pct
        else:
            derived['risk_adjusted_momentum'] = None

        return derived

    # Helper methods for technical indicators

    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> tuple:
        """Calculate MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def _calculate_bollinger_pct_b(self, prices: pd.Series, period=20, std_dev=2) -> pd.Series:
        """Calculate Bollinger %B."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        pct_b = (prices - lower_band) / (upper_band - lower_band)
        return pct_b

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
        """Calculate ATR."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr

    @staticmethod
    def compute_credit_to_width_ratio(
        credit: float, spread_width: float,
    ) -> Optional[float]:
        """Compute credit-to-width ratio for a spread trade.

        Args:
            credit: Net credit received (e.g. 0.65 for $0.65).
            spread_width: Width of the spread in dollars (e.g. 5.0).

        Returns:
            Ratio (0-1 range for credit spreads), or None if inputs invalid.
        """
        if spread_width is None or spread_width <= 0 or credit is None:
            return None
        return credit / spread_width

    def get_feature_names(self) -> list:
        """
        Get list of all feature names for model training.

        Dropped features (Phase 3 cleanup):
          - spread_width: constant for a given config, zero predictive value
          - vix_percentile_*: redundant with iv_rank
          - ma*_slope_*: noisy, low signal-to-noise ratio
        """
        return [
            # Technical
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bollinger_pct_b', 'atr_pct', 'volume_ratio',
            'return_5d', 'return_10d', 'return_20d',
            'dist_from_sma20_pct', 'dist_from_sma50_pct', 'dist_from_sma200_pct',

            # Volatility
            'realized_vol_10d', 'realized_vol_20d', 'realized_vol_60d',
            'iv_rank', 'current_iv', 'rv_iv_spread',
            'put_call_skew_ratio', 'put_skew_steepness',

            # Market
            'vix_level', 'vix_change_1d', 'vix_change_5d',
            'put_call_ratio',
            'spy_return_5d', 'spy_return_20d', 'spy_realized_vol',

            # Tier 1 alpha: market structure
            'vix_contango_ratio',    # VIX term structure (VIX3M/VIX or VIX/RV proxy)
            'spy_tlt_corr_20d',      # SPY/TLT 20d rolling correlation
            'hyg_lqd_ratio',         # HY/IG credit ratio — credit stress indicator
            'hyg_lqd_ratio_5d_chg',  # 5-day change in credit ratio

            # Trade structure (new Phase 3)
            'credit_to_width_ratio',

            # Event risk
            'days_to_earnings', 'days_to_fomc', 'days_to_cpi',
            'event_risk_score',

            # Seasonal
            'day_of_week', 'days_to_opex', 'is_monday', 'is_month_end',

            # Tier 1 alpha: price action
            'opening_gap_pct',       # (open - prev_close) / prev_close

            # Regime
            'regime_id', 'regime_confidence', 'regime_duration_days',
            'regime_low_vol_trending', 'regime_high_vol_trending',
            'regime_mean_reverting', 'regime_crisis',

            # Derived
            'rsi_oversold', 'rsi_overbought',
            'iv_rank_high', 'iv_rank_low',
            'vol_premium', 'vol_premium_pct',
            'risk_adjusted_momentum',
        ]
