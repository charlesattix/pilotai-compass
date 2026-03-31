"""
Implied Volatility Surface Analysis Module

Analyzes IV surface structure, skew steepness, and term structure for
optimal credit spread selection.

Based on research:
- Gatheral (2006): "The Volatility Surface: A Practitioner's Guide"
- Bollen & Whaley (2004): "Does Net Buying Pressure Affect the Shape of Implied Volatility Functions?"
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from shared.indicators import calculate_iv_rank as _shared_iv_rank

logger = logging.getLogger(__name__)


class IVAnalyzer:
    """
    Advanced implied volatility surface analyzer.

    Analyzes:
    - Skew steepness (25-delta put vs call IV)
    - Term structure slope (near-term vs far-term IV)
    - IV rank and percentile (current IV vs 1-year history)
    - Skew inversion (unusual patterns indicating opportunities)
    """

    def __init__(self, lookback_days: int = 252, data_cache=None, data_provider=None):
        """
        Initialize IV analyzer.

        Args:
            lookback_days: Historical IV lookback period
            data_cache: Deprecated alias for data_provider.
            data_provider: Data provider (DataCache or IronVault-compatible)
                           with get_history(ticker, period) method.
        """
        self.lookback_days = lookback_days
        self.data_provider = data_provider or data_cache
        self.iv_history_cache = {}
        self.cache_timestamp = {}

        logger.info(f"IVAnalyzer initialized (lookback={lookback_days} days)")

    def analyze_surface(
        self,
        ticker: str,
        options_chain: pd.DataFrame,
        current_price: float
    ) -> Dict:
        """
        Comprehensive IV surface analysis.

        Args:
            ticker: Stock ticker
            options_chain: Options chain DataFrame
            current_price: Current stock price

        Returns:
            Dictionary with IV metrics and signals
        """
        try:
            if options_chain.empty:
                return self._get_default_analysis()

            # Compute IV metrics
            skew_metrics = self._compute_skew_metrics(options_chain, current_price)
            term_structure = self._compute_term_structure(options_chain, current_price)
            iv_rank_percentile = self._compute_iv_rank_percentile(ticker, options_chain)

            # Generate trading signals
            signals = self._generate_signals(skew_metrics, term_structure, iv_rank_percentile)

            result = {
                'ticker': ticker,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'skew': skew_metrics,
                'term_structure': term_structure,
                'iv_rank_percentile': iv_rank_percentile,
                'signals': signals,
            }

            logger.debug(
                f"{ticker} IV Analysis: "
                f"Skew={skew_metrics.get('put_call_skew_ratio', 1.0):.3f}, "
                f"IV_Rank={iv_rank_percentile.get('iv_rank', 50):.1f}%, "
                f"Signal={signals.get('overall_signal', 'neutral')}"
            )

            return result

        except Exception as e:
            logger.error(f"Error analyzing IV surface for {ticker}: {e}", exc_info=True)
            return self._get_default_analysis()

    def _compute_skew_metrics(self, options_chain: pd.DataFrame, current_price: float) -> Dict:
        """
        Compute volatility skew metrics.

        Key metric: Put skew / Call skew ratio
        - Ratio > 1.1: Steep put skew (good for bull put spreads)
        - Ratio < 0.9: Steep call skew (good for bear call spreads)
        """
        try:
            # Filter for options with good liquidity
            mask = (options_chain['bid'] > 0) & (options_chain['ask'] > 0)
            if 'volume' in options_chain.columns:
                mask = mask & (options_chain['volume'] > 10)
            chain = options_chain[mask].copy()

            if chain.empty or 'iv' not in chain.columns:
                return {'available': False}

            # Separate puts and calls
            puts = chain[chain['type'] == 'put'].copy()
            calls = chain[chain['type'] == 'call'].copy()

            if puts.empty or calls.empty:
                return {'available': False}

            # Calculate moneyness (strike / spot)
            puts['moneyness'] = puts['strike'] / current_price
            calls['moneyness'] = calls['strike'] / current_price

            # Focus on near-the-money options (0.90 - 1.10)
            puts_ntm = puts[(puts['moneyness'] >= 0.90) & (puts['moneyness'] <= 1.10)]
            calls_ntm = calls[(calls['moneyness'] >= 0.90) & (calls['moneyness'] <= 1.10)]

            if puts_ntm.empty or calls_ntm.empty:
                # Fallback to all options
                puts_ntm = puts
                calls_ntm = calls

            # ATM IV (strike closest to current price)
            atm_put_iv = puts_ntm.iloc[(puts_ntm['strike'] - current_price).abs().argsort()[:1]]['iv'].mean()
            atm_call_iv = calls_ntm.iloc[(calls_ntm['strike'] - current_price).abs().argsort()[:1]]['iv'].mean()

            # 25-delta approximate (OTM by ~10%)
            put_25d_strike = current_price * 0.90
            call_25d_strike = current_price * 1.10

            # Find closest strikes
            put_25d_iv = puts.iloc[(puts['strike'] - put_25d_strike).abs().argsort()[:1]]['iv'].mean()
            call_25d_iv = calls.iloc[(calls['strike'] - call_25d_strike).abs().argsort()[:1]]['iv'].mean()

            # Compute skew steepness
            put_skew = put_25d_iv - atm_call_iv  # OTM put IV vs ATM call IV
            call_skew = call_25d_iv - atm_call_iv  # OTM call IV vs ATM call IV

            # Skew ratio (key metric)
            if call_skew > 0:
                skew_ratio = put_skew / call_skew
            else:
                skew_ratio = 2.0 if put_skew > 0 else 1.0

            # Put-call IV differential
            put_call_diff = atm_put_iv - atm_call_iv

            metrics = {
                'available': True,
                'atm_put_iv': float(atm_put_iv * 100),
                'atm_call_iv': float(atm_call_iv * 100),
                'put_25d_iv': float(put_25d_iv * 100),
                'call_25d_iv': float(call_25d_iv * 100),
                'put_skew_steepness': float(put_skew * 100),
                'call_skew_steepness': float(call_skew * 100),
                'put_call_skew_ratio': float(skew_ratio),
                'put_call_iv_diff': float(put_call_diff * 100),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error computing skew metrics: {e}", exc_info=True)
            return {'available': False}

    def _compute_term_structure(self, options_chain: pd.DataFrame, current_price: float) -> Dict:
        """
        Analyze IV term structure (near-term vs far-term).

        Contango (near < far): Normal, favor selling near-term
        Backwardation (near > far): Fear, avoid new trades or adjust
        """
        try:
            if 'expiration' not in options_chain.columns or 'iv' not in options_chain.columns:
                return {'available': False}

            # Calculate DTE for each option (copy to avoid mutating caller's DataFrame)
            now = datetime.now(timezone.utc)
            options_chain = options_chain.copy()
            exp_col = pd.to_datetime(options_chain['expiration'])
            if exp_col.dt.tz is None:
                exp_col = exp_col.dt.tz_localize('UTC')
            options_chain['dte'] = (exp_col - now).dt.days

            # Filter ATM options
            atm_options = options_chain[
                (abs(options_chain['strike'] - current_price) / current_price < 0.05)
            ].copy()

            if atm_options.empty:
                return {'available': False}

            # Near-term: 20-40 DTE
            near_term = atm_options[(atm_options['dte'] >= 20) & (atm_options['dte'] <= 40)]

            # Far-term: 60-90 DTE
            far_term = atm_options[(atm_options['dte'] >= 60) & (atm_options['dte'] <= 90)]

            if near_term.empty or far_term.empty:
                return {'available': False}

            near_term_iv = near_term['iv'].mean() * 100
            far_term_iv = far_term['iv'].mean() * 100

            # Term structure slope
            slope = far_term_iv - near_term_iv
            slope_pct = (slope / near_term_iv * 100) if near_term_iv > 0 else 0

            # Classify term structure
            if slope > 2:
                structure_type = 'contango'  # Normal
            elif slope < -2:
                structure_type = 'backwardation'  # Fear/event risk
            else:
                structure_type = 'flat'

            metrics = {
                'available': True,
                'near_term_iv': float(near_term_iv),
                'far_term_iv': float(far_term_iv),
                'slope': float(slope),
                'slope_pct': float(slope_pct),
                'structure_type': structure_type,
            }

            return metrics

        except Exception as e:
            logger.error(f"Error computing term structure: {e}", exc_info=True)
            return {'available': False}

    def _compute_iv_rank_percentile(self, ticker: str, options_chain: pd.DataFrame) -> Dict:
        """
        Calculate IV rank and percentile using historical data.

        IV Rank: Where current IV sits in its 52-week range
        IV Percentile: What % of days had lower IV
        """
        try:
            # Get current ATM IV
            if 'iv' not in options_chain.columns:
                return {'available': False}

            current_iv = options_chain['iv'].median() * 100

            # Fetch historical IV (using HV as proxy)
            iv_history = self._get_iv_history(ticker)

            if iv_history is None or len(iv_history) < 50:
                return {'available': False, 'current_iv': float(current_iv)}

            # Delegate core calculation to shared implementation
            shared_result = _shared_iv_rank(iv_history, current_iv)

            # Mean reversion signal
            # High IV (>70% rank) suggests mean reversion down (good for credit spreads)
            # Low IV (<30% rank) suggests expansion risk (be cautious)

            metrics = {
                'available': True,
                'current_iv': float(current_iv),
                'iv_rank': float(shared_result['iv_rank']),
                'iv_percentile': float(shared_result['iv_percentile']),
                'iv_min_52w': float(shared_result['iv_min']),
                'iv_max_52w': float(shared_result['iv_max']),
                'iv_mean_52w': float(iv_history.mean()),
                'iv_std_52w': float(iv_history.std()),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error computing IV rank/percentile: {e}", exc_info=True)
            return {'available': False}

    def _get_iv_history(self, ticker: str) -> Optional[pd.Series]:
        """
        Get historical implied volatility (using HV as proxy).

        Requires data_provider. Returns None on cache miss.
        Caches results for 1 day.
        """
        # Check in-memory cache
        if ticker in self.iv_history_cache:
            cache_age = (datetime.now(timezone.utc) - self.cache_timestamp.get(ticker, datetime.min.replace(tzinfo=timezone.utc))).total_seconds()
            if cache_age < 86400:  # 24 hours
                return self.iv_history_cache[ticker]

        try:
            if not self.data_provider:
                logger.warning("No data_provider — cannot compute IV history for %s", ticker)
                return None

            stock = self.data_provider.get_history(ticker, period='1y')

            if stock is None or stock.empty:
                return None

            # Calculate 20-day historical volatility as IV proxy
            returns = stock['Close'].pct_change()
            hv = returns.rolling(window=20).std() * np.sqrt(252) * 100
            hv = hv.dropna()

            # Cache result
            self.iv_history_cache[ticker] = hv
            self.cache_timestamp[ticker] = datetime.now(timezone.utc)

            return hv

        except Exception as e:
            logger.error(f"Error fetching IV history for {ticker}: {e}", exc_info=True)
            return None

    def _generate_signals(
        self,
        skew_metrics: Dict,
        term_structure: Dict,
        iv_rank_percentile: Dict
    ) -> Dict:
        """
        Generate trading signals from IV analysis.

        Returns:
            Dictionary with signals for bull put and bear call spreads
        """
        signals = {
            'bull_put_favorable': False,
            'bear_call_favorable': False,
            'overall_signal': 'neutral',
            'reasoning': [],
        }

        # Check data availability
        if not skew_metrics.get('available'):
            signals['reasoning'].append('Insufficient skew data')
            return signals

        # Skew-based signals
        skew_ratio = skew_metrics.get('put_call_skew_ratio', 1.0)

        if skew_ratio > 1.15:
            signals['bull_put_favorable'] = True
            signals['reasoning'].append(f'Steep put skew (ratio={skew_ratio:.2f})')

        if skew_ratio < 0.85:
            signals['bear_call_favorable'] = True
            signals['reasoning'].append(f'Steep call skew (ratio={skew_ratio:.2f})')

        # IV rank signals
        if iv_rank_percentile.get('available'):
            iv_rank = iv_rank_percentile.get('iv_rank', 50)

            if iv_rank > 70:
                signals['bull_put_favorable'] = True
                signals['bear_call_favorable'] = True
                signals['reasoning'].append(f'High IV rank ({iv_rank:.0f}%) - mean reversion opportunity')

            if iv_rank < 30:
                signals['reasoning'].append(f'Low IV rank ({iv_rank:.0f}%) - expansion risk')

        # Term structure signals
        if term_structure.get('available'):
            structure_type = term_structure.get('structure_type')

            if structure_type == 'backwardation':
                signals['reasoning'].append('IV backwardation - caution, possible event risk')

            if structure_type == 'contango':
                signals['reasoning'].append('IV contango - normal structure')

        # Overall signal
        if signals['bull_put_favorable'] and signals['bear_call_favorable']:
            signals['overall_signal'] = 'favorable_both'
        elif signals['bull_put_favorable']:
            signals['overall_signal'] = 'favorable_bull_put'
        elif signals['bear_call_favorable']:
            signals['overall_signal'] = 'favorable_bear_call'

        return signals

    def _get_default_analysis(self) -> Dict:
        """
        Return default analysis when data unavailable.
        """
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'skew': {'available': False},
            'term_structure': {'available': False},
            'iv_rank_percentile': {'available': False},
            'signals': {
                'bull_put_favorable': False,
                'bear_call_favorable': False,
                'overall_signal': 'neutral',
                'reasoning': ['Insufficient data'],
            },
        }
