"""
Position Sizing Module

Implements Kelly Criterion with ML confidence adjustment and correlation constraints.

Based on research:
- Kelly (1956): "A New Interpretation of Information Rate"
- Thorp (2008): "The Kelly Criterion in Blackjack Sports Betting and the Stock Market"
- MacLean et al. (2011): "The Kelly Capital Growth Investment Criterion"
"""

import logging
from collections import Counter
from typing import Dict, List, Optional

from shared.types import PositionSizeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upgrade 3: IV-Scaled Position Sizing (standalone functions)
# ---------------------------------------------------------------------------

def calculate_dynamic_risk(
    account_value: float,
    iv_rank: float,
    current_portfolio_risk: float,
    flat_risk_pct: Optional[float] = None,
    max_risk_pct: Optional[float] = None,
    ml_confidence_multiplier: float = 1.0,
) -> float:
    """Return the dollar risk budget for one new trade.

    When flat_risk_pct is provided, it is used directly as the per-trade risk
    percentage (matching backtester flat/sizing_mode behaviour).  When omitted,
    the IV-Rank tier schedule is applied instead.

    Sizing tiers (IV-rank mode only):
      IVR < 20  (low edge)      → 0.5× baseline = 1% of account
      IVR 20-50 (standard edge) → 1× baseline    = 2% of account
      IVR > 50  (high edge)     → up to 1.5×     = up to 3% (linear)

    In both modes a 40% portfolio heat cap is enforced: if adding this trade
    would push total open risk above 40% of account_value, the budget is
    reduced to whatever room remains (floored at 0).

    Args:
        account_value: Current account balance in dollars.
        iv_rank: Current IV Rank 0-100.
        current_portfolio_risk: Dollar value of max-loss exposure already
                                 committed across all open positions.
        flat_risk_pct: If set, use this fixed percentage (e.g. 10.0 for 10%)
                       instead of the IV-rank tiers.  Matches
                       max_risk_per_trade from config.yaml.
        max_risk_pct: If set, caps the IV-rank-derived target at this percentage
                      (e.g. 10.0 for 10%). Has no effect when flat_risk_pct is
                      set. Allows iv_scaled mode to respect max_risk_per_trade.
        ml_confidence_multiplier: ML V2 sizing multiplier (default 1.0 = no effect).
                                  Scales the trade dollar risk by ML confidence.

    Returns:
        Dollar risk budget for the new trade (>= 0).
    """
    base_risk_pct = 0.02  # default IV-rank baseline (2% of account in standard regime)
    # NOTE: This 40% cap is independent of the backtester's max_portfolio_exposure_pct
    # config param, which defaults to 100%.  In iv_scaled mode the effective exposure
    # ceiling is min(max_portfolio_exposure_pct, 40%).  flat mode bypasses this function
    # entirely so the heat cap does NOT apply there.
    max_portfolio_heat = 0.40

    if flat_risk_pct is not None:
        target_risk_pct = flat_risk_pct / 100.0
    elif iv_rank < 20:
        target_risk_pct = base_risk_pct * 0.5
    elif iv_rank <= 50:
        target_risk_pct = base_risk_pct
    else:
        multiplier = min(1.5, 1.0 + ((iv_rank - 50) / 100.0))
        target_risk_pct = base_risk_pct * multiplier

    # P0-A fix: cap IV-rank result at configured max_risk_per_trade when provided.
    # flat_risk_pct already IS the configured value, so cap only applies to iv_scaled mode.
    if max_risk_pct is not None and flat_risk_pct is None:
        target_risk_pct = min(target_risk_pct, max_risk_pct / 100.0)

    trade_dollar_risk = account_value * target_risk_pct
    trade_dollar_risk *= ml_confidence_multiplier  # ML V2 scaling

    heat_budget = account_value * max_portfolio_heat - current_portfolio_risk
    if trade_dollar_risk > heat_budget:
        return max(0.0, heat_budget)

    return trade_dollar_risk


def get_contract_size(
    trade_dollar_risk: float,
    spread_width: float,
    credit_received: float,
    max_contracts: int = 5,
) -> int:
    """Convert a dollar risk budget into a contract count.

    Args:
        trade_dollar_risk: Dollar risk budget from calculate_dynamic_risk().
        spread_width: Width of the spread in dollars (e.g. 5 for a $5-wide spread).
        credit_received: Net credit per spread in dollars (e.g. 0.65).
        max_contracts: Hard cap on contract count (default 5 for backward compat).

    Returns:
        Number of contracts (integer, minimum 0, capped at max_contracts).
    """
    max_loss_per_contract = (spread_width - credit_received) * 100
    if max_loss_per_contract <= 0:
        return 0
    contracts = int(trade_dollar_risk // max_loss_per_contract)
    return min(contracts, max_contracts)


class PositionSizer:
    """
    Adaptive position sizing using Kelly Criterion.

    Features:
    - Kelly criterion for optimal position size
    - ML confidence adjustment (fractional Kelly)
    - Correlation-based position limits
    - Maximum drawdown constraints
    - Risk parity across positions
    """

    def __init__(
        self,
        max_position_size: float = 0.10,
        kelly_fraction: float = 0.25,
        max_portfolio_risk: float = 0.20,
        max_correlated_exposure: float = 0.15,
    ):
        """
        Initialize position sizer.

        Args:
            max_position_size: Maximum size for any single position (fraction of portfolio)
            kelly_fraction: Fraction of Kelly criterion to use (0.25 = quarter Kelly)
            max_portfolio_risk: Maximum total portfolio risk exposure
            max_correlated_exposure: Maximum exposure to correlated positions
        """
        self.max_position_size = max_position_size
        self.kelly_fraction = kelly_fraction
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlated_exposure = max_correlated_exposure
        self.fallback_counter: Counter = Counter()

        logger.info(
            f"PositionSizer initialized: "
            f"max_pos={max_position_size:.2%}, "
            f"kelly_frac={kelly_fraction:.2f}, "
            f"max_risk={max_portfolio_risk:.2%}"
        )

    def calculate_position_size(
        self,
        win_probability: float,
        expected_return: float,
        expected_loss: float,
        ml_confidence: float,
        current_positions: Optional[List[Dict]] = None,
        ticker: str = '',
    ) -> PositionSizeResult:
        """
        Calculate optimal position size for a new trade.

        Args:
            win_probability: Probability of profit (from ML model)
            expected_return: Expected gain if profitable (e.g., 0.30 = 30%)
            expected_loss: Expected loss if unprofitable (e.g., -1.00 = -100%)
            ml_confidence: ML model confidence (0-1)
            current_positions: List of existing positions
            ticker: Ticker symbol for correlation analysis

        Returns:
            Dictionary with position size recommendation and reasoning
        """
        try:
            # 1. Calculate base Kelly size
            kelly_size = self._calculate_kelly(
                win_probability, expected_return, abs(expected_loss)
            )

            # 2. Apply Kelly fraction (for safety)
            fractional_kelly = kelly_size * self.kelly_fraction

            # 3. Adjust by ML confidence
            confidence_adjusted = fractional_kelly * ml_confidence

            # 4. Apply maximum position limit
            size_capped = min(confidence_adjusted, self.max_position_size)

            # 5. Check portfolio-level constraints
            if current_positions:
                size_final = self._apply_portfolio_constraints(
                    size_capped, ticker, current_positions
                )
            else:
                size_final = size_capped

            # 6. Ensure non-negative
            size_final = max(0.0, size_final)

            result = {
                'recommended_size': round(size_final, 4),
                'kelly_size': round(kelly_size, 4),
                'fractional_kelly': round(fractional_kelly, 4),
                'confidence_adjusted': round(confidence_adjusted, 4),
                'capped_size': round(size_capped, 4),
                'applied_constraints': [],
                'expected_value': round(
                    win_probability * expected_return +
                    (1 - win_probability) * expected_loss,
                    4
                ),
                'kelly_fraction_used': self.kelly_fraction,
                'ml_confidence': round(ml_confidence, 3),
            }

            # Add reasoning
            if kelly_size <= 0:
                result['applied_constraints'].append('Negative expected value - no position')
            if confidence_adjusted < fractional_kelly:
                result['applied_constraints'].append(f'Reduced by ML confidence ({ml_confidence:.2f})')
            if size_capped < confidence_adjusted:
                result['applied_constraints'].append(f'Capped at max position size ({self.max_position_size:.2%})')
            if size_final < size_capped:
                result['applied_constraints'].append('Reduced due to portfolio constraints')

            logger.info(
                f"Position size for {ticker}: {size_final:.2%} "
                f"(Kelly={kelly_size:.2%}, Conf={ml_confidence:.2f})"
            )

            return result

        except Exception as e:
            self.fallback_counter['calculate_position_size'] += 1
            count = self.fallback_counter['calculate_position_size']
            logger.error(f"Error calculating position size for {ticker} (fallback #{count}): {e}", exc_info=True)
            if count >= 10:
                logger.critical(f"PositionSizer calculate_position_size has fallen back {count} times — investigate")
            return self._get_default_sizing()

    def _calculate_kelly(
        self,
        win_prob: float,
        win_amount: float,
        loss_amount: float
    ) -> float:
        """
        Calculate Kelly Criterion position size.

        Kelly% = (p * b - q) / b
        where:
            p = probability of winning
            q = probability of losing = 1 - p
            b = win_amount / loss_amount (odds ratio)

        For credit spreads:
            win_amount = premium received (e.g., 0.30 = 30% return on risk)
            loss_amount = max loss (typically 1.00 = 100% of collateral)
        """
        try:
            # Validate inputs
            if win_prob <= 0 or win_prob >= 1:
                return 0.0
            if win_amount <= 0 or loss_amount <= 0:
                return 0.0

            # Calculate Kelly
            p = win_prob
            q = 1 - win_prob
            b = win_amount / loss_amount

            kelly = (p * b - q) / b

            # Kelly can be negative if expected value is negative
            return max(0.0, kelly)

        except Exception as e:
            logger.error(f"Error in Kelly calculation: {e}", exc_info=True)
            return 0.0

    def _apply_portfolio_constraints(
        self,
        proposed_size: float,
        ticker: str,
        current_positions: List[Dict]
    ) -> float:
        """
        Apply portfolio-level constraints.

        Constraints:
        1. Total portfolio risk <= max_portfolio_risk
        2. Correlated exposure <= max_correlated_exposure
        """
        try:
            # Calculate current total risk
            total_current_risk = sum(
                pos.get('position_size', 0) for pos in current_positions
            )

            # Check if adding this position would exceed max portfolio risk
            if total_current_risk + proposed_size > self.max_portfolio_risk:
                available_risk = max(0.0, self.max_portfolio_risk - total_current_risk)
                proposed_size = min(proposed_size, available_risk)

            # Check correlation constraints (simplified)
            # In production, use actual return correlations
            correlated_tickers = self._get_correlated_tickers(ticker)

            correlated_exposure = sum(
                pos.get('position_size', 0)
                for pos in current_positions
                if pos.get('ticker', '') in correlated_tickers
            )

            if correlated_exposure + proposed_size > self.max_correlated_exposure:
                available_corr_risk = max(0.0, self.max_correlated_exposure - correlated_exposure)
                proposed_size = min(proposed_size, available_corr_risk)

            return proposed_size

        except Exception as e:
            logger.error(f"Error applying portfolio constraints: {e}", exc_info=True)
            return proposed_size

    def _get_correlated_tickers(self, ticker: str) -> List[str]:
        """
        Get list of tickers correlated with given ticker.

        Simplified version - in production, calculate actual correlations.
        """
        # Major index ETFs (highly correlated)
        index_etfs = ['SPY', 'QQQ', 'IWM', 'DIA']

        # Tech stocks (correlated)
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'QQQ']

        # Financial stocks
        financials = ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'XLF']

        if ticker in index_etfs:
            return [t for t in index_etfs if t != ticker]
        elif ticker in tech_stocks:
            return [t for t in tech_stocks if t != ticker]
        elif ticker in financials:
            return [t for t in financials if t != ticker]
        else:
            # Assume correlation with major index
            return ['SPY']

    def calculate_portfolio_risk(self, positions: List[Dict]) -> Dict:
        """
        Calculate total portfolio risk metrics.

        Args:
            positions: List of position dictionaries

        Returns:
            Dictionary with risk metrics
        """
        try:
            if not positions:
                return {
                    'total_risk': 0.0,
                    'n_positions': 0,
                    'largest_position': 0.0,
                    'concentration': 0.0,
                    'available_capacity': self.max_portfolio_risk,
                }

            position_sizes = [pos.get('position_size', 0) for pos in positions]

            total_risk = sum(position_sizes)
            n_positions = len(positions)
            largest_position = max(position_sizes) if position_sizes else 0

            # Concentration (HHI - Herfindahl-Hirschman Index)
            concentration = sum(s**2 for s in position_sizes) if total_risk > 0 else 0

            # Available capacity
            available_capacity = max(0.0, self.max_portfolio_risk - total_risk)

            metrics = {
                'total_risk': round(total_risk, 4),
                'n_positions': n_positions,
                'largest_position': round(largest_position, 4),
                'concentration': round(concentration, 4),
                'available_capacity': round(available_capacity, 4),
                'risk_utilization': round(total_risk / self.max_portfolio_risk * 100, 1),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}", exc_info=True)
            return {}

    def rebalance_positions(
        self,
        positions: List[Dict],
        ml_predictions: Dict[str, float]
    ) -> List[Dict]:
        """
        Rebalance existing positions based on updated ML predictions.

        Args:
            positions: Current positions
            ml_predictions: Updated probability predictions {ticker: probability}

        Returns:
            List of rebalancing recommendations
        """
        try:
            recommendations = []

            for pos in positions:
                ticker = pos.get('ticker', '')
                current_size = pos.get('position_size', 0)

                if ticker not in ml_predictions:
                    continue

                # Get updated probability
                new_prob = ml_predictions[ticker]

                # Recalculate optimal size
                win_prob = new_prob
                expected_return = pos.get('expected_return', 0.30)
                expected_loss = pos.get('expected_loss', -1.0)
                ml_confidence = abs(new_prob - 0.5) * 2  # Simple confidence

                sizing = self.calculate_position_size(
                    win_probability=win_prob,
                    expected_return=expected_return,
                    expected_loss=expected_loss,
                    ml_confidence=ml_confidence,
                    current_positions=positions,
                    ticker=ticker,
                )

                recommended_size = sizing['recommended_size']

                # Check if rebalancing needed (>20% difference)
                if current_size == 0:
                    continue
                if abs(recommended_size - current_size) / current_size > 0.20:
                    recommendations.append({
                        'ticker': ticker,
                        'current_size': current_size,
                        'recommended_size': recommended_size,
                        'action': 'increase' if recommended_size > current_size else 'decrease',
                        'change_pct': (recommended_size - current_size) / current_size * 100,
                        'reason': f"Probability updated to {new_prob:.2%}",
                    })

            logger.info(f"Generated {len(recommendations)} rebalancing recommendations")

            return recommendations

        except Exception as e:
            logger.error(f"Error rebalancing positions: {e}", exc_info=True)
            return []

    def calculate_optimal_leverage(
        self,
        position_sizes: List[float],
        win_probabilities: List[float],
        expected_returns: List[float],
    ) -> float:
        """
        Calculate optimal portfolio leverage using Kelly for multiple positions.

        Args:
            position_sizes: Current position sizes
            win_probabilities: Win probability for each position
            expected_returns: Expected return for each position

        Returns:
            Optimal leverage multiplier
        """
        try:
            # Simplified: sum of individual Kelly ratios
            # In practice, would use covariance matrix

            total_kelly = 0.0

            for size, prob, ret in zip(position_sizes, win_probabilities, expected_returns):
                kelly = self._calculate_kelly(prob, ret, 1.0)
                total_kelly += kelly * size

            # Leverage = actual exposure / Kelly exposure
            total_size = sum(position_sizes)

            if total_kelly > 0:
                optimal_leverage = total_size / (total_kelly * self.kelly_fraction)
            else:
                optimal_leverage = 0.0

            return round(optimal_leverage, 2)

        except Exception as e:
            logger.error(f"Error calculating optimal leverage: {e}", exc_info=True)
            return 1.0

    def _get_default_sizing(self) -> PositionSizeResult:
        """
        Return default sizing when calculation fails.
        """
        return {
            'recommended_size': 0.0,
            'kelly_size': 0.0,
            'fractional_kelly': 0.0,
            'confidence_adjusted': 0.0,
            'capped_size': 0.0,
            'applied_constraints': ['Error in calculation'],
            'expected_value': 0.0,
            'kelly_fraction_used': self.kelly_fraction,
            'ml_confidence': 0.0,
        }

    def get_fallback_stats(self) -> Dict[str, int]:
        """Return fallback counts for monitoring."""
        return dict(self.fallback_counter)

    def get_size_recommendation_text(self, sizing_result: Dict, portfolio_value: float) -> str:
        """
        Get human-readable sizing recommendation.

        Args:
            sizing_result: Result from calculate_position_size()
            portfolio_value: Total portfolio value in dollars

        Returns:
            Human-readable recommendation string
        """
        size_pct = sizing_result['recommended_size']
        size_dollars = size_pct * portfolio_value

        # Convert to number of contracts (assuming ~$1000 per contract)
        contracts = int(size_dollars / 1000)

        text = f"Recommended position size: {size_pct:.2%} (${size_dollars:,.0f}, ~{contracts} contracts)\n"
        text += f"Expected value: {sizing_result['expected_value']:.2%}\n"

        if sizing_result['applied_constraints']:
            text += "Constraints applied:\n"
            for constraint in sizing_result['applied_constraints']:
                text += f"  - {constraint}\n"

        return text
