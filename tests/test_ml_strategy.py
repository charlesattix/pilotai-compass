"""
Tests for MLEnhancedStrategy — ML confidence-gated strategy wrapper.

Verifies:
  1. High-confidence signals pass through
  2. Low-confidence signals are filtered
  3. Fallback predictions (untrained model) pass all signals
  4. ML metadata is attached to surviving signals
  5. Position management delegates to base strategy
  6. Sizing delegates to base strategy
  7. Feature-miss signals are dropped
  8. Stats tracking is accurate
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from compass.ml_strategy import MLEnhancedStrategy, DEFAULT_CONFIDENCE_THRESHOLD
from compass.signal_model import SignalModel
from strategies.base import (
    BaseStrategy,
    LegType,
    MarketSnapshot,
    ParamDef,
    PortfolioState,
    Position,
    PositionAction,
    Signal,
    TradeDirection,
    TradeLeg,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_signal(ticker: str = "SPY", score: float = 50.0) -> Signal:
    """Create a minimal valid Signal for testing."""
    exp = datetime(2026, 4, 18)
    return Signal(
        strategy_name="TestStrategy",
        ticker=ticker,
        direction=TradeDirection.SHORT,
        legs=[
            TradeLeg(LegType.SHORT_PUT, strike=440.0, expiration=exp, entry_price=2.50),
            TradeLeg(LegType.LONG_PUT, strike=428.0, expiration=exp, entry_price=0.80),
        ],
        net_credit=1.70,
        max_loss=10.30,
        max_profit=1.70,
        score=score,
        signal_date=datetime(2026, 3, 21),
        expiration=exp,
        dte=28,
        metadata={'spread_type': 'bull_put'},
    )


def _make_market_snapshot() -> MarketSnapshot:
    """Create a minimal MarketSnapshot."""
    idx = pd.bdate_range("2026-01-02", periods=60)
    prices = pd.Series(range(440, 500), index=idx[:60], name="Close", dtype=float)
    df = pd.DataFrame({"Close": prices, "Open": prices, "High": prices + 1,
                        "Low": prices - 1, "Volume": [1_000_000] * 60})
    return MarketSnapshot(
        date=datetime(2026, 3, 21),
        price_data={"SPY": df},
        prices={"SPY": 480.0},
        vix=18.0,
        iv_rank={"SPY": 45.0},
        realized_vol={"SPY": 15.0},
        rsi={"SPY": 52.0},
        regime="bull",
    )


def _make_position() -> Position:
    """Create a minimal Position for manage_position tests."""
    exp = datetime(2026, 4, 18)
    return Position(
        id="pos_001",
        strategy_name="TestStrategy",
        ticker="SPY",
        direction=TradeDirection.SHORT,
        legs=[
            TradeLeg(LegType.SHORT_PUT, strike=440.0, expiration=exp, entry_price=2.50),
            TradeLeg(LegType.LONG_PUT, strike=428.0, expiration=exp, entry_price=0.80),
        ],
        contracts=2,
        entry_date=datetime(2026, 3, 21),
        net_credit=1.70,
    )


def _make_portfolio_state() -> PortfolioState:
    """Create a minimal PortfolioState."""
    return PortfolioState(
        equity=100_000.0,
        starting_capital=100_000.0,
        cash=90_000.0,
    )


class StubStrategy(BaseStrategy):
    """Concrete stub strategy for testing the wrapper."""

    def __init__(self, signals: List[Signal] = None, action: PositionAction = PositionAction.HOLD,
                 contracts: int = 2):
        super().__init__({})
        self._signals = signals or []
        self._action = action
        self._contracts = contracts

    def generate_signals(self, market_data: MarketSnapshot) -> List[Signal]:
        return list(self._signals)

    def manage_position(self, position: Position, market_data: MarketSnapshot) -> PositionAction:
        return self._action

    def size_position(self, signal: Signal, portfolio_state: PortfolioState) -> int:
        return self._contracts

    @classmethod
    def get_param_space(cls) -> List[ParamDef]:
        return []


def _make_ml_strategy(
    signals: List[Signal] = None,
    prediction: Dict[str, Any] = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    feature_engine=None,
    action: PositionAction = PositionAction.HOLD,
    contracts: int = 2,
) -> MLEnhancedStrategy:
    """Wire up an MLEnhancedStrategy with controlled mocks."""
    if signals is None:
        signals = [_make_signal()]

    base = StubStrategy(signals=signals, action=action, contracts=contracts)

    model = MagicMock(spec=SignalModel)
    if prediction is None:
        prediction = {
            'prediction': 1,
            'probability': 0.75,
            'confidence': 0.50,
            'signal': 'bullish',
            'signal_strength': 75.0,
            'timestamp': '2026-03-21T00:00:00+00:00',
        }
    model.predict.return_value = prediction

    return MLEnhancedStrategy(
        base_strategy=base,
        signal_model=model,
        feature_engine=feature_engine,
        ml_config={'confidence_threshold': threshold},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLGatePassThrough:
    """Test 1: High-confidence signals pass through."""

    def test_high_confidence_passes(self):
        """Signal with confidence=0.50 passes threshold=0.30."""
        ml_strat = _make_ml_strategy(
            prediction={'prediction': 1, 'probability': 0.75, 'confidence': 0.50,
                        'signal': 'bullish', 'signal_strength': 75.0,
                        'timestamp': '2026-03-21T00:00:00+00:00'},
            threshold=0.30,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 1
        assert signals[0].ticker == "SPY"

    def test_multiple_signals_all_pass(self):
        """All signals pass when all have high confidence."""
        sigs = [_make_signal("SPY"), _make_signal("SPY")]
        ml_strat = _make_ml_strategy(
            signals=sigs,
            prediction={'prediction': 1, 'probability': 0.80, 'confidence': 0.60,
                        'signal': 'bullish', 'signal_strength': 80.0,
                        'timestamp': '2026-03-21T00:00:00+00:00'},
            threshold=0.30,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 2


class TestMLGateFiltering:
    """Test 2: Low-confidence signals are filtered."""

    def test_low_confidence_filtered(self):
        """Signal with confidence=0.10 is dropped at threshold=0.30."""
        ml_strat = _make_ml_strategy(
            prediction={'prediction': 0, 'probability': 0.55, 'confidence': 0.10,
                        'signal': 'neutral', 'signal_strength': 55.0,
                        'timestamp': '2026-03-21T00:00:00+00:00'},
            threshold=0.30,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 0

    def test_exact_threshold_passes(self):
        """Signal with confidence exactly at threshold passes."""
        ml_strat = _make_ml_strategy(
            prediction={'prediction': 1, 'probability': 0.65, 'confidence': 0.30,
                        'signal': 'bullish', 'signal_strength': 65.0,
                        'timestamp': '2026-03-21T00:00:00+00:00'},
            threshold=0.30,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 1

    def test_partial_filtering(self):
        """Only high-confidence signals survive from a mixed batch."""
        sigs = [_make_signal("SPY"), _make_signal("SPY")]
        model = MagicMock(spec=SignalModel)
        # First call: high confidence, second: low confidence
        model.predict.side_effect = [
            {'prediction': 1, 'probability': 0.80, 'confidence': 0.60,
             'signal': 'bullish', 'signal_strength': 80.0,
             'timestamp': '2026-03-21T00:00:00+00:00'},
            {'prediction': 0, 'probability': 0.52, 'confidence': 0.04,
             'signal': 'neutral', 'signal_strength': 52.0,
             'timestamp': '2026-03-21T00:00:00+00:00'},
        ]
        base = StubStrategy(signals=sigs)
        ml_strat = MLEnhancedStrategy(
            base_strategy=base,
            signal_model=model,
            ml_config={'confidence_threshold': 0.30},
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 1
        assert signals[0].metadata['ml_confidence'] == 0.60


class TestFallbackBehavior:
    """Test 3: Untrained model fallback passes all signals."""

    def test_fallback_passes_signal(self):
        """When model returns fallback=True, signal passes through."""
        ml_strat = _make_ml_strategy(
            prediction={
                'prediction': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'signal': 'neutral',
                'signal_strength': 50.0,
                'timestamp': '2026-03-21T00:00:00+00:00',
                'fallback': True,
            },
            threshold=0.30,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 1
        assert signals[0].metadata['ml_gated'] is False

    def test_fallback_with_zero_confidence_passes(self):
        """Fallback with confidence=0.0 still passes (unlike normal prediction)."""
        ml_strat = _make_ml_strategy(
            prediction={
                'prediction': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'signal': 'neutral',
                'signal_strength': 50.0,
                'timestamp': '2026-03-21T00:00:00+00:00',
                'fallback': True,
            },
            threshold=0.50,  # High threshold
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 1


class TestMetadataEnrichment:
    """Test 4: ML metadata is attached to surviving signals."""

    def test_ml_prediction_in_metadata(self):
        pred = {'prediction': 1, 'probability': 0.72, 'confidence': 0.44,
                'signal': 'bullish', 'signal_strength': 72.0,
                'timestamp': '2026-03-21T00:00:00+00:00'}
        ml_strat = _make_ml_strategy(prediction=pred, threshold=0.20)
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 1
        meta = signals[0].metadata
        assert meta['ml_prediction'] == pred
        assert meta['ml_gated'] is True
        assert meta['ml_confidence'] == 0.44
        assert meta['ml_probability'] == 0.72

    def test_original_metadata_preserved(self):
        """Original signal metadata (spread_type etc.) is not overwritten."""
        ml_strat = _make_ml_strategy(threshold=0.10)
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert signals[0].metadata['spread_type'] == 'bull_put'


class TestDelegation:
    """Test 5: Position management and sizing delegate to base strategy."""

    def test_manage_position_delegates(self):
        ml_strat = _make_ml_strategy(action=PositionAction.CLOSE_PROFIT)
        snap = _make_market_snapshot()
        pos = _make_position()
        result = ml_strat.manage_position(pos, snap)
        assert result == PositionAction.CLOSE_PROFIT

    def test_manage_position_hold(self):
        ml_strat = _make_ml_strategy(action=PositionAction.HOLD)
        snap = _make_market_snapshot()
        pos = _make_position()
        result = ml_strat.manage_position(pos, snap)
        assert result == PositionAction.HOLD

    def test_size_position_delegates(self):
        ml_strat = _make_ml_strategy(contracts=3)
        sig = _make_signal()
        ps = _make_portfolio_state()
        result = ml_strat.size_position(sig, ps)
        assert result == 3

    def test_size_position_zero_skip(self):
        ml_strat = _make_ml_strategy(contracts=0)
        sig = _make_signal()
        ps = _make_portfolio_state()
        result = ml_strat.size_position(sig, ps)
        assert result == 0


class TestFeatureMiss:
    """Test 7: Feature-miss signals are dropped."""

    def test_feature_engine_returns_none(self):
        """When FeatureEngine.build_features returns None, signal is dropped."""
        fe = MagicMock()
        fe.build_features.return_value = None

        ml_strat = _make_ml_strategy(feature_engine=fe, threshold=0.0)
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 0

    def test_missing_price_drops_signal(self):
        """When ticker price is missing from snapshot, signal is dropped."""
        fe = MagicMock()
        fe.build_features.return_value = None

        ml_strat = _make_ml_strategy(
            signals=[_make_signal("QQQ")],  # QQQ not in snapshot prices
            feature_engine=fe,
            threshold=0.0,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 0


class TestStatsTracking:
    """Test 8: ML gating stats are tracked correctly."""

    def test_stats_after_mixed_filtering(self):
        sigs = [_make_signal(), _make_signal(), _make_signal()]
        model = MagicMock(spec=SignalModel)
        model.predict.side_effect = [
            # Pass
            {'prediction': 1, 'probability': 0.80, 'confidence': 0.60,
             'signal': 'bullish', 'signal_strength': 80.0,
             'timestamp': '2026-03-21T00:00:00+00:00'},
            # Filter
            {'prediction': 0, 'probability': 0.52, 'confidence': 0.04,
             'signal': 'neutral', 'signal_strength': 52.0,
             'timestamp': '2026-03-21T00:00:00+00:00'},
            # Pass
            {'prediction': 1, 'probability': 0.70, 'confidence': 0.40,
             'signal': 'bullish', 'signal_strength': 70.0,
             'timestamp': '2026-03-21T00:00:00+00:00'},
        ]
        base = StubStrategy(signals=sigs)
        ml_strat = MLEnhancedStrategy(
            base_strategy=base,
            signal_model=model,
            ml_config={'confidence_threshold': 0.30},
        )
        snap = _make_market_snapshot()
        ml_strat.generate_signals(snap)

        stats = ml_strat.get_ml_stats()
        assert stats['total_signals'] == 3
        assert stats['passed_signals'] == 2
        assert stats['filtered_signals'] == 1
        assert stats['feature_miss_signals'] == 0
        assert abs(stats['pass_rate'] - 2 / 3) < 0.01

    def test_empty_base_signals(self):
        """No signals from base → no ML processing."""
        ml_strat = _make_ml_strategy(signals=[])
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 0
        stats = ml_strat.get_ml_stats()
        assert stats['total_signals'] == 0


class TestNameAndParamSpace:
    """Verify wrapper naming and param space."""

    def test_wrapped_name(self):
        ml_strat = _make_ml_strategy()
        assert ml_strat.name.startswith("ML_")

    def test_param_space_has_threshold(self):
        params = MLEnhancedStrategy.get_param_space()
        names = [p.name for p in params]
        assert 'ml_confidence_threshold' in names

    def test_default_threshold(self):
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.30
