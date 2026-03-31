"""
Tests for ML V2: Regime-Specific Confidence Sizing.

Verifies:
  1. confidence_to_size_multiplier mapping
  2. RegimeModelRouter routing and fallback
  3. V2 confidence sizing mode (never drops signals)
  4. Kill switch (ml_sizing=False → V1, ml_sizing=True → V2)
  5. Backward compatibility with V1
"""

from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from compass.ml_strategy import (
    MLEnhancedStrategy,
    RegimeModelRouter,
    confidence_to_size_multiplier,
    DEFAULT_CONFIDENCE_THRESHOLD,
)
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
# Fixtures (reuse patterns from test_ml_strategy.py)
# ---------------------------------------------------------------------------

def _make_signal(ticker: str = "SPY", score: float = 50.0) -> Signal:
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


def _make_market_snapshot(regime: str = "bull") -> MarketSnapshot:
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
        regime=regime,
    )


def _make_portfolio_state() -> PortfolioState:
    return PortfolioState(
        equity=100_000.0,
        starting_capital=100_000.0,
        cash=90_000.0,
    )


class StubStrategy(BaseStrategy):
    """Concrete stub strategy for testing the wrapper."""

    def __init__(self, signals=None, action=PositionAction.HOLD, contracts=2):
        super().__init__({})
        self._signals = signals or []
        self._action = action
        self._contracts = contracts

    def generate_signals(self, market_data):
        return list(self._signals)

    def manage_position(self, position, market_data):
        return self._action

    def size_position(self, signal, portfolio_state):
        return self._contracts

    @classmethod
    def get_param_space(cls):
        return []


def _make_prediction(confidence: float = 0.50, probability: float = 0.75):
    """Build a mock prediction dict with given confidence."""
    return {
        'prediction': 1,
        'probability': probability,
        'confidence': confidence,
        'signal': 'bullish',
        'signal_strength': probability * 100,
        'timestamp': '2026-03-21T00:00:00+00:00',
    }


def _make_v2_strategy(
    signals=None,
    prediction=None,
    ml_sizing=True,
    feature_engine=None,
    contracts=4,
) -> MLEnhancedStrategy:
    """Wire up a V2 MLEnhancedStrategy with controlled mocks."""
    if signals is None:
        signals = [_make_signal()]
    if prediction is None:
        prediction = _make_prediction(confidence=0.60)

    base = StubStrategy(signals=signals, contracts=contracts)
    model = MagicMock(spec=SignalModel)
    model.predict.return_value = prediction
    model.load.return_value = False  # Prevent regime model loading

    return MLEnhancedStrategy(
        base_strategy=base,
        signal_model=model,
        feature_engine=feature_engine,
        ml_config={'ml_sizing': ml_sizing},
    )


# ---------------------------------------------------------------------------
# TestConfidenceToMultiplier
# ---------------------------------------------------------------------------

class TestConfidenceToMultiplier:
    """Test confidence_to_size_multiplier linear mapping.

    Default: min_mult=0.25, max_mult=1.25
    Formula: 0.25 + (1.25 - 0.25) * confidence = 0.25 + 1.0 * confidence
    """

    def test_high_confidence(self):
        # 0.25 + 1.0 * 0.85 = 1.10
        assert confidence_to_size_multiplier(0.85) == pytest.approx(1.10)

    def test_medium_confidence(self):
        # 0.25 + 1.0 * 0.60 = 0.85
        assert confidence_to_size_multiplier(0.60) == pytest.approx(0.85)

    def test_low_confidence(self):
        # 0.25 + 1.0 * 0.40 = 0.65
        assert confidence_to_size_multiplier(0.40) == pytest.approx(0.65)

    def test_zero_confidence_returns_min(self):
        assert confidence_to_size_multiplier(0.0) == pytest.approx(0.25)

    def test_full_confidence_returns_max(self):
        assert confidence_to_size_multiplier(1.0) == pytest.approx(1.25)

    def test_clamps_negative_to_zero(self):
        """Negative confidence clamped to 0 → returns min_mult."""
        assert confidence_to_size_multiplier(-0.5) == pytest.approx(0.25)

    def test_clamps_above_one(self):
        """Confidence > 1 clamped to 1 → returns max_mult."""
        assert confidence_to_size_multiplier(1.5) == pytest.approx(1.25)

    def test_custom_min_max(self):
        # min=0.50, max=1.00, confidence=0.5 → 0.50 + 0.50*0.5 = 0.75
        assert confidence_to_size_multiplier(0.5, min_mult=0.50, max_mult=1.00) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# TestRegimeModelRouter
# ---------------------------------------------------------------------------

class TestRegimeModelRouter:
    """Test RegimeModelRouter routing and fallback."""

    def test_default_model_used_when_no_regime(self):
        """regime=None → default model."""
        default = MagicMock(spec=SignalModel)
        default.predict.return_value = _make_prediction(0.80)

        router = RegimeModelRouter(default_model=default)
        result = router.predict({'vix_level': 18.0}, regime=None)

        default.predict.assert_called_once()
        assert result['confidence'] == 0.80

    def test_regime_model_used_when_available(self):
        """regime="bull" → bull model when loaded."""
        default = MagicMock(spec=SignalModel)
        default.predict.return_value = _make_prediction(0.50)

        bull_model = MagicMock(spec=SignalModel)
        bull_model.predict.return_value = _make_prediction(0.90)

        router = RegimeModelRouter(default_model=default)
        router.regime_models['bull'] = bull_model

        result = router.predict({'vix_level': 18.0}, regime='bull')

        bull_model.predict.assert_called_once()
        default.predict.assert_not_called()
        assert result['confidence'] == 0.90

    def test_fallback_to_default_on_missing_regime(self):
        """regime="crash" with no crash model → default model."""
        default = MagicMock(spec=SignalModel)
        default.predict.return_value = _make_prediction(0.40)

        router = RegimeModelRouter(default_model=default)
        result = router.predict({'vix_level': 30.0}, regime='crash')

        default.predict.assert_called_once()
        assert result['confidence'] == 0.40

    def test_all_regimes_routed(self):
        """Each loaded regime model is used for its regime."""
        default = MagicMock(spec=SignalModel)
        default.predict.return_value = _make_prediction(0.10)

        regimes = ['bull', 'bear', 'high_vol', 'low_vol']
        router = RegimeModelRouter(default_model=default)

        for regime in regimes:
            model = MagicMock(spec=SignalModel)
            model.predict.return_value = _make_prediction(0.80)
            router.regime_models[regime] = model

        for regime in regimes:
            result = router.predict({'vix_level': 18.0}, regime=regime)
            assert result['confidence'] == 0.80
            router.regime_models[regime].predict.assert_called_once()

        # Default should not have been called
        default.predict.assert_not_called()

    def test_defensive_regime_used_for_bear(self):
        """bear regime falls back to defensive model when no bear model loaded."""
        default = MagicMock(spec=SignalModel)
        default.predict.return_value = _make_prediction(0.10)

        defensive_model = MagicMock(spec=SignalModel)
        defensive_model.predict.return_value = _make_prediction(0.70)

        router = RegimeModelRouter(default_model=default)
        router.regime_models['defensive'] = defensive_model

        result = router.predict({'vix_level': 25.0}, regime='bear')
        defensive_model.predict.assert_called_once()
        default.predict.assert_not_called()
        assert result['confidence'] == 0.70

    def test_defensive_regime_used_for_high_vol(self):
        """high_vol regime falls back to defensive model."""
        default = MagicMock(spec=SignalModel)
        default.predict.return_value = _make_prediction(0.10)

        defensive_model = MagicMock(spec=SignalModel)
        defensive_model.predict.return_value = _make_prediction(0.65)

        router = RegimeModelRouter(default_model=default)
        router.regime_models['defensive'] = defensive_model

        result = router.predict({'vix_level': 30.0}, regime='high_vol')
        defensive_model.predict.assert_called_once()
        assert result['confidence'] == 0.65

    def test_exact_regime_takes_priority_over_defensive(self):
        """If both bear and defensive models exist, bear model wins."""
        default = MagicMock(spec=SignalModel)
        bear_model = MagicMock(spec=SignalModel)
        bear_model.predict.return_value = _make_prediction(0.90)
        defensive_model = MagicMock(spec=SignalModel)
        defensive_model.predict.return_value = _make_prediction(0.50)

        router = RegimeModelRouter(default_model=default)
        router.regime_models['bear'] = bear_model
        router.regime_models['defensive'] = defensive_model

        result = router.predict({'vix_level': 25.0}, regime='bear')
        bear_model.predict.assert_called_once()
        defensive_model.predict.assert_not_called()
        assert result['confidence'] == 0.90

    def test_defensive_does_not_apply_to_bull(self):
        """bull regime does NOT fall back to defensive — goes to default."""
        default = MagicMock(spec=SignalModel)
        default.predict.return_value = _make_prediction(0.30)
        defensive_model = MagicMock(spec=SignalModel)
        defensive_model.predict.return_value = _make_prediction(0.80)

        router = RegimeModelRouter(default_model=default)
        router.regime_models['defensive'] = defensive_model

        result = router.predict({'vix_level': 15.0}, regime='bull')
        default.predict.assert_called_once()
        defensive_model.predict.assert_not_called()
        assert result['confidence'] == 0.30


# ---------------------------------------------------------------------------
# TestMLSizingMode
# ---------------------------------------------------------------------------

class TestMLSizingMode:
    """Test V2 confidence sizing behavior."""

    def test_v2_never_drops_signals(self):
        """V2 mode keeps ALL signals regardless of confidence."""
        # Use very low confidence that V1 would drop
        sigs = [_make_signal(), _make_signal(), _make_signal()]
        ml_strat = _make_v2_strategy(
            signals=sigs,
            prediction=_make_prediction(confidence=0.05),
            ml_sizing=True,
        )
        snap = _make_market_snapshot()
        result = ml_strat.generate_signals(snap)
        assert len(result) == 3

    def test_v2_attaches_multiplier_metadata(self):
        """V2 signals have ml_size_multiplier in metadata."""
        ml_strat = _make_v2_strategy(
            prediction=_make_prediction(confidence=0.60),
            ml_sizing=True,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 1

        meta = signals[0].metadata
        assert 'ml_size_multiplier' in meta
        assert meta['ml_size_multiplier'] == pytest.approx(0.85)  # 0.25 + 1.0*0.60
        assert meta['ml_confidence'] == 0.60
        assert meta['ml_gated'] is True

    def test_v2_feature_miss_gets_minimum_size(self):
        """Feature miss → 0.25 multiplier, not dropped."""
        fe = MagicMock()
        fe.build_features.return_value = None

        ml_strat = _make_v2_strategy(
            feature_engine=fe,
            ml_sizing=True,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        assert len(signals) == 1
        assert signals[0].metadata['ml_size_multiplier'] == 0.25
        assert signals[0].metadata['ml_confidence'] == 0.0
        assert signals[0].metadata['ml_gated'] is False

    def test_v2_size_position_applies_multiplier(self):
        """size_position scales base contracts by ml_size_multiplier."""
        ml_strat = _make_v2_strategy(
            prediction=_make_prediction(confidence=0.60),
            ml_sizing=True,
            contracts=4,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        sig = signals[0]

        ps = _make_portfolio_state()
        result = ml_strat.size_position(sig, ps)
        # 4 * 0.85 = 3.4 → round → 3
        assert result == 3

    def test_v2_size_position_never_zero(self):
        """Low multiplier with 1 base contract → 1 (not 0)."""
        ml_strat = _make_v2_strategy(
            prediction=_make_prediction(confidence=0.10),  # → 0.35 multiplier
            ml_sizing=True,
            contracts=1,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        sig = signals[0]

        ps = _make_portfolio_state()
        result = ml_strat.size_position(sig, ps)
        # 1 * 0.35 = 0.35 → round → 0 → max(1, 0) → 1
        assert result == 1


# ---------------------------------------------------------------------------
# TestKillSwitch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    """Test ml_sizing kill switch."""

    def test_kill_switch_false_uses_v1(self):
        """ml_sizing=False → binary gating (low confidence dropped)."""
        base = StubStrategy(signals=[_make_signal()])
        model = MagicMock(spec=SignalModel)
        model.predict.return_value = _make_prediction(confidence=0.10)

        ml_strat = MLEnhancedStrategy(
            base_strategy=base,
            signal_model=model,
            ml_config={'ml_sizing': False, 'confidence_threshold': 0.30},
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        # V1: confidence 0.10 < threshold 0.30 → dropped
        assert len(signals) == 0

    def test_kill_switch_true_uses_v2(self):
        """ml_sizing=True → confidence sizing (low confidence kept with small multiplier)."""
        ml_strat = _make_v2_strategy(
            prediction=_make_prediction(confidence=0.10),
            ml_sizing=True,
        )
        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        # V2: never drops, confidence 0.10 → multiplier 0.35
        assert len(signals) == 1
        assert signals[0].metadata['ml_size_multiplier'] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# TestBackwardCompatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Verify V1 behavior is unchanged."""

    def test_default_config_is_v1(self):
        """No ml_sizing key → V1 binary gating."""
        base = StubStrategy(signals=[_make_signal()])
        model = MagicMock(spec=SignalModel)
        model.predict.return_value = _make_prediction(confidence=0.10)

        ml_strat = MLEnhancedStrategy(
            base_strategy=base,
            signal_model=model,
            ml_config={'confidence_threshold': 0.30},
        )
        assert ml_strat.ml_sizing_enabled is False
        assert ml_strat.regime_router is None

        snap = _make_market_snapshot()
        signals = ml_strat.generate_signals(snap)
        # V1: low confidence dropped
        assert len(signals) == 0

    def test_v1_behavior_unchanged(self):
        """V1 high-confidence pass and low-confidence drop still work."""
        sigs = [_make_signal(), _make_signal()]
        model = MagicMock(spec=SignalModel)
        model.predict.side_effect = [
            _make_prediction(confidence=0.60),  # pass
            _make_prediction(confidence=0.10),  # drop
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


# ---------------------------------------------------------------------------
# TestSizingIntegration
# ---------------------------------------------------------------------------

class TestSizingIntegration:
    """Test sizing.calculate_dynamic_risk ml_confidence_multiplier."""

    def test_ml_multiplier_default_no_effect(self):
        from compass.sizing import calculate_dynamic_risk
        # Default ml_confidence_multiplier=1.0 → no change
        base = calculate_dynamic_risk(100_000, 35.0, 0.0)
        with_default = calculate_dynamic_risk(100_000, 35.0, 0.0, ml_confidence_multiplier=1.0)
        assert base == with_default

    def test_ml_multiplier_scales_risk(self):
        from compass.sizing import calculate_dynamic_risk
        base = calculate_dynamic_risk(100_000, 35.0, 0.0)
        half = calculate_dynamic_risk(100_000, 35.0, 0.0, ml_confidence_multiplier=0.5)
        assert abs(half - base * 0.5) < 0.01
