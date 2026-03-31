"""
Tests for compass/regime_gate.py

Coverage:
  - Default gate decisions for all 6 regimes
  - Configurable halt_regimes and allow_regimes
  - Configurable min_win_rate threshold
  - should_trade() convenience method
  - position_scale() returns correct values
  - Regime enum interop (accepts compass.regime.Regime values)
  - String input: lowercase, uppercase, underscore variants
  - Unknown regime: allow (default) and halt actions
  - GateDecision.to_dict() shape
  - regime_summary() completeness
  - Edge cases: None input, crash regime, neutral
"""

import pytest

from compass.regime_gate import (
    BEAR,
    BULL,
    CRASH,
    HIGH_VOL,
    LOW_VOL,
    NEUTRAL,
    HISTORICAL_WIN_RATES,
    GateDecision,
    RegimeGate,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gate():
    """Default RegimeGate with no custom config."""
    return RegimeGate()


@pytest.fixture
def strict_gate():
    """Gate that also halts neutral regime."""
    return RegimeGate({"halt_regimes": [BEAR, HIGH_VOL, CRASH, NEUTRAL]})


@pytest.fixture
def low_threshold_gate():
    """Gate with a lower min_win_rate threshold (0.60)."""
    return RegimeGate({"min_win_rate": 0.60})


# ── Default gate decisions ────────────────────────────────────────────────────

class TestDefaultGate:

    def test_bull_allows_trade(self, gate):
        assert gate.should_trade(BULL) is True

    def test_low_vol_allows_trade(self, gate):
        assert gate.should_trade(LOW_VOL) is True

    def test_neutral_allows_trade(self, gate):
        assert gate.should_trade(NEUTRAL) is True

    def test_bear_halts_trade(self, gate):
        assert gate.should_trade(BEAR) is False

    def test_high_vol_halts_trade(self, gate):
        assert gate.should_trade(HIGH_VOL) is False

    def test_crash_halts_trade(self, gate):
        assert gate.should_trade(CRASH) is False


# ── GateDecision content ──────────────────────────────────────────────────────

class TestGateDecisionContent:

    def test_bear_decision_has_reason(self, gate):
        d = gate.evaluate(BEAR)
        assert isinstance(d.reason, str)
        assert len(d.reason) > 10
        assert "bear" in d.reason.lower() or "Bear" in d.reason

    def test_bear_decision_position_scale_zero(self, gate):
        d = gate.evaluate(BEAR)
        assert d.position_scale == 0.0

    def test_bull_decision_full_scale(self, gate):
        d = gate.evaluate(BULL)
        assert d.position_scale == 1.0

    def test_neutral_decision_reduced_scale(self, gate):
        d = gate.evaluate(NEUTRAL)
        assert 0.0 < d.position_scale < 1.0

    def test_decision_regime_is_normalized(self, gate):
        d = gate.evaluate("BEAR")  # uppercase input
        assert d.regime == "bear"

    def test_bear_historical_wr_present(self, gate):
        d = gate.evaluate(BEAR)
        assert d.historical_wr is not None
        assert 0.0 < d.historical_wr < 1.0

    def test_crash_historical_wr_none(self, gate):
        # crash has no CS historical data
        d = gate.evaluate(CRASH)
        assert d.historical_wr is None

    def test_high_vol_not_trading(self, gate):
        d = gate.evaluate(HIGH_VOL)
        assert d.should_trade is False
        assert d.position_scale == 0.0

    def test_bull_should_trade_and_scale_consistent(self, gate):
        d = gate.evaluate(BULL)
        assert d.should_trade is True
        assert d.position_scale > 0.0

    def test_to_dict_has_required_keys(self, gate):
        d = gate.evaluate(BULL).to_dict()
        for key in ("should_trade", "regime", "reason", "position_scale",
                    "historical_wr", "confidence"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_position_scale_rounded(self, gate):
        d = gate.evaluate(NEUTRAL).to_dict()
        # Should be rounded to 2 dp
        assert d["position_scale"] == round(d["position_scale"], 2)


# ── position_scale() helper ───────────────────────────────────────────────────

class TestPositionScale:

    def test_bull_full_scale(self, gate):
        assert gate.position_scale(BULL) == 1.0

    def test_low_vol_full_scale(self, gate):
        assert gate.position_scale(LOW_VOL) == 1.0

    def test_bear_zero_scale(self, gate):
        assert gate.position_scale(BEAR) == 0.0

    def test_crash_zero_scale(self, gate):
        assert gate.position_scale(CRASH) == 0.0

    def test_high_vol_zero_scale(self, gate):
        assert gate.position_scale(HIGH_VOL) == 0.0

    def test_neutral_partial_scale(self, gate):
        s = gate.position_scale(NEUTRAL)
        assert 0.0 < s < 1.0


# ── Regime enum interop ───────────────────────────────────────────────────────

class TestRegimeEnumInterop:

    def test_regime_enum_bear(self):
        """Should work with compass.regime.Regime enum values."""
        from compass.regime import Regime  # type: ignore[import]
        gate = RegimeGate()
        assert gate.should_trade(Regime.BEAR) is False

    def test_regime_enum_bull(self):
        from compass.regime import Regime
        gate = RegimeGate()
        assert gate.should_trade(Regime.BULL) is True

    def test_regime_enum_high_vol(self):
        from compass.regime import Regime
        gate = RegimeGate()
        assert gate.should_trade(Regime.HIGH_VOL) is False

    def test_regime_enum_low_vol(self):
        from compass.regime import Regime
        gate = RegimeGate()
        assert gate.should_trade(Regime.LOW_VOL) is True

    def test_regime_enum_crash(self):
        from compass.regime import Regime
        gate = RegimeGate()
        assert gate.should_trade(Regime.CRASH) is False


# ── String normalization ──────────────────────────────────────────────────────

class TestStringNormalization:

    def test_uppercase_bear(self, gate):
        assert gate.should_trade("BEAR") is False

    def test_uppercase_bull(self, gate):
        assert gate.should_trade("BULL") is True

    def test_mixed_case_high_vol(self, gate):
        assert gate.should_trade("High_Vol") is False

    def test_whitespace_stripped(self, gate):
        assert gate.should_trade("  bull  ") is True


# ── Configurable halt / allow sets ───────────────────────────────────────────

class TestConfigurableRegimeSets:

    def test_strict_gate_halts_neutral(self, strict_gate):
        assert strict_gate.should_trade(NEUTRAL) is False

    def test_strict_gate_still_allows_bull(self, strict_gate):
        assert strict_gate.should_trade(BULL) is True

    def test_custom_allow_list_with_cleared_halt(self):
        """Bear regime can be allowed by clearing halt_regimes and setting allow_regimes."""
        # halt_regimes takes highest priority, so both must be configured
        permissive = RegimeGate({
            "halt_regimes": [],   # clear all halts
            "allow_regimes": [BEAR, BULL, NEUTRAL, LOW_VOL, HIGH_VOL, CRASH],
        })
        assert permissive.should_trade(BEAR) is True

    def test_custom_halt_single_regime(self):
        gate = RegimeGate({"halt_regimes": [BULL]})
        assert gate.should_trade(BULL) is False

    def test_custom_position_scale_override(self):
        gate = RegimeGate({"position_scales": {NEUTRAL: 0.5}})
        assert gate.position_scale(NEUTRAL) == 0.5

    def test_custom_position_scale_does_not_affect_other_regimes(self):
        gate = RegimeGate({"position_scales": {NEUTRAL: 0.5}})
        assert gate.position_scale(BULL) == 1.0


# ── min_win_rate threshold ────────────────────────────────────────────────────

class TestMinWinRateThreshold:

    def test_low_threshold_allows_bear_via_wr_check(self):
        """With min_win_rate=0.60, bear (64.7% WR) should be allowed if not in halt set."""
        gate = RegimeGate({
            "min_win_rate": 0.60,
            "halt_regimes": [HIGH_VOL, CRASH],  # bear not explicitly halted
            "allow_regimes": [BULL, LOW_VOL, NEUTRAL],
        })
        assert gate.should_trade(BEAR) is True

    def test_high_threshold_halts_bear_via_wr_check(self):
        """With min_win_rate=0.90 and bear not in halt set, bear should be halted by WR."""
        gate = RegimeGate({
            "min_win_rate": 0.90,
            "halt_regimes": [HIGH_VOL, CRASH],
            "allow_regimes": [BULL, LOW_VOL],
        })
        # BEAR is neither in halt nor allow, so WR check applies: 64.7% < 90% → halt
        assert gate.should_trade(BEAR) is False

    def test_high_threshold_halts_neutral_via_wr_check(self):
        """With min_win_rate=0.90, neutral (84% WR) should be halted."""
        gate = RegimeGate({
            "min_win_rate": 0.90,
            "halt_regimes": [BEAR, HIGH_VOL, CRASH],
            "allow_regimes": [BULL, LOW_VOL],
        })
        assert gate.should_trade(NEUTRAL) is False


# ── Unknown regime handling ───────────────────────────────────────────────────

class TestUnknownRegime:

    def test_unknown_regime_default_allows(self, gate):
        d = gate.evaluate("sideways_chop")
        assert d.should_trade is True

    def test_unknown_regime_zero_confidence(self, gate):
        d = gate.evaluate("sideways_chop")
        assert d.confidence == 0.0

    def test_unknown_regime_halt_config(self):
        gate = RegimeGate({"unknown_regime_action": "halt"})
        assert gate.should_trade("sideways_chop") is False

    def test_invalid_unknown_regime_action_raises(self):
        with pytest.raises(ValueError, match="unknown_regime_action"):
            RegimeGate({"unknown_regime_action": "maybe"})


# ── regime_summary() ─────────────────────────────────────────────────────────

class TestRegimeSummary:

    def test_summary_contains_all_regimes(self, gate):
        summary = gate.regime_summary()
        for regime in (BULL, BEAR, HIGH_VOL, LOW_VOL, CRASH, NEUTRAL):
            assert regime in summary

    def test_summary_values_are_dicts(self, gate):
        summary = gate.regime_summary()
        for v in summary.values():
            assert isinstance(v, dict)
            assert "should_trade" in v

    def test_summary_bear_not_trading(self, gate):
        assert gate.regime_summary()[BEAR]["should_trade"] is False

    def test_summary_bull_trading(self, gate):
        assert gate.regime_summary()[BULL]["should_trade"] is True


# ── Confidence scores ─────────────────────────────────────────────────────────

class TestConfidence:

    def test_bull_has_high_confidence(self, gate):
        """Bull has n=208 data points — should have high confidence."""
        d = gate.evaluate(BULL)
        assert d.confidence >= 0.7

    def test_high_vol_has_low_confidence(self, gate):
        """high_vol has n=2 data points — should have low confidence."""
        d = gate.evaluate(HIGH_VOL)
        assert d.confidence <= 0.5

    def test_crash_has_low_confidence(self, gate):
        d = gate.evaluate(CRASH)
        assert d.confidence <= 0.3

    def test_confidence_in_valid_range(self, gate):
        for regime in (BULL, BEAR, HIGH_VOL, LOW_VOL, CRASH, NEUTRAL):
            c = gate.evaluate(regime).confidence
            assert 0.0 <= c <= 1.0, f"Confidence out of range for {regime}: {c}"


# ── Historical win rates ──────────────────────────────────────────────────────

class TestHistoricalWinRates:

    def test_bull_wr_above_80pct(self):
        assert HISTORICAL_WIN_RATES[BULL] > 0.80

    def test_bear_wr_below_75pct(self):
        assert HISTORICAL_WIN_RATES[BEAR] < 0.75

    def test_high_vol_wr_at_or_below_55pct(self):
        assert HISTORICAL_WIN_RATES[HIGH_VOL] <= 0.55

    def test_crash_wr_is_none(self):
        assert HISTORICAL_WIN_RATES[CRASH] is None


# ── Integration: simulated trade sequence ────────────────────────────────────

class TestIntegration:

    def test_regime_sequence_blocks_bear(self, gate):
        """In a simulated year with regime shifts, bear blocks should fire."""
        sequence = [BULL, BULL, BEAR, BEAR, BULL, HIGH_VOL, BULL, NEUTRAL]
        results = [gate.should_trade(r) for r in sequence]
        expected = [True, True, False, False, True, False, True, True]
        assert results == expected

    def test_2022_bear_year_simulation(self, gate):
        """
        2022 analogue: sustained bear regime all year.
        Gate should block all 20 CS trades, preventing 8 losses.
        """
        trades_2022 = [BEAR] * 20
        taken = [gate.should_trade(r) for r in trades_2022]
        assert sum(taken) == 0   # all blocked

    def test_2021_bull_year_simulation(self, gate):
        """2021 analogue: sustained bull regime. All 70 trades should be allowed."""
        trades_2021 = [BULL] * 70
        taken = [gate.should_trade(r) for r in trades_2021]
        assert sum(taken) == 70  # all allowed

    def test_position_scale_sequence(self, gate):
        """Position scales for a mixed regime sequence."""
        regimes = [BULL, NEUTRAL, BEAR, CRASH]
        scales = [gate.position_scale(r) for r in regimes]
        assert scales[0] == 1.0          # bull: full
        assert 0.0 < scales[1] < 1.0    # neutral: partial
        assert scales[2] == 0.0          # bear: zero
        assert scales[3] == 0.0          # crash: zero
