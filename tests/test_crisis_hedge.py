"""
Comprehensive tests for compass/crisis_hedge.py.

Coverage:
  - CrisisHedgeConfig dataclass defaults and custom values
  - _vix_scale() piecewise linear function at all breakpoints + interior
  - position_scale_factor(): normal/crash/high_vol regimes, term structure
  - stop_loss_multiplier(): linear interpolation, boundary conditions, regime gates
  - get_audit_metadata(): all keys, is_throttled/is_halted flags, reason strings
  - Monotonicity: scale/stop must be monotone w.r.t. VIX
  - Boundary clamping: outputs always in valid range
  - Hysteresis state tracking
  - log_decisions=False silences log output
"""

from __future__ import annotations

import math
import pytest

from compass.crisis_hedge import CrisisHedgeConfig, CrisisHedgeController


# ─── Fixtures ─────────────────────────────────────────────────────────────────

# Pin to explicit floor=20/ceiling=50 config for breakpoint math tests.
# These tests verify the _vix_scale algorithm, not the current defaults.
_MATH_TEST_CONFIG = CrisisHedgeConfig(
    vix_scale_floor=20.0, vix_scale_ceiling=50.0,
    vix_stop_floor=20.0, vix_stop_ceiling=45.0,
    log_decisions=False,
)


@pytest.fixture
def default_ctrl() -> CrisisHedgeController:
    return CrisisHedgeController(_MATH_TEST_CONFIG)


@pytest.fixture
def silent_ctrl() -> CrisisHedgeController:
    return CrisisHedgeController(_MATH_TEST_CONFIG)


# ─── CrisisHedgeConfig ────────────────────────────────────────────────────────

class TestCrisisHedgeConfig:

    def test_default_vix_scale_floor(self):
        assert CrisisHedgeConfig().vix_scale_floor == 12.0

    def test_default_vix_scale_ceiling(self):
        assert CrisisHedgeConfig().vix_scale_ceiling == 35.0

    def test_default_base_stop_multiplier(self):
        assert CrisisHedgeConfig().base_stop_multiplier == 3.5

    def test_default_min_stop_multiplier(self):
        assert CrisisHedgeConfig().min_stop_multiplier == 1.5

    def test_default_vix_stop_floor(self):
        assert CrisisHedgeConfig().vix_stop_floor == 12.0

    def test_default_vix_stop_ceiling(self):
        assert CrisisHedgeConfig().vix_stop_ceiling == 25.8

    def test_default_crash_regime_scale(self):
        assert CrisisHedgeConfig().crash_regime_scale == 0.0

    def test_default_high_vol_regime_scale(self):
        assert CrisisHedgeConfig().high_vol_regime_scale == 0.25

    def test_default_use_vix_term_structure(self):
        assert CrisisHedgeConfig().use_vix_term_structure is True

    def test_default_vix_ts_backwardation_penalty(self):
        assert CrisisHedgeConfig().vix_ts_backwardation_penalty == 0.25

    def test_default_recovery_hysteresis_vix(self):
        assert CrisisHedgeConfig().recovery_hysteresis_vix == 3.0

    def test_default_log_decisions(self):
        assert CrisisHedgeConfig().log_decisions is True

    def test_custom_config_overrides(self):
        cfg = CrisisHedgeConfig(
            vix_scale_floor=15.0,
            vix_scale_ceiling=40.0,
            base_stop_multiplier=4.0,
            min_stop_multiplier=2.0,
            crash_regime_scale=0.0,
            high_vol_regime_scale=0.10,
            log_decisions=False,
        )
        assert cfg.vix_scale_floor == 15.0
        assert cfg.vix_scale_ceiling == 40.0
        assert cfg.base_stop_multiplier == 4.0
        assert cfg.min_stop_multiplier == 2.0
        assert cfg.high_vol_regime_scale == 0.10
        assert cfg.log_decisions is False


# ─── _vix_scale() breakpoints and interior ────────────────────────────────────

class TestVixScalePrivate:
    """Test _vix_scale() in isolation at every documented breakpoint."""

    def test_at_or_below_floor_is_one(self, default_ctrl):
        assert default_ctrl._vix_scale(10.0) == 1.0
        assert default_ctrl._vix_scale(20.0) == 1.0

    def test_at_or_above_ceiling_is_zero(self, default_ctrl):
        assert default_ctrl._vix_scale(50.0) == 0.0
        assert default_ctrl._vix_scale(80.0) == 0.0

    def test_midpoint_segment1_vix25(self, default_ctrl):
        # Segment 1 (20–30): at VIX=25 (halfway) → 1.0 - 0.50*(5/10) = 0.75
        assert default_ctrl._vix_scale(25.0) == pytest.approx(0.75, abs=1e-9)

    def test_boundary_segment1_end_vix30(self, default_ctrl):
        # End of segment 1 → 0.50
        assert default_ctrl._vix_scale(30.0) == pytest.approx(0.50, abs=1e-6)

    def test_midpoint_segment2_vix35(self, default_ctrl):
        # Segment 2 (30–40): at VIX=35 (halfway) → 0.50 - 0.40*(5/10) = 0.30
        assert default_ctrl._vix_scale(35.0) == pytest.approx(0.30, abs=1e-9)

    def test_boundary_segment2_end_vix40(self, default_ctrl):
        # End of segment 2 → 0.10
        assert default_ctrl._vix_scale(40.0) == pytest.approx(0.10, abs=1e-6)

    def test_midpoint_segment3_vix45(self, default_ctrl):
        # Segment 3 (40–50): at VIX=45 (halfway) → 0.10 - 0.10*(5/10) = 0.05
        assert default_ctrl._vix_scale(45.0) == pytest.approx(0.05, abs=1e-9)

    def test_monotone_across_range(self, default_ctrl):
        vixes = [5, 15, 20, 22, 25, 28, 30, 33, 35, 38, 40, 42, 45, 48, 50, 60]
        scales = [default_ctrl._vix_scale(v) for v in vixes]
        for i in range(1, len(scales)):
            assert scales[i] <= scales[i - 1], (
                f"_vix_scale not monotone: vix={vixes[i]} scale={scales[i]} "
                f"> vix={vixes[i-1]} scale={scales[i-1]}"
            )

    def test_always_in_zero_one(self, default_ctrl):
        for vix in range(0, 100):
            s = default_ctrl._vix_scale(float(vix))
            assert 0.0 <= s <= 1.0, f"_vix_scale({vix}) = {s} out of [0,1]"

    def test_custom_floor_ceiling(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(
            vix_scale_floor=10.0, vix_scale_ceiling=40.0, log_decisions=False
        ))
        assert ctrl._vix_scale(10.0) == 1.0
        assert ctrl._vix_scale(40.0) == 0.0
        # Midpoint of segment 1: (10+40)/3 offset = 10.0 → VIX=20 is halfway of seg1
        # seg=10, t=10 → t==seg → boundary between seg1 and seg2 → 0.50
        assert ctrl._vix_scale(20.0) == pytest.approx(0.50, abs=1e-6)


# ─── position_scale_factor() ──────────────────────────────────────────────────

class TestPositionScaleFactor:

    def test_normal_regime_below_floor_is_one(self, silent_ctrl):
        assert silent_ctrl.position_scale_factor(vix=15.0) == pytest.approx(1.0)

    def test_normal_regime_above_ceiling_is_zero(self, silent_ctrl):
        assert silent_ctrl.position_scale_factor(vix=55.0) == 0.0

    def test_neutral_regime_same_as_no_regime(self, silent_ctrl):
        assert (
            silent_ctrl.position_scale_factor(vix=25.0, regime="neutral")
            == silent_ctrl.position_scale_factor(vix=25.0, regime=None)
        )

    def test_bull_regime_same_as_no_regime(self, silent_ctrl):
        assert (
            silent_ctrl.position_scale_factor(vix=25.0, regime="bull")
            == silent_ctrl.position_scale_factor(vix=25.0)
        )

    def test_crash_regime_always_zero(self, silent_ctrl):
        for vix in [10, 15, 20, 30, 50]:
            assert silent_ctrl.position_scale_factor(vix=float(vix), regime="crash") == 0.0

    def test_crash_regime_case_insensitive(self, silent_ctrl):
        assert silent_ctrl.position_scale_factor(vix=15.0, regime="CRASH") == 0.0
        assert silent_ctrl.position_scale_factor(vix=15.0, regime="Crash") == 0.0

    def test_high_vol_regime_capped_at_0_25(self, silent_ctrl):
        # In high_vol, scale = min(vix_scale, 0.25)
        # At VIX=15 (below floor): vix_scale=1.0 → min(1.0, 0.25) = 0.25
        assert silent_ctrl.position_scale_factor(vix=15.0, regime="high_vol") == pytest.approx(0.25)

    def test_high_vol_regime_vix_scale_wins_when_smaller(self, silent_ctrl):
        # At VIX=45 (vix_scale≈0.05 < 0.25): should use vix_scale
        scale = silent_ctrl.position_scale_factor(vix=45.0, regime="high_vol")
        assert scale < 0.25
        assert scale >= 0.0

    def test_result_always_in_zero_one(self, silent_ctrl):
        regimes = [None, "bull", "bear", "neutral", "high_vol", "crash", "low_vol"]
        for vix in [5, 15, 20, 25, 30, 35, 40, 45, 50, 60]:
            for r in regimes:
                s = silent_ctrl.position_scale_factor(vix=float(vix), regime=r)
                assert 0.0 <= s <= 1.0, f"scale={s} for vix={vix} regime={r}"

    def test_result_rounded_to_4_decimal_places(self, silent_ctrl):
        s = silent_ctrl.position_scale_factor(vix=27.3)
        # Should be rounded to 4dp
        assert s == round(s, 4)

    # VIX term structure tests

    def test_contango_no_penalty(self, silent_ctrl):
        # vix3m > vix = contango → no penalty
        s_no_ts = silent_ctrl.position_scale_factor(vix=25.0)
        s_contango = silent_ctrl.position_scale_factor(vix=25.0, vix3m=30.0)
        assert s_contango == pytest.approx(s_no_ts, abs=1e-4)

    def test_backwardation_applies_penalty(self, silent_ctrl):
        # vix3m < vix = backwardation → scale should be lower
        s_no_ts = silent_ctrl.position_scale_factor(vix=25.0)
        s_backward = silent_ctrl.position_scale_factor(vix=25.0, vix3m=20.0)
        assert s_backward < s_no_ts

    def test_backwardation_penalty_capped_at_config(self):
        # inversion_depth = 0.4 → penalty = min(0.25, 0.4*2=0.80) = 0.25
        ctrl = CrisisHedgeController(CrisisHedgeConfig(
            vix_scale_floor=20.0, vix_scale_ceiling=50.0,
            vix_ts_backwardation_penalty=0.25, log_decisions=False
        ))
        # vix=25 (scale=0.75), vix3m=15 (ratio=0.6, inversion=0.4)
        # penalty = min(0.25, 0.4*2) = 0.25 → scale = 0.75 * 0.75 = 0.5625
        s = ctrl.position_scale_factor(vix=25.0, vix3m=15.0)
        assert s == pytest.approx(0.75 * 0.75, abs=1e-3)

    def test_backwardation_small_inversion(self):
        # inversion_depth=0.05 → penalty = min(0.25, 0.10) = 0.10
        ctrl = CrisisHedgeController(CrisisHedgeConfig(
            vix_scale_floor=20.0, vix_scale_ceiling=50.0,
            vix_ts_backwardation_penalty=0.25, log_decisions=False
        ))
        s_flat = ctrl.position_scale_factor(vix=25.0)     # vix=vix3m → no penalty
        s_inv = ctrl.position_scale_factor(vix=25.0, vix3m=23.75)  # ratio=0.95
        # inversion_depth=0.05, penalty=min(0.25, 0.1)=0.10
        # scale = 0.75 * (1 - 0.10) = 0.675
        assert s_inv == pytest.approx(0.75 * 0.90, abs=1e-3)

    def test_term_structure_disabled(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(
            vix_scale_floor=20.0, vix_scale_ceiling=50.0,
            use_vix_term_structure=False, log_decisions=False
        ))
        s_no_ts = ctrl.position_scale_factor(vix=25.0)
        s_backward = ctrl.position_scale_factor(vix=25.0, vix3m=15.0)
        assert s_no_ts == pytest.approx(s_backward, abs=1e-9)

    def test_vix3m_none_skips_term_structure(self, silent_ctrl):
        s1 = silent_ctrl.position_scale_factor(vix=25.0, vix3m=None)
        s2 = silent_ctrl.position_scale_factor(vix=25.0)
        assert s1 == pytest.approx(s2, abs=1e-9)

    def test_monotone_with_vix(self, silent_ctrl):
        vixes = list(range(10, 65, 2))
        scales = [silent_ctrl.position_scale_factor(vix=float(v)) for v in vixes]
        for i in range(1, len(scales)):
            assert scales[i] <= scales[i - 1] + 1e-9, (
                f"Not monotone: vix={vixes[i]} scale={scales[i]} > prev scale={scales[i-1]}"
            )


# ─── stop_loss_multiplier() ───────────────────────────────────────────────────

class TestStopLossMultiplier:

    def test_below_stop_floor_returns_base(self, silent_ctrl):
        assert silent_ctrl.stop_loss_multiplier(vix=10.0) == pytest.approx(3.5)
        assert silent_ctrl.stop_loss_multiplier(vix=20.0) == pytest.approx(3.5)

    def test_above_stop_ceiling_returns_min(self, silent_ctrl):
        assert silent_ctrl.stop_loss_multiplier(vix=45.0) == pytest.approx(1.5)
        assert silent_ctrl.stop_loss_multiplier(vix=80.0) == pytest.approx(1.5)

    def test_midpoint_vix_32_5(self, silent_ctrl):
        # floor=20, ceiling=45, midpoint=32.5 → t=0.5 → mult = 3.5 - 0.5*(3.5-1.5) = 2.5
        m = silent_ctrl.stop_loss_multiplier(vix=32.5)
        assert m == pytest.approx(2.5, abs=1e-3)

    def test_linear_interpolation_at_vix_25(self, silent_ctrl):
        # floor=20, ceiling=45, vix=25 → t = 5/25 = 0.2 → 3.5 - 0.2*2.0 = 3.1
        m = silent_ctrl.stop_loss_multiplier(vix=25.0)
        assert m == pytest.approx(3.1, abs=1e-3)

    def test_crash_regime_always_min(self, silent_ctrl):
        for vix in [10, 15, 20, 30, 50]:
            assert silent_ctrl.stop_loss_multiplier(float(vix), regime="crash") == pytest.approx(1.5)

    def test_crash_regime_case_insensitive(self, silent_ctrl):
        assert silent_ctrl.stop_loss_multiplier(15.0, regime="CRASH") == pytest.approx(1.5)

    def test_neutral_regime_same_as_none(self, silent_ctrl):
        assert (
            silent_ctrl.stop_loss_multiplier(30.0, regime="neutral")
            == silent_ctrl.stop_loss_multiplier(30.0, regime=None)
        )

    def test_high_vol_regime_not_special_cased(self, silent_ctrl):
        # high_vol regime only affects position_scale_factor, not stop_loss_multiplier
        m_neutral = silent_ctrl.stop_loss_multiplier(25.0, regime="neutral")
        m_high_vol = silent_ctrl.stop_loss_multiplier(25.0, regime="high_vol")
        assert m_neutral == m_high_vol

    def test_always_in_valid_range(self, silent_ctrl):
        for vix in range(0, 100):
            m = silent_ctrl.stop_loss_multiplier(float(vix))
            assert 1.5 <= m <= 3.5, f"stop mult={m} at vix={vix} out of [1.5, 3.5]"

    def test_monotone_decreasing_with_vix(self, silent_ctrl):
        vixes = [5, 10, 15, 20, 22, 25, 28, 30, 35, 40, 45, 50, 60]
        mults = [silent_ctrl.stop_loss_multiplier(float(v)) for v in vixes]
        for i in range(1, len(mults)):
            assert mults[i] <= mults[i - 1] + 1e-9, (
                f"Not monotone: vix={vixes[i]} mult={mults[i]} > prev mult={mults[i-1]}"
            )

    def test_custom_config_base_and_min(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(
            base_stop_multiplier=4.0,
            min_stop_multiplier=2.0,
            vix_stop_floor=25.0,
            vix_stop_ceiling=50.0,
            log_decisions=False,
        ))
        assert ctrl.stop_loss_multiplier(vix=20.0) == pytest.approx(4.0)
        assert ctrl.stop_loss_multiplier(vix=50.0) == pytest.approx(2.0)
        # Midpoint: t=0.5 → 4.0 - 0.5*(4.0-2.0) = 3.0
        assert ctrl.stop_loss_multiplier(vix=37.5) == pytest.approx(3.0, abs=1e-3)


# ─── get_audit_metadata() ────────────────────────────────────────────────────

class TestGetAuditMetadata:

    def test_all_keys_present(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=15.0)
        expected_keys = {
            "scale_factor", "stop_multiplier", "regime", "vix", "vix3m",
            "ts_ratio", "is_backwardated", "is_throttled", "is_halted", "reason",
        }
        assert expected_keys <= set(meta.keys())

    def test_normal_conditions(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=15.0, regime="neutral")
        assert meta["scale_factor"] == pytest.approx(1.0)
        assert meta["stop_multiplier"] == pytest.approx(3.5)
        assert meta["is_throttled"] is False
        assert meta["is_halted"] is False
        assert meta["is_backwardated"] is False
        assert meta["reason"] == "NORMAL"

    def test_halted_on_crash_regime(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=15.0, regime="crash")
        assert meta["scale_factor"] == 0.0
        assert meta["is_halted"] is True
        assert meta["is_throttled"] is True
        assert "HALTED" in meta["reason"]

    def test_heavy_throttle_high_vix(self, silent_ctrl):
        # VIX=45 → scale≈0.05 (< 0.5) → HEAVY_THROTTLE
        meta = silent_ctrl.get_audit_metadata(vix=45.0)
        assert meta["scale_factor"] < 0.5
        assert meta["is_throttled"] is True
        assert "HEAVY_THROTTLE" in meta["reason"] or "HALTED" in meta["reason"]

    def test_light_throttle_moderate_vix(self, silent_ctrl):
        # VIX=25 → scale=0.75 (between 0.5 and 1.0) → LIGHT_THROTTLE
        meta = silent_ctrl.get_audit_metadata(vix=25.0)
        assert 0.5 <= meta["scale_factor"] < 1.0
        assert meta["is_throttled"] is True
        assert meta["is_halted"] is False
        assert "LIGHT_THROTTLE" in meta["reason"]

    def test_stop_tightened_in_reason(self, silent_ctrl):
        # VIX=35 → stop < 3.5 → STOP_TIGHTENED in reason
        meta = silent_ctrl.get_audit_metadata(vix=35.0)
        assert meta["stop_multiplier"] < 3.5
        assert "STOP_TIGHTENED" in meta["reason"]

    def test_backwardation_flag(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=25.0, vix3m=20.0)
        assert meta["is_backwardated"] is True
        assert meta["ts_ratio"] == pytest.approx(20.0 / 25.0, abs=1e-6)
        assert "VIX_BACKWARDATED" in meta["reason"]

    def test_contango_not_backwardated(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=15.0, vix3m=20.0)
        assert meta["is_backwardated"] is False
        assert "VIX_BACKWARDATED" not in meta["reason"]

    def test_vix3m_none_ts_ratio_none(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=15.0, vix3m=None)
        assert meta["vix3m"] is None
        assert meta["ts_ratio"] is None
        assert meta["is_backwardated"] is False

    def test_regime_stored_correctly(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=15.0, regime="bear")
        assert meta["regime"] == "bear"

    def test_none_regime_stored_as_neutral(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=15.0, regime=None)
        assert meta["regime"] == "neutral"

    def test_vix_stored_correctly(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=27.3)
        assert meta["vix"] == 27.3

    def test_reason_not_empty(self, silent_ctrl):
        for vix in [10, 25, 35, 50]:
            meta = silent_ctrl.get_audit_metadata(vix=float(vix))
            assert len(meta["reason"]) > 0

    def test_high_vol_regime_metadata(self, silent_ctrl):
        meta = silent_ctrl.get_audit_metadata(vix=15.0, regime="high_vol")
        assert meta["scale_factor"] == pytest.approx(0.25, abs=1e-4)
        assert meta["is_throttled"] is True
        assert meta["is_halted"] is False


# ─── Hysteresis state tracking ────────────────────────────────────────────────

class TestHysteresisState:

    def test_initial_last_scale_factor_is_one(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        assert ctrl._last_scale_factor == 1.0

    def test_initial_below_hysteresis_true(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        assert ctrl._below_hysteresis_threshold is True

    def test_last_scale_factor_updated_after_non_regime_call(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        ctrl.position_scale_factor(vix=25.0)
        expected = ctrl._vix_scale(25.0)
        assert ctrl._last_scale_factor == pytest.approx(expected, abs=1e-6)

    def test_last_scale_factor_not_updated_on_crash_regime(self):
        ctrl = CrisisHedgeController(CrisisHedgeConfig(log_decisions=False))
        ctrl.position_scale_factor(vix=10.0)  # below floor=12 → sets _last_scale_factor=1.0
        ctrl.position_scale_factor(vix=10.0, regime="crash")
        # crash returns early without updating _last_scale_factor
        assert ctrl._last_scale_factor == pytest.approx(1.0, abs=1e-6)


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_vix_zero_returns_full_scale(self, silent_ctrl):
        assert silent_ctrl.position_scale_factor(vix=0.0) == pytest.approx(1.0)

    def test_vix_very_large_returns_zero_scale(self, silent_ctrl):
        assert silent_ctrl.position_scale_factor(vix=1000.0) == 0.0

    def test_stop_vix_zero_returns_base(self, silent_ctrl):
        assert silent_ctrl.stop_loss_multiplier(vix=0.0) == pytest.approx(3.5)

    def test_stop_vix_very_large_returns_min(self, silent_ctrl):
        assert silent_ctrl.stop_loss_multiplier(vix=1000.0) == pytest.approx(1.5)

    def test_controller_initialises_without_config(self):
        ctrl = CrisisHedgeController()
        assert ctrl.cfg.vix_scale_floor == 12.0

    def test_controller_initialises_with_none_config(self):
        ctrl = CrisisHedgeController(config=None)
        assert ctrl.cfg.base_stop_multiplier == 3.5

    def test_unknown_regime_treated_as_neutral(self, silent_ctrl):
        s_neutral = silent_ctrl.position_scale_factor(vix=25.0, regime="neutral")
        s_unknown = silent_ctrl.position_scale_factor(vix=25.0, regime="whatever")
        assert s_neutral == pytest.approx(s_unknown, abs=1e-6)

    def test_whitespace_in_regime_stripped(self, silent_ctrl):
        s = silent_ctrl.position_scale_factor(vix=15.0, regime="  crash  ")
        assert s == 0.0

    def test_floor_equals_ceiling_degenerate_config(self):
        """Degenerate config where floor == ceiling should not crash.

        When floor == ceiling, the <= floor check fires before >= ceiling,
        so VIX exactly at the boundary returns 1.0; VIX just above returns 0.0.
        The key invariant is that the code does not raise.
        """
        ctrl = CrisisHedgeController(CrisisHedgeConfig(
            vix_scale_floor=30.0, vix_scale_ceiling=30.0, log_decisions=False
        ))
        # Below floor → 1.0
        assert ctrl._vix_scale(25.0) == 1.0
        # At floor (== ceiling): <= floor wins → 1.0
        assert ctrl._vix_scale(30.0) == 1.0
        # Above ceiling → 0.0
        assert ctrl._vix_scale(31.0) == 0.0


# ─── COVID scenario verification ─────────────────────────────────────────────

class TestCovidScenario:
    """Verify the scaling curve produces expected values from the design doc."""

    def test_vix_15_full_size(self, silent_ctrl):
        """Pre-crash: full size."""
        assert silent_ctrl.position_scale_factor(vix=15.0) == pytest.approx(1.0)

    def test_vix_25_moderate_throttle(self, silent_ctrl):
        """Day 4-7 of COVID crash (VIX ~25): scale should be ~0.75."""
        s = silent_ctrl.position_scale_factor(vix=25.0)
        assert 0.5 < s < 1.0

    def test_vix_40_near_shutdown(self, silent_ctrl):
        """Day 10+ of COVID (VIX ~40): scale should be ~0.10."""
        s = silent_ctrl.position_scale_factor(vix=40.0)
        assert s == pytest.approx(0.10, abs=1e-6)

    def test_vix_82_full_shutdown(self, silent_ctrl):
        """COVID peak VIX=82: no new entries."""
        s = silent_ctrl.position_scale_factor(vix=82.0)
        assert s == 0.0

    def test_stop_at_vix_15_is_base(self, silent_ctrl):
        assert silent_ctrl.stop_loss_multiplier(vix=15.0) == pytest.approx(3.5)

    def test_stop_at_vix_40_tighter(self, silent_ctrl):
        m = silent_ctrl.stop_loss_multiplier(vix=40.0)
        assert m < 3.5
        assert m >= 1.5

    def test_stop_at_vix_82_is_minimum(self, silent_ctrl):
        assert silent_ctrl.stop_loss_multiplier(vix=82.0) == pytest.approx(1.5)
