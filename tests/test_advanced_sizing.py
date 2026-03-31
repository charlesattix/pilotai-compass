"""Tests for compass/advanced_sizing.py — regime-adaptive fractional Kelly."""

from __future__ import annotations

import math

import pytest

from compass.advanced_sizing import (
    AdvancedPositionSizer,
    SizingConfig,
    SizingResult,
    correlation_scale,
    drawdown_scale,
    kelly_criterion,
)


# ── kelly_criterion ──────────────────────────────────────────────────────


class TestKellyCriterion:

    def test_positive_edge(self):
        # 85% win rate, 0.5 return on risk, 1.0 loss → positive Kelly
        k = kelly_criterion(0.85, 0.50, 1.00)
        assert k > 0

    def test_no_edge(self):
        # 50% win, equal payoff → zero
        k = kelly_criterion(0.50, 1.0, 1.0)
        assert k == 0.0

    def test_negative_edge(self):
        # 30% win, 0.2 return, 1.0 loss → no bet
        k = kelly_criterion(0.30, 0.20, 1.00)
        assert k == 0.0

    def test_high_probability(self):
        k = kelly_criterion(0.90, 0.50, 1.00)
        assert k > 0.5

    def test_invalid_prob_zero(self):
        assert kelly_criterion(0.0, 0.30, 1.00) == 0.0

    def test_invalid_prob_one(self):
        assert kelly_criterion(1.0, 0.30, 1.00) == 0.0

    def test_invalid_returns(self):
        assert kelly_criterion(0.60, 0.0, 1.00) == 0.0
        assert kelly_criterion(0.60, 0.30, 0.0) == 0.0
        assert kelly_criterion(0.60, -0.10, 1.00) == 0.0

    def test_clamped_to_one(self):
        # Very high edge: should be clamped at 1.0
        k = kelly_criterion(0.99, 10.0, 0.01)
        assert k <= 1.0

    def test_typical_credit_spread(self):
        # 70% WR, 30% return on risk, 100% max loss
        k = kelly_criterion(0.70, 0.30, 1.00)
        # Kelly = (0.7*0.3 - 0.3) / 0.3 = (0.21 - 0.3)/0.3 = -0.3 → 0
        # Wait: p*b - q / b = (0.7 * 0.3 - 0.3) / 0.3 = -0.09/0.3 = -0.3
        # Negative → 0. With higher win_return it becomes positive.
        assert k == 0.0

    def test_credit_spread_positive_edge(self):
        # 85% WR, 50% return on risk, 100% max loss
        k = kelly_criterion(0.85, 0.50, 1.00)
        # b = 0.5, Kelly = (0.85*0.5 - 0.15) / 0.5 = (0.425 - 0.15)/0.5 = 0.55
        assert 0.5 < k < 0.6

    def test_moderate_credit_spread(self):
        # 75% WR, 40% return, 100% loss → Kelly = 0.125
        k = kelly_criterion(0.75, 0.40, 1.00)
        assert 0.12 < k < 0.13


# ── drawdown_scale ───────────────────────────────────────────────────────


class TestDrawdownScale:

    def test_no_drawdown(self):
        assert drawdown_scale(0.0, 30.0) == 1.0

    def test_below_start(self):
        # DD = 10%, max = 30%, start = 50% of 30 = 15
        assert drawdown_scale(10.0, 30.0) == 1.0

    def test_at_start(self):
        # DD = 15%, exactly at start threshold (50% of 30)
        assert drawdown_scale(15.0, 30.0) == 1.0

    def test_midpoint(self):
        # DD = 21%, start=15, exit=27 → midpoint → scale ≈ 0.5
        scale = drawdown_scale(21.0, 30.0)
        assert 0.45 < scale < 0.55

    def test_at_exit(self):
        # DD = 27%, exactly at exit threshold (90% of 30)
        assert drawdown_scale(27.0, 30.0) == 0.0

    def test_above_exit(self):
        assert drawdown_scale(29.0, 30.0) == 0.0

    def test_max_dd_zero(self):
        # Edge case: no max DD → always full size
        assert drawdown_scale(10.0, 0.0) == 1.0

    def test_negative_dd_treated_as_positive(self):
        # Negative DD (passed as -8.0) → abs → 8.0
        assert drawdown_scale(-8.0, 30.0) == 1.0

    def test_linear_scaling(self):
        # Verify linearity between start and exit
        s1 = drawdown_scale(18.0, 30.0)
        s2 = drawdown_scale(21.0, 30.0)
        s3 = drawdown_scale(24.0, 30.0)
        assert s1 > s2 > s3
        # Equal spacing → equal decrements
        assert abs((s1 - s2) - (s2 - s3)) < 0.01

    def test_custom_thresholds(self):
        # Start at 30%, exit at 80%
        assert drawdown_scale(5.0, 20.0, scale_start=0.30, scale_exit=0.80) == 1.0
        assert drawdown_scale(16.0, 20.0, scale_start=0.30, scale_exit=0.80) == 0.0


# ── correlation_scale ────────────────────────────────────────────────────


class TestCorrelationScale:

    def test_below_threshold(self):
        assert correlation_scale(0.40, threshold=0.60) == 1.0

    def test_at_threshold(self):
        assert correlation_scale(0.60, threshold=0.60) == 1.0

    def test_above_threshold(self):
        scale = correlation_scale(0.70, threshold=0.60, penalty_per_0_1=0.15)
        # Excess = 0.10, penalty = 1 * 0.15 = 0.15, scale = 0.85
        assert abs(scale - 0.85) < 0.01

    def test_high_correlation(self):
        scale = correlation_scale(0.90, threshold=0.60, penalty_per_0_1=0.15)
        # Excess = 0.30, penalty = 3 * 0.15 = 0.45, scale = 0.55
        assert abs(scale - 0.55) < 0.01

    def test_extreme_correlation(self):
        # 1.0 correlation with aggressive penalty
        scale = correlation_scale(1.0, threshold=0.50, penalty_per_0_1=0.25)
        # Excess = 0.50, penalty = 5 * 0.25 = 1.25, scale = max(0, -0.25) = 0
        assert scale == 0.0

    def test_zero_correlation(self):
        assert correlation_scale(0.0) == 1.0

    def test_negative_correlation(self):
        # Negative correlation → below threshold → full size
        assert correlation_scale(-0.30, threshold=0.60) == 1.0


# ── SizingConfig ─────────────────────────────────────────────────────────


class TestSizingConfig:

    def test_defaults(self):
        cfg = SizingConfig()
        assert cfg.max_position_pct == 10.0
        assert cfg.max_dd_pct == 30.0
        assert cfg.dd_scale_start == 0.50
        assert cfg.dd_scale_exit == 0.90
        assert cfg.correlation_threshold == 0.60

    def test_regime_fractions_defaults(self):
        cfg = SizingConfig()
        assert cfg.regime_fractions["bull"] == 0.75
        assert cfg.regime_fractions["crash"] == 0.25
        assert cfg.regime_fractions["bear"] == 0.50

    def test_custom_config(self):
        cfg = SizingConfig(max_dd_pct=20.0, max_position_pct=5.0)
        assert cfg.max_dd_pct == 20.0
        assert cfg.max_position_pct == 5.0


# ── AdvancedPositionSizer.compute ────────────────────────────────────────


class TestAdvancedPositionSizerCompute:

    @pytest.fixture
    def sizer(self):
        return AdvancedPositionSizer()

    def test_basic_bull_regime(self, sizer):
        r = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.00,
            regime="bull",
        )
        assert r.position_fraction > 0
        assert r.kelly_fraction == 0.75
        assert r.dd_scale == 1.0
        assert r.corr_scale == 1.0
        assert r.regime == "bull"

    def test_crash_regime_smaller(self, sizer):
        bull = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.00,
            regime="bull",
        )
        crash = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.00,
            regime="crash",
        )
        assert crash.position_fraction < bull.position_fraction
        assert crash.kelly_fraction == 0.25
        assert bull.kelly_fraction == 0.75

    def test_bear_between_bull_and_crash(self, sizer):
        bull = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, regime="bull")
        bear = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, regime="bear")
        crash = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, regime="crash")
        assert crash.position_fraction <= bear.position_fraction <= bull.position_fraction

    def test_drawdown_reduces_size(self, sizer):
        normal = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, regime="bull", current_dd_pct=0.0)
        stressed = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, regime="bull", current_dd_pct=20.0)
        assert stressed.position_fraction < normal.position_fraction
        assert stressed.dd_scale < 1.0

    def test_deep_drawdown_exits(self, sizer):
        r = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.00,
            regime="bull", current_dd_pct=28.0,  # > 90% of 30
        )
        assert r.position_fraction == 0.0
        assert r.dd_scale == 0.0

    def test_correlation_reduces_size(self, sizer):
        low_corr = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, portfolio_correlation=0.30)
        high_corr = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, portfolio_correlation=0.80)
        assert high_corr.position_fraction < low_corr.position_fraction
        assert high_corr.corr_scale < 1.0

    def test_combined_dd_and_correlation(self, sizer):
        r = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.00,
            regime="high_vol", current_dd_pct=18.0, portfolio_correlation=0.75,
        )
        assert r.dd_scale < 1.0
        assert r.corr_scale < 1.0
        assert r.kelly_fraction == 0.25
        # Should be very small
        assert r.position_fraction < 0.05

    def test_negative_edge_returns_zero(self, sizer):
        r = sizer.compute(win_prob=0.30, win_return=0.20, loss_return=1.00, regime="bull")
        assert r.position_fraction == 0.0
        assert r.kelly_raw == 0.0

    def test_max_position_cap(self):
        cfg = SizingConfig(max_position_pct=2.0)
        sizer = AdvancedPositionSizer(config=cfg)
        r = sizer.compute(win_prob=0.95, win_return=5.0, loss_return=0.10, regime="bull")
        assert r.position_fraction <= 0.02
        assert r.capped

    def test_constraints_logged(self, sizer):
        r = sizer.compute(
            win_prob=0.75, win_return=0.40, loss_return=1.00,
            regime="bull", current_dd_pct=20.0, portfolio_correlation=0.75,
        )
        assert len(r.constraints) >= 2
        assert any("DD" in c for c in r.constraints)
        assert any("Corr" in c for c in r.constraints)

    def test_unknown_regime_uses_base_fraction(self):
        sizer = AdvancedPositionSizer()
        r = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, regime="sideways")
        assert r.kelly_fraction == 0.50  # base_kelly_fraction default

    def test_result_is_dataclass(self, sizer):
        r = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00)
        assert isinstance(r, SizingResult)


# ── Per-experiment overrides ─────────────────────────────────────────────


class TestExperimentOverrides:

    def test_override_applied(self):
        default_cfg = SizingConfig(max_dd_pct=30.0)
        exp401_cfg = SizingConfig(max_dd_pct=20.0, max_position_pct=5.0)
        sizer = AdvancedPositionSizer(
            config=default_cfg,
            experiment_overrides={"EXP-401": exp401_cfg},
        )
        # Default experiment
        r_default = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00)
        # EXP-401
        r_401 = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, experiment_id="EXP-401")
        # EXP-401 should have tighter cap
        assert sizer.get_config("EXP-401").max_position_pct == 5.0
        assert sizer.get_config().max_position_pct == 10.0

    def test_unknown_experiment_uses_default(self):
        sizer = AdvancedPositionSizer(
            experiment_overrides={"EXP-401": SizingConfig(max_dd_pct=15.0)},
        )
        cfg = sizer.get_config("EXP-999")
        assert cfg.max_dd_pct == 30.0  # default

    def test_override_regime_fractions(self):
        conservative = SizingConfig(
            regime_fractions={"bull": 0.50, "bear": 0.25, "crash": 0.10,
                              "high_vol": 0.10, "low_vol": 0.50},
        )
        sizer = AdvancedPositionSizer(config=conservative)
        r = sizer.compute(win_prob=0.75, win_return=0.40, loss_return=1.00, regime="bull")
        assert r.kelly_fraction == 0.50


# ── compute_contracts ────────────────────────────────────────────────────


class TestComputeContracts:

    def test_basic_contract_count(self):
        sizer = AdvancedPositionSizer()
        contracts = sizer.compute_contracts(
            account_value=100_000, spread_width=5.0, credit_received=0.65,
            win_prob=0.75, win_return=0.40, loss_return=1.00, regime="bull",
        )
        assert isinstance(contracts, int)
        assert contracts >= 0

    def test_max_contracts_cap(self):
        sizer = AdvancedPositionSizer()
        contracts = sizer.compute_contracts(
            account_value=1_000_000, spread_width=5.0, credit_received=0.65,
            max_contracts=3,
            win_prob=0.95, win_return=5.0, loss_return=0.10, regime="bull",
        )
        assert contracts <= 3

    def test_zero_when_no_edge(self):
        sizer = AdvancedPositionSizer()
        contracts = sizer.compute_contracts(
            account_value=100_000, spread_width=5.0, credit_received=0.65,
            win_prob=0.30, win_return=0.20, loss_return=1.00, regime="bull",
        )
        assert contracts == 0

    def test_zero_when_deep_drawdown(self):
        sizer = AdvancedPositionSizer()
        contracts = sizer.compute_contracts(
            account_value=100_000, spread_width=5.0, credit_received=0.65,
            win_prob=0.75, win_return=0.40, loss_return=1.00,
            regime="bull", current_dd_pct=28.0,
        )
        assert contracts == 0

    def test_zero_max_loss(self):
        sizer = AdvancedPositionSizer()
        # credit >= width → max_loss_per ≤ 0
        contracts = sizer.compute_contracts(
            account_value=100_000, spread_width=5.0, credit_received=6.0,
            win_prob=0.75, win_return=0.40, loss_return=1.00,
        )
        assert contracts == 0
