"""
Unit tests for COMPASS position sizing utilities.

Target module: compass/sizing.py (post-Phase 2 move)
Pre-move source: ml/position_sizer.py (top-level functions only)

Tests cover:
  - calculate_dynamic_risk() — IV-rank tiers, flat override, heat cap, max_risk_pct
  - get_contract_size() — standard case, max cap, degenerate inputs

Blueprint spec: 5+ tests, all green (Phase 3 exit criteria).
"""

import pytest

from compass.sizing import calculate_dynamic_risk, get_contract_size

from tests.compass_helpers import (
    ACCOUNT_100K,
    ACCOUNT_10K,
    SPREAD_WIDTH_5,
    CREDIT_065,
    MAX_LOSS_PER_CONTRACT_5_065,
)


# ══════════════════════════════════════════════════════════════════════════════
# A. calculate_dynamic_risk() — IV-rank tier schedule
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicRiskIVRankTiers:
    """IV-rank mode: IVR < 20 -> 1%, 20-50 -> 2%, >50 -> up to 3%."""

    def test_low_ivr_returns_1_pct(self):
        """IVR=10 (< 20) -> 0.5x baseline = 1% of account -> $1,000 on $100K."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=10.0, current_portfolio_risk=0.0
        )
        assert result == pytest.approx(1000.0)

    def test_standard_ivr_returns_2_pct(self):
        """IVR=35 (20-50) -> 1x baseline = 2% of account -> $2,000 on $100K."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=0.0
        )
        assert result == pytest.approx(2000.0)

    def test_high_ivr_returns_up_to_3_pct(self):
        """IVR=75 (> 50) -> between 2% and 3% of account."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=75.0, current_portfolio_risk=0.0
        )
        assert 2000.0 < result < 3000.0

    def test_ivr_100_returns_max_tier(self):
        """IVR=100 -> 1.5x baseline = 3% of account -> $3,000 on $100K."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=100.0, current_portfolio_risk=0.0
        )
        assert result == pytest.approx(3000.0)

    def test_ivr_boundary_20_is_standard(self):
        """IVR=20 (inclusive) -> standard tier (2%)."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=20.0, current_portfolio_risk=0.0
        )
        assert result == pytest.approx(2000.0)

    def test_ivr_boundary_50_is_standard(self):
        """IVR=50 (inclusive) -> standard tier (2%)."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=50.0, current_portfolio_risk=0.0
        )
        assert result == pytest.approx(2000.0)


# ══════════════════════════════════════════════════════════════════════════════
# B. calculate_dynamic_risk() — flat_risk_pct override
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicRiskFlatOverride:

    def test_flat_risk_pct_overrides_iv_tiers(self):
        """flat_risk_pct=5.0 -> 5% of account regardless of IVR."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=10.0, current_portfolio_risk=0.0,
            flat_risk_pct=5.0,
        )
        assert result == pytest.approx(5000.0)

    def test_flat_risk_pct_ignores_max_risk_pct(self):
        """max_risk_pct has no effect when flat_risk_pct is set."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=10.0, current_portfolio_risk=0.0,
            flat_risk_pct=5.0, max_risk_pct=1.0,
        )
        # flat_risk_pct=5.0 should dominate, max_risk_pct=1.0 is ignored
        assert result == pytest.approx(5000.0)


# ══════════════════════════════════════════════════════════════════════════════
# C. calculate_dynamic_risk() — portfolio heat cap (40%)
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicRiskHeatCap:

    def test_heat_cap_reduces_budget(self):
        """38% heat used -> budget reduced to fit under 40% ceiling."""
        # 38% of $100K = $38K already used. Max heat = 40% = $40K.
        # Room left = $2K. A standard IVR=35 would want $2K, so exactly fits.
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=38_000.0
        )
        assert result == pytest.approx(2000.0)

        # If we use 39% ($39K), only $1K room remains
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=39_000.0
        )
        assert result == pytest.approx(1000.0)

    def test_heat_cap_full_returns_zero(self):
        """Portfolio risk at 40%+ of account -> returns 0.0."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=50.0, current_portfolio_risk=40_000.0
        )
        assert result == 0.0

        # Also above 40%
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=50.0, current_portfolio_risk=45_000.0
        )
        assert result == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# D. calculate_dynamic_risk() — max_risk_pct cap
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicRiskMaxRiskPct:

    def test_max_risk_pct_caps_iv_rank_result(self):
        """max_risk_pct=1.5 caps the IV-rank-derived budget at 1.5%."""
        # IVR=35 would give 2% ($2K), but max_risk_pct=1.5 caps to $1.5K
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=0.0,
            max_risk_pct=1.5,
        )
        assert result == pytest.approx(1500.0)

    def test_max_risk_pct_no_effect_on_flat(self):
        """max_risk_pct is ignored when flat_risk_pct is provided."""
        result = calculate_dynamic_risk(
            account_value=ACCOUNT_100K, iv_rank=35.0, current_portfolio_risk=0.0,
            flat_risk_pct=3.0, max_risk_pct=1.5,
        )
        assert result == pytest.approx(3000.0)


# ══════════════════════════════════════════════════════════════════════════════
# E. get_contract_size()
# ══════════════════════════════════════════════════════════════════════════════

class TestGetContractSize:

    def test_standard_case(self):
        """$1,000 risk, $5 spread, $0.65 credit -> floor($1000 / $435) = 2 contracts."""
        result = get_contract_size(
            trade_dollar_risk=1000.0,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
        )
        # max_loss = (5.0 - 0.65) * 100 = $435
        # floor(1000 / 435) = 2
        assert result == 2

    def test_max_contracts_cap(self):
        """Large risk budget -> capped at max_contracts (default 5)."""
        result = get_contract_size(
            trade_dollar_risk=10_000.0,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
        )
        # floor(10000 / 435) = 22, but capped at 5
        assert result == 5

    def test_custom_max_contracts(self):
        """max_contracts=10 allows more contracts when budget permits."""
        result = get_contract_size(
            trade_dollar_risk=5000.0,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
            max_contracts=10,
        )
        # floor(5000 / 435) = 11, but capped at 10
        assert result == 10

    def test_credit_exceeds_spread_returns_zero(self):
        """credit_received >= spread_width -> max_loss_per_contract <= 0 -> 0 contracts."""
        result = get_contract_size(
            trade_dollar_risk=1000.0,
            spread_width=5.0,
            credit_received=5.0,
        )
        assert result == 0

    def test_zero_risk_budget_returns_zero(self):
        """trade_dollar_risk=0 -> 0 contracts."""
        result = get_contract_size(
            trade_dollar_risk=0.0,
            spread_width=SPREAD_WIDTH_5,
            credit_received=CREDIT_065,
        )
        assert result == 0
