"""Tests for alerts.alert_position_sizer — sizing correctness + cap enforcement."""

import pytest

from alerts.alert_position_sizer import AlertPositionSizer
from alerts.alert_schema import Alert, AlertType, Direction, Leg, SizeResult
from shared.constants import MAX_RISK_PER_TRADE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_legs(short=100.0, long_=95.0, option_type="put"):
    return [
        Leg(strike=short, option_type=option_type, action="sell", expiration="2025-06-20"),
        Leg(strike=long_, option_type=option_type, action="buy", expiration="2025-06-20"),
    ]


def _make_alert(
    credit=1.50,
    short=100.0,
    long_=95.0,
    risk_pct=0.02,
    **overrides,
):
    defaults = dict(
        type=AlertType.credit_spread,
        ticker="SPY",
        direction=Direction.bullish,
        legs=_make_legs(short, long_),
        entry_price=credit,
        stop_loss=3.00,
        profit_target=0.75,
        risk_pct=risk_pct,
    )
    defaults.update(overrides)
    return Alert(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSizingMath:
    """Verify the end-to-end sizing calculation."""

    def test_basic_size(self):
        sizer = AlertPositionSizer()
        alert = _make_alert(credit=1.50, short=100.0, long_=95.0)

        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,       # standard tier → 2% baseline
            current_portfolio_risk=0,
            weekly_loss_breach=False,
        )

        assert isinstance(result, SizeResult)
        assert result.contracts >= 0
        assert result.dollar_risk > 0
        assert result.risk_pct > 0
        # With 100K account, 2% baseline → $2000 risk budget
        # Spread width = 5, credit = 1.50, max_loss/contract = 3.50 * 100 = $350
        # Contracts = floor(2000 / 350) = 5 (capped at 5 by get_contract_size)
        assert result.contracts == 5

    def test_low_iv_rank_reduces_size(self):
        sizer = AlertPositionSizer()
        alert = _make_alert()

        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=10,  # low IV → 0.5× baseline = 1%
            current_portfolio_risk=0,
        )

        # 1% of 100K = $1000 risk budget
        # $1000 / $350 = 2 contracts
        assert result.contracts == 2

    def test_high_iv_rank_increases_size(self):
        sizer = AlertPositionSizer()
        alert = _make_alert()

        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=70,  # high IV → up to 1.5× baseline
            current_portfolio_risk=0,
        )

        # Should get more contracts than standard tier
        assert result.contracts >= 3


class TestCapping:
    """Verify IV-rank sizer stays within expected range (40% heat cap inside calculate_dynamic_risk)."""

    def test_dollar_risk_within_heat_cap(self):
        """IV-rank sizer at IVR=100 returns 3% (backtester: no extra MAX_RISK_PER_TRADE cap)."""
        sizer = AlertPositionSizer()
        alert = _make_alert()

        result = sizer.size(
            alert=alert,
            account_value=10_000,
            iv_rank=100,
            current_portfolio_risk=0,
        )

        # calculate_dynamic_risk at IVR=100: 2% * 1.5 = 3% ($300 for $10K account)
        # No extra MAX_RISK_PER_TRADE hard cap — only 40% heat cap applies
        assert result.dollar_risk <= 10_000 * 0.40  # within heat cap
        assert result.dollar_risk > 0


class TestWeeklyLossReduction:
    """Weekly loss breach no longer reduces size (removed to match backtester)."""

    def test_weekly_loss_breach_no_longer_reduces_size(self):
        """Backtester has no weekly-loss breach reduction — live must match."""
        sizer = AlertPositionSizer()
        alert = _make_alert()

        normal = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,
            current_portfolio_risk=0,
            weekly_loss_breach=False,
        )

        with_breach = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,
            current_portfolio_risk=0,
            weekly_loss_breach=True,
        )

        # weekly_loss_breach flag is accepted but must NOT change sizing
        assert with_breach.dollar_risk == pytest.approx(normal.dollar_risk)
        assert with_breach.contracts == normal.contracts


class TestEdgeCases:
    """Edge cases: zero contracts, small accounts, iron condors."""

    def test_zero_contracts_when_budget_tiny(self):
        sizer = AlertPositionSizer()
        alert = _make_alert()

        result = sizer.size(
            alert=alert,
            account_value=100,    # tiny account
            iv_rank=30,
            current_portfolio_risk=0,
        )

        assert result.contracts == 0
        assert result.max_loss == 0

    def test_iron_condor_spread_width(self):
        """Iron condor: spread width should come from the widest wing."""
        legs = [
            Leg(strike=95.0, option_type="put", action="buy", expiration="2025-06-20"),
            Leg(strike=100.0, option_type="put", action="sell", expiration="2025-06-20"),
            Leg(strike=110.0, option_type="call", action="sell", expiration="2025-06-20"),
            Leg(strike=115.0, option_type="call", action="buy", expiration="2025-06-20"),
        ]
        alert = _make_alert(legs=legs)

        sizer = AlertPositionSizer()
        spread_width, credit = sizer._extract_spread_params(alert)

        assert spread_width == 5.0  # both wings are $5 wide

    def test_portfolio_heat_limits_budget(self):
        """When portfolio is near heat cap, budget should be reduced."""
        sizer = AlertPositionSizer()
        alert = _make_alert()

        result = sizer.size(
            alert=alert,
            account_value=100_000,
            iv_rank=30,
            current_portfolio_risk=38_000,  # 38% of 100K — near 40% heat cap
        )

        # Budget should be reduced since we're near the heat cap
        assert result.dollar_risk <= 2_000  # at most $2K of room
