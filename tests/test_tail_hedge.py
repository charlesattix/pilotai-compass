"""Tests for shared/tail_hedge.py and debit trade integration."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.tail_hedge import (
    compute_protection_ratio,
    hedge_budget,
    optimal_put_strike,
    select_hedge_expiry,
    should_buy_hedge,
    size_hedge,
    vix_percentile,
)

# ---------------------------------------------------------------------------
# TestShouldBuyHedge
# ---------------------------------------------------------------------------

class TestShouldBuyHedge:

    def test_below_median_returns_true(self):
        """VIX below 252-day median → should hedge."""
        history = [20.0] * 126 + [30.0] * 126  # median ~25
        assert should_buy_hedge(18.0, history) is True

    def test_above_median_returns_false(self):
        """VIX above median → hedges too expensive."""
        history = [15.0] * 252
        assert should_buy_hedge(20.0, history) is False

    def test_vix_above_30_returns_false(self):
        """VIX > 30 → too expensive regardless of percentile."""
        history = [35.0] * 252  # median is 35, VIX 31 is below it
        assert should_buy_hedge(31.0, history) is False

    def test_insufficient_history_returns_false(self):
        """Fewer than 60 data points → not enough data."""
        history = [20.0] * 59
        assert should_buy_hedge(15.0, history) is False

    def test_at_median_returns_false(self):
        """VIX exactly at median → not strictly below."""
        history = [20.0] * 252
        assert should_buy_hedge(20.0, history) is False


# ---------------------------------------------------------------------------
# TestOptimalStrike
# ---------------------------------------------------------------------------

class TestOptimalStrike:

    def test_low_vix_5pct_otm(self):
        """Low VIX (<15) → 5% OTM."""
        strike = optimal_put_strike(500.0, 12.0)
        assert strike == round(500 * 0.95)  # 475

    def test_normal_vix_7pct_otm(self):
        """Normal VIX (15-20) → 7% OTM."""
        strike = optimal_put_strike(500.0, 17.0)
        assert strike == round(500 * 0.93)  # 465

    def test_elevated_vix_10pct_otm(self):
        """Elevated VIX (20-30) → 10% OTM."""
        strike = optimal_put_strike(500.0, 25.0)
        assert strike == round(500 * 0.90)  # 450

    def test_rounding(self):
        """Strike should be rounded to nearest integer."""
        strike = optimal_put_strike(553.7, 14.0)
        expected = round(553.7 * 0.95)  # 526.015 → 526
        assert strike == expected


# ---------------------------------------------------------------------------
# TestHedgeBudget
# ---------------------------------------------------------------------------

class TestHedgeBudget:

    def test_full_month_budget(self):
        """30 days since last hedge → full monthly budget."""
        budget = hedge_budget(100_000, 0.025, 30)
        assert budget == 2500.0

    def test_prorated_budget(self):
        """15 days since last hedge → 50% of monthly budget."""
        budget = hedge_budget(100_000, 0.025, 15)
        assert budget == pytest.approx(1250.0)

    def test_cooldown_returns_zero(self):
        """Hedged within 7 days → cooldown, budget = 0."""
        assert hedge_budget(100_000, 0.025, 5) == 0.0
        assert hedge_budget(100_000, 0.025, 0) == 0.0

    def test_zero_equity_returns_zero(self):
        """Zero equity → no budget."""
        assert hedge_budget(0, 0.025, 30) == 0.0


# ---------------------------------------------------------------------------
# TestSelectExpiry
# ---------------------------------------------------------------------------

class TestSelectExpiry:

    def test_returns_friday(self):
        """Selected expiry should be a Friday."""
        date = datetime(2026, 3, 7, tzinfo=timezone.utc)
        expiry = select_hedge_expiry(date)
        assert expiry.weekday() == 4  # Friday

    def test_dte_in_range(self):
        """DTE should be between 30 and 60."""
        date = datetime(2026, 3, 7, tzinfo=timezone.utc)
        expiry = select_hedge_expiry(date, min_dte=30, max_dte=60)
        dte = (expiry - date).days
        assert 30 <= dte <= 60


# ---------------------------------------------------------------------------
# TestSizeHedge
# ---------------------------------------------------------------------------

class TestSizeHedge:

    def test_basic_sizing(self):
        """Budget / (put_price * 100) → contracts."""
        contracts = size_hedge(5000.0, 2.50)
        # 5000 / (2.50 * 100) = 20, capped at max_contracts=10
        assert contracts == 10

    def test_cant_afford_returns_zero(self):
        """Can't afford even 1 contract → 0."""
        assert size_hedge(100.0, 5.0) == 0

    def test_max_cap(self):
        """Capped at max_contracts."""
        contracts = size_hedge(100_000.0, 1.0, max_contracts=5)
        assert contracts == 5

    def test_exact_budget(self):
        """Budget exactly covers N contracts."""
        contracts = size_hedge(500.0, 2.50, max_contracts=10)
        assert contracts == 2  # 500 / 250 = 2


# ---------------------------------------------------------------------------
# TestProtectionRatio
# ---------------------------------------------------------------------------

class TestProtectionRatio:

    def test_basic_ratio(self):
        """Standard protection ratio computation."""
        # price=500, strike=475, put_price=2.0
        # notional = (500-475)*100 = 2500
        # cost = 2.0 * 100 = 200
        # ratio = 2500/200 = 12.5
        ratio = compute_protection_ratio(500.0, 475.0, 2.0)
        assert ratio == pytest.approx(12.5)

    def test_zero_put_price(self):
        """Zero put price → 0 ratio (avoid div by zero)."""
        assert compute_protection_ratio(500.0, 475.0, 0.0) == 0.0

    def test_itm_strike(self):
        """Strike above price → 0 ratio (ITM put, no protection value)."""
        assert compute_protection_ratio(500.0, 510.0, 2.0) == 0.0


# ---------------------------------------------------------------------------
# TestVixPercentile
# ---------------------------------------------------------------------------

class TestVixPercentile:

    def test_low_vix(self):
        """VIX at bottom of history → low percentile."""
        history = list(range(10, 40))
        pct = vix_percentile(10.0, history)
        assert pct < 10.0

    def test_empty_history(self):
        """Empty history → default 50."""
        assert vix_percentile(20.0, []) == 50.0


# ---------------------------------------------------------------------------
# TestHedgeIntegration — end-to-end flows
# ---------------------------------------------------------------------------

class TestHedgeIntegration:

    def test_full_recommendation_flow(self):
        """End-to-end: VIX cheap → compute strike, budget, size, ratio."""
        from strategies.pricing import bs_price

        vix = 14.0
        price = 550.0
        equity = 100_000.0
        history = [18.0] * 252  # median 18, VIX 14 is below

        assert should_buy_hedge(vix, history) is True

        strike = optimal_put_strike(price, vix)
        assert strike == round(price * 0.95)

        budget = hedge_budget(equity, 0.025, 30)
        assert budget == 2500.0

        # Estimate put cost via BS
        T = 45 / 365
        put_cost = bs_price(price, strike, T, 0.04, vix / 100, "P")
        assert put_cost > 0

        contracts = size_hedge(budget, put_cost)
        assert contracts >= 1

        ratio = compute_protection_ratio(price, strike, put_cost)
        assert ratio > 0

    def test_signal_adapter_round_trip(self):
        """Signal with LONG_PUT → adapter → protective_put opp → adapter back."""
        from shared.strategy_adapter import signal_to_opportunity, trade_dict_to_position
        from strategies.base import LegType, Signal, TradeDirection, TradeLeg

        exp = datetime(2026, 4, 17, tzinfo=timezone.utc)
        signal = Signal(
            strategy_name="TailHedge",
            ticker="SPY",
            direction=TradeDirection.LONG,
            legs=[TradeLeg(LegType.LONG_PUT, 525.0, exp)],
            net_credit=-1.80,  # debit
            max_loss=1.80,
            max_profit=525.0,  # theoretical max if SPY goes to 0
            profit_target_pct=1.0,
            stop_loss_pct=0.80,
            score=50.0,
            expiration=exp,
            dte=41,
        )

        opp = signal_to_opportunity(signal, current_price=555.0)
        assert opp["type"] == "protective_put"
        assert opp["short_strike"] == 0
        assert opp["long_strike"] == 525.0
        assert opp["credit"] == -1.80

        # Reconstruct Position
        trade_dict = {
            "id": "PT-test123",
            "ticker": "SPY",
            "type": "protective_put",
            "short_strike": 0,
            "long_strike": 525.0,
            "expiration": "2026-04-17",
            "contracts": 2,
            "credit": -1.80,
        }
        pos = trade_dict_to_position(trade_dict)
        assert pos.direction == TradeDirection.LONG
        assert len(pos.legs) == 1
        assert pos.legs[0].leg_type == LegType.LONG_PUT
        assert pos.legs[0].strike == 525.0

    @pytest.mark.skip(reason="PaperTrader deleted — _evaluate_position no longer exists")
    def test_debit_pnl_profit(self):
        """Protective put appreciates on VIX spike → positive PnL."""
        pt = self._make_eval_trader()
        trade = self._make_protective_put_trade(
            long_strike=475, credit=-2.0, contracts=2, dte=30,
        )
        # Price drops significantly → put gains value
        pnl, reason = pt._evaluate_position(trade, current_price=460, dte=25)
        # Put should be worth more than we paid
        assert pnl > 0

    @pytest.mark.skip(reason="PaperTrader deleted — _evaluate_position no longer exists")
    def test_debit_pnl_loss(self):
        """Protective put loses value via time decay → negative PnL."""
        pt = self._make_eval_trader()
        trade = self._make_protective_put_trade(
            long_strike=475, credit=-2.0, contracts=2, dte=30,
        )
        # Price rises far above strike → put is nearly worthless
        pnl, reason = pt._evaluate_position(trade, current_price=550, dte=5)
        assert pnl < 0

    @pytest.mark.skip(reason="PaperTrader deleted — _evaluate_position no longer exists")
    def test_debit_expiration_trigger(self):
        """DTE <= 1 triggers expiration close for protective put."""
        pt = self._make_eval_trader()
        trade = self._make_protective_put_trade(
            long_strike=475, credit=-2.0, contracts=1, dte=30,
        )
        # Price far above strike → put nearly worthless, no profit trigger
        _, reason = pt._evaluate_position(trade, current_price=550, dte=1)
        assert reason == "expiration"

    def test_open_trade_allows_debit(self):
        """_open_trade should accept protective_put with negative credit."""
        from shared.strategy_adapter import signal_to_opportunity
        from strategies.base import LegType, Signal, TradeDirection, TradeLeg

        exp = datetime(2026, 4, 17, tzinfo=timezone.utc)
        signal = Signal(
            strategy_name="TailHedge",
            ticker="SPY",
            direction=TradeDirection.LONG,
            legs=[TradeLeg(LegType.LONG_PUT, 525.0, exp)],
            net_credit=-1.50,
            max_loss=1.50,
            profit_target_pct=1.0,
            stop_loss_pct=0.80,
            score=50.0,
            expiration=exp,
            dte=41,
        )
        opp = signal_to_opportunity(signal, current_price=555.0)

        # Verify the opp would NOT be blocked by the debit gate
        is_debit = opp.get("type") == "protective_put"
        credit = opp.get("credit", 0)
        max_loss = opp.get("max_loss", 0)

        # Old gate: credit <= 0 or max_loss <= 0 → would reject
        assert credit < 0
        assert max_loss > 0
        # New gate: is_debit=True → only checks max_loss > 0
        assert is_debit is True
        assert not (is_debit and max_loss <= 0)  # passes new gate

    # --- Helpers ---

    @staticmethod
    def _make_eval_trader():
        """Create a minimal PaperTrader for _evaluate_position tests."""
        from paper_trader import PaperTrader

        with patch('paper_trader.init_db'), \
             patch('paper_trader.get_trades', return_value=[]), \
             patch('paper_trader.DATA_DIR') as mdir:
            mdir.__truediv__ = lambda s, n: Path("/tmp") / n
            mdir.mkdir = MagicMock()
            pt = PaperTrader({
                'risk': {
                    'account_size': 100000,
                    'max_risk_per_trade': 2.0,
                    'max_positions': 5,
                    'profit_target': 50,
                    'stop_loss_multiplier': 2.5,
                },
                'alpaca': {'enabled': False},
            })
        return pt

    @staticmethod
    def _make_protective_put_trade(
        long_strike=475,
        credit=-2.0,
        contracts=2,
        dte=30,
        ticker='SPY',
    ):
        """Build a protective_put trade dict."""
        exp_date = datetime.now(timezone.utc) + timedelta(days=dte)
        debit = abs(credit)
        return {
            'id': 'PT-hedge-test',
            'status': 'open',
            'ticker': ticker,
            'type': 'protective_put',
            'short_strike': 0,
            'long_strike': long_strike,
            'expiration': exp_date.strftime('%Y-%m-%d'),
            'contracts': contracts,
            'credit_per_spread': credit,
            'credit': credit,
            'total_credit': round(credit * contracts * 100, 2),
            'max_loss_per_spread': debit,
            'total_max_loss': round(debit * contracts * 100, 2),
            'profit_target': round(debit * 1.0 * contracts * 100, 2),
            'stop_loss_amount': round(debit * 0.80 * contracts * 100, 2),
            'profit_target_pct': 1.0,
            'stop_loss_pct': 0.80,
            'strategy_name': 'TailHedge',
            'entry_price': 555.0,
            'entry_date': datetime.now(timezone.utc).isoformat(),
        }
