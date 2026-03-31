"""Tests for straddle/strangle execution pipeline.

Covers:
- AlpacaProvider.submit_single_leg() / close_single_leg()
- ExecutionEngine._submit_straddle() debit/credit handling
- PositionMonitor straddle close reconciliation (dual-leg)
- P&L calculation for long/short straddles
- Dry-run logging with straddle-specific fields
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

from shared.database import init_db, upsert_trade, get_trade_by_id
from execution.execution_engine import ExecutionEngine
from execution.position_monitor import PositionMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_alpaca(**overrides):
    """Build a mock AlpacaProvider with sensible defaults."""
    m = MagicMock()
    m.get_market_clock.return_value = {"is_open": True}
    m.submit_single_leg.return_value = {
        "status": "submitted",
        "order_id": "ord-call-001",
    }
    m.close_single_leg.return_value = {
        "status": "submitted",
        "order_id": "close-call-001",
    }
    m.cancel_order.return_value = True
    m.get_positions.return_value = []
    m.get_order_status.return_value = {
        "status": "filled",
        "filled_avg_price": "3.00",
        "filled_at": "2026-03-15T12:00:00Z",
        "filled_qty": "1",
    }
    m._build_occ_symbol.side_effect = lambda t, e, s, ot: f"{t}{e}{s}{ot}"
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


def _make_straddle_opp(**overrides):
    """Build a straddle/strangle opportunity dict."""
    opp = {
        "ticker": "SPY",
        "type": "short_straddle",
        "call_strike": 500.0,
        "put_strike": 500.0,
        "expiration": "2026-04-17",
        "credit": 8.50,
        "contracts": 2,
        "is_debit": False,
    }
    opp.update(overrides)
    return opp


def _make_long_straddle_opp(**overrides):
    """Build a long (debit) straddle opportunity."""
    opp = {
        "ticker": "SPY",
        "type": "long_straddle",
        "call_strike": 500.0,
        "put_strike": 500.0,
        "expiration": "2026-04-17",
        "credit": -6.00,  # negative = debit
        "contracts": 1,
        "is_debit": True,
        "event_type": "fomc",
    }
    opp.update(overrides)
    return opp


def _make_straddle_trade(trade_id="ss-001", **overrides):
    """Build a straddle trade dict as stored in DB."""
    trade = {
        "id": trade_id,
        "ticker": "SPY",
        "strategy_type": "short_straddle",
        "status": "open",
        "short_strike": 0,
        "long_strike": 0,
        "call_strike": 500.0,
        "put_strike": 500.0,
        "expiration": "2026-04-17",
        "credit": 8.50,
        "contracts": 2,
        "entry_date": "2026-03-15T10:00:00Z",
        "is_debit": False,
    }
    trade.update(overrides)
    return trade


def _setup_db():
    """Create a temp DB and return its path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    init_db(path)
    return path


def _monitor(alpaca=None, db_path=None, config=None):
    """Build a PositionMonitor with sensible defaults."""
    return PositionMonitor(
        alpaca_provider=alpaca or _make_alpaca(),
        config=config or {
            "risk": {"profit_target": 50, "stop_loss_multiplier": 3.5},
            "strategy": {"manage_dte": 0},
            "execution": {"commission_per_contract": 0},
        },
        db_path=db_path,
    )


# ===========================================================================
# AlpacaProvider.submit_single_leg tests
# ===========================================================================

class TestSubmitSingleLeg(unittest.TestCase):
    """Test the submit_single_leg method on AlpacaProvider."""

    @patch("strategy.alpaca_provider.AlpacaProvider.__init__", return_value=None)
    def test_submit_single_leg_buy_call(self, mock_init):
        """Buy a single call option."""
        from strategy.alpaca_provider import AlpacaProvider

        provider = AlpacaProvider.__new__(AlpacaProvider)
        provider._circuit_breaker = MagicMock()
        provider.client = MagicMock()

        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_order.client_order_id = "sl-SPY-abc"
        mock_order.status = "accepted"
        mock_order.submitted_at = "2026-03-15T10:00:00Z"
        provider._circuit_breaker.call.return_value = mock_order

        # Mock find_option_symbol to return a symbol
        provider.find_option_symbol = MagicMock(return_value="SPY260417C00500000")

        result = provider.submit_single_leg(
            ticker="SPY",
            strike=500.0,
            expiration="2026-04-17",
            option_type="call",
            side="buy",
            contracts=1,
            limit_price=3.50,
            client_order_id="sl-SPY-abc",
        )

        assert result["status"] == "submitted"
        assert result["order_id"] == "order-123"
        assert result["side"] == "buy"
        assert result["option_type"] == "call"
        assert result["strike"] == 500.0

    @patch("strategy.alpaca_provider.AlpacaProvider.__init__", return_value=None)
    def test_submit_single_leg_sell_put(self, mock_init):
        """Sell a single put option."""
        from strategy.alpaca_provider import AlpacaProvider

        provider = AlpacaProvider.__new__(AlpacaProvider)
        provider._circuit_breaker = MagicMock()
        provider.client = MagicMock()

        mock_order = MagicMock()
        mock_order.id = "order-456"
        mock_order.client_order_id = "sl-SPY-def"
        mock_order.status = "accepted"
        mock_order.submitted_at = "2026-03-15T10:00:00Z"
        provider._circuit_breaker.call.return_value = mock_order
        provider.find_option_symbol = MagicMock(return_value="SPY260417P00500000")

        result = provider.submit_single_leg(
            ticker="SPY",
            strike=500.0,
            expiration="2026-04-17",
            option_type="put",
            side="sell",
            contracts=2,
        )

        assert result["status"] == "submitted"
        assert result["side"] == "sell"
        assert result["option_type"] == "put"

    @patch("strategy.alpaca_provider.AlpacaProvider.__init__", return_value=None)
    def test_submit_single_leg_symbol_resolve_failure(self, mock_init):
        """Returns error if option symbol cannot be resolved."""
        from strategy.alpaca_provider import AlpacaProvider

        provider = AlpacaProvider.__new__(AlpacaProvider)
        provider.find_option_symbol = MagicMock(return_value=None)

        result = provider.submit_single_leg(
            ticker="SPY", strike=500.0, expiration="2026-04-17",
            option_type="call", side="buy",
        )

        assert result["status"] == "error"
        assert "Could not resolve" in result["message"]

    @patch("strategy.alpaca_provider.AlpacaProvider.__init__", return_value=None)
    def test_close_single_leg_uses_close_intents(self, mock_init):
        """close_single_leg uses BUY_TO_CLOSE / SELL_TO_CLOSE intents."""
        from strategy.alpaca_provider import AlpacaProvider, PositionIntent

        provider = AlpacaProvider.__new__(AlpacaProvider)
        provider._circuit_breaker = MagicMock()
        provider.client = MagicMock()

        mock_order = MagicMock()
        mock_order.id = "close-789"
        mock_order.status = "accepted"
        provider._circuit_breaker.call.return_value = mock_order
        provider.find_option_symbol = MagicMock(return_value="SPY260417C00500000")

        result = provider.close_single_leg(
            ticker="SPY", strike=500.0, expiration="2026-04-17",
            option_type="call", side="buy", contracts=1,
        )

        assert result["status"] == "submitted"
        # Verify the order request used BUY_TO_CLOSE
        order_req = provider._circuit_breaker.call.call_args[0][1]
        assert order_req.position_intent == PositionIntent.BUY_TO_CLOSE

    @patch("strategy.alpaca_provider.AlpacaProvider.__init__", return_value=None)
    def test_submit_single_leg_rounds_strike(self, mock_init):
        """Strike is rounded to 2 decimals before lookup."""
        from strategy.alpaca_provider import AlpacaProvider

        provider = AlpacaProvider.__new__(AlpacaProvider)
        provider._circuit_breaker = MagicMock()
        provider.client = MagicMock()

        mock_order = MagicMock()
        mock_order.id = "ord-round"
        mock_order.client_order_id = "test"
        mock_order.status = "accepted"
        mock_order.submitted_at = "2026-03-15"
        provider._circuit_breaker.call.return_value = mock_order
        provider.find_option_symbol = MagicMock(return_value="SPY260417C00683000")

        result = provider.submit_single_leg(
            ticker="SPY", strike=682.9999999, expiration="2026-04-17",
            option_type="call", side="buy",
        )

        # Should have rounded to 683.0
        provider.find_option_symbol.assert_called_with("SPY", "2026-04-17", 683.0, "call")
        assert result["strike"] == 683.0


# ===========================================================================
# ExecutionEngine._submit_straddle tests
# ===========================================================================

class TestSubmitStraddle(unittest.TestCase):
    """Test the _submit_straddle method in ExecutionEngine."""

    def test_short_straddle_sells_both_legs(self):
        """Short straddle: sell call + sell put."""
        alpaca = _make_alpaca()
        call_counter = {"n": 0}

        def mock_submit(**kwargs):
            call_counter["n"] += 1
            return {"status": "submitted", "order_id": f"ord-{call_counter['n']}"}

        alpaca.submit_single_leg.side_effect = lambda **kw: mock_submit(**kw)
        db = _setup_db()
        engine = ExecutionEngine(alpaca_provider=alpaca, db_path=db)

        opp = _make_straddle_opp()
        result = engine._submit_straddle(opp, contracts=2, credit=8.50, client_id="test-001")

        assert result["status"] == "submitted"
        assert alpaca.submit_single_leg.call_count == 2

        # First call: sell call
        first_call = alpaca.submit_single_leg.call_args_list[0]
        assert first_call.kwargs["side"] == "sell"
        assert first_call.kwargs["option_type"] == "call"

        # Second call: sell put
        second_call = alpaca.submit_single_leg.call_args_list[1]
        assert second_call.kwargs["side"] == "sell"
        assert second_call.kwargs["option_type"] == "put"

    def test_long_straddle_buys_both_legs(self):
        """Long straddle: buy call + buy put (debit)."""
        alpaca = _make_alpaca()
        alpaca.submit_single_leg.return_value = {"status": "submitted", "order_id": "ord-long"}
        db = _setup_db()
        engine = ExecutionEngine(alpaca_provider=alpaca, db_path=db)

        opp = _make_long_straddle_opp()
        result = engine._submit_straddle(opp, contracts=1, credit=-6.00, client_id="test-long")

        assert result["status"] == "submitted"
        # Both calls should be "buy"
        for call_args in alpaca.submit_single_leg.call_args_list:
            assert call_args.kwargs["side"] == "buy"

    def test_limit_price_split_evenly(self):
        """Each leg gets half the total credit/debit as limit price."""
        alpaca = _make_alpaca()
        alpaca.submit_single_leg.return_value = {"status": "submitted", "order_id": "ord-1"}
        db = _setup_db()
        engine = ExecutionEngine(alpaca_provider=alpaca, db_path=db)

        opp = _make_straddle_opp(credit=10.00)
        engine._submit_straddle(opp, contracts=1, credit=10.00, client_id="test-split")

        for call_args in alpaca.submit_single_leg.call_args_list:
            assert call_args.kwargs["limit_price"] == 5.00

    def test_debit_limit_price_uses_abs(self):
        """Debit (negative credit): limit price is positive abs(credit/2)."""
        alpaca = _make_alpaca()
        alpaca.submit_single_leg.return_value = {"status": "submitted", "order_id": "ord-1"}
        db = _setup_db()
        engine = ExecutionEngine(alpaca_provider=alpaca, db_path=db)

        opp = _make_long_straddle_opp(credit=-8.00)
        engine._submit_straddle(opp, contracts=1, credit=-8.00, client_id="test-debit")

        for call_args in alpaca.submit_single_leg.call_args_list:
            assert call_args.kwargs["limit_price"] == 4.00  # abs(-8/2)

    def test_second_leg_failure_cancels_first(self):
        """If put leg fails, call leg gets cancelled (rollback)."""
        alpaca = _make_alpaca()
        alpaca.submit_single_leg.side_effect = [
            {"status": "submitted", "order_id": "ord-call-ok"},
            {"status": "error", "message": "insufficient buying power"},
        ]
        db = _setup_db()
        engine = ExecutionEngine(alpaca_provider=alpaca, db_path=db)

        opp = _make_straddle_opp()
        result = engine._submit_straddle(opp, contracts=1, credit=8.50, client_id="test-rollback")

        assert result["status"] == "partial_error"
        alpaca.cancel_order.assert_called_once_with("ord-call-ok")

    def test_first_leg_failure_no_cancel(self):
        """If call leg fails, no cancel is needed (nothing to roll back)."""
        alpaca = _make_alpaca()
        alpaca.submit_single_leg.return_value = {"status": "error", "message": "bad symbol"}
        db = _setup_db()
        engine = ExecutionEngine(alpaca_provider=alpaca, db_path=db)

        opp = _make_straddle_opp()
        result = engine._submit_straddle(opp, contracts=1, credit=8.50, client_id="test-fail1")

        assert result["status"] == "partial_error"
        alpaca.cancel_order.assert_not_called()


# ===========================================================================
# ExecutionEngine dry-run straddle logging
# ===========================================================================

class TestStraddleDryRun(unittest.TestCase):
    """Test that straddle dry-run logs straddle-specific fields."""

    def test_dry_run_straddle_returns_dry_run_status(self):
        """Dry-run mode (no alpaca) returns dry_run status for straddles."""
        db = _setup_db()
        engine = ExecutionEngine(alpaca_provider=None, db_path=db)
        opp = _make_straddle_opp()
        result = engine.submit_opportunity(opp)

        assert result["status"] == "dry_run"
        assert "client_order_id" in result

    def test_dry_run_long_straddle(self):
        """Dry-run mode for long straddle (debit)."""
        db = _setup_db()
        engine = ExecutionEngine(alpaca_provider=None, db_path=db)
        opp = _make_long_straddle_opp()
        result = engine.submit_opportunity(opp)

        assert result["status"] == "dry_run"

    def test_trade_record_stores_straddle_fields(self):
        """Trade record in DB has call_strike, put_strike, is_debit."""
        db = _setup_db()
        engine = ExecutionEngine(alpaca_provider=None, db_path=db)
        opp = _make_long_straddle_opp(call_strike=505.0, put_strike=495.0)
        result = engine.submit_opportunity(opp)

        trade = get_trade_by_id(result["client_order_id"], path=db)
        assert trade is not None
        assert trade["call_strike"] == 505.0
        assert trade["put_strike"] == 495.0
        assert trade["is_debit"] is True


# ===========================================================================
# PositionMonitor straddle close reconciliation
# ===========================================================================

class TestStraddleCloseReconciliation(unittest.TestCase):
    """Test dual-leg close reconciliation for straddles."""

    def test_dual_leg_both_filled_records_combined_pnl(self):
        """When both call and put close orders fill, combined P&L is recorded."""
        db = _setup_db()
        alpaca = _make_alpaca()

        # Call fills at 2.00, put fills at 1.50 → combined 3.50
        def mock_order_status(order_id):
            if "call" in order_id or order_id == "close-call-001":
                return {"status": "filled", "filled_avg_price": "2.00",
                        "filled_at": "2026-03-15", "filled_qty": "2"}
            else:
                return {"status": "filled", "filled_avg_price": "1.50",
                        "filled_at": "2026-03-15", "filled_qty": "2"}

        alpaca.get_order_status.side_effect = mock_order_status

        mon = _monitor(alpaca=alpaca, db_path=db)

        trade = _make_straddle_trade(
            trade_id="ss-recon-1",
            status="pending_close",
            credit=8.50,  # short straddle, received 8.50 credit
            contracts=2,
            close_order_id="close-call-001",
            close_put_order_id="close-put-001",
            exit_reason="profit_target",
        )
        upsert_trade(trade, source="execution", path=db)

        mon._reconcile_pending_closes()

        updated = get_trade_by_id("ss-recon-1", path=db)
        # P&L = (credit - combined_fill) * contracts * 100
        # = (8.50 - 3.50) * 2 * 100 = 1000.0
        assert updated["status"] in ("closed_profit", "closed_loss", "closed")
        assert updated["pnl"] == 1000.0

    def test_dual_leg_one_failed_resets_to_open(self):
        """If one leg terminally fails, position resets to open."""
        db = _setup_db()
        alpaca = _make_alpaca()

        def mock_order_status(order_id):
            if "call" in order_id:
                return {"status": "filled", "filled_avg_price": "2.00",
                        "filled_at": "2026-03-15", "filled_qty": "1"}
            else:
                return {"status": "cancelled"}

        alpaca.get_order_status.side_effect = mock_order_status

        mon = _monitor(alpaca=alpaca, db_path=db)

        trade = _make_straddle_trade(
            trade_id="ss-recon-2",
            status="pending_close",
            close_order_id="close-call-002",
            close_put_order_id="close-put-002",
            exit_reason="stop_loss",
        )
        upsert_trade(trade, source="execution", path=db)

        mon._reconcile_pending_closes()

        updated = get_trade_by_id("ss-recon-2", path=db)
        assert updated["status"] == "open"
        assert updated.get("close_order_id") is None

    def test_dual_leg_partial_fill_waits(self):
        """If one leg filled but other pending, position stays pending_close."""
        db = _setup_db()
        alpaca = _make_alpaca()

        def mock_order_status(order_id):
            if "call" in order_id:
                return {"status": "filled", "filled_avg_price": "2.00",
                        "filled_at": "2026-03-15", "filled_qty": "1"}
            else:
                return {"status": "accepted"}

        alpaca.get_order_status.side_effect = mock_order_status

        mon = _monitor(alpaca=alpaca, db_path=db)

        trade = _make_straddle_trade(
            trade_id="ss-recon-3",
            status="pending_close",
            close_order_id="close-call-003",
            close_put_order_id="close-put-003",
            exit_reason="profit_target",
        )
        upsert_trade(trade, source="execution", path=db)

        mon._reconcile_pending_closes()

        updated = get_trade_by_id("ss-recon-3", path=db)
        assert updated["status"] == "pending_close"  # still waiting


# ===========================================================================
# Straddle P&L calculation
# ===========================================================================

class TestStraddlePnL(unittest.TestCase):
    """Test P&L calculations for straddle positions."""

    def test_short_straddle_profit(self):
        """Short straddle profit: credit received > cost to close."""
        db = _setup_db()
        alpaca = _make_alpaca()
        alpaca.get_order_status.return_value = {
            "status": "filled", "filled_avg_price": "3.00",
            "filled_at": "2026-03-15", "filled_qty": "2",
        }
        mon = _monitor(alpaca=alpaca, db_path=db)

        trade = _make_straddle_trade(
            trade_id="ss-pnl-short",
            status="pending_close",
            credit=8.50,
            contracts=2,
            close_order_id="ord-pnl-1",
            exit_reason="profit_target",
        )
        upsert_trade(trade, source="execution", path=db)

        mon._reconcile_pending_closes()

        updated = get_trade_by_id("ss-pnl-short", path=db)
        # P&L = (8.50 - 3.00) * 2 * 100 = 1100.0
        assert updated["pnl"] == 1100.0

    def test_long_straddle_profit(self):
        """Long straddle profit: sell proceeds > debit paid."""
        db = _setup_db()
        alpaca = _make_alpaca()
        alpaca.get_order_status.return_value = {
            "status": "filled", "filled_avg_price": "8.00",
            "filled_at": "2026-03-15", "filled_qty": "1",
        }
        mon = _monitor(alpaca=alpaca, db_path=db)

        trade = _make_straddle_trade(
            trade_id="ss-pnl-long",
            strategy_type="long_straddle",
            status="pending_close",
            credit=-6.00,  # debit
            is_debit=True,
            contracts=1,
            close_order_id="ord-pnl-2",
            exit_reason="profit_target",
        )
        upsert_trade(trade, source="execution", path=db)

        mon._reconcile_pending_closes()

        updated = get_trade_by_id("ss-pnl-long", path=db)
        # P&L = (8.00 - abs(-6.00)) * 1 * 100 = 200.0
        assert updated["pnl"] == 200.0

    def test_long_straddle_loss(self):
        """Long straddle loss: sell proceeds < debit paid."""
        db = _setup_db()
        alpaca = _make_alpaca()
        alpaca.get_order_status.return_value = {
            "status": "filled", "filled_avg_price": "2.00",
            "filled_at": "2026-03-15", "filled_qty": "1",
        }
        mon = _monitor(alpaca=alpaca, db_path=db)

        trade = _make_straddle_trade(
            trade_id="ss-pnl-loss",
            strategy_type="long_straddle",
            status="pending_close",
            credit=-6.00,
            is_debit=True,
            contracts=1,
            close_order_id="ord-pnl-3",
            exit_reason="stop_loss",
        )
        upsert_trade(trade, source="execution", path=db)

        mon._reconcile_pending_closes()

        updated = get_trade_by_id("ss-pnl-loss", path=db)
        # P&L = (2.00 - 6.00) * 1 * 100 = -400.0
        assert updated["pnl"] == -400.0


# ===========================================================================
# Combine straddle fills helper
# ===========================================================================

class TestCombineStraddleFills(unittest.TestCase):
    """Test _combine_straddle_fills helper."""

    def test_combines_fill_prices(self):
        db = _setup_db()
        mon = _monitor(db_path=db)

        call_order = {"filled_avg_price": "3.50", "filled_at": "2026-03-15", "filled_qty": "2"}
        put_order = {"filled_avg_price": "2.25", "filled_at": "2026-03-15", "filled_qty": "2"}

        combined = mon._combine_straddle_fills(call_order, put_order)

        assert combined["status"] == "filled"
        assert float(combined["filled_avg_price"]) == 5.75
        assert combined["filled_qty"] == "2"

    def test_handles_missing_fill_prices(self):
        db = _setup_db()
        mon = _monitor(db_path=db)

        call_order = {"filled_avg_price": None, "filled_at": None, "filled_qty": None}
        put_order = {"filled_avg_price": "2.00", "filled_at": "2026-03-15", "filled_qty": "1"}

        combined = mon._combine_straddle_fills(call_order, put_order)

        assert combined["status"] == "filled"
        assert float(combined["filled_avg_price"]) == 2.0


if __name__ == "__main__":
    unittest.main()
