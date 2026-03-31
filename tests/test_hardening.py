"""
Pre-live hardening tests — verification checklist gaps.

Covers:
  H1:  Non-retryable error detection (_is_non_retryable)
  H2:  _retry_with_backoff skips retry on 4xx errors
  H3:  AlpacaProvider.get_market_clock() happy path and failure path
  H4:  AlpacaProvider.get_order_status() returns filled_qty and qty
  H5:  ExecutionEngine market-closed guard blocks order submission
  H6:  ExecutionEngine market-open or unknown passes through
  H7:  PositionMonitor._reconcile_pending_opens() calls PositionReconciler
  H8:  PositionMonitor partial fill detection (filled_qty != contracts)
  H9:  PositionMonitor partial fill adjusts contracts in pos dict
  H10: PositionMonitor._detect_assignment() warns on equity position
  H11: PositionMonitor._detect_assignment() ignores option positions
  H12: PositionMonitor._detect_assignment() ignores unrelated tickers
  H13: _check_positions() calls _reconcile_pending_opens before _reconcile_pending_closes
  H14: pending_open trade promoted to open by intra-cycle reconciliation
  H15: market_closed result does NOT update DB status
"""

from datetime import datetime, timezone
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

try:
    from alpaca.trading.requests import OptionLegRequest  # noqa: F401
except ImportError:
    pytest.skip("OptionLegRequest not available in this alpaca-py version", allow_module_level=True)

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------


def _config(profit_target: float = 50.0, sl_mult: float = 3.5, manage_dte: int = 21) -> Dict:
    return {
        "risk": {"profit_target": profit_target, "stop_loss_multiplier": sl_mult},
        "strategy": {"manage_dte": manage_dte},
    }


def _make_alpaca(
    positions=None,
    order_status=None,
    market_clock=None,
):
    mock = MagicMock()
    mock.get_positions.return_value = positions or []
    mock.get_order_status.return_value = order_status or {}
    mock.close_spread.return_value = {"status": "submitted", "order_id": "ord-001"}
    mock.close_iron_condor.return_value = {"status": "submitted", "order_id": "ord-ic-001"}
    mock.get_market_clock.return_value = market_clock or {"is_open": True}

    def _build_occ(ticker, expiration, strike, opt_type):
        cp = "C" if opt_type.lower().startswith("c") else "P"
        strike_int = int(round(float(strike) * 1000))
        return f"{ticker.upper()}{cp}{strike_int:08d}"

    mock._build_occ_symbol.side_effect = _build_occ
    return mock


def _make_trade(
    trade_id="t1",
    ticker="SPY",
    strategy_type="bull_put",
    status="open",
    short_strike=450.0,
    long_strike=445.0,
    expiration="2099-12-31",
    credit=1.00,
    contracts=1,
    **extra,
) -> Dict:
    t = dict(
        id=trade_id,
        ticker=ticker,
        strategy_type=strategy_type,
        status=status,
        short_strike=short_strike,
        long_strike=long_strike,
        expiration=expiration,
        credit=credit,
        contracts=contracts,
        entry_date=datetime.now(timezone.utc).isoformat(),
    )
    t.update(extra)
    return t


def _monitor(alpaca=None, db_path=None, **cfg_overrides):
    from execution.position_monitor import PositionMonitor
    cfg = _config(**cfg_overrides)
    return PositionMonitor(
        alpaca_provider=alpaca or _make_alpaca(),
        config=cfg,
        db_path=db_path,
    )


def _setup_db(tmp_path):
    from shared.database import init_db
    db = str(tmp_path / "test.db")
    init_db(db)
    return db


# ---------------------------------------------------------------------------
# H1-H2: Non-retryable error classification
# ---------------------------------------------------------------------------


class TestNonRetryableErrors:
    """_is_non_retryable and _retry_with_backoff correctly identify 4xx as non-retryable."""

    def test_422_is_non_retryable(self):
        from strategy.alpaca_provider import _is_non_retryable
        exc = Exception("APIError 422 Unprocessable Entity: invalid symbol")
        assert _is_non_retryable(exc) is True

    def test_403_is_non_retryable(self):
        from strategy.alpaca_provider import _is_non_retryable
        exc = Exception("403 Forbidden: insufficient buying power")
        assert _is_non_retryable(exc) is True

    def test_400_is_non_retryable(self):
        from strategy.alpaca_provider import _is_non_retryable
        exc = Exception("400 Bad Request")
        assert _is_non_retryable(exc) is True

    def test_409_is_non_retryable(self):
        from strategy.alpaca_provider import _is_non_retryable
        exc = Exception("409 Conflict: duplicate client_order_id")
        assert _is_non_retryable(exc) is True

    def test_500_is_retryable(self):
        from strategy.alpaca_provider import _is_non_retryable
        exc = Exception("500 Internal Server Error")
        assert _is_non_retryable(exc) is False

    def test_503_is_retryable(self):
        from strategy.alpaca_provider import _is_non_retryable
        exc = Exception("503 Service Unavailable")
        assert _is_non_retryable(exc) is False

    def test_connection_error_is_retryable(self):
        from strategy.alpaca_provider import _is_non_retryable
        exc = ConnectionError("Network timeout")
        assert _is_non_retryable(exc) is False

    def test_retry_decorator_skips_retry_on_422(self):
        """When a 422 error is raised, the function must NOT be retried."""
        from strategy.alpaca_provider import _retry_with_backoff

        call_count = [0]

        @_retry_with_backoff(max_retries=2, base_delay=0.001)
        def flaky():
            call_count[0] += 1
            raise Exception("422 invalid symbol")

        with pytest.raises(Exception, match="422"):
            flaky()

        # Called exactly once — no retry
        assert call_count[0] == 1

    def test_retry_decorator_retries_on_500(self):
        """Server errors should be retried up to max_retries times."""
        from strategy.alpaca_provider import _retry_with_backoff

        call_count = [0]

        @_retry_with_backoff(max_retries=2, base_delay=0.001)
        def flaky():
            call_count[0] += 1
            raise Exception("500 server error")

        with pytest.raises(Exception, match="500"):
            flaky()

        # Called 3 times: 1 original + 2 retries
        assert call_count[0] == 3


# ---------------------------------------------------------------------------
# H3: AlpacaProvider.get_market_clock
# ---------------------------------------------------------------------------


class TestGetMarketClock:
    def _make_provider(self, is_open=True):
        """Create a partially-mocked AlpacaProvider."""
        from strategy.alpaca_provider import AlpacaProvider

        with patch.object(AlpacaProvider, "_verify_connection"):
            provider = AlpacaProvider.__new__(AlpacaProvider)
            provider._api_key = "K"
            provider._api_secret = "S"
            provider._base_url = "https://paper-api.alpaca.markets"

            from shared.circuit_breaker import CircuitBreaker
            provider._circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)

            mock_clock = MagicMock()
            mock_clock.is_open = is_open
            mock_clock.next_open = "2026-03-09T09:30:00-05:00"
            mock_clock.next_close = "2026-03-09T16:00:00-05:00"
            mock_clock.timestamp = "2026-03-09T08:00:00-05:00"

            mock_client = MagicMock()
            mock_client.get_clock.return_value = mock_clock
            provider.client = mock_client

        return provider

    def test_market_open_returns_is_open_true(self):
        provider = self._make_provider(is_open=True)
        clock = provider.get_market_clock()
        assert clock["is_open"] is True
        assert "next_open" in clock

    def test_market_closed_returns_is_open_false(self):
        provider = self._make_provider(is_open=False)
        clock = provider.get_market_clock()
        assert clock["is_open"] is False

    def test_failure_returns_is_open_none(self):
        from strategy.alpaca_provider import AlpacaProvider

        with patch.object(AlpacaProvider, "_verify_connection"):
            provider = AlpacaProvider.__new__(AlpacaProvider)
            provider._api_key = "K"
            provider._api_secret = "S"
            from shared.circuit_breaker import CircuitBreaker
            provider._circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)
            mock_client = MagicMock()
            mock_client.get_clock.side_effect = Exception("network error")
            provider.client = mock_client

        result = provider.get_market_clock()
        assert result["is_open"] is None


# ---------------------------------------------------------------------------
# H4: get_order_status includes filled_qty and qty
# ---------------------------------------------------------------------------


class TestGetOrderStatusFields:
    def test_filled_qty_and_qty_returned(self):
        from strategy.alpaca_provider import AlpacaProvider

        with patch.object(AlpacaProvider, "_verify_connection"):
            provider = AlpacaProvider.__new__(AlpacaProvider)
            from shared.circuit_breaker import CircuitBreaker
            provider._circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)

            mock_order = MagicMock()
            mock_order.id = "ord-999"
            mock_order.status = "filled"
            mock_order.filled_avg_price = "1.50"
            mock_order.filled_at = "2026-03-09T10:00:00Z"
            mock_order.filled_qty = "2"
            mock_order.qty = "2"

            mock_client = MagicMock()
            mock_client.get_order_by_id.return_value = mock_order
            provider.client = mock_client

        result = provider.get_order_status("ord-999")
        assert result["filled_qty"] == "2"
        assert result["qty"] == "2"
        assert result["status"] == "filled"

    def test_null_filled_qty_returns_none(self):
        from strategy.alpaca_provider import AlpacaProvider

        with patch.object(AlpacaProvider, "_verify_connection"):
            provider = AlpacaProvider.__new__(AlpacaProvider)
            from shared.circuit_breaker import CircuitBreaker
            provider._circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)

            mock_order = MagicMock()
            mock_order.id = "ord-111"
            mock_order.status = "new"
            mock_order.filled_avg_price = None
            mock_order.filled_at = None
            mock_order.filled_qty = None
            mock_order.qty = "3"

            mock_client = MagicMock()
            mock_client.get_order_by_id.return_value = mock_order
            provider.client = mock_client

        result = provider.get_order_status("ord-111")
        assert result["filled_qty"] is None
        assert result["qty"] == "3"


# ---------------------------------------------------------------------------
# H5-H6: ExecutionEngine market hours guard
# ---------------------------------------------------------------------------


class TestExecutionEngineMarketGuard:
    def _make_engine(self, is_open, db_path):
        from execution.execution_engine import ExecutionEngine
        mock_alpaca = _make_alpaca(market_clock={"is_open": is_open})
        mock_alpaca.submit_credit_spread.return_value = {"status": "submitted", "order_id": "x"}
        return ExecutionEngine(alpaca_provider=mock_alpaca, db_path=db_path)

    def test_market_closed_blocks_submission(self, tmp_path):
        engine = self._make_engine(is_open=False, db_path=str(tmp_path / "db.db"))
        opp = {
            "ticker": "SPY", "type": "bull_put", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50, "contracts": 1,
        }
        result = engine.submit_opportunity(opp)
        assert result["status"] == "market_closed"
        assert "next_open" in result["message"]
        # Alpaca submit should NOT have been called
        engine.alpaca.submit_credit_spread.assert_not_called()

    def test_market_open_allows_submission(self, tmp_path):
        engine = self._make_engine(is_open=True, db_path=str(tmp_path / "db.db"))
        opp = {
            "ticker": "SPY", "type": "bull_put", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50, "contracts": 1,
        }
        result = engine.submit_opportunity(opp)
        assert result["status"] == "submitted"
        engine.alpaca.submit_credit_spread.assert_called_once()

    def test_market_clock_unknown_does_not_block(self, tmp_path):
        """When market clock returns is_open=None (API failure), fail open."""
        from execution.execution_engine import ExecutionEngine
        mock_alpaca = _make_alpaca(market_clock={"is_open": None})
        mock_alpaca.submit_credit_spread.return_value = {"status": "submitted", "order_id": "x"}
        engine = ExecutionEngine(alpaca_provider=mock_alpaca, db_path=str(tmp_path / "db.db"))
        opp = {
            "ticker": "SPY", "type": "bull_put", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50, "contracts": 1,
        }
        result = engine.submit_opportunity(opp)
        # Should NOT be blocked (fail open)
        assert result["status"] != "market_closed"

    def test_dry_run_skips_market_clock_check(self, tmp_path):
        """In dry-run mode (no alpaca), market clock check is not called."""
        from execution.execution_engine import ExecutionEngine
        engine = ExecutionEngine(alpaca_provider=None, db_path=str(tmp_path / "db.db"))
        opp = {
            "ticker": "SPY", "type": "bull_put", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50, "contracts": 1,
        }
        result = engine.submit_opportunity(opp)
        assert result["status"] == "dry_run"

    def test_market_closed_still_writes_db_record(self, tmp_path):
        """Even on market_closed, DB record is written (write-before-submit pattern)."""
        from shared.database import get_trades
        db_path = str(tmp_path / "db.db")
        engine = self._make_engine(is_open=False, db_path=db_path)
        opp = {
            "ticker": "SPY", "type": "bull_put", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50, "contracts": 1,
        }
        engine.submit_opportunity(opp)
        trades = get_trades(path=db_path)
        # DB record is written in pending_open state before market-hours check
        assert len(trades) == 1
        assert trades[0]["status"] == "pending_open"


# ---------------------------------------------------------------------------
# H7: PositionMonitor._reconcile_pending_opens calls PositionReconciler
# ---------------------------------------------------------------------------


class TestReconcilePendingOpens:
    def test_reconcile_pending_opens_calls_reconciler(self):
        """_reconcile_pending_opens must delegate to PositionReconciler."""
        mon = _monitor(alpaca=_make_alpaca())

        mock_result = MagicMock()
        mock_result.pending_resolved = 1
        mock_result.pending_failed = 0

        # PositionReconciler is imported *inside* the function with `from shared.reconciler import`
        # so we must patch it at the source module.
        with patch("shared.reconciler.PositionReconciler") as MockReconciler:
            MockReconciler.return_value.reconcile_pending_only.return_value = mock_result
            mon._reconcile_pending_opens()
            MockReconciler.assert_called_once()
            MockReconciler.return_value.reconcile_pending_only.assert_called_once()

    def test_reconcile_pending_opens_handles_exception_gracefully(self):
        """Exception inside _reconcile_pending_opens must not propagate."""
        mon = _monitor(alpaca=_make_alpaca())

        with patch("shared.reconciler.PositionReconciler", side_effect=Exception("DB gone")):
            # Must not raise
            mon._reconcile_pending_opens()

    def test_check_positions_calls_pending_opens_before_pending_closes(self):
        """_check_positions must call _reconcile_pending_opens before _reconcile_pending_closes."""
        mon = _monitor(alpaca=_make_alpaca())
        call_order = []

        with patch.object(mon, "_reconcile_pending_opens", side_effect=lambda: call_order.append("opens")):
            with patch.object(mon, "_reconcile_pending_closes", side_effect=lambda: call_order.append("closes")):
                with patch.object(mon, "_is_market_hours", return_value=True):
                    with patch("execution.position_monitor.get_trades", return_value=[]):
                        mon._check_positions()

        assert "opens" in call_order
        assert "closes" in call_order
        assert call_order.index("opens") < call_order.index("closes")

    def test_pending_open_promoted_to_open_intraday(self, tmp_path):
        """Integration: pending_open trade becomes 'open' within the same session.

        We test only the reconciliation step (pending_open → open), isolating from
        the external-close and exit-condition logic that runs later in _check_positions.
        """
        from shared.database import get_trades, upsert_trade

        db = _setup_db(tmp_path)

        # Write a pending_open trade
        trade = _make_trade(
            trade_id="intraday-1",
            status="pending_open",
            alpaca_client_order_id="cs-abc123",
        )
        upsert_trade(trade, source="execution", path=db)

        # Mock: Alpaca says the order is filled
        alpaca = _make_alpaca()
        alpaca.get_order_by_client_id.return_value = {
            "id": "ord-fill-1",
            "status": "filled",
            "filled_avg_price": "1.50",
            "filled_at": datetime.now(timezone.utc).isoformat(),
            "legs": [],
        }
        alpaca.get_positions.return_value = []

        mon = _monitor(alpaca=alpaca, db_path=db)

        # Call only the reconciliation step directly — this is the core behaviour we're testing
        mon._reconcile_pending_opens()

        trades = get_trades(path=db)
        assert trades[0]["status"] == "open"


# ---------------------------------------------------------------------------
# H8-H9: Partial fill detection in _reconcile_pending_closes
# ---------------------------------------------------------------------------


class TestPartialFillDetection:
    def _setup_db(self, tmp_path):
        return _setup_db(tmp_path)

    def test_partial_fill_logged_and_contracts_adjusted(self, tmp_path):
        """When filled_qty < expected contracts, log warning and adjust contracts."""

        db = self._setup_db(tmp_path)
        from shared.database import upsert_trade

        # Position with 3 contracts expected
        pos = _make_trade(
            trade_id="partial-1",
            status="pending_close",
            contracts=3,
            credit=1.00,
            close_order_id="ord-partial",
            exit_reason="profit_target",
        )
        upsert_trade(pos, source="execution", path=db)

        # Alpaca says only 2 filled
        alpaca = _make_alpaca(order_status={
            "id": "ord-partial",
            "status": "filled",
            "filled_avg_price": "0.40",
            "filled_at": datetime.now(timezone.utc).isoformat(),
            "filled_qty": "2",   # partial!
            "qty": "3",
        })
        mon = _monitor(alpaca=alpaca, db_path=db)

        with patch.object(mon, "_record_close_pnl") as mock_record:
            mon._reconcile_pending_closes()
            # _record_close_pnl was called with the pos dict
            assert mock_record.called
            # The pos dict should have contracts adjusted to 2
            pos_arg = mock_record.call_args[0][0]
            assert pos_arg["contracts"] == 2

    def test_full_fill_no_adjustment(self, tmp_path):
        """When filled_qty == contracts, pos dict is unchanged."""
        db = self._setup_db(tmp_path)
        from shared.database import upsert_trade

        pos = _make_trade(
            trade_id="full-1",
            status="pending_close",
            contracts=2,
            credit=1.00,
            close_order_id="ord-full",
            exit_reason="profit_target",
        )
        upsert_trade(pos, source="execution", path=db)

        alpaca = _make_alpaca(order_status={
            "status": "filled",
            "filled_avg_price": "0.40",
            "filled_at": datetime.now(timezone.utc).isoformat(),
            "filled_qty": "2",
            "qty": "2",
        })
        mon = _monitor(alpaca=alpaca, db_path=db)

        with patch.object(mon, "_record_close_pnl") as mock_record:
            mon._reconcile_pending_closes()
            pos_arg = mock_record.call_args[0][0]
            # Contracts unchanged
            assert pos_arg["contracts"] == 2

    def test_missing_filled_qty_does_not_crash(self, tmp_path):
        """When filled_qty is absent from order response, no crash."""
        db = self._setup_db(tmp_path)
        from shared.database import upsert_trade

        pos = _make_trade(
            trade_id="no-qty-1",
            status="pending_close",
            contracts=1,
            credit=1.00,
            close_order_id="ord-nq",
            exit_reason="stop_loss",
        )
        upsert_trade(pos, source="execution", path=db)

        # No filled_qty in response
        alpaca = _make_alpaca(order_status={
            "status": "filled",
            "filled_avg_price": "3.00",
            "filled_at": datetime.now(timezone.utc).isoformat(),
        })
        mon = _monitor(alpaca=alpaca, db_path=db)

        # Must not raise
        mon._reconcile_pending_closes()
        from shared.database import get_trades
        trades = get_trades(path=db)
        assert trades[0]["status"] in ("closed_profit", "closed_loss")


# ---------------------------------------------------------------------------
# H10-H12: Assignment detection
# ---------------------------------------------------------------------------


class TestAssignmentDetection:
    def _make_open_pos(self, ticker="SPY"):
        return _make_trade(ticker=ticker, strategy_type="bull_put")

    def test_equity_position_matching_ticker_triggers_warning(self, caplog):
        """Equity position for managed ticker logs a WARNING."""
        import logging

        mon = _monitor()
        open_positions = [self._make_open_pos("SPY")]
        # Alpaca positions dict: SPY equity position (no 'option' in asset_class)
        alpaca_positions = {
            "SPY": {
                "symbol": "SPY",
                "qty": "-100",
                "asset_class": "us_equity",
                "market_value": "-55000",
            }
        }

        with caplog.at_level(logging.WARNING, logger="execution.position_monitor"):
            mon._detect_assignment(open_positions, alpaca_positions)

        assert any("POSSIBLE ASSIGNMENT" in r.message for r in caplog.records)
        assert any("SPY" in r.message for r in caplog.records)

    def test_option_positions_not_flagged(self, caplog):
        """Regular option positions should not trigger assignment warning."""
        import logging

        mon = _monitor()
        open_positions = [self._make_open_pos("SPY")]
        alpaca_positions = {
            "SPY260320P00540000": {
                "symbol": "SPY260320P00540000",
                "qty": "-1",
                "asset_class": "us_option",
                "market_value": "-150",
            }
        }

        with caplog.at_level(logging.WARNING, logger="execution.position_monitor"):
            mon._detect_assignment(open_positions, alpaca_positions)

        assert not any("POSSIBLE ASSIGNMENT" in r.message for r in caplog.records)

    def test_equity_position_unrelated_ticker_not_flagged(self, caplog):
        """Equity position for a different ticker does not trigger warning."""
        import logging

        mon = _monitor()
        open_positions = [self._make_open_pos("SPY")]
        # QQQ equity position — not related to SPY spread
        alpaca_positions = {
            "QQQ": {
                "symbol": "QQQ",
                "qty": "50",
                "asset_class": "us_equity",
                "market_value": "22000",
            }
        }

        with caplog.at_level(logging.WARNING, logger="execution.position_monitor"):
            mon._detect_assignment(open_positions, alpaca_positions)

        assert not any("POSSIBLE ASSIGNMENT" in r.message for r in caplog.records)

    def test_no_open_positions_skips_detection(self):
        """With no managed positions, _detect_assignment is a no-op."""
        mon = _monitor()
        # Should not raise
        mon._detect_assignment([], {"SPY": {"asset_class": "us_equity", "qty": "-100"}})

    def test_detect_assignment_called_in_check_positions(self):
        """_check_positions must call _detect_assignment every cycle (when open positions exist)."""
        alpaca = _make_alpaca(positions=[])
        mon = _monitor(alpaca=alpaca)

        # _detect_assignment is called AFTER the open-positions early-return guard.
        # Provide a non-empty open-positions list so the guard passes.
        fake_open_pos = [_make_trade(trade_id="p1", status="open")]

        with patch.object(mon, "_detect_assignment") as mock_detect, \
             patch.object(mon, "_is_market_hours", return_value=True), \
             patch("execution.position_monitor.get_trades", return_value=fake_open_pos), \
             patch.object(mon, "_reconcile_pending_opens"), \
             patch.object(mon, "_reconcile_pending_closes"), \
             patch.object(mon, "_reconcile_external_closes"), \
             patch.object(mon, "_check_exit_conditions", return_value=None):
            mon._check_positions()

        mock_detect.assert_called_once()

    def test_detect_assignment_multiple_managed_tickers(self, caplog):
        """Assignment detection works correctly when managing multiple tickers."""
        import logging

        mon = _monitor()
        open_positions = [
            self._make_open_pos("SPY"),
            self._make_open_pos("QQQ"),
        ]
        # Both tickers have equity positions
        alpaca_positions = {
            "SPY": {"symbol": "SPY", "qty": "-100", "asset_class": "us_equity", "market_value": "-55000"},
            "QQQ": {"symbol": "QQQ", "qty": "-100", "asset_class": "us_equity", "market_value": "-45000"},
        }

        with caplog.at_level(logging.WARNING, logger="execution.position_monitor"):
            mon._detect_assignment(open_positions, alpaca_positions)

        warning_messages = [r.message for r in caplog.records if "POSSIBLE ASSIGNMENT" in r.message]
        assert len(warning_messages) == 2
        assert any("SPY" in m for m in warning_messages)
        assert any("QQQ" in m for m in warning_messages)


# ---------------------------------------------------------------------------
# H13: Check cycle ordering (covered in H7 above)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# H14: Integration — full intra-day order lifecycle
# ---------------------------------------------------------------------------


class TestIntraDayOrderLifecycle:
    """Integration test: order placed → pending_open → promoted to open → stop-loss checked."""

    def test_pending_open_order_monitored_for_stop_loss_same_session(self, tmp_path):
        """
        A trade placed intra-day (pending_open) should become 'open' and be
        checked for stop-loss within the same running session.

        We verify the two-step sequence:
          Step 1: _reconcile_pending_opens promotes pending_open → open
          Step 2: On next _check_positions, the now-open position is checked for exits
        """
        from execution.position_monitor import PositionMonitor
        from shared.database import get_trades, upsert_trade

        db = _setup_db(tmp_path)

        # Place a pending_open trade
        trade = _make_trade(
            trade_id="live-1",
            status="pending_open",
            credit=1.00,
            contracts=2,
            short_strike=450.0,
            long_strike=445.0,
            expiration="2099-12-31",
            alpaca_client_order_id="cs-live-1",
        )
        upsert_trade(trade, source="execution", path=db)

        alpaca = _make_alpaca()
        alpaca.get_order_by_client_id.return_value = {
            "id": "ord-live-1",
            "status": "filled",
            "filled_avg_price": "1.00",
            "filled_at": datetime.now(timezone.utc).isoformat(),
            "legs": [],
        }

        cfg = _config(sl_mult=3.5)
        monitor = PositionMonitor(alpaca_provider=alpaca, config=cfg, db_path=db)

        # Step 1: Call reconcile directly to promote the trade
        monitor._reconcile_pending_opens()

        trades = get_trades(path=db)
        assert trades[0]["status"] == "open", "Trade must be promoted to open"

        # Step 2: Now it's open — verify _check_exit_conditions is called for it.
        # SL threshold = (1 + 3.5) * 1.00 = $4.50; spread_width=5 → cap=4.50 → threshold=4.50
        # We mock _get_spread_value to return 4.60 (> 4.50) to trigger SL.
        with patch.object(PositionMonitor, "_is_market_hours", return_value=True), \
             patch.object(monitor, "_get_spread_value", return_value=4.60), \
             patch.object(monitor, "_reconcile_external_closes"), \
             patch.object(monitor, "_detect_assignment"), \
             patch.object(monitor.alpaca, "get_positions", return_value=[]):
            monitor._check_positions()

        trades = get_trades(path=db)
        # Stop-loss detected (4.60 >= 4.50 threshold) → close submitted → pending_close
        assert trades[0]["status"] == "pending_close"
        assert trades[0]["exit_reason"] == "stop_loss"
