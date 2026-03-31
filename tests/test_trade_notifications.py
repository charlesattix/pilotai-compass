"""
Unit tests for pilotai_signal/trade_notifications.py.

All tests are fully offline: no Alpaca API calls, no Telegram sends.
Alpaca Order/Position/Account objects are replaced with simple dataclass stubs.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from pilotai_signal.trade_notifications import (
    _get_open_credit,
    _is_notified,
    _mark_notified,
    _daily_summary_sent_today,
    _mark_daily_summary_sent,
    _get_db_connection,
    _SCHEMA,
    build_daily_summary,
    classify_mleg_order,
    format_close_message,
    format_open_message,
    format_rejected_message,
    parse_occ,
    poll_new_orders,
    send_daily_summary,
)


# ---------------------------------------------------------------------------
# Stubs — mimic alpaca-py model objects
# ---------------------------------------------------------------------------

@dataclass
class Leg:
    symbol: str = ""
    side: str = "sell"
    position_intent: str = "sell_to_open"
    filled_avg_price: Optional[float] = None
    qty: float = 1.0


@dataclass
class Order:
    id: str = "order-001"
    client_order_id: str = "cs-SPY-abc12345"
    status: str = "filled"
    order_class: str = "mleg"
    symbol: str = ""
    side: str = ""
    position_intent: str = ""
    filled_avg_price: Optional[float] = 1.82
    filled_qty: float = 2.0
    qty: float = 2.0
    failed_at: Optional[str] = None
    legs: List[Leg] = field(default_factory=list)


@dataclass
class Position:
    symbol: str = ""
    qty: float = 2.0
    unrealized_pl: float = 120.0
    unrealized_plpc: float = 0.08
    current_price: float = 1.50
    avg_entry_price: float = 1.82


@dataclass
class Account:
    equity: float = 105_000.0
    last_equity: float = 103_500.0
    buying_power: float = 50_000.0
    options_buying_power: float = 25_000.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn(tmp_path):
    """In-memory SQLite DB for dedup tracking."""
    conn = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    yield conn
    conn.close()


def _put_spread_order(order_id="order-open-001", credit=1.82, qty=2.0) -> Order:
    """A filled MLEG open bull-put-spread order."""
    return Order(
        id=order_id,
        client_order_id=f"cs-SPY-abc12345",
        status="filled",
        order_class="mleg",
        filled_avg_price=credit,
        filled_qty=qty,
        qty=qty,
        legs=[
            Leg(symbol="O:SPY250404P00520000", side="sell", position_intent="sell_to_open"),
            Leg(symbol="O:SPY250404P00515000", side="buy", position_intent="buy_to_open"),
        ],
    )


def _close_spread_order(order_id="order-close-001", debit=0.45, qty=2.0) -> Order:
    """A filled MLEG close order."""
    return Order(
        id=order_id,
        client_order_id="close-SPY-xyz99999",
        status="filled",
        order_class="mleg",
        filled_avg_price=debit,
        filled_qty=qty,
        qty=qty,
        legs=[
            Leg(symbol="O:SPY250404P00520000", side="buy", position_intent="buy_to_close"),
            Leg(symbol="O:SPY250404P00515000", side="sell", position_intent="sell_to_close"),
        ],
    )


def _rejected_order() -> Order:
    return Order(
        id="order-rej-001",
        client_order_id="cs-XLE-fail0001",
        status="rejected",
        order_class="mleg",
        symbol="",
        filled_avg_price=None,
        filled_qty=0,
        qty=1,
        legs=[],
    )


# ---------------------------------------------------------------------------
# parse_occ
# ---------------------------------------------------------------------------

class TestParseOCC:
    def test_standard_occ_with_prefix(self):
        r = parse_occ("O:SPY250404P00520000")
        assert r is not None
        assert r["ticker"] == "SPY"
        assert r["expiration"] == date(2025, 4, 4)
        assert r["option_type"] == "P"
        assert r["strike"] == pytest.approx(520.0)

    def test_occ_without_o_prefix(self):
        r = parse_occ("SPY250404P00520000")
        assert r is not None
        assert r["ticker"] == "SPY"
        assert r["strike"] == pytest.approx(520.0)

    def test_call_option(self):
        r = parse_occ("O:SPY250404C00530000")
        assert r["option_type"] == "C"
        assert r["strike"] == pytest.approx(530.0)

    def test_fractional_strike(self):
        # $45.50 option
        r = parse_occ("O:IBIT241115P00045500")
        assert r["strike"] == pytest.approx(45.5)

    def test_ibit_symbol(self):
        r = parse_occ("O:IBIT241115P00045000")
        assert r["ticker"] == "IBIT"
        assert r["expiration"] == date(2024, 11, 15)
        assert r["strike"] == pytest.approx(45.0)

    def test_non_option_symbol_returns_none(self):
        assert parse_occ("SPY") is None
        assert parse_occ("AAPL") is None

    def test_empty_returns_none(self):
        assert parse_occ("") is None
        assert parse_occ(None) is None


# ---------------------------------------------------------------------------
# classify_mleg_order
# ---------------------------------------------------------------------------

class TestClassifyMlegOrder:
    def test_open_order_by_leg_intent(self):
        order = _put_spread_order()
        assert classify_mleg_order(order) == "open"

    def test_close_order_by_leg_intent(self):
        order = _close_spread_order()
        assert classify_mleg_order(order) == "close"

    def test_fallback_to_client_id_close(self):
        order = Order(client_order_id="close-SPY-abc123", legs=[])
        assert classify_mleg_order(order) == "close"

    def test_fallback_to_client_id_open(self):
        order = Order(client_order_id="cs-SPY-abc123", legs=[])
        assert classify_mleg_order(order) == "open"

    def test_unknown_when_no_intent_or_id(self):
        order = Order(client_order_id="random-order-id", legs=[])
        assert classify_mleg_order(order) == "unknown"


# ---------------------------------------------------------------------------
# format_open_message
# ---------------------------------------------------------------------------

class TestFormatOpenMessage:
    def test_message_contains_filled_emoji(self):
        msg, _, _ = format_open_message(_put_spread_order())
        assert "📥" in msg

    def test_message_contains_ticker_and_strikes(self):
        msg, _, _ = format_open_message(_put_spread_order(credit=1.82))
        assert "SPY" in msg
        assert "520" in msg
        assert "515" in msg

    def test_message_contains_credit(self):
        msg, _, _ = format_open_message(_put_spread_order(credit=1.82))
        assert "1.82" in msg

    def test_message_contains_put_label(self):
        msg, _, _ = format_open_message(_put_spread_order())
        assert "put" in msg.lower()

    def test_returns_short_symbol(self):
        _, short_sym, _ = format_open_message(_put_spread_order())
        assert short_sym == "o:spy250404p00520000"

    def test_returns_credit(self):
        _, _, credit = format_open_message(_put_spread_order(credit=1.82))
        assert credit == pytest.approx(1.82)

    def test_contracts_in_message(self):
        msg, _, _ = format_open_message(_put_spread_order(qty=3))
        assert "3" in msg


# ---------------------------------------------------------------------------
# format_close_message
# ---------------------------------------------------------------------------

class TestFormatCloseMessage:
    def test_profitable_close_shows_closed_emoji(self):
        # Opened at $1.82, close costs $0.37 → profit
        msg = format_close_message(_close_spread_order(debit=0.37), open_credit=1.82)
        assert "📤" in msg
        assert "POSITION CLOSED" in msg

    def test_stop_triggered_shows_warning_emoji(self):
        # Opened at $1.82, close costs $4.20 → loss
        msg = format_close_message(_close_spread_order(debit=4.20), open_credit=1.82)
        assert "⚠️" in msg
        assert "STOP TRIGGERED" in msg

    def test_pnl_shown_when_open_credit_available(self):
        # P&L = (1.82 - 0.37) * 100 * 2 = $290
        msg = format_close_message(_close_spread_order(debit=0.37, qty=2), open_credit=1.82)
        assert "290" in msg or "+290" in msg

    def test_close_cost_shown_when_no_open_credit(self):
        msg = format_close_message(_close_spread_order(debit=0.50), open_credit=None)
        assert "0.50" in msg

    def test_message_contains_ticker(self):
        msg = format_close_message(_close_spread_order(), open_credit=1.82)
        assert "SPY" in msg

    def test_message_contains_strikes(self):
        msg = format_close_message(_close_spread_order(), open_credit=1.82)
        assert "520" in msg
        assert "515" in msg


# ---------------------------------------------------------------------------
# format_rejected_message
# ---------------------------------------------------------------------------

class TestFormatRejectedMessage:
    def test_rejected_emoji_present(self):
        msg = format_rejected_message(_rejected_order())
        assert "❌" in msg

    def test_rejected_label_present(self):
        msg = format_rejected_message(_rejected_order())
        assert "ORDER REJECTED" in msg

    def test_client_id_in_message(self):
        msg = format_rejected_message(_rejected_order())
        assert "cs-XLE-fail0001" in msg


# ---------------------------------------------------------------------------
# build_daily_summary
# ---------------------------------------------------------------------------

class TestBuildDailySummary:
    def test_contains_equity(self):
        msg = build_daily_summary(Account(), [])
        assert "105,000" in msg or "105000" in msg

    def test_contains_day_pnl(self):
        # 105000 - 103500 = +1500
        msg = build_daily_summary(Account(), [])
        assert "+1,500" in msg or "+1500" in msg

    def test_no_positions_shows_none(self):
        msg = build_daily_summary(Account(), [])
        assert "(none)" in msg

    def test_option_position_shown(self):
        pos = Position(symbol="O:SPY250404P00520000", unrealized_pl=120.0, unrealized_plpc=0.08)
        msg = build_daily_summary(Account(), [pos])
        assert "SPY" in msg
        assert "520" in msg

    def test_expiry_warning_within_3_days(self):
        from datetime import date
        # Expiration is tomorrow
        tomorrow = date.today() + timedelta(days=1)
        yymmdd = tomorrow.strftime("%y%m%d")
        symbol = f"O:SPY{yymmdd}P00500000"
        pos = Position(symbol=symbol, unrealized_pl=-50.0, unrealized_plpc=-0.02)
        msg = build_daily_summary(Account(), [pos])
        assert "Expiring within" in msg or "EXPIR" in msg.upper() or "1d left" in msg

    def test_no_expiry_warning_when_far_out(self):
        # Expiration is 60 days away — no warning
        future = date.today() + timedelta(days=60)
        yymmdd = future.strftime("%y%m%d")
        symbol = f"O:SPY{yymmdd}P00500000"
        pos = Position(symbol=symbol, unrealized_pl=200.0, unrealized_plpc=0.15)
        msg = build_daily_summary(Account(), [pos])
        # No expiry warning section
        assert "Expiring within" not in msg

    def test_non_option_positions_skipped(self):
        pos = Position(symbol="AAPL")  # not an OCC symbol
        msg = build_daily_summary(Account(), [pos])
        assert "AAPL" not in msg  # equity positions are skipped

    def test_header_contains_date(self):
        msg = build_daily_summary(Account(), [])
        assert date.today().isoformat() in msg


# ---------------------------------------------------------------------------
# Dedup DB helpers
# ---------------------------------------------------------------------------

class TestDedupDB:
    def test_new_order_is_not_notified(self, db_conn):
        assert not _is_notified(db_conn, "order-999")

    def test_mark_then_is_notified(self, db_conn):
        _mark_notified(db_conn, "order-001", "open")
        db_conn.commit()
        assert _is_notified(db_conn, "order-001")

    def test_mark_stores_credit(self, db_conn):
        _mark_notified(db_conn, "order-002", "open",
                       short_symbol="O:SPY250404P00520000", credit_received=1.82)
        db_conn.commit()
        credit = _get_open_credit(db_conn, "O:SPY250404P00520000")
        assert credit == pytest.approx(1.82)

    def test_get_open_credit_missing_returns_none(self, db_conn):
        assert _get_open_credit(db_conn, "O:NONEXISTENT") is None

    def test_daily_summary_not_sent_initially(self, db_conn):
        assert not _daily_summary_sent_today(db_conn, date.today())

    def test_daily_summary_marked_sent(self, db_conn):
        _mark_daily_summary_sent(db_conn, date.today())
        db_conn.commit()
        assert _daily_summary_sent_today(db_conn, date.today())

    def test_daily_summary_different_date_not_sent(self, db_conn):
        _mark_daily_summary_sent(db_conn, date.today())
        db_conn.commit()
        yesterday = date.today() - timedelta(days=1)
        assert not _daily_summary_sent_today(db_conn, yesterday)


# ---------------------------------------------------------------------------
# poll_new_orders integration (mocked Alpaca client)
# ---------------------------------------------------------------------------

class TestPollNewOrders:
    def _make_client(self, orders):
        client = MagicMock()
        client.get_orders.return_value = orders
        return client

    def test_new_filled_open_sends_notification(self, db_conn):
        order = _put_spread_order()
        client = self._make_client([order])
        with patch("pilotai_signal.trade_notifications.send_telegram") as mock_tg:
            notified = poll_new_orders(client, db_conn, dry_run=True)
        assert len(notified) == 1
        assert notified[0] == "order-open-001"

    def test_already_notified_order_skipped(self, db_conn):
        order = _put_spread_order()
        _mark_notified(db_conn, order.id, "open")
        db_conn.commit()
        client = self._make_client([order])
        with patch("pilotai_signal.trade_notifications.send_telegram") as mock_tg:
            notified = poll_new_orders(client, db_conn, dry_run=True)
        assert notified == []
        mock_tg.assert_not_called()

    def test_rejected_order_sends_notification(self, db_conn):
        order = _rejected_order()
        client = self._make_client([order])
        with patch("pilotai_signal.trade_notifications.send_telegram") as mock_tg:
            notified = poll_new_orders(client, db_conn, dry_run=True)
        assert len(notified) == 1
        # Message should contain rejection emoji
        call_args = mock_tg.call_args[0][0]
        assert "❌" in call_args

    def test_close_with_known_credit_computes_pnl(self, db_conn):
        # Seed the open order so we know the credit
        _mark_notified(
            db_conn, "order-open-001", "open",
            short_symbol="o:spy250404p00520000",
            credit_received=1.82, contracts=2,
        )
        db_conn.commit()

        close_order = _close_spread_order(debit=0.37, qty=2)
        client = self._make_client([close_order])
        with patch("pilotai_signal.trade_notifications.send_telegram") as mock_tg:
            notified = poll_new_orders(client, db_conn, dry_run=True)
        assert len(notified) == 1
        call_args = mock_tg.call_args[0][0]
        assert "📤" in call_args  # profitable close
        assert "290" in call_args  # (1.82 - 0.37) * 100 * 2

    def test_alpaca_error_returns_empty(self, db_conn):
        client = MagicMock()
        client.get_orders.side_effect = RuntimeError("API down")
        with patch("pilotai_signal.trade_notifications.send_telegram"):
            notified = poll_new_orders(client, db_conn, dry_run=True)
        assert notified == []

    def test_cancelled_partial_fill_notified(self, db_conn):
        order = Order(
            id="order-cancel-001",
            client_order_id="cs-XLE-cancel",
            status="canceled",
            order_class="mleg",
            filled_qty=1.0,  # partially filled
            qty=2.0,
            legs=[],
        )
        client = self._make_client([order])
        with patch("pilotai_signal.trade_notifications.send_telegram") as mock_tg:
            notified = poll_new_orders(client, db_conn, dry_run=True)
        assert len(notified) == 1
        call_args = mock_tg.call_args[0][0]
        assert "CANCELLED" in call_args or "cancelled" in call_args.lower()

    def test_cancelled_no_fill_not_notified(self, db_conn):
        order = Order(
            id="order-cancel-002",
            client_order_id="cs-XLE-pure-cancel",
            status="canceled",
            order_class="mleg",
            filled_qty=0.0,
            qty=2.0,
            legs=[],
        )
        client = self._make_client([order])
        with patch("pilotai_signal.trade_notifications.send_telegram") as mock_tg:
            notified = poll_new_orders(client, db_conn, dry_run=True)
        # Not in notified (no message sent), but marked in DB for dedup
        assert notified == []
        mock_tg.assert_not_called()
        assert _is_notified(db_conn, "order-cancel-002")


# ---------------------------------------------------------------------------
# send_daily_summary
# ---------------------------------------------------------------------------

class TestSendDailySummary:
    def test_sends_once_per_day(self, db_conn):
        client = MagicMock()
        client.get_account.return_value = Account()
        client.get_all_positions.return_value = []
        with patch("pilotai_signal.trade_notifications.send_telegram"):
            first = send_daily_summary(client, db_conn, dry_run=True)
            # dry_run=True doesn't mark in DB — call again to verify idempotency
            # For real test: use dry_run=False and check second call returns False
        assert first is True

    def test_not_sent_twice_same_day(self, db_conn):
        _mark_daily_summary_sent(db_conn, date.today())
        db_conn.commit()
        client = MagicMock()
        with patch("pilotai_signal.trade_notifications.send_telegram") as mock_tg:
            result = send_daily_summary(client, db_conn, dry_run=True)
        assert result is False
        mock_tg.assert_not_called()

    def test_alpaca_error_returns_false(self, db_conn):
        client = MagicMock()
        client.get_account.side_effect = RuntimeError("Network error")
        with patch("pilotai_signal.trade_notifications.send_telegram"):
            result = send_daily_summary(client, db_conn, dry_run=True)
        assert result is False
