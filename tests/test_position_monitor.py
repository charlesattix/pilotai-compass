"""Tests for execution/position_monitor.py covering the 3 critical bug fixes:

  Bug 1: Iron condor (4-leg) pricing and closing
  Bug 2: P&L recording after close order fills
  Bug 3: External close detection (stale DB positions)
  +    : Market hours gate
"""

import pytest

try:
    from alpaca.trading.requests import OptionLegRequest  # noqa: F401
except ImportError:
    pytest.skip("OptionLegRequest not available in this alpaca-py version", allow_module_level=True)

import tempfile
from datetime import datetime, timezone
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

from execution.position_monitor import PositionMonitor
from shared.database import get_trades, init_db, upsert_trade

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(profit_target: float = 50.0, sl_mult: float = 3.5, manage_dte: int = 21) -> Dict:
    return {
        "risk": {
            "profit_target": profit_target,
            "stop_loss_multiplier": sl_mult,
        },
        "strategy": {
            "manage_dte": manage_dte,
        },
    }


def _make_alpaca(positions: List[Dict] = None, order_status: Dict = None):
    """Return a mock AlpacaProvider."""
    mock = MagicMock()
    mock.get_positions.return_value = positions or []
    mock.get_order_status.return_value = order_status or {}
    mock.close_spread.return_value = {"status": "submitted", "order_id": "ord-spread-001"}
    mock.close_iron_condor.return_value = {"status": "submitted", "order_id": "ord-ic-001"}
    # Mirror _build_occ_symbol from real provider
    def _build_occ(ticker, expiration, strike, opt_type):
        cp = "C" if opt_type.lower().startswith("c") else "P"
        strike_int = int(round(float(strike) * 1000))
        # simplify: just ticker+cp+strike for test readability
        return f"{ticker.upper()}{cp}{strike_int:08d}"
    mock._build_occ_symbol.side_effect = _build_occ
    return mock


def _make_trade(
    trade_id: str = "t1",
    ticker: str = "SPY",
    strategy_type: str = "bull_put",
    status: str = "open",
    short_strike: float = 450.0,
    long_strike: float = 445.0,
    expiration: str = "2099-12-31",   # far future so DTE never triggers
    credit: float = 1.00,
    contracts: int = 1,
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


def _make_ic_trade(
    trade_id: str = "ic1",
    ticker: str = "SPY",
    status: str = "open",
    put_short: float = 440.0,
    put_long: float = 435.0,
    call_short: float = 470.0,
    call_long: float = 475.0,
    expiration: str = "2099-12-31",
    credit: float = 2.00,
    contracts: int = 1,
    **extra,
) -> Dict:
    t = _make_trade(
        trade_id=trade_id,
        ticker=ticker,
        strategy_type="iron_condor",
        status=status,
        short_strike=put_short,
        long_strike=put_long,
        expiration=expiration,
        credit=credit,
        contracts=contracts,
    )
    t.update({
        "put_short_strike": put_short,
        "put_long_strike": put_long,
        "call_short_strike": call_short,
        "call_long_strike": call_long,
    })
    t.update(extra)
    return t


def _alpaca_pos(symbol: str, market_value: float) -> Dict:
    """Simulate an Alpaca position dict."""
    return {"symbol": symbol, "market_value": str(market_value)}


def _monitor(alpaca=None, db_path: Optional[str] = None, **config_overrides) -> PositionMonitor:
    cfg = _config(**config_overrides)
    return PositionMonitor(
        alpaca_provider=alpaca or _make_alpaca(),
        config=cfg,
        db_path=db_path,
    )


# ---------------------------------------------------------------------------
# Bug 1: Iron condor pricing (_get_ic_value)
# ---------------------------------------------------------------------------

class TestIronCondorPricing:
    """_get_ic_value sums both wings; returns None on missing strikes."""

    def _monitor_and_occ(self, positions: List[Dict]):
        alpaca = _make_alpaca(positions=positions)
        mon = _monitor(alpaca=alpaca)
        return mon, alpaca

    def _occ(self, alpaca, ticker, exp, strike, opt_type):
        return alpaca._build_occ_symbol(ticker, exp, strike, opt_type)

    def test_ic_value_sums_both_wings(self):
        exp = "2099-12-31"
        pos = _make_ic_trade(
            put_short=440.0, put_long=435.0,
            call_short=470.0, call_long=475.0,
            expiration=exp, credit=2.00, contracts=1,
        )
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        # put short: owe $80 (short, negative MV); put long: worth $20 (long, positive)
        # call short: owe $30; call long: worth $10
        put_short_sym = mon.alpaca._build_occ_symbol("SPY", exp, 440.0, "put")
        put_long_sym = mon.alpaca._build_occ_symbol("SPY", exp, 435.0, "put")
        call_short_sym = mon.alpaca._build_occ_symbol("SPY", exp, 470.0, "call")
        call_long_sym = mon.alpaca._build_occ_symbol("SPY", exp, 475.0, "call")

        alpaca_positions = {
            put_short_sym:  _alpaca_pos(put_short_sym, -80.0),   # short: negative MV
            put_long_sym:   _alpaca_pos(put_long_sym, 20.0),     # long: positive MV
            call_short_sym: _alpaca_pos(call_short_sym, -30.0),
            call_long_sym:  _alpaca_pos(call_long_sym, 10.0),
        }

        val = mon._get_ic_value(pos, alpaca_positions)
        # put wing cost: (80 - 20) / (1*100) = 0.60
        # call wing cost: (30 - 10) / (1*100) = 0.20
        # total = 0.80
        assert val is not None
        assert abs(val - 0.80) < 1e-9

    def test_ic_value_none_when_call_wing_missing(self):
        exp = "2099-12-31"
        pos = _make_ic_trade(expiration=exp)
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        put_short_sym = mon.alpaca._build_occ_symbol("SPY", exp, 440.0, "put")
        put_long_sym = mon.alpaca._build_occ_symbol("SPY", exp, 435.0, "put")

        # Only put wing present — call wing missing
        alpaca_positions = {
            put_short_sym: _alpaca_pos(put_short_sym, -80.0),
            put_long_sym:  _alpaca_pos(put_long_sym, 20.0),
        }

        val = mon._get_ic_value(pos, alpaca_positions)
        assert val is None

    def test_ic_value_none_when_wing_strikes_missing(self):
        """If IC record missing call_short_strike/call_long_strike, return None."""
        pos = _make_ic_trade()
        pos.pop("call_short_strike")
        pos.pop("call_long_strike")
        mon = _monitor()
        val = mon._get_ic_value(pos, {})
        assert val is None

    def test_ic_exit_triggered_at_profit_target(self):
        """IC should trigger profit_target exit when combined value is low enough."""
        exp = "2099-12-31"
        pos = _make_ic_trade(credit=2.00, expiration=exp)
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        put_short_sym = mon.alpaca._build_occ_symbol("SPY", exp, 440.0, "put")
        put_long_sym  = mon.alpaca._build_occ_symbol("SPY", exp, 435.0, "put")
        call_short_sym = mon.alpaca._build_occ_symbol("SPY", exp, 470.0, "call")
        call_long_sym  = mon.alpaca._build_occ_symbol("SPY", exp, 475.0, "call")

        # Combined value = 0.90 → pnl% = (2.00-0.90)/2.00*100 = 55% > 50% PT
        alpaca_positions = {
            put_short_sym:  _alpaca_pos(put_short_sym, -50.0),   # 0.50/share cost
            put_long_sym:   _alpaca_pos(put_long_sym, 5.0),
            call_short_sym: _alpaca_pos(call_short_sym, -44.0),  # 0.44/share cost (after netting)
            call_long_sym:  _alpaca_pos(call_long_sym, 1.0),
        }
        # put wing: (50-5)/100 = 0.45; call wing: (44-1)/100 = 0.43; total = 0.88
        reason = mon._check_exit_conditions(pos, alpaca_positions)
        assert reason == "profit_target"


# ---------------------------------------------------------------------------
# Bug 1: Iron condor close (_close_position routes to close_iron_condor)
# ---------------------------------------------------------------------------

class TestIronCondorClose:
    """_close_position must call close_iron_condor for IC positions."""

    def test_ic_close_calls_close_iron_condor(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db = f.name
        init_db(db)

        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_ic_trade(
            put_short=440.0, put_long=435.0,
            call_short=470.0, call_long=475.0,
            expiration="2099-12-31",
        )
        upsert_trade(pos, source="execution", path=db)

        mon._close_position(pos, "profit_target")

        alpaca.close_iron_condor.assert_called_once()
        kwargs = alpaca.close_iron_condor.call_args[1]
        assert kwargs["put_short_strike"] == 440.0
        assert kwargs["put_long_strike"] == 435.0
        assert kwargs["call_short_strike"] == 470.0
        assert kwargs["call_long_strike"] == 475.0
        alpaca.close_spread.assert_not_called()

    def test_regular_spread_close_calls_close_spread(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db = f.name
        init_db(db)

        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_trade(strategy_type="bull_put", short_strike=450.0, long_strike=445.0)
        upsert_trade(pos, source="execution", path=db)

        mon._close_position(pos, "stop_loss")

        alpaca.close_spread.assert_called_once()
        alpaca.close_iron_condor.assert_not_called()

    def test_ic_close_stores_order_id_in_db(self):
        """close_order_id must be saved so _reconcile_pending_closes can poll fill."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db = f.name
        init_db(db)

        alpaca = _make_alpaca()
        alpaca.close_iron_condor.return_value = {"status": "submitted", "order_id": "ord-99"}
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_ic_trade(expiration="2099-12-31")
        upsert_trade(pos, source="execution", path=db)

        mon._close_position(pos, "dte_management")

        trades = get_trades(status="pending_close", path=db)
        assert len(trades) == 1
        assert trades[0].get("close_order_id") == "ord-99"

    def test_ic_missing_call_strikes_returns_error(self):
        """If IC trade record missing call strikes, close should log error not crash."""
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            db = f.name
        init_db(db)

        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_ic_trade(expiration="2099-12-31")
        pos.pop("call_short_strike", None)
        pos.pop("call_long_strike", None)
        upsert_trade(pos, source="execution", path=db)

        # Should not raise
        mon._close_position(pos, "profit_target")
        alpaca.close_iron_condor.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 2: P&L recording after close fill (_reconcile_pending_closes)
# ---------------------------------------------------------------------------

class TestPnLRecording:
    """_reconcile_pending_closes must record realized P&L when order fills."""

    def _setup_db(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db = f.name
        f.close()
        init_db(db)
        return db

    def test_filled_order_records_profit(self):
        db = self._setup_db()
        fill_price = 0.40   # closed at 0.40, opened at credit=1.00 → profit
        alpaca = _make_alpaca(order_status={
            "id": "ord-001",
            "status": "filled",
            "filled_avg_price": str(fill_price),
            "filled_at": datetime.now(timezone.utc).isoformat(),
        })
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_trade(
            trade_id="t1", status="pending_close",
            credit=1.00, contracts=2,
            close_order_id="ord-001",
            exit_reason="profit_target",
        )
        upsert_trade(pos, source="execution", path=db)

        mon._reconcile_pending_closes()

        trades = get_trades(path=db)
        assert len(trades) == 1
        t = trades[0]
        # gross pnl = (1.00 - 0.40) * 2 * 100 = 120
        # commission = 0.65 * 2 contracts * 2 legs * 2 sides = $5.20 (default 0.65)
        # net pnl = 120 - 5.20 = 114.80
        assert t["status"] == "closed_profit"
        assert abs(t["pnl"] - 114.80) < 0.01
        assert t["exit_date"] is not None

    def test_filled_order_records_loss(self):
        db = self._setup_db()
        fill_price = 2.80   # closed at 2.80, opened at credit=1.00 → loss
        alpaca = _make_alpaca(order_status={
            "status": "filled",
            "filled_avg_price": str(fill_price),
            "filled_at": datetime.now(timezone.utc).isoformat(),
        })
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_trade(
            trade_id="t2", status="pending_close",
            credit=1.00, contracts=1,
            close_order_id="ord-002",
            exit_reason="stop_loss",
        )
        upsert_trade(pos, source="execution", path=db)

        mon._reconcile_pending_closes()

        trades = get_trades(path=db)
        t = trades[0]
        # gross pnl = (1.00 - 2.80) * 1 * 100 = -180
        # commission = 0.65 * 1 contract * 2 legs * 2 sides = $2.60 (default 0.65)
        # net pnl = -180 - 2.60 = -182.60
        assert t["status"] == "closed_loss"
        assert abs(t["pnl"] - (-182.60)) < 0.01

    def test_pending_order_left_unchanged(self):
        db = self._setup_db()
        alpaca = _make_alpaca(order_status={
            "status": "new",   # not yet filled
            "filled_avg_price": None,
        })
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_trade(
            trade_id="t3", status="pending_close",
            close_order_id="ord-003",
        )
        upsert_trade(pos, source="execution", path=db)

        mon._reconcile_pending_closes()

        trades = get_trades(path=db)
        assert trades[0]["status"] == "pending_close"   # unchanged

    def test_cancelled_order_resets_to_open(self):
        db = self._setup_db()
        alpaca = _make_alpaca(order_status={
            "status": "cancelled",
            "filled_avg_price": None,
        })
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_trade(
            trade_id="t4", status="pending_close",
            close_order_id="ord-004",
            exit_reason="profit_target",
        )
        upsert_trade(pos, source="execution", path=db)

        mon._reconcile_pending_closes()

        trades = get_trades(path=db)
        assert trades[0]["status"] == "open"

    def test_no_order_id_skipped(self):
        """Positions with no close_order_id are skipped gracefully."""
        db = self._setup_db()
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_trade(trade_id="t5", status="pending_close")
        # no close_order_id set
        upsert_trade(pos, source="execution", path=db)

        mon._reconcile_pending_closes()

        # status unchanged — no order to poll
        trades = get_trades(path=db)
        assert trades[0]["status"] == "pending_close"
        alpaca.get_order_status.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 3: External close detection (_reconcile_external_closes)
# ---------------------------------------------------------------------------

class TestExternalCloseDetection:

    def _setup_db(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db = f.name
        f.close()
        init_db(db)
        return db

    def test_missing_legs_marked_closed_external(self):
        db = self._setup_db()
        alpaca = _make_alpaca(positions=[])   # empty Alpaca — position gone
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_trade(
            trade_id="ext1",
            strategy_type="bull_put",
            short_strike=450.0, long_strike=445.0,
            expiration="2025-06-20",    # past expiration to simulate real scenario
        )
        upsert_trade(pos, source="execution", path=db)

        open_positions = get_trades(status="open", path=db)
        mon._reconcile_external_closes(open_positions, {})

        trades = get_trades(path=db)
        assert trades[0]["status"] == "closed_external"
        assert trades[0]["exit_reason"] == "closed_external"
        assert trades[0]["exit_date"] is not None

    def test_in_memory_status_mutated_for_exit_loop_skip(self):
        """_reconcile_external_closes must mutate pos['status'] so exit loop skips it."""
        alpaca = _make_alpaca(positions=[])
        mon = _monitor(alpaca=alpaca)

        pos = _make_trade(
            strategy_type="bull_put",
            short_strike=450.0, long_strike=445.0,
            expiration="2025-06-20",
        )
        open_positions = [pos]
        mon._reconcile_external_closes(open_positions, {})

        assert pos["status"] == "closed_external"

    def test_present_position_not_marked_external(self):
        db = self._setup_db()
        exp = "2025-06-20"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, db_path=db)

        short_sym = mon.alpaca._build_occ_symbol("SPY", exp, 450.0, "put")
        long_sym  = mon.alpaca._build_occ_symbol("SPY", exp, 445.0, "put")
        alpaca_positions = {
            short_sym: _alpaca_pos(short_sym, -100.0),
            long_sym:  _alpaca_pos(long_sym, 30.0),
        }

        pos = _make_trade(
            trade_id="present1",
            strategy_type="bull_put",
            short_strike=450.0, long_strike=445.0,
            expiration=exp,
        )
        upsert_trade(pos, source="execution", path=db)
        open_positions = [pos]

        mon._reconcile_external_closes(open_positions, alpaca_positions)

        assert pos["status"] == "open"    # unchanged

    def test_ic_all_legs_missing_marked_external(self):
        db = self._setup_db()
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_ic_trade(trade_id="ic-ext", expiration="2025-06-20")
        upsert_trade(pos, source="execution", path=db)
        open_positions = [pos]

        # Empty alpaca positions → all 4 legs missing
        mon._reconcile_external_closes(open_positions, {})

        assert pos["status"] == "closed_external"

    def test_ic_partial_legs_not_marked_external(self):
        """If some IC legs still exist, do NOT mark as externally closed."""
        exp = "2025-06-20"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        pos = _make_ic_trade(expiration=exp)

        # Only put short leg present — call wing gone (could be partial assignment)
        put_short_sym = mon.alpaca._build_occ_symbol("SPY", exp, 440.0, "put")
        alpaca_positions = {put_short_sym: _alpaca_pos(put_short_sym, -80.0)}

        result = mon._all_legs_missing(pos, alpaca_positions)
        assert result is False  # not ALL legs missing → don't mark external


# ---------------------------------------------------------------------------
# Market hours gate
# ---------------------------------------------------------------------------

class TestMarketHoursGate:

    def test_market_open_weekday(self):
        # Patch to Wednesday 10:30 ET
        with patch("execution.position_monitor.datetime") as mock_dt:
            datetime(2026, 3, 4, 10, 30, tzinfo=None)   # weekday=2
            mock_dt.now.return_value = MagicMock(
                weekday=lambda: 2,  # Wednesday
                hour=10, minute=30,
            )
            assert PositionMonitor._is_market_hours() is True

    def test_market_closed_before_open(self):
        with patch("execution.position_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(
                weekday=lambda: 1,  # Tuesday
                hour=9, minute=15,   # before 9:30
            )
            assert PositionMonitor._is_market_hours() is False

    def test_market_closed_after_close(self):
        with patch("execution.position_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(
                weekday=lambda: 3,  # Thursday
                hour=16, minute=0,   # exactly 4:00 PM — closed (exclusive)
            )
            assert PositionMonitor._is_market_hours() is False

    def test_market_closed_weekend(self):
        with patch("execution.position_monitor.datetime") as mock_dt:
            mock_dt.now.return_value = MagicMock(
                weekday=lambda: 5,  # Saturday
                hour=12, minute=0,
            )
            assert PositionMonitor._is_market_hours() is False

    def test_check_skipped_when_market_closed(self):
        """_check_positions must return immediately when market is closed."""
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        with patch.object(PositionMonitor, "_is_market_hours", return_value=False):
            mon._check_positions()

        alpaca.get_positions.assert_not_called()
        alpaca.get_order_status.assert_not_called()

    def test_check_runs_when_market_open(self):
        """_check_positions proceeds past market gate when open."""
        alpaca = _make_alpaca(positions=[])
        mon = _monitor(alpaca=alpaca)

        with patch.object(PositionMonitor, "_is_market_hours", return_value=True):
            mon._check_positions()

        # Even with no open DB positions, the gate was passed (get_positions called via reconcile)
        # _reconcile_pending_closes calls get_order_status only if there are pending positions
        # Here there are none — just verify no exception and market gate passed
        # (get_positions is called during _check_positions step 3 only if open_positions exist)


# ---------------------------------------------------------------------------
# Integration: full cycle with IC
# ---------------------------------------------------------------------------

class TestFullIcCycle:
    """Integration: IC position goes from open → pending_close → closed_profit."""

    def _setup_db(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db = f.name
        f.close()
        init_db(db)
        return db

    def test_ic_profit_target_close_and_pnl_recorded(self):
        db = self._setup_db()
        exp = "2099-12-31"

        alpaca = _make_alpaca()
        alpaca.close_iron_condor.return_value = {
            "status": "submitted", "order_id": "ic-close-999"
        }
        alpaca.get_order_status.return_value = {
            "status": "filled",
            "filled_avg_price": "0.80",   # closed at 0.80, credit=2.00 → profit
            "filled_at": datetime.now(timezone.utc).isoformat(),
        }

        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_ic_trade(
            trade_id="ic-full-1",
            put_short=440.0, put_long=435.0,
            call_short=470.0, call_long=475.0,
            expiration=exp, credit=2.00, contracts=2,
        )
        upsert_trade(pos, source="execution", path=db)

        # Simulate current IC value = 0.90 (55% profit → triggers PT)
        put_short_sym  = mon.alpaca._build_occ_symbol("SPY", exp, 440.0, "put")
        put_long_sym   = mon.alpaca._build_occ_symbol("SPY", exp, 435.0, "put")
        call_short_sym = mon.alpaca._build_occ_symbol("SPY", exp, 470.0, "call")
        call_long_sym  = mon.alpaca._build_occ_symbol("SPY", exp, 475.0, "call")

        alpaca_positions = {
            put_short_sym:  _alpaca_pos(put_short_sym, -50.0),
            put_long_sym:   _alpaca_pos(put_long_sym, 5.0),
            call_short_sym: _alpaca_pos(call_short_sym, -44.0),
            call_long_sym:  _alpaca_pos(call_long_sym, 1.0),
        }
        # put wing cost: (50-5)/200 = 0.225; call wing cost: (44-1)/200 = 0.215; total ≈ 0.44
        # pnl% = (2.00-0.44)/2.00 * 100 = 78% > 50% → PT triggers
        alpaca.get_positions.return_value = list(alpaca_positions.values())
        for sym, p in alpaca_positions.items():
            p["symbol"] = sym

        with patch.object(PositionMonitor, "_is_market_hours", return_value=True):
            # Cycle 1: detect PT, submit close
            mon._check_positions()

        trades = get_trades(path=db)
        assert trades[0]["status"] == "pending_close"
        alpaca.close_iron_condor.assert_called_once()

        # Cycle 2: fill comes in, P&L recorded
        with patch.object(PositionMonitor, "_is_market_hours", return_value=True):
            # No open positions in this cycle
            alpaca.get_positions.return_value = []
            mon._check_positions()

        trades = get_trades(path=db)
        t = trades[0]
        assert t["status"] == "closed_profit"
        # gross pnl = (2.00 - 0.80) * 2 * 100 = 240
        # commission = 0.65 * 2 contracts * 4 legs * 2 sides = $10.40 (default 0.65)
        # net pnl = 240 - 10.40 = 229.60
        assert abs(t["pnl"] - 229.60) < 0.01


# ---------------------------------------------------------------------------
# Straddle/Strangle support
# ---------------------------------------------------------------------------

def _make_straddle_trade(
    trade_id: str = "ss1",
    ticker: str = "SPY",
    strategy_type: str = "short_straddle",
    call_strike: float = 500.0,
    put_strike: float = 500.0,
    expiration: str = "2099-12-31",
    credit: float = 5.00,
    contracts: int = 1,
    is_debit: bool = False,
    **extra,
) -> Dict:
    t = dict(
        id=trade_id,
        ticker=ticker,
        strategy_type=strategy_type,
        status="open",
        call_strike=call_strike,
        put_strike=put_strike,
        expiration=expiration,
        credit=credit,
        contracts=contracts,
        is_debit=is_debit,
        entry_date=datetime.now(timezone.utc).isoformat(),
    )
    t.update(extra)
    return t


class TestStraddleValuation:
    """_get_straddle_value pricing for long and short straddles."""

    def test_short_straddle_value(self):
        """Short straddle: cost to close = sum of absolute market values."""
        exp = "2099-12-31"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        pos = _make_straddle_trade(
            strategy_type="short_straddle",
            call_strike=500.0, put_strike=500.0,
            credit=5.00, contracts=2,
        )

        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        put_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "put")
        alpaca_positions = {
            call_sym: _alpaca_pos(call_sym, -300.0),  # short call liability
            put_sym: _alpaca_pos(put_sym, -200.0),     # short put liability
        }

        val = mon._get_straddle_value(pos, alpaca_positions)
        # (300 + 200) / (2 * 100) = 2.50 per share
        assert val is not None
        assert abs(val - 2.50) < 0.01

    def test_long_straddle_value(self):
        """Long straddle: value = sum of market values (assets we hold)."""
        exp = "2099-12-31"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        pos = _make_straddle_trade(
            strategy_type="long_straddle",
            call_strike=500.0, put_strike=500.0,
            credit=-4.00, contracts=1, is_debit=True,
        )

        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        put_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "put")
        alpaca_positions = {
            call_sym: _alpaca_pos(call_sym, 350.0),   # long call asset
            put_sym: _alpaca_pos(put_sym, 200.0),      # long put asset
        }

        val = mon._get_straddle_value(pos, alpaca_positions)
        # (350 + 200) / (1 * 100) = 5.50 per share
        assert val is not None
        assert abs(val - 5.50) < 0.01

    def test_straddle_value_none_when_leg_missing(self):
        """Returns None when one leg is absent from Alpaca."""
        exp = "2099-12-31"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        pos = _make_straddle_trade()
        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        # Only call leg present, put missing
        alpaca_positions = {call_sym: _alpaca_pos(call_sym, -100.0)}

        val = mon._get_straddle_value(pos, alpaca_positions)
        assert val is None

    def test_spread_value_routes_to_straddle(self):
        """_get_spread_value routes straddle/strangle types correctly."""
        exp = "2099-12-31"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        for stype in ["short_straddle", "long_strangle", "short_strangle"]:
            pos = _make_straddle_trade(strategy_type=stype)
            call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
            put_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "put")
            alpaca_positions = {
                call_sym: _alpaca_pos(call_sym, -100.0),
                put_sym: _alpaca_pos(put_sym, -100.0),
            }
            val = mon._get_spread_value(pos, alpaca_positions)
            assert val is not None


class TestStraddleExitConditions:
    """Exit logic for credit (short) and debit (long) straddle positions."""

    def test_short_straddle_profit_target(self):
        """Short straddle hits PT when spread value drops below credit."""
        exp = "2099-12-31"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, profit_target=50.0)

        pos = _make_straddle_trade(
            strategy_type="short_straddle", credit=5.00, contracts=1,
        )

        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        put_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "put")
        # value = (100+50)/100 = 1.50; pnl_pct = (5.00-1.50)/5.00 = 70% > 50% PT
        alpaca_positions = {
            call_sym: _alpaca_pos(call_sym, -100.0),
            put_sym: _alpaca_pos(put_sym, -50.0),
        }

        reason = mon._check_exit_conditions(pos, alpaca_positions)
        assert reason == "profit_target"

    def test_short_straddle_no_exit(self):
        """Short straddle within thresholds — no exit."""
        exp = "2099-12-31"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, profit_target=50.0)

        pos = _make_straddle_trade(
            strategy_type="short_straddle", credit=5.00, contracts=1,
        )

        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        put_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "put")
        # value = (200+200)/100 = 4.00; pnl_pct = (5.00-4.00)/5.00 = 20% < 50% PT
        alpaca_positions = {
            call_sym: _alpaca_pos(call_sym, -200.0),
            put_sym: _alpaca_pos(put_sym, -200.0),
        }

        reason = mon._check_exit_conditions(pos, alpaca_positions)
        assert reason is None

    def test_long_straddle_profit_target(self):
        """Long straddle hits PT when value exceeds debit paid by target %."""
        exp = "2099-12-31"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, profit_target=50.0)

        pos = _make_straddle_trade(
            strategy_type="long_straddle", credit=-4.00, contracts=1,
            is_debit=True, profit_target_pct=50.0,
        )

        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        put_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "put")
        # value = (500+300)/100 = 8.00; debit=4.00; profit_pct = (8-4)/4 = 100% > 50%
        alpaca_positions = {
            call_sym: _alpaca_pos(call_sym, 500.0),
            put_sym: _alpaca_pos(put_sym, 300.0),
        }

        reason = mon._check_exit_conditions(pos, alpaca_positions)
        assert reason == "profit_target"

    def test_long_straddle_stop_loss(self):
        """Long straddle hits SL when value drops below debit by stop %."""
        exp = "2099-12-31"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca, sl_mult=3.5)

        pos = _make_straddle_trade(
            strategy_type="long_straddle", credit=-4.00, contracts=1,
            is_debit=True, stop_loss_pct=50.0,
        )

        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        put_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "put")
        # value = (50+50)/100 = 1.00; debit=4.00; loss_pct = (4-1)/4 = 75% > 50%
        alpaca_positions = {
            call_sym: _alpaca_pos(call_sym, 50.0),
            put_sym: _alpaca_pos(put_sym, 50.0),
        }

        reason = mon._check_exit_conditions(pos, alpaca_positions)
        assert reason == "stop_loss"

    def test_long_straddle_no_exit(self):
        """Long straddle within thresholds — no exit."""
        exp = "2099-12-31"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        pos = _make_straddle_trade(
            strategy_type="long_straddle", credit=-4.00, contracts=1,
            is_debit=True, profit_target_pct=50.0, stop_loss_pct=50.0,
        )

        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        put_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "put")
        # value = (250+150)/100 = 4.00; debit=4.00; pnl=0% — flat
        alpaca_positions = {
            call_sym: _alpaca_pos(call_sym, 250.0),
            put_sym: _alpaca_pos(put_sym, 150.0),
        }

        reason = mon._check_exit_conditions(pos, alpaca_positions)
        assert reason is None


class TestStraddleExternalCloseDetection:
    """_all_legs_missing and _detect_orphans for straddle positions."""

    def test_straddle_all_legs_missing(self):
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        pos = _make_straddle_trade(expiration="2025-06-20")
        result = mon._all_legs_missing(pos, {})
        assert result is True

    def test_straddle_one_leg_present(self):
        exp = "2025-06-20"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        pos = _make_straddle_trade(expiration=exp)
        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        alpaca_positions = {call_sym: _alpaca_pos(call_sym, -100.0)}

        result = mon._all_legs_missing(pos, alpaca_positions)
        assert result is False

    def test_straddle_both_legs_present(self):
        exp = "2025-06-20"
        alpaca = _make_alpaca()
        mon = _monitor(alpaca=alpaca)

        pos = _make_straddle_trade(expiration=exp)
        call_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "call")
        put_sym = mon.alpaca._build_occ_symbol("SPY", exp, 500.0, "put")
        alpaca_positions = {
            call_sym: _alpaca_pos(call_sym, -100.0),
            put_sym: _alpaca_pos(put_sym, -100.0),
        }

        result = mon._all_legs_missing(pos, alpaca_positions)
        assert result is False


class TestStraddleCloseAndPnL:
    """Closing and P&L recording for straddle positions."""

    def _setup_db(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        db = f.name
        f.close()
        init_db(db)
        return db

    def test_close_routes_to_submit_straddle_close(self):
        """_close_position routes straddle to _submit_straddle_close."""
        db = self._setup_db()
        alpaca = _make_alpaca()
        alpaca.submit_single_leg.return_value = {"status": "submitted", "order_id": "sl-001"}
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_straddle_trade(trade_id="ss-close-1")
        upsert_trade(pos, source="execution", path=db)

        mon._close_position(pos, "profit_target")

        # Should call submit_single_leg for call AND put
        assert alpaca.submit_single_leg.call_count == 2

    def test_debit_pnl_recording(self):
        """P&L for a debit (long) straddle: pnl = (fill_price - debit_paid) * contracts * 100."""
        db = self._setup_db()
        alpaca = _make_alpaca(order_status={
            "status": "filled",
            "filled_avg_price": "6.50",
        })
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_straddle_trade(
            trade_id="ss-pnl-1", strategy_type="long_straddle",
            credit=-4.00, contracts=2, is_debit=True,
            status="pending_close", close_order_id="ord-ss-001",
        )
        upsert_trade(pos, source="execution", path=db)

        mon._reconcile_pending_closes()

        trades = get_trades(path=db)
        t = trades[0]
        assert t["status"] == "closed_profit"
        # gross pnl = (6.50 - 4.00) * 2 * 100 = 500
        # commission = 0.65 * 2 contracts * 2 legs * 2 sides = $5.20 (default 0.65)
        # net pnl = 500 - 5.20 = 494.80
        assert abs(t["pnl"] - 494.80) < 0.01

    def test_credit_pnl_recording_for_short_straddle(self):
        """P&L for a credit (short) straddle: pnl = (credit - fill_price) * contracts * 100."""
        db = self._setup_db()
        alpaca = _make_alpaca(order_status={
            "status": "filled",
            "filled_avg_price": "2.00",
        })
        mon = _monitor(alpaca=alpaca, db_path=db)

        pos = _make_straddle_trade(
            trade_id="ss-pnl-2", strategy_type="short_straddle",
            credit=5.00, contracts=1,
            status="pending_close", close_order_id="ord-ss-002",
        )
        upsert_trade(pos, source="execution", path=db)

        mon._reconcile_pending_closes()

        trades = get_trades(path=db)
        t = trades[0]
        assert t["status"] == "closed_profit"
        # gross pnl = (5.00 - 2.00) * 1 * 100 = 300
        # commission = 0.65 * 1 contract * 2 legs * 2 sides = $2.60 (default 0.65)
        # net pnl = 300 - 2.60 = 297.40
        assert abs(t["pnl"] - 297.40) < 0.01
