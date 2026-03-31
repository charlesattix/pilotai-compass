"""
Hardening Pass 3 — tests for the final sweep of gaps:
  - Stop-loss threshold: spread_width cap + backtester-aligned formula
  - Close order failure → immediate reset to open
  - DB write failure → WAL recovery file
  - Startup reconciliation: phantom positions (DB open, Alpaca missing)
  - Startup reconciliation: orphan positions (Alpaca has, DB missing)
  - Consecutive Alpaca API failure alerting
  - Wide bid-ask: all closes use market orders (explicit test)
"""

import pytest

try:
    from alpaca.trading.requests import OptionLegRequest  # noqa: F401
except ImportError:
    pytest.skip("OptionLegRequest not available in this alpaca-py version", allow_module_level=True)

import json
import os
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monitor(tmp_path, config=None):
    from execution.position_monitor import PositionMonitor
    from shared.database import init_db
    db = str(tmp_path / "trades.db")
    init_db(db)
    cfg = config or {"risk": {"profit_target": 50, "stop_loss_multiplier": 3.5}, "strategy": {}}
    alpaca = MagicMock()
    alpaca.get_positions.return_value = []
    monitor = PositionMonitor(alpaca_provider=alpaca, config=cfg, db_path=db)
    return monitor, db


def _make_open_pos(tmp_path, trade_id="pos-001", credit=1.50, short_strike=545.0,
                   long_strike=540.0, spread_type="bull_put", db=None):
    from shared.database import upsert_trade
    pos = {
        "id": trade_id,
        "ticker": "SPY",
        "strategy_type": spread_type,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "expiration": "2026-12-18",
        "credit": credit,
        "contracts": 1,
        "status": "open",
    }
    if db:
        upsert_trade(pos, source="execution", path=db)
    return pos


# ===========================================================================
# Stop-Loss Threshold
# ===========================================================================

class TestStopLossThreshold:
    """Verify SL uses formula-only threshold matching the backtester.

    Backtester: fires when spread_value >= (1 + stop_loss_mult) × credit.
    The 90% spread-width cap was removed (backtester has no width cap).
    """

    def _check_sl(self, monitor, pos, current_value):
        """Call _check_exit_conditions with a mocked spread value."""
        with patch.object(monitor, "_get_spread_value", return_value=current_value):
            return monitor._check_exit_conditions(pos, {})

    def test_narrow_spread_sl_does_not_fire_below_formula_threshold(self, tmp_path):
        """$5 spread, $1.50 credit: formula threshold = (1+3.5)×$1.50 = $6.75.
        Since spread_width ($5) < threshold ($6.75), SL via formula is unreachable.
        At $4.50 (old 90% cap level), SL must NOT fire — no width cap anymore."""
        monitor, _ = _make_monitor(tmp_path)
        pos = _make_open_pos(tmp_path, credit=1.50, short_strike=545.0, long_strike=540.0)
        # Value at $4.50 — below formula threshold $6.75 → no SL (no width cap)
        result = self._check_sl(monitor, pos, current_value=4.50)
        assert result != "stop_loss"

    def test_narrow_spread_sl_fires_at_formula_threshold(self, tmp_path):
        """SL fires when spread_value reaches formula threshold (1+mult)×credit."""
        monitor, _ = _make_monitor(tmp_path)
        pos = _make_open_pos(tmp_path, credit=1.50, short_strike=545.0, long_strike=540.0)
        # Threshold = (1+3.5) × 1.50 = 6.75 — fires exactly at threshold
        result = self._check_sl(monitor, pos, current_value=6.75)
        assert result == "stop_loss"

    def test_wide_spread_sl_fires_on_loss_formula(self, tmp_path):
        """$20 wide spread, $3.00 credit: formula SL = (1+3.5) × $3 = $13.50."""
        monitor, _ = _make_monitor(tmp_path)
        pos = _make_open_pos(tmp_path, credit=3.00, short_strike=560.0, long_strike=540.0)
        # At $13.50 — formula threshold, SL fires
        result = self._check_sl(monitor, pos, current_value=13.50)
        assert result == "stop_loss"

    def test_wide_spread_sl_does_not_fire_below_formula(self, tmp_path):
        monitor, _ = _make_monitor(tmp_path)
        pos = _make_open_pos(tmp_path, credit=3.00, short_strike=560.0, long_strike=540.0)
        # $13.49 → just below formula threshold
        result = self._check_sl(monitor, pos, current_value=13.49)
        assert result != "stop_loss"

    def test_missing_strike_data_falls_back_to_formula(self, tmp_path):
        """When strike data is missing, no spread_width cap; formula-only threshold."""
        monitor, _ = _make_monitor(tmp_path)
        pos = {
            "id": "no-strikes",
            "ticker": "SPY",
            "strategy_type": "bull_put",
            "short_strike": None,
            "long_strike": None,
            "expiration": "2026-12-18",
            "credit": 1.50,
            "contracts": 1,
            "status": "open",
        }
        # Without spread_width, formula SL = 4.5 × $1.50 = $6.75
        result = self._check_sl(monitor, pos, current_value=6.75)
        assert result == "stop_loss"

    def test_sl_threshold_matches_backtester_formula(self, tmp_path):
        """Confirm the formula (1 + mult) × credit matches what backtester uses.
        Backtester fires when: loss = spread_value - credit >= mult × credit
        i.e., spread_value >= (1 + mult) × credit."""
        monitor, _ = _make_monitor(tmp_path, config={
            "risk": {"profit_target": 50, "stop_loss_multiplier": 3.5},
            "strategy": {},
        })
        # Use wide spread ($50) so cap doesn't interfere
        pos = _make_open_pos(tmp_path, credit=2.00, short_strike=600.0, long_strike=550.0)
        # (1 + 3.5) × $2.00 = $9.00 — backtester fires exactly here
        result = self._check_sl(monitor, pos, current_value=9.00)
        assert result == "stop_loss"
        result_below = self._check_sl(monitor, pos, current_value=8.99)
        assert result_below != "stop_loss"

    def test_profit_target_still_fires_normally(self, tmp_path):
        """Ensure profit target check is not broken by SL changes."""
        monitor, _ = _make_monitor(tmp_path, config={
            "risk": {"profit_target": 50, "stop_loss_multiplier": 3.5},
            "strategy": {"manage_dte": 21},
        })
        pos = _make_open_pos(tmp_path, credit=2.00, short_strike=545.0, long_strike=540.0)
        # At $1.00 current_value, pnl = $1.00, pnl% = 50% → profit target fires
        result = self._check_sl(monitor, pos, current_value=1.00)
        assert result == "profit_target"


# ===========================================================================
# Close Order Failure → Reset to Open
# ===========================================================================

class TestCloseOrderFailureReset:

    def _open_trade_in_db(self, db):
        from shared.database import upsert_trade
        pos = {
            "id": "close-fail-001",
            "ticker": "SPY",
            "strategy_type": "bull_put",
            "short_strike": 545.0,
            "long_strike": 540.0,
            "expiration": "2026-12-18",
            "credit": 1.50,
            "contracts": 1,
            "status": "open",
        }
        upsert_trade(pos, source="execution", path=db)
        return dict(pos)

    def test_close_failure_resets_position_to_open(self, tmp_path):
        """If Alpaca rejects a close order, the position must revert to open."""
        from shared.database import get_trades
        monitor, db = _make_monitor(tmp_path)
        pos = self._open_trade_in_db(db)

        monitor.alpaca.close_spread.return_value = {
            "status": "error",
            "message": "422: invalid symbol",
        }

        monitor._close_position(pos, reason="stop_loss")

        trades = get_trades(path=db)
        t = next(x for x in trades if x["id"] == "close-fail-001")
        assert t["status"] == "open"
        assert t.get("close_order_id") is None
        assert t.get("exit_reason") is None

    def test_close_failure_does_not_leave_pending_close(self, tmp_path):
        """After a failed close, the position must NOT be stuck in pending_close."""
        from shared.database import get_trades
        monitor, db = _make_monitor(tmp_path)
        pos = self._open_trade_in_db(db)
        pos["status"] = "pending_close"
        pos["exit_reason"] = "stop_loss"

        monitor.alpaca.close_spread.return_value = {
            "status": "error",
            "message": "500: internal server error",
        }

        monitor._close_position(pos, reason="stop_loss")

        trades = get_trades(path=db)
        t = next(x for x in trades if x["id"] == "close-fail-001")
        assert t["status"] == "open"

    def test_successful_close_keeps_pending_close_status(self, tmp_path):
        """Sanity check: a successful close submission leaves pending_close intact."""
        from shared.database import get_trades
        monitor, db = _make_monitor(tmp_path)
        pos = self._open_trade_in_db(db)

        monitor.alpaca.close_spread.return_value = {
            "status": "submitted",
            "order_id": "ord-close-001",
        }

        monitor._close_position(pos, reason="profit_target")

        trades = get_trades(path=db)
        t = next(x for x in trades if x["id"] == "close-fail-001")
        assert t["status"] == "pending_close"


# ===========================================================================
# WAL Recovery on DB Write Failure
# ===========================================================================

class TestWALRecovery:

    def test_wal_entry_written_on_close_trade_failure(self, tmp_path):
        """If close_trade() raises, a WAL recovery entry must be written."""
        wal_file = str(tmp_path / "recovery.wal")
        monitor, db = _make_monitor(tmp_path, config={
            "risk": {"profit_target": 50, "stop_loss_multiplier": 3.5},
            "strategy": {},
            "execution": {"wal_path": wal_file},
        })

        pos = {
            "id": "wal-test-001",
            "ticker": "SPY",
            "strategy_type": "bull_put",
            "credit": 1.50,
            "contracts": 1,
            "status": "pending_close",
            "exit_reason": "profit_target",
        }
        order = {
            "status": "filled",
            "filled_avg_price": "0.75",
            "filled_qty": "1",
            "qty": "1",
        }

        with patch("execution.position_monitor.close_trade", side_effect=Exception("SQLite write error")):
            monitor._record_close_pnl(pos, order)

        assert os.path.exists(wal_file), "WAL file must be created on DB failure"
        with open(wal_file) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["type"] == "close_trade"
        assert entry["trade_id"] == "wal-test-001"
        assert entry["_processed"] is False

    def test_wal_replay_returns_unprocessed_entries(self, tmp_path):
        """replay_wal() must return all entries where _processed=False."""
        from shared.wal import replay_wal, write_wal_entry
        wal_file = str(tmp_path / "recovery.wal")

        write_wal_entry({"type": "close_trade", "trade_id": "a"}, wal_path=wal_file)
        write_wal_entry({"type": "close_trade", "trade_id": "b"}, wal_path=wal_file)

        entries = replay_wal(wal_path=wal_file)
        assert len(entries) == 2
        trade_ids = {e["trade_id"] for e in entries}
        assert trade_ids == {"a", "b"}

    def test_wal_clear_removes_file(self, tmp_path):
        from shared.wal import clear_wal, write_wal_entry
        wal_file = str(tmp_path / "recovery.wal")
        write_wal_entry({"type": "test"}, wal_path=wal_file)
        assert os.path.exists(wal_file)
        clear_wal(wal_path=wal_file)
        assert not os.path.exists(wal_file)

    def test_replay_wal_empty_when_no_file(self, tmp_path):
        from shared.wal import replay_wal
        entries = replay_wal(wal_path=str(tmp_path / "nonexistent.wal"))
        assert entries == []

    def test_wal_write_does_not_raise_on_success(self, tmp_path):
        from shared.wal import write_wal_entry
        wal_file = str(tmp_path / "recovery.wal")
        write_wal_entry({"type": "test", "trade_id": "x"}, wal_path=wal_file)  # must not raise


# ===========================================================================
# Startup Reconciliation — Phantom Positions
# ===========================================================================

class TestStartupReconciliationPhantom:
    """DB says open, Alpaca has no matching position → needs_investigation."""

    def _make_reconciler(self, db):
        from shared.reconciler import PositionReconciler
        alpaca = MagicMock()
        # _build_occ_symbol: return deterministic symbols
        alpaca._build_occ_symbol.side_effect = lambda t, e, s, o: f"{t}{e}{o}{int(s)}"
        return PositionReconciler(alpaca=alpaca, db_path=db)

    def test_open_trade_not_in_alpaca_marked_needs_investigation(self, tmp_path):
        from shared.database import get_trades, init_db, upsert_trade
        db = str(tmp_path / "trades.db")
        init_db(db)

        # Insert an open trade
        upsert_trade({
            "id": "phantom-001",
            "ticker": "SPY",
            "strategy_type": "bull_put",
            "short_strike": 545.0,
            "long_strike": 540.0,
            "expiration": "2026-01-16",
            "credit": 1.50,
            "contracts": 1,
            "status": "open",
        }, source="execution", path=db)

        reconciler = self._make_reconciler(db)
        # Alpaca returns empty positions (trade expired/closed while system was down)
        reconciler.alpaca.get_positions.return_value = []

        result = reconciler.reconcile()

        assert result.phantom_resolved == 1
        trades = get_trades(path=db)
        t = next(x for x in trades if x["id"] == "phantom-001")
        assert t["status"] == "needs_investigation"

    def test_open_trade_still_in_alpaca_not_touched(self, tmp_path):
        from shared.database import get_trades, init_db, upsert_trade
        db = str(tmp_path / "trades.db")
        init_db(db)

        upsert_trade({
            "id": "live-001",
            "ticker": "SPY",
            "strategy_type": "bull_put",
            "short_strike": 545.0,
            "long_strike": 540.0,
            "expiration": "2026-01-16",
            "credit": 1.50,
            "contracts": 1,
            "status": "open",
        }, source="execution", path=db)

        reconciler = self._make_reconciler(db)
        # Alpaca returns the matching symbols
        short_sym = reconciler.alpaca._build_occ_symbol("SPY", "2026-01-16", 545.0, "put")
        long_sym  = reconciler.alpaca._build_occ_symbol("SPY", "2026-01-16", 540.0, "put")
        reconciler.alpaca.get_positions.return_value = [
            {"symbol": short_sym, "qty": "-1", "asset_class": "us_option",
             "market_value": "-120", "avg_entry_price": "1.20", "current_price": "1.20",
             "unrealized_pl": "0", "side": "short"},
            {"symbol": long_sym,  "qty": "1",  "asset_class": "us_option",
             "market_value": "60",  "avg_entry_price": "0.60", "current_price": "0.60",
             "unrealized_pl": "0", "side": "long"},
        ]

        result = reconciler.reconcile()

        assert result.phantom_resolved == 0
        trades = get_trades(path=db)
        t = next(x for x in trades if x["id"] == "live-001")
        assert t["status"] == "open"

    def test_multiple_phantoms_all_resolved(self, tmp_path):
        from shared.database import init_db, upsert_trade
        db = str(tmp_path / "trades.db")
        init_db(db)

        for i in range(3):
            upsert_trade({
                "id": f"phantom-{i}",
                "ticker": "SPY",
                "strategy_type": "bull_put",
                "short_strike": 545.0 + i,
                "long_strike": 540.0 + i,
                "expiration": "2026-01-16",
                "credit": 1.50,
                "contracts": 1,
                "status": "open",
            }, source="execution", path=db)

        reconciler = self._make_reconciler(db)
        reconciler.alpaca.get_positions.return_value = []

        result = reconciler.reconcile()
        assert result.phantom_resolved == 3


# ===========================================================================
# Startup Reconciliation — Orphan Positions
# ===========================================================================

class TestStartupReconciliationOrphans:
    """Alpaca has option position not in DB → create unmanaged record."""

    def _make_reconciler(self, db):
        from shared.reconciler import PositionReconciler
        alpaca = MagicMock()
        alpaca._build_occ_symbol.side_effect = lambda t, e, s, o: f"{t}{e}{o}{int(s)}"
        return PositionReconciler(alpaca=alpaca, db_path=db)

    def test_orphan_option_position_creates_unmanaged_record(self, tmp_path):
        from shared.database import get_trades, init_db
        db = str(tmp_path / "trades.db")
        init_db(db)

        reconciler = self._make_reconciler(db)
        reconciler.alpaca.get_positions.return_value = [
            {
                "symbol": "QQQ260320P00450000",
                "qty": "-1",
                "asset_class": "us_option",
                "market_value": "-80",
                "avg_entry_price": "0.80",
                "current_price": "0.80",
                "unrealized_pl": "0",
                "side": "short",
            }
        ]

        result = reconciler.reconcile()

        assert result.orphans_detected == 1
        trades = get_trades(path=db)
        orphan = next((t for t in trades if "orphan" in t.get("id", "")), None)
        assert orphan is not None
        assert orphan["status"] == "unmanaged"

    def test_equity_positions_not_flagged_as_orphans(self, tmp_path):
        from shared.database import init_db
        db = str(tmp_path / "trades.db")
        init_db(db)

        reconciler = self._make_reconciler(db)
        reconciler.alpaca.get_positions.return_value = [
            {
                "symbol": "SPY",
                "qty": "100",
                "asset_class": "us_equity",
                "market_value": "54500",
                "avg_entry_price": "545",
                "current_price": "545",
                "unrealized_pl": "0",
                "side": "long",
            }
        ]

        result = reconciler.reconcile()
        assert result.orphans_detected == 0

    def test_managed_positions_not_flagged_as_orphans(self, tmp_path):
        """A position in DB and in Alpaca should NOT be flagged as orphan."""
        from shared.database import init_db, upsert_trade
        db = str(tmp_path / "trades.db")
        init_db(db)

        upsert_trade({
            "id": "managed-001",
            "ticker": "SPY",
            "strategy_type": "bull_put",
            "short_strike": 545.0,
            "long_strike": 540.0,
            "expiration": "2026-01-16",
            "credit": 1.50,
            "contracts": 1,
            "status": "open",
        }, source="execution", path=db)

        reconciler = self._make_reconciler(db)
        # Build the expected symbols
        short_sym = reconciler.alpaca._build_occ_symbol("SPY", "2026-01-16", 545.0, "put")
        long_sym  = reconciler.alpaca._build_occ_symbol("SPY", "2026-01-16", 540.0, "put")
        reconciler.alpaca.get_positions.return_value = [
            {"symbol": short_sym, "qty": "-1", "asset_class": "us_option",
             "market_value": "-120", "avg_entry_price": "1.20", "current_price": "1.20",
             "unrealized_pl": "0", "side": "short"},
            {"symbol": long_sym,  "qty": "1",  "asset_class": "us_option",
             "market_value": "60",  "avg_entry_price": "0.60", "current_price": "0.60",
             "unrealized_pl": "0", "side": "long"},
        ]

        result = reconciler.reconcile()
        assert result.orphans_detected == 0


# ===========================================================================
# Consecutive API Failure Alerting
# ===========================================================================

class TestConsecutiveAPIFailureAlerting:

    def test_first_failure_logs_error_not_critical(self, tmp_path, caplog):
        import logging
        monitor, _ = _make_monitor(tmp_path)
        monitor.alpaca.get_positions.side_effect = Exception("connection refused")

        with patch.object(monitor, "_is_market_hours", return_value=True), \
             patch.object(monitor, "_reconcile_pending_opens"), \
             patch.object(monitor, "_reconcile_pending_closes"), \
             caplog.at_level(logging.ERROR):
            monitor._check_positions()

        assert monitor._consecutive_api_failures == 1
        # No CRITICAL log on first failure
        assert not any(r.levelno == logging.CRITICAL for r in caplog.records)

    def test_third_consecutive_failure_logs_critical(self, tmp_path, caplog):
        import logging
        monitor, _ = _make_monitor(tmp_path)
        monitor.alpaca.get_positions.side_effect = Exception("unreachable")
        monitor._consecutive_api_failures = 2  # simulate 2 prior failures

        with patch.object(monitor, "_is_market_hours", return_value=True), \
             patch.object(monitor, "_reconcile_pending_opens"), \
             patch.object(monitor, "_reconcile_pending_closes"), \
             caplog.at_level(logging.CRITICAL):
            monitor._check_positions()

        assert monitor._consecutive_api_failures == 3
        assert any(r.levelno == logging.CRITICAL for r in caplog.records)
        assert any("unreachable" in r.message.lower() or "consecutive" in r.message.lower()
                   for r in caplog.records if r.levelno == logging.CRITICAL)

    def test_recovery_resets_failure_counter(self, tmp_path):
        monitor, _ = _make_monitor(tmp_path)
        monitor._consecutive_api_failures = 5  # had failures

        # Now succeeds
        monitor.alpaca.get_positions.return_value = []
        monitor.alpaca.get_positions.side_effect = None

        with patch.object(monitor, "_is_market_hours", return_value=True), \
             patch.object(monitor, "_reconcile_pending_opens"), \
             patch.object(monitor, "_reconcile_pending_closes"), \
             patch.object(monitor, "_detect_assignment"), \
             patch.object(monitor, "_detect_orphans"), \
             patch.object(monitor, "_reconcile_external_closes"):
            monitor._check_positions()

        assert monitor._consecutive_api_failures == 0

    def test_counter_increments_on_each_failure(self, tmp_path):
        monitor, _ = _make_monitor(tmp_path)
        monitor.alpaca.get_positions.side_effect = Exception("timeout")

        for i in range(1, 4):
            with patch.object(monitor, "_is_market_hours", return_value=True), \
                 patch.object(monitor, "_reconcile_pending_opens"), \
                 patch.object(monitor, "_reconcile_pending_closes"):
                monitor._check_positions()
            assert monitor._consecutive_api_failures == i


# ===========================================================================
# Wide Bid-Ask: All Closes Use Market Orders
# ===========================================================================

class TestCloseOrdersUseMarketOrders:
    """All close submissions must pass limit_price=None (market order)
    to guarantee fills even with wide bid-ask spreads."""

    def test_close_spread_called_with_market_order(self, tmp_path):
        from shared.database import upsert_trade
        monitor, db = _make_monitor(tmp_path)
        pos = {
            "id": "close-mkt-001",
            "ticker": "SPY",
            "strategy_type": "bull_put",
            "short_strike": 545.0,
            "long_strike": 540.0,
            "expiration": "2026-12-18",
            "credit": 1.50,
            "contracts": 1,
            "status": "open",
        }
        upsert_trade(pos, source="execution", path=db)

        monitor.alpaca.close_spread.return_value = {
            "status": "submitted",
            "order_id": "ord-mkt-001",
        }

        monitor._close_position(dict(pos), reason="stop_loss")

        call_kwargs = monitor.alpaca.close_spread.call_args
        # limit_price must be None → market order
        limit = call_kwargs.kwargs.get("limit_price", call_kwargs.args[5] if len(call_kwargs.args) > 5 else None)
        assert limit is None, "Close orders must use market orders (limit_price=None)"

    def test_ic_close_called_with_market_order(self, tmp_path):
        from shared.database import upsert_trade
        monitor, db = _make_monitor(tmp_path)
        pos = {
            "id": "close-ic-mkt",
            "ticker": "SPY",
            "strategy_type": "iron_condor",
            "put_short_strike": 530.0,
            "put_long_strike": 525.0,
            "call_short_strike": 560.0,
            "call_long_strike": 565.0,
            "short_strike": 530.0,
            "long_strike": 525.0,
            "expiration": "2026-12-18",
            "credit": 2.00,
            "contracts": 1,
            "status": "open",
        }
        upsert_trade(pos, source="execution", path=db)

        monitor.alpaca.close_iron_condor.return_value = {
            "status": "submitted",
            "order_id": "ord-ic-mkt",
        }

        monitor._close_position(dict(pos), reason="dte_management")

        call_kwargs = monitor.alpaca.close_iron_condor.call_args
        limit = call_kwargs.kwargs.get("limit_price")
        assert limit is None, "IC close orders must use market orders (limit_price=None)"
