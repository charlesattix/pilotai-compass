"""
Tests for the 5 reconciler fixes:

Fix 1: IC client_order_id suffix mismatch — reconciler now looks up wing IDs
Fix 2: Batch fetch limit 100 → 500
Fix 3: Belt-and-suspenders wing suffix search before age-based failed_open
Fix 4: Recovery path — failed_open with live Alpaca position → promoted to open
Fix 5: Periodic orphan detection from reconcile_pending_only() (30-min throttle)
"""

import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from shared.database import get_trades, init_db, upsert_trade
from shared.reconciler import PositionReconciler, _ORPHAN_CHECK_INTERVAL_MINUTES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _db(tmp_path):
    path = str(tmp_path / "test.db")
    init_db(path)
    return path


def _pending_trade(db_path, trade_id="cs-abc123", spread_type="bull_put", dry_run=False):
    """Insert a pending_open trade and return it."""
    t = {
        "id": trade_id,
        "ticker": "SPY",
        "strategy_type": spread_type,
        "status": "pending_open",
        "short_strike": 540.0,
        "long_strike": 535.0,
        "expiration": "2026-04-18",
        "credit": 1.50,
        "contracts": 1,
        "entry_date": datetime.now(timezone.utc).isoformat(),
        "alpaca_client_order_id": trade_id,
    }
    if dry_run:
        t["dry_run"] = True
    upsert_trade(t, source="execution", path=db_path)
    return t


def _pending_ic_trade(db_path, trade_id="cs-ic001"):
    """Insert a pending_open iron condor trade."""
    t = {
        "id": trade_id,
        "ticker": "SPY",
        "strategy_type": "iron_condor",
        "status": "pending_open",
        "short_strike": 540.0,
        "long_strike": 535.0,
        "put_short_strike": 540.0,
        "put_long_strike": 535.0,
        "call_short_strike": 570.0,
        "call_long_strike": 575.0,
        "expiration": "2026-04-18",
        "credit": 3.00,
        "contracts": 1,
        "entry_date": datetime.now(timezone.utc).isoformat(),
        "alpaca_client_order_id": trade_id,
        "alpaca_put_order_id": trade_id + "-put",
        "alpaca_call_order_id": trade_id + "-call",
    }
    upsert_trade(t, source="execution", path=db_path)
    return t


def _mock_alpaca(batch_orders=None):
    alpaca = MagicMock()
    alpaca.get_orders.return_value = batch_orders or []
    alpaca.get_order_by_client_id.return_value = None
    alpaca.get_positions.return_value = []
    return alpaca


# ---------------------------------------------------------------------------
# Fix 1: IC wing IDs stored in DB by execution_engine
# ---------------------------------------------------------------------------

class TestFix1ICWingIDsStoredInDB:
    def test_wing_ids_stored_after_successful_ic_submission(self, tmp_path):
        """ExecutionEngine stores alpaca_put_order_id / alpaca_call_order_id on IC submit."""
        from execution.execution_engine import ExecutionEngine

        mock_alpaca = MagicMock()
        mock_alpaca.get_market_clock.return_value = {"is_open": True}
        mock_alpaca.submit_credit_spread.return_value = {
            "status": "submitted", "order_id": "ord-put-1",
        }
        db_path = str(tmp_path / "test.db")
        engine = ExecutionEngine(alpaca_provider=mock_alpaca, db_path=db_path)

        opp = {
            "ticker": "SPY", "type": "iron_condor", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 3.00, "contracts": 1,
            "put_short_strike": 540.0, "put_long_strike": 535.0,
            "call_short_strike": 570.0, "call_long_strike": 575.0,
        }
        result = engine.submit_opportunity(opp)
        assert result["status"] == "submitted"

        from shared.database import get_trade_by_id
        trade = get_trade_by_id(result["client_order_id"], path=db_path)
        assert trade is not None
        cid = result["client_order_id"]
        assert trade.get("alpaca_put_order_id") == cid + "-put"
        assert trade.get("alpaca_call_order_id") == cid + "-call"

    def test_wing_ids_not_stored_for_non_ic(self, tmp_path):
        """Wing IDs should NOT be added for regular bull_put spreads."""
        from execution.execution_engine import ExecutionEngine

        mock_alpaca = MagicMock()
        mock_alpaca.get_market_clock.return_value = {"is_open": True}
        mock_alpaca.submit_credit_spread.return_value = {
            "status": "submitted", "order_id": "ord-1",
        }
        db_path = str(tmp_path / "test.db")
        engine = ExecutionEngine(alpaca_provider=mock_alpaca, db_path=db_path)

        opp = {
            "ticker": "SPY", "type": "bull_put", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 1.50, "contracts": 1,
        }
        result = engine.submit_opportunity(opp)
        assert result["status"] == "submitted"

        from shared.database import get_trade_by_id
        trade = get_trade_by_id(result["client_order_id"], path=db_path)
        assert trade.get("alpaca_put_order_id") is None
        assert trade.get("alpaca_call_order_id") is None

    def test_reconciler_promotes_ic_when_both_wings_filled(self, tmp_path):
        """Reconciler promotes IC pending_open → open when both wings filled."""
        db_path = _db(tmp_path)
        trade_id = "cs-ic001"
        _pending_ic_trade(db_path, trade_id)

        put_order = {"client_order_id": trade_id + "-put", "status": "filled",
                     "filled_avg_price": "1.50", "id": "put-ord-1"}
        call_order = {"client_order_id": trade_id + "-call", "status": "filled",
                      "filled_avg_price": "1.50", "id": "call-ord-1"}

        alpaca = _mock_alpaca(batch_orders=[put_order, call_order])
        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile_pending_only()

        assert result.pending_resolved == 1
        assert result.pending_failed == 0
        trade = get_trades(status="open", path=db_path)
        assert len(trade) == 1
        assert trade[0]["alpaca_fill_price"] == pytest.approx(3.0)

    def test_reconciler_marks_ic_failed_when_put_wing_cancelled(self, tmp_path):
        """IC is marked failed_open if put wing is in terminal state."""
        db_path = _db(tmp_path)
        trade_id = "cs-ic002"
        _pending_ic_trade(db_path, trade_id)

        put_order = {"client_order_id": trade_id + "-put", "status": "cancelled", "id": "p-ord"}
        call_order = {"client_order_id": trade_id + "-call", "status": "filled",
                      "filled_avg_price": "1.50", "id": "c-ord"}

        alpaca = _mock_alpaca(batch_orders=[put_order, call_order])
        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile_pending_only()

        assert result.pending_failed == 1
        assert result.pending_resolved == 0
        trades = get_trades(status="failed_open", path=db_path)
        assert len(trades) == 1

    def test_reconciler_ic_leaves_pending_when_wings_in_flight(self, tmp_path):
        """IC stays pending_open when wings are submitted but not yet filled."""
        db_path = _db(tmp_path)
        trade_id = "cs-ic003"
        _pending_ic_trade(db_path, trade_id)

        put_order = {"client_order_id": trade_id + "-put", "status": "submitted", "id": "p-ord"}
        call_order = {"client_order_id": trade_id + "-call", "status": "submitted", "id": "c-ord"}

        alpaca = _mock_alpaca(batch_orders=[put_order, call_order])
        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile_pending_only()

        assert result.pending_resolved == 0
        assert result.pending_failed == 0
        trades = get_trades(status="pending_open", path=db_path)
        assert len(trades) == 1

    def test_reconciler_ic_uses_derived_suffix_as_fallback(self, tmp_path):
        """Without stored wing IDs, reconciler derives cid+'-put'/'-call' as fallback."""
        db_path = _db(tmp_path)
        trade_id = "cs-ic004"
        # IC trade WITHOUT alpaca_put/call_order_id stored (old-style record)
        t = {
            "id": trade_id, "ticker": "SPY", "strategy_type": "iron_condor",
            "status": "pending_open", "short_strike": 540.0, "long_strike": 535.0,
            "put_short_strike": 540.0, "put_long_strike": 535.0,
            "call_short_strike": 570.0, "call_long_strike": 575.0,
            "expiration": "2026-04-18", "credit": 3.00, "contracts": 1,
            "entry_date": datetime.now(timezone.utc).isoformat(),
            "alpaca_client_order_id": trade_id,
            # No alpaca_put_order_id / alpaca_call_order_id
        }
        upsert_trade(t, source="execution", path=db_path)

        put_order = {"client_order_id": trade_id + "-put", "status": "filled",
                     "filled_avg_price": "1.50", "id": "p-ord"}
        call_order = {"client_order_id": trade_id + "-call", "status": "filled",
                      "filled_avg_price": "1.50", "id": "c-ord"}

        alpaca = _mock_alpaca(batch_orders=[put_order, call_order])
        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile_pending_only()

        assert result.pending_resolved == 1
        trades = get_trades(status="open", path=db_path)
        assert len(trades) == 1


# ---------------------------------------------------------------------------
# Fix 2: Batch fetch limit 500
# ---------------------------------------------------------------------------

class TestFix2BatchLimit500:
    def test_batch_fetch_uses_limit_500(self, tmp_path):
        """_fetch_recent_orders_by_client_id should request limit=500."""
        alpaca = _mock_alpaca()
        reconciler = PositionReconciler(alpaca=alpaca, db_path=_db(tmp_path))
        reconciler._fetch_recent_orders_by_client_id()
        alpaca.get_orders.assert_called_once_with(status="all", limit=500)


# ---------------------------------------------------------------------------
# Fix 3: Belt-and-suspenders wing suffix search for non-IC trades
# ---------------------------------------------------------------------------

class TestFix3WingSuffixFallback:
    def test_regular_spread_not_marked_failed_when_wing_suffix_found(self, tmp_path):
        """A regular spread trade (bull_put) is left pending if wing suffix found in batch.

        This protects against a future data corruption scenario where a non-IC trade
        ends up with suffixed orders.  More importantly it prevents false failures for
        legacy IC records whose strategy_type might not include 'condor'.
        """
        db_path = _db(tmp_path)
        trade_id = "cs-bp001"
        # Very old pending trade (age > 4h) — would normally be marked failed_open
        old_entry = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        t = {
            "id": trade_id, "ticker": "SPY", "strategy_type": "bull_put",
            "status": "pending_open", "short_strike": 540.0, "long_strike": 535.0,
            "expiration": "2026-04-18", "credit": 1.50, "contracts": 1,
            "entry_date": old_entry,
            "alpaca_client_order_id": trade_id,
        }
        upsert_trade(t, source="execution", path=db_path)

        # Wing-suffixed order found in batch — prevents age-based failure
        wing_order = {"client_order_id": trade_id + "-put", "status": "submitted"}
        alpaca = _mock_alpaca(batch_orders=[wing_order])
        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile_pending_only()

        assert result.pending_failed == 0
        # Trade must still be pending_open
        trades = get_trades(status="pending_open", path=db_path)
        assert len(trades) == 1

    def test_regular_spread_marked_failed_when_old_and_no_wing_found(self, tmp_path):
        """A regular spread trade > 4h old with no orders is still marked failed_open."""
        db_path = _db(tmp_path)
        trade_id = "cs-bp002"
        old_entry = (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat()
        t = {
            "id": trade_id, "ticker": "SPY", "strategy_type": "bull_put",
            "status": "pending_open", "short_strike": 540.0, "long_strike": 535.0,
            "expiration": "2026-04-18", "credit": 1.50, "contracts": 1,
            "entry_date": old_entry, "alpaca_client_order_id": trade_id,
        }
        upsert_trade(t, source="execution", path=db_path)

        # Empty batch + per-trade lookup returns None
        alpaca = _mock_alpaca(batch_orders=[])
        alpaca.get_order_by_client_id.return_value = None
        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile_pending_only()

        assert result.pending_failed == 1
        trades = get_trades(status="failed_open", path=db_path)
        assert len(trades) == 1


# ---------------------------------------------------------------------------
# Fix 4: Recovery — failed_open promoted to open when matching Alpaca position found
# ---------------------------------------------------------------------------

class TestFix4FailedOpenRecovery:
    def _build_occ_symbol(self, ticker, exp, strike, opt_type):
        """Replicate the OCC symbol format for testing."""
        dt = exp.replace("-", "")
        strike_int = round(strike * 1000)
        return f"{ticker:<6}{dt}{opt_type[0].upper()}{strike_int:08d}"

    def test_failed_open_promoted_to_open_when_position_found(self, tmp_path):
        """A failed_open IC trade is recovered when its legs are still live in Alpaca."""
        db_path = _db(tmp_path)
        trade_id = "cs-failed-ic"

        # Insert a failed_open IC trade (simulates the Bug 1 outcome)
        t = {
            "id": trade_id, "ticker": "SPY", "strategy_type": "iron_condor",
            "status": "failed_open", "exit_reason": "ic_wings_not_found_in_alpaca",
            "short_strike": 540.0, "long_strike": 535.0,
            "put_short_strike": 540.0, "put_long_strike": 535.0,
            "call_short_strike": 570.0, "call_long_strike": 575.0,
            "expiration": "2026-04-18", "credit": 3.00, "contracts": 1,
            "entry_date": datetime.now(timezone.utc).isoformat(),
            "alpaca_client_order_id": trade_id,
        }
        upsert_trade(t, source="execution", path=db_path)

        # Mock Alpaca: one put leg of the IC is visible as a live position
        put_sym = self._build_occ_symbol("SPY", "2026-04-18", 540.0, "put")
        alpaca = MagicMock()
        alpaca.get_positions.return_value = [
            {"symbol": put_sym, "asset_class": "us_option", "qty": "-1",
             "market_value": "-50.0"},
        ]
        alpaca.get_orders.return_value = []
        alpaca.get_order_by_client_id.return_value = None
        alpaca._build_occ_symbol.side_effect = lambda t, e, s, o: (
            self._build_occ_symbol(t, e, s, o)
        )

        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile()

        recovered = get_trades(status="open", path=db_path)
        assert len(recovered) == 1, "failed_open trade should be promoted to open"
        assert result.pending_resolved == 1

    def test_no_recovery_when_no_matching_failed_open(self, tmp_path):
        """An Alpaca position with no matching failed_open trade creates unmanaged record."""
        db_path = _db(tmp_path)
        put_sym = self._build_occ_symbol("SPY", "2026-04-18", 540.0, "put")
        alpaca = MagicMock()
        alpaca.get_positions.return_value = [
            {"symbol": put_sym, "asset_class": "us_option", "qty": "-1",
             "market_value": "-50.0"},
        ]
        alpaca.get_orders.return_value = []
        alpaca.get_order_by_client_id.return_value = None
        alpaca._build_occ_symbol.side_effect = lambda t, e, s, o: (
            self._build_occ_symbol(t, e, s, o)
        )

        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile()

        unmanaged = get_trades(status="unmanaged", path=db_path)
        assert len(unmanaged) == 1
        assert result.orphans_detected == 1


# ---------------------------------------------------------------------------
# Fix 5: Periodic orphan detection from reconcile_pending_only()
# ---------------------------------------------------------------------------

class TestFix5PeriodicOrphanDetection:
    def test_orphan_detection_runs_when_no_prior_check(self, tmp_path):
        """First call to reconcile_pending_only() always runs orphan detection."""
        db_path = _db(tmp_path)
        put_sym = "SPY   260418P00540000"
        alpaca = MagicMock()
        alpaca.get_positions.return_value = [
            {"symbol": put_sym, "asset_class": "us_option", "qty": "-1",
             "market_value": "-50.0"},
        ]
        alpaca.get_orders.return_value = []
        alpaca.get_order_by_client_id.return_value = None
        alpaca._build_occ_symbol.side_effect = lambda t, e, s, o: put_sym

        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile_pending_only()

        # Orphan detection should have run and created an unmanaged record
        unmanaged = get_trades(status="unmanaged", path=db_path)
        assert len(unmanaged) == 1
        assert result.orphans_detected == 1

    def test_orphan_detection_skipped_when_recently_run(self, tmp_path):
        """reconcile_pending_only() skips orphan detection within the 30-min window."""
        from shared.database import save_scanner_state
        db_path = _db(tmp_path)

        # Simulate a recent orphan check (1 minute ago)
        recent = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        save_scanner_state("last_orphan_check", recent, path=db_path)

        put_sym = "SPY   260418P00540000"
        alpaca = MagicMock()
        alpaca.get_positions.return_value = [
            {"symbol": put_sym, "asset_class": "us_option", "qty": "-1",
             "market_value": "-50.0"},
        ]
        alpaca.get_orders.return_value = []
        alpaca.get_order_by_client_id.return_value = None

        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile_pending_only()

        # Orphan detection should have been skipped
        assert result.orphans_detected == 0
        unmanaged = get_trades(status="unmanaged", path=db_path)
        assert len(unmanaged) == 0

    def test_orphan_detection_runs_when_interval_elapsed(self, tmp_path):
        """reconcile_pending_only() runs orphan detection when 30+ min have elapsed."""
        from shared.database import save_scanner_state
        db_path = _db(tmp_path)

        # Simulate last check 31 minutes ago — interval exceeded
        old_check = (
            datetime.now(timezone.utc) - timedelta(minutes=_ORPHAN_CHECK_INTERVAL_MINUTES + 1)
        ).isoformat()
        save_scanner_state("last_orphan_check", old_check, path=db_path)

        put_sym = "SPY   260418P00540000"
        alpaca = MagicMock()
        alpaca.get_positions.return_value = [
            {"symbol": put_sym, "asset_class": "us_option", "qty": "-1",
             "market_value": "-50.0"},
        ]
        alpaca.get_orders.return_value = []
        alpaca.get_order_by_client_id.return_value = None
        alpaca._build_occ_symbol.side_effect = lambda t, e, s, o: put_sym

        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        result = reconciler.reconcile_pending_only()

        assert result.orphans_detected == 1
        unmanaged = get_trades(status="unmanaged", path=db_path)
        assert len(unmanaged) == 1

    def test_last_orphan_check_timestamp_persisted_after_run(self, tmp_path):
        """After orphan detection runs, the timestamp is saved to scanner_state."""
        from shared.database import load_scanner_state
        db_path = _db(tmp_path)

        alpaca = _mock_alpaca()
        reconciler = PositionReconciler(alpaca=alpaca, db_path=db_path)
        reconciler.reconcile_pending_only()

        saved = load_scanner_state("last_orphan_check", path=db_path)
        assert saved is not None
        # Should parse as a valid ISO datetime
        dt = datetime.fromisoformat(saved)
        assert (datetime.now(timezone.utc) - dt.replace(tzinfo=timezone.utc)).total_seconds() < 10

    def test_should_run_orphan_check_returns_true_when_no_saved_state(self, tmp_path):
        """_should_run_orphan_check() returns True when no saved timestamp exists."""
        db_path = _db(tmp_path)
        reconciler = PositionReconciler(alpaca=_mock_alpaca(), db_path=db_path)
        assert reconciler._should_run_orphan_check() is True

    def test_should_run_orphan_check_returns_false_when_recent(self, tmp_path):
        """_should_run_orphan_check() returns False within the throttle window."""
        from shared.database import save_scanner_state
        db_path = _db(tmp_path)
        recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        save_scanner_state("last_orphan_check", recent, path=db_path)
        reconciler = PositionReconciler(alpaca=_mock_alpaca(), db_path=db_path)
        assert reconciler._should_run_orphan_check() is False

    def test_should_run_orphan_check_returns_true_when_stale(self, tmp_path):
        """_should_run_orphan_check() returns True after the throttle window expires."""
        from shared.database import save_scanner_state
        db_path = _db(tmp_path)
        stale = (
            datetime.now(timezone.utc) - timedelta(minutes=_ORPHAN_CHECK_INTERVAL_MINUTES + 5)
        ).isoformat()
        save_scanner_state("last_orphan_check", stale, path=db_path)
        reconciler = PositionReconciler(alpaca=_mock_alpaca(), db_path=db_path)
        assert reconciler._should_run_orphan_check() is True


# ---------------------------------------------------------------------------
# Integration: IC trade full lifecycle with all fixes
# ---------------------------------------------------------------------------

class TestICLifecycleIntegration:
    def test_ic_lifecycle_submit_fill_reconcile(self, tmp_path):
        """End-to-end: IC submitted → wing IDs stored → reconciler promotes to open."""
        from execution.execution_engine import ExecutionEngine

        db_path = str(tmp_path / "test.db")

        # Step 1: Submit IC via ExecutionEngine
        mock_alpaca = MagicMock()
        mock_alpaca.get_market_clock.return_value = {"is_open": True}
        mock_alpaca.submit_credit_spread.return_value = {
            "status": "submitted", "order_id": "alpaca-ord-put",
        }
        engine = ExecutionEngine(alpaca_provider=mock_alpaca, db_path=db_path)
        opp = {
            "ticker": "SPY", "type": "iron_condor", "expiration": "2026-04-18",
            "short_strike": 540.0, "long_strike": 535.0, "credit": 3.00, "contracts": 1,
            "put_short_strike": 540.0, "put_long_strike": 535.0,
            "call_short_strike": 570.0, "call_long_strike": 575.0,
        }
        result = engine.submit_opportunity(opp)
        cid = result["client_order_id"]
        assert result["status"] == "submitted"

        # Step 2: Verify wing IDs stored in DB
        from shared.database import get_trade_by_id
        trade = get_trade_by_id(cid, path=db_path)
        assert trade["alpaca_put_order_id"] == cid + "-put"
        assert trade["alpaca_call_order_id"] == cid + "-call"

        # Step 3: Reconcile — both wings filled
        put_order = {"client_order_id": cid + "-put", "status": "filled",
                     "filled_avg_price": "1.50", "id": "po-1"}
        call_order = {"client_order_id": cid + "-call", "status": "filled",
                      "filled_avg_price": "1.50", "id": "co-1"}
        alpaca_rec = _mock_alpaca(batch_orders=[put_order, call_order])
        reconciler = PositionReconciler(alpaca=alpaca_rec, db_path=db_path)
        rec_result = reconciler.reconcile_pending_only()

        assert rec_result.pending_resolved == 1
        open_trades = get_trades(status="open", path=db_path)
        assert len(open_trades) == 1
        assert open_trades[0]["id"] == cid
        assert open_trades[0]["alpaca_fill_price"] == pytest.approx(3.0)
