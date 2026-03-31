"""Tests for shared.database — uses real SQLite via tmp_path, no mocks."""

import time

from shared.database import (
    close_trade,
    get_latest_alerts,
    get_trades,
    init_db,
    insert_alert,
    insert_regime_snapshot,
    upsert_trade,
)


def _make_trade(**overrides):
    """Return a sample trade dict, with optional overrides."""
    base = {
        "id": "test-1",
        "ticker": "SPY",
        "type": "bull_put_spread",
        "status": "open",
        "short_strike": 450,
        "long_strike": 445,
        "expiration": "2025-06-20",
        "credit": 1.50,
        "contracts": 2,
        "entry_date": "2025-01-15T10:00:00",
        "custom_field": "extra",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestInitDb
# ---------------------------------------------------------------------------


class TestInitDb:
    """Database initialisation creates tables and is idempotent."""

    def test_creates_tables(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        # Verify all three tables exist by querying sqlite_master
        import sqlite3

        conn = sqlite3.connect(db_path)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()

        assert "trades" in tables
        assert "alerts" in tables
        assert "regime_snapshots" in tables

    def test_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)
        init_db(path=db_path)  # second call should not raise


# ---------------------------------------------------------------------------
# TestTrades
# ---------------------------------------------------------------------------


class TestTrades:
    """Trade upsert, retrieval, filtering, closing, and metadata handling."""

    def test_upsert_and_get_round_trip(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        trade = _make_trade()
        upsert_trade(trade, source="scanner", path=db_path)

        rows = get_trades(path=db_path)
        assert len(rows) == 1

        t = rows[0]
        assert t["id"] == "test-1"
        assert t["ticker"] == "SPY"
        assert t["strategy_type"] == "bull_put_spread"
        assert t["status"] == "open"
        assert t["short_strike"] == 450
        assert t["long_strike"] == 445
        assert t["expiration"] == "2025-06-20"
        assert t["credit"] == 1.50
        assert t["contracts"] == 2
        assert t["entry_date"] == "2025-01-15T10:00:00"

    def test_update_existing_trade(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        upsert_trade(_make_trade(), source="scanner", path=db_path)
        # Update with a new status
        upsert_trade(
            _make_trade(status="pending_close"), source="scanner", path=db_path
        )

        rows = get_trades(path=db_path)
        assert len(rows) == 1
        assert rows[0]["status"] == "pending_close"

    def test_filter_by_status(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        upsert_trade(_make_trade(id="t-open", status="open"), source="scanner", path=db_path)
        upsert_trade(
            _make_trade(id="t-closed", status="closed_profit"),
            source="scanner",
            path=db_path,
        )

        open_trades = get_trades(status="open", path=db_path)
        assert len(open_trades) == 1
        assert open_trades[0]["id"] == "t-open"

        closed_trades = get_trades(status="closed_profit", path=db_path)
        assert len(closed_trades) == 1
        assert closed_trades[0]["id"] == "t-closed"

    def test_filter_by_source(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        upsert_trade(_make_trade(id="t1"), source="scanner", path=db_path)
        upsert_trade(_make_trade(id="t2"), source="manual", path=db_path)

        scanner_trades = get_trades(source="scanner", path=db_path)
        assert len(scanner_trades) == 1
        assert scanner_trades[0]["id"] == "t1"

    def test_close_trade_profit(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        upsert_trade(_make_trade(id="t-profit"), source="scanner", path=db_path)
        close_trade("t-profit", pnl=120.0, reason="target_hit", path=db_path)

        rows = get_trades(path=db_path)
        t = rows[0]
        assert t["status"] == "closed_profit"
        assert t["pnl"] == 120.0
        assert t["exit_reason"] == "target_hit"
        assert t["exit_date"] is not None

    def test_close_trade_loss(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        upsert_trade(_make_trade(id="t-loss"), source="scanner", path=db_path)
        close_trade("t-loss", pnl=-80.0, reason="stop_loss", path=db_path)

        rows = get_trades(status="closed_loss", path=db_path)
        assert len(rows) == 1
        assert rows[0]["status"] == "closed_loss"
        assert rows[0]["pnl"] == -80.0

    def test_close_trade_expiry(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        upsert_trade(_make_trade(id="t-exp"), source="scanner", path=db_path)
        close_trade("t-exp", pnl=0.0, reason="expired", path=db_path)

        rows = get_trades(status="closed_expiry", path=db_path)
        assert len(rows) == 1
        assert rows[0]["status"] == "closed_expiry"

    def test_close_trade_manual(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        upsert_trade(_make_trade(id="t-man"), source="scanner", path=db_path)
        close_trade("t-man", pnl=50.0, reason="manual", path=db_path)

        rows = get_trades(status="closed_manual", path=db_path)
        assert len(rows) == 1
        assert rows[0]["status"] == "closed_manual"

    def test_metadata_round_trip(self, tmp_path):
        """Extra keys not in the schema columns are stored as metadata JSON
        and merged back into the returned dict."""
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        trade = _make_trade(custom_field="extra", notes="my note", score=0.87)
        upsert_trade(trade, source="scanner", path=db_path)

        rows = get_trades(path=db_path)
        t = rows[0]
        assert t["custom_field"] == "extra"
        assert t["notes"] == "my note"
        assert t["score"] == 0.87
        # metadata column itself should NOT appear in the flattened dict
        assert "metadata" not in t


# ---------------------------------------------------------------------------
# TestAlerts
# ---------------------------------------------------------------------------


class TestAlerts:
    """Alert insertion and retrieval."""

    def test_insert_and_get_round_trip(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        alert = {"ticker": "AAPL", "signal": "bullish", "strength": 0.9}
        insert_alert(alert, path=db_path)

        rows = get_latest_alerts(path=db_path)
        assert len(rows) == 1
        assert rows[0]["ticker"] == "AAPL"
        assert rows[0]["signal"] == "bullish"
        assert rows[0]["strength"] == 0.9
        assert "id" in rows[0]
        assert "created_at" in rows[0]

    def test_limit_parameter(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        # Insert three alerts with different tickers so IDs don't collide
        for ticker in ("AAPL", "MSFT", "GOOG"):
            insert_alert({"ticker": ticker, "signal": "neutral"}, path=db_path)
            # Tiny sleep so the timestamp-based id is unique
            time.sleep(0.01)

        all_alerts = get_latest_alerts(limit=50, path=db_path)
        assert len(all_alerts) == 3

        limited = get_latest_alerts(limit=2, path=db_path)
        assert len(limited) == 2


# ---------------------------------------------------------------------------
# TestRegimeSnapshots
# ---------------------------------------------------------------------------


class TestRegimeSnapshots:
    """Regime snapshot insertion."""

    def test_insert_without_error(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        insert_regime_snapshot(
            regime="bull",
            confidence=0.85,
            features={"vix": 15.2, "ma_slope": 0.03},
            path=db_path,
        )

        # Verify row exists by direct query
        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM regime_snapshots").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["regime"] == "bull"
        assert rows[0]["confidence"] == 0.85

    def test_insert_without_features(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(path=db_path)

        insert_regime_snapshot(
            regime="bear",
            confidence=0.70,
            path=db_path,
        )

        import sqlite3

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM regime_snapshots").fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0]["regime"] == "bear"
