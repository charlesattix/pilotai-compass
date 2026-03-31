"""Tests for shared.feature_logger (ML-1)."""

import os
import sqlite3
import tempfile

import pytest

from shared.feature_logger import FeatureLogger, _extract_features_from_opportunity


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


class TestFeatureLogger:
    def test_create_table(self, db_path):
        fl = FeatureLogger(db_path=db_path)
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "trade_features" in tables

    def test_log_entry(self, db_path):
        fl = FeatureLogger(db_path=db_path)
        features = {
            "timestamp": "2025-01-15T10:00:00",
            "ticker": "SPY",
            "strategy_type": "bull_put",
            "direction": "bullish",
            "regime": "bull",
            "vix": 18.5,
            "vix_rank": 35.0,
            "vix_percentile": 40.0,
            "iv_rank": 45.0,
            "rsi": 55.0,
            "ma200_distance": 5.2,
            "dte": 30,
            "otm_pct": 3.5,
            "spread_width": 5.0,
            "credit_received": 0.85,
            "max_loss": 415.0,
            "realized_vol_20d": 0.15,
            "realized_vol_5d": 0.12,
            "vix_vix3m_ratio": 0.92,
            "vol_premium_zscore": 1.2,
            "score": 72.5,
        }
        fl.log_entry("trade-001", features)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trade_features WHERE trade_id = ?", ("trade-001",)).fetchone()
        conn.close()

        assert row is not None
        assert row["ticker"] == "SPY"
        assert row["vix"] == 18.5
        assert row["score"] == 72.5
        assert row["outcome"] is None  # not yet filled

    def test_log_outcome(self, db_path):
        fl = FeatureLogger(db_path=db_path)
        fl.log_entry("trade-002", {"timestamp": "2025-01-15T10:00:00", "ticker": "QQQ"})
        fl.log_outcome("trade-002", "win", 18.5, 7.3)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM trade_features WHERE trade_id = ?", ("trade-002",)).fetchone()
        conn.close()

        assert row["outcome"] == "win"
        assert row["pnl_pct"] == 18.5
        assert row["hold_days"] == 7.3

    def test_get_stats_empty(self, db_path):
        fl = FeatureLogger(db_path=db_path)
        stats = fl.get_stats()
        assert stats["total_features"] == 0
        assert stats["class_balance"] == {}

    def test_get_stats_with_data(self, db_path):
        fl = FeatureLogger(db_path=db_path)
        fl.log_entry("t1", {"timestamp": "2025-01-10", "ticker": "SPY"})
        fl.log_entry("t2", {"timestamp": "2025-01-15", "ticker": "QQQ"})
        fl.log_entry("t3", {"timestamp": "2025-01-20", "ticker": "IWM"})
        fl.log_outcome("t1", "win", 10.0, 5.0)
        fl.log_outcome("t2", "loss", -15.0, 3.0)

        stats = fl.get_stats()
        assert stats["total_features"] == 3
        assert stats["class_balance"]["win"] == 1
        assert stats["class_balance"]["loss"] == 1
        assert stats["class_balance"]["pending"] == 1

    def test_log_entry_does_not_raise(self, db_path):
        """Feature logging must never crash the scanner."""
        fl = FeatureLogger(db_path="/nonexistent/path/db.sqlite")
        # Should not raise — just logs a warning
        fl.log_entry("trade-bad", {"ticker": "SPY"})

    def test_log_outcome_does_not_raise(self, db_path):
        fl = FeatureLogger(db_path="/nonexistent/path/db.sqlite")
        fl.log_outcome("trade-bad", "win", 10.0, 5.0)

    def test_upsert_on_duplicate(self, db_path):
        fl = FeatureLogger(db_path=db_path)
        fl.log_entry("t1", {"timestamp": "2025-01-10", "ticker": "SPY", "score": 50.0})
        fl.log_entry("t1", {"timestamp": "2025-01-10", "ticker": "SPY", "score": 75.0})

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM trade_features WHERE trade_id = ?", ("t1",)).fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0]["score"] == 75.0


class TestExtractFeatures:
    def test_basic_extraction(self):
        opp = {
            "ticker": "SPY",
            "type": "bull_put_spread",
            "short_strike": 480,
            "current_price": 500,
            "credit": 0.85,
            "spread_width": 5,
            "max_loss": 415,
            "dte": 30,
            "score": 72.5,
            "_ml_features": {
                "current_price": 500,
                "regime": "bull",
                "vix": 18.5,
                "iv_rank": 45.0,
                "rsi": 55.0,
            },
        }
        features = _extract_features_from_opportunity(opp)
        assert features["ticker"] == "SPY"
        assert features["direction"] == "bullish"
        assert features["regime"] == "bull"
        assert features["vix"] == 18.5
        assert features["otm_pct"] == pytest.approx(4.0, abs=0.01)
        assert features["score"] == 72.5

    def test_missing_context(self):
        opp = {"ticker": "AAPL", "type": "bear_call_spread", "score": 60}
        features = _extract_features_from_opportunity(opp)
        assert features["ticker"] == "AAPL"
        assert features["direction"] == "bearish"
        assert features["vix"] is None
