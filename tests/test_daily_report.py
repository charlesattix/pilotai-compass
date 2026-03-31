"""Tests for scripts/daily_report.py (INF-4)."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.daily_report import (
    _compute_stats,
    _hold_days,
    _dte_remaining,
    _pnl_class,
    _pnl_str,
    _pct_str,
    collect_report_data,
    generate_html,
    get_daily_summary_metrics,
    load_config,
    load_env_file,
    send_html_report_telegram,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite DB with the trades schema and sample data."""
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE trades (
        id TEXT PRIMARY KEY, source TEXT NOT NULL, ticker TEXT NOT NULL,
        strategy_type TEXT, status TEXT DEFAULT 'open',
        short_strike REAL, long_strike REAL, expiration TEXT,
        credit REAL, contracts INTEGER DEFAULT 1,
        entry_date TEXT, exit_date TEXT, exit_reason TEXT,
        pnl REAL, metadata JSON,
        created_at TEXT DEFAULT (datetime('now')),
        updated_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE TABLE alerts (
        id TEXT PRIMARY KEY, ticker TEXT NOT NULL,
        data JSON NOT NULL, created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE TABLE regime_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT, regime TEXT,
        confidence REAL, features JSON,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE TABLE scanner_state (
        key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL,
        updated_at TEXT DEFAULT (datetime('now'))
    )""")

    # Sample trades
    trades = [
        ("t1", "scanner", "SPY", "credit_spread", "open", 580, 568, "2026-04-01",
         1.50, 2, "2026-03-10", None, None, None, "{}"),
        ("t2", "scanner", "SPY", "iron_condor", "closed_profit", 590, 578, "2026-03-20",
         2.00, 1, "2026-03-01", "2026-03-15T10:00:00", "profit_target", 120.0, "{}"),
        ("t3", "scanner", "SPY", "credit_spread", "closed_loss", 570, 558, "2026-03-18",
         1.80, 1, "2026-03-05", "2026-03-15T14:00:00", "stop_loss", -80.0, "{}"),
        ("t4", "scanner", "QQQ", "credit_spread", "closed_profit", 480, 468, "2026-03-10",
         1.20, 1, "2026-02-20", "2026-03-08T11:00:00", "profit_target", 60.0, "{}"),
    ]
    conn.executemany(
        "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),datetime('now'))",
        trades,
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def sample_config(tmp_db):
    return {
        "experiment_id": "EXP-TEST",
        "db_path": tmp_db,
        "risk": {"account_size": 100_000},
        "alpaca": {"base_url": "https://paper-api.alpaca.markets"},
    }


# ---------------------------------------------------------------------------
# Unit tests — helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_pnl_class(self):
        assert _pnl_class(100) == "profit"
        assert _pnl_class(-50) == "loss"
        assert _pnl_class(0) == "neutral"

    def test_pnl_str(self):
        assert _pnl_str(150.5) == "+$150.50"
        assert _pnl_str(-42.1) == "$-42.10"
        assert _pnl_str(0) == "$0.00"

    def test_pct_str(self):
        assert _pct_str(5.5) == "+5.50%"
        assert _pct_str(-3.2) == "-3.20%"

    def test_hold_days(self):
        assert _hold_days({"entry_date": "2026-03-01", "exit_date": "2026-03-15"}) == "14"
        assert _hold_days({"entry_date": "", "exit_date": ""}) == "—"
        assert _hold_days({}) == "—"

    def test_dte_remaining(self):
        assert _dte_remaining({"expiration": "2026-04-01"}, "2026-03-15") == "17"
        assert _dte_remaining({"expiration": "2026-03-10"}, "2026-03-15") == "0"
        assert _dte_remaining({}, "2026-03-15") == "—"

    def test_compute_stats_empty(self):
        s = _compute_stats([], 100_000)
        assert s["count"] == 0
        assert s["win_rate"] == 0

    def test_compute_stats(self):
        trades = [
            {"pnl": 100}, {"pnl": 200}, {"pnl": -50},
        ]
        s = _compute_stats(trades, 100_000)
        assert s["count"] == 3
        assert s["wins"] == 2
        assert s["losses"] == 1
        assert s["total_pnl"] == 250
        assert abs(s["win_rate"] - 66.67) < 0.1
        assert s["avg_win"] == 150.0
        assert s["avg_loss"] == -50.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestCollectReportData:
    @patch("scripts.daily_report.get_alpaca_equity", return_value=None)
    def test_collect_data(self, mock_eq, sample_config):
        os.environ["PILOTAI_DB_PATH"] = sample_config["db_path"]
        data = collect_report_data(sample_config, report_date="2026-03-15")

        assert data["experiment_id"] == "EXP-TEST"
        assert data["date"] == "2026-03-15"
        assert len(data["open_trades"]) == 1
        assert len(data["closed_today"]) == 2  # t2 and t3
        assert len(data["all_closed"]) == 3

    @patch("scripts.daily_report.get_alpaca_equity", return_value=None)
    def test_rolling_aggregates(self, mock_eq, sample_config):
        os.environ["PILOTAI_DB_PATH"] = sample_config["db_path"]
        data = collect_report_data(sample_config, report_date="2026-03-15")

        # 7-day: t2, t3 (closed on 3/15), t4 closed 3/8 (within 7 days of 3/15)
        assert len(data["closed_7d"]) == 3
        # 30-day: all 3 closed trades
        assert len(data["closed_30d"]) == 3


class TestGenerateHTML:
    @patch("scripts.daily_report.get_alpaca_equity", return_value=105_000.0)
    def test_html_contains_sections(self, mock_eq, sample_config):
        os.environ["PILOTAI_DB_PATH"] = sample_config["db_path"]
        data = collect_report_data(sample_config, report_date="2026-03-15")
        html = generate_html(data)

        assert "Daily Report" in html
        assert "EXP-TEST" in html
        assert "Open Positions" in html
        assert "Closed Today" in html
        assert "Account Summary" in html
        assert "Strategy Breakdown" in html
        assert "Rolling Aggregates" in html
        assert "Upcoming Economic Events" in html
        assert "SPY" in html

    @patch("scripts.daily_report.get_alpaca_equity", return_value=None)
    def test_html_no_open_positions(self, mock_eq, tmp_path):
        """Config with empty DB produces valid HTML."""
        db_path = str(tmp_path / "empty.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE trades (
            id TEXT PRIMARY KEY, source TEXT, ticker TEXT, strategy_type TEXT,
            status TEXT, short_strike REAL, long_strike REAL, expiration TEXT,
            credit REAL, contracts INTEGER, entry_date TEXT, exit_date TEXT,
            exit_reason TEXT, pnl REAL, metadata JSON,
            created_at TEXT, updated_at TEXT)""")
        conn.execute("""CREATE TABLE alerts (
            id TEXT PRIMARY KEY, ticker TEXT, data JSON, created_at TEXT)""")
        conn.execute("""CREATE TABLE regime_snapshots (
            id INTEGER PRIMARY KEY, regime TEXT, confidence REAL,
            features JSON, created_at TEXT)""")
        conn.execute("""CREATE TABLE scanner_state (
            key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)""")
        conn.commit()
        conn.close()

        config = {"experiment_id": "EMPTY", "db_path": db_path,
                  "risk": {"account_size": 50_000}, "alpaca": {}}
        data = collect_report_data(config, report_date="2026-03-15")
        html = generate_html(data)
        assert "No open positions" in html
        assert "No trades closed today" in html


class TestSendTelegram:
    @patch("scripts.daily_report.requests.post")
    def test_send_success(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        mock_post.return_value.raise_for_status = MagicMock()

        os.environ["TELEGRAM_BOT_TOKEN"] = "fake-token"
        os.environ["TELEGRAM_CHAT_ID"] = "12345"

        result = send_html_report_telegram("<html>test</html>", "2026-03-15", "EXP-TEST")
        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "sendDocument" in call_args[0][0]

    def test_send_no_credentials(self):
        os.environ.pop("TELEGRAM_BOT_TOKEN", None)
        os.environ.pop("TELEGRAM_CHAT_ID", None)
        result = send_html_report_telegram("<html>test</html>", "2026-03-15", "EXP-TEST")
        assert result is False


class TestLoadEnvFile:
    def test_load_env(self, tmp_path):
        env_file = tmp_path / ".env.test"
        env_file.write_text('MY_TEST_KEY=hello\n# comment\nANOTHER="world"\n')
        # Clear if exists
        os.environ.pop("MY_TEST_KEY", None)
        os.environ.pop("ANOTHER", None)

        load_env_file(str(env_file))
        assert os.environ.get("MY_TEST_KEY") == "hello"
        assert os.environ.get("ANOTHER") == "world"

        # Cleanup
        os.environ.pop("MY_TEST_KEY", None)
        os.environ.pop("ANOTHER", None)


class TestLegacyMetrics:
    def test_get_daily_summary_metrics(self, sample_config):
        os.environ["PILOTAI_DB_PATH"] = sample_config["db_path"]
        metrics = get_daily_summary_metrics(report_date="2026-03-15", account_size=100_000)

        assert metrics["date"] == "2026-03-15"
        assert metrics["closed_today"] == 2
        assert metrics["wins"] == 1
        assert metrics["losses"] == 1
        assert metrics["day_pnl"] == 40.0  # 120 + (-80)
