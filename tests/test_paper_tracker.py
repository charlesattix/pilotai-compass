"""Tests for compass/paper_tracker.py — paper trading performance tracker.

Covers:
  - classify_trades: status classification
  - compute_metrics: P&L, win rate, Sharpe, drawdown, edge cases
  - _compute_daily_pnl: aggregation by exit date
  - _compute_sharpe: insufficient data, zero std
  - _compute_max_drawdown: flat, winning, losing curves
  - load_trades: valid DB, missing DB, empty DB
  - collect_experiment_data: integration with registry
  - generate_html: structure and content
  - _fmt: None handling, formatting
"""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from compass.paper_tracker import (
    _compute_daily_pnl,
    _compute_max_drawdown,
    _compute_sharpe,
    _fmt,
    _status_badge,
    classify_trades,
    collect_experiment_data,
    compute_metrics,
    generate_html,
    load_trades,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _trade(
    id_="t1", status="closed_profit", pnl=50.0, credit=1.5,
    entry_date="2026-03-20T10:00:00", exit_date="2026-03-25T10:00:00",
    ticker="SPY", strategy_type="bull_put_spread", contracts=1,
    exit_reason="profit_target", **kwargs,
):
    """Build a trade dict with defaults."""
    d = {
        "id": id_, "status": status, "pnl": pnl, "credit": credit,
        "entry_date": entry_date, "exit_date": exit_date,
        "ticker": ticker, "strategy_type": strategy_type,
        "contracts": contracts, "exit_reason": exit_reason,
    }
    d.update(kwargs)
    return d


def _create_test_db(path: Path, trades=None):
    """Create a SQLite DB with the trades table schema and optional rows."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE trades (
            id TEXT PRIMARY KEY,
            source TEXT,
            ticker TEXT,
            strategy_type TEXT,
            status TEXT DEFAULT 'open',
            short_strike REAL,
            long_strike REAL,
            expiration TEXT,
            credit REAL,
            contracts INTEGER DEFAULT 1,
            entry_date TEXT,
            exit_date TEXT,
            exit_reason TEXT,
            pnl REAL,
            metadata JSON,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    if trades:
        for t in trades:
            conn.execute(
                "INSERT INTO trades (id, ticker, strategy_type, status, credit, "
                "contracts, entry_date, exit_date, exit_reason, pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (t["id"], t.get("ticker", "SPY"), t.get("strategy_type"),
                 t.get("status", "open"), t.get("credit"), t.get("contracts", 1),
                 t.get("entry_date"), t.get("exit_date"),
                 t.get("exit_reason"), t.get("pnl")),
            )
    conn.commit()
    conn.close()


# ── classify_trades ──────────────────────────────────────────────────────


class TestClassifyTrades:
    def test_open_trade(self):
        result = classify_trades([_trade(status="open")])
        assert len(result["open"]) == 1
        assert len(result["closed"]) == 0

    def test_closed_profit(self):
        result = classify_trades([_trade(status="closed_profit")])
        assert len(result["closed"]) == 1

    def test_closed_loss(self):
        result = classify_trades([_trade(status="closed_loss")])
        assert len(result["closed"]) == 1

    def test_closed_external(self):
        result = classify_trades([_trade(status="closed_external")])
        assert len(result["closed"]) == 1

    def test_unmanaged(self):
        result = classify_trades([_trade(status="unmanaged")])
        assert len(result["unmanaged"]) == 1

    def test_mixed(self):
        trades = [
            _trade(id_="t1", status="open"),
            _trade(id_="t2", status="closed_profit"),
            _trade(id_="t3", status="unmanaged"),
            _trade(id_="t4", status="closed_loss"),
        ]
        result = classify_trades(trades)
        assert len(result["open"]) == 1
        assert len(result["closed"]) == 2
        assert len(result["unmanaged"]) == 1

    def test_empty_list(self):
        result = classify_trades([])
        assert result == {"open": [], "closed": [], "unmanaged": []}


# ── compute_metrics ──────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_no_trades(self):
        m = compute_metrics([], [])
        assert m["total_trades"] == 0
        assert m["win_rate"] is None
        assert m["sharpe"] is None
        assert m["total_pnl"] == 0.0

    def test_one_winner(self):
        closed = [_trade(pnl=100.0)]
        m = compute_metrics(closed, [])
        assert m["wins"] == 1
        assert m["losses"] == 0
        assert m["win_rate"] == 1.0
        assert m["total_pnl"] == 100.0

    def test_one_loser(self):
        closed = [_trade(pnl=-50.0)]
        m = compute_metrics(closed, [])
        assert m["wins"] == 0
        assert m["losses"] == 1
        assert m["win_rate"] == 0.0
        assert m["total_pnl"] == -50.0

    def test_mixed_trades(self):
        closed = [
            _trade(id_="t1", pnl=100.0, exit_date="2026-03-20"),
            _trade(id_="t2", pnl=-30.0, exit_date="2026-03-21"),
            _trade(id_="t3", pnl=50.0, exit_date="2026-03-22"),
        ]
        m = compute_metrics(closed, [])
        assert m["wins"] == 2
        assert m["losses"] == 1
        assert m["win_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert m["total_pnl"] == pytest.approx(120.0)
        assert m["best_trade"] == 100.0
        assert m["worst_trade"] == -30.0

    def test_open_trades_counted(self):
        closed = [_trade(id_="t1", pnl=50.0)]
        open_t = [_trade(id_="t2", status="open", pnl=None)]
        m = compute_metrics(closed, open_t)
        assert m["total_trades"] == 2
        assert m["open_trades"] == 1
        assert m["closed_trades"] == 1

    def test_cumulative_return_pct(self):
        closed = [_trade(pnl=5000.0)]
        m = compute_metrics(closed, [], starting_capital=100_000)
        assert m["cumulative_return_pct"] == pytest.approx(5.0)

    def test_null_pnl_trades_excluded(self):
        closed = [_trade(id_="t1", pnl=None), _trade(id_="t2", pnl=100.0)]
        m = compute_metrics(closed, [])
        assert m["trades_with_pnl"] == 1
        assert m["total_pnl"] == 100.0

    def test_hold_days(self):
        closed = [_trade(
            entry_date="2026-03-20T10:00:00",
            exit_date="2026-03-25T10:00:00",
            pnl=50.0,
        )]
        m = compute_metrics(closed, [])
        assert m["avg_hold_days"] == 5.0


# ── _compute_daily_pnl ──────────────────────────────────────────────────


class TestComputeDailyPnl:
    def test_empty(self):
        assert _compute_daily_pnl([], 100_000) == []

    def test_single_trade(self):
        trades = [_trade(pnl=100.0, exit_date="2026-03-20")]
        daily = _compute_daily_pnl(trades, 100_000)
        assert daily == [100.0]

    def test_same_day_aggregation(self):
        trades = [
            _trade(id_="t1", pnl=50.0, exit_date="2026-03-20"),
            _trade(id_="t2", pnl=30.0, exit_date="2026-03-20"),
        ]
        daily = _compute_daily_pnl(trades, 100_000)
        assert daily == [80.0]

    def test_sorted_by_date(self):
        trades = [
            _trade(id_="t1", pnl=50.0, exit_date="2026-03-22"),
            _trade(id_="t2", pnl=-20.0, exit_date="2026-03-20"),
        ]
        daily = _compute_daily_pnl(trades, 100_000)
        assert daily == [-20.0, 50.0]

    def test_null_pnl_skipped(self):
        trades = [
            _trade(id_="t1", pnl=None, exit_date="2026-03-20"),
            _trade(id_="t2", pnl=50.0, exit_date="2026-03-21"),
        ]
        daily = _compute_daily_pnl(trades, 100_000)
        assert daily == [50.0]


# ── _compute_sharpe ──────────────────────────────────────────────────────


class TestComputeSharpe:
    def test_empty(self):
        assert _compute_sharpe([]) is None

    def test_single_day(self):
        assert _compute_sharpe([100.0]) is None

    def test_positive_sharpe(self):
        daily = [100, 120, 80, 110, 90, 130, 70, 140]
        sharpe = _compute_sharpe(daily)
        assert sharpe is not None
        assert sharpe > 0

    def test_zero_std(self):
        assert _compute_sharpe([100.0, 100.0, 100.0]) is None


# ── _compute_max_drawdown ────────────────────────────────────────────────


class TestComputeMaxDrawdown:
    def test_empty(self):
        dd, pct = _compute_max_drawdown([], 100_000)
        assert dd == 0.0
        assert pct is None

    def test_only_wins(self):
        dd, pct = _compute_max_drawdown([100, 200, 300], 100_000)
        assert dd == 0.0
        assert pct == 0.0

    def test_drawdown_then_recovery(self):
        daily = [100, -300, 200, 100]  # equity: 100100 → 99800 → 100000 → 100100
        dd, pct = _compute_max_drawdown(daily, 100_000)
        assert dd < 0
        assert pct < 0

    def test_deep_drawdown(self):
        daily = [-10000]  # 10% loss
        dd, pct = _compute_max_drawdown(daily, 100_000)
        assert dd == -10000
        assert pct == pytest.approx(-10.0, abs=0.1)


# ── load_trades ──────────────────────────────────────────────────────────


class TestLoadTrades:
    def test_missing_db(self, tmp_path):
        result = load_trades(tmp_path / "nonexistent.db")
        assert result == []

    def test_empty_db(self, tmp_path):
        db_path = tmp_path / "empty.db"
        _create_test_db(db_path)
        result = load_trades(db_path)
        assert result == []

    def test_with_trades(self, tmp_path):
        db_path = tmp_path / "test.db"
        _create_test_db(db_path, [
            _trade(id_="t1", pnl=100.0),
            _trade(id_="t2", pnl=-50.0),
        ])
        result = load_trades(db_path)
        assert len(result) == 2
        assert result[0]["id"] == "t1"


# ── _fmt ─────────────────────────────────────────────────────────────────


class TestFmt:
    def test_none(self):
        assert _fmt(None) == "—"

    def test_float(self):
        assert _fmt(42.567) == "42.57"

    def test_prefix_suffix(self):
        assert _fmt(100.0, ",.2f", prefix="$") == "$100.00"
        assert _fmt(5.0, ".2f", suffix="%") == "5.00%"

    def test_percent_format(self):
        assert _fmt(0.75, ".1%") == "75.0%"


# ── _status_badge ────────────────────────────────────────────────────────


class TestStatusBadge:
    def test_active(self):
        m = {"trades_with_pnl": 15, "open_trades": 2}
        assert "active" in _status_badge(m).lower()

    def test_early(self):
        m = {"trades_with_pnl": 3, "open_trades": 1}
        assert "early" in _status_badge(m).lower()

    def test_pending(self):
        m = {"trades_with_pnl": 0, "open_trades": 2}
        assert "pending" in _status_badge(m).lower()

    def test_waiting(self):
        m = {"trades_with_pnl": 0, "open_trades": 0}
        assert "awaiting" in _status_badge(m).lower()


# ── collect_experiment_data ──────────────────────────────────────────────


class TestCollectExperimentData:
    def test_filters_paper_trading_only(self):
        registry = {
            "EXP-1": {"id": "EXP-1", "status": "paper_trading", "name": "Test"},
            "EXP-2": {"id": "EXP-2", "status": "retired", "name": "Old"},
            "EXP-3": {"id": "EXP-3", "status": "in_development", "name": "Dev"},
        }
        result = collect_experiment_data(registry)
        assert len(result) == 1
        assert result[0]["id"] == "EXP-1"

    def test_empty_registry(self):
        result = collect_experiment_data({})
        assert result == []

    def test_missing_db_produces_empty_metrics(self):
        registry = {
            "EXP-999": {
                "id": "EXP-999", "status": "paper_trading",
                "name": "No DB", "ticker": "SPY",
            },
        }
        result = collect_experiment_data(registry)
        assert len(result) == 1
        assert result[0]["metrics"]["total_trades"] == 0


# ── generate_html ────────────────────────────────────────────────────────


class TestGenerateHTML:
    def _make_experiments(self):
        closed = [
            _trade(id_="t1", pnl=100.0, exit_date="2026-03-20"),
            _trade(id_="t2", pnl=-30.0, exit_date="2026-03-21"),
        ]
        open_t = [_trade(id_="t3", status="open", pnl=None)]
        metrics = compute_metrics(closed, open_t)
        return [{
            "id": "EXP-TEST",
            "name": "Test Experiment",
            "ticker": "SPY",
            "live_since": "2026-03-15",
            "description": "A test experiment",
            "db_path": "/tmp/test.db",
            "trades_total": 3,
            "classified": {"open": open_t, "closed": closed, "unmanaged": []},
            "metrics": metrics,
        }]

    def test_valid_html(self):
        html = generate_html(self._make_experiments())
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_experiment_id(self):
        html = generate_html(self._make_experiments())
        assert "EXP-TEST" in html

    def test_contains_comparison_table(self):
        html = generate_html(self._make_experiments())
        assert "Experiment Comparison" in html
        assert "Win Rate" in html
        assert "Sharpe" in html

    def test_contains_detail_section(self):
        html = generate_html(self._make_experiments())
        assert "Experiment Details" in html
        assert "Test Experiment" in html

    def test_contains_trade_rows(self):
        html = generate_html(self._make_experiments())
        assert "t1" in html
        assert "$100.00" in html

    def test_no_external_resources(self):
        html = generate_html(self._make_experiments())
        assert "http://" not in html
        assert "https://" not in html

    def test_empty_experiments(self):
        html = generate_html([])
        assert "<!DOCTYPE html>" in html
        assert "0" in html  # should show 0 experiments

    def test_no_data_experiment(self):
        exp = [{
            "id": "EXP-EMPTY",
            "name": "Empty",
            "ticker": "SPY",
            "live_since": "2026-03-29",
            "description": "No trades yet",
            "db_path": None,
            "trades_total": 0,
            "classified": {"open": [], "closed": [], "unmanaged": []},
            "metrics": compute_metrics([], []),
        }]
        html = generate_html(exp)
        assert "No trades recorded yet" in html
