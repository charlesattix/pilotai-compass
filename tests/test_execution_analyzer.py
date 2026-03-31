"""Tests for compass/execution_analyzer.py — execution quality analysis.

Covers:
  - load_trades_from_db: DB read, metadata JSON parsing, missing DB
  - load_trades_from_csv: CSV read, column mapping, missing file
  - SlippageMetrics: computation, edge cases
  - fill_rate_by_dimension: groupby, binning, missing columns
  - compute_dimension_breakdowns: all standard dimensions
  - compute_outcome_metrics: P&L, win rate, hold days
  - Chart rendering: non-empty base64 output
  - generate_html: structure, KPIs, sections
  - generate_execution_report: end-to-end
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.execution_analyzer import (
    SlippageMetrics,
    compute_dimension_breakdowns,
    compute_outcome_metrics,
    compute_slippage,
    fill_rate_by_dimension,
    generate_execution_report,
    generate_html,
    load_trades_from_csv,
    load_trades_from_db,
    _render_pnl_distribution,
    _render_cumulative_pnl,
    _render_breakdown_chart,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _create_test_db(path, trades=None):
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE trades (
            id TEXT PRIMARY KEY, ticker TEXT, strategy_type TEXT,
            status TEXT, credit REAL, contracts INTEGER,
            short_strike REAL, long_strike REAL, expiration TEXT,
            entry_date TEXT, exit_date TEXT, exit_reason TEXT,
            pnl REAL, alpaca_fill_price REAL, alpaca_status TEXT,
            metadata TEXT
        )
    """)
    if trades:
        for t in trades:
            meta = json.dumps(t.get("metadata", {})) if t.get("metadata") else None
            conn.execute(
                "INSERT INTO trades (id, ticker, strategy_type, status, credit, "
                "contracts, entry_date, exit_date, pnl, metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (t["id"], t.get("ticker", "SPY"), t.get("strategy_type", "CS"),
                 t.get("status", "closed_profit"), t.get("credit"),
                 t.get("contracts", 1), t.get("entry_date"),
                 t.get("exit_date"), t.get("pnl"), meta),
            )
    conn.commit()
    conn.close()


def _make_csv_data(n=50, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    pnl = rng.normal(30, 150, n)
    return pd.DataFrame({
        "entry_date": dates - pd.Timedelta(days=5),
        "exit_date": dates,
        "net_credit": rng.uniform(0.5, 2.0, n),
        "pnl": pnl,
        "return_pct": pnl / 500,
        "win": (pnl > 0).astype(int),
        "contracts": rng.randint(1, 5, n),
        "spread_width": 5.0,
        "max_loss_per_unit": rng.uniform(3, 5, n),
        "strategy_type": rng.choice(["CS", "SS"], n),
        "spread_type": rng.choice(["bull_put", "bear_call"], n),
        "exit_reason": rng.choice(["profit_target", "stop_loss", "expiry"], n),
        "vix": rng.uniform(12, 40, n),
        "spy_price": rng.uniform(400, 550, n),
        "dte_at_entry": rng.randint(7, 60, n),
        "hold_days": rng.randint(1, 30, n),
        "day_of_week": rng.randint(0, 5, n),
        "short_strike": rng.uniform(400, 550, n),
        "otm_pct": rng.uniform(2, 12, n),
    })


# ── load_trades_from_db ──────────────────────────────────────────────────


class TestLoadTradesFromDB:
    def test_missing_db(self, tmp_path):
        result = load_trades_from_db(tmp_path / "nonexistent.db")
        assert result.empty

    def test_empty_db(self, tmp_path):
        _create_test_db(tmp_path / "empty.db")
        result = load_trades_from_db(tmp_path / "empty.db")
        assert result.empty

    def test_with_trades(self, tmp_path):
        db = tmp_path / "test.db"
        _create_test_db(db, [
            {"id": "t1", "credit": 1.50, "pnl": 100, "entry_date": "2024-03-20"},
            {"id": "t2", "credit": 2.00, "pnl": -50, "entry_date": "2024-03-21"},
        ])
        result = load_trades_from_db(db)
        assert len(result) == 2
        assert "credit" in result.columns

    def test_metadata_parsing(self, tmp_path):
        db = tmp_path / "test.db"
        meta = {"signal_credit": 1.80, "mid_price": 1.75, "bid": 1.70, "ask": 1.80}
        _create_test_db(db, [
            {"id": "t1", "credit": 1.50, "pnl": 100, "entry_date": "2024-03-20",
             "metadata": meta},
        ])
        result = load_trades_from_db(db)
        assert result.iloc[0]["signal_credit"] == 1.80
        assert result.iloc[0]["mid_price"] == 1.75


# ── load_trades_from_csv ─────────────────────────────────────────────────


class TestLoadTradesFromCSV:
    def test_missing_csv(self, tmp_path):
        result = load_trades_from_csv(tmp_path / "nonexistent.csv")
        assert result.empty

    def test_with_csv(self, tmp_path):
        csv = tmp_path / "test.csv"
        _make_csv_data(30).to_csv(csv, index=False)
        result = load_trades_from_csv(csv)
        assert len(result) == 30
        assert "credit" in result.columns
        assert "pnl" in result.columns

    def test_missing_required_columns(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv, index=False)
        result = load_trades_from_csv(csv)
        assert result.empty


# ── compute_slippage ─────────────────────────────────────────────────────


class TestComputeSlippage:
    def test_empty_trades(self):
        s = compute_slippage(pd.DataFrame())
        assert s.n_trades == 0
        assert s.mean_slippage is None

    def test_with_signal_and_fill(self):
        trades = pd.DataFrame({
            "signal_credit": [2.00, 1.80, 1.50],
            "credit": [1.90, 1.75, 1.55],
            "contracts": [1, 2, 1],
        })
        s = compute_slippage(trades)
        assert s.n_with_slippage_data == 3
        assert s.mean_slippage is not None
        # Slippage = signal - fill: (0.10, 0.05, -0.05) → mean = 0.033
        assert abs(s.mean_slippage - 0.0333) < 0.01

    def test_no_slippage_data(self):
        trades = pd.DataFrame({
            "pnl": [100, -50],
            "contracts": [1, 1],
        })
        s = compute_slippage(trades)
        assert s.n_trades == 2
        assert s.n_with_slippage_data == 0

    def test_backtest_mode(self):
        """In backtest mode, credit=fill=signal, so slippage is 0."""
        trades = pd.DataFrame({
            "credit": [1.50, 2.00],
            "spread_width": [5.0, 5.0],
            "contracts": [1, 1],
        })
        s = compute_slippage(trades)
        assert s.n_with_slippage_data == 2
        assert s.mean_slippage == pytest.approx(0.0)


# ── fill_rate_by_dimension ───────────────────────────────────────────────


class TestFillRateByDimension:
    def test_categorical_groupby(self):
        trades = pd.DataFrame({
            "strategy_type": ["CS", "CS", "SS", "SS"],
            "win": [1, 0, 1, 1],
            "pnl": [100, -50, 80, 60],
            "credit": [1.5, 1.2, 2.0, 1.8],
            "return_pct": [10, -5, 8, 6],
        })
        result = fill_rate_by_dimension(trades, "strategy_type")
        assert len(result) == 2
        assert "count" in result.columns
        assert "win_rate" in result.columns

    def test_binned_dimension(self):
        trades = pd.DataFrame({
            "vix": [12, 18, 22, 28, 40],
            "win": [1, 1, 0, 1, 0],
            "pnl": [100, 50, -30, 80, -100],
            "credit": [1.0] * 5,
            "return_pct": [10, 5, -3, 8, -10],
        })
        result = fill_rate_by_dimension(
            trades, "vix_bucket",
            bins=[0, 15, 25, 50],
            bin_labels=["Low", "Med", "High"],
        )
        assert len(result) == 3

    def test_missing_column(self):
        trades = pd.DataFrame({"pnl": [100]})
        result = fill_rate_by_dimension(trades, "nonexistent")
        assert result.empty


# ── compute_dimension_breakdowns ─────────────────────────────────────────


class TestComputeDimensionBreakdowns:
    def test_all_dimensions(self):
        df = _make_csv_data(100)
        breakdowns = compute_dimension_breakdowns(df)
        assert "day_of_week" in breakdowns or "vix_regime" in breakdowns
        assert len(breakdowns) >= 3  # at least day, vix, strategy

    def test_empty_trades(self):
        breakdowns = compute_dimension_breakdowns(pd.DataFrame())
        assert len(breakdowns) == 0


# ── compute_outcome_metrics ──────────────────────────────────────────────


class TestComputeOutcomeMetrics:
    def test_empty(self):
        m = compute_outcome_metrics(pd.DataFrame())
        assert m["n_trades"] == 0

    def test_with_data(self, tmp_path):
        csv = tmp_path / "test.csv"
        _make_csv_data(50).to_csv(csv, index=False)
        df = load_trades_from_csv(csv)  # maps net_credit → credit
        m = compute_outcome_metrics(df)
        assert m["n_trades"] == 50
        assert "total_pnl" in m
        assert "win_rate" in m
        assert "avg_hold_days" in m
        assert "avg_credit" in m

    def test_win_rate_correct(self):
        trades = pd.DataFrame({"win": [1, 1, 0], "pnl": [100, 50, -30]})
        m = compute_outcome_metrics(trades)
        assert m["win_rate"] == pytest.approx(2 / 3, abs=0.01)


# ── Chart rendering ──────────────────────────────────────────────────────


class TestChartRendering:
    def test_pnl_distribution(self):
        df = _make_csv_data(50)
        b64 = _render_pnl_distribution(df)
        assert len(b64) > 1000

    def test_cumulative_pnl(self):
        df = _make_csv_data(50)
        b64 = _render_cumulative_pnl(df)
        assert len(b64) > 1000

    def test_breakdown_chart(self):
        df = _make_csv_data(50)
        breakdowns = compute_dimension_breakdowns(df)
        b64 = _render_breakdown_chart(breakdowns)
        assert len(b64) > 1000

    def test_empty_data_no_crash(self):
        b64 = _render_pnl_distribution(pd.DataFrame())
        assert b64 == ""
        b64 = _render_cumulative_pnl(pd.DataFrame())
        assert b64 == ""


# ── generate_html ────────────────────────────────────────────────────────


class TestGenerateHTML:
    def test_valid_html(self):
        df = _make_csv_data(30)
        slippage = compute_slippage(df)
        outcomes = compute_outcome_metrics(df)
        breakdowns = compute_dimension_breakdowns(df)
        html = generate_html("TEST", df, slippage, outcomes, breakdowns, {})
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_sections(self):
        df = _make_csv_data(30)
        slippage = compute_slippage(df)
        outcomes = compute_outcome_metrics(df)
        breakdowns = compute_dimension_breakdowns(df)
        html = generate_html("TEST", df, slippage, outcomes, breakdowns, {})
        assert "Implementation Shortfall" in html
        assert "Trade Outcomes" in html
        assert "Execution Quality" in html

    def test_no_external_resources(self):
        df = _make_csv_data(30)
        slippage = compute_slippage(df)
        outcomes = compute_outcome_metrics(df)
        html = generate_html("TEST", df, slippage, outcomes, {}, {})
        assert "http://" not in html
        assert "https://" not in html


# ── generate_execution_report end-to-end ─────────────────────────────────


class TestGenerateExecutionReport:
    def test_end_to_end_csv(self, tmp_path):
        csv = tmp_path / "trades.csv"
        _make_csv_data(60).to_csv(csv, index=False)
        out = str(tmp_path / "report.html")
        result = generate_execution_report(csv_path=str(csv), output=out,
                                           experiment_label="Test")
        assert Path(result).exists()
        content = Path(result).read_text()
        assert "<!DOCTYPE html>" in content
        assert "data:image/png;base64," in content
        assert len(content) > 5000

    def test_end_to_end_db(self, tmp_path):
        db = tmp_path / "test.db"
        _create_test_db(db, [
            {"id": "t1", "credit": 1.5, "pnl": 100,
             "entry_date": "2024-03-20", "exit_date": "2024-03-25"},
            {"id": "t2", "credit": 2.0, "pnl": -50,
             "entry_date": "2024-03-21", "exit_date": "2024-03-26"},
        ])
        out = str(tmp_path / "report.html")
        result = generate_execution_report(db_path=str(db), output=out)
        assert Path(result).exists()

    def test_no_data_no_crash(self, tmp_path):
        out = str(tmp_path / "report.html")
        result = generate_execution_report(
            csv_path=str(tmp_path / "no.csv"),
            output=out,
        )
        assert Path(result).exists()
