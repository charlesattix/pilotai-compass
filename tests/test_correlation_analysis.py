"""Tests for compass/correlation_analysis.py — cross-experiment correlation analysis.

Covers:
  - daily_pnl_from_db: DB read, empty DB, missing DB
  - daily_pnl_from_csv: CSV read, missing columns
  - compute_risk_metrics: Sharpe, Sortino, Calmar, edge cases
  - build_return_matrix: alignment, common dates
  - compute_correlation_matrix: identity, known values
  - compute_rolling_correlation: window, pair labels
  - generate_html: structure and content
  - generate_correlation_report: end-to-end with synthetic data
"""

import math
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.correlation_analysis import (
    build_return_matrix,
    compute_correlation_matrix,
    compute_risk_metrics,
    compute_rolling_correlation,
    daily_pnl_from_csv,
    daily_pnl_from_db,
    generate_correlation_report,
    generate_html,
    load_all_daily_pnl,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _create_db(path: Path, trades=None):
    """Create a test SQLite DB with trades table."""
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE trades (
            id TEXT PRIMARY KEY,
            exit_date TEXT,
            pnl REAL,
            status TEXT DEFAULT 'closed_profit'
        )
    """)
    if trades:
        for t in trades:
            conn.execute(
                "INSERT INTO trades (id, exit_date, pnl, status) VALUES (?, ?, ?, ?)",
                (t["id"], t["exit_date"], t["pnl"], t.get("status", "closed_profit")),
            )
    conn.commit()
    conn.close()


def _create_csv(path: Path, n=50, seed=42):
    """Create a test CSV with exit_date and pnl columns."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    df = pd.DataFrame({
        "entry_date": dates - pd.Timedelta(days=5),
        "exit_date": dates,
        "pnl": rng.normal(50, 200, n),
        "year": 2024,
        "win": (rng.normal(50, 200, n) > 0).astype(int),
    })
    df.to_csv(path, index=False)


def _make_daily_pnl(n=100, mean=50, std=200, seed=42, start="2024-01-02"):
    """Generate a synthetic daily P&L series."""
    rng = np.random.RandomState(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(rng.normal(mean, std, n), index=idx, name="daily_pnl")


# ── daily_pnl_from_db ───────────────────────────────────────────────────


class TestDailyPnlFromDB:
    def test_missing_db(self, tmp_path):
        result = daily_pnl_from_db(tmp_path / "nonexistent.db")
        assert len(result) == 0

    def test_empty_db(self, tmp_path):
        db = tmp_path / "empty.db"
        _create_db(db)
        result = daily_pnl_from_db(db)
        assert len(result) == 0

    def test_with_trades(self, tmp_path):
        db = tmp_path / "test.db"
        _create_db(db, [
            {"id": "t1", "exit_date": "2024-03-20", "pnl": 100.0},
            {"id": "t2", "exit_date": "2024-03-21", "pnl": -50.0},
        ])
        result = daily_pnl_from_db(db)
        assert len(result) >= 2
        assert result.loc["2024-03-20"] == 100.0
        assert result.loc["2024-03-21"] == -50.0

    def test_same_day_aggregation(self, tmp_path):
        db = tmp_path / "test.db"
        _create_db(db, [
            {"id": "t1", "exit_date": "2024-03-20", "pnl": 100.0},
            {"id": "t2", "exit_date": "2024-03-20", "pnl": 50.0},
        ])
        result = daily_pnl_from_db(db)
        assert result.loc["2024-03-20"] == 150.0

    def test_null_pnl_skipped(self, tmp_path):
        db = tmp_path / "test.db"
        _create_db(db, [
            {"id": "t1", "exit_date": "2024-03-20", "pnl": None},
        ])
        result = daily_pnl_from_db(db)
        assert len(result) == 0


# ── daily_pnl_from_csv ──────────────────────────────────────────────────


class TestDailyPnlFromCSV:
    def test_missing_csv(self, tmp_path):
        result = daily_pnl_from_csv(tmp_path / "nonexistent.csv")
        assert len(result) == 0

    def test_with_csv(self, tmp_path):
        csv = tmp_path / "test.csv"
        _create_csv(csv, n=30)
        result = daily_pnl_from_csv(csv)
        assert len(result) >= 30  # filled to bday calendar
        assert not result.isna().any()

    def test_missing_columns(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"x": [1, 2]}).to_csv(csv, index=False)
        result = daily_pnl_from_csv(csv)
        assert len(result) == 0


# ── compute_risk_metrics ─────────────────────────────────────────────────


class TestComputeRiskMetrics:
    def test_empty_series(self):
        m = compute_risk_metrics(pd.Series(dtype=float))
        assert m["sharpe"] is None
        assert m["sortino"] is None
        assert m["calmar"] is None
        assert m["n_days"] == 0

    def test_single_day(self):
        m = compute_risk_metrics(pd.Series([100.0]))
        assert m["sharpe"] is None
        assert m["n_days"] == 1

    def test_positive_returns(self):
        pnl = _make_daily_pnl(252, mean=200, std=100, seed=42)
        m = compute_risk_metrics(pnl)
        assert m["sharpe"] is not None
        assert m["sharpe"] > 0
        assert m["sortino"] is not None
        assert m["sortino"] > 0
        assert m["annual_return_pct"] > 0
        assert m["total_return_pct"] > 0
        assert m["n_days"] == 252

    def test_negative_returns(self):
        pnl = _make_daily_pnl(100, mean=-200, std=100, seed=42)
        m = compute_risk_metrics(pnl)
        assert m["sharpe"] < 0
        assert m["max_drawdown_pct"] < 0
        assert m["total_return_pct"] < 0

    def test_calmar_positive_for_good_strategy(self):
        # Need enough variance to produce drawdowns (std > mean)
        pnl = _make_daily_pnl(252, mean=100, std=500, seed=42)
        m = compute_risk_metrics(pnl)
        # Calmar is CAGR / |max_dd|; should exist with this vol level
        if m["max_drawdown_pct"] is not None and m["max_drawdown_pct"] < 0:
            assert m["calmar"] is not None

    def test_win_rate_daily(self):
        # 80 positive days, 20 negative
        pnl = pd.Series([100] * 80 + [-50] * 20, dtype=float)
        m = compute_risk_metrics(pnl)
        assert m["win_rate_daily"] == pytest.approx(0.80)

    def test_max_drawdown(self):
        # Start at capital, drop 10%, recover
        pnl = pd.Series([0, -10000, -5000, 5000, 10000, 5000], dtype=float)
        m = compute_risk_metrics(pnl, starting_capital=100_000)
        assert m["max_drawdown_pct"] < 0
        assert m["max_drawdown_pct"] > -20  # should be ~-15%


# ── build_return_matrix ──────────────────────────────────────────────────


class TestBuildReturnMatrix:
    def test_empty_input(self):
        result = build_return_matrix({})
        assert result.empty

    def test_single_experiment(self):
        pnl = _make_daily_pnl(50)
        result = build_return_matrix({"EXP-1": pnl})
        assert len(result.columns) == 1
        assert "EXP-1" in result.columns

    def test_aligned_experiments(self):
        pnl1 = _make_daily_pnl(50, seed=42)
        pnl2 = _make_daily_pnl(50, seed=43)
        result = build_return_matrix({"A": pnl1, "B": pnl2})
        assert len(result.columns) == 2
        assert len(result) == 50

    def test_misaligned_keeps_common_only(self):
        pnl1 = _make_daily_pnl(50, start="2024-01-02", seed=42)
        pnl2 = _make_daily_pnl(50, start="2024-02-01", seed=43)
        result = build_return_matrix({"A": pnl1, "B": pnl2})
        # Only overlapping dates
        assert len(result) < 50
        assert not result.isna().any().any()


# ── compute_correlation_matrix ───────────────────────────────────────────


class TestComputeCorrelationMatrix:
    def test_empty(self):
        result = compute_correlation_matrix(pd.DataFrame())
        assert result.empty

    def test_too_few_points(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        result = compute_correlation_matrix(df)
        assert result.empty

    def test_identity_diagonal(self):
        pnl1 = _make_daily_pnl(50, seed=42)
        pnl2 = _make_daily_pnl(50, seed=43)
        rm = build_return_matrix({"A": pnl1, "B": pnl2})
        corr = compute_correlation_matrix(rm)
        assert corr.loc["A", "A"] == pytest.approx(1.0)
        assert corr.loc["B", "B"] == pytest.approx(1.0)

    def test_perfect_correlation(self):
        pnl = _make_daily_pnl(50, seed=42)
        rm = build_return_matrix({"A": pnl, "B": pnl})
        corr = compute_correlation_matrix(rm)
        assert corr.loc["A", "B"] == pytest.approx(1.0)

    def test_negative_correlation(self):
        pnl1 = _make_daily_pnl(50, seed=42)
        pnl2 = -pnl1  # opposite
        rm = build_return_matrix({"A": pnl1, "B": pnl2})
        corr = compute_correlation_matrix(rm)
        assert corr.loc["A", "B"] == pytest.approx(-1.0)


# ── compute_rolling_correlation ──────────────────────────────────────────


class TestComputeRollingCorrelation:
    def test_single_experiment_empty(self):
        pnl = _make_daily_pnl(50)
        rm = build_return_matrix({"A": pnl})
        result = compute_rolling_correlation(rm, window=10)
        assert result == {}

    def test_two_experiments(self):
        pnl1 = _make_daily_pnl(60, seed=42)
        pnl2 = _make_daily_pnl(60, seed=43)
        rm = build_return_matrix({"A": pnl1, "B": pnl2})
        result = compute_rolling_correlation(rm, window=10)
        assert "A vs B" in result
        assert len(result["A vs B"]) > 0

    def test_three_experiments_gives_three_pairs(self):
        pnl1 = _make_daily_pnl(60, seed=1)
        pnl2 = _make_daily_pnl(60, seed=2)
        pnl3 = _make_daily_pnl(60, seed=3)
        rm = build_return_matrix({"A": pnl1, "B": pnl2, "C": pnl3})
        result = compute_rolling_correlation(rm, window=10)
        assert len(result) == 3  # A vs B, A vs C, B vs C

    def test_values_in_range(self):
        pnl1 = _make_daily_pnl(60, seed=42)
        pnl2 = _make_daily_pnl(60, seed=43)
        rm = build_return_matrix({"A": pnl1, "B": pnl2})
        result = compute_rolling_correlation(rm, window=10)
        for series in result.values():
            assert series.min() >= -1.01
            assert series.max() <= 1.01


# ── generate_html ────────────────────────────────────────────────────────


class TestGenerateHTML:
    def _make_report_data(self):
        pnl1 = _make_daily_pnl(100, seed=42)
        pnl2 = _make_daily_pnl(100, seed=43)
        daily_pnls = {"EXP-A": pnl1, "EXP-B": pnl2}
        metrics = {
            k: compute_risk_metrics(v) for k, v in daily_pnls.items()
        }
        rm = build_return_matrix(daily_pnls)
        corr = compute_correlation_matrix(rm)
        rolling = compute_rolling_correlation(rm, 21)
        return daily_pnls, metrics, corr, rolling

    def test_valid_html(self):
        daily_pnls, metrics, corr, rolling = self._make_report_data()
        html = generate_html(daily_pnls, metrics, corr, rolling, "", "", "", 21)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_sections(self):
        daily_pnls, metrics, corr, rolling = self._make_report_data()
        html = generate_html(daily_pnls, metrics, corr, rolling, "", "", "", 21)
        assert "Risk Metrics Summary" in html
        assert "Equity Curves" in html
        assert "Correlation Matrix" in html
        assert "Portfolio Optimization Takeaways" in html

    def test_contains_experiment_names(self):
        daily_pnls, metrics, corr, rolling = self._make_report_data()
        html = generate_html(daily_pnls, metrics, corr, rolling, "", "", "", 21)
        assert "EXP-A" in html
        assert "EXP-B" in html

    def test_no_external_resources(self):
        daily_pnls, metrics, corr, rolling = self._make_report_data()
        html = generate_html(daily_pnls, metrics, corr, rolling, "", "", "", 21)
        assert "http://" not in html
        assert "https://" not in html

    def test_empty_experiments(self):
        html = generate_html({}, {}, pd.DataFrame(), {}, "", "", "", 21)
        assert "<!DOCTYPE html>" in html
        assert "Insufficient overlapping data" in html


# ── generate_correlation_report end-to-end ───────────────────────────────


class TestGenerateCorrelationReport:
    def test_end_to_end_with_csvs(self, tmp_path):
        csv_a = tmp_path / "a.csv"
        csv_b = tmp_path / "b.csv"
        _create_csv(csv_a, n=60, seed=42)
        _create_csv(csv_b, n=60, seed=43)

        sources = {
            "EXP-A": {"db": tmp_path / "no.db", "csv": csv_a, "ticker": "SPY"},
            "EXP-B": {"db": tmp_path / "no.db", "csv": csv_b, "ticker": "SPY"},
        }
        out = str(tmp_path / "report.html")
        result = generate_correlation_report(out, sources=sources)

        assert Path(result).exists()
        content = Path(result).read_text()
        assert "<!DOCTYPE html>" in content
        assert "EXP-A" in content
        assert "EXP-B" in content
        assert "data:image/png;base64," in content
        assert len(content) > 5000

    def test_single_experiment_no_crash(self, tmp_path):
        csv = tmp_path / "a.csv"
        _create_csv(csv, n=60, seed=42)
        sources = {"EXP-A": {"db": tmp_path / "no.db", "csv": csv, "ticker": "SPY"}}
        out = str(tmp_path / "report.html")
        result = generate_correlation_report(out, sources=sources)
        assert Path(result).exists()

    def test_no_data_no_crash(self, tmp_path):
        sources = {"EXP-X": {"db": tmp_path / "no.db", "csv": None, "ticker": "SPY"}}
        out = str(tmp_path / "report.html")
        result = generate_correlation_report(out, sources=sources)
        assert Path(result).exists()
