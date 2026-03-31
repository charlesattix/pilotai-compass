"""Tests for compass/backtest_reconciler.py — backtest vs paper reconciliation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.backtest_reconciler import (
    BacktestReconciler,
    ReconciliationResult,
    RootCauseSummary,
    TradeDiscrepancy,
    TradePairComparison,
    classify_root_cause,
    classify_severity,
    compute_reconciliation_score,
    match_trades,
    SEVERITY_SCORE,
    _fmt_pct,
    _fmt_dollar,
    _score_color,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_trades(
    n: int = 30,
    seed: int = 42,
    price_noise: float = 0.02,
    pnl_noise: float = 15.0,
    time_noise_hours: float = 0.5,
    with_trade_id: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate paired backtest and paper trade data with controlled noise."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    exit_dates = dates + pd.Timedelta(days=2)

    bt_entry = rng.uniform(3.0, 8.0, n)
    bt_exit = bt_entry + rng.normal(0.5, 1.0, n)
    bt_pnl = (bt_exit - bt_entry) * 100  # 1 contract * 100 multiplier

    pp_entry = bt_entry + rng.normal(0, price_noise, n)
    pp_exit = bt_exit + rng.normal(0, price_noise, n)
    pp_pnl = bt_pnl + rng.normal(0, pnl_noise, n)

    time_offsets = pd.to_timedelta(rng.normal(0, time_noise_hours, n), unit="h")
    pp_entry_dates = dates + time_offsets.round("min")
    pp_exit_dates = exit_dates + time_offsets.round("min")

    bt = pd.DataFrame({
        "entry_price": bt_entry,
        "exit_price": bt_exit,
        "pnl": bt_pnl,
        "entry_date": dates,
        "exit_date": exit_dates,
    })
    pp = pd.DataFrame({
        "entry_price": pp_entry,
        "exit_price": pp_exit,
        "pnl": pp_pnl,
        "entry_date": pp_entry_dates,
        "exit_date": pp_exit_dates,
    })

    if with_trade_id:
        ids = [f"T-{i:04d}" for i in range(n)]
        bt["trade_id"] = ids
        pp["trade_id"] = ids

    return bt, pp


from typing import Tuple


@pytest.fixture
def matched_trades():
    return _make_trades(30, seed=42)


@pytest.fixture
def noisy_trades():
    """Trades with high noise — many discrepancies expected."""
    return _make_trades(30, seed=99, price_noise=0.5, pnl_noise=200.0, time_noise_hours=4.0)


@pytest.fixture
def perfect_trades():
    """Identical BT and PP trades — score should be 100."""
    bt, _ = _make_trades(20, seed=42, price_noise=0.0, pnl_noise=0.0, time_noise_hours=0.0)
    return bt, bt.copy()


@pytest.fixture
def reconciler(matched_trades):
    bt, pp = matched_trades
    return BacktestReconciler(bt, pp)


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_basic_init(self, matched_trades):
        bt, pp = matched_trades
        rec = BacktestReconciler(bt, pp)
        assert len(rec.bt) == 30
        assert len(rec.pp) == 30

    def test_missing_columns_bt(self):
        bt = pd.DataFrame({"entry_price": [1.0]})
        pp = pd.DataFrame({"entry_price": [1.0], "exit_price": [1.1],
                            "pnl": [10], "entry_date": ["2024-01-02"],
                            "exit_date": ["2024-01-04"]})
        with pytest.raises(ValueError, match="backtest.*missing"):
            BacktestReconciler(bt, pp)

    def test_missing_columns_pp(self):
        bt = pd.DataFrame({"entry_price": [1.0], "exit_price": [1.1],
                            "pnl": [10], "entry_date": ["2024-01-02"],
                            "exit_date": ["2024-01-04"]})
        pp = pd.DataFrame({"entry_price": [1.0]})
        with pytest.raises(ValueError, match="paper.*missing"):
            BacktestReconciler(bt, pp)

    def test_custom_thresholds(self, matched_trades):
        bt, pp = matched_trades
        rec = BacktestReconciler(bt, pp, thresholds={"pnl_tol_dollars": 50.0})
        assert rec.thresholds["pnl_tol_dollars"] == 50.0
        # defaults still present
        assert rec.thresholds["entry_price_tol_pct"] == 0.5


# ── Trade matching tests ─────────────────────────────────────────────────


class TestMatchTrades:
    def test_match_by_trade_id(self, matched_trades):
        bt, pp = matched_trades
        pairs, ubt, upp = match_trades(bt, pp)
        assert len(pairs) == 30
        assert len(ubt) == 0
        assert len(upp) == 0

    def test_match_by_date_fallback(self):
        bt, pp = _make_trades(10, with_trade_id=False)
        pairs, ubt, upp = match_trades(bt, pp)
        assert len(pairs) > 0  # most should match
        assert len(pairs) + len(ubt) == len(bt)

    def test_partial_match(self):
        bt, pp = _make_trades(20, with_trade_id=True)
        # Remove some paper trades
        pp_partial = pp.iloc[:15].copy()
        pairs, ubt, upp = match_trades(bt, pp_partial)
        assert len(pairs) == 15
        assert len(ubt) == 5

    def test_extra_paper_trades(self):
        bt, pp = _make_trades(10, with_trade_id=True)
        # Add extra paper trade
        extra = pp.iloc[[0]].copy()
        extra["trade_id"] = "EXTRA-001"
        pp_extended = pd.concat([pp, extra], ignore_index=True)
        pairs, ubt, upp = match_trades(bt, pp_extended)
        assert len(pairs) == 10
        assert len(upp) == 1


# ── Root cause classification tests ──────────────────────────────────────


class TestRootCause:
    def test_slippage_small_price_diff(self):
        assert classify_root_cause("entry_price", 0.3, 5.0) == "slippage"

    def test_fill_quality_medium_price_diff(self):
        assert classify_root_cause("entry_price", 2.0, 5.0) == "fill_quality"

    def test_data_staleness_large_price_diff(self):
        assert classify_root_cause("exit_price", 5.0, 5.0) == "data_staleness"

    def test_timing_drift(self):
        assert classify_root_cause("timing", 0, 45) == "timing_drift"

    def test_model_divergence_timing(self):
        assert classify_root_cause("timing", 0, 120) == "model_divergence"

    def test_pnl_slippage(self):
        assert classify_root_cause("pnl", 2.0, 5.0) == "slippage"

    def test_pnl_model_divergence(self):
        assert classify_root_cause("pnl", 50.0, 5.0) == "model_divergence"


# ── Severity classification tests ────────────────────────────────────────


class TestSeverity:
    def test_low_price(self):
        assert classify_severity(0.3, "entry_price") == "low"

    def test_medium_price(self):
        assert classify_severity(1.0, "exit_price") == "medium"

    def test_high_price(self):
        assert classify_severity(3.0, "entry_price") == "high"

    def test_low_pnl(self):
        assert classify_severity(3.0, "pnl") == "low"

    def test_high_pnl(self):
        assert classify_severity(25.0, "pnl") == "high"

    def test_severity_scores(self):
        assert SEVERITY_SCORE["low"] < SEVERITY_SCORE["medium"] < SEVERITY_SCORE["high"]


# ── Score computation tests ──────────────────────────────────────────────


class TestScore:
    def test_perfect_score(self, perfect_trades):
        bt, pp = perfect_trades
        rec = BacktestReconciler(bt, pp)
        result = rec.reconcile()
        assert result.reconciliation_score == 100.0

    def test_score_range(self, reconciler):
        result = reconciler.reconcile()
        assert 0 <= result.reconciliation_score <= 100

    def test_score_breakdown_sums(self, reconciler):
        result = reconciler.reconcile()
        bd = result.score_breakdown
        total = sum(bd.values())
        assert abs(total - result.reconciliation_score) < 0.1

    def test_no_trades_score(self):
        score, bd = compute_reconciliation_score([], 0, 0, {
            "entry_price_tol_pct": 0.5, "exit_price_tol_pct": 0.5,
            "pnl_tol_dollars": 20, "timing_tol_minutes": 30,
        })
        assert score == 100.0

    def test_no_matches_low_score(self):
        score, bd = compute_reconciliation_score([], 10, 10, {
            "entry_price_tol_pct": 0.5, "exit_price_tol_pct": 0.5,
            "pnl_tol_dollars": 20, "timing_tol_minutes": 30,
        })
        assert score == 0.0

    def test_noisy_lower_than_clean(self, matched_trades, noisy_trades):
        bt_c, pp_c = matched_trades
        bt_n, pp_n = noisy_trades
        r_clean = BacktestReconciler(bt_c, pp_c).reconcile()
        r_noisy = BacktestReconciler(bt_n, pp_n).reconcile()
        assert r_clean.reconciliation_score > r_noisy.reconciliation_score


# ── Full reconciliation tests ────────────────────────────────────────────


class TestReconciliation:
    def test_reconcile_returns_result(self, reconciler):
        result = reconciler.reconcile()
        assert isinstance(result, ReconciliationResult)
        assert result.n_backtest_trades == 30
        assert result.n_paper_trades == 30
        assert result.n_matched == 30

    def test_discrepancies_found(self, reconciler):
        result = reconciler.reconcile()
        # With default noise, some discrepancies expected
        assert isinstance(result.discrepancies, list)

    def test_root_cause_summary(self, reconciler):
        result = reconciler.reconcile()
        for rcs in result.root_cause_summary:
            assert isinstance(rcs, RootCauseSummary)
            assert rcs.count > 0
            assert 0 <= rcs.pct_of_total <= 1

    def test_noisy_has_more_discrepancies(self, matched_trades, noisy_trades):
        bt_c, pp_c = matched_trades
        bt_n, pp_n = noisy_trades
        r_clean = BacktestReconciler(bt_c, pp_c).reconcile()
        r_noisy = BacktestReconciler(bt_n, pp_n).reconcile()
        assert len(r_noisy.discrepancies) >= len(r_clean.discrepancies)

    def test_comparison_fields(self, reconciler):
        result = reconciler.reconcile()
        if result.comparisons:
            c = result.comparisons[0]
            assert isinstance(c, TradePairComparison)
            assert c.is_matched is True
            assert isinstance(c.entry_price_diff_pct, float)


# ── TradeDiscrepancy tests ───────────────────────────────────────────────


class TestTradeDiscrepancy:
    def test_to_dict(self):
        td = TradeDiscrepancy(
            trade_id="T-0001", field_name="entry_price",
            backtest_value=5.0, paper_value=5.05,
            diff=0.05, diff_pct=1.0,
            root_cause="slippage", severity="low",
        )
        d = td.to_dict()
        assert d["trade_id"] == "T-0001"
        assert d["root_cause"] == "slippage"
        assert "diff_pct" in d


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generate_report_creates_file(self, reconciler):
        result = reconciler.reconcile()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "test_recon.html"
            path = BacktestReconciler.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Reconciliation" in content

    def test_report_contains_score(self, reconciler):
        result = reconciler.reconcile()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            BacktestReconciler.generate_report(result, out)
            content = out.read_text()
            assert "Reconciliation Score" in content

    def test_report_contains_table(self, reconciler):
        result = reconciler.reconcile()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            BacktestReconciler.generate_report(result, out)
            content = out.read_text()
            assert "Trade-Level" in content
            assert "<table" in content

    def test_report_contains_svg(self, reconciler):
        result = reconciler.reconcile()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            BacktestReconciler.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content

    def test_fmt_helpers(self):
        assert _fmt_pct(12.34) == "12.34%"
        assert _fmt_dollar(1234.5) == "$1,234.50"

    def test_score_color(self):
        assert _score_color(90) == "#3fb950"
        assert _score_color(70) == "#d29922"
        assert _score_color(40) == "#f85149"

    def test_report_default_path(self, reconciler):
        result = reconciler.reconcile()
        path = BacktestReconciler.generate_report(result)
        assert path.exists()
        assert "reconciliation.html" in str(path)
        path.unlink(missing_ok=True)
