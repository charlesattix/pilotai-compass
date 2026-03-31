"""Tests for compass/execution_quality.py — execution quality analysis."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.execution_quality import (
    CostAttribution,
    ExecutionQualityAnalyzer,
    ExecutionSummary,
    RegimeExecStats,
    SlippageStats,
    TimeOfDayBucket,
    estimate_cost_attribution,
    estimate_slippage_bps,
    recommend_execution_time,
)

ROOT = Path(__file__).resolve().parent.parent
EXP400_CSV = ROOT / "compass" / "training_data_exp400.csv"


def _make_trades(n: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "entry_date": pd.bdate_range("2023-01-02", periods=n),
        "exit_date": pd.bdate_range("2023-01-09", periods=n),
        "net_credit": rng.uniform(0.30, 1.50, n),
        "spread_width": np.full(n, 5.0),
        "vix": rng.uniform(12, 45, n),
        "contracts": rng.randint(1, 6, n),
        "regime": rng.choice(["bull", "bear", "high_vol", "low_vol"], n),
        "day_of_week": rng.randint(0, 5, n),
        "pnl": rng.normal(50, 300, n),
        "strategy_type": "CS",
    })


# ── estimate_slippage_bps ────────────────────────────────────────────────


class TestEstimateSlippage:

    def test_positive_for_valid_inputs(self):
        s = estimate_slippage_bps(0.65, 5.0, 20.0, 1)
        assert s > 0

    def test_higher_vix_more_slippage(self):
        low = estimate_slippage_bps(0.65, 5.0, 15.0, 1)
        high = estimate_slippage_bps(0.65, 5.0, 40.0, 1)
        assert high > low

    def test_more_contracts_more_slippage(self):
        one = estimate_slippage_bps(0.65, 5.0, 20.0, 1)
        ten = estimate_slippage_bps(0.65, 5.0, 20.0, 10)
        assert ten > one

    def test_zero_spread_width_returns_zero(self):
        assert estimate_slippage_bps(0.65, 0.0, 20.0) == 0.0

    def test_zero_credit_returns_zero(self):
        assert estimate_slippage_bps(0.0, 5.0, 20.0) == 0.0

    def test_returns_float(self):
        assert isinstance(estimate_slippage_bps(0.65, 5.0, 20.0), float)


# ── estimate_cost_attribution ────────────────────────────────────────────


class TestCostAttribution:

    def test_components_sum_to_total(self):
        c = estimate_cost_attribution(0.65, 5.0, 25.0, 2)
        assert abs(c.spread_cost_bps + c.timing_cost_bps + c.impact_cost_bps - c.total_cost_bps) < 0.1

    def test_spread_is_largest_component(self):
        c = estimate_cost_attribution(0.65, 5.0, 20.0, 1)
        assert c.spread_cost_bps > c.timing_cost_bps
        assert c.spread_cost_bps > c.impact_cost_bps

    def test_high_vix_increases_spread_cost(self):
        low = estimate_cost_attribution(0.65, 5.0, 15.0, 1)
        high = estimate_cost_attribution(0.65, 5.0, 40.0, 1)
        assert high.spread_cost_bps > low.spread_cost_bps

    def test_zero_width_returns_empty(self):
        c = estimate_cost_attribution(0.65, 0.0, 20.0)
        assert c.total_cost_bps == 0.0


# ── recommend_execution_time ─────────────────────────────────────────────


class TestRecommendExecTime:

    def test_bull_recommendation(self):
        r = recommend_execution_time("bull")
        assert "10:00" in r

    def test_crash_recommendation(self):
        r = recommend_execution_time("crash")
        assert "limit" in r.lower() or "11:00" in r

    def test_unknown_regime_returns_default(self):
        r = recommend_execution_time("unknown_regime")
        assert "10:00" in r

    def test_all_regimes_have_text(self):
        for regime in ["bull", "bear", "high_vol", "low_vol", "crash"]:
            r = recommend_execution_time(regime)
            assert len(r) > 10


# ── ExecutionQualityAnalyzer ─────────────────────────────────────────────


class TestAnalyzerFit:

    def test_fit_returns_self(self):
        a = ExecutionQualityAnalyzer()
        result = a.fit(_make_trades())
        assert result is a

    def test_summary_populated(self):
        a = ExecutionQualityAnalyzer()
        a.fit(_make_trades())
        s = a.summary()
        assert isinstance(s, ExecutionSummary)
        assert s.total_trades == 50

    def test_slippage_stats(self):
        a = ExecutionQualityAnalyzer()
        a.fit(_make_trades())
        s = a.summary().overall_slippage
        assert s.n_trades == 50
        assert s.mean_slippage_bps > 0
        assert s.p95_slippage_bps >= s.mean_slippage_bps

    def test_cost_attribution_populated(self):
        a = ExecutionQualityAnalyzer()
        a.fit(_make_trades())
        ca = a.summary().cost_attribution
        assert ca.total_cost_bps > 0
        assert ca.spread_cost_bps > 0

    def test_regime_stats_populated(self):
        a = ExecutionQualityAnalyzer()
        a.fit(_make_trades())
        rs = a.summary().regime_stats
        assert len(rs) >= 2
        regimes = {r.regime for r in rs}
        assert "bull" in regimes

    def test_time_of_day_populated(self):
        a = ExecutionQualityAnalyzer()
        a.fit(_make_trades())
        tod = a.summary().time_of_day
        assert len(tod) >= 2

    def test_empty_dataframe(self):
        a = ExecutionQualityAnalyzer()
        a.fit(pd.DataFrame())
        assert a.summary().total_trades == 0

    def test_missing_columns_handled(self):
        df = pd.DataFrame({"entry_date": pd.bdate_range("2023-01-02", periods=5)})
        a = ExecutionQualityAnalyzer()
        a.fit(df)
        assert a.summary().total_trades == 5

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_real_data(self):
        df = pd.read_csv(EXP400_CSV)
        a = ExecutionQualityAnalyzer()
        a.fit(df)
        s = a.summary()
        assert s.total_trades > 200
        assert s.overall_slippage.mean_slippage_bps > 0
        assert len(s.regime_stats) >= 2


# ── HTML report ──────────────────────────────────────────────────────────


class TestGenerateReport:

    def test_returns_html(self):
        a = ExecutionQualityAnalyzer()
        a.fit(_make_trades())
        html = a.generate_report()
        assert "<!DOCTYPE html>" in html
        assert "Execution Quality" in html

    def test_contains_sections(self):
        a = ExecutionQualityAnalyzer()
        a.fit(_make_trades())
        html = a.generate_report()
        assert "Cost Attribution" in html
        assert "Slippage Distribution" in html
        assert "Day of Week" in html
        assert "Regime-Conditioned" in html

    def test_writes_to_file(self):
        a = ExecutionQualityAnalyzer()
        a.fit(_make_trades())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "report.html")
            html = a.generate_report(path)
            assert Path(path).exists()
            assert Path(path).stat().st_size > 3000

    def test_unfitted_returns_placeholder(self):
        a = ExecutionQualityAnalyzer()
        html = a.generate_report()
        assert "No data" in html
