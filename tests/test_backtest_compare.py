"""Tests for compass/backtest_compare.py — cross-experiment comparison."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from compass.backtest_compare import (
    ExperimentMetrics,
    build_daily_return_matrix,
    compute_all_metrics,
    compute_correlation,
    compute_metrics,
    generate_html,
    load_all_trades,
    load_trades,
)

ROOT = Path(__file__).resolve().parent.parent
EXP400_CSV = ROOT / "compass" / "training_data_exp400.csv"
EXP401_CSV = ROOT / "compass" / "training_data_exp401.csv"


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_trades(n: int = 50, seed: int = 42, regime: str = "bull") -> pd.DataFrame:
    """Generate synthetic trade data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n * 3, freq="B")
    entry_dates = dates[::3][:n]
    exit_dates = dates[2::3][:n]
    pnl = rng.normal(50, 200, n)
    win = (pnl > 0).astype(int)
    return pd.DataFrame({
        "entry_date": entry_dates,
        "exit_date": exit_dates,
        "year": [d.year for d in exit_dates],
        "pnl": pnl,
        "return_pct": pnl / 1000 * 100,
        "win": win,
        "regime": regime,
        "strategy_type": "CS",
        "vix": rng.uniform(15, 30, n),
    })


@pytest.fixture
def two_experiments() -> Dict[str, pd.DataFrame]:
    return {
        "EXP-A": _make_trades(50, seed=42, regime="bull"),
        "EXP-B": _make_trades(40, seed=99, regime="bear"),
    }


@pytest.fixture
def multi_regime_trades() -> pd.DataFrame:
    frames = []
    for regime in ["bull", "bear", "high_vol", "crash"]:
        frames.append(_make_trades(15, seed=hash(regime) % 2**31, regime=regime))
    return pd.concat(frames, ignore_index=True)


# ── ExperimentMetrics ────────────────────────────────────────────────────


class TestExperimentMetrics:

    def test_default_creation(self):
        m = ExperimentMetrics(experiment="TEST", period="2023")
        assert m.experiment == "TEST"
        assert m.n_trades == 0
        assert m.win_rate == 0.0

    def test_fields_accessible(self):
        m = ExperimentMetrics(
            experiment="EXP-400", period="Overall", n_trades=100,
            win_rate=65.0, sharpe=1.5, total_pnl=25000.0,
        )
        assert m.win_rate == 65.0
        assert m.sharpe == 1.5


# ── compute_metrics ──────────────────────────────────────────────────────


class TestComputeMetrics:

    def test_basic_metrics(self):
        df = _make_trades(30)
        m = compute_metrics(df, "2023", "TEST")
        assert m.n_trades == 30
        assert 0 <= m.win_rate <= 100
        assert m.experiment == "TEST"
        assert m.period == "2023"

    def test_empty_trades(self):
        df = pd.DataFrame(columns=["entry_date", "exit_date", "pnl", "year"])
        m = compute_metrics(df, "2023", "TEST")
        assert m.n_trades == 0
        assert m.win_rate == 0.0
        assert m.sharpe == 0.0

    def test_all_winners(self):
        df = _make_trades(20, seed=42)
        df["pnl"] = abs(df["pnl"]) + 1  # all positive
        m = compute_metrics(df, "2023", "TEST")
        assert m.win_rate == 100.0
        assert m.total_pnl > 0
        assert m.profit_factor == 0.0  # no losses → denominator 0

    def test_all_losers(self):
        df = _make_trades(20, seed=42)
        df["pnl"] = -(abs(df["pnl"]) + 1)  # all negative
        m = compute_metrics(df, "2023", "TEST")
        assert m.win_rate == 0.0
        assert m.total_pnl < 0

    def test_sharpe_sign(self):
        # Mostly winning trades
        df = _make_trades(50, seed=42)
        df["pnl"] = abs(df["pnl"]) + 100
        m = compute_metrics(df, "2023", "TEST")
        assert m.sharpe > 0

    def test_max_dd_negative(self):
        df = _make_trades(50, seed=42)
        m = compute_metrics(df, "2023", "TEST")
        assert m.max_dd_pct <= 0

    def test_regime_passed_through(self):
        df = _make_trades(10)
        m = compute_metrics(df, "Overall", "TEST", regime="high_vol")
        assert m.regime == "high_vol"

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_real_exp400_data(self):
        df = load_trades(EXP400_CSV)
        m = compute_metrics(df, "Overall", "EXP-400")
        assert m.n_trades > 200
        assert m.sharpe != 0

    @pytest.mark.skipif(not EXP401_CSV.exists(), reason="data not available")
    def test_real_exp401_data(self):
        df = load_trades(EXP401_CSV)
        m = compute_metrics(df, "Overall", "EXP-401")
        assert m.n_trades > 300


# ── compute_all_metrics ──────────────────────────────────────────────────


class TestComputeAllMetrics:

    def test_returns_three_lists(self, two_experiments):
        overall, yearly, regime = compute_all_metrics(two_experiments)
        assert isinstance(overall, list)
        assert isinstance(yearly, list)
        assert isinstance(regime, list)

    def test_overall_one_per_experiment(self, two_experiments):
        overall, _, _ = compute_all_metrics(two_experiments)
        assert len(overall) == 2
        names = {m.experiment for m in overall}
        assert names == {"EXP-A", "EXP-B"}

    def test_yearly_entries_exist(self, two_experiments):
        _, yearly, _ = compute_all_metrics(two_experiments)
        assert len(yearly) >= 2  # at least one year per experiment

    def test_regime_entries(self, two_experiments):
        _, _, regime = compute_all_metrics(two_experiments)
        assert len(regime) >= 2  # one per regime per experiment

    def test_multi_regime_breakdown(self, multi_regime_trades):
        data = {"EXP-MULTI": multi_regime_trades}
        _, _, regime = compute_all_metrics(data)
        regimes = {m.regime for m in regime}
        assert "bull" in regimes
        assert "bear" in regimes

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_real_data_all_metrics(self):
        trades = load_all_trades()
        overall, yearly, regime = compute_all_metrics(trades)
        assert len(overall) >= 2
        assert len(yearly) >= 10  # multiple years per experiment
        assert len(regime) >= 4  # multiple regimes


# ── build_daily_return_matrix ────────────────────────────────────────────


class TestBuildDailyReturnMatrix:

    def test_shape(self, two_experiments):
        matrix = build_daily_return_matrix(two_experiments)
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape[1] == 2
        assert "EXP-A" in matrix.columns
        assert "EXP-B" in matrix.columns

    def test_no_nans(self, two_experiments):
        matrix = build_daily_return_matrix(two_experiments)
        assert matrix.isna().sum().sum() == 0

    def test_empty_input(self):
        matrix = build_daily_return_matrix({})
        assert matrix.empty


# ── compute_correlation ──────────────────────────────────────────────────


class TestComputeCorrelation:

    def test_symmetric(self, two_experiments):
        matrix = build_daily_return_matrix(two_experiments)
        corr = compute_correlation(matrix)
        assert corr.shape == (2, 2)
        assert abs(corr.loc["EXP-A", "EXP-B"] - corr.loc["EXP-B", "EXP-A"]) < 1e-10

    def test_diagonal_is_one(self, two_experiments):
        matrix = build_daily_return_matrix(two_experiments)
        corr = compute_correlation(matrix)
        for col in corr.columns:
            assert abs(corr.loc[col, col] - 1.0) < 1e-10

    def test_values_in_range(self, two_experiments):
        matrix = build_daily_return_matrix(two_experiments)
        corr = compute_correlation(matrix)
        assert (corr.values >= -1.0001).all()
        assert (corr.values <= 1.0001).all()

    def test_single_experiment_empty(self):
        matrix = pd.DataFrame({"A": [1, 2, 3]})
        corr = compute_correlation(matrix)
        assert corr.empty

    def test_empty_matrix(self):
        corr = compute_correlation(pd.DataFrame())
        assert corr.empty

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_real_data_correlation(self):
        trades = load_all_trades()
        matrix = build_daily_return_matrix(trades)
        corr = compute_correlation(matrix)
        assert corr.shape[0] >= 2
        assert corr.shape[0] == corr.shape[1]


# ── load_trades / load_all_trades ────────────────────────────────────────


class TestLoadTrades:

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_load_exp400(self):
        df = load_trades(EXP400_CSV)
        assert len(df) > 200
        assert "pnl" in df.columns
        assert "exit_date" in df.columns
        assert "regime" in df.columns

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_load_all_has_400_and_401(self):
        trades = load_all_trades()
        assert "EXP-400" in trades
        assert "EXP-401" in trades


# ── generate_html ────────────────────────────────────────────────────────


class TestGenerateHtml:

    @pytest.fixture
    def rendered_html(self, two_experiments):
        overall, yearly, regime = compute_all_metrics(two_experiments)
        matrix = build_daily_return_matrix(two_experiments)
        corr = compute_correlation(matrix)
        return generate_html(two_experiments, overall, yearly, regime, corr)

    def test_valid_html(self, rendered_html):
        assert rendered_html.startswith("<!DOCTYPE html>")
        assert "</html>" in rendered_html

    def test_contains_title(self, rendered_html):
        assert "Backtest Comparison Report" in rendered_html

    def test_contains_experiment_cards(self, rendered_html):
        assert "EXP-A" in rendered_html
        assert "EXP-B" in rendered_html

    def test_contains_overall_section(self, rendered_html):
        assert "Overall Performance" in rendered_html

    def test_contains_yearly_section(self, rendered_html):
        assert "Year-by-Year" in rendered_html

    def test_contains_correlation_section(self, rendered_html):
        assert "Correlation" in rendered_html

    def test_contains_regime_section(self, rendered_html):
        assert "Regime-Conditioned" in rendered_html

    def test_contains_metric_names(self, rendered_html):
        assert "Win Rate" in rendered_html
        assert "Sharpe" in rendered_html
        assert "Max DD" in rendered_html
        assert "Profit Factor" in rendered_html

    def test_contains_highlights(self, rendered_html):
        assert "Best Sharpe" in rendered_html

    def test_writes_to_file(self, rendered_html):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_report.html"
            path.write_text(rendered_html)
            assert path.exists()
            assert path.stat().st_size > 3000

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_real_data_html(self):
        trades = load_all_trades()
        overall, yearly, regime = compute_all_metrics(trades)
        matrix = build_daily_return_matrix(trades)
        corr = compute_correlation(matrix)
        html = generate_html(trades, overall, yearly, regime, corr)
        assert "EXP-400" in html
        assert "EXP-401" in html
        assert len(html) > 10000


# ── Correlation heatmap rendering ────────────────────────────────────────


class TestCorrelationHeatmap:

    def test_empty_correlation(self, two_experiments):
        from compass.backtest_compare import _correlation_heatmap_html
        html = _correlation_heatmap_html(pd.DataFrame())
        assert "Insufficient data" in html

    def test_valid_heatmap(self, two_experiments):
        from compass.backtest_compare import _correlation_heatmap_html
        matrix = build_daily_return_matrix(two_experiments)
        corr = compute_correlation(matrix)
        html = _correlation_heatmap_html(corr)
        assert "EXP-A" in html
        assert "EXP-B" in html
        assert "1.00" in html  # diagonal
