"""Tests for compass/regime_backtest.py — regime-conditioned backtesting."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from compass.regime_backtest import (
    BacktestSummary,
    RegimeBacktester,
    RegimePerformance,
    RegimeSelection,
    SwitchingResult,
    TransitionPerformance,
    _compute_max_dd,
    _compute_sharpe,
    _detect_transitions,
    _regime_metrics,
    _simulate_switching,
)

ROOT = Path(__file__).resolve().parent.parent
EXP400_CSV = ROOT / "compass" / "training_data_exp400.csv"
EXP401_CSV = ROOT / "compass" / "training_data_exp401.csv"


def _make_trades(
    n: int = 60, seed: int = 42, regimes: list = None,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    if regimes is None:
        regimes = ["bull", "bear", "high_vol"]
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({
        "entry_date": dates,
        "exit_date": dates + pd.Timedelta(days=7),
        "regime": [regimes[i % len(regimes)] for i in range(n)],
        "strategy_type": "CS",
        "hold_days": rng.randint(5, 25, n),
        "pnl": rng.normal(50, 300, n),
        "return_pct": rng.normal(5, 30, n),
        "win": (rng.normal(50, 300, n) > 0).astype(int),
        "vix": rng.uniform(14, 35, n),
        "net_credit": rng.uniform(0.30, 1.50, n),
        "spread_width": np.full(n, 5.0),
    })


@pytest.fixture
def two_experiments() -> Dict[str, pd.DataFrame]:
    return {
        "EXP-A": _make_trades(60, seed=42),
        "EXP-B": _make_trades(50, seed=99, regimes=["bull", "bear", "low_vol"]),
    }


# ── Helpers ──────────────────────────────────────────────────────────────


class TestHelpers:

    def test_compute_sharpe_positive(self):
        pnls = pd.Series([100, 50, -30, 80, 60])
        s = _compute_sharpe(pnls)
        assert s > 0

    def test_compute_sharpe_empty(self):
        assert _compute_sharpe(pd.Series(dtype=float)) == 0.0

    def test_compute_max_dd_negative(self):
        pnls = np.array([100, -200, 50, -300, 100])
        dd = _compute_max_dd(pnls)
        assert dd < 0

    def test_compute_max_dd_all_positive(self):
        pnls = np.array([100, 200, 300])
        dd = _compute_max_dd(pnls)
        assert dd == 0.0


# ── _regime_metrics ──────────────────────────────────────────────────────


class TestRegimeMetrics:

    def test_basic_metrics(self):
        df = _make_trades(30)
        m = _regime_metrics(df[df["regime"] == "bull"], "TEST", "bull")
        assert isinstance(m, RegimePerformance)
        assert m.n_trades > 0
        assert m.experiment == "TEST"
        assert m.regime == "bull"

    def test_empty_df(self):
        df = pd.DataFrame(columns=["pnl", "hold_days", "exit_date"])
        m = _regime_metrics(df, "TEST", "bear")
        assert m.n_trades == 0
        assert m.win_rate == 0.0

    def test_win_rate_range(self):
        df = _make_trades(50)
        m = _regime_metrics(df, "TEST", "all")
        assert 0 <= m.win_rate <= 100


# ── _detect_transitions ─────────────────────────────────────────────────


class TestDetectTransitions:

    def test_finds_transitions(self):
        df = _make_trades(30, regimes=["bull", "bear", "bull"])
        trans = _detect_transitions(df)
        assert len(trans) >= 2

    def test_no_transitions_single_regime(self):
        df = _make_trades(20, regimes=["bull"])
        assert len(_detect_transitions(df)) == 0

    def test_transition_has_correct_format(self):
        df = _make_trades(30, regimes=["bull", "bear"])
        trans = _detect_transitions(df)
        assert len(trans) > 0
        from_r, to_r, date = trans[0]
        assert from_r == "bull"
        assert to_r == "bear"


# ── RegimeBacktester ─────────────────────────────────────────────────────


class TestRegimeBacktester:

    def test_fit_returns_self(self, two_experiments):
        bt = RegimeBacktester()
        assert bt.fit(two_experiments) is bt

    def test_summary_populated(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        s = bt.summary()
        assert isinstance(s, BacktestSummary)
        assert len(s.experiments) == 2

    def test_regime_perf_has_entries(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        assert len(bt.summary().regime_perf) >= 4

    def test_selections_populated(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        sels = bt.summary().selections
        assert len(sels) >= 2
        regimes = {s.regime for s in sels}
        assert "bull" in regimes

    def test_switching_result(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        sw = bt.summary().switching
        assert sw is not None
        assert sw.n_trades > 0
        assert len(sw.equity_curve) > 0

    def test_transitions_detected(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        # May have transitions if regimes alternate
        assert isinstance(bt.summary().transitions, list)

    def test_unfitted_summary_empty(self):
        bt = RegimeBacktester()
        s = bt.summary()
        assert len(s.regime_perf) == 0

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_real_data(self):
        trades = {}
        if EXP400_CSV.exists():
            trades["EXP-400"] = pd.read_csv(EXP400_CSV)
        if EXP401_CSV.exists():
            trades["EXP-401"] = pd.read_csv(EXP401_CSV)
        bt = RegimeBacktester()
        bt.fit(trades)
        s = bt.summary()
        assert s.switching.n_trades > 50
        assert len(s.selections) >= 2


# ── HTML report ──────────────────────────────────────────────────────────


class TestReport:

    def test_returns_html(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        html = bt.generate_report()
        assert "<!DOCTYPE html>" in html
        assert "Regime Backtest" in html

    def test_contains_sections(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        html = bt.generate_report()
        assert "Per-Regime Performance" in html
        assert "Optimal Strategy" in html
        assert "Transition" in html
        assert "Switching" in html

    def test_writes_to_file(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "report.html")
            bt.generate_report(p)
            assert Path(p).exists()
            assert Path(p).stat().st_size > 2000

    def test_unfitted(self):
        html = RegimeBacktester().generate_report()
        assert "No data" in html

    def test_experiment_names_in_html(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        html = bt.generate_report()
        assert "EXP-A" in html
        assert "EXP-B" in html


# ── Edge cases and integration ───────────────────────────────────────────


class TestEdgeCases:

    def test_single_experiment(self):
        bt = RegimeBacktester()
        bt.fit({"SOLO": _make_trades(40)})
        s = bt.summary()
        assert len(s.experiments) == 1
        assert s.switching.n_trades > 0

    def test_single_regime(self):
        bt = RegimeBacktester()
        bt.fit({"A": _make_trades(30, regimes=["bull"])})
        s = bt.summary()
        assert len(s.selections) == 1
        assert s.selections[0].regime == "bull"

    def test_switching_equity_curve_monotonic_dates(self, two_experiments):
        bt = RegimeBacktester()
        bt.fit(two_experiments)
        curve = bt.summary().switching.equity_curve
        if len(curve) > 1:
            dates = [d for d, _ in curve]
            assert dates == sorted(dates)
