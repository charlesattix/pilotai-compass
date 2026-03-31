"""Tests for compass/portfolio_simulator.py — multi-experiment portfolio sim."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from compass.portfolio_simulator import (
    DailySnapshot,
    ExperimentContribution,
    PortfolioMetrics,
    PortfolioSimulator,
    RebalanceEntry,
    SimulationResult,
    compute_metrics_from_daily,
)

ROOT = Path(__file__).resolve().parent.parent
EXP400_CSV = ROOT / "compass" / "training_data_exp400.csv"
EXP401_CSV = ROOT / "compass" / "training_data_exp401.csv"


def _make_trades(n: int = 80, seed: int = 42, regime: str = "bull") -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    entry = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "entry_date": entry,
        "exit_date": entry + pd.Timedelta(days=7),
        "regime": regime,
        "strategy_type": "CS",
        "hold_days": rng.randint(5, 20, n),
        "pnl": rng.normal(60, 250, n),
        "return_pct": rng.normal(5, 25, n),
        "win": (rng.normal(60, 250, n) > 0).astype(int),
        "vix": rng.uniform(14, 30, n),
        "net_credit": rng.uniform(0.30, 1.20, n),
        "spread_width": np.full(n, 5.0),
    })


@pytest.fixture
def two_exps() -> Dict[str, pd.DataFrame]:
    return {
        "EXP-A": _make_trades(80, seed=42, regime="bull"),
        "EXP-B": _make_trades(60, seed=99, regime="bear"),
    }


# ── compute_metrics_from_daily ───────────────────────────────────────────


class TestMetrics:

    def test_positive_returns(self):
        daily = pd.Series([100, 50, 80, 60, 70])
        m = compute_metrics_from_daily(daily)
        assert m.total_return_pct > 0
        assert m.sharpe > 0

    def test_negative_returns(self):
        daily = pd.Series([-100, -50, -80, -60, -70])
        m = compute_metrics_from_daily(daily)
        assert m.total_return_pct < 0
        assert m.max_dd_pct < 0

    def test_empty_series(self):
        m = compute_metrics_from_daily(pd.Series(dtype=float))
        assert m.total_return_pct == 0.0

    def test_max_dd_computed(self):
        daily = pd.Series([100, -200, 50, -300, 100])
        m = compute_metrics_from_daily(daily)
        assert m.max_dd_pct < 0

    def test_sortino_computed(self):
        daily = pd.Series([100, -50, 80, -30, 120])
        m = compute_metrics_from_daily(daily)
        assert m.sortino != 0


# ── PortfolioSimulator init ──────────────────────────────────────────────


class TestSimulatorInit:

    def test_default_equal_weight(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        assert abs(sim.base_allocations["EXP-A"] - 0.5) < 0.01
        assert abs(sim.base_allocations["EXP-B"] - 0.5) < 0.01

    def test_custom_weights(self, two_exps):
        sim = PortfolioSimulator(
            experiments=two_exps,
            allocations={"EXP-A": 0.7, "EXP-B": 0.3},
        )
        assert sim.base_allocations["EXP-A"] == 0.7

    def test_custom_styles(self, two_exps):
        sim = PortfolioSimulator(
            experiments=two_exps,
            styles={"EXP-A": "momentum", "EXP-B": "defensive"},
        )
        assert sim.styles["EXP-A"] == "momentum"


# ── run() ────────────────────────────────────────────────────────────────


class TestSimulatorRun:

    def test_run_returns_result(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_portfolio_metrics_populated(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        sim.run()
        pm = sim.result().portfolio_metrics
        assert pm.n_trades > 0
        assert pm.sharpe != 0
        assert pm.max_dd_pct <= 0

    def test_daily_snapshots_created(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        sim.run()
        snaps = sim.result().daily_snapshots
        assert len(snaps) > 50

    def test_snapshots_have_equity(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        sim.run()
        for s in sim.result().daily_snapshots[:5]:
            assert isinstance(s, DailySnapshot)
            assert s.equity > 0

    def test_per_experiment_contributions(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        sim.run()
        per_exp = sim.result().per_experiment
        assert len(per_exp) == 2
        names = {e.name for e in per_exp}
        assert "EXP-A" in names

    def test_rebalance_log_populated(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps, rebalance_freq_weeks=1)
        sim.run()
        log = sim.result().rebalance_log
        assert len(log) >= 1
        assert isinstance(log[0], RebalanceEntry)

    def test_comparison_includes_portfolio_and_experiments(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        sim.run()
        comp = sim.result().comparison
        assert "Portfolio" in comp
        assert "EXP-A" in comp
        assert "EXP-B" in comp

    def test_empty_experiments(self):
        sim = PortfolioSimulator(experiments={})
        result = sim.run()
        assert result.portfolio_metrics.n_trades == 0

    def test_single_experiment(self):
        sim = PortfolioSimulator(experiments={"SOLO": _make_trades(40)})
        sim.run()
        assert sim.result().portfolio_metrics.n_trades == 40


# ── Regime-adaptive allocation ───────────────────────────────────────────


class TestRegimeAdaptive:

    def test_weights_change_with_regime(self, two_exps):
        sim = PortfolioSimulator(
            experiments=two_exps,
            styles={"EXP-A": "momentum", "EXP-B": "defensive"},
            regime_adaptive=True,
        )
        # Simulate different regimes
        corr = pd.DataFrame({"EXP-A": [1, 0.3], "EXP-B": [0.3, 1]}, index=["EXP-A", "EXP-B"])
        bull_w = sim._compute_weights("bull", 1.0, corr)
        bear_w = sim._compute_weights("bear", 1.0, corr)
        # In bull: momentum should be overweighted
        assert bull_w["EXP-A"] > bear_w["EXP-A"]
        # In bear: defensive should be overweighted
        assert bear_w["EXP-B"] > bull_w["EXP-B"]

    def test_event_scaling_reduces_weights(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps, event_scaling=True)
        corr = pd.DataFrame({"EXP-A": [1, 0.3], "EXP-B": [0.3, 1]}, index=["EXP-A", "EXP-B"])
        normal = sim._compute_weights("bull", 1.0, corr)
        event = sim._compute_weights("bull", 0.7, corr)
        assert sum(event.values()) < sum(normal.values())

    def test_correlation_penalty(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps, correlation_threshold=0.5)
        low_corr = pd.DataFrame({"EXP-A": [1, 0.3], "EXP-B": [0.3, 1]}, index=["EXP-A", "EXP-B"])
        high_corr = pd.DataFrame({"EXP-A": [1, 0.9], "EXP-B": [0.9, 1]}, index=["EXP-A", "EXP-B"])
        w_low = sim._compute_weights("bull", 1.0, low_corr)
        w_high = sim._compute_weights("bull", 1.0, high_corr)
        assert sum(w_high.values()) <= sum(w_low.values())

    def test_weights_normalised(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        corr = pd.DataFrame({"EXP-A": [1, 0.3], "EXP-B": [0.3, 1]}, index=["EXP-A", "EXP-B"])
        w = sim._compute_weights("bull", 1.0, corr)
        assert sum(w.values()) <= 1.01


# ── HTML report ──────────────────────────────────────────────────────────


class TestReport:

    def test_returns_html(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        sim.run()
        html = sim.generate_report()
        assert "<!DOCTYPE html>" in html

    def test_contains_sections(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        sim.run()
        html = sim.generate_report()
        assert "Contribution" in html
        assert "Portfolio vs" in html
        assert "Equity Curve" in html
        assert "Rebalance" in html

    def test_writes_to_file(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        sim.run()
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "report.html")
            sim.generate_report(p)
            assert Path(p).exists()

    def test_not_run(self):
        sim = PortfolioSimulator(experiments={"A": _make_trades(10)})
        html = sim.generate_report()
        assert "not run" in html.lower()

    def test_experiment_names_in_html(self, two_exps):
        sim = PortfolioSimulator(experiments=two_exps)
        sim.run()
        html = sim.generate_report()
        assert "EXP-A" in html
        assert "EXP-B" in html


# ── Real data integration ────────────────────────────────────────────────


class TestRealData:

    @pytest.mark.skipif(not EXP400_CSV.exists(), reason="data not available")
    def test_real_data_runs(self):
        exps = {
            "EXP-400": pd.read_csv(EXP400_CSV),
            "EXP-401": pd.read_csv(EXP401_CSV),
        }
        sim = PortfolioSimulator(
            experiments=exps,
            allocations={"EXP-400": 0.6, "EXP-401": 0.4},
            styles={"EXP-400": "defensive", "EXP-401": "momentum"},
        )
        sim.run()
        r = sim.result()
        assert r.portfolio_metrics.n_trades > 500
        assert len(r.daily_snapshots) > 100
        assert len(r.rebalance_log) >= 10
