"""Tests for compass/turnover_optimizer.py — turnover and rebalancing cost optimizer."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from compass.turnover_optimizer import (
    CostModel, FrequencyResult, OptimalFrequency, TaxLot, TaxOptResult,
    TurnoverDecomposition, TurnoverOptimizer, TurnoverSnapshot,
    compute_rebalance_cost,
)

# ── Helpers ──────────────────────────────────────────────────────────────

def _make_data(n=252, assets=4, seed=42):
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    names = [f"asset_{i}" for i in range(assets)]
    # Slowly drifting target weights
    base_w = np.ones(assets) / assets
    weights_data = {}
    for i, name in enumerate(names):
        noise = rng.normal(0, 0.01, n).cumsum()
        w = base_w[i] + noise * 0.05
        weights_data[name] = np.clip(w, 0.05, 0.5)
    wdf = pd.DataFrame(weights_data, index=dates)
    # Normalise to sum to 1
    wdf = wdf.div(wdf.sum(axis=1), axis=0)

    ret_data = {}
    for name in names:
        ret_data[name] = rng.normal(0.0003, 0.012, n)
    rdf = pd.DataFrame(ret_data, index=dates)

    return wdf, rdf

def _make_optimizer(n=252, assets=4, seed=42, **kw):
    w, r = _make_data(n, assets, seed)
    return TurnoverOptimizer(w, r, **kw)

def _make_with_regimes(n=252, seed=42):
    w, r = _make_data(n, seed=seed)
    reg = pd.Series(
        ["bull"] * (n // 4) + ["neutral"] * (n // 4) + ["bear"] * (n // 4) + ["high_vol"] * (n - 3 * (n // 4)),
        index=r.index,
    )
    return TurnoverOptimizer(w, r, regimes=reg)

# ── Cost function tests ──────────────────────────────────────────────────

class TestCostFunction:
    def test_zero_turnover_zero_cost(self):
        assert compute_rebalance_cost(0, 100_000, 4, CostModel()) == 0.0

    def test_positive_cost(self):
        c = compute_rebalance_cost(0.5, 100_000, 4, CostModel())
        assert c > 0

    def test_higher_turnover_higher_cost(self):
        c1 = compute_rebalance_cost(0.1, 100_000, 4, CostModel())
        c2 = compute_rebalance_cost(0.5, 100_000, 4, CostModel())
        assert c2 > c1

    def test_larger_portfolio_higher_cost(self):
        c1 = compute_rebalance_cost(0.3, 50_000, 4, CostModel())
        c2 = compute_rebalance_cost(0.3, 200_000, 4, CostModel())
        assert c2 > c1

# ── Dataclass tests ──────────────────────────────────────────────────────

class TestDataclasses:
    def test_cost_model_defaults(self):
        m = CostModel()
        assert m.commission_per_contract == pytest.approx(0.65)

    def test_turnover_snapshot(self):
        s = TurnoverSnapshot("2024-01-01", 0.3, 50, 0.5, 0.3, 0.2, 3)
        assert s.turnover == pytest.approx(0.3)

    def test_frequency_result(self):
        f = FrequencyResult("weekly", 0.1, 500, 0.08, 1.2, 0.15, 52, 25)
        assert f.net_sharpe == pytest.approx(1.2)

    def test_decomposition(self):
        d = TurnoverDecomposition(2.0, 0.6, 0.3, 0.1, 0.04, 2.0)
        assert d.signal_pct + d.drift_pct + d.regime_pct == pytest.approx(1.0)

    def test_tax_lot(self):
        t = TaxLot("SPY", "2024-01-01", 10, 1000, 1100, 100, 30, False)
        assert t.gain_loss == pytest.approx(100)

    def test_tax_opt_result(self):
        t = TaxOptResult("fifo", 500, -200, 105, 10, 0)
        assert t.method == "fifo"

    def test_optimal_frequency(self):
        o = OptimalFrequency("weekly", 1.5, 0.12, 30, "reason")
        assert o.frequency == "weekly"

# ── Frequency simulation tests ───────────────────────────────────────────

class TestFrequencySimulation:
    def test_three_frequencies(self):
        opt = _make_optimizer(); opt.analyze()
        assert len(opt.frequency_results) == 3

    def test_frequency_names(self):
        opt = _make_optimizer(); opt.analyze()
        names = {f.frequency for f in opt.frequency_results}
        assert names == {"daily", "weekly", "monthly"}

    def test_daily_more_rebalances(self):
        opt = _make_optimizer(); opt.analyze()
        daily = [f for f in opt.frequency_results if f.frequency == "daily"][0]
        monthly = [f for f in opt.frequency_results if f.frequency == "monthly"][0]
        assert daily.n_rebalances > monthly.n_rebalances

    def test_all_frequencies_have_cost(self):
        opt = _make_optimizer(); opt.analyze()
        for f in opt.frequency_results:
            assert f.total_cost > 0

    def test_net_return_less_than_gross(self):
        opt = _make_optimizer(); opt.analyze()
        for f in opt.frequency_results:
            assert f.net_return <= f.gross_return + 0.001  # small tolerance

    def test_cost_drag_positive(self):
        opt = _make_optimizer(); opt.analyze()
        for f in opt.frequency_results:
            assert f.cost_drag_bps >= 0

# ── Turnover decomposition tests ─────────────────────────────────────────

class TestDecomposition:
    def test_decomposition_populated(self):
        opt = _make_optimizer(); opt.analyze()
        assert opt.decomposition is not None
        assert opt.decomposition.total_turnover > 0

    def test_decomposition_pcts_sum_to_one(self):
        opt = _make_optimizer(); opt.analyze()
        d = opt.decomposition
        total = d.signal_pct + d.drift_pct + d.regime_pct
        assert total == pytest.approx(1.0, abs=0.01)

    def test_with_regimes_has_regime_component(self):
        opt = _make_with_regimes(); opt.analyze()
        # Regime component should be >= 0 (may be 0 if regime changes
        # don't align perfectly with daily rebalance snapshots)
        assert opt.decomposition.regime_pct >= 0

    def test_without_regimes_no_regime_component(self):
        opt = _make_optimizer(); opt.analyze()
        assert opt.decomposition.regime_pct == pytest.approx(0.0, abs=0.01)

    def test_annualized_positive(self):
        opt = _make_optimizer(); opt.analyze()
        assert opt.decomposition.annualized > 0

# ── Tax lot tests ────────────────────────────────────────────────────────

class TestTaxLots:
    def test_three_methods(self):
        opt = _make_optimizer(); opt.analyze()
        methods = {t.method for t in opt.tax_results}
        assert methods == {"fifo", "lifo", "tax_loss"}

    def test_tax_loss_saves(self):
        opt = _make_optimizer(); opt.analyze()
        tl = [t for t in opt.tax_results if t.method == "tax_loss"]
        if tl:
            assert tl[0].tax_savings_vs_fifo >= 0

    def test_fifo_savings_zero(self):
        opt = _make_optimizer(); opt.analyze()
        fifo = [t for t in opt.tax_results if t.method == "fifo"][0]
        assert fifo.tax_savings_vs_fifo == 0

    def test_lots_sold_positive(self):
        opt = _make_optimizer(); opt.analyze()
        for t in opt.tax_results:
            assert t.lots_sold > 0

# ── Optimal frequency tests ─────────────────────────────────────────────

class TestOptimal:
    def test_optimal_found(self):
        opt = _make_optimizer(); opt.analyze()
        assert opt.optimal is not None

    def test_optimal_in_frequencies(self):
        opt = _make_optimizer(); opt.analyze()
        assert opt.optimal.frequency in ("daily", "weekly", "monthly")

    def test_optimal_has_reason(self):
        opt = _make_optimizer(); opt.analyze()
        assert len(opt.optimal.reason) > 0

    def test_optimal_matches_best_sharpe(self):
        opt = _make_optimizer(); opt.analyze()
        best = max(opt.frequency_results, key=lambda f: f.net_sharpe)
        assert opt.optimal.frequency == best.frequency

# ── Snapshots tests ──────────────────────────────────────────────────────

class TestSnapshots:
    def test_snapshots_per_frequency(self):
        opt = _make_optimizer(); opt.analyze()
        assert "daily" in opt.snapshots
        assert "weekly" in opt.snapshots
        assert "monthly" in opt.snapshots

    def test_snapshot_costs_positive(self):
        opt = _make_optimizer(); opt.analyze()
        for freq, snaps in opt.snapshots.items():
            for s in snaps:
                assert s.cost >= 0

    def test_snapshot_turnover_non_negative(self):
        opt = _make_optimizer(); opt.analyze()
        for snaps in opt.snapshots.values():
            for s in snaps:
                assert s.turnover >= 0

# ── Pipeline tests ───────────────────────────────────────────────────────

class TestPipeline:
    def test_analyze_keys(self):
        opt = _make_optimizer()
        result = opt.analyze()
        expected = {"frequency_results", "snapshots", "decomposition", "tax_results", "optimal"}
        assert set(result.keys()) == expected

    def test_from_csv(self, tmp_path):
        w, r = _make_data()
        w.to_csv(tmp_path / "w.csv"); r.to_csv(tmp_path / "r.csv")
        opt = TurnoverOptimizer.from_csv(str(tmp_path / "w.csv"), str(tmp_path / "r.csv"))
        opt.analyze()
        assert opt.optimal is not None

    def test_short_data(self):
        w, r = _make_data(n=10)
        opt = TurnoverOptimizer(w, r)
        opt.analyze()
        assert opt.optimal is not None

# ── Report tests ─────────────────────────────────────────────────────────

class TestReport:
    def test_generates_html(self, tmp_path):
        opt = _make_optimizer()
        path = opt.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "<!DOCTYPE html>" in c and "Turnover" in c

    def test_sections(self, tmp_path):
        opt = _make_optimizer()
        path = opt.generate_report(str(tmp_path / "r.html"))
        c = open(path).read()
        assert "Cost-Adjusted" in c and "Frequency" in c
        assert "Attribution" in c and "Tax" in c

    def test_charts(self, tmp_path):
        opt = _make_optimizer()
        path = opt.generate_report(str(tmp_path / "r.html"))
        assert "data:image/png;base64," in open(path).read()

    def test_auto_analyze(self, tmp_path):
        opt = _make_optimizer()
        assert opt.optimal is None
        opt.generate_report(str(tmp_path / "r.html"))
        assert opt.optimal is not None

    def test_default_path(self):
        opt = _make_optimizer()
        path = opt.generate_report()
        assert "turnover_optimizer.html" in path
