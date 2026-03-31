"""Tests for compass.dynamic_hedge – dynamic hedging engine."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.dynamic_hedge import (
    ALL_REGIMES,
    BEAR,
    BULL,
    CRASH,
    HIGH_VOL,
    LOW_VOL,
    CostBenefit,
    CrossHedge,
    DeltaHedge,
    DynamicHedgeEngine,
    HedgeConfig,
    HedgeHistory,
    HedgePnL,
    HedgeSnapshot,
    VIXCallOverlay,
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _make_returns(n: int = 200, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(rng.randn(n) * 0.01, index=idx)


def _make_correlated_returns(n: int = 200, corr: float = 0.85, seed: int = 42) -> dict:
    """Two experiments with controlled correlation."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    base = rng.randn(n) * 0.01
    noise = rng.randn(n) * 0.01
    r_a = base
    r_b = corr * base + np.sqrt(1 - corr**2) * noise
    return {
        "EXP-1": pd.Series(r_a, index=idx),
        "EXP-2": pd.Series(r_b, index=idx),
    }


def _make_uncorrelated_returns(n: int = 200, seed: int = 42) -> dict:
    return {
        "EXP-1": _make_returns(n, seed=42),
        "EXP-2": _make_returns(n, seed=99),
    }


# ── Constructor ─────────────────────────────────────────────────────────────
class TestDynamicHedgeEngineInit:
    def test_default_config(self):
        e = DynamicHedgeEngine()
        assert e.config.vix_call_budget_pct == 0.02
        assert e.config.delta_hedge_threshold == 0.10

    def test_custom_config(self):
        cfg = HedgeConfig(vix_call_budget_pct=0.05, delta_hedge_threshold=0.20)
        e = DynamicHedgeEngine(config=cfg)
        assert e.config.vix_call_budget_pct == 0.05

    def test_regime_scales_present(self):
        e = DynamicHedgeEngine()
        for r in ALL_REGIMES:
            assert r in e.config.regime_vix_scale
            assert r in e.config.regime_delta_aggression


# ── compute_hedges ──────────────────────────────────────────────────────────
class TestComputeHedges:
    def test_returns_hedge_snapshot(self):
        snap = DynamicHedgeEngine().compute_hedges(
            portfolio_value=100_000, vix=20.0, regime=BULL,
        )
        assert isinstance(snap, HedgeSnapshot)

    def test_snapshot_fields(self):
        snap = DynamicHedgeEngine().compute_hedges(
            portfolio_value=100_000, vix=25.0, regime=BEAR,
        )
        assert snap.regime == BEAR
        assert snap.vix_level == 25.0
        assert snap.vix_overlay is not None
        assert snap.delta_hedge is not None
        assert snap.cost_benefit is not None
        assert len(snap.generated_at) > 0

    def test_all_regimes_accepted(self):
        engine = DynamicHedgeEngine()
        for regime in ALL_REGIMES:
            snap = engine.compute_hedges(100_000, 20.0, regime)
            assert snap.regime == regime


# ── VIX call overlay ────────────────────────────────────────────────────────
class TestVIXCallOverlay:
    def test_contracts_nonnegative(self):
        snap = DynamicHedgeEngine().compute_hedges(100_000, 20.0, BULL)
        assert snap.vix_overlay.n_contracts >= 0

    def test_crash_regime_more_contracts(self):
        engine = DynamicHedgeEngine()
        bull = engine.compute_hedges(100_000, 20.0, BULL)
        crash = engine.compute_hedges(100_000, 20.0, CRASH)
        assert crash.vix_overlay.n_contracts >= bull.vix_overlay.n_contracts

    def test_regime_multiplier_matches(self):
        engine = DynamicHedgeEngine()
        snap = engine.compute_hedges(100_000, 20.0, HIGH_VOL)
        expected = engine.config.regime_vix_scale[HIGH_VOL]
        assert snap.vix_overlay.regime_multiplier == expected

    def test_higher_vix_higher_cost(self):
        engine = DynamicHedgeEngine()
        low = engine.compute_hedges(100_000, 15.0, BULL)
        high = engine.compute_hedges(100_000, 40.0, BULL)
        # Higher VIX = higher premium per contract
        if low.vix_overlay.n_contracts > 0 and high.vix_overlay.n_contracts > 0:
            cost_per_low = low.vix_overlay.estimated_cost / max(low.vix_overlay.n_contracts, 1)
            cost_per_high = high.vix_overlay.estimated_cost / max(high.vix_overlay.n_contracts, 1)
            assert cost_per_high >= cost_per_low

    def test_expected_payoff_positive(self):
        snap = DynamicHedgeEngine().compute_hedges(100_000, 20.0, BEAR)
        if snap.vix_overlay.n_contracts > 0:
            assert snap.vix_overlay.expected_payoff_at_spike > 0

    def test_zero_portfolio_zero_contracts(self):
        snap = DynamicHedgeEngine().compute_hedges(0, 20.0, BULL)
        assert snap.vix_overlay.n_contracts == 0


# ── Delta hedge ─────────────────────────────────────────────────────────────
class TestDeltaHedge:
    def test_no_hedge_below_threshold(self):
        snap = DynamicHedgeEngine().compute_hedges(
            100_000, 20.0, BULL, portfolio_delta=0.05,
        )
        assert snap.delta_hedge.spy_shares == 0

    def test_hedge_above_threshold(self):
        snap = DynamicHedgeEngine().compute_hedges(
            100_000, 20.0, BEAR, portfolio_delta=0.30,
        )
        assert snap.delta_hedge.spy_shares != 0

    def test_hedge_direction_opposes_delta(self):
        snap = DynamicHedgeEngine().compute_hedges(
            100_000, 20.0, BEAR, portfolio_delta=0.50,
        )
        # Positive delta → negative hedge delta (short SPY)
        assert snap.delta_hedge.hedge_delta < 0

    def test_crash_regime_full_aggression(self):
        engine = DynamicHedgeEngine()
        snap = engine.compute_hedges(100_000, 40.0, CRASH, portfolio_delta=0.50)
        assert snap.delta_hedge.regime_aggression == 1.0

    def test_bull_regime_low_aggression(self):
        engine = DynamicHedgeEngine()
        snap = engine.compute_hedges(100_000, 15.0, BULL, portfolio_delta=0.50)
        assert snap.delta_hedge.regime_aggression == 0.3

    def test_negative_delta_positive_hedge(self):
        snap = DynamicHedgeEngine().compute_hedges(
            100_000, 20.0, BEAR, portfolio_delta=-0.30,
        )
        assert snap.delta_hedge.hedge_delta > 0


# ── Cross-hedge ─────────────────────────────────────────────────────────────
class TestCrossHedge:
    def test_correlated_experiments_flagged(self):
        returns = _make_correlated_returns(corr=0.85)
        weights = {"EXP-1": 0.5, "EXP-2": 0.5}
        snap = DynamicHedgeEngine().compute_hedges(
            100_000, 20.0, BULL,
            experiment_returns=returns, experiment_weights=weights,
        )
        assert len(snap.cross_hedges) > 0
        assert snap.cross_hedges[0].hedge_action == "reduce_both"

    def test_uncorrelated_no_cross_hedge(self):
        returns = _make_uncorrelated_returns()
        weights = {"EXP-1": 0.5, "EXP-2": 0.5}
        snap = DynamicHedgeEngine().compute_hedges(
            100_000, 20.0, BULL,
            experiment_returns=returns, experiment_weights=weights,
        )
        assert len(snap.cross_hedges) == 0

    def test_cross_hedge_fields(self):
        returns = _make_correlated_returns(corr=0.90)
        weights = {"EXP-1": 0.5, "EXP-2": 0.5}
        snap = DynamicHedgeEngine().compute_hedges(
            100_000, 20.0, BULL,
            experiment_returns=returns, experiment_weights=weights,
        )
        if snap.cross_hedges:
            ch = snap.cross_hedges[0]
            assert ch.exp_a == "EXP-1"
            assert ch.exp_b == "EXP-2"
            assert ch.correlation > 0.7
            assert 0 < ch.recommended_weight_adj <= 1.0

    def test_no_experiments_no_cross_hedge(self):
        snap = DynamicHedgeEngine().compute_hedges(100_000, 20.0, BULL)
        assert snap.cross_hedges == []

    def test_single_experiment_no_cross_hedge(self):
        returns = {"EXP-1": _make_returns()}
        weights = {"EXP-1": 1.0}
        snap = DynamicHedgeEngine().compute_hedges(
            100_000, 20.0, BULL,
            experiment_returns=returns, experiment_weights=weights,
        )
        assert snap.cross_hedges == []


# ── Cost-benefit ────────────────────────────────────────────────────────────
class TestCostBenefit:
    def test_cost_benefit_present(self):
        snap = DynamicHedgeEngine().compute_hedges(100_000, 20.0, BULL)
        assert snap.cost_benefit is not None

    def test_cost_nonnegative(self):
        snap = DynamicHedgeEngine().compute_hedges(100_000, 20.0, BEAR)
        assert snap.cost_benefit.total_hedge_cost >= 0

    def test_recommendation_valid(self):
        snap = DynamicHedgeEngine().compute_hedges(100_000, 20.0, BULL)
        assert snap.cost_benefit.recommendation in ["hedge", "partial_hedge", "no_hedge"]

    def test_crash_has_high_dd_reduction(self):
        snap = DynamicHedgeEngine().compute_hedges(
            1_000_000, 40.0, CRASH, portfolio_delta=0.50,
        )
        # Crash regime should produce substantial DD reduction estimate
        assert snap.cost_benefit.expected_dd_reduction_pct > 0.10

    def test_low_vol_cheaper_than_crash(self):
        engine = DynamicHedgeEngine()
        low = engine.compute_hedges(100_000, 12.0, LOW_VOL)
        crash = engine.compute_hedges(100_000, 40.0, CRASH)
        assert low.cost_benefit.total_hedge_cost < crash.cost_benefit.total_hedge_cost


# ── Hedge P&L tracking ─────────────────────────────────────────────────────
class TestHedgePnL:
    def test_track_pnl_returns_result(self):
        engine = DynamicHedgeEngine()
        snap = engine.compute_hedges(100_000, 20.0, BEAR, portfolio_delta=0.30)
        pnl = engine.track_pnl(alpha_pnl=500, vix_change=5.0, spy_return=-0.02, snapshot=snap)
        assert isinstance(pnl, HedgePnL)

    def test_alpha_pnl_preserved(self):
        engine = DynamicHedgeEngine()
        snap = engine.compute_hedges(100_000, 20.0, BULL)
        pnl = engine.track_pnl(alpha_pnl=1000, snapshot=snap)
        assert pnl.alpha_pnl == 1000

    def test_total_pnl_sum(self):
        engine = DynamicHedgeEngine()
        snap = engine.compute_hedges(100_000, 25.0, BEAR, portfolio_delta=0.20)
        pnl = engine.track_pnl(alpha_pnl=500, vix_change=3.0, spy_return=-0.01, snapshot=snap)
        assert pnl.total_pnl == pytest.approx(
            pnl.alpha_pnl + pnl.total_hedge_pnl,
        )

    def test_vix_spike_positive_hedge_pnl(self):
        engine = DynamicHedgeEngine()
        snap = engine.compute_hedges(500_000, 25.0, CRASH, portfolio_delta=0.50)
        pnl = engine.track_pnl(alpha_pnl=-5000, vix_change=20.0, spy_return=-0.05, snapshot=snap)
        # VIX call should produce positive P&L on a big spike
        if snap.vix_overlay.n_contracts > 0:
            assert pnl.vix_call_pnl > 0

    def test_no_snapshot_returns_pnl(self):
        engine = DynamicHedgeEngine()
        pnl = engine.track_pnl(alpha_pnl=500)
        assert pnl.alpha_pnl == 500
        assert pnl.total_hedge_pnl == 0


# ── History ─────────────────────────────────────────────────────────────────
class TestHistory:
    def test_history_tracks_snapshots(self):
        engine = DynamicHedgeEngine()
        engine.compute_hedges(100_000, 20.0, BULL)
        engine.compute_hedges(100_000, 30.0, HIGH_VOL)
        history = engine.get_history()
        assert len(history.snapshots) == 2

    def test_empty_history(self):
        engine = DynamicHedgeEngine()
        history = engine.get_history()
        assert len(history.snapshots) == 0
        assert history.cumulative_pnl is None


# ── HTML report ─────────────────────────────────────────────────────────────
class TestHTMLReport:
    def test_report_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = DynamicHedgeEngine()
            snap = engine.compute_hedges(100_000, 25.0, BEAR, portfolio_delta=0.30)
            path = engine.generate_report(snap, output_path=Path(tmp) / "h.html")
            assert path.exists()
            assert path.stat().st_size > 0

    def test_report_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = DynamicHedgeEngine()
            returns = _make_correlated_returns(corr=0.85)
            weights = {"EXP-1": 0.5, "EXP-2": 0.5}
            snap = engine.compute_hedges(
                100_000, 30.0, CRASH, portfolio_delta=0.40,
                experiment_returns=returns, experiment_weights=weights,
            )
            hist = engine.get_history()
            path = engine.generate_report(snap, history=hist, output_path=Path(tmp) / "r.html")
            html = path.read_text()
            assert "Dynamic Hedge Dashboard" in html
            assert "VIX Call Overlay" in html
            assert "SPY Delta Hedge" in html
            assert "Cost" in html
            assert "Correlation" in html

    def test_report_valid_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = DynamicHedgeEngine()
            snap = engine.compute_hedges(100_000, 20.0, BULL)
            path = engine.generate_report(snap, output_path=Path(tmp) / "v.html")
            html = path.read_text()
            assert html.startswith("<!DOCTYPE html>")
            assert "</html>" in html


# ── Dataclasses ─────────────────────────────────────────────────────────────
class TestDataclasses:
    def test_hedge_config_defaults(self):
        c = HedgeConfig()
        assert c.vix_call_budget_pct == 0.02
        assert CRASH in c.regime_vix_scale

    def test_vix_call_overlay(self):
        v = VIXCallOverlay(0.04, 5, 3.0, 800.0, 5000.0, 2.0)
        assert v.n_contracts == 5

    def test_delta_hedge(self):
        d = DeltaHedge(0.3, -0.24, 0.8, -53, 0.8)
        assert d.spy_shares == -53

    def test_cross_hedge(self):
        c = CrossHedge("A", "B", 0.85, "reduce_both", 0.7)
        assert c.hedge_action == "reduce_both"

    def test_cost_benefit(self):
        cb = CostBenefit(0.03, 0.10, 0.3, 0.03, 0.07, "hedge")
        assert cb.recommendation == "hedge"

    def test_hedge_pnl_defaults(self):
        p = HedgePnL()
        assert p.alpha_pnl == 0.0
        assert p.total_pnl == 0.0

    def test_hedge_snapshot_defaults(self):
        s = HedgeSnapshot(regime=BULL, vix_level=20.0)
        assert s.cross_hedges == []

    def test_hedge_history_defaults(self):
        h = HedgeHistory()
        assert h.snapshots == []
