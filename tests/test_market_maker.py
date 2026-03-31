"""Tests for compass/market_maker.py — market-making simulator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.market_maker import (
    AdverseSelectionMetrics,
    FillEvent,
    MMConfig,
    MarketMakerSimulator,
    PnLDecomposition,
    QuoteState,
    SimulationResult,
    SpreadAnalysis,
    compute_half_spread,
    decompose_pnl,
    detect_adverse_selection,
    fill_probability,
    optimal_spread,
    reservation_price,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _mid_prices(n: int = 500, seed: int = 42, drift: float = 0.0) -> pd.Series:
    """Generate synthetic mid prices via geometric Brownian motion."""
    rng = np.random.RandomState(seed)
    log_rets = rng.normal(drift, 0.01, n)
    prices = 5.0 * np.exp(np.cumsum(log_rets))
    return pd.Series(prices, index=pd.bdate_range("2024-01-02", periods=n), name="mid")


def _trending_prices(n: int = 500, seed: int = 42) -> pd.Series:
    """Prices with strong upward trend (tests inventory asymmetry)."""
    return _mid_prices(n, seed, drift=0.002)


@pytest.fixture
def mid_prices():
    return _mid_prices()


@pytest.fixture
def config():
    return MMConfig(gamma=0.1, k=1.5, sigma=0.01, position_limit=50)


@pytest.fixture
def simulator(config):
    return MarketMakerSimulator(config, seed=42)


# ── Avellaneda-Stoikov model tests ───────────────────────────────────────


class TestAvellanedaStoikov:
    def test_reservation_price_zero_inventory(self):
        r = reservation_price(5.0, 0, 0.1, 0.01, 0.5)
        assert r == 5.0  # no inventory adjustment

    def test_reservation_price_long_inventory(self):
        r = reservation_price(5.0, 10, 0.1, 0.01, 0.5)
        assert r < 5.0  # long inventory → lower reservation (want to sell)

    def test_reservation_price_short_inventory(self):
        r = reservation_price(5.0, -10, 0.1, 0.01, 0.5)
        assert r > 5.0  # short inventory → higher reservation (want to buy)

    def test_reservation_scales_with_gamma(self):
        r_low = reservation_price(5.0, 10, 0.01, 0.01, 0.5)
        r_high = reservation_price(5.0, 10, 1.0, 0.01, 0.5)
        assert r_low > r_high  # higher risk aversion → more adjustment

    def test_optimal_spread_positive(self):
        s = optimal_spread(0.1, 0.01, 0.5, 1.5)
        assert s > 0

    def test_optimal_spread_scales_with_sigma(self):
        s_low = optimal_spread(0.1, 0.005, 0.5, 1.5)
        s_high = optimal_spread(0.1, 0.02, 0.5, 1.5)
        assert s_high > s_low

    def test_optimal_spread_positive_all_gammas(self):
        """Spread is positive for all reasonable gamma values."""
        for g in [0.01, 0.1, 0.5, 1.0, 5.0]:
            s = optimal_spread(g, 0.01, 0.5, 1.5)
            assert s > 0


# ── Half-spread and fill probability tests ───────────────────────────────


class TestSpreadAndFill:
    def test_half_spread_floor(self):
        hs = compute_half_spread(5.0, 0.0001, 2.0)
        assert hs >= 5.0 * 2.0 / 10_000

    def test_half_spread_no_floor(self):
        hs = compute_half_spread(5.0, 1.0, 2.0)
        assert hs == 0.5  # 1.0 / 2 is way above floor

    def test_fill_prob_positive(self):
        p = fill_probability(0.01, 5.0, 1.5, 0.3)
        assert 0 < p <= 0.3

    def test_fill_prob_decreases_with_spread(self):
        p_tight = fill_probability(0.001, 5.0, 1.5, 0.3)
        p_wide = fill_probability(0.1, 5.0, 1.5, 0.3)
        assert p_tight > p_wide

    def test_fill_prob_zero_price(self):
        assert fill_probability(0.01, 0.0, 1.5, 0.3) == 0.0

    def test_fill_prob_bounded(self):
        p = fill_probability(0.0001, 5.0, 1.5, 0.3)
        assert 0 <= p <= 1.0


# ── Adverse selection tests ──────────────────────────────────────────────


class TestAdverseSelection:
    def test_empty_fills(self):
        mids = np.array([5.0, 5.1, 5.2])
        a = detect_adverse_selection([], mids)
        assert a.toxicity_score == 0.0
        assert a.n_total_fills == 0

    def test_no_adverse_on_flat(self):
        mids = np.array([5.0] * 20)
        fills = [FillEvent(step=5, side="buy", price=4.99, quantity=1,
                           mid_price=5.0, inventory_after=1)]
        a = detect_adverse_selection(fills, mids)
        assert a.n_adverse_fills == 0

    def test_adverse_on_drop_after_buy(self):
        mids = np.concatenate([np.array([5.0] * 5), np.array([4.5] * 10)])
        fills = [FillEvent(step=3, side="buy", price=4.99, quantity=1,
                           mid_price=5.0, inventory_after=1)]
        a = detect_adverse_selection(fills, mids, lookforward=5)
        assert a.n_adverse_fills == 1
        assert a.pct_adverse_fills == 1.0

    def test_toxicity_bounded(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        assert 0 <= result.adverse_selection.toxicity_score <= 1.0


# ── PnL decomposition tests ─────────────────────────────────────────────


class TestPnLDecomposition:
    def test_empty_fills(self):
        d = decompose_pnl([], np.array([5.0]), 0)
        assert d.total_pnl == 0.0
        assert d.spread_capture == 0.0

    def test_spread_capture_positive_round_trip(self):
        """Buy at bid, sell at ask → positive spread capture."""
        fills = [
            FillEvent(step=0, side="buy", price=4.99, quantity=1,
                      mid_price=5.0, inventory_after=1),
            FillEvent(step=1, side="sell", price=5.01, quantity=1,
                      mid_price=5.0, inventory_after=0),
        ]
        d = decompose_pnl(fills, np.array([5.0, 5.0]), 0)
        assert d.spread_capture > 0

    def test_components_sum_to_total(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        d = result.pnl_decomp
        component_sum = d.spread_capture + d.inventory_risk + d.adverse_selection
        assert abs(component_sum - d.total_pnl) < 0.01


# ── Simulation integration tests ─────────────────────────────────────────


class TestSimulation:
    def test_returns_result(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        assert isinstance(result, SimulationResult)
        assert result.n_steps == len(mid_prices)

    def test_fills_generated(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        assert len(result.fills) > 0

    def test_quotes_match_steps(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        assert len(result.quotes) == len(mid_prices)

    def test_inventory_path_shape(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        assert len(result.inventory_path) == len(mid_prices)

    def test_inventory_within_limits(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        assert result.max_inventory <= simulator.config.position_limit + simulator.config.lot_size
        assert result.min_inventory >= -(simulator.config.position_limit + simulator.config.lot_size)

    def test_pnl_path_shape(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        assert len(result.pnl_path) == len(mid_prices)

    def test_fill_rate_bounded(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        assert 0 <= result.fill_rate <= 1.0

    def test_bid_below_ask(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        for q in result.quotes:
            assert q.bid < q.ask

    def test_spread_analysis_populated(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        s = result.spread_analysis
        assert isinstance(s, SpreadAnalysis)
        assert s.avg_half_spread_bps > 0

    def test_too_short_raises(self, simulator):
        with pytest.raises(ValueError, match="at least 2"):
            simulator.simulate(pd.Series([5.0]))

    def test_reproducible(self, config, mid_prices):
        s1 = MarketMakerSimulator(config, seed=42)
        s2 = MarketMakerSimulator(config, seed=42)
        r1 = s1.simulate(mid_prices)
        r2 = s2.simulate(mid_prices)
        assert r1.final_pnl == r2.final_pnl
        assert len(r1.fills) == len(r2.fills)

    def test_different_seeds_differ(self, config, mid_prices):
        r1 = MarketMakerSimulator(config, seed=1).simulate(mid_prices)
        r2 = MarketMakerSimulator(config, seed=99).simulate(mid_prices)
        # Very unlikely to have same fill count with different seeds
        assert r1.final_pnl != r2.final_pnl or len(r1.fills) != len(r2.fills)


# ── Config tests ─────────────────────────────────────────────────────────


class TestConfig:
    def test_default_config(self):
        cfg = MMConfig()
        assert cfg.gamma == 0.1
        assert cfg.position_limit == 100

    def test_custom_config(self):
        cfg = MMConfig(gamma=0.5, k=2.0, position_limit=20)
        assert cfg.gamma == 0.5
        assert cfg.position_limit == 20

    def test_tighter_limits_fewer_fills(self, mid_prices):
        tight = MarketMakerSimulator(MMConfig(position_limit=5), seed=42)
        loose = MarketMakerSimulator(MMConfig(position_limit=200), seed=42)
        r_tight = tight.simulate(mid_prices)
        r_loose = loose.simulate(mid_prices)
        # Tighter limits should result in fewer or equal fills
        assert len(r_tight.fills) <= len(r_loose.fills) + 10  # small tolerance


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "mm.html"
            path = MarketMakerSimulator.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Market Maker Simulation" in content

    def test_contains_pnl_attribution(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MarketMakerSimulator.generate_report(result, out)
            content = out.read_text()
            assert "PnL Attribution" in content
            assert "Spread Capture" in content

    def test_contains_charts(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MarketMakerSimulator.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Inventory" in content

    def test_contains_spread_analysis(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MarketMakerSimulator.generate_report(result, out)
            content = out.read_text()
            assert "Spread Analysis" in content
            assert "Adverse Selection" in content

    def test_default_path(self, simulator, mid_prices):
        result = simulator.simulate(mid_prices)
        path = MarketMakerSimulator.generate_report(result)
        assert path.exists()
        assert "market_maker.html" in str(path)
        path.unlink(missing_ok=True)
