"""
Tests for compass/alpha_research.py — Alpha research framework.

Covers:
  - Signal zoo (all 22 signals produce correct-length arrays)
  - Individual signal functions (momentum, mean-reversion, vol, etc.)
  - Spearman IC computation (known correlation, zero-signal, short data)
  - Walk-forward IC (fold structure, output fields)
  - Turnover computation
  - IC decay and half-life
  - Marginal Sharpe contribution
  - Capacity score
  - Interaction detection (synergy, correlation)
  - AlphaResearcher (evaluate_signal, evaluate_zoo, rankings)
  - HTML report generation
  - Edge cases (short data, constant signals)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from compass.alpha_research import (
    SIGNAL_ZOO,
    AlphaResearcher,
    InteractionResult,
    ResearchResult,
    SignalDefinition,
    SignalEvaluation,
    compute_capacity_score,
    compute_ic_decay,
    compute_marginal_sharpe,
    compute_spearman_ic,
    compute_turnover,
    compute_walk_forward_ic,
    generate_report,
    test_interaction as _test_interaction,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def market_data(rng):
    """500-day synthetic market data."""
    n = 500
    returns = rng.normal(0.0004, 0.012, n)
    prices = 100 * np.cumprod(1 + returns)
    vix = 20.0 + np.cumsum(rng.normal(0, 0.3, n))
    vix = np.clip(vix, 10, 80)
    return prices, returns, vix


@pytest.fixture
def researcher(market_data):
    """Ready-to-use AlphaResearcher."""
    prices, returns, vix = market_data
    return AlphaResearcher(prices, returns, vix, n_folds=5, max_lag=10, top_n=5)


# ── Signal zoo tests ──────────────────────────────────────────────────────────

class TestSignalZoo:
    def test_zoo_has_at_least_20_signals(self):
        assert len(SIGNAL_ZOO) >= 20

    def test_all_categories_represented(self):
        cats = {sd.category for sd in SIGNAL_ZOO}
        assert "momentum" in cats
        assert "mean_reversion" in cats
        assert "volatility" in cats
        assert "cross_asset" in cats
        assert "calendar" in cats
        assert "microstructure" in cats

    def test_all_signals_produce_correct_length(self, market_data):
        prices, returns, vix = market_data
        for sd in SIGNAL_ZOO:
            signal = sd.func(prices, returns, vix)
            assert len(signal) == len(prices), f"{sd.name} length mismatch"

    def test_all_signals_finite(self, market_data):
        prices, returns, vix = market_data
        for sd in SIGNAL_ZOO:
            signal = sd.func(prices, returns, vix)
            assert np.all(np.isfinite(signal)), f"{sd.name} has non-finite values"

    def test_signal_definition_fields(self):
        sd = SIGNAL_ZOO[0]
        assert isinstance(sd, SignalDefinition)
        assert sd.name != ""
        assert sd.category != ""
        assert callable(sd.func)


# ── Individual signal tests ───────────────────────────────────────────────────

class TestIndividualSignals:
    def test_momentum_5d_zero_for_first_5(self, market_data):
        prices, returns, vix = market_data
        from compass.alpha_research import sig_momentum_5d
        sig = sig_momentum_5d(prices, returns, vix)
        assert all(sig[:5] == 0.0)

    def test_rsi_14_between_0_and_100(self, market_data):
        prices, returns, vix = market_data
        from compass.alpha_research import sig_rsi_14
        rsi = sig_rsi_14(prices, returns, vix)
        valid = rsi[14:]
        assert np.all(valid >= 0)
        assert np.all(valid <= 100)

    def test_bollinger_centered_around_zero(self, market_data):
        prices, returns, vix = market_data
        from compass.alpha_research import sig_bollinger_pctb
        bb = sig_bollinger_pctb(prices, returns, vix)
        valid = bb[20:]
        assert abs(np.mean(valid)) < 0.5

    def test_day_of_week_periodic(self, market_data):
        prices, returns, vix = market_data
        from compass.alpha_research import sig_day_of_week
        dow = sig_day_of_week(prices, returns, vix)
        # Should repeat every 5 periods
        assert abs(dow[0] - dow[5]) < 1e-10


# ── IC computation tests ─────────────────────────────────────────────────────

class TestSpearmanIC:
    def test_perfect_positive_correlation(self):
        signal = np.arange(100, dtype=float)
        returns = np.arange(100, dtype=float)
        ic = compute_spearman_ic(signal, returns)
        assert abs(ic - 1.0) < 0.01

    def test_perfect_negative_correlation(self):
        signal = np.arange(100, dtype=float)
        returns = -np.arange(100, dtype=float)
        ic = compute_spearman_ic(signal, returns)
        assert abs(ic - (-1.0)) < 0.01

    def test_zero_for_random(self, rng):
        signal = rng.randn(1000)
        returns = rng.randn(1000)
        ic = compute_spearman_ic(signal, returns)
        assert abs(ic) < 0.1  # should be near zero for independent

    def test_constant_signal_returns_zero(self):
        signal = np.ones(100)
        returns = np.arange(100, dtype=float)
        ic = compute_spearman_ic(signal, returns)
        assert ic == 0.0

    def test_short_data_returns_zero(self):
        ic = compute_spearman_ic(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert ic == 0.0


class TestWalkForwardIC:
    def test_returns_ic_result(self, rng):
        signal = rng.randn(500)
        fwd = rng.randn(500)
        ic = compute_walk_forward_ic(signal, fwd, n_folds=5)
        assert hasattr(ic, "ic_mean")
        assert hasattr(ic, "ic_std")
        assert hasattr(ic, "icir")
        assert hasattr(ic, "ic_by_fold")

    def test_n_folds_minus_one_results(self, rng):
        signal = rng.randn(500)
        fwd = rng.randn(500)
        ic = compute_walk_forward_ic(signal, fwd, n_folds=5)
        assert len(ic.ic_by_fold) == 4  # n_folds - 1

    def test_short_data_empty_folds(self):
        signal = np.arange(10, dtype=float)
        fwd = np.arange(10, dtype=float)
        ic = compute_walk_forward_ic(signal, fwd, n_folds=5)
        assert ic.ic_mean == 0.0


# ── Turnover tests ────────────────────────────────────────────────────────────

class TestTurnover:
    def test_constant_signal_zero_turnover(self):
        signal = np.ones(100)
        assert compute_turnover(signal) == 0.0

    def test_alternating_signal_full_turnover(self):
        signal = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1], dtype=float)
        assert compute_turnover(signal) == 1.0

    def test_turnover_between_zero_and_one(self, rng):
        signal = rng.randn(500)
        t = compute_turnover(signal)
        assert 0 <= t <= 1.0


# ── IC decay tests ────────────────────────────────────────────────────────────

class TestICDecay:
    def test_returns_curve_and_halflife(self, rng):
        signal = rng.randn(300)
        returns = rng.randn(300)
        curve, half_life = compute_ic_decay(signal, returns, max_lag=10)
        assert len(curve) == 10
        assert isinstance(half_life, float)

    def test_decay_curve_length_matches_max_lag(self, rng):
        signal = rng.randn(200)
        returns = rng.randn(200)
        curve, _ = compute_ic_decay(signal, returns, max_lag=15)
        assert len(curve) == 15


# ── Marginal Sharpe tests ────────────────────────────────────────────────────

class TestMarginalSharpe:
    def test_nonzero_for_correlated_signal(self):
        rng = np.random.RandomState(42)
        n = 500
        returns = rng.normal(0, 0.01, n)
        # Signal that is correlated with returns (not perfectly, but noticeably)
        signal = returns + rng.normal(0, 0.005, n)
        ms = compute_marginal_sharpe(signal, returns)
        assert ms != 0.0

    def test_zero_for_constant_signal(self, rng):
        signal = np.ones(200)
        returns = rng.randn(200) * 0.01
        ms = compute_marginal_sharpe(signal, returns)
        assert ms == 0.0


# ── Capacity score tests ─────────────────────────────────────────────────────

class TestCapacityScore:
    def test_range_0_to_100(self):
        for to in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]:
            for hl in [1.0, 5.0, 20.0, 100.0, float("inf")]:
                score = compute_capacity_score(to, hl)
                assert 0 <= score <= 100

    def test_low_turnover_high_capacity(self):
        low = compute_capacity_score(0.05, 50.0)
        high = compute_capacity_score(0.80, 2.0)
        assert low > high


# ── Interaction tests ─────────────────────────────────────────────────────────

class TestInteraction:
    def test_returns_interaction_result(self, rng):
        a = rng.randn(200)
        b = rng.randn(200)
        fwd = rng.randn(200)
        ir = _test_interaction(a, b, fwd)
        assert isinstance(ir, InteractionResult)

    def test_synergy_computation(self, rng):
        a = rng.randn(200)
        b = rng.randn(200)
        fwd = rng.randn(200)
        ir = _test_interaction(a, b, fwd)
        expected = ir.joint_ic - max(abs(ir.marginal_ic_a), abs(ir.marginal_ic_b))
        assert abs(ir.synergy - expected) < 1e-5

    def test_correlation_range(self, rng):
        a = rng.randn(200)
        b = rng.randn(200)
        fwd = rng.randn(200)
        ir = _test_interaction(a, b, fwd)
        assert -1.0 <= ir.correlation <= 1.0


# ── AlphaResearcher tests ────────────────────────────────────────────────────

class TestAlphaResearcher:
    def test_construction(self, market_data):
        prices, returns, vix = market_data
        ar = AlphaResearcher(prices, returns, vix)
        assert ar.n == len(prices)

    def test_too_few_points_raises(self, rng):
        with pytest.raises(ValueError, match="at least 50"):
            AlphaResearcher(rng.randn(10), rng.randn(10), rng.randn(10))

    def test_length_mismatch_raises(self, rng):
        with pytest.raises(ValueError, match="same length"):
            AlphaResearcher(rng.randn(100), rng.randn(50), rng.randn(100))

    def test_evaluate_signal(self, researcher):
        sd = SIGNAL_ZOO[0]
        ev = researcher.evaluate_signal(sd)
        assert isinstance(ev, SignalEvaluation)
        assert ev.signal_name == sd.name
        assert ev.category == sd.category

    def test_evaluate_zoo_returns_result(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=False)
        assert isinstance(result, ResearchResult)
        assert len(result.evaluations) == len(SIGNAL_ZOO)

    def test_rankings_assigned(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=False)
        for ev in result.evaluations:
            assert ev.rank_ic > 0
            assert ev.rank_turnover_ic > 0
            assert ev.rank_marginal_sharpe > 0
            assert ev.composite_rank > 0

    def test_sorted_by_composite_rank(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=False)
        ranks = [ev.composite_rank for ev in result.evaluations]
        assert ranks == sorted(ranks)

    def test_top_signals_populated(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=False)
        assert len(result.top_signals) <= researcher.top_n
        assert len(result.top_signals) > 0

    def test_interactions_detected(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=True, interaction_top_n=3)
        assert len(result.interactions) > 0

    def test_summary_keys(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=False)
        assert "n_signals" in result.summary
        assert "top_signal" in result.summary
        assert "best_ic" in result.summary

    def test_custom_signal_list(self, researcher):
        custom = SIGNAL_ZOO[:3]
        result = researcher.evaluate_zoo(signals=custom, test_interactions=False)
        assert len(result.evaluations) == 3


# ── HTML report tests ─────────────────────────────────────────────────────────

class TestHTMLReport:
    def test_generates_valid_html(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=True, interaction_top_n=3)
        html = researcher.generate_html(result)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_contains_ranking_table(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=False)
        html = researcher.generate_html(result)
        assert "Signal Ranking" in html
        assert "IC" in html
        assert "Turnover" in html

    def test_contains_decay_curves(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=False)
        html = researcher.generate_html(result)
        assert "IC Decay" in html
        assert "<svg" in html

    def test_contains_interactions(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=True, interaction_top_n=3)
        html = researcher.generate_html(result)
        assert "Feature Interactions" in html
        assert "Synergy" in html

    def test_contains_signal_names(self, researcher):
        result = researcher.evaluate_zoo(test_interactions=False)
        html = researcher.generate_html(result)
        assert "momentum_5d" in html


# ── Convenience function test ─────────────────────────────────────────────────

class TestGenerateReport:
    def test_writes_file(self, market_data):
        prices, returns, vix = market_data
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "alpha.html")
            result = generate_report(prices, returns, vix, output_path=path,
                                     n_folds=3, max_lag=5, top_n=3)
            assert os.path.isfile(path)
            assert isinstance(result, ResearchResult)
