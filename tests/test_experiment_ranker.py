"""Tests for compass/experiment_ranker.py — multi-criteria ranking."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.experiment_ranker import (
    DEFAULT_WEIGHTS,
    ExperimentMetrics,
    ExperimentRanker,
    NormalizedScores,
    RankerResult,
    TierClassification,
    check_demotion,
    check_promotion,
    classify_tier,
    compute_composite,
    compute_metrics_from_trades,
    normalize_metrics,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_metrics(name: str, sharpe: float = 2.0, ret: float = 50.0,
                  dd: float = 10.0, hit: float = 0.55, n: int = 50) -> ExperimentMetrics:
    return ExperimentMetrics(
        name=name, sharpe=sharpe, annual_return_pct=ret,
        max_drawdown_pct=dd, calmar=ret / max(dd, 0.01),
        sortino=sharpe * 1.3, hit_rate=hit, profit_factor=1.5,
        capacity_score=0.7, signal_decay_rate=0.01,
        oos_degradation=0.1, n_trades=n, total_pnl=ret * 1000,
    )


def _make_experiments(n: int = 5) -> list:
    return [
        _make_metrics("EXP-S", sharpe=4.0, ret=80.0, dd=5.0, hit=0.65, n=100),
        _make_metrics("EXP-A", sharpe=2.5, ret=50.0, dd=10.0, hit=0.58, n=80),
        _make_metrics("EXP-B", sharpe=1.5, ret=30.0, dd=15.0, hit=0.52, n=60),
        _make_metrics("EXP-D", sharpe=0.3, ret=5.0, dd=25.0, hit=0.42, n=40),
        _make_metrics("EXP-F", sharpe=-0.5, ret=-10.0, dd=35.0, hit=0.30, n=20),
    ][:n]


def _make_trades(n: int = 80, seed: int = 42, win_rate: float = 0.55) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    wins = rng.random(n) < win_rate
    pnl = np.where(wins, rng.uniform(50, 500, n), rng.uniform(-400, -50, n))
    return pd.DataFrame({"pnl": pnl})


@pytest.fixture
def experiments():
    return _make_experiments()


@pytest.fixture
def ranker():
    return ExperimentRanker()


# ── Metric computation tests ─────────────────────────────────────────────


class TestMetricComputation:
    def test_from_trades(self):
        trades = _make_trades(80, win_rate=0.60)
        m = compute_metrics_from_trades("TEST", trades)
        assert m.name == "TEST"
        assert m.n_trades == 80
        assert m.hit_rate > 0.4

    def test_sharpe_sign(self):
        # Mostly winning → positive Sharpe
        trades = _make_trades(100, win_rate=0.70)
        m = compute_metrics_from_trades("WIN", trades)
        assert m.sharpe > 0

    def test_losing_strategy(self):
        trades = _make_trades(100, win_rate=0.20, seed=99)
        m = compute_metrics_from_trades("LOSE", trades)
        assert m.sharpe < 1.0

    def test_empty_trades(self):
        m = compute_metrics_from_trades("EMPTY", pd.DataFrame())
        assert m.n_trades == 0
        assert m.sharpe == 0.0

    def test_drawdown_positive(self):
        trades = _make_trades(80)
        m = compute_metrics_from_trades("TEST", trades)
        assert m.max_drawdown_pct >= 0

    def test_profit_factor_capped(self):
        # All wins → PF should be capped at 10
        df = pd.DataFrame({"pnl": [100.0] * 20})
        m = compute_metrics_from_trades("WIN", df)
        assert m.profit_factor <= 10.0

    def test_capacity_score_bounded(self):
        trades = _make_trades(100)
        m = compute_metrics_from_trades("CAP", trades)
        assert 0.0 <= m.capacity_score <= 1.0

    def test_oos_degradation_bounded(self):
        trades = _make_trades(100)
        m = compute_metrics_from_trades("OOS", trades)
        assert 0.0 <= m.oos_degradation <= 1.0


# ── Normalization tests ──────────────────────────────────────────────────


class TestNormalization:
    def test_output_length(self, experiments):
        normed = normalize_metrics(experiments)
        assert len(normed) == len(experiments)

    def test_scores_bounded(self, experiments):
        normed = normalize_metrics(experiments)
        for ns in normed:
            assert 0.0 <= ns.sharpe <= 1.0
            assert 0.0 <= ns.annual_return <= 1.0
            assert 0.0 <= ns.max_drawdown <= 1.0
            assert 0.0 <= ns.hit_rate <= 1.0

    def test_best_gets_highest(self, experiments):
        normed = normalize_metrics(experiments)
        # EXP-S has highest Sharpe → should get highest normalized Sharpe
        s_idx = next(i for i, ns in enumerate(normed) if ns.name == "EXP-S")
        assert normed[s_idx].sharpe == max(ns.sharpe for ns in normed)

    def test_single_experiment(self):
        normed = normalize_metrics([_make_metrics("SOLO")])
        assert len(normed) == 1
        assert normed[0].sharpe == 0.5  # single experiment → 0.5

    def test_empty(self):
        assert normalize_metrics([]) == []


# ── Composite scoring tests ──────────────────────────────────────────────


class TestComposite:
    def test_composite_bounded(self, experiments):
        normed = normalize_metrics(experiments)
        normed = compute_composite(normed, DEFAULT_WEIGHTS)
        for ns in normed:
            assert 0.0 <= ns.composite <= 1.0

    def test_weights_influence(self):
        exps = [
            _make_metrics("HIGH_SHARPE", sharpe=5.0, ret=10.0),
            _make_metrics("HIGH_RET", sharpe=1.0, ret=100.0),
        ]
        # With Sharpe weight = 1.0, HIGH_SHARPE should rank higher
        normed = normalize_metrics(exps)
        w_sharpe = {k: 0.0 for k in DEFAULT_WEIGHTS}
        w_sharpe["sharpe"] = 1.0
        normed = compute_composite(normed, w_sharpe)
        sharpe_exp = next(ns for ns in normed if ns.name == "HIGH_SHARPE")
        ret_exp = next(ns for ns in normed if ns.name == "HIGH_RET")
        assert sharpe_exp.composite > ret_exp.composite


# ── Tier classification tests ────────────────────────────────────────────


class TestTierClassification:
    def test_s_tier(self):
        assert classify_tier(0.90) == "S"

    def test_a_tier(self):
        assert classify_tier(0.75) == "A"

    def test_b_tier(self):
        assert classify_tier(0.60) == "B"

    def test_c_tier(self):
        assert classify_tier(0.45) == "C"

    def test_d_tier(self):
        assert classify_tier(0.30) == "D"

    def test_f_tier(self):
        assert classify_tier(0.10) == "F"


# ── Promotion tests ─────────────────────────────────────────────────────


class TestPromotion:
    def test_eligible(self):
        m = _make_metrics("GOOD", sharpe=2.0, ret=50, dd=10, hit=0.55, n=50)
        ok, reasons = check_promotion(m, "A")
        assert ok is True
        assert len(reasons) == 0

    def test_not_eligible_low_tier(self):
        m = _make_metrics("BAD", sharpe=2.0, ret=50, dd=10, hit=0.55, n=50)
        ok, reasons = check_promotion(m, "D")
        assert ok is False
        assert any("tier" in r for r in reasons)

    def test_not_eligible_low_sharpe(self):
        m = _make_metrics("LOW", sharpe=0.5, ret=50, dd=10, hit=0.55, n=50)
        ok, reasons = check_promotion(m, "B")
        assert ok is False
        assert any("Sharpe" in r for r in reasons)

    def test_not_eligible_few_trades(self):
        m = _make_metrics("FEW", sharpe=2.0, ret=50, dd=10, hit=0.55, n=10)
        ok, reasons = check_promotion(m, "B")
        assert ok is False
        assert any("trades" in r for r in reasons)


# ── Demotion tests ───────────────────────────────────────────────────────


class TestDemotion:
    def test_not_triggered_good(self):
        m = _make_metrics("GOOD", sharpe=2.0, dd=10, hit=0.55)
        triggered, reasons = check_demotion(m, "A")
        assert triggered is False

    def test_triggered_bad_tier(self):
        m = _make_metrics("BAD", sharpe=2.0, dd=10, hit=0.55)
        triggered, reasons = check_demotion(m, "D")
        assert triggered is True

    def test_triggered_high_drawdown(self):
        m = _make_metrics("DD", sharpe=2.0, dd=35.0, hit=0.55)
        triggered, reasons = check_demotion(m, "B")
        assert triggered is True
        assert any("DD" in r for r in reasons)

    def test_triggered_negative_sharpe(self):
        m = _make_metrics("NEG", sharpe=-1.0, dd=10, hit=0.55)
        triggered, reasons = check_demotion(m, "B")
        assert triggered is True


# ── Full ranking tests ───────────────────────────────────────────────────


class TestFullRanking:
    def test_returns_result(self, ranker, experiments):
        result = ranker.rank(experiments)
        assert isinstance(result, RankerResult)
        assert result.n_experiments == 5

    def test_sorted_by_composite(self, ranker, experiments):
        result = ranker.rank(experiments)
        scores = [c.composite_score for c in result.classifications]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_sequential(self, ranker, experiments):
        result = ranker.rank(experiments)
        ranks = [c.rank for c in result.classifications]
        assert ranks == list(range(1, len(experiments) + 1))

    def test_tier_counts(self, ranker, experiments):
        result = ranker.rank(experiments)
        total = sum(result.tier_counts.values())
        assert total == len(experiments)

    def test_best_experiment_rank_1(self, ranker, experiments):
        result = ranker.rank(experiments)
        assert result.classifications[0].name == "EXP-S"

    def test_worst_experiment_last(self, ranker, experiments):
        result = ranker.rank(experiments)
        assert result.classifications[-1].name == "EXP-F"

    def test_empty_experiments(self, ranker):
        result = ranker.rank([])
        assert result.n_experiments == 0

    def test_history_tracked(self, ranker, experiments):
        ranker.rank(experiments)
        ranker.rank(experiments)
        result = ranker.rank(experiments)
        assert len(result.history) == 3

    def test_rank_from_trades(self):
        ranker = ExperimentRanker()
        trades = {
            "A": _make_trades(60, seed=1, win_rate=0.60),
            "B": _make_trades(50, seed=2, win_rate=0.45),
        }
        result = ranker.rank_from_trades(trades)
        assert result.n_experiments == 2
        # A should rank higher (better win rate)
        assert result.classifications[0].name == "A"

    def test_custom_weights(self, experiments):
        w = {k: 0.0 for k in DEFAULT_WEIGHTS}
        w["hit_rate"] = 1.0
        ranker = ExperimentRanker(weights=w)
        result = ranker.rank(experiments)
        # EXP-S has highest hit rate → should be #1
        assert result.classifications[0].name == "EXP-S"


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, ranker, experiments):
        result = ranker.rank(experiments)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "rank.html"
            path = ExperimentRanker.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Leaderboard" in content

    def test_contains_tiers(self, ranker, experiments):
        result = ranker.rank(experiments)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            ExperimentRanker.generate_report(result, out)
            content = out.read_text()
            assert "[S]" in content or "[A]" in content

    def test_contains_radar(self, ranker, experiments):
        result = ranker.rank(experiments)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            ExperimentRanker.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content
            assert "Radar" in content

    def test_contains_experiment_names(self, ranker, experiments):
        result = ranker.rank(experiments)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "r.html"
            ExperimentRanker.generate_report(result, out)
            content = out.read_text()
            assert "EXP-S" in content
            assert "EXP-F" in content

    def test_default_path(self, ranker, experiments):
        result = ranker.rank(experiments)
        path = ExperimentRanker.generate_report(result)
        assert path.exists()
        assert "experiment_ranker.html" in str(path)
        path.unlink(missing_ok=True)
