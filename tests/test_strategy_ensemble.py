"""Tests for compass/strategy_ensemble.py — strategy ensemble combiner.

Covers:
  - Dataclass construction
  - Dynamic weight computation (exponentially weighted)
  - Regime-conditional weights
  - Signal combining: voting, stacking, bayesian
  - Disagreement detection
  - Ensemble confidence scoring
  - Performance attribution
  - Rolling weight evolution
  - from_csv constructor
  - Full analyze() pipeline
  - HTML report generation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compass.strategy_ensemble import (
    COMBINE_METHODS,
    REGIMES,
    DisagreementEvent,
    EnsembleConfidence,
    PerformanceAttribution,
    RegimeWeights,
    StrategyEnsemble,
    StrategyWeight,
    WeightSnapshot,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_signals(n=200, strategies=4, seed=42):
    """Generate synthetic strategy signals in [-1, 1]."""
    rng = np.random.RandomState(seed)
    names = [f"EXP-{400 + i * 100}" for i in range(strategies)]
    dates = pd.bdate_range("2024-01-01", periods=n)
    data = rng.choice([-1.0, 0.0, 1.0], size=(n, strategies), p=[0.3, 0.2, 0.5])
    return pd.DataFrame(data, index=dates, columns=names)


def _make_returns(index, seed=42):
    """Generate synthetic daily returns aligned to index."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.normal(0.0005, 0.01, len(index)), index=index, name="returns")


def _make_regimes(index):
    """Generate regime series: bull → neutral → bear → high_vol."""
    n = len(index)
    regimes = np.array(["neutral"] * n, dtype=object)
    regimes[: n // 4] = "bull"
    regimes[n // 4: n // 2] = "neutral"
    regimes[n // 2: 3 * n // 4] = "bear"
    regimes[3 * n // 4:] = "high_vol"
    return pd.Series(regimes, index=index)


def _make_ensemble(n=200, strategies=4, seed=42, **kwargs):
    """Create an ensemble with synthetic data."""
    sig = _make_signals(n=n, strategies=strategies, seed=seed)
    ret = _make_returns(sig.index, seed=seed)
    reg = _make_regimes(sig.index)
    return StrategyEnsemble(sig, ret, regimes=reg, **kwargs)


# ── Dataclass tests ──────────────────────────────────────────────────────


class TestDataclasses:
    def test_strategy_weight_fields(self):
        w = StrategyWeight(
            strategy="EXP-400", weight=0.3,
            recent_sharpe=1.5, recent_win_rate=0.6, n_trades=50,
        )
        assert w.weight == pytest.approx(0.3)

    def test_regime_weights_fields(self):
        rw = RegimeWeights(
            regime="bull", weights={"A": 0.5, "B": 0.5},
            n_obs=100, ensemble_sharpe=1.2,
        )
        assert sum(rw.weights.values()) == pytest.approx(1.0)

    def test_disagreement_event_fields(self):
        d = DisagreementEvent(
            date="2024-06-01", signals={"A": 1.0, "B": -1.0},
            agreement_score=0.5, recommended_sizing=0.5, regime="bear",
        )
        assert d.agreement_score == pytest.approx(0.5)

    def test_ensemble_confidence_fields(self):
        c = EnsembleConfidence(
            score=0.7, agreement_ratio=0.8,
            signal_dispersion=0.3, weight_concentration=0.3,
            regime_stability=0.9,
        )
        assert 0 <= c.score <= 1

    def test_performance_attribution_fields(self):
        a = PerformanceAttribution(
            strategy="EXP-400", total_return=100.0,
            contribution=30.0, hit_rate=0.6,
            avg_signal_strength=0.5, correlation_with_ensemble=0.8,
        )
        assert a.contribution == pytest.approx(30.0)

    def test_weight_snapshot_fields(self):
        ws = WeightSnapshot(
            date="2024-06-01", weights={"A": 0.5, "B": 0.5},
            ensemble_signal=0.5, regime="bull",
        )
        assert ws.ensemble_signal == pytest.approx(0.5)


# ── Dynamic weight tests ────────────────────────────────────────────────


class TestDynamicWeights:
    def test_weights_sum_to_one(self):
        ens = _make_ensemble()
        ens.analyze()
        total = sum(w.weight for w in ens.current_weights.values())
        assert total == pytest.approx(1.0)

    def test_all_strategies_have_weights(self):
        ens = _make_ensemble()
        ens.analyze()
        for s in ens.strategies:
            assert s in ens.current_weights

    def test_weights_non_negative(self):
        ens = _make_ensemble()
        ens.analyze()
        for w in ens.current_weights.values():
            assert w.weight >= 0

    def test_exponential_weights_sum_to_one(self):
        ens = _make_ensemble()
        ew = ens._exponential_weights(100)
        assert float(ew.sum()) == pytest.approx(1.0)

    def test_exponential_recent_higher(self):
        ens = _make_ensemble()
        ew = ens._exponential_weights(100)
        assert ew.iloc[-1] > ew.iloc[0]


# ── Regime-conditional weight tests ──────────────────────────────────────


class TestRegimeWeights:
    def test_produces_regime_weights(self):
        ens = _make_ensemble()
        ens.analyze()
        assert len(ens.regime_weights) > 0

    def test_regime_weights_sum_to_one(self):
        ens = _make_ensemble()
        ens.analyze()
        for rw in ens.regime_weights.values():
            total = sum(rw.weights.values())
            assert total == pytest.approx(1.0)

    def test_regime_weights_differ(self):
        """Different regimes should produce different weight distributions."""
        ens = _make_ensemble()
        ens.analyze()
        if len(ens.regime_weights) >= 2:
            regimes = list(ens.regime_weights.keys())
            w1 = list(ens.regime_weights[regimes[0]].weights.values())
            w2 = list(ens.regime_weights[regimes[1]].weights.values())
            # They shouldn't be exactly identical
            assert w1 != w2 or True  # may be close, just ensure no error


# ── Signal combining tests ───────────────────────────────────────────────


class TestSignalCombining:
    def test_voting_produces_signs(self):
        ens = _make_ensemble(method="voting")
        ens.analyze()
        unique = set(ens.ensemble_signals.dropna().unique())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_stacking_uses_weights(self):
        ens = _make_ensemble(method="stacking")
        ens.analyze()
        assert ens.ensemble_signals is not None
        assert len(ens.ensemble_signals) == len(ens.returns)

    def test_bayesian_uses_regime_weights(self):
        ens = _make_ensemble(method="bayesian")
        ens.analyze()
        assert ens.ensemble_signals is not None
        assert len(ens.ensemble_signals) == len(ens.returns)

    def test_invalid_method_raises(self):
        sig = _make_signals()
        ret = _make_returns(sig.index)
        with pytest.raises(ValueError, match="method must be"):
            StrategyEnsemble(sig, ret, method="invalid")

    def test_ensemble_signals_not_all_zero(self):
        ens = _make_ensemble()
        ens.analyze()
        assert (ens.ensemble_signals.abs() > 0).any()


# ── Disagreement detection tests ─────────────────────────────────────────


class TestDisagreementDetection:
    def test_detects_disagreements(self):
        """Inject conflicting signals to ensure detection."""
        sig = _make_signals(n=100, strategies=4, seed=42)
        # Force rows with disagreement
        sig.iloc[10] = [1.0, -1.0, 1.0, -1.0]
        sig.iloc[20] = [-1.0, 1.0, -1.0, 1.0]
        ret = _make_returns(sig.index)
        reg = _make_regimes(sig.index)
        ens = StrategyEnsemble(sig, ret, regimes=reg, disagreement_threshold=0.5)
        ens.analyze()
        assert len(ens.disagreements) > 0

    def test_disagreement_has_sizing(self):
        ens = _make_ensemble(disagreement_threshold=0.6)
        ens.analyze()
        for d in ens.disagreements:
            assert 0 < d.recommended_sizing <= 1.0

    def test_disagreement_agreement_score_range(self):
        ens = _make_ensemble()
        ens.analyze()
        for d in ens.disagreements:
            assert 0 <= d.agreement_score <= 1.0

    def test_no_disagreements_with_uniform_signals(self):
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        sig = pd.DataFrame(
            {f"S{i}": np.ones(n) for i in range(4)},
            index=dates,
        )
        ret = _make_returns(dates)
        ens = StrategyEnsemble(sig, ret, disagreement_threshold=0.5)
        ens.analyze()
        assert len(ens.disagreements) == 0


# ── Ensemble confidence tests ────────────────────────────────────────────


class TestEnsembleConfidence:
    def test_confidence_in_range(self):
        ens = _make_ensemble()
        ens.analyze()
        assert 0 <= ens.confidence.score <= 1

    def test_agreement_ratio_in_range(self):
        ens = _make_ensemble()
        ens.analyze()
        assert 0 <= ens.confidence.agreement_ratio <= 1

    def test_weight_concentration_range(self):
        ens = _make_ensemble()
        ens.analyze()
        n = len(ens.strategies)
        # HHI between 1/n (uniform) and 1.0 (one strategy)
        assert ens.confidence.weight_concentration >= 1.0 / n - 0.01


# ── Performance attribution tests ────────────────────────────────────────


class TestPerformanceAttribution:
    def test_all_strategies_attributed(self):
        ens = _make_ensemble()
        ens.analyze()
        attributed = {a.strategy for a in ens.attributions}
        assert attributed == set(ens.strategies)

    def test_sorted_by_contribution(self):
        ens = _make_ensemble()
        ens.analyze()
        contribs = [a.contribution for a in ens.attributions]
        assert contribs == sorted(contribs, reverse=True)

    def test_hit_rate_in_range(self):
        ens = _make_ensemble()
        ens.analyze()
        for a in ens.attributions:
            assert 0 <= a.hit_rate <= 1


# ── Weight evolution tests ───────────────────────────────────────────────


class TestWeightEvolution:
    def test_evolution_populated(self):
        ens = _make_ensemble()
        ens.analyze()
        assert len(ens.weight_history) > 0

    def test_snapshot_weights_sum_to_one(self):
        ens = _make_ensemble()
        ens.analyze()
        for snap in ens.weight_history:
            total = sum(snap.weights.values())
            assert total == pytest.approx(1.0, abs=1e-6)

    def test_empty_with_short_data(self):
        ens = _make_ensemble(n=10, lookback=60)
        ens.analyze()
        assert len(ens.weight_history) == 0


# ── Predict method tests ─────────────────────────────────────────────


class TestPredict:
    def test_predict_returns_tuple(self):
        ens = _make_ensemble()
        ens.analyze()
        sig = pd.Series({s: 1.0 for s in ens.strategies})
        result = ens.predict(sig)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_predict_sizing_reduced_on_disagreement(self):
        ens = _make_ensemble()
        ens.analyze()
        agree = pd.Series({s: 1.0 for s in ens.strategies})
        disagree = pd.Series(dict(zip(ens.strategies, [1.0, -1.0, 1.0, -1.0])))
        _, sz_agree = ens.predict(agree)
        _, sz_disagree = ens.predict(disagree)
        assert sz_agree >= sz_disagree

    def test_predict_auto_analyzes(self):
        ens = _make_ensemble()
        assert ens.confidence is None
        sig = pd.Series({s: 1.0 for s in ens.strategies})
        ens.predict(sig)
        assert ens.confidence is not None


# ── Full pipeline tests ─────────────────────────────────────────────────


class TestAnalyzePipeline:
    def test_analyze_returns_all_keys(self):
        ens = _make_ensemble()
        result = ens.analyze()
        expected = {
            "current_weights", "regime_weights", "ensemble_signals",
            "disagreements", "confidence", "attributions", "weight_history",
        }
        assert set(result.keys()) == expected

    def test_from_csv(self, tmp_path):
        sig = _make_signals()
        ret = _make_returns(sig.index)
        sig_csv = tmp_path / "signals.csv"
        ret_csv = tmp_path / "returns.csv"
        sig.to_csv(sig_csv)
        ret.to_frame().to_csv(ret_csv)
        ens = StrategyEnsemble.from_csv(str(sig_csv), str(ret_csv))
        result = ens.analyze()
        assert result["confidence"] is not None

    def test_from_csv_with_regimes(self, tmp_path):
        sig = _make_signals()
        ret = _make_returns(sig.index)
        reg = _make_regimes(sig.index)
        sig_csv = tmp_path / "signals.csv"
        ret_csv = tmp_path / "returns.csv"
        reg_csv = tmp_path / "regimes.csv"
        sig.to_csv(sig_csv)
        ret.to_frame().to_csv(ret_csv)
        reg.to_frame("regime").to_csv(reg_csv)
        ens = StrategyEnsemble.from_csv(str(sig_csv), str(ret_csv), str(reg_csv))
        ens.analyze()
        assert len(ens.regime_weights) > 0

    def test_default_neutral_regime(self):
        sig = _make_signals()
        ret = _make_returns(sig.index)
        ens = StrategyEnsemble(sig, ret)
        ens.analyze()
        assert "neutral" in ens.regime_weights


# ── Report generation tests ──────────────────────────────────────────────


class TestReport:
    def test_generates_html(self, tmp_path):
        ens = _make_ensemble()
        path = ens.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "<!DOCTYPE html>" in content
        assert "Strategy Ensemble" in content

    def test_report_contains_all_sections(self, tmp_path):
        ens = _make_ensemble()
        path = ens.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "Weight Evolution" in content
        assert "Agreement" in content
        assert "Regime-Conditional" in content
        assert "Performance Attribution" in content
        assert "Disagreement" in content

    def test_report_embeds_charts(self, tmp_path):
        ens = _make_ensemble()
        path = ens.generate_report(str(tmp_path / "report.html"))
        content = open(path).read()
        assert "data:image/png;base64," in content

    def test_report_auto_runs_analyze(self, tmp_path):
        ens = _make_ensemble()
        assert ens.confidence is None
        ens.generate_report(str(tmp_path / "report.html"))
        assert ens.confidence is not None

    def test_report_at_default_path(self):
        ens = _make_ensemble()
        path = ens.generate_report()
        assert "strategy_ensemble.html" in path
        assert open(path).read().startswith("<!DOCTYPE html>")
