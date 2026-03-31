"""Tests for compass/meta_learner.py — ensemble meta-learner."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compass.meta_learner import (
    BaseModelStats,
    DisagreementFeatures,
    LiftAnalysis,
    MetaLearner,
    MetaLearnerResult,
    RegimeWeights,
    WalkForwardFold,
    compute_disagreement_features,
    compute_lift,
    evaluate_base_model,
    fit_regime_weights,
    logistic_fit,
    logistic_predict_proba,
    ridge_fit,
    ridge_predict,
    walk_forward_meta,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_data(
    n: int = 300, n_models: int = 3, seed: int = 42, base_acc: float = 0.6,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate synthetic base model predictions and actuals."""
    rng = np.random.RandomState(seed)
    actuals = rng.randint(0, 2, n).astype(float)

    preds = {}
    for i in range(n_models):
        name = ["XGB", "RF", "LR", "MLP", "LASSO"][i % 5]
        # Each model has base_acc + noise probability of being correct
        correct = rng.random(n) < (base_acc + rng.uniform(-0.05, 0.05))
        p = np.where(correct, actuals * 0.7 + 0.15, (1 - actuals) * 0.7 + 0.15)
        p += rng.normal(0, 0.1, n)
        preds[name] = np.clip(p, 0.01, 0.99)

    predictions = pd.DataFrame(preds)
    regimes = rng.choice(["bull", "bear", "sideways"], n)
    return predictions, actuals, regimes


@pytest.fixture
def data():
    return _make_data()


@pytest.fixture
def predictions(data):
    return data[0]


@pytest.fixture
def actuals(data):
    return data[1]


@pytest.fixture
def regimes(data):
    return data[2]


@pytest.fixture
def meta_learner(predictions, actuals, regimes):
    return MetaLearner(predictions, actuals, regimes=regimes, n_folds=3)


# ── Logistic regression tests ────────────────────────────────────────────


class TestLogistic:
    def test_fit_returns_weights(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 3))
        y = (X @ [1, -0.5, 0.3] > 0).astype(float)
        w = logistic_fit(X, y)
        assert len(w) == 4  # 3 features + intercept

    def test_predict_bounded(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (50, 2))
        w = np.array([1.0, -0.5, 0.1])
        probs = logistic_predict_proba(X, w)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_fit_improves_accuracy(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (200, 3))
        y = (X @ [1, -0.5, 0.3] > 0).astype(float)
        w = logistic_fit(X, y, n_iter=300)
        preds = logistic_predict_proba(X, w)
        acc = ((preds > 0.5).astype(int) == y).mean()
        assert acc > 0.7


# ── Ridge regression tests ───────────────────────────────────────────────


class TestRidge:
    def test_fit_returns_weights(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 3))
        y = X @ [0.5, 0.3, -0.2] + rng.normal(0, 0.1, 100)
        w = ridge_fit(X, y)
        assert len(w) == 4  # 3 + intercept

    def test_predict_shape(self):
        X = np.array([[1, 2], [3, 4], [5, 6]]).astype(float)
        w = np.array([0.5, 0.3, 0.1])
        preds = ridge_predict(X, w)
        assert len(preds) == 3

    def test_regularization_shrinks_weights(self):
        rng = np.random.RandomState(42)
        X = rng.normal(0, 1, (100, 3))
        y = X @ [0.5, 0.3, -0.2] + rng.normal(0, 0.1, 100)
        w_low = ridge_fit(X, y, alpha=0.01)
        w_high = ridge_fit(X, y, alpha=100.0)
        assert np.abs(w_high[:-1]).sum() < np.abs(w_low[:-1]).sum()


# ── Disagreement tests ───────────────────────────────────────────────────


class TestDisagreement:
    def test_basic(self, predictions):
        feat, arr = compute_disagreement_features(predictions)
        assert isinstance(feat, DisagreementFeatures)
        assert feat.avg_disagreement >= 0
        assert feat.n_unanimous + feat.n_split == len(predictions)

    def test_unanimous_all_agree(self):
        df = pd.DataFrame({"A": [0.9, 0.9], "B": [0.8, 0.8]})
        feat, _ = compute_disagreement_features(df)
        assert feat.n_unanimous == 2

    def test_split_disagreement(self):
        df = pd.DataFrame({"A": [0.9, 0.1], "B": [0.1, 0.9]})
        feat, _ = compute_disagreement_features(df)
        assert feat.n_split == 2

    def test_empty(self):
        feat, arr = compute_disagreement_features(pd.DataFrame({"A": [], "B": []}))
        assert feat.avg_disagreement == 0


# ── Base model evaluation tests ──────────────────────────────────────────


class TestBaseModelEval:
    def test_perfect_model(self):
        preds = np.array([0.9, 0.1, 0.8, 0.2])
        actuals = np.array([1, 0, 1, 0])
        stats = evaluate_base_model(preds, actuals, "perfect")
        assert stats.accuracy == 1.0
        assert stats.auc_approx == 1.0

    def test_random_model(self):
        rng = np.random.RandomState(42)
        preds = rng.random(100)
        actuals = rng.randint(0, 2, 100).astype(float)
        stats = evaluate_base_model(preds, actuals, "random")
        assert 0.3 < stats.accuracy < 0.7  # roughly 50%

    def test_brier_bounded(self, predictions, actuals):
        stats = evaluate_base_model(predictions.iloc[:, 0].values, actuals, "test")
        assert 0 <= stats.brier_score <= 1.0

    def test_empty(self):
        stats = evaluate_base_model(np.array([]), np.array([]), "empty")
        assert stats.n_predictions == 0


# ── Regime weights tests ─────────────────────────────────────────────────


class TestRegimeWeights:
    def test_fit_returns_regimes(self, predictions, actuals, regimes):
        rws = fit_regime_weights(predictions, actuals, regimes)
        assert len(rws) > 0
        assert all(isinstance(rw, RegimeWeights) for rw in rws)

    def test_all_regimes_present(self, predictions, actuals, regimes):
        rws = fit_regime_weights(predictions, actuals, regimes)
        regime_names = {rw.regime for rw in rws}
        assert regime_names == set(np.unique(regimes))

    def test_weights_are_dicts(self, predictions, actuals, regimes):
        rws = fit_regime_weights(predictions, actuals, regimes)
        for rw in rws:
            assert isinstance(rw.weights, dict)
            assert set(rw.weights.keys()) == set(predictions.columns)


# ── Walk-forward tests ───────────────────────────────────────────────────


class TestWalkForward:
    def test_folds_created(self, predictions, actuals):
        folds, preds = walk_forward_meta(predictions, actuals, n_folds=3)
        assert len(folds) > 0
        assert len(preds) == len(actuals)

    def test_expanding_window(self, predictions, actuals):
        folds, _ = walk_forward_meta(predictions, actuals, n_folds=3)
        if len(folds) >= 2:
            assert folds[1].n_train > folds[0].n_train

    def test_oos_predictions_shape(self, predictions, actuals):
        _, preds = walk_forward_meta(predictions, actuals, n_folds=3)
        assert len(preds) == len(actuals)

    def test_short_data(self):
        df = pd.DataFrame({"A": [0.5, 0.6], "B": [0.4, 0.7]})
        folds, _ = walk_forward_meta(df, np.array([1, 0]), n_folds=3)
        assert folds == []

    def test_logistic_method(self, predictions, actuals):
        folds, _ = walk_forward_meta(predictions, actuals, n_folds=2, meta_method="logistic")
        assert len(folds) > 0


# ── Lift analysis tests ──────────────────────────────────────────────────


class TestLift:
    def test_positive_lift(self):
        stats = [BaseModelStats("A", 0.6, 0.65, 0.2, 3.0, 100, 0.5),
                 BaseModelStats("B", 0.55, 0.6, 0.25, 2.5, 100, 0.5)]
        lift = compute_lift(stats, 0.65, 4.0)
        assert lift.lift_pct > 0
        assert lift.best_base_name == "A"

    def test_negative_lift(self):
        stats = [BaseModelStats("A", 0.7, 0.75, 0.15, 5.0, 100, 0.5)]
        lift = compute_lift(stats, 0.65, 4.0)
        assert lift.lift_pct < 0

    def test_empty_stats(self):
        lift = compute_lift([], 0.6, 3.0)
        assert lift.best_base_name == ""


# ── Constructor tests ─────────────────────────────────────────────────────


class TestConstructor:
    def test_basic(self, predictions, actuals):
        ml = MetaLearner(predictions, actuals)
        assert ml.n_models == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            MetaLearner(pd.DataFrame(), np.array([]))

    def test_length_mismatch_raises(self, predictions):
        with pytest.raises(ValueError, match="same length"):
            MetaLearner(predictions, np.array([1, 0]))

    def test_single_model_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            MetaLearner(pd.DataFrame({"A": [0.5, 0.6]}), np.array([1, 0]))

    def test_bad_method_raises(self, predictions, actuals):
        with pytest.raises(ValueError, match="meta_method"):
            MetaLearner(predictions, actuals, meta_method="magic")


# ── Full fit tests ───────────────────────────────────────────────────────


class TestFullFit:
    def test_returns_result(self, meta_learner):
        result = meta_learner.fit()
        assert isinstance(result, MetaLearnerResult)
        assert result.n_samples == 300
        assert result.n_base_models == 3

    def test_meta_weights_sum(self, meta_learner):
        result = meta_learner.fit()
        total = sum(abs(v) for v in result.meta_weights.values())
        assert abs(total - 1.0) < 0.01

    def test_base_stats_populated(self, meta_learner):
        result = meta_learner.fit()
        assert len(result.base_model_stats) == 3
        for s in result.base_model_stats:
            assert 0 <= s.accuracy <= 1.0

    def test_regime_weights_present(self, meta_learner):
        result = meta_learner.fit()
        assert len(result.regime_weights) > 0

    def test_wf_folds_present(self, meta_learner):
        result = meta_learner.fit()
        assert len(result.walk_forward_folds) > 0

    def test_lift_computed(self, meta_learner):
        result = meta_learner.fit()
        assert isinstance(result.lift, LiftAnalysis)

    def test_predictions_shape(self, meta_learner):
        result = meta_learner.fit()
        assert len(result.meta_predictions) == 300

    def test_ridge_method(self, predictions, actuals):
        ml = MetaLearner(predictions, actuals, meta_method="ridge", n_folds=2)
        result = ml.fit()
        assert result.meta_method == "ridge"

    def test_logistic_method(self, predictions, actuals):
        ml = MetaLearner(predictions, actuals, meta_method="logistic", n_folds=2)
        result = ml.fit()
        assert result.meta_method == "logistic"


# ── HTML report tests ────────────────────────────────────────────────────


class TestHTMLReport:
    def test_generates_file(self, meta_learner):
        result = meta_learner.fit()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "ml.html"
            path = MetaLearner.generate_report(result, out)
            assert path.exists()
            content = path.read_text()
            assert "Meta-Learner" in content

    def test_contains_model_table(self, meta_learner):
        result = meta_learner.fit()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MetaLearner.generate_report(result, out)
            content = out.read_text()
            assert "XGB" in content
            assert "Accuracy" in content

    def test_contains_svg(self, meta_learner):
        result = meta_learner.fit()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MetaLearner.generate_report(result, out)
            content = out.read_text()
            assert "<svg" in content

    def test_contains_lift(self, meta_learner):
        result = meta_learner.fit()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "report.html"
            MetaLearner.generate_report(result, out)
            content = out.read_text()
            assert "Lift" in content

    def test_default_path(self, meta_learner):
        result = meta_learner.fit()
        path = MetaLearner.generate_report(result)
        assert path.exists()
        assert "meta_learner.html" in str(path)
        path.unlink(missing_ok=True)
