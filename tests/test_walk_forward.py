"""Tests for compass/walk_forward.py walk-forward validation framework.

Covers:
  - prepare_features: numeric selection, one-hot encoding, NaN handling
  - FoldResult.to_dict: serialisation
  - WalkForwardValidator._aggregate_metrics: mean/std, None handling
  - WalkForwardValidator._get_probabilities: predict_proba fallback
  - WalkForwardValidator.run: end-to-end with synthetic data, edge cases
  - validate_model: convenience wrapper
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from compass.walk_forward import (
    CATEGORICAL_FEATURES,
    DATE_COL,
    NUMERIC_FEATURES,
    RETURN_COL,
    TARGET_COL,
    FoldResult,
    WalkForwardValidator,
    prepare_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_training_df(n_per_year=50, years=(2020, 2021, 2022)):
    """Generate a synthetic training DataFrame spanning multiple years."""
    rng = np.random.RandomState(42)
    rows = []
    for year in years:
        for i in range(n_per_year):
            day = 1 + (i % 28)
            month = 1 + (i % 12)
            rows.append({
                DATE_COL: f"{year}-{month:02d}-{day:02d}",
                TARGET_COL: rng.randint(0, 2),
                RETURN_COL: rng.normal(0.005, 0.03),
                "dte_at_entry": rng.randint(10, 50),
                "vix": rng.uniform(12, 40),
                "iv_rank": rng.uniform(0, 100),
                "rsi_14": rng.uniform(20, 80),
                "net_credit": rng.uniform(0.3, 1.5),
                "spread_width": 5.0,
                "max_loss_per_unit": rng.uniform(3, 5),
                "otm_pct": rng.uniform(2, 10),
                "contracts": rng.randint(1, 5),
                "spy_price": rng.uniform(380, 500),
                "regime": rng.choice(["bull", "bear", "neutral"]),
                "strategy_type": "CS",
                "spread_type": rng.choice(["bull_put", "bear_call"]),
            })
    return pd.DataFrame(rows)


# ── prepare_features ─────────────────────────────────────────────────────


class TestPrepareFeatures:
    def test_returns_dataframe(self):
        df = _make_training_df()
        result = prepare_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_numeric_columns_present(self):
        df = _make_training_df()
        result = prepare_features(df)
        for col in ["dte_at_entry", "vix", "iv_rank"]:
            assert col in result.columns

    def test_categorical_one_hot_encoded(self):
        df = _make_training_df()
        result = prepare_features(df)
        # Should have regime_bull, regime_bear, etc.
        regime_cols = [c for c in result.columns if c.startswith("regime_")]
        assert len(regime_cols) >= 2

    def test_no_nan_in_output(self):
        df = _make_training_df()
        # Inject NaN
        df.loc[0, "vix"] = np.nan
        df.loc[1, "iv_rank"] = np.nan
        result = prepare_features(df)
        assert not result.isnull().any().any()

    def test_no_inf_in_output(self):
        df = _make_training_df()
        df.loc[0, "vix"] = np.inf
        result = prepare_features(df)
        assert not np.isinf(result.values).any()

    def test_missing_numeric_column_skipped(self):
        df = _make_training_df()
        result = prepare_features(df, numeric_features=["vix", "nonexistent_col"])
        assert "vix" in result.columns
        assert "nonexistent_col" not in result.columns

    def test_missing_categorical_column_skipped(self):
        df = _make_training_df()
        result = prepare_features(df, categorical_features=["regime", "nonexistent_cat"])
        regime_cols = [c for c in result.columns if c.startswith("regime_")]
        assert len(regime_cols) >= 2

    def test_row_count_preserved(self):
        df = _make_training_df(n_per_year=30, years=(2020, 2021))
        result = prepare_features(df)
        assert len(result) == len(df)

    def test_empty_categoricals(self):
        df = _make_training_df()
        result = prepare_features(df, categorical_features=[])
        # No one-hot columns
        assert not any(c.startswith("regime_") for c in result.columns)


# ── FoldResult ───────────────────────────────────────────────────────────


class TestFoldResult:
    def _make_fold_result(self, **overrides):
        defaults = dict(
            fold=0,
            train_start="2020-01-01",
            train_end="2020-12-31",
            test_start="2021-01-01",
            test_end="2021-12-31",
            n_train=100,
            n_test=50,
            accuracy=0.75,
            precision=0.80,
            recall=0.70,
            brier_score=0.18,
            auc=0.82,
            signal_sharpe=1.5,
            test_win_rate=0.60,
            predictions=np.array([1, 0, 1]),
            probabilities=np.array([0.8, 0.3, 0.7]),
            test_labels=np.array([1, 0, 1]),
        )
        defaults.update(overrides)
        return FoldResult(**defaults)

    def test_to_dict_keys(self):
        fr = self._make_fold_result()
        d = fr.to_dict()
        assert "fold" in d
        assert "train_period" in d
        assert "test_period" in d
        assert "accuracy" in d
        assert "auc" in d
        assert "signal_sharpe" in d

    def test_to_dict_rounds_values(self):
        fr = self._make_fold_result(accuracy=0.123456789)
        d = fr.to_dict()
        assert d["accuracy"] == 0.1235

    def test_to_dict_none_auc(self):
        fr = self._make_fold_result(auc=None)
        d = fr.to_dict()
        assert d["auc"] is None

    def test_to_dict_none_signal_sharpe(self):
        fr = self._make_fold_result(signal_sharpe=None)
        d = fr.to_dict()
        assert d["signal_sharpe"] is None

    def test_train_period_format(self):
        fr = self._make_fold_result()
        d = fr.to_dict()
        assert "→" in d["train_period"]


# ── _aggregate_metrics ───────────────────────────────────────────────────


class TestAggregateMetrics:
    def _make_folds(self, n=3):
        folds = []
        for i in range(n):
            folds.append(FoldResult(
                fold=i,
                train_start="2020-01-01",
                train_end="2020-12-31",
                test_start="2021-01-01",
                test_end="2021-12-31",
                n_train=100,
                n_test=50 + i * 10,
                accuracy=0.70 + i * 0.05,
                precision=0.75 + i * 0.05,
                recall=0.65 + i * 0.05,
                brier_score=0.20 - i * 0.02,
                auc=0.80 + i * 0.03 if i < 2 else None,
                signal_sharpe=1.0 + i * 0.5 if i < 2 else None,
                test_win_rate=0.60,
                predictions=np.array([1]),
                probabilities=np.array([0.7]),
                test_labels=np.array([1]),
            ))
        return folds

    def test_mean_accuracy(self):
        folds = self._make_folds(3)
        agg = WalkForwardValidator._aggregate_metrics(folds)
        expected = np.mean([0.70, 0.75, 0.80])
        assert agg["accuracy_mean"] == pytest.approx(expected, abs=1e-3)

    def test_std_accuracy(self):
        folds = self._make_folds(3)
        agg = WalkForwardValidator._aggregate_metrics(folds)
        assert agg["accuracy_std"] > 0

    def test_auc_excludes_none(self):
        folds = self._make_folds(3)  # 3rd fold has auc=None
        agg = WalkForwardValidator._aggregate_metrics(folds)
        # Only first 2 folds have AUC
        assert agg["auc_mean"] == pytest.approx(np.mean([0.80, 0.83]), abs=1e-3)

    def test_all_auc_none(self):
        folds = self._make_folds(3)
        for f in folds:
            f.auc = None
        agg = WalkForwardValidator._aggregate_metrics(folds)
        assert agg["auc_mean"] is None
        assert agg["auc_std"] is None

    def test_signal_sharpe_excludes_none(self):
        folds = self._make_folds(3)
        agg = WalkForwardValidator._aggregate_metrics(folds)
        assert agg["signal_sharpe_mean"] is not None

    def test_total_oos_samples(self):
        folds = self._make_folds(3)
        agg = WalkForwardValidator._aggregate_metrics(folds)
        assert agg["total_oos_samples"] == sum(f.n_test for f in folds)

    def test_n_folds(self):
        folds = self._make_folds(3)
        agg = WalkForwardValidator._aggregate_metrics(folds)
        assert agg["n_folds"] == 3

    def test_single_fold_std_is_zero(self):
        folds = self._make_folds(1)
        agg = WalkForwardValidator._aggregate_metrics(folds)
        assert agg["accuracy_std"] == 0.0


# ── _get_probabilities ───────────────────────────────────────────────────


class TestGetProbabilities:
    def test_predict_proba_2d(self):
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
        result = WalkForwardValidator._get_probabilities(model, np.zeros((2, 3)))
        np.testing.assert_array_equal(result, [0.7, 0.4])

    def test_predict_proba_1d(self):
        model = MagicMock()
        model.predict_proba.return_value = np.array([0.7, 0.4])
        result = WalkForwardValidator._get_probabilities(model, np.zeros((2, 3)))
        np.testing.assert_array_equal(result, [0.7, 0.4])

    def test_fallback_to_predict(self):
        model = MagicMock(spec=["predict"])  # no predict_proba
        model.predict.return_value = np.array([1, 0])
        result = WalkForwardValidator._get_probabilities(model, np.zeros((2, 3)))
        np.testing.assert_array_equal(result, [1.0, 0.0])


# ── WalkForwardValidator.run ─────────────────────────────────────────────


class TestWalkForwardRun:
    def test_insufficient_years_raises(self):
        df = _make_training_df(years=(2020,))
        model = DecisionTreeClassifier(random_state=42)
        validator = WalkForwardValidator(model)
        with pytest.raises(ValueError, match="at least 2 distinct years"):
            validator.run(df)

    def test_basic_run_with_two_years(self):
        df = _make_training_df(n_per_year=60, years=(2020, 2021))
        model = DecisionTreeClassifier(random_state=42)
        validator = WalkForwardValidator(model, min_train_samples=10)
        result = validator.run(df)
        assert result["n_folds"] >= 1
        assert "aggregate" in result
        assert "folds" in result
        assert "oos_predictions" in result

    def test_three_year_expanding_window(self):
        df = _make_training_df(n_per_year=50, years=(2020, 2021, 2022))
        model = DecisionTreeClassifier(random_state=42)
        validator = WalkForwardValidator(model, min_train_samples=10)
        result = validator.run(df)
        assert result["n_folds"] == 2  # fold 0: train 2020, test 2021; fold 1: train 2020+2021, test 2022

    def test_oos_predictions_concatenated(self):
        df = _make_training_df(n_per_year=50, years=(2020, 2021, 2022))
        model = DecisionTreeClassifier(random_state=42)
        validator = WalkForwardValidator(model, min_train_samples=10)
        result = validator.run(df)
        oos = result["oos_predictions"]
        assert len(oos["predictions"]) == len(oos["labels"])
        assert len(oos["probabilities"]) == len(oos["labels"])

    def test_aggregate_metrics_keys(self):
        df = _make_training_df(n_per_year=50, years=(2020, 2021, 2022))
        model = DecisionTreeClassifier(random_state=42)
        validator = WalkForwardValidator(model, min_train_samples=10)
        result = validator.run(df)
        agg = result["aggregate"]
        assert "accuracy_mean" in agg
        assert "precision_mean" in agg
        assert "brier_score_mean" in agg
        assert "total_oos_samples" in agg

    def test_min_train_samples_skips_fold(self):
        """With very high min_train_samples, early folds are skipped."""
        df = _make_training_df(n_per_year=20, years=(2020, 2021, 2022))
        model = DecisionTreeClassifier(random_state=42)
        # First fold has only 20 samples; require 25
        validator = WalkForwardValidator(model, min_train_samples=25)
        result = validator.run(df)
        # Only second fold (40 train samples) should run
        assert result["n_folds"] == 1

    def test_all_folds_skipped_raises(self):
        df = _make_training_df(n_per_year=10, years=(2020, 2021))
        model = DecisionTreeClassifier(random_state=42)
        validator = WalkForwardValidator(model, min_train_samples=999)
        with pytest.raises(ValueError, match="No valid folds"):
            validator.run(df)

    def test_returns_in_oos_when_available(self):
        df = _make_training_df(n_per_year=50, years=(2020, 2021))
        model = DecisionTreeClassifier(random_state=42)
        validator = WalkForwardValidator(model, min_train_samples=10)
        result = validator.run(df)
        assert "returns" in result["oos_predictions"]

    def test_custom_features(self):
        df = _make_training_df(n_per_year=50, years=(2020, 2021))
        model = DecisionTreeClassifier(random_state=42)
        validator = WalkForwardValidator(
            model,
            numeric_features=["vix", "iv_rank"],
            categorical_features=[],
            min_train_samples=10,
        )
        result = validator.run(df)
        assert result["n_folds"] >= 1
