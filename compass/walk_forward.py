"""
Walk-Forward Validation Framework for COMPASS ML Models

Chronological expanding-window validation that prevents lookahead bias.
Splits trade data into sequential folds: train on years 1..N, test on year N+1.

Usage:
    from compass.walk_forward import WalkForwardValidator, validate_model

    df = pd.read_csv("compass/training_data_exp401.csv")
    results = validate_model(df)
    print(results["aggregate"])

References:
- Bailey et al. (2014): Probability of backtest overfitting
- de Prado (2018): Advances in Financial Machine Learning, Ch. 12
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from shared.indicators import sanitize_features

logger = logging.getLogger(__name__)

# Features used by collect_training_data.py — non-target numeric columns
# suitable for ML.  Categorical columns (regime, strategy_type, spread_type)
# are one-hot encoded automatically.
NUMERIC_FEATURES = [
    "dte_at_entry",
    "hold_days",
    "day_of_week",
    "days_since_last_trade",
    "rsi_14",
    "momentum_5d_pct",
    "momentum_10d_pct",
    "vix",
    "vix_percentile_20d",
    "vix_percentile_50d",
    "vix_percentile_100d",
    "iv_rank",
    "spy_price",
    "dist_from_ma20_pct",
    "dist_from_ma50_pct",
    "dist_from_ma80_pct",
    "dist_from_ma200_pct",
    "ma20_slope_ann_pct",
    "ma50_slope_ann_pct",
    "realized_vol_atr20",
    "realized_vol_5d",
    "realized_vol_10d",
    "realized_vol_20d",
    "net_credit",
    "spread_width",
    "max_loss_per_unit",
    "otm_pct",
    "contracts",
]

CATEGORICAL_FEATURES = [
    "regime",
    "strategy_type",
    "spread_type",
]

TARGET_COL = "win"
DATE_COL = "entry_date"
RETURN_COL = "return_pct"


# ─────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    numeric_features: Sequence[str] = NUMERIC_FEATURES,
    categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
) -> pd.DataFrame:
    """Build a clean feature matrix from a training-data DataFrame.

    Selects numeric + one-hot-encoded categorical columns, fills NaNs with 0,
    and sanitizes inf values.  Returns a DataFrame with deterministic column
    order so train/test splits stay aligned.
    """
    parts: list[pd.DataFrame] = []

    # Numeric
    num_cols = [c for c in numeric_features if c in df.columns]
    num_df = df[num_cols].copy().fillna(0.0)
    parts.append(num_df)

    # Categorical → one-hot
    for col in categorical_features:
        if col not in df.columns:
            continue
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
        parts.append(dummies)

    features = pd.concat(parts, axis=1)
    # Sanitize inf / nan in the combined matrix
    clean = sanitize_features(features.values.astype(float))
    features = pd.DataFrame(clean, columns=features.columns, index=features.index)
    return features


# ─────────────────────────────────────────────────────────────────────────
# Fold result
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Metrics for a single walk-forward fold."""

    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    accuracy: float
    precision: float
    recall: float
    brier_score: float
    auc: Optional[float]
    signal_sharpe: Optional[float]
    test_win_rate: float
    predictions: np.ndarray = field(repr=False)
    probabilities: np.ndarray = field(repr=False)
    test_labels: np.ndarray = field(repr=False)
    test_returns: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold,
            "train_period": f"{self.train_start} → {self.train_end}",
            "test_period": f"{self.test_start} → {self.test_end}",
            "n_train": self.n_train,
            "n_test": self.n_test,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "brier_score": round(self.brier_score, 4),
            "auc": round(self.auc, 4) if self.auc is not None else None,
            "signal_sharpe": round(self.signal_sharpe, 4) if self.signal_sharpe is not None else None,
            "test_win_rate": round(self.test_win_rate, 4),
        }


# ─────────────────────────────────────────────────────────────────────────
# Walk-forward validator
# ─────────────────────────────────────────────────────────────────────────

class WalkForwardValidator:
    """Expanding-window walk-forward validation for trade-level ML models.

    Splits data chronologically by year into expanding training windows:
        Fold 0: train [year_0],           test [year_1]
        Fold 1: train [year_0, year_1],   test [year_2]
        ...
        Fold N: train [year_0 .. year_N], test [year_N+1]

    This mirrors realistic model deployment: you always train on all
    available history and test on the next unseen period.
    """

    def __init__(
        self,
        model: BaseEstimator,
        *,
        numeric_features: Sequence[str] = NUMERIC_FEATURES,
        categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
        target_col: str = TARGET_COL,
        date_col: str = DATE_COL,
        return_col: str = RETURN_COL,
        min_train_samples: int = 30,
    ) -> None:
        """
        Args:
            model: Any sklearn-compatible estimator with fit/predict/predict_proba.
            numeric_features: Numeric column names to use.
            categorical_features: Categorical column names to one-hot encode.
            target_col: Binary target column (1=win, 0=loss).
            date_col: Date column for chronological splitting.
            return_col: Per-trade return column for Sharpe calculation.
            min_train_samples: Skip fold if training set is smaller than this.
        """
        self.model = model
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.target_col = target_col
        self.date_col = date_col
        self.return_col = return_col
        self.min_train_samples = min_train_samples

    # ── Public API ────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute walk-forward validation over all year-based folds.

        Args:
            df: Training data DataFrame (from collect_training_data.py).
                Must contain date_col, target_col, and feature columns.

        Returns:
            Dictionary with keys:
                folds     — list of FoldResult.to_dict()
                aggregate — averaged metrics across all folds
                n_folds   — number of folds evaluated
                oos_predictions — concatenated out-of-sample predictions
        """
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        years = sorted(df[self.date_col].dt.year.unique())
        if len(years) < 2:
            raise ValueError(
                f"Need at least 2 distinct years for walk-forward; got {years}"
            )

        # Build feature matrix once on full data to get consistent one-hot columns
        features_full = prepare_features(
            df,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
        )
        feature_cols = list(features_full.columns)

        fold_results: List[FoldResult] = []
        all_oos_preds: List[np.ndarray] = []
        all_oos_probas: List[np.ndarray] = []
        all_oos_labels: List[np.ndarray] = []
        all_oos_returns: List[np.ndarray] = []

        for fold_idx in range(len(years) - 1):
            train_years = years[: fold_idx + 1]
            test_year = years[fold_idx + 1]

            train_mask = df[self.date_col].dt.year.isin(train_years)
            test_mask = df[self.date_col].dt.year == test_year

            n_train = train_mask.sum()
            n_test = test_mask.sum()

            if n_train < self.min_train_samples:
                logger.info(
                    "Fold %d: skipping, only %d train samples (min=%d)",
                    fold_idx, n_train, self.min_train_samples,
                )
                continue

            if n_test == 0:
                logger.info("Fold %d: skipping, no test samples for year %d", fold_idx, test_year)
                continue

            X_train = features_full.loc[train_mask, feature_cols].values
            y_train = df.loc[train_mask, self.target_col].values.astype(int)
            X_test = features_full.loc[test_mask, feature_cols].values
            y_test = df.loc[test_mask, self.target_col].values.astype(int)

            test_returns = None
            if self.return_col in df.columns:
                test_returns = df.loc[test_mask, self.return_col].values

            train_dates = df.loc[train_mask, self.date_col]
            test_dates = df.loc[test_mask, self.date_col]

            # Train a fresh clone per fold
            fold_model = clone(self.model)
            fold_model.fit(X_train, y_train)

            y_pred = fold_model.predict(X_test)
            y_proba = self._get_probabilities(fold_model, X_test)

            result = self._compute_fold_metrics(
                fold_idx=fold_idx,
                y_test=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                test_returns=test_returns,
                train_dates=train_dates,
                test_dates=test_dates,
                n_train=n_train,
                n_test=n_test,
            )
            fold_results.append(result)
            all_oos_preds.append(y_pred)
            all_oos_probas.append(y_proba)
            all_oos_labels.append(y_test)
            if test_returns is not None:
                all_oos_returns.append(test_returns)

            logger.info(
                "Fold %d [%s → %s] train=%d test=%d | "
                "acc=%.3f prec=%.3f rec=%.3f brier=%.3f auc=%s sharpe=%s",
                fold_idx,
                result.train_start, result.test_end,
                n_train, n_test,
                result.accuracy, result.precision, result.recall,
                result.brier_score,
                f"{result.auc:.3f}" if result.auc is not None else "N/A",
                f"{result.signal_sharpe:.3f}" if result.signal_sharpe is not None else "N/A",
            )

        if not fold_results:
            raise ValueError("No valid folds produced — check data size and min_train_samples")

        # Aggregate metrics
        aggregate = self._aggregate_metrics(fold_results)

        # Concatenate all OOS predictions
        oos = {
            "predictions": np.concatenate(all_oos_preds),
            "probabilities": np.concatenate(all_oos_probas),
            "labels": np.concatenate(all_oos_labels),
        }
        if all_oos_returns:
            oos["returns"] = np.concatenate(all_oos_returns)

        logger.info(
            "Walk-forward complete: %d folds, OOS accuracy=%.3f, OOS Brier=%.3f",
            len(fold_results),
            aggregate["accuracy_mean"],
            aggregate["brier_score_mean"],
        )

        return {
            "folds": [r.to_dict() for r in fold_results],
            "aggregate": aggregate,
            "n_folds": len(fold_results),
            "oos_predictions": oos,
        }

    # ── Internals ─────────────────────────────────────────────────────

    @staticmethod
    def _get_probabilities(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """Extract class-1 probabilities, falling back to predictions."""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2:
                return proba[:, 1]
            return proba
        # Models without predict_proba (e.g. SVM without probability=True)
        return model.predict(X).astype(float)

    def _compute_fold_metrics(
        self,
        fold_idx: int,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        test_returns: Optional[np.ndarray],
        train_dates: pd.Series,
        test_dates: pd.Series,
        n_train: int,
        n_test: int,
    ) -> FoldResult:
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        brier = brier_score_loss(y_test, y_proba)

        # AUC requires both classes present
        auc = None
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba)

        # Signal Sharpe: annualized Sharpe of returns on trades the model
        # predicted as wins (probability > 0.5).
        signal_sharpe = None
        if test_returns is not None:
            signal_mask = y_proba > 0.5
            if signal_mask.sum() >= 2:
                signal_returns = test_returns[signal_mask]
                mean_r = np.mean(signal_returns)
                std_r = np.std(signal_returns, ddof=1)
                if std_r > 0:
                    # Annualize assuming ~52 weekly trades
                    signal_sharpe = (mean_r / std_r) * np.sqrt(52)

        return FoldResult(
            fold=fold_idx,
            train_start=str(train_dates.min().date()),
            train_end=str(train_dates.max().date()),
            test_start=str(test_dates.min().date()),
            test_end=str(test_dates.max().date()),
            n_train=n_train,
            n_test=n_test,
            accuracy=acc,
            precision=prec,
            recall=rec,
            brier_score=brier,
            auc=auc,
            signal_sharpe=signal_sharpe,
            test_win_rate=float(y_test.mean()),
            predictions=y_pred,
            probabilities=y_proba,
            test_labels=y_test,
            test_returns=test_returns,
        )

    @staticmethod
    def _aggregate_metrics(folds: List[FoldResult]) -> Dict[str, Any]:
        """Compute mean and std of per-fold metrics."""
        metric_names = ["accuracy", "precision", "recall", "brier_score"]
        agg: Dict[str, Any] = {}

        for m in metric_names:
            vals = [getattr(f, m) for f in folds]
            agg[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{m}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)

        # AUC (may be None for some folds)
        auc_vals = [f.auc for f in folds if f.auc is not None]
        if auc_vals:
            agg["auc_mean"] = round(float(np.mean(auc_vals)), 4)
            agg["auc_std"] = round(float(np.std(auc_vals, ddof=1)) if len(auc_vals) > 1 else 0.0, 4)
        else:
            agg["auc_mean"] = None
            agg["auc_std"] = None

        # Signal Sharpe
        sharpe_vals = [f.signal_sharpe for f in folds if f.signal_sharpe is not None]
        if sharpe_vals:
            agg["signal_sharpe_mean"] = round(float(np.mean(sharpe_vals)), 4)
            agg["signal_sharpe_std"] = round(float(np.std(sharpe_vals, ddof=1)) if len(sharpe_vals) > 1 else 0.0, 4)
        else:
            agg["signal_sharpe_mean"] = None
            agg["signal_sharpe_std"] = None

        agg["total_oos_samples"] = sum(f.n_test for f in folds)
        agg["n_folds"] = len(folds)

        return agg


# ─────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────

def validate_model(
    df: pd.DataFrame,
    model: Optional[BaseEstimator] = None,
    *,
    min_train_samples: int = 30,
) -> Dict[str, Any]:
    """One-call walk-forward validation with sensible defaults.

    Args:
        df: Training data DataFrame (from collect_training_data.py).
        model: sklearn-compatible classifier. Defaults to XGBClassifier
               with the same hyperparameters used in SignalModel.train().
        min_train_samples: Minimum training samples per fold.

    Returns:
        Walk-forward results dict (see WalkForwardValidator.run).
    """
    if model is None:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for the default model. "
                "Install with: pip install xgboost"
            ) from exc

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            max_depth=6,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss",
        )

    validator = WalkForwardValidator(
        model=model,
        min_train_samples=min_train_samples,
    )
    return validator.run(df)
