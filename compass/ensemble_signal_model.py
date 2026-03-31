"""
Ensemble Signal Model — Multi-classifier trade prediction with walk-forward weighting.

Combines XGBoost, RandomForest, and ExtraTrees into a weighted-average ensemble.
Each base learner is individually calibrated (CalibratedClassifierCV) so the
ensemble averages well-calibrated probabilities rather than raw scores.

Ensemble weights are set by walk-forward validation: the training data is split
into K chronological folds; each model is trained on folds 1..k and scored on
fold k+1.  The per-model AUC across held-out folds determines the voting weight.
This avoids look-ahead bias that random-shuffle CV would introduce on time-series
trade data.

Implements the same public interface as SignalModel (predict, predict_batch, train,
save, load, backtest) so it can be used as a drop-in replacement anywhere
SignalModel is accepted — including MLEnhancedStrategy and RegimeModelRouter.

Based on research:
- Dietterich (2000): Ensemble Methods in Machine Learning
- Lakshminarayanan et al. (2017): Simple and Scalable Predictive Uncertainty Estimation
"""

import json
import logging
import os
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

# sklearn 1.6+ deprecated cv='prefit' in favour of FrozenEstimator.
try:
    from sklearn.frozen import FrozenEstimator  # type: ignore[import-untyped]
    _HAS_FROZEN = True
except ImportError:
    _HAS_FROZEN = False
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from shared.indicators import sanitize_features
from shared.types import PredictionResult

logger = logging.getLogger(__name__)

# ── Walk-forward validation defaults ─────────────────────────────────────────
DEFAULT_WF_FOLDS = 5           # number of chronological folds
MIN_FOLD_SAMPLES = 30          # skip folds with fewer samples than this
EQUAL_WEIGHT_FALLBACK = True   # if walk-forward fails, fall back to equal weights

# ── Base model configs ───────────────────────────────────────────────────────
_XGB_PARAMS = {
    'objective': 'binary:logistic',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'eval_metric': 'logloss',
}

_RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}

_ET_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}


def _calibrate_prefit(estimator: object) -> CalibratedClassifierCV:
    """Wrap a pre-fitted estimator for probability calibration.

    Uses FrozenEstimator on sklearn >= 1.6 (where cv='prefit' is deprecated),
    falls back to cv='prefit' on older versions.
    """
    if _HAS_FROZEN:
        return CalibratedClassifierCV(FrozenEstimator(estimator), method='sigmoid')
    return CalibratedClassifierCV(estimator, method='sigmoid', cv='prefit')


def _build_base_models() -> List[Tuple[str, object]]:
    """Instantiate the three base classifiers.

    Returns list of (name, unfitted_estimator) tuples.
    XGBoost is skipped if the package is not installed.
    """
    models: List[Tuple[str, object]] = []
    if xgb is not None:
        models.append(('xgboost', xgb.XGBClassifier(**_XGB_PARAMS)))
    else:
        logger.warning("XGBoost unavailable — ensemble will use RF + ET only")
    models.append(('random_forest', RandomForestClassifier(**_RF_PARAMS)))
    models.append(('extra_trees', ExtraTreesClassifier(**_ET_PARAMS)))
    return models


# ── Walk-forward weight computation ──────────────────────────────────────────

def _walk_forward_weights(
    X: np.ndarray,
    y: np.ndarray,
    base_models: List[Tuple[str, object]],
    n_folds: int = DEFAULT_WF_FOLDS,
) -> Dict[str, float]:
    """Compute per-model weights via walk-forward (expanding-window) validation.

    The data is split into ``n_folds`` chronological slices.  For each fold k
    (k >= 1), models are trained on slices 0..k-1 and scored on slice k.
    The mean AUC across held-out folds sets the relative weight.

    Args:
        X: Feature matrix (n_samples, n_features), assumed chronologically ordered.
        y: Binary labels.
        base_models: List of (name, estimator) tuples.
        n_folds: Number of chronological folds.

    Returns:
        {model_name: weight} dict, weights sum to 1.0.
    """
    n = len(y)
    fold_size = n // n_folds
    if fold_size < MIN_FOLD_SAMPLES:
        logger.warning(
            "Too few samples per fold (%d) for walk-forward — using equal weights",
            fold_size,
        )
        return {name: 1.0 / len(base_models) for name, _ in base_models}

    # Accumulate per-model AUC scores across folds
    model_aucs: Dict[str, List[float]] = {name: [] for name, _ in base_models}

    for k in range(1, n_folds):
        train_end = fold_size * k
        val_end = min(fold_size * (k + 1), n)
        X_tr, y_tr = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
            continue  # skip degenerate folds

        for name, base_est in base_models:
            try:
                from sklearn.base import clone
                est = clone(base_est)
                if name == 'xgboost':
                    est.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    est.fit(X_tr, y_tr)
                proba = est.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, proba)
                model_aucs[name].append(auc)
            except Exception as exc:
                logger.warning("Walk-forward fold %d failed for %s: %s", k, name, exc)

    # Compute mean AUC per model
    mean_aucs: Dict[str, float] = {}
    for name in model_aucs:
        scores = model_aucs[name]
        if scores:
            mean_aucs[name] = float(np.mean(scores))
        else:
            mean_aucs[name] = 0.5  # chance-level fallback

    logger.info("Walk-forward AUCs: %s", {k: f"{v:.4f}" for k, v in mean_aucs.items()})

    # Convert AUC to weights: subtract 0.5 (chance level) so only above-chance
    # performance contributes, then normalise.
    edges = {name: max(0.0, auc - 0.5) for name, auc in mean_aucs.items()}
    total_edge = sum(edges.values())

    if total_edge < 1e-9:
        # All models at or below chance — equal weights
        weights = {name: 1.0 / len(base_models) for name, _ in base_models}
    else:
        weights = {name: edge / total_edge for name, edge in edges.items()}

    logger.info("Ensemble weights: %s", {k: f"{v:.3f}" for k, v in weights.items()})
    return weights


class EnsembleSignalModel:
    """
    Ensemble signal model combining XGBoost, RandomForest, and ExtraTrees.

    Each base learner is individually calibrated via CalibratedClassifierCV
    (sigmoid method on a held-out calibration set).  The ensemble prediction
    is a weighted average of calibrated probabilities, where weights are
    derived from walk-forward validation AUC.

    Drop-in replacement for SignalModel — same train/predict/save/load API.
    """

    def __init__(self, model_dir: str = 'ml/models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Per-model state: {name: calibrated_estimator}
        self.calibrated_models: Dict[str, object] = {}
        self.ensemble_weights: Dict[str, float] = {}
        self.feature_names: Optional[List[str]] = None
        self.trained: bool = False
        self.training_stats: Dict = {}
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        self._fallback_lock = threading.Lock()
        self.fallback_counter: Counter = Counter()

        logger.info("EnsembleSignalModel initialized (model_dir=%s)", model_dir)

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        calibrate: bool = True,
        save_model: bool = True,
        n_wf_folds: int = DEFAULT_WF_FOLDS,
    ) -> Dict:
        """Train the ensemble on labelled trade data.

        Steps:
            1. Hold out 20% test set (stratified).
            2. On the remaining 80%, run walk-forward validation to set weights.
            3. Retrain each base model on the full 80% training set.
            4. Calibrate each model on a held-out calibration split.
            5. Evaluate ensemble on the 20% test set.

        Args:
            features_df: Feature DataFrame (same format as SignalModel.train).
            labels:      Binary labels (1=profitable, 0=unprofitable).
            calibrate:   Whether to calibrate probabilities (recommended True).
            save_model:  Whether to persist the ensemble to disk.
            n_wf_folds:  Number of chronological folds for walk-forward weighting.

        Returns:
            Training statistics dictionary.
        """
        try:
            logger.info("Training ensemble on %d samples...", len(features_df))

            self.feature_names = list(features_df.columns)
            X = sanitize_features(features_df.values)
            y = labels

            # 1. Hold out 20% test set
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y,
            )

            # 2. Walk-forward weights (on training portion only)
            base_models = _build_base_models()
            self.ensemble_weights = _walk_forward_weights(
                X_train, y_train, base_models, n_folds=n_wf_folds,
            )

            # 3 & 4. Final training + calibration
            if calibrate:
                X_fit, X_cal, y_fit, y_cal = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
                )
            else:
                X_fit, y_fit = X_train, y_train
                X_cal, y_cal = None, None

            self.calibrated_models = {}
            per_model_stats: Dict[str, Dict] = {}

            for name, base_est in base_models:
                from sklearn.base import clone
                est = clone(base_est)

                # Train
                if name == 'xgboost':
                    # Use a small validation split for early stopping
                    X_inner, X_val, y_inner, y_val = train_test_split(
                        X_fit, y_fit, test_size=0.15, random_state=42, stratify=y_fit,
                    )
                    est.fit(
                        X_inner, y_inner,
                        eval_set=[(X_val, y_val)],
                        verbose=False,
                    )
                else:
                    est.fit(X_fit, y_fit)

                # Calibrate
                if calibrate and X_cal is not None:
                    cal_model = _calibrate_prefit(est)
                    cal_model.fit(X_cal, y_cal)
                    self.calibrated_models[name] = cal_model
                else:
                    self.calibrated_models[name] = est

                # Per-model test metrics
                model = self.calibrated_models[name]
                proba = model.predict_proba(X_test)[:, 1]
                pred = (proba > 0.5).astype(int)
                per_model_stats[name] = {
                    'test_auc': float(roc_auc_score(y_test, proba)),
                    'test_accuracy': float(accuracy_score(y_test, pred)),
                    'test_precision': float(precision_score(y_test, pred, zero_division=0)),
                    'test_recall': float(recall_score(y_test, pred, zero_division=0)),
                    'weight': self.ensemble_weights.get(name, 0.0),
                }

            # 5. Ensemble test metrics
            ensemble_proba = self._weighted_predict_proba(X_test)
            ensemble_pred = (ensemble_proba > 0.5).astype(int)

            stats = {
                'ensemble_test_auc': float(roc_auc_score(y_test, ensemble_proba)),
                'ensemble_test_accuracy': float(accuracy_score(y_test, ensemble_pred)),
                'ensemble_test_precision': float(precision_score(y_test, ensemble_pred, zero_division=0)),
                'ensemble_test_recall': float(recall_score(y_test, ensemble_pred, zero_division=0)),
                'n_train': len(X_fit),
                'n_calibration': len(X_cal) if X_cal is not None else 0,
                'n_test': len(X_test),
                'n_features': X.shape[1],
                'positive_rate': float(y.mean()),
                'n_wf_folds': n_wf_folds,
                'per_model': per_model_stats,
                'ensemble_weights': dict(self.ensemble_weights),
                # Backward-compat aliases for online_retrain.py (GAP-1, GAP-2):
                'test_auc': float(roc_auc_score(y_test, ensemble_proba)),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }

            self.training_stats = stats
            self.trained = True

            # Feature distribution stats for drift monitoring
            self.feature_means = np.mean(X, axis=0)
            self.feature_stds = np.std(X, axis=0)

            logger.info("Ensemble training complete")
            logger.info("  Ensemble Test AUC:       %.4f", stats['ensemble_test_auc'])
            logger.info("  Ensemble Test Accuracy:  %.4f", stats['ensemble_test_accuracy'])
            logger.info("  Ensemble Test Precision: %.4f", stats['ensemble_test_precision'])
            logger.info("  Ensemble Test Recall:    %.4f", stats['ensemble_test_recall'])
            for name, ms in per_model_stats.items():
                logger.info(
                    "  %s — AUC: %.4f  weight: %.3f",
                    name, ms['test_auc'], ms['weight'],
                )

            if save_model:
                self.save(
                    f"ensemble_model_{datetime.now(timezone.utc).strftime('%Y%m%d')}.joblib"
                )

            return stats

        except Exception as exc:
            logger.error("Error training ensemble: %s", exc, exc_info=True)
            return {}

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, features: Dict) -> PredictionResult:
        """Predict profitability for a single trade.

        Returns the same dict shape as SignalModel.predict:
        {prediction, probability, confidence, signal, signal_strength, timestamp}
        with an optional ``fallback`` key when the model is unavailable.
        """
        if not self.trained:
            logger.warning("Ensemble not trained, attempting load...")
            if not self.load():
                return self._get_default_prediction()

        try:
            X = self._features_to_array(features)
            if X is None:
                return self._get_default_prediction()

            self._check_feature_distribution(X)

            probability = float(self._weighted_predict_proba(X)[0])
            prediction = int(probability > 0.5)
            confidence = abs(probability - 0.5) * 2

            return {
                'prediction': prediction,
                'probability': round(probability, 4),
                'confidence': round(confidence, 4),
                'signal': (
                    'bullish' if probability > 0.55
                    else 'bearish' if probability < 0.45
                    else 'neutral'
                ),
                'signal_strength': round(probability * 100, 1),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }

        except Exception as exc:
            with self._fallback_lock:
                self.fallback_counter['predict'] += 1
                count = self.fallback_counter['predict']
            logger.error("Ensemble prediction error (fallback #%d): %s", count, exc, exc_info=True)
            if count >= 10:
                logger.critical("EnsembleSignalModel predict has fallen back %d times", count)
            return self._get_default_prediction()

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for a batch of trades.

        Args:
            features_df: Feature DataFrame with columns matching self.feature_names.

        Returns:
            1-D numpy array of probabilities.
        """
        if not self.trained:
            logger.warning("Ensemble not trained")
            return np.ones(len(features_df)) * 0.5

        try:
            X = sanitize_features(features_df[self.feature_names].values)
            self._check_feature_distribution(X)
            return self._weighted_predict_proba(X)

        except Exception as exc:
            with self._fallback_lock:
                self.fallback_counter['predict_batch'] += 1
                count = self.fallback_counter['predict_batch']
            logger.error("Ensemble batch prediction error (fallback #%d): %s", count, exc, exc_info=True)
            if count >= 10:
                logger.critical("EnsembleSignalModel predict_batch has fallen back %d times", count)
            return np.ones(len(features_df)) * 0.5

    def backtest(self, features_df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Backtest the ensemble on historical data.

        Args:
            features_df: Historical features.
            labels:      Historical binary outcomes.

        Returns:
            Backtest metrics dictionary.
        """
        try:
            if not self.trained:
                logger.error("Ensemble not trained")
                return {}

            logger.info("Backtesting ensemble on %d trades...", len(labels))

            probabilities = self.predict_batch(features_df)
            predictions = (probabilities > 0.5).astype(int)

            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            auc = roc_auc_score(labels, probabilities)

            confidence_thresholds = [0.6, 0.7, 0.8]
            threshold_results = {}
            for thresh in confidence_thresholds:
                confident_mask = np.abs(probabilities - 0.5) * 2 >= (thresh - 0.5)
                if confident_mask.sum() > 0:
                    threshold_results[f'conf_{thresh:.1f}'] = {
                        'accuracy': float(accuracy_score(labels[confident_mask], predictions[confident_mask])),
                        'count': int(confident_mask.sum()),
                        'pct_trades': float(confident_mask.sum() / len(labels) * 100),
                    }

            results = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'auc': float(auc),
                'n_trades': len(labels),
                'win_rate': float(labels.mean()),
                'threshold_results': threshold_results,
            }

            logger.info("Ensemble backtest complete — AUC: %.4f  Accuracy: %.4f", auc, accuracy)
            return results

        except Exception as exc:
            logger.error("Error in ensemble backtest: %s", exc, exc_info=True)
            return {}

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, filename: str) -> None:
        """Save the trained ensemble to disk."""
        try:
            filepath = self.model_dir / filename

            model_data = {
                'calibrated_models': self.calibrated_models,
                'ensemble_weights': self.ensemble_weights,
                'feature_names': self.feature_names,
                'training_stats': self.training_stats,
                'feature_means': self.feature_means,
                'feature_stds': self.feature_stds,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model_type': 'ensemble',
            }

            joblib.dump(model_data, filepath)

            if self.feature_means is not None and self.feature_stds is not None:
                stats_path = filepath.with_suffix('.feature_stats.json')
                stats_data = {
                    'feature_names': self.feature_names,
                    'feature_means': self.feature_means.tolist(),
                    'feature_stds': self.feature_stds.tolist(),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                }
                with open(stats_path, 'w') as f:
                    json.dump(stats_data, f, indent=2)

            logger.info("Ensemble saved to %s", filepath)

        except Exception as exc:
            logger.error("Error saving ensemble: %s", exc, exc_info=True)

    def load(self, filename: Optional[str] = None) -> bool:
        """Load a trained ensemble from disk.

        Args:
            filename: Model filename.  If None, loads the most recent
                      ``ensemble_model_*.joblib`` in model_dir.

        Returns:
            True on success, False on failure.
        """
        try:
            if filename is None:
                model_files = list(self.model_dir.glob('ensemble_model_*.joblib'))
                if not model_files:
                    logger.warning("No saved ensemble models found")
                    return False
                filepath = max(model_files, key=lambda p: p.stat().st_mtime)
            else:
                filepath = self.model_dir / filename

            if not filepath.exists():
                logger.warning("Ensemble file not found: %s", filepath)
                return False

            # Path-traversal guard (mirrors SignalModel.load SEC-DATA-03)
            resolved_path = os.path.realpath(filepath)
            expected_dir = os.path.realpath(self.model_dir)
            if not resolved_path.startswith(expected_dir + os.sep) and resolved_path != expected_dir:
                logger.error(
                    "SECURITY: Model path '%s' is outside expected directory '%s'. "
                    "Refusing to load.",
                    resolved_path, expected_dir,
                )
                return False

            logger.warning(
                "Loading serialized ensemble from disk: %s. "
                "joblib.load can execute arbitrary code — only load models from trusted sources.",
                filepath,
            )
            model_data = joblib.load(filepath)

            self.calibrated_models = model_data['calibrated_models']
            self.ensemble_weights = model_data['ensemble_weights']
            self.feature_names = model_data['feature_names']
            self.training_stats = model_data.get('training_stats', {})
            self.trained = True

            loaded_means = model_data.get('feature_means')
            loaded_stds = model_data.get('feature_stds')
            if loaded_means is not None and loaded_stds is not None:
                self.feature_means = np.asarray(loaded_means)
                self.feature_stds = np.asarray(loaded_stds)
            else:
                self.feature_means = None
                self.feature_stds = None

            # Staleness check
            model_timestamp = model_data.get('timestamp')
            if model_timestamp:
                try:
                    trained_at = datetime.fromisoformat(model_timestamp)
                    age_days = (datetime.now(timezone.utc) - trained_at).days
                    if age_days > 30:
                        logger.warning(
                            "Ensemble is %d days old (trained %s). Consider retraining.",
                            age_days, model_timestamp,
                        )
                except (ValueError, TypeError):
                    pass

            logger.info("Ensemble loaded from %s (%d models)", filepath, len(self.calibrated_models))
            return True

        except Exception as exc:
            logger.error("Error loading ensemble: %s", exc, exc_info=True)
            return False

    # ── Monitoring ───────────────────────────────────────────────────────────

    def get_fallback_stats(self) -> Dict[str, int]:
        """Return fallback counts for monitoring."""
        return dict(self.fallback_counter)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _weighted_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute weighted-average probability across all calibrated models.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            1-D array of ensemble probabilities, shape (n_samples,).
        """
        weighted_sum = np.zeros(X.shape[0])
        total_weight = 0.0

        for name, model in self.calibrated_models.items():
            weight = self.ensemble_weights.get(name, 0.0)
            if weight <= 0:
                continue
            proba = model.predict_proba(X)[:, 1]
            weighted_sum += weight * proba
            total_weight += weight

        if total_weight < 1e-9:
            # All weights zero — equal fallback
            n_models = len(self.calibrated_models)
            for name, model in self.calibrated_models.items():
                proba = model.predict_proba(X)[:, 1]
                weighted_sum += proba
            return weighted_sum / max(n_models, 1)

        return weighted_sum / total_weight

    def _features_to_array(self, features: Dict) -> Optional[np.ndarray]:
        """Convert a feature dict to a (1, n_features) numpy array.

        Mirrors SignalModel._features_to_array exactly so that predict()
        accepts the same input format.
        """
        if self.feature_names is None:
            logger.error("Feature names not set")
            return None

        try:
            feature_values = []
            missing = []
            for name in self.feature_names:
                if name not in features:
                    missing.append(name)
                    feature_values.append(0.0)
                    continue
                value = features[name]
                if value is None or np.isnan(value):
                    value = 0.0
                feature_values.append(value)
            if missing:
                logger.warning("Missing %d features (filled with 0.0): %s", len(missing), missing)

            X = np.array(feature_values).reshape(1, -1)
            return sanitize_features(X)

        except Exception as exc:
            logger.error("Error converting features to array: %s", exc, exc_info=True)
            return None

    def _check_feature_distribution(self, X: np.ndarray) -> None:
        """Log warnings for features >3 std devs from training mean."""
        try:
            if self.feature_means is None or self.feature_stds is None:
                return
            if self.feature_names is None:
                return

            for row in X:
                for i, (value, mean, std) in enumerate(
                    zip(row, self.feature_means, self.feature_stds)
                ):
                    if std == 0 or np.isnan(std) or np.isnan(mean):
                        continue
                    n_stds = abs(value - mean) / std
                    if n_stds > 3.0:
                        feat_name = (
                            self.feature_names[i]
                            if i < len(self.feature_names)
                            else f"feature_{i}"
                        )
                        logger.warning(
                            "Feature '%s' value %.3f is %.1f std devs from training mean",
                            feat_name, value, n_stds,
                        )
        except Exception:
            pass  # monitoring never breaks prediction

    @staticmethod
    def _get_default_prediction() -> PredictionResult:
        """Return neutral fallback prediction."""
        return {
            'prediction': 0,
            'probability': 0.5,
            'confidence': 0.0,
            'signal': 'neutral',
            'signal_strength': 50.0,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fallback': True,
        }
