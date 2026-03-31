"""
IBIT-specific ML signal model for EXP-601.

Adapts the SignalModel pattern for crypto/IBIT characteristics:
  - Uses crypto-native features from :mod:`compass.ibit_features`
  - Higher VIX thresholds for regime detection (BTC is 3-4× more volatile)
  - Weekend gap awareness (BTC trades 24/7 but IBIT doesn't)
  - Walk-forward validation with IBIT-appropriate time windows
  - Feature importance ranking specific to IBIT dynamics
  - Save/load interface compatible with :class:`ml.regime_model_router.RegimeModelRouter`

Usage::

    from compass.ibit_signal_model import IBITSignalModel
    model = IBITSignalModel(model_dir="ml/models/ibit")
    stats = model.train(features_df, labels)
    result = model.predict(features_dict)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from compass.ibit_features import IBIT_FEATURE_NAMES, IBITFeatureEngine
from shared.indicators import sanitize_features

logger = logging.getLogger(__name__)


# ── IBIT-specific XGBoost hyperparameters ────────────────────────────────
# Tuned for small sample sizes (249 IBIT trades) and high class imbalance

_IBIT_XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 3,             # shallower than SPY (6) — prevent overfit on 249 trades
    "learning_rate": 0.03,      # slower learning for small data
    "n_estimators": 150,
    "min_child_weight": 10,     # higher than SPY (5) — need more evidence per leaf
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "gamma": 2,                 # higher regularization
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "eval_metric": "logloss",
    "scale_pos_weight": 1.0,    # adjusted during training if imbalanced
}

# IBIT regime thresholds (BTC is more volatile than SPY)
IBIT_REGIME_CONFIG = {
    "vix_low": 20.0,            # SPY uses 15
    "vix_high": 40.0,           # SPY uses 30
    "vix_extreme": 60.0,        # SPY uses 45
    "btc_corr_threshold": 0.5,  # BTC-ETH correlation below this = regime shift
    "weekend_gap_pct": 3.0,     # flag gaps > 3% (BTC trades but IBIT doesn't)
}


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""
    fold: int
    train_size: int
    test_size: int
    auc: float
    accuracy: float
    precision: float
    recall: float
    feature_importances: Dict[str, float]


@dataclass
class IBITTrainingStats:
    """Comprehensive training statistics."""
    test_auc: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    n_train: int
    n_test: int
    n_features: int
    positive_rate: float
    scale_pos_weight: float
    walk_forward_folds: List[WalkForwardFold]
    wf_mean_auc: float
    wf_std_auc: float
    feature_importances: Dict[str, float]
    timestamp: str


# ── IBITSignalModel ──────────────────────────────────────────────────────


class IBITSignalModel:
    """XGBoost binary classifier for IBIT trade win/loss prediction.

    Compatible with RegimeModelRouter's model interface (predict, predict_batch,
    train, save, load, trained, training_stats, feature_names).

    Args:
        model_dir: Directory for model persistence.
        xgb_params: Override XGBoost parameters.
    """

    def __init__(
        self,
        model_dir: str = "ml/models/ibit",
        xgb_params: Optional[Dict] = None,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.xgb_params = dict(_IBIT_XGB_PARAMS)
        if xgb_params:
            self.xgb_params.update(xgb_params)

        self.model = None
        self.trained: bool = False
        self.training_stats: Dict[str, Any] = {}
        self.feature_names: Optional[List[str]] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self._ibit_stats: Optional[IBITTrainingStats] = None

    # ── Training ─────────────────────────────────────────────────────

    def train(
        self,
        features_df: pd.DataFrame,
        labels: np.ndarray,
        calibrate: bool = False,
        save_model: bool = True,
        n_wf_folds: int = 3,
    ) -> Dict[str, Any]:
        """Train the IBIT signal model with walk-forward validation.

        Args:
            features_df: Feature DataFrame (columns = feature names).
            labels: Binary labels (1=win, 0=loss).
            calibrate: Whether to calibrate probabilities (not used for IBIT
                       due to small sample size).
            save_model: Persist to disk after training.
            n_wf_folds: Number of walk-forward folds.

        Returns:
            Training statistics dictionary.
        """
        import xgboost as xgb

        self.feature_names = list(features_df.columns)
        X = sanitize_features(features_df.values.astype(np.float64))
        y = labels.astype(int)

        # Adjust scale_pos_weight for class imbalance
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        if n_pos > 0 and n_neg > 0:
            self.xgb_params["scale_pos_weight"] = n_neg / n_pos

        # Feature distribution stats
        self.feature_means = np.nanmean(X, axis=0)
        self.feature_stds = np.nanstd(X, axis=0)

        # Walk-forward validation
        wf_folds = self._walk_forward_validate(X, y, n_wf_folds)

        # Final model: train on 80%, test on 20%
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = xgb.XGBClassifier(**self.xgb_params)
        self.model.fit(X_train, y_train,
                       eval_set=[(X_test, y_test)], verbose=False)

        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > 0.5).astype(int)

        # Feature importance
        gain_raw = self.model.get_booster().get_score(importance_type="gain")
        importances: Dict[str, float] = {}
        for fname, score in gain_raw.items():
            if fname.startswith("f"):
                idx = int(fname[1:])
                if idx < len(self.feature_names):
                    importances[self.feature_names[idx]] = score
        total = sum(importances.values())
        if total > 0:
            importances = {k: round(v / total, 4) for k, v in importances.items()}

        # Metrics
        has_two_classes = len(np.unique(y_test)) >= 2
        auc = float(roc_auc_score(y_test, y_proba)) if has_two_classes else 0.5

        wf_aucs = [f.auc for f in wf_folds]
        wf_mean = float(np.mean(wf_aucs)) if wf_aucs else 0.5
        wf_std = float(np.std(wf_aucs, ddof=1)) if len(wf_aucs) > 1 else 0.0

        self._ibit_stats = IBITTrainingStats(
            test_auc=round(auc, 4),
            test_accuracy=round(float(accuracy_score(y_test, y_pred)), 4),
            test_precision=round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            test_recall=round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            n_train=len(X_train),
            n_test=len(X_test),
            n_features=X.shape[1],
            positive_rate=round(float(y.mean()), 4),
            scale_pos_weight=round(self.xgb_params["scale_pos_weight"], 4),
            walk_forward_folds=wf_folds,
            wf_mean_auc=round(wf_mean, 4),
            wf_std_auc=round(wf_std, 4),
            feature_importances=importances,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        self.training_stats = {
            "test_auc": self._ibit_stats.test_auc,
            "test_accuracy": self._ibit_stats.test_accuracy,
            "wf_mean_auc": self._ibit_stats.wf_mean_auc,
            "n_train": self._ibit_stats.n_train,
            "n_test": self._ibit_stats.n_test,
            "timestamp": self._ibit_stats.timestamp,
            "feature_importances": importances,
        }
        self.trained = True

        if save_model:
            self.save()

        logger.info("IBIT model trained: AUC=%.4f, WF AUC=%.4f±%.4f",
                     auc, wf_mean, wf_std)
        return self.training_stats

    def _walk_forward_validate(
        self, X: np.ndarray, y: np.ndarray, n_folds: int,
    ) -> List[WalkForwardFold]:
        """Time-series walk-forward validation."""
        import xgboost as xgb

        n = len(y)
        folds: List[WalkForwardFold] = []

        if n < 50 or n_folds < 2:
            return folds

        tscv = TimeSeriesSplit(n_splits=n_folds)
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            model = xgb.XGBClassifier(**self.xgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

            y_proba = model.predict_proba(X_te)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)

            # Per-fold feature importance
            gain = model.get_booster().get_score(importance_type="gain")
            fi: Dict[str, float] = {}
            for fname, score in gain.items():
                if fname.startswith("f"):
                    idx = int(fname[1:])
                    if self.feature_names and idx < len(self.feature_names):
                        fi[self.feature_names[idx]] = score
            total = sum(fi.values())
            if total > 0:
                fi = {k: round(v / total, 4) for k, v in fi.items()}

            folds.append(WalkForwardFold(
                fold=fold_idx,
                train_size=len(X_tr),
                test_size=len(X_te),
                auc=round(float(roc_auc_score(y_te, y_proba)), 4),
                accuracy=round(float(accuracy_score(y_te, y_pred)), 4),
                precision=round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
                recall=round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
                feature_importances=fi,
            ))

        return folds

    # ── Prediction ───────────────────────────────────────────────────

    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict win probability for a single IBIT trade.

        Compatible with RegimeModelRouter's expected interface.
        """
        if not self.trained or self.model is None:
            return self._default_prediction()

        try:
            X = self._dict_to_array(features)
            if X is None:
                return self._default_prediction()

            proba = float(self.model.predict_proba(X)[0, 1])
            pred = int(proba > 0.5)
            confidence = abs(proba - 0.5) * 2

            return {
                "prediction": pred,
                "probability": round(proba, 4),
                "confidence": round(confidence, 4),
                "signal": "bullish" if proba > 0.55 else "bearish" if proba < 0.45 else "neutral",
                "signal_strength": round(proba * 100, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            logger.error("IBIT prediction error: %s", exc)
            return self._default_prediction()

    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Batch prediction. Returns 1D array of probabilities."""
        if not self.trained or self.model is None:
            return np.full(len(features_df), 0.5)

        try:
            cols = self.feature_names or list(features_df.columns)
            available = [c for c in cols if c in features_df.columns]
            X = features_df[available].fillna(0).values.astype(np.float64)
            X = sanitize_features(X)
            return self.model.predict_proba(X)[:, 1]
        except Exception as exc:
            logger.error("IBIT batch prediction error: %s", exc)
            return np.full(len(features_df), 0.5)

    def _dict_to_array(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        if self.feature_names is None:
            return None
        vals = [features.get(f, 0.0) or 0.0 for f in self.feature_names]
        X = np.array(vals, dtype=np.float64).reshape(1, -1)
        return sanitize_features(X)

    @staticmethod
    def _default_prediction() -> Dict[str, Any]:
        return {
            "prediction": 0, "probability": 0.5, "confidence": 0.0,
            "signal": "neutral", "signal_strength": 50.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fallback": True,
        }

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, filename: str = "ibit_signal_model.joblib") -> None:
        """Save model to disk."""
        path = self.model_dir / filename
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "training_stats": self.training_stats,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "xgb_params": self.xgb_params,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_type": "ibit_signal",
        }
        joblib.dump(data, path)
        logger.info("IBIT model saved to %s", path)

    def load(self, filename: str = "ibit_signal_model.joblib") -> bool:
        """Load model from disk. Returns True on success."""
        path = self.model_dir / filename
        if not path.exists():
            files = sorted(self.model_dir.glob("ibit_signal_model*.joblib"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                return False
            path = files[0]

        # Path traversal guard
        resolved = os.path.realpath(path)
        expected = os.path.realpath(self.model_dir)
        if not resolved.startswith(expected + os.sep) and resolved != expected:
            logger.error("Path traversal blocked: %s", resolved)
            return False

        try:
            data = joblib.load(path)
            self.model = data["model"]
            self.feature_names = data["feature_names"]
            self.training_stats = data.get("training_stats", {})
            self.feature_means = data.get("feature_means")
            self.feature_stds = data.get("feature_stds")
            self.trained = True
            logger.info("IBIT model loaded from %s", path)
            return True
        except Exception as exc:
            logger.error("Failed to load IBIT model: %s", exc)
            return False

    # ── Feature importance ───────────────────────────────────────────

    def get_feature_importance(self, top_n: int = 15) -> List[Tuple[str, float]]:
        """Return top features sorted by importance."""
        imp = self.training_stats.get("feature_importances", {})
        return sorted(imp.items(), key=lambda x: -x[1])[:top_n]

    # ── IBIT regime detection ────────────────────────────────────────

    @staticmethod
    def classify_ibit_regime(
        vix: float,
        btc_corr: Optional[float] = None,
        gap_pct: Optional[float] = None,
    ) -> str:
        """Classify IBIT-specific market regime.

        Uses higher VIX thresholds than SPY (BTC is more volatile).
        """
        cfg = IBIT_REGIME_CONFIG
        if vix >= cfg["vix_extreme"]:
            return "crash"
        if vix >= cfg["vix_high"]:
            return "high_vol"
        if vix < cfg["vix_low"]:
            return "low_vol"
        if btc_corr is not None and btc_corr < cfg["btc_corr_threshold"]:
            return "decorrelating"
        if gap_pct is not None and abs(gap_pct) > cfg["weekend_gap_pct"]:
            return "gap_risk"
        return "normal"

    # ── HTML Report ──────────────────────────────────────────────────

    def generate_report(self, output: str = str(Path("reports/ibit_model.html"))) -> str:
        """Generate HTML model report."""
        if not self._ibit_stats:
            logger.warning("No training stats — run train() first")
            return ""

        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out.resolve())

    def _render_charts(self) -> Dict[str, str]:
        import matplotlib
        matplotlib.use("Agg")
        charts: Dict[str, str] = {}
        charts["importance"] = self._chart_importance()
        charts["wf_auc"] = self._chart_wf_auc()
        return charts

    def _fig_to_b64(self, fig) -> str:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _chart_importance(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = self.get_feature_importance(15)
        if not top:
            return ""
        names, vals = zip(*top)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(names))))
        y = range(len(names))
        ax.barh(y, vals, color="#3b82f6", alpha=0.85, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance (gain)")
        ax.set_title("IBIT Feature Importance", fontsize=12)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_wf_auc(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        stats = self._ibit_stats
        if not stats or not stats.walk_forward_folds:
            return ""
        folds = stats.walk_forward_folds
        fig, ax = plt.subplots(figsize=(7, 4))
        x = [f.fold for f in folds]
        aucs = [f.auc for f in folds]
        ax.bar(x, aucs, color="#2563eb", alpha=0.85, edgecolor="white")
        ax.axhline(0.5, color="#dc2626", ls="--", lw=1, label="Random (0.5)")
        ax.axhline(stats.wf_mean_auc, color="#16a34a", ls="-", lw=1.5,
                    label=f"Mean ({stats.wf_mean_auc:.3f})")
        ax.set_xlabel("Fold")
        ax.set_ylabel("AUC")
        ax.set_title("Walk-Forward AUC per Fold", fontsize=12)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        stats = self._ibit_stats

        def _img(key):
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ''

        # WF fold table
        fold_rows = ""
        for f in (stats.walk_forward_folds if stats else []):
            auc_cls = "good" if f.auc > 0.55 else "bad" if f.auc < 0.5 else ""
            fold_rows += (
                f'<tr><td>{f.fold}</td><td>{f.train_size}</td><td>{f.test_size}</td>'
                f'<td class="{auc_cls}">{f.auc:.4f}</td><td>{f.accuracy:.4f}</td>'
                f'<td>{f.precision:.4f}</td><td>{f.recall:.4f}</td></tr>\n'
            )

        auc_cls = "good" if stats and stats.test_auc > 0.55 else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>IBIT Signal Model — EXP-601</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 1.5em; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .kpi-row {{ display: flex; gap: 1.2em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 120px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.5em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.75em; color: #64748b; margin-top: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #f1f5f9; padding: 8px 10px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>IBIT Signal Model — EXP-601</h1>
<div class="meta">Generated {now} &middot; {stats.n_features if stats else 0} features &middot; {stats.n_train + stats.n_test if stats else 0} trades</div>

<div class="kpi-row">
  <div class="kpi"><div class="value {auc_cls}">{stats.test_auc if stats else '—'}</div><div class="label">Test AUC</div></div>
  <div class="kpi"><div class="value">{stats.wf_mean_auc if stats else '—'}</div><div class="label">WF Mean AUC</div></div>
  <div class="kpi"><div class="value">{stats.test_accuracy if stats else '—'}</div><div class="label">Accuracy</div></div>
  <div class="kpi"><div class="value">{stats.positive_rate if stats else '—'}</div><div class="label">Win Rate (base)</div></div>
  <div class="kpi"><div class="value">{stats.n_features if stats else '—'}</div><div class="label">Features</div></div>
</div>

<h2>1. Feature Importance</h2>
{_img("importance")}

<h2>2. Walk-Forward Validation</h2>
{_img("wf_auc")}
<table>
<thead><tr><th>Fold</th><th>Train</th><th>Test</th><th>AUC</th><th>Accuracy</th><th>Precision</th><th>Recall</th></tr></thead>
<tbody>{fold_rows}</tbody>
</table>

<h2>3. IBIT Regime Thresholds</h2>
<table>
<tbody>
<tr><td>VIX Low (low_vol)</td><td>&lt; {IBIT_REGIME_CONFIG["vix_low"]}</td></tr>
<tr><td>VIX High (high_vol)</td><td>&ge; {IBIT_REGIME_CONFIG["vix_high"]}</td></tr>
<tr><td>VIX Extreme (crash)</td><td>&ge; {IBIT_REGIME_CONFIG["vix_extreme"]}</td></tr>
<tr><td>BTC Corr Threshold</td><td>&lt; {IBIT_REGIME_CONFIG["btc_corr_threshold"]}</td></tr>
<tr><td>Weekend Gap Alert</td><td>&gt; {IBIT_REGIME_CONFIG["weekend_gap_pct"]}%</td></tr>
</tbody>
</table>

<footer>Generated by <code>compass/ibit_signal_model.py</code></footer>
</body></html>"""
        return html
