"""
Ensemble meta-learner — learns optimal model combination weights.

Components:
  - Base model prediction storage (XGB, RF, LR per trade)
  - Stacking with logistic regression + ridge as meta-models
  - Walk-forward meta-training (expanding window, never look ahead)
  - Feature engineering from base model disagreements
  - Regime-conditional meta-weights
  - Performance comparison (meta vs best single model)
  - Lift analysis (meta improvement over best base)

HTML report at reports/meta_learner.html with model contribution chart,
regime weight heatmap, lift analysis.

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.meta_learner import MetaLearner
    ml = MetaLearner(base_predictions, actuals)
    result = ml.fit()
    MetaLearner.generate_report(result)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "meta_learner.html"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class BaseModelStats:
    """Performance stats for one base model."""

    name: str
    accuracy: float
    auc_approx: float  # concordance-based approximation
    brier_score: float
    sharpe: float
    n_predictions: int
    meta_weight: float


@dataclass
class DisagreementFeatures:
    """Features engineered from base model disagreements."""

    avg_disagreement: float
    max_disagreement: float
    n_unanimous: int
    n_split: int
    disagreement_ratio: float


@dataclass
class RegimeWeights:
    """Meta-weights per regime."""

    regime: str
    weights: Dict[str, float]
    n_samples: int
    regime_accuracy: float


@dataclass
class WalkForwardFold:
    """One fold of walk-forward meta-training."""

    fold: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    meta_accuracy: float
    best_base_accuracy: float
    lift: float


@dataclass
class LiftAnalysis:
    """Meta-learner lift over best single model."""

    meta_accuracy: float
    best_base_accuracy: float
    best_base_name: str
    lift_pct: float
    meta_sharpe: float
    best_base_sharpe: float
    sharpe_lift: float


@dataclass
class MetaLearnerResult:
    """Full result from meta-learner."""

    base_model_stats: List[BaseModelStats]
    meta_weights: Dict[str, float]  # final learned weights
    meta_method: str
    disagreement: DisagreementFeatures
    regime_weights: List[RegimeWeights]
    walk_forward_folds: List[WalkForwardFold]
    lift: LiftAnalysis
    meta_predictions: np.ndarray
    n_samples: int
    n_base_models: int


# ── Logistic regression (pure numpy) ─────────────────────────────────────


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_fit(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_iter: int = 200,
    reg: float = 0.01,
) -> np.ndarray:
    """Fit logistic regression via gradient descent with L2 regularization.

    Returns weight vector including intercept as last element.
    """
    n, p = X.shape
    X_b = np.column_stack([X, np.ones(n)])
    w = np.zeros(p + 1)

    for _ in range(n_iter):
        pred = _sigmoid(X_b @ w)
        error = pred - y
        grad = X_b.T @ error / n + reg * w
        grad[-1] -= reg * w[-1]  # don't regularize intercept
        w -= lr * grad

    return w


def logistic_predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict probabilities from logistic weights."""
    X_b = np.column_stack([X, np.ones(X.shape[0])])
    return _sigmoid(X_b @ w)


# ── Ridge regression (pure numpy) ────────────────────────────────────────


def ridge_fit(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit ridge regression. Returns weight vector (no intercept column needed)."""
    X_b = np.column_stack([X, np.ones(X.shape[0])])
    n_f = X_b.shape[1]
    reg = alpha * np.eye(n_f)
    reg[-1, -1] = 0  # don't regularize intercept
    try:
        w = np.linalg.solve(X_b.T @ X_b + reg, X_b.T @ y)
    except np.linalg.LinAlgError:
        w = np.zeros(n_f)
    return w


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict from ridge weights."""
    X_b = np.column_stack([X, np.ones(X.shape[0])])
    return X_b @ w


# ── Disagreement features ────────────────────────────────────────────────


def compute_disagreement_features(
    predictions: pd.DataFrame,
) -> Tuple[DisagreementFeatures, np.ndarray]:
    """Engineer features from base model disagreements.

    Returns: (features_summary, per-row disagreement array)
    """
    vals = predictions.values
    n = len(vals)
    if n == 0:
        return DisagreementFeatures(0, 0, 0, 0, 0), np.array([])

    # Disagreement = std of predictions across models per row
    row_std = vals.std(axis=1)
    row_range = vals.max(axis=1) - vals.min(axis=1)

    # Unanimous: all models agree on direction (>0.5 or <0.5)
    binary = (vals > 0.5).astype(int)
    unanimous = (binary.sum(axis=1) == 0) | (binary.sum(axis=1) == vals.shape[1])
    n_unanimous = int(unanimous.sum())

    return DisagreementFeatures(
        avg_disagreement=float(row_std.mean()),
        max_disagreement=float(row_std.max()),
        n_unanimous=n_unanimous,
        n_split=n - n_unanimous,
        disagreement_ratio=(n - n_unanimous) / n if n > 0 else 0.0,
    ), row_std


# ── Base model evaluation ────────────────────────────────────────────────


def evaluate_base_model(
    predictions: np.ndarray,
    actuals: np.ndarray,
    name: str,
    meta_weight: float = 0.0,
) -> BaseModelStats:
    """Evaluate a single base model."""
    n = len(predictions)
    if n == 0:
        return BaseModelStats(name, 0, 0, 0, 0, 0, meta_weight)

    # Accuracy (binary threshold 0.5)
    binary_pred = (predictions > 0.5).astype(int)
    accuracy = float((binary_pred == actuals).mean())

    # Brier score
    brier = float(np.mean((predictions - actuals) ** 2))

    # AUC approximation (concordance)
    pos = predictions[actuals == 1]
    neg = predictions[actuals == 0]
    if len(pos) > 0 and len(neg) > 0:
        concordant = 0
        total = len(pos) * len(neg)
        for p in pos:
            concordant += (neg < p).sum()
        auc = concordant / total
    else:
        auc = 0.5

    # Sharpe from PnL proxy: correct predictions = +1, wrong = -1
    pnl = np.where(binary_pred == actuals, 1.0, -1.0)
    mu = pnl.mean()
    std = pnl.std(ddof=1) if len(pnl) > 1 else 1.0
    sharpe = float(mu / std * math.sqrt(252)) if std > 1e-12 else 0.0

    return BaseModelStats(
        name=name, accuracy=accuracy, auc_approx=float(auc),
        brier_score=brier, sharpe=sharpe,
        n_predictions=n, meta_weight=meta_weight,
    )


# ── Regime-conditional weights ───────────────────────────────────────────


def fit_regime_weights(
    predictions: pd.DataFrame,
    actuals: np.ndarray,
    regimes: np.ndarray,
    alpha: float = 1.0,
) -> List[RegimeWeights]:
    """Fit separate meta-weights per regime."""
    unique_regimes = np.unique(regimes)
    results: List[RegimeWeights] = []

    for regime in unique_regimes:
        mask = regimes == regime
        X = predictions.values[mask]
        y = actuals[mask]

        if len(y) < 10:
            # Too few samples, equal weight
            w = np.ones(X.shape[1]) / X.shape[1]
        else:
            w_full = ridge_fit(X, y, alpha)
            w = w_full[:-1]  # strip intercept
            abs_sum = np.abs(w).sum()
            if abs_sum > 1e-12:
                w = w / abs_sum

        weights_dict = {col: float(w[i]) for i, col in enumerate(predictions.columns)}
        acc = float((((X @ w[:X.shape[1]]) > 0.5).astype(int) == y).mean()) if len(y) > 0 else 0.0

        results.append(RegimeWeights(
            regime=str(regime), weights=weights_dict,
            n_samples=int(mask.sum()), regime_accuracy=acc,
        ))

    return results


# ── Walk-forward meta-training ───────────────────────────────────────────


def walk_forward_meta(
    predictions: pd.DataFrame,
    actuals: np.ndarray,
    n_folds: int = 5,
    meta_method: str = "ridge",
    alpha: float = 1.0,
) -> Tuple[List[WalkForwardFold], np.ndarray]:
    """Walk-forward expanding-window meta-training.

    Returns: (folds, out-of-sample meta predictions)
    """
    n = len(actuals)
    if n < 20 or n_folds < 1:
        return [], np.full(n, 0.5)

    fold_size = n // (n_folds + 1)
    if fold_size < 5:
        return [], np.full(n, 0.5)

    folds: List[WalkForwardFold] = []
    oos_preds = np.full(n, 0.5)
    X = predictions.values

    for f in range(n_folds):
        train_end = fold_size * (f + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)

        if test_end <= test_start:
            continue

        X_train, y_train = X[:train_end], actuals[:train_end]
        X_test, y_test = X[test_start:test_end], actuals[test_start:test_end]

        # Fit meta-model
        if meta_method == "logistic":
            w = logistic_fit(X_train, y_train)
            test_preds = logistic_predict_proba(X_test, w)
        else:
            w = ridge_fit(X_train, y_train, alpha)
            test_preds = ridge_predict(X_test, w)
            test_preds = np.clip(test_preds, 0, 1)

        oos_preds[test_start:test_end] = test_preds

        meta_acc = float(((test_preds > 0.5).astype(int) == y_test).mean())

        # Best single base model accuracy on test set
        best_base = 0.0
        for col_idx in range(X_test.shape[1]):
            base_acc = float(((X_test[:, col_idx] > 0.5).astype(int) == y_test).mean())
            best_base = max(best_base, base_acc)

        lift = (meta_acc - best_base) / best_base * 100 if best_base > 0 else 0.0

        folds.append(WalkForwardFold(
            fold=f + 1, train_end=train_end,
            test_start=test_start, test_end=test_end,
            n_train=train_end, n_test=test_end - test_start,
            meta_accuracy=meta_acc, best_base_accuracy=best_base,
            lift=lift,
        ))

    return folds, oos_preds


# ── Lift analysis ────────────────────────────────────────────────────────


def compute_lift(
    base_stats: List[BaseModelStats],
    meta_accuracy: float,
    meta_sharpe: float,
) -> LiftAnalysis:
    """Compare meta-learner vs best single model."""
    if not base_stats:
        return LiftAnalysis(meta_accuracy, 0, "", 0, meta_sharpe, 0, 0)

    best = max(base_stats, key=lambda s: s.accuracy)
    lift_pct = (meta_accuracy - best.accuracy) / best.accuracy * 100 if best.accuracy > 0 else 0.0
    sharpe_lift = meta_sharpe - best.sharpe

    return LiftAnalysis(
        meta_accuracy=meta_accuracy,
        best_base_accuracy=best.accuracy,
        best_base_name=best.name,
        lift_pct=lift_pct,
        meta_sharpe=meta_sharpe,
        best_base_sharpe=best.sharpe,
        sharpe_lift=sharpe_lift,
    )


# ── Core engine ──────────────────────────────────────────────────────────


class MetaLearner:
    """Ensemble meta-learner for optimal model combination.

    Args:
        predictions: DataFrame where each column is a base model's
                     probability predictions (0-1).
        actuals: Series/array of actual outcomes (0/1).
        regimes: Optional array of regime labels per observation.
        meta_method: 'ridge' or 'logistic'.
        n_folds: walk-forward folds.
        alpha: regularization strength.
    """

    def __init__(
        self,
        predictions: pd.DataFrame,
        actuals: np.ndarray,
        regimes: Optional[np.ndarray] = None,
        meta_method: str = "ridge",
        n_folds: int = 5,
        alpha: float = 1.0,
    ):
        if predictions.empty:
            raise ValueError("predictions DataFrame must not be empty")
        if len(actuals) != len(predictions):
            raise ValueError("predictions and actuals must have same length")
        if predictions.shape[1] < 2:
            raise ValueError("Need at least 2 base models")
        if meta_method not in ("ridge", "logistic"):
            raise ValueError(f"meta_method must be 'ridge' or 'logistic', got {meta_method!r}")

        self.predictions = predictions.copy()
        self.actuals = np.asarray(actuals, dtype=float)
        self.regimes = regimes
        self.meta_method = meta_method
        self.n_folds = n_folds
        self.alpha = alpha
        self.model_names = list(predictions.columns)
        self.n_models = len(self.model_names)

    def fit(self) -> MetaLearnerResult:
        """Fit meta-learner and evaluate."""
        X = self.predictions.values
        y = self.actuals

        # Walk-forward
        wf_folds, oos_preds = walk_forward_meta(
            self.predictions, y, self.n_folds, self.meta_method, self.alpha,
        )

        # Fit final meta-weights on all data
        if self.meta_method == "logistic":
            w_full = logistic_fit(X, y)
            final_preds = logistic_predict_proba(X, w_full)
            weights_raw = w_full[:-1]
        else:
            w_full = ridge_fit(X, y, self.alpha)
            final_preds = np.clip(ridge_predict(X, w_full), 0, 1)
            weights_raw = w_full[:-1]

        # Normalize weights
        abs_sum = np.abs(weights_raw).sum()
        if abs_sum > 1e-12:
            weights_norm = weights_raw / abs_sum
        else:
            weights_norm = np.ones(self.n_models) / self.n_models

        meta_weights = {name: float(weights_norm[i]) for i, name in enumerate(self.model_names)}

        # Base model stats
        base_stats = []
        for i, name in enumerate(self.model_names):
            stats = evaluate_base_model(X[:, i], y, name, float(weights_norm[i]))
            base_stats.append(stats)

        # Meta accuracy and Sharpe
        meta_binary = (oos_preds > 0.5).astype(int)
        meta_acc = float((meta_binary == y).mean())
        meta_pnl = np.where(meta_binary == y, 1.0, -1.0)
        mu, std = meta_pnl.mean(), meta_pnl.std(ddof=1) if len(meta_pnl) > 1 else 1.0
        meta_sharpe = float(mu / std * math.sqrt(252)) if std > 1e-12 else 0.0

        # Disagreement
        disagree, disagree_arr = compute_disagreement_features(self.predictions)

        # Regime weights
        regime_wts: List[RegimeWeights] = []
        if self.regimes is not None:
            regime_wts = fit_regime_weights(
                self.predictions, y, self.regimes, self.alpha
            )

        # Lift
        lift = compute_lift(base_stats, meta_acc, meta_sharpe)

        return MetaLearnerResult(
            base_model_stats=base_stats,
            meta_weights=meta_weights,
            meta_method=self.meta_method,
            disagreement=disagree,
            regime_weights=regime_wts,
            walk_forward_folds=wf_folds,
            lift=lift,
            meta_predictions=oos_preds,
            n_samples=len(y),
            n_base_models=self.n_models,
        )

    @staticmethod
    def generate_report(
        result: MetaLearnerResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _f(v: float, d: int = 3) -> str:
    return f"{v:.{d}f}"


def _fp(v: float) -> str:
    return f"{v:.1%}"


def _weight_bars_svg(weights: Dict[str, float]) -> str:
    if not weights:
        return ""
    names = list(weights.keys())
    vals = [weights[n] for n in names]
    n = len(names)
    w, h = 500, n * 36 + 40
    pad_l = 80
    abs_max = max(abs(v) for v in vals) if vals else 1.0
    if abs_max == 0:
        abs_max = 1.0
    bar_area = (w - pad_l - 40) / 2
    mid_x = pad_l + bar_area
    bar_h = 24

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="18" text-anchor="middle" class="svg-title">Meta-Weights by Model</text>')
    parts.append(f'<line x1="{mid_x:.0f}" y1="25" x2="{mid_x:.0f}" y2="{h - 5}" stroke="#30363d"/>')

    for i in range(n):
        y = 30 + i * 36
        bw = abs(vals[i]) / abs_max * bar_area
        color = "#3fb950" if vals[i] >= 0 else "#f85149"
        bx = mid_x if vals[i] >= 0 else mid_x - bw
        parts.append(f'<text x="{pad_l - 5}" y="{y + 16:.0f}" text-anchor="end" font-size="10" fill="#8b949e">{names[i][:10]}</text>')
        parts.append(f'<rect x="{bx:.0f}" y="{y}" width="{bw:.0f}" height="{bar_h}" fill="{color}" rx="3" opacity="0.85"/>')
        parts.append(f'<text x="{bx + bw + 4:.0f}" y="{y + 16:.0f}" font-size="9" fill="#c9d1d9">{vals[i]:+.3f}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _regime_heatmap_svg(regime_wts: List[RegimeWeights]) -> str:
    if not regime_wts:
        return ""
    regimes = [r.regime for r in regime_wts]
    models = list(regime_wts[0].weights.keys()) if regime_wts else []
    if not models:
        return ""

    cell = 50
    lbl_l, lbl_t = 80, 25
    w = lbl_l + len(models) * cell + 10
    h = lbl_t + len(regimes) * cell + 30

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="16" text-anchor="middle" class="svg-title">Regime Weight Heatmap</text>')

    for i, rw in enumerate(regime_wts):
        for j, model in enumerate(models):
            val = rw.weights.get(model, 0)
            intensity = min(abs(val) * 2, 1.0)
            if val >= 0:
                r, g, b = int(255 * (1 - intensity * 0.5)), 255, int(255 * (1 - intensity * 0.5))
            else:
                r, g, b = 255, int(255 * (1 - intensity * 0.5)), int(255 * (1 - intensity * 0.5))
            x = lbl_l + j * cell
            y = lbl_t + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({r},{g},{b})" stroke="#30363d" stroke-width="0.5"/>')
            parts.append(f'<text x="{x + cell // 2}" y="{y + cell // 2 + 4}" text-anchor="middle" font-size="9" fill="#000">{val:.2f}</text>')

    for j, m in enumerate(models):
        parts.append(f'<text x="{lbl_l + j * cell + cell // 2}" y="{lbl_t - 3}" text-anchor="middle" font-size="8" fill="#8b949e">{m[:8]}</text>')
    for i, reg in enumerate(regimes):
        parts.append(f'<text x="{lbl_l - 4}" y="{lbl_t + i * cell + cell // 2 + 3}" text-anchor="end" font-size="9" fill="#8b949e">{reg}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _base_table(stats: List[BaseModelStats]) -> str:
    if not stats:
        return ""
    rows = ""
    for s in sorted(stats, key=lambda x: x.accuracy, reverse=True):
        rows += f"<tr><td style='text-align:left'>{s.name}</td><td>{_fp(s.accuracy)}</td><td>{_f(s.auc_approx)}</td><td>{_f(s.brier_score)}</td><td>{_f(s.sharpe, 2)}</td><td>{_f(s.meta_weight)}</td></tr>"
    return f"""<table class="data-table"><tr><th style='text-align:left'>Model</th><th>Accuracy</th><th>AUC</th><th>Brier</th><th>Sharpe</th><th>Meta Weight</th></tr>{rows}</table>"""


def _wf_table(folds: List[WalkForwardFold]) -> str:
    if not folds:
        return "<p class='meta'>Insufficient data for walk-forward.</p>"
    rows = ""
    for f in folds:
        color = "#3fb950" if f.lift > 0 else "#f85149"
        rows += f"<tr><td>{f.fold}</td><td>{f.n_train}</td><td>{f.n_test}</td><td>{_fp(f.meta_accuracy)}</td><td>{_fp(f.best_base_accuracy)}</td><td style='color:{color}'>{f.lift:+.1f}%</td></tr>"
    return f"""<table class="data-table"><tr><th>Fold</th><th>Train N</th><th>Test N</th><th>Meta Acc</th><th>Best Base</th><th>Lift</th></tr>{rows}</table>"""


def _build_html(result: MetaLearnerResult) -> str:
    l = result.lift
    lift_color = "#3fb950" if l.lift_pct > 0 else "#f85149"
    d = result.disagreement

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Meta-Learner Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
              gap: 12px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 12px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.8em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.15em; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }}
  .metrics-grid .label {{ color: #8b949e; font-size: 0.85em; }}
  .metrics-grid .value {{ color: #f0f6fc; font-weight: 600; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  .chart {{ width: 100%; max-width: 700px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Ensemble Meta-Learner</h1>
<p class="meta">{result.n_base_models} base models &middot; {result.n_samples} samples &middot;
   Method: {result.meta_method} &middot; {len(result.walk_forward_folds)} WF folds</p>

<div class="summary">
  <div class="stat"><div class="label">Meta Accuracy</div><div class="value">{_fp(l.meta_accuracy)}</div></div>
  <div class="stat"><div class="label">Best Base Acc</div><div class="value">{_fp(l.best_base_accuracy)}</div></div>
  <div class="stat"><div class="label">Lift</div><div class="value" style="color:{lift_color}">{l.lift_pct:+.1f}%</div></div>
  <div class="stat"><div class="label">Meta Sharpe</div><div class="value">{_f(l.meta_sharpe, 2)}</div></div>
  <div class="stat"><div class="label">Sharpe Lift</div><div class="value" style="color:{lift_color}">{l.sharpe_lift:+.2f}</div></div>
  <div class="stat"><div class="label">Disagreement</div><div class="value">{_fp(d.disagreement_ratio)}</div></div>
</div>

<h2>Base Model Comparison</h2>
{_base_table(result.base_model_stats)}

<div class="two-col">
  {_weight_bars_svg(result.meta_weights)}
  {_regime_heatmap_svg(result.regime_weights)}
</div>

<div class="card">
  <h3>Disagreement Analysis</h3>
  <div class="metrics-grid">
    <div><span class="label">Avg Disagreement</span><span class="value">{_f(d.avg_disagreement)}</span></div>
    <div><span class="label">Max Disagreement</span><span class="value">{_f(d.max_disagreement)}</span></div>
    <div><span class="label">Unanimous</span><span class="value">{d.n_unanimous}</span></div>
    <div><span class="label">Split</span><span class="value">{d.n_split}</span></div>
    <div><span class="label">Disagree Ratio</span><span class="value">{_fp(d.disagreement_ratio)}</span></div>
  </div>
</div>

<h2>Walk-Forward Validation</h2>
{_wf_table(result.walk_forward_folds)}

<div class="card">
  <h3>Lift Analysis</h3>
  <p>Meta-learner accuracy <strong>{_fp(l.meta_accuracy)}</strong> vs best base model
     <strong>{l.best_base_name}</strong> at <strong>{_fp(l.best_base_accuracy)}</strong>
     &rarr; <span style="color:{lift_color}"><strong>{l.lift_pct:+.1f}%</strong> lift</span></p>
  <p>Meta Sharpe <strong>{_f(l.meta_sharpe, 2)}</strong> vs best base Sharpe
     <strong>{_f(l.best_base_sharpe, 2)}</strong>
     &rarr; <span style="color:{lift_color}"><strong>{l.sharpe_lift:+.2f}</strong> Sharpe improvement</span></p>
</div>

</body>
</html>"""
