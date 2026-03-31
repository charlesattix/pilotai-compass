"""
Model diagnostics dashboard — self-contained HTML report with embedded charts.

Runs walk-forward validation on trade-level training data and produces:
  1. Calibration curve (predicted probability vs actual win rate, 10 bins)
  2. ROC curves per fold overlaid on one plot
  3. Confusion matrix at optimal threshold per fold
  4. Feature importance bar chart (top 15 features by gain)
  5. Prediction distribution histogram (win vs loss probability overlap)
  6. Summary table: per-fold AUC, accuracy, precision, recall, Brier score

All matplotlib charts are embedded as base64 PNGs inside a single HTML file —
no external dependencies needed to view the report.

Usage::

    from compass.model_diagnostics import generate_diagnostics
    generate_diagnostics("compass/training_data_combined.csv", "reports/model_diagnostics.html")
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from compass.walk_forward import (
    CATEGORICAL_FEATURES,
    DATE_COL,
    NUMERIC_FEATURES,
    TARGET_COL,
    prepare_features,
)

logger = logging.getLogger(__name__)

# ── XGBoost defaults (match SignalModel.train / feature_importance.py) ────

_XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "eval_metric": "logloss",
}


def _build_model():
    import xgboost as xgb
    return xgb.XGBClassifier(**_XGB_PARAMS)


# ── Walk-forward data collector ──────────────────────────────────────────


def _collect_fold_data(
    df: pd.DataFrame,
    min_train_samples: int = 30,
) -> List[Dict[str, Any]]:
    """Run walk-forward folds and collect all data needed for diagnostics.

    Returns list of per-fold dicts with keys:
        fold, train_years, test_year, n_train, n_test,
        y_test, y_proba, y_pred, auc, accuracy, precision, recall, brier,
        fpr, tpr, thresholds, optimal_threshold,
        gain_importance (dict: feature_name → normalized gain),
        model (the trained XGBoost model)
    """
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    years = sorted(df[DATE_COL].dt.year.unique())

    if len(years) < 2:
        raise ValueError(f"Need ≥2 years; got {years}")

    features_full = prepare_features(
        df,
        numeric_features=[c for c in NUMERIC_FEATURES if c in df.columns],
        categorical_features=CATEGORICAL_FEATURES,
    )
    feature_cols = list(features_full.columns)
    folds: List[Dict[str, Any]] = []

    for fold_idx in range(len(years) - 1):
        train_years = years[:fold_idx + 1]
        test_year = years[fold_idx + 1]
        train_mask = df[DATE_COL].dt.year.isin(train_years)
        test_mask = df[DATE_COL].dt.year == test_year
        n_train, n_test = int(train_mask.sum()), int(test_mask.sum())

        if n_train < min_train_samples or n_test < 5:
            continue

        X_train = features_full.loc[train_mask].values
        y_train = df.loc[train_mask, TARGET_COL].values.astype(int)
        X_test = features_full.loc[test_mask].values
        y_test = df.loc[test_mask, TARGET_COL].values.astype(int)

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        model = _build_model()
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        # Youden's J optimal threshold
        j_scores = tpr - fpr
        opt_idx = int(np.argmax(j_scores))
        optimal_threshold = float(thresholds[opt_idx])

        y_pred_opt = (y_proba >= optimal_threshold).astype(int)

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred_opt)
        prec = precision_score(y_test, y_pred_opt, zero_division=0)
        rec = recall_score(y_test, y_pred_opt, zero_division=0)
        brier = brier_score_loss(y_test, y_proba)

        # Gain importance
        gain_raw = model.get_booster().get_score(importance_type="gain")
        gain_dict: Dict[str, float] = {}
        for fname, score in gain_raw.items():
            if fname.startswith("f"):
                idx = int(fname[1:])
                if idx < len(feature_cols):
                    gain_dict[feature_cols[idx]] = score
        total_gain = sum(gain_dict.values())
        if total_gain > 0:
            gain_dict = {k: v / total_gain for k, v in gain_dict.items()}

        folds.append({
            "fold": fold_idx,
            "train_years": [int(y) for y in train_years],
            "test_year": int(test_year),
            "n_train": n_train,
            "n_test": n_test,
            "y_test": y_test,
            "y_proba": y_proba,
            "y_pred": y_pred_opt,
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "brier": round(brier, 4),
            "fpr": fpr,
            "tpr": tpr,
            "optimal_threshold": round(optimal_threshold, 3),
            "gain_importance": gain_dict,
        })

        logger.info(
            "Fold %d [→%d]: AUC=%.4f acc=%.3f prec=%.3f rec=%.3f brier=%.4f thresh=%.3f",
            fold_idx, test_year, auc, acc, prec, rec, brier, optimal_threshold,
        )

    return folds


# ── Chart rendering ──────────────────────────────────────────────────────


def _fig_to_base64(fig) -> str:
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _render_roc_curves(folds: List[Dict]) -> str:
    """Overlay all fold ROC curves on one plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(folds), 1)))

    for i, f in enumerate(folds):
        ax.plot(
            f["fpr"], f["tpr"],
            color=colors[i],
            lw=1.8,
            label=f"Fold {f['fold']} (test {f['test_year']}, AUC={f['auc']:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")

    mean_auc = np.mean([f["auc"] for f in folds])
    ax.set_title(f"ROC Curves — Walk-Forward Folds  (mean AUC = {mean_auc:.4f})", fontsize=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _render_calibration_curve(folds: List[Dict]) -> str:
    """Calibration curve: pooled OOS predictions vs actual win rate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_y = np.concatenate([f["y_test"] for f in folds])
    all_p = np.concatenate([f["y_proba"] for f in folds])

    n_bins = 10
    prob_true, prob_pred = calibration_curve(all_y, all_p, n_bins=n_bins, strategy="uniform")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(prob_pred, prob_true, "s-", color="#2563eb", lw=2, markersize=7, label="Model")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfectly calibrated")
    ax1.set_title("Calibration Curve (pooled out-of-sample predictions)", fontsize=12)
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Actual win rate")
    ax1.legend(fontsize=9)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)

    # Bin counts histogram
    ax2.hist(all_p, bins=20, color="#94a3b8", edgecolor="white", alpha=0.8)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Prediction distribution (all folds pooled)", fontsize=10)

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _render_confusion_matrices(folds: List[Dict]) -> str:
    """Grid of confusion matrices (one per fold) at optimal threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(folds)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, f in enumerate(folds):
        cm = confusion_matrix(f["y_test"], f["y_pred"])
        ax = axes[i]

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(
            f"Fold {f['fold']} (test {f['test_year']})\nthresh={f['optimal_threshold']:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Loss", "Win"])
        ax.set_yticklabels(["Loss", "Win"])

        # Annotate cells
        for row_idx in range(2):
            for col_idx in range(2):
                val = cm[row_idx, col_idx]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(col_idx, row_idx, str(val), ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Confusion Matrices at Optimal Threshold (Youden's J)", fontsize=12, y=1.01)
    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _render_feature_importance(folds: List[Dict], top_n: int = 15) -> str:
    """Bar chart of top features by mean gain importance across folds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Aggregate gain importance across folds
    all_features: Dict[str, List[float]] = {}
    for f in folds:
        for feat, val in f["gain_importance"].items():
            all_features.setdefault(feat, []).append(val)

    if not all_features:
        return ""

    mean_imp = {feat: np.mean(vals) for feat, vals in all_features.items()}
    std_imp = {feat: np.std(vals, ddof=1) if len(vals) > 1 else 0.0
               for feat, vals in all_features.items()}

    sorted_feats = sorted(mean_imp.items(), key=lambda x: -x[1])[:top_n]
    names = [f[0] for f in sorted_feats]
    means = [f[1] for f in sorted_feats]
    stds = [std_imp[f[0]] for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(names))))
    y_pos = np.arange(len(names))

    bars = ax.barh(y_pos, means, xerr=stds, height=0.65,
                   color="#3b82f6", edgecolor="white", capsize=3, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Gain Importance (normalized)")
    ax.set_title(f"Top {top_n} Features by XGBoost Gain (across {len(folds)} folds)", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


def _render_prediction_distributions(folds: List[Dict]) -> str:
    """Overlaid histograms of predicted probabilities for wins vs losses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_y = np.concatenate([f["y_test"] for f in folds])
    all_p = np.concatenate([f["y_proba"] for f in folds])

    win_probs = all_p[all_y == 1]
    loss_probs = all_p[all_y == 0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 25)

    ax.hist(win_probs, bins=bins, alpha=0.6, color="#16a34a", label=f"Wins (n={len(win_probs)})",
            density=True, edgecolor="white")
    ax.hist(loss_probs, bins=bins, alpha=0.6, color="#dc2626", label=f"Losses (n={len(loss_probs)})",
            density=True, edgecolor="white")

    # Overlap region shading
    mean_thresh = np.mean([f["optimal_threshold"] for f in folds])
    ax.axvline(mean_thresh, color="#1e293b", linestyle="--", lw=1.5,
               label=f"Mean optimal threshold ({mean_thresh:.3f})")

    ax.set_xlabel("Predicted probability of win")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Distributions: Wins vs Losses (pooled OOS)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    b64 = _fig_to_base64(fig)
    plt.close(fig)
    return b64


# ── HTML assembly ────────────────────────────────────────────────────────


def _build_html(
    folds: List[Dict],
    roc_b64: str,
    cal_b64: str,
    cm_b64: str,
    fi_b64: str,
    dist_b64: str,
    dataset_path: str,
    n_trades: int,
) -> str:
    """Assemble the full self-contained HTML report."""
    mean_auc = np.mean([f["auc"] for f in folds])
    mean_acc = np.mean([f["accuracy"] for f in folds])
    mean_prec = np.mean([f["precision"] for f in folds])
    mean_rec = np.mean([f["recall"] for f in folds])
    mean_brier = np.mean([f["brier"] for f in folds])
    std_auc = np.std([f["auc"] for f in folds], ddof=1) if len(folds) > 1 else 0.0

    def _auc_class(auc: float) -> str:
        if auc >= 0.80:
            return "good"
        if auc >= 0.65:
            return "ok"
        return "poor"

    # Summary table rows
    summary_rows = ""
    for f in folds:
        cls = _auc_class(f["auc"])
        train_str = ", ".join(str(y) for y in f["train_years"])
        summary_rows += (
            f'<tr>'
            f'<td>{f["fold"]}</td>'
            f'<td>{train_str}</td>'
            f'<td>{f["test_year"]}</td>'
            f'<td>{f["n_train"]}</td>'
            f'<td>{f["n_test"]}</td>'
            f'<td class="{cls}">{f["auc"]:.4f}</td>'
            f'<td>{f["accuracy"]:.4f}</td>'
            f'<td>{f["precision"]:.4f}</td>'
            f'<td>{f["recall"]:.4f}</td>'
            f'<td>{f["brier"]:.4f}</td>'
            f'<td>{f["optimal_threshold"]:.3f}</td>'
            f'</tr>\n'
        )

    # Averages row
    summary_rows += (
        f'<tr class="avg-row">'
        f'<td colspan="5"><strong>Mean ± Std</strong></td>'
        f'<td class="{_auc_class(mean_auc)}"><strong>{mean_auc:.4f} ± {std_auc:.4f}</strong></td>'
        f'<td><strong>{mean_acc:.4f}</strong></td>'
        f'<td><strong>{mean_prec:.4f}</strong></td>'
        f'<td><strong>{mean_rec:.4f}</strong></td>'
        f'<td><strong>{mean_brier:.4f}</strong></td>'
        f'<td></td>'
        f'</tr>\n'
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Model Diagnostics Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         margin: 0; padding: 2em 3em; background: #f8fafc; color: #1e293b; }}
  h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4em; }}
  h2 {{ color: #334155; margin-top: 2.5em; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 2em; }}
  .kpi-row {{ display: flex; gap: 1.5em; flex-wrap: wrap; margin: 1.5em 0; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
          padding: 1em 1.5em; min-width: 150px; flex: 1; text-align: center; }}
  .kpi .value {{ font-size: 1.8em; font-weight: 700; }}
  .kpi .label {{ font-size: 0.8em; color: #64748b; margin-top: 0.2em; }}
  .kpi .good {{ color: #16a34a; }}
  .kpi .ok {{ color: #d97706; }}
  .kpi .poor {{ color: #dc2626; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.9em; }}
  th {{ background: #f1f5f9; padding: 10px 12px; text-align: left;
       border-bottom: 2px solid #cbd5e1; font-weight: 600; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; }}
  tr:hover {{ background: #f8fafc; }}
  .avg-row td {{ background: #f1f5f9; border-top: 2px solid #cbd5e1; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .ok {{ color: #d97706; font-weight: 600; }}
  .poor {{ color: #dc2626; font-weight: 600; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; }}
  @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Model Diagnostics Dashboard</h1>
<div class="meta">
  <strong>Dataset:</strong> {dataset_path} ({n_trades} trades) &middot;
  <strong>Folds:</strong> {len(folds)} (year-based expanding window) &middot;
  <strong>Generated:</strong> {now}
</div>

<div class="kpi-row">
  <div class="kpi">
    <div class="value {_auc_class(mean_auc)}">{mean_auc:.4f}</div>
    <div class="label">Mean AUC (± {std_auc:.4f})</div>
  </div>
  <div class="kpi">
    <div class="value">{mean_acc:.1%}</div>
    <div class="label">Mean Accuracy</div>
  </div>
  <div class="kpi">
    <div class="value">{mean_prec:.1%}</div>
    <div class="label">Mean Precision</div>
  </div>
  <div class="kpi">
    <div class="value">{mean_rec:.1%}</div>
    <div class="label">Mean Recall</div>
  </div>
  <div class="kpi">
    <div class="value">{mean_brier:.4f}</div>
    <div class="label">Mean Brier Score</div>
  </div>
</div>

<h2>1. Per-Fold Summary</h2>
<table>
<thead>
<tr>
  <th>Fold</th><th>Train Years</th><th>Test Year</th>
  <th>Train N</th><th>Test N</th><th>AUC</th>
  <th>Accuracy</th><th>Precision</th><th>Recall</th>
  <th>Brier</th><th>Threshold</th>
</tr>
</thead>
<tbody>
{summary_rows}
</tbody>
</table>

<h2>2. ROC Curves</h2>
<div class="chart">
  <img src="data:image/png;base64,{roc_b64}" alt="ROC Curves">
</div>

<h2>3. Calibration &amp; Prediction Distribution</h2>
<div class="chart">
  <img src="data:image/png;base64,{cal_b64}" alt="Calibration Curve">
</div>

<h2>4. Prediction Distributions: Wins vs Losses</h2>
<div class="chart">
  <img src="data:image/png;base64,{dist_b64}" alt="Prediction Distributions">
</div>

<h2>5. Confusion Matrices (Optimal Threshold)</h2>
<div class="chart">
  <img src="data:image/png;base64,{cm_b64}" alt="Confusion Matrices">
</div>

<h2>6. Feature Importance (Top 15 by Gain)</h2>
<div class="chart">
  <img src="data:image/png;base64,{fi_b64}" alt="Feature Importance">
</div>

<footer>
  Generated by <code>compass/model_diagnostics.py</code> &middot; Walk-forward XGBoost diagnostics
</footer>

</body>
</html>"""

    return html


# ── Public API ───────────────────────────────────────────────────────────


def generate_diagnostics(
    csv_path: str,
    output_path: str = "reports/model_diagnostics.html",
    min_train_samples: int = 30,
) -> str:
    """Generate a self-contained HTML diagnostics report.

    Args:
        csv_path: Path to training data CSV (must have walk-forward columns).
        output_path: Where to write the HTML report.
        min_train_samples: Minimum training samples per fold.

    Returns:
        Absolute path to the generated report.
    """
    logger.info("Loading %s", csv_path)
    df = pd.read_csv(csv_path)
    n_trades = len(df)
    logger.info("Loaded %d trades", n_trades)

    # Collect per-fold data
    logger.info("Running walk-forward folds...")
    folds = _collect_fold_data(df, min_train_samples=min_train_samples)
    logger.info("Completed %d folds", len(folds))

    if not folds:
        raise ValueError("No valid folds produced — check data size")

    # Render all charts
    logger.info("Rendering charts...")
    roc_b64 = _render_roc_curves(folds)
    cal_b64 = _render_calibration_curve(folds)
    cm_b64 = _render_confusion_matrices(folds)
    fi_b64 = _render_feature_importance(folds)
    dist_b64 = _render_prediction_distributions(folds)

    # Assemble HTML
    html = _build_html(folds, roc_b64, cal_b64, cm_b64, fi_b64, dist_b64, csv_path, n_trades)

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    logger.info("Report written to %s", out)

    return str(out.resolve())


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    csv = sys.argv[1] if len(sys.argv) > 1 else "compass/training_data_combined.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "reports/model_diagnostics.html"
    generate_diagnostics(csv, out)
