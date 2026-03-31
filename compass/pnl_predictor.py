"""Pre-trade P&L prediction engine — gradient boosted model predicting trade
P&L distribution from market state features, with confidence intervals,
go/no-go recommendations, calibration tracking, and position sizer integration.

Provides:
  1. Feature engineering from market state (VIX, term structure, returns, volume)
  2. Gradient boosted model predicting P&L distribution
  3. Confidence intervals on predicted P&L
  4. Go/no-go recommendation based on predicted risk-adjusted return
  5. Calibration tracking (predicted vs actual)
  6. Position sizer integration (scale size by confidence)
  7. HTML report with accuracy, calibration, importance, prediction log
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "vix", "vix_term_structure", "return_1d", "return_5d",
    "return_20d", "volume_ratio", "iv_rank", "spread_width",
]

DEFAULT_GO_THRESHOLD = 0.5   # minimum predicted risk-adj return
DEFAULT_CONFIDENCE = 0.80    # confidence level for intervals


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class Prediction:
    """Single trade P&L prediction."""
    predicted_pnl: float
    confidence_low: float
    confidence_high: float
    confidence_level: float
    go_decision: bool
    risk_adjusted_return: float
    recommended_size: float      # 0–1 scale factor
    feature_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class CalibrationBin:
    """One bin on the calibration curve."""
    bin_mid: float
    avg_predicted: float
    avg_actual: float
    n_obs: int


@dataclass
class FeatureImportance:
    """Feature importance for P&L prediction."""
    feature: str
    importance: float
    rank: int


@dataclass
class PredictionLogEntry:
    """Historical prediction vs actual."""
    timestamp: str
    predicted_pnl: float
    actual_pnl: float
    error: float
    go_decision: bool
    was_correct: bool            # go + profitable, or no-go + unprofitable


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    mae: float = 0.0
    rmse: float = 0.0
    r_squared: float = 0.0
    direction_accuracy: float = 0.0  # % correct sign
    go_precision: float = 0.0       # % of go decisions that were profitable
    n_train: int = 0
    n_test: int = 0


@dataclass
class PredictorResult:
    """Complete predictor output."""
    metrics: Optional[ModelMetrics] = None
    importances: List[FeatureImportance] = field(default_factory=list)
    calibration: List[CalibrationBin] = field(default_factory=list)
    prediction_log: List[PredictionLogEntry] = field(default_factory=list)
    n_predictions: int = 0
    generated_at: str = ""


# ── Core predictor ──────────────────────────────────────────────────────────
class PnLPredictor:
    """Pre-trade P&L prediction with gradient boosting."""

    def __init__(
        self,
        features: Optional[List[str]] = None,
        go_threshold: float = DEFAULT_GO_THRESHOLD,
        confidence_level: float = DEFAULT_CONFIDENCE,
        n_estimators: int = 100,
        max_depth: int = 4,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        self.features = features or list(DEFAULT_FEATURES)
        self.go_threshold = go_threshold
        self.confidence_level = confidence_level
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state

        self._model: Optional[GradientBoostingRegressor] = None
        self._residual_std: float = 0.0
        self._importances: List[FeatureImportance] = []
        self._log: List[PredictionLogEntry] = []

    # ── Training ────────────────────────────────────────────────────────────
    def fit(
        self,
        features_df: pd.DataFrame,
        pnl_series: pd.Series,
    ) -> PredictorResult:
        """Train the model and return evaluation metrics.

        Parameters
        ----------
        features_df : pd.DataFrame
            Columns matching self.features.
        pnl_series : pd.Series
            Actual trade P&L aligned to same index.
        """
        cols = [c for c in self.features if c in features_df.columns]
        if not cols or len(features_df) < 20:
            return PredictorResult(generated_at=self._now())

        common = features_df.index.intersection(pnl_series.index)
        X = features_df.loc[common, cols].dropna()
        y = pnl_series.loc[X.index]

        if len(X) < 20:
            return PredictorResult(generated_at=self._now())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            shuffle=False,
        )

        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self._model.fit(X_train, y_train)

        # Residual std for confidence intervals
        train_pred = self._model.predict(X_train)
        residuals = y_train.values - train_pred
        self._residual_std = float(np.std(residuals))

        # Evaluate on test
        test_pred = self._model.predict(X_test)
        test_actual = y_test.values

        metrics = self._compute_metrics(test_pred, test_actual, len(X_train), len(X_test))

        # Importances
        imp = self._model.feature_importances_
        ranked = sorted(zip(cols, imp), key=lambda x: -x[1])
        self._importances = [
            FeatureImportance(f, float(v), i + 1)
            for i, (f, v) in enumerate(ranked)
        ]

        # Calibration from test set
        calibration = self._calibration_curve(test_pred, test_actual)

        # Prediction log from test set
        log_entries: List[PredictionLogEntry] = []
        for i in range(len(test_pred)):
            pred = float(test_pred[i])
            actual = float(test_actual[i])
            go = pred > self.go_threshold
            correct = (go and actual > 0) or (not go and actual <= 0)
            log_entries.append(PredictionLogEntry(
                timestamp=str(y_test.index[i]),
                predicted_pnl=pred,
                actual_pnl=actual,
                error=actual - pred,
                go_decision=go,
                was_correct=correct,
            ))
        self._log = log_entries

        return PredictorResult(
            metrics=metrics,
            importances=self._importances,
            calibration=calibration,
            prediction_log=log_entries,
            n_predictions=len(log_entries),
            generated_at=self._now(),
        )

    # ── Prediction ──────────────────────────────────────────────────────────
    def predict(self, features: Dict[str, float]) -> Prediction:
        """Predict P&L for a single trade.

        Parameters
        ----------
        features : dict
            Feature name → value for current market state.
        """
        if self._model is None:
            return Prediction(0, 0, 0, self.confidence_level, False, 0, 0, features)

        # Use the exact feature columns the model was trained on
        trained_cols = list(self._model.feature_names_in_) if hasattr(self._model, "feature_names_in_") else self.features
        X = np.array([[features.get(c, 0.0) for c in trained_cols]])
        pred = float(self._model.predict(X)[0])

        # Confidence interval from residual distribution
        z = _z_score(self.confidence_level)
        ci_low = pred - z * self._residual_std
        ci_high = pred + z * self._residual_std

        # Risk-adjusted return: pred / std (like Sharpe of this trade)
        rar = pred / self._residual_std if self._residual_std > 1e-9 else 0.0

        # Go decision
        go = rar > self.go_threshold

        # Position size recommendation: scale by confidence
        if go:
            # Scale from 0.5 to 1.0 based on how far above threshold
            excess = min(rar / self.go_threshold, 2.0) - 1.0  # 0 to 1
            size = 0.5 + excess * 0.5
        else:
            size = 0.0

        return Prediction(
            predicted_pnl=pred,
            confidence_low=ci_low,
            confidence_high=ci_high,
            confidence_level=self.confidence_level,
            go_decision=go,
            risk_adjusted_return=rar,
            recommended_size=round(size, 2),
            feature_values=features,
        )

    def record_actual(self, predicted_pnl: float, actual_pnl: float, go: bool) -> None:
        """Record an actual outcome for calibration tracking."""
        correct = (go and actual_pnl > 0) or (not go and actual_pnl <= 0)
        self._log.append(PredictionLogEntry(
            timestamp=self._now(),
            predicted_pnl=predicted_pnl,
            actual_pnl=actual_pnl,
            error=actual_pnl - predicted_pnl,
            go_decision=go,
            was_correct=correct,
        ))

    def get_log(self) -> List[PredictionLogEntry]:
        return list(self._log)

    # ── Metrics ─────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_metrics(
        pred: np.ndarray, actual: np.ndarray, n_train: int, n_test: int,
    ) -> ModelMetrics:
        errors = actual - pred
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        # Direction accuracy
        dir_acc = float(np.mean(np.sign(pred) == np.sign(actual)))

        # Go precision: of trades where pred > 0, how many actual > 0
        go_mask = pred > 0
        if go_mask.sum() > 0:
            go_prec = float(np.mean(actual[go_mask] > 0))
        else:
            go_prec = 0.0

        return ModelMetrics(
            mae=mae, rmse=rmse, r_squared=r2,
            direction_accuracy=dir_acc, go_precision=go_prec,
            n_train=n_train, n_test=n_test,
        )

    @staticmethod
    def _calibration_curve(
        pred: np.ndarray, actual: np.ndarray, n_bins: int = 10,
    ) -> List[CalibrationBin]:
        if len(pred) < n_bins:
            return []
        bins = np.linspace(pred.min(), pred.max(), n_bins + 1)
        result: List[CalibrationBin] = []
        for i in range(n_bins):
            mask = (pred >= bins[i]) & (pred < bins[i + 1])
            if i == n_bins - 1:
                mask = (pred >= bins[i]) & (pred <= bins[i + 1])
            n = int(mask.sum())
            if n == 0:
                continue
            result.append(CalibrationBin(
                bin_mid=float((bins[i] + bins[i + 1]) / 2),
                avg_predicted=float(pred[mask].mean()),
                avg_actual=float(actual[mask].mean()),
                n_obs=n,
            ))
        return result

    # ── Report ──────────────────────────────────────────────────────────────
    def generate_report(
        self,
        result: PredictorResult,
        output_path: str | Path = "reports/pnl_predictor.html",
    ) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("PnL predictor report written to %s", path)
        return path

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML ────────────────────────────────────────────────────────────────
    def _build_html(self, r: PredictorResult) -> str:
        cards = self._html_cards(r)
        metrics_tbl = self._html_metrics(r.metrics)
        imp_chart = self._svg_importances(r.importances)
        cal_chart = self._svg_calibration(r.calibration)
        log_tbl = self._html_log(r.prediction_log)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>P&L Predictor</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}
svg{{display:block;margin:0 auto}}
</style>
</head>
<body>
<h1>Pre-Trade P&L Predictor</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_predictions} predictions evaluated</p>
{cards}
{metrics_tbl}
<div class="sec"><h2>Feature Importance</h2>{imp_chart}</div>
<div class="sec"><h2>Calibration (Predicted vs Actual)</h2>{cal_chart}</div>
{log_tbl}
</body>
</html>"""

    @staticmethod
    def _html_cards(r: PredictorResult) -> str:
        m = r.metrics
        if not m:
            return ""
        return f"""<div class="grid">
<div class="card"><div class="lbl">MAE</div><div class="val">{m.mae:.2f}</div></div>
<div class="card"><div class="lbl">RMSE</div><div class="val">{m.rmse:.2f}</div></div>
<div class="card"><div class="lbl">R-squared</div><div class="val">{m.r_squared:.3f}</div></div>
<div class="card"><div class="lbl">Direction Acc</div><div class="val">{m.direction_accuracy:.0%}</div></div>
<div class="card"><div class="lbl">Go Precision</div><div class="val">{m.go_precision:.0%}</div></div>
<div class="card"><div class="lbl">Predictions</div><div class="val">{r.n_predictions}</div></div>
</div>"""

    @staticmethod
    def _html_metrics(m: Optional[ModelMetrics]) -> str:
        if not m:
            return ""
        return f"""<div class="sec"><h2>Model Performance</h2>
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td>Mean Absolute Error</td><td>{m.mae:.4f}</td></tr>
<tr><td>Root Mean Squared Error</td><td>{m.rmse:.4f}</td></tr>
<tr><td>R-squared</td><td>{m.r_squared:.4f}</td></tr>
<tr><td>Direction Accuracy</td><td>{m.direction_accuracy:.1%}</td></tr>
<tr><td>Go Precision</td><td>{m.go_precision:.1%}</td></tr>
<tr><td>Train / Test Split</td><td>{m.n_train} / {m.n_test}</td></tr>
</tbody></table></div>"""

    @staticmethod
    def _svg_importances(imps: List[FeatureImportance]) -> str:
        if not imps:
            return "<p>No data.</p>"
        w = 480
        h_chart = 28 * len(imps) + 20
        pl = 140
        max_imp = max(f.importance for f in imps) or 1.0
        bars = ""
        for i, f in enumerate(imps):
            y = 10 + i * 28
            bw = max(2, f.importance / max_imp * (w - pl - 40))
            bars += (
                f'<text x="{pl - 5}" y="{y + 14}" text-anchor="end" font-size="11" fill="#e2e8f0">{f.feature}</text>'
                f'<rect x="{pl}" y="{y}" width="{bw:.0f}" height="20" rx="3" fill="#38bdf8" opacity="0.8"/>'
                f'<text x="{pl + bw + 5}" y="{y + 14}" font-size="10" fill="#94a3b8">{f.importance:.3f}</text>'
            )
        return f'<svg viewBox="0 0 {w} {h_chart}" width="{w}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'

    @staticmethod
    def _svg_calibration(cal: List[CalibrationBin]) -> str:
        if not cal:
            return "<p>No calibration data.</p>"
        w, h = 300, 300
        pad = 40
        ch = h - 2 * pad
        # Perfect line
        diag = f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{pad}" stroke="#475569" stroke-width="1" stroke-dasharray="4"/>'
        vals = [c.avg_predicted for c in cal] + [c.avg_actual for c in cal]
        mn, mx = min(vals), max(vals)
        rng = mx - mn or 1

        dots = ""
        for c in cal:
            x = pad + (c.avg_predicted - mn) / rng * ch
            y = h - pad - (c.avg_actual - mn) / rng * ch
            dots += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="5" fill="#f97316"/>'

        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'{diag}{dots}'
            f'<text x="{w // 2}" y="{h - 5}" text-anchor="middle" font-size="10" fill="#94a3b8">Predicted</text>'
            f'<text x="10" y="{h // 2}" font-size="10" fill="#94a3b8" transform="rotate(-90 10 {h // 2})">Actual</text>'
            f'</svg>'
        )

    @staticmethod
    def _html_log(log: List[PredictionLogEntry]) -> str:
        if not log:
            return ""
        rows = ""
        for e in log[-30:]:
            p_cls = "pos" if e.predicted_pnl > 0 else "neg"
            a_cls = "pos" if e.actual_pnl > 0 else "neg"
            c_cls = "pos" if e.was_correct else "neg"
            rows += (
                f'<tr><td>{e.timestamp}</td>'
                f'<td class="{p_cls}">{e.predicted_pnl:.2f}</td>'
                f'<td class="{a_cls}">{e.actual_pnl:.2f}</td>'
                f'<td>{e.error:.2f}</td>'
                f'<td>{"GO" if e.go_decision else "NO-GO"}</td>'
                f'<td class="{c_cls}">{"Yes" if e.was_correct else "No"}</td></tr>'
            )
        return f"""<div class="sec"><h2>Recent Prediction Log</h2>
<table><thead><tr><th>Time</th><th>Predicted</th><th>Actual</th><th>Error</th><th>Decision</th><th>Correct</th></tr></thead>
<tbody>{rows}</tbody></table></div>"""


# ── Utilities ───────────────────────────────────────────────────────────────
def _z_score(confidence: float) -> float:
    """Approximate z-score for a confidence level."""
    # Common values
    table = {0.80: 1.282, 0.85: 1.440, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    if confidence in table:
        return table[confidence]
    # Rough approximation for others
    p = (1 + confidence) / 2
    # Rational approximation (Abramowitz & Stegun)
    if p >= 1.0:
        return 3.5
    t = math.sqrt(-2 * math.log(1 - p))
    return t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (
        1 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t
    )
