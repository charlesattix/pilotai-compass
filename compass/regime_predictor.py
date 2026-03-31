"""Market regime predictor – forecasts regime transitions using macro features
with multi-horizon predictions, transition probability matrices, Platt-scaled
confidence calibration, and feature importance analysis.

Provides:
  1. Next-day regime prediction from macro features
  2. Multi-horizon forecasts (1d, 5d, 20d)
  3. Regime transition probability matrix from history
  4. Confidence calibration via Platt scaling (logistic sigmoid)
  5. Feature importance ranking
  6. HTML report with heatmaps, accuracy charts, and calibration curves
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Regime labels (aligned with compass.regime.Regime)
REGIMES: List[str] = ["bull", "bear", "high_vol", "low_vol", "crash"]

# Default macro feature columns
DEFAULT_FEATURES: List[str] = [
    "vix",
    "vix_term_structure",   # VIX3M / VIX ratio
    "credit_spread",        # HY OAS or similar
    "yield_curve_slope",    # 10Y - 2Y
    "momentum_20d",         # SPY 20-day momentum
]

DEFAULT_HORIZONS: List[int] = [1, 5, 20]


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class RegimeForecast:
    """Single-horizon regime forecast."""
    horizon_days: int
    predicted_regime: str
    probabilities: Dict[str, float]   # regime → probability
    confidence: float                 # calibrated confidence (0-1)


@dataclass
class TransitionMatrix:
    """Regime transition probability matrix."""
    matrix: Dict[str, Dict[str, float]]   # from_regime → to_regime → prob
    regimes: List[str]
    n_transitions: int
    stationary_dist: Dict[str, float]     # long-run regime distribution


@dataclass
class FeatureImportance:
    """Feature importance for regime prediction."""
    feature: str
    importance: float
    rank: int


@dataclass
class CalibrationPoint:
    """One bin on the calibration curve."""
    bin_mid: float
    predicted_prob: float
    actual_freq: float
    n_obs: int


@dataclass
class HorizonAccuracy:
    """Accuracy metrics at one forecast horizon."""
    horizon_days: int
    accuracy: float
    top2_accuracy: float    # correct regime in top-2 predictions
    n_test: int


@dataclass
class PredictorResult:
    """Complete regime prediction output."""
    forecasts: List[RegimeForecast] = field(default_factory=list)
    transition_matrix: Optional[TransitionMatrix] = None
    feature_importances: List[FeatureImportance] = field(default_factory=list)
    calibration_curve: List[CalibrationPoint] = field(default_factory=list)
    horizon_accuracies: List[HorizonAccuracy] = field(default_factory=list)
    current_regime: str = ""
    n_training_samples: int = 0
    generated_at: str = ""


# ── Core predictor ──────────────────────────────────────────────────────────
class RegimePredictor:
    """Predicts market regime using macro features with calibrated confidence."""

    def __init__(
        self,
        features: Optional[List[str]] = None,
        horizons: Optional[List[int]] = None,
        n_estimators: int = 100,
        max_depth: int = 4,
        test_size: float = 0.20,
        random_state: int = 42,
    ) -> None:
        self.features = features or list(DEFAULT_FEATURES)
        self.horizons = horizons or list(DEFAULT_HORIZONS)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.test_size = test_size
        self.random_state = random_state

        self._models: Dict[int, GradientBoostingClassifier] = {}
        self._calibrators: Dict[int, LogisticRegression] = {}
        self._classes: List[str] = []
        self._importances: List[FeatureImportance] = []

    # ── Public API ──────────────────────────────────────────────────────────
    def fit(
        self,
        features_df: pd.DataFrame,
        regime_series: pd.Series,
    ) -> PredictorResult:
        """Train models, compute transition matrix, calibrate, and return
        full analysis result.

        Parameters
        ----------
        features_df : pd.DataFrame
            Columns matching self.features, indexed by date.
        regime_series : pd.Series
            Regime labels aligned to same index.
        """
        features_df, regime_series = self._align(features_df, regime_series)
        if len(features_df) < 30:
            logger.warning("Too few samples (%d) for training", len(features_df))
            return PredictorResult(generated_at=self._now())

        self._classes = sorted(regime_series.unique().tolist())

        # Transition matrix from full history
        tm = self._compute_transition_matrix(regime_series)

        # Train per-horizon models
        accuracies: List[HorizonAccuracy] = []
        cal_points: List[CalibrationPoint] = []

        for h in self.horizons:
            X, y = self._prepare_horizon(features_df, regime_series, h)
            if len(X) < 20:
                continue
            acc, cal = self._train_horizon(h, X, y)
            accuracies.append(acc)
            if h == self.horizons[0]:
                cal_points = cal  # calibration from shortest horizon

        # Feature importance from first model
        self._importances = self._extract_importances()

        # Generate forecasts from latest observation
        forecasts = self._predict_latest(features_df)

        current = str(regime_series.iloc[-1]) if len(regime_series) > 0 else ""

        return PredictorResult(
            forecasts=forecasts,
            transition_matrix=tm,
            feature_importances=self._importances,
            calibration_curve=cal_points,
            horizon_accuracies=accuracies,
            current_regime=current,
            n_training_samples=len(features_df),
            generated_at=self._now(),
        )

    def predict(self, features_row: pd.DataFrame) -> List[RegimeForecast]:
        """Predict regimes for a single observation across all horizons."""
        return self._predict_latest(features_row)

    def generate_report(
        self,
        result: PredictorResult,
        output_path: str | Path = "reports/regime_predictions.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Regime prediction report written to %s", path)
        return path

    # ── Alignment ───────────────────────────────────────────────────────────
    def _align(
        self, features_df: pd.DataFrame, regime_series: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        cols = [c for c in self.features if c in features_df.columns]
        if not cols:
            return pd.DataFrame(), pd.Series(dtype=str)
        df = features_df[cols].copy()
        idx = df.index.intersection(regime_series.index)
        df = df.loc[idx].dropna()
        rs = regime_series.loc[df.index]
        return df, rs

    # ── Transition matrix ───────────────────────────────────────────────────
    @staticmethod
    def _compute_transition_matrix(regimes: pd.Series) -> TransitionMatrix:
        labels = sorted(regimes.unique().tolist())
        counts: Dict[str, Dict[str, int]] = {r: {r2: 0 for r2 in labels} for r in labels}
        vals = regimes.values
        n_trans = 0
        for i in range(len(vals) - 1):
            fr, to = str(vals[i]), str(vals[i + 1])
            if fr in counts and to in counts[fr]:
                counts[fr][to] += 1
                n_trans += 1

        # Normalize to probabilities
        matrix: Dict[str, Dict[str, float]] = {}
        for fr in labels:
            row_sum = sum(counts[fr].values())
            matrix[fr] = {
                to: counts[fr][to] / row_sum if row_sum > 0 else 0.0
                for to in labels
            }

        # Stationary distribution via power iteration
        stat = _stationary_distribution(matrix, labels)

        return TransitionMatrix(
            matrix=matrix,
            regimes=labels,
            n_transitions=n_trans,
            stationary_dist=stat,
        )

    # ── Horizon data prep ───────────────────────────────────────────────────
    @staticmethod
    def _prepare_horizon(
        features_df: pd.DataFrame, regimes: pd.Series, horizon: int,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        y = regimes.shift(-horizon).dropna()
        common = features_df.index.intersection(y.index)
        return features_df.loc[common], y.loc[common]

    # ── Training ────────────────────────────────────────────────────────────
    def _train_horizon(
        self, horizon: int, X: pd.DataFrame, y: pd.Series,
    ) -> Tuple[HorizonAccuracy, List[CalibrationPoint]]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            shuffle=False,
        )

        model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)
        self._models[horizon] = model

        # Accuracy
        preds = model.predict(X_test)
        accuracy = float(np.mean(preds == y_test))

        # Top-2 accuracy
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            top2_acc = self._top_k_accuracy(proba, y_test, model.classes_, k=2)
        else:
            top2_acc = accuracy

        # Platt scaling (calibration)
        cal_points: List[CalibrationPoint] = []
        if hasattr(model, "predict_proba") and len(X_test) > 10:
            train_proba = model.predict_proba(X_train)
            cal = LogisticRegression(random_state=self.random_state, max_iter=1000)
            cal.fit(train_proba, y_train)
            self._calibrators[horizon] = cal

            # Calibration curve from test set
            test_proba = model.predict_proba(X_test)
            max_probs = np.max(test_proba, axis=1)
            correct = (preds == y_test.values).astype(float)
            cal_points = self._calibration_curve(max_probs, correct)

        return (
            HorizonAccuracy(
                horizon_days=horizon,
                accuracy=accuracy,
                top2_accuracy=top2_acc,
                n_test=len(X_test),
            ),
            cal_points,
        )

    @staticmethod
    def _top_k_accuracy(
        proba: np.ndarray, y_true: pd.Series, classes: np.ndarray, k: int,
    ) -> float:
        top_k_classes = classes[np.argsort(-proba, axis=1)[:, :k]]
        hits = sum(1 for i, true in enumerate(y_true) if true in top_k_classes[i])
        return hits / len(y_true) if len(y_true) > 0 else 0.0

    @staticmethod
    def _calibration_curve(
        predicted_probs: np.ndarray, correct: np.ndarray, n_bins: int = 10,
    ) -> List[CalibrationPoint]:
        bins = np.linspace(0, 1, n_bins + 1)
        points: List[CalibrationPoint] = []
        for i in range(n_bins):
            mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
            n = int(mask.sum())
            if n == 0:
                continue
            points.append(CalibrationPoint(
                bin_mid=float((bins[i] + bins[i + 1]) / 2),
                predicted_prob=float(predicted_probs[mask].mean()),
                actual_freq=float(correct[mask].mean()),
                n_obs=n,
            ))
        return points

    # ── Feature importance ──────────────────────────────────────────────────
    def _extract_importances(self) -> List[FeatureImportance]:
        if not self._models:
            return []
        model = next(iter(self._models.values()))
        if not hasattr(model, "feature_importances_"):
            return []
        imp = model.feature_importances_
        cols = self.features[:len(imp)]
        ranked = sorted(zip(cols, imp), key=lambda x: -x[1])
        return [
            FeatureImportance(feature=f, importance=float(v), rank=i + 1)
            for i, (f, v) in enumerate(ranked)
        ]

    # ── Prediction ──────────────────────────────────────────────────────────
    def _predict_latest(self, features_df: pd.DataFrame) -> List[RegimeForecast]:
        if features_df.empty or not self._models:
            return []
        cols = [c for c in self.features if c in features_df.columns]
        row = features_df[cols].iloc[[-1]].dropna(axis=1)
        if row.empty:
            return []

        forecasts: List[RegimeForecast] = []
        for h in self.horizons:
            model = self._models.get(h)
            if model is None:
                continue
            # Ensure feature alignment
            missing = set(model.feature_names_in_) - set(row.columns) if hasattr(model, "feature_names_in_") else set()
            if missing:
                continue

            pred = str(model.predict(row)[0])
            proba = model.predict_proba(row)[0]
            prob_dict = {str(c): float(p) for c, p in zip(model.classes_, proba)}

            # Calibrated confidence
            confidence = float(max(proba))
            calibrator = self._calibrators.get(h)
            if calibrator is not None:
                try:
                    cal_proba = calibrator.predict_proba(proba.reshape(1, -1))
                    confidence = float(np.max(cal_proba))
                except Exception:
                    pass

            forecasts.append(RegimeForecast(
                horizon_days=h,
                predicted_regime=pred,
                probabilities=prob_dict,
                confidence=confidence,
            ))
        return forecasts

    # ── Helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: PredictorResult) -> str:
        cards = self._html_cards(r)
        tm_heatmap = self._svg_transition_heatmap(r.transition_matrix)
        acc_tbl = self._html_accuracy(r.horizon_accuracies)
        imp_bars = self._svg_importance_bars(r.feature_importances)
        cal_chart = self._svg_calibration(r.calibration_curve)
        forecast_tbl = self._html_forecasts(r.forecasts)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Regime Predictions</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
.sec{{margin-bottom:32px}}
.sec h2{{font-size:1.1rem;margin-bottom:12px;color:#38bdf8}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8;font-weight:600}}
tr:hover{{background:#1e293b}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}.warn{{color:#fbbf24}}
svg{{display:block;margin:0 auto}}
</style>
</head>
<body>
<h1>Market Regime Predictions</h1>
<p class="sub">Generated {r.generated_at or 'N/A'} &middot; {r.n_training_samples} training samples &middot; Current: <strong>{r.current_regime.upper() or 'N/A'}</strong></p>

{cards}
{forecast_tbl}

<div class="sec">
<h2>Transition Probability Matrix</h2>
{tm_heatmap}
</div>

{acc_tbl}

<div class="sec">
<h2>Feature Importance</h2>
{imp_bars}
</div>

<div class="sec">
<h2>Calibration Curve</h2>
{cal_chart}
</div>

</body>
</html>"""

    @staticmethod
    def _html_cards(r: PredictorResult) -> str:
        f1 = r.forecasts[0] if r.forecasts else None
        pred = f1.predicted_regime.upper() if f1 else "N/A"
        conf = f"{f1.confidence:.0%}" if f1 else "N/A"
        acc1 = next((a for a in r.horizon_accuracies if a.horizon_days == 1), None)
        acc_str = f"{acc1.accuracy:.1%}" if acc1 else "N/A"
        return f"""<div class="grid">
<div class="card"><div class="lbl">Next-Day Forecast</div><div class="val">{pred}</div></div>
<div class="card"><div class="lbl">Confidence</div><div class="val">{conf}</div></div>
<div class="card"><div class="lbl">1d Accuracy</div><div class="val">{acc_str}</div></div>
<div class="card"><div class="lbl">Training Samples</div><div class="val">{r.n_training_samples}</div></div>
</div>"""

    @staticmethod
    def _html_forecasts(forecasts: List[RegimeForecast]) -> str:
        if not forecasts:
            return ""
        rows = ""
        for f in forecasts:
            probs = " / ".join(f"{r}: {p:.0%}" for r, p in sorted(f.probabilities.items(), key=lambda x: -x[1])[:3])
            rows += (
                f"<tr><td>{f.horizon_days}d</td>"
                f"<td><strong>{f.predicted_regime}</strong></td>"
                f"<td>{f.confidence:.1%}</td>"
                f"<td>{probs}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Multi-Horizon Forecasts</h2>
<table>
<thead><tr><th>Horizon</th><th>Predicted</th><th>Confidence</th><th>Top Probabilities</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _svg_transition_heatmap(tm: Optional[TransitionMatrix]) -> str:
        if not tm:
            return "<p>No transition data.</p>"
        regs = tm.regimes
        n = len(regs)
        cell = 60
        lbl_w = 70
        w = lbl_w + n * cell + 10
        h = 30 + n * cell + 30

        cells = ""
        for i, fr in enumerate(regs):
            # Row label
            cells += (
                f'<text x="{lbl_w - 5}" y="{35 + i * cell + cell // 2}" '
                f'text-anchor="end" font-size="11" fill="#e2e8f0">{fr}</text>'
            )
            for j, to in enumerate(regs):
                p = tm.matrix[fr][to]
                x = lbl_w + j * cell
                y = 30 + i * cell
                intensity = min(255, int(p * 400))
                colour = f"rgb({30},{intensity},{80})"
                cells += (
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{colour}" stroke="#0f172a" stroke-width="1"/>'
                    f'<text x="{x + cell // 2}" y="{y + cell // 2 + 4}" '
                    f'text-anchor="middle" font-size="10" fill="#e2e8f0">{p:.0%}</text>'
                )
        # Column headers
        for j, to in enumerate(regs):
            x = lbl_w + j * cell + cell // 2
            cells += f'<text x="{x}" y="20" text-anchor="middle" font-size="10" fill="#94a3b8">{to}</text>'

        return f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">{cells}</svg>'

    @staticmethod
    def _html_accuracy(accs: List[HorizonAccuracy]) -> str:
        if not accs:
            return ""
        rows = ""
        for a in accs:
            rows += (
                f"<tr><td>{a.horizon_days}d</td>"
                f"<td>{a.accuracy:.1%}</td>"
                f"<td>{a.top2_accuracy:.1%}</td>"
                f"<td>{a.n_test}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Forecast Accuracy by Horizon</h2>
<table>
<thead><tr><th>Horizon</th><th>Accuracy</th><th>Top-2 Accuracy</th><th>Test Samples</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    @staticmethod
    def _svg_importance_bars(imps: List[FeatureImportance]) -> str:
        if not imps:
            return "<p>No importance data.</p>"
        w, h_chart = 500, 30 * len(imps) + 20
        pl = 140
        max_imp = max(f.importance for f in imps) or 1.0
        bars = ""
        for i, f in enumerate(imps):
            y = 10 + i * 30
            bw = f.importance / max_imp * (w - pl - 30)
            bars += (
                f'<text x="{pl - 5}" y="{y + 14}" text-anchor="end" font-size="11" fill="#e2e8f0">{f.feature}</text>'
                f'<rect x="{pl}" y="{y}" width="{bw:.0f}" height="20" rx="3" fill="#38bdf8" opacity="0.8"/>'
                f'<text x="{pl + bw + 5}" y="{y + 14}" font-size="10" fill="#94a3b8">{f.importance:.3f}</text>'
            )
        return f'<svg viewBox="0 0 {w} {h_chart}" width="{w}" xmlns="http://www.w3.org/2000/svg">{bars}</svg>'

    @staticmethod
    def _svg_calibration(cal: List[CalibrationPoint]) -> str:
        if not cal:
            return "<p>No calibration data.</p>"
        w, h = 300, 300
        pad = 40
        ch = h - 2 * pad

        # Perfect calibration line
        diag = f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{pad}" stroke="#475569" stroke-width="1" stroke-dasharray="4"/>'

        pts = ""
        for c in cal:
            x = pad + c.predicted_prob * ch
            y = h - pad - c.actual_freq * ch
            pts += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="5" fill="#f97316"/>'

        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'{diag}{pts}'
            f'<text x="{w // 2}" y="{h - 5}" text-anchor="middle" font-size="10" fill="#94a3b8">Predicted</text>'
            f'<text x="10" y="{h // 2}" font-size="10" fill="#94a3b8" transform="rotate(-90 10 {h // 2})">Actual</text>'
            f'</svg>'
        )


# ── Utility: stationary distribution ────────────────────────────────────────
def _stationary_distribution(
    matrix: Dict[str, Dict[str, float]], regimes: List[str], iters: int = 100,
) -> Dict[str, float]:
    """Power iteration for stationary distribution of transition matrix."""
    n = len(regimes)
    if n == 0:
        return {}
    P = np.zeros((n, n))
    for i, fr in enumerate(regimes):
        for j, to in enumerate(regimes):
            P[i, j] = matrix.get(fr, {}).get(to, 0.0)

    pi = np.ones(n) / n
    for _ in range(iters):
        pi = pi @ P
        s = pi.sum()
        if s > 0:
            pi /= s

    return {r: float(pi[i]) for i, r in enumerate(regimes)}
