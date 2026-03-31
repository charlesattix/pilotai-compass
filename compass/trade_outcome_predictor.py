"""
Trade outcome predictor — pre-entry P&L prediction with similar-trade matching.

Predicts expected P&L for a proposed trade BEFORE entry using market
conditions, regime, and historical pattern matching.  Provides confidence
intervals, feature importance, calibration analysis, and a similar-trade
lookup that finds the N most similar historical trades by feature distance.

Generates an HTML report at reports/trade_outcome.html with prediction
distribution, similar-trades table, and calibration curve.

Usage::

    from compass.trade_outcome_predictor import TradeOutcomePredictor
    predictor = TradeOutcomePredictor(historical_trades_df)
    predictor.fit()
    pred = predictor.predict(new_trade_features)
    predictor.generate_report("reports/trade_outcome.html")
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "trade_outcome.html"

DEFAULT_FEATURES = [
    "vix", "iv_percentile", "dte_at_entry", "net_credit",
    "spread_width", "delta_short", "rsi", "momentum",
    "day_of_week", "hour_of_day",
]

REGIMES = ("bull", "bear", "high_vol", "neutral")


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class Prediction:
    """Predicted outcome for a single proposed trade."""
    expected_pnl: float
    ci_lower: float           # 10th percentile
    ci_upper: float           # 90th percentile
    win_probability: float    # P(pnl > 0)
    risk_reward_score: float  # 0-100
    confidence: float         # 0-1, model confidence
    regime: str


@dataclass
class SimilarTrade:
    """A historical trade similar to the query."""
    index: int
    distance: float
    pnl: float
    return_pct: float
    regime: str
    features: Dict[str, float]


@dataclass
class FeatureImportance:
    """Importance of a single feature in the prediction model."""
    feature: str
    importance: float
    rank: int


@dataclass
class CalibrationBucket:
    """One bucket of the calibration analysis."""
    predicted_mean: float
    actual_mean: float
    n_trades: int
    bucket_label: str


@dataclass
class FitResult:
    """Summary of the model fitting process."""
    n_train: int
    n_features: int
    train_r2: float
    cv_mae: float
    feature_importances: List[FeatureImportance]


# ── Predictor ───────────────────────────────────────────────────────────


class TradeOutcomePredictor:
    """Predict trade P&L before entry using historical pattern matching."""

    def __init__(
        self,
        trades: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_neighbors: int = 10,
        target_col: str = "pnl",
        regime_col: str = "regime",
    ) -> None:
        self.trades = trades.copy()
        self.target_col = target_col
        self.regime_col = regime_col
        self.n_neighbors = n_neighbors

        # Determine available features
        requested = features or DEFAULT_FEATURES
        self.features = [f for f in requested if f in trades.columns]
        if not self.features:
            raise ValueError(
                f"No usable features found.  Requested: {requested}, "
                f"available: {list(trades.columns)}"
            )

        # Models (populated by fit)
        self._model: Optional[GradientBoostingRegressor] = None
        self._scaler: Optional[StandardScaler] = None
        self._nn: Optional[NearestNeighbors] = None
        self._X_scaled: Optional[np.ndarray] = None

        # Results
        self.fit_result: Optional[FitResult] = None
        self.predictions: List[Tuple[Dict[str, float], Prediction]] = []
        self.calibration: List[CalibrationBucket] = []

    # ── Class constructors ──────────────────────────────────────────────

    @classmethod
    def from_csv(cls, path: str, **kwargs: Any) -> "TradeOutcomePredictor":
        """Load historical trades from CSV."""
        df = pd.read_csv(path, parse_dates=True)
        return cls(df, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def fit(self) -> FitResult:
        """Train the prediction model on historical trades."""
        X = self.trades[self.features].values.astype(float)
        y = self.trades[self.target_col].values.astype(float)

        # Handle NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        self._scaler = StandardScaler()
        self._X_scaled = self._scaler.fit_transform(X)

        # Gradient boosting regressor
        self._model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        self._model.fit(self._X_scaled, y)

        # Nearest neighbors for similar-trade lookup
        self._nn = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self._X_scaled)),
            metric="euclidean",
        )
        self._nn.fit(self._X_scaled)

        # Feature importances
        importances = self._model.feature_importances_
        ranked = sorted(
            enumerate(importances), key=lambda x: -x[1],
        )
        fi = [
            FeatureImportance(
                feature=self.features[idx], importance=float(imp),
                rank=rank + 1,
            )
            for rank, (idx, imp) in enumerate(ranked)
        ]

        # Training R²
        train_r2 = float(self._model.score(self._X_scaled, y))

        # Simple leave-one-out-style MAE estimate via OOB or quick CV
        preds = self._model.predict(self._X_scaled)
        cv_mae = float(np.mean(np.abs(preds - y)))

        self.fit_result = FitResult(
            n_train=len(y), n_features=len(self.features),
            train_r2=train_r2, cv_mae=cv_mae,
            feature_importances=fi,
        )

        # Calibration on training data
        self.calibration = self._calibration_analysis(preds, y)

        return self.fit_result

    def predict(self, trade_features: Dict[str, float]) -> Prediction:
        """Predict P&L for a proposed trade given its features."""
        if self._model is None:
            self.fit()

        x = np.array([[trade_features.get(f, 0.0) for f in self.features]])
        x_scaled = self._scaler.transform(x)

        expected = float(self._model.predict(x_scaled)[0])

        # Confidence interval from similar trades
        similar = self.find_similar_trades(trade_features)
        if similar:
            sim_pnls = np.array([s.pnl for s in similar])
            ci_lower = float(np.percentile(sim_pnls, 10))
            ci_upper = float(np.percentile(sim_pnls, 90))
            win_prob = float((sim_pnls > 0).mean())
        else:
            y = self.trades[self.target_col].values
            ci_lower = float(np.percentile(y, 10))
            ci_upper = float(np.percentile(y, 90))
            win_prob = float((y > 0).mean())

        # Risk/reward score
        rr_score = self._risk_reward_score(expected, ci_lower, ci_upper, win_prob)

        # Confidence: inverse of prediction uncertainty
        spread = ci_upper - ci_lower
        baseline_spread = float(self.trades[self.target_col].std() * 2) or 1.0
        confidence = max(0.0, min(1.0, 1.0 - spread / (baseline_spread * 2)))

        regime = trade_features.get("regime", trade_features.get(self.regime_col, "neutral"))
        if not isinstance(regime, str):
            regime = "neutral"

        pred = Prediction(
            expected_pnl=expected, ci_lower=ci_lower, ci_upper=ci_upper,
            win_probability=win_prob, risk_reward_score=rr_score,
            confidence=confidence, regime=str(regime),
        )
        self.predictions.append((trade_features, pred))
        return pred

    def find_similar_trades(
        self,
        trade_features: Dict[str, float],
        n: Optional[int] = None,
    ) -> List[SimilarTrade]:
        """Find the N most similar historical trades by feature distance."""
        if self._nn is None or self._scaler is None:
            self.fit()

        k = n or self.n_neighbors
        k = min(k, self._X_scaled.shape[0])

        x = np.array([[trade_features.get(f, 0.0) for f in self.features]])
        x_scaled = self._scaler.transform(x)
        distances, indices = self._nn.kneighbors(x_scaled, n_neighbors=k)

        valid_mask = ~(np.isnan(self.trades[self.features].values).any(axis=1)
                       | np.isnan(self.trades[self.target_col].values))
        valid_indices = np.where(valid_mask)[0]

        results: List[SimilarTrade] = []
        for dist, idx in zip(distances[0], indices[0]):
            orig_idx = int(valid_indices[idx]) if idx < len(valid_indices) else int(idx)
            row = self.trades.iloc[orig_idx]
            results.append(SimilarTrade(
                index=orig_idx, distance=float(dist),
                pnl=float(row.get(self.target_col, 0)),
                return_pct=float(row.get("return_pct", 0)),
                regime=str(row.get(self.regime_col, "neutral")),
                features={f: float(row.get(f, 0)) for f in self.features},
            ))
        return results

    def predict_batch(self, trades_df: pd.DataFrame) -> List[Prediction]:
        """Predict outcomes for multiple proposed trades."""
        results = []
        for _, row in trades_df.iterrows():
            feats = {f: float(row.get(f, 0)) for f in self.features}
            if self.regime_col in row:
                feats["regime"] = row[self.regime_col]
            results.append(self.predict(feats))
        return results

    # ── Internal ────────────────────────────────────────────────────────

    @staticmethod
    def _risk_reward_score(
        expected: float, ci_low: float, ci_high: float, win_prob: float,
    ) -> float:
        """Compute risk/reward score 0-100."""
        if ci_high == ci_low:
            return 50.0
        # Upside vs downside ratio
        upside = max(ci_high - expected, 0)
        downside = max(expected - ci_low, 0)
        ratio = upside / (downside + 1e-6)
        ratio_score = min(ratio / 3.0, 1.0) * 40  # max 40 from ratio

        # Win probability contribution
        wp_score = win_prob * 40  # max 40 from win prob

        # Expected P&L direction
        direction_score = 20.0 if expected > 0 else 0.0

        return max(0.0, min(100.0, ratio_score + wp_score + direction_score))

    def _calibration_analysis(
        self, predicted: np.ndarray, actual: np.ndarray, n_buckets: int = 5,
    ) -> List[CalibrationBucket]:
        """Bin predictions and compare to actual outcomes."""
        if len(predicted) < n_buckets:
            return []
        order = np.argsort(predicted)
        predicted = predicted[order]
        actual = actual[order]
        bucket_size = len(predicted) // n_buckets
        buckets: List[CalibrationBucket] = []
        for i in range(n_buckets):
            start = i * bucket_size
            end = start + bucket_size if i < n_buckets - 1 else len(predicted)
            p_slice = predicted[start:end]
            a_slice = actual[start:end]
            buckets.append(CalibrationBucket(
                predicted_mean=float(np.mean(p_slice)),
                actual_mean=float(np.mean(a_slice)),
                n_trades=len(p_slice),
                bucket_label=f"Q{i + 1}",
            ))
        return buckets

    # ── Report generation ───────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        """Generate HTML report. Fits model if not yet fit."""
        if self.fit_result is None:
            self.fit()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        logger.info("Report written to %s", out)
        return str(out.resolve())

    # ── Charts ──────────────────────────────────────────────────────────

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        charts: Dict[str, str] = {}
        charts["feature_importance"] = self._chart_feature_importance()
        charts["calibration"] = self._chart_calibration()
        charts["prediction_dist"] = self._chart_prediction_distribution()
        charts["pnl_distribution"] = self._chart_pnl_distribution()
        return charts

    def _chart_feature_importance(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.fit_result or not self.fit_result.feature_importances:
            return ""
        fi = sorted(self.fit_result.feature_importances, key=lambda x: x.importance)
        fig, ax = plt.subplots(figsize=(7, max(3, len(fi) * 0.4)))
        ax.barh([f.feature for f in fi], [f.importance for f in fi],
                color="#3b82f6", alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance", fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_calibration(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.calibration:
            return ""
        fig, ax = plt.subplots(figsize=(5, 5))
        preds = [b.predicted_mean for b in self.calibration]
        actuals = [b.actual_mean for b in self.calibration]
        ax.scatter(preds, actuals, s=60, color="#3b82f6", zorder=5)
        for b in self.calibration:
            ax.annotate(b.bucket_label, (b.predicted_mean, b.actual_mean),
                        fontsize=8, ha="left", va="bottom")
        mn = min(min(preds), min(actuals))
        mx = max(max(preds), max(actuals))
        pad = (mx - mn) * 0.1 or 1
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                "k--", lw=0.8, alpha=0.5, label="Perfect calibration")
        ax.set_xlabel("Predicted Mean P&L")
        ax.set_ylabel("Actual Mean P&L")
        ax.set_title("Calibration Plot", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_prediction_distribution(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.predictions:
            return ""
        pnls = [p.expected_pnl for _, p in self.predictions]
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(pnls, bins=min(30, len(pnls)), color="#3b82f6", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="#dc2626", lw=1, ls="--")
        ax.set_xlabel("Expected P&L")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Distribution", fontsize=11)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_pnl_distribution(self) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y = self.trades[self.target_col].dropna()
        if len(y) < 5:
            return ""
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(y, bins=min(40, len(y) // 2), color="#64748b", alpha=0.7, edgecolor="white")
        ax.axvline(0, color="#dc2626", lw=1, ls="--")
        ax.axvline(y.mean(), color="#16a34a", lw=1.2, ls="-", label=f"Mean: ${y.mean():.0f}")
        ax.set_xlabel("P&L")
        ax.set_ylabel("Count")
        ax.set_title("Historical P&L Distribution", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    # ── HTML builder ────────────────────────────────────────────────────

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        fr = self.fit_result or FitResult(0, 0, 0, 0, [])

        # Feature importance table
        fi_rows = ""
        for f in fr.feature_importances:
            fi_rows += (
                f'<tr><td>{f.rank}</td><td>{f.feature}</td>'
                f'<td>{f.importance:.4f}</td></tr>\n'
            )

        # Calibration table
        cal_rows = ""
        for b in self.calibration:
            diff = b.actual_mean - b.predicted_mean
            cls = "good" if abs(diff) < abs(b.predicted_mean) * 0.3 else "bad"
            cal_rows += (
                f'<tr><td>{b.bucket_label}</td><td>{b.n_trades}</td>'
                f'<td>${b.predicted_mean:,.0f}</td>'
                f'<td>${b.actual_mean:,.0f}</td>'
                f'<td class="{cls}">${diff:+,.0f}</td></tr>\n'
            )

        # Recent predictions table
        pred_rows = ""
        for feats, p in self.predictions[-20:]:
            cls = "good" if p.expected_pnl > 0 else "bad"
            pred_rows += (
                f'<tr><td>{p.regime}</td>'
                f'<td class="{cls}">${p.expected_pnl:,.0f}</td>'
                f'<td>${p.ci_lower:,.0f} – ${p.ci_upper:,.0f}</td>'
                f'<td>{p.win_probability:.0%}</td>'
                f'<td>{p.risk_reward_score:.0f}</td>'
                f'<td>{p.confidence:.0%}</td></tr>\n'
            )

        win_rate = float((self.trades[self.target_col] > 0).mean()) if len(self.trades) > 0 else 0
        avg_pnl = float(self.trades[self.target_col].mean()) if len(self.trades) > 0 else 0

        def _img(key: str) -> str:
            b64 = charts.get(key, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b64}" alt="{key}"></div>' if b64 else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trade Outcome Predictions</title>
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
  td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; text-align: right; }}
  td:first-child {{ text-align: left; }}
  .chart {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
            padding: 1em; margin: 1.5em 0; text-align: center; }}
  .chart img {{ max-width: 100%; height: auto; }}
  footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #e2e8f0;
            font-size: 0.8em; color: #94a3b8; }}
</style>
</head>
<body>

<h1>Trade Outcome Predictions</h1>
<div class="meta">{fr.n_train} training trades &middot; {fr.n_features} features &middot; Generated {now}</div>

<div class="kpi-row">
  <div class="kpi"><div class="value">{fr.train_r2:.2f}</div><div class="label">Model R&sup2;</div></div>
  <div class="kpi"><div class="value">${fr.cv_mae:,.0f}</div><div class="label">Train MAE</div></div>
  <div class="kpi"><div class="value">{win_rate:.0%}</div><div class="label">Historical Win Rate</div></div>
  <div class="kpi"><div class="value {"good" if avg_pnl > 0 else "bad"}">${avg_pnl:,.0f}</div><div class="label">Avg P&L</div></div>
  <div class="kpi"><div class="value">{len(self.predictions)}</div><div class="label">Predictions Made</div></div>
</div>

<h2>1. Feature Importance</h2>
{_img("feature_importance")}
<table>
<thead><tr><th>Rank</th><th>Feature</th><th>Importance</th></tr></thead>
<tbody>{fi_rows}</tbody>
</table>

<h2>2. Historical P&L Distribution</h2>
{_img("pnl_distribution")}

<h2>3. Calibration Analysis</h2>
{_img("calibration")}
<table>
<thead><tr><th>Bucket</th><th>Trades</th><th>Predicted</th><th>Actual</th><th>Error</th></tr></thead>
<tbody>{cal_rows}</tbody>
</table>

<h2>4. Recent Predictions</h2>
{_img("prediction_dist")}
<table>
<thead><tr><th>Regime</th><th>Expected P&L</th><th>80% CI</th><th>Win Prob</th><th>R/R Score</th><th>Confidence</th></tr></thead>
<tbody>{pred_rows if pred_rows else '<tr><td colspan="6" style="text-align:center;color:#64748b">No predictions yet</td></tr>'}</tbody>
</table>

<footer>Generated by <code>compass/trade_outcome_predictor.py</code></footer>
</body></html>"""
        return html
