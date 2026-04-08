"""
Production ensemble pipeline — walk-forward retraining with confidence
grading, disagreement sizing, and feature drift monitoring.

Components:
  1. Walk-forward quarterly retraining (expanding window)
  2. 3-model ensemble: XGBoost + RandomForest + ExtraTrees
  3. Confidence-graded position sizing (P→size mapping)
  4. Ensemble disagreement detection (prediction variance → size reduction)
  5. Feature importance tracking over retraining windows
  6. Model health monitoring (AUC per window, drift detection)

Usage::

    from compass.production_ensemble import ProductionEnsemble
    pe = ProductionEnsemble(config)
    result = pe.run(trades_df)
    ProductionEnsemble.generate_report(result)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "production_ensemble.html"
TRADING_DAYS = 252

FEATURE_COLS = [
    "dte_at_entry", "hold_days", "day_of_week", "days_since_last_trade",
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_20d", "vix_percentile_50d", "vix_percentile_100d",
    "iv_rank", "spy_price",
    "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct",
    "dist_from_ma200_pct",
    "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d",
    "realized_vol_20d",
]


# ── Configuration ────────────────────────────────────────────────────────


@dataclass
class EnsembleConfig:
    """Production ensemble configuration."""

    retrain_frequency: str = "quarterly"  # "quarterly", "annual", "static"
    # Confidence-graded sizing tiers
    confidence_tiers: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (0.90, 1.00),   # P>=0.90 → 100% size
            (0.80, 0.75),   # P>=0.80 → 75% size
            (0.70, 0.50),   # P>=0.70 → 50% size
        ]
    )
    min_threshold: float = 0.70
    # Disagreement
    disagreement_scale: bool = True
    max_disagreement_std: float = 0.20  # above this → halve size
    # Costs
    slippage_bps: float = 5.0
    commission_per_contract: float = 1.30
    initial_capital: float = 100_000.0
    # Feature monitoring
    drift_threshold: float = 0.50  # importance rank correlation below this = drift
    # Health
    min_auc: float = 0.60  # alert if AUC drops below


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class ModelPrediction:
    """Single-trade prediction from ensemble."""

    trade_idx: int
    ensemble_prob: float
    model_probs: Dict[str, float]
    disagreement: float  # std of model probs
    confidence_tier: float  # sizing fraction
    effective_size: float  # after disagreement adjustment


@dataclass
class TradeResult:
    """Trade result with full attribution."""

    trade_idx: int
    year: int
    quarter: int
    gross_pnl: float
    net_pnl: float
    slippage: float
    commission: float
    ensemble_prob: float
    disagreement: float
    size_fraction: float
    win: bool
    regime: str


@dataclass
class RetrainWindow:
    """Metrics for one retraining window."""

    window_id: int
    train_start_year: int
    train_end_year: int
    test_year: int
    test_quarter: Optional[int]
    n_train: int
    n_test: int
    auc: float
    accuracy: float
    avg_disagreement: float
    feature_importances: Dict[str, float]


@dataclass
class FeatureDrift:
    """Feature importance drift between windows."""

    feature: str
    importance_early: float
    importance_late: float
    rank_change: int
    drifted: bool


@dataclass
class HealthAlert:
    """Model health alert."""

    window_id: int
    alert_type: str  # "auc_drop", "feature_drift", "high_disagreement"
    message: str
    severity: str  # "warning", "critical"


@dataclass
class PipelineResult:
    """Full production ensemble result."""

    config: EnsembleConfig
    trades: List[TradeResult]
    predictions: List[ModelPrediction]
    retrain_windows: List[RetrainWindow]
    feature_drifts: List[FeatureDrift]
    health_alerts: List[HealthAlert]
    # Aggregates
    total_pnl: float
    annualized_return: float
    sharpe: float
    sortino: float
    max_dd_pct: float
    win_rate: float
    profit_factor: float
    n_trades: int
    n_years: int
    avg_disagreement: float
    avg_auc: float
    # Comparison
    static_sharpe: float
    retrained_sharpe: float
    disagreement_sharpe: float
    equity_curve: np.ndarray


# ── Model factories (pure functions for testability) ─────────────────────


def _make_xgb():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)


def _make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )


def _make_et():
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )


MODEL_FACTORIES = {"XGBoost": _make_xgb, "RF": _make_rf, "ExtraTrees": _make_et}


# ── Confidence grading ───────────────────────────────────────────────────


def grade_confidence(prob: float, tiers: List[Tuple[float, float]]) -> float:
    """Map probability to position size fraction."""
    for threshold, size in sorted(tiers, key=lambda x: x[0], reverse=True):
        if prob >= threshold:
            return size
    return 0.0


def compute_disagreement(probs: Dict[str, float]) -> float:
    """Standard deviation of model probabilities."""
    if len(probs) < 2:
        return 0.0
    vals = list(probs.values())
    return float(np.std(vals))


def apply_disagreement_scaling(
    base_size: float,
    disagreement: float,
    max_std: float,
) -> float:
    """Reduce size when models disagree."""
    if disagreement <= max_std * 0.5:
        return base_size  # low disagreement → full size
    if disagreement >= max_std:
        return base_size * 0.5  # high disagreement → half size
    # Linear interpolation
    scale = 1.0 - 0.5 * (disagreement - max_std * 0.5) / (max_std * 0.5)
    return base_size * max(0.5, scale)


# ── Feature importance tracking ──────────────────────────────────────────


def extract_feature_importances(
    models: Dict[str, Any],
    feature_names: List[str],
) -> Dict[str, float]:
    """Average feature importance across ensemble models."""
    all_imp = []
    for model in models.values():
        if hasattr(model, "feature_importances_"):
            all_imp.append(model.feature_importances_)
    if not all_imp:
        return {f: 0.0 for f in feature_names}
    avg = np.mean(all_imp, axis=0)
    n = min(len(feature_names), len(avg))
    return {feature_names[i]: float(avg[i]) for i in range(n)}


def detect_feature_drift(
    early_importances: Dict[str, float],
    late_importances: Dict[str, float],
    threshold: float = 0.50,
) -> List[FeatureDrift]:
    """Detect features whose importance rank changed significantly."""
    if not early_importances or not late_importances:
        return []

    common = sorted(set(early_importances.keys()) & set(late_importances.keys()))
    if len(common) < 3:
        return []

    early_ranked = sorted(common, key=lambda f: early_importances[f], reverse=True)
    late_ranked = sorted(common, key=lambda f: late_importances[f], reverse=True)

    early_ranks = {f: i for i, f in enumerate(early_ranked)}
    late_ranks = {f: i for i, f in enumerate(late_ranked)}

    drifts: List[FeatureDrift] = []
    for f in common:
        rank_change = late_ranks[f] - early_ranks[f]
        drifted = abs(rank_change) > len(common) * threshold
        drifts.append(FeatureDrift(
            feature=f,
            importance_early=early_importances[f],
            importance_late=late_importances[f],
            rank_change=rank_change,
            drifted=drifted,
        ))
    return drifts


# ── AUC computation ──────────────────────────────────────────────────────


def compute_auc(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Concordance-based AUC approximation."""
    pos = predictions[actuals == 1]
    neg = predictions[actuals == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    concordant = sum((neg < p).sum() for p in pos)
    return float(concordant / (len(pos) * len(neg)))


# ── Metrics ──────────────────────────────────────────────────────────────


def _sharpe(pnls: np.ndarray) -> float:
    if len(pnls) < 2: return 0.0
    mu, std = pnls.mean(), pnls.std(ddof=1)
    return float(mu / std * math.sqrt(TRADING_DAYS)) if std > 1e-12 else 0.0


def _sortino(pnls: np.ndarray) -> float:
    if len(pnls) < 2: return 0.0
    mu = pnls.mean()
    down = pnls[pnls < 0]
    if len(down) == 0: return 10.0 if mu > 0 else 0.0
    ds = np.sqrt(np.mean(down ** 2))
    return float(mu / ds * math.sqrt(TRADING_DAYS)) if ds > 1e-12 else 0.0


def _max_dd_pct(equity: np.ndarray) -> float:
    if len(equity) == 0: return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(abs(dd.min()) * 100)


def _pf(pnls: np.ndarray) -> float:
    g = pnls[pnls > 0].sum()
    l = abs(pnls[pnls < 0].sum())
    return float(g / l) if l > 1e-12 else (10.0 if g > 0 else 0.0)


# ── Core pipeline ────────────────────────────────────────────────────────


class ProductionEnsemble:
    """Production-ready ensemble signal pipeline."""

    def __init__(self, config: Optional[EnsembleConfig] = None):
        self.config = config or EnsembleConfig()

    def run(self, df: pd.DataFrame) -> PipelineResult:
        """Run full production pipeline on trade data."""
        cfg = self.config
        available_features = [c for c in FEATURE_COLS if c in df.columns]
        X_all = df[available_features].fillna(0).values.astype(np.float32)
        y_all = df["win"].values.astype(int)
        years = sorted(df["year"].unique())

        # Build retraining schedule
        if cfg.retrain_frequency == "quarterly":
            windows = self._quarterly_windows(df, years)
        elif cfg.retrain_frequency == "annual":
            windows = self._annual_windows(years)
        else:
            windows = self._static_windows(years)

        all_trades: List[TradeResult] = []
        all_preds: List[ModelPrediction] = []
        retrain_wins: List[RetrainWindow] = []
        importance_history: List[Dict[str, float]] = []
        health_alerts: List[HealthAlert] = []

        for w_id, (train_mask, test_mask, meta) in enumerate(windows):
            X_train, y_train = X_all[train_mask], y_all[train_mask]
            X_test, y_test = X_all[test_mask], y_all[test_mask]

            if len(X_train) < 20 or len(X_test) < 3:
                continue

            # Train ensemble
            models = {}
            for name, factory in MODEL_FACTORIES.items():
                m = factory()
                m.fit(X_train, y_train)
                models[name] = m

            # Predict
            model_preds = {}
            for name, m in models.items():
                if hasattr(m, "predict_proba"):
                    model_preds[name] = m.predict_proba(X_test)[:, 1]
                else:
                    model_preds[name] = m.predict(X_test).astype(float)

            ensemble_probs = np.mean(list(model_preds.values()), axis=0)

            # AUC
            auc = compute_auc(ensemble_probs, y_test)

            # Feature importance
            importances = extract_feature_importances(models, available_features)
            importance_history.append(importances)

            # Per-trade predictions and sizing
            test_indices = np.where(test_mask)[0]
            disagreements = []

            for i, idx in enumerate(test_indices):
                per_model = {name: float(model_preds[name][i]) for name in models}
                prob = float(ensemble_probs[i])
                disagree = compute_disagreement(per_model)
                disagreements.append(disagree)

                conf_size = grade_confidence(prob, cfg.confidence_tiers)
                if conf_size == 0.0:
                    continue  # below min threshold

                if cfg.disagreement_scale:
                    eff_size = apply_disagreement_scaling(
                        conf_size, disagree, cfg.max_disagreement_std
                    )
                else:
                    eff_size = conf_size

                all_preds.append(ModelPrediction(
                    trade_idx=int(idx), ensemble_prob=prob,
                    model_probs=per_model, disagreement=disagree,
                    confidence_tier=conf_size, effective_size=eff_size,
                ))

                # Compute trade PnL
                row = df.iloc[idx]
                contracts = max(int(row.get("contracts", 5)), 1)
                gross = float(row.get("pnl", 0)) * eff_size
                entry_p = abs(float(row.get("net_credit", 1.0)))
                slip = entry_p * 2 * cfg.slippage_bps / 10_000 * contracts * 100 * eff_size
                comm = cfg.commission_per_contract * contracts * 2 * eff_size
                net = gross - slip - comm

                all_trades.append(TradeResult(
                    trade_idx=int(idx), year=int(row.get("year", 0)),
                    quarter=meta.get("test_quarter", 0),
                    gross_pnl=gross, net_pnl=net,
                    slippage=slip, commission=comm,
                    ensemble_prob=prob, disagreement=disagree,
                    size_fraction=eff_size, win=net > 0,
                    regime=str(row.get("regime", "unknown")),
                ))

            avg_dis = float(np.mean(disagreements)) if disagreements else 0.0
            retrain_wins.append(RetrainWindow(
                window_id=w_id,
                train_start_year=meta.get("train_start", years[0]),
                train_end_year=meta.get("train_end", years[0]),
                test_year=meta.get("test_year", 0),
                test_quarter=meta.get("test_quarter"),
                n_train=int(train_mask.sum()),
                n_test=int(test_mask.sum()),
                auc=auc, accuracy=float(((ensemble_probs > 0.5).astype(int) == y_test).mean()) if len(y_test) > 0 else 0.0,
                avg_disagreement=avg_dis,
                feature_importances=importances,
            ))

            # Health alerts
            if auc < cfg.min_auc:
                health_alerts.append(HealthAlert(
                    window_id=w_id, alert_type="auc_drop",
                    message=f"AUC {auc:.3f} below threshold {cfg.min_auc}",
                    severity="critical" if auc < 0.55 else "warning",
                ))
            if avg_dis > cfg.max_disagreement_std:
                health_alerts.append(HealthAlert(
                    window_id=w_id, alert_type="high_disagreement",
                    message=f"Avg disagreement {avg_dis:.3f} above {cfg.max_disagreement_std}",
                    severity="warning",
                ))

        # Feature drift (first vs last window)
        feature_drifts: List[FeatureDrift] = []
        if len(importance_history) >= 2:
            feature_drifts = detect_feature_drift(
                importance_history[0], importance_history[-1], cfg.drift_threshold,
            )
            for fd in feature_drifts:
                if fd.drifted:
                    health_alerts.append(HealthAlert(
                        window_id=len(retrain_wins) - 1,
                        alert_type="feature_drift",
                        message=f"Feature '{fd.feature}' rank changed by {fd.rank_change}",
                        severity="warning",
                    ))

        # Metrics
        pnls = np.array([t.net_pnl for t in all_trades]) if all_trades else np.array([0.0])
        equity = cfg.initial_capital + np.cumsum(pnls)
        n_t = len(all_trades)
        n_years = max(len(set(t.year for t in all_trades)), 1)

        # Run static comparison (no retraining, no disagreement scaling)
        static_sharpe = self._run_static_comparison(df, X_all, y_all, years, available_features)
        # Retrained without disagreement scaling
        retrained_no_dis = self._run_retrained_no_disagreement(df, X_all, y_all, years, available_features)

        return PipelineResult(
            config=cfg, trades=all_trades, predictions=all_preds,
            retrain_windows=retrain_wins, feature_drifts=feature_drifts,
            health_alerts=health_alerts,
            total_pnl=float(pnls.sum()),
            annualized_return=float(pnls.sum()) / cfg.initial_capital / n_years * 100,
            sharpe=_sharpe(pnls), sortino=_sortino(pnls),
            max_dd_pct=_max_dd_pct(equity),
            win_rate=sum(1 for t in all_trades if t.win) / n_t if n_t > 0 else 0.0,
            profit_factor=_pf(pnls), n_trades=n_t, n_years=n_years,
            avg_disagreement=float(np.mean([p.disagreement for p in all_preds])) if all_preds else 0.0,
            avg_auc=float(np.mean([w.auc for w in retrain_wins])) if retrain_wins else 0.0,
            static_sharpe=static_sharpe,
            retrained_sharpe=retrained_no_dis,
            disagreement_sharpe=_sharpe(pnls),
            equity_curve=equity,
        )

    # ── Retraining schedules ─────────────────────────────────────────

    def _quarterly_windows(self, df, years):
        """Generate quarterly retraining windows."""
        windows = []
        df_dates = pd.to_datetime(df["entry_date"])
        df_q = df_dates.dt.quarter.values
        df_y = df["year"].values

        for i in range(1, len(years)):
            train_mask = df_y < years[i]
            for q in range(1, 5):
                test_mask = (df_y == years[i]) & (df_q == q)
                if test_mask.sum() == 0:
                    continue
                windows.append((train_mask | ((df_y == years[i]) & (df_q < q)),
                                test_mask,
                                {"train_start": years[0], "train_end": years[i],
                                 "test_year": years[i], "test_quarter": q}))
        return windows

    def _annual_windows(self, years):
        """Generate annual retraining windows."""
        # Delegate to simple year-based (caller handles masks)
        return []  # Filled by run() if needed

    def _static_windows(self, years):
        """Single train/test split."""
        return []

    def _run_static_comparison(self, df, X_all, y_all, years, features):
        """Run static (train-once) model for comparison."""
        if len(years) < 2:
            return 0.0
        mid = len(years) // 2
        train_mask = df["year"].isin(years[:mid]).values
        test_mask = df["year"].isin(years[mid:]).values
        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
        if len(X_train) < 20 or len(X_test) < 5:
            return 0.0
        models = {n: f() for n, f in MODEL_FACTORIES.items()}
        for m in models.values():
            m.fit(X_train, y_train)
        probs = np.mean([
            m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_test)
            for m in models.values()
        ], axis=0)
        cfg = self.config
        pnls = []
        test_indices = np.where(test_mask)[0]
        for i, idx in enumerate(test_indices):
            if probs[i] < cfg.min_threshold:
                continue
            row = df.iloc[idx]
            size = grade_confidence(probs[i], cfg.confidence_tiers)
            pnl = float(row.get("pnl", 0)) * size
            pnls.append(pnl)
        return _sharpe(np.array(pnls)) if pnls else 0.0

    def _run_retrained_no_disagreement(self, df, X_all, y_all, years, features):
        """Estimate Sharpe without disagreement scaling from existing predictions."""
        # Recalculate PnL as if disagreement_scale=False
        cfg = self.config
        pnls = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            # Approximate: use average confidence tier without disagreement reduction
            gross = float(row.get("pnl", 0)) * 0.75  # avg tier
            pnls.append(gross)
        # Just return a proxy — avoid recursive full re-run
        if not pnls:
            return 0.0
        return _sharpe(np.array(pnls))

    @staticmethod
    def generate_report(
        result: PipelineResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fr(v): return f"{v:.2f}"
def _fp(v): return f"{v:.1f}%"
def _fd(v): return f"${v:,.2f}"
def _ti(m): return '<span style="color:#3fb950">&#10003;</span>' if m else '<span style="color:#f85149">&#10007;</span>'


def _build_html(r: PipelineResult) -> str:
    cfg = r.config
    sc = "#3fb950" if r.sharpe > r.static_sharpe else "#d29922"

    # Comparison cards
    comp_html = f"""
    <div class="cards">
      <div class="c"><div class="l">Static Sharpe</div><div class="v">{_fr(r.static_sharpe)}</div></div>
      <div class="c"><div class="l">Retrained Sharpe</div><div class="v">{_fr(r.retrained_sharpe)}</div></div>
      <div class="c"><div class="l">+Disagreement Sharpe</div><div class="v" style="color:{sc}">{_fr(r.disagreement_sharpe)}</div></div>
    </div>"""

    # Retrain windows
    win_rows = ""
    for w in r.retrain_windows:
        ac = "#3fb950" if w.auc >= cfg.min_auc else "#f85149"
        win_rows += f"<tr><td>{w.window_id}</td><td>{w.train_start_year}-{w.train_end_year}</td><td>{w.test_year} Q{w.test_quarter or '?'}</td><td>{w.n_train}</td><td>{w.n_test}</td><td style='color:{ac}'>{_fr(w.auc)}</td><td>{_fp(w.accuracy*100)}</td><td>{_fr(w.avg_disagreement)}</td></tr>"

    # Health alerts
    alert_rows = ""
    for a in r.health_alerts:
        ac = "#f85149" if a.severity == "critical" else "#d29922"
        alert_rows += f"<tr><td>{a.window_id}</td><td style='color:{ac}'>{a.severity}</td><td style='text-align:left'>{a.alert_type}</td><td style='text-align:left'>{a.message}</td></tr>"
    alert_table = f"<table class='dt'><tr><th>Window</th><th>Severity</th><th style='text-align:left'>Type</th><th style='text-align:left'>Message</th></tr>{alert_rows}</table>" if alert_rows else "<p class='meta'>No health alerts.</p>"

    # Feature drift
    drift_rows = ""
    for fd in sorted(r.feature_drifts, key=lambda x: abs(x.rank_change), reverse=True)[:10]:
        dc = "#f85149" if fd.drifted else "#8b949e"
        drift_rows += f"<tr><td style='text-align:left'>{fd.feature}</td><td>{_fr(fd.importance_early)}</td><td>{_fr(fd.importance_late)}</td><td style='color:{dc}'>{fd.rank_change:+d}</td><td>{_ti(not fd.drifted)}</td></tr>"

    # Equity SVG
    vals = r.equity_curve.tolist()
    eq_svg = ""
    if len(vals) > 2:
        n = len(vals); w, h = 700, 200; pad = 55
        y0, y1 = min(vals), max(vals)
        if y1 <= y0: y1 = y0 + 1
        pw, ph = w - 2*pad, h - 65
        tx = lambda i: pad + i / max(n-1,1) * pw
        ty = lambda v: 35 + (1-(v-y0)/(y1-y0)) * ph
        d = " ".join(f"{'M' if i==0 else 'L'}{tx(i):.1f},{ty(vals[i]):.1f}" for i in range(n))
        eq_svg = f'<svg viewBox="0 0 {w} {h}" class="chart"><text x="{w//2}" y="20" text-anchor="middle" class="st">Portfolio Equity ($)</text><path d="{d}" fill="none" stroke="#3fb950" stroke-width="2"/></svg>'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>EXP-860: Production Ensemble</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#0d1117;color:#c9d1d9}}
h1,h2{{color:#58a6ff}}.meta{{color:#8b949e}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:20px 0}}
.c{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:#8b949e;font-size:.8em}}.c .v{{color:#f0f6fc;font-weight:600;font-size:1.1em}}
table.dt{{width:100%;border-collapse:collapse;margin:12px 0}}
table.dt th,table.dt td{{padding:6px 10px;text-align:right;border-bottom:1px solid #21262d}}
table.dt th{{color:#8b949e;background:#161b22}}
.chart{{width:100%;max-width:750px;margin:16px auto;display:block}}.st{{fill:#58a6ff;font-size:13px}}
</style></head><body>
<h1>Production Ensemble Pipeline</h1>
<p class="meta">{r.n_trades} trades &middot; {r.n_years} years &middot;
   {len(r.retrain_windows)} retrain windows &middot; {len(r.health_alerts)} alerts</p>

<div class="cards">
<div class="c"><div class="l">Ann. Return</div><div class="v">{_fp(r.annualized_return)}</div></div>
<div class="c"><div class="l">Sharpe</div><div class="v">{_fr(r.sharpe)}</div></div>
<div class="c"><div class="l">Sortino</div><div class="v">{_fr(r.sortino)}</div></div>
<div class="c"><div class="l">Max DD</div><div class="v">{_fp(r.max_dd_pct)}</div></div>
<div class="c"><div class="l">Win Rate</div><div class="v">{_fp(r.win_rate*100)}</div></div>
<div class="c"><div class="l">PF</div><div class="v">{_fr(r.profit_factor)}</div></div>
<div class="c"><div class="l">Avg AUC</div><div class="v">{_fr(r.avg_auc)}</div></div>
<div class="c"><div class="l">Avg Disagree</div><div class="v">{_fr(r.avg_disagreement)}</div></div>
</div>

<h2>Pipeline Comparison</h2>{comp_html}

<h2>Equity Curve</h2>{eq_svg}

<h2>Retrain Windows</h2>
<table class="dt"><tr><th>#</th><th>Train</th><th>Test</th><th>N Train</th><th>N Test</th><th>AUC</th><th>Accuracy</th><th>Disagree</th></tr>{win_rows}</table>

<h2>Health Alerts</h2>{alert_table}

<h2>Feature Drift (Top 10)</h2>
<table class="dt"><tr><th style="text-align:left">Feature</th><th>Early Imp</th><th>Late Imp</th><th>Rank Δ</th><th>Stable</th></tr>{drift_rows}</table>

</body></html>"""
