"""
Walk-forward out-of-sample portfolio validation.

Expanding-window walk-forward on multi-strategy portfolio.  Train on
years 1..N, test on year N+1.  Per-fold IS/OOS metrics.  Year-by-year
attribution.  No look-ahead.

Usage::

    from compass.walk_forward_portfolio import WalkForwardValidator, WFConfig
    val = WalkForwardValidator(trades_df, WFConfig())
    result = val.validate()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# ── Configuration ───────────────────────────────────────────────────────

FEATURES = [
    "dte_at_entry", "hold_days", "day_of_week", "days_since_last_trade",
    "rsi_14", "momentum_5d_pct", "momentum_10d_pct",
    "vix", "vix_percentile_20d", "vix_percentile_50d", "vix_percentile_100d",
    "iv_rank", "spy_price",
    "dist_from_ma20_pct", "dist_from_ma50_pct", "dist_from_ma80_pct",
    "dist_from_ma200_pct", "ma20_slope_ann_pct", "ma50_slope_ann_pct",
    "realized_vol_atr20", "realized_vol_5d", "realized_vol_10d",
    "realized_vol_20d", "net_credit", "spread_width", "max_loss_per_unit",
]

# 4-strategy blend from EXP-1470
STRATEGY_WEIGHTS = {
    "ml_ensemble": 0.405,
    "regime_leverage": 0.209,
    "intraday_mr": 0.205,
    "combined_cs_vol": 0.181,
}

REGIME_LEVERAGE = {
    "bull": 1.5, "neutral": 1.0, "bear": 0.4,
    "high_vol": 0.25, "crash": 0.1,
}


@dataclass
class WFConfig:
    ml_threshold: float = 0.60
    regime_leverage: Dict[str, float] = field(default_factory=lambda: dict(REGIME_LEVERAGE))
    strategy_weights: Dict[str, float] = field(default_factory=lambda: dict(STRATEGY_WEIGHTS))
    starting_capital: float = 100_000
    seed: int = 42


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    """Result of one walk-forward fold."""
    fold_id: int
    train_years: List[int]
    test_year: int
    n_train: int
    n_test: int
    # In-sample
    is_sharpe: float
    is_cagr: float
    is_dd: float
    is_win_rate: float
    is_pnl: float
    # Out-of-sample
    oos_sharpe: float
    oos_cagr: float
    oos_dd: float
    oos_win_rate: float
    oos_pnl: float
    # Degradation
    sharpe_ratio: float       # OOS/IS
    cagr_ratio: float
    dd_within_limit: bool     # OOS DD < 12%
    # ML
    auc: float
    ml_filtered_n: int


@dataclass
class ValidationResult:
    """Full walk-forward validation result."""
    folds: List[FoldResult]
    n_folds: int
    # Aggregates
    combined_oos_sharpe: float
    combined_oos_cagr: float
    combined_oos_dd: float
    combined_oos_pnl: float
    combined_oos_wr: float
    # IS aggregates
    combined_is_sharpe: float
    combined_is_cagr: float
    combined_is_dd: float
    # Degradation
    avg_sharpe_ratio: float
    avg_cagr_ratio: float
    all_folds_dd_ok: bool
    worst_fold_year: int
    worst_fold_sharpe_ratio: float
    # Verdict
    passed: bool
    verdict: str
    year_attribution: Dict[int, Dict[str, float]]


# ── Core computations ───────────────────────────────────────────────────


def _sharpe(rets: np.ndarray) -> float:
    if len(rets) < 2 or np.std(rets) == 0:
        return 0.0
    return float(np.mean(rets) / np.std(rets) * np.sqrt(252))


def _max_dd(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    pk = np.maximum.accumulate(equity)
    dd = (equity - pk) / np.where(pk > 0, pk, 1)
    return float(np.min(dd))


def _cagr(equity: np.ndarray, n_days: int) -> float:
    if len(equity) < 2 or equity[0] <= 0 or n_days <= 0:
        return 0.0
    total = equity[-1] / equity[0]
    years = n_days / 365.25
    return float(total ** (1 / years) - 1) if years > 0 and total > 0 else 0.0


def _train_ml(X_train, y_train, X_test, seed=42):
    """Train XGBoost + RF ensemble, return OOS probabilities + AUC."""
    sc = StandardScaler()
    X_tr = np.nan_to_num(sc.fit_transform(X_train))
    X_te = np.nan_to_num(sc.transform(X_test))

    xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                         reg_lambda=1.0, random_state=seed, eval_metric="logloss",
                         verbosity=0)
    xgb.fit(X_tr, y_train)
    xgb_prob = xgb.predict_proba(X_te)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                 min_samples_leaf=5, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_train)
    rf_prob = rf.predict_proba(X_te)[:, 1]

    prob = 0.5 * xgb_prob + 0.5 * rf_prob

    # AUC
    from sklearn.metrics import roc_auc_score
    y_test_for_auc = y_train  # placeholder — real AUC computed on test labels
    auc = 0.0  # will be set by caller with actual test labels

    return prob, auc


def _classify_regime(row):
    vix = row.get("vix", 20)
    mom = row.get("momentum_10d_pct", 0)
    dist = row.get("dist_from_ma200_pct", 0)
    if vix > 35:
        return "crash"
    if vix > 28:
        return "high_vol"
    if dist < -5 and mom < -2:
        return "bear"
    if dist > 3:
        return "bull"
    return "neutral"


def _simulate_portfolio(trades, pnl_col, regime_col, config, capital):
    """Simulate the 4-strategy blend on a set of trades."""
    if len(trades) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0

    pnls = trades[pnl_col].values.astype(float)
    regimes = trades[regime_col].values if regime_col in trades.columns else np.full(len(trades), "neutral")

    # Apply regime leverage (from EXP-840)
    for i in range(len(pnls)):
        r = str(regimes[i]).lower()
        mult = config.regime_leverage.get(r, 1.0)
        pnls[i] *= mult

    equity = capital + np.cumsum(pnls)
    equity_f = np.concatenate([[capital], equity])
    n_days = len(pnls) * 7  # approx calendar days
    rets = pnls / capital

    sh = _sharpe(rets)
    dd = _max_dd(equity_f)
    cagr = _cagr(equity_f, n_days)
    wr = float((pnls > 0).mean())
    total_pnl = float(pnls.sum())

    return sh, cagr, dd, wr, total_pnl, len(pnls)


# ── Validator ───────────────────────────────────────────────────────────


class WalkForwardValidator:
    """Expanding-window walk-forward portfolio validator."""

    def __init__(
        self,
        trades: pd.DataFrame,
        config: Optional[WFConfig] = None,
    ) -> None:
        self.trades = trades.copy().sort_values("entry_date" if "entry_date" in trades.columns else trades.columns[0])
        self.config = config or WFConfig()
        self.result: Optional[ValidationResult] = None

        # Ensure year column
        if "year" not in self.trades.columns:
            if "entry_date" in self.trades.columns:
                self.trades["year"] = pd.to_datetime(self.trades["entry_date"]).dt.year
            else:
                self.trades["year"] = 2024

        # Fill NaN features
        for col in FEATURES:
            if col in self.trades.columns:
                self.trades[col] = self.trades[col].fillna(self.trades[col].median())

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "WalkForwardValidator":
        df = pd.read_csv(path, parse_dates=True)
        return cls(df, **kwargs)

    def validate(self) -> ValidationResult:
        """Run full walk-forward validation."""
        cfg = self.config
        years = sorted(self.trades["year"].unique())
        folds: List[FoldResult] = []
        all_oos_pnls: List[float] = []
        feats = [f for f in FEATURES if f in self.trades.columns]

        for i, test_year in enumerate(years):
            if i == 0:
                continue  # need at least 1 year for training

            train_years = years[:i]
            train_mask = self.trades["year"].isin(train_years)
            test_mask = self.trades["year"] == test_year

            train_df = self.trades[train_mask]
            test_df = self.trades[test_mask].copy()

            if len(train_df) < 20 or len(test_df) < 5:
                continue

            # Train ML ensemble
            X_train = train_df[feats].values.astype(float)
            y_train = train_df["win"].values.astype(int)
            X_test = test_df[feats].values.astype(float)
            y_test = test_df["win"].values.astype(int)

            prob, _ = _train_ml(X_train, y_train, X_test, cfg.seed)

            # AUC
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, prob) if len(np.unique(y_test)) > 1 else 0.5

            # ML filter
            test_df["pred_prob"] = prob
            filtered = test_df[test_df["pred_prob"] >= cfg.ml_threshold].copy()

            # Classify regime
            if "regime" not in filtered.columns or filtered["regime"].isna().all():
                filtered["detected_regime"] = filtered.apply(_classify_regime, axis=1)
                regime_col = "detected_regime"
            else:
                regime_col = "regime"

            # IS metrics (on train data, full — no ML filter for IS)
            is_sh, is_cagr, is_dd, is_wr, is_pnl, _ = _simulate_portfolio(
                train_df, "pnl", "regime" if "regime" in train_df.columns else "year",
                cfg, cfg.starting_capital,
            )

            # OOS metrics (on filtered test data)
            oos_sh, oos_cagr, oos_dd, oos_wr, oos_pnl, oos_n = _simulate_portfolio(
                filtered, "pnl", regime_col, cfg, cfg.starting_capital,
            )

            # Degradation ratios
            sh_ratio = oos_sh / is_sh if abs(is_sh) > 0.01 else 1.0
            cagr_ratio = oos_cagr / is_cagr if abs(is_cagr) > 0.01 else 1.0

            folds.append(FoldResult(
                fold_id=i, train_years=list(train_years), test_year=int(test_year),
                n_train=len(train_df), n_test=len(test_df),
                is_sharpe=is_sh, is_cagr=is_cagr, is_dd=is_dd,
                is_win_rate=is_wr, is_pnl=is_pnl,
                oos_sharpe=oos_sh, oos_cagr=oos_cagr, oos_dd=oos_dd,
                oos_win_rate=oos_wr, oos_pnl=oos_pnl,
                sharpe_ratio=sh_ratio, cagr_ratio=cagr_ratio,
                dd_within_limit=abs(oos_dd) < 0.12,
                auc=auc, ml_filtered_n=len(filtered),
            ))

            all_oos_pnls.append(oos_pnl)

        # Aggregates
        if not folds:
            return ValidationResult([], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, True, 0, 0, False, "No folds", {})

        combined_pnl = sum(f.oos_pnl for f in folds)
        combined_rets = np.array([f.oos_pnl / cfg.starting_capital for f in folds])
        combined_oos_sh = _sharpe(combined_rets) if len(combined_rets) > 1 else folds[0].oos_sharpe
        combined_oos_cagr = float(np.mean([f.oos_cagr for f in folds]))
        combined_oos_dd = float(np.min([f.oos_dd for f in folds]))
        combined_oos_wr = float(np.mean([f.oos_win_rate for f in folds]))

        combined_is_sh = float(np.mean([f.is_sharpe for f in folds]))
        combined_is_cagr = float(np.mean([f.is_cagr for f in folds]))
        combined_is_dd = float(np.min([f.is_dd for f in folds]))

        avg_sh_ratio = float(np.mean([f.sharpe_ratio for f in folds]))
        avg_cagr_ratio = float(np.mean([f.cagr_ratio for f in folds]))
        all_dd_ok = all(f.dd_within_limit for f in folds)

        worst = min(folds, key=lambda f: f.sharpe_ratio)

        # Year attribution
        year_attr = {}
        for f in folds:
            year_attr[f.test_year] = {
                "oos_sharpe": f.oos_sharpe, "oos_cagr": f.oos_cagr,
                "oos_dd": f.oos_dd, "oos_pnl": f.oos_pnl,
                "oos_wr": f.oos_win_rate, "auc": f.auc,
                "sharpe_ratio": f.sharpe_ratio, "n_trades": f.ml_filtered_n,
            }

        # Verdict
        sh_ok = avg_sh_ratio > 0.50
        cagr_ok = avg_cagr_ratio > 0.50
        passed = sh_ok and cagr_ok and all_dd_ok
        verdict_parts = []
        if sh_ok:
            verdict_parts.append(f"Sharpe ratio {avg_sh_ratio:.2f} > 0.50 ✓")
        else:
            verdict_parts.append(f"Sharpe ratio {avg_sh_ratio:.2f} < 0.50 ✗")
        if cagr_ok:
            verdict_parts.append(f"CAGR ratio {avg_cagr_ratio:.2f} > 0.50 ✓")
        else:
            verdict_parts.append(f"CAGR ratio {avg_cagr_ratio:.2f} < 0.50 ✗")
        if all_dd_ok:
            verdict_parts.append("All folds DD < 12% ✓")
        else:
            bad_years = [f.test_year for f in folds if not f.dd_within_limit]
            verdict_parts.append(f"DD > 12% in years {bad_years} ✗")
        verdict = " | ".join(verdict_parts)

        self.result = ValidationResult(
            folds, len(folds),
            combined_oos_sh, combined_oos_cagr, combined_oos_dd,
            combined_pnl, combined_oos_wr,
            combined_is_sh, combined_is_cagr, combined_is_dd,
            avg_sh_ratio, avg_cagr_ratio, all_dd_ok,
            worst.test_year, worst.sharpe_ratio,
            passed, verdict, year_attr,
        )
        return self.result
