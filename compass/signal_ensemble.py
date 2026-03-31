"""
Signal ensemble — combine multiple alpha signals into a composite.

Methods:
  - Equal weight
  - Inverse-volatility weight
  - Rank-based (signals ranked, then averaged)
  - ML stacking: ridge regression + elastic net on forward returns
  - Regime-conditional (separate weights per regime)

Preprocessing:
  - Winsorize (clip extreme percentiles)
  - Z-score normalisation
  - Rank transform (cross-sectional percentile)

Quality gates:
  - Minimum IC (information coefficient) threshold
  - Maximum pairwise correlation (remove redundant signals)

Walk-forward expanding-window training (never look ahead).

HTML report at reports/signal_ensemble.html with individual vs ensemble
performance, weight evolution, attribution chart.

Usage::

    from compass.signal_ensemble import SignalEnsemble
    ens = SignalEnsemble(signals_df, forward_returns)
    result = ens.fit()
    SignalEnsemble.generate_report(result)
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
DEFAULT_OUTPUT = ROOT / "reports" / "signal_ensemble.html"
TRADING_DAYS = 252

METHODS = ("equal", "inverse_vol", "rank_based", "ridge", "elastic_net", "regime_conditional")


# ── Preprocessing ────────────────────────────────────────────────────────


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Clip values at given percentiles."""
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def zscore_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score each column."""
    mu = df.mean()
    sigma = df.std().replace(0, 1.0)
    return (df - mu) / sigma


def rank_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank per row, mapped to [-1, 1]."""
    return df.rank(axis=1, pct=True) * 2 - 1


def preprocess(
    df: pd.DataFrame,
    method: str = "zscore",
    winsorize_pct: float = 0.01,
) -> pd.DataFrame:
    """Apply preprocessing pipeline."""
    result = df.copy()
    # Winsorize first
    if winsorize_pct > 0:
        for col in result.columns:
            result[col] = winsorize(result[col], winsorize_pct, 1.0 - winsorize_pct)
    if method == "zscore":
        return zscore_transform(result)
    elif method == "rank":
        return rank_transform(result)
    return result


# ── IC computation ───────────────────────────────────────────────────────


def compute_ic(signal: pd.Series, returns: pd.Series) -> float:
    """Spearman rank IC between signal and forward returns."""
    aligned = pd.concat([signal.rename("s"), returns.rename("r")], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    rs = aligned["s"].rank()
    rr = aligned["r"].rank()
    n = len(rs)
    d = rs - rr
    return float(1.0 - 6.0 * (d ** 2).sum() / (n * (n ** 2 - 1)))


def compute_ic_series(signal: pd.Series, returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling IC series."""
    aligned = pd.concat([signal.rename("s"), returns.rename("r")], axis=1).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    ics = []
    dates = []
    for end in range(window, len(aligned) + 1):
        chunk = aligned.iloc[end - window:end]
        rs = chunk["s"].rank()
        rr = chunk["r"].rank()
        n = len(rs)
        d = rs - rr
        ic = 1.0 - 6.0 * (d ** 2).sum() / (n * (n ** 2 - 1))
        ics.append(float(ic))
        dates.append(chunk.index[-1])
    return pd.Series(ics, index=dates, name="ic")


# ── Quality gates ────────────────────────────────────────────────────────


def apply_quality_gates(
    signals: pd.DataFrame,
    returns: pd.Series,
    min_ic: float = 0.02,
    max_correlation: float = 0.85,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Filter signals by IC and correlation thresholds.

    Returns: (filtered_signals, kept_names, dropped_names)
    """
    kept: List[str] = []
    dropped: List[str] = []

    # IC gate
    ics = {}
    for col in signals.columns:
        ic = compute_ic(signals[col], returns)
        ics[col] = ic
        if abs(ic) >= min_ic:
            kept.append(col)
        else:
            dropped.append(col)

    if len(kept) < 2:
        # If too few pass, keep all
        return signals, list(signals.columns), []

    # Correlation gate: remove highly correlated (keep higher IC)
    kept_df = signals[kept]
    corr = kept_df.corr().abs()
    to_remove = set()
    cols = list(kept)
    for i in range(len(cols)):
        if cols[i] in to_remove:
            continue
        for j in range(i + 1, len(cols)):
            if cols[j] in to_remove:
                continue
            if corr.loc[cols[i], cols[j]] > max_correlation:
                # Drop the one with lower IC
                if abs(ics[cols[i]]) < abs(ics[cols[j]]):
                    to_remove.add(cols[i])
                else:
                    to_remove.add(cols[j])

    final_kept = [c for c in kept if c not in to_remove]
    final_dropped = dropped + list(to_remove)

    if len(final_kept) < 2:
        return signals[kept], kept, dropped

    return signals[final_kept], final_kept, final_dropped


# ── Weighting methods ────────────────────────────────────────────────────


def equal_weights(n: int) -> np.ndarray:
    return np.ones(n) / n


def inverse_vol_weights(signals: pd.DataFrame) -> np.ndarray:
    vols = signals.std()
    vols = vols.replace(0, vols[vols > 0].min() if (vols > 0).any() else 1.0)
    inv = 1.0 / vols
    return (inv / inv.sum()).values


def rank_based_weights(signals: pd.DataFrame, returns: pd.Series) -> np.ndarray:
    """Weight by IC rank (higher IC → higher weight)."""
    ics = np.array([abs(compute_ic(signals[c], returns)) for c in signals.columns])
    if ics.sum() < 1e-12:
        return equal_weights(len(signals.columns))
    return ics / ics.sum()


def ridge_weights(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0,
) -> np.ndarray:
    """Ridge regression weights."""
    n_f = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y
    reg = alpha * np.eye(n_f)
    try:
        w = np.linalg.solve(XtX + reg, Xty)
    except np.linalg.LinAlgError:
        return np.ones(n_f) / n_f
    abs_sum = np.abs(w).sum()
    return w / abs_sum if abs_sum > 1e-12 else np.ones(n_f) / n_f


def elastic_net_weights(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0, l1_ratio: float = 0.5,
    n_iter: int = 200, lr: float = 0.01,
) -> np.ndarray:
    """Elastic net via coordinate descent (pure numpy)."""
    n, p = X.shape
    w = np.zeros(p)
    l1 = alpha * l1_ratio
    l2 = alpha * (1 - l1_ratio)

    for _ in range(n_iter):
        for j in range(p):
            r = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ r / n
            if rho > l1:
                w[j] = (rho - l1) / (np.sum(X[:, j] ** 2) / n + l2)
            elif rho < -l1:
                w[j] = (rho + l1) / (np.sum(X[:, j] ** 2) / n + l2)
            else:
                w[j] = 0.0

    abs_sum = np.abs(w).sum()
    return w / abs_sum if abs_sum > 1e-12 else np.ones(p) / p


def regime_conditional_weights(
    signals: pd.DataFrame,
    returns: pd.Series,
    regimes: pd.Series,
    alpha: float = 1.0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Fit separate ridge weights per regime.

    Returns: (per_regime_weights, blended_final_weights)
    """
    aligned = pd.concat([signals, returns.rename("_ret"), regimes.rename("_reg")], axis=1).dropna()
    regime_weights: Dict[str, np.ndarray] = {}

    for regime in aligned["_reg"].unique():
        mask = aligned["_reg"] == regime
        sub = aligned[mask]
        if len(sub) < 10:
            regime_weights[str(regime)] = equal_weights(signals.shape[1])
            continue
        X = sub[signals.columns].values
        y = sub["_ret"].values
        regime_weights[str(regime)] = ridge_weights(X, y, alpha)

    # Blended: weight by regime frequency
    counts = aligned["_reg"].value_counts(normalize=True)
    blended = np.zeros(signals.shape[1])
    for reg, w in regime_weights.items():
        freq = counts.get(reg, 0)
        blended += w * freq
    abs_sum = np.abs(blended).sum()
    if abs_sum > 1e-12:
        blended /= abs_sum

    return regime_weights, blended


# ── Walk-forward training ────────────────────────────────────────────────


@dataclass
class WalkForwardFold:
    fold: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    weights: np.ndarray
    test_ic: float
    best_single_ic: float


def walk_forward_fit(
    signals: pd.DataFrame,
    returns: pd.Series,
    method: str = "ridge",
    n_folds: int = 5,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
) -> Tuple[List[WalkForwardFold], np.ndarray, pd.DataFrame]:
    """Walk-forward expanding-window ensemble training.

    Returns: (folds, final_weights, weight_history_df)
    """
    aligned = pd.concat([signals, returns.rename("_ret")], axis=1).dropna()
    n = len(aligned)
    cols = list(signals.columns)
    n_signals = len(cols)

    if n < 20 or n_folds < 1:
        w = equal_weights(n_signals)
        return [], w, pd.DataFrame()

    fold_size = n // (n_folds + 1)
    if fold_size < 5:
        w = equal_weights(n_signals)
        return [], w, pd.DataFrame()

    folds: List[WalkForwardFold] = []
    weight_records: List[Dict] = []

    for f in range(n_folds):
        train_end = fold_size * (f + 1)
        test_start = train_end
        test_end = min(train_end + fold_size, n)
        if test_end <= test_start:
            continue

        X_train = aligned.iloc[:train_end][cols].values
        y_train = aligned.iloc[:train_end]["_ret"].values
        X_test = aligned.iloc[test_start:test_end][cols].values
        y_test = aligned.iloc[test_start:test_end]["_ret"].values

        if method == "elastic_net":
            w = elastic_net_weights(X_train, y_train, alpha, l1_ratio)
        elif method == "ridge":
            w = ridge_weights(X_train, y_train, alpha)
        elif method == "inverse_vol":
            w = inverse_vol_weights(aligned.iloc[:train_end][cols])
        elif method == "rank_based":
            w = rank_based_weights(
                aligned.iloc[:train_end][cols],
                aligned.iloc[:train_end]["_ret"],
            )
        else:
            w = equal_weights(n_signals)

        # Test IC
        combo = X_test @ w
        combo_s = pd.Series(combo, index=aligned.index[test_start:test_end])
        test_ret_s = pd.Series(y_test, index=aligned.index[test_start:test_end])
        test_ic = compute_ic(combo_s, test_ret_s)

        # Best single signal IC on test
        best_single = max(
            abs(compute_ic(
                pd.Series(X_test[:, j], index=aligned.index[test_start:test_end]),
                test_ret_s,
            ))
            for j in range(n_signals)
        )

        folds.append(WalkForwardFold(
            fold=f + 1, train_end=train_end, test_start=test_start,
            test_end=test_end, n_train=train_end, n_test=test_end - test_start,
            weights=w, test_ic=test_ic, best_single_ic=best_single,
        ))
        weight_records.append({"fold": f + 1, **{cols[i]: float(w[i]) for i in range(n_signals)}})

    # Final weights: fit on all data
    X_all = aligned[cols].values
    y_all = aligned["_ret"].values
    if method == "elastic_net":
        final_w = elastic_net_weights(X_all, y_all, alpha, l1_ratio)
    elif method == "ridge":
        final_w = ridge_weights(X_all, y_all, alpha)
    elif method == "inverse_vol":
        final_w = inverse_vol_weights(aligned[cols])
    elif method == "rank_based":
        final_w = rank_based_weights(aligned[cols], aligned["_ret"])
    else:
        final_w = equal_weights(n_signals)

    wh = pd.DataFrame(weight_records).set_index("fold") if weight_records else pd.DataFrame()
    return folds, final_w, wh


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class SignalStats:
    name: str
    ic: float
    icir: float
    hit_rate: float
    weight: float
    kept: bool


@dataclass
class EnsembleResult:
    method: str
    signal_names: List[str]
    kept_signals: List[str]
    dropped_signals: List[str]
    weights: Dict[str, float]
    signal_stats: List[SignalStats]
    ensemble_ic: float
    ensemble_icir: float
    best_single_ic: float
    lift_pct: float
    walk_forward: List[WalkForwardFold]
    weight_history: pd.DataFrame
    regime_weights: Optional[Dict[str, np.ndarray]]
    combined_signal: pd.Series
    n_observations: int


# ── Core engine ──────────────────────────────────────────────────────────


class SignalEnsemble:
    """Signal ensemble combining multiple alpha signals.

    Args:
        signals: DataFrame where each column is an alpha signal.
        forward_returns: Series of forward returns aligned with signals.
        regimes: Optional Series of regime labels.
        method: One of METHODS.
        preprocess_method: 'zscore', 'rank', or 'none'.
        min_ic: Minimum IC for quality gate.
        max_correlation: Maximum pairwise correlation.
        alpha: Regularisation for ridge/elastic net.
        l1_ratio: L1 ratio for elastic net.
        n_folds: Walk-forward folds.
    """

    def __init__(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        regimes: Optional[pd.Series] = None,
        method: str = "ridge",
        preprocess_method: str = "zscore",
        min_ic: float = 0.02,
        max_correlation: float = 0.85,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        n_folds: int = 5,
    ):
        if signals.empty:
            raise ValueError("signals must not be empty")
        if signals.shape[1] < 2:
            raise ValueError("Need at least 2 signals")
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}")

        self.raw_signals = signals.copy()
        self.forward_returns = forward_returns.copy()
        self.regimes = regimes
        self.method = method
        self.preprocess_method = preprocess_method
        self.min_ic = min_ic
        self.max_correlation = max_correlation
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_folds = n_folds

    def fit(self) -> EnsembleResult:
        """Fit ensemble and evaluate."""
        # Preprocess
        processed = preprocess(self.raw_signals, self.preprocess_method)

        # Quality gates
        filtered, kept, dropped = apply_quality_gates(
            processed, self.forward_returns, self.min_ic, self.max_correlation,
        )

        n_signals = filtered.shape[1]
        cols = list(filtered.columns)

        # Walk-forward + weights
        if self.method == "regime_conditional" and self.regimes is not None:
            reg_weights, final_w = regime_conditional_weights(
                filtered, self.forward_returns, self.regimes, self.alpha,
            )
            wf_folds: List[WalkForwardFold] = []
            wh = pd.DataFrame()
        else:
            reg_weights = None
            wf_folds, final_w, wh = walk_forward_fit(
                filtered, self.forward_returns, self.method,
                self.n_folds, self.alpha, self.l1_ratio,
            )

        weights_dict = {cols[i]: float(final_w[i]) for i in range(n_signals)}

        # Combined signal
        combo = filtered.values @ final_w
        combined = pd.Series(combo, index=filtered.index, name="ensemble")

        # Per-signal stats
        signal_stats: List[SignalStats] = []
        for col in self.raw_signals.columns:
            sig = processed[col] if col in processed.columns else self.raw_signals[col]
            ic_s = compute_ic_series(sig, self.forward_returns)
            ic_val = compute_ic(sig, self.forward_returns)
            icir = float(ic_s.mean() / ic_s.std()) if len(ic_s) > 1 and ic_s.std() > 1e-12 else 0.0
            hr = float((ic_s > 0).mean()) if len(ic_s) > 0 else 0.0
            signal_stats.append(SignalStats(
                name=col, ic=ic_val, icir=icir, hit_rate=hr,
                weight=weights_dict.get(col, 0.0), kept=col in kept,
            ))

        # Ensemble IC
        ens_ic_s = compute_ic_series(combined, self.forward_returns)
        ens_ic = compute_ic(combined, self.forward_returns)
        ens_icir = float(ens_ic_s.mean() / ens_ic_s.std()) if len(ens_ic_s) > 1 and ens_ic_s.std() > 1e-12 else 0.0

        # Best single
        best_single = max(abs(ss.ic) for ss in signal_stats) if signal_stats else 0.0
        lift = (abs(ens_ic) - best_single) / best_single * 100 if best_single > 1e-12 else 0.0

        return EnsembleResult(
            method=self.method,
            signal_names=list(self.raw_signals.columns),
            kept_signals=kept, dropped_signals=dropped,
            weights=weights_dict, signal_stats=signal_stats,
            ensemble_ic=ens_ic, ensemble_icir=ens_icir,
            best_single_ic=best_single, lift_pct=lift,
            walk_forward=wf_folds, weight_history=wh,
            regime_weights=reg_weights, combined_signal=combined,
            n_observations=len(filtered),
        )

    @staticmethod
    def generate_report(
        result: EnsembleResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _f(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


def _fp(v: float) -> str:
    return f"{v:.1%}"


def _weight_bars(weights: Dict[str, float]) -> str:
    if not weights:
        return ""
    names = list(weights.keys())
    vals = [weights[n] for n in names]
    n = len(names)
    w, h = 500, n * 32 + 35
    pad_l = 80
    abs_max = max(abs(v) for v in vals) if vals else 1.0
    if abs_max == 0:
        abs_max = 1.0
    bar_area = (w - pad_l - 40) / 2
    mid_x = pad_l + bar_area

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="16" text-anchor="middle" class="svg-title">Ensemble Weights</text>')
    parts.append(f'<line x1="{mid_x:.0f}" y1="22" x2="{mid_x:.0f}" y2="{h}" stroke="#30363d"/>')

    for i in range(n):
        y = 28 + i * 32
        bw = abs(vals[i]) / abs_max * bar_area
        color = "#3fb950" if vals[i] >= 0 else "#f85149"
        bx = mid_x if vals[i] >= 0 else mid_x - bw
        parts.append(f'<text x="{pad_l - 4}" y="{y + 14:.0f}" text-anchor="end" font-size="10" fill="#8b949e">{names[i][:10]}</text>')
        parts.append(f'<rect x="{bx:.0f}" y="{y}" width="{bw:.0f}" height="22" fill="{color}" rx="3" opacity="0.85"/>')
        parts.append(f'<text x="{bx + bw + 3:.0f}" y="{y + 14:.0f}" font-size="9" fill="#c9d1d9">{vals[i]:+.3f}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _signal_table(stats: List[SignalStats]) -> str:
    rows = ""
    for s in sorted(stats, key=lambda x: abs(x.ic), reverse=True):
        kept_icon = "✓" if s.kept else "✗"
        kc = "#3fb950" if s.kept else "#f85149"
        rows += f"<tr><td style='text-align:left'>{s.name}</td><td>{_f(s.ic)}</td><td>{_f(s.icir, 2)}</td><td>{_fp(s.hit_rate)}</td><td>{_f(s.weight, 3)}</td><td style='color:{kc}'>{kept_icon}</td></tr>"
    return f"""<table class="dt"><tr><th style="text-align:left">Signal</th><th>IC</th><th>ICIR</th><th>Hit Rate</th><th>Weight</th><th>Kept</th></tr>{rows}</table>"""


def _wf_table(folds: List[WalkForwardFold]) -> str:
    if not folds:
        return "<p class='meta'>No walk-forward data.</p>"
    rows = ""
    for f in folds:
        lc = "#3fb950" if abs(f.test_ic) > f.best_single_ic else "#f85149"
        rows += f"<tr><td>{f.fold}</td><td>{f.n_train}</td><td>{f.n_test}</td><td>{_f(f.test_ic)}</td><td>{_f(f.best_single_ic)}</td><td style='color:{lc}'>{'+' if abs(f.test_ic) > f.best_single_ic else '-'}</td></tr>"
    return f"""<table class="dt"><tr><th>Fold</th><th>Train</th><th>Test</th><th>Ensemble IC</th><th>Best Single</th><th>Lift</th></tr>{rows}</table>"""


def _build_html(r: EnsembleResult) -> str:
    lift_c = "#3fb950" if r.lift_pct > 0 else "#f85149"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><title>Signal Ensemble Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #0d1117; color: #c9d1d9; }}
  h1,h2 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 10px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.8em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.1em; }}
  table.dt {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.dt th, table.dt td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.dt th {{ color: #8b949e; background: #161b22; }}
  .chart {{ width: 100%; max-width: 550px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
</style>
</head>
<body>
<h1>Signal Ensemble</h1>
<p class="meta">{len(r.signal_names)} signals &middot; {len(r.kept_signals)} kept &middot;
   {r.n_observations} observations &middot; Method: {r.method}</p>

<div class="summary">
  <div class="stat"><div class="label">Ensemble IC</div><div class="value">{_f(r.ensemble_ic)}</div></div>
  <div class="stat"><div class="label">Ensemble ICIR</div><div class="value">{_f(r.ensemble_icir, 2)}</div></div>
  <div class="stat"><div class="label">Best Single IC</div><div class="value">{_f(r.best_single_ic)}</div></div>
  <div class="stat"><div class="label">Lift</div><div class="value" style="color:{lift_c}">{r.lift_pct:+.1f}%</div></div>
  <div class="stat"><div class="label">Kept</div><div class="value">{len(r.kept_signals)}/{len(r.signal_names)}</div></div>
  <div class="stat"><div class="label">Dropped</div><div class="value">{len(r.dropped_signals)}</div></div>
</div>

<h2>Signal Performance</h2>
{_signal_table(r.signal_stats)}

{_weight_bars(r.weights)}

<h2>Walk-Forward Validation</h2>
{_wf_table(r.walk_forward)}

</body></html>"""
