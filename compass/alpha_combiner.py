"""
Alpha signal combiner — merges multiple alpha signals into a composite.

Combination methods:
  - Equal weight
  - Inverse-volatility weight
  - ML-learned weights (ridge regression on forward returns)
  - Correlation-aware de-weighting (penalise redundant signals)
  - Dynamic weight adjustment (recent IC-based re-weighting)

Signal processing:
  - Z-score normalisation
  - Cross-sectional rank normalisation

Evaluation:
  - IC  (information coefficient — Spearman rank correlation with returns)
  - ICIR (IC information ratio — mean IC / std IC)
  - Turnover (period-to-period weight change)
  - Out-of-sample expanding-window evaluation

HTML report at reports/alpha_combiner.html with correlation matrix heatmap,
weight evolution chart, and IC dashboard.

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.alpha_combiner import AlphaCombiner
    ac = AlphaCombiner(signals_df, forward_returns)
    result = ac.combine()
    AlphaCombiner.generate_report(result)
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
DEFAULT_OUTPUT = ROOT / "reports" / "alpha_combiner.html"


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class SignalMetrics:
    """Per-signal quality metrics."""

    name: str
    mean_ic: float
    std_ic: float
    icir: float
    hit_rate: float  # fraction of periods with positive IC
    avg_turnover: float
    weight: float


@dataclass
class CombinedMetrics:
    """Metrics for the combined alpha signal."""

    method: str
    mean_ic: float
    std_ic: float
    icir: float
    hit_rate: float
    avg_turnover: float


@dataclass
class OOSResult:
    """Out-of-sample evaluation result."""

    n_periods: int
    train_ic: float
    test_ic: float
    train_icir: float
    test_icir: float
    ic_decay: float  # test_ic / train_ic


@dataclass
class CombinerResult:
    """Full result from alpha combination."""

    signal_names: List[str]
    weights: np.ndarray
    method: str
    signal_metrics: List[SignalMetrics]
    combined_metrics: CombinedMetrics
    correlation_matrix: pd.DataFrame
    weight_history: Optional[pd.DataFrame]  # for dynamic methods
    oos_result: Optional[OOSResult]
    combined_signal: pd.DataFrame


# ── Normalisation ────────────────────────────────────────────────────────


def zscore_normalize(signals: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalise each signal column (cross-time)."""
    mu = signals.mean()
    sigma = signals.std()
    sigma = sigma.replace(0, 1.0)
    return (signals - mu) / sigma


def rank_normalize(signals: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank normalisation (per row) mapped to [-1, 1]."""
    ranked = signals.rank(axis=1, pct=True)
    return ranked * 2 - 1


# ── IC / ICIR / turnover ────────────────────────────────────────────────


def compute_ic_series(
    signal: pd.Series,
    forward_returns: pd.Series,
) -> pd.Series:
    """Compute rolling IC (Spearman rank correlation) per period.

    For time-series signals we compute the sign-agreement per period.
    Returns a Series of per-period IC values.
    """
    aligned = pd.concat(
        [signal.rename("sig"), forward_returns.rename("ret")], axis=1
    ).dropna()
    if len(aligned) < 3:
        return pd.Series(dtype=float)

    # Rolling IC: use expanding windows of at least 20 periods
    # For simplicity, compute single-period sign-agreement IC
    ic_vals = []
    ic_dates = []
    window = min(20, len(aligned) // 2)
    if window < 3:
        window = 3
    for end in range(window, len(aligned) + 1):
        chunk = aligned.iloc[end - window : end]
        # Spearman rank correlation
        rk_sig = chunk["sig"].rank()
        rk_ret = chunk["ret"].rank()
        n = len(rk_sig)
        if n < 3:
            continue
        d = rk_sig - rk_ret
        spearman = 1.0 - 6.0 * (d**2).sum() / (n * (n**2 - 1))
        ic_vals.append(spearman)
        ic_dates.append(chunk.index[-1])

    return pd.Series(ic_vals, index=ic_dates, name="ic")


def compute_ic(signal: pd.Series, forward_returns: pd.Series) -> float:
    """Single IC value (Spearman rank correlation)."""
    aligned = pd.concat(
        [signal.rename("sig"), forward_returns.rename("ret")], axis=1
    ).dropna()
    if len(aligned) < 3:
        return 0.0
    rk_sig = aligned["sig"].rank()
    rk_ret = aligned["ret"].rank()
    n = len(rk_sig)
    d = rk_sig - rk_ret
    return float(1.0 - 6.0 * (d**2).sum() / (n * (n**2 - 1)))


def compute_icir(ic_series: pd.Series) -> float:
    """IC Information Ratio = mean(IC) / std(IC)."""
    if len(ic_series) < 2:
        return 0.0
    std = float(ic_series.std())
    if std < 1e-12:
        return 0.0
    return float(ic_series.mean() / std)


def compute_turnover(weights_history: pd.DataFrame) -> float:
    """Average absolute weight change between consecutive periods."""
    if len(weights_history) < 2:
        return 0.0
    diffs = weights_history.diff().iloc[1:]
    return float(diffs.abs().sum(axis=1).mean())


def compute_signal_turnover(signal: pd.Series) -> float:
    """Average absolute change in signal value."""
    if len(signal) < 2:
        return 0.0
    return float(signal.diff().abs().mean())


# ── Weighting methods ────────────────────────────────────────────────────


def equal_weights(n: int) -> np.ndarray:
    """Equal weight across n signals."""
    return np.ones(n) / n


def inverse_vol_weights(signals: pd.DataFrame) -> np.ndarray:
    """Weight inversely proportional to signal volatility."""
    vols = signals.std()
    vols = vols.replace(0, vols[vols > 0].min() if (vols > 0).any() else 1.0)
    inv = 1.0 / vols
    return (inv / inv.sum()).values


def ridge_weights(
    signals: pd.DataFrame,
    forward_returns: pd.Series,
    alpha: float = 1.0,
) -> np.ndarray:
    """ML-learned weights via ridge regression."""
    aligned = pd.concat(
        [signals, forward_returns.rename("_ret")], axis=1
    ).dropna()
    if len(aligned) < 10:
        return equal_weights(signals.shape[1])

    X = aligned[signals.columns].values
    y = aligned["_ret"].values

    # Ridge: w = (X'X + alpha*I)^{-1} X'y
    XtX = X.T @ X
    Xty = X.T @ y
    n_features = X.shape[1]
    try:
        w = np.linalg.solve(XtX + alpha * np.eye(n_features), Xty)
    except np.linalg.LinAlgError:
        return equal_weights(n_features)

    # Normalise to sum to 1 (allow negative for hedging, but normalise abs)
    abs_sum = np.abs(w).sum()
    if abs_sum < 1e-12:
        return equal_weights(n_features)
    return w / abs_sum


def correlation_deweight(
    signals: pd.DataFrame,
    base_weights: np.ndarray,
    threshold: float = 0.7,
) -> np.ndarray:
    """Penalise correlated signals to reduce redundancy.

    For each pair with |corr| > threshold, reduce the lower-IC signal's weight.
    """
    corr = signals.corr().values
    n = len(base_weights)
    penalties = np.ones(n)

    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) > threshold:
                # Penalise both, but more on the one with lower absolute weight
                if abs(base_weights[i]) < abs(base_weights[j]):
                    penalties[i] *= 1.0 - abs(corr[i, j]) * 0.5
                else:
                    penalties[j] *= 1.0 - abs(corr[i, j]) * 0.5

    adjusted = base_weights * penalties
    abs_sum = np.abs(adjusted).sum()
    if abs_sum < 1e-12:
        return equal_weights(n)
    return adjusted / abs_sum


def dynamic_weights(
    signals: pd.DataFrame,
    forward_returns: pd.Series,
    lookback: int = 63,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Compute time-varying weights based on rolling IC performance.

    Uses expanding window with minimum `lookback` periods.
    Returns DataFrame of weights over time.
    """
    aligned = pd.concat(
        [signals, forward_returns.rename("_ret")], axis=1
    ).dropna()

    records = []
    for end in range(lookback, len(aligned) + 1):
        train = aligned.iloc[:end]
        X = train[signals.columns].values
        y = train["_ret"].values

        # Ridge regression on expanding window
        XtX = X.T @ X
        Xty = X.T @ y
        n_f = X.shape[1]
        try:
            w = np.linalg.solve(XtX + ridge_alpha * np.eye(n_f), Xty)
        except np.linalg.LinAlgError:
            w = np.ones(n_f) / n_f

        abs_sum = np.abs(w).sum()
        if abs_sum < 1e-12:
            w = np.ones(n_f) / n_f
        else:
            w = w / abs_sum

        records.append(
            {"date": train.index[-1], **dict(zip(signals.columns, w))}
        )

    if not records:
        return pd.DataFrame(columns=["date"] + list(signals.columns))
    return pd.DataFrame(records).set_index("date")


# ── Out-of-sample evaluation ─────────────────────────────────────────────


def oos_evaluate(
    signals: pd.DataFrame,
    forward_returns: pd.Series,
    train_frac: float = 0.7,
    ridge_alpha: float = 1.0,
) -> OOSResult:
    """Expanding-window out-of-sample evaluation."""
    aligned = pd.concat(
        [signals, forward_returns.rename("_ret")], axis=1
    ).dropna()
    n = len(aligned)
    split = int(n * train_frac)

    if split < 10 or n - split < 5:
        return OOSResult(
            n_periods=n, train_ic=0.0, test_ic=0.0,
            train_icir=0.0, test_icir=0.0, ic_decay=0.0,
        )

    train = aligned.iloc[:split]
    test = aligned.iloc[split:]

    # Fit on train
    X_train = train[signals.columns].values
    y_train = train["_ret"].values
    n_f = X_train.shape[1]
    try:
        w = np.linalg.solve(
            X_train.T @ X_train + ridge_alpha * np.eye(n_f),
            X_train.T @ y_train,
        )
    except np.linalg.LinAlgError:
        w = np.ones(n_f) / n_f

    # Compute combined signal
    train_combo = pd.Series(
        X_train @ w, index=train.index, name="combo"
    )
    test_combo = pd.Series(
        test[signals.columns].values @ w, index=test.index, name="combo"
    )

    train_ic_series = compute_ic_series(train_combo, train["_ret"])
    test_ic_series = compute_ic_series(test_combo, test["_ret"])

    train_ic = float(train_ic_series.mean()) if len(train_ic_series) > 0 else 0.0
    test_ic = float(test_ic_series.mean()) if len(test_ic_series) > 0 else 0.0
    train_icir = compute_icir(train_ic_series)
    test_icir = compute_icir(test_ic_series)

    ic_decay = test_ic / train_ic if abs(train_ic) > 1e-12 else 0.0

    return OOSResult(
        n_periods=n,
        train_ic=train_ic,
        test_ic=test_ic,
        train_icir=train_icir,
        test_icir=test_icir,
        ic_decay=ic_decay,
    )


# ── Core combiner ────────────────────────────────────────────────────────


class AlphaCombiner:
    """Combines multiple alpha signals into a composite."""

    METHODS = ("equal", "inverse_vol", "ridge", "correlation_aware", "dynamic")

    def __init__(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        method: str = "ridge",
        ridge_alpha: float = 1.0,
        corr_threshold: float = 0.7,
        dynamic_lookback: int = 63,
        normalize: str = "zscore",
    ):
        """
        Args:
            signals: DataFrame where each column is an alpha signal.
            forward_returns: Series of forward returns aligned with signals.
            method: One of METHODS.
            ridge_alpha: Regularisation for ridge regression.
            corr_threshold: Correlation threshold for de-weighting.
            dynamic_lookback: Minimum lookback for dynamic weights.
            normalize: 'zscore', 'rank', or 'none'.
        """
        if signals.empty:
            raise ValueError("signals DataFrame must not be empty")
        if signals.shape[1] < 2:
            raise ValueError("Need at least 2 signals to combine")
        if method not in self.METHODS:
            raise ValueError(f"Unknown method {method!r}, choose from {self.METHODS}")

        self.raw_signals = signals.copy()
        self.forward_returns = forward_returns.copy()
        self.method = method
        self.ridge_alpha = ridge_alpha
        self.corr_threshold = corr_threshold
        self.dynamic_lookback = dynamic_lookback
        self.normalize = normalize
        self.signal_names = list(signals.columns)
        self.n_signals = len(self.signal_names)

        # Normalise
        if normalize == "zscore":
            self.signals = zscore_normalize(signals)
        elif normalize == "rank":
            self.signals = rank_normalize(signals)
        else:
            self.signals = signals.copy()

    def _compute_weights(self) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        """Compute signal weights based on selected method."""
        weight_history = None

        if self.method == "equal":
            w = equal_weights(self.n_signals)
        elif self.method == "inverse_vol":
            w = inverse_vol_weights(self.signals)
        elif self.method == "ridge":
            w = ridge_weights(self.signals, self.forward_returns, self.ridge_alpha)
        elif self.method == "correlation_aware":
            base = ridge_weights(self.signals, self.forward_returns, self.ridge_alpha)
            w = correlation_deweight(self.signals, base, self.corr_threshold)
        elif self.method == "dynamic":
            wh = dynamic_weights(
                self.signals, self.forward_returns,
                self.dynamic_lookback, self.ridge_alpha,
            )
            weight_history = wh
            w = wh.iloc[-1].values if len(wh) > 0 else equal_weights(self.n_signals)
        else:
            w = equal_weights(self.n_signals)

        return w, weight_history

    def _compute_signal_metrics(self, weights: np.ndarray) -> List[SignalMetrics]:
        """Compute per-signal quality metrics."""
        metrics = []
        for i, name in enumerate(self.signal_names):
            sig = self.signals[name]
            ic_series = compute_ic_series(sig, self.forward_returns)
            mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
            std_ic = float(ic_series.std()) if len(ic_series) > 1 else 0.0
            icir = compute_icir(ic_series)
            hit = float((ic_series > 0).mean()) if len(ic_series) > 0 else 0.0
            turnover = compute_signal_turnover(sig)

            metrics.append(SignalMetrics(
                name=name,
                mean_ic=mean_ic,
                std_ic=std_ic,
                icir=icir,
                hit_rate=hit,
                avg_turnover=turnover,
                weight=float(weights[i]),
            ))
        return metrics

    def combine(self) -> CombinerResult:
        """Run full alpha combination."""
        weights, weight_history = self._compute_weights()
        signal_metrics = self._compute_signal_metrics(weights)
        corr_matrix = self.signals.corr()

        # Build combined signal
        combined = self.signals.values @ weights
        combined_series = pd.Series(combined, index=self.signals.index, name="combined")
        combined_df = self.signals.copy()
        combined_df["combined"] = combined_series

        # Combined metrics
        ic_series = compute_ic_series(combined_series, self.forward_returns)
        mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else 0.0
        std_ic = float(ic_series.std()) if len(ic_series) > 1 else 0.0
        icir = compute_icir(ic_series)
        hit = float((ic_series > 0).mean()) if len(ic_series) > 0 else 0.0

        wh_turnover = 0.0
        if weight_history is not None and len(weight_history) > 1:
            wh_turnover = compute_turnover(weight_history)

        combined_metrics = CombinedMetrics(
            method=self.method,
            mean_ic=mean_ic,
            std_ic=std_ic,
            icir=icir,
            hit_rate=hit,
            avg_turnover=wh_turnover,
        )

        # OOS evaluation
        oos = oos_evaluate(
            self.signals, self.forward_returns,
            ridge_alpha=self.ridge_alpha,
        )

        return CombinerResult(
            signal_names=self.signal_names,
            weights=weights,
            method=self.method,
            signal_metrics=signal_metrics,
            combined_metrics=combined_metrics,
            correlation_matrix=corr_matrix,
            weight_history=weight_history,
            oos_result=oos,
            combined_signal=combined_df,
        )

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: CombinerResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def _fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def _corr_heatmap_svg(corr: pd.DataFrame) -> str:
    """Inline SVG heatmap of signal correlations."""
    names = list(corr.columns)
    n = len(names)
    if n == 0:
        return ""

    cell = 50
    label_pad = 80
    w = label_pad + n * cell + 10
    h = label_pad + n * cell + 10

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(
        f'<text x="{w // 2}" y="15" text-anchor="middle" class="svg-title">'
        f"Signal Correlation Matrix</text>"
    )

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            # Color: blue (negative) → white (0) → red (positive)
            if val >= 0:
                r_c = 255
                g_c = int(255 * (1 - val))
                b_c = int(255 * (1 - val))
            else:
                r_c = int(255 * (1 + val))
                g_c = int(255 * (1 + val))
                b_c = 255

            x = label_pad + j * cell
            y = 25 + i * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="rgb({r_c},{g_c},{b_c})" stroke="#30363d" stroke-width="0.5"/>'
            )
            parts.append(
                f'<text x="{x + cell // 2}" y="{y + cell // 2 + 4}" '
                f'text-anchor="middle" font-size="10" fill="#000">{val:.2f}</text>'
            )

    # Labels
    for i, name in enumerate(names):
        # Top labels
        x = label_pad + i * cell + cell // 2
        parts.append(
            f'<text x="{x}" y="{23}" text-anchor="middle" font-size="9" '
            f'fill="#8b949e">{name[:8]}</text>'
        )
        # Left labels
        y = 25 + i * cell + cell // 2 + 4
        parts.append(
            f'<text x="{label_pad - 5}" y="{y}" text-anchor="end" font-size="9" '
            f'fill="#8b949e">{name[:8]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _weight_evolution_svg(wh: Optional[pd.DataFrame]) -> str:
    """Stacked line chart of weight evolution over time."""
    if wh is None or wh.empty:
        return "<p class='meta'>Weight history not available (static method).</p>"

    w, h = 700, 280
    pad = 55
    pw = w - 2 * pad
    ph = h - 70

    n = len(wh)
    cols = list(wh.columns)
    colors = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff",
              "#f0883e", "#8b949e", "#da3633", "#1f6feb", "#238636"]

    all_vals = wh.values.flatten()
    y_min = float(np.nanmin(all_vals)) * 1.1
    y_max = float(np.nanmax(all_vals)) * 1.1
    if y_max <= y_min:
        y_max = y_min + 0.1

    def tx(i: int) -> float:
        return pad + i / max(n - 1, 1) * pw

    def ty(v: float) -> float:
        return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(
        f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">'
        f"Weight Evolution Over Time</text>"
    )

    # Zero line
    if y_min < 0 < y_max:
        zy = ty(0)
        parts.append(
            f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" '
            f'stroke="#30363d" stroke-dasharray="3,3"/>'
        )

    for ci, col in enumerate(cols):
        vals = wh[col].values
        color = colors[ci % len(colors)]
        d = " ".join(
            f"{'M' if j == 0 else 'L'}{tx(j):.1f},{ty(float(vals[j])):.1f}"
            for j in range(n)
            if not np.isnan(vals[j])
        )
        parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        # Legend
        lx = pad + ci * 90
        parts.append(
            f'<rect x="{lx}" y="{h - 15}" width="10" height="10" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{lx + 14}" y="{h - 6}" font-size="9" '
            f'fill="#8b949e">{col[:8]}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _ic_dashboard(
    signal_metrics: List[SignalMetrics],
    combined: CombinedMetrics,
) -> str:
    """IC metrics table and combined stats."""
    rows = ""
    for sm in signal_metrics:
        rows += f"""<tr>
          <td style="text-align:left">{sm.name}</td>
          <td>{_fmt(sm.mean_ic)}</td>
          <td>{_fmt(sm.std_ic)}</td>
          <td>{_fmt(sm.icir, 2)}</td>
          <td>{_fmt_pct(sm.hit_rate)}</td>
          <td>{_fmt(sm.avg_turnover)}</td>
          <td>{_fmt(sm.weight, 3)}</td>
        </tr>"""

    return f"""
    <div class="card">
      <h3>IC Dashboard</h3>
      <div class="combined-stats">
        <span>Combined IC: <strong>{_fmt(combined.mean_ic)}</strong></span>
        <span>ICIR: <strong>{_fmt(combined.icir, 2)}</strong></span>
        <span>Hit Rate: <strong>{_fmt_pct(combined.hit_rate)}</strong></span>
        <span>Method: <strong>{combined.method}</strong></span>
      </div>
      <table class="data-table">
        <tr><th style="text-align:left">Signal</th><th>Mean IC</th><th>Std IC</th>
            <th>ICIR</th><th>Hit Rate</th><th>Turnover</th><th>Weight</th></tr>
        {rows}
      </table>
    </div>"""


def _oos_card(oos: Optional[OOSResult]) -> str:
    if oos is None:
        return ""
    decay_color = "#3fb950" if oos.ic_decay > 0.5 else "#d29922" if oos.ic_decay > 0 else "#f85149"
    return f"""
    <div class="card">
      <h3>Out-of-Sample Evaluation</h3>
      <div class="metrics-grid">
        <div><span class="label">Train IC</span><span class="value">{_fmt(oos.train_ic)}</span></div>
        <div><span class="label">Test IC</span><span class="value">{_fmt(oos.test_ic)}</span></div>
        <div><span class="label">Train ICIR</span><span class="value">{_fmt(oos.train_icir, 2)}</span></div>
        <div><span class="label">Test ICIR</span><span class="value">{_fmt(oos.test_icir, 2)}</span></div>
        <div><span class="label">IC Decay</span>
          <span class="value" style="color:{decay_color}">{_fmt(oos.ic_decay, 2)}</span></div>
        <div><span class="label">Periods</span><span class="value">{oos.n_periods}</span></div>
      </div>
    </div>"""


def _weights_bar_svg(names: List[str], weights: np.ndarray) -> str:
    """Horizontal bar chart of final weights."""
    n = len(names)
    if n == 0:
        return ""
    bar_h = 24
    gap = 6
    pad_l = 90
    w = 500
    h = n * (bar_h + gap) + 30

    abs_max = max(abs(weights).max(), 0.01)

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(
        f'<text x="{w // 2}" y="15" text-anchor="middle" class="svg-title">'
        f"Final Signal Weights</text>"
    )

    mid_x = pad_l + (w - pad_l - 30) / 2
    parts.append(
        f'<line x1="{mid_x:.0f}" y1="22" x2="{mid_x:.0f}" y2="{h}" '
        f'stroke="#30363d" stroke-width="1"/>'
    )

    bar_area = (w - pad_l - 30) / 2
    for i in range(n):
        y = 25 + i * (bar_h + gap)
        bw = abs(weights[i]) / abs_max * bar_area
        color = "#3fb950" if weights[i] >= 0 else "#f85149"
        bx = mid_x if weights[i] >= 0 else mid_x - bw

        parts.append(
            f'<text x="{pad_l - 5}" y="{y + bar_h * 0.7:.0f}" text-anchor="end" '
            f'font-size="10" fill="#8b949e">{names[i][:10]}</text>'
        )
        parts.append(
            f'<rect x="{bx:.0f}" y="{y}" width="{bw:.0f}" height="{bar_h}" '
            f'fill="{color}" rx="3" opacity="0.85"/>'
        )
        parts.append(
            f'<text x="{bx + bw + 4:.0f}" y="{y + bar_h * 0.7:.0f}" '
            f'font-size="9" fill="#c9d1d9">{weights[i]:+.3f}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _build_html(result: CombinerResult) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Alpha Signal Combiner Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; }}
  .combined-stats {{ display: flex; gap: 24px; margin-bottom: 12px; flex-wrap: wrap; }}
  .combined-stats span {{ color: #8b949e; }}
  .combined-stats strong {{ color: #f0f6fc; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
  .metrics-grid .label {{ color: #8b949e; font-size: 0.85em; }}
  .metrics-grid .value {{ color: #f0f6fc; font-weight: 600; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  .svg-label {{ fill: #8b949e; font-size: 10px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Alpha Signal Combiner</h1>
<p class="meta">{len(result.signal_names)} signals &middot; Method: {result.method} &middot;
   {len(result.combined_signal)} observations</p>

{_ic_dashboard(result.signal_metrics, result.combined_metrics)}

<div class="two-col">
  {_corr_heatmap_svg(result.correlation_matrix)}
  {_weights_bar_svg(result.signal_names, result.weights)}
</div>

<h2>Weight Evolution</h2>
{_weight_evolution_svg(result.weight_history)}

{_oos_card(result.oos_result)}

</body>
</html>"""
