"""
Cross-asset signal generator.

Components:
  - Cointegration testing: Engle-Granger (ADF on residuals), Johansen (trace)
  - Lead-lag detection across asset pairs (cross-correlation at lags)
  - Spread z-score signals with pair trading entry/exit rules
  - Rolling correlation regime detection (risk-on/risk-off/transition)
  - Cross-asset momentum spillover (relative strength transmission)

HTML report at reports/cross_asset_signal.html with cointegration table,
lead-lag heatmap, spread z-score charts.

This is READ-ONLY analysis.  No broker connections, no trade placement.

Usage::

    from compass.cross_asset_signal import CrossAssetSignalEngine
    engine = CrossAssetSignalEngine(prices_df)
    result = engine.analyze()
    CrossAssetSignalEngine.generate_report(result)
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
DEFAULT_OUTPUT = ROOT / "reports" / "cross_asset_signal.html"


# ── ADF test (pure numpy) ────────────────────────────────────────────────


def adf_test(series: np.ndarray) -> Tuple[float, float]:
    """Simplified ADF test.  Returns (adf_stat, approx_p_value)."""
    y = np.asarray(series, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 15:
        return 0.0, 1.0

    dy = np.diff(y)
    y_lag = y[:-1]
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, dy, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    b = beta[1]
    resid = dy - X @ beta
    se = np.sqrt(np.sum(resid ** 2) / max(len(dy) - 2, 1))
    denom = np.sqrt(np.sum((y_lag - y_lag.mean()) ** 2))
    if denom < 1e-15:
        return 0.0, 1.0
    se_b = se / denom
    adf = float(b / se_b)

    # Approximate p-value (MacKinnon asymptotic)
    if adf < -3.43:
        p = 0.005
    elif adf < -2.86:
        p = 0.03
    elif adf < -2.57:
        p = 0.07
    elif adf < -1.94:
        p = 0.15
    else:
        p = min(1.0, 0.5 + 0.3 * (adf + 1.94))
    return adf, max(0.0, min(1.0, p))


# ── Data classes ─────────────────────────────────────────────────────────


@dataclass
class CointegrationResult:
    """Cointegration test between two assets."""

    asset_a: str
    asset_b: str
    method: str  # "engle_granger" or "johansen_trace"
    cointegrated: bool
    test_stat: float
    p_value: float
    hedge_ratio: float
    half_life: float


@dataclass
class LeadLagResult:
    """Lead-lag relationship between two assets."""

    leader: str
    lagger: str
    optimal_lag: int
    correlation_at_lag: float
    direction: str  # "positive" or "negative"


@dataclass
class SpreadSignal:
    """Z-score pair trading signal."""

    asset_a: str
    asset_b: str
    current_z: float
    signal: str  # "long_spread", "short_spread", "neutral"
    hedge_ratio: float
    half_life: float
    entry_threshold: float
    exit_threshold: float


@dataclass
class CorrelationRegime:
    """Rolling correlation regime between two assets."""

    asset_a: str
    asset_b: str
    current_corr: float
    rolling_mean: float
    rolling_std: float
    z_score: float
    regime: str  # "high_corr", "low_corr", "normal"


@dataclass
class MomentumSpillover:
    """Cross-asset momentum spillover signal."""

    source: str
    target: str
    source_momentum: float
    spillover_beta: float
    predicted_move: float
    signal: str  # "bullish", "bearish", "neutral"


@dataclass
class AnalysisResult:
    """Full result from cross-asset analysis."""

    asset_names: List[str]
    cointegrations: List[CointegrationResult]
    lead_lags: List[LeadLagResult]
    spread_signals: List[SpreadSignal]
    correlation_regimes: List[CorrelationRegime]
    momentum_spillovers: List[MomentumSpillover]
    lead_lag_matrix: pd.DataFrame
    correlation_matrix: pd.DataFrame
    n_assets: int
    n_observations: int


# ── Engle-Granger cointegration ──────────────────────────────────────────


def engle_granger_test(
    y1: np.ndarray,
    y2: np.ndarray,
    p_threshold: float = 0.05,
) -> Tuple[bool, float, float, float, float]:
    """Engle-Granger two-step cointegration test.

    Returns: (cointegrated, adf_stat, p_value, hedge_ratio, half_life)
    """
    n = min(len(y1), len(y2))
    if n < 20:
        return False, 0.0, 1.0, 0.0, float("inf")

    y1, y2 = y1[:n], y2[:n]

    # Step 1: OLS regression y1 = a + beta * y2
    X = np.column_stack([np.ones(n), y2])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y1, rcond=None)
    except np.linalg.LinAlgError:
        return False, 0.0, 1.0, 0.0, float("inf")

    hedge_ratio = float(beta[1])
    residuals = y1 - X @ beta

    # Step 2: ADF on residuals
    adf_stat, p_val = adf_test(residuals)

    # Half-life of mean reversion
    half_life = _estimate_half_life(residuals)

    return p_val < p_threshold, adf_stat, p_val, hedge_ratio, half_life


def _estimate_half_life(residuals: np.ndarray) -> float:
    """Estimate mean-reversion half-life via AR(1) regression."""
    y = residuals[1:]
    y_lag = residuals[:-1]
    if len(y) < 5:
        return float("inf")

    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")

    phi = beta[1]
    if phi >= 1.0 or phi <= 0:
        return float("inf")
    return float(-math.log(2) / math.log(phi))


# ── Johansen trace test (simplified) ─────────────────────────────────────


def johansen_trace_test(
    y1: np.ndarray,
    y2: np.ndarray,
    p_threshold: float = 0.05,
) -> Tuple[bool, float, float, float]:
    """Simplified Johansen trace test for 2 variables.

    Uses VAR(1) residuals and canonical correlation approach.
    Returns: (cointegrated, trace_stat, approx_p_value, hedge_ratio)
    """
    n = min(len(y1), len(y2))
    if n < 30:
        return False, 0.0, 1.0, 0.0

    y1, y2 = y1[:n], y2[:n]
    Y = np.column_stack([y1, y2])
    dY = np.diff(Y, axis=0)
    Y_lag = Y[:-1]

    # Regress dY and Y_lag on constant, get residuals
    n_obs = len(dY)
    ones = np.ones((n_obs, 1))

    try:
        # Residuals of dY on constant
        b0, _, _, _ = np.linalg.lstsq(ones, dY, rcond=None)
        R0 = dY - ones @ b0

        # Residuals of Y_lag on constant
        b1, _, _, _ = np.linalg.lstsq(ones, Y_lag, rcond=None)
        R1 = Y_lag - ones @ b1
    except np.linalg.LinAlgError:
        return False, 0.0, 1.0, 0.0

    # Cross-product matrices
    S00 = R0.T @ R0 / n_obs
    S01 = R0.T @ R1 / n_obs
    S10 = R1.T @ R0 / n_obs
    S11 = R1.T @ R1 / n_obs

    try:
        S11_inv = np.linalg.inv(S11)
        S00_inv = np.linalg.inv(S00)
    except np.linalg.LinAlgError:
        return False, 0.0, 1.0, 0.0

    M = S11_inv @ S10 @ S00_inv @ S01
    eigenvalues = np.sort(np.real(np.linalg.eigvals(M)))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)

    # Trace statistic: -n * sum(ln(1 - lambda_i)) for i > r
    trace_stat = float(-n_obs * np.sum(np.log(np.maximum(1.0 - eigenvalues, 1e-15))))

    # Critical values for 2 variables (approx: 5% = 15.41 for r=0)
    critical_5pct = 15.41
    cointegrated = trace_stat > critical_5pct
    p_approx = 0.01 if trace_stat > 20.0 else (0.03 if trace_stat > critical_5pct else 0.10)

    # Hedge ratio from eigenvector
    try:
        _, vecs = np.linalg.eig(M)
        hedge = float(vecs[1, 0] / vecs[0, 0]) if abs(vecs[0, 0]) > 1e-12 else 0.0
    except Exception:
        hedge = 0.0

    return cointegrated, trace_stat, p_approx, hedge


# ── Lead-lag detection ───────────────────────────────────────────────────


def detect_lead_lag(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    max_lag: int = 10,
) -> Tuple[int, float]:
    """Find optimal lag via cross-correlation.

    Positive lag = a leads b.  Returns (optimal_lag, correlation_at_lag).
    """
    n = min(len(returns_a), len(returns_b))
    if n < max_lag + 5:
        return 0, 0.0

    ra = returns_a[:n]
    rb = returns_b[:n]

    best_lag = 0
    best_corr = 0.0

    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x = ra[:-lag]
            y = rb[lag:]
        elif lag < 0:
            x = ra[-lag:]
            y = rb[:lag]
        else:
            x = ra
            y = rb

        if len(x) < 5:
            continue

        c = np.corrcoef(x, y)[0, 1]
        if abs(c) > abs(best_corr):
            best_corr = float(c)
            best_lag = lag

    return best_lag, best_corr


def build_lead_lag_matrix(
    returns: pd.DataFrame,
    max_lag: int = 10,
) -> pd.DataFrame:
    """Build NxN matrix of optimal lag correlations."""
    assets = list(returns.columns)
    n = len(assets)
    matrix = pd.DataFrame(0.0, index=assets, columns=assets)

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix.iloc[i, j] = 1.0
                continue
            _, corr = detect_lead_lag(
                returns.iloc[:, i].values,
                returns.iloc[:, j].values,
                max_lag,
            )
            matrix.iloc[i, j] = corr

    return matrix


# ── Spread z-score signals ───────────────────────────────────────────────


def compute_spread_zscore(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    hedge_ratio: float,
    window: int = 60,
) -> Tuple[np.ndarray, float]:
    """Compute z-score of the spread = prices_a - hedge_ratio * prices_b.

    Returns: (zscore_series, current_zscore)
    """
    spread = prices_a - hedge_ratio * prices_b
    n = len(spread)
    if n < window:
        return np.zeros(n), 0.0

    zscores = np.zeros(n)
    for i in range(window, n):
        chunk = spread[i - window : i]
        mu = chunk.mean()
        std = chunk.std()
        if std > 1e-12:
            zscores[i] = (spread[i] - mu) / std

    return zscores, float(zscores[-1]) if n > window else 0.0


def pair_trade_signal(
    z: float,
    entry: float = 2.0,
    exit_thresh: float = 0.5,
) -> str:
    """Generate pair trade signal from z-score."""
    if z > entry:
        return "short_spread"
    elif z < -entry:
        return "long_spread"
    elif abs(z) < exit_thresh:
        return "neutral"
    return "neutral"


# ── Rolling correlation regime ───────────────────────────────────────────


def rolling_correlation_regime(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 60,
    high_threshold: float = 1.5,
    low_threshold: float = -1.5,
) -> Tuple[float, float, float, float, str]:
    """Detect correlation regime from rolling correlation z-score.

    Returns: (current_corr, rolling_mean, rolling_std, z_score, regime)
    """
    rolling_corr = returns_a.rolling(window).corr(returns_b).dropna()
    if len(rolling_corr) < 5:
        return 0.0, 0.0, 0.0, 0.0, "normal"

    current = float(rolling_corr.iloc[-1])
    mu = float(rolling_corr.mean())
    std = float(rolling_corr.std())
    z = (current - mu) / std if std > 1e-12 else 0.0

    if z > high_threshold:
        regime = "high_corr"
    elif z < low_threshold:
        regime = "low_corr"
    else:
        regime = "normal"

    return current, mu, std, z, regime


# ── Momentum spillover ───────────────────────────────────────────────────


def compute_momentum_spillover(
    returns_source: pd.Series,
    returns_target: pd.Series,
    momentum_window: int = 21,
    regression_window: int = 63,
) -> Tuple[float, float, float, str]:
    """Detect momentum spillover from source to target.

    Returns: (source_momentum, spillover_beta, predicted_move, signal)
    """
    if len(returns_source) < regression_window + momentum_window:
        return 0.0, 0.0, 0.0, "neutral"

    # Source momentum
    src_mom = float(returns_source.iloc[-momentum_window:].sum())

    # Regress target returns on lagged source returns (1-day lag)
    src_lag = returns_source.shift(1).dropna()
    aligned = pd.concat(
        [returns_target.rename("tgt"), src_lag.rename("src")], axis=1
    ).dropna()
    train = aligned.iloc[-regression_window:]

    if len(train) < 20:
        return src_mom, 0.0, 0.0, "neutral"

    X = np.column_stack([np.ones(len(train)), train["src"].values])
    y = train["tgt"].values
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return src_mom, 0.0, 0.0, "neutral"

    spillover_beta = float(beta[1])
    predicted = spillover_beta * src_mom

    if predicted > 0.005:
        signal = "bullish"
    elif predicted < -0.005:
        signal = "bearish"
    else:
        signal = "neutral"

    return src_mom, spillover_beta, predicted, signal


# ── Core engine ──────────────────────────────────────────────────────────


class CrossAssetSignalEngine:
    """Cross-asset signal generator."""

    def __init__(
        self,
        prices: pd.DataFrame,
        window: int = 60,
        max_lag: int = 10,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        coint_p_threshold: float = 0.05,
    ):
        if prices.empty:
            raise ValueError("prices DataFrame must not be empty")
        if prices.shape[1] < 2:
            raise ValueError("Need at least 2 assets")

        self.prices = prices.copy()
        self.returns = prices.pct_change().dropna()
        self.asset_names = list(prices.columns)
        self.n_assets = len(self.asset_names)
        self.window = window
        self.max_lag = max_lag
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.coint_p = coint_p_threshold

    def analyze(self) -> AnalysisResult:
        """Run full cross-asset analysis."""
        coints = self._cointegration_tests()
        lead_lags = self._lead_lag_detection()
        spreads = self._spread_signals(coints)
        corr_regimes = self._correlation_regimes()
        spillovers = self._momentum_spillovers()
        ll_matrix = build_lead_lag_matrix(self.returns, self.max_lag)
        corr_matrix = self.returns.corr()

        return AnalysisResult(
            asset_names=self.asset_names,
            cointegrations=coints,
            lead_lags=lead_lags,
            spread_signals=spreads,
            correlation_regimes=corr_regimes,
            momentum_spillovers=spillovers,
            lead_lag_matrix=ll_matrix,
            correlation_matrix=corr_matrix,
            n_assets=self.n_assets,
            n_observations=len(self.prices),
        )

    def _cointegration_tests(self) -> List[CointegrationResult]:
        results: List[CointegrationResult] = []
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                a, b = self.asset_names[i], self.asset_names[j]
                pa = self.prices[a].dropna().values
                pb = self.prices[b].dropna().values
                n = min(len(pa), len(pb))
                pa, pb = pa[:n], pb[:n]

                # Engle-Granger
                coint, adf, pv, hr, hl = engle_granger_test(pa, pb, self.coint_p)
                results.append(CointegrationResult(
                    asset_a=a, asset_b=b, method="engle_granger",
                    cointegrated=coint, test_stat=adf, p_value=pv,
                    hedge_ratio=hr, half_life=hl,
                ))

                # Johansen trace
                jcoint, trace, jpv, jhr = johansen_trace_test(pa, pb, self.coint_p)
                results.append(CointegrationResult(
                    asset_a=a, asset_b=b, method="johansen_trace",
                    cointegrated=jcoint, test_stat=trace, p_value=jpv,
                    hedge_ratio=jhr, half_life=hl,  # reuse EG half-life
                ))

        return results

    def _lead_lag_detection(self) -> List[LeadLagResult]:
        results: List[LeadLagResult] = []
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                a, b = self.asset_names[i], self.asset_names[j]
                ra = self.returns[a].values
                rb = self.returns[b].values
                lag, corr = detect_lead_lag(ra, rb, self.max_lag)
                if abs(corr) < 0.03:
                    continue
                if lag > 0:
                    leader, lagger = a, b
                elif lag < 0:
                    leader, lagger = b, a
                    lag = -lag
                else:
                    leader, lagger = a, b
                results.append(LeadLagResult(
                    leader=leader, lagger=lagger, optimal_lag=lag,
                    correlation_at_lag=corr,
                    direction="positive" if corr > 0 else "negative",
                ))
        return results

    def _spread_signals(
        self, coints: List[CointegrationResult]
    ) -> List[SpreadSignal]:
        results: List[SpreadSignal] = []
        seen = set()
        for c in coints:
            if not c.cointegrated:
                continue
            pair = (c.asset_a, c.asset_b)
            if pair in seen:
                continue
            seen.add(pair)

            pa = self.prices[c.asset_a].dropna().values
            pb = self.prices[c.asset_b].dropna().values
            n = min(len(pa), len(pb))
            _, current_z = compute_spread_zscore(
                pa[:n], pb[:n], c.hedge_ratio, self.window
            )
            sig = pair_trade_signal(current_z, self.zscore_entry, self.zscore_exit)

            results.append(SpreadSignal(
                asset_a=c.asset_a, asset_b=c.asset_b,
                current_z=current_z, signal=sig,
                hedge_ratio=c.hedge_ratio, half_life=c.half_life,
                entry_threshold=self.zscore_entry,
                exit_threshold=self.zscore_exit,
            ))
        return results

    def _correlation_regimes(self) -> List[CorrelationRegime]:
        results: List[CorrelationRegime] = []
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                a, b = self.asset_names[i], self.asset_names[j]
                curr, mu, std, z, regime = rolling_correlation_regime(
                    self.returns[a], self.returns[b], self.window,
                )
                results.append(CorrelationRegime(
                    asset_a=a, asset_b=b, current_corr=curr,
                    rolling_mean=mu, rolling_std=std, z_score=z,
                    regime=regime,
                ))
        return results

    def _momentum_spillovers(self) -> List[MomentumSpillover]:
        results: List[MomentumSpillover] = []
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                if i == j:
                    continue
                src, tgt = self.asset_names[i], self.asset_names[j]
                mom, beta, pred, sig = compute_momentum_spillover(
                    self.returns[src], self.returns[tgt],
                )
                if abs(beta) < 0.01:
                    continue
                results.append(MomentumSpillover(
                    source=src, target=tgt, source_momentum=mom,
                    spillover_beta=beta, predicted_move=pred, signal=sig,
                ))
        return results

    # ── HTML report ──────────────────────────────────────────────────

    @staticmethod
    def generate_report(
        result: AnalysisResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


def _coint_table(coints: List[CointegrationResult]) -> str:
    if not coints:
        return "<p class='meta'>No cointegration results.</p>"
    rows = ""
    for c in coints:
        cls = "coint-yes" if c.cointegrated else ""
        label = "YES" if c.cointegrated else "no"
        rows += f"""<tr class="{cls}">
          <td>{c.asset_a}/{c.asset_b}</td><td>{c.method}</td>
          <td>{label}</td><td>{_fmt(c.test_stat, 2)}</td>
          <td>{_fmt(c.p_value, 3)}</td><td>{_fmt(c.hedge_ratio, 3)}</td>
          <td>{c.half_life:.0f}d</td></tr>"""
    return f"""
    <table class="data-table">
      <tr><th>Pair</th><th>Method</th><th>Coint?</th><th>Stat</th>
          <th>p-value</th><th>Hedge Ratio</th><th>Half-Life</th></tr>
      {rows}
    </table>"""


def _heatmap_svg(matrix: pd.DataFrame, title: str) -> str:
    names = list(matrix.columns)
    n = len(names)
    if n == 0:
        return ""
    cell = 50
    lbl = 70
    w = lbl + n * cell + 10
    h = lbl + n * cell + 10

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(
        f'<text x="{w // 2}" y="15" text-anchor="middle" class="svg-title">{title}</text>'
    )
    for i in range(n):
        for j in range(n):
            val = float(matrix.iloc[i, j])
            intensity = min(abs(val), 1.0)
            if val >= 0:
                r, g, b = int(255 * (1 - intensity * 0.6)), int(255 * (1 - intensity * 0.3)), 255
            else:
                r, g, b = 255, int(255 * (1 - intensity * 0.3)), int(255 * (1 - intensity * 0.6))
            x = lbl + j * cell
            y = 25 + i * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="rgb({r},{g},{b})" stroke="#30363d" stroke-width="0.5"/>'
            )
            parts.append(
                f'<text x="{x + cell // 2}" y="{y + cell // 2 + 4}" text-anchor="middle" '
                f'font-size="9" fill="#000">{val:.2f}</text>'
            )
    for i, nm in enumerate(names):
        parts.append(
            f'<text x="{lbl + i * cell + cell // 2}" y="23" text-anchor="middle" '
            f'font-size="8" fill="#8b949e">{nm[:6]}</text>'
        )
        parts.append(
            f'<text x="{lbl - 4}" y="{25 + i * cell + cell // 2 + 3}" text-anchor="end" '
            f'font-size="8" fill="#8b949e">{nm[:6]}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def _spread_signals_table(spreads: List[SpreadSignal]) -> str:
    if not spreads:
        return "<p class='meta'>No cointegrated pairs for spread signals.</p>"
    rows = ""
    for s in spreads:
        sig_color = "#3fb950" if "long" in s.signal else "#f85149" if "short" in s.signal else "#8b949e"
        rows += f"""<tr>
          <td>{s.asset_a}/{s.asset_b}</td>
          <td>{_fmt(s.current_z, 2)}</td>
          <td style="color:{sig_color}">{s.signal}</td>
          <td>{_fmt(s.hedge_ratio, 3)}</td>
          <td>{s.half_life:.0f}d</td></tr>"""
    return f"""
    <table class="data-table">
      <tr><th>Pair</th><th>Z-Score</th><th>Signal</th><th>Hedge Ratio</th><th>Half-Life</th></tr>
      {rows}
    </table>"""


def _regime_table(regimes: List[CorrelationRegime]) -> str:
    if not regimes:
        return ""
    rows = ""
    for r in regimes:
        color = "#f85149" if r.regime == "high_corr" else "#3fb950" if r.regime == "low_corr" else "#8b949e"
        rows += f"""<tr>
          <td>{r.asset_a}/{r.asset_b}</td>
          <td>{_fmt(r.current_corr, 3)}</td><td>{_fmt(r.z_score, 2)}</td>
          <td style="color:{color}">{r.regime}</td></tr>"""
    return f"""
    <table class="data-table">
      <tr><th>Pair</th><th>Current Corr</th><th>Z-Score</th><th>Regime</th></tr>
      {rows}
    </table>"""


def _build_html(result: AnalysisResult) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Cross-Asset Signal Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  table.data-table th, table.data-table td {{ padding: 6px 10px; text-align: right;
                                               border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  table.data-table td:first-child {{ text-align: left; }}
  table.data-table th:first-child {{ text-align: left; }}
  .coint-yes td {{ color: #3fb950; font-weight: 600; }}
  .chart {{ width: 100%; max-width: 700px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
</style>
</head>
<body>
<h1>Cross-Asset Signal Analysis</h1>
<p class="meta">{result.n_assets} assets &middot; {result.n_observations} observations &middot;
   {len(result.cointegrations)} cointegration tests &middot;
   {len(result.lead_lags)} lead-lag pairs</p>

<h2>Cointegration Tests</h2>
{_coint_table(result.cointegrations)}

<h2>Spread Trading Signals</h2>
{_spread_signals_table(result.spread_signals)}

<h2>Lead-Lag Heatmap</h2>
{_heatmap_svg(result.lead_lag_matrix, "Lead-Lag Cross-Correlation")}

<h2>Correlation Matrix</h2>
{_heatmap_svg(result.correlation_matrix, "Return Correlation Matrix")}

<h2>Correlation Regimes</h2>
{_regime_table(result.correlation_regimes)}

</body>
</html>"""
