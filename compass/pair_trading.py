"""
Statistical pair trading engine.

Components:
  - Cointegration testing: Engle-Granger (ADF on residuals) + Johansen (trace)
  - Z-score spread trading with configurable entry/exit thresholds
  - Dynamic hedge ratio estimation via Kalman filter
  - Half-life of mean reversion estimation
  - Entry/exit signal generation with position tracking
  - Pair selection from a universe (rank by cointegration strength)
  - Backtest with realistic slippage + commission costs
  - P&L attribution (spread capture vs cost drag)

HTML report at reports/pair_trading.html with spread chart, z-score,
hedge ratio evolution, P&L curve.

This is READ-ONLY simulation.  No broker connections, no trade placement.

Usage::

    from compass.pair_trading import PairTradingEngine
    engine = PairTradingEngine(prices_a, prices_b)
    result = engine.run()
    PairTradingEngine.generate_report(result)
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
DEFAULT_OUTPUT = ROOT / "reports" / "pair_trading.html"


# ── ADF test ─────────────────────────────────────────────────────────────


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
    adf = float(b / (se / denom))
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
    """Cointegration test output."""

    method: str
    cointegrated: bool
    test_stat: float
    p_value: float
    hedge_ratio: float
    half_life: float


@dataclass
class KalmanState:
    """Kalman filter state for dynamic hedge ratio."""

    hedge_ratios: np.ndarray
    intercepts: np.ndarray
    spreads: np.ndarray
    timestamps: np.ndarray


@dataclass
class Signal:
    """Entry/exit signal at a point in time."""

    index: int
    signal_type: str   # "long_spread", "short_spread", "close"
    z_score: float
    spread: float
    hedge_ratio: float


@dataclass
class Trade:
    """A completed pair trade."""

    entry_idx: int
    exit_idx: int
    direction: str     # "long_spread" or "short_spread"
    entry_spread: float
    exit_spread: float
    entry_z: float
    exit_z: float
    hedge_ratio: float
    gross_pnl: float
    slippage: float
    commission: float
    net_pnl: float
    hold_periods: int


@dataclass
class PairCandidate:
    """A pair candidate from universe selection."""

    asset_a: str
    asset_b: str
    coint_pvalue: float
    half_life: float
    hedge_ratio: float
    spread_vol: float
    score: float


@dataclass
class PairTradingResult:
    """Full result from pair trading analysis."""

    asset_a: str
    asset_b: str
    cointegration_eg: CointegrationResult
    cointegration_jh: CointegrationResult
    kalman: KalmanState
    signals: List[Signal]
    trades: List[Trade]
    # Metrics
    total_pnl: float
    total_return_pct: float
    sharpe: float
    win_rate: float
    profit_factor: float
    n_trades: int
    avg_hold: float
    total_slippage: float
    total_commission: float
    max_drawdown_pct: float
    # Series
    spread_series: np.ndarray
    zscore_series: np.ndarray
    hedge_ratio_series: np.ndarray
    pnl_curve: np.ndarray
    n_observations: int


# ── Engle-Granger cointegration ──────────────────────────────────────────


def engle_granger(
    y1: np.ndarray, y2: np.ndarray, p_threshold: float = 0.05,
) -> CointegrationResult:
    """Engle-Granger two-step cointegration test."""
    n = min(len(y1), len(y2))
    if n < 20:
        return CointegrationResult("engle_granger", False, 0.0, 1.0, 0.0, float("inf"))
    y1, y2 = y1[:n], y2[:n]
    X = np.column_stack([np.ones(n), y2])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, y1, rcond=None)
    except np.linalg.LinAlgError:
        return CointegrationResult("engle_granger", False, 0.0, 1.0, 0.0, float("inf"))
    hedge = float(beta[1])
    residuals = y1 - X @ beta
    adf_stat, pv = adf_test(residuals)
    hl = _half_life(residuals)
    return CointegrationResult("engle_granger", pv < p_threshold, adf_stat, pv, hedge, hl)


def _half_life(residuals: np.ndarray) -> float:
    """AR(1) half-life of mean reversion."""
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
    return float(-math.log(2) / math.log(abs(phi)))


# ── Johansen trace test (simplified, 2 variables) ────────────────────────


def johansen_trace(
    y1: np.ndarray, y2: np.ndarray, p_threshold: float = 0.05,
) -> CointegrationResult:
    """Simplified Johansen trace test for 2 variables."""
    n = min(len(y1), len(y2))
    if n < 30:
        return CointegrationResult("johansen_trace", False, 0.0, 1.0, 0.0, float("inf"))
    y1, y2 = y1[:n], y2[:n]
    Y = np.column_stack([y1, y2])
    dY = np.diff(Y, axis=0)
    Y_lag = Y[:-1]
    n_obs = len(dY)
    ones = np.ones((n_obs, 1))
    try:
        b0, _, _, _ = np.linalg.lstsq(ones, dY, rcond=None)
        R0 = dY - ones @ b0
        b1, _, _, _ = np.linalg.lstsq(ones, Y_lag, rcond=None)
        R1 = Y_lag - ones @ b1
    except np.linalg.LinAlgError:
        return CointegrationResult("johansen_trace", False, 0.0, 1.0, 0.0, float("inf"))
    S00 = R0.T @ R0 / n_obs
    S01 = R0.T @ R1 / n_obs
    S10 = R1.T @ R0 / n_obs
    S11 = R1.T @ R1 / n_obs
    try:
        S11_inv = np.linalg.inv(S11)
        S00_inv = np.linalg.inv(S00)
    except np.linalg.LinAlgError:
        return CointegrationResult("johansen_trace", False, 0.0, 1.0, 0.0, float("inf"))
    M = S11_inv @ S10 @ S00_inv @ S01
    eigenvalues = np.sort(np.real(np.linalg.eigvals(M)))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    trace_stat = float(-n_obs * np.sum(np.log(np.maximum(1.0 - eigenvalues, 1e-15))))
    critical = 15.41
    coint = trace_stat > critical
    pv = 0.01 if trace_stat > 20 else (0.03 if coint else 0.10)
    try:
        _, vecs = np.linalg.eig(M)
        hedge = float(vecs[1, 0] / vecs[0, 0]) if abs(vecs[0, 0]) > 1e-12 else 0.0
    except Exception:
        hedge = 0.0
    # Use EG half-life
    residuals = y1 - hedge * y2
    hl = _half_life(residuals)
    return CointegrationResult("johansen_trace", coint, trace_stat, pv, hedge, hl)


# ── Kalman filter for dynamic hedge ratio ────────────────────────────────


def kalman_hedge_ratio(
    y1: np.ndarray,
    y2: np.ndarray,
    delta: float = 1e-4,
    ve: float = 1e-3,
) -> KalmanState:
    """Online Kalman filter estimating dynamic hedge ratio and intercept.

    State = [intercept, hedge_ratio].
    Observation: y1[t] = intercept + hedge_ratio * y2[t] + noise.

    Args:
        delta: state transition covariance scale.
        ve: observation noise variance.
    """
    n = min(len(y1), len(y2))
    y1, y2 = y1[:n].astype(float), y2[:n].astype(float)

    # State: [intercept, hedge]
    theta = np.zeros(2)
    P = np.eye(2) * 1.0
    Q = np.eye(2) * delta
    R = ve

    hedge_ratios = np.zeros(n)
    intercepts = np.zeros(n)
    spreads = np.zeros(n)

    for t in range(n):
        x = np.array([1.0, y2[t]])

        # Predict
        P = P + Q

        # Update
        y_hat = x @ theta
        e = y1[t] - y_hat
        S = x @ P @ x + R
        K = P @ x / S
        theta = theta + K * e
        P = P - np.outer(K, x) @ P

        intercepts[t] = theta[0]
        hedge_ratios[t] = theta[1]
        spreads[t] = e

    return KalmanState(
        hedge_ratios=hedge_ratios,
        intercepts=intercepts,
        spreads=spreads,
        timestamps=np.arange(n),
    )


# ── Z-score computation ──────────────────────────────────────────────────


def compute_zscore(spread: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling z-score of spread."""
    n = len(spread)
    z = np.zeros(n)
    for i in range(window, n):
        chunk = spread[i - window:i]
        mu = chunk.mean()
        std = chunk.std()
        if std > 1e-12:
            z[i] = (spread[i] - mu) / std
    return z


# ── Signal generation ────────────────────────────────────────────────────


def generate_signals(
    zscore: np.ndarray,
    spread: np.ndarray,
    hedge_ratios: np.ndarray,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> List[Signal]:
    """Generate entry/exit signals from z-score series."""
    signals: List[Signal] = []
    position = 0  # 0=flat, 1=long_spread, -1=short_spread

    for i in range(len(zscore)):
        z = zscore[i]
        hr = hedge_ratios[i] if i < len(hedge_ratios) else hedge_ratios[-1]

        if position == 0:
            if z < -entry_z:
                position = 1
                signals.append(Signal(i, "long_spread", z, spread[i], hr))
            elif z > entry_z:
                position = -1
                signals.append(Signal(i, "short_spread", z, spread[i], hr))
        elif position == 1:
            if z > -exit_z or z > stop_z or z < -stop_z:
                position = 0
                signals.append(Signal(i, "close", z, spread[i], hr))
        elif position == -1:
            if z < exit_z or z < -stop_z or z > stop_z:
                position = 0
                signals.append(Signal(i, "close", z, spread[i], hr))

    return signals


# ── Backtest ─────────────────────────────────────────────────────────────


def backtest_signals(
    signals: List[Signal],
    spread: np.ndarray,
    capital: float = 100_000.0,
    slippage_bps: float = 5.0,
    commission: float = 2.60,
) -> List[Trade]:
    """Convert signals into closed trades with costs."""
    trades: List[Trade] = []
    pending: Optional[Signal] = None

    for sig in signals:
        if sig.signal_type in ("long_spread", "short_spread"):
            pending = sig
        elif sig.signal_type == "close" and pending is not None:
            entry_s = pending.spread
            exit_s = sig.spread

            if pending.signal_type == "long_spread":
                gross = (exit_s - entry_s) * 100
            else:
                gross = (entry_s - exit_s) * 100

            slip = abs(entry_s + exit_s) * slippage_bps / 10_000 * 100
            net = gross - slip - commission

            trades.append(Trade(
                entry_idx=pending.index, exit_idx=sig.index,
                direction=pending.signal_type,
                entry_spread=entry_s, exit_spread=exit_s,
                entry_z=pending.z_score, exit_z=sig.z_score,
                hedge_ratio=pending.hedge_ratio,
                gross_pnl=gross, slippage=slip, commission=commission,
                net_pnl=net, hold_periods=sig.index - pending.index,
            ))
            pending = None

    return trades


# ── Pair selection from universe ─────────────────────────────────────────


def select_pairs(
    prices: pd.DataFrame,
    max_pairs: int = 10,
    p_threshold: float = 0.05,
    max_half_life: float = 60.0,
) -> List[PairCandidate]:
    """Select best cointegrated pairs from a universe of assets."""
    assets = list(prices.columns)
    candidates: List[PairCandidate] = []

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i], assets[j]
            pa = prices[a].dropna().values
            pb = prices[b].dropna().values
            n = min(len(pa), len(pb))
            if n < 60:
                continue

            result = engle_granger(pa[:n], pb[:n], p_threshold)
            if result.p_value >= p_threshold:
                continue
            if result.half_life > max_half_life or result.half_life <= 0:
                continue

            residuals = pa[:n] - result.hedge_ratio * pb[:n]
            spread_vol = float(np.std(residuals))

            # Score: lower p-value and shorter half-life are better
            score = (1.0 - result.p_value) / max(result.half_life, 1.0)

            candidates.append(PairCandidate(
                asset_a=a, asset_b=b,
                coint_pvalue=result.p_value,
                half_life=result.half_life,
                hedge_ratio=result.hedge_ratio,
                spread_vol=spread_vol,
                score=score,
            ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_pairs]


# ── Metrics ──────────────────────────────────────────────────────────────


def _sharpe(pnls: np.ndarray) -> float:
    if len(pnls) < 2:
        return 0.0
    mu, std = pnls.mean(), pnls.std(ddof=1)
    return float(mu / std * math.sqrt(252)) if std > 1e-12 else 0.0


def _profit_factor(pnls: np.ndarray) -> float:
    gains = pnls[pnls > 0].sum()
    losses = abs(pnls[pnls < 0].sum())
    if losses < 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _max_dd_pct(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1)
    return float(abs(dd.min()))


# ── Core engine ──────────────────────────────────────────────────────────


class PairTradingEngine:
    """Statistical pair trading engine.

    Args:
        prices_a: Price series for asset A.
        prices_b: Price series for asset B.
        name_a: Asset A name.
        name_b: Asset B name.
        zscore_window: Rolling window for z-score.
        entry_z: Z-score threshold for entry.
        exit_z: Z-score threshold for exit.
        stop_z: Z-score stop-loss threshold.
        slippage_bps: Slippage in basis points.
        commission: Round-trip commission per trade.
        capital: Starting capital.
    """

    def __init__(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        name_a: str = "A",
        name_b: str = "B",
        zscore_window: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        slippage_bps: float = 5.0,
        commission: float = 2.60,
        capital: float = 100_000.0,
    ):
        if len(prices_a) < 30 or len(prices_b) < 30:
            raise ValueError("Need at least 30 price observations per asset")
        self.pa = prices_a.values.astype(float)
        self.pb = prices_b.values.astype(float)
        n = min(len(self.pa), len(self.pb))
        self.pa, self.pb = self.pa[:n], self.pb[:n]
        self.name_a = name_a
        self.name_b = name_b
        self.zscore_window = zscore_window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.slippage_bps = slippage_bps
        self.commission = commission
        self.capital = capital

    def run(self) -> PairTradingResult:
        """Run full pair trading analysis and backtest."""
        n = len(self.pa)

        # Cointegration tests
        eg = engle_granger(self.pa, self.pb)
        jh = johansen_trace(self.pa, self.pb)

        # Kalman filter
        kalman = kalman_hedge_ratio(self.pa, self.pb)

        # Spread and z-score using Kalman hedge ratio
        spread = kalman.spreads
        zscore = compute_zscore(spread, self.zscore_window)

        # Signals
        signals = generate_signals(
            zscore, spread, kalman.hedge_ratios,
            self.entry_z, self.exit_z, self.stop_z,
        )

        # Backtest
        trades = backtest_signals(
            signals, spread, self.capital, self.slippage_bps, self.commission,
        )

        # Metrics
        pnls = np.array([t.net_pnl for t in trades]) if trades else np.array([0.0])
        equity = self.capital + np.cumsum(pnls)
        n_trades = len(trades)
        wins = sum(1 for t in trades if t.net_pnl > 0)
        total_pnl = float(pnls.sum())

        return PairTradingResult(
            asset_a=self.name_a, asset_b=self.name_b,
            cointegration_eg=eg, cointegration_jh=jh,
            kalman=kalman, signals=signals, trades=trades,
            total_pnl=total_pnl,
            total_return_pct=total_pnl / self.capital * 100,
            sharpe=_sharpe(pnls),
            win_rate=wins / n_trades if n_trades > 0 else 0.0,
            profit_factor=_profit_factor(pnls),
            n_trades=n_trades,
            avg_hold=float(np.mean([t.hold_periods for t in trades])) if trades else 0.0,
            total_slippage=float(sum(t.slippage for t in trades)),
            total_commission=float(sum(t.commission for t in trades)),
            max_drawdown_pct=_max_dd_pct(equity),
            spread_series=spread,
            zscore_series=zscore,
            hedge_ratio_series=kalman.hedge_ratios,
            pnl_curve=equity,
            n_observations=n,
        )

    @staticmethod
    def generate_report(
        result: PairTradingResult,
        output_path: Path = DEFAULT_OUTPUT,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        html = _build_html(result)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Report written to %s", output_path)
        return output_path


# ── HTML generation ──────────────────────────────────────────────────────


def _f(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}"


def _fd(v: float) -> str:
    return f"${v:,.2f}"


def _fp(v: float) -> str:
    return f"{v:.1f}%"


def _svg_line(values, title, color="#58a6ff", w=700, h=200, threshold=None):
    vals = list(values) if not isinstance(values, list) else values
    if len(vals) < 2:
        return ""
    n = len(vals)
    pad = 55
    pw = w - 2 * pad
    ph = h - 65
    y_min = min(vals)
    y_max = max(vals)
    if threshold is not None:
        y_min = min(y_min, -abs(threshold))
        y_max = max(y_max, abs(threshold))
    if y_max <= y_min:
        y_max = y_min + 0.01

    def tx(i): return pad + i / max(n - 1, 1) * pw
    def ty(v): return 35 + (1 - (v - y_min) / (y_max - y_min)) * ph

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    parts.append(f'<text x="{w // 2}" y="20" text-anchor="middle" class="svg-title">{title}</text>')
    if y_min < 0 < y_max:
        zy = ty(0)
        parts.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" stroke="#30363d" stroke-dasharray="3,3"/>')
    if threshold is not None:
        for th in [threshold, -threshold]:
            thy = ty(th)
            parts.append(f'<line x1="{pad}" y1="{thy:.0f}" x2="{w - pad}" y2="{thy:.0f}" stroke="#f85149" stroke-dasharray="4,3" stroke-width="1"/>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(vals[i]):.1f}" for i in range(n))
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def _trade_table(trades: List[Trade]) -> str:
    if not trades:
        return "<p class='meta'>No trades executed.</p>"
    rows = ""
    for t in trades[:50]:
        color = "#3fb950" if t.net_pnl > 0 else "#f85149"
        rows += f"<tr><td>{t.entry_idx}</td><td>{t.exit_idx}</td><td>{t.direction}</td><td>{_f(t.entry_z)}</td><td>{_f(t.exit_z)}</td><td>{_f(t.hedge_ratio, 3)}</td><td style='color:{color}'>{_fd(t.net_pnl)}</td><td>{t.hold_periods}</td></tr>"
    return f"""<table class="data-table"><tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry Z</th><th>Exit Z</th><th>Hedge</th><th>Net PnL</th><th>Hold</th></tr>{rows}</table>"""


def _coint_card(eg: CointegrationResult, jh: CointegrationResult) -> str:
    eg_color = "#3fb950" if eg.cointegrated else "#f85149"
    jh_color = "#3fb950" if jh.cointegrated else "#f85149"
    return f"""
    <div class="card">
      <h3>Cointegration Tests</h3>
      <div class="metrics-grid">
        <div><span class="label">Engle-Granger</span><span class="value" style="color:{eg_color}">{"YES" if eg.cointegrated else "NO"} (p={_f(eg.p_value, 3)})</span></div>
        <div><span class="label">Johansen Trace</span><span class="value" style="color:{jh_color}">{"YES" if jh.cointegrated else "NO"} (stat={_f(jh.test_stat)})</span></div>
        <div><span class="label">EG Hedge Ratio</span><span class="value">{_f(eg.hedge_ratio, 4)}</span></div>
        <div><span class="label">Half-Life</span><span class="value">{_f(eg.half_life, 1)} days</span></div>
      </div>
    </div>"""


def _build_html(result: PairTradingResult) -> str:
    eg = result.cointegration_eg
    jh = result.cointegration_jh
    sharpe_color = "#3fb950" if result.sharpe > 0 else "#f85149"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Pair Trading: {result.asset_a}/{result.asset_b}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #0d1117;
         color: #c9d1d9; }}
  h1, h2, h3 {{ color: #58a6ff; }}
  .meta {{ color: #8b949e; margin-bottom: 20px; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
              gap: 10px; margin: 20px 0; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 10px; text-align: center; }}
  .stat .label {{ color: #8b949e; font-size: 0.8em; }}
  .stat .value {{ color: #f0f6fc; font-weight: 600; font-size: 1.1em; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
           padding: 16px; margin: 16px 0; }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
  .metrics-grid .label {{ color: #8b949e; font-size: 0.85em; }}
  .metrics-grid .value {{ color: #f0f6fc; font-weight: 600; }}
  table.data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }}
  table.data-table th, table.data-table td {{ padding: 5px 8px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.data-table th {{ color: #8b949e; background: #161b22; }}
  .chart {{ width: 100%; max-width: 750px; margin: 16px auto; display: block; }}
  .svg-title {{ fill: #58a6ff; font-size: 13px; }}
</style>
</head>
<body>
<h1>Pair Trading: {result.asset_a} / {result.asset_b}</h1>
<p class="meta">{result.n_observations} observations &middot; {result.n_trades} trades &middot;
   Half-life: {_f(eg.half_life, 1)} days</p>

<div class="summary">
  <div class="stat"><div class="label">Total PnL</div><div class="value">{_fd(result.total_pnl)}</div></div>
  <div class="stat"><div class="label">Return</div><div class="value">{_fp(result.total_return_pct)}</div></div>
  <div class="stat"><div class="label">Sharpe</div><div class="value" style="color:{sharpe_color}">{_f(result.sharpe)}</div></div>
  <div class="stat"><div class="label">Win Rate</div><div class="value">{_fp(result.win_rate * 100)}</div></div>
  <div class="stat"><div class="label">Profit Factor</div><div class="value">{_f(result.profit_factor)}</div></div>
  <div class="stat"><div class="label">Max DD</div><div class="value">{_fp(result.max_drawdown_pct * 100)}</div></div>
  <div class="stat"><div class="label">Avg Hold</div><div class="value">{_f(result.avg_hold, 1)}d</div></div>
  <div class="stat"><div class="label">Slippage</div><div class="value">{_fd(result.total_slippage)}</div></div>
</div>

{_coint_card(eg, jh)}

<h2>Spread</h2>
{_svg_line(result.spread_series.tolist(), "Spread (Kalman Residual)", "#d29922")}

<h2>Z-Score</h2>
{_svg_line(result.zscore_series.tolist(), "Spread Z-Score", "#58a6ff", threshold=result.cointegration_eg.half_life if False else 2.0)}

<h2>Hedge Ratio Evolution</h2>
{_svg_line(result.hedge_ratio_series.tolist(), "Kalman Hedge Ratio", "#3fb950")}

<h2>Equity Curve</h2>
{_svg_line(result.pnl_curve.tolist(), "Equity ($)", "#3fb950")}

<h2>Trades</h2>
{_trade_table(result.trades)}

</body>
</html>"""
