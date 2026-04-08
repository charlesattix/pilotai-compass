"""
Portfolio correlation monitor — detects diversification breakdown
and auto-suggests delevering during correlation spikes.

Features:
  1. Multi-window rolling correlations (20/60/120 day)
  2. DCC-GARCH-style dynamic correlation estimation
  3. Alerts when pairwise correlations exceed threshold
  4. Correlation regime classification (normal/elevated/crisis)
  5. Auto-delever sizing based on correlation state
  6. Backtest: delevering on correlation spikes vs static

All methods work on pre-loaded data — no API calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


class CorrRegime(str, Enum):
    NORMAL = "normal"        # avg corr < 0.30
    ELEVATED = "elevated"    # 0.30 - 0.50
    CRISIS = "crisis"        # > 0.50


@dataclass
class CorrSnapshot:
    """Point-in-time correlation state."""
    date: datetime
    avg_corr_20d: float
    avg_corr_60d: float
    avg_corr_120d: float
    max_pairwise: float
    max_pair: str
    regime: CorrRegime
    size_multiplier: float
    n_pairs_above_threshold: int


@dataclass
class CorrAlert:
    date: datetime
    pair: str
    correlation: float
    window: int
    message: str


@dataclass
class DCCEstimate:
    """Dynamic Conditional Correlation estimate."""
    date: datetime
    dcc_matrix: np.ndarray
    avg_dcc: float
    persistence: float      # how sticky is current correlation level


@dataclass
class CorrBacktestResult:
    """Comparison: correlation-aware vs static sizing."""
    static_return: float
    static_sharpe: float
    static_max_dd: float
    adaptive_return: float
    adaptive_sharpe: float
    adaptive_max_dd: float
    dd_reduction: float
    n_delever_days: int
    avg_corr_at_delever: float
    snapshots: List[CorrSnapshot] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_strategy_returns(
    n_days: int = 1512, n_strategies: int = 4, seed: int = 42,
) -> pd.DataFrame:
    """Generate multi-strategy returns with regime-dependent correlations.

    Normal: pairwise corr ~0.15
    Crisis (COVID/2022): pairwise corr spikes to ~0.65
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_days)
    names = [f"strat_{i}" for i in range(n_strategies)]

    # Base independent returns
    indep = rng.normal(0.0004, 0.008, (n_days, n_strategies))

    # Common factor (market)
    common = rng.normal(0, 0.01, n_days)

    # Normal correlation loading: each strategy loads ~0.15 on common
    normal_loading = np.array([0.15, 0.12, 0.18, 0.10][:n_strategies])

    # Crisis correlation loading: spike to ~0.65
    crisis_loading = np.array([0.65, 0.60, 0.70, 0.55][:n_strategies])

    # Crisis periods
    crisis = np.zeros(n_days)
    if n_days > 80:
        crisis[50:80] = 1.0                          # COVID
    if n_days > 650:
        crisis[530:620] = 0.7                         # 2022
    if n_days > 1200:
        crisis[1100:1120] = 0.5                       # mini-crisis

    result = np.zeros((n_days, n_strategies))
    for i in range(n_days):
        loading = normal_loading * (1 - crisis[i]) + crisis_loading * crisis[i]
        for j in range(n_strategies):
            result[i, j] = indep[i, j] + loading[j] * common[i]

    # COVID crash returns
    if n_days > 75:
        result[55:75] += rng.normal(-0.008, 0.015, (20, n_strategies))

    return pd.DataFrame(result, index=idx, columns=names)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class CorrelationMonitor:
    """Portfolio correlation monitor with auto-delevering.

    Args:
        windows: Rolling windows for correlation computation.
        alert_threshold: Pairwise correlation above this triggers alert.
        crisis_threshold: Average correlation above this = crisis regime.
        delever_threshold: Start delevering above this avg correlation.
    """

    def __init__(
        self,
        windows: Tuple[int, ...] = (20, 60, 120),
        alert_threshold: float = 0.50,
        crisis_threshold: float = 0.50,
        delever_threshold: float = 0.35,
    ) -> None:
        self.windows = windows
        self.alert_threshold = alert_threshold
        self.crisis_threshold = crisis_threshold
        self.delever_threshold = delever_threshold

    # ------------------------------------------------------------------
    # Rolling correlations
    # ------------------------------------------------------------------

    @staticmethod
    def rolling_corr_matrix(
        returns: pd.DataFrame, window: int,
    ) -> List[Tuple[datetime, pd.DataFrame]]:
        """Compute rolling correlation matrices."""
        results = []
        for end in range(window, len(returns) + 1):
            chunk = returns.iloc[end - window:end]
            corr = chunk.corr()
            results.append((chunk.index[-1], corr))
        return results

    @staticmethod
    def avg_pairwise_corr(corr_matrix: pd.DataFrame) -> float:
        """Average of upper-triangle pairwise correlations."""
        n = len(corr_matrix)
        if n < 2:
            return 0.0
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                vals.append(corr_matrix.iloc[i, j])
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def max_pairwise(corr_matrix: pd.DataFrame) -> Tuple[float, str]:
        """Maximum pairwise correlation and the pair."""
        n = len(corr_matrix)
        if n < 2:
            return 0.0, ""
        best_val = -1.0
        best_pair = ""
        cols = corr_matrix.columns
        for i in range(n):
            for j in range(i + 1, n):
                v = corr_matrix.iloc[i, j]
                if v > best_val:
                    best_val = v
                    best_pair = f"{cols[i]}×{cols[j]}"
        return float(best_val), best_pair

    # ------------------------------------------------------------------
    # DCC-GARCH style dynamic correlations
    # ------------------------------------------------------------------

    def dcc_estimate(
        self, returns: pd.DataFrame, decay: float = 0.94,
    ) -> List[DCCEstimate]:
        """Exponentially-weighted dynamic correlation (DCC-like).

        Q_t = (1-decay) × eps_t × eps_t' + decay × Q_{t-1}
        R_t = diag(Q_t)^{-1/2} × Q_t × diag(Q_t)^{-1/2}
        """
        X = returns.values
        N, K = X.shape
        if N < 10 or K < 2:
            return []

        # Standardise
        mu = X.mean(0)
        std = X.std(0)
        std[std < 1e-8] = 1e-8
        eps = (X - mu) / std

        Q = np.corrcoef(eps[:20], rowvar=False) if N >= 20 else np.eye(K)
        results = []

        for t in range(max(20, 1), N):
            outer = np.outer(eps[t], eps[t])
            Q = (1 - decay) * outer + decay * Q

            # Normalise to correlation
            d = np.sqrt(np.diag(Q))
            d[d < 1e-8] = 1e-8
            R = Q / np.outer(d, d)
            np.fill_diagonal(R, 1.0)
            R = np.clip(R, -1, 1)

            # Average off-diagonal
            upper = [R[i, j] for i in range(K) for j in range(i + 1, K)]
            avg = float(np.mean(upper)) if upper else 0.0

            persistence = decay  # in simple EWMA, persistence = decay parameter

            results.append(DCCEstimate(
                date=returns.index[t], dcc_matrix=R.copy(),
                avg_dcc=avg, persistence=persistence,
            ))

        return results

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def classify_regime(self, avg_corr: float) -> CorrRegime:
        if avg_corr >= self.crisis_threshold:
            return CorrRegime.CRISIS
        if avg_corr >= self.delever_threshold:
            return CorrRegime.ELEVATED
        return CorrRegime.NORMAL

    def size_multiplier(self, avg_corr: float) -> float:
        """Position size adjustment based on correlation level."""
        if avg_corr < self.delever_threshold:
            return 1.0
        if avg_corr >= self.crisis_threshold:
            return 0.3
        # Linear interpolation between threshold and crisis
        frac = (avg_corr - self.delever_threshold) / (self.crisis_threshold - self.delever_threshold)
        return max(0.3, 1.0 - frac * 0.7)

    # ------------------------------------------------------------------
    # Full monitoring
    # ------------------------------------------------------------------

    def monitor(
        self, returns: pd.DataFrame,
    ) -> List[CorrSnapshot]:
        """Run full correlation monitoring."""
        snapshots: List[CorrSnapshot] = []
        min_window = min(self.windows)

        for end in range(min_window, len(returns) + 1):
            dt = returns.index[end - 1]
            corrs = {}
            for w in self.windows:
                if end >= w:
                    chunk = returns.iloc[end - w:end]
                    corr = chunk.corr()
                    corrs[w] = self.avg_pairwise_corr(corr)
                else:
                    corrs[w] = 0.0

            # Use shortest window for reactivity
            short_w = min(self.windows)
            if end >= short_w:
                short_corr = returns.iloc[end - short_w:end].corr()
                max_val, max_pair = self.max_pairwise(short_corr)
                n_above = sum(1 for i in range(len(short_corr))
                               for j in range(i + 1, len(short_corr))
                               if short_corr.iloc[i, j] > self.alert_threshold)
            else:
                max_val, max_pair = 0.0, ""
                n_above = 0

            avg_short = corrs.get(self.windows[0], 0)
            regime = self.classify_regime(avg_short)
            size_mult = self.size_multiplier(avg_short)

            snapshots.append(CorrSnapshot(
                date=dt,
                avg_corr_20d=corrs.get(20, corrs.get(self.windows[0], 0)),
                avg_corr_60d=corrs.get(60, corrs.get(self.windows[1] if len(self.windows) > 1 else self.windows[0], 0)),
                avg_corr_120d=corrs.get(120, corrs.get(self.windows[-1], 0)),
                max_pairwise=max_val,
                max_pair=max_pair,
                regime=regime,
                size_multiplier=size_mult,
                n_pairs_above_threshold=n_above,
            ))

        return snapshots

    def generate_alerts(
        self, returns: pd.DataFrame,
    ) -> List[CorrAlert]:
        """Generate alerts for correlation threshold breaches."""
        alerts: List[CorrAlert] = []
        w = min(self.windows)
        cols = returns.columns

        for end in range(w, len(returns) + 1):
            dt = returns.index[end - 1]
            chunk = returns.iloc[end - w:end]
            corr = chunk.corr()
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    v = corr.iloc[i, j]
                    if v > self.alert_threshold:
                        alerts.append(CorrAlert(
                            dt, f"{cols[i]}×{cols[j]}", float(v), w,
                            f"Correlation {v:.2f} > {self.alert_threshold:.2f}"))
        return alerts

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def backtest(
        self, returns: pd.DataFrame,
    ) -> CorrBacktestResult:
        """Compare correlation-aware delevering vs static sizing."""
        snapshots = self.monitor(returns)
        n = len(returns)
        cols = returns.columns
        k = len(cols)

        static_rets = np.zeros(n)
        adaptive_rets = np.zeros(n)
        n_delever = 0
        delever_corrs: List[float] = []

        snap_map = {s.date: s for s in snapshots}

        for i in range(n):
            dt = returns.index[i]
            day_ret = returns.iloc[i].mean()  # equal-weight portfolio

            static_rets[i] = day_ret

            snap = snap_map.get(dt)
            if snap:
                adaptive_rets[i] = day_ret * snap.size_multiplier
                if snap.size_multiplier < 1.0:
                    n_delever += 1
                    delever_corrs.append(snap.avg_corr_20d)
            else:
                adaptive_rets[i] = day_ret

        def _metrics(r):
            eq = np.cumprod(1 + r)
            total = float(eq[-1] - 1)
            n_yr = len(r) / TRADING_DAYS
            annual = (1 + total) ** (1 / max(n_yr, 0.01)) - 1
            mu, std = float(r.mean()), float(r.std())
            sharpe = mu / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0
            dd = float((1 - eq / np.maximum.accumulate(eq)).max())
            return annual, sharpe, dd

        s_ret, s_sh, s_dd = _metrics(static_rets)
        a_ret, a_sh, a_dd = _metrics(adaptive_rets)

        return CorrBacktestResult(
            static_return=s_ret, static_sharpe=s_sh, static_max_dd=s_dd,
            adaptive_return=a_ret, adaptive_sharpe=a_sh, adaptive_max_dd=a_dd,
            dd_reduction=s_dd - a_dd,
            n_delever_days=n_delever,
            avg_corr_at_delever=float(np.mean(delever_corrs)) if delever_corrs else 0,
            snapshots=snapshots,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self, result: CorrBacktestResult,
        output_path: str = "reports/correlation_monitor.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Correlation timeline SVG
        corr_svg = ""
        if result.snapshots and len(result.snapshots) > 10:
            vals = [s.avg_corr_20d for s in result.snapshots]
            n = len(vals)
            w, h = 750, 200
            pad = 50
            pw, ph = w - 2 * pad, h - 60
            def tx(i): return pad + i / max(n - 1, 1) * pw
            def ty(v): return 28 + (1 - v) * ph  # 0..1 → bottom..top
            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                      f'style="background:#fff;border:1px solid #e2e8f0;border-radius:6px;margin:.5rem 0">']
            parts.append(f'<text x="{w // 2}" y="16" text-anchor="middle" font-size="12" '
                          f'font-weight="bold" fill="#0f172a">Rolling 20d Avg Pairwise Correlation</text>')
            # Threshold lines
            thy = ty(self.alert_threshold)
            parts.append(f'<line x1="{pad}" y1="{thy:.0f}" x2="{w - pad}" y2="{thy:.0f}" '
                          f'stroke="#dc2626" stroke-dasharray="4,3"/>')
            parts.append(f'<text x="{w - pad + 3}" y="{thy + 4:.0f}" font-size="8" fill="#dc2626">{self.alert_threshold}</text>')
            dly = ty(self.delever_threshold)
            parts.append(f'<line x1="{pad}" y1="{dly:.0f}" x2="{w - pad}" y2="{dly:.0f}" '
                          f'stroke="#d97706" stroke-dasharray="4,3"/>')
            d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(vals[i]):.1f}" for i in range(n))
            parts.append(f'<path d="{d}" fill="none" stroke="#2563eb" stroke-width="1.5"/>')
            parts.append("</svg>")
            corr_svg = "\n".join(parts)

        # Regime distribution
        regimes = {}
        for s in result.snapshots:
            regimes[s.regime.value] = regimes.get(s.regime.value, 0) + 1

        r = result
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Correlation Monitor</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fff; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; }}
h2 {{ color: #334155; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; border-bottom: 2px solid #e2e8f0; }}
th:first-child {{ text-align: left; }}
td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #f1f5f9; }}
td:first-child {{ text-align: left; font-weight: 500; }}
.card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }}
.green {{ color: #059669; font-weight: 700; }}
</style></head><body>
<h1>EXP-1410-max: Portfolio Correlation Monitor</h1>
<div class="card">
<p><strong>DD Reduction:</strong> <span class="green">{r.dd_reduction:.1%}</span> |
<strong>Delever Days:</strong> {r.n_delever_days} |
<strong>Avg Corr at Delever:</strong> {r.avg_corr_at_delever:.2f}</p>
<p>Regime Distribution: {' | '.join(f'{k}: {v}d' for k, v in regimes.items())}</p>
</div>

{corr_svg}

<h2>Static vs Correlation-Aware Sizing</h2>
<table>
<tr><th>Metric</th><th>Static</th><th>Adaptive</th><th>Improvement</th></tr>
<tr><td>Annual Return</td><td>{r.static_return:.2%}</td><td>{r.adaptive_return:.2%}</td>
<td>{r.adaptive_return - r.static_return:+.2%}</td></tr>
<tr><td>Sharpe</td><td>{r.static_sharpe:.2f}</td><td>{r.adaptive_sharpe:.2f}</td>
<td>{r.adaptive_sharpe - r.static_sharpe:+.2f}</td></tr>
<tr><td>Max Drawdown</td><td>{r.static_max_dd:.2%}</td><td>{r.adaptive_max_dd:.2%}</td>
<td class="green">{r.dd_reduction:+.2%}</td></tr>
</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)
