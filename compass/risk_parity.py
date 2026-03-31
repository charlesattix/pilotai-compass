"""
Risk parity portfolio optimizer — dedicated module.

Methods:
  1. Equal Risk Contribution (ERC) — true risk-budget optimization
  2. Hierarchical Risk Parity (HRP) — dendrogram-based bisection
  3. Inverse-volatility weighting
  4. Maximum diversification ratio
  5. Minimum variance

Regime-conditional risk parity: selects a different method per regime.
Backtest harness: run all methods vs equal-weight benchmark and compare.

All methods work on pre-loaded return data — no network calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Enums & dataclasses
# ---------------------------------------------------------------------------

class RPMethod(str, Enum):
    ERC = "erc"
    HRP = "hrp"
    INVERSE_VOL = "inverse_vol"
    MAX_DIV = "max_div"
    MIN_VAR = "min_var"


@dataclass
class RPWeights:
    """Risk parity result."""
    weights: Dict[str, float]
    method: str
    expected_return: float = 0.0
    expected_vol: float = 0.0
    sharpe: float = 0.0
    diversification_ratio: float = 0.0


@dataclass
class RiskContribution:
    asset: str
    weight: float
    risk_contrib: float
    pct_of_total: float


@dataclass
class BacktestRow:
    date: datetime
    method: str
    daily_return: float
    cumulative: float


@dataclass
class MethodComparison:
    method: str
    total_return: float
    annual_return: float
    sharpe: float
    max_drawdown: float
    volatility: float


# ---------------------------------------------------------------------------
# Core optimizers
# ---------------------------------------------------------------------------

class RiskParityOptimizer:
    """Risk parity portfolio optimizer.

    Args:
        risk_free_rate: Annualised rate for Sharpe computation.
    """

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # 1. Equal Risk Contribution (ERC)
    # ------------------------------------------------------------------

    def erc(
        self, returns: pd.DataFrame, budget: Optional[Dict[str, float]] = None,
    ) -> RPWeights:
        """True ERC via optimization: each asset contributes equally to risk.

        Minimizes: sum_i (RC_i - target_i)^2
        where RC_i = w_i * (Cov @ w)_i / sigma_p
        """
        assets = returns.columns.tolist()
        n = len(assets)
        if n == 0:
            return RPWeights(weights={}, method="erc")

        cov = returns.cov().values * TRADING_DAYS
        target_rc = np.ones(n) / n
        if budget:
            target_rc = np.array([budget.get(a, 1.0 / n) for a in assets])
            target_rc /= target_rc.sum()

        def objective(w):
            w = np.maximum(w, 1e-8)
            port_var = w @ cov @ w
            port_vol = np.sqrt(port_var) if port_var > 0 else 1e-8
            rc = w * (cov @ w) / port_vol
            rc_pct = rc / rc.sum() if rc.sum() > 0 else rc
            return float(((rc_pct - target_rc) ** 2).sum())

        x0 = np.ones(n) / n
        bounds = [(0.01, 0.60)] * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 500})

        w = res.x if res.success else x0
        w = np.maximum(w, 0)
        w /= w.sum()

        return self._build_result(assets, w, cov, returns, "erc")

    # ------------------------------------------------------------------
    # 2. HRP
    # ------------------------------------------------------------------

    def hrp(self, returns: pd.DataFrame) -> RPWeights:
        """Hierarchical Risk Parity via dendrogram bisection."""
        assets = returns.columns.tolist()
        n = len(assets)
        if n <= 1:
            w = {a: 1.0 for a in assets}
            return RPWeights(weights=w, method="hrp")

        corr = returns.corr().values
        dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
        order = leaves_list(link).tolist()

        cov = returns.cov().values * TRADING_DAYS
        weights = np.ones(n)

        def _cluster_var(idx):
            sub_cov = cov[np.ix_(idx, idx)]
            inv_d = 1.0 / np.diag(sub_cov).clip(1e-12)
            inv_d /= inv_d.sum()
            return float(inv_d @ sub_cov @ inv_d)

        def _bisect(items):
            if len(items) <= 1:
                return
            mid = len(items) // 2
            left, right = items[:mid], items[mid:]
            vl = _cluster_var(left)
            vr = _cluster_var(right)
            total = vl + vr
            alpha = 1 - vl / total if total > 0 else 0.5
            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= (1 - alpha)
            _bisect(left)
            _bisect(right)

        _bisect(order)
        weights /= weights.sum()
        return self._build_result(assets, weights, cov, returns, "hrp")

    # ------------------------------------------------------------------
    # 3. Inverse-vol
    # ------------------------------------------------------------------

    def inverse_vol(self, returns: pd.DataFrame) -> RPWeights:
        assets = returns.columns.tolist()
        if not assets:
            return RPWeights(weights={}, method="inverse_vol")
        vols = returns.std().values * np.sqrt(TRADING_DAYS)
        vols = np.maximum(vols, 1e-8)
        w = (1.0 / vols)
        w /= w.sum()
        cov = returns.cov().values * TRADING_DAYS
        return self._build_result(assets, w, cov, returns, "inverse_vol")

    # ------------------------------------------------------------------
    # 4. Max diversification
    # ------------------------------------------------------------------

    def max_diversification(self, returns: pd.DataFrame) -> RPWeights:
        assets = returns.columns.tolist()
        n = len(assets)
        if n == 0:
            return RPWeights(weights={}, method="max_div")
        cov = returns.cov().values * TRADING_DAYS
        vols = np.sqrt(np.diag(cov))

        def neg_dr(w):
            pv = np.sqrt(w @ cov @ w)
            return -(w @ vols) / pv if pv > 1e-12 else 0.0

        x0 = np.ones(n) / n
        bounds = [(0.01, 0.60)] * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = minimize(neg_dr, x0, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 500})
        w = res.x if res.success else x0
        w = np.maximum(w, 0)
        w /= w.sum()
        dr = float((w @ vols) / np.sqrt(w @ cov @ w)) if np.sqrt(w @ cov @ w) > 1e-12 else 1.0
        result = self._build_result(assets, w, cov, returns, "max_div")
        result.diversification_ratio = dr
        return result

    # ------------------------------------------------------------------
    # 5. Minimum variance
    # ------------------------------------------------------------------

    def min_variance(self, returns: pd.DataFrame) -> RPWeights:
        assets = returns.columns.tolist()
        n = len(assets)
        if n == 0:
            return RPWeights(weights={}, method="min_var")
        cov = returns.cov().values * TRADING_DAYS

        def port_var(w):
            return float(w @ cov @ w)

        x0 = np.ones(n) / n
        bounds = [(0.01, 0.60)] * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = minimize(port_var, x0, method="SLSQP", bounds=bounds,
                        constraints=cons, options={"maxiter": 500})
        w = res.x if res.success else x0
        w = np.maximum(w, 0)
        w /= w.sum()
        return self._build_result(assets, w, cov, returns, "min_var")

    # ------------------------------------------------------------------
    # Unified dispatcher
    # ------------------------------------------------------------------

    def optimize(
        self, returns: pd.DataFrame, method: RPMethod = RPMethod.ERC,
    ) -> RPWeights:
        dispatch = {
            RPMethod.ERC: self.erc,
            RPMethod.HRP: self.hrp,
            RPMethod.INVERSE_VOL: self.inverse_vol,
            RPMethod.MAX_DIV: self.max_diversification,
            RPMethod.MIN_VAR: self.min_variance,
        }
        fn = dispatch.get(method, self.erc)
        return fn(returns)

    # ------------------------------------------------------------------
    # Risk contributions
    # ------------------------------------------------------------------

    @staticmethod
    def risk_contributions(
        returns: pd.DataFrame, weights: Dict[str, float],
    ) -> List[RiskContribution]:
        assets = returns.columns.tolist()
        w = np.array([weights.get(a, 0.0) for a in assets])
        cov = returns.cov().values * TRADING_DAYS
        port_vol = np.sqrt(w @ cov @ w) if w @ cov @ w > 0 else 1e-8
        cov_w = cov @ w
        results: List[RiskContribution] = []
        for i, a in enumerate(assets):
            rc = float(w[i] * cov_w[i] / port_vol)
            pct = rc / port_vol if port_vol > 1e-12 else 0.0
            results.append(RiskContribution(a, float(w[i]), rc, pct))
        return results

    # ------------------------------------------------------------------
    # Regime-conditional
    # ------------------------------------------------------------------

    def regime_optimize(
        self,
        returns: pd.DataFrame,
        regimes: pd.Series,
        method_map: Optional[Dict[str, RPMethod]] = None,
    ) -> Dict[str, RPWeights]:
        """Different RP method per regime."""
        default = {
            "bull": RPMethod.ERC, "bear": RPMethod.MIN_VAR,
            "high_vol": RPMethod.INVERSE_VOL, "low_vol": RPMethod.MAX_DIV,
            "crash": RPMethod.MIN_VAR,
        }
        mm = method_map or default
        aligned = returns.copy()
        aligned["_regime"] = regimes
        results: Dict[str, RPWeights] = {}
        for regime, grp in aligned.groupby("_regime"):
            r_str = str(regime)
            method = mm.get(r_str, RPMethod.ERC)
            sub = grp.drop(columns="_regime")
            if len(sub) < 5:
                results[r_str] = self.inverse_vol(sub)
            else:
                results[r_str] = self.optimize(sub, method)
        return results

    # ------------------------------------------------------------------
    # Backtest all methods
    # ------------------------------------------------------------------

    def backtest_all(
        self, returns: pd.DataFrame, rebalance_freq: int = 21,
    ) -> Tuple[Dict[str, List[BacktestRow]], List[MethodComparison]]:
        """Backtest all RP methods + equal-weight benchmark."""
        methods = list(RPMethod) + ["equal_weight"]
        assets = returns.columns.tolist()
        n_assets = len(assets)
        if n_assets == 0:
            return {}, []

        all_rows: Dict[str, List[BacktestRow]] = {str(m): [] for m in methods}
        all_rets: Dict[str, List[float]] = {str(m): [] for m in methods}

        # Pre-compute weights at each rebalance
        weight_cache: Dict[str, Dict[str, float]] = {}
        for m in RPMethod:
            weight_cache[m.value] = {}
        weight_cache["equal_weight"] = {a: 1.0 / n_assets for a in assets}

        cum = {str(m): 1.0 for m in methods}

        for i in range(len(returns)):
            dt = returns.index[i]

            # Rebalance?
            if i % rebalance_freq == 0 and i >= 60:
                hist = returns.iloc[max(0, i - 252):i]
                for m in RPMethod:
                    try:
                        pw = self.optimize(hist, m)
                        weight_cache[m.value] = pw.weights
                    except Exception:
                        pass

            day_ret = returns.iloc[i]
            for m_key in [str(m) for m in methods]:
                w = weight_cache.get(m_key, {a: 1.0 / n_assets for a in assets})
                port_r = sum(w.get(a, 0) * float(day_ret.get(a, 0)) for a in assets)
                cum[m_key] *= (1 + port_r)
                all_rets[m_key].append(port_r)
                all_rows[m_key].append(BacktestRow(
                    date=dt, method=m_key, daily_return=port_r,
                    cumulative=cum[m_key] - 1,
                ))

        comparisons: List[MethodComparison] = []
        for m_key in [str(m) for m in methods]:
            rets = np.array(all_rets[m_key])
            if len(rets) < 2:
                continue
            total = float(np.prod(1 + rets) - 1)
            n_years = len(rets) / TRADING_DAYS
            annual = (1 + total) ** (1 / max(n_years, 0.01)) - 1
            vol = float(rets.std() * np.sqrt(TRADING_DAYS))
            mu = float(rets.mean())
            std = float(rets.std())
            sharpe = mu / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
            eq = np.cumprod(1 + rets)
            dd = float((1 - eq / np.maximum.accumulate(eq)).max())
            comparisons.append(MethodComparison(
                method=m_key, total_return=total, annual_return=annual,
                sharpe=sharpe, max_drawdown=dd, volatility=vol,
            ))
        comparisons.sort(key=lambda c: c.sharpe, reverse=True)
        return all_rows, comparisons

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self, assets, w, cov, returns, method,
    ) -> RPWeights:
        mu = returns.mean().values * TRADING_DAYS
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 1e-12 else 0.0
        return RPWeights(
            weights=dict(zip(assets, w.tolist())),
            method=method, expected_return=ret,
            expected_vol=vol, sharpe=sharpe,
        )

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_line(
        series_map: Dict[str, List[float]], title: str,
        width: int = 720, height: int = 220,
    ) -> str:
        if not series_map or all(len(v) < 2 for v in series_map.values()):
            return ""
        colors = ["#2980b9", "#e74c3c", "#27ae60", "#e67e22", "#8e44ad", "#1abc9c"]
        all_v = [v for vals in series_map.values() for v in vals]
        vmin, vmax = min(all_v), max(all_v)
        if vmax <= vmin:
            vmax = vmin + 0.01
        n = max(len(v) for v in series_map.values())
        pad_l, pad_r, pad_t, pad_b = 50, 15, 28, 35
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b
        def tx(i): return pad_l + i / max(n - 1, 1) * pw
        def ty(v): return pad_t + (1 - (v - vmin) / (vmax - vmin)) * ph

        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="background:#fff;border:1px solid #ddd;'
             f'border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')
        for ci, (label, vals) in enumerate(series_map.items()):
            c = colors[ci % len(colors)]
            d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                          for i, v in enumerate(vals))
            p.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2"/>')
            lx = pad_l + ci * 100
            p.append(f'<rect x="{lx}" y="{height - 16}" width="10" height="10" fill="{c}"/>')
            p.append(f'<text x="{lx + 14}" y="{height - 7}" font-size="9" fill="#333">{label}</text>')
        p.append("</svg>")
        return "\n".join(p)

    @staticmethod
    def _svg_pie(
        slices: List[Tuple[str, float, str]],
        width: int = 260, height: int = 260, title: str = "",
    ) -> str:
        if not slices:
            return ""
        cx, cy, r = width // 2, height // 2 - 8, min(width, height) // 2 - 35
        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="background:#fff;border:1px solid #ddd;'
             f'border-radius:6px;margin:.5rem 0">']
        if title:
            p.append(f'<text x="{cx}" y="16" text-anchor="middle" font-size="12" '
                     f'font-weight="bold" fill="#1a1a2e">{title}</text>')
        angle = -90.0
        for label, frac, color in slices:
            if frac <= 0:
                continue
            s_rad = np.radians(angle)
            sweep = frac * 360
            e_rad = np.radians(angle + sweep)
            lg = 1 if sweep > 180 else 0
            x1 = cx + r * np.cos(s_rad)
            y1 = cy + r * np.sin(s_rad)
            x2 = cx + r * np.cos(e_rad)
            y2 = cy + r * np.sin(e_rad)
            p.append(f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} '
                     f'A{r},{r} 0 {lg} 1 {x2:.1f},{y2:.1f} Z" fill="{color}"/>')
            angle += sweep
        lx, ly = 5, height - 14
        for label, frac, color in slices:
            if frac <= 0:
                continue
            p.append(f'<rect x="{lx}" y="{ly}" width="8" height="8" fill="{color}"/>')
            p.append(f'<text x="{lx + 11}" y="{ly + 7}" font-size="8" fill="#333">{label} {frac:.0%}</text>')
            lx += max(len(label) * 6 + 40, 55)
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        comparisons: List[MethodComparison],
        all_rows: Optional[Dict[str, List[BacktestRow]]] = None,
        risk_contribs: Optional[List[RiskContribution]] = None,
        regime_portfolios: Optional[Dict[str, RPWeights]] = None,
        output_path: str = "reports/risk_parity.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Equity curves
        eq_svg = ""
        if all_rows:
            series = {m: [r.cumulative for r in rows] for m, rows in all_rows.items()
                       if len(rows) > 2}
            eq_svg = self._svg_line(series, "Cumulative Return Comparison")

        # Drawdown curves
        dd_svg = ""
        if all_rows:
            dd_series: Dict[str, List[float]] = {}
            for m, rows in all_rows.items():
                if len(rows) < 2:
                    continue
                eq = np.array([1 + r.cumulative for r in rows])
                hwm = np.maximum.accumulate(eq)
                dd = (1 - eq / hwm).tolist()
                dd_series[m] = dd
            dd_svg = self._svg_line(dd_series, "Drawdown Comparison")

        # Risk contribution pie
        rc_svg = ""
        if risk_contribs:
            palette = ["#2980b9", "#e74c3c", "#27ae60", "#e67e22", "#8e44ad"]
            slices = [(rc.asset, rc.pct_of_total, palette[i % len(palette)])
                       for i, rc in enumerate(risk_contribs)]
            rc_svg = self._svg_pie(slices, title="Risk Contribution")

        # Comparison table
        comp_rows = [
            f"<tr><td style='text-align:left'>{c.method}</td>"
            f"<td>{c.total_return:.2%}</td><td>{c.annual_return:.2%}</td>"
            f"<td>{c.sharpe:.2f}</td><td>{c.max_drawdown:.2%}</td>"
            f"<td>{c.volatility:.2%}</td></tr>"
            for c in comparisons
        ]

        # Regime table
        regime_html = ""
        if regime_portfolios:
            rows = []
            for reg, pw in sorted(regime_portfolios.items()):
                top = sorted(pw.weights.items(), key=lambda x: x[1], reverse=True)[:3]
                top_str = ", ".join(f"{a}:{w:.0%}" for a, w in top)
                rows.append(f"<tr><td>{reg}</td><td>{pw.method}</td>"
                             f"<td>{pw.sharpe:.2f}</td><td>{top_str}</td></tr>")
            regime_html = f"""
<h2>Regime Allocations</h2>
<table><tr><th>Regime</th><th>Method</th><th>Sharpe</th><th>Top Weights</th></tr>
{''.join(rows)}</table>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Risk Parity</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.charts {{ display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }}
</style></head><body>
<h1>Risk Parity Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Methods tested:</strong> {len(comparisons)}</p>
</div>

{eq_svg}
{dd_svg}

<div class="charts">{rc_svg}</div>

<h2>Method Comparison</h2>
<table><tr><th style='text-align:left'>Method</th><th>Total</th><th>Annual</th>
<th>Sharpe</th><th>Max DD</th><th>Vol</th></tr>
{''.join(comp_rows)}</table>

{regime_html}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Risk parity report -> %s", path)
        return str(path)
