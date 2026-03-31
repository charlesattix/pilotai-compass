"""Tail risk analyzer – CVaR, EVT/GPD fitting, stress VaR, and per-experiment
tail contribution for credit spread portfolios.

Provides:
  1. CVaR (Conditional Value at Risk) at 95% and 99% confidence
  2. Expected Shortfall decomposition by experiment
  3. Extreme Value Theory (EVT) fitting via Generalized Pareto Distribution (GPD)
  4. Per-experiment tail risk contribution to portfolio
  5. Stress VaR — VaR computed on worst historical periods only
  6. HTML report with waterfall chart, tail distribution fit, per-experiment
     contributions, and stress VaR comparison
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
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────
DEFAULT_CONFIDENCE_LEVELS: Tuple[float, ...] = (0.95, 0.99)
PERIODS_PER_YEAR: int = 252
# Stress period: worst N-day drawdown windows
DEFAULT_STRESS_WINDOW: int = 20
DEFAULT_STRESS_TOP_N: int = 5


# ── Data classes ────────────────────────────────────────────────────────────
@dataclass
class VaRCVaR:
    """Value-at-Risk and Conditional VaR at a single confidence level."""
    confidence: float
    var: float          # positive number = loss magnitude
    cvar: float         # expected shortfall beyond VaR
    n_obs: int
    horizon_days: int = 1


@dataclass
class GPDFit:
    """Generalized Pareto Distribution fit to tail losses."""
    xi: float           # shape parameter (>0 = heavy tail)
    beta: float         # scale parameter
    threshold: float    # exceedance threshold (u)
    n_exceedances: int
    ks_pvalue: float    # Kolmogorov–Smirnov goodness-of-fit p-value
    evt_var_95: float   # GPD-implied VaR at 95%
    evt_var_99: float   # GPD-implied VaR at 99%
    evt_cvar_99: float  # GPD-implied CVaR at 99%


@dataclass
class ExperimentTailContrib:
    """Tail risk contribution of one experiment to the portfolio."""
    experiment_id: str
    weight: float
    standalone_cvar_95: float
    standalone_cvar_99: float
    marginal_cvar_99: float    # change in portfolio CVaR when experiment removed
    pct_contribution: float    # fraction of portfolio CVaR attributable


@dataclass
class StressVaR:
    """VaR computed over worst historical periods only."""
    confidence: float
    stress_var: float
    normal_var: float
    stress_ratio: float     # stress_var / normal_var
    n_stress_obs: int


@dataclass
class TailRiskResult:
    """Complete tail risk analysis output."""
    var_cvar: List[VaRCVaR] = field(default_factory=list)
    gpd_fit: Optional[GPDFit] = None
    experiment_contribs: List[ExperimentTailContrib] = field(default_factory=list)
    stress_vars: List[StressVaR] = field(default_factory=list)
    portfolio_returns: Optional[pd.Series] = None
    generated_at: str = ""


# ── GPD helper functions ────────────────────────────────────────────────────
def _fit_gpd_mle(exceedances: np.ndarray) -> Tuple[float, float]:
    """Fit GPD shape (xi) and scale (beta) via profile maximum likelihood.

    Uses the Grimshaw (1993) approach: for a grid of xi values find the beta
    that maximises the log-likelihood, then pick the best xi.
    """
    n = len(exceedances)
    if n < 5:
        return (0.0, float(np.mean(exceedances)))

    mean_exc = float(np.mean(exceedances))

    # Grid search over xi
    best_ll = -np.inf
    best_xi = 0.0
    best_beta = mean_exc

    for xi_candidate in np.linspace(-0.5, 2.0, 251):
        if abs(xi_candidate) < 1e-8:
            # Exponential case
            beta_c = mean_exc
            if beta_c <= 0:
                continue
            ll = -n * np.log(beta_c) - np.sum(exceedances) / beta_c
        else:
            beta_c = mean_exc * xi_candidate / (
                np.mean((1 + xi_candidate * exceedances / mean_exc) ** 0 ) - 1
            ) if abs(xi_candidate) > 1e-8 else mean_exc
            # MLE beta given xi
            beta_c = xi_candidate * mean_exc / ((1 + xi_candidate) - 1) if abs(xi_candidate) > 1e-8 else mean_exc
            # Direct MLE: beta = xi * mean(exceedances) only for xi>0
            # More robust: use the MLE relationship
            inner = 1 + xi_candidate * exceedances / (mean_exc * max(xi_candidate, 0.01))
            if np.any(inner <= 0):
                continue
            # beta from MLE first-order condition
            beta_c = float(xi_candidate / n * np.sum(exceedances / inner))
            if beta_c <= 1e-12:
                continue
            inner2 = 1 + xi_candidate * exceedances / beta_c
            if np.any(inner2 <= 0):
                continue
            ll = -n * np.log(beta_c) - (1 + 1 / xi_candidate) * np.sum(np.log(inner2))

        if np.isfinite(ll) and ll > best_ll:
            best_ll = ll
            best_xi = xi_candidate
            best_beta = beta_c

    return (float(best_xi), float(max(best_beta, 1e-12)))


def _gpd_cdf(x: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """GPD cumulative distribution function."""
    if abs(xi) < 1e-8:
        return 1.0 - np.exp(-x / beta)
    inner = 1 + xi * x / beta
    inner = np.maximum(inner, 1e-12)
    return 1.0 - inner ** (-1.0 / xi)


def _gpd_quantile(p: float, xi: float, beta: float) -> float:
    """GPD quantile function (inverse CDF)."""
    if abs(xi) < 1e-8:
        return -beta * np.log(1 - p)
    return beta / xi * ((1 - p) ** (-xi) - 1)


def _ks_test_gpd(exceedances: np.ndarray, xi: float, beta: float) -> float:
    """Kolmogorov–Smirnov test p-value for GPD fit."""
    n = len(exceedances)
    if n < 2:
        return 0.0
    sorted_exc = np.sort(exceedances)
    empirical = np.arange(1, n + 1) / n
    theoretical = _gpd_cdf(sorted_exc, xi, beta)
    d_stat = float(np.max(np.abs(empirical - theoretical)))
    # Approximate p-value (asymptotic)
    lam = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * d_stat
    if lam <= 0:
        return 1.0
    # Kolmogorov distribution approximation
    p_val = 2.0 * np.sum([
        (-1) ** (k - 1) * np.exp(-2 * k * k * lam * lam)
        for k in range(1, 20)
    ])
    return float(np.clip(p_val, 0.0, 1.0))


# ── Core analyzer ───────────────────────────────────────────────────────────
class TailRiskAnalyzer:
    """Computes tail risk metrics for credit spread portfolios."""

    def __init__(
        self,
        confidence_levels: Tuple[float, ...] = DEFAULT_CONFIDENCE_LEVELS,
        stress_window: int = DEFAULT_STRESS_WINDOW,
        stress_top_n: int = DEFAULT_STRESS_TOP_N,
        gpd_threshold_pct: float = 90.0,
    ) -> None:
        self.confidence_levels = confidence_levels
        self.stress_window = stress_window
        self.stress_top_n = stress_top_n
        self.gpd_threshold_pct = gpd_threshold_pct

    # ── Public API ──────────────────────────────────────────────────────────
    def analyze(
        self,
        experiment_returns: Dict[str, pd.Series],
        weights: Optional[Dict[str, float]] = None,
    ) -> TailRiskResult:
        """Run full tail risk analysis.

        Parameters
        ----------
        experiment_returns : dict
            Mapping of experiment_id → daily return series.
        weights : dict, optional
            Portfolio weights per experiment.  Equal-weight if None.
        """
        if not experiment_returns:
            return TailRiskResult(generated_at=self._now())

        ids = list(experiment_returns.keys())
        if weights is None:
            w = 1.0 / len(ids)
            weights = {eid: w for eid in ids}

        # Align all series
        aligned = self._align_returns(experiment_returns)
        if aligned.empty or len(aligned) < 10:
            logger.warning("Too few aligned observations (%d)", len(aligned))
            return TailRiskResult(generated_at=self._now())

        port_ret = self._portfolio_returns(aligned, weights)

        var_cvar = self._compute_var_cvar(port_ret)
        gpd_fit = self._fit_evt(port_ret)
        contribs = self._experiment_contributions(aligned, weights, port_ret)
        stress = self._stress_var(port_ret)

        return TailRiskResult(
            var_cvar=var_cvar,
            gpd_fit=gpd_fit,
            experiment_contribs=contribs,
            stress_vars=stress,
            portfolio_returns=port_ret,
            generated_at=self._now(),
        )

    def generate_report(
        self,
        result: TailRiskResult,
        output_path: str | Path = "reports/tail_risk.html",
    ) -> Path:
        """Write self-contained HTML report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html = self._build_html(result)
        path.write_text(html, encoding="utf-8")
        logger.info("Tail risk report written to %s", path)
        return path

    # ── VaR / CVaR via historical simulation ────────────────────────────────
    def _compute_var_cvar(self, returns: pd.Series) -> List[VaRCVaR]:
        results: List[VaRCVaR] = []
        losses = -returns.values  # positive = loss
        for conf in self.confidence_levels:
            alpha = 1 - conf  # e.g. 0.05 for 95%
            var = float(np.percentile(losses, (1 - alpha) * 100))
            tail = losses[losses >= var]
            cvar = float(np.mean(tail)) if len(tail) > 0 else var
            results.append(VaRCVaR(
                confidence=conf, var=var, cvar=cvar,
                n_obs=len(returns), horizon_days=1,
            ))
        return results

    # ── EVT / GPD ───────────────────────────────────────────────────────────
    def _fit_evt(self, returns: pd.Series) -> GPDFit:
        losses = -returns.values
        threshold = float(np.percentile(losses, self.gpd_threshold_pct))
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < 5:
            return GPDFit(
                xi=0.0, beta=0.0, threshold=threshold,
                n_exceedances=len(exceedances), ks_pvalue=0.0,
                evt_var_95=0.0, evt_var_99=0.0, evt_cvar_99=0.0,
            )

        xi, beta = _fit_gpd_mle(exceedances)
        ks_p = _ks_test_gpd(exceedances, xi, beta)

        n = len(losses)
        n_u = len(exceedances)
        ratio = n_u / n

        # GPD-implied VaR: u + GPD_quantile( (1 - (n/n_u)*(1-conf)) )
        def _evt_var(conf: float) -> float:
            p = 1 - (1 - conf) / ratio if ratio > 1e-9 else 0.99
            p = min(p, 0.9999)
            return threshold + _gpd_quantile(p, xi, beta)

        evt_var_95 = _evt_var(0.95)
        evt_var_99 = _evt_var(0.99)

        # GPD-implied CVaR at 99%
        if xi < 1.0:
            evt_cvar_99 = evt_var_99 / (1 - xi) + (beta - xi * threshold) / (1 - xi)
        else:
            evt_cvar_99 = evt_var_99 * 2.0  # heavy tail fallback

        return GPDFit(
            xi=xi, beta=beta, threshold=threshold,
            n_exceedances=n_u, ks_pvalue=ks_p,
            evt_var_95=evt_var_95, evt_var_99=evt_var_99,
            evt_cvar_99=evt_cvar_99,
        )

    # ── Per-experiment tail contribution ────────────────────────────────────
    def _experiment_contributions(
        self,
        aligned: pd.DataFrame,
        weights: Dict[str, float],
        port_ret: pd.Series,
    ) -> List[ExperimentTailContrib]:
        port_losses = -port_ret.values
        var_99 = float(np.percentile(port_losses, 99))
        tail_mask = port_losses >= var_99
        port_cvar_99 = float(np.mean(port_losses[tail_mask])) if tail_mask.any() else 0.0

        contribs: List[ExperimentTailContrib] = []
        for eid in aligned.columns:
            w = weights.get(eid, 0.0)
            exp_losses = -aligned[eid].values

            # Standalone CVaR
            var_95_s = float(np.percentile(exp_losses, 95))
            var_99_s = float(np.percentile(exp_losses, 99))
            tail_s_95 = exp_losses[exp_losses >= var_95_s]
            tail_s_99 = exp_losses[exp_losses >= var_99_s]
            cvar_95 = float(np.mean(tail_s_95)) if len(tail_s_95) > 0 else var_95_s
            cvar_99 = float(np.mean(tail_s_99)) if len(tail_s_99) > 0 else var_99_s

            # Marginal CVaR: portfolio CVaR without this experiment
            other_cols = [c for c in aligned.columns if c != eid]
            if other_cols:
                other_weights = {c: weights.get(c, 0.0) for c in other_cols}
                total_w = sum(other_weights.values())
                if total_w > 1e-9:
                    other_weights = {c: v / total_w for c, v in other_weights.items()}
                reduced_ret = sum(aligned[c] * other_weights[c] for c in other_cols)
                reduced_losses = -reduced_ret.values
                rv99 = float(np.percentile(reduced_losses, 99))
                rt = reduced_losses[reduced_losses >= rv99]
                reduced_cvar = float(np.mean(rt)) if len(rt) > 0 else rv99
                marginal = port_cvar_99 - reduced_cvar
            else:
                marginal = port_cvar_99

            pct = marginal / port_cvar_99 if port_cvar_99 > 1e-12 else 0.0

            contribs.append(ExperimentTailContrib(
                experiment_id=eid,
                weight=w,
                standalone_cvar_95=cvar_95,
                standalone_cvar_99=cvar_99,
                marginal_cvar_99=marginal,
                pct_contribution=pct,
            ))
        return contribs

    # ── Stress VaR ──────────────────────────────────────────────────────────
    def _stress_var(self, returns: pd.Series) -> List[StressVaR]:
        # Identify worst drawdown windows
        rolling_ret = returns.rolling(window=self.stress_window).sum()
        rolling_ret = rolling_ret.dropna()

        if len(rolling_ret) < self.stress_top_n:
            return []

        worst_ends = rolling_ret.nsmallest(self.stress_top_n).index
        stress_mask = pd.Series(False, index=returns.index)
        for end_idx in worst_ends:
            pos = returns.index.get_loc(end_idx)
            start_pos = max(0, pos - self.stress_window + 1)
            stress_mask.iloc[start_pos:pos + 1] = True

        stress_returns = returns[stress_mask]
        if len(stress_returns) < 5:
            return []

        stress_losses = -stress_returns.values
        normal_losses = -returns.values

        results: List[StressVaR] = []
        for conf in self.confidence_levels:
            alpha = 1 - conf
            pct = (1 - alpha) * 100
            s_var = float(np.percentile(stress_losses, pct))
            n_var = float(np.percentile(normal_losses, pct))
            ratio = s_var / n_var if abs(n_var) > 1e-12 else 0.0
            results.append(StressVaR(
                confidence=conf,
                stress_var=s_var,
                normal_var=n_var,
                stress_ratio=ratio,
                n_stress_obs=len(stress_returns),
            ))
        return results

    # ── Alignment and portfolio returns ─────────────────────────────────────
    @staticmethod
    def _align_returns(returns: Dict[str, pd.Series]) -> pd.DataFrame:
        frames = {eid: s.rename(eid) for eid, s in returns.items()}
        df = pd.concat(frames.values(), axis=1, join="inner")
        return df.dropna()

    @staticmethod
    def _portfolio_returns(
        aligned: pd.DataFrame, weights: Dict[str, float],
    ) -> pd.Series:
        port = pd.Series(0.0, index=aligned.index)
        for col in aligned.columns:
            port += aligned[col] * weights.get(col, 0.0)
        return port

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # ── HTML report ─────────────────────────────────────────────────────────
    def _build_html(self, r: TailRiskResult) -> str:
        cards = self._html_cards(r)
        waterfall = self._svg_waterfall(r.var_cvar, r.gpd_fit)
        tail_fit = self._html_gpd_section(r.gpd_fit)
        contrib_table = self._html_contrib_table(r.experiment_contribs)
        stress_section = self._html_stress(r.stress_vars)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Tail Risk Analysis</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.6rem;margin-bottom:4px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:28px}}
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
<h1>Tail Risk Analysis</h1>
<p class="sub">Generated {r.generated_at or 'N/A'}</p>

{cards}

<div class="sec">
<h2>VaR / CVaR Waterfall</h2>
{waterfall}
</div>

{tail_fit}
{contrib_table}
{stress_section}

</body>
</html>"""

    # ── Cards ───────────────────────────────────────────────────────────────
    @staticmethod
    def _html_cards(r: TailRiskResult) -> str:
        var95 = next((v for v in r.var_cvar if v.confidence == 0.95), None)
        var99 = next((v for v in r.var_cvar if v.confidence == 0.99), None)
        xi = f"{r.gpd_fit.xi:.3f}" if r.gpd_fit else "N/A"
        ks = f"{r.gpd_fit.ks_pvalue:.3f}" if r.gpd_fit else "N/A"
        n_obs = var95.n_obs if var95 else 0
        return f"""<div class="grid">
<div class="card"><div class="lbl">VaR 95%</div><div class="val">{(var95.var if var95 else 0):.4f}</div></div>
<div class="card"><div class="lbl">CVaR 95%</div><div class="val">{(var95.cvar if var95 else 0):.4f}</div></div>
<div class="card"><div class="lbl">VaR 99%</div><div class="val">{(var99.var if var99 else 0):.4f}</div></div>
<div class="card"><div class="lbl">CVaR 99%</div><div class="val">{(var99.cvar if var99 else 0):.4f}</div></div>
<div class="card"><div class="lbl">GPD Shape (xi)</div><div class="val">{xi}</div></div>
<div class="card"><div class="lbl">KS p-value</div><div class="val">{ks}</div></div>
<div class="card"><div class="lbl">Observations</div><div class="val">{n_obs}</div></div>
</div>"""

    # ── Waterfall SVG ───────────────────────────────────────────────────────
    @staticmethod
    def _svg_waterfall(
        var_cvar: List[VaRCVaR], gpd: Optional[GPDFit],
    ) -> str:
        items: List[Tuple[str, float]] = []
        for vc in var_cvar:
            items.append((f"VaR {vc.confidence:.0%}", vc.var))
            items.append((f"CVaR {vc.confidence:.0%}", vc.cvar))
        if gpd and gpd.evt_var_99 > 0:
            items.append(("EVT VaR 99%", gpd.evt_var_99))
            items.append(("EVT CVaR 99%", gpd.evt_cvar_99))

        if not items:
            return "<p>No VaR data.</p>"

        w, h = 560, 240
        pad_l, pad_b, pad_t = 80, 50, 20
        chart_h = h - pad_b - pad_t
        n = len(items)
        max_val = max(v for _, v in items) or 0.01
        bar_w = min(48, (w - pad_l) // n - 8)

        bars = ""
        colours = ["#f87171", "#ef4444", "#fb923c", "#f97316", "#a78bfa", "#8b5cf6", "#c084fc", "#a855f7"]
        for i, (label, val) in enumerate(items):
            x = pad_l + i * ((w - pad_l) // n) + 4
            bar_h = max(2, val / max_val * (chart_h * 0.85))
            y = pad_t + chart_h - bar_h
            c = colours[i % len(colours)]
            bars += (
                f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
                f'rx="3" fill="{c}" opacity="0.85"/>'
                f'<text x="{x + bar_w // 2}" y="{y - 5}" text-anchor="middle" '
                f'font-size="10" fill="#e2e8f0">{val:.4f}</text>'
                f'<text x="{x + bar_w // 2}" y="{h - 8}" text-anchor="middle" '
                f'font-size="9" fill="#94a3b8" transform="rotate(-25 {x + bar_w // 2} {h - 8})">'
                f'{label}</text>'
            )

        baseline = pad_t + chart_h
        return (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
            f'<line x1="{pad_l}" y1="{baseline}" x2="{w}" y2="{baseline}" '
            f'stroke="#475569" stroke-width="1"/>'
            f'{bars}</svg>'
        )

    # ── GPD section ─────────────────────────────────────────────────────────
    @staticmethod
    def _html_gpd_section(gpd: Optional[GPDFit]) -> str:
        if not gpd or gpd.n_exceedances < 5:
            return ""
        tail_type = "Heavy" if gpd.xi > 0.1 else "Light" if gpd.xi < -0.1 else "Medium"
        return f"""<div class="sec">
<h2>Extreme Value Theory — GPD Fit</h2>
<table>
<thead><tr><th>Parameter</th><th>Value</th><th>Interpretation</th></tr></thead>
<tbody>
<tr><td>Shape (ξ)</td><td>{gpd.xi:.4f}</td><td class="{'neg' if gpd.xi > 0.3 else 'warn' if gpd.xi > 0 else 'pos'}">{tail_type} tail</td></tr>
<tr><td>Scale (β)</td><td>{gpd.beta:.6f}</td><td>Dispersion of exceedances</td></tr>
<tr><td>Threshold (u)</td><td>{gpd.threshold:.6f}</td><td>P{int(100 - 0)}th percentile of losses</td></tr>
<tr><td>Exceedances</td><td>{gpd.n_exceedances}</td><td>Observations above threshold</td></tr>
<tr><td>KS p-value</td><td>{gpd.ks_pvalue:.4f}</td><td class="{'pos' if gpd.ks_pvalue > 0.05 else 'neg'}">{'Good fit' if gpd.ks_pvalue > 0.05 else 'Poor fit'}</td></tr>
<tr><td>EVT VaR 95%</td><td>{gpd.evt_var_95:.6f}</td><td>GPD-implied</td></tr>
<tr><td>EVT VaR 99%</td><td>{gpd.evt_var_99:.6f}</td><td>GPD-implied</td></tr>
<tr><td>EVT CVaR 99%</td><td>{gpd.evt_cvar_99:.6f}</td><td>GPD-implied expected shortfall</td></tr>
</tbody>
</table>
</div>"""

    # ── Contribution table ──────────────────────────────────────────────────
    @staticmethod
    def _html_contrib_table(contribs: List[ExperimentTailContrib]) -> str:
        if not contribs:
            return ""
        rows = ""
        for c in sorted(contribs, key=lambda x: -abs(x.pct_contribution)):
            cls = "neg" if c.pct_contribution > 0.3 else ""
            rows += (
                f"<tr><td>{c.experiment_id}</td>"
                f"<td>{c.weight:.1%}</td>"
                f"<td>{c.standalone_cvar_95:.4f}</td>"
                f"<td>{c.standalone_cvar_99:.4f}</td>"
                f"<td>{c.marginal_cvar_99:.4f}</td>"
                f'<td class="{cls}">{c.pct_contribution:.1%}</td></tr>'
            )
        return f"""<div class="sec">
<h2>Per-Experiment Tail Risk Contribution</h2>
<table>
<thead><tr><th>Experiment</th><th>Weight</th><th>CVaR 95%</th><th>CVaR 99%</th><th>Marginal CVaR</th><th>% Contribution</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""

    # ── Stress VaR ──────────────────────────────────────────────────────────
    @staticmethod
    def _html_stress(stress: List[StressVaR]) -> str:
        if not stress:
            return ""
        rows = ""
        for s in stress:
            ratio_cls = "neg" if s.stress_ratio > 2.0 else "warn" if s.stress_ratio > 1.5 else ""
            rows += (
                f"<tr><td>{s.confidence:.0%}</td>"
                f"<td>{s.normal_var:.4f}</td>"
                f"<td>{s.stress_var:.4f}</td>"
                f'<td class="{ratio_cls}">{s.stress_ratio:.2f}x</td>'
                f"<td>{s.n_stress_obs}</td></tr>"
            )
        return f"""<div class="sec">
<h2>Stress VaR Comparison</h2>
<table>
<thead><tr><th>Confidence</th><th>Normal VaR</th><th>Stress VaR</th><th>Ratio</th><th>Stress Obs</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>"""
