"""
Factor exposure analyzer — decomposes strategy returns into standard
risk factor exposures and generates neutralisation overlays.

Factors (Fama-French-like, proxied via ETFs):
  1. Market  (SPY)  — equity beta
  2. Size    (IWM)  — small-cap premium
  3. Value   (IWD)  — value premium
  4. Momentum(MTUM) — trend factor
  5. Low Vol (USMV) — defensive / min-vol
  6. Quality (QUAL) — earnings quality

Components:
  - Full-sample OLS regression → factor betas + alpha
  - Rolling factor betas (detect regime-dependent tilts)
  - Factor attribution (how much of return came from each factor)
  - Factor-neutral overlay (hedge out unintended exposures)
  - Regime-conditional analysis (betas per regime)

All methods work on pre-loaded data — no API calls.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

FACTOR_NAMES = ["market", "size", "value", "momentum", "low_vol", "quality"]
FACTOR_ETFS = {
    "market": "SPY", "size": "IWM", "value": "IWD",
    "momentum": "MTUM", "low_vol": "USMV", "quality": "QUAL",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FactorBetas:
    """Full-sample factor regression result."""
    alpha: float                 # annualised alpha (intercept × 252)
    alpha_t_stat: float
    betas: Dict[str, float]     # {factor_name: beta}
    t_stats: Dict[str, float]
    r_squared: float
    residual_vol: float          # annualised unexplained vol


@dataclass
class FactorAttribution:
    """Return attributed to each factor."""
    total_return: float
    alpha_contribution: float
    factor_contributions: Dict[str, float]
    residual: float


@dataclass
class RollingBeta:
    """Rolling beta for one factor at one date."""
    date: datetime
    factor: str
    beta: float
    t_stat: float


@dataclass
class RegimeFactorProfile:
    """Factor betas within a specific market regime."""
    regime: str
    n_days: int
    betas: Dict[str, float]
    alpha: float
    r_squared: float


@dataclass
class NeutralOverlay:
    """Hedge ratios to neutralise factor exposures."""
    hedges: Dict[str, float]     # {factor: shares/contracts to hedge}
    residual_beta: Dict[str, float]  # remaining exposure after hedge
    cost_annual_pct: float


@dataclass
class FactorAnalysisResult:
    """Complete factor analysis output."""
    betas: FactorBetas
    attribution: FactorAttribution
    rolling: pd.DataFrame          # date × factor betas
    regime_profiles: List[RegimeFactorProfile]
    overlay: Optional[NeutralOverlay]


# ---------------------------------------------------------------------------
# Synthetic factor data generator
# ---------------------------------------------------------------------------

def generate_factor_returns(
    n_days: int = 1512, seed: int = 42,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Generate strategy returns + factor returns calibrated to 2020-2025.

    Returns (strategy_returns, factor_returns_df).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-02", periods=n_days)

    # Factor returns (calibrated to real ETF stats)
    factors = pd.DataFrame(index=idx)
    factors["market"] = rng.normal(0.0004, 0.012, n_days)   # SPY ~10%, 19% vol
    factors["size"] = rng.normal(0.0003, 0.015, n_days)      # IWM
    factors["value"] = rng.normal(0.0002, 0.011, n_days)     # IWD
    factors["momentum"] = rng.normal(0.0003, 0.010, n_days)  # MTUM
    factors["low_vol"] = rng.normal(0.0003, 0.008, n_days)   # USMV
    factors["quality"] = rng.normal(0.0003, 0.009, n_days)   # QUAL

    # Add cross-correlations (market drives all)
    for f in ["size", "value", "low_vol", "quality"]:
        factors[f] += factors["market"] * rng.uniform(0.3, 0.7)
    factors["momentum"] += factors["market"] * 0.2  # lower corr

    # COVID crash
    if n_days > 80:
        factors.iloc[50:70] *= rng.uniform(-2, -1, (20, 6))

    # Strategy returns: alpha + factor exposures + noise
    true_betas = {"market": -0.15, "size": 0.05, "value": 0.08,
                   "momentum": -0.03, "low_vol": 0.12, "quality": 0.04}
    true_alpha = 0.0008  # ~20% annualised
    noise = rng.normal(0, 0.005, n_days)

    strat = true_alpha + noise
    for f, beta in true_betas.items():
        strat += beta * factors[f].values

    return pd.Series(strat, index=idx, name="strategy"), factors


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class FactorExposureAnalyzer:
    """Factor exposure analysis and neutralisation.

    Args:
        rolling_window: Days for rolling beta estimation.
        risk_free_rate: Annualised risk-free rate.
    """

    def __init__(
        self,
        rolling_window: int = 63,
        risk_free_rate: float = 0.045,
    ) -> None:
        self.rolling_window = rolling_window
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Full-sample regression
    # ------------------------------------------------------------------

    def estimate_betas(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> FactorBetas:
        """OLS regression: strategy = alpha + sum(beta_i × factor_i) + eps."""
        aligned = pd.concat([strategy_returns.rename("strat"), factor_returns], axis=1).dropna()
        if len(aligned) < 30:
            return FactorBetas(0, 0, {f: 0 for f in FACTOR_NAMES}, {f: 0 for f in FACTOR_NAMES}, 0, 0)

        rf_daily = self.risk_free_rate / TRADING_DAYS
        y = aligned["strat"].values - rf_daily
        cols = [c for c in FACTOR_NAMES if c in aligned.columns]
        X = aligned[cols].values - rf_daily
        X_c = np.column_stack([np.ones(len(y)), X])

        try:
            betas_raw, residuals, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
        except np.linalg.LinAlgError:
            return FactorBetas(0, 0, {f: 0 for f in cols}, {f: 0 for f in cols}, 0, 0)

        alpha_daily = float(betas_raw[0])
        factor_betas = {cols[i]: float(betas_raw[i + 1]) for i in range(len(cols))}

        # Stats
        pred = X_c @ betas_raw
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else 0

        resid = y - pred
        resid_vol = float(resid.std() * math.sqrt(TRADING_DAYS))

        # t-stats (approximate)
        n, k = len(y), len(betas_raw)
        mse = ss_res / max(n - k, 1)
        try:
            cov_matrix = mse * np.linalg.inv(X_c.T @ X_c)
            se = np.sqrt(np.diag(cov_matrix))
            t_stats = {cols[i]: float(betas_raw[i + 1] / se[i + 1]) if se[i + 1] > 1e-12 else 0
                        for i in range(len(cols))}
            alpha_t = float(betas_raw[0] / se[0]) if se[0] > 1e-12 else 0
        except np.linalg.LinAlgError:
            t_stats = {f: 0 for f in cols}
            alpha_t = 0

        return FactorBetas(
            alpha=alpha_daily * TRADING_DAYS,
            alpha_t_stat=alpha_t,
            betas=factor_betas,
            t_stats=t_stats,
            r_squared=r2,
            residual_vol=resid_vol,
        )

    # ------------------------------------------------------------------
    # Factor attribution
    # ------------------------------------------------------------------

    def factor_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        betas: Optional[FactorBetas] = None,
    ) -> FactorAttribution:
        """Decompose total return into factor contributions."""
        if betas is None:
            betas = self.estimate_betas(strategy_returns, factor_returns)

        aligned = pd.concat([strategy_returns.rename("strat"), factor_returns], axis=1).dropna()
        total = float(aligned["strat"].sum())

        contributions = {}
        explained = 0.0
        for f, beta in betas.betas.items():
            if f in aligned.columns:
                c = beta * float(aligned[f].sum())
                contributions[f] = c
                explained += c

        alpha_contrib = betas.alpha / TRADING_DAYS * len(aligned)
        residual = total - explained - alpha_contrib

        return FactorAttribution(total, alpha_contrib, contributions, residual)

    # ------------------------------------------------------------------
    # Rolling betas
    # ------------------------------------------------------------------

    def rolling_betas(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute rolling factor betas over time."""
        aligned = pd.concat([strategy_returns.rename("strat"), factor_returns], axis=1).dropna()
        cols = [c for c in FACTOR_NAMES if c in aligned.columns]
        w = self.rolling_window

        if len(aligned) < w + 10:
            return pd.DataFrame()

        records = []
        for end in range(w, len(aligned) + 1):
            chunk = aligned.iloc[end - w:end]
            y = chunk["strat"].values
            X = chunk[cols].values
            X_c = np.column_stack([np.ones(len(y)), X])
            try:
                b, _, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
            except np.linalg.LinAlgError:
                b = np.zeros(1 + len(cols))
            row = {"date": chunk.index[-1], "alpha": float(b[0]) * TRADING_DAYS}
            for i, f in enumerate(cols):
                row[f] = float(b[i + 1])
            records.append(row)

        return pd.DataFrame(records).set_index("date")

    # ------------------------------------------------------------------
    # Regime-conditional betas
    # ------------------------------------------------------------------

    def regime_profiles(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        regimes: pd.Series,
    ) -> List[RegimeFactorProfile]:
        """Factor betas per regime."""
        aligned = pd.concat([
            strategy_returns.rename("strat"),
            factor_returns,
            regimes.rename("regime"),
        ], axis=1).dropna()

        results: List[RegimeFactorProfile] = []
        for regime, grp in aligned.groupby("regime"):
            if len(grp) < 20:
                results.append(RegimeFactorProfile(str(regime), len(grp), {}, 0, 0))
                continue
            sub_strat = grp["strat"]
            sub_factors = grp[[c for c in FACTOR_NAMES if c in grp.columns]]
            fb = self.estimate_betas(sub_strat, sub_factors)
            results.append(RegimeFactorProfile(
                str(regime), len(grp), fb.betas, fb.alpha, fb.r_squared))

        return results

    # ------------------------------------------------------------------
    # Factor-neutral overlay
    # ------------------------------------------------------------------

    @staticmethod
    def neutral_overlay(
        betas: FactorBetas,
        portfolio_value: float = 100000,
        hedge_cost_bps: float = 5,
    ) -> NeutralOverlay:
        """Compute hedge ratios to neutralise factor exposures.

        For each significant beta, short/long the corresponding ETF
        to offset the exposure.
        """
        hedges: Dict[str, float] = {}
        residual: Dict[str, float] = {}

        for f, beta in betas.betas.items():
            if abs(beta) > 0.05:  # only hedge significant exposures
                # Shares needed: beta × portfolio / ETF_price (assume $100)
                shares = -beta * portfolio_value / 100
                hedges[f] = round(shares, 0)
                residual[f] = 0.0  # fully hedged
            else:
                hedges[f] = 0.0
                residual[f] = beta

        n_hedges = sum(1 for v in hedges.values() if v != 0)
        annual_cost = n_hedges * hedge_cost_bps / 10000 * portfolio_value

        return NeutralOverlay(hedges, residual, annual_cost / portfolio_value)

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        portfolio_value: float = 100000,
    ) -> FactorAnalysisResult:
        """Complete factor analysis."""
        betas = self.estimate_betas(strategy_returns, factor_returns)
        attribution = self.factor_attribution(strategy_returns, factor_returns, betas)
        rolling = self.rolling_betas(strategy_returns, factor_returns)
        regime_prof = self.regime_profiles(strategy_returns, factor_returns, regimes) if regimes is not None else []
        overlay = self.neutral_overlay(betas, portfolio_value)

        return FactorAnalysisResult(betas, attribution, rolling, regime_prof, overlay)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def generate_report(
        self,
        result: FactorAnalysisResult,
        output_path: str = "reports/factor_exposure.html",
    ) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fb = result.betas
        fa = result.attribution

        # Rolling beta SVG
        rolling_svg = ""
        if not result.rolling.empty and len(result.rolling) > 10:
            cols = [c for c in FACTOR_NAMES if c in result.rolling.columns]
            n = len(result.rolling)
            w, h = 750, 220
            pad = 55
            pw, ph = w - 2 * pad, h - 65
            all_vals = result.rolling[cols].values.flatten()
            vmin = float(np.nanmin(all_vals)) * 1.1
            vmax = float(np.nanmax(all_vals)) * 1.1
            if vmax <= vmin:
                vmax = vmin + 0.1
            def tx(i): return pad + i / max(n - 1, 1) * pw
            def ty(v): return 28 + (1 - (v - vmin) / (vmax - vmin)) * ph

            colors = {"market": "#dc2626", "size": "#2563eb", "value": "#059669",
                       "momentum": "#d97706", "low_vol": "#7c3aed", "quality": "#0891b2"}
            parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                      f'style="background:#fff;border:1px solid #e2e8f0;border-radius:6px;margin:.5rem 0">']
            parts.append(f'<text x="{w // 2}" y="16" text-anchor="middle" font-size="12" '
                          f'font-weight="bold" fill="#0f172a">Rolling Factor Betas ({self.rolling_window}d)</text>')
            zy = ty(0)
            parts.append(f'<line x1="{pad}" y1="{zy:.0f}" x2="{w - pad}" y2="{zy:.0f}" stroke="#e2e8f0"/>')
            for ci, col in enumerate(cols):
                vals = result.rolling[col].values
                c = colors.get(col, "#999")
                d = " ".join(f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(float(vals[i])):.1f}"
                              for i in range(n) if not np.isnan(vals[i]))
                parts.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="1.5"/>')
                lx = pad + ci * 100
                parts.append(f'<rect x="{lx}" y="{h - 16}" width="8" height="8" fill="{c}"/>')
                parts.append(f'<text x="{lx + 11}" y="{h - 8}" font-size="9" fill="#334155">{col}</text>')
            parts.append("</svg>")
            rolling_svg = "\n".join(parts)

        # Beta table
        beta_rows = []
        for f in FACTOR_NAMES:
            b = fb.betas.get(f, 0)
            t = fb.t_stats.get(f, 0)
            sig = abs(t) > 2
            color = "#059669" if sig else "#94a3b8"
            etf = FACTOR_ETFS.get(f, "")
            beta_rows.append(
                f"<tr><td style='text-align:left'>{f} ({etf})</td>"
                f"<td>{b:+.3f}</td><td>{t:+.2f}</td>"
                f"<td style='color:{color};font-weight:700'>{'SIG' if sig else 'ns'}</td></tr>")

        # Attribution table
        attr_rows = [
            f"<tr><td style='text-align:left'>{f}</td><td>{c * 10000:+.1f}</td></tr>"
            for f, c in fa.factor_contributions.items()
        ]

        # Overlay table
        overlay_rows = []
        if result.overlay:
            for f, shares in result.overlay.hedges.items():
                if shares != 0:
                    etf = FACTOR_ETFS.get(f, "")
                    overlay_rows.append(
                        f"<tr><td style='text-align:left'>{f} ({etf})</td>"
                        f"<td>{shares:+.0f}</td></tr>")

        # Regime table
        regime_rows = []
        for rp in result.regime_profiles:
            top_beta = max(rp.betas.items(), key=lambda x: abs(x[1]), default=("", 0))
            regime_rows.append(
                f"<tr><td style='text-align:left'>{rp.regime}</td><td>{rp.n_days}</td>"
                f"<td>{rp.alpha:+.1%}</td><td>{rp.r_squared:.2f}</td>"
                f"<td>{top_beta[0]}: {top_beta[1]:+.2f}</td></tr>")

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Factor Exposure</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; background: #fff; color: #1e293b; }}
h1 {{ color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; }}
h2 {{ color: #334155; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th {{ background: #f1f5f9; padding: 10px 12px; text-align: right; border-bottom: 2px solid #e2e8f0; }}
th:first-child {{ text-align: left; }}
td {{ padding: 9px 12px; text-align: right; border-bottom: 1px solid #f1f5f9; }}
td:first-child {{ text-align: left; }}
.card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.2rem; margin: 1rem 0; }}
</style></head><body>
<h1>EXP-1260-max: Factor Exposure Analysis</h1>
<div class="card">
<p><strong>Alpha:</strong> {fb.alpha:+.1%}/yr (t={fb.alpha_t_stat:.2f}) |
<strong>R²:</strong> {fb.r_squared:.2f} |
<strong>Residual Vol:</strong> {fb.residual_vol:.1%}</p>
</div>

{rolling_svg}

<h2>Factor Betas</h2>
<table><tr><th style='text-align:left'>Factor (ETF)</th><th>Beta</th><th>t-stat</th><th>Sig?</th></tr>
{''.join(beta_rows)}</table>

<h2>Return Attribution (bps)</h2>
<table><tr><th style='text-align:left'>Source</th><th>Contribution</th></tr>
<tr><td style='text-align:left'><strong>Alpha</strong></td><td>{fa.alpha_contribution * 10000:+.1f}</td></tr>
{''.join(attr_rows)}
<tr><td style='text-align:left'>Residual</td><td>{fa.residual * 10000:+.1f}</td></tr>
<tr><td style='text-align:left'><strong>Total</strong></td><td>{fa.total_return * 10000:+.1f}</td></tr></table>

{self._overlay_html(overlay_rows, result.overlay)}
{self._regime_html(regime_rows)}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        return str(path)

    @staticmethod
    def _overlay_html(rows: List[str], overlay: Optional[NeutralOverlay]) -> str:
        if not rows:
            return ""
        cost = f"{overlay.cost_annual_pct:.2%}" if overlay else "0%"
        joined = "".join(rows)
        return (f'<h2>Factor-Neutral Overlay</h2>'
                f'<table><tr><th style="text-align:left">Factor</th>'
                f'<th>Hedge (shares)</th></tr>{joined}</table>'
                f'<p>Annual cost: {cost}</p>')

    @staticmethod
    def _regime_html(rows: List[str]) -> str:
        if not rows:
            return ""
        joined = "".join(rows)
        return (f'<h2>Regime Profiles</h2>'
                f'<table><tr><th style="text-align:left">Regime</th>'
                f'<th>Days</th><th>Alpha</th><th>R²</th><th>Top Beta</th></tr>'
                f'{joined}</table>')
