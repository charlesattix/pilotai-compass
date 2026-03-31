"""
compass/greeks_sensitivity.py — Options Greeks sensitivity analysis for
credit spreads.

Computes how credit-spread P&L responds to changes in underlying price
(delta), implied volatility (vega), time decay (theta), and interest
rates (rho) using a closed-form Black-Scholes approximation.

Scenario matrix: sweeps underlying ±5% in 0.5% steps × IV ±30% in 5%
steps × DTE 0-45 in 5-day steps → ~2,800 cells per spread.

Usage::

    from compass.greeks_sensitivity import GreeksSensitivityAnalyzer

    analyzer = GreeksSensitivityAnalyzer()
    analyzer.fit(trades_df)
    analyzer.generate_report("reports/greeks_sensitivity.html")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Black-Scholes helpers ────────────────────────────────────────────────

_SQRT2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT2PI


def bs_put_price(
    S: float, K: float, T: float, sigma: float, r: float = 0.045,
) -> float:
    """Black-Scholes European put price.  T in years, sigma annualised."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_call_price(
    S: float, K: float, T: float, sigma: float, r: float = 0.045,
) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def put_spread_value(
    S: float, K_short: float, K_long: float,
    T: float, sigma: float, r: float = 0.045,
) -> float:
    """Value of a bull-put credit spread (short higher-strike put, long lower)."""
    return bs_put_price(S, K_short, T, sigma, r) - bs_put_price(S, K_long, T, sigma, r)


def call_spread_value(
    S: float, K_short: float, K_long: float,
    T: float, sigma: float, r: float = 0.045,
) -> float:
    """Value of a bear-call credit spread (short lower-strike call, long higher)."""
    return bs_call_price(S, K_short, T, sigma, r) - bs_call_price(S, K_long, T, sigma, r)


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class GreeksSnapshot:
    """Greeks for a single spread at a single point."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0   # per day
    vega: float = 0.0     # per 1% IV move
    rho: float = 0.0      # per 1% rate move


@dataclass
class ScenarioCell:
    """One cell in the scenario matrix."""
    price_pct: float      # underlying change from entry (%)
    iv_pct: float          # IV change from entry (%)
    dte: int
    pnl_pct: float         # P&L as % of max profit


@dataclass
class OptimalEntry:
    """Conditions historically associated with best outcomes."""
    regime: str
    iv_rank_range: Tuple[float, float]
    dte_range: Tuple[int, int]
    otm_pct_range: Tuple[float, float]
    avg_return_pct: float
    win_rate: float
    n_trades: int


@dataclass
class GreekProfile:
    """Aggregate Greek profile for a regime."""
    regime: str
    n_trades: int
    avg_delta: float
    avg_theta_day: float
    avg_vega: float
    avg_gamma: float


@dataclass
class SensitivitySummary:
    """Full analysis output."""
    n_trades: int = 0
    scenario_matrix: List[ScenarioCell] = field(default_factory=list)
    optimal_entries: List[OptimalEntry] = field(default_factory=list)
    regime_profiles: List[GreekProfile] = field(default_factory=list)
    greeks_at_entry: Optional[GreeksSnapshot] = None


# ── Greeks computation ───────────────────────────────────────────────────


def compute_greeks(
    S: float, K_short: float, K_long: float,
    T: float, sigma: float, r: float = 0.045,
    spread_type: str = "bull_put",
) -> GreeksSnapshot:
    """Compute approximate Greeks for a credit spread via finite differences."""
    dS = S * 0.001
    dT = 1.0 / 252.0
    dSigma = 0.01
    dR = 0.01

    val_fn = put_spread_value if "put" in spread_type.lower() else call_spread_value

    v0 = val_fn(S, K_short, K_long, T, sigma, r)
    v_up = val_fn(S + dS, K_short, K_long, T, sigma, r)
    v_dn = val_fn(S - dS, K_short, K_long, T, sigma, r)

    delta = (v_up - v_dn) / (2 * dS)
    gamma = (v_up - 2 * v0 + v_dn) / (dS ** 2)

    if T > dT:
        theta = (val_fn(S, K_short, K_long, T - dT, sigma, r) - v0) / (-dT)
        theta /= 252  # per calendar day
    else:
        theta = 0.0

    vega = (val_fn(S, K_short, K_long, T, sigma + dSigma, r) - v0) / dSigma
    rho = (val_fn(S, K_short, K_long, T, sigma, r + dR) - v0) / dR

    return GreeksSnapshot(
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        theta=round(theta, 6),
        vega=round(vega, 6),
        rho=round(rho, 6),
    )


def build_scenario_matrix(
    S: float, K_short: float, K_long: float,
    sigma: float, credit: float, r: float = 0.045,
    spread_type: str = "bull_put",
    price_range_pct: float = 5.0,
    price_step_pct: float = 0.5,
    iv_range_pct: float = 30.0,
    iv_step_pct: float = 5.0,
    dte_max: int = 45,
    dte_step: int = 5,
) -> List[ScenarioCell]:
    """Build the scenario matrix sweeping price, IV, and DTE."""
    val_fn = put_spread_value if "put" in spread_type.lower() else call_spread_value
    max_profit = credit * 100
    cells: List[ScenarioCell] = []

    price_steps = np.arange(-price_range_pct, price_range_pct + 0.01, price_step_pct)
    iv_steps = np.arange(-iv_range_pct, iv_range_pct + 0.01, iv_step_pct)
    dte_steps = list(range(0, dte_max + 1, dte_step))
    if 0 not in dte_steps:
        dte_steps.insert(0, 0)

    entry_value = val_fn(S, K_short, K_long, dte_max / 252.0, sigma, r)

    for dp in price_steps:
        for dv in iv_steps:
            for dte in dte_steps:
                S_new = S * (1 + dp / 100)
                sigma_new = max(0.01, sigma * (1 + dv / 100))
                T_new = max(dte, 0) / 252.0
                new_val = val_fn(S_new, K_short, K_long, T_new, sigma_new, r)
                pnl = (entry_value - new_val) * 100  # per contract
                pnl_pct = (pnl / max_profit * 100) if max_profit > 0 else 0.0
                cells.append(ScenarioCell(
                    price_pct=round(dp, 1),
                    iv_pct=round(dv, 1),
                    dte=dte,
                    pnl_pct=round(pnl_pct, 1),
                ))
    return cells


# ── Analyzer ─────────────────────────────────────────────────────────────


class GreeksSensitivityAnalyzer:
    """Full Greeks sensitivity analysis from historical trade data."""

    def __init__(self) -> None:
        self._summary: Optional[SensitivitySummary] = None
        self._fitted = False

    def fit(self, trades_df: pd.DataFrame) -> "GreeksSensitivityAnalyzer":
        """Analyse Greeks sensitivity from trade data.

        Expects columns: spy_price, short_strike, spread_width, net_credit,
        dte_at_entry, vix, iv_rank, otm_pct, regime, pnl, win.
        """
        df = trades_df.copy()
        if len(df) == 0:
            self._summary = SensitivitySummary()
            self._fitted = True
            return self

        # Fill defaults
        for col, default in [
            ("spy_price", 450.0), ("short_strike", 440.0), ("spread_width", 5.0),
            ("net_credit", 0.65), ("dte_at_entry", 21), ("vix", 20.0),
            ("iv_rank", 50.0), ("otm_pct", 3.0), ("regime", "bull"),
            ("pnl", 0.0), ("win", 0), ("spread_type", "bull_put"),
        ]:
            if col not in df.columns:
                df[col] = default

        df["vix"] = pd.to_numeric(df["vix"], errors="coerce").fillna(20.0)
        df["iv_rank"] = pd.to_numeric(df["iv_rank"], errors="coerce").fillna(50.0)
        df["dte_at_entry"] = pd.to_numeric(df["dte_at_entry"], errors="coerce").fillna(21).astype(int)

        # Compute Greeks at entry for median trade
        med = df.median(numeric_only=True)
        S = float(med.get("spy_price", 450))
        K_short = float(med.get("short_strike", 440))
        K_long = K_short - float(med.get("spread_width", 5))
        sigma = float(med.get("vix", 20)) / 100.0 * 1.2  # rough VIX→IV conversion
        T = float(med.get("dte_at_entry", 21)) / 252.0
        credit = float(med.get("net_credit", 0.65))

        greeks_entry = compute_greeks(S, K_short, K_long, T, sigma, spread_type="bull_put")

        # Scenario matrix for the median trade
        scenario = build_scenario_matrix(
            S, K_short, K_long, sigma, credit,
            spread_type="bull_put",
        )

        # Optimal entry conditions
        optimal = self._find_optimal_entries(df)

        # Per-regime Greek profiles
        profiles = self._compute_regime_profiles(df)

        self._summary = SensitivitySummary(
            n_trades=len(df),
            scenario_matrix=scenario,
            optimal_entries=optimal,
            regime_profiles=profiles,
            greeks_at_entry=greeks_entry,
        )
        self._fitted = True
        return self

    def summary(self) -> SensitivitySummary:
        if not self._fitted:
            return SensitivitySummary()
        return self._summary

    def generate_report(self, path: Optional[str] = None) -> str:
        """Generate HTML report."""
        if not self._fitted:
            return "<html><body><p>No data.</p></body></html>"
        html = self._render_html()
        if path:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html)
        return html

    # ── Private ───────────────────────────────────────────────────────

    def _find_optimal_entries(self, df: pd.DataFrame) -> List[OptimalEntry]:
        results = []
        for regime in sorted(df["regime"].dropna().unique()):
            sub = df[df["regime"] == regime]
            if len(sub) < 5:
                continue

            winners = sub[sub["win"] == 1]
            if len(winners) < 3:
                continue

            iv_lo = float(winners["iv_rank"].quantile(0.25))
            iv_hi = float(winners["iv_rank"].quantile(0.75))
            dte_lo = int(winners["dte_at_entry"].quantile(0.25))
            dte_hi = int(winners["dte_at_entry"].quantile(0.75))
            otm_lo = float(winners["otm_pct"].quantile(0.25)) if "otm_pct" in winners.columns else 1.0
            otm_hi = float(winners["otm_pct"].quantile(0.75)) if "otm_pct" in winners.columns else 5.0

            results.append(OptimalEntry(
                regime=str(regime),
                iv_rank_range=(round(iv_lo, 1), round(iv_hi, 1)),
                dte_range=(dte_lo, dte_hi),
                otm_pct_range=(round(otm_lo, 1), round(otm_hi, 1)),
                avg_return_pct=round(float(sub["pnl"].mean() / max(sub["net_credit"].mean() * 100, 1) * 100), 1),
                win_rate=round(float(sub["win"].mean()) * 100, 1),
                n_trades=len(sub),
            ))
        return results

    def _compute_regime_profiles(self, df: pd.DataFrame) -> List[GreekProfile]:
        profiles = []
        for regime in sorted(df["regime"].dropna().unique()):
            sub = df[df["regime"] == regime]
            if len(sub) < 3:
                continue

            deltas, thetas, vegas, gammas = [], [], [], []
            for _, row in sub.iterrows():
                S = float(row.get("spy_price", 450))
                K_short = float(row.get("short_strike", 440))
                K_long = K_short - float(row.get("spread_width", 5))
                sigma = max(float(row.get("vix", 20)) / 100 * 1.2, 0.05)
                T = max(int(row.get("dte_at_entry", 21)), 1) / 252.0
                sp = str(row.get("spread_type", "bull_put"))
                g = compute_greeks(S, K_short, K_long, T, sigma, spread_type=sp)
                deltas.append(g.delta)
                thetas.append(g.theta)
                vegas.append(g.vega)
                gammas.append(g.gamma)

            profiles.append(GreekProfile(
                regime=str(regime),
                n_trades=len(sub),
                avg_delta=round(float(np.mean(deltas)), 4),
                avg_theta_day=round(float(np.mean(thetas)), 4),
                avg_vega=round(float(np.mean(vegas)), 4),
                avg_gamma=round(float(np.mean(gammas)), 6),
            ))
        return profiles

    def _render_html(self) -> str:
        s = self._summary
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        g = s.greeks_at_entry or GreeksSnapshot()

        # Cards
        cards = (
            f'<div class="cards">'
            f'<div class="card"><div class="ct">Trades</div><div class="cv">{s.n_trades:,}</div></div>'
            f'<div class="card"><div class="ct">Delta</div><div class="cv">{g.delta:.4f}</div></div>'
            f'<div class="card"><div class="ct">Theta/day</div><div class="cv">{g.theta:.4f}</div></div>'
            f'<div class="card"><div class="ct">Vega</div><div class="cv">{g.vega:.4f}</div></div>'
            f'<div class="card"><div class="ct">Gamma</div><div class="cv">{g.gamma:.6f}</div></div>'
            f'<div class="card"><div class="ct">Scenarios</div><div class="cv">{len(s.scenario_matrix):,}</div></div>'
            f'</div>'
        )

        # Scenario heatmap for DTE=20 (price x IV)
        heatmap = self._render_heatmap(s.scenario_matrix, target_dte=20)

        # Optimal entry table
        opt_rows = ""
        for o in s.optimal_entries:
            opt_rows += (
                f'<tr><td style="font-weight:600">{o.regime}</td>'
                f'<td>{o.iv_rank_range[0]:.0f}-{o.iv_rank_range[1]:.0f}</td>'
                f'<td>{o.dte_range[0]}-{o.dte_range[1]}</td>'
                f'<td>{o.otm_pct_range[0]:.1f}-{o.otm_pct_range[1]:.1f}%</td>'
                f'<td>{o.win_rate:.0f}%</td>'
                f'<td>{o.avg_return_pct:+.1f}%</td>'
                f'<td>{o.n_trades}</td></tr>'
            )
        opt_table = (
            f'<table><thead><tr><th>Regime</th><th>IV Rank</th><th>DTE</th>'
            f'<th>OTM%</th><th>Win Rate</th><th>Avg Return</th><th>N</th></tr></thead>'
            f'<tbody>{opt_rows}</tbody></table>'
        ) if opt_rows else "<p>Insufficient data.</p>"

        # Regime profiles
        prof_rows = ""
        for p in s.regime_profiles:
            prof_rows += (
                f'<tr><td style="font-weight:600">{p.regime}</td>'
                f'<td>{p.n_trades}</td>'
                f'<td>{p.avg_delta:.4f}</td>'
                f'<td>{p.avg_theta_day:.4f}</td>'
                f'<td>{p.avg_vega:.4f}</td>'
                f'<td>{p.avg_gamma:.6f}</td></tr>'
            )
        prof_table = (
            f'<table><thead><tr><th>Regime</th><th>N</th><th>Avg Delta</th>'
            f'<th>Avg Theta/d</th><th>Avg Vega</th><th>Avg Gamma</th></tr></thead>'
            f'<tbody>{prof_rows}</tbody></table>'
        )

        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Greeks Sensitivity Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:#f8fafc;color:#1e293b;line-height:1.5;padding:24px;max-width:1200px;margin:0 auto}}
h1{{font-size:1.6em;font-weight:700;margin-bottom:4px}}
h2{{font-size:1.15em;font-weight:600;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}}
.sub{{color:#64748b;font-size:0.9em;margin-bottom:20px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;min-width:130px;flex:1}}
.ct{{font-size:0.75em;color:#64748b;text-transform:uppercase;letter-spacing:.5px}}
.cv{{font-size:1.4em;font-weight:700}}
table{{border-collapse:collapse;width:100%;font-size:0.85em;margin-bottom:16px}}
th{{background:#f1f5f9;padding:6px 8px;text-align:left;font-weight:600;border-bottom:2px solid #e2e8f0}}
td{{padding:5px 8px;border-bottom:1px solid #f1f5f9;text-align:center}}
.hm td{{padding:3px 4px;font-size:0.78em;min-width:42px}}
hr{{margin:28px 0;border:none;border-top:1px solid #e2e8f0}}
</style></head><body>

<h1>Greeks Sensitivity Report</h1>
<p class="sub">Median-trade analysis &middot; {len(s.scenario_matrix):,} scenario cells &middot; {now}</p>
{cards}

<h2>P&amp;L Scenario Heatmap (DTE=20)</h2>
<p style="font-size:0.82em;color:#64748b;margin-bottom:8px">
Rows = underlying price change, columns = IV change. Values = P&amp;L as % of max profit.
Green = profit, red = loss.</p>
{heatmap}

<h2>Optimal Entry Conditions (from Winners)</h2>
<p style="font-size:0.82em;color:#64748b;margin-bottom:8px">
IQR of IV rank, DTE, and OTM% among winning trades per regime.</p>
{opt_table}

<h2>Per-Regime Greek Profiles</h2>
<p style="font-size:0.82em;color:#64748b;margin-bottom:8px">
Average Greeks at entry across historical trades per regime.</p>
{prof_table}

<hr><p style="font-size:0.75em;color:#94a3b8">Generated by <code>compass/greeks_sensitivity.py</code></p>
</body></html>"""

    def _render_heatmap(self, cells: List[ScenarioCell], target_dte: int = 20) -> str:
        filtered = [c for c in cells if c.dte == target_dte]
        if not filtered:
            # Fall back to closest DTE
            dtes = sorted(set(c.dte for c in cells))
            closest = min(dtes, key=lambda d: abs(d - target_dte)) if dtes else 0
            filtered = [c for c in cells if c.dte == closest]
        if not filtered:
            return "<p>No scenario data.</p>"

        prices = sorted(set(c.price_pct for c in filtered))
        ivs = sorted(set(c.iv_pct for c in filtered))
        lookup = {(c.price_pct, c.iv_pct): c.pnl_pct for c in filtered}

        header = '<th>Price\\IV</th>' + ''.join(f'<th>{v:+.0f}%</th>' for v in ivs)
        rows = ""
        for p in prices:
            cells_html = f'<td style="font-weight:600;text-align:left">{p:+.1f}%</td>'
            for v in ivs:
                val = lookup.get((p, v), 0)
                if val > 50:
                    bg = "#bbf7d0"
                elif val > 0:
                    bg = "#dcfce7"
                elif val > -50:
                    bg = "#fef9c3"
                elif val > -100:
                    bg = "#fecaca"
                else:
                    bg = "#fca5a5"
                cells_html += f'<td style="background:{bg}">{val:+.0f}</td>'
            rows += f'<tr>{cells_html}</tr>'

        return f'<table class="hm"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table>'
