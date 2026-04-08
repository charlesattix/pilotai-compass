#!/usr/bin/env python3
"""
Deep Correlation Matrix & Portfolio Optimization.

Builds NxN correlation matrix for all real-data strategies, then computes:
  1. Full heatmap with clustering
  2. Minimum-variance portfolio
  3. Maximum-Sharpe portfolio
  4. Risk parity portfolio
  5. Risk contribution per strategy
  6. Efficient frontier

Output: reports/strategy_correlation_deep.html + .json
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.correlation_analyzer import (
    STRATEGIES, StrategySpec, build_daily_returns, compute_correlation_matrix,
    TRADING_DAYS, N_YEARS,
)

REPORT_PATH = ROOT / "reports" / "strategy_correlation_deep.html"
JSON_PATH = ROOT / "reports" / "strategy_correlation_deep.json"
RF = 0.045  # risk-free rate


# ═══════════════════════════════════════════════════════════════════════════
# Add EXP-1660 to the strategy catalog
# ═══════════════════════════════════════════════════════════════════════════

STRATEGIES["EXP-1660 Vol Premium"] = StrategySpec(
    "EXP-1660 Vol Premium", "1660-VRP",
    cagr=0.022, sharpe=1.83, max_dd=0.017, spy_corr=-0.70, verdict="PROMISING",
    yearly_rets={2020: 0.0, 2021: 0.005, 2022: 0.0, 2023: 0.025, 2024: 0.046, 2025: 0.0},
)


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio optimization (analytical, no scipy needed)
# ═══════════════════════════════════════════════════════════════════════════

def _portfolio_stats(
    weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
) -> Tuple[float, float, float]:
    """Return (return, vol, sharpe) for given weights."""
    ret = float(weights @ mu)
    var = float(weights @ cov @ weights)
    vol = math.sqrt(max(var, 1e-16))
    sharpe = (ret - RF) / vol if vol > 1e-8 else 0.0
    return ret, vol, sharpe


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal risk contribution of each asset."""
    port_var = float(weights @ cov @ weights)
    port_vol = math.sqrt(max(port_var, 1e-16))
    marginal = (cov @ weights) / port_vol
    rc = weights * marginal
    return rc / rc.sum() if rc.sum() > 1e-12 else rc


def _min_variance(mu: np.ndarray, cov: np.ndarray, n: int,
                  min_w: float = 0.0) -> np.ndarray:
    """Find min-variance portfolio via grid search."""
    best_vol = 1e9
    best_w = np.ones(n) / n
    rng = np.random.RandomState(42)
    for _ in range(50_000):
        w = rng.dirichlet(np.ones(n))
        if min_w > 0:
            w = np.clip(w, min_w, 1.0)
            w /= w.sum()
        _, vol, _ = _portfolio_stats(w, mu, cov)
        if vol < best_vol:
            best_vol = vol
            best_w = w.copy()
    return best_w


def _max_sharpe(mu: np.ndarray, cov: np.ndarray, n: int,
                min_w: float = 0.0) -> np.ndarray:
    """Find max-Sharpe portfolio via grid search."""
    best_sharpe = -1e9
    best_w = np.ones(n) / n
    rng = np.random.RandomState(77)
    for _ in range(50_000):
        w = rng.dirichlet(np.ones(n))
        if min_w > 0:
            w = np.clip(w, min_w, 1.0)
            w /= w.sum()
        _, _, sharpe = _portfolio_stats(w, mu, cov)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w.copy()
    return best_w


def _risk_parity(cov: np.ndarray, n: int) -> np.ndarray:
    """Inverse-vol risk parity weights."""
    vols = np.sqrt(np.diag(cov))
    inv_vol = 1.0 / np.where(vols > 1e-10, vols, 1e-10)
    w = inv_vol / inv_vol.sum()
    return w


def _efficient_frontier(
    mu: np.ndarray, cov: np.ndarray, n: int, n_points: int = 30,
) -> List[Dict]:
    """Generate points on the efficient frontier via target-return sweeps."""
    rng = np.random.RandomState(123)
    min_ret = float(mu.min())
    max_ret = float(mu.max())
    targets = np.linspace(max(min_ret, 0), max_ret, n_points)

    points = []
    for target in targets:
        best_vol = 1e9
        best_w = None
        for _ in range(20_000):
            w = rng.dirichlet(np.ones(n))
            r, v, s = _portfolio_stats(w, mu, cov)
            if r >= target - 0.005 and v < best_vol:
                best_vol = v
                best_w = w.copy()
        if best_w is not None:
            r, v, s = _portfolio_stats(best_w, mu, cov)
            points.append({"return": round(r, 4), "vol": round(v, 4), "sharpe": round(s, 2)})

    return points


# ═══════════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("DEEP CORRELATION MATRIX & PORTFOLIO OPTIMIZATION")
    print("=" * 60)

    # Build returns
    print("\n[1/6] Building daily return series...")
    returns = build_daily_returns()
    corr_matrix, names = compute_correlation_matrix(returns)
    n = len(names)
    short_names = [STRATEGIES[nm].short for nm in names]
    print(f"  {n} strategies, {len(returns[names[0]])} daily returns")

    # Annualized return and covariance
    ret_matrix = np.column_stack([returns[nm] for nm in names])
    mu = np.array([ret_matrix[:, i].mean() * TRADING_DAYS for i in range(n)])
    cov = np.cov(ret_matrix, rowvar=False) * TRADING_DAYS

    # Strategy summary
    print("\n[2/6] Strategy summary:")
    for i, nm in enumerate(names):
        s = STRATEGIES[nm]
        print(f"  {s.short:10s}  μ={mu[i]:+7.2%}  σ={math.sqrt(cov[i,i]):6.2%}  "
              f"Sharpe={(mu[i]-RF)/math.sqrt(cov[i,i]):5.2f}  {s.verdict}")

    # Key correlation pairs
    print("\n[3/6] Correlation extremes:")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((corr_matrix[i, j], short_names[i], short_names[j]))
    pairs.sort()
    print("  Most negative:")
    for c, a, b in pairs[:5]:
        print(f"    {a} ↔ {b}: {c:+.3f}")
    print("  Most positive:")
    for c, a, b in pairs[-5:]:
        print(f"    {a} ↔ {b}: {c:+.3f}")
    print(f"  Avg pairwise: {np.mean([p[0] for p in pairs]):+.3f}")

    # Min-variance portfolio
    print("\n[4/6] Optimizing portfolios...")
    print("  Min-variance (50K samples)...")
    w_minvar = _min_variance(mu, cov, n)
    r_mv, v_mv, s_mv = _portfolio_stats(w_minvar, mu, cov)
    rc_mv = _risk_contributions(w_minvar, cov)
    print(f"    Return={r_mv:.2%}, Vol={v_mv:.2%}, Sharpe={s_mv:.2f}")

    # Max-Sharpe portfolio
    print("  Max-Sharpe (50K samples)...")
    w_maxsharpe = _max_sharpe(mu, cov, n)
    r_ms, v_ms, s_ms = _portfolio_stats(w_maxsharpe, mu, cov)
    rc_ms = _risk_contributions(w_maxsharpe, cov)
    print(f"    Return={r_ms:.2%}, Vol={v_ms:.2%}, Sharpe={s_ms:.2f}")

    # Risk parity
    print("  Risk parity (inverse vol)...")
    w_rp = _risk_parity(cov, n)
    r_rp, v_rp, s_rp = _portfolio_stats(w_rp, mu, cov)
    rc_rp = _risk_contributions(w_rp, cov)
    print(f"    Return={r_rp:.2%}, Vol={v_rp:.2%}, Sharpe={s_rp:.2f}")

    # Equal weight
    w_ew = np.ones(n) / n
    r_ew, v_ew, s_ew = _portfolio_stats(w_ew, mu, cov)
    rc_ew = _risk_contributions(w_ew, cov)

    # Efficient frontier
    print("\n[5/6] Computing efficient frontier (30 points)...")
    frontier = _efficient_frontier(mu, cov, n, n_points=30)
    print(f"  {len(frontier)} frontier points computed")

    # Generate report
    print("\n[6/6] Generating report...")
    data = {
        "n_strategies": n,
        "names": names,
        "short_names": short_names,
        "correlation_matrix": corr_matrix.tolist(),
        "annualized_returns": mu.tolist(),
        "annualized_vols": [math.sqrt(cov[i, i]) for i in range(n)],
        "portfolios": {
            "equal_weight": {
                "weights": {names[i]: round(float(w_ew[i]), 4) for i in range(n)},
                "return": round(r_ew, 4), "vol": round(v_ew, 4), "sharpe": round(s_ew, 2),
                "risk_contrib": {names[i]: round(float(rc_ew[i]), 4) for i in range(n)},
            },
            "min_variance": {
                "weights": {names[i]: round(float(w_minvar[i]), 4) for i in range(n)},
                "return": round(r_mv, 4), "vol": round(v_mv, 4), "sharpe": round(s_mv, 2),
                "risk_contrib": {names[i]: round(float(rc_mv[i]), 4) for i in range(n)},
            },
            "max_sharpe": {
                "weights": {names[i]: round(float(w_maxsharpe[i]), 4) for i in range(n)},
                "return": round(r_ms, 4), "vol": round(v_ms, 4), "sharpe": round(s_ms, 2),
                "risk_contrib": {names[i]: round(float(rc_ms[i]), 4) for i in range(n)},
            },
            "risk_parity": {
                "weights": {names[i]: round(float(w_rp[i]), 4) for i in range(n)},
                "return": round(r_rp, 4), "vol": round(v_rp, 4), "sharpe": round(s_rp, 2),
                "risk_contrib": {names[i]: round(float(rc_rp[i]), 4) for i in range(n)},
            },
        },
        "efficient_frontier": frontier,
        "strategy_details": {
            nm: {
                "short": STRATEGIES[nm].short,
                "cagr": STRATEGIES[nm].cagr,
                "sharpe": STRATEGIES[nm].sharpe,
                "max_dd": STRATEGIES[nm].max_dd,
                "spy_corr": STRATEGIES[nm].spy_corr,
                "verdict": STRATEGIES[nm].verdict,
                "ann_return": round(float(mu[i]), 4),
                "ann_vol": round(math.sqrt(float(cov[i, i])), 4),
            }
            for i, nm in enumerate(names)
        },
    }

    html = generate_html(data)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  HTML: {REPORT_PATH}")

    JSON_PATH.write_text(json.dumps(data, indent=2, default=str))
    print(f"  JSON: {JSON_PATH}")

    # Print portfolio comparison
    print("\n" + "=" * 60)
    print("PORTFOLIO COMPARISON")
    print("=" * 60)
    for pname in ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]:
        p = data["portfolios"][pname]
        top3 = sorted(p["weights"].items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"{STRATEGIES[k].short} {v:.0%}" for k, v in top3)
        print(f"  {pname:15s}: Ret={p['return']:+.2%}  Vol={p['vol']:.2%}  Sharpe={p['sharpe']:.2f}  Top3: {top3_str}")


# ═══════════════════════════════════════════════════════════════════════════
# HTML Report (white background)
# ═══════════════════════════════════════════════════════════════════════════

def _hc(v: float) -> str:
    """Heatmap color: blue (neg) → white (0) → red (pos)."""
    if abs(v) > 0.95:
        return "#1e293b"
    if v > 0:
        g = int(255 * (1 - min(v, 1)))
        return f"rgb(255,{g},{g})"
    else:
        r = int(255 * (1 + max(v, -1)))
        return f"rgb({r},{r},255)"


def generate_html(data: Dict) -> str:
    n = data["n_strategies"]
    names = data["names"]
    short = data["short_names"]
    corr = np.array(data["correlation_matrix"])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Heatmap
    hdr = "".join(f'<th style="writing-mode:vertical-lr;text-orientation:mixed;padding:3px 1px;font-size:.6rem;white-space:nowrap">{s}</th>' for s in short)
    hm_rows = ""
    for i in range(n):
        cells = f'<td style="text-align:left;font-size:.68rem;font-weight:600;white-space:nowrap">{short[i]}</td>'
        for j in range(n):
            v = corr[i][j]
            bg = _hc(v)
            tc = "#fff" if abs(v) > 0.5 or i == j else "#111"
            cells += f'<td style="background:{bg};color:{tc};text-align:center;font-size:.6rem;padding:2px;min-width:32px">{v:+.2f}</td>'
        hm_rows += f"<tr>{cells}</tr>\n"

    # Strategy overview table
    strat_rows = ""
    for i, nm in enumerate(names):
        s = STRATEGIES[nm]
        vc = {"LIVE": "#059669", "PROMISING": "#2563eb", "MARGINAL": "#d97706"}.get(s.verdict, "#6b7280")
        strat_rows += (
            f'<tr><td style="text-align:left"><span style="color:{vc};font-weight:600">{s.short}</span></td>'
            f'<td style="color:{vc}">{s.verdict}</td>'
            f'<td style="color:{"#059669" if s.cagr>0 else "#dc2626"}">{s.cagr:+.1%}</td>'
            f'<td>{s.sharpe:.2f}</td><td>{s.max_dd:.1%}</td>'
            f'<td style="color:{"#059669" if abs(s.spy_corr)<0.3 else "#d97706"}">{s.spy_corr:+.2f}</td>'
            f'<td>{data["annualized_returns"][i]:+.2%}</td>'
            f'<td>{data["annualized_vols"][i]:.2%}</td></tr>\n'
        )

    # Portfolio comparison table
    port_rows = ""
    for pname, label in [("equal_weight", "Equal Weight"), ("min_variance", "Min Variance"),
                          ("max_sharpe", "Max Sharpe"), ("risk_parity", "Risk Parity")]:
        p = data["portfolios"][pname]
        sc = "#059669" if p["sharpe"] > 3 else ("#d97706" if p["sharpe"] > 1 else "#6b7280")
        port_rows += (
            f'<tr><td style="text-align:left"><strong>{label}</strong></td>'
            f'<td style="color:{"#059669" if p["return"]>0 else "#dc2626"}">{p["return"]:+.2%}</td>'
            f'<td>{p["vol"]:.2%}</td>'
            f'<td style="color:{sc}"><strong>{p["sharpe"]:.2f}</strong></td></tr>\n'
        )

    # Weight tables per portfolio
    weight_sections = ""
    for pname, label in [("max_sharpe", "Max Sharpe"), ("min_variance", "Min Variance"),
                          ("risk_parity", "Risk Parity")]:
        p = data["portfolios"][pname]
        w_rows = ""
        sorted_w = sorted(p["weights"].items(), key=lambda x: -x[1])
        for nm, w in sorted_w:
            if w < 0.005:
                continue
            rc = p["risk_contrib"].get(nm, 0)
            s = STRATEGIES[nm]
            bar_w = min(w * 300, 200)
            rc_bar = min(abs(rc) * 300, 200)
            w_rows += (
                f'<tr><td style="text-align:left">{s.short}</td>'
                f'<td>{w:.1%}</td>'
                f'<td><div style="background:#3b82f6;height:8px;width:{bar_w}px;border-radius:4px"></div></td>'
                f'<td>{rc:.1%}</td>'
                f'<td><div style="background:#f59e0b;height:8px;width:{rc_bar}px;border-radius:4px"></div></td>'
                f'</tr>\n'
            )
        weight_sections += f"""
        <div class="section-card">
        <h3>{label} Portfolio (Sharpe {p["sharpe"]:.2f}, Return {p["return"]:+.2%}, Vol {p["vol"]:.2%})</h3>
        <table>
        <thead><tr><th>Strategy</th><th>Weight</th><th></th><th>Risk %</th><th></th></tr></thead>
        <tbody>{w_rows}</tbody></table>
        </div>"""

    # Most/least correlated pairs
    pair_list = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_list.append((corr[i][j], short[i], short[j]))
    pair_list.sort()
    neg_rows = ""
    for c, a, b in pair_list[:7]:
        clr = "#2563eb" if c < -0.2 else "#6b7280"
        neg_rows += f'<tr><td>{a} ↔ {b}</td><td style="color:{clr}"><strong>{c:+.3f}</strong></td></tr>\n'
    pos_rows = ""
    for c, a, b in pair_list[-7:]:
        clr = "#dc2626" if c > 0.5 else "#6b7280"
        pos_rows += f'<tr><td>{a} ↔ {b}</td><td style="color:{clr}"><strong>{c:+.3f}</strong></td></tr>\n'

    # Efficient frontier SVG
    if data["efficient_frontier"]:
        pts = data["efficient_frontier"]
        min_vol = min(p["vol"] for p in pts) if pts else 0
        max_vol = max(p["vol"] for p in pts) if pts else 1
        min_ret = min(p["return"] for p in pts) if pts else 0
        max_ret = max(p["return"] for p in pts) if pts else 1
        vol_range = max(max_vol - min_vol, 0.001)
        ret_range = max(max_ret - min_ret, 0.001)

        svg_w, svg_h = 500, 300
        pad = 40
        def _sx(v): return pad + (v - min_vol) / vol_range * (svg_w - 2 * pad)
        def _sy(r): return svg_h - pad - (r - min_ret) / ret_range * (svg_h - 2 * pad)

        circles = ""
        for p in pts:
            x, y = _sx(p["vol"]), _sy(p["return"])
            circles += f'<circle cx="{x:.0f}" cy="{y:.0f}" r="3" fill="#3b82f6" opacity="0.7"/>'

        # Plot the 4 portfolios
        port_dots = ""
        colors = {"max_sharpe": "#059669", "min_variance": "#2563eb",
                  "risk_parity": "#f59e0b", "equal_weight": "#6b7280"}
        labels_map = {"max_sharpe": "Max SR", "min_variance": "MinVar",
                      "risk_parity": "RP", "equal_weight": "EW"}
        for pname in ["max_sharpe", "min_variance", "risk_parity", "equal_weight"]:
            pdata = data["portfolios"][pname]
            px, py = _sx(pdata["vol"]), _sy(pdata["return"])
            c = colors[pname]
            port_dots += f'<circle cx="{px:.0f}" cy="{py:.0f}" r="6" fill="{c}" stroke="#fff" stroke-width="1.5"/>'
            port_dots += f'<text x="{px+8:.0f}" y="{py+4:.0f}" font-size="9" fill="{c}">{labels_map[pname]}</text>'

        frontier_svg = f"""
        <svg width="{svg_w}" height="{svg_h}" style="background:#f8f9fa;border:1px solid #e5e7eb;border-radius:8px">
          <text x="{svg_w//2}" y="15" text-anchor="middle" font-size="11" fill="#374151" font-weight="600">Efficient Frontier</text>
          <text x="{svg_w//2}" y="{svg_h-5}" text-anchor="middle" font-size="9" fill="#6b7280">Volatility →</text>
          <text x="10" y="{svg_h//2}" text-anchor="middle" font-size="9" fill="#6b7280" transform="rotate(-90,10,{svg_h//2})">Return →</text>
          {circles}
          {port_dots}
        </svg>"""
    else:
        frontier_svg = "<p>No frontier computed</p>"

    avg_corr = np.mean([p[0] for p in pair_list])
    ms = data["portfolios"]["max_sharpe"]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Strategy Correlation Deep Analysis</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e5e7eb;--text:#111827;--muted:#6b7280;--green:#059669;--red:#dc2626}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;max-width:1400px;margin:0 auto;padding:24px}}
h1{{font-size:1.5rem;font-weight:800;margin-bottom:4px}}
h2{{font-size:1.1rem;font-weight:700;margin:28px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--border)}}
h3{{font-size:.95rem;font-weight:600;margin:12px 0 8px}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:16px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:var(--muted);font-size:.65rem;font-weight:600;text-transform:uppercase}}.c .v{{font-size:1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:.76rem}}
th,td{{padding:4px 6px;text-align:right;border-bottom:1px solid var(--border)}}
th{{background:#f1f5f9;color:var(--muted);font-size:.66rem;font-weight:600;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
.section-card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin:12px 0}}
.note{{color:var(--muted);font-size:.8rem;margin:4px 0}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
@media(max-width:900px){{.grid2,.grid3{{grid-template-columns:1fr}}}}
.callout{{background:var(--card);border-left:4px solid var(--green);padding:12px;margin:12px 0;font-size:.82rem;border-radius:4px}}
.footer{{margin-top:36px;text-align:center;font-size:.72rem;color:var(--muted);border-top:1px solid var(--border);padding-top:12px}}
</style></head><body>

<h1>Strategy Correlation & Portfolio Optimization</h1>
<div class="subtitle">{n} real-data strategies &bull; NxN correlation matrix &bull; Min-variance / Max-Sharpe / Risk-parity optimization &bull; {ts}</div>

<div class="cards">
  <div class="c"><div class="l">Strategies</div><div class="v">{n}</div></div>
  <div class="c"><div class="l">Avg Pairwise Corr</div><div class="v" style="color:var(--green)">{avg_corr:+.3f}</div></div>
  <div class="c"><div class="l">Max Sharpe Port.</div><div class="v" style="color:var(--green)">{ms["sharpe"]:.2f}</div></div>
  <div class="c"><div class="l">Max SR Return</div><div class="v">{ms["return"]:+.2%}</div></div>
  <div class="c"><div class="l">Max SR Vol</div><div class="v">{ms["vol"]:.2%}</div></div>
  <div class="c"><div class="l">Most Negative Pair</div><div class="v" style="color:#2563eb">{pair_list[0][0]:+.3f}</div></div>
</div>

<h2>1. Correlation Heatmap</h2>
<p class="note">Blue = negative correlation (diversifiers), Red = positive (redundant), White = uncorrelated</p>
<div style="overflow-x:auto">
<table style="font-size:.65rem">
<thead><tr><th></th>{hdr}</tr></thead>
<tbody>{hm_rows}</tbody></table>
</div>

<h2>2. Strategy Overview</h2>
<table>
<thead><tr><th>Strategy</th><th>Status</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>SPY Corr</th><th>Ann Return</th><th>Ann Vol</th></tr></thead>
<tbody>{strat_rows}</tbody></table>

<h2>3. Correlation Extremes</h2>
<div class="grid2">
<div class="section-card">
<h3>Most Negatively Correlated (Best Diversifiers)</h3>
<table><thead><tr><th>Pair</th><th>Correlation</th></tr></thead>
<tbody>{neg_rows}</tbody></table>
</div>
<div class="section-card">
<h3>Most Positively Correlated (Redundant)</h3>
<table><thead><tr><th>Pair</th><th>Correlation</th></tr></thead>
<tbody>{pos_rows}</tbody></table>
</div>
</div>

<h2>4. Portfolio Comparison</h2>
<table>
<thead><tr><th>Method</th><th>Ann Return</th><th>Ann Vol</th><th>Sharpe</th></tr></thead>
<tbody>{port_rows}</tbody></table>

<h2>5. Efficient Frontier</h2>
<div style="text-align:center;margin:16px 0">{frontier_svg}</div>

<h2>6. Optimal Allocations & Risk Contributions</h2>
<div class="grid3">{weight_sections}</div>

<div class="callout">
<strong>Key Insight:</strong> The max-Sharpe portfolio achieves {ms["sharpe"]:.2f} Sharpe at {ms["return"]:+.2%} annual return with {ms["vol"]:.2%} vol.
The average pairwise correlation of {avg_corr:+.3f} across {n} strategies confirms strong diversification potential.
The most valuable diversifiers are those with negative SPY correlation (Vol-TS at -0.32, 1660-VRP at -0.70, TLT-IC at -0.20)
because they hedge the dominant EXP-1220 exposure.
</div>

<div class="footer">
  Deep Correlation Analysis &bull; {n} strategies &bull; {ts} &bull; PilotAI Compass
</div>
</body></html>"""


if __name__ == "__main__":
    main()
