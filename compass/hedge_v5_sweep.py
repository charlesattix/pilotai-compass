#!/usr/bin/env python3
"""2x + Crisis Alpha hedge sweep — v4 vs v5 side-by-side.

Re-runs the 2× EXP-1220 + hedge parameter sweep, this time with the
hedge-optimized v5 stream (compass/crisis_alpha_v5.py) and compares it
directly to the v4 baseline stream.

- Sweep hedge % 5-30% in 2.5% steps at 2× leverage (both v4 and v5).
- Sweep leverage 1.5-2.5× at each stream's optimal hedge %.
- Report DD reduction at the user's key operating points.

Rule Zero: all data from real Yahoo Finance. No synthetic series.
EXP-1220 stream reused from scripts.ultimate_portfolio.load_exp1220_dynamic
(real SPY + ^VIX + ^VIX3M + TailRiskProtector).

Output: compass/reports/2x_hedge_v5_comparison.html
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.metrics import full_metrics
from compass.crisis_alpha_v3 import load_universe_v3
from compass.crisis_alpha_v4 import ConfigV4, backtest_v4
from compass.crisis_alpha_v5 import (
    HedgeConfigV5,
    backtest_v5,
    score_hedge,
    search_hedge_configs,
    select_best_hedge,
)

REPORT_PATH = ROOT / "compass" / "reports" / "2x_hedge_v5_comparison.html"


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_exp1220() -> pd.Series:
    """EXP-1220 dynamic-leverage stream used in the prior 2× hedge deep dive."""
    from scripts.ultimate_portfolio import load_exp1220_dynamic
    s = load_exp1220_dynamic()
    s.index = pd.DatetimeIndex(s.index)
    return s


def build_v4_hedge(prices: pd.DataFrame) -> Tuple[pd.Series, ConfigV4]:
    """Rebuild the v4 baseline (matches the 'v4 baseline' in crisis_alpha_v5.py)."""
    cfg = ConfigV4(
        name="v4_baseline",
        lookback_preset="v2_round",
        vol_target=0.05,
        leverage=1.0,
        dd_brake_threshold=0.05,
        dd_brake_zone=0.03,
        max_weight=0.20,
        require_confirmation=False,
    )
    r = backtest_v4(prices, cfg)
    return r.daily_returns, r


def build_v5_hedge(prices: pd.DataFrame, exp1220: pd.Series
                    ) -> Tuple[pd.Series, HedgeConfigV5, Dict[str, float]]:
    """Run v5 hedge-objective grid search and pick the best hedge."""
    configs = search_hedge_configs(prices, exp1220)
    configs.sort(key=lambda c: c.hedge_score)
    best = select_best_hedge(configs)
    scores = score_hedge(best.daily_returns, exp1220)
    return best.daily_returns, best, scores


# ═══════════════════════════════════════════════════════════════════════════
# Sweeps
# ═══════════════════════════════════════════════════════════════════════════

def combine(exp1220: pd.Series, hedge: pd.Series,
             leverage: float, hedge_pct: float) -> pd.Series:
    common = exp1220.index.intersection(hedge.index).sort_values()
    a = exp1220.reindex(common).fillna(0)
    b = hedge.reindex(common).fillna(0)
    return (1 - hedge_pct) * a * leverage + hedge_pct * b


def hedge_pct_sweep(exp1220: pd.Series, hedge: pd.Series,
                     leverage: float = 2.0) -> List[Dict]:
    out = []
    for hp in np.arange(0.05, 0.305, 0.025):
        rets = combine(exp1220, hedge, leverage, float(hp))
        m = full_metrics(rets.values)
        m["leverage"] = leverage
        m["hedge_pct"] = round(float(hp), 4)
        out.append(m)
    return out


def leverage_sweep(exp1220: pd.Series, hedge: pd.Series,
                    optimal_hedge_pct: float) -> List[Dict]:
    out = []
    for lev in [1.5, 1.75, 2.0, 2.25, 2.5]:
        rets = combine(exp1220, hedge, lev, optimal_hedge_pct)
        m = full_metrics(rets.values)
        m["leverage"] = lev
        m["hedge_pct"] = optimal_hedge_pct
        out.append(m)
    return out


def pick_optimal(sweep: List[Dict], dd_max: float = 15.0) -> Dict:
    feasible = [r for r in sweep if r["max_dd_pct"] < dd_max]
    if not feasible:
        return max(sweep, key=lambda r: r["sharpe"])
    return max(feasible, key=lambda r: r["sharpe"])


def correlations(exp1220: pd.Series, hedge: pd.Series) -> Dict[str, float]:
    """Full-sample and stress-period correlation of hedge vs EXP-1220."""
    common = exp1220.index.intersection(hedge.index)
    a = exp1220.reindex(common).fillna(0)
    b = hedge.reindex(common).fillna(0)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return {"corr_full": 0.0, "corr_dd": 0.0, "downside_capture_pct": 0.0}

    scores = score_hedge(b, a, dd_threshold=0.03)
    return {
        "corr_full": round(scores["corr_full"], 3),
        "corr_dd": round(scores["corr_dd"], 3),
        "downside_capture_pct": round(scores["downside_capture"] * 100, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def _sweep_rows(sweep: List[Dict], optimal_hp: float) -> str:
    rows = ""
    for r in sweep:
        is_opt = abs(r["hedge_pct"] - optimal_hp) < 0.001
        hl = ' style="background:#ecfdf5"' if is_opt else ""
        meets = r["max_dd_pct"] < 15
        cagr_color = "#16a34a" if meets else "#dc2626"
        rows += (
            f"<tr{hl}>"
            f"<td>{r['hedge_pct']*100:.1f}%</td>"
            f"<td style='color:{cagr_color};font-weight:600'>{r['cagr_pct']:.1f}%</td>"
            f"<td style='font-weight:700'>{r['sharpe']:.2f}</td>"
            f"<td>{r['max_dd_pct']:.1f}%</td>"
            f"<td>{r['calmar']:.2f}</td>"
            f"<td>{r['vol_pct']:.1f}%</td>"
            f"</tr>"
        )
    return rows


def _lev_rows(sweep: List[Dict]) -> str:
    rows = ""
    for r in sweep:
        meets = r["max_dd_pct"] < 15
        cagr_color = "#16a34a" if meets else "#dc2626"
        rows += (
            f"<tr>"
            f"<td style='font-weight:700'>{r['leverage']}×</td>"
            f"<td>{r['hedge_pct']*100:.1f}%</td>"
            f"<td style='color:{cagr_color};font-weight:600'>{r['cagr_pct']:.1f}%</td>"
            f"<td style='font-weight:700'>{r['sharpe']:.2f}</td>"
            f"<td>{r['max_dd_pct']:.1f}%</td>"
            f"<td>{r['calmar']:.2f}</td>"
            f"</tr>"
        )
    return rows


def _sbs_rows(v4: List[Dict], v5: List[Dict]) -> str:
    rows = ""
    for a, b in zip(v4, v5):
        dd_delta = b["max_dd_pct"] - a["max_dd_pct"]
        sh_delta = b["sharpe"] - a["sharpe"]
        cagr_delta = b["cagr_pct"] - a["cagr_pct"]
        dd_color = "#16a34a" if dd_delta < 0 else "#dc2626"
        sh_color = "#16a34a" if sh_delta > 0 else "#dc2626"
        rows += (
            f"<tr>"
            f"<td>{a['hedge_pct']*100:.1f}%</td>"
            f"<td>{a['cagr_pct']:.1f}%</td>"
            f"<td>{a['sharpe']:.2f}</td>"
            f"<td>{a['max_dd_pct']:.1f}%</td>"
            f"<td>{b['cagr_pct']:.1f}%</td>"
            f"<td>{b['sharpe']:.2f}</td>"
            f"<td>{b['max_dd_pct']:.1f}%</td>"
            f"<td style='color:{'#16a34a' if cagr_delta>0 else '#dc2626'}'>{cagr_delta:+.1f}</td>"
            f"<td style='color:{sh_color}'>{sh_delta:+.2f}</td>"
            f"<td style='color:{dd_color};font-weight:700'>{dd_delta:+.1f}</td>"
            f"</tr>"
        )
    return rows


def build_html(
    exp1220_m: Dict,
    v4_sweep: List[Dict], v5_sweep: List[Dict],
    v4_lev: List[Dict], v5_lev: List[Dict],
    v4_opt: Dict, v5_opt: Dict,
    v4_corr: Dict, v5_corr: Dict,
    v4_name: str, v5_name: str,
    unhedged_2x: Dict,
) -> str:
    v4_sb = _sweep_rows(v4_sweep, v4_opt["hedge_pct"])
    v5_sb = _sweep_rows(v5_sweep, v5_opt["hedge_pct"])
    v4_lb = _lev_rows(v4_lev)
    v5_lb = _lev_rows(v5_lev)
    sbs = _sbs_rows(v4_sweep, v5_sweep)

    v4_10 = next(r for r in v4_sweep if abs(r["hedge_pct"] - 0.10) < 1e-4)
    v5_10 = next(r for r in v5_sweep if abs(r["hedge_pct"] - 0.10) < 1e-4)
    dd_reduction_v4 = unhedged_2x["max_dd_pct"] - v4_10["max_dd_pct"]
    dd_reduction_v5 = unhedged_2x["max_dd_pct"] - v5_10["max_dd_pct"]
    winner = "v5" if dd_reduction_v5 > dd_reduction_v4 else "v4"
    verdict_color = "#16a34a" if dd_reduction_v5 > 0 else "#dc2626"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2× + Crisis Alpha Hedge — v4 vs v5 Comparison</title>
<style>
* {{ box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       max-width:1200px;margin:0 auto;padding:28px;background:#fff;color:#1e293b;line-height:1.5; }}
h1 {{ font-size:1.8em;color:#0f172a;margin-bottom:4px; }}
h2 {{ color:#334155;margin-top:2.4em;padding-bottom:8px;border-bottom:2px solid #e2e8f0; }}
h3 {{ color:#475569;margin-top:1.6em; }}
.subtitle {{ color:#64748b;font-size:0.92rem;margin-bottom:24px; }}
.verdict {{ background:#ecfdf5;border:2px solid {verdict_color};border-radius:10px;padding:20px;margin:24px 0; }}
.verdict h3 {{ margin-top:0;color:#065f46; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:16px;margin:16px 0;font-size:0.84rem;line-height:1.7; }}
.kpi-row {{ display:flex;gap:14px;flex-wrap:wrap;margin:20px 0; }}
.kpi {{ background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center;flex:1;min-width:130px; }}
.kpi .value {{ font-size:1.55em;font-weight:800;color:#0f172a; }}
.kpi .label {{ font-size:0.72em;color:#64748b;margin-top:4px;text-transform:uppercase; }}
table {{ width:100%;border-collapse:collapse;margin:14px 0;font-size:0.84em; }}
th {{ background:#f1f5f9;padding:10px 12px;text-align:right;font-weight:600;color:#475569;
     border-bottom:2px solid #cbd5e1;font-size:0.74em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:8px 12px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
tr:hover {{ background:#f8fafc; }}
.grid2 {{ display:grid;grid-template-columns:1fr 1fr;gap:20px; }}
.footer {{ margin-top:3em;padding-top:1em;border-top:1px solid #e2e8f0;font-size:0.78em;color:#94a3b8;text-align:center; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:12px 0; }}
</style></head><body>

<h1>2× EXP-1220 + Crisis Alpha Hedge — v4 vs v5</h1>
<div class="subtitle">Hedge parameter sweep + leverage sweep · Crisis Alpha v5 hedge-optimized
objective vs v4 baseline · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="sources">
<strong>Rule Zero — all data real:</strong><br>
EXP-1220 stream: <code>scripts.ultimate_portfolio.load_exp1220_dynamic</code>
(real Yahoo SPY + ^VIX + ^VIX3M, TailRiskProtector-scaled)<br>
v4 hedge: <code>compass.crisis_alpha_v4.ConfigV4(v2_round,vt=0.05,l=1.0,brake 0.05+0.03)</code>
— the baseline referenced in <code>compass/crisis_alpha_v5.py</code><br>
v5 hedge: <code>compass.crisis_alpha_v5.search_hedge_configs+select_best_hedge</code>
— grid optimized on <code>hedge_score = corr_dd − 5·downside_capture</code><br>
Metrics via <code>compass.metrics.full_metrics</code> (correct arithmetic-mean Sharpe)
</div>

<div class="verdict">
<h3>Headline: does v5 actually reduce 2× EXP-1220 drawdown?</h3>
At 2× leverage + 10% hedge (the reference point):<br>
&nbsp;&nbsp;• <strong>Unhedged 2×:</strong> DD = {unhedged_2x['max_dd_pct']:.1f}% · Sharpe {unhedged_2x['sharpe']:.2f} · CAGR {unhedged_2x['cagr_pct']:.1f}%<br>
&nbsp;&nbsp;• <strong>+ 10% v4 hedge:</strong> DD = {v4_10['max_dd_pct']:.1f}% (Δ {-dd_reduction_v4:+.1f}pp) · Sharpe {v4_10['sharpe']:.2f} · CAGR {v4_10['cagr_pct']:.1f}%<br>
&nbsp;&nbsp;• <strong>+ 10% v5 hedge:</strong> DD = {v5_10['max_dd_pct']:.1f}% (Δ {-dd_reduction_v5:+.1f}pp) · Sharpe {v5_10['sharpe']:.2f} · CAGR {v5_10['cagr_pct']:.1f}%<br>
<strong>Winner on DD reduction: {winner}</strong>
(v5 beats v4 by {dd_reduction_v5 - dd_reduction_v4:+.1f}pp of DD at the 10% hedge point).
</div>

<h2>1. Hedge stream metrics (standalone)</h2>
<div class="grid2">
<div>
<h3>v4 baseline</h3>
<div class="kpi-row">
<div class="kpi"><div class="value">{v4_corr['corr_full']:+.3f}</div><div class="label">Full corr</div></div>
<div class="kpi"><div class="value">{v4_corr['corr_dd']:+.3f}</div><div class="label">Stress corr</div></div>
<div class="kpi"><div class="value">{v4_corr['downside_capture_pct']:+.3f}%</div><div class="label">Downside cap</div></div>
</div>
<p><code>{v4_name}</code></p>
</div>
<div>
<h3>v5 hedge-optimized</h3>
<div class="kpi-row">
<div class="kpi"><div class="value">{v5_corr['corr_full']:+.3f}</div><div class="label">Full corr</div></div>
<div class="kpi"><div class="value">{v5_corr['corr_dd']:+.3f}</div><div class="label">Stress corr</div></div>
<div class="kpi"><div class="value">{v5_corr['downside_capture_pct']:+.3f}%</div><div class="label">Downside cap</div></div>
</div>
<p><code>{v5_name}</code></p>
</div>
</div>
<div class="note">
<strong>Interpretation:</strong> The stress-period correlation (<code>corr_dd</code>) is the key
number — it is the correlation computed only on days when EXP-1220 is in ≥3% drawdown.
A hedge is useful <em>only</em> on these days; a strategy that is negatively correlated
with EXP-1220 overall but positively correlated during stress is not a real hedge.
</div>

<h2>2. Hedge % sweep at 2× leverage — side-by-side</h2>
<table>
<thead><tr>
<th rowspan="2">Hedge %</th>
<th colspan="3">v4 baseline</th>
<th colspan="3">v5 hedge-optimized</th>
<th colspan="3">Delta (v5 − v4)</th>
</tr><tr>
<th>CAGR</th><th>Sharpe</th><th>DD</th>
<th>CAGR</th><th>Sharpe</th><th>DD</th>
<th>ΔCAGR</th><th>ΔSharpe</th><th>ΔDD</th>
</tr></thead>
<tbody>{sbs}</tbody>
</table>

<h3>v4 hedge sweep (2× leverage)</h3>
<table>
<thead><tr><th>Hedge %</th><th>CAGR</th><th>Sharpe</th><th>DD</th><th>Calmar</th><th>Vol</th></tr></thead>
<tbody>{v4_sb}</tbody>
</table>
<p><strong>v4 optimal (max Sharpe, DD&lt;15%):</strong>
{v4_opt['hedge_pct']*100:.1f}% hedge → CAGR {v4_opt['cagr_pct']:.1f}%,
Sharpe {v4_opt['sharpe']:.2f}, DD {v4_opt['max_dd_pct']:.1f}%</p>

<h3>v5 hedge sweep (2× leverage)</h3>
<table>
<thead><tr><th>Hedge %</th><th>CAGR</th><th>Sharpe</th><th>DD</th><th>Calmar</th><th>Vol</th></tr></thead>
<tbody>{v5_sb}</tbody>
</table>
<p><strong>v5 optimal (max Sharpe, DD&lt;15%):</strong>
{v5_opt['hedge_pct']*100:.1f}% hedge → CAGR {v5_opt['cagr_pct']:.1f}%,
Sharpe {v5_opt['sharpe']:.2f}, DD {v5_opt['max_dd_pct']:.1f}%</p>

<h2>3. Leverage sweep at each stream's optimal hedge %</h2>
<div class="grid2">
<div>
<h3>v4 @ {v4_opt['hedge_pct']*100:.1f}% hedge</h3>
<table>
<thead><tr><th>Lev</th><th>Hedge</th><th>CAGR</th><th>Sharpe</th><th>DD</th><th>Calmar</th></tr></thead>
<tbody>{v4_lb}</tbody>
</table>
</div>
<div>
<h3>v5 @ {v5_opt['hedge_pct']*100:.1f}% hedge</h3>
<table>
<thead><tr><th>Lev</th><th>Hedge</th><th>CAGR</th><th>Sharpe</th><th>DD</th><th>Calmar</th></tr></thead>
<tbody>{v5_lb}</tbody>
</table>
</div>
</div>

<h2>4. EXP-1220 reference</h2>
<div class="kpi-row">
<div class="kpi"><div class="value">{exp1220_m['cagr_pct']:.1f}%</div><div class="label">CAGR</div></div>
<div class="kpi"><div class="value">{exp1220_m['sharpe']:.2f}</div><div class="label">Sharpe</div></div>
<div class="kpi"><div class="value">{exp1220_m['max_dd_pct']:.1f}%</div><div class="label">Max DD</div></div>
<div class="kpi"><div class="value">{exp1220_m['vol_pct']:.1f}%</div><div class="label">Vol</div></div>
<div class="kpi"><div class="value">{exp1220_m['n_days']}</div><div class="label">Days</div></div>
</div>

<div class="footer">
scripts/hedge_sweep_v5_comparison.py · Crisis Alpha v4 (d041b1d baseline) vs v5 ·
Real Yahoo Finance + TailRiskProtector only · No synthetic data · Rule Zero.
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("2× + Crisis Alpha hedge sweep — v4 vs v5 comparison")
    print("=" * 72)

    print("\n[1/6] Loading real Yahoo universe (v3 ETF set)...")
    prices = load_universe_v3(start="2014-01-01", end="2026-01-01")
    print(f"      {len(prices)} days × {len(prices.columns)} assets")

    print("\n[2/6] Loading EXP-1220 dynamic stream...")
    exp1220 = load_exp1220()
    exp1220_m = full_metrics(exp1220.values)
    print(f"      EXP-1220: {len(exp1220)} days  CAGR {exp1220_m['cagr_pct']:+.1f}%  "
          f"Sharpe {exp1220_m['sharpe']:.2f}  DD {exp1220_m['max_dd_pct']:.1f}%")

    print("\n[3/6] Building v4 baseline hedge...")
    v4_hedge, v4_cfg = build_v4_hedge(prices)
    v4_corr = correlations(exp1220, v4_hedge)
    print(f"      v4: full corr {v4_corr['corr_full']:+.3f}  "
          f"stress corr {v4_corr['corr_dd']:+.3f}  "
          f"downside cap {v4_corr['downside_capture_pct']:+.3f}%")

    print("\n[4/6] Searching v5 hedge-optimized grid...")
    v5_hedge, v5_best, v5_corr = build_v5_hedge(prices, exp1220)
    v5_corr_disp = {
        "corr_full": round(v5_corr["corr_full"], 3),
        "corr_dd": round(v5_corr["corr_dd"], 3),
        "downside_capture_pct": round(v5_corr["downside_capture"] * 100, 4),
    }
    print(f"      v5 BEST: {v5_best.name}")
    print(f"      v5: full corr {v5_corr_disp['corr_full']:+.3f}  "
          f"stress corr {v5_corr_disp['corr_dd']:+.3f}  "
          f"downside cap {v5_corr_disp['downside_capture_pct']:+.3f}%")

    print("\n[5/6] Hedge % sweep 5%–30% at 2× leverage...")
    v4_sweep = hedge_pct_sweep(exp1220, v4_hedge, leverage=2.0)
    v5_sweep = hedge_pct_sweep(exp1220, v5_hedge, leverage=2.0)

    print("  v4 results:")
    for r in v4_sweep:
        m = "*" if r["max_dd_pct"] < 15 else " "
        print(f"    {m} {r['hedge_pct']*100:5.1f}%  "
              f"CAGR {r['cagr_pct']:7.1f}%  "
              f"Sharpe {r['sharpe']:5.2f}  "
              f"DD {r['max_dd_pct']:5.1f}%")

    print("  v5 results:")
    for r in v5_sweep:
        m = "*" if r["max_dd_pct"] < 15 else " "
        print(f"    {m} {r['hedge_pct']*100:5.1f}%  "
              f"CAGR {r['cagr_pct']:7.1f}%  "
              f"Sharpe {r['sharpe']:5.2f}  "
              f"DD {r['max_dd_pct']:5.1f}%")

    v4_opt = pick_optimal(v4_sweep)
    v5_opt = pick_optimal(v5_sweep)
    print(f"\n  v4 optimal: {v4_opt['hedge_pct']*100:.1f}% hedge  "
          f"Sharpe {v4_opt['sharpe']:.2f}  DD {v4_opt['max_dd_pct']:.1f}%")
    print(f"  v5 optimal: {v5_opt['hedge_pct']*100:.1f}% hedge  "
          f"Sharpe {v5_opt['sharpe']:.2f}  DD {v5_opt['max_dd_pct']:.1f}%")

    print("\n[6/6] Leverage sweep 1.5–2.5× at each optimal hedge...")
    v4_lev = leverage_sweep(exp1220, v4_hedge, v4_opt["hedge_pct"])
    v5_lev = leverage_sweep(exp1220, v5_hedge, v5_opt["hedge_pct"])
    for r in v4_lev:
        print(f"  v4  {r['leverage']:.2f}×  CAGR {r['cagr_pct']:7.1f}%  "
              f"Sharpe {r['sharpe']:5.2f}  DD {r['max_dd_pct']:.1f}%")
    for r in v5_lev:
        print(f"  v5  {r['leverage']:.2f}×  CAGR {r['cagr_pct']:7.1f}%  "
              f"Sharpe {r['sharpe']:5.2f}  DD {r['max_dd_pct']:.1f}%")

    # Unhedged 2× EXP-1220 reference
    unhedged_rets = exp1220 * 2.0
    unhedged_2x = full_metrics(unhedged_rets.values)
    print(f"\n  Unhedged 2× EXP-1220: CAGR {unhedged_2x['cagr_pct']:.1f}%  "
          f"Sharpe {unhedged_2x['sharpe']:.2f}  DD {unhedged_2x['max_dd_pct']:.1f}%")

    html = build_html(
        exp1220_m=exp1220_m,
        v4_sweep=v4_sweep, v5_sweep=v5_sweep,
        v4_lev=v4_lev, v5_lev=v5_lev,
        v4_opt=v4_opt, v5_opt=v5_opt,
        v4_corr=v4_corr, v5_corr=v5_corr_disp,
        v4_name=v4_cfg.name, v5_name=v5_best.name,
        unhedged_2x=unhedged_2x,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")
    print(f"Size: {len(html)/1024:.0f} KB")


if __name__ == "__main__":
    main()
