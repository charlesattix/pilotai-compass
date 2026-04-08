"""
compass/exp2040_leveraged_calendars.py — EXP-2040 Leveraged GLD/SLV Calendar Spread Scaling.

CONTEXT:
  EXP-1770 (commodity calendar spreads) showed that the GLD−GC=F and
  SLV−SI=F pairs harvest a real, persistent roll-yield drift:
      GLD pair  : Sharpe 2.70, CAGR 15.2%, Max DD ~2.5%
      SLV pair  : Sharpe 2.27, CAGR ~25%, larger DD
  Both run on real Yahoo data, walk-forward 2015-2025, and have low
  correlation to EXP-1220 SPY put-credit-spreads.

QUESTION:
  Can we lift the GLD/SLV sleeve allocation from the current 7.5% per
  leg to 15-25% (combined) and lever it 1.5×-2× to boost the portfolio
  Sharpe, while keeping max DD < 12% (the masterplan ceiling)?

PROTOCOL (all REAL data, Rule Zero):
  1. Reuse `compass.exp1770_commodity_calendars.load_pair` and
     `walk_forward` to obtain the canonical OOS daily-return series for
     GLD−GC=F and SLV−SI=F. Yahoo only.
  2. Build leverage variants ×1.0 / ×1.5 / ×2.0 (linear scaling of the
     daily returns — calendars have no embedded option leverage and the
     positions are dollar-neutral spreads, so linear leverage models the
     economics correctly when assuming sufficient margin).
  3. Build a GLD/SLV mix sweep at vol-weighted, equal-weight, 60/40,
     40/60.
  4. Build the combined portfolio with EXP-1220 (loaded from
     scripts.ultimate_portfolio.load_exp1220_dynamic — the canonical
     dynamic-leverage daily return stream).
  5. Sweep allocations: 10%, 15%, 20%, 25% to the (GLD+SLV) sleeve,
     remainder to EXP-1220, with each leverage choice.
  6. Compute portfolio metrics on the OVERLAP window where both
     EXP-1220 and the calendar streams have data.

OUTPUTS:
  compass/reports/exp2040_leveraged_calendars.json
  compass/reports/exp2040_leveraged_calendars.html

Run::
    python3 -m compass.exp2040_leveraged_calendars
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp1770_commodity_calendars import load_pair, walk_forward

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2040_leveraged_calendars.json"
REPORT_HTML = REPORT_DIR / "exp2040_leveraged_calendars.html"

TRADING_DAYS = 252
DD_CEILING = 0.12   # 12% portfolio drawdown ceiling


# ═══════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_calendar_streams() -> Dict[str, pd.Series]:
    """Run EXP-1770 walk-forward to get OOS daily return streams for
    GLD and SLV calendar spreads. Returns dict keyed by ETF symbol."""
    streams: Dict[str, pd.Series] = {}
    for etf, fut in [("GLD", "GC=F"), ("SLV", "SI=F")]:
        print(f"  loading {etf}/{fut} (Yahoo, real)...")
        df = load_pair(etf, fut)
        bt = walk_forward(etf, df)
        s = bt.daily_returns.dropna()
        s.name = etf
        streams[etf] = s
        m = bt.metrics
        print(f"    {etf}: n_days={m['n_days']} CAGR={m['cagr']*100:.2f}% "
              f"Sharpe={m['sharpe']:.2f} MaxDD={m['max_dd']*100:.2f}%")
    return streams


def load_exp1220_stream() -> pd.Series:
    """Load EXP-1220 canonical daily return stream."""
    print("  loading EXP-1220 daily returns (canonical)...")
    from scripts.ultimate_portfolio import load_exp1220_dynamic
    s = load_exp1220_dynamic().dropna()
    s.name = "EXP-1220"
    print(f"    EXP-1220: {len(s)} days  {s.index.min().date()} → {s.index.max().date()}")
    return s


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(rets: pd.Series) -> Dict[str, float]:
    r = rets.dropna()
    n = len(r)
    if n < 5:
        return {"n_days": n, "cagr_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "vol_pct": 0.0, "hit_rate_pct": 0.0}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = (1.0 + r).cumprod()
    yrs = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    pk = eq.cummax()
    dd = (eq - pk) / pk
    return {
        "n_days": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(float(dd.min()) * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
        "hit_rate_pct": round(float((r != 0).mean()) * 100, 1),
    }


def correlation(a: pd.Series, b: pd.Series) -> Optional[float]:
    common = a.index.intersection(b.index)
    if len(common) < 30:
        return None
    aa = a.reindex(common).fillna(0).values
    bb = b.reindex(common).fillna(0).values
    if aa.std() == 0 or bb.std() == 0:
        return None
    c = float(np.corrcoef(aa, bb)[0, 1])
    return None if math.isnan(c) else round(c, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Variant builders
# ═══════════════════════════════════════════════════════════════════════════

def lever(s: pd.Series, mult: float) -> pd.Series:
    """Linear leverage. Calendars are spread positions — leverage 2×
    means double the dollar exposure → daily return doubles. Assumes
    sufficient margin (calendar spreads have low Reg-T requirements)."""
    return (s * mult).rename(s.name)


def vol_weight_mix(g: pd.Series, s: pd.Series) -> Tuple[float, float]:
    """Inverse-vol weights so each leg contributes equal risk."""
    vg = g.std() if g.std() > 1e-12 else 1.0
    vs = s.std() if s.std() > 1e-12 else 1.0
    wg_raw = 1.0 / vg
    ws_raw = 1.0 / vs
    tot = wg_raw + ws_raw
    return (wg_raw / tot, ws_raw / tot)


def blend(streams: Dict[str, pd.Series],
            weights: Dict[str, float]) -> pd.Series:
    """Daily-rebalanced linear blend."""
    idx = None
    for s in streams.values():
        idx = s.index if idx is None else idx.union(s.index)
    out = pd.Series(0.0, index=idx)
    for k, w in weights.items():
        if k in streams:
            out = out.add(streams[k].reindex(idx).fillna(0.0) * w, fill_value=0.0)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Experiment workflow
# ═══════════════════════════════════════════════════════════════════════════

def run_solo_leverage_sweep(streams: Dict[str, pd.Series]) -> Dict:
    """(a) and (b): Solo leverage sweep on GLD and SLV individually."""
    out: Dict[str, Dict] = {}
    for etf, s in streams.items():
        leg: Dict[str, Dict] = {}
        for mult in [1.0, 1.5, 2.0]:
            lev_s = lever(s, mult)
            m = compute_metrics(lev_s)
            leg[f"{mult}x"] = {
                "leverage": mult,
                **m,
                "passes_dd_ceiling": m["max_dd_pct"] > -DD_CEILING * 100,
            }
        out[etf] = leg
    return out


def run_mix_sweep(streams: Dict[str, pd.Series]) -> Dict:
    """(c): Optimal GLD/SLV mix sweep, including vol-weighted."""
    g = streams["GLD"]
    s = streams["SLV"]
    vw_g, vw_s = vol_weight_mix(g, s)

    mixes = {
        "equal_50_50":      {"GLD": 0.50, "SLV": 0.50},
        "gld_heavy_60_40":  {"GLD": 0.60, "SLV": 0.40},
        "slv_heavy_40_60":  {"GLD": 0.40, "SLV": 0.60},
        f"vol_weight ({vw_g:.2f}/{vw_s:.2f})": {"GLD": vw_g, "SLV": vw_s},
    }

    out: Dict[str, Dict] = {}
    for name, w in mixes.items():
        for mult in [1.0, 1.5, 2.0]:
            lev_streams = {k: lever(streams[k], mult) for k in ("GLD", "SLV")}
            mix = blend(lev_streams, w)
            m = compute_metrics(mix)
            out[f"{name} ×{mult}"] = {
                "weights": w,
                "leverage": mult,
                **m,
                "passes_dd_ceiling": m["max_dd_pct"] > -DD_CEILING * 100,
            }
    return out


def run_portfolio_integration(
    exp1220: pd.Series,
    streams: Dict[str, pd.Series],
) -> Dict:
    """(d): Sweep EXP-1220 + (GLD/SLV sleeve) at 10/15/20/25% allocations
    with each calendar leverage 1.0×/1.5×/2.0×.

    The sleeve is itself an inverse-vol blend of GLD and SLV (so each
    contributes equal RISK to the sleeve, not equal capital). The
    benchmark is 100% EXP-1220 over the same overlap window.
    """
    g = streams["GLD"]
    s = streams["SLV"]
    vw_g, vw_s = vol_weight_mix(g, s)
    sleeve_weights = {"GLD": vw_g, "SLV": vw_s}

    # Overlap window where ALL three series are defined
    common = exp1220.index.intersection(g.index).intersection(s.index)
    common = common.sort_values()
    if len(common) < 250:
        raise RuntimeError(f"insufficient overlap window: {len(common)} days")

    e = exp1220.reindex(common).fillna(0.0)
    g_a = g.reindex(common).fillna(0.0)
    s_a = s.reindex(common).fillna(0.0)

    # Benchmark: 100% EXP-1220 over the overlap
    bench = e.copy()
    bench_m = compute_metrics(bench)

    sweeps: Dict[str, Dict] = {}
    for alloc in [0.10, 0.15, 0.20, 0.25]:
        for mult in [1.0, 1.5, 2.0]:
            sleeve_g = g_a * mult * sleeve_weights["GLD"]
            sleeve_s = s_a * mult * sleeve_weights["SLV"]
            sleeve = sleeve_g + sleeve_s
            port = (1.0 - alloc) * e + alloc * sleeve
            m = compute_metrics(port)
            corr_to_bench = correlation(port, bench)
            corr_sleeve_to_bench = correlation(sleeve, bench)
            sweeps[f"alloc_{int(alloc*100)}_lev_{mult}x"] = {
                "alloc_to_sleeve": alloc,
                "alloc_to_exp1220": round(1.0 - alloc, 3),
                "sleeve_leverage": mult,
                "sleeve_weights": sleeve_weights,
                **m,
                "passes_dd_ceiling": m["max_dd_pct"] > -DD_CEILING * 100,
                "delta_sharpe_vs_baseline": round(m["sharpe"] - bench_m["sharpe"], 3),
                "delta_cagr_vs_baseline": round(m["cagr_pct"] - bench_m["cagr_pct"], 3),
                "corr_port_to_baseline": corr_to_bench,
                "corr_sleeve_to_baseline": corr_sleeve_to_bench,
            }

    return {
        "overlap_window": {
            "start": str(common.min().date()),
            "end": str(common.max().date()),
            "n_days": int(len(common)),
        },
        "baseline_100pct_exp1220": bench_m,
        "sweeps": sweeps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def render_html(payload: Dict) -> str:
    def cls_dd(v: float) -> str:
        return "good" if v > -12 else "bad"

    def fmt(v, dp=2):
        return f"{v:.{dp}f}" if isinstance(v, (int, float)) else str(v)

    # Solo leverage rows
    solo_rows = ""
    for etf, lvls in payload["solo_leverage"].items():
        for k, m in lvls.items():
            solo_rows += f"""<tr>
                <td><strong>{etf}</strong></td><td>{m['leverage']}×</td>
                <td>{m['cagr_pct']:.2f}%</td>
                <td>{m['sharpe']:.2f}</td>
                <td class="{cls_dd(m['max_dd_pct'])}">{m['max_dd_pct']:.2f}%</td>
                <td>{m['vol_pct']:.2f}%</td>
                <td>{'✅' if m['passes_dd_ceiling'] else '❌'}</td>
            </tr>"""

    # Mix rows
    mix_rows = ""
    for name, m in payload["gld_slv_mix"].items():
        mix_rows += f"""<tr>
            <td>{name}</td>
            <td>{m['cagr_pct']:.2f}%</td>
            <td>{m['sharpe']:.2f}</td>
            <td class="{cls_dd(m['max_dd_pct'])}">{m['max_dd_pct']:.2f}%</td>
            <td>{m['vol_pct']:.2f}%</td>
            <td>{'✅' if m['passes_dd_ceiling'] else '❌'}</td>
        </tr>"""

    # Portfolio integration rows
    integ = payload["portfolio_integration"]
    bench = integ["baseline_100pct_exp1220"]
    integ_rows = ""
    for k, m in integ["sweeps"].items():
        integ_rows += f"""<tr>
            <td>{int(m['alloc_to_sleeve']*100)}%</td>
            <td>{m['sleeve_leverage']}×</td>
            <td>{m['cagr_pct']:.2f}%</td>
            <td>{m['sharpe']:.2f}</td>
            <td class="{cls_dd(m['max_dd_pct'])}">{m['max_dd_pct']:.2f}%</td>
            <td>{m['vol_pct']:.2f}%</td>
            <td>{m['delta_sharpe_vs_baseline']:+.2f}</td>
            <td>{m['delta_cagr_vs_baseline']:+.2f}%</td>
            <td>{m['corr_port_to_baseline'] if m['corr_port_to_baseline'] is not None else '—'}</td>
            <td>{'✅' if m['passes_dd_ceiling'] else '❌'}</td>
        </tr>"""

    # Best variant
    best = payload.get("best_portfolio", {})
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2040 Leveraged GLD/SLV Calendar Scaling</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1200px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.5em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:140px; }}
  .kpi .value {{ font-size:1.7em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.78em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  td.good {{ color:#16a34a; font-weight:600; }}
  td.bad  {{ color:#dc2626; font-weight:600; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
</style></head><body>

<h1>EXP-2040 — Leveraged GLD/SLV Calendar Spread Scaling</h1>
<div class="subtitle">Real Yahoo Finance data, 2015-2025, walk-forward via EXP-1770
| {payload['timestamp']}</div>

<div class="note">
    <strong>Question:</strong> Can the GLD+SLV sleeve be lifted from 7.5% per leg
    to 15-25% combined, levered to 1.5×-2×, to lift portfolio Sharpe while
    holding max DD &lt; {int(DD_CEILING*100)}%?<br>
    <strong>Data:</strong> EXP-1770 walk-forward OOS daily streams (real Yahoo
    GLD/SLV/GC=F/SI=F), and the canonical EXP-1220 dynamic-leverage daily
    return series. No synthetic data, no random fills.
</div>

<h2>Baseline (100% EXP-1220 over overlap window)</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value">{bench['cagr_pct']:.2f}%</div><div class="label">CAGR</div></div>
    <div class="kpi"><div class="value">{bench['sharpe']:.2f}</div><div class="label">Sharpe</div></div>
    <div class="kpi"><div class="value">{bench['max_dd_pct']:.2f}%</div><div class="label">Max DD</div></div>
    <div class="kpi"><div class="value">{bench['vol_pct']:.2f}%</div><div class="label">Vol</div></div>
    <div class="kpi"><div class="value">{integ['overlap_window']['n_days']}</div><div class="label">Overlap Days</div></div>
</div>

<h2>(a/b) Solo leverage sweep — GLD and SLV calendars</h2>
<table>
    <thead><tr><th>Pair</th><th>Lev</th><th>CAGR</th><th>Sharpe</th>
    <th>Max DD</th><th>Vol</th><th>≤{int(DD_CEILING*100)}% DD?</th></tr></thead>
    <tbody>{solo_rows}</tbody>
</table>

<h2>(c) GLD/SLV mix sweep</h2>
<table>
    <thead><tr><th>Mix</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th>
    <th>Vol</th><th>≤{int(DD_CEILING*100)}% DD?</th></tr></thead>
    <tbody>{mix_rows}</tbody>
</table>

<h2>(d) Portfolio integration with EXP-1220</h2>
<table>
    <thead><tr><th>Sleeve %</th><th>Lev</th><th>CAGR</th><th>Sharpe</th>
    <th>Max DD</th><th>Vol</th><th>ΔSharpe</th><th>ΔCAGR</th>
    <th>ρ vs base</th><th>≤{int(DD_CEILING*100)}% DD?</th></tr></thead>
    <tbody>{integ_rows}</tbody>
</table>

<h2>Best portfolio variant (passes DD ceiling, max Sharpe)</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value">{best.get('label','—')}</div><div class="label">Variant</div></div>
    <div class="kpi"><div class="value">{best.get('cagr_pct', 0):.2f}%</div><div class="label">CAGR</div></div>
    <div class="kpi"><div class="value">{best.get('sharpe', 0):.2f}</div><div class="label">Sharpe</div></div>
    <div class="kpi"><div class="value">{best.get('max_dd_pct', 0):.2f}%</div><div class="label">Max DD</div></div>
    <div class="kpi"><div class="value">{best.get('delta_sharpe_vs_baseline', 0):+.2f}</div><div class="label">vs Baseline</div></div>
</div>

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2040 — compass/exp2040_leveraged_calendars.py · Real Yahoo data via EXP-1770
walk-forward · Linear leverage on dollar-neutral calendar spreads
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2040 — Leveraged GLD/SLV Calendar Spread Scaling")
    print("=" * 72)

    print("\n[1/5] Loading calendar streams (EXP-1770 walk-forward)...")
    streams = load_calendar_streams()

    print("\n[2/5] Loading EXP-1220 daily return stream (canonical)...")
    exp1220 = load_exp1220_stream()

    print("\n[3/5] (a/b) Solo leverage sweep on GLD and SLV...")
    solo = run_solo_leverage_sweep(streams)
    for etf, lvls in solo.items():
        for k, m in lvls.items():
            print(f"  {etf} {k}: CAGR={m['cagr_pct']:.2f}% "
                  f"Sharpe={m['sharpe']:.2f} DD={m['max_dd_pct']:.2f}% "
                  f"{'PASS' if m['passes_dd_ceiling'] else 'FAIL'}")

    print("\n[4/5] (c) GLD/SLV mix sweep...")
    mix = run_mix_sweep(streams)
    for name, m in mix.items():
        print(f"  {name}: CAGR={m['cagr_pct']:.2f}% Sharpe={m['sharpe']:.2f} "
              f"DD={m['max_dd_pct']:.2f}% "
              f"{'PASS' if m['passes_dd_ceiling'] else 'FAIL'}")

    print("\n[5/5] (d) Portfolio integration sweep with EXP-1220...")
    integ = run_portfolio_integration(exp1220, streams)
    print(f"  baseline 100% EXP-1220: "
          f"CAGR={integ['baseline_100pct_exp1220']['cagr_pct']:.2f}% "
          f"Sharpe={integ['baseline_100pct_exp1220']['sharpe']:.2f} "
          f"DD={integ['baseline_100pct_exp1220']['max_dd_pct']:.2f}%")
    for k, m in integ["sweeps"].items():
        flag = "PASS" if m["passes_dd_ceiling"] else "FAIL"
        print(f"  {k}: CAGR={m['cagr_pct']:.2f}% Sharpe={m['sharpe']:.2f} "
              f"DD={m['max_dd_pct']:.2f}% ΔSharpe={m['delta_sharpe_vs_baseline']:+.2f} "
              f"ρ={m['corr_port_to_baseline']} {flag}")

    # Pick best variant: max Sharpe among DD-passing
    eligible = [(k, m) for k, m in integ["sweeps"].items() if m["passes_dd_ceiling"]]
    if eligible:
        best_k, best_m = max(eligible, key=lambda kv: kv[1]["sharpe"])
        best_payload = {
            "label": best_k,
            **best_m,
        }
        print(f"\n  WINNER (passes DD ceiling): {best_k}")
        print(f"  Sharpe={best_m['sharpe']}  CAGR={best_m['cagr_pct']}%  "
              f"DD={best_m['max_dd_pct']}%  ΔSharpe={best_m['delta_sharpe_vs_baseline']:+.2f}")
    else:
        best_payload = {}
        print("\n  No variant passes the 12% DD ceiling.")

    payload = {
        "experiment": "EXP-2040",
        "title": "Leveraged GLD/SLV Calendar Spread Scaling",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data_sources": {
            "GLD_calendar": "compass.exp1770_commodity_calendars walk-forward (Yahoo GLD vs GC=F)",
            "SLV_calendar": "compass.exp1770_commodity_calendars walk-forward (Yahoo SLV vs SI=F)",
            "EXP-1220":     "scripts.ultimate_portfolio.load_exp1220_dynamic (canonical)",
        },
        "dd_ceiling_pct": DD_CEILING * 100,
        "solo_leverage": solo,
        "gld_slv_mix": mix,
        "portfolio_integration": integ,
        "best_portfolio": best_payload,
        "rule_zero": (
            "All inputs are real: EXP-1770 walk-forward OOS streams from Yahoo "
            "GLD/SLV/GC=F/SI=F, plus the canonical EXP-1220 dynamic-leverage "
            "daily return stream from scripts.ultimate_portfolio. No synthetic "
            "data, no random fills, no Black-Scholes pricing."
        ),
    }

    print("\nWriting reports...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  → {REPORT_JSON}")
    REPORT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"  → {REPORT_HTML}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
