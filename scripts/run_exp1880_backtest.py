#!/usr/bin/env python3
"""
EXP-1880 — Integrated overlay backtest.

Runs EXP-1220 credit spreads on real IronVault data four ways:
  1. baseline                       (no overlay)
  2. fomc_only                      (EXP-1740 filters)
  3. pcr_only                       (EXP-1750 filters)
  4. integrated (fomc + pcr + vol-stress)   ← EXP-1880 production config

Writes:
  compass/reports/exp1880_integrated.json
  compass/reports/exp1880_integrated.html
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp1220_standalone import run_exp1220_trades
from compass.exp1880_integrated_overlays import (
    IntegratedEntryOverlay,
    OverlayConfig,
)
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp1880_integrated.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp1880_integrated.html"


def _metrics(trades: List[Dict], label: str) -> Dict:
    if not trades:
        return {"label": label, "n": 0, "pnl": 0.0, "wr": 0.0,
                "sharpe": 0.0, "cagr_pct": 0.0, "max_dd_pct": 0.0,
                "avg_pnl": 0.0, "trades_per_yr": 0.0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    wins = (pnl > 0).sum()
    equity = 100_000 + pnl.cumsum()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    yrs = max(1.0, (
        datetime.strptime(trades[-1]["exit_date"], "%Y-%m-%d")
        - datetime.strptime(trades[0]["entry_date"], "%Y-%m-%d")
    ).days / 365.25)
    tpy = len(pnl) / yrs
    rets = pnl / 100_000
    mu, sd = rets.mean(), (rets.std(ddof=1) if len(rets) > 1 else 0.0)
    sharpe = (mu / sd) * math.sqrt(tpy) if sd > 1e-12 else 0.0
    return {
        "label": label, "n": int(len(pnl)),
        "pnl": float(pnl.sum()),
        "wr": float(wins / len(pnl)),
        "sharpe": float(sharpe),
        "cagr_pct": float((equity[-1] / 100_000) ** (1 / yrs) * 100 - 100),
        "max_dd_pct": float(-dd.min() * 100),
        "avg_pnl": float(pnl.mean()),
        "trades_per_yr": float(tpy),
    }


def _walk_forward(trades: List[Dict], overlay: IntegratedEntryOverlay) -> List[Dict]:
    by_year: Dict[int, List[Dict]] = {}
    for t in trades:
        by_year.setdefault(int(t["entry_date"][:4]), []).append(t)
    rows = []
    for y in sorted(by_year):
        base = _metrics(by_year[y], f"{y} baseline")
        filt = _metrics(overlay.filter_trades(by_year[y]), f"{y} integrated")
        rows.append({
            "year": y,
            "baseline_n": base["n"], "baseline_sharpe": base["sharpe"],
            "baseline_pnl": base["pnl"],
            "filtered_n": filt["n"], "filtered_sharpe": filt["sharpe"],
            "filtered_pnl": filt["pnl"],
            "delta_sharpe": filt["sharpe"] - base["sharpe"],
        })
    return rows


def main():
    print("[1/4] loading IronVault and SPY/VIX history …")
    import yfinance as yf
    hd = IronVault.instance()
    spy = yf.download("SPY", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index)
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)

    print("[2/4] running EXP-1220 baseline trades …")
    trades = run_exp1220_trades(hd, spy, vix)
    print(f"      {len(trades)} trades")

    print("[3/4] building integrated overlay panel …")
    fomc_only = IntegratedEntryOverlay.from_config(
        OverlayConfig(use_fomc=True, use_vix_slope=True,
                      use_pcr=False, use_vix_inversion=False, use_put_zspike=False),
        hd=hd,
    )
    pcr_only = IntegratedEntryOverlay.from_config(
        OverlayConfig(use_fomc=False, use_vix_slope=False,
                      use_pcr=True, use_vix_inversion=True, use_put_zspike=True),
        hd=hd,
    )
    integrated = IntegratedEntryOverlay.from_config(OverlayConfig(), hd=hd)

    print("[4/4] applying overlays …")
    base_m = _metrics(trades, "baseline")
    fomc_m = _metrics(fomc_only.filter_trades(trades), "fomc_only")
    pcr_m  = _metrics(pcr_only.filter_trades(trades),  "pcr_only")
    int_m  = _metrics(integrated.filter_trades(trades), "integrated")
    wf     = _walk_forward(trades, integrated)

    payload = {
        "experiment": "EXP-1880",
        "name": "Integrated FOMC + PCR Entry Overlays for EXP-1220",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data_sources": {
            "fomc": "data/fomc/ (federalreserve.gov 2015-2025, 89 meetings)",
            "pcr": "IronVault options_cache.db (real Polygon SPY option_daily volume)",
            "vix": "Yahoo Finance ^VIX, ^VIX3M, ^VIX9D",
            "options": "IronVault SPY chains via run_exp1220_trades",
        },
        "baseline":   base_m,
        "fomc_only":  fomc_m,
        "pcr_only":   pcr_m,
        "integrated": int_m,
        "delta_vs_baseline": {
            "fomc_only":  round(fomc_m["sharpe"] - base_m["sharpe"], 3),
            "pcr_only":   round(pcr_m["sharpe"]  - base_m["sharpe"], 3),
            "integrated": round(int_m["sharpe"]  - base_m["sharpe"], 3),
        },
        "walk_forward_integrated": wf,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    base = p["baseline"]
    rows = "".join(
        f"<tr><td>{name}</td><td>{p[name]['n']}</td><td>{p[name]['wr']*100:.1f}%</td>"
        f"<td>{p[name]['sharpe']:.2f}</td><td>{p[name]['cagr_pct']:.2f}%</td>"
        f"<td>{p[name]['max_dd_pct']:.2f}%</td><td>${p[name]['pnl']:.0f}</td>"
        f"<td class='{ 'ok' if p['delta_vs_baseline'].get(name,0)>=0 else 'bad'}'>"
        f"{p['delta_vs_baseline'].get(name, 0):+.2f}</td></tr>"
        for name in ("baseline", "fomc_only", "pcr_only", "integrated")
    )
    rows_w = "".join(
        f"<tr><td>{r['year']}</td><td>{r['baseline_n']}</td><td>{r['baseline_sharpe']:.2f}</td>"
        f"<td>{r['filtered_n']}</td><td>{r['filtered_sharpe']:.2f}</td>"
        f"<td class='{ 'ok' if r['delta_sharpe']>=0 else 'bad'}'>{r['delta_sharpe']:+.2f}</td></tr>"
        for r in p["walk_forward_integrated"]
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-1880 — Integrated Overlays</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .bad{{color:#b80000;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-1880 — Integrated FOMC + Put/Call Entry Overlays</h1>
<p class='small'>Generated {p['generated']} · 100% real data
 (federalreserve.gov FOMC + IronVault SPY P/C + Yahoo VIX) · Rule-Zero clean.</p>

<h2>Baseline vs each overlay</h2>
<table>
<tr><th>Variant</th><th>n</th><th>WR</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>PnL</th><th>ΔSharpe</th></tr>
{rows}
</table>

<h2>Walk-forward (integrated overlay) by year</h2>
<table>
<tr><th>Year</th><th>Base n</th><th>Base Sharpe</th><th>Filt n</th><th>Filt Sharpe</th><th>ΔSharpe</th></tr>
{rows_w}
</table>

<h2>Notes</h2>
<ul>
<li>FOMC filter: HD score ≥ {OverlayConfig().fomc_hawkish_thresh:.2f} blocks entries
    within {OverlayConfig().fomc_block_calendar_days}d window.</li>
<li>VIX-slope filter: requires ^VIX3M − ^VIX ≥ 0 (contango).</li>
<li>PCR filter: bottom {int(OverlayConfig().pcr_low_pct*100)}% rolling pct rank → block;
    top {int((1-OverlayConfig().pcr_high_pct)*100)}% → 1.30× size.</li>
<li>Vol-stress: VIX inversion or put-volume z &gt; 2 → block.</li>
<li>All filters independently switchable via <code>OverlayConfig</code>.</li>
</ul>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
