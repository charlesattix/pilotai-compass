"""
EXP-2120 — T+V+F Triple Overlay Integration for EXP-1220
=========================================================

Formal integration of the three overlays validated in Waves 2–3:
  T  = VIX term structure filter (EXP-2070): ratio = ^VIX/^VIX3M < 0.90
  V  = Vol-of-vol sizing         (EXP-1970): vvol z ≤ 1 full, 1–2 half, >2 zero
  F  = FOMC sentiment window     (EXP-1740): HD ≥ 0.30 within 7d → block

Runs the canonical EXP-1220 trade tape (171 real-IronVault trades) through
every non-empty combination of {T, V, F} and reports trade-level Sharpe,
n, win rate, CAGR, max DD, PnL for each.

Reference V+F baseline (from EXP-2070): Sharpe 1.00.
EXP-2070 also already computed T-alone < 0.90 → 2.08 and T+V+F < 0.90 → 2.42.
This experiment reproduces those numbers alongside the missing pair-wise
combinations (T+V, T+F) so every subset of the three overlays is on one page.

Rule Zero clean — same real data sources as the component experiments:
  * Yahoo Finance ^VIX / ^VIX3M
  * federalreserve.gov FOMC minutes 2015-2025 (89 meetings)
  * IronVault options_cache.db (real Polygon SPY chains)

Outputs
  compass/reports/exp2120_triple_overlay.json
  compass/reports/exp2120_triple_overlay.html
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp1220_standalone import run_exp1220_trades
from compass.exp2070_term_structure import (
    apply_filters,
    backwardation_filter,      # noqa: F401 (imported for parity)
    fomc_filter,
    load_fomc_hd,
    load_term_structure,
    term_filter,
    vvol_filter,
)
from compass.exp1970_vol_of_vol import build_vvol_panel
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2120_triple_overlay.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2120_triple_overlay.html"

TRADING_DAYS = 252
CAPITAL = 100_000
TERM_THRESHOLD = 0.90
VVOL_Z_MAX = 2.0
FOMC_HAWK_THRESH = 0.30
FOMC_BLOCK_DAYS = 7


def _metrics(trades: List[Dict], label: str) -> Dict:
    if not trades:
        return {"label": label, "n": 0, "pnl": 0.0, "wr": 0.0, "sharpe": 0.0,
                "cagr_pct": 0.0, "max_dd_pct": 0.0, "avg_pnl": 0.0, "trades_per_yr": 0.0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    eq = CAPITAL + pnl.cumsum()
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    yrs = max(1.0, (
        datetime.strptime(trades[-1]["exit_date"], "%Y-%m-%d")
        - datetime.strptime(trades[0]["entry_date"], "%Y-%m-%d")
    ).days / 365.25)
    tpy = len(pnl) / yrs
    rets = pnl / CAPITAL
    mu, sd = rets.mean(), (rets.std(ddof=1) if len(rets) > 1 else 0.0)
    sharpe = (mu / sd) * math.sqrt(tpy) if sd > 1e-12 else 0.0
    return {
        "label": label, "n": int(len(pnl)),
        "pnl": float(pnl.sum()), "wr": float((pnl > 0).mean()),
        "sharpe": round(float(sharpe), 3),
        "cagr_pct": round(float((eq[-1] / CAPITAL) ** (1 / yrs) * 100 - 100), 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "avg_pnl": round(float(pnl.mean()), 2),
        "trades_per_yr": round(float(tpy), 2),
    }


def _walk_forward(trades: List[Dict], filtered: List[Dict]) -> List[Dict]:
    def _yr(trs):
        out: Dict[int, List[Dict]] = {}
        for t in trs:
            out.setdefault(int(t["entry_date"][:4]), []).append(t)
        return out
    b = _yr(trades); f = _yr(filtered)
    rows = []
    for y in sorted(set(b) | set(f)):
        mb = _metrics(b.get(y, []), f"{y} base")
        mf = _metrics(f.get(y, []), f"{y} filt")
        rows.append({
            "year": y,
            "base_n": mb["n"], "base_sharpe": mb["sharpe"],
            "filt_n": mf["n"], "filt_sharpe": mf["sharpe"],
            "delta_sharpe": round(mf["sharpe"] - mb["sharpe"], 3),
        })
    return rows


def main():
    import yfinance as yf

    print("[1/4] building unified overlay panel …")
    ts_df = load_term_structure("2019-06-01", "2026-07-01")
    vix = ts_df["vix"]
    vvol = build_vvol_panel(vix)
    fomc = load_fomc_hd("2019-06-01", "2026-07-01")
    panel = ts_df[["vix", "vix3m", "ratio", "contango"]].join(
        vvol[["vvol", "vvol_z"]], how="left"
    ).join(fomc, how="left")

    print("[2/4] running EXP-1220 baseline on real IronVault …")
    hd = IronVault.instance()
    spy = yf.download("SPY", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).normalize()
    vix_daily = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix_daily, pd.DataFrame):
        vix_daily = vix_daily.iloc[:, 0]
    vix_daily.index = pd.to_datetime(vix_daily.index).normalize()
    trades = run_exp1220_trades(hd, spy, vix_daily)
    print(f"      {len(trades)} baseline trades")
    baseline = _metrics(trades, "baseline")

    print("[3/4] evaluating every {T,V,F} subset …")
    filters: Dict[str, Callable] = {
        "T": term_filter(TERM_THRESHOLD),
        "V": vvol_filter(),
        "F": fomc_filter(hawk_thresh=FOMC_HAWK_THRESH, window_days=FOMC_BLOCK_DAYS),
    }

    results: Dict[str, Dict] = {"baseline": baseline}
    # order: single, pair, triple
    ordered_subsets = []
    for k in (1, 2, 3):
        for combo in combinations(["T", "V", "F"], k):
            ordered_subsets.append("+".join(combo))

    for subset in ordered_subsets:
        keys = subset.split("+")
        stack = [filters[k] for k in keys]
        filt = apply_filters(trades, panel, stack)
        results[subset] = _metrics(filt, subset)

    # ranking
    ranked = sorted(
        [(k, v) for k, v in results.items() if k != "baseline"],
        key=lambda kv: kv[1]["sharpe"],
        reverse=True,
    )
    best_label, best = ranked[0]
    delta = round(best["sharpe"] - baseline["sharpe"], 3)

    # incremental decomposition — does the triple beat the best pair?
    best_pair_label, best_pair = max(
        [(k, v) for k, v in results.items() if "+" in k and len(k.split("+")) == 2],
        key=lambda kv: kv[1]["sharpe"],
    )
    triple = results["T+V+F"]
    incremental_over_best_pair = round(triple["sharpe"] - best_pair["sharpe"], 3)

    # walk-forward on the winning subset
    winning_filters = [filters[k] for k in best_label.split("+")]
    wf = _walk_forward(trades, apply_filters(trades, panel, winning_filters))

    print(f"[4/4] best subset: {best_label} Sharpe {best['sharpe']}  (Δ {delta:+.2f})")

    payload = {
        "experiment": "EXP-2120",
        "name": "Triple Overlay Integration (T+V+F) for EXP-1220",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "components": {
            "T": f"VIX/VIX3M ratio < {TERM_THRESHOLD}  (EXP-2070)",
            "V": f"vvol 20d/252d-z ≤ {VVOL_Z_MAX}    (EXP-1970)",
            "F": f"FOMC HD ≥ {FOMC_HAWK_THRESH} within {FOMC_BLOCK_DAYS}d → block  (EXP-1740)",
        },
        "data_sources": {
            "vix_vix3m": "Yahoo Finance ^VIX / ^VIX3M",
            "fomc":      "federalreserve.gov FOMC minutes 2015-2025 (89 meetings)",
            "options":   "IronVault options_cache.db (real Polygon SPY chains)",
        },
        "baseline": baseline,
        "variants": {k: results[k] for k in ["baseline"] + ordered_subsets},
        "ranked_by_sharpe": [
            {"subset": k, "sharpe": v["sharpe"], "n": v["n"],
             "cagr_pct": v["cagr_pct"], "max_dd_pct": v["max_dd_pct"]}
            for k, v in ranked
        ],
        "best": {"subset": best_label, **best,
                 "delta_sharpe_vs_baseline": delta},
        "triple_vs_best_pair": {
            "triple": {"sharpe": triple["sharpe"], "n": triple["n"]},
            "best_pair_label": best_pair_label,
            "best_pair": {"sharpe": best_pair["sharpe"], "n": best_pair["n"]},
            "delta_sharpe": incremental_over_best_pair,
            "triple_adds_incremental": incremental_over_best_pair > 0,
        },
        "walk_forward_best": wf,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    def row(k, v):
        return (f"<tr><td>{k}</td><td>{v['n']}</td><td>{v['wr']*100:.1f}%</td>"
                f"<td>{v['sharpe']:.2f}</td><td>{v['cagr_pct']:.2f}%</td>"
                f"<td>{v['max_dd_pct']:.2f}%</td><td>${v['pnl']:.0f}</td></tr>")

    rows_var = "".join(row(k, v) for k, v in p["variants"].items())
    rows_rank = "".join(
        f"<tr><td>{i+1}</td><td>{r['subset']}</td><td>{r['n']}</td>"
        f"<td>{r['sharpe']:.2f}</td><td>{r['cagr_pct']:.2f}%</td>"
        f"<td>{r['max_dd_pct']:.2f}%</td></tr>"
        for i, r in enumerate(p["ranked_by_sharpe"])
    )
    rows_wf = "".join(
        f"<tr><td>{r['year']}</td><td>{r['base_n']}</td><td>{r['base_sharpe']:.2f}</td>"
        f"<td>{r['filt_n']}</td><td>{r['filt_sharpe']:.2f}</td>"
        f"<td class='{ 'ok' if r['delta_sharpe']>=0 else 'bad'}'>{r['delta_sharpe']:+.2f}</td></tr>"
        for r in p["walk_forward_best"]
    )
    tvb = p["triple_vs_best_pair"]
    inc_cls = "ok" if tvb["triple_adds_incremental"] else "warn"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2120 — T+V+F Triple Overlay</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .warn{{color:#b86b00;font-weight:600}} .bad{{color:#b80000;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-2120 — T+V+F Triple Overlay Integration</h1>
<p class='small'>Generated {p['generated']} · Rule-Zero clean · Real Yahoo
 ^VIX/^VIX3M, real federalreserve.gov FOMC, real IronVault SPY chains.</p>

<h2>Components</h2>
<ul>
<li><b>T</b> — {p['components']['T']}</li>
<li><b>V</b> — {p['components']['V']}</li>
<li><b>F</b> — {p['components']['F']}</li>
</ul>

<h2>Every subset</h2>
<table>
<tr><th>Subset</th><th>n</th><th>WR</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>PnL</th></tr>
{rows_var}
</table>

<h2>Ranked by Sharpe</h2>
<table>
<tr><th>#</th><th>Subset</th><th>n</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th></tr>
{rows_rank}
</table>

<h2>Does the triple beat the best pair?</h2>
<p>Best pair: <b>{tvb['best_pair_label']}</b> — Sharpe {tvb['best_pair']['sharpe']:.2f} (n={tvb['best_pair']['n']})<br>
   T+V+F triple: Sharpe {tvb['triple']['sharpe']:.2f} (n={tvb['triple']['n']})<br>
   Δ Sharpe: <span class='{inc_cls}'>{tvb['delta_sharpe']:+.2f}</span>
   — {'triple ADDS incremental alpha' if tvb['triple_adds_incremental'] else 'triple does NOT add over best pair'}</p>

<h2>Walk-forward (winning subset: {p['best']['subset']})</h2>
<table>
<tr><th>Year</th><th>Base n</th><th>Base Sharpe</th><th>Filt n</th><th>Filt Sharpe</th><th>Δ</th></tr>
{rows_wf}
</table>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
