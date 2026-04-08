"""
EXP-1970 — Vol-of-Vol Overlay for EXP-1220 Credit Spreads
==========================================================

Hypothesis
----------
When the realised volatility of VIX itself is low, selling premium is
exceptionally safe — the vol surface is stable and mean-reverting.
When vol-of-vol is elevated, VIX can spike unpredictably → de-risk.

Signal
------
  1. vvol  = 20-day annualised std-dev of VIX daily log-returns
  2. zscore = rolling-252d z-score of vvol
  3. Position-size multiplier schedule:
        z <= 1  → 1.0   (calm / normal)
        1 < z <= 2 → 0.5 (elevated — half size)
        z >  2  → 0.0   (panic — no new entries)

Application
-----------
  * Overlay mode:    applied to the canonical 171-trade EXP-1220 tape
                     (real IronVault SPY chains). Sizes scaled on entry.
  * Standalone mode: same engine, but *only* enter on z <= 0 days so
                     we can measure the "calm vol" premium in isolation.

Outputs
-------
  compass/reports/exp1970_vol_of_vol.json
  compass/reports/exp1970_vol_of_vol.html

Rule Zero: VIX is real Yahoo ^VIX, options are real IronVault SPY.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp1220_standalone import run_exp1220_trades
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp1970_vol_of_vol.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp1970_vol_of_vol.html"

TRADING_DAYS = 252
CAPITAL = 100_000


# ─────────────────────────────────────────────────────────────────────────────
# Signal: vol-of-vol z-score
# ─────────────────────────────────────────────────────────────────────────────
def build_vvol_panel(vix_close: pd.Series,
                     realised_window: int = 20,
                     z_window: int = 252) -> pd.DataFrame:
    """Returns a DataFrame with columns:
         vix, vix_logret, vvol, vvol_z, size_mult
    """
    s = vix_close.dropna().copy()
    s.index = pd.to_datetime(s.index).normalize()
    logret = np.log(s / s.shift(1))
    vvol = logret.rolling(realised_window).std(ddof=1) * math.sqrt(TRADING_DAYS)
    mu = vvol.rolling(z_window).mean()
    sd = vvol.rolling(z_window).std(ddof=1)
    z = (vvol - mu) / sd
    size = pd.Series(1.0, index=s.index)
    size[z > 1.0] = 0.5
    size[z > 2.0] = 0.0
    return pd.DataFrame({
        "vix": s,
        "vix_logret": logret,
        "vvol": vvol,
        "vvol_z": z,
        "size_mult": size,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Overlay: scale each trade's pnl by the z-based multiplier at entry date
# ─────────────────────────────────────────────────────────────────────────────
def apply_overlay(trades: List[Dict], panel: pd.DataFrame,
                  *, standalone_only_calm: bool = False) -> List[Dict]:
    kept = []
    for t in trades:
        ed = pd.Timestamp(t["entry_date"]).normalize()
        if ed in panel.index:
            row = panel.loc[ed]
        else:
            idx = panel.index.searchsorted(ed) - 1
            if idx < 0:
                continue
            row = panel.iloc[idx]
        z = row["vvol_z"]
        mult = row["size_mult"]
        if pd.isna(z) or pd.isna(mult):
            # warmup — be permissive, full size
            mult = 1.0
            z = 0.0
        if standalone_only_calm and (pd.isna(z) or z > 0):
            continue
        if mult == 0.0:
            continue
        if mult != 1.0:
            nt = dict(t)
            nt["pnl"] = round(t["pnl"] * mult, 2)
            nt["contracts"] = max(1, int(round(t["contracts"] * mult)))
            nt["vvol_size_mult"] = float(mult)
            nt["vvol_z"] = float(z)
            kept.append(nt)
        else:
            nt = dict(t)
            nt["vvol_size_mult"] = 1.0
            nt["vvol_z"] = float(z)
            kept.append(nt)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(trades: List[Dict], label: str) -> Dict:
    if not trades:
        return {"label": label, "n": 0, "pnl": 0.0, "wr": 0.0, "sharpe": 0.0,
                "cagr_pct": 0.0, "max_dd_pct": 0.0, "avg_pnl": 0.0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    wins = int((pnl > 0).sum())
    equity = CAPITAL + pnl.cumsum()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    yrs = max(1.0, (
        datetime.strptime(trades[-1]["exit_date"], "%Y-%m-%d")
        - datetime.strptime(trades[0]["entry_date"], "%Y-%m-%d")
    ).days / 365.25)
    tpy = len(pnl) / yrs
    rets = pnl / CAPITAL
    mu, sd = rets.mean(), (rets.std(ddof=1) if len(rets) > 1 else 0.0)
    sharpe = (mu / sd) * math.sqrt(tpy) if sd > 1e-12 else 0.0
    return {
        "label": label, "n": len(pnl),
        "pnl": float(pnl.sum()), "wr": wins / len(pnl),
        "sharpe": round(float(sharpe), 3),
        "cagr_pct": round(float((equity[-1] / CAPITAL) ** (1 / yrs) * 100 - 100), 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "avg_pnl": round(float(pnl.mean()), 2),
        "trades_per_yr": round(float(tpy), 2),
    }


def walk_forward(baseline: List[Dict], overlaid: List[Dict]) -> List[Dict]:
    def _by_year(trs):
        out = {}
        for t in trs:
            out.setdefault(int(t["entry_date"][:4]), []).append(t)
        return out
    b_y = _by_year(baseline); o_y = _by_year(overlaid)
    years = sorted(set(b_y) | set(o_y))
    rows = []
    for y in years:
        b = metrics(b_y.get(y, []), f"{y} baseline")
        o = metrics(o_y.get(y, []), f"{y} overlay")
        rows.append({
            "year": y,
            "baseline_n": b["n"], "baseline_sharpe": b["sharpe"], "baseline_pnl": b["pnl"],
            "overlay_n":  o["n"], "overlay_sharpe":  o["sharpe"], "overlay_pnl":  o["pnl"],
            "delta_sharpe": round(o["sharpe"] - b["sharpe"], 3),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import yfinance as yf
    print("[1/5] downloading VIX from Yahoo …")
    vix = yf.download("^VIX", start="2018-01-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).normalize()
    print(f"      {len(vix)} daily closes")

    print("[2/5] building vol-of-vol panel …")
    panel = build_vvol_panel(vix)
    live = panel.dropna(subset=["vvol_z"])
    calm_pct = float((live["vvol_z"] <= 0).mean()) * 100
    half_pct = float(((live["vvol_z"] > 1) & (live["vvol_z"] <= 2)).mean()) * 100
    flat_pct = float((live["vvol_z"] > 2).mean()) * 100
    print(f"      regime mix — calm(z<=0): {calm_pct:.1f}%  "
          f"half(1<z<=2): {half_pct:.1f}%  flat(z>2): {flat_pct:.1f}%")

    print("[3/5] running EXP-1220 baseline trades on real IronVault …")
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
    print(f"      {len(trades)} trades")

    print("[4/5] applying overlay and standalone modes …")
    baseline = metrics(trades, "baseline")
    overlaid_trades = apply_overlay(trades, panel, standalone_only_calm=False)
    overlay = metrics(overlaid_trades, "overlay_sized")
    standalone_trades = apply_overlay(trades, panel, standalone_only_calm=True)
    standalone = metrics(standalone_trades, "standalone_calm_only")

    wf = walk_forward(trades, overlaid_trades)

    # regime-conditional trade stats
    regime_rows = []
    for label, sel in [
        ("z <= 0",  lambda z: z <= 0),
        ("0 < z <= 1", lambda z: (z > 0) & (z <= 1)),
        ("1 < z <= 2", lambda z: (z > 1) & (z <= 2)),
        ("z > 2",   lambda z: z > 2),
    ]:
        sub = []
        for t in trades:
            ed = pd.Timestamp(t["entry_date"]).normalize()
            if ed not in panel.index:
                continue
            zv = panel.loc[ed, "vvol_z"]
            if pd.isna(zv):
                continue
            if sel(zv):
                sub.append(t)
        regime_rows.append({"regime": label, **metrics(sub, label)})

    print("[5/5] writing report …")
    payload = {
        "experiment": "EXP-1970",
        "name": "Vol-of-Vol Overlay for EXP-1220 Credit Spreads",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data_sources": {
            "vix":     "Yahoo Finance ^VIX daily",
            "options": "IronVault options_cache.db (real Polygon SPY chains)",
        },
        "signal": {
            "realised_vol_window": 20,
            "z_score_window":      252,
            "size_schedule":       {"z<=1": 1.0, "1<z<=2": 0.5, "z>2": 0.0},
            "regime_mix_pct": {
                "calm_z_le_0":  round(calm_pct, 2),
                "half_1_lt_z_le_2": round(half_pct, 2),
                "flat_z_gt_2":  round(flat_pct, 2),
            },
        },
        "baseline":   baseline,
        "overlay":    overlay,
        "standalone": standalone,
        "delta_sharpe_overlay_vs_baseline": round(overlay["sharpe"] - baseline["sharpe"], 3),
        "target_overlay_delta": 0.5,
        "target_overlay_met": (overlay["sharpe"] - baseline["sharpe"]) >= 0.5,
        "target_standalone_sharpe": 2.0,
        "target_standalone_met": standalone["sharpe"] >= 2.0,
        "walk_forward_overlay": wf,
        "regime_conditional_baseline": regime_rows,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    rows_v = "".join(
        f"<tr><td>{p[k]['label']}</td><td>{p[k]['n']}</td>"
        f"<td>{p[k]['wr']*100:.1f}%</td><td>{p[k]['sharpe']:.2f}</td>"
        f"<td>{p[k]['cagr_pct']:.2f}%</td><td>{p[k]['max_dd_pct']:.2f}%</td>"
        f"<td>${p[k]['pnl']:.0f}</td></tr>"
        for k in ("baseline", "overlay", "standalone")
    )
    rows_w = "".join(
        f"<tr><td>{r['year']}</td><td>{r['baseline_n']}</td><td>{r['baseline_sharpe']:.2f}</td>"
        f"<td>{r['overlay_n']}</td><td>{r['overlay_sharpe']:.2f}</td>"
        f"<td class='{ 'ok' if r['delta_sharpe']>=0 else 'bad'}'>{r['delta_sharpe']:+.2f}</td></tr>"
        for r in p["walk_forward_overlay"]
    )
    rows_r = "".join(
        f"<tr><td>{r['regime']}</td><td>{r['n']}</td><td>{r['wr']*100:.1f}%</td>"
        f"<td>{r['sharpe']:.2f}</td><td>{r['cagr_pct']:.2f}%</td>"
        f"<td>${r['pnl']:.0f}</td></tr>"
        for r in p["regime_conditional_baseline"]
    )
    ov_cls = "ok" if p["target_overlay_met"] else "warn"
    sa_cls = "ok" if p["target_standalone_met"] else "warn"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-1970 — Vol-of-Vol Overlay</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .warn{{color:#b86b00;font-weight:600}} .bad{{color:#b80000;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-1970 — Vol-of-Vol Overlay</h1>
<p class='small'>Generated {p['generated']} · Real Yahoo ^VIX · Real IronVault SPY chains · Rule Zero clean.</p>

<h2>Headline</h2>
<table>
<tr><th>Variant</th><th>n</th><th>WR</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>PnL</th></tr>
{rows_v}
</table>
<p>Overlay ΔSharpe vs baseline: <b>{p['delta_sharpe_overlay_vs_baseline']:+.2f}</b>
  &nbsp;·&nbsp; target +0.50 <span class='{ov_cls}'>{'MET' if p['target_overlay_met'] else 'NOT MET'}</span>
  &nbsp;·&nbsp; standalone Sharpe target 2.0 <span class='{sa_cls}'>{'MET' if p['target_standalone_met'] else 'NOT MET'}</span></p>

<h2>Walk-forward (overlay) by year</h2>
<table>
<tr><th>Year</th><th>Base n</th><th>Base Sharpe</th><th>Ovly n</th><th>Ovly Sharpe</th><th>Δ</th></tr>
{rows_w}
</table>

<h2>Regime-conditional baseline (descriptive only)</h2>
<table>
<tr><th>Regime</th><th>n</th><th>WR</th><th>Sharpe</th><th>CAGR</th><th>PnL</th></tr>
{rows_r}
</table>

<h2>Notes</h2>
<ul>
<li>vvol = 20d annualised std-dev of log(VIX); z = 252d rolling z-score.</li>
<li>Regime mix over full sample — calm (z≤0): {p['signal']['regime_mix_pct']['calm_z_le_0']}% ·
    half (1&lt;z≤2): {p['signal']['regime_mix_pct']['half_1_lt_z_le_2']}% ·
    flat (z&gt;2): {p['signal']['regime_mix_pct']['flat_z_gt_2']}%.</li>
<li>Overlay scales each trade's pnl AND contract count by the size multiplier at its entry date; half/flat regimes cut exposure rather than leverage up the others.</li>
<li>Standalone mode keeps only trades entered on calm (z≤0) days, unscaled.</li>
</ul>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
