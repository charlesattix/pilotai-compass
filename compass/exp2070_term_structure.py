"""
EXP-2070 — VIX Term Structure Alpha as an EXP-1220 Entry Overlay
================================================================

Hypothesis
----------
The VIX/VIX3M ratio summarises the shape of the short-end VIX curve:
  ratio < 1  ⇒ contango (near-vol below 3m) — a calm, complacent regime
               where credit-spread sellers are paid a stable premium.
  ratio > 1  ⇒ backwardation (near-vol above 3m) — a panic/stress regime
               where vol can spike further and credit spreads blow up.

Selling premium *only* when the near end of the curve is below the
three-month point should therefore tilt the entry tape toward calm
regimes and lift risk-adjusted returns.

Procedure
---------
1. Build a daily ^VIX / ^VIX3M ratio from Yahoo Finance (real).
2. Run the canonical 171-trade EXP-1220 tape on real IronVault SPY chains.
3. Apply the term-structure filter at entry date — skip trades when the
   ratio is above threshold. Threshold sweep: 0.80, 0.85, 0.90, 0.95,
   plus a skip-backwardation-only variant (ratio > 1.0 blocks).
4. Walk-forward by year.
5. Stack on top of V+F (vol-of-vol + FOMC hawkish window) to see
   whether the term-structure signal adds *incremental* Sharpe beyond
   the Wave-2 overlays already in production.

Outputs
-------
  compass/reports/exp2070_term_structure.json
  compass/reports/exp2070_term_structure.html

Rule Zero clean.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp1220_standalone import run_exp1220_trades
from compass.exp1740_sentiment_filter import parse_fomc_minutes
from compass.exp1970_vol_of_vol import build_vvol_panel
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2070_term_structure.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2070_term_structure.html"

TRADING_DAYS = 252
CAPITAL = 100_000


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_term_structure(start: str, end: str) -> pd.DataFrame:
    """Daily ^VIX / ^VIX3M ratio from real Yahoo data."""
    import yfinance as yf
    out = {}
    for sym, key in [("^VIX", "vix"), ("^VIX3M", "vix3m")]:
        d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        out[key] = d["Close"]
    df = pd.DataFrame(out).dropna()
    df.index = pd.to_datetime(df.index).normalize()
    df["ratio"] = df["vix"] / df["vix3m"]
    df["contango"] = (df["ratio"] < 1.0).astype(int)
    return df


def load_fomc_hd(start: str, end: str) -> pd.Series:
    """Per-trading-day FOMC HD-score using latest release at-or-before date."""
    import yfinance as yf
    idx = yf.download("SPY", start=start, end=end, progress=False)
    if isinstance(idx.columns, pd.MultiIndex):
        idx.columns = idx.columns.get_level_values(0)
    idx.index = pd.to_datetime(idx.index).normalize()
    idx = idx.index
    feats = parse_fomc_minutes()
    fd = pd.DataFrame([f.__dict__ for f in feats])
    fd["date"] = pd.to_datetime(fd["date"])
    fd = fd.sort_values("date").reset_index(drop=True)
    hd_col = np.full(len(idx), np.nan)
    ds_col = np.full(len(idx), np.nan)
    j = 0
    for i, day in enumerate(idx):
        while j + 1 < len(fd) and fd.iloc[j + 1]["date"] <= day:
            j += 1
        if fd.iloc[j]["date"] <= day:
            hd_col[i] = fd.iloc[j]["hd_score"]
            ds_col[i] = (day - fd.iloc[j]["date"]).days
    return pd.DataFrame({"fomc_hd": hd_col, "fomc_days_since": ds_col}, index=idx)


# ─────────────────────────────────────────────────────────────────────────────
# Filters
# ─────────────────────────────────────────────────────────────────────────────
def term_filter(threshold: float) -> Callable[[pd.Series], bool]:
    """Allow only when VIX/VIX3M ratio < threshold (i.e. contango, calm)."""
    def _f(row: pd.Series) -> bool:
        r = row.get("ratio")
        if r is None or pd.isna(r):
            return True   # permissive during warmup / missing data
        return bool(r < threshold)
    return _f


def backwardation_filter() -> Callable[[pd.Series], bool]:
    """Allow except when curve is inverted (ratio > 1.0)."""
    def _f(row: pd.Series) -> bool:
        r = row.get("ratio")
        if r is None or pd.isna(r):
            return True
        return bool(r <= 1.0)
    return _f


def vvol_filter() -> Callable[[pd.Series], bool]:
    """EXP-1970 vol-of-vol: block when vvol_z > 2."""
    def _f(row: pd.Series) -> bool:
        z = row.get("vvol_z")
        if z is None or pd.isna(z):
            return True
        return bool(z <= 2.0)
    return _f


def fomc_filter(hawk_thresh: float = 0.30, window_days: int = 7) -> Callable[[pd.Series], bool]:
    """EXP-1740 FOMC hawkish-window block."""
    def _f(row: pd.Series) -> bool:
        hd = row.get("fomc_hd")
        ds = row.get("fomc_days_since")
        if hd is None or pd.isna(hd) or ds is None or pd.isna(ds):
            return True
        return not (hd >= hawk_thresh and ds <= window_days)
    return _f


def apply_filters(trades: List[Dict], panel: pd.DataFrame,
                  filters: List[Callable[[pd.Series], bool]]) -> List[Dict]:
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
        if all(f(row) for f in filters):
            kept.append(t)
    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(trades: List[Dict], label: str) -> Dict:
    if not trades:
        return {"label": label, "n": 0, "pnl": 0.0, "wr": 0.0, "sharpe": 0.0,
                "cagr_pct": 0.0, "max_dd_pct": 0.0, "avg_pnl": 0.0}
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


def walk_forward(trades: List[Dict], filtered: List[Dict]) -> List[Dict]:
    def _yr(trs):
        out = {}
        for t in trs:
            out.setdefault(int(t["entry_date"][:4]), []).append(t)
        return out
    b = _yr(trades); f = _yr(filtered)
    rows = []
    for y in sorted(set(b) | set(f)):
        mb = metrics(b.get(y, []), f"{y} base")
        mf = metrics(f.get(y, []), f"{y} filt")
        rows.append({
            "year": y,
            "base_n": mb["n"], "base_sharpe": mb["sharpe"], "base_pnl": mb["pnl"],
            "filt_n": mf["n"], "filt_sharpe": mf["sharpe"], "filt_pnl": mf["pnl"],
            "delta_sharpe": round(mf["sharpe"] - mb["sharpe"], 3),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import yfinance as yf

    print("[1/5] downloading VIX term structure …")
    ts = load_term_structure("2019-06-01", "2026-07-01")
    print(f"      {len(ts)} days  contango days: {int(ts['contango'].sum())} "
          f"({ts['contango'].mean()*100:.1f}%)")

    print("[2/5] building unified overlay panel (term + vvol + FOMC) …")
    vix = ts["vix"]
    vvol = build_vvol_panel(vix).rename_axis("date")
    fomc = load_fomc_hd("2019-06-01", "2026-07-01").rename_axis("date")
    panel = ts[["vix", "vix3m", "ratio", "contango"]].copy()
    panel = panel.join(vvol[["vvol", "vvol_z"]], how="left")
    panel = panel.join(fomc, how="left")

    print("[3/5] running EXP-1220 baseline on real IronVault …")
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
    baseline = metrics(trades, "baseline")

    print("[4/5] threshold sweep and stacked comparisons …")
    variants = {}

    # Threshold sweep
    for thr in (0.80, 0.85, 0.90, 0.95):
        f = apply_filters(trades, panel, [term_filter(thr)])
        variants[f"term_only_ratio_lt_{thr:.2f}"] = metrics(f, f"term < {thr:.2f}")

    # Backwardation-only: block when ratio > 1.0
    f_back = apply_filters(trades, panel, [backwardation_filter()])
    variants["skip_backwardation_only"] = metrics(f_back, "skip_backwardation_only")

    # Standalone V and F
    f_v = apply_filters(trades, panel, [vvol_filter()])
    variants["V_only_vvol"] = metrics(f_v, "V_only_vvol")

    f_f = apply_filters(trades, panel, [fomc_filter()])
    variants["F_only_fomc"] = metrics(f_f, "F_only_fomc")

    # Stacked V+F
    f_vf = apply_filters(trades, panel, [vvol_filter(), fomc_filter()])
    variants["V_plus_F"] = metrics(f_vf, "V_plus_F")

    # Stacked V+F+T at each threshold — does term add?
    for thr in (0.85, 0.90, 0.95):
        f_vft = apply_filters(trades, panel,
                               [vvol_filter(), fomc_filter(), term_filter(thr)])
        variants[f"V_plus_F_plus_T_lt_{thr:.2f}"] = metrics(
            f_vft, f"V+F+T term<{thr:.2f}")

    # Pick best single-T threshold and best stacked result
    term_only = {k: v for k, v in variants.items() if k.startswith("term_only")}
    best_term = max(term_only.values(), key=lambda v: v["sharpe"])
    best_stacked = max(
        (v for k, v in variants.items() if k.startswith("V_plus_F_plus_T")),
        key=lambda v: v["sharpe"],
    )
    vf_base = variants["V_plus_F"]
    incremental = round(best_stacked["sharpe"] - vf_base["sharpe"], 3)

    # Walk-forward on the best single-term threshold
    best_thr_label = next(k for k, v in term_only.items() if v is best_term)
    best_thr = float(best_thr_label.split("_")[-1])
    wf_term = walk_forward(trades, apply_filters(trades, panel, [term_filter(best_thr)]))

    # Regime breakdown on baseline (descriptive)
    regime_rows = []
    for label, sel in [
        ("ratio < 0.85",       lambda r: r < 0.85),
        ("0.85 ≤ ratio < 0.90", lambda r: 0.85 <= r < 0.90),
        ("0.90 ≤ ratio < 0.95", lambda r: 0.90 <= r < 0.95),
        ("0.95 ≤ ratio < 1.00", lambda r: 0.95 <= r < 1.00),
        ("ratio ≥ 1.00 (inv)", lambda r: r >= 1.00),
    ]:
        sub = []
        for t in trades:
            ed = pd.Timestamp(t["entry_date"]).normalize()
            if ed not in panel.index:
                continue
            r = panel.loc[ed, "ratio"]
            if pd.isna(r):
                continue
            if sel(float(r)):
                sub.append(t)
        regime_rows.append({"regime": label, **metrics(sub, label)})

    print("[5/5] writing report …")
    payload = {
        "experiment": "EXP-2070",
        "name": "VIX Term Structure Alpha — EXP-1220 Entry Overlay",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data_sources": {
            "term_structure": "Yahoo Finance ^VIX / ^VIX3M daily ratio",
            "fomc": "data/fomc/ (federalreserve.gov 2015-2025, 89 meetings)",
            "vvol": "Yahoo ^VIX → 20d realised / 252d z (EXP-1970)",
            "options": "IronVault options_cache.db (real Polygon SPY chains)",
        },
        "baseline": baseline,
        "variants": variants,
        "best_term_only": {"label": best_thr_label, **best_term},
        "v_plus_f_baseline": vf_base,
        "best_stacked_vft": best_stacked,
        "incremental_sharpe_over_vf": incremental,
        "term_adds_incremental": incremental > 0,
        "walk_forward_best_term_only": wf_term,
        "regime_conditional_baseline": regime_rows,
        "target_delta_sharpe": 0.5,
        "target_met_term_only": (best_term["sharpe"] - baseline["sharpe"]) >= 0.5,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    base = p["baseline"]
    def drow(m, key_label=None):
        return (f"<tr><td>{key_label or m['label']}</td><td>{m['n']}</td>"
                f"<td>{m['wr']*100:.1f}%</td><td>{m['sharpe']:.2f}</td>"
                f"<td>{m['cagr_pct']:.2f}%</td><td>{m['max_dd_pct']:.2f}%</td>"
                f"<td>${m['pnl']:.0f}</td></tr>")
    rows_var = drow(base, "baseline")
    for k, v in p["variants"].items():
        rows_var += drow(v, k)
    rows_wf = "".join(
        f"<tr><td>{r['year']}</td><td>{r['base_n']}</td><td>{r['base_sharpe']:.2f}</td>"
        f"<td>{r['filt_n']}</td><td>{r['filt_sharpe']:.2f}</td>"
        f"<td class='{ 'ok' if r['delta_sharpe']>=0 else 'bad'}'>{r['delta_sharpe']:+.2f}</td></tr>"
        for r in p["walk_forward_best_term_only"]
    )
    rows_reg = "".join(drow(r, r["regime"]) for r in p["regime_conditional_baseline"])
    inc_cls = "ok" if p["term_adds_incremental"] else "warn"
    tgt_cls = "ok" if p["target_met_term_only"] else "warn"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2070 — VIX Term Structure Overlay</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .warn{{color:#b86b00;font-weight:600}} .bad{{color:#b80000;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-2070 — VIX Term Structure Alpha</h1>
<p class='small'>Generated {p['generated']} · Real Yahoo ^VIX &amp; ^VIX3M ·
  Real IronVault SPY chains · Rule Zero clean.</p>

<h2>All variants vs baseline</h2>
<table>
<tr><th>Variant</th><th>n</th><th>WR</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>PnL</th></tr>
{rows_var}
</table>

<h2>Best term-only</h2>
<p>Best threshold: <b>{p['best_term_only']['label']}</b>,
   Sharpe {p['best_term_only']['sharpe']:.2f}
   (baseline {base['sharpe']:.2f},
   Δ {p['best_term_only']['sharpe']-base['sharpe']:+.2f}) —
   target +0.50: <span class='{tgt_cls}'>{'MET' if p['target_met_term_only'] else 'NOT MET'}</span></p>

<h2>Incremental over V+F (vol-of-vol + FOMC)</h2>
<p>V+F stacked Sharpe: <b>{p['v_plus_f_baseline']['sharpe']:.2f}</b> (n={p['v_plus_f_baseline']['n']})<br>
   Best V+F+T stacked: <b>{p['best_stacked_vft']['label']}</b>, Sharpe <b>{p['best_stacked_vft']['sharpe']:.2f}</b>
   (n={p['best_stacked_vft']['n']})<br>
   Δ Sharpe from adding T on top of V+F: <span class='{inc_cls}'>{p['incremental_sharpe_over_vf']:+.2f}</span>
   ({'T adds' if p['term_adds_incremental'] else 'T does NOT add'} incremental alpha)</p>

<h2>Walk-forward (best term-only)</h2>
<table>
<tr><th>Year</th><th>Base n</th><th>Base Sharpe</th><th>Filt n</th><th>Filt Sharpe</th><th>Δ</th></tr>
{rows_wf}
</table>

<h2>Regime-conditional baseline (descriptive)</h2>
<table>
<tr><th>Regime</th><th>n</th><th>WR</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>PnL</th></tr>
{rows_reg}
</table>

<h2>Notes</h2>
<ul>
<li>Term ratio = ^VIX / ^VIX3M (real Yahoo daily close).</li>
<li>Threshold-lt filter allows entries only on days where ratio &lt; threshold.</li>
<li>Skip-backwardation variant blocks only when ratio ≥ 1.0.</li>
<li>Stacked V+F+T applies all three filters in AND; every filter is
    independently switchable so production can pick any subset.</li>
</ul>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
