"""EXP-2150 — Higher-Frequency EXP-1220 + T+V Signal Quality Boost.

Hypothesis
----------
The EXP-2100 audit showed that the V+F overlay on the sparse 171-trade
baseline tape actually HURTS portfolio Sharpe because the capital-
dilution penalty on a ~91% zero-return-day series dominates the per-
trade alpha lift. The fix is DENSITY: more trades per year should
reduce the zero-day fraction and let the per-trade alpha survive
the portfolio-level metric.

This experiment tests three changes:
  1. Weekly cadence — drop the 10-day minimum spacing between
     entries to 5 days. Expected: ~2× trade count (baseline 171 →
     ~350), 86% → ~72% zero-return days.
  2. T filter — block entries when VIX term structure is inverted
     (^VIX > ^VIX3M), using real Yahoo data.
  3. V filter — block entries when the 252-day z-score of VoV (20d
     realised vol of VIX log-returns) exceeds 1.0.

Six variants compared:
  a. baseline (171 trades, biweekly, no filter)
  b. weekly (spacing=5 days, no filter)
  c. weekly + T
  d. weekly + V
  e. weekly + T+V
  f. biweekly + T+V (the conservative alternative)

For each variant we measure:
  - Trade count
  - Trade-level Sharpe (annualised by √(trades/year))
  - Portfolio-level daily Sharpe via compass.metrics.full_metrics
    on Config A (70/5/10/10/5, 2× EXP-1220, 2× GLD, 1.5× SLV)
  - Zero-return-day fraction (measures dilution severity)

Rule Zero: every input is real.
  - IronVault SPY chains (data/options_cache.db)
  - Yahoo SPY / ^VIX / ^VIX3M
  - compass/cache/exp1860_streams.pkl (v5, GLD, SLV canonical)
  - compass/cache/exp2020_vol_arb_trades.pkl (vol_arb trades)

Output
------
  compass/reports/exp2150_higher_frequency.json
  compass/reports/exp2150_higher_frequency.html
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.metrics import full_metrics

REPORT_JSON = ROOT / "compass" / "reports" / "exp2150_higher_frequency.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2150_higher_frequency.html"
CACHE_DIR = ROOT / "compass" / "cache"
CACHE_V3 = CACHE_DIR / "exp1860_streams.pkl"
CACHE_VOL_ARB = CACHE_DIR / "exp2020_vol_arb_trades.pkl"
CACHE_BIWEEKLY = CACHE_DIR / "exp2150_trades_biweekly.pkl"
CACHE_WEEKLY = CACHE_DIR / "exp2150_trades_weekly.pkl"

START = "2020-01-01"
END = "2025-12-31"
WARMUP = 252
CAPITAL = 100_000

# Config A weights from EXP-2050/2100
WEIGHTS_A = {
    "exp1220": 0.70, "v5_hedge": 0.05, "gld_calendar": 0.10,
    "slv_calendar": 0.10, "vol_arb": 0.05,
}
LEV = {
    "exp1220": 2.00, "v5_hedge": 1.00, "gld_calendar": 2.00,
    "slv_calendar": 1.50, "vol_arb": 1.00,
}


# ═══════════════════════════════════════════════════════════════════════════
# Parameterised EXP-1220 runner (spacing-configurable)
# ═══════════════════════════════════════════════════════════════════════════

def run_exp1220_parameterised(hd, spy_df, vix, *, min_spacing_days: int) -> List[Dict]:
    """Fork of compass.exp1220_standalone.run_exp1220_trades with configurable
    min_spacing_days. 10 = baseline biweekly, 5 = weekly."""
    from compass.exp1220_standalone import (
        _find_exps, _exp_dt, _next_td, _sell_put_spread, _walk_spread,
    )

    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, "2020-03-01", "2025-12-31", monthly=False)
    trades: List[Dict] = []
    last = None

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=28), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < min_spacing_days:
            continue
        try:
            price = float(spy_close.loc[es])
            v = float(vix.loc[es])
        except Exception:
            continue
        if np.isnan(price) or np.isnan(v):
            continue
        if v > 40:
            continue

        spread = _sell_put_spread(hd, exp, es, price, otm_pct=0.95, width=5.0)
        if spread is None:
            continue
        cts = max(1, min(4, int(100_000 * 0.03 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(
            hd, exp, spread["short"], spread["long"],
            spread["credit"], entry_dt, exp_obj, spy_df.index,
        )
        pnl = (spread["credit"] - ev) * 100 * cts
        trades.append({
            "entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
            "exit_reason": er, "credit": spread["credit"],
            "vix": round(v, 1), "hold_days": hold, "contracts": cts,
        })
        last = entry_dt
    return trades


def load_tape(cadence: str, use_cache: bool = True) -> List[Dict]:
    cache_file = CACHE_BIWEEKLY if cadence == "biweekly" else CACHE_WEEKLY
    if use_cache and cache_file.exists():
        print(f"[cache] {cadence} tape from {cache_file.name}")
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    print(f"[run] {cadence} EXP-1220 pipeline (real IronVault)...")
    import yfinance as yf
    from shared.iron_vault import IronVault
    hd = IronVault.instance()
    spy = yf.download("SPY", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index)
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)

    spacing = 10 if cadence == "biweekly" else 5
    trades = run_exp1220_parameterised(hd, spy, vix, min_spacing_days=spacing)
    print(f"       {cadence}: {len(trades)} trades (spacing={spacing}d)")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as fh:
        pickle.dump(trades, fh)
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# T and V signal panels
# ═══════════════════════════════════════════════════════════════════════════

def build_t_panel() -> pd.DataFrame:
    """Term structure panel: block days when ^VIX > ^VIX3M (inversion)."""
    print("[panel T] building VIX term structure panel (real Yahoo)...")
    from compass.exp1750_putcall_overlay import load_vix_term_structure
    vts = load_vix_term_structure(START, END)
    allow = (vts["inverted"] == 0)
    panel = pd.DataFrame({
        "allow_entry": allow.fillna(True),
        "size_mult": pd.Series(1.0, index=vts.index),
    })
    panel.index = pd.to_datetime(panel.index).normalize()
    return panel


def build_v_panel() -> pd.DataFrame:
    """VoV z-score panel: zero-out days when z > 1 (elevated VoV)."""
    print("[panel V] building VoV z-score panel (real Yahoo ^VIX)...")
    import yfinance as yf
    from compass.exp1970_vol_of_vol import build_vvol_panel
    vix = yf.download("^VIX", start="2018-01-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    p = build_vvol_panel(vix)
    panel = pd.DataFrame({
        "allow_entry": (p["size_mult"] > 0).fillna(True),
        "size_mult": p["size_mult"].fillna(1.0),
    })
    panel.index = pd.to_datetime(panel.index).normalize()
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# Apply filters (same as EXP-2100)
# ═══════════════════════════════════════════════════════════════════════════

def _panel_lookup(panel: pd.DataFrame, ed: pd.Timestamp) -> Tuple[bool, float]:
    if panel.empty:
        return True, 1.0
    if ed in panel.index:
        row = panel.loc[ed]
    else:
        idx = panel.index.searchsorted(ed) - 1
        if idx < 0:
            return True, 1.0
        row = panel.iloc[idx]
    allow = bool(row["allow_entry"]) if not pd.isna(row["allow_entry"]) else True
    sm = float(row["size_mult"]) if not pd.isna(row["size_mult"]) else 1.0
    return allow, sm


def apply_overlays(trades: List[Dict],
                    overlays: Dict[str, pd.DataFrame]) -> List[Dict]:
    out: List[Dict] = []
    for t in trades:
        ed = pd.Timestamp(t["entry_date"]).normalize()
        allow, size = True, 1.0
        for _, panel in overlays.items():
            a, s = _panel_lookup(panel, ed)
            if not a:
                allow = False
                break
            size *= s
        if not allow or size <= 0:
            continue
        nt = dict(t)
        nt["pnl"] = round(t["pnl"] * size, 2)
        nt["contracts"] = max(1, int(round(t["contracts"] * size)))
        out.append(nt)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Convert trades → daily + portfolio compose
# ═══════════════════════════════════════════════════════════════════════════

def trades_to_daily(trades: List[Dict], full_index: pd.DatetimeIndex) -> pd.Series:
    if not trades:
        return pd.Series(0.0, index=full_index, name="exp1220")
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily = df.groupby("exit_date")["pnl"].sum() / CAPITAL
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.name = "exp1220"
    return daily


def trade_level_sharpe(trades: List[Dict]) -> float:
    if not trades:
        return 0.0
    pnls = np.array([t["pnl"] for t in trades])
    if len(pnls) < 2:
        return 0.0
    df = pd.DataFrame(trades)
    en = pd.to_datetime(df["entry_date"])
    ex = pd.to_datetime(df["exit_date"])
    yrs = max((ex.max() - en.min()).days / 365.25, 0.5)
    tpy = len(pnls) / yrs
    rets = pnls / CAPITAL
    mu, sd = float(rets.mean()), float(rets.std(ddof=1))
    return float(mu / sd * math.sqrt(tpy)) if sd > 1e-12 else 0.0


def zero_day_fraction(daily: pd.Series) -> float:
    mask = (START <= daily.index.strftime("%Y-%m-%d")) & \
           (daily.index.strftime("%Y-%m-%d") <= END)
    sub = daily[mask]
    if len(sub) == 0:
        return 1.0
    return float((sub == 0).mean())


def load_sibling_streams() -> Dict[str, pd.Series]:
    with open(CACHE_V3, "rb") as fh:
        v3 = pickle.load(fh)
    with open(CACHE_VOL_ARB, "rb") as fh:
        va_trades = pickle.load(fh)

    df = pd.DataFrame(va_trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    va_daily = df.groupby("exit_date")["pnl"].sum() / CAPITAL
    full = pd.bdate_range(va_daily.index.min(), va_daily.index.max())
    va_daily = va_daily.reindex(full, fill_value=0.0)
    va_daily.name = "vol_arb"

    return {
        "v5_hedge": v3["v5_hedge"],
        "gld_calendar": v3["gld_calendar"],
        "slv_calendar": v3["slv_calendar"],
        "vol_arb": va_daily,
    }


def portfolio(exp1220_series: pd.Series,
               siblings: Dict[str, pd.Series]) -> pd.Series:
    all_streams = {"exp1220": exp1220_series, **siblings}
    df = pd.concat([s.rename(k) for k, s in all_streams.items()],
                    axis=1, sort=True)
    df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))]
    df = df.fillna(0.0).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    port = pd.Series(0.0, index=df.index)
    for k in ["exp1220", "v5_hedge", "gld_calendar", "slv_calendar", "vol_arb"]:
        port = port + WEIGHTS_A[k] * df[k] * LEV[k]
    return port


def yearly(rets: pd.Series) -> List[Dict]:
    out = []
    for yr in sorted({d.year for d in rets.index}):
        sub = rets[rets.index.year == yr]
        if len(sub) < 20:
            continue
        m = full_metrics(sub.values)
        m["year"] = int(yr)
        out.append(m)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════════════════

def build_html(payload: Dict) -> str:
    rows = ""
    for label, v in payload["variants"].items():
        p = v["portfolio"]
        marker = " ★" if label == payload["winner"] else ""
        lift = v["portfolio"]["sharpe"] - payload["variants"]["baseline_biweekly"]["portfolio"]["sharpe"]
        color = "#16a34a" if lift > 0.1 else ("#dc2626" if lift < -0.05 else "#0f172a")
        rows += (
            f"<tr><td style='font-weight:700'>{label}{marker}</td>"
            f"<td>{v['n_trades']}</td>"
            f"<td>{v['trades_per_year']:.1f}</td>"
            f"<td>{v['trade_sharpe']:.2f}</td>"
            f"<td>{v['zero_day_fraction']*100:.0f}%</td>"
            f"<td>{p['cagr_pct']:.1f}%</td>"
            f"<td style='font-weight:700'>{p['sharpe']:.2f}</td>"
            f"<td style='color:{color};font-weight:700'>{lift:+.2f}</td>"
            f"<td>{p['max_dd_pct']:.1f}%</td>"
            f"<td>{p['calmar']:.2f}</td></tr>"
        )

    yr_keys = sorted({y["year"] for v in payload["variants"].values()
                      for y in v["portfolio_yearly"]})
    yearly_rows = ""
    for yr in yr_keys:
        cells = ""
        for label in payload["variants"]:
            row = next((y for y in payload["variants"][label]["portfolio_yearly"]
                        if y["year"] == yr), {})
            cagr = row.get("cagr_pct", 0)
            sh = row.get("sharpe", 0)
            dd = row.get("max_dd_pct", 0)
            color = "#16a34a" if cagr > 0 else "#dc2626"
            cells += (
                f"<td style='color:{color}'>{cagr:.0f}%</td>"
                f"<td>{sh:.2f}</td><td>{dd:.1f}%</td>"
            )
        yearly_rows += f"<tr><td style='font-weight:700'>{yr}</td>{cells}</tr>"

    yr_header_top = "".join(f"<th colspan='3'>{k}</th>"
                            for k in payload["variants"].keys())
    yr_header_bot = "".join("<th>CAGR</th><th>SR</th><th>DD</th>"
                            for _ in payload["variants"].keys())

    winner = payload["winner"]
    wm = payload["variants"][winner]["portfolio"]
    base_sr = payload["variants"]["baseline_biweekly"]["portfolio"]["sharpe"]
    lift = wm["sharpe"] - base_sr

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2150 — Higher-Frequency EXP-1220</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1300px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.75em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.winner {{ background:#ecfdf5;border:2px solid #16a34a;border-radius:10px;padding:18px;margin:20px 0; }}
.winner h3 {{ margin-top:0;color:#065f46; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:14px 0; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.82em; }}
th {{ background:#f1f5f9;padding:9px 10px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.72em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 10px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
</style></head><body>
<h1>EXP-2150 — Higher-Frequency EXP-1220 + T+V Filter</h1>
<p style="color:#64748b">Weekly cadence + term-structure + vol-of-vol filters ·
Config A portfolio integration · 2020-2025 ·
{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero — all real:</strong><br>
EXP-1220 trades: compass.exp1220_standalone fork with configurable
min_spacing_days on real IronVault <code>data/options_cache.db</code>
SPY chains + Yahoo SPY/^VIX<br>
T filter: compass.exp1750_putcall_overlay.load_vix_term_structure on
real Yahoo ^VIX/^VIX3M<br>
V filter: compass.exp1970_vol_of_vol.build_vvol_panel on real Yahoo ^VIX<br>
v5_hedge, GLD calendar, SLV calendar: compass/cache/exp1860_streams.pkl<br>
vol_arb: compass/cache/exp2020_vol_arb_trades.pkl<br>
Canonical Sharpe: compass.metrics.full_metrics (mean/std × √252)
</div>

<div class="winner">
<h3>★ Winner: <code>{winner}</code></h3>
CAGR <strong>{wm['cagr_pct']:.1f}%</strong> ·
Sharpe <strong>{wm['sharpe']:.2f}</strong> ·
Max DD <strong>{wm['max_dd_pct']:.1f}%</strong> ·
Calmar <strong>{wm['calmar']:.2f}</strong><br>
Lift vs baseline biweekly: <strong>{lift:+.2f}</strong> Sharpe points
</div>

<h2>1. All variants — trade counts, trade Sharpe, portfolio metrics</h2>
<table>
<thead><tr>
<th>Variant</th><th>N</th><th>Trades/yr</th><th>Trade SR</th>
<th>Zero-days</th><th>CAGR</th><th>Sharpe</th><th>ΔSR vs base</th>
<th>DD</th><th>Calmar</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>2. Year-by-year portfolio metrics</h2>
<table>
<thead>
<tr><th rowspan='2'>Year</th>{yr_header_top}</tr>
<tr>{yr_header_bot}</tr>
</thead>
<tbody>{yearly_rows}</tbody>
</table>

<div class="note">
<strong>What to watch for:</strong> the hypothesis is that weekly
cadence reduces the zero-return-day fraction (MASTERPLAN Bug 3 penalty)
enough that the per-trade alpha survives into the portfolio daily
Sharpe. The T and V filters are signal-quality gates: they should
raise trade Sharpe, but only if the remaining trades are DENSE enough
to avoid re-inflicting the dilution penalty. Compare the "Zero-days"
column to the Sharpe column to see whether density matters more than
per-trade quality on this metric.
</div>

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2150_higher_frequency.py · Rule Zero ·
real IronVault + Yahoo only
</p>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2150 — Higher-Frequency EXP-1220 + T+V Filter")
    print("=" * 72)

    # Load the two baseline tapes
    biweekly = load_tape("biweekly")
    weekly = load_tape("weekly")

    # Build signal panels once
    t_panel = build_t_panel()
    v_panel = build_v_panel()

    # Sibling streams + common index
    siblings = load_sibling_streams()
    full_idx = pd.bdate_range(START, END)

    # Six variants
    configs = [
        ("baseline_biweekly",  biweekly, {}),
        ("weekly",             weekly,   {}),
        ("weekly_T",           weekly,   {"T": t_panel}),
        ("weekly_V",           weekly,   {"V": v_panel}),
        ("weekly_T+V",         weekly,   {"T": t_panel, "V": v_panel}),
        ("biweekly_T+V",       biweekly, {"T": t_panel, "V": v_panel}),
    ]

    results: Dict[str, Dict] = {}
    print("\n[run] evaluating variants...")
    print(f"  {'variant':22s} n_trades  /yr   tradeSR  zero%  portCAGR  portSR  portDD")
    for label, tape, overlays in configs:
        if overlays:
            trades = apply_overlays(tape, overlays)
        else:
            trades = tape
        ts = trade_level_sharpe(trades)
        daily = trades_to_daily(trades, full_idx)
        zero_frac = zero_day_fraction(daily)
        port_rets = portfolio(daily, siblings)
        port_oos = port_rets.iloc[WARMUP:]
        port_m = full_metrics(port_oos.values)

        if trades:
            df = pd.DataFrame(trades)
            en = pd.to_datetime(df["entry_date"])
            ex = pd.to_datetime(df["exit_date"])
            yrs = max((ex.max() - en.min()).days / 365.25, 0.5)
            tpy = len(trades) / yrs
        else:
            tpy = 0.0

        results[label] = {
            "n_trades": len(trades),
            "trades_per_year": round(tpy, 2),
            "trade_sharpe": round(ts, 3),
            "zero_day_fraction": round(zero_frac, 4),
            "portfolio": port_m,
            "portfolio_yearly": yearly(port_oos),
            "overlays": list(overlays.keys()),
        }
        print(f"  {label:22s} {len(trades):5d}  {tpy:5.1f}   "
              f"{ts:5.2f}  {zero_frac*100:4.0f}%  "
              f"{port_m['cagr_pct']:+6.1f}%  "
              f"{port_m['sharpe']:5.2f}  "
              f"{port_m['max_dd_pct']:5.1f}%")

    # Winner: highest portfolio Sharpe
    winner = max(results.keys(),
                 key=lambda k: results[k]["portfolio"]["sharpe"])
    base_sr = results["baseline_biweekly"]["portfolio"]["sharpe"]
    lift = results[winner]["portfolio"]["sharpe"] - base_sr
    print(f"\n[winner] {winner} — Sharpe {results[winner]['portfolio']['sharpe']:.2f} "
          f"(baseline {base_sr:.2f}, lift {lift:+.2f})")

    payload = {
        "experiment": "EXP-2150",
        "title": "Higher-Frequency EXP-1220 + T+V Signal Quality Boost",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "exp1220_pipeline": "compass.exp1220_standalone fork with configurable min_spacing_days",
            "ironvault": "data/options_cache.db SPY chains",
            "yahoo_spy_vix": "real Yahoo Finance",
            "t_filter": "compass.exp1750_putcall_overlay.load_vix_term_structure (^VIX > ^VIX3M = blocked)",
            "v_filter": "compass.exp1970_vol_of_vol.build_vvol_panel (z > 1 = blocked)",
            "sibling_streams": "compass/cache/exp1860_streams.pkl + exp2020_vol_arb_trades.pkl",
            "sharpe_formula": "compass.metrics.full_metrics",
        },
        "config": "A_70/5/10/10/5",
        "weights": WEIGHTS_A,
        "leverage": LEV,
        "variants": results,
        "winner": winner,
        "baseline_portfolio_sharpe": round(base_sr, 3),
        "winner_portfolio_sharpe": round(results[winner]["portfolio"]["sharpe"], 3),
        "sharpe_lift": round(lift, 3),
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    REPORT_HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


if __name__ == "__main__":
    main()
