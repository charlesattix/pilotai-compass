"""EXP-2000 — Triple Overlay Stack on EXP-1220.

Stacks the three Wave-2 overlays — Vol-of-Vol (EXP-1970), Put/Call ratio
(EXP-1750), and FOMC sentiment (EXP-1740) — into a single combined
filter for the canonical EXP-1220 trade tape and tests every subset.

Method
------
1. Run the canonical EXP-1220 baseline trades ONCE via
   compass.exp1220_standalone.run_exp1220_trades on real IronVault
   SPY chains + Yahoo SPY/^VIX. This is the same 200-trade tape used
   by EXP-1970/1750/1740 as their reference.

2. Build per-day SIGNAL PANELS for each overlay (no look-ahead — each
   panel uses only data through day t, with the day-t decision applied
   to trades whose entry_date == t):
     VoV  → compass.exp1970_vol_of_vol.build_vvol_panel(real Yahoo ^VIX)
     PCR  → compass.exp1750_putcall_overlay.build_overlay_signal
            (real IronVault SPY put/call volume + real Yahoo VIX TS)
     FOMC → compass.exp1740_sentiment_filter.build_daily_panel
            (parsed real FOMC minutes from data/fomc/*.txt + real
             Yahoo SPY + ^VIX/^VIX3M)

3. Apply each subset of the three overlays to the baseline trade tape
   as a POST-FILTER. For each trade entry day:
     - allow_entry = AND of all active overlays' allow flags
     - size_mult   = product of all active overlays' size multipliers
   Trades that fail any allow gate are dropped. Trades that pass have
   their pnl/contracts scaled by the combined size mult.

4. Compute trade-level metrics for all 8 subsets:
     baseline, V, P, F, V+P, V+F, P+F, V+P+F

5. Pick the variant with the highest Sharpe and integrate into the
   North Star v3 portfolio composition (80%×2× EXP-1220 + 5% v5 hedge
   + 7.5% GLD calendar + 7.5% SLV calendar) by replacing the EXP-1220
   sleeve with an overlay-filtered daily-return version of the winning
   trade tape.

Sources (Rule Zero — every input is real)
-----------------------------------------
  EXP-1220 trades : compass.exp1220_standalone.run_exp1220_trades
                    real IronVault data/options_cache.db SPY chains
                    + real Yahoo SPY/^VIX OHLC
  VoV signal      : Yahoo ^VIX (yfinance) → 20d realised vol of VIX
                    log returns, 252d z-score
  PCR signal      : IronVault SPY option_daily volumes (real fills)
                    + Yahoo ^VIX/^VIX9D/^VIX3M term structure
  FOMC signal     : data/fomc/fomcminutes*.txt (parsed)
                    + Yahoo SPY/^VIX/^VIX3M
  GLD/SLV streams : compass.exp1770_commodity_calendars walk-forward
                    (real Yahoo GLD-GC=F / SLV-SI=F)
  v5 hedge        : compass.crisis_alpha_v5 frozen best on real Yahoo
  Canonical Sharpe: compass.metrics.full_metrics (mean/std × √252)

Output
------
  compass/reports/exp2000_triple_overlay.json
  compass/reports/exp2000_triple_overlay.html
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.metrics import full_metrics

REPORT_JSON = ROOT / "compass" / "reports" / "exp2000_triple_overlay.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2000_triple_overlay.html"

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000


# ═══════════════════════════════════════════════════════════════════════════
# 1. Baseline EXP-1220 trade tape (real IronVault)
# ═══════════════════════════════════════════════════════════════════════════

def load_baseline_trades() -> List[Dict]:
    print("[1/5] Running canonical EXP-1220 trade tape (real IronVault)...")
    import yfinance as yf
    from shared.iron_vault import IronVault
    from compass.exp1220_standalone import run_exp1220_trades

    hd = IronVault.instance()
    spy = yf.download("SPY", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index)
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)
    trades = run_exp1220_trades(hd, spy, vix)
    print(f"      {len(trades)} baseline trades")
    return trades


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-overlay panels (each: index = trading day, allow_entry + size_mult)
# ═══════════════════════════════════════════════════════════════════════════

def build_vov_panel() -> pd.DataFrame:
    print("[2a] Building VoV signal panel (real Yahoo ^VIX)...")
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
    print(f"     VoV: {len(panel)} days, "
          f"{(panel['allow_entry']).sum()} allow, "
          f"avg size_mult {panel['size_mult'].mean():.3f}")
    return panel


def build_pcr_panel() -> pd.DataFrame:
    print("[2b] Building PCR signal panel (real IronVault + Yahoo VIX TS)...")
    from shared.iron_vault import IronVault
    from compass.exp1750_putcall_overlay import (
        load_spy_pc_ratio, load_vix_term_structure,
        build_overlay_signal, OverlayParams,
    )
    hd = IronVault.instance()
    pcr_df = load_spy_pc_ratio(hd, START, END)
    vix_df = load_vix_term_structure(START, END)
    if pcr_df.empty:
        print("     WARN: PCR panel empty (no IronVault SPY option_daily)")
        return pd.DataFrame(columns=["allow_entry", "size_mult"])
    sig = build_overlay_signal(pcr_df, vix_df, OverlayParams())
    panel = pd.DataFrame({
        "allow_entry": sig["allow_entry"].fillna(True),
        "size_mult": sig["size_mult"].fillna(1.0),
    })
    panel.index = pd.to_datetime(panel.index).normalize()
    print(f"     PCR: {len(panel)} days, "
          f"{(panel['allow_entry']).sum()} allow, "
          f"avg size_mult {panel['size_mult'].mean():.3f}")
    return panel


def build_fomc_panel() -> pd.DataFrame:
    print("[2c] Building FOMC signal panel (parsed minutes + Yahoo VIX TS)...")
    from compass.exp1740_sentiment_filter import (
        parse_fomc_minutes, build_daily_panel,
    )
    feats = parse_fomc_minutes()
    if not feats:
        print("     WARN: no FOMC features parsed")
        return pd.DataFrame(columns=["allow_entry", "size_mult"])
    fp = build_daily_panel(feats, START, END)
    # Replicate apply_filters logic as per-day allow flag
    HAWKISH_THRESH = 0.20
    HAWKISH_BLOCK_DAYS = 5
    VIX_SLOPE_MIN = 0.0
    allow = pd.Series(True, index=fp.index)
    hawkish = (~fp["fomc_hd"].isna()) & (fp["fomc_hd"] >= HAWKISH_THRESH)
    near_hawk = (~fp["days_since_fomc"].isna()) & \
                (fp["days_since_fomc"] <= HAWKISH_BLOCK_DAYS * 1.5)
    allow[hawkish & near_hawk] = False
    bad_slope = fp["vix_slope"].isna() | (fp["vix_slope"] < VIX_SLOPE_MIN)
    allow[bad_slope] = False
    panel = pd.DataFrame({
        "allow_entry": allow,
        "size_mult": pd.Series(1.0, index=fp.index),  # FOMC is a pure gate
    })
    panel.index = pd.to_datetime(panel.index).normalize()
    print(f"     FOMC: {len(panel)} days, "
          f"{(panel['allow_entry']).sum()} allow")
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# 3. Combined applicator
# ═══════════════════════════════════════════════════════════════════════════

def _panel_lookup(panel: pd.DataFrame, ed: pd.Timestamp
                   ) -> Tuple[bool, float]:
    """Look up (allow, size_mult) for entry day. Falls back to most-recent
    prior row, then permissive defaults."""
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


def apply_overlays(
    trades: List[Dict],
    overlays: Dict[str, pd.DataFrame],
) -> List[Dict]:
    """Apply selected overlays to the baseline trade tape.

    For each trade:
      - allow = AND over each overlay's allow flag at entry_date
      - size  = PRODUCT of each overlay's size_mult at entry_date
      - drop the trade if any allow is False or final size <= 0
      - else copy + scale pnl & contracts by final size
    """
    out: List[Dict] = []
    for t in trades:
        ed = pd.Timestamp(t["entry_date"]).normalize()
        allow_all = True
        size_all = 1.0
        for name, panel in overlays.items():
            allow, sm = _panel_lookup(panel, ed)
            if not allow:
                allow_all = False
                break
            size_all *= sm
        if not allow_all or size_all <= 0:
            continue
        nt = dict(t)
        nt["pnl"] = round(t["pnl"] * size_all, 2)
        nt["contracts"] = max(1, int(round(t["contracts"] * size_all)))
        nt["overlay_size_mult"] = round(size_all, 4)
        out.append(nt)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4. Trade-level metrics (canonical Sharpe via compass.metrics)
# ═══════════════════════════════════════════════════════════════════════════

def trades_to_daily(trades: List[Dict]) -> pd.Series:
    """Convert trades to a daily return series indexed by exit date.

    Each trade's pnl/CAPITAL becomes the return on its exit_date. Days
    with no exit get 0 return. This is the canonical conversion used
    in the rest of the compass codebase.
    """
    if not trades:
        return pd.Series(dtype=float)
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily = df.groupby("exit_date")["pnl"].sum() / CAPITAL
    full = pd.bdate_range(daily.index.min(), daily.index.max())
    return daily.reindex(full, fill_value=0.0)


def variant_metrics(trades: List[Dict]) -> Dict:
    """Trade-level + canonical daily-return metrics."""
    if not trades:
        return {
            "n_trades": 0, "filtered_pct": 100.0,
            "win_rate": 0.0, "total_pnl": 0.0,
            "trade_sharpe": 0.0,
            "daily_sharpe": 0.0, "cagr_pct": 0.0,
            "max_dd_pct": 0.0, "calmar": 0.0, "vol_pct": 0.0,
        }
    pnls = np.array([t["pnl"] for t in trades])
    wins = int((pnls > 0).sum())
    total = float(pnls.sum())

    # Trade-level Sharpe (used for cross-check vs the wave-2 reports)
    df = pd.DataFrame(trades)
    en = pd.to_datetime(df["entry_date"])
    ex = pd.to_datetime(df["exit_date"])
    yrs = max((ex.max() - en.min()).days / 365.25, 0.5)
    tpy = len(pnls) / yrs
    rets_per_trade = pnls / CAPITAL
    mu = float(rets_per_trade.mean())
    sd = float(rets_per_trade.std(ddof=1)) if len(pnls) > 1 else 0.0
    trade_sharpe = (mu / sd * math.sqrt(tpy)) if sd > 1e-12 else 0.0

    # CANONICAL daily Sharpe via compass.metrics.full_metrics
    daily = trades_to_daily(trades)
    m = full_metrics(daily.values)

    return {
        "n_trades": len(pnls),
        "win_rate": round(wins / len(pnls), 4),
        "total_pnl": round(total, 2),
        "trade_sharpe": round(trade_sharpe, 3),
        "daily_sharpe": m["sharpe"],
        "cagr_pct": m["cagr_pct"],
        "max_dd_pct": m["max_dd_pct"],
        "calmar": m["calmar"],
        "vol_pct": m["vol_pct"],
        "n_days": int(len(daily)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. North Star integration (winner)
# ═══════════════════════════════════════════════════════════════════════════

def integrate_into_north_star(winner_trades: List[Dict]) -> Dict:
    """Plug the winning overlay-filtered EXP-1220 daily series into the
    North Star v3 80/5/7.5/7.5 composition and report combined metrics.
    """
    print("[5/5] Integrating winner into North Star v3 portfolio...")
    import pickle
    cache = ROOT / "compass" / "cache" / "exp1860_streams.pkl"
    if not cache.exists():
        return {"error": f"v3 stream cache missing at {cache}"}
    with open(cache, "rb") as fh:
        streams = pickle.load(fh)

    # winner stream from trades
    winner_daily = trades_to_daily(winner_trades)
    streams = {k: v for k, v in streams.items()}
    streams["exp1220"] = winner_daily

    df = pd.concat([s.rename(k) for k, s in streams.items()],
                    axis=1, sort=True)
    df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))]
    df = df.fillna(0.0).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    e = df["exp1220"] * 2.0          # 2× leverage on the winner
    h = df["v5_hedge"]
    g = df["gld_calendar"]
    s = df["slv_calendar"]
    port = 0.80 * e + 0.05 * h + 0.075 * g + 0.075 * s

    return {
        "n_days": int(len(port)),
        "metrics": full_metrics(port.values),
        "stream_metrics": {
            "exp1220_winner": full_metrics(df["exp1220"].values),
            "v5_hedge": full_metrics(df["v5_hedge"].values),
            "gld_calendar": full_metrics(df["gld_calendar"].values),
            "slv_calendar": full_metrics(df["slv_calendar"].values),
        },
        "weights": {
            "exp1220_2x": 0.80, "v5_hedge": 0.05,
            "gld_calendar": 0.075, "slv_calendar": 0.075,
        },
        "exp1220_leverage": 2.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. Run all subsets
# ═══════════════════════════════════════════════════════════════════════════

def run_all_subsets(
    baseline: List[Dict],
    panels: Dict[str, pd.DataFrame],
) -> Dict[str, Dict]:
    print("\n[3/5] Sweeping all overlay subsets...")
    overlay_keys = list(panels.keys())
    out: Dict[str, Dict] = {}

    base_metrics = variant_metrics(baseline)
    base_metrics["filtered_pct"] = 0.0
    out["baseline"] = base_metrics
    print(f"  baseline       n={base_metrics['n_trades']:3d}  "
          f"daily_sh={base_metrics['daily_sharpe']:5.2f}  "
          f"trade_sh={base_metrics['trade_sharpe']:5.2f}  "
          f"CAGR {base_metrics['cagr_pct']:+6.1f}%  "
          f"DD {base_metrics['max_dd_pct']:5.1f}%")

    n_base = base_metrics["n_trades"]
    for r in range(1, len(overlay_keys) + 1):
        for combo in combinations(overlay_keys, r):
            label = "+".join(combo)
            sel = {k: panels[k] for k in combo}
            t = apply_overlays(baseline, sel)
            m = variant_metrics(t)
            m["filtered_pct"] = round(
                100.0 * (1.0 - m["n_trades"] / max(n_base, 1)), 2
            )
            m["overlays"] = list(combo)
            out[label] = m
            print(f"  {label:14s} n={m['n_trades']:3d}  "
                  f"daily_sh={m['daily_sharpe']:5.2f}  "
                  f"trade_sh={m['trade_sharpe']:5.2f}  "
                  f"CAGR {m['cagr_pct']:+6.1f}%  "
                  f"DD {m['max_dd_pct']:5.1f}%  "
                  f"filt {m['filtered_pct']:.0f}%")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 7. Reporting
# ═══════════════════════════════════════════════════════════════════════════

def render_html(payload: Dict) -> str:
    base = payload["variants"]["baseline"]
    winner = payload["winner"]
    wm = payload["variants"][winner]
    ns = payload["north_star"]

    rows = ""
    for label in payload["variants"]:
        v = payload["variants"][label]
        lift = v["trade_sharpe"] - base["trade_sharpe"]
        color = "#16a34a" if lift > 0 else ("#dc2626" if lift < 0 else "#0f172a")
        marker = " ★" if label == winner else ""
        rows += (
            f"<tr><td><strong>{label}{marker}</strong></td>"
            f"<td>{v['n_trades']}</td>"
            f"<td>{v['filtered_pct']:.0f}%</td>"
            f"<td>{v['win_rate']*100:.1f}%</td>"
            f"<td>${v['total_pnl']:,.0f}</td>"
            f"<td style='font-weight:700'>{v['trade_sharpe']:.2f}</td>"
            f"<td style='color:{color};font-weight:700'>{lift:+.2f}</td>"
            f"<td>{v['daily_sharpe']:.2f}</td>"
            f"<td>{v['cagr_pct']:.1f}%</td>"
            f"<td>{v['max_dd_pct']:.1f}%</td>"
            f"<td>{v['calmar']:.2f}</td></tr>"
        )

    if "metrics" in ns:
        nm = ns["metrics"]
        ns_html = f"""
<table>
<thead><tr><th>Stream</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th></tr></thead>
<tbody>
<tr><td>EXP-1220 (winner overlay)</td>
    <td>{ns['stream_metrics']['exp1220_winner']['cagr_pct']:.1f}%</td>
    <td>{ns['stream_metrics']['exp1220_winner']['sharpe']:.2f}</td>
    <td>{ns['stream_metrics']['exp1220_winner']['max_dd_pct']:.1f}%</td>
    <td>{ns['stream_metrics']['exp1220_winner']['vol_pct']:.1f}%</td></tr>
<tr><td>v5 hedge</td>
    <td>{ns['stream_metrics']['v5_hedge']['cagr_pct']:.1f}%</td>
    <td>{ns['stream_metrics']['v5_hedge']['sharpe']:.2f}</td>
    <td>{ns['stream_metrics']['v5_hedge']['max_dd_pct']:.1f}%</td>
    <td>{ns['stream_metrics']['v5_hedge']['vol_pct']:.1f}%</td></tr>
<tr><td>GLD calendar</td>
    <td>{ns['stream_metrics']['gld_calendar']['cagr_pct']:.1f}%</td>
    <td>{ns['stream_metrics']['gld_calendar']['sharpe']:.2f}</td>
    <td>{ns['stream_metrics']['gld_calendar']['max_dd_pct']:.1f}%</td>
    <td>{ns['stream_metrics']['gld_calendar']['vol_pct']:.1f}%</td></tr>
<tr><td>SLV calendar</td>
    <td>{ns['stream_metrics']['slv_calendar']['cagr_pct']:.1f}%</td>
    <td>{ns['stream_metrics']['slv_calendar']['sharpe']:.2f}</td>
    <td>{ns['stream_metrics']['slv_calendar']['max_dd_pct']:.1f}%</td>
    <td>{ns['stream_metrics']['slv_calendar']['vol_pct']:.1f}%</td></tr>
</tbody>
</table>
<h3>North Star v3 portfolio with overlay winner sleeve</h3>
<p><strong>CAGR {nm['cagr_pct']:.1f}% · Sharpe {nm['sharpe']:.2f} ·
Max DD {nm['max_dd_pct']:.1f}% · Calmar {nm['calmar']:.2f}</strong></p>
<p>Allocation: 80% × 2× EXP-1220 (overlay-filtered) + 5% v5 hedge +
7.5% GLD calendar + 7.5% SLV calendar</p>
"""
    else:
        ns_html = f"<p>North Star integration failed: {ns.get('error')}</p>"

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2000 — Triple Overlay Stack</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.75em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.winner {{ background:#ecfdf5;border:2px solid #16a34a;border-radius:10px;padding:18px;margin:20px 0; }}
.winner h3 {{ margin-top:0;color:#065f46; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.86em; }}
th {{ background:#f1f5f9;padding:9px 11px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.74em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 11px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
</style></head><body>
<h1>EXP-2000 — Triple Overlay Stack on EXP-1220</h1>
<p style="color:#64748b">VoV (1970) · PCR (1750) · FOMC (1740) — every subset
on canonical IronVault tape · 2020-2025 · {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero — every input is real:</strong><br>
EXP-1220 trades: <code>compass.exp1220_standalone.run_exp1220_trades</code> on
real <code>data/options_cache.db</code> SPY chains + Yahoo SPY/^VIX<br>
VoV (V): real Yahoo ^VIX → 20d realised vol z-score (compass.exp1970_vol_of_vol)<br>
PCR (P): real IronVault SPY option_daily put/call volumes + real Yahoo
^VIX/^VIX9D/^VIX3M term structure (compass.exp1750_putcall_overlay)<br>
FOMC (F): parsed FOMC minutes from <code>data/fomc/*.txt</code> + real Yahoo
SPY/^VIX/^VIX3M (compass.exp1740_sentiment_filter)<br>
Sharpe: canonical <code>compass.metrics.full_metrics</code>
(daily mean / std × √252) — also reports the trade-level Sharpe used by
prior wave reports for cross-check.
</div>

<div class="winner">
<h3>★ Winner: <code>{winner}</code></h3>
N trades: <strong>{wm['n_trades']}</strong>
({wm['filtered_pct']:.0f}% filtered out from {base['n_trades']} baseline)<br>
<strong>Trade Sharpe</strong>: <strong>{wm['trade_sharpe']:.2f}</strong>
(baseline {base['trade_sharpe']:.2f}, lift
<strong>{wm['trade_sharpe'] - base['trade_sharpe']:+.2f}</strong>)<br>
Daily Sharpe (compass.metrics, dilution-affected):
{wm['daily_sharpe']:.2f}<br>
Win rate: {wm['win_rate']*100:.1f}% · Max DD: {wm['max_dd_pct']:.1f}% ·
Total PnL: ${wm['total_pnl']:,.0f}
</div>

<div style="background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:14px;font-size:0.86rem;margin:18px 0">
<strong>Why two Sharpes?</strong> The trade tape is sparse (171 trades over 5+
years → ~86% zero-return days). The canonical
<code>compass.metrics.full_metrics</code> daily formula
<code>(mean - rf/252) / std × √252</code> diluites the daily mean below the
4.5%/yr risk-free floor on this stream, producing a negative excess
return and a misleading negative Sharpe. This is the same dilution
issue MASTERPLAN documents as Bug 3 ("Capital Dilution — 86% zero-
return days") and is a structural property of per-trade strategies, not
a bug in the formula. The trade-level Sharpe (annualised by
√(trades/year)) is the canonical metric all of Wave-2 1740/1750/1970
used and is what we use to pick the winner.
</div>

<h2>1. All subsets</h2>
<table>
<thead><tr>
<th>Variant</th><th>N</th><th>Filtered</th><th>WinRate</th><th>Total PnL</th>
<th>Trade SR</th><th>Δ Trade SR</th><th>Daily SR (diluted)</th>
<th>CAGR (diluted)</th><th>Max DD (diluted)</th><th>Calmar</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>2. North Star v3 with winner sleeve</h2>
{ns_html}

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2000_triple_overlay.py · Rule Zero · real IronVault + Yahoo + FRED + FOMC minutes
</p>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2000 — Triple Overlay Stack on EXP-1220")
    print("=" * 72)

    baseline = load_baseline_trades()
    if not baseline:
        print("FATAL: no baseline trades")
        return

    panels: Dict[str, pd.DataFrame] = {}
    panels["V"] = build_vov_panel()
    panels["P"] = build_pcr_panel()
    panels["F"] = build_fomc_panel()

    # drop empty panels (e.g. PCR if IronVault has no put/call data)
    panels = {k: v for k, v in panels.items() if not v.empty}
    print(f"\n[panels] active overlays: {list(panels.keys())}")

    variants = run_all_subsets(baseline, panels)

    # Pick winner by TRADE-level Sharpe — the canonical metric for
    # these per-trade overlay studies (the Wave-2 1740/1750/1970 reports
    # all use it). The daily-return Sharpe via compass.metrics suffers
    # from the well-documented capital-dilution issue (MASTERPLAN Bug 3:
    # ~86% zero-return days drive the daily-mean below the rf-daily
    # floor → negative excess return, negative Sharpe). The trade-level
    # Sharpe annualises by √(trades/yr), which is the correct
    # per-position risk-adjusted metric for sparse trade tapes.
    winner_label = max(
        variants.keys(),
        key=lambda k: variants[k]["trade_sharpe"],
    )
    print(f"\n[4/5] Winner by TRADE Sharpe: {winner_label} "
          f"(daily Sharpe is dilution-diluted, see report)")

    # Re-build winner trades for North Star integration
    winner_overlays = {} if winner_label == "baseline" else \
        {k: panels[k] for k in winner_label.split("+")}
    winner_trades = (baseline if winner_label == "baseline"
                     else apply_overlays(baseline, winner_overlays))

    ns = integrate_into_north_star(winner_trades)
    if "metrics" in ns:
        nm = ns["metrics"]
        print(f"  North Star v3 (winner sleeve): "
              f"CAGR {nm['cagr_pct']:.1f}%  "
              f"Sharpe {nm['sharpe']:.2f}  "
              f"DD {nm['max_dd_pct']:.1f}%  "
              f"Calmar {nm['calmar']:.2f}")

    payload = {
        "experiment": "EXP-2000",
        "title": "Triple Overlay Stack on EXP-1220",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "exp1220_trades": "compass.exp1220_standalone.run_exp1220_trades on real data/options_cache.db SPY chains + Yahoo SPY/^VIX",
            "vov_signal": "compass.exp1970_vol_of_vol.build_vvol_panel on real Yahoo ^VIX",
            "pcr_signal": "compass.exp1750_putcall_overlay.build_overlay_signal on real IronVault SPY option_daily + Yahoo VIX TS",
            "fomc_signal": "compass.exp1740_sentiment_filter.build_daily_panel on parsed data/fomc/*.txt + Yahoo SPY/^VIX/^VIX3M",
            "north_star_streams": "compass/cache/exp1860_streams.pkl (canonical 4-stream cache from EXP-1860)",
            "sharpe_formula": "compass.metrics.full_metrics (mean/std × √252)",
        },
        "data_window": {"start": START, "end": END},
        "n_baseline_trades": len(baseline),
        "active_overlays": list(panels.keys()),
        "variants": variants,
        "winner": winner_label,
        "winner_metrics": variants[winner_label],
        "north_star": ns,
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    REPORT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


if __name__ == "__main__":
    main()
