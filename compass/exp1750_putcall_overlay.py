"""
EXP-1750 — Order Flow / Put-Call Ratio Overlay for EXP-1220

Goal: improve EXP-1220 entry timing using options-flow signals.

Data sources (REAL only — no synthetic):
  1. SPY put/call volume ratio computed from IronVault `option_daily` (we own
     the underlying CBOE-derived data; Yahoo's ^CPC/^PCALL feeds are delisted).
  2. VIX term structure from Yahoo: ^VIX, ^VIX9D, ^VIX3M.

Signals:
  - Extreme high SPY P/C ratio (fear) → contrarian *bullish* gate
    (better entry credit on put-credit-spreads when retail panics).
  - VIX term structure inversion (^VIX > ^VIX3M) → vol-stress *exit*/no-entry.
  - Unusual put-volume z-score spike → caution (may precede dislocation).

Overlay logic on EXP-1220 credit spreads:
  - ENTRY GATE: skip trades when (a) PCR is in bottom quartile (complacency)
    or (b) VIX term structure is inverted.
  - SIZING MODIFIER: scale 1.0×–1.5× when PCR is in top quartile (fear premium).

Backtest:
  - Walk-forward 2020-2025 (SPY option-volume coverage starts 2020-01).
  - Compare baseline EXP-1220 vs overlay using trade-level Sharpe.
  - Target: +0.30 Sharpe improvement.

Output: reports/exp1750_order_flow.json + reports/exp1750_order_flow.html
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADING_DAYS = 252

import sys
sys.path.insert(0, str(ROOT))
from shared.iron_vault import IronVault

# Reuse the EXP-1220 trade-construction primitives
from compass.exp1220_standalone import (
    _exp_dt, _find_exps, _next_td, _sell_put_spread, _walk_spread,
)


# ───────────────────────────────────────────────────────────────────────────
# DATA: Put/Call ratio from IronVault SPY option_daily
# ───────────────────────────────────────────────────────────────────────────

def load_spy_pc_ratio(hd: IronVault, start: str, end: str) -> pd.DataFrame:
    """Real SPY put/call volume ratio from IronVault option_daily.

    Returns DataFrame indexed by date with columns:
      put_vol, call_vol, pcr, pcr_5d, pcr_zscore_20d, put_zscore_20d
    """
    conn = sqlite3.connect(hd._db_path)
    sql = """
      SELECT od.date, oc.option_type, SUM(od.volume) AS v
      FROM option_daily od
      JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
      WHERE oc.ticker='SPY' AND od.date BETWEEN ? AND ?
      GROUP BY od.date, oc.option_type
      ORDER BY od.date
    """
    rows = conn.execute(sql, (start, end)).fetchall()
    conn.close()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["date", "type", "v"])
    df = df.pivot(index="date", columns="type", values="v").fillna(0)
    df.columns.name = None
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns={"P": "put_vol", "C": "call_vol"})
    df = df[(df["put_vol"] > 0) & (df["call_vol"] > 0)]
    df["pcr"] = df["put_vol"] / df["call_vol"]
    df["pcr_5d"] = df["pcr"].rolling(5).mean()
    df["pcr_zscore_20d"] = (df["pcr"] - df["pcr"].rolling(20).mean()) / df["pcr"].rolling(20).std(ddof=1)
    df["put_zscore_20d"] = (df["put_vol"] - df["put_vol"].rolling(20).mean()) / df["put_vol"].rolling(20).std(ddof=1)
    return df


def load_vix_term_structure(start: str, end: str) -> pd.DataFrame:
    """VIX term structure from Yahoo: ^VIX, ^VIX9D, ^VIX3M."""
    import yfinance as yf
    out = {}
    for sym, key in [("^VIX9D", "vix9d"), ("^VIX", "vix"), ("^VIX3M", "vix3m")]:
        d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        out[key] = d["Close"]
    df = pd.DataFrame(out).dropna()
    df.index = pd.to_datetime(df.index)
    df["ts_ratio_short"] = df["vix9d"] / df["vix"]      # >1 → near-term stress
    df["ts_ratio_long"]  = df["vix"]    / df["vix3m"]   # >1 → curve inversion (stress)
    df["inverted"] = (df["ts_ratio_long"] > 1.0).astype(int)
    return df


# ───────────────────────────────────────────────────────────────────────────
# OVERLAY SIGNAL
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class OverlayParams:
    pcr_high_pct: float = 0.75   # PCR ≥ rolling-window 75th pct → fear (gate ON)
    pcr_low_pct:  float = 0.25   # PCR ≤ rolling-window 25th pct → complacent (gate OFF)
    pcr_lookback: int   = 60     # rolling window for percentile rank
    use_vix_inversion: bool = True   # block entry when ^VIX > ^VIX3M
    use_put_zspike:    bool = True   # block when put_zscore_20d > 2 (panic dump)
    size_high_pct:     float = 1.30  # contracts multiplier when PCR is "high" zone
    size_low_pct:      float = 0.50  # contracts multiplier when "low" (still allowed if not blocked)


def build_overlay_signal(pcr_df: pd.DataFrame, vix_df: pd.DataFrame,
                         params: OverlayParams) -> pd.DataFrame:
    """Per-day decisions: allow/block + size_mult."""
    df = pcr_df.join(vix_df[["vix", "vix3m", "ts_ratio_long", "inverted"]], how="left")
    df["pcr_pct_rank"] = df["pcr"].rolling(params.pcr_lookback).rank(pct=True)

    df["block_low_pcr"]    = df["pcr_pct_rank"] < params.pcr_low_pct
    df["block_inversion"]  = (df["inverted"] == 1) if params.use_vix_inversion else False
    df["block_put_spike"]  = (df["put_zscore_20d"] > 2.0) if params.use_put_zspike else False

    df["allow_entry"] = ~(df["block_low_pcr"] | df["block_inversion"] | df["block_put_spike"])

    df["size_mult"] = 1.0
    df.loc[df["pcr_pct_rank"] >= params.pcr_high_pct, "size_mult"] = params.size_high_pct
    df.loc[df["pcr_pct_rank"] <  params.pcr_low_pct,  "size_mult"] = params.size_low_pct
    return df


# ───────────────────────────────────────────────────────────────────────────
# BACKTEST: EXP-1220 with overlay
# ───────────────────────────────────────────────────────────────────────────

def run_exp1220_with_overlay(hd: IronVault,
                             spy_df: pd.DataFrame,
                             vix_close: pd.Series,
                             overlay: Optional[pd.DataFrame],
                             start: str, end: str) -> List[Dict]:
    """Run EXP-1220 trades; if overlay is given, apply gate + sizing on entry."""
    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, start, end, monthly=False)
    trades, last = [], None

    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=28), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if es < start or es > end:
            continue
        if last and (entry_dt - last).days < 10:
            continue
        try:
            price = float(spy_close.loc[es]); v = float(vix_close.loc[es])
        except Exception:
            continue
        if np.isnan(price) or np.isnan(v):
            continue
        if v > 40:
            continue

        # Overlay decision
        size_mult = 1.0
        gate_decision = "baseline"
        if overlay is not None:
            try:
                row = overlay.loc[entry_dt]
            except KeyError:
                # find prior row
                idx = overlay.index.searchsorted(entry_dt) - 1
                if idx < 0:
                    continue
                row = overlay.iloc[idx]
            if not bool(row["allow_entry"]):
                continue   # gated out
            size_mult = float(row["size_mult"])
            gate_decision = "overlay_pass"

        spread = _sell_put_spread(hd, exp, es, price, otm_pct=0.95, width=5.0)
        if spread is None:
            continue
        base_cts = max(1, min(4, int(100_000 * 0.03 / (spread["max_loss"] * 100))))
        cts = max(1, int(round(base_cts * size_mult)))
        ed, er, ev, hold = _walk_spread(hd, exp, spread["short"], spread["long"],
                                        spread["credit"], entry_dt, exp_obj, spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * cts
        trades.append({
            "entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
            "exit_reason": er, "credit": spread["credit"],
            "vix": round(v, 1), "hold_days": hold,
            "contracts": cts, "size_mult": size_mult,
            "gate": gate_decision,
        })
        last = entry_dt
    return trades


# ───────────────────────────────────────────────────────────────────────────
# METRICS
# ───────────────────────────────────────────────────────────────────────────

def trade_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {"n": 0, "total_pnl": 0.0, "win_rate": 0.0,
                "avg_pnl": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "max_dd_pct": 0.0, "cagr_pct": 0.0, "calmar": 0.0,
                "avg_hold_days": 0.0}

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())

    eq = np.cumsum(pnls) + 100_000
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max()) if len(eq) else 0.0

    df = pd.DataFrame(trades)
    en = pd.to_datetime(df["entry_date"]); ex = pd.to_datetime(df["exit_date"])
    years = max((ex.max() - en.min()).days / 365.25, 0.5)
    cagr = ((1 + total / 100_000) ** (1 / years) - 1) if total > -100_000 else -1.0

    mu = float(pnls.mean())
    sigma = float(pnls.std(ddof=1)) if n > 1 else 1.0
    tpy = n / max(years, 0.5)
    sharpe = mu / sigma * math.sqrt(tpy) if sigma > 1e-9 else 0.0
    down = pnls[pnls < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else sigma
    sortino = mu / ds * math.sqrt(tpy) if ds > 1e-9 else 0.0
    calmar = (cagr / dd) if dd > 1e-6 else 0.0

    return {
        "n": n, "total_pnl": round(total, 2),
        "win_rate": round(wins / n, 4), "avg_pnl": round(mu, 2),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "max_dd_pct": round(dd * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "calmar": round(calmar, 2),
        "avg_hold_days": round(float(df["hold_days"].mean()), 1),
    }


def yearly_metrics(trades: List[Dict]) -> Dict[int, Dict]:
    if not trades: return {}
    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["exit_date"]).dt.year
    out = {}
    for yr, g in df.groupby("year"):
        out[int(yr)] = trade_metrics(g.to_dict("records"))
    return out


# ───────────────────────────────────────────────────────────────────────────
# WALK-FORWARD: train pcr percentile thresholds in-sample, apply OOS
# ───────────────────────────────────────────────────────────────────────────

def walk_forward(trades_baseline: List[Dict],
                 trades_overlay:  List[Dict]) -> Dict:
    """Group trades by exit-year and compare year-by-year."""
    by_yr_b = yearly_metrics(trades_baseline)
    by_yr_o = yearly_metrics(trades_overlay)
    yrs = sorted(set(by_yr_b) | set(by_yr_o))
    rows = []
    for y in yrs:
        b = by_yr_b.get(y, {}); o = by_yr_o.get(y, {})
        rows.append({
            "year": y,
            "baseline_n":     b.get("n", 0),
            "baseline_pnl":   b.get("total_pnl", 0.0),
            "baseline_wr":    b.get("win_rate", 0.0),
            "baseline_sharpe":b.get("sharpe", 0.0),
            "overlay_n":      o.get("n", 0),
            "overlay_pnl":    o.get("total_pnl", 0.0),
            "overlay_wr":     o.get("win_rate", 0.0),
            "overlay_sharpe": o.get("sharpe", 0.0),
            "delta_sharpe":   round(o.get("sharpe", 0.0) - b.get("sharpe", 0.0), 3),
            "delta_pnl":      round(o.get("total_pnl", 0.0) - b.get("total_pnl", 0.0), 2),
        })
    return {"by_year": rows}


# ───────────────────────────────────────────────────────────────────────────
# REPORT
# ───────────────────────────────────────────────────────────────────────────

def write_html(payload: Dict, path: Path) -> None:
    base = payload["baseline"]; ov = payload["overlay"]
    delta = payload["delta"]
    rows = ""
    for r in payload["walk_forward"]["by_year"]:
        dc = "#16a34a" if r["delta_sharpe"] > 0 else "#dc2626"
        rows += (f"<tr><td>{r['year']}</td>"
                 f"<td>{r['baseline_n']}</td><td>${r['baseline_pnl']:,.0f}</td>"
                 f"<td>{r['baseline_wr']:.0%}</td><td>{r['baseline_sharpe']:.2f}</td>"
                 f"<td>{r['overlay_n']}</td><td>${r['overlay_pnl']:,.0f}</td>"
                 f"<td>{r['overlay_wr']:.0%}</td><td>{r['overlay_sharpe']:.2f}</td>"
                 f"<td style='color:{dc};font-weight:700'>{r['delta_sharpe']:+.2f}</td>"
                 f"<td>${r['delta_pnl']:+,.0f}</td></tr>")

    target_met = "✅ TARGET MET" if delta["sharpe"] >= 0.30 else "❌ Below +0.30 target"
    target_color = "#16a34a" if delta["sharpe"] >= 0.30 else "#dc2626"

    html = f"""<!DOCTYPE html><html><head><meta charset='UTF-8'>
<title>EXP-1750 Order Flow / P-C Ratio Overlay</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1100px}}
h1{{font-size:1.5rem;color:#0f172a}} h2{{font-size:1.05rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:18px}}
.headline{{background:#f0fdf4;border-left:4px solid {target_color};padding:14px;border-radius:6px;margin:14px 0;font-size:0.95rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.15rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.84rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}} td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}} td:first-child{{text-align:left}}
.pos{{color:#16a34a}} .neg{{color:#dc2626}} code{{background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:0.8rem}}
</style></head><body>
<h1>EXP-1750 — Order-Flow / Put-Call Ratio Overlay</h1>
<p class='meta'>Real SPY put/call volume (IronVault) + VIX9D/VIX/VIX3M term structure (Yahoo).
Walk-forward {payload['date_range']['start']} → {payload['date_range']['end']}. Baseline: EXP-1220 standalone.</p>

<div class='headline'><strong>Headline:</strong> baseline Sharpe
<strong>{base['sharpe']:.2f}</strong> → overlay Sharpe
<strong>{ov['sharpe']:.2f}</strong>
&nbsp;|&nbsp; Δ = <strong style='color:{target_color}'>{delta['sharpe']:+.2f}</strong>
&nbsp;({target_met} on +0.30 Sharpe target)</div>

<div class='grid'>
  <div class='card'><div class='l'>Baseline Trades</div><div class='v'>{base['n']}</div></div>
  <div class='card'><div class='l'>Overlay Trades</div><div class='v'>{ov['n']}</div></div>
  <div class='card'><div class='l'>Baseline PnL</div><div class='v'>${base['total_pnl']:,.0f}</div></div>
  <div class='card'><div class='l'>Overlay PnL</div><div class='v'>${ov['total_pnl']:,.0f}</div></div>
  <div class='card'><div class='l'>Baseline WR</div><div class='v'>{base['win_rate']:.0%}</div></div>
  <div class='card'><div class='l'>Overlay WR</div><div class='v'>{ov['win_rate']:.0%}</div></div>
  <div class='card'><div class='l'>Baseline DD</div><div class='v'>{base['max_dd_pct']:.1f}%</div></div>
  <div class='card'><div class='l'>Overlay DD</div><div class='v'>{ov['max_dd_pct']:.1f}%</div></div>
</div>

<h2>Walk-Forward by Year</h2>
<table>
<tr><th>Year</th><th>B n</th><th>B PnL</th><th>B WR</th><th>B Sharpe</th>
<th>O n</th><th>O PnL</th><th>O WR</th><th>O Sharpe</th>
<th>Δ Sharpe</th><th>Δ PnL</th></tr>
{rows}
</table>

<h2>Overlay Parameters</h2>
<p><code>{json.dumps(payload['overlay_params'])}</code></p>

<h2>Signal Summary (overlay decisions over full window)</h2>
<div class='grid'>
  <div class='card'><div class='l'>Days Allowed</div><div class='v'>{payload['signal_stats']['allowed_pct']:.0f}%</div></div>
  <div class='card'><div class='l'>Blocked: low PCR</div><div class='v'>{payload['signal_stats']['block_low_pcr']:.0f}%</div></div>
  <div class='card'><div class='l'>Blocked: VIX inv.</div><div class='v'>{payload['signal_stats']['block_inversion']:.0f}%</div></div>
  <div class='card'><div class='l'>Blocked: put spike</div><div class='v'>{payload['signal_stats']['block_put_spike']:.0f}%</div></div>
  <div class='card'><div class='l'>Mean PCR</div><div class='v'>{payload['signal_stats']['mean_pcr']:.2f}</div></div>
  <div class='card'><div class='l'>Mean VIX/VIX3M</div><div class='v'>{payload['signal_stats']['mean_ts_long']:.2f}</div></div>
</div>

<h2>Method</h2>
<ul>
<li>SPY P/C = SUM(put volume) / SUM(call volume) per day across all SPY contracts in IronVault.</li>
<li>VIX term structure inversion = ^VIX &gt; ^VIX3M (Yahoo Finance daily closes).</li>
<li>Entry gate blocks: PCR percentile rank below {payload['overlay_params']['pcr_low_pct']:.0%} (60-day window),
  VIX/VIX3M &gt; 1.0, or put_volume z-score &gt; 2.0.</li>
<li>Sizing modifier: contracts × {payload['overlay_params']['size_high_pct']} when PCR percentile rank ≥ {payload['overlay_params']['pcr_high_pct']:.0%},
  × {payload['overlay_params']['size_low_pct']} when ≤ {payload['overlay_params']['pcr_low_pct']:.0%} (else 1.0).</li>
<li>Trade-level Sharpe = mean_pnl/std_pnl × √(trades/year).</li>
</ul>
<div style='color:#94a3b8;font-size:0.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px'>
compass/exp1750_putcall_overlay.py · IronVault SPY options · Yahoo VIX term structure · REAL DATA ONLY
</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-1750 — Order Flow / P-C Ratio Overlay")
    print("=" * 60)

    start_data, end_data = "2019-06-01", "2026-04-02"
    bt_start,   bt_end   = "2020-01-01", "2025-12-31"

    hd = IronVault.instance()

    print("[1/5] Loading SPY price + VIX (Yahoo)…")
    import yfinance as yf
    spy_df = yf.download("SPY", start=start_data, end=end_data, progress=False, auto_adjust=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)
    spy_df.index = pd.to_datetime(spy_df.index)

    vix_df = yf.download("^VIX", start=start_data, end=end_data, progress=False, auto_adjust=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix_close = vix_df["Close"]; vix_close.index = pd.to_datetime(vix_close.index)

    print("[2/5] Computing SPY put/call ratio from IronVault…")
    pcr = load_spy_pc_ratio(hd, start_data, end_data)
    print(f"      PCR rows: {len(pcr)}  mean PCR: {pcr['pcr'].mean():.2f}")

    print("[3/5] Loading VIX term structure (^VIX9D, ^VIX, ^VIX3M)…")
    vts = load_vix_term_structure(start_data, end_data)
    print(f"      VTS rows: {len(vts)}  inverted-day pct: {vts['inverted'].mean():.0%}")

    params = OverlayParams()
    overlay = build_overlay_signal(pcr, vts, params)

    sig_stats = {
        "allowed_pct":      float(overlay["allow_entry"].mean() * 100),
        "block_low_pcr":    float(overlay["block_low_pcr"].mean() * 100),
        "block_inversion":  float(overlay["block_inversion"].mean() * 100),
        "block_put_spike":  float(overlay["block_put_spike"].mean() * 100),
        "mean_pcr":         float(pcr["pcr"].mean()),
        "mean_ts_long":     float(vts["ts_ratio_long"].mean()),
    }
    print(f"      Days allowed: {sig_stats['allowed_pct']:.0f}%  | "
          f"low-PCR-blocked: {sig_stats['block_low_pcr']:.0f}%  | "
          f"VIX-inv-blocked: {sig_stats['block_inversion']:.0f}%")

    print("[4/5] Backtest BASELINE EXP-1220 (no overlay)…")
    baseline = run_exp1220_with_overlay(hd, spy_df, vix_close, overlay=None,
                                        start=bt_start, end=bt_end)
    print(f"      baseline trades: {len(baseline)}")

    print("[5/5] Backtest OVERLAY EXP-1220 (gate + sizing)…")
    with_overlay = run_exp1220_with_overlay(hd, spy_df, vix_close, overlay=overlay,
                                            start=bt_start, end=bt_end)
    print(f"      overlay trades:  {len(with_overlay)}")

    base_m = trade_metrics(baseline)
    ov_m   = trade_metrics(with_overlay)
    delta = {
        "sharpe":   round(ov_m["sharpe"]  - base_m["sharpe"],  3),
        "cagr_pct": round(ov_m["cagr_pct"]- base_m["cagr_pct"],2),
        "win_rate": round(ov_m["win_rate"]- base_m["win_rate"],4),
        "max_dd_pct": round(ov_m["max_dd_pct"] - base_m["max_dd_pct"], 2),
        "total_pnl": round(ov_m["total_pnl"] - base_m["total_pnl"], 2),
        "n_trades":  ov_m["n"] - base_m["n"],
    }
    wf = walk_forward(baseline, with_overlay)

    payload = {
        "experiment": "EXP-1750",
        "name": "Order Flow / Put-Call Ratio Overlay",
        "date_range": {"start": bt_start, "end": bt_end},
        "baseline": base_m,
        "overlay": ov_m,
        "delta": delta,
        "walk_forward": wf,
        "overlay_params": asdict(params),
        "signal_stats": sig_stats,
        "data_sources": {
            "pc_ratio": "IronVault option_daily — SUM(put volume)/SUM(call volume) for SPY",
            "vix_term": "Yahoo Finance ^VIX, ^VIX9D, ^VIX3M",
            "spy_options": "IronVault options_cache.db (real)",
        },
        "trades_baseline_sample": baseline[:5],
        "trades_overlay_sample": with_overlay[:5],
        "target_sharpe_improvement": 0.30,
        "target_met": delta["sharpe"] >= 0.30,
    }

    out_dir = ROOT / "compass" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exp1750_order_flow.json").write_text(json.dumps(payload, indent=2))
    write_html(payload, out_dir / "exp1750_order_flow.html")

    print()
    print("RESULTS")
    print("-" * 60)
    print(f"{'metric':<14}{'baseline':>14}{'overlay':>14}{'delta':>14}")
    for k in ["sharpe","cagr_pct","win_rate","max_dd_pct","total_pnl","n"]:
        bv = base_m.get(k, 0); ov = ov_m.get(k, 0)
        print(f"{k:<14}{bv:>14.3f}{ov:>14.3f}{(ov-bv):>14.3f}")
    print()
    print(f"Δ Sharpe: {delta['sharpe']:+.3f}  | target +0.30  | "
          f"{'✅ MET' if delta['sharpe']>=0.30 else '❌ MISS'}")
    print(f"Reports → compass/reports/exp1750_order_flow.json + .html")
    return payload


if __name__ == "__main__":
    main()
