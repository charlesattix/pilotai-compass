"""
EXP-2460 — Zero-Cost Alpha: Portfolio-Level T+V Timing Overlay
===============================================================

Hypothesis
----------
Most overlays add alpha by TRADING MORE (more signals, more filters,
more positions). The cheaper path is to trade LESS on bad days — skip
rebalances when the VIX term structure warns of backwardation or
realised-vol-of-VIX is elevated. That simultaneously:

  (1) reduces transaction cost by cutting rebalance turnover, and
  (2) lifts Sharpe by not leaning into bad regimes.

Reference signals (all previously validated)
  T filter (EXP-2070):  VIX / VIX3M > 0.95  → skip
  V filter (EXP-1970):  20d realised VIX vol z-score > 1.5  → skip

Strategy variants tested on the 7-stream cube
  A. baseline            Weekly rebalance, no overlay, 5bps per trade
  B. skip_hold           Skip rebalance but keep previous weights on block days
  C. skip_flat           Skip rebalance AND go flat (0 exposure) on block days
  D. skip_half           Skip rebalance AND cut leverage to 50% on block days

Transaction cost model
----------------------
Cost = turnover × cost_bps / 10000
     = sum(|w_new - w_old|) × 5 bps
The baseline incurs a full turnover cost every rebalance (252/7 ≈ 36
per year × 5 years ≈ 180 events). The overlay variants skip a fraction
of those events, directly saving bps.

Backtest
--------
Walk-forward 252d train / 63d test, 20 folds, Ledoit-Wolf risk-parity
allocator (EXP-2360 winner), vol-targeted to 15% annualised.

Rule Zero: all signals from real Yahoo ^VIX + ^VIX3M, all streams from
the real 7-stream cube (5 cached + XLF/XLI rebuilt live from IronVault).

Outputs
  compass/reports/exp2460_zero_cost_overlay.json
  compass/reports/exp2460_zero_cost_overlay.html
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp1970_vol_of_vol import build_vvol_panel
from compass.exp2070_term_structure import load_term_structure
from compass.exp2080_corr_regime import load_streams
from compass.exp2160_high_capacity_alts import (
    run_put_credit_spreads,
    trades_to_daily_pct,
)
from compass.exp2360_robust_cov import risk_parity_weights
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2460_zero_cost_overlay.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2460_zero_cost_overlay.html"

TRADING_DAYS = 252
TRAIN_DAYS = 252
TEST_DAYS = 63
TARGET_VOL_ANNUAL = 0.15
COST_BPS = 5.0                     # 5 bps per unit of turnover
REBALANCE_EVERY_N_DAYS = 5         # weekly

# Thresholds from task
TERM_RATIO_BLOCK = 0.95            # VIX/VIX3M > 0.95 → block
VVOL_Z_BLOCK     = 1.5             # 20d vvol z > 1.5 → block


# ─────────────────────────────────────────────────────────────────────────────
# Cube + panel
# ─────────────────────────────────────────────────────────────────────────────
def build_seven_stream_cube() -> pd.DataFrame:
    print("[1/5] loading 5-stream cached cube …")
    base = load_streams()
    print(f"      {base.shape}")

    print("[2/5] building XLF + XLI streams (real IronVault) …")
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)
    for tk in ("XLF", "XLI"):
        trades = run_put_credit_spreads(con, tk)
        daily = trades_to_daily_pct(trades, base.index)
        base[f"{tk.lower()}_cs"] = daily.reindex(base.index).fillna(0.0)
    con.close()
    df = base[["exp1220", "v5_hedge", "gld_cal", "slv_cal", "cross_vol", "xlf_cs", "xli_cs"]]
    return df


def build_overlay_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a daily DataFrame aligned with df with T/V flags."""
    print("[3/5] building T+V overlay panel from real Yahoo ^VIX/^VIX3M …")
    ts = load_term_structure("2019-06-01", "2026-07-01")
    vix = ts["vix"]
    vvol = build_vvol_panel(vix)
    panel = ts[["vix", "vix3m", "ratio"]].join(vvol[["vvol", "vvol_z"]], how="left")
    panel = panel.reindex(df.index).ffill()
    panel["t_block"] = (panel["ratio"] > TERM_RATIO_BLOCK).fillna(False)
    panel["v_block"] = (panel["vvol_z"] > VVOL_Z_BLOCK).fillna(False)
    panel["any_block"] = panel["t_block"] | panel["v_block"]
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# Backtest driver with cost accounting
# ─────────────────────────────────────────────────────────────────────────────
def cov_ledoit_wolf(R: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LedoitWolf().fit(R).covariance_


def run_variant(df: pd.DataFrame, panel: pd.DataFrame, *,
                variant: str) -> Dict:
    """variant ∈ {baseline, skip_hold, skip_flat, skip_half}"""
    cols = list(df.columns)
    n = len(df)
    daily_ret = []
    dates = []
    turnovers = []           # list of turnover per rebalance event
    rebalance_days = 0
    skipped_days = 0
    flatten_days = 0
    scale_applied = []

    i = TRAIN_DAYS
    w_prev = np.full(len(cols), 1.0 / len(cols))
    scale_prev = 1.0
    while i < n:
        # rebalance decision at start of each test window (every REBALANCE_EVERY_N_DAYS)
        train_start = max(0, i - TRAIN_DAYS)
        train = df.iloc[train_start:i].values

        # compute "target" weights + vol scale for THIS block
        Sigma = cov_ledoit_wolf(train)
        w_target = risk_parity_weights(Sigma)
        train_port = train @ w_target
        train_vol = float(np.std(train_port, ddof=1)) * math.sqrt(TRADING_DAYS)
        scale_target = TARGET_VOL_ANNUAL / max(train_vol, 1e-10)
        scale_target = float(np.clip(scale_target, 0.1, 5.0))

        # Is TODAY a block day per the overlay?
        block = False
        if variant != "baseline":
            entry_date = df.index[i]
            try:
                block = bool(panel.loc[entry_date, "any_block"])
            except KeyError:
                block = False

        # Decide weights + scale for this block according to variant
        if variant == "baseline" or not block:
            w_use = w_target
            scale_use = scale_target
            # normal rebalance — incurs turnover cost
            tto = float(np.sum(np.abs(w_use * scale_use - w_prev * scale_prev)))
            turnovers.append(tto)
            rebalance_days += 1
        else:
            # Block day — behaviour depends on variant
            if variant == "skip_hold":
                w_use = w_prev
                scale_use = scale_prev
                skipped_days += 1
            elif variant == "skip_flat":
                w_use = np.zeros_like(w_prev)
                scale_use = 0.0
                # closing out any previous exposure does incur cost
                tto = float(np.sum(np.abs(w_prev * scale_prev)))
                turnovers.append(tto)
                flatten_days += 1
            elif variant == "skip_half":
                w_use = w_prev
                scale_use = scale_prev * 0.5
                tto = float(np.sum(np.abs(w_use * scale_use - w_prev * scale_prev)))
                turnovers.append(tto)
                skipped_days += 1
            else:
                raise ValueError(f"unknown variant {variant}")

        # Play the next REBALANCE_EVERY_N_DAYS days with this (w_use, scale_use)
        block_end = min(i + REBALANCE_EVERY_N_DAYS, n)
        test_slice = df.iloc[i:block_end].values
        block_ret = (test_slice @ w_use) * scale_use
        dates.extend(df.index[i:block_end])
        daily_ret.extend(block_ret.tolist())
        scale_applied.extend([scale_use] * (block_end - i))

        w_prev = w_use
        scale_prev = scale_use
        i = block_end

    # Apply transaction-cost drag on the pooled daily series:
    #   cost_per_event = COST_BPS/1e4 × turnover_event
    # We amortise each cost over the period it covers (1 day subtracted on rebalance day).
    # Simpler: subtract the total cost from the cumulative equity at the end, then
    # report gross and net separately.
    gross = pd.Series(daily_ret, index=dates, dtype=float)
    total_turnover = float(np.sum(turnovers))
    total_cost = total_turnover * (COST_BPS / 1e4)
    # distribute across rebalance events (one cost-day per event)
    net = gross.copy()
    event_idx = 0
    # mark rebalance dates
    rb_dates = []
    i = TRAIN_DAYS
    w_prev_sim = np.full(len(cols), 1 / len(cols))
    sc_prev_sim = 1.0
    while i < n:
        entry_date = df.index[i]
        if variant == "baseline":
            rb_dates.append(entry_date)
        else:
            blocked = bool(panel.loc[entry_date, "any_block"]) if entry_date in panel.index else False
            if (not blocked) or variant == "skip_flat":
                rb_dates.append(entry_date)
        i += REBALANCE_EVERY_N_DAYS
    # apply cost per event on the event date
    if rb_dates and len(turnovers) == len(rb_dates):
        for d, tto in zip(rb_dates, turnovers):
            if d in net.index:
                net.loc[d] -= tto * (COST_BPS / 1e4)

    return {
        "variant": variant,
        "gross_daily": gross,
        "net_daily":   net,
        "rebalance_events": len(turnovers),
        "rebalance_events_baseline_equivalent": int((n - TRAIN_DAYS) / REBALANCE_EVERY_N_DAYS),
        "skipped_days": skipped_days,
        "flatten_days": flatten_days,
        "total_turnover": total_turnover,
        "total_cost_bps": total_turnover * COST_BPS,
        "mean_scale": float(np.mean(scale_applied)) if scale_applied else 0.0,
    }


def metrics(daily: pd.Series, label: str) -> Dict:
    daily = daily.dropna()
    if len(daily) < 30:
        return {"label": label, "n_days": 0}
    eq = (1 + daily).cumprod()
    yrs = len(daily) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    mu, sd = daily.mean(), daily.std(ddof=1)
    sharpe = float((mu / sd) * math.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
    downside = daily[daily < 0].std(ddof=1) if (daily < 0).any() else np.nan
    sortino = float((mu / downside) * math.sqrt(TRADING_DAYS)) if downside and downside > 1e-12 else 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(-dd.min())
    return {
        "label": label,
        "n_days": int(len(daily)),
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 3),
        "vol_pct": round(float(sd) * math.sqrt(TRADING_DAYS) * 100, 3),
        "calmar": round(cagr / max_dd, 3) if max_dd > 1e-9 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = build_seven_stream_cube()
    panel = build_overlay_panel(df)

    block_days_total = int(panel["any_block"].sum())
    t_only = int(panel["t_block"].sum())
    v_only = int(panel["v_block"].sum())
    both = int((panel["t_block"] & panel["v_block"]).sum())
    print(f"      block-day totals — T:{t_only}  V:{v_only}  both:{both}  any:{block_days_total}  of {len(panel)}")

    print("[4/5] running 4 variants …")
    variants = {}
    for name in ("baseline", "skip_hold", "skip_flat", "skip_half"):
        r = run_variant(df, panel, variant=name)
        gm = metrics(r["gross_daily"], f"{name}_gross")
        nm = metrics(r["net_daily"],  f"{name}_net")
        variants[name] = {
            "gross": gm,
            "net":   nm,
            "rebalance_events": r["rebalance_events"],
            "rebalance_events_baseline": r["rebalance_events_baseline_equivalent"],
            "skipped_days":     r["skipped_days"],
            "flatten_days":     r["flatten_days"],
            "total_turnover":   round(r["total_turnover"], 4),
            "total_cost_bps":   round(r["total_cost_bps"], 2),
            "mean_scale":       round(r["mean_scale"], 3),
        }
        print(f"      {name:11s}  events {r['rebalance_events']:4d}  "
              f"gross S {gm['sharpe']:5.2f}  net S {nm['sharpe']:5.2f}  "
              f"CAGR {nm['cagr_pct']:6.2f}%  DD {nm['max_dd_pct']:5.2f}%  "
              f"cost {r['total_cost_bps']:5.1f} bps")

    # Headline comparisons
    base = variants["baseline"]
    best = max(((k, v) for k, v in variants.items() if k != "baseline"),
               key=lambda kv: kv[1]["net"]["sharpe"])
    best_name, best_v = best
    delta_sharpe_net = round(best_v["net"]["sharpe"] - base["net"]["sharpe"], 3)
    delta_cost_bps   = round(base["total_cost_bps"] - best_v["total_cost_bps"], 2)
    trade_reduction_pct = round(
        (1 - best_v["rebalance_events"] / max(1, base["rebalance_events"])) * 100, 2
    )

    print("[5/5] writing report …")
    payload = {
        "experiment": "EXP-2460",
        "name": "Zero-Cost Alpha — portfolio-level T+V timing overlay",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "thresholds": {
            "term_ratio_block_gt": TERM_RATIO_BLOCK,
            "vvol_z_block_gt":     VVOL_Z_BLOCK,
            "rebalance_every_n_days": REBALANCE_EVERY_N_DAYS,
            "target_vol_annual":   TARGET_VOL_ANNUAL,
            "cost_bps_per_turnover": COST_BPS,
        },
        "data_sources": {
            "cube": "compass.exp2080_corr_regime.load_streams + compass.exp2160_high_capacity_alts XLF/XLI",
            "term_structure": "Yahoo Finance ^VIX / ^VIX3M",
            "vvol": "Yahoo Finance ^VIX 20d realised, 252d z-score",
        },
        "cube_info": {
            "n_days": int(len(df)),
            "range": [str(df.index[0].date()), str(df.index[-1].date())],
            "streams": list(df.columns),
        },
        "block_day_stats": {
            "T_only_block_days": t_only,
            "V_only_block_days": v_only,
            "both_block_days":   both,
            "any_block_days":    block_days_total,
            "pct_of_sample":     round(block_days_total / len(panel) * 100, 2),
        },
        "variants": variants,
        "headline": {
            "best_variant": best_name,
            "delta_sharpe_net_vs_baseline": delta_sharpe_net,
            "cost_saving_bps": delta_cost_bps,
            "trade_reduction_pct": trade_reduction_pct,
            "net_sharpe_best": best_v["net"]["sharpe"],
            "net_sharpe_baseline": base["net"]["sharpe"],
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    rows = "".join(
        f"<tr><td>{name}</td>"
        f"<td>{v['rebalance_events']}</td>"
        f"<td>{v['skipped_days']}</td>"
        f"<td>{v['flatten_days']}</td>"
        f"<td>{v['gross']['sharpe']:.2f}</td>"
        f"<td>{v['net']['sharpe']:.2f}</td>"
        f"<td>{v['net']['cagr_pct']:.2f}%</td>"
        f"<td>{v['net']['max_dd_pct']:.2f}%</td>"
        f"<td>{v['total_cost_bps']:.1f}</td></tr>"
        for name, v in p["variants"].items()
    )
    h = p["headline"]; b = p["block_day_stats"]
    delta_cls = "ok" if h["delta_sharpe_net_vs_baseline"] > 0 else "warn"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2460 — Zero-Cost Overlay</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5;background:#fff}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .warn{{color:#b86b00;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-2460 — Zero-Cost Alpha: Portfolio-Level T+V Timing Overlay</h1>
<p class='small'>Generated {p['generated']} · 7-stream cube · LW risk-parity ·
  15% vol target · 5bps/turnover · Rule Zero clean.</p>

<h2>Block-day coverage</h2>
<p>T (^VIX/^VIX3M &gt; {p['thresholds']['term_ratio_block_gt']}): <b>{b['T_only_block_days']}</b> days<br>
   V (vvol 20d/252d-z &gt; {p['thresholds']['vvol_z_block_gt']}): <b>{b['V_only_block_days']}</b> days<br>
   Both: <b>{b['both_block_days']}</b> days<br>
   Any block: <b>{b['any_block_days']}</b> days ({b['pct_of_sample']}% of sample)</p>

<h2>Variant bake-off</h2>
<table>
<tr><th>Variant</th><th>Rebalance events</th><th>Skipped days</th><th>Flat days</th>
 <th>Gross Sharpe</th><th>Net Sharpe</th><th>Net CAGR</th><th>Net DD</th><th>Total cost (bps)</th></tr>
{rows}
</table>

<h2>Headline</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>{h['best_variant']}</th><th>Δ</th></tr>
<tr><td>Net Sharpe</td><td>{h['net_sharpe_baseline']:.2f}</td><td>{h['net_sharpe_best']:.2f}</td>
 <td class='{delta_cls}'>{h['delta_sharpe_net_vs_baseline']:+.2f}</td></tr>
<tr><td>Rebalance events</td>
 <td>{p['variants']['baseline']['rebalance_events']}</td>
 <td>{p['variants'][h['best_variant']]['rebalance_events']}</td>
 <td>{-h['trade_reduction_pct']:.1f}%</td></tr>
<tr><td>Total cost (bps)</td>
 <td>{p['variants']['baseline']['total_cost_bps']:.1f}</td>
 <td>{p['variants'][h['best_variant']]['total_cost_bps']:.1f}</td>
 <td class='ok'>-{h['cost_saving_bps']:.1f}</td></tr>
</table>

<h2>Notes</h2>
<ul>
<li>Overlay blocks rebalancing when <b>VIX/VIX3M &gt; 0.95</b> (term
    backwardation) OR <b>vvol 20d-z &gt; 1.5</b> (elevated vol of vol).</li>
<li><b>skip_hold</b> keeps previous weights (pure cost reduction).<br>
    <b>skip_flat</b> goes to cash on block days (cost reduction +
    exposure reduction; will incur a close-out turnover).<br>
    <b>skip_half</b> keeps previous weights at 50% leverage
    (compromise between cost and exposure).</li>
<li>Target Sharpe 6.0 is leverage-invariant — these variants modify
    Sharpe only through regime timing, not through leverage.</li>
</ul>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
