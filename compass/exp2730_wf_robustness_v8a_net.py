"""EXP-2730 — Walk-Forward Robustness Deep Dive on v8a NET.

The EXP-2600 headline claim is:
    v8a @ vt=0.12 → pooled OOS net Sharpe 6.16, CAGR 125.7%, DD 7.1%

This is a POOLED metric — all 20 OOS folds concatenated into one series,
then Sharpe computed on the concatenation. That number can hide large
per-fold variability. EXP-2730 runs the same pipeline but reports
PER-FOLD net metrics, then aggregates to answer the deployment question:

    Is the MEDIAN fold net Sharpe above 6.0, or is the pooled 6.16 a
    statistical artifact of a few lucky folds?

METHOD
======
  1. Rebuild v8a cube (EXP-2450 sparse 7-stream + QQQ from EXP-2590)
  2. 20-fold walk-forward:
       - Expanding-window train: all data from day 0 to fold cutoff
         (contrast with EXP-2280/EXP-2600 rolling 252d)
       - Test: 63 trading days after cutoff
       - Ledoit-Wolf covariance on train → risk-parity weights
       - Vol target 0.12, scale cap 20×
  3. Per-fold NET metrics (gross - flat 890 bps drag):
       net Sharpe, net CAGR, net max DD, weights, scale factor
  4. Aggregate:
       - Per-fold Sharpe distribution (min, p25, median, p75, max, mean, std)
       - % folds above Sharpe 6.0
       - % folds above CAGR 100%
       - % folds below DD 12%
       - Pooled OOS metrics (matches EXP-2600 published)
  5. Ship/don't-ship decision:
       IF median fold net Sharpe >= 6.0 AND pooled >= 6.0 → SHIP
       IF only pooled >= 6.0 → honest gap report
       ELSE → do not ship

Also runs a ROLLING-window variant (EXP-2600 methodology) for A/B
comparison to confirm the expanding vs rolling distinction is material.

Rule Zero: sparse v8a cube from EXP-2450 + cached EXP-2250 QQQ trades,
EXP-2570 drag rate (890.3 bps). No synthetic.

OUTPUT
  compass/reports/exp2730_wf_robustness_v8a_net.json
  compass/reports/exp2730_wf_robustness_v8a_net.html
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_JSON = ROOT / "compass" / "reports" / "exp2730_wf_robustness_v8a_net.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2730_wf_robustness_v8a_net.html"
QQQ_TRADES_PKL = ROOT / "compass" / "cache" / "exp2250_qqq_trades.pkl"

CAPITAL = 100_000
TRADING_DAYS = 252
TRAIN_DAYS_ROLLING = 252     # EXP-2280/2600 rolling window
TRAIN_DAYS_MIN_EXPAND = 252  # expanding-window minimum
TEST_DAYS = 63
TARGET_VOL = 0.12
SCALE_CAP = 20.0

# EXP-2570 drag (Alpaca commission-free + execution optimization)
NET_DRAG_BPS = 890.3
NET_DRAG_PCT = NET_DRAG_BPS / 100.0   # 8.903%


# ═══════════════════════════════════════════════════════════════════════════
# Cube builder (reuses EXP-2600 builder)
# ═══════════════════════════════════════════════════════════════════════════

def build_v8a_cube() -> pd.DataFrame:
    from compass.exp2450_sparse_combined_honest import build_sparse_seven_stream_cube
    base = build_sparse_seven_stream_cube()
    if not QQQ_TRADES_PKL.exists():
        raise FileNotFoundError(f"{QQQ_TRADES_PKL} missing")
    qqq_trades = pickle.load(QQQ_TRADES_PKL.open("rb"))
    qqq = pd.Series(0.0, index=base.index, name="qqq_cs")
    for t in qqq_trades:
        d = pd.Timestamp(t["exit_date"])
        if d in qqq.index:
            qqq.loc[d] += float(t["pnl"]) / CAPITAL
    cube = base.copy()
    cube["qqq_cs"] = qqq
    cols = ["exp1220", "v5_hedge", "gld_cal", "slv_cal",
            "cross_vol", "xlf_cs", "xli_cs", "qqq_cs"]
    return cube[cols]


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward drivers (rolling + expanding)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FoldResult:
    fold: int
    window_type: str              # 'rolling' or 'expanding'
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    vol_scale: float
    weights: Dict[str, float]
    gross_metrics: Dict[str, float]
    net_metrics: Dict[str, float]


def fold_metrics(r: pd.Series) -> Dict:
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {"n": n, "cagr_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "vol_pct": 0.0}
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sh = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = (1 + r).cumprod()
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1) if eq.iloc[-1] > 0 else -1.0
    dd = float((1 - eq / eq.cummax()).max())
    return {
        "n": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sh, 3),
        "max_dd_pct": round(dd * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
        "calmar": round(cagr / dd, 3) if dd > 1e-9 else 0.0,
    }


def walk_forward_v8a(cube: pd.DataFrame,
                       window_type: str = "rolling",
                       target_vol: float = TARGET_VOL,
                       ) -> Tuple[List[FoldResult], pd.Series]:
    """Run the 20-fold walk-forward.

    rolling:   train = last TRAIN_DAYS_ROLLING days ending at cutoff
    expanding: train = all data from day 0 to cutoff (min TRAIN_DAYS_MIN_EXPAND)

    Per-fold net returns subtract flat daily drag (NET_DRAG_PCT / 252).
    Returns (fold_results, pooled_net_series).
    """
    from compass.exp2360_robust_cov import cov_ledoit_wolf, risk_parity_weights

    cols = list(cube.columns)
    n = len(cube)
    folds: List[FoldResult] = []
    pooled_idx: List = []
    pooled_net_vals: List[float] = []

    daily_drag = NET_DRAG_PCT / 100.0 / TRADING_DAYS

    # Fold cutoffs — 20 folds, 63-day stride starting after initial training
    i = max(TRAIN_DAYS_ROLLING, TRAIN_DAYS_MIN_EXPAND)
    fold_ix = 0
    while i + TEST_DAYS <= n:
        if window_type == "rolling":
            train = cube.iloc[i - TRAIN_DAYS_ROLLING:i]
        elif window_type == "expanding":
            train = cube.iloc[:i]
        else:
            raise ValueError(f"unknown window_type: {window_type}")

        test = cube.iloc[i:i + TEST_DAYS]
        Sigma = cov_ledoit_wolf(train.values)
        w = risk_parity_weights(Sigma)
        train_port = train.values @ w
        train_vol = float(np.std(train_port, ddof=1)) * math.sqrt(TRADING_DAYS)
        scale = target_vol / train_vol if train_vol > 1e-10 else 1.0
        scale = float(np.clip(scale, 0.1, SCALE_CAP))
        gross_oos = pd.Series(test.values @ w * scale, index=test.index)
        net_oos = gross_oos - daily_drag

        fr = FoldResult(
            fold=fold_ix,
            window_type=window_type,
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            n_train=len(train),
            n_test=len(test),
            vol_scale=round(scale, 4),
            weights={cols[j]: round(float(w[j]), 4) for j in range(len(cols))},
            gross_metrics=fold_metrics(gross_oos),
            net_metrics=fold_metrics(net_oos),
        )
        folds.append(fr)
        pooled_idx.extend(test.index.tolist())
        pooled_net_vals.extend(net_oos.tolist())
        i += TEST_DAYS
        fold_ix += 1

    pooled_net = pd.Series(pooled_net_vals, index=pooled_idx, dtype=float)
    return folds, pooled_net


# ═══════════════════════════════════════════════════════════════════════════
# Distribution stats
# ═══════════════════════════════════════════════════════════════════════════

def distribution_stats(values: List[float], label: str) -> Dict:
    arr = np.array(values, dtype=float)
    if len(arr) == 0:
        return {"label": label, "n": 0}
    return {
        "label": label,
        "n": int(len(arr)),
        "min": round(float(arr.min()), 3),
        "p10": round(float(np.percentile(arr, 10)), 3),
        "p25": round(float(np.percentile(arr, 25)), 3),
        "median": round(float(np.median(arr)), 3),
        "mean": round(float(arr.mean()), 3),
        "p75": round(float(np.percentile(arr, 75)), 3),
        "p90": round(float(np.percentile(arr, 90)), 3),
        "max": round(float(arr.max()), 3),
        "std": round(float(arr.std(ddof=1)), 3) if len(arr) > 1 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Ship decision
# ═══════════════════════════════════════════════════════════════════════════

def ship_decision(pooled_net_sharpe: float, median_fold_sharpe: float,
                    pct_folds_above_6: float) -> Dict:
    """Ship only if BOTH median fold Sharpe AND pooled Sharpe clear 6.0
    AND at least 50% of folds are above 6.0."""
    median_ok = median_fold_sharpe >= 6.0
    pooled_ok = pooled_net_sharpe >= 6.0
    consistency_ok = pct_folds_above_6 >= 50.0

    if median_ok and pooled_ok and consistency_ok:
        decision = "SHIP"
        reason = (f"Median fold net Sharpe {median_fold_sharpe:.2f} ≥ 6.0, "
                   f"pooled {pooled_net_sharpe:.2f} ≥ 6.0, "
                   f"{pct_folds_above_6:.0f}% folds ≥ 6.0. "
                   f"All three criteria satisfied.")
    elif pooled_ok and not median_ok:
        decision = "GAP_REPORT"
        reason = (f"Pooled {pooled_net_sharpe:.2f} ≥ 6.0 but median "
                   f"{median_fold_sharpe:.2f} < 6.0. The pooled headline "
                   f"is a statistical artifact of a few high-Sharpe folds; "
                   f"typical deployment should expect ~{median_fold_sharpe:.2f}.")
    elif median_ok and not consistency_ok:
        decision = "GAP_REPORT"
        reason = (f"Median {median_fold_sharpe:.2f} ≥ 6.0 but only "
                   f"{pct_folds_above_6:.0f}% of folds ≥ 6.0 (need 50%+). "
                   f"Distribution is bimodal — high tail drags median up.")
    else:
        decision = "DO_NOT_SHIP"
        reason = (f"Neither pooled ({pooled_net_sharpe:.2f}) nor median "
                   f"({median_fold_sharpe:.2f}) clears the 6.0 gate.")
    return {
        "decision": decision,
        "reason": reason,
        "median_fold_sharpe": round(median_fold_sharpe, 3),
        "pooled_net_sharpe": round(pooled_net_sharpe, 3),
        "pct_folds_above_6": round(pct_folds_above_6, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2730 — Walk-Forward Robustness Deep Dive on v8a NET")
    print("=" * 72)

    print("\n[1/4] Building v8a 8-stream cube...")
    cube = build_v8a_cube()
    print(f"       shape {cube.shape}  cols: {list(cube.columns)}")
    print(f"       range {cube.index[0].date()} → {cube.index[-1].date()}")

    results = {}

    for window_type in ["rolling", "expanding"]:
        print(f"\n[2/4] Walk-forward ({window_type} window, target_vol={TARGET_VOL}, "
              f"drag={NET_DRAG_BPS}bps)...")
        folds, pooled_net = walk_forward_v8a(cube, window_type=window_type)
        pooled_m = fold_metrics(pooled_net)

        fold_sharpes = [f.net_metrics["sharpe"] for f in folds]
        fold_cagrs = [f.net_metrics["cagr_pct"] for f in folds]
        fold_dds = [f.net_metrics["max_dd_pct"] for f in folds]
        fold_gross_sharpes = [f.gross_metrics["sharpe"] for f in folds]

        sharpe_dist = distribution_stats(fold_sharpes, "net_sharpe")
        cagr_dist = distribution_stats(fold_cagrs, "net_cagr_pct")
        dd_dist = distribution_stats(fold_dds, "net_max_dd_pct")
        gross_dist = distribution_stats(fold_gross_sharpes, "gross_sharpe")

        pct_above_6 = float(np.mean(np.array(fold_sharpes) >= 6.0) * 100)
        pct_above_5 = float(np.mean(np.array(fold_sharpes) >= 5.0) * 100)
        pct_above_4 = float(np.mean(np.array(fold_sharpes) >= 4.0) * 100)
        pct_cagr_above_100 = float(np.mean(np.array(fold_cagrs) >= 100) * 100)
        pct_dd_under_12 = float(np.mean(np.array(fold_dds) <= 12) * 100)

        decision = ship_decision(pooled_m["sharpe"],
                                   sharpe_dist["median"], pct_above_6)

        # Print per-fold table
        print(f"\n{window_type.upper()} per-fold results:")
        print(f"  {'fold':>4}  {'test_start':>10}→{'test_end':<10}  "
              f"{'scale':>6}  {'gross_SR':>8}  {'net_SR':>7}  "
              f"{'net_CAGR':>9}  {'net_DD':>7}")
        for f in folds:
            g = f.gross_metrics
            n = f.net_metrics
            flag = "★" if n["sharpe"] >= 6.0 else ("+" if n["sharpe"] >= 5.0 else " ")
            print(f"  {flag} {f.fold:2d}  {f.test_start}→{f.test_end}  "
                  f"{f.vol_scale:6.2f}  {g['sharpe']:8.2f}  {n['sharpe']:7.2f}  "
                  f"{n['cagr_pct']:+8.1f}%  {n['max_dd_pct']:6.2f}%")

        print(f"\n{window_type.upper()} aggregate:")
        print(f"  pooled net Sharpe:  {pooled_m['sharpe']:.3f}")
        print(f"  pooled net CAGR:    {pooled_m['cagr_pct']:+.2f}%")
        print(f"  pooled net DD:      {pooled_m['max_dd_pct']:.2f}%")
        print(f"  median fold SR:     {sharpe_dist['median']:.3f}")
        print(f"  fold SR dist:       min {sharpe_dist['min']:.2f}  "
              f"p25 {sharpe_dist['p25']:.2f}  median {sharpe_dist['median']:.2f}  "
              f"p75 {sharpe_dist['p75']:.2f}  max {sharpe_dist['max']:.2f}")
        print(f"  % folds ≥ SR 6.0:   {pct_above_6:.0f}%")
        print(f"  % folds ≥ SR 5.0:   {pct_above_5:.0f}%")
        print(f"  % folds ≥ SR 4.0:   {pct_above_4:.0f}%")
        print(f"  % folds CAGR ≥ 100: {pct_cagr_above_100:.0f}%")
        print(f"  % folds DD < 12:    {pct_dd_under_12:.0f}%")
        print(f"  ship decision:      {decision['decision']}")
        print(f"                      {decision['reason']}")

        results[window_type] = {
            "pooled_net_metrics": pooled_m,
            "fold_distribution": {
                "net_sharpe": sharpe_dist,
                "gross_sharpe": gross_dist,
                "net_cagr_pct": cagr_dist,
                "net_max_dd_pct": dd_dist,
            },
            "fold_counts": {
                "n_folds": len(folds),
                "pct_above_sharpe_6": round(pct_above_6, 2),
                "pct_above_sharpe_5": round(pct_above_5, 2),
                "pct_above_sharpe_4": round(pct_above_4, 2),
                "pct_cagr_above_100": round(pct_cagr_above_100, 2),
                "pct_dd_under_12": round(pct_dd_under_12, 2),
            },
            "decision": decision,
            "folds": [
                {
                    "fold": f.fold,
                    "train_window": f"{f.train_start}→{f.train_end}",
                    "n_train": f.n_train,
                    "test_window": f"{f.test_start}→{f.test_end}",
                    "vol_scale": f.vol_scale,
                    "gross_sharpe": f.gross_metrics["sharpe"],
                    "net_sharpe": f.net_metrics["sharpe"],
                    "net_cagr_pct": f.net_metrics["cagr_pct"],
                    "net_max_dd_pct": f.net_metrics["max_dd_pct"],
                    "weights": f.weights,
                }
                for f in folds
            ],
        }

    # ── Final verdict
    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    for wt in ["rolling", "expanding"]:
        d = results[wt]["decision"]
        print(f"  {wt:10s}  {d['decision']:12s}  "
              f"pooled {d['pooled_net_sharpe']:.2f}  "
              f"median {d['median_fold_sharpe']:.2f}  "
              f"%≥6 {d['pct_folds_above_6']:.0f}%")
    print()
    primary_decision = results["rolling"]["decision"]["decision"]
    if primary_decision == "SHIP":
        print("  → SHIP: rolling-window walk-forward passes all three criteria.")
    elif primary_decision == "GAP_REPORT":
        print("  → GAP REPORT: the pooled 6.16 headline is not fully")
        print("    supported by per-fold median. See decision.reason.")
    else:
        print("  → DO NOT SHIP: walk-forward does not support the 6.16 claim.")

    # ── JSON
    payload = {
        "experiment": "EXP-2730",
        "title": "Walk-Forward Robustness Deep Dive on v8a NET",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "cube": "EXP-2450 sparse 7-stream + EXP-2250 cached QQQ trades",
            "cov_estimator": "compass.exp2360_robust_cov.cov_ledoit_wolf",
            "risk_parity": "compass.exp2360_robust_cov.risk_parity_weights",
            "drag": f"EXP-2570 {NET_DRAG_BPS} bps (Alpaca commfree + execution optimization)",
        },
        "config": {
            "target_vol": TARGET_VOL,
            "scale_cap": SCALE_CAP,
            "train_days_rolling": TRAIN_DAYS_ROLLING,
            "train_days_min_expanding": TRAIN_DAYS_MIN_EXPAND,
            "test_days": TEST_DAYS,
            "drag_pct_annual": NET_DRAG_PCT,
        },
        "baseline_claim_from_exp2600": {
            "pooled_net_sharpe": 6.164,
            "pooled_net_cagr_pct": 125.702,
            "pooled_net_max_dd_pct": 7.109,
        },
        "results": results,
        "primary_decision": results["rolling"]["decision"],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    REPORT_HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


def build_html(p: Dict) -> str:
    def render_fold_rows(folds: List[Dict]) -> str:
        out = ""
        for f in folds:
            ns = f["net_sharpe"]
            if ns >= 6.0:
                color = "#16a34a"
                marker = "★"
            elif ns >= 5.0:
                color = "#84cc16"
                marker = "+"
            elif ns >= 4.0:
                color = "#f59e0b"
                marker = ""
            else:
                color = "#dc2626"
                marker = ""
            out += (
                f"<tr><td>{f['fold']}{marker}</td>"
                f"<td>{f['test_window']}</td>"
                f"<td>{f['vol_scale']:.2f}×</td>"
                f"<td>{f['gross_sharpe']:.2f}</td>"
                f"<td style='color:{color};font-weight:700'>{ns:.2f}</td>"
                f"<td>{f['net_cagr_pct']:+.1f}%</td>"
                f"<td>{f['net_max_dd_pct']:.2f}%</td></tr>"
            )
        return out

    def render_dist(d: Dict) -> str:
        return (
            f"<tr><td>{d['label']}</td>"
            f"<td>{d['min']:.2f}</td>"
            f"<td>{d['p10']:.2f}</td>"
            f"<td>{d['p25']:.2f}</td>"
            f"<td style='font-weight:700'>{d['median']:.2f}</td>"
            f"<td>{d['mean']:.2f}</td>"
            f"<td>{d['p75']:.2f}</td>"
            f"<td>{d['p90']:.2f}</td>"
            f"<td>{d['max']:.2f}</td>"
            f"<td>{d['std']:.2f}</td></tr>"
        )

    def window_section(window_type: str) -> str:
        r = p["results"][window_type]
        d = r["decision"]
        pooled = r["pooled_net_metrics"]
        dist = r["fold_distribution"]
        cnt = r["fold_counts"]

        folds_html = render_fold_rows(r["folds"])
        dist_rows = "".join([
            render_dist(dist["net_sharpe"]),
            render_dist(dist["gross_sharpe"]),
            render_dist(dist["net_cagr_pct"]),
            render_dist(dist["net_max_dd_pct"]),
        ])

        color_decision = {
            "SHIP": "#16a34a",
            "GAP_REPORT": "#f59e0b",
            "DO_NOT_SHIP": "#dc2626",
        }.get(d["decision"], "#0f172a")

        return f"""
<h2>{window_type.upper()}-window walk-forward</h2>

<div style="background:#fff;border:2px solid {color_decision};border-radius:10px;padding:14px;margin:14px 0;">
<strong style="color:{color_decision};">Decision: {d['decision']}</strong><br>
{d['reason']}
</div>

<h3>Pooled OOS net metrics</h3>
<p>Sharpe <strong>{pooled['sharpe']:.3f}</strong> ·
CAGR <strong>{pooled['cagr_pct']:+.2f}%</strong> ·
Max DD <strong>{pooled['max_dd_pct']:.2f}%</strong> ·
Vol <strong>{pooled['vol_pct']:.2f}%</strong></p>

<h3>Per-fold distribution</h3>
<table>
<thead><tr>
<th>Metric</th><th>min</th><th>p10</th><th>p25</th><th>median</th>
<th>mean</th><th>p75</th><th>p90</th><th>max</th><th>std</th>
</tr></thead>
<tbody>{dist_rows}</tbody>
</table>

<h3>Fold counts</h3>
<p>
Total folds: <strong>{cnt['n_folds']}</strong><br>
% folds net Sharpe ≥ 6.0: <strong>{cnt['pct_above_sharpe_6']:.0f}%</strong><br>
% folds net Sharpe ≥ 5.0: <strong>{cnt['pct_above_sharpe_5']:.0f}%</strong><br>
% folds net Sharpe ≥ 4.0: <strong>{cnt['pct_above_sharpe_4']:.0f}%</strong><br>
% folds CAGR ≥ 100%:       <strong>{cnt['pct_cagr_above_100']:.0f}%</strong><br>
% folds DD &lt; 12%:           <strong>{cnt['pct_dd_under_12']:.0f}%</strong>
</p>

<h3>Per-fold table</h3>
<table>
<thead><tr>
<th>Fold</th><th>Test window</th><th>Vol scale</th>
<th>Gross SR</th><th>Net SR</th><th>Net CAGR</th><th>Net DD</th>
</tr></thead>
<tbody>{folds_html}</tbody>
</table>
"""

    rolling_html = window_section("rolling")
    expanding_html = window_section("expanding")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2730 — WF Robustness v8a NET</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1250px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.8em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
h3 {{ margin-top:1.5em;color:#475569; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.claim {{ background:#fefce8;border:1px solid #fde047;border-radius:8px;padding:14px;font-size:0.88rem;margin:14px 0; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.85em; }}
th {{ background:#f1f5f9;padding:8px 10px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.72em;text-transform:uppercase; }}
th:first-child, th:nth-child(2) {{ text-align:left; }}
td {{ padding:7px 10px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child, td:nth-child(2) {{ text-align:left; }}
</style></head><body>

<h1>EXP-2730 — Walk-Forward Robustness Deep Dive (v8a NET)</h1>
<p style="color:#64748b">20-fold walk-forward on the full NET pipeline —
Ledoit-Wolf covariance + risk-parity + 12% vol target + 890 bps drag ·
{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero:</strong> EXP-2450 sparse 7-stream cube + EXP-2250 cached
QQQ trades (real IronVault). EXP-2570 drag rate (Alpaca commission-free +
execution optimization). Ledoit-Wolf cov from compass.exp2360_robust_cov.
</div>

<div class="claim">
<strong>Claim from EXP-2600:</strong> v8a @ vt=0.12 → pooled net Sharpe
<strong>6.164</strong>, CAGR 125.7%, DD 7.1%. EXP-2730 validates per-fold.
</div>

{rolling_html}

{expanding_html}

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2730_wf_robustness_v8a_net.py · Rule Zero · all real data
</p>
</body></html>"""


if __name__ == "__main__":
    main()
