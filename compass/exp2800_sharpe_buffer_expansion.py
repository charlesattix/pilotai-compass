"""EXP-2800 — Sharpe Buffer Expansion via XLE 9th Stream.

EXP-2740 showed the v8a pooled net Sharpe 6.165 sits only 0.165 above
the 6.0 gate. 3 of 28 sensitivity perturbations breach:
    vol_target_-20%   5.998
    slippage_+50%     5.944
    v5_hedge_+20%     5.822  (worst)
The fragile buffer means any small execution or regime drift drops
the portfolio below the 6.0 ship gate.

This experiment tests whether adding XLE credit spreads (EXP-2710,
standalone trade Sharpe 1.87, WR 96.2%, correlation to exp1220
−0.016) as a 9th stream expands the buffer. XLE has extraordinarily
low correlation to every existing stream (|max corr| 0.023), so
Ledoit-Wolf risk-parity should load it and improve diversification.

METHOD
======
1. Build v9 cube = v8a (8 streams) + xle_cs as the 9th stream
   (sparse exit-date convention, aligned to the same business-day
   index as EXP-2450/2730)
2. Walk-forward Ledoit-Wolf risk-parity 20 folds at vt=0.12,
   scale_cap=20, same drag (EXP-2570 890.3 bps)
3. Measure new baseline pooled net Sharpe vs EXP-2740's 6.165
4. Run the SAME 28 sensitivity perturbations on the v9 cube:
   vol_target ±10/±20%, shrinkage ±20/±50%, per-stream weight ±20%,
   slippage +25/+50/+100%, spread widening
5. Compare buffer expansion: min perturbation Sharpe on v8a (5.822)
   vs on v9. Target: buffer improvement > 0.3 AND all perturbations
   clear 6.0.

Rule Zero: reuses EXP-2450 sparse cube, EXP-2250 cached QQQ trades,
freshly-built XLE trades via compass.exp2710_xle_integration
.run_xle_credit_spreads, EXP-2570 drag rate.

OUTPUT
  compass/reports/exp2800_sharpe_buffer_expansion.json
  compass/reports/exp2800_sharpe_buffer_expansion.html
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
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

REPORT_JSON = ROOT / "compass" / "reports" / "exp2800_sharpe_buffer_expansion.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2800_sharpe_buffer_expansion.html"

QQQ_TRADES_PKL = ROOT / "compass" / "cache" / "exp2250_qqq_trades.pkl"
XLE_TRADES_PKL = ROOT / "compass" / "cache" / "exp2800_xle_trades.pkl"
DB_PATH = ROOT / "data" / "options_cache.db"

CAPITAL = 100_000
TRADING_DAYS = 252
TRAIN_DAYS = 252
TEST_DAYS = 63
TARGET_VOL = 0.12
SCALE_CAP = 20.0
NET_DRAG_PCT = 8.903   # EXP-2570 Alpaca commfree + exec opt


# ═══════════════════════════════════════════════════════════════════════════
# Build v9 cube = v8a + XLE
# ═══════════════════════════════════════════════════════════════════════════

def load_xle_trades() -> List:
    if XLE_TRADES_PKL.exists():
        print(f"[cache] XLE trades from {XLE_TRADES_PKL.name}")
        return pickle.load(XLE_TRADES_PKL.open("rb"))
    print("[run] EXP-2710 XLE credit spread backtest (real IronVault)...")
    from compass.exp2710_xle_integration import run_xle_credit_spreads
    con = sqlite3.connect(str(DB_PATH))
    try:
        trades = run_xle_credit_spreads(con)
    finally:
        con.close()
    print(f"       {len(trades)} XLE trades")
    XLE_TRADES_PKL.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(trades, XLE_TRADES_PKL.open("wb"))
    return trades


def build_v9_cube() -> pd.DataFrame:
    from compass.exp2450_sparse_combined_honest import build_sparse_seven_stream_cube
    base = build_sparse_seven_stream_cube()
    print(f"[cube] base 7-stream shape: {base.shape}")

    # QQQ from EXP-2250 cache
    qqq_trades = pickle.load(QQQ_TRADES_PKL.open("rb"))
    qqq = pd.Series(0.0, index=base.index, name="qqq_cs")
    for t in qqq_trades:
        d = pd.Timestamp(t["exit_date"])
        if d in qqq.index:
            qqq.loc[d] += float(t["pnl"]) / CAPITAL

    # XLE from EXP-2710 (real IronVault run, cached after first call)
    xle_trades = load_xle_trades()
    xle = pd.Series(0.0, index=base.index, name="xle_cs")
    for t in xle_trades:
        d = pd.Timestamp(t.expiration)
        if d in xle.index:
            xle.loc[d] += float(t.pnl_pct_capital)

    v9 = base.copy()
    v9["qqq_cs"] = qqq
    v9["xle_cs"] = xle
    cols = ["exp1220", "v5_hedge", "gld_cal", "slv_cal",
            "cross_vol", "xlf_cs", "xli_cs", "qqq_cs", "xle_cs"]
    return v9[cols]


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward LW with parameterised drag/vol/slippage/shrinkage
# ═══════════════════════════════════════════════════════════════════════════

def fold_metrics(r: pd.Series) -> Dict:
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {"n": n, "sharpe": 0.0, "cagr_pct": 0.0,
                "max_dd_pct": 0.0, "vol_pct": 0.0}
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    sh = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = (1 + r).cumprod()
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1) if eq.iloc[-1] > 0 else -1.0
    dd = float((1 - eq / eq.cummax()).max())
    return {
        "n": n,
        "sharpe": round(sh, 3),
        "cagr_pct": round(cagr * 100, 3),
        "max_dd_pct": round(dd * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
    }


def walk_forward_lw(cube: pd.DataFrame,
                     target_vol: float = TARGET_VOL,
                     drag_pct_annual: float = NET_DRAG_PCT,
                     shrinkage_mult: float = 1.0,
                     weight_perturbation: Optional[Dict[str, float]] = None,
                     stream_vol_mult: Optional[Dict[str, float]] = None,
                     ) -> Tuple[pd.Series, List[Dict]]:
    """Walk-forward with LW risk-parity + optional perturbations.

    Parameters
    ----------
    target_vol        : annualised vol target
    drag_pct_annual   : flat annual drag subtracted from net returns
    shrinkage_mult    : multiply LW shrinkage toward sample cov
                        (>1 = more shrinkage, <1 = less)
    weight_perturbation : {stream: multiplier} applied to the LW weights
                          AFTER the optimizer (then renormalised)
    stream_vol_mult   : {stream: multiplier} applied to that stream's
                          returns before the optimizer sees them (used
                          to simulate higher slippage on a sleeve)
    """
    from compass.exp2360_robust_cov import cov_ledoit_wolf, risk_parity_weights

    # Apply stream-level vol multiplier (models slippage/cost increase
    # on a specific sleeve by scaling down its mean and vol proportionally)
    work = cube.copy()
    if stream_vol_mult:
        for col, mult in stream_vol_mult.items():
            if col in work.columns:
                work[col] = work[col] * mult

    cols = list(work.columns)
    n = len(work)
    pooled_idx: List = []
    pooled_vals: List[float] = []
    folds: List[Dict] = []

    daily_drag = drag_pct_annual / 100.0 / TRADING_DAYS

    i = TRAIN_DAYS
    fold_ix = 0
    while i + TEST_DAYS <= n:
        train = work.iloc[i - TRAIN_DAYS:i]
        test = work.iloc[i:i + TEST_DAYS]
        Sigma = cov_ledoit_wolf(train.values)

        # Optional shrinkage adjustment: blend Sigma with diag(Sigma)
        # shrinkage_mult > 1.0 → more diagonal (more shrinkage)
        if shrinkage_mult != 1.0:
            diag = np.diag(np.diag(Sigma))
            blend = max(0.0, min(1.0, (shrinkage_mult - 1.0) * 0.5))
            Sigma = (1.0 - blend) * Sigma + blend * diag

        w = risk_parity_weights(Sigma)

        # Post-hoc weight perturbation (renormalised)
        if weight_perturbation:
            w_adj = w.copy()
            for j, c in enumerate(cols):
                if c in weight_perturbation:
                    w_adj[j] *= weight_perturbation[c]
            w_adj = np.clip(w_adj, 0, None)
            if w_adj.sum() > 1e-9:
                w = w_adj / w_adj.sum()

        train_port = train.values @ w
        train_vol = float(np.std(train_port, ddof=1)) * math.sqrt(TRADING_DAYS)
        scale = target_vol / train_vol if train_vol > 1e-10 else 1.0
        scale = float(np.clip(scale, 0.1, SCALE_CAP))
        gross = pd.Series(test.values @ w * scale, index=test.index)
        net = gross - daily_drag

        folds.append({
            "fold": fold_ix,
            "test_start": str(test.index[0].date()),
            "test_end": str(test.index[-1].date()),
            "vol_scale": round(scale, 3),
            "weights": {cols[j]: round(float(w[j]), 4) for j in range(len(cols))},
            "gross_metrics": fold_metrics(gross),
            "net_metrics": fold_metrics(net),
        })
        pooled_idx.extend(test.index.tolist())
        pooled_vals.extend(net.tolist())
        i += TEST_DAYS
        fold_ix += 1

    pooled = pd.Series(pooled_vals, index=pooled_idx, dtype=float)
    return pooled, folds


def pooled_summary(pooled: pd.Series, folds: List[Dict]) -> Dict:
    m = fold_metrics(pooled)
    fold_sharpes = [f["net_metrics"]["sharpe"] for f in folds]
    m["median_fold_sharpe"] = round(float(np.median(fold_sharpes)), 3)
    m["pct_folds_above_6"] = round(float(np.mean(np.array(fold_sharpes) >= 6.0) * 100), 2)
    m["pct_folds_above_5"] = round(float(np.mean(np.array(fold_sharpes) >= 5.0) * 100), 2)
    m["ships"] = m["sharpe"] >= 6.0 and m["median_fold_sharpe"] >= 6.0
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity sweep — mirrors EXP-2740's 28 perturbations
# ═══════════════════════════════════════════════════════════════════════════

def run_sensitivity_sweep(cube: pd.DataFrame, label: str) -> Dict:
    print(f"\n[sensitivity] {label}: running 28-perturbation sweep...")
    results = {"vol_target": [], "shrinkage": [], "stream_weights": [],
                "slippage": [], "spread": []}

    # 1. Baseline
    pooled, folds = walk_forward_lw(cube)
    baseline = pooled_summary(pooled, folds)
    print(f"  baseline            pooled SR {baseline['sharpe']:.3f}  "
          f"median fold {baseline['median_fold_sharpe']:.2f}  "
          f"ships {baseline['ships']}")

    # 2. Vol target ±10/±20%
    for mult, label_suffix in [(0.8, "vol_target_-20%"),
                                 (0.9, "vol_target_-10%"),
                                 (1.1, "vol_target_+10%"),
                                 (1.2, "vol_target_+20%")]:
        pooled, folds = walk_forward_lw(cube, target_vol=TARGET_VOL * mult)
        m = pooled_summary(pooled, folds)
        m["label"] = label_suffix
        m["perturbation"] = f"target_vol × {mult}"
        results["vol_target"].append(m)
        print(f"  {label_suffix:22s}  pooled {m['sharpe']:.3f}  "
              f"median {m['median_fold_sharpe']:.2f}  ships {m['ships']}")

    # 3. Shrinkage ±20/±50%
    for mult, label_suffix in [(0.5, "shrinkage_-50%"),
                                 (0.8, "shrinkage_-20%"),
                                 (1.2, "shrinkage_+20%"),
                                 (1.5, "shrinkage_+50%")]:
        pooled, folds = walk_forward_lw(cube, shrinkage_mult=mult)
        m = pooled_summary(pooled, folds)
        m["label"] = label_suffix
        m["perturbation"] = f"LW shrinkage × {mult}"
        results["shrinkage"].append(m)
        print(f"  {label_suffix:22s}  pooled {m['sharpe']:.3f}  "
              f"median {m['median_fold_sharpe']:.2f}  ships {m['ships']}")

    # 4. Per-stream weight perturbation ±20%
    for col in cube.columns:
        for mult, suffix in [(0.8, "-20%"), (1.2, "+20%")]:
            pooled, folds = walk_forward_lw(
                cube, weight_perturbation={col: mult})
            m = pooled_summary(pooled, folds)
            m["label"] = f"{col}_{suffix}"
            m["perturbation"] = f"weight({col}) × {mult}"
            results["stream_weights"].append(m)
            ships_mark = "✓" if m["ships"] else " "
            print(f"  {ships_mark} {col}_{suffix:4s}        pooled {m['sharpe']:.3f}  "
                  f"median {m['median_fold_sharpe']:.2f}")

    # 5. Slippage +25/+50/+100% (modeled as extra drag)
    for mult, suffix in [(1.25, "slippage_+25%"),
                           (1.5,  "slippage_+50%"),
                           (2.0,  "slippage_+100%")]:
        pooled, folds = walk_forward_lw(
            cube, drag_pct_annual=NET_DRAG_PCT * mult)
        m = pooled_summary(pooled, folds)
        m["label"] = suffix
        m["perturbation"] = f"drag × {mult}"
        results["slippage"].append(m)
        ships_mark = "✓" if m["ships"] else " "
        print(f"  {ships_mark} {suffix:22s}  pooled {m['sharpe']:.3f}  "
              f"median {m['median_fold_sharpe']:.2f}")

    # 6. Spread widening — modelled as vol multiplier on all spread sleeves
    for mult, suffix in [(1.1, "spread_+10%"),
                           (1.2, "spread_+20%")]:
        spread_streams = {c: mult for c in cube.columns
                          if c.endswith("_cs")}
        pooled, folds = walk_forward_lw(
            cube, stream_vol_mult=spread_streams)
        m = pooled_summary(pooled, folds)
        m["label"] = suffix
        m["perturbation"] = f"credit-spread streams vol × {mult}"
        results["spread"].append(m)
        ships_mark = "✓" if m["ships"] else " "
        print(f"  {ships_mark} {suffix:22s}  pooled {m['sharpe']:.3f}  "
              f"median {m['median_fold_sharpe']:.2f}")

    return {"baseline": baseline, "perturbations": results}


def summarize_buffer(sensitivity: Dict, label: str) -> Dict:
    all_items = []
    for section in ("vol_target", "shrinkage", "stream_weights",
                     "slippage", "spread"):
        all_items.extend(sensitivity["perturbations"][section])
    n = len(all_items)
    sharpes = [item["sharpe"] for item in all_items]
    min_sr = min(sharpes)
    worst = min(all_items, key=lambda x: x["sharpe"])
    below_6 = [x for x in all_items if x["sharpe"] < 6.0]
    baseline = sensitivity["baseline"]
    buffer = min_sr - 6.0   # distance from worst perturbation to 6.0 gate

    return {
        "label": label,
        "baseline_pooled_sharpe": baseline["sharpe"],
        "baseline_median_fold_sharpe": baseline["median_fold_sharpe"],
        "n_perturbations": n,
        "min_perturbation_sharpe": round(float(min_sr), 3),
        "worst_perturbation": worst["label"],
        "buffer_above_6": round(float(buffer), 3),
        "n_below_6": len(below_6),
        "breaches": [{"label": b["label"], "sharpe": b["sharpe"]}
                      for b in below_6],
        "all_ship": len(below_6) == 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2800 — Sharpe Buffer Expansion via XLE 9th Stream")
    print("=" * 72)

    print("\n[1/4] Building v9 cube (v8a + XLE)...")
    v9 = build_v9_cube()
    print(f"       shape {v9.shape}  cols: {list(v9.columns)}")
    xle_nonzero = int((v9["xle_cs"] != 0).sum())
    print(f"       XLE nonzero days: {xle_nonzero}")

    # Also build v8a (without XLE) for A/B comparison
    v8a = v9.drop(columns=["xle_cs"])
    print(f"[1/4] v8a comparison cube shape: {v8a.shape}")

    print("\n[2/4] Baseline walk-forward (v8a vs v9)...")
    for label, cube in [("v8a (8-stream)", v8a), ("v9 (9-stream +XLE)", v9)]:
        pooled, folds = walk_forward_lw(cube)
        m = pooled_summary(pooled, folds)
        print(f"  {label:24s}  pooled SR {m['sharpe']:.3f}  "
              f"median {m['median_fold_sharpe']:.2f}  "
              f"%≥6 {m['pct_folds_above_6']:.0f}%  "
              f"CAGR {m['cagr_pct']:+6.1f}%  DD {m['max_dd_pct']:.2f}%")

    print("\n[3/4] Running 28-perturbation sensitivity sweep on BOTH cubes...")
    v8a_sens = run_sensitivity_sweep(v8a, "v8a baseline")
    v9_sens = run_sensitivity_sweep(v9, "v9 +XLE")

    print("\n[4/4] Buffer analysis...")
    v8a_buf = summarize_buffer(v8a_sens, "v8a")
    v9_buf = summarize_buffer(v9_sens, "v9")
    buffer_delta = v9_buf["buffer_above_6"] - v8a_buf["buffer_above_6"]

    print(f"\n{'':30s}  {'v8a':>10s}  {'v9':>10s}  {'Δ':>8s}")
    print(f"  {'baseline pooled SR':30s}  "
          f"{v8a_buf['baseline_pooled_sharpe']:10.3f}  "
          f"{v9_buf['baseline_pooled_sharpe']:10.3f}  "
          f"{v9_buf['baseline_pooled_sharpe'] - v8a_buf['baseline_pooled_sharpe']:+8.3f}")
    print(f"  {'median fold Sharpe':30s}  "
          f"{v8a_buf['baseline_median_fold_sharpe']:10.3f}  "
          f"{v9_buf['baseline_median_fold_sharpe']:10.3f}  "
          f"{v9_buf['baseline_median_fold_sharpe'] - v8a_buf['baseline_median_fold_sharpe']:+8.3f}")
    print(f"  {'min perturbation Sharpe':30s}  "
          f"{v8a_buf['min_perturbation_sharpe']:10.3f}  "
          f"{v9_buf['min_perturbation_sharpe']:10.3f}  "
          f"{v9_buf['min_perturbation_sharpe'] - v8a_buf['min_perturbation_sharpe']:+8.3f}")
    print(f"  {'buffer above 6.0 (min-6)':30s}  "
          f"{v8a_buf['buffer_above_6']:+10.3f}  "
          f"{v9_buf['buffer_above_6']:+10.3f}  "
          f"{buffer_delta:+8.3f}")
    print(f"  {'# perturbations below 6':30s}  "
          f"{v8a_buf['n_below_6']:10d}  {v9_buf['n_below_6']:10d}")
    print(f"  {'all perturbations ship':30s}  "
          f"{'yes' if v8a_buf['all_ship'] else 'NO':>10s}  "
          f"{'yes' if v9_buf['all_ship'] else 'NO':>10s}")

    print(f"\n  Worst v8a perturbation: {v8a_buf['worst_perturbation']} "
          f"SR {v8a_buf['min_perturbation_sharpe']:.3f}")
    print(f"  Worst v9  perturbation: {v9_buf['worst_perturbation']} "
          f"SR {v9_buf['min_perturbation_sharpe']:.3f}")

    # Decision
    buffer_improved_by_0_3 = buffer_delta > 0.3
    all_ship = v9_buf["all_ship"]
    if all_ship and buffer_improved_by_0_3:
        decision = "PROMOTE_V9"
        reason = (f"XLE expands buffer by {buffer_delta:+.2f} Sharpe points "
                  f"(> 0.3 target) AND all 28 perturbations clear 6.0.")
    elif all_ship and not buffer_improved_by_0_3:
        decision = "MARGINAL_V9"
        reason = (f"All 28 perturbations clear 6.0 but buffer improvement "
                  f"{buffer_delta:+.2f} is below the 0.3 target.")
    elif buffer_improved_by_0_3 and not all_ship:
        decision = "PARTIAL_V9"
        reason = (f"Buffer expands by {buffer_delta:+.2f} but "
                  f"{v9_buf['n_below_6']} perturbations still breach 6.0.")
    else:
        decision = "KEEP_V8A"
        reason = (f"XLE does not materially improve buffer "
                  f"({buffer_delta:+.2f}) and {v9_buf['n_below_6']} "
                  f"perturbations still breach.")
    print(f"\n[decision] {decision}")
    print(f"  {reason}")

    # ── JSON
    payload = {
        "experiment": "EXP-2800",
        "title": "Sharpe Buffer Expansion via XLE 9th Stream",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "v8a_cube": "EXP-2450 sparse 7-stream + EXP-2250 QQQ trades",
            "xle_trades": "compass.exp2710_xle_integration.run_xle_credit_spreads (real IronVault)",
            "drag": f"EXP-2570 {NET_DRAG_PCT}% annual (Alpaca commfree + exec opt)",
            "walk_forward": "LW risk-parity + 12% vol target + 890 bps drag",
        },
        "config": {
            "target_vol": TARGET_VOL,
            "scale_cap": SCALE_CAP,
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "drag_pct_annual": NET_DRAG_PCT,
        },
        "exp2740_baseline_reference": {
            "pooled_net_sharpe": 6.165,
            "median_fold_sharpe": 6.943,
            "pct_folds_above_6": 70.0,
            "worst_perturbation": "v5_hedge_+20%",
            "worst_sharpe": 5.822,
            "n_breaches": 3,
        },
        "v8a_sensitivity": v8a_sens,
        "v9_sensitivity": v9_sens,
        "v8a_buffer": v8a_buf,
        "v9_buffer": v9_buf,
        "buffer_delta": round(float(buffer_delta), 3),
        "decision": decision,
        "reason": reason,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    REPORT_HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


def build_html(p: Dict) -> str:
    v8a = p["v8a_buffer"]
    v9 = p["v9_buffer"]
    delta = p["buffer_delta"]
    decision = p["decision"]
    decision_color = {
        "PROMOTE_V9": "#16a34a",
        "MARGINAL_V9": "#f59e0b",
        "PARTIAL_V9": "#f59e0b",
        "KEEP_V8A": "#dc2626",
    }.get(decision, "#0f172a")

    def perturbation_rows(sens: Dict, section_name: str) -> str:
        rows = ""
        items = sens["perturbations"].get(section_name, [])
        for item in items:
            sr = item["sharpe"]
            color = "#16a34a" if sr >= 6.0 else "#dc2626"
            rows += (
                f"<tr><td>{item.get('label','?')}</td>"
                f"<td style='color:{color};font-weight:700'>{sr:.3f}</td>"
                f"<td>{item.get('median_fold_sharpe',0):.2f}</td>"
                f"<td>{item.get('pct_folds_above_6',0):.0f}%</td>"
                f"<td>{item.get('cagr_pct',0):.0f}%</td>"
                f"<td>{item.get('max_dd_pct',0):.1f}%</td></tr>"
            )
        return rows

    sections = []
    for name in ("vol_target", "shrinkage", "stream_weights", "slippage", "spread"):
        v8_rows = perturbation_rows(p["v8a_sensitivity"], name)
        v9_rows = perturbation_rows(p["v9_sensitivity"], name)
        sections.append(f"""
<h3>{name}</h3>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
<div><strong>v8a</strong>
<table><thead><tr><th>Label</th><th>Sharpe</th><th>Med fold</th><th>%≥6</th><th>CAGR</th><th>DD</th></tr></thead>
<tbody>{v8_rows}</tbody></table></div>
<div><strong>v9 (+XLE)</strong>
<table><thead><tr><th>Label</th><th>Sharpe</th><th>Med fold</th><th>%≥6</th><th>CAGR</th><th>DD</th></tr></thead>
<tbody>{v9_rows}</tbody></table></div>
</div>""")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2800 — Sharpe Buffer Expansion</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.8em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
h3 {{ color:#475569;margin-top:1.5em; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.decision {{ background:#fff;border:2px solid {decision_color};border-radius:10px;padding:16px;margin:16px 0; }}
.decision h3 {{ margin-top:0;color:{decision_color}; }}
table {{ width:100%;border-collapse:collapse;margin:10px 0;font-size:0.82em; }}
th {{ background:#f1f5f9;padding:7px 9px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.7em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:6px 9px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
</style></head><body>

<h1>EXP-2800 — Sharpe Buffer Expansion via XLE 9th Stream</h1>
<p style="color:#64748b">Adding XLE credit spreads as the 9th stream to
expand the Sharpe 6.0 buffer · {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero:</strong> EXP-2450 sparse 7-stream + EXP-2250 QQQ cache
+ compass.exp2710_xle_integration.run_xle_credit_spreads on real IronVault
data (26 XLE trades, WR 96.2%, trade Sharpe 1.87, corr to exp1220 -0.016).
EXP-2570 drag model (890.3 bps annual).
</div>

<div class="decision">
<h3>Decision: {decision}</h3>
{p['reason']}
</div>

<h2>Buffer comparison</h2>
<table>
<thead><tr><th>Metric</th><th>v8a (8-stream)</th><th>v9 (+XLE)</th><th>Δ</th></tr></thead>
<tbody>
<tr><td>baseline pooled SR</td>
<td>{v8a['baseline_pooled_sharpe']:.3f}</td>
<td>{v9['baseline_pooled_sharpe']:.3f}</td>
<td>{v9['baseline_pooled_sharpe'] - v8a['baseline_pooled_sharpe']:+.3f}</td></tr>
<tr><td>median fold Sharpe</td>
<td>{v8a['baseline_median_fold_sharpe']:.3f}</td>
<td>{v9['baseline_median_fold_sharpe']:.3f}</td>
<td>{v9['baseline_median_fold_sharpe'] - v8a['baseline_median_fold_sharpe']:+.3f}</td></tr>
<tr><td>min perturbation SR (worst case)</td>
<td>{v8a['min_perturbation_sharpe']:.3f}</td>
<td>{v9['min_perturbation_sharpe']:.3f}</td>
<td style='font-weight:700'>{v9['min_perturbation_sharpe'] - v8a['min_perturbation_sharpe']:+.3f}</td></tr>
<tr><td><strong>buffer above 6.0</strong></td>
<td>{v8a['buffer_above_6']:+.3f}</td>
<td>{v9['buffer_above_6']:+.3f}</td>
<td style='font-weight:700;color:{decision_color}'>{delta:+.3f}</td></tr>
<tr><td># perturbations below 6</td>
<td>{v8a['n_below_6']}</td>
<td>{v9['n_below_6']}</td>
<td>{v9['n_below_6'] - v8a['n_below_6']:+d}</td></tr>
<tr><td>worst perturbation</td>
<td>{v8a['worst_perturbation']}</td>
<td>{v9['worst_perturbation']}</td>
<td>—</td></tr>
</tbody>
</table>

<h2>Full sensitivity matrix</h2>
{''.join(sections)}

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2800_sharpe_buffer_expansion.py · Rule Zero · real data
</p>
</body></html>"""


if __name__ == "__main__":
    main()
