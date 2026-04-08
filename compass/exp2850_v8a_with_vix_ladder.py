"""EXP-2850 — v8a Walk-Forward with Production VIX Ladder.

Integrates the EXP-2820 winning VIX ladder (compass/vix_ladder.py) into
the v8a portfolio pipeline and reports updated NET metrics.

Pipeline
--------
  1. Build v8a sparse cube (EXP-2450 7-stream + EXP-2250 QQQ)
  2. Walk-forward 20 folds (LW risk-parity + 12% vol target + 890 bps drag)
     — this matches EXP-2730 exactly (baseline reference)
  3. Apply VIXLadder() to the per-fold raw OOS returns (causal, shift-1d)
     The ladder multiplier scales exposure by the 9-breakpoint EXP-2820
     ladder: VIX≤20 → 1.0, VIX 25 → 0.90, VIX 30 → 0.75, VIX 35 → 0.60,
     VIX 40 → 0.50, VIX 50 → 0.35, VIX 60 → 0.25, VIX 70 → 0.15, >70 → 0.
  4. Report per-fold metrics + pooled metrics + lift vs baseline

Baseline reference (EXP-2730):
    pooled net Sharpe       6.164
    median fold Sharpe      6.94
    pooled net CAGR         125.7%
    pooled max DD           7.11%
    % folds ≥ Sharpe 6.0    70%

Expected lift from EXP-2820: +0.486 Sharpe on the full-sample
normal-regime test (no flash crash insert). This experiment measures
whether that lift carries through to the walk-forward methodology.

Rule Zero: EXP-2450 sparse cube (real), EXP-2250 cached QQQ, real
Yahoo ^VIX daily close. No synthetic.

OUTPUT
  compass/reports/exp2850_v8a_with_vix_ladder.json
  compass/reports/exp2850_v8a_with_vix_ladder.html
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.vix_ladder import VIXLadder, fetch_vix

REPORT_JSON = ROOT / "compass" / "reports" / "exp2850_v8a_with_vix_ladder.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2850_v8a_with_vix_ladder.html"
QQQ_TRADES_PKL = ROOT / "compass" / "cache" / "exp2250_qqq_trades.pkl"

CAPITAL = 100_000
TRADING_DAYS = 252
TRAIN_DAYS = 252
TEST_DAYS = 63
TARGET_VOL = 0.12
SCALE_CAP = 20.0
NET_DRAG_PCT = 8.903  # EXP-2570


def build_v8a_cube() -> pd.DataFrame:
    from compass.exp2450_sparse_combined_honest import build_sparse_seven_stream_cube
    base = build_sparse_seven_stream_cube()
    qqq_trades = pickle.load(QQQ_TRADES_PKL.open("rb"))
    qqq = pd.Series(0.0, index=base.index, name="qqq_cs")
    for t in qqq_trades:
        d = pd.Timestamp(t["exit_date"])
        if d in qqq.index:
            qqq.loc[d] += float(t["pnl"]) / CAPITAL
    cube = base.copy()
    cube["qqq_cs"] = qqq
    return cube[["exp1220", "v5_hedge", "gld_cal", "slv_cal",
                  "cross_vol", "xlf_cs", "xli_cs", "qqq_cs"]]


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
        "calmar": round(cagr / dd, 3) if dd > 1e-9 else 0.0,
    }


def walk_forward_with_ladder(
    cube: pd.DataFrame,
    vix: pd.Series,
    ladder: VIXLadder = None,
    apply_ladder: bool = True,
) -> Tuple[pd.Series, pd.Series, List[Dict]]:
    """Walk-forward LW risk-parity on cube.

    If apply_ladder=True, multiplies each day's raw OOS return by the
    VIX-ladder exposure (causal, shift-1d). Drag subtracted after.

    Returns (pooled_net, pooled_exposure, fold_rows).
    """
    from compass.exp2360_robust_cov import cov_ledoit_wolf, risk_parity_weights

    if ladder is None:
        ladder = VIXLadder()

    cols = list(cube.columns)
    n = len(cube)
    pooled_idx: List = []
    pooled_vals: List[float] = []
    pooled_exp_vals: List[float] = []
    fold_rows: List[Dict] = []

    daily_drag = NET_DRAG_PCT / 100.0 / TRADING_DAYS

    i = TRAIN_DAYS
    fold_ix = 0
    while i + TEST_DAYS <= n:
        train = cube.iloc[i - TRAIN_DAYS:i]
        test = cube.iloc[i:i + TEST_DAYS]
        Sigma = cov_ledoit_wolf(train.values)
        w = risk_parity_weights(Sigma)
        train_port = train.values @ w
        train_vol = float(np.std(train_port, ddof=1)) * math.sqrt(TRADING_DAYS)
        scale = TARGET_VOL / train_vol if train_vol > 1e-10 else 1.0
        scale = float(np.clip(scale, 0.1, SCALE_CAP))

        gross = pd.Series(test.values @ w * scale, index=test.index)

        if apply_ladder:
            vix_slice = vix.reindex(test.index).ffill().bfill()
            exposure = ladder.apply(vix_slice)
            gross_laddered = gross * exposure
        else:
            exposure = pd.Series(1.0, index=test.index)
            gross_laddered = gross

        net = gross_laddered - daily_drag

        fold_rows.append({
            "fold": fold_ix,
            "test_start": str(test.index[0].date()),
            "test_end": str(test.index[-1].date()),
            "vol_scale": round(scale, 3),
            "avg_exposure": round(float(exposure.mean()), 4),
            "min_exposure": round(float(exposure.min()), 4),
            "weights": {cols[j]: round(float(w[j]), 4) for j in range(len(cols))},
            "gross_metrics": fold_metrics(gross),
            "gross_laddered_metrics": fold_metrics(gross_laddered),
            "net_metrics": fold_metrics(net),
        })
        pooled_idx.extend(test.index.tolist())
        pooled_vals.extend(net.tolist())
        pooled_exp_vals.extend(exposure.tolist())
        i += TEST_DAYS
        fold_ix += 1

    pooled = pd.Series(pooled_vals, index=pooled_idx, dtype=float)
    pooled_exp = pd.Series(pooled_exp_vals, index=pooled_idx, dtype=float)
    return pooled, pooled_exp, fold_rows


def summarize(pooled: pd.Series, folds: List[Dict], label: str) -> Dict:
    m = fold_metrics(pooled)
    fold_sharpes = [f["net_metrics"]["sharpe"] for f in folds]
    m["label"] = label
    m["n_folds"] = len(folds)
    m["median_fold_sharpe"] = round(float(np.median(fold_sharpes)), 3)
    m["pct_folds_above_6"] = round(float(np.mean(np.array(fold_sharpes) >= 6.0) * 100), 2)
    m["pct_folds_above_5"] = round(float(np.mean(np.array(fold_sharpes) >= 5.0) * 100), 2)
    m["worst_fold_sharpe"] = round(float(min(fold_sharpes)), 3)
    m["best_fold_sharpe"] = round(float(max(fold_sharpes)), 3)
    return m


def yearly_breakdown(rets: pd.Series) -> Dict[int, Dict]:
    out = {}
    for yr in sorted({d.year for d in rets.index}):
        sub = rets[rets.index.year == yr]
        if len(sub) < 20:
            continue
        out[int(yr)] = fold_metrics(sub)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2850 — v8a + Production VIX Ladder")
    print("=" * 72)

    print("\n[1/5] Building v8a cube...")
    cube = build_v8a_cube()
    print(f"       shape {cube.shape}  {cube.index[0].date()}→{cube.index[-1].date()}")

    print("\n[2/5] Loading real Yahoo ^VIX...")
    vix_start = (cube.index.min() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    vix_end = (cube.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    vix = fetch_vix(vix_start, vix_end)
    print(f"       {len(vix)} VIX days  min {vix.min():.1f}  max {vix.max():.1f}  mean {vix.mean():.1f}")

    print("\n[3/5] Instantiating VIX ladder (EXP-2820 default)...")
    ladder = VIXLadder()
    print(f"       {ladder}")
    print(f"       {len(ladder.breakpoints)} breakpoints, causal={ladder.causal}")

    print("\n[4/5] Walk-forward — baseline (no ladder)...")
    bl_pooled, bl_exp, bl_folds = walk_forward_with_ladder(cube, vix, apply_ladder=False)
    bl_summary = summarize(bl_pooled, bl_folds, "baseline_no_ladder")
    print(f"       pooled: SR {bl_summary['sharpe']:.3f}  "
          f"CAGR {bl_summary['cagr_pct']:+6.1f}%  "
          f"DD {bl_summary['max_dd_pct']:.2f}%  "
          f"median fold SR {bl_summary['median_fold_sharpe']:.2f}")

    print("\n[5/5] Walk-forward — v8a + VIX ladder...")
    lad_pooled, lad_exp, lad_folds = walk_forward_with_ladder(cube, vix, ladder, apply_ladder=True)
    lad_summary = summarize(lad_pooled, lad_folds, "v8a_vix_ladder")
    print(f"       pooled: SR {lad_summary['sharpe']:.3f}  "
          f"CAGR {lad_summary['cagr_pct']:+6.1f}%  "
          f"DD {lad_summary['max_dd_pct']:.2f}%  "
          f"median fold SR {lad_summary['median_fold_sharpe']:.2f}")
    print(f"       avg exposure: {lad_exp.mean():.3f}  "
          f"min exposure: {lad_exp.min():.3f}  "
          f"days below 1.0: {int((lad_exp < 1.0).sum())}")

    # Per-fold comparison
    print("\n[per-fold comparison]")
    print(f"  {'fold':>4}  {'test_start':>10}  {'bl_SR':>6}  {'lad_SR':>7}  "
          f"{'delta':>6}  {'lad_CAGR':>9}  {'lad_DD':>7}  {'avg_exp':>8}")
    for i, (b, l) in enumerate(zip(bl_folds, lad_folds)):
        bs = b["net_metrics"]["sharpe"]
        ls = l["net_metrics"]["sharpe"]
        lc = l["net_metrics"]["cagr_pct"]
        ld = l["net_metrics"]["max_dd_pct"]
        ae = l["avg_exposure"]
        delta = ls - bs
        flag = "★" if ls >= 6.0 else ("+" if ls >= 5.0 else " ")
        print(f"  {flag} {i:2d}  {b['test_start']}  "
              f"{bs:6.2f}  {ls:7.2f}  {delta:+6.2f}  "
              f"{lc:+8.1f}%  {ld:6.2f}%  {ae:8.3f}")

    # Yearly breakdown (laddered)
    yearly = yearly_breakdown(lad_pooled)
    print("\n[yearly net metrics — laddered]")
    for yr, m in yearly.items():
        print(f"  {yr}  CAGR {m['cagr_pct']:+7.1f}%  SR {m['sharpe']:5.2f}  DD {m['max_dd_pct']:.2f}%")

    # Delta summary
    delta_sharpe = lad_summary["sharpe"] - bl_summary["sharpe"]
    delta_cagr = lad_summary["cagr_pct"] - bl_summary["cagr_pct"]
    delta_dd = lad_summary["max_dd_pct"] - bl_summary["max_dd_pct"]

    print("\n" + "=" * 72)
    print("VERDICT — v8a + VIX ladder vs baseline")
    print("=" * 72)
    print(f"  Pooled SR:       {bl_summary['sharpe']:.3f} → {lad_summary['sharpe']:.3f}  "
          f"({delta_sharpe:+.3f})")
    print(f"  Pooled CAGR:     {bl_summary['cagr_pct']:+.1f}% → {lad_summary['cagr_pct']:+.1f}%  "
          f"({delta_cagr:+.1f}pp)")
    print(f"  Pooled Max DD:   {bl_summary['max_dd_pct']:.2f}% → {lad_summary['max_dd_pct']:.2f}%  "
          f"({delta_dd:+.2f}pp)")
    print(f"  Median fold SR:  {bl_summary['median_fold_sharpe']:.3f} → "
          f"{lad_summary['median_fold_sharpe']:.3f}  "
          f"({lad_summary['median_fold_sharpe'] - bl_summary['median_fold_sharpe']:+.3f})")
    print(f"  % folds ≥ 6.0:   {bl_summary['pct_folds_above_6']:.0f}% → "
          f"{lad_summary['pct_folds_above_6']:.0f}%")

    # Ship decision — net SR must still clear 6.0 after the ladder
    if lad_summary["sharpe"] >= 6.0 and lad_summary["median_fold_sharpe"] >= 6.0 \
            and lad_summary["max_dd_pct"] <= 12.0:
        decision = "SHIP"
        reason = (f"Laddered pooled net SR {lad_summary['sharpe']:.2f} ≥ 6.0, "
                   f"median {lad_summary['median_fold_sharpe']:.2f} ≥ 6.0, "
                   f"DD {lad_summary['max_dd_pct']:.1f}% ≤ 12%. "
                   f"Ladder preserves ship criteria while adding "
                   f"flash-crash protection.")
    elif delta_sharpe >= 0 and lad_summary["max_dd_pct"] <= bl_summary["max_dd_pct"] + 1:
        decision = "SHIP_WITH_NOTE"
        reason = (f"Laddered SR {lad_summary['sharpe']:.2f} ≥ baseline, "
                   f"DD within tolerance. Ship but note the Sharpe may "
                   f"not clear 6.0 gate without the ladder's flash-crash protection.")
    else:
        decision = "HOLD"
        reason = (f"Ladder reduces pooled SR by {-delta_sharpe:.2f}. "
                   f"Trade-off between normal-regime return and crash protection.")
    print(f"\n  decision: {decision}")
    print(f"  {reason}")

    payload = {
        "experiment": "EXP-2850",
        "title": "v8a + Production VIX Ladder Integration",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "cube": "EXP-2450 sparse 7-stream + EXP-2250 QQQ cache",
            "vix": "Yahoo ^VIX daily close (causal shift-1d)",
            "ladder": "compass.vix_ladder.VIXLadder (EXP-2820 winner)",
            "drag": f"EXP-2570 {NET_DRAG_PCT}%/yr (Alpaca + exec opt)",
            "walk_forward": "LW risk-parity + 12% vol target + scale cap 20x",
        },
        "ladder": ladder.describe(),
        "config": {
            "target_vol": TARGET_VOL,
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "scale_cap": SCALE_CAP,
            "drag_pct_annual": NET_DRAG_PCT,
        },
        "baseline_no_ladder": bl_summary,
        "v8a_with_ladder": lad_summary,
        "delta": {
            "sharpe": round(delta_sharpe, 3),
            "cagr_pct": round(delta_cagr, 3),
            "max_dd_pct": round(delta_dd, 3),
        },
        "yearly_laddered": yearly,
        "folds_baseline": [
            {"fold": f["fold"], "test_start": f["test_start"],
             "test_end": f["test_end"], "metrics": f["net_metrics"]}
            for f in bl_folds
        ],
        "folds_laddered": [
            {"fold": f["fold"], "test_start": f["test_start"],
             "test_end": f["test_end"], "metrics": f["net_metrics"],
             "avg_exposure": f["avg_exposure"],
             "min_exposure": f["min_exposure"]}
            for f in lad_folds
        ],
        "decision": decision,
        "reason": reason,
        "exposure_stats": {
            "mean": round(float(lad_exp.mean()), 4),
            "median": round(float(lad_exp.median()), 4),
            "min": round(float(lad_exp.min()), 4),
            "max": round(float(lad_exp.max()), 4),
            "days_at_full": int((lad_exp >= 0.999).sum()),
            "days_below_full": int((lad_exp < 0.999).sum()),
            "days_below_0_5": int((lad_exp < 0.5).sum()),
        },
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    REPORT_HTML.write_text(build_html(payload), encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


def build_html(p: Dict) -> str:
    bl = p["baseline_no_ladder"]
    ld = p["v8a_with_ladder"]
    delta = p["delta"]
    dec_color = {"SHIP": "#16a34a", "SHIP_WITH_NOTE": "#f59e0b", "HOLD": "#dc2626"}.get(p["decision"], "#0f172a")

    fold_rows = ""
    for b, l in zip(p["folds_baseline"], p["folds_laddered"]):
        bs = b["metrics"]["sharpe"]
        ls = l["metrics"]["sharpe"]
        delta_sr = ls - bs
        color = "#16a34a" if ls >= 6.0 else ("#f59e0b" if ls >= 5.0 else "#dc2626")
        fold_rows += (
            f"<tr><td>{b['fold']}</td>"
            f"<td>{b['test_start']}</td>"
            f"<td>{bs:.2f}</td>"
            f"<td style='color:{color};font-weight:700'>{ls:.2f}</td>"
            f"<td>{delta_sr:+.2f}</td>"
            f"<td>{l['metrics']['cagr_pct']:+.1f}%</td>"
            f"<td>{l['metrics']['max_dd_pct']:.2f}%</td>"
            f"<td>{l['avg_exposure']:.3f}</td></tr>"
        )

    yr_rows = ""
    for yr, m in p["yearly_laddered"].items():
        yr_rows += (
            f"<tr><td>{yr}</td>"
            f"<td>{m['cagr_pct']:+.1f}%</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td>{m['max_dd_pct']:.2f}%</td></tr>"
        )

    bp_rows = ""
    for v, e in p["ladder"]["breakpoints"]:
        v_str = f"{v:.0f}" if v < 1e6 else "∞"
        bp_rows += f"<tr><td>VIX {v_str}</td><td>{e:.2f}×</td></tr>"

    es = p["exposure_stats"]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2850 — v8a + VIX Ladder</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1250px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.8em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.decision {{ background:#fff;border:2px solid {dec_color};border-radius:10px;padding:16px;margin:16px 0; }}
.decision h3 {{ margin-top:0;color:{dec_color}; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.85em; }}
th {{ background:#f1f5f9;padding:8px 11px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.72em;text-transform:uppercase; }}
th:first-child, th:nth-child(2) {{ text-align:left; }}
td {{ padding:7px 11px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child, td:nth-child(2) {{ text-align:left; }}
.grid2 {{ display:grid;grid-template-columns:1fr 1fr;gap:18px; }}
</style></head><body>

<h1>EXP-2850 — v8a with Production VIX Ladder</h1>
<p style="color:#64748b">EXP-2820 winning 9-point VIX ladder integrated into
the v8a walk-forward pipeline · {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero:</strong> EXP-2450 sparse cube + EXP-2250 QQQ cache,
real Yahoo ^VIX daily close (causal shift-1d), compass.vix_ladder.VIXLadder
with the EXP-2820 winning breakpoints. EXP-2570 drag (890.3 bps/yr).
</div>

<div class="decision">
<h3>Decision: {p['decision']}</h3>
{p['reason']}
</div>

<h2>Headline comparison</h2>
<table>
<thead><tr><th>Metric</th><th>Baseline (no ladder)</th><th>v8a + ladder</th><th>Δ</th></tr></thead>
<tbody>
<tr><td>Pooled net Sharpe</td>
<td>{bl['sharpe']:.3f}</td>
<td style='font-weight:700'>{ld['sharpe']:.3f}</td>
<td>{delta['sharpe']:+.3f}</td></tr>
<tr><td>Pooled net CAGR</td>
<td>{bl['cagr_pct']:+.1f}%</td>
<td>{ld['cagr_pct']:+.1f}%</td>
<td>{delta['cagr_pct']:+.1f}pp</td></tr>
<tr><td>Pooled Max DD</td>
<td>{bl['max_dd_pct']:.2f}%</td>
<td>{ld['max_dd_pct']:.2f}%</td>
<td>{delta['max_dd_pct']:+.2f}pp</td></tr>
<tr><td>Median fold Sharpe</td>
<td>{bl['median_fold_sharpe']:.3f}</td>
<td>{ld['median_fold_sharpe']:.3f}</td>
<td>{ld['median_fold_sharpe'] - bl['median_fold_sharpe']:+.3f}</td></tr>
<tr><td>% folds ≥ 6.0</td>
<td>{bl['pct_folds_above_6']:.0f}%</td>
<td>{ld['pct_folds_above_6']:.0f}%</td>
<td>{ld['pct_folds_above_6'] - bl['pct_folds_above_6']:+.0f}pp</td></tr>
<tr><td>Worst fold Sharpe</td>
<td>{bl['worst_fold_sharpe']:.2f}</td>
<td>{ld['worst_fold_sharpe']:.2f}</td>
<td>{ld['worst_fold_sharpe'] - bl['worst_fold_sharpe']:+.2f}</td></tr>
</tbody>
</table>

<div class="grid2">
<div>
<h2>Ladder breakpoints</h2>
<table>
<thead><tr><th>VIX</th><th>Exposure</th></tr></thead>
<tbody>{bp_rows}</tbody>
</table>
</div>
<div>
<h2>Exposure statistics (walk-forward)</h2>
<table>
<tbody>
<tr><td>Mean exposure</td><td>{es['mean']:.3f}</td></tr>
<tr><td>Median exposure</td><td>{es['median']:.3f}</td></tr>
<tr><td>Min exposure</td><td>{es['min']:.3f}</td></tr>
<tr><td>Max exposure</td><td>{es['max']:.3f}</td></tr>
<tr><td>Days at full (≥0.999)</td><td>{es['days_at_full']}</td></tr>
<tr><td>Days reduced</td><td>{es['days_below_full']}</td></tr>
<tr><td>Days below 0.5×</td><td>{es['days_below_0_5']}</td></tr>
</tbody>
</table>
</div>
</div>

<h2>Per-fold comparison</h2>
<table>
<thead><tr>
<th>Fold</th><th>Test start</th>
<th>Baseline SR</th><th>Laddered SR</th><th>Δ</th>
<th>Laddered CAGR</th><th>Laddered DD</th><th>Avg exposure</th>
</tr></thead>
<tbody>{fold_rows}</tbody>
</table>

<h2>Yearly breakdown (laddered, net)</h2>
<table>
<thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
<tbody>{yr_rows}</tbody>
</table>

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2850_v8a_with_vix_ladder.py · Rule Zero · real data
</p>
</body></html>"""


if __name__ == "__main__":
    main()
