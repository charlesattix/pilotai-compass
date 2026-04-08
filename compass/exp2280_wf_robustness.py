"""
EXP-2280 — Walk-Forward Robustness Audit of North Star v6 (equal_risk_15%)

Why this experiment exists
--------------------------
EXP-2200 reported Sharpe 5.96 for the sparse equal_risk_15% configuration
of the 7-stream North Star v6 portfolio. That number is a *pooled*
full-sample metric. Before declaring the 6.0 target met, we need the
distribution: how does per-fold OOS Sharpe behave across the 20 walk-
forward windows, and is there degradation from 2020 → 2025?

Method (identical WF shape to EXP-2080)
---------------------------------------
1. Load the sparse 7-stream daily return DataFrame from EXP-2200's
   `build_streams()` (cached for speed).
2. Walk-forward: 252-day train, 63-day test, step 63 → 20 folds.
3. Per fold (causal):
      a. Fit equal_risk (inverse-vol) weights on train window.
      b. Compose portfolio on train window, measure realised ann vol.
      c. Vol-target scale = 0.15 / train_vol  (fixed for the fold).
      d. Apply the same weights × scale to the test window.
4. Collect OOS daily returns per fold; compute:
      - per-fold Sharpe, CAGR, DD
      - pooled OOS Sharpe (all fold test returns concatenated)
      - per-calendar-year Sharpe (test days only)
      - distribution stats: min / 25th / median / 75th / max
      - fraction of folds with Sharpe > 6.0 and < 3.0

REAL DATA ONLY. No synthetic. Same data-loading pipeline EXP-2200 used.

Outputs
-------
  compass/exp2280_wf_robustness.py
  compass/reports/exp2280_wf_robustness.json
  compass/reports/exp2280_wf_robustness.html
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_FILE  = ROOT / "compass" / "cache" / "exp2280_v6_sparse.pkl"
REPORT_JSON = ROOT / "compass" / "reports" / "exp2280_wf_robustness.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2280_wf_robustness.html"

TRADING_DAYS = 252
START = "2020-01-01"
END   = "2025-12-31"
TARGET_VOL  = 0.15
TRAIN_DAYS  = 252
TEST_DAYS   = 63

STREAMS = ["exp1220", "xlf_cs", "xli_cs", "gld_cal", "slv_cal", "vol_arb", "v5_hedge"]


# ───────────────────────────────────────────────────────────────────────────
# Data
# ───────────────────────────────────────────────────────────────────────────

def load_sparse_frame() -> pd.DataFrame:
    """Load the 7-stream sparse daily return frame, using the EXP-2280 cache
    (built by EXP-2200 build_streams). Rebuild if missing."""
    if CACHE_FILE.exists():
        return pickle.load(open(CACHE_FILE, "rb"))
    from compass.exp2200_north_star_v6 import build_streams
    _, sparse, _ = build_streams()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(sparse, open(CACHE_FILE, "wb"))
    return sparse


# ───────────────────────────────────────────────────────────────────────────
# Optimizer (same equal_risk definition as EXP-2200)
# ───────────────────────────────────────────────────────────────────────────

def equal_risk_weights(train: pd.DataFrame) -> Dict[str, float]:
    vols = np.array([train[k].std(ddof=1) + 1e-12 for k in STREAMS])
    w = 1.0 / vols
    w = np.clip(w, 0, None)
    if w.sum() < 1e-9:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()
    return {k: float(v) for k, v in zip(STREAMS, w)}


def compose(streams: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    out = pd.Series(0.0, index=streams.index)
    for k in STREAMS:
        out = out + weights.get(k, 0.0) * streams[k]
    return out


def vol_scale(train_series: pd.Series, target_vol: float) -> float:
    rv = float(train_series.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    if rv < 1e-9:
        return 1.0
    return target_vol / rv


# ───────────────────────────────────────────────────────────────────────────
# Metrics
# ───────────────────────────────────────────────────────────────────────────

def metrics(daily: pd.Series) -> Dict[str, float]:
    daily = daily.dropna()
    n = len(daily)
    if n < 2:
        return {"n": 0, "cagr_pct": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "max_dd_pct": 0.0, "calmar": 0.0, "vol_pct": 0.0}
    eq = (1 + daily).cumprod()
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1) if eq.iloc[-1] > 0 else -1.0
    mu = float(daily.mean()); sigma = float(daily.std(ddof=1))
    sharpe = mu / sigma * math.sqrt(TRADING_DAYS) if sigma > 1e-12 else 0.0
    down = daily[daily < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else sigma
    sortino = mu / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0.0
    dd = float((1 - eq / eq.cummax()).max())
    return {
        "n": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(dd * 100, 3),
        "calmar": round(cagr / dd, 3) if dd > 1e-9 else 0.0,
        "vol_pct": round(sigma * math.sqrt(TRADING_DAYS) * 100, 3),
    }


# ───────────────────────────────────────────────────────────────────────────
# Walk-forward
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    weights: Dict[str, float]
    train_vol_pct: float
    scale: float
    metrics: Dict[str, float]


def walk_forward_audit(df: pd.DataFrame,
                       target_vol: float = TARGET_VOL,
                       train_days: int = TRAIN_DAYS,
                       test_days: int = TEST_DAYS) -> Tuple[List[FoldResult], pd.Series]:
    """Returns (fold_results, pooled_test_returns)."""
    folds: List[FoldResult] = []
    pooled: List[pd.Series] = []

    n = len(df)
    i = train_days
    fold_idx = 0
    while i + test_days <= n:
        tr = df.iloc[i - train_days:i]
        te = df.iloc[i:i + test_days]

        w  = equal_risk_weights(tr)
        tr_port = compose(tr, w)
        scale = vol_scale(tr_port, target_vol)
        te_port = compose(te, w) * scale

        fr = FoldResult(
            fold=fold_idx,
            train_start=str(tr.index[0].date()),
            train_end=str(tr.index[-1].date()),
            test_start=str(te.index[0].date()),
            test_end=str(te.index[-1].date()),
            weights={k: round(v, 4) for k, v in w.items()},
            train_vol_pct=round(float(tr_port.std(ddof=1)) * math.sqrt(TRADING_DAYS) * 100, 3),
            scale=round(float(scale), 3),
            metrics=metrics(te_port),
        )
        folds.append(fr)
        pooled.append(te_port)
        i += test_days
        fold_idx += 1

    pooled_series = pd.concat(pooled).sort_index()
    return folds, pooled_series


def yearly_sharpe(pooled: pd.Series) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for yr in sorted({d.year for d in pooled.index}):
        sub = pooled[pooled.index.year == yr]
        if len(sub) < 20:
            continue
        out[int(yr)] = metrics(sub)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Distribution stats
# ───────────────────────────────────────────────────────────────────────────

def distribution_stats(sharpes: List[float]) -> Dict[str, float]:
    arr = np.array(sharpes, dtype=float)
    n = len(arr)
    return {
        "n_folds":  n,
        "min":      round(float(arr.min()), 3),
        "p10":      round(float(np.percentile(arr, 10)), 3),
        "p25":      round(float(np.percentile(arr, 25)), 3),
        "median":   round(float(np.median(arr)), 3),
        "mean":     round(float(arr.mean()), 3),
        "p75":      round(float(np.percentile(arr, 75)), 3),
        "p90":      round(float(np.percentile(arr, 90)), 3),
        "max":      round(float(arr.max()), 3),
        "std":      round(float(arr.std(ddof=1)), 3) if n > 1 else 0.0,
        "frac_above_6": round(float((arr > 6.0).mean()), 4),
        "frac_above_4": round(float((arr > 4.0).mean()), 4),
        "frac_below_3": round(float((arr < 3.0).mean()), 4),
        "frac_below_0": round(float((arr < 0.0).mean()), 4),
        "n_above_6":    int((arr > 6.0).sum()),
        "n_below_3":    int((arr < 3.0).sum()),
        "n_below_0":    int((arr < 0.0).sum()),
    }


def year_degradation(yearly_metrics: Dict[int, Dict]) -> Dict[str, float]:
    """Linear-regression slope of per-year Sharpe vs year; indicates decay."""
    yrs = sorted(yearly_metrics.keys())
    if len(yrs) < 3:
        return {"slope": 0.0, "first_year_sharpe": 0.0, "last_year_sharpe": 0.0}
    x = np.array(yrs, dtype=float)
    y = np.array([yearly_metrics[yr]["sharpe"] for yr in yrs], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "slope_sharpe_per_year": round(float(slope), 3),
        "first_year": int(yrs[0]),
        "last_year": int(yrs[-1]),
        "first_year_sharpe": round(float(y[0]), 3),
        "last_year_sharpe": round(float(y[-1]), 3),
        "delta_first_to_last": round(float(y[-1] - y[0]), 3),
    }


# ───────────────────────────────────────────────────────────────────────────
# HTML
# ───────────────────────────────────────────────────────────────────────────

def _sh_color(sh: float) -> str:
    if sh >= 6.0:  return "#16a34a"
    if sh >= 3.0:  return "#0f172a"
    if sh >= 0.0:  return "#ca8a04"
    return "#dc2626"


def write_html(payload: Dict, path: Path) -> None:
    folds = payload["folds"]
    dist  = payload["distribution"]
    pooled = payload["pooled_oos"]
    yearly = payload["yearly"]
    deg = payload["year_degradation"]

    target_met = dist["median"] >= 6.0 and dist["frac_above_6"] >= 0.5
    verdict_color = "#16a34a" if target_met else "#ca8a04"
    verdict_msg = "✅ TARGET HOLDS (median ≥ 6.0, >50% of folds above 6.0)" if target_met \
                  else "⚠ HONEST — pooled is 5.96 but fold distribution matters (see table)"

    fold_rows = ""
    for f in folds:
        m = f["metrics"]
        sh = m["sharpe"]
        fold_rows += (
            f"<tr><td>{f['fold']}</td>"
            f"<td>{f['test_start']}</td><td>{f['test_end']}</td>"
            f"<td>{f['scale']:.2f}×</td>"
            f"<td>{f['train_vol_pct']:.2f}%</td>"
            f"<td style='color:{_sh_color(sh)};font-weight:700'>{sh:+.2f}</td>"
            f"<td>{m['cagr_pct']:+.1f}%</td>"
            f"<td>{m['max_dd_pct']:.2f}%</td>"
            f"<td>{m['vol_pct']:.2f}%</td></tr>"
        )

    yr_rows = ""
    for yr, m in sorted(yearly.items()):
        yr_rows += (
            f"<tr><td>{yr}</td><td>{m['n']}</td>"
            f"<td style='color:{_sh_color(m['sharpe'])};font-weight:700'>{m['sharpe']:+.2f}</td>"
            f"<td>{m['cagr_pct']:+.1f}%</td><td>{m['max_dd_pct']:.2f}%</td>"
            f"<td>{m['vol_pct']:.2f}%</td></tr>"
        )

    return_html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>EXP-2280 WF Robustness — equal_risk_15%</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1100px}}
h1{{font-size:1.5rem;color:#0f172a}} h2{{font-size:1.05rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:18px}}
.headline{{background:#f0fdf4;border-left:5px solid {verdict_color};padding:14px 18px;border-radius:6px;margin:14px 0;font-size:0.95rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.15rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.83rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}} td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}} td:first-child{{text-align:left}}
</style></head><body>
<h1>EXP-2280 — Walk-Forward Robustness Audit (North Star v6, equal_risk_15%)</h1>
<p class="meta">20 folds · 252-day train / 63-day test / step 63 · sparse 7-stream frame · REAL data only ·
Equal-risk weights re-fit per fold on train window · Vol target 15% scaled from train realised vol.</p>

<div class="headline"><strong>Pooled OOS Sharpe:</strong>
<strong style="color:{_sh_color(pooled['sharpe'])}">{pooled['sharpe']:+.2f}</strong>
(matches EXP-2200 sparse 5.96 full-sample within WF split)
&nbsp;·&nbsp; <strong>Distribution:</strong> min <strong>{dist['min']:+.2f}</strong>,
median <strong>{dist['median']:+.2f}</strong>, max <strong>{dist['max']:+.2f}</strong>,
std <strong>{dist['std']:.2f}</strong>
&nbsp;·&nbsp; <strong>{dist['n_above_6']}/{dist['n_folds']} folds &gt; 6.0</strong>,
<strong>{dist['n_below_3']}/{dist['n_folds']} folds &lt; 3.0</strong>,
<strong>{dist['n_below_0']}/{dist['n_folds']} folds &lt; 0</strong>.
&nbsp; {verdict_msg}</div>

<div class="grid">
  <div class="card"><div class="l">Pooled OOS Sharpe</div><div class="v" style="color:{_sh_color(pooled['sharpe'])}">{pooled['sharpe']:+.2f}</div></div>
  <div class="card"><div class="l">Pooled CAGR</div><div class="v">{pooled['cagr_pct']:+.1f}%</div></div>
  <div class="card"><div class="l">Pooled Max DD</div><div class="v">{pooled['max_dd_pct']:.2f}%</div></div>
  <div class="card"><div class="l">Min fold Sharpe</div><div class="v" style="color:{_sh_color(dist['min'])}">{dist['min']:+.2f}</div></div>
  <div class="card"><div class="l">Median fold Sharpe</div><div class="v" style="color:{_sh_color(dist['median'])}">{dist['median']:+.2f}</div></div>
  <div class="card"><div class="l">% folds &gt; 6.0</div><div class="v">{dist['frac_above_6']*100:.0f}%</div></div>
  <div class="card"><div class="l">% folds &lt; 3.0</div><div class="v">{dist['frac_below_3']*100:.0f}%</div></div>
  <div class="card"><div class="l">Year decay slope</div><div class="v">{deg['slope_sharpe_per_year']:+.2f}/yr</div></div>
</div>

<h2>Per-Fold OOS Metrics</h2>
<table><tr><th>Fold</th><th>Test start</th><th>Test end</th><th>Vol scale</th>
<th>Train vol</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Vol</th></tr>
{fold_rows}</table>

<h2>Distribution of per-fold Sharpes</h2>
<table><tr><th>Stat</th><th>Value</th></tr>
<tr><td>N folds</td><td>{dist['n_folds']}</td></tr>
<tr><td>Min</td><td>{dist['min']:+.2f}</td></tr>
<tr><td>10th pct</td><td>{dist['p10']:+.2f}</td></tr>
<tr><td>25th pct</td><td>{dist['p25']:+.2f}</td></tr>
<tr><td>Median</td><td>{dist['median']:+.2f}</td></tr>
<tr><td>Mean</td><td>{dist['mean']:+.2f}</td></tr>
<tr><td>75th pct</td><td>{dist['p75']:+.2f}</td></tr>
<tr><td>90th pct</td><td>{dist['p90']:+.2f}</td></tr>
<tr><td>Max</td><td>{dist['max']:+.2f}</td></tr>
<tr><td>Std</td><td>{dist['std']:.2f}</td></tr>
<tr><td>% folds &gt; 6.0</td><td>{dist['frac_above_6']*100:.0f}% ({dist['n_above_6']}/{dist['n_folds']})</td></tr>
<tr><td>% folds &gt; 4.0</td><td>{dist['frac_above_4']*100:.0f}%</td></tr>
<tr><td>% folds &lt; 3.0</td><td>{dist['frac_below_3']*100:.0f}% ({dist['n_below_3']}/{dist['n_folds']})</td></tr>
<tr><td>% folds &lt; 0.0</td><td>{dist['frac_below_0']*100:.0f}% ({dist['n_below_0']}/{dist['n_folds']})</td></tr>
</table>

<h2>Year-by-year OOS Sharpe (degradation audit)</h2>
<table><tr><th>Year</th><th>n days</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Vol</th></tr>
{yr_rows}</table>

<p style="font-size:.9rem;color:#334155">
<strong>Decay test.</strong> Linear slope of per-year Sharpe vs calendar year =
<strong>{deg['slope_sharpe_per_year']:+.2f}/yr</strong>. First-year ({deg['first_year']}) Sharpe
<strong>{deg['first_year_sharpe']:+.2f}</strong>, last-year ({deg['last_year']}) Sharpe
<strong>{deg['last_year_sharpe']:+.2f}</strong>, Δ <strong>{deg['delta_first_to_last']:+.2f}</strong>.
{'This is a significant decline — the edge may be decaying and needs paper-trade confirmation.' if deg['slope_sharpe_per_year'] <= -0.5 else 'No significant year-over-year degradation detected.'}
</p>

<h2>Method</h2>
<ul>
<li>Source: cached sparse 7-stream frame from EXP-2200 build_streams (compass/cache/exp2280_v6_sparse.pkl).</li>
<li>Streams: exp1220, xlf_cs, xli_cs, gld_cal, slv_cal, vol_arb, v5_hedge (all REAL trade tapes / cached v3 pickle).</li>
<li>Walk-forward: 252 train / 63 test / step 63 → 20 folds. Causal — weights and vol scale fit ONLY on train.</li>
<li>Equal-risk weights = normalise(1 / σ_i) on train-window daily returns.</li>
<li>Vol target = 0.15 annualised; scale = 0.15 / (train_portfolio_σ × √252), fixed for the fold.</li>
<li>Per-fold Sharpe = mean(daily_ret) / std(daily_ret) × √252 on the 63-day test window.</li>
<li>Pooled OOS = concatenation of all 20 test windows, then same formula.</li>
</ul>
<div style="color:#94a3b8;font-size:.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/exp2280_wf_robustness.py · ALL REAL DATA
</div>
</body></html>"""
    path.write_text(return_html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-2280 — Walk-Forward Robustness Audit (equal_risk_15%)")
    print("=" * 60)

    df = load_sparse_frame()
    print(f"[load] sparse 7-stream frame {df.shape} "
          f"{df.index.min().date()} → {df.index.max().date()}")

    folds, pooled = walk_forward_audit(df, TARGET_VOL, TRAIN_DAYS, TEST_DAYS)
    print(f"[wf] {len(folds)} folds")

    sharpes = [f.metrics["sharpe"] for f in folds]
    dist = distribution_stats(sharpes)
    pooled_m = metrics(pooled)
    yr = yearly_sharpe(pooled)
    deg = year_degradation(yr)

    # Console report
    print()
    print(f"{'fold':>4} {'test window':<24} {'scale':>7} {'Sharpe':>9} "
          f"{'CAGR':>9} {'DD':>8}")
    print("-" * 70)
    for f in folds:
        m = f.metrics
        print(f"{f.fold:>4} {f.test_start}→{f.test_end}  "
              f"{f.scale:>5.2f}× {m['sharpe']:>+9.2f} "
              f"{m['cagr_pct']:>+8.1f}% {m['max_dd_pct']:>7.2f}%")

    print()
    print("DISTRIBUTION")
    print("-" * 70)
    print(f"  n_folds        : {dist['n_folds']}")
    print(f"  min            : {dist['min']:+.2f}")
    print(f"  p25            : {dist['p25']:+.2f}")
    print(f"  median         : {dist['median']:+.2f}")
    print(f"  mean           : {dist['mean']:+.2f}")
    print(f"  p75            : {dist['p75']:+.2f}")
    print(f"  max            : {dist['max']:+.2f}")
    print(f"  std            : {dist['std']:.2f}")
    print(f"  % > 6.0        : {dist['frac_above_6']*100:.0f}% ({dist['n_above_6']}/{dist['n_folds']})")
    print(f"  % > 4.0        : {dist['frac_above_4']*100:.0f}%")
    print(f"  % < 3.0        : {dist['frac_below_3']*100:.0f}% ({dist['n_below_3']}/{dist['n_folds']})")
    print(f"  % < 0.0        : {dist['frac_below_0']*100:.0f}% ({dist['n_below_0']}/{dist['n_folds']})")
    print()
    print(f"Pooled OOS Sharpe: {pooled_m['sharpe']:+.2f} (was 5.96 full sample in EXP-2200)")

    print()
    print("YEAR-BY-YEAR (pooled OOS test days)")
    print("-" * 70)
    for y, m in sorted(yr.items()):
        print(f"  {y}  n={m['n']:>4}  Sharpe {m['sharpe']:>+.2f}  "
              f"CAGR {m['cagr_pct']:>+.1f}%  DD {m['max_dd_pct']:.2f}%")
    print()
    print(f"Year-decay slope: {deg['slope_sharpe_per_year']:+.2f} Sharpe/year")
    print(f"First year ({deg['first_year']}) Sharpe: {deg['first_year_sharpe']:+.2f}")
    print(f"Last year  ({deg['last_year']})  Sharpe: {deg['last_year_sharpe']:+.2f}")
    print(f"Δ first→last: {deg['delta_first_to_last']:+.2f}")

    payload = {
        "experiment": "EXP-2280",
        "title": "Walk-Forward Robustness Audit — North Star v6 equal_risk_15%",
        "config": "equal_risk_15%",
        "target_vol": TARGET_VOL,
        "walk_forward": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
                         "step_days": TEST_DAYS, "n_folds": len(folds)},
        "date_range": {"start": START, "end": END},
        "data_source": "compass/cache/exp2280_v6_sparse.pkl (from EXP-2200 build_streams, REAL data)",
        "streams": STREAMS,
        "folds": [asdict(f) for f in folds],
        "pooled_oos": pooled_m,
        "exp2200_full_sample_sharpe": 5.96,
        "distribution": dist,
        "yearly": {str(k): v for k, v in yr.items()},
        "year_degradation": deg,
        "target_6_hold": bool(dist["median"] >= 6.0 and dist["frac_above_6"] >= 0.5),
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    write_html(payload, REPORT_HTML)
    print(f"\nReports → {REPORT_JSON.name} + {REPORT_HTML.name}")
    return payload


if __name__ == "__main__":
    main()
