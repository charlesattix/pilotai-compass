"""
EXP-2370 — Portfolio DD Circuit Breaker for Walk-Forward

Problem statement (from EXP-2280)
---------------------------------
The pooled 20-fold walk-forward OOS drawdown on the North Star v6
equal_risk_15% portfolio is **24.4%** — far worse than the 5.7%
full-sample figure and above the 12% North Star ceiling. The
single worst fold is the 2021-12 → 2022-03 inflation shock window,
inside which intra-fold DD ran to 24%.

Hypothesis
----------
A CAUSAL trailing-DD circuit breaker on the portfolio series can
cap pooled OOS DD under 12% without destroying Sharpe. The circuit
watches a 20-day trailing portfolio drawdown; when it crosses a
threshold, it cuts leverage to a reduced level until the trailing
DD recovers below the threshold.

Two de-levering modes tested
----------------------------
  FLATTEN    : cut leverage to 0× (flat cash) while tripped
  REDUCE     : cut leverage to 1× (base) while tripped

Four thresholds: 3%, 5%, 8%, 10%.  Total: 8 configs × 20 folds.

Causality
---------
The circuit reads only *past* portfolio returns. On day t the
trailing-DD window is days [t-20 .. t-1]; scaling applied on day t.
Trigger on day t uses information available at the end of day t-1.

Method
------
1. Load the cached 7-stream sparse frame (EXP-2280 cache).
2. Walk-forward 252-day train / 63-day test, step 63 → 20 folds
   (same shape as EXP-2080 / EXP-2280).
3. Per fold (causal):
     a. Fit equal_risk (inverse-vol) weights on training window.
     b. Compute training-window realised vol; set scale = 0.15 / σ_train.
     c. Compose raw scaled OOS portfolio on test window.
     d. Run the causal trailing-DD circuit on the OOS series.
4. Aggregate per-fold and pooled OOS metrics for every config.

Success criterion
-----------------
Pooled OOS max DD < 12% AND pooled OOS Sharpe ≥ 4.0.

REAL DATA ONLY — uses EXP-2200 build_streams sparse 7-stream frame.

Outputs
-------
  compass/exp2370_dd_circuit_breaker.py
  compass/reports/exp2370_dd_circuit_breaker.json
  compass/reports/exp2370_dd_circuit_breaker.html
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_FILE  = ROOT / "compass" / "cache" / "exp2280_v6_sparse.pkl"
REPORT_JSON = ROOT / "compass" / "reports" / "exp2370_dd_circuit_breaker.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2370_dd_circuit_breaker.html"

TRADING_DAYS = 252
START       = "2020-01-01"
END         = "2025-12-31"
TARGET_VOL  = 0.15
TRAIN_DAYS  = 252
TEST_DAYS   = 63
DD_WINDOW   = 20              # trailing days for DD watch

STREAMS = ["exp1220", "xlf_cs", "xli_cs", "gld_cal",
           "slv_cal", "vol_arb", "v5_hedge"]

THRESHOLDS = [0.03, 0.05, 0.08, 0.10]
MODES = ["flatten", "reduce"]   # flatten=0x, reduce=1x


# ───────────────────────────────────────────────────────────────────────────
# Data
# ───────────────────────────────────────────────────────────────────────────

def load_sparse_frame() -> pd.DataFrame:
    if CACHE_FILE.exists():
        return pickle.load(open(CACHE_FILE, "rb"))
    from compass.exp2200_north_star_v6 import build_streams
    _, sparse, _ = build_streams()
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(sparse, open(CACHE_FILE, "wb"))
    return sparse


# ───────────────────────────────────────────────────────────────────────────
# Weights + base portfolio
# ───────────────────────────────────────────────────────────────────────────

def equal_risk_weights(train: pd.DataFrame) -> Dict[str, float]:
    vols = np.array([train[k].std(ddof=1) + 1e-12 for k in STREAMS])
    w = 1.0 / vols
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 1e-9 else np.ones_like(w) / len(w)
    return {k: float(v) for k, v in zip(STREAMS, w)}


def compose(streams: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    out = pd.Series(0.0, index=streams.index)
    for k in STREAMS:
        out = out + weights.get(k, 0.0) * streams[k]
    return out


def vol_scale_from_train(train_port: pd.Series, target_vol: float) -> float:
    rv = float(train_port.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    return target_vol / rv if rv > 1e-9 else 1.0


# ───────────────────────────────────────────────────────────────────────────
# Circuit breaker (causal)
# ───────────────────────────────────────────────────────────────────────────

def apply_circuit_breaker(raw: pd.Series,
                          threshold: float,
                          mode: str,
                          window: int = DD_WINDOW,
                          base_leverage: float = 1.0
                          ) -> Tuple[pd.Series, pd.Series]:
    """Run the causal trailing-DD circuit.

    On day t:
      1. Compute equity curve over days [t-window .. t-1] from raw returns.
      2. Trailing DD = max-to-trough over that window.
      3. If trailing DD > threshold → leverage set to 0× (flatten) or 1× (reduce).
         Else → leverage set to `base_leverage` (the vol-targeted scale).
      4. Today's realised return = leverage_t × raw_t.

    The circuit uses ONLY returns through t-1 — fully causal.

    Returns (levered_series, leverage_path).
    """
    raw_vals = raw.values.astype(float)
    n = len(raw_vals)
    lev_path = np.full(n, base_leverage, dtype=float)
    out_vals = np.zeros(n, dtype=float)

    # Floor leverage while tripped
    trip_lev = 0.0 if mode == "flatten" else 1.0

    # Rolling equity curve from t-1 back `window` days
    for t in range(n):
        start = max(0, t - window)
        if start >= t - 1:
            # Insufficient history — use base leverage
            lev = base_leverage
        else:
            window_rets = raw_vals[start:t]          # days [t-window .. t-1]
            # Local equity curve (reset to 1.0 at window start)
            eq = np.cumprod(1.0 + window_rets)
            peak = np.maximum.accumulate(eq)
            dd = (peak - eq) / peak               # array of drawdowns
            trailing_dd = float(dd.max())
            lev = trip_lev if trailing_dd > threshold else base_leverage
        lev_path[t] = lev
        out_vals[t] = lev * raw_vals[t]

    return (pd.Series(out_vals, index=raw.index),
            pd.Series(lev_path, index=raw.index))


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
    sharpe  = mu / sigma * math.sqrt(TRADING_DAYS) if sigma > 1e-12 else 0.0
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
# Walk-forward with circuit breaker
# ───────────────────────────────────────────────────────────────────────────

def walk_forward_with_circuit(df: pd.DataFrame,
                              threshold: Optional[float],
                              mode: Optional[str]) -> Tuple[List[Dict], pd.Series, pd.Series]:
    """Run the 20-fold WF. If threshold is None → baseline (no circuit).

    Returns (fold_results, pooled_series, pooled_lev).
    """
    folds: List[Dict] = []
    pooled: List[pd.Series] = []
    pooled_lev: List[pd.Series] = []

    n = len(df)
    i = TRAIN_DAYS
    idx = 0
    while i + TEST_DAYS <= n:
        tr = df.iloc[i - TRAIN_DAYS:i]
        te = df.iloc[i:i + TEST_DAYS]

        w = equal_risk_weights(tr)
        tr_port = compose(tr, w)
        scale = vol_scale_from_train(tr_port, TARGET_VOL)

        # Raw OOS scaled portfolio (pre-circuit)
        raw_oos = compose(te, w) * scale

        if threshold is None:
            levered, lev = raw_oos, pd.Series(1.0, index=raw_oos.index)
        else:
            levered, lev = apply_circuit_breaker(
                raw_oos, threshold=threshold, mode=mode,
                window=DD_WINDOW, base_leverage=1.0
            )

        folds.append({
            "fold": idx,
            "test_start": str(te.index[0].date()),
            "test_end":   str(te.index[-1].date()),
            "metrics":    metrics(levered),
            "trip_count": int((lev < 1.0 - 1e-9).sum()) if threshold is not None else 0,
            "trip_pct":   round(float((lev < 1.0 - 1e-9).mean() * 100), 2)
                          if threshold is not None else 0.0,
        })
        pooled.append(levered)
        pooled_lev.append(lev)
        i += TEST_DAYS
        idx += 1

    pooled_series = pd.concat(pooled).sort_index()
    pooled_lev_series = pd.concat(pooled_lev).sort_index()
    return folds, pooled_series, pooled_lev_series


# ───────────────────────────────────────────────────────────────────────────
# HTML report
# ───────────────────────────────────────────────────────────────────────────

def write_html(payload: Dict, path: Path) -> None:
    base = payload["baseline"]["pooled"]
    configs = payload["configs"]

    # Find best config
    # Success criterion: DD < 12% AND Sharpe highest
    best = None
    for c in configs:
        p = c["pooled"]
        if p["max_dd_pct"] < 12.0:
            if best is None or p["sharpe"] > best["pooled"]["sharpe"]:
                best = c
    if best is None:
        best = min(configs, key=lambda c: c["pooled"]["max_dd_pct"])

    verdict_ok = (best["pooled"]["max_dd_pct"] < 12.0
                  and best["pooled"]["sharpe"] >= 4.0)
    color = "#16a34a" if verdict_ok else "#ca8a04"
    msg = ("✅ TARGET MET: pooled DD < 12% and Sharpe ≥ 4.0"
           if verdict_ok else
           "⚠ Best config: DD and Sharpe tradeoff — see table")

    # Config comparison table
    cfg_rows = ""
    for c in configs:
        p = c["pooled"]
        is_best = (c["label"] == best["label"])
        marker = " ★" if is_best else ""
        sh_color = "#16a34a" if p["sharpe"] >= 4.0 else "#ca8a04"
        dd_color = "#16a34a" if p["max_dd_pct"] < 12.0 else "#dc2626"
        cfg_rows += (
            f"<tr{'  style=background:#f0fdf4' if is_best else ''}>"
            f"<td><strong>{c['label']}{marker}</strong></td>"
            f"<td>{c['threshold']*100:.0f}%</td>"
            f"<td>{c['mode']}</td>"
            f"<td>{p['n']}</td>"
            f"<td style='color:{sh_color};font-weight:700'>{p['sharpe']:+.2f}</td>"
            f"<td>{p['cagr_pct']:+.1f}%</td>"
            f"<td style='color:{dd_color};font-weight:700'>{p['max_dd_pct']:.2f}%</td>"
            f"<td>{p['vol_pct']:.2f}%</td>"
            f"<td>{c['trip_pct_avg']:.1f}%</td></tr>"
        )
    base_row = (f"<tr style='background:#f8fafc'>"
                f"<td><strong>baseline (no circuit)</strong></td>"
                f"<td>—</td><td>—</td>"
                f"<td>{base['n']}</td>"
                f"<td>{base['sharpe']:+.2f}</td>"
                f"<td>{base['cagr_pct']:+.1f}%</td>"
                f"<td style='color:#dc2626;font-weight:700'>{base['max_dd_pct']:.2f}%</td>"
                f"<td>{base['vol_pct']:.2f}%</td><td>0.0%</td></tr>")

    # Best config fold-by-fold detail
    best_fold_rows = ""
    for f in best["folds"]:
        m = f["metrics"]
        best_fold_rows += (
            f"<tr><td>{f['fold']}</td><td>{f['test_start']}</td><td>{f['test_end']}</td>"
            f"<td>{m['sharpe']:+.2f}</td><td>{m['cagr_pct']:+.1f}%</td>"
            f"<td>{m['max_dd_pct']:.2f}%</td><td>{m['vol_pct']:.2f}%</td>"
            f"<td>{f['trip_pct']:.0f}%</td></tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>EXP-2370 DD Circuit Breaker</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1100px}}
h1{{font-size:1.5rem;color:#0f172a}} h2{{font-size:1.05rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:18px}}
.headline{{background:#f0fdf4;border-left:5px solid {color};padding:14px 18px;border-radius:6px;margin:14px 0;font-size:0.95rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.15rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.83rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}} td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}} td:first-child{{text-align:left}}
</style></head><body>
<h1>EXP-2370 — Portfolio DD Circuit Breaker</h1>
<p class="meta">20-fold walk-forward on sparse 7-stream frame · equal_risk + 15% vol target
· causal 20-day trailing DD circuit · REAL DATA only</p>

<div class="headline">
<strong>Best config:</strong> {best['label']} →
pooled Sharpe <strong>{best['pooled']['sharpe']:+.2f}</strong>
· pooled CAGR <strong>{best['pooled']['cagr_pct']:+.1f}%</strong>
· pooled Max DD <strong>{best['pooled']['max_dd_pct']:.2f}%</strong>
· circuit tripped <strong>{best['trip_pct_avg']:.1f}%</strong> of OOS days.
&nbsp;({msg})</div>

<div class="grid">
  <div class="card"><div class="l">Baseline DD</div><div class="v" style="color:#dc2626">{base['max_dd_pct']:.1f}%</div></div>
  <div class="card"><div class="l">Best DD</div><div class="v">{best['pooled']['max_dd_pct']:.1f}%</div></div>
  <div class="card"><div class="l">Baseline Sharpe</div><div class="v">{base['sharpe']:+.2f}</div></div>
  <div class="card"><div class="l">Best Sharpe</div><div class="v">{best['pooled']['sharpe']:+.2f}</div></div>
  <div class="card"><div class="l">Baseline CAGR</div><div class="v">{base['cagr_pct']:+.1f}%</div></div>
  <div class="card"><div class="l">Best CAGR</div><div class="v">{best['pooled']['cagr_pct']:+.1f}%</div></div>
  <div class="card"><div class="l">Trip rate (best)</div><div class="v">{best['trip_pct_avg']:.1f}%</div></div>
</div>

<h2>All circuit breaker configurations (pooled OOS)</h2>
<p class="meta">Same WF shape as EXP-2280 · target: DD &lt; 12% AND Sharpe ≥ 4.0</p>
<table><tr><th>Config</th><th>Threshold</th><th>Mode</th><th>n</th>
<th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Vol</th><th>Trip %</th></tr>
{base_row}
{cfg_rows}</table>

<h2>Best config — per-fold detail ({best['label']})</h2>
<table><tr><th>Fold</th><th>Test start</th><th>Test end</th>
<th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Vol</th><th>Trip %</th></tr>
{best_fold_rows}</table>

<h2>Method</h2>
<ul>
<li>Cached sparse 7-stream frame from EXP-2200 build_streams.</li>
<li>Walk-forward 252 train / 63 test / step 63 → 20 folds (same as EXP-2280).</li>
<li>Per fold: equal_risk weights + 15% vol target (scale = 0.15 / σ_train × √252).</li>
<li>Causal circuit: on day t, compute trailing-DD over days [t-20 .. t-1] from raw scaled
    portfolio returns. If DD &gt; threshold → leverage scaled to 0× (flatten) or 1× (reduce);
    else base 1.0× (= already vol-targeted).</li>
<li>Thresholds tested: 3%, 5%, 8%, 10% × modes flatten / reduce = 8 configs.</li>
<li>Trip rate = fraction of OOS days where the circuit was engaged.</li>
</ul>
<div style="color:#94a3b8;font-size:.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/exp2370_dd_circuit_breaker.py · ALL REAL DATA
</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-2370 — Portfolio DD Circuit Breaker")
    print("=" * 60)

    df = load_sparse_frame()
    print(f"[load] sparse {df.shape} {df.index.min().date()} → {df.index.max().date()}")

    # Baseline: no circuit
    print("\n[baseline] equal_risk + 15% vol target, NO circuit")
    base_folds, base_pooled, _ = walk_forward_with_circuit(df, None, None)
    base_m = metrics(base_pooled)
    print(f"  pooled Sharpe {base_m['sharpe']:+.2f}  "
          f"CAGR {base_m['cagr_pct']:+.1f}%  DD {base_m['max_dd_pct']:.2f}%")

    # All 8 configs
    configs_out: List[Dict] = []
    print("\n[circuit configs]")
    for mode in MODES:
        for thr in THRESHOLDS:
            label = f"{mode}_{int(thr*100)}pct"
            folds, pooled, lev = walk_forward_with_circuit(df, thr, mode)
            pm = metrics(pooled)
            trip_avg = float((lev < 1.0 - 1e-9).mean() * 100)
            configs_out.append({
                "label": label,
                "threshold": thr,
                "mode": mode,
                "pooled": pm,
                "trip_pct_avg": round(trip_avg, 3),
                "folds": folds,
            })
            print(f"  {label:<18}  Sh {pm['sharpe']:>+.2f}  "
                  f"CAGR {pm['cagr_pct']:>+7.1f}%  DD {pm['max_dd_pct']:>6.2f}%  "
                  f"trip {trip_avg:>5.1f}%")

    # Find best: DD<12% AND maximise Sharpe
    ok_cfgs = [c for c in configs_out if c["pooled"]["max_dd_pct"] < 12.0]
    if ok_cfgs:
        best = max(ok_cfgs, key=lambda c: c["pooled"]["sharpe"])
        print(f"\n[best] {best['label']} (DD<12% AND max Sharpe)")
    else:
        best = min(configs_out, key=lambda c: c["pooled"]["max_dd_pct"])
        print(f"\n[best] {best['label']} (min DD — no config breached 12% floor)")
    bp = best["pooled"]
    print(f"  Sh {bp['sharpe']:+.2f}  CAGR {bp['cagr_pct']:+.1f}%  "
          f"DD {bp['max_dd_pct']:.2f}%")

    target_met = bp["max_dd_pct"] < 12.0 and bp["sharpe"] >= 4.0
    print(f"\nTarget (DD < 12 AND Sharpe >= 4.0): {'✅ MET' if target_met else '⚠ MISS'}")

    payload = {
        "experiment": "EXP-2370",
        "title": "Portfolio DD Circuit Breaker — 20-fold Walk-Forward",
        "date_range": {"start": START, "end": END},
        "walk_forward": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
                         "step_days": TEST_DAYS, "n_folds": len(base_folds),
                         "target_vol": TARGET_VOL},
        "circuit_spec": {
            "dd_window_days": DD_WINDOW,
            "thresholds": THRESHOLDS,
            "modes": MODES,
            "causal": True,
            "base_leverage": 1.0,
        },
        "streams": STREAMS,
        "data_source": "compass/cache/exp2280_v6_sparse.pkl (REAL EXP-2200 build_streams)",
        "baseline": {
            "folds": base_folds,
            "pooled": base_m,
        },
        "configs": configs_out,
        "best_config_label": best["label"],
        "best_config_pooled": bp,
        "target_met": target_met,
        "success_criterion": "pooled_max_dd_pct < 12.0 AND pooled_sharpe >= 4.0",
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    write_html(payload, REPORT_HTML)
    print(f"\nReports → {REPORT_JSON.name} + {REPORT_HTML.name}")
    return payload


if __name__ == "__main__":
    main()
