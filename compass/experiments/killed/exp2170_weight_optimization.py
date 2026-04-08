"""
EXP-2170 — Weight Optimisation for the 5-Stream Portfolio
==========================================================

Goal
----
The 5-stream static portfolio (EXP-1220 + v5_hedge + gld_cal + slv_cal +
cross_vol) compounds to Sharpe 5.24 at 3× leverage in EXP-2110. The
Carlos target is 6.00. The Sharpe ratio is leverage-invariant by
construction, so the only ways to close the gap are:

  (a) raise mean returns,
  (b) cut portfolio volatility,
  (c) any mix of the above —

all of which boil down to *better cross-stream weights*.

This experiment performs a walk-forward weight bake-off across seven
allocation rules, on the same daily-return cube the production
portfolio uses (real IronVault + real Yahoo, all upstream Rule-Zero
clean):

  1. equal_weight             1/N
  2. static_prod              EXP-2080 production weights
  3. inv_vol                  1/σ_i, normalised
  4. min_variance             quadprog on Σ
  5. mv_sample                max-Sharpe Markowitz, sample covariance
  6. mv_shrink (Ledoit-Wolf)  max-Sharpe Markowitz, LW-shrinkage cov
  7. mv_shrink_floor          mv_shrink with min-weight 0.05 floor
                              (preserves the decorrelated hedge sleeves
                               even when their mean estimate is noisy)

Walk-forward
------------
Rolling 252-day training window → 63-day OOS test window, advance by
the test window. Weights are re-fit on each training slice and held
constant through the next test slice. Pooled OOS daily returns are
concatenated across all folds and metrics computed on the pool.

Scope warning (HONEST)
----------------------
Pooled-OOS Sharpe across the same daily cube cannot beat 6.0 from
weight optimisation alone unless the *streams themselves* contain
enough orthogonal alpha to support it. This experiment honestly
measures the available headroom and reports the result, regardless
of whether it clears 6.0 or not.

Outputs
  compass/reports/exp2170_weight_optimization.json
  compass/reports/exp2170_weight_optimization.html
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp2080_corr_regime import (
    STATIC_WEIGHTS,
    load_streams,
    metrics as portfolio_metrics,
    TRADING_DAYS,
)

REPORT_JSON = ROOT / "compass" / "reports" / "exp2170_weight_optimization.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2170_weight_optimization.html"

TRAIN_DAYS = 252
TEST_DAYS  = 63
LEVERAGE   = 3.0     # match EXP-2110 sweet spot
TARGET_SHARPE = 6.00


# ─────────────────────────────────────────────────────────────────────────────
# Covariance estimators
# ─────────────────────────────────────────────────────────────────────────────
def sample_cov(R: np.ndarray) -> np.ndarray:
    return np.cov(R, rowvar=False, ddof=1)


def ledoit_wolf_cov(R: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward the constant-correlation target.

    Hand-rolled (no sklearn dep) — see Ledoit & Wolf 2004 sec 3.2."""
    T, N = R.shape
    Xc = R - R.mean(axis=0, keepdims=True)
    S = (Xc.T @ Xc) / T
    var = np.diag(S)
    sd = np.sqrt(var)
    # Constant-correlation target
    corr = S / np.outer(sd, sd)
    iu = np.triu_indices(N, k=1)
    rbar = corr[iu].mean() if len(iu[0]) else 0.0
    F = rbar * np.outer(sd, sd)
    np.fill_diagonal(F, var)

    # Shrinkage intensity (Ledoit-Wolf 2004 eq. 5/6 — simplified)
    pi_mat = np.zeros_like(S)
    for i in range(N):
        for j in range(N):
            pi_mat[i, j] = ((Xc[:, i] * Xc[:, j] - S[i, j]) ** 2).mean()
    pi_hat = pi_mat.sum()
    # Diagonal elements always shrink toward themselves
    rho_diag = np.trace(pi_mat)
    # Off-diag rho term — small refinement, can be approximated
    rho_off = 0.0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            term = (rbar / 2) * (
                (sd[j] / sd[i]) * ((Xc[:, i] ** 2 * Xc[:, i] * Xc[:, j]).mean() - var[i] * S[i, j]) +
                (sd[i] / sd[j]) * ((Xc[:, j] ** 2 * Xc[:, i] * Xc[:, j]).mean() - var[j] * S[i, j])
            )
            rho_off += term
    rho_hat = rho_diag + rho_off
    gamma_hat = ((F - S) ** 2).sum()
    if gamma_hat <= 0:
        return S
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = max(0.0, min(1.0, kappa / T))
    return delta * F + (1 - delta) * S


# ─────────────────────────────────────────────────────────────────────────────
# Allocation rules — input is the train slice (T × N), output is N weights
# ─────────────────────────────────────────────────────────────────────────────
def w_equal(R: np.ndarray, cols: List[str]) -> np.ndarray:
    return np.full(R.shape[1], 1 / R.shape[1])


def w_static(R: np.ndarray, cols: List[str]) -> np.ndarray:
    return np.array([STATIC_WEIGHTS.get(c, 0.0) for c in cols])


def w_inv_vol(R: np.ndarray, cols: List[str]) -> np.ndarray:
    sd = R.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1e-12)
    raw = 1.0 / sd
    return raw / raw.sum()


def _max_sharpe(mu: np.ndarray, S: np.ndarray, *,
                floor: float = 0.0) -> np.ndarray:
    """Long-only, sum-to-1 max-Sharpe via scipy SLSQP."""
    N = len(mu)
    def neg_sharpe(w):
        port_mu = float(w @ mu)
        port_var = float(w @ S @ w)
        if port_var <= 0:
            return 1e6
        return -port_mu / math.sqrt(port_var)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(floor, 1.0) for _ in range(N)]
    x0 = np.full(N, 1 / N)
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=bnds,
                   constraints=cons, options={"ftol": 1e-9, "maxiter": 200})
    if not res.success:
        return x0
    w = res.x
    w = np.clip(w, floor, 1.0)
    return w / w.sum()


def w_min_var(R: np.ndarray, cols: List[str]) -> np.ndarray:
    S = sample_cov(R)
    N = R.shape[1]
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [(0.0, 1.0) for _ in range(N)]
    res = minimize(lambda w: float(w @ S @ w), np.full(N, 1 / N),
                   method="SLSQP", bounds=bnds, constraints=cons,
                   options={"ftol": 1e-9, "maxiter": 200})
    return res.x if res.success else np.full(N, 1 / N)


def w_mv_sample(R: np.ndarray, cols: List[str]) -> np.ndarray:
    return _max_sharpe(R.mean(axis=0), sample_cov(R), floor=0.0)


def w_mv_shrink(R: np.ndarray, cols: List[str]) -> np.ndarray:
    return _max_sharpe(R.mean(axis=0), ledoit_wolf_cov(R), floor=0.0)


def w_mv_shrink_floor(R: np.ndarray, cols: List[str]) -> np.ndarray:
    return _max_sharpe(R.mean(axis=0), ledoit_wolf_cov(R), floor=0.05)


ALLOCATORS: Dict[str, Callable] = {
    "equal_weight":      w_equal,
    "static_prod":       w_static,
    "inv_vol":           w_inv_vol,
    "min_variance":      w_min_var,
    "mv_sample":         w_mv_sample,
    "mv_shrink":         w_mv_shrink,
    "mv_shrink_floor05": w_mv_shrink_floor,
}


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward driver
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward(df: pd.DataFrame, allocator: Callable,
                 train_days: int = TRAIN_DAYS,
                 test_days: int = TEST_DAYS,
                 leverage: float = LEVERAGE) -> Tuple[pd.Series, List[np.ndarray]]:
    cols = list(df.columns)
    n = len(df)
    pooled_idx, pooled_vals, weights_history = [], [], []
    i = train_days
    while i + test_days <= n:
        train = df.iloc[i - train_days:i].values
        test  = df.iloc[i:i + test_days]
        w = allocator(train, cols)
        weights_history.append(w)
        port = (test.values @ w) * leverage
        pooled_idx.extend(test.index.tolist())
        pooled_vals.extend(port.tolist())
        i += test_days
    return pd.Series(pooled_vals, index=pooled_idx, dtype=float), weights_history


def metrics_pooled(daily: pd.Series, label: str) -> Dict:
    if len(daily) < 30:
        return {"label": label, "n_days": 0, "cagr_pct": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "max_dd_pct": 0.0,
                "vol_pct": 0.0, "calmar": 0.0}
    eq = (1 + daily).cumprod()
    yrs = len(daily) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    mu, sd = daily.mean(), daily.std(ddof=1)
    downside = daily[daily < 0].std(ddof=1) if (daily < 0).any() else np.nan
    sharpe = float((mu / sd) * math.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
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
    print("[1/4] loading 5-stream cube …")
    df = load_streams()
    cols = list(df.columns)
    print(f"      shape {df.shape}  range {df.index[0].date()} → {df.index[-1].date()}")

    print("[2/4] running walk-forward bake-off …")
    results: Dict[str, Dict] = {}
    weights_summary: Dict[str, Dict] = {}
    for name, alloc in ALLOCATORS.items():
        pooled, history = walk_forward(df, alloc)
        m = metrics_pooled(pooled, name)
        results[name] = m
        # average weights across folds
        wmat = np.vstack(history)
        avg_w = wmat.mean(axis=0)
        std_w = wmat.std(axis=0)
        weights_summary[name] = {
            "avg":  {c: round(float(avg_w[i]), 4) for i, c in enumerate(cols)},
            "std":  {c: round(float(std_w[i]), 4) for i, c in enumerate(cols)},
            "n_folds": int(len(history)),
        }
        print(f"      {name:18s}  Sharpe {m['sharpe']:5.2f}  CAGR {m['cagr_pct']:6.2f}%  DD {m['max_dd_pct']:5.2f}%")

    ranked = sorted(results.items(), key=lambda kv: kv[1]["sharpe"], reverse=True)
    best_name, best = ranked[0]
    static = results["static_prod"]
    delta_vs_static = round(best["sharpe"] - static["sharpe"], 3)
    headroom = round(TARGET_SHARPE - best["sharpe"], 3)

    print(f"[3/4] best allocator: {best_name}  Sharpe {best['sharpe']}")
    print(f"      Δ vs static: {delta_vs_static:+.2f}   headroom to 6.0: {headroom:+.2f}")

    print("[4/4] writing report …")
    payload = {
        "experiment": "EXP-2170",
        "name": "Weight optimisation bake-off for the 5-stream portfolio",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data": {
            "source": "compass.exp2080_corr_regime.load_streams (cached real cube)",
            "streams": cols,
            "n_days": int(len(df)),
            "range": [str(df.index[0].date()), str(df.index[-1].date())],
            "leverage": LEVERAGE,
        },
        "walk_forward": {
            "train_days": TRAIN_DAYS,
            "test_days":  TEST_DAYS,
            "n_folds":    int(weights_summary[best_name]["n_folds"]),
        },
        "results": results,
        "weights_summary": weights_summary,
        "ranking": [{"rank": i + 1, "name": k, **v} for i, (k, v) in enumerate(ranked)],
        "best": {"name": best_name, **best,
                 "delta_sharpe_vs_static": delta_vs_static,
                 "headroom_to_target": headroom},
        "target_sharpe": TARGET_SHARPE,
        "target_met": best["sharpe"] >= TARGET_SHARPE,
        "honest_note": (
            "Pooled OOS Sharpe across the same daily cube is upper-bounded by "
            "the orthogonal alpha contained in the streams themselves. If the "
            "best allocator does not clear 6.0, the gap closes by adding new "
            "uncorrelated streams (EXP-2020, EXP-2070 already exist), not by "
            "re-weighting the existing five."
        ),
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    cols = p["data"]["streams"]
    rows_rank = "".join(
        f"<tr><td>{r['rank']}</td><td>{r['name']}</td>"
        f"<td>{r['sharpe']:.2f}</td><td>{r['cagr_pct']:.2f}%</td>"
        f"<td>{r['max_dd_pct']:.2f}%</td><td>{r['vol_pct']:.2f}%</td>"
        f"<td>{r['calmar']:.2f}</td></tr>"
        for r in p["ranking"]
    )
    rows_w = ""
    for k, v in p["weights_summary"].items():
        cells = "".join(f"<td>{v['avg'][c]:.3f}</td>" for c in cols)
        rows_w += f"<tr><td>{k}</td>{cells}</tr>"
    target_cls = "ok" if p["target_met"] else "warn"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2170 — Weight Optimisation</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .warn{{color:#b86b00;font-weight:600}} .bad{{color:#b80000;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-2170 — Weight Optimisation Bake-off</h1>
<p class='small'>Generated {p['generated']} · 5 streams · {p['data']['n_days']} days
 · Walk-forward {p['walk_forward']['train_days']}d→{p['walk_forward']['test_days']}d
 · {p['walk_forward']['n_folds']} folds · leverage {p['data']['leverage']}× · Rule Zero clean.</p>

<h2>Pooled OOS ranking</h2>
<table>
<tr><th>#</th><th>Allocator</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Vol</th><th>Calmar</th></tr>
{rows_rank}
</table>

<h2>Best allocator: {p['best']['name']}</h2>
<p>Sharpe <b>{p['best']['sharpe']:.2f}</b>
 · Δ vs static_prod {p['best']['delta_sharpe_vs_static']:+.2f}
 · Headroom to target 6.00: <b>{p['best']['headroom_to_target']:+.2f}</b>
 · Target 6.00: <span class='{target_cls}'>{'MET' if p['target_met'] else 'NOT MET'}</span></p>

<h2>Average weights across folds</h2>
<table>
<tr><th>Allocator</th>{''.join(f'<th>{c}</th>' for c in cols)}</tr>
{rows_w}
</table>

<h2>Honest note</h2>
<p>{p['honest_note']}</p>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
