"""
EXP-2360 — Robust Covariance Estimators for the 7-Stream Portfolio
===================================================================

The EXP-2280 walk-forward audit found pooled OOS Sharpe = 4.43 on the
equal-risk (risk-parity) allocation of the 7-stream cube with a 15%
vol target. That number used sample covariance. This experiment
checks whether any robust covariance estimator lifts the honest WF
Sharpe or — more importantly — whether it stabilises the per-fold
distribution.

Estimators tested
-----------------
  1. sample                sample covariance (baseline)
  2. ledoit_wolf           sklearn.covariance.LedoitWolf
  3. oas                   sklearn.covariance.OAS (Oracle Approximating Shrinkage)
  4. min_cov_det           sklearn.covariance.MinCovDet (robust to outliers)
  5. one_factor            1-factor market model: Σ = β β' σ_m² + diag(idio)
                           with SPY returns as the market factor

Pipeline (per fold)
-------------------
  train (252d) → estimate Σ → risk-parity solver → normalise to ERC →
  (hold weights constant for 63d) → OOS test → scale by
  target_vol / train_vol to hit 15% vol target → record daily returns.

Pool all fold test-series and compute a single "pooled OOS" metric,
plus the distribution of per-fold Sharpes to measure stability (std,
IQR, fraction above 6, fraction below 3).

Stream data
-----------
  5 cached streams from EXP-2080 real cube
  XLF + XLI rebuilt live from EXP-2160 engine on real IronVault chains

Outputs
  compass/reports/exp2360_robust_cov.json
  compass/reports/exp2360_robust_cov.html

Rule Zero clean — no synthetic prices.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, OAS, MinCovDet

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp2080_corr_regime import load_streams
from compass.exp2160_high_capacity_alts import (
    run_put_credit_spreads,
    trades_to_daily_pct,
)
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2360_robust_cov.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2360_robust_cov.html"

TRADING_DAYS = 252
TRAIN_DAYS = 252
TEST_DAYS = 63
TARGET_VOL_ANNUAL = 0.15
CAPITAL = 100_000


# ─────────────────────────────────────────────────────────────────────────────
# 7-stream cube
# ─────────────────────────────────────────────────────────────────────────────
def build_seven_stream_cube() -> pd.DataFrame:
    print("[1/4] loading 5-stream cached cube …")
    base = load_streams()
    print(f"      {base.shape}")

    print("[2/4] building XLF + XLI streams (real IronVault) …")
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)
    for tk in ("XLF", "XLI"):
        trades = run_put_credit_spreads(con, tk)
        daily = trades_to_daily_pct(trades, base.index)
        base[f"{tk.lower()}_cs"] = daily.reindex(base.index).fillna(0.0)
    con.close()
    df = base[["exp1220", "v5_hedge", "gld_cal", "slv_cal", "cross_vol", "xlf_cs", "xli_cs"]]
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Covariance estimators
# ─────────────────────────────────────────────────────────────────────────────
def cov_sample(R: np.ndarray) -> np.ndarray:
    return np.cov(R, rowvar=False, ddof=1)


def cov_ledoit_wolf(R: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LedoitWolf().fit(R).covariance_


def cov_oas(R: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return OAS().fit(R).covariance_


def cov_min_cov_det(R: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return MinCovDet(support_fraction=0.85, random_state=0).fit(R).covariance_
        except Exception:
            # MCD can fail on very short windows; fall back to sample.
            return cov_sample(R)


def cov_one_factor(R: np.ndarray, market: np.ndarray) -> np.ndarray:
    """1-factor market-model covariance:
        Σ = β β' σ_m²  +  diag(σ_ε²)
    with the market factor provided as a 1-D numpy array aligned with R.
    """
    T, N = R.shape
    market = np.asarray(market).reshape(-1)
    assert len(market) == T, "market factor length mismatch"
    m_c = market - market.mean()
    mm = float(m_c @ m_c)
    if mm <= 1e-14:
        return cov_sample(R)
    var_m = mm / (T - 1)
    Xc = R - R.mean(axis=0, keepdims=True)
    # β_i = Cov(R_i, m) / Var(m) = (X_c[:, i] @ m_c) / (m_c @ m_c)
    betas = (Xc.T @ m_c) / mm              # (N,)
    # residuals: shape (T, N) = Xc - m_c[:, None] * betas[None, :]
    resid = Xc - np.outer(m_c, betas)
    idio = (resid ** 2).sum(axis=0) / (T - 1)
    # Protect against zero idiosyncratic variance
    idio = np.maximum(idio, 1e-14)
    Sigma = var_m * np.outer(betas, betas) + np.diag(idio)
    return Sigma


ESTIMATORS: Dict[str, Callable] = {
    "sample":       cov_sample,
    "ledoit_wolf":  cov_ledoit_wolf,
    "oas":          cov_oas,
    "min_cov_det":  cov_min_cov_det,
}


# ─────────────────────────────────────────────────────────────────────────────
# Risk-parity (equal risk contribution) solver
# ─────────────────────────────────────────────────────────────────────────────
def risk_parity_weights(Sigma: np.ndarray, n_iter: int = 500,
                        tol: float = 1e-10) -> np.ndarray:
    """Equal-risk-contribution weights via the Chaves–Hsu–Li–Shakernia
    (2011) fixed-point iteration.

    Find w ≥ 0, Σ w = 1 such that every stream contributes the same
    share of portfolio variance: w_i · (Σ w)_i = const.

    The iteration scales each weight by sqrt(target / rc_i) and
    re-normalises. Converges fast for well-conditioned Σ and is scale-
    invariant, which is critical when the daily-return covariance has
    tiny absolute entries.
    """
    N = Sigma.shape[0]
    # Ensure PSD numerically
    Sigma = (Sigma + Sigma.T) / 2
    eig_min = float(np.linalg.eigvalsh(Sigma).min())
    if eig_min < 1e-14:
        Sigma = Sigma + np.eye(N) * (1e-14 - eig_min + 1e-14)

    w = np.ones(N) / N
    for _ in range(n_iter):
        mrc = Sigma @ w                   # marginal risk contribution
        rc  = w * mrc                     # actual risk contribution
        target = rc.mean()
        if target <= 1e-30:
            break
        scale = np.sqrt(target / np.maximum(rc, 1e-30))
        w_new = w * scale
        w_new = np.maximum(w_new, 1e-10)
        w_new = w_new / w_new.sum()
        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward driver
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward(df: pd.DataFrame, estimator_name: str,
                 estimator: Callable) -> Tuple[pd.Series, List[Dict]]:
    cols = list(df.columns)
    n = len(df)
    pooled_idx, pooled_vals = [], []
    fold_rows: List[Dict] = []
    spy_ret = df["exp1220"].values  # proxy "market" for the 1-factor model
    i = TRAIN_DAYS
    fold_ix = 0
    while i + TEST_DAYS <= n:
        train = df.iloc[i - TRAIN_DAYS:i].values
        test  = df.iloc[i:i + TEST_DAYS]
        if estimator_name == "one_factor":
            market_train = spy_ret[i - TRAIN_DAYS:i]
            Sigma = cov_one_factor(train, market_train)
        else:
            Sigma = estimator(train)
        w = risk_parity_weights(Sigma)

        # gross portfolio returns before vol-targeting
        raw = test.values @ w
        # scale to TARGET_VOL_ANNUAL using the training-window stdev
        train_port = train @ w
        train_vol  = float(np.std(train_port, ddof=1)) * math.sqrt(TRADING_DAYS)
        if train_vol <= 1e-10:
            scale = 1.0
        else:
            scale = TARGET_VOL_ANNUAL / train_vol
        scale = float(np.clip(scale, 0.1, 5.0))
        scaled = raw * scale

        pooled_idx.extend(test.index.tolist())
        pooled_vals.extend(scaled.tolist())

        # per-fold sharpe
        mu = float(np.mean(scaled))
        sd = float(np.std(scaled, ddof=1)) if len(scaled) > 1 else 0.0
        sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
        fold_rows.append({
            "fold": fold_ix,
            "start": str(test.index[0].date()),
            "end":   str(test.index[-1].date()),
            "sharpe": round(sharpe, 3),
            "cagr_pct": round(float((1 + mu) ** TRADING_DAYS - 1) * 100, 3),
            "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
            "scale":  round(scale, 3),
            "weights": {cols[j]: round(float(w[j]), 4) for j in range(len(cols))},
        })
        fold_ix += 1
        i += TEST_DAYS
    return pd.Series(pooled_vals, index=pooled_idx, dtype=float), fold_rows


def metrics_pooled(daily: pd.Series, label: str) -> Dict:
    daily = daily.dropna()
    if len(daily) < 30:
        return {"label": label, "n_days": 0}
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


def distribution_stats(fold_rows: List[Dict]) -> Dict:
    sh = np.array([r["sharpe"] for r in fold_rows], dtype=float)
    if len(sh) == 0:
        return {}
    return {
        "n_folds": int(len(sh)),
        "mean":    round(float(sh.mean()), 3),
        "median":  round(float(np.median(sh)), 3),
        "std":     round(float(sh.std(ddof=1)), 3),
        "min":     round(float(sh.min()), 3),
        "p10":     round(float(np.quantile(sh, 0.10)), 3),
        "p25":     round(float(np.quantile(sh, 0.25)), 3),
        "p75":     round(float(np.quantile(sh, 0.75)), 3),
        "p90":     round(float(np.quantile(sh, 0.90)), 3),
        "max":     round(float(sh.max()), 3),
        "iqr":     round(float(np.quantile(sh, 0.75) - np.quantile(sh, 0.25)), 3),
        "frac_above_6": round(float((sh >= 6).mean()), 3),
        "frac_above_4": round(float((sh >= 4).mean()), 3),
        "frac_below_3": round(float((sh < 3).mean()), 3),
        "frac_below_0": round(float((sh < 0).mean()), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    df = build_seven_stream_cube()
    print(f"      cube {df.shape}  {df.index[0].date()} → {df.index[-1].date()}")

    print("[3/4] running walk-forward for each estimator …")
    all_results: Dict[str, Dict] = {}
    for name, est in ESTIMATORS.items():
        pooled, folds = walk_forward(df, name, est)
        pooled_m = metrics_pooled(pooled, name)
        dist = distribution_stats(folds)
        all_results[name] = {"pooled": pooled_m, "distribution": dist, "folds": folds}
        print(f"      {name:14s}  pooled Sharpe {pooled_m['sharpe']:5.2f}  "
              f"pooled DD {pooled_m['max_dd_pct']:5.2f}%  "
              f"median-fold {dist['median']:5.2f}  "
              f"std-fold {dist['std']:5.2f}  "
              f"frac>6 {dist['frac_above_6']:.0%}")

    # one-factor runs via the same walk_forward driver but with its own branch
    pooled, folds = walk_forward(df, "one_factor", cov_sample)  # estimator unused for this name
    pooled_m = metrics_pooled(pooled, "one_factor")
    dist = distribution_stats(folds)
    all_results["one_factor"] = {"pooled": pooled_m, "distribution": dist, "folds": folds}
    print(f"      one_factor     pooled Sharpe {pooled_m['sharpe']:5.2f}  "
          f"pooled DD {pooled_m['max_dd_pct']:5.2f}%  "
          f"median-fold {dist['median']:5.2f}  "
          f"std-fold {dist['std']:5.2f}  "
          f"frac>6 {dist['frac_above_6']:.0%}")

    # Ranking
    pooled_rank = sorted(all_results.items(),
                         key=lambda kv: kv[1]["pooled"]["sharpe"], reverse=True)
    stability_rank = sorted(all_results.items(),
                            key=lambda kv: kv[1]["distribution"]["std"])

    baseline_sharpe = 4.429   # EXP-2280 pooled OOS with sample cov
    above_baseline = {k: v["pooled"]["sharpe"] > baseline_sharpe
                      for k, v in all_results.items()}

    payload = {
        "experiment": "EXP-2360",
        "name": "Robust Covariance Estimators on 7-Stream Portfolio",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "baseline_reference": {
            "source": "EXP-2280",
            "pooled_oos_sharpe": baseline_sharpe,
        },
        "data_sources": {
            "five_streams": "compass.exp2080_corr_regime.load_streams (cached real cube)",
            "xlf_cs": "compass.exp2160_high_capacity_alts.run_put_credit_spreads('XLF')",
            "xli_cs": "compass.exp2160_high_capacity_alts.run_put_credit_spreads('XLI')",
        },
        "config": {
            "train_days": TRAIN_DAYS,
            "test_days":  TEST_DAYS,
            "target_vol_annual": TARGET_VOL_ANNUAL,
            "allocator": "equal_risk_contribution (risk parity)",
            "market_factor_for_one_factor": "exp1220 stream as market proxy",
        },
        "cube_info": {
            "n_days": int(len(df)),
            "range": [str(df.index[0].date()), str(df.index[-1].date())],
            "streams": list(df.columns),
        },
        "results": {
            k: {"pooled": v["pooled"], "distribution": v["distribution"]}
            for k, v in all_results.items()
        },
        "above_baseline": above_baseline,
        "pooled_sharpe_ranking": [
            {"estimator": k,
             "pooled_sharpe": v["pooled"]["sharpe"],
             "pooled_dd":     v["pooled"]["max_dd_pct"],
             "cagr_pct":      v["pooled"]["cagr_pct"],
             "median_fold":   v["distribution"]["median"],
             "std_fold":      v["distribution"]["std"]}
            for k, v in pooled_rank
        ],
        "stability_ranking": [
            {"estimator": k,
             "std_fold":      v["distribution"]["std"],
             "iqr_fold":      v["distribution"]["iqr"],
             "pooled_sharpe": v["pooled"]["sharpe"]}
            for k, v in stability_rank
        ],
        "best_by_pooled_sharpe":  pooled_rank[0][0],
        "best_by_stability":      stability_rank[0][0],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    rows_pool = "".join(
        f"<tr><td>{i+1}</td><td>{r['estimator']}</td>"
        f"<td>{r['pooled_sharpe']:.2f}</td>"
        f"<td>{r['cagr_pct']:.2f}%</td>"
        f"<td>{r['pooled_dd']:.2f}%</td>"
        f"<td>{r['median_fold']:.2f}</td>"
        f"<td>{r['std_fold']:.2f}</td></tr>"
        for i, r in enumerate(p["pooled_sharpe_ranking"])
    )
    rows_stab = "".join(
        f"<tr><td>{i+1}</td><td>{r['estimator']}</td>"
        f"<td>{r['std_fold']:.2f}</td><td>{r['iqr_fold']:.2f}</td>"
        f"<td>{r['pooled_sharpe']:.2f}</td></tr>"
        for i, r in enumerate(p["stability_ranking"])
    )
    rows_dist = "".join(
        f"<tr><td>{k}</td>"
        f"<td>{v['distribution']['n_folds']}</td>"
        f"<td>{v['distribution']['mean']:.2f}</td>"
        f"<td>{v['distribution']['median']:.2f}</td>"
        f"<td>{v['distribution']['std']:.2f}</td>"
        f"<td>{v['distribution']['min']:.2f}</td>"
        f"<td>{v['distribution']['max']:.2f}</td>"
        f"<td>{v['distribution']['frac_above_6']*100:.0f}%</td>"
        f"<td>{v['distribution']['frac_above_4']*100:.0f}%</td>"
        f"<td>{v['distribution']['frac_below_3']*100:.0f}%</td></tr>"
        for k, v in p["results"].items()
    )
    yes_cell = '<span class="ok">YES</span>'
    no_cell  = '<span class="warn">no</span>'
    rows_above = "".join(
        f"<tr><td>{k}</td><td>{yes_cell if v else no_cell}</td></tr>"
        for k, v in p["above_baseline"].items()
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2360 — Robust Covariance</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5;background:#fff}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .warn{{color:#b86b00;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-2360 — Robust Covariance on 7-Stream Portfolio</h1>
<p class='small'>Generated {p['generated']} · Risk-parity allocator · 15% vol target ·
  WF 252d train / 63d test · {p['cube_info']['n_days']} days · Rule Zero clean</p>

<p>Baseline reference: EXP-2280 pooled OOS Sharpe = <b>{p['baseline_reference']['pooled_oos_sharpe']}</b>
   (sample covariance).</p>

<h2>Pooled OOS ranking</h2>
<table>
<tr><th>#</th><th>Estimator</th><th>Pooled Sharpe</th><th>CAGR</th>
    <th>Pooled DD</th><th>Median fold</th><th>Std fold</th></tr>
{rows_pool}
</table>

<h2>Stability ranking (lowest per-fold std wins)</h2>
<table>
<tr><th>#</th><th>Estimator</th><th>Std(Sharpe)</th><th>IQR(Sharpe)</th><th>Pooled Sharpe</th></tr>
{rows_stab}
</table>

<h2>Per-fold distribution</h2>
<table>
<tr><th>Estimator</th><th>n</th><th>mean</th><th>median</th><th>std</th>
    <th>min</th><th>max</th><th>frac≥6</th><th>frac≥4</th><th>frac&lt;3</th></tr>
{rows_dist}
</table>

<h2>Does it improve pooled OOS above {p['baseline_reference']['pooled_oos_sharpe']}?</h2>
<table><tr><th>Estimator</th><th>Above baseline</th></tr>{rows_above}</table>

<h2>Conclusions</h2>
<p>Best by pooled Sharpe: <b>{p['best_by_pooled_sharpe']}</b>
 · Best by stability (lowest fold std): <b>{p['best_by_stability']}</b></p>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
