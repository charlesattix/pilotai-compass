"""
EXP-1980 — Correlation Regime Switching / Dynamic Hedge Ratio

Hypothesis
----------
Crisis Alpha v5 has a -1.07% CAGR standalone — it is a drag in normal
markets. But its negative-beta property during stress is valuable. A
*static* 5% allocation always pays the drag whether or not stress is
present. A *dynamic* allocation, conditioned on the rolling correlation
between EXP-1220 and SPY, should:

  - Cut hedge weight to ~0% when EXP-1220's risk is idiosyncratic
    (low correlation with SPY) — saving the drag.
  - Lift hedge weight to 15-20% when EXP-1220's risk is systemic
    (high correlation with SPY) — buying real protection.

Same or better max DD vs the static-5% portfolio, but higher CAGR
because the drag is paid only when it earns its keep.

Streams loaded (REAL data, cached from EXP-1850 + EXP-1770)
-----------------------------------------------------------
  exp1220   — EXP-1220 dynamic credit spread proxy        (cached)
  v5_hedge  — Crisis Alpha v5 best frozen hedge config    (cached)
  gld_cal   — GLD calendar (ETF − GC=F front future)      (exp1770)
  slv_cal   — SLV calendar (ETF − SI=F front future)      (exp1770)
  spy       — Yahoo SPY daily returns                     (yfinance)

Walk-forward
------------
252-day train / 63-day OOS test, step 63 days. Each train fold
optimizes the three corr→weight knee points by grid search; the OOS
fold applies them.

Outputs
-------
  compass/exp1980_dynamic_hedge.py
  compass/reports/exp1980_dynamic_hedge.json
  compass/reports/exp1980_dynamic_hedge.html
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass, asdict, field
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_FILE = ROOT / "compass" / "cache" / "exp1850_streams.pkl"
REPORT_JSON = ROOT / "compass" / "reports" / "exp1980_dynamic_hedge.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp1980_dynamic_hedge.html"

TRADING_DAYS = 252
START = "2020-01-01"
END = "2025-12-31"


# ───────────────────────────────────────────────────────────────────────────
# Data loading (REAL only — cached streams + Yahoo + exp1770 calendars)
# ───────────────────────────────────────────────────────────────────────────

def load_streams() -> pd.DataFrame:
    """Combine all return streams into one daily-aligned DataFrame."""
    print("[load] cached EXP-1850 streams (exp1220 + v5_hedge)")
    if not CACHE_FILE.exists():
        from compass.exp1850_regime_portfolio import load_real_streams
        load_real_streams()
    cached = pickle.load(open(CACHE_FILE, "rb"))
    exp1220 = cached["exp1220"].rename("exp1220")
    v5      = cached["v5_hedge"].rename("v5_hedge")

    print("[load] GLD/SLV calendar streams from exp1770")
    from compass.exp1770_commodity_calendars import (
        load_pair, walk_forward, PAIRS,
    )
    gld_etf, gld_fut, _ = PAIRS["GLD"]
    slv_etf, slv_fut, _ = PAIRS["SLV"]
    gld_df = load_pair(gld_etf, gld_fut)
    slv_df = load_pair(slv_etf, slv_fut)
    gld_cal = walk_forward("GLD", gld_df).daily_returns.rename("gld_cal")
    slv_cal = walk_forward("SLV", slv_df).daily_returns.rename("slv_cal")

    print("[load] SPY daily returns from Yahoo")
    import yfinance as yf
    spy = yf.download("SPY", start="2018-01-01", end="2026-04-01",
                      progress=False, auto_adjust=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy_ret = spy["Close"].pct_change().rename("spy")

    df = pd.concat([exp1220, v5, gld_cal, slv_cal, spy_ret], axis=1)
    df = df.loc[START:END].fillna(0.0).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ───────────────────────────────────────────────────────────────────────────
# Hedge schedule
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class HedgeSchedule:
    """Maps rolling corr to v5 hedge weight via 3 knee points."""
    low_thr:    float = 0.20
    high_thr:   float = 0.50
    weight_low:  float = 0.00     # corr < low_thr
    weight_mid:  float = 0.07     # low_thr ≤ corr < high_thr
    weight_high: float = 0.18     # corr ≥ high_thr

    def weight(self, corr: float) -> float:
        if not np.isfinite(corr):
            return self.weight_mid
        if corr < self.low_thr:
            return self.weight_low
        if corr < self.high_thr:
            return self.weight_mid
        return self.weight_high

    def series(self, corr: pd.Series) -> pd.Series:
        return corr.apply(self.weight)


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 60) -> pd.Series:
    return a.rolling(window).corr(b)


# ───────────────────────────────────────────────────────────────────────────
# Portfolio synthesis
# ───────────────────────────────────────────────────────────────────────────

def synth_portfolio(df: pd.DataFrame, hedge_w: pd.Series,
                    base_w_exp1220: float = 1.0,
                    extra: Optional[Dict[str, float]] = None) -> pd.Series:
    """Daily portfolio return = base_w_exp1220·exp1220 + hedge_w·v5_hedge
    + Σ extra_k·stream_k (for GLD/SLV calendar diversifiers).
    """
    out = base_w_exp1220 * df["exp1220"] + hedge_w * df["v5_hedge"]
    if extra:
        for k, w in extra.items():
            out = out + w * df[k]
    return out


# ───────────────────────────────────────────────────────────────────────────
# Metrics
# ───────────────────────────────────────────────────────────────────────────

def metrics(daily: pd.Series) -> Dict[str, float]:
    daily = daily.dropna()
    n = len(daily)
    if n < 2:
        return {"n": 0, "cagr_pct": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "max_dd_pct": 0.0, "calmar": 0.0, "vol_pct": 0.0,
                "total_ret_pct": 0.0}
    eq = (1 + daily).cumprod()
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else -1.0
    mu = float(daily.mean()); sigma = float(daily.std(ddof=1))
    sharpe = mu / sigma * math.sqrt(TRADING_DAYS) if sigma > 1e-12 else 0.0
    down = daily[daily < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else sigma
    sortino = mu / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0.0
    hwm = eq.cummax()
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-9 else 0.0
    vol = sigma * math.sqrt(TRADING_DAYS)
    return {
        "n": n, "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "max_dd_pct": round(dd * 100, 3),
        "calmar": round(calmar, 3),
        "vol_pct": round(vol * 100, 3),
        "total_ret_pct": round(float(eq.iloc[-1] - 1) * 100, 3),
    }


# ───────────────────────────────────────────────────────────────────────────
# Walk-forward optimization
# ───────────────────────────────────────────────────────────────────────────

GRID_LOW_THR  = [0.10, 0.15, 0.20, 0.25, 0.30]
GRID_HIGH_THR = [0.40, 0.50, 0.60]
GRID_W_LOW    = [0.00, 0.02]
GRID_W_MID    = [0.05, 0.07, 0.10]
GRID_W_HIGH   = [0.15, 0.18, 0.20]


def all_schedules() -> List[HedgeSchedule]:
    out = []
    for lt, ht, wl, wm, wh in product(GRID_LOW_THR, GRID_HIGH_THR,
                                       GRID_W_LOW, GRID_W_MID, GRID_W_HIGH):
        if lt >= ht or wl >= wm or wm >= wh:
            continue
        out.append(HedgeSchedule(lt, ht, wl, wm, wh))
    return out


def fit_best_schedule(df: pd.DataFrame, corr: pd.Series) -> HedgeSchedule:
    """Pick the schedule that maximises Sharpe on the given (training) slice."""
    best, best_sh = None, -1e9
    for sch in all_schedules():
        hw = sch.series(corr)
        port = synth_portfolio(df, hw)
        sh = metrics(port)["sharpe"]
        if sh > best_sh:
            best_sh = sh; best = sch
    return best or HedgeSchedule()


def walk_forward(df: pd.DataFrame, train_days: int = 252,
                 test_days: int = 63) -> Dict:
    """Expanding/rolling WF: train fits schedule, test applies it OOS."""
    corr = rolling_corr(df["exp1220"], df["spy"], 60).fillna(0.0)
    n = len(df)
    folds = []
    test_returns_dyn:    List[pd.Series] = []
    test_returns_static: List[pd.Series] = []
    test_returns_base:   List[pd.Series] = []
    test_hedge_weights:  List[pd.Series] = []

    i = train_days
    while i + test_days <= n:
        tr_slice = df.iloc[i - train_days:i]
        te_slice = df.iloc[i:i + test_days]
        tr_corr = corr.iloc[i - train_days:i]
        te_corr = corr.iloc[i:i + test_days]

        sch = fit_best_schedule(tr_slice, tr_corr)
        hw_te = sch.series(te_corr)
        dyn = synth_portfolio(te_slice, hw_te)
        static = synth_portfolio(te_slice, pd.Series(0.05, index=te_slice.index))
        base = te_slice["exp1220"]

        folds.append({
            "train_start": str(tr_slice.index[0].date()),
            "train_end":   str(tr_slice.index[-1].date()),
            "test_start":  str(te_slice.index[0].date()),
            "test_end":    str(te_slice.index[-1].date()),
            "schedule":    asdict(sch),
            "mean_test_corr": round(float(te_corr.mean()), 3),
            "mean_test_hedge_w": round(float(hw_te.mean()), 4),
            "dynamic":  metrics(dyn),
            "static":   metrics(static),
            "baseline": metrics(base),
        })
        test_returns_dyn.append(dyn)
        test_returns_static.append(static)
        test_returns_base.append(base)
        test_hedge_weights.append(hw_te)
        i += test_days

    pooled_dyn    = pd.concat(test_returns_dyn).sort_index()
    pooled_static = pd.concat(test_returns_static).sort_index()
    pooled_base   = pd.concat(test_returns_base).sort_index()
    pooled_hw     = pd.concat(test_hedge_weights).sort_index()

    return {
        "folds": folds,
        "pooled_oos": {
            "dynamic":  metrics(pooled_dyn),
            "static_5": metrics(pooled_static),
            "baseline": metrics(pooled_base),
            "mean_hedge_weight":   round(float(pooled_hw.mean()), 4),
            "median_hedge_weight": round(float(pooled_hw.median()), 4),
            "frac_zero_hedge":     round(float((pooled_hw < 0.01).mean()), 4),
            "frac_high_hedge":     round(float((pooled_hw > 0.12).mean()), 4),
        },
        # Stash for HTML cards (don't serialise full series)
        "_pooled_dyn": pooled_dyn,
        "_pooled_static": pooled_static,
        "_pooled_hw": pooled_hw,
    }


# ───────────────────────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────────────────────

def write_html(payload: Dict, path: Path) -> None:
    p = payload["pooled_oos"]
    dyn = p["dynamic"]; sta = p["static_5"]; bas = p["baseline"]

    delta_cagr = round(dyn["cagr_pct"] - sta["cagr_pct"], 2)
    delta_dd   = round(dyn["max_dd_pct"] - sta["max_dd_pct"], 2)
    delta_sh   = round(dyn["sharpe"] - sta["sharpe"], 2)
    success = (dyn["max_dd_pct"] <= sta["max_dd_pct"] + 0.5
               and dyn["cagr_pct"] >= sta["cagr_pct"])
    color = "#16a34a" if success else "#dc2626"
    msg = "✅ TARGET MET" if success else "❌ Did not meet DD/CAGR target"

    fold_rows = ""
    for f in payload["walk_forward"]["folds"]:
        d = f["dynamic"]; s = f["static"]
        ds = round(d["sharpe"] - s["sharpe"], 2)
        c = "#16a34a" if ds > 0 else "#dc2626"
        fold_rows += (
            f"<tr><td>{f['test_start']}</td><td>{f['test_end']}</td>"
            f"<td>{f['mean_test_corr']:+.2f}</td>"
            f"<td>{f['mean_test_hedge_w']:.2%}</td>"
            f"<td>{s['sharpe']:.2f}</td><td>{d['sharpe']:.2f}</td>"
            f"<td style='color:{c};font-weight:700'>{ds:+.2f}</td>"
            f"<td>{s['max_dd_pct']:.1f}%</td><td>{d['max_dd_pct']:.1f}%</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset='UTF-8'>
<title>EXP-1980 Dynamic Hedge Ratio</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1100px}}
h1{{font-size:1.5rem;color:#0f172a}} h2{{font-size:1.05rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:18px}}
.headline{{background:#f0fdf4;border-left:4px solid {color};padding:14px;border-radius:6px;margin:14px 0;font-size:0.95rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.15rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.83rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}} td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}} td:first-child{{text-align:left}}
</style></head><body>
<h1>EXP-1980 — Correlation Regime Switching / Dynamic Hedge Ratio</h1>
<p class='meta'>Streams: EXP-1220 · v5_hedge · GLD/SLV calendars · SPY (all real).
Walk-forward 252/63. Hedge schedule re-fit per fold.</p>

<div class='headline'><strong>Pooled OOS:</strong>
&nbsp;Static-5% CAGR <strong>{sta['cagr_pct']:+.2f}%</strong> Sharpe <strong>{sta['sharpe']:.2f}</strong> DD <strong>{sta['max_dd_pct']:.1f}%</strong>
→ Dynamic CAGR <strong>{dyn['cagr_pct']:+.2f}%</strong> Sharpe <strong>{dyn['sharpe']:.2f}</strong> DD <strong>{dyn['max_dd_pct']:.1f}%</strong>
&nbsp;|&nbsp; ΔCAGR <strong>{delta_cagr:+.2f}pp</strong>, ΔSharpe <strong>{delta_sh:+.2f}</strong>, ΔDD <strong>{delta_dd:+.2f}pp</strong>
&nbsp;({msg})</div>

<div class='grid'>
  <div class='card'><div class='l'>Mean hedge w</div><div class='v'>{p['mean_hedge_weight']:.1%}</div></div>
  <div class='card'><div class='l'>Median hedge w</div><div class='v'>{p['median_hedge_weight']:.1%}</div></div>
  <div class='card'><div class='l'>Days at 0% hedge</div><div class='v'>{p['frac_zero_hedge']:.0%}</div></div>
  <div class='card'><div class='l'>Days at &gt;12% hedge</div><div class='v'>{p['frac_high_hedge']:.0%}</div></div>
  <div class='card'><div class='l'>Baseline CAGR</div><div class='v'>{bas['cagr_pct']:+.2f}%</div></div>
  <div class='card'><div class='l'>Baseline DD</div><div class='v'>{bas['max_dd_pct']:.1f}%</div></div>
</div>

<h2>Walk-Forward Folds</h2>
<table><tr><th>Test start</th><th>Test end</th>
<th>Mean test ρ</th><th>Mean hedge w</th>
<th>Static Sh</th><th>Dyn Sh</th><th>Δ Sh</th>
<th>Static DD</th><th>Dyn DD</th></tr>
{fold_rows}</table>

<h2>Method</h2>
<ul>
<li>Streams: cached EXP-1850 (exp1220 + v5_hedge) + exp1770 GLD/SLV calendars + Yahoo SPY.</li>
<li>Rolling 60-day corr(EXP-1220, SPY) gates a 3-knee hedge schedule
   (low_thr, high_thr, w_low, w_mid, w_high). Grid search per train fold,
   apply OOS.</li>
<li>Walk-forward 252-day train, 63-day test, expanding step.</li>
<li>Pooled OOS metrics aggregate every test fold's daily returns.</li>
</ul>
<div style='color:#94a3b8;font-size:0.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px'>
compass/exp1980_dynamic_hedge.py · ALL REAL DATA</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-1980 — Correlation Regime Switching / Dynamic Hedge Ratio")
    print("=" * 60)

    df = load_streams()
    print(f"[align] {len(df)} aligned days {df.index.min().date()} → "
          f"{df.index.max().date()}")
    print(f"[streams] columns: {list(df.columns)}")
    for c in df.columns:
        m = metrics(df[c])
        print(f"  {c:<10} CAGR {m['cagr_pct']:+6.2f}%  Sharpe {m['sharpe']:+.2f}  "
              f"DD {m['max_dd_pct']:.1f}%")

    print("\n[corr] computing 60-day rolling corr(EXP-1220, SPY)...")
    corr = rolling_corr(df["exp1220"], df["spy"], 60)
    print(f"  mean={float(corr.mean()):+.3f}  median={float(corr.median()):+.3f}  "
          f"min={float(corr.min()):+.3f}  max={float(corr.max()):+.3f}")
    print(f"  frac > +0.50: {float((corr > 0.5).mean()):.0%}")
    print(f"  frac < +0.20: {float((corr < 0.2).mean()):.0%}")

    print("\n[walk-forward] 252/63 with grid-search per fold...")
    wf = walk_forward(df, train_days=252, test_days=63)
    print(f"  {len(wf['folds'])} folds")

    p = wf["pooled_oos"]
    print()
    print("POOLED OOS RESULTS")
    print("-" * 60)
    print(f"{'metric':<14}{'baseline':>14}{'static_5%':>14}{'dynamic':>14}")
    for k in ["cagr_pct", "sharpe", "sortino", "max_dd_pct", "calmar", "vol_pct"]:
        b = p["baseline"].get(k, 0); s = p["static_5"].get(k, 0); d = p["dynamic"].get(k, 0)
        print(f"{k:<14}{b:>14.3f}{s:>14.3f}{d:>14.3f}")
    print()
    print(f"Mean hedge weight (dynamic):   {p['mean_hedge_weight']:.2%}")
    print(f"Frac of days at 0% hedge:      {p['frac_zero_hedge']:.0%}")
    print(f"Frac of days at >12% hedge:    {p['frac_high_hedge']:.0%}")

    delta_cagr = p["dynamic"]["cagr_pct"] - p["static_5"]["cagr_pct"]
    delta_dd   = p["dynamic"]["max_dd_pct"] - p["static_5"]["max_dd_pct"]
    target_met = (p["dynamic"]["max_dd_pct"] <= p["static_5"]["max_dd_pct"] + 0.5
                  and delta_cagr >= 0)
    print(f"\nΔ CAGR: {delta_cagr:+.2f}pp   Δ DD: {delta_dd:+.2f}pp   "
          f"target {'✅ MET' if target_met else '❌ MISS'}")

    payload = {
        "experiment": "EXP-1980",
        "title": "Correlation Regime Switching — Dynamic Hedge Ratio",
        "date_range": {"start": START, "end": END},
        "data_sources": {
            "exp1220":  "compass/cache/exp1850_streams.pkl (REAL)",
            "v5_hedge": "compass/cache/exp1850_streams.pkl (REAL)",
            "gld_cal":  "compass.exp1770_commodity_calendars walk_forward GLD-GC=F (REAL)",
            "slv_cal":  "compass.exp1770_commodity_calendars walk_forward SLV-SI=F (REAL)",
            "spy":      "Yahoo Finance SPY (REAL)",
        },
        "stream_metrics": {c: metrics(df[c]) for c in df.columns},
        "rolling_corr_stats": {
            "window": 60,
            "mean":   round(float(corr.mean()), 4),
            "median": round(float(corr.median()), 4),
            "min":    round(float(corr.min()), 4),
            "max":    round(float(corr.max()), 4),
            "frac_above_0_5": round(float((corr > 0.5).mean()), 4),
            "frac_below_0_2": round(float((corr < 0.2).mean()), 4),
        },
        "grid": {
            "low_thr":  GRID_LOW_THR,
            "high_thr": GRID_HIGH_THR,
            "w_low":    GRID_W_LOW,
            "w_mid":    GRID_W_MID,
            "w_high":   GRID_W_HIGH,
        },
        "walk_forward": {
            "train_days": 252, "test_days": 63,
            "n_folds": len(wf["folds"]),
            "folds":   wf["folds"],
        },
        "pooled_oos": p,
        "deltas": {
            "cagr_pp_dynamic_vs_static_5": round(delta_cagr, 3),
            "dd_pp_dynamic_vs_static_5":   round(delta_dd,   3),
            "sharpe_dynamic_vs_static_5":  round(p["dynamic"]["sharpe"]
                                                 - p["static_5"]["sharpe"], 3),
        },
        "target_met": bool(target_met),
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    write_html({**payload, "walk_forward": wf}, REPORT_HTML)
    print(f"\nReports → {REPORT_JSON.name} + {REPORT_HTML.name}")
    return payload


if __name__ == "__main__":
    main()
