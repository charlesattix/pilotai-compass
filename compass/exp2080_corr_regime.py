"""
EXP-2080 — Correlation Regime Switching for Portfolio Allocation

Hypothesis
----------
Pairwise correlations between alpha streams are not stationary. When the
market enters systemic stress, previously-uncorrelated streams co-move
with SPY and the diversification benefit collapses; when streams
*decorrelate* (or go anti-correlated), the diversification benefit grows
and the portfolio can safely take more risk.

This experiment builds a correlation-regime detector on the live
multi-stream portfolio and dynamically allocates weight as a function of
the detected regime.

Streams (REAL data, all canonical loaders)
------------------------------------------
  exp1220     EXP-1220 dynamic credit spread proxy   (cached EXP-1850)
  v5_hedge    Crisis Alpha v5 best frozen config     (cached EXP-1850)
  gld_cal     GLD calendar (ETF − GC=F front)        (compass.exp1770)
  slv_cal     SLV calendar (ETF − SI=F front)        (compass.exp1770)
  cross_vol   Cross-sectional vol arbitrage trades   (compass.exp2020)

Regime taxonomy
---------------
We compute the rolling-60-day mean off-diagonal pairwise correlation
across the 5 streams and classify each day:

  STRESS         mean_off_diag >= +0.50      (correlations spike)
  DECORRELATION  mean_off_diag <= -0.05      (true diversification)
  NORMAL         everything else

Static weights (baseline)
-------------------------
  exp1220 0.40 · gld_cal 0.20 · slv_cal 0.20 · cross_vol 0.15 · v5_hedge 0.05

Dynamic weights
---------------
  STRESS:        cut equity-heavy, lift hedge
                 exp1220 0.20 · gld_cal 0.20 · slv_cal 0.15 ·
                 cross_vol 0.15 · v5_hedge 0.30
  DECORRELATION: increase risk budget (1.3x leverage on the static set)
  NORMAL:        static weights, 1.0x leverage

Walk-forward 2020-2025: dynamic vs static, 252-day train / 63-day test.
We focus on whether DD < 12% with Sharpe ≥ 4.0.

ALL REAL DATA. No synthetic.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_PKL_1850   = ROOT / "compass" / "cache" / "exp1850_streams.pkl"
CACHE_2080       = ROOT / "compass" / "cache" / "exp2080_streams.pkl"
REPORT_JSON      = ROOT / "compass" / "reports" / "exp2080_corr_regime.json"
REPORT_HTML      = ROOT / "compass" / "reports" / "exp2080_corr_regime.html"

TRADING_DAYS = 252
START = "2020-01-01"
END   = "2025-12-31"

STREAM_NAMES = ["exp1220", "v5_hedge", "gld_cal", "slv_cal", "cross_vol"]


# ───────────────────────────────────────────────────────────────────────────
# Stream loaders (REAL data, cached)
# ───────────────────────────────────────────────────────────────────────────

def _trades_to_daily(trades: List[Dict],
                     index: pd.DatetimeIndex,
                     capital: float = 100_000.0) -> pd.Series:
    """Convert (entry/exit/pnl) trades to a daily-return series on `index`."""
    s = pd.Series(0.0, index=index)
    for t in trades:
        try:
            d = pd.Timestamp(t["exit_date"])
            if d in s.index:
                s.loc[d] += float(t["pnl"]) / capital
        except Exception:
            pass
    return s


def load_streams() -> pd.DataFrame:
    """Load + cache the 5-stream daily-return DataFrame."""
    if CACHE_2080.exists():
        print(f"[cache] loading {CACHE_2080.name}")
        return pickle.load(open(CACHE_2080, "rb"))

    # 1. EXP-1220 + v5_hedge from cached pickle
    print("[load] cached exp1220 + v5_hedge from EXP-1850 pickle")
    if not CACHE_PKL_1850.exists():
        from compass.exp1850_regime_portfolio import load_real_streams
        load_real_streams()
    cached = pickle.load(open(CACHE_PKL_1850, "rb"))
    exp1220 = cached["exp1220"].rename("exp1220")
    v5_hedge = cached["v5_hedge"].rename("v5_hedge")

    # 2. GLD/SLV calendar streams from exp1770
    print("[load] GLD/SLV calendar streams (exp1770)")
    from compass.exp1770_commodity_calendars import load_pair, walk_forward, PAIRS
    gld_etf, gld_fut, _ = PAIRS["GLD"]
    slv_etf, slv_fut, _ = PAIRS["SLV"]
    gld_cal = walk_forward("GLD", load_pair(gld_etf, gld_fut)).daily_returns.rename("gld_cal")
    slv_cal = walk_forward("SLV", load_pair(slv_etf, slv_fut)).daily_returns.rename("slv_cal")

    # 3. Cross-Vol Arb daily returns (built from exp2020 trades on the
    #    union of stream business days, exit-date PnL / capital)
    print("[load] Cross-Sectional Vol Arb trades (exp2020)")
    from compass.exp2020_cross_vol_arb import (
        UNIVERSE, load_prices, weekly_signal_panel, build_trades,
    )
    from shared.iron_vault import IronVault
    prices = load_prices(UNIVERSE)
    hd = IronVault.instance()
    panel = weekly_signal_panel(prices, hd)
    cv_trades = build_trades(panel, prices)
    print(f"      cross_vol trades: {len(cv_trades)}")

    # Build a common business-day index spanning the union of streams.
    all_idx = pd.date_range(START, END, freq="B")
    cross_vol = _trades_to_daily(cv_trades, all_idx).rename("cross_vol")

    df = pd.concat([exp1220, v5_hedge, gld_cal, slv_cal, cross_vol], axis=1)
    df = df.loc[START:END].fillna(0.0).sort_index()
    df = df[~df.index.duplicated(keep="last")]

    CACHE_2080.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(df, open(CACHE_2080, "wb"))
    print(f"[cache] saved → {CACHE_2080}")
    return df


# ───────────────────────────────────────────────────────────────────────────
# Correlation regime detection
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeThresholds:
    stress_min:        float = 0.50
    decorrelation_max: float = -0.05
    window:            int   = 60


REGIME_NORMAL = "normal"
REGIME_STRESS = "stress"
REGIME_DECORR = "decorrelation"


def rolling_mean_off_diag_corr(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling mean of off-diagonal pairwise correlations across all columns."""
    out = pd.Series(np.nan, index=df.index)
    cols = list(df.columns)
    n_pairs = len(cols) * (len(cols) - 1) // 2
    if n_pairs == 0:
        return out
    arr = df.values
    for i in range(window, len(df) + 1):
        sub = arr[i - window:i]
        if sub.shape[0] < window:
            continue
        # Per-column std; if any column is degenerate, fall back to nan
        c = np.corrcoef(sub.T)
        if not np.all(np.isfinite(c)):
            continue
        # Mean of upper triangle off-diagonals
        iu = np.triu_indices(c.shape[0], k=1)
        out.iloc[i - 1] = float(np.mean(c[iu]))
    return out.ffill().fillna(0.0)


def classify_regime(mean_corr: float, thr: RegimeThresholds) -> str:
    if mean_corr >= thr.stress_min:
        return REGIME_STRESS
    if mean_corr <= thr.decorrelation_max:
        return REGIME_DECORR
    return REGIME_NORMAL


# ───────────────────────────────────────────────────────────────────────────
# Allocation schedules
# ───────────────────────────────────────────────────────────────────────────

STATIC_WEIGHTS: Dict[str, float] = {
    "exp1220":   0.40,
    "gld_cal":   0.20,
    "slv_cal":   0.20,
    "cross_vol": 0.15,
    "v5_hedge":  0.05,
}

# Stress-regime weights: cut equity-heavy, lift hedge
STRESS_WEIGHTS: Dict[str, float] = {
    "exp1220":   0.20,
    "gld_cal":   0.20,
    "slv_cal":   0.15,
    "cross_vol": 0.15,
    "v5_hedge":  0.30,
}

DECORR_LEVERAGE = 1.30   # boost on the static schedule when diversification is real
NORMAL_LEVERAGE = 1.00
STRESS_LEVERAGE = 1.00


def regime_weights_and_lev(regime: str) -> Tuple[Dict[str, float], float]:
    if regime == REGIME_STRESS:
        return STRESS_WEIGHTS, STRESS_LEVERAGE
    if regime == REGIME_DECORR:
        return STATIC_WEIGHTS, DECORR_LEVERAGE
    return STATIC_WEIGHTS, NORMAL_LEVERAGE


# ───────────────────────────────────────────────────────────────────────────
# Portfolio construction
# ───────────────────────────────────────────────────────────────────────────

def static_portfolio(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(0.0, index=df.index)
    for k, w in STATIC_WEIGHTS.items():
        if k in df.columns:
            out = out + w * df[k]
    return out


def dynamic_portfolio(df: pd.DataFrame,
                      regime_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Apply per-day regime weights+leverage. Returns (daily_ret, lev_series)."""
    out = pd.Series(0.0, index=df.index)
    lev_out = pd.Series(1.0, index=df.index)
    for d in df.index:
        r = regime_series.get(d, REGIME_NORMAL) if d in regime_series.index else REGIME_NORMAL
        w, lev = regime_weights_and_lev(str(r))
        day_ret = 0.0
        for k, wk in w.items():
            if k in df.columns:
                day_ret += wk * float(df.at[d, k])
        out.loc[d] = lev * day_ret
        lev_out.loc[d] = lev
    return out, lev_out


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
    }


# ───────────────────────────────────────────────────────────────────────────
# Walk-forward
# ───────────────────────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, train_days: int = 252, test_days: int = 63
                 ) -> Dict:
    """Per-fold: compute regime series on data up to test cut, apply OOS."""
    n = len(df)
    folds = []
    pooled_dyn:    List[pd.Series] = []
    pooled_static: List[pd.Series] = []
    pooled_regime: List[pd.Series] = []
    pooled_lev:    List[pd.Series] = []
    thr = RegimeThresholds()

    i = train_days
    while i + test_days <= n:
        slc_full = df.iloc[:i + test_days]
        # Compute rolling mean off-diag corr through the FULL slice; the
        # value at row k uses only rows [k-window+1 .. k] so this is causal.
        m = rolling_mean_off_diag_corr(slc_full, thr.window)
        regime_series = m.apply(lambda x: classify_regime(x, thr))
        te_slice = df.iloc[i:i + test_days]
        te_regime = regime_series.iloc[i:i + test_days]

        dyn, lev = dynamic_portfolio(te_slice, te_regime)
        sta = static_portfolio(te_slice)

        # Per-fold regime distribution
        rdist = te_regime.value_counts(normalize=True).to_dict()

        folds.append({
            "test_start": str(te_slice.index[0].date()),
            "test_end":   str(te_slice.index[-1].date()),
            "regime_dist": {k: round(float(v), 3) for k, v in rdist.items()},
            "mean_corr":   round(float(m.iloc[i:i + test_days].mean()), 3),
            "static":  metrics(sta),
            "dynamic": metrics(dyn),
        })
        pooled_dyn.append(dyn)
        pooled_static.append(sta)
        pooled_regime.append(te_regime)
        pooled_lev.append(lev)
        i += test_days

    pooled_d = pd.concat(pooled_dyn).sort_index()
    pooled_s = pd.concat(pooled_static).sort_index()
    pooled_r = pd.concat(pooled_regime).sort_index()
    pooled_l = pd.concat(pooled_lev).sort_index()
    rdist_full = pooled_r.value_counts(normalize=True).to_dict()

    return {
        "folds": folds,
        "pooled_oos": {
            "static":  metrics(pooled_s),
            "dynamic": metrics(pooled_d),
            "regime_distribution": {k: round(float(v), 4) for k, v in rdist_full.items()},
            "mean_leverage": round(float(pooled_l.mean()), 4),
            "frac_stress":  round(float((pooled_r == REGIME_STRESS).mean()), 4),
            "frac_decorr":  round(float((pooled_r == REGIME_DECORR).mean()), 4),
            "frac_normal":  round(float((pooled_r == REGIME_NORMAL).mean()), 4),
        },
    }


# ───────────────────────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────────────────────

def write_html(payload: Dict, path: Path) -> None:
    p = payload["walk_forward"]["pooled_oos"]
    sta = p["static"]; dyn = p["dynamic"]
    target_ok = (dyn["max_dd_pct"] < 12.0 and dyn["sharpe"] >= 4.0)
    color = "#16a34a" if target_ok else "#dc2626"
    msg = "✅ TARGET MET (DD < 12% and Sharpe ≥ 4.0)" if target_ok \
          else "⚠ Target not met (need DD < 12% AND Sharpe ≥ 4.0)"

    fold_rows = ""
    for f in payload["walk_forward"]["folds"]:
        s = f["static"]; d = f["dynamic"]
        rds = " ".join(f"{k[0]}={v:.0%}" for k, v in f["regime_dist"].items())
        ds = round(d["sharpe"] - s["sharpe"], 2)
        c = "#16a34a" if ds > 0 else "#dc2626"
        fold_rows += (
            f"<tr><td>{f['test_start']}</td><td>{f['test_end']}</td>"
            f"<td>{f['mean_corr']:+.2f}</td>"
            f"<td>{s['sharpe']:.2f}</td><td>{d['sharpe']:.2f}</td>"
            f"<td style='color:{c};font-weight:700'>{ds:+.2f}</td>"
            f"<td>{s['max_dd_pct']:.1f}%</td><td>{d['max_dd_pct']:.1f}%</td>"
            f"<td style='font-size:0.78rem'>{rds}</td></tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset='UTF-8'>
<title>EXP-2080 Correlation Regime Switching</title>
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
<h1>EXP-2080 — Correlation Regime Switching</h1>
<p class='meta'>5 streams: exp1220 · v5_hedge · gld_cal · slv_cal · cross_vol · all REAL.
Walk-forward 252/63 with rolling-60d mean off-diagonal pairwise corr regime detector.</p>

<div class='headline'><strong>Pooled OOS:</strong>
&nbsp;Static  Sharpe <strong>{sta['sharpe']:.2f}</strong>  CAGR <strong>{sta['cagr_pct']:+.2f}%</strong>  DD <strong>{sta['max_dd_pct']:.2f}%</strong>
&nbsp;|&nbsp; Dynamic Sharpe <strong>{dyn['sharpe']:.2f}</strong>  CAGR <strong>{dyn['cagr_pct']:+.2f}%</strong>  DD <strong>{dyn['max_dd_pct']:.2f}%</strong>
&nbsp; ({msg})</div>

<div class='grid'>
  <div class='card'><div class='l'>% Normal</div><div class='v'>{p['frac_normal']:.0%}</div></div>
  <div class='card'><div class='l'>% Stress</div><div class='v'>{p['frac_stress']:.0%}</div></div>
  <div class='card'><div class='l'>% Decorr</div><div class='v'>{p['frac_decorr']:.0%}</div></div>
  <div class='card'><div class='l'>Mean lev</div><div class='v'>{p['mean_leverage']:.2f}x</div></div>
  <div class='card'><div class='l'>Static vol</div><div class='v'>{sta['vol_pct']:.1f}%</div></div>
  <div class='card'><div class='l'>Dyn vol</div><div class='v'>{dyn['vol_pct']:.1f}%</div></div>
</div>

<h2>Walk-Forward Folds</h2>
<table><tr><th>Test start</th><th>Test end</th>
<th>Mean ρ</th><th>Static Sh</th><th>Dyn Sh</th><th>Δ Sh</th>
<th>Static DD</th><th>Dyn DD</th><th>Regime mix</th></tr>
{fold_rows}</table>

<h2>Method</h2>
<ul>
<li>Streams: exp1220 + v5_hedge from cached EXP-1850 pickle; gld_cal /
   slv_cal from compass.exp1770 walk_forward; cross_vol from
   compass.exp2020 build_trades (exit-date PnL → daily series).</li>
<li>Regime: rolling-60d mean off-diagonal pairwise correlation across
   all 5 streams. STRESS if ≥ +0.50, DECORRELATION if ≤ -0.05, NORMAL
   otherwise. (Causal: row k uses rows [k-59..k] only.)</li>
<li>Static schedule: exp1220 0.40 / gld_cal 0.20 / slv_cal 0.20 /
   cross_vol 0.15 / v5_hedge 0.05.</li>
<li>Stress schedule: exp1220 0.20 / gld_cal 0.20 / slv_cal 0.15 /
   cross_vol 0.15 / v5_hedge 0.30 (cut equity-heavy, lift hedge).</li>
<li>Decorrelation: keep static weights, lift leverage 1.0x → 1.30x.</li>
<li>Walk-forward 252/63, dynamic vs static measured on the same OOS days.</li>
<li>Target: max DD &lt; 12% AND Sharpe ≥ 4.0.</li>
</ul>
<div style='color:#94a3b8;font-size:0.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px'>
compass/exp2080_corr_regime.py · ALL REAL DATA</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-2080 — Correlation Regime Switching for Portfolio Allocation")
    print("=" * 60)

    df = load_streams()
    print(f"[align] {len(df)} aligned business days "
          f"{df.index.min().date()} → {df.index.max().date()}")
    print(f"[streams] {list(df.columns)}")
    for c in df.columns:
        m = metrics(df[c])
        print(f"  {c:<10} CAGR {m['cagr_pct']:+7.2f}%  Sharpe {m['sharpe']:+.2f}  "
              f"DD {m['max_dd_pct']:.2f}%")

    thr = RegimeThresholds()
    mc = rolling_mean_off_diag_corr(df, thr.window)
    print(f"\n[corr] mean off-diag corr (60d)  mean={float(mc.mean()):+.3f}  "
          f"min={float(mc.min()):+.3f}  max={float(mc.max()):+.3f}")
    rdist_full = mc.apply(lambda x: classify_regime(x, thr)).value_counts(normalize=True)
    for k, v in rdist_full.items():
        print(f"  {k:<14}: {v:.0%}")

    print("\n[walk-forward] 252/63...")
    wf = walk_forward(df, train_days=252, test_days=63)
    p = wf["pooled_oos"]
    print(f"  {len(wf['folds'])} folds")
    print()
    print("POOLED OOS RESULTS")
    print("-" * 60)
    print(f"{'metric':<14}{'static':>14}{'dynamic':>14}{'delta':>14}")
    for k in ["cagr_pct", "sharpe", "sortino", "max_dd_pct", "calmar", "vol_pct"]:
        s = p["static"].get(k, 0); d = p["dynamic"].get(k, 0)
        print(f"{k:<14}{s:>14.3f}{d:>14.3f}{(d-s):>14.3f}")
    print()
    print(f"Regime distribution (OOS): "
          f"Normal {p['frac_normal']:.0%}  "
          f"Stress {p['frac_stress']:.0%}  "
          f"Decorr {p['frac_decorr']:.0%}")
    print(f"Mean dynamic leverage: {p['mean_leverage']:.2f}x")

    target_ok = (p["dynamic"]["max_dd_pct"] < 12.0 and p["dynamic"]["sharpe"] >= 4.0)
    print(f"\nTarget (DD < 12% AND Sharpe ≥ 4.0): "
          f"{'✅ MET' if target_ok else '⚠ MISS'}")

    payload = {
        "experiment": "EXP-2080",
        "title": "Correlation Regime Switching for Portfolio Allocation",
        "date_range": {"start": START, "end": END},
        "data_sources": {
            "exp1220":  "compass/cache/exp1850_streams.pkl (REAL)",
            "v5_hedge": "compass/cache/exp1850_streams.pkl (REAL)",
            "gld_cal":  "compass.exp1770_commodity_calendars walk_forward (REAL)",
            "slv_cal":  "compass.exp1770_commodity_calendars walk_forward (REAL)",
            "cross_vol":"compass.exp2020_cross_vol_arb build_trades (REAL)",
        },
        "regime_thresholds": asdict(thr),
        "static_weights":    STATIC_WEIGHTS,
        "stress_weights":    STRESS_WEIGHTS,
        "decorr_leverage":   DECORR_LEVERAGE,
        "stream_metrics":    {c: metrics(df[c]) for c in df.columns},
        "rolling_corr_stats": {
            "mean":   round(float(mc.mean()), 4),
            "median": round(float(mc.median()), 4),
            "min":    round(float(mc.min()), 4),
            "max":    round(float(mc.max()), 4),
        },
        "regime_distribution_full": {str(k): round(float(v), 4)
                                     for k, v in rdist_full.items()},
        "walk_forward": wf,
        "target": {"max_dd_pct_lt": 12.0, "sharpe_gte": 4.0,
                   "met": bool(target_ok)},
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    write_html(payload, REPORT_HTML)
    print(f"\nReports → {REPORT_JSON.name} + {REPORT_HTML.name}")
    return payload


if __name__ == "__main__":
    main()
