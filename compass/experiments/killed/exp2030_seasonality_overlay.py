"""
EXP-2030 — Intraweek Seasonality Overlay for EXP-1220

Hypothesis
----------
Academic options-market research documents persistent calendar effects:
day-of-week (Monday/Friday), week-of-month, opex week, and FOMC-week
anomalies. We test whether EXP-1220's 5-year real IronVault trade record
exhibits any of these effects strongly enough to be worth filtering on.

Method
------
1. Run baseline EXP-1220 (REAL IronVault SPY options) 2020-2025.
2. Tag every trade with calendar features at *entry date*:
     - day_of_week        (0=Mon … 4=Fri)
     - week_of_month      (1..5, Mon-anchored)
     - week_to_opex       (calendar-weeks distance to next 3rd-Friday)
     - week_to_fomc       (calendar-weeks distance to nearest FOMC meeting)
     - is_opex_week       (entry within 4 cal-days of monthly opex Friday)
     - is_fomc_week       (entry within 4 cal-days of an FOMC meeting)
3. Tabulate per-bucket trade Sharpe / win rate / mean PnL.
4. Build a `skip_set` of (feature, value) pairs whose Sharpe is below
   a chosen quantile in training data (worst 25%).
5. Walk-forward expanding train: fit `skip_set` on data through year-1,
   apply to year. Pool OOS trades and report.

ALL REAL DATA — FOMC dates are public Fed calendar facts (not synthetic).
Outputs
-------
  compass/exp2030_seasonality_overlay.py
  compass/reports/exp2030_seasonality_overlay.json
  compass/reports/exp2030_seasonality_overlay.html
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault
from compass.exp1220_standalone import (
    _exp_dt, _find_exps, _next_td, _sell_put_spread, _walk_spread,
)

TRADING_DAYS = 252
START_DATA = "2019-06-01"
END_DATA = "2026-04-02"
BT_START = "2020-01-01"
BT_END = "2025-12-31"


# ───────────────────────────────────────────────────────────────────────────
# Calendar facts: FOMC meeting dates (public Federal Reserve calendar)
# ───────────────────────────────────────────────────────────────────────────

# Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
# Two-day meetings → use the Day-2 (decision) date.
FOMC_DECISION_DATES: List[str] = [
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",  # incl emergency
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
]

_FOMC_DT = pd.DatetimeIndex(pd.to_datetime(FOMC_DECISION_DATES)).sort_values()


def _third_friday(year: int, month: int) -> date:
    """Monthly equity-options OPEX = third Friday of the month."""
    d = date(year, month, 1)
    # weekday(): Mon=0..Sun=6  Friday=4
    offset = (4 - d.weekday()) % 7
    first_friday = d + timedelta(days=offset)
    return first_friday + timedelta(days=14)


def _opex_dates(start: str, end: str) -> pd.DatetimeIndex:
    sy, sm = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]),   int(end[5:7])
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(_third_friday(y, m))
        m += 1
        if m > 12:
            m = 1; y += 1
    return pd.DatetimeIndex(pd.to_datetime(out))


_OPEX_DT = _opex_dates("2019-01", "2026-12")


# ───────────────────────────────────────────────────────────────────────────
# Calendar feature tagging
# ───────────────────────────────────────────────────────────────────────────

def _nearest_days(d: pd.Timestamp, ref: pd.DatetimeIndex) -> int:
    """Signed distance in calendar days to the nearest ref date (abs)."""
    if len(ref) == 0:
        return 999
    diffs = np.abs((ref - d).days)
    return int(diffs.min())


def _next_days_forward(d: pd.Timestamp, ref: pd.DatetimeIndex) -> int:
    """Calendar days to the next ref date >= d (or 999 if none)."""
    later = ref[ref >= d]
    if len(later) == 0:
        return 999
    return int((later[0] - d).days)


def tag_trade(trade: Dict) -> Dict:
    """Add calendar features to a trade dict in-place; return it."""
    d = pd.Timestamp(trade["entry_date"])
    trade["dow"]            = int(d.weekday())          # 0..4
    trade["week_of_month"]  = int((d.day - 1) // 7) + 1 # 1..5
    trade["days_to_opex"]   = _next_days_forward(d, _OPEX_DT)
    trade["days_to_fomc"]   = _nearest_days(d, _FOMC_DT)
    trade["is_opex_week"]   = int(trade["days_to_opex"] <= 4)
    trade["is_fomc_week"]   = int(trade["days_to_fomc"] <= 4)
    return trade


# ───────────────────────────────────────────────────────────────────────────
# Run EXP-1220 baseline
# ───────────────────────────────────────────────────────────────────────────

def run_baseline_trades(hd: IronVault, spy_df: pd.DataFrame,
                        vix_close: pd.Series, start: str, end: str) -> List[Dict]:
    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, start, end, monthly=False)
    trades, last = [], None
    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=28), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if es < start or es > end:
            continue
        if last and (entry_dt - last).days < 10:
            continue
        try:
            price = float(spy_close.loc[es]); v = float(vix_close.loc[es])
        except Exception:
            continue
        if np.isnan(price) or np.isnan(v) or v > 40:
            continue
        spread = _sell_put_spread(hd, exp, es, price, otm_pct=0.95, width=5.0)
        if spread is None:
            continue
        cts = max(1, min(4, int(100_000 * 0.03 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(hd, exp, spread["short"], spread["long"],
                                        spread["credit"], entry_dt, exp_obj, spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * cts
        t = {"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
             "exit_reason": er, "credit": spread["credit"], "vix": round(v, 1),
             "hold_days": hold, "contracts": cts}
        trades.append(tag_trade(t))
        last = entry_dt
    return trades


# ───────────────────────────────────────────────────────────────────────────
# Bucket statistics + skip-set fitting
# ───────────────────────────────────────────────────────────────────────────

def _bucket_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {"n": 0, "mean_pnl": 0.0, "win_rate": 0.0, "sharpe": 0.0}
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls); mu = float(pnls.mean()); sigma = float(pnls.std(ddof=1)) if n > 1 else 1.0
    sharpe = mu / sigma * math.sqrt(n) if sigma > 1e-9 else 0.0
    return {
        "n": n,
        "mean_pnl": round(mu, 2),
        "win_rate": round(float((pnls > 0).mean()), 4),
        "sharpe":   round(sharpe, 3),
    }


FEATURE_KEYS = ["dow", "week_of_month", "is_opex_week", "is_fomc_week"]


def bucket_table(trades: List[Dict]) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for key in FEATURE_KEYS:
        vals = sorted({t[key] for t in trades})
        out[key] = {str(v): _bucket_metrics([t for t in trades if t[key] == v])
                    for v in vals}
    return out


@dataclass
class SkipSet:
    """Set of (feature, value) pairs whose entries should be filtered out."""
    pairs: List[Tuple[str, int]] = field(default_factory=list)
    min_bucket_n: int = 5

    def should_skip(self, trade: Dict) -> bool:
        for k, v in self.pairs:
            if int(trade[k]) == v:
                return True
        return False

    def to_serialisable(self) -> List[Dict]:
        return [{"feature": k, "value": v} for k, v in self.pairs]


def fit_skip_set(trades: List[Dict],
                 quantile: float = 0.25,
                 min_bucket_n: int = 5) -> SkipSet:
    """Identify (feature, value) buckets whose Sharpe is in the worst quantile.

    To avoid spurious tiny buckets we require min_bucket_n trades.
    """
    candidates: List[Tuple[str, int, float]] = []
    for key in FEATURE_KEYS:
        vals = sorted({t[key] for t in trades})
        for v in vals:
            sub = [t for t in trades if t[key] == v]
            if len(sub) < min_bucket_n:
                continue
            m = _bucket_metrics(sub)
            candidates.append((key, int(v), m["sharpe"]))

    if not candidates:
        return SkipSet([], min_bucket_n)

    sharpes = [c[2] for c in candidates]
    cutoff = float(np.quantile(sharpes, quantile))
    pairs = [(k, v) for k, v, s in candidates if s <= cutoff and s < 0.0]
    return SkipSet(pairs, min_bucket_n)


def apply_skip_set(trades: List[Dict], skip: SkipSet) -> List[Dict]:
    return [t for t in trades if not skip.should_skip(t)]


# ───────────────────────────────────────────────────────────────────────────
# Trade-level metrics
# ───────────────────────────────────────────────────────────────────────────

def trade_metrics(trades: List[Dict]) -> Dict:
    if not trades:
        return {"n": 0, "total_pnl": 0.0, "win_rate": 0.0, "avg_pnl": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "max_dd_pct": 0.0,
                "cagr_pct": 0.0, "calmar": 0.0, "avg_hold_days": 0.0}
    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls); total = float(pnls.sum()); wins = int((pnls > 0).sum())
    eq = np.cumsum(pnls) + 100_000
    peak = np.maximum.accumulate(eq)
    dd = float(((peak - eq) / peak).max())
    df = pd.DataFrame(trades)
    en = pd.to_datetime(df["entry_date"]); ex = pd.to_datetime(df["exit_date"])
    years = max((ex.max() - en.min()).days / 365.25, 0.5)
    cagr = ((1 + total / 100_000) ** (1 / years) - 1) if total > -100_000 else -1.0
    mu = float(pnls.mean()); sigma = float(pnls.std(ddof=1)) if n > 1 else 1.0
    tpy = n / max(years, 0.5)
    sharpe = mu / sigma * math.sqrt(tpy) if sigma > 1e-9 else 0.0
    down = pnls[pnls < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else sigma
    sortino = mu / ds * math.sqrt(tpy) if ds > 1e-9 else 0.0
    calmar = cagr / dd if dd > 1e-6 else 0.0
    return {
        "n": n, "total_pnl": round(total, 2),
        "win_rate": round(wins / n, 4), "avg_pnl": round(mu, 2),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "max_dd_pct": round(dd * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "calmar": round(calmar, 2),
        "avg_hold_days": round(float(df["hold_days"].mean()), 1),
    }


# ───────────────────────────────────────────────────────────────────────────
# Walk-forward
# ───────────────────────────────────────────────────────────────────────────

def walk_forward(all_trades: List[Dict]) -> Dict:
    """Expanding train: fit skip_set on trades with exit year < ty, test on ty."""
    df = pd.DataFrame(all_trades)
    df["exit_year"] = pd.to_datetime(df["exit_date"]).dt.year
    test_years = sorted(df["exit_year"].unique())

    folds = []
    pooled_b: List[Dict] = []
    pooled_o: List[Dict] = []

    for ty in test_years:
        train = [t for t in all_trades
                 if pd.to_datetime(t["exit_date"]).year < ty]
        test  = [t for t in all_trades
                 if pd.to_datetime(t["exit_date"]).year == ty]
        if len(train) < 20:
            # Not enough history → no skipping in this fold
            skip = SkipSet([])
        else:
            skip = fit_skip_set(train, quantile=0.25, min_bucket_n=5)

        kept = apply_skip_set(test, skip)
        folds.append({
            "year": int(ty),
            "skip_set": skip.to_serialisable(),
            "n_train": len(train),
            "baseline": trade_metrics(test),
            "overlay":  trade_metrics(kept),
        })
        pooled_b.extend(test)
        pooled_o.extend(kept)

    return {
        "folds": folds,
        "pooled_oos": {
            "baseline": trade_metrics(pooled_b),
            "overlay":  trade_metrics(pooled_o),
        },
    }


# ───────────────────────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────────────────────

def write_html(payload: Dict, path: Path) -> None:
    base = payload["walk_forward"]["pooled_oos"]["baseline"]
    ov   = payload["walk_forward"]["pooled_oos"]["overlay"]
    delta_sh = round(ov["sharpe"] - base["sharpe"], 3)
    color = "#16a34a" if delta_sh >= 0.30 else ("#ca8a04" if delta_sh > 0 else "#dc2626")
    target_msg = "✅ TARGET MET" if delta_sh >= 0.30 else "⚠ Below +0.30 target"

    fold_rows = ""
    for f in payload["walk_forward"]["folds"]:
        b = f["baseline"]; o = f["overlay"]
        ds = round(o["sharpe"] - b["sharpe"], 2)
        c = "#16a34a" if ds > 0 else "#dc2626"
        skip_txt = ", ".join(f"{p['feature']}={p['value']}" for p in f["skip_set"]) or "—"
        fold_rows += (
            f"<tr><td>{f['year']}</td>"
            f"<td>{b['n']}</td><td>{b['sharpe']:.2f}</td>"
            f"<td>{o['n']}</td><td>{o['sharpe']:.2f}</td>"
            f"<td style='color:{c};font-weight:700'>{ds:+.2f}</td>"
            f"<td style='font-size:0.78rem'>{skip_txt}</td></tr>"
        )

    bucket_rows = ""
    for feat, by_val in payload["bucket_table_full"].items():
        for v, m in by_val.items():
            bucket_rows += (
                f"<tr><td>{feat}</td><td>{v}</td>"
                f"<td>{m['n']}</td><td>${m['mean_pnl']:,.0f}</td>"
                f"<td>{m['win_rate']:.0%}</td>"
                f"<td>{m['sharpe']:+.2f}</td></tr>"
            )

    html = f"""<!DOCTYPE html><html><head><meta charset='UTF-8'>
<title>EXP-2030 Seasonality Overlay</title>
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
<h1>EXP-2030 — Intraweek Seasonality Overlay for EXP-1220</h1>
<p class='meta'>Real IronVault SPY options · 2020-2025 · FOMC dates from public Fed calendar.</p>

<div class='headline'><strong>Pooled OOS Sharpe:</strong>
&nbsp;baseline <strong>{base['sharpe']:.2f}</strong>
→ overlay <strong>{ov['sharpe']:.2f}</strong>
&nbsp;|&nbsp; Δ = <strong style='color:{color}'>{delta_sh:+.2f}</strong>
&nbsp;({target_msg} on +0.30 target)</div>

<div class='grid'>
  <div class='card'><div class='l'>Baseline n</div><div class='v'>{base['n']}</div></div>
  <div class='card'><div class='l'>Overlay n</div><div class='v'>{ov['n']}</div></div>
  <div class='card'><div class='l'>Baseline PnL</div><div class='v'>${base['total_pnl']:,.0f}</div></div>
  <div class='card'><div class='l'>Overlay PnL</div><div class='v'>${ov['total_pnl']:,.0f}</div></div>
  <div class='card'><div class='l'>Baseline WR</div><div class='v'>{base['win_rate']:.0%}</div></div>
  <div class='card'><div class='l'>Overlay WR</div><div class='v'>{ov['win_rate']:.0%}</div></div>
  <div class='card'><div class='l'>Baseline DD</div><div class='v'>{base['max_dd_pct']:.1f}%</div></div>
  <div class='card'><div class='l'>Overlay DD</div><div class='v'>{ov['max_dd_pct']:.1f}%</div></div>
</div>

<h2>Walk-Forward Folds (per-year, expanding train)</h2>
<table><tr><th>Year</th><th>B n</th><th>B Sharpe</th>
<th>O n</th><th>O Sharpe</th><th>Δ Sharpe</th><th>Skip set (trained on prior years)</th></tr>
{fold_rows}</table>

<h2>Per-Bucket Stats (full sample, all years pooled)</h2>
<table><tr><th>Feature</th><th>Value</th><th>n</th><th>Mean PnL</th>
<th>Win%</th><th>Bucket Sharpe</th></tr>
{bucket_rows}</table>

<h2>Method</h2>
<ul>
<li>Tag every EXP-1220 entry with: day-of-week, week-of-month,
    is_opex_week (within 4 days of monthly 3rd-Friday OPEX),
    is_fomc_week (within 4 days of an FOMC decision date).</li>
<li>For each fold (test year), fit skip-set on prior trades:
    require ≥5 trades per bucket and bucket Sharpe in worst-25% AND &lt; 0.</li>
<li>Apply skip-set to OOS year. Pool OOS trades across folds for the
    headline result.</li>
<li>FOMC dates are public Fed calendar facts (federalreserve.gov).</li>
</ul>
<div style='color:#94a3b8;font-size:0.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px'>
compass/exp2030_seasonality_overlay.py · ALL REAL DATA</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-2030 — Intraweek Seasonality Overlay")
    print("=" * 60)

    hd = IronVault.instance()

    print("[1/4] Loading SPY + VIX (Yahoo, real)...")
    import yfinance as yf
    spy_df = yf.download("SPY", start=START_DATA, end=END_DATA,
                         progress=False, auto_adjust=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)
    spy_df.index = pd.to_datetime(spy_df.index)
    vix_df = yf.download("^VIX", start=START_DATA, end=END_DATA,
                         progress=False, auto_adjust=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix_close = vix_df["Close"]; vix_close.index = pd.to_datetime(vix_close.index)

    print("[2/4] Running baseline EXP-1220 trades on real IronVault...")
    trades = run_baseline_trades(hd, spy_df, vix_close, BT_START, BT_END)
    print(f"      {len(trades)} trades")

    print("[3/4] Building per-bucket seasonality table (full sample)...")
    bt = bucket_table(trades)
    for k, by_v in bt.items():
        print(f"  {k}:")
        for v, m in by_v.items():
            print(f"    {k}={v:<3} n={m['n']:>3}  WR={m['win_rate']:>5.0%}  "
                  f"meanP=${m['mean_pnl']:>7,.0f}  Sh={m['sharpe']:+.2f}")

    print("[4/4] Walk-forward (expanding train, per-year fold)...")
    wf = walk_forward(trades)

    base_m = wf["pooled_oos"]["baseline"]
    ov_m   = wf["pooled_oos"]["overlay"]
    delta_sh = round(ov_m["sharpe"] - base_m["sharpe"], 3)

    print()
    print("POOLED OOS (walk-forward concatenated)")
    print("-" * 60)
    print(f"{'metric':<14}{'baseline':>14}{'overlay':>14}{'delta':>14}")
    for k in ["sharpe", "cagr_pct", "win_rate", "max_dd_pct", "total_pnl", "n"]:
        bv = base_m.get(k, 0); ov = ov_m.get(k, 0)
        print(f"{k:<14}{bv:>14.3f}{ov:>14.3f}{(ov-bv):>14.3f}")
    print()
    print(f"Δ Sharpe (pooled OOS): {delta_sh:+.3f} | target +0.30 | "
          f"{'✅ MET' if delta_sh>=0.30 else '⚠ MISS'}")
    print()
    print("Per-fold detail:")
    for f in wf["folds"]:
        b = f["baseline"]; o = f["overlay"]
        skip_txt = ", ".join(f"{p['feature']}={p['value']}" for p in f["skip_set"]) or "(none)"
        print(f"  {f['year']}: B[{b['n']:>3}t S={b['sharpe']:+.2f}] "
              f"O[{o['n']:>3}t S={o['sharpe']:+.2f}]  Δ={o['sharpe']-b['sharpe']:+.2f}"
              f"  skip: {skip_txt}")

    payload = {
        "experiment": "EXP-2030",
        "title": "Intraweek Seasonality Overlay",
        "date_range": {"start": BT_START, "end": BT_END},
        "data_sources": {
            "spy_options":  "IronVault options_cache.db (REAL)",
            "spy":          "Yahoo Finance SPY (REAL)",
            "vix":          "Yahoo Finance ^VIX (REAL)",
            "fomc_dates":   "federalreserve.gov public FOMC calendar (REAL)",
            "opex_dates":   "Computed deterministically from calendar (3rd Friday)",
        },
        "n_baseline_trades": len(trades),
        "feature_keys": FEATURE_KEYS,
        "bucket_table_full": bt,
        "walk_forward": wf,
        "delta_sharpe_pooled_oos": delta_sh,
        "target_sharpe_lift": 0.30,
        "target_met": delta_sh >= 0.30,
    }

    out = ROOT / "compass" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp2030_seasonality_overlay.json").write_text(json.dumps(payload, indent=2, default=str))
    write_html(payload, out / "exp2030_seasonality_overlay.html")
    print(f"\nReports → exp2030_seasonality_overlay.{{json,html}}")
    return payload


if __name__ == "__main__":
    main()
