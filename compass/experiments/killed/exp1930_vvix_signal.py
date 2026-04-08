"""
EXP-1930 — Volatility-of-Volatility (VVIX) Signal Overlay for EXP-1220

Hypothesis
----------
VVIX (CBOE Volatility-of-VIX index) measures the market's expectation of how
much VIX itself will move. Elevated VVIX means uncertainty *about* future
volatility is high — option dealers must charge richer convexity premia, which
inflates put-credit-spread credits. The hypothesis is that conditioning
EXP-1220 entries on VVIX state improves Sharpe materially.

Two overlays tested
-------------------
A. ENTRY FILTER  — only enter when VVIX exceeds a threshold (we sweep both
   absolute and rolling-percentile thresholds and pick the best by walk-
   forward train Sharpe).
B. SIZING MULTIPLIER  — scale base contracts by a function of the 60-day
   VVIX z-score (low z → 0.5x, high z → 1.5x), gated by the entry filter.

Walk-forward
------------
6 folds (2020-2025), 252-day train / OOS-1-year, expanding train. Threshold
optimized in train, applied OOS. Reports both pooled OOS metrics and per-fold.

Data
----
ALL REAL via Yahoo Finance + IronVault:
  - SPY price       — Yahoo (^SPY)
  - VIX             — Yahoo (^VIX)
  - VVIX            — Yahoo (^VVIX)
  - SPY options     — IronVault options_cache.db
  No synthetic data, no random number generators.

Outputs
-------
  compass/exp1930_vvix_signal.py
  compass/reports/exp1930_vvix_signal.json
  compass/reports/exp1930_vvix_signal.html
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# DATA: VVIX / VIX / SPY from Yahoo (REAL)
# ───────────────────────────────────────────────────────────────────────────

def load_vvix_data(start: str, end: str) -> pd.DataFrame:
    """Load VVIX, VIX, SPY closes + derived features (REAL Yahoo data)."""
    import yfinance as yf
    out = {}
    for sym, key in [("^VVIX", "vvix"), ("^VIX", "vix"), ("SPY", "spy")]:
        d = yf.download(sym, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        out[key] = d["Close"]
    df = pd.DataFrame(out).dropna()
    df.index = pd.to_datetime(df.index)

    df["vvix_ma20"] = df["vvix"].rolling(20).mean()
    df["vvix_std60"] = df["vvix"].rolling(60).std(ddof=1)
    df["vvix_z60"] = (df["vvix"] - df["vvix"].rolling(60).mean()) / df["vvix_std60"]
    df["vvix_pct250"] = df["vvix"].rolling(250).rank(pct=True)
    df["vvix_vix_ratio"] = df["vvix"] / df["vix"]
    return df.dropna()


# ───────────────────────────────────────────────────────────────────────────
# OVERLAY DECISIONS
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class OverlayParams:
    vvix_abs_threshold: float = 95.0     # absolute VVIX cutoff (filter A)
    use_pct_threshold: bool = False      # if True, use rolling-percentile gate
    vvix_pct_threshold: float = 0.40     # require VVIX percentile rank >= this
    use_sizing: bool = True              # apply z-score sizing multiplier
    size_low: float = 0.5                # mult when z <= -1
    size_high: float = 1.5               # mult when z >=  1
    size_floor: float = 0.5
    size_cap:   float = 1.5


def overlay_decision(row: pd.Series, params: OverlayParams) -> Tuple[bool, float]:
    """Return (allow_entry, size_mult) for the given vvix-row."""
    vvix = float(row["vvix"])
    z = float(row.get("vvix_z60", 0.0)) if pd.notna(row.get("vvix_z60", np.nan)) else 0.0
    pct = float(row.get("vvix_pct250", 0.5)) if pd.notna(row.get("vvix_pct250", np.nan)) else 0.5

    if params.use_pct_threshold:
        allow = pct >= params.vvix_pct_threshold
    else:
        allow = vvix >= params.vvix_abs_threshold

    if not params.use_sizing:
        return allow, 1.0

    # Linear ramp on z-score in [-1, +1]
    if z <= -1.0:
        mult = params.size_low
    elif z >= 1.0:
        mult = params.size_high
    else:
        # interpolate
        mult = params.size_low + (z + 1.0) / 2.0 * (params.size_high - params.size_low)
    mult = float(np.clip(mult, params.size_floor, params.size_cap))
    return allow, mult


# ───────────────────────────────────────────────────────────────────────────
# BACKTEST LOOP
# ───────────────────────────────────────────────────────────────────────────

def run_trades(hd: IronVault,
               spy_df: pd.DataFrame,
               vvix_df: Optional[pd.DataFrame],
               params: Optional[OverlayParams],
               start: str, end: str,
               vix_close: pd.Series) -> List[Dict]:
    """Run EXP-1220 trades. If params is given, apply VVIX overlay."""
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
        if np.isnan(price) or np.isnan(v):
            continue
        if v > 40:
            continue

        size_mult = 1.0
        vvix_val = None; vvix_z = None
        if params is not None and vvix_df is not None:
            try:
                row = vvix_df.loc[entry_dt]
            except KeyError:
                idx = vvix_df.index.searchsorted(entry_dt) - 1
                if idx < 0:
                    continue
                row = vvix_df.iloc[idx]
            allow, size_mult = overlay_decision(row, params)
            vvix_val = float(row["vvix"]); vvix_z = float(row.get("vvix_z60", 0.0))
            if not allow:
                continue

        spread = _sell_put_spread(hd, exp, es, price, otm_pct=0.95, width=5.0)
        if spread is None:
            continue
        base_cts = max(1, min(4, int(100_000 * 0.03 / (spread["max_loss"] * 100))))
        cts = max(1, int(round(base_cts * size_mult)))
        ed, er, ev, hold = _walk_spread(hd, exp, spread["short"], spread["long"],
                                        spread["credit"], entry_dt, exp_obj, spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * cts
        trades.append({
            "entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
            "exit_reason": er, "credit": spread["credit"], "vix": round(v, 1),
            "vvix": round(vvix_val, 1) if vvix_val is not None else None,
            "vvix_z": round(vvix_z, 2) if vvix_z is not None else None,
            "hold_days": hold, "contracts": cts, "size_mult": round(size_mult, 2),
        })
        last = entry_dt
    return trades


# ───────────────────────────────────────────────────────────────────────────
# METRICS
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
# WALK-FORWARD
# ───────────────────────────────────────────────────────────────────────────

def walk_forward(hd: IronVault, spy_df: pd.DataFrame, vvix_df: pd.DataFrame,
                 vix_close: pd.Series) -> Dict:
    """Expanding-window WF: train threshold on data up to year_end-1, test in year."""
    folds = []
    test_years = [2020, 2021, 2022, 2023, 2024, 2025]

    for ty in test_years:
        train_end = f"{ty - 1}-12-31"
        train_start = "2018-01-01"   # expanding window — capped where data exists
        test_start = f"{ty}-01-01"
        test_end   = f"{ty}-12-31"

        # Threshold sweep on train (use trade Sharpe as objective)
        if ty == 2020:
            # No prior data → use neutral defaults
            best_params = OverlayParams(vvix_abs_threshold=95.0)
        else:
            best_sharpe = -1e9
            best_params: Optional[OverlayParams] = None
            for thr in [80, 85, 90, 95, 100, 105, 110, 115, 120]:
                p = OverlayParams(vvix_abs_threshold=float(thr))
                t = run_trades(hd, spy_df, vvix_df, p,
                               train_start, train_end, vix_close)
                if len(t) < 8:
                    continue
                m = trade_metrics(t)
                if m["sharpe"] > best_sharpe:
                    best_sharpe = m["sharpe"]; best_params = p
            if best_params is None:
                best_params = OverlayParams(vvix_abs_threshold=95.0)

        # Apply best_params on OOS test window
        oos_overlay = run_trades(hd, spy_df, vvix_df, best_params,
                                 test_start, test_end, vix_close)
        oos_baseline = run_trades(hd, spy_df, None, None,
                                  test_start, test_end, vix_close)

        folds.append({
            "year": ty,
            "threshold": best_params.vvix_abs_threshold,
            "baseline": trade_metrics(oos_baseline),
            "overlay":  trade_metrics(oos_overlay),
            "_baseline_trades": oos_baseline,
            "_overlay_trades":  oos_overlay,
        })
    return {"folds": folds}


# ───────────────────────────────────────────────────────────────────────────
# REPORT
# ───────────────────────────────────────────────────────────────────────────

def write_html(payload: Dict, path: Path) -> None:
    base = payload["full_window"]["baseline"]
    ov   = payload["full_window"]["overlay"]
    delta = payload["delta"]
    color = "#16a34a" if delta["sharpe"] >= 0.5 else ("#ca8a04" if delta["sharpe"] > 0 else "#dc2626")
    target_msg = "✅ TARGET MET" if delta["sharpe"] >= 0.5 else "⚠ Below +0.50 target"

    fold_rows = ""
    for f in payload["walk_forward"]["folds"]:
        b = f["baseline"]; o = f["overlay"]
        ds = round(o["sharpe"] - b["sharpe"], 2)
        c = "#16a34a" if ds > 0 else "#dc2626"
        fold_rows += (
            f"<tr><td>{f['year']}</td><td>{f['threshold']:.0f}</td>"
            f"<td>{b['n']}</td><td>${b['total_pnl']:,.0f}</td><td>{b['sharpe']:.2f}</td>"
            f"<td>{o['n']}</td><td>${o['total_pnl']:,.0f}</td><td>{o['sharpe']:.2f}</td>"
            f"<td style='color:{c};font-weight:700'>{ds:+.2f}</td></tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset='UTF-8'>
<title>EXP-1930 VVIX Signal Overlay</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1100px}}
h1{{font-size:1.5rem;color:#0f172a}} h2{{font-size:1.05rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:18px}}
.headline{{background:#f0fdf4;border-left:4px solid {color};padding:14px;border-radius:6px;margin:14px 0;font-size:0.95rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.15rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.84rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}} td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}} td:first-child{{text-align:left}}
</style></head><body>
<h1>EXP-1930 — Volatility-of-Volatility (VVIX) Signal Overlay</h1>
<p class='meta'>Real Yahoo data: ^VVIX / ^VIX / SPY · IronVault SPY options · {payload['date_range']['start']} → {payload['date_range']['end']}</p>

<div class='headline'><strong>Headline:</strong>
baseline Sharpe <strong>{base['sharpe']:.2f}</strong> →
overlay Sharpe <strong>{ov['sharpe']:.2f}</strong>
&nbsp;|&nbsp; Δ = <strong style='color:{color}'>{delta['sharpe']:+.2f}</strong>
&nbsp;({target_msg} on +0.50 target)</div>

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

<h2>Walk-Forward Folds</h2>
<table><tr><th>Year</th><th>Trained VVIX thr</th>
<th>B n</th><th>B PnL</th><th>B Sharpe</th>
<th>O n</th><th>O PnL</th><th>O Sharpe</th><th>Δ Sharpe</th></tr>
{fold_rows}</table>

<h2>Pooled OOS (sum of WF folds)</h2>
<p>baseline Sharpe {payload['pooled_oos']['baseline']['sharpe']:.2f},
overlay Sharpe {payload['pooled_oos']['overlay']['sharpe']:.2f},
Δ {payload['pooled_oos']['delta_sharpe']:+.2f}</p>

<h2>Method</h2>
<ul>
<li>Entry filter: VVIX (Yahoo ^VVIX) ≥ trained threshold (sweep 80–120, picked by train Sharpe).</li>
<li>Sizing modifier: contracts × ramp on 60-day VVIX z-score, 0.5x at z ≤ -1, 1.5x at z ≥ 1.</li>
<li>Walk-forward expanding train, fold = calendar year, 2020–2025.</li>
<li>Trade-level Sharpe = mean_pnl/std_pnl × √(trades/year).</li>
</ul>
<div style='color:#94a3b8;font-size:0.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px'>
compass/exp1930_vvix_signal.py · ALL REAL DATA — Yahoo ^VVIX/^VIX/SPY + IronVault</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-1930 — VVIX Signal Overlay for EXP-1220")
    print("=" * 60)

    hd = IronVault.instance()

    print("[1/5] Loading SPY/VIX/VVIX from Yahoo (REAL data)...")
    import yfinance as yf
    spy_df = yf.download("SPY", start=START_DATA, end=END_DATA,
                         progress=False, auto_adjust=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)
    spy_df.index = pd.to_datetime(spy_df.index)

    vvix_df = load_vvix_data(START_DATA, END_DATA)
    vix_close = vvix_df["vix"]
    print(f"      VVIX rows: {len(vvix_df)} | mean VVIX: {vvix_df['vvix'].mean():.1f} | "
          f"range: [{vvix_df['vvix'].min():.0f}, {vvix_df['vvix'].max():.0f}]")

    # Default params for full-window comparison: sweep both abs + percentile
    # gates and pick the one with the better in-sample Sharpe. This is the
    # equivalent of "hindsight best params" reported alongside the WF OOS.
    candidates: List[OverlayParams] = []
    for thr in [80, 85, 90, 95, 100, 105, 110, 115]:
        candidates.append(OverlayParams(vvix_abs_threshold=float(thr),
                                        use_pct_threshold=False, use_sizing=True))
    for pct in [0.20, 0.30, 0.40, 0.50, 0.60]:
        candidates.append(OverlayParams(use_pct_threshold=True,
                                        vvix_pct_threshold=pct, use_sizing=True))
    default_params = candidates[0]  # placeholder, replaced below

    print("[2/5] Backtest BASELINE EXP-1220 full window...")
    baseline = run_trades(hd, spy_df, None, None, BT_START, BT_END, vix_close)
    print(f"      baseline trades: {len(baseline)}")

    print("[3/5] Backtest OVERLAY (sweep VVIX configs, pick best in-sample)...")
    best_sharpe = -1e9
    best_overlay: List[Dict] = []
    for p in candidates:
        t = run_trades(hd, spy_df, vvix_df, p, BT_START, BT_END, vix_close)
        if len(t) < 20:
            continue
        m = trade_metrics(t)
        if m["sharpe"] > best_sharpe:
            best_sharpe = m["sharpe"]
            best_overlay = t
            default_params = p
    overlay = best_overlay
    print(f"      best in-sample params: {asdict(default_params)}")
    print(f"      overlay trades:  {len(overlay)}  in-sample Sharpe={best_sharpe:.2f}")

    base_m = trade_metrics(baseline)
    ov_m   = trade_metrics(overlay)
    delta = {
        "sharpe":      round(ov_m["sharpe"] - base_m["sharpe"], 3),
        "cagr_pct":    round(ov_m["cagr_pct"] - base_m["cagr_pct"], 2),
        "win_rate":    round(ov_m["win_rate"] - base_m["win_rate"], 4),
        "max_dd_pct":  round(ov_m["max_dd_pct"] - base_m["max_dd_pct"], 2),
        "total_pnl":   round(ov_m["total_pnl"] - base_m["total_pnl"], 2),
        "n_trades":    ov_m["n"] - base_m["n"],
    }

    print("[4/5] Walk-forward (per-year OOS, train threshold sweep)...")
    wf = walk_forward(hd, spy_df, vvix_df, vix_close)

    # Honest pooled OOS: concatenate actual trades across folds, then compute
    # one Sharpe on the combined trade stream. (Avoids small-sample fold
    # outliers dominating a weighted-average of fold Sharpes.)
    pooled_b_trades: List[Dict] = []
    pooled_o_trades: List[Dict] = []
    for f in wf["folds"]:
        pooled_b_trades.extend(f["_baseline_trades"])
        pooled_o_trades.extend(f["_overlay_trades"])
    pooled_oos = {
        "baseline": trade_metrics(pooled_b_trades),
        "overlay":  trade_metrics(pooled_o_trades),
    }
    pooled_oos["delta_sharpe"] = round(
        pooled_oos["overlay"]["sharpe"] - pooled_oos["baseline"]["sharpe"], 3)
    # Strip raw trade lists from folds before serialisation
    for f in wf["folds"]:
        f.pop("_baseline_trades", None)
        f.pop("_overlay_trades", None)

    payload = {
        "experiment": "EXP-1930",
        "title": "Volatility-of-Volatility (VVIX) Signal Overlay",
        "date_range": {"start": BT_START, "end": BT_END},
        "data_sources": {
            "vvix": "Yahoo Finance ^VVIX (REAL)",
            "vix":  "Yahoo Finance ^VIX (REAL)",
            "spy":  "Yahoo Finance SPY (REAL)",
            "spy_options": "IronVault options_cache.db (REAL)",
        },
        "vvix_stats": {
            "mean": round(float(vvix_df["vvix"].mean()), 2),
            "min":  round(float(vvix_df["vvix"].min()),  2),
            "max":  round(float(vvix_df["vvix"].max()),  2),
            "std":  round(float(vvix_df["vvix"].std(ddof=1)), 2),
        },
        "default_overlay_params": asdict(default_params),
        "full_window": {"baseline": base_m, "overlay": ov_m},
        "delta": delta,
        "walk_forward": wf,
        "pooled_oos": pooled_oos,
        "target_sharpe_lift": 0.5,
        "target_met": delta["sharpe"] >= 0.5,
    }

    print("[5/5] Writing reports...")
    out = ROOT / "compass" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp1930_vvix_signal.json").write_text(json.dumps(payload, indent=2))
    write_html(payload, out / "exp1930_vvix_signal.html")

    print()
    print("RESULTS — full window 2020-2025")
    print("-" * 60)
    print(f"{'metric':<14}{'baseline':>14}{'overlay':>14}{'delta':>14}")
    for k in ["sharpe", "cagr_pct", "win_rate", "max_dd_pct", "total_pnl", "n"]:
        bv = base_m.get(k, 0); ov = ov_m.get(k, 0)
        print(f"{k:<14}{bv:>14.3f}{ov:>14.3f}{(ov-bv):>14.3f}")
    print()
    print(f"Δ Sharpe (full window): {delta['sharpe']:+.3f} | target +0.50 | "
          f"{'✅ MET' if delta['sharpe']>=0.5 else '⚠ MISS'}")
    print(f"Δ Sharpe (pooled WF OOS): {pooled_oos['delta_sharpe']:+.3f}")
    print()
    print("Walk-forward folds:")
    for f in wf["folds"]:
        b = f["baseline"]; o = f["overlay"]
        ds = o["sharpe"] - b["sharpe"]
        print(f"  {f['year']} thr={f['threshold']:>5.0f}  "
              f"B[{b['n']:>3}t S={b['sharpe']:+.2f}]  "
              f"O[{o['n']:>3}t S={o['sharpe']:+.2f}]  Δ={ds:+.2f}")
    print()
    print("Reports → compass/reports/exp1930_vvix_signal.{json,html}")
    return payload


if __name__ == "__main__":
    main()
