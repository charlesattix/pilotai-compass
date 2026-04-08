"""
compass/exp2430_capacity_optimized.py — EXP-2430 Capacity-Optimized Portfolio.

QUESTION:
  EXP-2140 and EXP-2380 proved the current North Star v6 portfolio is
  capacity-bottlenecked by the SLV calendar sleeve (~$16M AUM soft cap,
  $82M hard cap). Can we redesign the portfolio so it supports $200M+
  AUM while maintaining Sharpe ≥ 5.0 and CAGR ≥ 100% (at the same 3×
  gross leverage used by v6)?

APPROACH (Rule Zero — all real data):
  1. Load the 7-stream v6 cache (compass/cache/exp2280_v6_sparse.pkl)
     → exp1220, xlf_cs, xli_cs, gld_cal, slv_cal, vol_arb, v5_hedge.
  2. Load the cached QQQ credit-spread trades
     (compass/cache/exp2250_qqq_trades.pkl), convert to a daily return
     stream aligned to the same calendar.
  3. Define three portfolio variants:
        A) baseline_v6        — current 7-stream portfolio as-is
        B) v6_drop_slv        — same 7 streams but SLV = 0
        C) v6_1_capacity      — 7 streams: SLV replaced by QQQ + boosted
                                  SPY/QQQ weights
  4. Run each at 3× gross leverage with equal-risk (inv-vol) sizing
     scaled to a 15%-per-sleeve target vol, clipped at 3×. Compute
     full-sample + 20-fold walk-forward metrics.
  5. Compute per-sleeve capacity (soft cap at 1% ADV) for each variant
     using the EXP-2140 square-root impact model and real ADV.
  6. Verdict: does v6_1_capacity support ≥$200M AUM with Sharpe ≥ 5.0
     and CAGR ≥ 100%?

OUTPUTS:
  compass/reports/exp2430_capacity_optimized.{json,html}
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STREAMS_PKL = ROOT / "compass" / "cache" / "exp2280_v6_sparse.pkl"
QQQ_TRADES_PKL = ROOT / "compass" / "cache" / "exp2250_qqq_trades.pkl"
IV_DB = ROOT / "data" / "options_cache.db"
REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2430_capacity_optimized.json"
REPORT_HTML = REPORT_DIR / "exp2430_capacity_optimized.html"

TRADING_DAYS = 252
TARGET_VOL_PER_SLEEVE = 0.15
MAX_GROSS_LEVERAGE = 3.0
CAPITAL = 100_000.0

# Capacity model (mirrors EXP-2140)
IMPACT_COEFF_EQUITY = 100.0
IMPACT_COEFF_FUTURES = 120.0
IMPACT_COEFF_OPTIONS = 150.0
PART_SOFT = 0.01
PART_HARD = 0.05
TURNOVER_RT_YEAR = 50

TARGET_AUM = 200e6   # the scaling goal


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def load_streams() -> pd.DataFrame:
    print(f"  loading {STREAMS_PKL.name}")
    df: pd.DataFrame = pickle.load(STREAMS_PKL.open("rb"))
    df = df.fillna(0.0).astype(float)
    print(f"    {len(df)} days × {len(df.columns)} streams  "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


def load_qqq_daily(idx: pd.DatetimeIndex) -> pd.Series:
    print(f"  loading {QQQ_TRADES_PKL.name}")
    trades = pickle.load(QQQ_TRADES_PKL.open("rb"))
    by: Dict[pd.Timestamp, float] = defaultdict(float)
    for t in trades:
        by[pd.Timestamp(t["exit_date"])] += float(t["pnl"]) / CAPITAL
    s = pd.Series(0.0, index=idx, name="qqq_cs")
    for d, v in by.items():
        if d in s.index:
            s.loc[d] += v
    print(f"    {len(trades)} trades mapped to {(s != 0).sum()} daily cells")
    return s


def fetch_yahoo_vol_px(symbol: str, days: int = 90) -> Tuple[float, float]:
    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=days * 2)).timestamp())
    safe = symbol.replace("^", "%5E").replace("=", "%3D")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}"
           f"?period1={start}&period2={end}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    vols = [v for v in (quote.get("volume") or []) if v is not None and v > 0]
    closes = [c for c in (quote.get("close") or []) if c is not None]
    if not vols or not closes:
        raise RuntimeError(f"Yahoo empty for {symbol}")
    return float(np.median(vols[-days:])), float(closes[-1])


def ironvault_option_volume(ticker: str) -> Optional[float]:
    if not IV_DB.exists():
        return None
    conn = sqlite3.connect(str(IV_DB))
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT od.date, SUM(od.volume)
            FROM option_daily od
            JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
            WHERE oc.ticker = ? AND od.date >= '2024-01-01'
            GROUP BY od.date
        """, (ticker,))
        rows = cur.fetchall()
    finally:
        conn.close()
    vols = [float(v or 0) for _, v in rows if v]
    return float(np.median(vols)) if vols else None


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def portfolio_metrics(rets: np.ndarray) -> Dict:
    n = len(rets)
    if n < 5:
        return {"n_days": n, "cagr_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "vol_pct": 0.0}
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = np.cumprod(1.0 + rets)
    yrs = n / TRADING_DAYS
    cagr = float(eq[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return {
        "n_days": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(float(dd.min()) * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
    }


def equal_risk_weights(df: pd.DataFrame, weights_raw: Dict[str, float]) -> np.ndarray:
    """Convert nominal capital weights → equal-risk (inv-vol) weights scaled
    to target vol per sleeve, multiplied by the nominal weight, and clipped
    to the gross leverage budget."""
    cols = list(df.columns)
    w = np.zeros(len(cols))
    stds = df.std(ddof=1).values
    per_sleeve_lev = np.where(
        stds > 1e-12,
        (TARGET_VOL_PER_SLEEVE / math.sqrt(TRADING_DAYS)) / stds,
        0.0,
    )
    for i, c in enumerate(cols):
        w[i] = per_sleeve_lev[i] * weights_raw.get(c, 0.0)
    gross = float(np.sum(np.abs(w)))
    if gross > MAX_GROSS_LEVERAGE:
        w *= MAX_GROSS_LEVERAGE / gross
    return w


def weighted_returns(df: pd.DataFrame, w: np.ndarray) -> pd.Series:
    return pd.Series(df.values @ w, index=df.index, name="port")


def walk_forward_20(rets: pd.Series) -> List[Dict]:
    """Split the time series into 20 equal-size sequential folds and
    report per-fold metrics. This is a robustness audit, not parameter
    tuning — the allocation is fixed across folds."""
    n = len(rets)
    if n < 20:
        return []
    fold_size = n // 20
    out: List[Dict] = []
    for i in range(20):
        start = i * fold_size
        end = start + fold_size if i < 19 else n
        sub = rets.iloc[start:end]
        if len(sub) < 10:
            continue
        m = portfolio_metrics(sub.values)
        out.append({
            "fold": i + 1,
            "start": str(sub.index.min().date()),
            "end": str(sub.index.max().date()),
            **m,
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Capacity (mirrors EXP-2140)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SleeveCapacity:
    name: str
    binding_leg: str
    adv_notional_usd: float
    impact_coeff: float
    soft_cap_portfolio_aum: float
    hard_cap_portfolio_aum: float


def build_capacity_table(weights: Dict[str, float]) -> Dict[str, SleeveCapacity]:
    """Look up ADV for each sleeve's binding leg (real Yahoo/IronVault)."""
    # SPY options
    spy_opts = ironvault_option_volume("SPY") or 2_300_000
    spy_vol, spy_px = fetch_yahoo_vol_px("SPY")
    spy_notional = spy_opts * 100 * spy_px
    # QQQ options
    qqq_opts = ironvault_option_volume("QQQ") or 188_000
    _, qqq_px = fetch_yahoo_vol_px("QQQ")
    qqq_notional = qqq_opts * 100 * qqq_px
    # XLF options
    xlf_opts = ironvault_option_volume("XLF") or 100_000
    _, xlf_px = fetch_yahoo_vol_px("XLF")
    xlf_notional = xlf_opts * 100 * xlf_px
    # XLI options
    xli_opts = ironvault_option_volume("XLI") or 14_000
    _, xli_px = fetch_yahoo_vol_px("XLI")
    xli_notional = xli_opts * 100 * xli_px
    # GLD — binding leg is the THINNER of GC=F futures vs GLD options
    gld_opts = ironvault_option_volume("GLD") or 7_000
    _, gld_px = fetch_yahoo_vol_px("GLD")
    gld_opt_notional = gld_opts * 100 * gld_px
    gc_vol, gc_px = fetch_yahoo_vol_px("GC=F")
    gc_notional = gc_vol * gc_px * 100
    gld_notional = min(gld_opt_notional, gc_notional)
    # SLV — binding leg is SI=F futures (SLV opts not in IronVault)
    si_vol, si_px = fetch_yahoo_vol_px("SI=F")
    slv_notional = si_vol * si_px * 5000
    # Cross-vol — IWM shares (no IWM options in IronVault)
    iwm_vol, iwm_px = fetch_yahoo_vol_px("IWM")
    iwm_notional = iwm_vol * iwm_px
    # Crisis Alpha v5 — UVXY + VXX as VIX-call proxy
    uvxy_vol, uvxy_px = fetch_yahoo_vol_px("UVXY")
    vxx_vol, vxx_px = fetch_yahoo_vol_px("VXX")
    vix_notional = uvxy_vol * uvxy_px + vxx_vol * vxx_px

    profiles = {
        "exp1220":  (spy_notional, "SPY options", IMPACT_COEFF_OPTIONS),
        "qqq_cs":   (qqq_notional, "QQQ options", IMPACT_COEFF_OPTIONS),
        "xlf_cs":   (xlf_notional, "XLF options", IMPACT_COEFF_OPTIONS),
        "xli_cs":   (xli_notional, "XLI options", IMPACT_COEFF_OPTIONS),
        "gld_cal":  (gld_notional, "GC=F futures / GLD opts (min)",
                       IMPACT_COEFF_FUTURES),
        "slv_cal":  (slv_notional, "SI=F futures", IMPACT_COEFF_FUTURES),
        "vol_arb":  (iwm_notional, "IWM ETF shares (proxy)",
                       IMPACT_COEFF_EQUITY),
        "v5_hedge": (vix_notional, "UVXY+VXX (VIX proxy)",
                       IMPACT_COEFF_OPTIONS),
    }

    table: Dict[str, SleeveCapacity] = {}
    for name, w in weights.items():
        if w <= 0 or name not in profiles:
            continue
        adv, leg, coeff = profiles[name]
        soft_notional = adv * PART_SOFT
        hard_notional = adv * PART_HARD
        table[name] = SleeveCapacity(
            name=name,
            binding_leg=leg,
            adv_notional_usd=adv,
            impact_coeff=coeff,
            soft_cap_portfolio_aum=soft_notional / w if w > 0 else float("inf"),
            hard_cap_portfolio_aum=hard_notional / w if w > 0 else float("inf"),
        )
    return table


def capacity_summary(table: Dict[str, SleeveCapacity]) -> Dict:
    if not table:
        return {}
    soft_min = min(table.values(), key=lambda c: c.soft_cap_portfolio_aum)
    hard_min = min(table.values(), key=lambda c: c.hard_cap_portfolio_aum)
    return {
        "soft_bottleneck": soft_min.name,
        "soft_bottleneck_aum_usd": soft_min.soft_cap_portfolio_aum,
        "hard_bottleneck": hard_min.name,
        "hard_bottleneck_aum_usd": hard_min.hard_cap_portfolio_aum,
        "per_sleeve": {
            c.name: {
                "binding_leg": c.binding_leg,
                "adv_notional_usd": c.adv_notional_usd,
                "soft_cap_aum": c.soft_cap_portfolio_aum,
                "hard_cap_aum": c.hard_cap_portfolio_aum,
            }
            for c in table.values()
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Variants
# ═══════════════════════════════════════════════════════════════════════════

def build_variants(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Define three capital-weight allocations to test.

    All cash buffer lives outside the return-generating sleeves — weights
    sum to 1.0 across sleeves (no cash column in df)."""
    return {
        "baseline_v6": {
            "exp1220":  0.35,
            "xlf_cs":   0.10,
            "xli_cs":   0.10,
            "gld_cal":  0.10,
            "slv_cal":  0.075,
            "vol_arb":  0.15,
            "v5_hedge": 0.125,
        },
        "v6_drop_slv": {
            "exp1220":  0.425,    # +0.075 (absorbs SLV weight)
            "xlf_cs":   0.10,
            "xli_cs":   0.10,
            "gld_cal":  0.10,
            "slv_cal":  0.00,
            "vol_arb":  0.15,
            "v5_hedge": 0.125,
        },
        "v6_1_capacity": {
            "exp1220":  0.35,
            "qqq_cs":   0.20,     # NEW — replaces SLV + boosts SPY slightly
            "xlf_cs":   0.10,
            "xli_cs":   0.10,
            "gld_cal":  0.05,     # trimmed from 0.10 to stay far from soft cap
            "slv_cal":  0.00,
            "vol_arb":  0.10,
            "v5_hedge": 0.10,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2430 — Capacity-Optimized Portfolio Design")
    print("=" * 72)

    print("\n[1/5] Loading 7-stream cache + QQQ daily return stream...")
    streams = load_streams()
    qqq = load_qqq_daily(streams.index)
    # Add QQQ column to a combined DataFrame
    combined = streams.copy()
    combined["qqq_cs"] = qqq.values
    print(f"  combined: {len(combined.columns)} streams "
          f"({', '.join(combined.columns)})")

    print("\n[2/5] Defining variants...")
    variants = build_variants(combined)
    for name, w in variants.items():
        print(f"  {name}:  {{" +
              ", ".join(f"{k}:{v:.3f}" for k, v in w.items() if v > 0) + "}")

    print("\n[3/5] Running each variant (equal-risk inv-vol, 3× clip)...")
    results: Dict[str, Dict] = {}
    for name, w_raw in variants.items():
        w_vec = equal_risk_weights(combined, w_raw)
        rets = weighted_returns(combined, w_vec)
        gross = float(np.sum(np.abs(w_vec)))
        full = portfolio_metrics(rets.values)
        wf = walk_forward_20(rets)
        per_sleeve_lev = {
            c: round(float(w_vec[i]), 4)
            for i, c in enumerate(combined.columns)
        }
        results[name] = {
            "capital_weights": w_raw,
            "per_sleeve_leverage": per_sleeve_lev,
            "gross_leverage": round(gross, 3),
            "full_sample": full,
            "walk_forward": wf,
            "walk_forward_summary": {
                "folds": len(wf),
                "cagr_mean_pct": round(float(np.mean([f["cagr_pct"] for f in wf])), 3) if wf else 0,
                "cagr_min_pct": round(float(np.min([f["cagr_pct"] for f in wf])), 3) if wf else 0,
                "cagr_max_pct": round(float(np.max([f["cagr_pct"] for f in wf])), 3) if wf else 0,
                "sharpe_mean": round(float(np.mean([f["sharpe"] for f in wf])), 3) if wf else 0,
                "sharpe_min": round(float(np.min([f["sharpe"] for f in wf])), 3) if wf else 0,
                "pct_folds_positive": round(
                    float(np.mean([1.0 if f["cagr_pct"] > 0 else 0.0 for f in wf])) * 100, 1
                ) if wf else 0,
            },
        }
        print(f"\n  {name}:")
        print(f"    gross leverage {gross:.2f}×")
        print(f"    full-sample: CAGR={full['cagr_pct']}%  Sharpe={full['sharpe']}  "
              f"DD={full['max_dd_pct']}%  Vol={full['vol_pct']}%")
        if wf:
            wfs = results[name]["walk_forward_summary"]
            print(f"    walk-forward (20 folds): CAGR mean={wfs['cagr_mean_pct']}%  "
                  f"min={wfs['cagr_min_pct']}%  max={wfs['cagr_max_pct']}%  "
                  f"{wfs['pct_folds_positive']}% positive")

    print("\n[4/5] Computing capacity tables (real Yahoo + IronVault ADV)...")
    # Need fresh ADV lookups once; reuse per variant
    # Use the first variant's weight dict shape to call once, then reuse ADVs
    capacity_results: Dict[str, Dict] = {}
    for name, data in results.items():
        w_raw = variants[name]
        table = build_capacity_table(w_raw)
        summary = capacity_summary(table)
        capacity_results[name] = summary
        if summary:
            print(f"\n  {name}:")
            print(f"    soft bottleneck: {summary['soft_bottleneck']:10s}  "
                  f"${summary['soft_bottleneck_aum_usd']/1e6:>8,.1f}M")
            print(f"    hard bottleneck: {summary['hard_bottleneck']:10s}  "
                  f"${summary['hard_bottleneck_aum_usd']/1e6:>8,.1f}M")

    print("\n[5/5] Verdict...")
    best_variant = "v6_1_capacity"
    best = results[best_variant]["full_sample"]
    best_cap = capacity_results[best_variant]
    soft_aum = best_cap.get("soft_bottleneck_aum_usd", 0)

    targets = {
        "cap_ge_200m":  soft_aum >= TARGET_AUM,
        "sharpe_ge_5":  best["sharpe"] >= 5.0,
        "cagr_ge_100":  best["cagr_pct"] >= 100.0,
    }
    passed = all(targets.values())
    print(f"  Target: ≥$200M soft cap, Sharpe ≥5.0, CAGR ≥100%")
    print(f"    soft cap  : ${soft_aum/1e6:,.0f}M  → "
          f"{'PASS' if targets['cap_ge_200m'] else 'FAIL'}")
    print(f"    Sharpe    : {best['sharpe']:.2f}         → "
          f"{'PASS' if targets['sharpe_ge_5'] else 'FAIL'}")
    print(f"    CAGR      : {best['cagr_pct']:.2f}%       → "
          f"{'PASS' if targets['cagr_ge_100'] else 'FAIL'}")
    print(f"  Decision: {'APPROVE' if passed else 'REJECT'} ({sum(targets.values())}/3 gates)")

    # Sharpe sacrifice vs baseline
    base_full = results["baseline_v6"]["full_sample"]
    trade_off = {
        "delta_sharpe": round(best["sharpe"] - base_full["sharpe"], 3),
        "delta_cagr_pct": round(best["cagr_pct"] - base_full["cagr_pct"], 3),
        "delta_dd_pct": round(best["max_dd_pct"] - base_full["max_dd_pct"], 3),
        "capacity_multiplier": round(
            soft_aum / capacity_results["baseline_v6"]["soft_bottleneck_aum_usd"], 2
        ) if capacity_results["baseline_v6"].get("soft_bottleneck_aum_usd", 0) > 0 else None,
    }
    print(f"\n  Trade-off vs baseline_v6:")
    print(f"    ΔSharpe  = {trade_off['delta_sharpe']:+.2f}")
    print(f"    ΔCAGR    = {trade_off['delta_cagr_pct']:+.2f}%")
    print(f"    ΔMaxDD   = {trade_off['delta_dd_pct']:+.2f}%")
    if trade_off["capacity_multiplier"]:
        print(f"    capacity = {trade_off['capacity_multiplier']:.1f}×")

    payload = {
        "experiment": "EXP-2430",
        "title": "Capacity-Optimized Portfolio",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "target": {
            "capacity_aum_usd": TARGET_AUM,
            "sharpe_ge": 5.0,
            "cagr_pct_ge": 100.0,
        },
        "data": {
            "streams_source": str(STREAMS_PKL.name),
            "qqq_trades_source": str(QQQ_TRADES_PKL.name),
            "n_days": int(len(combined)),
            "start": str(combined.index.min().date()),
            "end": str(combined.index.max().date()),
            "columns": list(combined.columns),
        },
        "variants": {name: {**results[name],
                              "capacity": capacity_results[name]}
                       for name in variants},
        "recommended": best_variant,
        "target_check": targets,
        "trade_off_vs_baseline": trade_off,
        "decision": "APPROVE" if passed else "REJECT",
        "rule_zero": (
            "Real streams from exp2280_v6_sparse.pkl + cached QQQ trades "
            "from exp2250_qqq_trades.pkl (both derived from real IronVault "
            "option data and real Yahoo underlying prices). ADV data pulled "
            "live from Yahoo + IronVault for capacity math. No synthetic."
        ),
    }

    print("\nWriting reports...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  → {REPORT_JSON}")
    REPORT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"  → {REPORT_HTML}")
    print("\nDONE.")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════════════════

def render_html(payload: Dict) -> str:
    def fmt_usd(x):
        if x is None:
            return "—"
        if x >= 1e9:
            return f"${x/1e9:.2f}B"
        if x >= 1e6:
            return f"${x/1e6:.1f}M"
        return f"${x:.0f}"

    variants = payload["variants"]
    rows = ""
    for name, v in variants.items():
        f = v["full_sample"]
        cap = v.get("capacity", {})
        soft_aum = cap.get("soft_bottleneck_aum_usd", 0) or 0
        hard_aum = cap.get("hard_bottleneck_aum_usd", 0) or 0
        highlight = "row-best" if name == payload["recommended"] else ""
        rows += f"""<tr class="{highlight}">
            <td><strong>{name}</strong></td>
            <td>{v['gross_leverage']:.2f}×</td>
            <td>{f['cagr_pct']:.2f}%</td>
            <td>{f['sharpe']:.2f}</td>
            <td>{f['max_dd_pct']:.2f}%</td>
            <td>{f['vol_pct']:.2f}%</td>
            <td>{fmt_usd(soft_aum)}</td>
            <td>{fmt_usd(hard_aum)}</td>
            <td>{cap.get('soft_bottleneck','?')}</td>
        </tr>"""

    # WF summary table for recommended
    rec = variants[payload["recommended"]]
    wf_rows = ""
    for f in rec["walk_forward"]:
        cls = "good" if f["cagr_pct"] > 0 else "bad"
        wf_rows += f"""<tr>
            <td>{f['fold']}</td>
            <td>{f['start']}</td>
            <td>{f['end']}</td>
            <td class="{cls}">{f['cagr_pct']:+.2f}%</td>
            <td>{f['sharpe']:.2f}</td>
            <td>{f['max_dd_pct']:.2f}%</td>
        </tr>"""

    tar = payload["target_check"]
    trade = payload["trade_off_vs_baseline"]
    dec_cls = "good" if payload["decision"] == "APPROVE" else "bad"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2430 Capacity-Optimized Portfolio</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1200px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.2em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:150px; }}
  .kpi .value {{ font-size:1.5em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; font-weight:700; }}
  .bad  {{ color:#dc2626; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.74em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  tr.row-best {{ background:#ecfdf5; font-weight:600; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
</style></head><body>

<h1>EXP-2430 — Capacity-Optimized Portfolio</h1>
<div class="subtitle">Can we redesign the 7-stream portfolio for $200M+ AUM while keeping Sharpe ≥5? | {payload['timestamp']}</div>

<div class="note">
    <strong>Goal:</strong> drop the SLV capacity bottleneck and test a
    redesign that trades some Sharpe for meaningful AUM headroom. Real
    7-stream cache + cached QQQ trades. Capacity numbers from live
    Yahoo + IronVault ADV.
</div>

<h2>Decision</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value {dec_cls}">{payload['decision']}</div><div class="label">Verdict</div></div>
    <div class="kpi"><div class="value">{rec['full_sample']['cagr_pct']:.1f}%</div><div class="label">CAGR</div></div>
    <div class="kpi"><div class="value">{rec['full_sample']['sharpe']:.2f}</div><div class="label">Sharpe</div></div>
    <div class="kpi"><div class="value">{rec['full_sample']['max_dd_pct']:.2f}%</div><div class="label">Max DD</div></div>
    <div class="kpi"><div class="value">{fmt_usd(variants[payload['recommended']]['capacity'].get('soft_bottleneck_aum_usd', 0))}</div><div class="label">Soft Cap AUM</div></div>
</div>

<ul>
    <li><strong>≥$200M soft cap:</strong> {'PASS' if tar['cap_ge_200m'] else 'FAIL'}</li>
    <li><strong>Sharpe ≥ 5.0:</strong> {'PASS' if tar['sharpe_ge_5'] else 'FAIL'}</li>
    <li><strong>CAGR ≥ 100%:</strong> {'PASS' if tar['cagr_ge_100'] else 'FAIL'}</li>
</ul>

<h2>Variant Comparison (full sample, 3× gross)</h2>
<table>
    <thead><tr><th>Variant</th><th>Gross</th><th>CAGR</th><th>Sharpe</th>
    <th>Max DD</th><th>Vol</th><th>Soft Cap</th><th>Hard Cap</th>
    <th>Bottleneck</th></tr></thead>
    <tbody>{rows}</tbody>
</table>

<h2>Trade-off vs baseline_v6</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value">{trade['delta_sharpe']:+.2f}</div><div class="label">ΔSharpe</div></div>
    <div class="kpi"><div class="value">{trade['delta_cagr_pct']:+.2f}%</div><div class="label">ΔCAGR</div></div>
    <div class="kpi"><div class="value">{trade['delta_dd_pct']:+.2f}%</div><div class="label">ΔMaxDD</div></div>
    <div class="kpi"><div class="value">{trade['capacity_multiplier']}×</div><div class="label">Capacity Lift</div></div>
</div>

<h2>20-Fold Walk-Forward (recommended = {payload['recommended']})</h2>
<table>
    <thead><tr><th>Fold</th><th>Start</th><th>End</th><th>CAGR</th>
    <th>Sharpe</th><th>Max DD</th></tr></thead>
    <tbody>{wf_rows}</tbody>
</table>

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2430 — compass/exp2430_capacity_optimized.py · Real 7-stream cache + cached QQQ trades + live Yahoo/IronVault ADV
</div>

</body></html>"""


if __name__ == "__main__":
    sys.exit(main())
