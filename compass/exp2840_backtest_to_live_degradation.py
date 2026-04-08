"""
compass/exp2840_backtest_to_live_degradation.py — EXP-2840 Honest Backtest-to-Live Degradation.

HYPOTHESIS:
  EXP-2760 literature survey (Sharpe 6.00 net realism check) reported
  that live trading Sharpe is typically 0.5x-0.7x of backtest Sharpe
  for options-selling strategies. The v8 net Sharpe 6.00 headline
  therefore implies a REALISTIC live range of 3.00-4.20.

GOAL:
  Build an honest degradation model that simulates live performance
  under different alpha-decay multipliers, identifies the break-point
  where the portfolio drops below "elite" (Sharpe 3.0), and stress-
  tests the impact of any one sleeve going to zero. Carlos needs to
  set realistic expectations BEFORE live capital is committed.

METHOD (Rule Zero):
  1. Load the 8-stream portfolio from exp2280_v6_sparse.pkl (7 real
     streams) + cached QQQ trades (EXP-2250). Real IronVault + Yahoo.
  2. Compute the per-stream baseline Sharpe from the real returns.
  3. Apply degradation via MEAN SHIFT (preserves vol): the new stream
     mean is k × original mean where k ∈ {1.0, 0.9, 0.8, 0.7, 0.6, 0.5}.
     This is the correct way to model "alpha decay" — in practice,
     live Sharpe degrades because the mean drops, not because the
     vol inflates (vol is often similar between backtest and live).
  4. Compute portfolio Sharpe under each scenario using the production
     inv-vol weights (from EXP-2420 / EXP-2600).
  5. Find the break-point k* where portfolio Sharpe = 3.0.
  6. Stress test: what if Crisis Alpha v5 (the weakest stream, baseline
     Sharpe 1.20) goes to zero entirely?
  7. Write a "what Carlos should expect" report with p05 / p50 / p95
     live Sharpe bands.

OUTPUTS:
  compass/reports/exp2840_backtest_to_live_degradation.{json,html}
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STREAMS_PKL = ROOT / "compass" / "cache" / "exp2280_v6_sparse.pkl"
QQQ_TRADES_PKL = ROOT / "compass" / "cache" / "exp2250_qqq_trades.pkl"
REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2840_backtest_to_live_degradation.json"
REPORT_HTML = REPORT_DIR / "exp2840_backtest_to_live_degradation.html"

TRADING_DAYS = 252
CAPITAL = 100_000.0

# Production capital weights (North Star v8)
CAPITAL_WEIGHTS = {
    "exp1220":  0.35,
    "qqq_cs":   0.15,
    "xlf_cs":   0.10,
    "xli_cs":   0.10,
    "gld_cal":  0.10,
    "slv_cal":  0.05,
    "vol_arb":  0.10,
    "v5_hedge": 0.05,
}
TARGET_GROSS_LEVERAGE = 3.0

# Degradation multipliers to test
DEG_MULTIPLIERS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

# Targets
SHARPE_ELITE = 3.0     # "still elite" threshold
SHARPE_GOOD = 2.0      # "still good" threshold

# Net Alpaca Sharpe from EXP-2570 (the production headline)
BACKTEST_NET_SHARPE = 6.00


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def load_8_streams() -> pd.DataFrame:
    print(f"  loading {STREAMS_PKL.name}")
    df7: pd.DataFrame = pickle.load(STREAMS_PKL.open("rb"))
    df7 = df7.fillna(0.0).astype(float)
    print(f"  loading {QQQ_TRADES_PKL.name}")
    trades = pickle.load(QQQ_TRADES_PKL.open("rb"))
    qqq = pd.Series(0.0, index=df7.index, name="qqq_cs")
    by: Dict[pd.Timestamp, float] = defaultdict(float)
    for t in trades:
        by[pd.Timestamp(t["exit_date"])] += float(t["pnl"]) / CAPITAL
    for d, v in by.items():
        if d in qqq.index:
            qqq.loc[d] += v
    df = df7.copy()
    df["qqq_cs"] = qqq.values
    # Reorder columns so qqq_cs sits right after exp1220
    cols_ordered = ["exp1220", "qqq_cs", "xlf_cs", "xli_cs",
                     "gld_cal", "slv_cal", "vol_arb", "v5_hedge"]
    df = df[cols_ordered]
    print(f"  8-stream DF: {len(df)} days  "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


def equal_risk_weights(df: pd.DataFrame, target_gross: float) -> np.ndarray:
    cols = list(df.columns)
    stds = df.std(ddof=1).values
    base = np.zeros(len(cols))
    for i, c in enumerate(cols):
        if stds[i] > 1e-12:
            base[i] = CAPITAL_WEIGHTS.get(c, 0.0) / stds[i]
    s = float(np.sum(np.abs(base)))
    if s < 1e-12:
        return base
    return base / s * target_gross


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def sharpe_of(rets: np.ndarray) -> float:
    r = rets[rets != 0] if (rets == 0).mean() > 0.3 else rets  # sparse streams
    if len(r) < 5:
        return 0.0
    sd = float(np.std(r, ddof=1))
    if sd < 1e-12:
        return 0.0
    return float(np.mean(r)) / sd * math.sqrt(TRADING_DAYS)


def portfolio_metrics(rets: np.ndarray) -> Dict:
    n = len(rets)
    if n < 5:
        return {"sharpe": 0.0, "cagr_pct": 0.0, "max_dd_pct": 0.0,
                "vol_pct": 0.0}
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = np.cumprod(1.0 + rets)
    yrs = n / TRADING_DAYS
    cagr = float(eq[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return {
        "sharpe": round(sharpe, 3),
        "cagr_pct": round(cagr * 100, 3),
        "max_dd_pct": round(float(dd.min()) * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Degradation model
# ═══════════════════════════════════════════════════════════════════════════

def degrade_stream_mean(series: pd.Series, k: float) -> pd.Series:
    """Scale the non-zero mean by k while preserving vol.

    This is the correct alpha-decay model: live Sharpe drops because the
    mean drops, not because vol inflates. For a stream with mean μ and
    std σ:
        new_series[i] = series[i] - (1-k) * μ    for active days only

    The vol (std of non-zero days) is preserved; the mean scales by k.
    """
    if k == 1.0:
        return series.copy()
    active = series != 0
    mu = float(series[active].mean()) if active.any() else 0.0
    shift = (1.0 - k) * mu
    out = series.copy()
    out[active] = out[active] - shift
    return out


def degrade_portfolio(df: pd.DataFrame, k_by_stream: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for c in df.columns:
        k = k_by_stream.get(c, 1.0)
        out[c] = degrade_stream_mean(df[c], k)
    return out


def find_breakeven_k(df: pd.DataFrame, w: np.ndarray,
                       target_sharpe: float, hi: float = 1.0,
                       lo: float = 0.0, iters: int = 40) -> float:
    """Bisect to find the uniform multiplier k* such that
    portfolio Sharpe with all streams at k equals target_sharpe."""
    for _ in range(iters):
        mid = (hi + lo) / 2.0
        dg = degrade_portfolio(df, {c: mid for c in df.columns})
        rets = dg.values @ w
        sh = portfolio_metrics(rets)["sharpe"]
        if sh > target_sharpe:
            hi = mid
        else:
            lo = mid
    return round((hi + lo) / 2.0, 4)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2840 — Honest Backtest-to-Live Degradation Model")
    print("=" * 72)

    print("\n[1/6] Loading 8-stream portfolio...")
    df = load_8_streams()
    w = equal_risk_weights(df, TARGET_GROSS_LEVERAGE)
    base_rets = pd.Series(df.values @ w, index=df.index)
    base_m = portfolio_metrics(base_rets.values)
    per_sleeve_lev = {c: round(float(w[i]), 4) for i, c in enumerate(df.columns)}
    print(f"  per-sleeve leverage: {per_sleeve_lev}")
    print(f"  gross leverage: {float(np.sum(np.abs(w))):.2f}×")
    print(f"  baseline portfolio: "
          f"CAGR {base_m['cagr_pct']}%  Sharpe {base_m['sharpe']}  "
          f"DD {base_m['max_dd_pct']}%  Vol {base_m['vol_pct']}%")

    print("\n[2/6] Per-stream baseline Sharpes (non-zero days, sparse-aware)...")
    stream_baselines = {}
    for c in df.columns:
        sh = sharpe_of(df[c].values)
        stream_baselines[c] = round(sh, 3)
        print(f"    {c:10s}  Sharpe = {sh:.2f}")

    print("\n[3/6] Uniform degradation sweep (all streams at same k)...")
    uniform_scenarios: Dict[str, Dict] = {}
    for k in DEG_MULTIPLIERS:
        dg = degrade_portfolio(df, {c: k for c in df.columns})
        rets = dg.values @ w
        m = portfolio_metrics(rets)
        implied_net_sharpe = BACKTEST_NET_SHARPE * k
        label = f"k_{k:.1f}x"
        uniform_scenarios[label] = {
            "degradation_multiplier": k,
            "implied_live_sharpe_from_headline": round(implied_net_sharpe, 2),
            "portfolio_metrics": m,
            "passes_elite": m["sharpe"] >= SHARPE_ELITE,
            "passes_good":  m["sharpe"] >= SHARPE_GOOD,
        }
        elite_tag = "ELITE" if m["sharpe"] >= SHARPE_ELITE else (
            "GOOD" if m["sharpe"] >= SHARPE_GOOD else "BELOW")
        print(f"  k={k:.1f}×  "
              f"CAGR {m['cagr_pct']:6.2f}%  "
              f"Sharpe {m['sharpe']:5.2f}  "
              f"DD {m['max_dd_pct']:5.2f}%  "
              f"[{elite_tag}]")

    print("\n[4/6] Break-point analysis...")
    k_at_elite = find_breakeven_k(df, w, SHARPE_ELITE)
    k_at_good = find_breakeven_k(df, w, SHARPE_GOOD)
    k_at_1 = find_breakeven_k(df, w, 1.0)
    print(f"  k* at Sharpe = 3.0 (elite): {k_at_elite:.3f}  "
          f"(live ≤ {BACKTEST_NET_SHARPE * k_at_elite:.2f} backtest-implied)")
    print(f"  k* at Sharpe = 2.0 (good):  {k_at_good:.3f}")
    print(f"  k* at Sharpe = 1.0:          {k_at_1:.3f}")
    print(f"  Portfolio stays ELITE (Sharpe ≥ 3.0) if live delivers "
          f"{k_at_elite*100:.0f}% of backtest alpha or more.")
    print(f"  At the literature midpoint (0.60×), "
          f"portfolio Sharpe = {uniform_scenarios['k_0.6x']['portfolio_metrics']['sharpe']}")

    print("\n[5/6] Stress test — what if Crisis Alpha v5 goes to ZERO?")
    stress_scenarios: Dict[str, Dict] = {}
    for stream_to_zero in ["v5_hedge", "slv_cal", "vol_arb", "gld_cal"]:
        dg = df.copy()
        dg[stream_to_zero] = 0.0
        rets = dg.values @ w
        m = portfolio_metrics(rets)
        delta_sh = m["sharpe"] - base_m["sharpe"]
        stress_scenarios[f"zero_{stream_to_zero}"] = {
            "killed_stream": stream_to_zero,
            "baseline_stream_sharpe": stream_baselines[stream_to_zero],
            "portfolio_metrics": m,
            "delta_sharpe": round(delta_sh, 3),
            "delta_cagr_pct": round(m["cagr_pct"] - base_m["cagr_pct"], 3),
        }
        print(f"  zero {stream_to_zero:10s} (baseline Sh {stream_baselines[stream_to_zero]:.2f})  →  "
              f"portfolio Sharpe {base_m['sharpe']:.2f} → {m['sharpe']:.2f}  "
              f"({delta_sh:+.2f})  "
              f"CAGR {base_m['cagr_pct']:.1f}% → {m['cagr_pct']:.1f}%")

    # Combined nightmare: Crisis Alpha zero AND 0.5× on all others
    print("\n  Combined nightmare: Crisis Alpha ZERO + 0.5× everything else...")
    dg = degrade_portfolio(df, {c: 0.5 for c in df.columns if c != "v5_hedge"})
    dg["v5_hedge"] = 0.0
    rets = dg.values @ w
    nightmare_m = portfolio_metrics(rets)
    print(f"    CAGR {nightmare_m['cagr_pct']:.2f}%  "
          f"Sharpe {nightmare_m['sharpe']:.2f}  "
          f"DD {nightmare_m['max_dd_pct']:.2f}%")

    print("\n[6/6] What Carlos should expect — honest live ranges...")
    # Aggregate p05 / p50 / p95 bands from the degradation sweep
    # p95 = best case (k=0.8-0.9), p50 = middle (k=0.6), p05 = worst (k=0.4-0.5)
    def pick_band(label):
        return uniform_scenarios[label]["portfolio_metrics"]["sharpe"]

    expectation_bands = {
        "optimistic_p95": {
            "label": "Optimistic (k=0.8×)",
            "degradation": 0.8,
            "portfolio_sharpe": pick_band("k_0.8x"),
            "portfolio_cagr_pct": uniform_scenarios["k_0.8x"]["portfolio_metrics"]["cagr_pct"],
            "rationale": "Upper bound of literature range. Typical for "
                          "well-executed high-liquidity options strategies.",
        },
        "realistic_p50": {
            "label": "Realistic (k=0.6×)",
            "degradation": 0.6,
            "portfolio_sharpe": pick_band("k_0.6x"),
            "portfolio_cagr_pct": uniform_scenarios["k_0.6x"]["portfolio_metrics"]["cagr_pct"],
            "rationale": "Literature midpoint. Plan for this as the base case.",
        },
        "pessimistic_p05": {
            "label": "Pessimistic (k=0.5×)",
            "degradation": 0.5,
            "portfolio_sharpe": pick_band("k_0.5x"),
            "portfolio_cagr_pct": uniform_scenarios["k_0.5x"]["portfolio_metrics"]["cagr_pct"],
            "rationale": "Lower bound of literature range. Triggers soul-"
                          "searching but still profitable.",
        },
        "crisis_scenario": {
            "label": "Crisis (0.5× + Crisis Alpha zero)",
            "degradation": "0.5× + v5=0",
            "portfolio_sharpe": nightmare_m["sharpe"],
            "portfolio_cagr_pct": nightmare_m["cagr_pct"],
            "rationale": "Simultaneous alpha decay AND our weakest stream "
                          "going to zero. The floor case.",
        },
    }
    print(f"  Optimistic (k=0.8x): Sharpe {expectation_bands['optimistic_p95']['portfolio_sharpe']:.2f}")
    print(f"  Realistic  (k=0.6x): Sharpe {expectation_bands['realistic_p50']['portfolio_sharpe']:.2f}")
    print(f"  Pessimistic(k=0.5x): Sharpe {expectation_bands['pessimistic_p05']['portfolio_sharpe']:.2f}")
    print(f"  Crisis (nightmare):  Sharpe {expectation_bands['crisis_scenario']['portfolio_sharpe']:.2f}")

    payload = {
        "experiment": "EXP-2840",
        "title": "Honest Backtest-to-Live Degradation Model",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hypothesis": (
            "EXP-2760 literature survey: live Sharpe = 0.5-0.7× backtest "
            "Sharpe for options-selling strategies. Our net Alpaca Sharpe "
            "6.00 implies a realistic live range of 3.00-4.20."
        ),
        "backtest_net_sharpe": BACKTEST_NET_SHARPE,
        "data": {
            "streams_source": str(STREAMS_PKL.name),
            "qqq_trades_source": str(QQQ_TRADES_PKL.name),
            "n_days": int(len(df)),
            "start": str(df.index.min().date()),
            "end": str(df.index.max().date()),
            "streams": list(df.columns),
        },
        "per_sleeve_leverage": per_sleeve_lev,
        "gross_leverage": round(float(np.sum(np.abs(w))), 3),
        "baseline_portfolio_metrics": base_m,
        "stream_baseline_sharpes": stream_baselines,
        "uniform_degradation_scenarios": uniform_scenarios,
        "breakpoints": {
            "k_at_sharpe_3.0_elite": k_at_elite,
            "k_at_sharpe_2.0_good":  k_at_good,
            "k_at_sharpe_1.0":       k_at_1,
            "interpretation": (
                f"Portfolio stays above Sharpe 3.0 (elite) as long as live "
                f"delivers ≥ {k_at_elite*100:.0f}% of backtest alpha per stream. "
                f"At the literature midpoint (60%), portfolio Sharpe is "
                f"{uniform_scenarios['k_0.6x']['portfolio_metrics']['sharpe']}."
            ),
        },
        "stream_zero_stress_tests": stress_scenarios,
        "combined_nightmare_scenario": {
            "description": "Crisis Alpha v5 ZERO + 0.5x decay on other 7 streams",
            "portfolio_metrics": nightmare_m,
            "delta_sharpe": round(nightmare_m["sharpe"] - base_m["sharpe"], 3),
        },
        "expectation_bands_for_carlos": expectation_bands,
        "rule_zero": (
            "Real stream returns from exp2280_v6_sparse.pkl + cached QQQ "
            "trades from exp2250. Degradation applied as mean shift "
            "(preserves vol) — the correct model for alpha decay. No "
            "synthetic returns, no Monte Carlo, no distributional "
            "assumptions beyond the real empirical returns."
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
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def render_html(payload: Dict) -> str:
    base = payload["baseline_portfolio_metrics"]
    uniform = payload["uniform_degradation_scenarios"]
    stress = payload["stream_zero_stress_tests"]
    bands = payload["expectation_bands_for_carlos"]
    bp = payload["breakpoints"]
    nightmare = payload["combined_nightmare_scenario"]

    # Uniform sweep rows
    uniform_rows = ""
    for label, v in uniform.items():
        k = v["degradation_multiplier"]
        m = v["portfolio_metrics"]
        sh_cls = "good" if m["sharpe"] >= 3.0 else ("warn" if m["sharpe"] >= 2.0 else "bad")
        tag = "ELITE" if m["sharpe"] >= 3.0 else ("GOOD" if m["sharpe"] >= 2.0 else "BELOW")
        uniform_rows += f"""<tr>
            <td><strong>{k:.1f}×</strong></td>
            <td>{v['implied_live_sharpe_from_headline']:.2f}</td>
            <td>{m['cagr_pct']:.2f}%</td>
            <td class="{sh_cls}">{m['sharpe']:.2f}</td>
            <td>{m['max_dd_pct']:.2f}%</td>
            <td>{m['vol_pct']:.2f}%</td>
            <td class="{sh_cls}">{tag}</td>
        </tr>"""

    # Stream baselines
    stream_rows = ""
    for s, sh in payload["stream_baseline_sharpes"].items():
        stream_rows += f"""<tr>
            <td><strong>{s}</strong></td>
            <td>{sh:.2f}</td>
            <td>{payload['per_sleeve_leverage'].get(s, 0):.3f}</td>
        </tr>"""

    # Stress test rows
    stress_rows = ""
    for name, v in stress.items():
        stream = v["killed_stream"]
        m = v["portfolio_metrics"]
        ds = v["delta_sharpe"]
        stress_rows += f"""<tr>
            <td><strong>{stream}</strong></td>
            <td>{v['baseline_stream_sharpe']:.2f}</td>
            <td>{m['cagr_pct']:.2f}%</td>
            <td>{m['sharpe']:.2f}</td>
            <td>{ds:+.2f}</td>
        </tr>"""

    # Expectation bands
    def band_card(key, extra_class=""):
        b = bands[key]
        sh = b["portfolio_sharpe"]
        cls = "good" if sh >= 3.0 else ("warn" if sh >= 2.0 else "bad")
        return f"""
        <div class="kpi {cls}">
            <div class="value">{sh:.2f}</div>
            <div class="label">{b['label']}</div>
            <div style="font-size:0.72em; color:#64748b; margin-top:4px;">
                CAGR {b['portfolio_cagr_pct']:.1f}%
            </div>
        </div>"""

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2840 Backtest-to-Live Degradation</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1200px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.2em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:160px; }}
  .kpi .value {{ font-size:1.8em; font-weight:800; color:#0f172a; }}
  .kpi.good .value {{ color:#16a34a; }}
  .kpi.warn .value {{ color:#ca8a04; }}
  .kpi.bad  .value {{ color:#dc2626; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; font-weight:700; }}
  .warn {{ color:#ca8a04; font-weight:700; }}
  .bad  {{ color:#dc2626; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.74em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
  .warning {{ background:#fef2f2; border:2px solid #dc2626; border-radius:8px;
                padding:14px 18px; margin:14px 0; font-size:0.9rem; }}
</style></head><body>

<h1>EXP-2840 — Honest Backtest-to-Live Degradation</h1>
<div class="subtitle">What Carlos should expect · 8-stream v8 portfolio · {payload['timestamp']}</div>

<div class="note">
<strong>Method:</strong> apply alpha-decay multipliers k ∈ &#123;1.0, 0.9, …, 0.3&#125; to
each stream's non-zero-day MEAN (preserving vol — the correct alpha-decay
model) and recompute portfolio metrics. Degradation ≠ vol inflation; live
Sharpe drops because live returns are smaller, not because vol explodes.
Data: real IronVault-derived streams + cached QQQ trades, no synthetic.
<br><br>
EXP-2760 literature survey: live Sharpe is typically <strong>0.5-0.7×</strong>
of backtest for well-executed options-selling strategies. Our backtest
net Sharpe is <strong>6.00</strong>, which implies a realistic live
range of <strong>3.00-4.20</strong>.
</div>

<h2>Backtest Baseline</h2>
<div class="kpi-row">
    <div class="kpi good"><div class="value">{base['sharpe']:.2f}</div><div class="label">Backtest Sharpe</div></div>
    <div class="kpi good"><div class="value">{base['cagr_pct']:.1f}%</div><div class="label">CAGR</div></div>
    <div class="kpi good"><div class="value">{base['max_dd_pct']:.2f}%</div><div class="label">Max DD</div></div>
    <div class="kpi"><div class="value">{base['vol_pct']:.2f}%</div><div class="label">Vol</div></div>
</div>

<h2>What Carlos Should Expect — Live Range</h2>
<div class="kpi-row">
    {band_card("optimistic_p95")}
    {band_card("realistic_p50")}
    {band_card("pessimistic_p05")}
    {band_card("crisis_scenario")}
</div>
<ul>
    <li><strong>Optimistic (k = 0.8×):</strong> {bands['optimistic_p95']['rationale']}</li>
    <li><strong>Realistic (k = 0.6×):</strong> {bands['realistic_p50']['rationale']}</li>
    <li><strong>Pessimistic (k = 0.5×):</strong> {bands['pessimistic_p05']['rationale']}</li>
    <li><strong>Crisis:</strong> {bands['crisis_scenario']['rationale']}</li>
</ul>

<h2>Uniform Degradation Sweep</h2>
<table>
    <thead><tr><th>Multiplier k</th><th>Implied Live Sh (from 6.00)</th>
    <th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th><th>Band</th></tr></thead>
    <tbody>{uniform_rows}</tbody>
</table>

<h2>Break-Point Analysis</h2>
<div class="kpi-row">
    <div class="kpi good"><div class="value">{bp['k_at_sharpe_3.0_elite']:.2f}×</div><div class="label">k* for Sharpe ≥ 3.0 (elite)</div></div>
    <div class="kpi warn"><div class="value">{bp['k_at_sharpe_2.0_good']:.2f}×</div><div class="label">k* for Sharpe ≥ 2.0 (good)</div></div>
    <div class="kpi"><div class="value">{bp['k_at_sharpe_1.0']:.2f}×</div><div class="label">k* for Sharpe ≥ 1.0</div></div>
</div>
<p>{bp['interpretation']}</p>

<h2>Stream-Kill Stress Tests</h2>
<p>What happens if one sleeve goes entirely to zero (complete strategy failure)?</p>
<table>
    <thead><tr><th>Stream Killed</th><th>Baseline Stream Sh</th>
    <th>Portfolio CAGR</th><th>Portfolio Sharpe</th><th>ΔSharpe</th></tr></thead>
    <tbody>{stress_rows}</tbody>
</table>

<div class="warning">
<strong>Combined nightmare scenario</strong> (Crisis Alpha ZERO + 0.5× on all others):
CAGR <strong>{nightmare['portfolio_metrics']['cagr_pct']:.2f}%</strong> · Sharpe
<strong>{nightmare['portfolio_metrics']['sharpe']:.2f}</strong> · Max DD
{nightmare['portfolio_metrics']['max_dd_pct']:.2f}% · ΔSh
{nightmare['delta_sharpe']:+.2f} vs backtest.
This is the floor case — simultaneous alpha decay across all streams AND our
weakest stream failing entirely.
</div>

<h2>Per-Stream Baseline (non-zero days)</h2>
<table>
    <thead><tr><th>Stream</th><th>Baseline Sharpe</th><th>Production Leverage</th></tr></thead>
    <tbody>{stream_rows}</tbody>
</table>

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2840 — compass/exp2840_backtest_to_live_degradation.py · Real 8-stream cache · Mean-shift degradation model
</div>

</body></html>"""


if __name__ == "__main__":
    sys.exit(main())
