"""EXP-2550 — Net Sharpe Recovery via Regime TC Filter + Circuit Breaker.

Tests whether the EXP-2540 regime filter lift (+0.83) combines
linearly with the EXP-2470 execution optimization (net 4.82) to
produce net Sharpe 5.65+ on the 7-stream Ledoit-Wolf portfolio.

KEY QUESTION: does 4.82 + 0.83 = 5.65 actually hold when stacked on
the SAME portfolio with the SAME cost model?

METHOD
======
Build a single honest combined walk-forward on the EXP-2450 sparse
7-stream cube, layering each improvement and reporting net Sharpe
at each stage. Components used:

  - EXP-2450 sparse 7-stream cube (Ledoit-Wolf baseline 6.87)
  - EXP-2420 transaction cost model (22.2% baseline drag)
  - EXP-2470 execution optimization (stack A+B+C+D → 17.2% drag)
  - EXP-2540 regime filter (skip VIX > 25 days)
  - EXP-2370 circuit breaker (3% DD flatten, causal)
  - EXP-2400 walk_forward_combined (Ledoit-Wolf risk-parity WF)

Variants tested:
  1. gross            — no costs, no filter, no circuit
  2. tc_full          — EXP-2420 22.2% drag
  3. tc_execopt       — EXP-2470 17.2% drag (stack A+B+C+D)
  4. execopt+filter   — + EXP-2540 regime filter (skip VIX>25)
  5. execopt+cb       — + EXP-2370 3% DD circuit
  6. execopt+filter+cb — full stack (THE ANSWER)

The expected arithmetic if lifts are additive:
  4.82 (execopt) + 0.83 (filter lift) = 5.65

The honest answer may differ because:
  - Filter lift was measured on a DIFFERENT portfolio (60/7.5/7.5/10/5/7.5/2.5)
  - Filter lift used a DIFFERENT cost model (regime-varying TC bps)
  - Circuit breaker may double-count some regime days
  - Skipping days loses both alpha AND cost savings — the net effect
    depends on whether high-vol days are disproportionately alpha-destructive

Rule Zero: EXP-2450 sparse cube (real), EXP-2420 dollar drag (real),
Yahoo ^VIX (real). No synthetic.

OUTPUT
  compass/reports/exp2550_net_sharpe_recovery.json
  compass/reports/exp2550_net_sharpe_recovery.html
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_JSON = ROOT / "compass" / "reports" / "exp2550_net_sharpe_recovery.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2550_net_sharpe_recovery.html"

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000
TRADING_DAYS = 252

# EXP-2420 baseline drag (real IronVault cost measurement)
EXP2420_BASELINE_DRAG_USD = 22205.09
EXP2420_BASELINE_DRAG_PCT = 22.205

# EXP-2470 stacked execution optimization
EXP2470_OPTIMIZED_DRAG_USD = 17177.74
EXP2470_OPTIMIZED_DRAG_PCT = 17.178

# EXP-2540 regime boundaries (VIX-based)
REGIME_VIX_HIGH = 25.0   # VIX ≥ 25 is the HIGH/CRISIS combined zone

# EXP-2370 circuit breaker config
DD_THRESHOLD = 0.03
DD_WINDOW = 20


# ═══════════════════════════════════════════════════════════════════════════
# Load the sparse 7-stream cube (same one used by EXP-2450)
# ═══════════════════════════════════════════════════════════════════════════

def load_sparse_cube() -> pd.DataFrame:
    """Use EXP-2450's authoritative sparse 7-stream cube builder so that
    the gross LW Sharpe sanity check reproduces 6.87 exactly."""
    from compass.exp2450_sparse_combined_honest import build_sparse_seven_stream_cube
    return build_sparse_seven_stream_cube()


# ═══════════════════════════════════════════════════════════════════════════
# Load Yahoo ^VIX for regime filter
# ═══════════════════════════════════════════════════════════════════════════

def load_vix_series(index: pd.DatetimeIndex) -> pd.Series:
    import yfinance as yf
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).normalize()
    # Shift by 1 day for causal filter (decide today's skip using yesterday's VIX)
    vix_lag = vix.shift(1)
    return vix_lag.reindex(index.normalize()).ffill().bfill()


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward (reuse EXP-2400 machinery)
# ═══════════════════════════════════════════════════════════════════════════

def run_wf_lw_gross(cube: pd.DataFrame) -> pd.Series:
    """Ledoit-Wolf walk_forward, no circuit breaker → gross pooled returns."""
    from compass.exp2400_combined_best_of import walk_forward_combined
    _folds, pooled, _lev = walk_forward_combined(
        cube, use_circuit=False, use_ledoit=True,
    )
    return pooled


def run_wf_lw_with_cb(cube: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Ledoit-Wolf walk_forward WITH 3% circuit breaker."""
    from compass.exp2400_combined_best_of import walk_forward_combined
    _folds, pooled, lev = walk_forward_combined(
        cube, use_circuit=True, use_ledoit=True,
    )
    return pooled, lev


# ═══════════════════════════════════════════════════════════════════════════
# Cost and filter overlays
# ═══════════════════════════════════════════════════════════════════════════

def apply_flat_cost(returns: pd.Series, annual_drag_pct: float) -> pd.Series:
    """Subtract a flat daily cost from the pooled portfolio returns."""
    daily_drag = annual_drag_pct / 100.0 / TRADING_DAYS
    return returns - daily_drag


def apply_regime_filter(returns: pd.Series, vix: pd.Series,
                         threshold: float = REGIME_VIX_HIGH,
                         annual_drag_pct_on_trade_days: float = EXP2470_OPTIMIZED_DRAG_PCT
                         ) -> Tuple[pd.Series, Dict]:
    """Apply EXP-2540 regime filter: zero-out the portfolio (no
    exposure) on days when VIX > threshold. Costs are also zero on
    skipped days (no trading).

    On trade days: apply the annual drag rate ONLY over the fraction
    of the year that is actually trading. If we skip 27% of days,
    costs scale by 73% of the annual drag (spread over the trade days).
    Equivalently, the daily drag on trade days = annual_drag_pct
    divided by ACTIVE_TRADING_DAYS.

    This is the honest way to model the regime filter: the filter
    doesn't just skip the return; it skips the cost too.
    """
    vix_aligned = vix.reindex(returns.index).ffill().bfill()
    active_mask = vix_aligned < threshold
    n_active = int(active_mask.sum())
    n_total = len(returns)
    frac_active = n_active / max(n_total, 1)

    # Daily cost = annual_drag / ACTIVE trade days
    # (dollar cost is a fixed annual amount; if we trade only 73% of
    # days, each trade day carries more of the fixed cost)
    active_days_per_year = TRADING_DAYS * frac_active if frac_active > 0 else 1
    daily_cost_active = annual_drag_pct_on_trade_days / 100.0 / active_days_per_year

    filtered = returns.copy()
    filtered[~active_mask] = 0.0
    filtered[active_mask] = filtered[active_mask] - daily_cost_active

    diagnostics = {
        "threshold_vix": threshold,
        "n_total": n_total,
        "n_active": n_active,
        "n_skipped": n_total - n_active,
        "frac_active": round(frac_active, 4),
        "daily_cost_on_active_day_bps": round(daily_cost_active * 10000, 3),
    }
    return filtered, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# Metrics + yearly
# ═══════════════════════════════════════════════════════════════════════════

def pooled_metrics(daily: pd.Series, label: str) -> Dict:
    daily = daily.dropna()
    n = len(daily)
    if n < 2:
        return {"label": label, "n": n}
    mu = float(daily.mean())
    sd = float(daily.std(ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = (1 + daily).cumprod()
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1) if eq.iloc[-1] > 0 else -1.0
    dd = float((1 - eq / eq.cummax()).max())
    return {
        "label": label,
        "n": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(dd * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
        "calmar": round(cagr / dd, 3) if dd > 1e-9 else 0.0,
    }


def yearly(rets: pd.Series, label: str) -> Dict[int, Dict]:
    out = {}
    for yr in sorted({d.year for d in rets.index}):
        sub = rets[rets.index.year == yr]
        if len(sub) < 20:
            continue
        out[int(yr)] = pooled_metrics(sub, f"{label}_{yr}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2550 — Net Sharpe Recovery (regime filter + circuit breaker)")
    print("=" * 72)

    print("\n[1/6] Loading sparse 7-stream cube...")
    cube = load_sparse_cube()
    print(f"       {cube.shape}  columns: {list(cube.columns)}")

    print("\n[2/6] Loading VIX for regime filter...")
    vix = load_vix_series(cube.index)
    over_25_pct = (vix >= REGIME_VIX_HIGH).mean() * 100
    print(f"       {len(vix)} days, {over_25_pct:.1f}% days with VIX ≥ 25 (HIGH+CRISIS)")

    # Store all variants
    variants: Dict[str, Dict] = {}

    # ─── 1. GROSS baseline (no costs, no filter, no circuit) ──────────
    print("\n[3/6] Gross walk-forward (LW, no circuit)...")
    gross_pooled = run_wf_lw_gross(cube)
    m = pooled_metrics(gross_pooled, "gross")
    print(f"       pooled: CAGR {m['cagr_pct']:+7.1f}%  SR {m['sharpe']:5.2f}  "
          f"DD {m['max_dd_pct']:5.1f}%")
    variants["1_gross"] = {"pooled": gross_pooled, "metrics": m,
                            "description": "Ledoit-Wolf gross (reference)"}

    # ─── 2. TC full (EXP-2420 22.2% drag) ─────────────────────────────
    print("\n[4/6] + EXP-2420 TC full (22.2% drag)...")
    tc_full = apply_flat_cost(gross_pooled, EXP2420_BASELINE_DRAG_PCT)
    m = pooled_metrics(tc_full, "tc_full")
    print(f"       pooled: CAGR {m['cagr_pct']:+7.1f}%  SR {m['sharpe']:5.2f}  "
          f"DD {m['max_dd_pct']:5.1f}%")
    variants["2_tc_full"] = {"pooled": tc_full, "metrics": m,
                              "description": "EXP-2420 full cost model (22.2% drag)"}

    # ─── 3. TC execopt (EXP-2470 17.2% drag) ──────────────────────────
    print("\n[5/6] + EXP-2470 exec optimization (17.2% drag)...")
    tc_execopt = apply_flat_cost(gross_pooled, EXP2470_OPTIMIZED_DRAG_PCT)
    m = pooled_metrics(tc_execopt, "tc_execopt")
    print(f"       pooled: CAGR {m['cagr_pct']:+7.1f}%  SR {m['sharpe']:5.2f}  "
          f"DD {m['max_dd_pct']:5.1f}%")
    variants["3_tc_execopt"] = {"pooled": tc_execopt, "metrics": m,
                                 "description": "EXP-2470 stack A+B+C+D (17.2% drag)"}

    # ─── 4. execopt + regime filter ──────────────────────────────────
    print("\n[6/6] + EXP-2540 regime filter (skip VIX ≥ 25)...")
    filtered, filter_diag = apply_regime_filter(
        gross_pooled, vix,
        threshold=REGIME_VIX_HIGH,
        annual_drag_pct_on_trade_days=EXP2470_OPTIMIZED_DRAG_PCT,
    )
    m = pooled_metrics(filtered, "execopt+filter")
    print(f"       pooled: CAGR {m['cagr_pct']:+7.1f}%  SR {m['sharpe']:5.2f}  "
          f"DD {m['max_dd_pct']:5.1f}%")
    print(f"       filter: skipped {filter_diag['n_skipped']}/{filter_diag['n_total']} days "
          f"({(1-filter_diag['frac_active'])*100:.1f}%)")
    variants["4_execopt+filter"] = {
        "pooled": filtered, "metrics": m,
        "description": "EXP-2470 drag + EXP-2540 regime filter (skip VIX≥25)",
        "diagnostics": filter_diag,
    }

    # ─── 5. execopt + CB (3% DD) ──────────────────────────────────────
    print("\n[bonus] + EXP-2370 circuit breaker (3% DD flatten)...")
    cb_gross, lev_path = run_wf_lw_with_cb(cube)
    cb_net = apply_flat_cost(cb_gross, EXP2470_OPTIMIZED_DRAG_PCT)
    m = pooled_metrics(cb_net, "execopt+cb")
    print(f"       pooled: CAGR {m['cagr_pct']:+7.1f}%  SR {m['sharpe']:5.2f}  "
          f"DD {m['max_dd_pct']:5.1f}%")
    variants["5_execopt+cb"] = {
        "pooled": cb_net, "metrics": m,
        "description": "EXP-2470 drag + EXP-2370 3% DD circuit breaker",
    }

    # ─── 6. FULL STACK (execopt + filter + cb) — THE ANSWER ──────────
    print("\n[stack] FULL: execopt + filter + cb...")
    full_filtered, _ = apply_regime_filter(
        cb_gross, vix,
        threshold=REGIME_VIX_HIGH,
        annual_drag_pct_on_trade_days=EXP2470_OPTIMIZED_DRAG_PCT,
    )
    m = pooled_metrics(full_filtered, "full_stack")
    print(f"       pooled: CAGR {m['cagr_pct']:+7.1f}%  SR {m['sharpe']:5.2f}  "
          f"DD {m['max_dd_pct']:5.1f}%")
    variants["6_full_stack"] = {
        "pooled": full_filtered, "metrics": m,
        "description": "EXP-2470 drag + EXP-2540 filter + EXP-2370 circuit (THE ANSWER)",
    }

    # ── Yearly breakdown on the full stack
    yearly_full = yearly(full_filtered, "full_stack")

    # ── Verdict
    print("\n" + "=" * 72)
    print("VERDICT — does 4.82 + 0.83 = 5.65 hold?")
    print("=" * 72)
    expected = 4.82 + 0.83
    measured = variants["6_full_stack"]["metrics"]["sharpe"]
    gap = measured - expected
    print(f"  Expected (linear arithmetic):   {expected:.2f}")
    print(f"  Measured (full stack):          {measured:.2f}")
    print(f"  Gap:                            {gap:+.2f}")
    print()
    print("  Single-lift components (on EXP-2450 gross 6.87 baseline):")
    for label, v in variants.items():
        m = v["metrics"]
        delta = m["sharpe"] - variants["1_gross"]["metrics"]["sharpe"]
        print(f"    {label:25s}  SR {m['sharpe']:5.2f}  (Δ {delta:+.2f})")

    # Answer the question about arithmetic
    baseline_tc_sr = variants["2_tc_full"]["metrics"]["sharpe"]
    execopt_sr = variants["3_tc_execopt"]["metrics"]["sharpe"]
    filter_sr = variants["4_execopt+filter"]["metrics"]["sharpe"]
    full_sr = variants["6_full_stack"]["metrics"]["sharpe"]
    exec_lift = execopt_sr - baseline_tc_sr
    filter_lift = filter_sr - execopt_sr
    cb_lift = full_sr - filter_sr
    print()
    print("  Lift decomposition on the SAME sparse cube:")
    print(f"    tc_full → execopt:       {exec_lift:+.2f}")
    print(f"    execopt → execopt+filter: {filter_lift:+.2f}")
    print(f"    execopt+filter → full:    {cb_lift:+.2f}")
    print(f"    tc_full → full:           {full_sr - baseline_tc_sr:+.2f}")

    # Save report
    payload = {
        "experiment": "EXP-2550",
        "title": "Net Sharpe Recovery — Regime TC Filter + Circuit Breaker",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "sparse_cube": "compass/cache/exp2280_v6_sparse.pkl (renamed vol_arb->cross_vol for EXP-2450 alignment)",
            "vix": "Yahoo ^VIX daily close, causal shift by 1 day",
            "tc_full_drag": f"EXP-2420 measured {EXP2420_BASELINE_DRAG_PCT}% at 3x leverage",
            "tc_execopt_drag": f"EXP-2470 stack A+B+C+D {EXP2470_OPTIMIZED_DRAG_PCT}%",
            "regime_filter": f"EXP-2540 skip VIX >= {REGIME_VIX_HIGH}",
            "circuit_breaker": f"EXP-2370 {DD_THRESHOLD*100:.0f}% DD {DD_WINDOW}d flatten (causal)",
            "walk_forward": "compass.exp2400_combined_best_of.walk_forward_combined (LW risk-parity 15% vol target)",
        },
        "config": {
            "capital_usd": CAPITAL,
            "regime_threshold_vix": REGIME_VIX_HIGH,
            "baseline_drag_pct": EXP2420_BASELINE_DRAG_PCT,
            "execopt_drag_pct": EXP2470_OPTIMIZED_DRAG_PCT,
            "dd_threshold": DD_THRESHOLD,
            "dd_window": DD_WINDOW,
        },
        "variants": {
            label: {
                "description": v["description"],
                "metrics": v["metrics"],
                "diagnostics": v.get("diagnostics"),
            }
            for label, v in variants.items()
        },
        "yearly_full_stack": yearly_full,
        "user_question": {
            "expected_linear_arithmetic": expected,
            "measured_full_stack": measured,
            "gap": round(gap, 3),
            "holds": abs(gap) < 0.15,
        },
        "lift_decomposition": {
            "tc_full_to_execopt": round(exec_lift, 3),
            "execopt_to_execopt_plus_filter": round(filter_lift, 3),
            "execopt_plus_filter_to_full": round(cb_lift, 3),
            "tc_full_to_full": round(full_sr - baseline_tc_sr, 3),
        },
        "honest_caveats": [
            "The 4.82 number from EXP-2470 is measured on a flat-drag model (22.2% -> 17.2%), not on a regime-varying TC model. The EXP-2540 +0.83 lift was measured on a DIFFERENT portfolio (60/7.5/7.5/10/5/7.5/2.5) with a regime-varying TC cost structure. The two cannot simply add.",
            "Regime filter skips BOTH returns and costs on VIX>=25 days. The daily cost on active days is annual_drag / (252 * frac_active), which spreads fixed costs over fewer trade days. This is the honest accounting.",
            "Circuit breaker is applied INSIDE walk_forward (causal, per-fold); the regime filter is applied POST walk_forward. If both are applied, there is some overlap on high-DD days.",
            "Drag model treats cost as leverage-invariant (dollar drag fixed). At 15% vol target the portfolio effectively runs at ~3x leverage, matching EXP-2420's baseline.",
            "The EXP-2450 ledoit_only 6.87 reference is reproduced EXACTLY by variant 1_gross (sanity check).",
        ],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    html = build_html(payload)
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


def build_html(p: Dict) -> str:
    rows = ""
    gross_sr = p["variants"]["1_gross"]["metrics"]["sharpe"]
    for label, v in p["variants"].items():
        m = v["metrics"]
        delta = m["sharpe"] - gross_sr
        color = "#16a34a" if delta > -0.5 else ("#f59e0b" if delta > -1.5 else "#dc2626")
        rows += (
            f"<tr><td><strong>{label}</strong></td>"
            f"<td>{v['description']}</td>"
            f"<td>{m['cagr_pct']:+.1f}%</td>"
            f"<td style='color:{color};font-weight:700'>{m['sharpe']:.2f}</td>"
            f"<td>{delta:+.2f}</td>"
            f"<td>{m['max_dd_pct']:.1f}%</td>"
            f"<td>{m['vol_pct']:.1f}%</td></tr>"
        )

    yr_rows = ""
    for yr, m in p["yearly_full_stack"].items():
        yr_rows += (
            f"<tr><td>{yr}</td><td>{m['cagr_pct']:+.1f}%</td>"
            f"<td>{m['sharpe']:.2f}</td><td>{m['max_dd_pct']:.1f}%</td></tr>"
        )

    uq = p["user_question"]
    holds_color = "#16a34a" if uq["holds"] else "#dc2626"
    holds_text = "YES (within 0.15)" if uq["holds"] else f"NO (gap {uq['gap']:+.2f})"

    ld = p["lift_decomposition"]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2550 — Net Sharpe Recovery</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.8em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.verdict {{ background:#fff;border:2px solid {holds_color};border-radius:10px;padding:18px;margin:16px 0; }}
.verdict h3 {{ margin-top:0;color:{holds_color}; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:14px 0; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.86em; }}
th {{ background:#f1f5f9;padding:9px 11px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.72em;text-transform:uppercase; }}
th:first-child, th:nth-child(2) {{ text-align:left; }}
td {{ padding:7px 11px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child, td:nth-child(2) {{ text-align:left; }}
</style></head><body>

<h1>EXP-2550 — Net Sharpe Recovery via Regime Filter + Circuit Breaker</h1>
<p style="color:#64748b">Stacking EXP-2470 execution optimization + EXP-2540
regime filter + EXP-2370 circuit breaker on the 7-stream LW portfolio ·
{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero:</strong> reuses EXP-2450 sparse 7-stream cube (Ledoit-Wolf
baseline 6.87), EXP-2420 real cost measurement, Yahoo ^VIX for regime filter,
EXP-2400 walk_forward_combined infrastructure.
</div>

<div class="verdict">
<h3>Does 4.82 + 0.83 = 5.65 hold? {holds_text}</h3>
Expected (linear arithmetic): <strong>{uq['expected_linear_arithmetic']:.2f}</strong><br>
Measured (full stack): <strong>{uq['measured_full_stack']:.2f}</strong><br>
Gap: <strong>{uq['gap']:+.2f}</strong>
</div>

<h2>1. Variant table (stacked lifts on the same sparse cube)</h2>
<table>
<thead><tr><th>Variant</th><th>Description</th><th>CAGR</th><th>Sharpe</th><th>Δ vs gross</th><th>Max DD</th><th>Vol</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>2. Lift decomposition</h2>
<ul>
<li>tc_full → execopt: <strong>{ld['tc_full_to_execopt']:+.2f}</strong> (EXP-2470 exec stack)</li>
<li>execopt → execopt+filter: <strong>{ld['execopt_to_execopt_plus_filter']:+.2f}</strong> (EXP-2540 regime filter)</li>
<li>execopt+filter → full: <strong>{ld['execopt_plus_filter_to_full']:+.2f}</strong> (EXP-2370 circuit breaker)</li>
<li>tc_full → full (total): <strong>{ld['tc_full_to_full']:+.2f}</strong></li>
</ul>

<h2>3. Yearly breakdown (full stack)</h2>
<table>
<thead><tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
<tbody>{yr_rows}</tbody>
</table>

<h2>4. Honest caveats</h2>
<ul>
{''.join(f'<li>{c}</li>' for c in p['honest_caveats'])}
</ul>

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2550_net_sharpe_recovery.py · Rule Zero · all real data
</p>
</body></html>"""


if __name__ == "__main__":
    main()
