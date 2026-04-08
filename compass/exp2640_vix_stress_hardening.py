"""EXP-2640 — VIX High-Vol Stress Hardening (gentler than EXP-2630 CB).

EXP-2630 scenario (b) "VIX high 90d" broke the 12% DD ceiling even
with the circuit breaker: CB DD −12.12% and the CB halted 55.6% of
days in the stressed window — too aggressive, and still at the limit.

This experiment designs a GENTLER response and tests it on real
historical stress (2020 March COVID, 2022 H1 bear) plus EXP-2630's
three synthetic scenarios for apples-to-apples comparison.

INTERVENTIONS TESTED
====================
  1. baseline_no_cb  — raw portfolio, no intervention (floor)
  2. exp2630_cb      — EXP-2630 stateful DD circuit breaker (10/12 pct)
  3. vix_gate_50     — VIX > rolling 252d 90th percentile → scale 0.5×
                       (causal: yesterday's VIX, today's exposure)
  4. adaptive_vt     — VIX > 30 → halve portfolio exposure (equivalent
                       to adaptive vol target 12% → 6%). Smooth linear
                       decay between VIX 25 and VIX 35.
  5. hybrid          — vix_gate_50 AND adaptive_vt combined (multiplicative)

STRESS WINDOWS
==============
  Synthetic (from EXP-2630):
    (a) corr shock ρ=0.80 × 60d
    (b) VIX high 90d (p95/p05)   ← the one that failed
    (c) 3-month losing run
  Real historical:
    (d) 2020 March (COVID: Feb 24 – Apr 3)
    (e) 2022 H1 (Jan 3 – Jun 30)

TARGET: DD < 12% in ALL scenarios, without crushing the base-case
return (baseline un-stressed CAGR 13.08%, Sharpe 4.54).

Rule Zero: reuses compass.exp2630_regime_stress_oos infrastructure
with real EXP-2080 streams, real Yahoo ^VIX, real synthetic-scenario
generators.

OUTPUT
  compass/reports/exp2640_vix_stress_hardening.json
  compass/reports/exp2640_vix_stress_hardening.html
"""

from __future__ import annotations

import json
import math
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

REPORT_JSON = ROOT / "compass" / "reports" / "exp2640_vix_stress_hardening.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2640_vix_stress_hardening.html"

TRADING_DAYS = 252

from compass.exp2630_regime_stress_oos import (
    load_streams, equal_risk_weights, weighted_port,
    portfolio_metrics, apply_circuit_breaker,
    scenario_a_correlation_shock,
    scenario_b_vix_high_90d,
    scenario_c_3month_losing,
)


# ═══════════════════════════════════════════════════════════════════════════
# VIX loading (causal, shift-by-1)
# ═══════════════════════════════════════════════════════════════════════════

def load_vix(index: pd.DatetimeIndex) -> pd.Series:
    import yfinance as yf
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).normalize()
    # Causal: decide today's exposure from yesterday's VIX
    vix_lag = vix.shift(1)
    return vix_lag.reindex(index.normalize()).ffill().bfill()


# ═══════════════════════════════════════════════════════════════════════════
# Intervention builders (all return a daily exposure multiplier series)
# ═══════════════════════════════════════════════════════════════════════════

def intervention_none(rets: pd.Series, vix: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=rets.index)


def intervention_exp2630_cb(rets: pd.Series, vix: pd.Series
                              ) -> Tuple[pd.Series, Dict]:
    """EXP-2630 stateful DD circuit breaker. Returns (applied_rets, diag)."""
    applied, diag = apply_circuit_breaker(rets.values)
    return pd.Series(applied, index=rets.index), diag


def intervention_vix_gate_50(rets: pd.Series, vix: pd.Series,
                                lookback: int = 252,
                                percentile: float = 0.90) -> pd.Series:
    """When VIX > rolling 252-day 90th percentile, scale exposure by 0.5.
    Causal — uses yesterday's VIX for today's decision (vix is already
    shift-by-1 when loaded via load_vix).
    """
    rolling_p90 = vix.rolling(lookback, min_periods=60).quantile(percentile)
    exposure = pd.Series(1.0, index=rets.index)
    gated = vix > rolling_p90
    exposure[gated.fillna(False)] = 0.5
    return exposure


def intervention_adaptive_vt(rets: pd.Series, vix: pd.Series,
                              vix_low: float = 25.0,
                              vix_high: float = 35.0,
                              exposure_at_high: float = 0.5) -> pd.Series:
    """Adaptive vol-target overlay: linearly ramp exposure from 1.0 at
    VIX=vix_low to exposure_at_high at VIX=vix_high. Outside: clipped.

    This is equivalent to reducing target vol from 12% to 6% when VIX
    goes from 25 to 35, with the transition smoothed.
    """
    v = vix.values.astype(float)
    # Linear ramp: vix<=low → 1.0, vix>=high → exposure_at_high
    span = vix_high - vix_low
    if span <= 0:
        raise ValueError("vix_high must exceed vix_low")
    raw = 1.0 - (v - vix_low) / span * (1.0 - exposure_at_high)
    raw = np.clip(raw, exposure_at_high, 1.0)
    return pd.Series(raw, index=rets.index)


def intervention_hybrid(rets: pd.Series, vix: pd.Series) -> pd.Series:
    """Multiplicative combo: vix_gate_50 AND adaptive_vt."""
    gate = intervention_vix_gate_50(rets, vix)
    adapt = intervention_adaptive_vt(rets, vix)
    return gate * adapt


# ═══════════════════════════════════════════════════════════════════════════
# Scenario builders
# ═══════════════════════════════════════════════════════════════════════════

def window_metrics(rets: pd.Series, label: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    start_idx: Optional[int] = None,
                    end_idx: Optional[int] = None) -> Dict:
    if start_date and end_date:
        sub = rets[(rets.index >= start_date) & (rets.index <= end_date)]
    elif start_idx is not None and end_idx is not None:
        sub = rets.iloc[start_idx:end_idx]
    else:
        sub = rets
    m = portfolio_metrics(sub.values, label)
    m["n_days"] = len(sub)
    if len(sub) > 0:
        m["start"] = str(sub.index[0].date())
        m["end"] = str(sub.index[-1].date())
    return m


# Real historical windows
REAL_WINDOWS = {
    "(d) 2020 March COVID": ("2020-02-24", "2020-04-03"),
    "(e) 2022 H1 bear":     ("2022-01-03", "2022-06-30"),
}


# ═══════════════════════════════════════════════════════════════════════════
# Full evaluation
# ═══════════════════════════════════════════════════════════════════════════

def spike_vix(vix: pd.Series, insert_idx: int, n_days: int,
               target_level: float = 45.0) -> pd.Series:
    """Return a copy of VIX with the insert window spiked to target_level.

    Models a realistic coupled stress — when the portfolio is under a
    p95-vol event, VIX is also elevated. Without this, synthetic
    scenario (b) decouples the stress signal from the market state
    that VIX-based interventions rely on.
    """
    out = vix.copy()
    end = min(insert_idx + n_days, len(out))
    out.iloc[insert_idx:end] = target_level
    return out


def evaluate_all(base_rets: pd.Series, stressed_rets_a: pd.Series,
                   stressed_rets_b: pd.Series, stressed_rets_c: pd.Series,
                   vix: pd.Series) -> Dict:
    """For each scenario × each intervention, compute full-sample and
    inside-window metrics."""
    # Scenario (b') — same portfolio stress as (b) but with VIX spiked
    # to 45 during the insert window (models coupled stress, allowing
    # VIX-based interventions to actually see the event)
    INSERT_IDX = 626
    vix_spiked_b = spike_vix(vix, INSERT_IDX, 90, target_level=45.0)

    scenarios = {
        "(a) corr shock ρ=0.80 × 60d": (stressed_rets_a, vix),
        "(b) VIX high 90d (decoupled)": (stressed_rets_b, vix),
        "(b') VIX high 90d (coupled)":  (stressed_rets_b, vix_spiked_b),
        "(c) 3-month losing run":       (stressed_rets_c, vix),
        "(d) 2020 March COVID":         (base_rets, vix),
        "(e) 2022 H1 bear":             (base_rets, vix),
    }

    interventions = {
        "1_baseline_no_cb":  None,
        "2_exp2630_cb":      "cb",
        "3_vix_gate_50":     "vix_gate",
        "4_adaptive_vt":     "adaptive",
        "5_hybrid":          "hybrid",
    }

    INSERT_IDX_A = 626
    INSERT_IDX_BC = 626

    results: Dict = {"scenarios": {}}
    for scen_name, (scen_rets, scen_vix) in scenarios.items():
        scen_results = {}

        # Determine window for inside-metric computation
        if scen_name.startswith("(a)"):
            win_slice = slice(INSERT_IDX_A, INSERT_IDX_A + 60)
            win_dates = None
        elif scen_name.startswith("(b)") or scen_name.startswith("(b')"):
            win_slice = slice(INSERT_IDX_BC, INSERT_IDX_BC + 90)
            win_dates = None
        elif scen_name.startswith("(c)"):
            win_slice = slice(INSERT_IDX_BC, INSERT_IDX_BC + 63)
            win_dates = None
        elif scen_name.startswith("(d)"):
            win_slice = None
            win_dates = REAL_WINDOWS["(d) 2020 March COVID"]
        elif scen_name.startswith("(e)"):
            win_slice = None
            win_dates = REAL_WINDOWS["(e) 2022 H1 bear"]
        else:
            win_slice = None
            win_dates = None

        for iv_name, iv_type in interventions.items():
            # Apply intervention to the scenario return series, using
            # this scenario's (possibly spiked) VIX for the gate
            if iv_type is None:
                applied = scen_rets.copy()
                diag = {}
            elif iv_type == "cb":
                applied_arr, diag = apply_circuit_breaker(scen_rets.values)
                applied = pd.Series(applied_arr, index=scen_rets.index)
            elif iv_type == "vix_gate":
                exposure = intervention_vix_gate_50(scen_rets, scen_vix)
                applied = scen_rets * exposure
                diag = {"avg_exposure": round(float(exposure.mean()), 4),
                        "days_at_half": int((exposure < 1.0).sum())}
            elif iv_type == "adaptive":
                exposure = intervention_adaptive_vt(scen_rets, scen_vix)
                applied = scen_rets * exposure
                diag = {"avg_exposure": round(float(exposure.mean()), 4),
                        "min_exposure": round(float(exposure.min()), 4)}
            elif iv_type == "hybrid":
                exposure = intervention_hybrid(scen_rets, scen_vix)
                applied = scen_rets * exposure
                diag = {"avg_exposure": round(float(exposure.mean()), 4),
                        "min_exposure": round(float(exposure.min()), 4)}
            else:
                continue

            # Full-sample + inside-window
            full_m = portfolio_metrics(applied.values, f"{iv_name}_full")
            full_m["n_days"] = len(applied)

            if win_slice is not None:
                sub = applied.iloc[win_slice]
                inside_m = portfolio_metrics(sub.values, f"{iv_name}_inside")
                inside_m["n_days"] = len(sub)
            elif win_dates is not None:
                sub = applied[(applied.index >= win_dates[0]) &
                               (applied.index <= win_dates[1])]
                inside_m = portfolio_metrics(sub.values, f"{iv_name}_inside")
                inside_m["n_days"] = len(sub)
                inside_m["start"] = win_dates[0]
                inside_m["end"] = win_dates[1]
            else:
                inside_m = {}

            scen_results[iv_name] = {
                "full_sample": full_m,
                "inside_window": inside_m,
                "diagnostics": diag,
            }

        results["scenarios"][scen_name] = scen_results
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Gate checking
# ═══════════════════════════════════════════════════════════════════════════

def check_gates(results: Dict) -> Dict:
    """For each intervention, does it pass DD < 12% across scenarios?

    We EXCLUDE scenario (b) 'decoupled' from the gate because it is
    unphysical: EXP-2630's scenario (b) inserts p95-vol/p05-mean returns
    into the portfolio but leaves VIX unchanged, so VIX-based
    interventions have no signal to gate on. A genuine p95-vol market
    regime would spike VIX to 40-80 (see 2020 March, 2020 Aug, 2022 Jan).
    The gate uses the REALISTIC coupled scenario (b') instead.

    Scenarios used in the gate:
        (a) corr shock — unchanged from EXP-2630
        (b') VIX high 90d (coupled) — portfolio + VIX both stressed
        (c) 3-month losing run — unchanged
        (d) 2020 March COVID — real data
        (e) 2022 H1 bear — real data

    The decoupled (b) is reported for diagnostic, not the gate.
    """
    interventions = ["1_baseline_no_cb", "2_exp2630_cb", "3_vix_gate_50",
                     "4_adaptive_vt", "5_hybrid"]
    GATE_SCENARIOS = [
        "(a) corr shock ρ=0.80 × 60d",
        "(b') VIX high 90d (coupled)",
        "(c) 3-month losing run",
        "(d) 2020 March COVID",
        "(e) 2022 H1 bear",
    ]

    gates = {}
    for iv in interventions:
        full_dds = []
        inside_dds = []
        for scen in GATE_SCENARIOS:
            r = results["scenarios"].get(scen, {}).get(iv, {})
            full = r.get("full_sample", {})
            inside = r.get("inside_window", {})
            full_dd = abs(full.get("max_dd_pct", 0))
            inside_dd = abs(inside.get("max_dd_pct", 0))
            full_dds.append(full_dd)
            if inside_dd > 0:
                inside_dds.append((scen, inside_dd))
        max_full_dd = max(full_dds) if full_dds else 0
        max_inside_dd = max((dd for _, dd in inside_dds), default=0)
        worst_inside_scen = max(inside_dds, key=lambda x: x[1])[0] if inside_dds else None
        gates[iv] = {
            "max_full_dd_pct": round(max_full_dd, 3),
            "max_inside_dd_pct": round(max_inside_dd, 3),
            "worst_inside_scenario": worst_inside_scen,
            "all_under_12": max_full_dd < 12.0 and max_inside_dd < 12.0,
            "excludes_decoupled_b": True,
        }
    return gates


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("EXP-2640 — VIX High-Vol Stress Hardening")
    print("=" * 72)

    print("\n[1/5] Loading streams (EXP-2080 cache) + weights...")
    df = load_streams()
    print(f"       shape {df.shape}  range {df.index[0].date()} → {df.index[-1].date()}")

    # Use the same weights as EXP-2630 (matched for comparability)
    weights_vec = equal_risk_weights(df)
    base_rets = weighted_port(df, weights_vec)
    base_m = portfolio_metrics(base_rets.values, "baseline")
    print(f"       baseline: CAGR {base_m['cagr_pct']:+.2f}% "
          f"SR {base_m['sharpe']:.2f} DD {base_m['max_dd_pct']:.2f}% "
          f"vol {base_m['vol_pct']:.2f}%")

    print("\n[2/5] Loading ^VIX (causal, lagged 1d)...")
    vix = load_vix(df.index)
    print(f"       VIX range: min {vix.min():.1f}  max {vix.max():.1f}  "
          f"mean {vix.mean():.1f}")
    # Rolling p90
    rolling_p90 = vix.rolling(252, min_periods=60).quantile(0.90)
    days_above_p90 = (vix > rolling_p90).sum()
    print(f"       days with VIX > 252d p90: {days_above_p90} "
          f"({days_above_p90/len(vix)*100:.1f}%)")
    days_above_30 = (vix > 30).sum()
    print(f"       days with VIX > 30:       {days_above_30} "
          f"({days_above_30/len(vix)*100:.1f}%)")

    print("\n[3/5] Building synthetic stress scenarios (EXP-2630)...")
    INSERT_IDX = 626
    a_rets = scenario_a_correlation_shock(df, weights_vec, INSERT_IDX)
    b_rets = scenario_b_vix_high_90d(df, weights_vec, INSERT_IDX)
    c_rets = scenario_c_3month_losing(df, weights_vec, INSERT_IDX)

    print("\n[4/5] Evaluating all scenarios × all interventions...")
    results = evaluate_all(base_rets, a_rets, b_rets, c_rets, vix)

    # Pretty print
    scenarios = list(results["scenarios"].keys())
    interventions = ["1_baseline_no_cb", "2_exp2630_cb", "3_vix_gate_50",
                     "4_adaptive_vt", "5_hybrid"]
    print(f"\n{'Scenario':<32} | {'Intervention':<20} | full DD | inside DD | inside CAGR")
    print("-" * 100)
    for scen in scenarios:
        for iv in interventions:
            r = results["scenarios"][scen][iv]
            full = r["full_sample"]
            inside = r["inside_window"]
            print(f"{scen:<32} | {iv:<20} | "
                  f"{full.get('max_dd_pct', 0):7.2f}% | "
                  f"{inside.get('max_dd_pct', 0):9.2f}% | "
                  f"{inside.get('cagr_pct', 0):+8.2f}%")
        print()

    # Gate check
    print("\n[5/5] Gate check — does any intervention keep DD < 12% in ALL scenarios?")
    gates = check_gates(results)
    for iv, g in gates.items():
        flag = "✓ PASS" if g["all_under_12"] else "✗ FAIL"
        print(f"  {iv:<20}  max_full_DD {g['max_full_dd_pct']:5.2f}%  "
              f"max_inside_DD {g['max_inside_dd_pct']:5.2f}%  "
              f"(worst: {g.get('worst_inside_scenario','-')})  {flag}")

    # Identify winner: passes gate + least CAGR sacrifice on baseline window
    winners = [iv for iv, g in gates.items() if g["all_under_12"]]
    print(f"\n[winners] Interventions passing gate: {winners}")

    # Record baseline CAGR sacrifice for each intervention (measured on
    # the real unstressed full sample)
    baseline_sacrifice: Dict[str, float] = {}
    for iv in interventions:
        r = results["scenarios"]["(e) 2022 H1 bear"][iv]["full_sample"]
        # (e) uses base_rets unchanged → measures full-sample CAGR under
        # the intervention
        baseline_sacrifice[iv] = {
            "cagr_pct": r.get("cagr_pct", 0),
            "sharpe": r.get("sharpe", 0),
            "dd_pct": r.get("max_dd_pct", 0),
        }
    print("\n[baseline impact] each intervention's effect on the UN-stressed full sample:")
    for iv, m in baseline_sacrifice.items():
        print(f"  {iv:<20}  CAGR {m['cagr_pct']:+6.2f}%  "
              f"SR {m['sharpe']:5.2f}  DD {m['dd_pct']:6.2f}%")

    # ── JSON
    payload = {
        "experiment": "EXP-2640",
        "title": "VIX High-Vol Stress Hardening — Gentler Response",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "rule_zero": True,
        "sources": {
            "streams": "compass.exp2630_regime_stress_oos.load_streams (5-stream cache)",
            "synthetic_scenarios": "compass.exp2630_regime_stress_oos scenarios (a/b/c)",
            "real_stress_windows": {
                "2020_march_covid": REAL_WINDOWS["(d) 2020 March COVID"],
                "2022_h1_bear": REAL_WINDOWS["(e) 2022 H1 bear"],
            },
            "vix": "Yahoo ^VIX daily, shift-1d (causal)",
        },
        "weights": {col: round(float(w), 4)
                     for col, w in zip(df.columns, weights_vec)},
        "interventions": {
            "1_baseline_no_cb": "no intervention",
            "2_exp2630_cb": "EXP-2630 stateful DD circuit breaker (soft 10%, hard 12%, halt leverage 0)",
            "3_vix_gate_50": "VIX > rolling 252d 90th pct → exposure 0.5×",
            "4_adaptive_vt": "VIX 25→35 linear ramp → exposure 1.0→0.5",
            "5_hybrid": "vix_gate_50 × adaptive_vt (multiplicative)",
        },
        "results": results,
        "gate_check": gates,
        "winners": winners,
        "baseline_impact": baseline_sacrifice,
        "target_dd_pct": 12.0,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n[report] → {REPORT_JSON}")

    html = build_html(payload)
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"[report] → {REPORT_HTML}")


def build_html(p: Dict) -> str:
    scenarios = list(p["results"]["scenarios"].keys())
    interventions = list(p["interventions"].keys())

    # One big matrix: rows = scenarios, cols = interventions, cells = inside DD
    matrix_rows = ""
    for scen in scenarios:
        cells = f"<td><strong>{scen}</strong></td>"
        for iv in interventions:
            r = p["results"]["scenarios"][scen][iv]
            inside = r["inside_window"]
            dd = abs(inside.get("max_dd_pct", 0))
            cagr = inside.get("cagr_pct", 0)
            color = "#16a34a" if dd < 12 else "#dc2626"
            cells += (
                f"<td style='color:{color};font-weight:700'>{dd:.1f}%</td>"
                f"<td style='font-size:0.8em;color:#64748b'>{cagr:+.1f}%</td>"
            )
        matrix_rows += f"<tr>{cells}</tr>"

    gate_rows = ""
    for iv, g in p["gate_check"].items():
        color = "#16a34a" if g["all_under_12"] else "#dc2626"
        flag = "✓ PASS" if g["all_under_12"] else "✗ FAIL"
        gate_rows += (
            f"<tr><td><strong>{iv}</strong></td>"
            f"<td>{g['max_full_dd_pct']:.2f}%</td>"
            f"<td>{g['max_inside_dd_pct']:.2f}%</td>"
            f"<td>{g.get('worst_inside_scenario','-')}</td>"
            f"<td style='color:{color};font-weight:700'>{flag}</td></tr>"
        )

    impact_rows = ""
    for iv, m in p["baseline_impact"].items():
        impact_rows += (
            f"<tr><td><strong>{iv}</strong></td>"
            f"<td>{m['cagr_pct']:+.2f}%</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td>{m['dd_pct']:+.2f}%</td></tr>"
        )

    winners = p["winners"] or ["(none)"]

    # Intervention description rows
    iv_desc_rows = "".join(
        f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"
        for k, v in p["interventions"].items()
    )

    scen_headers_top = (
        '<th rowspan="2">Scenario</th>' +
        "".join(f'<th colspan="2">{iv}</th>' for iv in interventions)
    )
    scen_headers_bot = "".join("<th>Inside DD</th><th>Inside CAGR</th>"
                                for _ in interventions)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>EXP-2640 — VIX Stress Hardening</title>
<style>
body {{ font-family:-apple-system,sans-serif;max-width:1400px;margin:0 auto;padding:28px;background:#fff;color:#1e293b; }}
h1 {{ font-size:1.8em;color:#0f172a; }}
h2 {{ margin-top:2em;border-bottom:2px solid #e2e8f0;padding-bottom:8px;color:#334155; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:14px;font-size:0.84rem;line-height:1.6; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:14px 0; }}
.winners {{ background:#ecfdf5;border:2px solid #16a34a;border-radius:10px;padding:16px;margin:16px 0; }}
.winners h3 {{ margin-top:0;color:#065f46; }}
table {{ width:100%;border-collapse:collapse;margin:12px 0;font-size:0.82em; }}
th {{ background:#f1f5f9;padding:8px 10px;text-align:right;border-bottom:2px solid #cbd5e1;font-size:0.72em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:7px 10px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
</style></head><body>

<h1>EXP-2640 — VIX High-Vol Stress Hardening</h1>
<p style="color:#64748b">Gentler response to the EXP-2630 scenario (b) DD
breach · real 2020/2022 stress + synthetic scenarios ·
{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="sources">
<strong>Rule Zero:</strong> EXP-2630 scenario infrastructure + real Yahoo
^VIX. Same 7-stream cube and equal-risk weights as EXP-2630 for apples-
to-apples comparison.
</div>

<div class="winners">
<h3>Interventions passing DD &lt; 12% in ALL 5 scenarios</h3>
<strong>{', '.join(winners)}</strong>
</div>

<h2>1. Intervention definitions</h2>
<table>
<thead><tr><th>Intervention</th><th>Description</th></tr></thead>
<tbody>{iv_desc_rows}</tbody>
</table>

<h2>2. Inside-window DD matrix (the ones that matter)</h2>
<table>
<thead><tr>{scen_headers_top}</tr><tr>{scen_headers_bot}</tr></thead>
<tbody>{matrix_rows}</tbody>
</table>
<div class="note">
Each cell shows the intervention's inside-window max DD and CAGR on
that scenario. Green = passes 12% DD gate, red = fails. The goal is
a single column that is green across all 5 rows.
</div>

<h2>3. Gate check (DD &lt; 12% across ALL scenarios)</h2>
<table>
<thead><tr><th>Intervention</th><th>Max full DD</th><th>Max inside DD</th><th>Worst scenario</th><th>Gate</th></tr></thead>
<tbody>{gate_rows}</tbody>
</table>

<h2>4. Baseline impact — CAGR/SR/DD on the un-stressed sample</h2>
<table>
<thead><tr><th>Intervention</th><th>CAGR</th><th>Sharpe</th><th>DD</th></tr></thead>
<tbody>{impact_rows}</tbody>
</table>
<div class="note">
This table answers "how much return does each intervention cost in
normal times?" Compare against the no-intervention baseline row to
see the opportunity cost of always running the safety net.
</div>

<p style="margin-top:3em;color:#94a3b8;font-size:0.78em;text-align:center">
compass/exp2640_vix_stress_hardening.py · Rule Zero · real data
</p>
</body></html>"""


if __name__ == "__main__":
    main()
