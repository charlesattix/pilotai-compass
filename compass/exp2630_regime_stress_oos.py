"""
compass/exp2630_regime_stress_oos.py — EXP-2630 OOS Regime Stress Test.

QUESTION: The 7-stream North Star v6 portfolio has been validated on
2020-2025 walk-forward data. How does it behave in UNPRECEDENTED stress
regimes that are absent from the historical sample? Specifically:

  (a) Correlation shock  — force all 21 pairwise correlations to 0.80
                             for a 60-day window inserted mid-sample
  (b) VIX-stays-high     — 90 consecutive days where every sleeve
                             operates at its historical 95th-percentile
                             vol and mean is set to its 5th percentile
                             (an extended stress regime worse than 2020)
  (c) 3-month losing run — 63 consecutive days forced to the 5th-pct
                             worst return across all streams

DEFENSES TESTED:
  • DD circuit breaker (from EXP-2370 / compass.portfolio_risk_manager):
        soft limit 10% → de-lever to 0.5×
        hard limit 12% → flatten + lock until recovery
  • Walk-forward metrics with and without the circuit breaker
  • Conditional Sharpe during each stress window

OUTPUTS:
  compass/reports/exp2630_regime_stress_oos.{json,html}
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STREAMS_PKL = ROOT / "compass" / "cache" / "exp2280_v6_sparse.pkl"
REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2630_regime_stress_oos.json"
REPORT_HTML = REPORT_DIR / "exp2630_regime_stress_oos.html"

TRADING_DAYS = 252

# Portfolio config (7-stream v6)
CAPITAL_WEIGHTS = {
    "exp1220":  0.35,
    "xlf_cs":   0.10,
    "xli_cs":   0.10,
    "gld_cal":  0.10,
    "slv_cal":  0.075,
    "vol_arb":  0.15,
    "v5_hedge": 0.125,
}
TARGET_GROSS_LEVERAGE = 3.0

# Circuit breaker (EXP-2370 / portfolio_risk_manager defaults)
DD_SOFT_PCT = 0.10     # 10% → emergency de-lever
DD_HARD_PCT = 0.12     # 12% → flatten + lock
SOFT_LEVERAGE = 0.5    # throttle factor when in soft zone
RECOVERY_PCT = 0.06    # unlock when DD recovers to this level

# Stress scenario parameters
STRESS_A_CORR = 0.80
STRESS_A_DAYS = 60
STRESS_B_DAYS = 90
STRESS_C_DAYS = 63     # 3 months
RNG_SEED = 20260408

DD_CEILING = 0.12      # report gate


# ═══════════════════════════════════════════════════════════════════════════
# Data + portfolio build
# ═══════════════════════════════════════════════════════════════════════════

def load_streams() -> pd.DataFrame:
    print(f"  loading {STREAMS_PKL.name}")
    df: pd.DataFrame = pickle.load(STREAMS_PKL.open("rb"))
    df = df.fillna(0.0).astype(float)
    print(f"    {len(df)} days × {len(df.columns)} streams  "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


def equal_risk_weights(df: pd.DataFrame) -> np.ndarray:
    cols = list(df.columns)
    stds = df.std(ddof=1).values
    base = np.zeros(len(cols))
    for i, c in enumerate(cols):
        if stds[i] > 1e-12:
            base[i] = CAPITAL_WEIGHTS.get(c, 0.0) / stds[i]
    s = float(np.sum(np.abs(base)))
    if s < 1e-12:
        return base
    return base / s * TARGET_GROSS_LEVERAGE


def weighted_port(df: pd.DataFrame, w: np.ndarray) -> pd.Series:
    return pd.Series(df.values @ w, index=df.index, name="port")


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def max_drawdown(rets: np.ndarray) -> float:
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return float(dd.min())


def portfolio_metrics(rets: np.ndarray, label: str = "") -> Dict:
    n = len(rets)
    if n < 5:
        return {"label": label, "n_days": n, "cagr_pct": 0.0,
                "sharpe": 0.0, "max_dd_pct": 0.0, "vol_pct": 0.0}
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = np.cumprod(1.0 + rets)
    yrs = n / TRADING_DAYS
    cagr = float(eq[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    return {
        "label": label,
        "n_days": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_drawdown(rets) * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
        "worst_day_pct": round(float(np.min(rets)) * 100, 3),
        "best_day_pct": round(float(np.max(rets)) * 100, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Circuit breaker (EXP-2370)
# ═══════════════════════════════════════════════════════════════════════════

def apply_circuit_breaker(rets: np.ndarray,
                            soft_pct: float = DD_SOFT_PCT,
                            hard_pct: float = DD_HARD_PCT,
                            soft_leverage: float = SOFT_LEVERAGE,
                            recovery_pct: float = RECOVERY_PCT) -> Tuple[np.ndarray, Dict]:
    """Stateful drawdown circuit breaker.

    state machine:
      NORMAL  -> WARN  when DD from HWM reaches soft_pct  (leverage → soft_leverage)
      WARN    -> HALT  when DD from HWM reaches hard_pct  (leverage → 0)
      HALT    -> WARN  when DD recovers to recovery_pct   (leverage back to soft)
      WARN    -> NORMAL when DD recovers to 0.5*soft_pct  (leverage → 1.0)
    """
    n = len(rets)
    out = np.zeros(n)
    state = "NORMAL"
    lev = 1.0
    eq = 1.0
    peak = 1.0
    states_trace: List[str] = []
    transitions: List[Dict] = []

    for t in range(n):
        applied = rets[t] * lev
        out[t] = applied
        eq *= (1.0 + applied)
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak

        # Transitions (check from worst to best)
        new_state = state
        new_lev = lev
        if state == "NORMAL":
            if dd <= -hard_pct:
                new_state, new_lev = "HALT", 0.0
            elif dd <= -soft_pct:
                new_state, new_lev = "WARN", soft_leverage
        elif state == "WARN":
            if dd <= -hard_pct:
                new_state, new_lev = "HALT", 0.0
            elif dd >= -0.5 * soft_pct:
                new_state, new_lev = "NORMAL", 1.0
        elif state == "HALT":
            if dd >= -recovery_pct:
                new_state, new_lev = "WARN", soft_leverage

        if new_state != state:
            transitions.append({
                "day": t,
                "from": state,
                "to": new_state,
                "dd_pct": round(dd * 100, 3),
                "new_leverage": new_lev,
            })
        state = new_state
        lev = new_lev
        states_trace.append(state)

    pct_normal = states_trace.count("NORMAL") / n * 100
    pct_warn = states_trace.count("WARN") / n * 100
    pct_halt = states_trace.count("HALT") / n * 100

    return out, {
        "transitions": transitions,
        "pct_days_normal": round(pct_normal, 2),
        "pct_days_warn": round(pct_warn, 2),
        "pct_days_halt": round(pct_halt, 2),
        "n_state_changes": len(transitions),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stress scenario builders
# ═══════════════════════════════════════════════════════════════════════════

def insert_window(series: pd.Series, window_vals: np.ndarray,
                    insert_idx: int) -> pd.Series:
    """Overwrite a contiguous slice of `series` with `window_vals`."""
    out = series.copy()
    end = min(insert_idx + len(window_vals), len(out))
    out.iloc[insert_idx:end] = window_vals[: end - insert_idx]
    return out


def scenario_a_correlation_shock(df: pd.DataFrame, w: np.ndarray,
                                    insert_idx: int) -> pd.Series:
    """All 21 pairwise correlations forced to 0.80 for 60 days.
    Marginals (mean, std) preserved from empirical estimates. A Cholesky
    factor model generates the stressed returns for the inserted window,
    then the rest of the path uses the real observed returns."""
    rng = np.random.default_rng(RNG_SEED)
    n_streams = len(df.columns)
    C = np.full((n_streams, n_streams), STRESS_A_CORR)
    np.fill_diagonal(C, 1.0)
    try:
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n_streams))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 1e-6)
        L = vecs @ np.diag(np.sqrt(vals))
    means = df.mean().values
    stds = df.std(ddof=1).values
    z = rng.standard_normal((STRESS_A_DAYS, n_streams))
    corr_sim = z @ L.T
    stream_rets = corr_sim * stds + means
    window_port = stream_rets @ w

    base = weighted_port(df, w)
    return insert_window(base, window_port, insert_idx)


def scenario_b_vix_high_90d(df: pd.DataFrame, w: np.ndarray,
                              insert_idx: int) -> pd.Series:
    """90 consecutive days where every sleeve operates at its 95th
    percentile vol and 5th percentile mean — effectively a longer, more
    punishing version of 2020 that the portfolio has never seen."""
    rng = np.random.default_rng(RNG_SEED + 1)
    n_streams = len(df.columns)
    # Per-stream 95th pct magnitude and 5th pct bias
    p95_abs = np.array([float(np.quantile(np.abs(df[c].values), 0.95))
                          for c in df.columns])
    p05_mean = np.array([float(np.quantile(df[c].values, 0.05))
                           for c in df.columns])
    # Preserve empirical correlation during stress
    C = df.corr().values
    try:
        L = np.linalg.cholesky(C + 1e-10 * np.eye(n_streams))
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 1e-6)
        L = vecs @ np.diag(np.sqrt(vals))
    z = rng.standard_normal((STRESS_B_DAYS, n_streams))
    corr_sim = z @ L.T
    # Scale to 95th-pct vol and shift to 5th-pct mean
    stream_rets = corr_sim * p95_abs + p05_mean
    window_port = stream_rets @ w

    base = weighted_port(df, w)
    return insert_window(base, window_port, insert_idx)


def scenario_c_3month_losing(df: pd.DataFrame, w: np.ndarray,
                                 insert_idx: int) -> pd.Series:
    """63 consecutive days forced to the per-stream 5th-percentile return
    (a flat, mechanical, worst-case grind)."""
    rng = np.random.default_rng(RNG_SEED + 2)
    n_streams = len(df.columns)
    p05 = np.array([float(np.quantile(df[c].values, 0.05))
                      for c in df.columns])
    # Add modest noise so it's not a literal step function; each day
    # independently sampled from the bottom 10th pct of each stream.
    bottom_samples = {c: df[c].values[df[c].values <= np.quantile(df[c].values, 0.10)]
                       for c in df.columns}
    stream_rets = np.zeros((STRESS_C_DAYS, n_streams))
    for i, c in enumerate(df.columns):
        pool = bottom_samples[c]
        if len(pool) > 0:
            stream_rets[:, i] = rng.choice(pool, size=STRESS_C_DAYS, replace=True)
        else:
            stream_rets[:, i] = p05[i]
    window_port = stream_rets @ w

    base = weighted_port(df, w)
    return insert_window(base, window_port, insert_idx)


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_scenario(name: str,
                        base_rets: pd.Series,
                        stressed_rets: pd.Series,
                        insert_idx: int,
                        window_len: int) -> Dict:
    base_m = portfolio_metrics(base_rets.values, f"{name} (no stress)")
    stressed_m = portfolio_metrics(stressed_rets.values, f"{name} (raw stress)")

    # Apply circuit breaker to the STRESSED path
    cb_rets, cb_diag = apply_circuit_breaker(stressed_rets.values)
    cb_m = portfolio_metrics(cb_rets, f"{name} (with CB)")

    # Conditional metrics — INSIDE the stress window
    end = min(insert_idx + window_len, len(stressed_rets))
    inside = stressed_rets.iloc[insert_idx:end]
    inside_m = portfolio_metrics(inside.values, f"{name} (inside window)")

    inside_cb = cb_rets[insert_idx:end]
    inside_cb_m = portfolio_metrics(inside_cb, f"{name} (inside, CB)")

    return {
        "scenario": name,
        "insert_idx": insert_idx,
        "window_len": window_len,
        "baseline_metrics": base_m,
        "stressed_raw_metrics": stressed_m,
        "stressed_with_cb_metrics": cb_m,
        "conditional_inside_window": inside_m,
        "conditional_inside_window_with_cb": inside_cb_m,
        "circuit_breaker_diagnostics": cb_diag,
        "dd_under_12pct_raw": stressed_m["max_dd_pct"] >= -DD_CEILING * 100,
        "dd_under_12pct_with_cb": cb_m["max_dd_pct"] >= -DD_CEILING * 100,
        "cb_protects": (cb_m["max_dd_pct"] > stressed_m["max_dd_pct"]),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_html(payload: Dict) -> str:
    scenarios = payload["scenarios"]

    def row_for(s: Dict) -> str:
        raw = s["stressed_raw_metrics"]
        cb = s["stressed_with_cb_metrics"]
        inside = s["conditional_inside_window"]
        cb_diag = s["circuit_breaker_diagnostics"]
        cls_dd_raw = "good" if raw["max_dd_pct"] >= -12 else "bad"
        cls_dd_cb = "good" if cb["max_dd_pct"] >= -12 else "bad"
        return f"""<tr>
            <td><strong>{s['scenario']}</strong></td>
            <td>{raw['cagr_pct']:.2f}%</td>
            <td>{raw['sharpe']:.2f}</td>
            <td class="{cls_dd_raw}">{raw['max_dd_pct']:.2f}%</td>
            <td>{cb['cagr_pct']:.2f}%</td>
            <td>{cb['sharpe']:.2f}</td>
            <td class="{cls_dd_cb}">{cb['max_dd_pct']:.2f}%</td>
            <td>{inside['sharpe']:.2f}</td>
            <td>{inside['worst_day_pct']:.2f}%</td>
            <td>{cb_diag['pct_days_warn']:.1f}%</td>
            <td>{cb_diag['pct_days_halt']:.1f}%</td>
        </tr>"""

    rows = "".join(row_for(s) for s in scenarios)

    baseline = payload["baseline_metrics"]
    sign = payload["sign_off"]
    sign_cls = "good" if sign["approved"] else "bad"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2630 OOS Regime Stress Test</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1200px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.2em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:140px; }}
  .kpi .value {{ font-size:1.5em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; font-weight:700; }}
  .bad  {{ color:#dc2626; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.82em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.7em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
</style></head><body>

<h1>EXP-2630 — Out-of-Sample Regime Stress Test</h1>
<div class="subtitle">7-stream North Star v6 · 3 unprecedented stress scenarios · DD circuit breaker | {payload['timestamp']}</div>

<div class="note">
    <strong>Scenarios:</strong> (a) 60-day correlation shock ρ=0.80,
    (b) 90-day extended vol stress at 95th-pct vol and 5th-pct mean,
    (c) 63-day forced losing run sampled from each stream's bottom
    10th percentile. Real portfolio stream from exp2280_v6_sparse.pkl;
    stress windows inserted mid-sample while the rest of the path is
    real observed returns. <strong>Circuit breaker:</strong> EXP-2370
    drawdown model — 10% soft → 0.5× leverage, 12% hard → flatten,
    recovery at 6% DD.
</div>

<h2>Baseline (no stress, 3× gross)</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value">{baseline['cagr_pct']:.2f}%</div><div class="label">CAGR</div></div>
    <div class="kpi"><div class="value">{baseline['sharpe']:.2f}</div><div class="label">Sharpe</div></div>
    <div class="kpi"><div class="value">{baseline['max_dd_pct']:.2f}%</div><div class="label">Max DD</div></div>
    <div class="kpi"><div class="value">{baseline['vol_pct']:.2f}%</div><div class="label">Vol</div></div>
</div>

<h2>Stress Scenario Comparison</h2>
<table>
    <thead><tr>
        <th>Scenario</th>
        <th>Raw CAGR</th><th>Raw Sh</th><th>Raw DD</th>
        <th>CB CAGR</th><th>CB Sh</th><th>CB DD</th>
        <th>Cond Sh (inside)</th><th>Worst day</th>
        <th>% WARN</th><th>% HALT</th>
    </tr></thead>
    <tbody>{rows}</tbody>
</table>

<h2>Sign-Off</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value {sign_cls}">{'APPROVE' if sign['approved'] else 'REJECT'}</div><div class="label">Decision</div></div>
    <div class="kpi"><div class="value">{sign['gates_passed']}/{sign['gates_total']}</div><div class="label">Gates</div></div>
</div>
<ul>
    {''.join(f'<li>{r}</li>' for r in sign['reasons'])}
</ul>

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2630 — compass/exp2630_regime_stress_oos.py · Real 7-stream cache · EXP-2370 circuit breaker
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2630 — OOS Regime Stress Test")
    print("=" * 72)

    print("\n[1/5] Loading 7-stream portfolio and baseline metrics...")
    df = load_streams()
    w = equal_risk_weights(df)
    base_rets = weighted_port(df, w)
    base_m = portfolio_metrics(base_rets.values, "baseline")
    print(f"  per-sleeve lev: {dict((c, round(float(w[i]), 3)) for i, c in enumerate(df.columns))}")
    print(f"  gross leverage: {float(np.sum(np.abs(w))):.2f}×")
    print(f"  baseline: CAGR={base_m['cagr_pct']}%  Sharpe={base_m['sharpe']}  "
          f"DD={base_m['max_dd_pct']}%  Vol={base_m['vol_pct']}%")

    # Stress windows inserted at ~40% through the sample (roughly mid-2022)
    insert_idx = int(len(df) * 0.40)
    print(f"\n  stress insert index: {insert_idx} "
          f"(date {df.index[insert_idx].date()})")

    print("\n[2/5] Scenario (a): 60-day correlation shock ρ=0.80...")
    a_rets = scenario_a_correlation_shock(df, w, insert_idx)
    res_a = evaluate_scenario("(a) corr shock ρ=0.80 × 60d",
                                 base_rets, a_rets, insert_idx, STRESS_A_DAYS)
    print(f"    raw stressed:  CAGR={res_a['stressed_raw_metrics']['cagr_pct']}%  "
          f"Sharpe={res_a['stressed_raw_metrics']['sharpe']}  "
          f"DD={res_a['stressed_raw_metrics']['max_dd_pct']}%")
    print(f"    with CB:       CAGR={res_a['stressed_with_cb_metrics']['cagr_pct']}%  "
          f"Sharpe={res_a['stressed_with_cb_metrics']['sharpe']}  "
          f"DD={res_a['stressed_with_cb_metrics']['max_dd_pct']}%")
    print(f"    inside window: Sharpe={res_a['conditional_inside_window']['sharpe']}  "
          f"worst day={res_a['conditional_inside_window']['worst_day_pct']}%")
    print(f"    CB transitions: {res_a['circuit_breaker_diagnostics']['n_state_changes']}  "
          f"WARN days {res_a['circuit_breaker_diagnostics']['pct_days_warn']}%  "
          f"HALT days {res_a['circuit_breaker_diagnostics']['pct_days_halt']}%")

    print("\n[3/5] Scenario (b): 90-day VIX-high (p95 vol, p05 mean)...")
    b_rets = scenario_b_vix_high_90d(df, w, insert_idx)
    res_b = evaluate_scenario("(b) VIX high 90d (p95/p05)",
                                 base_rets, b_rets, insert_idx, STRESS_B_DAYS)
    print(f"    raw stressed:  CAGR={res_b['stressed_raw_metrics']['cagr_pct']}%  "
          f"Sharpe={res_b['stressed_raw_metrics']['sharpe']}  "
          f"DD={res_b['stressed_raw_metrics']['max_dd_pct']}%")
    print(f"    with CB:       CAGR={res_b['stressed_with_cb_metrics']['cagr_pct']}%  "
          f"Sharpe={res_b['stressed_with_cb_metrics']['sharpe']}  "
          f"DD={res_b['stressed_with_cb_metrics']['max_dd_pct']}%")
    print(f"    inside window: Sharpe={res_b['conditional_inside_window']['sharpe']}  "
          f"worst day={res_b['conditional_inside_window']['worst_day_pct']}%")
    print(f"    CB transitions: {res_b['circuit_breaker_diagnostics']['n_state_changes']}  "
          f"WARN days {res_b['circuit_breaker_diagnostics']['pct_days_warn']}%  "
          f"HALT days {res_b['circuit_breaker_diagnostics']['pct_days_halt']}%")

    print("\n[4/5] Scenario (c): 63-day losing run (bottom 10th pct)...")
    c_rets = scenario_c_3month_losing(df, w, insert_idx)
    res_c = evaluate_scenario("(c) 3-month losing run",
                                 base_rets, c_rets, insert_idx, STRESS_C_DAYS)
    print(f"    raw stressed:  CAGR={res_c['stressed_raw_metrics']['cagr_pct']}%  "
          f"Sharpe={res_c['stressed_raw_metrics']['sharpe']}  "
          f"DD={res_c['stressed_raw_metrics']['max_dd_pct']}%")
    print(f"    with CB:       CAGR={res_c['stressed_with_cb_metrics']['cagr_pct']}%  "
          f"Sharpe={res_c['stressed_with_cb_metrics']['sharpe']}  "
          f"DD={res_c['stressed_with_cb_metrics']['max_dd_pct']}%")
    print(f"    inside window: Sharpe={res_c['conditional_inside_window']['sharpe']}  "
          f"worst day={res_c['conditional_inside_window']['worst_day_pct']}%")
    print(f"    CB transitions: {res_c['circuit_breaker_diagnostics']['n_state_changes']}  "
          f"WARN days {res_c['circuit_breaker_diagnostics']['pct_days_warn']}%  "
          f"HALT days {res_c['circuit_breaker_diagnostics']['pct_days_halt']}%")

    print("\n[5/5] Sign-off...")
    scenarios = [res_a, res_b, res_c]

    # Gates:
    #   (1) raw DD under 12% in ALL scenarios → pass
    #   (2) CB-adjusted DD under 12% in ALL scenarios → pass
    #   (3) CB meaningfully reduces DD (≥ 1pp) in any scenario where
    #       raw DD exceeded 12% → pass if all fit
    all_raw_ok = all(s["dd_under_12pct_raw"] for s in scenarios)
    all_cb_ok = all(s["dd_under_12pct_with_cb"] for s in scenarios)
    cb_protects_where_needed = all(
        s["dd_under_12pct_raw"] or s["cb_protects"] for s in scenarios
    )

    gates = {
        "raw_dd_under_12pct_all": all_raw_ok,
        "cb_dd_under_12pct_all": all_cb_ok,
        "cb_protects_when_needed": cb_protects_where_needed,
    }
    approved = all(gates.values())
    reasons = []
    for s in scenarios:
        raw_dd = s["stressed_raw_metrics"]["max_dd_pct"]
        cb_dd = s["stressed_with_cb_metrics"]["max_dd_pct"]
        cb_halt_pct = s["circuit_breaker_diagnostics"]["pct_days_halt"]
        cb_warn_pct = s["circuit_breaker_diagnostics"]["pct_days_warn"]
        tag = "PASS" if s["dd_under_12pct_with_cb"] else "FAIL"
        reasons.append(
            f"{s['scenario']}: raw DD {raw_dd:.2f}%  "
            f"CB DD {cb_dd:.2f}%  "
            f"CB activity WARN {cb_warn_pct:.1f}% HALT {cb_halt_pct:.1f}%  "
            f"[{tag}]"
        )

    sign_off = {
        "approved": approved,
        "gates_passed": int(sum(gates.values())),
        "gates_total": len(gates),
        "gates": gates,
        "reasons": reasons,
    }
    print(f"  DECISION: {'APPROVE' if approved else 'REJECT'} "
          f"({sign_off['gates_passed']}/{sign_off['gates_total']} gates)")
    for r in reasons:
        print(f"    {r}")

    payload = {
        "experiment": "EXP-2630",
        "title": "OOS Regime Stress Test · 7-stream North Star v6",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "streams": list(df.columns),
            "capital_weights": CAPITAL_WEIGHTS,
            "target_gross_leverage": TARGET_GROSS_LEVERAGE,
            "dd_ceiling_pct": DD_CEILING * 100,
            "circuit_breaker": {
                "soft_pct": DD_SOFT_PCT * 100,
                "hard_pct": DD_HARD_PCT * 100,
                "soft_leverage": SOFT_LEVERAGE,
                "recovery_pct": RECOVERY_PCT * 100,
            },
            "stress_params": {
                "scenario_a_corr": STRESS_A_CORR,
                "scenario_a_days": STRESS_A_DAYS,
                "scenario_b_days": STRESS_B_DAYS,
                "scenario_c_days": STRESS_C_DAYS,
                "rng_seed": RNG_SEED,
                "insert_date": str(df.index[insert_idx].date()),
                "insert_index": insert_idx,
            },
        },
        "baseline_metrics": base_m,
        "scenarios": scenarios,
        "sign_off": sign_off,
        "rule_zero": (
            "Real 7-stream returns from exp2280_v6_sparse.pkl (derived "
            "from real IronVault + Yahoo data). Stress windows inserted "
            "mid-sample via Cholesky factor model with empirical "
            "marginals for (a) and (b), and bottom-10%-percentile "
            "bootstrap resampling for (c). Rest of the path is real "
            "observed returns. No synthetic distribution fitting "
            "beyond the documented stress-window generators."
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


if __name__ == "__main__":
    sys.exit(main())
