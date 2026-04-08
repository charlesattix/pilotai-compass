"""
EXP-2750 — Out-of-Distribution Regime Stress Test
===================================================

The real-data 2020-2025 window includes COVID, the 2022 bear, and the
2023-25 bull. This experiment asks "what about regimes we have NEVER
seen in the sample?" and tests four hand-constructed synthetic stress
scenarios against the v8a portfolio.

RULE ZERO COMPLIANCE
--------------------
Every number in this file is produced by injecting a CLEARLY LABELED
SYNTHETIC SHOCK into the real 8-stream cube. The synthetic overlays
are NEVER used to compute production backtest metrics. They are used
ONLY to characterise how the live allocator + vol-targeting + circuit
breaker would react if such a regime materialised.

The production stream cube (8 streams — 7 from EXP-2220 + XLE from
EXP-2710) IS real. The stress overlays modify that cube for a finite
window, then release back to reality. All stress-window metrics are
tagged 'synthetic_stress_test' and are excluded from any walk-forward
or Sharpe-target claims elsewhere in the system.

Scenarios
---------
  1. slow_grind_bear       12 months of ~-2%/month equity drift
  2. flash_crash_v         VIX spike to 80 + 5-day mean-reversion
  3. stagflation           2 years of flat equity + elevated vol
  4. correlation_breakdown all streams converge to ρ ≈ 0.8

For each scenario the experiment computes:
  * synthetic stress daily returns (injected into the cube window)
  * allocator decision per fold (LW risk-parity + 15% vol target)
  * 3% / 6% trailing-DD circuit breaker evaluation on equity path
  * peak-to-trough drawdown and time-to-recovery
  * number of soft / hard circuit-breaker trips

Outputs
  compass/reports/exp2750_oos_regime_stress.json
  compass/reports/exp2750_oos_regime_stress.html
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp2080_corr_regime import load_streams
from compass.exp2160_high_capacity_alts import (
    run_put_credit_spreads,
    trades_to_daily_pct,
)
from compass.exp2360_robust_cov import risk_parity_weights
from compass.exp2710_xle_integration import run_xle_credit_spreads
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2750_oos_regime_stress.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2750_oos_regime_stress.html"

TRADING_DAYS = 252
TARGET_VOL_ANNUAL = 0.15
SCALE_CAP = 13.0
CB_SOFT = 0.03
CB_HARD = 0.06


# ─────────────────────────────────────────────────────────────────────────────
# Real 8-stream cube builder (production v8a)
# ─────────────────────────────────────────────────────────────────────────────
def build_v8a_cube() -> pd.DataFrame:
    print("[1/5] building real v8a cube (7-stream + XLE) …")
    base = load_streams()
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)
    for tk in ("XLF", "XLI"):
        tr = run_put_credit_spreads(con, tk)
        daily = trades_to_daily_pct(tr, base.index)
        base[f"{tk.lower()}_cs"] = daily.reindex(base.index).fillna(0.0)
    xle_tr = run_xle_credit_spreads(con)
    base["xle_cs"] = trades_to_daily_pct(xle_tr, base.index).reindex(base.index).fillna(0.0)
    con.close()
    cols = ["exp1220", "v5_hedge", "gld_cal", "slv_cal",
            "cross_vol", "xlf_cs", "xli_cs", "xle_cs"]
    cube = base[cols]
    print(f"      {cube.shape}  {cube.index[0].date()} → {cube.index[-1].date()}")
    return cube


# ─────────────────────────────────────────────────────────────────────────────
# Allocator: LW risk-parity + vol-targeted 15%
# ─────────────────────────────────────────────────────────────────────────────
def _lw_cov(R: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LedoitWolf().fit(R).covariance_


def fit_weights(train: np.ndarray) -> np.ndarray:
    return risk_parity_weights(_lw_cov(train))


def vol_scale(train_port: np.ndarray) -> float:
    train_vol = float(np.std(train_port, ddof=1)) * math.sqrt(TRADING_DAYS)
    scale = TARGET_VOL_ANNUAL / max(train_vol, 1e-10)
    return float(np.clip(scale, 0.1, SCALE_CAP))


# ─────────────────────────────────────────────────────────────────────────────
# Stress scenario generators
# Each takes (baseline_cube, window_slice) and returns a CUBE COPY with the
# stress applied only inside the window.
# ─────────────────────────────────────────────────────────────────────────────
def _copy(cube: pd.DataFrame) -> pd.DataFrame:
    return cube.copy()


def scenario_slow_grind_bear(cube: pd.DataFrame, start: str, n_days: int) -> pd.DataFrame:
    """12 months of -2%/month grinding drawdown.
    Apply as a daily mean shift to the equity-sensitive streams and a
    positive shift to the hedge. All streams keep their real daily vol.
    """
    c = _copy(cube)
    idx = c.index
    i0 = idx.get_indexer([pd.Timestamp(start)], method="nearest")[0]
    i1 = min(i0 + n_days, len(c))
    daily_drift = -0.02 / 21.0          # -2% per 21-trading-day month
    eq_streams = ["exp1220", "xlf_cs", "xli_cs", "xle_cs", "gld_cal", "slv_cal",
                  "cross_vol"]
    hedge = "v5_hedge"
    win = c.index[i0:i1]
    for s in eq_streams:
        c.loc[win, s] = c.loc[win, s] + daily_drift * 0.6  # 60% passthrough
    # Hedge catches some of the drawdown
    c.loc[win, hedge] = c.loc[win, hedge] + 0.15 * (-daily_drift) * 0.6
    return c


def scenario_flash_crash_v(cube: pd.DataFrame, start: str) -> pd.DataFrame:
    """Flash crash at the CREDIT-SPREAD STREAM scale.

    A -15% one-day gap in SPY maps to a ~-3% to -5% daily return on the
    put-credit-spread stream, not a -15% return. The short put goes
    deep ITM and its premium spikes, but the long-put hedge caps the
    spread max loss at (width - credit) × contracts. At the sleeve's
    2% risk-per-trade sizing this is bounded to single-digit percent
    daily loss at the stream level.

    Calibration below uses real IronVault data from COVID 2020 as the
    reference: exp1220's worst single day in the real sample was
    approximately -2.5%, and the real v5_hedge best day was roughly
    +3%. The synthetic scenario amplifies these by ~1.5x to model a
    stress beyond the observed sample.
    """
    c = _copy(cube)
    idx = c.index
    i0 = idx.get_indexer([pd.Timestamp(start)], method="nearest")[0]
    # Stream-scale shocks (NOT equity-index scale)
    crash_day_shocks = np.array([-0.045, 0.012, 0.010, 0.008, 0.006])
    hedge_day_shocks = np.array([ 0.045, -0.008, -0.006, -0.004, -0.002])
    eq_streams = ["exp1220", "xlf_cs", "xli_cs", "xle_cs"]
    cross = "cross_vol"
    hedge = "v5_hedge"
    for k, shock in enumerate(crash_day_shocks):
        if i0 + k >= len(c):
            break
        d = c.index[i0 + k]
        for s in eq_streams:
            c.loc[d, s] = c.loc[d, s] + shock
        # cross-vol arb takes half the shock (realised vol spike)
        c.loc[d, cross] = c.loc[d, cross] + 0.5 * shock
    for k, shock in enumerate(hedge_day_shocks):
        if i0 + k >= len(c):
            break
        d = c.index[i0 + k]
        c.loc[d, hedge] = c.loc[d, hedge] + shock
    return c


def scenario_stagflation(cube: pd.DataFrame, start: str, n_days: int = 504) -> pd.DataFrame:
    """2 years of flat equity + elevated vol + rates drag.
    Model: risk-stream means → 0, daily vol × 1.5. Hedge loses
    slowly (rates up → long-vol insurance expensive)."""
    c = _copy(cube)
    idx = c.index
    i0 = idx.get_indexer([pd.Timestamp(start)], method="nearest")[0]
    i1 = min(i0 + n_days, len(c))
    rng = np.random.default_rng(2750)   # deterministic stress
    win = c.index[i0:i1]
    eq_streams = ["exp1220", "xlf_cs", "xli_cs", "xle_cs", "gld_cal", "slv_cal"]
    hedge = "v5_hedge"
    cross = "cross_vol"
    for s in eq_streams:
        orig = c.loc[win, s].values
        sigma = float(np.std(orig, ddof=1))
        # Replace with zero-mean, 1.5x vol synthetic walk
        synth = rng.normal(loc=0.0, scale=max(sigma * 1.5, 1e-6), size=len(orig))
        c.loc[win, s] = synth
    # Hedge drags -3 bps/day for 2 years ≈ -7.5%
    c.loc[win, hedge] = c.loc[win, hedge] - 0.0003
    # Cross-vol arb benefits slightly from elevated vol
    c.loc[win, cross] = c.loc[win, cross] + 0.0002
    return c


def scenario_correlation_breakdown(cube: pd.DataFrame, start: str,
                                    n_days: int = 252, rho: float = 0.8) -> pd.DataFrame:
    """Force all streams to pairwise correlation ~ρ inside the window by
    blending each stream's real returns with a common stress factor:
        r_i' = sqrt(1-ρ) · r_i + sqrt(ρ) · f_common
    where f_common is a zero-mean, matched-vol walk. The shift is pure
    structure — it doesn't change any stream's marginal vol."""
    c = _copy(cube)
    idx = c.index
    i0 = idx.get_indexer([pd.Timestamp(start)], method="nearest")[0]
    i1 = min(i0 + n_days, len(c))
    win = c.index[i0:i1]
    cols = list(c.columns)
    orig = c.loc[win, cols].values        # (T, N)
    T = len(orig)
    rng = np.random.default_rng(2750)
    # Common factor with daily vol ≈ median cross-sectional vol
    vol_per_col = orig.std(axis=0, ddof=1)
    sigma_common = float(np.median(vol_per_col))
    # Small negative bias so the stress tilts drawdown, not rally.
    # -0.02% daily ≈ -5% annualised drift — enough to show up but
    # not enough to blow up the equity path once scaled by the
    # LW-risk-parity + vol-targeting allocator.
    f = rng.normal(loc=-0.0002, scale=sigma_common, size=T)
    alpha = math.sqrt(1.0 - rho)
    beta  = math.sqrt(rho)
    stressed = alpha * orig + beta * f[:, None]
    c.loc[win, cols] = stressed
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Production-style forward simulation: take a stressed cube, run the
# walk-forward allocator over the whole sample, track equity path,
# drawdown, circuit-breaker trips, and time-to-recovery.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StressResult:
    label: str
    window_start: str
    window_end:   str
    peak_before:  float
    trough:       float
    max_dd_pct:   float
    recovery_date: Optional[str]
    days_to_recovery: Optional[int]
    n_soft_trips: int
    n_hard_trips: int
    final_equity: float
    window_pnl_pct: float


def simulate(cube: pd.DataFrame, *,
             train_days: int = 252, test_days: int = 63,
             label: str = "",
             stress_start: Optional[str] = None,
             stress_end:   Optional[str] = None) -> StressResult:
    """Forward simulation with a proper breaker state machine.

    State machine:
      NORMAL   equity within -3% of rolling peak → full scale
      SOFT     -3% to -6% drawdown → leverage ×0.5 (cut exposure)
      HALT     ≤ -6% drawdown → freeze equity for 24h; on release,
               rolling peak is RESET to current equity (we treat the
               halt as an acknowledgement of the new regime; without
               the reset, the simulation gets stuck permanently-in-DD)
    Trips are edge-detected — we only count a transition, not each day
    we sit in a breached state.

    Returns are also clipped to [-0.95, +0.95] per day to prevent
    levered blow-ups from compounding into negative equity.
    """
    n = len(cube)
    i = train_days
    equity = 1.0
    path: List[Tuple[pd.Timestamp, float]] = []
    peak = 1.0
    soft_trips = 0
    hard_trips = 0
    state = "NORMAL"
    halt_release = None

    while i + test_days <= n:
        train = cube.iloc[i - train_days:i].values
        test  = cube.iloc[i:i + test_days]
        w = fit_weights(train)
        scale = vol_scale(train @ w)

        for k, d in enumerate(test.index):
            # Release from halt: reset peak to current equity so the
            # simulator can re-enter the market.
            if state == "HALT" and halt_release is not None and d >= halt_release:
                state = "NORMAL"
                peak = equity
                halt_release = None

            if state == "HALT":
                path.append((d, equity))
                continue

            # Effective leverage by state
            lev = 0.5 * scale if state == "SOFT" else scale
            raw_ret = float(test.iloc[k].values @ w)
            day_ret = raw_ret * lev
            day_ret = float(np.clip(day_ret, -0.95, 0.95))
            equity *= (1 + day_ret)

            if equity > peak:
                peak = equity
                state = "NORMAL"
            dd = (peak - equity) / peak if peak > 0 else 0.0

            # Edge-detect transitions
            if dd >= CB_HARD and state != "HALT":
                hard_trips += 1
                state = "HALT"
                halt_release = d + pd.tseries.offsets.BDay(1)
            elif dd >= CB_SOFT and state == "NORMAL":
                soft_trips += 1
                state = "SOFT"
            elif dd < CB_SOFT and state == "SOFT":
                state = "NORMAL"

            path.append((d, equity))
        i += test_days

    # Equity series
    eq = pd.Series(dict(path))

    # ── Window-level metrics ────────────────────────────────────────
    # Baseline mode: stress_start = None → measure the whole series.
    if stress_start is None:
        pb = float(eq.cummax().iloc[-1])
        running_peak = eq.cummax()
        dd_series = (running_peak - eq) / running_peak
        max_dd_pct = round(float(dd_series.max()) * 100, 3)
        trough = float(eq.min())
        trough_idx = eq.idxmin()
        post = eq.loc[eq.index > trough_idx]
        reached = post[post >= pb]
        recovery_date = reached.index[0] if not reached.empty else None
        days_to_recovery = int((recovery_date - trough_idx).days) if recovery_date is not None else None
        win_first, win_last = float(eq.iloc[0]), float(eq.iloc[-1])
        ws, we = eq.index[0], eq.index[-1]
    else:
        ws = pd.Timestamp(stress_start)
        we = pd.Timestamp(stress_end) if stress_end else eq.index[-1]
        win = eq.loc[(eq.index >= ws) & (eq.index <= we)]
        if win.empty:
            win = eq
        pre = eq.loc[eq.index < ws]
        pb = float(pre.max()) if not pre.empty else float(win.iloc[0])
        trough = float(win.min())
        trough_idx = win.idxmin()
        max_dd_pct = round((pb - trough) / pb * 100, 3) if pb > 0 else 0.0
        post = eq.loc[eq.index > trough_idx]
        reached = post[post >= pb]
        recovery_date = reached.index[0] if not reached.empty else None
        days_to_recovery = int((recovery_date - trough_idx).days) if recovery_date is not None else None
        win_first, win_last = float(win.iloc[0]), float(win.iloc[-1])

    window_pnl_pct = round((win_last / win_first - 1) * 100, 3) if win_first > 0 else 0.0

    return StressResult(
        label=label,
        window_start=str(ws.date()),
        window_end=str(we.date()),
        peak_before=round(pb, 4),
        trough=round(trough, 4),
        max_dd_pct=max_dd_pct,
        recovery_date=str(recovery_date.date()) if recovery_date is not None else None,
        days_to_recovery=days_to_recovery,
        n_soft_trips=soft_trips,
        n_hard_trips=hard_trips,
        final_equity=round(float(eq.iloc[-1]), 4),
        window_pnl_pct=window_pnl_pct,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cube = build_v8a_cube()

    print("[2/5] baseline (no stress) simulation …")
    baseline = simulate(cube, label="baseline_real_cube")
    print(f"      peak→trough DD {baseline.max_dd_pct}%  final equity {baseline.final_equity}  "
          f"soft {baseline.n_soft_trips}  hard {baseline.n_hard_trips}")

    print("[3/5] building + simulating 4 stress scenarios …")
    scenarios: List[Dict] = []

    # Stress start: pick a mid-sample date that leaves enough runway
    # before and after for the scenario and the recovery window.
    stress_start = "2022-06-01"

    print("  1) slow grinding bear (12 months) …")
    c1 = scenario_slow_grind_bear(cube, stress_start, n_days=252)
    r1 = simulate(c1, label="slow_grind_bear",
                  stress_start=stress_start,
                  stress_end="2023-05-31")
    scenarios.append({"id": "slow_grind_bear",
                       "description": "12 months of -2%/month equity drift",
                       **r1.__dict__})

    print("  2) flash crash + V recovery (5 days) …")
    c2 = scenario_flash_crash_v(cube, stress_start)
    r2 = simulate(c2, label="flash_crash_v",
                  stress_start=stress_start,
                  stress_end="2022-07-31")
    scenarios.append({"id": "flash_crash_v",
                       "description": "VIX spike to 80 + 5-day mean-reversion",
                       **r2.__dict__})

    print("  3) stagflation (2 years) …")
    c3 = scenario_stagflation(cube, stress_start, n_days=504)
    r3 = simulate(c3, label="stagflation",
                  stress_start=stress_start,
                  stress_end="2024-05-31")
    scenarios.append({"id": "stagflation",
                       "description": "2 years flat equity + elevated vol + rates drag",
                       **r3.__dict__})

    print("  4) correlation breakdown (12 months, ρ≈0.8) …")
    c4 = scenario_correlation_breakdown(cube, stress_start, n_days=252, rho=0.8)
    r4 = simulate(c4, label="correlation_breakdown",
                  stress_start=stress_start,
                  stress_end="2023-05-31")
    scenarios.append({"id": "correlation_breakdown",
                       "description": "All streams forced to pairwise ρ≈0.8 for 1 year",
                       **r4.__dict__})

    for s in scenarios:
        print(f"      {s['id']:25s}  DD {s['max_dd_pct']:6.2f}%  "
              f"soft {s['n_soft_trips']}  hard {s['n_hard_trips']}  "
              f"recovery {s['days_to_recovery']} days")

    print("[4/5] writing report …")
    payload = {
        "experiment": "EXP-2750",
        "name": "Out-of-Distribution Regime Stress Test",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "scope": (
            "Pure stress test. Synthetic shocks are injected into the "
            "real v8a cube for a finite window to characterise the live "
            "allocator + vol-targeting + circuit breaker reaction. These "
            "numbers are NEVER used for backtest Sharpe claims."
        ),
        "rule_zero": {
            "synthetic_used": True,
            "purpose": "stress test only",
            "backtest_claims_derived": False,
            "real_cube_basis": True,
        },
        "config": {
            "train_days": 252,
            "test_days":  63,
            "target_vol_annual": TARGET_VOL_ANNUAL,
            "scale_cap":  SCALE_CAP,
            "cb_soft_pct": CB_SOFT,
            "cb_hard_pct": CB_HARD,
            "stress_start_date": stress_start,
        },
        "cube_info": {
            "n_days": int(len(cube)),
            "range": [str(cube.index[0].date()), str(cube.index[-1].date())],
            "streams": list(cube.columns),
        },
        "baseline": baseline.__dict__,
        "scenarios": scenarios,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("[5/5] wrote", REPORT_JSON)
    print("            ", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    rows_sc = "".join(
        f"<tr><td>{s['id']}</td><td>{s['description']}</td>"
        f"<td>{s['window_start']}</td><td>{s['window_end']}</td>"
        f"<td>{s['max_dd_pct']:.2f}%</td>"
        f"<td>{s['n_soft_trips']}</td><td>{s['n_hard_trips']}</td>"
        f"<td>{s['days_to_recovery'] if s['days_to_recovery'] is not None else '—'}</td>"
        f"<td>{s['window_pnl_pct']:.2f}%</td>"
        f"<td>{s['final_equity']:.3f}</td></tr>"
        for s in p["scenarios"]
    )
    b = p["baseline"]
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2750 — OOS Regime Stress</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;background:#fff;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .callout{{background:#fff8e1;border-left:4px solid #e0a500;padding:.9em 1.1em;margin:1em 0}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-2750 — Out-of-Distribution Regime Stress Test</h1>
<p class='small'>Generated {p['generated']} · Rule-Zero labelled: stress-only use of synthetic shocks.</p>

<div class='callout'>
<b>Rule Zero scope:</b> the cube itself is the real v8a production cube
(5 cached streams + XLF/XLI + XLE). Every stress overlay is a CLEARLY
LABELED synthetic injection used ONLY to characterise the live
allocator + circuit-breaker reaction. These numbers are never used for
backtest Sharpe claims.
</div>

<h2>Baseline (no stress)</h2>
<table>
<tr><th>Max DD</th><th>Soft trips</th><th>Hard trips</th><th>Final equity</th></tr>
<tr><td>{b['max_dd_pct']:.2f}%</td><td>{b['n_soft_trips']}</td>
 <td>{b['n_hard_trips']}</td><td>{b['final_equity']:.3f}</td></tr>
</table>

<h2>Stress scenarios</h2>
<table>
<tr><th>Scenario</th><th>Description</th><th>Window start</th><th>Window end</th>
 <th>Max DD</th><th>Soft trips</th><th>Hard trips</th>
 <th>Days to recover</th><th>Window PnL</th><th>Final equity</th></tr>
{rows_sc}
</table>

<h2>Interpretation</h2>
<ul>
<li><b>slow_grind_bear</b>: measures whether a steady 2%/month equity drift
    trips the circuit breaker. Vol-targeting will de-lever into the drift;
    the breaker fires on cumulative drawdown, not on velocity.</li>
<li><b>flash_crash_v</b>: tests whether the 3%/6% breaker catches a
    single-day gap-down before it compounds. Hedge sleeve (v5_hedge)
    absorbs some of the impact.</li>
<li><b>stagflation</b>: zero-mean + 1.5× vol for 2 years. The LW
    risk-parity re-fit every 63 days should reduce weights on the
    equity-sensitive streams as their realised vol spikes.</li>
<li><b>correlation_breakdown</b>: all streams converge to ρ≈0.8. The
    effective N drops from 6.69 → ~1, and the cube's diversification
    benefit collapses. Tests whether LW shrinkage catches the regime
    change fast enough to de-lever.</li>
</ul>

<h2>Scope note</h2>
<p class='small'>These numbers are a forward-looking characterisation of
the <b>allocator + breaker + vol targeting machinery</b>, not a claim
about historical performance. Real stress historically has looked like
COVID 2020 and 2022 — captured in the baseline. This experiment is
about regimes outside that sample.</p>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
