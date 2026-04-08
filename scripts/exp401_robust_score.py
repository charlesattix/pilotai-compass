#!/usr/bin/env python3
"""
exp401_robust_score.py — ROBUST overfit scoring for EXP-401 (The Blend)

Runs the full overfit detection suite adapted for the portfolio blend:
  A. Cross-year consistency (≥5/6 years profitable)
  B. Walk-forward validation (3-fold rolling, median ratio ≥0.50)
  C. Parameter sensitivity (jitter CS risk ±20%, SS risk ±20%, regime scales ±20%)
  D. Trade count gate (≥30 trades/year)
  E. Regime diversity (monthly coverage ≥40%, no P&L concentration)
  F. Drawdown reality (<50% max DD, <15 loss streak)
  G. Composite overfit score (weighted A-F, ≥0.70 = ROBUST)

Uses the Phase 4 best config: CS(12%) + SS(3%) with regime scales.

Output: output/exp401_robust_score.json

Usage:
    PYTHONPATH=. python3 scripts/exp401_robust_score.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.portfolio_backtester import PortfolioBacktester
from scripts.portfolio_blend import get_strategy_params
from scripts.validate_params import (
    check_a_consistency,
    check_b_walkforward,
    check_d_trade_count,
    check_e_regime_diversity,
    check_f_drawdown,
    check_h_data_consistency,
    compute_overfit_score,
)
from strategies import STRATEGY_REGISTRY

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — Phase 4 best config (same as final_validation.py)
# ═══════════════════════════════════════════════════════════════════════════════

TICKERS = ["SPY"]
STARTING_CAPITAL = 100_000
ALL_YEARS = list(range(2020, 2026))

CS_BASE_RISK = 0.12
SS_BASE_RISK = 0.03

CS_REGIME_SCALES = {
    "regime_scale_bull": 1.0,
    "regime_scale_bear": 0.3,
    "regime_scale_high_vol": 0.3,
    "regime_scale_low_vol": 0.8,
    "regime_scale_crash": 0.0,
}

SS_REGIME_SCALES = {
    "regime_scale_bull": 1.5,
    "regime_scale_bear": 1.5,
    "regime_scale_high_vol": 2.5,
    "regime_scale_low_vol": 1.0,
    "regime_scale_crash": 0.5,
}


def build_params(
    cs_risk: float = CS_BASE_RISK,
    ss_risk: float = SS_BASE_RISK,
    cs_regime: Dict = None,
    ss_regime: Dict = None,
) -> Tuple[Dict, Dict]:
    """Build CS and SS param dicts."""
    cs_params = get_strategy_params("credit_spread", risk_override=cs_risk)
    cs_params.update(cs_regime or CS_REGIME_SCALES)

    ss_params = get_strategy_params("straddle_strangle", risk_override=ss_risk)
    ss_params.update(ss_regime or SS_REGIME_SCALES)

    return cs_params, ss_params


def run_blend_year(cs_params: Dict, ss_params: Dict, year: int) -> Dict:
    """Run CS+SS blend for one year. Returns full combined results dict."""
    cs_cls = STRATEGY_REGISTRY["credit_spread"]
    ss_cls = STRATEGY_REGISTRY["straddle_strangle"]

    bt = PortfolioBacktester(
        strategies=[("credit_spread", cs_cls(dict(cs_params))),
                    ("straddle_strangle", ss_cls(dict(ss_params)))],
        tickers=TICKERS,
        start_date=datetime(year, 1, 1),
        end_date=datetime(year, 12, 31),
        starting_capital=STARTING_CAPITAL,
        max_positions=10,
        max_positions_per_strategy=5,
    )
    raw = bt.run()
    combined = raw.get("combined", raw)
    return combined


def run_all_years(
    cs_params: Dict, ss_params: Dict, years: List[int] = None,
) -> Dict[str, Dict]:
    """Run blend across years. Returns {str_year: results_dict}."""
    if years is None:
        years = ALL_YEARS
    results = {}
    for year in years:
        results[str(year)] = run_blend_year(cs_params, ss_params, year)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Check C — Parameter sensitivity (blend-adapted jitter test)
# ═══════════════════════════════════════════════════════════════════════════════

def check_c_blend_sensitivity(
    base_results: Dict[str, Dict],
    base_cs_risk: float,
    base_ss_risk: float,
    base_cs_regime: Dict,
    base_ss_regime: Dict,
) -> Dict:
    """
    Perturb key blend parameters by ±20% and measure return stability.

    Jitter targets:
      1. CS max_risk_pct (12% ± 20% → 9.6% and 14.4%)
      2. SS max_risk_pct (3% ± 20% → 2.4% and 3.6%)
      3. Each regime scale ± 20% (10 scales × 2 perturbations = 20 runs)

    Score = avg_jittered_return / base_return, capped at [0, 1].
    Pass if score ≥ 0.60 and no cliff params.
    """
    base_avg = sum(
        r.get("return_pct", 0) for r in base_results.values() if "error" not in r
    ) / max(1, sum(1 for r in base_results.values() if "error" not in r))

    years = [int(y) for y in base_results.keys() if "error" not in base_results[y]]
    jitter_results = []
    cliff_params = []

    def _run_jitter(label: str, cs_risk, ss_risk, cs_regime, ss_regime):
        cs_p, ss_p = build_params(cs_risk, ss_risk, cs_regime, ss_regime)
        results = run_all_years(cs_p, ss_p, years)
        avg_ret = sum(
            r.get("return_pct", 0) for r in results.values() if "error" not in r
        ) / max(1, len(results))
        return {"label": label, "avg_return": round(avg_ret, 2)}

    # ── Jitter CS risk ±20% ──────────────────────────────────────────────
    for pct in [-0.20, +0.20]:
        new_risk = base_cs_risk * (1 + pct)
        label = f"cs_risk={new_risk:.3f} ({pct:+.0%})"
        entry = _run_jitter(label, new_risk, base_ss_risk, base_cs_regime, base_ss_regime)
        jitter_results.append(entry)
        print(f"    {label}: {entry['avg_return']:+.1f}%")

    # ── Jitter SS risk ±20% ──────────────────────────────────────────────
    for pct in [-0.20, +0.20]:
        new_risk = base_ss_risk * (1 + pct)
        label = f"ss_risk={new_risk:.3f} ({pct:+.0%})"
        entry = _run_jitter(label, base_cs_risk, new_risk, base_cs_regime, base_ss_regime)
        jitter_results.append(entry)
        print(f"    {label}: {entry['avg_return']:+.1f}%")

    # ── Jitter each regime scale ±20% ────────────────────────────────────
    for scale_name, base_val in base_cs_regime.items():
        if base_val == 0:
            continue  # skip crash=0 (can't perturb 0 meaningfully)
        for pct in [-0.20, +0.20]:
            new_val = round(base_val * (1 + pct), 3)
            jittered_regime = dict(base_cs_regime)
            jittered_regime[scale_name] = new_val
            label = f"cs_{scale_name}={new_val} ({pct:+.0%})"
            entry = _run_jitter(label, base_cs_risk, base_ss_risk, jittered_regime, base_ss_regime)
            jitter_results.append(entry)
            print(f"    {label}: {entry['avg_return']:+.1f}%")

    for scale_name, base_val in base_ss_regime.items():
        if base_val == 0:
            continue
        for pct in [-0.20, +0.20]:
            new_val = round(base_val * (1 + pct), 3)
            jittered_regime = dict(base_ss_regime)
            jittered_regime[scale_name] = new_val
            label = f"ss_{scale_name}={new_val} ({pct:+.0%})"
            entry = _run_jitter(label, base_cs_risk, base_ss_risk, base_cs_regime, jittered_regime)
            jitter_results.append(entry)
            print(f"    {label}: {entry['avg_return']:+.1f}%")

    # ── Compute score ────────────────────────────────────────────────────
    jitter_avg = sum(r["avg_return"] for r in jitter_results) / len(jitter_results)
    score = min(1.0, max(0.0, jitter_avg / base_avg)) if base_avg > 0 else 0.5

    # Detect cliffs: any jitter causing >50% return drop
    for r in jitter_results:
        if base_avg > 0 and r["avg_return"] / base_avg < 0.50:
            cliff_params.append(r["label"])

    passed = score >= 0.60 and not cliff_params

    note = f"Jitter avg={jitter_avg:+.1f}% vs base={base_avg:+.1f}% (ratio={score:.3f})"
    if cliff_params:
        note += f" | CLIFF: {cliff_params}"

    return {
        "check": "C_sensitivity",
        "base_avg_return": round(base_avg, 2),
        "jitter_avg_return": round(jitter_avg, 2),
        "score": round(score, 3),
        "passed": passed,
        "cliff_params": cliff_params,
        "jitter_runs": jitter_results,
        "total_jitter_runs": len(jitter_results),
        "note": note,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("EXP-401 ROBUST Overfit Scoring")
    print("=" * 70)
    print(f"Config: CS={CS_BASE_RISK:.0%} + SS={SS_BASE_RISK:.0%} with regime scales")

    t_total = time.time()

    # ── Run baseline 6-year backtest ─────────────────────────────────────
    print("\n[0] Running baseline 6-year backtest...")
    t0 = time.time()
    cs_params, ss_params = build_params()
    base_results = run_all_years(cs_params, ss_params)
    print(f"  Done in {time.time()-t0:.0f}s")

    for year in ALL_YEARS:
        r = base_results[str(year)]
        print(f"    {year}: ret={r.get('return_pct', 0):+.1f}%  DD={r.get('max_drawdown', 0):.1f}%  "
              f"trades={r.get('total_trades', 0)}  WR={r.get('win_rate', 0):.1f}%")

    # ── Run all checks ───────────────────────────────────────────────────
    print("\n[1] Running overfit checks...")

    # H: Data consistency
    print("  [H] Data consistency...", end=" ", flush=True)
    check_h = check_h_data_consistency(base_results)
    print(f"{check_h['score']:.2f}  {'✓' if check_h['passed'] else '⚠ ' + check_h['note']}")

    # A: Cross-year consistency
    print("  [A] Cross-year consistency...", end=" ", flush=True)
    check_a = check_a_consistency(base_results)
    print(f"{check_a['score']:.2f}  {'✓' if check_a['passed'] else '✗'}  ({check_a['note']})")

    # B: Walk-forward validation
    print("  [B] Walk-forward validation...", end=" ", flush=True)
    check_b = check_b_walkforward({}, False, "SPY", base_results)
    print(f"{check_b['score']:.2f}  {'✓' if check_b['passed'] else '✗'}  ({check_b['note']})")

    # C: Parameter sensitivity (blend-adapted)
    print("  [C] Parameter sensitivity (blend jitter)...")
    t0 = time.time()
    check_c = check_c_blend_sensitivity(
        base_results, CS_BASE_RISK, SS_BASE_RISK, CS_REGIME_SCALES, SS_REGIME_SCALES,
    )
    print(f"    → {check_c['score']:.2f}  {'✓' if check_c['passed'] else '✗'}  "
          f"({check_c['note']})  [{time.time()-t0:.0f}s]")

    # D: Trade count gate
    print("  [D] Trade count gate...", end=" ", flush=True)
    check_d = check_d_trade_count(base_results)
    print(f"{check_d['score']:.2f}  {'✓' if check_d['passed'] else '✗'}  ({check_d['note']})")

    # E: Regime diversity
    print("  [E] Regime diversity...", end=" ", flush=True)
    check_e = check_e_regime_diversity(base_results)
    print(f"{check_e['score']:.2f}  {'✓' if check_e['passed'] else '✗'}  ({check_e['note']})")

    # F: Drawdown reality
    print("  [F] Drawdown reality...", end=" ", flush=True)
    check_f = check_f_drawdown(base_results)
    print(f"{check_f['score']:.2f}  {'✓' if check_f['passed'] else '✗'}  ({check_f['note']})")

    # ── Compute composite score ──────────────────────────────────────────
    checks = {
        "H_data_consistency": check_h,
        "A_consistency": check_a,
        "B_walkforward": check_b,
        "C_sensitivity": check_c,
        "D_trade_count": check_d,
        "E_regime_diversity": check_e,
        "F_drawdown": check_f,
    }

    overfit_score, verdict, gates_failed = compute_overfit_score(checks)

    icon = {"ROBUST": "ROBUST", "SUSPECT": "SUSPECT", "OVERFIT": "OVERFIT"}[verdict]
    gate_note = f" [GATES FAILED: {', '.join(gates_failed)}]" if gates_failed else ""

    print(f"\n  {'=' * 50}")
    print(f"  COMPOSITE OVERFIT SCORE: {overfit_score:.3f}  →  {icon}{gate_note}")
    print(f"  {'=' * 50}")
    print("\n  Component scores:")
    print(f"    A (consistency):  {check_a['score']:.3f} × 0.25 = {check_a['score']*0.25:.3f}")
    print(f"    B (walkforward):  {check_b['score']:.3f} × 0.30 = {check_b['score']*0.30:.3f}")
    print(f"    C (sensitivity):  {check_c['score']:.3f} × 0.25 = {check_c['score']*0.25:.3f}")
    print(f"    D (trade_count):  {check_d['score']:.3f} × 0.10 = {check_d['score']*0.10:.3f}")
    print(f"    E (diversity):    {check_e['score']:.3f} × 0.10 = {check_e['score']*0.10:.3f}")

    print(f"\n  Total time: {time.time()-t_total:.0f}s")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "experiment": "EXP-401",
        "description": "Phase 4 regime-optimized blend: CS(12%) + SS(3%)",
        "generated": datetime.now().isoformat(),
        "config": {
            "tickers": TICKERS,
            "starting_capital": STARTING_CAPITAL,
            "cs_base_risk": CS_BASE_RISK,
            "ss_base_risk": SS_BASE_RISK,
            "cs_regime_scales": CS_REGIME_SCALES,
            "ss_regime_scales": SS_REGIME_SCALES,
        },
        "baseline_yearly": {
            yr: {
                "return_pct": r.get("return_pct", 0),
                "max_drawdown": r.get("max_drawdown", 0),
                "total_trades": r.get("total_trades", 0),
                "win_rate": r.get("win_rate", 0),
                "sharpe_ratio": r.get("sharpe_ratio", 0),
            }
            for yr, r in base_results.items()
        },
        "checks": {k: v for k, v in checks.items()},
        "overfit_score": overfit_score,
        "verdict": verdict,
        "gates_failed": gates_failed,
    }

    out_path = ROOT / "output" / "exp401_robust_score.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
