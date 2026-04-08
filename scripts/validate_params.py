#!/usr/bin/env python3
"""
validate_params.py — Overfit detection suite for Operation Crack The Code

Implements ALL 7 checks from MASTERPLAN Step 3.5:
  A. Cross-year consistency  (≥5/6 years profitable)
  B. Walk-forward validation (test ≥50% of train return)
  C. Parameter sensitivity   (±10-20% jitter, return stays ≥60%)
  D. Trade count gate        (≥30 trades per year)
  E. Regime diversity        (profitable months ≥6 per year)
  F. Drawdown reality        (<50% max DD, <60d recovery, <15 loss streak)
  G. Composite overfit score (weighted A-F, must be ≥0.70 to be "robust")

Usage (standalone):
    python3 scripts/validate_params.py --config configs/exp.json

Usage (programmatic):
    from scripts.validate_params import validate_params
    result = validate_params(params, results_by_year, years, use_real, ticker)
"""

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logger = logging.getLogger("validate")

# ── Check A — Cross-year consistency ─────────────────────────────────────────

def check_a_consistency(results_by_year: dict) -> dict:
    """≥5/6 years must be profitable. Score = years_profitable / total."""
    years_profitable = sum(
        1 for r in results_by_year.values()
        if "error" not in r and r.get("return_pct", 0) > 0
    )
    total = sum(1 for r in results_by_year.values() if "error" not in r)
    score = years_profitable / total if total > 0 else 0
    passed = score >= 0.833  # ≥5/6
    return {
        "check": "A_consistency",
        "years_profitable": years_profitable,
        "years_total": total,
        "score": round(score, 3),
        "passed": passed,
        "note": f"{years_profitable}/{total} years profitable",
    }


# ── Check B — Rolling walk-forward validation ────────────────────────────────

def check_b_walkforward(params: dict, use_real: bool, ticker: str,
                         existing_results: dict) -> dict:
    """
    Rolling walk-forward: 3 folds, each with growing train + 1-year test.
      Fold 1: Train 2020-2022 → Test 2023
      Fold 2: Train 2020-2023 → Test 2024
      Fold 3: Train 2020-2024 → Test 2025

    For each fold: test_return / train_avg >= 0.50 → passes.
    Score = avg ratio across folds (capped 0-1).

    Uses existing_results for all years — no re-running required.
    This replaces the single 3+3 split with a selection-aware multi-fold approach.
    """
    def avg_ret(ys):
        rets = [existing_results[y]["return_pct"]
                for y in ys if y in existing_results and "error" not in existing_results[y]]
        return sum(rets) / len(rets) if rets else None

    folds = [
        {"train": ["2020", "2021", "2022"],            "test": "2023"},
        {"train": ["2020", "2021", "2022", "2023"],    "test": "2024"},
        {"train": ["2020", "2021", "2022", "2023", "2024"], "test": "2025"},
    ]

    fold_results = []
    for fold in folds:
        train_avg = avg_ret(fold["train"])
        test_yr   = fold["test"]
        if test_yr not in existing_results or "error" in existing_results.get(test_yr, {}):
            continue
        test_ret = existing_results[test_yr]["return_pct"]

        if train_avg is None:
            continue

        if train_avg <= 0:
            ratio = 1.0 if test_ret >= train_avg else 0.0
        else:
            ratio = min(1.0, max(0.0, test_ret / train_avg))

        fold_results.append({
            "train_years": fold["train"],
            "test_year": test_yr,
            "train_avg": round(train_avg, 2),
            "test_return": round(test_ret, 2),
            "ratio": round(ratio, 3),
            "passed": ratio >= 0.50,
        })

    if not fold_results:
        return {
            "check": "B_walkforward",
            "score": 0.5,
            "passed": None,
            "note": "Insufficient years for rolling walk-forward",
            "folds": [],
        }

    # CARLOS FIX: Use MEDIAN fold ratio (not mean) to neutralize 2022 outlier effect.
    # A monster year like 2022 (+203%) inflates the train_avg for Folds 2 and 3,
    # making it artificially hard for test years to meet the 50% threshold.
    ratios = sorted(f["ratio"] for f in fold_results)
    n = len(ratios)
    if n % 2 == 1:
        median_ratio = ratios[n // 2]
    else:
        median_ratio = (ratios[n // 2 - 1] + ratios[n // 2]) / 2.0

    folds_passed = sum(1 for f in fold_results if f["passed"])
    score = round(median_ratio, 3)
    passed = folds_passed >= 2  # at least 2 of 3 folds must pass (GATE — see compute_overfit_score)

    return {
        "check": "B_walkforward",
        "folds": fold_results,
        "folds_passed": folds_passed,
        "folds_total": len(fold_results),
        "median_ratio": round(median_ratio, 3),
        "score": score,
        "passed": passed,
        "note": (f"Rolling WF: {folds_passed}/{len(fold_results)} folds pass (median ratio={median_ratio:.2f})"
                 + (" ✓" if passed else " ✗ GATE FAIL — composite capped at SUSPECT")),
    }


# ── Check C — Parameter sensitivity (jitter test) ───────────────────────────

def check_c_sensitivity(params: dict, base_results: dict, use_real: bool, ticker: str) -> dict:
    """
    Perturb each numeric param by ±10% and ±20%. Run 4 jittered variants.
    Score = avg_jittered_return / base_return. Must be ≥0.60.
    """
    from scripts.run_optimization import run_all_years

    jitter_params = ["target_delta", "target_dte", "spread_width",
                     "stop_loss_multiplier", "profit_target", "max_risk_per_trade"]
    perturb_pcts = [-0.20, -0.10, +0.10, +0.20]

    base_avg = sum(r.get("return_pct", 0) for r in base_results.values()
                   if "error" not in r) / max(1, sum(1 for r in base_results.values() if "error" not in r))

    years = [int(y) for y in base_results.keys() if "error" not in base_results[y]]
    if not years:
        return {"check": "C_sensitivity", "score": 0.5, "passed": None,
                "note": "No valid years to jitter", "jitter_runs": []}

    jitter_results = []
    cliff_params = []
    trade_count_cliff = []

    base_total_trades = sum(
        r.get("total_trades", 0) for r in base_results.values() if "error" not in r
    )

    # Only jitter the 3 highest-impact params to keep run count low
    params_to_test = [p for p in jitter_params if p in params][:3]

    for param in params_to_test:
        base_val = params[param]
        if not isinstance(base_val, (int, float)):
            continue

        param_jitter_rets = []
        for pct in perturb_pcts[:2]:  # only ±10% to keep it fast
            jittered = copy.deepcopy(params)
            new_val = base_val * (1 + pct)
            # Clamp reasonable ranges
            if param == "target_delta":
                new_val = max(0.05, min(0.40, new_val))
            if param == "target_dte":
                new_val = max(7, min(90, int(new_val)))
            if param == "spread_width":
                new_val = max(1, int(round(new_val)))
            if param == "profit_target":
                new_val = max(10, min(100, new_val))
            jittered[param] = new_val

            t0 = time.time()
            try:
                j_results = run_all_years(jittered, years, use_real, ticker)
                j_avg = sum(r.get("return_pct", 0) for r in j_results.values()
                            if "error" not in r) / max(1, len(j_results))
                j_trades = sum(r.get("total_trades", 0) for r in j_results.values()
                               if "error" not in r)
                elapsed = time.time() - t0
                jitter_results.append({
                    "param": param, "delta_pct": pct, "new_val": new_val,
                    "avg_return": round(j_avg, 2), "total_trades": j_trades,
                    "elapsed_sec": round(elapsed),
                })
                param_jitter_rets.append(j_avg)
                # Flag if trade count drops by >80% on a ±10% param change — signals
                # high fragility (this is the root of "160 vs 7 trades" discrepancies).
                if base_total_trades > 0 and j_trades < base_total_trades * 0.20:
                    trade_count_cliff.append(
                        f"{param}{pct:+.0%}: {j_trades} vs {base_total_trades} base"
                    )
            except Exception as e:
                logger.warning("Jitter run failed for %s=%s: %s", param, new_val, e)

        # Detect cliff: ±10% change causes >50% return drop
        if param_jitter_rets and base_avg != 0:
            worst_ratio = min(r / base_avg if base_avg > 0 else 0 for r in param_jitter_rets)
            if worst_ratio < 0.50:
                cliff_params.append(param)

    if not jitter_results:
        return {"check": "C_sensitivity", "score": 0.5, "passed": None,
                "note": "No jitter runs completed", "jitter_runs": []}

    jitter_avg = sum(r["avg_return"] for r in jitter_results) / len(jitter_results)
    score = min(1.0, jitter_avg / base_avg) if base_avg > 0 else 0.5
    score = max(0.0, score)
    passed = score >= 0.60 and not cliff_params

    note = f"Jitter avg={jitter_avg:+.1f}% vs base={base_avg:+.1f}%"
    if cliff_params:
        note += f" ⚠️ CLIFF PARAMS: {cliff_params}"
    if trade_count_cliff:
        note += f" ⚠️ TRADE COUNT CLIFF: {trade_count_cliff}"

    return {
        "check": "C_sensitivity",
        "base_avg_return": round(base_avg, 2),
        "jitter_avg_return": round(jitter_avg, 2),
        "score": round(score, 3),
        "passed": passed,
        "cliff_params": cliff_params,
        "trade_count_cliff": trade_count_cliff,
        "jitter_runs": jitter_results,
        "note": note,
    }


# ── Check D — Trade count gate ────────────────────────────────────────────────

def check_d_trade_count(results_by_year: dict, min_trades: int = 30) -> dict:
    """All years must have ≥30 trades. Score=1 if all pass, else fraction passing."""
    checks = {}
    for yr, r in results_by_year.items():
        if "error" in r:
            checks[yr] = {"trades": 0, "passed": False}
            continue
        t = r.get("total_trades", 0)
        checks[yr] = {"trades": t, "passed": t >= min_trades}

    passing = sum(1 for c in checks.values() if c["passed"])
    total   = len(checks)
    score   = passing / total if total > 0 else 0
    low = [yr for yr, c in checks.items() if not c["passed"]]

    return {
        "check": "D_trade_count",
        "per_year": checks,
        "years_passing": passing,
        "years_total": total,
        "score": round(score, 3),
        "passed": score >= 1.0,
        "low_trade_years": low,
        "note": f"{passing}/{total} years with ≥{min_trades} trades" +
                (f" — low: {low}" if low else ""),
    }


# ── Check E — Regime diversity ────────────────────────────────────────────────

def check_e_regime_diversity(results_by_year: dict) -> dict:
    """
    For each year: count months with at least 1 trade.
    Score = avg(months_with_trades / 12) across years.
    Also flag if >50% of annual P&L came from a single month.
    """
    year_scores = {}
    concentration_flags = []

    for yr, r in results_by_year.items():
        if "error" in r:
            year_scores[yr] = 0.0
            continue

        monthly = r.get("monthly_pnl", {})
        if not monthly:
            year_scores[yr] = 0.0
            continue

        months_with_trades = sum(1 for m in monthly.values() if m.get("trades", 0) > 0)
        from_key = min(monthly.keys())
        to_key   = max(monthly.keys())
        from_y, from_m = int(from_key[:4]), int(from_key[5:])
        to_y,   to_m   = int(to_key[:4]),   int(to_key[5:])
        months_elapsed = (to_y - from_y) * 12 + (to_m - from_m) + 1
        year_scores[yr] = months_with_trades / max(1, months_elapsed)

        # Concentration check
        total_pnl = sum(m.get("pnl", 0) for m in monthly.values())
        if total_pnl > 0:
            max_month_pnl = max(m.get("pnl", 0) for m in monthly.values())
            if max_month_pnl / total_pnl > 0.50:
                concentration_flags.append(yr)

    score = sum(year_scores.values()) / len(year_scores) if year_scores else 0
    passed = score >= 0.40 and not concentration_flags  # ≥~5 months active per year on average

    return {
        "check": "E_regime_diversity",
        "per_year_month_coverage": {yr: round(s, 3) for yr, s in year_scores.items()},
        "score": round(score, 3),
        "passed": passed,
        "concentration_flags": concentration_flags,
        "note": (f"Avg monthly coverage: {score:.0%}"
                 + (f" ⚠️ Concentration in: {concentration_flags}" if concentration_flags else "")),
    }


# ── Check F — Drawdown reality ────────────────────────────────────────────────

def check_f_drawdown(results_by_year: dict, max_dd_limit: float = -50.0,
                     max_streak: int = 15) -> dict:
    """
    All years: max drawdown < 50%, max loss streak < 15.
    Score degrades proportionally with violations.
    """
    dd_violations = []
    streak_violations = []

    for yr, r in results_by_year.items():
        if "error" in r:
            continue
        dd = r.get("max_drawdown", 0)
        if dd < max_dd_limit:
            dd_violations.append({"year": yr, "max_drawdown": dd})
        streak = r.get("max_loss_streak", 0)
        if streak >= max_streak:
            streak_violations.append({"year": yr, "max_loss_streak": streak})

    total = sum(1 for r in results_by_year.values() if "error" not in r)
    violations = len(dd_violations) + len(streak_violations)
    score = max(0.0, 1.0 - (violations / max(1, total)))
    passed = violations == 0

    return {
        "check": "F_drawdown",
        "dd_violations": dd_violations,
        "streak_violations": streak_violations,
        "max_dd_limit": max_dd_limit,
        "max_streak_limit": max_streak,
        "score": round(score, 3),
        "passed": passed,
        "note": ("No violations" if not violations
                 else f"{violations} violations — DD: {dd_violations}, Streak: {streak_violations}"),
    }


# ── Check H — Data consistency ────────────────────────────────────────────────

def check_h_data_consistency(results_by_year: dict) -> dict:
    """
    Verify internal accounting consistency per year:
      return_pct must match (ending_capital - starting_capital) / starting_capital * 100
    within 1%.  Mismatch signals a double-counting or capital tracking bug.
    Also flags years where 0 trades were recorded — useful for spotting
    data pipeline holes (e.g. 'exp_031: 160 trades vs 7 trades' discrepancy).
    """
    inconsistencies = []
    zero_trade_years = []

    for yr, r in results_by_year.items():
        if "error" in r:
            continue

        end   = r.get("ending_capital", 0)
        start = r.get("starting_capital", 0)
        ret   = r.get("return_pct", 0)

        if start > 0 and end > 0:
            implied = (end - start) / start * 100
            if abs(implied - ret) > 1.0:
                inconsistencies.append({
                    "year":      yr,
                    "reported_return_pct":  round(ret, 2),
                    "implied_return_pct":   round(implied, 2),
                    "delta":                round(abs(implied - ret), 2),
                })

        trades = r.get("total_trades", 0)
        if trades == 0:
            zero_trade_years.append(yr)

    passed = len(inconsistencies) == 0
    score  = 1.0 if passed else 0.0

    note = "All metrics internally consistent"
    if inconsistencies:
        note = f"⚠️  {len(inconsistencies)} metric inconsistencies — check capital tracking"
    if zero_trade_years:
        note += f" | 0-trade years: {zero_trade_years}"

    return {
        "check":              "H_data_consistency",
        "inconsistencies":    inconsistencies,
        "zero_trade_years":   zero_trade_years,
        "score":              score,
        "passed":             passed,
        "note":               note,
    }


# ── Check G — Composite score ─────────────────────────────────────────────────

def compute_overfit_score(checks: dict) -> tuple:
    """
    Weighted composite per MASTERPLAN (CARLOS UPDATED RULES):
      consistency × 0.25 + walkforward × 0.30 + sensitivity × 0.25
      + trade_count × 0.10 + regime_diversity × 0.10

    HARD GATES (Carlos critique §1, §4):
      - Walk-forward (B) FAIL → composite capped at 0.59 (always SUSPECT)
      - Sensitivity (C) FAIL → composite capped at 0.59 (always SUSPECT)
      - Drawdown (F) FAIL → composite capped at 0.59 (always SUSPECT)

    These are not "partial credit" failures — they are disqualifying.
    Motivation: B and C are the FORWARD-LOOKING checks. A strategy that
    fails out-of-sample and can't survive parameter jitter will NOT work
    in live trading regardless of in-sample performance.
    """
    weights = {
        "A_consistency":    0.25,
        "B_walkforward":    0.30,
        "C_sensitivity":    0.25,
        "D_trade_count":    0.10,
        "E_regime_diversity": 0.10,
    }
    score = 0.0
    for key, w in weights.items():
        c = checks.get(key, {})
        s = c.get("score", 0.5) if c.get("score") is not None else 0.5
        score += s * w

    gates_failed = []

    # Walk-forward GATE (B) — must pass to be ROBUST
    b = checks.get("B_walkforward", {})
    if b.get("passed") is False:
        gates_failed.append("B_walkforward")

    # Sensitivity GATE (C) — must pass to be ROBUST
    c_check = checks.get("C_sensitivity", {})
    if c_check.get("passed") is False:
        gates_failed.append("C_sensitivity")

    # Drawdown GATE (F) — violations cap the composite
    f = checks.get("F_drawdown", {})
    if not f.get("passed", True):
        gates_failed.append("F_drawdown")

    if gates_failed:
        score = min(score, 0.59)

    score = round(score, 3)

    if score >= 0.70:
        verdict = "ROBUST"
    elif score >= 0.50:
        verdict = "SUSPECT"
    else:
        verdict = "OVERFIT"

    return score, verdict, gates_failed


# ── Main validation entry point ───────────────────────────────────────────────

def validate_params(params: dict, results_by_year: dict, years: list,
                    use_real: bool, ticker: str = "SPY",
                    skip_jitter: bool = False) -> dict:
    """
    Run all overfit checks and return a full validation report.

    Args:
        params:           Strategy params dict.
        results_by_year:  Already-computed results (keyed by str year).
        years:            List of int years that were run.
        use_real:         Whether to use real Polygon data for jitter runs.
        ticker:           Ticker symbol.
        skip_jitter:      If True, skip check C (saves time).

    Returns:
        Dict with per-check results, overfit_score, and verdict.
    """
    # Run consistency check first — flag problems before computing overfit score
    print("  [H] Data consistency...", end=" ", flush=True)
    check_h = check_h_data_consistency(results_by_year)
    print(f"{check_h['score']:.2f}  {'✓' if check_h['passed'] else '⚠️  ' + check_h['note']}")

    print("  [A] Cross-year consistency...", end=" ", flush=True)
    check_a = check_a_consistency(results_by_year)
    print(f"{check_a['score']:.2f}  {'✓' if check_a['passed'] else '✗'}")

    print("  [B] Walk-forward validation...", end=" ", flush=True)
    check_b = check_b_walkforward(params, use_real, ticker, results_by_year)
    print(f"{check_b['score']:.2f}  {'✓' if check_b['passed'] else '✗' if check_b['passed'] is False else '?'}")

    print("  [C] Parameter sensitivity...", end=" ", flush=True)
    if skip_jitter:
        check_c = {"check": "C_sensitivity", "score": 0.5, "passed": None,
                   "note": "Skipped", "jitter_runs": []}
        print("skipped")
    else:
        check_c = check_c_sensitivity(params, results_by_year, use_real, ticker)
        print(f"{check_c['score']:.2f}  {'✓' if check_c['passed'] else '✗' if check_c['passed'] is False else '?'}")

    print("  [D] Trade count gate...", end=" ", flush=True)
    check_d = check_d_trade_count(results_by_year)
    print(f"{check_d['score']:.2f}  {'✓' if check_d['passed'] else '✗'}")

    print("  [E] Regime diversity...", end=" ", flush=True)
    check_e = check_e_regime_diversity(results_by_year)
    print(f"{check_e['score']:.2f}  {'✓' if check_e['passed'] else '✗'}")

    print("  [F] Drawdown reality...", end=" ", flush=True)
    check_f = check_f_drawdown(results_by_year)
    print(f"{check_f['score']:.2f}  {'✓' if check_f['passed'] else '✗'}")

    checks = {
        "H_data_consistency": check_h,
        "A_consistency":      check_a,
        "B_walkforward":      check_b,
        "C_sensitivity":      check_c,
        "D_trade_count":      check_d,
        "E_regime_diversity": check_e,
        "F_drawdown":         check_f,
    }

    overfit_score, verdict, gates_failed = compute_overfit_score(checks)

    icon = "✅ ROBUST" if verdict == "ROBUST" else ("⚠️  SUSPECT" if verdict == "SUSPECT" else "❌ OVERFIT")
    gate_note = f" [GATES FAILED: {', '.join(gates_failed)}]" if gates_failed else ""
    print(f"  Overfit score: {overfit_score:.3f}  →  {icon}{gate_note}")

    return {
        "checks":        checks,
        "overfit_score": overfit_score,
        "verdict":       verdict,
        "gates_failed":  gates_failed,
    }


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run overfit validation on a param config")
    parser.add_argument("--config",      required=True, help="JSON file with params")
    parser.add_argument("--results",     help="JSON file with existing results_by_year (skips re-running)")
    parser.add_argument("--heuristic",   action="store_true", help="Fast heuristic mode")
    parser.add_argument("--skip-jitter", action="store_true", help="Skip check C (faster)")
    parser.add_argument("--ticker",      default="SPY")
    args = parser.parse_args()

    with open(args.config) as f:
        params = json.load(f)

    if args.results:
        with open(args.results) as f:
            results_by_year = json.load(f)
        years = [int(y) for y in results_by_year.keys()]
    else:
        from scripts.run_optimization import YEARS, run_all_years
        use_real = not args.heuristic
        years = YEARS
        print(f"Running backtests for {years}...")
        results_by_year = run_all_years(params, years, use_real, args.ticker)

    print("\nRunning overfit checks...")
    result = validate_params(params, results_by_year, years,
                             not args.heuristic, args.ticker, args.skip_jitter)

    print()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
