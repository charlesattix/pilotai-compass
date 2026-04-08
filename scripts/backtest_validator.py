#!/usr/bin/env python3
"""
Backtest Integrity Validator
============================
Post-backtest validation script. Reads leaderboard.json or a single run result
and checks for commission errors, margin violations, leverage abuse, unrealistic
returns, and volume infeasibility.

Usage:
    # Validate entire leaderboard
    python scripts/backtest_validator.py

    # Validate a single leaderboard entry by run_id
    python scripts/backtest_validator.py --run-id <run_id>

    # Validate a raw result JSON file (from a single run)
    python scripts/backtest_validator.py --result-file path/to/result.json

Exit codes:
    0 — all entries PASS or WARN
    1 — at least one entry FAILS (blocks pipeline)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from shared.realistic_benchmarks import (
    COMMISSION_PER_CONTRACT_DEFAULT,
    SHARPE_RATIO_FANTASY_THRESHOLD,
    SHARPE_RATIO_REALISTIC_MAX,
    WIN_RATE_FANTASY_THRESHOLD_PCT,
    WIN_RATE_REALISTIC_MAX_PCT,
    compute_leverage_ratio,
    grade_annual_returns,
    is_leverage_realistic,
    is_volume_feasible,
)

LEADERBOARD_PATH = ROOT / "output" / "leaderboard.json"


# ---------------------------------------------------------------------------
# Individual check functions — return (grade, message)
# ---------------------------------------------------------------------------

def check_mode(result_year: dict) -> tuple:
    """Heuristic mode gives 100% WR and cannot reflect real fills."""
    mode = result_year.get("mode", "unknown")
    if mode == "heuristic":
        return "WARN", "Mode=heuristic: 100% WR is baked in by design. Results are NOT real-data verified."
    return "PASS", f"Mode={mode}: real data."


def check_win_rate(result_year: dict) -> tuple:
    wr = result_year.get("win_rate", 0)
    mode = result_year.get("mode", "unknown")
    if wr >= WIN_RATE_FANTASY_THRESHOLD_PCT and mode != "heuristic":
        return "FAIL", f"Win rate {wr:.1f}% in real-data mode is implausible (≥{WIN_RATE_FANTASY_THRESHOLD_PCT}%). Likely a data error."
    if wr >= WIN_RATE_REALISTIC_MAX_PCT:
        return "WARN", f"Win rate {wr:.1f}% is very high (≥{WIN_RATE_REALISTIC_MAX_PCT}%). Review carefully."
    return "PASS", f"Win rate {wr:.1f}% is within expected range."


def check_return_bound(return_pct: float, year: str) -> tuple:
    """Flag any single year > 200% for mandatory human review."""
    if return_pct > 500:
        return "FAIL", f"Year {year}: return={return_pct:.1f}% exceeds 500%. Almost certainly a leverage/margin bug."
    if return_pct > 200:
        return "WARN", (
            f"Year {year}: return={return_pct:.1f}% exceeds 200%. "
            f"Verify margin model, position sizing, and commission scaling."
        )
    return "PASS", f"Year {year}: return={return_pct:.1f}% is within expected range."


def check_commission_math(
    result_year: dict,
    params: dict,
    commission_per_contract: float = COMMISSION_PER_CONTRACT_DEFAULT,
) -> tuple:
    """Estimate expected commissions and check if they're plausible.

    Expected commission = trades × avg_contracts × comm_per_contract × legs × 2 (round trip)

    We infer avg_contracts from max_risk_per_trade / risk_per_spread_estimate.
    If we can't compute exactly, we do a loose sanity check.
    """
    total_trades = result_year.get("total_trades", 0)
    if total_trades == 0:
        return "PASS", "No trades — commission check skipped."

    # Estimate average contracts per trade from params
    max_contracts = params.get("max_contracts", 5)
    params.get("spread_width", 5)
    params.get("max_risk_per_trade", 2.0)
    starting_capital = result_year.get("starting_capital", 100_000)
    iron_condor_enabled = params.get("iron_condor_enabled", False)
    legs = 4 if iron_condor_enabled else 2

    # Conservative estimate: assume avg contracts = max_contracts
    # (real avg is lower, but this gives us the worst-case expected commission drain)
    expected_commission_min = total_trades * 1 * commission_per_contract * legs * 2        # 1 contract
    expected_commission_max = total_trades * max_contracts * commission_per_contract * legs * 2  # max_contracts

    # Infer actual commission paid from capital math
    # total_pnl = ending_capital - starting_capital (gross, before commissions)
    # But the backtester nets commissions into trades. We can only check bounds.
    ending_capital = result_year.get("ending_capital", starting_capital)
    ending_capital - starting_capital

    # Commission check: for max_contracts > 1, expected commissions should be significant
    if max_contracts > 10:
        # At 100 contracts, 200 trades, round-trip = 200 × 100 × $0.65 × 4 legs × 2 = $104,000
        # This should visibly reduce returns compared to 1-contract baseline
        commission_at_max = total_trades * max_contracts * commission_per_contract * legs * 2
        commission_pct_of_capital = (commission_at_max / starting_capital) * 100
        if commission_pct_of_capital > 50:
            return "WARN", (
                f"Expected commission at max_contracts={max_contracts}: "
                f"${commission_at_max:,.0f} ({commission_pct_of_capital:.1f}% of starting capital). "
                f"Verify commissions are scaled by contract count in the backtester."
            )

    return "PASS", (
        f"Commission range: ${expected_commission_min:.0f}–${expected_commission_max:.0f} "
        f"({total_trades} trades, 1–{max_contracts} contracts, {legs} legs round-trip)."
    )


def check_margin_and_leverage(
    params: dict,
    starting_capital: float,
) -> tuple:
    """Check theoretical worst-case leverage vs starting capital.

    Leverage = (max_positions × max_contracts × spread_width × 100) / starting_capital

    Real brokers typically require full spread_width × 100 × contracts as maintenance margin
    for defined-risk spreads. If leverage > 3x, the backtester is operating with phantom capital.
    """
    max_positions = params.get("max_positions", 50)
    max_contracts = params.get("max_contracts", 5)
    spread_width = params.get("spread_width", 5)

    leverage = compute_leverage_ratio(max_positions, max_contracts, spread_width, starting_capital)
    ok, label = is_leverage_realistic(leverage)

    if label == "FAIL":
        return "FAIL", (
            f"Leverage={leverage:.1f}x: max_positions={max_positions} × max_contracts={max_contracts} "
            f"× spread_width=${spread_width} × 100 = ${max_positions * max_contracts * spread_width * 100:,.0f} "
            f"required margin vs ${starting_capital:,.0f} capital. "
            f"No real broker supports this. Results are not achievable."
        )
    if label == "WARN":
        return "WARN", (
            f"Leverage={leverage:.1f}x: acceptable but aggressive. "
            f"Max margin required: ${max_positions * max_contracts * spread_width * 100:,.0f}."
        )
    return "PASS", f"Leverage={leverage:.1f}x: within realistic bounds."


def check_volume_feasibility(params: dict) -> tuple:
    """Check if max_contracts is feasible given typical SPY OTM option daily volume."""
    max_contracts = params.get("max_contracts", 5)
    ok, label, msg = is_volume_feasible(max_contracts)
    if not ok:
        return "FAIL", msg
    if label == "WARN":
        return "WARN", msg
    return "PASS", msg


def check_sharpe(result_year: dict, year: str) -> tuple:
    sharpe = result_year.get("sharpe_ratio", 0)
    if sharpe > SHARPE_RATIO_FANTASY_THRESHOLD:
        return "FAIL", (
            f"Year {year}: Sharpe={sharpe:.2f} exceeds {SHARPE_RATIO_FANTASY_THRESHOLD}. "
            f"This is not achievable by any real options strategy. Commission or margin bug likely."
        )
    if sharpe > SHARPE_RATIO_REALISTIC_MAX:
        return "WARN", f"Year {year}: Sharpe={sharpe:.2f} is high (>{SHARPE_RATIO_REALISTIC_MAX}). Review carefully."
    return "PASS", f"Year {year}: Sharpe={sharpe:.2f}."


def check_position_count_vs_config(result_year: dict, params: dict) -> tuple:
    """Verify the reported max simultaneous positions is consistent with config."""
    # We don't have peak simultaneous positions in the result dict, so check trades/days
    total_trades = result_year.get("total_trades", 0)
    max_positions = params.get("max_positions", 50)
    target_dte = params.get("min_dte", params.get("target_dte", 35))

    # Rough check: if total_trades > (252 / DTE) × max_positions × 3, something is off
    # 252 trading days, each position lasts ~DTE days, so ~252/DTE "slots" per year
    # max simultaneous positions × slots = theoretical max trades
    theoretical_max = int((252 / max(target_dte, 5)) * max_positions * 3)
    if total_trades > theoretical_max:
        return "WARN", (
            f"total_trades={total_trades} seems high vs DTE={target_dte}, "
            f"max_positions={max_positions} (theoretical max ≈{theoretical_max})."
        )
    return "PASS", f"Trade count {total_trades} is consistent with DTE={target_dte}, max_positions={max_positions}."


# ---------------------------------------------------------------------------
# Entry-level validator
# ---------------------------------------------------------------------------

_GRADE_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


def _worst(grades: List[str]) -> str:
    return max(grades, key=lambda g: _GRADE_ORDER.get(g, 0))


def validate_entry(entry: dict) -> dict:
    """Validate a single leaderboard entry. Returns a validation report dict."""
    params = entry.get("params", {})
    results = entry.get("results", {})
    run_id = entry.get("run_id", "unknown")
    mode = entry.get("mode", "heuristic")

    findings = []
    year_grades = []

    # Pull config values (fall back to run_optimization defaults)
    commission_per_contract = 0.65
    starting_capital = 100_000  # default; overridden per year below

    # ── Per-year checks ──────────────────────────────────────────────────────
    returns_by_year = {}
    for year_str, yr in results.items():
        if not isinstance(yr, dict) or "error" in yr:
            continue

        sc = yr.get("starting_capital", starting_capital)
        ret = yr.get("return_pct", 0)
        returns_by_year[year_str] = ret

        for check_fn, kwargs in [
            (check_mode, {"result_year": yr}),
            (check_win_rate, {"result_year": yr}),
            (check_return_bound, {"return_pct": ret, "year": year_str}),
            (check_commission_math, {"result_year": yr, "params": params,
                                      "commission_per_contract": commission_per_contract}),
            (check_margin_and_leverage, {"params": params, "starting_capital": sc}),
            (check_sharpe, {"result_year": yr, "year": year_str}),
            (check_position_count_vs_config, {"result_year": yr, "params": params}),
        ]:
            grade, msg = check_fn(**kwargs)
            findings.append({"year": year_str, "check": check_fn.__name__, "grade": grade, "message": msg})
            year_grades.append(grade)

    # ── Volume feasibility (param-level, not per-year) ────────────────────
    grade, msg = check_volume_feasibility(params)
    findings.append({"year": "ALL", "check": "check_volume_feasibility", "grade": grade, "message": msg})
    year_grades.append(grade)

    # ── Return benchmark grade ────────────────────────────────────────────
    if returns_by_year:
        avg_ret = sum(returns_by_year.values()) / len(returns_by_year)
        has_ic = params.get("iron_condor_enabled", False)
        bench = grade_annual_returns(avg_ret, has_iron_condors=has_ic)
        findings.append({
            "year": "ALL",
            "check": "benchmark_grade",
            "grade": bench.grade if bench.grade == "REALISTIC" else ("WARN" if bench.grade == "OPTIMISTIC" else "FAIL"),
            "message": bench.message,
        })
        year_grades.append(bench.grade if bench.grade == "REALISTIC" else ("WARN" if bench.grade == "OPTIMISTIC" else "FAIL"))

    overall = _worst(year_grades)

    return {
        "run_id": run_id,
        "overall_grade": overall,
        "findings": findings,
        "params_summary": {
            "max_contracts": params.get("max_contracts"),
            "max_positions": params.get("max_positions", 50),
            "spread_width": params.get("spread_width"),
            "mode": mode,
        },
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_GRADE_COLORS = {
    "PASS": "\033[32m",   # green
    "WARN": "\033[33m",   # yellow
    "FAIL": "\033[31m",   # red
    "RESET": "\033[0m",
}


def _color(grade: str, text: str) -> str:
    """Colorize text based on grade (only when stdout is a tty)."""
    if not sys.stdout.isatty():
        return text
    return f"{_GRADE_COLORS.get(grade, '')}{text}{_GRADE_COLORS['RESET']}"


def print_report(report: dict, verbose: bool = False):
    run_id = report["run_id"]
    overall = report["overall_grade"]
    print(f"\n{'═' * 72}")
    print(f"  Run: {run_id}")
    print(f"  Overall: {_color(overall, overall)}")
    params = report.get("params_summary", {})
    print(f"  Params: max_contracts={params.get('max_contracts')}  "
          f"max_positions={params.get('max_positions')}  "
          f"spread_width=${params.get('spread_width')}  "
          f"mode={params.get('mode')}")
    print(f"{'─' * 72}")

    if verbose or overall in ("WARN", "FAIL"):
        for f in report["findings"]:
            grade = f["grade"]
            if not verbose and grade == "PASS":
                continue
            prefix = f"  [{_color(grade, grade):>4}]"
            print(f"{prefix}  {f['year']:>4}  {f['check']}")
            print(f"          {f['message']}")
    elif overall == "PASS":
        pass_count = sum(1 for f in report["findings"] if f["grade"] == "PASS")
        print(f"  All {pass_count} checks PASSED.")

    print(f"{'═' * 72}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate backtest results for commission, margin, and realism."
    )
    parser.add_argument("--run-id", help="Validate a single leaderboard entry by run_id.")
    parser.add_argument("--result-file", help="Validate a raw result JSON file (dict or list of dicts).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all checks, including PASSes.")
    parser.add_argument("--leaderboard", default=str(LEADERBOARD_PATH),
                        help="Path to leaderboard.json (default: output/leaderboard.json)")
    args = parser.parse_args()

    entries = []

    if args.result_file:
        with open(args.result_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            entries = data
        else:
            entries = [data]
    else:
        lb_path = Path(args.leaderboard)
        if not lb_path.exists():
            print(f"Leaderboard not found: {lb_path}", file=sys.stderr)
            sys.exit(1)
        with open(lb_path) as f:
            entries = json.load(f)

    if args.run_id:
        entries = [e for e in entries if e.get("run_id") == args.run_id]
        if not entries:
            print(f"No entry found with run_id={args.run_id}", file=sys.stderr)
            sys.exit(1)

    if not entries:
        print("No entries to validate.", file=sys.stderr)
        sys.exit(0)

    any_fail = False
    for entry in entries:
        report = validate_entry(entry)
        print_report(report, verbose=args.verbose)
        if report["overall_grade"] == "FAIL":
            any_fail = True

    if any_fail:
        print("\nVALIDATION FAILED — one or more entries have FAIL-grade issues.", file=sys.stderr)
        sys.exit(1)

    print("\nValidation complete — no FAIL-grade issues found.")
    sys.exit(0)


if __name__ == "__main__":
    main()
