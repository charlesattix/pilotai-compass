#!/usr/bin/env python3
"""
run_sweep.py — Batch parameter sweep for real-data optimization.

DATA GUARANTEE: All runs use IronVault (shared/iron_vault.py) exclusively.
No heuristic/synthetic pricing. run_year() hard-fails if options_cache.db is missing.

Usage:
    python3 scripts/run_sweep.py --sweep configs/real_data_sweep.json --phase phase1_dte_risk
    python3 scripts/run_sweep.py --sweep configs/real_data_sweep.json --phase phase1_dte_risk --dry-run
    python3 scripts/run_sweep.py --sweep configs/real_data_sweep.json --phase phase1_dte_risk --years 2022,2023,2024
    python3 scripts/run_sweep.py --sweep configs/real_data_sweep.json --phase phase1_dte_risk --max-runs 10
    python3 scripts/run_sweep.py --sweep configs/real_data_sweep.json --list-phases

Resuming: Already-completed combos are detected by checking the leaderboard for matching
params+years. Re-run interrupted sweeps safely — completed runs are never duplicated.
"""

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from scripts.run_optimization import (
    YEARS,
    _build_config,
    append_to_leaderboard,
    compute_summary,
    load_leaderboard,
    run_all_years,
)

OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)


# ── IronVault verification ───────────────────────────────────────────────────

def _verify_ironvault():
    """Hard-fail if IronVault isn't available. Called once at startup."""
    from shared.iron_vault import IronVault, IronVaultError
    try:
        hd = IronVault.instance()
        # Confirm it's using offline mode (no live Polygon calls)
        assert getattr(hd._hd, "offline_mode", False), (
            "IronVault is NOT in offline_mode — live API calls would fire during sweep!"
        )
        print(f"  IronVault: OK  (offline_mode=True, DB connected)")
        return hd
    except IronVaultError as e:
        print(f"\n  ERROR: {e}")
        print("  Run: python3 scripts/iron_vault_setup.py --verbose")
        sys.exit(1)


# ── Combo generation ─────────────────────────────────────────────────────────

def _generate_combos(phase_cfg: dict, global_fixed: dict) -> list[dict]:
    """
    Generate all parameter combinations for a phase.

    target_dte and min_dte are PAIRED (not cartesian) when both appear in
    the grid — their index positions must match. The sweep config encodes
    this as two parallel lists.
    """
    grid = phase_cfg.get("grid", {})
    phase_fixed = phase_cfg.get("fixed_params", {})

    # Merge global fixed → phase fixed (phase overrides global)
    fixed = {k: v for k, v in global_fixed.items() if not k.startswith("_")}
    fixed.update({k: v for k, v in phase_fixed.items() if not k.startswith("_")})

    # Handle paired DTE keys — zip instead of cartesian product
    paired_keys = []
    free_keys = []
    if "target_dte" in grid and "min_dte" in grid:
        paired_keys = ["target_dte", "min_dte"]

    for key in grid:
        if key not in paired_keys:
            free_keys.append(key)

    # Build list of (key, value) choices for free keys
    free_choices = [[(k, v) for v in grid[k]] for k in free_keys]

    # Paired DTE combinations (zip the two lists)
    if paired_keys:
        dte_pairs = list(zip(grid["target_dte"], grid["min_dte"]))
    else:
        dte_pairs = [None]

    combos = []
    for dte_pair in dte_pairs:
        for free_vals in (itertools.product(*free_choices) if free_choices else [()]):
            params = dict(fixed)
            if dte_pair is not None:
                params["target_dte"] = dte_pair[0]
                params["min_dte"]    = dte_pair[1]
            for k, v in free_vals:
                params[k] = v
            combos.append(params)

    return combos


# ── Deduplication ─────────────────────────────────────────────────────────────

def _params_fingerprint(params: dict, years: list) -> str:
    """
    Stable fingerprint for a (params, years) pair.
    Used to detect already-completed runs in the leaderboard.
    """
    # Only key the fields that actually affect backtester behavior
    relevant_keys = [
        "target_dte", "min_dte", "spread_width", "stop_loss_multiplier",
        "profit_target", "max_risk_per_trade", "direction", "trend_ma_period",
        "regime_mode", "compound", "sizing_mode", "use_delta_selection",
        "otm_pct", "iron_condor_enabled", "ic_neutral_regime_only",
        "ic_risk_per_trade", "ic_min_combined_credit_pct", "min_credit_pct",
    ]
    key_dict = {k: params.get(k) for k in relevant_keys if k in params}
    key_dict["_years"] = sorted(years)
    raw = json.dumps(key_dict, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _build_completed_set(leaderboard: list, years: list) -> set:
    """Return set of fingerprints for already-completed real-data runs."""
    completed = set()
    for entry in leaderboard:
        if entry.get("mode") != "real":
            continue
        if sorted(entry.get("years_run", [])) != sorted(years):
            continue
        fp = _params_fingerprint(entry.get("params", {}), years)
        completed.add(fp)
    return completed


# ── Single combo runner ───────────────────────────────────────────────────────

def _run_combo(combo_idx: int, total: int, params: dict, years: list,
               ticker: str = "SPY", no_validate: bool = True) -> dict:
    """Run one parameter combo across all years. Returns leaderboard entry."""
    import uuid
    run_id = f"sweep_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    t0 = time.time()
    print(f"\n[{combo_idx}/{total}] {run_id}")
    print(f"  dte={params.get('target_dte')}/{params.get('min_dte')}  "
          f"width={params.get('spread_width')}  "
          f"sl={params.get('stop_loss_multiplier')}x  "
          f"pt={params.get('profit_target')}%  "
          f"risk={params.get('max_risk_per_trade')}%  "
          f"dir={params.get('direction', 'both')}  "
          f"ma={params.get('trend_ma_period', 80)}  "
          f"otm={params.get('otm_pct', 0.03)}")

    results_by_year = run_all_years(params, years, ticker=ticker)
    summary = compute_summary(results_by_year)
    elapsed = time.time() - t0

    flag = "✓" if summary["years_profitable"] >= 5 else ("~" if summary["years_profitable"] >= 3 else "✗")
    print(f"  {flag} avg={summary['avg_return']:+.1f}%  "
          f"dd={summary['worst_dd']:.1f}%  "
          f"years={summary['years_profitable']}/{summary['years_total']}  "
          f"({elapsed:.0f}s)")

    # Overfit validation only for promising results to keep sweep fast
    overfit_score = None
    verdict = None
    if not no_validate and summary["avg_return"] > 10 and len(years) >= 4:
        try:
            from scripts.validate_params import validate_params
            val = validate_params(params, results_by_year, years, True, ticker)
            overfit_score = val["overfit_score"]
            verdict = val["verdict"]
            print(f"  overfit={overfit_score:.3f}  {verdict}")
        except Exception as e:
            verdict = f"VALIDATION_ERROR: {e}"

    def _slim(r):
        return {k: v for k, v in r.items() if k not in ("trades", "equity_curve")}

    entry = {
        "run_id":       run_id,
        "timestamp":    datetime.utcnow().isoformat(),
        "params":       params,
        "ticker":       ticker,
        "mode":         "real",
        "years_run":    years,
        "results":      {yr: _slim(r) for yr, r in results_by_year.items()},
        "summary":      summary,
        "overfit_score": overfit_score,
        "verdict":      verdict,
        "elapsed_sec":  round(elapsed),
        "note":         f"sweep:{params.get('_sweep_phase', 'unknown')}",
    }

    append_to_leaderboard(entry)
    return entry


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch parameter sweep using real data (IronVault enforced)"
    )
    parser.add_argument("--sweep",       default="configs/real_data_sweep.json",
                        help="Path to sweep config JSON")
    parser.add_argument("--phase",       help="Phase name to run (e.g. phase1_dte_risk)")
    parser.add_argument("--list-phases", action="store_true", help="List available phases and exit")
    parser.add_argument("--years",       help="Comma-separated years (e.g. 2022,2023,2024)")
    parser.add_argument("--dry-run",     action="store_true", help="Show combo count and params, don't run")
    parser.add_argument("--max-runs",    type=int, default=0, help="Stop after N runs (0 = unlimited)")
    parser.add_argument("--ticker",      default="SPY")
    parser.add_argument("--no-skip",     action="store_true",
                        help="Re-run already-completed combos (default: skip them)")
    parser.add_argument("--validate",    action="store_true",
                        help="Run overfit validation for promising results (slow)")
    args = parser.parse_args()

    # Load sweep config
    sweep_path = ROOT / args.sweep
    if not sweep_path.exists():
        print(f"ERROR: sweep config not found: {sweep_path}")
        sys.exit(1)
    sweep_cfg = json.loads(sweep_path.read_text())

    if args.list_phases:
        phases = sweep_cfg.get("phases", {})
        print("\nAvailable phases:")
        for name, ph in phases.items():
            if name.startswith("_"):
                continue
            est_runs = ph.get("estimated_runs", "?")
            est_min  = ph.get("estimated_minutes", "?")
            desc     = ph.get("_description", "")
            prio     = ph.get("_priority", "?")
            print(f"  [{prio}] {name}: {est_runs} runs ~{est_min}min — {desc}")
        return

    if not args.phase:
        parser.error("--phase is required (use --list-phases to see options)")

    phases = sweep_cfg.get("phases", {})
    if args.phase not in phases:
        print(f"ERROR: phase '{args.phase}' not found. Available: {list(phases.keys())}")
        sys.exit(1)

    phase_cfg    = phases[args.phase]
    global_fixed = {k: v for k, v in sweep_cfg.get("fixed_params", {}).items()
                    if not k.startswith("_")}

    # Years
    if args.years:
        years = [int(y.strip()) for y in args.years.split(",")]
    else:
        years = YEARS

    print()
    print("═" * 72)
    print("  run_sweep.py — Real-Data Parameter Sweep")
    print(f"  Sweep config : {sweep_path.name}")
    print(f"  Phase        : {args.phase}")
    print(f"  Description  : {phase_cfg.get('_description', '')}")
    print(f"  Years        : {years}")
    print(f"  Ticker       : {args.ticker}")
    print(f"  Dry run      : {args.dry_run}")
    print("═" * 72)

    # IronVault check (before generating combos)
    if not args.dry_run:
        print()
        _verify_ironvault()

    # Generate combos
    combos = _generate_combos(phase_cfg, global_fixed)

    # Tag each combo with the phase name (stored in leaderboard note)
    for c in combos:
        c["_sweep_phase"] = args.phase

    print(f"\n  Generated {len(combos)} parameter combinations")

    if args.dry_run:
        print(f"\n  DRY RUN — first 5 combos:")
        for i, c in enumerate(combos[:5]):
            print(f"    [{i+1}] {json.dumps({k: v for k, v in c.items() if not k.startswith('_')})}")
        if len(combos) > 5:
            print(f"    ... and {len(combos)-5} more")
        elapsed_est = len(combos) * 27 / 60
        print(f"\n  Estimated runtime: {elapsed_est:.0f} min ({elapsed_est/60:.1f} hr) @ ~27s/run")
        return

    # Check already-completed combos
    if not args.no_skip:
        leaderboard = load_leaderboard()
        completed   = _build_completed_set(leaderboard, years)
        before      = len(combos)
        combos      = [c for c in combos
                       if _params_fingerprint(c, years) not in completed]
        skipped = before - len(combos)
        if skipped:
            print(f"  Skipping {skipped} already-completed combos "
                  f"({len(combos)} remaining)")

    if args.max_runs and len(combos) > args.max_runs:
        print(f"  Limiting to first {args.max_runs} combos (--max-runs)")
        combos = combos[:args.max_runs]

    if not combos:
        print("\n  All combos already completed. Use --no-skip to re-run.")
        return

    total          = len(combos)
    elapsed_est    = total * 27 / 60
    print(f"  Running {total} combos (est. {elapsed_est:.0f} min)")
    print()

    t_start    = time.time()
    results    = []
    best_avg   = None
    best_combo = None

    for i, params in enumerate(combos, 1):
        try:
            entry = _run_combo(
                i, total, params, years,
                ticker=args.ticker,
                no_validate=not args.validate,
            )
            results.append(entry)

            avg = entry["summary"]["avg_return"]
            if best_avg is None or avg > best_avg:
                best_avg   = avg
                best_combo = params

        except KeyboardInterrupt:
            print(f"\n  Interrupted at combo {i}/{total}. {i-1} runs saved to leaderboard.")
            break
        except Exception as e:
            print(f"\n  ERROR on combo {i}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    elapsed_total = time.time() - t_start
    n_done        = len(results)

    print()
    print("═" * 72)
    print(f"  SWEEP COMPLETE — {n_done}/{total} combos run in {elapsed_total/60:.1f} min")
    if results:
        # Sort results by avg return
        sorted_results = sorted(results, key=lambda r: r["summary"]["avg_return"], reverse=True)
        print(f"\n  Top 10 results:")
        print(f"  {'Avg':>8}  {'DD':>7}  {'Yrs':>5}  Params")
        print("  " + "─" * 68)
        for r in sorted_results[:10]:
            p = r["params"]
            print(f"  {r['summary']['avg_return']:>+7.1f}%  "
                  f"{r['summary']['worst_dd']:>6.1f}%  "
                  f"{r['summary']['years_profitable']}/{r['summary']['years_total']}  "
                  f"dte={p.get('target_dte')}/{p.get('min_dte')} "
                  f"w={p.get('spread_width')} "
                  f"sl={p.get('stop_loss_multiplier')}x "
                  f"pt={p.get('profit_target')}% "
                  f"risk={p.get('max_risk_per_trade')}%")

        if best_combo:
            print(f"\n  Best avg return: {best_avg:+.1f}%")
            print(f"  Best params: {json.dumps({k: v for k, v in best_combo.items() if not k.startswith('_')})}")

    print()
    print(f"  All results saved to output/leaderboard.json")
    print("═" * 72)
    print()


if __name__ == "__main__":
    main()
