#!/usr/bin/env python3
"""
endless_optimizer.py — Autonomous optimization daemon.

Reads current state, picks the next experiment, runs it, logs, repeats.

Phases (auto-escalation):
  Phase 1 — Single strategy optimization (explore each strategy alone)
  Phase 2 — Multi-strategy blending (combine top strategies)
  Phase 3 — Regime-conditional allocation (dynamic per regime)

Usage:
    python3 scripts/endless_optimizer.py                     # run forever
    python3 scripts/endless_optimizer.py --max-runs 100      # stop after 100
    python3 scripts/endless_optimizer.py --phase 2           # start at phase 2
    python3 scripts/endless_optimizer.py --strategies credit_spread,iron_condor
    python3 scripts/endless_optimizer.py --dry-run           # show what it would do
"""

import argparse
import logging
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from engine.optimizer import Optimizer
from scripts.run_optimization import (
    YEARS,
    _build_entry,
    _flatten_params,
    append_to_leaderboard,
    append_to_opt_log,
    build_strategies_config,
    compute_summary,
    extract_yearly_results,
    get_current_best,
    load_leaderboard,
    load_state,
    print_results_table,
    run_full,
    save_state,
)
from strategies import STRATEGY_REGISTRY

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("endless")

DEFAULT_TICKERS = ["SPY"]

# Phase escalation thresholds
PLATEAU_WINDOW = 20       # Check last N runs for improvement
PLATEAU_MIN_IMPROVEMENT = 0.5  # Must improve avg_return by ≥0.5% to not plateau
PHASE_2_MIN_RUNS = 30    # Minimum single-strategy runs before escalating
PHASE_3_MIN_RUNS = 20    # Minimum blending runs before escalating

# Stop flag for graceful shutdown
_shutdown = False


def _signal_handler(sig, frame):
    global _shutdown
    print("\n  Shutdown requested — finishing current run...")
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Phase 1: Single Strategy Optimization ────────────────────────────────────

def phase1_experiment(
    state: Dict,
    strategy_names: List[str],
    years: List[int],
    tickers: List[str],
) -> Dict:
    """Run a single-strategy optimization experiment.

    Cycles through strategies round-robin, using Optimizer.suggest()
    to pick params for each.
    """
    history = state.get("phase1_history", {})

    # Pick next strategy (round-robin based on run counts)
    run_counts = {name: len(history.get(name, [])) for name in strategy_names}
    strategy_name = min(run_counts, key=run_counts.get)

    # Build optimizer for this strategy
    opt = Optimizer(strategy_name=strategy_name)
    strat_history = history.get(strategy_name, [])
    params = opt.suggest(strat_history)

    # Run backtest with just this strategy
    strategies_config = {strategy_name: params}
    results = run_full(strategies_config, years, tickers)
    score = Optimizer.compute_score(results)

    # Record in history
    if strategy_name not in history:
        history[strategy_name] = []
    history[strategy_name].append({"params": params, "score": score})
    state["phase1_history"] = history

    return {
        "strategies_config": strategies_config,
        "results": results,
        "score": score,
        "strategy_name": strategy_name,
    }


# ── Phase 2: Multi-Strategy Blending ─────────────────────────────────────────

def phase2_experiment(
    state: Dict,
    strategy_names: List[str],
    years: List[int],
    tickers: List[str],
) -> Dict:
    """Combine top strategies with optimized params.

    Uses the best params found in Phase 1 for each strategy,
    then adds/removes strategies and perturbs params.
    """
    phase1_history = state.get("phase1_history", {})
    phase2_history = state.get("phase2_history", [])

    # Find best params per strategy from phase 1
    best_per_strategy = {}
    for name in strategy_names:
        strat_hist = phase1_history.get(name, [])
        if strat_hist:
            best = max(strat_hist, key=lambda h: h["score"])
            best_per_strategy[name] = best["params"]

    if not best_per_strategy:
        # No phase 1 data — use defaults
        best_per_strategy = build_strategies_config(strategy_names)

    # Decide which strategies to include (2-5 strategies, or all if fewer)
    import random
    n_strategies = random.randint(min(2, len(strategy_names)), min(5, len(strategy_names)))
    # Weighted by phase 1 score (better strategies more likely)
    scores = []
    for name in strategy_names:
        strat_hist = phase1_history.get(name, [])
        best_score = max((h["score"] for h in strat_hist), default=0)
        scores.append((name, best_score))
    scores.sort(key=lambda x: x[1], reverse=True)

    # Always include top 2, random sample for rest
    selected = [s[0] for s in scores[:2]]
    remaining = [s[0] for s in scores[2:]]
    if remaining and n_strategies > 2:
        extra = random.sample(remaining, min(n_strategies - 2, len(remaining)))
        selected.extend(extra)

    # Build config with best or perturbed params
    strategies_config = {}
    for name in selected:
        if name in best_per_strategy:
            opt = Optimizer(strategy_name=name)
            if random.random() < 0.5 and len(phase2_history) > 5:
                strategies_config[name] = opt.sample_near_best(best_per_strategy[name])
            else:
                strategies_config[name] = dict(best_per_strategy[name])
        else:
            cls = STRATEGY_REGISTRY[name]
            strategies_config[name] = cls.get_default_params()

    results = run_full(strategies_config, years, tickers)
    score = Optimizer.compute_score(results)

    phase2_history.append({"strategies": list(strategies_config.keys()), "score": score})
    state["phase2_history"] = phase2_history

    return {
        "strategies_config": strategies_config,
        "results": results,
        "score": score,
    }


# ── Phase 3: Regime-Conditional Allocation ───────────────────────────────────

def phase3_experiment(
    state: Dict,
    strategy_names: List[str],
    years: List[int],
    tickers: List[str],
) -> Dict:
    """Regime-conditional strategy allocation.

    Different strategy combos for different regimes. The backtester
    already tags each day with a regime (via engine.regime), so
    strategies can use snapshot.regime in generate_signals().

    For Phase 3, we vary the strategy set and params, relying on
    each strategy's internal regime awareness.
    """
    phase1_history = state.get("phase1_history", {})
    phase3_history = state.get("phase3_history", [])

    import random

    # Use broader strategy combos + aggressive param exploration
    n_strategies = random.randint(3, min(7, len(strategy_names)))
    selected = random.sample(strategy_names, n_strategies)

    strategies_config = {}
    for name in selected:
        opt = Optimizer(strategy_name=name)
        strat_hist = phase1_history.get(name, [])

        if strat_hist and random.random() < 0.6:
            best = max(strat_hist, key=lambda h: h["score"])
            strategies_config[name] = opt.sample_near_best(best["params"], noise=0.20)
        else:
            strategies_config[name] = opt.sample_params()

    results = run_full(strategies_config, years, tickers)
    score = Optimizer.compute_score(results)

    phase3_history.append({"strategies": list(strategies_config.keys()), "score": score})
    state["phase3_history"] = phase3_history

    return {
        "strategies_config": strategies_config,
        "results": results,
        "score": score,
    }


# ── Plateau Detection ────────────────────────────────────────────────────────

def is_plateaued(scores: List[float], window: int = PLATEAU_WINDOW,
                 min_improvement: float = PLATEAU_MIN_IMPROVEMENT) -> bool:
    """Check if recent scores show no improvement."""
    if len(scores) < window:
        return False
    recent = scores[-window:]
    first_half = sum(recent[: window // 2]) / (window // 2)
    second_half = sum(recent[window // 2 :]) / (window // 2 - window % 2 + window % 2)
    return (second_half - first_half) < min_improvement


# ── Progress Report ──────────────────────────────────────────────────────────

def print_progress(state: Dict, run_number: int):
    """Print progress report every 100 runs."""
    print()
    print("=" * 72)
    print(f"  PROGRESS REPORT — Run #{run_number}")
    print(f"  Phase: {state.get('current_phase', '?')}")
    print(f"  Total runs: {state.get('total_runs', 0)}")
    print(f"  Best avg return: {state.get('best_avg_return', 'N/A')}")
    print(f"  Best overfit score: {state.get('best_overfit_score', 'N/A')}")
    print(f"  Best run ID: {state.get('best_run_id', 'N/A')}")

    # Phase 1 per-strategy counts
    p1 = state.get("phase1_history", {})
    if p1:
        print("  Phase 1 runs per strategy:")
        for name, hist in sorted(p1.items()):
            best = max((h["score"] for h in hist), default=0)
            print(f"    {name}: {len(hist)} runs, best score={best:.4f}")

    # Phase 2/3 counts
    p2 = state.get("phase2_history", [])
    p3 = state.get("phase3_history", [])
    if p2:
        print(f"  Phase 2 blending runs: {len(p2)}")
    if p3:
        print(f"  Phase 3 regime runs: {len(p3)}")

    print("=" * 72)
    print()


# ── Main Loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Endless optimizer daemon")
    parser.add_argument("--max-runs", type=int, default=0, help="Stop after N runs (0=forever)")
    parser.add_argument("--phase", type=int, default=0, help="Start at phase (1/2/3, 0=auto)")
    parser.add_argument("--strategies", help="Comma-separated strategy names")
    parser.add_argument("--years", help="Comma-separated years")
    parser.add_argument("--tickers", help="Comma-separated tickers")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    parser.add_argument("--report-interval", type=int, default=100,
                        help="Print progress every N runs")
    args = parser.parse_args()

    # Config
    strategy_names = (
        [s.strip() for s in args.strategies.split(",")]
        if args.strategies
        else list(STRATEGY_REGISTRY.keys())
    )
    years = [int(y.strip()) for y in args.years.split(",")] if args.years else YEARS
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else DEFAULT_TICKERS

    # Load state
    state = load_state()
    all_scores: List[float] = []

    # Determine starting phase
    if args.phase > 0:
        current_phase = args.phase
    elif state.get("current_phase_num"):
        current_phase = state["current_phase_num"]
    else:
        current_phase = 1

    state["current_phase"] = f"Phase {current_phase}"
    state["current_phase_num"] = current_phase

    print()
    print("=" * 72)
    print("  ENDLESS OPTIMIZER — Autonomous Optimization Daemon")
    print(f"  Strategies : {', '.join(strategy_names)}")
    print(f"  Years      : {years}")
    print(f"  Tickers    : {tickers}")
    print(f"  Phase      : {current_phase}")
    print(f"  Max runs   : {args.max_runs or 'unlimited'}")
    print("=" * 72)
    print()

    if args.dry_run:
        print("DRY RUN — would start optimization loop.")
        print(f"  Phase 1: Single-strategy optimization ({len(strategy_names)} strategies)")
        print(f"  Phase 2: Multi-strategy blending (after {PHASE_2_MIN_RUNS}+ runs plateau)")
        print(f"  Phase 3: Regime-conditional allocation (after {PHASE_3_MIN_RUNS}+ blending runs)")
        return

    run_number = 0
    start_total = state.get("total_runs", 0)

    while not _shutdown:
        run_number += 1

        if args.max_runs and run_number > args.max_runs:
            print(f"\n  Reached max runs ({args.max_runs}). Stopping.")
            break

        run_id = f"endless_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

        print(f"\n--- Run #{run_number} (phase {current_phase}) [{run_id}] ---")

        # Pick experiment based on phase
        t0 = time.time()
        try:
            if current_phase == 1:
                exp = phase1_experiment(state, strategy_names, years, tickers)
            elif current_phase == 2:
                exp = phase2_experiment(state, strategy_names, years, tickers)
            else:
                exp = phase3_experiment(state, strategy_names, years, tickers)
        except Exception as e:
            logger.exception("Experiment failed: %s", e)
            print(f"  ERROR: {e}")
            time.sleep(1)
            continue

        elapsed = time.time() - t0
        strategies_config = exp["strategies_config"]
        results = exp["results"]
        score = exp["score"]
        all_scores.append(score)

        # Extract yearly + summary
        results_by_year = extract_yearly_results(results)
        summary = compute_summary(results_by_year)

        # Quick validation (skip jitter for speed)
        overfit_score = None
        verdict = None
        if len(years) >= 4:
            try:
                from scripts.validate_params import validate_params
                flat_params = _flatten_params(strategies_config)
                val = validate_params(
                    flat_params, results_by_year, years,
                    use_real=False, ticker=tickers[0], skip_jitter=True,
                )
                overfit_score = val["overfit_score"]
                verdict = val["verdict"]
            except Exception as e:
                logger.warning("Validation failed: %s", e)

        # Print results
        print_results_table(run_id, strategies_config, results_by_year,
                            summary, overfit_score, verdict)

        # Save to leaderboard
        entry = _build_entry(
            run_id, strategies_config, results, results_by_year,
            summary, overfit_score, verdict, {},
            tickers, years,
            note=f"endless phase{current_phase}",
            elapsed_sec=elapsed,
        )
        append_to_leaderboard(entry)

        # Log experiment
        log_entry = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "phase": f"Phase {current_phase}",
            "strategies": list(strategies_config.keys()),
            "score": score,
            "avg_return": summary["avg_return"],
            "overfit_score": overfit_score,
            "verdict": verdict,
            "elapsed_sec": round(elapsed),
            "status": "complete",
        }
        append_to_opt_log(log_entry)

        # Update state
        state["total_runs"] = start_total + run_number
        lb = load_leaderboard()
        best = get_current_best(lb)
        if best:
            state["best_run_id"] = best["run_id"]
            state["best_avg_return"] = best["summary"]["avg_return"]
            state["best_overfit_score"] = best.get("overfit_score")
        save_state(state)

        # Progress report
        if run_number % args.report_interval == 0:
            print_progress(state, run_number)

        # Phase escalation check (only escalate if multiple strategies available)
        if current_phase == 1 and len(strategy_names) >= 2:
            p1_total = sum(len(h) for h in state.get("phase1_history", {}).values())
            p1_scores = []
            for hist in state.get("phase1_history", {}).values():
                p1_scores.extend(h["score"] for h in hist)
            if p1_total >= PHASE_2_MIN_RUNS and is_plateaued(p1_scores):
                print("\n  >>> PHASE 1 PLATEAUED — Escalating to Phase 2 (blending)")
                current_phase = 2
                state["current_phase"] = "Phase 2"
                state["current_phase_num"] = 2
                save_state(state)

        elif current_phase == 2 and len(strategy_names) >= 3:
            p2_hist = state.get("phase2_history", [])
            p2_scores = [h["score"] for h in p2_hist]
            if len(p2_hist) >= PHASE_3_MIN_RUNS and is_plateaued(p2_scores):
                print("\n  >>> PHASE 2 PLATEAUED — Escalating to Phase 3 (regime switching)")
                current_phase = 3
                state["current_phase"] = "Phase 3"
                state["current_phase_num"] = 3
                save_state(state)

    # Graceful shutdown
    print("\n  Saving final state...")
    save_state(state)
    print_progress(state, run_number)
    print("  Endless optimizer stopped.")


if __name__ == "__main__":
    main()
