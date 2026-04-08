#!/usr/bin/env python3
"""
run_crypto_sweep.py — Run IBIT/BTC crypto credit spread parameter sweep.

Runs combos 0-799 (first half of the 1600-combo IBIT sweep) against
real Deribit BTC data (2020-2024). Maps IBIT sweep params to
BTCCreditSpreadBacktester config.

Usage:
    python3 scripts/run_crypto_sweep.py --start 0 --end 799
    python3 scripts/run_crypto_sweep.py --start 800 --end 1599

Output:
    output/leaderboard.json        — all results, sorted by avg return
    output/optimization_state.json — progress checkpoint every 50 combos
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.btc_credit_spread_backtester import BTCCreditSpreadBacktester
from backtest.crypto_param_sweep import FULL_SWEEP

OUTPUT_DIR = ROOT / "output"
LEADERBOARD_PATH = OUTPUT_DIR / "crypto_sweep_leaderboard.json"
STATE_PATH = OUTPUT_DIR / "optimization_state.json"

YEARS = [2020, 2021, 2022, 2023, 2024]
TRAIN_YEARS = [2020, 2021, 2022]
TEST_YEARS  = [2023, 2024]

# Gate 2 criteria
GATE2_AVG_RETURN = 12.0     # avg annual return ≥ 12%
GATE2_MAX_DD     = -15.0    # max drawdown > -15% (i.e. abs(dd) < 15%)
GATE2_OVERFIT    = 0.70     # overfit score ≥ 0.70

# ---------------------------------------------------------------------------
# Param mapping: IBIT sweep → BTCCreditSpreadBacktester
# ---------------------------------------------------------------------------

# delta → OTM% approximation for BTC puts
# BTC has very high vol, so 0.15-delta ≈ 7% OTM, 0.30-delta ≈ 15% OTM
DELTA_TO_OTM = {
    0.10: 0.05,   # ~5% OTM — deep OTM, conservative
    0.15: 0.07,   # ~7% OTM
    0.20: 0.10,   # ~10% OTM
    0.30: 0.15,   # ~15% OTM — closer to money
}

# Spread width in dollars → % of spot approximation
# BTC ~$30k-$60k, so $500 width ≈ 1-1.5% OTM spread. We use a fixed pct.
# IBIT spread_width in the sweep is 2 (default), not varied — so fixed 5%
SPREAD_WIDTH_PCT = 0.05   # 5% of spot

BASE_RISK_PCT = 0.05      # 5% per trade (from BASE_PARAMS max_risk_per_trade)
BASE_STARTING_CAPITAL = 100_000.0


def regime_risk_scale(params: Dict[str, Any]) -> float:
    """
    Compute an effective risk-per-trade scale from the regime profile.

    Since BTCCreditSpreadBacktester doesn't have a crypto composite regime
    detector, we approximate the regime profile effect by taking the average
    of all 5 regime-band scales. This is a reasonable first-order approximation
    for a diverse multi-year backtest where each regime appears some fraction
    of the time.

    For a more accurate model, we weight by approximate BTC regime distribution:
      extreme_fear  ~15%  (BTC spends ~15% of time in extreme fear)
      cautious      ~20%
      neutral       ~30%
      bullish       ~25%
      extreme_greed ~10%
    """
    weights = {
        "regime_scale_extreme_fear":  0.15,
        "regime_scale_cautious":      0.20,
        "regime_scale_neutral":       0.30,
        "regime_scale_bullish":       0.25,
        "regime_scale_extreme_greed": 0.10,
    }
    scale = sum(params.get(k, 1.0) * w for k, w in weights.items())
    return scale


def sweep_params_to_bt_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """Map a crypto_param_sweep combo dict to BTCCreditSpreadBacktester config."""
    delta = params["target_delta"]
    otm_pct = DELTA_TO_OTM.get(delta, 0.07)

    profit_target_pct = params["profit_target"] / 100.0
    stop_loss_mult = params["stop_loss_multiplier"]
    dte_target = params["target_dte"]

    risk_scale = regime_risk_scale(params)
    effective_risk = BASE_RISK_PCT * risk_scale
    # Clamp risk to reasonable range
    effective_risk = max(0.01, min(effective_risk, 0.15))

    return {
        "starting_capital":     BASE_STARTING_CAPITAL,
        "otm_pct":              otm_pct,
        "spread_width_pct":     SPREAD_WIDTH_PCT,
        "min_credit_pct":       8.0,
        "stop_loss_multiplier": stop_loss_mult,
        "profit_target_pct":    profit_target_pct,
        "dte_target":           dte_target,
        "dte_min":              max(2, dte_target - 7),
        "risk_per_trade_pct":   effective_risk,
        "max_contracts":        20,
        "compound":             True,   # BASE_PARAMS has compound=True
    }


def run_backtest(config: Dict[str, Any], years: List[int]) -> Optional[Dict[str, Any]]:
    """Run backtest and return results dict, or None on error."""
    try:
        bt = BTCCreditSpreadBacktester(config=config)
        return bt.run(years)
    except Exception as e:
        return None


def compute_overfit_score(
    train_results: Optional[Dict],
    test_results: Optional[Dict],
) -> float:
    """
    overfit_score = test_avg_return / train_avg_return.
    Closer to 1.0 = less overfit. Clamped to [-1, 2].
    """
    if not train_results or not test_results:
        return 0.0

    train_yr = train_results.get("year_stats", {})
    test_yr  = test_results.get("year_stats", {})

    train_rets = [train_yr[y]["return_pct"] for y in TRAIN_YEARS if y in train_yr]
    test_rets  = [test_yr[y]["return_pct"]  for y in TEST_YEARS  if y in test_yr]

    train_avg = sum(train_rets) / len(train_rets) if train_rets else 0.0
    test_avg  = sum(test_rets)  / len(test_rets)  if test_rets  else 0.0

    if abs(train_avg) < 0.01:
        return 0.0

    score = test_avg / train_avg
    return round(max(-1.0, min(2.0, score)), 4)


def build_record(
    combo_idx: int,
    params: Dict[str, Any],
    full_results: Dict,
    train_results: Dict,
    test_results: Dict,
    overfit_score: float,
) -> Dict[str, Any]:
    """Build a leaderboard record from backtest results."""
    yr_stats = full_results.get("year_stats", {})

    per_year_returns = {
        str(y): round(yr_stats[y]["return_pct"], 2) if y in yr_stats else None
        for y in YEARS
    }

    year_returns = [yr_stats[y]["return_pct"] for y in YEARS if y in yr_stats]
    avg_return = sum(year_returns) / len(year_returns) if year_returns else 0.0

    # Gate 2 check
    max_dd = full_results.get("max_drawdown", -999.0)
    win_rate = full_results.get("win_rate", 0.0)
    profit_factor = full_results.get("profit_factor", 0.0)

    gate2_pass = (
        avg_return >= GATE2_AVG_RETURN
        and max_dd >= GATE2_MAX_DD          # dd is negative, so >= -15% means within gate
        and overfit_score >= GATE2_OVERFIT
    )

    # Param summary (just the swept keys)
    param_summary = {
        "target_dte":           params["target_dte"],
        "target_delta":         params["target_delta"],
        "profit_target":        params["profit_target"],
        "stop_loss_multiplier": params["stop_loss_multiplier"],
        "regime_profile":       params["regime_profile"],
        "spread_width":         params.get("spread_width", 2),
    }

    return {
        "combo_idx":      combo_idx,
        "params":         param_summary,
        "per_year_returns": per_year_returns,
        "avg_return":     round(avg_return, 2),
        "max_drawdown":   round(max_dd, 2),
        "win_rate":       round(win_rate, 2),
        "profit_factor":  round(profit_factor, 4),
        "overfit_score":  overfit_score,
        "total_trades":   full_results.get("total_trades", 0),
        "gate2_pass":     gate2_pass,
    }


def load_existing_leaderboard() -> List[Dict]:
    """Load existing crypto sweep leaderboard if present."""
    if LEADERBOARD_PATH.exists():
        try:
            with open(LEADERBOARD_PATH) as f:
                data = json.load(f)
            if isinstance(data, list):
                # Validate it has combo_idx format (crypto sweep format)
                if data and "combo_idx" in data[0]:
                    return data
        except Exception:
            pass
    return []


def save_leaderboard(records: List[Dict]):
    """Save leaderboard sorted by avg_return descending."""
    sorted_records = sorted(records, key=lambda r: r.get("avg_return", -999), reverse=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEADERBOARD_PATH, "w") as f:
        json.dump(sorted_records, f, indent=2)


def save_state(combo_idx: int, start: int, end: int, n_done: int, n_gate2: int, elapsed: float):
    """Save optimization state checkpoint."""
    state = {
        "last_completed_idx": combo_idx,
        "range_start":        start,
        "range_end":          end,
        "n_completed":        n_done,
        "n_gate2_pass":       n_gate2,
        "elapsed_seconds":    round(elapsed, 1),
        "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="IBIT/BTC crypto credit spread sweep")
    parser.add_argument("--start", type=int, default=0,   help="Start combo index (inclusive)")
    parser.add_argument("--end",   type=int, default=799, help="End combo index (inclusive)")
    parser.add_argument("--checkpoint", type=int, default=50, help="Save every N combos")
    args = parser.parse_args()

    start_idx = args.start
    end_idx   = min(args.end, len(FULL_SWEEP) - 1)
    combos    = FULL_SWEEP[start_idx : end_idx + 1]

    print(f"\n{'='*60}")
    print(f"BTC/IBIT Credit Spread Parameter Sweep")
    print(f"Combos: {start_idx} - {end_idx}  ({len(combos)} runs)")
    print(f"Years:  {YEARS}")
    print(f"Train:  {TRAIN_YEARS}  |  Test: {TEST_YEARS}")
    print(f"Gate 2: avg_return >= {GATE2_AVG_RETURN}% | "
          f"max_dd >= {GATE2_MAX_DD}% | overfit >= {GATE2_OVERFIT}")
    print(f"{'='*60}\n")

    # Load existing results (may have records from a prior run)
    existing = load_existing_leaderboard()
    existing_idxs = {r["combo_idx"] for r in existing}
    records = list(existing)

    n_done  = 0
    n_gate2 = sum(1 for r in existing if r.get("gate2_pass"))
    t_start = time.time()

    for i, params in enumerate(combos):
        combo_idx = start_idx + i

        # Skip if already done
        if combo_idx in existing_idxs:
            n_done += 1
            continue

        bt_config = sweep_params_to_bt_config(params)

        # Run full backtest (all 5 years)
        full_res = run_backtest(bt_config, YEARS)
        if full_res is None or full_res.get("total_trades", 0) == 0:
            # Record empty result
            records.append({
                "combo_idx":      combo_idx,
                "params":         {
                    "target_dte":           params["target_dte"],
                    "target_delta":         params["target_delta"],
                    "profit_target":        params["profit_target"],
                    "stop_loss_multiplier": params["stop_loss_multiplier"],
                    "regime_profile":       params["regime_profile"],
                },
                "per_year_returns": {},
                "avg_return":     -999.0,
                "max_drawdown":   -999.0,
                "win_rate":       0.0,
                "profit_factor":  0.0,
                "overfit_score":  0.0,
                "total_trades":   0,
                "gate2_pass":     False,
            })
            n_done += 1
            continue

        # Train / test split for overfit score
        train_res = run_backtest(bt_config, TRAIN_YEARS)
        test_res  = run_backtest(bt_config, TEST_YEARS)
        overfit   = compute_overfit_score(train_res, test_res)

        record = build_record(combo_idx, params, full_res, train_res, test_res, overfit)
        records.append(record)
        n_done += 1

        if record["gate2_pass"]:
            n_gate2 += 1
            marker = " *** GATE2 PASS ***"
        else:
            marker = ""

        elapsed = time.time() - t_start
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = (len(combos) - n_done) / rate if rate > 0 else 0

        yr = full_res.get("year_stats", {})
        yr_str = "  ".join(
            f"{y}:{yr[y]['return_pct']:+.0f}%"
            for y in YEARS if y in yr
        )

        print(
            f"[{combo_idx:4d}] "
            f"DTE={params['target_dte']:2d} "
            f"d={params['target_delta']:.2f} "
            f"PT={params['profit_target']:2d}% "
            f"SL={params['stop_loss_multiplier']:.1f}x "
            f"reg={params['regime_profile'][:8]:<8}  "
            f"avg={record['avg_return']:+6.1f}% "
            f"dd={record['max_drawdown']:+6.1f}% "
            f"ovf={overfit:.2f} "
            f"| {yr_str}"
            f"{marker}"
        )

        # Checkpoint every N combos
        if n_done % args.checkpoint == 0:
            save_leaderboard(records)
            save_state(combo_idx, start_idx, end_idx, n_done, n_gate2, elapsed)
            top3 = sorted(records, key=lambda r: r.get("avg_return", -999), reverse=True)[:3]
            print(f"\n  --- Checkpoint @ combo {combo_idx} | "
                  f"{n_done}/{len(combos)} done | "
                  f"Gate2: {n_gate2} passing | "
                  f"ETA: {remaining/60:.1f}min ---")
            for rank, r in enumerate(top3, 1):
                p = r["params"]
                print(f"  #{rank}: DTE={p.get('target_dte')} d={p.get('target_delta')} "
                      f"PT={p.get('profit_target')} SL={p.get('stop_loss_multiplier')} "
                      f"reg={p.get('regime_profile')}  "
                      f"avg={r['avg_return']:+.1f}% dd={r['max_drawdown']:+.1f}% "
                      f"ovf={r['overfit_score']:.2f}")
            print()

    # Final save
    elapsed = time.time() - t_start
    save_leaderboard(records)
    save_state(end_idx, start_idx, end_idx, n_done, n_gate2, elapsed)

    # Final summary
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE: combos {start_idx}-{end_idx}")
    print(f"Elapsed: {elapsed/60:.1f} min  |  {n_done} combos run")
    print(f"Gate 2 passes: {n_gate2}")
    print(f"{'='*60}")

    # Top 10 Gate 2 passes
    gate2_records = [r for r in records if r.get("gate2_pass")]
    gate2_sorted  = sorted(gate2_records, key=lambda r: r.get("avg_return", -999), reverse=True)

    if gate2_sorted:
        print(f"\nTop {min(10, len(gate2_sorted))} Gate 2 Results:")
        print(f"{'Rank':<5} {'Idx':>5} {'DTE':>4} {'Delta':>6} {'PT%':>4} {'SL':>4} "
              f"{'Regime':<12} {'AvgRet':>8} {'MaxDD':>8} {'Ovf':>6} {'WR%':>6}")
        print("-" * 80)
        for rank, r in enumerate(gate2_sorted[:10], 1):
            p = r["params"]
            print(
                f"{rank:<5} {r['combo_idx']:>5} "
                f"{p.get('target_dte', '?'):>4} "
                f"{p.get('target_delta', '?'):>6} "
                f"{p.get('profit_target', '?'):>4} "
                f"{p.get('stop_loss_multiplier', '?'):>4} "
                f"{p.get('regime_profile', '?'):<12} "
                f"{r['avg_return']:>+8.1f}% "
                f"{r['max_drawdown']:>+8.1f}% "
                f"{r['overfit_score']:>6.2f} "
                f"{r['win_rate']:>6.1f}%"
            )
        print()

        # Per-year breakdown for top 3
        print("Per-year returns for top 3:")
        for rank, r in enumerate(gate2_sorted[:3], 1):
            yr_str = "  ".join(
                f"{y}:{v:+.0f}%" if v is not None else f"{y}:N/A"
                for y, v in sorted(r.get("per_year_returns", {}).items())
            )
            print(f"  #{rank} [{r['combo_idx']}]: {yr_str}")
    else:
        print("\nNo Gate 2 passes found in this range.")
        # Show top 10 overall
        top10 = sorted(records, key=lambda r: r.get("avg_return", -999), reverse=True)[:10]
        print(f"\nTop 10 overall (regardless of Gate 2):")
        print(f"{'Rank':<5} {'Idx':>5} {'DTE':>4} {'Delta':>6} {'PT%':>4} {'SL':>4} "
              f"{'Regime':<12} {'AvgRet':>8} {'MaxDD':>8} {'Ovf':>6}")
        print("-" * 75)
        for rank, r in enumerate(top10, 1):
            p = r["params"]
            print(
                f"{rank:<5} {r['combo_idx']:>5} "
                f"{p.get('target_dte', '?'):>4} "
                f"{p.get('target_delta', '?'):>6} "
                f"{p.get('profit_target', '?'):>4} "
                f"{p.get('stop_loss_multiplier', '?'):>4} "
                f"{p.get('regime_profile', '?'):<12} "
                f"{r['avg_return']:>+8.1f}% "
                f"{r['max_drawdown']:>+8.1f}% "
                f"{r['overfit_score']:>6.2f}"
            )

    print(f"\nLeaderboard saved to: {LEADERBOARD_PATH}")
    print(f"State saved to:       {STATE_PATH}")


if __name__ == "__main__":
    main()
