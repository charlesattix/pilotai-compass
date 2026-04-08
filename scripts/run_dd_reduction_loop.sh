#!/bin/bash
# DD-Reduction Experiment Loop — Phase 5
# Runs experiments 082-088 sequentially after exp_081 finishes
# Usage: bash scripts/run_dd_reduction_loop.sh

set -e
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_exp() {
    local config="$1"
    local run_id="$2"
    local hypothesis="$3"
    local note="$4"
    log "=== Starting $run_id ==="
    python3 scripts/run_optimization.py \
        --config "$config" \
        --run-id "$run_id" \
        --hypothesis "$hypothesis" \
        --note "$note"
    log "=== Completed $run_id ==="
}

log "DD-Reduction Loop starting — exp_082 through exp_088"

# exp_082: 5% risk, no compound, circuit breaker=30% from start
# NOTE: In non-compound mode, CB is from starting_capital (not peak).
# If 2020 profits first take equity above $100k, CB may never trigger.
# Including for completeness.
run_exp \
    configs/exp_082_risk5_nocompound_cb30.json \
    exp_082_risk5_nocompound_cb30 \
    "5% risk, no compound, CB=30% from starting capital. Baseline exp_076 has 2020 DD=-42.9% from equity peak. CB is measured from starting capital in non-compound mode, so fires when equity < 70k. If 2020 profits build equity above 100k before the crash, CB fires later. Testing whether this has any impact on max DD." \
    "Phase 5 DD reduction: tighter CB=30%"

# exp_083: 5% risk, no compound, tighter stop loss 1.5x
# Reduces per-trade max loss: stops at 1.5x credit vs 2.5x credit.
# Estimated savings: ~14 losing trades × ~$300 savings = $4,200 < $3,800 needed.
run_exp \
    configs/exp_083_risk5_nocompound_sl15x.json \
    exp_083_risk5_nocompound_sl15x \
    "5% risk, no compound, stop_loss_multiplier=1.5x (vs 2.5x baseline). Tighter stops reduce per-trade max loss: at 2.5x stop, loss=(2.5-1)xcredit; at 1.5x, loss=(1.5-1)xcredit — 3x reduction. With ~14 losing trades in 2020, estimated savings = 14 × 300 = ~$4200, enough to reduce 2020 DD from -42.9% to < -40%. Risk: more false stops on trades that would have recovered." \
    "Phase 5 DD reduction: tighter stop loss 1.5x"

# exp_084: 5% risk, no compound, MA50 (faster trend detection)
# MA50 turns bearish before MA200 in a crash, reducing bull put entries
run_exp \
    configs/exp_084_risk5_nocompound_ma50.json \
    exp_084_risk5_nocompound_ma50 \
    "5% risk, no compound, trend_ma_period=50 (vs MA200 baseline). Faster bearish signal: SPY 50-day MA would have turned bearish in early March 2020 (before the worst of the crash), reducing new bull put entries. Trade-off: MA50 also flips more during 2022 bear market recoveries, potentially stopping bear calls prematurely." \
    "Phase 5 DD reduction: faster trend MA50"

# exp_085: 5% risk, no compound, VIX dynamic (full<20, half<28, qtr<35) + CB=35%
# Combo approach
run_exp \
    configs/exp_085_risk5_nocompound_vix_cb35.json \
    exp_085_risk5_nocompound_vix_cb35 \
    "5% risk, no compound, VIX dynamic sizing (full<20, half<28, qtr<35) + CB=35%. Combo: VIX scaling reduces size during crash, tighter CB reduces new entries faster. Note: exp_081 (same VIX thresholds) cut returns heavily (2020: +9.9%, 2021: +4.8%). With CB=35% added, may not help if VIX scaling already over-reduces returns." \
    "Phase 5 DD reduction: VIX dynamic + CB=35% combo"

# exp_086: 5% risk, no compound, portfolio exposure cap=30%
# Hard cap on total concurrent max-loss exposure at 30% of equity
# Limits to ~6 concurrent positions vs ~24 in baseline
run_exp \
    configs/exp_086_risk5_nocompound_exposure30.json \
    exp_086_risk5_nocompound_exposure30 \
    "5% risk, no compound, max_portfolio_exposure_pct=30. Hard cap: total max-loss across all open positions <= 30% of equity. At 5% per trade, max 6 concurrent positions (30%/5%). Theoretical max DD = 30% if all positions hit max loss simultaneously. Expected: 2020 DD drops dramatically but annual trades drop from ~172 to ~43, cutting returns from +57% to maybe +15%. Too conservative?" \
    "Phase 5 DD reduction: portfolio exposure cap 30%"

# exp_087: 5% risk, no compound, portfolio exposure cap=50%
# Less restrictive - allows 10 concurrent positions
run_exp \
    configs/exp_087_risk5_nocompound_exposure50.json \
    exp_087_risk5_nocompound_exposure50 \
    "5% risk, no compound, max_portfolio_exposure_pct=50. Softer exposure cap: max 10 concurrent positions (50%/5%). Theoretical max DD = 50% if all positions hit max loss — but with 90% win rate, actual max DD much lower. Expected: 2020 DD drops to ~20-25%, 2022 DD drops to ~20%, returns drop moderately (172→~100 trades/year). Should still pass P50>30% target." \
    "Phase 5 DD reduction: portfolio exposure cap 50%"

# exp_088: 5% risk, no compound, VIX dynamic LOOSE (full<25, half<35, qtr<50)
# Much gentler thresholds: only reduces size during genuine stress (VIX>25)
run_exp \
    configs/exp_088_risk5_nocompound_vix_loose.json \
    exp_088_risk5_nocompound_vix_loose \
    "5% risk, no compound, VIX dynamic sizing LOOSE (full<25, half<35, qtr<50). More realistic thresholds: full size up to VIX=25 (normal range), half size VIX 25-35 (elevated), quarter size VIX 35-50 (very high), minimal above 50 (extreme like March 2020 peak). Should preserve 2021 returns (VIX mostly <25), reduce 2022 returns moderately, and cut 2020 crash exposure significantly (VIX hit 85)." \
    "Phase 5 DD reduction: VIX dynamic loose thresholds full<25 half<35 qtr<50"

log "All DD-reduction experiments complete!"
log "Check output/leaderboard.json for results."
