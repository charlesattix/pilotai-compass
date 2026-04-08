#!/bin/bash
# Watch exp_095 (MA100) and exp_096 (MA150) 2023+2024 tests
# When both complete, analyze and launch full 6yr with the winner
set -euo pipefail
cd /Users/charlesbot/projects/pilotai-credit-spreads
LOG=output/ma_sweep_watcher.log

echo "[$(date)] MA sweep watcher started. Monitoring exp_095 (MA100) and exp_096 (MA150)" | tee -a "$LOG"

wait_for_run() {
    local run_id="$1"
    echo "[$(date)] Waiting for $run_id to appear in leaderboard..." | tee -a "$LOG"
    while true; do
        found=$(python3 -c "
import json
with open('output/leaderboard.json') as f: lb=json.load(f)
matches=[e for e in lb if '$run_id'==e.get('run_id','')]
print(len(matches))
" 2>/dev/null || echo "0")
        if [ "$found" -gt "0" ]; then
            echo "[$(date)] $run_id found in leaderboard." | tee -a "$LOG"
            break
        fi
        sleep 120
    done
}

get_result() {
    local run_id="$1"
    python3 -c "
import json
with open('output/leaderboard.json') as f: lb=json.load(f)
m=[e for e in lb if '$run_id'==e.get('run_id','')][-1]
results=m['results']
for yr in ['2023','2024']:
    r=results.get(yr,{})
    print(f'{yr}: return={r.get(\"return_pct\",\"NA\")}%  trades={r.get(\"total_trades\",\"NA\")}  bear_call_wr={r.get(\"bear_call_win_rate\",\"NA\")}%  DD={r.get(\"max_drawdown\",\"NA\")}')
" 2>&1
}

both_positive() {
    local run_id="$1"
    python3 -c "
import json, sys
with open('output/leaderboard.json') as f: lb=json.load(f)
m=[e for e in lb if '$run_id'==e.get('run_id','')][-1]
r=m['results']
r23=r.get('2023',{}).get('return_pct',None)
r24=r.get('2024',{}).get('return_pct',None)
if r23 is not None and r24 is not None and r23>0 and r24>0:
    sys.exit(0)  # both positive
sys.exit(1)  # at least one negative
" 2>/dev/null
}

wait_for_run "exp_095_ma100_2023_2024"
echo "[$(date)] exp_095 MA100 results:" | tee -a "$LOG"
get_result "exp_095_ma100_2023_2024" | tee -a "$LOG"

wait_for_run "exp_096_ma150_2023_2024"
echo "[$(date)] exp_096 MA150 results:" | tee -a "$LOG"
get_result "exp_096_ma150_2023_2024" | tee -a "$LOG"

echo "[$(date)] Both complete. Evaluating Goldilocks winner..." | tee -a "$LOG"

# Determine winner: prefer MA100 if both years positive, else MA150, else report no winner
if both_positive "exp_095_ma100_2023_2024"; then
    WINNER_CONFIG="configs/exp_095_ma100.json"
    WINNER_RUN="exp_095_ma100_6yr"
    WINNER_LABEL="MA100"
    echo "[$(date)] WINNER: MA100 (both 2023+2024 positive)" | tee -a "$LOG"
elif both_positive "exp_096_ma150_2023_2024"; then
    WINNER_CONFIG="configs/exp_096_ma150.json"
    WINNER_RUN="exp_096_ma150_6yr"
    WINNER_LABEL="MA150"
    echo "[$(date)] WINNER: MA150 (both 2023+2024 positive, MA100 failed)" | tee -a "$LOG"
else
    echo "[$(date)] NO WINNER — neither MA100 nor MA150 makes both 2023+2024 positive. Need manual analysis." | tee -a "$LOG"
    get_result "exp_095_ma100_2023_2024" | tee -a "$LOG"
    get_result "exp_096_ma150_2023_2024" | tee -a "$LOG"
    exit 0
fi

echo "[$(date)] Launching $WINNER_LABEL full 6yr run: $WINNER_RUN" | tee -a "$LOG"
python3 scripts/run_optimization.py \
    --config "$WINNER_CONFIG" \
    --run-id "$WINNER_RUN" \
    --years 2020,2021,2022,2023,2024,2025 \
    --no-validate \
    --hypothesis "$WINNER_LABEL Goldilocks 6yr — both 2023 and 2024 confirmed positive in sweep test. Verify full 6yr avg>30% and all DDs<40%." \
    --note "Phase 5 MA sweet spot winner — full 6yr validation" \
    2>&1 | tee "output/${WINNER_RUN}.log"

echo "[$(date)] $WINNER_RUN complete. Check output/${WINNER_RUN}.log for results." | tee -a "$LOG"
