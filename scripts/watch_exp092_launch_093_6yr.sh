#!/bin/bash
# Watch for exp_092 6yr completion; then launch exp_093 MA50 full 6yr run

set -euo pipefail
cd /Users/charlesbot/projects/pilotai-credit-spreads

PID_092=97920
LOG=output/exp_093_6yr_watcher.log

echo "[$(date)] Watcher started. Monitoring exp_092 PID $PID_092" | tee -a "$LOG"

while kill -0 $PID_092 2>/dev/null; do
    echo "[$(date)] exp_092 still running..." | tee -a "$LOG"
    sleep 300
done

echo "[$(date)] exp_092 (PID $PID_092) has exited. Launching exp_093 MA50 full 6yr..." | tee -a "$LOG"
sleep 10

python3 scripts/run_optimization.py \
    --config configs/exp_093_ma50_2023fix.json \
    --run-id exp_093_ma50_6yr \
    --years 2020,2021,2022,2023,2024,2025 \
    --no-validate \
    --hypothesis 'MA50 full 6yr — 2023 confirmed +10.07% (vs exp_090 -12%). Check 2020 COVID DD and overall 6yr avg. Bear call WR in 2023 went from 46%->71% with MA50.' \
    --note 'Phase 5 MA50 6yr validation. If 6yr avg>30% and all DDs<40%, this becomes new champion.' \
    2>&1 | tee output/exp_093_6yr.log

echo "[$(date)] exp_093_6yr complete. Check output/exp_093_6yr.log for results." | tee -a "$LOG"
