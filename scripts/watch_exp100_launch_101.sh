#!/bin/bash
# Watch for exp_100 completion; then launch exp_101 full 6yr walkforward

set -euo pipefail
cd /Users/charlesbot/projects/pilotai-credit-spreads

LOG=output/exp_101_watcher.log

PID_100=${1:-$(cat /tmp/exp_100.pid 2>/dev/null || echo "")}

if [ -z "$PID_100" ]; then
    echo "[$(date)] ERROR: No PID for exp_100. Pass PID as first argument." | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] Watcher started. Monitoring exp_100 PID $PID_100" | tee -a "$LOG"

while kill -0 "$PID_100" 2>/dev/null; do
    echo "[$(date)] exp_100 still running..." | tee -a "$LOG"
    sleep 300
done

echo "[$(date)] exp_100 (PID $PID_100) has exited. Launching exp_101 full 6yr..." | tee -a "$LOG"
sleep 10

python3 scripts/run_optimization.py \
    --config configs/exp_101_mega_6yr.json \
    --run-id exp_101_mega_6yr \
    --years 2020,2021,2022,2023,2024,2025 \
    --no-validate \
    --hypothesis 'exp_101 full 6yr walkforward: combo v2 regime detector, 12% risk, ICs enabled, compounding, drawdown CB 35%. This is the ultimate walk-forward validation. Train context: exp_098 (2023+2024 pilot) + exp_099 (2020+2022 bear). This is the out-of-sample confirmation run for the full Phase 6 combo regime system.' \
    --note 'Phase 6 exp_101: full 6yr walkforward. combo v2 regime, 12% risk, IC enabled, compounding, CB=35%.' \
    2>&1 | tee output/exp_101_6yr.log

echo "[$(date)] exp_101 complete. MISSION COMPLETE — check output/exp_101_6yr.log" | tee -a "$LOG"
