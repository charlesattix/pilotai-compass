#!/bin/bash
# Watch for exp_099 completion; then launch exp_100 seasonal overlay

set -euo pipefail
cd /Users/charlesbot/projects/pilotai-credit-spreads

LOG=output/exp_100_watcher.log

# PID_099 is passed as $1 or read from the sentinel file
PID_099=${1:-$(cat /tmp/exp_099.pid 2>/dev/null || echo "")}

if [ -z "$PID_099" ]; then
    echo "[$(date)] ERROR: No PID for exp_099. Pass PID as first argument." | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] Watcher started. Monitoring exp_099 PID $PID_099" | tee -a "$LOG"

while kill -0 "$PID_099" 2>/dev/null; do
    echo "[$(date)] exp_099 still running..." | tee -a "$LOG"
    sleep 300
done

echo "[$(date)] exp_099 (PID $PID_099) has exited. Launching exp_100 seasonal..." | tee -a "$LOG"
sleep 10

python3 scripts/run_optimization.py \
    --config configs/exp_100_mega_seasonal.json \
    --run-id exp_100_mega_seasonal \
    --years 2023,2024 \
    --no-validate \
    --hypothesis 'exp_100 seasonal overlay: same as exp_098 pilot + seasonal_sizing boosts summer months (May/Jun/Jul +20%) and reduces winter months (Dec/Jan/Feb -20%). Hypothesis: summer bull markets generate more reliable spreads; winter volatility higher so reduce sizing.' \
    --note 'Phase 6 exp_100: seasonal sizing overlay on combo v2 regime. 12% risk, ICs enabled, compounding.' \
    2>&1 | tee output/exp_100_seasonal.log &

PID_100=$!
echo $PID_100 > /tmp/exp_100.pid
echo "[$(date)] exp_100 launched (PID $PID_100). Starting chain watcher for exp_101..." | tee -a "$LOG"

# Immediately start the next watcher
bash scripts/watch_exp100_launch_101.sh "$PID_100" &

echo "[$(date)] Chain watcher for exp_101 started in background." | tee -a "$LOG"
