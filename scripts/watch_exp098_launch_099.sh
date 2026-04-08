#!/bin/bash
# Watch for exp_098 completion; then launch exp_099 bear-optimized variant
# Usage: bash scripts/watch_exp098_launch_099.sh <PID_098>

set -euo pipefail
cd /Users/charlesbot/projects/pilotai-credit-spreads

LOG=output/exp_099_watcher.log

PID_098=${1:-$(cat /tmp/exp_098.pid 2>/dev/null || echo "")}

if [ -z "$PID_098" ]; then
    echo "[$(date)] ERROR: No PID for exp_098. Pass PID as first argument." | tee -a "$LOG"
    exit 1
fi

echo "[$(date)] Watcher started. Monitoring exp_098 PID $PID_098" | tee -a "$LOG"

while kill -0 "$PID_098" 2>/dev/null; do
    echo "[$(date)] exp_098 still running..." | tee -a "$LOG"
    sleep 300
done

echo "[$(date)] exp_098 (PID $PID_098) has exited. Launching exp_099 bear-optimized..." | tee -a "$LOG"
sleep 10

python3 scripts/run_optimization.py \
    --config configs/exp_099_mega_bear.json \
    --run-id exp_099_mega_bear \
    --years 2020,2022 \
    --no-validate \
    --hypothesis 'exp_099 bear-optimized: 2020+2022 focus. IC disabled, 10% risk, combo v2 regime. Testing if v2 unanimous BEAR signals correctly flip to bear_call in genuine bear markets (2022 sustained, 2020 COVID spike). Expect BEAR=all3/3 for months in 2022, VIX circuit breaker in COVID March 2020.' \
    --note 'Phase 6 exp_099: bear market validation. IC off, 10% risk, combo v2.' \
    2>&1 | tee output/exp_099_bear.log &

PID_099=$!
echo $PID_099 > /tmp/exp_099.pid
echo "[$(date)] exp_099 launched (PID $PID_099). Starting chain watcher for exp_100..." | tee -a "$LOG"

# Immediately start the next watcher
bash scripts/watch_exp099_launch_100.sh "$PID_099" &

echo "[$(date)] Chain watcher for exp_100 started in background." | tee -a "$LOG"
