#!/bin/bash
# Watch for exp_093 completion; parse 2023 result; auto-launch exp_094 if 2023 still negative

set -euo pipefail
cd /Users/charlesbot/projects/pilotai-credit-spreads

PID_093=97966
LOG=/Users/charlesbot/projects/pilotai-credit-spreads/output/exp_093_watcher.log

echo "[$(date)] Watcher started. Monitoring PID $PID_093 (exp_093 MA50 2023-only)" | tee -a "$LOG"

# Wait for process to finish
while kill -0 $PID_093 2>/dev/null; do
    echo "[$(date)] exp_093 still running..." | tee -a "$LOG"
    sleep 120
done

echo "[$(date)] exp_093 (PID $PID_093) has exited." | tee -a "$LOG"

# Give it a moment for file writes
sleep 5

# Pull 2023 result from leaderboard
RESULT_2023=$(python3 - << 'PYEOF'
import json, sys
with open("output/leaderboard.json") as f:
    lb = json.load(f)
# Find most recent exp_093 entry
matches = [e for e in lb if "exp_093" in e.get("run_id","")]
if not matches:
    print("NOT_FOUND")
    sys.exit(0)
latest = matches[-1]
years = latest.get("yearly_results", {})
r2023 = years.get("2023", {})
ret = r2023.get("return_pct", r2023.get("total_return_pct", None))
trades = r2023.get("total_trades", r2023.get("trade_count", "?"))
dd = r2023.get("max_drawdown", r2023.get("max_drawdown_pct", "?"))
print(f"return={ret} trades={trades} dd={dd}")
PYEOF
)

echo "[$(date)] exp_093 2023 result: $RESULT_2023" | tee -a "$LOG"

# Parse return value
RETURN=$(echo "$RESULT_2023" | grep -oE 'return=[-0-9.]+' | cut -d= -f2 || echo "UNKNOWN")

echo "[$(date)] 2023 return_pct=$RETURN" | tee -a "$LOG"

# Check if negative
if python3 -c "import sys; r=float('${RETURN}'); sys.exit(0 if r < 0 else 1)" 2>/dev/null; then
    echo "[$(date)] 2023 STILL NEGATIVE ($RETURN%). Launching exp_094 (OTM=4%, MA200)..." | tee -a "$LOG"
    python3 scripts/run_optimization.py \
        --config configs/exp_094_otm4pct.json \
        --run-id exp_094_otm4pct_2023only \
        --years 2023 \
        --no-validate \
        --hypothesis 'OTM=4% instead of 3% for more cushion. MA50 (exp_093) did not fix 2023. More OTM = lower credit but higher WR buffer.' \
        --note 'Phase 5 2023 fix: wider OTM to push strikes further from market' \
        2>&1 | tee output/exp_094.log
    echo "[$(date)] exp_094 complete." | tee -a "$LOG"
else
    echo "[$(date)] 2023 POSITIVE ($RETURN%)! MA50 fixed it. No need for exp_094." | tee -a "$LOG"
    echo "[$(date)] Next step: run exp_093 config on full 6yr to confirm." | tee -a "$LOG"
fi

echo "[$(date)] Watcher done." | tee -a "$LOG"
