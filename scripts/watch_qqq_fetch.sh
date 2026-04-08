#!/usr/bin/env bash
# watch_qqq_fetch.sh — polls QQQ fetch progress and auto-launches backtest when done.
# Usage: bash scripts/watch_qqq_fetch.sh [config] [log_prefix]
#
# Waits until the QQQ fetch reaches 100% (or no fetch process is running),
# then fires run_optimization.py --ticker QQQ for each provided config.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

FETCH_LOG="${FETCH_LOG:-/tmp/qqq_fetch.log}"
POLL_SECONDS=120  # check every 2 min

configs=(
  "configs/exp_249_qqq_champion.json:exp_249_qqq_champion"
  "configs/exp_250_qqq_dte40.json:exp_250_qqq_dte40"
  "configs/exp_251_qqq_base.json:exp_251_qqq_base"
)

is_fetch_done() {
  # Done if: last progress line shows 100%, OR no fetch process running
  if ! pgrep -f "fetch_sector_options.py.*QQQ" > /dev/null 2>&1; then
    return 0
  fi
  local last_pct
  last_pct=$(grep -o '[0-9]*\.[0-9]*%' "$FETCH_LOG" 2>/dev/null | tail -1 | tr -d '%')
  if [[ -n "$last_pct" ]] && awk "BEGIN{exit !($last_pct >= 99.9)}"; then
    return 0
  fi
  return 1
}

echo "$(date) — Watching QQQ fetch ($FETCH_LOG)"
echo "Will auto-launch: ${configs[*]}"

while ! is_fetch_done; do
  last_line=$(grep "Progress:" "$FETCH_LOG" 2>/dev/null | tail -1 || echo "no progress yet")
  echo "$(date +%H:%M:%S) — fetch in progress: $last_line"
  sleep "$POLL_SECONDS"
done

echo "$(date) — QQQ fetch complete! Loading .env and launching backtests..."
set -a && source "$ROOT/.env" && set +a

for entry in "${configs[@]}"; do
  cfg="${entry%%:*}"
  lbl="${entry##*:}"
  log="/tmp/${lbl}.log"
  echo "Launching: $cfg → $log"
  python3 scripts/run_optimization.py \
    --config "$cfg" \
    --ticker QQQ \
    --note "QQQ champion (exp_231 params) — first QQQ run after full data fetch" \
    > "$log" 2>&1 &
  echo "  PID: $!"
done

echo "$(date) — All QQQ backtests launched. Logs: /tmp/exp_249_qqq_champion.log"
