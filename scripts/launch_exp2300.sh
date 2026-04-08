#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# launch_exp2300.sh — EXP-2300 North Star v6 Paper Trading Launcher
# ═══════════════════════════════════════════════════════════════════════════
#
# Deployment launcher for the 7-stream EXP-2200 equal_risk_15% portfolio.
# Target: Mac Studio, Charles.
#
# Usage:
#   ./scripts/launch_exp2300.sh smoke    # load configs, verify access, no trades
#   ./scripts/launch_exp2300.sh dry      # smoke + signal computation, no trades
#   ./scripts/launch_exp2300.sh start    # start paper runner + monitor
#   ./scripts/launch_exp2300.sh stop     # stop background processes
#   ./scripts/launch_exp2300.sh status   # report status + health snapshot
#   ./scripts/launch_exp2300.sh logs     # tail runner log
#
# Reads configs/exp2300_north_star_v6_paper.yaml. Writes pid + health
# under logs/exp2300/. Loads env from .env.exp2300 (preferred) or .env.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/exp2300_north_star_v6_paper.yaml"
LOG_DIR="logs/exp2300"
PID_FILE="$LOG_DIR/runner.pid"
HEALTH_FILE="$LOG_DIR/health.json"
RUNNER_LOG="$LOG_DIR/runner.log"

mkdir -p "$LOG_DIR"

# ─── Environment loading ─────────────────────────────────────────────────
ENV_FILE=".env"
[ -f ".env.exp2300" ] && ENV_FILE=".env.exp2300"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[exp2300] env loaded from $ENV_FILE"
else
    echo "[exp2300] WARN: no $ENV_FILE (credentials must be in shell env)"
fi

# ─── Env var preflight ───────────────────────────────────────────────────
require_env() {
    local missing=0
    for var in "$@"; do
        if [ -z "${!var:-}" ]; then
            echo "[exp2300] MISSING env var: $var"
            missing=1
        fi
    done
    return $missing
}

# ─── Mode dispatch ───────────────────────────────────────────────────────
MODE="${1:-}"
if [ -z "$MODE" ]; then
    echo "Usage: $0 {smoke|dry|start|stop|status|logs}"
    exit 1
fi

case "$MODE" in
    smoke)
        echo "[exp2300] smoke check"
        python3 -m compass.exp2300_portfolio_runner \
            --config "$CONFIG" --mode smoke
        ;;

    dry)
        echo "[exp2300] dry run"
        python3 -m compass.exp2300_portfolio_runner \
            --config "$CONFIG" --mode dry
        ;;

    start)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "[exp2300] already running, pid $(cat "$PID_FILE")"
            exit 0
        fi
        require_env ALPACA_API_KEY_PAPER ALPACA_API_SECRET_PAPER || {
            echo "[exp2300] REFUSING TO START — missing credentials"
            exit 1
        }
        echo "[exp2300] starting paper runner in background"
        nohup python3 -m compass.exp2300_portfolio_runner \
            --config "$CONFIG" --mode paper \
            > "$RUNNER_LOG" 2>&1 &
        RPID=$!
        echo "$RPID" > "$PID_FILE"
        sleep 2
        if kill -0 "$RPID" 2>/dev/null; then
            echo "[exp2300] started, pid $RPID, logging to $RUNNER_LOG"
        else
            echo "[exp2300] FAILED — process died immediately, check $RUNNER_LOG"
            rm -f "$PID_FILE"
            exit 1
        fi
        ;;

    stop)
        if [ ! -f "$PID_FILE" ]; then
            echo "[exp2300] no pid file — nothing to stop"
            exit 0
        fi
        RPID="$(cat "$PID_FILE")"
        if kill -0 "$RPID" 2>/dev/null; then
            echo "[exp2300] stopping pid $RPID"
            kill -TERM "$RPID"
            sleep 3
            if kill -0 "$RPID" 2>/dev/null; then
                echo "[exp2300] still running after SIGTERM — SIGKILL"
                kill -KILL "$RPID" || true
            fi
        fi
        rm -f "$PID_FILE"
        echo "[exp2300] stopped"
        ;;

    status)
        if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
            echo "[exp2300] RUNNING pid $(cat "$PID_FILE")"
        else
            echo "[exp2300] STOPPED"
        fi
        if [ -f "$HEALTH_FILE" ]; then
            echo "[exp2300] health snapshot ($HEALTH_FILE):"
            python3 -c "
import json, sys
d = json.load(open('$HEALTH_FILE'))
print('  experiment:', d.get('experiment'))
print('  mode:', d.get('mode'))
print('  generated:', d.get('generated'))
print('  all_checks_passed:', d.get('all_checks_passed'))
print('  sleeves:')
for s in d.get('sleeves', []):
    print(f\"    {s['id']:16s}  w={s['weight']:.4f}  mode={s['mode']:12s}  \"
          f\"status={s.get('status','?')}\")
"
        else
            echo "[exp2300] no health file yet"
        fi
        ;;

    logs)
        if [ -f "$RUNNER_LOG" ]; then
            tail -f "$RUNNER_LOG"
        else
            echo "[exp2300] no runner log at $RUNNER_LOG"
        fi
        ;;

    *)
        echo "[exp2300] unknown mode: $MODE"
        echo "Usage: $0 {smoke|dry|start|stop|status|logs}"
        exit 1
        ;;
esac
