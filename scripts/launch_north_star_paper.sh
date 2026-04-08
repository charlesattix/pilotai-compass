#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# launch_north_star_paper.sh — EXP-1900 North Star paper deployment launcher
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/launch_north_star_paper.sh smoke    # smoke check (no trades)
#   ./scripts/launch_north_star_paper.sh dry      # dry-run (no order submit)
#   ./scripts/launch_north_star_paper.sh start    # start paper engine + monitor
#   ./scripts/launch_north_star_paper.sh stop     # stop both processes
#   ./scripts/launch_north_star_paper.sh status   # report status
#   ./scripts/launch_north_star_paper.sh logs     # tail combined logs
#   ./scripts/launch_north_star_paper.sh monitor  # run monitor in foreground
#
# Reads configs/north_star_paper.yaml. Writes pid + health files under
# logs/north_star/. Loads env from .env or .env.north_star (if present).
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/north_star_paper.yaml"
LOG_DIR="logs/north_star"
PID_FILE="$LOG_DIR/north_star.pid"
MON_PID_FILE="$LOG_DIR/monitor.pid"
HEALTH_FILE="$LOG_DIR/health.json"
MAIN_LOG="$LOG_DIR/north_star.log"
MON_LOG="$LOG_DIR/monitor.log"

mkdir -p "$LOG_DIR"

# ─── Load environment ────────────────────────────────────────────────────
ENV_FILE=".env"
[ -f ".env.north_star" ] && ENV_FILE=".env.north_star"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[launch] env loaded from $ENV_FILE"
else
    echo "[launch] WARN: no $ENV_FILE — Alpaca/Telegram credentials must be in shell env"
fi

# ─── Required env vars ───────────────────────────────────────────────────
require_env() {
    local missing=0
    for v in "$@"; do
        if [ -z "${!v:-}" ]; then
            echo "[launch] ERROR: env var $v is unset"
            missing=1
        fi
    done
    return $missing
}

PYTHON="${PYTHON:-python3}"

cmd_smoke() {
    echo "[launch] SMOKE: validating config + dependencies"
    [ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }
    "$PYTHON" -c "import yaml,sys; yaml.safe_load(open('$CONFIG'))" || exit 1
    "$PYTHON" - <<'PY'
import importlib, sys
mods = [
    "compass.tail_risk_hedge",
    "compass.gld_tlt_relval",
    "compass.crisis_alpha_v5",
    "compass.exp1750_putcall_overlay",
    "compass.risk_overlay",
    "compass.telegram_alerter",
]
ok = True
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  ✓ {m}")
    except Exception as e:
        print(f"  ✗ {m}: {e}")
        ok = False
sys.exit(0 if ok else 1)
PY
    require_env ALPACA_API_KEY_PAPER ALPACA_API_SECRET_PAPER || true
    require_env TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID || true
    echo "[launch] smoke OK"
}

cmd_dry() {
    echo "[launch] DRY-RUN: scanning, no order submission"
    require_env ALPACA_API_KEY_PAPER ALPACA_API_SECRET_PAPER
    "$PYTHON" -m compass.paper_engine --config "$CONFIG" --dry-run --once \
        2>&1 | tee -a "$MAIN_LOG"
}

cmd_start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "[launch] already running (pid $(cat "$PID_FILE"))"
        exit 0
    fi
    require_env ALPACA_API_KEY_PAPER ALPACA_API_SECRET_PAPER
    require_env TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID
    echo "[launch] starting paper engine"
    nohup "$PYTHON" -m compass.paper_engine --config "$CONFIG" \
        >> "$MAIN_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1

    echo "[launch] starting monitor (5-min interval)"
    nohup "$PYTHON" "$SCRIPT_DIR/north_star_monitor.py" \
        --config "$CONFIG" --interval 300 \
        >> "$MON_LOG" 2>&1 &
    echo $! > "$MON_PID_FILE"

    echo "[launch] paper engine pid=$(cat "$PID_FILE") monitor pid=$(cat "$MON_PID_FILE")"
    echo "[launch] tail -f $MAIN_LOG  $MON_LOG"
}

cmd_stop() {
    for f in "$PID_FILE" "$MON_PID_FILE"; do
        if [ -f "$f" ]; then
            pid=$(cat "$f")
            if kill -0 "$pid" 2>/dev/null; then
                echo "[launch] stopping pid $pid"
                kill "$pid"
                sleep 1
                kill -0 "$pid" 2>/dev/null && kill -9 "$pid" || true
            fi
            rm -f "$f"
        fi
    done
    echo "[launch] stopped"
}

cmd_status() {
    echo "[launch] status:"
    for label in engine monitor; do
        f="$PID_FILE"
        [ "$label" = "monitor" ] && f="$MON_PID_FILE"
        if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
            echo "  $label: RUNNING (pid $(cat "$f"))"
        else
            echo "  $label: STOPPED"
        fi
    done
    if [ -f "$HEALTH_FILE" ]; then
        echo "  health: $HEALTH_FILE"
        "$PYTHON" -c "import json; print(json.dumps(json.load(open('$HEALTH_FILE')), indent=2))" 2>/dev/null || cat "$HEALTH_FILE"
    fi
}

cmd_logs() {
    tail -F "$MAIN_LOG" "$MON_LOG" 2>/dev/null
}

cmd_monitor() {
    require_env ALPACA_API_KEY_PAPER ALPACA_API_SECRET_PAPER
    "$PYTHON" "$SCRIPT_DIR/north_star_monitor.py" \
        --config "$CONFIG" --interval 300 --foreground
}

case "${1:-smoke}" in
    smoke)   cmd_smoke ;;
    dry)     cmd_dry ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    logs)    cmd_logs ;;
    monitor) cmd_monitor ;;
    *)       echo "usage: $0 {smoke|dry|start|stop|status|logs|monitor}"; exit 2 ;;
esac
