#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# launch_north_star_v6.sh — EXP-2290 North Star v6 launcher (Mac Studio)
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/launch_north_star_v6.sh smoke    # env+import+config validation
#   ./scripts/launch_north_star_v6.sh dry      # scan loop, no order submit
#   ./scripts/launch_north_star_v6.sh start    # foreground paper engine
#   ./scripts/launch_north_star_v6.sh daemon   # background engine + monitor
#   ./scripts/launch_north_star_v6.sh stop     # stop engine + monitor
#   ./scripts/launch_north_star_v6.sh status   # summary + health.json
#   ./scripts/launch_north_star_v6.sh logs     # tail engine + monitor
#   ./scripts/launch_north_star_v6.sh report   # run the daily P&L report now
#   ./scripts/launch_north_star_v6.sh close-all  # flatten every open sleeve
#   ./scripts/launch_north_star_v6.sh install-launchd  # macOS LaunchAgent
#
# Loads `.env` or `.env.north_star_v6`. Writes pid/log files under
# logs/north_star_v6/. Fails fast if required env vars are missing.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/north_star_v6_prod.yaml"
LOG_DIR="logs/north_star_v6"
ENGINE_PID="$LOG_DIR/engine.pid"
MONITOR_PID="$LOG_DIR/monitor.pid"
HEALTH_FILE="$LOG_DIR/health.json"
ENGINE_LOG="$LOG_DIR/engine.log"
MONITOR_LOG="$LOG_DIR/monitor.log"
REPORT_LOG="$LOG_DIR/report.log"

mkdir -p "$LOG_DIR" "reports/north_star_v6"

# ─── Environment ────────────────────────────────────────────────────────
ENV_FILE=".env"
[ -f ".env.north_star_v6" ] && ENV_FILE=".env.north_star_v6"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[launch] loaded $ENV_FILE"
else
    echo "[launch] WARNING: no $ENV_FILE — Alpaca/Telegram creds must be in shell env"
fi

PYTHON="${PYTHON:-python3}"

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

# ─── Commands ───────────────────────────────────────────────────────────

cmd_smoke() {
    echo "[launch] SMOKE: validating config + modules"
    [ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }

    "$PYTHON" - <<PY
import yaml, sys
cfg = yaml.safe_load(open("$CONFIG"))
assert cfg["experiment_id"] == "EXP-2290", f"wrong experiment_id {cfg['experiment_id']}"
strategies = [s for s in cfg["strategies"] if s.get("enabled") and s["id"] != "cash_buffer"]
assert len(strategies) == 7, f"expected 7 active sleeves, got {len(strategies)}"
total_w = sum(s["weight"] for s in cfg["strategies"])
assert 0.99 < total_w < 1.01, f"weights sum to {total_w:.3f}"
print(f"  [cfg]  7 active sleeves, weights sum={total_w:.3f}")
print(f"  [cfg]  sleeves: {[s['id'] for s in strategies]}")
PY

    "$PYTHON" - <<'PY'
import importlib, sys
mods = [
    "compass.tail_risk_hedge",
    "compass.exp1770_commodity_calendars",
    "compass.crisis_alpha_v5",
    "compass.exp1750_putcall_overlay",
    "compass.risk_overlay",
    "compass.portfolio_risk_manager",
    "compass.telegram_alerter",
]
ok = True
for m in mods:
    try:
        importlib.import_module(m)
        print(f"  [mod] ✓ {m}")
    except Exception as e:
        print(f"  [mod] ✗ {m}: {e}")
        ok = False
sys.exit(0 if ok else 1)
PY

    require_env ALPACA_API_KEY_PAPER ALPACA_API_SECRET_PAPER || \
        echo "[launch] note: Alpaca vars missing (ok for smoke)"
    require_env TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID || \
        echo "[launch] note: Telegram vars missing (will log-fallback)"
    echo "[launch] smoke OK"
}

cmd_dry() {
    require_env ALPACA_API_KEY_PAPER ALPACA_API_SECRET_PAPER
    echo "[launch] DRY-RUN: scan loop, no order submission"
    "$PYTHON" -m compass.paper_engine --config "$CONFIG" --dry-run --once \
        2>&1 | tee -a "$ENGINE_LOG"
}

cmd_start() {
    require_env ALPACA_API_KEY_PAPER ALPACA_API_SECRET_PAPER
    require_env TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID
    echo "[launch] FOREGROUND paper engine (Ctrl-C to stop)"
    exec "$PYTHON" -m compass.paper_engine --config "$CONFIG"
}

cmd_daemon() {
    if [ -f "$ENGINE_PID" ] && kill -0 "$(cat "$ENGINE_PID")" 2>/dev/null; then
        echo "[launch] engine already running (pid $(cat "$ENGINE_PID"))"
        exit 0
    fi
    require_env ALPACA_API_KEY_PAPER ALPACA_API_SECRET_PAPER
    require_env TELEGRAM_BOT_TOKEN TELEGRAM_CHAT_ID

    echo "[launch] starting paper engine (daemon)"
    nohup "$PYTHON" -m compass.paper_engine --config "$CONFIG" \
        >> "$ENGINE_LOG" 2>&1 &
    echo $! > "$ENGINE_PID"
    sleep 1

    echo "[launch] starting health monitor"
    nohup "$PYTHON" "$SCRIPT_DIR/north_star_v6_monitor.py" \
        --config "$CONFIG" --interval 300 \
        >> "$MONITOR_LOG" 2>&1 &
    echo $! > "$MONITOR_PID"

    echo "[launch] engine pid=$(cat "$ENGINE_PID") monitor pid=$(cat "$MONITOR_PID")"
    echo "[launch] tail -F $ENGINE_LOG $MONITOR_LOG"
}

cmd_stop() {
    for f in "$ENGINE_PID" "$MONITOR_PID"; do
        if [ -f "$f" ]; then
            pid=$(cat "$f")
            if kill -0 "$pid" 2>/dev/null; then
                echo "[launch] stopping pid $pid"
                kill "$pid" || true
                sleep 1
                kill -0 "$pid" 2>/dev/null && kill -9 "$pid" || true
            fi
            rm -f "$f"
        fi
    done
    echo "[launch] stopped"
}

cmd_status() {
    echo "[launch] north_star_v6 status:"
    for name in engine monitor; do
        f="$ENGINE_PID"
        [ "$name" = "monitor" ] && f="$MONITOR_PID"
        if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
            echo "  $name: RUNNING (pid $(cat "$f"))"
        else
            echo "  $name: STOPPED"
        fi
    done
    if [ -f "$HEALTH_FILE" ]; then
        echo "  health: $HEALTH_FILE"
        "$PYTHON" - <<PY
import json
d = json.load(open("$HEALTH_FILE"))
print(f"    equity      : \${d.get('equity', 0):,.0f}")
print(f"    pnl_total   : \${d.get('pnl_total', 0):+,.0f}  "
      f"({d.get('pnl_total_pct', 0):+.2f}%)")
print(f"    open_posns  : {d.get('open_positions', 0)}")
print(f"    breaches    : {len(d.get('breaches', []))}")
for s in d.get('sleeves', []):
    print(f"    {s['id']:28s} "
          f"posns={s.get('n_positions', 0):>2}  "
          f"pnl_today=\${s.get('pnl_today', 0):+,.0f}")
PY
    fi
}

cmd_logs() {
    tail -F "$ENGINE_LOG" "$MONITOR_LOG" 2>/dev/null
}

cmd_report() {
    echo "[launch] running daily P&L report"
    "$PYTHON" "$SCRIPT_DIR/north_star_v6_daily_report.py" \
        --config "$CONFIG" 2>&1 | tee -a "$REPORT_LOG"
}

cmd_close_all() {
    echo "[launch] FLATTEN: closing every open position (confirm with CLOSE_ALL=1)"
    if [ "${CLOSE_ALL:-}" != "1" ]; then
        echo "  dry-run (set CLOSE_ALL=1 to actually flatten)"
        "$PYTHON" -m compass.paper_engine --config "$CONFIG" --close-all --dry-run
    else
        "$PYTHON" -m compass.paper_engine --config "$CONFIG" --close-all
    fi
}

cmd_install_launchd() {
    PLIST="$HOME/Library/LaunchAgents/com.pilotai.northstar.v6.plist"
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.pilotai.northstar.v6</string>
  <key>ProgramArguments</key>
  <array>
    <string>$SCRIPT_DIR/launch_north_star_v6.sh</string>
    <string>daemon</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key>
  <dict><key>SuccessfulExit</key><false/></dict>
  <key>StandardOutPath</key><string>$REPO_ROOT/$ENGINE_LOG</string>
  <key>StandardErrorPath</key><string>$REPO_ROOT/$ENGINE_LOG</string>
</dict>
</plist>
PLIST
    echo "[launch] wrote $PLIST"
    echo "        load with: launchctl load -w $PLIST"
}

case "${1:-smoke}" in
    smoke)            cmd_smoke ;;
    dry)              cmd_dry ;;
    start)            cmd_start ;;
    daemon)           cmd_daemon ;;
    stop)             cmd_stop ;;
    status)           cmd_status ;;
    logs)             cmd_logs ;;
    report)           cmd_report ;;
    close-all)        cmd_close_all ;;
    install-launchd)  cmd_install_launchd ;;
    *)
        echo "usage: $0 {smoke|dry|start|daemon|stop|status|logs|report|close-all|install-launchd}"
        exit 2 ;;
esac
