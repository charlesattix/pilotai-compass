#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# launch_exp2520.sh — EXP-2520 Paper-Trading Deployment Launcher
# ═══════════════════════════════════════════════════════════════════════════
#
# Final paper-trading deployment for the 7-stream portfolio on Mac Studio.
#
# Commands
#   ./scripts/launch_exp2520.sh smoke          env + import + config validation
#   ./scripts/launch_exp2520.sh dry            scan loop, NO order submit
#   ./scripts/launch_exp2520.sh start          foreground paper engine
#   ./scripts/launch_exp2520.sh daemon         background engine + monitor + dashboard
#   ./scripts/launch_exp2520.sh stop           stop all running components
#   ./scripts/launch_exp2520.sh restart        stop + daemon
#   ./scripts/launch_exp2520.sh status         engine/monitor/dashboard summary
#   ./scripts/launch_exp2520.sh logs           tail engine + monitor logs
#   ./scripts/launch_exp2520.sh report         run the daily P&L report now
#   ./scripts/launch_exp2520.sh dashboard      run the risk dashboard once
#   ./scripts/launch_exp2520.sh close-all      flatten every open sleeve
#   ./scripts/launch_exp2520.sh install-launchd  macOS LaunchAgent (auto-start)
#
# Loads .env or .env.exp2520. Writes pid/log files under logs/exp2520/.
# Fails fast if required env vars are missing.
#
# References
#   configs/exp2410_production_paper.yaml   — upstream config (EXP-2410)
#   scripts/exp2520_monitor.py              — 5-min health poller
#   scripts/exp2520_risk_dashboard.py       — live risk dashboard
#   scripts/exp2520_daily_report.py         — end-of-day P&L + attribution
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/exp2410_production_paper.yaml"
LOG_DIR="logs/exp2520"
ENGINE_PID="$LOG_DIR/engine.pid"
MONITOR_PID="$LOG_DIR/monitor.pid"
DASHBOARD_PID="$LOG_DIR/dashboard.pid"
ENGINE_LOG="$LOG_DIR/engine.log"
MONITOR_LOG="$LOG_DIR/monitor.log"
DASHBOARD_LOG="$LOG_DIR/dashboard.log"
HEALTH_JSON="$LOG_DIR/health.json"
STATE_JSON="$LOG_DIR/state.json"

mkdir -p "$LOG_DIR"

# ── Env loading ───────────────────────────────────────────────────────────
load_env() {
    local loaded=0
    if [ -f .env.exp2520 ]; then
        set -o allexport; source .env.exp2520; set +o allexport; loaded=1
    elif [ -f .env ]; then
        set -o allexport; source .env; set +o allexport; loaded=1
    fi
    if [ "$loaded" -eq 0 ]; then
        echo "WARN: no .env or .env.exp2520 file found — relying on inherited env." >&2
    fi
}

require_env() {
    local var="$1"
    if [ -z "${!var:-}" ]; then
        echo "ERROR: env var $var is not set. See .env.exp2520.example" >&2
        exit 2
    fi
}

check_required_env() {
    require_env ALPACA_API_KEY_PAPER
    require_env ALPACA_API_SECRET_PAPER
    # Telegram is optional but highly recommended
    if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
        echo "WARN: TELEGRAM_BOT_TOKEN not set — alerts will log-only." >&2
    fi
}

# ── Command: smoke ────────────────────────────────────────────────────────
cmd_smoke() {
    load_env
    echo "[smoke] config: $CONFIG"
    test -f "$CONFIG" || { echo "ERROR: config not found"; exit 3; }

    check_required_env
    echo "[smoke] env OK"

    python3 - <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open("configs/exp2410_production_paper.yaml"))
assert cfg["experiment_id"] == "EXP-2410", f"unexpected experiment_id {cfg['experiment_id']}"
assert cfg["mode"] == "paper", f"mode must be paper, got {cfg['mode']}"
# check every strategy module is importable
sys.path.insert(0, str(Path.cwd()))
for s in cfg["strategies"]:
    if not s.get("enabled", True): continue
    mod = s.get("module")
    if not mod or mod == "passive": continue
    try:
        __import__(mod)
        print(f"  ✓ {s['id']:25s}  {mod}")
    except Exception as e:
        print(f"  ✗ {s['id']:25s}  {mod}  ← {type(e).__name__}: {e}")
        sys.exit(4)
# validate allocator ref
alloc_mod = cfg["portfolio"]["allocator"]["module"]
__import__(alloc_mod)
print(f"  ✓ allocator               {alloc_mod}")
# validate risk manager
rm_mod = cfg["risk_manager"]["module"]
try:
    __import__(rm_mod)
    print(f"  ✓ risk_manager            {rm_mod}")
except Exception as e:
    print(f"  ! risk_manager            {rm_mod}  ({type(e).__name__}) — may resolve at runtime")
# honest targets echo
t = cfg["targets"]["honest_walk_forward"]
print(f"  honest targets: Sharpe {t['pooled_sharpe']}, CAGR {t['pooled_cagr_pct']}%, DD {t['pooled_max_dd_pct']}%")
print("  smoke OK")
PY
    echo "[smoke] done"
}

# ── Command: dry ──────────────────────────────────────────────────────────
cmd_dry() {
    load_env; check_required_env
    echo "[dry] scanning with DRY_RUN=1 — no orders will be submitted"
    DRY_RUN=1 python3 -m compass.paper_engine \
        --config "$CONFIG" \
        --dry-run \
        --log-file "$LOG_DIR/dry.log"
}

# ── Command: start (foreground) ───────────────────────────────────────────
cmd_start() {
    load_env; check_required_env
    echo "[start] foreground paper engine (Ctrl-C to stop)"
    python3 -m compass.paper_engine \
        --config "$CONFIG" \
        --log-file "$ENGINE_LOG" \
        --state-file "$STATE_JSON"
}

# ── Command: daemon (background engine + monitor + dashboard) ────────────
cmd_daemon() {
    load_env; check_required_env

    if [ -f "$ENGINE_PID" ] && kill -0 "$(cat "$ENGINE_PID")" 2>/dev/null; then
        echo "[daemon] engine already running (pid $(cat "$ENGINE_PID"))"
        return 0
    fi

    echo "[daemon] starting engine …"
    nohup python3 -m compass.paper_engine \
        --config "$CONFIG" \
        --log-file "$ENGINE_LOG" \
        --state-file "$STATE_JSON" \
        >"$ENGINE_LOG" 2>&1 &
    echo $! > "$ENGINE_PID"
    sleep 1

    echo "[daemon] starting monitor …"
    nohup python3 scripts/exp2520_monitor.py \
        --config "$CONFIG" \
        --log-file "$MONITOR_LOG" \
        --health-file "$HEALTH_JSON" \
        >"$MONITOR_LOG" 2>&1 &
    echo $! > "$MONITOR_PID"

    echo "[daemon] starting risk dashboard refresher …"
    nohup python3 scripts/exp2520_risk_dashboard.py \
        --config "$CONFIG" \
        --loop \
        --log-file "$DASHBOARD_LOG" \
        >"$DASHBOARD_LOG" 2>&1 &
    echo $! > "$DASHBOARD_PID"

    echo "[daemon] all three components started:"
    echo "         engine    pid $(cat "$ENGINE_PID")    log $ENGINE_LOG"
    echo "         monitor   pid $(cat "$MONITOR_PID")   log $MONITOR_LOG"
    echo "         dashboard pid $(cat "$DASHBOARD_PID") log $DASHBOARD_LOG"
}

# ── Command: stop ─────────────────────────────────────────────────────────
_stop_one() {
    local name="$1" pid_file="$2"
    if [ -f "$pid_file" ]; then
        local pid; pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" && echo "[stop] $name (pid $pid) stopped"
            for _ in 1 2 3 4 5; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" && echo "[stop] $name force-killed"
            fi
        fi
        rm -f "$pid_file"
    else
        echo "[stop] $name not running"
    fi
}

cmd_stop() {
    _stop_one engine    "$ENGINE_PID"
    _stop_one monitor   "$MONITOR_PID"
    _stop_one dashboard "$DASHBOARD_PID"
}

cmd_restart() {
    cmd_stop
    sleep 1
    cmd_daemon
}

# ── Command: status ──────────────────────────────────────────────────────
cmd_status() {
    echo "── EXP-2520 deployment status ──────────────────────────────────"
    for pair in "engine:$ENGINE_PID" "monitor:$MONITOR_PID" "dashboard:$DASHBOARD_PID"; do
        name="${pair%%:*}"; pidf="${pair##*:}"
        if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
            echo "  $name:    RUNNING (pid $(cat "$pidf"))"
        else
            echo "  $name:    stopped"
        fi
    done
    if [ -f "$HEALTH_JSON" ]; then
        echo
        echo "── Latest health.json ──"
        python3 -c "import json; d=json.load(open('$HEALTH_JSON'))
for k in ('last_poll','equity','leverage','trailing_dd_pct','circuit_breaker_state','alert_count_24h'):
    print(f'  {k:22s} {d.get(k)}')"
    fi
}

# ── Command: logs ────────────────────────────────────────────────────────
cmd_logs() {
    echo "── tailing engine + monitor + dashboard (Ctrl-C to exit) ──"
    tail -F "$ENGINE_LOG" "$MONITOR_LOG" "$DASHBOARD_LOG" 2>/dev/null || true
}

# ── Command: report ──────────────────────────────────────────────────────
cmd_report() {
    load_env
    python3 scripts/exp2520_daily_report.py --config "$CONFIG"
}

# ── Command: dashboard ───────────────────────────────────────────────────
cmd_dashboard() {
    load_env
    python3 scripts/exp2520_risk_dashboard.py --config "$CONFIG" --once
}

# ── Command: close-all ───────────────────────────────────────────────────
cmd_close_all() {
    load_env; check_required_env
    echo "[close-all] flattening every open sleeve via engine API"
    python3 -m compass.paper_engine --config "$CONFIG" --close-all
}

# ── Command: install-launchd (macOS) ─────────────────────────────────────
cmd_install_launchd() {
    local plist="$HOME/Library/LaunchAgents/com.pilotai.exp2520.plist"
    cat > "$plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.pilotai.exp2520</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$REPO_ROOT/scripts/launch_exp2520.sh</string>
    <string>daemon</string>
  </array>
  <key>WorkingDirectory</key><string>$REPO_ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$REPO_ROOT/$LOG_DIR/launchd.stdout.log</string>
  <key>StandardErrorPath</key><string>$REPO_ROOT/$LOG_DIR/launchd.stderr.log</string>
</dict></plist>
PLIST
    launchctl unload "$plist" 2>/dev/null || true
    launchctl load "$plist"
    echo "[install-launchd] installed $plist"
}

# ── Dispatch ─────────────────────────────────────────────────────────────
case "${1:-help}" in
    smoke)           cmd_smoke ;;
    dry)             cmd_dry ;;
    start)           cmd_start ;;
    daemon)          cmd_daemon ;;
    stop)            cmd_stop ;;
    restart)         cmd_restart ;;
    status)          cmd_status ;;
    logs)            cmd_logs ;;
    report)          cmd_report ;;
    dashboard)       cmd_dashboard ;;
    close-all)       cmd_close_all ;;
    install-launchd) cmd_install_launchd ;;
    *)
        grep -E '^\s*\./scripts/launch_exp2520\.sh' "$0" | sed 's/^#\s\?//'
        exit 1
        ;;
esac
