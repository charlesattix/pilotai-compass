#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# deploy_paper.sh — One-command paper trading launch
# EXP-1220 at 1.5× static leverage, 7-day cadence
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_DIR/configs/deploy_exp1220_1.5x.yaml"
LOG_DIR="$PROJECT_DIR/logs"
PID_FILE="$LOG_DIR/paper_trading.pid"

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

usage() {
    echo "Usage: $0 {start|stop|status|check}"
    echo ""
    echo "  start   — Launch paper trading daemon"
    echo "  stop    — Stop the daemon"
    echo "  status  — Show current positions and P&L"
    echo "  check   — Verify environment and config"
    exit 1
}

check_env() {
    echo -e "${YELLOW}Checking environment...${NC}"
    local ok=true

    # Python
    if ! command -v python3 &>/dev/null; then
        echo -e "  ${RED}✗ python3 not found${NC}"; ok=false
    else
        echo -e "  ${GREEN}✓ python3: $(python3 --version)${NC}"
    fi

    # Required packages
    for pkg in numpy pandas; do
        if python3 -c "import $pkg" 2>/dev/null; then
            echo -e "  ${GREEN}✓ $pkg installed${NC}"
        else
            echo -e "  ${RED}✗ $pkg missing — run: pip3 install $pkg${NC}"; ok=false
        fi
    done

    # Alpaca credentials
    if [[ -z "${ALPACA_API_KEY:-}" ]]; then
        echo -e "  ${RED}✗ ALPACA_API_KEY not set${NC}"
        echo -e "    Export it: export ALPACA_API_KEY='your-key-here'"
        ok=false
    else
        echo -e "  ${GREEN}✓ ALPACA_API_KEY set (${ALPACA_API_KEY:0:8}...)${NC}"
    fi

    if [[ -z "${ALPACA_SECRET_KEY:-}" ]]; then
        echo -e "  ${RED}✗ ALPACA_SECRET_KEY not set${NC}"
        echo -e "    Export it: export ALPACA_SECRET_KEY='your-secret-here'"
        ok=false
    else
        echo -e "  ${GREEN}✓ ALPACA_SECRET_KEY set${NC}"
    fi

    # Config file
    if [[ -f "$CONFIG" ]]; then
        echo -e "  ${GREEN}✓ Config: $CONFIG${NC}"
    else
        echo -e "  ${RED}✗ Config not found: $CONFIG${NC}"; ok=false
    fi

    # Log directory
    mkdir -p "$LOG_DIR"
    echo -e "  ${GREEN}✓ Log dir: $LOG_DIR${NC}"

    if $ok; then
        echo -e "\n${GREEN}All checks passed. Ready to deploy.${NC}"
    else
        echo -e "\n${RED}Fix the above issues before deploying.${NC}"
        exit 1
    fi
}

start_trading() {
    check_env

    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo -e "${YELLOW}Paper trading is already running (PID $(cat "$PID_FILE"))${NC}"
        exit 1
    fi

    echo -e "\n${GREEN}Starting EXP-1220 paper trading...${NC}"
    echo -e "  Config:   $CONFIG"
    echo -e "  Leverage: 1.5× static"
    echo -e "  Cadence:  7-day (Monday scan)"
    echo -e "  Hedge:    None"
    echo -e "  Log:      $LOG_DIR/paper_trading.log"

    # Launch the standalone EXP-1220 scanner
    cd "$PROJECT_DIR"
    PYTHONPATH="$PROJECT_DIR" nohup python3 scripts/run_exp1220.py \
        >> "$LOG_DIR/paper_trading.log" 2>&1 &

    echo $! > "$PID_FILE"
    echo -e "${GREEN}Started with PID $(cat "$PID_FILE")${NC}"
    echo -e "  Tail log: tail -f $LOG_DIR/paper_trading.log"
}

stop_trading() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}Stopping paper trading (PID $PID)...${NC}"
            kill "$PID"
            rm -f "$PID_FILE"
            echo -e "${GREEN}Stopped.${NC}"
        else
            echo -e "${YELLOW}Process $PID not running. Cleaning up.${NC}"
            rm -f "$PID_FILE"
        fi
    else
        echo -e "${YELLOW}No PID file found. Paper trading not running.${NC}"
    fi
}

show_status() {
    echo -e "${YELLOW}Paper Trading Status${NC}"
    echo "─────────────────────────────────"

    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo -e "  Status: ${GREEN}RUNNING${NC} (PID $(cat "$PID_FILE"))"
    else
        echo -e "  Status: ${RED}STOPPED${NC}"
    fi

    # Show recent log entries
    if [[ -f "$LOG_DIR/paper_trading.log" ]]; then
        echo ""
        echo "  Last 10 log entries:"
        tail -10 "$LOG_DIR/paper_trading.log" | sed 's/^/    /'
    fi

    # Show trade journal if exists
    if [[ -f "$LOG_DIR/trade_journal.csv" ]]; then
        echo ""
        echo "  Recent trades:"
        tail -5 "$LOG_DIR/trade_journal.csv" | sed 's/^/    /'
    fi
}

case "${1:-}" in
    start)  start_trading ;;
    stop)   stop_trading ;;
    status) show_status ;;
    check)  check_env ;;
    *)      usage ;;
esac
