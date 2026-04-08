#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# launch_exp1220.sh — One-command launcher for EXP-1220 paper trading
# ═══════════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/launch_exp1220.sh smoke      # smoke test (no trades)
#   ./scripts/launch_exp1220.sh dry        # dry-run scan (no submissions)
#   ./scripts/launch_exp1220.sh scan       # live paper scan (submits orders)
#   ./scripts/launch_exp1220.sh status     # show positions
#   ./scripts/launch_exp1220.sh health     # write health file
#   ./scripts/launch_exp1220.sh close-all  # close all open positions
#   ./scripts/launch_exp1220.sh install    # install LaunchAgent (macOS)
#   ./scripts/launch_exp1220.sh uninstall  # remove LaunchAgent (macOS)
#
# Loads environment from .env (if present) and runs the scanner.
# Exits non-zero on any error.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ─── Load environment ───────────────────────────────────────────────────────
ENV_FILE=".env"
if [ -f ".env.exp1220" ]; then
    ENV_FILE=".env.exp1220"
fi

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
    echo "[launch] Loaded environment from $ENV_FILE"
else
    echo "[launch] WARNING: No $ENV_FILE found. Copy .env.exp1220.example to .env and fill in credentials."
fi

# ─── Prerequisite check ─────────────────────────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
    echo "[launch] ERROR: python3 not found on PATH"
    exit 1
fi

# Check required packages
python3 -c "import yaml" 2>/dev/null || {
    echo "[launch] ERROR: pyyaml not installed. Run: pip3 install pyyaml"
    exit 1
}

# alpaca-py is only required for live scans (not dry-run or smoke-test)
check_alpaca() {
    python3 -c "import alpaca" 2>/dev/null || {
        echo "[launch] ERROR: alpaca-py not installed. Run: pip3 install alpaca-py"
        exit 1
    }
}

# ─── Dispatch ───────────────────────────────────────────────────────────────
CMD="${1:-smoke}"

case "$CMD" in
    smoke|smoke-test)
        echo "[launch] Running smoke test..."
        python3 scripts/run_exp1220.py --smoke-test
        ;;
    dry|dry-run)
        echo "[launch] Running dry-run scan (no orders submitted)..."
        python3 scripts/run_exp1220.py --dry-run --force-scan
        ;;
    scan|live|paper)
        check_alpaca
        if [ -z "${ALPACA_API_KEY:-}" ] || [ -z "${ALPACA_SECRET_KEY:-}" ]; then
            echo "[launch] ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"
            echo "[launch]        Copy .env.exp1220.example to .env and fill in keys"
            exit 1
        fi
        echo "[launch] Running live paper scan..."
        python3 scripts/run_exp1220.py
        ;;
    force-scan)
        check_alpaca
        echo "[launch] Running FORCED paper scan..."
        python3 scripts/run_exp1220.py --force-scan
        ;;
    status)
        python3 scripts/run_exp1220.py --status
        ;;
    health)
        python3 scripts/run_exp1220.py --health
        ;;
    close-all)
        check_alpaca
        echo "[launch] WARNING: This will close ALL open positions."
        read -rp "Type 'CLOSE' to confirm: " confirm
        if [ "$confirm" != "CLOSE" ]; then
            echo "[launch] Aborted."
            exit 1
        fi
        python3 scripts/run_exp1220.py --close-all
        ;;
    install)
        if [[ "$OSTYPE" != "darwin"* ]]; then
            echo "[launch] ERROR: LaunchAgent install is macOS only"
            exit 1
        fi
        echo "[launch] Installing LaunchAgent..."

        # Check plist path matches actual repo
        PLIST="deploy/com.pilotai.exp1220.plist"
        if grep -q "/Users/charles/pilotai" "$PLIST" && [ "$REPO_ROOT" != "/Users/charles/pilotai" ]; then
            echo "[launch] ERROR: plist has hard-coded /Users/charles/pilotai"
            echo "[launch]        Edit $PLIST and replace with actual path: $REPO_ROOT"
            exit 1
        fi

        DEST="$HOME/Library/LaunchAgents/com.pilotai.exp1220.plist"
        mkdir -p "$HOME/Library/LaunchAgents"
        cp "$PLIST" "$DEST"
        chmod 644 "$DEST"
        launchctl unload "$DEST" 2>/dev/null || true
        launchctl load "$DEST"
        echo "[launch] OK — LaunchAgent installed at $DEST"
        echo "[launch] Verify with: launchctl list | grep pilotai"
        echo "[launch] Manual trigger: launchctl start com.pilotai.exp1220"
        ;;
    uninstall)
        if [[ "$OSTYPE" != "darwin"* ]]; then
            echo "[launch] ERROR: LaunchAgent uninstall is macOS only"
            exit 1
        fi
        DEST="$HOME/Library/LaunchAgents/com.pilotai.exp1220.plist"
        launchctl unload "$DEST" 2>/dev/null || true
        rm -f "$DEST"
        echo "[launch] OK — LaunchAgent removed"
        ;;
    logs)
        tail -f logs/exp1220.log
        ;;
    help|--help|-h)
        head -30 "$0" | grep -E "^#"
        ;;
    *)
        echo "[launch] Unknown command: $CMD"
        echo "[launch] Run '$0 help' for usage"
        exit 1
        ;;
esac
