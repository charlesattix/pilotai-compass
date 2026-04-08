#!/bin/bash
# =============================================================================
# sync_dashboard_data.sh — Cron wrapper for dashboard data sync
# =============================================================================
#
# Exports all experiment DBs to data/dashboard_export.json and pushes to Railway.
# Designed to run every 5 minutes during market hours.
#
# SETUP:
#   1. Create .env.sync in the project root with:
#        RAILWAY_URL=https://attix-credit-spreads-production.up.railway.app
#        RAILWAY_ADMIN_TOKEN=your_token_here
#
#   2. Install the cron job (edit with: crontab -e):
#        */5 9-16 * * 1-5 /Users/charlesbot/projects/attix-credit-spreads/scripts/sync_dashboard_data.sh >> ~/logs/sync_dashboard.log 2>&1
#
#      Breakdown: every 5 min, hours 9-16, Mon-Fri
#      Note: 9-16 covers 9:00-16:59 → catches pre-market open (9:30) through
#            after-close (4 PM). Adjust to 9-17 if you want 5 PM coverage.
#
#   3. Make executable:
#        chmod +x scripts/sync_dashboard_data.sh
#
#   4. Test manually:
#        scripts/sync_dashboard_data.sh
#
# LOG: ~/logs/sync_dashboard.log (rotate with logrotate or manually)
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/sync_dashboard_data.py"
LOG_PREFIX="[sync_dashboard $(date '+%Y-%m-%d %H:%M:%S')]"

# ── Python interpreter: prefer venv, fall back to system python3 ─────────────

VENV_PYTHON="$PROJECT_DIR/.venv/bin/python3"
if [ -f "$VENV_PYTHON" ]; then
    PYTHON="$VENV_PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "$LOG_PREFIX ERROR: python3 not found" >&2
    exit 1
fi

# ── Check script exists ───────────────────────────────────────────────────────

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "$LOG_PREFIX ERROR: sync script not found at $PYTHON_SCRIPT" >&2
    exit 1
fi

# ── Market hours guard (optional — redundant with cron schedule but explicit) ─
# Uncomment to enforce strictly within the script itself.
#
# DAY_OF_WEEK=$(date '+%u')   # 1=Mon … 7=Sun
# HOUR=$(date '+%-H')
# if [ "$DAY_OF_WEEK" -gt 5 ] || [ "$HOUR" -lt 9 ] || [ "$HOUR" -gt 16 ]; then
#     echo "$LOG_PREFIX Outside market hours — skipping"
#     exit 0
# fi

# ── Run ──────────────────────────────────────────────────────────────────────

echo "$LOG_PREFIX Starting sync..."

cd "$PROJECT_DIR"

# Decide whether to push based on env var (set RAILWAY_PUSH=false to disable)
PUSH_FLAG=""
if [ "${RAILWAY_PUSH:-true}" = "true" ]; then
    PUSH_FLAG="--push"
fi

"$PYTHON" "$PYTHON_SCRIPT" $PUSH_FLAG --quiet
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "$LOG_PREFIX Done."
elif [ $EXIT_CODE -eq 2 ]; then
    # Push failed but local export succeeded — not fatal for monitoring
    echo "$LOG_PREFIX WARNING: Local export OK but Railway push failed (exit 2)." >&2
    exit 2
else
    echo "$LOG_PREFIX ERROR: Sync failed (exit $EXIT_CODE)." >&2
    exit $EXIT_CODE
fi
