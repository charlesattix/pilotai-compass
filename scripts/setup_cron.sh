#!/usr/bin/env bash
# ============================================================================
# setup_cron.sh — Install crontab entry for daily SPY options data update
# ============================================================================
#
# Schedules daily_data_update.sh to run at 6 PM ET (22:00 UTC) on weekdays.
# This is ~1 hour after US market close, ensuring all daily bars are settled.
#
# Usage:
#   ./scripts/setup_cron.sh           # install cron entry
#   ./scripts/setup_cron.sh --remove  # remove cron entry
#   ./scripts/setup_cron.sh --status  # show current cron entries
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
UPDATE_SCRIPT="$PROJECT_DIR/scripts/daily_data_update.sh"

# Cron schedule: 22:00 UTC Mon-Fri (= 6 PM ET / 5 PM CT)
CRON_SCHEDULE="0 22 * * 1-5"
CRON_COMMENT="# pilotai-credit-spreads: daily SPY options data update"
CRON_ENTRY="$CRON_SCHEDULE $UPDATE_SCRIPT >> $PROJECT_DIR/data/daily_update_cron.log 2>&1"

ACTION="${1:-install}"

case "$ACTION" in
    --status|-s)
        echo "Current crontab entries matching pilotai:"
        crontab -l 2>/dev/null | grep -i "pilotai" || echo "  (none found)"
        echo
        echo "Full crontab:"
        crontab -l 2>/dev/null || echo "  (no crontab)"
        ;;

    --remove|-r)
        if crontab -l 2>/dev/null | grep -q "daily_data_update"; then
            crontab -l 2>/dev/null | grep -v "daily_data_update" | grep -v "$CRON_COMMENT" | crontab -
            echo "Removed daily_data_update cron entry."
        else
            echo "No daily_data_update cron entry found."
        fi
        ;;

    install|--install|-i|"")
        # Verify the script exists and is executable
        if [ ! -x "$UPDATE_SCRIPT" ]; then
            echo "ERROR: $UPDATE_SCRIPT is not executable."
            echo "Run: chmod +x $UPDATE_SCRIPT"
            exit 1
        fi

        # Verify .env exists
        if [ ! -f "$PROJECT_DIR/.env" ]; then
            echo "ERROR: $PROJECT_DIR/.env not found."
            echo "Create it with: POLYGON_API_KEY=your_key"
            exit 1
        fi

        # Check if already installed
        if crontab -l 2>/dev/null | grep -q "daily_data_update"; then
            echo "Cron entry already exists. Updating..."
            # Remove old entry first
            crontab -l 2>/dev/null | grep -v "daily_data_update" | grep -v "$CRON_COMMENT" | crontab -
        fi

        # Install new entry
        (
            crontab -l 2>/dev/null || true
            echo ""
            echo "$CRON_COMMENT"
            echo "$CRON_ENTRY"
        ) | crontab -

        echo "Cron entry installed:"
        echo "  Schedule: $CRON_SCHEDULE (22:00 UTC / 6 PM ET, Mon-Fri)"
        echo "  Command:  $UPDATE_SCRIPT"
        echo "  Log:      $PROJECT_DIR/data/daily_update_cron.log"
        echo
        echo "Verify with: crontab -l"
        echo "Remove with: $0 --remove"
        echo
        echo "Test now with: $UPDATE_SCRIPT --dry-run"
        ;;

    *)
        echo "Usage: $0 [install|--remove|--status]"
        exit 1
        ;;
esac
