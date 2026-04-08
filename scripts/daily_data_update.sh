#!/usr/bin/env bash
# ============================================================================
# daily_data_update.sh — Daily SPY Options Data Update
# ============================================================================
#
# Fetches new SPY options data from Polygon and validates the cache DB.
# Designed to be idempotent and safe to run via cron.
#
# Steps:
#   1. Check if today is a US trading day (skip weekends & holidays)
#   2. Source .env for POLYGON_API_KEY
#   3. Acquire lock to prevent concurrent runs
#   4. Run backfill_polygon_cache.py to fetch new data (with retries)
#   5. Run iron_vault_setup.py to validate the DB
#   6. Log all output with timestamps to data/daily_update.log
#   7. Rotate logs if > 10 MB
#
# Crontab entry (6 PM ET = 22:00 UTC on trading days):
#   0 22 * * 1-5 /path/to/pilotai-credit-spreads/scripts/daily_data_update.sh
#
# Usage:
#   ./scripts/daily_data_update.sh              # normal run
#   ./scripts/daily_data_update.sh --dry-run    # discovery only, no fetching
#   ./scripts/daily_data_update.sh --force      # skip trading-day check
#
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOCK_FILE="$PROJECT_DIR/data/.daily_update.lock"
LOG_FILE="$PROJECT_DIR/data/daily_update.log"
MAX_LOG_SIZE=$((10 * 1024 * 1024))  # 10 MB
MAX_RETRIES=3
RETRY_DELAY=60  # seconds between retries

DRY_RUN=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --force)   FORCE=true ;;
    esac
done

# ── Logging ────────────────────────────────────────────────────────────────

log() {
    local msg="[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_error() {
    log "ERROR: $1"
}

# ── Log rotation ──────────────────────────────────────────────────────────

rotate_log() {
    if [ -f "$LOG_FILE" ]; then
        local size
        size=$(stat -c%s "$LOG_FILE" 2>/dev/null || stat -f%z "$LOG_FILE" 2>/dev/null || echo 0)
        if [ "$size" -gt "$MAX_LOG_SIZE" ]; then
            mv "$LOG_FILE" "${LOG_FILE}.1"
            log "Log rotated (was ${size} bytes)"
        fi
    fi
}

# ── Trading day check ────────────────────────────────────────────────────
# Skips weekends.  Also skips major US market holidays (NYSE closures).
# The --force flag bypasses this check.

is_trading_day() {
    local dow
    dow=$(date -u '+%u')  # 1=Monday ... 7=Sunday

    # Skip weekends
    if [ "$dow" -ge 6 ]; then
        return 1
    fi

    # Check major US market holidays (approximate — covers most closures).
    # Format: MM-DD for fixed dates, or python check for floating ones.
    local today
    today=$(date -u '+%m-%d')

    # Fixed holidays: New Year's Day, Independence Day, Christmas
    case "$today" in
        01-01|07-04|12-25)
            return 1
            ;;
    esac

    # Floating holidays are harder to check in pure bash.  For production
    # accuracy, defer to a Python helper that checks a calendar.  For now,
    # the cron schedule (Mon-Fri) + fixed holidays covers ~95% of cases.
    # The script is idempotent, so running on a half-day or holiday just
    # means no new data is fetched (Polygon returns nothing new).

    return 0
}

# ── Lock (idempotency) ────────────────────────────────────────────────────

cleanup() {
    rm -f "$LOCK_FILE"
}

if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Another daily_data_update is already running (PID $OLD_PID). Exiting."
        exit 0
    fi
    rm -f "$LOCK_FILE"
fi

echo $$ > "$LOCK_FILE"
trap cleanup EXIT

# ── Setup ──────────────────────────────────────────────────────────────────

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Rotate log if needed
rotate_log

log "=========================================="
log "Daily Data Update — START"
log "=========================================="

# ── Trading day gate ─────────────────────────────────────────────────────

if [ "$FORCE" = false ] && [ "$DRY_RUN" = false ]; then
    if ! is_trading_day; then
        log "Not a trading day (weekend or holiday). Skipping. Use --force to override."
        log "=========================================="
        log "Daily Data Update — SKIPPED (non-trading day)"
        log "=========================================="
        exit 0
    fi
fi

# ── Source environment ─────────────────────────────────────────────────────

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$PROJECT_DIR/.env"
    set +a
    log "Sourced .env"
else
    log_error ".env file not found at $PROJECT_DIR/.env"
    exit 1
fi

if [ -z "${POLYGON_API_KEY:-}" ]; then
    log_error "POLYGON_API_KEY is not set in .env"
    exit 1
fi

log "POLYGON_API_KEY is set (${POLYGON_API_KEY:0:4}...)"

# ── Step 1: Backfill Polygon Cache (with retries) ────────────────────────

BACKFILL_ARGS="--workers 4"
if [ "$DRY_RUN" = true ]; then
    BACKFILL_ARGS="$BACKFILL_ARGS --dry-run"
fi

log "Step 1: Fetching new SPY options data (backfill_polygon_cache.py $BACKFILL_ARGS)"

backfill_success=false
for attempt in $(seq 1 "$MAX_RETRIES"); do
    if python3 "$PROJECT_DIR/scripts/backfill_polygon_cache.py" $BACKFILL_ARGS >> "$LOG_FILE" 2>&1; then
        log "Step 1: PASSED — backfill completed successfully (attempt $attempt/$MAX_RETRIES)"
        backfill_success=true
        break
    else
        BACKFILL_EXIT=$?
        if [ "$attempt" -lt "$MAX_RETRIES" ]; then
            log "Step 1: FAILED (attempt $attempt/$MAX_RETRIES, exit code $BACKFILL_EXIT) — retrying in ${RETRY_DELAY}s..."
            sleep "$RETRY_DELAY"
        else
            log_error "Step 1: FAILED after $MAX_RETRIES attempts (exit code $BACKFILL_EXIT)"
        fi
    fi
done

if [ "$backfill_success" = false ]; then
    log_error "Backfill exhausted all retries. Check data/daily_update.log for details."
    # Continue to validation — partial data may still be useful
fi

# ── Step 2: Validate Iron Vault DB ───────────────────────────────────────

log "Step 2: Validating options_cache.db (iron_vault_setup.py)"

if python3 "$PROJECT_DIR/scripts/iron_vault_setup.py" >> "$LOG_FILE" 2>&1; then
    log "Step 2: PASSED — Iron Vault validation successful"
else
    VAULT_EXIT=$?
    log_error "Step 2: FAILED — Iron Vault validation exited with code $VAULT_EXIT"
fi

# ── Step 3: Report DB size ───────────────────────────────────────────────

DB_FILE="$PROJECT_DIR/data/options_cache.db"
if [ -f "$DB_FILE" ]; then
    DB_SIZE=$(du -h "$DB_FILE" | cut -f1)
    log "DB size: $DB_SIZE ($DB_FILE)"
fi

# ── Done ──────────────────────────────────────────────────────────────────

if [ "$backfill_success" = true ]; then
    log "=========================================="
    log "Daily Data Update — COMPLETE"
    log "=========================================="
    exit 0
else
    log "=========================================="
    log "Daily Data Update — COMPLETE WITH ERRORS"
    log "=========================================="
    exit 1
fi
