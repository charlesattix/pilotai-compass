#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# PilotAI Signal Service — Cron wrapper
#
# Cron schedule (9:35 AM ET Mon–Fri):
#   35 9 * * 1-5 /Users/charlesbot/projects/pilotai-credit-spreads/scripts/run_signal_service.sh
#
# Optional afternoon digest (4:00 PM ET):
#   0 16 * * 1-5 .../run_signal_service.sh --digest-only
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/signal_service_$(date +%Y%m).log"

mkdir -p "$LOG_DIR"

# ── Load environment ──────────────────────────────────────────────────────────
# Source .env if it exists (local development)
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Required env vars — fail fast if missing
: "${TELEGRAM_BOT_TOKEN:?TELEGRAM_BOT_TOKEN is required}"
: "${TELEGRAM_CHAT_ID:?TELEGRAM_CHAT_ID is required}"

export PILOTAI_API_KEY="${PILOTAI_API_KEY:-cZZP6he1Qez8Lb6njh6w5vUe}"
export PILOTAI_DB_PATH="${PILOTAI_DB_PATH:-$PROJECT_ROOT/data/pilotai_signal.db}"

# ── Activate virtualenv if present ───────────────────────────────────────────
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$PROJECT_ROOT/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# ── Parse arguments ───────────────────────────────────────────────────────────
DIGEST_ONLY=false
DRY_RUN=false
FORCE=false

for arg in "$@"; do
    case "$arg" in
        --digest-only) DIGEST_ONLY=true ;;
        --dry-run)     DRY_RUN=true ;;
        --force)       FORCE=true ;;
    esac
done

# ── Log helper ────────────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "=== Signal Service starting (digest_only=$DIGEST_ONLY dry_run=$DRY_RUN) ==="

cd "$PROJECT_ROOT"

if [ "$DIGEST_ONLY" = true ]; then
    CMD="python -m pilotai_signal digest"
    [ "$DRY_RUN" = true ] && CMD="$CMD --dry-run"
    log "Running: $CMD"
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
else
    CMD="python -m pilotai_signal run"
    [ "$FORCE" = true ]    && CMD="$CMD --force"
    [ "$DRY_RUN" = true ]  && CMD="$CMD --dry-run"
    log "Running: $CMD"
    eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
fi

EXIT_CODE=${PIPESTATUS[0]}
log "=== Signal Service complete (exit=$EXIT_CODE) ==="
exit $EXIT_CODE
