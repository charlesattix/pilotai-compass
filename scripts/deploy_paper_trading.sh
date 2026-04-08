#!/bin/bash
# ============================================================================
# deploy_paper_trading.sh — One-Command Paper Trading Deployment
# ============================================================================
# Deploys the Ultimate Portfolio paper trading system.
# Handles: directory setup, DB init, data checks, cron install, first scan.
#
# Prerequisites (Charles must do BEFORE running this):
#   1. Create .env.ultimate_v4 with Alpaca + Polygon keys
#   2. Clean orphan positions in existing Alpaca accounts
#
# Usage:
#   chmod +x scripts/deploy_paper_trading.sh
#   ./scripts/deploy_paper_trading.sh
# ============================================================================

set -euo pipefail

# Auto-detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
DATA_DIR="${PROJECT_DIR}/data"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }
info() { echo -e "  ${BLUE}→${NC} $1"; }

ERRORS=0

echo ""
echo "============================================================"
echo "  PilotAI — Paper Trading Deployment"
echo "  Ultimate Portfolio (4 strategies at adaptive leverage)"
echo "============================================================"
echo ""

# ── Phase 1: Pre-flight checks ──────────────────────────────────────────

echo "Phase 1: Pre-flight checks"

# Python
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1)
    ok "Python: $PY_VER"
else
    fail "Python3 not found"
    ERRORS=$((ERRORS + 1))
fi

# Required packages
for pkg in numpy pandas; do
    if python3 -c "import $pkg" 2>/dev/null; then
        ok "Package: $pkg"
    else
        fail "Package missing: $pkg (pip install $pkg)"
        ERRORS=$((ERRORS + 1))
    fi
done

# Options cache DB
DB_PATH="${DATA_DIR}/options_cache.db"
if [ -f "$DB_PATH" ]; then
    DB_SIZE=$(stat -c%s "$DB_PATH" 2>/dev/null || stat -f%z "$DB_PATH" 2>/dev/null || echo 0)
    DB_MB=$((DB_SIZE / 1048576))
    if [ "$DB_MB" -gt 900 ]; then
        ok "IronVault DB: ${DB_MB}MB"
    else
        warn "IronVault DB small: ${DB_MB}MB (expected >900MB)"
    fi
else
    fail "IronVault DB not found at $DB_PATH"
    ERRORS=$((ERRORS + 1))
fi

# Config file
CONFIG="${PROJECT_DIR}/configs/paper_ultimate_v4.yaml"
if [ -f "$CONFIG" ]; then
    ok "Config: paper_ultimate_v4.yaml"
else
    fail "Config missing: configs/paper_ultimate_v4.yaml"
    ERRORS=$((ERRORS + 1))
fi

# Env file
ENV_FILE="${PROJECT_DIR}/.env.ultimate_v4"
if [ -f "$ENV_FILE" ]; then
    ok "Env file: .env.ultimate_v4"
    # Check for required keys
    for key in ALPACA_API_KEY ALPACA_API_SECRET POLYGON_API_KEY; do
        if grep -q "^${key}=" "$ENV_FILE" 2>/dev/null; then
            ok "  $key present"
        else
            fail "  $key missing in .env.ultimate_v4"
            ERRORS=$((ERRORS + 1))
        fi
    done
else
    fail "Env file missing: .env.ultimate_v4"
    echo ""
    echo "  Create it with:"
    echo "    cat > .env.ultimate_v4 << 'EOF'"
    echo "    ALPACA_API_KEY=<your-paper-api-key>"
    echo "    ALPACA_API_SECRET=<your-paper-api-secret>"
    echo "    ALPACA_BASE_URL=https://paper-api.alpaca.markets"
    echo "    POLYGON_API_KEY=<your-polygon-key>"
    echo "    EOF"
    echo ""
    ERRORS=$((ERRORS + 1))
fi

# Check existing experiment env files
for envf in .env.exp400 .env.exp401; do
    if [ -f "${PROJECT_DIR}/${envf}" ]; then
        ok "Existing env: $envf"
    else
        warn "Missing env: $envf (existing experiments won't run)"
    fi
done

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    fail "Pre-flight failed with $ERRORS errors. Fix the above issues and re-run."
    exit 1
fi

echo ""
ok "All pre-flight checks passed"

# ── Phase 2: Directory setup ────────────────────────────────────────────

echo ""
echo "Phase 2: Directory setup"

mkdir -p "$LOG_DIR"
ok "Logs: $LOG_DIR"

mkdir -p "${DATA_DIR}/ultimate_v4"
ok "Data: ${DATA_DIR}/ultimate_v4"

# ── Phase 3: Database initialization ────────────────────────────────────

echo ""
echo "Phase 3: Database initialization"

PAPER_DB="${DATA_DIR}/ultimate_v4/paper.db"
if [ -f "$PAPER_DB" ]; then
    ok "Database exists: $PAPER_DB"
else
    python3 -c "
import sqlite3
conn = sqlite3.connect('${PAPER_DB}')
cur = conn.cursor()
cur.execute('''CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    ticker TEXT NOT NULL,
    direction TEXT,
    entry_date TEXT,
    exit_date TEXT,
    entry_price REAL,
    exit_price REAL,
    contracts INTEGER,
    pnl REAL,
    status TEXT DEFAULT 'open',
    metadata TEXT
)''')
cur.execute('''CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    equity REAL,
    cash REAL,
    positions_value REAL
)''')
cur.execute('''CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    strategy TEXT,
    signal_type TEXT,
    details TEXT
)''')
conn.commit()
conn.close()
print('Database initialized successfully')
" 2>&1 && ok "Database initialized: $PAPER_DB" || fail "Database init failed"
fi

# ── Phase 4: Data freshness check ───────────────────────────────────────

echo ""
echo "Phase 4: Data freshness check"

LAST_DATE=$(python3 -c "
import sqlite3
conn = sqlite3.connect('${DB_PATH}')
cur = conn.cursor()
cur.execute('SELECT MAX(date) FROM daily_bars WHERE ticker=\"SPY\"')
row = cur.fetchone()
print(row[0] if row and row[0] else 'unknown')
conn.close()
" 2>/dev/null || echo "unknown")

info "Last SPY data: $LAST_DATE"

# Check if data is more than 3 days old
if [ "$LAST_DATE" != "unknown" ]; then
    DAYS_OLD=$(python3 -c "
from datetime import datetime
try:
    last = datetime.strptime('${LAST_DATE}', '%Y-%m-%d')
    diff = (datetime.now() - last).days
    print(diff)
except:
    print(999)
" 2>/dev/null || echo 999)

    if [ "$DAYS_OLD" -gt 3 ]; then
        warn "Data is ${DAYS_OLD} days old — running update"
        if [ -x "${SCRIPT_DIR}/daily_data_update.sh" ]; then
            info "Running daily_data_update.sh..."
            bash "${SCRIPT_DIR}/daily_data_update.sh" >> "${LOG_DIR}/data_update.log" 2>&1 && \
                ok "Data updated" || warn "Data update had issues (check logs/data_update.log)"
        else
            warn "daily_data_update.sh not executable — skipping"
        fi
    else
        ok "Data is ${DAYS_OLD} days old (fresh enough)"
    fi
fi

# ── Phase 5: Cron installation ──────────────────────────────────────────

echo ""
echo "Phase 5: Cron installation"

# Build cron entries
SCAN_SCRIPT="${SCRIPT_DIR}/run_ultimate_scan.sh"

# Create the scanner wrapper script
cat > "$SCAN_SCRIPT" << SCANEOF
#!/bin/bash
# Auto-generated scanner for Ultimate Portfolio paper trading
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR}"
LOG_FILE="${LOG_DIR}/scan-ultimate-v4.log"

# Skip weekends
DOW=\$(TZ=America/New_York date +%u)
if [ "\$DOW" -gt 5 ]; then
    exit 0
fi

cd "\$PROJECT_DIR"

echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting Ultimate v4 scan" >> "\$LOG_FILE"

# Source env
set -a
source .env.ultimate_v4
set +a

# Run scanner
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from compass.paper_trading_v4 import PaperTradingV4
    engine = PaperTradingV4('configs/paper_ultimate_v4.yaml')
    engine.run_scan_cycle()
    print('Scan complete')
except Exception as e:
    print(f'Scan error: {e}', file=sys.stderr)
" >> "\$LOG_FILE" 2>&1

echo "[\$(date -u +%Y-%m-%dT%H:%M:%SZ)] Scan finished" >> "\$LOG_FILE"
SCANEOF

chmod +x "$SCAN_SCRIPT"
ok "Scanner script: $SCAN_SCRIPT"

# Check if cron is available
if command -v crontab &>/dev/null; then
    # Build crontab entries
    CRON_ENTRIES=$(cat << CRONEOF
# PilotAI Paper Trading — Ultimate Portfolio v4
# Scanner: every 30 min during market hours (Mon-Fri, 9:30-16:00 ET)
*/30 9-15 * * 1-5 ${SCAN_SCRIPT} >> ${LOG_DIR}/cron.log 2>&1
# Data refresh: daily at 16:30 ET (after market close)
30 16 * * 1-5 cd ${PROJECT_DIR} && bash scripts/daily_data_update.sh >> ${LOG_DIR}/data_update.log 2>&1
CRONEOF
)

    # Check if already installed
    EXISTING=$(crontab -l 2>/dev/null || true)
    if echo "$EXISTING" | grep -q "ultimate_v4\|run_ultimate_scan"; then
        ok "Cron entries already installed"
    else
        # Append to existing crontab
        (echo "$EXISTING"; echo ""; echo "$CRON_ENTRIES") | crontab -
        ok "Cron entries installed"
    fi

    info "Cron schedule:"
    info "  Scanner: every 30 min, Mon-Fri 9:30-15:30 ET"
    info "  Data refresh: daily at 16:30 ET"
else
    warn "crontab not available — set up manually or use systemd timer"
fi

# ── Phase 6: First scan (smoke test) ────────────────────────────────────

echo ""
echo "Phase 6: Smoke test"

info "Running first scanner cycle..."
bash "$SCAN_SCRIPT" 2>&1 && ok "First scan completed" || warn "First scan had issues (check logs)"

# ── Phase 7: Status report ──────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  DEPLOYMENT COMPLETE"
echo "============================================================"
echo ""
ok "Ultimate Portfolio v4 paper trading deployed"
echo ""
info "Directories:"
info "  Config:   $CONFIG"
info "  Database: $PAPER_DB"
info "  Logs:     $LOG_DIR"
info "  Scanner:  $SCAN_SCRIPT"
echo ""
info "Next steps:"
info "  1. Wait for next market day"
info "  2. Check logs: tail -20 ${LOG_DIR}/scan-ultimate-v4.log"
info "  3. Check trades: sqlite3 ${PAPER_DB} 'SELECT * FROM trades'"
info "  4. Monitor: open reports/production_dashboard.html"
echo ""
info "To stop paper trading:"
info "  crontab -l | grep -v ultimate | crontab -"
echo ""
