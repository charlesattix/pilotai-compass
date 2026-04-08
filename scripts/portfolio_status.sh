#!/bin/bash
# portfolio_status.sh — Bulletproof credential & portfolio discovery
#
# Auto-discovers all .env.exp* files, validates Alpaca + Polygon connectivity,
# and prints a complete account/position summary for every experiment.
#
# Usage:  bash scripts/portfolio_status.sh
# Exit:   0 if all accounts reachable, 1 if any fail
#
# IMPORTANT — env var names vs HTTP header names:
#   Env files use:  ALPACA_API_KEY  and  ALPACA_API_SECRET
#   Alpaca headers: APCA-API-KEY-ID and  APCA-API-SECRET-KEY  (different!)

set -uo pipefail

ALPACA_BASE="https://paper-api.alpaca.markets"
POLYGON_BASE="https://api.polygon.io"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT=15

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
info() { echo -e "  ${CYAN}→${NC} $*"; }

_get_key() {
  # Extract KEY=value from an env file, stripping comments and quotes
  local file="$1" varname="$2"
  grep -E "^${varname}=" "$file" 2>/dev/null \
    | head -1 \
    | sed 's/^[^=]*=//; s/^["'"'"']//; s/["'"'"']$//'
}

_alpaca() {
  local endpoint="$1" key="$2" secret="$3"
  curl -sf --max-time "$TIMEOUT" \
    -H "APCA-API-KEY-ID: ${key}" \
    -H "APCA-API-SECRET-KEY: ${secret}" \
    "${ALPACA_BASE}${endpoint}" 2>/dev/null
}

_check_polygon() {
  local key="$1"
  if [ -z "$key" ]; then fail "POLYGON_API_KEY not set"; return 1; fi
  local resp
  resp=$(curl -sf --max-time "$TIMEOUT" \
    "${POLYGON_BASE}/v2/aggs/ticker/SPY/range/1/day/2024-01-02/2024-01-02?apiKey=${key}" \
    2>/dev/null || echo "")
  if echo "$resp" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); exit(0 if d.get('resultsCount',0)>0 else 1)" 2>/dev/null; then
    ok "Polygon key valid  (key=${key:0:8}...)"
  else
    fail "Polygon INVALID/unreachable  (key=${key:0:8}...)"
    return 1
  fi
}

_check_account() {
  local env_file="$1"
  # Derive experiment name: .env.exp036 → exp036
  local exp
  exp=$(python3 -c "
import re, sys
name = sys.argv[1]
m = re.search(r'\.env\.(.+)$', name)
print(m.group(1) if m else name)
" "$(basename "$env_file")")

  echo ""
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
  echo -e "${BOLD}  Experiment: ${CYAN}${exp}${NC}   ${env_file}"
  echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

  local api_key api_secret polygon_key
  api_key=$(_get_key "$env_file" "ALPACA_API_KEY")
  api_secret=$(_get_key "$env_file" "ALPACA_API_SECRET")
  polygon_key=$(_get_key "$env_file" "POLYGON_API_KEY")

  if [ -z "$api_key" ] || [ -z "$api_secret" ]; then
    fail "ALPACA_API_KEY or ALPACA_API_SECRET missing in ${env_file}"
    return 1
  fi
  info "Key: ${api_key:0:8}...  (${#api_key} chars)"

  local acct
  acct=$(_alpaca "/v2/account" "$api_key" "$api_secret")
  if [ -z "$acct" ]; then
    fail "Alpaca API unreachable or auth failed  (key=${api_key:0:8}...)"
    return 1
  fi

  # Parse account fields with python3
  python3 - <<PYEOF
import json, sys
d = json.loads("""${acct}""")
acct_num = d.get('account_number', '?')
status   = d.get('status', '?')
equity   = float(d.get('equity', 0) or 0)
cash     = float(d.get('cash', 0) or 0)
bp       = float(d.get('buying_power', 0) or 0)
upl      = float(d.get('unrealized_pl', 0) or 0)
sign     = '+' if upl >= 0 else ''
print(f"  \033[32m✓\033[0m Account {acct_num}  status={status}")
print(f"     Equity:         \${equity:>12,.2f}")
print(f"     Cash:           \${cash:>12,.2f}")
print(f"     Buying Power:   \${bp:>12,.2f}")
print(f"     Unrealized P&L: {sign}\${upl:>11,.2f}")
PYEOF

  local positions
  positions=$(_alpaca "/v2/positions" "$api_key" "$api_secret")
  if [ -n "$positions" ]; then
    python3 - <<PYEOF
import json, sys
positions = json.loads("""${positions}""")
if not positions:
    print("     Positions:  none")
else:
    print(f"     Positions ({len(positions)}):")
    for p in positions:
        sym  = p.get('symbol','?')
        side = p.get('side','?')
        qty  = p.get('qty','?')
        mv   = float(p.get('market_value', 0) or 0)
        upl  = float(p.get('unrealized_pl', 0) or 0)
        s    = '+' if upl >= 0 else ''
        print(f"       {sym:<28} {side:<5} qty={qty:<6} mv=\${mv:>10,.2f}  uPnL={s}\${upl:>9,.2f}")
PYEOF
  fi

  if [ -n "$polygon_key" ]; then
    _check_polygon "$polygon_key" || true
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║    PilotAI Portfolio Status — $(date '+%Y-%m-%d %H:%M %Z')   ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"

# Auto-discover .env.exp* — exclude .env.example, .env.local, plain .env
mapfile -d '' env_files < <(find "$PROJECT_DIR" -maxdepth 1 -name '.env.exp*' -print0 2>/dev/null | sort -z)

if [ ${#env_files[@]} -eq 0 ]; then
  fail "No .env.exp* files found in ${PROJECT_DIR}"
  echo "  Expected: .env.exp036  .env.exp059  .env.exp154  .env.exp305"
  exit 1
fi

info "Discovered ${#env_files[@]} experiment env files"

FAILED=0
for f in "${env_files[@]}"; do
  _check_account "$f" || FAILED=$((FAILED + 1))
done

# Also check shared Polygon key from root .env
SHARED_ENV="${PROJECT_DIR}/.env"
if [ -f "$SHARED_ENV" ]; then
  POLY=$(_get_key "$SHARED_ENV" "POLYGON_API_KEY")
  if [ -n "$POLY" ]; then
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Shared: Polygon API${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    _check_polygon "$POLY" || FAILED=$((FAILED + 1))
  fi
fi

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ "$FAILED" -eq 0 ]; then
  echo -e "  ${GREEN}${BOLD}All accounts OK${NC}"
  exit 0
else
  echo -e "  ${RED}${BOLD}${FAILED} account(s) FAILED — check credentials/connectivity${NC}"
  exit 1
fi
