#!/bin/bash
# Sync local scanner trades to Railway via the /api/import-trades endpoint.
# This endpoint supports full field control (status, pnl, exit_date etc).
#
# Usage:
#   bash scripts/sync_trades_to_railway.sh
#
# Requires: the /api/import-trades endpoint to be deployed on Railway.

set -euo pipefail

BASE_URL="${RAILWAY_URL:-https://pilotai-credit-spreads-production.up.railway.app}"
DB_PATH="${PILOTAI_DB:-data/pilotai.db}"

echo "Extracting trades from local database..."

# Extract trades as JSON from local SQLite
TRADES_JSON=$(sqlite3 "$DB_PATH" -json "
  SELECT id, ticker, strategy_type, status,
         short_strike, long_strike, expiration,
         credit, contracts, entry_date, exit_date,
         CASE WHEN exit_reason IS NULL THEN 'stop_loss' ELSE exit_reason END as exit_reason,
         pnl, metadata
  FROM trades
  WHERE source='scanner'
  ORDER BY created_at DESC
")

COUNT=$(echo "$TRADES_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "Found ${COUNT} scanner trades"

# Transform to the import format
IMPORT_PAYLOAD=$(python3 -c "
import json, sys

trades = json.loads('''${TRADES_JSON}''')
import_trades = []
for t in trades:
    meta = json.loads(t.get('metadata', '{}') or '{}')
    import_trades.append({
        'id': t['id'],
        'ticker': t['ticker'],
        'strategy_type': t.get('strategy_type', ''),
        'status': t.get('status', 'open'),
        'short_strike': t.get('short_strike', 0),
        'long_strike': t.get('long_strike', 0),
        'expiration': t.get('expiration', ''),
        'credit': t.get('credit', 0),
        'contracts': t.get('contracts', 1),
        'entry_date': t.get('entry_date', ''),
        'exit_date': t.get('exit_date'),
        'exit_reason': t.get('exit_reason'),
        'pnl': t.get('pnl'),
        'metadata': meta,
    })

print(json.dumps({'trades': import_trades, 'clear_existing': True}))
")

echo "Sending ${COUNT} trades to Railway..."
RESP=$(curl -s -X POST "${BASE_URL}/api/import-trades" \
  -H "Content-Type: application/json" \
  -d "$IMPORT_PAYLOAD")

echo "Response: ${RESP}"

# Verify
echo ""
echo "--- Verifying ---"
curl -s "${BASE_URL}/api/paper-trades" | python3 -c "
import sys, json
data = json.load(sys.stdin)
trades = data.get('trades', [])
stats = data.get('stats', {})
print(f'Total: {stats.get(\"total_trades\", len(trades))}  Open: {stats.get(\"open_trades\", 0)}  Closed: {stats.get(\"closed_trades\", 0)}')
print(f'Realized PnL: \${stats.get(\"total_realized_pnl\", 0):,.2f}')
for t in trades:
    rpnl = t.get('realized_pnl') or t.get('unrealized_pnl') or 0
    print(f'  {t[\"ticker\"]:4s} {t.get(\"short_strike\",0)}/{t.get(\"long_strike\",0)} x{t.get(\"contracts\",1)} {t.get(\"status\",\"?\")} pnl=\${rpnl:,.2f}')
"
