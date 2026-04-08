#!/usr/bin/env bash
# check_keys.sh — Minimal key validation. One line per account.
# Run this BEFORE ever reporting key issues to Carlos.
# Exit 0 = all keys work. Exit 1 = at least one failed.
set -uo pipefail

cd "$(dirname "$0")/.."
python3 -c "
from shared.credentials import get_all_portfolios, check_portfolio
ok = True
for name, info in get_all_portfolios().items():
    if not info['has_keys']:
        print(f'{name}: NO_KEYS (no api key in {info[\"env_file\"]})')
        continue
    r = check_portfolio(info['env_file'])
    if r['status'] == 'OK':
        print(f'{name}: ✅ OK — {r[\"account_number\"]} — \${r[\"equity\"]:,.2f}')
    else:
        print(f'{name}: ❌ {r[\"status\"]} — {r.get(\"error\",\"\")[:60]}')
        ok = False
exit(0 if ok else 1)
"
