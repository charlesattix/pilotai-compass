# Paper Trading Deployment — Ultimate Portfolio

**Last Updated:** 2026-04-05
**Status:** NOT DEPLOYED — all infrastructure built, waiting on operations

---

## TL;DR for Charles

Run one command to deploy everything:

```bash
./scripts/deploy_paper_trading.sh
```

This will: create directories, check env files, initialize databases, install cron,
run pre-flight checks, and start the first scanner cycle.

**Before running, you MUST:**
1. Create `.env.ultimate_v4` (template below)
2. Verify `.env.exp400` and `.env.exp401` have current Alpaca keys
3. Close or document the 30 orphan positions in Alpaca accounts

---

## 1. Current State (Honest Assessment)

### What Works
| Component | Tests | Status |
|-----------|-------|--------|
| Paper Trading Engine v4 | 55 pass | Feature-complete |
| Core Paper Engine | 57 pass | Feature-complete |
| Production Monitor | 87 pass | Dashboard + alerts |
| Risk Management | 71 pass | Circuit breakers, kill switch |
| Paper Reconciler | 11 pass | 6-dimension comparison |
| Telegram Alerting | 3 pass | Trade/Risk/Model/Kill alerts |
| Data Pipeline | ✓ | daily_data_update.sh ready |
| **Total** | **284 tests** | **Code is ready** |

### What's Broken
| Issue | Severity | Blocker? |
|-------|----------|----------|
| **Cron not installed** | CRITICAL | YES — scanners never run |
| **EXP-503/600 no env files** | CRITICAL | YES — can't start |
| **30 orphan positions** | HIGH | YES — position limits saturated |
| **Data staleness (GLD/QQQ/TLT)** | MEDIUM | Blocks multi-ticker pairs |
| **deploy.sh hardcoded paths** | MEDIUM | Won't run on non-macOS |

### Validation Clock
```
Mar 15 ─── EXP-400/401 launched
           ↓ (20 days, 2 trades — should have been 80-120)
Apr 05 ─── TODAY (36% elapsed)
           ↓ (36 days remaining)
May 11 ─── 8-week mark (decision point)
```

**With immediate action, 30+ days of real data remain — enough for validation.**

---

## 2. What Exactly Needs to Happen

### Step 1: Environment Files (Charles, 5 minutes)

Create `.env.ultimate_v4` in project root:

```bash
# .env.ultimate_v4 — Ultimate Portfolio Paper Trading
ALPACA_API_KEY=<your-paper-api-key>
ALPACA_API_SECRET=<your-paper-api-secret>
ALPACA_BASE_URL=https://paper-api.alpaca.markets
POLYGON_API_KEY=<your-polygon-key>

# Optional: Telegram alerts
TELEGRAM_BOT_TOKEN=<bot-token>
TELEGRAM_CHAT_ID=<chat-id>
```

**Also verify** `.env.exp400` and `.env.exp401` have current keys. If stale, regenerate in Alpaca dashboard.

### Step 2: Orphan Position Cleanup (Charles, 30 minutes)

Check both accounts for mystery positions:

```bash
# List all positions in each account
python3 -c "
from alpaca.trading.client import TradingClient
import os
for env in ['.env.exp400', '.env.exp401']:
    # Load keys from env file
    print(f'--- {env} ---')
    # Check positions via Alpaca API
"
```

Options:
- **Close all orphans** — safest, clears position limits
- **Document as manual trades** — if intentional, mark in reconciler
- **Filter in scanner** — add orphan exclusion (already built in paper engine)

### Step 3: Run Deploy Script (Charles, 1 command)

```bash
chmod +x scripts/deploy_paper_trading.sh
./scripts/deploy_paper_trading.sh
```

This handles everything else automatically (see Section 4).

### Step 4: Verify (Charles, 5 minutes next trading day)

```bash
# Check logs
tail -20 logs/scan-ultimate-v4.log

# Check trades
python3 -c "
import sqlite3
conn = sqlite3.connect('data/ultimate_v4/paper.db')
cur = conn.cursor()
cur.execute('SELECT COUNT(*) FROM trades')
print(f'Trades: {cur.fetchone()[0]}')
"
```

---

## 3. Infrastructure Inventory

### Existing (Ready to Use)
```
configs/paper_ultimate_v4.yaml     ← Portfolio config (298 lines)
compass/paper_trading_v4.py        ← 5-strategy engine (632 lines)
compass/paper_trading_engine.py    ← Core engine (876 lines)
compass/production_monitor.py      ← Health dashboard (37K)
compass/paper_reconciler.py        ← Backtest/paper comparison (44K)
compass/live_bridge.py             ← Signal → order translation (631 lines)
shared/telegram_alerts.py          ← Alert types
alerts/telegram_bot.py             ← Bot gateway
scripts/daily_data_update.sh       ← Polygon data refresh
scripts/setup_cron.sh              ← Cron installer
```

### Missing (Created by Deploy Script)
```
data/ultimate_v4/                  ← Database directory
data/ultimate_v4/paper.db          ← Trade/position database
logs/                              ← Log directory
.env.ultimate_v4                   ← Credentials (Charles creates)
```

### Charles Must Provide
```
✗ Alpaca API key + secret (paper trading account)
✗ Polygon API key (if not already in .env.exp400)
✗ Telegram bot token + chat ID (optional but recommended)
✗ Decision: which portfolio variant to trade
    Option A: Ultimate 1.6x (101.6% CAGR, 11.4% DD)
    Option B: Adaptive + Hedge (102% CAGR, 7.5% DD)
    Option C: Hedged v3 (82.4% CAGR, 6.4% DD)
    Option D: SPY-Only high-cap (27.9% CAGR, 4.1% DD, $1B+ cap)
```

---

## 4. Deploy Script Details

The `scripts/deploy_paper_trading.sh` script does:

```
Phase 1: Pre-flight checks
  ✓ Python 3.8+ available
  ✓ Required packages (numpy, pandas, yfinance)
  ✓ options_cache.db exists and >900MB
  ✓ .env.ultimate_v4 exists (created by Charles)
  ✓ configs/paper_ultimate_v4.yaml exists

Phase 2: Directory setup
  ✓ Create data/ultimate_v4/
  ✓ Create logs/
  ✓ Set permissions

Phase 3: Database initialization
  ✓ Initialize paper.db with schema
  ✓ Verify IronVault connection

Phase 4: Data freshness check
  ✓ Check last date in options_cache.db
  ✓ Run daily_data_update.sh if stale

Phase 5: Cron installation
  ✓ Add scanner cron (every 30 min during market hours)
  ✓ Add data refresh cron (daily at 16:30 ET)
  ✓ Add health check cron (daily at 08:00 ET)

Phase 6: First scan
  ✓ Run one scanner cycle to verify it works
  ✓ Check logs for errors

Phase 7: Status report
  ✓ Print summary of what was deployed
  ✓ Print next steps
```

---

## 5. Monitoring After Deployment

### Daily Checks
```bash
# Check scanner ran
tail -5 logs/scan-ultimate-v4.log

# Check for new trades
sqlite3 data/ultimate_v4/paper.db "SELECT * FROM trades ORDER BY entry_date DESC LIMIT 5"

# Check health dashboard
open reports/production_dashboard.html
```

### Weekly Checks
```bash
# Run reconciler (compare paper vs backtest)
python3 -c "from compass.paper_reconciler import run; run()"

# Check data freshness
python3 -c "
import sqlite3
conn = sqlite3.connect('data/options_cache.db')
cur = conn.cursor()
cur.execute('SELECT MAX(date) FROM daily_bars WHERE ticker=\"SPY\"')
print(f'Last SPY data: {cur.fetchone()[0]}')
"
```

### Alert Thresholds
| Alert | Threshold | Action |
|-------|-----------|--------|
| Kill switch | DD > 20% | Auto-flatten all positions |
| Risk alert | DD > 10% | Telegram notification |
| Stale data | >2 days | Telegram warning |
| Scanner failure | 3 consecutive | Telegram critical |

---

## 6. Rollback Plan

If anything goes wrong:

```bash
# Stop all scanners
crontab -r  # removes all cron jobs

# Or just remove the paper trading cron entries
crontab -l | grep -v "scan-ultimate" | crontab -

# Flatten positions (if needed)
python3 -c "
# This closes all paper positions in the account
print('Manual: go to Alpaca dashboard and close all positions')
"
```

---

## 7. Expected Results

After successful deployment, expect:

| Metric | First Week | 4 Weeks | 8 Weeks |
|--------|-----------|---------|---------|
| Trades | 3-5 | 15-25 | 30-50 |
| Signals generated | 20-30 | 80-120 | 160-240 |
| Positions open | 2-4 | 3-6 | 3-6 |
| P&L data points | 5-7 | 20-28 | 40-56 |

**Target performance (from backtest):**
- CAGR: 80-100% annualized
- Max DD: <12%
- Win rate: 80-90%
- Sharpe: 4-9 (depending on variant)
