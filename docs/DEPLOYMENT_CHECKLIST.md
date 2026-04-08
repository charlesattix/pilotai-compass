# Deployment Checklist — Paper Trading Launch

**For:** EXP-880 (Crisis Hedge V2) and EXP-1470 (North Star Combined)
**Operator:** Carlos / Charles
**Last Updated:** 2026-04-03

---

## Phase 0: Before You Start

- [ ] Read this entire checklist once before doing anything
- [ ] Confirm the Mac Studio is powered on and connected to internet
- [ ] Confirm you have admin access to the Mac Studio terminal
- [ ] Confirm you have Alpaca paper trading account credentials
- [ ] Block 2 hours for initial setup (one-time)

---

## Phase 1: Environment Setup (One-Time)

### 1.1 Create Environment File

Create a file called `.env.exp880` in the project root:

```
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_API_SECRET=your_paper_api_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
POLYGON_API_KEY=your_polygon_key_here
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
EXPERIMENT_ID=EXP-880
```

**Where to get these:**
- **Alpaca:** Log in to https://app.alpaca.markets → Paper Trading → API Keys
- **Polygon:** https://polygon.io/dashboard → API Keys (free tier is fine)
- **Telegram:** Talk to @BotFather on Telegram → /newbot → copy token. Send a message to the bot, then visit `https://api.telegram.org/bot<TOKEN>/getUpdates` to get your chat_id

### 1.2 Verify Python Environment

Open terminal on Mac Studio and run:
```bash
cd /path/to/pilotai-credit-spreads
python3 --version   # should be 3.11+
pip3 install -r requirements.txt  # if not already done
```

### 1.3 Create Data Directories

```bash
mkdir -p data/exp880 output logs
```

---

## Phase 2: Config Verification

### 2.1 Check Config File Exists

```bash
ls configs/paper_exp880.yaml
```

If missing, something is wrong — do NOT proceed. Contact engineering.

### 2.2 Verify Config Contents

Open `configs/paper_exp880.yaml` and confirm:
- [ ] `paper_mode: true` (MUST be true)
- [ ] `tickers: [SPY]`
- [ ] `strategy.min_dte: 15`
- [ ] `strategy.max_dte: 25`
- [ ] Risk section has `max_drawdown: 0.12` (12% kill switch)
- [ ] Leverage section has `base_leverage: 2.0`

### 2.3 Verify API Keys Work

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv('.env.exp880')
import os, requests
r = requests.get('https://paper-api.alpaca.markets/v2/account',
    headers={'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
             'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']})
print(f'Status: {r.status_code}')
if r.status_code == 200:
    d = r.json()
    print(f'Account: {d[\"account_number\"]}')
    print(f'Equity: \${float(d[\"equity\"]):,.2f}')
    print(f'Buying Power: \${float(d[\"buying_power\"]):,.2f}')
    print('API OK')
else:
    print(f'ERROR: {r.text}')
"
```

Expected: `API OK` with equity showing. If error, double-check API keys.

---

## Phase 3: Pre-Flight Checks

### 3.1 Run Deployment Validator

```bash
python3 -c "
from compass.deployment_validator import DeploymentValidator
dv = DeploymentValidator()
report = dv.validate(skip_alpaca=False)
for c in report.checks:
    icon = 'PASS' if c.passed else 'FAIL'
    print(f'  [{icon}] {c.name}: {c.detail}')
print(f'\nResult: {\"ALL PASSED\" if report.all_passed else \"FAILED — DO NOT PROCEED\"}\')
"
```

**All checks must pass before proceeding.** If any fail:
- `module_imports` fail → Python packages missing, run `pip3 install -r requirements.txt`
- `config_schema` fail → Config file corrupted, restore from git
- `alpaca_connectivity` fail → API keys wrong, check `.env.exp880`
- `model_files` fail → Model not trained, contact engineering
- `directory_permissions` fail → Run `mkdir -p data/exp880 output logs`

### 3.2 Run Dry Trade Test

```bash
python3 -c "
from compass.deployment_validator import DeploymentValidator
dv = DeploymentValidator()
result = dv.check_dry_trade()
print(f'Dry trade: {\"PASS\" if result.passed else \"FAIL\"} — {result.detail}')
"
```

Must show `PASS`. This confirms the signal → risk check → order pipeline works.

---

## Phase 4: Launch

### 4.1 Start Paper Trading

```bash
python3 main.py scheduler --config configs/paper_exp880.yaml --env-file .env.exp880
```

Or if using LaunchAgent (Mac Studio auto-start):
```bash
# Load the LaunchAgent
launchctl load ~/Library/LaunchAgents/com.pilotai.exp880.plist
```

### 4.2 Verify It's Running

Wait 2 minutes, then check:
```bash
# Check process
ps aux | grep "main.py"

# Check logs
tail -20 logs/exp880.log

# Check Telegram — you should have received a startup message
```

Expected in logs: `[INFO] Scheduler started for EXP-880` and `[INFO] Next signal check at ...`

### 4.3 Confirm First Signal Check

The system checks for signals at market open (9:35 ET). After the first trading day:
```bash
# Check if any signals were generated
grep "signal" logs/exp880.log | tail -5
```

---

## Phase 5: Monitoring Setup

### 5.1 Telegram Alerts (Automatic)

Once running, you'll receive these automatically:
- **Trade alerts** — every entry and exit with P&L
- **Risk alerts** — if drawdown approaches threshold
- **Daily summary** — after market close (P&L, positions, hedge state)
- **Weekly report** — Sundays (vs backtest expectations)

### 5.2 Daily Monitoring Routine (5 minutes)

Every trading day, check:

1. **Telegram daily summary** — arrived? P&L reasonable?
2. **Open positions** — how many? Any stuck?
3. **Drawdown** — below 5%? (warn at 5%, halt at 12%)
4. **Kill switch** — still ARMED? (should always be armed)

### 5.3 Health Check Command

```bash
python3 -c "
from dotenv import load_dotenv; load_dotenv('.env.exp880')
import os, requests
r = requests.get('https://paper-api.alpaca.markets/v2/positions',
    headers={'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
             'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET']})
positions = r.json()
print(f'Open positions: {len(positions)}')
for p in positions:
    print(f'  {p[\"symbol\"]}: {p[\"qty\"]} @ {p[\"avg_entry_price\"]} (P&L: {p[\"unrealized_pl\"]})')
"
```

---

## Phase 6: Week 1 Watchlist

**The first week is critical.** Watch for:

| Issue | What to Look For | Action |
|-------|-----------------|--------|
| No trades | Zero signals after 3 days | Check signal pipeline, IV rank may be low |
| Too many trades | >2 trades per day | Check entry filter (P ≥ 0.75) |
| Large loss | Single trade > -$500 | Normal if stop triggered, review exit |
| Kill switch | Triggered message in Telegram | DO NOT restart. See escalation below |
| No Telegram | No daily summary | Check bot token, restart process |
| Alpaca error | API errors in logs | Check API keys, Alpaca status page |

**Normal first-week expectations:**
- 0-3 trades
- P&L between -$500 and +$500
- No kill switch triggers
- All Telegram alerts arriving

---

## Phase 7: Escalation Procedures

### Kill Switch Triggered
1. **Do NOT restart** for at least 24 hours
2. Check Telegram for the trigger reason (drawdown or daily loss)
3. Review logs: `grep "kill_switch" logs/exp880.log`
4. Contact engineering for root cause analysis
5. Only restart after explicit approval

### Anomalous Fill
1. Check Alpaca dashboard for the fill details
2. If fill price seems wrong (>2% from expected), screenshot it
3. Contact Alpaca support if fill looks erroneous
4. Log the incident in `logs/incidents.md`

### System Down (No Heartbeat)
1. Check if process is running: `ps aux | grep main.py`
2. Check logs for crash: `tail -50 logs/exp880.log`
3. Restart: `python3 main.py scheduler --config configs/paper_exp880.yaml --env-file .env.exp880`
4. If it crashes again, contact engineering

### Market Crash (VIX > 40)
1. System should auto-delever (Crisis Hedge V2)
2. Verify via Telegram: look for "scale reduction" messages
3. Do NOT manually override — let the system work
4. If no Telegram alerts, check system is running

---

## Phase 8: Victory Conditions (8-Week Targets)

After 8 weeks of paper trading, evaluate against these criteria:

| Criterion | Target | How to Check |
|-----------|--------|-------------|
| **Cumulative P&L** | Positive after week 4 | Telegram weekly reports |
| **Win rate** | > 65% | Weekly report |
| **Max drawdown** | < 12% | Daily summary drawdown field |
| **Kill switch triggers** | 0 | Would be a Telegram CRITICAL alert |
| **Paper vs backtest drift** | < 30% | Weekly report "vs expected" field |
| **Trades per month** | 2-8 | Count from trade alerts |
| **Telegram reliability** | 100% daily summaries received | Check every day |

### Passing = Proceed to Live

If ALL criteria pass after 8 weeks:
- [ ] Schedule review meeting
- [ ] Prepare live trading account ($10K initial)
- [ ] Update config: `paper_mode: false`
- [ ] Start with 1.0x leverage (no leverage for first live month)
- [ ] Follow same monitoring routine but with heightened attention

### Failing = Investigate

If any criterion fails:
- [ ] Document which criterion failed and by how much
- [ ] Contact engineering with weekly report data
- [ ] Extend paper trading by 4 weeks after fix
- [ ] Do NOT proceed to live until all criteria pass

---

## Quick Reference

| What | Command |
|------|---------|
| Start | `python3 main.py scheduler --config configs/paper_exp880.yaml --env-file .env.exp880` |
| Stop | `Ctrl+C` or `launchctl unload ~/Library/LaunchAgents/com.pilotai.exp880.plist` |
| Check logs | `tail -50 logs/exp880.log` |
| Check positions | See health check command in Phase 5.3 |
| Run pre-flight | See Phase 3.1 |
| Emergency stop | Kill the process: `pkill -f "main.py.*exp880"` |

---

*Questions? Contact engineering. Do not modify config files or code without approval.*
