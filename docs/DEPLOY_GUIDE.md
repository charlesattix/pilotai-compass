# EXP-1220 Deployment Guide — Paper Trading on Mac Studio

## Overview

**Strategy:** EXP-1220 SPY credit spreads with tail risk protection
**Leverage:** 1.5× static (validated: 99.2% CAGR, 3.83 Sharpe, 11.2% DD)
**Cadence:** Enter new trade every 7 days (Monday scan)
**Max concurrent:** 5 positions

---

## 1. Prerequisites

### Hardware
- Mac Studio (M1/M2/M3 — any Apple Silicon)
- Stable internet connection
- Running during market hours (9:30 AM - 4:00 PM ET)

### Software
```bash
# Python 3.10+
python3 --version

# Required packages
pip3 install numpy pandas pyyaml alpaca-py requests
```

### Alpaca Account
1. Sign up at https://alpaca.markets
2. Create a **Paper Trading** account first
3. Go to API Keys → Generate new key pair
4. Note your:
   - **API Key ID** (starts with `PK...`)
   - **Secret Key** (starts with a long alphanumeric string)

---

## 2. Environment Setup

### Set API Credentials

Add to your `~/.zshrc` (or `~/.bash_profile`):

```bash
# Alpaca Paper Trading API
export ALPACA_API_KEY='PK...'           # your API key
export ALPACA_SECRET_KEY='...'          # your secret key
export ALPACA_BASE_URL='https://paper-api.alpaca.markets'  # paper trading

# Optional: Telegram alerts
export TELEGRAM_BOT_TOKEN='...'
export TELEGRAM_CHAT_ID='...'
```

Then reload:
```bash
source ~/.zshrc
```

### Clone the Repo
```bash
cd ~/Projects
git clone <repo-url> pilotai-credit-spreads
cd pilotai-credit-spreads
```

### Verify Setup
```bash
./scripts/deploy_paper.sh check
```

This verifies Python, packages, API keys, and config file.

---

## 3. Launch Paper Trading

### One-Command Start
```bash
./scripts/deploy_paper.sh start
```

### Monitor
```bash
# Watch live logs
tail -f logs/paper_trading.log

# Check status
./scripts/deploy_paper.sh status

# Stop
./scripts/deploy_paper.sh stop
```

---

## 4. macOS LaunchAgent (Auto-Start)

To run paper trading automatically on login:

```bash
mkdir -p ~/Library/LaunchAgents
cat > ~/Library/LaunchAgents/com.pilotai.paper-trading.plist << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pilotai.paper-trading</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>cd ~/Projects/pilotai-credit-spreads && ./scripts/deploy_paper.sh start</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>EnvironmentVariables</key>
    <dict>
        <key>ALPACA_API_KEY</key>
        <string>YOUR_KEY_HERE</string>
        <key>ALPACA_SECRET_KEY</key>
        <string>YOUR_SECRET_HERE</string>
        <key>ALPACA_BASE_URL</key>
        <string>https://paper-api.alpaca.markets</string>
    </dict>
    <key>StandardOutPath</key>
    <string>/tmp/pilotai-paper-trading.out</string>
    <key>StandardErrorPath</key>
    <string>/tmp/pilotai-paper-trading.err</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>25</integer>
    </dict>
</dict>
</plist>
PLIST
```

**Replace `YOUR_KEY_HERE` and `YOUR_SECRET_HERE`** with actual credentials.

Load the agent:
```bash
launchctl load ~/Library/LaunchAgents/com.pilotai.paper-trading.plist
```

Unload:
```bash
launchctl unload ~/Library/LaunchAgents/com.pilotai.paper-trading.plist
```

---

## 5. How Leverage Works on Alpaca (Credit Spreads)

### Key Concept: 1.5× Leverage Does NOT Mean Margin Borrowing

For **defined-risk credit spreads**, leverage is purely about **position sizing**.

A bull put spread has:
- **Max loss** = (spread width - credit received) × 100 per contract
- **Buying power reduction** = max loss (NOT the notional value)

### Example: $100K Account at 1.5× Leverage

| Parameter | 1.0× (base) | 1.5× (deployed) |
|-----------|-------------|-----------------|
| Risk per trade | 1.0% = $1,000 | 1.5% = $1,500 |
| Spread width | $5 | $5 |
| Credit received | ~$0.50 | ~$0.50 |
| Max loss/contract | ~$450 | ~$450 |
| Contracts | 2 | 3 |
| Max concurrent | 5 | 5 |
| Max total risk | $4,500 (4.5%) | $6,750 (6.75%) |
| Buying power used | 4.5% | 6.75% |

### Why Alpaca Allows This
- Credit spreads are **defined risk** — max loss is known at entry
- Alpaca's margin requirement = max loss (not notional)
- With 5 positions × $1,500 risk = $7,500 total = **7.5% of account**
- Plenty of buying power headroom (Reg T gives ~50% of account)
- **No margin interest** because you're not borrowing — you're sizing positions larger

### What "1.5× leverage" Really Means
- Base strategy sizes at 1% risk per trade
- At 1.5×, we size at 1.5% risk per trade (50% larger positions)
- Total portfolio never risks more than 8% simultaneously
- The "leverage" is conceptual, not mechanical margin leverage

---

## 6. What to Monitor

### Daily (16:05 ET — automated report)
- Total P&L for the day
- Number of open positions
- Current drawdown from peak

### Weekly (Monday morning before scan)
- Review trade journal (`logs/trade_journal.csv`)
- Check win rate (target: >75%)
- Verify VIX level (skip scan if VIX > 35)

### Alert Thresholds
| Alert | Threshold | Action |
|-------|-----------|--------|
| VIX spike | > 30 | Review positions, tighten stops |
| Daily loss | > 2% | Check for stop-loss issues |
| Weekly loss | > 5% | Halt new entries for 1 week |
| Drawdown | > 10% | **HALT ALL TRADING** until review |
| Position opened | Any | Verify correct strikes and sizing |
| Stop loss hit | Any | Log reason, check if systematic |

### Red Flags (Call Carlos)
- Drawdown exceeds 10%
- 3+ consecutive losing trades
- Unusual position sizing (>5 contracts)
- VIX > 40 with open positions
- Any unrecognized trades in account

---

## 7. Config Reference

Full config: `configs/deploy_exp1220_1.5x.yaml`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Leverage | 1.5× static | Pareto-optimal for DD<12% (see dcf617c) |
| Cadence | 7 days | Balances capital utilization vs overtrading |
| Max concurrent | 5 | ~5-6 weeks of overlapping 30-DTE positions |
| Spread width | $5 | Standard SPY spread, good liquidity |
| Target DTE | 30 | Sweet spot for theta decay vs gamma risk |
| OTM % | 5% | ~85-delta short strike, safe distance |
| Profit target | 50% | Close at half max profit (proven optimal) |
| Stop loss | 2× credit | Cap losses at 2× credit received |
| VIX max entry | 35 | Don't sell premium in extreme fear |
| Hedge | None | Static leverage doesn't benefit from hedge overlay |

---

## 8. Transitioning to Live

After 90 days of paper trading with:
- Positive total P&L
- Win rate > 70%
- Max DD < 12%
- No system errors

Then:
1. Change `mode: paper` → `mode: live` in config
2. Change `ALPACA_BASE_URL` to `https://api.alpaca.markets`
3. Start with **50% of target capital** for first 30 days
4. Scale to full capital after confirming live fills match paper

---

*Generated by PilotAI deployment pipeline. Config validated against walk-forward OOS 2022-2025.*
