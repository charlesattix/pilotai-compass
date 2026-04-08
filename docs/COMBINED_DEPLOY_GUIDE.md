# Combined Portfolio Deployment Guide

**For:** Charles (Mac Studio operator)
**Portfolio:** EXP-1220 + EXP-1780 + EXP-1820 + EXP-1660
**Architecture:** North Star v2 — regime-switching portfolio
**Expected Performance (validated, commit `55aa09f`):**
CAGR +101.6%, Sharpe 4.48, IS 107%/3.70 → OOS 96%/5.40

> **⛔ RULE ZERO:** This portfolio uses REAL market data only. Alpaca live
> quotes + Yahoo Finance VIX/SPY + IronVault historical cache. No synthetic
> data anywhere. Any deviation = FAIL.

---

## Architecture Overview

This is a **regime-switching portfolio** that allocates capital across four
strategies based on the current market regime (BULL / NEUTRAL / BEAR / HIGH_VOL).

| Regime | Conditions | EXP-1220 | EXP-1780 | EXP-1820 | EXP-1660 | Cash |
|--------|-----------|---------:|---------:|---------:|---------:|-----:|
| BULL | VIX < 20, uptrend | **90%** | 0% | 0% | 0% | 10% |
| NEUTRAL | Range-bound | **80%** | 0% | 10% | 0% | 10% |
| BEAR | Downtrend, VIX 20-30 | 50% | **30%** | 0% | 0% | 20% |
| HIGH_VOL | VIX ≥ 30 | 40% | **30%** | 0% | **20%** | 10% |

**Execution model:**
- **EXP-1220** is **autonomous** — runs via its own scanner (`scripts/run_exp1220.py`)
  with its own LaunchAgent. The combined runner does not submit 1220 orders directly.
- **EXP-1780, EXP-1820, EXP-1660** are **advisory** — the combined runner produces
  daily signal recommendations which Charles reviews and executes manually in the
  Alpaca paper UI.

---

## Deployment Package Contents

```
pilotai-credit-spreads/
├── configs/
│   ├── combined_portfolio.yaml       ← combined portfolio config (NEW)
│   └── deploy_exp1220_1.5x.yaml      ← EXP-1220's own config
├── scripts/
│   ├── run_combined.py                ← combined runner (NEW)
│   ├── run_exp1220.py                 ← autonomous EXP-1220 scanner
│   └── launch_exp1220.sh              ← EXP-1220 launcher
├── deploy/
│   ├── com.pilotai.exp1220.plist     ← existing LaunchAgent for EXP-1220
│   └── README_launchagent.md
├── docs/
│   └── COMBINED_DEPLOY_GUIDE.md       ← this file
├── compass/
│   └── crisis_alpha_v3.py             ← used by combined runner for EXP-1780 signals
├── shared/
│   └── telegram_alerts.py             ← alert delivery
└── .env.exp1220.example               ← env template (works for combined too)
```

Verify after `git pull`:
```bash
git checkout maximus/clean-features
git pull
ls configs/combined_portfolio.yaml \
   scripts/run_combined.py \
   scripts/run_exp1220.py \
   docs/COMBINED_DEPLOY_GUIDE.md
```

---

## Phase 0 — Prerequisites

### 0.1 Mac Studio
- [ ] Powered on, network connected
- [ ] Timezone: `America/New_York` (verify with `date`)
- [ ] Display sleep OK, system sleep disabled (or `pmset repeat`)
- [ ] Admin terminal access

### 0.2 Repo + Python
```bash
cd ~/pilotai  # or your actual checkout path
git fetch origin
git checkout maximus/clean-features
git pull

python3 --version  # >= 3.9
pip3 install pyyaml alpaca-py requests numpy pandas yfinance
```

### 0.3 Alpaca Paper Account
- Log in to https://app.alpaca.markets/paper/dashboard/overview
- Confirm your paper account is active (e.g. **PA3YFVQCXTD6** or your own)
- Starting equity: $100,000
- Copy `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` from the dashboard

### 0.4 Telegram Bot (optional but strongly recommended)
- On Telegram, message `@BotFather` → `/newbot` → copy token
- Send a message to your new bot, then visit:
  `https://api.telegram.org/bot<TOKEN>/getUpdates`
- Copy your `chat_id` from the JSON

---

## Phase 1 — Configuration

### 1.1 Create `.env`
```bash
cp .env.exp1220.example .env
nano .env  # or vim
```

Required:
```
ALPACA_API_KEY=PKxxxxxxxxxxxxxxxxxxxx
ALPACA_SECRET_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ALPACA_BASE_URL=https://paper-api.alpaca.markets
EXPERIMENT_ID=COMBINED
```

For Telegram alerts:
```
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=987654321
```

Secure it:
```bash
chmod 600 .env
```

### 1.2 Review `configs/combined_portfolio.yaml`
Open and confirm:
- `account.starting_capital` matches your paper account
- `risk.max_drawdown_halt_pct` is 10% (default)
- Telegram alerts are enabled

No edits normally required — defaults are calibrated from the validated backtest.

---

## Phase 2 — Pre-Launch Validation

### 2.1 Smoke test
```bash
source .env
python3 scripts/run_combined.py --smoke-test
```

Expected output:
```
[1/6] Loading config...              OK
[2/6] Checking state file...         OK
[3/6] Checking log directory...      OK
[4/6] Fetching VIX & SPY...          OK  VIX = xx.xx
                                      OK  SPY = $xxx.xx
[5/6] Running regime detection...    OK  Regime: BULL/NEUTRAL/BEAR/HIGH_VOL
                                      OK  Target allocations: ...
[6/6] Checking Alpaca connectivity... OK  Account xxx, equity $100,000

SMOKE TEST PASSED
```

**STOP if any check fails.**

### 2.2 Dry-run
```bash
python3 scripts/run_combined.py --dry-run
```

This runs the full daily cycle without submitting orders. You should see:
- Current regime detected
- Target allocations printed
- Advisory signals generated (if regime warrants)
- Note about EXP-1220 delegation

### 2.3 EXP-1220 scanner smoke test
EXP-1220 is autonomous — verify its scanner independently:
```bash
./scripts/launch_exp1220.sh smoke
./scripts/launch_exp1220.sh dry
```

See `DEPLOY_CHECKLIST.md` for full EXP-1220 setup details.

---

## Phase 3 — Manual Operation (Week 1)

### 3.1 Daily manual run
Run each morning after market open (~09:35 ET):
```bash
source .env
python3 scripts/run_combined.py
```

This will:
1. Fetch VIX and SPY from Yahoo
2. Detect the current regime
3. Check if regime changed (triggers rebalance)
4. Generate advisory signals for EXP-1780/1820/1660 if allocated
5. Log P&L to `logs/combined_pnl_journal.csv`
6. Update health file `logs/combined_health.json`
7. Send Telegram alerts if configured

### 3.2 Review advisory signals
The runner prints signal recommendations. **These are ADVISORY** — they
are NOT submitted automatically. Charles must review and manually execute
in Alpaca paper UI:

Example EXP-1780 output:
```
1 advisory signal(s) for manual review:
  [EXP-1780] {
    'as_of': '2026-04-02',
    'target_weights': {'TLT': 0.2, 'GLD': 0.15, 'EFA': 0.1, ...},
    'gross_exposure': 2.00,
  }
```

To execute: place equivalent ETF orders in Alpaca paper matching the weights.

### 3.3 EXP-1220 runs independently
EXP-1220 has its own scanner. Trigger it manually in week 1:
```bash
./scripts/launch_exp1220.sh scan
./scripts/launch_exp1220.sh status
```

### 3.4 Daily status check
```bash
python3 scripts/run_combined.py --status
```
Prints current regime, allocations, P&L, halt status.

### 3.5 Weekly P&L report
```bash
python3 scripts/run_combined.py --report --report-days 7
```

---

## Phase 4 — Automated Daily Run (Week 2+)

Only automate after 3-5 successful manual runs.

### 4.1 Create LaunchAgent plist
Save to `~/Library/LaunchAgents/com.pilotai.combined.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.pilotai.combined</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/env</string>
        <string>bash</string>
        <string>-lc</string>
        <string>cd /Users/charles/pilotai &amp;&amp; set -a &amp;&amp; source .env &amp;&amp; set +a &amp;&amp; /usr/bin/python3 scripts/run_combined.py</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/charles/pilotai</string>

    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>9</integer><key>Minute</key><integer>40</integer></dict>
    </array>

    <key>StandardOutPath</key>
    <string>/Users/charles/pilotai/logs/combined_launchd.out.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/charles/pilotai/logs/combined_launchd.err.log</string>

    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
        <key>PYTHONUNBUFFERED</key>
        <string>1</string>
    </dict>
</dict>
</plist>
```

**IMPORTANT:** Replace `/Users/charles/pilotai` with your actual checkout path.

### 4.2 Install
```bash
chmod 644 ~/Library/LaunchAgents/com.pilotai.combined.plist
launchctl unload ~/Library/LaunchAgents/com.pilotai.combined.plist 2>/dev/null
launchctl load   ~/Library/LaunchAgents/com.pilotai.combined.plist
launchctl list | grep pilotai
```

### 4.3 Test
```bash
launchctl start com.pilotai.combined
sleep 5
tail -20 logs/combined_portfolio.log
```

Both the combined runner AND EXP-1220 scanner should run daily. Verify:
```bash
launchctl list | grep pilotai
# should show both:
#   com.pilotai.combined
#   com.pilotai.exp1220
```

---

## Phase 5 — Monitoring

### 5.1 Key files
| File | Purpose |
|------|---------|
| `logs/combined_portfolio.log` | Main runner log (rotating, 50MB max, 5 backups) |
| `logs/combined_state.json` | Current regime, allocations, halt status |
| `logs/combined_health.json` | Latest health status (ok/warning/error/halted) |
| `logs/combined_pnl_journal.csv` | Daily P&L history |
| `logs/combined_launchd.err.log` | LaunchAgent stderr (scheduler errors) |

### 5.2 Daily health check
```bash
python3 scripts/run_combined.py --health
cat logs/combined_health.json
```
Exit codes: 0=OK, 1=warning, 2=error.

### 5.3 Telegram alerts you'll receive
| Event | Example |
|-------|---------|
| Regime change | `BULL → BEAR, VIX 25.5, SPY 650` |
| Rebalance | `Rebalance executed: BEAR regime, target allocations ...` |
| DD warning | `Warning: Portfolio DD 5.2% (warning threshold 5.0%)` |
| DD critical | `CRITICAL: Portfolio DD 8.5% crossed 8.0%` |
| Halt | `HALT: Portfolio DD 10.2% >= 10% limit. All trading suspended.` |
| Recovery | `RESUMED: DD recovered to 4.8%. Trading active.` |
| Runner error | `Daily run ERROR: <exception>` |

### 5.4 Metrics to watch
| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| Portfolio DD | < 3% | 3-5% | ≥ 5% |
| Daily P&L | > -1% | -1% to -2% | < -2% |
| Weekly P&L | > -3% | -3% to -5% | < -5% |
| Health status | `ok` | `warning` | `error` / `halted` |
| Days since run | 0-1 | 2-3 | ≥ 4 |

### 5.5 Drawdown thresholds (configured)
- **5%** — Telegram warning
- **8%** — Telegram critical alert
- **10%** — Automatic HALT (all trading suspended)
- **5%** — Recovery threshold (resume trading)

---

## Phase 6 — Emergency Procedures

### Kill switch (combined portfolio)
The combined runner halts automatically on 10% DD. To manually force a halt:
```bash
python3 -c "
import json, pathlib
p = pathlib.Path('logs/combined_state.json')
s = json.loads(p.read_text()) if p.exists() else {}
s['halted'] = True
s['halt_reason'] = 'manual halt'
p.write_text(json.dumps(s, indent=2))
print('Combined portfolio HALTED')
"
```
To resume: set `halted` back to `false` in the same file.

### Kill switch (EXP-1220)
```bash
./scripts/launch_exp1220.sh close-all
```
Prompts for confirmation, then submits BTC+STC for every open EXP-1220 position.

### Disable schedulers
```bash
launchctl unload ~/Library/LaunchAgents/com.pilotai.combined.plist
launchctl unload ~/Library/LaunchAgents/com.pilotai.exp1220.plist
```

### Runner won't start
```bash
cat logs/combined_launchd.err.log
python3 scripts/run_combined.py --smoke-test
```

### Alpaca auth errors
```bash
env | grep ALPACA  # must show 3 entries after `source .env`
python3 -c "
import os
from alpaca.trading.client import TradingClient
tc = TradingClient(os.environ['ALPACA_API_KEY'], os.environ['ALPACA_SECRET_KEY'], paper=True)
a = tc.get_account()
print('OK:', a.status, 'equity:', a.equity)
"
```

### Regime detection returning wrong answer
Yahoo Finance can occasionally be flaky. Check:
```bash
python3 -c "
from scripts.run_combined import get_vix, get_spy_price, get_spy_ma50_slope
print('VIX:', get_vix())
print('SPY:', get_spy_price())
print('Slope:', get_spy_ma50_slope())
"
```
If values look wrong, the runner defaults to NEUTRAL and continues.

### State reset
If state file gets corrupted:
```bash
mv logs/combined_state.json logs/combined_state.json.bak
python3 scripts/run_combined.py --dry-run
```

---

## Quick Reference

```bash
# Pre-launch
python3 scripts/run_combined.py --smoke-test
python3 scripts/run_combined.py --dry-run

# Daily operation
python3 scripts/run_combined.py                # live run
python3 scripts/run_combined.py --status       # current state
python3 scripts/run_combined.py --health       # health check
python3 scripts/run_combined.py --report       # weekly P&L

# EXP-1220 (autonomous, runs via its own scanner)
./scripts/launch_exp1220.sh scan
./scripts/launch_exp1220.sh status
./scripts/launch_exp1220.sh close-all

# LaunchAgent management
launchctl load ~/Library/LaunchAgents/com.pilotai.combined.plist
launchctl start com.pilotai.combined
launchctl list | grep pilotai
```

---

## Success Criteria (Week 8 Review)

After 8 weeks of paper trading:

| Metric | Target | Minimum |
|--------|--------|---------|
| Weekly runs executed | ≥ 40 | 30 |
| EXP-1220 trades filled | ≥ 10 | 6 |
| Advisory signals acted on | N/A (optional) | 0 (OK) |
| Max DD | < 8% | < 12% |
| Realized CAGR (annualized) | ≥ 50% | ≥ 30% |
| Telegram alerts received | > 0 | > 0 |
| Runner errors | 0 | ≤ 2 |
| Health: OK days | ≥ 95% | ≥ 90% |

**Decision point:** If all minimums met → promote to live trading.

---

## Honest Caveats

1. **Only EXP-1220 is autonomous.** The other three strategies produce
   advisory signals that Charles must manually execute. Automating them
   is backlog (requires building per-strategy scanners).

2. **The 101.6% CAGR / 4.48 Sharpe target** comes from yearly-bar backtest.
   Real execution on daily bars will have higher realized vol and worse
   Sharpe. Budget for 30-60% of backtest Sharpe in live (so 1.5-3.0
   realized Sharpe in paper).

3. **Regime detection relies on Yahoo Finance.** If Yahoo is down, the
   runner defaults to NEUTRAL and continues — not a halt. Monitor the
   regime field in `combined_state.json` to catch staleness.

4. **EXP-1780 advisory signals involve 12+ ETF positions.** Manual
   execution is laborious. Consider skipping EXP-1780 initially and
   using only EXP-1220 + EXP-1820 + EXP-1660 until automation catches up.

5. **No monthly/daily stress test** yet on the combined portfolio. The
   commit `55aa09f` acknowledges this. Treat paper weeks 1-4 as the
   real stress test.

---

## Contacts
- **Operator:** Charles (Mac Studio)
- **Owner:** Carlos
- **Strategist:** Maximus (branch `maximus/clean-features`)

For questions, check in this order:
1. `logs/combined_health.json`
2. `logs/combined_portfolio.log`
3. This guide (Phase 6 troubleshooting)
4. `DEPLOY_CHECKLIST.md` (EXP-1220 specifics)
5. `deploy/README_launchagent.md`

---

*Combined Portfolio Deployment Guide — 2026-04-06*
*Runner: scripts/run_combined.py v1.0.0*
*Config: configs/combined_portfolio.yaml v1.0.0*
*Source commit: 55aa09f (North Star v2)*
