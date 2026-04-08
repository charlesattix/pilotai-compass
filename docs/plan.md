# PilotAI Credit Spread Trading System
## Project Plan & Architecture

---

## 1. Executive Summary

PilotAI is an automated credit spread trading system that scans the options market for high-probability credit spread opportunities, generates actionable alerts, and executes paper trades automatically. The system targets a **90%+ probability of profit** per trade using disciplined delta-based strike selection on highly liquid ETFs.

**Current Status:** Live and operational. Dashboard deployed. Paper trading engine running. Real broker integration (Alpaca) in progress.

---

## 2. What Is a Credit Spread?

A credit spread is an options strategy where you simultaneously sell a closer-to-the-money option and buy a further-out-of-the-money option. You collect a net credit upfront. If the underlying stays away from your strikes by expiration, you keep the credit as profit.

**Two types we trade:**
- **Bull Put Spread** (Bullish) — Sell a put, buy a lower put. Profits if price stays above the short strike.
- **Bear Call Spread** (Bearish) — Sell a call, buy a higher call. Profits if price stays below the short strike.

**Why credit spreads?**
- Defined risk (max loss = spread width minus credit)
- High probability of profit (we target 85-90%+)
- Time decay works in our favor (theta positive)
- Don't need to predict direction precisely — just need price to stay in a range

---

## 3. Strategy Rules

| Parameter | Value |
|---|---|
| **Underlyings** | SPY, QQQ, IWM |
| **Short Strike Delta** | 0.10 – 0.15 (85-90% probability OTM) |
| **Days to Expiration** | 30 – 45 DTE |
| **Spread Width** | $5 |
| **Minimum Credit** | 20% of spread width ($1.00 on $5 spread) |
| **IV Rank Minimum** | 25+ (prefer elevated volatility for richer premiums) |
| **Profit Target** | 50% of max credit |
| **Stop Loss** | 2.5× the credit received |
| **Max Concurrent Positions** | 5 |
| **Max Risk Per Trade** | 2% of account |

### Entry Criteria
1. Technical analysis confirms directional bias (RSI, moving averages, volume)
2. IV Rank ≥ 25 (options are relatively expensive)
3. Short strike delta between 0.10-0.15
4. Credit received ≥ 20% of spread width
5. 30-45 DTE for optimal theta decay
6. Liquidity check — sufficient open interest and tight bid-ask

### Exit Rules
- **Winner:** Close at 50% of max profit (don't get greedy)
- **Loser:** Close if loss reaches 2.5× credit received
- **Time:** Close at 7 DTE regardless (avoid gamma risk near expiration)
- **Emergency:** Close if short strike is breached

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────┐
│                   DATA LAYER                     │
│  yfinance (free/delayed) ← current default       │
│  Tradier API (real-time, real Greeks) ← pending   │
│  Polygon.io (historical bars) ← built            │
│  Alpaca (paper trading execution) ← in progress  │
└─────────────┬───────────────────────┬────────────┘
              │                       │
              ▼                       ▼
┌─────────────────────┐  ┌─────────────────────────┐
│  STRATEGY ENGINE     │  │  PAPER TRADING ENGINE    │
│  • Technical Analysis│  │  • Auto-execute signals  │
│  • Options Analyzer  │  │  • Position tracking     │
│  • Opportunity Scorer│  │  • P&L monitoring        │
│  • Signal Generator  │  │  • Auto-exit management  │
└─────────┬───────────┘  │  • Alpaca API (live sim)  │
          │               └──────────┬────────────────┘
          ▼                          │
┌─────────────────────┐              │
│  ALERT GENERATOR     │              │
│  • JSON/CSV/TXT      │              │
│  • Telegram push     │              │
│  • Score + PoP + Risk│              │
└─────────┬───────────┘              │
          │                          │
          ▼                          ▼
┌─────────────────────────────────────────────────┐
│              WEB DASHBOARD                       │
│  "Alerts by PilotAI"                             │
│  • Live alerts feed with expandable cards        │
│  • Paper trading portfolio view                  │
│  • Real-time ticker tape (TradingView)           │
│  • Performance stats (from real trade data)      │
│  • Hosted on Railway (permanent URL)             │
│  https://pilotai-alerts-production.up.railway.app│
└─────────────────────────────────────────────────┘
```

---

## 5. What's Built (Complete)

### 5.1 Core Python System
- **17 Python files, ~3,500+ lines**
- Scanner finds opportunities across SPY, QQQ, IWM
- Black-Scholes delta calculation for accurate strike selection
- Technical analysis: RSI, moving averages, volume analysis
- Scoring algorithm ranks opportunities by probability, credit, and risk
- Alert generation in JSON, CSV, and plain text formats
- Full backtest module for historical validation

### 5.2 Paper Trading Engine
- Auto-executes top-scored signals on each scan
- Tracks entries, exits, P&L per trade
- Position sizing: max 2% risk per trade
- Auto-closes at 50% profit or 2.5× stop loss
- Stores all data in JSON for dashboard consumption

### 5.3 Web Dashboard
- **Next.js 14** with TypeScript and Tailwind CSS
- PilotAI branding: purple→pink→orange gradient, light theme
- Live ticker tape via TradingView widget
- Alert cards with full trade details, expandable
- Paper trading portfolio page with positions and P&L
- Deployed on Railway at permanent URL
- Mobile responsive (900px, 600px breakpoints)

### 5.4 Automated Scanning
- Cron job: every 30 minutes, 10 AM – 3:30 PM ET, Mon-Fri
- Auto-scans, auto-trades, and pushes summary to Telegram

### 5.5 Data Providers Built
- **yfinance** — working (free, 15-min delay)
- **Tradier** — provider built, pending API key
- **Polygon.io** — provider built, free tier limited to historical data

---

## 6. What's In Progress

### 6.1 Alpaca Paper Trading Integration
**Status: Account created, completing setup**

Alpaca provides a real paper trading environment with:
- Simulated order execution against live market data
- Real fill prices based on actual bid/ask
- Options trading enabled by default on paper accounts
- Full REST API for programmatic trading
- Free — no cost for paper trading

**What this gives us:**
- Instead of our JSON-based simulation, trades execute on Alpaca's paper trading platform
- Real order fills, realistic slippage
- Proper options chain data
- Portfolio tracking through Alpaca's dashboard + our dashboard

### 6.2 Telegram Alerts
- Bot module exists, needs bot token configuration
- Will push alerts with trade details on each scan

---

## 7. What's Next

| Priority | Task | Timeline |
|---|---|---|
| **P0** | Complete Alpaca integration — real paper trading | This week |
| **P0** | First paper trades on Alpaca during market hours | Next trading day |
| **P1** | Telegram alert bot — push notifications | This week |
| **P1** | Dashboard: live P&L from Alpaca positions | This week |
| **P2** | Backtest validation — run 6-month historical test | Next week |
| **P2** | Add more underlyings (AAPL, MSFT, AMZN) | Next week |
| **P3** | Iron condor strategy (neutral market) | Week 3 |
| **P3** | Live trading readiness (Alpaca live account) | After paper validation |

---

## 8. Risk Management

### Per-Trade Risk
- Max 2% of account per trade
- Defined max loss (spread width - credit) × contracts
- Stop loss at 2.5× credit to cut losers early

### Portfolio Risk
- Max 5 concurrent positions
- Diversified across SPY, QQQ, IWM
- No single underlying > 40% of portfolio risk
- Dashboard shows real-time portfolio risk bar

### Strategy Edge
- Selling options with 85-90% probability of expiring worthless
- Time decay (theta) generates daily income
- IV Rank filter ensures we sell when premiums are rich
- 50% profit target locks in gains early, improves win rate further

---

## 9. Access & URLs

| Resource | URL |
|---|---|
| **Dashboard** | https://pilotai-alerts-production.up.railway.app |
| **Paper Trading** | https://pilotai-alerts-production.up.railway.app/paper-trading |
| **Alpaca Dashboard** | https://app.alpaca.markets (atlas@attix.com) |
| **System Files** | Mac Studio: ~/credit-spread-system/ |

---

## 10. Technology Stack

| Component | Technology |
|---|---|
| Strategy Engine | Python 3.9, scipy, numpy, pandas |
| Options Data | yfinance (current), Tradier/Alpaca (planned) |
| Technical Analysis | Custom RSI, SMA, EMA + ta-lib |
| Paper Trading | Alpaca Markets API |
| Web Dashboard | Next.js 14, TypeScript, Tailwind CSS |
| Hosting | Railway |
| Alerts | Telegram Bot API |
| Scheduling | OpenClaw cron (30-min intervals) |

---

*Document generated by Atlas for PilotAI — February 12, 2026*
