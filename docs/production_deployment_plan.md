# Production Deployment Plan — Ultimate Portfolio

**Version:** 1.0
**Date:** 2026-04-05
**Status:** DRAFT — Pending paper trading validation
**Target:** Backtest → Paper → Live AUM

---

## Executive Summary

This document is the complete roadmap for deploying the Ultimate Portfolio from
backtested strategies to live capital. It covers architecture, capital allocation,
risk management, execution, alerting, failover, and a phased launch checklist.

**Portfolio composition:**

| Strategy | Weight | Source Module | Sharpe | CAGR |
|----------|--------|--------------|--------|------|
| EXP-1220 Tail Risk Protection | 95.0% | `compass/tail_risk_hedge.py` | 5.78 | ~55% |
| Cross-Asset Pairs (TLT-QQQ) | 1.67% | `compass/gld_tlt_relval.py` | 5.06 OOS | — |
| TLT Iron Condors | 1.67% | `compass/iron_condor_optimizer.py` | 2.69 | 10.2% |
| Vol Term Structure | 1.67% | `compass/vol_term_structure_deep_dive.py` | 2.81 OOS | 0.55% |

**Backtest performance (1507 days):** CAGR 55.56%, Sharpe 4.10, Max DD 7.21%
**OOS walk-forward (2022–2025):** Avg CAGR 89.6%, Avg Sharpe 3.57, Max DD 11.4%
**Target live leverage:** 1.6x (yields ~100% CAGR at 11.4% max DD in backtest)

---

## 1. Architecture

### 1.1 Design: Single Scheduler + Signal Aggregator

```
┌──────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                       │
│              compass/live_bridge.py                   │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐  │
│  │ EXP-1220 │ │ X-Asset  │ │ TLT ICs  │ │VolTerm │  │
│  │ Tail Risk│ │  Pairs   │ │          │ │  Str.  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───┬────┘  │
│       │             │            │            │       │
│       └──────┬──────┴─────┬──────┘            │       │
│              ▼            ▼                    ▼       │
│        Signal Aggregator (merge, dedupe, weight)      │
│              │                                        │
│              ▼                                        │
│     ┌────────────────┐                                │
│     │  Risk Overlay   │  compass/risk_overlay.py      │
│     │  (5 layers)     │                               │
│     └───────┬────────┘                                │
│             ▼                                         │
│     ┌────────────────┐                                │
│     │ Order Manager   │  compass/order_manager.py     │
│     └───────┬────────┘                                │
│             ▼                                         │
│     ┌────────────────┐                                │
│     │ Execution Algo  │  compass/execution_algo.py    │
│     │ (TWAP/VWAP/IS) │                               │
│     └───────┬────────┘                                │
│             ▼                                         │
│     ┌────────────────┐                                │
│     │ Broker Adapter  │  Alpaca REST API              │
│     └───────┬────────┘                                │
│             ▼                                         │
│     ┌────────────────┐                                │
│     │ Prod Monitor    │  compass/prod_monitor.py      │
│     │ + Telegram      │  shared/telegram_alerts.py    │
│     └────────────────┘                                │
└──────────────────────────────────────────────────────┘
```

**Why single process, not separate processes per strategy:**

1. **Capital is shared** — strategies compete for the same margin. A single process
   prevents over-allocation when two strategies signal simultaneously.
2. **Risk overlay requires portfolio-level view** — DD circuit breaker, delta limits,
   and correlation checks need to see all positions at once.
3. **Simpler ops** — one cron job, one log stream, one health check endpoint.

**Process model:**

- **Cron-scheduled runner** (`scripts/run_portfolio.sh`) executes once daily at
  market open (9:31 AM ET) and once at close (3:50 PM ET).
- Each run: refresh data → compute signals → aggregate → risk check → execute → report.
- Intraday monitoring via a lightweight heartbeat loop (every 5 min) that checks
  positions, Greeks, and P&L without generating new signals.

### 1.2 Module Responsibilities

| Module | Role | Frequency |
|--------|------|-----------|
| `compass/tail_risk_hedge.py` | Compute crisis score, leverage target, hedge allocation | Daily open |
| `compass/gld_tlt_relval.py` | GLD/TLT z-score signal, spread selection via IronVault | Daily open |
| `compass/iron_condor_optimizer.py` | TLT IC entry/exit signals | Monthly (new positions) |
| `compass/vol_term_structure_deep_dive.py` | Contango/backwardation signal | Daily open |
| `compass/risk_overlay.py` | Apply all 5 risk layers to aggregated signals | Every signal |
| `compass/events.py` | FOMC/CPI/NFP event gate calendar | Daily pre-check |
| `compass/live_bridge.py` | Signal → Order translation, position reconciliation | Every signal |
| `compass/order_manager.py` | Order lifecycle, fill tracking, cost attribution | Per order |
| `compass/execution_algo.py` | TWAP/VWAP/IS algo selection, slice scheduling | Per order |
| `compass/prod_monitor.py` | Greeks, P&L, margin, latency monitoring | 5-min loop |
| `shared/circuit_breaker.py` | Broker API fault tolerance | Per API call |

### 1.3 Data Flow

```
Market data (Polygon/Alpaca)
  │
  ├─→ SPY/VIX/VIX3M prices → Regime classifier → Dynamic leverage
  ├─→ GLD/TLT/XLF prices   → Z-score ratios    → Pair signals
  ├─→ TLT option chains     → IC optimizer       → IC entry signals
  ├─→ VIX term structure     → Contango signal    → Vol term signal
  │
  └─→ Signal Aggregator
        │
        ├─→ Weight by portfolio allocation (95/1.67/1.67/1.67)
        ├─→ Risk overlay: leverage × event gate × DD breaker
        ├─→ Position delta check (net < 50)
        ├─→ Margin check (utilization < 80%)
        │
        └─→ Order generation → Broker
```

---

## 2. Capital Allocation

### 2.1 Strategy Weights

From `reports/ultimate_portfolio.json`, optimized via walk-forward:

| Strategy | Weight | Capital ($100K) | Margin Reserved | Max Positions |
|----------|--------|-----------------|-----------------|---------------|
| EXP-1220 Tail Risk | 95.0% | $95,000 | Overlay (no margin) | N/A (hedge overlay) |
| Cross-Asset Pairs | 1.67% | $1,670 | $835 | 2 |
| TLT Iron Condors | 1.67% | $1,670 | $835 | 2 |
| Vol Term Structure | 1.67% | $1,670 | $835 | 2 |
| **Cash buffer** | — | — | $2,500 | — |

**Note:** EXP-1220 is a leverage/hedge overlay on the entire portfolio, not a
separate capital allocation. It modulates the leverage applied to all positions
and purchases tail protection hedges (SPY puts + VIX calls) using 2% annual budget.

### 2.2 Leverage Implementation

- **Target leverage:** 1.6x (from Monte Carlo optimization)
- **Implementation:** Alpaca 2.0x margin account; actual leverage modulated daily
  by the risk overlay's dynamic leverage engine (0.3x–1.8x range)
- **Effective capital deployment:** $100K equity × 1.6x = $160K notional
- **Margin buffer:** Always maintain >20% cash margin to avoid margin calls

### 2.3 Rebalancing

- **Frequency:** Quarterly (aligned with walk-forward window expansion)
- **Trigger conditions for off-cycle rebalance:**
  - Any strategy weight drifts >5pp from target
  - Portfolio DD exceeds 8% (reduce leverage, rebalance to cash)
  - New strategy validated and added to portfolio
- **Rebalance method:** Sell overweight positions first, then buy underweight.
  Never rebalance within 2 days of FOMC/CPI/NFP (event gate prevents this).

### 2.4 Scaling Plan

| Phase | AUM | Leverage | Strategies | Capacity Limit |
|-------|-----|----------|------------|----------------|
| Paper | $100K virtual | 1.6x | 4 strategies | N/A |
| Seed | $25K–$50K | 1.0x | 4 strategies | GLD OI (~$700K) |
| Growth | $50K–$200K | 1.2x–1.6x | 4 strategies + XLI ICs | TLT OI (~$2M) |
| Scale | $200K–$1M | 1.6x | All validated pairs | Multi-broker split |

---

## 3. Risk Monitoring

### 3.1 Real-Time DD Tracking

The `compass/risk_overlay.py` DD circuit breaker provides portfolio-level protection:

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| Portfolio DD | 5% | 10% | 10% → cut to 0.5x leverage until recovery to <5% |
| Daily P&L loss | -$2,500 | -$5,000 | Critical → halt new entries for 24h |
| Strategy DD | 3% | 5% | 5% → disable strategy, log, alert |
| VIX spike | >28 | >40 | >40 → crisis mode (0.4x leverage, 80% hedge ratio) |

**Implementation:** `compass/prod_monitor.py` runs every 5 minutes during market hours:

```python
MonitorConfig(
    max_daily_loss=5000.0,
    max_drawdown_pct=0.10,
    max_delta=500.0,
    max_gamma=100.0,
    max_vega=5000.0,
    max_margin_utilization=0.80,
    min_fill_rate=0.80,
    max_avg_slippage_bps=10.0,
    max_signal_age_minutes=60.0,
    max_latency_ms=500.0,
)
```

### 3.2 Position Limits

| Limit | Value | Scope | Enforcement |
|-------|-------|-------|-------------|
| Max total positions | 20 | Portfolio | `paper_trading_engine.py` EngineConfig |
| Max per strategy | 10 | Per strategy | Pre-trade risk check |
| Max contracts per position | 50 | Per position | Order manager |
| Max portfolio delta | ±50 | Net portfolio | Greeks monitor |
| Max portfolio vega | $5,000 | Net portfolio | Greeks monitor |
| Margin utilization | <80% | Portfolio | Pre-trade + continuous |
| Correlation limit | <0.80 | Between positions | Pre-trade check |

### 3.3 Per-Strategy Kill Switches

Each strategy runs through the `shared/circuit_breaker.py` fault tolerance wrapper:

```
Strategy Kill Switch States:
  ARMED     → Normal operation
  TRIGGERED → Strategy halted after threshold breach
  MANUAL    → Operator halted via Telegram /kill command
  DISARMED  → Operator explicitly resumed
```

**Trigger conditions per strategy:**

| Strategy | Max DD | Max Consecutive Losses | Staleness (no signal) |
|----------|--------|----------------------|----------------------|
| EXP-1220 Tail Risk | N/A (overlay) | N/A | 24 hours |
| Cross-Asset Pairs | 5% | 5 | 48 hours |
| TLT Iron Condors | 5% | 3 | 7 days (monthly) |
| Vol Term Structure | 3% | 5 | 48 hours |

### 3.4 Risk Overlay Integration

The `compass/risk_overlay.py` module wraps all portfolio returns through 5 layers:

1. **Dynamic leverage** — VIX/TS/rvol-aware scaling (0.3x–1.8x)
2. **Tail risk hedging** — SPY puts + VIX calls (2% annual budget)
3. **Event gates** — FOMC (0.5x), CPI (0.65x), NFP (0.75x) on day-of
4. **Position stops** — 3% fixed + 5% trailing per position
5. **DD circuit breaker** — 10% DD → 0.5x until recovery to 5%

All layers are independently toggleable via `RiskOverlayConfig`.

---

## 4. Execution

### 4.1 Order Routing

```
Signal → Risk Check → Order Generation → Algo Selection → Broker
```

**Algo selection** (from `compass/execution_algo.py`):

| Urgency | Algorithm | Use Case |
|---------|-----------|----------|
| LOW | TWAP | Monthly IC entries, rebalances |
| MEDIUM | VWAP | Daily pair signals, vol term entries |
| HIGH | Implementation Shortfall | Hedge adjustments during elevated vol |
| CRITICAL | Market | Crisis hedge buys, circuit breaker liquidations |

**Order types:**

- **Credit spreads (pairs, ICs):** Limit order at mid-price, step toward natural
  after 30 seconds, cancel-replace after 60 seconds. Max 3 attempts.
- **Hedge buys (SPY puts, VIX calls):** Market order when crisis score > 0.7.
  Limit order at ask + $0.05 when crisis score 0.3–0.7.
- **Emergency liquidation:** Market order, all positions, immediate.

### 4.2 Slippage Budget

| Strategy | Expected Slippage | Budget | Notes |
|----------|------------------|--------|-------|
| Cross-Asset Pairs | $0.02–0.04/contract | 5 bps | GLD/TLT monthly options, moderate OI |
| TLT Iron Condors | $0.03–0.05/contract | 5 bps | Monthly expirations, good liquidity |
| Vol Term Structure | $0.02–0.03/contract | 3 bps | SPY options, deep liquidity |
| Hedge positions | $0.05–0.10/contract | 10 bps | Speed over price in crisis |

**Total cost model** (from walk-forward production config):
- Spread: $0.50/contract
- Commission: $0.005/contract
- Effective cost: 10.6 bps per turnover event
- Annual cost budget for hedges: 2% of AUM

### 4.3 Fill Monitoring

`compass/prod_monitor.py` tracks fill quality continuously:

- **Fill rate** — target >95%, alert at <80%
- **Avg slippage** — target <5 bps, alert at >10 bps
- **Partial fills** — auto-cancel remainder after 120 seconds
- **Rejected orders** — immediate Telegram alert + log investigation

**Fill reconciliation** (daily at 4:15 PM ET):
- Compare expected fills (from order manager) with actual fills (from broker API)
- Flag any discrepancy >$10 or >1 contract
- Log to `data/fill_reconciliation.db`

---

## 5. Alerting

### 5.1 Telegram Integration

All alerts route through `shared/telegram_alerts.py` to a dedicated Telegram channel.

**Alert categories:**

| Level | Category | Example | Rate Limit |
|-------|----------|---------|------------|
| INFO | Trade entry | "OPEN: TLT IC 92/90/88/86 × 3 contracts, $0.42 credit" | Per trade |
| INFO | Trade exit | "CLOSE: GLD-TLT pair, +$180 (profit target)" | Per trade |
| INFO | Daily summary | "EOD: +$340, DD 2.1%, 4 positions, leverage 1.4x" | Once daily 4:30pm |
| WARNING | Slippage | "Fill slippage 12 bps on TLT IC (budget: 5 bps)" | 5-min cooldown |
| WARNING | Event gate | "FOMC tomorrow — scaling to 0.6x" | Per event |
| WARNING | Strategy stale | "EXP-1630 no signal for 48h" | 6-hour cooldown |
| CRITICAL | DD breach | "DD at 8.2% — approaching 10% circuit breaker" | Immediate |
| CRITICAL | Kill switch | "CIRCUIT BREAKER: DD 10.3% — leverage cut to 0.5x" | Immediate |
| CRITICAL | Strategy error | "Cross-Asset Pairs: exception in signal generation" | Immediate |
| CRITICAL | Broker down | "Alpaca API unreachable for 5 minutes" | 5-min cooldown |

### 5.2 Alert Format

```
🔴 CRITICAL — DD Circuit Breaker Activated
Portfolio DD: 10.3% (threshold: 10%)
Leverage: 1.4x → 0.5x
Action: No new entries until DD < 5%
Positions: 6 open, no liquidation (stops handle exits)
Time: 2026-05-15 14:22 ET
```

```
🟢 INFO — Daily Summary
Date: 2026-05-15
Daily P&L: +$340 (+0.34%)
Portfolio DD: 2.1%
Leverage: 1.4x | VIX: 16.2
Positions: 4 | Hedge: active (SPY puts)
Strategies: 1220 ✅ | Pairs ✅ | TLT-IC ✅ | VolTerm ✅
```

### 5.3 Operator Commands (via Telegram)

| Command | Action |
|---------|--------|
| `/status` | Current portfolio state, P&L, positions |
| `/kill` | Emergency halt — cancel all orders, freeze entries |
| `/kill <strategy>` | Halt one strategy only |
| `/resume` | Re-arm after manual halt |
| `/positions` | List all open positions with Greeks |
| `/risk` | Current risk state: DD, leverage, VIX, event gates |

---

## 6. Failover

### 6.1 Strategy Error

```
Strategy throws exception during signal generation
  │
  ├─→ Log full traceback to strategy log file
  ├─→ Telegram CRITICAL alert with error message
  ├─→ Skip strategy for this cycle (other strategies continue)
  ├─→ Existing positions remain open (managed by stops/expiry)
  ├─→ Circuit breaker: after 3 consecutive failures → disable strategy
  └─→ Operator must /resume after investigation
```

**Design principle:** A single strategy failure must never bring down the portfolio.
The orchestrator catches exceptions per-strategy and continues processing the rest.

### 6.2 Data Feed Failure

```
Polygon/Alpaca data feed unreachable
  │
  ├─→ Retry 3 times with exponential backoff (5s, 15s, 45s)
  ├─→ If all retries fail:
  │     ├─→ Use last known prices (cache in data/price_cache.db)
  │     ├─→ Telegram CRITICAL: "Data feed down — using stale prices"
  │     ├─→ Reduce leverage to 0.5x (stale data = uncertain risk)
  │     ├─→ No new entries permitted
  │     └─→ Existing positions managed by broker-side stops
  └─→ On recovery: resume normal operation, log gap duration
```

**Staleness limits:**
- Prices >15 min old: warning, reduce to 0.8x leverage
- Prices >60 min old: critical, reduce to 0.5x, no new entries
- Prices >4 hours old: halt all activity, alert operator

### 6.3 Broker API Down

```
Alpaca API unreachable (shared/circuit_breaker.py manages)
  │
  ├─→ Circuit breaker: closed → open after 5 consecutive failures
  ├─→ When open: all orders queued locally (not submitted)
  ├─→ Telegram CRITICAL: "Broker API down — orders queued"
  ├─→ After 60 seconds: attempt half-open (single test call)
  │     ├─→ Success: flush queued orders, resume
  │     └─→ Failure: remain open, retry in 60s
  └─→ If down >30 minutes: operator page, manual intervention
```

**Broker-side protection (always active regardless of our system):**
- Alpaca enforces per-account position limits
- Alpaca enforces PDT and margin rules
- Options auto-exercise/expire at expiration
- GTC orders persist on broker side even if our system is down

### 6.4 System Crash / Restart

```
Process dies unexpectedly
  │
  ├─→ Cron relaunches at next scheduled run (9:31 AM or 3:50 PM)
  ├─→ On startup:
  │     ├─→ Load state from SQLite (data/portfolio_state.db)
  │     ├─→ Reconcile positions: local state vs broker API
  │     ├─→ If mismatch: log + alert, use broker as source of truth
  │     └─→ Resume normal operation
  └─→ Positions survive restart (they live at the broker)
```

**State persistence:**
- All position state in SQLite (`data/pilotai_ultimate.db`)
- Equity curve, trade history, risk state persisted per cycle
- On crash, worst case is one missed signal cycle (not a position loss)

### 6.5 Failover Matrix

| Failure | Impact | Recovery Time | Data Loss |
|---------|--------|---------------|-----------|
| Strategy exception | One strategy skipped | Next cycle (minutes) | None |
| Data feed down | Stale prices, reduced leverage | Auto on recovery | None |
| Broker API down | Orders queued | Auto on recovery (60s) | None |
| System crash | Missed signal cycle | Next cron (hours) | None |
| Database corruption | State mismatch | Manual reconciliation | Position state |
| Network outage | All external services down | Manual + broker | None (broker has positions) |

---

## 7. Pre-Launch Checklist

### Phase 0: Infrastructure (Week 0)

- [ ] Install cron jobs for daily scanner + data update on production host
- [ ] Create `.env.ultimate` with Alpaca paper credentials
- [ ] Create `data/pilotai_ultimate.db` SQLite database
- [ ] Configure Telegram bot + channel, test message delivery
- [ ] Verify Python environment: all dependencies installed, tests passing
- [ ] Verify IronVault `options_cache.db` accessible and current

### Phase 1: Paper Trading Validation (Weeks 1–8)

- [ ] Deploy all 4 strategies in paper mode (dry_run=True in live_bridge)
- [ ] **Week 1–2:** Verify signals match backtest expectations
  - [ ] EXP-1220: crisis score tracks VIX movements correctly
  - [ ] Cross-Asset Pairs: z-score signals on GLD/TLT generate expected entries
  - [ ] TLT ICs: monthly IC entries at correct strikes/expirations
  - [ ] Vol Term Structure: contango signals align with VIX term data
- [ ] **Week 3–4:** Verify execution quality
  - [ ] Fill rate >95%
  - [ ] Avg slippage <5 bps
  - [ ] Order routing (TWAP/VWAP) working correctly
  - [ ] Position reconciliation: paper engine matches broker state
- [ ] **Week 5–6:** Verify risk management
  - [ ] Risk overlay layers firing correctly (check each layer independently)
  - [ ] Event gates: confirm FOMC/CPI/NFP scaling applied on correct dates
  - [ ] Position stops: verify stop triggers execute at correct thresholds
  - [ ] DD circuit breaker: inject synthetic DD to test activation/recovery
- [ ] **Week 7–8:** Full system validation
  - [ ] Continuous operation >14 days without manual intervention
  - [ ] Telegram alerts: all categories received and formatted correctly
  - [ ] Daily P&L tracking matches broker-reported values (±$5 tolerance)
  - [ ] No orphan positions, no stale signals, no uncaught exceptions

**Go/No-Go Criteria for Paper → Live:**

| Metric | Threshold | Source |
|--------|-----------|--------|
| Paper trading days | ≥ 56 (8 weeks) | Trade log |
| Paper trades executed | ≥ 30 | Trade log |
| Fill rate | ≥ 90% | Fill monitor |
| Max paper DD | < 15% | Equity curve |
| Signal-to-backtest alignment | > 70% of signals match | Manual review |
| System uptime | > 95% | Health monitor |
| Zero critical bugs | 0 unresolved | Issue tracker |
| Operator comfort | Manual sign-off | Carlos |

### Phase 2: Stress Testing (Week 6–8, parallel with paper)

- [ ] Run Monte Carlo stress test (10K simulations): confirm 0% ruin probability
- [ ] Run historical crisis scenarios through risk overlay:
  - [ ] COVID-2020: portfolio DD < 20% with hedges active
  - [ ] 2022 Bear: portfolio DD < 15%
  - [ ] Flash crash: recovery within 5 days
  - [ ] VIX spike (VIX > 50): leverage cuts to <0.5x automatically
- [ ] Verify DD circuit breaker fires at exactly 10% DD
- [ ] Verify kill switch halts all activity within 1 second
- [ ] Run 24-hour soak test: continuous operation with simulated data gaps

### Phase 3: Seed Deployment (Week 9–12)

- [ ] Switch from paper to live Alpaca account
- [ ] Start with **$25K at 1.0x leverage** (conservative)
- [ ] First 2 weeks: monitor every trade manually
- [ ] Verify live fills match paper fill quality (±2 bps)
- [ ] Verify Greeks and margin calculations match broker reports
- [ ] Confirm Telegram alerts for every entry/exit
- [ ] Weekly review: P&L, DD, slippage, system health

### Phase 4: Growth (Week 13+)

- [ ] After 4 weeks of stable live operation:
  - [ ] Increase to $50K
  - [ ] Enable 1.2x leverage
- [ ] After 8 weeks:
  - [ ] Increase to full allocation
  - [ ] Enable target 1.6x leverage
  - [ ] Add XLI ICs and TLT-XLF pair if validated
- [ ] Quarterly reviews:
  - [ ] Walk-forward re-optimization of weights
  - [ ] Backfill IronVault with new option data
  - [ ] Review and update risk thresholds

---

## Appendix A: Configuration Files

### A.1 Production Environment (`.env.ultimate`)

```bash
# Broker
ALPACA_API_KEY=<live_key>
ALPACA_API_SECRET=<live_secret>
ALPACA_BASE_URL=https://api.alpaca.markets

# Data
POLYGON_API_KEY=<polygon_key>

# Alerting
TELEGRAM_BOT_TOKEN=<bot_token>
TELEGRAM_CHAT_ID=<chat_id>

# Portfolio
EXPERIMENT_ID=ULTIMATE
STARTING_CAPITAL=100000
TARGET_LEVERAGE=1.6
MAX_DRAWDOWN_PCT=0.12

# Risk Overlay
ENABLE_DYNAMIC_LEVERAGE=true
ENABLE_TAIL_HEDGE=true
ENABLE_EVENT_GATES=true
ENABLE_POSITION_STOPS=true
ENABLE_DD_CIRCUIT_BREAKER=true
HEDGE_ANNUAL_COST_PCT=2.0
DD_BREAKER_THRESHOLD=0.10
DD_RECOVERY_THRESHOLD=0.05
```

### A.2 Cron Schedule

```crontab
# Data update (6:00 AM ET)
0 10 * * 1-5 cd /path/to/pilotai-credit-spreads && ./scripts/daily_data_update.sh >> logs/data.log 2>&1

# Morning signal run (9:31 AM ET)
31 13 * * 1-5 cd /path/to/pilotai-credit-spreads && python3 -m compass.live_bridge --config .env.ultimate >> logs/signals.log 2>&1

# Intraday monitor (every 5 min, market hours)
*/5 13-20 * * 1-5 cd /path/to/pilotai-credit-spreads && python3 -m compass.prod_monitor --config .env.ultimate >> logs/monitor.log 2>&1

# Afternoon close run (3:50 PM ET)
50 19 * * 1-5 cd /path/to/pilotai-credit-spreads && python3 -m compass.live_bridge --mode close --config .env.ultimate >> logs/close.log 2>&1

# Daily reconciliation (4:15 PM ET)
15 20 * * 1-5 cd /path/to/pilotai-credit-spreads && python3 -m compass.paper_reconciler --config .env.ultimate >> logs/reconcile.log 2>&1

# Weekly health report (Sunday 6 PM ET)
0 22 * * 0 cd /path/to/pilotai-credit-spreads && python3 -m compass.prod_monitor --weekly-report >> logs/weekly.log 2>&1
```

### A.3 Risk Overlay Config

```python
RiskOverlayConfig(
    # Dynamic leverage
    target_leverage=1.8,
    min_leverage=0.3,
    vix_calm=15.0,
    vix_crisis=35.0,
    leverage_smoothing_halflife=5,

    # Tail hedge
    hedge_annual_cost_budget_pct=2.0,
    normal_hedge_ratio=0.30,
    crisis_hedge_ratio=0.80,

    # Event gates
    fomc_scaling={5: 1.0, 4: 0.9, 3: 0.8, 2: 0.7, 1: 0.6, 0: 0.5},
    cpi_scaling={2: 1.0, 1: 0.75, 0: 0.65},
    nfp_scaling={2: 1.0, 1: 0.80, 0: 0.75},

    # Position stops
    stop_loss_pct=0.03,
    trailing_stop_pct=0.05,

    # DD circuit breaker
    dd_breaker_threshold=0.10,
    dd_breaker_leverage=0.50,
    dd_recovery_threshold=0.05,
)
```

---

## Appendix B: Key File Paths

| Component | File | Tests |
|-----------|------|-------|
| Risk overlay | `compass/risk_overlay.py` | `tests/test_risk_overlay.py` (71) |
| Tail risk hedge | `compass/tail_risk_hedge.py` | `tests/test_tail_risk_hedge.py` |
| Dynamic leverage | `compass/dynamic_leverage.py` | `tests/test_dynamic_leverage.py` |
| Event gates | `compass/events.py` | `tests/test_event_gate.py` |
| Paper engine | `compass/paper_trading_engine.py` | 57 tests |
| Production monitor | `compass/prod_monitor.py` | 87 tests |
| Live bridge | `compass/live_bridge.py` | — |
| Order manager | `compass/order_manager.py` | — |
| Execution algo | `compass/execution_algo.py` | — |
| Circuit breaker | `shared/circuit_breaker.py` | — |
| Credentials | `shared/credentials.py` | — |
| Telegram | `shared/telegram_alerts.py` | — |
| Ultimate Portfolio results | `reports/ultimate_portfolio.json` | — |
| Walk-forward results | `reports/ultimate_portfolio_walkforward.json` | — |
| EXP-1630 optimization | `reports/exp1630_optimization.json` | — |

---

## Appendix C: Emergency Runbook

### C.1 Portfolio DD > 10%

1. **Automatic:** DD circuit breaker fires → leverage cut to 0.5x
2. **Verify:** Check Telegram for CRITICAL alert
3. **Assess:** Is DD from market move or system error?
4. **If market:** Let positions expire or hit stops. Do NOT panic sell.
5. **If error:** `/kill` all strategies. Investigate. Fix. Reconcile. `/resume`

### C.2 Strategy Producing Wrong Signals

1. `/kill <strategy>` via Telegram
2. Check signal logs: `logs/signals.log`
3. Compare signal against manual calculation
4. If data issue: check `data/options_cache.db` freshness
5. If code bug: fix, test, redeploy, `/resume`

### C.3 Broker API Outage

1. Circuit breaker handles automatically (orders queued)
2. If >30 minutes: check Alpaca status page
3. Existing positions are safe (live at broker, not in our system)
4. On recovery: reconcile positions before resuming

### C.4 Complete System Failure

1. **Positions are safe** — they live at the broker
2. Options expire worthless or exercised on broker side
3. Restart system, reconcile from broker state
4. Worst case: miss one signal cycle (a few hours of inaction, not a loss)
