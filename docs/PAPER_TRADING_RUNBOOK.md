# EXP-880 Paper Trading Runbook

Operational guide for running the Crisis Hedge V2 Ultra-Safe paper trading
deployment.  This is the definitive reference for startup, monitoring,
incident response, and graduation to live trading.

---

## 1. Pre-Launch Checklist

Complete every item before the first session.

### Credentials & Environment

- [ ] Alpaca paper trading account created at <https://app.alpaca.markets>
- [ ] `.env.exp880` file created with:
  ```
  ALPACA_API_KEY=PK...
  ALPACA_API_SECRET=...
  POLYGON_API_KEY=...
  TELEGRAM_BOT_TOKEN=...    # optional but recommended
  TELEGRAM_CHAT_ID=...      # optional but recommended
  ```
- [ ] Verify Alpaca key is **paper** (not live): key starts with `PK`
- [ ] Polygon API key has options data access (Basic plan minimum)
- [ ] Telegram bot added to monitoring chat (test with `/start`)

### Configuration

- [ ] `configs/paper_exp880.yaml` reviewed — settings match EXP-880 backtest:
  - `paper_mode: true`
  - Base leverage: `2.0x`
  - Crisis hedge: `min_scale: 0.20`, DD delevering `2% → 7%`
  - Signal filter: `P >= 0.75` ensemble threshold
  - Hard DD stop: `12%`
  - Max positions: `8`
- [ ] Database path exists: `data/exp880/` directory created
- [ ] Log directory exists: `logs/` writable

### Dry Run

- [ ] Run system in dry-run mode (no order submission):
  ```bash
  python main.py scheduler --config configs/paper_exp880.yaml \
                           --env-file .env.exp880 \
                           --dry-run
  ```
- [ ] Verify: features computed without error
- [ ] Verify: regime detected and logged
- [ ] Verify: ML ensemble scores returned
- [ ] Verify: crisis hedge scale factor computed
- [ ] Verify: Telegram test alert received (if configured)
- [ ] Verify: no `ImportError` or `KeyError` in logs

### Model Readiness

- [ ] Ensemble models exist at expected paths (XGB + RF + ET)
- [ ] Model files are not stale (< 30 days old)
- [ ] `compass/online_retrain.py` retrain schedule configured
- [ ] Feature pipeline (`compass/features.py`) produces 27 features

---

## 2. Startup Procedure

### First-Time Launch

```bash
# 1. Activate environment
cd /home/node/.openclaw/workspace/pilotai-credit-spreads
source .venv/bin/activate  # or conda activate pilotai

# 2. Run database migrations (if any)
python -c "from compass.paper_trading_engine import PaperTradingEngine; print('OK')"

# 3. Start the scheduler
python main.py scheduler --config configs/paper_exp880.yaml \
                         --env-file .env.exp880

# 4. Verify first heartbeat in Telegram (within 1 min)
# 5. Verify log output: "EXP-880 paper trading started"
```

### Daily Restart (if needed)

```bash
# Check if already running
ps aux | grep "main.py scheduler"

# If not running, restart
nohup python main.py scheduler --config configs/paper_exp880.yaml \
                               --env-file .env.exp880 \
                               >> logs/exp880.log 2>&1 &

# Verify
tail -f logs/exp880.log
```

### After a Crash

1. Check `logs/exp880.log` for the error
2. Fix the root cause (see Section 5)
3. Restart using the daily restart procedure
4. Verify existing positions are still tracked in the database
5. Send manual Telegram message: "EXP-880 restarted after crash"

---

## 3. Daily Monitoring Tasks

### Morning (9:15 ET — before market open)

| Task | How | What to Check |
|------|-----|--------------|
| Process running | `ps aux \| grep scheduler` | Process exists |
| Last heartbeat | Telegram or `logs/exp880.log` | < 24h old |
| Regime status | Log: "Current regime:" | Makes sense vs market |
| VIX level | Any market data source | If VIX > 25: expect reduced sizing |
| Open positions | `data/exp880/pilotai_exp880.db` | Count ≤ 8 |
| Account equity | Alpaca dashboard | No unexpected changes |

### Intraday (check 2-3 times)

| Task | Trigger | Action |
|------|---------|--------|
| New trade alert | Telegram `TradeAlert` | Verify makes sense |
| Risk alert | Telegram `RiskAlert` | Check immediately (see §4) |
| Model alert | Telegram `ModelAlert` | Check within 1 hour |
| No alerts all day | End of day | Normal — not every day has trades |

### Evening (after 16:00 ET)

| Task | How |
|------|-----|
| Review day's trades | `SELECT * FROM trades WHERE date=today` |
| Check daily P&L | Alpaca dashboard → portfolio history |
| Log drawdown | Compare equity to high-water mark |
| Compare to backtest | Is daily return within expected range? |

---

## 4. Alert Response Procedures

### TradeAlert — New Trade Executed

**Severity:** Informational
**Response time:** Next convenient check

```
📊 EXP-880 TRADE: SELL SPY 540P/535P @ $2.15
  Signal: P=0.82, Regime: bull, Scale: 1.0
  Size: 3 contracts (confidence: high)
```

**Actions:**
1. Verify the trade appears in Alpaca dashboard
2. Check that sizing makes sense (3 contracts at 2× leverage)
3. If trade looks wrong (wrong ticker, absurd size), trigger kill switch

### RiskAlert — Risk Limit Approached

**Severity:** High
**Response time:** Within 15 minutes

```
⚠️ RISK: DD at 4.2% (delever start: 2%, max: 7%)
  Scale: 0.63, Positions: 6, VIX: 22
```

**Actions:**
1. Check Alpaca dashboard for current equity
2. If DD < 7%: monitoring only, system is auto-delevering
3. If DD 7-12%: system at maximum delever, watch closely
4. If DD > 12%: **hard stop triggered automatically** — verify all positions closed

### RiskAlert — Kill Switch Triggered

**Severity:** Critical
**Response time:** Immediately

```
🚨 KILL SWITCH: DD=12.3% exceeds 12% hard stop
  Action: ALL POSITIONS CLOSED
```

**Actions:**
1. Verify all positions are actually closed in Alpaca
2. If any positions remain open, close them manually
3. Do NOT restart until root cause analysis is complete
4. Document the event in `experiments/EXP-880-paper/incidents/`

### ModelAlert — Model Health Issue

**Severity:** Medium
**Response time:** Within 1 hour

```
🔧 MODEL: Feature drift detected — rvol_20d z-score 3.2
  Ensemble staleness: 15 days (retrain at 30)
```

**Actions:**
1. If feature drift: check if market conditions changed (VIX spike, etc.)
2. If model stale (> 30 days): trigger manual retrain:
   ```bash
   python -c "from compass.online_retrain import OnlineRetrainer; \
              OnlineRetrainer().retrain()"
   ```
3. If ensemble disagreement high: reduce position size manually until resolved

---

## 5. Failure Scenarios and Recovery

### Scenario 1: Process Crash

**Symptoms:** No Telegram heartbeat, no log output
**Cause:** Python exception, OOM, server restart

**Recovery:**
1. Check logs: `tail -100 logs/exp880.log`
2. If OOM: increase memory or reduce position count
3. If exception: fix code, restart
4. Existing positions are safe (held at Alpaca, not in-memory)
5. Restart using §2 procedure

### Scenario 2: Alpaca API Outage

**Symptoms:** `ConnectionError` or `HTTPError` in logs
**Cause:** Alpaca maintenance, rate limiting, network issue

**Recovery:**
1. Check Alpaca status: <https://status.alpaca.markets>
2. If Alpaca is down: wait, system will retry automatically
3. If rate-limited: reduce scan frequency in config
4. Existing positions are safe (held at Alpaca)
5. After recovery: verify position state matches database

### Scenario 3: Polygon Data Outage

**Symptoms:** `No options data` or `PolygonError` in logs
**Cause:** Polygon maintenance, API key issue

**Recovery:**
1. Check Polygon status
2. System should skip signal generation (no trades, not crash)
3. If prolonged (> 4h): manually check VIX and regime
4. After recovery: system resumes automatically

### Scenario 4: Model Degradation

**Symptoms:** Win rate drops below 50% over 20+ trades
**Cause:** Market regime shift, feature drift, model staleness

**Recovery:**
1. Check current regime — are we in a regime the model handles poorly?
2. If regime = crash: system should already be at min_scale (0.20)
3. If feature drift: trigger manual retrain
4. If persistent (> 30 trades at < 50% WR): pause trading, investigate
5. Compare paper results to backtest expectations per regime

### Scenario 5: Flash Crash / VIX Spike

**Symptoms:** VIX > 35, multiple risk alerts
**Cause:** Market event

**Recovery:**
1. Crisis hedge activates automatically at VIX > 25
2. At VIX > 35: system at minimum scale (0.20)
3. At VIX > 50: put overlay activates
4. Do NOT manually override — let the hedge work
5. Monitor DD — if approaching 12%, kill switch triggers
6. After event: wait for recovery signal (10d momentum + VIX < 22)

---

## 6. Weekly Review Process

Every Friday after market close:

### Performance Review

```sql
-- Weekly P&L
SELECT SUM(realized_pnl) as weekly_pnl,
       COUNT(*) as n_trades,
       AVG(CASE WHEN realized_pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate
FROM trades
WHERE date >= date('now', '-7 days');
```

| Metric | Check Against | Action if Outside |
|--------|--------------|-------------------|
| Weekly return | ±3% of backtest weekly avg | Investigate if 2 consecutive weeks outside |
| Win rate | > 55% rolling 20 trades | Flag if < 50% for 2 weeks |
| Max DD this week | < 5% | Reduce leverage if > 5% |
| Avg position size | Within 20% of target | Recalibrate if drifting |
| Regime accuracy | Matches reality | Manual override if obviously wrong |

### Model Health Check

- Ensemble agreement rate (should be > 70%)
- Feature drift z-scores (all should be < 2.5)
- Model age (retrain if > 30 days)
- Signal quality: IC of recent predictions vs outcomes

### Comparison to Backtest

- Plot cumulative paper P&L vs backtest equity curve
- Compute tracking error (should be < 5% annualised)
- If paper is > 2× worse than backtest for 3 weeks: pause and investigate

### Document in Weekly Log

Create `experiments/EXP-880-paper/weekly/YYYY-WW.md` with:
- P&L summary
- Notable trades (best/worst)
- Regime classification accuracy
- Any incidents or alerts
- Decision: continue / adjust / pause

---

## 7. Victory Conditions for Live Trading

Graduate from paper to live when ALL of the following are met:

### Minimum Duration
- [ ] 30 calendar days of paper trading completed
- [ ] At least 20 round-trip trades executed

### Performance
- [ ] Cumulative paper return > 0% (profitable)
- [ ] Paper Sharpe > 1.5 (annualised)
- [ ] Paper max DD < 12%
- [ ] Win rate > 50% over all trades

### Execution Quality
- [ ] Signal-to-fill latency < 5 minutes (95th percentile)
- [ ] No missed signals due to system failures
- [ ] Fill prices within 2% of expected (slippage acceptable)

### Tracking
- [ ] Paper P&L tracks backtest expectations within 50% tolerance
- [ ] No regime classification errors that caused material losses
- [ ] Crisis hedge triggered correctly during any VIX > 25 event

### Operational
- [ ] Zero unrecovered system crashes in last 14 days
- [ ] All weekly reviews completed and documented
- [ ] Telegram alerts functioning for full duration
- [ ] Kill switch tested (manually triggered and verified)

### Approval
- [ ] Weekly review log reviewed by Carlos
- [ ] Written approval to proceed to live
- [ ] Live account funded with initial allocation
- [ ] Live config created (`configs/live_exp880.yaml` with `paper_mode: false`)

---

## 8. Kill Switch Procedure

### Automatic Kill Switch

The `PaperTradingEngine` triggers automatically when:
- Drawdown exceeds 12% hard stop
- `CrisisHedgeController` scale reaches 0.0

When triggered:
1. All open positions are closed at market
2. `TelegramAlerter.kill_switch_alert()` fires
3. No new trades are placed
4. System enters "halted" state

### Manual Kill Switch

If you need to stop immediately:

```bash
# Option 1: Graceful shutdown
kill -SIGTERM $(pgrep -f "main.py scheduler")

# Option 2: Close all positions via Alpaca API
python -c "
from alpaca.trading.client import TradingClient
client = TradingClient('YOUR_KEY', 'YOUR_SECRET', paper=True)
client.close_all_positions(cancel_orders=True)
print('All positions closed')
"

# Option 3: Nuclear — close everything via Alpaca dashboard
# Go to https://app.alpaca.markets → Positions → Close All
```

### After Kill Switch

1. **Do NOT restart immediately**
2. Document what happened in `experiments/EXP-880-paper/incidents/`
3. Analyse: was the kill switch correct? Or false alarm?
4. If false alarm: fix the trigger, restart
5. If real drawdown: wait for market stabilisation, review with Carlos
6. Resume only after root cause is understood and documented

---

## Appendix: Key File Paths

| File | Purpose |
|------|---------|
| `configs/paper_exp880.yaml` | Strategy configuration |
| `.env.exp880` | Credentials (gitignored) |
| `data/exp880/pilotai_exp880.db` | Trade database |
| `logs/exp880.log` | System logs |
| `compass/paper_trading_engine.py` | `PaperTradingEngine` class |
| `compass/crisis_hedge.py` | `CrisisHedgeController`, `CrisisHedgeConfig` |
| `compass/telegram_alerter.py` | `TelegramAlerter`, `TradeAlert`, `RiskAlert` |
| `compass/online_retrain.py` | `OnlineRetrainer` |
| `compass/ensemble_signal_model.py` | 3-model ensemble |
| `compass/ml_strategy.py` | `RegimeModelRouter`, `ShadowEnsemble` |
| `compass/features.py` | Feature engineering pipeline |
| `compass/regime.py` | Regime classifier |
| `main.py` | Entry point (`scheduler` command) |

## Appendix: Key Parameters (EXP-880 V2 Ultra-Safe)

| Parameter | Value | Source |
|-----------|-------|--------|
| `paper_mode` | `true` | Safety |
| `base_leverage` | `2.0x` | EXP-840/980 |
| `ensemble_threshold` | `P >= 0.75` | EXP-860 |
| `min_scale` | `0.20` | EXP-880 |
| `dd_delever_start` | `2%` | EXP-880 |
| `dd_delever_max` | `7%` | EXP-880 |
| `hard_dd_stop` | `12%` | EXP-890 |
| `vix_reduce` | `25` | EXP-880 |
| `vix_minimum` | `35` | EXP-880 |
| `vix_full_hedge` | `50` | EXP-880 |
| `max_positions` | `8` | Risk management |
| `recovery_signal` | `10d momentum + VIX < 22` | EXP-880 |
| `recovery_ramp` | `20 days` | EXP-880 |
