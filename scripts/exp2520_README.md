# EXP-2520 — Paper Trading Deployment Package

**Final paper-trading deployment for the 7-stream North-Star portfolio.**

Supersedes the EXP-2290 (v6) launch. Uses the EXP-2410 production config
as the single source of truth. Designed to run on a Mac Studio under
`launchd`, with optional Telegram alerting.

## Components

| File | Purpose |
|---|---|
| `configs/exp2410_production_paper.yaml` | The config (EXP-2410). Not in this package — referenced. |
| `scripts/launch_exp2520.sh` | Bash launcher: smoke / dry / start / daemon / stop / status / logs / report / dashboard / close-all / install-launchd |
| `scripts/exp2520_monitor.py` | 5-min health poller + Telegram alerts. Evaluates the 3% trailing-DD circuit breaker against rolling peak. Read-only — never submits orders. |
| `scripts/exp2520_risk_dashboard.py` | Live HTML risk dashboard at `reports/exp2520/risk_dashboard.html`. Auto-refresh 2 minutes. |
| `scripts/exp2520_daily_report.py` | End-of-day P&L + per-sleeve attribution. Writes `reports/exp2520/daily/YYYY-MM-DD.{html,json}`. |

## Environment

Create `.env.exp2520` at the repo root with:

```bash
ALPACA_API_KEY_PAPER=PK...
ALPACA_API_SECRET_PAPER=...
TELEGRAM_BOT_TOKEN=123456:ABC...     # optional
TELEGRAM_CHAT_ID=-100xxxxxxxxxx      # optional
```

## Bring-up sequence (Mac Studio)

```bash
# 1. Validate env + config + imports
./scripts/launch_exp2520.sh smoke

# 2. Dry-run (scan loop, no orders submitted)
./scripts/launch_exp2520.sh dry

# 3. Start in foreground first to watch the engine come alive
./scripts/launch_exp2520.sh start

# 4. Once stable, switch to background daemon
./scripts/launch_exp2520.sh daemon

# 5. Install macOS LaunchAgent for auto-start on reboot
./scripts/launch_exp2520.sh install-launchd

# 6. Daily operations
./scripts/launch_exp2520.sh status
./scripts/launch_exp2520.sh logs
./scripts/launch_exp2520.sh report
./scripts/launch_exp2520.sh dashboard
```

## What the 3 background components do

### 1. `compass.paper_engine` (the engine)

The only writer to the Alpaca paper broker. Loads the EXP-2410 config,
runs the rebalance cadence for each sleeve, applies the Ledoit-Wolf
risk-parity allocator every 63 days, vol-targets to 15% daily with a
13× hard scale cap, and wires the T+V entry overlay into the EXP-1220
sleeve only (via `CreditSpreadStrategy.entry_overlay`).

### 2. `scripts/exp2520_monitor.py` (5-min poller)

Read-only. Every 5 minutes it:

- Fetches Alpaca account equity and open positions
- Reads engine `state.json` for leverage, scale factor, last weights
- Evaluates the **3% trailing-DD circuit breaker** against the rolling
  peak stored in state.json (EXP-2370 spec)
- Sends Telegram alerts on position changes, circuit-breaker trips,
  VIX spikes, and leverage-cap breaches
- Writes `logs/exp2520/health.json` for the dashboard and the launcher
  `status` command

### 3. `scripts/exp2520_risk_dashboard.py` (HTML dashboard)

Re-renders `reports/exp2520/risk_dashboard.html` every 2 minutes from
the monitor's `health.json` and the engine's `state.json`. The HTML
auto-refreshes every 2 minutes in the browser so you can point a
spare monitor at it.

## Circuit-breaker spec (EXP-2370)

| Trigger | Threshold | Action |
|---|---|---|
| Soft | equity ≤ rolling peak − **3%** | reduce leverage to 50% of target |
| Hard | equity ≤ rolling peak − **6%** | close every position + halt 24h |
| Daily override | any single day down ≥ 2% | soft trigger immediately |
| Recovery | equity ≥ rolling peak − 1.5% | resume normal leverage |

Rolling peak is tracked in `logs/exp2520/state.json` by the monitor
and is never decreased. The engine consumes the breaker state on each
rebalance decision.

## Paper-to-live promotion criteria (Charles signs off)

From `configs/exp2410_production_paper.yaml → promotion:`:

- `paper_duration_days >= 90`
- `paper_sharpe_vs_backtest_ratio >= 0.70`
- `max_realised_drawdown_pct <= 5`
- `circuit_breaker_trips_max = 1` (soft), `0` (hard)
- `deviation_alert_count_max = 3`
- Clean 90-day test: Sharpe ≥ 3.0, DD ≤ 8%
- Zero Rule-Zero violations

## Honest targets (EXP-2280 walk-forward, the only un-biased number)

- Pooled OOS Sharpe: **4.43**
- Per-fold mean / median: **5.97 / 6.26**
- Frac folds ≥ 6: **60%**
- Pooled OOS CAGR: **170.4%**
- Pooled OOS Max DD: **24.4%** → tightened via 3% circuit breaker

The 13–33 Sharpe figures from full-sample optimisers (EXP-2200, EXP-2360)
are look-ahead biased and **NOT** the production target. The walk-
forward distribution above is.

## Rule Zero

Every executable quote traces to Alpaca live, IronVault `option_daily`,
Yahoo `^VIX`/`^VIX3M`, or federalreserve.gov. Paper-mode fills apply a
calibrated slippage model around **real** quotes. Missing quote = skip
the trade. Never extrapolate.
