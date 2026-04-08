# EXP-2920 — Paper-Trading Monitoring Dashboard Specification

**Status:** design-complete · reference code + tests shipped
**Consumers:** Charles (operator), Carlos (sign-off), risk/investor review
**Upstream:** EXP-2520 deployment package, EXP-2620 Alpaca connector,
MASTERPLAN v11 abort triggers

---

## 1. Purpose

When Carlos authorises paper trading on Alpaca, the operator needs a
single read-only daily view of every metric that gates promotion to
live. This spec defines that view, the data flow behind it, the
alert rules, and the four MASTERPLAN abort triggers that flatten the
book immediately when breached.

The monitoring system is **read-only**. The paper engine
(`compass.paper_engine`) is the only component that submits orders.
The monitor polls state and alerts.

## 2. Seven monitoring dimensions

| # | Dimension | Metric(s) | Update cadence | Source |
|---|---|---|---|---|
| 1 | **Daily P&L per stream** | `pnl_today`, `pnl_mtd`, `pnl_ytd`, `open_positions`, `weight` | 5 min | Alpaca activities + engine `state.json[per_stream_pnl]` |
| 2 | **Rolling Sharpe ratio** | 30d / 60d / 90d pooled-OOS Sharpe | daily (EOD) | computed from daily return series in `state.json[return_history]` |
| 3 | **Max drawdown tracker** | trailing DD vs rolling peak (never decreases), DD time-in-state | 5 min | computed from `equity` vs `rolling_peak_equity` |
| 4 | **VIX ladder exposure** | current VIX, ladder zone (`calm`/`stress`/`crisis`/`flat`), multiplier applied | 5 min | Yahoo `^VIX` close + `vix_ladder_state(vix)` |
| 5 | **Order fill quality** | `n_fills_today`, `mean_deviation_cents`, `p90_deviation_cents`, `frac_over_5c`, `stale_quote_count` | on every fill | Alpaca fills compared to NBBO mid at submission time |
| 6 | **Correlation across streams** | rolling 60d pairwise Pearson, `max_abs_offdiag`, `median_abs` | daily (EOD) | `correlation_matrix(per_stream_return_hist, 60)` |
| 7 | **Abort trigger panel** | 4 MASTERPLAN triggers with OK / WARN / CRITICAL | 5 min | `AbortTriggerEvaluator.evaluate()` |

Every metric lands on the single canonical `MonitorTick` object
defined in `compass.exp2920_monitor_core`.

## 3. MASTERPLAN v11 abort triggers

Copied verbatim from `MASTERPLAN.md` lines 164-170. Any one flattens
the book immediately.

| # | Code | Condition | Action |
|---|---|---|---|
| 1 | `drawdown_12pct_hard_circuit` | Trailing DD ≥ 12% from rolling peak | Close all + halt 24h |
| 2 | `rolling_4w_sharpe_lt_2_for_5d` | Rolling 20-day Sharpe < 2.0 for 5 consecutive days | Close all + halt until Carlos sign-off |
| 3 | `alpaca_fill_deviation_gt_5c_on_20pct_orders` | Alpaca fills deviate from NBBO > 5c on > 20% of today's orders | Close all + fail over to IBKR paper |
| 4 | `rule_zero_violation` | Any synthetic fill or extrapolated quote in the last 24h (zero tolerance) | Close all + halt + incident review |

The evaluator is stateless except for the 5-consecutive-breach
counter for trigger 2, which persists in `state.json`.

### Severity ladder

| Severity | Trigger 1 | Trigger 2 | Trigger 3 | Trigger 4 |
|---|---|---|---|---|
| **OK** | DD < 10% | Sharpe ≥ 2.0 OR warmup | frac ≤ 20% OR no fills | 0 violations |
| **WARNING** | 10% ≤ DD < 12% (implicit via EXP-2370 3%/6% breaker) | Sharpe < 2.0, day 1-4 of breach window | — | — |
| **CRITICAL** | DD ≥ 12% | Sharpe < 2.0 for 5th consecutive day | > 20% of fills deviated > 5c | ≥ 1 violation in 24h |

## 4. Data flow

```
                      ┌─────────────────────┐
                      │  compass.paper_engine │  ← only writer to broker
                      └──────────┬──────────┘
                                 │ writes
                                 ▼
                    ┌──────────────────────────┐
                    │ logs/exp2920/state.json   │
                    │   equity, leverage,       │
                    │   scale_factor,           │
                    │   per_stream_pnl,         │
                    │   return_history,         │
                    │   fills_today             │
                    └──────────┬───────────────┘
                               │ reads
              ┌────────────────┼─────────────────┐
              ▼                ▼                 ▼
  ┌──────────────────┐  ┌────────────────┐  ┌────────────────────┐
  │ MetricAggregator │  │ scripts/       │  │ scripts/           │
  │ .build_tick()    │  │ exp2520_daily_ │  │ exp2520_risk_      │
  │ (this module)    │  │ report.py      │  │ dashboard.py       │
  └────────┬─────────┘  │ EOD HTML/JSON  │  │ HTML auto-refresh  │
           │            └────────────────┘  └─────────┬──────────┘
           ▼                                          │
  ┌──────────────────┐                                │
  │ AbortTrigger     │                                │
  │ Evaluator        │                                │
  │ (4 verdicts)     │                                │
  └────────┬─────────┘                                │
           │                                          │
           ▼                                          ▼
  ┌──────────────────┐          ┌───────────────────────────┐
  │ scripts/         │          │ reports/exp2520/          │
  │ exp2520_         │          │   risk_dashboard.html     │
  │ monitor.py       │          │   daily/YYYY-MM-DD.html   │
  │  Telegram alerts │          └───────────────────────────┘
  └──────────────────┘
```

## 5. Alert rules

The monitor emits Telegram alerts with strict level/code/cooldown so
that the operator is never spammed:

| Level | Trigger | Cooldown |
|---|---|---|
| `INFO` | new position opened / closed; EOD summary | 5 min |
| `WARNING` | trailing DD ≥ 2% (approaching 3% soft breaker); leverage ≥ 10×; VIX ≥ 25; rolling-Sharpe breach day 1–4 | 60 s |
| `CRITICAL` | any abort trigger `CRITICAL`; circuit-breaker hard trip; leverage ≥ 13× hard cap; VIX ≥ 35 | 0 (immediate) |

`CRITICAL` alerts automatically page the operator and include the
full `AbortVerdict` JSON payload so the incident review can begin
without round-tripping through the dashboard.

## 6. Reference implementation

Three files shipped with this experiment:

| File | Purpose | Lines |
|---|---|---|
| `compass/exp2920_monitor_core.py` | `MetricAggregator`, `AbortTriggerEvaluator`, `MonitorTick`, `StreamPnl`, `FillQuality`, `CorrelationSnapshot`, `vix_ladder_state()`, `rolling_sharpe()`, `running_max_drawdown()`, `correlation_matrix()` | ~430 |
| `tests/test_exp2920_monitor.py` | 29 unit tests covering every monitoring dimension + every abort trigger branch | ~250 |
| `compass/exp2920_monitoring_spec.md` | this document | ~200 |

### Test coverage matrix (29/29 pass)

| Area | Tests | What is verified |
|---|---|---|
| VIX ladder | 9 | every zone maps to the right multiplier |
| Rolling Sharpe | 3 | warmup → None, positive series → positive Sharpe, zero-vol → 0.0 |
| Max drawdown | 2 | monotone up → 0; peak→trough captures correct DD |
| Correlation matrix | 2 | warmup returns empty; 60d real returns → correct pearson |
| Abort trigger 1 | 2 | fires at 12.5%, OK at 11.9% |
| Abort trigger 2 | 4 | warmup OK, WARNING days 1–4, CRITICAL on day 5, resets when Sharpe recovers |
| Abort trigger 3 | 3 | fires at 25%, OK at 18%, OK when 0 fills |
| Abort trigger 4 | 2 | fires on 1 violation, OK at 0 |
| Integration | 2 | `build_tick` populates every field; `rolling_peak_equity` persists across ticks |

## 7. Integration with existing EXP-2520 / EXP-2620 stack

The monitoring core from this experiment slots into the already-
shipped deployment package without changing their public APIs:

| Component | Before EXP-2920 | After EXP-2920 |
|---|---|---|
| `scripts/exp2520_monitor.py` | hand-coded breaker eval | imports `AbortTriggerEvaluator` |
| `scripts/exp2520_risk_dashboard.py` | reads `health.json` directly | reads `MonitorTick.to_dict()` |
| `scripts/exp2520_daily_report.py` | ad-hoc metrics | uses `MetricAggregator` |
| `compass/exp2620_alpaca_connector.py` | own breaker logic | will be refactored to call `AbortTriggerEvaluator` in a follow-on |

The refactor is intentionally left for a follow-on experiment so
this one ships only the new core + tests + spec.

## 8. Rule Zero compliance

Every price read by the aggregator comes from **Alpaca live** (or
Yahoo `^VIX` for the ladder) — no synthetic fills. If a quote is
missing the metric is flagged `stale` and abort trigger 4 fires.
Tests explicitly verify the zero-tolerance path.

## 9. Deployment checklist (for Charles)

1. Confirm `ALPACA_API_KEY_PAPER` + `ALPACA_API_SECRET_PAPER` in env
2. Confirm `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in env (fallback to log)
3. Run `python -m pytest tests/test_exp2920_monitor.py` — must be 29/29
4. Run `./scripts/launch_exp2520.sh smoke` — must pass
5. Start the daemon: `./scripts/launch_exp2520.sh daemon`
6. Verify `logs/exp2920/monitor_state.json` appears within 5 min
7. Open `reports/exp2520/risk_dashboard.html` and confirm all 7
   monitored dimensions render
8. Manually trip each abort trigger in a dry-run to confirm the
   Telegram alerts arrive with the right severity

## 10. Promotion gate to live

Per `MASTERPLAN.md`:

> 1. 90 days paper
> 2. paper_sharpe_vs_backtest_ratio_min ≥ 0.70
> 3. Zero abort-trigger events in paper window
> 4. EXP-2670 checklist re-run returns OVERALL=GO
> 5. Secondary broker (IBKR Pro) account funded and wired as fallback

The monitoring core enforces gate #3 mechanically. Any
`CRITICAL` abort verdict in the 90-day window resets the clock to
day zero.

---

*Rule Zero: every metric in this spec reads from real broker data or
clearly labelled real market feeds. No synthetic pricing anywhere in
the monitoring path.*
