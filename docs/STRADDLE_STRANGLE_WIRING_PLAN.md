# Straddle/Strangle Wiring Plan

**Status:** Planning
**Branch:** `maximus/champion-config`
**Scope:** Wire `strategies/straddle_strangle.py` into the live paper-trading scan loop (EXP-401)
**Do NOT implement until this doc is approved.**

---

## Executive Summary

The execution, monitoring, and database layers are **fully straddle/strangle-ready**. The scanner entry point is **not wired**. The gap is entirely in the alert-generation layer: no scanner calls `StraddleStrangleStrategy`, no `AlertType` enum value exists for it, and `Alert.from_opportunity()` has no conversion path for straddle opportunities. Everything downstream (Alpaca order submission, position monitoring, P&L recording, Telegram alerts) already works.

**Estimated total effort: ~3–4 days of focused work across 6 tasks.**

---

## Current Architecture: What Exists

### The working scan pipeline (credit spreads)

```
ScanScheduler.run_forever()              shared/scheduler.py:92
  └─ scan_fn(slot_type)                                    :151
       └─ CreditSpreadSystem.scan_opportunities()  main.py:257
            └─ _analyze_ticker(ticker)                     :274  [thread pool]
                 └─ strategy.evaluate_spread_opportunity() :403
                      ├─ find_bull_put_spreads()   spread_strategy.py:107
                      ├─ find_bear_call_spreads()           :115
                      └─ find_iron_condors()                :122
                           └─ Alert.from_opportunity()  alert_schema.py:132
                                └─ alert_router.route()  alert_router.py:~50
                                     └─ ExecutionEngine.submit_opportunity()
                                                   execution_engine.py:57
                                          ├─ _submit_credit_spread()  :160
                                          ├─ _submit_iron_condor()    :189
                                          └─ _submit_straddle()       :252  ← already exists
```

### What's already implemented (no changes needed)

| Component | File | Lines | Notes |
|-----------|------|-------|-------|
| Order submission | `execution/execution_engine.py` | 252–320 | `_submit_straddle()` — 2 single-leg orders, rollback on partial fill |
| Position monitoring | `execution/position_monitor.py` | 311–456 | Debit/credit P&L, per-trade targets, straddle-aware stop (no width cap) |
| Straddle pricing | `execution/position_monitor.py` | 546–600 | `_get_straddle_value()` — prices each leg separately from Alpaca |
| Closing logic | `execution/position_monitor.py` | 783–840 | `_submit_straddle_close()` — two single-leg close orders |
| DB schema | `shared/database.py` | 40–58 | `call_strike`, `put_strike`, `is_debit` stored in `metadata` JSON |
| Signal builder | `strategies/straddle_strangle.py` | 41–203 | `_build_long()` + `_build_short()` with Black-Scholes pricing |
| Signal→Opportunity | `shared/strategy_adapter.py` | 24–68 | `signal_to_opportunity()` handles straddle leg types |
| Regime detector | `ml/combo_regime_detector.py` | 34–200 | BULL/BEAR/NEUTRAL — not yet consulted by straddle path |
| Config section | `configs/paper_exp401.yaml` | 75–86 | Full `straddle_strangle:` block (enabled, mode, DTE, IV params, targets) |

### What's missing (the actual work)

| Gap | File | What's needed |
|-----|------|---------------|
| AlertType enum | `alerts/alert_schema.py:25` | Add `straddle_strangle = "straddle_strangle"` |
| from_opportunity() | `alerts/alert_schema.py:132` | Add straddle conversion path |
| Scanner hook | `strategy/spread_strategy.py:74` | Add `_find_straddles()` call in `evaluate_spread_opportunity()` |
| Event calendar | `strategy/spread_strategy.py` | Wire economic/earnings event lookup into scanner |
| Regime gating | `strategy/spread_strategy.py` | Optional: gate short straddles by regime |
| Telegram label | `alerts/formatters/telegram.py:11` | Add emoji + label for new AlertType |

---

## Detailed Gap Analysis

### Gap 1 — AlertType enum (trivial, ~15 min)

**File:** `alerts/alert_schema.py:25–30`

```python
# Current:
class AlertType(str, Enum):
    credit_spread = "credit_spread"
    momentum_swing = "momentum_swing"
    iron_condor = "iron_condor"
    earnings_play = "earnings_play"
    gamma_lotto = "gamma_lotto"

# Add:
    straddle_strangle = "straddle_strangle"
```

Downstream consumers that switch on `AlertType` must also be updated:
- `alerts/formatters/telegram.py:11` — `_TYPE_EMOJI`, `_TYPE_LABEL` dicts
- `alerts/alert_generator.py` — any type-specific filtering
- `alerts/alert_router.py` — routing logic (check if it has a whitelist)

---

### Gap 2 — Alert.from_opportunity() conversion path (~2 hrs)

**File:** `alerts/alert_schema.py:132–289`

The existing `from_opportunity()` method handles `gamma_lotto` and `iron_condor` via special cases. Straddle/strangle needs a parallel path.

An opportunity dict for a straddle/strangle looks like (from `strategy_adapter.py:37–68`):
```python
{
    "type": "short_straddle",       # or short_strangle / long_straddle
    "ticker": "SPY",
    "call_strike": 450.0,
    "put_strike": 450.0,            # same as call for straddle, different for strangle
    "expiration": "2025-03-21",
    "credit": 3.20,                 # negative if debit (long)
    "is_debit": False,
    "max_loss": 960.0,              # calculated from strike distance or credit × multiplier
    "dte": 5,
    "score": 68.0,
    "event_type": "fomc",           # what triggered the entry
}
```

The conversion needs to:
1. Detect `"straddle" in opp["type"] or "strangle" in opp["type"]`
2. Build two `Leg` objects: call leg + put leg
3. Set `direction = Direction.neutral`
4. Set `entry_price` = credit received (short) or debit paid (long)
5. Set `stop_loss` from config `stop_loss_pct` or `stop_loss_multiplier`
6. Set `profit_target` from config `profit_target_pct`
7. Set `AlertType.straddle_strangle`

Risk: `from_opportunity()` is 157 lines of dense branching. Easy to introduce bugs. Needs unit tests alongside the change.

---

### Gap 3 — Scanner hook in evaluate_spread_opportunity() (~1 day)

**File:** `strategy/spread_strategy.py:74–141`

This is the largest gap. The method currently:
```python
def evaluate_spread_opportunity(self, ticker, price_data, vix_data, tlt_data, scan_date):
    # ... regime detection ...
    opportunities = []

    if regime in (BULL, NEUTRAL):
        opps = self.find_bull_put_spreads(...)
        opportunities.extend(opps)

    if regime in (BEAR, NEUTRAL):
        opps = self.find_bear_call_spreads(...)
        opportunities.extend(opps)

    if iron_condor_config.get("enabled"):
        opps = self.find_iron_condors(...)
        opportunities.extend(opps)

    return sorted(opportunities, key=lambda x: x["score"], reverse=True)
```

**New section to add:**
```python
    ss_config = self.config.get("strategy", {}).get("straddle_strangle", {})
    if ss_config.get("enabled"):
        opps = self._find_straddles(ticker, price_data, vix_data, scan_date, ss_config)
        opportunities.extend(opps)
```

**`_find_straddles()` must implement:**

1. **Event lookup** — check if today is within the entry window for a known event:
   - `mode: short_post_event` → look for events in the past 1–3 days (IV crush window)
   - `mode: long_pre_event` → look for events in the next 5–10 days (vol expansion setup)
   - `mode: both` → check both windows

2. **Event source** — the existing `shared/macro_event_gate.py` and `shared/economic_calendar.py` have event data. Specifically `get_upcoming_events()` in `shared/macro_state_db.py` returns FOMC/CPI dates. Filter by `event_types: fomc_cpi`.

3. **Strike selection** — ATM for straddle, or `otm_pct` OTM for strangle:
   - Call strike: `round(price × (1 + otm_pct) / 5) × 5` (round to nearest $5)
   - Put strike: `round(price × (1 - otm_pct) / 5) × 5`
   - For straddle: both strikes = ATM

4. **Expiration selection** — use existing `_nearest_friday_expiration()` or `_nearest_mwf_expiration()` targeting `target_dte` days out (typically 5 for post-event, 10 for pre-event)

5. **Pricing** — reuse `strategies/straddle_strangle.py`'s `_build_short()` / `_build_long()` methods, or call Black-Scholes directly. The IV estimates:
   - Post-event (short): use `vix / sqrt(252) × sqrt(dte) × iv_crush_pct` (crushed IV)
   - Pre-event (long): use `vix / sqrt(252) × sqrt(dte) × (1 + event_iv_boost)` (boosted IV)

6. **Dedup** — `strategies/straddle_strangle.py` generates Signal objects. The existing `_open_keys` set in the backtester prevents same-expiration re-entry. In the paper trader, the DB dedup check (`client_order_id` determinism) handles this.

7. **Return format** — opportunity dict matching what `strategy_adapter.signal_to_opportunity()` produces (keys: type, ticker, call_strike, put_strike, expiration, credit, is_debit, max_loss, dte, score, event_type)

**Regime gating recommendation for straddles:**
- Short straddle/strangle: allow in NEUTRAL or BEAR (post-spike IV crush)
- Long straddle: allow pre-event regardless of regime (volatility expansion is the thesis)
- Do NOT block long straddles in BULL regime — IV expansion before events is regime-independent

---

### Gap 4 — Economic event calendar integration (~4 hrs)

**Current state:** `alerts/earnings_scanner.py` has an earnings calendar but is not called from the main scan loop. `shared/economic_calendar.py` has FOMC/CPI hardcoded dates. `shared/macro_state_db.py:get_upcoming_events()` returns scheduled macro events.

**Needed in `_find_straddles()`:**
```python
from shared.macro_event_gate import get_upcoming_events

events = get_upcoming_events(horizon_days=14)  # next 2 weeks
recent_events = get_upcoming_events(horizon_days=-3)  # past 3 days (for post-event shorts)
```

**Filter by `event_types` config:**
- `"all"` → use all events
- `"fomc_only"` → filter for type == "fomc"
- `"fomc_cpi"` → filter for type in ("fomc", "cpi")

**Caveat:** `get_upcoming_events()` uses the macro state DB which may be empty in paper trading if the macro snapshot engine isn't running. Need a fallback to the hardcoded `shared/economic_calendar.py` FOMC/CPI dates.

---

### Gap 5 — Regime gating for straddles (optional, ~2 hrs)

**File:** `strategy/spread_strategy.py` (inside `_find_straddles()`)

The regime scales already exist in `configs/paper_exp401.yaml:163–176`:
```yaml
ss_regime_scale_bull: 1.5
ss_regime_scale_bear: 1.5
ss_regime_scale_high_vol: 2.5
ss_regime_scale_low_vol: 1.0
ss_regime_scale_crash: 0.0
```

These should be applied to position sizing (not entry blocking) for straddles. A `scale=0.0` effectively blocks entry (0 contracts).

**Implementation sketch:**
```python
regime = technical_signals.get("combo_regime", "NEUTRAL")
scale = {
    "BULL": ss_config.get("ss_regime_scale_bull", 1.0),
    "BEAR": ss_config.get("ss_regime_scale_bear", 1.0),
    "NEUTRAL": 1.0,
}.get(regime, 1.0)

if scale == 0.0:
    return []  # blocked by regime

# Pass scale into opportunity dict for sizer to apply
opp["regime_scale"] = scale
```

The `AlertPositionSizer` already reads `regime_scale` from opportunities for credit spreads (`alert_position_sizer.py:~120`).

---

### Gap 6 — Telegram formatter (~30 min)

**File:** `alerts/formatters/telegram.py:11–25`

```python
# Add to _TYPE_EMOJI:
AlertType.straddle_strangle: "🟣",

# Add to _TYPE_LABEL:
AlertType.straddle_strangle: "STRADDLE/STRANGLE",
```

The formatter's `format_entry_alert()` method at line 35 is already generic — it iterates over `alert.legs` and prints each one. A 2-leg straddle (SELL CALL + SELL PUT) will format correctly without other changes.

The one wrinkle: the existing message template references `direction` as "BULLISH" or "BEARISH". For straddles, `direction = Direction.neutral`, which should render as "NEUTRAL" — confirm this path exists in the formatter at line 49.

---

## Monitoring and Exit Strategy

### How straddle exits differ from credit spreads

| Dimension | Credit Spread | Straddle/Strangle |
|-----------|--------------|-------------------|
| Max loss defined? | Yes (spread width) | No (short: theoretically unlimited; capped at `max_loss` in config) |
| Stop loss formula | `credit × stop_loss_multiplier` or `pct_of_width` | `debit × (1 + stop_loss_pct)` (long) or `credit × stop_loss_multiplier` (short) |
| Profit target | % of credit received | % of credit received (short) or debit paid (long) |
| Position value | Mid of spread | Sum of both legs (short) or max of both legs (long) |
| DTE exit | Optional `manage_dte` | 0 DTE forced close (already in position monitor) |
| Delta hedging? | Not applicable | **Not implemented, not planned** — we are volatility/event players, not delta-neutral market makers |

### Why NOT delta hedge

Delta hedging adds:
- Continuous monitoring overhead (real-time data)
- Commission drag from frequent small hedge trades
- Complexity without clear edge for event-driven short straddles

Our thesis is **IV crush after binary events** (FOMC, CPI). After the event, IV collapses fast and we close at 50% profit. We are not running a delta-neutral book — we want the decay to work quickly after the event, then exit. No hedging needed.

### The actual monitoring flow (already wired)

`PositionMonitor._check_exit_conditions()` at `execution/position_monitor.py:311`:

```
For short straddle (credit position, is_debit=False):
  spread_value = call_price + put_price           (current cost to buy back)
  pnl_pct = (credit - spread_value) / credit

  if pnl_pct >= profit_target_pct:  → CLOSE (profit target hit)
  if spread_value >= credit × (1 + stop_loss_multiplier):  → CLOSE (stop hit)
  if dte == 0:  → CLOSE (expiration)

For long straddle (debit position, is_debit=True):
  spread_value = max(call_price, put_price)        (best leg wins)
  loss_pct = (debit - spread_value) / debit
  pnl_pct = (spread_value - debit) / debit

  if pnl_pct >= profit_target_pct:  → CLOSE (profit target hit)
  if loss_pct >= stop_loss_pct:  → CLOSE (stop hit)
  if dte == 0:  → CLOSE (expiration)
```

Note at `position_monitor.py:419`: `is_straddle = "straddle" in spread_type or "strangle" in spread_type` — this already bypasses the `spread_width` stop formula that's used for credit spreads. The straddle stop is purely credit/debit based.

---

## Risk Budget Interaction with Credit Spreads

### Current risk budget (EXP-401, `paper_exp401.yaml`)

```yaml
risk:
  max_risk_per_trade: 12.0          # Credit spread budget (per trade)
  straddle_strangle_risk_pct: 3.0   # Straddle budget (per trade)
  max_positions: 12
  max_positions_per_ticker: 3
  portfolio_risk:
    max_portfolio_risk_pct: 40      # Total open risk cap
    max_same_expiration: 4
```

### How they interact

The `AlertPositionSizer` (`alerts/alert_position_sizer.py:46`) selects `risk_pct` based on `alert.type`:
- `credit_spread` / `iron_condor` → uses `max_risk_per_trade` (12%)
- `straddle_strangle` → needs to use `straddle_strangle_risk_pct` (3%)

**This routing is NOT yet implemented** in `alert_position_sizer.py`. It currently uses `alert.risk_pct` which comes from `from_opportunity()`. As long as `from_opportunity()` sets `risk_pct = ss_config["max_risk_pct"]` (3.0), the sizer will use 3% correctly.

The shared `max_portfolio_risk_pct: 40%` cap is enforced by the portfolio heat tracker regardless of position type — it accumulates `max_loss / account_size` across ALL open positions. A 3% straddle and a 12% credit spread both count against the 40% cap.

**Potential conflict:** On a high-vol event day, we might want to enter:
- 2 credit spreads (2 × 12% = 24% risk)
- 1 short straddle post-event (3%)
- Total: 27% → under 40% cap ✓

The risk budget interaction is **safe as designed**. No changes needed to the portfolio heat tracker.

**Recommendation:** Add a separate `max_concurrent_straddles: 2` config option to limit straddle exposure independently, since straddle max loss is theoretically uncapped (unlike credit spreads with defined width).

---

## Task Breakdown

### Task 1: AlertType enum + Telegram formatter
**Files:** `alerts/alert_schema.py`, `alerts/formatters/telegram.py`
**Effort:** ~1 hour
**Risk:** Low — purely additive
**Steps:**
1. Add `straddle_strangle = "straddle_strangle"` to `AlertType` at `alert_schema.py:30`
2. Add emoji `🟣` and label `"STRADDLE/STRANGLE"` in `telegram.py:11–25`
3. Check `alert_router.py` for any AlertType whitelist that must be updated
4. Run `tests/test_alert_schema.py` + `tests/test_telegram_formatter.py`

---

### Task 2: Alert.from_opportunity() straddle path
**Files:** `alerts/alert_schema.py:132–289`
**Effort:** ~3 hours
**Risk:** Medium — dense branching; existing tests rely on this method
**Steps:**
1. Add detection: `opp_type = opp.get("type", ""); is_straddle = "straddle" in opp_type or "strangle" in opp_type`
2. Build two `Leg` objects (call + put)
3. Set `direction = Direction.neutral`
4. Map `entry_price`, `stop_loss`, `profit_target` from opp fields
5. Set `AlertType.straddle_strangle`
6. Write unit tests: long straddle, short straddle, short strangle round-trip

---

### Task 3: _find_straddles() scanner
**Files:** `strategy/spread_strategy.py`, possibly `shared/economic_calendar.py`
**Effort:** ~1 day
**Risk:** Medium — new code in the hot scan path; must be fast
**Steps:**
1. Add private `_find_straddles(ticker, price_data, vix_data, scan_date, ss_config) -> List[Dict]`
2. Event lookup: call `get_upcoming_events()`, filter by `event_types` config, check entry windows
3. Fallback to hardcoded FOMC/CPI calendar if macro DB is empty
4. Strike selection: ATM or OTM per `otm_pct` config
5. Expiration: nearest Friday at `target_dte` using existing `_nearest_friday_expiration()`
6. Pricing: call `strategies/straddle_strangle.py` pricing helpers or inline Black-Scholes
7. Return opportunity dict(s); empty list if no qualifying events
8. Call from `evaluate_spread_opportunity()` at `spread_strategy.py:~130`
9. Write tests: post-event short, pre-event long, no-event returns empty

---

### Task 4: Event calendar wiring
**Files:** `shared/macro_event_gate.py`, `shared/economic_calendar.py`, `strategy/spread_strategy.py`
**Effort:** ~4 hours
**Risk:** Low-medium — depends on whether macro DB is populated in paper trading
**Steps:**
1. Audit `get_upcoming_events()` — does it work without a populated macro DB?
2. If not: implement fallback to hardcoded `FOMC_DATES` + `CPI_DATES` in `shared/economic_calendar.py`
3. Implement `get_recent_events(days_ago=3)` helper for post-event detection (if not present)
4. Wire filter logic by `event_types` config value
5. Test both paths (DB populated and empty)

---

### Task 5: Regime gating and size scaling
**Files:** `strategy/spread_strategy.py` (inside `_find_straddles()`)
**Effort:** ~2 hours
**Risk:** Low — additive only
**Steps:**
1. Read `ss_regime_scale_*` from config
2. Apply scale to contracts (0.0 = skip entry entirely)
3. Add `regime_scale` key to opportunity dict
4. Confirm `AlertPositionSizer` reads `regime_scale` for straddle type
5. Test: NEUTRAL regime → scale 1.0, BULL → 1.5, crash → 0.0 (no entry)

---

### Task 6: Integration test + paper trading validation
**Files:** `tests/test_spread_strategy_full.py` or new file
**Effort:** ~4 hours
**Risk:** Low — validation only
**Steps:**
1. Mock an FOMC event in the past 2 days
2. Run `evaluate_spread_opportunity()` and verify a straddle opportunity is returned
3. Run through `Alert.from_opportunity()` → verify `AlertType.straddle_strangle`
4. Mock `ExecutionEngine.submit_opportunity()` → verify `_submit_straddle()` is called
5. Verify Telegram message format includes correct emoji and "NEUTRAL" direction
6. In paper trading (EXP-401 session): watch for first FOMC/CPI date and confirm entry + exit cycle

---

## Effort Summary

| Task | Est. Effort | Risk | Dependency |
|------|-------------|------|------------|
| T1: AlertType enum + Telegram | 1 hr | Low | None |
| T2: from_opportunity() path | 3 hrs | Medium | T1 |
| T3: _find_straddles() scanner | 1 day | Medium | T2 |
| T4: Event calendar wiring | 4 hrs | Low-Med | T3 |
| T5: Regime gating + scaling | 2 hrs | Low | T3 |
| T6: Integration tests + validation | 4 hrs | Low | T1–T5 |
| **Total** | **~3.5 days** | | |

---

## What This Spec Does NOT Cover

- **Delta hedging** — deliberately excluded; our thesis is event-driven IV crush, not delta-neutral market making
- **Rolling positions** — not supported by `PositionMonitor`; intentional (short DTE, close and re-enter)
- **Live data pricing** — we use Alpaca's live option quotes (already done in `_get_straddle_value()`); no Polygon needed for paper trading
- **Vega-based sizing** — flat dollar risk is sufficient for 3% per trade; vega-normalized sizing adds complexity without clear edge at this size
- **Cross-ticker straddle correlation** — not needed; straddles are independent event plays per ticker

---

## Open Questions (Resolve Before Implementing)

1. **Event calendar source in paper trading:** Is the macro state DB populated when EXP-401 runs, or do we need the hardcoded calendar fallback? Check `data/pilotai_exp401.db` for `macro_events` table rows.

2. **Short vs long mode for EXP-401:** Config says `mode: short_post_event`. Is this still the intent, or should we also support `long_pre_event`? Long straddles before FOMC are a different risk profile (capped loss, large upside if big move).

3. **Max concurrent straddles:** Should we add `max_concurrent_straddles: 2` to config? Without it, multiple straddles could accumulate near the 40% portfolio cap quickly during volatile periods.

4. **Strangle vs straddle preference:** ATM straddle has higher premium and higher delta risk if the market moves before the event. OTM strangle has lower premium and wider breakevens. Which does the backtest support for post-FOMC IV crush? (The `otm_pct: 0.04` config implies strangle preference.)

5. **IV source for live pricing:** `_build_short()` in `strategies/straddle_strangle.py` uses Black-Scholes with VIX-derived IV. In paper trading, should we use Polygon live IV quotes instead for better accuracy?

---

*Last updated: 2026-03-15*
*Author: Maximus architecture review*
