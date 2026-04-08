# Paper Trading Status Report

**Date:** 2026-04-04
**Validation Window:** Mar 15 → May 11 (8-week clock)
**Days Elapsed:** 20 of 56

---

## Executive Summary

| Status | Detail |
|--------|--------|
| **EXP-400** | 1 closed trade, 16 orphan positions detected. DB at `data/pilotai.db`. |
| **EXP-401** | 1 open trade (bear call, 14 contracts), 14 orphans. DB at `data/pilotai_exp401.db`. |
| **EXP-503** | **No database exists.** Config references `data/exp503/pilotai_exp503.db` — never created. |
| **EXP-600** | **No database exists.** Config references `data/exp600/pilotai_exp600.db` — never created. |
| **Cron (data)** | `scripts/daily_data_update.sh` + `scripts/setup_cron.sh` exist but **cron is NOT installed** (`crontab -l` → empty). |
| **Cron (scan)** | `scripts/scan-cron.sh` exists, references all 4 experiments, but **cron is NOT installed**. Hardcoded macOS path (`/Users/charlesbot/projects/...`) — won't work on this machine. |

**Overall Assessment: CRITICAL — 2 of 4 experiments have no database at all; cron jobs are not running; orphan positions dominate the existing databases.**

---

## Per-Experiment Detail

### EXP-400: The Champion (SPY)

- **Config:** `configs/paper_champion.yaml`
- **DB:** `data/pilotai.db`
- **Alpaca Account:** PA36XFVLG0WE
- **Live Since:** 2026-03-15
- **Env File:** `.env.exp400` (exists)

#### Trade Log

| Metric | Value |
|--------|-------|
| Total records | 17 |
| `closed_external` | 1 |
| `unmanaged` (orphans) | 16 |
| Intentional trades | **1** |
| Date range | 2026-03-24 → 2026-03-27 |

**The single closed trade:**
- ID: `t1`
- Type: `bull_put` (SPY 450/445 put spread)
- Entry: 2026-03-24, Credit: $1.00, Contracts: 1
- Exit: 2026-04-02 (closed_external)
- PnL: `null` (not recorded)
- Expiration: 2025-06-20 (note: expiration is in the PAST — anomaly)

**16 orphan positions** detected on 2026-03-27. All are SPY call options (C) with strikes in the 649-681 range, expirations 2026-04-10 and 2026-04-17. All have `strategy_type: unknown`, zero contracts, no credit.

#### Anomalies
1. **Only 1 intentional trade in 20 days** — the scanner should be generating multiple trades per week
2. **Expiration date anomaly** — the closed trade has expiration `2025-06-20` which is ~9 months before entry date (2026-03-24). This is clearly a data error.
3. **16 orphan positions** — the reconciler is detecting positions in Alpaca that weren't placed by the system. These are likely from manual trading or another process on the same Alpaca account.
4. **PnL not recorded** on the closed trade
5. **Scanner state shows `peak_equity: 125000`** but no trades justify this

### EXP-401: The Blend (SPY)

- **Config:** `configs/paper_exp401.yaml`
- **DB:** `data/pilotai_exp401.db`
- **Alpaca Account:** PA3Y2XDYB9I3
- **Live Since:** 2026-03-15
- **Env File:** `.env.exp401` (exists)

#### Trade Log

| Metric | Value |
|--------|-------|
| Total records | 15 |
| `open` | 1 |
| `unmanaged` (orphans) | 14 |
| Intentional trades | **1** |
| Date range | 2026-03-27 (single day) |

**The open trade:**
- ID: `cs-cba3f19eac66c910`
- Type: `bear_call` (SPY 649/661 call spread)
- Entry: 2026-03-27 18:00 UTC, Credit: $3.84/contract, Contracts: 14
- Expiration: 2026-04-17
- Status: `open` — still live
- **Note:** 14 contracts × $3.84 credit = $5,376 total credit. On a $12-wide spread, max loss = $16,800. This is a significant position.

**14 orphan positions** detected same day — identical call strikes to EXP-400 orphans (SPY 660-681 range, 2026-04-10/04-17 expirations).

#### Anomalies
1. **Only 1 real trade in 20 days** — same signal drought as EXP-400
2. **Shared orphan pattern** — EXP-400 and EXP-401 detect the same orphan contracts. This suggests the Alpaca accounts may share positions or the orphan detection has a bug.
3. **Bear call in a bull market?** — SPY in late March 2026 was in a strong rally. A 649/661 bear call spread entered on 2026-03-27 is aggressively bearish. The combo regime may have classified this period as "bear" (incorrectly).
4. **1 trade_features row** — the ML feature logging is working but only captured one entry

### EXP-503: ML V2 Aggressive (SPY)

- **Config:** `configs/paper_exp503.yaml`
- **DB:** `data/exp503/pilotai_exp503.db` — **DOES NOT EXIST**
- **Alpaca Account:** PA3Z9PLVYUL5
- **Live Since:** 2026-03-22
- **Env File:** `.env.exp503` — **NOT FOUND** (only `.env.exp400` and `.env.exp401` exist)

#### Status: NOT DEPLOYED

The database directory `data/exp503/` was never created. There is no `.env.exp503` file. The experiment exists in the registry and has a config file, but **was never actually started**.

### EXP-600: IBIT Adaptive (Crypto)

- **Config:** `configs/paper_exp600.yaml`
- **DB:** `data/exp600/pilotai_exp600.db` — **DOES NOT EXIST**
- **Alpaca Account:** PA3O14JAJHJ0
- **Live Since:** 2026-03-22
- **Env File:** `.env.exp600` — **NOT FOUND**

#### Status: NOT DEPLOYED

Same as EXP-503 — the database and env file were never created. The config exists but the experiment was never started.

---

## Polygon Data Update Cron

### Scripts Available

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/daily_data_update.sh` | Fetch new SPY options data from Polygon | Exists, executable |
| `scripts/setup_cron.sh` | Install/remove crontab entry for daily update | Exists, executable |
| `scripts/backfill_polygon_cache.py` | Backfill options_cache.db from Polygon API | Exists |
| `scripts/scan-cron.sh` | Run scanner for all 4 paper experiments | Exists, but **hardcoded macOS path** |

### Cron Status: NOT INSTALLED

```
$ crontab -l
→ no crontab for node
```

Neither the data update cron nor the scanner cron is installed. The setup infrastructure exists (`setup_cron.sh`) but was never run.

### `scan-cron.sh` Portability Issue

Line 12: `PROJECT_DIR="/Users/charlesbot/projects/pilotai-credit-spreads"` — hardcoded to a macOS path. This script will fail on any other machine. Should use `$(dirname "$(dirname "$(readlink -f "$0")")")` or similar.

---

## Missed Trades Analysis

### Expected vs Actual

| Experiment | Expected Trades (20 days) | Actual Trades | Gap |
|------------|---------------------------|---------------|-----|
| EXP-400 | ~10-15 (based on heuristic backtest: ~67/year) | **1** | **~12 missed** |
| EXP-401 | ~10-15 | **1** | **~12 missed** |
| EXP-503 | ~10-15 | **0** (not deployed) | **All missed** |
| EXP-600 | ~8-10 (IBIT, lower frequency) | **0** (not deployed) | **All missed** |

### Root Causes of Missed Trades

1. **Cron not installed** — the scanner (`scan-cron.sh`) is not being run on a schedule. Without scheduled scans, the system only trades when manually triggered.
2. **EXP-503/600 never deployed** — no DB, no env file. These were announced in the MASTERPLAN on 2026-03-22 but the deployment was never completed.
3. **Scanner runs but doesn't find opportunities** — the 1 trade each in EXP-400/401 suggests the scanner DID run at least once (around 2026-03-27) but the regime/technical filters are too restrictive for the current market conditions.
4. **Orphan position contamination** — 16 orphan positions in EXP-400 and 14 in EXP-401 suggest the Alpaca accounts have positions from other sources, which may be consuming the `max_positions` limit.

---

## Recommendations

### Immediate Actions (P0)

1. **Install the cron jobs** — Run `scripts/setup_cron.sh` on the production machine. Also set up scan-cron.sh on a schedule (fix the hardcoded path first).
2. **Deploy EXP-503 and EXP-600** — Create `data/exp503/` and `data/exp600/` directories, set up `.env.exp503` and `.env.exp600` with Alpaca credentials.
3. **Investigate orphan positions** — 30 orphan positions across 2 accounts suggest shared account usage. Either (a) use dedicated Alpaca accounts per experiment, or (b) filter orphans from the position count so they don't block new trades.
4. **Fix the EXP-400 expiration anomaly** — the `t1` trade has expiration `2025-06-20` (9 months before entry). Investigate whether this is a data entry bug or a system error.

### Short-Term (P1)

5. **Fix scan-cron.sh portability** — Replace hardcoded `/Users/charlesbot/projects/` with relative path detection.
6. **Verify Polygon data is current** — Run `scripts/daily_data_update.sh --dry-run` to check if options_cache.db has recent data. Without fresh data, the scanner can't find valid option contracts.
7. **Add monitoring** — Set up a daily heartbeat check that alerts if no scan has run in 24 hours.

### Validation Clock

The 8-week validation window (Mar 16 → May 11) is 36% elapsed with essentially **zero meaningful data**. If cron is installed today and issues are resolved, there are ~5 weeks remaining — still enough time for a meaningful paper trading validation if trade frequency improves.
