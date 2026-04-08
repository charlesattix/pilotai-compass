# IronVault Data Inventory

**Last Updated:** 2026-04-05 (post-TLT backfill)
**Database:** `data/options_cache.db` (976 MB)
**Source:** Polygon.io historical options data

---

## Database Summary

| Metric | Count |
|--------|-------|
| Total option contracts | 261,770 |
| Total daily bars | 6,275,226 |
| Total intraday bars | 1,591,036 |
| Database size | 976 MB |

---

## Per-Ticker Coverage

### Option Contracts

| Ticker | Contracts | Expirations | First Exp | Last Exp | Status |
|--------|-----------|-------------|-----------|----------|--------|
| **SPY** | 193,272 | 646 | 2020-01-29 | 2026-06-30 | **FULL** — weeklies + monthlies |
| **XLI** | 17,287 | 320 | 2020-01-03 | 2026-06-18 | **FULL** |
| **GLD** | ~14,700+ | 250+ | 2020-01-17 | **2025-01-17** | **EXTENDED** — backfilled to Jan 2025 (was Oct 2024) |
| **XLF** | 9,256 | 320 | 2020-01-03 | 2026-06-30 | **FULL** |
| **QQQ** | ~12,200+ | 230+ | 2020-01-03 | **2025-12-19** | **EXTENDED** — backfilled from Apr 2023 to Dec 2025 |
| **TLT** | ~10,700+ | 198 | 2020-01-17 | **2025-12-19** | **EXTENDED** — backfilled Aug 2024 to Dec 2025 |
| **SOXX** | 3,460 | 70 | 2020-07-17 | 2026-06-18 | Partial |
| **XLK** | 2,680 | 242 | 2020-01-17 | 2026-06-18 | Partial (few strikes) |
| **XLE** | 1,757 | 181 | 2020-04-17 | 2026-06-30 | Partial (few strikes) |

### Daily Bars (with volume > 0)

| Ticker | Daily Bars | Trading Days | First Date | Last Date |
|--------|------------|-------------|------------|-----------|
| SPY | 4,423,353 | 1,732 | 2019-03-04 | 2026-04-02 |
| QQQ | 392,739+ | 1,500+ | 2020-01-02 | 2025-12-19 |
| XLF | 243,583 | 1,571 | 2020-01-02 | 2026-04-02 |
| XLI | 200,761 | 1,571 | 2020-01-02 | 2026-04-02 |
| TLT | 293,446 | 1,501 | 2020-01-02 | **2025-12-19** |
| GLD | 190,933+ | 1,200+ | 2020-01-02 | 2025-01-17 |
| SOXX | 37,229 | 804 | 2020-02-06 | 2026-04-02 |
| XLE | 20,542 | 1,359 | 2020-01-23 | 2026-04-02 |
| XLK | 18,702 | 1,431 | 2020-01-02 | 2026-04-02 |

### Intraday Bars (5-minute)

| Ticker | Bars | Days | First | Last |
|--------|------|------|-------|------|
| SPY | 1,426,178 | 1,540 | 2020-01-02 | 2026-02-24 |

SPY has 5-minute intraday bars from 09:30 to 16:00 (~78 bars/day).
No intraday data for other tickers.

---

## Critical Data Gaps

### 1. GLD Options — BACKFILLED (2026-04-05)

- **Status:** Extended from Oct 2024 → Jan 2025 via `scripts/backfill_gld_qqq.py`
- **New data:** 623 contracts, 11,026 bars added
- **Coverage:** Now through Jan 2025 (monthly expirations)
- **Remaining gap:** Feb-Dec 2025 (11 months, ~1,100 contracts)
- **Severity:** MEDIUM — 5 years of data available, recent gap is shrinking

### 2. QQQ Options — BACKFILLED (2026-04-05)

- **Status:** Extended from Apr 2023 → Dec 2025 via `scripts/backfill_gld_qqq.py`
- **New data:** 3,000 contracts, 88,659 bars added
- **Coverage:** Now through Dec 2025 (monthly expirations May 2023 onward)
- **Remaining gap:** None for monthly expirations. Weekly expirations not backfilled.
- **Severity:** RESOLVED for strategy backtesting purposes

### 3. TLT Options — BACKFILLED (2026-04-05)

- **Status:** Extended from Jul 2024 → Dec 2025 via `scripts/backfill_tlt.py`
- **Method:** OCC symbol construction (strikes $70-$115, 17 monthly expirations, puts + calls)
- **New data:** 1,564 contracts registered, 3,082 bars fetched from Polygon (0 errors)
- **Coverage:** Now through Dec 2025 (198 expirations, 293,446 daily bars, 1,501 trading days)
- **Remaining gap:** None for monthly expirations in the $70-$115 strike range
- **Severity:** RESOLVED — TLT Iron Condors and TLT-based pairs can now be validated on 2024-2025 data

### 4. SPY Intraday — Missing Mar 2026 to Present

- **Impact:** Intraday strategies (EXP-1000, gap fade) lack most recent data
- **Severity:** LOW — 5+ years of intraday data available

---

## Polygon API Status

| Parameter | Value |
|-----------|-------|
| API Key | Set in `.env` (starts with `y3y07kPI...`) |
| Key Status | **ACTIVE** — daily bars work, market status works |
| Options Contract Enumeration | **NOT AVAILABLE** — returns 0 results (likely Starter tier) |
| Individual Contract Bars | **WORKS** — can fetch specific contract OHLCV |
| Rate Limit | ~5 calls/sec (free/starter tier) |

### Backfill Strategy

The current Polygon tier can fetch daily bars for individual contracts but
cannot enumerate new contracts. To backfill:

**Option A: Upgrade to Options tier**
- Enables `/v3/reference/options/contracts` enumeration
- Cost: ~$200/month
- Time to backfill: ~3 hours for all gaps

**Option B: Manual contract enumeration**
- Build OCC symbols from known strike grids + expiration dates
- Use `/v2/aggs/ticker/{OCC_SYMBOL}/range/1/day/{from}/{to}`
- Works with current tier but requires strike guessing

**Option C: Alternative data source**
- CBOE DataShop, OptionMetrics, or IVolatility
- Higher cost but complete data

---

## Backfill Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/backfill_polygon_cache.py` | SPY option daily bars | Complete (SPY up to date) |
| `scripts/backfill_tlt.py` | TLT option backfill (OCC symbol construction) | **Complete** — Jul 2024 to Dec 2025 |
| `scripts/backfill_gap.py` | Targeted gap backfill for existing contracts | Working |
| `scripts/fetch_sector_options.py` | Multi-ticker option fetch | Available but needs Options tier |
| `scripts/iron_vault_setup.py` | DB setup and validation | Working |

### Running a Backfill

```bash
# If Polygon Options tier is available:
python3 scripts/fetch_sector_options.py --ticker GLD --start 2024-03-16 --end 2025-12-31
python3 scripts/fetch_sector_options.py --ticker TLT --start 2024-07-20 --end 2025-12-31
python3 scripts/fetch_sector_options.py --ticker QQQ --start 2023-04-22 --end 2025-12-31

# SPY is already up to date — no backfill needed
```

---

## Data Quality Notes

1. **All option prices are from Polygon** — real market data, not synthetic
2. **Volume data available** for SPY (reliable), sector ETFs (partial)
3. **Open Interest** is NULL for most contracts (Polygon starter tier limitation)
4. **IronVault policy:** cache miss returns `None` (trade skipped), NEVER falls back to synthetic
5. **Split/adjustment:** Polygon handles corporate actions automatically
6. **Expiration types:** Mix of monthlies (3rd Friday) and weeklies (every Friday for SPY)
