# Data Architecture — Iron Vault

## The Golden Rule

**ALL pricing data must come from IronVault. NEVER synthetic/heuristic pricing.**

Any code path that generates an option price without querying the real `options_cache.db` is a critical bug. This is a Carlos directive, non-negotiable.

---

## Data Sources

### 1. options_cache.db (Primary — ~900MB)

SQLite database at `data/options_cache.db`. Pre-populated via Polygon API, never modified at backtest time (`offline_mode=True`).

| Table | Rows (approx) | Description |
|-------|--------------|-------------|
| `option_contracts` | 168K | Strike/expiration reference data per ticker |
| `option_daily` | 5.67M | Daily OHLCV bars per option contract |
| `option_intraday` | 1.59M | 5-min intraday bars per option contract |

**Coverage**: SPY 2020-01-02 → 2026-02-25, plus sector ETFs (XLE, XLK, SOXX, etc.)

### 2. macro_state.db (Macro/COMPASS signals)

SQLite database at `data/macro_state.db`. Contains sector COMPASS scores, macro regime signals, and economic calendar data. Accessed via `shared/macro_state_db.py`.

### 3. Live price data (Yahoo Finance)

Used exclusively for **underlying prices** (SPY close, VIX) during backtesting. Fetched via curl-based yfinance wrapper to bypass LibreSSL issues (`backtest/backtester.py:_yf_history_safe`). This is real market data, not synthetic.

---

## IronVault — The Single Access Point

`shared/iron_vault.py` is the authoritative entry point for all options data access.

```python
from shared.iron_vault import IronVault, IronVaultError

# Get the singleton (validates DB on first call)
hd = IronVault.instance()

# Pass to Backtester
bt = Backtester(config, historical_data=hd)
```

### What IronVault Does

- Validates `options_cache.db` exists and contains data at startup
- Raises `IronVaultError` if DB is missing or empty
- Delegates all queries to `HistoricalOptionsData(offline_mode=True)` — cache-only, no live Polygon calls
- Provides `coverage_report()` for DB health checks

### What IronVault Does NOT Do

- Make live Polygon API calls (offline_mode=True always)
- Return synthetic/heuristic prices — EVER
- Raise on per-contract cache misses (None = "skip this trade", not an error)

### Per-Contract Cache Misses

When `get_spread_prices()` returns `None` for a specific contract/date, the backtester **skips that trade** (correct behavior — we just didn't have that contract in cache that day). This is NOT the same as synthetic fallback. The distinction:

| Behavior | Correct? |
|----------|----------|
| Cache miss → skip trade | ✅ YES |
| Cache miss → fabricate price from Black-Scholes | ❌ NO (heuristic) |
| Cache miss → use fixed % of spread width as credit | ❌ NO (heuristic) |

---

## Setup From Scratch

### Step 1: Set API Key

```bash
echo "POLYGON_API_KEY=your_key_here" >> .env
```

### Step 2: Validate Setup

```bash
python scripts/iron_vault_setup.py --verbose
```

This checks:
- POLYGON_API_KEY presence
- DB existence and size
- Per-ticker contract counts and year coverage
- Flags any critical gaps

### Step 3: Backfill Missing Data (if needed)

```bash
# SPY (main backtest ticker)
python scripts/fetch_polygon_options.py --ticker SPY --resume

# Sector ETFs (COMPASS portfolio backtests)
python scripts/fetch_sector_options.py --ticker XLE --resume
python scripts/fetch_sector_options.py --ticker XLK --resume
```

### Step 4: Run a Backtest

```bash
python scripts/run_optimization.py --config configs/champion.json
```

No `--heuristic` flag exists. All backtests use real data.

---

## Architecture: How Data Flows

```
IronVault.instance()
    └── HistoricalOptionsData(offline_mode=True)
            └── options_cache.db
                    ├── option_contracts (strike lookup)
                    ├── option_daily     (entry/exit prices)
                    └── option_intraday  (intraday scan exits)

Backtester
    ├── historical_data = IronVault.instance()
    ├── _find_real_spread() → get_available_strikes() → get_spread_prices()
    ├── _manage_positions() → get_spread_prices() / get_intraday_spread_prices()
    └── _close_at_expiration_real() → get_spread_prices() → underlying intrinsic fallback
```

---

## Coverage Map (as of 2026-03-16)

| Ticker | Contracts | Daily Bars | Intraday Bars | Years |
|--------|-----------|-----------|---------------|-------|
| SPY | ~166K | ~5.6M | ~1.59M | 2020-2025 |
| XLE | ~1K | ~15K | 0 | 2020-2025 |
| XLK | ~1.8K | 0 | 0 | fetching |
| SOXX | ~70 | 0 | 0 | monthly only |

**Note**: Sector ETFs use heuristic mode in the portfolio backtester because their data is too sparse for reliable spread pricing (5-10 strikes/expiration, $1-wide strikes). Only SPY has full coverage.

---

## Files

| File | Purpose |
|------|---------|
| `shared/iron_vault.py` | Singleton access point, validation |
| `scripts/iron_vault_setup.py` | Setup validation, coverage report |
| `backtest/historical_data.py` | Raw DB queries (wrapped by IronVault) |
| `backtest/backtester.py` | Uses IronVault as `historical_data` param |
| `scripts/run_optimization.py` | Always calls `IronVault.instance()` |
| `scripts/run_portfolio_backtest.py` | Same |
| `data/options_cache.db` | The 900MB source of truth |
