"""
Macro Snapshot Engine
=====================
Powers both historical generation and live snapshots.

Computes:
  - Sector relative strength (3M, 12M) for 11 SPDR sectors + 4 thematic ETFs
  - RRG quadrant classification (Leading / Weakening / Lagging / Improving)
  - Macro score (4 dimensions: growth, inflation, fed policy, risk appetite)
  - Cross-sectional sector rankings

Data sources:
  - Polygon REST API  : adjusted daily OHLCV for ETFs
  - FRED REST API     : macro time series

Lookahead prevention:
  RELEASE_LAG_DAYS is applied per FRED series so that when generating a snapshot
  for date D, only FRED observations available on D are used.  Daily series have
  a 1-day lag; monthly releases have a 30-45 day lag depending on the series.
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ── Polygon base URL ───────────────────────────────────────────────────────────
POLYGON_BASE = "https://api.polygon.io"
FRED_BASE = "https://api.stlouisfed.org/fred"

# ── Ticker universe ────────────────────────────────────────────────────────────
SECTOR_ETFS: Dict[str, str] = {
    "XLC":  "Communication Services",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLV":  "Health Care",
    "XLI":  "Industrials",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
    "XLK":  "Technology",
    "XLU":  "Utilities",
}

THEMATIC_ETFS: Dict[str, str] = {
    "SOXX": "Semiconductors",
    "XBI":  "Biotech",
    "PAVE": "Infrastructure",
    "ITA":  "Defense & Aerospace",
}

ALL_ETF_TICKERS: Dict[str, str] = {**SECTOR_ETFS, **THEMATIC_ETFS}
BENCHMARK = "SPY"

# ── FRED series config ─────────────────────────────────────────────────────────
# lag_days: minimum calendar days after the observation period end before the
#           value is published on FRED.  Used to block lookahead.
FRED_SERIES: Dict[str, Dict] = {
    "T10Y2Y":       {"name": "10Y-2Y Yield Spread",              "lag_days": 1,  "freq": "daily"},
    "VIXCLS":       {"name": "VIX Close",                         "lag_days": 1,  "freq": "daily"},
    "BAMLH0A0HYM2": {"name": "HY OAS Spread (%)",                 "lag_days": 1,  "freq": "daily"},
    "T5YIE":        {"name": "5Y Breakeven Inflation",            "lag_days": 1,  "freq": "daily"},
    "FEDFUNDS":     {"name": "Effective Fed Funds Rate",          "lag_days": 35, "freq": "monthly"},
    "CFNAI":        {"name": "Chicago Fed Natl Activity Index",   "lag_days": 25, "freq": "monthly"},
    "PAYEMS":       {"name": "Nonfarm Payrolls (000s)",           "lag_days": 35, "freq": "monthly"},
    "CPIAUCSL":     {"name": "CPI All Items (NSA)",               "lag_days": 35, "freq": "monthly"},
    "CPILFESL":     {"name": "Core CPI (NSA)",                    "lag_days": 35, "freq": "monthly"},
}
# Note: NAPM (ISM Manufacturing PMI) was removed from FRED in June 2016.
# CFNAI (Chicago Fed National Activity Index) is used instead — it is a composite
# of 85 monthly indicators and is a superior comprehensive growth proxy.

# ── RRG / RS lookback constants ────────────────────────────────────────────────
RS_3M_DAYS   = 63    # trading-day lookback for 3-month RS
RS_12M_DAYS  = 252   # trading-day lookback for 12-month RS
RS_MOM_DAYS  = 21    # lookback for RS-Ratio momentum (4 weeks)
WARMUP_DAYS  = 280   # calendar days before first snapshot date needed in cache

# ── Scoring defaults ─────────────────────────────────────────────────────────
NEUTRAL_SCORE_DEFAULT = 50.0   # score returned when underlying data is missing

# ── HTTP / rate-limit config ─────────────────────────────────────────────────
POLYGON_RATE_LIMIT_INTERVAL = 0.25  # seconds between Polygon calls (4 req/sec)
HTTP_CONNECT_TIMEOUT = 5            # seconds
HTTP_READ_TIMEOUT = 30              # seconds
MAX_PAGINATION_PAGES = 20           # Polygon pagination safety limit
FRED_RATE_LIMIT_SLEEP = 0.5        # seconds between FRED CSV fetches

# ── Price lookback helpers ───────────────────────────────────────────────────
TRADING_TO_CALENDAR_RATIO = 1.6  # multiply trading-day lookback by this for calendar days
MONTHLY_RELEASE_OFFSET = 31      # calendar days from FRED obs_date to approx month-end

# ── RRG (Relative Rotation Graph) ────────────────────────────────────────────
RRG_EMA_SPAN = 10             # EMA span for RS-Ratio smoothing
RRG_NORM_SCALE = 10           # standard-deviation multiplier for cross-sectional normalization
RRG_NORM_CENTER = 100         # center value after normalization
RRG_QUADRANT_THRESHOLD = 100  # RS-Ratio / RS-Momentum above this → Leading or Weakening
SORT_SENTINEL = -9999         # placeholder for tickers missing RS data (sorts last)

# ── Growth dimension scoring ─────────────────────────────────────────────────
# CFNAI: 3-month moving average; 0 = trend, positive = expansion, negative = contraction
CFNAI_SCORE_XP = [-2.5, -1.5, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 0.75, 1.0, 1.5]
CFNAI_SCORE_FP = [   0,    8,   18,   30,   40,  52,  62,  72,   82,  90,  100]
# Nonfarm payrolls: 3-month average monthly job gains (thousands)
PAYROLL_SCORE_XP = [-300, -100,  0,  50, 100, 150, 250, 400]
PAYROLL_SCORE_FP = [   0,   10, 25,  40,  55,  70,  85, 100]
GROWTH_CFNAI_WEIGHT = 0.5
GROWTH_PAYROLL_WEIGHT = 0.5

# ── Inflation dimension scoring ──────────────────────────────────────────────
# "Goldilocks" curve: peaks at 2.0-2.5% YoY (used for both headline and core CPI)
INFLATION_GOLDILOCKS_XP = [-1,  0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 9.0]
INFLATION_GOLDILOCKS_FP = [30, 40,  55,  70,  85, 100,  85,  60,  45,  25,   5]
# 5-year breakeven inflation rate: moderate expectations are healthy
BREAKEVEN_SCORE_XP = [0.5, 1.0, 1.5, 2.0, 2.3, 2.7, 3.5]
BREAKEVEN_SCORE_FP = [ 20,  40,  60,  85, 100,  75,  35]
INFLATION_CPI_WEIGHT = 0.35
INFLATION_CORE_WEIGHT = 0.40
INFLATION_BREAKEVEN_WEIGHT = 0.25

# ── Fed policy dimension scoring ─────────────────────────────────────────────
# 10Y-2Y yield spread: positive = steep curve = expansionary
YIELD_CURVE_SCORE_XP = [-2.0, -1.0, -0.5, 0.0, 0.3, 0.75, 1.5, 2.5, 3.5]
YIELD_CURVE_SCORE_FP = [   5,   15,   25,  40,  52,   65,  80,  92,  100]
# Effective fed funds rate: low = accommodative, high = restrictive
FED_FUNDS_SCORE_XP = [0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
FED_FUNDS_SCORE_FP = [ 80,  85,  75,  65,  55,  40,  28,  15]
FED_YIELD_CURVE_WEIGHT = 0.55
FED_FUNDS_RATE_WEIGHT = 0.45

# ── Risk appetite dimension scoring ──────────────────────────────────────────
# VIX: low vol = high risk appetite; inverted mapping
VIX_SCORE_XP = [  9,  12,  15,  18,  22,  27,  35,  50,  70]
VIX_SCORE_FP = [100,  92,  82,  68,  52,  35,  20,   8,   0]
# HY OAS spread (percentage points): tight spreads = risk-on
HY_SPREAD_SCORE_XP = [ 1.5, 2.5, 3.5, 4.5, 6.0, 8.0, 12.0, 20.0]
HY_SPREAD_SCORE_FP = [ 100,  90,  75,  55,  38,  20,    8,    0]
RISK_VIX_WEIGHT = 0.50
RISK_HY_WEIGHT = 0.50

# ── Overall macro score weights ──────────────────────────────────────────────
MACRO_GROWTH_WEIGHT = 0.25
MACRO_INFLATION_WEIGHT = 0.25
MACRO_FED_POLICY_WEIGHT = 0.25
MACRO_RISK_APPETITE_WEIGHT = 0.25

# ── Score range bounds ─────────────────────────────────────────────────────
SCORE_MIN = 0.0
SCORE_MAX = 100.0

# ── Growth scoring lookbacks ───────────────────────────────────────────────
GROWTH_FRED_LOOKBACK = 4        # months of CFNAI / payroll history to fetch
GROWTH_MA_MONTHS = 3            # months for moving-average smoothing
GROWTH_MIN_PAYROLL_OBS = 2      # minimum payroll observations needed

# ── Inflation scoring lookbacks ────────────────────────────────────────────
INFLATION_FRED_LOOKBACK = 14    # months of CPI data to fetch
INFLATION_YOY_MONTHS = 13       # offset for year-over-year calculation

# ── Sector RS helpers ──────────────────────────────────────────────────────
RS_PRICE_BUFFER_DAYS = 30       # extra calendar days fetched beyond RS lookback
RS_MIN_DATA_POINTS = 5          # minimum extra data points for valid RS calc
RS_PCT_MULTIPLIER = 100.0       # multiply ratio to get percentage

# ── Snapshot defaults ──────────────────────────────────────────────────────
SNAPSHOT_SPY_LOOKBACK = 5       # trading days of SPY to fetch for snapshot

# ── FRED defaults ──────────────────────────────────────────────────────────
FRED_DEFAULT_N_OBS = 13         # default observation window for FRED queries

# ── HTTP retry config ─────────────────────────────────────────────────────
HTTP_RETRY_TOTAL = 4            # max retries on transient errors
HTTP_RETRY_BACKOFF = 1.0        # exponential backoff factor (seconds)
HTTP_RETRY_JITTER = 0.3         # random jitter added to backoff

# ── Price / Polygon defaults ──────────────────────────────────────────────
REFRESH_PRICE_DAYS_BACK = 20    # default calendar days for price cache refresh
PRICE_SERIES_DEFAULT_LOOKBACK = 300  # default trading-day lookback for price series
POLYGON_AGGS_LIMIT = 50_000    # max bars per Polygon aggs request


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score(value: float, xp: List[float], fp: List[float]) -> float:
    """Map a raw value to a 0-100 score via piecewise-linear interpolation."""
    if value is None or np.isnan(value):
        return NEUTRAL_SCORE_DEFAULT
    return float(np.clip(np.interp(value, xp, fp), SCORE_MIN, SCORE_MAX))


# ─────────────────────────────────────────────────────────────────────────────
# MacroSnapshotEngine
# ─────────────────────────────────────────────────────────────────────────────

class MacroSnapshotEngine:
    """
    Single engine for macro snapshot generation.

    Usage pattern:
        engine = MacroSnapshotEngine(polygon_key="...", fred_key="...")
        engine.prefetch_all_data(start_date=date(2019, 7, 1), end_date=date.today())
        snap = engine.generate_snapshot(date(2024, 1, 5))
    """

    def __init__(
        self,
        polygon_key: str,
        fred_key: Optional[str] = None,  # unused — kept for API compatibility
        cache_dir: str = "data/macro_cache",
    ):
        if not polygon_key:
            raise ValueError("polygon_key is required")

        self.polygon_key = polygon_key
        self.fred_key = fred_key  # not required; public CSV endpoint used instead
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # HTTP session: retries on 429 / 5xx
        # E12: separate connect timeout (5s) and read timeout (30s) to avoid
        # hanging indefinitely on slow or unresponsive endpoints
        self._session = requests.Session()
        _retry = Retry(
            total=HTTP_RETRY_TOTAL,
            backoff_factor=HTTP_RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_jitter=HTTP_RETRY_JITTER,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=_retry))
        self._last_polygon_call = 0.0
        self._polygon_min_interval = POLYGON_RATE_LIMIT_INTERVAL
        self._http_timeout = (HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT)

        # SQLite cache path — connections are opened per-call (E1: thread-safe)
        self._db_path = self.cache_dir / "macro_cache.db"
        self._init_db()

        logger.info("MacroSnapshotEngine initialized (cache: %s)", self._db_path)

    # ── DB connection — per-call (E1: thread-safe) ─────────────────────────────

    @contextmanager
    def _cache_conn(self):
        """Open a fresh connection to the macro cache DB per call.

        E1 fix: removes the shared self._conn that was not thread-safe.
        Each call to a DB-backed method opens, uses, and closes its own
        connection, making concurrent snapshot generation safe.
        """
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── DB schema ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._cache_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS price_cache (
                    ticker  TEXT NOT NULL,
                    date    TEXT NOT NULL,
                    close   REAL NOT NULL,
                    PRIMARY KEY (ticker, date)
                );

                CREATE TABLE IF NOT EXISTS fred_cache (
                    series_id TEXT NOT NULL,
                    obs_date  TEXT NOT NULL,
                    value     REAL,
                    PRIMARY KEY (series_id, obs_date)
                );

                CREATE TABLE IF NOT EXISTS fetch_log (
                    key   TEXT PRIMARY KEY,
                    value TEXT
                );
            """)

    def _log_fetched(self, key: str, value: str = "1") -> None:
        with self._cache_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO fetch_log (key, value) VALUES (?, ?)",
                (key, value),
            )

    def _is_fetched(self, key: str) -> bool:
        with self._cache_conn() as conn:
            row = conn.execute(
                "SELECT value FROM fetch_log WHERE key = ?", (key,)
            ).fetchone()
        return row is not None

    # ── Polygon: price data ────────────────────────────────────────────────────

    def _polygon_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_polygon_call
        if elapsed < self._polygon_min_interval:
            time.sleep(self._polygon_min_interval - elapsed)
        self._last_polygon_call = time.monotonic()

    def _fetch_polygon_aggs(
        self, ticker: str, from_date: date, to_date: date
    ) -> List[Dict]:
        """Fetch adjusted daily OHLCV bars from Polygon for a single ticker."""
        self._polygon_rate_limit()
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")
        path = f"/v2/aggs/ticker/{ticker}/range/1/day/{from_str}/{to_str}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": POLYGON_AGGS_LIMIT,
            "apiKey": self.polygon_key,
        }
        try:
            resp = self._session.get(
                f"{POLYGON_BASE}{path}", params=params, timeout=self._http_timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("Polygon fetch failed for %s: %s", ticker, exc)
            return []

        results = data.get("results") or []

        # Handle pagination (unlikely for 6yr daily but be safe)
        next_url = data.get("next_url")
        pages = 0
        while next_url and pages < MAX_PAGINATION_PAGES:
            self._polygon_rate_limit()
            try:
                resp = self._session.get(
                    next_url, params={"apiKey": self.polygon_key}, timeout=self._http_timeout
                )
                resp.raise_for_status()
                page = resp.json()
                results.extend(page.get("results") or [])
                next_url = page.get("next_url")
                pages += 1
            except requests.RequestException as exc:
                logger.warning("Polygon pagination failed: %s", exc)
                break

        return results

    def _store_prices(self, ticker: str, bars: List[Dict]) -> None:
        rows = []
        for bar in bars:
            ts_ms = bar.get("t")
            close = bar.get("c")
            if ts_ms is None or close is None:
                continue
            dt = date.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
            rows.append((ticker, dt, float(close)))
        if rows:
            with self._cache_conn() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO price_cache (ticker, date, close) VALUES (?, ?, ?)",
                    rows,
                )

    def prefetch_prices(
        self, tickers: List[str], start_date: date, end_date: date
    ) -> None:
        """Download and cache adjusted daily closes for all tickers.

        E6: On fetch failure (empty result), we do NOT mark the key as fetched
        so that the next run will retry the download for that ticker.
        """
        all_tickers = list(set(tickers + [BENCHMARK]))
        for ticker in all_tickers:
            key = f"prices:{ticker}:{start_date}:{end_date}"
            if self._is_fetched(key):
                logger.debug("Price cache hit for %s — skipping fetch", ticker)
                continue
            logger.info("Fetching prices: %s  %s → %s", ticker, start_date, end_date)
            bars = self._fetch_polygon_aggs(ticker, start_date, end_date)
            if bars:
                self._store_prices(ticker, bars)
                self._log_fetched(key)  # E6: only mark fetched on success
                logger.info("  Stored %d bars for %s", len(bars), ticker)
            else:
                # E6: do NOT log as fetched — next run will retry
                logger.warning("  No bars returned for %s — will retry next run", ticker)

    def _get_price_series(
        self, ticker: str, as_of_date: date, lookback_days: int = PRICE_SERIES_DEFAULT_LOOKBACK
    ) -> pd.Series:
        """
        Return a date-indexed close price Series ending on or before as_of_date,
        covering at least lookback_days trading days (we fetch a bit extra).
        """
        earliest = as_of_date - timedelta(days=int(lookback_days * TRADING_TO_CALENDAR_RATIO))
        with self._cache_conn() as conn:
            rows = conn.execute(
                """
                SELECT date, close FROM price_cache
                WHERE ticker = ?
                  AND date >= ?
                  AND date <= ?
                ORDER BY date ASC
                """,
                (ticker, earliest.strftime("%Y-%m-%d"), as_of_date.strftime("%Y-%m-%d")),
            ).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        dates = [r["date"] for r in rows]
        closes = [r["close"] for r in rows]
        return pd.Series(closes, index=pd.to_datetime(dates))

    # ── FRED: macro data (public CSV endpoint, no API key required) ───────────
    # FRED's public download URL returns the full series as CSV:
    #   https://fred.stlouisfed.org/graph/fredgraph.csv?id={SERIES_ID}
    # This endpoint is freely available without authentication.

    def _fetch_fred_public_csv(self, series_id: str) -> List[Tuple[str, Optional[float]]]:
        """
        Fetch full history of a FRED series via the public CSV download endpoint.
        Returns list of (date_str, value) tuples.  No API key required.
        """
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            resp = self._session.get(url, timeout=self._http_timeout)
            resp.raise_for_status()
            rows = []
            for i, line in enumerate(resp.text.strip().splitlines()):
                if i == 0:
                    continue  # skip header "observation_date,{series_id}"
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                dt_str, val_str = parts[0].strip(), parts[1].strip()
                try:
                    value: Optional[float] = float(val_str) if val_str not in (".", "") else None
                except ValueError:
                    value = None
                rows.append((dt_str, value))
            return rows
        except requests.RequestException as exc:
            logger.error("FRED public CSV fetch failed for %s: %s", series_id, exc)
            return []

    def _store_fred(self, series_id: str, rows: List[Tuple[str, Optional[float]]]) -> None:
        with self._cache_conn() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO fred_cache (series_id, obs_date, value) VALUES (?, ?, ?)",
                [(series_id, dt, val) for dt, val in rows],
            )

    def prefetch_fred(self, start_date: date, end_date: date) -> None:
        """Download and cache all FRED series via the public CSV endpoint."""
        for series_id in FRED_SERIES:
            key = f"fred_csv:{series_id}"
            if self._is_fetched(key):
                logger.debug("FRED cache hit for %s — skipping", series_id)
                continue
            logger.info("Fetching FRED (public CSV): %s  [%s]", series_id, FRED_SERIES[series_id]["name"])
            rows = self._fetch_fred_public_csv(series_id)
            if rows:
                self._store_fred(series_id, rows)
                self._log_fetched(key)
                logger.info("  Stored %d observations for %s", len(rows), series_id)
            else:
                logger.warning("  No observations returned for %s", series_id)
            time.sleep(FRED_RATE_LIMIT_SLEEP)

    def prefetch_all_data(self, start_date: date, end_date: date) -> None:
        """Bulk download all prices and FRED series (call once before generate loop)."""
        logger.info("Prefetching all data: %s → %s", start_date, end_date)
        self.prefetch_prices(list(ALL_ETF_TICKERS.keys()), start_date, end_date)
        self.prefetch_fred(start_date, end_date)
        logger.info("Prefetch complete.")

    def _get_fred_value(
        self, series_id: str, as_of_date: date
    ) -> Optional[float]:
        """
        Return the most recent FRED observation for series_id that was
        actually published on or before as_of_date (applying release lag).
        """
        lag_days = FRED_SERIES[series_id]["lag_days"]
        # For a monthly series with obs_date = first-of-month (e.g., 2024-01-01),
        # the data covers January 2024 and is published ~lag_days after month-end
        # (i.e., 2024-01-31 + lag_days).  We approximate: available_date ≈
        # obs_date + 28 (days in shortest month) + lag_days for monthly series.
        freq = FRED_SERIES[series_id]["freq"]
        if freq == "monthly":
            # obs_date is the first of the reference month; add ~31 days for month-end
            # then add lag_days for publication delay
            cutoff_offset = MONTHLY_RELEASE_OFFSET + lag_days
        else:
            cutoff_offset = lag_days

        cutoff_date = (as_of_date - timedelta(days=cutoff_offset)).strftime("%Y-%m-%d")

        with self._cache_conn() as conn:
            row = conn.execute(
                """
                SELECT value FROM fred_cache
                WHERE series_id = ?
                  AND obs_date <= ?
                  AND value IS NOT NULL
                ORDER BY obs_date DESC
                LIMIT 1
                """,
                (series_id, cutoff_date),
            ).fetchone()
        return float(row["value"]) if row else None

    def _get_fred_series_window(
        self, series_id: str, as_of_date: date, n_obs: int = FRED_DEFAULT_N_OBS
    ) -> pd.Series:
        """Return the last n_obs FRED observations available as of as_of_date."""
        lag_days = FRED_SERIES[series_id]["lag_days"]
        freq = FRED_SERIES[series_id]["freq"]
        cutoff_offset = (MONTHLY_RELEASE_OFFSET + lag_days) if freq == "monthly" else lag_days
        cutoff_date = (as_of_date - timedelta(days=cutoff_offset)).strftime("%Y-%m-%d")

        with self._cache_conn() as conn:
            rows = conn.execute(
                """
                SELECT obs_date, value FROM fred_cache
                WHERE series_id = ?
                  AND obs_date <= ?
                  AND value IS NOT NULL
                ORDER BY obs_date DESC
                LIMIT ?
                """,
                (series_id, cutoff_date, n_obs),
            ).fetchall()
        if not rows:
            return pd.Series(dtype=float)
        rows = list(reversed(rows))
        dates = [r["obs_date"] for r in rows]
        values = [r["value"] for r in rows]
        return pd.Series(values, index=pd.to_datetime(dates))

    # ── Sector RS computation ──────────────────────────────────────────────────

    def _compute_sector_rs(self, as_of_date: date) -> List[Dict]:
        """
        Compute 3M / 12M relative strength, RRG quadrant, and rankings
        for all ETF tickers vs SPY.

        Returns a list of dicts sorted by rs_3m descending.
        """
        spy = self._get_price_series(BENCHMARK, as_of_date, RS_12M_DAYS + RS_PRICE_BUFFER_DAYS)
        if spy.empty or len(spy) < RS_3M_DAYS + RS_MIN_DATA_POINTS:
            logger.warning("Insufficient SPY data for %s", as_of_date)
            return []

        spy_now = spy.iloc[-1]
        spy_3m = spy.iloc[-RS_3M_DAYS] if len(spy) >= RS_3M_DAYS else None
        spy_12m = spy.iloc[-RS_12M_DAYS] if len(spy) >= RS_12M_DAYS else None

        results = []
        rrg_raw = {}  # ticker → (rs_ratio_raw, rs_mom_raw) for cross-sectional norm

        for ticker, name in ALL_ETF_TICKERS.items():
            s = self._get_price_series(ticker, as_of_date, RS_12M_DAYS + RS_PRICE_BUFFER_DAYS)
            if s.empty or len(s) < RS_3M_DAYS + RS_MIN_DATA_POINTS:
                continue

            close_now = s.iloc[-1]

            # Simple RS: how much did ticker outperform SPY?
            rs_3m: Optional[float] = None
            rs_12m: Optional[float] = None
            if spy_3m is not None and len(s) >= RS_3M_DAYS:
                tick_3m = s.iloc[-RS_3M_DAYS]
                if tick_3m > 0 and spy_3m > 0:
                    tick_ret_3m = close_now / tick_3m
                    spy_ret_3m = spy_now / spy_3m
                    rs_3m = (tick_ret_3m / spy_ret_3m - 1.0) * RS_PCT_MULTIPLIER  # % outperformance

            if spy_12m is not None and len(s) >= RS_12M_DAYS:
                tick_12m = s.iloc[-RS_12M_DAYS]
                if tick_12m > 0 and spy_12m > 0:
                    tick_ret_12m = close_now / tick_12m
                    spy_ret_12m = spy_now / spy_12m
                    rs_12m = (tick_ret_12m / spy_ret_12m - 1.0) * RS_PCT_MULTIPLIER

            # RRG RS-Ratio: ratio of ticker/SPY, smoothed
            # Build a daily rel-strength series for the past ~100 trading days
            rel_series = s / spy.reindex(s.index).ffill()
            rel_series = rel_series.dropna()
            if len(rel_series) >= RS_MOM_DAYS + RRG_EMA_SPAN:
                rs_ratio_val = float(rel_series.ewm(span=RRG_EMA_SPAN, adjust=False).mean().iloc[-1])
                # Momentum = RS-Ratio change over last RS_MOM_DAYS
                rs_ratio_series = rel_series.ewm(span=RRG_EMA_SPAN, adjust=False).mean()
                if len(rs_ratio_series) >= RS_MOM_DAYS:
                    rs_mom_val = float(
                        (rs_ratio_series.iloc[-1] / rs_ratio_series.iloc[-RS_MOM_DAYS] - 1.0) * RS_PCT_MULTIPLIER
                    )
                else:
                    rs_mom_val = 0.0
                rrg_raw[ticker] = (rs_ratio_val, rs_mom_val)

            results.append({
                "ticker": ticker,
                "name": name,
                "category": "sector" if ticker in SECTOR_ETFS else "thematic",
                "close": round(float(close_now), 2),
                "rs_3m": round(rs_3m, 2) if rs_3m is not None else None,
                "rs_12m": round(rs_12m, 2) if rs_12m is not None else None,
                "rrg_quadrant": None,  # filled below
                "rs_ratio": None,
                "rs_momentum": None,
            })

        # Cross-sectional normalize RRG values to center at 100
        if rrg_raw:
            ratio_vals = [v[0] for v in rrg_raw.values()]
            mom_vals = [v[1] for v in rrg_raw.values()]
            ratio_mean, ratio_std = np.mean(ratio_vals), np.std(ratio_vals) or 1.0
            mom_mean, mom_std = np.mean(mom_vals), np.std(mom_vals) or 1.0

            for row in results:
                t = row["ticker"]
                if t in rrg_raw:
                    raw_ratio, raw_mom = rrg_raw[t]
                    norm_ratio = (raw_ratio - ratio_mean) / ratio_std * RRG_NORM_SCALE + RRG_NORM_CENTER
                    norm_mom = (raw_mom - mom_mean) / mom_std * RRG_NORM_SCALE + RRG_NORM_CENTER
                    row["rs_ratio"] = round(float(norm_ratio), 2)
                    row["rs_momentum"] = round(float(norm_mom), 2)
                    # Quadrant classification
                    if norm_ratio >= RRG_QUADRANT_THRESHOLD and norm_mom >= RRG_QUADRANT_THRESHOLD:
                        row["rrg_quadrant"] = "Leading"
                    elif norm_ratio >= RRG_QUADRANT_THRESHOLD and norm_mom < RRG_QUADRANT_THRESHOLD:
                        row["rrg_quadrant"] = "Weakening"
                    elif norm_ratio < RRG_QUADRANT_THRESHOLD and norm_mom < RRG_QUADRANT_THRESHOLD:
                        row["rrg_quadrant"] = "Lagging"
                    else:
                        row["rrg_quadrant"] = "Improving"

        # Rank by rs_3m (tickers without rs_3m go to end)
        results.sort(
            key=lambda x: x["rs_3m"] if x["rs_3m"] is not None else SORT_SENTINEL,
            reverse=True,
        )
        for i, row in enumerate(results):
            row["rank_3m"] = i + 1

        # Also add rank_12m
        sorted_12m = sorted(
            results,
            key=lambda x: x["rs_12m"] if x["rs_12m"] is not None else SORT_SENTINEL,
            reverse=True,
        )
        rank_map = {row["ticker"]: i + 1 for i, row in enumerate(sorted_12m)}
        for row in results:
            row["rank_12m"] = rank_map.get(row["ticker"])

        return results

    # ── Macro score ────────────────────────────────────────────────────────────

    def _score_growth(self, as_of_date: date) -> Tuple[float, Dict]:
        """
        Growth score from CFNAI (Chicago Fed National Activity Index) + nonfarm payrolls.

        CFNAI is a composite of 85 monthly indicators centered at 0:
          positive = above-trend growth, negative = below-trend.
          A 3-month MA above +0.70 historically signals expansion with inflation pressure.
          A 3-month MA below -0.70 historically signals recession risk.
        """
        cfnai_series = self._get_fred_series_window("CFNAI", as_of_date, GROWTH_FRED_LOOKBACK)
        payems_series = self._get_fred_series_window("PAYEMS", as_of_date, GROWTH_FRED_LOOKBACK)

        # Use 3-month moving average of CFNAI for stability
        cfnai_3m: Optional[float] = None
        cfnai_score = NEUTRAL_SCORE_DEFAULT
        if len(cfnai_series) >= 1:
            cfnai_3m = float(cfnai_series.iloc[-min(GROWTH_MA_MONTHS, len(cfnai_series)):].mean())
            cfnai_score = _score(cfnai_3m, CFNAI_SCORE_XP, CFNAI_SCORE_FP)

        # 3-month avg of monthly job additions (in thousands)
        payroll_3m_avg: Optional[float] = None
        payroll_score = NEUTRAL_SCORE_DEFAULT
        if len(payems_series) >= GROWTH_MIN_PAYROLL_OBS:
            diffs = payems_series.diff().dropna()
            if len(diffs) >= 1:
                payroll_3m_avg = float(diffs.iloc[-min(GROWTH_MA_MONTHS, len(diffs)):].mean())
                payroll_score = _score(payroll_3m_avg, PAYROLL_SCORE_XP, PAYROLL_SCORE_FP)

        score = cfnai_score * GROWTH_CFNAI_WEIGHT + payroll_score * GROWTH_PAYROLL_WEIGHT
        indicators = {
            "cfnai_3m": round(cfnai_3m, 3) if cfnai_3m is not None else None,
            "payrolls_3m_avg_k": round(payroll_3m_avg, 1) if payroll_3m_avg is not None else None,
        }
        return round(score, 1), indicators

    def _score_inflation(self, as_of_date: date) -> Tuple[float, Dict]:
        """
        Inflation score: Goldilocks (1.5-3%) is best.
        Deflation (<0%) and high inflation (>6%) both score low.
        """
        cpi_series = self._get_fred_series_window("CPIAUCSL", as_of_date, INFLATION_FRED_LOOKBACK)
        core_series = self._get_fred_series_window("CPILFESL", as_of_date, INFLATION_FRED_LOOKBACK)
        breakeven = self._get_fred_value("T5YIE", as_of_date)

        def yoy_pct(s: pd.Series) -> Optional[float]:
            if len(s) < INFLATION_YOY_MONTHS:
                return None
            return float((s.iloc[-1] / s.iloc[-INFLATION_YOY_MONTHS] - 1.0) * RS_PCT_MULTIPLIER)

        cpi_yoy = yoy_pct(cpi_series)
        core_yoy = yoy_pct(core_series)

        cpi_score = _score(cpi_yoy, INFLATION_GOLDILOCKS_XP, INFLATION_GOLDILOCKS_FP)
        core_score = _score(core_yoy, INFLATION_GOLDILOCKS_XP, INFLATION_GOLDILOCKS_FP)
        be_score = _score(breakeven, BREAKEVEN_SCORE_XP, BREAKEVEN_SCORE_FP)

        weights = [INFLATION_CPI_WEIGHT, INFLATION_CORE_WEIGHT, INFLATION_BREAKEVEN_WEIGHT]
        scores = [cpi_score, core_score, be_score]
        score = sum(w * s for w, s in zip(weights, scores))

        indicators = {
            "cpi_yoy_pct": round(cpi_yoy, 2) if cpi_yoy else None,
            "core_cpi_yoy_pct": round(core_yoy, 2) if core_yoy else None,
            "breakeven_5y": round(breakeven, 3) if breakeven else None,
        }
        return round(score, 1), indicators

    def _score_fed_policy(self, as_of_date: date) -> Tuple[float, Dict]:
        """
        Fed policy score: accommodative rates + steep yield curve = good for equities.
        """
        t10y2y = self._get_fred_value("T10Y2Y", as_of_date)
        fedfunds = self._get_fred_value("FEDFUNDS", as_of_date)

        # Steep/positive yield curve is expansionary
        yc_score = _score(t10y2y, YIELD_CURVE_SCORE_XP, YIELD_CURVE_SCORE_FP)

        # Low rates are accommodative; tight rates reduce equity multiples
        ff_score = _score(fedfunds, FED_FUNDS_SCORE_XP, FED_FUNDS_SCORE_FP)

        score = yc_score * FED_YIELD_CURVE_WEIGHT + ff_score * FED_FUNDS_RATE_WEIGHT
        indicators = {
            "t10y2y": round(t10y2y, 3) if t10y2y else None,
            "fedfunds": round(fedfunds, 3) if fedfunds else None,
        }
        return round(score, 1), indicators

    def _score_risk_appetite(self, as_of_date: date) -> Tuple[float, Dict]:
        """
        Risk appetite score from VIX + HY spreads.
        Low vol and tight spreads = high risk appetite.
        """
        vix = self._get_fred_value("VIXCLS", as_of_date)
        hy_spread = self._get_fred_value("BAMLH0A0HYM2", as_of_date)

        vix_score = _score(vix, VIX_SCORE_XP, VIX_SCORE_FP)

        # HY OAS in percentage points (FRED stores as %)
        hy_score = _score(hy_spread, HY_SPREAD_SCORE_XP, HY_SPREAD_SCORE_FP)

        score = vix_score * RISK_VIX_WEIGHT + hy_score * RISK_HY_WEIGHT
        indicators = {
            "vix": round(vix, 2) if vix else None,
            "hy_oas_pct": round(hy_spread, 3) if hy_spread else None,
        }
        return round(score, 1), indicators

    def _compute_macro_score(self, as_of_date: date) -> Dict:
        """Compute all 4 macro score dimensions. Returns dict with overall + dimensions."""
        growth_score, growth_ind = self._score_growth(as_of_date)
        inflation_score, inflation_ind = self._score_inflation(as_of_date)
        fed_score, fed_ind = self._score_fed_policy(as_of_date)
        risk_score, risk_ind = self._score_risk_appetite(as_of_date)

        # Validate individual dimension scores (guard against NaN propagation)
        def _safe_score(s: float, name: str) -> float:
            if s is None or np.isnan(s):
                logger.warning(
                    "Dimension score '%s' is NaN for %s — using neutral %.1f",
                    name, as_of_date, NEUTRAL_SCORE_DEFAULT,
                )
                return NEUTRAL_SCORE_DEFAULT
            return float(np.clip(s, SCORE_MIN, SCORE_MAX))

        growth_score    = _safe_score(growth_score,    "growth")
        inflation_score = _safe_score(inflation_score, "inflation")
        fed_score       = _safe_score(fed_score,       "fed_policy")
        risk_score      = _safe_score(risk_score,      "risk_appetite")

        overall = round(
            float(np.clip(
                growth_score * MACRO_GROWTH_WEIGHT
                + inflation_score * MACRO_INFLATION_WEIGHT
                + fed_score * MACRO_FED_POLICY_WEIGHT
                + risk_score * MACRO_RISK_APPETITE_WEIGHT,
                SCORE_MIN, SCORE_MAX,
            )),
            1,
        )

        all_indicators = {
            **growth_ind,
            **inflation_ind,
            **fed_ind,
            **risk_ind,
        }

        return {
            "overall": overall,
            "growth": growth_score,
            "inflation": inflation_score,
            "fed_policy": fed_score,
            "risk_appetite": risk_score,
            "indicators": all_indicators,
        }

    # ── Main: generate_snapshot ────────────────────────────────────────────────

    def generate_snapshot(self, as_of_date: date) -> Dict:
        """
        Generate a complete macro snapshot for as_of_date.

        Returns a dict with:
          - date
          - spy_close
          - sector_rankings  (sorted by rs_3m)
          - macro_score
          - top_sector_3m, top_sector_12m
          - leading_sectors, lagging_sectors
        """
        spy_series = self._get_price_series(BENCHMARK, as_of_date, SNAPSHOT_SPY_LOOKBACK)
        spy_close = float(spy_series.iloc[-1]) if not spy_series.empty else None

        sector_rankings = self._compute_sector_rs(as_of_date)
        macro_score = self._compute_macro_score(as_of_date)

        # E8: Snapshot completeness validation
        n_sectors = len(sector_rankings)
        expected_sectors = len(ALL_ETF_TICKERS)
        if n_sectors < expected_sectors:
            logger.warning(
                "Incomplete snapshot for %s: got %d/%d sectors — "
                "some ETF price data may be missing",
                as_of_date, n_sectors, expected_sectors,
            )
        if spy_close is None:
            logger.warning("Snapshot for %s: SPY close is missing", as_of_date)
        for dim in ("growth", "inflation", "fed_policy", "risk_appetite"):
            if macro_score.get(dim) is None:
                logger.warning(
                    "Snapshot for %s: macro_score.%s is None (FRED data missing)", as_of_date, dim
                )

        # E9: Week-over-week velocity (score_velocity, risk_app_velocity)
        # Look up the prior week's snapshot from macro_state.db to compute delta
        try:
            from compass.macro_db import get_db as _get_state_db
            with _get_state_db() as _sc:
                prior = _sc.execute(
                    "SELECT overall, risk_appetite FROM macro_score "
                    "WHERE date < ? ORDER BY date DESC LIMIT 1",
                    (as_of_date.strftime("%Y-%m-%d"),),
                ).fetchone()
            if prior and prior["overall"] is not None and macro_score.get("overall") is not None:
                macro_score["score_velocity"] = round(
                    float(macro_score["overall"]) - float(prior["overall"]), 2
                )
            else:
                macro_score["score_velocity"] = None
            if prior and prior["risk_appetite"] is not None and macro_score.get("risk_appetite") is not None:
                macro_score["risk_app_velocity"] = round(
                    float(macro_score["risk_appetite"]) - float(prior["risk_appetite"]), 2
                )
            else:
                macro_score["risk_app_velocity"] = None
        except Exception as exc:
            logger.debug("Velocity computation skipped: %s", exc)
            macro_score.setdefault("score_velocity", None)
            macro_score.setdefault("risk_app_velocity", None)

        # Convenience summaries
        top_sector_3m = sector_rankings[0]["ticker"] if sector_rankings else None
        ranked_12m = sorted(
            [s for s in sector_rankings if s["rs_12m"] is not None],
            key=lambda x: x["rs_12m"],
            reverse=True,
        )
        top_sector_12m = ranked_12m[0]["ticker"] if ranked_12m else None

        leading = [s["ticker"] for s in sector_rankings if s["rrg_quadrant"] == "Leading"]
        lagging = [s["ticker"] for s in sector_rankings if s["rrg_quadrant"] == "Lagging"]

        return {
            "date": as_of_date.strftime("%Y-%m-%d"),
            "spy_close": round(spy_close, 2) if spy_close else None,
            "top_sector_3m": top_sector_3m,
            "top_sector_12m": top_sector_12m,
            "leading_sectors": leading,
            "lagging_sectors": lagging,
            "sector_rankings": sector_rankings,
            "macro_score": macro_score,
        }

    def save_to_db(self, snap: Dict, db_path: Optional[str] = None) -> None:
        """
        Persist a snapshot dict into macro_state.db.
        Thin wrapper around shared.macro_state_db.save_snapshot().
        Includes score_velocity and risk_app_velocity if computed by generate_snapshot.
        """
        from compass.macro_db import save_snapshot, set_state
        save_snapshot(snap, db_path=db_path)
        set_state("last_weekly_snapshot", snap["date"], db_path=db_path)
        logger.info("Snapshot %s saved to macro_state.db", snap["date"])

    def refresh_price_cache(self, days_back: int = REFRESH_PRICE_DAYS_BACK) -> None:
        """
        Fetch and cache the most recent N calendar days of price data.
        Used by the weekly job to keep the price cache current without a full re-fetch.
        """
        end = date.today()
        start = end - timedelta(days=days_back)
        logger.info("Refreshing price cache: last %d days (%s → %s)", days_back, start, end)
        # Clear fetch_log entries so prefetch_prices will re-fetch recent bars
        tickers = list(ALL_ETF_TICKERS.keys()) + [BENCHMARK]
        for ticker in tickers:
            # Insert new prices; IGNORE means existing bars won't duplicate
            bars = self._fetch_polygon_aggs(ticker, start, end)
            if bars:
                self._store_prices(ticker, bars)
        logger.info("Price cache refresh complete.")

    def refresh_fred_cache(self) -> None:
        """
        Re-fetch all FRED series (full history refresh).
        Existing rows are preserved (INSERT OR IGNORE); new obs are added.
        The fetch_log keys for fred_csv are cleared so new data lands.
        """
        with self._cache_conn() as conn:
            for series_id in FRED_SERIES:
                key = f"fred_csv:{series_id}"
                conn.execute("DELETE FROM fetch_log WHERE key = ?", (key,))
        self.prefetch_fred(date(2000, 1, 1), date.today())

    def close(self) -> None:
        """Close HTTP session (DB connections are per-call; no persistent conn to close)."""
        try:
            self._session.close()
        except Exception:
            pass
