"""
Historical composite score engine for backtesting.

Reconstructs the composite score for any past date using locally cached data.

Data sources
------------
  Source              What we store                   How far back
  ------------------  ------------------------------  -----------------
  alternative.me      Fear & Greed index (daily)      2018-02-01
  CoinGecko           BTC close + BTC market cap      2013-04-28
  CoinGecko           Total crypto market cap         2013-04-28
  OKX history API     BTC perp funding settlements    ~2019-09-13
  (derived)           Realized vol, MA200, dominance  computed locally

Lookahead rules
---------------
When computing the score for date D (backtest scan at ~9:15 AM ET = 14:15 UTC):

  Signal               Source date / cutoff
  -------------------  -------------------------------------------------
  fear_greed_index     D  (published midnight UTC — before market open)
  ma200_position       Closes through D-1  (D's close not yet known)
  realized_vol (→IV)   Closes through D-1
  funding_rate         Most recent OKX settlement before 14:00 UTC on D
  btc_dominance        D  (CoinGecko market cap updates from prior close)

Historical IV proxy
-------------------
Deribit IV history is not freely available for multi-year backtests.
We approximate:

    iv_rv_spread = realized_vol × IV_PREMIUM

where IV_PREMIUM = 0.10 (10% constant premium above RV).  This matches
the long-run crypto IV/RV ratio of ~1.05-1.20.  Effect: high RV periods
(crashes, breakouts) generate a large spread → fear component → lower score.

Usage
-----
    from datetime import date
    from compass.crypto.historical_score import HistoricalScoreCache, build_score_series

    cache = HistoricalScoreCache("data/crypto_score_cache.db")
    cache.populate(date(2020, 1, 1), date(2025, 12, 31))   # one-time, ~20 s

    scores = build_score_series(date(2020, 1, 1), date(2025, 12, 31), cache)
    for d, result in sorted(scores.items()):
        print(d, result["score"], result["band"], result["regime"])
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from compass.crypto.composite_score import compute_composite_score
from compass.crypto.realized_vol import compute_realized_vol
from compass.crypto.regime import compute_ma200_position

_LOG = logging.getLogger(__name__)

# Proxy: IV = RV × (1 + IV_PREMIUM).  iv_rv_spread = IV - RV = RV × IV_PREMIUM.
_DEFAULT_IV_PREMIUM: float = 0.10

# How many BTC close prices to keep in the rolling buffer:
# 200 (MA200) + 30 (RV window) + 5 (buffer for missing trading days)
_PRICE_LOOKBACK_DAYS: int = 235

# CoinGecko public free tier: 30 req/min → 2 s between calls
_CG_THROTTLE_SECS: float = 2.0
_CG_BASE_URL = "https://api.coingecko.com/api/v3"
_CG_TIMEOUT   = 30

# OKX public market data
_OKX_BASE_URL  = "https://www.okx.com/api/v5/public"
_OKX_TIMEOUT   = 15
_OKX_PAGE_SIZE = 100          # max records per page
_OKX_INST_BTC  = "BTC-USDT-SWAP"

# alternative.me Fear & Greed (limit 2000 ≈ 5.5 years)
_FNG_URL       = "https://api.alternative.me/fng/"
_FNG_TIMEOUT   = 10
_FNG_MAX_LIMIT = 2000


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class HistoricalScoreCache:
    """SQLite-backed cache for historical crypto score inputs.

    Workflow:
        1. Call ``populate(start_date, end_date)`` once to fetch and store all
           necessary data from external APIs.
        2. Call ``build_historical_score`` / ``build_score_series`` — these
           read **only** from the local DB; no network access.

    Safe to call ``populate`` multiple times: already-cached rows are skipped
    (INSERT OR REPLACE / INSERT OR IGNORE semantics).

    Args:
        db_path: SQLite file path.  Use ``":memory:"`` for in-process testing.
    """

    def __init__(self, db_path: str = "data/crypto_score_cache.db") -> None:
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        self._last_cg_call: float = 0.0

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS btc_daily (
                date              TEXT PRIMARY KEY,   -- YYYY-MM-DD
                close             REAL NOT NULL,
                btc_market_cap_usd REAL
            );
            CREATE TABLE IF NOT EXISTS total_market_daily (
                date           TEXT PRIMARY KEY,      -- YYYY-MM-DD
                market_cap_usd REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS fear_greed_daily (
                date  TEXT PRIMARY KEY,               -- YYYY-MM-DD
                value INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS btc_funding_settlements (
                settlement_time_ms INTEGER PRIMARY KEY,
                funding_rate       REAL NOT NULL
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Population — API → DB
    # ------------------------------------------------------------------

    def populate(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> None:
        """Fetch all historical signal data for ``[start_date, end_date]``.

        Extends ``start_date`` backwards by ``_PRICE_LOOKBACK_DAYS`` so that
        MA200 and RV computations are warm on the very first backtest date.

        Args:
            start_date: Earliest backtest date (inclusive).
            end_date:   Latest backtest date (inclusive).
        """
        fetch_start = start_date - datetime.timedelta(days=_PRICE_LOOKBACK_DAYS)
        _LOG.info(
            "Populating cache %s–%s (price lookback from %s)",
            start_date, end_date, fetch_start,
        )
        self._populate_fear_greed(start_date, end_date)
        self._populate_btc_prices(fetch_start, end_date)
        self._populate_total_market_cap(fetch_start, end_date)
        self._populate_funding_rates(fetch_start, end_date)
        _LOG.info("Cache population complete.")

    # --- Fear & Greed ---

    def _populate_fear_greed(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> None:
        cached_dates = self._cached_dates("fear_greed_daily")
        if cached_dates and min(cached_dates) <= start_date and max(cached_dates) >= end_date:
            _LOG.debug("Fear & Greed already cached for range.")
            return

        days_needed = (end_date - start_date).days + 30
        limit = min(max(days_needed, 365), _FNG_MAX_LIMIT)
        _LOG.info("Fetching %d days of Fear & Greed history ...", limit)

        try:
            resp = requests.get(
                _FNG_URL,
                params={"limit": limit, "date_format": "us"},
                timeout=_FNG_TIMEOUT,
            )
            resp.raise_for_status()
            entries = resp.json().get("data", [])
        except Exception as exc:
            _LOG.error("Fear & Greed fetch failed: %s", exc)
            return

        rows: List[Tuple[str, int]] = []
        for entry in entries:
            try:
                ts    = int(entry["timestamp"])
                value = int(entry["value"])
                rows.append((datetime.date.fromtimestamp(ts).isoformat(), value))
            except (KeyError, TypeError, ValueError) as exc:
                _LOG.warning("Skipping malformed F&G entry: %s", exc)

        if rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO fear_greed_daily (date, value) VALUES (?, ?)",
                rows,
            )
            self._conn.commit()
            _LOG.info("Stored %d Fear & Greed rows.", len(rows))

    # --- BTC prices + market cap ---

    def _populate_btc_prices(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> None:
        cached_dates = self._cached_dates("btc_daily")
        if cached_dates and min(cached_dates) <= start_date and max(cached_dates) >= end_date:
            _LOG.debug("BTC prices already cached for range.")
            return

        from_ts = int(datetime.datetime(
            start_date.year, start_date.month, start_date.day, 0, 0, 0
        ).timestamp())
        to_ts   = int(datetime.datetime(
            end_date.year, end_date.month, end_date.day, 23, 59, 59
        ).timestamp())

        _LOG.info("Fetching BTC price history %s–%s from CoinGecko ...", start_date, end_date)
        self._cg_throttle()

        try:
            resp = requests.get(
                f"{_CG_BASE_URL}/coins/bitcoin/market_chart/range",
                params={"vs_currency": "usd", "from": from_ts, "to": to_ts},
                timeout=_CG_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            _LOG.error("CoinGecko BTC price fetch failed: %s", exc)
            return

        prices_raw = payload.get("prices", [])
        mcaps_raw  = payload.get("market_caps", [])

        close_by_date: Dict[str, float] = {}
        mcap_by_date:  Dict[str, float] = {}

        for ts_ms, price in prices_raw:
            date_str = datetime.date.fromtimestamp(ts_ms / 1000.0).isoformat()
            close_by_date[date_str] = float(price)

        for ts_ms, mcap in mcaps_raw:
            date_str = datetime.date.fromtimestamp(ts_ms / 1000.0).isoformat()
            mcap_by_date[date_str] = float(mcap)

        rows = [
            (d, close, mcap_by_date.get(d))
            for d, close in close_by_date.items()
        ]
        if rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO btc_daily "
                "(date, close, btc_market_cap_usd) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()
            _LOG.info("Stored %d BTC daily rows.", len(rows))

    # --- Total market cap ---

    def _populate_total_market_cap(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> None:
        cached_dates = self._cached_dates("total_market_daily")
        if cached_dates and min(cached_dates) <= start_date and max(cached_dates) >= end_date:
            _LOG.debug("Total market cap already cached for range.")
            return

        days = (datetime.date.today() - start_date).days + 5
        _LOG.info("Fetching total market cap history (%d days) from CoinGecko ...", days)
        self._cg_throttle()

        try:
            resp = requests.get(
                f"{_CG_BASE_URL}/global/market_cap_chart",
                params={"vs_currency": "usd", "days": days},
                timeout=_CG_TIMEOUT,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            _LOG.error("CoinGecko total market cap fetch failed: %s", exc)
            return

        raw = payload.get("market_cap_chart", {}).get("market_cap", [])
        rows = [
            (datetime.date.fromtimestamp(ts_ms / 1000.0).isoformat(), float(mcap))
            for ts_ms, mcap in raw
        ]
        if rows:
            self._conn.executemany(
                "INSERT OR REPLACE INTO total_market_daily (date, market_cap_usd) VALUES (?, ?)",
                rows,
            )
            self._conn.commit()
            _LOG.info("Stored %d total market cap rows.", len(rows))

    # --- OKX funding rates ---

    def _populate_funding_rates(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> None:
        first_needed_ms = int(datetime.datetime(
            start_date.year, start_date.month, start_date.day
        ).timestamp() * 1000)
        last_needed_ms = int(datetime.datetime(
            end_date.year, end_date.month, end_date.day, 23, 59
        ).timestamp() * 1000)

        covered_min, covered_max = self._funding_coverage()
        if (
            covered_min is not None
            and covered_max is not None
            and covered_min <= first_needed_ms
            and covered_max >= last_needed_ms
        ):
            _LOG.debug("Funding rates already cached for range.")
            return

        _LOG.info("Fetching OKX BTC funding rate history %s–%s ...", start_date, end_date)
        all_rows: List[Tuple[int, float]] = []
        # Paginate backwards from end of range
        cursor_before: Optional[int] = last_needed_ms + 1

        while cursor_before is None or cursor_before > first_needed_ms:
            params: Dict[str, Any] = {
                "instId": _OKX_INST_BTC,
                "limit":  _OKX_PAGE_SIZE,
            }
            if cursor_before is not None:
                params["before"] = str(cursor_before)

            try:
                resp = requests.get(
                    f"{_OKX_BASE_URL}/funding-rate-history",
                    params=params,
                    timeout=_OKX_TIMEOUT,
                )
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("code") != "0":
                    _LOG.error("OKX API error: %s", payload.get("msg"))
                    break
                data = payload.get("data", [])
            except Exception as exc:
                _LOG.error("OKX funding rate fetch failed: %s", exc)
                break

            if not data:
                break

            oldest_ts: Optional[int] = None
            for entry in data:
                try:
                    ts_ms = int(entry["fundingTime"])
                    rate  = float(entry.get("realizedRate") or entry["fundingRate"])
                    all_rows.append((ts_ms, rate))
                    if oldest_ts is None or ts_ms < oldest_ts:
                        oldest_ts = ts_ms
                except (KeyError, TypeError, ValueError) as exc:
                    _LOG.warning("Skipping malformed funding entry: %s", exc)

            if oldest_ts is None or oldest_ts <= first_needed_ms:
                break
            cursor_before = oldest_ts

        if all_rows:
            self._conn.executemany(
                "INSERT OR IGNORE INTO btc_funding_settlements "
                "(settlement_time_ms, funding_rate) VALUES (?, ?)",
                all_rows,
            )
            self._conn.commit()
            _LOG.info("Stored %d OKX funding settlement rows.", len(all_rows))

    # ------------------------------------------------------------------
    # Read methods — DB → caller
    # ------------------------------------------------------------------

    def get_fear_greed(self, date: datetime.date) -> Optional[int]:
        """Fear & Greed value (0-100) for ``date``, or None if not cached."""
        row = self._conn.execute(
            "SELECT value FROM fear_greed_daily WHERE date = ?",
            (date.isoformat(),),
        ).fetchone()
        return int(row[0]) if row else None

    def get_btc_closes(
        self,
        up_to_date: datetime.date,
        n_days: int,
    ) -> List[float]:
        """Up to ``n_days`` BTC close prices ending on ``up_to_date`` (inclusive).

        Returns prices in chronological order (oldest first).

        Args:
            up_to_date: Last date to include.  Pass D-1 to avoid lookahead.
            n_days:     Maximum number of days to return.
        """
        rows = self._conn.execute(
            "SELECT close FROM btc_daily WHERE date <= ? ORDER BY date DESC LIMIT ?",
            (up_to_date.isoformat(), n_days),
        ).fetchall()
        return [r[0] for r in reversed(rows)]

    def get_funding_rate_before(self, before_dt: datetime.datetime) -> Optional[float]:
        """Most recently settled BTC funding rate strictly before ``before_dt`` (UTC).

        Args:
            before_dt: Timezone-aware UTC datetime.  Settlements at exactly this
                       timestamp are excluded.
        """
        threshold_ms = int(before_dt.timestamp() * 1000)
        row = self._conn.execute(
            """SELECT funding_rate FROM btc_funding_settlements
               WHERE settlement_time_ms < ?
               ORDER BY settlement_time_ms DESC LIMIT 1""",
            (threshold_ms,),
        ).fetchone()
        return float(row[0]) if row else None

    def get_btc_dominance(self, date: datetime.date) -> Optional[float]:
        """BTC market-cap dominance as % for ``date``, or None if missing.

        Computed as btc_market_cap / total_market_cap × 100.
        """
        row = self._conn.execute(
            """SELECT b.btc_market_cap_usd, t.market_cap_usd
               FROM btc_daily b
               JOIN total_market_daily t ON b.date = t.date
               WHERE b.date = ?""",
            (date.isoformat(),),
        ).fetchone()
        if row is None or row[0] is None or row[1] is None or row[1] == 0.0:
            return None
        return round(float(row[0]) / float(row[1]) * 100.0, 2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    # SECURITY: whitelist of valid table names for _cached_dates.
    # SQLite does not support parameterized identifiers, so we validate
    # against this set before interpolating into SQL (SECURITY_AUDIT.md finding #3).
    _ALLOWED_SCORE_TABLES: frozenset = frozenset({
        "btc_daily",
        "total_market_daily",
        "fear_greed_daily",
        "btc_funding_settlements",
    })

    def _cached_dates(self, table: str) -> List[datetime.date]:
        """Return all dates present in ``table``."""
        if table not in self._ALLOWED_SCORE_TABLES:
            raise ValueError(f"Unknown score table: {table!r}")
        cur = self._conn.execute(f"SELECT date FROM {table}")  # safe: whitelisted above
        return [datetime.date.fromisoformat(r[0]) for r in cur.fetchall()]

    def _funding_coverage(self) -> Tuple[Optional[int], Optional[int]]:
        row = self._conn.execute(
            "SELECT MIN(settlement_time_ms), MAX(settlement_time_ms) "
            "FROM btc_funding_settlements"
        ).fetchone()
        return (row[0], row[1]) if row and row[0] is not None else (None, None)

    def _cg_throttle(self) -> None:
        """Enforce CoinGecko rate limit (30 calls/min)."""
        elapsed = time.monotonic() - self._last_cg_call
        if elapsed < _CG_THROTTLE_SECS:
            time.sleep(_CG_THROTTLE_SECS - elapsed)
        self._last_cg_call = time.monotonic()


# ---------------------------------------------------------------------------
# Scoring — cache → composite score
# ---------------------------------------------------------------------------

def build_historical_score(
    date: datetime.date,
    cache: HistoricalScoreCache,
    rv_window: int = 30,
    iv_premium: float = _DEFAULT_IV_PREMIUM,
) -> Dict[str, Any]:
    """Compute the composite score for a historical ``date`` with no lookahead.

    Args:
        date:       The backtest date (D) — score reflects what was knowable
                    before the open (i.e., at approximately 9:15 AM ET).
        cache:      A populated ``HistoricalScoreCache``.
        rv_window:  Rolling window in days for realized vol (default 30).
        iv_premium: IV-over-RV premium fraction used to proxy iv_rv_spread
                    (default 0.10 = 10%).

    Returns:
        The ``compute_composite_score`` result dict extended with:
            date        (datetime.date)
            regime      (str)  — regime label matching compass.crypto.regime
            diagnostics (dict) — rv, closes count, price_as_of, iv_premium

    Raises:
        ValueError: If no signals are available for ``date`` (cache empty /
                    date predates history).
    """
    prev_day = date - datetime.timedelta(days=1)

    # --- Prices through D-1 (strict no-lookahead) ---
    closes = cache.get_btc_closes(up_to_date=prev_day, n_days=_PRICE_LOOKBACK_DAYS)

    ma200: Optional[str]   = compute_ma200_position(closes) if len(closes) >= 200 else None
    rv:    Optional[float] = (
        compute_realized_vol(closes, window=rv_window)
        if len(closes) >= rv_window + 1
        else None
    )
    # iv_rv_spread = IV - RV = RV*(1+premium) - RV = RV*premium
    iv_rv_spread: Optional[float] = (rv * iv_premium) if rv is not None else None

    # --- Fear & Greed (published midnight UTC, available at open on D) ---
    fg_raw      = cache.get_fear_greed(date)
    fear_greed  = float(fg_raw) if fg_raw is not None else None

    # --- Funding rate: most recent settlement before 14:00 UTC on D ---
    scan_utc    = datetime.datetime(
        date.year, date.month, date.day, 14, 0, 0,
        tzinfo=datetime.timezone.utc,
    )
    funding_rate = cache.get_funding_rate_before(scan_utc)

    # --- BTC dominance: from D's market cap entry (prior-night close) ---
    btc_dominance = cache.get_btc_dominance(date)

    # --- Score (gracefully drops missing signals) ---
    result = compute_composite_score(
        fear_greed_index=fear_greed,
        iv_rv_spread=iv_rv_spread,
        funding_rate=funding_rate,
        ma200_position=ma200,
        btc_dominance=btc_dominance,
    )

    result["date"]   = date
    result["regime"] = _score_to_regime(result["score"])
    result["diagnostics"] = {
        "rv_30d":          round(rv, 6) if rv is not None else None,
        "iv_premium_used": iv_premium,
        "closes_count":    len(closes),
        "price_as_of":     prev_day.isoformat(),
    }
    return result


def build_score_series(
    start_date: datetime.date,
    end_date: datetime.date,
    cache: HistoricalScoreCache,
    rv_window: int = 30,
    iv_premium: float = _DEFAULT_IV_PREMIUM,
) -> Dict[datetime.date, Dict[str, Any]]:
    """Build composite scores for every calendar date in ``[start_date, end_date]``.

    Dates with no signal data are silently omitted (no partial-data errors).

    Args:
        start_date: First date to compute (inclusive).
        end_date:   Last date to compute (inclusive).
        cache:      Populated ``HistoricalScoreCache``.
        rv_window:  RV window in days (default 30).
        iv_premium: IV-over-RV premium fraction (default 0.10).

    Returns:
        ``{date: result_dict}`` for every date that had at least one signal.
        Keys are ``datetime.date`` objects; iterate ``sorted(scores.items())``
        for chronological order.
    """
    if start_date > end_date:
        return {}

    results: Dict[datetime.date, Dict[str, Any]] = {}
    current = start_date

    while current <= end_date:
        try:
            results[current] = build_historical_score(
                current, cache, rv_window, iv_premium
            )
        except ValueError:
            # No signals at all for this date — skip without noise
            pass
        except Exception as exc:
            _LOG.warning("Skipping %s — unexpected error: %s", current, exc)
        current += datetime.timedelta(days=1)

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _score_to_regime(score: float) -> str:
    """Map a 0-100 composite score to the regime label."""
    if score < 25.0:
        return "extreme_fear"
    elif score < 40.0:
        return "cautious"
    elif score < 60.0:
        return "neutral"
    elif score < 75.0:
        return "bullish"
    else:
        return "extreme_greed"
