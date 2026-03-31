"""
Tests for compass.crypto.historical_score.

Strategy:
  - HistoricalScoreCache uses ":memory:" SQLite so every test is isolated.
  - populate() methods are tested by patching requests.get with fake payloads.
  - build_historical_score / build_score_series tests use direct DB inserts
    to avoid network calls — they verify pure computation logic only.
"""

from __future__ import annotations

import datetime
import math
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from compass.crypto.historical_score import (
    HistoricalScoreCache,
    _DEFAULT_IV_PREMIUM,
    _PRICE_LOOKBACK_DAYS,
    _score_to_regime,
    build_historical_score,
    build_score_series,
)
from compass.crypto.realized_vol import compute_realized_vol


# ===========================================================================
# Helpers
# ===========================================================================

def _make_cache() -> HistoricalScoreCache:
    """In-memory cache; fully isolated per test."""
    return HistoricalScoreCache(":memory:")


def _insert_closes(
    cache: HistoricalScoreCache,
    start_date: datetime.date,
    prices: List[float],
    btc_market_caps: List[float] | None = None,
) -> None:
    """Insert daily close prices starting from start_date."""
    rows = []
    for i, price in enumerate(prices):
        d = (start_date + datetime.timedelta(days=i)).isoformat()
        mcap = btc_market_caps[i] if btc_market_caps else None
        rows.append((d, price, mcap))
    cache._conn.executemany(
        "INSERT OR REPLACE INTO btc_daily (date, close, btc_market_cap_usd) VALUES (?, ?, ?)",
        rows,
    )
    cache._conn.commit()


def _insert_total_market_caps(
    cache: HistoricalScoreCache,
    start_date: datetime.date,
    mcaps: List[float],
) -> None:
    rows = [
        ((start_date + datetime.timedelta(days=i)).isoformat(), v)
        for i, v in enumerate(mcaps)
    ]
    cache._conn.executemany(
        "INSERT OR REPLACE INTO total_market_daily (date, market_cap_usd) VALUES (?, ?)",
        rows,
    )
    cache._conn.commit()


def _insert_fear_greed(
    cache: HistoricalScoreCache,
    start_date: datetime.date,
    values: List[int],
) -> None:
    rows = [
        ((start_date + datetime.timedelta(days=i)).isoformat(), v)
        for i, v in enumerate(values)
    ]
    cache._conn.executemany(
        "INSERT OR REPLACE INTO fear_greed_daily (date, value) VALUES (?, ?)",
        rows,
    )
    cache._conn.commit()


def _insert_funding(
    cache: HistoricalScoreCache,
    settlements: List[tuple],
) -> None:
    """Insert (settlement_time_ms, funding_rate) rows."""
    cache._conn.executemany(
        "INSERT OR IGNORE INTO btc_funding_settlements "
        "(settlement_time_ms, funding_rate) VALUES (?, ?)",
        settlements,
    )
    cache._conn.commit()


def _settlement_ms(date: datetime.date, hour: int = 8) -> int:
    """UTC ms timestamp for a settlement at ``hour:00`` on ``date``."""
    dt = datetime.datetime(
        date.year, date.month, date.day, hour, 0, 0,
        tzinfo=datetime.timezone.utc,
    )
    return int(dt.timestamp() * 1000)


def _flat_prices(n: int, value: float = 50_000.0) -> List[float]:
    return [value] * n


def _trending_prices(n: int, start: float = 50_000.0, drift: float = 0.001) -> List[float]:
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + drift))
    return prices


# ===========================================================================
# HistoricalScoreCache — schema and basic reads
# ===========================================================================

class TestCacheSchema:

    def test_tables_created_on_init(self):
        cache = _make_cache()
        tables = {r[0] for r in cache._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "btc_daily" in tables
        assert "total_market_daily" in tables
        assert "fear_greed_daily" in tables
        assert "btc_funding_settlements" in tables
        cache.close()

    def test_multiple_inits_idempotent(self):
        """CREATE TABLE IF NOT EXISTS — safe to call twice."""
        cache = _make_cache()
        cache._create_tables()  # second call must not raise
        cache.close()


# ===========================================================================
# HistoricalScoreCache — read methods
# ===========================================================================

class TestCacheReads:

    def setup_method(self):
        self.cache = _make_cache()
        self.today = datetime.date(2024, 6, 15)

    def teardown_method(self):
        self.cache.close()

    # --- get_fear_greed ---

    def test_fear_greed_returns_value(self):
        _insert_fear_greed(self.cache, self.today, [42])
        assert self.cache.get_fear_greed(self.today) == 42

    def test_fear_greed_missing_returns_none(self):
        assert self.cache.get_fear_greed(self.today) is None

    def test_fear_greed_exact_date_only(self):
        other = self.today + datetime.timedelta(days=1)
        _insert_fear_greed(self.cache, other, [77])
        assert self.cache.get_fear_greed(self.today) is None

    # --- get_btc_closes ---

    def test_btc_closes_returns_chronological(self):
        prices = [100.0, 110.0, 120.0, 130.0, 140.0]
        _insert_closes(self.cache, self.today, prices)
        end = self.today + datetime.timedelta(days=4)
        result = self.cache.get_btc_closes(up_to_date=end, n_days=5)
        assert result == prices

    def test_btc_closes_respects_up_to_date(self):
        """up_to_date=D should NOT return D+1."""
        prices = [100.0, 200.0, 300.0]  # day0, day1, day2
        _insert_closes(self.cache, self.today, prices)
        # Request up to day1 — should not include day2 (300)
        d1 = self.today + datetime.timedelta(days=1)
        result = self.cache.get_btc_closes(up_to_date=d1, n_days=10)
        assert 300.0 not in result
        assert 200.0 in result

    def test_btc_closes_n_days_limit(self):
        prices = _flat_prices(10)
        _insert_closes(self.cache, self.today, prices)
        end = self.today + datetime.timedelta(days=9)
        result = self.cache.get_btc_closes(up_to_date=end, n_days=3)
        assert len(result) == 3

    def test_btc_closes_empty_returns_empty(self):
        result = self.cache.get_btc_closes(up_to_date=self.today, n_days=10)
        assert result == []

    # --- get_funding_rate_before ---

    def test_funding_rate_before_returns_most_recent(self):
        d = self.today
        # Three settlements: 00:00, 08:00, 16:00 UTC
        _insert_funding(self.cache, [
            (_settlement_ms(d, 0),  0.0001),
            (_settlement_ms(d, 8),  0.0002),
            (_settlement_ms(d, 16), 0.0003),
        ])
        before = datetime.datetime(
            d.year, d.month, d.day, 14, 0, 0, tzinfo=datetime.timezone.utc
        )
        rate = self.cache.get_funding_rate_before(before)
        assert rate == pytest.approx(0.0002)  # 08:00 settlement

    def test_funding_rate_before_excludes_exact_timestamp(self):
        d = self.today
        ts = _settlement_ms(d, 14)  # exactly 14:00
        _insert_funding(self.cache, [(ts, 0.0005)])
        before = datetime.datetime(
            d.year, d.month, d.day, 14, 0, 0, tzinfo=datetime.timezone.utc
        )
        # strictly less than — 14:00 settlement excluded
        rate = self.cache.get_funding_rate_before(before)
        assert rate is None

    def test_funding_rate_before_no_data_returns_none(self):
        before = datetime.datetime(
            self.today.year, self.today.month, self.today.day, 14, 0, 0,
            tzinfo=datetime.timezone.utc,
        )
        assert self.cache.get_funding_rate_before(before) is None

    def test_funding_rate_before_negative_rate(self):
        d = self.today
        _insert_funding(self.cache, [(_settlement_ms(d, 0), -0.0001)])
        before = datetime.datetime(
            d.year, d.month, d.day, 12, 0, 0, tzinfo=datetime.timezone.utc
        )
        assert self.cache.get_funding_rate_before(before) == pytest.approx(-0.0001)

    # --- get_btc_dominance ---

    def test_dominance_computed_correctly(self):
        btc_mcap   = 1_000_000.0
        total_mcap = 2_000_000.0
        _insert_closes(
            self.cache, self.today, [50_000.0],
            btc_market_caps=[btc_mcap],
        )
        _insert_total_market_caps(self.cache, self.today, [total_mcap])
        dom = self.cache.get_btc_dominance(self.today)
        assert dom == pytest.approx(50.0)

    def test_dominance_missing_total_returns_none(self):
        _insert_closes(self.cache, self.today, [50_000.0], btc_market_caps=[1e12])
        # No total_market_daily row
        assert self.cache.get_btc_dominance(self.today) is None

    def test_dominance_zero_total_returns_none(self):
        _insert_closes(self.cache, self.today, [50_000.0], btc_market_caps=[1e12])
        _insert_total_market_caps(self.cache, self.today, [0.0])
        assert self.cache.get_btc_dominance(self.today) is None

    def test_dominance_missing_btc_mcap_returns_none(self):
        _insert_closes(self.cache, self.today, [50_000.0])  # btc_market_cap_usd = NULL
        _insert_total_market_caps(self.cache, self.today, [2e12])
        assert self.cache.get_btc_dominance(self.today) is None

    def test_dominance_reasonable_range(self):
        """Typical 2024 BTC dominance is 50-55%."""
        _insert_closes(
            self.cache, self.today, [70_000.0],
            btc_market_caps=[1.38e12],
        )
        _insert_total_market_caps(self.cache, self.today, [2.5e12])
        dom = self.cache.get_btc_dominance(self.today)
        assert 40.0 <= dom <= 70.0


# ===========================================================================
# HistoricalScoreCache — populate() with mocked HTTP
# ===========================================================================

class TestCachePopulate:

    def setup_method(self):
        self.cache = _make_cache()
        self.start = datetime.date(2024, 1, 1)
        self.end   = datetime.date(2024, 1, 7)

    def teardown_method(self):
        self.cache.close()

    def _fg_payload(self, n_days: int = 10) -> Dict:
        base_ts = int(datetime.datetime(2024, 1, 7).timestamp())
        entries = [
            {"value": str(50 + i), "timestamp": str(base_ts - i * 86400)}
            for i in range(n_days)
        ]
        return {"data": entries}

    def _cg_price_payload(self) -> Dict:
        base_ms = int(datetime.datetime(2024, 1, 1).timestamp() * 1000)
        prices  = [[base_ms + i * 86_400_000, 40000.0 + i * 100] for i in range(10)]
        mcaps   = [[base_ms + i * 86_400_000, 7.8e11 + i * 1e9]  for i in range(10)]
        return {"prices": prices, "market_caps": mcaps, "total_volumes": []}

    def _cg_global_payload(self) -> Dict:
        base_ms = int(datetime.datetime(2024, 1, 1).timestamp() * 1000)
        mcaps   = [[base_ms + i * 86_400_000, 1.6e12 + i * 1e9] for i in range(10)]
        return {"market_cap_chart": {"market_cap": mcaps, "volume": []}}

    def _okx_page(self, n: int = 5, oldest_ts: int = 0) -> Dict:
        base_ms = int(datetime.datetime(2024, 1, 7).timestamp() * 1000)
        data = [
            {
                "fundingTime":  str(base_ms - i * 28_800_000),
                "fundingRate":  "0.0001",
                "realizedRate": "0.0001",
                "instId":       "BTC-USDT-SWAP",
            }
            for i in range(n)
        ]
        return {"code": "0", "data": data}

    def test_populate_fear_greed_stores_rows(self):
        payload = self._fg_payload(7)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = payload

        with patch("requests.get", return_value=mock_resp):
            self.cache._populate_fear_greed(self.start, self.end)

        count = self.cache._conn.execute(
            "SELECT COUNT(*) FROM fear_greed_daily"
        ).fetchone()[0]
        assert count == 7

    def test_populate_fear_greed_error_graceful(self):
        """HTTP error must not raise — just log and return."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("timeout")

        with patch("requests.get", return_value=mock_resp):
            self.cache._populate_fear_greed(self.start, self.end)  # must not raise

        count = self.cache._conn.execute(
            "SELECT COUNT(*) FROM fear_greed_daily"
        ).fetchone()[0]
        assert count == 0

    def test_populate_btc_prices_stores_rows(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = self._cg_price_payload()

        with patch("requests.get", return_value=mock_resp):
            self.cache._populate_btc_prices(self.start, self.end)

        count = self.cache._conn.execute(
            "SELECT COUNT(*) FROM btc_daily"
        ).fetchone()[0]
        assert count == 10  # 10 entries in mock payload

    def test_populate_btc_prices_error_graceful(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("network error")

        with patch("requests.get", return_value=mock_resp):
            self.cache._populate_btc_prices(self.start, self.end)

        count = self.cache._conn.execute("SELECT COUNT(*) FROM btc_daily").fetchone()[0]
        assert count == 0

    def test_populate_total_market_cap_stores_rows(self):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = self._cg_global_payload()

        with patch("requests.get", return_value=mock_resp):
            self.cache._populate_total_market_cap(self.start, self.end)

        count = self.cache._conn.execute(
            "SELECT COUNT(*) FROM total_market_daily"
        ).fetchone()[0]
        assert count == 10

    def test_populate_funding_rates_stores_rows(self):
        """Single-page OKX response should store all records."""
        page = self._okx_page(n=5)

        # Second call returns empty page to stop pagination
        empty_page = {"code": "0", "data": []}
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = [page, empty_page]

        with patch("requests.get", return_value=mock_resp):
            self.cache._populate_funding_rates(self.start, self.end)

        count = self.cache._conn.execute(
            "SELECT COUNT(*) FROM btc_funding_settlements"
        ).fetchone()[0]
        assert count == 5

    def test_populate_funding_rates_okx_error_graceful(self):
        error_resp = {"code": "1", "msg": "Server error", "data": []}
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = error_resp

        with patch("requests.get", return_value=mock_resp):
            self.cache._populate_funding_rates(self.start, self.end)

        count = self.cache._conn.execute(
            "SELECT COUNT(*) FROM btc_funding_settlements"
        ).fetchone()[0]
        assert count == 0

    def test_populate_skips_already_cached_fear_greed(self):
        """Second populate call with same range must not make another HTTP request."""
        payload = self._fg_payload(7)
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = payload

        with patch("requests.get", return_value=mock_resp) as mock_get:
            self.cache._populate_fear_greed(self.start, self.end)
            calls_first = mock_get.call_count

        # Insert covers the range — second call should skip
        with patch("requests.get", return_value=mock_resp) as mock_get2:
            self.cache._populate_fear_greed(self.start, self.end)
            assert mock_get2.call_count == 0, "Second populate should be a cache hit"


# ===========================================================================
# build_historical_score — computation correctness
# ===========================================================================

class TestBuildHistoricalScore:

    def setup_method(self):
        self.cache = _make_cache()
        self.date  = datetime.date(2024, 6, 15)
        # A date with plenty of history: fill 240 days ending D-1
        prev = self.date - datetime.timedelta(days=1)
        start = prev - datetime.timedelta(days=239)
        prices = _trending_prices(240, start=40_000.0, drift=0.0005)
        _insert_closes(self.cache, start, prices)

    def teardown_method(self):
        self.cache.close()

    def _add_fear_greed(self, value: int = 55) -> None:
        _insert_fear_greed(self.cache, self.date, [value])

    def _add_funding(self, rate: float = 0.0001) -> None:
        # Settlement at 08:00 UTC on D — before our 14:00 scan cutoff
        _insert_funding(self.cache, [(_settlement_ms(self.date, 8), rate)])

    # --- output structure ---

    def test_returns_all_required_keys(self):
        self._add_fear_greed()
        result = build_historical_score(self.date, self.cache)
        for key in ("score", "band", "signals", "timestamp", "date", "regime", "diagnostics"):
            assert key in result, f"Missing key: {key}"

    def test_date_field_matches_input(self):
        self._add_fear_greed()
        result = build_historical_score(self.date, self.cache)
        assert result["date"] == self.date

    def test_score_in_valid_range(self):
        self._add_fear_greed()
        self._add_funding()
        result = build_historical_score(self.date, self.cache)
        assert 0.0 <= result["score"] <= 100.0

    def test_regime_matches_score(self):
        self._add_fear_greed()
        result = build_historical_score(self.date, self.cache)
        score = result["score"]
        regime = result["regime"]
        if score < 25:
            assert regime == "extreme_fear"
        elif score < 40:
            assert regime == "cautious"
        elif score < 60:
            assert regime == "neutral"
        elif score < 75:
            assert regime == "bullish"
        else:
            assert regime == "extreme_greed"

    # --- lookahead prevention ---

    def test_price_data_as_of_prev_day(self):
        """diagnostics.price_as_of must be D-1, not D."""
        self._add_fear_greed()
        result = build_historical_score(self.date, self.cache)
        expected = (self.date - datetime.timedelta(days=1)).isoformat()
        assert result["diagnostics"]["price_as_of"] == expected

    def test_d_close_not_used(self):
        """Inserting an extreme D close must NOT change the score."""
        self._add_fear_greed()
        result_before = build_historical_score(self.date, self.cache)

        # Insert D's close at a wildly different price
        _insert_closes(self.cache, self.date, [1.0])  # price crash
        result_after = build_historical_score(self.date, self.cache)

        assert result_before["score"] == result_after["score"]

    def test_iv_rv_spread_is_rv_times_premium(self):
        """iv_rv_spread signal should equal RV × iv_premium."""
        self._add_fear_greed()
        result = build_historical_score(self.date, self.cache, iv_premium=0.10)

        diag = result["diagnostics"]
        rv   = diag["rv_30d"]
        assert rv is not None

        # The spread actually used = rv * 0.10
        expected_spread = rv * 0.10

        # Verify it appears in signals
        if "iv_rv_spread" in result["signals"]:
            assert result["signals"]["iv_rv_spread"]["raw"] == pytest.approx(
                expected_spread, rel=1e-4
            )

    def test_custom_iv_premium_changes_spread(self):
        """Higher iv_premium → larger raw spread → lower (more-fear) component."""
        # Need volatile prices so RV > 0; create a separate cache with alternating prices
        vol_cache = _make_cache()
        prev = self.date - datetime.timedelta(days=1)
        start = prev - datetime.timedelta(days=239)
        # Alternating ±2% to produce meaningful realized vol
        prices = []
        p = 40_000.0
        for i in range(240):
            p = p * 1.02 if i % 2 == 0 else p * 0.98
            prices.append(p)
        _insert_closes(vol_cache, start, prices)
        _insert_fear_greed(vol_cache, self.date, [50])

        r1 = build_historical_score(self.date, vol_cache, iv_premium=0.05)
        r2 = build_historical_score(self.date, vol_cache, iv_premium=0.50)

        assert "iv_rv_spread" in r1["signals"], "iv_rv_spread signal missing with volatile prices"
        assert "iv_rv_spread" in r2["signals"]
        # Higher premium → larger raw spread → inverted sigmoid gives lower component
        assert r1["signals"]["iv_rv_spread"]["raw"] < r2["signals"]["iv_rv_spread"]["raw"]
        assert r1["signals"]["iv_rv_spread"]["component"] > \
               r2["signals"]["iv_rv_spread"]["component"]
        vol_cache.close()

    # --- funding rate cutoff ---

    def test_funding_settlement_before_14_utc_used(self):
        """Settlements at 00:00 and 08:00 should be used (before 14:00 scan)."""
        self._add_fear_greed()
        _insert_funding(self.cache, [
            (_settlement_ms(self.date, 0),  0.0001),
            (_settlement_ms(self.date, 8),  0.0002),   # most recent before 14:00
            (_settlement_ms(self.date, 16), 0.9999),   # AFTER scan — must not be used
        ])
        result = build_historical_score(self.date, self.cache)
        assert result["signals"]["funding_rate"]["raw"] == pytest.approx(0.0002)

    def test_funding_settlement_after_14_utc_excluded(self):
        """Only the 16:00 settlement exists; it's after 14:00 → funding absent."""
        self._add_fear_greed()
        _insert_funding(self.cache, [(_settlement_ms(self.date, 16), 0.0005)])
        result = build_historical_score(self.date, self.cache)
        assert "funding_rate" not in result["signals"]

    # --- graceful degradation ---

    def test_only_prices_produces_valid_score(self):
        """No fear_greed, no funding → still scores using MA200 + iv_rv_spread."""
        result = build_historical_score(self.date, self.cache)
        assert 0.0 <= result["score"] <= 100.0
        assert "ma200_position" in result["signals"]
        assert "iv_rv_spread" in result["signals"]

    def test_no_signals_raises_value_error(self):
        """Empty cache (no prices, no fear_greed, no funding) → ValueError."""
        empty_cache = _make_cache()
        with pytest.raises(ValueError):
            build_historical_score(self.date, empty_cache)
        empty_cache.close()

    def test_insufficient_prices_for_ma200_skips_ma200(self):
        """Only 50 days of prices → MA200 signal skipped (not enough history)."""
        short_cache = _make_cache()
        start = self.date - datetime.timedelta(days=51)
        _insert_closes(short_cache, start, _flat_prices(50))
        _insert_fear_greed(short_cache, self.date, [50])
        result = build_historical_score(self.date, short_cache)
        assert "ma200_position" not in result["signals"]
        short_cache.close()

    def test_insufficient_prices_for_rv_skips_iv_rv_spread(self):
        """Only 20 days of prices with rv_window=30 → iv_rv_spread skipped."""
        short_cache = _make_cache()
        start = self.date - datetime.timedelta(days=21)
        _insert_closes(short_cache, start, _flat_prices(20))
        _insert_fear_greed(short_cache, self.date, [50])
        result = build_historical_score(self.date, short_cache, rv_window=30)
        assert "iv_rv_spread" not in result["signals"]
        short_cache.close()

    def test_btc_dominance_signal_present_when_both_mcaps_available(self):
        # Insert a btc_daily row for D itself (dominance is queried for D, not D-1)
        self.cache._conn.execute(
            "INSERT OR REPLACE INTO btc_daily (date, close, btc_market_cap_usd) "
            "VALUES (?, ?, ?)",
            (self.date.isoformat(), 70_000.0, 1.0e12),
        )
        self.cache._conn.commit()
        # Insert matching total market cap for D
        _insert_total_market_caps(self.cache, self.date, [2.0e12])

        result = build_historical_score(self.date, self.cache)
        assert "btc_dominance" in result["signals"]
        assert result["signals"]["btc_dominance"]["raw"] == pytest.approx(50.0)

    # --- diagnostics ---

    def test_diagnostics_structure(self):
        self._add_fear_greed()
        result = build_historical_score(self.date, self.cache)
        diag = result["diagnostics"]
        assert "rv_30d" in diag
        assert "iv_premium_used" in diag
        assert "closes_count" in diag
        assert "price_as_of" in diag

    def test_diagnostics_closes_count_is_correct(self):
        """closes_count should be ≤ PRICE_LOOKBACK_DAYS and match what was fetched."""
        self._add_fear_greed()
        result = build_historical_score(self.date, self.cache)
        # We inserted 240 closes ending on D-1; count should be ≤ 235 (PRICE_LOOKBACK_DAYS)
        assert result["diagnostics"]["closes_count"] <= _PRICE_LOOKBACK_DAYS

    def test_flat_prices_give_zero_rv(self):
        """Flat prices → RV=0 → iv_rv_spread=0."""
        flat_cache = _make_cache()
        start = self.date - datetime.timedelta(days=240)
        _insert_closes(flat_cache, start, _flat_prices(240))
        _insert_fear_greed(flat_cache, self.date, [50])
        result = build_historical_score(self.date, flat_cache)
        assert result["diagnostics"]["rv_30d"] == pytest.approx(0.0, abs=1e-10)
        flat_cache.close()


# ===========================================================================
# build_score_series
# ===========================================================================

class TestBuildScoreSeries:

    def setup_method(self):
        self.cache = _make_cache()
        # 300 closes ending on 2024-06-30
        end    = datetime.date(2024, 6, 30)
        start  = end - datetime.timedelta(days=299)
        prices = _trending_prices(300, 40_000.0, drift=0.0003)
        _insert_closes(self.cache, start, prices)

        # Fear & Greed for Jun 1-30
        fg_start = datetime.date(2024, 6, 1)
        _insert_fear_greed(self.cache, fg_start, [50] * 30)

    def teardown_method(self):
        self.cache.close()

    def test_returns_dict_keyed_by_date(self):
        result = build_score_series(
            datetime.date(2024, 6, 10),
            datetime.date(2024, 6, 12),
            self.cache,
        )
        for k in result:
            assert isinstance(k, datetime.date)

    def test_all_dates_in_range_present(self):
        start = datetime.date(2024, 6, 10)
        end   = datetime.date(2024, 6, 15)
        result = build_score_series(start, end, self.cache)
        for i in range((end - start).days + 1):
            d = start + datetime.timedelta(days=i)
            assert d in result, f"Missing date {d}"

    def test_start_after_end_returns_empty(self):
        result = build_score_series(
            datetime.date(2024, 6, 30),
            datetime.date(2024, 6, 1),
            self.cache,
        )
        assert result == {}

    def test_same_start_and_end(self):
        result = build_score_series(
            datetime.date(2024, 6, 15),
            datetime.date(2024, 6, 15),
            self.cache,
        )
        assert len(result) == 1

    def test_dates_with_no_signals_skipped(self):
        """Dates for which the cache has no data at all should be omitted."""
        # Use a completely empty cache — no prices, no fear_greed, no funding
        empty = _make_cache()
        target = datetime.date(2024, 6, 15)
        result = build_score_series(target, target, empty)
        assert target not in result
        empty.close()

    def test_scores_in_valid_range(self):
        result = build_score_series(
            datetime.date(2024, 6, 10),
            datetime.date(2024, 6, 20),
            self.cache,
        )
        for d, r in result.items():
            assert 0.0 <= r["score"] <= 100.0, f"Out-of-range score on {d}: {r['score']}"

    def test_each_result_has_date_field_matching_key(self):
        result = build_score_series(
            datetime.date(2024, 6, 10),
            datetime.date(2024, 6, 15),
            self.cache,
        )
        for d, r in result.items():
            assert r["date"] == d

    def test_regime_field_present_in_every_result(self):
        result = build_score_series(
            datetime.date(2024, 6, 10),
            datetime.date(2024, 6, 14),
            self.cache,
        )
        for d, r in result.items():
            assert "regime" in r

    def test_consistent_scores_for_flat_prices(self):
        """Flat prices + constant fear_greed → all scores identical."""
        flat_cache = _make_cache()
        end   = datetime.date(2024, 6, 30)
        start = end - datetime.timedelta(days=299)
        _insert_closes(flat_cache, start, _flat_prices(300))
        fg_start = datetime.date(2024, 6, 1)
        _insert_fear_greed(flat_cache, fg_start, [50] * 30)

        result = build_score_series(
            datetime.date(2024, 6, 10),
            datetime.date(2024, 6, 20),
            flat_cache,
        )
        scores = [r["score"] for r in result.values()]
        assert len(set(scores)) == 1, "Flat prices + constant FG should produce identical scores"
        flat_cache.close()

    def test_custom_rv_window_propagated(self):
        """rv_window parameter must reach the diagnostics."""
        result = build_score_series(
            datetime.date(2024, 6, 15),
            datetime.date(2024, 6, 15),
            self.cache,
            rv_window=7,
        )
        d = datetime.date(2024, 6, 15)
        if d in result:
            # 7-day window should match compute_realized_vol output
            prev = d - datetime.timedelta(days=1)
            closes = self.cache.get_btc_closes(prev, 235)
            expected_rv = compute_realized_vol(closes, window=7)
            assert result[d]["diagnostics"]["rv_30d"] == pytest.approx(
                expected_rv, rel=1e-6
            )

    def test_custom_iv_premium_propagated(self):
        result = build_score_series(
            datetime.date(2024, 6, 15),
            datetime.date(2024, 6, 15),
            self.cache,
            iv_premium=0.25,
        )
        d = datetime.date(2024, 6, 15)
        if d in result:
            assert result[d]["diagnostics"]["iv_premium_used"] == 0.25


# ===========================================================================
# _score_to_regime
# ===========================================================================

class TestScoreToRegime:

    def test_below_25_extreme_fear(self):
        assert _score_to_regime(0.0)  == "extreme_fear"
        assert _score_to_regime(24.9) == "extreme_fear"

    def test_25_to_40_cautious(self):
        assert _score_to_regime(25.0) == "cautious"
        assert _score_to_regime(39.9) == "cautious"

    def test_40_to_60_neutral(self):
        assert _score_to_regime(40.0) == "neutral"
        assert _score_to_regime(59.9) == "neutral"

    def test_60_to_75_bullish(self):
        assert _score_to_regime(60.0) == "bullish"
        assert _score_to_regime(74.9) == "bullish"

    def test_75_plus_extreme_greed(self):
        assert _score_to_regime(75.0)  == "extreme_greed"
        assert _score_to_regime(100.0) == "extreme_greed"
