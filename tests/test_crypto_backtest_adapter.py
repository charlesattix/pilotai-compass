"""
Unit tests for CryptoDataAdapter.

Uses an in-memory SQLite DB seeded with fixture data so tests are fully
offline and deterministic.  No real data/DB files are required.
"""

import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from backtest.crypto_data_adapter import CryptoDataAdapter, CryptoDataError

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE crypto_option_contracts (
    ticker TEXT,
    expiration TEXT,
    strike REAL,
    option_type TEXT,
    contract_symbol TEXT,
    as_of_date TEXT
);
CREATE TABLE crypto_option_daily (
    contract_symbol TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    open_interest INTEGER,
    bid REAL,
    ask REAL,
    mid REAL,
    iv REAL,
    delta REAL,
    gamma REAL,
    theta REAL,
    vega REAL,
    underlying_price REAL
);
CREATE TABLE crypto_underlying_daily (
    ticker TEXT,
    date TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER
);
"""

_CONTRACTS = [
    # (ticker, expiration, strike, option_type, contract_symbol, as_of_date)
    ("IBIT", "2024-11-15", 40.0, "P", "O:IBIT241115P00040000", "2024-10-01"),
    ("IBIT", "2024-11-15", 45.0, "P", "O:IBIT241115P00045000", "2024-10-01"),
    ("IBIT", "2024-11-15", 50.0, "P", "O:IBIT241115P00050000", "2024-10-01"),
    ("IBIT", "2024-11-15", 55.0, "C", "O:IBIT241115C00055000", "2024-10-01"),
    ("IBIT", "2024-11-15", 60.0, "C", "O:IBIT241115C00060000", "2024-10-01"),
]

_DAILY = [
    # (contract_symbol, date, open, high, low, close, volume, oi, bid, ask, mid, iv, ...)
    ("O:IBIT241115P00045000", "2024-10-15", 1.0, 1.2, 0.9, 1.10, 500, 1000, 1.05, 1.15, 1.10, 0.55, None, None, None, None, 48.0),
    ("O:IBIT241115P00040000", "2024-10-15", 0.3, 0.4, 0.25, 0.35, 200, 400, 0.32, 0.38, 0.35, 0.52, None, None, None, None, 48.0),
    ("O:IBIT241115C00055000", "2024-10-15", 0.8, 0.9, 0.7, 0.85, 300, 600, 0.82, 0.88, 0.85, 0.58, None, None, None, None, 48.0),
    ("O:IBIT241115C00060000", "2024-10-15", 0.2, 0.3, 0.18, 0.25, 100, 200, 0.22, 0.28, 0.25, 0.54, None, None, None, None, 48.0),
    # IV series for get_iv_series test
    ("O:IBIT241115P00045000", "2024-10-10", 1.2, 1.3, 1.0, 1.15, 450, 900, 1.10, 1.20, 1.15, 0.60, None, None, None, None, 47.0),
    ("O:IBIT241115P00045000", "2024-10-11", 1.1, 1.2, 0.95, 1.05, 480, 950, 1.02, 1.08, 1.05, 0.58, None, None, None, None, 47.5),
]

_UNDERLYING = [
    # (ticker, date, open, high, low, close, volume)
    ("IBIT", "2024-10-01", 44.0, 45.5, 43.8, 45.2, 10000000),
    ("IBIT", "2024-10-02", 45.2, 46.0, 44.9, 45.8, 9500000),
    ("IBIT", "2024-10-15", 47.5, 48.2, 47.0, 47.8, 11000000),
    ("IBIT", "2024-10-16", 47.8, 48.5, 47.3, 48.1, 10500000),
]


def _make_db(path: str) -> None:
    """Create a fixture DB at path with test data."""
    conn = sqlite3.connect(path)
    conn.executescript(_DDL)
    conn.executemany(
        "INSERT INTO crypto_option_contracts VALUES (?,?,?,?,?,?)", _CONTRACTS
    )
    conn.executemany(
        "INSERT INTO crypto_option_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        _DAILY,
    )
    conn.executemany(
        "INSERT INTO crypto_underlying_daily VALUES (?,?,?,?,?,?,?)", _UNDERLYING
    )
    conn.commit()
    conn.close()


@pytest.fixture
def adapter(tmp_path):
    db_path = str(tmp_path / "crypto_options_cache.db")
    _make_db(db_path)
    adp = CryptoDataAdapter(db_path)
    yield adp
    adp.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_occ_symbol():
    sym = CryptoDataAdapter.build_occ_symbol("IBIT", "2024-11-15", 45.0, "P")
    assert sym == "O:IBIT241115P00045000"


def test_build_occ_symbol_call():
    sym = CryptoDataAdapter.build_occ_symbol("IBIT", "2024-11-15", 55.0, "C")
    assert sym == "O:IBIT241115C00055000"


def test_missing_db_raises():
    with pytest.raises(CryptoDataError):
        CryptoDataAdapter("/nonexistent/path/crypto_options_cache.db")


def test_get_contract_price(adapter):
    price = adapter.get_contract_price("O:IBIT241115P00045000", "2024-10-15")
    assert price == pytest.approx(1.10)


def test_get_contract_price_missing(adapter):
    price = adapter.get_contract_price("O:IBIT241115P00045000", "2024-12-01")
    assert price is None


def test_get_available_strikes_put(adapter):
    strikes = adapter.get_available_strikes("IBIT", "2024-11-15", "2024-10-15", "P")
    assert strikes == [40.0, 45.0, 50.0]


def test_get_available_strikes_call(adapter):
    strikes = adapter.get_available_strikes("IBIT", "2024-11-15", "2024-10-15", "C")
    assert strikes == [55.0, 60.0]


def test_get_available_strikes_no_lookahead(adapter):
    # as_of_date before any contracts seeded (2024-10-01) → no results
    strikes = adapter.get_available_strikes("IBIT", "2024-11-15", "2024-09-30", "P")
    assert strikes == []


def test_get_spread_prices(adapter):
    result = adapter.get_spread_prices(
        "IBIT", "2024-11-15", 45.0, 40.0, "P", "2024-10-15"
    )
    assert result is not None
    assert result["spread_value"] == pytest.approx(1.10 - 0.35)
    assert result["short_price"] == pytest.approx(1.10)
    assert result["long_price"] == pytest.approx(0.35)
    assert result["slippage"] == 0.0


def test_get_spread_prices_missing_leg(adapter):
    # long_strike 99.0 does not exist
    result = adapter.get_spread_prices(
        "IBIT", "2024-11-15", 45.0, 99.0, "P", "2024-10-15"
    )
    assert result is None


def test_get_intraday_falls_back_to_daily(adapter):
    """get_intraday_spread_prices must delegate to daily when no intraday bars exist."""
    intraday = adapter.get_intraday_spread_prices(
        "IBIT", "2024-11-15", 45.0, 40.0, "P", "2024-10-15", 9, 30
    )
    daily = adapter.get_spread_prices(
        "IBIT", "2024-11-15", 45.0, 40.0, "P", "2024-10-15"
    )
    assert intraday == daily


def test_get_prev_daily_volume(adapter):
    vol = adapter.get_prev_daily_volume("O:IBIT241115P00045000", "2024-10-16")
    assert vol == 500


def test_get_prev_daily_oi(adapter):
    oi = adapter.get_prev_daily_oi("O:IBIT241115P00045000", "2024-10-16")
    assert oi == 1000


def test_get_underlying_prices_returns_ohlcv(adapter):
    df = adapter.get_underlying_prices(
        "IBIT",
        datetime(2024, 10, 1),
        datetime(2024, 10, 16),
    )
    assert not df.empty
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns)
    assert len(df) == 4  # 4 rows seeded
    # Index should be DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)
    # Spot check a value
    row = df.loc[pd.Timestamp("2024-10-15")]
    assert row["Close"] == pytest.approx(47.8)


def test_get_underlying_prices_date_filter(adapter):
    df = adapter.get_underlying_prices(
        "IBIT",
        datetime(2024, 10, 15),
        datetime(2024, 10, 16),
    )
    assert len(df) == 2


def test_get_iv_series(adapter):
    ivs = adapter.get_iv_series("IBIT", "2024-10-16", window=90)
    # 3 dates with iv data: 2024-10-10, 2024-10-11, 2024-10-15
    assert len(ivs) == 3
    # Returned oldest-first
    assert ivs[0] == pytest.approx(0.60)
    # 2024-10-15 has 4 contracts: avg(0.55, 0.52, 0.58, 0.54) = 0.5475
    assert ivs[-1] == pytest.approx(0.5475)


def test_get_iv_series_empty_before_date(adapter):
    ivs = adapter.get_iv_series("IBIT", "2024-10-09", window=90)
    assert ivs == []
