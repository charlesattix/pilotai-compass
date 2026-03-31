"""
Shared test fixtures and utilities for COMPASS module tests.

Provides reusable helpers for:
  - tests/test_regime_classifier.py
  - tests/test_event_gate.py
  - tests/test_sizing.py

All data is deterministic (no randomness) so tests are fully reproducible.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Price / VIX data builders
# ---------------------------------------------------------------------------

def mock_spy_prices(
    start_date: str = "2024-01-02",
    days: int = 100,
    trend: float = 0.0,
    base: float = 450.0,
    volatility: float = 0.0,
) -> pd.Series:
    """Build a deterministic SPY close price series.

    Args:
        start_date: First business day (string, e.g. "2024-01-02").
        days: Number of business days to generate.
        trend: Annualized return in percent (e.g. +20.0 for 20% annual uptrend,
               -30.0 for 30% annual decline). Applied as a smooth daily drift.
        base: Starting price level.
        volatility: Daily noise amplitude in dollars. 0 = perfectly smooth.
                    Useful for testing trend detection thresholds.

    Returns:
        pd.Series with DatetimeIndex (business days), name="Close".
    """
    idx = pd.bdate_range(start=start_date, periods=days)
    daily_return = (1 + trend / 100) ** (1 / 252) - 1

    prices = np.empty(days)
    prices[0] = base
    for i in range(1, days):
        prices[i] = prices[i - 1] * (1 + daily_return)
        if volatility > 0:
            # Deterministic "noise" based on day index (no random seed needed)
            prices[i] += volatility * np.sin(i * 0.7)

    return pd.Series(prices, index=idx, name="Close")


def mock_vix_series(
    start_date: str = "2024-01-02",
    days: int = 100,
    base_level: float = 20.0,
    levels: Optional[List[float]] = None,
) -> pd.Series:
    """Build a deterministic VIX close series.

    Args:
        start_date: First business day.
        days: Number of business days.
        base_level: Constant VIX level (used when `levels` is None).
        levels: If provided, must be length `days`. Overrides `base_level`
                for scenario-based testing (e.g. VIX spike sequences).

    Returns:
        pd.Series with DatetimeIndex, name="VIX_Close".
    """
    idx = pd.bdate_range(start=start_date, periods=days)
    if levels is not None:
        assert len(levels) == days, f"levels length {len(levels)} != days {days}"
        values = levels
    else:
        values = [base_level] * days

    return pd.Series(values, index=idx, name="VIX_Close", dtype=float)


def mock_spy_dataframe(prices: pd.Series) -> pd.DataFrame:
    """Wrap a price Series into a minimal SPY OHLCV DataFrame.

    Required by RegimeClassifier.classify_series() which reads df["Close"].
    """
    return pd.DataFrame({"Close": prices}, index=prices.index)


# ---------------------------------------------------------------------------
# Regime scenario presets
# ---------------------------------------------------------------------------
# Each preset defines VIX + trend parameters that should produce the
# expected regime, along with enough history for the default 50-day
# trend window to work (100 days).

class RegimeScenario:
    """Bundles VIX, trend, and expected regime for parametrized tests."""

    def __init__(
        self,
        name: str,
        vix: float,
        trend: float,
        expected_regime: str,
        days: int = 100,
        base: float = 450.0,
        sharp_decline: bool = False,
    ):
        self.name = name
        self.vix = vix
        self.trend = trend
        self.expected_regime = expected_regime
        self.days = days
        self.base = base
        self.sharp_decline = sharp_decline

    def build_prices(self, start: str = "2023-06-01") -> pd.Series:
        """Build spy prices for this scenario.

        If sharp_decline is True, injects a >5% drop in the last 10 days
        (needed for CRASH classification).
        """
        if self.sharp_decline:
            # Build 90 days of normal prices, then 10 days of sharp decline
            normal = mock_spy_prices(start, days=self.days - 10, trend=self.trend, base=self.base)
            last_price = float(normal.iloc[-1])
            # Drop 7% over 10 days (> 5% threshold)
            decline_idx = pd.bdate_range(start=normal.index[-1] + timedelta(days=1), periods=10)
            decline_prices = np.linspace(last_price, last_price * 0.93, 10)
            decline = pd.Series(decline_prices, index=decline_idx, name="Close")
            return pd.concat([normal, decline])
        return mock_spy_prices(start, days=self.days, trend=self.trend, base=self.base)


# Pre-built scenarios matching the 5 regimes
REGIME_SCENARIOS: Dict[str, RegimeScenario] = {
    "bull": RegimeScenario(
        name="bull",
        vix=16.0,
        trend=25.0,  # strong uptrend
        expected_regime="bull",
    ),
    "bear": RegimeScenario(
        name="bear",
        vix=27.0,
        trend=-25.0,  # strong downtrend + elevated VIX
        expected_regime="bear",
    ),
    "high_vol": RegimeScenario(
        name="high_vol",
        vix=35.0,
        trend=5.0,  # any trend, VIX > 30
        expected_regime="high_vol",
    ),
    "low_vol": RegimeScenario(
        name="low_vol",
        vix=12.0,
        trend=0.0,  # flat market, very low VIX
        expected_regime="low_vol",
    ),
    "crash": RegimeScenario(
        name="crash",
        vix=45.0,
        trend=-40.0,  # extreme VIX + sharp decline
        expected_regime="crash",
        sharp_decline=True,
    ),
}


# ---------------------------------------------------------------------------
# Known event dates for deterministic event gate testing
# ---------------------------------------------------------------------------
# All dates below are real dates from the hardcoded FOMC calendar and
# algorithmic CPI/NFP computation. They allow tests to use fixed as_of
# dates without depending on date.today().

KNOWN_FOMC_DATES = {
    # 2026 FOMC decision dates (from shared/macro_event_gate.py FOMC_DATES_2026)
    "2026_jan": date(2026, 1, 29),
    "2026_mar": date(2026, 3, 19),
    "2026_may": date(2026, 5, 7),
    "2026_jun": date(2026, 6, 18),
    "2026_jul": date(2026, 7, 30),
    "2026_sep": date(2026, 9, 17),
    "2026_nov": date(2026, 11, 5),
    "2026_dec": date(2026, 12, 17),
    # Emergency dates (2020)
    "2020_emergency_1": date(2020, 3, 3),
    "2020_emergency_2": date(2020, 3, 15),
}

# CPI release dates (computed: 12th of M+1, advanced past weekends)
# CPI for Jan 2026 → Feb 12, 2026 (Thursday = weekday, no shift)
# CPI for Nov 2025 → Dec 12, 2025 (Friday = weekday, no shift)
# CPI for May 2026 → Jun 12, 2026 (Friday = weekday, no shift)
KNOWN_CPI_DATES = {
    "jan_2026": date(2026, 2, 12),   # CPI for Jan released Feb 12
    "nov_2025": date(2025, 12, 12),  # CPI for Nov released Dec 12
    "may_2026": date(2026, 6, 12),   # CPI for May released Jun 12
}

# NFP release dates (first Friday of M+1)
# NFP for Jan 2026 → first Friday of Feb 2026 = Feb 6 (Friday)
# NFP for Dec 2025 → first Friday of Jan 2026 = Jan 2 (Friday)
# NFP for Jun 2026 → first Friday of Jul 2026 = Jul 3 (Friday)
KNOWN_NFP_DATES = {
    "jan_2026": date(2026, 2, 6),   # NFP for Jan released first Fri of Feb
    "dec_2025": date(2026, 1, 2),   # NFP for Dec released first Fri of Jan 2026
    "jun_2026": date(2026, 7, 3),   # NFP for Jun released first Fri of Jul
}

# Scaling tables (mirrored from source for test assertions)
FOMC_SCALING = {5: 1.00, 4: 0.90, 3: 0.80, 2: 0.70, 1: 0.60, 0: 0.50}
CPI_SCALING = {2: 1.00, 1: 0.75, 0: 0.65}
NFP_SCALING = {2: 1.00, 1: 0.80, 0: 0.75}

# Post-event buffer scaling factors
POST_FOMC_SCALING = 0.70
POST_CPI_SCALING = 0.80
POST_NFP_SCALING = 0.80


# ---------------------------------------------------------------------------
# Sizing test constants
# ---------------------------------------------------------------------------

ACCOUNT_100K = 100_000.0
ACCOUNT_10K = 10_000.0

# Standard spread: $5-wide, $0.65 credit
# max_loss_per_contract = (5.0 - 0.65) * 100 = $435
SPREAD_WIDTH_5 = 5.0
CREDIT_065 = 0.65
MAX_LOSS_PER_CONTRACT_5_065 = (SPREAD_WIDTH_5 - CREDIT_065) * 100  # $435.0


# ---------------------------------------------------------------------------
# Mock macro DB
# ---------------------------------------------------------------------------

def mock_macro_db(tmp_path) -> str:
    """Create an in-memory-equivalent SQLite DB with the macro_state schema.

    Uses tmp_path (pytest fixture) to create an isolated DB file.
    Returns the DB path string for passing to functions that accept db_path.

    The DB is initialized with the full schema (snapshots, sector_rs,
    macro_score, macro_events, macro_state tables) matching
    shared/macro_state_db.init_db().
    """
    import sqlite3

    db_path = str(tmp_path / "test_macro_state.db")
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            date            TEXT PRIMARY KEY,
            spy_close       REAL,
            top_sector_3m   TEXT,
            top_sector_12m  TEXT,
            leading_sectors TEXT,
            rrg_quadrants   TEXT,
            raw_json        TEXT
        );

        CREATE TABLE IF NOT EXISTS sector_rs (
            date         TEXT NOT NULL,
            ticker       TEXT NOT NULL,
            name         TEXT,
            category     TEXT,
            close        REAL,
            rs_3m        REAL,
            rs_12m       REAL,
            rrg_quadrant TEXT,
            PRIMARY KEY (date, ticker)
        );

        CREATE TABLE IF NOT EXISTS macro_score (
            date              TEXT PRIMARY KEY,
            overall           REAL,
            growth            REAL,
            inflation         REAL,
            fed_policy        REAL,
            risk_appetite     REAL,
            overall_velocity  REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS macro_events (
            event_date     TEXT NOT NULL,
            event_type     TEXT NOT NULL,
            description    TEXT,
            days_out       INTEGER,
            scaling_factor REAL,
            is_emergency   INTEGER DEFAULT 0,
            PRIMARY KEY (event_date, event_type)
        );

        CREATE TABLE IF NOT EXISTS macro_state (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()
    return db_path


def seed_macro_db(db_path: str, macro_score: float = 55.0, regime: str = "NEUTRAL") -> None:
    """Insert minimal seed data into a test macro DB.

    Useful for tests that need a non-empty DB without constructing
    full snapshots.
    """
    import sqlite3
    from datetime import date as _date

    today = _date.today().strftime("%Y-%m-%d")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT OR REPLACE INTO macro_score (date, overall, growth, inflation, fed_policy, risk_appetite) "
        "VALUES (?, ?, 50.0, 50.0, 50.0, 50.0)",
        (today, macro_score),
    )
    conn.execute(
        "INSERT OR REPLACE INTO macro_state (key, value) VALUES ('regime', ?)",
        (regime,),
    )
    conn.commit()
    conn.close()
