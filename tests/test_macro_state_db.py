"""
Unit tests for shared/macro_state_db.py
"""


import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compass.macro_db import (
    backfill_macro_score_velocities,
    get_current_macro_score,
    get_db,
    get_eligible_underlyings,
    get_event_scaling_factor,
    get_latest_snapshot_date,
    get_sector_rankings,
    get_snapshot_count,
    get_state,
    init_db,
    save_snapshot,
    set_state,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixture: isolated in-memory DB for each test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path):
    """Return path string for a fresh, initialised macro_state.db."""
    db = str(tmp_path / "macro_state.db")
    init_db(db)
    return db


def _make_snap(snap_date: str, overall: float = 60.0, risk_appetite: float = 70.0) -> dict:
    """Build a minimal snapshot dict accepted by save_snapshot()."""
    return {
        "date": snap_date,
        "spy_close": 450.0,
        "top_sector_3m": "XLK",
        "top_sector_12m": "XLK",
        "leading_sectors": ["XLK", "XLF"],
        "lagging_sectors": ["XLE"],
        "macro_score": {
            "overall": overall,
            "growth": 55.0,
            "inflation": 60.0,
            "fed_policy": 50.0,
            "risk_appetite": risk_appetite,
            "indicators": {
                "vix": 18.5,
                "t10y2y": 0.25,
                "hy_oas_pct": 3.5,
                "cpi_yoy_pct": 3.1,
                "core_cpi_yoy_pct": 3.4,
                "fedfunds": 5.25,
                "cfnai_3m": -0.1,
                "payrolls_3m_avg_k": 180.0,
                "breakeven_5y": 2.3,
            },
        },
        "sector_rankings": [
            {
                "ticker": "XLK",
                "name": "Technology",
                "category": "sector",
                "close": 195.0,
                "rs_3m": 4.5,
                "rs_12m": 12.0,
                "rs_ratio": 102.1,
                "rs_momentum": 101.0,
                "rrg_quadrant": "Leading",
                "rank_3m": 1,
                "rank_12m": 1,
            },
            {
                "ticker": "XLF",
                "name": "Financials",
                "category": "sector",
                "close": 38.0,
                "rs_3m": 2.1,
                "rs_12m": 6.5,
                "rs_ratio": 100.5,
                "rs_momentum": 100.2,
                "rrg_quadrant": "Leading",
                "rank_3m": 2,
                "rank_12m": 2,
            },
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Schema / init
# ─────────────────────────────────────────────────────────────────────────────

def test_init_db_creates_tables(tmp_db):
    conn = get_db(tmp_db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    conn.close()
    assert {"snapshots", "sector_rs", "macro_score", "macro_events", "macro_state"} <= tables


def test_init_db_idempotent(tmp_db):
    """Calling init_db twice should not raise."""
    init_db(tmp_db)
    init_db(tmp_db)


# ─────────────────────────────────────────────────────────────────────────────
# State key-value store
# ─────────────────────────────────────────────────────────────────────────────

def test_set_and_get_state(tmp_db):
    set_state("test_key", "hello", db_path=tmp_db)
    assert get_state("test_key", db_path=tmp_db) == "hello"


def test_get_state_missing_returns_default(tmp_db):
    assert get_state("nonexistent", default="fallback", db_path=tmp_db) == "fallback"


def test_set_state_overwrites(tmp_db):
    set_state("k", "v1", db_path=tmp_db)
    set_state("k", "v2", db_path=tmp_db)
    assert get_state("k", db_path=tmp_db) == "v2"


# ─────────────────────────────────────────────────────────────────────────────
# save_snapshot + retrieval
# ─────────────────────────────────────────────────────────────────────────────

def test_save_snapshot_inserts_rows(tmp_db):
    snap = _make_snap("2024-01-05")
    save_snapshot(snap, db_path=tmp_db)
    assert get_snapshot_count(tmp_db) == 1
    assert get_latest_snapshot_date(tmp_db) == "2024-01-05"


def test_save_snapshot_idempotent(tmp_db):
    snap = _make_snap("2024-01-05")
    save_snapshot(snap, db_path=tmp_db)
    save_snapshot(snap, db_path=tmp_db)
    assert get_snapshot_count(tmp_db) == 1


def test_save_snapshot_populates_macro_score(tmp_db):
    snap = _make_snap("2024-01-05", overall=72.5, risk_appetite=80.0)
    save_snapshot(snap, db_path=tmp_db)

    conn = get_db(tmp_db)
    row = conn.execute("SELECT * FROM macro_score WHERE date = '2024-01-05'").fetchone()
    conn.close()

    assert row is not None
    assert abs(row["overall"] - 72.5) < 0.01
    assert abs(row["overall_v2"] - 72.5) < 0.01
    assert row["updated_at"] is not None


def test_save_snapshot_computes_velocity_on_second_row(tmp_db):
    save_snapshot(_make_snap("2024-01-05", overall=60.0, risk_appetite=70.0), db_path=tmp_db)
    save_snapshot(_make_snap("2024-01-12", overall=63.0, risk_appetite=73.0), db_path=tmp_db)

    conn = get_db(tmp_db)
    row = conn.execute("SELECT * FROM macro_score WHERE date = '2024-01-12'").fetchone()
    conn.close()

    assert abs(row["score_velocity"] - 3.0) < 0.01
    assert abs(row["risk_app_velocity"] - 3.0) < 0.01


def test_save_snapshot_first_row_velocity_zero(tmp_db):
    """First row with no prior week should get 0.0 velocity, not NULL."""
    save_snapshot(_make_snap("2024-01-05", overall=60.0), db_path=tmp_db)
    conn = get_db(tmp_db)
    row = conn.execute("SELECT score_velocity, risk_app_velocity FROM macro_score WHERE date = '2024-01-05'").fetchone()
    conn.close()
    assert row["score_velocity"] == 0.0
    assert row["risk_app_velocity"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Integration API
# ─────────────────────────────────────────────────────────────────────────────

def test_get_current_macro_score_no_data(tmp_db):
    assert get_current_macro_score(db_path=tmp_db) == 50.0


def test_get_current_macro_score_returns_latest(tmp_db):
    save_snapshot(_make_snap("2024-01-05", overall=55.0), db_path=tmp_db)
    save_snapshot(_make_snap("2024-01-12", overall=68.0), db_path=tmp_db)
    assert abs(get_current_macro_score(db_path=tmp_db) - 68.0) < 0.01


def test_get_sector_rankings_empty(tmp_db):
    assert get_sector_rankings(db_path=tmp_db) == []


def test_get_sector_rankings_returns_sorted(tmp_db):
    save_snapshot(_make_snap("2024-01-05"), db_path=tmp_db)
    rankings = get_sector_rankings(db_path=tmp_db)
    assert len(rankings) == 2
    assert rankings[0]["ticker"] == "XLK"
    assert rankings[0]["rank_3m"] == 1


def test_get_event_scaling_factor_default(tmp_db):
    assert get_event_scaling_factor(db_path=tmp_db) == 1.0


def test_get_eligible_underlyings_base_always_included(tmp_db):
    tickers = get_eligible_underlyings("neutral", db_path=tmp_db)
    assert "SPY" in tickers
    assert "QQQ" in tickers
    assert "IWM" in tickers


def test_get_eligible_underlyings_bull_adds_top_sectors(tmp_db):
    save_snapshot(_make_snap("2024-01-05", overall=70.0), db_path=tmp_db)
    tickers = get_eligible_underlyings("bull", db_path=tmp_db)
    assert "XLK" in tickers  # rank_3m=1 → top 4


def test_get_eligible_underlyings_bear_macro_contracts(tmp_db):
    save_snapshot(_make_snap("2024-01-05", overall=40.0), db_path=tmp_db)
    tickers = get_eligible_underlyings("bull", db_path=tmp_db)
    # macro score < 45 → base universe only
    assert set(tickers) == {"SPY", "QQQ", "IWM"}


# ─────────────────────────────────────────────────────────────────────────────
# backfill_macro_score_velocities
# ─────────────────────────────────────────────────────────────────────────────

def test_backfill_velocities_updates_all_rows(tmp_db):
    save_snapshot(_make_snap("2024-01-05", overall=60.0, risk_appetite=70.0), db_path=tmp_db)
    save_snapshot(_make_snap("2024-01-12", overall=63.0, risk_appetite=74.0), db_path=tmp_db)
    save_snapshot(_make_snap("2024-01-19", overall=61.0, risk_appetite=72.0), db_path=tmp_db)

    # Manually NULL out the velocity columns to simulate old data
    conn = get_db(tmp_db)
    conn.execute("UPDATE macro_score SET score_velocity = NULL, risk_app_velocity = NULL, overall_v2 = NULL")
    conn.commit()
    conn.close()

    n = backfill_macro_score_velocities(db_path=tmp_db)
    assert n == 3

    conn = get_db(tmp_db)
    rows = conn.execute(
        "SELECT date, overall_v2, score_velocity, risk_app_velocity FROM macro_score ORDER BY date"
    ).fetchall()
    conn.close()

    # Row 1: no prior week → velocities 0.0 (cleaner than NULL), overall_v2 populated
    assert rows[0]["score_velocity"] == 0.0
    assert rows[0]["risk_app_velocity"] == 0.0
    assert abs(rows[0]["overall_v2"] - 60.0) < 0.01

    # Row 2: delta = 63 - 60 = +3
    assert abs(rows[1]["score_velocity"] - 3.0) < 0.01
    assert abs(rows[1]["risk_app_velocity"] - 4.0) < 0.01

    # Row 3: delta = 61 - 63 = -2
    assert abs(rows[2]["score_velocity"] - (-2.0)) < 0.01
    assert abs(rows[2]["risk_app_velocity"] - (-2.0)) < 0.01


def test_backfill_velocities_empty_db(tmp_db):
    assert backfill_macro_score_velocities(db_path=tmp_db) == 0
