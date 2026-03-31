"""
Unit tests for api/macro_api.py (FastAPI endpoints)
Uses FastAPI TestClient with a temporary DB injected via env var override.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set a test API key before importing the app
os.environ.setdefault("MACRO_API_KEYS", "test-key-123")

from fastapi.testclient import TestClient

import api.macro_api as _api_module
import compass.macro_db as _canonical_db_module
import shared.macro_state_db as _db_module
from api.macro_api import MACRO_CACHE_DB, app
from compass.macro_db import init_db, save_snapshot

# Whether the real price cache is available (required for /regime endpoint)
_CACHE_AVAILABLE = MACRO_CACHE_DB.exists()
requires_cache = pytest.mark.skipif(
    not _CACHE_AVAILABLE, reason="macro_cache.db not available"
)

TEST_KEY = "test-key-123"
AUTH = {"X-API-Key": TEST_KEY}


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: patch MACRO_DB_PATH to a temp file for each test
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Redirect all DB calls to a fresh temp DB for the duration of each test."""
    db = str(tmp_path / "macro_state.db")
    init_db(db)

    monkeypatch.setattr(_canonical_db_module, "MACRO_DB_PATH", Path(db))
    monkeypatch.setattr(_db_module, "MACRO_DB_PATH", Path(db))
    monkeypatch.setattr(_api_module, "MACRO_DB_PATH", Path(db))

    # Patch the functions used by endpoints to use tmp db
    orig_get_db = _canonical_db_module.get_db

    def patched_get_db(path=None):
        return orig_get_db(db)

    monkeypatch.setattr(_canonical_db_module, "get_db", patched_get_db)
    monkeypatch.setattr(_db_module, "get_db", patched_get_db)
    monkeypatch.setattr(_api_module, "get_db", patched_get_db)

    yield db


def _make_snap(snap_date: str, overall: float = 65.0, risk_appetite: float = 75.0) -> dict:
    return {
        "date": snap_date,
        "spy_close": 500.0,
        "top_sector_3m": "XLK",
        "top_sector_12m": "XLK",
        "leading_sectors": ["XLK"],
        "lagging_sectors": ["XLE"],
        "macro_score": {
            "overall": overall,
            "growth": 60.0,
            "inflation": 55.0,
            "fed_policy": 50.0,
            "risk_appetite": risk_appetite,
            "indicators": {
                "vix": 15.0,
                "t10y2y": 0.5,
                "hy_oas_pct": 3.0,
                "cpi_yoy_pct": 2.8,
                "core_cpi_yoy_pct": 3.1,
                "fedfunds": 4.5,
                "cfnai_3m": 0.1,
                "payrolls_3m_avg_k": 200.0,
                "breakeven_5y": 2.2,
            },
        },
        "sector_rankings": [
            {
                "ticker": "XLK", "name": "Technology", "category": "sector",
                "close": 200.0, "rs_3m": 5.0, "rs_12m": 15.0,
                "rs_ratio": 103.0, "rs_momentum": 102.0, "rrg_quadrant": "Leading",
                "rank_3m": 1, "rank_12m": 1,
            },
        ],
    }


client = TestClient(app, raise_server_exceptions=True)


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_no_auth_required():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "version" in body
    assert "snapshot_count" in body


def test_health_returns_ok_with_data(isolated_db):
    save_snapshot(_make_snap("2024-06-07"), db_path=isolated_db)
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["snapshot_count"] == 1
    assert r.json()["latest_date"] == "2024-06-07"


# ─────────────────────────────────────────────────────────────────────────────
# Auth enforcement
# ─────────────────────────────────────────────────────────────────────────────

def test_auth_missing_key_returns_401():
    r = client.get("/api/v1/macro/score")
    assert r.status_code == 401


def test_auth_invalid_key_returns_401():
    r = client.get("/api/v1/macro/score", headers={"X-API-Key": "bad-key"})
    assert r.status_code == 401


def test_auth_valid_key_passes():
    r = client.get("/api/v1/macro/score", headers=AUTH)
    # 404 (no data) is fine — it got past auth
    assert r.status_code in (200, 404)


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/macro/score
# ─────────────────────────────────────────────────────────────────────────────

def test_macro_score_404_when_empty():
    r = client.get("/api/v1/macro/score", headers=AUTH)
    assert r.status_code == 404


def test_macro_score_returns_data(isolated_db):
    save_snapshot(_make_snap("2024-06-07", overall=67.5, risk_appetite=78.0), db_path=isolated_db)
    r = client.get("/api/v1/macro/score", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert abs(body["overall"] - 67.5) < 0.01
    assert body["regime"] == "BULL_MACRO"
    assert body["as_of_date"] == "2024-06-07"
    assert "indicators" in body
    assert body["indicators"]["vix"] == 15.0


def test_macro_score_includes_velocity(isolated_db):
    save_snapshot(_make_snap("2024-06-07", overall=60.0, risk_appetite=70.0), db_path=isolated_db)
    save_snapshot(_make_snap("2024-06-14", overall=64.0, risk_appetite=74.0), db_path=isolated_db)
    r = client.get("/api/v1/macro/score", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert abs(body["score_velocity"] - 4.0) < 0.01
    assert abs(body["risk_app_velocity"] - 4.0) < 0.01
    assert abs(body["overall_v2"] - 64.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/macro/snapshot/latest
# ─────────────────────────────────────────────────────────────────────────────

def test_latest_snapshot_404_when_empty():
    r = client.get("/api/v1/macro/snapshot/latest", headers=AUTH)
    assert r.status_code == 404


def test_latest_snapshot_returns_full_data(isolated_db):
    save_snapshot(_make_snap("2024-06-07", overall=65.0), db_path=isolated_db)
    r = client.get("/api/v1/macro/snapshot/latest", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["date"] == "2024-06-07"
    assert abs(body["spy_close"] - 500.0) < 0.01
    assert len(body["sector_rankings"]) == 1
    assert body["sector_rankings"][0]["ticker"] == "XLK"
    assert body["macro_score"]["regime"] == "BULL_MACRO"


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/macro/snapshot/{date}
# ─────────────────────────────────────────────────────────────────────────────

def test_snapshot_by_date_found(isolated_db):
    save_snapshot(_make_snap("2024-06-07"), db_path=isolated_db)
    r = client.get("/api/v1/macro/snapshot/2024-06-07", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["date"] == "2024-06-07"


def test_snapshot_by_date_not_found(isolated_db):
    r = client.get("/api/v1/macro/snapshot/2020-01-01", headers=AUTH)
    assert r.status_code == 404


def test_snapshot_by_date_invalid_format():
    r = client.get("/api/v1/macro/snapshot/not-a-date", headers=AUTH)
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/macro/sectors
# ─────────────────────────────────────────────────────────────────────────────

def test_sectors_404_when_empty():
    r = client.get("/api/v1/macro/sectors", headers=AUTH)
    assert r.status_code == 404


def test_sectors_returns_ranked_list(isolated_db):
    save_snapshot(_make_snap("2024-06-07"), db_path=isolated_db)
    r = client.get("/api/v1/macro/sectors", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert len(body) >= 1
    assert body[0]["ticker"] == "XLK"
    assert body[0]["rrg_quadrant"] == "Leading"


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/macro/eligible
# ─────────────────────────────────────────────────────────────────────────────

def test_eligible_invalid_regime_422():
    r = client.get("/api/v1/macro/eligible?regime=SIDEWAYS", headers=AUTH)
    assert r.status_code == 422


def test_eligible_returns_base_universe(isolated_db):
    r = client.get("/api/v1/macro/eligible?regime=NEUTRAL", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert "SPY" in body["tickers"]
    assert "QQQ" in body["tickers"]
    assert "IWM" in body["tickers"]


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/macro/history
# ─────────────────────────────────────────────────────────────────────────────

def test_history_returns_summaries(isolated_db):
    save_snapshot(_make_snap("2024-06-07", overall=65.0), db_path=isolated_db)
    save_snapshot(_make_snap("2024-06-14", overall=67.0), db_path=isolated_db)
    r = client.get("/api/v1/macro/history?from=2024-06-01&to=2024-06-30", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["date"] == "2024-06-07"
    assert body[1]["date"] == "2024-06-14"


def test_history_invalid_date_422():
    r = client.get("/api/v1/macro/history?from=bad&to=2024-12-31", headers=AUTH)
    assert r.status_code == 422


def test_history_range_too_large_422():
    r = client.get("/api/v1/macro/history?from=2015-01-01&to=2024-12-31", headers=AUTH)
    assert r.status_code == 422


def test_history_to_before_from_422():
    r = client.get("/api/v1/macro/history?from=2024-06-30&to=2024-06-01", headers=AUTH)
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# END-TO-END INTEGRATION: full stack with real temp DB, every endpoint
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def e2e_client(tmp_path, monkeypatch):
    """
    Stand up a fresh DB with realistic multi-snapshot data, patch the app to
    use it, and return a TestClient.  This exercises the full request→DB→response
    stack for every endpoint.
    """
    db = str(tmp_path / "e2e.db")
    init_db(db)

    # Seed two weekly snapshots (week apart, different overall scores)
    snap1 = _make_snap("2025-01-03", overall=55.0, risk_appetite=62.0)
    snap2 = _make_snap("2025-01-10", overall=68.0, risk_appetite=74.0)
    # Give snap2 a second sector so sectors endpoint has >1 item
    snap2["sector_rankings"].append({
        "ticker": "XLF", "name": "Financials", "category": "sector",
        "close": 42.0, "rs_3m": 1.5, "rs_12m": 5.0,
        "rs_ratio": 99.5, "rs_momentum": 100.1, "rrg_quadrant": "Improving",
        "rank_3m": 2, "rank_12m": 2,
    })
    save_snapshot(snap1, db_path=db)
    save_snapshot(snap2, db_path=db)

    monkeypatch.setattr(_canonical_db_module, "MACRO_DB_PATH", Path(db))
    monkeypatch.setattr(_db_module, "MACRO_DB_PATH", Path(db))
    monkeypatch.setattr(_api_module, "MACRO_DB_PATH", Path(db))

    orig_get_db = _canonical_db_module.get_db

    def patched_get_db(path=None):
        return orig_get_db(db)

    monkeypatch.setattr(_canonical_db_module, "get_db", patched_get_db)
    monkeypatch.setattr(_db_module, "get_db", patched_get_db)
    monkeypatch.setattr(_api_module, "get_db", patched_get_db)

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestE2EFullStack:
    """Integration tests: real temp DB, every endpoint exercised end-to-end."""

    def test_health_reports_two_snapshots(self, e2e_client):
        r = e2e_client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["snapshot_count"] == 2
        assert body["latest_date"] == "2025-01-10"
        assert "version" in body
        assert "db_path" in body
        assert "timestamp" in body

    def test_macro_score_latest_and_velocity(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/score", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        # Should be the 2025-01-10 snapshot (overall=68)
        assert abs(body["overall"] - 68.0) < 0.01
        assert abs(body["overall_v2"] - 68.0) < 0.01
        assert body["regime"] == "BULL_MACRO"
        assert body["as_of_date"] == "2025-01-10"
        # velocity = 68 - 55 = +13
        assert abs(body["score_velocity"] - 13.0) < 0.01
        assert abs(body["risk_app_velocity"] - 12.0) < 0.01
        # indicators nested object
        ind = body["indicators"]
        assert ind["vix"] == 15.0
        assert ind["fedfunds"] == 4.5
        assert ind["cpi_yoy_pct"] == 2.8

    def test_latest_snapshot_full_structure(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/snapshot/latest", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert body["date"] == "2025-01-10"
        assert abs(body["spy_close"] - 500.0) < 0.01
        assert body["top_sector_3m"] == "XLK"
        assert "XLK" in body["leading_sectors"]
        # macro_score nested
        ms = body["macro_score"]
        assert ms["regime"] == "BULL_MACRO"
        assert abs(ms["score_velocity"] - 13.0) < 0.01
        # sector_rankings
        assert len(body["sector_rankings"]) == 2
        tickers = [s["ticker"] for s in body["sector_rankings"]]
        assert tickers[0] == "XLK"   # rank_3m=1 first
        assert tickers[1] == "XLF"

    def test_snapshot_by_date_both_dates(self, e2e_client):
        for d in ("2025-01-03", "2025-01-10"):
            r = e2e_client.get(f"/api/v1/macro/snapshot/{d}", headers=AUTH)
            assert r.status_code == 200, f"Expected 200 for {d}"
            assert r.json()["date"] == d

    def test_snapshot_by_date_missing(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/snapshot/2020-01-01", headers=AUTH)
        assert r.status_code == 404

    def test_sectors_sorted_by_rank(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/sectors", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        assert body[0]["ticker"] == "XLK"
        assert body[0]["rank_3m"] == 1
        assert body[1]["ticker"] == "XLF"
        assert body[1]["rrg_quadrant"] == "Improving"
        # All required fields present
        for item in body:
            assert "rs_3m" in item
            assert "rs_12m" in item
            assert "rrg_quadrant" in item

    def test_sectors_category_filter(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/sectors?category=sector", headers=AUTH)
        assert r.status_code == 200
        assert all(s["category"] == "sector" for s in r.json())

    def test_eligible_bull_regime(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/eligible?regime=BULL", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert set(["SPY", "QQQ", "IWM"]) <= set(body["tickers"])
        assert body["macro_regime"] == "BULL_MACRO"
        assert abs(body["macro_score"] - 68.0) < 0.01
        assert body["event_scaling"] == 1.0   # no events seeded
        assert body["as_of_date"] == "2025-01-10"
        # XLK is rank_3m=1, within top 4 → should be added
        assert "XLK" in body["tickers"]

    def test_eligible_bear_regime_param(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/eligible?regime=BEAR", headers=AUTH)
        assert r.status_code == 200
        # macro score=68 (>45) so not bear-macro contracted
        assert "SPY" in r.json()["tickers"]

    def test_eligible_neutral_regime(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/eligible?regime=NEUTRAL", headers=AUTH)
        assert r.status_code == 200
        assert "SPY" in r.json()["tickers"]

    def test_history_full_range(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/history?from=2025-01-01&to=2025-01-31", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        # Ordered ascending
        assert body[0]["date"] == "2025-01-03"
        assert body[1]["date"] == "2025-01-10"
        # Summary fields present
        row = body[1]
        assert abs(row["macro_overall"] - 68.0) < 0.01
        assert row["regime"] == "BULL_MACRO"
        assert row["top_sector_3m"] == "XLK"
        assert abs(row["spy_close"] - 500.0) < 0.01

    def test_history_single_date_range(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/history?from=2025-01-03&to=2025-01-03", headers=AUTH)
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["date"] == "2025-01-03"
        assert abs(body[0]["macro_overall"] - 55.0) < 0.01

    def test_history_empty_range(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/history?from=2020-01-01&to=2020-12-31", headers=AUTH)
        assert r.status_code == 200
        assert r.json() == []

    def test_auth_rejected_on_all_protected_endpoints(self, e2e_client):
        protected = [
            "/api/v1/macro/score",
            "/api/v1/macro/snapshot/latest",
            "/api/v1/macro/snapshot/2025-01-10",
            "/api/v1/macro/sectors",
            "/api/v1/macro/eligible",
            "/api/v1/macro/events",
            "/api/v1/macro/history?from=2025-01-01&to=2025-01-31",
        ]
        for url in protected:
            r = e2e_client.get(url)
            assert r.status_code == 401, f"Expected 401 without auth on {url}"

    def test_events_endpoint_returns_list(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/events", headers=AUTH)
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_events_endpoint_custom_horizon(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/events?horizon_days=30", headers=AUTH)
        assert r.status_code == 200

    def test_first_snapshot_velocity_is_zero(self, e2e_client):
        """First row velocity should be 0.0, not NULL (fix #2)."""
        r = e2e_client.get("/api/v1/macro/snapshot/2025-01-03", headers=AUTH)
        assert r.status_code == 200
        ms = r.json()["macro_score"]
        # save_snapshot now stores 0.0 for first row
        assert ms["score_velocity"] == 0.0
        assert ms["risk_app_velocity"] == 0.0

    def test_auth_rejected_on_regime_endpoint(self, e2e_client):
        r = e2e_client.get("/api/v1/macro/regime")
        assert r.status_code == 401


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/macro/regime — unit tests (auth / param validation, no cache needed)
# ─────────────────────────────────────────────────────────────────────────────

def test_regime_requires_auth():
    r = client.get("/api/v1/macro/regime")
    assert r.status_code == 401


def test_regime_unsupported_underlying_422():
    r = client.get("/api/v1/macro/regime?underlying=QQQ", headers=AUTH)
    assert r.status_code == 422
    assert "QQQ" in r.json()["detail"]


def test_regime_unsupported_underlying_lists_valid_options():
    r = client.get("/api/v1/macro/regime?underlying=TSLA", headers=AUTH)
    assert r.status_code == 422
    detail = r.json()["detail"]
    # Should mention supported tickers
    assert "SPY" in detail


def test_regime_lowercase_underlying_normalized():
    """Lowercase ticker should be normalized to uppercase before validation."""
    # 'spy' is not in REGIME_SUPPORTED_UNDERLYINGS as-is — endpoint uppercases it
    # It should either succeed (if cache present) or give 422 for bad tickers
    r = client.get("/api/v1/macro/regime?underlying=spy", headers=AUTH)
    # If cache present → 200; if not → 503 (not 422, since 'SPY' is valid after upper())
    assert r.status_code in (200, 503)


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/macro/regime — integration tests (require real macro_cache.db)
# ─────────────────────────────────────────────────────────────────────────────

@requires_cache
def test_regime_spy_returns_200():
    r = client.get("/api/v1/macro/regime?underlying=SPY", headers=AUTH)
    assert r.status_code == 200


@requires_cache
def test_regime_response_structure():
    r = client.get("/api/v1/macro/regime?underlying=SPY", headers=AUTH)
    assert r.status_code == 200
    body = r.json()

    # Top-level required fields
    assert body["underlying"] == "SPY"
    assert body["regime"] in ("bull", "bear", "neutral")
    assert isinstance(body["bull_votes"], int)
    assert isinstance(body["bear_votes"], int)
    assert isinstance(body["vix_circuit_breaker"], bool)
    assert isinstance(body["hysteresis_active"], bool)
    assert "as_of" in body
    assert "timestamp" in body

    # VIX field (may be None)
    assert "vix" in body

    # Signals nested object
    sigs = body["signals"]
    assert "ma200" in sigs
    assert "rsi" in sigs
    assert "vix_structure" in sigs


@requires_cache
def test_regime_ma200_signal_fields():
    r = client.get("/api/v1/macro/regime?underlying=SPY", headers=AUTH)
    ma200 = r.json()["signals"]["ma200"]
    assert ma200["vote"] in ("BULL", "BEAR", "ABSTAIN")
    assert ma200["price"] is not None and ma200["price"] > 0
    assert ma200["ma200"] is not None and ma200["ma200"] > 0
    assert ma200["band_upper"] > ma200["band_lower"]
    assert "detail" in ma200


@requires_cache
def test_regime_rsi_signal_fields():
    r = client.get("/api/v1/macro/regime?underlying=SPY", headers=AUTH)
    rsi = r.json()["signals"]["rsi"]
    assert rsi["vote"] in ("BULL", "BEAR", "NEUTRAL", "ABSTAIN")
    assert rsi["rsi"] is not None
    assert 0 <= rsi["rsi"] <= 100
    assert rsi["threshold_bull"] == 55.0
    assert rsi["threshold_bear"] == 45.0


@requires_cache
def test_regime_vix_structure_always_abstains():
    """VIX3M unavailable → vix_structure must always report ABSTAIN."""
    r = client.get("/api/v1/macro/regime?underlying=SPY", headers=AUTH)
    vs = r.json()["signals"]["vix_structure"]
    assert vs["vote"] == "ABSTAIN"
    assert vs["ratio"] is None
    assert vs["vix3m_available"] is False
    assert "VIX3M" in vs["detail"]


@requires_cache
def test_regime_vote_counts_consistent_with_signals():
    """bull_votes and bear_votes must match the signal votes."""
    r = client.get("/api/v1/macro/regime?underlying=SPY", headers=AUTH)
    body = r.json()
    sigs = body["signals"]

    expected_bull = sum(
        1 for s in [sigs["ma200"]["vote"], sigs["rsi"]["vote"]]
        if s == "BULL"
    )
    expected_bear = sum(
        1 for s in [sigs["ma200"]["vote"], sigs["rsi"]["vote"]]
        if s == "BEAR"
    )
    assert body["bull_votes"] == expected_bull
    assert body["bear_votes"] == expected_bear


@requires_cache
def test_regime_as_of_is_valid_date():
    r = client.get("/api/v1/macro/regime?underlying=SPY", headers=AUTH)
    from datetime import date as _date
    as_of = r.json()["as_of"]
    parsed = _date.fromisoformat(as_of)  # raises if invalid
    # Should be recent (within the last few months)
    assert parsed.year >= 2025


@requires_cache
def test_regime_sector_etf_works():
    """Non-SPY ticker from price cache should also return 200."""
    r = client.get("/api/v1/macro/regime?underlying=XLE", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["underlying"] == "XLE"
    assert body["regime"] in ("bull", "bear", "neutral")


@requires_cache
def test_regime_default_underlying_is_spy():
    r_default = client.get("/api/v1/macro/regime", headers=AUTH)
    r_spy     = client.get("/api/v1/macro/regime?underlying=SPY", headers=AUTH)
    assert r_default.status_code == 200
    assert r_default.json()["underlying"] == "SPY"
    # Both requests hit the same data — regime should be identical
    assert r_default.json()["regime"] == r_spy.json()["regime"]


@requires_cache
def test_regime_hysteresis_and_raw_regime_logic():
    """
    If hysteresis_active=True, raw signals alone don't meet the current regime's
    voting threshold. Verify the relationship is internally consistent.
    """
    r = client.get("/api/v1/macro/regime?underlying=SPY", headers=AUTH)
    body = r.json()
    if not body["hysteresis_active"] or body["vix_circuit_breaker"]:
        pytest.skip("Hysteresis not currently active — nothing to verify")

    regime = body["regime"]
    bull_votes = body["bull_votes"]
    bear_votes = body["bear_votes"]

    # If hysteresis is active, the raw signals disagree with the current regime
    if regime == "BULL":
        # Raw signals don't meet BULL threshold (< 2 bull votes)
        assert bull_votes < 2
    elif regime == "NEUTRAL":
        # Raw signals would have flipped to BULL or BEAR
        assert bull_votes >= 2 or bear_votes >= 3
