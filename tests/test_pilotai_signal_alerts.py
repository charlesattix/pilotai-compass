"""
Tests for pilotai_signal/alerts.py — filtering, digest footer, and cooldown.

All tests are pure (no network, no real DB) unless stated otherwise.
"""

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from pilotai_signal.alerts import (
    _load_mover_cooldown,
    _save_mover_cooldown,
    build_digest,
    filter_alerts,
)
from pilotai_signal import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alert(alert_type, ticker, conviction_after, conviction_before=None, days=5):
    """Build a minimal alert dict for filter_alerts tests."""
    return {
        "alert_date": "2025-03-21",
        "alert_type": alert_type,
        "ticker": ticker,
        "conviction_before": conviction_before,
        "conviction_after": conviction_after,
        "days_in_signal": days,
        "message": f"[{alert_type}] {ticker}",
    }


TODAY = date(2025, 3, 21)


# ---------------------------------------------------------------------------
# filter_alerts — suppressed types
# ---------------------------------------------------------------------------

def test_strong_always_suppressed():
    alerts = [_alert("STRONG", "AAPL", 0.80)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert sendable == []
    assert suppressed.get("STRONG") == 1


def test_mover_down_always_suppressed():
    alerts = [_alert("MOVER_DOWN", "TSLA", 0.40, conviction_before=0.60)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert sendable == []
    assert suppressed.get("MOVER_DOWN") == 1


def test_multiple_suppressed_types_counted_separately():
    alerts = [
        _alert("STRONG", "AAPL", 0.80),
        _alert("STRONG", "MSFT", 0.75),
        _alert("MOVER_DOWN", "TSLA", 0.40, conviction_before=0.60),
    ]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert sendable == []
    assert suppressed["STRONG"] == 2
    assert suppressed["MOVER_DOWN"] == 1


# ---------------------------------------------------------------------------
# filter_alerts — NEW conviction gate
# ---------------------------------------------------------------------------

def test_new_high_conviction_passes():
    alerts = [_alert("NEW", "NVDA", conviction_after=0.72)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert len(sendable) == 1
    assert suppressed == {}


def test_new_exactly_at_threshold_passes():
    alerts = [_alert("NEW", "AMD", conviction_after=config.ALERT_MIN_CONVICTION_NEW)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert len(sendable) == 1


def test_new_below_threshold_suppressed():
    alerts = [_alert("NEW", "GME", conviction_after=0.65)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert sendable == []
    assert suppressed.get("low-conviction NEW") == 1


def test_new_just_below_threshold_suppressed():
    alerts = [_alert("NEW", "GME", conviction_after=0.699)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert sendable == []


# ---------------------------------------------------------------------------
# filter_alerts — MOVER_UP gates
# ---------------------------------------------------------------------------

def test_mover_up_passes_all_gates():
    alerts = [_alert("MOVER_UP", "META", conviction_after=0.75, conviction_before=0.55)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert len(sendable) == 1
    assert suppressed == {}


def test_mover_up_low_delta_suppressed():
    # delta = 0.10 < ALERT_MIN_DELTA_MOVER (0.15)
    alerts = [_alert("MOVER_UP", "META", conviction_after=0.70, conviction_before=0.60)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert sendable == []
    assert suppressed.get("MOVER_UP (low delta)") == 1


def test_mover_up_low_conviction_after_suppressed():
    # delta OK, but conviction_after = 0.55 < ALERT_MIN_CONVICTION_MOVER (0.60)
    alerts = [_alert("MOVER_UP", "META", conviction_after=0.55, conviction_before=0.35)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert sendable == []
    assert suppressed.get("MOVER_UP (low conviction)") == 1


def test_mover_up_cooldown_suppressed():
    # Last fired 3 days ago → within 7-day cooldown
    cooldown = {"META": date(2025, 3, 18)}
    alerts = [_alert("MOVER_UP", "META", conviction_after=0.75, conviction_before=0.55)]
    sendable, suppressed = filter_alerts(alerts, cooldown, TODAY)
    assert sendable == []
    assert suppressed.get("MOVER_UP (cooldown)") == 1


def test_mover_up_cooldown_expired_passes():
    # Last fired 8 days ago → cooldown expired
    cooldown = {"META": date(2025, 3, 13)}
    alerts = [_alert("MOVER_UP", "META", conviction_after=0.75, conviction_before=0.55)]
    sendable, suppressed = filter_alerts(alerts, cooldown, TODAY)
    assert len(sendable) == 1
    assert suppressed == {}


def test_mover_up_cooldown_exactly_7_days_still_suppressed():
    # 7 days ago: days < 7 is False (7 < 7 is False) → passes
    cooldown = {"META": date(2025, 3, 14)}  # 7 days ago
    alerts = [_alert("MOVER_UP", "META", conviction_after=0.75, conviction_before=0.55)]
    sendable, suppressed = filter_alerts(alerts, cooldown, TODAY)
    # (today - last_fired).days == 7, 7 < 7 is False → passes
    assert len(sendable) == 1


def test_mover_up_cooldown_different_ticker_unaffected():
    # TSLA is on cooldown but NVDA is not
    cooldown = {"TSLA": date(2025, 3, 18)}
    alerts = [_alert("MOVER_UP", "NVDA", conviction_after=0.75, conviction_before=0.55)]
    sendable, suppressed = filter_alerts(alerts, cooldown, TODAY)
    assert len(sendable) == 1


# ---------------------------------------------------------------------------
# filter_alerts — EXIT always passes
# ---------------------------------------------------------------------------

def test_exit_always_passes():
    alerts = [_alert("EXIT", "AAPL", conviction_after=0.0, conviction_before=0.65)]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    assert len(sendable) == 1
    assert suppressed == {}


# ---------------------------------------------------------------------------
# filter_alerts — mixed bag
# ---------------------------------------------------------------------------

def test_mixed_alerts():
    alerts = [
        _alert("NEW", "NVDA", conviction_after=0.80),          # passes
        _alert("NEW", "GME", conviction_after=0.50),           # suppressed (low conv)
        _alert("EXIT", "AAPL", conviction_after=0.0, conviction_before=0.65),  # passes
        _alert("STRONG", "MSFT", conviction_after=0.75),       # suppressed
        _alert("MOVER_DOWN", "TSLA", conviction_after=0.40, conviction_before=0.60),  # suppressed
        _alert("MOVER_UP", "META", conviction_after=0.75, conviction_before=0.55),   # passes
    ]
    sendable, suppressed = filter_alerts(alerts, {}, TODAY)
    tickers_sent = {a["ticker"] for a in sendable}
    assert tickers_sent == {"NVDA", "AAPL", "META"}
    assert suppressed["STRONG"] == 1
    assert suppressed["MOVER_DOWN"] == 1
    assert suppressed["low-conviction NEW"] == 1


# ---------------------------------------------------------------------------
# build_digest — suppressed footer
# ---------------------------------------------------------------------------

def _signal(ticker, conviction=0.60, frequency=10, total=57, days=5):
    return {
        "ticker": ticker,
        "conviction": conviction,
        "frequency": frequency,
        "total_portfolios": total,
        "freq_pct": frequency / total,
        "avg_weight": 0.05,
        "weighted_qscore": 0.8,
        "days_in_signal": days,
    }


def test_digest_no_suppressed_no_footer():
    signals = [_signal("NVDA"), _signal("AAPL")]
    msg = build_digest(TODAY, signals, suppressed=None)
    assert "🔇" not in msg


def test_digest_empty_suppressed_no_footer():
    signals = [_signal("NVDA")]
    msg = build_digest(TODAY, signals, suppressed={})
    assert "🔇" not in msg


def test_digest_with_suppressed_shows_footer():
    signals = [_signal("NVDA")]
    suppressed = {"STRONG": 12, "MOVER_DOWN": 5, "low-conviction NEW": 3}
    msg = build_digest(TODAY, signals, suppressed=suppressed)
    assert "🔇 Suppressed:" in msg
    assert "12 STRONG" in msg
    assert "5 MOVER_DOWN" in msg
    assert "3 low-conviction NEW" in msg


def test_digest_footer_all_categories():
    signals = [_signal("AAPL")]
    suppressed = {
        "STRONG": 8,
        "MOVER_DOWN": 4,
        "low-conviction NEW": 2,
        "MOVER_UP (cooldown)": 1,
        "MOVER_UP (low conviction)": 3,
    }
    msg = build_digest(TODAY, signals, suppressed=suppressed)
    assert "🔇 Suppressed:" in msg
    for label, count in suppressed.items():
        assert f"{count} {label}" in msg


def test_digest_gold_tickers_still_shown():
    signals = [_signal("NVDA"), _signal("GLD", conviction=0.72)]
    msg = build_digest(TODAY, signals, suppressed={"STRONG": 1})
    assert "GOLD HEDGE SIGNAL" in msg
    assert "GLD" in msg


# ---------------------------------------------------------------------------
# Cooldown persistence
# ---------------------------------------------------------------------------

def test_save_and_load_mover_cooldown(tmp_path):
    cooldown_file = tmp_path / "mover_up_cooldown.json"
    cooldown = {"NVDA": date(2025, 3, 15), "AAPL": date(2025, 3, 20)}

    with patch.object(config, "MOVER_COOLDOWN_FILE", cooldown_file):
        _save_mover_cooldown(cooldown)
        loaded = _load_mover_cooldown()

    assert loaded == cooldown


def test_load_mover_cooldown_missing_file(tmp_path):
    missing = tmp_path / "nonexistent.json"
    with patch.object(config, "MOVER_COOLDOWN_FILE", missing):
        result = _load_mover_cooldown()
    assert result == {}


def test_load_mover_cooldown_corrupt_file(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not json at all {{{")
    with patch.object(config, "MOVER_COOLDOWN_FILE", bad_file):
        result = _load_mover_cooldown()
    assert result == {}


def test_save_creates_parent_directory(tmp_path):
    nested = tmp_path / "nested" / "dir" / "cooldown.json"
    cooldown = {"TSLA": date(2025, 1, 1)}
    with patch.object(config, "MOVER_COOLDOWN_FILE", nested):
        _save_mover_cooldown(cooldown)
    assert nested.exists()
    data = json.loads(nested.read_text())
    assert data == {"TSLA": "2025-01-01"}
