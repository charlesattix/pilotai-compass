"""Crypto Fear & Greed Index collector (alternative.me).

API docs: https://alternative.me/crypto/fear-and-greed-index/#api

All functions return None / [] on any error — never raise.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import requests

_LOG = logging.getLogger(__name__)

_BASE_URL = "https://api.alternative.me/fng/"
_TIMEOUT = 10
_MAX_RETRIES = 3

Classification = str  # "Extreme Fear" | "Fear" | "Neutral" | "Greed" | "Extreme Greed"


def _get(params: Optional[Dict] = None):
    """GET the Fear & Greed endpoint with retries."""
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(_BASE_URL, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                _LOG.warning(
                    "Fear & Greed request failed (attempt %d/%d): %s — retry in %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                _LOG.error(
                    "Fear & Greed request failed after %d attempts: %s", _MAX_RETRIES, exc
                )
    return None


def _parse_entry(entry: Dict) -> Optional[Dict]:
    """Parse a single API data entry into {value, classification, timestamp}."""
    try:
        value = int(entry["value"])
        classification: Classification = (
            entry.get("value_classification") or _classify(value)
        )
        return {
            "value": value,
            "classification": classification,
            "timestamp": int(entry["timestamp"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        _LOG.warning("Skipping malformed Fear & Greed entry %s: %s", entry, exc)
        return None


def _classify(value: int) -> Classification:
    """Map a 0–100 value to the canonical classification label."""
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 75:
        return "Greed"
    return "Extreme Greed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_current() -> Optional[Dict]:
    """Return the current Fear & Greed reading as::

        {"value": int, "classification": str, "timestamp": int}

    Returns None on error.
    """
    data = _get({"limit": 1})
    if data is None:
        return None
    try:
        return _parse_entry(data["data"][0])
    except (KeyError, IndexError, TypeError) as exc:
        _LOG.error("Unexpected Fear & Greed current payload: %s", exc)
        return None


def get_history(days: int = 30) -> List[Dict]:
    """Return up to *days* daily readings, newest first.

    Each entry: ``{"value": int, "classification": str, "timestamp": int}``.
    Returns [] on error.
    """
    data = _get({"limit": days})
    if data is None:
        return []
    try:
        entries = data["data"]
    except (KeyError, TypeError) as exc:
        _LOG.error("Unexpected Fear & Greed history payload: %s", exc)
        return []

    result = []
    for entry in entries:
        parsed = _parse_entry(entry)
        if parsed is not None:
            result.append(parsed)
    return result
