"""CoinGecko API client for BTC/ETH price, history, and dominance data.

Free tier: max 30 calls/min.  A 2-second inter-call delay is enforced
automatically by _throttle().  Every function returns None (or []) on
any error — never raises.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import requests

_LOG = logging.getLogger(__name__)

_BASE_URL = "https://api.coingecko.com/api/v3"
_TIMEOUT = 10
_MAX_RETRIES = 3
_MIN_INTERVAL = 2.0  # 30 calls/min → 2 s between calls

# Module-level last-call tracker for rate limiting
_last_call_ts: float = 0.0


def _throttle() -> None:
    """Block until at least _MIN_INTERVAL seconds have elapsed since the last call."""
    global _last_call_ts
    elapsed = time.monotonic() - _last_call_ts
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_ts = time.monotonic()


def _get(path: str, params: Optional[Dict] = None):
    """GET {_BASE_URL}/{path} with retries and throttling.

    Returns parsed JSON dict/list on success, None on permanent failure.
    """
    url = f"{_BASE_URL}/{path}"
    for attempt in range(_MAX_RETRIES):
        try:
            _throttle()
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                _LOG.warning(
                    "CoinGecko request failed (attempt %d/%d): %s — retry in %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                _LOG.error(
                    "CoinGecko request failed after %d attempts: %s", _MAX_RETRIES, exc
                )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_btc_price() -> Optional[float]:
    """Return current BTC/USD price as a float, or None on error."""
    data = _get("simple/price", {"ids": "bitcoin", "vs_currencies": "usd"})
    if data is None:
        return None
    try:
        return float(data["bitcoin"]["usd"])
    except (KeyError, TypeError, ValueError) as exc:
        _LOG.error("Unexpected CoinGecko price payload: %s", exc)
        return None


def get_eth_price() -> Optional[float]:
    """Return current ETH/USD price as a float, or None on error."""
    data = _get("simple/price", {"ids": "ethereum", "vs_currencies": "usd"})
    if data is None:
        return None
    try:
        return float(data["ethereum"]["usd"])
    except (KeyError, TypeError, ValueError) as exc:
        _LOG.error("Unexpected CoinGecko price payload: %s", exc)
        return None


def _parse_ohlc(raw) -> List[Dict]:
    """Convert raw [[ts, o, h, l, c], ...] to list of dicts."""
    result = []
    for row in raw:
        try:
            result.append(
                {
                    "time": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                }
            )
        except (IndexError, TypeError, ValueError) as exc:
            _LOG.warning("Skipping malformed OHLC row %s: %s", row, exc)
    return result


def get_btc_history(days: int = 365) -> List[Dict]:
    """Return daily OHLC for BTC as a list of dicts with keys
    {time, open, high, low, close}.  Returns [] on error.
    """
    data = _get("coins/bitcoin/ohlc", {"vs_currency": "usd", "days": days})
    if data is None:
        return []
    try:
        return _parse_ohlc(data)
    except TypeError as exc:
        _LOG.error("Unexpected CoinGecko BTC OHLC payload: %s", exc)
        return []


def get_eth_history(days: int = 365) -> List[Dict]:
    """Return daily OHLC for ETH as a list of dicts with keys
    {time, open, high, low, close}.  Returns [] on error.
    """
    data = _get("coins/ethereum/ohlc", {"vs_currency": "usd", "days": days})
    if data is None:
        return []
    try:
        return _parse_ohlc(data)
    except TypeError as exc:
        _LOG.error("Unexpected CoinGecko ETH OHLC payload: %s", exc)
        return []


def get_btc_dominance() -> Optional[float]:
    """Return BTC market-cap dominance as a percentage (0–100), or None on error."""
    data = _get("global")
    if data is None:
        return None
    try:
        return float(data["data"]["market_cap_percentage"]["btc"])
    except (KeyError, TypeError, ValueError) as exc:
        _LOG.error("Unexpected CoinGecko global payload: %s", exc)
        return None
