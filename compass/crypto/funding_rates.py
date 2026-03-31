"""OKX perpetual futures funding rate collector.

Public endpoints — no API key required.
Docs: https://www.okx.com/docs-v5/en/#public-data-rest-api-get-funding-rate
      https://www.okx.com/docs-v5/en/#public-data-rest-api-get-funding-rate-history

Replaces the previous Binance collector (HTTP 451 from US IPs) and Bybit
(HTTP 403 from US IPs). OKX public market data endpoints are US-accessible.

All functions return None / [] on any error — never raise.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Dict, List, Optional

import requests

_LOG = logging.getLogger(__name__)

_BASE_URL = "https://www.okx.com/api/v5/public"
_TIMEOUT = 10
_MAX_RETRIES = 3

# OKX history endpoint returns max 100 records per page.
# 100 records at 3 settlements/day covers ~33 days — sufficient for default window.
_MAX_PAGE_SIZE = 100

# Instrument IDs for USDT-margined perpetual swaps
_BTC_INST = "BTC-USDT-SWAP"
_ETH_INST = "ETH-USDT-SWAP"


def _get(path: str, params: Optional[Dict] = None):
    """GET {_BASE_URL}/{path} with retries.

    Returns the ``data`` list on success, None on permanent failure.
    OKX wraps responses in ``{"code": "0", "data": [...]}``.
    """
    url = f"{_BASE_URL}/{path}"
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("code") != "0":
                _LOG.error(
                    "OKX API error for %s: code=%s msg=%s",
                    path, payload.get("code"), payload.get("msg"),
                )
                return None
            return payload.get("data")
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                _LOG.warning(
                    "OKX request failed (attempt %d/%d): %s — retry in %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                _LOG.error(
                    "OKX request failed after %d attempts: %s", _MAX_RETRIES, exc
                )
    return None


def _current_funding(inst_id: str) -> Optional[float]:
    """Return the most recently *settled* funding rate for *inst_id*.

    OKX's /funding-rate endpoint returns the current period's data including
    ``settFundingRate`` (the last settled rate) and ``fundingRate`` (the
    current period's estimated rate). We return ``settFundingRate`` to match
    the convention of the previous Binance/Bybit collectors.
    """
    data = _get("funding-rate", {"instId": inst_id})
    if not data:
        return None
    try:
        entry = data[0]
        rate_str = entry.get("settFundingRate") or entry.get("fundingRate")
        if rate_str is None:
            _LOG.warning("OKX funding-rate response missing rate field for %s", inst_id)
            return None
        return float(rate_str)
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        _LOG.error("Unexpected OKX funding rate payload for %s: %s", inst_id, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_btc_funding() -> Optional[float]:
    """Return the current BTC/USDT perpetual funding rate as a float, or None on error."""
    return _current_funding(_BTC_INST)


def get_eth_funding() -> Optional[float]:
    """Return the current ETH/USDT perpetual funding rate as a float, or None on error."""
    return _current_funding(_ETH_INST)


def get_funding_history(symbol: str, days: int = 30) -> List[Dict]:
    """Return historical funding rates for *symbol* over the past *days* days.

    OKX settles funding every 8 hours (3 × per day), so 30 days ≈ 90 records.

    *symbol* accepts either OKX instrument IDs (``"BTC-USDT-SWAP"``) or the
    legacy Binance-style symbol names (``"BTCUSDT"``, ``"ETHUSDT"``), which are
    silently normalised to the corresponding OKX instrument ID.

    Each entry::

        {"symbol": str, "funding_rate": float, "funding_time": int (ms epoch)}

    Returns [] on error.
    """
    inst_id = _normalise_symbol(symbol)
    limit = min(days * 3 + 1, _MAX_PAGE_SIZE)
    data = _get("funding-rate-history", {"instId": inst_id, "limit": limit})
    if data is None:
        return []
    try:
        return [
            {
                "symbol": entry["instId"],
                "funding_rate": float(entry.get("realizedRate") or entry["fundingRate"]),
                "funding_time": int(entry["fundingTime"]),
            }
            for entry in data
        ]
    except (KeyError, TypeError, ValueError) as exc:
        _LOG.error("Unexpected OKX funding history payload: %s", exc)
        return []


def _normalise_symbol(symbol: str) -> str:
    """Map legacy Binance-style symbols to OKX instrument IDs."""
    _MAP = {
        "BTCUSDT": _BTC_INST,
        "ETHUSDT": _ETH_INST,
    }
    return _MAP.get(symbol.upper(), symbol)
