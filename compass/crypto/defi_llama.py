"""DeFiLlama stablecoin supply collector.

Docs: https://stablecoins.llama.fi/

Tracks total circulating supply for USDT, USDC, and DAI — the three largest
USD stablecoins.  Rising stablecoin supply signals dry-powder inflows; falling
supply signals capital rotation out of crypto.

All functions return None / [] on any error — never raise.
"""

from __future__ import annotations

import datetime
import logging
import time
from typing import Dict, List, Optional

import requests

_LOG = logging.getLogger(__name__)

_BASE_URL = "https://stablecoins.llama.fi"
_TIMEOUT = 10
_MAX_RETRIES = 3

# Symbols to track for "total" supply.
_TRACKED_SYMBOLS = {"USDT", "USDC", "DAI"}


def _get(path: str, params: dict | None = None):
    """GET {_BASE_URL}/{path} with retries."""
    url = f"{_BASE_URL}/{path}"
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                _LOG.warning(
                    "DeFiLlama request failed (attempt %d/%d): %s — retry in %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                _LOG.error(
                    "DeFiLlama request failed after %d attempts: %s", _MAX_RETRIES, exc
                )
    return None


def _extract_circulating(asset: dict) -> float:
    """Pull the peggedUSD circulating supply from an asset dict."""
    circ = asset.get("circulating") or {}
    # DeFiLlama returns {"peggedUSD": <number>} nested under "circulating"
    return float(circ.get("peggedUSD") or 0)


def _get_asset_list() -> list[dict] | None:
    """Fetch the full stablecoin list from /stablecoins."""
    data = _get("stablecoins")
    if data is None:
        return None
    try:
        return data["peggedAssets"]
    except (KeyError, TypeError) as exc:
        _LOG.error("Unexpected DeFiLlama stablecoins payload: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_total_stablecoin_supply() -> float | None:
    """Return total circulating supply (USD) for USDT + USDC + DAI combined.

    Returns None on error.
    """
    assets = _get_asset_list()
    if assets is None:
        return None
    try:
        total = 0.0
        found = set()
        for asset in assets:
            symbol = (asset.get("symbol") or "").upper()
            if symbol in _TRACKED_SYMBOLS:
                total += _extract_circulating(asset)
                found.add(symbol)
        if not found:
            _LOG.warning("No tracked stablecoins (USDT/USDC/DAI) found in DeFiLlama response")
            return None
        return total
    except (TypeError, ValueError) as exc:
        _LOG.error("Error summing stablecoin supply: %s", exc)
        return None


def get_stablecoin_history(days: int = 90) -> list[dict]:
    """Return combined daily supply history for USDT + USDC + DAI.

    Each entry::

        {"date": int (unix timestamp), "total_supply": float (USD)}

    Entries are sorted by date ascending.  Days with missing data for a
    stablecoin are skipped for that coin only (others still contribute).
    Returns [] on error.
    """
    assets = _get_asset_list()
    if assets is None:
        return []

    # Collect IDs for tracked symbols
    tracked_ids: dict[str, str] = {}  # symbol → id
    try:
        for asset in assets:
            symbol = (asset.get("symbol") or "").upper()
            if symbol in _TRACKED_SYMBOLS:
                asset_id = asset.get("id")
                if asset_id is not None:
                    tracked_ids[symbol] = str(asset_id)
    except (TypeError, KeyError) as exc:
        _LOG.error("Error parsing DeFiLlama asset list: %s", exc)
        return []

    if not tracked_ids:
        _LOG.warning("No tracked stablecoins found in DeFiLlama asset list")
        return []

    # Cutoff: only include dates within the last `days` days
    cutoff_ts = int(
        (datetime.datetime.utcnow() - datetime.timedelta(days=days)).timestamp()
    )

    # date_ts → cumulative supply across all tracked assets
    supply_by_date: dict[int, float] = {}

    for symbol, asset_id in tracked_ids.items():
        history = _get(f"stablecoin/{asset_id}")
        if history is None:
            _LOG.warning("Could not fetch history for %s (id=%s)", symbol, asset_id)
            continue
        try:
            tokens = history.get("tokens") or []
            for entry in tokens:
                ts = int(entry.get("date") or 0)
                if ts < cutoff_ts:
                    continue
                supply = float((entry.get("circulating") or {}).get("peggedUSD") or 0)
                supply_by_date[ts] = supply_by_date.get(ts, 0.0) + supply
        except (TypeError, ValueError, KeyError) as exc:
            _LOG.warning("Error parsing history for %s: %s", symbol, exc)
            continue

    if not supply_by_date:
        return []

    return [
        {"date": ts, "total_supply": supply}
        for ts, supply in sorted(supply_by_date.items())
    ]
