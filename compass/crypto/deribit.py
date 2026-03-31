"""Deribit options data collector (public API, no auth required).

Docs: https://docs.deribit.com/#public-get_book_summary_by_currency

Provides P/C ratio, max pain, and OI by strike for BTC options.
All functions return None / {} / [] on any error — never raise.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import requests

_LOG = logging.getLogger(__name__)

_BASE_URL = "https://www.deribit.com/api/v2/public"
_TIMEOUT = 10
_MAX_RETRIES = 3


def _get(method: str, params: Optional[Dict] = None):
    """Call a Deribit public method and return the ``result`` field."""
    url = f"{_BASE_URL}/{method}"
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            if "result" not in payload:
                err = payload.get("error", "unknown error")
                _LOG.error("Deribit API error for %s: %s", method, err)
                return None
            return payload["result"]
        except requests.RequestException as exc:
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                _LOG.warning(
                    "Deribit request failed (attempt %d/%d): %s — retry in %ds",
                    attempt + 1, _MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
            else:
                _LOG.error(
                    "Deribit request failed after %d attempts: %s", _MAX_RETRIES, exc
                )
    return None


def _option_summaries(currency: str = "BTC") -> Optional[List[Dict]]:
    """Fetch all option book summaries for *currency*."""
    return _get("get_book_summary_by_currency", {"currency": currency, "kind": "option"})


def _parse_instrument(name: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Parse an instrument name like ``'BTC-25MAR25-100000-C'``.

    Returns ``(expiry, strike, opt_type)`` where opt_type is ``'C'`` or ``'P'``,
    or ``(None, None, None)`` if the name cannot be parsed.
    """
    parts = name.split("-")
    if len(parts) != 4:
        return None, None, None
    _, expiry, strike_str, opt_type = parts
    try:
        strike = float(strike_str)
    except ValueError:
        return None, None, None
    if opt_type not in ("C", "P"):
        return None, None, None
    return expiry, strike, opt_type


def _build_oi_map(summaries: List[Dict], expiry: str) -> Dict[float, Dict]:
    """Build ``{strike: {"C": call_oi, "P": put_oi}}`` for the given expiry."""
    oi_map: Dict[float, Dict] = {}
    for item in summaries:
        name = item.get("instrument_name", "")
        inst_expiry, strike, opt_type = _parse_instrument(name)
        if inst_expiry != expiry or strike is None:
            continue
        if strike not in oi_map:
            oi_map[strike] = {"C": 0.0, "P": 0.0}
        oi = float(item.get("open_interest") or 0)
        oi_map[strike][opt_type] += oi  # type: ignore[index]
    return oi_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_btc_put_call_ratio() -> Optional[float]:
    """Return current BTC options Put/Call ratio by open interest, or None on error."""
    summaries = _option_summaries("BTC")
    if summaries is None:
        return None
    try:
        call_oi = 0.0
        put_oi = 0.0
        for item in summaries:
            name = item.get("instrument_name", "")
            _, _, opt_type = _parse_instrument(name)
            oi = float(item.get("open_interest") or 0)
            if opt_type == "C":
                call_oi += oi
            elif opt_type == "P":
                put_oi += oi
        if call_oi == 0:
            _LOG.warning("No BTC call OI found — cannot compute P/C ratio")
            return None
        return round(put_oi / call_oi, 4)
    except (TypeError, ValueError) as exc:
        _LOG.error("Error computing BTC P/C ratio: %s", exc)
        return None


def get_btc_max_pain(expiry: str) -> Optional[float]:
    """Return the max-pain strike for BTC options expiring on *expiry* (e.g. ``'25MAR25'``).

    Max pain = the settlement price at which total intrinsic value paid to option
    holders is **minimised** (i.e. option writers keep the most premium).

    Returns None on error or if no options are found for the given expiry.
    """
    summaries = _option_summaries("BTC")
    if summaries is None:
        return None
    try:
        oi_map = _build_oi_map(summaries, expiry)
        if not oi_map:
            _LOG.warning("No BTC options found for expiry %s", expiry)
            return None

        strikes = sorted(oi_map.keys())
        min_pain = float("inf")
        max_pain_strike: Optional[float] = None

        for candidate in strikes:
            pain = 0.0
            for k, oi in oi_map.items():
                if k < candidate:
                    # ITM call at this settlement: holder receives (candidate - k) per contract
                    pain += (candidate - k) * oi["C"]
                elif k > candidate:
                    # ITM put at this settlement: holder receives (k - candidate) per contract
                    pain += (k - candidate) * oi["P"]
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = candidate

        return max_pain_strike
    except (TypeError, ValueError) as exc:
        _LOG.error("Error computing BTC max pain for expiry %s: %s", expiry, exc)
        return None


def get_btc_oi_by_strike(expiry: str) -> dict:
    """Return OI distribution by strike for BTC options expiring on *expiry*.

    Returns::

        {
            100000.0: {"call_oi": 42.5, "put_oi": 18.0},
            ...
        }

    Returns {} on error.
    """
    summaries = _option_summaries("BTC")
    if summaries is None:
        return {}
    try:
        raw = _build_oi_map(summaries, expiry)
        # Re-key to the documented public interface names
        return {
            strike: {"call_oi": oi["C"], "put_oi": oi["P"]}
            for strike, oi in raw.items()
        }
    except (TypeError, ValueError) as exc:
        _LOG.error("Error fetching BTC OI by strike for expiry %s: %s", expiry, exc)
        return {}
