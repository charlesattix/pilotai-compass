#!/usr/bin/env python3
"""
Phase 1: ETF Options Liquidity Probe
======================================
Tests each target ETF against Polygon to determine:
  1. Are historical options contracts available? (2021, 2022, 2023)
  2. How many strikes per expiration? (chain depth)
  3. Is daily OHLCV price data available for those contracts?
  4. Are weekly expirations available (not just monthly)?

Output: liquidity tier classification + saves to output/etf_liquidity_probe.json

Run time: ~8-12 minutes (1 call/sec rate limit, ~70-90 API calls total)

Usage:
    python3 scripts/probe_etf_liquidity.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import requests

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("probe")

API_KEY = os.environ.get("POLYGON_API_KEY", "")
BASE_URL = "https://api.polygon.io"

# Target ETFs — order by expected liquidity (most liquid first)
TICKERS = ["QQQ", "IWM", "XLE", "XLK", "XLF", "SOXX", "XLI", "XLY", "XLC", "XBI", "SMH"]

# Known monthly expirations (3rd Fridays) spanning different regimes
MONTHLY_EXPS = [
    ("2021-01-15", "Jan2021_bull"),
    ("2021-06-18", "Jun2021_bull"),
    ("2022-06-17", "Jun2022_bear"),
    ("2023-06-16", "Jun2023_AI"),
]

# A non-3rd-Friday to test for weekly options availability
WEEKLY_TEST_DATE = "2023-06-09"   # 2nd Friday of Jun 2023

# Approximate underlying prices near these expirations (for OTM context)
# Used only for display — does not affect probe logic
APPROX_PRICES = {
    "QQQ":  {"2021-01-15": 315, "2021-06-18": 355, "2022-06-17": 283, "2023-06-16": 362},
    "IWM":  {"2021-01-15": 214, "2021-06-18": 224, "2022-06-17": 171, "2023-06-16": 185},
    "XLE":  {"2021-01-15": 42,  "2021-06-18": 53,  "2022-06-17": 81,  "2023-06-16": 84},
    "XLK":  {"2021-01-15": 134, "2021-06-18": 146, "2022-06-17": 132, "2023-06-16": 169},
    "XLF":  {"2021-01-15": 30,  "2021-06-18": 36,  "2022-06-17": 32,  "2023-06-16": 34},
    "SOXX": {"2021-01-15": 320, "2021-06-18": 390, "2022-06-17": 330, "2023-06-16": 590},
    "XLI":  {"2021-01-15": 92,  "2021-06-18": 101, "2022-06-17": 87,  "2023-06-16": 104},
    "XLY":  {"2021-01-15": 160, "2021-06-18": 183, "2022-06-17": 140, "2023-06-16": 175},
    "XLC":  {"2021-01-15": 66,  "2021-06-18": 79,  "2022-06-17": 56,  "2023-06-16": 71},
    "XBI":  {"2021-01-15": 155, "2021-06-18": 153, "2022-06-17": 76,  "2023-06-16": 83},
    "SMH":  {"2021-01-15": 230, "2021-06-18": 272, "2022-06-17": 196, "2023-06-16": 145},
}

_last_call = 0.0


def api_get(path: str, params: Optional[Dict] = None, timeout: int = 20) -> Optional[Dict]:
    """Rate-limited Polygon GET request (1 call/sec minimum)."""
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)

    p = dict(params or {})
    p["apiKey"] = API_KEY

    try:
        resp = requests.get(f"{BASE_URL}{path}", params=p, timeout=timeout)
        _last_call = time.time()
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 403:
            logger.warning("403 Forbidden for %s — API key issue or data not licensed", path)
            return None
        if resp.status_code == 404:
            return None
        logger.warning("HTTP %d for %s", resp.status_code, path)
        return None
    except requests.exceptions.Timeout:
        logger.warning("Timeout: %s", path)
        _last_call = time.time()
        return None
    except Exception as e:
        logger.warning("Request failed for %s: %s", path, e)
        _last_call = time.time()
        return None


def probe_expiration(ticker: str, exp_date: str, label: str) -> Dict:
    """Query contract listings + verify daily data for one put contract."""
    # ── 1. Get put contracts for this expiration ──────────────────────────────
    data = api_get("/v3/reference/options/contracts", {
        "underlying_ticker": ticker,
        "expiration_date": exp_date,
        "contract_type": "put",
        "as_of": exp_date,
        "limit": 1000,
        "order": "asc",
        "sort": "strike_price",
    })

    if data is None:
        return {"label": label, "status": "api_error", "puts": 0}

    puts = data.get("results", [])
    n_puts = len(puts)

    if n_puts == 0:
        return {"label": label, "status": "no_contracts", "puts": 0}

    # ── 2. Get call count (separate query, quick scan) ───────────────────────
    call_data = api_get("/v3/reference/options/contracts", {
        "underlying_ticker": ticker,
        "expiration_date": exp_date,
        "contract_type": "call",
        "as_of": exp_date,
        "limit": 1000,
        "order": "asc",
        "sort": "strike_price",
    })
    n_calls = len(call_data.get("results", [])) if call_data else 0

    strikes = sorted(c["strike_price"] for c in puts if "strike_price" in c)
    approx_price = APPROX_PRICES.get(ticker, {}).get(exp_date)

    # Count strikes in OTM range (80–100% of spot for puts = 0–20% OTM)
    otm_strikes = [s for s in strikes if approx_price and 0.80 * approx_price <= s <= approx_price]

    # ── 3. Verify daily price data for a ~5% OTM put ─────────────────────────
    daily_ok = False
    sample_sym = None
    sample_price = None

    if puts and approx_price:
        # Find the strike closest to 95% of spot (5% OTM put)
        target_strike = approx_price * 0.95
        closest = min(puts, key=lambda c: abs(c.get("strike_price", 0) - target_strike))
        sym = closest.get("ticker", "")

        if sym:
            price_data = api_get(
                f"/v2/aggs/ticker/{sym}/range/1/day/{exp_date}/{exp_date}",
                {"adjusted": "true", "sort": "asc", "limit": 10},
            )
            if price_data and price_data.get("results"):
                daily_ok = True
                sample_sym = sym
                sample_price = price_data["results"][0].get("c")

    return {
        "label": label,
        "status": "ok",
        "puts": n_puts,
        "calls": n_calls,
        "min_strike": min(strikes) if strikes else None,
        "max_strike": max(strikes) if strikes else None,
        "otm_strikes_5pct_range": len(otm_strikes),  # strikes within 5–20% OTM band
        "daily_data_ok": daily_ok,
        "sample_symbol": sample_sym,
        "sample_price": sample_price,
    }


def check_weeklies(ticker: str) -> bool:
    """Test if weekly (non-3rd-Friday) expirations exist."""
    data = api_get("/v3/reference/options/contracts", {
        "underlying_ticker": ticker,
        "expiration_date": WEEKLY_TEST_DATE,
        "contract_type": "put",
        "as_of": WEEKLY_TEST_DATE,
        "limit": 5,
    })
    return bool(data and data.get("results"))


def probe_ticker(ticker: str) -> Dict:
    """Full probe for one ticker: expirations × (contracts + daily data) + weekly check."""
    print(f"\n  Probing {ticker}...", flush=True)
    result = {
        "ticker": ticker,
        "expirations": [],
        "has_weeklies": False,
        "any_daily_data": False,
        "total_puts_sampled": 0,
    }

    for exp_date, label in MONTHLY_EXPS:
        print(f"    {label} ({exp_date})...", flush=True)
        exp_result = probe_expiration(ticker, exp_date, label)
        result["expirations"].append(exp_result)
        if exp_result.get("daily_data_ok"):
            result["any_daily_data"] = True
        if exp_result.get("puts", 0) > 0:
            result["total_puts_sampled"] += exp_result["puts"]

    # Weekly check (1 call)
    print(f"    Weekly check ({WEEKLY_TEST_DATE})...", flush=True)
    result["has_weeklies"] = check_weeklies(ticker)

    return result


def classify_tier(result: Dict) -> Tuple[str, str]:
    """
    Tier 1: Deep chain (50+ puts/exp), daily data confirmed, has weeklies
    Tier 2: Adequate chain (20+ puts/exp), daily data confirmed, monthly only
    Tier 3 (marginal): Thin chain (<20 puts/exp) or no daily data
    Tier 3 (exclude): No contracts or no daily data at all
    """
    exps = result["expirations"]
    ok_exps = [e for e in exps if e.get("status") == "ok" and e.get("puts", 0) > 0]

    if not ok_exps or not result["any_daily_data"]:
        return "Tier 3", "exclude — no contracts or no daily data"

    avg_puts = sum(e["puts"] for e in ok_exps) / len(ok_exps)
    avg_otm = sum(e.get("otm_strikes_5pct_range", 0) for e in ok_exps) / len(ok_exps)

    if avg_puts >= 50 and result["any_daily_data"] and result["has_weeklies"]:
        return "Tier 1", f"deep ({avg_puts:.0f} puts/exp avg, weeklies available)"
    if avg_puts >= 30 and result["any_daily_data"] and result["has_weeklies"]:
        return "Tier 1", f"adequate ({avg_puts:.0f} puts/exp avg, weeklies available)"
    if avg_puts >= 20 and result["any_daily_data"]:
        return "Tier 2", f"monthly-only ({avg_puts:.0f} puts/exp avg, {avg_otm:.0f} OTM strikes)"
    if avg_puts >= 5 and result["any_daily_data"]:
        return "Tier 3", f"marginal ({avg_puts:.0f} puts/exp avg — thin chain)"
    return "Tier 3", "exclude — insufficient depth"


def main():
    if not API_KEY:
        print("ERROR: POLYGON_API_KEY not set. Check .env file.")
        sys.exit(1)

    n_calls = len(TICKERS) * (len(MONTHLY_EXPS) * 3 + 1)  # 3 calls/exp (puts + calls + daily) + weekly
    print("=" * 72)
    print("PHASE 1: ETF Options Liquidity Probe")
    print(f"Testing {len(TICKERS)} tickers × {len(MONTHLY_EXPS)} expirations + weekly check")
    print(f"Estimated API calls: ~{n_calls}  |  Estimated time: ~{n_calls // 60 + 1} min")
    print("=" * 72)

    all_results = []
    start_ts = time.time()

    for ticker in TICKERS:
        r = probe_ticker(ticker)
        tier, reason = classify_tier(r)
        r["tier"] = tier
        r["tier_reason"] = reason
        all_results.append(r)

    elapsed = time.time() - start_ts

    # Save to output/
    out_path = ROOT / "output" / "etf_liquidity_probe.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed),
            "results": all_results,
        }, f, indent=2)

    # Print summary table
    print("\n" + "=" * 72)
    print("LIQUIDITY TIER RESULTS")
    print("=" * 72)
    print(f"{'Ticker':8s} {'Tier':10s} {'Puts/exp':10s} {'Weeklies':10s} {'Daily$':8s}  Notes")
    print("-" * 72)

    for r in all_results:
        ok_exps = [e for e in r["expirations"] if e.get("status") == "ok"]
        avg_puts = sum(e["puts"] for e in ok_exps) / max(len(ok_exps), 1)
        weeklies = "yes" if r["has_weeklies"] else "no"
        daily = "yes" if r["any_daily_data"] else "NO"
        # Sample price from first expiration that has data
        sample_price = next(
            (f"${e['sample_price']:.2f}" for e in r["expirations"] if e.get("sample_price")),
            "n/a"
        )
        print(f"{r['ticker']:8s} {r['tier']:10s} {avg_puts:8.0f}   {weeklies:8s}   {daily:4s}  {sample_price}  {r.get('tier_reason', '')}")

    print(f"\nElapsed: {elapsed:.0f}s  |  Results: {out_path}")
    print("\nRecommended fetch order (highest alpha first):")
    tier1 = [r["ticker"] for r in all_results if r.get("tier") == "Tier 1"]
    tier2 = [r["ticker"] for r in all_results if r.get("tier") == "Tier 2"]
    print(f"  Tier 1 (fetch immediately): {tier1}")
    print(f"  Tier 2 (fetch after Tier 1): {tier2}")


if __name__ == "__main__":
    main()
