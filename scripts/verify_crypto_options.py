"""
Gate 1 Verification: Polygon options data availability for crypto ETFs.
Checks IBIT, ETHA, and BITO.

Usage:
    python3 scripts/verify_crypto_options.py

Loads POLYGON_API_KEY from .env.exp400
"""

import os
import sys
import time
import json
from datetime import date, datetime
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env.exp400")

API_KEY = os.getenv("POLYGON_API_KEY", "")
if not API_KEY:
    print("ERROR: POLYGON_API_KEY not found in .env.exp400")
    sys.exit(1)

BASE_URL = "https://api.polygon.io"
TICKERS = ["IBIT", "ETHA", "BITO"]
TODAY = date.today().isoformat()
_last_call = 0.0


def _get(url: str, params: dict) -> dict:
    """Rate-limited GET with retry."""
    global _last_call
    elapsed = time.monotonic() - _last_call
    if elapsed < 0.25:
        time.sleep(0.25 - elapsed)
    params["apiKey"] = API_KEY
    resp = requests.get(url, params=params, timeout=30)
    _last_call = time.monotonic()
    if resp.status_code == 429:
        print("  [rate limited] sleeping 5s...")
        time.sleep(5)
        return _get(url.split("?")[0], {k: v for k, v in params.items() if k != "apiKey"})
    resp.raise_for_status()
    return resp.json()


def fetch_contracts(ticker: str) -> list:
    """Fetch all reference contracts for a ticker (paginated)."""
    url = f"{BASE_URL}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": ticker,
        "limit": 1000,
        "expired": "false",
    }
    contracts = []
    pages = 0
    while url and pages < 20:
        data = _get(url, params)
        results = data.get("results", [])
        contracts.extend(results)
        next_url = data.get("next_url")
        url = next_url
        params = {}  # next_url already contains all params except apiKey
        pages += 1
        if not results:
            break
    return contracts


def fetch_snapshot(ticker: str, limit: int = 50) -> list:
    """Fetch live option chain snapshot (greeks, bid/ask, OI, volume)."""
    url = f"{BASE_URL}/v3/snapshot/options/{ticker}"
    params = {
        "limit": limit,
        "expiration_date.gte": TODAY,
    }
    try:
        data = _get(url, params)
        return data.get("results", [])
    except requests.HTTPError as e:
        print(f"  Snapshot error: {e}")
        return []


def fetch_historical_contracts(ticker: str) -> list:
    """Also fetch expired contracts to see historical data range."""
    url = f"{BASE_URL}/v3/reference/options/contracts"
    params = {
        "underlying_ticker": ticker,
        "limit": 1000,
        "expired": "true",
    }
    contracts = []
    pages = 0
    while url and pages < 10:
        data = _get(url, params)
        results = data.get("results", [])
        contracts.extend(results)
        next_url = data.get("next_url")
        url = next_url
        params = {}
        pages += 1
        if not results:
            break
    return contracts


def analyze_contracts(ticker: str, active: list, expired: list) -> dict:
    all_contracts = active + expired
    if not all_contracts:
        return {"ticker": ticker, "total_contracts": 0, "verdict": "NO DATA"}

    # Date range
    exp_dates = sorted(set(c.get("expiration_date", "") for c in all_contracts if c.get("expiration_date")))
    earliest = exp_dates[0] if exp_dates else "N/A"
    latest = exp_dates[-1] if exp_dates else "N/A"

    # Expiry type breakdown (weekly = Fri/non-standard vs monthly = 3rd Fri)
    expirations = sorted(set(exp_dates))
    weekly_count = 0
    monthly_count = 0
    for exp in expirations:
        try:
            d = datetime.strptime(exp, "%Y-%m-%d")
            day_of_month = d.day
            # Monthly = 3rd Friday: day between 15-21 and weekday==4 (Friday)
            if d.weekday() == 4 and 15 <= day_of_month <= 21:
                monthly_count += 1
            else:
                weekly_count += 1
        except ValueError:
            pass

    # Strike breakdown
    strikes_by_expiry = defaultdict(set)
    put_count = 0
    call_count = 0
    for c in all_contracts:
        exp = c.get("expiration_date", "")
        strike = c.get("strike_price")
        ctype = c.get("contract_type", "")
        if strike and exp:
            strikes_by_expiry[exp].add(strike)
        if ctype == "put":
            put_count += 1
        elif ctype == "call":
            call_count += 1

    strikes_per_expiry = [len(v) for v in strikes_by_expiry.values()]
    avg_strikes = sum(strikes_per_expiry) / len(strikes_per_expiry) if strikes_per_expiry else 0
    max_strikes = max(strikes_per_expiry) if strikes_per_expiry else 0

    return {
        "ticker": ticker,
        "total_contracts": len(all_contracts),
        "active_contracts": len(active),
        "expired_contracts": len(expired),
        "unique_expirations": len(expirations),
        "earliest_expiration": earliest,
        "latest_expiration": latest,
        "weekly_expiries": weekly_count,
        "monthly_expiries": monthly_count,
        "puts": put_count,
        "calls": call_count,
        "avg_strikes_per_expiry": round(avg_strikes, 1),
        "max_strikes_per_expiry": max_strikes,
        "verdict": "DATA FOUND" if all_contracts else "NO DATA",
    }


def analyze_snapshot(ticker: str, snapshots: list) -> dict:
    if not snapshots:
        return {"ticker": ticker, "snapshot_count": 0, "verdict": "NO LIVE QUOTES"}

    bids, asks, vols, ois, spreads, ivs = [], [], [], [], [], []
    for s in snapshots:
        q = s.get("last_quote") or s.get("day", {})
        bid = q.get("bid") if q else None
        ask = q.get("ask") if q else None
        vol = (s.get("day") or {}).get("volume")
        oi = s.get("open_interest")
        greeks = s.get("greeks") or {}
        iv = greeks.get("implied_volatility") or s.get("implied_volatility")

        if bid is not None:
            bids.append(bid)
        if ask is not None:
            asks.append(ask)
        if vol is not None:
            vols.append(vol)
        if oi is not None:
            ois.append(oi)
        if bid is not None and ask is not None and bid > 0:
            spread_pct = (ask - bid) / ((ask + bid) / 2) * 100
            spreads.append(spread_pct)
        if iv is not None and iv > 0:
            ivs.append(iv)

    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else None

    def total(lst):
        return int(sum(lst)) if lst else 0

    # Sample first 3 contracts for display
    samples = []
    for s in snapshots[:3]:
        det = s.get("details") or {}
        q = s.get("last_quote") or {}
        day = s.get("day") or {}
        greeks = s.get("greeks") or {}
        samples.append({
            "symbol": det.get("ticker", "?"),
            "type": det.get("contract_type", "?"),
            "strike": det.get("strike_price"),
            "expiry": det.get("expiration_date"),
            "bid": q.get("bid"),
            "ask": q.get("ask"),
            "volume": day.get("volume"),
            "oi": s.get("open_interest"),
            "iv": greeks.get("implied_volatility"),
        })

    return {
        "ticker": ticker,
        "snapshot_count": len(snapshots),
        "avg_bid": avg(bids),
        "avg_ask": avg(asks),
        "avg_bid_ask_spread_pct": avg(spreads),
        "total_volume": total(vols),
        "total_oi": total(ois),
        "avg_iv": avg(ivs),
        "samples": samples,
        "verdict": "LIQUID" if (avg(spreads) or 999) < 15 else "ILLIQUID (wide spreads)",
    }


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    print(f"\nGATE 1 VERIFICATION — Crypto ETF Options on Polygon")
    print(f"Date: {TODAY}")
    print(f"API Key: {API_KEY[:6]}...{API_KEY[-4:]}")

    all_results = {}

    for ticker in TICKERS:
        print_section(f"{ticker}")

        print(f"  Fetching active contracts...")
        active = fetch_contracts(ticker)
        print(f"  → {len(active)} active contracts found")

        print(f"  Fetching expired contracts (historical range)...")
        expired = fetch_historical_contracts(ticker)
        print(f"  → {len(expired)} expired contracts found")

        contract_analysis = analyze_contracts(ticker, active, expired)

        print(f"  Fetching live snapshot (bid/ask/volume/OI)...")
        snapshots = fetch_snapshot(ticker, limit=50)
        print(f"  → {len(snapshots)} snapshot records found")

        snapshot_analysis = analyze_snapshot(ticker, snapshots)

        all_results[ticker] = {
            "contracts": contract_analysis,
            "snapshot": snapshot_analysis,
        }

        # Print summary for this ticker
        ca = contract_analysis
        sa = snapshot_analysis
        print(f"\n  CONTRACT COVERAGE:")
        print(f"    Total contracts:     {ca['total_contracts']} ({ca['active_contracts']} active, {ca['expired_contracts']} expired)")
        print(f"    Date range:          {ca['earliest_expiration']} → {ca['latest_expiration']}")
        print(f"    Unique expirations:  {ca['unique_expirations']} ({ca['weekly_expiries']} weekly, {ca['monthly_expiries']} monthly)")
        print(f"    Put/Call:            {ca['puts']} puts / {ca['calls']} calls")
        print(f"    Avg strikes/expiry:  {ca['avg_strikes_per_expiry']}  (max: {ca['max_strikes_per_expiry']})")
        print(f"    Verdict:             {ca['verdict']}")

        print(f"\n  LIQUIDITY (live snapshot):")
        print(f"    Snapshot records:    {sa['snapshot_count']}")
        print(f"    Avg bid-ask spread:  {sa['avg_bid_ask_spread_pct']}%")
        print(f"    Total volume:        {sa['total_volume']:,}")
        print(f"    Total OI:            {sa['total_oi']:,}")
        print(f"    Avg IV:              {sa['avg_iv']}")
        print(f"    Verdict:             {sa['verdict']}")

        if sa.get("samples"):
            print(f"\n  SAMPLE CONTRACTS:")
            for s in sa["samples"]:
                print(f"    {s['symbol']}")
                print(f"      type={s['type']}, strike={s['strike']}, expiry={s['expiry']}")
                print(f"      bid={s['bid']}, ask={s['ask']}, vol={s['volume']}, OI={s['oi']}, IV={s['iv']}")

    # Final gate verdict
    print_section("GATE 1 VERDICT")
    for ticker in TICKERS:
        ca = all_results[ticker]["contracts"]
        sa = all_results[ticker]["snapshot"]
        status = "✅ PASS" if ca["total_contracts"] > 0 else "❌ FAIL"
        liquid = "✅ LIQUID" if sa["snapshot_count"] > 0 else "⚠️  NO LIVE QUOTES"
        print(f"  {ticker}: {status}  {liquid}")
        print(f"         {ca['total_contracts']} contracts, {ca['unique_expirations']} expirations, {ca['avg_strikes_per_expiry']} avg strikes/exp")

    # Save full results
    out_path = ROOT / "output" / "crypto_options_gate1.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full results saved to: output/crypto_options_gate1.json")


if __name__ == "__main__":
    main()
