#!/usr/bin/env python3
"""
account_status.py — Detailed status for all Alpaca paper trading accounts.

Extends check_accounts.py with:
  - Recent orders (filled + open) since a given date
  - Full position details (symbol, qty, unrealized P&L)
  - Summary by experiment

Usage:
    python3 scripts/account_status.py                     # all accounts, 30 days
    python3 scripts/account_status.py --since 2026-03-15  # since paper trading launch
    python3 scripts/account_status.py --exp 400            # just one experiment

Loads credentials from .env.expNNN files. Never hardcodes API keys.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.check_accounts import (
    BASE_URL, _discover_accounts, _headers, _fetch_account, _fetch_positions,
)


def _fetch_orders(key: str, secret: str, since: str, status: str = "all",
                   limit: int = 500) -> list:
    """Fetch /v2/orders filtered by date. Returns list or raises on error."""
    resp = requests.get(
        f"{BASE_URL}/v2/orders",
        headers=_headers(key, secret),
        params={
            "status": status,       # "open", "closed", or "all"
            "after": since,          # ISO date
            "limit": limit,
            "direction": "desc",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def _summarize_positions(positions: list) -> dict:
    """Aggregate position stats: total unrealized P&L, by asset class."""
    total_unrealized = 0.0
    total_market_value = 0.0
    by_class = {"option": 0, "us_equity": 0, "other": 0}

    for p in positions:
        try:
            total_unrealized += float(p.get("unrealized_pl") or 0)
            total_market_value += float(p.get("market_value") or 0)
        except (TypeError, ValueError):
            pass
        asset_class = (p.get("asset_class") or "").lower()
        if "option" in asset_class:
            by_class["option"] += 1
        elif "equity" in asset_class:
            by_class["us_equity"] += 1
        else:
            by_class["other"] += 1

    return {
        "total_unrealized_pl": total_unrealized,
        "total_market_value": total_market_value,
        "by_class": by_class,
    }


def _summarize_orders(orders: list) -> dict:
    """Count orders by status and side."""
    filled = [o for o in orders if o.get("status") == "filled"]
    open_orders = [o for o in orders if o.get("status") in ("new", "accepted", "partially_filled")]
    canceled = [o for o in orders if o.get("status") in ("canceled", "expired")]
    rejected = [o for o in orders if o.get("status") == "rejected"]

    return {
        "total": len(orders),
        "filled": len(filled),
        "open": len(open_orders),
        "canceled": len(canceled),
        "rejected": len(rejected),
        "filled_list": filled,
        "open_list": open_orders,
    }


def check_account_detailed(exp: str, key: str, secret: str,
                             since: str) -> dict:
    """Full detailed check for one account."""
    result = {
        "exp": exp,
        "status": "ERROR",
        "error": None,
        "account": None,
        "positions": [],
        "orders": {"total": 0, "filled": 0, "open": 0, "canceled": 0, "rejected": 0},
        "summary": {},
    }

    try:
        acct = _fetch_account(key, secret)
        if "code" in acct:
            result["error"] = acct.get("message", f"code {acct.get('code')}")
            return result

        result["account"] = {
            "id": acct.get("id", ""),
            "status": acct.get("status", ""),
            "equity": float(acct.get("equity") or 0),
            "last_equity": float(acct.get("last_equity") or 0),
            "buying_power": float(acct.get("buying_power") or 0),
            "portfolio_value": float(acct.get("portfolio_value") or 0),
            "cash": float(acct.get("cash") or 0),
            "created_at": acct.get("created_at", ""),
        }

        positions = _fetch_positions(key, secret)
        result["positions"] = positions
        result["summary"] = _summarize_positions(positions)

        orders = _fetch_orders(key, secret, since)
        result["orders"] = _summarize_orders(orders)
        result["status"] = "OK"

    except requests.exceptions.HTTPError as exc:
        result["error"] = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
    except requests.exceptions.ConnectionError as exc:
        result["error"] = f"Connection error: {exc}"
    except requests.exceptions.Timeout:
        result["error"] = "Request timed out"
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)

    return result


def _print_detailed(results: list) -> None:
    """Print detailed per-account status."""
    print("\n" + "=" * 78)
    print("  PAPER TRADING STATUS — All Alpaca Accounts")
    print("=" * 78)

    alive = [r for r in results if r["status"] == "OK"]
    dead = [r for r in results if r["status"] != "OK"]

    # Summary table
    print(f"\n{'EXP':<6} {'EQUITY':>13} {'CHANGE':>12} {'UNRLZ P&L':>11} "
          f"{'POS':>5} {'FILLED':>7} {'OPEN':>5}")
    print("-" * 78)

    grand_equity = 0.0
    grand_change = 0.0
    grand_pl = 0.0

    for r in alive:
        a = r["account"]
        s = r["summary"]
        o = r["orders"]
        equity = a["equity"]
        last = a["last_equity"]
        change = equity - last if last > 0 else 0
        grand_equity += equity
        grand_change += change
        grand_pl += s.get("total_unrealized_pl", 0)

        print(f"{r['exp']:<6} ${equity:>11,.2f} ${change:>10,.2f} "
              f"${s.get('total_unrealized_pl', 0):>9,.2f} "
              f"{len(r['positions']):>5} {o['filled']:>7} {o['open']:>5}")

    print("-" * 78)
    print(f"{'TOTAL':<6} ${grand_equity:>11,.2f} ${grand_change:>10,.2f} "
          f"${grand_pl:>9,.2f}")

    # Per-account details
    for r in alive:
        print(f"\n  --- {r['exp'].upper()} ({r['account']['id'][:12]}...) ---")
        a = r["account"]
        s = r["summary"]
        o = r["orders"]
        print(f"    Status:       {a['status']}")
        print(f"    Equity:       ${a['equity']:,.2f}")
        print(f"    Cash:         ${a['cash']:,.2f}")
        print(f"    Buying power: ${a['buying_power']:,.2f}")
        print(f"    Market value: ${s.get('total_market_value', 0):,.2f}")
        print(f"    Unrealized P&L: ${s.get('total_unrealized_pl', 0):,.2f}")
        print(f"    Positions: {len(r['positions'])} "
              f"(options={s['by_class']['option']}, equities={s['by_class']['us_equity']})")
        print(f"    Orders:    {o['total']} total ({o['filled']} filled, "
              f"{o['open']} open, {o['canceled']} canceled)")

        # Show position details (top 10)
        if r["positions"]:
            print(f"\n    Top positions:")
            for p in r["positions"][:10]:
                sym = p.get("symbol", "?")
                qty = p.get("qty", "?")
                mv = float(p.get("market_value") or 0)
                upl = float(p.get("unrealized_pl") or 0)
                print(f"      {sym:<22} qty={qty:>6} mv=${mv:>9,.2f} upl=${upl:>+9,.2f}")
            if len(r["positions"]) > 10:
                print(f"      ... and {len(r['positions']) - 10} more")

        # Show recent orders (last 5 filled)
        if o.get("filled_list"):
            print(f"\n    Recent filled orders:")
            for ord in o["filled_list"][:5]:
                t = (ord.get("filled_at") or ord.get("submitted_at") or "")[:19]
                sym = ord.get("symbol", "?")
                side = ord.get("side", "?")
                qty = ord.get("filled_qty") or ord.get("qty") or "?"
                price = ord.get("filled_avg_price") or "?"
                print(f"      {t}  {side.upper():<4} {sym:<22} qty={qty} @ {price}")

    if dead:
        print(f"\n  --- FAILED ACCOUNTS ---")
        for r in dead:
            print(f"    {r['exp']:<10} {r['error']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Detailed Alpaca account status")
    parser.add_argument("--since", default=None,
                        help="ISO date (default: 2026-03-15 paper trading launch)")
    parser.add_argument("--exp", default=None,
                        help="Filter to one experiment (e.g. 400)")
    args = parser.parse_args()

    since = args.since or "2026-03-15"

    accounts = _discover_accounts()
    if not accounts:
        print("No .env.exp* files found.")
        return 1

    # Filter accounts
    if args.exp:
        target = f"exp{args.exp}"
        accounts = {k: v for k, v in accounts.items() if k == target}
        if not accounts:
            print(f"No credentials found for {target}. Expected .env.{target} in project root.")
            return 1

    print(f"Querying {len(accounts)} account(s), orders since {since}...")

    results = []
    for exp, creds in accounts.items():
        r = check_account_detailed(exp, creds["key"], creds["secret"], since)
        results.append(r)

    _print_detailed(results)

    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_err = sum(1 for r in results if r["status"] != "OK")
    print(f"Summary: {n_ok} OK, {n_err} errors")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
