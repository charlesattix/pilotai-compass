#!/usr/bin/env python3
"""
close_all_positions.py — Market-close all open positions in an Alpaca paper account.

Usage:
    python3 scripts/close_all_positions.py .env.exp059           # shows positions, asks for confirmation
    python3 scripts/close_all_positions.py .env.exp059 --force   # skips confirmation prompt

Safety:
    - Always prints current positions before acting
    - Requires --force flag or interactive confirmation
    - Only works against paper accounts (checks ALPACA_PAPER=true)
    - Polls for fill status after submitting close orders
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALPACA_BASE = "https://paper-api.alpaca.markets"
TIMEOUT = 20
POLL_INTERVAL = 2   # seconds between fill checks
POLL_MAX = 60       # max seconds to wait for fills


# ── Color ──────────────────────────────────────────────────────────────────────
class C:
    BOLD  = "\033[1m"
    GREEN = "\033[0;32m"
    RED   = "\033[0;31m"
    YEL   = "\033[1;33m"
    CYAN  = "\033[0;36m"
    DIM   = "\033[2m"
    NC    = "\033[0m"


# ── Env file ───────────────────────────────────────────────────────────────────
def read_env(path: Path) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result


# ── Alpaca HTTP helpers ────────────────────────────────────────────────────────
def _request(method: str, endpoint: str, key: str, secret: str, body: dict = None):
    url = f"{ALPACA_BASE}{endpoint}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body_bytes = e.read()
        try:
            err = json.loads(body_bytes)
        except Exception:
            err = {"message": body_bytes.decode(errors="replace")}
        return {"_http_error": e.code, "_message": err.get("message", str(err))}
    except (urllib.error.URLError, OSError) as e:
        return {"_error": str(e)}


def get(endpoint, key, secret):
    return _request("GET", endpoint, key, secret)


def delete(endpoint, key, secret):
    return _request("DELETE", endpoint, key, secret)


# ── Main logic ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Market-close all open positions in an Alpaca paper account.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("env_file", help="Path to .env file (e.g. .env.exp059)")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    env_path = Path(args.env_file)
    if not env_path.exists():
        env_path = ROOT / args.env_file
    if not env_path.exists():
        sys.exit(f"ERROR: env file not found: {args.env_file}")

    creds = read_env(env_path)
    api_key    = creds.get("ALPACA_API_KEY", "")
    api_secret = creds.get("ALPACA_API_SECRET", "")
    is_paper   = creds.get("ALPACA_PAPER", "false").lower() == "true"

    if not api_key or not api_secret:
        sys.exit(f"ERROR: ALPACA_API_KEY or ALPACA_API_SECRET missing in {env_path}")

    if not is_paper:
        sys.exit("SAFETY HALT: ALPACA_PAPER is not 'true' in env file. "
                 "This script only runs against paper accounts.")

    print()
    print(f"{C.BOLD}╔══════════════════════════════════════════════════╗{C.NC}")
    print(f"{C.BOLD}║  close_all_positions.py                          ║{C.NC}")
    print(f"{C.BOLD}╚══════════════════════════════════════════════════╝{C.NC}")
    print(f"  Env file:  {env_path}")
    print(f"  API key:   {api_key[:8]}...")
    print(f"  Mode:      {C.YEL}PAPER TRADING{C.NC}")
    print()

    # ── Fetch account ──
    acct = get("/v2/account", api_key, api_secret)
    if "_error" in acct or "_http_error" in acct:
        msg = acct.get("_message") or acct.get("_error", "unknown")
        sys.exit(f"ERROR: Could not reach Alpaca account: {msg}")

    acct_num = acct.get("account_number", "?")
    equity   = float(acct.get("equity", 0) or 0)
    print(f"  Account:   {acct_num}")
    print(f"  Equity:    ${equity:,.2f}")
    print()

    # ── Fetch positions ──
    positions = get("/v2/positions", api_key, api_secret)
    if isinstance(positions, dict) and ("_error" in positions or "_http_error" in positions):
        msg = positions.get("_message") or positions.get("_error", "unknown")
        sys.exit(f"ERROR: Could not fetch positions: {msg}")

    if not positions:
        print(f"  {C.GREEN}✓ No open positions. Nothing to close.{C.NC}")
        print()
        return

    print(f"  {C.BOLD}Open positions ({len(positions)}):{C.NC}")
    print(f"  {'Symbol':<34} {'Side':<5} {'Qty':>6}  {'Market Value':>14}  {'Unreal P&L':>12}")
    print(f"  {'─'*34} {'─'*5} {'─'*6}  {'─'*14}  {'─'*12}")
    total_mv  = 0.0
    total_upl = 0.0
    for p in positions:
        sym  = p.get("symbol", "?")
        side = p.get("side", "?")
        qty  = p.get("qty", "?")
        mv   = float(p.get("market_value", 0) or 0)
        upl  = float(p.get("unrealized_pl", 0) or 0)
        total_mv  += mv
        total_upl += upl
        upl_color = C.GREEN if upl >= 0 else C.RED
        print(f"  {sym:<34} {side:<5} {qty:>6}  ${mv:>13,.2f}  "
              f"{upl_color}{'+' if upl >= 0 else ''}{upl:>+11,.2f}{C.NC}")

    print(f"  {'─'*34} {'─'*5} {'─'*6}  {'─'*14}  {'─'*12}")
    upl_color = C.GREEN if total_upl >= 0 else C.RED
    print(f"  {'TOTAL':<34} {'':5} {'':>6}  ${total_mv:>13,.2f}  "
          f"{upl_color}{'+' if total_upl >= 0 else ''}{total_upl:>+11,.2f}{C.NC}")
    print()

    # ── Confirmation ──
    if not args.force:
        print(f"  {C.YEL}{C.BOLD}WARNING: This will market-close ALL {len(positions)} positions above.{C.NC}")
        print("  Type 'yes' to confirm, anything else to abort: ", end="")
        answer = input().strip().lower()
        if answer != "yes":
            print(f"  {C.DIM}Aborted.{C.NC}")
            sys.exit(0)
        print()
    else:
        print(f"  {C.YEL}--force flag set — skipping confirmation.{C.NC}")
        print()

    # ── Submit close orders ──
    print(f"  {C.BOLD}Submitting close orders...{C.NC}")
    submitted = []
    failed = []

    for p in positions:
        sym = p.get("symbol", "?")
        resp = delete(f"/v2/positions/{sym}", api_key, api_secret)
        if isinstance(resp, dict) and ("_error" in resp or "_http_error" in resp):
            msg = resp.get("_message") or resp.get("_error", "unknown")
            print(f"  {C.RED}✗{C.NC} {sym}: {msg}")
            failed.append(sym)
        else:
            order_id = resp.get("id", "?") if isinstance(resp, dict) else "?"
            print(f"  {C.GREEN}✓{C.NC} {sym}: close order submitted (order_id={order_id[:8]}...)")
            submitted.append({"symbol": sym, "order_id": order_id, "pre_mv": float(p.get("market_value", 0) or 0)})

    print()

    if not submitted:
        print(f"  {C.RED}All close orders failed.{C.NC}")
        sys.exit(1)

    # ── Poll for fills ──
    print(f"  {C.BOLD}Waiting for fills (max {POLL_MAX}s)...{C.NC}")
    deadline = time.time() + POLL_MAX
    filled = {}

    while time.time() < deadline and len(filled) < len(submitted):
        orders_resp = get("/v2/orders?status=closed&limit=50", api_key, api_secret)
        if isinstance(orders_resp, list):
            for o in orders_resp:
                oid = o.get("id", "")
                for s in submitted:
                    if s["order_id"] == oid and s["symbol"] not in filled:
                        filled[s["symbol"]] = {
                            "status":   o.get("status", "?"),
                            "filled_at": o.get("filled_at", "?"),
                            "filled_qty": o.get("filled_qty", "?"),
                            "filled_avg_price": o.get("filled_avg_price"),
                        }
        remaining = len(submitted) - len(filled)
        if remaining > 0:
            sys.stdout.write(f"\r  Waiting... {len(filled)}/{len(submitted)} filled")
            sys.stdout.flush()
            time.sleep(POLL_INTERVAL)

    sys.stdout.write("\r" + " " * 50 + "\r")
    sys.stdout.flush()

    # ── Summary ──
    print(f"  {C.BOLD}{'─'*52}{C.NC}")
    print(f"  {C.BOLD}CLOSE SUMMARY{C.NC}")
    print(f"  {C.BOLD}{'─'*52}{C.NC}")
    for s in submitted:
        sym = s["symbol"]
        if sym in filled:
            fill = filled[sym]
            avg_px = fill.get("filled_avg_price")
            px_str = f" @ ${float(avg_px):.4f}" if avg_px else ""
            print(f"  {C.GREEN}✓{C.NC} {sym}: FILLED{px_str} (qty={fill['filled_qty']})")
        else:
            print(f"  {C.YEL}?{C.NC} {sym}: order submitted but fill not confirmed in {POLL_MAX}s")

    if failed:
        for sym in failed:
            print(f"  {C.RED}✗{C.NC} {sym}: close order FAILED")

    print()

    # ── Verify final position count ──
    time.sleep(2)
    final_positions = get("/v2/positions", api_key, api_secret)
    if isinstance(final_positions, list):
        remaining = len(final_positions)
        if remaining == 0:
            print(f"  {C.GREEN}{C.BOLD}✓ Account clear — 0 open positions.{C.NC}")
        else:
            print(f"  {C.YEL}⚠ {remaining} position(s) still open after close attempt:{C.NC}")
            for p in final_positions:
                print(f"    {p.get('symbol', '?')}  qty={p.get('qty','?')}")
    else:
        print(f"  {C.DIM}Could not verify final position count.{C.NC}")

    acct_final = get("/v2/account", api_key, api_secret)
    if "_error" not in acct_final and "_http_error" not in acct_final:
        equity_final = float(acct_final.get("equity", 0) or 0)
        delta = equity_final - equity
        sign = "+" if delta >= 0 else ""
        print(f"  Final equity:  ${equity_final:,.2f}  ({sign}${delta:,.2f} vs before close)")

    print()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
