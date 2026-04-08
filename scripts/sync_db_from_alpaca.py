"""
scripts/sync_db_from_alpaca.py
------------------------------
Compare Alpaca paper positions with DB open trades. Report orphans.
Optionally insert orphans as unmanaged records so RiskGate sees them.

Usage:
    python3 scripts/sync_db_from_alpaca.py                   # all accounts, report only
    python3 scripts/sync_db_from_alpaca.py --fix             # insert orphans into DB
    python3 scripts/sync_db_from_alpaca.py --account exp401  # single account
"""

import argparse
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT     = Path(__file__).resolve().parent.parent
BASE_URL = "https://paper-api.alpaca.markets"

# ---------------------------------------------------------------------------
# Account map: id → env file + db path
# ---------------------------------------------------------------------------

def _creds(env_file):
    cfg = dotenv_values(ROOT / env_file)
    return cfg.get("ALPACA_API_KEY", ""), cfg.get("ALPACA_API_SECRET", "")

ACCOUNTS = {
    "exp400": {
        "env":  ".env.exp400",
        "db":   ROOT / "data" / "pilotai_exp400.db",
        "label": "EXP-400 The Champion",
    },
    "exp401": {
        "env":  ".env.exp401",
        "db":   ROOT / "data" / "pilotai_exp401.db",
        "label": "EXP-401 The Blend",
    },
    "exp503": {
        "env":  ".env.exp503",
        "db":   ROOT / "data" / "exp503" / "pilotai_exp503.db",
        "label": "EXP-503 ML V2 Aggressive",
    },
    "exp600": {
        "env":  ".env.exp600",
        "db":   ROOT / "data" / "exp600" / "pilotai_exp600.db",
        "label": "EXP-600 IBIT Adaptive",
    },
}

# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

def alpaca_positions(env_file):
    key, secret = _creds(env_file)
    if not key or not secret:
        raise ValueError(f"Missing credentials in {env_file}")
    resp = requests.get(
        f"{BASE_URL}/v2/positions",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()

# ---------------------------------------------------------------------------
# OCC symbol helpers
# ---------------------------------------------------------------------------

def occ_from_trade(trade):
    """Build OCC symbols for both legs of a DB trade. Returns set of symbols."""
    ticker  = trade["ticker"] or ""
    exp_raw = str(trade["expiration"] or "").split("T")[0]
    short   = trade["short_strike"]
    long    = trade["long_strike"]
    stype   = str(trade["strategy_type"] or "").lower()
    opt     = "P" if "put" in stype else "C"

    if not ticker or not exp_raw or len(exp_raw) < 10:
        return set()

    yy, mm, dd = exp_raw[2:4], exp_raw[5:7], exp_raw[8:10]

    syms = set()
    for strike in (short, long):
        if strike is not None:
            syms.add(f"{ticker}{yy}{mm}{dd}{opt}{int(strike * 1000):08d}")
    return syms


def parse_occ(symbol):
    """Parse OCC symbol → dict with ticker, expiration, option_type, strike."""
    m = re.match(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$", symbol.strip())
    if not m:
        return None
    ticker, yymmdd, cp, strike_raw = m.groups()
    return {
        "ticker":      ticker,
        "expiration":  f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:6]}",
        "option_type": "call" if cp == "C" else "put",
        "strike":      int(strike_raw) / 1000.0,
    }

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

MANAGED_STATUSES = ("open", "pending_open", "unmanaged")


def db_managed_symbols(db_path):
    """Return set of OCC symbols currently tracked in the DB (open + pending_open + unmanaged)."""
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" * len(MANAGED_STATUSES))
    rows = conn.execute(
        f"SELECT ticker, strategy_type, short_strike, long_strike, expiration "
        f"FROM trades WHERE status IN ({placeholders})",
        MANAGED_STATUSES,
    ).fetchall()
    conn.close()

    syms = set()
    for r in rows:
        syms |= occ_from_trade(dict(r))
    return syms


def insert_unmanaged(db_path, symbol, pos):
    """Insert one orphan Alpaca position as an unmanaged trade record."""
    info = parse_occ(symbol)
    if not info:
        return False

    strike   = info["strike"]
    opt_type = info["option_type"]
    side     = pos.get("side", "")
    qty      = abs(int(pos.get("qty", 0) or 0))
    stype    = ("bear_call" if opt_type == "call" else "bull_put")

    trade_id = f"unmanaged-{symbol}"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        INSERT INTO trades
            (id, source, ticker, strategy_type, status,
             short_strike, long_strike, expiration, credit, contracts,
             entry_date, updated_at)
        VALUES (?, 'sync_script', ?, ?, 'unmanaged',
                ?, ?, ?, 0.0, ?,
                ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            short_strike = excluded.short_strike,
            long_strike  = excluded.long_strike,
            expiration   = excluded.expiration,
            contracts    = excluded.contracts,
            updated_at   = datetime('now')
    """, (
        trade_id,
        info["ticker"],
        stype,
        strike if side == "short" else None,
        strike if side == "long"  else None,
        info["expiration"],
        qty,
        datetime.now(timezone.utc).isoformat(),
    ))
    conn.commit()
    conn.close()
    return True

# ---------------------------------------------------------------------------
# Core: compare one account
# ---------------------------------------------------------------------------

def check_account(name, acct):
    result = {"name": name, "label": acct["label"], "error": None,
              "alpaca_count": 0, "orphans": []}

    try:
        positions = alpaca_positions(acct["env"])
    except Exception as e:
        result["error"] = str(e)
        return result

    option_positions = [p for p in positions if "option" in str(p.get("asset_class", "")).lower()]
    result["alpaca_count"] = len(option_positions)

    managed = db_managed_symbols(acct["db"])

    for pos in option_positions:
        if pos["symbol"] not in managed:
            result["orphans"].append(pos)

    return result

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results):
    print()
    for r in results:
        print(f"{'─'*60}")
        print(f"  {r['label']}  ({r['name']})")
        print(f"{'─'*60}")
        if r["error"]:
            print(f"  ERROR: {r['error']}\n")
            continue
        print(f"  Alpaca option positions : {r['alpaca_count']}")
        print(f"  Orphans (not in DB)     : {len(r['orphans'])}")
        if r["orphans"]:
            print()
            for p in r["orphans"]:
                mval = float(p.get("market_value") or 0)
                upl  = float(p.get("unrealized_pl") or 0)
                sign = "+" if upl >= 0 else ""
                print(f"    {p['symbol']:<36}  {p['side']:<5}  qty={p['qty']:<6}"
                      f"  mkt=${mval:>9,.2f}  P&L={sign}${upl:>8,.2f}")
        print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix",     action="store_true", help="Insert orphans into DB as unmanaged")
    parser.add_argument("--account", choices=list(ACCOUNTS.keys()), help="Single account only")
    args = parser.parse_args()

    run = {args.account: ACCOUNTS[args.account]} if args.account else ACCOUNTS

    results = []
    for name, acct in run.items():
        print(f"Checking {name}...", flush=True)
        results.append(check_account(name, acct))

    print_report(results)

    if args.fix:
        print(f"{'='*60}")
        print("  APPLYING FIXES")
        print(f"{'='*60}")
        for r in results:
            if r["error"] or not r["orphans"]:
                continue
            acct = run[r["name"]]
            for pos in r["orphans"]:
                sym = pos["symbol"]
                ok  = insert_unmanaged(acct["db"], sym, pos)
                tag = "inserted" if ok else "SKIPPED (parse failed)"
                print(f"  [{r['name']}] {tag}: {sym}  side={pos['side']}  qty={pos['qty']}")
        print()
    elif any(r["orphans"] for r in results):
        print("Run with --fix to insert orphans into the DB as unmanaged records.")


if __name__ == "__main__":
    main()
