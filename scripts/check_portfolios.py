#!/usr/bin/env python3
"""check_portfolios.py — Query all Alpaca paper trading accounts.

Loads .env files for all live experiments, queries each Alpaca paper account,
and reports equity, P&L, and open positions. Safe to run during market hours
or as a heartbeat check — errors on one account never affect the others.

Usage:
    python scripts/check_portfolios.py            # full report
    python scripts/check_portfolios.py --json     # JSON output (for automation)
    python scripts/check_portfolios.py --summary  # summary table only

Exit codes:
    0  all accounts queried successfully
    1  one or more accounts had errors (still prints successful ones)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── third-party (pip install python-dotenv alpaca-py) ────────────────────────
try:
    from dotenv import dotenv_values
except ImportError:
    sys.exit("ERROR: python-dotenv not installed. Run: pip install python-dotenv")

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
except ImportError:
    sys.exit("ERROR: alpaca-py not installed. Run: pip install alpaca-py")

# ── paths ────────────────────────────────────────────────────────────────────

PROJECT       = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT / "experiments" / "registry.json"
PAPER_BASE    = "https://paper-api.alpaca.markets"

# Ordered list: (env_file, fallback_exp_id, is_legacy)
# is_legacy=True  → shown in report but errors don't affect exit code
# .env.champion   → legacy alias for EXP-400; credentials revoked as of 2026-03-22,
#                   superseded by .env.exp400 which holds the live credentials.
ENV_SLOTS = [
    (".env.champion", "EXP-400", True),   # legacy — stale credentials
    (".env.exp400",   "EXP-400", False),
    (".env.exp401",   "EXP-401", False),
    (".env.exp503",   "EXP-503", False),
    (".env.exp600",   "EXP-600", False),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _exp_label(exp_id: str, registry: dict) -> str:
    """Return 'EXP-400 — The Champion' style label."""
    exp = registry.get("experiments", {}).get(exp_id, {})
    name = exp.get("name", "")
    creator = exp.get("created_by", "")
    parts = [exp_id]
    if name:
        parts.append(name)
    if creator:
        parts.append(f"by {creator}")
    return " — ".join(parts)


def _clean_url(url: str) -> str:
    """Strip trailing /v2 — alpaca-py appends it internally."""
    return url.rstrip("/").removesuffix("/v2").rstrip("/")


def _val(v) -> float:
    """Safe float conversion from Alpaca field (may be Decimal or str)."""
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _side(v) -> str:
    return v.value if hasattr(v, "value") else str(v)


# ── per-account query ────────────────────────────────────────────────────────

def query_account(env_file: str, fallback_exp_id: str, registry: dict,
                  is_legacy: bool = False) -> dict:
    """
    Query one Alpaca paper account. Always returns a dict; puts errors in
    result["error"] rather than raising. Never throws.
    """
    path = PROJECT / env_file
    result: dict = {
        "env_file":    env_file,
        "exp_id":      fallback_exp_id,
        "label":       _exp_label(fallback_exp_id, registry),
        "is_legacy":   is_legacy,
        "error":       None,
        "account_number": None,
        "status":      None,
        "equity":      None,
        "cash":        None,
        "buying_power": None,
        "last_equity": None,
        "day_pnl":     None,
        "day_pnl_pct": None,
        "unrealized_pl": None,
        "positions":   [],
        "open_orders": [],
        "queried_at":  datetime.now(timezone.utc).isoformat(),
    }

    if not path.exists():
        result["error"] = f"env file not found: {env_file}"
        return result

    try:
        cfg        = dotenv_values(path)
        api_key    = cfg.get("ALPACA_API_KEY", "").strip()
        api_secret = cfg.get("ALPACA_API_SECRET", "").strip()
        raw_url    = cfg.get("ALPACA_BASE_URL", "").strip()
        exp_id     = cfg.get("EXPERIMENT_ID", fallback_exp_id).strip()
    except Exception as e:
        result["error"] = f"Failed to parse env file: {e}"
        return result

    result["exp_id"] = exp_id
    result["label"]  = _exp_label(exp_id, registry)

    if not api_key or not api_secret:
        result["error"] = "Missing ALPACA_API_KEY or ALPACA_API_SECRET"
        return result

    # Build URL override only if non-default
    url_override: Optional[str] = None
    if raw_url:
        cleaned = _clean_url(raw_url)
        if cleaned and cleaned != PAPER_BASE:
            url_override = cleaned

    try:
        client = TradingClient(
            api_key, api_secret,
            paper=True,
            url_override=url_override,
        )

        acct        = client.get_account()
        positions   = client.get_all_positions()
        open_orders = client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=50)
        )

        equity      = _val(acct.equity)
        last_equity = _val(acct.last_equity)
        day_pnl     = equity - last_equity
        day_pnl_pct = (day_pnl / last_equity * 100) if last_equity else 0.0

        result.update({
            "account_number": str(getattr(acct, "account_number", "—")),
            "status":         _side(acct.status),
            "equity":         equity,
            "cash":           _val(acct.cash),
            "buying_power":   _val(acct.buying_power),
            "last_equity":    last_equity,
            "day_pnl":        day_pnl,
            "day_pnl_pct":    day_pnl_pct,
            "unrealized_pl":  sum(_val(p.unrealized_pl) for p in positions),
            "positions": [
                {
                    "symbol":      p.symbol,
                    "qty":         str(p.qty),
                    "side":        _side(p.side),
                    "avg_entry":   _val(p.avg_entry_price),
                    "market_val":  _val(p.market_value),
                    "unrealized_pl":   _val(p.unrealized_pl),
                    "unrealized_plpc": _val(p.unrealized_plpc) * 100,
                }
                for p in positions
            ],
            "open_orders": [
                {
                    "symbol":    o.symbol,
                    "side":      _side(o.side),
                    "qty":       str(o.qty),
                    "type":      _side(getattr(o, "order_type", None) or getattr(o, "type", "?")),
                    "status":    _side(o.status),
                    "submitted": str(o.submitted_at)[:19] if o.submitted_at else "—",
                }
                for o in open_orders
            ],
        })

    except Exception as e:
        result["error"] = str(e)

    return result


# ── deduplication ─────────────────────────────────────────────────────────────

def deduplicate(results: list[dict]) -> list[dict]:
    """
    If two env files resolve to the same account_number, keep the first
    and skip the duplicate (marks it skipped in the output).
    """
    seen: set[str] = set()
    out = []
    for r in results:
        acct_no = r.get("account_number")
        if acct_no and acct_no != "—" and acct_no in seen:
            r = dict(r)
            r["error"] = f"duplicate of account {acct_no} (already shown above)"
        elif acct_no and acct_no != "—":
            seen.add(acct_no)
        out.append(r)
    return out


# ── formatting ────────────────────────────────────────────────────────────────

def _sign(v: float) -> str:
    return "+" if v >= 0 else ""


def _pct(v: float) -> str:
    s = "+" if v >= 0 else ""
    return f"{s}{v:.2f}%"


def print_full_report(results: list[dict]) -> None:
    W = 68
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'═'*W}")
    print(f"  PILOTAI — PAPER TRADING PORTFOLIO CHECK")
    print(f"  {now}")
    print(f"{'═'*W}")

    for r in results:
        print(f"\n{'─'*W}")
        legacy_tag = "  [LEGACY]" if r.get("is_legacy") else ""
        print(f"  {r['label']}  ({r['env_file']}){legacy_tag}")
        print(f"{'─'*W}")

        if r["error"]:
            tag = "legacy/stale" if r.get("is_legacy") else "ERROR"
            print(f"  ⚠  {tag}: {r['error']}")
            continue

        dp  = r["day_pnl"]
        up  = r["unrealized_pl"]
        eq  = r["equity"]
        last = r["last_equity"]

        print(f"  Account      : {r['account_number']}  [{r['status']}]")
        print(f"  Equity       : ${eq:>13,.2f}")
        print(f"  Cash         : ${r['cash']:>13,.2f}")
        print(f"  Buying Power : ${r['buying_power']:>13,.2f}")
        print(f"  Day P&L      : {_sign(dp)}${dp:>12,.2f}   ({_pct(r['day_pnl_pct'])})  prev close ${last:,.2f}")
        print(f"  Unrealized   : {_sign(up)}${up:>12,.2f}")

        positions = r["positions"]
        if positions:
            print(f"\n  Open Positions ({len(positions)}):")
            for p in positions:
                pl   = p["unrealized_pl"]
                plpc = p["unrealized_plpc"]
                print(f"    {p['symbol']:<36}  {p['side']:<5}  ×{p['qty']:<5}"
                      f"  P&L: {_sign(pl)}${pl:>8,.2f}  ({_sign(plpc)}{plpc:.1f}%)")
        else:
            print(f"\n  Open Positions : none")

        orders = r["open_orders"]
        if orders:
            print(f"\n  Open Orders ({len(orders)}):")
            for o in orders:
                print(f"    {o['symbol']:<36}  {o['side']:<5}  ×{o['qty']:<5}"
                      f"  {o['type']:<10}  {o['status']}")
        else:
            print(f"  Open Orders    : none")

    _print_summary(results, W)


def print_summary_only(results: list[dict]) -> None:
    W = 68
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'═'*W}")
    print(f"  PILOTAI — PORTFOLIO SUMMARY  ({now})")
    print(f"{'═'*W}")
    _print_summary(results, W, standalone=True)


def _print_summary(results: list[dict], W: int, standalone: bool = False) -> None:
    ok  = [r for r in results if not r["error"]]
    err = [r for r in results if r["error"] and not r.get("is_legacy")]
    leg = [r for r in results if r["error"] and r.get("is_legacy")]

    tot_equity  = sum(r["equity"]        for r in ok)
    tot_dpnl    = sum(r["day_pnl"]       for r in ok)
    tot_unreal  = sum(r["unrealized_pl"] for r in ok)
    tot_pos     = sum(len(r["positions"]) for r in ok)
    tot_orders  = sum(len(r["open_orders"]) for r in ok)

    print(f"\n{'═'*W}")
    if standalone:
        label_w = 36
        print(f"  {'Experiment':<{label_w}}  {'Equity':>12}  {'Day P&L':>10}  {'Unrealized':>11}  {'Pos':>4}")
        print(f"  {'─'*label_w}  {'─'*12}  {'─'*10}  {'─'*11}  {'─'*4}")
        for r in results:
            lbl = r["label"][:label_w]
            if r["error"]:
                print(f"  {lbl:<{label_w}}  {'ERROR':>12}  {r['error'][:26]}")
            else:
                dp = r["day_pnl"]
                up = r["unrealized_pl"]
                print(f"  {lbl:<{label_w}}  ${r['equity']:>11,.0f}  "
                      f"{_sign(dp)}${dp:>9,.0f}  {_sign(up)}${up:>10,.0f}  "
                      f"{len(r['positions']):>4}")
        print(f"  {'─'*label_w}  {'─'*12}  {'─'*10}  {'─'*11}  {'─'*4}")

    print(f"  Combined Equity      : ${tot_equity:>13,.2f}")
    print(f"  Combined Day P&L     : {_sign(tot_dpnl)}${tot_dpnl:>12,.2f}")
    print(f"  Combined Unrealized  : {_sign(tot_unreal)}${tot_unreal:>12,.2f}")
    print(f"  Open Positions       : {tot_pos}")
    print(f"  Open Orders          : {tot_orders}")
    print(f"  Accounts OK / Total  : {len(ok)} / {len(results)}")
    if err:
        print(f"\n  Errors:")
        for r in err:
            print(f"    ✗  {r['label']}: {r['error']}")
    if leg:
        print(f"\n  Legacy/stale (not counted):")
        for r in leg:
            print(f"    ~  {r['label']} ({r['env_file']}): {r['error']}")
    print(f"{'═'*W}\n")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Check all Alpaca paper trading accounts.")
    parser.add_argument("--json",    action="store_true", help="Output JSON (for automation)")
    parser.add_argument("--summary", action="store_true", help="Summary table only")
    args = parser.parse_args()

    registry = _load_registry()
    results  = [query_account(env, fb, registry, legacy) for env, fb, legacy in ENV_SLOTS]
    results  = deduplicate(results)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    elif args.summary:
        print_summary_only(results)
    else:
        print_full_report(results)

    # Only non-legacy errors count toward exit code
    has_errors = any(r["error"] for r in results if not r.get("is_legacy"))
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
