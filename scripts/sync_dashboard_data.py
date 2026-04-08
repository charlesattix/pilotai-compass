#!/usr/bin/env python3
"""
sync_dashboard_data.py — Dashboard Data Sync
=============================================
Reads all live experiment SQLite DBs on this Mac, exports to
data/dashboard_export.json, and (optionally) pushes to Railway.

Usage:
    # Export locally only
    python scripts/sync_dashboard_data.py

    # Export + push to Railway
    python scripts/sync_dashboard_data.py --push

    # Export + push with explicit URL / token
    python scripts/sync_dashboard_data.py --push \\
        --railway-url https://attix-credit-spreads-production.up.railway.app \\
        --token $RAILWAY_ADMIN_TOKEN

    # Dry run (print JSON, no writes)
    python scripts/sync_dashboard_data.py --dry-run

Environment variables (read from .env.sync if present):
    RAILWAY_URL          — Railway app base URL
    RAILWAY_ADMIN_TOKEN  — Bearer token (same as API_AUTH_TOKEN on Railway)

Cron example (every 5 min, market hours):
    */5 9-16 * * 1-5 /path/to/sync_dashboard_data.sh
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

PROJECT_ROOT   = Path(__file__).parent.parent
REGISTRY_PATH  = PROJECT_ROOT / "experiments" / "registry.json"
OUTPUT_PATH    = PROJECT_ROOT / "data" / "dashboard_export.json"
SCHEMA_VERSION = "1.2"

STARTING_EQUITY    = 100_000.0
ALPACA_PAPER_URL   = "https://paper-api.alpaca.markets"

# Backtest expectations (from MASTERPLAN.md / registry notes)
BACKTEST_EXPECTATIONS = {
    "EXP-400": {"avg_return": 32.7,  "max_dd": -12.1, "robust": 0.870},
    "EXP-401": {"avg_return": 40.7,  "max_dd": -7.0,  "robust": None},
    "EXP-503": {"avg_return": None,   "max_dd": None,  "robust": None},
    "EXP-600": {"avg_return": 139.2, "max_dd": -19.4, "robust": 0.950},
}


# ---------------------------------------------------------------------------
# Alpaca helpers
# ---------------------------------------------------------------------------

def _load_exp_env(exp: dict) -> dict:
    """Parse ALPACA_API_KEY / ALPACA_API_SECRET from .env.expNNN file."""
    num = exp["id"].replace("EXP-", "").lower()      # "EXP-400" → "400"
    env_path = PROJECT_ROOT / f".env.exp{num}"
    result: dict = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def _fetch_alpaca_account(api_key: str, api_secret: str) -> dict:
    """
    Fetch live account equity and open positions from Alpaca paper trading API.
    Returns a dict with equity, unrealized_pl, portfolio_value, positions, etc.
    """
    import urllib.request
    import urllib.error

    headers = {
        "APCA-API-KEY-ID":     api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }

    result: dict = {
        "equity":          None,
        "last_equity":     None,
        "unrealized_pl":   None,
        "portfolio_value": None,
        "cash":            None,
        "buying_power":    None,
        "day_pl":          None,
        "positions":       [],
        "error":           None,
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
    }

    # ── Account ────────────────────────────────────────────────────────────
    try:
        req = urllib.request.Request(
            f"{ALPACA_PAPER_URL}/v2/account",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            acct = json.loads(resp.read().decode("utf-8"))
        result["equity"]          = float(acct.get("equity") or 0)
        result["last_equity"]     = float(acct.get("last_equity") or 0)
        result["unrealized_pl"]   = float(acct.get("unrealized_pl") or 0)
        result["portfolio_value"] = float(acct.get("portfolio_value") or 0)
        result["cash"]            = float(acct.get("cash") or 0)
        result["buying_power"]    = float(acct.get("buying_power") or 0)
        # Day P&L = equity - last_equity
        if result["equity"] is not None and result["last_equity"] is not None:
            result["day_pl"] = round(result["equity"] - result["last_equity"], 2)
    except Exception as e:
        result["error"] = f"account: {e}"
        return result

    # ── Positions ──────────────────────────────────────────────────────────
    try:
        req = urllib.request.Request(
            f"{ALPACA_PAPER_URL}/v2/positions",
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw_positions = json.loads(resp.read().decode("utf-8"))
        if isinstance(raw_positions, list):
            result["positions"] = [
                {
                    "symbol":           p.get("symbol"),
                    "qty":              float(p.get("qty") or 0),
                    "market_value":     float(p.get("market_value") or 0),
                    "cost_basis":       float(p.get("cost_basis") or 0),
                    "unrealized_pl":    float(p.get("unrealized_pl") or 0),
                    "unrealized_plpc":  round(float(p.get("unrealized_plpc") or 0) * 100, 4),
                    "current_price":    float(p.get("current_price") or 0),
                    "avg_entry_price":  float(p.get("avg_entry_price") or 0),
                    "side":             p.get("side"),
                }
                for p in raw_positions
            ]
            # Alpaca account-level unrealized_pl is often 0 for options;
            # compute from individual positions as a reliable fallback.
            pos_unrealized = sum(float(p.get("unrealized_pl") or 0) for p in raw_positions)
            if result["unrealized_pl"] == 0 and pos_unrealized != 0:
                result["unrealized_pl"] = round(pos_unrealized, 2)
    except Exception as e:
        result["error"] = f"positions: {e}"

    return result


# ---------------------------------------------------------------------------
# DB path resolution (mirrors paper_trading_report.py)
# ---------------------------------------------------------------------------

def _resolve_db_path(exp: dict) -> Optional[Path]:
    """
    Try config yaml → data/expNNN/attix_expNNN.db → data/attix_expNNN.db.
    Returns the first path that exists AND has a trades table.
    Falls back to first existing path if none has trades.
    """
    candidates: list[Path] = []

    paper_cfg = exp.get("paper_config")
    if paper_cfg:
        try:
            import yaml
            cfg_file = PROJECT_ROOT / paper_cfg
            if cfg_file.exists():
                with open(cfg_file) as f:
                    cfg = yaml.safe_load(f)
                db_from_yaml = cfg.get("db_path", "")
                if db_from_yaml:
                    candidates.append(PROJECT_ROOT / db_from_yaml)
        except Exception:
            pass

    num = exp["id"].replace("EXP-", "").lower()
    candidates += [
        PROJECT_ROOT / f"data/exp{num}/attix_exp{num}.db",
        PROJECT_ROOT / f"data/attix_exp{num}.db",
    ]

    first_existing: Optional[Path] = None
    for p in candidates:
        if not p.exists():
            continue
        if first_existing is None:
            first_existing = p
        try:
            conn = sqlite3.connect(str(p))
            conn.execute("SELECT 1 FROM trades LIMIT 1")
            conn.close()
            return p
        except sqlite3.OperationalError:
            try:
                conn.close()
            except Exception:
                pass
            continue
        except Exception:
            pass

    return first_existing


# ---------------------------------------------------------------------------
# DB querying
# ---------------------------------------------------------------------------

_CLOSED_STATUSES = (
    "closed_profit", "closed_loss", "closed_manual",
    "closed_expiry", "closed_external",
)


def _week_start(ref: datetime) -> str:
    monday = ref - timedelta(days=ref.weekday())
    return monday.strftime("%Y-%m-%d")


def _query_experiment(exp: dict, report_date: str) -> dict:
    """Extract all stats + trade data from one experiment DB."""
    exp_id    = exp["id"]
    db_path   = _resolve_db_path(exp)
    ref_dt    = datetime.strptime(report_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    week_start = _week_start(ref_dt)

    result: dict = {
        "id":          exp_id,
        "name":        exp.get("name", exp_id),
        "ticker":      exp.get("ticker", "SPY"),
        "creator":     exp.get("created_by", "—"),
        "live_since":  exp.get("live_since", "—"),
        "account_id":  exp.get("account_id", "—"),
        "notes":       exp.get("notes", ""),
        "backtest":    BACKTEST_EXPECTATIONS.get(exp_id, {}),
        "db_path":     str(db_path) if db_path else None,
        "error":       None,
        "alpaca":      None,   # populated below
        "stats": {
            "total_closed":    0,
            "wins":            0,
            "losses":          0,
            "win_rate":        0.0,
            "total_pnl":       0.0,
            "total_return_pct": 0.0,
            "max_dd_pct":      0.0,
            "max_dd_dollars":  0.0,
            "open_count":      0,
            "avg_pnl":         0.0,
            "trades_week":     0,
            "last_trade_date": None,
            "profit_factor":   None,
        },
        "equity_curve":       [],  # [{date, cumulative_pnl, cumulative_pnl_pct}]
        "open_positions":     [],
        "recent_trades":      [],  # last 20 closed
        "strategy_breakdown": {},  # {strategy_type: {count, wins, pnl}}
    }

    if not db_path:
        result["error"] = "Database not found"
        return result

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # ── Closed trades (ordered for equity curve) ───────────────────────
        placeholders = ",".join("?" * len(_CLOSED_STATUSES))
        closed_rows = conn.execute(
            f"SELECT id, pnl, strategy_type, exit_date, entry_date, ticker, "
            f"       short_strike, long_strike, contracts, credit, exit_reason "
            f"FROM trades WHERE status IN ({placeholders}) ORDER BY exit_date ASC",
            _CLOSED_STATUSES,
        ).fetchall()

        pnls    = [float(r["pnl"] or 0) for r in closed_rows]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        win_rate  = (len(wins) / len(pnls) * 100) if pnls else 0.0
        avg_pnl   = (total_pnl / len(pnls)) if pnls else 0.0

        # Profit factor
        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

        # Max drawdown + equity curve
        equity_curve: list[dict] = []
        cumulative   = 0.0
        peak         = 0.0
        max_dd_d     = 0.0
        for r in closed_rows:
            p = float(r["pnl"] or 0)
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd_d:
                max_dd_d = dd
            date_str = str(r["exit_date"] or "")[:10]
            equity_curve.append({
                "date":              date_str,
                "cumulative_pnl":    round(cumulative, 2),
                "cumulative_pnl_pct": round(cumulative / STARTING_EQUITY * 100, 4),
            })
        max_dd_pct = max_dd_d / STARTING_EQUITY * 100

        # Trades this week
        trades_week = sum(
            1 for r in closed_rows
            if str(r["exit_date"] or "")[:10] >= week_start
        )
        last_trade = str(closed_rows[-1]["exit_date"] or "")[:10] if closed_rows else None

        # Strategy breakdown
        breakdown: dict[str, dict] = {}
        for r in closed_rows:
            st  = (r["strategy_type"] or "unknown")
            p   = float(r["pnl"] or 0)
            if st not in breakdown:
                breakdown[st] = {"count": 0, "wins": 0, "pnl": 0.0}
            breakdown[st]["count"] += 1
            if p > 0:
                breakdown[st]["wins"] += 1
            breakdown[st]["pnl"] = round(breakdown[st]["pnl"] + p, 2)
        for st in breakdown:
            bd = breakdown[st]
            bd["win_rate"] = round(bd["wins"] / bd["count"] * 100, 1) if bd["count"] else 0.0

        # Recent closed trades (last 20, newest first)
        recent_closed = conn.execute(
            f"SELECT id, pnl, strategy_type, exit_date, entry_date, ticker, "
            f"       short_strike, long_strike, contracts, credit, exit_reason, status "
            f"FROM trades WHERE status IN ({placeholders}) "
            f"ORDER BY exit_date DESC LIMIT 20",
            _CLOSED_STATUSES,
        ).fetchall()

        recent_trades = []
        for r in recent_closed:
            recent_trades.append({
                "id":            r["id"],
                "ticker":        r["ticker"],
                "strategy_type": r["strategy_type"],
                "entry_date":    str(r["entry_date"] or "")[:10],
                "exit_date":     str(r["exit_date"] or "")[:10],
                "short_strike":  r["short_strike"],
                "long_strike":   r["long_strike"],
                "contracts":     r["contracts"],
                "credit":        r["credit"],
                "pnl":           round(float(r["pnl"] or 0), 2),
                "exit_reason":   r["exit_reason"],
            })

        # Open positions
        open_rows = conn.execute(
            "SELECT id, ticker, strategy_type, entry_date, expiration, "
            "       short_strike, long_strike, contracts, credit, metadata "
            "FROM trades WHERE status = 'open'"
        ).fetchall()

        open_positions = []
        for r in open_rows:
            open_positions.append({
                "id":            r["id"],
                "ticker":        r["ticker"],
                "strategy_type": r["strategy_type"],
                "entry_date":    str(r["entry_date"] or "")[:10],
                "expiration":    str(r["expiration"] or "")[:10],
                "short_strike":  r["short_strike"],
                "long_strike":   r["long_strike"],
                "contracts":     r["contracts"],
                "credit":        r["credit"],
            })

        conn.close()

        result["stats"] = {
            "total_closed":     len(pnls),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate":         round(win_rate, 2),
            "total_pnl":        round(total_pnl, 2),
            "total_return_pct": round(total_pnl / STARTING_EQUITY * 100, 4),
            "max_dd_pct":       round(max_dd_pct, 2),
            "max_dd_dollars":   round(max_dd_d, 2),
            "open_count":       len(open_positions),
            "avg_pnl":          round(avg_pnl, 2),
            "trades_week":      trades_week,
            "last_trade_date":  last_trade,
            "profit_factor":    round(profit_factor, 3) if profit_factor is not None else None,
        }
        result["equity_curve"]       = equity_curve
        result["open_positions"]     = open_positions
        result["recent_trades"]      = recent_trades
        result["strategy_breakdown"] = breakdown

    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            result["error"] = "No trades yet"
        else:
            result["error"] = str(e)
    except Exception as e:
        result["error"] = str(e)

    # ── Alpaca live data ────────────────────────────────────────────────────
    env_vars = _load_exp_env(exp)
    api_key    = env_vars.get("ALPACA_API_KEY", "")
    api_secret = env_vars.get("ALPACA_API_SECRET", "")
    if api_key and api_secret:
        result["alpaca"] = _fetch_alpaca_account(api_key, api_secret)
    else:
        result["alpaca"] = {
            "error":      "No Alpaca credentials found",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    return result


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def build_export(report_date: str) -> dict:
    """Read registry + all experiment DBs, build the full export payload."""
    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    live_exps = [
        exp for exp in registry["experiments"].values()
        if exp.get("status") == "paper_trading"
    ]
    live_exps.sort(key=lambda e: e["id"])

    experiments = []
    for exp in live_exps:
        stats = _query_experiment(exp, report_date)
        experiments.append(stats)

    now = datetime.now(timezone.utc)

    return {
        "schema_version":  SCHEMA_VERSION,
        "generated_at":    now.isoformat(),
        "generated_epoch": int(now.timestamp()),
        "report_date":     report_date,
        "starting_equity": STARTING_EQUITY,
        "experiments":     experiments,
        "summary": {
            "total_experiments": len(experiments),
            "with_trades":       sum(1 for e in experiments if e["stats"]["total_closed"] > 0),
            "total_open":        sum(e["stats"]["open_count"] for e in experiments),
            "total_closed":      sum(e["stats"]["total_closed"] for e in experiments),
            "combined_pnl":      round(sum(e["stats"]["total_pnl"] for e in experiments), 2),
            "combined_equity":   round(sum(
                e["alpaca"]["equity"] or 0
                for e in experiments
                if e.get("alpaca") and e["alpaca"].get("equity") is not None
            ), 2),
            "combined_unrealized_pl": round(sum(
                e["alpaca"]["unrealized_pl"] or 0
                for e in experiments
                if e.get("alpaca") and e["alpaca"].get("unrealized_pl") is not None
            ), 2),
        },
    }


# ---------------------------------------------------------------------------
# Railway push
# ---------------------------------------------------------------------------

def push_to_railway(payload: dict, railway_url: str, token: str, verbose: bool = True) -> bool:
    """
    POST the export JSON to Railway's /api/admin/push-data endpoint.
    Returns True on success.
    """
    import urllib.request
    import urllib.error

    url = railway_url.rstrip("/") + "/api/admin/push-data"
    body = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "X-API-Key": token,
            "User-Agent":    "attix-sync/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body = resp.read().decode("utf-8")
            if verbose:
                print(f"  Railway: HTTP {resp.status} — {resp_body[:120]}")
            return resp.status == 200
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:200]
        if verbose:
            print(f"  Railway push FAILED: HTTP {e.code} — {err_body}", file=sys.stderr)
        return False
    except Exception as e:
        if verbose:
            print(f"  Railway push FAILED: {e}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# .env.sync loader
# ---------------------------------------------------------------------------

def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export experiment data to JSON and optionally push to Railway.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_PATH),
        help="Local JSON output path (default: data/dashboard_export.json)",
    )
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Report date YYYY-MM-DD (default: today UTC)",
    )
    parser.add_argument(
        "--push", action="store_true",
        help="Push exported JSON to Railway after writing locally",
    )
    parser.add_argument(
        "--railway-url",
        default=None,
        help="Railway base URL (overrides RAILWAY_URL env var)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Railway admin token (overrides RAILWAY_ADMIN_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print JSON to stdout, do not write or push",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    # Load env files (try .env.sync, then .env)
    _load_env_file(PROJECT_ROOT / ".env.sync")
    _load_env_file(PROJECT_ROOT / ".env")

    def log(*msg):
        if not args.quiet:
            print(*msg)

    # ── Build export ────────────────────────────────────────────────────────
    log(f"[sync] Building dashboard export for {args.date}...")

    payload = build_export(args.date)

    exp_count = len(payload["experiments"])
    for exp in payload["experiments"]:
        st = exp["stats"]
        alp = exp.get("alpaca") or {}
        err = exp.get("error") or ""
        alp_err = alp.get("error") or ""
        equity_str = ""
        if alp.get("equity") is not None:
            equity_str = f"  alpaca_equity=${alp['equity']:,.0f}"
            if alp.get("unrealized_pl") is not None:
                equity_str += f"  unreal_pl={alp['unrealized_pl']:+,.0f}"
        elif alp_err:
            equity_str = f"  alpaca=({alp_err})"
        tag = f"({err})" if err else f"closed={st['total_closed']} open={st['open_count']} pnl={st['total_pnl']:+.2f}"
        log(f"  {exp['id']:8s}  {tag}{equity_str}")

    log(f"  Summary: {exp_count} experiments | "
        f"{payload['summary']['total_closed']} closed | "
        f"{payload['summary']['total_open']} open | "
        f"combined PnL {payload['summary']['combined_pnl']:+.2f}")

    # ── Dry run ─────────────────────────────────────────────────────────────
    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    # ── Write locally ────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"[sync] Written to {out_path}  ({out_path.stat().st_size:,} bytes)")

    # ── Push to Railway ──────────────────────────────────────────────────────
    if args.push:
        railway_url = args.railway_url or os.environ.get("RAILWAY_URL", "")
        token       = args.token       or os.environ.get("RAILWAY_ADMIN_TOKEN", "")

        if not railway_url:
            print("[sync] ERROR: --railway-url or RAILWAY_URL env var required for --push",
                  file=sys.stderr)
            return 1
        if not token:
            print("[sync] ERROR: --token or RAILWAY_ADMIN_TOKEN env var required for --push",
                  file=sys.stderr)
            return 1

        log(f"[sync] Pushing to {railway_url}...")
        ok = push_to_railway(payload, railway_url, token, verbose=not args.quiet)
        if not ok:
            return 2
        log("[sync] Push complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
