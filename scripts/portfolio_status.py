#!/usr/bin/env python3
"""
portfolio_status.py — PilotAI Paper Trading Dashboard

Reads experiments.yaml registry, checks tmux session state, pings each
Alpaca paper account, and prints a live status dashboard.

Usage:
    python3 scripts/portfolio_status.py              # all experiments
    python3 scripts/portfolio_status.py exp400       # single experiment
    python3 scripts/portfolio_status.py --no-color   # plain output (for logs)
    python3 scripts/portfolio_status.py --active     # active only

Requirements: PyYAML (pip install pyyaml) — already in project dependencies
"""

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Union

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML not installed. Run: pip install pyyaml")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "experiments.yaml"
ALPACA_BASE = "https://paper-api.alpaca.markets"
TIMEOUT = 15


# ── ANSI colors ────────────────────────────────────────────────────────────────
class C:
    BOLD  = "\033[1m"
    GREEN = "\033[0;32m"
    RED   = "\033[0;31m"
    YEL   = "\033[1;33m"
    CYAN  = "\033[0;36m"
    DIM   = "\033[2m"
    NC    = "\033[0m"


def disable_color() -> None:
    for attr in list(vars(C)):
        if not attr.startswith("_"):
            setattr(C, attr, "")


# ── Registry ───────────────────────────────────────────────────────────────────
def load_registry() -> Dict[str, dict]:
    if not REGISTRY_PATH.exists():
        sys.exit(f"ERROR: Registry not found: {REGISTRY_PATH}\n"
                 f"       Create experiments.yaml at the project root.")
    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    experiments = data.get("experiments")
    if not experiments:
        sys.exit("ERROR: experiments.yaml has no 'experiments' key.")
    return experiments


# ── Env file parsing ───────────────────────────────────────────────────────────
def read_env_file(env_path: Path) -> Dict[str, str]:
    """Parse a KEY=value env file. Skips comments and blank lines."""
    result: dict[str, str] = {}
    if not env_path.exists():
        return result
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result


# ── tmux ──────────────────────────────────────────────────────────────────────
def tmux_running(session: Optional[str]) -> bool:
    if not session:
        return False
    try:
        r = subprocess.run(
            ["tmux", "has-session", "-t", session],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Alpaca API ─────────────────────────────────────────────────────────────────
def alpaca_get(endpoint: str, key: str, secret: str) -> Optional[Union[dict, list]]:
    url = f"{ALPACA_BASE}{endpoint}"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # Surface auth failures clearly
        if e.code in (401, 403):
            return {"_error": f"Auth failed (HTTP {e.code})"}
        return {"_error": f"HTTP {e.code}"}
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None


# ── Formatting helpers ─────────────────────────────────────────────────────────
def days_since(date_str: str) -> str:
    try:
        d = date.fromisoformat(date_str)
        n = (date.today() - d).days
        if n == 0:
            return "started today"
        elif n == 1:
            return "1 day ago"
        else:
            return f"{n} days ago"
    except (ValueError, TypeError):
        return date_str or "unknown start date"


def pnl_color(v: float) -> str:
    return C.GREEN if v >= 0 else C.RED


def fmt_upl(v: float) -> str:
    sign = "+" if v >= 0 else ""
    color = pnl_color(v)
    return f"{color}{sign}${v:,.2f}{C.NC}"


# ── Status label for an experiment ────────────────────────────────────────────
def state_label(cfg: dict, is_running: bool) -> str:
    status = cfg.get("status", "unknown")
    tmux = cfg.get("tmux_session") or ""
    if status == "active" and is_running:
        return f"{C.GREEN}{C.BOLD}RUNNING{C.NC} (tmux:{tmux})"
    elif status == "active" and not is_running:
        return f"{C.RED}{C.BOLD}DOWN{C.NC} {C.DIM}(registry=active but tmux:{tmux!r} not found){C.NC}"
    elif status == "stopped" and is_running:
        return f"{C.YEL}{C.BOLD}RUNNING{C.NC} {C.DIM}(registry=stopped — update registry){C.NC}"
    elif status == "archived":
        return f"{C.DIM}ARCHIVED{C.NC}"
    else:
        return f"{C.DIM}STOPPED{C.NC}"


# ── Print one experiment block ─────────────────────────────────────────────────
def print_experiment(name: str, cfg: dict) -> bool:
    """Print full status block for one experiment. Returns True if Alpaca OK."""
    env_file = ROOT / cfg.get("env_file", "MISSING")
    is_running = tmux_running(cfg.get("tmux_session"))
    label = state_label(cfg, is_running)

    acct_id = cfg.get("alpaca_account_id", "?")
    start = cfg.get("start_date", "")

    print()
    print(f"{C.BOLD}{'─' * 62}{C.NC}")
    print(f"  {C.BOLD}{C.CYAN}{name}{C.NC}  {label}  {C.DIM}account:{acct_id}{C.NC}")
    print(f"  {C.DIM}Desc:{C.NC}    {cfg.get('description', '')}")
    print(f"  {C.DIM}Config:{C.NC}  {cfg.get('config_file', '?')}")
    print(f"  {C.DIM}Env:{C.NC}     {cfg.get('env_file', '?')}")
    print(f"  {C.DIM}DB:{C.NC}      {cfg.get('db_path', '?')}")
    print(f"  {C.DIM}Start:{C.NC}   {start}  ({days_since(start)})")
    if cfg.get("backtest_ref"):
        print(f"  {C.DIM}Backtest:{C.NC} {cfg['backtest_ref']}")
    if cfg.get("notes"):
        # Wrap long notes
        note = str(cfg["notes"]).strip().replace("\n", " ")
        print(f"  {C.DIM}Notes:{C.NC}   {note}")

    # ── Credentials ──
    creds = read_env_file(env_file)
    api_key = creds.get("ALPACA_API_KEY", "")
    api_secret = creds.get("ALPACA_API_SECRET", "")

    if not api_key or not api_secret:
        print(f"  {C.RED}✗{C.NC} Credentials missing in {cfg.get('env_file','?')}")
        return False

    # ── Alpaca account ──
    acct = alpaca_get("/v2/account", api_key, api_secret)
    if acct is None:
        print(f"  {C.RED}✗{C.NC} Alpaca unreachable (key={api_key[:8]}...)")
        return False
    if isinstance(acct, dict) and "_error" in acct:
        print(f"  {C.RED}✗{C.NC} Alpaca error: {acct['_error']} (key={api_key[:8]}...)")
        return False

    equity  = float(acct.get("equity", 0) or 0)
    cash    = float(acct.get("cash", 0) or 0)
    bp      = float(acct.get("buying_power", 0) or 0)
    upl     = float(acct.get("unrealized_pl", 0) or 0)
    daytpl  = float(acct.get("last_equity", 0) or 0)
    day_pnl = equity - daytpl
    acct_num = acct.get("account_number", acct_id)
    acct_status = acct.get("status", "?")

    starting_capital = 100_000.0
    total_pnl = equity - starting_capital
    total_pct = (total_pnl / starting_capital) * 100

    print(f"  {C.GREEN}✓{C.NC} {acct_num}  status={acct_status}")
    print(f"    Equity:         ${equity:>12,.2f}  "
          f"({C.GREEN if total_pnl >= 0 else C.RED}{'+' if total_pnl >= 0 else ''}"
          f"${total_pnl:,.2f} / {'+' if total_pct >= 0 else ''}{total_pct:.2f}% vs $100K start{C.NC})")
    print(f"    Cash:           ${cash:>12,.2f}")
    print(f"    Buying Power:   ${bp:>12,.2f}")
    print(f"    Unrealized P&L: {fmt_upl(upl)}")
    print(f"    Day P&L:        {fmt_upl(day_pnl)}")

    # ── Positions ──
    positions = alpaca_get("/v2/positions", api_key, api_secret)
    if positions is None:
        print(f"  {C.RED}✗{C.NC} Could not fetch positions")
    elif not positions:
        print(f"    Positions:  {C.DIM}none open{C.NC}")
    else:
        print(f"    Positions ({len(positions)}):")
        for p in positions:
            sym  = p.get("symbol", "?")
            side = p.get("side", "?")
            qty  = p.get("qty", "?")
            mv   = float(p.get("market_value", 0) or 0)
            pupl = float(p.get("unrealized_pl", 0) or 0)
            pc   = pnl_color(pupl)
            sign = "+" if pupl >= 0 else ""
            print(f"      {sym:<32} {side:<5} qty={qty:<6} "
                  f"mv=${mv:>10,.2f}  uPnL={pc}{sign}${pupl:>8,.2f}{C.NC}")

    return True


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(experiments: Dict[str, dict]) -> None:
    """Print one-line-per-experiment summary table."""
    print()
    print(f"{C.BOLD}{'─' * 62}{C.NC}")
    print(f"{C.BOLD}  SUMMARY TABLE{C.NC}")
    print(f"{C.BOLD}{'─' * 62}{C.NC}")
    header = f"  {'Experiment':<12} {'Status':<10} {'Account':<14} {'tmux':<12} {'Config'}"
    print(f"{C.DIM}{header}{C.NC}")
    for name, cfg in experiments.items():
        status = cfg.get("status", "?")
        acct = cfg.get("alpaca_account_id", "?")
        tmux = cfg.get("tmux_session") or "—"
        config = Path(cfg.get("config_file", "?")).name
        is_running = tmux_running(cfg.get("tmux_session"))
        if status == "active" and is_running:
            sc = C.GREEN
        elif status == "active":
            sc = C.RED
        elif status == "stopped":
            sc = C.DIM
        else:
            sc = C.DIM
        print(f"  {sc}{name:<12}{C.NC} {sc}{status:<10}{C.NC} {acct:<14} {tmux:<12} {config}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="PilotAI paper trading portfolio dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment", nargs="?",
        help="Show only this experiment (e.g. exp400). Omit for all.",
    )
    parser.add_argument(
        "--active", action="store_true",
        help="Show only experiments with status=active",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color output",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Show summary table only (no Alpaca API calls)",
    )
    args = parser.parse_args()

    if args.no_color:
        disable_color()

    experiments = load_registry()

    # Filter
    if args.experiment:
        if args.experiment not in experiments:
            print(f"ERROR: '{args.experiment}' not in experiments.yaml.")
            print(f"  Known: {', '.join(experiments)}")
            return 1
        experiments = {args.experiment: experiments[args.experiment]}
    elif args.active:
        experiments = {k: v for k, v in experiments.items()
                       if v.get("status") == "active"}

    # Header
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M %Z")
    print()
    print(f"{C.BOLD}╔══════════════════════════════════════════════════════════╗{C.NC}")
    print(f"{C.BOLD}║  PilotAI Portfolio Dashboard  —  {now:<25}║{C.NC}")
    print(f"{C.BOLD}╚══════════════════════════════════════════════════════════╝{C.NC}")
    print(f"  Registry: {REGISTRY_PATH}")

    if args.summary:
        print_summary(experiments)
        return 0

    # Sort: active first, then stopped, then archived
    order = {"active": 0, "stopped": 1, "archived": 2}
    sorted_exps = sorted(
        list(experiments.items()),
        key=lambda kv: (order.get(kv[1].get("status", ""), 9), kv[0]),
    )

    failed = 0
    for name, cfg in sorted_exps:
        ok = print_experiment(name, cfg)
        if not ok:
            failed += 1

    print()
    print(f"{C.BOLD}{'─' * 62}{C.NC}")
    if failed == 0:
        print(f"  {C.GREEN}{C.BOLD}All accounts reachable.{C.NC}")
    else:
        print(f"  {C.RED}{C.BOLD}{failed} account(s) failed — check credentials/connectivity.{C.NC}")

    print_summary(experiments)
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
