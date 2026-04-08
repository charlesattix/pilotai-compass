#!/usr/bin/env python3
"""
orphan_check.py — Detect orphan positions in paper trading accounts.

An orphan is defined as: an Alpaca account with open option positions
but NO corresponding running tmux session.

This can happen when:
  - A tmux session crashes or is killed while positions are open
  - Someone kills the process without closing positions first
  - A paper config fails to start after a machine restart

Usage:
    python3 scripts/orphan_check.py           # check all active experiments
    python3 scripts/orphan_check.py --all     # include stopped/archived too
    python3 scripts/orphan_check.py --no-color

Exit codes:
    0  — all active accounts healthy (tmux running OR no positions)
    1  — at least one orphan found (positions without a running process)
"""

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    sys.exit("ERROR: PyYAML not installed. Run: pip install pyyaml")

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "experiments.yaml"
ALPACA_BASE = "https://paper-api.alpaca.markets"
TIMEOUT = 15


# ── Color ──────────────────────────────────────────────────────────────────────
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
        sys.exit(f"ERROR: Registry not found: {REGISTRY_PATH}")
    with open(REGISTRY_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("experiments", {})


# ── Env file ───────────────────────────────────────────────────────────────────
def read_env(env_path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
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


# ── Alpaca ─────────────────────────────────────────────────────────────────────
def alpaca_get(endpoint: str, key: str, secret: str) -> Optional[object]:
    url = f"{ALPACA_BASE}{endpoint}"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    })
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}"}
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
        return {"_error": str(e)}


def count_option_positions(positions) -> int:
    """Count positions whose symbol looks like an OCC option symbol."""
    if not isinstance(positions, list):
        return 0
    count = 0
    for p in positions:
        sym = p.get("symbol", "")
        # OCC symbols are long: e.g. SPY260417C00580000
        if len(sym) > 10 or p.get("asset_class", "").lower() == "us_option":
            count += 1
    return count


# ── Per-experiment check ────────────────────────────────────────────────────────
class CheckResult:
    def __init__(self, name: str):
        self.name = name
        self.status = ""           # registry status
        self.tmux_session = ""
        self.tmux_alive = False
        self.account_id = ""
        self.equity = 0.0
        self.option_positions = 0  # count of open option positions
        self.api_reachable = False
        self.api_error = ""
        self.is_orphan = False
        self.is_healthy = False
        self.verdict = ""          # HEALTHY | ORPHAN | DOWN | OFFLINE | ARCHIVED


def check_experiment(name: str, cfg: dict) -> CheckResult:
    result = CheckResult(name)
    result.status = cfg.get("status", "unknown")
    result.tmux_session = cfg.get("tmux_session") or ""
    result.account_id = cfg.get("alpaca_account_id", "?")

    result.tmux_alive = tmux_running(result.tmux_session)

    # Load credentials
    env_file = ROOT / cfg.get("env_file", "")
    creds = read_env(env_file)
    api_key    = creds.get("ALPACA_API_KEY", "")
    api_secret = creds.get("ALPACA_API_SECRET", "")

    if not api_key or not api_secret:
        result.api_error = f"credentials missing in {cfg.get('env_file','?')}"
        result.verdict = "OFFLINE"
        return result

    # Fetch account
    acct = alpaca_get("/v2/account", api_key, api_secret)
    if isinstance(acct, dict) and "_error" in acct:
        result.api_error = acct["_error"]
        result.verdict = "OFFLINE"
        return result

    result.api_reachable = True
    result.equity = float(acct.get("equity", 0) or 0)

    # Fetch positions
    positions = alpaca_get("/v2/positions", api_key, api_secret)
    if isinstance(positions, dict) and "_error" in positions:
        result.api_error = positions["_error"]
    else:
        result.option_positions = count_option_positions(positions)

    # ── Classify ──
    if result.status in ("archived", "stopped"):
        if result.option_positions > 0:
            result.is_orphan = True
            result.verdict = "ORPHAN"  # stopped/archived but still has positions
        else:
            result.verdict = "ARCHIVED" if result.status == "archived" else "STOPPED_CLEAN"
        return result

    # Active experiment
    if result.tmux_alive and result.option_positions >= 0:
        result.is_healthy = True
        result.verdict = "HEALTHY"
    elif not result.tmux_alive and result.option_positions > 0:
        result.is_orphan = True
        result.verdict = "ORPHAN"   # process died, positions still open
    elif not result.tmux_alive and result.option_positions == 0:
        result.verdict = "DOWN"     # process not running, no positions (safe but registry wrong)
    else:
        result.verdict = "HEALTHY"

    return result


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect orphan positions in Alpaca paper accounts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Check ALL experiments including stopped and archived",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI color output",
    )
    args = parser.parse_args()

    if args.no_color:
        disable_color()

    experiments = load_registry()

    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print()
    print(f"{C.BOLD}╔══════════════════════════════════════════════════╗{C.NC}")
    print(f"{C.BOLD}║  Orphan Position Check  —  {now:<22}║{C.NC}")
    print(f"{C.BOLD}╚══════════════════════════════════════════════════╝{C.NC}")
    print()

    results: List[CheckResult] = []

    for name, cfg in experiments.items():
        status = cfg.get("status", "unknown")
        if not args.all and status in ("archived", "stopped"):
            continue

        sys.stdout.write(f"  Checking {name}... ")
        sys.stdout.flush()
        r = check_experiment(name, cfg)
        results.append(r)

        if r.verdict == "HEALTHY":
            label = f"{C.GREEN}HEALTHY{C.NC}"
        elif r.verdict == "ORPHAN":
            label = f"{C.RED}{C.BOLD}ORPHAN ⚠{C.NC}"
        elif r.verdict == "DOWN":
            label = f"{C.YEL}DOWN{C.NC} (no tmux, no positions)"
        elif r.verdict == "OFFLINE":
            label = f"{C.YEL}OFFLINE{C.NC} ({r.api_error})"
        elif r.verdict == "ARCHIVED":
            label = f"{C.DIM}ARCHIVED (clean){C.NC}"
        elif r.verdict == "STOPPED_CLEAN":
            label = f"{C.DIM}STOPPED (clean){C.NC}"
        else:
            label = r.verdict

        print(label)

        if r.verdict in ("HEALTHY", "ORPHAN", "DOWN", "OFFLINE"):
            tmux_str = f"tmux:{r.tmux_session}" if r.tmux_session else "no tmux"
            tmux_color = C.GREEN if r.tmux_alive else C.DIM
            print(f"    {tmux_color}{tmux_str} {'✓ running' if r.tmux_alive else '✗ NOT running'}{C.NC}")
            print(f"    Account: {r.account_id}  Equity: ${r.equity:,.2f}")
            pos_color = C.RED if r.option_positions > 0 and not r.tmux_alive else C.NC
            print(f"    {pos_color}Open option positions: {r.option_positions}{C.NC}")

        if r.is_orphan:
            print(f"    {C.RED}ACTION REQUIRED: Run close_all_positions.py during market hours{C.NC}")
            print(f"      python3 scripts/close_all_positions.py {cfg.get('env_file','?')} --force")

    # ── Summary ──
    orphans  = [r for r in results if r.is_orphan]
    healthy  = [r for r in results if r.verdict == "HEALTHY"]
    down     = [r for r in results if r.verdict == "DOWN"]
    offline  = [r for r in results if r.verdict == "OFFLINE"]

    print()
    print(f"{C.BOLD}{'─' * 52}{C.NC}")
    print(f"{C.BOLD}  SUMMARY{C.NC}")
    print(f"{C.BOLD}{'─' * 52}{C.NC}")
    print(f"  Healthy:  {C.GREEN}{len(healthy)}{C.NC}")
    print(f"  Orphans:  {C.RED if orphans else C.NC}{len(orphans)}{C.NC}"
          + (f"  ← {C.RED}{C.BOLD}ACTION REQUIRED{C.NC}" if orphans else ""))
    if down:
        print(f"  Down:     {C.YEL}{len(down)}{C.NC}  (process not running, no positions)")
    if offline:
        print(f"  Offline:  {C.YEL}{len(offline)}{C.NC}  (API unreachable)")
    print()

    if orphans:
        print(f"  {C.RED}{C.BOLD}ORPHANS DETECTED:{C.NC}")
        for r in orphans:
            print(f"    • {r.name} ({r.account_id}): {r.option_positions} open positions, no running process")
        print()
        print("  Run during market hours (Mon-Fri 9:30-16:00 ET):")
        for r in orphans:
            exp_cfg = experiments.get(r.name, {})
            print(f"    python3 scripts/close_all_positions.py {exp_cfg.get('env_file','?')} --force")
        print()
        return 1

    print(f"  {C.GREEN}{C.BOLD}All active accounts healthy — no orphans.{C.NC}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
