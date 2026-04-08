"""
check_accounts.py — Programmatic status check for all Alpaca paper trading accounts.

Credentials are loaded from .env.expNNN files in the project root (never hardcoded).
SECURITY AUDIT #1: removed hardcoded API key/secret/account-ID triplets.

Usage:
    python3 scripts/check_accounts.py

Importable:
    from scripts.check_accounts import check_all_accounts
    results = check_all_accounts()
"""

import os
from pathlib import Path

import requests

BASE_URL = "https://paper-api.alpaca.markets"

# Project root is one level above this script.
_PROJECT_ROOT = Path(__file__).parent.parent


def _load_env_file(path: Path) -> dict:
    """Parse a simple KEY=VALUE env file. Ignores comments and blank lines."""
    result = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    except OSError:
        pass
    return result


def _discover_accounts() -> dict:
    """
    Discover all .env.expNNN files in the project root and extract Alpaca credentials.
    Returns a dict keyed by experiment name (e.g. 'exp036').
    """
    accounts = {}
    for env_file in sorted(_PROJECT_ROOT.glob(".env.exp*")):
        exp_name = env_file.name.lstrip(".")  # '.env.exp036' -> 'env.exp036' -> strip further
        # Keep just 'exp036' from '.env.exp036'
        exp_name = env_file.stem  # stem of '.env.exp036' is '.env.exp036' on some Pythons
        # Path.stem strips only one suffix; use name manipulation instead:
        name = env_file.name  # '.env.exp036'
        if name.startswith(".env."):
            exp_name = name[len(".env."):]  # 'exp036'
        else:
            continue

        env = _load_env_file(env_file)
        key = env.get("ALPACA_API_KEY")
        secret = env.get("ALPACA_API_SECRET")
        if key and secret:
            accounts[exp_name] = {"key": key, "secret": secret}

    return accounts


def _headers(key: str, secret: str) -> dict:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _fetch_account(key: str, secret: str) -> dict:
    """Fetch /v2/account. Returns parsed JSON dict or raises on error."""
    resp = requests.get(
        f"{BASE_URL}/v2/account",
        headers=_headers(key, secret),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_positions(key: str, secret: str) -> list:
    """Fetch /v2/positions. Returns list of position dicts or raises on error."""
    resp = requests.get(
        f"{BASE_URL}/v2/positions",
        headers=_headers(key, secret),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def check_all_accounts() -> dict:
    """
    Check all Alpaca paper trading accounts discovered from .env.expNNN files.

    Returns a dict keyed by exp name, e.g.:
        {
          "exp036": {
            "account_id": "PA3D6UPXF5F2",
            "status": "OK",           # or "ERROR"
            "equity": 100000.0,
            "buying_power": 50000.0,
            "positions_count": 3,
            "positions": [...],        # raw list from Alpaca
            "error": None,             # or error message string
          },
          ...
        }
    """
    accounts = _discover_accounts()
    if not accounts:
        print("No .env.exp* files found in project root.")
        return {}

    results = {}

    for exp, creds in accounts.items():
        key = creds["key"]
        secret = creds["secret"]

        entry: dict = {
            "account_id": None,
            "status": "ERROR",
            "equity": None,
            "buying_power": None,
            "positions_count": 0,
            "positions": [],
            "error": None,
        }

        try:
            acct = _fetch_account(key, secret)

            # Alpaca returns an error dict (with "code") for auth failures even on 200
            if "code" in acct:
                entry["error"] = acct.get("message", "API error (code: {})".format(acct["code"]))
                results[exp] = entry
                continue

            entry["account_id"] = acct.get("id", "")
            entry["equity"] = float(acct.get("equity") or 0)
            entry["buying_power"] = float(acct.get("buying_power") or 0)

            positions = _fetch_positions(key, secret)
            entry["positions"] = positions
            entry["positions_count"] = len(positions)
            entry["status"] = "OK"

        except requests.exceptions.HTTPError as exc:
            entry["error"] = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        except requests.exceptions.ConnectionError as exc:
            entry["error"] = f"Connection error: {exc}"
        except requests.exceptions.Timeout:
            entry["error"] = "Request timed out"
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc)

        results[exp] = entry

    return results


def _print_summary(results: dict) -> None:
    print("\n=== PilotAI Portfolio Status ===")
    header = f"{'EXP':<8} {'ACCOUNT':<14} {'STATUS':<8} {'EQUITY':>13} {'BUYING PWR':>13} {'POSITIONS':>9}"
    separator = f"{'----':<8} {'-----------':<14} {'------':<8} {'-----------':>13} {'-----------':>13} {'---------':>9}"
    print(header)
    print(separator)

    has_error = False
    for exp, data in results.items():
        exp_short = exp.replace("exp", "")
        account_id = data.get("account_id") or "unknown"
        status = data["status"]

        if status == "OK":
            equity_str = "${:,.2f}".format(data["equity"])
            bp_str = "${:,.2f}".format(data["buying_power"])
            pos_str = str(data["positions_count"])
        else:
            equity_str = data.get("error") or "error"
            # Truncate long error messages for table display
            if len(equity_str) > 13:
                equity_str = equity_str[:10] + "..."
            bp_str = "-"
            pos_str = "-"
            has_error = True

        print(
            f"{exp_short:<8} {account_id:<14} {status:<8} {equity_str:>13} {bp_str:>13} {pos_str:>9}"
        )

    print()

    if has_error:
        print("WARNING: One or more accounts reported errors.")
        for exp, data in results.items():
            if data["status"] == "ERROR":
                print(f"  {exp} ({data.get('account_id')}): {data.get('error')}")
        print()


if __name__ == "__main__":
    results = check_all_accounts()
    _print_summary(results)
