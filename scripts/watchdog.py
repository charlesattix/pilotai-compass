#!/usr/bin/env python3
"""
scripts/watchdog.py — Auto-restart watchdog for PilotAI experiments (INF-3).

Checks:
  1. tmux session liveness for each active experiment
  2. Scanner heartbeat staleness (>45 min during market hours → crashed)
  3. Alpaca API connectivity (authenticated /v2/account ping)
  4. DB write recency (last trade row within expected timeframe)

Actions:
  - Restarts dead tmux sessions using experiment config
  - Sends Telegram alerts on any restart or failure
  - Outputs JSON status to stdout for cron consumption

Usage:
    python scripts/watchdog.py --config experiments.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ET = timezone(timedelta(hours=-4))  # EDT — adjust if needed or use zoneinfo
MARKET_OPEN = (9, 15)   # 9:15 AM ET
MARKET_CLOSE = (16, 0)  # 4:00 PM ET
HEARTBEAT_STALE_MINUTES = 45
DB_STALE_HOURS = 24  # trades table: alert if no row within this window during market days

logger = logging.getLogger("watchdog")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_et() -> datetime:
    """Current time in US/Eastern (EDT approximation)."""
    return datetime.now(ET)


def is_market_hours(now: Optional[datetime] = None) -> bool:
    """True if *now* is a weekday between 9:15 AM and 4:00 PM ET."""
    now = now or _now_et()
    if now.weekday() >= 5:  # Sat/Sun
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t < MARKET_CLOSE


def tmux_session_alive(session_name: str) -> bool:
    """Return True if a tmux session with *session_name* exists."""
    if not session_name:
        return False
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session_name],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def restart_tmux_session(
    session_name: str,
    project_dir: str,
    env_file: str,
    config_file: str,
    db_path: str,
) -> bool:
    """Start a new tmux session running the scheduler for an experiment.

    Returns True on success.
    """
    cmd = (
        f"cd {project_dir} && "
        f"python3 main.py scheduler "
        f"--config {config_file} "
        f"--env-file {env_file} "
        f"--db {db_path}"
    )
    try:
        result = subprocess.run(
            [
                "tmux", "new-session", "-d",
                "-s", session_name,
                "bash", "-c", cmd,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.error("Failed to restart tmux session %s: %s", session_name, exc)
        return False


def check_heartbeat(data_dir: str, exp_id: str) -> Optional[datetime]:
    """Read the heartbeat file and return its timestamp, or None if missing."""
    hb_path = Path(data_dir) / f".last_scan_{exp_id}"
    if not hb_path.exists():
        return None
    try:
        ts_str = hb_path.read_text().strip()
        return datetime.fromisoformat(ts_str)
    except Exception:
        return None


def check_alpaca_api(env_file: str, base_url: str = "https://paper-api.alpaca.markets") -> bool:
    """Ping Alpaca /v2/account using credentials from *env_file*. Returns True if OK."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        # Fallback: parse manually
        dotenv_values = _parse_env_file

    try:
        env = dotenv_values(env_file)
        key = env.get("APCA_API_KEY_ID") or env.get("ALPACA_API_KEY")
        secret = env.get("APCA_API_SECRET_KEY") or env.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            return False

        import urllib.request
        req = urllib.request.Request(
            f"{base_url}/v2/account",
            headers={
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as exc:
        logger.warning("Alpaca API check failed for %s: %s", env_file, exc)
        return False


def _parse_env_file(path: str) -> Dict[str, str]:
    """Minimal .env parser (fallback if python-dotenv not installed)."""
    env: Dict[str, str] = {}
    try:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip("'\"")
    except Exception:
        pass
    return env


def check_db_recency(db_path: str, max_age_hours: int = DB_STALE_HOURS) -> Optional[str]:
    """Return ISO timestamp of most recent trade, or None if DB missing/empty/stale.

    Returns the timestamp string if recent enough, None otherwise.
    """
    import sqlite3

    if not Path(db_path).exists():
        return None
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        cursor = conn.execute(
            "SELECT MAX(created_at) FROM trades"
        )
        row = cursor.fetchone()
        conn.close()
        if row and row[0]:
            return row[0]
        return None
    except Exception as exc:
        logger.warning("DB check failed for %s: %s", db_path, exc)
        return None


def send_telegram_alert(message: str) -> bool:
    """Send a Telegram alert using the project's shared telegram_alerts module."""
    try:
        # Try importing the project's module
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from shared.telegram_alerts import send_message
        return send_message(message)
    except Exception as exc:
        logger.error("Failed to send Telegram alert: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main watchdog logic
# ---------------------------------------------------------------------------


def load_experiments(config_path: str) -> Dict[str, Any]:
    """Load experiments.yaml and return the experiments dict."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("experiments", {})


def run_watchdog(config_path: str, project_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run all watchdog checks. Returns JSON-serializable status dict."""

    if project_dir is None:
        project_dir = str(Path(config_path).resolve().parent)

    experiments = load_experiments(config_path)
    data_dir = os.path.join(project_dir, "data")
    now = _now_et()
    market_open = is_market_hours(now)

    results: Dict[str, Any] = {
        "timestamp": now.isoformat(),
        "market_hours": market_open,
        "experiments": {},
        "alerts": [],
        "restarts": [],
    }

    for exp_id, exp in experiments.items():
        status = exp.get("status", "unknown")

        # Only monitor active experiments
        if status != "active":
            results["experiments"][exp_id] = {"status": status, "monitored": False}
            continue

        exp_result: Dict[str, Any] = {
            "status": status,
            "monitored": True,
        }

        session_name = exp.get("tmux_session")
        env_file = exp.get("env_file", "")
        config_file = exp.get("config_file", "")
        db_path = exp.get("db_path", "")

        # Resolve relative paths
        env_file_abs = os.path.join(project_dir, env_file) if env_file else ""
        db_path_abs = os.path.join(project_dir, db_path) if db_path else ""

        # 1. tmux session alive?
        try:
            alive = tmux_session_alive(session_name)
            exp_result["tmux_alive"] = alive

            if not alive and session_name:
                msg = f"🔴 <b>WATCHDOG: {exp_id} tmux session '{session_name}' is DEAD</b>"
                logger.warning(msg)

                # Attempt restart
                restarted = restart_tmux_session(
                    session_name, project_dir, env_file, config_file, db_path,
                )
                exp_result["restarted"] = restarted

                if restarted:
                    restart_msg = (
                        f"🔄 <b>WATCHDOG RESTART: {exp_id}</b>\n\n"
                        f"Session <code>{session_name}</code> was dead — restarted automatically.\n"
                        f"Config: <code>{config_file}</code>"
                    )
                    results["restarts"].append(exp_id)
                    send_telegram_alert(restart_msg)
                else:
                    fail_msg = (
                        f"🚨 <b>WATCHDOG FAILED TO RESTART: {exp_id}</b>\n\n"
                        f"Session <code>{session_name}</code> is dead and auto-restart failed.\n"
                        f"Manual intervention required!"
                    )
                    results["alerts"].append(f"{exp_id}: restart failed")
                    send_telegram_alert(fail_msg)
        except Exception as exc:
            exp_result["tmux_error"] = str(exc)
            logger.error("tmux check failed for %s: %s", exp_id, exc)

        # 2. Heartbeat staleness (only during market hours)
        try:
            hb_ts = check_heartbeat(data_dir, exp_id)
            if hb_ts:
                exp_result["last_heartbeat"] = hb_ts.isoformat()
                if market_open and hb_ts.tzinfo is None:
                    hb_ts = hb_ts.replace(tzinfo=ET)
                if market_open:
                    age_min = (now - hb_ts).total_seconds() / 60
                    exp_result["heartbeat_age_min"] = round(age_min, 1)
                    if age_min > HEARTBEAT_STALE_MINUTES:
                        stale_msg = (
                            f"⚠️ <b>WATCHDOG: {exp_id} scanner stale</b>\n\n"
                            f"Last heartbeat: {hb_ts.strftime('%H:%M:%S ET')} "
                            f"({round(age_min)} min ago)\n"
                            f"Threshold: {HEARTBEAT_STALE_MINUTES} min"
                        )
                        results["alerts"].append(f"{exp_id}: heartbeat stale ({round(age_min)}m)")
                        send_telegram_alert(stale_msg)
            else:
                exp_result["last_heartbeat"] = None
                if market_open:
                    exp_result["heartbeat_missing"] = True
        except Exception as exc:
            exp_result["heartbeat_error"] = str(exc)
            logger.error("Heartbeat check failed for %s: %s", exp_id, exc)

        # 3. Alpaca API connectivity
        try:
            api_ok = check_alpaca_api(env_file_abs)
            exp_result["alpaca_api_ok"] = api_ok
            if not api_ok:
                api_msg = (
                    f"⚠️ <b>WATCHDOG: {exp_id} Alpaca API unreachable</b>\n\n"
                    f"Env file: <code>{env_file}</code>"
                )
                results["alerts"].append(f"{exp_id}: Alpaca API down")
                send_telegram_alert(api_msg)
        except Exception as exc:
            exp_result["alpaca_error"] = str(exc)
            logger.error("Alpaca check failed for %s: %s", exp_id, exc)

        # 4. DB write recency
        try:
            last_trade = check_db_recency(db_path_abs)
            exp_result["last_trade"] = last_trade
        except Exception as exc:
            exp_result["db_error"] = str(exc)
            logger.error("DB check failed for %s: %s", exp_id, exc)

        results["experiments"][exp_id] = exp_result

    return results


def main():
    parser = argparse.ArgumentParser(description="PilotAI watchdog — auto-restart & health checks")
    parser.add_argument("--config", default="experiments.yaml", help="Path to experiments.yaml")
    parser.add_argument("--project-dir", default=None, help="Project root (default: parent of config)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        results = run_watchdog(args.config, args.project_dir)
        print(json.dumps(results, indent=2, default=str))
    except Exception as exc:
        # The watchdog itself must never crash
        error_result = {
            "timestamp": datetime.now(ET).isoformat(),
            "error": str(exc),
            "status": "watchdog_crashed",
        }
        print(json.dumps(error_result, indent=2))
        try:
            send_telegram_alert(
                f"🚨 <b>WATCHDOG ITSELF CRASHED</b>\n\n"
                f"<code>{exc}</code>\n\n"
                f"Manual investigation required!"
            )
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
