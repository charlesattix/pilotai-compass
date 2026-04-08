# Portfolio Manager Framework — Design Document
**PilotAI Credit Spreads Trading System**  
**Version 1.0**  
**Last Updated:** March 6, 2026  
**Status:** Architecture Design

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [CLI Commands](#3-cli-commands)
4. [Pre-Flight Validation](#4-pre-flight-validation)
5. [Runtime Safety](#5-runtime-safety)
6. [Observability](#6-observability)
7. [Failure Scenarios](#7-failure-scenarios)
8. [File Structure](#8-file-structure)
9. [Implementation Plan](#9-implementation-plan)
10. [Config Schema](#10-config-schema)

---

## 1. EXECUTIVE SUMMARY

### Problem Statement

The PilotAI credit spreads trading system currently operates three live paper-trading portfolios:

- **exp_036** (Alpaca account `PA3QQDNADQU9`) — `.env.exp036`
- **exp_059** (Alpaca account `PA35U64WQBWM`) — `.env` (main)
- **exp_154** (Alpaca account `PA3UNOV58WGK`) — `.env.exp154`

Each portfolio has distinct configuration parameters:

| Portfolio | Risk | SL | Features | Config File |
|-----------|------|-----|----------|-------------|
| exp_036 | 10% | 2.5x | Compound, directional, MA200 filter | `config_exp036.yaml` + `.env.exp036` |
| exp_059 | 10% | 2.5x | ICs enabled, neutral regime | `config.yaml` + `.env` |
| exp_154 | 5% | 3.5x | 12% IC credit floor, ComboRegime v2 | `config_exp154.yaml` + `.env.exp154` |

**Current issues:**

1. **Hand-wired setup**: Each portfolio requires manual shell script management (`run_exp154.sh`), separate terminal sessions, and manual process tracking.
2. **No standardization**: No central registry — configuration scattered across YAML files, env files, and shell scripts.
3. **Audit findings (Feb 2026)**: System is *alert-only* with no automated execution safeguards. No pre-flight validation, no stop-loss enforcement, no circuit breakers, no position monitoring. Paper trading orders submit to Alpaca but positions are never actively managed for profit-taking or stop-loss.
4. **Crash vulnerability**: Process crashes leave orphaned positions with no reconciliation. Database state can diverge from Alpaca reality between scanner runs.
5. **Risk exposure**: No portfolio-level caps on total capital at risk, no max position limits, no duplicate trade prevention.

### Why This Framework Exists

**The Portfolio Manager Framework standardizes, isolates, and hardens multi-portfolio operation.**

Core principles:

- **Single source of truth**: `portfolios.json` registry tracks every portfolio's lifecycle, configuration, and state.
- **Complete isolation**: Each portfolio gets its own `.env`, SQLite database, config file, log directory, and Alpaca paper account — no shared state.
- **Safety-first architecture**: Every portfolio launch is gated by comprehensive pre-flight validation. Runtime includes active position monitoring, stop-loss/profit-target enforcement, and circuit breakers.
- **Production-grade observability**: Per-portfolio logs, health checks, Telegram alerts, and crash recovery.
- **Hedge fund standards**: Treat each portfolio as an independent trading strategy with isolated P&L, risk controls, and audit trails.

### Goal

Transform PilotAI from a research prototype into a robust, production-grade multi-portfolio system capable of managing dozens of concurrent strategies with institutional-level risk management and operational discipline.

---

## 2. ARCHITECTURE OVERVIEW

### 2.1 Portfolio Registry

**Central data structure:** `portfolios/portfolios.json`

```json
{
  "portfolios": [
    {
      "portfolio_id": "exp_036",
      "name": "Experiment 036 — Directional Compound",
      "status": "running",
      "alpaca_account": "PA3QQDNADQU9",
      "created_at": "2026-02-26T15:47:00Z",
      "config_file": "portfolios/exp_036/config.yaml",
      "env_file": "portfolios/exp_036/.env",
      "db_path": "portfolios/exp_036/pilotai.db",
      "log_dir": "portfolios/exp_036/logs",
      "pid": 12345,
      "last_heartbeat": "2026-03-06T16:20:00Z",
      "tags": ["directional", "compound", "ma200"],
      "metadata": {
        "risk_per_trade": 0.10,
        "stop_loss_multiplier": 2.5,
        "regime_mode": "ma200",
        "backtested_sharpe": 2.27,
        "notes": "2025 validation: +15.11%, 217 trades"
      }
    },
    {
      "portfolio_id": "exp_154",
      "name": "Experiment 154 — ComboRegime IC Focus",
      "status": "running",
      "alpaca_account": "PA3UNOV58WGK",
      "created_at": "2026-03-06T04:56:00Z",
      "config_file": "portfolios/exp_154/config.yaml",
      "env_file": "portfolios/exp_154/.env",
      "db_path": "portfolios/exp_154/pilotai.db",
      "log_dir": "portfolios/exp_154/logs",
      "pid": 12346,
      "last_heartbeat": "2026-03-06T16:21:00Z",
      "tags": ["iron_condor", "regime_v2", "conservative"],
      "metadata": {
        "risk_per_trade": 0.05,
        "ic_min_credit_pct": 12,
        "stop_loss_multiplier": 3.5,
        "regime_mode": "combo_v2",
        "backtested_sharpe": null,
        "notes": "Phase 6 regime detector validation"
      }
    }
  ],
  "schema_version": "1.0"
}
```

**Registry operations:**

- **Create**: Add new portfolio entry with validated config
- **Update**: Modify status (running → paused → stopped), record PID, update heartbeat
- **Delete**: Archive portfolio and move data to `portfolios/archive/`
- **Query**: List portfolios by status, tag, or creation date

### 2.2 Isolation Model

**Every portfolio is a completely isolated universe:**

```
portfolios/
├── portfolios.json              # Central registry
├── exp_036/                      # Portfolio-specific directory
│   ├── config.yaml               # Strategy params (DTE, risk, SL, etc.)
│   ├── .env                      # Alpaca API keys, Telegram, Polygon
│   ├── pilotai.db                # SQLite DB (trades, alerts, reconciliation)
│   ├── logs/                     # Timestamped log files
│   │   ├── scanner_2026-03-06.log
│   │   ├── health_check.log
│   │   └── position_monitor.log
│   └── state/                    # Runtime state (PIDs, circuit breaker flags)
│       ├── circuit_breaker.json
│       └── reconciliation_state.json
├── exp_154/
│   ├── config.yaml
│   ├── .env
│   ├── pilotai.db
│   ├── logs/
│   └── state/
└── archive/                      # Deleted portfolios
    └── exp_099_2026-02-28/
```

**Isolation guarantees:**

1. **No shared databases**: Each portfolio writes to its own SQLite file. Trades from exp_036 never mix with exp_154.
2. **No shared .env files**: API keys, secrets, and config variables are scoped per-portfolio.
3. **No shared logs**: Each portfolio logs to its own directory — no interleaved output.
4. **Independent processes**: Each portfolio runs as a separate OS process with its own PID.

**Why isolation matters:**

- **Blast radius containment**: Bug in exp_036 config cannot affect exp_154.
- **Independent rollbacks**: Pause or stop one portfolio without touching others.
- **Clean P&L attribution**: Each portfolio's trades and performance are 100% isolated.
- **Parallel testing**: Run experimental configs alongside production without contamination.

### 2.3 Multi-Portfolio Process Architecture

**High-level process topology:**

```
┌───────────────────────────────────────────────────────────┐
│  Portfolio Manager (CLI)                                   │
│  - Validates configs, checks API keys, enforces policies  │
│  - Spawns & monitors child processes                       │
│  - Aggregates health checks                                │
└───────────────────────────────────────────────────────────┘
           │
           ├──> Scanner Process: exp_036 (PID 12345)
           │    ├── Entry: 14x/day market-hours scan
           │    ├── DB: portfolios/exp_036/pilotai.db
           │    └── Logs: portfolios/exp_036/logs/
           │
           ├──> Scanner Process: exp_154 (PID 12346)
           │    ├── Entry: 14x/day market-hours scan
           │    ├── DB: portfolios/exp_154/pilotai.db
           │    └── Logs: portfolios/exp_154/logs/
           │
           └──> Position Monitor (shared, multi-portfolio)
                ├── Runs every 5 minutes
                ├── Queries all "running" portfolios from registry
                ├── For each: check PT/SL thresholds, enforce DTE exits
                └── Logs: logs/position_monitor.log
```

**Process lifecycle:**

1. **Launch**: `portfolio start exp_154` validates config, checks APIs, spawns `python main.py scheduler --portfolio exp_154`
2. **Heartbeat**: Each scanner writes timestamp to registry every 10 minutes
3. **Monitoring**: Position monitor reads registry, queries Alpaca for each portfolio's positions
4. **Shutdown**: `portfolio stop exp_154` sends SIGTERM, waits for graceful exit, updates registry status

**Process supervision:**

- **systemd** (Linux) or **launchd** (macOS) can auto-restart crashed processes
- **Health check endpoint**: HTTP server on per-portfolio port (8001, 8002, ...) returns status
- **Stale heartbeat detection**: If heartbeat age > 15 minutes, mark portfolio as "degraded"

---

## 3. CLI COMMANDS

**Central command-line interface:** `portfolio` (Python script or symlink to `scripts/portfolio_cli.py`)

### 3.1 `portfolio create`

**Purpose:** Initialize a new portfolio with validated configuration.

**Usage:**
```bash
portfolio create --id exp_160 \
  --name "Risk 5% IC-Only Experiment" \
  --alpaca-account PA3ABCD1234 \
  --config configs/exp_160_risk5_ic.json \
  --env /path/to/.env.exp160 \
  --tags ic_only,risk5,regime_v2
```

**Actions:**
1. Validate `--id` is unique (not in registry)
2. Verify Alpaca account ID via API handshake
3. Parse config file, check required fields (see Config Schema)
4. Run pre-flight validation suite (API keys, DB path, logs writable)
5. Create portfolio directory structure: `portfolios/exp_160/`
6. Copy config and .env to portfolio directory
7. Initialize empty SQLite database
8. Add entry to `portfolios.json`
9. Output: `✅ Portfolio exp_160 created. Launch with: portfolio start exp_160`

**Flags:**
- `--dry-run`: Validate without creating files
- `--force`: Skip confirmation prompts

### 3.2 `portfolio list`

**Purpose:** Display all portfolios with status and metadata.

**Usage:**
```bash
portfolio list                   # All portfolios
portfolio list --status running  # Filter by status
portfolio list --tags ic_only    # Filter by tag
```

**Output:**
```
ID        NAME                              STATUS    ALPACA ACCT     PID     HEARTBEAT
exp_036   Directional Compound              running   PA3QQDNADQU9    12345   2m ago
exp_154   ComboRegime IC Focus              running   PA3UNOV58WGK    12346   1m ago
exp_160   Risk 5% IC-Only Experiment        stopped   PA3ABCD1234     —       —
exp_099   (archived)                        deleted   —               —       —

Legend: ● running | ◐ paused | ○ stopped | ✗ degraded
```

**Flags:**
- `--json`: Output as JSON for scripting
- `--verbose`: Include full metadata (risk params, backtest stats)

### 3.3 `portfolio status`

**Purpose:** Detailed status for a single portfolio.

**Usage:**
```bash
portfolio status exp_154
```

**Output:**
```
Portfolio: exp_154 (ComboRegime IC Focus)
Status: ● RUNNING (healthy)
Alpaca Account: PA3UNOV58WGK
Process PID: 12346
Uptime: 4h 32m
Last Heartbeat: 23 seconds ago

Configuration:
  Risk per trade: 5%
  Stop-loss: 3.5x credit
  Iron condors: Enabled (12% min combined credit)
  Regime mode: combo_v2

Database: portfolios/exp_154/pilotai.db (47 trades, 12 open positions)
Logs: portfolios/exp_154/logs/scanner_2026-03-06.log (3.2 MB)

Circuit Breaker: ARMED (trigger at -40% drawdown)
Current Drawdown: -8.3% (safe)

Recent Activity:
  15:42 — Opened SPY 580/575 bull put (2 contracts, $0.82 credit)
  15:38 — Closed QQQ 520/515 bull put at 50% profit ($164 P&L)
  15:30 — Position monitor checked 12 positions (all healthy)
```

**Flags:**
- `--tail`: Live-tail log output
- `--positions`: Show detailed position table

### 3.4 `portfolio start`

**Purpose:** Launch scanner process for a portfolio.

**Usage:**
```bash
portfolio start exp_154              # Start with existing config
portfolio start exp_154 --validate   # Run pre-flight before launch
```

**Pre-flight validation (see Section 4):**
- API key connectivity test
- SL/PT/CB config present
- Database writable
- No shared .env or DB with other portfolios
- Regime detector config valid
- Position monitoring enabled in config

**Actions:**
1. Check portfolio status (must be "stopped" or "paused")
2. Load config and .env from portfolio directory
3. Run pre-flight validation (BLOCKING if any check fails)
4. Spawn scanner process: `python main.py scheduler --portfolio exp_154`
5. Record PID in registry
6. Set status to "running"
7. Start heartbeat monitoring thread
8. Output: `✅ Portfolio exp_154 started (PID 12346)`

**Flags:**
- `--skip-validation`: Skip pre-flight (DANGEROUS — use only in emergency recovery)
- `--dry-run`: Show what would be launched without starting process

### 3.5 `portfolio pause`

**Purpose:** Temporarily halt new entries without stopping the process.

**Usage:**
```bash
portfolio pause exp_036 --reason "Testing config change"
```

**Actions:**
1. Write pause flag to `portfolios/exp_036/state/pause.flag`
2. Scanner checks for pause flag before every scan — if present, skip opportunity generation
3. Position monitor continues to enforce SL/PT on existing positions
4. Update registry status to "paused"

**Resume:**
```bash
portfolio resume exp_036
```

**Use cases:**
- Testing new config changes without killing process
- Temporarily halting entries during high-impact events (FOMC, earnings)
- Manual risk management intervention

### 3.6 `portfolio stop`

**Purpose:** Gracefully shut down scanner process.

**Usage:**
```bash
portfolio stop exp_154
portfolio stop --all    # Stop all running portfolios
```

**Actions:**
1. Send SIGTERM to PID from registry
2. Wait up to 30 seconds for graceful exit
3. If still running, send SIGKILL
4. Update registry status to "stopped"
5. Clear PID field

**Flags:**
- `--force`: SIGKILL immediately (use if process is hung)

### 3.7 `portfolio validate`

**Purpose:** Run comprehensive pre-flight validation without launching.

**Usage:**
```bash
portfolio validate exp_154
```

**Output:**
```
Validating portfolio: exp_154
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Config file exists and parses correctly
✅ Alpaca API keys valid (account PA3UNOV58WGK accessible)
✅ Polygon API key valid (rate limit: 5 req/min remaining)
✅ Stop-loss config present (3.5x multiplier)
✅ Profit target config present (50% of credit)
✅ Circuit breaker config present (40% drawdown)
✅ Database path writable (portfolios/exp_154/pilotai.db)
✅ Log directory writable (portfolios/exp_154/logs/)
✅ No shared .env detected (unique keys)
✅ No shared database detected (unique DB path)
✅ Regime detector config valid (combo_v2 mode)
✅ Position monitoring enabled (check_interval: 5min)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ All checks passed. Safe to launch.
```

**Exit codes:**
- `0`: All checks passed
- `1`: One or more checks failed (output shows failures)

### 3.8 `portfolio delete`

**Purpose:** Archive a portfolio and remove from active registry.

**Usage:**
```bash
portfolio delete exp_160 --confirm
```

**Actions:**
1. Verify portfolio is stopped (refuse if running)
2. Move `portfolios/exp_160/` to `portfolios/archive/exp_160_2026-03-06/`
3. Remove from `portfolios.json`
4. Preserve all data for audit purposes (never delete SQLite DB or logs)

**Flags:**
- `--purge`: Permanently delete archived data (requires admin confirmation)

---

## 4. PRE-FLIGHT VALIDATION

**Every portfolio launch is BLOCKED until all checks pass.**

### 4.1 Validation Checklist

| # | Check | Description | Blocking? |
|---|-------|-------------|-----------|
| 1 | **Config file exists** | `config.yaml` present and valid YAML syntax | ✅ YES |
| 2 | **Required config fields** | All mandatory params present (see Config Schema) | ✅ YES |
| 3 | **Alpaca API keys valid** | Test connection, verify account number matches registry | ✅ YES |
| 4 | **Alpaca account type** | Must be paper trading account | ✅ YES |
| 5 | **Polygon API key valid** | Test `/v2/aggs/ticker/SPY/range/1/day` endpoint | ✅ YES |
| 6 | **Telegram bot token** | Validate token format, test send (optional if disabled) | ⚠️ WARNING |
| 7 | **Stop-loss config present** | `risk.stop_loss_multiplier` must be set (> 0) | ✅ YES |
| 8 | **Profit target present** | `risk.profit_target` must be set (> 0) | ✅ YES |
| 9 | **Circuit breaker config** | `risk.drawdown_cb_pct` must be set (< 100) | ✅ YES |
| 10 | **Database path unique** | No other portfolio shares this DB path | ✅ YES |
| 11 | **Database writable** | Create test table, write row, delete | ✅ YES |
| 12 | **Shared .env detection** | Check if .env Alpaca keys match any other portfolio | ✅ YES |
| 13 | **Log directory writable** | Create test log file | ✅ YES |
| 14 | **Regime detector config** | If `regime_mode` != `none`, validate parameters | ✅ YES |
| 15 | **Position monitoring enabled** | `position_monitor.enabled: true` in config | ✅ YES |
| 16 | **Position monitor interval** | `check_interval` >= 1 minute (avoid rate limits) | ✅ YES |
| 17 | **Max positions cap** | `risk.max_positions` must be set and reasonable (< 100) | ✅ YES |
| 18 | **Max capital at risk** | `risk.max_portfolio_risk_pct` must be < 100 | ✅ YES |
| 19 | **DTE range valid** | `min_dte < max_dte`, both > 0 | ✅ YES |
| 20 | **Spread width > 0** | Ensure `strategy.spread_width` > 0 | ✅ YES |

### 4.2 Validation Implementation

**Code structure:**

```python
# scripts/preflight_validator.py

from typing import Dict, List, Tuple
from pathlib import Path
import yaml
import sqlite3
import requests
from alpaca.trading.client import TradingClient


class ValidationResult:
    def __init__(self, passed: bool, message: str, blocking: bool = True):
        self.passed = passed
        self.message = message
        self.blocking = blocking

    def __repr__(self):
        icon = "✅" if self.passed else ("❌" if self.blocking else "⚠️")
        return f"{icon} {self.message}"


class PreFlightValidator:
    def __init__(self, portfolio_id: str, config_path: Path, env_path: Path):
        self.portfolio_id = portfolio_id
        self.config_path = config_path
        self.env_path = env_path
        self.config = None
        self.env = None

    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation checks. Returns (all_passed, results)."""
        results = []

        results.append(self._check_config_exists())
        if not results[-1].passed:
            return False, results  # Cannot continue without config

        results.append(self._check_config_syntax())
        if not results[-1].passed:
            return False, results

        self.config = self._load_config()
        self.env = self._load_env()

        results.extend([
            self._check_required_config_fields(),
            self._check_alpaca_keys(),
            self._check_alpaca_account_type(),
            self._check_polygon_key(),
            self._check_telegram_config(),
            self._check_stop_loss_config(),
            self._check_profit_target_config(),
            self._check_circuit_breaker_config(),
            self._check_database_unique(),
            self._check_database_writable(),
            self._check_shared_env(),
            self._check_log_dir_writable(),
            self._check_regime_detector(),
            self._check_position_monitor_enabled(),
            self._check_position_monitor_interval(),
            self._check_max_positions(),
            self._check_max_risk(),
            self._check_dte_range(),
            self._check_spread_width(),
        ])

        all_passed = all(r.passed or not r.blocking for r in results)
        return all_passed, results

    def _check_config_exists(self) -> ValidationResult:
        if not self.config_path.exists():
            return ValidationResult(False, f"Config file not found: {self.config_path}")
        return ValidationResult(True, "Config file exists")

    def _check_config_syntax(self) -> ValidationResult:
        try:
            with open(self.config_path) as f:
                yaml.safe_load(f)
            return ValidationResult(True, "Config file parses correctly")
        except yaml.YAMLError as e:
            return ValidationResult(False, f"Config syntax error: {e}")

    def _check_alpaca_keys(self) -> ValidationResult:
        api_key = self.env.get("ALPACA_API_KEY")
        api_secret = self.env.get("ALPACA_API_SECRET")

        if not api_key or not api_secret:
            return ValidationResult(False, "Alpaca API keys missing from .env")

        try:
            client = TradingClient(api_key, api_secret, paper=True)
            account = client.get_account()
            expected_account = self._get_alpaca_account_from_registry()
            if expected_account and account.account_number != expected_account:
                return ValidationResult(
                    False,
                    f"Alpaca account mismatch: expected {expected_account}, got {account.account_number}"
                )
            return ValidationResult(
                True,
                f"Alpaca API keys valid (account {account.account_number[-4:]})"
            )
        except Exception as e:
            return ValidationResult(False, f"Alpaca API test failed: {e}")

    def _check_stop_loss_config(self) -> ValidationResult:
        sl = self.config.get("risk", {}).get("stop_loss_multiplier")
        if not sl or sl <= 0:
            return ValidationResult(False, "Stop-loss config missing or invalid")
        return ValidationResult(True, f"Stop-loss config present ({sl}x multiplier)")

    def _check_circuit_breaker_config(self) -> ValidationResult:
        cb = self.config.get("risk", {}).get("drawdown_cb_pct")
        if not cb or cb <= 0 or cb >= 100:
            return ValidationResult(False, "Circuit breaker config missing or invalid")
        return ValidationResult(True, f"Circuit breaker config present ({cb}% drawdown)")

    def _check_shared_env(self) -> ValidationResult:
        """Check if this .env's Alpaca keys match any other portfolio."""
        api_key = self.env.get("ALPACA_API_KEY")
        registry = self._load_registry()

        for p in registry.get("portfolios", []):
            if p["portfolio_id"] == self.portfolio_id:
                continue  # Skip self

            other_env = self._load_env_from_path(Path(p["env_file"]))
            if other_env.get("ALPACA_API_KEY") == api_key:
                return ValidationResult(
                    False,
                    f"Shared .env detected: {p['portfolio_id']} uses same Alpaca keys"
                )

        return ValidationResult(True, "No shared .env detected (unique keys)")

    def _check_database_unique(self) -> ValidationResult:
        """Check if DB path is unique across all portfolios."""
        db_path = str(self.config.get("database", {}).get("path", "pilotai.db"))
        registry = self._load_registry()

        for p in registry.get("portfolios", []):
            if p["portfolio_id"] == self.portfolio_id:
                continue
            if p.get("db_path") == db_path:
                return ValidationResult(
                    False,
                    f"Shared database detected: {p['portfolio_id']} uses {db_path}"
                )

        return ValidationResult(True, "Database path unique")

    def _check_position_monitor_enabled(self) -> ValidationResult:
        enabled = self.config.get("position_monitor", {}).get("enabled", False)
        if not enabled:
            return ValidationResult(
                False,
                "Position monitoring disabled (required for runtime safety)"
            )
        return ValidationResult(True, "Position monitoring enabled")

    # Additional check methods omitted for brevity...

    def _load_config(self) -> Dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _load_env(self) -> Dict:
        env = {}
        with open(self.env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
        return env

    def _load_registry(self) -> Dict:
        registry_path = Path("portfolios/portfolios.json")
        if not registry_path.exists():
            return {"portfolios": []}
        with open(registry_path) as f:
            return json.load(f)
```

**Usage in CLI:**

```python
# scripts/portfolio_cli.py

def cmd_start(args):
    portfolio_id = args.portfolio_id
    portfolio = load_portfolio_from_registry(portfolio_id)

    if not args.skip_validation:
        validator = PreFlightValidator(
            portfolio_id,
            Path(portfolio["config_file"]),
            Path(portfolio["env_file"]),
        )
        all_passed, results = validator.validate_all()

        print(f"\nValidating portfolio: {portfolio_id}")
        print("━" * 60)
        for result in results:
            print(result)
        print("━" * 60)

        if not all_passed:
            print("❌ Pre-flight validation FAILED. Cannot launch.")
            sys.exit(1)

        print("✅ All checks passed. Launching...\n")

    # Proceed with launch...
```

### 4.3 Failure Handling

**If validation fails:**

1. **Exit immediately** — never launch with failed checks
2. **Output detailed error message** with remediation steps
3. **Log failure** to `logs/preflight_failures.log` with timestamp and portfolio ID
4. **Suggested fixes** in output (e.g., "Set `risk.stop_loss_multiplier` in config.yaml")

**Example failure output:**

```
Validating portfolio: exp_160
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Config file exists and parses correctly
✅ Alpaca API keys valid (account PA3ABCD1234 accessible)
❌ Stop-loss config missing or invalid
   → Fix: Add `risk.stop_loss_multiplier: 2.5` to config.yaml
✅ Profit target config present (50% of credit)
❌ Shared database detected: exp_036 uses portfolios/shared.db
   → Fix: Set unique `database.path` in config (e.g., portfolios/exp_160/pilotai.db)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Pre-flight validation FAILED. Cannot launch.
```

---

## 5. RUNTIME SAFETY

**Active protection systems that run continuously after launch.**

### 5.1 Position Monitoring Loop

**Purpose:** Enforce profit targets, stop-losses, and DTE-based exits on open positions.

**Architecture:**

```python
# scripts/position_monitor.py

import time
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from shared.database import get_trades
from strategy.alpaca_provider import AlpacaProvider


logger = logging.getLogger(__name__)


class PositionMonitor:
    """Monitors all open positions across portfolios and enforces exit rules."""

    def __init__(self, registry_path: str = "portfolios/portfolios.json"):
        self.registry_path = registry_path
        self.check_interval = 300  # 5 minutes

    def run_forever(self):
        """Main monitoring loop."""
        logger.info("Position monitor started (check interval: 5 min)")

        while True:
            try:
                self._monitor_cycle()
            except Exception as e:
                logger.error(f"Monitor cycle error: {e}", exc_info=True)

            time.sleep(self.check_interval)

    def _monitor_cycle(self):
        """Single monitoring pass across all running portfolios."""
        registry = self._load_registry()
        running_portfolios = [
            p for p in registry.get("portfolios", [])
            if p["status"] == "running"
        ]

        if not running_portfolios:
            logger.debug("No running portfolios to monitor")
            return

        logger.info(f"Monitoring {len(running_portfolios)} portfolio(s)")

        for portfolio in running_portfolios:
            try:
                self._monitor_portfolio(portfolio)
            except Exception as e:
                logger.error(
                    f"Error monitoring {portfolio['portfolio_id']}: {e}",
                    exc_info=True
                )

    def _monitor_portfolio(self, portfolio: Dict):
        """Monitor all open positions for a single portfolio."""
        portfolio_id = portfolio["portfolio_id"]
        db_path = portfolio["db_path"]
        config = self._load_config(portfolio["config_file"])
        env = self._load_env(portfolio["env_file"])

        # Get open positions from database
        open_positions = get_trades(status="open", path=db_path)

        if not open_positions:
            logger.debug(f"{portfolio_id}: No open positions")
            return

        logger.info(f"{portfolio_id}: Checking {len(open_positions)} position(s)")

        # Connect to Alpaca for this portfolio
        alpaca = AlpacaProvider(
            api_key=env["ALPACA_API_KEY"],
            api_secret=env["ALPACA_API_SECRET"],
            paper=True,
        )

        for position in open_positions:
            self._check_position(position, alpaca, config, db_path)

    def _check_position(
        self,
        position: Dict,
        alpaca: AlpacaProvider,
        config: Dict,
        db_path: str,
    ):
        """Check a single position and close if exit conditions met."""
        position_id = position["position_id"]
        ticker = position["ticker"]

        # Get current option prices from Alpaca
        current_price = self._get_spread_price(position, alpaca)

        if current_price is None:
            logger.warning(f"{position_id}: Could not fetch current price")
            return

        # Calculate P&L
        entry_credit = position["credit"]
        current_cost = current_price
        unrealized_pnl = (entry_credit - current_cost) * position["contracts"] * 100

        # Exit condition checks
        exit_reason = None

        # 1. Profit target: 50% of credit
        profit_target_pct = config.get("risk", {}).get("profit_target", 50) / 100
        if current_cost <= entry_credit * (1 - profit_target_pct):
            exit_reason = f"profit_target_{int(profit_target_pct * 100)}pct"

        # 2. Stop-loss: multiplier × credit
        stop_loss_mult = config.get("risk", {}).get("stop_loss_multiplier", 2.5)
        max_loss = entry_credit * stop_loss_mult
        if current_cost >= entry_credit + max_loss:
            exit_reason = f"stop_loss_{stop_loss_mult}x"

        # 3. DTE threshold: close at 21 DTE
        manage_dte = config.get("strategy", {}).get("manage_dte", 21)
        expiration = datetime.fromisoformat(position["expiration"])
        dte = (expiration - datetime.now(timezone.utc)).days
        if dte <= manage_dte:
            exit_reason = f"dte_{dte}"

        if not exit_reason:
            logger.debug(
                f"{position_id}: Healthy — P&L ${unrealized_pnl:.2f}, "
                f"DTE {dte}, price ${current_cost:.2f}"
            )
            return

        # Close position
        logger.warning(
            f"{position_id}: Exit triggered ({exit_reason}) — "
            f"closing at ${current_cost:.2f}"
        )

        try:
            self._close_position(position, alpaca, current_cost, exit_reason, db_path)
        except Exception as e:
            logger.error(f"{position_id}: Close failed: {e}", exc_info=True)

    def _close_position(
        self,
        position: Dict,
        alpaca: AlpacaProvider,
        exit_price: float,
        exit_reason: str,
        db_path: str,
    ):
        """Submit close order and update database."""
        result = alpaca.close_spread(
            ticker=position["ticker"],
            short_strike=position["short_strike"],
            long_strike=position["long_strike"],
            expiration=position["expiration"],
            spread_type=position["type"],
            contracts=position["contracts"],
            limit_price=exit_price,
        )

        if result["status"] != "submitted":
            raise RuntimeError(f"Close order rejected: {result.get('message')}")

        # Update database
        realized_pnl = (position["credit"] - exit_price) * position["contracts"] * 100
        from shared.database import upsert_trade

        position["status"] = "closed"
        position["exit_price"] = exit_price
        position["exit_reason"] = exit_reason
        position["exit_date"] = datetime.now(timezone.utc).isoformat()
        position["pnl"] = realized_pnl

        upsert_trade(position, source="position_monitor", path=db_path)

        logger.info(
            f"{position['position_id']}: Closed successfully "
            f"(P&L ${realized_pnl:.2f}, reason: {exit_reason})"
        )

    def _get_spread_price(self, position: Dict, alpaca: AlpacaProvider) -> float:
        """Get current bid/ask midpoint for the spread."""
        # Implementation: query Alpaca for option quotes, compute spread mid price
        # Details omitted for brevity
        pass

    def _load_registry(self) -> Dict:
        with open(self.registry_path) as f:
            return json.load(f)

    def _load_config(self, path: str) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)

    def _load_env(self, path: str) -> Dict:
        env = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
        return env
```

**Deployment:**

```bash
# Run as systemd service (Linux)
# /etc/systemd/system/pilotai-position-monitor.service

[Unit]
Description=PilotAI Position Monitor
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/pilotai-credit-spreads
ExecStart=/usr/bin/python3 scripts/position_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Monitoring schedule:**

- **Check interval:** Every 5 minutes during market hours (9:30 AM – 4:00 PM ET)
- **After-hours:** Every 30 minutes (for assignment risk near expiry)

### 5.2 Circuit Breaker

**Purpose:** Halt new entries when drawdown exceeds threshold.

**Implementation:**

```python
# shared/circuit_breaker.py

from datetime import datetime, timezone
from pathlib import Path
import json


class CircuitBreaker:
    """Per-portfolio circuit breaker for drawdown protection."""

    def __init__(self, portfolio_dir: Path, threshold_pct: float):
        self.portfolio_dir = portfolio_dir
        self.threshold_pct = threshold_pct
        self.state_file = portfolio_dir / "state" / "circuit_breaker.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def check(self, current_equity: float, peak_equity: float) -> bool:
        """Check if circuit breaker should trip.

        Returns:
            True if trading allowed, False if circuit breaker triggered.
        """
        drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100

        if drawdown_pct >= self.threshold_pct:
            self._trip(drawdown_pct)
            return False

        return True

    def _trip(self, drawdown_pct: float):
        """Record circuit breaker trip."""
        state = {
            "tripped": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drawdown_pct": drawdown_pct,
            "threshold_pct": self.threshold_pct,
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

        logger.critical(
            f"🚨 CIRCUIT BREAKER TRIGGERED: {drawdown_pct:.1f}% drawdown "
            f"(threshold: {self.threshold_pct}%)"
        )

    def is_tripped(self) -> bool:
        """Check if circuit breaker is currently tripped."""
        if not self.state_file.exists():
            return False

        with open(self.state_file) as f:
            state = json.load(f)
            return state.get("tripped", False)

    def reset(self):
        """Manually reset circuit breaker (admin only)."""
        if self.state_file.exists():
            self.state_file.unlink()
        logger.info("Circuit breaker reset")
```

**Integration:**

```python
# In scanner process before generating opportunities

cb = CircuitBreaker(
    Path(f"portfolios/{portfolio_id}"),
    threshold_pct=config["risk"]["drawdown_cb_pct"],
)

if cb.is_tripped():
    logger.warning("Circuit breaker active — skipping scan")
    return

# Calculate current drawdown from Alpaca account
account = alpaca.get_account()
current_equity = account["equity"]
peak_equity = get_peak_equity_from_db(db_path)  # Track high-water mark

if not cb.check(current_equity, peak_equity):
    logger.critical("Circuit breaker TRIGGERED — halting new entries")
    send_telegram_alert(f"🚨 Portfolio {portfolio_id} circuit breaker triggered")
    return

# Proceed with scan...
```

### 5.3 Duplicate Trade Prevention

**Problem:** Scanner could generate the same opportunity multiple times per day.

**Solution:**

```python
# shared/duplicate_checker.py

def is_duplicate_trade(
    ticker: str,
    short_strike: float,
    long_strike: float,
    expiration: str,
    db_path: str,
) -> bool:
    """Check if this exact spread already exists as an open position."""
    from shared.database import get_trades

    open_positions = get_trades(status="open", path=db_path)

    for pos in open_positions:
        if (
            pos["ticker"] == ticker
            and pos["short_strike"] == short_strike
            and pos["long_strike"] == long_strike
            and pos["expiration"].startswith(expiration)
        ):
            return True

    return False
```

**Usage in scanner:**

```python
# Before submitting order

if is_duplicate_trade(
    opp["ticker"],
    opp["short_strike"],
    opp["long_strike"],
    opp["expiration"],
    db_path,
):
    logger.info(f"Skipping duplicate trade: {opp['ticker']} {opp['short_strike']}/{opp['long_strike']}")
    continue
```

### 5.4 Market Hours Enforcement

**Rule:** Only submit orders during market hours (9:30 AM – 4:00 PM ET).

```python
# shared/market_hours.py

from datetime import datetime, time
from zoneinfo import ZoneInfo


def is_market_hours() -> bool:
    """Check if current time is within market hours (9:30 AM - 4:00 PM ET)."""
    et_tz = ZoneInfo("America/New_York")
    now = datetime.now(et_tz)

    # Check if weekday (Mon-Fri)
    if now.weekday() >= 5:  # Sat=5, Sun=6
        return False

    # Check time window
    market_open = time(9, 30)
    market_close = time(16, 0)

    return market_open <= now.time() <= market_close
```

**Enforcement:**

```python
# In scanner before order submission

if not is_market_hours():
    logger.info("Outside market hours — skipping order submission")
    return
```

### 5.5 Max Positions Cap

**Rule:** Never exceed `max_positions` limit per portfolio.

```python
# Before opening new position

open_positions = get_trades(status="open", path=db_path)
max_positions = config["risk"]["max_positions"]

if len(open_positions) >= max_positions:
    logger.warning(f"Max positions reached ({max_positions}) — skipping entry")
    return
```

### 5.6 Max Capital at Risk Cap

**Rule:** Total capital at risk across all positions cannot exceed `max_portfolio_risk_pct`.

```python
# Calculate total capital at risk

total_risk = sum(
    pos["max_loss"] * pos["contracts"] * 100
    for pos in open_positions
)

account_value = alpaca.get_account()["equity"]
max_risk = account_value * (config["risk"]["max_portfolio_risk_pct"] / 100)

if total_risk >= max_risk:
    logger.warning(
        f"Max portfolio risk reached (${total_risk:.0f} / ${max_risk:.0f}) "
        f"— skipping entry"
    )
    return
```

### 5.7 Crash Recovery

**Problem:** Process crashes leave positions unmonitored. Database may be stale relative to Alpaca reality.

**Solution:** Reconciliation on startup.

```python
# In scanner startup (main.py)

from shared.reconciler import PositionReconciler

def startup_reconciliation(portfolio_id: str):
    """Reconcile database state against Alpaca on process start."""
    logger.info(f"Running crash recovery reconciliation for {portfolio_id}")

    portfolio = load_portfolio_from_registry(portfolio_id)
    alpaca = load_alpaca_provider(portfolio)
    reconciler = PositionReconciler(alpaca, db_path=portfolio["db_path"])

    result = reconciler.reconcile()

    logger.info(f"Reconciliation complete: {result}")

    if result.pending_resolved > 0:
        send_telegram_alert(
            f"⚠️ {portfolio_id}: Reconciled {result.pending_resolved} "
            f"pending_open trades on startup"
        )

    if result.pending_failed > 0:
        send_telegram_alert(
            f"❌ {portfolio_id}: {result.pending_failed} trades failed to open "
            f"(orders not found in Alpaca)"
        )
```

**Reconciliation logic:**

- **`pending_open` → `open`**: If Alpaca order is filled
- **`pending_open` → `failed_open`**: If Alpaca order is rejected/cancelled/not found
- **`open` → validate**: Confirm position still exists in Alpaca

---

## 6. OBSERVABILITY

**Comprehensive visibility into system health and performance.**

### 6.1 Per-Portfolio Log Files

**Log rotation:** Daily rotation with 30-day retention.

```
portfolios/exp_154/logs/
├── scanner_2026-03-06.log       # Today's scan activity
├── scanner_2026-03-05.log       # Yesterday's logs
├── position_monitor_2026-03-06.log
├── health_check_2026-03-06.log
└── errors.log                   # Persistent error log (no rotation)
```

**Log format:**

```
2026-03-06 15:42:33 [INFO] [exp_154] scanner: Found 12 opportunities (SPY, QQQ, IWM)
2026-03-06 15:42:34 [INFO] [exp_154] scanner: Top opportunity — SPY 580/575 bull put, score 72
2026-03-06 15:42:35 [INFO] [exp_154] alpaca: Order submitted (ID: abc123, status: submitted)
2026-03-06 15:42:36 [INFO] [exp_154] database: Trade inserted (position_id: SPY_bull_put_20260306_154235)
2026-03-06 15:47:01 [INFO] [exp_154] position_monitor: Checked 12 positions (all healthy)
2026-03-06 15:52:01 [INFO] [exp_154] position_monitor: Position SPY_bull_put_20260305_143210 hit profit target (50%) — closing
2026-03-06 15:52:02 [INFO] [exp_154] alpaca: Close order submitted (ID: xyz789)
2026-03-06 15:52:03 [INFO] [exp_154] database: Position closed (P&L: $164)
```

**Logging configuration (per portfolio):**

```python
# utils.py

def setup_portfolio_logging(portfolio_id: str, log_dir: Path):
    """Configure logging for a portfolio."""
    log_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"scanner_{today}.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(portfolio_id)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(lambda record: setattr(record, "portfolio_id", portfolio_id) or True)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
```

### 6.2 Health Check Endpoint

**HTTP server per portfolio on unique port:**

```python
# scripts/health_check_server.py

from flask import Flask, jsonify
from datetime import datetime, timezone
import json


def create_health_check_app(portfolio_id: str, registry_path: str):
    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        """Return health status for this portfolio."""
        registry = load_registry(registry_path)
        portfolio = find_portfolio(registry, portfolio_id)

        if not portfolio:
            return jsonify({"status": "error", "message": "Portfolio not found"}), 404

        last_heartbeat = datetime.fromisoformat(portfolio["last_heartbeat"])
        age_seconds = (datetime.now(timezone.utc) - last_heartbeat).total_seconds()

        status = "healthy" if age_seconds < 600 else "degraded"  # 10min threshold

        return jsonify({
            "portfolio_id": portfolio_id,
            "status": status,
            "pid": portfolio.get("pid"),
            "last_heartbeat": portfolio["last_heartbeat"],
            "heartbeat_age_seconds": age_seconds,
            "uptime_seconds": _get_process_uptime(portfolio.get("pid")),
        })

    @app.route("/positions", methods=["GET"])
    def positions():
        """Return open positions for this portfolio."""
        portfolio = load_portfolio_from_registry(portfolio_id)
        open_positions = get_trades(status="open", path=portfolio["db_path"])
        return jsonify({
            "portfolio_id": portfolio_id,
            "open_positions": len(open_positions),
            "positions": open_positions,
        })

    return app


if __name__ == "__main__":
    import sys
    portfolio_id = sys.argv[1]
    port = int(sys.argv[2])

    app = create_health_check_app(portfolio_id, "portfolios/portfolios.json")
    app.run(host="0.0.0.0", port=port)
```

**Start health check server:**

```bash
# Automatically started when portfolio launches
python scripts/health_check_server.py exp_154 8002 &
```

**Query health:**

```bash
curl http://localhost:8002/health
# {"portfolio_id": "exp_154", "status": "healthy", "heartbeat_age_seconds": 42, ...}
```

### 6.3 Telegram Alerts

**Alert triggers:**

| Event | Priority | Example Message |
|-------|----------|-----------------|
| **System start** | Info | `✅ Portfolio exp_154 started (PID 12346)` |
| **System stop** | Info | `⏹️ Portfolio exp_154 stopped gracefully` |
| **System crash** | Critical | `💥 Portfolio exp_154 process died (PID 12346 not responding)` |
| **Circuit breaker trip** | Critical | `🚨 Portfolio exp_154 circuit breaker TRIGGERED (-42.3% drawdown)` |
| **Stop-loss hit** | Warning | `🛑 exp_154: SPY 580/575 bull put stopped out at 2.5x ($205 loss)` |
| **Profit target** | Success | `💰 exp_154: QQQ 520/515 bull put closed at 50% profit ($164 gain)` |
| **Position opened** | Info | `📈 exp_154: Opened SPY 580/575 bull put (2 contracts, $0.82 credit)` |
| **Position closed** | Info | `📉 exp_154: Closed SPY 580/575 at DTE 21 ($58 profit)` |
| **Daily P&L summary** | Info | `📊 exp_154 Daily Summary: 3 trades, +$327 (+0.33%)` |
| **Pre-flight failure** | Error | `❌ exp_154 launch BLOCKED: Stop-loss config missing` |
| **API error** | Error | `⚠️ exp_154: Alpaca API rate limit hit (retrying in 60s)` |

**Implementation:**

```python
# alerts/telegram_bot.py (enhanced)

def send_portfolio_alert(
    portfolio_id: str,
    event_type: str,
    message: str,
    priority: str = "info",
):
    """Send Telegram alert for a portfolio event."""
    # Load .env from portfolio directory to get bot token
    portfolio = load_portfolio_from_registry(portfolio_id)
    env = load_env(portfolio["env_file"])

    bot_token = env.get("TELEGRAM_BOT_TOKEN")
    chat_id = env.get("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logger.warning(f"Telegram not configured for {portfolio_id}")
        return

    # Format message with priority icon
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "critical": "🚨",
    }
    icon = icons.get(priority, "ℹ️")

    full_message = f"{icon} **{portfolio_id}** | {message}"

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": full_message,
                "parse_mode": "Markdown",
            },
            timeout=10,
        )
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")
```

**Daily P&L summary (EOD):**

```python
# Run at 4:30 PM ET daily

def send_daily_summary(portfolio_id: str):
    """Send end-of-day P&L summary."""
    portfolio = load_portfolio_from_registry(portfolio_id)
    db_path = portfolio["db_path"]

    today = datetime.now(timezone.utc).date()
    trades_today = [
        t for t in get_trades(path=db_path)
        if datetime.fromisoformat(t["entry_date"]).date() == today
    ]

    closed_today = [t for t in trades_today if t["status"] == "closed"]
    total_pnl = sum(t["pnl"] for t in closed_today)

    open_positions = get_trades(status="open", path=db_path)

    message = (
        f"📊 **Daily Summary** ({today})\n\n"
        f"• Trades opened: {len(trades_today)}\n"
        f"• Trades closed: {len(closed_today)}\n"
        f"• Realized P&L: ${total_pnl:.2f}\n"
        f"• Open positions: {len(open_positions)}\n"
    )

    send_portfolio_alert(portfolio_id, "daily_summary", message, priority="info")
```

### 6.4 Metrics Dashboard (Web UI)

**Future enhancement:** Real-time dashboard showing all portfolios.

```
┌─────────────────────────────────────────────────────────────┐
│  PilotAI Portfolio Manager Dashboard                         │
├─────────────────────────────────────────────────────────────┤
│  Portfolio       Status      Open Pos   P&L Today   Uptime  │
│  exp_036         ● Running   8          +$412       4h 32m  │
│  exp_154         ● Running   12         -$89        2h 18m  │
│  exp_160         ○ Stopped   —          —           —       │
├─────────────────────────────────────────────────────────────┤
│  Total Open Positions: 20                                   │
│  Total Capital at Risk: $8,450 / $40,000 (21%)              │
│  Combined P&L Today: +$323 (+0.32%)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. FAILURE SCENARIOS

**Comprehensive failure analysis with mitigations.**

### 7.1 API Rejection Mid-Order

**Scenario:** Alpaca rejects order after submission (insufficient buying power, contract unavailable, market closed).

**Impact:** Position marked `pending_open` in DB but never fills. Capital reserved but not deployed.

**Detection:**
- Position monitor queries Alpaca order status
- If order status is `rejected`, `cancelled`, or `expired` → mark trade as `failed_open`

**Mitigation:**
1. **Pre-flight buying power check**: Before submitting, verify `alpaca.get_account()["options_buying_power"]` sufficient for max loss
2. **Immediate status poll**: After order submission, wait 5 seconds and poll status — if rejected, log immediately
3. **Reconciliation sweep**: On next startup, reconciler flags all `pending_open` trades older than 4 hours
4. **Telegram alert**: Notify on order rejection with reason

**Code:**

```python
# In scanner order submission

result = alpaca.submit_credit_spread(...)

if result["status"] == "error":
    logger.error(f"Order rejected: {result['message']}")
    send_portfolio_alert(
        portfolio_id,
        "order_rejected",
        f"Order rejected: {result['message']}",
        priority="error"
    )
    return  # Do not persist to DB

# Wait and verify
time.sleep(5)
order_status = alpaca.get_order_status(result["order_id"])

if order_status["status"] in {"rejected", "cancelled"}:
    logger.error(f"Order failed post-submission: {order_status}")
    # Mark as failed_open immediately
    trade["status"] = "failed_open"
    trade["exit_reason"] = f"alpaca_{order_status['status']}"
    upsert_trade(trade, source="scanner", path=db_path)
```

### 7.2 Crash Between Leg 1 and Leg 2

**Scenario:** Process crashes after selling short option but before buying long option. Naked short exposure.

**Impact:** **CATASTROPHIC** — unbounded risk. Alpaca paper trading won't execute this in production, but real trading would.

**Root cause:** This cannot happen with multi-leg orders (MLEG). Alpaca executes both legs atomically or rejects the entire order.

**Mitigation:**
1. **Always use MLEG orders**: Never submit legs separately (already implemented in `AlpacaProvider.submit_credit_spread`)
2. **Pre-flight validation**: Check `order_class == "mleg"` in all spread submissions
3. **Reconciliation**: If a `pending_open` trade resolves to only one filled leg in Alpaca (impossible with MLEG but check anyway), immediately close the naked leg and alert

**Verification:**

```python
# In reconciler

def _verify_both_legs_filled(position: Dict, alpaca: AlpacaProvider) -> bool:
    """Check that both legs of a spread filled."""
    order = alpaca.get_order_by_client_id(position["alpaca_client_order_id"])

    if not order or not order.get("legs"):
        return False

    filled_legs = [leg for leg in order["legs"] if leg["status"] == "filled"]

    if len(filled_legs) == 1:
        # One leg filled, one not → EMERGENCY
        logger.critical(
            f"PARTIAL FILL DETECTED: {position['position_id']} — "
            f"only one leg filled! Closing naked position."
        )
        send_portfolio_alert(
            position["portfolio_id"],
            "partial_fill",
            f"🚨 PARTIAL FILL: {position['ticker']} spread incomplete — "
            f"closing naked leg immediately",
            priority="critical",
        )
        # Close the filled leg immediately
        # (implementation depends on which leg filled)
        return False

    return len(filled_legs) == 2
```

### 7.3 Stale Polygon Data

**Scenario:** Polygon API returns cached or delayed data. Scanner sees outdated prices, opens trades with bad pricing.

**Impact:** Adverse selection — entering at worse prices than expected. P&L drag.

**Detection:**
- Check `timestamp` field in Polygon responses
- If data age > 5 minutes during market hours → reject

**Mitigation:**

```python
# In data_cache.py

def get_latest_price(self, ticker: str) -> Optional[float]:
    """Get latest price with staleness check."""
    data = self._fetch_from_polygon(ticker)

    if not data:
        return None

    # Check data age
    data_timestamp = datetime.fromisoformat(data["timestamp"])
    age_seconds = (datetime.now(timezone.utc) - data_timestamp).total_seconds()

    if age_seconds > 300:  # 5 minutes
        logger.warning(
            f"Stale data for {ticker}: {age_seconds:.0f}s old — rejecting"
        )
        return None

    return data["price"]
```

**Fallback:** If Polygon data is stale, fall back to yfinance (free but 15-minute delay). Log data source in trade record.

### 7.4 Circuit Breaker Trigger During Monitoring

**Scenario:** Drawdown exceeds threshold. Circuit breaker trips. Scanner stops generating entries. Position monitor continues.

**Expected behavior:** This is CORRECT. Circuit breaker only halts new entries. Existing positions must still be monitored for exits.

**Mitigation:** None needed — this is the design. Alert user of CB trip and continue monitoring.

### 7.5 Wrong .env Loaded

**Scenario:** Portfolio exp_154 accidentally loads exp_036's .env. Orders submit to wrong Alpaca account. Trades intermingle.

**Impact:** **SEVERE** — cross-portfolio contamination. P&L attribution broken. Reconciliation chaos.

**Detection:**
- Pre-flight validation checks .env uniqueness (see Section 4)
- On startup, verify Alpaca account number from API matches registry entry

**Mitigation:**

```python
# In startup

def verify_correct_env_loaded(portfolio_id: str, env: Dict):
    """Verify .env loaded is correct for this portfolio."""
    portfolio = load_portfolio_from_registry(portfolio_id)
    expected_account = portfolio["alpaca_account"]

    # Test API
    alpaca = AlpacaProvider(env["ALPACA_API_KEY"], env["ALPACA_API_SECRET"])
    account = alpaca.get_account()

    if account["account_number"] != expected_account:
        raise RuntimeError(
            f"ENV MISMATCH: Portfolio {portfolio_id} expected account "
            f"{expected_account}, but .env connects to {account['account_number']}. "
            f"ABORTING."
        )

    logger.info(f"✅ Verified correct .env loaded for {portfolio_id}")
```

**If mismatch detected:** Exit immediately with critical alert. Do not proceed.

### 7.6 Partial Fills on MLEG

**Scenario:** Alpaca MLEG order partially fills (e.g., 3 of 5 spreads executed).

**Impact:** Position size mismatch. Risk calculations off.

**Detection:**
- After order submission, poll `filled_qty` vs `qty` requested
- If `filled_qty < qty` after 60 seconds → order partially filled or pending

**Mitigation:**

```python
# After order submission

def wait_for_fill(order_id: str, alpaca: AlpacaProvider, timeout: int = 60) -> Dict:
    """Wait for order to fill or timeout."""
    start = time.time()

    while time.time() - start < timeout:
        order = alpaca.get_order_status(order_id)

        if order["status"] == "filled":
            return order

        if order["status"] in {"rejected", "cancelled", "expired"}:
            raise RuntimeError(f"Order failed: {order['status']}")

        time.sleep(2)

    # Timeout
    order = alpaca.get_order_status(order_id)
    filled_qty = int(order.get("filled_qty", 0))
    requested_qty = int(order["qty"])

    if filled_qty < requested_qty:
        logger.warning(
            f"Partial fill: {filled_qty}/{requested_qty} spreads filled after {timeout}s"
        )
        # Accept partial fill and record actual quantity
        return order

    raise TimeoutError(f"Order {order_id} not filled after {timeout}s")
```

**Recording partial fills:**

```python
# In trade record

trade["contracts"] = filled_qty  # Actual filled quantity, not requested
trade["partial_fill"] = True if filled_qty < requested_qty else False
```

### 7.7 Assignment Risk Near Expiry

**Scenario:** Short option goes ITM near expiration. Risk of assignment if not closed.

**Impact:** Position converts to naked short stock. Requires immediate buy-to-close of stock.

**Detection:**
- Position monitor checks DTE
- If DTE <= 3 days and short strike is ITM → urgent close

**Mitigation:**

```python
# In position monitor

def _check_assignment_risk(position: Dict, current_price: float) -> bool:
    """Check if position has assignment risk."""
    dte = (datetime.fromisoformat(position["expiration"]) - datetime.now(timezone.utc)).days

    if dte > 3:
        return False  # Safe

    short_strike = position["short_strike"]
    spread_type = position["type"]

    # Check if short strike is ITM
    if "put" in spread_type.lower():
        itm = current_price < short_strike
    else:  # call
        itm = current_price > short_strike

    if itm:
        logger.warning(
            f"{position['position_id']}: ASSIGNMENT RISK — "
            f"{dte} DTE and short strike ITM. Closing immediately."
        )
        return True

    return False
```

**Rule:** Close all positions at DTE 3 (or earlier if ITM), no exceptions.

### 7.8 Alpaca Rate Limits Across 3 Portfolios

**Scenario:** Three portfolios submit orders simultaneously. Combined request rate exceeds Alpaca API limit (200 req/min).

**Impact:** Orders rejected with 429 Too Many Requests. Opportunities missed.

**Detection:**
- Alpaca SDK raises `alpaca.common.exceptions.APIError` with status 429
- Circuit breaker in `AlpacaProvider` trips after 5 consecutive 429s

**Mitigation:**

```python
# In AlpacaProvider (already implemented)

from shared.circuit_breaker import CircuitBreaker

class AlpacaProvider:
    def __init__(self, ...):
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)

    @_retry_with_backoff(max_retries=2, base_delay=1.0)
    def submit_credit_spread(self, ...):
        try:
            order = self._circuit_breaker.call(self.client.submit_order, order_req)
        except CircuitOpenError:
            logger.error("Alpaca circuit breaker OPEN — too many failures")
            raise
```

**Additional mitigation:**

1. **Stagger scans**: Offset portfolio scan times by 2-3 minutes
2. **Shared rate limiter**: Global token bucket for Alpaca requests across all portfolios (future enhancement)
3. **Exponential backoff**: Already implemented in `_retry_with_backoff` decorator

**Scan schedule:**

```python
# In scheduler

PORTFOLIO_SCAN_OFFSETS = {
    "exp_036": 0,      # Scan at :00, :15, :30, :45
    "exp_059": 1,      # Scan at :01, :16, :31, :46
    "exp_154": 2,      # Scan at :02, :17, :32, :47
}
```

---

## 8. FILE STRUCTURE

**Complete directory layout for multi-portfolio system:**

```
pilotai-credit-spreads/
│
├── main.py                       # Main entry point (scanner, backtest, dashboard)
├── config.yaml.example           # Template config (do not use directly)
├── .env.example                  # Template .env (never commit real keys)
├── requirements.txt              # Python dependencies
├── README.md
├── MASTERPLAN.md                 # Optimization roadmap
├── CLAUDE.md                     # Coding guidelines
│
├── portfolios/                   # ** NEW: Portfolio isolation root **
│   ├── portfolios.json           # Central registry (source of truth)
│   ├── exp_036/                  # Portfolio-specific directory
│   │   ├── config.yaml           # Strategy configuration
│   │   ├── .env                  # Alpaca/Polygon/Telegram keys
│   │   ├── pilotai.db            # SQLite database (trades, alerts)
│   │   ├── logs/                 # Log files
│   │   │   ├── scanner_2026-03-06.log
│   │   │   ├── position_monitor_2026-03-06.log
│   │   │   └── errors.log
│   │   └── state/                # Runtime state
│   │       ├── circuit_breaker.json
│   │       ├── reconciliation_state.json
│   │       └── pause.flag        # Presence of file = paused
│   ├── exp_154/
│   │   ├── config.yaml
│   │   ├── .env
│   │   ├── pilotai.db
│   │   ├── logs/
│   │   └── state/
│   └── archive/                  # Deleted portfolios
│       └── exp_099_2026-02-28/
│
├── scripts/                      # CLI and automation tools
│   ├── portfolio_cli.py          # ** NEW: Main CLI (portfolio create/start/stop/etc.) **
│   ├── preflight_validator.py   # ** NEW: Pre-flight validation suite **
│   ├── position_monitor.py      # ** NEW: Runtime position monitoring loop **
│   ├── health_check_server.py   # ** NEW: HTTP health check endpoint **
│   ├── run_optimization.py      # Backtest optimization harness
│   └── validate_params.py       # Overfit detection
│
├── shared/                       # Shared utilities
│   ├── database.py               # SQLite ORM
│   ├── data_cache.py             # Price data caching
│   ├── reconciler.py             # Position reconciliation
│   ├── circuit_breaker.py        # ** ENHANCED: Circuit breaker class **
│   ├── scheduler.py              # Market-hours scheduler
│   ├── market_hours.py           # ** NEW: Market hours enforcement **
│   ├── duplicate_checker.py     # ** NEW: Duplicate trade prevention **
│   ├── exceptions.py             # Custom exceptions
│   └── constants.py              # Global constants
│
├── strategy/                     # Trading strategy core
│   ├── credit_spread_strategy.py
│   ├── technical_analyzer.py
│   ├── options_analyzer.py
│   ├── alpaca_provider.py        # Alpaca API client
│   └── regime_detector.py        # Market regime detection
│
├── tracker/                      # Trade tracking (legacy, replaced by DB)
│   ├── trade_tracker.py
│   └── pnl_dashboard.py
│
├── alerts/                       # Alert generation
│   ├── alert_generator.py
│   ├── telegram_bot.py           # ** ENHANCED: Per-portfolio alerts **
│   ├── risk_gate.py
│   └── formatters/
│
├── backtest/                     # Backtesting engine
│   ├── backtester.py
│   ├── historical_data.py
│   └── performance_metrics.py
│
├── ml/                           # ML pipeline (regime detection, scoring)
│   ├── ml_pipeline.py
│   ├── regime_detector.py
│   └── feature_engine.py
│
├── web/                          # Web dashboard (future)
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── configs/                      # Backtest experiment configs (archive)
│   ├── exp_001_bull_put_only.json
│   ├── exp_154_risk5_icr12_sl35.json
│   └── ...
│
├── data/                         # Shared data cache (read-only)
│   ├── price_history/
│   └── options_cache/
│
├── logs/                         # System-level logs (not portfolio-specific)
│   ├── position_monitor.log      # Global position monitor
│   ├── preflight_failures.log    # Pre-flight validation failures
│   └── portfolio_manager.log     # CLI activity
│
├── output/                       # Backtest results, leaderboard
│   ├── leaderboard.json
│   ├── optimization_log.json
│   └── backtest_reports/
│
├── tests/                        # Unit and integration tests
│   ├── test_preflight_validator.py  # ** NEW **
│   ├── test_position_monitor.py     # ** NEW **
│   ├── test_circuit_breaker.py      # ** NEW **
│   └── ...
│
└── docs/                         # Documentation
    ├── PORTFOLIO_MANAGER_DESIGN.md   # ** THIS DOCUMENT **
    ├── API.md
    └── BACKTEST_GUIDE.md
```

**Key principles:**

1. **Portfolio isolation**: Everything portfolio-specific lives under `portfolios/<id>/`
2. **No shared state**: Each portfolio has its own DB, logs, config, .env
3. **Central registry**: `portfolios.json` is single source of truth
4. **Backward compatibility**: Existing code in `strategy/`, `backtest/`, `shared/` unchanged
5. **CLI-driven**: All portfolio operations via `portfolio` command

---

## 9. IMPLEMENTATION PLAN

**Total estimated effort: 34 hours**

### Phase 1: Portfolio Registry + CLI + Pre-Flight (8 hours)

**Goal:** Establish portfolio registry and CLI framework with comprehensive validation.

**Deliverables:**

1. **Portfolio registry schema** (`portfolios.json`)
   - JSON schema with validation
   - CRUD operations (create, read, update, delete)
   - Locking mechanism for concurrent access

2. **CLI commands** (`scripts/portfolio_cli.py`)
   - `portfolio create`
   - `portfolio list`
   - `portfolio status`
   - `portfolio delete`
   - Argument parsing with `argparse`
   - User-friendly output formatting

3. **Pre-flight validator** (`scripts/preflight_validator.py`)
   - All 20 validation checks (see Section 4)
   - Blocking vs. warning logic
   - Detailed error messages with remediation
   - Integration with CLI `portfolio start --validate`

4. **Portfolio directory structure generator**
   - Scaffold `portfolios/<id>/` on creation
   - Copy config and .env templates
   - Initialize empty SQLite DB
   - Create log and state directories

**Testing:**
- Unit tests for registry CRUD
- Integration test: create portfolio, validate, list, delete
- Negative tests: duplicate IDs, missing config, invalid API keys

**Acceptance criteria:**
- [ ] Can create portfolio with `portfolio create --id exp_160 ...`
- [ ] Pre-flight validation blocks launch on missing SL config
- [ ] `portfolio list` shows status and metadata
- [ ] All tests pass

---

### Phase 2: Position Monitoring + SL/PT Enforcement (12 hours)

**Goal:** Build active position monitoring loop with profit-target and stop-loss enforcement.

**Deliverables:**

1. **Position monitor daemon** (`scripts/position_monitor.py`)
   - Multi-portfolio loop (query registry for running portfolios)
   - 5-minute check interval
   - Query Alpaca for current spread prices
   - Calculate unrealized P&L
   - Check PT (50% of credit) and SL (multiplier × credit)
   - Check DTE threshold (close at 21 DTE)
   - Submit close orders via `AlpacaProvider.close_spread()`
   - Update database with exit details

2. **Assignment risk detection**
   - Check if short strike is ITM with DTE <= 3
   - Force close immediately if ITM near expiry

3. **Position monitor logging**
   - Per-portfolio log files: `portfolios/<id>/logs/position_monitor_<date>.log`
   - Log every check cycle with status of all positions

4. **Telegram alerts for exits**
   - Profit target hit
   - Stop-loss triggered
   - DTE-based close
   - Assignment risk close

5. **Systemd service file** (Linux deployment)
   - Auto-restart on crash
   - Run as non-root user

**Testing:**
- Unit tests: PT/SL threshold calculations
- Integration test: Mock Alpaca responses, verify close orders submitted
- Simulate ITM near expiry, verify urgent close

**Acceptance criteria:**
- [ ] Position monitor runs continuously without crash
- [ ] Closes position at 50% profit (verified in paper trading)
- [ ] Closes position at 2.5x loss (verified)
- [ ] Closes all positions at 21 DTE
- [ ] Telegram alerts sent for all exit events

---

### Phase 3: Circuit Breaker + Crash Recovery (8 hours)

**Goal:** Implement drawdown circuit breaker and startup reconciliation.

**Deliverables:**

1. **Circuit breaker class** (`shared/circuit_breaker.py`)
   - Per-portfolio state tracking
   - Check current equity vs. peak equity
   - Trip at configured threshold (e.g., 40% drawdown)
   - Write trip flag to `portfolios/<id>/state/circuit_breaker.json`
   - Scanner checks trip flag before generating opportunities

2. **Circuit breaker integration in scanner**
   - On startup: load CB state
   - Before each scan: check if tripped
   - If tripped: skip opportunity generation, log warning
   - Manual reset via `portfolio resume`

3. **Startup reconciliation** (`shared/reconciler.py` — enhance existing)
   - Run on every scanner startup
   - Reconcile `pending_open` trades against Alpaca orders
   - Promote to `open` if filled
   - Mark as `failed_open` if rejected/not found
   - Telegram alert on reconciliation findings

4. **Duplicate trade prevention** (`shared/duplicate_checker.py`)
   - Check if exact spread already open before submitting order
   - Log duplicate skip

5. **Market hours enforcement** (`shared/market_hours.py`)
   - Block order submission outside 9:30 AM – 4:00 PM ET
   - Skip scans on weekends

**Testing:**
- Unit tests: CB threshold calculations
- Integration test: Simulate 40% drawdown, verify CB trips
- Test reconciliation: create `pending_open` trades, verify status updates
- Test duplicate detection: try to open same spread twice

**Acceptance criteria:**
- [ ] CB trips at 40% drawdown, scanner skips opportunities
- [ ] Reconciliation resolves `pending_open` on startup
- [ ] Duplicate trades are blocked
- [ ] Orders blocked outside market hours

---

### Phase 4: Full Observability (6 hours)

**Goal:** Complete logging, health checks, and alerting.

**Deliverables:**

1. **Per-portfolio logging** (enhance `utils.py`)
   - Daily log rotation
   - Log format with portfolio ID prefix
   - Separate logs for scanner, position monitor, health check

2. **Health check HTTP server** (`scripts/health_check_server.py`)
   - Flask app per portfolio on unique port
   - `/health` endpoint: PID, heartbeat age, status
   - `/positions` endpoint: open positions list
   - Auto-start with portfolio launch

3. **Telegram alert enhancements** (`alerts/telegram_bot.py`)
   - Per-portfolio bot configuration
   - All event types (see Section 6.3)
   - Daily P&L summary at 4:30 PM ET

4. **CLI enhancements**
   - `portfolio start`: spawn health check server, record port in registry
   - `portfolio stop`: kill health check server
   - `portfolio status --tail`: live-tail logs

5. **Documentation**
   - Update README with portfolio manager usage
   - Write `docs/PORTFOLIO_MANAGER_GUIDE.md` with examples

**Testing:**
- Integration test: start portfolio, query health endpoint
- Test Telegram alerts for all event types
- Verify log rotation works (create 31 daily files, check oldest deleted)

**Acceptance criteria:**
- [ ] Health check endpoint returns valid JSON
- [ ] Telegram daily summary sent at EOD
- [ ] Logs rotate daily, 30-day retention
- [ ] Documentation complete

---

### Phase 5: Production Hardening (Optional, +8 hours)

**Advanced features for production deployment:**

1. **Multi-portfolio dashboard** (web UI)
   - Real-time status for all portfolios
   - Aggregated P&L chart
   - Circuit breaker status indicators
   - One-click pause/resume

2. **Shared rate limiter** for Alpaca API
   - Token bucket across all portfolios
   - Global request counter
   - Auto-throttle if approaching limit

3. **Performance monitoring** (Prometheus/Grafana)
   - Metrics: scan latency, order submission success rate, position count
   - Alerts on anomalies

4. **Backup and disaster recovery**
   - Daily SQLite backups to S3/Railway
   - Config versioning in git
   - Rollback procedures

---

## 10. CONFIG SCHEMA

**Required configuration fields for every portfolio.**

### 10.1 YAML Config Structure

```yaml
# portfolios/exp_154/config.yaml

# Portfolio metadata (optional, for human reference)
portfolio:
  id: exp_154
  name: "ComboRegime IC Focus"
  description: "Risk 5%, IC-only, 12% combined credit floor, SL 3.5x"
  tags: [iron_condor, regime_v2, conservative]

# Tickers to monitor (REQUIRED)
tickers:
  - SPY
  - QQQ
  - IWM

# Strategy parameters (REQUIRED)
strategy:
  min_dte: 25                      # Minimum days to expiration (int, > 0)
  max_dte: 45                      # Maximum days to expiration (int, > min_dte)
  manage_dte: 21                   # Close positions at this DTE (int, > 0, < min_dte)
  use_delta_selection: false       # Use delta-based strike selection (bool)
  target_delta: 0.12               # Target short-strike delta (float, 0-1)
  spread_width: 5                  # Spread width in dollars (int, > 0)
  min_iv_rank: 12                  # Minimum IV rank (int, 0-100)
  min_credit_pct: 8                # Min credit as % of spread width (float, 0-100)

  # Iron condor config
  iron_condor:
    enabled: true                  # Enable iron condors (bool)
    min_combined_credit_pct: 12    # Min combined credit % (float, 0-100)
    rsi_min: 30                    # RSI lower bound for neutral (int, 0-100)
    rsi_max: 70                    # RSI upper bound for neutral (int, 0-100)

  # Regime detector (REQUIRED if direction: both)
  regime_mode: combo_v2            # Options: none, ma200, combo_v2 (str)
  combo_regime:                    # Required if regime_mode: combo_v2
    ma_period: 200                 # Moving average period (int, > 0)
    rsi_period: 14                 # RSI period (int, > 0)
    vix_threshold: 40              # VIX panic threshold (float, > 0)
    cooldown_days: 10              # Hysteresis cooldown (int, >= 0)

# Risk management (REQUIRED)
risk:
  account_size: 100000             # Starting capital (float, > 0)
  max_risk_per_trade: 5.0          # Max risk % per trade (float, 0-100)
  max_positions: 50                # Max concurrent positions (int, > 0, < 100)
  max_positions_per_ticker: 2      # Max positions per ticker (int, > 0)
  profit_target: 50                # Profit target % of credit (float, > 0)
  stop_loss_multiplier: 3.5        # Stop-loss as credit multiplier (float, > 0)
  drawdown_cb_pct: 40              # Circuit breaker drawdown % (float, 0-100)
  max_portfolio_risk_pct: 40       # Max total capital at risk % (float, 0-100)
  enable_rolling: false            # Enable position rolling (bool)

# Position monitoring (REQUIRED)
position_monitor:
  enabled: true                    # Enable active monitoring (bool, MUST BE TRUE)
  check_interval: 5                # Check interval in minutes (int, >= 1)
  assignment_risk_dte: 3           # Close ITM positions at this DTE (int, >= 0)

# Alpaca configuration (REQUIRED — keys in .env)
alpaca:
  enabled: true                    # Enable paper trading (bool)
  paper: true                      # Paper trading mode (bool, MUST BE TRUE for validation)
  # API keys loaded from .env: ALPACA_API_KEY, ALPACA_API_SECRET

# Data provider (REQUIRED — keys in .env)
data:
  provider: polygon                # Options: polygon, tradier, yfinance (str)
  # API keys loaded from .env: POLYGON_API_KEY or TRADIER_API_KEY

# Alerts (OPTIONAL — keys in .env)
alerts:
  telegram:
    enabled: true                  # Enable Telegram alerts (bool)
    # Bot token/chat ID loaded from .env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# Database (OPTIONAL — defaults to portfolios/<id>/pilotai.db)
database:
  path: portfolios/exp_154/pilotai.db  # SQLite DB path (str)

# Logging (OPTIONAL)
logging:
  level: INFO                      # Log level (str: DEBUG, INFO, WARNING, ERROR)
  dir: portfolios/exp_154/logs     # Log directory (str)
```

### 10.2 Validation Rules

**Enforced by `PreFlightValidator`:**

| Field | Type | Validation | Error Message |
|-------|------|-----------|---------------|
| `tickers` | list[str] | Non-empty, all uppercase | "Tickers list empty or invalid" |
| `strategy.min_dte` | int | > 0, < max_dte | "min_dte must be > 0 and < max_dte" |
| `strategy.max_dte` | int | > min_dte | "max_dte must be > min_dte" |
| `strategy.manage_dte` | int | > 0, < min_dte | "manage_dte must be > 0 and < min_dte" |
| `strategy.spread_width` | int | > 0 | "spread_width must be > 0" |
| `strategy.min_credit_pct` | float | 0-100 | "min_credit_pct must be 0-100" |
| `risk.max_risk_per_trade` | float | 0-100 | "max_risk_per_trade must be 0-100" |
| `risk.max_positions` | int | > 0, < 100 | "max_positions must be 1-99" |
| `risk.profit_target` | float | > 0 | "profit_target must be > 0" |
| `risk.stop_loss_multiplier` | float | > 0 | "stop_loss_multiplier must be > 0" |
| `risk.drawdown_cb_pct` | float | 0-100 | "drawdown_cb_pct must be 0-100" |
| `risk.max_portfolio_risk_pct` | float | 0-100 | "max_portfolio_risk_pct must be 0-100" |
| `position_monitor.enabled` | bool | Must be `true` | "Position monitoring required for safety" |
| `position_monitor.check_interval` | int | >= 1 | "check_interval must be >= 1 minute" |
| `alpaca.enabled` | bool | — | (No constraint) |
| `alpaca.paper` | bool | Must be `true` | "Live trading not allowed (paper must be true)" |
| `database.path` | str | Writable, unique across portfolios | "DB path not writable or shared with another portfolio" |

### 10.3 .env Schema

**Required environment variables per portfolio:**

```bash
# portfolios/exp_154/.env

# Alpaca API (REQUIRED)
ALPACA_API_KEY=PK...              # Paper trading key
ALPACA_API_SECRET=...             # Paper trading secret
ALPACA_PAPER=true                 # Must be true

# Polygon.io API (REQUIRED if data.provider: polygon)
POLYGON_API_KEY=...               # Free or paid key

# Tradier API (OPTIONAL, if data.provider: tradier)
TRADIER_API_KEY=...

# Telegram (OPTIONAL, if alerts.telegram.enabled: true)
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Database path override (OPTIONAL — defaults to portfolios/<id>/pilotai.db)
PILOTAI_DB_PATH=portfolios/exp_154/pilotai.db
```

**Validation:**

- **Alpaca keys**: Must start with `PK` (paper key prefix)
- **Telegram token**: Format `\d+:[A-Za-z0-9_-]+`
- **Telegram chat ID**: Integer or `@channel_name`

---

## Appendix A: Migration from Current System

**How to migrate existing portfolios (exp_036, exp_059, exp_154) to new framework:**

### Step 1: Create Portfolio Entries

```bash
# Migrate exp_036
portfolio create \
  --id exp_036 \
  --name "Directional Compound" \
  --alpaca-account PA3QQDNADQU9 \
  --config config_exp036.yaml \
  --env .env.exp036 \
  --tags directional,compound,ma200

# Migrate exp_059
portfolio create \
  --id exp_059 \
  --name "Iron Condor Neutral" \
  --alpaca-account PA35U64WQBWM \
  --config config.yaml \
  --env .env \
  --tags ic_enabled,neutral,ma200

# Migrate exp_154
portfolio create \
  --id exp_154 \
  --name "ComboRegime IC Focus" \
  --alpaca-account PA3UNOV58WGK \
  --config config_exp154.yaml \
  --env .env.exp154 \
  --tags ic_only,regime_v2,conservative
```

### Step 2: Copy Existing Databases

```bash
# Copy existing SQLite DBs to portfolio directories
cp data/pilotai.db portfolios/exp_036/pilotai.db
cp data/pilotai_exp059.db portfolios/exp_059/pilotai.db
cp data/pilotai_exp154.db portfolios/exp_154/pilotai.db
```

### Step 3: Stop Old Processes

```bash
# Kill existing scanner processes
ps aux | grep "main.py scheduler" | awk '{print $2}' | xargs kill
```

### Step 4: Launch via New CLI

```bash
# Start all portfolios with validation
portfolio start exp_036 --validate
portfolio start exp_059 --validate
portfolio start exp_154 --validate

# Verify status
portfolio list
```

### Step 5: Start Position Monitor

```bash
# Start global position monitor daemon
nohup python scripts/position_monitor.py > logs/position_monitor.log 2>&1 &
```

**Rollback plan:** If migration fails, revert to old shell scripts (`run_exp154.sh`) and original DB paths.

---

## Appendix B: Future Enhancements

**Post-MVP features (not in initial 34h plan):**

1. **Portfolio cloning**: `portfolio clone exp_154 --id exp_155 --change risk.stop_loss_multiplier=4.0`
2. **A/B testing framework**: Run two portfolios with different configs, compare results
3. **Portfolio groups**: Tag portfolios with strategy families, bulk operations (`portfolio stop --group conservative`)
4. **Risk aggregation**: Calculate portfolio-level Greeks (delta, theta, vega) across all positions
5. **Automated parameter tuning**: ML-driven config optimization within safety bounds
6. **Multi-account support**: Manage multiple Alpaca accounts (real + paper) in one system
7. **Order execution analytics**: Track slippage, fill rates, time-to-fill per portfolio
8. **Trade replay**: Replay historical trades in backtester with exact portfolio config
9. **Disaster recovery**: One-click restore from backup (DB + config snapshots)
10. **Compliance reporting**: Audit trail export for regulatory review

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-06 | Claude (Subagent) | Initial design document |

---

**End of Portfolio Manager Framework Design Document**
