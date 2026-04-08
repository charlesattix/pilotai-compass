# Architecture Proposal Review — Maximus Cruz Proposal (March 13, 2026)
**Reviewer:** Claude
**Date:** 2026-03-13
**Verdict: Mostly wrong diagnosis. Proposal is ~60% overengineered. Two real problems worth fixing.**

---

## CRITICAL FINDING: P2 Is Based on a False Premise

This is the most important thing in this review.

**Maximus says:**
> All experiments share the same Alpaca paper account. The PositionMonitor in EXP-400 might try to manage EXP-401's positions. PositionReconciler syncs ALL Alpaca positions into the local DB — cross-contaminating everything.

**Reality:** Every experiment already has its own Alpaca account with separate API credentials.

```
.env.champion → ALPACA_API_KEY=PK4SGNFT3BGN...  → Account PA3D6UPXF5F2
.env.exp401   → ALPACA_API_KEY=PKJKRGVQZIA6...  → Account PA3Y2XDYB9I3
.env.exp305   → ALPACA_API_KEY=PKSPAM5732NK...  → Account PA3W9FZKK6XD
.env.exp059   → ALPACA_API_KEY=PK6URS6OBCS...   → Account PA3LP867WNGU
.env.exp154   → ALPACA_API_KEY=PKANAYVKHZX...   → Account PA3UNOV58WGK
```

Five distinct accounts. Five distinct API keys. One account per experiment — a rule Carlos has been following from the start.

The entire `client_order_id` tagging scheme, the `PositionMonitor` filtering logic, and the `PositionReconciler` filtering are **solutions to a problem that does not exist**. Each experiment's reconciler already talks to its own Alpaca account and can only see its own positions. There is zero cross-contamination risk.

The proposal spends its most detailed technical section (Section 6, "Isolation Model") solving the wrong problem. And implementing it would mean modifying `execution_engine.py`, `position_monitor.py` (1,121 lines), and `reconciler.py` (416 lines) — live trading components — for zero benefit.

---

## Problem-by-Problem Verdict

### P1 — No Experiment Registry
**STATUS: REAL — AND ALREADY FIXED TODAY**

This was a genuine gap as of this morning. It was fixed by:
- `experiments.yaml` at the project root: maps every dimension (name ↔ env file ↔ config ↔ account ↔ tmux ↔ DB ↔ status ↔ start date)
- `scripts/portfolio_status.py`: reads the registry, checks tmux sessions live, pings all Alpaca accounts, shows equity/positions/P&L

Maximus's proposed replacement — a SQLite `experiment_registry.db` with an `experiments` and `experiment_events` table — is heavier and harder to edit by hand. The YAML file is human-readable and version-controlled. The SQLite approach requires `pilotctl` to modify it; the YAML approach works with any text editor.

**Verdict: solved. Don't build the SQLite registry.**

---

### P2 — Alpaca Account Collision
**STATUS: DOES NOT EXIST**

See above. All accounts are already separate. The proposal's diagnosis is wrong.

**Verdict: do not implement order tagging. It solves a phantom problem and touches live trading code.**

---

### P3 — No Process Orchestration
**STATUS: REAL, BUT PROPOSED SOLUTION IS OVERENGINEERED**

The problem is real: there's no way to start all experiments together, restart crashed ones, or show process status beyond `tmux ls`. But the solution doesn't need to be `pilotctl` with a `ProcessSupervisor` class and exponential backoff restart logic.

The actual need for a 2-5 experiment system:
- `pilotctl start exp400 exp401` → runs the two `tmux new` + `python main.py` commands
- `pilotctl stop exp400` → runs `tmux kill-session`
- `pilotctl status` → already solved by `portfolio_status.py`

A 50-line shell script handles this. A Python `ProcessSupervisor` with auto-restart is actually **dangerous** for a live trading system: if `position_monitor.py` crashes during a VIX spike because of an API timeout, auto-restart means the system immediately re-enters the market without anyone reviewing what happened. You want the system to page you and wait.

**Verdict: a thin `pilotctl.py` wrapper (50-100 lines) around tmux commands would be enough. Skip the process supervisor.**

---

### P4 — No Centralized Comparison Dashboard
**STATUS: REAL, BUT PREMATURE**

Both active experiments (exp400 and exp401) started today. exp400 has 0 trades. exp401 has 5 trades. There is nothing meaningful to compare yet. The `ComparisonEngine` proposed would read two empty/near-empty databases and produce a table showing `$0 vs $5`.

The comparison engine is a reasonable future need. It does not need to be a 500-line `orchestrator/comparison_engine.py` — it can be `scripts/compare_experiments.py`, a 50-line script that opens each experiment's SQLite DB, runs a query, and prints a table.

The **daily digest via Telegram** is a genuinely useful feature, but Telegram isn't even configured right now (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are empty strings in `.env`). Building an `ExperimentAlertRouter` for a notification channel that isn't connected is backwards.

**Verdict: revisit in 2-4 weeks when there are actual trades to compare. Build a script, not a package.**

---

### P5 — Configuration Drift
**STATUS: MINOR REAL RISK, WRONG SOLUTION**

Config drift (200-line YAML files manually maintained) is a real risk. But `scripts/validate_params.py` already exists (448 lines) and validates configs. The config factory + template + override system is elegant in theory but introduces a new problem: generated YAML files in `configs/generated/` that are harder to read and audit than hand-written ones.

The actual risk: someone edits `paper_champion.yaml` and accidentally changes `profit_target: 55` to `profit_target: 0.55` (wrong unit). That would be caught by better validation in `validate_params.py`, not by a config factory.

**Verdict: extend `validate_params.py` to catch unit errors and warn on values that diverge from backtest. Don't build a config factory.**

---

### P6 — Heartbeat & Healthcheck Collision
**STATUS: PARTIALLY REAL, SMALLER THAN CLAIMED**

**Heartbeat (real problem):** The heartbeat path is hardcoded at module load time:
```python
# shared/scheduler.py line 34
_HEARTBEAT_PATH = Path(os.environ.get("DATA_DIR", "data")) / "heartbeat.json"
```
If exp400 and exp401 run without a `DATA_DIR` env var, they both write to `data/heartbeat.json`. Each overwrites the other. Fix: one line in each `.env.*` file: `DATA_DIR=data/exp400` and `DATA_DIR=data/exp401`. This is a 2-minute fix.

**Healthcheck HTTP server (phantom problem):** `shared/healthcheck.py` exists with port 8080 hardcoded. But it is **never instantiated** in `main.py` or `scheduler.py`. The comment in `scheduler.py` at line 102 mentions it as a future capability, not a current one. There is no port collision because the server is never started.

**Verdict: fix `DATA_DIR` per experiment env file (2 minutes). Skip the healthcheck multiplexer.**

---

## Problems Maximus MISSED

These are real issues not mentioned in the proposal:

**1. Commission formula bug (uncommitted fix)**
`execution/execution_engine.py` does NOT have the `* contracts` commission fix in any committed commit. The fix is in the working tree only. This is the most pressing actual issue.

**2. Heartbeat DATA_DIR not set in env files**
The two active experiments both have empty `DATA_DIR` → both write to the same `data/heartbeat.json`. Fix: add `DATA_DIR=data/exp400` to `.env.champion` and `DATA_DIR=data/exp401` to `.env.exp401`.

**3. Telegram is not configured**
Both `.env.champion` and `.env.exp401` have `TELEGRAM_BOT_TOKEN=` (empty). Telegram alerts are silently disabled for both active experiments. When a position hits stop-loss at 2 PM, nobody gets notified. This should be wired up before comparing experiment architectures.

**4. No `scripts/compare_experiments.py` yet**
Useful once there are trades. Takes 30 minutes to build as a script.

---

## The Core Overengineering Problem

The proposal asks for a 6-day engineering project to build:
- `pilotctl.py`
- `orchestrator/experiment_manager.py`
- `orchestrator/experiment_registry.py`
- `orchestrator/config_factory.py`
- `orchestrator/process_supervisor.py`
- `orchestrator/comparison_engine.py`
- `orchestrator/alert_router.py`
- Modifications to `execution_engine.py`, `position_monitor.py`, `reconciler.py`

For a system with **2 active experiments**, both started today, running on a laptop in tmux.

The analogy is: you have two pizza ovens in a restaurant kitchen. They each have their own gas line, their own timer, their own oven temperature dial. Maximus is proposing to build a centralized oven orchestration platform with auto-restart capability, a config factory to generate oven temperature settings from templates, and a shared order-tagging system to prevent the ovens from cooking each other's pizzas — even though the ovens are in separate rooms and have never seen each other's food.

The correct response to 2 well-isolated processes is not 7 new Python modules.

---

## What's Actually Worth Building (MVI)

In priority order:

| Priority | Work | Effort | Benefit |
|---|---|---|---|
| 🔴 NOW | Commit the commission `* contracts` fix | 5 min | Fixes active paper trade P&L calculation |
| 🔴 NOW | Add `DATA_DIR=data/expNNN` to `.env.champion` and `.env.exp401` | 5 min | Fixes heartbeat collision |
| 🔴 NOW | Wire up Telegram (put real tokens in env files) | 30 min | Actually get notified of trades |
| 🟡 SOON | `scripts/compare_experiments.py` — reads both DBs, prints table | 1 hr | Useful once trades accumulate |
| 🟡 SOON | `pilotctl.py` — thin wrapper for start/stop/restart via tmux | 2 hr | Convenience, not necessity |
| 🟢 LATER | Telegram experiment name prefix (5-line change) | 30 min | Cleaner notifications |
| 🟢 LATER | Extend `validate_params.py` to check unit consistency | 2 hr | Guards against config drift |
| ⬜ SKIP | SQLite experiment registry | — | experiments.yaml does the job |
| ⬜ SKIP | `client_order_id` tagging / position filtering | — | Accounts are already separate |
| ⬜ SKIP | ProcessSupervisor with auto-restart | — | Dangerous for live trading |
| ⬜ SKIP | Config factory / template system | — | Adds complexity, solved by validate_params |
| ⬜ SKIP | Healthcheck multiplexer | — | Server not running; not needed |
| ⬜ SKIP | Weekly HTML comparison report | — | Premature; build after 3+ months of data |

**Total MVI effort: ~4 hours.** Not 6 days.

---

## On Maximus's Architecture Skills

To be fair: the proposal is well-structured, clearly written, and the individual components (`pilotctl`, process lifecycle, comparison engine) are genuinely useful concepts for a large multi-experiment system. The design is appropriate for a system running 20+ experiments, or one that needs to be handed off to a team.

For this system — 2 live experiments on a single machine managed by one person with full system access — the overhead of the orchestrator layer exceeds its benefit. The correct time to build this is when you're running 5+ experiments simultaneously and spending more than 10 minutes per day managing them manually.

The biggest error in the proposal is the Alpaca account sharing diagnosis. That false premise drives the most invasive part of the implementation (modifying execution_engine.py, position_monitor.py, reconciler.py). Getting that wrong means the riskiest work was entirely unnecessary.

---

*All claims in this review verified against: `.env.*` credential files (5 separate API keys confirmed), `execution/execution_engine.py` (commission formula), `shared/scheduler.py` (heartbeat path), `shared/healthcheck.py` (never instantiated), `shared/reconciler.py` (account-scoped), `data/pilotai_*.db` (0 and 5 trades in active DBs), today's `experiments.yaml` and `scripts/portfolio_status.py` (P1 already solved).*
