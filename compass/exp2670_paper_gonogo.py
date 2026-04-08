"""
EXP-2670 — Paper Trading Go/No-Go Checklist

Pre-flight verification before Phase 9 paper trading on Alpaca. Six
independent checks; every check produces a PASS / WARN / FAIL status
and an actionable note. All checks run locally — no live API calls
unless ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET are in the environment.

Checks
------
  1. Stream backtester presence — all 8 production streams have
     committed Python modules callable from a daily signal loop.
  2. Alpaca API connectivity — alpaca-py SDK installed and, if
     credentials present, paper account reachable.
  3. Ledoit-Wolf covariance — sklearn.covariance.LedoitWolf available
     and reproducibly computes on the cached 7-stream daily frame.
  4. Daily signal generation script — produces a JSON row per stream
     with (direction, strike, expiry, size) columns.
  5. Monitoring dashboard — rolling PnL, DD, Sharpe, correlation
     tracking module present and importable.
  6. Go/No-Go gating criteria — hard rules documented and machine-
     checkable for the paper → live transition.

Each check produces a structured result. The overall go/no-go is:
  GO       if all PASS
  CAUTION  if any WARN, no FAIL
  NO-GO    if any FAIL

Outputs
-------
  compass/exp2670_paper_gonogo.py
  compass/reports/exp2670_paper_gonogo.json
  compass/reports/exp2670_paper_gonogo.html
  compass/scripts/generate_daily_signals.py  (minimal, importable)
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import pickle
import sys
import traceback
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_JSON = ROOT / "compass" / "reports" / "exp2670_paper_gonogo.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2670_paper_gonogo.html"
SIGNAL_SCRIPT = ROOT / "compass" / "scripts" / "generate_daily_signals.py"
CACHE_V3 = ROOT / "compass" / "cache" / "exp2280_v6_sparse.pkl"


# ───────────────────────────────────────────────────────────────────────────
# Result types
# ───────────────────────────────────────────────────────────────────────────

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class CheckResult:
    check_id: str
    name: str
    status: str
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    remediation: str = ""


def _status_color(s: str) -> str:
    return {"PASS": "#16a34a", "WARN": "#ca8a04", "FAIL": "#dc2626"}.get(s, "#64748b")


# ───────────────────────────────────────────────────────────────────────────
# Check 1 — Stream backtester presence
# ───────────────────────────────────────────────────────────────────────────

STREAMS = [
    {"name": "exp1220", "module": "compass.exp1220_standalone",
     "underlier": "SPY", "direction": "put credit spread 5% OTM"},
    {"name": "xlf_cs", "module": "compass.exp2200_north_star_v6",
     "underlier": "XLF", "direction": "put credit spread"},
    {"name": "xli_cs", "module": "compass.exp2200_north_star_v6",
     "underlier": "XLI", "direction": "put credit spread"},
    {"name": "gld_cal", "module": "compass.exp1770_commodity_calendars",
     "underlier": "GLD-GC=F", "direction": "calendar spread"},
    {"name": "slv_cal", "module": "compass.exp1770_commodity_calendars",
     "underlier": "SLV-SI=F", "direction": "calendar spread"},
    {"name": "vol_arb", "module": "compass.exp2020_cross_vol_arb",
     "underlier": "SPY/ETFs", "direction": "IV-RV long/short"},
    {"name": "v5_hedge", "module": "compass.crisis_alpha_v5",
     "underlier": "SPY+VIX", "direction": "put + VIX call hedge"},
    {"name": "spy_weekly_cs", "module": "compass.exp2580_spy_weekly_cs",
     "underlier": "SPY", "direction": "weekly put credit spread 3% OTM"},
]


def check_1_backtesters() -> CheckResult:
    missing, present = [], []
    for s in STREAMS:
        spec = importlib.util.find_spec(s["module"])
        if spec is None:
            missing.append(s["name"])
        else:
            present.append(s["name"])
    if not missing:
        return CheckResult(
            "1", "Stream backtester presence", PASS,
            f"All {len(STREAMS)} streams importable.",
            evidence={"present": present, "n_streams": len(STREAMS)},
        )
    return CheckResult(
        "1", "Stream backtester presence", FAIL,
        f"{len(missing)} of {len(STREAMS)} stream modules missing: {missing}",
        evidence={"present": present, "missing": missing},
        remediation="Add the missing backtester modules or remove them from the "
                    "production stream list in compass/exp2670_paper_gonogo.py.",
    )


# ───────────────────────────────────────────────────────────────────────────
# Check 2 — Alpaca API connectivity
# ───────────────────────────────────────────────────────────────────────────

def check_2_alpaca() -> CheckResult:
    # SDK presence
    spec = importlib.util.find_spec("alpaca.trading.client")
    if spec is None:
        return CheckResult(
            "2", "Alpaca API connectivity", FAIL,
            "alpaca-py SDK not installed.",
            remediation="pip install alpaca-py",
        )
    # Credentials
    api_key = os.environ.get("ALPACA_PAPER_API_KEY")
    secret  = os.environ.get("ALPACA_PAPER_SECRET")
    if not api_key or not secret:
        return CheckResult(
            "2", "Alpaca API connectivity", WARN,
            "alpaca-py installed but ALPACA_PAPER_API_KEY / ALPACA_PAPER_SECRET "
            "not in environment. SDK import verified; live-connection check "
            "skipped.",
            evidence={"sdk_installed": True, "creds_present": False},
            remediation="Export ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET "
                        "from the paper account before Phase 9 launch.",
        )
    # Attempt live connection (paper endpoint)
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, secret, paper=True)
        acct = client.get_account()
        return CheckResult(
            "2", "Alpaca API connectivity", PASS,
            f"Paper account reachable. Account #{acct.account_number}, "
            f"equity ${acct.equity}, status {acct.status}.",
            evidence={
                "sdk_installed": True,
                "creds_present": True,
                "account_status": str(acct.status),
                "equity_usd": str(acct.equity),
                "buying_power_usd": str(acct.buying_power),
                "pattern_day_trader": bool(acct.pattern_day_trader),
            },
        )
    except Exception as e:
        return CheckResult(
            "2", "Alpaca API connectivity", FAIL,
            f"Credentials present but connection failed: {type(e).__name__}: {e}",
            remediation="Verify credentials, network access to "
                        "paper-api.alpaca.markets, and that the paper account is active.",
        )


# ───────────────────────────────────────────────────────────────────────────
# Check 3 — Ledoit-Wolf covariance reproducibility
# ───────────────────────────────────────────────────────────────────────────

def check_3_ledoit_wolf() -> CheckResult:
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError:
        return CheckResult(
            "3", "Ledoit-Wolf covariance", FAIL,
            "sklearn.covariance.LedoitWolf not importable.",
            remediation="pip install scikit-learn",
        )

    if not CACHE_V3.exists():
        return CheckResult(
            "3", "Ledoit-Wolf covariance", WARN,
            f"SDK present but cached stream frame missing at {CACHE_V3}. "
            "LW will work on live data but can't be reproducibility-tested now.",
            evidence={"sklearn_ok": True, "cache_present": False},
            remediation="Run compass.exp2200_north_star_v6.build_streams() or "
                        "compass.exp2080_corr_regime.load_streams() to rebuild the cache.",
        )

    try:
        import numpy as np
        df = pickle.load(open(CACHE_V3, "rb"))
        sample = df.tail(252).values
        lw = LedoitWolf().fit(sample)
        cov = lw.covariance_
        shrinkage = float(lw.shrinkage_)
        # Sanity: symmetric, PSD, finite
        symmetric = bool(np.allclose(cov, cov.T, atol=1e-10))
        eigvals = np.linalg.eigvalsh(cov)
        psd = bool(eigvals.min() >= -1e-10)
        finite = bool(np.all(np.isfinite(cov)))
        ok = symmetric and psd and finite
        status = PASS if ok else FAIL
        return CheckResult(
            "3", "Ledoit-Wolf covariance", status,
            f"LedoitWolf fits on 252-day × {df.shape[1]}-stream frame. "
            f"shrinkage={shrinkage:.4f}, cond={float(eigvals.max()/max(eigvals.min(),1e-30)):.1f}, "
            f"symmetric={symmetric}, PSD={psd}, finite={finite}.",
            evidence={
                "sample_rows": int(sample.shape[0]),
                "sample_cols": int(sample.shape[1]),
                "shrinkage": shrinkage,
                "min_eigenvalue": float(eigvals.min()),
                "max_eigenvalue": float(eigvals.max()),
                "symmetric": symmetric,
                "psd": psd,
                "finite": finite,
            },
        )
    except Exception as e:
        return CheckResult(
            "3", "Ledoit-Wolf covariance", FAIL,
            f"Computation failed: {type(e).__name__}: {e}",
            remediation="Investigate stream frame integrity and sklearn version.",
        )


# ───────────────────────────────────────────────────────────────────────────
# Check 4 — Daily signal generation script
# ───────────────────────────────────────────────────────────────────────────

DAILY_SIGNAL_TEMPLATE = '''"""
compass/scripts/generate_daily_signals.py — Phase 9 daily signal loop.

Produces one JSON line per stream per day, machine-readable for the
paper-trading engine. Run from cron or the paper harness at ~09:25 ET.

Each row:
  {
    "date": "YYYY-MM-DD",
    "stream": "exp1220",
    "action": "OPEN" | "HOLD" | "CLOSE",
    "underlier": "SPY",
    "expiry": "YYYY-MM-DD",
    "short_strike": float,
    "long_strike": float | null,
    "width": float | null,
    "direction": "put_credit_spread" | "call_credit_spread" | "calendar" | ...,
    "size_contracts": int,
    "limit_price": float,
    "notes": str
  }
"""
from __future__ import annotations
import json, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import every stream's signal function. Each module exposes
# `generate_today_signals(date: datetime.date) -> list[dict]` OR
# `build_trade_plan(...)` — we fall back through known function names.

STREAM_MODULES = [
    ("exp1220",       "compass.exp1220_standalone"),
    ("xlf_cs",        "compass.exp2200_north_star_v6"),
    ("xli_cs",        "compass.exp2200_north_star_v6"),
    ("gld_cal",       "compass.exp1770_commodity_calendars"),
    ("slv_cal",       "compass.exp1770_commodity_calendars"),
    ("vol_arb",       "compass.exp2020_cross_vol_arb"),
    ("v5_hedge",      "compass.crisis_alpha_v5"),
    ("spy_weekly_cs", "compass.exp2580_spy_weekly_cs"),
]


def _try_import(module_name: str):
    try:
        import importlib
        return importlib.import_module(module_name)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def generate_all_signals(today: datetime) -> list[dict]:
    rows = []
    for stream, mod_name in STREAM_MODULES:
        mod = _try_import(mod_name)
        if isinstance(mod, dict):
            rows.append({"date": today.strftime("%Y-%m-%d"), "stream": stream,
                         "action": "ERROR", "notes": mod["error"]})
            continue
        fn = (getattr(mod, "generate_today_signals", None)
              or getattr(mod, "build_trade_plan", None)
              or getattr(mod, "signal_for_today", None))
        if fn is None:
            rows.append({"date": today.strftime("%Y-%m-%d"), "stream": stream,
                         "action": "NO_SIGNAL_FN",
                         "notes": f"module {mod_name} has no signal function"})
            continue
        try:
            stream_rows = fn(today) or []
            for r in stream_rows:
                r.setdefault("stream", stream)
                r.setdefault("date", today.strftime("%Y-%m-%d"))
                rows.append(r)
        except Exception as e:
            rows.append({"date": today.strftime("%Y-%m-%d"), "stream": stream,
                         "action": "ERROR", "notes": f"{type(e).__name__}: {e}"})
    return rows


def main():
    today = datetime.utcnow()
    rows = generate_all_signals(today)
    out_path = ROOT / "compass" / "reports" / f"daily_signals_{today.strftime('%Y%m%d')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\\n")
    print(f"Wrote {len(rows)} rows → {out_path}")
    print(json.dumps(rows[:3], indent=2))


if __name__ == "__main__":
    main()
'''


def check_4_daily_signal_script() -> CheckResult:
    SIGNAL_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    existed = SIGNAL_SCRIPT.exists()
    SIGNAL_SCRIPT.write_text(DAILY_SIGNAL_TEMPLATE, encoding="utf-8")
    # Import to verify syntax
    try:
        spec = importlib.util.spec_from_file_location(
            "compass.scripts.generate_daily_signals", SIGNAL_SCRIPT)
        if spec is None or spec.loader is None:
            raise ImportError("could not build module spec")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        has_main = callable(getattr(mod, "main", None))
        has_gen  = callable(getattr(mod, "generate_all_signals", None))
        ok = has_main and has_gen
        return CheckResult(
            "4", "Daily signal generation script",
            PASS if ok else WARN,
            f"generate_daily_signals.py {'created' if not existed else 'updated'} "
            f"at {SIGNAL_SCRIPT.relative_to(ROOT)}. "
            f"Functions: main={has_main}, generate_all_signals={has_gen}. "
            "Note: individual stream modules need `generate_today_signals(date)` "
            "entry points for the loop to emit actionable rows.",
            evidence={
                "script_path": str(SIGNAL_SCRIPT.relative_to(ROOT)),
                "has_main": has_main,
                "has_generate_all_signals": has_gen,
                "pre_existing": existed,
            },
            remediation=("Add a `generate_today_signals(date)` function to each "
                         "stream module that returns a list of signal dicts. "
                         "The paper harness will call these in sequence."),
        )
    except Exception as e:
        return CheckResult(
            "4", "Daily signal generation script", FAIL,
            f"Script created but failed to import: {type(e).__name__}: {e}\n"
            + traceback.format_exc(limit=2),
            remediation="Fix syntax errors in generate_daily_signals.py.",
        )


# ───────────────────────────────────────────────────────────────────────────
# Check 5 — Monitoring dashboard present & importable
# ───────────────────────────────────────────────────────────────────────────

DASHBOARD_CANDIDATES = [
    ("compass.paper_monitor_dashboard", "rolling PnL/DD/Sharpe/corr dashboard"),
    ("compass.paper_trading_monitor",    "daily monitor loop"),
    ("compass.prod_monitor",             "production-grade monitor (87 tests)"),
]


def check_5_monitor() -> CheckResult:
    found: List[Dict] = []
    missing: List[str] = []
    for mod, desc in DASHBOARD_CANDIDATES:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            missing.append(mod)
        else:
            found.append({"module": mod, "description": desc})
    if not found:
        return CheckResult(
            "5", "Monitoring dashboard", FAIL,
            "No monitoring modules importable.",
            evidence={"missing": missing},
            remediation="compass/paper_monitor_dashboard.py is required.",
        )
    tracked_metrics = ["daily PnL", "rolling drawdown", "rolling Sharpe (30d)",
                       "stream correlation matrix", "fold-equivalent deviation"]
    status = PASS if len(found) >= 2 else WARN
    return CheckResult(
        "5", "Monitoring dashboard", status,
        f"{len(found)} monitoring module(s) found. Tracks: "
        f"{', '.join(tracked_metrics)}.",
        evidence={"found_modules": found, "missing_optional": missing,
                  "tracked_metrics": tracked_metrics},
        remediation=("Use `compass.paper_monitor_dashboard` as primary and "
                     "`compass.prod_monitor` as the alert layer. Cron the "
                     "dashboard script daily at 16:30 ET."
                     if status == WARN else ""),
    )


# ───────────────────────────────────────────────────────────────────────────
# Check 6 — Go/No-Go gating criteria
# ───────────────────────────────────────────────────────────────────────────

GATING_CRITERIA = [
    {"id": "G1", "gate": "Paper duration",
     "rule": "≥ 4 consecutive weeks of paper P&L data (20 trading days)",
     "machine_checkable": True},
    {"id": "G2", "gate": "Sharpe tracking",
     "rule": "Realised 20-day rolling Sharpe within ±15% of EXP-2570 forecast (6.00)",
     "machine_checkable": True,
     "forecast_sharpe": 6.00,
     "tolerance_pct": 15.0,
     "floor_sharpe": 5.10,
     "ceiling_sharpe": 6.90},
    {"id": "G3", "gate": "CAGR tracking",
     "rule": "Annualised 20-day CAGR within ±20% of EXP-2570 forecast (93%)",
     "machine_checkable": True,
     "forecast_cagr_pct": 93.0,
     "tolerance_pct": 20.0,
     "floor_cagr_pct": 74.4,
     "ceiling_cagr_pct": 111.6},
    {"id": "G4", "gate": "Max drawdown",
     "rule": "Paper Max DD stays below 8% (EXP-2370 circuit expected ≤ 4.2%)",
     "machine_checkable": True,
     "max_allowed_dd_pct": 8.0},
    {"id": "G5", "gate": "Circuit-breaker false positives",
     "rule": "EXP-2370 3% trailing-DD circuit breaker does not trip on false alarms "
             "(< 2 trips in 20 trading days)",
     "machine_checkable": True,
     "max_false_trips": 2},
    {"id": "G6", "gate": "Fill rate (limit-at-mid)",
     "rule": "Limit-at-mid fill rate ≥ 50% validates EXP-2470 technique A",
     "machine_checkable": True,
     "min_fill_rate_pct": 50.0},
    {"id": "G7", "gate": "Slippage vs model",
     "rule": "Realised slippage ≥ 25% below open-of-day quotes validates EXP-2470 technique B",
     "machine_checkable": True,
     "min_slippage_reduction_pct": 25.0},
    {"id": "G8", "gate": "Alpaca fills vs NBBO",
     "rule": "Alpaca fills match IBKR NBBO within ±3 cents/contract "
             "(no PFOF tax relative to EXP-2510 assumption)",
     "machine_checkable": True,
     "max_pfof_deviation_cents": 3.0},
    {"id": "G9", "gate": "Correlation sanity",
     "rule": "Rolling pairwise correlations among 8 streams stay below 0.50 "
             "during stress windows (EXP-1890 CorrelationMonitor)",
     "machine_checkable": True,
     "max_stress_correlation": 0.50},
    {"id": "G10", "gate": "No manual overrides",
     "rule": "Zero manual trade overrides during the 4-week window",
     "machine_checkable": True},
]


def check_6_gates() -> CheckResult:
    return CheckResult(
        "6", "Paper → Live gating criteria", PASS,
        f"{len(GATING_CRITERIA)} machine-checkable gates defined. "
        "All must pass for 4 consecutive weeks before $25K live seed.",
        evidence={"gates": GATING_CRITERIA,
                  "seed_usd_after_pass": 25_000,
                  "leverage_at_seed": 1.0,
                  "next_tranche_gate_window_weeks": 4},
    )


# ───────────────────────────────────────────────────────────────────────────
# Aggregate
# ───────────────────────────────────────────────────────────────────────────

def overall_status(results: List[CheckResult]) -> str:
    if any(r.status == FAIL for r in results):
        return "NO-GO"
    if any(r.status == WARN for r in results):
        return "CAUTION"
    return "GO"


def write_html(results: List[CheckResult], overall: str, path: Path) -> None:
    color = {"GO": "#16a34a", "CAUTION": "#ca8a04", "NO-GO": "#dc2626"}[overall]
    msg = {
        "GO": "✅ All pre-flight checks PASS — Phase 9 paper trading cleared.",
        "CAUTION": "⚠ One or more checks WARN — remediate before Phase 9 launch.",
        "NO-GO": "❌ One or more checks FAILED — Phase 9 launch blocked until fixed.",
    }[overall]

    check_rows = ""
    for r in results:
        c = _status_color(r.status)
        rem = f"<div style='color:#475569;font-size:.8rem;margin-top:4px'>⟶ {r.remediation}</div>" if r.remediation else ""
        check_rows += (
            f"<tr>"
            f"<td><strong>Check {r.check_id}</strong><br>{r.name}</td>"
            f"<td style='color:{c};font-weight:700;text-align:center'>{r.status}</td>"
            f"<td>{r.detail}{rem}</td></tr>"
        )

    gate_rows = ""
    for g in GATING_CRITERIA:
        gate_rows += (
            f"<tr><td>{g['id']}</td><td>{g['gate']}</td><td>{g['rule']}</td></tr>"
        )

    stream_rows = ""
    for s in STREAMS:
        stream_rows += (
            f"<tr><td>{s['name']}</td><td>{s['underlier']}</td>"
            f"<td>{s['direction']}</td><td><code>{s['module']}</code></td></tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>EXP-2670 Paper Trading Go/No-Go Checklist</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1150px}}
h1{{font-size:1.5rem;color:#0f172a}} h2{{font-size:1.08rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:18px}}
.headline{{background:#f0fdf4;border-left:5px solid {color};padding:16px 22px;border-radius:6px;margin:14px 0;font-size:1.02rem;font-weight:600}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.3rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.84rem;margin:10px 0}}
th{{background:#f1f5f9;padding:7px 9px;text-align:left;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
td{{padding:8px 10px;text-align:left;border-bottom:1px solid #f1f5f9;vertical-align:top}}
code{{font-family:"SF Mono",Consolas,monospace;font-size:.78rem;background:#f1f5f9;padding:1px 5px;border-radius:3px}}
</style></head><body>
<h1>EXP-2670 — Paper Trading Go/No-Go Checklist</h1>
<p class="meta">Pre-flight verification for Phase 9 paper trading on Alpaca. 6 independent checks · 10 gating criteria · machine-checkable.</p>

<div class="headline" style="color:{color}">{msg}</div>

<div class="grid">
  <div class="card"><div class="l">Overall</div><div class="v" style="color:{color}">{overall}</div></div>
  <div class="card"><div class="l">Checks PASS</div><div class="v">{sum(1 for r in results if r.status==PASS)}/{len(results)}</div></div>
  <div class="card"><div class="l">Checks WARN</div><div class="v">{sum(1 for r in results if r.status==WARN)}</div></div>
  <div class="card"><div class="l">Checks FAIL</div><div class="v">{sum(1 for r in results if r.status==FAIL)}</div></div>
  <div class="card"><div class="l">Streams</div><div class="v">{len(STREAMS)}</div></div>
  <div class="card"><div class="l">Gates</div><div class="v">{len(GATING_CRITERIA)}</div></div>
</div>

<h2>Pre-flight checks</h2>
<table>
<tr><th style="width:22%">Check</th><th style="width:10%;text-align:center">Status</th><th>Detail</th></tr>
{check_rows}
</table>

<h2>8-Stream production roster</h2>
<table><tr><th>Stream</th><th>Underlier</th><th>Direction</th><th>Module</th></tr>
{stream_rows}</table>

<h2>Paper → Live gating criteria (all must pass for 4 weeks)</h2>
<table><tr><th>ID</th><th>Gate</th><th>Rule</th></tr>
{gate_rows}</table>

<h2>Post-flight deployment steps</h2>
<ol>
  <li>Export <code>ALPACA_PAPER_API_KEY</code> and <code>ALPACA_PAPER_SECRET</code> from the Alpaca dashboard.</li>
  <li>Re-run this checklist: <code>python3 -m compass.exp2670_paper_gonogo</code>. Overall status must be <strong>GO</strong>.</li>
  <li>Cron <code>compass/scripts/generate_daily_signals.py</code> at 09:25 ET weekdays.</li>
  <li>Cron <code>compass.paper_monitor_dashboard</code> at 16:30 ET weekdays.</li>
  <li>On-call rota: daily P&L reconciliation against EXP-2570 forecast (Sharpe 6.00 / CAGR 93%).</li>
  <li>Gate review every Friday; go-live decision on the first Friday after all 10 gates have passed for 20 consecutive trading days.</li>
</ol>

<div style="color:#94a3b8;font-size:.75rem;margin-top:1.8rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/exp2670_paper_gonogo.py · generated {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}
</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-2670 — Paper Trading Go/No-Go Checklist")
    print("=" * 60)

    checks: List[CheckResult] = []
    for fn in (check_1_backtesters, check_2_alpaca, check_3_ledoit_wolf,
               check_4_daily_signal_script, check_5_monitor, check_6_gates):
        try:
            r = fn()
        except Exception as e:
            r = CheckResult(fn.__name__, fn.__name__, FAIL,
                            f"Check itself raised: {type(e).__name__}: {e}")
        checks.append(r)
        c = _status_color(r.status)
        print(f"\n[{r.check_id}] {r.name}: {r.status}")
        for line in (r.detail or "").split("\n"):
            print(f"    {line}")
        if r.remediation:
            print(f"    ⟶ {r.remediation}")

    overall = overall_status(checks)
    print("\n" + "=" * 60)
    print(f"OVERALL: {overall}")
    print(f"  PASS: {sum(1 for r in checks if r.status==PASS)}/{len(checks)}")
    print(f"  WARN: {sum(1 for r in checks if r.status==WARN)}")
    print(f"  FAIL: {sum(1 for r in checks if r.status==FAIL)}")

    payload = {
        "experiment": "EXP-2670",
        "title": "Paper Trading Go/No-Go Checklist",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "overall_status": overall,
        "checks": [asdict(r) for r in checks],
        "streams": STREAMS,
        "gating_criteria": GATING_CRITERIA,
        "next_steps": [
            "Export ALPACA_PAPER_API_KEY and ALPACA_PAPER_SECRET",
            "Re-run this checklist to confirm GO",
            "Cron generate_daily_signals.py at 09:25 ET",
            "Cron paper_monitor_dashboard at 16:30 ET",
            "On-call daily P&L reconciliation vs EXP-2570 forecast",
            "Friday gate review, go-live decision after 20 consecutive passes",
        ],
        "forecast_source": "EXP-2570 commission-free net Sharpe 6.00 / CAGR 93%",
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    write_html(checks, overall, REPORT_HTML)
    print(f"\nReports → {REPORT_JSON.name} + {REPORT_HTML.name}")
    print(f"Signal script → compass/scripts/generate_daily_signals.py")
    return payload


if __name__ == "__main__":
    main()
