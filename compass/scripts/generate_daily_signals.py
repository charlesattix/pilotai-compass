"""
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

# EXP-2690: 8-stream signal registry.
#
# Each entry is (stream_id, source_module, attribute_name). The driver
# tries the source_module's attribute first, then falls back to the
# central registry in compass.exp2690_signal_generators. This gives TWO
# valid entry points per stream: (a) per-module generate_today_signals
# (added by EXP-2690 as a delegating stub on each source file), and
# (b) the central registry's per-sleeve function. The driver uses the
# central registry so that multi-ticker modules (exp2160, exp1770)
# emit distinct per-sleeve rows in the daily log.
STREAM_MODULES = [
    ("exp1220",  "compass.exp2690_signal_generators", "exp1220_signals"),
    ("xlf_cs",   "compass.exp2690_signal_generators", "xlf_cs_signals"),
    ("xli_cs",   "compass.exp2690_signal_generators", "xli_cs_signals"),
    ("qqq_cs",   "compass.exp2690_signal_generators", "qqq_cs_signals"),
    ("gld_cal",  "compass.exp2690_signal_generators", "gld_cal_signals"),
    ("slv_cal",  "compass.exp2690_signal_generators", "slv_cal_signals"),
    ("vol_arb",  "compass.exp2690_signal_generators", "vol_arb_signals"),
    ("v5_hedge", "compass.exp2690_signal_generators", "v5_hedge_signals"),
]


def _try_import(module_name: str):
    try:
        import importlib
        return importlib.import_module(module_name)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def generate_all_signals(today: datetime) -> list[dict]:
    rows = []
    for stream, mod_name, attr_name in STREAM_MODULES:
        mod = _try_import(mod_name)
        if isinstance(mod, dict):
            rows.append({"date": today.strftime("%Y-%m-%d"), "stream": stream,
                         "action": "ERROR", "notes": mod["error"]})
            continue
        fn = getattr(mod, attr_name, None)
        if fn is None:
            # Fallback to the common generate_today_signals entry point
            # (per-source-module stubs added by EXP-2690)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None,
                        help="ISO date to generate signals for (default: today UTC)")
    args = parser.parse_args()

    if args.date:
        today = datetime.strptime(args.date[:10], "%Y-%m-%d")
    else:
        today = datetime.utcnow()

    rows = generate_all_signals(today)
    out_path = ROOT / "compass" / "reports" / f"daily_signals_{today.strftime('%Y%m%d')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"Wrote {len(rows)} rows → {out_path}")
    # Summary per-stream
    by_stream = {}
    for r in rows:
        s = r.get("stream", "?")
        by_stream.setdefault(s, []).append(r.get("action", "?"))
    print("\nPer-stream summary:")
    for s, actions in by_stream.items():
        print(f"  {s:12s}  {len(actions):3d} signals  "
              f"{', '.join(sorted(set(actions)))}")


if __name__ == "__main__":
    main()
