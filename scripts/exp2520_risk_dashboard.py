"""
scripts/exp2520_risk_dashboard.py — EXP-2520 Risk Dashboard
============================================================

Live risk dashboard for the EXP-2520 paper-trading deployment. Reads
the paper engine's state.json, the monitor's health.json, and the
latest 7-stream weights from the Ledoit-Wolf allocator, then renders
an HTML page at reports/exp2520/risk_dashboard.html and a terse JSON
snapshot alongside.

Runs in two modes:
  --once    render a single snapshot and exit
  --loop    re-render every 2 minutes until killed

The dashboard is read-only — it submits nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("exp2520_dashboard")


# ═══════════════════════════════════════════════════════════════════════════
def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _render_html(snapshot: Dict, out: Path) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    h = snapshot.get("health", {})
    s = snapshot.get("state", {})
    weights = s.get("last_weights", {})
    streams = list(weights.keys())
    cb_state = h.get("circuit_breaker_state", "unknown")
    cb_color = {"OK": "#0a7a0a", "SOFT_REDUCE": "#b86b00",
                "HARD_HALT": "#b80000"}.get(cb_state, "#555")

    rows_w = "".join(
        f"<tr><td>{k}</td><td>{weights.get(k, 0):.4f}</td></tr>"
        for k in streams
    )
    rows_a = "".join(
        f"<tr><td>{a.get('level', '?')}</td><td>{a.get('code', '?')}</td>"
        f"<td>{a.get('msg', '')}</td></tr>"
        for a in h.get("alerts", [])
    ) or "<tr><td colspan='3' class='small'>no active alerts</td></tr>"

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta http-equiv='refresh' content='120'>
<title>EXP-2520 — Risk Dashboard</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5;background:#fff}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em;margin-top:0}}
 h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92em}}
 th,td{{border:1px solid #bbb;padding:6px 9px;text-align:left}}
 th{{background:#f0f0f0}}
 .bignum{{font-size:1.5em;font-weight:600}}
 .small{{color:#666;font-size:.88em}}
 .state{{display:inline-block;padding:2px 8px;border-radius:6px;color:white;font-weight:600}}
</style></head><body>
<h1>EXP-2520 — Paper-Trading Risk Dashboard</h1>
<p class='small'>Snapshot {now} · auto-refresh 2m · read-only.</p>

<h2>Circuit breaker</h2>
<p><span class='state' style='background:{cb_color}'>{cb_state}</span>
  &nbsp;trailing DD <b>{h.get('trailing_dd_pct', 0):.2f}%</b>
  &nbsp;(soft 3% / hard 6%)</p>
<p>Last action: <code>{h.get('circuit_breaker_action', 'none')}</code></p>

<h2>Equity & leverage</h2>
<table>
<tr><th>Equity</th><td class='bignum'>${h.get('equity', 0):,.0f}</td></tr>
<tr><th>Rolling peak</th><td>${h.get('rolling_peak', 0):,.0f}</td></tr>
<tr><th>Leverage</th><td>{h.get('leverage', 1.0):.2f}× (cap 13×)</td></tr>
<tr><th>Scale factor (vol target)</th><td>{h.get('scale_factor', 1.0):.3f}</td></tr>
<tr><th>Last scale refit</th><td>{h.get('last_scale_refit', 'unknown')}</td></tr>
<tr><th>VIX last</th><td>{h.get('vix_last', 0):.2f}</td></tr>
<tr><th>Open positions</th><td>{h.get('n_open_positions', 0)}</td></tr>
</table>

<h2>Current Ledoit-Wolf risk-parity weights</h2>
<table><tr><th>Stream</th><th>Weight</th></tr>{rows_w}</table>

<h2>Active alerts</h2>
<table><tr><th>Level</th><th>Code</th><th>Message</th></tr>{rows_a}</table>

<h2>Configuration</h2>
<p class='small'>Config: <code>{snapshot.get('config_path', '?')}</code> ·
  Allocator: Ledoit-Wolf risk-parity (EXP-2360) ·
  Vol target 15% / scale cap 13× (EXP-2340) ·
  3% trailing DD breaker (EXP-2370) ·
  T+V entry overlay on exp1220 sleeve only (EXP-2120).</p>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)


def render(config_path: Path, out_html: Path, out_json: Path) -> None:
    cfg = yaml.safe_load(open(config_path))
    health = _load_json(Path(cfg.get("monitoring", {})
                              .get("health_file", "logs/exp2520/health.json")))
    state  = _load_json(Path(cfg.get("monitoring", {})
                              .get("state_file",  "logs/exp2520/state.json")))
    snapshot = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "health": health,
        "state":  state,
    }
    _render_html(snapshot, out_html)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(snapshot, indent=2, default=str))
    LOG.info("wrote %s and %s", out_html, out_json)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp2410_production_paper.yaml")
    ap.add_argument("--html", default="reports/exp2520/risk_dashboard.html")
    ap.add_argument("--json", default="reports/exp2520/risk_dashboard.json")
    ap.add_argument("--log-file", default="logs/exp2520/dashboard.log")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        handlers=[logging.FileHandler(args.log_file),
                                  logging.StreamHandler()])

    cfg_path = Path(args.config)
    out_html = Path(args.html)
    out_json = Path(args.json)
    if args.once or not args.loop:
        render(cfg_path, out_html, out_json)
        return 0

    while True:
        try:
            render(cfg_path, out_html, out_json)
        except Exception as e:
            LOG.exception("dashboard render failed: %s", e)
        time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())
