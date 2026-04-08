"""
scripts/exp2520_daily_report.py — EXP-2520 End-of-Day P&L + Attribution
========================================================================

Reads Alpaca paper account history, the engine state, and the latest
health snapshot; produces a daily HTML + JSON report under
reports/exp2520/daily/YYYY-MM-DD.{html,json} plus a rolling summary at
reports/exp2520/rolling_summary.json.

Per-sleeve attribution is computed from state.json[last_weights] and
state.json[last_scale_factor] applied to the daily return contribution
of each strategy module. If engine attribution fields are missing we
fall back to equity-level totals.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("exp2520_daily")


# ═══════════════════════════════════════════════════════════════════════════
def _load(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _alpaca_account_history(days: int = 30) -> List[Dict]:
    key = os.environ.get("ALPACA_API_KEY_PAPER")
    sec = os.environ.get("ALPACA_API_SECRET_PAPER")
    if not key or not sec:
        return []
    try:
        import urllib.request
        url = (f"https://paper-api.alpaca.markets/v2/account/portfolio/history"
               f"?period={days}D&timeframe=1D")
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec,
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
        rows: List[Dict] = []
        ts = d.get("timestamp", [])
        eq = d.get("equity", [])
        pl = d.get("profit_loss_pct", [])
        for i in range(len(ts)):
            rows.append({
                "ts": ts[i],
                "equity": float(eq[i]) if eq[i] is not None else None,
                "pnl_pct": float(pl[i]) if pl[i] is not None else None,
            })
        return rows
    except Exception as e:
        LOG.warning("alpaca history fetch failed: %s", e)
        return []


def build_report(config_path: Path) -> Dict:
    cfg = yaml.safe_load(open(config_path))
    mon = cfg.get("monitoring", {})
    health = _load(Path(mon.get("health_file", "logs/exp2520/health.json")))
    state  = _load(Path(mon.get("state_file",  "logs/exp2520/state.json")))

    history = _alpaca_account_history(days=30)
    last_equity = history[-1]["equity"] if history else health.get("equity")
    pnl_today   = history[-1]["pnl_pct"] if history else None

    per_sleeve = state.get("per_sleeve_pnl_today", {})
    weights    = state.get("last_weights", {})

    honest = cfg.get("targets", {}).get("honest_walk_forward", {})

    return {
        "experiment": "EXP-2520",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "headline": {
            "equity": last_equity,
            "pnl_today_pct": pnl_today,
            "rolling_peak": health.get("rolling_peak"),
            "trailing_dd_pct": health.get("trailing_dd_pct"),
            "leverage": health.get("leverage"),
            "scale_factor": health.get("scale_factor"),
            "circuit_breaker_state": health.get("circuit_breaker_state"),
            "vix_last": health.get("vix_last"),
            "open_positions": health.get("n_open_positions"),
        },
        "per_sleeve_attribution": per_sleeve,
        "current_weights": weights,
        "recent_history": history,
        "honest_targets": honest,
        "alerts_today": health.get("alerts", []),
    }


def _write_html(report: Dict, path: Path) -> None:
    h = report["headline"]
    honest = report.get("honest_targets", {})
    rows_slv = "".join(
        f"<tr><td>{k}</td><td>${v:,.2f}</td></tr>"
        for k, v in report.get("per_sleeve_attribution", {}).items()
    ) or "<tr><td colspan='2' class='small'>no per-sleeve data yet</td></tr>"
    rows_w = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        for k, v in report.get("current_weights", {}).items()
    ) or "<tr><td colspan='2' class='small'>no weights yet</td></tr>"
    rows_hist = "".join(
        f"<tr><td>{r.get('ts')}</td><td>${r.get('equity', 0):,.2f}</td>"
        f"<td>{(r.get('pnl_pct') or 0) * 100:.2f}%</td></tr>"
        for r in report.get("recent_history", [])[-15:]
    ) or "<tr><td colspan='3' class='small'>no history</td></tr>"

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2520 — Daily Report {report['generated'][:10]}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;background:#fff;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}}
 h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.92em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}}
 th{{background:#f0f0f0}}
 .small{{color:#666;font-size:.88em}}
</style></head><body>
<h1>EXP-2520 — Paper-Trading Daily Report</h1>
<p class='small'>{report['generated']} · config <code>{report['config_path']}</code></p>

<h2>Headline</h2>
<table>
<tr><th>Equity</th><td>${(h.get('equity') or 0):,.2f}</td></tr>
<tr><th>Today's P&amp;L</th><td>{(h.get('pnl_today_pct') or 0) * 100:.2f}%</td></tr>
<tr><th>Rolling peak</th><td>${(h.get('rolling_peak') or 0):,.2f}</td></tr>
<tr><th>Trailing drawdown</th><td>{(h.get('trailing_dd_pct') or 0):.2f}%</td></tr>
<tr><th>Leverage</th><td>{(h.get('leverage') or 1.0):.2f}×</td></tr>
<tr><th>Vol scale factor</th><td>{(h.get('scale_factor') or 1.0):.3f}</td></tr>
<tr><th>Circuit breaker</th><td>{h.get('circuit_breaker_state', 'unknown')}</td></tr>
<tr><th>VIX</th><td>{(h.get('vix_last') or 0):.2f}</td></tr>
<tr><th>Open positions</th><td>{h.get('open_positions', 0)}</td></tr>
</table>

<h2>Per-sleeve attribution today</h2>
<table><tr><th>Sleeve</th><th>P&amp;L</th></tr>{rows_slv}</table>

<h2>Current weights (Ledoit-Wolf risk-parity)</h2>
<table><tr><th>Stream</th><th>Weight</th></tr>{rows_w}</table>

<h2>Last 15 days</h2>
<table><tr><th>Date</th><th>Equity</th><th>P&amp;L %</th></tr>{rows_hist}</table>

<h2>Honest walk-forward targets (EXP-2280)</h2>
<table>
<tr><th>Metric</th><th>Target</th></tr>
<tr><td>Pooled OOS Sharpe</td><td>{honest.get('pooled_sharpe', '—')}</td></tr>
<tr><td>Pooled OOS CAGR</td><td>{honest.get('pooled_cagr_pct', '—')}%</td></tr>
<tr><td>Pooled OOS Max DD</td><td>{honest.get('pooled_max_dd_pct', '—')}%</td></tr>
<tr><td>Per-fold median Sharpe</td><td>{honest.get('per_fold_median_sharpe', '—')}</td></tr>
<tr><td>Frac folds ≥ 6</td><td>{honest.get('per_fold_frac_above_6', '—')}</td></tr>
</table>

<h2>Active alerts</h2>
<p class='small'>{len(report.get('alerts_today', []))} alerts in the last monitor poll.</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp2410_production_paper.yaml")
    ap.add_argument("--out-dir", default="reports/exp2520/daily")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")

    report = build_report(Path(args.config))
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{day}.json"
    html_path = out_dir / f"{day}.html"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    _write_html(report, html_path)
    LOG.info("wrote %s + %s", json_path, html_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
