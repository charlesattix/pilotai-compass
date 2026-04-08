"""
scripts/north_star_v6_daily_report.py — EXP-2290 daily P&L report generator.

Produces the end-of-day report for the 7-stream North Star v6 portfolio:

  • Reads the live health.json snapshot (written by the monitor)
  • Appends today's closing equity and per-sleeve P&L to the rolling
    equity log (CSV) at logs/north_star_v6/equity_log.csv
  • Computes day / week / MTD / YTD / total P&L and drawdown
  • Renders an HTML + JSON report under reports/north_star_v6/
  • Optionally sends a Telegram summary via shared.telegram_alerts

Usage:
    python scripts/north_star_v6_daily_report.py --config configs/north_star_v6_prod.yaml
    python scripts/north_star_v6_daily_report.py --send-telegram
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class ReportPaths:
    log_dir: Path
    health_file: Path
    equity_log: Path
    report_dir: Path

    @classmethod
    def from_cfg(cls, cfg: Dict) -> "ReportPaths":
        mon = cfg.get("monitoring", {}) or {}
        log_dir = ROOT / mon.get("log_dir", "logs/north_star_v6")
        report_dir = ROOT / (cfg.get("reports", {})
                              .get("daily_pnl", {})
                              .get("output_dir", "reports/north_star_v6"))
        return cls(
            log_dir=log_dir,
            health_file=log_dir / "health.json",
            equity_log=log_dir / "equity_log.csv",
            report_dir=report_dir,
        )


def load_config(path: Path) -> Dict:
    return yaml.safe_load(path.read_text())


def load_health(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def append_equity_log(equity_log: Path, date: str, equity: float,
                        sleeves: List[Dict]) -> None:
    equity_log.parent.mkdir(parents=True, exist_ok=True)
    existing_dates = set()
    rows: List[Dict] = []
    if equity_log.exists():
        with equity_log.open() as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                existing_dates.add(r["date"])
                rows.append(r)
    if date in existing_dates:
        return  # already logged for today
    row = {"date": date, "equity": f"{equity:.2f}"}
    for s in sleeves:
        row[f"{s['id']}_pnl_today"] = f"{s.get('pnl_today', 0):.2f}"
        row[f"{s['id']}_positions"] = str(s.get("n_positions", 0))
    rows.append(row)
    all_fields = sorted({k for r in rows for k in r.keys()})
    with equity_log.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=all_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in all_fields})


def read_equity_series(equity_log: Path) -> List[Dict]:
    if not equity_log.exists():
        return []
    with equity_log.open() as fh:
        return list(csv.DictReader(fh))


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_period_returns(series: List[Dict],
                              start_capital: float) -> Dict[str, Dict]:
    if not series:
        return {}
    # parse to (date, equity)
    points = []
    for r in series:
        try:
            d = datetime.strptime(r["date"], "%Y-%m-%d").date()
            e = float(r["equity"])
            points.append((d, e))
        except Exception:
            continue
    if not points:
        return {}
    points.sort()
    today, cur = points[-1]

    def at_or_before(cutoff):
        for d, e in reversed(points):
            if d <= cutoff:
                return e
        return start_capital

    day_anchor = at_or_before(today - timedelta(days=1))
    week_anchor = at_or_before(today - timedelta(days=7))
    mtd_anchor = at_or_before(today.replace(day=1) - timedelta(days=1))
    ytd_anchor = at_or_before(today.replace(month=1, day=1) - timedelta(days=1))

    def pct(cur, anchor):
        return (cur / anchor - 1.0) * 100.0 if anchor > 0 else 0.0

    # Rolling drawdown
    equities = [e for _, e in points]
    peak = 0.0
    max_dd = 0.0
    for e in equities:
        peak = max(peak, e)
        if peak > 0:
            dd = (e - peak) / peak
            if dd < max_dd:
                max_dd = dd

    # Realized per-day returns for Sharpe
    rets = []
    for i in range(1, len(equities)):
        if equities[i - 1] > 0:
            rets.append(equities[i] / equities[i - 1] - 1.0)
    sharpe = 0.0
    if len(rets) >= 5:
        import statistics
        mu = statistics.mean(rets)
        sd = statistics.pstdev(rets)
        sharpe = (mu / sd * math.sqrt(252)) if sd > 1e-12 else 0.0

    return {
        "as_of": today.isoformat(),
        "equity": round(cur, 2),
        "starting_capital": round(start_capital, 2),
        "total_pnl": round(cur - start_capital, 2),
        "total_pct": round((cur / start_capital - 1.0) * 100.0, 3) if start_capital > 0 else 0.0,
        "day_pnl": round(cur - day_anchor, 2),
        "day_pct": round(pct(cur, day_anchor), 3),
        "week_pnl": round(cur - week_anchor, 2),
        "week_pct": round(pct(cur, week_anchor), 3),
        "mtd_pnl": round(cur - mtd_anchor, 2),
        "mtd_pct": round(pct(cur, mtd_anchor), 3),
        "ytd_pnl": round(cur - ytd_anchor, 2),
        "ytd_pct": round(pct(cur, ytd_anchor), 3),
        "max_dd_pct": round(max_dd * 100.0, 3),
        "sharpe_since_launch": round(sharpe, 3),
        "n_days": len(points),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_html(health: Dict, metrics: Dict) -> str:
    sleeves_rows = ""
    for s in health.get("sleeves", []):
        status_cls = {"OK": "ok", "WARN": "warn",
                       "BREACH": "bad"}.get(s.get("status", "OK"), "")
        pnl_cls = "good" if s.get("pnl_today", 0) >= 0 else "bad"
        sleeves_rows += f"""<tr>
            <td><strong>{s['id']}</strong></td>
            <td>{s.get('ticker','?')}</td>
            <td>{s.get('weight', 0)*100:.1f}%</td>
            <td>{s.get('n_positions', 0)}</td>
            <td>${s.get('market_value', 0):,.0f}</td>
            <td class="{pnl_cls}">${s.get('pnl_today', 0):+,.0f}</td>
            <td>{s.get('baseline_sharpe', 0):.2f}</td>
            <td class="{status_cls}">{s.get('status', 'OK')}</td>
        </tr>"""

    breaches_rows = ""
    for b in health.get("breaches", []):
        sev = b.get("severity", "info")
        breaches_rows += f"""<tr>
            <td class="{sev}">{b.get('code','?')}</td>
            <td>{b.get('severity','?')}</td>
            <td>{b.get('message','?')}</td>
        </tr>"""
    if not breaches_rows:
        breaches_rows = "<tr><td colspan='3' style='color:#16a34a'>No active breaches ✓</td></tr>"

    day_cls = "good" if metrics.get("day_pnl", 0) >= 0 else "bad"
    total_cls = "good" if metrics.get("total_pnl", 0) >= 0 else "bad"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2290 North Star v6 — Daily P&L {metrics.get('as_of','')}</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1100px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }}
  h2 {{ color:#334155; margin-top:2em; padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:140px; }}
  .kpi .value {{ font-size:1.6em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; font-weight:700; }}
  .bad  {{ color:#dc2626; font-weight:700; }}
  .warn {{ color:#ca8a04; font-weight:700; }}
  .ok   {{ color:#16a34a; font-weight:600; }}
  .critical {{ color:#dc2626; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.78em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child, td:nth-child(2) {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
</style></head><body>

<h1>EXP-2290 — North Star v6 Daily P&L</h1>
<div class="subtitle">As-of {metrics.get('as_of', health.get('timestamp', '?'))} ·
7-stream portfolio · Paper mode</div>

<h2>Headline</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value">${health.get('equity', 0):,.0f}</div><div class="label">Equity</div></div>
    <div class="kpi"><div class="value {day_cls}">${metrics.get('day_pnl', 0):+,.0f}</div><div class="label">Day P&L ({metrics.get('day_pct', 0):+.2f}%)</div></div>
    <div class="kpi"><div class="value">${metrics.get('week_pnl', 0):+,.0f}</div><div class="label">Week ({metrics.get('week_pct', 0):+.2f}%)</div></div>
    <div class="kpi"><div class="value">${metrics.get('mtd_pnl', 0):+,.0f}</div><div class="label">MTD ({metrics.get('mtd_pct', 0):+.2f}%)</div></div>
    <div class="kpi"><div class="value {total_cls}">${metrics.get('total_pnl', 0):+,.0f}</div><div class="label">Since launch ({metrics.get('total_pct', 0):+.2f}%)</div></div>
</div>
<div class="kpi-row">
    <div class="kpi"><div class="value">{metrics.get('max_dd_pct', 0):.2f}%</div><div class="label">Max DD</div></div>
    <div class="kpi"><div class="value">{metrics.get('sharpe_since_launch', 0):.2f}</div><div class="label">Sharpe (live)</div></div>
    <div class="kpi"><div class="value">{health.get('open_positions', 0)}</div><div class="label">Open Positions</div></div>
    <div class="kpi"><div class="value">{metrics.get('n_days', 0)}</div><div class="label">Days Tracked</div></div>
</div>

<h2>Sleeve Breakdown</h2>
<table>
    <thead><tr><th>Sleeve</th><th>Ticker</th><th>Weight</th><th>Positions</th>
    <th>Market Value</th><th>Day P&L</th><th>Baseline Sh</th><th>Status</th></tr></thead>
    <tbody>{sleeves_rows}</tbody>
</table>

<h2>Active Breaches</h2>
<table>
    <thead><tr><th>Code</th><th>Severity</th><th>Message</th></tr></thead>
    <tbody>{breaches_rows}</tbody>
</table>

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2290 — scripts/north_star_v6_daily_report.py · Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

</body></html>"""


def build_telegram_summary(health: Dict, metrics: Dict) -> str:
    lines = [
        f"📊 *EXP-2290 North Star v6 — Daily Report {metrics.get('as_of','')}*",
        f"Equity: ${health.get('equity', 0):,.0f}",
        f"Day: {'🟢' if metrics.get('day_pnl', 0) >= 0 else '🔴'} "
            f"${metrics.get('day_pnl', 0):+,.0f} ({metrics.get('day_pct', 0):+.2f}%)",
        f"Week: ${metrics.get('week_pnl', 0):+,.0f} ({metrics.get('week_pct', 0):+.2f}%)",
        f"MTD:  ${metrics.get('mtd_pnl', 0):+,.0f} ({metrics.get('mtd_pct', 0):+.2f}%)",
        f"Total:${metrics.get('total_pnl', 0):+,.0f} ({metrics.get('total_pct', 0):+.2f}%)",
        f"Max DD: {metrics.get('max_dd_pct', 0):.2f}%   "
            f"Sharpe: {metrics.get('sharpe_since_launch', 0):.2f}",
        "",
        "*Sleeves:*",
    ]
    for s in health.get("sleeves", []):
        emoji = "🟢" if s.get("pnl_today", 0) >= 0 else "🔴"
        lines.append(
            f"  {emoji} {s['id']:22s} ${s.get('pnl_today', 0):+,.0f}  "
            f"[{s.get('status', 'OK')}]"
        )
    return "\n".join(lines)


def send_telegram(message: str) -> None:
    try:
        from shared.telegram_alerts import send_telegram_alert
        send_telegram_alert(message, severity="info")
        return
    except Exception:
        pass
    try:
        from compass.telegram_alerter import TelegramAlerter, Priority
        TelegramAlerter().send(message, priority=Priority.INFO)
        return
    except Exception:
        pass
    print("[report] (telegram unavailable, printing)")
    print(message)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="EXP-2290 daily P&L report")
    ap.add_argument("--config", default=str(ROOT / "configs" / "north_star_v6_prod.yaml"))
    ap.add_argument("--send-telegram", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    paths = ReportPaths.from_cfg(cfg)
    start_capital = float((cfg.get("account", {}) or {}).get("starting_capital", 100_000))

    health = load_health(paths.health_file)
    if health is None:
        print(f"[report] ERROR: no health file at {paths.health_file}")
        print("         is the monitor running?")
        return 1

    today = datetime.utcnow().date().isoformat()
    append_equity_log(paths.equity_log, today, health["equity"], health.get("sleeves", []))

    series = read_equity_series(paths.equity_log)
    metrics = compute_period_returns(series, start_capital)

    paths.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = paths.report_dir / f"daily_pnl_{today}.json"
    html_path = paths.report_dir / f"daily_pnl_{today}.html"
    latest_json = paths.report_dir / "latest.json"
    latest_html = paths.report_dir / "latest.html"

    payload = {
        "experiment": "EXP-2290",
        "as_of": today,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "health_snapshot": health,
        "metrics": metrics,
    }
    json_text = json.dumps(payload, indent=2, default=str)
    json_path.write_text(json_text)
    latest_json.write_text(json_text)
    print(f"  → {json_path}")

    html_text = render_html(health, metrics)
    html_path.write_text(html_text, encoding="utf-8")
    latest_html.write_text(html_text, encoding="utf-8")
    print(f"  → {html_path}")

    summary = build_telegram_summary(health, metrics)
    print("\n" + summary)
    if args.send_telegram:
        send_telegram(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
