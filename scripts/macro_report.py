"""
Macro Report Generator
======================
Generates a human-readable markdown report from the latest macro snapshot in DB.

Usage:
  python3 scripts/macro_report.py                  # markdown to stdout
  python3 scripts/macro_report.py --date 2024-06-07  # specific historical date
  python3 scripts/macro_report.py --html            # generate HTML report

The report covers:
  1. Macro score summary (4 dimensions + regime)
  2. Sector rankings table (all 15 ETFs with RS and RRG quadrant)
  3. Active themes (Leading/Improving sectors)
  4. Upcoming macro events and scaling factor
  5. Trade ideas (eligible underlyings given current regime)
  6. Historical context (how does today compare to history)
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# ── Path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from compass.events import compute_composite_scaling, get_upcoming_events
from compass.macro_db import get_db

QUADRANT_EMOJI = {
    "Leading":   "✅",
    "Improving": "↗️",
    "Weakening": "↘️",
    "Lagging":   "❌",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_snapshot(target_date: Optional[str] = None) -> Optional[Dict]:
    """Load a snapshot from macro_state.db for the given date (or latest)."""
    conn = get_db()
    try:
        if target_date is None:
            target_date = conn.execute(
                "SELECT MAX(date) FROM snapshots"
            ).fetchone()[0]
        if not target_date:
            return None

        snap_row = conn.execute(
            "SELECT * FROM snapshots WHERE date = ?", (target_date,)
        ).fetchone()
        if not snap_row:
            return None

        ms_row = conn.execute(
            "SELECT * FROM macro_score WHERE date = ?", (target_date,)
        ).fetchone()

        sector_rows = conn.execute(
            """
            SELECT * FROM sector_rs WHERE date = ?
            ORDER BY rank_3m ASC
            """,
            (target_date,),
        ).fetchall()

        return {
            "snap": dict(snap_row),
            "macro_score": dict(ms_row) if ms_row else {},
            "sector_rankings": [dict(r) for r in sector_rows],
            "date": target_date,
        }
    finally:
        conn.close()


def load_historical_context(as_of_date: str, lookback_weeks: int = 52) -> List[Dict]:
    """Load macro_score rows for the past N weeks for context."""
    conn = get_db()
    try:
        cutoff = (date.fromisoformat(as_of_date) - timedelta(weeks=lookback_weeks)).strftime("%Y-%m-%d")
        rows = conn.execute(
            """
            SELECT date, overall, growth, inflation, fed_policy, risk_appetite, regime
            FROM macro_score
            WHERE date >= ? AND date <= ?
            ORDER BY date ASC
            """,
            (cutoff, as_of_date),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Report sections
# ─────────────────────────────────────────────────────────────────────────────

def _regime_description(overall: Optional[float]) -> str:
    if overall is None:
        return "UNKNOWN"
    if overall >= 65:
        return "BULL_MACRO — favorable conditions, full risk allocation"
    if overall >= 55:
        return "NEUTRAL+ — above-average conditions, standard sizing"
    if overall >= 45:
        return "NEUTRAL — mixed signals, standard sizing"
    if overall >= 35:
        return "NEUTRAL- — below-average, consider reducing exposure"
    return "BEAR_MACRO — unfavorable conditions, reduce directional exposure"


def _score_bar(value: Optional[float], width: int = 20) -> str:
    """ASCII progress bar for a 0–100 score."""
    if value is None:
        return "[" + "?" * width + "]"
    filled = int(round(value / 100 * width))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:.1f}"


def _percentile_label(value: Optional[float], history: List[Dict], field: str) -> str:
    """Show where this value ranks in the historical context."""
    vals = [r[field] for r in history if r.get(field) is not None]
    if not vals or value is None:
        return ""
    pct = sum(1 for v in vals if v <= value) / len(vals) * 100
    return f"(52-wk pctile: {pct:.0f}%)"


def generate_markdown(data: Dict, events: List[Dict]) -> str:
    snap = data["snap"]
    ms = data["macro_score"]
    rankings = data["sector_rankings"]
    snap_date = data["date"]

    history = load_historical_context(snap_date)
    overall = ms.get("overall")
    scaling = compute_composite_scaling(events)

    lines = [
        f"# Macro Intelligence Report — {snap_date}",
        "",
        f"**SPY Close:** ${snap.get('spy_close', 'N/A'):.2f}" if snap.get("spy_close") else "**SPY Close:** N/A",
        f"**Event Scaling Factor:** {scaling:.2f}",
        f"**Snapshots in DB:** {_count_snapshots()}",
        "",
        "---",
        "",
        "## 1. Macro Score",
        "",
        f"**Overall: {overall:.1f} / 100**" if overall is not None else "**Overall: N/A**",
        f"**Regime: {_regime_description(overall)}**",
        "",
    ]

    if overall is not None:
        lines += [
            "| Dimension | Score | Bar | 52-Week Percentile |",
            "|-----------|-------|-----|--------------------|",
        ]
        for dim, field in [
            ("Growth",       "growth"),
            ("Inflation",    "inflation"),
            ("Fed Policy",   "fed_policy"),
            ("Risk Appetite","risk_appetite"),
        ]:
            val = ms.get(field)
            bar = _score_bar(val, width=15)
            pctile = _percentile_label(val, history, field)
            lines.append(f"| {dim:<15} | {val:.1f} | {bar} | {pctile} |" if val else f"| {dim:<15} | N/A | — | — |")

        lines += [
            "",
            "### Key Indicators",
            "",
            "| Indicator | Value |",
            "|-----------|-------|",
        ]
        indicator_map = [
            ("VIX",              "vix",              "{:.2f}"),
            ("10Y-2Y Spread",    "t10y2y",           "{:+.3f}%"),
            ("HY OAS Spread",    "hy_oas_pct",       "{:.3f}%"),
            ("CPI YoY",          "cpi_yoy_pct",      "{:.2f}%"),
            ("Core CPI YoY",     "core_cpi_yoy_pct", "{:.2f}%"),
            ("5Y Breakeven",     "breakeven_5y",     "{:.3f}%"),
            ("Fed Funds Rate",   "fedfunds",         "{:.2f}%"),
            ("CFNAI 3M Avg",     "cfnai_3m",         "{:+.3f}"),
            ("Payrolls 3M Avg",  "payrolls_3m_avg_k","{:+.0f}K"),
        ]
        for label, field, fmt in indicator_map:
            val = ms.get(field)
            if val is not None:
                try:
                    lines.append(f"| {label} | {fmt.format(val)} |")
                except Exception:
                    lines.append(f"| {label} | {val} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Sector Rankings",
        "",
        "| # | Ticker | Name | RS 3M | RS 12M | RRG Quadrant | Close |",
        "|---|--------|------|-------|--------|--------------|-------|",
    ]

    for item in rankings:
        rs3  = f"{item['rs_3m']:+.1f}%"  if item.get("rs_3m")  is not None else "—"
        rs12 = f"{item['rs_12m']:+.1f}%" if item.get("rs_12m") is not None else "—"
        quad = item.get("rrg_quadrant") or "—"
        emoji = QUADRANT_EMOJI.get(quad, "")
        close = f"${item['close']:.2f}" if item.get("close") else "—"
        lines.append(
            f"| {item['rank_3m']} | **{item['ticker']}** | {item['name']} "
            f"| {rs3} | {rs12} | {emoji} {quad} | {close} |"
        )

    # Active themes from RRG
    leading   = [r for r in rankings if r.get("rrg_quadrant") == "Leading"]
    improving = [r for r in rankings if r.get("rrg_quadrant") == "Improving"]
    weakening = [r for r in rankings if r.get("rrg_quadrant") == "Weakening"]
    lagging   = [r for r in rankings if r.get("rrg_quadrant") == "Lagging"]

    lines += [
        "",
        "---",
        "",
        "## 3. RRG Quadrant Summary",
        "",
        "**Leading** ✅ (strong RS, gaining momentum): "
        + (", ".join(r["ticker"] for r in leading) or "None"),
        "",
        "**Improving** ↗️ (weak RS, gaining momentum): "
        + (", ".join(r["ticker"] for r in improving) or "None"),
        "",
        "**Weakening** ↘️ (strong RS, losing momentum): "
        + (", ".join(r["ticker"] for r in weakening) or "None"),
        "",
        "**Lagging** ❌ (weak RS, losing momentum): "
        + (", ".join(r["ticker"] for r in lagging) or "None"),
    ]

    # Events
    lines += [
        "",
        "---",
        "",
        "## 4. Upcoming Macro Events",
        "",
    ]
    if events:
        lines += [
            "| Event | Date | Days Out | Position Scaling |",
            "|-------|------|----------|-----------------|",
        ]
        for ev in events:
            lines.append(
                f"| {ev['event_type']} | {ev['event_date']} | T-{ev['days_out']} | {ev['scaling_factor']:.2f}x |"
            )
        lines += [
            "",
            f"**Composite scaling factor: {scaling:.2f}x** "
            f"({'no reduction' if scaling == 1.0 else f'{(1-scaling)*100:.0f}% size reduction'})",
        ]
    else:
        lines.append("No scheduled events in the next 14 days. Scaling factor: **1.00x**")

    # Trade ideas
    from compass.macro_db import get_eligible_underlyings
    eligible = get_eligible_underlyings(regime="BULL" if (overall or 50) >= 55 else "NEUTRAL")

    lines += [
        "",
        "---",
        "",
        "## 5. Trade Ideas",
        "",
        f"**Eligible underlyings (current macro + regime):** {', '.join(eligible)}",
        "",
    ]

    if overall is not None and overall >= 65:
        lines.append("**Macro bias:** BULLISH — favor bull put spreads on leading sectors")
    elif overall is not None and overall < 45:
        lines.append("**Macro bias:** BEARISH — favor bear call spreads; contract universe to SPY/QQQ/IWM")
    else:
        lines.append("**Macro bias:** NEUTRAL — standard bull put spreads on SPY/QQQ/IWM")

    if leading:
        lines.append("\n**Top conviction plays (Leading + top RS):**")
        for r in leading[:3]:
            if r.get("rs_3m") is not None and r["rs_3m"] > 3:
                lines.append(f"  - **{r['ticker']}** ({r['name']}): RS 3M = {r['rs_3m']:+.1f}%, {r['rrg_quadrant']}")

    if scaling < 0.80:
        lines += [
            "",
            f"⚠️ **Event risk active** — reduce position sizes by {(1-scaling)*100:.0f}% "
            f"(event scaling = {scaling:.2f}x)",
        ]

    # Historical context
    if history:
        scores = [r["overall"] for r in history if r.get("overall") is not None]
        if scores and overall is not None:
            avg_score = sum(scores) / len(scores)
            pct = sum(1 for s in scores if s <= overall) / len(scores) * 100
            lines += [
                "",
                "---",
                "",
                "## 6. Historical Context (52 Weeks)",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Current overall score | {overall:.1f} |",
                f"| 52-week average | {avg_score:.1f} |",
                f"| 52-week percentile | {pct:.0f}% |",
                f"| 52-week min | {min(scores):.1f} |",
                f"| 52-week max | {max(scores):.1f} |",
            ]

            # Regime distribution
            regime_counts: Dict[str, int] = {}
            for r in history:
                reg = r.get("regime") or "UNKNOWN"
                regime_counts[reg] = regime_counts.get(reg, 0) + 1
            n = len(history) or 1
            lines += [
                "",
                "**Regime distribution (52 weeks):**",
                "",
                "| Regime | Weeks | % |",
                "|--------|-------|---|",
            ]
            for reg, cnt in sorted(regime_counts.items(), key=lambda x: -x[1]):
                lines.append(f"| {reg} | {cnt} | {cnt/n*100:.0f}% |")

    lines += ["", "---", "", f"*Generated by PilotAI Macro Intelligence — {snap_date}*", ""]
    return "\n".join(lines)


def _count_snapshots() -> int:
    from compass.macro_db import get_snapshot_count
    return get_snapshot_count()


def markdown_to_html(md: str, title: str = "Macro Report") -> str:
    """Convert markdown report to a clean HTML page."""
    # Simple but readable conversion — no external deps
    import re

    html_body = md
    # Headers
    html_body = re.sub(r"^#### (.+)$", r"<h4>\1</h4>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^### (.+)$",  r"<h3>\1</h3>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^## (.+)$",   r"<h2>\1</h2>", html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^# (.+)$",    r"<h1>\1</h1>", html_body, flags=re.MULTILINE)
    # Bold
    html_body = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html_body)
    # Tables: basic pipe-table → HTML
    lines = html_body.split("\n")
    out = []
    in_table = False
    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                out.append("<table class='data-table'>")
                in_table = True
            if re.match(r"^\s*\|[-| :]+\|\s*$", line):
                continue  # separator row
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            tag = "th" if not in_table else "td"
            row_html = "".join(f"<{tag}>{c}</{tag}>" for c in cells)
            out.append(f"<tr>{row_html}</tr>")
        else:
            if in_table:
                out.append("</table>")
                in_table = False
            out.append(line)
    if in_table:
        out.append("</table>")
    html_body = "\n".join(out)
    # Paragraphs and line breaks
    html_body = re.sub(r"\n\n", "</p><p>", html_body)
    html_body = f"<p>{html_body}</p>"
    html_body = re.sub(r"<p>\s*<h", "<h", html_body)
    html_body = re.sub(r"</h(\d)>\s*</p>", r"</h\1>", html_body)
    html_body = re.sub(r"<p>\s*---\s*</p>", "<hr>", html_body)
    html_body = re.sub(r"<p>\s*</p>", "", html_body)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #fff; color: #1a1a1a; line-height: 1.6; padding: 32px; max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; border-bottom: 2px solid #0066cc; padding-bottom: 8px; margin-bottom: 16px; color: #003d99; }}
  h2 {{ font-size: 1.3rem; color: #0066cc; margin: 28px 0 12px; border-left: 4px solid #0066cc; padding-left: 10px; }}
  h3 {{ font-size: 1.1rem; color: #333; margin: 20px 0 8px; }}
  hr {{ border: none; border-top: 1px solid #e0e0e0; margin: 20px 0; }}
  p {{ margin: 8px 0; }}
  strong {{ color: #1a1a1a; }}
  .data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9rem; }}
  .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 7px 12px; text-align: left; }}
  .data-table th {{ background: #f0f4ff; font-weight: 600; color: #003d99; }}
  .data-table tr:nth-child(even) {{ background: #f9f9f9; }}
  .data-table tr:hover {{ background: #eef2ff; }}
  code {{ background: #f4f4f4; padding: 1px 5px; border-radius: 3px; font-family: monospace; font-size: 0.9em; }}
  ul {{ margin: 8px 0 8px 24px; }}
  li {{ margin: 3px 0; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate macro report from DB")
    p.add_argument("--date", metavar="YYYY-MM-DD", help="Report for specific date (default: latest)")
    p.add_argument("--html", action="store_true", help="Output HTML instead of markdown")
    p.add_argument("--out",  metavar="FILE", help="Write output to file (default: stdout)")
    return p.parse_args()


def main():
    args = parse_args()

    data = load_snapshot(args.date)
    if not data:
        logger.error("No snapshot found for date: %s", args.date or "latest")
        sys.exit(1)

    events = get_upcoming_events(
        as_of=date.fromisoformat(data["date"]),
        horizon_days=14,
    )

    md = generate_markdown(data, events)

    if args.html:
        output = markdown_to_html(md, title=f"Macro Report — {data['date']}")
    else:
        output = md

    if args.out:
        Path(args.out).write_text(output)
        print(f"Report written to: {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
