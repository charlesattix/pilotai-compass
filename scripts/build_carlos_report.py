#!/usr/bin/env python3
"""Build comprehensive experiment report for Carlos — 2026-04-06."""

from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════
# Experiment catalog (real numbers from recent commits)
# ═══════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    # ── WINNERS (deploy) ───────────────────────────────────────────────
    {
        "id": "EXP-1220",
        "name": "Credit Spread @ 1.5x (Baseline)",
        "type": "Regime-adaptive credit spread",
        "data": "IronVault (SPY) + Yahoo (VIX/VIX3M)",
        "cagr": "99.2%",
        "sharpe": "3.83",
        "max_dd": "-11.2%",
        "wr": "~87%",
        "corr_1220": "1.000",
        "verdict": "DEPLOYED",
        "tier": "winner",
        "notes": "The core strategy. Live paper trading since Mar 15. 98.6% CAGR observed.",
        "commit": "a43b49d",
    },

    # ── PROMISING (needs work) ─────────────────────────────────────────
    {
        "id": "EXP-1710",
        "name": "1-3 DTE SPY Iron Condors",
        "type": "Short-dated theta harvesting",
        "data": "IronVault (SPY options) + Yahoo (SPY/VIX)",
        "cagr": "+53.6%",
        "sharpe": "5.58 → 1.7 (decaying)",
        "max_dd": "-0.9%",
        "wr": "94%",
        "corr_1220": "-0.002",
        "verdict": "PROMISING",
        "tier": "promising",
        "notes": "Pivoted from 0DTE SPX (no data). Sharpe decaying 37→11→1.7 over 2023-2025. Adaptive overlay sizes down when trailing Sharpe < 1.0 (47% of 2025 trades). Use as 20% diversifier.",
        "commit": "3b35b90",
    },
    {
        "id": "EXP-1820",
        "name": "Dispersion (Relative Vol Premium)",
        "type": "Sector ETF vol > SPY vol signal",
        "data": "IronVault (SPY/XLF/XLI/XLK/XLE) + Yahoo spot",
        "cagr": "+0.8%",
        "sharpe": "1.94 (IS 1.89 / OOS 1.88)",
        "max_dd": "-0.4%",
        "wr": "76%",
        "corr_1220": "-0.032 (monthly)",
        "verdict": "PROMISING",
        "tier": "promising",
        "notes": "89 trades, zero OOS degradation, no parameter cliff. Capacity-constrained at ~$5M (XLI ADV). Best sector: XLI Sharpe 2.82, 97% WR. Pure diversifier candidate.",
        "commit": "a6fcd4f",
    },
    {
        "id": "EXP-1780",
        "name": "Crisis Alpha / Trend Following",
        "type": "Multi-asset momentum (13 tickers)",
        "data": "Yahoo (SPY/IWM/EFA/EEM/QQQ/TLT/LQD/HYG/GLD/USO/DBA/DBB/UUP)",
        "cagr": "+8.04%",
        "sharpe": "0.31 (corrected)",
        "max_dd": "-23.6%",
        "wr": "—",
        "corr_1220": "-0.146 overall / -0.420 during DD",
        "verdict": "PROMISING",
        "tier": "promising",
        "notes": "v3 grid: 40/72 passing configs. Best: v2_round/vol=0.06/1.5x. Real crisis outperf: +18.6% vs SPY across 37 DD periods. Only useful at 5% allocation as insurance.",
        "commit": "55ff43d",
    },
    {
        "id": "EXP-1660",
        "name": "Variance Risk Premium (VRP)",
        "type": "Short vol — SPY/QQQ/XLF/XLI/XLK",
        "data": "IronVault (5 tickers)",
        "cagr": "Variable per ticker",
        "sharpe": "3 viable survivors",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "Moderate",
        "verdict": "PROMISING",
        "tier": "promising",
        "notes": "60-config grid, 3 survivors after hardening. XLI is the new winner. Portfolio integration test: VRP is NEUTRAL, not additive to EXP-1220.",
        "commit": "9ac4f8d",
    },

    # ── KILLED (honest failures) ───────────────────────────────────────
    {
        "id": "EXP-1700",
        "name": "VIX Futures Roll Yield",
        "type": "Contango harvesting",
        "data": "Yahoo (VIXY/VXX)",
        "cagr": "minimal",
        "sharpe": "0.03",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "—",
        "verdict": "KILLED",
        "tier": "killed",
        "notes": "VIXY/VXX contango premium already priced in. No alpha after costs.",
        "commit": "4f5b39c",
    },
    {
        "id": "EXP-1720",
        "name": "Sector ETF Pairs Trading",
        "type": "Cointegration mean-reversion",
        "data": "Yahoo (XLF/XLK, XLE/XLU, etc.)",
        "cagr": "negative",
        "sharpe": "-0.46",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "—",
        "verdict": "KILLED",
        "tier": "killed",
        "notes": "No stable cointegration relationships. Pairs decay rapidly.",
        "commit": "766e2f1",
    },
    {
        "id": "EXP-1730",
        "name": "Treasury Curve Trades",
        "type": "TLT/IEF/SHY butterfly",
        "data": "Yahoo (TLT/IEF/SHY)",
        "cagr": "negative",
        "sharpe": "negative",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "—",
        "verdict": "KILLED",
        "tier": "killed",
        "notes": "Rate regime changes crush butterflies. Even hedge overlay salvage attempt failed.",
        "commit": "dd1ac1c",
    },
    {
        "id": "EXP-1790",
        "name": "Overnight Drift",
        "type": "Close-to-open gap fade",
        "data": "Yahoo (SPY intraday)",
        "cagr": "marginal",
        "sharpe": "weak",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "—",
        "verdict": "KILLED",
        "tier": "killed",
        "notes": "Academic edge confirmed in gross terms but eliminated by transaction costs.",
        "commit": "e802687",
    },
    {
        "id": "EXP-1800",
        "name": "Event IV Crush",
        "type": "FOMC/CPI IV crush capture",
        "data": "IronVault (SPY macro proxy)",
        "cagr": "N/A",
        "sharpe": "N/A",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "—",
        "verdict": "KILLED",
        "tier": "killed",
        "notes": "Data gap: no reliable FOMC-dated options. Index proxy approach did not work.",
        "commit": "8ccf251",
    },
    {
        "id": "EXP-1810",
        "name": "Gamma Scalping",
        "type": "Long gamma + delta hedge",
        "data": "IronVault (SPY options)",
        "cagr": "negative",
        "sharpe": "-2.28",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "—",
        "verdict": "KILLED",
        "tier": "killed",
        "notes": "Honest miss — hedging costs exceeded gamma income. Not useful as hedge either.",
        "commit": "9b3c4e7",
    },
    {
        "id": "EXP-1830",
        "name": "Momentum Rotation (Long-Short)",
        "type": "Sector momentum rotation",
        "data": "Yahoo (sector ETFs)",
        "cagr": "0%",
        "sharpe": "~0",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "low (uncorrelated)",
        "verdict": "KILLED",
        "tier": "killed",
        "notes": "Uncorrelated to EXP-1220 but zero alpha. 28 tests passing, strategy is a no-op.",
        "commit": "a24fe50",
    },
    {
        "id": "EXP-1840",
        "name": "IV Spike Entry",
        "type": "Short vol after IV spikes",
        "data": "IronVault (SPY options)",
        "cagr": "marginal",
        "sharpe": "weak",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "—",
        "verdict": "KILLED",
        "tier": "killed",
        "notes": "IV spikes revert but not consistently enough to beat costs. Honest result.",
        "commit": "e3567cf",
    },
    {
        "id": "EXP-1760",
        "name": "Earnings Vol Crush (Index Proxy)",
        "type": "IV crush around earnings",
        "data": "IronVault (SPY macro proxy)",
        "cagr": "N/A",
        "sharpe": "N/A",
        "max_dd": "—",
        "wr": "—",
        "corr_1220": "—",
        "verdict": "KILLED",
        "tier": "killed",
        "notes": "Data gap — need individual stock options for proper earnings play. Index proxy is too diluted.",
        "commit": "192a876",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# HTML Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_row(e):
    tier_colors = {
        "winner": ("#dcfce7", "#16a34a"),
        "promising": ("#dbeafe", "#2563eb"),
        "killed": ("#fee2e2", "#dc2626"),
    }
    bg, fg = tier_colors.get(e["tier"], ("#f8fafc", "#64748b"))

    return f"""<tr>
        <td style="text-align:left;font-weight:700;color:{fg}">{e['id']}</td>
        <td style="text-align:left;font-weight:600">{e['name']}</td>
        <td style="text-align:left;font-size:0.72rem;color:#64748b">{e['type']}</td>
        <td style="text-align:left;font-size:0.72rem;color:#475569">{e['data']}</td>
        <td>{e['cagr']}</td>
        <td>{e['sharpe']}</td>
        <td>{e['max_dd']}</td>
        <td>{e['wr']}</td>
        <td style="font-size:0.72rem">{e['corr_1220']}</td>
        <td><span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;font-weight:700;font-size:0.72rem">{e['verdict']}</span></td>
    </tr>"""


def build_notes_row(e):
    return f"""<div style="margin:8px 0;padding:10px;background:#f8fafc;border-left:3px solid #94a3b8;border-radius:4px">
        <div style="font-weight:700;color:#1e293b;font-size:0.85rem">{e['id']} — {e['name']}</div>
        <div style="font-size:0.78rem;color:#475569;margin-top:4px">{e['notes']}</div>
        <div style="font-size:0.68rem;color:#94a3b8;margin-top:4px">Commit: {e['commit']} | Data: {e['data']}</div>
    </div>"""


def build_report():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    date_str = "2026-04-06"

    winners = [e for e in EXPERIMENTS if e["tier"] == "winner"]
    promising = [e for e in EXPERIMENTS if e["tier"] == "promising"]
    killed = [e for e in EXPERIMENTS if e["tier"] == "killed"]

    # Summary table (all experiments)
    all_rows = "".join(build_row(e) for e in winners + promising + killed)

    # Section tables
    winner_rows = "".join(build_row(e) for e in winners)
    promising_rows = "".join(build_row(e) for e in promising)
    killed_rows = "".join(build_row(e) for e in killed)

    # Detailed notes
    winner_notes = "".join(build_notes_row(e) for e in winners)
    promising_notes = "".join(build_notes_row(e) for e in promising)
    killed_notes = "".join(build_notes_row(e) for e in killed)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PilotAI Experiment Report — {date_str}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin: 0; padding: 32px; background: #f1f5f9; color: #0f172a;
    max-width: 1400px; margin-left: auto; margin-right: auto;
  }}
  .header {{
    background: linear-gradient(135deg, #1e40af 0%, #0f172a 100%);
    color: white; padding: 32px; border-radius: 12px; margin-bottom: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }}
  .header h1 {{ margin: 0 0 8px; font-size: 1.8rem; }}
  .header .subtitle {{ color: #cbd5e1; font-size: 0.9rem; }}
  .header .meta {{ color: #94a3b8; font-size: 0.78rem; margin-top: 8px; }}

  .summary {{
    background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  }}
  .summary-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
  }}
  .summary-card {{
    background: #f8fafc; padding: 14px; border-radius: 8px;
    border-left: 3px solid #2563eb;
  }}
  .summary-card.winner {{ border-color: #16a34a; }}
  .summary-card.promising {{ border-color: #2563eb; }}
  .summary-card.killed {{ border-color: #dc2626; }}
  .summary-card .label {{
    font-size: 0.68rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.05em; font-weight: 600;
  }}
  .summary-card .value {{
    font-size: 1.6rem; font-weight: 700; color: #0f172a; margin-top: 4px;
  }}

  .section {{
    background: white; padding: 24px; border-radius: 10px; margin-bottom: 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  }}
  .section h2 {{
    margin: 0 0 16px; padding-bottom: 8px; font-size: 1.15rem;
    border-bottom: 2px solid #e2e8f0;
  }}
  .section h2.winners {{ color: #16a34a; border-color: #86efac; }}
  .section h2.promising {{ color: #2563eb; border-color: #93c5fd; }}
  .section h2.killed {{ color: #dc2626; border-color: #fca5a5; }}

  table {{
    width: 100%; border-collapse: collapse; font-size: 0.8rem;
  }}
  th {{
    background: #f1f5f9; padding: 8px 10px; text-align: right;
    font-size: 0.7rem; color: #475569; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.03em;
    border-bottom: 2px solid #cbd5e1;
  }}
  th:first-child, th:nth-child(2), th:nth-child(3), th:nth-child(4) {{ text-align: left; }}
  td {{
    padding: 8px 10px; text-align: right; border-bottom: 1px solid #f1f5f9;
    color: #1e293b;
  }}
  tr:hover td {{ background: #f8fafc; }}

  .rule-zero {{
    background: #fef2f2; border: 2px solid #dc2626; border-radius: 10px;
    padding: 16px; margin: 16px 0; color: #7f1d1d;
  }}
  .rule-zero strong {{ color: #991b1b; }}

  .portfolio-reco {{
    background: #f0fdf4; border: 2px solid #16a34a; border-radius: 10px;
    padding: 20px; margin: 20px 0;
  }}
  .portfolio-reco h3 {{
    margin: 0 0 12px; color: #166534; font-size: 1.05rem;
  }}

  .footer {{
    text-align: center; padding: 20px; color: #94a3b8; font-size: 0.72rem;
    margin-top: 32px;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>PilotAI Experiment Report</h1>
  <div class="subtitle">Comprehensive strategy evaluation — {date_str}</div>
  <div class="meta">Generated {ts} | For Carlos | All data from real market sources</div>
</div>

<div class="rule-zero">
  <strong>RULE ZERO COMPLIANCE:</strong> Every result below is from real market data.
  Option prices from IronVault (Polygon historical options cache). Spot prices from
  Yahoo Finance. Zero synthetic pricing. All Sharpe ratios use the corrected
  arithmetic daily mean formula — NOT CAGR-derived (which inflates by 1.2-2.4x).
</div>

<div class="summary">
  <div class="summary-grid">
    <div class="summary-card winner">
      <div class="label">Total Experiments</div>
      <div class="value">{len(EXPERIMENTS)}</div>
    </div>
    <div class="summary-card winner">
      <div class="label">Deployed</div>
      <div class="value" style="color:#16a34a">{len(winners)}</div>
    </div>
    <div class="summary-card promising">
      <div class="label">Promising</div>
      <div class="value" style="color:#2563eb">{len(promising)}</div>
    </div>
    <div class="summary-card killed">
      <div class="label">Killed (honest)</div>
      <div class="value" style="color:#dc2626">{len(killed)}</div>
    </div>
  </div>
</div>

<div class="portfolio-reco">
  <h3>Recommended Portfolio Construction</h3>
  <table>
    <thead><tr><th>Role</th><th>Strategy</th><th>Weight</th><th>Expected Contribution</th></tr></thead>
    <tbody>
      <tr>
        <td style="text-align:left"><strong>Core</strong></td>
        <td style="text-align:left">EXP-1220 @ 1.5x leverage</td>
        <td>75%</td>
        <td style="text-align:left">~75% CAGR, Sharpe 3.8, DD -11%</td>
      </tr>
      <tr>
        <td style="text-align:left"><strong>Uncorrelated Alpha</strong></td>
        <td style="text-align:left">EXP-1710 1DTE IC (adaptive)</td>
        <td>15%</td>
        <td style="text-align:left">Near-zero correlation, decay protection overlay</td>
      </tr>
      <tr>
        <td style="text-align:left"><strong>Diversifier</strong></td>
        <td style="text-align:left">EXP-1820 Dispersion</td>
        <td>5%</td>
        <td style="text-align:left">Capacity-limited to ~$5M, -0.03 monthly corr</td>
      </tr>
      <tr>
        <td style="text-align:left"><strong>Crisis Insurance</strong></td>
        <td style="text-align:left">EXP-1780 v3 (Crisis Alpha)</td>
        <td>5%</td>
        <td style="text-align:left">Activates only in crashes: +18.6% vs SPY during 37 DD periods</td>
      </tr>
    </tbody>
  </table>
</div>

<div class="section">
  <h2>All Experiments Summary</h2>
  <table>
    <thead><tr>
      <th>ID</th><th>Name</th><th>Type</th><th>Data Source</th>
      <th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>WR</th>
      <th>ρ EXP-1220</th><th>Verdict</th>
    </tr></thead>
    <tbody>{all_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2 class="winners">WINNERS — Deploy</h2>
  <table>
    <thead><tr>
      <th>ID</th><th>Name</th><th>Type</th><th>Data Source</th>
      <th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>WR</th>
      <th>ρ EXP-1220</th><th>Verdict</th>
    </tr></thead>
    <tbody>{winner_rows}</tbody>
  </table>
  <div style="margin-top:16px">{winner_notes}</div>
</div>

<div class="section">
  <h2 class="promising">PROMISING — Needs Work</h2>
  <table>
    <thead><tr>
      <th>ID</th><th>Name</th><th>Type</th><th>Data Source</th>
      <th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>WR</th>
      <th>ρ EXP-1220</th><th>Verdict</th>
    </tr></thead>
    <tbody>{promising_rows}</tbody>
  </table>
  <div style="margin-top:16px">{promising_notes}</div>
</div>

<div class="section">
  <h2 class="killed">KILLED — Honest Failures</h2>
  <table>
    <thead><tr>
      <th>ID</th><th>Name</th><th>Type</th><th>Data Source</th>
      <th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>WR</th>
      <th>ρ EXP-1220</th><th>Verdict</th>
    </tr></thead>
    <tbody>{killed_rows}</tbody>
  </table>
  <div style="margin-top:16px">{killed_notes}</div>
</div>

<div class="section">
  <h2>Methodology Notes</h2>
  <ul style="font-size:0.85rem;line-height:1.6;color:#475569">
    <li><strong>Sharpe formula:</strong> arithmetic mean of daily excess returns / std × sqrt(252).
      NOT (CAGR - rf) / vol, which overstates by 1.2-2.4x at high return levels.</li>
    <li><strong>Walk-forward:</strong> expanding window where possible, with IS = pre-2023 and OOS = 2023-2025
      for strategies with sufficient data.</li>
    <li><strong>Transaction costs:</strong> Commissions included ($0.65/leg). Bid-ask implicit in IronVault close prices.</li>
    <li><strong>Correlation:</strong> Yearly where sample is small; monthly where possible (EXP-1820).</li>
    <li><strong>Capacity:</strong> Max AUM = 5% of ATM ADV participation per trade.</li>
    <li><strong>Failures are honest:</strong> 9 strategies killed for zero or negative edge. This is how the process works.</li>
  </ul>
</div>

<div class="footer">
  PilotAI Multi-Strategy Research | Real IronVault + Yahoo Finance data |
  Corrected Sharpe formula throughout | {date_str}
</div>

</body>
</html>"""


def main():
    html = build_report()
    out = ROOT / "reports" / "full_experiment_report_20260406.html"
    out.write_text(html, encoding="utf-8")
    print(f"Report written to: {out}")

    # Stats
    winners = sum(1 for e in EXPERIMENTS if e["tier"] == "winner")
    promising = sum(1 for e in EXPERIMENTS if e["tier"] == "promising")
    killed = sum(1 for e in EXPERIMENTS if e["tier"] == "killed")
    print(f"  Total: {len(EXPERIMENTS)}")
    print(f"  Winners: {winners}")
    print(f"  Promising: {promising}")
    print(f"  Killed: {killed}")


if __name__ == "__main__":
    main()
