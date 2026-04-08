#!/usr/bin/env python3
"""
Run backtest_auditor on ALL experiments. Generate full_audit_report.html.
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.backtest_auditor import BacktestAuditor, AuditReport

EXPERIMENTS_DIR = ROOT / "experiments"
REPORTS_DIR = ROOT / "reports"

# Map experiment IDs to their source code files for synthetic/look-ahead checks
CODE_MAP = {
    "EXP-880": "compass/ml_strategy.py",
    "EXP-1220": "compass/tail_risk_hedge.py",
    "EXP-1230": "compass/microstructure_alpha.py",
    "EXP-1270": "compass/adaptive_stoploss.py",
    "EXP-1320": "compass/intraday_vol_clustering.py",
    "EXP-1470": "compass/north_star_integrator.py",
    "EXP-1630": "compass/gld_tlt_relval.py",
    "EXP-1640": "compass/cross_asset_momentum.py",
    "EXP-1650": "compass/iron_condor_optimizer.py",
    "EXP-1660": "compass/vrp_harvester.py",
}

# Known experiment metadata (data source, ticker)
KNOWN_META = {
    "EXP-880-real":  {"data": "IronVault", "ticker": "SPY"},
    "EXP-880-max":   {"data": "synthetic/heuristic", "ticker": "SPY"},
    "EXP-1220-real": {"data": "Yahoo Finance (real)", "ticker": "SPY"},
    "EXP-1220-max":  {"data": "synthetic", "ticker": "SPY"},
    "EXP-1230-real": {"data": "IronVault + Yahoo", "ticker": "SPY"},
    "EXP-1270-real": {"data": "IronVault", "ticker": "SPY"},
    "EXP-1320-real": {"data": "IronVault", "ticker": "SPY"},
    "EXP-1470-real": {"data": "IronVault + Polygon", "ticker": "SPY"},
    "EXP-1470-max":  {"data": "synthetic", "ticker": "SPY"},
    "EXP-1630-max":  {"data": "IronVault (real)", "ticker": "GLD"},
    "EXP-1640-max":  {"data": "IronVault + Yahoo", "ticker": "XLF"},
    "EXP-1650-max":  {"data": "IronVault", "ticker": "XLF"},
    "EXP-1660-max":  {"data": "IronVault", "ticker": "SPY"},
}


def load_summary(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def extract_trades(summary: dict) -> list:
    """Extract trade list from various summary formats."""
    if "trades" in summary and isinstance(summary["trades"], list):
        return summary["trades"]
    # Some summaries store trades per year
    if "yearly" in summary:
        trades = []
        for yr_data in summary["yearly"].values():
            if isinstance(yr_data, dict) and "trades" in yr_data:
                trades.extend(yr_data["trades"])
        if trades:
            return trades
    return []


def get_exp_name(exp_dir: str) -> str:
    """Extract experiment ID from directory name."""
    return exp_dir.replace("/results/summary.json", "").split("/")[-1]


def audit_experiment(exp_id: str, summary: dict) -> AuditReport:
    """Run full audit on one experiment."""
    auditor = BacktestAuditor()

    trades = extract_trades(summary)
    sharpe = summary.get("sharpe", summary.get("sharpe_ratio",
             summary.get("full_sharpe", summary.get("oos_sharpe", 0))))
    cagr = summary.get("cagr", summary.get("cagr_pct", summary.get("return_pct", 0)))
    if isinstance(cagr, (int, float)) and cagr > 5:
        cagr = cagr / 100  # convert percentage to decimal

    # Determine data source
    meta = KNOWN_META.get(exp_id, {})
    data_source = meta.get("data", "")
    if not data_source:
        # Infer from -real vs -max suffix
        if "-real" in exp_id:
            data_source = "IronVault (real)"
        elif "-max" in exp_id:
            data_source = "synthetic/heuristic"

    ticker = meta.get("ticker", "SPY")

    # Find source code
    base_id = re.sub(r"-(max|real|validation|paper)$", "", exp_id)
    code_path = CODE_MAP.get(base_id, "")
    if code_path:
        code_path = str(ROOT / code_path)

    # Check for commissions/slippage in summary
    has_comm = summary.get("commission_per_contract", 0) > 0 or "commission" in str(summary).lower()
    has_slip = summary.get("slippage", 0) > 0 or "slippage" in str(summary).lower()

    n_trades = summary.get("n_trades", summary.get("total_trades", len(trades)))
    n_days = summary.get("n_days", summary.get("trading_days", 1260))

    return auditor.audit(
        results=summary,
        trades=trades,
        reported_sharpe=sharpe,
        reported_cagr=cagr,
        data_source=data_source,
        code_path=code_path,
        ticker=ticker,
        has_commissions=has_comm,
        has_slippage=has_slip,
        n_days=n_days,
    )


def generate_full_report(results: list, output_path: str) -> str:
    """Generate consolidated HTML report for all experiments."""

    # Count by grade
    grades = {}
    for exp_id, report in results:
        g = report.overall_grade
        grades[g] = grades.get(g, 0) + 1

    n_total = len(results)
    n_pass = sum(1 for _, r in results if r.overall_grade in ("A", "B+"))
    n_warn = sum(1 for _, r in results if r.overall_grade in ("B", "C"))
    n_fail = sum(1 for _, r in results if r.overall_grade in ("D", "F"))

    # Summary table
    summary_rows = ""
    for exp_id, report in sorted(results, key=lambda x: {"A":0,"B+":1,"B":2,"C":3,"D":4,"F":5}.get(x[1].overall_grade, 9)):
        gc = {"A": "#059669", "B+": "#059669", "B": "#d97706", "C": "#d97706",
              "D": "#dc2626", "F": "#dc2626"}.get(report.overall_grade, "#6b7280")
        check_icons = ""
        for c in report.checks:
            ic = {"PASS": "#059669", "WARNING": "#d97706", "FAIL": "#dc2626", "CRITICAL": "#dc2626"}[c.severity]
            icon = {"PASS": "&#10003;", "WARNING": "!", "FAIL": "&#10007;", "CRITICAL": "!!"}[c.severity]
            short = c.name.replace(" Check", "")[:8]
            check_icons += f'<span style="color:{ic}" title="{c.name}: {c.message}">{icon}</span> '

        # Get key issues
        issues = [c.name.replace(" Check","") for c in report.checks if c.severity in ("FAIL","CRITICAL")]
        warnings = [c.name.replace(" Check","") for c in report.checks if c.severity == "WARNING"]
        issue_str = ", ".join(issues) if issues else (", ".join(warnings) if warnings else "—")
        issue_color = "#dc2626" if issues else ("#d97706" if warnings else "#6b7280")

        suffix = ""
        if "-real" in exp_id:
            suffix = '<span style="background:#dcfce7;color:#166534;padding:1px 5px;border-radius:3px;font-size:.7em;margin-left:4px">REAL</span>'
        elif "-max" in exp_id:
            suffix = '<span style="background:#fee2e2;color:#991b1b;padding:1px 5px;border-radius:3px;font-size:.7em;margin-left:4px">SYNTH</span>'

        summary_rows += (
            f'<tr>'
            f'<td style="text-align:left"><strong>{exp_id}</strong>{suffix}</td>'
            f'<td style="color:{gc};font-weight:700;font-size:1.1em;text-align:center">{report.overall_grade}</td>'
            f'<td style="text-align:center">{report.n_passed}</td>'
            f'<td style="text-align:center;color:#d97706">{report.n_warnings}</td>'
            f'<td style="text-align:center;color:#dc2626">{report.n_failed + report.n_critical}</td>'
            f'<td style="text-align:left">{check_icons}</td>'
            f'<td style="text-align:left;color:{issue_color};font-size:.82em">{issue_str}</td>'
            f'</tr>\n'
        )

    # Per-check failure matrix
    check_names = ["Dilution", "Synthetic Data", "Look-Ahead", "Sharpe Formula",
                   "Survivorship", "Transaction Cost", "Capacity"]
    matrix_rows = ""
    for exp_id, report in sorted(results, key=lambda x: x[0]):
        cells = ""
        for c in report.checks:
            ic = {"PASS": "#059669", "WARNING": "#d97706", "FAIL": "#dc2626", "CRITICAL": "#dc2626"}[c.severity]
            cells += f'<td style="text-align:center;color:{ic};font-weight:600">{c.severity[0]}</td>'
        matrix_rows += f'<tr><td style="text-align:left;font-size:.8em">{exp_id}</td>{cells}</tr>\n'

    # Grade distribution
    grade_dist = ""
    for g in ["A", "B+", "B", "C", "D", "F"]:
        cnt = grades.get(g, 0)
        gc = {"A": "#059669", "B+": "#059669", "B": "#d97706", "C": "#d97706",
              "D": "#dc2626", "F": "#dc2626"}.get(g, "#6b7280")
        bar_w = max(cnt * 8, 0)
        grade_dist += (
            f'<tr><td style="color:{gc};font-weight:700;width:40px">{g}</td>'
            f'<td style="text-align:left"><div style="background:{gc};height:18px;width:{bar_w}px;'
            f'border-radius:3px;display:inline-block"></div> {cnt}</td></tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Full Backtest Audit — All Experiments</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e5e7eb;--text:#111827;--muted:#6b7280}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;max-width:1300px;margin:0 auto;padding:24px}}
h1{{font-size:1.5rem;font-weight:800;margin-bottom:4px}}
h2{{font-size:1.1rem;font-weight:700;margin:28px 0 10px;border-bottom:2px solid var(--border);padding-bottom:6px}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:16px}}
.hero{{background:var(--card);border:2px solid var(--border);border-radius:10px;padding:20px;text-align:center;margin:16px 0}}
.hero .big{{font-size:1.8rem;font-weight:800}}
.hero .sub{{color:var(--muted);font-size:.9rem;margin-top:4px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:16px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;text-align:center}}
.c .l{{color:var(--muted);font-size:.72em;text-transform:uppercase}}.c .v{{font-weight:700;font-size:1.1em;margin-top:2px}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.82em}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--border)}}
th{{background:#f1f5f9;color:var(--muted);font-size:.72em;font-weight:600;text-transform:uppercase;text-align:left}}
tr:hover td{{background:#f8fafc}}
.note{{color:var(--muted);font-size:.82em;margin:8px 0}}
.box{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin:14px 0}}
</style></head><body>

<h1>Full Backtest Integrity Audit</h1>
<p class="subtitle">{n_total} experiments audited &bull; 7 checks per experiment &bull; {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="hero">
  <div class="big" style="color:{'#059669' if n_pass > n_fail else '#dc2626'}">{n_pass} PASS &bull; {n_warn} WARN &bull; {n_fail} FAIL</div>
  <div class="sub">Grade A/B+: trustworthy &bull; Grade B/C: use with caution &bull; Grade D/F: do not use</div>
</div>

<div class="cards">
  <div class="c"><div class="l">Total Experiments</div><div class="v">{n_total}</div></div>
  <div class="c"><div class="l">Grade A/B+</div><div class="v" style="color:#059669">{n_pass}</div></div>
  <div class="c"><div class="l">Grade B/C</div><div class="v" style="color:#d97706">{n_warn}</div></div>
  <div class="c"><div class="l">Grade D/F</div><div class="v" style="color:#dc2626">{n_fail}</div></div>
  <div class="c"><div class="l">Real Data</div><div class="v">{sum(1 for e,_ in results if '-real' in e)}</div></div>
  <div class="c"><div class="l">Synthetic</div><div class="v">{sum(1 for e,_ in results if '-max' in e)}</div></div>
</div>

<h2>Grade Distribution</h2>
<table style="max-width:400px"><tbody>{grade_dist}</tbody></table>

<h2>Full Scorecard (sorted by grade)</h2>
<table>
<thead><tr><th>Experiment</th><th style="text-align:center">Grade</th><th style="text-align:center">Pass</th><th style="text-align:center">Warn</th><th style="text-align:center">Fail</th><th>Checks</th><th>Key Issues</th></tr></thead>
<tbody>{summary_rows}</tbody></table>

<h2>Check Matrix (P=Pass, W=Warning, F=Fail, C=Critical)</h2>
<table>
<thead><tr><th>Experiment</th>{''.join(f'<th style="text-align:center;font-size:.68em">{n[:6]}</th>' for n in check_names)}</tr></thead>
<tbody>{matrix_rows}</tbody></table>

<div class="box">
<h2 style="border:none;margin-top:0">Audit Methodology</h2>
<p class="note">Each experiment is checked for 7 integrity issues discovered during the project:</p>
<ol style="padding-left:20px;font-size:.85em;line-height:2">
<li><strong>Dilution</strong> — &gt;50% zero-return days inflate Sharpe by reducing measured vol</li>
<li><strong>Synthetic Data</strong> — np.random, Black-Scholes, or heuristic pricing instead of real market data</li>
<li><strong>Look-Ahead Bias</strong> — future data used in entry/exit decisions</li>
<li><strong>Sharpe Formula</strong> — must use arithmetic returns, not CAGR-based (which inflates 1.07-2.4&times;)</li>
<li><strong>Survivorship Bias</strong> — expired/delisted options properly handled</li>
<li><strong>Transaction Costs</strong> — bid-ask spreads + commissions included</li>
<li><strong>Capacity</strong> — trades within &lt;5% of ATM daily volume</li>
</ol>
<p class="note" style="margin-top:10px">Grading: A = all pass, B+ = 1 warning, B = 2 warnings, C = 1 fail, D = 2+ fails, F = any critical.</p>
</div>

<p class="note" style="text-align:center;margin-top:32px;padding-top:12px;border-top:1px solid var(--border)">
  Backtest Integrity Audit &bull; compass/backtest_auditor.py &bull; {datetime.now().strftime('%Y-%m-%d')}
</p>
</body></html>"""

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p)


REPORT_EXPERIMENTS = {
    "EXP-1630 (GLD/TLT)": {"path": "reports/exp1630_gld_tlt_relval.json",
        "data": "IronVault (real)", "ticker": "GLD", "code": "compass/gld_tlt_relval.py"},
    "Vol Term Structure": {"path": "reports/vol_term_structure_deep_dive.json",
        "data": "IronVault (real)", "ticker": "SPY", "code": "compass/vol_term_structure_deep_dive.py"},
    "Cross-Asset Pairs": {"path": "reports/cross_asset_pairs_deep_validation.json",
        "data": "IronVault (real)", "ticker": "TLT", "code": ""},
    "XLI Iron Condors": {"path": "reports/xlf_iron_condor_optimization.json",
        "data": "IronVault (real)", "ticker": "XLI", "code": "compass/iron_condor_optimizer.py"},
}


def main():
    print("Scanning experiments...")
    summaries = sorted(EXPERIMENTS_DIR.glob("*/results/summary.json"))
    print(f"Found {len(summaries)} experiment summaries\n")

    results = []
    for path in summaries:
        exp_id = path.parent.parent.name
        summary = load_summary(path)
        if not summary:
            print(f"  {exp_id}: empty/invalid summary — SKIP")
            continue

        report = audit_experiment(exp_id, summary)
        results.append((exp_id, report))

    # Also audit report-based experiments (the key validated ones)
    print("\n  --- Report-based experiments ---")
    auditor = BacktestAuditor()
    for exp_id, meta in REPORT_EXPERIMENTS.items():
        rp = ROOT / meta["path"]
        if not rp.exists():
            continue
        summary = load_summary(rp)
        if not summary:
            continue
        code = str(ROOT / meta["code"]) if meta["code"] else ""
        report = auditor.audit(
            results=summary, data_source=meta["data"], code_path=code,
            ticker=meta["ticker"], has_slippage=True,
        )
        results.append((exp_id, report))

        checks_str = " ".join(
            {"PASS": ".", "WARNING": "W", "FAIL": "F", "CRITICAL": "!"}[c.severity]
            for c in report.checks
        )
        print(f"  {exp_id:<25} Grade {report.overall_grade:<3} [{checks_str}]  "
              f"P={report.n_passed} W={report.n_warnings} F={report.n_failed} C={report.n_critical}")

    # Summary
    print(f"\n{'='*60}")
    print(f"AUDIT COMPLETE: {len(results)} experiments")
    grades = {}
    for _, r in results:
        grades[r.overall_grade] = grades.get(r.overall_grade, 0) + 1
    for g in ["A", "B+", "B", "C", "D", "F"]:
        if g in grades:
            print(f"  Grade {g}: {grades[g]}")

    # Generate report
    out = str(REPORTS_DIR / "full_audit_report.html")
    generate_full_report(results, out)
    print(f"\nReport: {out}")

    return results


if __name__ == "__main__":
    main()
