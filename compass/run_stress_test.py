"""
compass/run_stress_test.py — Phase 6 Stress Test Runner

Derives daily returns from the trade-level training data CSVs (no backtester
needed), runs StressTester for EXP-400, EXP-401, and a blended portfolio, then
generates:
  - reports/stress_test_results.json   raw results (all three experiments)
  - reports/stress_test_report.html    formatted HTML report

Usage:
    python3 -m compass.run_stress_test
    # or
    python3 compass/run_stress_test.py

Success criteria (from MASTERPLAN):
  - MC 5th-percentile max DD <= 30%
  - All crisis scenarios DD <= 40%
  - No cliff parameters in sensitivity
  - Report generated
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.stress_test import StressTester
from compass.crisis_hedge import CrisisHedgeConfig, CrisisHedgeController, get_hedge_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_stress_test")

STARTING_CAPITAL = 100_000
N_SIMULATIONS = 10_000


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Derive daily returns from trade-level CSVs
# ─────────────────────────────────────────────────────────────────────────────

def _load_daily_returns(csv_path: Path, starting_capital: float) -> pd.Series:
    """Aggregate trade PnL by exit_date and convert to daily return series.

    Trades are attributed to their exit date (realised P&L date).  All
    trading weekdays in the date range get a return; days with no exits get 0.
    Return = sum(pnl_exiting_that_day) / starting_capital.
    """
    df = pd.read_csv(csv_path, parse_dates=["exit_date", "entry_date"])
    daily_pnl = df.groupby("exit_date")["pnl"].sum()

    # Full weekday calendar spanning the dataset
    start = df["entry_date"].min()
    end = df["exit_date"].max()
    all_days = pd.bdate_range(start=start, end=end)

    daily_pnl = daily_pnl.reindex(all_days, fill_value=0.0)
    daily_returns = daily_pnl / starting_capital
    daily_returns.index.name = "date"
    return daily_returns


def _blend_returns(r1: pd.Series, r2: pd.Series) -> pd.Series:
    """Equal-weight blend of two daily return series on their common date range."""
    combined = pd.concat([r1.rename("r1"), r2.rename("r2")], axis=1, sort=True).fillna(0.0)
    blended = 0.5 * combined["r1"] + 0.5 * combined["r2"]
    blended.index.name = "date"
    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b: Crisis-hedge-adjusted daily returns
# ─────────────────────────────────────────────────────────────────────────────

def _load_daily_returns_hedged(
    csv_path: Path,
    starting_capital: float,
    controller: CrisisHedgeController,
) -> pd.Series:
    """Same as _load_daily_returns but with CrisisHedgeController applied.

    For each trade row:
      - position_scale_factor(vix, regime) scales PnL (proxy for sizing reduction)
      - For stop-loss exits, additionally applies stop_loss_multiplier tightening:
        an early exit ratio = stop_multiplier / base_stop_multiplier caps the loss

    The hedged PnL is then aggregated by exit_date, same as the unhedged path.
    """
    base_stop = controller.cfg.base_stop_multiplier
    df = pd.read_csv(csv_path, parse_dates=["exit_date", "entry_date"])

    hedged_pnl = []
    for _, row in df.iterrows():
        vix = float(row.get("vix", 20.0)) if not pd.isna(row.get("vix", float("nan"))) else 20.0
        regime = str(row.get("regime", "neutral")) if not pd.isna(row.get("regime", float("nan"))) else "neutral"
        pnl = float(row["pnl"])

        scale = controller.position_scale_factor(vix=vix, regime=regime)

        # For losing stop-loss exits, tighten cap via stop multiplier
        if pnl < 0 and str(row.get("exit_reason", "")).lower() == "stop_loss":
            stop_mult = controller.stop_loss_multiplier(vix=vix, regime=regime)
            # Tighter stop = less loss: cap loss at stop_mult/base_stop fraction
            tighten_ratio = stop_mult / base_stop  # 1.0 normal → <1.0 tighter
            pnl = max(pnl * tighten_ratio, pnl)    # tighten can only reduce loss

        hedged_pnl.append(pnl * scale)

    df["hedged_pnl"] = hedged_pnl
    daily_pnl = df.groupby("exit_date")["hedged_pnl"].sum()

    start = df["entry_date"].min()
    end = df["exit_date"].max()
    all_days = pd.bdate_range(start=start, end=end)

    daily_pnl = daily_pnl.reindex(all_days, fill_value=0.0)
    daily_returns = daily_pnl / starting_capital
    daily_returns.index.name = "date"
    return daily_returns


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Run StressTester for one experiment
# ─────────────────────────────────────────────────────────────────────────────

def _run_stress(
    name: str,
    daily_returns: pd.Series,
    crisis_hedge_config: Optional[CrisisHedgeConfig] = None,
) -> Dict[str, Any]:
    log.info("Running stress test for %s (%d trading days)…", name, len(daily_returns))
    tester = StressTester(
        daily_returns.values,
        starting_capital=STARTING_CAPITAL,
        n_simulations=N_SIMULATIONS,
        block_size=5,
        seed=42,
    )
    results = tester.run_all(crisis_hedge_config=crisis_hedge_config)
    results["experiment"] = name
    results["n_trading_days"] = int(len(daily_returns))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: MASTERPLAN success-criteria checker
# ─────────────────────────────────────────────────────────────────────────────

def _check_criteria(results: Dict[str, Any]) -> Dict[str, bool]:
    mc = results["monte_carlo"]
    crisis = results["crisis_scenarios"]

    p5_dd = abs(mc["max_drawdown"]["percentiles_pct"].get("p5", 0))
    # Use hedged DD for pass/fail when available, otherwise fall back to unhedged
    crisis_dds = []
    for c in crisis:
        hedged = c.get("hedged_portfolio_drawdown_pct")
        crisis_dds.append(abs(hedged) if hedged is not None else abs(c["portfolio_drawdown_pct"]))
    max_crisis_dd = max(crisis_dds) if crisis_dds else 0

    # Cliff detection: look for sensitivity Sharpe dropping >50% across adjacent values
    no_cliffs = True
    for param_data in results["sensitivity"].values():
        sharpes = [r["sharpe"] for r in param_data["results"] if r["sharpe"] != 0]
        for i in range(1, len(sharpes)):
            if sharpes[i - 1] != 0 and abs(sharpes[i] - sharpes[i - 1]) / abs(sharpes[i - 1]) > 0.5:
                no_cliffs = False
                break

    return {
        "mc_p5_dd_le_30pct": p5_dd <= 30.0,
        "all_crisis_dd_le_40pct": max_crisis_dd <= 40.0,
        "no_cliff_parameters": no_cliffs,
        "p5_dd": round(p5_dd, 2),
        "max_crisis_dd": round(max_crisis_dd, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: HTML report generation
# ─────────────────────────────────────────────────────────────────────────────

_RISK_COLOR = {
    "LOW":      ("#d4edda", "#155724"),
    "MODERATE": ("#fff3cd", "#856404"),
    "HIGH":     ("#f8d7da", "#721c24"),
    "CRITICAL": ("#f5c6cb", "#491217"),
}


def _risk_badge(rating: str) -> str:
    bg, fg = _RISK_COLOR.get(rating, ("#e2e3e5", "#383d41"))
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:4px;font-weight:600;font-size:0.85em;">{rating}</span>'
    )


def _fmt_pct(v: float, decimals: int = 1) -> str:
    return f"{v:+.{decimals}f}%"


def _fmt_usd(v: float) -> str:
    return f"${v:,.0f}"


def _mc_table(mc: Dict) -> str:
    tw = mc["terminal_wealth"]
    dd = mc["max_drawdown"]
    sr = mc["sharpe_ratio"]
    rows = [
        ("Terminal Wealth — P5 / Median / P95",
         f"{_fmt_usd(tw['percentiles']['p5'])} / {_fmt_usd(tw['percentiles']['p50'])} / {_fmt_usd(tw['percentiles']['p95'])}"),
        ("Max Drawdown — P5 / Median / P95",
         f"{dd['percentiles_pct']['p5']:.1f}% / {dd['median_pct']:.1f}% / {dd['percentiles_pct']['p95']:.1f}%"),
        ("Sharpe Ratio (median)",   f"{sr['median']:.3f}"),
        ("Prob of Profit",          f"{mc['prob_profit']*100:.1f}%"),
        ("Prob of Ruin (−50%+)",    f"{mc['prob_ruin_50pct']*100:.2f}%"),
        ("Horizon (trading days)",  str(mc["horizon_days"])),
        ("MC Paths",                f"{mc['n_simulations']:,}"),
    ]
    body = "".join(
        f'<tr><td style="color:#555;padding:5px 10px;">{k}</td>'
        f'<td style="padding:5px 10px;font-weight:500;">{v}</td></tr>'
        for k, v in rows
    )
    return f'<table style="border-collapse:collapse;width:100%">{body}</table>'


def _crisis_table(crisis: List[Dict]) -> str:
    has_hedged = any(c.get("hedged_portfolio_drawdown_pct") is not None for c in crisis)
    hedged_header = (
        "<th style='padding:8px;text-align:right;'>Hedged DD</th>" if has_hedged else ""
    )
    header = (
        "<tr style='background:#f8f9fa'>"
        "<th style='padding:8px;text-align:left;'>Scenario</th>"
        "<th style='padding:8px;text-align:right;'>Underlying DD</th>"
        "<th style='padding:8px;text-align:right;'>Unhedged DD</th>"
        f"{hedged_header}"
        "<th style='padding:8px;text-align:right;'>Trough Value</th>"
        "<th style='padding:8px;text-align:right;'>Recovery (days)</th>"
        "<th style='padding:8px;text-align:right;'>VIX</th>"
        "<th style='padding:8px;text-align:center;'>Pass?</th>"
        "</tr>"
    )
    rows = ""
    for c in crisis:
        # Use hedged DD for pass/fail check when available
        hedged_dd = c.get("hedged_portfolio_drawdown_pct")
        effective_dd = abs(hedged_dd) if hedged_dd is not None else abs(c["portfolio_drawdown_pct"])
        passed = effective_dd <= 40.0
        tick = "✅" if passed else "❌"
        unhedged_dd = abs(c["portfolio_drawdown_pct"])
        unhedged_style = "color:#721c24;font-weight:600;" if unhedged_dd > 40.0 else ""
        hedged_cell = ""
        if has_hedged:
            if hedged_dd is not None:
                hedged_style = "color:#721c24;font-weight:600;" if abs(hedged_dd) > 40.0 else "color:#155724;font-weight:600;"
                hedged_cell = f"<td style='padding:8px;text-align:right;{hedged_style}'>{hedged_dd:.1f}%</td>"
            else:
                hedged_cell = "<td style='padding:8px;text-align:right;color:#999'>N/A</td>"
        rows += (
            f"<tr style='border-top:1px solid #dee2e6'>"
            f"<td style='padding:8px'><strong>{c['name']}</strong><br>"
            f"<small style='color:#666'>{c['description']}</small></td>"
            f"<td style='padding:8px;text-align:right;'>{c['underlying_drawdown_pct']:.1f}%</td>"
            f"<td style='padding:8px;text-align:right;{unhedged_style}'>{c['portfolio_drawdown_pct']:.1f}%</td>"
            f"{hedged_cell}"
            f"<td style='padding:8px;text-align:right;'>{_fmt_usd(c['trough_value'])}</td>"
            f"<td style='padding:8px;text-align:right;'>"
            f"{'N/A' if c['estimated_recovery_days'] is None else c['estimated_recovery_days']}</td>"
            f"<td style='padding:8px;text-align:right;'>{c['vix_start']}→{c['vix_peak']}</td>"
            f"<td style='padding:8px;text-align:center;font-size:1.2em;'>{tick}</td>"
            f"</tr>"
        )
    return f'<table style="border-collapse:collapse;width:100%;font-size:0.9em">{header}{rows}</table>'


def _sensitivity_section(sensitivity: Dict) -> str:
    sections = []
    for param_name, param_data in sensitivity.items():
        label = param_data["label"]
        desc = param_data["description"]
        results = param_data["results"]

        # Header row
        header_cells = "".join(
            f'<th style="padding:6px 10px;text-align:right;'
            f'{"background:#e8f4fd;" if r["is_baseline"] else ""}">'
            f'{"★ " if r["is_baseline"] else ""}{r["value"]}</th>'
            for r in results
        )

        # Sharpe row — color-coded
        def _sharpe_bg(s: float) -> str:
            if s >= 1.5:   return "#d4edda"
            if s >= 0.8:   return "#fff3cd"
            if s >= 0.0:   return "#fde8c8"
            return "#f8d7da"

        sharpe_cells = "".join(
            f'<td style="padding:6px 10px;text-align:right;background:{_sharpe_bg(r["sharpe"])};'
            f'{"font-weight:700;" if r["is_baseline"] else ""}">{r["sharpe"]:.2f}</td>'
            for r in results
        )

        # DD row
        dd_cells = "".join(
            f'<td style="padding:6px 10px;text-align:right;">{r["max_dd_pct"]:.1f}%</td>'
            for r in results
        )

        # CAGR row
        cagr_cells = "".join(
            f'<td style="padding:6px 10px;text-align:right;">{r["cagr_pct"]:.1f}%</td>'
            for r in results
        )

        table = (
            f'<table style="border-collapse:collapse;width:100%;font-size:0.85em;margin-top:6px">'
            f'<tr style="background:#f8f9fa">'
            f'<th style="padding:6px 10px;text-align:left;min-width:100px"></th>'
            f'{header_cells}</tr>'
            f'<tr><td style="padding:6px 10px;color:#555;">Sharpe</td>{sharpe_cells}</tr>'
            f'<tr style="background:#f8f9fa"><td style="padding:6px 10px;color:#555;">Max DD</td>{dd_cells}</tr>'
            f'<tr><td style="padding:6px 10px;color:#555;">CAGR</td>{cagr_cells}</tr>'
            f'</table>'
        )

        sections.append(
            f'<div style="margin-bottom:24px">'
            f'<h4 style="margin:0 0 4px;font-size:0.95em">{label}</h4>'
            f'<p style="margin:0 0 6px;font-size:0.8em;color:#666">{desc} '
            f'<em>(★ = baseline)</em></p>'
            f'{table}</div>'
        )
    return "\n".join(sections)


def _summary_card(exp_results: Dict, criteria: Dict) -> str:
    name = exp_results["experiment"]
    s = exp_results["summary"]
    hist = s["historical"]
    mc_conf = s["monte_carlo_confidence"]
    worst = s["worst_crisis"]
    rating = s["risk_rating"]
    bg, fg = _RISK_COLOR.get(rating, ("#e2e3e5", "#383d41"))

    return (
        f'<div style="border:1px solid #dee2e6;border-radius:8px;padding:20px;'
        f'background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.06)">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start">'
        f'<h3 style="margin:0 0 12px;font-size:1.1em">{name}</h3>'
        f'{_risk_badge(rating)}</div>'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.88em">'
        f'<tr><td style="color:#555;padding:3px 0">Trading days</td>'
        f'<td style="text-align:right;font-weight:500">{hist["n_days"]}</td></tr>'
        f'<tr><td style="color:#555;padding:3px 0">Hist. Sharpe</td>'
        f'<td style="text-align:right;font-weight:500">{hist["sharpe"]:.3f}</td></tr>'
        f'<tr><td style="color:#555;padding:3px 0">Hist. Max DD</td>'
        f'<td style="text-align:right;font-weight:500">{hist["max_drawdown_pct"]:.1f}%</td></tr>'
        f'<tr><td style="color:#555;padding:3px 0">Hist. CAGR</td>'
        f'<td style="text-align:right;font-weight:500">{hist["cagr_pct"]:.1f}%</td></tr>'
        f'<tr><td style="color:#555;padding:3px 0">MC P5 DD</td>'
        f'<td style="text-align:right;font-weight:500;'
        f'{"color:#721c24;" if not criteria["mc_p5_dd_le_30pct"] else ""}">'
        f'{mc_conf["median_max_dd_pct"]:.1f}% med / '
        f'{criteria["p5_dd"]:.1f}% P5</td></tr>'
        f'<tr><td style="color:#555;padding:3px 0">Prob Profit</td>'
        f'<td style="text-align:right;font-weight:500">{mc_conf["prob_profit_pct"]:.1f}%</td></tr>'
        f'<tr><td style="color:#555;padding:3px 0">Prob Ruin</td>'
        f'<td style="text-align:right;font-weight:500">{mc_conf["prob_ruin_pct"]:.2f}%</td></tr>'
        f'<tr><td style="color:#555;padding:3px 0">Worst crisis</td>'
        f'<td style="text-align:right;font-weight:500;'
        f'{"color:#721c24;" if not criteria["all_crisis_dd_le_40pct"] else ""}">'
        f'{worst["name"].split("(")[0].strip()}: {worst["portfolio_drawdown_pct"]:.1f}%</td></tr>'
        f'</table>'
        f'<div style="margin-top:12px;padding:8px;border-radius:4px;font-size:0.8em;'
        f'background:{bg};color:{fg}">'
        f'<strong>MASTERPLAN checks:</strong> '
        f'{"✅" if criteria["mc_p5_dd_le_30pct"] else "❌"} MC P5 DD ≤ 30%&nbsp; '
        f'{"✅" if criteria["all_crisis_dd_le_40pct"] else "❌"} Crisis DD ≤ 40%&nbsp; '
        f'{"✅" if criteria["no_cliff_parameters"] else "❌"} No cliff params'
        f'</div>'
        f'</div>'
    )


def _comparison_table(all_results: List[Dict]) -> str:
    rows_html = ""
    metrics = [
        ("Hist. Sharpe",   lambda r: f"{r['summary']['historical']['sharpe']:.3f}"),
        ("Hist. Max DD",   lambda r: f"{r['summary']['historical']['max_drawdown_pct']:.1f}%"),
        ("Hist. CAGR",     lambda r: f"{r['summary']['historical']['cagr_pct']:.1f}%"),
        ("MC Median DD",   lambda r: f"{r['summary']['monte_carlo_confidence']['median_max_dd_pct']:.1f}%"),
        ("MC P5 DD",       lambda r: f"{abs(r['monte_carlo']['max_drawdown']['percentiles_pct']['p5']):.1f}%"),
        ("P5 Terminal",    lambda r: _fmt_usd(r['monte_carlo']['terminal_wealth']['percentiles']['p5'])),
        ("P50 Terminal",   lambda r: _fmt_usd(r['monte_carlo']['terminal_wealth']['percentiles']['p50'])),
        ("P95 Terminal",   lambda r: _fmt_usd(r['monte_carlo']['terminal_wealth']['percentiles']['p95'])),
        ("Prob Profit",    lambda r: f"{r['monte_carlo']['prob_profit']*100:.1f}%"),
        ("Prob Ruin",      lambda r: f"{r['monte_carlo']['prob_ruin_50pct']*100:.2f}%"),
        ("Risk Rating",    lambda r: _risk_badge(r['summary']['risk_rating'])),
    ]

    header = "".join(
        f'<th style="padding:8px 14px;text-align:center;font-weight:600">{r["experiment"]}</th>'
        for r in all_results
    )

    for label, fn in metrics:
        cells = "".join(
            f'<td style="padding:8px 14px;text-align:center">{fn(r)}</td>'
            for r in all_results
        )
        rows_html += (
            f'<tr style="border-top:1px solid #dee2e6">'
            f'<td style="padding:8px 14px;color:#555;font-size:0.9em">{label}</td>'
            f'{cells}</tr>'
        )

    return (
        f'<table style="border-collapse:collapse;width:100%;font-size:0.9em">'
        f'<thead><tr style="background:#f8f9fa">'
        f'<th style="padding:8px 14px;text-align:left;font-weight:600">Metric</th>'
        f'{header}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table>'
    )


def _hedge_impact_table(hedge_impact: List[Dict]) -> str:
    """Render before/after crisis hedge comparison table."""
    header = (
        "<tr style='background:#f8f9fa'>"
        "<th style='padding:8px 14px;text-align:left'>Experiment</th>"
        "<th style='padding:8px 14px;text-align:right'>Unhedged Sharpe</th>"
        "<th style='padding:8px 14px;text-align:right'>Hedged Sharpe</th>"
        "<th style='padding:8px 14px;text-align:right'>Unhedged MC P5 DD</th>"
        "<th style='padding:8px 14px;text-align:right'>Hedged MC P5 DD</th>"
        "<th style='padding:8px 14px;text-align:right'>Unhedged Crisis DD</th>"
        "<th style='padding:8px 14px;text-align:right'>Hedged Crisis DD</th>"
        "<th style='padding:8px 14px;text-align:center'>DD Improved?</th>"
        "</tr>"
    )
    rows = ""
    for h in hedge_impact:
        improved = h["hedged_crisis_dd"] < h["unhedged_crisis_dd"]
        tick = "✅" if improved else "—"
        rows += (
            f"<tr style='border-top:1px solid #dee2e6'>"
            f"<td style='padding:8px 14px;font-weight:500'>{h['name']}</td>"
            f"<td style='padding:8px 14px;text-align:right'>{h['unhedged_sharpe']:.3f}</td>"
            f"<td style='padding:8px 14px;text-align:right'>{h['hedged_sharpe']:.3f}</td>"
            f"<td style='padding:8px 14px;text-align:right'>{h['unhedged_p5_dd']:.1f}%</td>"
            f"<td style='padding:8px 14px;text-align:right'>{h['hedged_p5_dd']:.1f}%</td>"
            f"<td style='padding:8px 14px;text-align:right'>{h['unhedged_crisis_dd']:.1f}%</td>"
            f"<td style='padding:8px 14px;text-align:right'>{h['hedged_crisis_dd']:.1f}%</td>"
            f"<td style='padding:8px 14px;text-align:center;font-size:1.1em'>{tick}</td>"
            f"</tr>"
        )
    return (
        f'<table style="border-collapse:collapse;width:100%;font-size:0.9em">'
        f'<thead>{header}</thead><tbody>{rows}</tbody></table>'
    )


def _generate_html(
    all_results: List[Dict],
    all_criteria: List[Dict],
    hedge_impact: Optional[List[Dict]] = None,
) -> str:
    # Build experiment order: EXP-400, EXP-401, Blended
    summary_cards = "".join(
        f'<div style="flex:1;min-width:260px">{_summary_card(r, c)}</div>'
        for r, c in zip(all_results, all_criteria)
    )

    # Crisis tables per experiment (stacked)
    crisis_sections = ""
    for r in all_results:
        crisis_sections += (
            f'<h3 style="margin:28px 0 10px;font-size:1em;color:#333">'
            f'{r["experiment"]} — Crisis Scenarios</h3>'
            f'{_crisis_table(r["crisis_scenarios"])}'
        )

    # Sensitivity sections per experiment
    sensitivity_sections = ""
    for r in all_results:
        sensitivity_sections += (
            f'<h3 style="margin:28px 0 10px;font-size:1em;color:#333">'
            f'{r["experiment"]} — Sensitivity Analysis</h3>'
            f'{_sensitivity_section(r["sensitivity"])}'
        )

    # MC details per experiment
    mc_sections = ""
    for r in all_results:
        mc_sections += (
            f'<h3 style="margin:28px 0 10px;font-size:1em;color:#333">'
            f'{r["experiment"]} — Monte Carlo ({r["monte_carlo"]["n_simulations"]:,} paths, '
            f'block={r["monte_carlo"]["block_size"]} days)</h3>'
            f'{_mc_table(r["monte_carlo"])}'
        )

    # Crisis hedge impact section
    hedge_section = ""
    if hedge_impact:
        hedge_section = (
            f'<h2>Crisis Hedge Impact</h2>'
            f'<p style="font-size:0.85em;color:#666;margin-bottom:12px">'
            f'CrisisHedgeController applied: VIX-adaptive position sizing (floor=20, ceiling=50) '
            f'+ stop-loss tightening (base=3.5×, min=1.5×). '
            f'Impact shows how hedge would have modified historical PnL stream.'
            f'</p>'
            f'{_hedge_impact_table(hedge_impact)}'
        )

    overall_pass = all(
        c["mc_p5_dd_le_30pct"] and c["all_crisis_dd_le_40pct"] and c["no_cliff_parameters"]
        for c in all_criteria
    )
    banner_bg = "#d4edda" if overall_pass else "#f8d7da"
    banner_fg = "#155724" if overall_pass else "#721c24"
    banner_icon = "✅" if overall_pass else "❌"
    banner_msg = "All MASTERPLAN criteria passed" if overall_pass else "Some MASTERPLAN criteria failed"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 6 Stress Test Report — PilotAI Credit Spreads</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #ffffff;
    color: #212529;
    line-height: 1.5;
    padding: 32px 24px;
    max-width: 1200px;
    margin: 0 auto;
  }}
  h1 {{ font-size: 1.7em; font-weight: 700; margin-bottom: 6px; }}
  h2 {{ font-size: 1.2em; font-weight: 600; margin: 32px 0 14px;
        padding-bottom: 8px; border-bottom: 2px solid #dee2e6; }}
  h3 {{ font-size: 1.0em; }}
  .subtitle {{ color: #666; font-size: 0.95em; margin-bottom: 24px; }}
  .banner {{
    border-radius: 6px; padding: 12px 18px; margin-bottom: 28px;
    font-size: 0.95em; font-weight: 500;
  }}
  .cards-row {{
    display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 32px;
  }}
  .section {{ margin-bottom: 36px; }}
  code {{ font-family: monospace; background: #f8f9fa;
          padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
  @media (max-width: 700px) {{
    .cards-row {{ flex-direction: column; }}
  }}
</style>
</head>
<body>

<h1>Phase 6 Stress Test Report</h1>
<p class="subtitle">
  PilotAI Credit Spreads — EXP-400 vs EXP-401 vs Blended Portfolio<br>
  Data: 2020-01-01 → 2025-12-31 &nbsp;|&nbsp;
  MC: {all_results[0]['monte_carlo']['n_simulations']:,} block-bootstrap paths &nbsp;|&nbsp;
  Starting capital: $100,000
</p>

<div class="banner" style="background:{banner_bg};color:{banner_fg}">
  {banner_icon} <strong>{banner_msg}</strong> —
  MC P5 DD ≤ 30%, all crisis DD ≤ 40%, no cliff parameters
</div>

<h2>Executive Summary</h2>
<div class="cards-row">
  {summary_cards}
</div>

<h2>Side-by-Side Comparison</h2>
{_comparison_table(all_results)}

<h2>Monte Carlo Results</h2>
<p style="font-size:0.85em;color:#666;margin-bottom:12px">
  Block-bootstrap resampling (block=5 days) preserves volatility clustering.
  Returns are derived from realised trade PnL attributed to exit date.
</p>
{mc_sections}

<h2>Crisis Scenario Survival</h2>
<p style="font-size:0.85em;color:#666;margin-bottom:12px">
  Applies 1.5× beta multiplier for credit spread short-gamma exposure.
  Pass criterion: portfolio drawdown ≤ 40%.
</p>
{crisis_sections}

{hedge_section}

<h2>Sensitivity Analysis</h2>
<p style="font-size:0.85em;color:#666;margin-bottom:12px">
  Heuristic scaling model (no full backtest re-run).
  ★ marks the baseline value.  Sharpe colour: green ≥ 1.5, yellow ≥ 0.8,
  orange ≥ 0, red &lt; 0.  "Cliff" = Sharpe drop &gt; 50% between adjacent values.
</p>
{sensitivity_sections}

<hr style="margin:36px 0;border:none;border-top:1px solid #dee2e6">
<p style="font-size:0.78em;color:#999">
  Generated by <code>compass/run_stress_test.py</code> &nbsp;|&nbsp;
  Raw results: <code>reports/stress_test_results.json</code>
</p>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # ── 1. Load daily returns ──────────────────────────────────────────────
    log.info("Loading training data CSVs…")
    r400 = _load_daily_returns(ROOT / "compass/training_data_exp400.csv", STARTING_CAPITAL)
    r401 = _load_daily_returns(ROOT / "compass/training_data_exp401.csv", STARTING_CAPITAL)
    r_blend = _blend_returns(r400, r401)

    log.info("EXP-400: %d days, total pnl=%.0f", len(r400), r400.sum() * STARTING_CAPITAL)
    log.info("EXP-401: %d days, total pnl=%.0f", len(r401), r401.sum() * STARTING_CAPITAL)
    log.info("Blended: %d days", len(r_blend))

    # ── 1b. Crisis-hedge-adjusted returns ─────────────────────────────────
    hedge_ctrl_400 = CrisisHedgeController(get_hedge_config("EXP-400"))
    hedge_ctrl_401 = CrisisHedgeController(get_hedge_config("EXP-401"))
    log.info("Loading hedged daily returns (CrisisHedgeController active)…")
    r400_hedged = _load_daily_returns_hedged(
        ROOT / "compass/training_data_exp400.csv", STARTING_CAPITAL, hedge_ctrl_400
    )
    r401_hedged = _load_daily_returns_hedged(
        ROOT / "compass/training_data_exp401.csv", STARTING_CAPITAL, hedge_ctrl_401
    )
    log.info("EXP-400 hedged: %d days, total pnl=%.0f", len(r400_hedged), r400_hedged.sum() * STARTING_CAPITAL)
    log.info("EXP-401 hedged: %d days, total pnl=%.0f", len(r401_hedged), r401_hedged.sum() * STARTING_CAPITAL)

    # ── 2. Run StressTester ────────────────────────────────────────────────
    hedge_cfg_400 = get_hedge_config("EXP-400")
    hedge_cfg_401 = get_hedge_config("EXP-401")
    results_400   = _run_stress("EXP-400 (Champion CS)",         r400,    crisis_hedge_config=hedge_cfg_400)
    results_401   = _run_stress("EXP-401 (CS + SS Blend)",       r401,    crisis_hedge_config=hedge_cfg_401)
    results_blend = _run_stress("Blended (50% EXP-400 + 50% EXP-401)", r_blend, crisis_hedge_config=hedge_cfg_400)

    results_400_hedged = _run_stress("EXP-400 (Hedged)", r400_hedged, crisis_hedge_config=hedge_cfg_400)
    results_401_hedged = _run_stress("EXP-401 (Hedged)", r401_hedged, crisis_hedge_config=hedge_cfg_401)

    all_results = [results_400, results_401, results_blend]

    # ── 3. MASTERPLAN criteria ─────────────────────────────────────────────
    criteria_400   = _check_criteria(results_400)
    criteria_401   = _check_criteria(results_401)
    criteria_blend = _check_criteria(results_blend)
    all_criteria   = [criteria_400, criteria_401, criteria_blend]

    # Build hedge impact comparison
    def _worst_crisis_dd(r: Dict) -> float:
        return max(abs(c["portfolio_drawdown_pct"]) for c in r["crisis_scenarios"])

    hedge_impact = [
        {
            "name": "EXP-400",
            "unhedged_sharpe": results_400["summary"]["historical"]["sharpe"],
            "hedged_sharpe":   results_400_hedged["summary"]["historical"]["sharpe"],
            "unhedged_p5_dd":  abs(results_400["monte_carlo"]["max_drawdown"]["percentiles_pct"]["p5"]),
            "hedged_p5_dd":    abs(results_400_hedged["monte_carlo"]["max_drawdown"]["percentiles_pct"]["p5"]),
            "unhedged_crisis_dd": _worst_crisis_dd(results_400),
            "hedged_crisis_dd":   _worst_crisis_dd(results_400_hedged),
        },
        {
            "name": "EXP-401",
            "unhedged_sharpe": results_401["summary"]["historical"]["sharpe"],
            "hedged_sharpe":   results_401_hedged["summary"]["historical"]["sharpe"],
            "unhedged_p5_dd":  abs(results_401["monte_carlo"]["max_drawdown"]["percentiles_pct"]["p5"]),
            "hedged_p5_dd":    abs(results_401_hedged["monte_carlo"]["max_drawdown"]["percentiles_pct"]["p5"]),
            "unhedged_crisis_dd": _worst_crisis_dd(results_401),
            "hedged_crisis_dd":   _worst_crisis_dd(results_401_hedged),
        },
    ]

    log.info(
        "Hedge impact — EXP-400: crisis DD %.1f%% → %.1f%%  EXP-401: %.1f%% → %.1f%%",
        hedge_impact[0]["unhedged_crisis_dd"], hedge_impact[0]["hedged_crisis_dd"],
        hedge_impact[1]["unhedged_crisis_dd"], hedge_impact[1]["hedged_crisis_dd"],
    )

    for name, c in [("EXP-400", criteria_400), ("EXP-401", criteria_401), ("Blended", criteria_blend)]:
        log.info(
            "%s criteria — MC P5 DD ≤ 30%%: %s (%.1f%%)  "
            "crisis DD ≤ 40%%: %s (%.1f%%)  no cliffs: %s",
            name,
            "PASS" if c["mc_p5_dd_le_30pct"] else "FAIL",
            c["p5_dd"],
            "PASS" if c["all_crisis_dd_le_40pct"] else "FAIL",
            c["max_crisis_dd"],
            "PASS" if c["no_cliff_parameters"] else "FAIL",
        )

    # ── 4. Save JSON ───────────────────────────────────────────────────────
    json_out = reports_dir / "stress_test_results.json"

    def _json_safe(obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        raise TypeError(f"Not serialisable: {type(obj)}")

    payload = {
        "experiments": {
            r["experiment"]: {
                k: v for k, v in r.items()
                if k != "sample_paths"       # skip huge path arrays from JSON
            }
            for r in all_results
        },
        "criteria": {
            all_results[i]["experiment"]: all_criteria[i]
            for i in range(len(all_results))
        },
        "crisis_hedge_impact": hedge_impact,
    }
    # Also remove sample_paths from nested mc dict
    for exp_data in payload["experiments"].values():
        if "monte_carlo" in exp_data:
            exp_data["monte_carlo"].pop("sample_paths", None)

    with open(json_out, "w") as fh:
        json.dump(payload, fh, indent=2, default=_json_safe)
    log.info("Saved raw results → %s", json_out)

    # ── 5. Generate HTML ───────────────────────────────────────────────────
    html_out = reports_dir / "stress_test_report.html"
    html = _generate_html(all_results, all_criteria, hedge_impact=hedge_impact)
    with open(html_out, "w") as fh:
        fh.write(html)
    log.info("Saved HTML report → %s", html_out)

    # ── 6. Print summary to stdout ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STRESS TEST COMPLETE")
    print("=" * 60)
    for r, c in zip(all_results, all_criteria):
        name = r["experiment"]
        rating = r["summary"]["risk_rating"]
        mc = r["monte_carlo"]
        p5_dd = abs(mc["max_drawdown"]["percentiles_pct"]["p5"])
        worst_crisis = max(r["crisis_scenarios"], key=lambda x: abs(x["portfolio_drawdown_pct"]))
        pass_all = c["mc_p5_dd_le_30pct"] and c["all_crisis_dd_le_40pct"] and c["no_cliff_parameters"]
        print(f"\n  {name}")
        print(f"    Risk rating:      {rating}")
        print(f"    MC P5 DD:         {p5_dd:.1f}%  {'✅' if c['mc_p5_dd_le_30pct'] else '❌'}")
        unhedged_dd = abs(worst_crisis['portfolio_drawdown_pct'])
        hedged_dd_val = worst_crisis.get('hedged_portfolio_drawdown_pct')
        if hedged_dd_val is not None:
            print(f"    Worst crisis DD:  {unhedged_dd:.1f}% (unhedged) → {abs(hedged_dd_val):.1f}% (hedged)  {'✅' if c['all_crisis_dd_le_40pct'] else '❌'}")
        else:
            print(f"    Worst crisis DD:  {unhedged_dd:.1f}%  {'✅' if c['all_crisis_dd_le_40pct'] else '❌'}")
        print(f"    No cliff params:  {'✅' if c['no_cliff_parameters'] else '❌'}")
        print(f"    OVERALL:          {'✅ PASS' if pass_all else '❌ FAIL'}")
    print()
    print(f"  HTML report:  {html_out}")
    print(f"  JSON results: {json_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
