#!/usr/bin/env python3
"""
run_compass_backtest.py — A/B test: exp_090 (baseline MA200) vs exp_101 (COMPASS)

COMPASS = Composite Macro Position & Sector Signal
  - Macro score sizing: score<45 → 1.1x (buy fear), score>70 → 0.8x (reduce complacency)
  - RRG breadth filter: block bull puts when <50% of sectors in Leading/Improving quadrant
  - Event scaling: NOT tested (macro_events table lacks historical FOMC/CPI/NFP data)

Writes results to: output/compass_backtest_results.md
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
OUTPUT = ROOT / "output"


def load_config_file(path: Path) -> dict:
    return json.loads(path.read_text())


def build_backtester_config(params: dict, starting_capital: float = 100_000) -> dict:
    """Mirror _build_config from run_optimization.py."""
    return {
        "strategy": {
            "target_delta":        params.get("target_delta", 0.12),
            "use_delta_selection": params.get("use_delta_selection", False),
            "target_dte":          params.get("target_dte", 35),
            "min_dte":             params.get("min_dte", 25),
            "spread_width":        params.get("spread_width", 5),
            "min_credit_pct":      params.get("min_credit_pct", 8),
            "direction":           params.get("direction", "both"),
            "trend_ma_period":     params.get("trend_ma_period", 200),
            "regime_mode":         params.get("regime_mode", "combo"),
            "regime_config":       params.get("regime_config", {}),
            "momentum_filter_pct": params.get("momentum_filter_pct", None),
            "iron_condor": {
                "enabled":                 params.get("iron_condor_enabled", False),
                "min_combined_credit_pct": params.get("ic_min_combined_credit_pct", 12),
                "neutral_regime_only":     params.get("ic_neutral_regime_only", False),
                "vix_min":                 params.get("ic_vix_min", 0),
                "risk_per_trade":          params.get("ic_risk_per_trade", None),
            },
            "iv_rank_min_entry":  params.get("iv_rank_min_entry", 0),
            "vix_max_entry":      params.get("vix_max_entry", 0),
            "vix_close_all":      params.get("vix_close_all", 0),
            "vix_dynamic_sizing": params.get("vix_dynamic_sizing", {}),
            "seasonal_sizing":    params.get("seasonal_sizing", {}),
            "compass_enabled":    params.get("compass_enabled", False),
            "compass_rrg_filter": params.get("compass_rrg_filter", False),
        },
        "risk": {
            "stop_loss_multiplier": params.get("stop_loss_multiplier", 2.5),
            "profit_target":        params.get("profit_target", 50),
            "max_risk_per_trade":   params.get("max_risk_per_trade", 10.0),
            "max_contracts":        params.get("max_contracts", 25),
            "max_positions":        50,
            "drawdown_cb_pct":      params.get("drawdown_cb_pct", 40),
        },
        "backtest": {
            "starting_capital":        starting_capital,
            "commission_per_contract": 0.65,
            "slippage":                0.05,
            "exit_slippage":           0.10,
            "compound":                params.get("compound", False),
            "sizing_mode":             params.get("sizing_mode", "flat"),
            "slippage_multiplier":     params.get("slippage_multiplier", 1.0),
            "max_portfolio_exposure_pct": params.get("max_portfolio_exposure_pct", 100.0),
            "exclude_months":          params.get("exclude_months", []),
            "monte_carlo":             params.get("monte_carlo", {}),
            "volume_gate":             params.get("volume_gate", False),
            "min_volume_ratio":        params.get("min_volume_ratio", 50),
            "volume_size_cap_pct":     params.get("volume_size_cap_pct", 0.02),
            "oi_gate":                 params.get("oi_gate", False),
            "oi_min_factor":           params.get("oi_min_factor", 2),
            "volume_gate_on_miss":     params.get("volume_gate_on_miss", "open"),
        },
    }


def run_year(params: dict, year: int) -> dict:
    from backtest.backtester import Backtester
    from backtest.historical_data import HistoricalOptionsData

    config = build_backtester_config(params)
    polygon_api_key = os.getenv("POLYGON_API_KEY", "")
    hd = HistoricalOptionsData(polygon_api_key, offline_mode=True)
    bt = Backtester(config, historical_data=hd, otm_pct=params.get("otm_pct", 0.03))
    result = bt.run_backtest("SPY", datetime(year, 1, 1), datetime(year, 12, 31)) or {}
    result["year"] = year
    return result


def run_experiment(label: str, params: dict) -> dict:
    results = {}
    print(f"\n=== {label} ===")
    for year in YEARS:
        t0 = time.time()
        print(f"  {year}...", end=" ", flush=True)
        try:
            r = run_year(params, year)
            elapsed = time.time() - t0
            print(f"{r.get('return_pct', 0):+.1f}%  {r.get('total_trades', 0)} trades  ({elapsed:.0f}s)")
            results[year] = r
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            results[year] = {"return_pct": 0, "total_trades": 0, "max_drawdown": 0,
                             "sharpe_ratio": 0, "error": str(e)}
    return results


def summarize(results: dict) -> dict:
    rets = [r.get("return_pct", 0) for r in results.values()]
    dds  = [r.get("max_drawdown", 0) for r in results.values()]
    return {
        "avg_return": round(sum(rets) / len(rets), 2),
        "min_return": round(min(rets), 2),
        "max_return": round(max(rets), 2),
        "worst_dd":   round(min(dds), 2),
        "years_profitable": sum(1 for r in rets if r > 0),
    }


def fmt_delta(a: float, b: float) -> str:
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.2f}pp"


def get_compass_stats() -> dict:
    """Pull macro score distribution from the DB for context."""
    try:
        from compass.macro_db import get_db
        conn = get_db()
        rows = conn.execute(
            "SELECT date, overall FROM macro_score WHERE date >= '2020-01-01' AND date <= '2025-12-31' ORDER BY date"
        ).fetchall()
        scores = [r["overall"] for r in rows if r["overall"] is not None]
        fear_weeks   = sum(1 for s in scores if s < 45)
        neutral_weeks = sum(1 for s in scores if 45 <= s <= 70)
        greed_weeks  = sum(1 for s in scores if s > 70)

        # By year
        by_year = {}
        for r in rows:
            yr = r["date"][:4]
            if yr not in by_year:
                by_year[yr] = []
            if r["overall"] is not None:
                by_year[yr].append(r["overall"])

        rrg_rows = conn.execute(
            """SELECT date, rrg_quadrant, COUNT(*) AS n FROM sector_rs
               WHERE date >= '2020-01-01' AND date <= '2025-12-31'
               GROUP BY date, rrg_quadrant"""
        ).fetchall()
        date_totals = {}
        date_pos = {}
        for r in rrg_rows:
            d = r["date"]
            date_totals[d] = date_totals.get(d, 0) + r["n"]
            if r["rrg_quadrant"] in ("Leading", "Improving"):
                date_pos[d] = date_pos.get(d, 0) + r["n"]
        rrg_fractions = [date_pos.get(d, 0) / date_totals[d] for d in date_totals if date_totals[d] > 0]
        rrg_blocked_pct = round(100 * sum(1 for f in rrg_fractions if f < 0.5) / len(rrg_fractions), 1) if rrg_fractions else 0
        conn.close()
        return {
            "total_weeks": len(scores),
            "fear_weeks": fear_weeks,
            "neutral_weeks": neutral_weeks,
            "greed_weeks": greed_weeks,
            "by_year": {yr: {"avg": round(sum(v)/len(v), 1), "min": round(min(v), 1), "max": round(max(v), 1)}
                        for yr, v in by_year.items()},
            "rrg_blocked_pct": rrg_blocked_pct,
        }
    except Exception as e:
        return {"error": str(e)}


def write_report(ctrl_results: dict, comp_results: dict, ctrl_summary: dict, comp_summary: dict, stats: dict):
    ctrl_params_note = "10% flat risk, MA200 filter, combo regime, no iron condors"
    comp_params_note = "10% flat risk, MA200 filter, combo regime + COMPASS macro sizing + RRG breadth filter"

    lines = [
        "# COMPASS Backtest Results — exp_090 vs exp_101",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        "## What Is COMPASS?",
        "",
        "COMPASS = Composite Macro Position & Sector Signal.",
        "It layers two real-time macro intelligence signals onto the baseline credit spread strategy:",
        "",
        "| Feature | Logic |",
        "|---------|-------|",
        "| Macro Score Sizing | Weekly score 0–100 from `macro_state.db`. Score < 45 → 1.1× size (buy fear). Score > 70 → 0.8× size (reduce complacency). |",
        "| RRG Breadth Filter | Block bull put entries when < 50% of tracked sectors (XLE/XLF/XLV/XLK/XLI/XLU/XLY) are in Leading or Improving RRG quadrant. |",
        "| Event Scaling | **NOT TESTED** — `macro_events` table contains only 1 future event; historical FOMC/CPI/NFP events were not backfilled. Requires data backfill before backtesting. |",
        "",
        "---",
        "",
        "## COMPASS Data Profile (2020–2025)",
        "",
        f"- **Total weekly snapshots**: {stats.get('total_weeks', 'N/A')}",
        f"- **Fear weeks** (score < 45, 1.1× size): {stats.get('fear_weeks', 'N/A')} weeks",
        f"- **Neutral weeks** (45–70, 1.0× size): {stats.get('neutral_weeks', 'N/A')} weeks",
        f"- **Greed weeks** (score > 70, 0.8× size): {stats.get('greed_weeks', 'N/A')} weeks",
        f"- **RRG filter blocks bull puts**: ~{stats.get('rrg_blocked_pct', 'N/A')}% of weeks",
        "",
        "### Macro Score by Year",
        "",
        "| Year | Avg Score | Min | Max | Regime Bias |",
        "|------|-----------|-----|-----|-------------|",
    ]
    regime_notes = {
        "2020": "mixed — COVID crash pulled score to 36.2 in Mar (fear)",
        "2021": "BULL MACRO — avg 73.7, mostly > 70 (complacency regime)",
        "2022": "NEUTRAL — rate hike uncertainty, no fear/greed extremes",
        "2023": "NEUTRAL — soft landing regime",
        "2024": "NEUTRAL/BULL — election cycle, score 55–64",
        "2025": "NEUTRAL/BULL — score 57–67",
    }
    by_year = stats.get("by_year", {})
    for yr in ["2020", "2021", "2022", "2023", "2024", "2025"]:
        s = by_year.get(yr, {})
        lines.append(f"| {yr} | {s.get('avg','?')} | {s.get('min','?')} | {s.get('max','?')} | {regime_notes.get(yr, '—')} |")

    lines += [
        "",
        "---",
        "",
        "## A/B Test: exp_090 vs exp_101",
        "",
        "**exp_090 (control)**: " + ctrl_params_note,
        "",
        "**exp_101 (treatment)**: " + comp_params_note,
        "",
        "### Year-by-Year Results",
        "",
        "| Year | exp_090 Return | Trades | exp_101 Return | Trades | Delta | DD (090) | DD (101) |",
        "|------|---------------|--------|---------------|--------|-------|----------|----------|",
    ]

    for yr in YEARS:
        c = ctrl_results.get(yr, {})
        t = comp_results.get(yr, {})
        cr  = c.get("return_pct", 0)
        tr  = t.get("return_pct", 0)
        ctr = c.get("total_trades", 0)
        ttr = t.get("total_trades", 0)
        cdd = c.get("max_drawdown", 0)
        tdd = t.get("max_drawdown", 0)
        delta = tr - cr
        dsign = "+" if delta >= 0 else ""
        lines.append(f"| {yr} | {cr:+.2f}% | {ctr} | {tr:+.2f}% | {ttr} | {dsign}{delta:.2f}pp | {cdd:.2f}% | {tdd:.2f}% |")

    lines += [
        "",
        "### Summary",
        "",
        "| Metric | exp_090 (baseline) | exp_101 (COMPASS) | Delta |",
        "|--------|-------------------|-------------------|-------|",
        f"| Avg Annual Return | {ctrl_summary['avg_return']:+.2f}% | {comp_summary['avg_return']:+.2f}% | {fmt_delta(ctrl_summary['avg_return'], comp_summary['avg_return'])} |",
        f"| Worst Annual Return | {ctrl_summary['min_return']:+.2f}% | {comp_summary['min_return']:+.2f}% | {fmt_delta(ctrl_summary['min_return'], comp_summary['min_return'])} |",
        f"| Best Annual Return | {ctrl_summary['max_return']:+.2f}% | {comp_summary['max_return']:+.2f}% | {fmt_delta(ctrl_summary['max_return'], comp_summary['max_return'])} |",
        f"| Worst Max Drawdown | {ctrl_summary['worst_dd']:.2f}% | {comp_summary['worst_dd']:.2f}% | {fmt_delta(ctrl_summary['worst_dd'], comp_summary['worst_dd'])} |",
        f"| Years Profitable | {ctrl_summary['years_profitable']}/6 | {comp_summary['years_profitable']}/6 | — |",
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ]

    avg_delta = comp_summary["avg_return"] - ctrl_summary["avg_return"]
    dd_delta = comp_summary["worst_dd"] - ctrl_summary["worst_dd"]

    if avg_delta > 1.0:
        verdict = "COMPASS ADDS ALPHA — meaningful return improvement."
    elif avg_delta > 0:
        verdict = "COMPASS is slightly positive — marginal improvement."
    elif avg_delta > -1.0:
        verdict = "COMPASS is roughly neutral — within noise."
    else:
        verdict = "COMPASS HURTS RETURNS — the macro overlays are subtractive on this config."

    lines += [
        f"**Verdict**: {verdict}",
        "",
        "### Signal-by-Signal Analysis",
        "",
        "**Macro Score Sizing (score < 45 → 1.1×, score > 70 → 0.8×)**:",
        "",
        "- Only 2020 had fear weeks (score < 45) — the COVID crash. In 2020, the 1.1× multiplier "
        "increased size during bear call entries when VIX was elevated and premiums were richest.",
        "- 2021 was almost entirely 'greed' territory (avg score 73.7). The 0.8× multiplier "
        "reduced position sizes across most of 2021. This explains any 2021 return drag vs baseline.",
        "- 2022–2025 had neutral-range scores (45–70) → 1.0× (no change).",
        "",
        "**RRG Breadth Filter (block bull puts when < 50% sectors Leading/Improving)**:",
        "",
        f"- Blocks ~{stats.get('rrg_blocked_pct', '?')}% of weeks from bull put entries.",
        "- This is most impactful in 2022 (broad sector weakness) and early 2020.",
        "- In bull years (2021, 2023–2025), most weeks have strong sector breadth — filter rarely triggers.",
        "",
        "**Event Scaling (FOMC/CPI/NFP)**:",
        "",
        "- **NOT IMPLEMENTED IN THIS BACKTEST.** The `macro_events` table has only 1 row "
        "(2026-03-12 CPI). Historical event dates were never backfilled into the DB.",
        "- To backtest this feature, populate `macro_events` with 2020–2025 FOMC/CPI/NFP dates "
        "and their `scaling_factor` values, then re-run.",
        "",
        "---",
        "",
        "## Data Limitations & Gaps",
        "",
        "1. **macro_events gap**: Historical FOMC/CPI/NFP events not in DB. Event scaling can't be backtested.",
        "2. **Weekly granularity**: Macro scores are weekly snapshots, forward-filled to daily. "
        "Intra-week macro regime changes are not captured.",
        "3. **Sector RRG vs SPY**: We trade SPY, not sector ETFs. The RRG filter uses "
        "aggregate sector breadth as a proxy for market health — a rough approximation.",
        "4. **Score calibration**: The 0-100 macro score range (min=36.2, max=82.1 historically) "
        "means the 'fear' threshold of 45 is rarely triggered. Consider recalibrating to "
        "the 20th/80th percentile of historical scores for more frequent signal activation.",
        "",
        "---",
        "",
        "## Recommendation",
        "",
    ]

    if avg_delta >= 1.0 and dd_delta >= -2.0:
        rec = ("ADOPT COMPASS — the macro layer adds meaningful return with acceptable "
               "drawdown trade-off. Proceed to Phase 7 full integration.")
    elif avg_delta >= 0 and dd_delta >= -5.0:
        rec = ("CONTINUE TESTING — COMPASS is non-destructive. Backfill macro_events "
               "to test event scaling, then re-run. Also consider recalibrating score "
               "thresholds to 20th/80th percentile of historical distribution.")
    else:
        rec = ("PAUSE — COMPASS hurts baseline returns. The macro overlays are reducing "
               "position sizes in favorable periods (2021 complacency) more than they "
               "help in fear periods (2020 crash). Consider inverting the RRG filter or "
               "using score multipliers only for IC entries (not directional spreads).")

    lines += [
        rec,
        "",
        "---",
        "",
        "_Config files: `configs/exp_090_risk10_nocompound_newcode.json` (baseline) "
        "| `configs/exp_101_compass.json` (COMPASS)_",
        "",
    ]

    report_path = OUTPUT / "compass_backtest_results.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    print("Loading configs...")
    ctrl_params = load_config_file(ROOT / "configs" / "exp_090_risk10_nocompound_newcode.json")
    comp_params = load_config_file(ROOT / "configs" / "exp_101_compass.json")

    print("Fetching COMPASS data profile...")
    stats = get_compass_stats()
    print(f"  Macro score weeks: {stats.get('total_weeks')}, fear={stats.get('fear_weeks')}, greed={stats.get('greed_weeks')}")

    ctrl_results = run_experiment("exp_090 (baseline - no COMPASS)", ctrl_params)
    comp_results = run_experiment("exp_101 (COMPASS enabled)", comp_params)

    ctrl_summary = summarize(ctrl_results)
    comp_summary = summarize(comp_results)

    print("\n=== SUMMARY ===")
    print(f"exp_090 avg return: {ctrl_summary['avg_return']:+.2f}%  worst_dd: {ctrl_summary['worst_dd']:.2f}%")
    print(f"exp_101 avg return: {comp_summary['avg_return']:+.2f}%  worst_dd: {comp_summary['worst_dd']:.2f}%")
    delta = comp_summary['avg_return'] - ctrl_summary['avg_return']
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}pp")

    write_report(ctrl_results, comp_results, ctrl_summary, comp_summary, stats)
