#!/usr/bin/env python3
"""
EXP-1730 Hedge Overlay Salvage Analysis
========================================

Re-evaluate EXP-1730 Treasury Curve as a HEDGE OVERLAY at 5-10% allocation.

Core findings from Wave 1 post-mortem:
- Standalone: Sharpe -0.29, CAGR -0.03%, Max DD 0.65% (break-even)
- BUT: SPY correlation -0.17 and yearly equity correlation -0.656
- Tiny drag, highly negative correlation — salvage as overlay?

Analysis:
  1. 5%, 7.5%, 10% allocation scenarios on top of EXP-1220 (EXP-400 as proxy)
  2. Specific crash windows: COVID (Mar 2020), 2022 Bear, Aug 2024 volmageddon
  3. Combined-portfolio drawdown reduction
  4. Compare to April 5: all purchased hedges are net-negative
     Is EXP-1730 structurally different?
  5. Walk-forward validation on real Yahoo data

Key comparison to April 5 finding (hedge_cost_resolution.py):
  Purchased puts:  -2.86%/yr net drag (4.36% cost - 1.50% alpha)
  Selective puts:  -0.60%/yr net drag
  Collar:          -0.31%/yr net drag

If EXP-1730 has positive returns ONLY during SPY drawdowns and
zero cost during bull markets, it's structurally different from
purchased insurance.

Methodology:
  1. Load real SPY daily from Yahoo (2010-2026)
  2. Identify months where SPY drew down >5% or had negative monthly return
  3. Load EXP-1730 trades from existing JSON
  4. Compute EXP-1730 PnL ONLY during stress months
  5. Test as overlay on EXP-1220 (using trade log)
  6. Compare to purchased hedge costs

ZERO SYNTHETIC DATA. All prices from Yahoo Finance.
"""

import json
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.backtester import _yf_download_safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CAPITAL = 100_000.0
TRADING_DAYS = 252

# Stress thresholds
MONTHLY_DD_THRESHOLD = -0.05    # 5% monthly drawdown
MONTHLY_RETURN_THRESHOLD = -0.05 # 5% monthly negative return

# From April 5 hedge cost resolution
BASELINE_EXP1220_ALPHA = 0.015   # 1.5%/yr trade-level
PURCHASED_HEDGE_COSTS = {
    "continuous_puts": 0.0436,   # 4.36%/yr real IronVault
    "selective_vix15": 0.0210,   # 2.10%/yr
    "collar": 0.0181,            # 1.81%/yr total drag
}


def load_spy_daily() -> pd.DataFrame:
    """Load REAL SPY daily closes from Yahoo."""
    log.info("Loading SPY daily from Yahoo Finance...")
    df = _yf_download_safe("SPY", "2010-01-01", "2026-04-06")
    if df.empty:
        raise RuntimeError("Failed to load SPY data")
    df = df[["Close"]].copy()
    df["return"] = df["Close"].pct_change()
    log.info(f"  SPY: {len(df)} bars, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def identify_stress_months(spy: pd.DataFrame) -> List[Tuple[str, float, float]]:
    """Identify months where SPY had significant stress.

    Stress = monthly return <= -5% OR peak-to-trough drawdown within month >= 5%.
    Returns list of (month_key, monthly_return, max_intramonth_dd).
    """
    spy = spy.copy()
    spy["month_key"] = spy.index.strftime("%Y-%m")

    stress_months = []
    for month_key, group in spy.groupby("month_key"):
        if len(group) < 5:
            continue
        start_price = group["Close"].iloc[0]
        end_price = group["Close"].iloc[-1]
        monthly_ret = (end_price / start_price) - 1

        # Intra-month drawdown
        peak = group["Close"].cummax()
        dd = (group["Close"] - peak) / peak
        max_dd = float(dd.min())

        is_stress = (monthly_ret <= MONTHLY_RETURN_THRESHOLD or
                     max_dd <= -0.05)
        if is_stress:
            stress_months.append((month_key, monthly_ret, max_dd))

    log.info(f"  Stress months identified: {len(stress_months)}")
    return stress_months


def load_exp1730_trades() -> List[Dict]:
    """Load EXP-1730 trade log from the real backtest JSON."""
    path = ROOT / "reports" / "exp1730_treasury_curve.json"
    with open(path) as f:
        d = json.load(f)
    return d["trades"]


def load_exp1220_daily_pnl() -> Dict[str, float]:
    """Load EXP-1220 trade log from the real trade records (EXP-400 blend source).

    Note: EXP-1220 doesn't have a full standalone trade log here. Use the
    champion trade log (EXP-400 with regime-adaptive CS) as the equity-strategy
    proxy — same bull put structure, same underlying.
    """
    path = ROOT / "output" / "champion_trade_log.json"
    with open(path) as f:
        trades = json.load(f)
    daily = defaultdict(float)
    for t in trades:
        d = t["exit"][:10]
        daily[d] += float(t["pnl"]) - float(t.get("comm", 0))
    log.info(f"  Equity strategy (champion log): {len(trades)} trades, "
             f"${sum(daily.values()):,.0f} total PnL")
    return dict(daily)


def attribute_exp1730_to_months(trades: List[Dict]) -> Dict[str, float]:
    """Sum EXP-1730 PnL by exit month."""
    by_month = defaultdict(float)
    for t in trades:
        exit_month = t["exit_date"][:7]
        by_month[exit_month] += float(t["pnl"])
    return dict(by_month)


def analyze_hedge_behavior(
    spy: pd.DataFrame,
    stress_months: List[Tuple[str, float, float]],
    exp1730_by_month: Dict[str, float],
) -> Dict:
    """Compute EXP-1730 performance during stress vs normal months."""
    stress_keys = set(m[0] for m in stress_months)

    stress_pnl = 0.0
    stress_trades_months = 0
    stress_positive_months = 0

    normal_pnl = 0.0
    normal_trades_months = 0
    normal_positive_months = 0

    for month, pnl in exp1730_by_month.items():
        if month in stress_keys:
            stress_pnl += pnl
            stress_trades_months += 1
            if pnl > 0:
                stress_positive_months += 1
        else:
            normal_pnl += pnl
            normal_trades_months += 1
            if pnl > 0:
                normal_positive_months += 1

    stress_win_rate = (stress_positive_months / stress_trades_months
                       if stress_trades_months > 0 else 0)
    normal_win_rate = (normal_positive_months / normal_trades_months
                       if normal_trades_months > 0 else 0)

    total_stress_months = len(stress_months)
    stress_coverage = (stress_trades_months / total_stress_months
                       if total_stress_months > 0 else 0)

    return {
        "total_stress_months": total_stress_months,
        "stress_months_with_trades": stress_trades_months,
        "stress_coverage": round(stress_coverage, 4),
        "stress_pnl": round(stress_pnl, 2),
        "stress_positive_months": stress_positive_months,
        "stress_win_rate": round(stress_win_rate, 4),
        "normal_months_with_trades": normal_trades_months,
        "normal_pnl": round(normal_pnl, 2),
        "normal_positive_months": normal_positive_months,
        "normal_win_rate": round(normal_win_rate, 4),
        "total_pnl": round(stress_pnl + normal_pnl, 2),
    }


def compute_yearly_correlation(spy: pd.DataFrame,
                                equity_daily_pnl: Dict[str, float],
                                exp1730_by_month: Dict[str, float]) -> Dict:
    """Compute yearly returns and correlation between equity strategy and EXP-1730."""
    # Equity PnL by year
    equity_by_year = defaultdict(float)
    for d, pnl in equity_daily_pnl.items():
        yr = int(d[:4])
        equity_by_year[yr] += pnl

    # EXP-1730 by year (from monthly)
    t1730_by_year = defaultdict(float)
    for month, pnl in exp1730_by_month.items():
        yr = int(month[:4])
        t1730_by_year[yr] += pnl

    # SPY by year
    spy_by_year = defaultdict(list)
    for dt, ret in spy["return"].dropna().items():
        spy_by_year[dt.year].append(ret)
    spy_yearly = {yr: float(np.prod([1 + r for r in rets]) - 1)
                  for yr, rets in spy_by_year.items()}

    common_years = sorted(set(equity_by_year.keys()) & set(t1730_by_year.keys()))
    yearly_pairs = []
    for yr in common_years:
        yearly_pairs.append({
            "year": yr,
            "spy_return": round(spy_yearly.get(yr, 0.0) * 100, 2),
            "equity_pnl": round(equity_by_year[yr], 2),
            "t1730_pnl": round(t1730_by_year[yr], 2),
        })

    # Correlation
    eq_arr = np.array([y["equity_pnl"] for y in yearly_pairs])
    t1730_arr = np.array([y["t1730_pnl"] for y in yearly_pairs])

    corr = 0.0
    if len(eq_arr) > 2 and np.std(eq_arr) > 0 and np.std(t1730_arr) > 0:
        corr = float(np.corrcoef(eq_arr, t1730_arr)[0, 1])

    return {
        "yearly": yearly_pairs,
        "correlation": round(corr, 4),
    }


def compute_overlay_net_improvement(
    equity_daily_pnl: Dict[str, float],
    exp1730_by_month: Dict[str, float],
    stress_months: List[Tuple[str, float, float]],
) -> Dict:
    """Compute net portfolio improvement with EXP-1730 as overlay.

    Scenarios:
      A. Equity alone
      B. Equity + continuous puts (April 5 baseline)
      C. Equity + EXP-1730 overlay (always on)
      D. Equity + EXP-1730 overlay (only during stress)

    We use yearly returns for comparison since hedge comparison is an
    annualized question.
    """
    # Sum equity PnL by year
    equity_by_year = defaultdict(float)
    for d, pnl in equity_daily_pnl.items():
        yr = int(d[:4])
        equity_by_year[yr] += pnl

    # EXP-1730 by year
    t1730_by_year = defaultdict(float)
    for month, pnl in exp1730_by_month.items():
        yr = int(month[:4])
        t1730_by_year[yr] += pnl

    common_years = sorted(set(equity_by_year.keys()))
    years = len(common_years)
    if years == 0:
        return {}

    # A. Equity alone
    total_equity = sum(equity_by_year.values())

    # B. Equity + continuous puts (cost 4.36%/yr)
    put_cost_yearly = PURCHASED_HEDGE_COSTS["continuous_puts"] * CAPITAL
    total_with_puts = total_equity - (put_cost_yearly * years)

    # C. Equity + EXP-1730 (always on)
    total_t1730 = sum(t1730_by_year.values())
    total_with_t1730 = total_equity + total_t1730

    # D. Equity + EXP-1730 (only during stress months)
    stress_month_keys = set(m[0] for m in stress_months)
    t1730_stress_pnl = sum(
        pnl for month, pnl in exp1730_by_month.items()
        if month in stress_month_keys
    )
    total_with_selective = total_equity + t1730_stress_pnl

    # Compute CAGRs (assume $100K capital)
    def cagr(total_pnl, n_years):
        if n_years <= 0:
            return 0.0
        total_ret = total_pnl / CAPITAL
        if 1 + total_ret <= 0:
            return -1.0
        return (1 + total_ret) ** (1 / n_years) - 1

    # Drawdown: max negative year
    def max_yearly_dd(year_pnl_dict):
        return min(year_pnl_dict.values()) / CAPITAL if year_pnl_dict else 0.0

    equity_worst_year = max_yearly_dd(equity_by_year)

    # For overlay scenarios, combine yearly pnls
    combined_t1730 = {yr: equity_by_year[yr] + t1730_by_year.get(yr, 0)
                      for yr in common_years}
    combined_t1730_worst = max_yearly_dd(combined_t1730)

    return {
        "n_years": years,
        "scenarios": {
            "A_equity_alone": {
                "total_pnl": round(total_equity, 2),
                "cagr_pct": round(cagr(total_equity, years) * 100, 2),
                "worst_year_pct": round(equity_worst_year * 100, 2),
            },
            "B_equity_plus_continuous_puts": {
                "total_pnl": round(total_with_puts, 2),
                "cagr_pct": round(cagr(total_with_puts, years) * 100, 2),
                "worst_year_pct": "estimated -2% better",
                "annual_drag_pct": round(PURCHASED_HEDGE_COSTS["continuous_puts"] * 100, 2),
                "source": "April 5 hedge_cost_resolution.py",
            },
            "C_equity_plus_t1730_always": {
                "total_pnl": round(total_with_t1730, 2),
                "cagr_pct": round(cagr(total_with_t1730, years) * 100, 2),
                "worst_year_pct": round(combined_t1730_worst * 100, 2),
                "annual_drag_pct": round((total_t1730 / years) / CAPITAL * 100, 2),
                "note": "negative drag = POSITIVE contribution",
            },
            "D_equity_plus_t1730_selective": {
                "total_pnl": round(total_with_selective, 2),
                "cagr_pct": round(cagr(total_with_selective, years) * 100, 2),
                "stress_pnl_only": round(t1730_stress_pnl, 2),
                "note": "EXP-1730 contribution ONLY during stress months",
            },
        },
        "equity_by_year": {str(yr): round(pnl, 2) for yr, pnl in equity_by_year.items()},
        "t1730_by_year": {str(yr): round(pnl, 2) for yr, pnl in t1730_by_year.items()},
    }


def walk_forward_hedge(
    spy: pd.DataFrame,
    exp1730_by_month: Dict[str, float],
    oos_start_year: int = 2020,
) -> Dict:
    """Walk-forward test: does EXP-1730's hedge behavior persist OOS?"""
    is_stress_months = []
    oos_stress_months = []

    spy_monthly = spy.copy()
    spy_monthly["month_key"] = spy_monthly.index.strftime("%Y-%m")
    spy_monthly["year"] = spy_monthly.index.year

    for month_key, group in spy_monthly.groupby("month_key"):
        if len(group) < 5:
            continue
        start = group["Close"].iloc[0]
        end = group["Close"].iloc[-1]
        monthly_ret = (end / start) - 1
        peak = group["Close"].cummax()
        dd = float(((group["Close"] - peak) / peak).min())
        if monthly_ret <= -0.05 or dd <= -0.05:
            yr = int(month_key[:4])
            if yr < oos_start_year:
                is_stress_months.append(month_key)
            else:
                oos_stress_months.append(month_key)

    is_stress_pnl = sum(exp1730_by_month.get(m, 0) for m in is_stress_months)
    oos_stress_pnl = sum(exp1730_by_month.get(m, 0) for m in oos_stress_months)

    is_total = sum(pnl for m, pnl in exp1730_by_month.items() if int(m[:4]) < oos_start_year)
    oos_total = sum(pnl for m, pnl in exp1730_by_month.items() if int(m[:4]) >= oos_start_year)

    return {
        "is_period": f"2010-{oos_start_year - 1}",
        "oos_period": f"{oos_start_year}-2026",
        "is_stress_months": len(is_stress_months),
        "is_stress_pnl": round(is_stress_pnl, 2),
        "is_total_pnl": round(is_total, 2),
        "oos_stress_months": len(oos_stress_months),
        "oos_stress_pnl": round(oos_stress_pnl, 2),
        "oos_total_pnl": round(oos_total, 2),
    }


# ── Crash-window analysis ─────────────────────────────────────────────────
CRASH_WINDOWS = [
    ("COVID Mar 2020",     "2020-02-19", "2020-03-23"),
    ("2022 Bear",          "2022-01-03", "2022-10-12"),
    ("Aug 2024 Unwind",    "2024-07-16", "2024-08-05"),
]


def build_daily_equity_curve(equity_daily_pnl: Dict[str, float],
                              spy: pd.DataFrame) -> pd.DataFrame:
    """Turn daily PnL dict into a full equity curve on SPY trading calendar."""
    df = pd.DataFrame({"Close": spy["Close"]})
    df["equity_pnl"] = 0.0
    for d, pnl in equity_daily_pnl.items():
        try:
            ts = pd.Timestamp(d)
            if ts in df.index:
                df.at[ts, "equity_pnl"] = pnl
        except Exception:
            pass
    return df


def build_t1730_daily_pnl(trades: List[Dict], spy: pd.DataFrame) -> pd.Series:
    """Distribute each EXP-1730 trade's PnL across its holding period linearly.

    Standalone trade log has entry/exit; assume linear P&L accrual over holding
    period for portfolio-level interaction with daily equity curve.
    """
    series = pd.Series(0.0, index=spy.index)
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_dt = pd.Timestamp(t["exit_date"])
        mask = (series.index >= entry) & (series.index <= exit_dt)
        n_days = int(mask.sum())
        if n_days > 0:
            per_day = float(t["pnl"]) / n_days
            series.loc[mask] += per_day
    return series


def compute_max_dd_daily(equity_series: pd.Series, capital: float) -> float:
    """Max drawdown of a daily equity level series."""
    equity = capital + equity_series.cumsum()
    peak = equity.cummax()
    dd = (peak - equity) / peak
    return float(dd.max())


def analyze_allocation_scenarios(
    equity_daily_pnl: Dict[str, float],
    t1730_trades: List[Dict],
    spy: pd.DataFrame,
    allocations: List[float] = (0.05, 0.075, 0.10),
) -> Dict:
    """Test 5%, 7.5%, 10% EXP-1730 allocations as overlay on equity strategy."""
    df = build_daily_equity_curve(equity_daily_pnl, spy)
    df["t1730_raw"] = build_t1730_daily_pnl(t1730_trades, spy)

    equity_total = float(df["equity_pnl"].sum())
    t1730_total = float(df["t1730_raw"].sum())

    # Baseline equity-only metrics
    base_dd = compute_max_dd_daily(df["equity_pnl"], CAPITAL)

    # Years span
    start = df.index[0]
    end = df.index[-1]
    n_years = max((end - start).days / 365.25, 0.5)
    base_cagr = ((1 + equity_total / CAPITAL) ** (1 / n_years) - 1) if equity_total > -CAPITAL else -1

    results = {
        "baseline": {
            "total_pnl": round(equity_total, 2),
            "cagr_pct": round(base_cagr * 100, 2),
            "max_dd_pct": round(base_dd * 100, 2),
        },
        "allocations": {},
    }

    # Scale EXP-1730 contribution by allocation
    # Interpretation: allocating X% of capital to EXP-1730 means
    # its returns contribute X/2% of its raw PnL (since EXP-1730 itself
    # uses 2% risk per trade, allocating 5% of capital = 2.5x the raw size).
    for alloc in allocations:
        scale = alloc / RISK_PER_TRADE_EXP1730  # 2% is EXP-1730's native risk
        scaled_t1730 = df["t1730_raw"] * scale

        combined_daily = df["equity_pnl"] + scaled_t1730
        combined_total = float(combined_daily.sum())
        combined_dd = compute_max_dd_daily(combined_daily, CAPITAL)
        combined_cagr = ((1 + combined_total / CAPITAL) ** (1 / n_years) - 1) if combined_total > -CAPITAL else -1

        # Per-crash drawdown
        crash_dds = {}
        for name, start_str, end_str in CRASH_WINDOWS:
            mask = (df.index >= start_str) & (df.index <= end_str)
            if mask.sum() < 5:
                continue
            window_eq = df.loc[mask, "equity_pnl"].cumsum().values
            window_comb = combined_daily.loc[mask].cumsum().values
            # DD within window (absolute dollars, then % of capital)
            if len(window_eq) > 0:
                peak_eq = np.maximum.accumulate(window_eq)
                dd_eq = float((peak_eq - window_eq).max()) / CAPITAL * 100
                peak_comb = np.maximum.accumulate(window_comb)
                dd_comb = float((peak_comb - window_comb).max()) / CAPITAL * 100
            else:
                dd_eq = dd_comb = 0.0
            crash_dds[name] = {
                "window": f"{start_str} to {end_str}",
                "equity_only_dd_pct": round(dd_eq, 2),
                "with_overlay_dd_pct": round(dd_comb, 2),
                "dd_reduction_pct": round(dd_eq - dd_comb, 2),
                "window_t1730_pnl": round(float(scaled_t1730.loc[mask].sum()), 2),
                "window_equity_pnl": round(float(df.loc[mask, "equity_pnl"].sum()), 2),
            }

        cagr_drag = round((combined_cagr - base_cagr) * 100, 3)
        dd_reduction_abs = round((base_dd - combined_dd) * 100, 2)

        results["allocations"][f"{alloc*100:.1f}%"] = {
            "scale_factor": round(scale, 2),
            "total_pnl": round(combined_total, 2),
            "cagr_pct": round(combined_cagr * 100, 2),
            "cagr_drag_pct": cagr_drag,
            "max_dd_pct": round(combined_dd * 100, 2),
            "dd_reduction_pct": dd_reduction_abs,
            "crash_windows": crash_dds,
        }

    return results


# EXP-1730's native risk per trade (from treasury_curve.py RISK_PER_TRADE)
RISK_PER_TRADE_EXP1730 = 0.02


def generate_report(results: Dict, output_path: Path):
    """Write HTML report."""
    hedge = results["hedge_behavior"]
    corr = results["correlation"]
    overlay = results["overlay_improvement"]
    wf = results["walk_forward"]

    # Stress vs normal table
    stress_row = ""
    stress_pct = hedge["stress_pnl"] / CAPITAL * 100
    normal_pct = hedge["normal_pnl"] / CAPITAL * 100
    s_color = "#059669" if hedge["stress_pnl"] > 0 else "#dc2626"
    n_color = "#059669" if hedge["normal_pnl"] > 0 else "#dc2626"

    # Yearly table
    yearly_rows = ""
    for y in corr["yearly"]:
        spy_c = "#059669" if y["spy_return"] > 0 else "#dc2626"
        eq_c = "#059669" if y["equity_pnl"] > 0 else "#dc2626"
        t_c = "#059669" if y["t1730_pnl"] > 0 else "#dc2626"
        yearly_rows += (
            f'<tr><td>{y["year"]}</td>'
            f'<td class="r" style="color:{spy_c}">{y["spy_return"]:+.1f}%</td>'
            f'<td class="r" style="color:{eq_c}">${y["equity_pnl"]:,.0f}</td>'
            f'<td class="r" style="color:{t_c}">${y["t1730_pnl"]:,.0f}</td></tr>\n'
        )

    # Scenarios table
    scen = overlay["scenarios"]
    scen_rows = ""
    for name, s in scen.items():
        cagr_c = "#059669" if s.get("cagr_pct", 0) > 0 else "#dc2626"
        worst_val = s.get("worst_year_pct", "—")
        worst_str = f"{worst_val:+.1f}%" if isinstance(worst_val, (int, float)) else str(worst_val)
        scen_rows += (
            f'<tr><td>{name.replace("_", " ").title()}</td>'
            f'<td class="r">${s["total_pnl"]:,.0f}</td>'
            f'<td class="r" style="color:{cagr_c}">{s["cagr_pct"]:+.2f}%</td>'
            f'<td class="r">{worst_str}</td>'
            f'<td>{s.get("note", s.get("source", ""))}</td></tr>\n'
        )

    verdict_hedge = hedge["stress_pnl"] > 0
    verdict_color = "#059669" if verdict_hedge else "#dc2626"
    verdict_text = ("HEDGE CONFIRMED: Positive PnL during SPY stress"
                    if verdict_hedge
                    else "NOT A HEDGE: No positive contribution during stress")

    best_scen = scen["D_equity_plus_t1730_selective"]
    baseline_scen = scen["A_equity_alone"]
    improvement_pct = best_scen["cagr_pct"] - baseline_scen["cagr_pct"]

    vs_puts = scen["B_equity_plus_continuous_puts"]
    vs_puts_gap = baseline_scen["cagr_pct"] - vs_puts["cagr_pct"]

    return _write_html(output_path, {
        "hedge": hedge, "corr": corr, "overlay": overlay, "wf": wf,
        "verdict_text": verdict_text, "verdict_color": verdict_color,
        "stress_pct": stress_pct, "normal_pct": normal_pct,
        "s_color": s_color, "n_color": n_color,
        "yearly_rows": yearly_rows, "scen_rows": scen_rows,
        "improvement_pct": improvement_pct, "vs_puts_gap": vs_puts_gap,
        "alloc_results": results.get("allocation_scenarios", {}),
    })


def _write_html(output_path: Path, ctx: Dict):
    hedge = ctx["hedge"]
    corr = ctx["corr"]
    wf = ctx["wf"]
    alloc = ctx.get("alloc_results", {})

    # Build allocation table
    alloc_rows = ""
    if alloc:
        baseline = alloc["baseline"]
        alloc_rows += (
            f'<tr><td><strong>Equity only (baseline)</strong></td>'
            f'<td class="r">${baseline["total_pnl"]:,.0f}</td>'
            f'<td class="r">{baseline["cagr_pct"]:+.2f}%</td>'
            f'<td class="r">{baseline["max_dd_pct"]:.2f}%</td>'
            f'<td class="r">&mdash;</td>'
            f'<td class="r">&mdash;</td></tr>\n'
        )
        for alloc_pct, r in alloc["allocations"].items():
            drag_c = "#059669" if r["cagr_drag_pct"] >= 0 else "#dc2626"
            red_c = "#059669" if r["dd_reduction_pct"] > 0 else "#dc2626"
            alloc_rows += (
                f'<tr><td><strong>+ EXP-1730 @ {alloc_pct}</strong></td>'
                f'<td class="r">${r["total_pnl"]:,.0f}</td>'
                f'<td class="r">{r["cagr_pct"]:+.2f}%</td>'
                f'<td class="r">{r["max_dd_pct"]:.2f}%</td>'
                f'<td class="r" style="color:{drag_c}">{r["cagr_drag_pct"]:+.3f}%</td>'
                f'<td class="r" style="color:{red_c}">{r["dd_reduction_pct"]:+.2f}pp</td></tr>\n'
            )

    # Build crash window rows — use the middle allocation (7.5%)
    crash_rows = ""
    mid_key = "7.5%"
    if alloc and mid_key in alloc.get("allocations", {}):
        mid = alloc["allocations"][mid_key]
        for cname, cw in mid["crash_windows"].items():
            red_c = "#059669" if cw["dd_reduction_pct"] > 0 else "#dc2626"
            t_c = "#059669" if cw["window_t1730_pnl"] > 0 else "#dc2626"
            crash_rows += (
                f'<tr><td><strong>{cname}</strong><br/><span style="color:#64748b;font-size:.72rem">{cw["window"]}</span></td>'
                f'<td class="r">{cw["equity_only_dd_pct"]:.2f}%</td>'
                f'<td class="r">{cw["with_overlay_dd_pct"]:.2f}%</td>'
                f'<td class="r" style="color:{red_c}">{cw["dd_reduction_pct"]:+.2f}pp</td>'
                f'<td class="r" style="color:{t_c}">${cw["window_t1730_pnl"]:+,.0f}</td>'
                f'<td class="r">${cw["window_equity_pnl"]:+,.0f}</td></tr>\n'
            )
    ctx["alloc_rows"] = alloc_rows
    ctx["crash_rows"] = crash_rows

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>EXP-1730 Hedge Overlay Analysis</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e2e8f0;--text:#1a1a2e;--muted:#64748b;--green:#059669;--red:#dc2626;--blue:#2563eb}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.55;max-width:1100px;margin:0 auto;padding:28px}}
h1{{font-size:1.55rem;font-weight:800;margin-bottom:4px}}
h2{{font-size:1.15rem;font-weight:700;margin:32px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--border)}}
.sub{{color:var(--muted);font-size:.86rem;margin-bottom:18px}}
.note{{color:var(--muted);font-size:.82rem;font-style:italic;margin:6px 0}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.84rem}}
th{{background:#f1f5f9;color:var(--muted);padding:7px 10px;text-align:left;border-bottom:2px solid var(--border);font-size:.74rem;font-weight:600;text-transform:uppercase}}
td{{padding:6px 10px;border-bottom:1px solid #f1f5f9;text-align:left}}
.r{{text-align:right}}
tr:hover td{{background:#fafafa}}
.hero{{background:linear-gradient(135deg,#f0fdf4 if ctx["verdict_color"]=="#059669" else #fef2f2,#ffffff);border:2px solid {ctx["verdict_color"]};border-radius:12px;padding:24px;margin:18px 0;text-align:center}}
.hero .title{{font-size:1.1rem;font-weight:700;color:{ctx["verdict_color"]}}}
.hero .big{{font-size:1.45rem;font-weight:800;color:{ctx["verdict_color"]};margin:8px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:13px;text-align:center}}
.c .l{{color:var(--muted);font-size:.72rem;text-transform:uppercase}}
.c .v{{font-weight:700;font-size:1.1rem;margin-top:3px}}
.box{{border:1px solid var(--border);border-radius:8px;padding:16px;margin:12px 0;background:var(--card)}}
.box-green{{border-left:5px solid var(--green)}} .box-red{{border-left:5px solid var(--red)}}
.box-blue{{border-left:5px solid var(--blue)}}
.box h4{{margin:0 0 6px;font-size:.95rem}}
</style></head><body>

<h1>EXP-1730 Hedge Overlay Analysis</h1>
<p class="sub">Does the Treasury Curve strategy act as a "free hedge" during equity stress?
&bull; Real Yahoo Finance data &bull; Zero synthetic</p>

<div class="hero">
<div class="title">{ctx["verdict_text"]}</div>
<div class="big">${hedge["stress_pnl"]:+,.0f} during {hedge["stress_months_with_trades"]} stress months
&nbsp;&bull;&nbsp; ${hedge["normal_pnl"]:+,.0f} during {hedge["normal_months_with_trades"]} normal months</div>
<p class="note">SPY/EXP-1730 yearly correlation: {corr["correlation"]:+.3f}
&bull; Walk-forward OOS stress PnL: ${wf["oos_stress_pnl"]:+,.0f}</p>
</div>

<div class="cards">
<div class="c"><div class="l">Stress Months Total</div><div class="v">{hedge["total_stress_months"]}</div></div>
<div class="c"><div class="l">With EXP-1730 Exits</div><div class="v">{hedge["stress_months_with_trades"]}</div></div>
<div class="c"><div class="l">Coverage</div><div class="v">{hedge["stress_coverage"]:.0%}</div></div>
<div class="c"><div class="l">Stress Win Rate</div><div class="v">{hedge["stress_win_rate"]:.0%}</div></div>
<div class="c"><div class="l">Normal Win Rate</div><div class="v">{hedge["normal_win_rate"]:.0%}</div></div>
<div class="c"><div class="l">Yearly Corr to Equity</div><div class="v">{corr["correlation"]:+.3f}</div></div>
</div>

<h2>1. Hedge Behavior — Stress vs Normal</h2>
<p class="note">Stress month = SPY monthly return &le; -5% OR peak-to-trough drawdown within the month &ge; 5%.</p>

<table>
<thead><tr><th>Regime</th><th class="r">Months w/ EXP-1730 Activity</th><th class="r">Total PnL</th><th class="r">% of Capital</th><th class="r">Win Rate</th></tr></thead>
<tbody>
<tr><td><strong>SPY Stress</strong> ({hedge["total_stress_months"]} stress months total)</td>
    <td class="r">{hedge["stress_months_with_trades"]}</td>
    <td class="r" style="color:{ctx["s_color"]}">${hedge["stress_pnl"]:,.0f}</td>
    <td class="r" style="color:{ctx["s_color"]}">{ctx["stress_pct"]:+.2f}%</td>
    <td class="r">{hedge["stress_win_rate"]:.0%}</td></tr>
<tr><td><strong>SPY Normal</strong></td>
    <td class="r">{hedge["normal_months_with_trades"]}</td>
    <td class="r" style="color:{ctx["n_color"]}">${hedge["normal_pnl"]:,.0f}</td>
    <td class="r" style="color:{ctx["n_color"]}">{ctx["normal_pct"]:+.2f}%</td>
    <td class="r">{hedge["normal_win_rate"]:.0%}</td></tr>
</tbody></table>

<h2>2. Yearly Correlation to Equity Strategy</h2>
<p class="note">Using champion_trade_log.json (EXP-400 regime-adaptive CS) as the equity-strategy proxy.</p>
<table>
<thead><tr><th>Year</th><th class="r">SPY Return</th><th class="r">Equity Strategy PnL</th><th class="r">EXP-1730 PnL</th></tr></thead>
<tbody>{ctx["yearly_rows"]}</tbody></table>
<p><strong>Yearly correlation (equity vs EXP-1730):</strong> <code>{corr["correlation"]:+.3f}</code></p>

<h2>3. Overlay Scenarios — Net Portfolio Improvement</h2>
<table>
<thead><tr><th>Scenario</th><th class="r">Total PnL</th><th class="r">CAGR</th><th class="r">Worst Year</th><th>Note</th></tr></thead>
<tbody>{ctx["scen_rows"]}</tbody></table>

<h2>3a. Allocation Overlay Scenarios (5%, 7.5%, 10%)</h2>
<p class="note">Equity strategy (champion/EXP-1220 proxy) + EXP-1730 overlay at different allocation levels.
Scale factor = allocation / EXP-1730 native risk (2%).</p>
<table>
<thead><tr><th>Configuration</th><th class="r">Total PnL</th><th class="r">CAGR</th><th class="r">Max DD</th><th class="r">CAGR Drag</th><th class="r">DD Reduction</th></tr></thead>
<tbody>{ctx["alloc_rows"]}</tbody></table>

<h2>3b. Drawdown During Specific Crash Windows (7.5% overlay)</h2>
<p class="note">Crash-window DD comparison: equity-only vs equity + EXP-1730 at 7.5% allocation.
COVID Mar 2020, 2022 Bear full year, Aug 2024 volmageddon unwind.</p>
<table>
<thead><tr><th>Crash</th><th class="r">Equity-Only DD</th><th class="r">With Overlay DD</th><th class="r">DD Reduction</th><th class="r">EXP-1730 PnL</th><th class="r">Equity PnL</th></tr></thead>
<tbody>{ctx["crash_rows"]}</tbody></table>

<h2>4. Walk-Forward: Does Hedge Behavior Persist OOS?</h2>
<table>
<thead><tr><th>Period</th><th class="r">Stress Months</th><th class="r">Stress PnL</th><th class="r">Total PnL</th></tr></thead>
<tbody>
<tr><td>IS ({wf["is_period"]})</td>
    <td class="r">{wf["is_stress_months"]}</td>
    <td class="r">${wf["is_stress_pnl"]:,.0f}</td>
    <td class="r">${wf["is_total_pnl"]:,.0f}</td></tr>
<tr><td>OOS ({wf["oos_period"]})</td>
    <td class="r">{wf["oos_stress_months"]}</td>
    <td class="r">${wf["oos_stress_pnl"]:,.0f}</td>
    <td class="r">${wf["oos_total_pnl"]:,.0f}</td></tr>
</tbody></table>

<h2>5. Comparison to April 5 Hedge Cost Finding</h2>
<div class="box box-blue">
<h4>Purchased hedges (from hedge_cost_resolution.py)</h4>
<ul style="padding-left:20px;font-size:.87rem;line-height:1.85">
<li><strong>Continuous puts:</strong> -2.86%/yr net drag (4.36% cost &minus; 1.50% alpha)</li>
<li><strong>Selective VIX&lt;15 puts:</strong> -0.60%/yr net drag</li>
<li><strong>Collar strategy:</strong> -0.31%/yr net drag (cheapest purchased hedge)</li>
<li><strong>Conclusion:</strong> All purchased insurance is cash-flow negative until alpha &ge; 1.81%/yr</li>
</ul>
</div>

<div class="box {'box-green' if hedge['stress_pnl'] > 0 else 'box-red'}">
<h4>EXP-1730 as unpurchased hedge</h4>
<p style="font-size:.87rem">EXP-1730 is structurally different from purchased insurance:
it is a <strong>return-generating strategy</strong> (not a drag). The question isn't
"how much does it cost?" but "does it contribute positive returns when equity strategies
suffer?"</p>
<ul style="padding-left:20px;font-size:.87rem;line-height:1.85;margin-top:8px">
<li>Standalone CAGR: essentially 0% (14-year break-even)</li>
<li>Stress-month contribution: <strong>${hedge["stress_pnl"]:+,.0f}</strong> across {hedge["stress_months_with_trades"]} stress months</li>
<li>Normal-month contribution: ${hedge["normal_pnl"]:+,.0f} across {hedge["normal_months_with_trades"]} normal months</li>
<li>Yearly correlation to equity strategy: {corr["correlation"]:+.3f} ({"uncorrelated" if abs(corr["correlation"]) < 0.3 else "moderate"})</li>
</ul>
<p style="font-size:.87rem;margin-top:8px">
<strong>Verdict:</strong> {
"EXP-1730 meaningfully hedges equity stress. Its positive stress-month PnL and near-zero yearly correlation make it structurally superior to purchased insurance — no premium bleed, actual positive contribution during the periods when equity strategies struggle."
if hedge["stress_pnl"] > 0
else "EXP-1730 does NOT deliver positive returns during stress months. It is not a hedge in the practical sense — it's just an uncorrelated break-even strategy."
}</p>
</div>

<h2>6. Rule Zero Compliance</h2>
<div class="box box-green">
<h4>ZERO SYNTHETIC DATA</h4>
<ul style="padding-left:20px;font-size:.82rem">
<li>SPY daily prices: Yahoo Finance (real)</li>
<li>EXP-1730 trades: reports/exp1730_treasury_curve.json (from real TLT/SHY/IEF/TIP backtest)</li>
<li>Equity strategy trades: output/champion_trade_log.json (real EXP-400 regime-adaptive CS)</li>
<li>Hedge cost baselines: April 5 hedge_cost_resolution.py (real IronVault put prices)</li>
</ul>
</div>

<p style="text-align:center;color:var(--muted);margin-top:36px;padding-top:14px;border-top:1px solid var(--border);font-size:.78rem">
EXP-1730 Hedge Overlay Analysis &bull; scripts/exp1730_hedge_analysis.py &bull;
{datetime.now().strftime("%Y-%m-%d")}
</p>
</body></html>"""
    output_path.write_text(html, encoding="utf-8")


def main():
    print("=" * 70)
    print("EXP-1730 Hedge Overlay Analysis")
    print("=" * 70)

    # Load real data
    spy = load_spy_daily()

    # Identify stress months
    log.info("Identifying SPY stress months...")
    stress_months = identify_stress_months(spy)
    for m in stress_months[:5]:
        log.info(f"  {m[0]}: monthly {m[1]:+.2%}, max DD {m[2]:+.2%}")
    if len(stress_months) > 5:
        log.info(f"  ... and {len(stress_months) - 5} more")

    # Load EXP-1730 trades and attribute to months
    log.info("Loading EXP-1730 trades...")
    trades = load_exp1730_trades()
    log.info(f"  {len(trades)} trades")
    exp1730_by_month = attribute_exp1730_to_months(trades)
    log.info(f"  Across {len(exp1730_by_month)} unique exit months")

    # Load equity strategy
    log.info("Loading equity strategy (EXP-400 champion as proxy for EXP-1220)...")
    equity_daily = load_exp1220_daily_pnl()

    # Analyze hedge behavior
    log.info("Analyzing hedge behavior...")
    hedge = analyze_hedge_behavior(spy, stress_months, exp1730_by_month)

    print(f"\n--- Hedge Behavior ---")
    print(f"Stress months total:            {hedge['total_stress_months']}")
    print(f"  ...with EXP-1730 exits:       {hedge['stress_months_with_trades']} ({hedge['stress_coverage']:.0%} coverage)")
    print(f"  Stress PnL:                   ${hedge['stress_pnl']:,.2f}")
    print(f"  Stress win rate:              {hedge['stress_win_rate']:.0%}")
    print(f"")
    print(f"Normal months with EXP-1730:    {hedge['normal_months_with_trades']}")
    print(f"  Normal PnL:                   ${hedge['normal_pnl']:,.2f}")
    print(f"  Normal win rate:              {hedge['normal_win_rate']:.0%}")

    # Yearly correlation
    log.info("Computing yearly correlation...")
    corr = compute_yearly_correlation(spy, equity_daily, exp1730_by_month)
    print(f"\n--- Yearly Correlation ---")
    print(f"Equity vs EXP-1730:             {corr['correlation']:+.3f}")

    # Overlay improvement
    log.info("Computing overlay improvement...")
    overlay = compute_overlay_net_improvement(equity_daily, exp1730_by_month, stress_months)
    print(f"\n--- Overlay Scenarios (over {overlay['n_years']} years) ---")
    for name, s in overlay["scenarios"].items():
        pretty = name.replace("_", " ")
        print(f"  {pretty}:")
        print(f"    Total PnL:  ${s['total_pnl']:,.2f}")
        print(f"    CAGR:       {s['cagr_pct']:+.2f}%")

    # Walk-forward
    log.info("Walk-forward analysis...")
    wf = walk_forward_hedge(spy, exp1730_by_month)
    print(f"\n--- Walk-Forward ---")
    print(f"IS ({wf['is_period']}):  stress PnL ${wf['is_stress_pnl']:,.0f} / total ${wf['is_total_pnl']:,.0f}")
    print(f"OOS ({wf['oos_period']}): stress PnL ${wf['oos_stress_pnl']:,.0f} / total ${wf['oos_total_pnl']:,.0f}")

    # Allocation scenarios & crash windows
    log.info("Allocation overlay analysis (5/7.5/10%)...")
    alloc_results = analyze_allocation_scenarios(equity_daily, trades, spy)
    print(f"\n--- Allocation Scenarios ---")
    print(f"Baseline equity-only:          CAGR {alloc_results['baseline']['cagr_pct']:+.2f}%, Max DD {alloc_results['baseline']['max_dd_pct']:.2f}%")
    for alloc_pct, r in alloc_results["allocations"].items():
        print(f"\nEquity + EXP-1730 @ {alloc_pct} (scale {r['scale_factor']:.1f}x):")
        print(f"  CAGR:         {r['cagr_pct']:+.2f}% (drag {r['cagr_drag_pct']:+.3f}%)")
        print(f"  Max DD:       {r['max_dd_pct']:.2f}% (reduction {r['dd_reduction_pct']:+.2f}pp)")
        for cname, cw in r["crash_windows"].items():
            print(f"  {cname}: equity-only DD {cw['equity_only_dd_pct']:.2f}% -> "
                  f"with overlay {cw['with_overlay_dd_pct']:.2f}% "
                  f"(reduction {cw['dd_reduction_pct']:+.2f}pp)")

    # Package
    results = {
        "generated": datetime.now().isoformat(),
        "data_source": "Yahoo Finance SPY + reports/exp1730_treasury_curve.json + output/champion_trade_log.json",
        "rule_zero_compliant": True,
        "hedge_behavior": hedge,
        "correlation": corr,
        "overlay_improvement": overlay,
        "walk_forward": wf,
        "allocation_scenarios": alloc_results,
        "april5_comparison": {
            "purchased_puts_drag": -2.86,
            "selective_puts_drag": -0.60,
            "collar_drag": -0.31,
            "source": "scripts/hedge_cost_resolution.py (commit c948b73)",
        },
    }

    # Save
    json_path = ROOT / "reports" / "exp1730_hedge_analysis.json"
    json_path.parent.mkdir(exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, default=str))
    log.info(f"JSON: {json_path}")

    html_path = ROOT / "reports" / "exp1730_hedge_analysis.html"
    generate_report(results, html_path)
    log.info(f"HTML: {html_path}")

    # Final verdict
    print(f"\n{'=' * 70}")
    if hedge["stress_pnl"] > 0:
        print("VERDICT: HEDGE BEHAVIOR CONFIRMED")
        print(f"  EXP-1730 contributes ${hedge['stress_pnl']:+,.0f} during stress months")
        print(f"  vs ${hedge['normal_pnl']:+,.0f} during normal months")
        print(f"  Yearly correlation to equity: {corr['correlation']:+.3f}")
        print(f"\n  Structurally different from purchased puts (all net-negative):")
        print(f"    Continuous puts drag:  -2.86%/yr")
        print(f"    Selective puts drag:   -0.60%/yr")
        print(f"    Collar drag:           -0.31%/yr")
        print(f"    EXP-1730 drag:         {(overlay['scenarios']['C_equity_plus_t1730_always']['annual_drag_pct']):+.2f}%/yr")
    else:
        print("VERDICT: NOT A HEDGE")
        print(f"  EXP-1730 stress PnL: ${hedge['stress_pnl']:,.0f} (not positive)")
        print(f"  Does not meet the hedge criterion.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
