#!/usr/bin/env python3
"""EXP-730-max: Short DTE Theta Decay Backtest

Simulates 0-7 DTE credit spreads on SPY, 2020-2025.
Generates realistic trades with theta decay modeling, event filtering,
and position sizing constraints.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ───────────────────────────────────────────────────────────────
STARTING_CAPITAL = 100_000.0
START_DATE = date(2020, 1, 2)
END_DATE = date(2025, 12, 31)

# Spread parameters
SPREAD_WIDTH_MIN = 1.0
SPREAD_WIDTH_MAX = 2.0
CREDIT_RATIO_MIN = 0.15   # 15% of spread width
CREDIT_RATIO_MAX = 0.30   # 30%
OTM_PCT_MIN = 5.0         # 5% OTM
OTM_PCT_MAX = 10.0        # 10% OTM
DTE_MIN = 0
DTE_MAX = 7

# Trade frequency
TRADES_PER_WEEK_MIN = 3
TRADES_PER_WEEK_MAX = 5

# Position sizing
MAX_POSITION_PCT = 0.02   # 2% of portfolio per trade
MAX_CONTRACTS = 20

# Exit rules
PROFIT_TARGET = 0.50      # 50% of max profit
STOP_LOSS_MULT = 2.0      # 2x credit received
TIME_STOP_DTE = 0         # close by expiration day

# Slippage
SLIPPAGE_PER_LEG_MIN = 0.03
SLIPPAGE_PER_LEG_MAX = 0.05

# Filter thresholds
IV_RANK_THRESHOLD = 20.0  # IV rank > 20th percentile (short DTE needs less IV edge)

# FOMC dates 2020-2025 (announcement days to avoid)
FOMC_DATES = {
    date(2020, 1, 29), date(2020, 3, 3), date(2020, 3, 15), date(2020, 3, 23),
    date(2020, 4, 29), date(2020, 6, 10), date(2020, 7, 29), date(2020, 9, 16),
    date(2020, 11, 5), date(2020, 12, 16),
    date(2021, 1, 27), date(2021, 3, 17), date(2021, 4, 28), date(2021, 6, 16),
    date(2021, 7, 28), date(2021, 9, 22), date(2021, 11, 3), date(2021, 12, 15),
    date(2022, 1, 26), date(2022, 3, 16), date(2022, 5, 4), date(2022, 6, 15),
    date(2022, 7, 27), date(2022, 9, 21), date(2022, 11, 2), date(2022, 12, 14),
    date(2023, 2, 1), date(2023, 3, 22), date(2023, 5, 3), date(2023, 6, 14),
    date(2023, 7, 26), date(2023, 9, 20), date(2023, 11, 1), date(2023, 12, 13),
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 17),
}

# CPI release dates (approx 2nd Wednesday of each month)
def _cpi_dates(start_year: int, end_year: int) -> set:
    dates = set()
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            d = date(y, m, 1)
            # Find 2nd Wednesday
            wed_count = 0
            for day in range(1, 22):
                candidate = date(y, m, day)
                if candidate.weekday() == 2:  # Wednesday
                    wed_count += 1
                    if wed_count == 2:
                        dates.add(candidate)
                        break
    return dates

# NFP: first Friday of each month
def _nfp_dates(start_year: int, end_year: int) -> set:
    dates = set()
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            for day in range(1, 8):
                candidate = date(y, m, day)
                if candidate.weekday() == 4:  # Friday
                    dates.add(candidate)
                    break
    return dates

EVENT_DATES = FOMC_DATES | _cpi_dates(2020, 2025) | _nfp_dates(2020, 2025)

# Also block day before and day after major events (FOMC only)
BLOCKED_DATES = set()
for d in FOMC_DATES:
    BLOCKED_DATES.add(d)
    BLOCKED_DATES.add(d - timedelta(days=1))
    BLOCKED_DATES.add(d + timedelta(days=1))
for d in _cpi_dates(2020, 2025) | _nfp_dates(2020, 2025):
    BLOCKED_DATES.add(d)


# ── Synthetic SPY price + VIX data ──────────────────────────────────────────
def generate_market_data(
    start: date, end: date, seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic SPY + VIX daily data using calibrated random walk."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # SPY: start at ~320 (Jan 2020), trending up with vol clusters
    spy_log_ret = rng.randn(n) * 0.012 + 0.0004  # ~10% annual return, ~19% vol

    # Add vol clustering (GARCH-like)
    vol = np.ones(n) * 0.012
    for i in range(1, n):
        vol[i] = 0.9 * vol[i - 1] + 0.1 * abs(spy_log_ret[i - 1]) + 0.002
        spy_log_ret[i] = rng.randn() * vol[i] + 0.0004

    # Inject known historical events
    spy_price = np.zeros(n)
    spy_price[0] = 320.0
    for i in range(1, n):
        spy_price[i] = spy_price[i - 1] * np.exp(spy_log_ret[i])

    # COVID crash: ~Feb 20 - Mar 23, 2020 (approx indices 35-55)
    covid_start = 35
    covid_end = 55
    if n > covid_end:
        for i in range(covid_start, covid_end):
            spy_log_ret[i] = -0.03 + rng.randn() * 0.04
        # Recovery
        for i in range(covid_end, min(covid_end + 60, n)):
            spy_log_ret[i] = 0.015 + rng.randn() * 0.02

    # 2022 bear market: roughly index 500-700
    bear_start = min(500, n - 1)
    bear_end = min(700, n)
    for i in range(bear_start, bear_end):
        spy_log_ret[i] = -0.003 + rng.randn() * 0.015

    # Rebuild prices with events
    spy_price[0] = 320.0
    for i in range(1, n):
        spy_price[i] = spy_price[i - 1] * np.exp(spy_log_ret[i])

    # VIX: inversely correlated with SPY, mean-reverting
    vix = np.zeros(n)
    vix[0] = 14.0
    for i in range(1, n):
        mean_rev = 0.03 * (18.0 - vix[i - 1])
        shock = -spy_log_ret[i] * 200 + rng.randn() * 1.5
        vix[i] = max(9.0, min(80.0, vix[i - 1] + mean_rev + shock))

    # IV rank: rolling percentile of VIX
    vix_series = pd.Series(vix)
    iv_rank = vix_series.rolling(60, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.min()) / max(x.max() - x.min(), 0.01) * 100
    ).fillna(50).values

    # Regime classification
    regimes = []
    for i in range(n):
        if vix[i] > 35:
            regimes.append("crash")
        elif vix[i] > 25:
            regimes.append("high_vol")
        elif vix[i] < 14 and spy_log_ret[max(0, i - 20):i + 1].mean() > 0:
            regimes.append("low_vol")
        elif spy_log_ret[max(0, i - 20):i + 1].mean() < -0.002:
            regimes.append("bear")
        else:
            regimes.append("bull")

    return pd.DataFrame({
        "date": dates[:n],
        "spy_price": spy_price,
        "spy_return": spy_log_ret,
        "vix": vix,
        "iv_rank": iv_rank,
        "realized_vol": pd.Series(spy_log_ret).rolling(20).std().fillna(0.012).values * np.sqrt(252),
        "regime": regimes,
    }).set_index("date")


# ── Black-Scholes theta decay model ────────────────────────────────────────
def bs_theta_factor(dte: int, total_dte: int) -> float:
    """Fraction of total premium captured from total_dte to dte remaining.

    Theta decay follows ~1/sqrt(T), so premium decays faster near expiry.
    """
    if total_dte <= 0:
        return 1.0
    t_start = total_dte / 365.0
    t_end = max(dte, 0) / 365.0
    if t_start <= 0:
        return 1.0
    return 1.0 - math.sqrt(t_end / t_start)


def simulate_trade_pnl(
    credit: float,
    spread_width: float,
    entry_dte: int,
    spy_return_during: float,
    otm_pct: float,
    rng: np.random.RandomState,
) -> Tuple[float, int, str]:
    """Simulate a single trade's P&L using theta decay + delta approximation.

    Returns (pnl_per_contract, hold_days, exit_reason).
    """
    max_loss = (spread_width - credit) * 100  # per contract
    max_profit = credit * 100

    # Day-by-day simulation
    hold_days = 0
    cumulative_move = 0.0
    daily_vol = 0.012

    for day in range(entry_dte + 1):
        hold_days += 1
        dte_remaining = entry_dte - day

        # Daily SPY move (partition total move + noise)
        daily_move = spy_return_during / max(entry_dte, 1) + rng.randn() * daily_vol * 0.3
        cumulative_move += daily_move

        # Theta captured so far
        theta_pnl = credit * 100 * bs_theta_factor(dte_remaining, entry_dte)

        # Delta P&L: approximate delta exposure
        # Short put spread: negative delta (hurts on down moves)
        # Short call spread: positive delta (hurts on up moves)
        # Use OTM distance as proxy for delta magnitude
        delta_magnitude = max(0.1, 0.5 - otm_pct / 20)  # ~0.15-0.40
        delta_pnl = -abs(cumulative_move) / (otm_pct / 100) * delta_magnitude * max_profit

        current_pnl = theta_pnl + delta_pnl

        # Check profit target
        if current_pnl >= max_profit * PROFIT_TARGET:
            return current_pnl, hold_days, "profit_target"

        # Check stop loss
        if current_pnl <= -max_loss * STOP_LOSS_MULT * credit / spread_width:
            return -max_loss * STOP_LOSS_MULT * credit / spread_width, hold_days, "stop_loss"

        # Check if underlying breaches short strike
        if abs(cumulative_move) > otm_pct / 100 * 0.9:
            loss = min(current_pnl, -max_profit * 0.5)
            return loss, hold_days, "strike_breach"

    # Expiration: collect remaining theta
    final_pnl = credit * 100 * bs_theta_factor(0, entry_dte) + delta_pnl
    final_pnl = max(-max_loss, min(max_profit, final_pnl))
    return final_pnl, hold_days, "expiration"


# ── Trade dataclass ─────────────────────────────────────────────────────────
@dataclass
class Trade:
    entry_date: str
    exit_date: str
    dte: int
    hold_days: int
    spread_type: str        # bull_put or bear_call
    spread_width: float
    credit: float
    contracts: int
    spy_price: float
    vix: float
    iv_rank: float
    regime: str
    short_strike: float
    otm_pct: float
    slippage: float
    exit_reason: str
    pnl: float              # after slippage
    return_pct: float
    win: bool


# ── Main backtest ───────────────────────────────────────────────────────────
def run_backtest(seed: int = 42) -> Dict[str, Any]:
    """Run the full EXP-730-max backtest."""
    rng = np.random.RandomState(seed)
    market = generate_market_data(START_DATE, END_DATE, seed=seed)

    capital = STARTING_CAPITAL
    equity_curve = [(str(START_DATE), capital)]
    trades: List[Trade] = []
    daily_pnl: List[Tuple[str, float]] = []

    peak = capital
    max_dd = 0.0
    weekly_trade_count = 0
    current_week = None

    trading_days = [d.date() for d in market.index]

    for i, today in enumerate(trading_days):
        if today < START_DATE or today > END_DATE:
            continue

        # Reset weekly counter
        week_num = today.isocalendar()[1]
        if current_week != week_num:
            current_week = week_num
            weekly_trade_count = 0

        # Skip weekends (shouldn't happen with bdate_range but safety)
        if today.weekday() >= 5:
            continue

        # Market data for today
        row = market.iloc[i]
        spy_price = row["spy_price"]
        vix = row["vix"]
        iv_rank = row["iv_rank"]
        regime = row["regime"]

        # ── Entry filters ──────────────────────────────────────────────
        # 1. Weekly trade limit
        if weekly_trade_count >= TRADES_PER_WEEK_MAX:
            continue

        # 2. IV rank filter
        if iv_rank < IV_RANK_THRESHOLD:
            continue

        # 3. Event filter
        if today in BLOCKED_DATES:
            continue

        # 4. Regime filter: skip crash, reduce in high_vol
        if regime == "crash":
            continue
        if regime == "high_vol" and rng.rand() > 0.3:
            continue

        # 5. Random entry (3-5 trades per week — high probability on eligible days)
        entry_prob = 0.85  # most eligible days → trade
        if rng.rand() > entry_prob:
            continue

        # ── Trade construction ─────────────────────────────────────────
        # Choose DTE
        dte = int(rng.randint(DTE_MIN, DTE_MAX + 1))
        if dte == 0:
            dte = 1  # minimum 1 day for simulation

        # Spread type: alternate bull_put (70%) and bear_call (30%)
        spread_type = "bull_put" if rng.rand() < 0.70 else "bear_call"

        # Spread width
        spread_width = round(rng.uniform(SPREAD_WIDTH_MIN, SPREAD_WIDTH_MAX), 0)
        if spread_width < 1:
            spread_width = 1.0

        # Credit
        credit_ratio = rng.uniform(CREDIT_RATIO_MIN, CREDIT_RATIO_MAX)
        credit = round(spread_width * credit_ratio, 2)

        # OTM distance
        otm_pct = rng.uniform(OTM_PCT_MIN, OTM_PCT_MAX)

        # Short strike
        if spread_type == "bull_put":
            short_strike = round(spy_price * (1 - otm_pct / 100), 0)
        else:
            short_strike = round(spy_price * (1 + otm_pct / 100), 0)

        # Position sizing: max 2% of capital risk per trade
        max_loss_per_contract = (spread_width - credit) * 100
        if max_loss_per_contract <= 0:
            continue
        max_contracts_by_risk = max(1, int(capital * MAX_POSITION_PCT / max_loss_per_contract))
        contracts = min(max_contracts_by_risk, MAX_CONTRACTS)
        contracts = max(1, contracts)

        # Slippage (per leg, 2 legs per spread)
        slippage_per_leg = rng.uniform(SLIPPAGE_PER_LEG_MIN, SLIPPAGE_PER_LEG_MAX)
        total_slippage = slippage_per_leg * 2 * contracts * 100  # in dollars

        # ── Simulate trade ─────────────────────────────────────────────
        # Get SPY return over the trade duration
        end_idx = min(i + dte, len(trading_days) - 1)
        spy_return = (market.iloc[end_idx]["spy_price"] / spy_price) - 1.0

        pnl_per_contract, hold_days, exit_reason = simulate_trade_pnl(
            credit, spread_width, dte, spy_return, otm_pct, rng,
        )

        total_pnl = pnl_per_contract * contracts - total_slippage
        return_pct = total_pnl / (max_loss_per_contract * contracts) * 100

        exit_date = trading_days[min(i + hold_days, len(trading_days) - 1)]

        trade = Trade(
            entry_date=str(today),
            exit_date=str(exit_date),
            dte=dte,
            hold_days=hold_days,
            spread_type=spread_type,
            spread_width=spread_width,
            credit=credit,
            contracts=contracts,
            spy_price=round(spy_price, 2),
            vix=round(vix, 2),
            iv_rank=round(iv_rank, 2),
            regime=regime,
            short_strike=short_strike,
            otm_pct=round(otm_pct, 2),
            slippage=round(total_slippage, 2),
            exit_reason=exit_reason,
            pnl=round(total_pnl, 2),
            return_pct=round(return_pct, 2),
            win=total_pnl > 0,
        )
        trades.append(trade)

        capital += total_pnl
        weekly_trade_count += 1
        daily_pnl.append((str(today), total_pnl))
        equity_curve.append((str(today), round(capital, 2)))

        # Track drawdown
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak
        if dd > max_dd:
            max_dd = dd

    # ── Compute summary metrics ────────────────────────────────────────
    total_trades = len(trades)
    wins = sum(1 for t in trades if t.win)
    losses = total_trades - wins
    win_rate = wins / total_trades if total_trades > 0 else 0

    total_pnl = sum(t.pnl for t in trades)
    total_return = (capital - STARTING_CAPITAL) / STARTING_CAPITAL

    # Annualized return
    years = (END_DATE - START_DATE).days / 365.25
    cagr = (capital / STARTING_CAPITAL) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe from daily P&L
    pnl_values = [p for _, p in daily_pnl]
    if len(pnl_values) > 1:
        sharpe = np.mean(pnl_values) / np.std(pnl_values) * np.sqrt(252) if np.std(pnl_values) > 0 else 0
    else:
        sharpe = 0

    avg_trade = total_pnl / total_trades if total_trades > 0 else 0
    avg_win = np.mean([t.pnl for t in trades if t.win]) if wins > 0 else 0
    avg_loss = np.mean([t.pnl for t in trades if not t.win]) if losses > 0 else 0
    profit_factor = abs(sum(t.pnl for t in trades if t.win) / sum(t.pnl for t in trades if not t.win)) if losses > 0 and sum(t.pnl for t in trades if not t.win) != 0 else 0
    avg_hold = np.mean([t.hold_days for t in trades]) if trades else 0
    avg_dte = np.mean([t.dte for t in trades]) if trades else 0

    # Per-year breakdown
    yearly = {}
    for t in trades:
        y = t.entry_date[:4]
        if y not in yearly:
            yearly[y] = {"trades": 0, "pnl": 0, "wins": 0}
        yearly[y]["trades"] += 1
        yearly[y]["pnl"] += t.pnl
        yearly[y]["wins"] += int(t.win)

    yearly_summary = {}
    for y, d in yearly.items():
        yearly_summary[y] = {
            "trades": d["trades"],
            "pnl": round(d["pnl"], 2),
            "win_rate": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] > 0 else 0,
            "return_pct": round(d["pnl"] / STARTING_CAPITAL * 100, 1),
        }

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Regime breakdown
    regime_stats = {}
    for t in trades:
        r = t.regime
        if r not in regime_stats:
            regime_stats[r] = {"trades": 0, "pnl": 0, "wins": 0}
        regime_stats[r]["trades"] += 1
        regime_stats[r]["pnl"] += t.pnl
        regime_stats[r]["wins"] += int(t.win)

    summary = {
        "experiment": "EXP-730-max",
        "name": "Short DTE Theta Decay",
        "period": f"{START_DATE} to {END_DATE}",
        "starting_capital": STARTING_CAPITAL,
        "ending_capital": round(capital, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar_ratio": round(cagr / max_dd, 2) if max_dd > 0 else 0,
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_trade_pnl": round(avg_trade, 2),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "avg_hold_days": round(float(avg_hold), 1),
        "avg_dte": round(float(avg_dte), 1),
        "total_slippage": round(sum(t.slippage for t in trades), 2),
        "exit_reasons": exit_reasons,
        "yearly": yearly_summary,
        "regime_stats": {k: {"trades": v["trades"], "pnl": round(v["pnl"], 2),
                             "win_rate": round(v["wins"] / v["trades"] * 100, 1) if v["trades"] > 0 else 0}
                         for k, v in regime_stats.items()},
        "success_criteria": {
            "annual_returns_gt_40": bool(cagr * 100 > 40),
            "max_dd_lt_15": bool(max_dd * 100 < 15),
            "win_rate_gt_70": bool(win_rate * 100 > 70),
            "avg_hold_lt_5": bool(float(avg_hold) < 5),
            "sharpe_gt_2_5": bool(sharpe > 2.5),
        },
    }

    return {
        "summary": summary,
        "trades": [asdict(t) for t in trades],
        "equity_curve": equity_curve,
    }


# ── HTML report generator ──────────────────────────────────────────────────
def generate_html_report(results: Dict[str, Any], output_path: str) -> None:
    s = results["summary"]
    eq = results["equity_curve"]
    trades = results["trades"]

    # SVG equity curve
    n = len(eq)
    w, h = 700, 250
    pl, pb, pt = 70, 35, 20
    cw, ch = w - pl, h - pb - pt
    values = [v for _, v in eq]
    mn, mx = min(values), max(values)
    rng_v = mx - mn or 1
    pts = []
    for i, (_, v) in enumerate(eq):
        x = pl + i / max(n - 1, 1) * cw
        y = pt + ch - (v - mn) / rng_v * ch
        pts.append(f"{x:.0f},{y:.0f}")
    eq_svg = (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{" ".join(pts)}" fill="none" stroke="#4ade80" stroke-width="2"/>'
        f'<text x="{pl-5}" y="{pt+4}" text-anchor="end" font-size="10" fill="#94a3b8">${mx:,.0f}</text>'
        f'<text x="{pl-5}" y="{pt+ch}" text-anchor="end" font-size="10" fill="#94a3b8">${mn:,.0f}</text>'
        f'<line x1="{pl}" y1="{pt+ch}" x2="{w}" y2="{pt+ch}" stroke="#475569" stroke-width="1"/>'
        f'</svg>'
    )

    # Yearly table
    yearly_rows = ""
    for y, d in sorted(s["yearly"].items()):
        cls = "pos" if d["pnl"] > 0 else "neg"
        yearly_rows += f'<tr><td>{y}</td><td>{d["trades"]}</td><td class="{cls}">${d["pnl"]:,.0f}</td><td>{d["win_rate"]}%</td><td>{d["return_pct"]}%</td></tr>'

    # Exit reasons
    exit_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(s["exit_reasons"].items()))

    # Regime stats
    regime_rows = ""
    for r, d in sorted(s["regime_stats"].items()):
        cls = "pos" if d["pnl"] > 0 else "neg"
        regime_rows += f'<tr><td>{r}</td><td>{d["trades"]}</td><td class="{cls}">${d["pnl"]:,.0f}</td><td>{d["win_rate"]}%</td></tr>'

    # Success criteria
    criteria_rows = ""
    for k, v in s["success_criteria"].items():
        cls = "pos" if v else "neg"
        criteria_rows += f'<tr><td>{k.replace("_"," ")}</td><td class="{cls}">{"PASS" if v else "FAIL"}</td></tr>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>EXP-730-max: Short DTE Theta Decay</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,sans-serif;background:#0f172a;color:#e2e8f0;padding:24px}}
h1{{font-size:1.8rem;margin-bottom:4px}}
h2{{font-size:1.1rem;color:#38bdf8;margin:20px 0 10px}}
.sub{{color:#94a3b8;font-size:.85rem;margin-bottom:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:28px}}
.card{{background:#1e293b;border-radius:10px;padding:18px}}
.card .lbl{{font-size:.75rem;color:#94a3b8;text-transform:uppercase}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:.85rem;margin-bottom:20px}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #334155}}
th{{color:#94a3b8}}
.pos{{color:#4ade80}}.neg{{color:#f87171}}
svg{{display:block;margin:0 auto 20px}}
</style>
</head>
<body>
<h1>EXP-730-max: Short DTE Theta Decay</h1>
<p class="sub">{s["period"]} &middot; {s["total_trades"]} trades &middot; Starting: ${s["starting_capital"]:,.0f}</p>

<div class="grid">
<div class="card"><div class="lbl">Ending Capital</div><div class="val {'pos' if s['total_return_pct']>0 else 'neg'}">${s["ending_capital"]:,.0f}</div></div>
<div class="card"><div class="lbl">Total Return</div><div class="val {'pos' if s['total_return_pct']>0 else 'neg'}">{s["total_return_pct"]}%</div></div>
<div class="card"><div class="lbl">CAGR</div><div class="val">{s["cagr_pct"]}%</div></div>
<div class="card"><div class="lbl">Sharpe</div><div class="val">{s["sharpe_ratio"]}</div></div>
<div class="card"><div class="lbl">Max DD</div><div class="val neg">{s["max_drawdown_pct"]}%</div></div>
<div class="card"><div class="lbl">Win Rate</div><div class="val">{s["win_rate_pct"]}%</div></div>
<div class="card"><div class="lbl">Profit Factor</div><div class="val">{s["profit_factor"]}</div></div>
<div class="card"><div class="lbl">Avg Hold</div><div class="val">{s["avg_hold_days"]}d</div></div>
<div class="card"><div class="lbl">Avg DTE</div><div class="val">{s["avg_dte"]}d</div></div>
<div class="card"><div class="lbl">Calmar</div><div class="val">{s["calmar_ratio"]}</div></div>
</div>

<h2>Equity Curve</h2>
{eq_svg}

<h2>Success Criteria</h2>
<table><thead><tr><th>Criterion</th><th>Status</th></tr></thead><tbody>{criteria_rows}</tbody></table>

<h2>Yearly Breakdown</h2>
<table><thead><tr><th>Year</th><th>Trades</th><th>P&L</th><th>Win Rate</th><th>Return</th></tr></thead><tbody>{yearly_rows}</tbody></table>

<h2>Exit Reasons</h2>
<table><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody>{exit_rows}</tbody></table>

<h2>Regime Performance</h2>
<table><thead><tr><th>Regime</th><th>Trades</th><th>P&L</th><th>Win Rate</th></tr></thead><tbody>{regime_rows}</tbody></table>

</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(html)


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running EXP-730-max backtest...")
    results = run_backtest()
    s = results["summary"]

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    with open(results_dir / "summary.json", "w") as f:
        json.dump(results["summary"], f, indent=2)
    print(f"Summary written to results/summary.json")

    generate_html_report(results, str(results_dir / "report.html"))
    print(f"Report written to results/report.html")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EXP-730-max: Short DTE Theta Decay — Results")
    print(f"{'='*60}")
    print(f"Period:          {s['period']}")
    print(f"Starting:        ${s['starting_capital']:,.0f}")
    print(f"Ending:          ${s['ending_capital']:,.0f}")
    print(f"Total Return:    {s['total_return_pct']}%")
    print(f"CAGR:            {s['cagr_pct']}%")
    print(f"Sharpe:          {s['sharpe_ratio']}")
    print(f"Max Drawdown:    {s['max_drawdown_pct']}%")
    print(f"Win Rate:        {s['win_rate_pct']}%")
    print(f"Total Trades:    {s['total_trades']}")
    print(f"Avg Hold Days:   {s['avg_hold_days']}")
    print(f"Profit Factor:   {s['profit_factor']}")
    print(f"Total Slippage:  ${s['total_slippage']:,.0f}")
    print(f"\nSuccess Criteria:")
    for k, v in s["success_criteria"].items():
        status = "✓" if v else "✗"
        print(f"  {status} {k.replace('_', ' ')}")
