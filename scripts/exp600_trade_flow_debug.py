#!/usr/bin/env python3
"""
EXP-600 Trade Flow Debug: Why 2 trades in 2021/2023 but 68 in 2022 and 58 in 2025?

For the winning config (DTE=45, W=$10, OTM=5%, PT=50%, SL=2.5x, risk=2%):
- 2020: 49 trades
- 2021: 2 trades   <-- WHY?
- 2022: 68 trades
- 2023: 2 trades   <-- WHY?
- 2024: 8 trades
- 2025: 58 trades

This script probes the DB and backtester logic to find the bottleneck.
"""

import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

DB_PATH = ROOT / "data" / "options_cache.db"
OUTPUT_PATH = ROOT / "results" / "exp600" / "trade_flow_debug.md"

TICKER = "SPY"
TARGET_DTE_MIN = 40
TARGET_DTE_MAX = 50
OTM_PCT = 0.05
SPREAD_WIDTH = 10
OPTION_TYPE = "P"  # Bull put spreads

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
SAMPLES_PER_YEAR = 15  # More samples for statistical power


def get_trading_days(year):
    """Get all weekday dates in a year (approximate trading days)."""
    days = []
    d = datetime(year, 1, 2)
    end = datetime(year, 12, 31)
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            days.append(d)
        d += timedelta(days=1)
    return days


def sample_days(year, n=SAMPLES_PER_YEAR):
    """Pick n evenly-spaced trading days across the year."""
    days = get_trading_days(year)
    if len(days) <= n:
        return days
    # Evenly spaced + some random
    step = len(days) // n
    sampled = [days[i * step] for i in range(n)]
    return sampled


def load_spy_prices():
    """Load SPY daily close prices using yfinance."""
    import yfinance as yf
    spy = yf.download("SPY", start="2019-12-01", end="2026-01-01", progress=False)
    prices = {}
    for idx, row in spy.iterrows():
        date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)[:10]
        close_val = row['Close']
        # Handle potential multi-index from yfinance
        if hasattr(close_val, 'iloc'):
            close_val = close_val.iloc[0]
        prices[date_str] = float(close_val)
    return prices


def probe_year(conn, year, report_lines, spy_prices=None):
    """Deep probe of a single year's data availability."""
    cur = conn.cursor()
    days = sample_days(year)

    report_lines.append(f"\n## Year {year}\n")

    # --- Level 1: How many contracts exist per expiration? ---
    cur.execute("""
        SELECT expiration, COUNT(DISTINCT strike) as n_strikes
        FROM option_contracts
        WHERE ticker = ? AND option_type = ?
          AND expiration BETWEEN ? AND ?
        GROUP BY expiration
        ORDER BY expiration
    """, (TICKER, OPTION_TYPE,
          f"{year}-01-01", f"{year}-12-31"))
    expirations = cur.fetchall()
    report_lines.append(f"### Contract Coverage")
    report_lines.append(f"- Total expirations with put contracts: **{len(expirations)}**")
    if expirations:
        avg_strikes = sum(r[1] for r in expirations) / len(expirations)
        report_lines.append(f"- Avg strikes per expiration: {avg_strikes:.0f}")
        report_lines.append(f"- First: {expirations[0][0]}, Last: {expirations[-1][0]}")

    # --- Level 2: How many daily bars exist per month? ---
    cur.execute("""
        SELECT substr(od.date, 1, 7) as month, COUNT(*) as n_bars
        FROM option_daily od
        JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
        WHERE oc.ticker = ? AND oc.option_type = ?
          AND od.date BETWEEN ? AND ?
          AND od.date != '0000-00-00'
          AND od.close IS NOT NULL
        GROUP BY month
        ORDER BY month
    """, (TICKER, OPTION_TYPE,
          f"{year}-01-01", f"{year}-12-31"))
    monthly_bars = cur.fetchall()
    report_lines.append(f"\n### Daily Bars by Month")
    report_lines.append(f"| Month | Bars | Status |")
    report_lines.append(f"|-------|------|--------|")
    total_bars = 0
    for month, n_bars in monthly_bars:
        status = "OK" if n_bars > 100 else ("SPARSE" if n_bars > 0 else "EMPTY")
        report_lines.append(f"| {month} | {n_bars:,} | {status} |")
        total_bars += n_bars
    report_lines.append(f"| **Total** | **{total_bars:,}** | |")

    # Check for months with zero bars (gaps)
    months_with_bars = {m for m, _ in monthly_bars}
    all_months = {f"{year}-{m:02d}" for m in range(1, 13)}
    missing_months = sorted(all_months - months_with_bars)
    if missing_months:
        report_lines.append(f"\n**MISSING MONTHS (zero bars)**: {', '.join(missing_months)}")

    # --- Level 3: Per-sample-day deep probe ---
    report_lines.append(f"\n### Per-Day Probe ({len(days)} sample days)")
    report_lines.append(f"| Date | DTE40-50 Exps | Strikes | Short Leg Bars | Long Leg Bars | Spread OK | Rejection |")
    report_lines.append(f"|------|--------------|---------|---------------|---------------|-----------|-----------|")

    stats = {
        "days_probed": 0,
        "days_with_expirations": 0,
        "days_with_strikes": 0,
        "days_with_short_bars": 0,
        "days_with_long_bars": 0,
        "days_with_spread": 0,
        "rejection_reasons": defaultdict(int),
    }

    for day in days:
        date_str = day.strftime("%Y-%m-%d")
        stats["days_probed"] += 1

        # Find expirations with DTE 40-50
        exp_start = (day + timedelta(days=TARGET_DTE_MIN)).strftime("%Y-%m-%d")
        exp_end = (day + timedelta(days=TARGET_DTE_MAX)).strftime("%Y-%m-%d")

        cur.execute("""
            SELECT DISTINCT expiration
            FROM option_contracts
            WHERE ticker = ? AND option_type = ?
              AND expiration BETWEEN ? AND ?
            ORDER BY expiration
        """, (TICKER, OPTION_TYPE, exp_start, exp_end))
        valid_exps = [r[0] for r in cur.fetchall()]

        if not valid_exps:
            report_lines.append(f"| {date_str} | 0 | - | - | - | NO | no DTE 40-50 expirations |")
            stats["rejection_reasons"]["no_expirations_in_DTE_range"] += 1
            continue

        stats["days_with_expirations"] += 1

        # For each valid expiration, check strikes
        best_result = None
        for exp in valid_exps:
            # Get price on this date from yfinance data
            price = None
            if spy_prices:
                price = spy_prices.get(date_str)
                if not price:
                    # Try previous few days
                    for offset in range(1, 5):
                        prev = (day - timedelta(days=offset)).strftime("%Y-%m-%d")
                        price = spy_prices.get(prev)
                        if price:
                            break

            if not price:
                best_result = (len(valid_exps), "-", "-", "-", "NO", "no underlying price")
                stats["rejection_reasons"]["no_underlying_price"] += 1
                break
            target_short = price * (1 - OTM_PCT)

            # Get available put strikes for this expiration
            cur.execute("""
                SELECT DISTINCT strike
                FROM option_contracts
                WHERE ticker = ? AND expiration = ? AND option_type = ?
                ORDER BY strike
            """, (TICKER, exp, OPTION_TYPE))
            all_strikes = [r[0] for r in cur.fetchall()]

            # Filter to OTM puts (strike <= target)
            otm_strikes = [s for s in all_strikes if s <= target_short]
            if not otm_strikes:
                continue  # Try next expiration

            short_strike = max(otm_strikes)
            long_strike = short_strike - SPREAD_WIDTH
            n_strikes = len(otm_strikes)

            # Check if daily bars exist for SHORT leg on this date
            from shared.iron_vault import IronVault
            hd = IronVault.instance()
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
            short_sym = hd.build_occ_symbol(TICKER, exp_dt, short_strike, OPTION_TYPE)
            long_sym = hd.build_occ_symbol(TICKER, exp_dt, long_strike, OPTION_TYPE)

            cur.execute("""
                SELECT close FROM option_daily
                WHERE contract_symbol = ? AND date = ?
            """, (short_sym, date_str))
            short_bar = cur.fetchone()

            cur.execute("""
                SELECT close FROM option_daily
                WHERE contract_symbol = ? AND date = ?
            """, (long_sym, date_str))
            long_bar = cur.fetchone()

            # Also check if contract has ANY bars at all
            cur.execute("""
                SELECT COUNT(*), MIN(date), MAX(date)
                FROM option_daily
                WHERE contract_symbol = ? AND date != '0000-00-00' AND close IS NOT NULL
            """, (short_sym,))
            short_any = cur.fetchone()

            cur.execute("""
                SELECT COUNT(*), MIN(date), MAX(date)
                FROM option_daily
                WHERE contract_symbol = ? AND date != '0000-00-00' AND close IS NOT NULL
            """, (long_sym,))
            long_any = cur.fetchone()

            short_bar_val = short_bar[0] if short_bar else None
            long_bar_val = long_bar[0] if long_bar else None

            has_short = short_bar_val is not None
            has_long = long_bar_val is not None

            rejection = ""
            if not has_short and not has_long:
                rejection = f"BOTH legs missing"
                if short_any and short_any[0] == 0:
                    rejection += f" (short {short_sym}: 0 bars ever)"
                else:
                    rejection += f" (short: {short_any[0]} bars {short_any[1]}→{short_any[2]})"
                if long_any and long_any[0] == 0:
                    rejection += f" (long {long_sym}: 0 bars ever)"
                else:
                    rejection += f" (long: {long_any[0]} bars {long_any[1]}→{long_any[2]})"
            elif not has_short:
                rejection = f"short leg missing ({short_sym}: {short_any[0]} bars total, range {short_any[1]}→{short_any[2]})"
            elif not has_long:
                rejection = f"long leg missing ({long_sym}: {long_any[0]} bars total, range {long_any[1]}→{long_any[2]})"
            elif short_bar_val <= 0 or long_bar_val <= 0:
                rejection = f"zero/negative close (short={short_bar_val}, long={long_bar_val})"
            else:
                credit = short_bar_val - long_bar_val
                if credit <= 0:
                    rejection = f"negative credit ${credit:.2f} (short=${short_bar_val:.2f}, long=${long_bar_val:.2f})"
                else:
                    min_credit = SPREAD_WIDTH * 0.10  # 10% min credit (config default)
                    if credit < min_credit:
                        rejection = f"credit ${credit:.2f} < min ${min_credit:.2f}"
                    else:
                        rejection = "PASS"

            short_info = f"${short_bar_val:.2f}" if has_short else "MISS"
            long_info = f"${long_bar_val:.2f}" if has_long else "MISS"
            spread_ok = "YES" if rejection == "PASS" else "NO"

            if has_short:
                stats["days_with_short_bars"] += 1
            if has_long:
                stats["days_with_long_bars"] += 1
            if rejection == "PASS":
                stats["days_with_spread"] += 1

            if rejection != "PASS":
                reason_key = rejection.split("(")[0].strip()
                stats["rejection_reasons"][reason_key] += 1

            stats["days_with_strikes"] += 1
            best_result = (len(valid_exps), n_strikes, short_info, long_info, spread_ok, rejection)
            break  # Use first valid expiration

        if best_result is None:
            report_lines.append(f"| {date_str} | {len(valid_exps)} | 0 OTM | - | - | NO | no OTM strikes at 5% |")
            stats["rejection_reasons"]["no_OTM_strikes"] += 1
        else:
            exps, strikes, short_info, long_info, spread_ok, rejection = best_result
            rej_short = rejection[:60] if len(rejection) > 60 else rejection
            report_lines.append(f"| {date_str} | {exps} | {strikes} | {short_info} | {long_info} | {spread_ok} | {rej_short} |")

    # Summary stats
    report_lines.append(f"\n### {year} Summary")
    report_lines.append(f"- Days probed: {stats['days_probed']}")
    report_lines.append(f"- Days with DTE 40-50 expirations: {stats['days_with_expirations']} ({stats['days_with_expirations']/max(1,stats['days_probed'])*100:.0f}%)")
    report_lines.append(f"- Days with OTM strikes: {stats['days_with_strikes']} ({stats['days_with_strikes']/max(1,stats['days_probed'])*100:.0f}%)")
    report_lines.append(f"- Days with short leg bar: {stats['days_with_short_bars']} ({stats['days_with_short_bars']/max(1,stats['days_probed'])*100:.0f}%)")
    report_lines.append(f"- Days with long leg bar: {stats['days_with_long_bars']} ({stats['days_with_long_bars']/max(1,stats['days_probed'])*100:.0f}%)")
    report_lines.append(f"- Days with valid spread: {stats['days_with_spread']} ({stats['days_with_spread']/max(1,stats['days_probed'])*100:.0f}%)")

    if stats["rejection_reasons"]:
        report_lines.append(f"\n**Rejection breakdown:**")
        for reason, count in sorted(stats["rejection_reasons"].items(), key=lambda x: -x[1]):
            report_lines.append(f"- {reason}: {count}")

    return stats


def probe_intraday_coverage(conn, report_lines):
    """Check intraday bar coverage since backtester uses intraday scan times."""
    cur = conn.cursor()
    report_lines.append(f"\n## Intraday Bar Coverage\n")
    report_lines.append(f"The backtester uses 14 scan times per day (9:30-16:00 ET).")
    report_lines.append(f"If intraday bars are missing, it falls back to daily bars.\n")

    report_lines.append(f"| Year | Intraday Bars | Contracts w/ Intraday | Dates w/ Intraday |")
    report_lines.append(f"|------|--------------|----------------------|-------------------|")

    for year in YEARS:
        cur.execute("""
            SELECT COUNT(*) FROM option_intraday oi
            JOIN option_contracts oc ON oi.contract_symbol = oc.contract_symbol
            WHERE oc.ticker = ? AND oc.option_type = ?
              AND oi.date BETWEEN ? AND ?
              AND oi.bar_time != 'FETCHED'
        """, (TICKER, OPTION_TYPE, f"{year}-01-01", f"{year}-12-31"))
        n_bars = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(DISTINCT oi.contract_symbol) FROM option_intraday oi
            JOIN option_contracts oc ON oi.contract_symbol = oc.contract_symbol
            WHERE oc.ticker = ? AND oc.option_type = ?
              AND oi.date BETWEEN ? AND ?
              AND oi.bar_time != 'FETCHED'
        """, (TICKER, OPTION_TYPE, f"{year}-01-01", f"{year}-12-31"))
        n_contracts = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(DISTINCT oi.date) FROM option_intraday oi
            JOIN option_contracts oc ON oi.contract_symbol = oc.contract_symbol
            WHERE oc.ticker = ? AND oc.option_type = ?
              AND oi.date BETWEEN ? AND ?
              AND oi.bar_time != 'FETCHED'
        """, (TICKER, OPTION_TYPE, f"{year}-01-01", f"{year}-12-31"))
        n_dates = cur.fetchone()[0]

        status = "DENSE" if n_bars > 10000 else ("SPARSE" if n_bars > 0 else "EMPTY")
        report_lines.append(f"| {year} | {n_bars:,} ({status}) | {n_contracts:,} | {n_dates} |")


def probe_expiration_patterns(conn, report_lines):
    """Check what expiration frequencies exist — daily/weekly/monthly."""
    cur = conn.cursor()
    report_lines.append(f"\n## Expiration Frequency by Year\n")
    report_lines.append(f"W=$10 OTM=5% DTE=45 targets specific expirations.")
    report_lines.append(f"If expirations are only monthly, there are fewer entry opportunities.\n")

    for year in YEARS:
        cur.execute("""
            SELECT DISTINCT expiration
            FROM option_contracts
            WHERE ticker = ? AND option_type = ?
              AND expiration BETWEEN ? AND ?
            ORDER BY expiration
        """, (TICKER, OPTION_TYPE, f"{year}-01-01", f"{year}-12-31"))
        exps = [r[0] for r in cur.fetchall()]

        if len(exps) < 2:
            report_lines.append(f"**{year}**: {len(exps)} expirations (NO DATA)")
            continue

        # Calculate intervals between expirations
        from datetime import date as date_type
        exp_dates = []
        for e in exps:
            try:
                exp_dates.append(datetime.strptime(e, "%Y-%m-%d").date())
            except ValueError:
                continue

        if len(exp_dates) < 2:
            report_lines.append(f"**{year}**: {len(exps)} expirations (parse error)")
            continue

        intervals = [(exp_dates[i+1] - exp_dates[i]).days for i in range(len(exp_dates)-1)]
        avg_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)

        freq = "DAILY" if avg_interval <= 2 else ("WEEKLY" if avg_interval <= 8 else "MONTHLY")
        report_lines.append(f"**{year}**: {len(exps)} expirations, avg interval {avg_interval:.1f}d "
                          f"(min={min_interval}d, max={max_interval}d) → **{freq}**")


def probe_daily_bar_density(conn, report_lines):
    """For contracts with DTE 40-50, how many daily bars exist per contract?"""
    cur = conn.cursor()
    report_lines.append(f"\n## Daily Bar Density for DTE 40-50 Contracts\n")
    report_lines.append(f"How many days of pricing data does each contract have?\n")

    report_lines.append(f"| Year | Contracts | Avg Bars/Contract | Median | Min | Max | 0-bar contracts |")
    report_lines.append(f"|------|-----------|-------------------|--------|-----|-----|-----------------|")

    for year in YEARS:
        # Get all contracts that would be in DTE 40-50 range during this year
        cur.execute("""
            SELECT oc.contract_symbol, oc.expiration, oc.strike
            FROM option_contracts oc
            WHERE oc.ticker = ? AND oc.option_type = ?
              AND oc.expiration BETWEEN ? AND ?
              AND oc.strike > 0
        """, (TICKER, OPTION_TYPE,
              f"{year}-01-01", f"{year}-12-31"))
        contracts = cur.fetchall()

        if not contracts:
            report_lines.append(f"| {year} | 0 | - | - | - | - | - |")
            continue

        # Sample up to 200 contracts to keep query time reasonable
        if len(contracts) > 200:
            random.seed(42 + year)
            contracts = random.sample(contracts, 200)

        bar_counts = []
        zero_bar = 0
        for sym, exp, strike in contracts:
            cur.execute("""
                SELECT COUNT(*) FROM option_daily
                WHERE contract_symbol = ? AND date != '0000-00-00' AND close IS NOT NULL
            """, (sym,))
            n = cur.fetchone()[0]
            bar_counts.append(n)
            if n == 0:
                zero_bar += 1

        if bar_counts:
            bar_counts.sort()
            avg_bars = sum(bar_counts) / len(bar_counts)
            median_bars = bar_counts[len(bar_counts) // 2]
            min_bars = bar_counts[0]
            max_bars = bar_counts[-1]
            report_lines.append(f"| {year} | {len(contracts)} sampled | {avg_bars:.1f} | {median_bars} | {min_bars} | {max_bars} | {zero_bar} ({zero_bar/len(contracts)*100:.0f}%) |")


def run_mini_backtest_trace(report_lines):
    """Actually run the backtester for a few days in 2021 and 2023 with verbose logging."""
    import logging
    from datetime import datetime as dt

    from backtest.backtester import Backtester
    from shared.iron_vault import IronVault

    report_lines.append(f"\n## Mini Backtest Trace (2021 & 2023)\n")
    report_lines.append(f"Running backtester on specific months with DEBUG logging to trace rejections.\n")

    config = {
        "strategy": {
            "target_delta": 0.12,
            "use_delta_selection": False,
            "target_dte": 45,
            "min_dte": 35,
            "spread_width": 10,
            "min_credit_pct": 10,
            "direction": "both",
            "trend_ma_period": 50,
            "regime_mode": "combo",
            "regime_config": {},
            "momentum_filter_pct": None,
            "iron_condor": {"enabled": False},
            "iv_rank_min_entry": 0,
            "vix_max_entry": 0,
            "vix_close_all": 0,
            "vix_dynamic_sizing": {},
            "seasonal_sizing": {},
            "compass_enabled": False,
            "compass_rrg_filter": False,
        },
        "risk": {
            "stop_loss_multiplier": 2.5,
            "profit_target": 50,
            "max_risk_per_trade": 2.0,
            "max_contracts": 25,
            "max_positions": 50,
            "drawdown_cb_pct": 25,
        },
        "backtest": {
            "starting_capital": 100_000,
            "commission_per_contract": 0.65,
            "slippage": 0.05,
            "exit_slippage": 0.10,
            "compound": False,
            "sizing_mode": "flat",
            "slippage_multiplier": 1.0,
            "max_portfolio_exposure_pct": 100.0,
            "exclude_months": [],
            "volume_gate": False,
            "oi_gate": False,
        },
    }

    # Set up logging to capture debug messages
    log_capture = []
    handler = logging.Handler()

    class ListHandler(logging.Handler):
        def emit(self, record):
            log_capture.append(self.format(record))

    list_handler = ListHandler()
    list_handler.setLevel(logging.DEBUG)
    list_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    bt_logger = logging.getLogger("backtest.backtester")
    bt_logger.setLevel(logging.DEBUG)
    bt_logger.addHandler(list_handler)

    hd = IronVault.instance()

    for year, month_label, start, end in [
        (2021, "Q1 Jan-Mar", dt(2021, 1, 1), dt(2021, 3, 31)),
        (2021, "Q2 Apr-Jun", dt(2021, 4, 1), dt(2021, 6, 30)),
        (2021, "Q3 Jul-Sep", dt(2021, 7, 1), dt(2021, 9, 30)),
        (2021, "Q4 Oct-Dec", dt(2021, 10, 1), dt(2021, 12, 31)),
        (2022, "Q1 Jan-Mar", dt(2022, 1, 1), dt(2022, 3, 31)),
        (2022, "Q2 Apr-Jun", dt(2022, 4, 1), dt(2022, 6, 30)),
        (2022, "Q3 Jul-Sep", dt(2022, 7, 1), dt(2022, 9, 30)),
        (2022, "Q4 Oct-Dec", dt(2022, 10, 1), dt(2022, 12, 31)),
        (2023, "Q1 Jan-Mar", dt(2023, 1, 1), dt(2023, 3, 31)),
        (2023, "Q2 Apr-Jun", dt(2023, 4, 1), dt(2023, 6, 30)),
        (2023, "Q3 Jul-Sep", dt(2023, 7, 1), dt(2023, 9, 30)),
        (2023, "Q4 Oct-Dec", dt(2023, 10, 1), dt(2023, 12, 31)),
        (2025, "Q1 Jan-Mar", dt(2025, 1, 1), dt(2025, 3, 31)),
        (2025, "Q2 Apr-Jun", dt(2025, 4, 1), dt(2025, 6, 30)),
        (2025, "Q3 Jul-Sep", dt(2025, 7, 1), dt(2025, 9, 30)),
        (2025, "Q4 Oct-Dec", dt(2025, 10, 1), dt(2025, 12, 31)),
    ]:
        log_capture.clear()
        bt = Backtester(config, historical_data=hd, otm_pct=0.05)
        result = bt.run_backtest(TICKER, start, end)
        result = result or {}

        trades = result.get("total_trades", 0)
        ret = result.get("return_pct", 0)

        # Count rejection types from debug logs
        no_strikes = sum(1 for l in log_capture if "No strikes" in l)
        no_price = sum(1 for l in log_capture if "No" in l and "price data" in l)
        below_min = sum(1 for l in log_capture if "below minimum" in l)
        negative_credit = sum(1 for l in log_capture if "credit" in l.lower() and "<=" in l)
        vol_gate = sum(1 for l in log_capture if "vol-gate" in l)
        opened = sum(1 for l in log_capture if "Opened" in l)

        report_lines.append(f"**{year} {month_label}**: {trades} trades, {ret:+.1f}% return")
        report_lines.append(f"  - Debug log entries: {len(log_capture)}")
        report_lines.append(f"  - 'No strikes': {no_strikes}")
        report_lines.append(f"  - 'No price data': {no_price}")
        report_lines.append(f"  - 'Below minimum credit': {below_min}")
        report_lines.append(f"  - 'Opened': {opened}")
        report_lines.append(f"  - Volume gate rejections: {vol_gate}")
        report_lines.append("")

    bt_logger.removeHandler(list_handler)


def probe_bar_availability_vs_dte(conn, report_lines):
    """Key question: at what DTE do daily bars actually appear?"""
    cur = conn.cursor()
    report_lines.append(f"\n## Daily Bar Availability vs DTE\n")
    report_lines.append(f"For each year, sample 50 OTM put contracts and check: at what DTE do daily bars first appear?\n")

    from shared.iron_vault import IronVault
    hd = IronVault.instance()

    report_lines.append(f"| Year | Contracts Sampled | Avg First-Bar DTE | Median | Min DTE | Max DTE | Never-Bars |")
    report_lines.append(f"|------|------------------|-------------------|--------|---------|---------|------------|")

    for year in YEARS:
        # Get OTM put contracts with known expirations
        cur.execute("""
            SELECT oc.contract_symbol, oc.expiration, oc.strike
            FROM option_contracts oc
            WHERE oc.ticker = ? AND oc.option_type = 'P'
              AND oc.expiration BETWEEN ? AND ?
              AND oc.strike > 0
            ORDER BY RANDOM()
            LIMIT 200
        """, (TICKER, f"{year}-01-01", f"{year}-12-31"))
        contracts = cur.fetchall()

        first_bar_dtes = []
        never_bars = 0

        for sym, exp, strike in contracts[:50]:
            # Get all daily bars for this contract
            cur.execute("""
                SELECT date FROM option_daily
                WHERE contract_symbol = ? AND date != '0000-00-00' AND close IS NOT NULL
                ORDER BY date ASC
                LIMIT 1
            """, (sym,))
            first_bar = cur.fetchone()

            if not first_bar:
                never_bars += 1
                continue

            try:
                first_date = datetime.strptime(first_bar[0], "%Y-%m-%d")
                exp_date = datetime.strptime(exp, "%Y-%m-%d")
                dte_at_first_bar = (exp_date - first_date).days
                first_bar_dtes.append(dte_at_first_bar)
            except ValueError:
                never_bars += 1

        if first_bar_dtes:
            first_bar_dtes.sort()
            avg_dte = sum(first_bar_dtes) / len(first_bar_dtes)
            median_dte = first_bar_dtes[len(first_bar_dtes) // 2]
            min_dte = min(first_bar_dtes)
            max_dte = max(first_bar_dtes)
            report_lines.append(f"| {year} | {len(contracts[:50])} | {avg_dte:.0f} | {median_dte} | {min_dte} | {max_dte} | {never_bars} |")
        else:
            report_lines.append(f"| {year} | {len(contracts[:50])} | N/A | N/A | N/A | N/A | {never_bars} |")

    report_lines.append("")
    report_lines.append("**Interpretation**: If avg first-bar DTE is ~20, bars only appear 20 days before expiration.")
    report_lines.append("With target_dte=45, the backtester tries to enter at 45 DTE but bars don't exist until ~20 DTE.")
    report_lines.append("Trades only happen on dates where bars coincidentally exist early (rare).")


def main():
    conn = sqlite3.connect(str(DB_PATH))
    report_lines = [
        "# EXP-600 Trade Flow Debug",
        f"Generated: {datetime.utcnow().isoformat()}",
        "",
        "## Mystery",
        "For config DTE=45, W=$10, OTM=5%, PT=50%, SL=2.5x, risk=2%:",
        "- 2020: 49 trades | 2021: **2 trades** | 2022: 68 trades",
        "- 2023: **2 trades** | 2024: 8 trades | 2025: 58 trades",
        "",
        "Hypothesis: DB has contracts listed but missing daily pricing bars for 2021/2023.",
        "",
    ]

    print("=" * 70)
    print("  EXP-600 Trade Flow Debug")
    print("=" * 70)

    print("\n[0/10] Loading SPY prices...")
    spy_prices = load_spy_prices()
    print(f"  Loaded {len(spy_prices)} daily SPY prices")

    # Probe 1: Expiration patterns
    print("\n[1/6] Checking expiration frequency patterns...")
    probe_expiration_patterns(conn, report_lines)

    # Probe 2: Daily bar coverage
    print("[2/6] Checking daily bar density per contract...")
    probe_daily_bar_density(conn, report_lines)

    # Probe 3: Intraday coverage
    print("[3/6] Checking intraday bar coverage...")
    probe_intraday_coverage(conn, report_lines)

    # Probe 4-9: Per-year deep probes
    all_stats = {}
    for i, year in enumerate(YEARS):
        print(f"[{4+i}/9] Deep probe year {year}...")
        stats = probe_year(conn, year, report_lines, spy_prices=spy_prices)
        all_stats[year] = stats

    # Probe: Bar availability vs DTE
    print("[10/11] Checking bar availability vs DTE...")
    probe_bar_availability_vs_dte(conn, report_lines)

    # Probe 11: Mini backtest trace
    print("[11/11] Running mini backtest traces...")
    run_mini_backtest_trace(report_lines)

    # Cross-year comparison
    report_lines.append(f"\n## Cross-Year Comparison\n")
    report_lines.append(f"| Year | Probed | Has Exps | Has Strikes | Short Bar | Long Bar | Valid Spread | Spread Rate |")
    report_lines.append(f"|------|--------|----------|-------------|-----------|----------|-------------|-------------|")
    for year in YEARS:
        s = all_stats[year]
        rate = s["days_with_spread"] / max(1, s["days_probed"]) * 100
        report_lines.append(
            f"| {year} | {s['days_probed']} | {s['days_with_expirations']} | {s['days_with_strikes']} | "
            f"{s['days_with_short_bars']} | {s['days_with_long_bars']} | {s['days_with_spread']} | {rate:.0f}% |"
        )

    # Write report
    report_text = "\n".join(report_lines)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write(report_text)

    print(f"\nReport saved to: {OUTPUT_PATH}")
    print(f"\n{'='*70}")
    print("  QUICK SUMMARY")
    print(f"{'='*70}")
    for year in YEARS:
        s = all_stats[year]
        rate = s["days_with_spread"] / max(1, s["days_probed"]) * 100
        top_reason = max(s["rejection_reasons"].items(), key=lambda x: x[1])[0] if s["rejection_reasons"] else "none"
        print(f"  {year}: {s['days_with_spread']}/{s['days_probed']} valid spreads ({rate:.0f}%) — top rejection: {top_reason}")

    conn.close()


if __name__ == "__main__":
    main()
