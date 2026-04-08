#!/usr/bin/env python3
"""
EXP-600 Diagnostic: Probe options_cache.db to understand what data is available.

For each year 2020-2025, pick 3 sample dates (Feb, Jun, Oct) and check:
1. How many contracts exist per expiration
2. What strike ranges are available (puts and calls)
3. What spread widths are feasible (strike spacing)
4. How many daily bars exist for those contracts
5. Whether short-DTE (15d) vs medium-DTE (30-45d) expirations have data

Output: results/exp600/diagnostic.json + human-readable summary to stdout
"""

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data" / "options_cache.db"
OUTPUT_DIR = ROOT / "results" / "exp600"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_db():
    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(str(DB_PATH))


def overall_stats(conn):
    """Get high-level DB stats."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM option_contracts")
    contracts = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM option_daily WHERE date != '0000-00-00'")
    daily_bars = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM option_contracts WHERE ticker = 'SPY'")
    spy_contracts = cur.fetchone()[0]

    # Contracts by ticker
    cur.execute("""
        SELECT ticker, COUNT(*) as cnt,
               MIN(expiration) as first_exp, MAX(expiration) as last_exp
        FROM option_contracts
        GROUP BY ticker ORDER BY cnt DESC
    """)
    by_ticker = []
    for row in cur.fetchall():
        by_ticker.append({
            "ticker": row[0], "contracts": row[1],
            "first_exp": row[2], "last_exp": row[3]
        })

    # Contracts by year for SPY
    cur.execute("""
        SELECT substr(expiration, 1, 4) as yr, COUNT(*) as cnt
        FROM option_contracts
        WHERE ticker = 'SPY'
        GROUP BY yr ORDER BY yr
    """)
    spy_by_year = {row[0]: row[1] for row in cur.fetchall()}

    return {
        "total_contracts": contracts,
        "total_daily_bars": daily_bars,
        "spy_contracts": spy_contracts,
        "by_ticker": by_ticker,
        "spy_contracts_by_year": spy_by_year,
    }


def probe_date(conn, ticker, sample_date_str):
    """For a given date, find what expirations and strikes are available."""
    cur = conn.cursor()
    sample_dt = datetime.strptime(sample_date_str, "%Y-%m-%d")

    results = {}

    # Find all expirations for this ticker where expiration > sample_date
    # and within 60 days
    max_exp = (sample_dt + timedelta(days=60)).strftime("%Y-%m-%d")
    cur.execute("""
        SELECT DISTINCT expiration, option_type, COUNT(*) as strike_count
        FROM option_contracts
        WHERE ticker = ? AND expiration > ? AND expiration <= ?
        GROUP BY expiration, option_type
        ORDER BY expiration, option_type
    """, (ticker, sample_date_str, max_exp))

    expirations = []
    for row in cur.fetchall():
        exp_date = row[0]
        opt_type = row[1]
        strike_count = row[2]
        dte = (datetime.strptime(exp_date, "%Y-%m-%d") - sample_dt).days

        # Get actual strikes
        cur.execute("""
            SELECT strike FROM option_contracts
            WHERE ticker = ? AND expiration = ? AND option_type = ?
            ORDER BY strike
        """, (ticker, exp_date, opt_type))
        strikes = [r[0] for r in cur.fetchall()]

        # Check how many of these contracts have daily bars on or near the sample date
        # Look for bars within 3 days of sample date
        bars_found = 0
        bars_checked = 0
        sample_contracts_with_bars = []
        for strike in strikes[:5]:  # Check first 5 strikes
            bars_checked += 1
            # Build OCC symbol
            exp_dt = datetime.strptime(exp_date, "%Y-%m-%d")
            exp_str = exp_dt.strftime("%y%m%d")
            strike_int = int(round(strike * 1000))
            symbol = f"O:{ticker}{exp_str}{opt_type}{strike_int:08d}"
            cur.execute("""
                SELECT date, close FROM option_daily
                WHERE contract_symbol = ? AND date >= ? AND date <= ?
                  AND date != '0000-00-00'
                ORDER BY date
            """, (symbol, sample_date_str,
                  (sample_dt + timedelta(days=5)).strftime("%Y-%m-%d")))
            rows = cur.fetchall()
            if rows:
                bars_found += 1
                sample_contracts_with_bars.append({
                    "symbol": symbol, "strike": strike,
                    "bars": [{"date": r[0], "close": r[1]} for r in rows[:3]]
                })

        # Strike spacing analysis
        if len(strikes) >= 2:
            spacings = [strikes[i+1] - strikes[i] for i in range(len(strikes)-1)]
            min_spacing = min(spacings)
            max_spacing = max(spacings)
            median_spacing = sorted(spacings)[len(spacings)//2]
        else:
            min_spacing = max_spacing = median_spacing = None

        expirations.append({
            "expiration": exp_date,
            "option_type": opt_type,
            "dte": dte,
            "strike_count": strike_count,
            "strike_min": min(strikes) if strikes else None,
            "strike_max": max(strikes) if strikes else None,
            "min_spacing": min_spacing,
            "max_spacing": max_spacing,
            "median_spacing": median_spacing,
            "bars_checked": bars_checked,
            "bars_found": bars_found,
            "sample_bars": sample_contracts_with_bars,
        })

    # Also check: for a typical 30-35 DTE Friday expiration, what does the full chain look like?
    target_exp_dt = sample_dt + timedelta(days=30)
    # Find nearest Friday
    days_to_friday = (4 - target_exp_dt.weekday()) % 7
    friday_exp = target_exp_dt + timedelta(days=days_to_friday)
    friday_exp_str = friday_exp.strftime("%Y-%m-%d")

    cur.execute("""
        SELECT strike, option_type FROM option_contracts
        WHERE ticker = ? AND expiration = ?
        ORDER BY option_type, strike
    """, (ticker, friday_exp_str))
    friday_chain = cur.fetchall()

    # Check daily bars for the Friday chain
    friday_bars_total = 0
    friday_bars_with_data = 0
    if friday_chain:
        for strike, opt_type in friday_chain[:20]:  # Sample 20
            friday_bars_total += 1
            exp_str = friday_exp.strftime("%y%m%d")
            strike_int = int(round(strike * 1000))
            symbol = f"O:{ticker}{exp_str}{opt_type}{strike_int:08d}"
            cur.execute("""
                SELECT 1 FROM option_daily
                WHERE contract_symbol = ? AND date = ? AND close IS NOT NULL
            """, (symbol, sample_date_str))
            if cur.fetchone():
                friday_bars_with_data += 1

    results["sample_date"] = sample_date_str
    results["expirations_found"] = len(expirations)
    results["expirations"] = expirations
    results["friday_30dte"] = {
        "target_expiration": friday_exp_str,
        "contracts_in_chain": len(friday_chain),
        "put_strikes": [r[0] for r in friday_chain if r[1] == 'P'],
        "call_strikes": [r[0] for r in friday_chain if r[1] == 'C'],
        "bars_sampled": friday_bars_total,
        "bars_with_data_on_sample_date": friday_bars_with_data,
    }

    # Check a 15-DTE expiration (champion config uses target_dte=15)
    target_15 = sample_dt + timedelta(days=15)
    days_to_friday_15 = (4 - target_15.weekday()) % 7
    friday_15 = target_15 + timedelta(days=days_to_friday_15)
    friday_15_str = friday_15.strftime("%Y-%m-%d")

    cur.execute("""
        SELECT strike, option_type FROM option_contracts
        WHERE ticker = ? AND expiration = ?
        ORDER BY option_type, strike
    """, (ticker, friday_15_str))
    chain_15 = cur.fetchall()

    results["friday_15dte"] = {
        "target_expiration": friday_15_str,
        "contracts_in_chain": len(chain_15),
        "put_strikes": [r[0] for r in chain_15 if r[1] == 'P'],
        "call_strikes": [r[0] for r in chain_15 if r[1] == 'C'],
    }

    return results


def simulate_spread_match(conn, ticker, sample_date_str, target_dte, spread_width, otm_pct):
    """Simulate what _find_real_spread would do: find strikes, check prices."""
    cur = conn.cursor()
    sample_dt = datetime.strptime(sample_date_str, "%Y-%m-%d")

    # Find nearest Friday expiration at target_dte
    target_exp = sample_dt + timedelta(days=target_dte)
    days_to_friday = (4 - target_exp.weekday()) % 7
    friday_exp = target_exp + timedelta(days=days_to_friday)
    friday_exp_str = friday_exp.strftime("%Y-%m-%d")

    # Also try MWF
    _MWF = {0, 2, 4}
    mwf_exp = target_exp
    for _ in range(4):
        if mwf_exp.weekday() in _MWF:
            break
        mwf_exp += timedelta(days=1)
    mwf_exp_str = mwf_exp.strftime("%Y-%m-%d")

    # We need SPY price on sample_date — get from Yahoo (or estimate)
    # For diagnostic, use rough SPY prices by year
    spy_prices = {
        "2020": 320, "2021": 430, "2022": 420, "2023": 450, "2024": 490, "2025": 560
    }
    year = sample_date_str[:4]
    approx_price = spy_prices.get(year, 450)

    results = {"sample_date": sample_date_str, "approx_spy_price": approx_price, "tests": []}

    for exp_str, exp_label in [(friday_exp_str, "friday"), (mwf_exp_str, "mwf")]:
        for opt_type, direction in [("P", "bull_put"), ("C", "bear_call")]:
            # Get available strikes
            cur.execute("""
                SELECT strike FROM option_contracts
                WHERE ticker = ? AND expiration = ? AND option_type = ?
                ORDER BY strike
            """, (ticker, exp_str, opt_type))
            strikes = [r[0] for r in cur.fetchall()]

            if not strikes:
                results["tests"].append({
                    "exp": exp_str, "exp_type": exp_label,
                    "direction": direction, "option_type": opt_type,
                    "strikes_found": 0, "result": "NO_CONTRACTS"
                })
                continue

            # Select short strike (OTM%)
            if opt_type == "P":
                target_short = approx_price * (1 - otm_pct)
                candidates = [s for s in strikes if s <= target_short]
                short_strike = max(candidates) if candidates else None
            else:
                target_short = approx_price * (1 + otm_pct)
                candidates = [s for s in strikes if s >= target_short]
                short_strike = min(candidates) if candidates else None

            if short_strike is None:
                results["tests"].append({
                    "exp": exp_str, "exp_type": exp_label,
                    "direction": direction, "option_type": opt_type,
                    "strikes_found": len(strikes),
                    "target_short": round(target_short, 1),
                    "result": "NO_OTM_STRIKE"
                })
                continue

            # Compute long strike
            long_strike = (short_strike - spread_width) if opt_type == "P" else (short_strike + spread_width)

            # Check if long strike exists in chain
            long_in_chain = long_strike in strikes

            # Check daily bars for both legs
            exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
            exp_ymd = exp_dt.strftime("%y%m%d")

            def check_bar(strike_val):
                strike_int = int(round(strike_val * 1000))
                symbol = f"O:{ticker}{exp_ymd}{opt_type}{strike_int:08d}"
                cur.execute("""
                    SELECT date, close FROM option_daily
                    WHERE contract_symbol = ? AND date = ? AND close IS NOT NULL
                """, (symbol, sample_date_str))
                row = cur.fetchone()
                if row:
                    return {"symbol": symbol, "has_bar": True, "close": row[1]}
                # Check nearby dates
                cur.execute("""
                    SELECT date, close FROM option_daily
                    WHERE contract_symbol = ? AND date >= ? AND date <= ?
                      AND date != '0000-00-00' AND close IS NOT NULL
                    ORDER BY date LIMIT 3
                """, (symbol,
                      (sample_dt - timedelta(days=5)).strftime("%Y-%m-%d"),
                      (sample_dt + timedelta(days=5)).strftime("%Y-%m-%d")))
                nearby = cur.fetchall()
                return {
                    "symbol": symbol, "has_bar": False,
                    "nearby_bars": [{"date": r[0], "close": r[1]} for r in nearby]
                }

            short_bar = check_bar(short_strike)
            long_bar = check_bar(long_strike)

            can_trade = short_bar["has_bar"] and long_bar["has_bar"]

            # Try adjacent strikes if primary fails
            alt_result = None
            if not can_trade:
                for offset in [1, -1, 2, -2]:
                    alt_short = short_strike + offset
                    alt_long = (alt_short - spread_width) if opt_type == "P" else (alt_short + spread_width)
                    alt_short_bar = check_bar(alt_short)
                    alt_long_bar = check_bar(alt_long)
                    if alt_short_bar["has_bar"] and alt_long_bar["has_bar"]:
                        alt_result = {
                            "offset": offset,
                            "short_strike": alt_short,
                            "long_strike": alt_long,
                            "short_close": alt_short_bar["close"],
                            "long_close": alt_long_bar["close"],
                        }
                        break

            results["tests"].append({
                "exp": exp_str, "exp_type": exp_label,
                "direction": direction, "option_type": opt_type,
                "dte": (exp_dt - sample_dt).days,
                "strikes_found": len(strikes),
                "short_strike": short_strike,
                "long_strike": long_strike,
                "long_in_chain": long_in_chain,
                "short_bar": short_bar,
                "long_bar": long_bar,
                "can_trade": can_trade,
                "alt_found": alt_result,
                "result": "TRADE_OK" if can_trade else ("ALT_FOUND" if alt_result else "NO_BARS"),
            })

    return results


def main():
    conn = get_db()
    print("=" * 80)
    print("  EXP-600 DIAGNOSTIC: Iron Vault DB Probe")
    print("=" * 80)

    # 1. Overall stats
    stats = overall_stats(conn)
    print(f"\n--- DB Overview ---")
    print(f"  Total contracts:   {stats['total_contracts']:,}")
    print(f"  Total daily bars:  {stats['total_daily_bars']:,}")
    print(f"  SPY contracts:     {stats['spy_contracts']:,}")
    print(f"\n  By ticker:")
    for t in stats["by_ticker"]:
        print(f"    {t['ticker']:6s}  {t['contracts']:>8,} contracts  ({t['first_exp']} → {t['last_exp']})")
    print(f"\n  SPY contracts by expiration year:")
    for yr, cnt in sorted(stats["spy_contracts_by_year"].items()):
        print(f"    {yr}: {cnt:>8,}")

    # 2. Per-year probes — 3 dates per year
    sample_dates = [
        "2020-02-10", "2020-06-15", "2020-10-15",
        "2021-02-10", "2021-06-15", "2021-10-15",
        "2022-02-10", "2022-06-15", "2022-10-15",
        "2023-02-12", "2023-06-15", "2023-10-16",
        "2024-02-12", "2024-06-17", "2024-10-15",
        "2025-01-13", "2025-02-10",
    ]

    all_probes = {}
    all_spread_tests = {}

    print(f"\n--- Expiration & Strike Probes ---")
    for sd in sample_dates:
        probe = probe_date(conn, "SPY", sd)
        all_probes[sd] = probe
        yr = sd[:4]

        friday_30 = probe["friday_30dte"]
        friday_15 = probe["friday_15dte"]
        print(f"\n  {sd}:")
        print(f"    Expirations within 60d: {probe['expirations_found']}")
        print(f"    30-DTE Friday ({friday_30['target_expiration']}): "
              f"{len(friday_30['put_strikes'])} puts, {len(friday_30['call_strikes'])} calls, "
              f"bars={friday_30['bars_with_data_on_sample_date']}/{friday_30['bars_sampled']}")
        print(f"    15-DTE Friday ({friday_15['target_expiration']}): "
              f"{len(friday_15['put_strikes'])} puts, {len(friday_15['call_strikes'])} calls")

        # Show strike spacing for first put expiration
        for exp in probe["expirations"][:3]:
            if exp["option_type"] == "P" and exp["strike_count"] > 0:
                print(f"      exp={exp['expiration']} (DTE {exp['dte']}): "
                      f"{exp['strike_count']} strikes, "
                      f"range ${exp['strike_min']}-${exp['strike_max']}, "
                      f"spacing ${exp['min_spacing']}-${exp['max_spacing']} "
                      f"(median ${exp['median_spacing']}), "
                      f"bars {exp['bars_found']}/{exp['bars_checked']}")
                break

    # 3. Spread matching simulation — test multiple configs
    print(f"\n--- Spread Matching Simulation ---")
    test_configs = [
        {"target_dte": 15, "spread_width": 12, "otm_pct": 0.02, "label": "champion (15d, $12w, 2% OTM)"},
        {"target_dte": 30, "spread_width": 5, "otm_pct": 0.03, "label": "conservative (30d, $5w, 3% OTM)"},
        {"target_dte": 35, "spread_width": 5, "otm_pct": 0.05, "label": "standard (35d, $5w, 5% OTM)"},
        {"target_dte": 30, "spread_width": 3, "otm_pct": 0.03, "label": "narrow (30d, $3w, 3% OTM)"},
        {"target_dte": 30, "spread_width": 1, "otm_pct": 0.03, "label": "tightest (30d, $1w, 3% OTM)"},
        {"target_dte": 45, "spread_width": 5, "otm_pct": 0.05, "label": "wide DTE (45d, $5w, 5% OTM)"},
    ]

    # Test a subset of dates per config
    test_dates = ["2020-02-10", "2020-06-15", "2021-06-15", "2022-06-15",
                  "2023-06-15", "2024-06-17", "2025-02-10"]

    for cfg in test_configs:
        print(f"\n  Config: {cfg['label']}")
        ok_count = 0
        total = 0
        for sd in test_dates:
            total += 1
            sm = simulate_spread_match(
                conn, "SPY", sd, cfg["target_dte"], cfg["spread_width"], cfg["otm_pct"]
            )
            # Check if any test produced TRADE_OK or ALT_FOUND
            any_ok = any(t["result"] in ("TRADE_OK", "ALT_FOUND") for t in sm["tests"])
            if any_ok:
                ok_count += 1
            best = next((t for t in sm["tests"] if t["result"] == "TRADE_OK"), None)
            alt = next((t for t in sm["tests"] if t["result"] == "ALT_FOUND"), None)
            status = "OK" if best else ("ALT" if alt else "MISS")

            detail = ""
            if best:
                detail = (f" short=${best['short_strike']} long=${best['long_strike']} "
                         f"short_px=${best['short_bar']['close']:.2f} long_px=${best['long_bar']['close']:.2f} "
                         f"credit=${best['short_bar']['close'] - best['long_bar']['close']:.2f}")
            elif alt:
                a = alt.get("alt_found", alt)  # handle both direct and nested
                detail = f" (alt: offset={a.get('offset','?')}, credit=${a.get('short_close',0) - a.get('long_close',0):.2f})"
            else:
                miss_reasons = [f"{t['exp_type']}/{t['direction']}={t['result']}" for t in sm["tests"]]
                detail = f" [{', '.join(miss_reasons[:4])}]"

            print(f"    {sd}: {status}{detail}")

        all_spread_tests[cfg["label"]] = {"ok": ok_count, "total": total}
        print(f"    → {ok_count}/{total} dates found trades")

    # 4. Deep dive: what does the DB actually have for daily bars by year?
    print(f"\n--- Daily Bars Distribution by Year (SPY options) ---")
    cur = conn.cursor()
    cur.execute("""
        SELECT substr(od.date, 1, 4) as yr, COUNT(*) as bar_count,
               COUNT(DISTINCT od.contract_symbol) as contract_count
        FROM option_daily od
        JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
        WHERE oc.ticker = 'SPY' AND od.date != '0000-00-00'
        GROUP BY yr ORDER BY yr
    """)
    for row in cur.fetchall():
        print(f"    {row[0]}: {row[1]:>10,} bars across {row[2]:>8,} contracts")

    # 5. Check specific: how many UNIQUE dates have bars for SPY options per year?
    print(f"\n--- Trading Days with SPY Option Bars by Year ---")
    cur.execute("""
        SELECT substr(od.date, 1, 4) as yr, COUNT(DISTINCT od.date) as trading_days
        FROM option_daily od
        JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
        WHERE oc.ticker = 'SPY' AND od.date != '0000-00-00' AND od.close IS NOT NULL
        GROUP BY yr ORDER BY yr
    """)
    for row in cur.fetchall():
        print(f"    {row[0]}: {row[1]:>4} trading days with bar data")

    # Save everything
    output = {
        "db_stats": stats,
        "probes": all_probes,
        "spread_test_summary": all_spread_tests,
    }
    output_path = OUTPUT_DIR / "diagnostic.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Full diagnostic saved to {output_path}")

    conn.close()
    print("\n" + "=" * 80)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
