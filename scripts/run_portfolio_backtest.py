#!/usr/bin/env python3
"""
run_portfolio_backtest.py — Multi-underlying COMPASS portfolio backtester

Option C architecture: runs separate per-ticker backtests with allocated capital,
combines results at the reporting layer. Zero changes to backtester.py.

Experiment series: exp_300+

COMPASS Universe Selection (per year):
  - SPY is always included (baseline anchor)
  - Each year: query macro_state.db for sectors ranked top-N by 3M RS
    AND in Leading/Improving RRG quadrant for a majority of weeks
  - Sectors only added when macro_score >= 45 (avoid BEAR_MACRO contraction)
  - Allocation: SPY floor + equal-weight among COMPASS sectors

Regime note: All tickers run with the SAME SPY-based combo regime detector.
  Leading sectors get the same bull/bear regime as SPY — this is correct for
  most years (XLE 2021 = BULL regime), but misses the XLE 2022 alpha (SPY
  BEAR but XLE Leading). Phase 2 will add per-ticker RRG regime override.

Config fields (in addition to standard backtest params from run_optimization.py):
  "portfolio_mode": "compass_top3"  # compass_top2, compass_top3, fixed_list
  "fixed_universe": ["SPY", "QQQ", "XLE"]  # when portfolio_mode="fixed_list"
  "spy_allocation": 0.40             # SPY floor allocation (0.0 = no SPY floor)
  "compass_min_leading_pct": 0.50    # Sector must be Leading/Improving >50% of year

Usage:
    python3 scripts/run_portfolio_backtest.py --config configs/exp_300.json
    python3 scripts/run_portfolio_backtest.py --config configs/exp_300.json --years 2021,2022
    python3 scripts/run_portfolio_backtest.py --portfolio compass_top3 --base-config configs/exp_126_risk8_sl35_ic_neutral_cb30_cd3.json
"""

import argparse
import json
import logging
import sqlite3
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("portfolio_bt")

OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

PORTFOLIO_LB_PATH = OUTPUT / "portfolio_leaderboard.json"
MACRO_DB_PATH = ROOT / "data" / "macro_state.db"

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Sectors eligible for COMPASS universe expansion.
# Ordered by historical alpha priority (highest → lowest based on proposal analysis).
COMPASS_SECTOR_POOL = ["XLE", "QQQ", "SOXX", "XLK", "XLF", "XLI", "XLY", "XLC", "IWM"]

# Minimum macro_score threshold to allow sector expansion.
# Below 45 = BEAR_MACRO: contract to SPY-only (matching get_eligible_underlyings() logic).
MACRO_BEAR_THRESHOLD = 45.0

CACHE_DB_PATH = ROOT / "data" / "options_cache.db"

# ─────────────────────────────────────────────────────────────────────────────
# Real data availability check
# ─────────────────────────────────────────────────────────────────────────────

def _ticker_has_real_data(ticker: str, min_contracts: int = 50) -> bool:
    """Check if ticker has enough options data via Iron Vault coverage report.

    Returns True if count >= min_contracts. Used for informational logging only —
    all backtests now use IronVault regardless.
    """
    if not CACHE_DB_PATH.exists():
        return False
    try:
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(str(CACHE_DB_PATH))
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM option_contracts WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        conn.close()
        count = row[0] if row else 0
        return count >= min_contracts
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# COMPASS universe selection (per-year, from macro_state.db)
# ─────────────────────────────────────────────────────────────────────────────

def get_macro_db() -> Optional[sqlite3.Connection]:
    if not MACRO_DB_PATH.exists():
        logger.warning("macro_state.db not found at %s", MACRO_DB_PATH)
        return None
    conn = sqlite3.connect(str(MACRO_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_year_macro_score(year: int) -> float:
    """Return the average macro_score for the given year. 50.0 on no data."""
    conn = get_macro_db()
    if not conn:
        return 50.0
    try:
        row = conn.execute(
            "SELECT AVG(overall) AS avg_score FROM macro_score "
            "WHERE date >= ? AND date <= ?",
            (f"{year}-01-01", f"{year}-12-31"),
        ).fetchone()
        return float(row["avg_score"]) if row and row["avg_score"] else 50.0
    finally:
        conn.close()


def get_year_sector_rankings(year: int) -> List[Dict]:
    """
    Return per-sector annual summary for a given year from macro_state.db.

    Aggregates all weekly snapshots in the year:
    - avg_rank_3m: lower = stronger relative strength
    - leading_pct: fraction of weeks in Leading or Improving quadrant
    - n_weeks: number of snapshots available

    Returns list of dicts sorted by avg_rank_3m ascending (best sector first).
    """
    conn = get_macro_db()
    if not conn:
        return []
    try:
        rows = conn.execute(
            """
            SELECT
                ticker,
                AVG(rank_3m) AS avg_rank_3m,
                COUNT(*) AS n_weeks,
                SUM(CASE WHEN rrg_quadrant IN ('Leading', 'Improving') THEN 1 ELSE 0 END)
                    AS n_leading_weeks
            FROM sector_rs
            WHERE date >= ? AND date <= ?
            GROUP BY ticker
            ORDER BY avg_rank_3m ASC
            """,
            (f"{year}-01-01", f"{year}-12-31"),
        ).fetchall()
        result = []
        for r in rows:
            n = r["n_weeks"] or 1
            result.append({
                "ticker": r["ticker"],
                "avg_rank_3m": round(r["avg_rank_3m"] or 99, 2),
                "n_weeks": r["n_weeks"],
                "leading_pct": round((r["n_leading_weeks"] or 0) / n, 3),
            })
        return result
    finally:
        conn.close()


def select_compass_universe(year: int, n_sectors: int = 3,
                             min_leading_pct: float = 0.50,
                             spy_allocation: float = 0.40) -> Dict[str, float]:
    """
    Build COMPASS universe allocation for a given year.

    Returns dict: {ticker: allocation_fraction}
    Total allocations always sum to 1.0.

    Logic:
    1. If avg macro_score < BEAR_MACRO_THRESHOLD → SPY only (1.0)
    2. Otherwise: SPY gets spy_allocation floor
    3. Top n_sectors sectors (from COMPASS_SECTOR_POOL) where leading_pct >= min_leading_pct
       split the remaining (1 - spy_allocation) equally
    4. If no COMPASS sectors qualify → SPY only (1.0)
    """
    # Step 1: Bear macro check
    avg_macro = get_year_macro_score(year)
    if avg_macro < MACRO_BEAR_THRESHOLD:
        logger.info("Year %d: BEAR_MACRO (score %.1f < %.0f) → SPY only",
                    year, avg_macro, MACRO_BEAR_THRESHOLD)
        return {"SPY": 1.0}

    # Step 2: Get sector rankings for the year
    rankings = get_year_sector_rankings(year)
    if not rankings:
        logger.warning("Year %d: no sector ranking data → SPY only", year)
        return {"SPY": 1.0}

    # Step 3: Filter to eligible sectors from the pool
    eligible = []
    for r in rankings:
        ticker = r["ticker"]
        if ticker not in COMPASS_SECTOR_POOL:
            continue
        if r["leading_pct"] < min_leading_pct:
            continue
        eligible.append(r)
        if len(eligible) >= n_sectors:
            break

    if not eligible:
        logger.info("Year %d: no sectors meet leading_pct >= %.0f%% → SPY only",
                    year, min_leading_pct * 100)
        return {"SPY": 1.0}

    # Step 4: Build allocation
    sector_share = (1.0 - spy_allocation) / len(eligible)
    alloc: Dict[str, float] = {"SPY": spy_allocation}
    for r in eligible:
        alloc[r["ticker"]] = round(sector_share, 4)

    # Normalize to exactly 1.0 (handle rounding)
    total = sum(alloc.values())
    if abs(total - 1.0) > 0.001:
        alloc["SPY"] = round(alloc["SPY"] + (1.0 - total), 4)

    logger.info("Year %d: COMPASS universe → %s (macro=%.1f)", year, alloc, avg_macro)
    return alloc


def select_fixed_universe(tickers: List[str], spy_allocation: float = 0.40) -> Dict[str, float]:
    """Return fixed allocation for a given list of tickers."""
    if not tickers or tickers == ["SPY"]:
        return {"SPY": 1.0}
    sectors = [t for t in tickers if t != "SPY"]
    alloc = {"SPY": spy_allocation}
    sector_share = (1.0 - spy_allocation) / len(sectors)
    for t in sectors:
        alloc[t] = round(sector_share, 4)
    return alloc


# ─────────────────────────────────────────────────────────────────────────────
# Single-year per-ticker backtest (delegates to run_optimization.run_year)
# ─────────────────────────────────────────────────────────────────────────────

def build_ticker_params(ticker: str, base_params: dict, year_rankings: List[Dict]) -> dict:
    """
    Build per-ticker params with regime override based on COMPASS RRG signal.

    SPY: unchanged (combo regime, direction from base_params)
    Sector ETFs:
      - leading_pct >= 0.50 → direction="bull_put", regime_mode="ma"
        (sector is in strong uptrend; use its own MA as entry filter)
      - leading_pct <= 0.25 → direction="bear_call", regime_mode="ma"
        (sector is in strong downtrend; use its own MA as entry filter)
      - otherwise → direction="both", regime_mode="ma"

    Using regime_mode="ma" instead of "combo" for sector ETFs bypasses the
    SPY-based combo regime detector and lets the sector's own price trend
    determine entry direction. This is critical for cases like XLE in 2022
    (XLE Leading +65% while SPY was BEAR — combo would block bull puts on XLE).
    """
    if ticker == "SPY":
        return base_params  # SPY unchanged

    # Find this ticker in year rankings
    ticker_data = next((r for r in year_rankings if r["ticker"] == ticker), None)
    if ticker_data is None:
        return base_params  # no COMPASS data → use base params unchanged

    leading_pct = ticker_data.get("leading_pct", 0.5)
    ticker_params = dict(base_params)

    if leading_pct >= 0.50:
        # Majority of year in Leading/Improving → sector is in an uptrend → only sell puts
        ticker_params["direction"]    = "bull_put"
        ticker_params["regime_mode"]  = "ma"
    elif leading_pct <= 0.25:
        # Majority of year in Lagging/Weakening → sector is in a downtrend → only sell calls
        ticker_params["direction"]    = "bear_call"
        ticker_params["regime_mode"]  = "ma"
    else:
        # Mixed quadrant history → trade both directions using sector's own MA trend
        ticker_params["direction"]    = "both"
        ticker_params["regime_mode"]  = "ma"

    # Disable ICs for sector ETFs: IC logic (ic_neutral_regime_only) only applies to
    # the combo regime detector. With regime_mode='ma', _ic_enabled stays True from
    # params and ICs would fire unconditionally. Sector ETFs should only trade
    # directional spreads based on their own MA trend.
    ticker_params["iron_condor_enabled"] = False

    return ticker_params


def run_portfolio_year(year: int, universe: Dict[str, float], params: dict,
                       use_real_data: bool = True,  # kept for call-site compatibility; always True
                       year_rankings: Optional[List[Dict]] = None) -> Dict:
    """
    Run one year of the portfolio backtest.

    For each ticker in universe:
      - allocate capital = starting_capital × allocation_fraction
      - apply per-ticker regime override (based on COMPASS leading_pct)
      - run run_year(ticker, year, ticker_params, ...)

    Returns combined result dict with:
      - return_pct: capital-weighted portfolio return (equity-based)
      - per-ticker breakdown
      - combined max_drawdown (allocation-weighted)
      - combined trade stats
    """
    from scripts.run_optimization import run_year

    starting_capital = 100_000
    ticker_results = {}

    for ticker, alloc_frac in universe.items():
        ticker_capital = starting_capital * alloc_frac

        # Per-ticker regime override based on COMPASS signal
        ticker_params = build_ticker_params(ticker, params, year_rankings or [])
        regime_note = ""
        if ticker != "SPY" and ticker_params.get("regime_mode") == "ma":
            regime_note = f" [{ticker_params.get('direction','both')}/ma]"

        has_data = _ticker_has_real_data(ticker)
        data_note = "" if has_data else " [sparse cache — heuristic skips expected]"

        print(f"    {ticker} ({alloc_frac:.0%} = ${ticker_capital:,.0f}){regime_note}{data_note}...",
              end=" ", flush=True)
        t0 = time.time()
        try:
            r = run_year(ticker, year, ticker_params, starting_capital=ticker_capital)
            elapsed = time.time() - t0
            ret = r.get("return_pct", 0)
            trades = r.get("total_trades", 0)
            print(f"{ret:+.1f}%  {trades}T  ({elapsed:.0f}s)")
            ticker_results[ticker] = {
                "return_pct": ret,
                "allocation_frac": alloc_frac,
                "total_trades": trades,
                "win_rate": r.get("win_rate", 0),
                "max_drawdown": r.get("max_drawdown", 0),
                "sharpe_ratio": r.get("sharpe_ratio", 0),
                "starting_capital": ticker_capital,
                "ending_capital": r.get("ending_capital", ticker_capital),
                "monthly_pnl": r.get("monthly_pnl", {}),
                "direction_used": ticker_params.get("direction", "both"),
                "regime_mode_used": ticker_params.get("regime_mode", "combo"),
            }
        except Exception as e:
            logger.exception("Ticker %s year %d failed: %s", ticker, year, e)
            print(f"ERROR: {e}")
            ticker_results[ticker] = {
                "return_pct": 0, "allocation_frac": alloc_frac,
                "total_trades": 0, "win_rate": 0, "max_drawdown": 0,
                "sharpe_ratio": 0, "error": str(e),
            }

    # ── Combine results ────────────────────────────────────────────────────────
    # Blended return: sum(alloc_frac × return_pct) — correct for fractional capital
    blended_return = sum(
        r["return_pct"] * r["allocation_frac"]
        for r in ticker_results.values()
    )
    total_trades = sum(r.get("total_trades", 0) for r in ticker_results.values())
    avg_win_rate = (
        sum(r.get("win_rate", 0) * r.get("total_trades", 0) for r in ticker_results.values())
        / max(total_trades, 1)
    )
    # Weighted max drawdown: each ticker's DD × its allocation fraction
    weighted_dd = sum(
        r.get("max_drawdown", 0) * r["allocation_frac"]
        for r in ticker_results.values()
    )
    # Portfolio-level equity: sum of ending capitals
    total_ending_capital = sum(
        r.get("ending_capital", starting_capital * r["allocation_frac"])
        for r in ticker_results.values()
    )
    portfolio_return_pct = (total_ending_capital - starting_capital) / starting_capital * 100

    return {
        "year": year,
        "universe": universe,
        "return_pct": round(portfolio_return_pct, 2),
        "blended_return_pct": round(blended_return, 2),
        "total_trades": total_trades,
        "win_rate": round(avg_win_rate, 2),
        "max_drawdown": round(weighted_dd, 2),
        "per_ticker": ticker_results,
        "starting_capital": starting_capital,
        "ending_capital": round(total_ending_capital, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio summary & reporting
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_summary(results_by_year: Dict) -> Dict:
    valid = [r for r in results_by_year.values() if "error" not in r]
    rets = [r["return_pct"] for r in valid]
    dds = [r["max_drawdown"] for r in valid]
    trades = [r["total_trades"] for r in valid]
    years_profitable = sum(1 for x in rets if x > 0)
    return {
        "avg_return": round(sum(rets) / len(rets), 2) if rets else 0,
        "min_return": round(min(rets), 2) if rets else 0,
        "max_return": round(max(rets), 2) if rets else 0,
        "worst_dd": round(min(dds), 2) if dds else 0,
        "avg_trades": round(sum(trades) / len(trades)) if trades else 0,
        "years_profitable": years_profitable,
        "years_total": len(rets),
        "consistency_score": round(years_profitable / len(rets), 3) if rets else 0,
    }


def print_portfolio_table(run_id: str, results_by_year: Dict, summary: Dict,
                           spy_baseline: Optional[Dict] = None):
    print()
    print("═" * 80)
    print(f"  Portfolio Run: {run_id}")
    print("─" * 80)
    hdr = f"  {'Year':<8} {'Portfolio':>10} {'Trades':>7} {'WR':>6} {'MaxDD':>8}"
    if spy_baseline:
        hdr += f"  {'SPY Baseline':>13}  {'Alpha':>7}"
    print(hdr)
    print("─" * 80)

    for yr in sorted(results_by_year.keys()):
        r = results_by_year[yr]
        if "error" in r:
            print(f"  {yr:<8} ERROR")
            continue
        ret = r["return_pct"]
        trades = r["total_trades"]
        wr = r["win_rate"]
        dd = r["max_drawdown"]
        flag = " ✓" if ret > 0 else " ✗"
        universe_str = "+".join(r.get("universe", {}).keys())

        row = f"  {yr:<8} {ret:>+9.1f}%  {trades:>6}  {wr:>5.1f}%  {dd:>7.1f}%{flag}  [{universe_str}]"
        if spy_baseline:
            spy_ret = spy_baseline.get(yr, {}).get("return_pct", 0)
            alpha = ret - spy_ret
            row += f"  SPY={spy_ret:>+6.1f}%  α={alpha:>+5.1f}%"
        print(row)

        # Per-ticker breakdown (indented)
        for ticker, tr in r.get("per_ticker", {}).items():
            t_ret = tr.get("return_pct", 0)
            alloc = tr.get("allocation_frac", 0)
            t_trades = tr.get("total_trades", 0)
            print(f"    {'':4s} {ticker:<6s} {alloc:.0%} → {t_ret:>+7.1f}%  ({t_trades}T)")

    print("─" * 80)
    print(f"  {'AVG':>8} {summary['avg_return']:>+9.1f}%  {summary['avg_trades']:>6}  "
          f"{'—':>6}   {summary['worst_dd']:>7.1f}%")
    print(f"  Profitable years: {summary['years_profitable']}/{summary['years_total']}  "
          f"Consistency: {summary['consistency_score']:.0%}")
    print("═" * 80)
    print()


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return default


def append_to_portfolio_leaderboard(entry: dict):
    lb = _load_json(PORTFOLIO_LB_PATH, [])
    lb.append(entry)
    lb.sort(key=lambda x: x.get("summary", {}).get("avg_return", 0), reverse=True)
    PORTFOLIO_LB_PATH.write_text(json.dumps(lb, indent=2, default=str))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-underlying COMPASS portfolio backtester (exp_300 series)"
    )
    parser.add_argument("--config",      help="JSON config file with params + portfolio settings")
    parser.add_argument("--base-config", help="JSON config with base strategy params (exp_126, etc.)")
    parser.add_argument("--portfolio",   default="compass_top3",
                        choices=["compass_top2", "compass_top3", "compass_top5",
                                 "spy_qqq", "spy_xle", "spy_qqq_xle", "fixed_list"],
                        help="Portfolio mode (overridden by --config portfolio_mode)")
    parser.add_argument("--universe",    help="Comma-separated fixed universe, e.g. SPY,QQQ,XLE")
    parser.add_argument("--years",       help="Comma-separated years, e.g. 2021,2022")
    parser.add_argument("--spy-alloc",   type=float, default=0.40, help="SPY allocation floor (0.40)")
    parser.add_argument("--run-id",      help="Override auto run ID")
    parser.add_argument("--note",        default="", help="Experiment note")
    args = parser.parse_args()

    # ── Load params ────────────────────────────────────────────────────────────
    from scripts.run_optimization import BASELINE_PARAMS
    params = dict(BASELINE_PARAMS)

    if args.base_config:
        with open(args.base_config) as f:
            params.update(json.load(f))

    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        # Portfolio-specific keys consumed here; rest goes to params
        portfolio_mode   = cfg.pop("portfolio_mode", args.portfolio)
        fixed_universe   = cfg.pop("fixed_universe", None)
        spy_allocation   = cfg.pop("spy_allocation", args.spy_alloc)
        min_leading_pct  = cfg.pop("compass_min_leading_pct", 0.50)
        params.update(cfg)
    else:
        portfolio_mode   = args.portfolio
        fixed_universe   = args.universe.split(",") if args.universe else None
        spy_allocation   = args.spy_alloc
        min_leading_pct  = 0.50

    # n_sectors from portfolio_mode
    n_sectors_map = {"compass_top2": 2, "compass_top3": 3, "compass_top5": 5}
    n_sectors = n_sectors_map.get(portfolio_mode, 3)

    years = [int(y.strip()) for y in args.years.split(",")] if args.years else YEARS
    run_id = args.run_id or f"port_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    print()
    print("═" * 80)
    print("  COMPASS PORTFOLIO BACKTEST — Multi-Underlying")
    print(f"  Run ID       : {run_id}")
    print(f"  Mode         : {portfolio_mode}")
    print(f"  Years        : {years}")
    print(f"  SPY floor    : {spy_allocation:.0%}")
    print(f"  Leading pct  : {min_leading_pct:.0%} min to qualify")
    print(f"  Data mode    : real data (Iron Vault)")
    print(f"  Note         : {args.note or '(none)'}")
    print("═" * 80)

    # Show COMPASS universe selection for each year
    print("\nCOMPASS Universe Selection:")
    print(f"  {'Year':<6} {'Macro':>7} {'Universe':<50} {'Allocation'}")
    print("  " + "-" * 72)
    year_universes = {}
    for year in sorted(years):
        if fixed_universe:
            uni = select_fixed_universe(fixed_universe, spy_allocation)
        elif portfolio_mode in ("spy_qqq",):
            uni = select_fixed_universe(["SPY", "QQQ"], spy_allocation)
        elif portfolio_mode == "spy_xle":
            uni = select_fixed_universe(["SPY", "XLE"], spy_allocation)
        elif portfolio_mode == "spy_qqq_xle":
            uni = select_fixed_universe(["SPY", "QQQ", "XLE"], spy_allocation)
        else:
            uni = select_compass_universe(year, n_sectors, min_leading_pct, spy_allocation)
        year_universes[year] = uni
        macro_score = get_year_macro_score(year)
        uni_str = " + ".join(f"{t}({v:.0%})" for t, v in uni.items())
        print(f"  {year:<6} {macro_score:>6.1f}  {uni_str}")

    print()
    print("Running portfolio backtests...")

    results_by_year = {}
    t_total = time.time()

    for year in years:
        universe = year_universes[year]
        year_rankings = get_year_sector_rankings(year)
        print(f"\n  Year {year} | Universe: {list(universe.keys())}")
        r = run_portfolio_year(year, universe, params, year_rankings=year_rankings)
        results_by_year[str(year)] = r
        print(f"  → Portfolio return: {r['return_pct']:+.1f}%  "
              f"Trades: {r['total_trades']}  DD: {r['max_drawdown']:.1f}%")

    elapsed = time.time() - t_total
    summary = compute_portfolio_summary(results_by_year)

    print_portfolio_table(run_id, results_by_year, summary)

    # Save to portfolio leaderboard
    entry = {
        "run_id": run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "portfolio_mode": portfolio_mode,
        "params": params,
        "spy_allocation": spy_allocation,
        "n_sectors": n_sectors,
        "min_leading_pct": min_leading_pct,
        "years_run": years,
        "year_universes": {str(y): u for y, u in year_universes.items()},
        "results": {yr: {k: v for k, v in r.items() if k != "per_ticker"}
                    for yr, r in results_by_year.items()},
        "results_per_ticker": {yr: r.get("per_ticker", {})
                               for yr, r in results_by_year.items()},
        "summary": summary,
        "elapsed_sec": round(elapsed),
        "note": args.note,
    }
    append_to_portfolio_leaderboard(entry)
    print(f"Results saved to {PORTFOLIO_LB_PATH}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
