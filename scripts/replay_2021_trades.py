#!/usr/bin/env python3
"""
replay_2021_trades.py — P6: 2021 100% Win Rate Spot-Check

Purpose:
    Carlos's concern: "2021 100% win rate (44 trades) needs to be VERIFIED."
    Run exp_059 for 2021 only, capture all trades, then fetch raw Polygon data
    for a sample of trades and verify:
    1. Entry credit is realistic vs Polygon chain mid-price
    2. Stop-loss COULD have triggered given actual price moves
    3. Exit prices are consistent with options data

Usage:
    python3 scripts/replay_2021_trades.py
    python3 scripts/replay_2021_trades.py --sample 10
    python3 scripts/replay_2021_trades.py --config configs/exp_059_friday_ic_risk10.json

Output:
    Prints trade-by-trade comparison: backtester record vs Polygon daily bar
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("replay")

DEFAULT_CONFIG = ROOT / "configs" / "exp_059_friday_ic_risk10.json"


def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def _spot_check_trade(trade: dict, hd, n: int, total: int):
    """Fetch Polygon daily bars for the short leg and compare to backtester record."""
    ticker = trade.get("ticker", "SPY")
    entry_date = trade.get("entry_date", "")
    exit_date = trade.get("exit_date", "")
    expiration = trade.get("expiration", "")
    short_strike = trade.get("short_strike", "")
    opt_type = trade.get("option_type", "P")
    credit = trade.get("credit", 0)
    pnl = trade.get("pnl", 0)
    exit_reason = trade.get("exit_reason", "?")
    contracts = trade.get("contracts", 1)

    # Build contract symbol for the short leg
    try:
        exp_dt = datetime.strptime(str(expiration)[:10], "%Y-%m-%d")
        exp_str_short = exp_dt.strftime("%y%m%d")
        strike_int = int(float(short_strike) * 1000)
        contract_symbol = f"O:{ticker}{exp_str_short}{opt_type}{strike_int:08d}"
    except Exception:
        contract_symbol = "UNKNOWN"

    print(f"\n  ─── Trade #{n}/{total} ─────────────────────────────────────────")
    print(f"  Entry:       {str(entry_date)[:10]}  →  Exit: {str(exit_date)[:10]}")
    print(f"  Type:        {trade.get('type', '?')}")
    print(f"  Strike:      {short_strike} {opt_type}  exp {str(expiration)[:10]}")
    print(f"  Credit:      ${credit:.4f}/share ({credit*100:.2f}¢)")
    print(f"  Contracts:   {contracts}")
    print(f"  Exit reason: {exit_reason}")
    print(f"  PnL:         ${pnl:+,.2f}")
    print(f"  Contract:    {contract_symbol}")

    # Fetch daily bars for the short leg from Polygon cache
    if hd is None:
        print("  Polygon:     [no HD — skipping raw data check]")
        return

    if contract_symbol == "UNKNOWN":
        print("  Polygon:     [no expiration in trade record — skipping raw data check]")
        return

    entry_str = str(entry_date)[:10]
    str(exit_date)[:10]

    # Get entry price from cache
    try:
        entry_prices = hd.get_spread_prices(
            ticker,
            exp_dt,
            float(short_strike),
            float(short_strike) - (5 if opt_type == "P" else -5),
            opt_type,
            entry_str,
        )
        if entry_prices:
            print(f"  Entry check: spread_value={entry_prices.get('spread_value', 'N/A'):.4f}  "
                  f"(backtester credit={credit:.4f})")
        else:
            print("  Entry check: no Polygon data for this date")
    except Exception as e:
        print(f"  Entry check: ERROR — {e}")

    # Get expiration price (did it expire worthless?)
    try:
        exp_str = str(expiration)[:10]
        exp_prices = hd.get_spread_prices(
            ticker,
            exp_dt,
            float(short_strike),
            float(short_strike) - (5 if opt_type == "P" else -5),
            opt_type,
            exp_str,
        )
        if exp_prices:
            final_val = exp_prices.get("spread_value", "N/A")
            print(f"  Expiry check: final_spread_value={final_val}  "
                  f"(should be ≈0 for full profit, credit was {credit:.4f})")
        else:
            print("  Expiry check: no Polygon data for expiration date")
    except Exception as e:
        print(f"  Expiry check: ERROR — {e}")


def main():
    parser = argparse.ArgumentParser(description="2021 100% WR spot-check (P6)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Config JSON file")
    parser.add_argument("--sample", type=int, default=10, help="Number of trades to spot-check")
    parser.add_argument("--all", action="store_true", help="Show all trades (not just sample)")
    parser.add_argument("--heuristic", action="store_true", help="No Polygon (skip raw data check)")
    args = parser.parse_args()

    params = _load_config(args.config)

    print(f"""
════════════════════════════════════════════════════════════════════════
  P6: 2021 100% WIN RATE SPOT-CHECK
  Config: {args.config}
════════════════════════════════════════════════════════════════════════
""")

    from scripts.run_optimization import run_year
    print("  Running 2021 backtest...")
    result = run_year("SPY", 2021, params, use_real_data=not args.heuristic)
    if not result:
        print("  ERROR: No results returned for 2021")
        return

    trades = result.get("trades", [])
    total_trades = result.get("total_trades", len(trades))
    win_rate     = result.get("win_rate", 0)
    total_return = result.get("return_pct", 0)
    max_dd       = result.get("max_drawdown", 0)
    friday_fb    = result.get("friday_fallback_count", "N/A")

    print(f"""
  ── 2021 Summary ─────────────────────────────────────────────────────
  Trades:        {total_trades}
  Win rate:      {win_rate:.1f}%
  Return:        {total_return:+.1f}%
  Max DD:        {max_dd:.1f}%
  Friday fallbacks: {friday_fb}
  ─────────────────────────────────────────────────────────────────────
""")

    if not trades:
        print("  No trades to replay.")
        return

    # Break down by type
    bull_puts    = [t for t in trades if t.get("type") == "bull_put_spread"]
    bear_calls   = [t for t in trades if t.get("type") == "bear_call_spread"]
    iron_condors = [t for t in trades if t.get("type") == "iron_condor"]

    print(f"  Bull puts:    {len(bull_puts)}")
    print(f"  Bear calls:   {len(bear_calls)}")
    print(f"  Iron condors: {len(iron_condors)}")
    print("  (Note: 2021 was a bull year — expecting mostly bull puts + ICs)")

    # Exit reason breakdown
    reasons = {}
    for t in trades:
        r = t.get("exit_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1
    print(f"\n  Exit reasons: {reasons}")
    print(f"  (Stop losses = {reasons.get('stop_loss', 0)} — should be 0 in 100% WR)")

    # Spot-check a sample
    import random
    random.seed(42)
    if args.all:
        sample = trades
    else:
        n_sample = min(args.sample, len(trades))
        sample = random.sample(trades, n_sample)

    print(f"\n  ── Spot-checking {len(sample)} trades against Polygon data ──────")

    hd = None
    if not args.heuristic:
        try:
            from backtest.historical_data import HistoricalOptionsData
            polygon_api_key = os.getenv("POLYGON_API_KEY", "")
            hd = HistoricalOptionsData(polygon_api_key)
        except Exception as e:
            print(f"  WARNING: Could not init Polygon: {e} — skipping raw data check")

    for i, trade in enumerate(sample, 1):
        _spot_check_trade(trade, hd, i, len(sample))

    print("""
════════════════════════════════════════════════════════════════════════
  VERDICT:
  - If ALL exit reasons are 'expiration' or 'profit_target' → legit WR
  - If stop_loss count > 0 but WR=100% → BUG (stops never triggered?)
  - If Polygon entry prices differ significantly → pricing model issue
════════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
