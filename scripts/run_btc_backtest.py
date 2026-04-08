#!/usr/bin/env python3
"""
Run BTC put credit spread backtest using real Deribit historical option data.

Usage:
    python3 scripts/run_btc_backtest.py
    python3 scripts/run_btc_backtest.py --years 2021 2022 2023
    python3 scripts/run_btc_backtest.py --otm-pct 0.07 --risk 0.05
    python3 scripts/run_btc_backtest.py --risk 0.08 --stop 3.0 --target 0.50
    python3 scripts/run_btc_backtest.py --compound false --years 2020 2021 2022 2023 2024
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtest.btc_credit_spread_backtester import BTCCreditSpreadBacktester, DEFAULT_CONFIG


def fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.1f}%"


def main():
    parser = argparse.ArgumentParser(description="BTC put credit spread backtest (Deribit real data)")
    parser.add_argument("--years",     type=int, nargs="+", default=list(range(2020, 2025)),
                        help="Years to backtest (default: 2020–2024)")
    parser.add_argument("--capital",   type=float, default=100_000, help="Starting capital USD")
    parser.add_argument("--otm-pct",   type=float, default=0.05,  help="Short put OTM %  (default: 0.05)")
    parser.add_argument("--width-pct", type=float, default=0.05,  help="Spread width % of spot (default: 0.05)")
    parser.add_argument("--min-credit",type=float, default=8.0,   help="Min credit %% of spread (default: 8.0)")
    parser.add_argument("--stop",      type=float, default=2.5,   help="Stop loss multiplier (default: 2.5)")
    parser.add_argument("--target",    type=float, default=0.50,  help="Profit target fraction (default: 0.50)")
    parser.add_argument("--dte",       type=int,   default=35,    help="Target DTE at entry (default: 35)")
    parser.add_argument("--risk",      type=float, default=0.05,  help="Risk %% of equity per trade (default: 0.05)")
    parser.add_argument("--max-contracts", type=int, default=20,  help="Max contracts per trade")
    parser.add_argument("--compound",  default="true",            help="Compound equity (default: true)")
    parser.add_argument("--per-year",  action="store_true",       help="Run each year independently")
    parser.add_argument("--save",      type=str, default=None,    help="Save results to JSON file")
    args = parser.parse_args()

    config = {
        "starting_capital":    args.capital,
        "otm_pct":             args.otm_pct,
        "spread_width_pct":    args.width_pct,
        "min_credit_pct":      args.min_credit,
        "stop_loss_multiplier": args.stop,
        "profit_target_pct":   args.target,
        "dte_target":          args.dte,
        "risk_per_trade_pct":  args.risk,
        "max_contracts":       args.max_contracts,
        "compound":            args.compound.lower() not in ("false", "0", "no"),
    }

    print("\n=== BTC Put Credit Spread Backtest (Deribit Real Data) ===")
    print(f"Years:          {args.years}")
    print(f"OTM pct:        {config['otm_pct']*100:.0f}%")
    print(f"Spread width:   {config['spread_width_pct']*100:.0f}% of spot")
    print(f"Min credit:     {config['min_credit_pct']:.0f}% of spread")
    print(f"Stop loss:      {config['stop_loss_multiplier']}× credit")
    print(f"Profit target:  {config['profit_target_pct']*100:.0f}% of credit captured")
    print(f"DTE target:     {config['dte_target']}")
    print(f"Risk per trade: {config['risk_per_trade_pct']*100:.1f}% of equity")
    print(f"Compound:       {config['compound']}")
    print(f"Starting cap:   ${config['starting_capital']:,.0f}")

    bt = BTCCreditSpreadBacktester(config=config)

    if args.per_year:
        # Each year independently (shows which years are profitable in isolation)
        print("\n--- Per-Year Results (independent, not compounded across years) ---")
        print(f"{'Year':<6} {'Return':>8} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8} {'PnL':>12}")
        print("-" * 55)
        summary_rets = []
        summary_dds  = []
        for year in args.years:
            yr_bt = BTCCreditSpreadBacktester(config=config)
            res = yr_bt.run([year])
            yr = res["year_stats"].get(year, {})
            ret = yr.get("return_pct", 0.0)
            dd  = yr.get("max_drawdown", 0.0)
            n   = yr.get("trade_count", 0)
            wr  = yr.get("win_rate", 0.0)
            pnl = yr.get("pnl_usd", 0.0)
            summary_rets.append(ret)
            summary_dds.append(dd)
            print(f"{year:<6} {fmt_pct(ret):>8} {fmt_pct(dd):>8} {n:>7} {wr:>7.1f}% {pnl:>+12,.0f}")
        print("-" * 55)
        avg_ret = sum(summary_rets) / len(summary_rets) if summary_rets else 0
        worst_dd = min(summary_dds) if summary_dds else 0
        n_pos = sum(1 for r in summary_rets if r > 0)
        print(f"{'Avg':<6} {fmt_pct(avg_ret):>8} {fmt_pct(worst_dd):>8} {'':>7} {'':>8} {'':>12}")
        print(f"\nProfitable years: {n_pos}/{len(args.years)}")
        return

    # Full multi-year run (capital carries over between years)
    results = bt.run(args.years)

    print(f"\n--- Results ({min(args.years)}–{max(args.years)}) ---")
    print(f"\n{'Year':<6} {'Return':>8} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8} {'PnL USD':>12}")
    print("-" * 55)
    yr_rets = []
    yr_dds  = []
    for year in args.years:
        ys = results["year_stats"].get(year, {})
        ret = ys.get("return_pct", 0.0)
        dd  = ys.get("max_drawdown", 0.0)
        n   = ys.get("trade_count", 0)
        wr  = ys.get("win_rate", 0.0)
        pnl = ys.get("pnl_usd", 0.0)
        yr_rets.append(ret)
        yr_dds.append(dd)
        print(f"{year:<6} {fmt_pct(ret):>8} {fmt_pct(dd):>8} {n:>7} {wr:>7.1f}% {pnl:>+12,.0f}")

    print("-" * 55)
    avg_ret  = sum(yr_rets) / len(yr_rets) if yr_rets else 0
    worst_dd = min(yr_dds) if yr_dds else 0
    n_pos = sum(1 for r in yr_rets if r > 0)
    print(f"{'Avg':<6} {fmt_pct(avg_ret):>8} {fmt_pct(worst_dd):>8} {'':>7}")

    print(f"\nOverall:")
    print(f"  Total return:     {fmt_pct(results['return_pct'])}")
    print(f"  Max drawdown:     {fmt_pct(results['max_drawdown'])}")
    print(f"  Total trades:     {results['total_trades']}")
    print(f"  Win rate:         {results['win_rate']:.1f}%")
    print(f"  Profit factor:    {results['profit_factor']:.2f}")
    print(f"  Avg win:         ${results['avg_win']:,.0f}")
    print(f"  Avg loss:        ${results['avg_loss']:,.0f}")
    print(f"  Profitable years: {n_pos}/{len(args.years)}")
    print(f"  Starting capital: ${results['starting_capital']:,.0f}")
    print(f"  Ending capital:   ${results['ending_capital']:,.0f}")
    if results.get("ruin_triggered"):
        print("  ⚠️  RUIN TRIGGERED")

    # Exit reason breakdown
    trades = results["trades"]
    if trades:
        reasons = {}
        for t in trades:
            reasons[t["exit_reason"]] = reasons.get(t["exit_reason"], 0) + 1
        print(f"\nExit reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            pct = count / len(trades) * 100
            print(f"  {reason:<20} {count:>3}  ({pct:.0f}%)")

    if args.save:
        out = {k: v for k, v in results.items() if k not in ("equity_curve", "trades")}
        out["trades"] = trades
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nResults saved to: {args.save}")


if __name__ == "__main__":
    main()
