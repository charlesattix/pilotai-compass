#!/usr/bin/env python3
"""
prob_of_ruin.py — Probability-of-Ruin Calculator for Credit Spread Strategies

Implements two methods:
  1. Bootstrap: resamples actual per-trade PnL from a 6-year backtest
  2. Parametric: uses win_rate + avg_win/loss to build trade distribution

CRITIQUE §7 (MEDIUM priority): Add probability-of-ruin calculation.

Usage:
    python3 scripts/prob_of_ruin.py --config configs/exp_059_friday_ic_risk10.json
    python3 scripts/prob_of_ruin.py --config configs/exp_059_friday_ic_risk10.json --n-sims 50000
    python3 scripts/prob_of_ruin.py --config configs/exp_059_friday_ic_risk10.json --heuristic  # fast mode
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("por")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Ruin = peak-to-trough drawdown exceeds this threshold
RUIN_THRESHOLD_PCT = 50.0  # 50% drawdown is "ruin"


def _collect_trades(params: dict, years: list, use_real_data: bool) -> List[dict]:
    """Run the backtest across all years and collect raw per-trade records."""
    from scripts.run_optimization import run_year

    all_trades = []
    for year in years:
        result = run_year("SPY", year, params, use_real_data=use_real_data)
        trades = result.get("trades", [])
        all_trades.extend(trades)
    return all_trades


def _extract_per_trade_pct(trades: List[dict]) -> List[float]:
    """
    Convert per-trade PnL to fractional return of max_risk.
    return_pct field = (pnl / max_risk) * 100 — already in the record.
    """
    pcts = []
    for t in trades:
        rp = t.get("return_pct")
        if rp is not None:
            pcts.append(rp)
    return pcts


def monte_carlo_ruin(
    trade_returns: List[float],
    n_sims: int,
    horizon: int,
    ruin_pct: float = RUIN_THRESHOLD_PCT,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap Monte Carlo probability-of-ruin.

    Each simulation:
    - Draws `horizon` trades from `trade_returns` (with replacement)
    - Converts return_pct to fractional equity change using risk_per_trade
    - Tracks peak equity; stops if drawdown > ruin_pct

    Returns (ruin_prob, median_final_equity, p5_final_equity)
    Each trade's return_pct is: (pnl / max_risk) * 100.
    Max_risk is risk_per_trade * equity, so:
        equity_change = equity * (risk_per_trade / 100) * (return_pct / 100)
    This is relative to current equity each trade (compounding).
    """
    rng = random.Random(seed)
    ruin_count = 0
    final_equities = []

    # Extract risk_per_trade from params — not available here, use 1.0 for normalized equity
    # The return_pct is already relative to max_risk, so:
    #   if win → pnl ~ +risk_pct * (return_pct / 100) of equity
    #   if loss → pnl ~ +risk_pct * (return_pct / 100) of equity (negative return_pct)
    # We need risk_per_trade to convert to equity fraction.
    # Since we only have return_pct and not abs PnL, we normalize per-simulation.
    # risk_per_trade is baked into the actual dollar amounts.
    # Simplification: simulate equity as a fraction, trade return = return_pct * risk_scale
    # where risk_scale is unknown. Instead, use the ACTUAL equity-relative returns.
    #
    # The backtester's return_pct = pnl / max_risk * 100.
    # max_risk = risk_per_trade% * equity (at entry).
    # So pnl = return_pct/100 * risk_per_trade/100 * equity
    # equity_new = equity * (1 + return_pct/100 * risk_per_trade/100)
    #
    # We pass risk_per_trade separately via a wrapper.
    # For now, this function receives scaled returns directly.

    for _ in range(n_sims):
        equity = 1.0
        peak = 1.0
        ruined = False
        for _ in range(horizon):
            r = rng.choice(trade_returns)  # already equity-relative (see wrapper)
            equity *= (1.0 + r)
            if equity > peak:
                peak = equity
            drawdown_pct = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
            if drawdown_pct >= ruin_pct:
                ruined = True
                break
        if ruined:
            ruin_count += 1
        final_equities.append(equity)

    ruin_prob = ruin_count / n_sims
    final_equities.sort()
    n = len(final_equities)
    median_eq = final_equities[n // 2]
    p5_eq = final_equities[int(n * 0.05)]
    return ruin_prob, median_eq, p5_eq


def _parametric_ruin(
    win_rate: float,
    avg_win_equity_frac: float,
    avg_loss_equity_frac: float,
    n_sims: int,
    horizon: int,
    ruin_pct: float = RUIN_THRESHOLD_PCT,
    seed: int = 99,
) -> Tuple[float, float, float]:
    """
    Parametric Monte Carlo using win_rate and avg win/loss as equity fractions.
    avg_win_equity_frac: e.g. 0.04 means +4% of equity on a win
    avg_loss_equity_frac: e.g. 0.10 means -10% of equity on a loss
    """
    rng = random.Random(seed)
    ruin_count = 0
    final_equities = []

    for _ in range(n_sims):
        equity = 1.0
        peak = 1.0
        ruined = False
        for _ in range(horizon):
            if rng.random() < win_rate:
                equity *= (1.0 + avg_win_equity_frac)
            else:
                equity *= (1.0 - avg_loss_equity_frac)
            if equity > peak:
                peak = equity
            drawdown_pct = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
            if drawdown_pct >= ruin_pct:
                ruined = True
                break
        if ruined:
            ruin_count += 1
        final_equities.append(equity)

    final_equities.sort()
    n = len(final_equities)
    return ruin_count / n_sims, final_equities[n // 2], final_equities[int(n * 0.05)]


def _kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Kelly criterion: f* = p - q/b where b = win/loss ratio."""
    q = 1.0 - win_rate
    if win_loss_ratio == 0:
        return 0.0
    return win_rate - q / win_loss_ratio


def main():
    parser = argparse.ArgumentParser(description="Probability-of-ruin calculator")
    parser.add_argument("--config", required=True, help="Config JSON file")
    parser.add_argument("--n-sims", type=int, default=10000, help="MC simulations (default: 10000)")
    parser.add_argument("--heuristic", action="store_true", help="Fast heuristic mode (no Polygon)")
    parser.add_argument("--years", default="2020,2021,2022,2023,2024,2025")
    args = parser.parse_args()

    years = [int(y) for y in args.years.split(",")]
    with open(args.config) as f:
        params = json.load(f)

    risk_per_trade = params.get("max_risk_per_trade", 5.0)  # % of account

    print(f"""
════════════════════════════════════════════════════════════════════════
  PROBABILITY-OF-RUIN ANALYSIS
  Config  : {args.config}
  Risk    : {risk_per_trade}% per trade
  Sims    : {args.n_sims:,} per horizon
  Ruin    : >{RUIN_THRESHOLD_PCT:.0f}% peak-to-trough drawdown
════════════════════════════════════════════════════════════════════════
""")

    # ── Step 1: Collect trades ───────────────────────────────────────────
    print("Running 6-year backtest to collect trade distribution...")
    use_real = not args.heuristic
    trades = _collect_trades(params, years, use_real_data=use_real)

    if not trades:
        print("ERROR: No trades collected — check config and data mode.")
        sys.exit(1)

    per_trade_returns = _extract_per_trade_pct(trades)

    # Convert return_pct (return as % of max_risk) to equity fraction
    # pnl = return_pct/100 * max_risk = return_pct/100 * risk_per_trade/100 * equity
    # equity_change = return_pct/100 * risk_per_trade/100
    equity_fracs = [r / 100.0 * risk_per_trade / 100.0 for r in per_trade_returns]

    wins = [r for r in per_trade_returns if r > 0]
    losses = [r for r in per_trade_returns if r < 0]
    win_rate = len(wins) / len(per_trade_returns) if per_trade_returns else 0
    avg_win_pct = sum(wins) / len(wins) if wins else 0
    avg_loss_pct = abs(sum(losses) / len(losses)) if losses else 0

    print(f"\n── Trade Statistics ({'real data' if use_real else 'heuristic'} mode) ────────────────────")
    print(f"  Total trades         : {len(trades)}")
    print(f"  Win rate             : {win_rate:.1%}")
    print(f"  Avg win (% max_risk) : +{avg_win_pct:.1f}%")
    print(f"  Avg loss (% max_risk): -{avg_loss_pct:.1f}%")
    print(f"  Win/loss ratio       : {avg_win_pct / avg_loss_pct:.2f}x" if avg_loss_pct > 0 else "  Win/loss ratio: ∞")

    avg_win_equity_frac = avg_win_pct / 100.0 * risk_per_trade / 100.0
    avg_loss_equity_frac = avg_loss_pct / 100.0 * risk_per_trade / 100.0
    kelly = _kelly_fraction(win_rate, avg_win_equity_frac / avg_loss_equity_frac if avg_loss_equity_frac > 0 else 999)

    print(f"\n  Kelly fraction       : {kelly:.1%} of account per trade")
    print(f"  Current risk/Kelly   : {risk_per_trade/100.0/kelly:.1f}x Kelly" if kelly > 0 else "  Kelly: undefined (no losses)")

    # ── Step 2: Bootstrap Monte Carlo POR ───────────────────────────────
    print(f"\n── Bootstrap Monte Carlo (resampling {len(trades)} actual trades) ────────")
    print(f"  {'Horizon':>8}  {'POR':>8}  {'Median final':>14}  {'P5 final':>10}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*10}")

    for horizon in [100, 250, 500, 1000]:
        ruin_prob, med_eq, p5_eq = monte_carlo_ruin(
            equity_fracs, args.n_sims, horizon, seed=42
        )
        print(f"  {horizon:>8}  {ruin_prob:>7.2%}  {med_eq:>+13.1%}  {p5_eq:>+9.1%}")

    # ── Step 3: Parametric POR (sanity check) ───────────────────────────
    if avg_loss_pct > 0:
        print(f"\n── Parametric Model (win={win_rate:.1%}, avg_win={avg_win_equity_frac:.2%}/trade, avg_loss={avg_loss_equity_frac:.2%}/trade) ─")
        print(f"  {'Horizon':>8}  {'POR':>8}  {'Median final':>14}  {'P5 final':>10}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*10}")
        for horizon in [100, 250, 500, 1000]:
            ruin_prob, med_eq, p5_eq = _parametric_ruin(
                win_rate, avg_win_equity_frac, avg_loss_equity_frac,
                args.n_sims, horizon, seed=99
            )
            print(f"  {horizon:>8}  {ruin_prob:>7.2%}  {med_eq:>+13.1%}  {p5_eq:>+9.1%}")

    # ── Step 4: Gap-Risk Stress Test ────────────────────────────────────
    # The backtester doesn't model overnight gap-downs that bypass stops.
    # A gap risk event = trade hits FULL max_loss (100% of max_risk) instead
    # of the stop loss. We inject p_gap probability per trade for realism.
    max_loss_equity_frac = risk_per_trade / 100.0  # full max_loss = 100% of risk cap

    print("\n── Gap-Risk Stress Test (backtester can't see overnight gaps) ──────────")
    print(f"  Max loss if gap occurs: {max_loss_equity_frac:.2%} of equity per trade")
    print(f"  {'Gap rate':>8}  {'POR@1000':>10}  {'Median@1000':>14}  {'P5@1000':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*10}")

    for p_gap in [0.001, 0.005, 0.01, 0.02]:
        # Blend: (1 - p_gap) chance of drawing from historical distribution,
        # p_gap chance of hitting full max_loss
        stressed_fracs = []
        for ef in equity_fracs:
            stressed_fracs.append(ef)  # base case
        # Add synthetic gap events to the pool
        n_gap = max(1, int(len(equity_fracs) * p_gap / (1 - p_gap)))
        for _ in range(n_gap):
            stressed_fracs.append(-max_loss_equity_frac)

        ruin_prob, med_eq, p5_eq = monte_carlo_ruin(
            stressed_fracs, args.n_sims, 1000, seed=77
        )
        print(f"  {p_gap:>7.1%}  {ruin_prob:>9.2%}  {med_eq:>+13.1%}  {p5_eq:>+9.1%}")

    # ── Step 5: Verdict ─────────────────────────────────────────────────
    bootstrap_por_1000, _, _ = monte_carlo_ruin(equity_fracs, args.n_sims, 1000, seed=42)
    # Stress test with 0.5% gap rate (realistic for monthly tail events)
    stressed_fracs_05 = list(equity_fracs)
    n_gap_05 = max(1, int(len(equity_fracs) * 0.005 / 0.995))
    for _ in range(n_gap_05):
        stressed_fracs_05.append(-max_loss_equity_frac)
    stressed_por_1000, _, _ = monte_carlo_ruin(stressed_fracs_05, args.n_sims, 1000, seed=77)

    print(f"""
════════════════════════════════════════════════════════════════════════
  VERDICT (1000-trade horizon)
  P(ruin, historical only)    : {bootstrap_por_1000:.2%}
  P(ruin, +0.5% gap rate)     : {stressed_por_1000:.2%}
  Ruin defined as: >{RUIN_THRESHOLD_PCT:.0f}% peak-to-trough drawdown

  Historical: {"✅ LOW RISK" if bootstrap_por_1000 < 0.01 else "⚠️  MODERATE" if bootstrap_por_1000 < 0.10 else "❌ HIGH RISK"}
  With gaps : {"✅ LOW RISK" if stressed_por_1000 < 0.01 else "⚠️  MODERATE — manage gap risk" if stressed_por_1000 < 0.10 else "❌ HIGH RISK — reduce position size"}
════════════════════════════════════════════════════════════════════════
""")


if __name__ == "__main__":
    main()
