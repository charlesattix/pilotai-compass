#!/usr/bin/env python3
"""
validate_real_vs_heuristic.py — EXP-400 / EXP-401 real-data validation

Runs both experiments through engine/portfolio_backtester.py twice:
  1. Heuristic mode  — Black-Scholes pricing (original results)
  2. Real data mode  — PolygonDataProvider (SQLite cache, offline)

The comparison answers: "Was the heuristic pricing lying to us?"

Key difference: PolygonDataProvider.get_spread_prices() returns valid=False if the
computed OTM% strike doesn't exist in the SQLite cache.  Trades with invalid prices
are SKIPPED.  Trade count going DOWN is the expected real-data result.

Usage:
    PYTHONPATH=. python3 scripts/validate_real_vs_heuristic.py

Output:
    output/real_vs_heuristic.json  — full results
    Prints ASCII comparison table
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from engine.portfolio_backtester import PortfolioBacktester
from strategies import STRATEGY_REGISTRY
from scripts.portfolio_blend import get_strategy_params

TICKERS = ["SPY"]
STARTING_CAPITAL = 100_000
YEARS = list(range(2020, 2026))

OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# ─── Reference numbers from previous heuristic runs ──────────────────────────

KNOWN_400 = {
    "2020": 8.9,  "2021": 101.5, "2022": -1.9,
    "2023": 37.5, "2024": 23.8,  "2025": 26.5,
}
KNOWN_401 = {
    "2020": 24.1, "2021": 107.4, "2022": 8.1,
    "2023": 43.2, "2024": 26.4,  "2025": 35.0,
}


# ─── Builders ────────────────────────────────────────────────────────────────

def _build_exp400_strategies() -> List[Tuple[str, object]]:
    """EXP-400: full champion multi-strategy lineup from champion.json."""
    champion_path = ROOT / "configs" / "champion.json"
    with open(champion_path) as f:
        champ = json.load(f)
    sp = champ.get("strategy_params", {})

    strategies = []
    for name in ["credit_spread", "iron_condor", "momentum_swing", "debit_spread", "zero_dte_spread"]:
        if name in STRATEGY_REGISTRY:
            params = sp.get(name, STRATEGY_REGISTRY[name].get_default_params())
            strategies.append((name, STRATEGY_REGISTRY[name](params)))
    return strategies


def _build_exp401_strategies(cs_risk: float = 0.12, ss_risk: float = 0.03) -> List[Tuple[str, object]]:
    """EXP-401: CS (12%) + straddle_strangle (3%) blend."""
    cs_params = get_strategy_params("credit_spread", risk_override=cs_risk)
    ss_params = get_strategy_params("straddle_strangle", risk_override=ss_risk)

    # Apply regime scales from paper_exp401.yaml
    cs_params.update({
        "regime_scale_bull": 1.0,
        "regime_scale_bear": 0.3,
        "regime_scale_high_vol": 0.3,
        "regime_scale_low_vol": 0.8,
        "regime_scale_crash": 0.0,
    })
    ss_params.update({
        "regime_scale_bull": 1.5,
        "regime_scale_bear": 1.5,
        "regime_scale_high_vol": 2.5,
        "regime_scale_low_vol": 1.0,
        "regime_scale_crash": 0.5,
    })

    return [
        ("credit_spread", STRATEGY_REGISTRY["credit_spread"](cs_params)),
        ("straddle_strangle", STRATEGY_REGISTRY["straddle_strangle"](ss_params)),
    ]


def _make_polygon_dp():
    """Create PolygonDataProvider wrapping SQLite cache (offline, no API calls)."""
    from backtest.data_provider import PolygonDataProvider
    from backtest.historical_data import HistoricalOptionsData
    api_key = os.getenv("POLYGON_API_KEY", "")
    hd = HistoricalOptionsData(api_key, offline_mode=True)
    return PolygonDataProvider(hd)


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_experiment(
    exp_name: str,
    strategies_fn,
    data_provider=None,
) -> Dict:
    """Run one experiment for all years; return per-year results dict."""
    mode = "real (PolygonDP)" if data_provider else "heuristic (BS)"
    print(f"\n  [{exp_name}] {mode}")

    yearly = {}
    for year in YEARS:
        t0 = time.time()
        strategies = strategies_fn()  # fresh instances per year
        bt = PortfolioBacktester(
            strategies=strategies,
            tickers=TICKERS,
            start_date=datetime(year, 1, 1),
            end_date=datetime(year, 12, 31),
            starting_capital=STARTING_CAPITAL,
            max_positions=10,
            max_positions_per_strategy=5,
            data_provider=data_provider,
        )
        raw = bt.run()
        combined = raw.get("combined", raw)
        elapsed = time.time() - t0

        result = {
            "return_pct":   round(combined.get("return_pct", 0), 2),
            "max_drawdown": round(combined.get("max_drawdown", 0), 2),
            "total_trades": combined.get("total_trades", 0),
            "win_rate":     round(combined.get("win_rate", 0), 2),
            "sharpe_ratio": round(combined.get("sharpe_ratio", 0), 3),
        }
        yearly[str(year)] = result
        ret = result["return_pct"]
        trades = result["total_trades"]
        print(f"    {year}: {ret:+.1f}%  {trades} trades  ({elapsed:.0f}s)")

    rets = [yearly[str(y)]["return_pct"] for y in YEARS]
    avg = sum(rets) / len(rets) if rets else 0
    worst_dd = min(yearly[str(y)]["max_drawdown"] for y in YEARS)
    print(f"    ─── avg={avg:+.1f}%  worst_DD={worst_dd:.1f}%")
    return yearly


# ─── Print table ─────────────────────────────────────────────────────────────

def print_comparison(exp_name: str, known: Dict, heuristic: Dict, real: Dict):
    yw = 6
    cw = 10

    header = (
        f"\n  {'Year':{yw}}  "
        f"{'Known BS':>{cw}}  "
        f"{'New BS':>{cw}}  "
        f"{'Real PolygonDP':>{cw+4}}  "
        f"{'BS vs Real':>{cw}}"
    )
    sep = "  " + "─" * (len(header) - 2)

    print(f"\n{'═'*70}")
    print(f"  {exp_name}")
    print(f"{'═'*70}")
    print(header)
    print(sep)

    heur_rets, real_rets, known_rets = [], [], []
    heur_trades, real_trades = [], []

    for year in [str(y) for y in YEARS]:
        k = known.get(year)
        h = heuristic.get(year, {}).get("return_pct", 0)
        r = real.get(year, {}).get("return_pct", 0)
        h_tr = heuristic.get(year, {}).get("total_trades", 0)
        r_tr = real.get(year, {}).get("total_trades", 0)
        diff = r - h

        def fmt(v):
            if v is None: return "  N/A"
            return f"{v:+.1f}%"

        diff_str = f"{diff:+.1f}%" if diff != 0 else "   0.0%"
        flag = " ⬆" if diff > 2 else (" ⬇" if diff < -2 else "")
        known_str = fmt(k)
        print(
            f"  {year:{yw}}  "
            f"{known_str:>{cw}}  "
            f"{fmt(h):>{cw}}  "
            f"{fmt(r):>{cw+4}}  "
            f"{diff_str:>{cw}}{flag}"
            f"  trades: BS={h_tr} DP={r_tr}"
        )

        if k is not None: known_rets.append(k)
        heur_rets.append(h)
        real_rets.append(r)
        heur_trades.append(h_tr)
        real_trades.append(r_tr)

    print(sep)
    k_avg  = sum(known_rets) / len(known_rets) if known_rets else 0
    h_avg  = sum(heur_rets) / len(heur_rets) if heur_rets else 0
    r_avg  = sum(real_rets) / len(real_rets) if real_rets else 0
    diff_avg = r_avg - h_avg
    print(
        f"  {'AVG':{yw}}  "
        f"{fmt(k_avg):>{cw}}  "
        f"{fmt(h_avg):>{cw}}  "
        f"{fmt(r_avg):>{cw+4}}  "
        f"{diff_avg:>+{cw}.1f}%"
        f"  trades: BS={sum(heur_trades)} DP={sum(real_trades)}"
    )

    heur_dd = min(heuristic.get(str(y), {}).get("max_drawdown", 0) for y in YEARS)
    real_dd  = min(real.get(str(y), {}).get("max_drawdown", 0) for y in YEARS)
    print(f"\n  Worst DD:  BS={heur_dd:.1f}%  PolygonDP={real_dd:.1f}%")

    print(f"\n  {'VERDICT':}")
    if abs(diff_avg) <= 3:
        print(f"  ✅ Results are CONSISTENT (avg diff {diff_avg:+.1f}%)")
        print(f"     Heuristic BS pricing was accurate enough.")
    elif diff_avg > 3:
        print(f"  ⚠️  Real data OUTPERFORMS heuristic by {diff_avg:+.1f}% avg")
        print(f"     Possible reason: BS underpriced options in cache strikes.")
    else:
        print(f"  ❌ Real data UNDERPERFORMS heuristic by {diff_avg:+.1f}% avg")
        print(f"     Heuristic was overstating edge.  Trade count reduced by "
              f"{sum(heur_trades)-sum(real_trades)} trades total.")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("""
════════════════════════════════════════════════════════════════════════
  REAL DATA vs HEURISTIC VALIDATION
  EXP-400 (Champion)  |  EXP-401 (CS+SS Blend)
  Data: PolygonDataProvider (SQLite cache, offline mode)
════════════════════════════════════════════════════════════════════════
""")

    dp = _make_polygon_dp()

    # ── EXP-400 ──────────────────────────────────────────────────────────────
    print("═" * 50)
    print("  EXP-400: Champion multi-strategy")
    print("═" * 50)

    exp400_heuristic = run_experiment("EXP-400", _build_exp400_strategies, data_provider=None)
    exp400_real      = run_experiment("EXP-400", _build_exp400_strategies, data_provider=dp)

    # ── EXP-401 ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  EXP-401: CS(12%) + SS(3%) Blend")
    print("═" * 50)

    exp401_heuristic = run_experiment("EXP-401", _build_exp401_strategies, data_provider=None)
    exp401_real      = run_experiment("EXP-401", _build_exp401_strategies, data_provider=dp)

    # ── Print comparison tables ───────────────────────────────────────────────
    print_comparison("EXP-400: Champion", KNOWN_400, exp400_heuristic, exp400_real)
    print_comparison("EXP-401: CS+SS Blend", KNOWN_401, exp401_heuristic, exp401_real)

    # ── Save results ─────────────────────────────────────────────────────────
    output = {
        "generated": datetime.now().isoformat(),
        "exp400": {
            "known":     KNOWN_400,
            "heuristic": exp400_heuristic,
            "real":      exp400_real,
        },
        "exp401": {
            "known":     KNOWN_401,
            "heuristic": exp401_heuristic,
            "real":      exp401_real,
        },
    }
    out_path = OUTPUT / "real_vs_heuristic.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved → {out_path}\n")


if __name__ == "__main__":
    main()
