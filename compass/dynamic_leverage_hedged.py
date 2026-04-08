"""
compass/dynamic_leverage_hedged.py — Dynamic Leverage + Crisis Alpha Hedge

Hypothesis: Dynamic leverage v3 got 169% CAGR @ 13.5% DD. Adding a
Crisis Alpha v4 hedge overlay should reduce DD further, allowing us
to INCREASE the leverage cap (1x-6x or 1x-7x) while staying under
15% DD.

Tests:
  1. Dynamic 1x-5x + 10% hedge (baseline extension)
  2. Dynamic 1x-6x + 10% hedge
  3. Dynamic 1x-7x + 10% hedge
  4. Dynamic 1x-5x + 15% hedge
  5. Compare all to static 2x + 10% hedge (current best balanced config)

All signals use t-1 lagged values (no look-ahead).
Rule Zero: all inputs from real Yahoo Finance data.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.dynamic_leverage_v3 import (
    fetch_yahoo_close, vix_band_leverage, term_structure_adjustment,
    drawdown_brake, regime_classify, regime_adjustment,
    leverage_to_tier, DailyState,
)
from compass.crisis_alpha_v3 import load_universe_v3
from compass.crisis_alpha_v4 import ConfigV4, backtest_v4
from compass.exp1780_exp1220_integration import build_exp1220_daily_returns

TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

V4_PRODUCTION = {
    "lookback_preset": "v2_round",
    "vol_target": 0.06,
    "leverage": 1.5,
    "dd_brake_threshold": 0.05,
    "dd_brake_zone": 0.05,
    "max_weight": 0.20,
    "require_confirmation": True,
}

# Leverage tier maps: list of (vix_band, tier_values) where tier_values
# defines the mapping from tier index → leverage.
LEVERAGE_CAPS = {
    "1x-5x": [1.0, 2.0, 3.0, 5.0],
    "1x-6x": [1.0, 2.0, 4.0, 6.0],
    "1x-7x": [1.0, 2.0, 4.0, 7.0],
}


def tier_to_leverage_custom(tier_idx: int, tiers: List[float]) -> float:
    """Convert tier index to leverage from a custom tier list."""
    return tiers[max(0, min(len(tiers) - 1, tier_idx))]


def vix_band_leverage_custom(vix_value: float, tiers: List[float]) -> float:
    """Map VIX level to base leverage using custom tier maximum."""
    max_lev = tiers[-1]
    mid_high = tiers[-2]
    mid_low = tiers[1]
    min_lev = tiers[0]

    if vix_value < 15:
        return max_lev
    if vix_value < 25:
        return mid_high
    if vix_value < 35:
        return mid_low
    return min_lev


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic leverage with custom cap
# ═══════════════════════════════════════════════════════════════════════════

def run_dynamic_backtest_custom(
    base_returns: pd.Series,
    vix: pd.Series,
    vix3m: pd.Series,
    spy: pd.Series,
    leverage_tiers: List[float],
) -> Tuple[pd.Series, List[DailyState]]:
    """Dynamic leverage backtest with a custom tier list (allows 1x-6x, 1x-7x)."""
    common = (base_returns.index.intersection(vix.index)
              .intersection(vix3m.index).intersection(spy.index))
    common = common.sort_values()
    rets = base_returns.reindex(common).fillna(0)
    vix_s = vix.reindex(common).ffill().bfill()
    vix3m_s = vix3m.reindex(common).ffill().bfill()
    spy_s = spy.reindex(common).ffill().bfill()
    spy_ma200 = spy_s.rolling(200, min_periods=50).mean()

    # t-1 lagged signals (rule zero: no look-ahead)
    vix_lagged = vix_s.shift(1).bfill()
    vix3m_lagged = vix3m_s.shift(1).bfill()
    spy_lagged = spy_s.shift(1).bfill()
    spy_ma200_lagged = spy_ma200.shift(1).bfill()

    min_lev = leverage_tiers[0]
    max_lev = leverage_tiers[-1]

    equity = 1.0
    peak = equity
    dynamic_rets = []
    states = []
    pos_day_history = []

    for i, dt in enumerate(common):
        v = float(vix_lagged.iloc[i])
        v3m = float(vix3m_lagged.iloc[i])
        ratio = v / max(v3m, 1.0)
        spy_now = float(spy_lagged.iloc[i])
        spy_ma = float(spy_ma200_lagged.iloc[i])

        dd = max(0, (peak - equity) / peak) if peak > 0 else 0

        if len(pos_day_history) >= 50:
            pos_day_rate = sum(pos_day_history[-50:]) / 50
        elif len(pos_day_history) > 0:
            pos_day_rate = sum(pos_day_history) / len(pos_day_history)
        else:
            pos_day_rate = 0.6

        # Base leverage from VIX bands (using custom tier max)
        base_lev = vix_band_leverage_custom(v, leverage_tiers)
        # Find base tier index in custom tiers
        base_tier = min(range(len(leverage_tiers)),
                        key=lambda j: abs(leverage_tiers[j] - base_lev))

        # Apply tier adjustments
        ts_adj = term_structure_adjustment(ratio)
        regime = regime_classify(spy_now, spy_ma, v)
        regime_adj = regime_adjustment(regime)
        wr_adj = -1 if pos_day_rate < 0.50 else 0

        adjusted_tier = base_tier + ts_adj + regime_adj + wr_adj
        adjusted_lev = tier_to_leverage_custom(adjusted_tier, leverage_tiers)

        # DD brake
        final_lev = drawdown_brake(dd, adjusted_lev)
        final_lev = max(min_lev, min(max_lev, final_lev))

        today_ret = float(rets.iloc[i]) * final_lev
        equity *= (1 + today_ret)
        if equity > peak:
            peak = equity

        pos_day_history.append(1 if today_ret > 0 else 0)
        if len(pos_day_history) > 100:
            pos_day_history.pop(0)

        dynamic_rets.append(today_ret)
        states.append(DailyState(
            date=dt, vix_used=v, vix_ratio_used=ratio,
            dd=dd, pos_day_rate=pos_day_rate, regime=regime,
            base_leverage=base_lev, final_leverage=final_lev,
            daily_return=today_ret,
        ))

    return pd.Series(dynamic_rets, index=common, name="dynamic"), states


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def corrected_sharpe(rets: np.ndarray, rf: float = 0.045) -> float:
    if len(rets) < 2:
        return 0.0
    r = np.asarray(rets, dtype=np.float64)
    rf_d = rf / TRADING_DAYS
    excess = float(np.mean(r)) - rf_d
    std = float(np.std(r, ddof=0))
    if std < 1e-12:
        return 0.0
    return excess / std * math.sqrt(TRADING_DAYS)


def compute_metrics(rets: np.ndarray) -> Dict[str, float]:
    if len(rets) < 2:
        return {"cagr": 0, "sharpe": 0, "dd": 0, "calmar": 0, "sortino": 0, "vol": 0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else -1
    sharpe = corrected_sharpe(rets)
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = rets[rets < 0]
    ds = float(np.std(down, ddof=0)) if len(down) > 1 else float(np.std(rets, ddof=0))
    sortino = float(np.mean(rets)) / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0
    vol = float(np.std(rets, ddof=0)) * math.sqrt(TRADING_DAYS)
    return {
        "cagr": float(cagr), "sharpe": float(sharpe), "dd": float(dd),
        "calmar": float(calmar), "sortino": float(sortino), "vol": float(vol),
    }


def yearly_breakdown(rets: pd.Series) -> Dict[int, Dict[str, float]]:
    yearly = {}
    for yr in sorted(set(rets.index.year)):
        mask = rets.index.year == yr
        yr_rets = rets[mask].values
        if len(yr_rets) < 5:
            continue
        m = compute_metrics(yr_rets)
        yearly[int(yr)] = {
            "cagr": round(m["cagr"] * 100, 2),
            "sharpe": round(m["sharpe"], 2),
            "dd": round(m["dd"] * 100, 2),
            "vol": round(m["vol"] * 100, 2),
        }
    return yearly


def crisis_dd(rets: pd.Series, start: str, end: str) -> float:
    mask = (rets.index >= start) & (rets.index <= end)
    sub = rets[mask].values
    if len(sub) < 2:
        return 0.0
    eq = np.cumprod(1 + sub)
    hwm = np.maximum.accumulate(eq)
    return float((1 - eq / hwm).max())


# ═══════════════════════════════════════════════════════════════════════════
# Combine dynamic core with hedge overlay
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ComboResult:
    name: str
    leverage_cap: str           # "1x-5x" / "1x-6x" / "1x-7x" / "static 2x"
    hedge_pct: float
    cagr: float
    sharpe: float
    max_dd: float
    covid_dd: float
    bear2022_dd: float
    sortino: float
    calmar: float
    vol: float
    avg_leverage: float
    yearly: Dict[int, Dict[str, float]]
    daily_returns: pd.Series = field(default=None, repr=False)


def build_combo(
    core_rets: pd.Series,
    hedge_rets: pd.Series,
    name: str,
    leverage_cap: str,
    hedge_pct: float,
    avg_leverage: float = 0.0,
) -> ComboResult:
    """Combine a dynamic (or static) core with a hedge allocation."""
    common = core_rets.index.intersection(hedge_rets.index)
    core = core_rets.loc[common]
    hedge = hedge_rets.loc[common]

    combined = core * (1.0 - hedge_pct) + hedge * hedge_pct

    m = compute_metrics(combined.values)
    cvd = crisis_dd(combined, "2020-02-19", "2020-03-23")
    bear = crisis_dd(combined, "2022-01-03", "2022-10-12")

    return ComboResult(
        name=name,
        leverage_cap=leverage_cap,
        hedge_pct=hedge_pct,
        cagr=round(m["cagr"] * 100, 2),
        sharpe=round(m["sharpe"], 2),
        max_dd=round(m["dd"] * 100, 2),
        covid_dd=round(cvd * 100, 2),
        bear2022_dd=round(bear * 100, 2),
        sortino=round(m["sortino"], 2),
        calmar=round(m["calmar"], 2),
        vol=round(m["vol"] * 100, 2),
        avg_leverage=round(avg_leverage, 2),
        yearly=yearly_breakdown(combined),
        daily_returns=combined,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline() -> Dict:
    print("[1/6] Loading real Yahoo data...")
    # For dynamic leverage core we need VIX/VIX3M/SPY + EXP-1220 base
    start = "2014-01-01"
    end = "2026-01-01"

    # Base returns from EXP-1220 proxy (calibrated on real SPY)
    # Use the same source the crisis_alpha v4 uses — ensures alignment
    prices = load_universe_v3(start=start, end=end)
    exp1220_base = build_exp1220_daily_returns(prices)
    print(f"      EXP-1220 base: {len(exp1220_base)} days "
          f"({exp1220_base.index[0].date()} → {exp1220_base.index[-1].date()})")

    # VIX / VIX3M / SPY for dynamic signals
    print("      ^VIX, ^VIX3M, SPY from Yahoo...")
    vix = fetch_yahoo_close("^VIX", start, end)
    vix3m = fetch_yahoo_close("^VIX3M", start, end)
    spy = fetch_yahoo_close("SPY", start, end)
    print(f"      VIX {len(vix)} days, VIX3M {len(vix3m)} days, SPY {len(spy)} days")

    print("\n[2/6] Running Crisis Alpha v4 (hedge signal)...")
    cfg = ConfigV4(
        name="v4_production",
        lookback_preset=V4_PRODUCTION["lookback_preset"],
        vol_target=V4_PRODUCTION["vol_target"],
        leverage=V4_PRODUCTION["leverage"],
        dd_brake_threshold=V4_PRODUCTION["dd_brake_threshold"],
        dd_brake_zone=V4_PRODUCTION["dd_brake_zone"],
        max_weight=V4_PRODUCTION["max_weight"],
        require_confirmation=V4_PRODUCTION["require_confirmation"],
    )
    cfg = backtest_v4(prices, cfg)
    hedge_rets = cfg.daily_returns
    print(f"      Crisis Alpha v4: CAGR {cfg.cagr:.1f}%  "
          f"Sharpe {cfg.sharpe:.2f}  DD {cfg.max_dd:.1f}%")

    print("\n[3/6] Running dynamic leverage cores (1x-5x, 1x-6x, 1x-7x)...")
    cores = {}
    for cap_name, tiers in LEVERAGE_CAPS.items():
        print(f"      {cap_name} ...", end=" ")
        dyn_rets, states = run_dynamic_backtest_custom(
            exp1220_base, vix, vix3m, spy, tiers,
        )
        avg_lev = np.mean([s.final_leverage for s in states])
        cores[cap_name] = {
            "returns": dyn_rets,
            "states": states,
            "avg_lev": avg_lev,
        }
        m = compute_metrics(dyn_rets.values)
        print(f"CAGR {m['cagr']*100:.1f}% "
              f"Sharpe {m['sharpe']:.2f} "
              f"DD {m['dd']*100:.1f}% "
              f"avg_lev {avg_lev:.2f}x")

    # Static 2x baseline
    print("      static 2x ...", end=" ")
    static_2x = exp1220_base * 2.0
    m = compute_metrics(static_2x.values)
    print(f"CAGR {m['cagr']*100:.1f}% "
          f"Sharpe {m['sharpe']:.2f} "
          f"DD {m['dd']*100:.1f}%")

    print("\n[4/6] Building combo portfolios...")
    combos = []

    # Test configurations
    test_configs = [
        ("Dynamic 1x-5x (unhedged)", "1x-5x", 0.00, cores["1x-5x"]["returns"], cores["1x-5x"]["avg_lev"]),
        ("Dynamic 1x-5x + 10% hedge", "1x-5x", 0.10, cores["1x-5x"]["returns"], cores["1x-5x"]["avg_lev"]),
        ("Dynamic 1x-5x + 15% hedge", "1x-5x", 0.15, cores["1x-5x"]["returns"], cores["1x-5x"]["avg_lev"]),
        ("Dynamic 1x-6x (unhedged)", "1x-6x", 0.00, cores["1x-6x"]["returns"], cores["1x-6x"]["avg_lev"]),
        ("Dynamic 1x-6x + 10% hedge", "1x-6x", 0.10, cores["1x-6x"]["returns"], cores["1x-6x"]["avg_lev"]),
        ("Dynamic 1x-7x (unhedged)", "1x-7x", 0.00, cores["1x-7x"]["returns"], cores["1x-7x"]["avg_lev"]),
        ("Dynamic 1x-7x + 10% hedge", "1x-7x", 0.10, cores["1x-7x"]["returns"], cores["1x-7x"]["avg_lev"]),
        ("Static 2x (unhedged)", "static 2x", 0.00, static_2x, 2.0),
        ("Static 2x + 10% hedge (baseline)", "static 2x", 0.10, static_2x, 2.0),
    ]

    for name, cap, hedge_pct, core, avg_lev in test_configs:
        combo = build_combo(core, hedge_rets, name, cap, hedge_pct, avg_lev)
        combos.append(combo)
        target = "PASS" if combo.max_dd < 15.0 else "FAIL"
        print(f"      {name:40s} CAGR {combo.cagr:+6.1f}% "
              f"Sharpe {combo.sharpe:5.2f} DD {combo.max_dd:5.1f}% "
              f"Calmar {combo.calmar:5.2f} [{target}]")

    # Find best config meeting <15% DD
    print("\n[5/6] Selecting optimal (DD <15% with max CAGR)...")
    qualifying = [c for c in combos if c.max_dd < 15.0]
    if qualifying:
        best = max(qualifying, key=lambda c: c.cagr)
        print(f"      Best: {best.name} → CAGR {best.cagr:+.1f}%, "
              f"DD {best.max_dd:.1f}%, Calmar {best.calmar:.2f}")
    else:
        best = min(combos, key=lambda c: c.max_dd)
        print(f"      None meet <15% DD. Lowest DD: {best.name} ({best.max_dd:.1f}%)")

    return {
        "combos": combos,
        "cores": cores,
        "static_2x": static_2x,
        "hedge_rets": hedge_rets,
        "v4_config": cfg,
        "best": best,
    }
