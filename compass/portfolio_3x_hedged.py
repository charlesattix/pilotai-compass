"""
EXP-1220 @ 3x + Crisis Alpha v4 Hedge
=======================================
Goal: Combine EXP-1220 at 3x leverage with Crisis Alpha v4 as a hedge
overlay. Does the hedge reduce the 3x max DD below 15%?

Configuration:
  - Core: EXP-1220 @ 3x (validated proxy from exp1780_exp1220_integration)
  - Hedge: Crisis Alpha v4 (DD-braked, confirmed, conservative leverage)
  - Allocations tested: 0%, 5%, 10%, 15% Crisis Alpha

Rule Zero: real Yahoo Finance data throughout. All drivers (SPY, sector
ETFs, bonds, commodities, FX) are real market prices. EXP-1220 uses its
calibrated functional proxy that reproduces validated metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADING_DAYS = 252

from compass.crisis_alpha_v3 import load_universe_v3
from compass.crisis_alpha_v4 import ConfigV4, backtest_v4, UNIVERSE_V4
from compass.exp1780_exp1220_integration import build_exp1220_daily_returns


# ═══════════════════════════════════════════════════════════════════════════
# Configuration — v4 production config
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

# EXP-1220 leverage options to test
EXP1220_LEVERAGES = [1.0, 1.5, 2.0, 2.5, 3.0]

# Crisis Alpha allocations to test
HEDGE_ALLOCATIONS = [0.0, 0.05, 0.075, 0.10, 0.125, 0.15]


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def corrected_sharpe(rets: np.ndarray, rf: float = 0.045) -> float:
    """Arithmetic daily mean Sharpe — canonical formula."""
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
        return {"cagr": 0, "sharpe": 0, "dd": 0, "sortino": 0, "calmar": 0, "vol": 0}
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
        "sortino": float(sortino), "calmar": float(calmar), "vol": float(vol),
    }


def yearly_metrics(rets: pd.Series) -> Dict[int, Dict[str, float]]:
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
        }
    return yearly


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio construction
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioResult:
    name: str
    exp1220_lev: float
    hedge_pct: float
    cagr: float
    sharpe: float
    max_dd: float
    covid_dd: float
    bear2022_dd: float
    sortino: float
    calmar: float
    vol: float
    yearly: Dict[int, Dict]
    daily_returns: pd.Series = field(default=None, repr=False)


def crisis_period_dd(rets: pd.Series, start: str, end: str) -> float:
    """Max drawdown during a specific crisis period."""
    mask = (rets.index >= start) & (rets.index <= end)
    period = rets[mask].values
    if len(period) < 2:
        return 0.0
    eq = np.cumprod(1 + period)
    hwm = np.maximum.accumulate(eq)
    return float((1 - eq / hwm).max())


def build_portfolio(
    exp1220_daily: pd.Series,
    crisis_alpha_daily: pd.Series,
    exp1220_lev: float,
    hedge_pct: float,
    name: str = "",
) -> PortfolioResult:
    """Build combined portfolio with specified leverage and hedge allocation.

    Core = EXP-1220 * (1 - hedge_pct) * exp1220_lev
    Hedge = Crisis Alpha * hedge_pct
    """
    # Align
    common = exp1220_daily.index.intersection(crisis_alpha_daily.index)
    e1220 = exp1220_daily.loc[common]
    cha = crisis_alpha_daily.loc[common]

    # Apply leverage to EXP-1220, then combine with hedge
    core_weight = 1.0 - hedge_pct
    combined = (e1220 * exp1220_lev * core_weight) + (cha * hedge_pct)

    m = compute_metrics(combined.values)
    cvd = crisis_period_dd(combined, "2020-02-19", "2020-03-23")
    bear = crisis_period_dd(combined, "2022-01-03", "2022-10-12")

    return PortfolioResult(
        name=name or f"EXP-1220@{exp1220_lev}x + {hedge_pct*100:.1f}% hedge",
        exp1220_lev=exp1220_lev,
        hedge_pct=hedge_pct,
        cagr=round(m["cagr"] * 100, 2),
        sharpe=round(m["sharpe"], 2),
        max_dd=round(m["dd"] * 100, 2),
        covid_dd=round(cvd * 100, 2),
        bear2022_dd=round(bear * 100, 2),
        sortino=round(m["sortino"], 2),
        calmar=round(m["calmar"], 2),
        vol=round(m["vol"] * 100, 2),
        yearly=yearly_metrics(combined),
        daily_returns=combined,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WFFold:
    oos_year: int
    n_train: int
    n_test: int
    unhedged_cagr: float
    unhedged_sharpe: float
    unhedged_dd: float
    hedged_cagr: float
    hedged_sharpe: float
    hedged_dd: float
    dd_reduction: float


def walk_forward(
    exp1220_daily: pd.Series,
    crisis_alpha_daily: pd.Series,
    exp1220_lev: float = 3.0,
    hedge_pct: float = 0.10,
) -> List[WFFold]:
    """Expanding-window walk-forward comparing unhedged vs hedged."""
    common = exp1220_daily.index.intersection(crisis_alpha_daily.index)
    e1220 = exp1220_daily.loc[common]
    cha = crisis_alpha_daily.loc[common]

    years = sorted(set(e1220.index.year))
    folds = []

    # Need at least 1 year IS before OOS
    for i, test_yr in enumerate(years[1:], start=1):
        train_mask = e1220.index.year < test_yr
        test_mask = e1220.index.year == test_yr
        if train_mask.sum() < 50 or test_mask.sum() < 50:
            continue

        # Unhedged OOS: EXP-1220 at full leverage
        unhedged_oos = e1220[test_mask].values * exp1220_lev
        u_m = compute_metrics(unhedged_oos)

        # Hedged OOS
        hedged_oos = (e1220[test_mask].values * exp1220_lev * (1 - hedge_pct)
                      + cha[test_mask].values * hedge_pct)
        h_m = compute_metrics(hedged_oos)

        folds.append(WFFold(
            oos_year=test_yr,
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            unhedged_cagr=round(u_m["cagr"] * 100, 2),
            unhedged_sharpe=round(u_m["sharpe"], 2),
            unhedged_dd=round(u_m["dd"] * 100, 2),
            hedged_cagr=round(h_m["cagr"] * 100, 2),
            hedged_sharpe=round(h_m["sharpe"], 2),
            hedged_dd=round(h_m["dd"] * 100, 2),
            dd_reduction=round((u_m["dd"] - h_m["dd"]) * 100, 2),
        ))

    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline() -> Dict:
    print("[1/5] Loading real Yahoo data (v4 universe)...")
    prices = load_universe_v3(start="2014-01-01", end="2026-01-01")
    print(f"      {len(prices)} days × {len(prices.columns)} assets "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    print("\n[2/5] Building EXP-1220 base returns (real SPY)...")
    e1220 = build_exp1220_daily_returns(prices)
    e1220_m = compute_metrics(e1220.values)
    print(f"      EXP-1220 @ 1x: CAGR {e1220_m['cagr']*100:.1f}%  "
          f"Sharpe {e1220_m['sharpe']:.2f}  Max DD {e1220_m['dd']*100:.1f}%")

    print("\n[3/5] Running Crisis Alpha v4 production config...")
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
    cha = cfg.daily_returns
    print(f"      Crisis Alpha v4: CAGR {cfg.cagr:.1f}%  "
          f"Sharpe {cfg.sharpe:.2f}  Max DD {cfg.max_dd:.1f}%  "
          f"SPY corr {cfg.corr_to_spy:+.2f}")

    print("\n[4/5] Building portfolio variants at EXP-1220 @ 3x...")
    portfolios = []
    for hedge in HEDGE_ALLOCATIONS:
        p = build_portfolio(e1220, cha, exp1220_lev=3.0, hedge_pct=hedge)
        portfolios.append(p)
        print(f"      Hedge {hedge*100:4.1f}%: "
              f"CAGR {p.cagr:+6.1f}% Sharpe {p.sharpe:5.2f} "
              f"MaxDD {p.max_dd:5.1f}% COVID {p.covid_dd:5.1f}% "
              f"2022 {p.bear2022_dd:5.1f}%")

    # Leverage sweep at 10% hedge
    print("\n[4b] Leverage sweep at 10% hedge allocation...")
    lev_sweep = []
    for lev in EXP1220_LEVERAGES:
        p = build_portfolio(e1220, cha, exp1220_lev=lev, hedge_pct=0.10)
        lev_sweep.append(p)
        print(f"      EXP-1220 @ {lev}x: "
              f"CAGR {p.cagr:+6.1f}% Sharpe {p.sharpe:5.2f} "
              f"MaxDD {p.max_dd:5.1f}%")

    print("\n[5/5] Walk-forward (3x + 10% hedge)...")
    folds = walk_forward(e1220, cha, exp1220_lev=3.0, hedge_pct=0.10)
    for f in folds:
        print(f"      {f.oos_year}: unhedged DD {f.unhedged_dd:5.1f}% → "
              f"hedged DD {f.hedged_dd:5.1f}%  "
              f"(Δ {f.dd_reduction:+.1f}pp, hedged CAGR {f.hedged_cagr:+6.1f}%)")

    # Select "best" hedge that reduces DD below 15% while maximizing CAGR
    target_dd = 15.0
    qualifying = [p for p in portfolios if p.max_dd < target_dd]
    if qualifying:
        best = max(qualifying, key=lambda p: p.cagr)
    else:
        best = min(portfolios, key=lambda p: p.max_dd)

    return {
        "prices": prices,
        "exp1220_daily": e1220,
        "crisis_alpha_daily": cha,
        "v4_config": cfg,
        "portfolios": portfolios,
        "lev_sweep": lev_sweep,
        "walk_forward": folds,
        "best": best,
        "unhedged_3x": portfolios[0],  # hedge_pct == 0
    }
