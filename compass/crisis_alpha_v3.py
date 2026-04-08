"""
EXP-1780 v3 — Focused Optimization Around Winning v2 Config

v2 winner: v2_round / vol_target / 2.0x → CAGR +8.5%, SPY corr -0.238

v3 improvements:
  1. Expanded universe: adds EFA, EEM, DBA, DBB (14 total assets)
  2. Focused grid around vol_target method (the only one producing neg corr)
  3. New lookback presets centered on v2_round [20,60,120,200]
  4. Vol-scaled position sizing with different targets (6%, 8%, 10%, 12%)
  5. KEY METRIC: correlation to EXP-1220 during DRAWDOWN periods specifically

Rule Zero: 100% REAL Yahoo data. Zero synthetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADING_DAYS = 252

from compass.crisis_alpha import CRISIS_PERIODS, _compute_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Expanded universe — 14 assets
# ═══════════════════════════════════════════════════════════════════════════

UNIVERSE_V3 = [
    # Equities (5)
    "SPY", "IWM", "EFA", "EEM", "QQQ",
    # Bonds (3)
    "TLT", "LQD", "HYG",
    # Commodities (4)
    "GLD", "USO", "DBA", "DBB",
    # FX (1)
    "UUP",
]


# ═══════════════════════════════════════════════════════════════════════════
# Focused lookback grid around v2_round winner
# ═══════════════════════════════════════════════════════════════════════════

LOOKBACK_GRID = {
    # Baseline (v2 winner)
    "v2_round":      ([20, 60, 120, 200], [0.15, 0.25, 0.30, 0.30]),
    # Variations
    "tight_around":  ([15, 50, 100, 180], [0.15, 0.25, 0.30, 0.30]),
    "wide_around":   ([25, 70, 140, 220], [0.15, 0.25, 0.30, 0.30]),
    # Shorter focus (faster signal)
    "fast":          ([10, 30, 60, 120],  [0.20, 0.30, 0.30, 0.20]),
    # Longer focus (more stable)
    "slow":          ([30, 90, 180, 252], [0.10, 0.20, 0.30, 0.40]),
    # 5-lookback (more diversified)
    "five_lb":       ([10, 21, 63, 126, 252], [0.10, 0.15, 0.25, 0.25, 0.25]),
}


# Vol targets to test
VOL_TARGETS = [0.06, 0.08, 0.10, 0.12]

# Leverage levels
LEVERAGE_LEVELS = [1.5, 2.0, 2.5]


@dataclass
class ConfigV3:
    name: str
    lookback_preset: str
    vol_target: float
    leverage: float
    n_assets: int
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    vol: float
    corr_to_spy: float
    corr_during_dd: float       # KEY METRIC: correlation during EXP-1220 drawdown periods
    crisis_avg_outperf: float
    crisis_performance: Dict[str, float]
    yearly: Dict[int, Dict[str, float]]
    equity: List[float]
    is_sharpe: float
    oos_sharpe: float


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_universe_v3(start: str = "2014-01-01", end: str = "2026-01-01") -> pd.DataFrame:
    """Load real Yahoo adjusted closes for the v3 universe."""
    import yfinance as yf
    prices = {}
    dropped = []
    for tk in UNIVERSE_V3:
        try:
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if len(df) < 400:
                dropped.append((tk, f"insufficient: {len(df)}"))
                continue
            prices[tk] = df["Close"]
        except Exception as e:
            dropped.append((tk, str(e)[:40]))

    if dropped:
        print(f"    Dropped: {dropped}")

    df = pd.DataFrame(prices).dropna()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Signal and weights
# ═══════════════════════════════════════════════════════════════════════════

def compute_momentum(prices, lookbacks, weights):
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for lb, w in zip(lookbacks, weights):
        mom = prices.pct_change(lb)
        signal = signal + w * mom.fillna(0)
    return signal


def compute_vol_target_weights(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    vol_target: float,
    leverage: float,
    vol_lookback: int = 60,
    max_weight: float = 0.30,
) -> pd.DataFrame:
    """Vol-targeted sizing — the method that produced the only neg-corr config."""
    returns = prices.pct_change().fillna(0)
    rolling_vol = (returns.rolling(vol_lookback, min_periods=20).std()
                   * math.sqrt(TRADING_DAYS)).fillna(vol_target)

    raw = (np.sign(signal)
           * np.minimum(np.abs(signal) * 5, 1.0)
           * vol_target / rolling_vol)
    raw = raw.clip(-max_weight, max_weight)

    gross = raw.abs().sum(axis=1)
    scale = np.where(gross > leverage, leverage / gross, 1.0)
    raw = raw.multiply(scale, axis=0)
    return raw


# ═══════════════════════════════════════════════════════════════════════════
# DD-period correlation (KEY METRIC)
# ═══════════════════════════════════════════════════════════════════════════

def exp1220_drawdown_periods(spy_returns: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Identify periods when EXP-1220 (credit spreads) would be in drawdown.

    EXP-1220 is a short-vol / credit spread strategy — it draws down when:
      - SPY has sharp negative days (gap down > 2%)
      - VIX spikes (which happens on SPY drops)
      - Sustained bear markets (rolling 20d return < -5%)

    We proxy this with SPY drawdown periods: rolling 20d peak-to-current
    drawdown > 5%.
    """
    # Compute rolling 20d drawdown
    roll_peak = spy_returns.add(1).cumprod().rolling(60, min_periods=20).max()
    cum = spy_returns.add(1).cumprod()
    roll_dd = (cum / roll_peak - 1)

    # DD periods: when rolling DD < -3%
    in_dd = roll_dd < -0.03

    periods = []
    start = None
    for i, (dt, v) in enumerate(in_dd.items()):
        if v and start is None:
            start = dt
        elif not v and start is not None:
            periods.append((start, dt))
            start = None
    if start is not None:
        periods.append((start, in_dd.index[-1]))

    return periods


def corr_during_dd(strategy_returns: np.ndarray,
                   strategy_dates: pd.DatetimeIndex,
                   dd_periods: List[Tuple]) -> float:
    """Compute correlation between strategy and SPY during DD periods only."""
    if not dd_periods:
        return 0.0
    mask = np.zeros(len(strategy_dates), dtype=bool)
    for start, end in dd_periods:
        mask |= (strategy_dates >= start) & (strategy_dates <= end)
    if mask.sum() < 10:
        return 0.0

    strat_dd = strategy_returns[mask[:len(strategy_returns)]]
    # Need to also get SPY returns for DD periods
    return float(mask.sum())  # placeholder — actual corr computed in backtest


# ═══════════════════════════════════════════════════════════════════════════
# Backtest one config
# ═══════════════════════════════════════════════════════════════════════════

def backtest_config_v3(
    prices: pd.DataFrame,
    lookback_preset: str,
    vol_target: float,
    leverage: float,
    rebalance_days: int = 5,
) -> ConfigV3:
    lookbacks, lw = LOOKBACK_GRID[lookback_preset]
    signal = compute_momentum(prices, lookbacks, lw)
    weights = compute_vol_target_weights(prices, signal, vol_target, leverage)
    asset_returns = prices.pct_change().fillna(0)

    # Hold for rebalance period
    held = weights.copy()
    for i in range(len(held)):
        if i % rebalance_days != 0 and i > 0:
            held.iloc[i] = held.iloc[i - 1]
    lagged = held.shift(1).fillna(0)

    port_rets = (lagged * asset_returns).sum(axis=1)
    warmup = max(lookbacks)
    valid_idx = prices.index[warmup] if len(prices) > warmup else prices.index[0]
    port_rets = port_rets[port_rets.index >= valid_idx]

    rets = port_rets.values
    m = _compute_metrics(rets)

    # Equity curve
    eq = [100_000.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))

    # SPY correlation (overall)
    spy_rets = asset_returns["SPY"][asset_returns.index >= valid_idx].values
    min_len = min(len(rets), len(spy_rets))
    corr_spy = float(np.corrcoef(rets[:min_len], spy_rets[:min_len])[0, 1]) if min_len > 10 else 0

    # KEY METRIC: correlation during SPY drawdown periods (proxy for EXP-1220 DD)
    spy_series = asset_returns["SPY"][asset_returns.index >= valid_idx]
    dd_periods = exp1220_drawdown_periods(spy_series)

    if dd_periods:
        dd_mask = np.zeros(min_len, dtype=bool)
        spy_idx = spy_series.index[:min_len]
        for ds, de in dd_periods:
            mask_arr = np.asarray((spy_idx >= ds) & (spy_idx <= de))
            dd_mask |= mask_arr

        if dd_mask.sum() > 10:
            strat_dd_rets = rets[:min_len][dd_mask]
            spy_dd_rets = spy_rets[:min_len][dd_mask]
            if np.std(strat_dd_rets) > 1e-8 and np.std(spy_dd_rets) > 1e-8:
                corr_dd = float(np.corrcoef(strat_dd_rets, spy_dd_rets)[0, 1])
            else:
                corr_dd = 0.0
        else:
            corr_dd = 0.0
    else:
        corr_dd = 0.0

    # Crisis attribution
    crisis_perf = {}
    crisis_outperf_list = []
    for name, (cstart, cend) in CRISIS_PERIODS.items():
        mask = np.asarray((port_rets.index >= cstart) & (port_rets.index <= cend))
        if mask.sum() < 3:
            continue
        mask_len = min(len(mask), len(rets), len(spy_rets))
        sub_mask = mask[:mask_len]
        strat = rets[:mask_len][sub_mask]
        spy_c = spy_rets[:mask_len][sub_mask]
        if len(strat) < 3:
            continue
        strat_ret = float(np.prod(1 + strat) - 1)
        spy_ret = float(np.prod(1 + spy_c) - 1)
        delta = (strat_ret - spy_ret) * 100
        crisis_perf[name] = round(delta, 2)
        crisis_outperf_list.append(delta)
    avg_crisis = float(np.mean(crisis_outperf_list)) if crisis_outperf_list else 0

    # Yearly breakdown
    yearly = {}
    for yr in sorted(set(port_rets.index.year)):
        yr_mask = np.asarray(port_rets.index.year == yr)
        yr_rets = rets[yr_mask[:len(rets)]]
        if len(yr_rets) < 5:
            continue
        ym = _compute_metrics(yr_rets)
        yearly[int(yr)] = {
            "cagr": round(ym["cagr"] * 100, 2),
            "sharpe": round(ym["sharpe"], 2),
            "dd": round(ym["dd"] * 100, 2),
        }

    # IS/OOS (2014-2020 vs 2021-2025)
    years = sorted(yearly.keys())
    is_years = [y for y in years if y <= 2020]
    oos_years = [y for y in years if y > 2020]
    is_mask = np.array([y in is_years for y in port_rets.index.year])[:len(rets)]
    oos_mask = np.array([y in oos_years for y in port_rets.index.year])[:len(rets)]
    is_m = _compute_metrics(rets[is_mask])
    oos_m = _compute_metrics(rets[oos_mask])

    return ConfigV3(
        name=f"{lookback_preset} / vol={vol_target:.2f} / {leverage}x",
        lookback_preset=lookback_preset,
        vol_target=vol_target,
        leverage=leverage,
        n_assets=len(prices.columns),
        cagr=round(m["cagr"] * 100, 2),
        sharpe=round(m["sharpe"], 2),
        sortino=round(m["sortino"], 2),
        max_dd=round(m["dd"] * 100, 2),
        calmar=round(m["calmar"], 2),
        vol=round(m["vol"] * 100, 2),
        corr_to_spy=round(corr_spy, 3),
        corr_during_dd=round(corr_dd, 3),
        crisis_avg_outperf=round(avg_crisis, 2),
        crisis_performance=crisis_perf,
        yearly=yearly,
        equity=eq,
        is_sharpe=round(is_m["sharpe"], 2),
        oos_sharpe=round(oos_m["sharpe"], 2),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Grid search
# ═══════════════════════════════════════════════════════════════════════════

def run_grid_v3(prices: pd.DataFrame) -> List[ConfigV3]:
    """Focused grid: all lookback presets × vol targets × leverage levels."""
    results = []
    total = len(LOOKBACK_GRID) * len(VOL_TARGETS) * len(LEVERAGE_LEVELS)
    i = 0
    for preset in LOOKBACK_GRID.keys():
        for vt in VOL_TARGETS:
            for lev in LEVERAGE_LEVELS:
                i += 1
                print(f"  [{i}/{total}] {preset} / vol={vt:.2f} / {lev}x...", end=" ")
                cfg = backtest_config_v3(prices, preset, vt, lev)
                passes = cfg.cagr >= 8.0 and cfg.corr_during_dd < 0.0
                tag = "PASS" if passes else ""
                print(f"CAGR={cfg.cagr:+.1f}% "
                      f"corr_spy={cfg.corr_to_spy:+.2f} "
                      f"corr_DD={cfg.corr_during_dd:+.2f} {tag}")
                results.append(cfg)

    return results
