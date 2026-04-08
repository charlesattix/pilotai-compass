"""
EXP-1780 v2 — Crisis Alpha with boosted CAGR

Improvements over v1:
  1. 11-asset universe (was 5): adds EEM, EFA, DBC, HYG, LQD, IWM
  2. Lookback sweep: test [20/60/120/200] vs v1 default [21/63/126/252]
  3. Leverage sweep: 1.0x, 1.5x, 2.0x
  4. Risk parity weighting (inverse-vol) vs equal-weight signal
  5. Walk-forward 2015-2025 with corrected Sharpe

Target: preserve negative SPY correlation while pushing CAGR to 10-15%.

Rule Zero: 100% REAL data from Yahoo Finance. Zero synthetic pricing.
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

# Reuse primitives from v1
from compass.crisis_alpha import (
    CRISIS_PERIODS, CrisisMetrics, WFFold, _sharpe, _compute_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
# Expanded universe
# ═══════════════════════════════════════════════════════════════════════════

# 11 assets spanning equities, bonds, commodities, FX, credit
ASSET_UNIVERSE_V2 = [
    # Equities (4)
    "SPY",    # S&P 500
    "IWM",    # Russell 2000 small-cap
    "EFA",    # Developed international
    "EEM",    # Emerging markets
    # Bonds (3)
    "TLT",    # 20+ year Treasury
    "LQD",    # Investment-grade corp
    "HYG",    # High-yield corp
    # Commodities (3)
    "GLD",    # Gold
    "USO",    # Oil
    "DBC",    # Broad commodities basket
    # FX (1)
    "UUP",    # US Dollar
]

# Alternate lookback sets to test
LOOKBACK_PRESETS = {
    "v1_default": ([21, 63, 126, 252], [0.15, 0.25, 0.30, 0.30]),
    "v2_round":   ([20, 60, 120, 200], [0.15, 0.25, 0.30, 0.30]),
    "short_bias": ([10, 20, 60, 120],  [0.20, 0.30, 0.30, 0.20]),
    "long_bias":  ([60, 120, 200, 252], [0.15, 0.25, 0.30, 0.30]),
}

# Weighting methods
WEIGHTING_METHODS = ["equal_signal", "risk_parity", "vol_target"]

# Leverage levels to sweep
LEVERAGE_LEVELS = [1.0, 1.5, 2.0]


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConfigResult:
    name: str
    lookback_preset: str
    weighting: str
    leverage: float
    n_assets: int
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    vol: float
    corr_to_spy: float
    crisis_avg_outperf: float
    passes_target: bool  # CAGR in [8, 18] and corr < 0.0
    yearly: Dict[int, Dict[str, float]]
    wf_folds: List[WFFold]
    equity: List[float]


@dataclass
class V2Result:
    all_configs: List[ConfigResult]
    best: ConfigResult
    universe_size: int
    n_days: int
    crisis_metrics_best: List[CrisisMetrics]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (REAL from Yahoo)
# ═══════════════════════════════════════════════════════════════════════════

def load_real_prices_v2(
    tickers: List[str] = None,
    start: str = "2014-01-01",
    end: str = "2026-01-01",
    min_days: int = 400,
) -> pd.DataFrame:
    """Load REAL daily adjusted closes. Drop tickers with insufficient history."""
    import yfinance as yf
    tickers = tickers or ASSET_UNIVERSE_V2
    prices = {}
    dropped = []
    for tk in tickers:
        try:
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if len(df) < min_days:
                dropped.append((tk, len(df)))
                continue
            prices[tk] = df["Close"]
        except Exception as e:
            dropped.append((tk, str(e)[:40]))
            continue

    if not prices:
        raise RuntimeError("No tickers loaded from Yahoo")
    if dropped:
        print(f"    Dropped: {dropped}")

    df = pd.DataFrame(prices).dropna()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Signal + weighting
# ═══════════════════════════════════════════════════════════════════════════

def compute_momentum_signal(
    prices: pd.DataFrame,
    lookbacks: List[int],
    weights: List[float],
) -> pd.DataFrame:
    """Multi-timeframe time-series momentum (same as v1 but parameterized)."""
    if len(lookbacks) != len(weights):
        raise ValueError("lookbacks and weights must have same length")
    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for lb, w in zip(lookbacks, weights):
        mom = prices.pct_change(lb)
        signal = signal + w * mom.fillna(0)
    return signal


def compute_weights(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    method: str,
    leverage: float,
    vol_lookback: int = 60,
    vol_target: float = 0.10,
    max_weight: float = 0.35,
) -> pd.DataFrame:
    """Compute portfolio weights from signal using specified method.

    - equal_signal: sign(signal) / n_active
    - risk_parity: inverse-vol weighted by sign(signal)
    - vol_target: v1 method (vol_target / rolling_vol × signal)
    """
    returns = prices.pct_change().fillna(0)
    rolling_vol = (returns.rolling(vol_lookback, min_periods=20).std()
                   * math.sqrt(TRADING_DAYS)).fillna(vol_target)

    if method == "equal_signal":
        # Simple: equal weight to all assets with positive signal, flat otherwise
        active = (signal > 0).astype(float)
        n_active = active.sum(axis=1).replace(0, 1)
        raw = active.div(n_active, axis=0) * leverage
        raw = raw.clip(0, max_weight)
    elif method == "risk_parity":
        # Risk parity: inverse vol, filtered by positive signal
        active = (signal > 0).astype(float)
        inv_vol = active / rolling_vol.replace(0, np.nan)
        inv_vol = inv_vol.fillna(0)
        row_sum = inv_vol.sum(axis=1).replace(0, 1)
        raw = inv_vol.div(row_sum, axis=0) * leverage
        raw = raw.clip(0, max_weight)
    elif method == "vol_target":
        # v1 method: vol-targeted, signed
        raw = (np.sign(signal)
               * np.minimum(np.abs(signal) * 5, 1.0)
               * vol_target / rolling_vol)
        raw = raw.clip(-max_weight, max_weight)
        gross = raw.abs().sum(axis=1)
        scale = np.where(gross > leverage, leverage / gross, 1.0)
        raw = raw.multiply(scale, axis=0)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return raw


# ═══════════════════════════════════════════════════════════════════════════
# Backtest
# ═══════════════════════════════════════════════════════════════════════════

def backtest_config(
    prices: pd.DataFrame,
    lookback_preset: str,
    weighting: str,
    leverage: float,
    rebalance_days: int = 5,
) -> ConfigResult:
    """Run one configuration through the full backtest pipeline."""
    lookbacks, lw = LOOKBACK_PRESETS[lookback_preset]
    signal = compute_momentum_signal(prices, lookbacks, lw)
    weights = compute_weights(prices, signal, weighting, leverage)
    asset_returns = prices.pct_change().fillna(0)

    # Hold weights for rebalance period
    held = weights.copy()
    for i in range(len(held)):
        if i % rebalance_days != 0 and i > 0:
            held.iloc[i] = held.iloc[i - 1]
    lagged = held.shift(1).fillna(0)

    port_rets = (lagged * asset_returns).sum(axis=1)
    # Skip warmup (longest lookback)
    warmup = max(lookbacks)
    valid_idx = prices.index[warmup] if len(prices) > warmup else prices.index[0]
    port_rets = port_rets[port_rets.index >= valid_idx]

    rets = port_rets.values
    m = _compute_metrics(rets)

    # Equity
    eq = [100_000.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))

    # SPY correlation
    spy_rets = asset_returns["SPY"][asset_returns.index >= valid_idx].values
    min_len = min(len(rets), len(spy_rets))
    if min_len > 10:
        corr = float(np.corrcoef(rets[:min_len], spy_rets[:min_len])[0, 1])
    else:
        corr = 0.0

    # Crisis attribution
    crisis_outperf = []
    for name, (start, end) in CRISIS_PERIODS.items():
        mask = np.asarray((port_rets.index >= start) & (port_rets.index <= end))
        if mask.sum() < 3:
            continue
        mask_len = min(len(mask), len(rets), len(spy_rets))
        sub_mask = mask[:mask_len]
        strat = rets[:mask_len][sub_mask]
        spy_c = spy_rets[:mask_len][sub_mask]
        if len(strat) < 3:
            continue
        delta = float(np.prod(1 + strat) - 1) - float(np.prod(1 + spy_c) - 1)
        crisis_outperf.append(delta * 100)
    avg_crisis = float(np.mean(crisis_outperf)) if crisis_outperf else 0

    # Yearly
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

    # Walk-forward expanding window
    years = sorted(yearly.keys())
    folds = []
    for i, test_yr in enumerate(years[1:], start=1):
        train_yrs = years[:i]
        train_mask = np.array([y in train_yrs for y in port_rets.index.year])[:len(rets)]
        test_mask = np.array([y == test_yr for y in port_rets.index.year])[:len(rets)]
        train_r = rets[train_mask]
        test_r = rets[test_mask]
        if len(train_r) < 50 or len(test_r) < 50:
            continue
        is_m = _compute_metrics(train_r)
        oos_m = _compute_metrics(test_r)
        folds.append(WFFold(
            test_year=test_yr, train_years=train_yrs,
            n_train=len(train_r), n_test=len(test_r),
            is_sharpe=round(is_m["sharpe"], 2),
            oos_sharpe=round(oos_m["sharpe"], 2),
            oos_cagr=round(oos_m["cagr"] * 100, 2),
            oos_dd=round(oos_m["dd"] * 100, 2),
        ))

    cagr_pct = round(m["cagr"] * 100, 2)
    passes = (8.0 <= cagr_pct <= 18.0) and (corr < 0.0)

    return ConfigResult(
        name=f"{lookback_preset} / {weighting} / {leverage}x",
        lookback_preset=lookback_preset, weighting=weighting,
        leverage=leverage, n_assets=len(prices.columns),
        cagr=cagr_pct, sharpe=round(m["sharpe"], 2),
        sortino=round(m["sortino"], 2),
        max_dd=round(m["dd"] * 100, 2),
        calmar=round(m["calmar"], 2),
        vol=round(m["vol"] * 100, 2),
        corr_to_spy=round(corr, 3),
        crisis_avg_outperf=round(avg_crisis, 2),
        passes_target=passes, yearly=yearly, wf_folds=folds, equity=eq,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Grid search
# ═══════════════════════════════════════════════════════════════════════════

def run_full_grid(prices: pd.DataFrame) -> V2Result:
    """Run all lookback × weighting × leverage combinations."""
    all_configs = []

    for preset in LOOKBACK_PRESETS.keys():
        for weighting in WEIGHTING_METHODS:
            for leverage in LEVERAGE_LEVELS:
                config = backtest_config(prices, preset, weighting, leverage)
                all_configs.append(config)

    # Score: prefer configs that hit target (negative corr + 8-18% CAGR)
    # Among qualifiers, maximize (CAGR × -corr_to_spy)
    def _score(c: ConfigResult) -> float:
        if c.passes_target:
            return (c.cagr / 100) * (-c.corr_to_spy + 0.1) * 10
        # Penalty for non-qualifying
        return -1.0 + (c.cagr / 100) * 0.1

    best = max(all_configs, key=_score)

    # Re-run best config to get crisis metrics
    lookbacks, lw = LOOKBACK_PRESETS[best.lookback_preset]
    signal = compute_momentum_signal(prices, lookbacks, lw)
    weights = compute_weights(prices, signal, best.weighting, best.leverage)
    asset_returns = prices.pct_change().fillna(0)
    held = weights.copy()
    for i in range(len(held)):
        if i % 5 != 0 and i > 0:
            held.iloc[i] = held.iloc[i - 1]
    lagged = held.shift(1).fillna(0)
    port_rets = (lagged * asset_returns).sum(axis=1)
    warmup = max(lookbacks)
    valid_idx = prices.index[warmup] if len(prices) > warmup else prices.index[0]
    port_rets = port_rets[port_rets.index >= valid_idx]
    rets = port_rets.values
    spy_rets = asset_returns["SPY"][asset_returns.index >= valid_idx].values

    crisis_metrics = []
    for name, (start, end) in CRISIS_PERIODS.items():
        mask = np.asarray((port_rets.index >= start) & (port_rets.index <= end))
        if mask.sum() < 3:
            continue
        mask_len = min(len(mask), len(rets), len(spy_rets))
        sub_mask = mask[:mask_len]
        strat = rets[:mask_len][sub_mask]
        spy_c = spy_rets[:mask_len][sub_mask]
        if len(strat) < 3:
            continue
        s_ret = float(np.prod(1 + strat) - 1)
        sp_ret = float(np.prod(1 + spy_c) - 1)
        crisis_metrics.append(CrisisMetrics(
            name=name, start=start, end=end, n_days=int(sub_mask.sum()),
            strategy_return=round(s_ret * 100, 2),
            spy_return=round(sp_ret * 100, 2),
            outperformance=round((s_ret - sp_ret) * 100, 2),
        ))

    return V2Result(
        all_configs=all_configs, best=best,
        universe_size=len(prices.columns),
        n_days=len(prices),
        crisis_metrics_best=crisis_metrics,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    result: V2Result,
    output_path: str = "reports/crisis_alpha_v2_grid.html",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    best = result.best

    # Best equity SVG
    eq = best.equity
    w, h = 780, 220
    pl, pr, pt, pb = 65, 20, 28, 28
    pw, ph = w - pl - pr, h - pt - pb
    n = len(eq)
    ym, yx = min(eq) * 0.95, max(eq) * 1.05
    step = max(1, n // 500)
    pts = [(i, eq[i]) for i in range(0, n, step)]
    if pts[-1][0] != n - 1:
        pts.append((n - 1, eq[-1]))

    def tx(i): return pl + i / max(n - 1, 1) * pw
    def ty(v): return pt + (1 - (v - ym) / max(yx - ym, 1)) * ph

    d = " ".join(f"{'M' if j == 0 else 'L'}{tx(i):.1f},{ty(v):.1f}"
                 for j, (i, v) in enumerate(pts))
    eq_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"
  style="border:1px solid #e2e8f0;border-radius:6px">
  <text x="{w//2}" y="16" text-anchor="middle" font-size="11" fill="#64748b">
    Best Config Equity: {best.name}
  </text>
  <path d="{d}" fill="none" stroke="#16a34a" stroke-width="1.5"/>
</svg>"""

    # Grid results table — sort by score (passes_target first, then CAGR)
    sorted_cfgs = sorted(result.all_configs,
                         key=lambda c: (not c.passes_target, -c.cagr))
    grid_rows = ""
    for c in sorted_cfgs:
        is_best = c.name == best.name
        bg = ' style="background:#f0fdf4"' if is_best else (
             ' style="background:#fef3c7"' if c.passes_target else "")
        star = " ★" if is_best else (" ✓" if c.passes_target else "")
        cc = "#16a34a" if c.cagr > 0 else "#dc2626"
        corr_c = "#16a34a" if c.corr_to_spy < 0 else ("#d97706" if c.corr_to_spy < 0.2 else "#dc2626")
        grid_rows += f"""<tr{bg}>
          <td>{c.lookback_preset}{star}</td>
          <td>{c.weighting}</td>
          <td>{c.leverage}x</td>
          <td style="color:{cc};font-weight:700">{c.cagr:+.1f}%</td>
          <td>{c.sharpe:.2f}</td>
          <td>{c.sortino:.2f}</td>
          <td>{c.max_dd:.1f}%</td>
          <td>{c.calmar:.1f}</td>
          <td>{c.vol:.1f}%</td>
          <td style="color:{corr_c};font-weight:700">{c.corr_to_spy:+.3f}</td>
          <td>{c.crisis_avg_outperf:+.1f}%</td>
        </tr>"""

    # Best yearly
    yr_rows = ""
    for yr, ym in sorted(best.yearly.items()):
        cc = "#16a34a" if ym["cagr"] > 0 else "#dc2626"
        yr_rows += f"""<tr>
          <td>{yr}</td>
          <td style="color:{cc};font-weight:700">{ym['cagr']:+.1f}%</td>
          <td>{ym['sharpe']:.2f}</td>
          <td>{ym['dd']:.1f}%</td>
        </tr>"""

    # Best walk-forward
    wf_rows = ""
    for f in best.wf_folds:
        oc = "#16a34a" if f.oos_sharpe > 0 else "#dc2626"
        wf_rows += f"""<tr>
          <td>{f.test_year}</td>
          <td>{len(f.train_years)}y</td>
          <td>{f.is_sharpe:.2f}</td>
          <td style="color:{oc};font-weight:700">{f.oos_sharpe:.2f}</td>
          <td>{f.oos_cagr:+.1f}%</td>
          <td>{f.oos_dd:.1f}%</td>
        </tr>"""

    # Crisis attribution for best
    crisis_rows = ""
    for c in result.crisis_metrics_best:
        sc = "#16a34a" if c.outperformance > 0 else "#dc2626"
        strat_c = "#16a34a" if c.strategy_return > 0 else "#dc2626"
        spy_c = "#16a34a" if c.spy_return > 0 else "#dc2626"
        crisis_rows += f"""<tr>
          <td>{c.name}</td>
          <td style="color:{strat_c};font-weight:700">{c.strategy_return:+.1f}%</td>
          <td style="color:{spy_c}">{c.spy_return:+.1f}%</td>
          <td style="color:{sc};font-weight:700">{c.outperformance:+.1f}%</td>
        </tr>"""

    n_pass = sum(1 for c in result.all_configs if c.passes_target)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>EXP-1780 v2 Crisis Alpha Grid</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b}}
h1{{font-size:1.4rem;color:#0f172a}}h2{{font-size:1rem;color:#334155;margin-top:1.5rem;border-bottom:1px solid #e2e8f0;padding-bottom:4px}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:16px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}}
.card .v{{font-size:1.1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;font-size:0.82rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}}
td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}}
td:first-child{{text-align:left}}
svg{{display:block;margin:0.5rem 0}}
.callout{{background:#eff6ff;border-left:4px solid #3b82f6;padding:12px;margin:12px 0;border-radius:4px;font-size:0.85rem}}
</style></head><body>
<h1>EXP-1780 v2 — Crisis Alpha Grid Search</h1>
<p class="meta">
  {result.universe_size}-asset universe | {len(LOOKBACK_PRESETS)} lookback presets ×
  {len(WEIGHTING_METHODS)} weightings × {len(LEVERAGE_LEVELS)} leverages =
  {len(result.all_configs)} configs | {result.n_days} days real Yahoo data
</p>

<div class="callout">
<strong>Rule Zero:</strong> All prices from Yahoo Finance (real market data).
Zero synthetic pricing. Universe expanded from 5 → 11 assets for better
diversification. Target: negative SPY correlation AND 8-18% CAGR.
</div>

<div class="grid">
  <div class="card"><div class="l">Best CAGR</div><div class="v" style="color:#16a34a">{best.cagr:+.1f}%</div></div>
  <div class="card"><div class="l">Best Sharpe</div><div class="v">{best.sharpe:.2f}</div></div>
  <div class="card"><div class="l">Best Sortino</div><div class="v">{best.sortino:.2f}</div></div>
  <div class="card"><div class="l">Best Max DD</div><div class="v">{best.max_dd:.1f}%</div></div>
  <div class="card"><div class="l">Best Calmar</div><div class="v">{best.calmar:.1f}</div></div>
  <div class="card"><div class="l">Corr to SPY</div><div class="v" style="color:{'#16a34a' if best.corr_to_spy < 0 else '#d97706'}">{best.corr_to_spy:+.3f}</div></div>
  <div class="card"><div class="l">Configs Passing</div><div class="v">{n_pass}/{len(result.all_configs)}</div></div>
  <div class="card"><div class="l">Universe</div><div class="v">{result.universe_size} assets</div></div>
</div>

<h2>Best Configuration: {best.name}</h2>
<p class="meta">
  Lookback: <strong>{best.lookback_preset}</strong> |
  Weighting: <strong>{best.weighting}</strong> |
  Leverage: <strong>{best.leverage}x</strong> |
  Passes target: <strong style="color:{'#16a34a' if best.passes_target else '#dc2626'}">
    {'YES' if best.passes_target else 'NO'}
  </strong>
</p>

{eq_svg}

<h2>Crisis Period Attribution (Best Config)</h2>
<table>
<tr><th>Crisis</th><th>Strategy</th><th>SPY</th><th>Outperformance</th></tr>
{crisis_rows}
</table>

<h2>Grid Search Results (sorted: passing first, then by CAGR)</h2>
<p class="meta">★ = best overall, ✓ = passes target (CAGR 8-18% AND corr &lt; 0)</p>
<table>
<tr><th>Lookback</th><th>Weighting</th><th>Leverage</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>SPY ρ</th><th>Crisis Δ</th></tr>
{grid_rows}
</table>

<h2>Best Config — Yearly Performance</h2>
<table>
<tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr>
{yr_rows}
</table>

<h2>Best Config — Walk-Forward (Expanding Window)</h2>
<table>
<tr><th>Test Year</th><th>Train</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS CAGR</th><th>OOS DD</th></tr>
{wf_rows}
</table>

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/crisis_alpha_v2.py | Yahoo Finance real data |
Corrected Sharpe: arithmetic mean × √252 / std(daily, ddof=1) |
Rule Zero compliant: zero synthetic pricing
</div>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis():
    print("EXP-1780 v2 — Crisis Alpha Grid Search")
    print("=" * 60)

    print("\n  [1/2] Loading REAL prices for 11-asset universe...")
    prices = load_real_prices_v2(start="2014-01-01", end="2026-01-01")
    print(f"    Loaded {len(prices)} days × {len(prices.columns)} assets")
    print(f"    Assets: {list(prices.columns)}")
    print(f"    Range: {prices.index[0].date()} to {prices.index[-1].date()}")

    print(f"\n  [2/2] Running grid search: "
          f"{len(LOOKBACK_PRESETS)} × {len(WEIGHTING_METHODS)} × {len(LEVERAGE_LEVELS)} "
          f"= {len(LOOKBACK_PRESETS) * len(WEIGHTING_METHODS) * len(LEVERAGE_LEVELS)} configs...")
    result = run_full_grid(prices)

    print(f"\n  Grid Results ({len(result.all_configs)} configs):")
    # Top 10 by CAGR
    sorted_cfgs = sorted(result.all_configs, key=lambda c: -c.cagr)
    print(f"\n  Top 10 by CAGR:")
    print(f"    {'Config':<35} {'CAGR':>7} {'Sharpe':>7} {'DD':>6} {'Corr':>7} {'Pass':>5}")
    for c in sorted_cfgs[:10]:
        tag = "✓" if c.passes_target else "-"
        print(f"    {c.name:<35} {c.cagr:>6.1f}% {c.sharpe:>7.2f} "
              f"{c.max_dd:>5.1f}% {c.corr_to_spy:>+7.3f} {tag:>5}")

    best = result.best
    print(f"\n  BEST OVERALL: {best.name}")
    print(f"    CAGR: {best.cagr:+.1f}% (target 8-18%)")
    print(f"    Sharpe: {best.sharpe:.2f}")
    print(f"    Sortino: {best.sortino:.2f}")
    print(f"    Max DD: {best.max_dd:.1f}%")
    print(f"    Calmar: {best.calmar:.1f}")
    print(f"    Corr to SPY: {best.corr_to_spy:+.3f} (target <0)")
    print(f"    Passes target: {'YES' if best.passes_target else 'NO'}")

    n_pass = sum(1 for c in result.all_configs if c.passes_target)
    print(f"\n  Configs passing target ({n_pass}/{len(result.all_configs)}):")
    for c in result.all_configs:
        if c.passes_target:
            print(f"    ✓ {c.name}: CAGR {c.cagr:+.1f}%, "
                  f"Sharpe {c.sharpe:.2f}, Corr {c.corr_to_spy:+.3f}")

    print(f"\n  Crisis Attribution (best config):")
    for c in result.crisis_metrics_best:
        tag = "OUT" if c.outperformance > 0 else "UNDER"
        print(f"    {c.name:<25s}: strat {c.strategy_return:+6.1f}% | "
              f"SPY {c.spy_return:+6.1f}% | Δ {c.outperformance:+6.1f}% [{tag}]")

    report = generate_report(result)
    print(f"\n  Report: {report}")
    return result


if __name__ == "__main__":
    run_analysis()
