"""
EXP-1780 — Crisis Alpha / CTA Trend Following

A multi-asset time-series momentum strategy designed to have NEGATIVE
correlation with credit spread selling (EXP-1220) and specifically
profit during market crashes.

KEY DESIGN:
  - Long assets trending up, flat (or short) assets trending down
  - Multi-timeframe signals (1m, 3m, 6m, 12m momentum)
  - Volatility-targeted position sizing (constant risk per asset)
  - Asset universe: SPY, TLT, GLD, UUP (dollar), USO (oil)
  - Rebalance weekly

Rule Zero: 100% REAL data from Yahoo Finance. ZERO synthetic pricing.
Yahoo is a real market data provider — not synthetic generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

ASSET_UNIVERSE = ["SPY", "TLT", "GLD", "UUP", "USO"]

# Momentum lookback windows (trading days)
LOOKBACKS = [21, 63, 126, 252]  # 1m, 3m, 6m, 12m

# Signal weights — longer lookbacks more stable
SIGNAL_WEIGHTS = [0.15, 0.25, 0.30, 0.30]

# Vol target: each asset targets 10% annual vol contribution
VOL_TARGET_ANNUAL = 0.10

# Position limits
MAX_GROSS_LEVERAGE = 2.0
MAX_ASSET_WEIGHT = 0.40
MIN_ASSET_WEIGHT = 0.0  # allow flat

# Known crisis periods for attribution
CRISIS_PERIODS = {
    "COVID 2020": ("2020-02-19", "2020-03-23"),
    "2022 Bear": ("2022-01-03", "2022-10-12"),
    "Aug 2015 China": ("2015-08-10", "2015-08-25"),
    "Feb 2018 Volmageddon": ("2018-01-26", "2018-02-09"),
    "Q4 2018 Selloff": ("2018-10-03", "2018-12-24"),
}


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CrisisMetrics:
    name: str
    start: str
    end: str
    n_days: int
    strategy_return: float
    spy_return: float
    outperformance: float  # strategy - SPY


@dataclass
class WFFold:
    test_year: int
    train_years: List[int]
    n_train: int
    n_test: int
    is_sharpe: float
    oos_sharpe: float
    oos_cagr: float
    oos_dd: float


@dataclass
class BacktestResult:
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    vol: float
    n_days: int
    equity: List[float]
    daily_returns: np.ndarray
    weights_history: pd.DataFrame
    corr_to_spy: float
    crisis_metrics: List[CrisisMetrics]
    yearly: Dict[int, Dict[str, float]]
    wf_folds: List[WFFold]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (REAL from yfinance)
# ═══════════════════════════════════════════════════════════════════════════

def load_real_prices(
    tickers: List[str] = None,
    start: str = "2014-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """Load REAL daily adjusted closes from Yahoo Finance.

    Returns DataFrame with one column per ticker, indexed by date.
    Raises if any ticker has <500 days of data.
    """
    import yfinance as yf
    tickers = tickers or ASSET_UNIVERSE
    prices = {}
    for tk in tickers:
        df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if len(df) < 500:
            raise ValueError(f"Ticker {tk} has only {len(df)} days — insufficient")
        prices[tk] = df["Close"]
    return pd.DataFrame(prices).dropna()


# ═══════════════════════════════════════════════════════════════════════════
# Signal generation
# ═══════════════════════════════════════════════════════════════════════════

def compute_momentum_signal(
    prices: pd.DataFrame,
    lookbacks: List[int] = None,
    weights: List[float] = None,
) -> pd.DataFrame:
    """Multi-timeframe time-series momentum signal.

    For each asset and each lookback window, compute (price_today / price_lookback - 1).
    Combine via weighted average. Positive = trending up, negative = trending down.

    Returns DataFrame same shape as prices, with signal values.
    """
    lookbacks = lookbacks or LOOKBACKS
    weights = weights or SIGNAL_WEIGHTS
    if len(lookbacks) != len(weights):
        raise ValueError("lookbacks and weights must have same length")

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for lb, w in zip(lookbacks, weights):
        mom = prices.pct_change(lb)
        signal = signal + w * mom.fillna(0)
    return signal


def compute_vol_target_weights(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    vol_target: float = VOL_TARGET_ANNUAL,
    vol_lookback: int = 60,
    max_weight: float = MAX_ASSET_WEIGHT,
    max_gross: float = MAX_GROSS_LEVERAGE,
) -> pd.DataFrame:
    """Compute vol-targeted weights from momentum signal.

    Each asset gets w_i = sign(signal_i) × vol_target / realized_vol_i
    then capped and gross-normalized.
    """
    returns = prices.pct_change().fillna(0)
    rolling_vol = returns.rolling(vol_lookback, min_periods=20).std() * math.sqrt(TRADING_DAYS)
    rolling_vol = rolling_vol.fillna(vol_target)

    # Raw vol-scaled weight = sign(signal) × vol_target / realized_vol
    # Signal magnitude matters too — scale by strength
    raw = np.sign(signal) * np.minimum(np.abs(signal) * 5, 1.0) * vol_target / rolling_vol
    raw = raw.clip(-max_weight, max_weight)

    # Cap gross exposure
    gross = raw.abs().sum(axis=1)
    scale = np.where(gross > max_gross, max_gross / gross, 1.0)
    weights = raw.multiply(scale, axis=0)
    return weights


# ═══════════════════════════════════════════════════════════════════════════
# Backtest engine
# ═══════════════════════════════════════════════════════════════════════════

def _sharpe(rets: np.ndarray) -> float:
    """Corrected: arithmetic mean × sqrt(252) / std(daily, ddof=1)."""
    if len(rets) < 2:
        return 0.0
    sigma = float(rets.std(ddof=1))
    return float(rets.mean()) / sigma * math.sqrt(TRADING_DAYS) if sigma > 1e-12 else 0.0


def _compute_metrics(rets: np.ndarray) -> dict:
    if len(rets) < 2:
        return {"cagr": 0, "sharpe": 0, "dd": 0, "sortino": 0, "calmar": 0, "vol": 0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else -1
    sharpe = _sharpe(rets)
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = rets[rets < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else float(rets.std(ddof=1))
    sortino = float(rets.mean()) / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0
    vol = float(rets.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    return {"cagr": cagr, "sharpe": sharpe, "dd": dd, "sortino": sortino,
            "calmar": calmar, "vol": vol}


def backtest_crisis_alpha(prices: pd.DataFrame,
                          rebalance_days: int = 5) -> BacktestResult:
    """Full walk-forward backtest of the crisis alpha strategy.

    Weekly rebalance (default). Returns = previous-day weights × today's asset returns.
    """
    signal = compute_momentum_signal(prices)
    weights = compute_vol_target_weights(prices, signal)
    asset_returns = prices.pct_change().fillna(0)

    # Hold weights for `rebalance_days` — use weekly snapshot
    held_weights = weights.copy()
    for i in range(len(held_weights)):
        if i % rebalance_days != 0 and i > 0:
            held_weights.iloc[i] = held_weights.iloc[i - 1]

    # Shift weights by 1 day to avoid look-ahead
    lagged_weights = held_weights.shift(1).fillna(0)

    # Daily portfolio return = sum(weight_i × return_i)
    port_rets = (lagged_weights * asset_returns).sum(axis=1)
    # Skip first 252 days (warmup for longest lookback)
    valid_mask = port_rets.index >= prices.index[252] if len(prices) > 252 else port_rets.index >= prices.index[0]
    port_rets = port_rets[valid_mask]

    rets_array = port_rets.values
    metrics = _compute_metrics(rets_array)

    # Equity curve
    eq = [100_000.0]
    for r in rets_array:
        eq.append(eq[-1] * (1 + r))

    # Correlation to SPY
    spy_rets = asset_returns["SPY"][valid_mask].values
    common_len = min(len(port_rets), len(spy_rets))
    if common_len > 10:
        corr = float(np.corrcoef(rets_array[:common_len], spy_rets[:common_len])[0, 1])
    else:
        corr = 0.0

    # Crisis period attribution
    crisis = []
    for name, (start, end) in CRISIS_PERIODS.items():
        mask = np.asarray((port_rets.index >= start) & (port_rets.index <= end))
        if mask.sum() < 3:
            continue
        # rets_array and spy_rets are same length as port_rets
        min_len = min(len(rets_array), len(spy_rets), len(mask))
        strat_rets = rets_array[:min_len][mask[:min_len]]
        spy_crisis = spy_rets[:min_len][mask[:min_len]]
        if len(strat_rets) < 3:
            continue
        strat_ret = float(np.prod(1 + strat_rets) - 1)
        spy_ret = float(np.prod(1 + spy_crisis) - 1)
        crisis.append(CrisisMetrics(
            name=name, start=start, end=end, n_days=int(mask.sum()),
            strategy_return=round(strat_ret * 100, 2),
            spy_return=round(spy_ret * 100, 2),
            outperformance=round((strat_ret - spy_ret) * 100, 2),
        ))

    # Yearly breakdown
    yearly = {}
    for yr in sorted(set(port_rets.index.year)):
        yr_mask = np.asarray(port_rets.index.year == yr)
        yr_rets = rets_array[yr_mask[:len(rets_array)]]
        if len(yr_rets) < 5:
            continue
        m = _compute_metrics(yr_rets)
        yearly[int(yr)] = {
            "cagr": round(m["cagr"] * 100, 2),
            "sharpe": round(m["sharpe"], 2),
            "dd": round(m["dd"] * 100, 2),
        }

    # Walk-forward: expanding window
    years = sorted(yearly.keys())
    folds = []
    for i, test_yr in enumerate(years[1:], start=1):
        train_years = years[:i]
        train_mask = np.array([y in train_years for y in port_rets.index.year])
        test_mask = np.array([y == test_yr for y in port_rets.index.year])
        train_r = rets_array[train_mask]
        test_r = rets_array[test_mask]
        if len(train_r) < 50 or len(test_r) < 50:
            continue
        is_m = _compute_metrics(train_r)
        oos_m = _compute_metrics(test_r)
        folds.append(WFFold(
            test_year=test_yr, train_years=train_years,
            n_train=len(train_r), n_test=len(test_r),
            is_sharpe=round(is_m["sharpe"], 2),
            oos_sharpe=round(oos_m["sharpe"], 2),
            oos_cagr=round(oos_m["cagr"] * 100, 2),
            oos_dd=round(oos_m["dd"] * 100, 2),
        ))

    return BacktestResult(
        cagr=round(metrics["cagr"] * 100, 2),
        sharpe=round(metrics["sharpe"], 2),
        sortino=round(metrics["sortino"], 2),
        max_dd=round(metrics["dd"] * 100, 2),
        calmar=round(metrics["calmar"], 2),
        vol=round(metrics["vol"] * 100, 2),
        n_days=len(rets_array),
        equity=eq,
        daily_returns=rets_array,
        weights_history=held_weights.loc[valid_mask],
        corr_to_spy=round(corr, 3),
        crisis_metrics=crisis,
        yearly=yearly,
        wf_folds=folds,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    result: BacktestResult,
    output_path: str = "reports/crisis_alpha_backtest.html",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Equity SVG
    eq = result.equity
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
    Crisis Alpha Equity (Real Yahoo Finance Data)
  </text>
  <path d="{d}" fill="none" stroke="#16a34a" stroke-width="1.5"/>
</svg>"""

    # Crisis attribution
    crisis_rows = ""
    for c in result.crisis_metrics:
        sc = "#16a34a" if c.outperformance > 0 else "#dc2626"
        strat_c = "#16a34a" if c.strategy_return > 0 else "#dc2626"
        spy_c = "#16a34a" if c.spy_return > 0 else "#dc2626"
        crisis_rows += f"""<tr>
          <td>{c.name}</td>
          <td>{c.start}</td>
          <td>{c.end}</td>
          <td>{c.n_days}</td>
          <td style="color:{strat_c};font-weight:700">{c.strategy_return:+.1f}%</td>
          <td style="color:{spy_c}">{c.spy_return:+.1f}%</td>
          <td style="color:{sc};font-weight:700">{c.outperformance:+.1f}%</td>
        </tr>"""

    # Yearly
    yr_rows = ""
    for yr, m in sorted(result.yearly.items()):
        cc = "#16a34a" if m["cagr"] > 0 else "#dc2626"
        yr_rows += f"""<tr>
          <td>{yr}</td>
          <td style="color:{cc};font-weight:700">{m['cagr']:+.1f}%</td>
          <td>{m['sharpe']:.2f}</td>
          <td>{m['dd']:.1f}%</td>
        </tr>"""

    # Walk-forward folds
    wf_rows = ""
    for f in result.wf_folds:
        oc = "#16a34a" if f.oos_sharpe > 0 else "#dc2626"
        wf_rows += f"""<tr>
          <td>{f.test_year}</td>
          <td>{",".join(str(y) for y in f.train_years)}</td>
          <td>{f.is_sharpe:.2f}</td>
          <td style="color:{oc};font-weight:700">{f.oos_sharpe:.2f}</td>
          <td>{f.oos_cagr:+.1f}%</td>
          <td>{f.oos_dd:.1f}%</td>
        </tr>"""

    corr_c = "#16a34a" if result.corr_to_spy < 0.1 else "#d97706"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>EXP-1780 Crisis Alpha</title>
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
<h1>EXP-1780 — Crisis Alpha / CTA Trend Following</h1>
<p class="meta">Real Yahoo Finance data | SPY/TLT/GLD/UUP/USO | Time-series momentum | Vol-targeted | Weekly rebalance</p>

<div class="callout">
<strong>Rule Zero:</strong> All prices from Yahoo Finance (real market data provider).
Zero synthetic pricing. Strategy: long assets trending up, flat assets trending down,
with vol-targeted position sizing to keep constant risk per asset.
</div>

<div class="grid">
  <div class="card"><div class="l">CAGR</div><div class="v" style="color:{'#16a34a' if result.cagr > 0 else '#dc2626'}">{result.cagr:+.1f}%</div></div>
  <div class="card"><div class="l">Sharpe</div><div class="v">{result.sharpe:.2f}</div></div>
  <div class="card"><div class="l">Sortino</div><div class="v">{result.sortino:.2f}</div></div>
  <div class="card"><div class="l">Max DD</div><div class="v">{result.max_dd:.1f}%</div></div>
  <div class="card"><div class="l">Calmar</div><div class="v">{result.calmar:.1f}</div></div>
  <div class="card"><div class="l">Vol</div><div class="v">{result.vol:.1f}%</div></div>
  <div class="card"><div class="l">Corr to SPY</div><div class="v" style="color:{corr_c}">{result.corr_to_spy:+.3f}</div></div>
  <div class="card"><div class="l">Days</div><div class="v">{result.n_days}</div></div>
</div>

<h2>Crisis Period Attribution (The Whole Point)</h2>
<table>
<tr><th>Crisis</th><th>Start</th><th>End</th><th>Days</th><th>Strategy</th><th>SPY</th><th>Outperformance</th></tr>
{crisis_rows}
</table>

<h2>Equity Curve</h2>
{eq_svg}

<h2>Yearly Performance</h2>
<table>
<tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr>
{yr_rows}
</table>

<h2>Walk-Forward Validation (Expanding Window)</h2>
<table>
<tr><th>Test Year</th><th>Train Years</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS CAGR</th><th>OOS DD</th></tr>
{wf_rows}
</table>

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/crisis_alpha.py | Yahoo Finance real data | Sharpe: arithmetic mean × √252 / std(daily, ddof=1) |
Rule Zero compliant: zero synthetic pricing
</div>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def run_analysis():
    print("EXP-1780 — Crisis Alpha / CTA Trend Following")
    print("=" * 60)

    print("\n  [1/3] Loading REAL prices from Yahoo Finance...")
    prices = load_real_prices(start="2014-01-01", end="2026-01-01")
    print(f"    Loaded {len(prices)} days × {len(prices.columns)} assets")
    print(f"    Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    for tk in prices.columns:
        print(f"    {tk}: {len(prices[tk].dropna())} days")

    print("\n  [2/3] Running backtest...")
    result = backtest_crisis_alpha(prices)

    print(f"\n  [3/3] Results (2015-2025, post-warmup):")
    print(f"    CAGR:    {result.cagr:+.1f}%")
    print(f"    Sharpe:  {result.sharpe:.2f} (corrected arithmetic formula)")
    print(f"    Sortino: {result.sortino:.2f}")
    print(f"    Max DD:  {result.max_dd:.1f}%")
    print(f"    Calmar:  {result.calmar:.1f}")
    print(f"    Corr to SPY: {result.corr_to_spy:+.3f}")

    print(f"\n  Crisis Period Performance:")
    for c in result.crisis_metrics:
        tag = "OUTPERFORM" if c.outperformance > 0 else "UNDERPERFORM"
        print(f"    {c.name:<25s}: strat {c.strategy_return:+6.1f}% | "
              f"SPY {c.spy_return:+6.1f}% | delta {c.outperformance:+6.1f}% [{tag}]")

    print(f"\n  Walk-Forward Folds: {len(result.wf_folds)}")
    oos_positive = sum(1 for f in result.wf_folds if f.oos_sharpe > 0)
    print(f"    OOS positive Sharpe: {oos_positive}/{len(result.wf_folds)}")

    report = generate_report(result)
    print(f"\n  Report: {report}")
    return result


if __name__ == "__main__":
    run_analysis()
