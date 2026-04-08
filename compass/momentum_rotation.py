"""
EXP-1830 — Momentum Factor Rotation (Sector ETFs + TLT + GLD)

Academic basis: Jegadeesh & Titman (1993), Asness-Moskowitz-Pedersen (2013),
Moskowitz-Ooi-Pedersen (2012). Cross-sectional momentum persistently earns
a premium across asset classes.

Strategy:
  Universe: 11 SPDR sector ETFs (XLF, XLK, XLE, XLV, XLI, XLU, XLC, XLB,
            XLP, XLY, XLRE) + TLT + GLD = 13 assets
  Signal:   Multi-horizon momentum (3/6/12 month, Asness-style blend)
  Entry:    Monthly rebalance, long top 3, short bottom 3 (long-only variant
            also tested: top 3 only)
  Sizing:   Equal weight within sleeves, leverage 1.0x gross
  Holding:  One month (21 trading days)

Variants tested:
  - long_short_3_3: long top 3, short bottom 3 (market-neutral)
  - long_only_top3: long top 3, cash for rest (lower vol, higher CAGR)
  - long_only_top5: long top 5 (more diversification)

Rule Zero: 100% real Yahoo Finance data. Zero synthetic pricing.
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


# ═══════════════════════════════════════════════════════════════════════════
# Universe
# ═══════════════════════════════════════════════════════════════════════════

# 11 SPDR sector ETFs (2010+ inception)
SECTOR_ETFS = [
    "XLF",   # Financials
    "XLK",   # Technology
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLRE",  # Real Estate (inception 2015)
    "XLC",   # Communication Services (inception 2018)
]

OTHER_ASSETS = ["TLT", "GLD"]  # Bonds + Gold

UNIVERSE = SECTOR_ETFS + OTHER_ASSETS


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MomentumConfig:
    """Parameters for the momentum rotation strategy."""
    lookback_months: List[int] = field(default_factory=lambda: [3, 6, 12])
    lookback_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    n_long: int = 3
    n_short: int = 3           # 0 = long only
    rebalance_days: int = 21   # monthly
    min_assets_for_signal: int = 5  # need this many with valid data
    allow_short: bool = True   # if False, n_short is forced to 0
    gross_leverage: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════


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
class VariantResult:
    name: str
    config: MomentumConfig
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    vol: float
    total_return: float
    n_days: int
    n_rebalances: int
    daily_returns: pd.Series
    equity: List[float]
    yearly: Dict[int, Dict[str, float]]
    wf_folds: List[WFFold]
    corr_to_spy: float
    corr_to_exp1220: Optional[float]
    corr_to_exp1780: Optional[float]
    weight_history: pd.DataFrame


# ═══════════════════════════════════════════════════════════════════════════
# Data loading — real Yahoo Finance
# ═══════════════════════════════════════════════════════════════════════════


def load_universe_prices(
    tickers: Optional[List[str]] = None,
    start: str = "2009-06-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """Load REAL daily adjusted closes for the universe.

    Returns a DataFrame with columns = tickers, index = dates.
    Tickers with insufficient history are dropped (<250 days = 1 year).
    """
    import yfinance as yf
    tickers = tickers or UNIVERSE
    data = {}
    dropped = []
    for tk in tickers:
        try:
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if len(df) < 250:
                dropped.append((tk, len(df)))
                continue
            data[tk] = df["Close"]
        except Exception as e:
            dropped.append((tk, str(e)[:40]))

    if not data:
        raise RuntimeError("No tickers loaded from Yahoo")
    if dropped:
        print(f"  Dropped tickers: {dropped}")

    # Outer join then forward-fill to align
    prices = pd.DataFrame(data).sort_index()
    return prices


# ═══════════════════════════════════════════════════════════════════════════
# Momentum signal
# ═══════════════════════════════════════════════════════════════════════════


def compute_momentum_signal(
    prices: pd.DataFrame,
    lookback_months: List[int],
    weights: List[float],
) -> pd.DataFrame:
    """Blended multi-horizon momentum signal.

    For each asset and date, compute:
      mom[t] = sum_i( w_i * (P[t] / P[t - 21*months_i] - 1) )

    Returns DataFrame same shape as prices with signal values.
    Early rows (before the longest lookback) are NaN.
    """
    if len(lookback_months) != len(weights):
        raise ValueError("lookback_months and weights must have same length")
    if abs(sum(weights) - 1.0) > 0.01:
        raise ValueError(f"weights must sum to 1.0, got {sum(weights)}")

    signal = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    valid_mask = pd.DataFrame(True, index=prices.index, columns=prices.columns)

    for months, w in zip(lookback_months, weights):
        lookback_days = months * 21  # approx trading days per month
        mom = prices.pct_change(lookback_days)
        signal = signal + w * mom.fillna(0)
        valid_mask = valid_mask & mom.notna()

    # Mask out rows where any lookback is NaN
    signal = signal.where(valid_mask)
    return signal


def compute_weights(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    config: MomentumConfig,
) -> pd.DataFrame:
    """Compute portfolio weights from the signal.

    For each row:
      - Rank assets by signal (ignoring NaN)
      - If fewer than min_assets valid, all weights = 0
      - Long the top n_long at +1/n_long
      - Short the bottom n_short at -1/n_short (if allow_short)
      - Scale by gross_leverage
    """
    cfg = config
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for idx in signal.index:
        row = signal.loc[idx].dropna()
        if len(row) < cfg.min_assets_for_signal:
            continue

        ranked = row.sort_values(ascending=False)
        top = ranked.iloc[:cfg.n_long]
        long_w = cfg.gross_leverage * 0.5 if cfg.allow_short and cfg.n_short > 0 else cfg.gross_leverage
        for tk in top.index:
            weights.at[idx, tk] = long_w / cfg.n_long

        if cfg.allow_short and cfg.n_short > 0:
            bottom = ranked.iloc[-cfg.n_short:]
            short_w = cfg.gross_leverage * 0.5
            for tk in bottom.index:
                weights.at[idx, tk] = -short_w / cfg.n_short

    return weights


def apply_rebalance_hold(weights: pd.DataFrame, rebalance_days: int) -> pd.DataFrame:
    """Hold weights for `rebalance_days` then refresh.

    This models a monthly rebalance: sample weights at day 0, 21, 42, ...
    and hold those weights constant in between.
    """
    held = weights.copy()
    last_row = None
    for i, idx in enumerate(held.index):
        if i % rebalance_days == 0 or last_row is None:
            last_row = weights.loc[idx].copy()
        else:
            held.loc[idx] = last_row
    return held


# ═══════════════════════════════════════════════════════════════════════════
# Backtest
# ═══════════════════════════════════════════════════════════════════════════


def backtest(
    prices: pd.DataFrame,
    config: Optional[MomentumConfig] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Run momentum rotation backtest.

    Returns:
        daily_returns: portfolio daily return series
        weight_history: DataFrame of daily weights (after rebalance hold)
    """
    cfg = config or MomentumConfig()

    signal = compute_momentum_signal(prices, cfg.lookback_months, cfg.lookback_weights)
    raw_weights = compute_weights(signal, prices, cfg)
    held_weights = apply_rebalance_hold(raw_weights, cfg.rebalance_days)

    # Shift by 1 day to avoid look-ahead
    lagged = held_weights.shift(1).fillna(0)

    asset_returns = prices.pct_change().fillna(0)
    port_rets = (lagged * asset_returns).sum(axis=1)

    # Skip warmup (longest lookback)
    warmup_days = max(cfg.lookback_months) * 21
    if warmup_days < len(prices):
        valid_idx = prices.index[warmup_days]
        port_rets = port_rets[port_rets.index >= valid_idx]
        held_weights = held_weights[held_weights.index >= valid_idx]

    return port_rets, held_weights


# ═══════════════════════════════════════════════════════════════════════════
# Metrics (corrected Sharpe)
# ═══════════════════════════════════════════════════════════════════════════


def compute_sharpe(rets: np.ndarray) -> float:
    """Arithmetic mean daily × sqrt(252) / std(daily, ddof=1)."""
    if len(rets) < 2:
        return 0.0
    sigma = float(rets.std(ddof=1))
    return float(rets.mean()) / sigma * math.sqrt(TRADING_DAYS) if sigma > 1e-12 else 0.0


def compute_metrics(rets: np.ndarray) -> dict:
    if len(rets) < 2:
        return {"cagr": 0, "sharpe": 0, "dd": 0, "sortino": 0, "calmar": 0, "vol": 0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else -1
    sharpe = compute_sharpe(rets)
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = rets[rets < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else float(rets.std(ddof=1))
    sortino = float(rets.mean()) / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0
    vol = float(rets.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    return {"cagr": cagr, "sharpe": sharpe, "dd": dd, "sortino": sortino,
            "calmar": calmar, "vol": vol}


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward
# ═══════════════════════════════════════════════════════════════════════════


def walk_forward(daily_rets: pd.Series) -> List[WFFold]:
    """Expanding-window year-by-year walk-forward."""
    if len(daily_rets) < 100:
        return []
    years = sorted(set(daily_rets.index.year))
    folds = []
    for test_yr in years[1:]:
        train_years = [y for y in years if y < test_yr]
        train_mask = daily_rets.index.year.isin(train_years)
        test_mask = daily_rets.index.year == test_yr
        train_r = daily_rets[train_mask].values
        test_r = daily_rets[test_mask].values
        if len(train_r) < 50 or len(test_r) < 50:
            continue
        is_m = compute_metrics(train_r)
        oos_m = compute_metrics(test_r)
        folds.append(WFFold(
            test_year=int(test_yr), train_years=train_years,
            n_train=len(train_r), n_test=len(test_r),
            is_sharpe=round(is_m["sharpe"], 2),
            oos_sharpe=round(oos_m["sharpe"], 2),
            oos_cagr=round(oos_m["cagr"] * 100, 2),
            oos_dd=round(oos_m["dd"] * 100, 2),
        ))
    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Correlation helpers
# ═══════════════════════════════════════════════════════════════════════════


def corr_to(daily_rets: pd.Series,
            reference: Optional[pd.Series]) -> Optional[float]:
    if reference is None or len(reference) < 10:
        return None
    common = daily_rets.index.intersection(reference.index)
    if len(common) < 10:
        return None
    a = daily_rets.reindex(common).fillna(0).values
    b = reference.reindex(common).fillna(0).values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def build_exp1220_reference(prices: pd.DataFrame) -> Optional[pd.Series]:
    """Proxy EXP-1220 (short gamma + theta) from SPY moves."""
    if "SPY" not in prices.columns:
        # Try loading SPY separately
        try:
            import yfinance as yf
            spy = yf.download("SPY",
                              start=prices.index[0].strftime("%Y-%m-%d"),
                              end=prices.index[-1].strftime("%Y-%m-%d"),
                              progress=False, auto_adjust=True)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.index = pd.to_datetime(spy.index)
            if spy.index.tz is not None:
                spy.index = spy.index.tz_localize(None)
            spy_rets = spy["Close"].pct_change().fillna(0)
        except Exception:
            return None
    else:
        spy_rets = prices["SPY"].pct_change().fillna(0)

    theta = 0.0002
    proxy = pd.Series(theta, index=spy_rets.index)
    proxy[spy_rets < -0.01] = theta + 1.5 * spy_rets[spy_rets < -0.01]
    proxy[spy_rets > 0.01] = theta + 0.3 * spy_rets[spy_rets > 0.01]
    return proxy


def build_exp1780_reference(start: str, end: str) -> Optional[pd.Series]:
    """Proxy EXP-1780 CTA trend from SPY + TLT trend following."""
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
        tlt = yf.download("TLT", start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        if isinstance(tlt.columns, pd.MultiIndex):
            tlt.columns = tlt.columns.get_level_values(0)
        spy.index = pd.to_datetime(spy.index)
        tlt.index = pd.to_datetime(tlt.index)
        if spy.index.tz is not None:
            spy.index = spy.index.tz_localize(None)
        if tlt.index.tz is not None:
            tlt.index = tlt.index.tz_localize(None)
        spy_ma = spy["Close"].rolling(200).mean()
        tlt_ma = tlt["Close"].rolling(200).mean()
        spy_long = (spy["Close"] > spy_ma).astype(float)
        tlt_long = (tlt["Close"] > tlt_ma).astype(float)
        spy_rets = spy["Close"].pct_change().fillna(0)
        tlt_rets = tlt["Close"].pct_change().fillna(0)
        return (0.5 * spy_long.shift(1).fillna(0) * spy_rets
                + 0.5 * tlt_long.shift(1).fillna(0) * tlt_rets)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Variant runner
# ═══════════════════════════════════════════════════════════════════════════


def run_variant(
    prices: pd.DataFrame,
    config: MomentumConfig,
    name: str,
    exp1220_ref: Optional[pd.Series],
    exp1780_ref: Optional[pd.Series],
) -> VariantResult:
    """Run one variant and compute all metrics."""
    port_rets, weights = backtest(prices, config)
    rets_array = port_rets.values
    m = compute_metrics(rets_array)

    eq = [100_000.0]
    for r in rets_array:
        eq.append(eq[-1] * (1 + r))

    # Yearly
    yearly = {}
    for yr in sorted(set(port_rets.index.year)):
        yr_mask = port_rets.index.year == yr
        yr_rets = port_rets[yr_mask].values
        if len(yr_rets) < 5:
            continue
        ym = compute_metrics(yr_rets)
        yearly[int(yr)] = {
            "cagr": round(ym["cagr"] * 100, 2),
            "sharpe": round(ym["sharpe"], 2),
            "dd": round(ym["dd"] * 100, 2),
        }

    folds = walk_forward(port_rets)

    # Correlations
    if "SPY" in prices.columns:
        spy_rets = prices["SPY"].pct_change().fillna(0)
    else:
        try:
            import yfinance as yf
            spy = yf.download("SPY",
                              start=prices.index[0].strftime("%Y-%m-%d"),
                              end=prices.index[-1].strftime("%Y-%m-%d"),
                              progress=False, auto_adjust=True)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.index = pd.to_datetime(spy.index)
            if spy.index.tz is not None:
                spy.index = spy.index.tz_localize(None)
            spy_rets = spy["Close"].pct_change().fillna(0)
        except Exception:
            spy_rets = None

    spy_corr = corr_to(port_rets, spy_rets) or 0.0
    c1220 = corr_to(port_rets, exp1220_ref)
    c1780 = corr_to(port_rets, exp1780_ref)

    n_rebalances = sum(1 for i in range(len(port_rets)) if i % config.rebalance_days == 0)

    return VariantResult(
        name=name, config=config,
        cagr=round(m["cagr"] * 100, 2),
        sharpe=round(m["sharpe"], 2),
        sortino=round(m["sortino"], 2),
        max_dd=round(m["dd"] * 100, 2),
        calmar=round(m["calmar"], 2),
        vol=round(m["vol"] * 100, 2),
        total_return=round((eq[-1] / eq[0] - 1) * 100, 2),
        n_days=len(rets_array),
        n_rebalances=n_rebalances,
        daily_returns=port_rets,
        equity=eq,
        yearly=yearly,
        wf_folds=folds,
        corr_to_spy=round(spy_corr, 3),
        corr_to_exp1220=round(c1220, 3) if c1220 is not None else None,
        corr_to_exp1780=round(c1780, 3) if c1780 is not None else None,
        weight_history=weights,
    )


def run_full_analysis() -> Dict[str, VariantResult]:
    """Run all 3 variants on real Yahoo data."""
    print("Loading real Yahoo Finance data for 13-asset universe...")
    prices = load_universe_prices(start="2009-06-01", end="2026-01-01")
    print(f"  Loaded {len(prices)} days × {len(prices.columns)} assets")
    print(f"  Assets: {list(prices.columns)}")
    print(f"  Range: {prices.index[0].date()} to {prices.index[-1].date()}")

    print("\nBuilding EXP-1220 reference (SPY short-gamma proxy)...")
    exp1220_ref = build_exp1220_reference(prices)

    print("Building EXP-1780 reference (SPY/TLT trend proxy)...")
    exp1780_ref = build_exp1780_reference(
        start=prices.index[0].strftime("%Y-%m-%d"),
        end=prices.index[-1].strftime("%Y-%m-%d"),
    )

    variants = {
        "long_short_3_3": MomentumConfig(
            n_long=3, n_short=3, allow_short=True,
        ),
        "long_only_top3": MomentumConfig(
            n_long=3, n_short=0, allow_short=False,
        ),
        "long_only_top5": MomentumConfig(
            n_long=5, n_short=0, allow_short=False,
        ),
    }

    results = {}
    for name, cfg in variants.items():
        print(f"\nRunning variant: {name}")
        r = run_variant(prices, cfg, name, exp1220_ref, exp1780_ref)
        results[name] = r
        c1220 = f"{r.corr_to_exp1220:+.3f}" if r.corr_to_exp1220 is not None else "N/A"
        print(f"  CAGR={r.cagr:+.1f}%, Sharpe={r.sharpe:.2f}, DD={r.max_dd:.1f}%, "
              f"ρ SPY={r.corr_to_spy:+.3f}, ρ 1220={c1220}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════


def generate_report(
    results: Dict[str, VariantResult],
    output_path: str = "reports/momentum_rotation_backtest.html",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]

    # Equity SVG — overlay all variants
    w, h = 780, 240
    pl, pr, pt, pb = 65, 20, 28, 28
    pw, ph = w - pl - pr, h - pt - pb
    all_eq = [v.equity for v in results.values()]
    all_vals = [x for eq in all_eq for x in eq]
    ym, yx = min(all_vals) * 0.95, max(all_vals) * 1.05
    max_len = max(len(eq) for eq in all_eq)

    colors = {
        "long_short_3_3": "#3b82f6",
        "long_only_top3": "#16a34a",
        "long_only_top5": "#d97706",
    }

    paths_svg = ""
    legend_items = []
    for name, var in results.items():
        eq = var.equity
        n = len(eq)
        step = max(1, n // 400)
        pts = [(j, eq[j]) for j in range(0, n, step)]
        if pts[-1][0] != n - 1:
            pts.append((n - 1, eq[-1]))

        def tx(x): return pl + x / max(max_len - 1, 1) * pw
        def ty(v): return pt + (1 - (v - ym) / max(yx - ym, 1)) * ph

        d = " ".join(f"{'M' if j == 0 else 'L'}{tx(x):.1f},{ty(v):.1f}"
                     for j, (x, v) in enumerate(pts))
        color = colors.get(name, "#64748b")
        paths_svg += f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.5"/>\n'
        legend_items.append(f'<span style="color:{color};margin-right:12px">■ {name}</span>')

    eq_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"
  style="border:1px solid #e2e8f0;border-radius:6px">
  <text x="{w//2}" y="16" text-anchor="middle" font-size="11" fill="#64748b">
    Momentum Rotation Equity Curves (Real Yahoo Finance Data)
  </text>
  {paths_svg}
</svg>
<p style="font-size:0.75rem;margin-top:4px">{''.join(legend_items)}</p>"""

    # Variant comparison
    var_rows = ""
    for name, var in results.items():
        is_best = name == best_name
        bg = ' style="background:#f0fdf4"' if is_best else ""
        star = " ★" if is_best else ""
        cc = "#16a34a" if var.cagr > 0 else "#dc2626"
        c1220 = f"{var.corr_to_exp1220:+.3f}" if var.corr_to_exp1220 is not None else "N/A"
        c1780 = f"{var.corr_to_exp1780:+.3f}" if var.corr_to_exp1780 is not None else "N/A"
        var_rows += f"""<tr{bg}>
          <td>{name}{star}</td>
          <td>{var.config.n_long}L / {var.config.n_short}S</td>
          <td style="color:{cc};font-weight:700">{var.cagr:+.1f}%</td>
          <td>{var.sharpe:.2f}</td>
          <td>{var.sortino:.2f}</td>
          <td>{var.max_dd:.1f}%</td>
          <td>{var.calmar:.1f}</td>
          <td>{var.vol:.1f}%</td>
          <td>{var.corr_to_spy:+.3f}</td>
          <td>{c1220}</td>
          <td>{c1780}</td>
        </tr>"""

    # Best variant yearly
    yr_rows = ""
    for yr, ym in sorted(best.yearly.items()):
        cc = "#16a34a" if ym["cagr"] > 0 else "#dc2626"
        yr_rows += f"""<tr>
          <td>{yr}</td>
          <td style="color:{cc};font-weight:700">{ym['cagr']:+.1f}%</td>
          <td>{ym['sharpe']:.2f}</td>
          <td>{ym['dd']:.1f}%</td>
        </tr>"""

    # Best variant walk-forward
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

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>EXP-1830 Momentum Rotation</title>
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
<h1>EXP-1830 — Momentum Factor Rotation</h1>
<p class="meta">13-asset universe | Real Yahoo Finance data | 2010-2025 walk-forward | Rule Zero compliant</p>

<div class="callout">
<strong>Academic basis:</strong> Jegadeesh & Titman (1993), Asness-Moskowitz-Pedersen (2013),
Moskowitz-Ooi-Pedersen (2012). Cross-sectional momentum persists across asset classes.
<br><br>
<strong>Universe (13):</strong> 11 SPDR sectors (XLF, XLK, XLE, XLV, XLI, XLU, XLB, XLP, XLY, XLRE, XLC)
+ TLT + GLD.
<br><br>
<strong>Signal:</strong> Blended 3/6/12-month momentum, weights 0.3/0.4/0.3.
Monthly rebalance (21 trading days), 1-day lag to avoid look-ahead.
</div>

<div class="grid">
  <div class="card"><div class="l">Best Variant</div><div class="v" style="color:#16a34a">{best_name}</div></div>
  <div class="card"><div class="l">Best CAGR</div><div class="v">{best.cagr:+.1f}%</div></div>
  <div class="card"><div class="l">Best Sharpe</div><div class="v">{best.sharpe:.2f}</div></div>
  <div class="card"><div class="l">Best Sortino</div><div class="v">{best.sortino:.2f}</div></div>
  <div class="card"><div class="l">Best Max DD</div><div class="v">{best.max_dd:.1f}%</div></div>
  <div class="card"><div class="l">Best Calmar</div><div class="v">{best.calmar:.1f}</div></div>
  <div class="card"><div class="l">Corr SPY</div><div class="v">{best.corr_to_spy:+.3f}</div></div>
  <div class="card"><div class="l">Corr 1220</div><div class="v">{f"{best.corr_to_exp1220:+.3f}" if best.corr_to_exp1220 is not None else "N/A"}</div></div>
</div>

<h2>Equity Curves (All Variants)</h2>
{eq_svg}

<h2>Variant Comparison</h2>
<table>
<tr><th>Variant</th><th>Legs</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>ρ SPY</th><th>ρ 1220</th><th>ρ 1780</th></tr>
{var_rows}
</table>

<h2>Best Variant — Yearly ({best_name})</h2>
<table>
<tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr>
{yr_rows}
</table>

<h2>Best Variant — Walk-Forward (Expanding Window)</h2>
<table>
<tr><th>Test Year</th><th>Train</th><th>IS SR</th><th>OOS SR</th><th>OOS CAGR</th><th>OOS DD</th></tr>
{wf_rows}
</table>

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/momentum_rotation.py | Real Yahoo Finance daily adjusted closes |
Sharpe: arithmetic mean × √252 / std(daily, ddof=1) |
Rule Zero compliant: zero synthetic pricing
</div>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print("EXP-1830 — Momentum Factor Rotation")
    print("=" * 60)

    results = run_full_analysis()

    print(f"\n{'Variant':<20} {'CAGR':>8} {'Sharpe':>7} {'DD':>7} {'Calmar':>7} {'ρ SPY':>7} {'ρ 1220':>7}")
    print("-" * 75)
    for name, r in results.items():
        c1220 = f"{r.corr_to_exp1220:+.2f}" if r.corr_to_exp1220 is not None else "N/A"
        print(f"{name:<20} {r.cagr:>+7.1f}% {r.sharpe:>7.2f} {r.max_dd:>6.1f}% "
              f"{r.calmar:>7.2f} {r.corr_to_spy:>+7.3f} {c1220:>7}")

    best = max(results, key=lambda k: results[k].sharpe)
    print(f"\nBEST: {best}")

    report = generate_report(results)
    print(f"\nReport: {report}")
    return results


if __name__ == "__main__":
    main()
