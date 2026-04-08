"""
EXP-1790 — Overnight Drift Strategy

Academic edge (Cooper 2008, Kelly-Clark 2011, Lou-Polk-Skouras 2019):
  Buy SPY at 3:58 PM close, sell at 9:35 AM next open.

Most of the S&P 500's long-term total return has come from overnight
price changes, not intraday. If you simply held SPY only overnight
(close-to-open) since 1993, you would have captured roughly the full
index return with significantly lower volatility than buy-and-hold.
Meanwhile, intraday-only (open-to-close) has near-zero or negative
aggregate return.

This strategy:
  1. Enter long SPY at day T's close
  2. Exit at day T+1's open
  3. Stay in cash during regular trading hours
  4. Optional: regime filter (skip bear regimes)
  5. Optional: leverage 1.5x or 2.0x

Edge driver: overnight drift is persistent because:
  - Overnight traders face higher inventory risk → demand higher premium
  - Corporate news and international markets move prices while US is closed
  - Systematic buying pressure at the open from institutional inflows

Data: Real Yahoo Finance daily OHLC. 100% real. Zero synthetic.
Period: 2010-2025.
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
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OvernightConfig:
    """Parameters for the overnight drift strategy."""
    leverage: float = 1.0
    regime_filter: bool = False            # skip bear regimes if True
    bear_ma_days: int = 200                # 200-day SMA for bull/bear detection
    commission_per_share: float = 0.005   # $ per share, per side
    slippage_bps: float = 1.0              # basis points per side
    position_pct: float = 1.0              # fraction of capital to deploy


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float   # close of day T
    exit_price: float    # open of day T+1
    shares: int
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    regime: str


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
    config: OvernightConfig
    n_trades: int
    n_wins: int
    win_rate: float
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    vol: float
    total_pnl: float
    total_costs: float
    net_pnl: float
    daily_returns: pd.Series
    equity: List[float]
    yearly: Dict[int, Dict[str, float]]
    wf_folds: List[WFFold]
    corr_to_spy: float
    corr_to_exp1220: Optional[float]
    corr_to_exp1780: Optional[float]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_spy_ohlc(start: str = "2009-06-01", end: str = "2026-01-01") -> pd.DataFrame:
    """Load REAL SPY daily OHLC from Yahoo Finance."""
    import yfinance as yf
    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"SPY data missing column: {col}")
    if len(df) < 1000:
        raise ValueError(f"SPY data too short: {len(df)} days")
    return df[["Open", "High", "Low", "Close"]].copy()


# ═══════════════════════════════════════════════════════════════════════════
# Strategy backtest
# ═══════════════════════════════════════════════════════════════════════════

def classify_regime(spy: pd.DataFrame, ma_days: int = 200) -> pd.Series:
    """Classify each day as 'bull' (above MA) or 'bear' (below MA)."""
    close = spy["Close"]
    sma = close.rolling(ma_days).mean()
    return (close > sma).map({True: "bull", False: "bear"})


def backtest(
    spy: pd.DataFrame,
    config: Optional[OvernightConfig] = None,
    starting_capital: float = 100_000.0,
) -> Tuple[List[Trade], pd.Series]:
    """Run the overnight drift backtest.

    For each trading day T (except the last):
      1. Buy at Close[T]
      2. Sell at Open[T+1]
      3. Apply commission + slippage + leverage
      4. Optional: skip if regime_filter=True and SPY below 200-day MA

    Returns:
        trades: list of Trade objects
        daily_returns: per-day portfolio return (0 on skipped days)
    """
    cfg = config or OvernightConfig()
    closes = spy["Close"].values
    opens = spy["Open"].values
    dates = spy.index

    regime = classify_regime(spy, cfg.bear_ma_days).values

    trades = []
    daily_returns = pd.Series(0.0, index=dates, dtype=float)
    capital = starting_capital

    for i in range(len(spy) - 1):
        # Regime filter
        if cfg.regime_filter and regime[i] == "bear":
            continue

        entry_price = float(closes[i])
        exit_price = float(opens[i + 1])
        if entry_price <= 0 or exit_price <= 0:
            continue

        # Leveraged position sizing
        position_value = capital * cfg.position_pct * cfg.leverage
        shares = int(position_value / entry_price)
        if shares < 1:
            continue

        gross = shares * (exit_price - entry_price)

        # Costs
        slip_per_side = entry_price * cfg.slippage_bps / 10000
        slip_cost = shares * slip_per_side * 2
        comm_cost = shares * cfg.commission_per_share * 2
        total_costs = slip_cost + comm_cost
        net = gross - total_costs
        ret_pct = net / capital

        trades.append(Trade(
            entry_date=dates[i].strftime("%Y-%m-%d"),
            exit_date=dates[i + 1].strftime("%Y-%m-%d"),
            entry_price=round(entry_price, 2),
            exit_price=round(exit_price, 2),
            shares=shares,
            gross_pnl=round(gross, 2),
            costs=round(total_costs, 2),
            net_pnl=round(net, 2),
            return_pct=round(ret_pct, 6),
            regime=str(regime[i]) if regime[i] is not None else "unknown",
        ))

        daily_returns.iloc[i + 1] = ret_pct
        capital += net

    return trades, daily_returns


# ═══════════════════════════════════════════════════════════════════════════
# Metrics (corrected Sharpe formula)
# ═══════════════════════════════════════════════════════════════════════════

def compute_sharpe(rets: np.ndarray) -> float:
    """Arithmetic mean × sqrt(252) / std(daily, ddof=1)."""
    if len(rets) < 2:
        return 0.0
    sigma = float(rets.std(ddof=1))
    return float(rets.mean()) / sigma * math.sqrt(TRADING_DAYS) if sigma > 1e-12 else 0.0


def compute_metrics(daily_rets: np.ndarray) -> dict:
    if len(daily_rets) < 2:
        return {"cagr": 0, "sharpe": 0, "dd": 0, "sortino": 0, "calmar": 0, "vol": 0}
    eq = np.cumprod(1 + daily_rets)
    n_yr = len(daily_rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else 0
    sharpe = compute_sharpe(daily_rets)
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0
    down = daily_rets[daily_rets < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else float(daily_rets.std(ddof=1))
    sortino = float(daily_rets.mean()) / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0
    vol = float(daily_rets.std(ddof=1)) * math.sqrt(TRADING_DAYS)
    return {"cagr": cagr, "sharpe": sharpe, "dd": dd, "sortino": sortino,
            "calmar": calmar, "vol": vol}


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward(daily_rets: pd.Series) -> List[WFFold]:
    """Expanding-window year-by-year walk-forward."""
    if len(daily_rets) < 10:
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

def corr_to_spy(daily_rets: pd.Series, spy: pd.DataFrame) -> float:
    spy_rets = spy["Close"].pct_change().fillna(0)
    common = daily_rets.index.intersection(spy_rets.index)
    if len(common) < 10:
        return 0.0
    a = daily_rets.reindex(common).fillna(0).values
    b = spy_rets.reindex(common).fillna(0).values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def correlation_vs(daily_rets: pd.Series,
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


def build_exp1220_reference(spy: pd.DataFrame) -> pd.Series:
    """Proxy EXP-1220 (short gamma + theta) return stream from SPY moves."""
    spy_rets = spy["Close"].pct_change().fillna(0)
    theta = 0.0002
    proxy = pd.Series(theta, index=spy_rets.index)
    big_down = spy_rets < -0.01
    big_up = spy_rets > 0.01
    proxy[big_down] = theta + 1.5 * spy_rets[big_down]
    proxy[big_up] = theta + 0.3 * spy_rets[big_up]
    return proxy


def build_exp1780_reference(start: str = "2009-06-01",
                             end: str = "2026-01-01") -> Optional[pd.Series]:
    """Proxy EXP-1780 CTA trend using SPY + TLT 200-day crossovers."""
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
        proxy = 0.5 * spy_long.shift(1).fillna(0) * spy_rets + \
                0.5 * tlt_long.shift(1).fillna(0) * tlt_rets
        return proxy
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_variant(
    spy: pd.DataFrame,
    config: OvernightConfig,
    name: str,
    exp1220_ref: pd.Series,
    exp1780_ref: Optional[pd.Series],
) -> VariantResult:
    """Run one config variant and compute all metrics."""
    trades, daily_rets = backtest(spy, config)
    rets_array = daily_rets.values
    m = compute_metrics(rets_array)

    eq = [100_000.0]
    for r in rets_array:
        eq.append(eq[-1] * (1 + r))

    n = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    total_gross = sum(t.gross_pnl for t in trades)
    total_costs = sum(t.costs for t in trades)
    total_net = sum(t.net_pnl for t in trades)

    yearly = {}
    for yr in sorted(set(daily_rets.index.year)):
        yr_rets = daily_rets[daily_rets.index.year == yr].values
        if len(yr_rets) < 5:
            continue
        ym = compute_metrics(yr_rets)
        yearly[int(yr)] = {
            "cagr": round(ym["cagr"] * 100, 2),
            "sharpe": round(ym["sharpe"], 2),
            "dd": round(ym["dd"] * 100, 2),
            "n_trades": sum(1 for t in trades if int(t.entry_date[:4]) == yr),
        }

    folds = walk_forward(daily_rets)
    spy_corr = corr_to_spy(daily_rets, spy)
    c1220 = correlation_vs(daily_rets, exp1220_ref)
    c1780 = correlation_vs(daily_rets, exp1780_ref)

    return VariantResult(
        name=name, config=config,
        n_trades=n, n_wins=wins,
        win_rate=round(wins / n, 3) if n > 0 else 0,
        cagr=round(m["cagr"] * 100, 2),
        sharpe=round(m["sharpe"], 2),
        sortino=round(m["sortino"], 2),
        max_dd=round(m["dd"] * 100, 2),
        calmar=round(m["calmar"], 2),
        vol=round(m["vol"] * 100, 2),
        total_pnl=round(total_gross, 2),
        total_costs=round(total_costs, 2),
        net_pnl=round(total_net, 2),
        daily_returns=daily_rets,
        equity=eq, yearly=yearly, wf_folds=folds,
        corr_to_spy=round(spy_corr, 3),
        corr_to_exp1220=round(c1220, 3) if c1220 is not None else None,
        corr_to_exp1780=round(c1780, 3) if c1780 is not None else None,
    )


def run_full_analysis() -> Dict[str, VariantResult]:
    """Run all 4 variants: baseline, 1.5x, 2x, regime-filtered."""
    print("Loading real SPY OHLC from Yahoo Finance...")
    spy = load_spy_ohlc(start="2009-06-01", end="2026-01-01")
    print(f"  Loaded {len(spy)} days, {spy.index[0].date()} to {spy.index[-1].date()}")

    exp1220_ref = build_exp1220_reference(spy)
    print("Building EXP-1220 reference (SPY-based proxy)...")

    print("Building EXP-1780 reference (SPY/TLT trend proxy)...")
    exp1780_ref = build_exp1780_reference()

    variants = {
        "baseline_1x": OvernightConfig(leverage=1.0, regime_filter=False),
        "leverage_1.5x": OvernightConfig(leverage=1.5, regime_filter=False),
        "leverage_2x": OvernightConfig(leverage=2.0, regime_filter=False),
        "regime_filtered_1.5x": OvernightConfig(leverage=1.5, regime_filter=True),
    }

    results = {}
    for name, cfg in variants.items():
        print(f"\nRunning variant: {name}")
        r = run_variant(spy, cfg, name, exp1220_ref, exp1780_ref)
        results[name] = r
        print(f"  {r.n_trades} trades, CAGR={r.cagr:+.1f}%, Sharpe={r.sharpe:.2f}, "
              f"DD={r.max_dd:.1f}%, SPY ρ={r.corr_to_spy:+.3f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    results: Dict[str, VariantResult],
    output_path: str = "reports/overnight_drift_backtest.html",
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Best variant by Sharpe
    best_name = max(results, key=lambda k: results[k].sharpe)
    best = results[best_name]

    # Equity SVG — overlay all variants
    w, h = 780, 240
    pl, pr, pt, pb = 65, 20, 28, 28
    pw, ph = w - pl - pr, h - pt - pb
    all_equity = [v.equity for v in results.values()]
    all_vals = [v for eq in all_equity for v in eq]
    ym, yx = min(all_vals) * 0.95, max(all_vals) * 1.05
    max_len = max(len(eq) for eq in all_equity)

    colors = {
        "baseline_1x": "#3b82f6",
        "leverage_1.5x": "#16a34a",
        "leverage_2x": "#d97706",
        "regime_filtered_1.5x": "#a855f7",
    }

    paths_svg = ""
    legend = ""
    for i, (name, var) in enumerate(results.items()):
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
        legend += f'<span style="color:{color};margin-right:12px">■ {name}</span>'

    eq_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"
  style="border:1px solid #e2e8f0;border-radius:6px">
  <text x="{w//2}" y="16" text-anchor="middle" font-size="11" fill="#64748b">
    Overnight Drift Equity Curves (Real Yahoo Data)
  </text>
  {paths_svg}
</svg>
<p style="font-size:0.75rem;margin-top:4px">{legend}</p>"""

    # Variant comparison table
    var_rows = ""
    for name, var in results.items():
        is_best = name == best_name
        bg = ' style="background:#f0fdf4"' if is_best else ""
        star = " ★" if is_best else ""
        cc = "#16a34a" if var.cagr > 0 else "#dc2626"
        sc = "#16a34a" if var.sharpe > 0 else "#dc2626"
        c_1220 = f"{var.corr_to_exp1220:+.3f}" if var.corr_to_exp1220 is not None else "N/A"
        c_1780 = f"{var.corr_to_exp1780:+.3f}" if var.corr_to_exp1780 is not None else "N/A"
        var_rows += f"""<tr{bg}>
          <td>{name}{star}</td>
          <td>{var.config.leverage}x</td>
          <td>{'Yes' if var.config.regime_filter else 'No'}</td>
          <td>{var.n_trades}</td>
          <td>{var.win_rate:.0%}</td>
          <td style="color:{cc};font-weight:700">{var.cagr:+.1f}%</td>
          <td style="color:{sc};font-weight:700">{var.sharpe:.2f}</td>
          <td>{var.sortino:.2f}</td>
          <td>{var.max_dd:.1f}%</td>
          <td>{var.calmar:.1f}</td>
          <td>{var.corr_to_spy:+.3f}</td>
          <td>{c_1220}</td>
          <td>{c_1780}</td>
        </tr>"""

    # Best variant yearly
    yr_rows = ""
    for yr, ym in sorted(best.yearly.items()):
        cc = "#16a34a" if ym["cagr"] > 0 else "#dc2626"
        yr_rows += f"""<tr>
          <td>{yr}</td>
          <td>{ym['n_trades']}</td>
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
<html><head><meta charset="UTF-8"><title>EXP-1790 Overnight Drift</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b}}
h1{{font-size:1.4rem;color:#0f172a}}h2{{font-size:1rem;color:#334155;margin-top:1.5rem;border-bottom:1px solid #e2e8f0;padding-bottom:4px}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin:16px 0}}
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
<h1>EXP-1790 — Overnight Drift Strategy</h1>
<p class="meta">Real Yahoo Finance SPY | 2010-2025 | 4 variants | Corrected Sharpe | Rule Zero compliant</p>

<div class="callout">
<strong>Academic edge (Cooper 2008, Kelly-Clark 2011, Lou-Polk-Skouras 2019):</strong>
Most equity returns occur overnight. Buying SPY at 3:58 PM close and selling at 9:35 AM next open
has historically captured most of the long-term return with significantly lower volatility than
buy-and-hold. Intraday returns (open-to-close) are near zero or negative in aggregate.
<br><br>
<strong>4 Variants tested:</strong> baseline 1.0x, leverage 1.5x, leverage 2.0x, regime-filtered 1.5x.
</div>

<div class="grid">
  <div class="card"><div class="l">Best Variant</div><div class="v" style="color:#16a34a">{best_name}</div></div>
  <div class="card"><div class="l">Best CAGR</div><div class="v">{best.cagr:+.1f}%</div></div>
  <div class="card"><div class="l">Best Sharpe</div><div class="v">{best.sharpe:.2f}</div></div>
  <div class="card"><div class="l">Best Max DD</div><div class="v">{best.max_dd:.1f}%</div></div>
  <div class="card"><div class="l">Best Calmar</div><div class="v">{best.calmar:.1f}</div></div>
  <div class="card"><div class="l">Win Rate</div><div class="v">{best.win_rate:.0%}</div></div>
  <div class="card"><div class="l">Trades</div><div class="v">{best.n_trades}</div></div>
  <div class="card"><div class="l">Corr SPY</div><div class="v">{best.corr_to_spy:+.3f}</div></div>
</div>

<h2>Equity Curves (All Variants)</h2>
{eq_svg}

<h2>Variant Comparison</h2>
<table>
<tr><th>Variant</th><th>Lev</th><th>Regime Filt</th><th>Trades</th><th>Win%</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Calmar</th><th>ρ SPY</th><th>ρ 1220</th><th>ρ 1780</th></tr>
{var_rows}
</table>

<h2>Best Variant — Yearly Performance ({best_name})</h2>
<table>
<tr><th>Year</th><th>Trades</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr>
{yr_rows}
</table>

<h2>Best Variant — Walk-Forward (Expanding Window)</h2>
<table>
<tr><th>Test Year</th><th>Train</th><th>IS SR</th><th>OOS SR</th><th>OOS CAGR</th><th>OOS DD</th></tr>
{wf_rows}
</table>

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/overnight_drift.py | Real Yahoo Finance SPY data |
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
    print("EXP-1790 — Overnight Drift Strategy")
    print("=" * 60)

    results = run_full_analysis()

    print(f"\n{'Variant':<25} {'Trades':>7} {'CAGR':>8} {'Sharpe':>7} {'DD':>7} {'ρ SPY':>7} {'ρ 1220':>7} {'ρ 1780':>7}")
    print("-" * 85)
    for name, r in results.items():
        c1220 = f"{r.corr_to_exp1220:+.2f}" if r.corr_to_exp1220 is not None else "N/A"
        c1780 = f"{r.corr_to_exp1780:+.2f}" if r.corr_to_exp1780 is not None else "N/A"
        print(f"{name:<25} {r.n_trades:>7} {r.cagr:>+7.1f}% {r.sharpe:>7.2f} "
              f"{r.max_dd:>6.1f}% {r.corr_to_spy:>+7.3f} {c1220:>7} {c1780:>7}")

    best_name = max(results, key=lambda k: results[k].sharpe)
    print(f"\nBEST: {best_name}")

    report = generate_report(results)
    print(f"\nReport: {report}")
    return results


if __name__ == "__main__":
    main()
