"""
Intraday Mean Reversion on SPY — Overnight Gap Fade

Edge hypothesis: When SPY opens materially away from the previous close,
the gap tends to partially close during the trading session. Specifically:
  - Gap down > 0.5% at open → BUY at open, SELL at close (long bias)
  - Gap up   > 0.5% at open → SELL at open, BUY at close (short bias)

This is a fundamentally different edge than credit spread selling (theta)
or trend-following (momentum). Mean reversion should be uncorrelated to
both EXP-1220 (vol selling) and EXP-1780 (trend following).

Data: Daily OHLC from Yahoo Finance. 100% real. Zero synthetic.
Period: 2015-2025 with expanding-window walk-forward validation.

Cost model:
  - Commission: $0.005/share × 2 sides = $0.01/share per round trip
  - Slippage: 1 bp (0.01%) per side = 2 bp round trip
  - Total friction: ~$0.02 + 0.02% on a $450 SPY = ~5 bps per trade
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
class MRConfig:
    """Parameters for the intraday mean-reversion strategy."""
    gap_threshold_pct: float = 0.5   # minimum gap to trigger a trade (in %)
    max_gap_pct: float = 5.0          # skip extreme gaps (news events)
    position_pct: float = 0.10        # 10% of capital per trade
    commission_per_share: float = 0.005  # $ per share, per side
    slippage_bps: float = 1.0          # basis points per side
    enable_long: bool = True           # trade gap-down fades
    enable_short: bool = True          # trade gap-up fades


# ═══════════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    date: str
    direction: str      # "long" or "short"
    gap_pct: float      # gap at open (signed, %)
    entry_price: float  # open
    exit_price: float   # close
    shares: int
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float


@dataclass
class WFFold:
    test_year: int
    train_years: List[int]
    n_train_trades: int
    n_test_trades: int
    is_sharpe: float
    oos_sharpe: float
    oos_cagr: float
    oos_dd: float
    oos_wr: float


@dataclass
class BacktestResult:
    trades: List[Trade]
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
    daily_returns: pd.Series       # indexed by date
    equity: List[float]
    yearly: Dict[int, Dict[str, float]]
    wf_folds: List[WFFold]
    corr_to_exp1220: Optional[float]
    corr_to_exp1780: Optional[float]
    corr_to_spy: float


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (real Yahoo Finance)
# ═══════════════════════════════════════════════════════════════════════════

def load_spy_ohlc(start: str = "2014-06-01", end: str = "2026-01-01") -> pd.DataFrame:
    """Load REAL SPY daily OHLC from Yahoo Finance. Zero synthetic."""
    import yfinance as yf
    df = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # Require OHLC columns
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"SPY data missing column: {col}")
    if len(df) < 500:
        raise ValueError(f"SPY data too short: {len(df)} days")
    return df[["Open", "High", "Low", "Close"]].copy()


# ═══════════════════════════════════════════════════════════════════════════
# Strategy backtest
# ═══════════════════════════════════════════════════════════════════════════

def backtest(
    spy: pd.DataFrame,
    config: Optional[MRConfig] = None,
    starting_capital: float = 100_000.0,
) -> Tuple[List[Trade], pd.Series]:
    """Run the intraday mean reversion backtest.

    For each trading day (after day 1):
      1. Compute gap = (Open_t - Close_{t-1}) / Close_{t-1}
      2. If |gap| > gap_threshold AND |gap| < max_gap:
         - gap down → long: buy at Open, sell at Close
         - gap up → short: short at Open, cover at Close
      3. Apply commission + slippage

    Returns:
        trades: list of Trade objects
        daily_returns: pd.Series of per-day portfolio returns (0 on no-trade days)
    """
    cfg = config or MRConfig()
    closes = spy["Close"].values
    opens = spy["Open"].values
    dates = spy.index

    trades = []
    daily_returns = pd.Series(0.0, index=dates, dtype=float)

    capital = starting_capital
    for i in range(1, len(spy)):
        prev_close = float(closes[i - 1])
        today_open = float(opens[i])
        today_close = float(closes[i])
        if prev_close <= 0 or today_open <= 0:
            continue

        gap_pct = (today_open - prev_close) / prev_close * 100

        if abs(gap_pct) < cfg.gap_threshold_pct:
            continue  # no trade
        if abs(gap_pct) > cfg.max_gap_pct:
            continue  # skip news gaps

        if gap_pct < 0 and not cfg.enable_long:
            continue
        if gap_pct > 0 and not cfg.enable_short:
            continue

        # Position sizing: 10% of capital
        position_value = capital * cfg.position_pct
        shares = int(position_value / today_open)
        if shares < 1:
            continue

        direction = "long" if gap_pct < 0 else "short"

        # Gross P&L
        if direction == "long":
            gross = shares * (today_close - today_open)
        else:
            gross = shares * (today_open - today_close)

        # Costs
        slippage_per_side = today_open * cfg.slippage_bps / 10000
        slip_cost = shares * slippage_per_side * 2  # both sides
        comm_cost = shares * cfg.commission_per_share * 2
        total_costs = slip_cost + comm_cost
        net = gross - total_costs
        ret_pct = net / capital

        trades.append(Trade(
            date=dates[i].strftime("%Y-%m-%d"),
            direction=direction,
            gap_pct=round(gap_pct, 3),
            entry_price=round(today_open, 2),
            exit_price=round(today_close, 2),
            shares=shares,
            gross_pnl=round(gross, 2),
            costs=round(total_costs, 2),
            net_pnl=round(net, 2),
            return_pct=round(ret_pct, 6),
        ))

        daily_returns.iloc[i] = ret_pct
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

def walk_forward(trades: List[Trade], daily_rets: pd.Series) -> List[WFFold]:
    """Expanding-window year-by-year walk-forward."""
    if not trades:
        return []

    df = pd.DataFrame([{"date": pd.Timestamp(t.date), "net_pnl": t.net_pnl,
                        "return_pct": t.return_pct} for t in trades])
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    folds = []

    for test_yr in years[1:]:
        train_years = [y for y in years if y < test_yr]
        train_df = df[df["year"].isin(train_years)]
        test_df = df[df["year"] == test_yr]

        if len(train_df) < 10 or len(test_df) < 10:
            continue

        # Use daily_rets restricted to each window for Sharpe computation
        train_mask = daily_rets.index.year.isin(train_years)
        test_mask = daily_rets.index.year == test_yr
        train_r = daily_rets[train_mask].values
        test_r = daily_rets[test_mask].values

        is_m = compute_metrics(train_r)
        oos_m = compute_metrics(test_r)

        folds.append(WFFold(
            test_year=int(test_yr), train_years=train_years,
            n_train_trades=len(train_df), n_test_trades=len(test_df),
            is_sharpe=round(is_m["sharpe"], 2),
            oos_sharpe=round(oos_m["sharpe"], 2),
            oos_cagr=round(oos_m["cagr"] * 100, 2),
            oos_dd=round(oos_m["dd"] * 100, 2),
            oos_wr=round(float((test_df["net_pnl"] > 0).sum()) / len(test_df), 3),
        ))

    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Correlation helpers
# ═══════════════════════════════════════════════════════════════════════════

def corr_to_spy(daily_rets: pd.Series, spy: pd.DataFrame) -> float:
    """Correlation to SPY daily returns."""
    spy_rets = spy["Close"].pct_change().fillna(0)
    common = daily_rets.index.intersection(spy_rets.index)
    if len(common) < 10:
        return 0.0
    a = daily_rets.reindex(common).fillna(0).values
    b = spy_rets.reindex(common).fillna(0).values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def correlation_to_reference(daily_rets: pd.Series,
                              reference_rets: pd.Series) -> Optional[float]:
    """Generic correlation vs a reference series."""
    if reference_rets is None or len(reference_rets) < 10:
        return None
    common = daily_rets.index.intersection(reference_rets.index)
    if len(common) < 10:
        return None
    a = daily_rets.reindex(common).fillna(0).values
    b = reference_rets.reindex(common).fillna(0).values
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def build_exp1220_reference(spy: pd.DataFrame) -> pd.Series:
    """Build a proxy EXP-1220 daily return series from SPY.

    EXP-1220 is a credit spread seller — short-gamma + theta. Its daily
    P&L is approximately a function of SPY daily moves:
      - Small moves (<0.5%): positive (theta decay)
      - Large down moves: negative (short-gamma hit)
      - Large up moves: slightly positive (premium erodes normally)
    """
    spy_rets = spy["Close"].pct_change().fillna(0)
    # Short-gamma payoff approximation
    theta = 0.0002  # ~5%/yr base theta
    proxy = pd.Series(theta, index=spy_rets.index)
    big_down = spy_rets < -0.01
    big_up = spy_rets > 0.01
    # Losses scale ~1.5x the underlying move on big down days
    proxy[big_down] = theta + 1.5 * spy_rets[big_down]
    # Small gains on big up days (premium captured faster)
    proxy[big_up] = theta + 0.3 * spy_rets[big_up]
    return proxy


def build_exp1780_reference(start: str = "2014-06-01",
                             end: str = "2026-01-01") -> Optional[pd.Series]:
    """Build EXP-1780 reference from real trend-following signal.

    Uses a simple SPY + TLT trend proxy: long SPY when above 200-day MA,
    long TLT when bonds trending up. This approximates the CTA trend
    strategy in compass/crisis_alpha.py without re-running the full backtest.
    """
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

        # Trend signal: 200-day MA crossover
        spy_ma = spy["Close"].rolling(200).mean()
        tlt_ma = tlt["Close"].rolling(200).mean()
        spy_long = (spy["Close"] > spy_ma).astype(float)
        tlt_long = (tlt["Close"] > tlt_ma).astype(float)

        spy_rets = spy["Close"].pct_change().fillna(0)
        tlt_rets = tlt["Close"].pct_change().fillna(0)

        # 50/50 trend portfolio with 1-day lag
        proxy = 0.5 * spy_long.shift(1).fillna(0) * spy_rets + \
                0.5 * tlt_long.shift(1).fillna(0) * tlt_rets
        return proxy
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Full analysis pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_full_analysis(config: Optional[MRConfig] = None) -> BacktestResult:
    cfg = config or MRConfig()

    # 1. Load real SPY data
    spy = load_spy_ohlc()

    # 2. Run backtest
    trades, daily_rets = backtest(spy, cfg)

    # 3. Compute metrics
    rets_array = daily_rets.values
    m = compute_metrics(rets_array)

    # 4. Equity curve
    eq = [100_000.0]
    for r in rets_array:
        eq.append(eq[-1] * (1 + r))

    # 5. Trade stats
    n = len(trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    total_gross = sum(t.gross_pnl for t in trades)
    total_costs = sum(t.costs for t in trades)
    total_net = sum(t.net_pnl for t in trades)

    # 6. Yearly breakdown
    yearly = {}
    for yr in sorted(set(daily_rets.index.year)):
        yr_mask = daily_rets.index.year == yr
        yr_rets = daily_rets[yr_mask].values
        if len(yr_rets) < 5:
            continue
        ym = compute_metrics(yr_rets)
        n_trades_yr = sum(1 for t in trades if int(t.date[:4]) == yr)
        yearly[int(yr)] = {
            "cagr": round(ym["cagr"] * 100, 2),
            "sharpe": round(ym["sharpe"], 2),
            "dd": round(ym["dd"] * 100, 2),
            "n_trades": n_trades_yr,
        }

    # 7. Walk-forward
    wf_folds = walk_forward(trades, daily_rets)

    # 8. Correlations
    spy_corr = corr_to_spy(daily_rets, spy)
    exp1220_ref = build_exp1220_reference(spy)
    exp1780_ref = build_exp1780_reference()
    corr_1220 = correlation_to_reference(daily_rets, exp1220_ref)
    corr_1780 = correlation_to_reference(daily_rets, exp1780_ref)

    return BacktestResult(
        trades=trades, n_trades=n, n_wins=wins,
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
        equity=eq,
        yearly=yearly,
        wf_folds=wf_folds,
        corr_to_exp1220=round(corr_1220, 3) if corr_1220 is not None else None,
        corr_to_exp1780=round(corr_1780, 3) if corr_1780 is not None else None,
        corr_to_spy=round(spy_corr, 3),
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(
    result: BacktestResult,
    output_path: str = "reports/intraday_mr_backtest.html",
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
    Intraday MR Equity Curve (Real Yahoo Data)
  </text>
  <path d="{d}" fill="none" stroke="#16a34a" stroke-width="1.5"/>
</svg>"""

    # Yearly table
    yr_rows = ""
    for yr, ym in sorted(result.yearly.items()):
        cc = "#16a34a" if ym["cagr"] > 0 else "#dc2626"
        yr_rows += f"""<tr>
          <td>{yr}</td>
          <td>{ym['n_trades']}</td>
          <td style="color:{cc};font-weight:700">{ym['cagr']:+.1f}%</td>
          <td>{ym['sharpe']:.2f}</td>
          <td>{ym['dd']:.1f}%</td>
        </tr>"""

    # Walk-forward folds
    wf_rows = ""
    for f in result.wf_folds:
        oc = "#16a34a" if f.oos_sharpe > 0 else "#dc2626"
        wf_rows += f"""<tr>
          <td>{f.test_year}</td>
          <td>{len(f.train_years)}y ({f.n_train_trades} trades)</td>
          <td>{f.n_test_trades}</td>
          <td>{f.is_sharpe:.2f}</td>
          <td style="color:{oc};font-weight:700">{f.oos_sharpe:.2f}</td>
          <td>{f.oos_cagr:+.1f}%</td>
          <td>{f.oos_dd:.1f}%</td>
          <td>{f.oos_wr:.0%}</td>
        </tr>"""

    # Long vs short breakdown
    longs = [t for t in result.trades if t.direction == "long"]
    shorts = [t for t in result.trades if t.direction == "short"]
    long_pnl = sum(t.net_pnl for t in longs)
    short_pnl = sum(t.net_pnl for t in shorts)
    long_wr = sum(1 for t in longs if t.net_pnl > 0) / max(len(longs), 1)
    short_wr = sum(1 for t in shorts if t.net_pnl > 0) / max(len(shorts), 1)

    # Correlation colors
    def _corr_color(c):
        if c is None:
            return "#64748b"
        if abs(c) < 0.15:
            return "#16a34a"
        if abs(c) < 0.30:
            return "#d97706"
        return "#dc2626"

    corr_1220_display = f"{result.corr_to_exp1220:+.3f}" if result.corr_to_exp1220 is not None else "N/A"
    corr_1780_display = f"{result.corr_to_exp1780:+.3f}" if result.corr_to_exp1780 is not None else "N/A"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Intraday MR — Overnight Gap Fade</title>
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
<h1>Intraday Mean Reversion — Overnight Gap Fade on SPY</h1>
<p class="meta">Real Yahoo Finance OHLC | 2015-2025 walk-forward | Corrected Sharpe formula | Rule Zero compliant</p>

<div class="callout">
<strong>Hypothesis:</strong> When SPY gaps materially away from the previous close
(>0.5%), the gap tends to partially close during the trading session.
<br><br>
<strong>Entry logic:</strong> If gap down > 0.5% at open → BUY at open, SELL at close.
If gap up > 0.5% → SHORT at open, COVER at close. Skip gaps > 5% (news events).
<br><br>
<strong>Edge type:</strong> Statistical mean reversion. Different from credit spread theta
(EXP-1220) and trend following (EXP-1780). Should provide diversification.
</div>

<div class="grid">
  <div class="card"><div class="l">CAGR</div><div class="v" style="color:{'#16a34a' if result.cagr > 0 else '#dc2626'}">{result.cagr:+.1f}%</div></div>
  <div class="card"><div class="l">Sharpe</div><div class="v">{result.sharpe:.2f}</div></div>
  <div class="card"><div class="l">Sortino</div><div class="v">{result.sortino:.2f}</div></div>
  <div class="card"><div class="l">Max DD</div><div class="v">{result.max_dd:.1f}%</div></div>
  <div class="card"><div class="l">Calmar</div><div class="v">{result.calmar:.1f}</div></div>
  <div class="card"><div class="l">Trades</div><div class="v">{result.n_trades}</div></div>
  <div class="card"><div class="l">Win Rate</div><div class="v">{result.win_rate:.0%}</div></div>
  <div class="card"><div class="l">Vol</div><div class="v">{result.vol:.1f}%</div></div>
  <div class="card"><div class="l">Corr SPY</div><div class="v" style="color:{_corr_color(result.corr_to_spy)}">{result.corr_to_spy:+.3f}</div></div>
  <div class="card"><div class="l">Corr 1220</div><div class="v" style="color:{_corr_color(result.corr_to_exp1220)}">{corr_1220_display}</div></div>
  <div class="card"><div class="l">Corr 1780</div><div class="v" style="color:{_corr_color(result.corr_to_exp1780)}">{corr_1780_display}</div></div>
</div>

<h2>Equity Curve</h2>
{eq_svg}

<h2>Long vs Short Breakdown</h2>
<table>
<tr><th>Direction</th><th>Trades</th><th>Wins</th><th>Win%</th><th>Net PnL</th></tr>
<tr><td>Long (gap-down fade)</td><td>{len(longs)}</td><td>{sum(1 for t in longs if t.net_pnl > 0)}</td><td>{long_wr:.0%}</td>
<td style="color:{'#16a34a' if long_pnl > 0 else '#dc2626'};font-weight:700">${long_pnl:,.0f}</td></tr>
<tr><td>Short (gap-up fade)</td><td>{len(shorts)}</td><td>{sum(1 for t in shorts if t.net_pnl > 0)}</td><td>{short_wr:.0%}</td>
<td style="color:{'#16a34a' if short_pnl > 0 else '#dc2626'};font-weight:700">${short_pnl:,.0f}</td></tr>
</table>

<h2>Costs Summary</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Gross PnL</td><td style="color:{'#16a34a' if result.total_pnl > 0 else '#dc2626'}">${result.total_pnl:,.0f}</td></tr>
<tr><td>Total Costs (commissions + slippage)</td><td style="color:#dc2626">-${result.total_costs:,.0f}</td></tr>
<tr><td>Net PnL</td><td style="color:{'#16a34a' if result.net_pnl > 0 else '#dc2626'};font-weight:700">${result.net_pnl:,.0f}</td></tr>
</table>

<h2>Yearly Performance</h2>
<table>
<tr><th>Year</th><th>Trades</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr>
{yr_rows}
</table>

<h2>Walk-Forward Validation (Expanding Window)</h2>
<table>
<tr><th>Test Year</th><th>Train</th><th>Test Trades</th><th>IS SR</th><th>OOS SR</th><th>OOS CAGR</th><th>OOS DD</th><th>OOS WR</th></tr>
{wf_rows}
</table>

<h2>Correlation Analysis</h2>
<table>
<tr><th>vs</th><th>Correlation</th><th>Notes</th></tr>
<tr><td>SPY daily returns</td><td style="color:{_corr_color(result.corr_to_spy)};font-weight:700">{result.corr_to_spy:+.3f}</td>
<td>Gap fade has natural hedge: long on down-opens, short on up-opens</td></tr>
<tr><td>EXP-1220 (vol selling proxy)</td><td style="color:{_corr_color(result.corr_to_exp1220)};font-weight:700">{corr_1220_display}</td>
<td>Target: low absolute correlation for diversification</td></tr>
<tr><td>EXP-1780 (CTA trend proxy)</td><td style="color:{_corr_color(result.corr_to_exp1780)};font-weight:700">{corr_1780_display}</td>
<td>Mean reversion is opposite of trend following</td></tr>
</table>

<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/intraday_mr.py | Real Yahoo Finance data | Sharpe: arithmetic mean × √252 / std(daily, ddof=1) |
Rule Zero compliant: zero synthetic pricing
</div>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("Intraday Mean Reversion — Overnight Gap Fade on SPY")
    print("=" * 60)

    print("\n  [1/4] Loading REAL SPY OHLC from Yahoo Finance...")
    spy = load_spy_ohlc()
    print(f"    Loaded {len(spy)} trading days, {spy.index[0].date()} to {spy.index[-1].date()}")

    print("\n  [2/4] Running backtest with default config (0.5% gap threshold)...")
    result = run_full_analysis()
    print(f"    {result.n_trades} trades executed")

    print(f"\n  [3/4] Results:")
    print(f"    CAGR:    {result.cagr:+.1f}%")
    print(f"    Sharpe:  {result.sharpe:.2f} (corrected formula)")
    print(f"    Sortino: {result.sortino:.2f}")
    print(f"    Max DD:  {result.max_dd:.1f}%")
    print(f"    Calmar:  {result.calmar:.1f}")
    print(f"    Win Rate: {result.win_rate:.0%}")
    print(f"    Gross PnL: ${result.total_pnl:,.0f}")
    print(f"    Costs:     ${result.total_costs:,.0f}")
    print(f"    Net PnL:   ${result.net_pnl:,.0f}")

    print(f"\n  Correlations:")
    print(f"    vs SPY:      {result.corr_to_spy:+.3f}")
    print(f"    vs EXP-1220: {result.corr_to_exp1220 if result.corr_to_exp1220 is not None else 'N/A'}")
    print(f"    vs EXP-1780: {result.corr_to_exp1780 if result.corr_to_exp1780 is not None else 'N/A'}")

    print(f"\n  Walk-forward: {len(result.wf_folds)} folds")
    oos_pos = sum(1 for f in result.wf_folds if f.oos_sharpe > 0)
    print(f"    OOS positive Sharpe: {oos_pos}/{len(result.wf_folds)}")

    print("\n  [4/4] Generating report...")
    report = generate_report(result)
    print(f"    Report: {report}")

    return result


if __name__ == "__main__":
    main()
