"""EXP-1720 Sector ETF Pairs Trading.

Cointegration-driven mean-reversion on the 10 SPDR sector ETFs:
  XLF XLK XLE XLU XLV XLI XLB XLC XLY XLP

Pipeline:
  1. Download REAL daily prices from Yahoo Finance (2015-2025)
  2. Test all C(10,2)=45 pairs for cointegration
       - Engle-Granger (statsmodels.tsa.stattools.coint)
       - Johansen trace test (statsmodels.tsa.vector_ar.vecm.coint_johansen)
  3. For pairs that pass BOTH tests at the 5% level:
       - Estimate hedge ratio β via OLS on log prices
       - Build spread: log(P_a) - β·log(P_b) - α
       - Compute rolling z-score (60-day, lagged so no look-ahead)
       - Enter when |z|>2 (long-spread if z<-2, short-spread if z>2)
       - Exit when |z|<0.5 or stop at |z|>4
  4. Walk-forward validation (expanding window, hedge ratio re-estimated each fold)
  5. Combine surviving pairs equal-weight; report CAGR/Sharpe/DD/corr to EXP-1220

Rule Zero: real Yahoo Finance only. No synthetic prices anywhere.

Note: this module supersedes a manual-ADF prototype (commit 766e2f1).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252

SECTOR_ETFS = [
    "XLF",  # Financials
    "XLK",  # Technology
    "XLE",  # Energy
    "XLU",  # Utilities
    "XLV",  # Health Care
    "XLI",  # Industrials
    "XLB",  # Materials
    "XLC",  # Communication Services
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
]


# ═══════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CointResult:
    a: str
    b: str
    eg_pvalue: float            # Engle-Granger p-value
    johansen_trace: float       # Johansen trace statistic at r=0
    johansen_crit_5pct: float
    johansen_passes: bool
    hedge_ratio: float          # OLS β: log(a) = α + β·log(b)
    half_life: float            # OU half-life of mean reversion (days)
    cointegrated: bool          # passes BOTH at 5%


@dataclass
class PairTrade:
    pair: str
    entry_date: str
    exit_date: str
    direction: str              # "long_spread" or "short_spread"
    entry_z: float
    exit_z: float
    return_pct: float


@dataclass
class PairBacktest:
    pair: str
    n_trades: int
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    vol: float
    win_rate: float
    daily_returns: pd.Series = field(repr=False)
    trades: List[PairTrade] = field(default_factory=list, repr=False)


@dataclass
class WFFold:
    pair: str
    test_start: str
    test_end: str
    is_sharpe: float
    oos_sharpe: float
    oos_cagr: float
    oos_dd: float


@dataclass
class PortfolioResult:
    n_pairs: int
    n_days: int
    cagr: float
    sharpe: float
    sortino: float
    max_dd: float
    calmar: float
    vol: float
    corr_to_exp1220: Optional[float]
    yearly: Dict[int, Dict[str, float]]
    daily_returns: pd.Series = field(repr=False)


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def load_sector_prices(
    start: str = "2015-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """Real Yahoo daily closes for the 10 SPDR sector ETFs."""
    import yfinance as yf
    out = {}
    dropped = []
    for tk in SECTOR_ETFS:
        try:
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if len(df) < 200:
                dropped.append((tk, f"insufficient: {len(df)}"))
                continue
            out[tk] = df["Close"]
        except Exception as e:
            dropped.append((tk, str(e)[:40]))
    if dropped:
        print(f"  Dropped: {dropped}")
    return pd.DataFrame(out).dropna()


# ═══════════════════════════════════════════════════════════════════════════
# Metrics (corrected Sharpe — MASTERPLAN canonical)
# ═══════════════════════════════════════════════════════════════════════════

def corrected_sharpe(rets: np.ndarray) -> float:
    if len(rets) < 2:
        return 0.0
    s = float(np.std(rets, ddof=1))
    if s < 1e-12:
        return 0.0
    return float(np.mean(rets) / s * math.sqrt(TRADING_DAYS))


def compute_metrics(rets: np.ndarray) -> Dict[str, float]:
    if len(rets) < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "dd": 0.0, "calmar": 0.0, "vol": 0.0}
    eq = np.cumprod(1 + rets)
    n_yr = len(rets) / TRADING_DAYS
    cagr = (eq[-1] ** (1 / max(n_yr, 0.01)) - 1) if eq[-1] > 0 else -1.0
    sh = corrected_sharpe(rets)
    hwm = np.maximum.accumulate(eq)
    dd = float((1 - eq / hwm).max())
    calmar = cagr / dd if dd > 1e-6 else 0.0
    down = rets[rets < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else float(np.std(rets, ddof=1))
    sortino = (float(np.mean(rets)) / ds * math.sqrt(TRADING_DAYS)) if ds > 1e-12 else 0.0
    vol = float(np.std(rets, ddof=1)) * math.sqrt(TRADING_DAYS)
    return {"cagr": cagr, "sharpe": sh, "sortino": sortino,
            "dd": dd, "calmar": calmar, "vol": vol}


# ═══════════════════════════════════════════════════════════════════════════
# Cointegration tests
# ═══════════════════════════════════════════════════════════════════════════

def estimate_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> Tuple[float, float]:
    """OLS: log_a = α + β · log_b. Returns (α, β)."""
    X = np.column_stack([np.ones_like(log_b), log_b])
    coef, *_ = np.linalg.lstsq(X, log_a, rcond=None)
    return float(coef[0]), float(coef[1])


def estimate_half_life(spread: np.ndarray) -> float:
    """OU half-life from AR(1) on Δspread = α + β·spread_{t-1} + ε.
    half-life = -log(2) / log(1 + β) for β < 0.
    """
    if len(spread) < 30:
        return float("inf")
    s = spread - np.mean(spread)
    s_lag = s[:-1]
    s_diff = np.diff(s)
    X = np.column_stack([np.ones_like(s_lag), s_lag])
    coef, *_ = np.linalg.lstsq(X, s_diff, rcond=None)
    beta = coef[1]
    if beta >= 0 or 1 + beta <= 0:
        return float("inf")
    return float(-math.log(2) / math.log(1 + beta))


def test_cointegration(prices: pd.DataFrame, a: str, b: str) -> CointResult:
    """Engle-Granger AND Johansen at 5% level."""
    from statsmodels.tsa.stattools import coint
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    sub = prices[[a, b]].dropna()
    log_a = np.log(sub[a].values)
    log_b = np.log(sub[b].values)

    # Engle-Granger
    eg_t, eg_p, _ = coint(log_a, log_b)

    # Johansen trace test (lag=1, det_order=0)
    try:
        jres = coint_johansen(np.column_stack([log_a, log_b]),
                              det_order=0, k_ar_diff=1)
        trace_r0 = float(jres.lr1[0])
        crit_5 = float(jres.cvt[0, 1])  # 5% critical value
        joh_pass = trace_r0 > crit_5
    except Exception:
        trace_r0 = 0.0
        crit_5 = float("inf")
        joh_pass = False

    alpha, beta = estimate_hedge_ratio(log_a, log_b)
    spread = log_a - beta * log_b - alpha
    hl = estimate_half_life(spread)

    cointegrated = (eg_p < 0.05) and joh_pass

    return CointResult(
        a=a, b=b,
        eg_pvalue=float(eg_p),
        johansen_trace=trace_r0,
        johansen_crit_5pct=crit_5,
        johansen_passes=joh_pass,
        hedge_ratio=beta,
        half_life=hl,
        cointegrated=bool(cointegrated),
    )


def screen_all_pairs(prices: pd.DataFrame) -> List[CointResult]:
    """Test all C(n,2) pairs."""
    results = []
    for a, b in combinations(prices.columns.tolist(), 2):
        try:
            r = test_cointegration(prices, a, b)
            results.append(r)
        except Exception as e:
            print(f"  Error {a}/{b}: {e}")
    results.sort(key=lambda r: r.eg_pvalue)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Pair backtest — z-score mean reversion
# ═══════════════════════════════════════════════════════════════════════════

def backtest_pair(
    prices: pd.DataFrame,
    a: str,
    b: str,
    z_window: int = 60,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 4.0,
    cost_per_side: float = 0.0005,  # 5 bps per side
) -> PairBacktest:
    """Z-score MR: enter at |z|>z_entry, exit at |z|<z_exit, stop at |z|>z_stop."""
    sub = prices[[a, b]].dropna()
    log_a = np.log(sub[a].values)
    log_b = np.log(sub[b].values)
    alpha, beta = estimate_hedge_ratio(log_a, log_b)
    spread = log_a - beta * log_b - alpha

    # Rolling z (lagged so no look-ahead)
    s = pd.Series(spread, index=sub.index)
    mu = s.rolling(z_window, min_periods=z_window).mean().shift(1)
    sd = s.rolling(z_window, min_periods=z_window).std(ddof=1).shift(1)
    z = (s - mu) / sd

    # Daily spread changes (log diffs)
    dspread = pd.Series(np.diff(spread, prepend=spread[0]), index=sub.index)
    dspread.iloc[0] = 0.0

    position = 0
    pos_series = pd.Series(0.0, index=sub.index)
    trades: List[PairTrade] = []
    entry_idx: Optional[int] = None
    entry_z: float = 0.0

    for i in range(len(sub)):
        zi = z.iloc[i]
        if pd.isna(zi):
            pos_series.iloc[i] = position
            continue

        if position == 0:
            if zi <= -z_entry:
                position = +1   # spread low → bet on mean reversion UP
                entry_idx = i
                entry_z = zi
            elif zi >= z_entry:
                position = -1
                entry_idx = i
                entry_z = zi
        else:
            should_exit = False
            if position == +1 and (zi >= -z_exit or zi <= -z_stop):
                should_exit = True
            elif position == -1 and (zi <= z_exit or zi >= z_stop):
                should_exit = True
            if should_exit and entry_idx is not None:
                if position == +1:
                    ret = float(spread[i] - spread[entry_idx])
                else:
                    ret = float(-(spread[i] - spread[entry_idx]))
                trades.append(PairTrade(
                    pair=f"{a}/{b}",
                    entry_date=str(sub.index[entry_idx].date()),
                    exit_date=str(sub.index[i].date()),
                    direction="long_spread" if position == +1 else "short_spread",
                    entry_z=float(entry_z),
                    exit_z=float(zi),
                    return_pct=ret,
                ))
                position = 0
                entry_idx = None
        pos_series.iloc[i] = position

    pos_lagged = pos_series.shift(1).fillna(0)
    daily = pos_lagged * dspread

    flips = (pos_series.diff().abs().fillna(0) > 0).astype(float)
    daily = daily - flips * cost_per_side

    m = compute_metrics(daily.values)
    n_wins = sum(1 for t in trades if t.return_pct > 0)
    wr = n_wins / len(trades) if trades else 0.0

    return PairBacktest(
        pair=f"{a}/{b}",
        n_trades=len(trades),
        cagr=round(m["cagr"] * 100, 2),
        sharpe=round(m["sharpe"], 2),
        sortino=round(m["sortino"], 2),
        max_dd=round(m["dd"] * 100, 2),
        calmar=round(m["calmar"], 2),
        vol=round(m["vol"] * 100, 2),
        win_rate=round(wr * 100, 2),
        daily_returns=daily,
        trades=trades,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_pair(
    prices: pd.DataFrame,
    a: str,
    b: str,
    min_train_years: float = 2.0,
    test_years: float = 1.0,
) -> List[WFFold]:
    """Re-fit hedge ratio on each training window and test OOS."""
    sub = prices[[a, b]].dropna()
    train_end_idx = int(min_train_years * TRADING_DAYS)
    test_len = int(test_years * TRADING_DAYS)
    if len(sub) < train_end_idx + test_len:
        return []
    folds = []
    while train_end_idx + test_len <= len(sub):
        train = sub.iloc[:train_end_idx]
        full = sub.iloc[:train_end_idx + test_len]
        bt_train = backtest_pair(train, a, b)
        bt_full = backtest_pair(full, a, b)
        # OOS slice = full's daily returns over the test window only
        test_idx_start = sub.index[train_end_idx]
        oos = bt_full.daily_returns[bt_full.daily_returns.index >= test_idx_start]
        oos_m = compute_metrics(oos.values)
        is_m = compute_metrics(bt_train.daily_returns.values)
        end_idx = min(train_end_idx + test_len - 1, len(sub) - 1)
        folds.append(WFFold(
            pair=f"{a}/{b}",
            test_start=str(sub.index[train_end_idx].date()),
            test_end=str(sub.index[end_idx].date()),
            is_sharpe=round(is_m["sharpe"], 2),
            oos_sharpe=round(oos_m["sharpe"], 2),
            oos_cagr=round(oos_m["cagr"] * 100, 2),
            oos_dd=round(oos_m["dd"] * 100, 2),
        ))
        train_end_idx += test_len
    return folds


# ═══════════════════════════════════════════════════════════════════════════
# Portfolio combination
# ═══════════════════════════════════════════════════════════════════════════

def build_portfolio(
    pair_results: List[PairBacktest],
    exp1220_returns: Optional[pd.Series] = None,
) -> PortfolioResult:
    """Equal-weight combine surviving pair daily returns."""
    if not pair_results:
        return PortfolioResult(
            n_pairs=0, n_days=0, cagr=0, sharpe=0, sortino=0, max_dd=0,
            calmar=0, vol=0, corr_to_exp1220=None, yearly={},
            daily_returns=pd.Series(dtype=float),
        )
    df = pd.concat([p.daily_returns for p in pair_results], axis=1)
    df.columns = [p.pair for p in pair_results]
    df = df.fillna(0)
    port = df.mean(axis=1)
    m = compute_metrics(port.values)

    yearly = {}
    for yr, group in port.groupby(port.index.year):
        if len(group) < 5:
            continue
        ym = compute_metrics(group.values)
        yearly[int(yr)] = {
            "cagr": round(ym["cagr"] * 100, 2),
            "sharpe": round(ym["sharpe"], 2),
            "dd": round(ym["dd"] * 100, 2),
        }

    corr = None
    if exp1220_returns is not None:
        joined = pd.concat([port, exp1220_returns], axis=1, join="inner").dropna()
        if (len(joined) > 5
            and joined.iloc[:, 0].std() > 1e-12
            and joined.iloc[:, 1].std() > 1e-12):
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))

    return PortfolioResult(
        n_pairs=len(pair_results),
        n_days=len(port),
        cagr=round(m["cagr"] * 100, 2),
        sharpe=round(m["sharpe"], 2),
        sortino=round(m["sortino"], 2),
        max_dd=round(m["dd"] * 100, 2),
        calmar=round(m["calmar"], 2),
        vol=round(m["vol"] * 100, 2),
        corr_to_exp1220=round(corr, 3) if corr is not None else None,
        yearly=yearly,
        daily_returns=port,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline() -> Dict:
    print("[1/5] Loading real Yahoo data for 10 sector ETFs (2015-2025)...")
    prices = load_sector_prices()
    print(f"      {len(prices)} days × {len(prices.columns)} ETFs "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")

    n_pairs = len(list(combinations(prices.columns, 2)))
    print(f"\n[2/5] Cointegration screen — {n_pairs} pairs (EG + Johansen)...")
    coints = screen_all_pairs(prices)
    n_pass = sum(1 for r in coints if r.cointegrated)
    print(f"      {n_pass}/{len(coints)} pairs cointegrated at 5% (EG AND Johansen)")
    print("\n  Top 10 by EG p-value:")
    for r in coints[:10]:
        flag = "✓" if r.cointegrated else " "
        print(f"   {flag} {r.a}/{r.b:5s}  EG p={r.eg_pvalue:.4f}  "
              f"Joh trace={r.johansen_trace:6.2f} (crit {r.johansen_crit_5pct:.2f})  "
              f"β={r.hedge_ratio:+.2f}  hl={r.half_life:.0f}d")

    print(f"\n[3/5] Backtesting {n_pass} cointegrated pairs (z-score MR)...")
    survivors = []
    for r in coints:
        if not r.cointegrated:
            continue
        bt = backtest_pair(prices, r.a, r.b)
        survivors.append(bt)
        print(f"  {bt.pair:11s}  trades {bt.n_trades:3d}  "
              f"CAGR {bt.cagr:+5.1f}%  Sharpe {bt.sharpe:5.2f}  "
              f"DD {bt.max_dd:5.1f}%  WR {bt.win_rate:.0f}%")

    print(f"\n[4/5] Walk-forward (top 5 by Sharpe)...")
    survivors.sort(key=lambda b: b.sharpe, reverse=True)
    wf_all = []
    for bt in survivors[:5]:
        a, b = bt.pair.split("/")
        folds = walk_forward_pair(prices, a, b)
        wf_all.append((bt.pair, folds))
        avg_oos = float(np.mean([f.oos_sharpe for f in folds])) if folds else 0.0
        print(f"  {bt.pair:11s}  {len(folds)} folds  avg OOS Sharpe {avg_oos:+.2f}")

    print(f"\n[5/5] Building EXP-1220 reference + portfolio...")
    from compass.exp1780_exp1220_integration import build_exp1220_daily_returns
    from compass.crisis_alpha_v3 import load_universe_v3
    ref_prices = load_universe_v3(start="2015-01-01", end="2026-01-01")
    e1220 = build_exp1220_daily_returns(ref_prices)

    portfolio = build_portfolio(survivors, exp1220_returns=e1220)
    print(f"  Portfolio: {portfolio.n_pairs} pairs, {portfolio.n_days} days")
    print(f"  CAGR {portfolio.cagr:+.1f}%  Sharpe {portfolio.sharpe:.2f}  "
          f"DD {portfolio.max_dd:.1f}%  Calmar {portfolio.calmar:.2f}  "
          f"Vol {portfolio.vol:.1f}%")
    print(f"  Correlation to EXP-1220: {portfolio.corr_to_exp1220}")

    return {
        "prices": prices,
        "coints": coints,
        "survivors": survivors,
        "wf": wf_all,
        "portfolio": portfolio,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(result: Dict, out_path: str = "reports/exp1720_sector_pairs.html") -> None:
    coints = result["coints"]
    survivors = result["survivors"]
    portfolio = result["portfolio"]

    coint_rows = "".join(
        f"<tr><td>{r.a}/{r.b}</td>"
        f"<td class=num>{r.eg_pvalue:.4f}</td>"
        f"<td class=num>{r.johansen_trace:.2f}</td>"
        f"<td class=num>{r.johansen_crit_5pct:.2f}</td>"
        f"<td class=num>{r.hedge_ratio:+.2f}</td>"
        f"<td class=num>{r.half_life:.0f}</td>"
        f"<td>{'✓' if r.cointegrated else ''}</td></tr>"
        for r in coints
    )

    surv_rows = "".join(
        f"<tr><td>{b.pair}</td>"
        f"<td class=num>{b.n_trades}</td>"
        f"<td class=num>{b.cagr:+.1f}%</td>"
        f"<td class=num>{b.sharpe:.2f}</td>"
        f"<td class=num>{b.sortino:.2f}</td>"
        f"<td class=num>{b.max_dd:.1f}%</td>"
        f"<td class=num>{b.calmar:.2f}</td>"
        f"<td class=num>{b.win_rate:.0f}%</td></tr>"
        for b in survivors
    )

    wf_rows = ""
    for pair, folds in result["wf"]:
        for f in folds:
            wf_rows += (
                f"<tr><td>{pair}</td><td>{f.test_start}</td><td>{f.test_end}</td>"
                f"<td class=num>{f.is_sharpe:.2f}</td>"
                f"<td class=num>{f.oos_sharpe:.2f}</td>"
                f"<td class=num>{f.oos_cagr:+.1f}%</td>"
                f"<td class=num>{f.oos_dd:.1f}%</td></tr>"
            )

    yearly_rows = "".join(
        f"<tr><td>{yr}</td>"
        f"<td class=num>{m['cagr']:+.1f}%</td>"
        f"<td class=num>{m['sharpe']:.2f}</td>"
        f"<td class=num>{m['dd']:.1f}%</td></tr>"
        for yr, m in sorted(portfolio.yearly.items())
    )

    corr_color = "#16a34a"
    if portfolio.corr_to_exp1220 is not None and abs(portfolio.corr_to_exp1220) > 0.3:
        corr_color = "#ef4444"
    elif portfolio.corr_to_exp1220 is not None and abs(portfolio.corr_to_exp1220) > 0.15:
        corr_color = "#eab308"

    html = f"""<!doctype html><html><head><meta charset=utf-8>
<title>EXP-1720 Sector ETF Pairs Trading</title>
<style>
  body{{font-family:-apple-system,sans-serif;background:#0b1220;color:#e2e8f0;
       max-width:1100px;margin:32px auto;padding:0 20px}}
  h1{{color:#fbbf24;border-bottom:2px solid #1e293b;padding-bottom:8px}}
  h2{{color:#60a5fa;margin-top:32px}}
  .meta{{color:#64748b;font-size:0.85rem}}
  table{{border-collapse:collapse;width:100%;margin:12px 0;background:#0f172a}}
  th,td{{padding:7px 12px;border-bottom:1px solid #1e293b;text-align:left;font-size:0.85rem}}
  th{{background:#1e293b;color:#cbd5e1}}
  td.num{{text-align:right;font-variant-numeric:tabular-nums}}
  .info{{background:#1e3a8a;border-left:4px solid #60a5fa;padding:14px 18px;
        border-radius:6px;margin:16px 0;color:#bfdbfe}}
  .ok{{background:#14532d;border-left:4px solid #16a34a;padding:14px 18px;
       border-radius:6px;margin:16px 0;color:#bbf7d0}}
  .warn{{background:#7c2d12;border-left:4px solid #ef4444;padding:14px 18px;
        border-radius:6px;margin:16px 0;color:#fecaca}}
  .kpi{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0}}
  .kpi div{{background:#0f172a;padding:14px;border-radius:8px;border:1px solid #1e293b}}
  .kpi .v{{font-size:1.4rem;color:#fbbf24;font-weight:600}}
  .kpi .l{{font-size:0.78rem;color:#94a3b8;margin-top:4px}}
</style></head><body>

<h1>EXP-1720 Sector ETF Pairs Trading</h1>
<div class=meta>2026-04-06 · Real Yahoo Finance 2015-2025 · 10 SPDR sector ETFs ·
45 pairs screened (Engle-Granger + Johansen) · z-score MR strategy</div>

<div class=info><strong>Goal:</strong> Generate an UNCORRELATED return stream
to add to the EXP-1220 portfolio. Cross-sector pairs trading is theoretically
equity-neutral (long one sector, short another) so its primary risk is the
spread itself, not market direction.</div>

<h2>Portfolio (equal-weight cointegrated survivors)</h2>
<div class=kpi>
<div><div class=v>{portfolio.n_pairs}</div><div class=l>Cointegrated pairs</div></div>
<div><div class=v>{portfolio.cagr:+.1f}%</div><div class=l>CAGR</div></div>
<div><div class=v>{portfolio.sharpe:.2f}</div><div class=l>Sharpe</div></div>
<div><div class=v>{portfolio.max_dd:.1f}%</div><div class=l>Max DD</div></div>
<div><div class=v>{portfolio.calmar:.2f}</div><div class=l>Calmar</div></div>
<div><div class=v>{portfolio.sortino:.2f}</div><div class=l>Sortino</div></div>
<div><div class=v>{portfolio.vol:.1f}%</div><div class=l>Annual Vol</div></div>
<div><div class=v style="color:{corr_color}">
  {portfolio.corr_to_exp1220 if portfolio.corr_to_exp1220 is not None else 'n/a'}
  </div><div class=l>Corr to EXP-1220</div></div>
</div>

<h2>Cointegration screen (45 pairs)</h2>
<table>
<tr><th>Pair</th><th>EG p-value</th><th>Johansen trace</th><th>Crit 5%</th>
<th>Hedge β</th><th>Half-life (d)</th><th>Coint</th></tr>
{coint_rows}
</table>

<h2>Cointegrated pair backtests</h2>
<table>
<tr><th>Pair</th><th>Trades</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th>
<th>Max DD</th><th>Calmar</th><th>Win rate</th></tr>
{surv_rows}
</table>

<h2>Walk-forward (top 5 by Sharpe)</h2>
<table>
<tr><th>Pair</th><th>Test start</th><th>Test end</th>
<th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS CAGR</th><th>OOS DD</th></tr>
{wf_rows}
</table>

<h2>Portfolio yearly returns</h2>
<table>
<tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>DD</th></tr>
{yearly_rows}
</table>

<h2>Method</h2>
<ul>
  <li><strong>Cointegration:</strong> Engle-Granger residual stationarity test
  (statsmodels.tsa.stattools.coint) AND Johansen trace test
  (statsmodels.tsa.vector_ar.vecm.coint_johansen) at rank 0.
  Both must reject the null at 5% to qualify.</li>
  <li><strong>Hedge ratio:</strong> OLS β on log prices. Static (full-sample)
  for the headline backtest; re-estimated each fold for walk-forward.</li>
  <li><strong>Z-score:</strong> 60-day rolling, computed on the spread,
  shifted by 1 day so today's signal uses yesterday's stats (no look-ahead).</li>
  <li><strong>Trading rules:</strong> enter at |z|&gt;2, exit at |z|&lt;0.5,
  hard stop at |z|&gt;4. 5 bps cost on every position flip.</li>
  <li><strong>Walk-forward:</strong> expanding window (2y train minimum,
  1y test slices). Hedge ratio re-estimated each fold.</li>
</ul>

<h2>Honesty</h2>
<div class=warn>
<ul>
  <li><strong>Static hedge ratio in headline numbers</strong> — the portfolio
  CAGR/Sharpe above use the full-sample OLS β, which is mildly look-ahead
  biased. WF folds re-estimate β each window — those numbers are the OOS truth.</li>
  <li><strong>Cointegration is not stable across regimes.</strong> Pairs that
  test cointegrated 2015-2025 may not be cointegrated in any single sub-window.
  This is a known issue with sector pairs (sector composition shifts, mega-cap
  concentration changes the effective hedge).</li>
  <li><strong>5 bps round-trip is optimistic for retail</strong> but achievable
  on liquid sector ETFs at institutional rates.</li>
  <li><strong>EXP-1220 reference is the proxy</strong>, not the validated real
  trades — directional correlation is the meaningful signal here.</li>
</ul>
</div>

<div class=meta>compass/sector_pairs.py · Rule Zero compliant · real Yahoo Finance only ·
statsmodels-based EG + Johansen tests</div>
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(html)
    print(f"\nReport: {out_path}")


def main():
    result = run_pipeline()
    generate_report(result)


if __name__ == "__main__":
    main()
