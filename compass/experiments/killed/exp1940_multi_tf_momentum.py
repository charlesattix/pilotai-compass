"""
compass/exp1940_multi_tf_momentum.py — EXP-1940 Multi-Timeframe Momentum.

HYPOTHESIS: Combining 1-month, 3-month, and 12-month momentum signals
across SPY/QQQ/IWM/EFA/EEM produces a robust monthly trend-following
alpha stream that is uncorrelated to EXP-1220 (SPY put-credit-spreads).

DATA POLICY (Rule Zero):
  • All ETF prices and dividends are pulled live from Yahoo Finance via
    the v8 chart API (urllib, no third-party SDK required) so the script
    is reproducible from a clean checkout.
  • No synthetic data. No np.random. If a fetch fails the script aborts.
  • Date range: 2015-01-01 → 2025-12-31 (10 calendar years; the warmup
    consumes the first 12 months → first signal is 2016-01-01).

STRATEGY:
  • Universe: SPY, QQQ, IWM, EFA, EEM (5 ETFs).
  • Each month-end:
        s_i = z(ret_1m_i) + z(ret_3m_i) + z(ret_12m_i)
    where z() is the cross-sectional z-score across the 5 ETFs.
  • Two variants are evaluated:
        long_only        — long top 3 equal-weight (no short)
        long_short       — long top 2, short bottom 2 (4-name dollar-neutral)
  • Rebalance once per month at the close of the last business day.
  • Returns are computed on the next month using REAL daily Yahoo closes
    (no look-ahead — signal date < holding period).

WALK-FORWARD VALIDATION:
  • Expanding window. The first OOS year is 2017 (fits 2016 in-sample
    only as warmup). Each subsequent year extends the in-sample by one
    year and re-evaluates the strategy on the next OOS year.
  • Since the strategy has no fitted parameters (the rule is fixed),
    the walk-forward acts as a robustness audit by reporting per-year
    OOS metrics rather than a single in-sample number.

OUTPUT:
  • compass/reports/exp1940_multi_tf_momentum.json — full results
    (per-variant metrics, walk-forward, correlation to EXP-1220, monthly
    return series, holdings history)

USAGE:
    python -m compass.exp1940_multi_tf_momentum
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp1940_multi_tf_momentum.json"

UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"
TRADING_DAYS = 252


# ═══════════════════════════════════════════════════════════════════════════
# Real data loader (Yahoo v8 chart API)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_yahoo_close(symbol: str, start: str, end: str) -> pd.Series:
    """Daily ADJUSTED closes (dividend + split adjusted) from Yahoo."""
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    safe = symbol.replace("^", "%5E")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}"
           f"?period1={start_ts}&period2={end_ts}&interval=1d&events=div%7Csplit")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    indicators = result["indicators"]
    # Prefer adjclose so total-return effects (dividends) are included.
    if "adjclose" in indicators and indicators["adjclose"]:
        closes = indicators["adjclose"][0]["adjclose"]
    else:
        closes = indicators["quote"][0]["close"]
    dates = [datetime.fromtimestamp(t).date() for t in ts]
    s = pd.Series(closes, index=pd.DatetimeIndex(dates), name=symbol).dropna()
    return s[~s.index.duplicated(keep="last")]


def load_universe() -> pd.DataFrame:
    print(f"  Loading {len(UNIVERSE)} ETFs from Yahoo {START_DATE} → {END_DATE}")
    series = {}
    for sym in UNIVERSE:
        s = fetch_yahoo_close(sym, START_DATE, END_DATE)
        print(f"    {sym:5s} {len(s):5d} bars  "
              f"{s.index.min().date()} → {s.index.max().date()}")
        series[sym] = s
    df = pd.DataFrame(series).sort_index().ffill().dropna()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Signal: cross-sectional momentum z-score composite
# ═══════════════════════════════════════════════════════════════════════════

def month_end_index(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Last business day of each calendar month present in the index."""
    ends = []
    seen = set()
    for d in prices.index:
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
    # collect last bar per (year, month)
    by_month: Dict[Tuple[int, int], pd.Timestamp] = {}
    for d in prices.index:
        by_month[(d.year, d.month)] = d
    return pd.DatetimeIndex(sorted(by_month.values()))


def momentum_features(prices: pd.DataFrame,
                       month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-rebalance composite z-score across 1m / 3m / 12m momentum."""
    rows = []
    me_list = list(month_ends)
    for i, dt in enumerate(me_list):
        if i < 12:
            continue
        # last 1, 3, 12 month-end bars
        d_1 = me_list[i - 1]
        d_3 = me_list[i - 3]
        d_12 = me_list[i - 12]

        ret_1 = prices.loc[dt] / prices.loc[d_1] - 1.0
        ret_3 = prices.loc[dt] / prices.loc[d_3] - 1.0
        ret_12 = prices.loc[dt] / prices.loc[d_12] - 1.0

        def zscore(s: pd.Series) -> pd.Series:
            mu = s.mean()
            sd = s.std(ddof=0)
            return (s - mu) / sd if sd > 1e-12 else s * 0.0

        comp = zscore(ret_1) + zscore(ret_3) + zscore(ret_12)
        for sym in UNIVERSE:
            rows.append({
                "rebalance_date": dt,
                "symbol": sym,
                "ret_1m": float(ret_1[sym]),
                "ret_3m": float(ret_3[sym]),
                "ret_12m": float(ret_12[sym]),
                "composite_z": float(comp[sym]),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Backtest: monthly holdings → daily return stream
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Holding:
    rebalance_date: pd.Timestamp
    next_rebalance: pd.Timestamp
    weights: Dict[str, float]


def build_holdings(features: pd.DataFrame,
                     month_ends: pd.DatetimeIndex,
                     variant: str) -> List[Holding]:
    """Pick top/bottom names each month and assign weights.

    variant ∈ {"long_only", "long_short"}
        long_only:  long top-3, equal-weight 1/3 each (gross 1.0)
        long_short: long top-2 +0.5 each, short bottom-2 -0.5 each
                    (gross 2.0, net 0.0)
    """
    me_list = list(month_ends)
    by_date: Dict[pd.Timestamp, pd.DataFrame] = {
        dt: g.copy() for dt, g in features.groupby("rebalance_date")
    }

    holdings: List[Holding] = []
    sorted_dates = sorted(by_date.keys())
    for i, dt in enumerate(sorted_dates):
        g = by_date[dt].sort_values("composite_z", ascending=False).reset_index(drop=True)
        weights: Dict[str, float] = {sym: 0.0 for sym in UNIVERSE}
        if variant == "long_only":
            for sym in g["symbol"].iloc[:3]:
                weights[sym] = 1.0 / 3.0
        elif variant == "long_short":
            for sym in g["symbol"].iloc[:2]:
                weights[sym] = 0.5
            for sym in g["symbol"].iloc[-2:]:
                weights[sym] = -0.5
        else:
            raise ValueError(f"unknown variant {variant!r}")

        # Holding period = until next rebalance date
        try:
            idx = me_list.index(dt)
            next_dt = me_list[idx + 1] if idx + 1 < len(me_list) else dt
        except ValueError:
            next_dt = dt
        holdings.append(Holding(rebalance_date=dt,
                                  next_rebalance=next_dt,
                                  weights=weights))
    return holdings


def daily_return_stream(prices: pd.DataFrame,
                          holdings: List[Holding]) -> pd.Series:
    """Stitch monthly holdings into a daily portfolio return series."""
    rets = prices.pct_change().fillna(0.0)
    out = pd.Series(0.0, index=rets.index, name="port")
    for h in holdings:
        # Hold from day AFTER rebalance (no look-ahead) through next rebalance
        start_idx = rets.index.searchsorted(h.rebalance_date) + 1
        end_idx = rets.index.searchsorted(h.next_rebalance)
        if start_idx >= len(rets) or end_idx <= start_idx:
            continue
        slice_dates = rets.index[start_idx:end_idx + 1]
        period = rets.loc[slice_dates]
        weight_vec = pd.Series(h.weights)
        port = period[UNIVERSE].mul(weight_vec, axis=1).sum(axis=1)
        out.loc[slice_dates] = port.values
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def metrics(daily: pd.Series) -> Dict:
    rets = daily.dropna().values
    n = len(rets)
    if n < 5:
        return {"n_days": n, "cagr_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "vol_pct": 0.0,
                "win_rate_pct": 0.0, "best_day_pct": 0.0, "worst_day_pct": 0.0}
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    years = n / TRADING_DAYS
    total = float(np.prod(1.0 + rets) - 1.0)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    cum = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return {
        "n_days": n,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(float(dd.min()) * 100, 2),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 2),
        "win_rate_pct": round(float((rets > 0).mean()) * 100, 1),
        "best_day_pct": round(float(np.max(rets)) * 100, 2),
        "worst_day_pct": round(float(np.min(rets)) * 100, 2),
    }


def walk_forward_yearly(daily: pd.Series) -> List[Dict]:
    out = []
    for yr in sorted(set(d.year for d in daily.index)):
        slice_ = daily[daily.index.year == yr]
        if len(slice_) < 20:
            continue
        m = metrics(slice_)
        out.append({"year": yr, **m})
    return out


def expanding_window_validation(daily: pd.Series) -> List[Dict]:
    """Expanding-window OOS audit. For each year Y from 2017 onward, the
    in-sample window is daily[<Y] and the OOS window is daily[Y]. We
    report metrics on each OOS window separately.
    """
    out = []
    years = sorted(set(d.year for d in daily.index))
    for yr in years:
        if yr < 2017:
            continue
        is_slice = daily[daily.index.year < yr]
        oos_slice = daily[daily.index.year == yr]
        if len(is_slice) < 50 or len(oos_slice) < 20:
            continue
        out.append({
            "year": yr,
            "in_sample_years": sorted(set(d.year for d in is_slice.index)),
            "in_sample": metrics(is_slice),
            "oos": metrics(oos_slice),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Correlation to EXP-1220 (best effort)
# ═══════════════════════════════════════════════════════════════════════════

def correlation_to_exp1220(daily: pd.Series) -> Optional[float]:
    try:
        from scripts.ultimate_portfolio import load_exp1220_dynamic
        exp1220 = load_exp1220_dynamic()
        common = daily.index.intersection(exp1220.index)
        if len(common) < 60:
            return None
        a = daily.reindex(common).fillna(0).values
        b = exp1220.reindex(common).fillna(0).values
        # Restrict to days where EITHER series is active
        mask = (np.abs(a) > 1e-9) | (np.abs(b) > 1e-9)
        if mask.sum() < 30:
            return None
        c = float(np.corrcoef(a[mask], b[mask])[0, 1])
        return None if math.isnan(c) else round(c, 4)
    except Exception as e:
        print(f"  EXP-1220 unavailable for correlation: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-1940 — Multi-Timeframe Momentum (SPY/QQQ/IWM/EFA/EEM)")
    print("=" * 72)

    print("\n[1/5] Loading real data from Yahoo Finance...")
    prices = load_universe()
    print(f"  {len(prices)} aligned daily bars  "
          f"{prices.index.min().date()} → {prices.index.max().date()}")

    print("\n[2/5] Computing month-end signals...")
    month_ends = month_end_index(prices)
    feats = momentum_features(prices, month_ends)
    print(f"  {feats['rebalance_date'].nunique()} rebalance dates "
          f"(first {feats['rebalance_date'].min().date()}, "
          f"last {feats['rebalance_date'].max().date()})")

    results: Dict[str, Dict] = {}
    print("\n[3/5] Running variants...")
    for variant in ["long_only", "long_short"]:
        print(f"\n  → variant = {variant}")
        holdings = build_holdings(feats, month_ends, variant)
        daily = daily_return_stream(prices, holdings)
        m = metrics(daily)
        wf = walk_forward_yearly(daily)
        ew = expanding_window_validation(daily)
        corr = correlation_to_exp1220(daily)
        print(f"    CAGR={m['cagr_pct']}%  Sharpe={m['sharpe']}  "
              f"DD={m['max_dd_pct']}%  vol={m['vol_pct']}%")
        if corr is not None:
            print(f"    ρ(EXP-1220) = {corr:+.3f}")
        results[variant] = {
            "metrics": m,
            "walk_forward_yearly": wf,
            "expanding_window": ew,
            "correlation_to_exp1220": corr,
            "n_holdings": len(holdings),
            "first_rebalance": str(holdings[0].rebalance_date.date()) if holdings else None,
            "last_rebalance": str(holdings[-1].rebalance_date.date()) if holdings else None,
            # Compact monthly returns for downstream
            "monthly_returns": [
                {"month": str(d.to_period("M")), "ret": round(float(v), 6)}
                for d, v in (daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)).items()
            ],
            # Holdings history (compact: only nonzero weights)
            "holdings_history": [
                {
                    "rebalance": str(h.rebalance_date.date()),
                    "weights": {k: round(v, 4) for k, v in h.weights.items() if v != 0},
                }
                for h in holdings
            ],
        }

    print("\n[4/5] Picking best variant by Sharpe...")
    best_name = max(results.keys(), key=lambda k: results[k]["metrics"]["sharpe"])
    bm = results[best_name]["metrics"]
    print(f"  WINNER: {best_name}  Sharpe={bm['sharpe']}  CAGR={bm['cagr_pct']}%  "
          f"DD={bm['max_dd_pct']}%")
    bcorr = results[best_name]["correlation_to_exp1220"]
    targets = {
        "sharpe_ge_2": bm["sharpe"] >= 2.0,
        "uncorrelated": bcorr is not None and abs(bcorr) < 0.20,
    }
    print(f"  Target check: Sharpe≥2 = {'PASS' if targets['sharpe_ge_2'] else 'FAIL'}  "
          f"|ρ|<0.20 = {'PASS' if targets['uncorrelated'] else 'FAIL/NA'}")

    print("\n[5/5] Writing JSON report...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "EXP-1940",
        "title": "Multi-Timeframe Momentum (SPY/QQQ/IWM/EFA/EEM)",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": {
            "universe": UNIVERSE,
            "start": str(prices.index.min().date()),
            "end": str(prices.index.max().date()),
            "n_bars": int(len(prices)),
            "source": "Yahoo Finance v8 chart API (adjclose, dividend+split adjusted)",
        },
        "signal": {
            "spec": "z(ret_1m) + z(ret_3m) + z(ret_12m) cross-sectional",
            "rebalance": "monthly (last business day of month)",
            "warmup_months": 12,
        },
        "variants": results,
        "best_variant": best_name,
        "target_check": targets,
        "rule_zero": (
            "All prices from Yahoo Finance v8 chart API. No synthetic data, "
            "no random fills, no extrapolation. If a Yahoo fetch fails the "
            "script aborts."
        ),
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  → {REPORT_JSON}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
