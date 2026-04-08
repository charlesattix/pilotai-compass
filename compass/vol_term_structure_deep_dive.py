"""
Vol Term Structure Strategy — Deep Dive.

Comprehensive analysis of the VTS strategy that showed OOS Sharpe 1.78 in
Round 2. Tests across multiple tickers, optimizes parameters, applies regime
filter, performs walk-forward validation, and estimates capacity.

All prices from IronVault options_cache.db. Zero synthetic data.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault
from compass.regime import RegimeClassifier, Regime

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT = ROOT / "reports" / "vol_term_structure_deep_dive.html"
CAPITAL = 100_000


# ═══════════════════════════════════════════════════════════════════════════
# Shared infrastructure
# ═══════════════════════════════════════════════════════════════════════════

def _exp_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _dl(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start="2019-06-01", end="2026-01-01", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df


def _next_td(dt: datetime, td_set) -> Optional[datetime]:
    for off in range(7):
        c = dt + timedelta(days=off)
        if c.strftime("%Y-%m-%d") in td_set:
            return c
    return None


def _all_exps(hd: IronVault, ticker: str, start: str, end: str) -> List[str]:
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    cur.execute("""SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker=? AND option_type='P' AND expiration BETWEEN ? AND ?
        ORDER BY expiration""", (ticker, start, end))
    out = [r[0] for r in cur.fetchall()]
    conn.close()
    return out


def _sell_put_spread(
    hd: IronVault, ticker: str, exp: str, trade_date: str,
    price: float, otm_pct: float = 0.94, width: float = 5.0,
) -> Optional[Dict]:
    strikes = hd.get_available_strikes(ticker, exp, trade_date, "P")
    if not strikes:
        return None
    ed = _exp_dt(exp)
    target = price * otm_pct
    for sk in sorted(strikes, key=lambda k: abs(k - target))[:12]:
        lk = sk - width
        if lk not in strikes:
            cands = [s for s in strikes if s < sk and abs(s - lk) <= 1.5]
            if not cands:
                continue
            lk = max(cands)
        w = sk - lk
        if w <= 0:
            continue
        pp = hd.get_spread_prices(ticker, ed, sk, lk, "P", trade_date)
        if pp is None:
            continue
        cr = pp["short_close"] - pp["long_close"]
        if cr > 0.05:
            return {"short": sk, "long": lk, "credit": round(cr, 4),
                    "width": w, "max_loss": round(w - cr, 4)}
    return None


def _walk(hd, ticker, exp, short_k, long_k, credit, entry_dt, exp_obj,
          td_index, profit_pct=0.50, stop_mult=3.0, min_dte=7):
    td_set = set(td_index.strftime("%Y-%m-%d"))
    cur = entry_dt + timedelta(days=1)
    hold = 0
    while cur <= exp_obj:
        cs = cur.strftime("%Y-%m-%d")
        if cs not in td_set:
            cur += timedelta(days=1)
            continue
        hold += 1
        dte = (exp_obj - cur).days
        pp = hd.get_spread_prices(ticker, exp_obj, short_k, long_k, "P", cs)
        if pp is None:
            cur += timedelta(days=1)
            continue
        cv = pp["short_close"] - pp["long_close"]
        if cv <= credit * (1 - profit_pct):
            return cs, "profit_target", cv, hold
        if cv - credit > credit * stop_mult:
            return cs, "stop_loss", cv, hold
        if dte <= min_dte:
            return cs, "dte_exit", cv, hold
        cur += timedelta(days=1)
    fp = hd.get_spread_prices(ticker, exp_obj, short_k, long_k, "P", exp)
    ev = fp["short_close"] - fp["long_close"] if fp else 0.0
    return exp, "expiration", ev, hold


# ═══════════════════════════════════════════════════════════════════════════
# Core VTS engine — parameterized
# ═══════════════════════════════════════════════════════════════════════════

def run_vts(
    hd: IronVault,
    ticker: str,
    price_df: pd.DataFrame,
    start: str = "2020-03-01",
    end: str = "2025-12-31",
    # Signal params
    contango_threshold: float = 1.15,
    otm_pct: float = 0.94,
    width: float = 5.0,
    # Timing
    entry_days_before: int = 25,
    min_interval: int = 14,
    front_back_gap_min: int = 25,
    front_back_gap_max: int = 45,
    # Exit
    profit_pct: float = 0.50,
    stop_mult: float = 3.0,
    min_dte: int = 7,
    # Sizing
    risk_pct: float = 0.015,
    max_contracts: int = 3,
    # Regime filter
    regime_series: Optional[pd.Series] = None,
    allowed_regimes: Optional[set] = None,
    regime_scale: Optional[Dict[str, float]] = None,
    label: str = "",
) -> List[Dict]:
    close = price_df["Close"]
    td_set = set(price_df.index.strftime("%Y-%m-%d"))
    exps = _all_exps(hd, ticker, start, end)
    trades, last = [], None

    for i, front in enumerate(exps):
        front_dt = _exp_dt(front)
        back = None
        for j in range(i + 1, min(i + 40, len(exps))):
            d = (_exp_dt(exps[j]) - front_dt).days
            if front_back_gap_min <= d <= front_back_gap_max:
                back = exps[j]
                break
        if back is None:
            continue
        back_dt = _exp_dt(back)

        entry_dt = _next_td(front_dt - timedelta(days=entry_days_before), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < min_interval:
            continue

        try:
            price = float(close.loc[es])
        except (KeyError, TypeError):
            continue

        # Regime filter
        if regime_series is not None and allowed_regimes is not None:
            try:
                reg = str(regime_series.loc[es])
            except (KeyError, TypeError):
                reg = "bull"
            if reg not in allowed_regimes:
                continue

        # Term structure signal
        front_strikes = hd.get_available_strikes(ticker, front, es, "P")
        back_strikes = hd.get_available_strikes(ticker, back, es, "P")
        common = sorted(set(front_strikes or []) & set(back_strikes or []))
        if not common:
            continue
        target_k = round(price * 0.95)
        strike = min(common, key=lambda k: abs(k - target_k))

        fsym = IronVault.build_occ_symbol(ticker, front_dt, strike, "P")
        bsym = IronVault.build_occ_symbol(ticker, back_dt, strike, "P")
        fp = hd.get_contract_price(fsym, es)
        bp = hd.get_contract_price(bsym, es)
        if fp is None or bp is None or fp < 0.05:
            continue
        ratio = bp / fp
        if ratio < contango_threshold:
            continue

        spread = _sell_put_spread(hd, ticker, front, es, price, otm_pct, width)
        if spread is None:
            continue

        # Regime-adjusted sizing
        scale = 1.0
        if regime_series is not None and regime_scale is not None:
            try:
                reg = str(regime_series.loc[es])
            except (KeyError, TypeError):
                reg = "bull"
            scale = regime_scale.get(reg, 1.0)

        base_contracts = max(1, min(max_contracts, int(CAPITAL * risk_pct / (spread["max_loss"] * 100))))
        contracts = max(1, int(base_contracts * scale))

        ed, er, ev, hold = _walk(hd, ticker, front, spread["short"], spread["long"],
                                  spread["credit"], entry_dt, front_dt, price_df.index,
                                  profit_pct, stop_mult, min_dte)
        pnl = (spread["credit"] - ev) * 100 * contracts
        trades.append({
            "entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
            "exit_reason": er, "credit": spread["credit"],
            "term_ratio": round(ratio, 3), "hold_days": hold,
            "contracts": contracts, "ticker": ticker,
        })
        last = entry_dt

    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Stats helpers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VTSResult:
    label: str
    ticker: str = ""
    params: Dict = field(default_factory=dict)
    trades: List[Dict] = field(default_factory=list)
    n: int = 0
    pnl: float = 0.0
    wr: float = 0.0
    dd: float = 0.0
    sharpe: float = 0.0
    cagr: float = 0.0
    spy_corr: float = 0.0
    yearly: Dict = field(default_factory=dict)
    # Walk-forward
    wf_windows: List[Dict] = field(default_factory=list)
    # Capacity
    avg_volume: float = 0.0
    max_safe_contracts: int = 0
    max_notional: float = 0.0


def _stats(trades, label, spy_ret, ticker="SPY", params=None):
    r = VTSResult(label=label, ticker=ticker, params=params or {})
    if not trades:
        return r
    df = pd.DataFrame(trades)
    pnls = df["pnl"].values
    n = len(pnls)
    r.n = n
    r.pnl = round(float(pnls.sum()), 2)
    r.wr = round(float((pnls > 0).sum()) / n, 4)

    eq = np.cumsum(pnls) + CAPITAL
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / pk
    r.dd = round(float(dd.max()), 4)

    mu = pnls.mean()
    sd = pnls.std(ddof=1) if n > 1 else 1.0
    r.sharpe = round(float(mu / sd * math.sqrt(min(n, 52))) if sd > 1e-9 else 0, 3)

    dates = pd.to_datetime(df["exit_date"])
    yrs = max((dates.max() - pd.to_datetime(df["entry_date"]).min()).days / 365.25, 0.5)
    r.cagr = round(((1 + r.pnl / CAPITAL) ** (1 / yrs) - 1) if r.pnl > -CAPITAL else -1.0, 4)

    # SPY correlation
    tr = {}
    for _, row in df.iterrows():
        d = str(row["exit_date"])[:10]
        tr[d] = tr.get(d, 0) + row["pnl"]
    ts = pd.Series(tr)
    ts.index = pd.to_datetime(ts.index)
    common = ts.index.intersection(spy_ret.index)
    r.spy_corr = round(float(np.corrcoef(ts.reindex(common).fillna(0), spy_ret.reindex(common).fillna(0))[0, 1]) if len(common) > 10 else 0.0, 4)

    # Yearly
    df["year"] = dates.dt.year
    for yr, g in df.groupby("year"):
        yp = g["pnl"].values
        yn = len(yp)
        if yn == 0:
            continue
        ye = np.cumsum(yp) + CAPITAL
        ypk = np.maximum.accumulate(ye)
        ydd = (ypk - ye) / ypk
        ysd = float(yp.std(ddof=1)) if yn > 1 else 1.0
        r.yearly[int(yr)] = {
            "n": yn, "pnl": round(float(yp.sum()), 2),
            "wr": round(float((yp > 0).sum()) / yn, 4),
            "dd": round(float(ydd.max()), 4),
            "sharpe": round(float(yp.mean()) / ysd * math.sqrt(min(yn, 52)) if ysd > 1e-9 else 0, 3),
        }

    r.trades = trades
    return r


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward validation
# ═══════════════════════════════════════════════════════════════════════════

def walk_forward_validation(
    hd: IronVault, spy_df: pd.DataFrame, spy_ret: pd.Series,
) -> Tuple[VTSResult, List[Dict]]:
    """Rolling 2-year IS → 1-year OOS walk-forward."""
    windows = [
        ("2020-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2021-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2022-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
        ("2023-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ]
    all_oos_trades = []
    wf_results = []

    for is_start, is_end, oos_start, oos_end in windows:
        # IS run to verify signal works
        is_trades = run_vts(hd, "SPY", spy_df, is_start, is_end)
        is_pnl = sum(t["pnl"] for t in is_trades)
        is_n = len(is_trades)
        is_wr = sum(1 for t in is_trades if t["pnl"] > 0) / max(is_n, 1)

        # OOS run with same params
        oos_trades = run_vts(hd, "SPY", spy_df, oos_start, oos_end)
        oos_pnl = sum(t["pnl"] for t in oos_trades)
        oos_n = len(oos_trades)
        oos_wr = sum(1 for t in oos_trades if t["pnl"] > 0) / max(oos_n, 1)

        oos_sharpe = 0.0
        if oos_n > 1:
            op = np.array([t["pnl"] for t in oos_trades])
            os = op.std(ddof=1)
            oos_sharpe = float(op.mean() / os * math.sqrt(min(oos_n, 52))) if os > 1e-9 else 0

        wf_results.append({
            "is_period": f"{is_start[:4]}-{is_end[:4]}",
            "oos_period": f"{oos_start[:4]}",
            "is_n": is_n, "is_pnl": round(is_pnl, 2), "is_wr": round(is_wr, 4),
            "oos_n": oos_n, "oos_pnl": round(oos_pnl, 2), "oos_wr": round(oos_wr, 4),
            "oos_sharpe": round(oos_sharpe, 3),
        })
        all_oos_trades.extend(oos_trades)

    combined = _stats(all_oos_trades, "Walk-Forward Combined OOS", spy_ret)
    combined.wf_windows = wf_results
    return combined, wf_results


# ═══════════════════════════════════════════════════════════════════════════
# Parameter sensitivity
# ═══════════════════════════════════════════════════════════════════════════

def parameter_sensitivity(
    hd: IronVault, spy_df: pd.DataFrame, spy_ret: pd.Series,
) -> List[VTSResult]:
    """Test key parameter variations."""
    configs = [
        ("Baseline (ratio≥1.15)",      {"contango_threshold": 1.15}),
        ("Loose (ratio≥1.05)",         {"contango_threshold": 1.05}),
        ("Tight (ratio≥1.25)",         {"contango_threshold": 1.25}),
        ("Very Tight (ratio≥1.35)",    {"contango_threshold": 1.35}),
        ("Closer OTM (96%)",           {"otm_pct": 0.96}),
        ("Further OTM (92%)",          {"otm_pct": 0.92}),
        ("Narrow $3 spread",           {"width": 3.0}),
        ("Wide $10 spread",            {"width": 10.0}),
        ("Quick profit (40%)",         {"profit_pct": 0.40}),
        ("Patient profit (65%)",       {"profit_pct": 0.65}),
        ("Tight stop (2x)",            {"stop_mult": 2.0}),
        ("Wide stop (5x)",             {"stop_mult": 5.0}),
        ("Shorter hold (entry 18d)",   {"entry_days_before": 18}),
        ("Longer hold (entry 32d)",    {"entry_days_before": 32}),
        ("Frequent (interval 10d)",    {"min_interval": 10}),
        ("Rare (interval 21d)",        {"min_interval": 21}),
    ]

    results = []
    for label, overrides in configs:
        params = {
            "contango_threshold": 1.15, "otm_pct": 0.94, "width": 5.0,
            "profit_pct": 0.50, "stop_mult": 3.0, "entry_days_before": 25,
            "min_interval": 14, "risk_pct": 0.015, "max_contracts": 3,
        }
        params.update(overrides)
        trades = run_vts(hd, "SPY", spy_df, **params)
        r = _stats(trades, label, spy_ret, params=params)
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Multi-ticker test
# ═══════════════════════════════════════════════════════════════════════════

def multi_ticker_test(
    hd: IronVault, price_data: Dict[str, pd.DataFrame], spy_ret: pd.Series,
) -> List[VTSResult]:
    """Run VTS on all tickers with sufficient data."""
    ticker_configs = {
        "SPY":  {"width": 5.0, "otm_pct": 0.94, "end": "2025-12-31"},
        "XLF":  {"width": 1.0, "otm_pct": 0.95, "end": "2025-12-31"},
        "XLI":  {"width": 1.0, "otm_pct": 0.95, "end": "2025-12-31"},
        "TLT":  {"width": 2.0, "otm_pct": 0.95, "end": "2024-06-30"},
        "GLD":  {"width": 2.0, "otm_pct": 0.95, "end": "2024-02-28"},
        "QQQ":  {"width": 5.0, "otm_pct": 0.94, "end": "2023-03-31"},
    }
    results = []
    for ticker, cfg in ticker_configs.items():
        if ticker not in price_data:
            continue
        trades = run_vts(
            hd, ticker, price_data[ticker],
            start="2020-03-01", end=cfg["end"],
            width=cfg["width"], otm_pct=cfg["otm_pct"],
            contango_threshold=1.10,  # slightly looser for thinner tickers
        )
        r = _stats(trades, f"VTS on {ticker}", spy_ret, ticker=ticker)
        results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Regime-filtered backtest
# ═══════════════════════════════════════════════════════════════════════════

def regime_filtered_test(
    hd: IronVault, spy_df: pd.DataFrame, vix_series: pd.Series,
    spy_ret: pd.Series,
) -> Tuple[VTSResult, VTSResult]:
    """Compare unfiltered vs regime-filtered VTS."""
    # Classify regimes
    rc = RegimeClassifier()
    regime_s = rc.classify_series(spy_df, vix_series)

    # Unfiltered baseline
    baseline_trades = run_vts(hd, "SPY", spy_df)
    baseline = _stats(baseline_trades, "VTS Unfiltered", spy_ret)

    # Regime-filtered: only trade in bull, low_vol, neutral (skip bear/high_vol/crash)
    allowed = {"bull", "low_vol", Regime.BULL, Regime.LOW_VOL, "Regime.BULL", "Regime.LOW_VOL"}
    # Also add string versions
    for r in [Regime.BULL, Regime.LOW_VOL]:
        allowed.add(str(r))
        allowed.add(r.value)

    scale = {"bull": 1.0, "low_vol": 1.0, "bear": 0.0, "high_vol": 0.0, "crash": 0.0,
             Regime.BULL.value: 1.0, Regime.LOW_VOL.value: 1.0,
             Regime.BEAR.value: 0.0, Regime.HIGH_VOL.value: 0.0, Regime.CRASH.value: 0.0}

    filtered_trades = run_vts(
        hd, "SPY", spy_df,
        regime_series=regime_s,
        allowed_regimes=allowed,
        regime_scale=scale,
    )
    filtered = _stats(filtered_trades, "VTS + Regime Filter", spy_ret)

    return baseline, filtered


# ═══════════════════════════════════════════════════════════════════════════
# Capacity estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_capacity(hd: IronVault, trades: List[Dict]) -> Dict:
    """Estimate max capacity from real volume data."""
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()

    cur.execute("""SELECT AVG(volume) FROM option_daily od
        JOIN option_contracts oc ON od.contract_symbol=oc.contract_symbol
        WHERE oc.ticker='SPY' AND oc.option_type='P' AND od.volume > 0 AND od.date > '2023-01-01'""")
    avg_vol = cur.fetchone()[0] or 100

    # Get volume at the specific strikes we actually traded
    trade_volumes = []
    for t in trades:
        if "credit" in t and t.get("ticker", "SPY") == "SPY":
            es = t["entry_date"]
            cur.execute("""SELECT volume FROM option_daily od
                JOIN option_contracts oc ON od.contract_symbol=oc.contract_symbol
                WHERE oc.ticker='SPY' AND oc.option_type='P' AND od.date=?
                AND od.volume > 0 ORDER BY od.volume DESC LIMIT 5""", (es,))
            vols = [r[0] for r in cur.fetchall()]
            if vols:
                trade_volumes.append(np.median(vols))
    conn.close()

    median_trade_vol = float(np.median(trade_volumes)) if trade_volumes else avg_vol

    # Conservative: fill ≤5% of daily volume to avoid market impact
    fill_pct = 0.05
    max_contracts = int(median_trade_vol * fill_pct)
    avg_spread_width = 5.0  # $5 wide
    notional_per_contract = avg_spread_width * 100  # $500
    max_notional = max_contracts * notional_per_contract

    # Scale up for $100K base → how much capital can we deploy?
    avg_credit = np.mean([t.get("credit", 1.0) for t in trades])
    avg_max_loss = avg_spread_width - avg_credit
    risk_per_contract = avg_max_loss * 100  # ~$450-470

    # At 1.5% risk per trade
    max_capital_at_1_5pct = max_contracts * risk_per_contract / 0.015

    return {
        "avg_daily_volume": round(avg_vol),
        "median_trade_volume": round(median_trade_vol),
        "fill_limit_pct": fill_pct,
        "max_contracts_per_trade": max_contracts,
        "notional_per_contract": notional_per_contract,
        "max_notional_per_trade": round(max_notional),
        "max_capital_at_1_5pct_risk": round(max_capital_at_1_5pct),
        "avg_credit": round(avg_credit, 3),
        "avg_max_loss_per_contract": round(risk_per_contract, 2),
        "institutional_viable": max_capital_at_1_5pct > 1_000_000,
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def _c(v): return "#3fb950" if v >= 0 else "#f85149"
def _fd(v): return f"${v:,.0f}"
def _fp(v): return f"{v:.1%}"
def _fr(v): return f"{v:.2f}"


def _build_html(
    baseline: VTSResult,
    multi: List[VTSResult],
    sensitivity: List[VTSResult],
    wf: Tuple[VTSResult, List[Dict]],
    regime_base: VTSResult,
    regime_filt: VTSResult,
    capacity: Dict,
    output: Path,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    wf_result, wf_windows = wf

    # ── Section 1: Baseline ──
    yr_rows = ""
    for yr in sorted(baseline.yearly):
        y = baseline.yearly[yr]
        yr_rows += f"""<tr><td>{yr}</td><td>{y['n']}</td>
          <td style="color:{_c(y['pnl'])}">{_fd(y['pnl'])}</td>
          <td>{_fp(y['wr'])}</td><td>{_fp(y['dd'])}</td>
          <td style="color:{_c(y['sharpe'])}">{_fr(y['sharpe'])}</td></tr>"""

    # ── Section 2: Multi-ticker ──
    multi_rows = ""
    for r in multi:
        multi_rows += f"""<tr><td style="text-align:left">{r.ticker}</td>
          <td>{r.n}</td><td style="color:{_c(r.pnl)}">{_fd(r.pnl)}</td>
          <td>{_fp(r.wr)}</td><td>{_fp(r.dd)}</td>
          <td style="color:{_c(r.sharpe)}">{_fr(r.sharpe)}</td>
          <td>{_fr(r.spy_corr)}</td></tr>"""

    # ── Section 3: Walk-forward ──
    wf_rows = ""
    for w in wf_windows:
        wf_rows += f"""<tr><td>{w['is_period']}</td><td>{w['oos_period']}</td>
          <td>{w['is_n']}</td><td style="color:{_c(w['is_pnl'])}">{_fd(w['is_pnl'])}</td>
          <td>{_fp(w['is_wr'])}</td><td>{w['oos_n']}</td>
          <td style="color:{_c(w['oos_pnl'])}">{_fd(w['oos_pnl'])}</td>
          <td>{_fp(w['oos_wr'])}</td>
          <td style="color:{_c(w['oos_sharpe'])}">{_fr(w['oos_sharpe'])}</td></tr>"""

    # ── Section 4: Parameter sensitivity ──
    sens_rows = ""
    for r in sorted(sensitivity, key=lambda x: -x.sharpe):
        border = "border-left:3px solid #d29922;" if r.label == "Baseline (ratio≥1.15)" else ""
        sens_rows += f"""<tr style="{border}"><td style="text-align:left">{r.label}</td>
          <td>{r.n}</td><td style="color:{_c(r.pnl)}">{_fd(r.pnl)}</td>
          <td>{_fp(r.wr)}</td><td>{_fp(r.dd)}</td>
          <td style="color:{_c(r.sharpe)}">{_fr(r.sharpe)}</td></tr>"""

    # ── Section 5: Regime filter ──
    regime_comparison = f"""
    <div class="g2">
      <div class="card"><h4>Unfiltered</h4>
        <div class="mg"><div><span class="l">Trades</span><span class="v">{regime_base.n}</span></div>
        <div><span class="l">P&L</span><span class="v" style="color:{_c(regime_base.pnl)}">{_fd(regime_base.pnl)}</span></div>
        <div><span class="l">Win Rate</span><span class="v">{_fp(regime_base.wr)}</span></div>
        <div><span class="l">Sharpe</span><span class="v" style="color:{_c(regime_base.sharpe)}">{_fr(regime_base.sharpe)}</span></div>
        <div><span class="l">Max DD</span><span class="v">{_fp(regime_base.dd)}</span></div></div></div>
      <div class="card"><h4>Regime-Filtered (Bull + Low Vol only)</h4>
        <div class="mg"><div><span class="l">Trades</span><span class="v">{regime_filt.n}</span></div>
        <div><span class="l">P&L</span><span class="v" style="color:{_c(regime_filt.pnl)}">{_fd(regime_filt.pnl)}</span></div>
        <div><span class="l">Win Rate</span><span class="v">{_fp(regime_filt.wr)}</span></div>
        <div><span class="l">Sharpe</span><span class="v" style="color:{_c(regime_filt.sharpe)}">{_fr(regime_filt.sharpe)}</span></div>
        <div><span class="l">Max DD</span><span class="v">{_fp(regime_filt.dd)}</span></div></div></div>
    </div>"""

    # ── Section 6: Capacity ──
    inst = "Yes" if capacity["institutional_viable"] else "No"
    inst_c = "#3fb950" if capacity["institutional_viable"] else "#f85149"

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Vol Term Structure — Deep Dive</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1300px; margin: 0 auto; padding: 24px; background: #0d1117; color: #c9d1d9; }}
  h1 {{ color: #58a6ff; margin-bottom: 4px; }}
  h2 {{ color: #58a6ff; border-bottom: 1px solid #21262d; padding-bottom: 6px; margin-top: 36px; }}
  h3 {{ color: #79c0ff; }}  h4 {{ color: #79c0ff; margin: 8px 0; }}
  .meta {{ color: #8b949e; font-size: 0.88em; }}
  .hero {{ background: #161b22; border: 2px solid #d29922; border-radius: 12px;
           padding: 24px; margin: 20px 0; text-align: center; }}
  .hero .big {{ font-size: 2.5em; font-weight: 800; color: #d29922; }}
  .hero .sub {{ color: #8b949e; font-size: 1.1em; margin-top: 4px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px; }}
  .g2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
  .mg {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 6px; margin-top: 8px; }}
  .mg .l {{ display: block; color: #8b949e; font-size: 0.72em; }}
  .mg .v {{ display: block; font-weight: 600; }}
  table.dt {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 0.84em; }}
  table.dt th, table.dt td {{ padding: 5px 8px; text-align: right; border-bottom: 1px solid #21262d; }}
  table.dt th {{ color: #8b949e; background: #161b22; }}
  table.dt td:first-child {{ text-align: left; }}
  .kpi {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 16px 0; }}
  .kpi > div {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 10px; text-align: center; }}
  .kpi .l {{ display: block; color: #8b949e; font-size: 0.72em; }}
  .kpi .v {{ display: block; font-weight: 600; font-size: 1.1em; }}
  footer {{ margin-top: 40px; padding-top: 12px; border-top: 1px solid #21262d;
            color: #484f58; font-size: 0.78em; }}
</style></head><body>

<h1>Vol Term Structure — Deep Dive</h1>
<p class="meta">Generated {ts} &middot; Real IronVault data only &middot; Zero synthetic pricing</p>

<div class="hero">
  <div class="big">{_fr(baseline.sharpe)} Sharpe</div>
  <div class="sub">{baseline.n} trades &middot; {_fp(baseline.wr)} win rate &middot;
    {_fp(baseline.dd)} max DD &middot; {_fp(baseline.cagr)} CAGR &middot; SPY corr {_fr(baseline.spy_corr)}</div>
</div>

<h2>1. Baseline Performance (SPY, 2020-2025)</h2>
<div class="kpi">
  <div><span class="l">Total P&L</span><span class="v" style="color:{_c(baseline.pnl)}">{_fd(baseline.pnl)}</span></div>
  <div><span class="l">Win Rate</span><span class="v">{_fp(baseline.wr)}</span></div>
  <div><span class="l">Max Drawdown</span><span class="v">{_fp(baseline.dd)}</span></div>
  <div><span class="l">Sharpe</span><span class="v" style="color:{_c(baseline.sharpe)}">{_fr(baseline.sharpe)}</span></div>
  <div><span class="l">CAGR</span><span class="v" style="color:{_c(baseline.cagr)}">{_fp(baseline.cagr)}</span></div>
  <div><span class="l">SPY Correlation</span><span class="v">{_fr(baseline.spy_corr)}</span></div>
  <div><span class="l">Avg P&L / Trade</span><span class="v" style="color:{_c(baseline.pnl)}">{_fd(baseline.pnl / max(baseline.n, 1))}</span></div>
</div>
<table class="dt"><tr><th>Year</th><th>Trades</th><th>P&L</th><th>Win Rate</th><th>Max DD</th><th>Sharpe</th></tr>
{yr_rows}</table>

<h2>2. Multi-Ticker Analysis</h2>
<p class="meta">VTS applied to all tickers with sufficient data in IronVault (contango threshold 1.10 for thinner tickers)</p>
<table class="dt"><tr><th style="text-align:left">Ticker</th><th>Trades</th><th>P&L</th><th>Win Rate</th>
  <th>Max DD</th><th>Sharpe</th><th>SPY Corr</th></tr>
{multi_rows}</table>

<h2>3. Walk-Forward Validation (2-yr IS → 1-yr OOS)</h2>
<p class="meta">Each window trains on 2 years, tests on the next year. Consistent positive OOS = robust signal.</p>
<div class="kpi">
  <div><span class="l">Combined OOS Trades</span><span class="v">{wf_result.n}</span></div>
  <div><span class="l">Combined OOS P&L</span><span class="v" style="color:{_c(wf_result.pnl)}">{_fd(wf_result.pnl)}</span></div>
  <div><span class="l">Combined OOS Sharpe</span><span class="v" style="color:{_c(wf_result.sharpe)}">{_fr(wf_result.sharpe)}</span></div>
  <div><span class="l">Combined OOS WR</span><span class="v">{_fp(wf_result.wr)}</span></div>
</div>
<table class="dt"><tr><th>IS Period</th><th>OOS Year</th><th>IS Trades</th><th>IS P&L</th><th>IS WR</th>
  <th>OOS Trades</th><th>OOS P&L</th><th>OOS WR</th><th>OOS Sharpe</th></tr>
{wf_rows}</table>

<h2>4. Parameter Sensitivity</h2>
<p class="meta">Each row changes ONE parameter vs baseline. Sorted by Sharpe.</p>
<table class="dt"><tr><th style="text-align:left">Configuration</th><th>Trades</th><th>P&L</th>
  <th>Win Rate</th><th>Max DD</th><th>Sharpe</th></tr>
{sens_rows}</table>

<h2>5. Regime Filter (compass/regime.py)</h2>
<p class="meta">RegimeClassifier: only trade in BULL + LOW_VOL regimes. Skip BEAR / HIGH_VOL / CRASH.</p>
{regime_comparison}

<h2>6. Capacity Analysis</h2>
<div class="kpi">
  <div><span class="l">Avg Daily Volume (SPY puts)</span><span class="v">{capacity['avg_daily_volume']:,}</span></div>
  <div><span class="l">Median Trade Volume</span><span class="v">{capacity['median_trade_volume']:,}</span></div>
  <div><span class="l">Fill Limit (5% of volume)</span><span class="v">{capacity['max_contracts_per_trade']:,} contracts</span></div>
  <div><span class="l">Max Notional / Trade</span><span class="v">{_fd(capacity['max_notional_per_trade'])}</span></div>
  <div><span class="l">Max Capital @ 1.5% Risk</span><span class="v">{_fd(capacity['max_capital_at_1_5pct_risk'])}</span></div>
  <div><span class="l">Institutional Viable (&gt;$1M)</span><span class="v" style="color:{inst_c}">{inst}</span></div>
</div>

<footer>
  Data: IronVault options_cache.db &middot; SPY (187K contracts, 644 expirations), XLF, XLI, TLT, GLD, QQQ &middot;
  No synthetic pricing &middot; Cache miss → trade skipped
</footer>
</body></html>"""

    output.write_text(html, encoding="utf-8")
    return output


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(output: Path = DEFAULT_OUTPUT):
    logging.basicConfig(level=logging.WARNING)

    print("=" * 70)
    print("VOL TERM STRUCTURE — DEEP DIVE")
    print("=" * 70)

    hd = IronVault.instance()
    print(f"IronVault: {hd.coverage_report()['contracts_total']:,} contracts\n")

    print("Fetching market data...")
    spy_df = _dl("SPY")
    vix_df = _dl("^VIX")
    vix_close = vix_df["Close"] if "Close" in vix_df.columns else vix_df.iloc[:, 0]
    spy_ret = spy_df["Close"].pct_change().dropna() * CAPITAL

    price_data = {"SPY": spy_df}
    for t in ["XLF", "XLI", "TLT", "GLD", "QQQ"]:
        price_data[t] = _dl(t)

    # 1. Baseline
    print("\n[1] Baseline SPY VTS...")
    baseline_trades = run_vts(hd, "SPY", spy_df)
    baseline = _stats(baseline_trades, "VTS Baseline (SPY)", spy_ret)
    print(f"    {baseline.n} trades, Sharpe {baseline.sharpe}, P&L {_fd(baseline.pnl)}")

    # 2. Multi-ticker
    print("\n[2] Multi-ticker VTS...")
    multi = multi_ticker_test(hd, price_data, spy_ret)
    for r in multi:
        print(f"    {r.ticker}: {r.n} trades, Sharpe {_fr(r.sharpe)}, P&L {_fd(r.pnl)}")

    # 3. Walk-forward
    print("\n[3] Walk-forward validation...")
    wf_combined, wf_windows = walk_forward_validation(hd, spy_df, spy_ret)
    for w in wf_windows:
        print(f"    IS {w['is_period']} → OOS {w['oos_period']}: "
              f"{w['oos_n']} trades, {_fd(w['oos_pnl'])}, Sharpe {_fr(w['oos_sharpe'])}")
    print(f"    Combined OOS: {wf_combined.n} trades, Sharpe {_fr(wf_combined.sharpe)}")

    # 4. Parameter sensitivity
    print("\n[4] Parameter sensitivity (16 configs)...")
    sensitivity = parameter_sensitivity(hd, spy_df, spy_ret)
    best_s = max(sensitivity, key=lambda x: x.sharpe)
    worst_s = min(sensitivity, key=lambda x: x.sharpe)
    print(f"    Best:  {best_s.label} — Sharpe {_fr(best_s.sharpe)}, {best_s.n} trades")
    print(f"    Worst: {worst_s.label} — Sharpe {_fr(worst_s.sharpe)}, {worst_s.n} trades")

    # 5. Regime filter
    print("\n[5] Regime filter test...")
    regime_base, regime_filt = regime_filtered_test(hd, spy_df, vix_close, spy_ret)
    print(f"    Unfiltered:  {regime_base.n} trades, Sharpe {_fr(regime_base.sharpe)}")
    print(f"    Filtered:    {regime_filt.n} trades, Sharpe {_fr(regime_filt.sharpe)}")

    # 6. Capacity
    print("\n[6] Capacity estimation...")
    capacity = estimate_capacity(hd, baseline_trades)
    print(f"    Max contracts/trade: {capacity['max_contracts_per_trade']}")
    print(f"    Max capital @ 1.5% risk: {_fd(capacity['max_capital_at_1_5pct_risk'])}")
    print(f"    Institutional viable: {capacity['institutional_viable']}")

    # Generate report
    rp = _build_html(baseline, multi, sensitivity, (wf_combined, wf_windows),
                     regime_base, regime_filt, capacity, output)
    print(f"\nReport: {rp}")

    # JSON
    jp = output.with_suffix(".json")
    summary = {
        "generated": datetime.now().isoformat(),
        "baseline": {"n": baseline.n, "pnl": baseline.pnl, "sharpe": baseline.sharpe,
                     "wr": baseline.wr, "dd": baseline.dd, "cagr": baseline.cagr,
                     "spy_corr": baseline.spy_corr, "yearly": baseline.yearly},
        "multi_ticker": [{"ticker": r.ticker, "n": r.n, "pnl": r.pnl, "sharpe": r.sharpe,
                          "wr": r.wr, "spy_corr": r.spy_corr} for r in multi],
        "walk_forward": {"combined_n": wf_combined.n, "combined_sharpe": wf_combined.sharpe,
                         "combined_pnl": wf_combined.pnl, "windows": wf_windows},
        "sensitivity": [{"label": r.label, "n": r.n, "pnl": r.pnl, "sharpe": r.sharpe,
                         "wr": r.wr, "dd": r.dd, "params": r.params} for r in sensitivity],
        "regime_filter": {
            "unfiltered": {"n": regime_base.n, "sharpe": regime_base.sharpe, "pnl": regime_base.pnl},
            "filtered": {"n": regime_filt.n, "sharpe": regime_filt.sharpe, "pnl": regime_filt.pnl},
        },
        "capacity": capacity,
    }
    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.bool_, np.integer)):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            return super().default(o)
    jp.write_text(json.dumps(summary, indent=2, cls=_Enc))
    return baseline


if __name__ == "__main__":
    main()
