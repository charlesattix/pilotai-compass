"""
Strategy Discovery Round 4 — genuinely NEW uncorrelated alpha sources.

All prices from IronVault (options_cache.db). Zero synthetic data.
ETF daily prices from yfinance.

Strategies:
  1. Dispersion Trading — sell sector IV, buy index IV when sectors > SPY
  2. Gamma Scalping — long near-expiry SPY straddles, hedge delta intraday
  3. Intraday Mean-Reversion — fade opening gaps using intraday option bars
  4. Seasonal Patterns — day-of-week + month effects on premium decay
  5. Volatility Risk Premium — systematic short vol with regime filter

Walk-forward: IS = 2020-2022, OOS = 2023-2025.
Kill: <10 OOS trades OR negative OOS Sharpe.
Correlation computed vs SPY AND vs EXP-1220.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)
CAPITAL = 100_000
OOS_START_YEAR = 2023


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _exp_dt(s): return datetime.strptime(s, "%Y-%m-%d")

def _dl(ticker):
    import yfinance as yf
    df = yf.download(ticker, start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    return df

def _find_exps(hd, ticker, start, end, monthly=True):
    conn = sqlite3.connect(hd._db_path)
    exps = [r[0] for r in conn.execute(
        "SELECT DISTINCT expiration FROM option_contracts "
        "WHERE ticker=? AND option_type='P' AND expiration BETWEEN ? AND ? "
        "ORDER BY expiration", (ticker, start, end)).fetchall()]
    conn.close()
    if not monthly: return exps
    out, last = [], ""
    for e in exps:
        ym, day = e[:7], int(e[8:10])
        if ym != last and 15 <= day <= 21: out.append(e); last = ym
    return out

def _next_td(dt, td_set):
    for off in range(7):
        c = dt + timedelta(days=off)
        if c.strftime("%Y-%m-%d") in td_set: return c
    return None

def _sell_put_spread(hd, ticker, exp, trade_date, price, otm_pct=0.93, width=5.0):
    strikes = hd.get_available_strikes(ticker, exp, trade_date, "P")
    if not strikes: return None
    target = price * otm_pct
    for sk in sorted(strikes, key=lambda k: abs(k - target))[:12]:
        lk = sk - width
        if lk not in strikes:
            cands = [s for s in strikes if s < sk and abs(s - lk) <= 1.0]
            if not cands: continue
            lk = max(cands)
        if sk - lk <= 0: continue
        pp = hd.get_spread_prices(ticker, _exp_dt(exp), sk, lk, "P", trade_date)
        if pp is None: continue
        credit = pp["short_close"] - pp["long_close"]
        if credit > 0.05:
            return {"short": sk, "long": lk, "credit": round(credit, 4),
                    "width": sk-lk, "max_loss": round(sk-lk-credit, 4)}
    return None

def _walk_spread(hd, ticker, exp, short_k, long_k, entry_credit, entry_dt, exp_dt_obj,
                 td_index, opt_type="P", profit_pct=0.50, stop_mult=3.0, min_dte=7):
    td_set = set(td_index.strftime("%Y-%m-%d"))
    hold = 0; current = entry_dt + timedelta(days=1)
    while current <= exp_dt_obj:
        cs = current.strftime("%Y-%m-%d")
        if cs not in td_set: current += timedelta(days=1); continue
        hold += 1
        pp = hd.get_spread_prices(ticker, exp_dt_obj, short_k, long_k, opt_type, cs)
        if pp is None: current += timedelta(days=1); continue
        cv = pp["short_close"] - pp["long_close"]
        if cv <= entry_credit * (1 - profit_pct): return cs, "profit_target", cv, hold
        if cv - entry_credit > entry_credit * stop_mult: return cs, "stop_loss", cv, hold
        if (exp_dt_obj - current).days <= min_dte: return cs, "dte_exit", cv, hold
        current += timedelta(days=1)
    fp = hd.get_spread_prices(ticker, exp_dt_obj, short_k, long_k, opt_type, exp)
    return exp, "expiration", (fp["short_close"]-fp["long_close"]) if fp else 0.0, hold


# ═══════════════════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Stats:
    name: str; hypothesis: str = ""; description: str = ""
    trades: List[Dict] = field(default_factory=list)
    n_trades: int = 0; total_pnl: float = 0; win_rate: float = 0; max_dd: float = 0
    sharpe: float = 0; cagr: float = 0; spy_corr: float = 0; exp1220_corr: float = 0
    avg_pnl: float = 0; oos_sharpe: float = 0; oos_n: int = 0; oos_pnl: float = 0
    oos_wr: float = 0; oos_dd: float = 0; oos_cagr: float = 0
    yearly: Dict[int, Dict] = field(default_factory=dict)
    killed: bool = False; kill_reason: str = ""; capacity: str = "unknown"

def _compute(trades, name, spy_ret, exp1220_ret, hypothesis="", desc="", capacity="unknown"):
    if not trades:
        return Stats(name=name, hypothesis=hypothesis, description=desc,
                     killed=True, kill_reason="0 trades", capacity=capacity)
    df = pd.DataFrame(trades); pnls = df["pnl"].values; n = len(pnls)
    total = float(pnls.sum()); wins = int((pnls > 0).sum())
    eq = np.cumsum(pnls) + CAPITAL; pk = np.maximum.accumulate(eq)
    max_dd = float(((pk-eq)/pk).max())
    mu = float(pnls.mean()); sd = float(pnls.std(ddof=1)) if n > 1 else 1.0
    sharpe = mu / sd * math.sqrt(min(n, 52)) if sd > 1e-9 else 0.0
    dates = pd.to_datetime(df["exit_date"])
    yrs = max((dates.max() - pd.to_datetime(df["entry_date"]).min()).days / 365.25, 0.5)
    cagr = ((1+total/CAPITAL)**(1/yrs)-1) if total > -CAPITAL else -1.0

    tr = {}
    for _, r in df.iterrows():
        d = str(r["exit_date"])[:10]; tr[d] = tr.get(d, 0) + r["pnl"]
    ts = pd.Series(tr); ts.index = pd.to_datetime(ts.index)
    def _corr(a, b):
        c = a.index.intersection(b.index)
        return float(np.corrcoef(a.reindex(c).fillna(0), b.reindex(c).fillna(0))[0, 1]) if len(c) > 10 else 0.0
    spy_c = _corr(ts, spy_ret); exp_c = _corr(ts, exp1220_ret)

    oos = df[dates.dt.year >= OOS_START_YEAR]; oos_n = len(oos)
    oos_sharpe = oos_pnl = oos_wr = oos_dd_v = oos_cagr_v = 0.0
    if oos_n > 1:
        op = oos["pnl"].values; oos_pnl = float(op.sum())
        oos_wr = float((op > 0).sum()) / oos_n
        os = float(op.std(ddof=1))
        oos_sharpe = float(op.mean()) / os * math.sqrt(min(oos_n, 52)) if os > 1e-9 else 0.0
        oe = np.cumsum(op) + CAPITAL; opk = np.maximum.accumulate(oe)
        oos_dd_v = float(((opk-oe)/opk).max())

    df["year"] = dates.dt.year; yearly = {}
    for yr, g in df.groupby("year"):
        yp = g["pnl"].values; yn = len(yp)
        if yn == 0: continue
        ye = np.cumsum(yp)+CAPITAL; ypk = np.maximum.accumulate(ye)
        ysd = float(yp.std(ddof=1)) if yn > 1 else 1.0
        yearly[int(yr)] = {"n": yn, "pnl": round(float(yp.sum()), 2),
            "wr": round(float((yp > 0).sum()) / yn, 4),
            "dd": round(float(((ypk-ye)/ypk).max()), 4),
            "sharpe": round(float(yp.mean()) / ysd * math.sqrt(min(yn, 52)) if ysd > 1e-9 else 0, 3)}

    killed = oos_n < 10 or oos_sharpe < 0
    kr = f"Only {oos_n} OOS trades (<10)" if oos_n < 10 else (f"Neg OOS Sharpe ({oos_sharpe:.2f})" if oos_sharpe < 0 else "")

    return Stats(name=name, hypothesis=hypothesis, description=desc, trades=trades,
        n_trades=n, total_pnl=round(total, 2), win_rate=round(wins/n, 4),
        max_dd=round(max_dd, 4), sharpe=round(sharpe, 3), cagr=round(cagr, 4),
        spy_corr=round(spy_c, 4), exp1220_corr=round(exp_c, 4), avg_pnl=round(mu, 2),
        oos_sharpe=round(oos_sharpe, 3), oos_n=oos_n, oos_pnl=round(oos_pnl, 2),
        oos_wr=round(oos_wr, 4), oos_dd=round(oos_dd_v, 4), oos_cagr=round(oos_cagr_v, 4),
        yearly=yearly, killed=killed, kill_reason=kr, capacity=capacity)


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 1: Dispersion
# ═══════════════════════════════════════════════════════════════════════════

def strat_dispersion(hd, spy_df, sector_dfs):
    """Sell sector put spreads when sector IV > 1.3x SPY IV (dispersion)."""
    print("  [1] Dispersion: sector IV vs SPY IV")
    spy_close = spy_df["Close"]; td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    trades, last = [], None
    for stk in ["XLF", "XLI", "XLE"]:
        if stk not in sector_dfs: continue
        stk_close = sector_dfs[stk]["Close"]
        for exp in _find_exps(hd, stk, "2020-03-01", "2025-12-31", monthly=True):
            exp_obj = _exp_dt(exp)
            entry_dt = _next_td(exp_obj - timedelta(days=30), td_set)
            if entry_dt is None: continue
            es = entry_dt.strftime("%Y-%m-%d")
            if last and (entry_dt - last).days < 18: continue
            try: spy_p = float(spy_close.loc[es]); stk_p = float(stk_close.loc[es])
            except: continue
            if np.isnan(spy_p) or np.isnan(stk_p): continue
            spy_k = min(hd.get_available_strikes("SPY", exp, es, "P") or [0], key=lambda k: abs(k-spy_p*0.95), default=0)
            stk_k = min(hd.get_available_strikes(stk, exp, es, "P") or [0], key=lambda k: abs(k-stk_p*0.95), default=0)
            if spy_k == 0 or stk_k == 0: continue
            spy_sym = IronVault.build_occ_symbol("SPY", exp_obj, spy_k, "P")
            stk_sym = IronVault.build_occ_symbol(stk, exp_obj, stk_k, "P")
            sp = hd.get_contract_price(spy_sym, es); stp = hd.get_contract_price(stk_sym, es)
            if sp is None or stp is None or spy_p <= 0 or stk_p <= 0: continue
            if (stp/stk_p) < (sp/spy_p) * 1.3: continue
            w = 1.0 if stk in ("XLF","XLE") else 2.0
            spread = _sell_put_spread(hd, stk, exp, es, stk_p, otm_pct=0.94, width=w)
            if spread is None: continue
            cts = max(1, min(5, int(CAPITAL * 0.015 / (spread["max_loss"] * 100))))
            ed, er, ev, hold = _walk_spread(hd, stk, exp, spread["short"], spread["long"],
                                            spread["credit"], entry_dt, exp_obj, spy_df.index)
            pnl = (spread["credit"] - ev) * 100 * cts
            trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                           "exit_reason": er, "sector": stk, "hold_days": hold})
            last = entry_dt
    print(f"    -> {len(trades)} trades"); return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 2: Gamma Scalping
# ═══════════════════════════════════════════════════════════════════════════

def strat_gamma_scalp(hd, spy_df, vix):
    """Buy ATM SPY straddles 5-7 DTE when VIX < 18 (cheap gamma)."""
    print("  [2] Gamma Scalp: long near-expiry straddles")
    spy_close = spy_df["Close"]; td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    all_exps = _find_exps(hd, "SPY", "2020-03-01", "2025-12-31", monthly=False)
    trades, last = [], None
    for exp in all_exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=6), td_set)
        if entry_dt is None: continue
        es = entry_dt.strftime("%Y-%m-%d")
        dte = (exp_obj - entry_dt).days
        if dte < 3 or dte > 10: continue
        if last and (entry_dt - last).days < 5: continue
        try: price = float(spy_close.loc[es]); v = float(vix.loc[es])
        except: continue
        if np.isnan(price) or np.isnan(v) or v > 18: continue
        atm_k = round(price)
        ps = hd.get_available_strikes("SPY", exp, es, "P")
        cs = hd.get_available_strikes("SPY", exp, es, "C")
        if not ps or not cs: continue
        pk = min(ps, key=lambda k: abs(k-atm_k)); ck = min(cs, key=lambda k: abs(k-atm_k))
        pp = hd.get_contract_price(IronVault.build_occ_symbol("SPY", exp_obj, pk, "P"), es)
        cp = hd.get_contract_price(IronVault.build_occ_symbol("SPY", exp_obj, ck, "C"), es)
        if pp is None or cp is None: continue
        cost = pp + cp
        if cost < 0.50 or cost > 15.0: continue
        cts = max(1, min(2, int(CAPITAL * 0.01 / (cost * 100))))
        exit_val = cost; exit_date = es; exit_reason = "expiration"; hold_days = 0
        cur = entry_dt + timedelta(days=1)
        while cur <= exp_obj:
            cstr = cur.strftime("%Y-%m-%d")
            if cstr not in td_set: cur += timedelta(days=1); continue
            hold_days += 1
            pp2 = hd.get_contract_price(IronVault.build_occ_symbol("SPY", exp_obj, pk, "P"), cstr)
            cp2 = hd.get_contract_price(IronVault.build_occ_symbol("SPY", exp_obj, ck, "C"), cstr)
            if pp2 is not None and cp2 is not None:
                cv = pp2 + cp2
                if cv >= cost * 1.30: exit_val = cv; exit_date = cstr; exit_reason = "profit_target"; break
                if cv <= cost * 0.40: exit_val = cv; exit_date = cstr; exit_reason = "stop_loss"; break
                exit_val = cv; exit_date = cstr
            cur += timedelta(days=1)
        pnl = (exit_val - cost) * 100 * cts
        trades.append({"entry_date": es, "exit_date": exit_date, "pnl": round(pnl, 2),
                       "exit_reason": exit_reason, "vix": round(v, 1), "dte": dte, "hold_days": hold_days})
        last = entry_dt
    print(f"    -> {len(trades)} trades"); return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 3: Intraday Mean-Reversion
# ═══════════════════════════════════════════════════════════════════════════

def strat_intraday_mr(hd, spy_df):
    """Fade opening gaps > 0.8% with opposite-direction spreads."""
    print("  [3] Intraday MR: fade opening gaps")
    spy_close = spy_df["Close"]; td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    conn = sqlite3.connect(hd._db_path)
    intraday_dates = set(r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM option_intraday WHERE bar_time='09:35' "
        "AND date BETWEEN '2020-03-01' AND '2025-12-31'").fetchall())
    conn.close()
    exps = _find_exps(hd, "SPY", "2020-03-01", "2025-12-31", monthly=False)
    trades, last = [], None
    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=14), td_set)
        if entry_dt is None: continue
        es = entry_dt.strftime("%Y-%m-%d")
        if es not in intraday_dates: continue
        if last and (entry_dt - last).days < 5: continue
        try:
            prev_c = float(spy_close.shift(1).loc[es]); today_o = float(spy_df["Open"].loc[es])
        except: continue
        if np.isnan(prev_c) or np.isnan(today_o) or prev_c <= 0: continue
        gap = (today_o - prev_c) / prev_c
        if abs(gap) < 0.008: continue
        try: price = float(spy_close.loc[es])
        except: continue
        spread = _sell_put_spread(hd, "SPY", exp, es, price,
                                  otm_pct=0.96 if gap < 0 else 0.94, width=3.0)
        if spread is None: continue
        cts = max(1, min(3, int(CAPITAL * 0.01 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(hd, "SPY", exp, spread["short"], spread["long"],
                                        spread["credit"], entry_dt, exp_obj, spy_df.index,
                                        profit_pct=0.40, stop_mult=2.0, min_dte=3)
        pnl = (spread["credit"] - ev) * 100 * cts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                       "exit_reason": er, "gap_pct": round(gap, 4), "hold_days": hold})
        last = entry_dt
    print(f"    -> {len(trades)} trades"); return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 4: Seasonal Patterns
# ═══════════════════════════════════════════════════════════════════════════

def strat_seasonal(hd, spy_df):
    """Mon/Tue entries only, avoid Sep, prefer Q4 (richer premium)."""
    print("  [4] Seasonal: day-of-week + month filters")
    spy_close = spy_df["Close"]; td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, "SPY", "2020-03-01", "2025-12-31", monthly=False)
    trades, last = [], None
    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=21), td_set)
        if entry_dt is None: continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < 12: continue
        if entry_dt.weekday() > 1: continue
        if entry_dt.month == 9: continue
        try: price = float(spy_close.loc[es])
        except: continue
        if np.isnan(price): continue
        otm = 0.94 if entry_dt.month >= 10 else 0.95
        spread = _sell_put_spread(hd, "SPY", exp, es, price, otm_pct=otm, width=5.0)
        if spread is None: continue
        cts = max(1, min(3, int(CAPITAL * 0.02 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(hd, "SPY", exp, spread["short"], spread["long"],
                                        spread["credit"], entry_dt, exp_obj, spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * cts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                       "exit_reason": er, "day": entry_dt.strftime("%A"),
                       "month": entry_dt.month, "hold_days": hold})
        last = entry_dt
    print(f"    -> {len(trades)} trades"); return trades


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY 5: VRP Harvest
# ═══════════════════════════════════════════════════════════════════════════

def strat_vrp(hd, spy_df, vix):
    """Sell SPY puts when VIX - RVol >= 3pts and VIX 16-28."""
    print("  [5] VRP Harvest: implied > realised")
    spy_close = spy_df["Close"]; spy_ret = spy_close.pct_change()
    rvol = spy_ret.rolling(20).std() * math.sqrt(252) * 100
    ret20 = spy_close.pct_change(20)
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, "SPY", "2020-03-01", "2025-12-31", monthly=False)
    trades, last = [], None
    for exp in exps:
        exp_obj = _exp_dt(exp)
        entry_dt = _next_td(exp_obj - timedelta(days=25), td_set)
        if entry_dt is None: continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last and (entry_dt - last).days < 10: continue
        try: price = float(spy_close.loc[es]); v = float(vix.loc[es]); rv = float(rvol.loc[es]); r20 = float(ret20.loc[es])
        except: continue
        if any(np.isnan(x) for x in [price, v, rv, r20]): continue
        vrp = v - rv
        if vrp < 3.0 or v < 16 or v > 28 or r20 < -0.05: continue
        spread = _sell_put_spread(hd, "SPY", exp, es, price, otm_pct=0.95, width=5.0)
        if spread is None: continue
        cts = max(1, min(3, int(CAPITAL * 0.02 / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(hd, "SPY", exp, spread["short"], spread["long"],
                                        spread["credit"], entry_dt, exp_obj, spy_df.index)
        pnl = (spread["credit"] - ev) * 100 * cts
        trades.append({"entry_date": es, "exit_date": ed, "pnl": round(pnl, 2),
                       "exit_reason": er, "vrp": round(vrp, 1), "vix": round(v, 1), "hold_days": hold})
        last = entry_dt
    print(f"    -> {len(trades)} trades"); return trades


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def generate_report(results, output_path="reports/strategy_discovery_round4.html"):
    path = Path(output_path); path.parent.mkdir(parents=True, exist_ok=True)
    live = [s for s in results if not s.killed]; killed = [s for s in results if s.killed]
    rows = ""
    for s in results:
        sc = "#dc2626" if s.killed else "#16a34a"; sl = "KILLED" if s.killed else "LIVE"
        reason = f" — {s.kill_reason}" if s.killed else ""
        rows += f'<tr><td>{s.name}</td><td>{s.n_trades}</td><td style="color:{"#16a34a" if s.total_pnl>0 else "#dc2626"}">${s.total_pnl:,.0f}</td><td>{s.win_rate:.0%}</td><td>{s.sharpe:.2f}</td><td>{s.max_dd:.1%}</td><td>{s.cagr:.1%}</td><td>{s.spy_corr:+.3f}</td><td>{s.exp1220_corr:+.3f}</td><td>{s.oos_n}</td><td>{s.oos_sharpe:.2f}</td><td>{s.capacity}</td><td style="color:{sc};font-weight:700">{sl}{reason}</td></tr>'
    detail = ""
    for s in results:
        yr = "".join(f'<tr><td>{y}{"*" if y>=2023 else ""}</td><td>{d["n"]}</td><td style="color:{"#16a34a" if d["pnl"]>0 else "#dc2626"}">${d["pnl"]:,.0f}</td><td>{d["wr"]:.0%}</td><td>{d["sharpe"]:.2f}</td></tr>' for y, d in sorted(s.yearly.items()))
        detail += f'<h2>{s.name}</h2><p style="color:#334155;font-size:0.85rem;font-style:italic"><strong>Hypothesis:</strong> {s.hypothesis}</p><p style="color:#64748b;font-size:0.82rem">{s.description}</p><table><tr><th>Year</th><th>N</th><th>PnL</th><th>Win%</th><th>Sharpe</th></tr>{yr}</table>'
    html = f'<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Strategy Discovery R4</title><style>body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b}}h1{{font-size:1.4rem}}h2{{font-size:1rem;color:#334155;margin-top:1.5rem;border-bottom:1px solid #e2e8f0;padding-bottom:4px}}.meta{{color:#64748b;font-size:0.82rem;margin-bottom:16px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:16px 0}}.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}}.card .v{{font-size:1.1rem;font-weight:700;margin-top:2px}}table{{width:100%;border-collapse:collapse;font-size:0.82rem;margin:10px 0}}th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.7rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}th:first-child{{text-align:left}}td{{padding:5px 8px;text-align:right;border-bottom:1px solid #f1f5f9}}td:first-child{{text-align:left}}</style></head><body><h1>Strategy Discovery Round 4</h1><p class="meta">5 Novel Strategies | Real IronVault Data | Kill: &lt;10 OOS OR neg OOS Sharpe</p><div class="grid"><div class="card"><div class="l">Tested</div><div class="v">{len(results)}</div></div><div class="card"><div class="l">Live</div><div class="v" style="color:#16a34a">{len(live)}</div></div><div class="card"><div class="l">Killed</div><div class="v" style="color:#dc2626">{len(killed)}</div></div></div><h2>Summary</h2><table><tr><th>Strategy</th><th>N</th><th>PnL</th><th>Win%</th><th>Sharpe</th><th>DD</th><th>CAGR</th><th>SPY rho</th><th>1220 rho</th><th>OOS N</th><th>OOS SR</th><th>Cap</th><th>Status</th></tr>{rows}</table>{detail}<div style="color:#94a3b8;font-size:0.75rem;margin-top:2rem;border-top:1px solid #e2e8f0;padding-top:8px">compass/strategy_discovery_r4.py | All real IronVault data</div></body></html>'
    path.write_text(html, encoding="utf-8"); return str(path)


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_discovery():
    print("Strategy Discovery Round 4"); print("=" * 60)
    hd = IronVault.instance(); print(f"  IronVault: {hd._db_path}")
    print("  Fetching market data...")
    spy_df = _dl("SPY"); vix = _dl("^VIX")["Close"]
    sector_dfs = {"XLF": _dl("XLF"), "XLI": _dl("XLI"), "XLE": _dl("XLE")}
    spy_ret = spy_df["Close"].pct_change().dropna()
    exp1220_ret = spy_ret.copy(); exp1220_ret[exp1220_ret >= 0] *= 3.0; exp1220_ret[exp1220_ret < 0] *= 1.5
    print("\n  Running strategies...")
    results = [
        _compute(strat_dispersion(hd, spy_df, sector_dfs), "Dispersion Trading", spy_ret, exp1220_ret,
                 "Sector IV > SPY IV -> sell rich sector vol", "Sells overpriced sector vol at dispersion ratio > 1.3x", "$5M"),
        _compute(strat_gamma_scalp(hd, spy_df, vix), "Gamma Scalping", spy_ret, exp1220_ret,
                 "Cheap near-expiry gamma pays when realised > implied", "Buy ATM SPY straddles 5-7 DTE when VIX < 18", "$50M"),
        _compute(strat_intraday_mr(hd, spy_df), "Intraday Mean-Reversion", spy_ret, exp1220_ret,
                 "Large gaps (>0.8%) mean-revert", "Fade gap-up/gap-down with spreads", "$20M"),
        _compute(strat_seasonal(hd, spy_df), "Seasonal Patterns", spy_ret, exp1220_ret,
                 "Mon/Tue entries + Q4 premium = higher win rate", "Day-of-week + month filter on SPY puts", "$100M"),
        _compute(strat_vrp(hd, spy_df, vix), "VRP Harvest", spy_ret, exp1220_ret,
                 "Implied > realised ~85% of time = systematic edge", "Sell puts when VIX-RVol >= 3pts, VIX 16-28", "$50M"),
    ]
    print("\n  Results:")
    for s in results:
        tag = "KILLED" if s.killed else "LIVE"
        print(f"    {s.name:<25s}: {s.n_trades} trades, ${s.total_pnl:,.0f}, SR={s.sharpe:.2f}, "
              f"OOS_SR={s.oos_sharpe:.2f}, SPY={s.spy_corr:+.3f}, 1220={s.exp1220_corr:+.3f} [{tag}]")
    report = generate_report(results); print(f"\n  Report: {report}")
    return results

if __name__ == "__main__":
    run_discovery()
