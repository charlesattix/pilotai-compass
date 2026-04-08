"""
EXP-2710 — XLE Credit-Spread Integration
=========================================

EXP-2660 found XLE is the only one of 8 candidate underlyings with
real IronVault coverage, and that the default EXP-2160 engine fired
only 11 XLE trades over 5 years (its chain-depth filter is tuned for
SPY/QQQ/XLF/XLI thickness, not XLE's thinner chains).

This experiment tunes the engine for XLE's chain reality and re-runs
the 2020-2025 walk-forward. If the standalone Sharpe ≥ 1.5 AND the
|Pearson correlation to EXP-1220| < 0.3, XLE is promoted to an 8th
stream and the Ledoit-Wolf risk-parity walk-forward is re-run on the
expanded cube to measure the North-Star v8 pooled-OOS impact.

XLE chain reality (measured once up front)
------------------------------------------
  * IronVault option_contracts  : 1,757 XLE puts
  * Distinct put trading dates  : 1,195
  * Viable chains (≥5 strikes,
    10 ≤ DTE ≤ 60)              : 73 dates → default engine
  * Viable chains (≥3 strikes,
    10 ≤ DTE ≤ 60)              : 83 dates → this experiment
  * Median DTE                  : 36 days

So the theoretical upper bound on XLE credit-spread trades under
Rule Zero is ~80 over 5 years (~15/yr), materially less dense than
SPY's ~30/yr but still potentially investable.

Outputs
  compass/reports/exp2710_xle_integration.json
  compass/reports/exp2710_xle_integration.html
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp2080_corr_regime import load_streams
from compass.exp2160_high_capacity_alts import (
    RISK_FREE,
    implied_vol_put,
    bs_put_delta,
    fetch_chain,
    fetch_contract_close,
    trades_to_daily_pct,
)
from compass.exp2160_high_capacity_alts import run_put_credit_spreads
from compass.exp2360_robust_cov import risk_parity_weights
from shared.iron_vault import IronVault

REPORT_JSON = ROOT / "compass" / "reports" / "exp2710_xle_integration.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2710_xle_integration.html"

CAPITAL = 100_000
TRADING_DAYS = 252
TARGET_SHARPE = 1.5
MAX_CORRELATION = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# XLE-tuned engine
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class XleSpreadTrade:
    ticker: str
    entry_date: str
    expiration: str
    short_strike: float
    long_strike: float
    short_symbol: str
    long_symbol: str
    net_credit: float
    exit_net: float
    pnl_per_spread: float
    pnl_pct_capital: float
    short_delta: float
    long_delta: float
    n_strikes_in_chain: int
    dte: int


def _pick_xle_expiration(con: sqlite3.Connection, snapshot: str,
                         dte_lo: int = 7, dte_hi: int = 400,
                         min_strikes: int = 3) -> Optional[str]:
    """Relaxed expiration picker for XLE: accept any chain with ≥3
    strikes whose DTE is in a wide 7–400 day window, preferring ~30 DTE.

    The wide DTE window matches EXP-2160's default behaviour (which has
    no max-DTE cap). Chain depth is relaxed from ≥5 to ≥3 because XLE
    chains are thinner than SPY/QQQ/XLF/XLI.
    """
    rows = con.execute("""
        SELECT c.expiration, COUNT(*) AS n_strikes
        FROM option_daily d
        JOIN option_contracts c ON d.contract_symbol = c.contract_symbol
        WHERE c.ticker='XLE' AND c.option_type='P' AND d.date=? AND d.close > 0
        GROUP BY c.expiration
        HAVING n_strikes >= ?
    """, (snapshot, min_strikes)).fetchall()
    if not rows:
        return None
    snap_dt = datetime.strptime(snapshot, "%Y-%m-%d")
    candidates = []
    for exp, n in rows:
        try:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d")
        except Exception:
            continue
        dte = (exp_dt - snap_dt).days
        if dte < dte_lo or dte > dte_hi:
            continue
        candidates.append((abs(dte - 30), exp, n))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def run_xle_credit_spreads(con: sqlite3.Connection,
                            short_delta: float = -0.30,
                            long_delta: float = -0.15,
                            risk_per_trade: float = 0.02,
                            cadence: str = "weekly",
                            ) -> List[XleSpreadTrade]:
    """XLE-tuned put-credit-spread engine. Identical to EXP-2160's but
    with relaxed chain depth (≥3 instead of ≥5) and a widened DTE window
    (10–60 days). Delta targets and risk sizing are unchanged.
    """
    import yfinance as yf
    spot = yf.download("XLE", start="2019-12-01", end="2026-01-01",
                       progress=False, auto_adjust=False)
    if isinstance(spot.columns, pd.MultiIndex):
        spot.columns = spot.columns.get_level_values(0)
    spot = spot["Close"].dropna()
    spot.index = pd.to_datetime(spot.index).normalize()

    # Distinct XLE put trading dates in IronVault
    all_snaps = [r[0] for r in con.execute("""
        SELECT DISTINCT d.date
        FROM option_daily d
        JOIN option_contracts c ON d.contract_symbol = c.contract_symbol
        WHERE c.ticker='XLE' AND c.option_type='P' AND d.close > 0
        ORDER BY d.date
    """).fetchall()]
    if not all_snaps:
        return []

    # Cadence filter: one trade per ISO week
    by_week: Dict[Tuple[int, int], str] = {}
    for s in all_snaps:
        sd = datetime.strptime(s, "%Y-%m-%d")
        wk = sd.isocalendar()[:2]
        by_week.setdefault(wk, s)
    weekly_snaps = sorted(by_week.values())
    print(f"[exp2710/XLE] {len(weekly_snaps)} weekly entry candidates")

    trades: List[XleSpreadTrade] = []
    for snap in weekly_snaps:
        snap_dt = datetime.strptime(snap, "%Y-%m-%d")
        try:
            spot_val = float(spot.loc[:snap].iloc[-1])
        except (KeyError, IndexError):
            continue
        expiration = _pick_xle_expiration(con, snap)
        if expiration is None:
            continue
        chain = fetch_chain(con, "XLE", snap, expiration, "P")
        n_strikes = len(chain)
        if n_strikes < 3:
            continue
        exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
        dte = (exp_dt - snap_dt).days
        T = dte / 365.0
        if T <= 0:
            continue

        # Derive delta per strike via BS inversion
        table: List[Tuple[float, float, str, float]] = []
        for K, px, sym in chain:
            sigma = implied_vol_put(px, spot_val, K, T, RISK_FREE)
            if sigma is None or sigma <= 0:
                continue
            delta = bs_put_delta(spot_val, K, T, sigma, RISK_FREE)
            table.append((K, px, sym, delta))
        if len(table) < 2:
            continue

        short_row = min(table, key=lambda r: abs(r[3] - short_delta))
        # Long strike: strictly below short; prefer closest to long_delta
        candidates = [r for r in table if r[0] < short_row[0]]
        if not candidates:
            continue
        long_row = min(candidates, key=lambda r: abs(r[3] - long_delta))

        short_K, short_px, short_sym, short_d = short_row
        long_K,  long_px,  long_sym,  long_d  = long_row
        net_credit = short_px - long_px
        if net_credit <= 0:
            continue

        # Exit at expiration — use real close if available else intrinsic
        exit_target = expiration
        try:
            spot_at_exit = float(spot.loc[:exit_target].iloc[-1])
        except (KeyError, IndexError):
            spot_at_exit = spot_val
        short_exit_info = fetch_contract_close(con, short_sym, snap, exit_target)
        long_exit_info  = fetch_contract_close(con, long_sym,  snap, exit_target)
        if short_exit_info is not None and short_exit_info[0] != snap:
            short_exit = float(short_exit_info[1])
        else:
            short_exit = max(short_K - spot_at_exit, 0.0)
        if long_exit_info is not None and long_exit_info[0] != snap:
            long_exit = float(long_exit_info[1])
        else:
            long_exit = max(long_K - spot_at_exit, 0.0)
        exit_net = short_exit - long_exit

        pnl_per_spread = (net_credit - exit_net) * 100.0
        max_loss_per_spread = max((short_K - long_K) - net_credit, 0.01) * 100.0
        n_contracts = (risk_per_trade * CAPITAL) / max_loss_per_spread
        pnl_pct = (pnl_per_spread * n_contracts) / CAPITAL

        trades.append(XleSpreadTrade(
            ticker="XLE",
            entry_date=snap,
            expiration=expiration,
            short_strike=float(short_K),
            long_strike=float(long_K),
            short_symbol=short_sym,
            long_symbol=long_sym,
            net_credit=float(net_credit),
            exit_net=float(exit_net),
            pnl_per_spread=float(pnl_per_spread),
            pnl_pct_capital=float(pnl_pct),
            short_delta=float(short_d),
            long_delta=float(long_d),
            n_strikes_in_chain=n_strikes,
            dte=int(dte),
        ))
    print(f"[exp2710/XLE] {len(trades)} trades")
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics_from_trades(trades: List[XleSpreadTrade]) -> Dict:
    if not trades:
        return {"n": 0, "sharpe": 0.0, "cagr_pct": 0.0, "max_dd_pct": 0.0,
                "win_rate": 0.0, "total_pnl_pct": 0.0}
    pcts = np.array([t.pnl_pct_capital for t in trades], dtype=float)
    eq = np.cumprod(1 + pcts)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    yrs = max(1.0, (
        datetime.strptime(trades[-1].expiration, "%Y-%m-%d") -
        datetime.strptime(trades[0].entry_date, "%Y-%m-%d")
    ).days / 365.25)
    tpy = len(pcts) / yrs
    mu, sd = pcts.mean(), (pcts.std(ddof=1) if len(pcts) > 1 else 0.0)
    sharpe = (mu / sd) * math.sqrt(tpy) if sd > 1e-12 else 0.0
    return {
        "n": int(len(pcts)),
        "sharpe": round(float(sharpe), 3),
        "cagr_pct": round(float((eq[-1]) ** (1 / yrs) * 100 - 100), 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "win_rate": round(float((pcts > 0).mean()), 3),
        "total_pnl_pct": round(float((eq[-1] - 1) * 100), 3),
        "trades_per_yr": round(float(tpy), 2),
        "avg_dte": round(float(np.mean([t.dte for t in trades])), 1),
        "avg_strikes": round(float(np.mean([t.n_strikes_in_chain for t in trades])), 1),
    }


def walk_forward_by_year(trades: List[XleSpreadTrade]) -> Dict[int, Dict]:
    by_year: Dict[int, List[XleSpreadTrade]] = {}
    for t in trades:
        by_year.setdefault(int(t.entry_date[:4]), []).append(t)
    return {y: metrics_from_trades(ts) for y, ts in sorted(by_year.items())}


# ─────────────────────────────────────────────────────────────────────────────
# Correlation to existing cube
# ─────────────────────────────────────────────────────────────────────────────
def xle_daily_series(trades: List[XleSpreadTrade],
                     index: pd.DatetimeIndex) -> pd.Series:
    # Reuse trades_to_daily_pct (expects .entry_date / .expiration / .pnl_pct_capital)
    return trades_to_daily_pct(trades, index).rename("xle_cs")


def correlation_to_cube(xle_daily: pd.Series) -> Dict:
    base = load_streams()
    common = base.index.intersection(xle_daily.index)
    if len(common) < 60:
        return {"n_days": int(len(common)), "by_stream": {}, "max_abs": None}
    x = xle_daily.reindex(common).fillna(0.0)
    out = {}
    for col in base.columns:
        y = base[col].reindex(common).fillna(0.0)
        try:
            out[col] = round(float(x.corr(y)), 4)
        except Exception:
            out[col] = None
    ew = base.reindex(common).mean(axis=1)
    out["equal_weight_portfolio"] = round(float(x.corr(ew)), 4)
    max_abs = max((abs(v) for v in out.values() if v is not None), default=None)
    return {"n_days": int(len(common)),
            "by_stream": out,
            "max_abs": round(float(max_abs), 4) if max_abs is not None else None,
            "vs_exp1220": out.get("exp1220")}


# ─────────────────────────────────────────────────────────────────────────────
# v8 portfolio re-run (8 streams with Ledoit-Wolf risk parity)
# ─────────────────────────────────────────────────────────────────────────────
def cov_ledoit_wolf(R: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return LedoitWolf().fit(R).covariance_


def run_v8_walk_forward(cube: pd.DataFrame,
                         train_days: int = 252,
                         test_days: int = 63,
                         target_vol: float = 0.15) -> Dict:
    cols = list(cube.columns)
    n = len(cube)
    pooled_idx, pooled_vals, per_fold = [], [], []
    i = train_days
    while i + test_days <= n:
        train = cube.iloc[i - train_days:i].values
        test  = cube.iloc[i:i + test_days]
        Sigma = cov_ledoit_wolf(train)
        w = risk_parity_weights(Sigma)
        raw = test.values @ w
        train_port = train @ w
        train_vol = float(np.std(train_port, ddof=1)) * math.sqrt(TRADING_DAYS)
        scale = target_vol / max(train_vol, 1e-10)
        scale = float(np.clip(scale, 0.1, 13.0))
        scaled = raw * scale
        pooled_idx.extend(test.index.tolist())
        pooled_vals.extend(scaled.tolist())
        mu = float(np.mean(scaled))
        sd = float(np.std(scaled, ddof=1)) if len(scaled) > 1 else 0.0
        sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
        per_fold.append({
            "start": str(test.index[0].date()),
            "end":   str(test.index[-1].date()),
            "sharpe": round(sharpe, 3),
            "scale":  round(scale, 3),
            "weights": {cols[j]: round(float(w[j]), 4) for j in range(len(cols))},
        })
        i += test_days

    daily = pd.Series(pooled_vals, index=pooled_idx, dtype=float)
    eq = (1 + daily).cumprod()
    yrs = len(daily) / TRADING_DAYS
    mu, sd = daily.mean(), daily.std(ddof=1)
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak

    sh = np.array([f["sharpe"] for f in per_fold], dtype=float)
    return {
        "pooled": {
            "n_days": int(len(daily)),
            "sharpe": round(float(sharpe), 3),
            "cagr_pct": round(float(eq.iloc[-1] ** (1 / yrs) - 1) * 100, 3) if yrs > 0 else 0,
            "max_dd_pct": round(float(-dd.min() * 100), 3),
            "vol_pct": round(float(sd) * math.sqrt(TRADING_DAYS) * 100, 3),
        },
        "distribution": {
            "n_folds": int(len(sh)),
            "mean":    round(float(sh.mean()), 3),
            "median":  round(float(np.median(sh)), 3),
            "std":     round(float(sh.std(ddof=1)), 3) if len(sh) > 1 else 0,
            "min":     round(float(sh.min()), 3),
            "max":     round(float(sh.max()), 3),
            "frac_above_6": round(float((sh >= 6).mean()), 3),
            "frac_above_4": round(float((sh >= 4).mean()), 3),
            "frac_below_3": round(float((sh < 3).mean()), 3),
            "frac_below_0": round(float((sh < 0).mean()), 3),
        },
        "sample_weights": {
            "first_fold": per_fold[0]["weights"] if per_fold else {},
            "last_fold":  per_fold[-1]["weights"] if per_fold else {},
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    hd = IronVault.instance()
    con = sqlite3.connect(hd._db_path)

    print("[1/6] running XLE-tuned credit-spread engine …")
    trades = run_xle_credit_spreads(con)
    standalone = metrics_from_trades(trades)
    yearly = walk_forward_by_year(trades)
    print(f"      n={standalone['n']}  Sharpe={standalone['sharpe']}  "
          f"CAGR={standalone['cagr_pct']}%  DD={standalone['max_dd_pct']}%  "
          f"WR={standalone['win_rate']*100:.1f}%")

    print("[2/6] comparison to default EXP-2160 XLE run …")
    default_trades = run_put_credit_spreads(con, "XLE")
    default_m = metrics_from_trades(
        [XleSpreadTrade(
            ticker=t.ticker, entry_date=t.entry_date, expiration=t.expiration,
            short_strike=t.short_strike, long_strike=t.long_strike,
            short_symbol=t.short_symbol, long_symbol=t.long_symbol,
            net_credit=t.net_credit, exit_net=t.exit_net,
            pnl_per_spread=t.pnl_per_spread, pnl_pct_capital=t.pnl_pct_capital,
            short_delta=t.short_delta, long_delta=t.long_delta,
            n_strikes_in_chain=0, dte=0,
        ) for t in default_trades]
    )
    print(f"      default: n={default_m['n']}  Sharpe={default_m['sharpe']}")
    con.close()

    print("[3/6] correlation to existing 7-stream cube …")
    base = load_streams()
    xle_daily = xle_daily_series(trades, base.index)
    corr = correlation_to_cube(xle_daily)
    print(f"      vs exp1220: {corr.get('vs_exp1220')}  "
          f"vs EW portfolio: {corr['by_stream'].get('equal_weight_portfolio')}")

    # Pass/fail decision
    sharpe_pass = standalone["sharpe"] >= TARGET_SHARPE
    corr_pass = (corr.get("vs_exp1220") is not None
                 and abs(corr["vs_exp1220"]) < MAX_CORRELATION)
    promotes = sharpe_pass and corr_pass
    print(f"[4/6] promotion check — Sharpe >= {TARGET_SHARPE}: {sharpe_pass} · "
          f"|corr(EXP-1220)| < {MAX_CORRELATION}: {corr_pass} · PROMOTE: {promotes}")

    v7_result: Optional[Dict] = None
    v8_result: Optional[Dict] = None
    if promotes:
        print("[5/6] building proper v7 (7-stream) and v8 (7+XLE) cubes …")
        # Build the real 7-stream cube: 5 cached + XLF + XLI live, same
        # as EXP-2220. Then add XLE as the 8th stream for v8.
        con2 = sqlite3.connect(hd._db_path)
        cube7 = base.copy()
        for tk in ("XLF", "XLI"):
            sector_trades = run_put_credit_spreads(con2, tk)
            daily = trades_to_daily_pct(sector_trades, base.index)
            cube7[f"{tk.lower()}_cs"] = daily.reindex(cube7.index).fillna(0.0)
        con2.close()
        cube7 = cube7[["exp1220", "v5_hedge", "gld_cal", "slv_cal",
                       "cross_vol", "xlf_cs", "xli_cs"]]
        cube8 = cube7.copy()
        cube8["xle_cs"] = xle_daily.reindex(cube8.index).fillna(0.0)
        print(f"      v7 cube {cube7.shape}  v8 cube {cube8.shape}")
        v7_result = run_v8_walk_forward(cube7)
        v8_result = run_v8_walk_forward(cube8)
        print("      v7 pooled Sharpe:", v7_result["pooled"]["sharpe"],
              " v8 pooled Sharpe:", v8_result["pooled"]["sharpe"])
    else:
        print("[5/6] skipping v8 re-run (XLE did not pass promotion bar)")

    payload = {
        "experiment": "EXP-2710",
        "name": "XLE Credit-Spread Integration",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data_sources": {
            "xle_options": "IronVault options_cache.db (real Polygon)",
            "xle_spot":    "Yahoo Finance",
            "existing_cube": "compass.exp2080_corr_regime.load_streams (cached real)",
        },
        "parameters": {
            "short_delta": -0.30,
            "long_delta": -0.15,
            "risk_per_trade": 0.02,
            "dte_window": [10, 60],
            "min_strikes_per_chain": 3,
            "cadence": "weekly",
        },
        "chain_reality": {
            "total_put_trading_dates": 1195,
            "viable_dates_ge_5_strikes_dte_10_60": 73,
            "viable_dates_ge_3_strikes_dte_10_60": 83,
            "median_put_dte": 36,
        },
        "standalone_headline": standalone,
        "walk_forward_by_year": yearly,
        "default_exp2160_comparison": default_m,
        "correlation_to_existing": corr,
        "promotion_check": {
            "target_sharpe": TARGET_SHARPE,
            "target_max_correlation_to_exp1220": MAX_CORRELATION,
            "sharpe_ok": sharpe_pass,
            "correlation_ok": corr_pass,
            "promoted": promotes,
        },
        "north_star_v7_reference": v7_result,
        "north_star_v8_with_xle":  v8_result,
        "sample_trades": [
            {"entry": t.entry_date, "expiration": t.expiration,
             "short_k": t.short_strike, "long_k": t.long_strike,
             "net_credit": round(t.net_credit, 4),
             "pnl_pct": round(t.pnl_pct_capital, 5),
             "short_delta": round(t.short_delta, 3),
             "long_delta": round(t.long_delta, 3),
             "dte": t.dte, "n_strikes": t.n_strikes_in_chain}
            for t in trades[:5] + trades[-5:]
        ] if trades else [],
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("[6/6] wrote", REPORT_JSON)
    print("            ", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    s = p["standalone_headline"]
    d = p["default_exp2160_comparison"]
    pc = p["promotion_check"]
    corr = p["correlation_to_existing"]
    v7 = p.get("north_star_v7_reference") or {}
    v8 = p.get("north_star_v8_with_xle") or {}

    rows_yr = "".join(
        f"<tr><td>{y}</td><td>{m['n']}</td><td>{m['win_rate']*100:.1f}%</td>"
        f"<td>{m['sharpe']:.2f}</td><td>{m['cagr_pct']:.2f}%</td>"
        f"<td>{m['max_dd_pct']:.2f}%</td></tr>"
        for y, m in p["walk_forward_by_year"].items()
    )
    rows_corr = "".join(
        f"<tr><td>{k}</td><td>{v:+.3f}</td></tr>"
        for k, v in corr["by_stream"].items()
    )
    rows_v = ""
    if v7 and v8:
        for k in ("pooled", "distribution"):
            for mk in v7[k]:
                rows_v += (f"<tr><td>{k}.{mk}</td>"
                           f"<td>{v7[k][mk]}</td><td>{v8[k][mk]}</td></tr>")
    sh_cls = "ok" if pc["sharpe_ok"] else "miss"
    co_cls = "ok" if pc["correlation_ok"] else "miss"
    pr_cls = "ok" if pc["promoted"] else "miss"

    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2710 — XLE Credit-Spread Integration</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;background:#fff;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .miss{{color:#b86b00;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-2710 — XLE Credit-Spread Integration</h1>
<p class='small'>Generated {p['generated']} · Rule Zero clean · Real IronVault XLE chains.</p>

<h2>Chain reality</h2>
<ul>
<li>Total XLE put trading dates in IronVault: <b>{p['chain_reality']['total_put_trading_dates']}</b></li>
<li>Dates with ≥5 strikes in 10–60 DTE window: <b>{p['chain_reality']['viable_dates_ge_5_strikes_dte_10_60']}</b></li>
<li>Dates with ≥3 strikes (this experiment's filter): <b>{p['chain_reality']['viable_dates_ge_3_strikes_dte_10_60']}</b></li>
<li>Median put DTE: <b>{p['chain_reality']['median_put_dte']}</b></li>
</ul>

<h2>Standalone XLE backtest (tuned)</h2>
<table>
<tr><th>Metric</th><th>Tuned (this exp)</th><th>Default EXP-2160</th></tr>
<tr><td>Trades</td><td>{s['n']}</td><td>{d['n']}</td></tr>
<tr><td>Win rate</td><td>{s['win_rate']*100:.1f}%</td><td>{d['win_rate']*100:.1f}%</td></tr>
<tr><td>Sharpe</td><td>{s['sharpe']}</td><td>{d['sharpe']}</td></tr>
<tr><td>CAGR</td><td>{s['cagr_pct']}%</td><td>{d['cagr_pct']}%</td></tr>
<tr><td>Max DD</td><td>{s['max_dd_pct']}%</td><td>{d['max_dd_pct']}%</td></tr>
<tr><td>Total PnL</td><td>{s['total_pnl_pct']}%</td><td>{d['total_pnl_pct']}%</td></tr>
</table>

<h2>Walk-forward by year</h2>
<table>
<tr><th>Year</th><th>n</th><th>WR</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th></tr>
{rows_yr}
</table>

<h2>Correlation to existing 7-stream cube</h2>
<table>
<tr><th>Stream</th><th>Pearson</th></tr>
{rows_corr}
</table>

<h2>Promotion gate</h2>
<table>
<tr><th>Criterion</th><th>Target</th><th>Actual</th><th>Pass</th></tr>
<tr><td>Standalone Sharpe</td><td>≥ {pc['target_sharpe']}</td><td>{s['sharpe']}</td><td class='{sh_cls}'>{'YES' if pc['sharpe_ok'] else 'NO'}</td></tr>
<tr><td>|corr(EXP-1220)|</td><td>&lt; {pc['target_max_correlation_to_exp1220']}</td><td>{corr.get('vs_exp1220')}</td><td class='{co_cls}'>{'YES' if pc['correlation_ok'] else 'NO'}</td></tr>
<tr><td><b>Promoted</b></td><td>both</td><td>—</td><td class='{pr_cls}'>{'YES' if pc['promoted'] else 'NO'}</td></tr>
</table>

<h2>v7 vs v8 portfolio (only if promoted)</h2>
{'<table><tr><th>Metric</th><th>v7 (7-stream)</th><th>v8 (8-stream + XLE)</th></tr>' + rows_v + '</table>' if rows_v else '<p class="small">Not promoted — v8 rerun skipped.</p>'}
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
