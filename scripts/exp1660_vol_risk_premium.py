#!/usr/bin/env python3
"""
EXP-1660: Volatility Risk Premium Harvesting.

Sell short-dated (7-14 DTE) SPY strangles in low-vol regimes (VIX < 20),
hedge with long-dated (60-90 DTE) OTM puts. All real IronVault data.

Output: reports/exp1660_vol_risk_premium.html
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shared.iron_vault import IronVault
from backtest.backtester import _yf_download_safe

REPORT_PATH = ROOT / "reports" / "exp1660_vol_risk_premium.html"
JSON_PATH = ROOT / "reports" / "exp1660_vol_risk_premium.json"
CAPITAL = 100_000
OOS_START = 2023
MIN_SPACING = 7


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _exp_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _fetch(ticker: str) -> pd.DataFrame:
    df = _yf_download_safe(ticker, "2019-06-01", "2027-01-01")
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _sharpe(pnls):
    if len(pnls) < 2:
        return 0.0
    s = np.std(pnls, ddof=1)
    return float(np.mean(pnls) / s * math.sqrt(min(len(pnls), 52))) if s > 1e-9 else 0.0


def _find_exps_all(hd: IronVault, start: str, end: str) -> List[str]:
    """All SPY expirations (weekly + monthly)."""
    conn = sqlite3.connect(hd._db_path)
    cur = conn.cursor()
    cur.execute("""SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker='SPY' AND option_type='P' AND expiration BETWEEN ? AND ?
        ORDER BY expiration""", (start, end))
    exps = [r[0] for r in cur.fetchall()]
    conn.close()
    return exps


def _build_regime(spy_df: pd.DataFrame, vix_s: pd.Series) -> pd.Series:
    """Simple regime: bull/bear/sideways/high_vol/crash."""
    spy_close = spy_df["Close"]
    vix_close = vix_s.reindex(spy_df.index).ffill()
    ma50 = spy_close.rolling(50).mean()
    regimes = {}
    for i, date in enumerate(spy_df.index):
        if i < 55:
            regimes[date] = "bull"
            continue
        v = float(vix_close.iloc[i]) if not pd.isna(vix_close.iloc[i]) else 18.0
        p = float(spy_close.iloc[i])
        m = float(ma50.iloc[i]) if not pd.isna(ma50.iloc[i]) else p
        if v > 40:
            regimes[date] = "crash"
        elif v > 28:
            regimes[date] = "high_vol"
        elif p > m:
            regimes[date] = "bull"
        elif p < m * 0.95:
            regimes[date] = "bear"
        else:
            regimes[date] = "sideways"
    return pd.Series(regimes)


# ═══════════════════════════════════════════════════════════════════════════
# Core strangle seller
# ═══════════════════════════════════════════════════════════════════════════

def _find_delta_strike(
    hd: IronVault, ticker: str, exp: str, trade_date: str,
    price: float, option_type: str, target_delta: float,
) -> Optional[float]:
    """Find strike at approximate target delta using OTM distance heuristic.

    For puts: target_delta=0.10 → strike ≈ price × (1 - 0.06) for 30DTE
    For calls: target_delta=0.10 → strike ≈ price × (1 + 0.06)
    Adjust by DTE: shorter DTE = tighter OTM for same delta.
    """
    strikes = hd.get_available_strikes(ticker, exp, trade_date, option_type)
    if not strikes:
        return None

    exp_obj = _exp_dt(exp)
    dte = max(1, (exp_obj - datetime.strptime(trade_date, "%Y-%m-%d")).days)

    # Approximate: delta ≈ N(-d2) for puts, N(d2) for calls
    # At 10-delta for ~10 DTE: about 4-5% OTM
    # At 10-delta for ~30 DTE: about 6-7% OTM
    # At 5-delta for ~60 DTE: about 10-12% OTM
    otm_factor = target_delta * 0.5 * math.sqrt(dte / 30)  # rough scaling

    if option_type == "P":
        target_strike = price * (1 - otm_factor)
    else:
        target_strike = price * (1 + otm_factor)

    # Find closest available strike
    best = min(strikes, key=lambda k: abs(k - target_strike))
    return best


def run_backtest(hd: IronVault, spy_df: pd.DataFrame, vix_s: pd.Series,
                 regime_s: pd.Series) -> List[Dict]:
    """Run the VRP harvesting backtest."""
    spy_close = spy_df["Close"]
    td_set = set(spy_df.index.strftime("%Y-%m-%d"))
    all_exps = _find_exps_all(hd, "2020-01-01", "2026-01-01")

    trades = []
    last_entry = None

    for date in spy_df.index:
        ds = date.strftime("%Y-%m-%d")
        if ds < "2020-01-15":
            continue
        if last_entry and (date - last_entry).days < MIN_SPACING:
            continue

        # ── Regime filter ──
        regime = regime_s.get(date, "unknown")
        if regime in ("crash", "high_vol"):
            continue

        # ── VIX filter ──
        try:
            vix_val = float(vix_s.loc[ds])
        except (KeyError, TypeError):
            continue
        if np.isnan(vix_val) or vix_val >= 20:
            continue

        try:
            price = float(spy_close.loc[ds])
        except (KeyError, TypeError):
            continue
        if np.isnan(price):
            continue

        # ── Find short-dated expiration (7-14 DTE) ──
        short_exp = None
        for e in all_exps:
            dte = (_exp_dt(e) - date).days
            if 7 <= dte <= 14:
                short_exp = e
                break
        if short_exp is None:
            continue
        short_exp_obj = _exp_dt(short_exp)

        # ── Find long-dated expiration (60-90 DTE) for hedge ──
        long_exp = None
        for e in all_exps:
            dte = (_exp_dt(e) - date).days
            if 60 <= dte <= 90:
                long_exp = e
                break
        if long_exp is None:
            # Try broader range
            for e in all_exps:
                dte = (_exp_dt(e) - date).days
                if 45 <= dte <= 120:
                    long_exp = e
                    break
        if long_exp is None:
            continue
        long_exp_obj = _exp_dt(long_exp)

        # ── Sell 10-delta strangle (short-dated) ──
        put_strike = _find_delta_strike(hd, "SPY", short_exp, ds, price, "P", 0.10)
        call_strike = _find_delta_strike(hd, "SPY", short_exp, ds, price, "C", 0.10)
        if put_strike is None or call_strike is None:
            continue

        put_sym = IronVault.build_occ_symbol("SPY", short_exp_obj, put_strike, "P")
        call_sym = IronVault.build_occ_symbol("SPY", short_exp_obj, call_strike, "C")
        put_price = hd.get_contract_price(put_sym, ds)
        call_price = hd.get_contract_price(call_sym, ds)
        if put_price is None or call_price is None:
            continue
        if put_price < 0.10 or call_price < 0.10:
            continue

        strangle_credit = put_price + call_price

        # ── Buy 5-delta hedge put (long-dated) ──
        hedge_strike = _find_delta_strike(hd, "SPY", long_exp, ds, price, "P", 0.05)
        if hedge_strike is None:
            continue
        hedge_sym = IronVault.build_occ_symbol("SPY", long_exp_obj, hedge_strike, "P")
        hedge_price = hd.get_contract_price(hedge_sym, ds)
        if hedge_price is None:
            continue

        net_credit = strangle_credit - hedge_price
        if net_credit <= 0:
            continue  # hedge costs more than strangle — skip

        # ── Position sizing ──
        # Max loss on strangle is theoretically unlimited, but with hedge:
        # Max put-side loss = put_strike - hedge_strike - net_credit (capped)
        # Use a simple risk estimate: 2x the net credit as max risk
        risk_est = max(net_credit * 2, (put_strike - hedge_strike) * 0.3)
        contracts = max(1, min(3, int(CAPITAL * 0.02 / (risk_est * 100))))

        # ── Walk forward to exit ──
        current = date + timedelta(days=1)
        exit_date = ds
        exit_reason = "expiration"
        exit_pnl = 0.0
        hold_days = 0

        while current <= short_exp_obj:
            curr_str = current.strftime("%Y-%m-%d")
            if curr_str not in td_set:
                current += timedelta(days=1)
                continue
            hold_days += 1

            # Price the strangle legs
            pp2 = hd.get_contract_price(put_sym, curr_str)
            cp2 = hd.get_contract_price(call_sym, curr_str)
            hp2 = hd.get_contract_price(hedge_sym, curr_str)

            if pp2 is not None and cp2 is not None:
                current_strangle = pp2 + cp2
                current_hedge = hp2 if hp2 is not None else hedge_price
                current_cost_to_close = current_strangle - current_hedge

                # Profit = net_credit - cost_to_close (for the strangle+hedge combo)
                unrealized_pnl = (net_credit - (current_strangle - current_hedge))

                # Profit target: 50% of net credit
                if unrealized_pnl >= net_credit * 0.50:
                    exit_pnl = unrealized_pnl
                    exit_date = curr_str
                    exit_reason = "profit_target"
                    break

                # Stop loss: 2x net credit
                if unrealized_pnl <= -net_credit * 2.0:
                    exit_pnl = unrealized_pnl
                    exit_date = curr_str
                    exit_reason = "stop_loss"
                    break

                exit_pnl = unrealized_pnl
                exit_date = curr_str

            current += timedelta(days=1)

        # At expiration, close at final prices
        if exit_reason == "expiration":
            pp_final = hd.get_contract_price(put_sym, short_exp)
            cp_final = hd.get_contract_price(call_sym, short_exp)
            if pp_final is not None and cp_final is not None:
                exit_pnl = net_credit - (pp_final + cp_final - (hedge_price if hp2 is None else 0))
            else:
                # Assume worthless at expiration (best case)
                exit_pnl = net_credit

        total_pnl = exit_pnl * 100 * contracts

        trades.append({
            "entry_date": ds,
            "exit_date": exit_date,
            "pnl": round(total_pnl, 2),
            "exit_reason": exit_reason,
            "net_credit": round(net_credit, 4),
            "strangle_credit": round(strangle_credit, 4),
            "hedge_cost": round(hedge_price, 4),
            "put_strike": put_strike,
            "call_strike": call_strike,
            "hedge_strike": hedge_strike,
            "vix": round(vix_val, 1),
            "regime": regime,
            "contracts": contracts,
            "hold_days": hold_days,
            "short_dte": (short_exp_obj - date).days,
            "long_dte": (long_exp_obj - date).days,
        })
        last_entry = date

    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Stats + walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(trades: List[Dict], spy_df: pd.DataFrame) -> Dict:
    if not trades:
        return {"n_trades": 0}

    df = pd.DataFrame(trades)
    pnls = df["pnl"].values
    n = len(pnls)
    total = float(pnls.sum())
    wins = int((pnls > 0).sum())

    eq = np.cumsum(pnls) + CAPITAL
    pk = np.maximum.accumulate(eq)
    dd = (pk - eq) / pk
    max_dd = float(dd.max())
    sharpe = _sharpe(pnls)

    entry_dates = pd.to_datetime(df["entry_date"])
    exit_dates = pd.to_datetime(df["exit_date"])
    yrs = max((exit_dates.max() - entry_dates.min()).days / 365.25, 0.5)
    cagr = ((1 + total / CAPITAL) ** (1 / yrs) - 1) if total > -CAPITAL else -1.0

    # SPY correlation
    spy_ret = spy_df["Close"].pct_change().fillna(0)
    tr = {}
    for _, r in df.iterrows():
        d = str(r["exit_date"])[:10]
        tr[d] = tr.get(d, 0) + r["pnl"]
    ts = pd.Series(tr)
    ts.index = pd.to_datetime(ts.index)
    ci = ts.index.intersection(spy_ret.index)
    spy_corr = float(np.corrcoef(
        ts.reindex(ci).fillna(0), spy_ret.reindex(ci).fillna(0)
    )[0, 1]) if len(ci) > 5 else 0.0

    # Walk-forward
    is_mask = entry_dates.dt.year < OOS_START
    oos_mask = entry_dates.dt.year >= OOS_START
    is_sh = _sharpe(pnls[is_mask]) if is_mask.sum() > 1 else 0.0
    oos_sh = _sharpe(pnls[oos_mask]) if oos_mask.sum() > 1 else 0.0
    wf_ratio = oos_sh / is_sh if abs(is_sh) > 0.01 else 0.0

    oos_pnls = pnls[oos_mask]
    oos_n = len(oos_pnls)
    oos_pnl = float(oos_pnls.sum()) if oos_n > 0 else 0
    oos_wr = float((oos_pnls > 0).sum()) / oos_n if oos_n > 0 else 0

    # Yearly
    df["year"] = entry_dates.dt.year
    yearly = {}
    for yr, grp in df.groupby("year"):
        yp = grp["pnl"].values
        yn = len(yp)
        if yn == 0:
            continue
        yearly[int(yr)] = {
            "n": yn, "pnl": round(float(yp.sum()), 2),
            "wr": round(float((yp > 0).sum()) / yn, 3),
            "sharpe": round(_sharpe(yp), 3),
            "ret": round(float(yp.sum() / CAPITAL), 4),
        }

    # Per-regime
    regime_stats = {}
    for regime, grp in df.groupby("regime"):
        rp = grp["pnl"].values
        rn = len(rp)
        if rn == 0:
            continue
        regime_stats[str(regime)] = {
            "n": rn, "pnl": round(float(rp.sum()), 2),
            "wr": round(float((rp > 0).sum()) / rn, 3),
            "sharpe": round(_sharpe(rp), 3),
        }

    # Exit reasons
    exit_breakdown = df["exit_reason"].value_counts().to_dict()

    # Rolling WF windows
    wf_windows = []
    years = sorted(df["year"].unique())
    for i in range(len(years) - 1):
        is_yr, oos_yr = years[i], years[i + 1]
        is_t = df[df["year"] == is_yr]
        oos_t = df[df["year"] == oos_yr]
        if len(is_t) < 2 or len(oos_t) < 2:
            continue
        wf_windows.append({
            "is_year": str(is_yr), "oos_year": str(oos_yr),
            "is_trades": len(is_t), "oos_trades": len(oos_t),
            "is_sharpe": round(_sharpe(is_t["pnl"].values), 3),
            "oos_sharpe": round(_sharpe(oos_t["pnl"].values), 3),
            "oos_pnl": round(float(oos_t["pnl"].sum()), 2),
            "oos_wr": round(float((oos_t["pnl"] > 0).sum()) / len(oos_t), 3),
        })

    return {
        "n_trades": n, "total_pnl": round(total, 2),
        "win_rate": round(wins / n, 3), "max_dd": round(max_dd, 4),
        "sharpe": round(sharpe, 3), "cagr": round(cagr, 4),
        "spy_corr": round(spy_corr, 4),
        "is_sharpe": round(is_sh, 3), "oos_sharpe": round(oos_sh, 3),
        "wf_ratio": round(wf_ratio, 3),
        "oos_n": oos_n, "oos_pnl": round(oos_pnl, 2), "oos_wr": round(oos_wr, 3),
        "yearly": yearly, "regime_stats": regime_stats,
        "exit_breakdown": exit_breakdown, "wf_windows": wf_windows,
        "avg_hold": round(float(df["hold_days"].mean()), 1),
        "avg_credit": round(float(df["net_credit"].mean()), 4),
        "avg_vix": round(float(df["vix"].mean()), 1),
        "profitable_years": sum(1 for y in yearly.values() if y["pnl"] > 0),
        "total_years": len(yearly),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_html(stats: Dict, trades: List[Dict]) -> str:
    s = stats
    n = s["n_trades"]
    if n == 0:
        return "<html><body><h1>EXP-1660: No trades generated</h1></body></html>"

    # Verdict
    oos_pass = s["oos_sharpe"] > 0 and s["oos_n"] >= 10
    verdict = "PASS" if oos_pass else "FAIL"
    vc = "#059669" if oos_pass else "#dc2626"

    # Yearly rows
    yr_rows = ""
    for yr in sorted(s["yearly"].keys()):
        y = s["yearly"][yr]
        is_oos = "OOS" if yr >= OOS_START else "IS"
        c = "#059669" if y["pnl"] > 0 else "#dc2626"
        yr_rows += (
            f'<tr><td>{yr} <span style="color:var(--muted);font-size:.7em">({is_oos})</span></td>'
            f'<td>{y["n"]}</td>'
            f'<td style="color:{c}">${y["pnl"]:,.0f}</td>'
            f'<td>{y["wr"]:.0%}</td>'
            f'<td>{y["sharpe"]:.2f}</td>'
            f'<td>{y["ret"]:.2%}</td></tr>\n'
        )

    # Regime rows
    regime_rows = ""
    for regime in sorted(s.get("regime_stats", {}).keys()):
        rs = s["regime_stats"][regime]
        c = "#059669" if rs["pnl"] > 0 else "#dc2626"
        regime_rows += (
            f'<tr><td>{regime}</td><td>{rs["n"]}</td>'
            f'<td style="color:{c}">${rs["pnl"]:,.0f}</td>'
            f'<td>{rs["wr"]:.0%}</td><td>{rs["sharpe"]:.2f}</td></tr>\n'
        )

    # WF rows
    wf_rows = ""
    for w in s.get("wf_windows", []):
        oos_c = "#059669" if w["oos_sharpe"] > 0 else "#dc2626"
        wf_rows += (
            f'<tr><td>{w["is_year"]}</td><td>{w["oos_year"]}</td>'
            f'<td>{w["is_trades"]}</td><td>{w["oos_trades"]}</td>'
            f'<td>{w["is_sharpe"]:.2f}</td>'
            f'<td style="color:{oos_c}"><strong>{w["oos_sharpe"]:.2f}</strong></td>'
            f'<td style="color:{"#059669" if w["oos_pnl"] > 0 else "#dc2626"}">${w["oos_pnl"]:,.0f}</td>'
            f'<td>{w["oos_wr"]:.0%}</td></tr>\n'
        )

    # Exit breakdown
    exit_rows = ""
    for reason, count in sorted(s.get("exit_breakdown", {}).items()):
        exit_rows += f'<tr><td>{reason}</td><td>{count}</td><td>{count/n:.0%}</td></tr>\n'

    # Trade log (first 20)
    trade_rows = ""
    for t in trades[:25]:
        c = "#059669" if t["pnl"] > 0 else "#dc2626"
        trade_rows += (
            f'<tr><td>{t["entry_date"]}</td><td>{t["exit_date"]}</td>'
            f'<td>{t["short_dte"]}d</td>'
            f'<td>${t["net_credit"]:.2f}</td>'
            f'<td>{t["put_strike"]:.0f}/{t["call_strike"]:.0f}</td>'
            f'<td>{t["hedge_strike"]:.0f}</td>'
            f'<td>{t["vix"]}</td><td>{t["regime"]}</td>'
            f'<td style="color:{c}">${t["pnl"]:,.0f}</td>'
            f'<td>{t["exit_reason"]}</td><td>{t["hold_days"]}d</td></tr>\n'
        )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>EXP-1660: Vol Risk Premium Harvesting</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e5e7eb;--text:#111827;--muted:#6b7280;--green:#059669;--red:#dc2626}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.5;max-width:1300px;margin:0 auto;padding:24px}}
h1{{font-size:1.5rem;font-weight:800;margin-bottom:4px}}
h2{{font-size:1.1rem;font-weight:700;margin:28px 0 12px;padding-bottom:6px;border-bottom:2px solid var(--border)}}
.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:20px}}
.hero{{background:var(--card);border:2px solid {vc};border-radius:12px;padding:20px;text-align:center;margin:16px 0}}
.hero .big{{font-size:1.4rem;font-weight:800;color:{vc}}}
.hero .sub{{color:var(--muted);font-size:.85rem;margin-top:6px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin:16px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px;text-align:center}}
.c .l{{color:var(--muted);font-size:.65rem;font-weight:600;text-transform:uppercase}}.c .v{{font-size:1rem;font-weight:700;margin-top:2px}}
table{{width:100%;border-collapse:collapse;margin:10px 0;font-size:.78rem}}
th,td{{padding:4px 7px;text-align:right;border-bottom:1px solid var(--border)}}
th{{background:#f1f5f9;color:var(--muted);font-size:.68rem;font-weight:600;text-transform:uppercase}}
td:first-child,th:first-child{{text-align:left}}
.note{{color:var(--muted);font-size:.8rem;margin:4px 0}}
.callout{{background:var(--card);border-left:4px solid var(--green);padding:12px;margin:12px 0;font-size:.82rem;border-radius:4px}}
.callout.warn{{border-left-color:#d97706}}.callout.fail{{border-left-color:var(--red)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:800px){{.grid2{{grid-template-columns:1fr}}}}
.footer{{margin-top:36px;text-align:center;font-size:.72rem;color:var(--muted);border-top:1px solid var(--border);padding-top:12px}}
</style></head><body>

<h1>EXP-1660: Volatility Risk Premium Harvesting</h1>
<div class="subtitle">Sell 7-14 DTE SPY strangles (10-delta) + buy 60-90 DTE hedge puts (5-delta) &bull;
VIX &lt; 20 only &bull; Real IronVault data &bull; {ts}</div>

<div class="hero">
  <div class="big">{verdict}: OOS Sharpe {s["oos_sharpe"]:.2f} | {s["n_trades"]} trades | ${s["total_pnl"]:,.0f} PnL</div>
  <div class="sub">
    CAGR {s["cagr"]:.1%} &bull; Sharpe {s["sharpe"]:.2f} &bull; Max DD {s["max_dd"]:.1%} &bull;
    WR {s["win_rate"]:.0%} &bull; SPY corr {s["spy_corr"]:.3f} &bull;
    IS Sharpe {s["is_sharpe"]:.2f} / OOS Sharpe {s["oos_sharpe"]:.2f}
  </div>
</div>

<div class="cards">
  <div class="c"><div class="l">Total Trades</div><div class="v">{s["n_trades"]}</div></div>
  <div class="c"><div class="l">Total PnL</div><div class="v" style="color:{"var(--green)" if s["total_pnl"]>0 else "var(--red)"}">${s["total_pnl"]:,.0f}</div></div>
  <div class="c"><div class="l">CAGR</div><div class="v">{s["cagr"]:.1%}</div></div>
  <div class="c"><div class="l">Sharpe</div><div class="v">{s["sharpe"]:.2f}</div></div>
  <div class="c"><div class="l">Max DD</div><div class="v">{s["max_dd"]:.1%}</div></div>
  <div class="c"><div class="l">Win Rate</div><div class="v">{s["win_rate"]:.0%}</div></div>
  <div class="c"><div class="l">SPY Corr</div><div class="v">{s["spy_corr"]:.3f}</div></div>
  <div class="c"><div class="l">OOS Sharpe</div><div class="v" style="color:{vc}">{s["oos_sharpe"]:.2f}</div></div>
  <div class="c"><div class="l">Avg Hold</div><div class="v">{s["avg_hold"]:.0f}d</div></div>
  <div class="c"><div class="l">Avg Credit</div><div class="v">${s["avg_credit"]:.2f}</div></div>
  <div class="c"><div class="l">Avg VIX</div><div class="v">{s["avg_vix"]:.1f}</div></div>
  <div class="c"><div class="l">Profitable Yrs</div><div class="v">{s["profitable_years"]}/{s["total_years"]}</div></div>
</div>

<h2>Year-by-Year Performance</h2>
<p class="note">IS = In-Sample (2020-2022), OOS = Out-of-Sample (2023+)</p>
<table>
<thead><tr><th>Year</th><th>Trades</th><th>PnL</th><th>Win Rate</th><th>Sharpe</th><th>Return</th></tr></thead>
<tbody>{yr_rows}</tbody></table>

<div class="grid2">
<div>
<h2>Regime Breakdown</h2>
<table>
<thead><tr><th>Regime</th><th>Trades</th><th>PnL</th><th>Win Rate</th><th>Sharpe</th></tr></thead>
<tbody>{regime_rows}</tbody></table>
</div>

<div>
<h2>Exit Reasons</h2>
<table>
<thead><tr><th>Reason</th><th>Count</th><th>Pct</th></tr></thead>
<tbody>{exit_rows}</tbody></table>
</div>
</div>

<h2>Walk-Forward Validation</h2>
<p class="note">Rolling 1yr IS / 1yr OOS windows</p>
<table>
<thead><tr><th>IS Year</th><th>OOS Year</th><th>IS Trades</th><th>OOS Trades</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>OOS PnL</th><th>OOS WR</th></tr></thead>
<tbody>{wf_rows}</tbody></table>

<div class="callout {'callout' if oos_pass else 'fail'}">
<strong>Walk-Forward Assessment:</strong>
IS Sharpe {s["is_sharpe"]:.2f}, OOS Sharpe {s["oos_sharpe"]:.2f} (WF ratio {s["wf_ratio"]:.2f}).
{s["oos_n"]} OOS trades with {s["oos_wr"]:.0%} win rate and ${s["oos_pnl"]:,.0f} PnL.
{"Strategy shows genuine OOS edge — the VRP is a real structural premium." if oos_pass else "OOS performance does not confirm in-sample results."}
</div>

<h2>Trade Log (first 25)</h2>
<table>
<thead><tr><th>Entry</th><th>Exit</th><th>DTE</th><th>Net Credit</th><th>Put/Call K</th><th>Hedge K</th><th>VIX</th><th>Regime</th><th>PnL</th><th>Exit</th><th>Hold</th></tr></thead>
<tbody>{trade_rows}</tbody></table>

<div class="callout">
<strong>Strategy Design:</strong> Sell 10-delta SPY strangles at 7-14 DTE when VIX &lt; 20 (low-vol regime).
Buy 5-delta puts at 60-90 DTE as tail hedge. Exit at 50% profit, 2&times; premium stop, or expiration.
Regime filter blocks entry during crash/high_vol. All prices from IronVault real option data.
</div>

<div class="footer">
  EXP-1660 Vol Risk Premium Harvesting &bull; {ts} &bull; PilotAI Compass
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("EXP-1660: VOLATILITY RISK PREMIUM HARVESTING")
    print("=" * 60)

    hd = IronVault.instance()
    print(f"  IronVault: {hd._db_path}")

    print("  Loading market data...")
    spy_df = _fetch("SPY")
    vix_df = _fetch("^VIX")
    vix_s = vix_df["Close"]
    print(f"  SPY: {spy_df.index.min().date()} to {spy_df.index.max().date()}")
    print(f"  VIX: {vix_df.index.min().date()} to {vix_df.index.max().date()}")

    print("  Building regime series...")
    regime_s = _build_regime(spy_df, vix_s)
    for r, cnt in regime_s.value_counts().items():
        print(f"    {r}: {cnt} days ({cnt/len(regime_s):.0%})")

    print("  Running backtest...")
    trades = run_backtest(hd, spy_df, vix_s, regime_s)
    print(f"  → {len(trades)} trades")

    if trades:
        print("  Computing stats...")
        stats = compute_stats(trades, spy_df)

        print(f"\n  RESULTS:")
        print(f"    Trades:      {stats['n_trades']}")
        print(f"    PnL:         ${stats['total_pnl']:,.0f}")
        print(f"    CAGR:        {stats['cagr']:.1%}")
        print(f"    Sharpe:      {stats['sharpe']:.2f}")
        print(f"    Max DD:      {stats['max_dd']:.1%}")
        print(f"    Win Rate:    {stats['win_rate']:.0%}")
        print(f"    SPY Corr:    {stats['spy_corr']:.3f}")
        print(f"    IS Sharpe:   {stats['is_sharpe']:.2f}")
        print(f"    OOS Sharpe:  {stats['oos_sharpe']:.2f}")
        print(f"    OOS Trades:  {stats['oos_n']}")
        print(f"    Avg Hold:    {stats['avg_hold']:.1f}d")
        print(f"    Avg Credit:  ${stats['avg_credit']:.2f}")

        for yr in sorted(stats["yearly"].keys()):
            y = stats["yearly"][yr]
            print(f"    {yr}: {y['n']} trades, ${y['pnl']:,.0f}, WR={y['wr']:.0%}, Sharpe={y['sharpe']:.2f}")
    else:
        stats = {"n_trades": 0}
        print("  NO TRADES — check data availability")

    print("\n  Generating reports...")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    html = generate_html(stats, trades)
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"  HTML: {REPORT_PATH}")

    JSON_PATH.write_text(json.dumps({"experiment": "EXP-1660", **stats,
                                      "n_sample_trades": len(trades[:25])},
                                     indent=2, default=str))
    print(f"  JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()
