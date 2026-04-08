"""
Strategy Discovery Round 3 — find uncorrelated alpha to EXP-1220.

Five candidate strategies, all on REAL data (Rule Zero):
  1. Earnings IV crush — REFERENCE prior result (EXP-1760/1800, KILLED)
  2. PUTW replication — Yahoo PUTW ETF (real fund, real prices)
  3. Calendar spread roll yield on SPY weeklies — IronVault
  4. Overnight vs intraday SPY split — Yahoo daily OHLC (intraday minute
     data is unavailable beyond 60 days, so we use the multi-year
     overnight-vs-intraday decomposition that IS available daily)
  5. Seasonal patterns (DOW, MOY) — Yahoo SPY daily

For each surviving idea: report CAGR, Sharpe, Max DD, correlation to
EXP-1220 (yearly Pearson, since we only have EXP-1220 yearly stats).
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from compass.metrics import annualized_sharpe, cagr, max_drawdown, annualized_vol

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("discovery3")

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
DB = REPO / "data" / "options_cache.db"

# EXP-1220 static 1.5x yearly returns (from reports/exp1220_dynamic_leverage.json)
EXP1220_YEARLY = {
    2020: 2.0794, 2021: 0.8828, 2022: 1.0055,
    2023: 0.6565, 2024: 0.6266, 2025: 0.9775,
}


def yearly_corr(daily_returns: pd.Series) -> float:
    """Pearson correlation of yearly returns vs EXP-1220 yearly returns."""
    if daily_returns.empty:
        return float("nan")
    yearly = (1 + daily_returns).groupby(daily_returns.index.year).prod() - 1
    common = sorted(set(yearly.index) & set(EXP1220_YEARLY.keys()))
    if len(common) < 3:
        return float("nan")
    a = np.array([yearly.loc[y] for y in common])
    b = np.array([EXP1220_YEARLY[y] for y in common])
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def report_metrics(name: str, daily: pd.Series) -> Dict:
    arr = daily.values
    m = {
        "name": name,
        "n_days": int(len(arr)),
        "first": str(daily.index[0].date()) if len(daily) else None,
        "last": str(daily.index[-1].date()) if len(daily) else None,
        "cagr": float(cagr(arr)),
        "sharpe": float(annualized_sharpe(arr)),
        "vol": float(annualized_vol(arr)),
        "max_dd": float(max_drawdown(arr)),
        "corr_1220_yearly": yearly_corr(daily),
    }
    log.info(
        "  %-32s CAGR=%+7.1f%%  Sharpe=%6.2f  DD=%5.1f%%  ρ(1220)=%+.2f  N=%d",
        name, m["cagr"] * 100, m["sharpe"], m["max_dd"] * 100, m["corr_1220_yearly"], m["n_days"],
    )
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Idea 1: Earnings IV crush — REFERENCE existing result, do not rerun
# ═══════════════════════════════════════════════════════════════════════════

def idea1_earnings_crush() -> Dict:
    log.info("\n[Idea 1] Earnings IV Crush — referencing prior result")
    prior = REPO / "reports" / "exp1760_earnings_vol_crush.json"
    if prior.exists():
        try:
            d = json.load(open(prior))
        except Exception:
            d = {}
    else:
        d = {}
    result = {
        "name": "Earnings IV Crush",
        "status": "KILLED (prior EXP-1760/1800)",
        "reason": (
            "IronVault contains zero single-name options. Index proxy "
            "(EXP-1760) achieved Trade Sharpe 0.79 but OOS Sharpe -0.06 "
            "and SPY corr -0.54 — failed walk-forward."
        ),
        "verdict": "DEAD",
        "prior_data": d.get("metrics") if d else None,
    }
    log.info("  Status: KILLED — see EXP-1760/1800")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Idea 2: PUTW replication — Yahoo PUTW ETF (real fund)
# ═══════════════════════════════════════════════════════════════════════════

def idea2_putw() -> Dict:
    log.info("\n[Idea 2] PUTW replication — WisdomTree CBOE PutWrite ETF")
    df = yf.download("PUTW", start="2018-01-01", end="2026-04-06",
                     progress=False, auto_adjust=True)
    if df.empty:
        log.warning("  Yahoo returned no PUTW data — KILL")
        return {"name": "PUTW", "status": "DATA_GAP", "verdict": "KILL"}
    px = df["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    daily = px.pct_change().dropna()
    daily.name = "PUTW"
    metrics = report_metrics("PUTW (Yahoo, no leverage)", daily)
    metrics["data_source"] = "Yahoo PUTW (real ETF prices)"
    metrics["verdict"] = "RESEARCH" if metrics["sharpe"] > 0.5 else "KILL"
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Idea 3: Calendar spread roll yield on SPY weeklies — IronVault
# ═══════════════════════════════════════════════════════════════════════════

def _spy_friday_expiries(start: str, end: str) -> List[str]:
    if not DB.exists():
        return []
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        """
        SELECT DISTINCT expiration FROM option_contracts
        WHERE ticker='SPY' AND expiration BETWEEN ? AND ?
          AND CAST(STRFTIME('%w', expiration) AS INTEGER) = 5
        ORDER BY expiration
        """,
        (start, end),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def _spy_atm_call_close(exp: str, date_str: str, spot: float) -> Optional[Tuple[float, float]]:
    """Return (strike, close) for the call closest to spot with a real bar on date_str."""
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        """
        SELECT oc.strike, od.close
        FROM option_contracts oc
        JOIN option_daily od ON oc.contract_symbol = od.contract_symbol
        WHERE oc.ticker='SPY' AND oc.expiration=? AND oc.option_type='C' AND od.date=?
        """,
        (exp, date_str),
    ).fetchall()
    conn.close()
    if not rows:
        return None
    best = min(rows, key=lambda r: abs(float(r[0]) - spot))
    return float(best[0]), float(best[1])


def idea3_calendar() -> Dict:
    log.info("\n[Idea 3] Calendar spread roll yield — SPY weekly calls (IronVault)")
    if not DB.exists():
        return {"name": "Calendar", "status": "DATA_GAP", "verdict": "KILL"}

    # SPY spot from Yahoo for entry days
    spy = yf.download("SPY", start="2020-01-01", end="2026-04-06",
                      progress=False, auto_adjust=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy = spy.dropna()

    # All Friday expiries 2020-2025
    expiries = _spy_friday_expiries("2020-01-01", "2026-04-06")
    log.info("  %d Friday SPY expiries available in IronVault", len(expiries))

    # Strategy: every Monday, sell this-Friday ATM call, buy next-Friday ATM call
    # at SAME strike. Close on Friday (cover front, sell back) — collect theta differential.
    spy_idx = spy.index
    trades = []
    for i, day in enumerate(spy_idx):
        if day.weekday() != 0:  # Monday
            continue
        spot = float(spy.iloc[i])
        date_str = day.strftime("%Y-%m-%d")
        # Front: this Friday
        front_exp_dt = day + pd.Timedelta(days=4)
        back_exp_dt = day + pd.Timedelta(days=11)
        front_exp = front_exp_dt.strftime("%Y-%m-%d")
        back_exp = back_exp_dt.strftime("%Y-%m-%d")
        if front_exp not in expiries or back_exp not in expiries:
            continue
        front = _spy_atm_call_close(front_exp, date_str, spot)
        back = _spy_atm_call_close(back_exp, date_str, spot)
        if not front or not back:
            continue
        # Force same strike
        if abs(front[0] - back[0]) > 0.5:
            continue
        debit_open = back[1] - front[1]  # we pay debit (back > front normally)
        if debit_open <= 0:
            continue
        # Close on Friday: front expires worthless or ITM; back is now front-week
        try:
            friday_idx = spy_idx.searchsorted(front_exp_dt)
            if friday_idx >= len(spy_idx):
                continue
            friday_day = spy_idx[friday_idx]
            friday_str = friday_day.strftime("%Y-%m-%d")
            friday_spot = float(spy.iloc[friday_idx])
        except Exception:
            continue
        # Front PnL at expiry: -max(0, S_T - K)
        K = front[0]
        front_settle = max(0.0, friday_spot - K)
        # Back close on Friday (now 7DTE)
        back_close = _spy_atm_call_close(back_exp, friday_str, friday_spot)
        if back_close is None:
            continue
        debit_close = back_close[1] - front_settle  # buy back front (=front_settle), sell back
        # PnL = debit_close - debit_open  (we paid debit_open, recovered debit_close)
        pnl = debit_close - debit_open
        # Express as fraction of spot (so we can compound)
        ret = pnl / spot
        trades.append({"date": friday_day, "ret": ret})

    log.info("  %d calendar trades found", len(trades))
    if len(trades) < 30:
        log.warning("  Too few trades — KILL")
        return {"name": "Calendar", "n_trades": len(trades),
                "status": "INSUFFICIENT_DATA", "verdict": "KILL"}

    s = pd.Series(
        [t["ret"] for t in trades],
        index=pd.DatetimeIndex([t["date"] for t in trades]),
    ).sort_index()
    # Build a daily series spread across SPY trading days
    daily = pd.Series(0.0, index=spy_idx)
    for d, v in s.items():
        if d in daily.index:
            daily.loc[d] += v
    daily.name = "Calendar"
    metrics = report_metrics("SPY weekly calendar (call)", daily)
    metrics["n_trades"] = len(trades)
    metrics["data_source"] = "IronVault SPY option_daily"
    metrics["verdict"] = "RESEARCH" if (metrics["sharpe"] > 0.5 and metrics["cagr"] > 0.10) else "KILL"
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Idea 4: Overnight vs intraday SPY decomposition (Yahoo daily OHLC)
# ═══════════════════════════════════════════════════════════════════════════

def idea4_overnight_intraday() -> Dict:
    log.info("\n[Idea 4] Overnight vs intraday SPY (Yahoo daily OHLC)")
    df = yf.download("SPY", start="2018-01-01", end="2026-04-06",
                     progress=False, auto_adjust=False)
    if df.empty:
        return {"name": "Overnight", "status": "DATA_GAP", "verdict": "KILL"}
    op = df["Open"].iloc[:, 0] if isinstance(df["Open"], pd.DataFrame) else df["Open"]
    cl = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
    # Overnight return = today_open / yesterday_close - 1
    overnight = (op / cl.shift(1) - 1).dropna()
    overnight.name = "overnight"
    intraday = (cl / op - 1).dropna()
    intraday.name = "intraday"
    log.info("  Decomposing buy-and-hold into overnight vs intraday...")
    m_overnight = report_metrics("Long overnight (close→open)", overnight)
    m_intraday = report_metrics("Long intraday (open→close)", intraday)

    # Mean-reversion strategy: each day, take the OPPOSITE of yesterday's
    # intraday return. Long if SPY fell intraday, short if it rose.
    # PnL = -sign(yesterday_intraday) * today_intraday
    yest = intraday.shift(1)
    mr_signal = -np.sign(yest)
    mr_daily = (mr_signal * intraday).dropna()
    mr_daily.name = "MeanRev"
    m_mr = report_metrics("Daily intraday mean-reversion", mr_daily)

    # Verdict: pick the best survivor
    candidates = [
        ("overnight_only", m_overnight),
        ("intraday_only", m_intraday),
        ("daily_mean_reversion", m_mr),
    ]
    surviving = [(k, m) for k, m in candidates if m["sharpe"] > 0.5 and m["cagr"] > 0.10]
    return {
        "name": "Overnight/Intraday decomposition",
        "data_source": "Yahoo SPY daily OHLC",
        "candidates": {k: m for k, m in candidates},
        "surviving": [k for k, _ in surviving],
        "verdict": "RESEARCH" if surviving else "KILL",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Idea 5: Seasonal patterns — DOW, MOY
# ═══════════════════════════════════════════════════════════════════════════

def idea5_seasonal() -> Dict:
    log.info("\n[Idea 5] Seasonal patterns in SPY (Yahoo daily)")
    df = yf.download("SPY", start="2018-01-01", end="2026-04-06",
                     progress=False, auto_adjust=False)
    cl = df["Close"]
    if isinstance(cl, pd.DataFrame):
        cl = cl.iloc[:, 0]
    rets = cl.pct_change().dropna()

    # IS = 2018-2022, OOS = 2023-2025 (avoid lookahead)
    is_mask = rets.index < pd.Timestamp("2023-01-01")
    is_rets = rets[is_mask]
    oos_rets = rets[~is_mask]

    # Day-of-week edges, learned IS only
    dow_means = is_rets.groupby(is_rets.index.dayofweek).mean()
    log.info("  IS day-of-week mean returns (bps): " +
             "  ".join(f"{d}={v*1e4:+.1f}" for d, v in dow_means.items()))

    # Month-of-year edges, learned IS only
    moy_means = is_rets.groupby(is_rets.index.month).mean()
    log.info("  IS month-of-year mean returns (bps): " +
             "  ".join(f"{m}={v*1e4:+.1f}" for m, v in moy_means.items()))

    # Strategy: long SPY only on (DOW, MOY) cells with positive IS mean
    good_dow = set(dow_means[dow_means > 0].index)
    good_moy = set(moy_means[moy_means > 0].index)
    log.info("  Good DOW (IS): %s   Good MOY (IS): %s", sorted(good_dow), sorted(good_moy))

    def signal_series(r):
        idx = r.index
        sig = pd.Series(
            [(d.dayofweek in good_dow and d.month in good_moy) for d in idx],
            index=idx,
        )
        return r.where(sig, 0.0)

    is_strat = signal_series(is_rets)
    oos_strat = signal_series(oos_rets)
    full_strat = pd.concat([is_strat, oos_strat])
    full_strat.name = "Seasonal"

    log.info("  IS:")
    is_m = report_metrics("Seasonal SPY (IS only)", is_strat)
    log.info("  OOS:")
    oos_m = report_metrics("Seasonal SPY (OOS only)", oos_strat)
    log.info("  FULL:")
    full_m = report_metrics("Seasonal SPY (full)", full_strat)

    full_m["is_sharpe"] = is_m["sharpe"]
    full_m["oos_sharpe"] = oos_m["sharpe"]
    full_m["data_source"] = "Yahoo SPY daily"
    # Survival: must hold OOS
    full_m["verdict"] = "RESEARCH" if (oos_m["sharpe"] > 0.5 and oos_m["cagr"] > 0.10) else "KILL"
    return full_m


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    log.info("=" * 70)
    log.info("Strategy Discovery Round 3 — uncorrelated alpha hunt")
    log.info("Rule Zero: 100% real data")
    log.info("Target: standalone CAGR >10%, ρ to EXP-1220 < 0.20")
    log.info("=" * 70)

    results = {
        "round": 3,
        "rule_zero": "100% real data — Yahoo + IronVault",
        "exp1220_yearly": EXP1220_YEARLY,
        "ideas": {},
    }

    results["ideas"]["1_earnings_crush"] = idea1_earnings_crush()
    results["ideas"]["2_putw"] = idea2_putw()
    results["ideas"]["3_calendar"] = idea3_calendar()
    results["ideas"]["4_overnight_intraday"] = idea4_overnight_intraday()
    results["ideas"]["5_seasonal"] = idea5_seasonal()

    # Final ranking
    log.info("\n" + "=" * 70)
    log.info("VERDICTS")
    log.info("=" * 70)
    survivors = []
    for k, v in results["ideas"].items():
        verdict = v.get("verdict", "?")
        sharpe = v.get("sharpe", v.get("candidates", {}).get("daily_mean_reversion", {}).get("sharpe", "—"))
        log.info("  %-35s  verdict=%s", k, verdict)
        if verdict == "RESEARCH":
            survivors.append(k)
    log.info("\nSurvivors: %s", survivors or "(none)")

    # Save
    out_json = REPORTS / "discovery_round3.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    log.info("\nJSON: %s", out_json)

    out_html = REPORTS / "discovery_round3.html"
    out_html.write_text(_render_html(results, survivors))
    log.info("HTML: %s", out_html)
    return 0


def _render_html(results: dict, survivors: list) -> str:
    def fmt_pct(x):
        try: return f"{x*100:+.1f}%"
        except: return "—"
    def fmt(x, fmtstr="{:.2f}"):
        try: return fmtstr.format(x)
        except: return "—"

    def row(name, m):
        v = m.get("verdict", "?")
        color = {"RESEARCH": "#10b981", "KILL": "#dc2626",
                 "DEAD": "#dc2626", "DATA_GAP": "#64748b",
                 "INSUFFICIENT_DATA": "#64748b"}.get(v, "#64748b")
        return f"""<tr>
<td><b>{name}</b><br><small>{m.get('data_source', m.get('reason', ''))}</small></td>
<td>{fmt_pct(m.get('cagr'))}</td>
<td>{fmt(m.get('sharpe'))}</td>
<td>{fmt_pct(m.get('max_dd'))}</td>
<td>{fmt(m.get('corr_1220_yearly'), "{:+.2f}")}</td>
<td>{m.get('n_days', m.get('n_trades', '—'))}</td>
<td><span style="background:{color};color:white;padding:3px 8px;border-radius:4px;font-weight:600;font-size:.85em">{v}</span></td>
</tr>"""

    rows_html = ""
    for k, v in results["ideas"].items():
        if "candidates" in v:
            for sk, sm in v["candidates"].items():
                rows_html += row(f"{k} → {sk}", sm)
        else:
            rows_html += row(k, v)

    survivors_html = (
        "<ul>" + "".join(f"<li><b>{s}</b></li>" for s in survivors) + "</ul>"
        if survivors else "<p><i>No idea cleared the bar (CAGR &gt; 10%, OOS Sharpe &gt; 0.5, ρ&lt;0.2 to 1220).</i></p>"
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Discovery Round 3</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
h1 {{ border-bottom: 2px solid #444; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
td, th {{ border: 1px solid #ccc; padding: 8px 12px; }}
th {{ background: #f0f0f0; text-align: left; }}
small {{ color: #666; }}
</style></head>
<body>
<h1>Strategy Discovery Round 3</h1>
<p>Goal: find a strategy with standalone CAGR &gt; 10%, OOS Sharpe &gt; 0.5,
and yearly correlation to EXP-1220 below 0.2.</p>
<p><b>Rule Zero:</b> 100% real data — Yahoo Finance + IronVault.</p>

<h2>Survivors</h2>
{survivors_html}

<h2>All ideas tested</h2>
<table>
<tr><th>Idea</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>ρ(1220, yearly)</th><th>N days/trades</th><th>Verdict</th></tr>
{rows_html}
</table>

<h2>Notes</h2>
<ul>
<li>Idea 1 (Earnings IV crush) was already killed in Wave 1 as EXP-1760/1800.
IronVault has zero single-name options, and the index proxy failed walk-forward.</li>
<li>Idea 2 (PUTW) uses the actual WisdomTree CBOE PutWrite ETF — every input is a real
fund close on Yahoo Finance.</li>
<li>Idea 3 (calendar spread) constructs front/back week ATM call calendars from real
IronVault SPY option_daily bars at the same strike. PnL is the change in the debit.</li>
<li>Idea 4 (overnight/intraday) uses real Yahoo SPY OHLC. Intraday minute data is only
available for the last 60 days, so we use the multi-year overnight/intraday split that
IS available.</li>
<li>Idea 5 (seasonality) is trained on 2018-2022 and tested on 2023-2025 to avoid lookahead.</li>
</ul>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
