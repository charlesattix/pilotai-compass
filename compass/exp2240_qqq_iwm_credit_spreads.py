"""
compass/exp2240_qqq_iwm_credit_spreads.py — EXP-2240 QQQ and IWM Credit Spreads.

CONTEXT: EXP-1220 (SPY), XLF, and XLI put-credit-spread strategies were
validated on real IronVault option data. This experiment tests whether
the SAME framework (30-day monthly entries, 5%-OTM short put, $5-wide
spread, 50% profit target, 2× stop, VIX<40 gate) works on QQQ and IWM.

QUESTION: If QQQ and IWM produce Grade A/B OOS results, we have a
9-stream portfolio (SPY, XLF, XLI, QQQ, IWM calendars + GLD/SLV
calendars + Crisis Alpha + cross-vol) with meaningfully larger
capacity.

DATA POLICY (Rule Zero):
  • All option prices from IronVault option_daily (real CBOE/Polygon
    data cached in data/options_cache.db).
  • Underlying spot from Yahoo Finance.
  • No synthetic prices, no Black-Scholes, no random fills. If a
    ticker has zero IronVault contracts, the script reports the
    honest data gap and does not attempt to simulate.

STRATEGY (parameters match EXP-1220 exactly):
  • Monthly cadence, min 10d spacing between entries
  • 28-day target DTE
  • 5% OTM short put, $5-wide spread (QQQ/IWM use $5 default; we also
    try $2 for IWM since it trades at a lower dollar price)
  • 3% risk per trade, contracts capped 1..4
  • Profit target 50% of credit
  • Stop loss 2× credit
  • DTE exit at 7 days
  • VIX > 40 blocks new entries

WALK-FORWARD: year-by-year OOS breakdown on 2020-2025 trades. The
strategy has no fitted parameters so year-over-year consistency is
the robustness check.

OUTPUTS:
  compass/reports/exp2240_qqq_iwm_credit_spreads.{json,html}

Run::
    python3 -m compass.exp2240_qqq_iwm_credit_spreads
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2240_qqq_iwm_credit_spreads.json"
REPORT_HTML = REPORT_DIR / "exp2240_qqq_iwm_credit_spreads.html"

TRADING_DAYS = 252
CAPITAL = 100_000.0


# ═══════════════════════════════════════════════════════════════════════════
# Parametrized EXP-1220 helpers (ticker-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

def _find_exps(hd, ticker: str, start: str, end: str) -> List[str]:
    conn = sqlite3.connect(hd._db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT expiration FROM option_contracts "
            "WHERE ticker=? AND option_type='P' AND expiration BETWEEN ? AND ? "
            "ORDER BY expiration",
            (ticker, start, end),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def _next_td(dt: datetime, td_set: set) -> Optional[datetime]:
    for off in range(7):
        c = dt + timedelta(days=off)
        if c.strftime("%Y-%m-%d") in td_set:
            return c
    return None


def _sell_put_spread(hd, ticker: str, exp: str, trade_date: str,
                      price: float, otm_pct: float, width: float) -> Optional[Dict]:
    strikes = hd.get_available_strikes(ticker, exp, trade_date, "P")
    if not strikes:
        return None
    target = price * otm_pct
    # Try 12 candidates closest to target
    for sk in sorted(strikes, key=lambda k: abs(k - target))[:12]:
        lk = sk - width
        if lk not in strikes:
            cands = [s for s in strikes if s < sk and abs(s - lk) <= 1.0]
            if not cands:
                continue
            lk = max(cands)
        if sk - lk <= 0:
            continue
        pp = hd.get_spread_prices(ticker, datetime.strptime(exp, "%Y-%m-%d"),
                                     sk, lk, "P", trade_date)
        if pp is None:
            continue
        credit = pp["short_close"] - pp["long_close"]
        if credit > 0.05:
            return {"short": sk, "long": lk, "credit": round(credit, 4),
                    "width": sk - lk, "max_loss": round(sk - lk - credit, 4)}
    return None


def _walk_spread(hd, ticker: str, exp: str, short_k: float, long_k: float,
                   entry_credit: float, entry_dt: datetime, exp_dt_obj: datetime,
                   td_index: pd.DatetimeIndex,
                   profit_pct: float = 0.50, stop_mult: float = 2.0,
                   min_dte: int = 7) -> Tuple[str, str, float, int]:
    td_set = set(td_index.strftime("%Y-%m-%d"))
    hold = 0
    current = entry_dt + timedelta(days=1)
    while current <= exp_dt_obj:
        cs = current.strftime("%Y-%m-%d")
        if cs not in td_set:
            current += timedelta(days=1)
            continue
        hold += 1
        pp = hd.get_spread_prices(ticker, exp_dt_obj, short_k, long_k, "P", cs)
        if pp is None:
            current += timedelta(days=1)
            continue
        cv = pp["short_close"] - pp["long_close"]
        if cv <= entry_credit * (1 - profit_pct):
            return cs, "profit", cv, hold
        if cv - entry_credit > entry_credit * stop_mult:
            return cs, "stop", cv, hold
        if (exp_dt_obj - current).days <= min_dte:
            return cs, "dte_exit", cv, hold
        current += timedelta(days=1)
    fp = hd.get_spread_prices(ticker, exp_dt_obj, short_k, long_k, "P", exp)
    ev = (fp["short_close"] - fp["long_close"]) if fp else 0.0
    return exp, "expiration", ev, hold


def run_credit_spread_trades(
    hd,
    ticker: str,
    underlying_df: pd.DataFrame,
    vix: pd.Series,
    *,
    width: float,
    otm_pct: float = 0.95,
    dte_target: int = 28,
    min_spacing: int = 10,
    risk_pct: float = 0.03,
    max_contracts: int = 4,
    vix_block: float = 40.0,
    start: str = "2020-03-01",
    end: str = "2025-12-31",
) -> List[Dict]:
    """Full EXP-1220-style credit spread loop for arbitrary ticker."""
    close = underlying_df["Close"]
    td_set = set(underlying_df.index.strftime("%Y-%m-%d"))
    exps = _find_exps(hd, ticker, start, end)
    trades: List[Dict] = []
    last_entry: Optional[datetime] = None

    for exp in exps:
        try:
            exp_obj = datetime.strptime(exp, "%Y-%m-%d")
        except ValueError:
            continue
        entry_dt = _next_td(exp_obj - timedelta(days=dte_target), td_set)
        if entry_dt is None:
            continue
        es = entry_dt.strftime("%Y-%m-%d")
        if last_entry is not None and (entry_dt - last_entry).days < min_spacing:
            continue
        try:
            price = float(close.loc[es])
            v = float(vix.loc[es])
        except Exception:
            continue
        if np.isnan(price) or np.isnan(v) or v > vix_block:
            continue

        spread = _sell_put_spread(hd, ticker, exp, es, price,
                                    otm_pct=otm_pct, width=width)
        if spread is None:
            continue
        contracts = max(1, min(max_contracts,
                                int(CAPITAL * risk_pct / (spread["max_loss"] * 100))))
        ed, er, ev, hold = _walk_spread(
            hd, ticker, exp, spread["short"], spread["long"],
            spread["credit"], entry_dt, exp_obj, underlying_df.index,
        )
        pnl = (spread["credit"] - ev) * 100 * contracts
        trades.append({
            "ticker": ticker,
            "entry_date": es,
            "exit_date": ed,
            "pnl": round(pnl, 2),
            "exit_reason": er,
            "credit": spread["credit"],
            "short_strike": spread["short"],
            "long_strike": spread["long"],
            "width": spread["width"],
            "vix": round(v, 1),
            "hold_days": hold,
            "contracts": contracts,
        })
        last_entry = entry_dt

    return trades


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def per_trade_metrics(trades: List[Dict], label: str) -> Dict:
    if not trades:
        return {"label": label, "n_trades": 0, "total_pnl": 0.0,
                "win_rate": 0.0, "sharpe": 0.0, "cagr_pct": 0.0,
                "max_dd_pct": 0.0, "avg_pnl": 0.0, "sortino": 0.0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    wins = int((pnl > 0).sum())
    equity = CAPITAL + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    first = datetime.strptime(trades[0]["entry_date"], "%Y-%m-%d")
    last = datetime.strptime(trades[-1]["exit_date"], "%Y-%m-%d")
    yrs = max(1.0, (last - first).days / 365.25)
    trades_per_yr = len(pnl) / yrs
    rets = pnl / CAPITAL
    mu = float(rets.mean())
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mu / sd) * math.sqrt(trades_per_yr) if sd > 1e-12 else 0.0
    down = rets[rets < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else 0.0
    sortino = (mu / ds) * math.sqrt(trades_per_yr) if ds > 1e-12 else 0.0
    cagr_pct = float((equity[-1] / CAPITAL) ** (1.0 / yrs) * 100 - 100)
    return {
        "label": label,
        "n_trades": int(len(pnl)),
        "total_pnl": round(float(pnl.sum()), 2),
        "win_rate": round(wins / len(pnl), 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr_pct, 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "avg_pnl": round(float(pnl.mean()), 2),
        "trades_per_yr": round(trades_per_yr, 2),
    }


def yearly_breakdown(trades: List[Dict]) -> List[Dict]:
    by_yr: Dict[int, List[float]] = defaultdict(list)
    for t in trades:
        by_yr[int(t["entry_date"][:4])].append(float(t["pnl"]))
    out = []
    for yr in sorted(by_yr.keys()):
        pnls = np.array(by_yr[yr])
        if len(pnls) < 2:
            continue
        mu = float(pnls.mean())
        sd = float(pnls.std(ddof=1))
        sh_trade = (mu / sd) * math.sqrt(12) if sd > 1e-12 else 0.0  # ~monthly
        out.append({
            "year": yr,
            "n_trades": int(len(pnls)),
            "pnl": round(float(pnls.sum()), 0),
            "return_pct": round(float(pnls.sum()) / CAPITAL * 100, 3),
            "win_rate": round(float((pnls > 0).mean()), 4),
            "sharpe_approx": round(sh_trade, 3),
        })
    return out


def daily_return_stream(trades: List[Dict], idx: pd.DatetimeIndex) -> pd.Series:
    """Exit-date-keyed daily returns for correlation analysis."""
    s = pd.Series(0.0, index=idx)
    for t in trades:
        d = pd.Timestamp(t["exit_date"])
        if d in s.index:
            s.loc[d] += t["pnl"] / CAPITAL
    return s


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def render_html(payload: Dict) -> str:
    def tk_row(r: Dict) -> str:
        m = r.get("metrics", {})
        if not m or m.get("n_trades", 0) == 0:
            return f"""<tr>
                <td><strong>{r['ticker']}</strong></td>
                <td colspan="8"><em>{r.get('status','no data')}</em></td>
            </tr>"""
        return f"""<tr>
            <td><strong>{r['ticker']}</strong></td>
            <td>{m['n_trades']}</td>
            <td>{m['win_rate']*100:.1f}%</td>
            <td>${m['total_pnl']:,.0f}</td>
            <td>{m['cagr_pct']:.2f}%</td>
            <td>{m['sharpe']:.2f}</td>
            <td>{m['sortino']:.2f}</td>
            <td>{m['max_dd_pct']:.2f}%</td>
            <td>{m['trades_per_yr']:.1f}</td>
        </tr>"""

    tk_rows = "".join(tk_row(r) for r in payload["tickers"])

    corr = payload.get("correlation_matrix", {})
    corr_tickers = list(corr.keys()) if corr else []
    corr_header = "<th>vs</th>" + "".join(f"<th>{t}</th>" for t in corr_tickers)
    corr_rows = ""
    for a in corr_tickers:
        cells = f"<td><strong>{a}</strong></td>"
        for b in corr_tickers:
            v = corr.get(a, {}).get(b)
            if v is None:
                cells += "<td>—</td>"
            elif a == b:
                cells += "<td>1.00</td>"
            else:
                cls = "low" if abs(v) < 0.2 else "med" if abs(v) < 0.5 else "high"
                cells += f'<td class="{cls}">{v:+.2f}</td>'
        corr_rows += f"<tr>{cells}</tr>"

    yearly_sections = ""
    for r in payload["tickers"]:
        if not r.get("yearly"):
            continue
        yr_rows = ""
        for y in r["yearly"]:
            yr_rows += f"""<tr>
                <td>{y['year']}</td>
                <td>{y['n_trades']}</td>
                <td>${y['pnl']:,.0f}</td>
                <td>{y['return_pct']:.2f}%</td>
                <td>{y['win_rate']*100:.1f}%</td>
                <td>{y['sharpe_approx']:.2f}</td>
            </tr>"""
        yearly_sections += f"""
            <h3>{r['ticker']} — Year-by-Year</h3>
            <table>
                <thead><tr><th>Year</th><th>Trades</th><th>P&amp;L</th>
                <th>Return</th><th>Win %</th><th>Sharpe approx</th></tr></thead>
                <tbody>{yr_rows}</tbody>
            </table>
        """

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2240 QQQ/IWM Credit Spreads</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1100px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.2em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  h3 {{ color:#475569; margin-top:1.4em; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.78em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  td.low {{ background:#dcfce7; font-weight:600; }}
  td.med {{ background:#fef9c3; }}
  td.high {{ background:#fee2e2; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
</style></head><body>

<h1>EXP-2240 — QQQ and IWM Credit Spreads</h1>
<div class="subtitle">Same EXP-1220 framework on QQQ/IWM | {payload['timestamp']}</div>

<div class="note">
    <strong>Parameters:</strong> 28-day target DTE, 5%-OTM short put,
    $5-wide spread (QQQ)/$2 (IWM if applicable), 50% profit target,
    2× stop, VIX&lt;40 entry, 3% risk per trade. Same parameters as
    EXP-1220. Real IronVault option_daily prices, Yahoo SPY/QQQ/IWM
    closes, Yahoo ^VIX. No synthetic data.
</div>

<h2>Per-ticker OOS metrics</h2>
<table>
    <thead><tr><th>Ticker</th><th>Trades</th><th>Win %</th><th>P&amp;L</th>
    <th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Trades/yr</th></tr></thead>
    <tbody>{tk_rows}</tbody>
</table>

<h2>Correlation Matrix (exit-date daily returns)</h2>
<table>
    <thead><tr>{corr_header}</tr></thead>
    <tbody>{corr_rows}</tbody>
</table>

{yearly_sections}

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2240 — compass/exp2240_qqq_iwm_credit_spreads.py · Real IronVault option_daily
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2240 — QQQ / IWM Credit Spreads")
    print("=" * 72)

    import yfinance as yf
    from shared.iron_vault import IronVault
    hd = IronVault.instance()

    # ── Data availability check ─────────────────────────────────────────
    print("\n[1/5] IronVault data availability check...")
    conn = sqlite3.connect(hd._db_path)
    availability: Dict[str, Dict] = {}
    for tk in ("SPY", "QQQ", "IWM", "XLF", "XLI"):
        n = conn.execute(
            "SELECT COUNT(*) FROM option_contracts WHERE ticker=?", (tk,)
        ).fetchone()[0]
        mn, mx = conn.execute(
            "SELECT MIN(od.date), MAX(od.date) "
            "FROM option_daily od JOIN option_contracts oc "
            "  ON od.contract_symbol=oc.contract_symbol WHERE oc.ticker=?",
            (tk,),
        ).fetchone()
        availability[tk] = {"contracts": int(n), "date_min": mn, "date_max": mx}
        print(f"  {tk}: {n:6d} contracts  {mn} → {mx}")
    conn.close()

    # ── Price / VIX ─────────────────────────────────────────────────────
    print("\n[2/5] Loading underlyings and VIX (Yahoo, real)...")
    start, end = "2019-06-01", "2026-01-01"
    underlyings: Dict[str, pd.DataFrame] = {}
    for tk in ("SPY", "QQQ", "IWM", "XLF", "XLI"):
        df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).normalize()
        underlyings[tk] = df
        print(f"  {tk}: {len(df)} bars")
    vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).normalize()

    # ── Run strategies ─────────────────────────────────────────────────
    print("\n[3/5] Running credit spread loops on QQQ and IWM (+ SPY/XLF/XLI as benchmarks)...")
    ticker_specs: List[Tuple[str, float]] = [
        ("SPY", 5.0),
        ("QQQ", 5.0),
        ("IWM", 2.0),   # IWM trades at lower $ — narrower spreads default
        ("XLF", 1.0),
        ("XLI", 2.0),
    ]
    results: List[Dict] = []
    trade_streams: Dict[str, List[Dict]] = {}
    for tk, width in ticker_specs:
        print(f"\n  → {tk} (width=${width})")
        if availability[tk]["contracts"] == 0:
            print(f"    SKIP: IronVault has zero {tk} contracts.")
            results.append({
                "ticker": tk,
                "width": width,
                "status": "SKIP: zero IronVault contracts (data gap)",
                "metrics": {},
                "yearly": [],
            })
            continue
        trades = run_credit_spread_trades(
            hd, tk, underlyings[tk], vix, width=width,
        )
        trade_streams[tk] = trades
        m = per_trade_metrics(trades, tk)
        y = yearly_breakdown(trades)
        print(f"    {m['n_trades']:3d} trades  WR={m['win_rate']*100:.1f}%  "
              f"Sh={m['sharpe']:.2f}  CAGR={m['cagr_pct']:.2f}%  "
              f"DD={m['max_dd_pct']:.2f}%  P&L=${m['total_pnl']:,.0f}")
        results.append({
            "ticker": tk,
            "width": width,
            "status": "OK",
            "metrics": m,
            "yearly": y,
            "trades": trades,
        })

    # ── Correlation matrix ─────────────────────────────────────────────
    print("\n[4/5] Computing cross-stream correlation matrix...")
    idx = pd.bdate_range("2020-01-01", "2026-01-01")
    streams = {
        tk: daily_return_stream(trs, idx)
        for tk, trs in trade_streams.items()
    }
    corr_mtx: Dict[str, Dict[str, Optional[float]]] = {}
    for a in streams:
        corr_mtx[a] = {}
        for b in streams:
            if a == b:
                corr_mtx[a][b] = 1.0
                continue
            sa = streams[a]
            sb = streams[b]
            mask = (sa != 0) | (sb != 0)
            if mask.sum() < 20:
                corr_mtx[a][b] = None
                continue
            try:
                c = float(np.corrcoef(sa[mask].values, sb[mask].values)[0, 1])
                corr_mtx[a][b] = None if math.isnan(c) else round(c, 4)
            except Exception:
                corr_mtx[a][b] = None
    for a, row in corr_mtx.items():
        disp = " ".join(f"{b}={row[b] if row[b] is not None else 'NA':>6}" for b in row)
        print(f"    {a}: {disp}")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n[5/5] Verdict — new streams that pass (Sharpe ≥ 1.0, WR ≥ 70%):")
    verdict: Dict[str, str] = {}
    for r in results:
        if r["metrics"] and r["ticker"] in ("QQQ", "IWM"):
            m = r["metrics"]
            passed = (m["sharpe"] >= 1.0 and m["win_rate"] >= 0.70
                       and m["n_trades"] >= 30)
            verdict[r["ticker"]] = (
                f"PASS (Sh {m['sharpe']:.2f}, WR {m['win_rate']*100:.0f}%, n={m['n_trades']})"
                if passed else
                f"REJECTED (Sh {m['sharpe']:.2f}, WR {m['win_rate']*100:.0f}%, n={m['n_trades']})"
            )
        elif r["ticker"] in ("QQQ", "IWM"):
            verdict[r["ticker"]] = r["status"]
    for tk, v in verdict.items():
        print(f"    {tk}: {v}")

    # ── Write report ────────────────────────────────────────────────────
    print("\nWriting reports...")
    payload = {
        "experiment": "EXP-2240",
        "title": "QQQ and IWM Credit Spreads",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "framework": "EXP-1220 put-credit-spread loop (28 DTE, 5% OTM, 50% profit target, 2× stop, VIX<40)",
        "ironvault_availability": availability,
        "tickers": [
            {k: v for k, v in r.items() if k != "trades"} for r in results
        ],
        "correlation_matrix": corr_mtx,
        "verdict": verdict,
        "rule_zero": (
            "All option prices from IronVault data/options_cache.db "
            "(real CBOE/Polygon). Underlying spot + VIX from Yahoo Finance. "
            "No synthetic data, no Black-Scholes, no random fills. If a "
            "ticker has zero IronVault contracts the script reports the "
            "data gap honestly and skips that ticker."
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  → {REPORT_JSON}")
    REPORT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"  → {REPORT_HTML}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ═══════════════════════════════════════════════════════════════════════════
# EXP-2690 — Production signal entry point
# ═══════════════════════════════════════════════════════════════════════════
def generate_today_signals(date):
    """Paper-trading scheduler entry point for the QQQ sleeve.
    Delegates to compass.exp2690_signal_generators.qqq_cs_signals."""
    from compass.exp2690_signal_generators import qqq_cs_signals
    return qqq_cs_signals(date)
