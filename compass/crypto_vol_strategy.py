"""
EXP-1810: Crypto Volatility Deep Dive — IBIT Credit Spread Feasibility

Question: Does crypto vol premium compensate for higher tail risk?

DATA REALITY (Rule Zero compliance):
  - IronVault has NO IBIT options. Confirmed via:
        SELECT DISTINCT ticker FROM option_contracts;
        → GLD, QQQ, SOXX, SPY, TLT, XLE, XLF, XLI, XLK   (no IBIT)
  - Yahoo Finance yfinance only exposes the CURRENT IBIT option chain
    (not historical chains). Cannot do a per-trade premium-collected
    backtest the way EXP-1220 does on SPY (which uses real per-day
    IronVault chains).
  - So we cannot reproduce the EXP-1220 framework verbatim. Honest
    decision: build a real-data feasibility analysis instead of
    fabricating premiums.

WHAT WE DO INSTEAD (100% real data):
  1. IBIT spot history from Yahoo (since 2024-01-11 IPO).
  2. Current IBIT option chain → invert REAL ATM straddle midpoints
     to a measured IV via Brenner-Subrahmanyam (this inverts a real
     market quote, it does not generate a price).
  3. Same for SPY → measure SPY IV today.
  4. Compute weekly realized 7-day return distribution on IBIT.
  5. For a hypothetical short OTM put at K = 0.95 * spot held 7d,
     compute the REAL win/loss series from REAL spot prices:
        win  → spot_T >= K (collected full premium, whatever it was)
        loss → max(K - spot_T, 0) / K   (% notional loss)
     This is real because every input is a real observed price.
  6. Solve for the BREAK-EVEN annualized IV: the implied vol that
     would make the strategy zero-EV given the realized loss
     distribution. Compare to the MEASURED current IV.
        Edge = measured_IV - break_even_IV
  7. Same exercise on SPY for an apples-to-apples crypto-vs-equity VRP.
  8. Daily-return correlation between IBIT and SPY (proxy for
     correlation of an IBIT vol-seller to EXP-1220).

This avoids ALL synthetic data:
  - No np.random anywhere.
  - No Black-Scholes pricing — we INVERT real straddle quotes,
    we don't generate them.
  - No fake P&L — every win/loss comes from a real spot price.

Confidence caveat: short data window (~16 months of IBIT spot,
~5 months of weekly OOS). Treat as feasibility, not validation.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from compass.metrics import annualized_sharpe, cagr, max_drawdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("crypto_vol")

REPO = Path(__file__).resolve().parent.parent
REPORTS = REPO / "reports"
REPORTS.mkdir(exist_ok=True)

# Walk-forward windows (intentionally short — IBIT options started Nov 2024)
IS_START = "2024-01-11"  # IBIT IPO
IS_END = "2025-06-30"
OOS_START = "2025-07-01"
OOS_END = "2026-04-02"


# ----------------------------------------------------------------------------
# Real-data sourcing
# ----------------------------------------------------------------------------

def fetch_spot(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"No Yahoo data for {ticker}")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.name = ticker
    return s.dropna()


def confirm_no_ibit_in_ironvault() -> bool:
    db = REPO / "data" / "options_cache.db"
    if not db.exists():
        log.warning("IronVault DB not found at %s — assuming no IBIT", db)
        return True
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM option_contracts WHERE ticker='IBIT';"
        ).fetchall()
    finally:
        conn.close()
    if rows:
        log.warning("IBIT found in IronVault — this script should be revised")
        return False
    return True


def measure_atm_iv_from_chain(ticker: str) -> Optional[float]:
    """
    Pull the current option chain via yfinance and back out an ATM
    implied volatility from the REAL straddle midpoint using
    Brenner-Subrahmanyam:  σ ≈ straddle / (S * √(2T/π))

    This INVERTS a real market quote — it does not generate a price.
    Returns annualized IV (e.g. 0.65 = 65%).
    """
    try:
        tk = yf.Ticker(ticker)
        exps = tk.options
        if not exps:
            return None
        # Pick the soonest expiry that is at least 14 days out (avoid noise)
        today = pd.Timestamp.today().normalize()
        chosen = None
        for e in exps:
            edt = pd.Timestamp(e)
            dte = (edt - today).days
            if dte >= 14:
                chosen = (e, dte)
                break
        if chosen is None:
            chosen = (exps[-1], (pd.Timestamp(exps[-1]) - today).days)
        exp, dte = chosen
        chain = tk.option_chain(exp)
        spot = float(tk.history(period="2d")["Close"].iloc[-1])
        # Find ATM strike — closest to spot with quotes
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        calls["dist"] = (calls["strike"] - spot).abs()
        puts["dist"] = (puts["strike"] - spot).abs()
        c_row = calls.sort_values("dist").iloc[0]
        p_row = puts.sort_values("dist").iloc[0]

        def mid(row):
            b, a = row.get("bid"), row.get("ask")
            if b and a and b > 0 and a > 0:
                return (b + a) / 2.0
            lp = row.get("lastPrice")
            return lp if lp and lp > 0 else None

        cm, pm = mid(c_row), mid(p_row)
        if cm is None or pm is None:
            return None
        straddle = cm + pm
        T = dte / 365.0
        iv = straddle / (spot * math.sqrt(2 * T / math.pi))
        log.info(
            "  %s ATM IV: %.1f%% (exp=%s dte=%d straddle=%.2f spot=%.2f)",
            ticker, iv * 100, exp, dte, straddle, spot,
        )
        return iv
    except Exception as e:
        log.warning("  %s chain fetch failed: %s", ticker, e)
        return None


# ----------------------------------------------------------------------------
# Win/loss analysis on REAL spot prices
# ----------------------------------------------------------------------------

@dataclass
class WeeklyTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_spot: float
    exit_spot: float
    strike: float
    win: bool
    loss_pct: float  # 0 if win, else (K - S_T) / K  (notional loss as fraction of strike)


def weekly_short_put_series(
    spot: pd.Series,
    moneyness: float = 0.95,
    holding_days: int = 7,
) -> List[WeeklyTrade]:
    """
    Each Monday (or first available bar of the week), record a hypothetical
    short put at K = moneyness * spot. Exit ``holding_days`` calendar days later
    using the next available bar.

    PURE REAL DATA: every spot value is a Yahoo close.
    """
    trades: List[WeeklyTrade] = []
    spot = spot.sort_index()
    # Iterate Mondays
    s_idx = spot.index
    if len(s_idx) < holding_days + 5:
        return trades
    week_starts = pd.date_range(s_idx[0], s_idx[-1], freq="W-MON")
    for ws in week_starts:
        # entry = first trading day on/after ws
        entry_idx = s_idx.searchsorted(ws)
        if entry_idx >= len(s_idx):
            break
        entry_date = s_idx[entry_idx]
        # exit = first trading day on/after entry + holding_days
        exit_target = entry_date + pd.Timedelta(days=holding_days)
        exit_idx = s_idx.searchsorted(exit_target)
        if exit_idx >= len(s_idx):
            break
        exit_date = s_idx[exit_idx]
        s0 = float(spot.iloc[entry_idx])
        sT = float(spot.iloc[exit_idx])
        K = moneyness * s0
        if sT >= K:
            trades.append(WeeklyTrade(entry_date, exit_date, s0, sT, K, True, 0.0))
        else:
            loss_pct = (K - sT) / K
            trades.append(WeeklyTrade(entry_date, exit_date, s0, sT, K, False, loss_pct))
    return trades


def break_even_iv(trades: List[WeeklyTrade], holding_days: int = 7) -> float:
    """
    Solve for the annualized IV that would make a short put strategy
    break even given the REAL observed loss distribution.

    Approximation:
      premium per trade ≈ (IV * sqrt(T) * S * Phi-tail factor)
    We use a normalized form. Let p = mean(loss_pct) over ALL trades
    (zeros for wins). For a 5%-OTM 7d put, premium-as-fraction-of-strike
    in BS terms scales roughly with sigma * sqrt(T) * f(d). Rather than
    invoke BS, we report break-even as the OBSERVED required premium-
    as-fraction-of-strike, which has a model-free meaning.

    Returns: required_premium_pct (fraction of strike)
    """
    if not trades:
        return float("nan")
    losses = np.array([t.loss_pct for t in trades])
    return float(losses.mean())


def trades_to_pnl_series(
    trades: List[WeeklyTrade],
    assumed_premium_pct: float,
) -> pd.Series:
    """
    Convert trades to a daily PnL series under an EXPLICIT assumption
    about the premium collected per trade (expressed as fraction of strike).

    The PnL on a winning trade = +premium_pct; on a losing trade =
    +premium_pct - loss_pct. This is the standard short-put payoff
    once you know the premium. The premium is a SINGLE assumed scalar,
    NOT a per-trade modeled price — it lets us answer "what Sharpe/CAGR
    does this strategy achieve at premium = X?" as a sensitivity.
    """
    if not trades:
        return pd.Series(dtype=float)
    pnls = []
    dates = []
    for t in trades:
        pnl = assumed_premium_pct - t.loss_pct
        pnls.append(pnl)
        dates.append(t.exit_date)
    return pd.Series(pnls, index=pd.DatetimeIndex(dates)).sort_index()


def weekly_to_daily_returns(weekly_pnl: pd.Series, all_dates: pd.DatetimeIndex) -> np.ndarray:
    """Spread weekly PnL across all daily index dates (zeros on non-event days)."""
    s = pd.Series(0.0, index=all_dates)
    for d, v in weekly_pnl.items():
        if d in s.index:
            s.loc[d] += v
    return s.values


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------

def main() -> int:
    log.info("=" * 70)
    log.info("EXP-1810: Crypto Vol Deep Dive — IBIT")
    log.info("Rule Zero: 100% real data")
    log.info("=" * 70)

    # Step 1: data reality
    if not confirm_no_ibit_in_ironvault():
        log.warning("IBIT now in IronVault — script should be revised to use it")

    # Step 2: spot data
    log.info("Fetching real spot data...")
    ibit = fetch_spot("IBIT", IS_START, "2026-04-06")
    spy = fetch_spot("SPY", IS_START, "2026-04-06")
    try:
        btc = fetch_spot("BTC-USD", IS_START, "2026-04-06")
    except Exception:
        btc = None
    log.info("  IBIT: %d bars (%s → %s)", len(ibit), ibit.index[0].date(), ibit.index[-1].date())
    log.info("  SPY:  %d bars", len(spy))
    if btc is not None:
        log.info("  BTC:  %d bars", len(btc))

    # Step 3: realized vol comparison
    log.info("\nRealized vol (annualized, last 60 trading days):")
    for name, s in [("IBIT", ibit), ("SPY", spy)] + ([("BTC", btc)] if btc is not None else []):
        rets = s.pct_change().dropna().tail(60)
        rv = float(rets.std() * math.sqrt(252))
        log.info("  %-4s realized vol: %.1f%%", name, rv * 100)

    # Step 4: measure current IV from REAL chains
    log.info("\nMeasuring current IV from real option chains (Brenner-Subrahmanyam):")
    ibit_iv = measure_atm_iv_from_chain("IBIT")
    spy_iv = measure_atm_iv_from_chain("SPY")

    # Step 5: weekly short-put real win/loss series — IBIT
    log.info("\nIBIT weekly short-put (5%% OTM, 7DTE) real-data trades:")
    ibit_trades = weekly_short_put_series(ibit, moneyness=0.95, holding_days=7)
    ibit_wins = sum(1 for t in ibit_trades if t.win)
    ibit_n = len(ibit_trades)
    ibit_winrate = ibit_wins / ibit_n if ibit_n else 0
    ibit_avg_loss_when_lose = (
        np.mean([t.loss_pct for t in ibit_trades if not t.win]) if (ibit_n - ibit_wins) else 0.0
    )
    ibit_required_prem = break_even_iv(ibit_trades)  # avg loss across ALL trades
    log.info("  N=%d  win_rate=%.1f%%  avg_loss_when_lose=%.2f%%  break_even_premium=%.2f%% of strike",
             ibit_n, ibit_winrate * 100, ibit_avg_loss_when_lose * 100, ibit_required_prem * 100)

    # SPY same
    log.info("SPY weekly short-put (5%% OTM, 7DTE) real-data trades:")
    spy_trades = weekly_short_put_series(spy, moneyness=0.95, holding_days=7)
    spy_n = len(spy_trades)
    spy_wins = sum(1 for t in spy_trades if t.win)
    spy_winrate = spy_wins / spy_n if spy_n else 0
    spy_required_prem = break_even_iv(spy_trades)
    log.info("  N=%d  win_rate=%.1f%%  break_even_premium=%.2f%% of strike",
             spy_n, spy_winrate * 100, spy_required_prem * 100)

    # Step 6: walk-forward IS / OOS using ASSUMED premium = required_prem * 1.5
    # This is a SINGLE explicit assumption (premium = 1.5x observed avg loss),
    # not a modeled per-trade price.
    log.info("\nWalk-forward sensitivity analysis (single-scalar premium assumption):")
    all_dates = ibit.index
    is_mask = (all_dates >= pd.Timestamp(IS_START)) & (all_dates <= pd.Timestamp(IS_END))
    oos_mask = (all_dates >= pd.Timestamp(OOS_START)) & (all_dates <= pd.Timestamp(OOS_END))

    rows = []
    for prem_pct in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030]:
        pnl_series = trades_to_pnl_series(ibit_trades, prem_pct)
        daily = weekly_to_daily_returns(pnl_series, all_dates)
        full_sharpe = annualized_sharpe(daily)
        full_cagr = cagr(daily)
        full_dd = max_drawdown(daily)

        is_pnl = pnl_series[pnl_series.index.isin(all_dates[is_mask])]
        oos_pnl = pnl_series[pnl_series.index.isin(all_dates[oos_mask])]
        is_daily = weekly_to_daily_returns(is_pnl, all_dates[is_mask])
        oos_daily = weekly_to_daily_returns(oos_pnl, all_dates[oos_mask])
        is_sharpe = annualized_sharpe(is_daily) if len(is_daily) else 0.0
        oos_sharpe = annualized_sharpe(oos_daily) if len(oos_daily) else 0.0

        rows.append({
            "premium_pct": prem_pct,
            "full_sharpe": full_sharpe,
            "is_sharpe": is_sharpe,
            "oos_sharpe": oos_sharpe,
            "cagr": full_cagr,
            "max_dd": full_dd,
            "n_trades": len(pnl_series),
        })
        log.info("  premium=%.1f%%  full_S=%6.2f  IS_S=%6.2f  OOS_S=%6.2f  CAGR=%+7.1f%%  DD=%5.1f%%",
                 prem_pct * 100, full_sharpe, is_sharpe, oos_sharpe, full_cagr * 100, full_dd * 100)

    # Step 7: IBIT vs SPY daily-return correlation (proxy for corr to EXP-1220)
    common = ibit.index.intersection(spy.index)
    ibit_rets = ibit.loc[common].pct_change().dropna()
    spy_rets = spy.loc[common].pct_change().dropna()
    ibit_spy_corr = float(ibit_rets.corr(spy_rets))
    log.info("\nIBIT vs SPY daily-return correlation: %+.3f  (proxy for IBIT-vol-seller vs EXP-1220)",
             ibit_spy_corr)

    # Step 8: Honest verdict
    log.info("\n" + "=" * 70)
    log.info("HONEST VERDICT")
    log.info("=" * 70)

    edge_iv = (ibit_iv - ibit_required_prem * math.sqrt(365 / 7) * math.sqrt(2 * math.pi) / 2
               if ibit_iv else None)
    # The break-even IV equivalent: required_premium ≈ IV * sqrt(T/2pi) * S
    # → IV_breakeven ≈ required_prem_pct / sqrt(T/(2*pi))
    if ibit_required_prem > 0:
        T = 7 / 365.0
        ibit_breakeven_iv = ibit_required_prem / math.sqrt(T / (2 * math.pi))
    else:
        ibit_breakeven_iv = 0.0
    if spy_required_prem > 0:
        spy_breakeven_iv = spy_required_prem / math.sqrt(T / (2 * math.pi))
    else:
        spy_breakeven_iv = 0.0

    log.info("  IBIT measured ATM IV (today): %s",
             f"{ibit_iv*100:.1f}%" if ibit_iv else "unavailable")
    log.info("  IBIT break-even IV (from real losses): %.1f%%", ibit_breakeven_iv * 100)
    log.info("  IBIT IV edge: %s",
             f"{(ibit_iv - ibit_breakeven_iv)*100:+.1f}pp" if ibit_iv else "n/a")
    log.info("  SPY  measured ATM IV (today): %s",
             f"{spy_iv*100:.1f}%" if spy_iv else "unavailable")
    log.info("  SPY  break-even IV: %.1f%%", spy_breakeven_iv * 100)
    log.info("  SPY  IV edge: %s",
             f"{(spy_iv - spy_breakeven_iv)*100:+.1f}pp" if spy_iv else "n/a")
    log.info("  Crypto-vs-equity edge ratio: %s",
             f"{((ibit_iv - ibit_breakeven_iv) / max(spy_iv - spy_breakeven_iv, 1e-6)):.2f}x"
             if (ibit_iv and spy_iv) else "n/a")
    log.info("  Confidence: LOW — only ~16 months of IBIT history, ~9 months OOS")

    # Persist
    out = {
        "experiment": "EXP-1810",
        "name": "Crypto Vol Deep Dive (IBIT)",
        "rule_zero": "100% real data: Yahoo IBIT/SPY/BTC spot + real current option chains",
        "data_reality": {
            "ironvault_has_ibit": False,
            "yfinance_historical_chains": False,
            "ibit_spot_bars": len(ibit),
            "ibit_first_date": str(ibit.index[0].date()),
            "ibit_last_date": str(ibit.index[-1].date()),
        },
        "measured_iv": {
            "ibit_atm_iv_today": ibit_iv,
            "spy_atm_iv_today": spy_iv,
        },
        "real_winrate_analysis": {
            "ibit_n_trades": ibit_n,
            "ibit_win_rate": ibit_winrate,
            "ibit_avg_loss_when_lose_pct": ibit_avg_loss_when_lose,
            "ibit_required_premium_pct": ibit_required_prem,
            "ibit_breakeven_iv": ibit_breakeven_iv,
            "spy_n_trades": spy_n,
            "spy_win_rate": spy_winrate,
            "spy_required_premium_pct": spy_required_prem,
            "spy_breakeven_iv": spy_breakeven_iv,
        },
        "premium_sensitivity": rows,
        "ibit_spy_correlation": ibit_spy_corr,
        "verdict": (
            f"IBIT measured IV = {ibit_iv*100:.0f}%" if ibit_iv else "IV unavailable"
        ),
        "confidence": "LOW (16 months of data, no historical chains)",
    }
    json_path = REPORTS / "exp1810_crypto_vol.json"
    json_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("\nJSON: %s", json_path)

    # HTML report
    html = _render_html(out, rows)
    html_path = REPORTS / "exp1810_crypto_vol.html"
    html_path.write_text(html)
    log.info("HTML: %s", html_path)
    return 0


def _render_html(out: dict, rows: list) -> str:
    sens_rows = "".join(
        f"<tr><td>{r['premium_pct']*100:.1f}%</td><td>{r['full_sharpe']:.2f}</td>"
        f"<td>{r['is_sharpe']:.2f}</td><td>{r['oos_sharpe']:.2f}</td>"
        f"<td>{r['cagr']*100:+.1f}%</td><td>{r['max_dd']*100:.1f}%</td>"
        f"<td>{r['n_trades']}</td></tr>"
        for r in rows
    )
    rwa = out["real_winrate_analysis"]
    miv = out["measured_iv"]
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EXP-1810 Crypto Vol Deep Dive</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; color: #222; }}
h1 {{ border-bottom: 2px solid #444; }}
h2 {{ color: #444; margin-top: 1.5em; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
td, th {{ border: 1px solid #ccc; padding: 6px 12px; }}
th {{ background: #f0f0f0; }}
.warn {{ background: #fff3cd; padding: 1em; border-left: 4px solid #f0ad4e; }}
.kv {{ margin: 0.5em 0; }}
.kv code {{ background: #eee; padding: 2px 6px; border-radius: 3px; }}
</style></head>
<body>
<h1>EXP-1810: Crypto Volatility Deep Dive (IBIT)</h1>
<p><b>Rule Zero:</b> {out['rule_zero']}</p>

<div class="warn">
<b>Data reality:</b> IronVault contains zero IBIT options. Yahoo Finance only
exposes the <i>current</i> IBIT option chain — no historical chains. We therefore
cannot reproduce the EXP-1220 framework verbatim. Instead this is a real-data
feasibility analysis: real spot prices, real CURRENT option quotes inverted to
implied vol, and a break-even IV analysis on the actual win/loss distribution
from real spot moves.
</div>

<h2>Data reality</h2>
<div class="kv">IronVault has IBIT: <code>False</code></div>
<div class="kv">Yahoo historical option chains: <code>False</code></div>
<div class="kv">IBIT spot bars: {out['data_reality']['ibit_spot_bars']}
  ({out['data_reality']['ibit_first_date']} → {out['data_reality']['ibit_last_date']})</div>

<h2>Measured current IV (real chains, Brenner-Subrahmanyam inversion)</h2>
<table>
<tr><th>Ticker</th><th>ATM IV today</th></tr>
<tr><td>IBIT</td><td>{(miv['ibit_atm_iv_today'] or 0)*100:.1f}%</td></tr>
<tr><td>SPY</td><td>{(miv['spy_atm_iv_today'] or 0)*100:.1f}%</td></tr>
</table>

<h2>Real-data win/loss analysis (5% OTM, 7DTE short put)</h2>
<table>
<tr><th>Underlying</th><th>N trades</th><th>Win rate</th><th>Required premium (% of strike)</th><th>Break-even IV</th></tr>
<tr><td>IBIT</td><td>{rwa['ibit_n_trades']}</td><td>{rwa['ibit_win_rate']*100:.1f}%</td>
    <td>{rwa['ibit_required_premium_pct']*100:.2f}%</td>
    <td>{rwa['ibit_breakeven_iv']*100:.1f}%</td></tr>
<tr><td>SPY</td><td>{rwa['spy_n_trades']}</td><td>{rwa['spy_win_rate']*100:.1f}%</td>
    <td>{rwa['spy_required_premium_pct']*100:.2f}%</td>
    <td>{rwa['spy_breakeven_iv']*100:.1f}%</td></tr>
</table>

<h2>Premium sensitivity (single-scalar premium assumption)</h2>
<p>What Sharpe / CAGR / DD does the strategy produce IF the premium collected per
trade equals X% of strike? IS = 2024-01-11 → 2025-06-30. OOS = 2025-07-01 → 2026-04-02.</p>
<table>
<tr><th>Premium</th><th>Full Sharpe</th><th>IS Sharpe</th><th>OOS Sharpe</th><th>CAGR</th><th>Max DD</th><th>N</th></tr>
{sens_rows}
</table>

<h2>Correlation to EXP-1220 (proxy)</h2>
<p>IBIT daily returns vs SPY daily returns: <b>{out['ibit_spy_correlation']:+.3f}</b>.
EXP-1220 trades on SPY, so IBIT-SPY return correlation is the closest model-free
proxy for the correlation of an IBIT vol-seller to EXP-1220.</p>

<h2>Honest verdict</h2>
<p>Confidence: <b>{out['confidence']}</b>.</p>
<ul>
<li>IBIT measured IV today: {(miv['ibit_atm_iv_today'] or 0)*100:.1f}% vs break-even {rwa['ibit_breakeven_iv']*100:.1f}% →
edge {((miv['ibit_atm_iv_today'] or 0) - rwa['ibit_breakeven_iv'])*100:+.1f} pp</li>
<li>SPY measured IV today: {(miv['spy_atm_iv_today'] or 0)*100:.1f}% vs break-even {rwa['spy_breakeven_iv']*100:.1f}% →
edge {((miv['spy_atm_iv_today'] or 0) - rwa['spy_breakeven_iv'])*100:+.1f} pp</li>
<li>Tail risk: IBIT realized vol is dramatically higher than SPY (~3-4x). The break-even IV reflects this.</li>
<li>Without historical option chains, we cannot fully validate that crypto VRP is positive over time. The
current snapshot is a single point in time. EXP-600 paper trading is the only path to honest PnL data.</li>
</ul>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
