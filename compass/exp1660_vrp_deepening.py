"""
EXP-1660 — VRP Deepening: Multi-asset variance risk premium backtest.

Expands VRP research beyond SPY to QQQ, IWM, and an international proxy (EEM,
used in place of EFA which has no public CBOE vol index on FRED/Yahoo).

Methodology (all real data, no synthetics — Rule Zero):
  * Underlying prices: Yahoo Finance daily OHLC for SPY, QQQ, IWM, EEM.
  * Implied vol: CBOE vol indices via FRED
      - VIXCLS   (SPY)   - VXNCLS  (QQQ)
      - RVXCLS   (IWM)   - VXEEMCLS (EEM, EFA proxy)
  * VRP signal   = IV² − RV²_backward_30d  (variance-swap style, Carr/Wu 2009)
  * Backtest P&L = IV²_entry − RV²_forward_30d  (the variance swap payoff)
    scaled by 1/(2·IV_entry) so units are "vol points" per position.

Walk-forward: rolling 12-month train window → 3-month OOS test, stepping
quarterly 2020-2025. Signal threshold selected on the training window
(best-performing percentile), applied OOS only.

Correlation vs EXP-1220: yearly returns are pulled from
experiments/EXP-1220-real/results/summary.json (protected equity curve
returns) and correlated against EXP-1660 per-ticker yearly P&L.

Outputs: compass/reports/exp1660_vrp_deepening.html

Run::
    python3 -m compass.exp1660_vrp_deepening
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT_DIR = os.path.join(os.path.dirname(__file__), "reports")
EXP1220_SUMMARY = os.path.join(
    ROOT, "experiments", "EXP-1220-real", "results", "summary.json"
)

START = "2020-01-01"
END = "2025-12-31"

# Ticker → (Yahoo price symbol, FRED vol-index code, display note)
PAIRS: Dict[str, Tuple[str, str, str]] = {
    "SPY":  ("SPY", "VIXCLS",   "S&P 500 ETF / ^VIX"),
    "QQQ":  ("QQQ", "VXNCLS",   "Nasdaq-100 ETF / ^VXN"),
    "IWM":  ("IWM", "RVXCLS",   "Russell 2000 ETF / ^RVX"),
    "EEM":  ("EEM", "VXEEMCLS", "EM ETF / ^VXEEM  (EFA proxy — no VXEFA series)"),
}

RV_WINDOW = 30          # trading days for realized vol
FWD_WINDOW = 30         # forward variance-swap horizon
TRAIN_DAYS = 252        # 1y train
TEST_DAYS = 63          # ~3m OOS
STEP_DAYS = 63          # quarterly walk-forward


# ── Data loaders ────────────────────────────────────────────────────────


def fetch_yahoo_prices(symbol: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(symbol, start=START, end=END, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError(f"Yahoo empty for {symbol}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = symbol
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna()


def fetch_fred_series(code: str) -> pd.Series:
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={code}&cosd={START}&coed={END}"
    )
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode()
    df = pd.read_csv(io.StringIO(raw))
    date_col = df.columns[0]
    val_col = df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    s = df.set_index(date_col)[val_col].dropna()
    s.name = code
    return s


def load_pair(ticker: str) -> pd.DataFrame:
    price_sym, fred_code, _ = PAIRS[ticker]
    px = fetch_yahoo_prices(price_sym)
    iv = fetch_fred_series(fred_code) / 100.0   # percentage → decimal annualised
    df = pd.concat([px.rename("close"), iv.rename("iv")], axis=1)
    df = df.dropna()
    df["ret"] = np.log(df["close"]).diff()
    return df.dropna()


# ── VRP computations ───────────────────────────────────────────────────


def backward_rv(returns: pd.Series, window: int = RV_WINDOW) -> pd.Series:
    return returns.rolling(window).std() * math.sqrt(252)


def forward_rv(returns: pd.Series, window: int = FWD_WINDOW) -> pd.Series:
    # annualised realised vol over the NEXT `window` trading days
    fwd = returns.shift(-1).rolling(window).std() * math.sqrt(252)
    return fwd.shift(-(window - 1))


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Attach VRP signal and variance-swap forward payoff."""
    out = df.copy()
    out["rv_back"] = backward_rv(out["ret"], RV_WINDOW)
    out["rv_fwd"] = forward_rv(out["ret"], FWD_WINDOW)
    # Carr-Wu variance swap payoff (short vol seller's gain if IV² > RV_fwd²)
    out["vrp_signal"] = out["iv"] ** 2 - out["rv_back"] ** 2      # observable at t
    out["vs_pnl_raw"] = out["iv"] ** 2 - out["rv_fwd"] ** 2       # realised OOS
    # Normalise to vol points: divide by 2·IV → units ≈ vol-points
    out["vs_pnl"] = out["vs_pnl_raw"] / (2.0 * out["iv"].clip(lower=1e-4))
    return out.dropna(subset=["vrp_signal", "vs_pnl"])


# ── Walk-forward backtest ──────────────────────────────────────────────


@dataclass
class FoldResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    threshold: float
    n_signals: int
    n_trades: int
    avg_pnl: float
    win_rate: float
    total_pnl: float


@dataclass
class TickerBacktest:
    ticker: str
    display: str
    fred_code: str
    n_days: int
    folds: List[FoldResult]
    trades_df: pd.DataFrame
    daily_pnl: pd.Series          # daily OOS pnl attributed to entry date
    yearly_pnl: Dict[int, float]
    metrics: Dict[str, float]


def walk_forward(ticker: str, sig: pd.DataFrame) -> TickerBacktest:
    idx = sig.index
    if len(idx) < TRAIN_DAYS + TEST_DAYS:
        raise RuntimeError(f"{ticker}: insufficient data ({len(idx)} days)")

    fold_results: List[FoldResult] = []
    all_trades: List[dict] = []
    oos_pnl = pd.Series(0.0, index=idx)

    start = TRAIN_DAYS
    while start + TEST_DAYS <= len(idx):
        train = sig.iloc[start - TRAIN_DAYS:start]
        test = sig.iloc[start:start + TEST_DAYS]

        # Choose VRP threshold = 60th percentile of positive signals in train
        pos_sig = train["vrp_signal"]
        thresh = float(np.nanpercentile(pos_sig, 60))
        thresh = max(thresh, 0.0)  # only sell vol when IV² > RV²

        # Grid-search best of {50, 60, 70, 80} on training P&L
        best_th, best_pnl = thresh, -np.inf
        for pct in (50, 60, 70, 80):
            th = float(np.nanpercentile(pos_sig, pct))
            th = max(th, 0.0)
            sel = train[train["vrp_signal"] >= th]
            pnl = float(sel["vs_pnl"].mean()) if len(sel) > 0 else -np.inf
            if pnl > best_pnl:
                best_pnl = pnl
                best_th = th

        # Apply OOS
        signals = test[test["vrp_signal"] >= best_th]
        n_sig = len(signals)
        if n_sig > 0:
            pnls = signals["vs_pnl"].values
            wins = int((pnls > 0).sum())
            total = float(pnls.sum())
            avg = float(pnls.mean())
            wr = wins / n_sig
            for dt, row in signals.iterrows():
                all_trades.append({
                    "ticker": ticker,
                    "date": dt,
                    "iv": float(row["iv"]),
                    "rv_back": float(row["rv_back"]),
                    "rv_fwd": float(row["rv_fwd"]),
                    "signal": float(row["vrp_signal"]),
                    "pnl": float(row["vs_pnl"]),
                })
                oos_pnl.loc[dt] = float(row["vs_pnl"])
        else:
            wins = 0
            total = 0.0
            avg = 0.0
            wr = 0.0

        fold_results.append(FoldResult(
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            threshold=best_th,
            n_signals=n_sig,
            n_trades=n_sig,
            avg_pnl=avg,
            win_rate=wr,
            total_pnl=total,
        ))
        start += STEP_DAYS

    trades_df = pd.DataFrame(all_trades)

    # Per-year OOS P&L
    yearly: Dict[int, float] = {}
    if not trades_df.empty:
        trades_df["year"] = pd.to_datetime(trades_df["date"]).dt.year
        for y, grp in trades_df.groupby("year"):
            yearly[int(y)] = float(grp["pnl"].sum())

    # Aggregate metrics
    if not trades_df.empty:
        pnls = trades_df["pnl"].values
        n = len(pnls)
        wr = float((pnls > 0).mean())
        mean_pnl = float(pnls.mean())
        std_pnl = float(pnls.std(ddof=1)) if n > 1 else 0.0
        sharpe = float(mean_pnl / std_pnl * math.sqrt(252 / FWD_WINDOW)) if std_pnl > 0 else 0.0
        total_pnl = float(pnls.sum())
    else:
        n, wr, mean_pnl, sharpe, total_pnl = 0, 0.0, 0.0, 0.0, 0.0

    metrics = {
        "n_trades": n,
        "win_rate": wr,
        "avg_pnl": mean_pnl,
        "total_pnl": total_pnl,
        "sharpe": sharpe,
    }

    return TickerBacktest(
        ticker=ticker,
        display=PAIRS[ticker][2],
        fred_code=PAIRS[ticker][1],
        n_days=len(idx),
        folds=fold_results,
        trades_df=trades_df,
        daily_pnl=oos_pnl,
        yearly_pnl=yearly,
        metrics=metrics,
    )


# ── EXP-1220 correlation ────────────────────────────────────────────────


def load_exp1220_yearly() -> Dict[int, float]:
    """Return EXP-1220 *protected* yearly percent returns."""
    if not os.path.exists(EXP1220_SUMMARY):
        return {}
    with open(EXP1220_SUMMARY) as f:
        data = json.load(f)
    out: Dict[int, float] = {}
    for y, blob in data.get("yearly", {}).items():
        try:
            out[int(y)] = float(blob["protected"]["return_pct"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def yearly_correlation(exp1220: Dict[int, float], exp1660: Dict[int, float]) -> Optional[float]:
    common = sorted(set(exp1220) & set(exp1660))
    if len(common) < 3:
        return None
    a = np.array([exp1220[y] for y in common], dtype=float)
    b = np.array([exp1660[y] for y in common], dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ── Report rendering ───────────────────────────────────────────────────


def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%" if np.isfinite(x) else "—"


def _fmt_num(x: float, dp: int = 3) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(
    results: Dict[str, TickerBacktest],
    exp1220_yearly: Dict[int, float],
    correlations: Dict[str, Optional[float]],
) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1200px;color:#111}
    h1{border-bottom:3px solid #0a3d62}
    h2{margin-top:2em;color:#0a3d62}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#0a3d62;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:2px 8px;border-radius:10px;background:#0a3d62;color:#fff;font-size:11px}
    """

    html = [f"<!doctype html><html><head><meta charset='utf-8'><title>EXP-1660 VRP Deepening</title><style>{css}</style></head><body>"]
    html.append("<h1>EXP-1660 — VRP Deepening</h1>")
    html.append(
        "<p class='muted'>Multi-asset Variance Risk Premium backtest. "
        "Real data only (Yahoo Finance for ETFs, FRED/CBOE for IV indices). "
        "Walk-forward 2020-2025, 12m train / 3m OOS, quarterly step.</p>"
    )
    html.append("<p><span class='pill'>Rule Zero ✓ no synthetic data</span></p>")

    # Summary per-ticker
    html.append("<h2>Per-ticker OOS results</h2>")
    html.append("<table><tr><th>Ticker</th><th>Pair</th><th>IV source</th>"
                "<th>Days</th><th>Trades</th><th>Win rate</th>"
                "<th>Avg P&L (vol-pts)</th><th>Total P&L (vol-pts)</th><th>Sharpe</th>"
                "<th>Corr vs EXP-1220 (yearly)</th></tr>")
    for tk, bt in results.items():
        m = bt.metrics
        corr = correlations.get(tk)
        corr_str = f"{corr:+.2f}" if corr is not None else "n/a"
        html.append(
            f"<tr><td class='l'><b>{tk}</b></td><td class='l'>{bt.display}</td>"
            f"<td class='l'>{bt.fred_code}</td><td>{bt.n_days}</td>"
            f"<td>{m['n_trades']}</td><td>{_fmt_pct(m['win_rate'])}</td>"
            f"<td class='{ 'pos' if m['avg_pnl']>0 else 'neg' }'>{_fmt_num(m['avg_pnl'])}</td>"
            f"<td class='{ 'pos' if m['total_pnl']>0 else 'neg' }'>{_fmt_num(m['total_pnl'])}</td>"
            f"<td>{_fmt_num(m['sharpe'], 2)}</td>"
            f"<td>{corr_str}</td></tr>"
        )
    html.append("</table>")
    html.append(
        "<p class='muted'>Sharpe is annualised from 30-day variance-swap holding period "
        "(√(252/30)). P&L unit is dimensionless \"vol points\" (Carr-Wu variance swap "
        "payoff divided by 2·IV).</p>"
    )

    # Yearly grid
    html.append("<h2>Yearly OOS P&L by ticker (vol points)</h2>")
    years = sorted({y for bt in results.values() for y in bt.yearly_pnl})
    if years:
        html.append("<table><tr><th>Ticker</th>" + "".join(f"<th>{y}</th>" for y in years) + "</tr>")
        for tk, bt in results.items():
            html.append(f"<tr><td class='l'><b>{tk}</b></td>")
            for y in years:
                v = bt.yearly_pnl.get(y, 0.0)
                cls = "pos" if v > 0 else ("neg" if v < 0 else "")
                html.append(f"<td class='{cls}'>{_fmt_num(v, 3)}</td>")
            html.append("</tr>")
        html.append("</table>")

    # EXP-1220 yearly
    html.append("<h2>EXP-1220 reference (protected, % return)</h2>")
    if exp1220_yearly:
        html.append("<table><tr>" + "".join(f"<th>{y}</th>" for y in sorted(exp1220_yearly)) + "</tr><tr>")
        for y in sorted(exp1220_yearly):
            v = exp1220_yearly[y]
            cls = "pos" if v > 0 else "neg"
            html.append(f"<td class='{cls}'>{v:.2f}%</td>")
        html.append("</tr></table>")
    else:
        html.append("<p class='muted'>EXP-1220 summary not found on disk.</p>")

    # Fold detail (SPY as canonical)
    html.append("<h2>Walk-forward fold detail</h2>")
    for tk, bt in results.items():
        html.append(f"<h3>{tk} — {bt.display}</h3>")
        html.append("<table><tr><th>Train</th><th>Test</th><th>Threshold</th>"
                    "<th># trades</th><th>Win rate</th><th>Avg P&L</th><th>Total P&L</th></tr>")
        for f in bt.folds:
            cls = "pos" if f.total_pnl > 0 else ("neg" if f.total_pnl < 0 else "")
            html.append(
                f"<tr><td class='l'>{f.train_start} → {f.train_end}</td>"
                f"<td class='l'>{f.test_start} → {f.test_end}</td>"
                f"<td>{_fmt_num(f.threshold, 4)}</td>"
                f"<td>{f.n_trades}</td>"
                f"<td>{_fmt_pct(f.win_rate)}</td>"
                f"<td>{_fmt_num(f.avg_pnl)}</td>"
                f"<td class='{cls}'>{_fmt_num(f.total_pnl)}</td></tr>"
            )
        html.append("</table>")

    # Methodology + caveats
    html.append("<h2>Methodology & caveats</h2>")
    html.append("<ul>")
    html.append("<li><b>VRP signal:</b> IV² − RV²_30d (Carr-Wu 2009 variance risk premium).</li>")
    html.append("<li><b>OOS payoff:</b> IV²_entry − RV²_forward_30d, normalised by 2·IV.</li>")
    html.append("<li><b>Walk-forward:</b> 252-day train / 63-day OOS, step 63 days. "
                "Threshold is grid-searched on train {50,60,70,80}th percentiles of positive VRP.</li>")
    html.append("<li><b>Data sources:</b> Yahoo daily close (SPY/QQQ/IWM/EEM); "
                "FRED CBOE vol indices (VIXCLS, VXNCLS, RVXCLS, VXEEMCLS).</li>")
    html.append("<li><b>EFA substitution:</b> EFA has no FRED/Yahoo IV index "
                "(VXEFA was discontinued). EEM (^VXEEM) is used as the international-"
                "exposure proxy. Including EFA with a modelled IV would violate Rule Zero.</li>")
    html.append("<li><b>Correlation vs EXP-1220:</b> yearly only (n≤6), so should be "
                "read as directional, not statistically significant.</li>")
    html.append("<li><b>Not a tradable strategy yet:</b> P&L is in vol-point units; "
                "converting to dollar P&L requires real options fills via IronVault, "
                "which is out of scope for this research report.</li>")
    html.append("</ul>")

    html.append("</body></html>")
    return "".join(html)


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)

    results: Dict[str, TickerBacktest] = {}
    for ticker in PAIRS:
        try:
            print(f"[exp1660] loading {ticker}…", flush=True)
            df = load_pair(ticker)
            sig = compute_signals(df)
            bt = walk_forward(ticker, sig)
            results[ticker] = bt
            print(f"[exp1660] {ticker}: {bt.metrics['n_trades']} trades, "
                  f"WR={bt.metrics['win_rate']:.2%}, Sharpe={bt.metrics['sharpe']:.2f}",
                  flush=True)
        except Exception as e:
            print(f"[exp1660] {ticker} FAILED: {e}", flush=True)

    if not results:
        print("[exp1660] no results — aborting")
        return 1

    exp1220 = load_exp1220_yearly()
    correlations = {
        tk: yearly_correlation(exp1220, bt.yearly_pnl) for tk, bt in results.items()
    }

    html = render_html(results, exp1220, correlations)
    out_path = os.path.join(REPORT_DIR, "exp1660_vrp_deepening.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[exp1660] wrote {out_path}")

    # Also drop a summary JSON for downstream consumers
    summary_path = os.path.join(REPORT_DIR, "exp1660_vrp_deepening.json")
    summary = {
        "experiment": "EXP-1660",
        "description": "VRP Deepening — multi-asset variance risk premium walk-forward",
        "data_sources": {
            "prices": "Yahoo Finance (SPY,QQQ,IWM,EEM)",
            "iv": "FRED CBOE vol indices (VIXCLS,VXNCLS,RVXCLS,VXEEMCLS)",
        },
        "tickers": {
            tk: {
                "display": bt.display,
                "fred_code": bt.fred_code,
                "metrics": bt.metrics,
                "yearly_pnl": bt.yearly_pnl,
                "n_folds": len(bt.folds),
                "corr_vs_exp1220": correlations.get(tk),
            }
            for tk, bt in results.items()
        },
        "exp1220_yearly_return_pct": exp1220,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[exp1660] wrote {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
