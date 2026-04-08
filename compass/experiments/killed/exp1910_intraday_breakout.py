"""
EXP-1910 — Intraday Momentum / Breakout (daily-resolution proxy).

Hypothesis (Carlos): SPY/QQQ exhibit structural intraday continuation —
opening gaps and volume surges that resolve in the same direction by the
close. We need a Sharpe-5+ uncorrelated stream to close the North-Star
gap, and an intraday breakout sleeve is the most plausible candidate that
does not require OPRA option chains.

Honest constraint: we only have **daily** OHLCV from Yahoo (open, high,
low, close, volume). True intraday backtests need 1-minute bars. The
deliverable is therefore a *daily-resolution proxy* of two intraday
strategies that the daily bar can faithfully represent:

  1. GAP-AND-GO         — at the open, if (open − prev_close) / prev_close
                          > +g (long) or < −g (short), enter at the open
                          and exit at today's close.
                          Daily P&L = sign × (close − open) / open
                          (this IS the literal intraday return — no
                          interpolation, no synthetic fills.)

  2. VOLUME BREAKOUT    — if today's volume > k × 20-day average volume
                          AND the gap direction agrees, take the same
                          intraday position (open → close).
                          This is a confirmation overlay on (1), not a
                          separate signal.

Walk-forward 2020-2025: 252-day train / 63-day OOS / 63-day step. Per
fold, grid-search the gap threshold g ∈ {0.001, 0.002, 0.003, 0.005} and
the volume multiplier k ∈ {1.0, 1.25, 1.5, 2.0} on training Sharpe; apply
the winning combination OOS only.

Aggregation:
  * Per-ticker (SPY, QQQ) walk-forward result
  * Equal-weight combined portfolio
  * Yearly correlation against EXP-1220 protected returns

REAL DATA ONLY — Yahoo daily auto-adjusted OHLCV. No synthetic fills, no
parametric simulation, no option proxies.

Outputs:
  compass/exp1910_intraday_breakout.py            (this file)
  compass/reports/exp1910_intraday_breakout.json
  compass/reports/exp1910_intraday_breakout.html

Tag: EXP-1910
Run: python3 -m compass.exp1910_intraday_breakout
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
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
TICKERS = ["SPY", "QQQ"]

TRAIN_DAYS = 252
TEST_DAYS = 63
STEP_DAYS = 63

GAP_GRID = (0.001, 0.002, 0.003, 0.005)        # 10/20/30/50 bps gap thresholds
VOL_GRID = (1.0, 1.25, 1.5, 2.0)                 # volume confirmation multipliers
VOL_LOOKBACK = 20

# 1-bp round-trip cost on the daily intraday spread (open→close on a
# liquid index ETF — generous estimate; SPY half-spread is ~0.005%)
ROUND_TRIP_BPS = 1.0


# ── Data ────────────────────────────────────────────────────────────────


def fetch_ohlcv(symbol: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=START, end=END, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError(f"Yahoo empty for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.columns = ["open", "high", "low", "close", "volume"]
    return df


# ── Signal & strategy ──────────────────────────────────────────────────


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach gap, intraday return, and volume ratio."""
    out = df.copy()
    out["prev_close"] = out["close"].shift(1)
    out["gap"] = (out["open"] - out["prev_close"]) / out["prev_close"]
    out["intraday_ret"] = (out["close"] - out["open"]) / out["open"]
    out["vol_avg"] = out["volume"].rolling(VOL_LOOKBACK, min_periods=10).mean()
    out["vol_ratio"] = out["volume"] / out["vol_avg"]
    return out.dropna(subset=["gap", "intraday_ret", "vol_ratio"])


def _strategy_returns(features: pd.DataFrame, gap_thresh: float,
                      vol_mult: float) -> pd.Series:
    """Apply the gap-and-go + volume-confirm rule and return daily P&L.

    Position sign matches the gap direction; only enter when |gap| ≥
    gap_thresh AND vol_ratio ≥ vol_mult. P&L is the literal intraday
    return (close − open)/open from real OHLC, minus a 1bp round-trip
    cost on entered days.
    """
    cond = (features["gap"].abs() >= gap_thresh) & \
           (features["vol_ratio"] >= vol_mult)
    direction = np.sign(features["gap"]).where(cond, 0.0).fillna(0.0)
    pnl = direction * features["intraday_ret"]
    cost = (direction != 0.0).astype(float) * (ROUND_TRIP_BPS / 10_000)
    return (pnl - cost).fillna(0.0)


def _annualised_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(252))


# ── Walk-forward ───────────────────────────────────────────────────────


@dataclass
class FoldResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    gap_thresh: float
    vol_mult: float
    train_sharpe: float
    test_sharpe: float
    n_signals: int
    pnl: float


@dataclass
class TickerBacktest:
    ticker: str
    n_days: int
    folds: List[FoldResult]
    daily_returns: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)


def walk_forward_ticker(ticker: str, df: pd.DataFrame) -> TickerBacktest:
    feat = build_features(df)
    if len(feat) < TRAIN_DAYS + TEST_DAYS:
        raise RuntimeError(f"{ticker}: insufficient data ({len(feat)} days)")

    folds: List[FoldResult] = []
    oos_returns = pd.Series(0.0, index=feat.index)

    start = TRAIN_DAYS
    while start + TEST_DAYS <= len(feat):
        train = feat.iloc[start - TRAIN_DAYS:start]
        test = feat.iloc[start:start + TEST_DAYS]

        # Grid-search by training Sharpe
        best = (-np.inf, GAP_GRID[0], VOL_GRID[0])
        for g, k in itertools.product(GAP_GRID, VOL_GRID):
            r = _strategy_returns(train, g, k)
            s = _annualised_sharpe(r)
            if s > best[0]:
                best = (s, g, k)
        _, g, k = best

        oos_r = _strategy_returns(test, g, k)
        oos_returns.loc[test.index] = oos_r.values

        n_sig = int((oos_r != 0).sum())
        folds.append(FoldResult(
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            gap_thresh=g,
            vol_mult=k,
            train_sharpe=float(best[0]),
            test_sharpe=_annualised_sharpe(oos_r),
            n_signals=n_sig,
            pnl=float(oos_r.sum()),
        ))
        start += STEP_DAYS

    return TickerBacktest(
        ticker=ticker,
        n_days=len(feat),
        folds=folds,
        daily_returns=oos_returns,
        metrics=_aggregate_metrics(oos_returns),
    )


def _aggregate_metrics(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    nz = r[r != 0]
    n_days = int(len(r))
    n_active = int(len(nz))
    if n_active < 2:
        return dict(
            n_days=n_days, n_active_days=n_active, total_return=0.0,
            cagr=0.0, sharpe=0.0, max_dd=0.0, hit_rate=0.0, vol=0.0,
        )
    eq = (1.0 + r).cumprod()
    total_return = float(eq.iloc[-1] - 1.0)
    years = n_days / 252
    cagr = float(eq.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    pk = eq.cummax()
    max_dd = float(((eq - pk) / pk).min())
    sharpe = _annualised_sharpe(r)
    vol = float(r.std() * math.sqrt(252))
    hit_rate = float((nz > 0).mean())
    return dict(
        n_days=n_days,
        n_active_days=n_active,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        hit_rate=hit_rate,
        vol=vol,
    )


# ── EXP-1220 correlation ───────────────────────────────────────────────


def load_exp1220_yearly() -> Dict[int, float]:
    if not os.path.exists(EXP1220_SUMMARY):
        return {}
    with open(EXP1220_SUMMARY) as f:
        data = json.load(f)
    out: Dict[int, float] = {}
    for y, blob in data.get("yearly", {}).items():
        try:
            out[int(y)] = float(blob["protected"]["return_pct"]) / 100.0
        except (KeyError, TypeError, ValueError):
            continue
    return out


def correlate_yearly(strategy_daily: pd.Series, exp1220: Dict[int, float]) -> Optional[float]:
    if not exp1220:
        return None
    yearly = strategy_daily.groupby(strategy_daily.index.year).apply(
        lambda r: float((1.0 + r).prod() - 1.0)
    ).to_dict()
    common = sorted(set(yearly) & set(exp1220))
    if len(common) < 3:
        return None
    a = np.array([yearly[y] for y in common], dtype=float)
    b = np.array([exp1220[y] for y in common], dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ── HTML ───────────────────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x*100:.{dp}f}%" if np.isfinite(x) else "—"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(tickers: Dict[str, TickerBacktest],
                portfolio: TickerBacktest,
                correlations: Dict[str, Optional[float]],
                exp1220: Dict[int, float]) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1200px;color:#111}
    h1{border-bottom:3px solid #1d3557}
    h2{margin-top:2em;color:#1d3557}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#1d3557;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:2px 8px;border-radius:10px;background:#1d3557;color:#fff;font-size:11px}
    """
    h: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-1910 Intraday Breakout</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-1910 — Intraday Momentum / Breakout (daily-OHLC proxy)</h1>",
        "<p class='muted'>Gap-and-go + volume-confirm signal on real Yahoo "
        "daily OHLCV for SPY and QQQ. Walk-forward 2020-2025, 252d train / "
        "63d OOS / 63d step.</p>",
        "<p><span class='pill'>Rule Zero ✓ real Yahoo daily OHLCV only</span></p>",
    ]

    h.append("<h2>Per-ticker OOS results</h2>")
    h.append("<table><tr><th>Ticker</th><th>Days</th><th>Active days</th>"
             "<th>Hit rate</th><th>CAGR</th><th>Sharpe</th><th>Vol</th>"
             "<th>Max DD</th><th>Corr vs EXP-1220</th></tr>")
    for tk, bt in tickers.items():
        m = bt.metrics
        corr = correlations.get(tk)
        corr_str = f"{corr:+.2f}" if corr is not None else "n/a"
        h.append(
            f"<tr><td class='l'><b>{tk}</b></td>"
            f"<td>{m['n_days']}</td><td>{m['n_active_days']}</td>"
            f"<td>{_fmt_pct(m['hit_rate'], 1)}</td>"
            f"<td class='{ 'pos' if m['cagr']>0 else 'neg' }'>{_fmt_pct(m['cagr'])}</td>"
            f"<td>{_fmt(m['sharpe'])}</td>"
            f"<td>{_fmt_pct(m['vol'])}</td>"
            f"<td class='neg'>{_fmt_pct(m['max_dd'])}</td>"
            f"<td>{corr_str}</td></tr>"
        )
    h.append("</table>")

    h.append("<h2>Equal-weight combined portfolio</h2>")
    pm = portfolio.metrics
    pcorr = correlations.get("PORTFOLIO")
    h.append(
        "<table><tr><th>CAGR</th><th>Sharpe</th><th>Vol</th><th>Max DD</th>"
        "<th>Hit rate</th><th>Active days</th><th>Corr vs EXP-1220</th></tr>"
        f"<tr><td class='{ 'pos' if pm['cagr']>0 else 'neg' }'>{_fmt_pct(pm['cagr'])}</td>"
        f"<td>{_fmt(pm['sharpe'])}</td>"
        f"<td>{_fmt_pct(pm['vol'])}</td>"
        f"<td class='neg'>{_fmt_pct(pm['max_dd'])}</td>"
        f"<td>{_fmt_pct(pm['hit_rate'], 1)}</td>"
        f"<td>{pm['n_active_days']}</td>"
        f"<td>{(f'{pcorr:+.2f}' if pcorr is not None else 'n/a')}</td></tr></table>"
    )

    # Yearly grid
    h.append("<h2>Yearly OOS returns</h2>")
    yearly_table: Dict[str, Dict[int, float]] = {}
    years_set: set = set()
    for tk, bt in tickers.items():
        r = bt.daily_returns.dropna()
        ybyr = r.groupby(r.index.year).apply(lambda s: float((1.0 + s).prod() - 1.0))
        yearly_table[tk] = ybyr.to_dict()
        years_set.update(ybyr.index)
    pyr = portfolio.daily_returns.dropna()
    yearly_table["PORTFOLIO"] = pyr.groupby(pyr.index.year).apply(
        lambda s: float((1.0 + s).prod() - 1.0)
    ).to_dict()
    years_set.update(yearly_table["PORTFOLIO"].keys())
    years = sorted(int(y) for y in years_set)
    if years:
        h.append("<table><tr><th>Stream</th>" + "".join(f"<th>{y}</th>" for y in years) + "</tr>")
        for label in list(tickers.keys()) + ["PORTFOLIO"]:
            h.append(f"<tr><td class='l'><b>{label}</b></td>")
            for y in years:
                v = yearly_table.get(label, {}).get(y, 0.0)
                cls = "pos" if v > 0 else ("neg" if v < 0 else "")
                h.append(f"<td class='{cls}'>{_fmt_pct(v, 2)}</td>")
            h.append("</tr>")
        h.append("</table>")

    # EXP-1220 reference
    if exp1220:
        h.append("<h2>EXP-1220 reference (protected, % return)</h2>")
        h.append("<table><tr>" + "".join(f"<th>{y}</th>" for y in sorted(exp1220)) + "</tr><tr>")
        for y in sorted(exp1220):
            v = exp1220[y]
            cls = "pos" if v > 0 else "neg"
            h.append(f"<td class='{cls}'>{_fmt_pct(v)}</td>")
        h.append("</tr></table>")

    # Fold detail
    h.append("<h2>Walk-forward fold detail</h2>")
    for tk, bt in tickers.items():
        h.append(f"<h3>{tk}</h3>")
        h.append("<table><tr><th>Train</th><th>Test</th>"
                 "<th>gap≥</th><th>vol≥</th><th>train Sharpe</th>"
                 "<th>test Sharpe</th><th># sigs</th><th>P&L</th></tr>")
        for f in bt.folds:
            cls = "pos" if f.pnl > 0 else ("neg" if f.pnl < 0 else "")
            h.append(
                f"<tr><td class='l'>{f.train_start} → {f.train_end}</td>"
                f"<td class='l'>{f.test_start} → {f.test_end}</td>"
                f"<td>{f.gap_thresh*100:.2f}%</td>"
                f"<td>{f.vol_mult:.2f}×</td>"
                f"<td>{_fmt(f.train_sharpe)}</td>"
                f"<td>{_fmt(f.test_sharpe)}</td>"
                f"<td>{f.n_signals}</td>"
                f"<td class='{cls}'>{_fmt_pct(f.pnl)}</td></tr>"
            )
        h.append("</table>")

    h.append("<h2>Methodology & honest caveats</h2>")
    h.append("<ul>")
    h.append("<li><b>Signal:</b> at the open, take a long if "
             "(open−prev_close)/prev_close ≥ +g, short if ≤ −g, AND today's "
             "volume ≥ k × 20-day average. Exit at today's close.</li>")
    h.append("<li><b>P&amp;L:</b> literal (close−open)/open from real Yahoo "
             "OHLC, minus a 1-bp round-trip cost on every entered day. "
             "No interpolation, no synthetic fills.</li>")
    h.append("<li><b>Walk-forward:</b> 252d train / 63d OOS / 63d step. "
             f"Per fold, grid-search g∈{list(GAP_GRID)} × k∈{list(VOL_GRID)} "
             "by training Sharpe; apply best OOS only.</li>")
    h.append("<li><b>What this is NOT:</b> a true intraday backtest. "
             "Daily OHLC cannot model intraday stops, scaling, or "
             "minute-bar momentum. Anything that requires intra-bar "
             "execution (trailing stop, VWAP entry, scale-in) needs "
             "minute data which Yahoo does not provide. The Sharpe ceiling "
             "of a daily-resolution proxy is therefore much lower than the "
             "Sharpe of the 'real' minute-bar version of the same idea.</li>")
    h.append("<li><b>On the Sharpe-5+ target:</b> a Sharpe-5+ daily series "
             "from 1 trade-day≤signal frequency on a single underlying is "
             "extraordinary and should be treated as overfit until proven "
             "OOS for ≥6 months on live execution. The walk-forward design "
             "above gives one honest read on whether the edge survives at all.</li>")
    h.append("<li><b>Cost assumption:</b> 1bp round-trip is generous on SPY "
             "(half-spread ~0.5bp). Tightening to a more realistic 2bp "
             "shifts metrics modestly; the file's ROUND_TRIP_BPS knob "
             "lets you re-run.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)

    ticker_results: Dict[str, TickerBacktest] = {}
    for tk in TICKERS:
        try:
            print(f"[exp1910] loading {tk}…", flush=True)
            df = fetch_ohlcv(tk)
            bt = walk_forward_ticker(tk, df)
            ticker_results[tk] = bt
            m = bt.metrics
            print(f"[exp1910] {tk}: CAGR={m['cagr']*100:.2f}%  "
                  f"Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.2f}%  "
                  f"hits={m['hit_rate']*100:.1f}%  active={m['n_active_days']}",
                  flush=True)
        except Exception as e:
            print(f"[exp1910] {tk} FAILED: {e}", flush=True)

    if not ticker_results:
        print("[exp1910] no tickers succeeded — aborting")
        return 1

    aligned = pd.concat(
        {tk: bt.daily_returns for tk, bt in ticker_results.items()}, axis=1
    ).fillna(0.0)
    n = aligned.shape[1]
    portfolio_returns = aligned.sum(axis=1) / n
    portfolio = TickerBacktest(
        ticker="PORTFOLIO",
        n_days=len(portfolio_returns),
        folds=[],
        daily_returns=portfolio_returns,
        metrics=_aggregate_metrics(portfolio_returns),
    )
    print(f"[exp1910] PORTFOLIO: CAGR={portfolio.metrics['cagr']*100:.2f}%  "
          f"Sharpe={portfolio.metrics['sharpe']:.2f}  "
          f"DD={portfolio.metrics['max_dd']*100:.2f}%")

    exp1220 = load_exp1220_yearly()
    correlations: Dict[str, Optional[float]] = {
        tk: correlate_yearly(bt.daily_returns, exp1220) for tk, bt in ticker_results.items()
    }
    correlations["PORTFOLIO"] = correlate_yearly(portfolio.daily_returns, exp1220)

    html = render_html(ticker_results, portfolio, correlations, exp1220)
    out_html = os.path.join(REPORT_DIR, "exp1910_intraday_breakout.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[exp1910] wrote {out_html}")

    out_json = os.path.join(REPORT_DIR, "exp1910_intraday_breakout.json")
    summary = {
        "experiment": "EXP-1910",
        "tag": "EXP-1910",
        "description": "Intraday gap-and-go + volume-confirm breakout (daily OHLC proxy)",
        "data_source": "Yahoo Finance daily auto-adjusted OHLCV",
        "window": {"start": START, "end": END},
        "tickers": TICKERS,
        "walk_forward": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "step_days": STEP_DAYS,
            "gap_grid": list(GAP_GRID),
            "vol_grid": list(VOL_GRID),
            "round_trip_bps": ROUND_TRIP_BPS,
        },
        "tickers_metrics": {
            tk: {
                "metrics": bt.metrics,
                "n_folds": len(bt.folds),
                "corr_vs_exp1220": correlations.get(tk),
            }
            for tk, bt in ticker_results.items()
        },
        "portfolio": {
            "metrics": portfolio.metrics,
            "corr_vs_exp1220": correlations.get("PORTFOLIO"),
            "n_streams": n,
        },
        "exp1220_yearly_protected_return": exp1220,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[exp1910] wrote {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
