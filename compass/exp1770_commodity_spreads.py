"""
EXP-1770 — Commodity Spreads (ETF-ratio mean reversion).

Companion to compass/exp1770_commodity_calendars.py. Where the calendars
script harvested ETF-vs-future roll yield, this one tests pairwise
mean-reversion on commodity ETF *ratios* — a calendar-spread-like trade
in the sense that two storable commodities tend to mean-revert to each
other once macro shocks fade.

Universe (per Carlos directive 2026-04-07):
    USO  (WTI crude oil)
    UNG  (US natural gas)
    GLD  (gold)
    DBA  (agriculture basket)

Methodology — REAL DATA ONLY (Rule Zero):
  * Daily adjusted close from Yahoo Finance, 2020-01-01 → 2025-12-31.
  * For every unordered pair (a, b), build the log-ratio
        x_t = log(a_t) - log(b_t).
  * Z-score x against its rolling 60-day mean & std.
  * Mean-reversion rule:
        z >  +z_entry → SHORT spread (sell a, buy b)
        z <  -z_entry → LONG spread  (buy a, sell b)
        |z| < z_exit  → flat
  * Walk-forward: 252-day train (~1y) → 63-day OOS (~3m), step 63d, run
    over the full 2020-2025 window. Per fold, grid-search z_entry on
    {1.0, 1.5, 2.0} and z_exit on {0.0, 0.25, 0.5} on the training window
    by Sharpe; apply best on OOS only.
  * Daily P&L = position × spread_return where spread_return is the
    daily change in log(a/b). Reported in decimal units (e.g. 0.01 = +1%).

Aggregation:
  * Per-pair OOS: CAGR, Sharpe, max drawdown, hit rate.
  * Combined portfolio: equal-weight average of all pairs (rebalanced daily).
  * Yearly correlation against EXP-1220 protected returns
    (experiments/EXP-1220-real/results/summary.json).

Outputs:
  compass/reports/exp1770_commodity_spreads.html
  compass/reports/exp1770_commodity_spreads.json   (gitignored)
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

TICKERS = ["USO", "UNG", "GLD", "DBA"]
DISPLAY = {
    "USO": "WTI Crude (USO)",
    "UNG": "US Natural Gas (UNG)",
    "GLD": "Gold (GLD)",
    "DBA": "Agriculture Basket (DBA)",
}

ZSCORE_WINDOW = 60          # rolling lookback for the spread z-score
TRAIN_DAYS = 252            # ~1y train
TEST_DAYS = 63              # ~3m OOS
STEP_DAYS = 63              # walk-forward step
Z_ENTRY_GRID = (1.0, 1.5, 2.0)
Z_EXIT_GRID = (0.0, 0.25, 0.5)


# ── Data ────────────────────────────────────────────────────────────────


def fetch_close(symbol: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(symbol, start=START, end=END, progress=False, auto_adjust=True)
    if df is None or len(df) == 0:
        raise RuntimeError(f"Yahoo empty for {symbol}")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    s.name = symbol
    return s.dropna()


def load_universe() -> pd.DataFrame:
    series = {tk: fetch_close(tk) for tk in TICKERS}
    df = pd.concat(series, axis=1, join="inner").dropna()
    return df


# ── Spread + signal ────────────────────────────────────────────────────


def build_spread(prices: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    out = pd.DataFrame(index=prices.index)
    out["log_ratio"] = np.log(prices[a]) - np.log(prices[b])
    out["spread_ret"] = out["log_ratio"].diff()
    mu = out["log_ratio"].rolling(ZSCORE_WINDOW).mean()
    sd = out["log_ratio"].rolling(ZSCORE_WINDOW).std(ddof=0)
    out["z"] = (out["log_ratio"] - mu) / sd.replace(0, np.nan)
    return out.dropna()


def _strategy_returns(spread: pd.DataFrame, z_entry: float, z_exit: float) -> pd.Series:
    """Return daily strategy P&L for a given (z_entry, z_exit) rule."""
    z = spread["z"].values
    ret = spread["spread_ret"].values
    pos = np.zeros_like(ret)
    cur = 0.0
    for i in range(len(z)):
        zi = z[i]
        if cur == 0.0:
            if zi > z_entry:
                cur = -1.0
            elif zi < -z_entry:
                cur = 1.0
        else:
            if abs(zi) <= z_exit:
                cur = 0.0
        pos[i] = cur
    # P&L: position established at end of day t earns spread_ret[t+1]
    pos_lag = np.concatenate([[0.0], pos[:-1]])
    pnl = pos_lag * ret
    return pd.Series(pnl, index=spread.index)


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
    z_entry: float
    z_exit: float
    n_days: int
    pnl: float
    sharpe: float


@dataclass
class PairBacktest:
    pair: str
    a: str
    b: str
    n_days: int
    folds: List[FoldResult]
    daily_returns: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)


def walk_forward_pair(a: str, b: str, prices: pd.DataFrame) -> PairBacktest:
    spread = build_spread(prices, a, b)
    if len(spread) < TRAIN_DAYS + TEST_DAYS:
        raise RuntimeError(f"{a}/{b}: insufficient data ({len(spread)} days)")

    folds: List[FoldResult] = []
    oos = pd.Series(0.0, index=spread.index)

    start = TRAIN_DAYS
    while start + TEST_DAYS <= len(spread):
        train = spread.iloc[start - TRAIN_DAYS:start]
        test = spread.iloc[start:start + TEST_DAYS]

        # Grid-search (z_entry, z_exit) by training Sharpe
        best_sharpe = -np.inf
        best_params: Tuple[float, float] = (Z_ENTRY_GRID[0], Z_EXIT_GRID[0])
        for ze in Z_ENTRY_GRID:
            for zx in Z_EXIT_GRID:
                if zx >= ze:
                    continue
                r = _strategy_returns(train, ze, zx)
                s = _annualised_sharpe(r)
                if s > best_sharpe:
                    best_sharpe = s
                    best_params = (ze, zx)

        ze, zx = best_params
        oos_r = _strategy_returns(test, ze, zx)
        oos.loc[test.index] = oos_r.values

        folds.append(FoldResult(
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            z_entry=ze,
            z_exit=zx,
            n_days=len(test),
            pnl=float(oos_r.sum()),
            sharpe=_annualised_sharpe(oos_r),
        ))
        start += STEP_DAYS

    return PairBacktest(
        pair=f"{a}/{b}",
        a=a, b=b,
        n_days=len(spread),
        folds=folds,
        daily_returns=oos,
        metrics=_aggregate_metrics(oos),
    )


def _aggregate_metrics(returns: pd.Series) -> Dict[str, float]:
    r = returns.dropna()
    nz = r[r != 0]
    n_days = int(len(r))
    n_active = int(len(nz))
    if n_active < 2:
        return dict(
            n_days=n_days, n_active_days=n_active, total_return=0.0,
            cagr=0.0, sharpe=0.0, max_dd=0.0, hit_rate=0.0,
        )
    eq = (1.0 + r).cumprod()
    total_return = float(eq.iloc[-1] - 1.0)
    years = n_days / 252
    cagr = float(eq.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    pk = eq.cummax()
    max_dd = float(((eq - pk) / pk).min())
    sharpe = _annualised_sharpe(r)
    hit_rate = float((nz > 0).mean())
    return dict(
        n_days=n_days,
        n_active_days=n_active,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_dd=max_dd,
        hit_rate=hit_rate,
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


# ── Report ─────────────────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x*100:.{dp}f}%" if np.isfinite(x) else "—"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(
    pairs: Dict[str, PairBacktest],
    portfolio: PairBacktest,
    correlations: Dict[str, Optional[float]],
    exp1220: Dict[int, float],
) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1200px;color:#111}
    h1{border-bottom:3px solid #4a4e2a}
    h2{margin-top:2em;color:#4a4e2a}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#4a4e2a;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:2px 8px;border-radius:10px;background:#4a4e2a;color:#fff;font-size:11px}
    """
    h: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-1770 Commodity Spreads — ETF Ratio Mean Reversion</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-1770 — Commodity Spreads (ETF Ratio Mean Reversion)</h1>",
        "<p class='muted'>Universe: USO / UNG / GLD / DBA. Pairwise log-ratio "
        "mean reversion, 60-day z-score, walk-forward 2020-2025 (252d train / "
        "63d OOS, quarterly step). Real Yahoo data only.</p>",
        "<p><span class='pill'>Rule Zero ✓ no synthetic data</span></p>",
    ]

    # Per-pair table
    h.append("<h2>Per-pair OOS metrics</h2>")
    h.append("<table><tr><th>Pair</th><th>Days</th><th>Active days</th>"
             "<th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Hit rate</th>"
             "<th>Corr vs EXP-1220</th></tr>")
    for pk, bt in pairs.items():
        m = bt.metrics
        corr = correlations.get(pk)
        corr_str = f"{corr:+.2f}" if corr is not None else "n/a"
        h.append(
            f"<tr><td class='l'><b>{pk}</b></td>"
            f"<td>{m['n_days']}</td><td>{m['n_active_days']}</td>"
            f"<td class='{ 'pos' if m['cagr']>0 else 'neg' }'>{_fmt_pct(m['cagr'])}</td>"
            f"<td>{_fmt(m['sharpe'])}</td>"
            f"<td class='neg'>{_fmt_pct(m['max_dd'])}</td>"
            f"<td>{_fmt_pct(m['hit_rate'], 1)}</td>"
            f"<td>{corr_str}</td></tr>"
        )
    h.append("</table>")

    # Portfolio
    h.append("<h2>Equal-weight combined portfolio</h2>")
    pm = portfolio.metrics
    pcorr = correlations.get("PORTFOLIO")
    h.append(
        "<table><tr><th>CAGR</th><th>Sharpe</th><th>Max DD</th>"
        "<th>Hit rate</th><th>Active days</th><th>Corr vs EXP-1220</th></tr>"
        f"<tr><td class='{ 'pos' if pm['cagr']>0 else 'neg' }'>{_fmt_pct(pm['cagr'])}</td>"
        f"<td>{_fmt(pm['sharpe'])}</td>"
        f"<td class='neg'>{_fmt_pct(pm['max_dd'])}</td>"
        f"<td>{_fmt_pct(pm['hit_rate'], 1)}</td>"
        f"<td>{pm['n_active_days']}</td>"
        f"<td>{(f'{pcorr:+.2f}' if pcorr is not None else 'n/a')}</td></tr></table>"
    )

    # Yearly grid
    h.append("<h2>Yearly OOS returns by pair</h2>")
    yearly_table: Dict[str, Dict[int, float]] = {}
    years_set: set = set()
    for pk, bt in pairs.items():
        r = bt.daily_returns.dropna()
        ybyr = r.groupby(r.index.year).apply(lambda s: float((1.0 + s).prod() - 1.0))
        yearly_table[pk] = ybyr.to_dict()
        years_set.update(ybyr.index)
    pyr = portfolio.daily_returns.dropna()
    yearly_table["PORTFOLIO"] = pyr.groupby(pyr.index.year).apply(
        lambda s: float((1.0 + s).prod() - 1.0)
    ).to_dict()
    years_set.update(yearly_table["PORTFOLIO"].keys())
    years = sorted(int(y) for y in years_set)
    if years:
        h.append("<table><tr><th>Pair</th>" + "".join(f"<th>{y}</th>" for y in years) + "</tr>")
        for label in list(pairs.keys()) + ["PORTFOLIO"]:
            h.append(f"<tr><td class='l'><b>{label}</b></td>")
            for y in years:
                v = yearly_table.get(label, {}).get(y, 0.0)
                cls = "pos" if v > 0 else ("neg" if v < 0 else "")
                h.append(f"<td class='{cls}'>{_fmt_pct(v, 2)}</td>")
            h.append("</tr>")
        h.append("</table>")

    # EXP-1220 reference
    h.append("<h2>EXP-1220 reference (protected, % return)</h2>")
    if exp1220:
        h.append("<table><tr>" + "".join(f"<th>{y}</th>" for y in sorted(exp1220)) + "</tr><tr>")
        for y in sorted(exp1220):
            v = exp1220[y]
            cls = "pos" if v > 0 else "neg"
            h.append(f"<td class='{cls}'>{_fmt_pct(v)}</td>")
        h.append("</tr></table>")

    # Methodology
    h.append("<h2>Methodology & caveats</h2>")
    h.append("<ul>")
    h.append("<li><b>Spread:</b> log(a) − log(b) on every unordered pair from "
             "{USO, UNG, GLD, DBA}.</li>")
    h.append("<li><b>Signal:</b> 60-day z-score of the log ratio.</li>")
    h.append("<li><b>Rule:</b> mean reversion. Enter at |z|>z_entry, exit at "
             "|z|<z_exit. Grid {z_entry∈(1.0,1.5,2.0), z_exit∈(0.0,0.25,0.5)}, "
             "best train-Sharpe wins per fold.</li>")
    h.append("<li><b>Walk-forward:</b> 252d train / 63d OOS / 63d step.</li>")
    h.append("<li><b>Data:</b> Yahoo Finance daily auto-adjusted close, "
             f"{START}→{END}. Real prices, no synthetics.</li>")
    h.append("<li><b>Caveats:</b> ETF ratio mean reversion is not a true "
             "calendar spread — it is a relative-value pair trade. Two ETFs "
             "tracking different commodities can drift apart for years (regime "
             "change), so this should be combined with a regime filter or hard "
             "stop before any production use.</li>")
    h.append("<li><b>Correlation vs EXP-1220:</b> yearly only (n≤6). Directional, "
             "not statistically significant.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("[exp1770-spreads] downloading universe…", flush=True)
    prices = load_universe()
    print(f"[exp1770-spreads] universe loaded: {prices.shape[0]} days × {prices.shape[1]} tickers")

    pair_results: Dict[str, PairBacktest] = {}
    for a, b in itertools.combinations(TICKERS, 2):
        try:
            bt = walk_forward_pair(a, b, prices)
            pair_results[bt.pair] = bt
            m = bt.metrics
            print(f"[exp1770-spreads] {bt.pair}: CAGR={m['cagr']*100:.2f}%  "
                  f"Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.2f}%",
                  flush=True)
        except Exception as e:
            print(f"[exp1770-spreads] {a}/{b} FAILED: {e}", flush=True)

    if not pair_results:
        print("[exp1770-spreads] no pairs succeeded — aborting")
        return 1

    aligned = pd.concat(
        {pk: bt.daily_returns for pk, bt in pair_results.items()}, axis=1
    ).fillna(0.0)
    n_pairs = aligned.shape[1]
    portfolio_returns = aligned.sum(axis=1) / n_pairs
    portfolio = PairBacktest(
        pair="PORTFOLIO", a="-", b="-",
        n_days=len(portfolio_returns),
        folds=[],
        daily_returns=portfolio_returns,
        metrics=_aggregate_metrics(portfolio_returns),
    )
    print(f"[exp1770-spreads] PORTFOLIO: CAGR={portfolio.metrics['cagr']*100:.2f}%  "
          f"Sharpe={portfolio.metrics['sharpe']:.2f}  "
          f"DD={portfolio.metrics['max_dd']*100:.2f}%")

    exp1220 = load_exp1220_yearly()
    correlations: Dict[str, Optional[float]] = {
        pk: correlate_yearly(bt.daily_returns, exp1220) for pk, bt in pair_results.items()
    }
    correlations["PORTFOLIO"] = correlate_yearly(portfolio.daily_returns, exp1220)

    html = render_html(pair_results, portfolio, correlations, exp1220)
    out_html = os.path.join(REPORT_DIR, "exp1770_commodity_spreads.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[exp1770-spreads] wrote {out_html}")

    out_json = os.path.join(REPORT_DIR, "exp1770_commodity_spreads.json")
    summary = {
        "experiment": "EXP-1770",
        "variant": "etf_ratio_mean_reversion",
        "description": "Commodity ETF pairwise log-ratio mean reversion",
        "universe": TICKERS,
        "data_source": "Yahoo Finance (real, no synthetics)",
        "window": {"start": START, "end": END},
        "walk_forward": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "step_days": STEP_DAYS,
            "z_entry_grid": list(Z_ENTRY_GRID),
            "z_exit_grid": list(Z_EXIT_GRID),
        },
        "pairs": {
            pk: {
                "metrics": bt.metrics,
                "n_folds": len(bt.folds),
                "corr_vs_exp1220": correlations.get(pk),
            }
            for pk, bt in pair_results.items()
        },
        "portfolio": {
            "metrics": portfolio.metrics,
            "corr_vs_exp1220": correlations.get("PORTFOLIO"),
            "n_pairs": n_pairs,
        },
        "exp1220_yearly_protected_return": exp1220,
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[exp1770-spreads] wrote {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
