"""
EXP-1770 — Commodity Calendar Spreads (roll-yield harvest).

Edge: storable commodities trade in persistent contango or backwardation.
The shape of the futures curve creates a structural drift between an ETF
that rolls front-month futures (USO, UNG, GLD, SLV) and the underlying
front-month future itself. That drift IS the calendar spread.

Methodology — all real, no synthetic prices (Rule Zero):
  * Underlying futures (continuous front-month) from Yahoo: CL=F, NG=F,
    GC=F, SI=F.
  * ETF NAVs from Yahoo: USO, UNG, GLD, SLV.
  * Daily roll-yield = ETF_return − Future_return. Cumulated, this isolates
    the calendar drift each ETF earns/pays from rolling.
  * Strategy: take a directional position on the (ETF − Future) spread
    based on a 60-day signal:
        - Mean-reversion: if 60d cum spread > +1σ → short (expect snapback)
        - Momentum:       if 60d cum spread > +1σ → long  (drift continues)
    Walk-forward chooses whichever rule beats buy-and-hold spread on the
    training window, then applies it OOS.
  * Walk-forward: 504-day train (~2y), 126-day test (~6m), step 126 days.
    Folds run 2015-2025.
  * Pair P&L is reported in spread-return units (decimal). The combined
    portfolio is an equal-weight blend of all four pairs (rebalanced daily).

Outputs:
  compass/reports/exp1770_commodity_calendars.html
  compass/reports/exp1770_commodity_calendars.json   (gitignored)

Run::
    python3 -m compass.exp1770_commodity_calendars
"""

from __future__ import annotations

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

START = "2015-01-01"
END = "2025-12-31"

# (ETF symbol, continuous front-month future, display name)
PAIRS: Dict[str, Tuple[str, str, str]] = {
    "USO": ("USO", "CL=F", "WTI Crude (USO − CL=F)"),
    "UNG": ("UNG", "NG=F", "Natural Gas (UNG − NG=F)"),
    "GLD": ("GLD", "GC=F", "Gold (GLD − GC=F)"),
    "SLV": ("SLV", "SI=F", "Silver (SLV − SI=F)"),
}

SIGNAL_WINDOW = 60        # rolling window for spread signal
TRAIN_DAYS = 504          # ~2y train
TEST_DAYS = 126           # ~6m OOS
STEP_DAYS = 126           # walk-forward step
Z_THRESH = 1.0            # signal z-score gate


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


def load_pair(etf: str, future: str) -> pd.DataFrame:
    e = fetch_close(etf)
    f = fetch_close(future)
    df = pd.concat([e.rename("etf"), f.rename("fut")], axis=1, join="inner").dropna()
    df["etf_ret"] = np.log(df["etf"]).diff()
    df["fut_ret"] = np.log(df["fut"]).diff()
    # Calendar-spread return: long ETF, short future. In persistent
    # contango this is negative (the roll drag the ETF eats); in
    # backwardation it is positive.
    df["spread_ret"] = df["etf_ret"] - df["fut_ret"]
    return df.dropna()


# ── Strategy ────────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    rule: str            # "mean_reversion", "momentum", or "flat"
    n_days: int
    pnl: float
    sharpe: float


@dataclass
class PairBacktest:
    pair: str
    display: str
    n_days: int
    folds: List[FoldResult]
    daily_returns: pd.Series        # OOS strategy daily return
    metrics: Dict[str, float] = field(default_factory=dict)


def _signal_zscore(spread_ret: pd.Series, window: int = SIGNAL_WINDOW) -> pd.Series:
    cum = spread_ret.rolling(window).sum()
    mu = cum.rolling(window).mean()
    sd = cum.rolling(window).std(ddof=0)
    return (cum - mu) / sd.replace(0, np.nan)


def _apply_rule(spread_ret: pd.Series, signal: pd.Series, rule: str) -> pd.Series:
    """Return strategy daily P&L given rule + signal.

    All positions are entered at end-of-day t and earn spread_ret[t+1].
    """
    if rule == "flat":
        return pd.Series(0.0, index=spread_ret.index)

    pos = pd.Series(0.0, index=spread_ret.index)
    if rule == "mean_reversion":
        pos[signal > Z_THRESH] = -1.0
        pos[signal < -Z_THRESH] = 1.0
    elif rule == "momentum":
        pos[signal > Z_THRESH] = 1.0
        pos[signal < -Z_THRESH] = -1.0
    else:
        raise ValueError(f"unknown rule {rule}")

    # Carry position forward when |signal| ≤ z (sticky position) to avoid
    # whipsaw — exit only when signal crosses zero.
    pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)
    pos[signal.abs() < 0.05] = 0.0  # exit deadzone

    return pos.shift(1).fillna(0.0) * spread_ret


def _annualised_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * math.sqrt(252))


def walk_forward(pair: str, df: pd.DataFrame) -> PairBacktest:
    if len(df) < TRAIN_DAYS + TEST_DAYS:
        raise RuntimeError(f"{pair}: insufficient data ({len(df)} days)")

    df = df.copy()
    df["signal"] = _signal_zscore(df["spread_ret"])
    df = df.dropna(subset=["signal"])

    fold_results: List[FoldResult] = []
    oos_returns = pd.Series(0.0, index=df.index)

    start = TRAIN_DAYS
    while start + TEST_DAYS <= len(df):
        train = df.iloc[start - TRAIN_DAYS:start]
        test = df.iloc[start:start + TEST_DAYS]

        # Score each rule on training window
        best_rule = "flat"
        best_score = -np.inf
        for rule in ("mean_reversion", "momentum", "flat"):
            r = _apply_rule(train["spread_ret"], train["signal"], rule)
            score = _annualised_sharpe(r)
            if score > best_score:
                best_score = score
                best_rule = rule

        oos_r = _apply_rule(test["spread_ret"], test["signal"], best_rule)
        oos_returns.loc[test.index] = oos_r.values

        fold_results.append(FoldResult(
            train_start=str(train.index[0].date()),
            train_end=str(train.index[-1].date()),
            test_start=str(test.index[0].date()),
            test_end=str(test.index[-1].date()),
            rule=best_rule,
            n_days=len(test),
            pnl=float(oos_r.sum()),
            sharpe=_annualised_sharpe(oos_r),
        ))
        start += STEP_DAYS

    metrics = _aggregate_metrics(oos_returns)
    return PairBacktest(
        pair=pair,
        display=PAIRS[pair][2],
        n_days=len(df),
        folds=fold_results,
        daily_returns=oos_returns,
        metrics=metrics,
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
    cagr = (eq.iloc[-1] ** (1 / years) - 1.0) if years > 0 else 0.0
    pk = eq.cummax()
    max_dd = float(((eq - pk) / pk).min())
    sharpe = _annualised_sharpe(r)
    hit_rate = float((nz > 0).mean())
    return dict(
        n_days=n_days,
        n_active_days=n_active,
        total_return=total_return,
        cagr=float(cagr),
        sharpe=sharpe,
        max_dd=max_dd,
        hit_rate=hit_rate,
    )


# ── EXP-1220 correlation ────────────────────────────────────────────────


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


def correlate_yearly(strategy_daily: pd.Series, exp1220_yearly: Dict[int, float]) -> Optional[float]:
    if not exp1220_yearly:
        return None
    yearly = strategy_daily.groupby(strategy_daily.index.year).apply(
        lambda r: float((1.0 + r).prod() - 1.0)
    ).to_dict()
    common = sorted(set(yearly) & set(exp1220_yearly))
    if len(common) < 3:
        return None
    a = np.array([yearly[y] for y in common], dtype=float)
    b = np.array([exp1220_yearly[y] for y in common], dtype=float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ── Report ──────────────────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x*100:.{dp}f}%" if np.isfinite(x) else "—"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(
    pairs: Dict[str, PairBacktest],
    portfolio: PairBacktest,
    correlations: Dict[str, Optional[float]],
    exp1220_yearly: Dict[int, float],
) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1200px;color:#111}
    h1{border-bottom:3px solid #5d3a00}
    h2{margin-top:2em;color:#5d3a00}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#5d3a00;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:2px 8px;border-radius:10px;background:#5d3a00;color:#fff;font-size:11px}
    """
    h: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-1770 Commodity Calendar Spreads</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-1770 — Commodity Calendar Spreads</h1>",
        "<p class='muted'>Roll-yield harvest on USO/UNG/GLD/SLV vs front-month "
        "futures (CL=F/NG=F/GC=F/SI=F). Walk-forward 2015-2025, 504d train / "
        "126d OOS, quarterly step. Real Yahoo data only.</p>",
        "<p><span class='pill'>Rule Zero ✓ no synthetic data</span></p>",
    ]

    # Per-pair metrics
    h.append("<h2>Per-pair OOS metrics</h2>")
    h.append("<table><tr><th>Pair</th><th>Description</th>"
             "<th>Days</th><th>Active days</th><th>CAGR</th><th>Sharpe</th>"
             "<th>Max DD</th><th>Hit rate</th><th>Corr vs EXP-1220 (yearly)</th></tr>")
    for pk, bt in pairs.items():
        m = bt.metrics
        corr = correlations.get(pk)
        corr_str = f"{corr:+.2f}" if corr is not None else "n/a"
        h.append(
            f"<tr><td class='l'><b>{pk}</b></td>"
            f"<td class='l'>{bt.display}</td>"
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
    years_set = set()
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
    if exp1220_yearly:
        h.append("<table><tr>" + "".join(f"<th>{y}</th>" for y in sorted(exp1220_yearly)) + "</tr><tr>")
        for y in sorted(exp1220_yearly):
            v = exp1220_yearly[y]
            cls = "pos" if v > 0 else "neg"
            h.append(f"<td class='{cls}'>{_fmt_pct(v)}</td>")
        h.append("</tr></table>")

    # Fold detail
    h.append("<h2>Walk-forward fold detail</h2>")
    for pk, bt in pairs.items():
        h.append(f"<h3>{pk} — {bt.display}</h3>")
        h.append("<table><tr><th>Train</th><th>Test</th><th>Selected rule</th>"
                 "<th>Days</th><th>P&L</th><th>Sharpe</th></tr>")
        for f in bt.folds:
            cls = "pos" if f.pnl > 0 else ("neg" if f.pnl < 0 else "")
            h.append(
                f"<tr><td class='l'>{f.train_start} → {f.train_end}</td>"
                f"<td class='l'>{f.test_start} → {f.test_end}</td>"
                f"<td class='l'>{f.rule}</td>"
                f"<td>{f.n_days}</td>"
                f"<td class='{cls}'>{_fmt_pct(f.pnl)}</td>"
                f"<td>{_fmt(f.sharpe)}</td></tr>"
            )
        h.append("</table>")

    # Methodology
    h.append("<h2>Methodology & caveats</h2>")
    h.append("<ul>")
    h.append("<li><b>Spread definition:</b> daily ETF return minus continuous "
             "front-month future return. Captures the structural roll yield "
             "(contango drag is negative, backwardation premium is positive).</li>")
    h.append("<li><b>Signal:</b> 60-day cumulative spread, z-scored against its "
             "own 60-day mean/std. ±1σ entry, zero-cross exit.</li>")
    h.append("<li><b>Rule selection:</b> per fold, pick whichever of "
             "{mean_reversion, momentum, flat} maximises training Sharpe; apply OOS.</li>")
    h.append("<li><b>Walk-forward:</b> 504d train, 126d OOS, step 126d.</li>")
    h.append("<li><b>Data sources:</b> Yahoo Finance daily adjusted close — "
             "ETFs (USO/UNG/GLD/SLV), continuous futures (CL=F/NG=F/GC=F/SI=F).</li>")
    h.append("<li><b>What this is NOT:</b> a real options calendar spread. "
             "It's a futures roll-yield harvest. Implementation in production "
             "would require futures execution, not options. Use this report as "
             "a research signal — not a tradable strategy until paired with a "
             "real futures execution path.</li>")
    h.append("<li><b>Correlation vs EXP-1220:</b> yearly only (n≤6). Directional, "
             "not statistically significant.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ────────────────────────────────────────────────────────────────


def main() -> int:
    os.makedirs(REPORT_DIR, exist_ok=True)

    pair_results: Dict[str, PairBacktest] = {}
    for pair, (etf, fut, _) in PAIRS.items():
        try:
            print(f"[exp1770] loading {pair} ({etf} vs {fut})…", flush=True)
            df = load_pair(etf, fut)
            bt = walk_forward(pair, df)
            pair_results[pair] = bt
            m = bt.metrics
            print(f"[exp1770] {pair}: CAGR={m['cagr']*100:.2f}%  "
                  f"Sharpe={m['sharpe']:.2f}  DD={m['max_dd']*100:.2f}%",
                  flush=True)
        except Exception as e:
            print(f"[exp1770] {pair} FAILED: {e}", flush=True)

    if not pair_results:
        print("[exp1770] no pairs succeeded — aborting")
        return 1

    # Combined portfolio: equal-weight blend of pair daily returns
    aligned = pd.concat(
        {pk: bt.daily_returns for pk, bt in pair_results.items()}, axis=1
    ).fillna(0.0)
    n_pairs = aligned.shape[1]
    portfolio_returns = aligned.sum(axis=1) / n_pairs
    portfolio = PairBacktest(
        pair="PORTFOLIO",
        display=f"Equal-weight {n_pairs}-pair blend",
        n_days=len(portfolio_returns),
        folds=[],
        daily_returns=portfolio_returns,
        metrics=_aggregate_metrics(portfolio_returns),
    )
    print(f"[exp1770] PORTFOLIO: CAGR={portfolio.metrics['cagr']*100:.2f}%  "
          f"Sharpe={portfolio.metrics['sharpe']:.2f}  "
          f"DD={portfolio.metrics['max_dd']*100:.2f}%")

    # Correlation vs EXP-1220
    exp1220 = load_exp1220_yearly()
    correlations: Dict[str, Optional[float]] = {
        pk: correlate_yearly(bt.daily_returns, exp1220) for pk, bt in pair_results.items()
    }
    correlations["PORTFOLIO"] = correlate_yearly(portfolio.daily_returns, exp1220)

    html = render_html(pair_results, portfolio, correlations, exp1220)
    out_html = os.path.join(REPORT_DIR, "exp1770_commodity_calendars.html")
    with open(out_html, "w") as f:
        f.write(html)
    print(f"[exp1770] wrote {out_html}")

    out_json = os.path.join(REPORT_DIR, "exp1770_commodity_calendars.json")
    summary = {
        "experiment": "EXP-1770",
        "description": "Commodity calendar spreads — futures roll-yield harvest",
        "data_sources": {
            "etfs": ["USO", "UNG", "GLD", "SLV"],
            "futures": ["CL=F", "NG=F", "GC=F", "SI=F"],
            "vendor": "Yahoo Finance",
        },
        "pairs": {
            pk: {
                "display": bt.display,
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
    print(f"[exp1770] wrote {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())


# ═══════════════════════════════════════════════════════════════════════════
# EXP-2690 — Production signal entry point (GLD + SLV calendar spreads)
# ═══════════════════════════════════════════════════════════════════════════
def generate_today_signals(date):
    """Paper-trading scheduler entry point. Returns signals for BOTH
    GLD and SLV futures-roll harvest sleeves."""
    from compass.exp2690_signal_generators import gld_cal_signals, slv_cal_signals
    return list(gld_cal_signals(date)) + list(slv_cal_signals(date))
