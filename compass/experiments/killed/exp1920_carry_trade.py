"""
EXP-1920 — Carry Trade / Interest-Rate-Differential Strategy
=============================================================

Hypothesis
----------
Systematic carry across currency- and commodity-tracking ETFs captures
the interest-rate-differential / roll-yield risk premium that the FX
carry-trade literature (Lustig & Verdelhan 2007, Koijen et al. 2018)
documents in spot FX and futures markets.

Universe (REAL Yahoo Finance only — Rule Zero)
----------------------------------------------
FX ETFs (carry signal = trailing-12m dividend yield, which is exactly
the interest income these grantor trusts pay through):
  FXA  Australian dollar       — usually positive carry
  FXC  Canadian dollar         — moderate
  FXE  Euro                    — varies by ECB cycle
  FXY  Japanese yen            — historically near zero  → funder
  FXB  British pound           — moderate
  UUP  US dollar bullish       — short-rate dependent

Commodity ETFs (carry signal = front-vs-deferred term-structure proxy
via trailing 3m return minus trailing 12m return; this is *roll yield*
in disguise — futures in backwardation roll positively, in contango
negatively):
  DBC  Broad commodity basket
  GLD  Gold
  USO  WTI oil
  UNG  Natural gas
  DBA  Agriculture basket
  SLV  Silver

Strategy
--------
Monthly rebalance, long top tercile by carry, short bottom tercile,
equal-weight within sleeve. FX and commodity sleeves are computed
separately and then averaged 50/50 so neither dominates.

Reports
-------
compass/reports/exp1920_carry_trade.json
compass/reports/exp1920_carry_trade.html
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_JSON = ROOT / "compass" / "reports" / "exp1920_carry_trade.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp1920_carry_trade.html"

FX_UNIVERSE        = ["FXA", "FXC", "FXE", "FXY", "FXB", "UUP"]
COMMODITY_UNIVERSE = ["DBC", "GLD", "USO", "UNG", "DBA", "SLV"]

START = "2019-12-01"   # need 12m warmup before first signal
END   = "2026-01-01"
TRADING_DAYS = 252


# ─────────────────────────────────────────────────────────────────────────────
# Data — REAL Yahoo Finance only
# ─────────────────────────────────────────────────────────────────────────────
def download_panel(symbols: List[str], start: str, end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (price_df, total_return_df). Both daily, indexed by date,
    columns = symbols. Uses Yahoo auto_adjust=False so we can decompose
    dividend income from price return."""
    import yfinance as yf
    raw = yf.download(symbols, start=start, end=end, progress=False,
                      auto_adjust=False, group_by="ticker")
    price_cols, tr_cols = {}, {}
    for sym in symbols:
        if sym in raw.columns.get_level_values(0):
            df = raw[sym]
            if "Close" in df.columns:
                price_cols[sym] = df["Close"]
            if "Adj Close" in df.columns:
                tr_cols[sym] = df["Adj Close"]
    price = pd.DataFrame(price_cols).dropna(how="all")
    tr    = pd.DataFrame(tr_cols).dropna(how="all")
    price.index = pd.to_datetime(price.index).normalize()
    tr.index    = pd.to_datetime(tr.index).normalize()
    return price, tr


# ─────────────────────────────────────────────────────────────────────────────
# Carry signals
# ─────────────────────────────────────────────────────────────────────────────
def fx_carry(price: pd.DataFrame, tr: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    """Trailing-12m dividend yield ≡ trailing total return − trailing price
    return. For grantor-trust FX ETFs this is exactly the interest income
    they pay out (= the carry). Bounded yields (no nonsense from data gaps)."""
    pr = price.pct_change(lookback_days)
    trr = tr.pct_change(lookback_days)
    yld = (trr - pr)
    yld = yld.clip(lower=-0.50, upper=0.50)
    return yld


def commodity_carry(tr: pd.DataFrame) -> pd.DataFrame:
    """Roll-yield proxy: trailing 3m total return − trailing 12m total return
    (annualised difference). Backwardated commodities (positive roll) tend
    to outperform contangoed ones — see Erb & Harvey 2006."""
    r3  = tr.pct_change(63)  * (252.0 / 63.0)
    r12 = tr.pct_change(252)
    return (r3 - r12).clip(lower=-2.0, upper=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio: monthly long top tercile / short bottom tercile, equal weight
# ─────────────────────────────────────────────────────────────────────────────
def long_short_weights(signal: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Per-row, set the top-k longs to +1/k and bottom-k shorts to -1/k."""
    w = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for dt, row in signal.iterrows():
        valid = row.dropna()
        if len(valid) < (2 * k):
            continue
        ranked = valid.sort_values()
        shorts = ranked.index[:k]
        longs  = ranked.index[-k:]
        w.loc[dt, longs]  = 1.0 / k
        w.loc[dt, shorts] = -1.0 / k
    return w


def monthly_rebalance(weights: pd.DataFrame) -> pd.DataFrame:
    """Hold each month's-end weight constant for the next month."""
    monthly = weights.resample("ME").last().shift(1)   # decision lag: act next month
    return monthly.reindex(weights.index, method="ffill").fillna(0.0)


def sleeve_returns(tr: pd.DataFrame, w_daily: pd.DataFrame) -> pd.Series:
    """Daily portfolio return = sum(weight_i × daily TR_i)."""
    rets = tr.pct_change().reindex(w_daily.index).fillna(0.0)
    common = w_daily.columns.intersection(rets.columns)
    return (w_daily[common] * rets[common]).sum(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(daily: pd.Series, label: str) -> Dict:
    daily = daily.dropna()
    if len(daily) < 30:
        return {"label": label, "n_days": 0, "cagr_pct": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "max_dd_pct": 0.0,
                "vol_pct": 0.0, "calmar": 0.0}
    eq = (1 + daily).cumprod()
    yrs = len(daily) / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    mu, sd = daily.mean(), daily.std(ddof=1)
    downside = daily[daily < 0].std(ddof=1) if (daily < 0).any() else np.nan
    sharpe = float((mu / sd) * math.sqrt(TRADING_DAYS)) if sd > 1e-12 else 0.0
    sortino = float((mu / downside) * math.sqrt(TRADING_DAYS)) if downside and downside > 1e-12 else 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(-dd.min())
    return {
        "label": label,
        "n_days": int(len(daily)),
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 3),
        "vol_pct": round(float(sd) * math.sqrt(TRADING_DAYS) * 100, 3),
        "calmar": round(cagr / max_dd, 3) if max_dd > 1e-9 else 0.0,
    }


def walk_forward_yearly(daily: pd.Series) -> List[Dict]:
    out = []
    for y, sub in daily.groupby(daily.index.year):
        m = metrics(sub, str(y))
        m["year"] = int(y)
        out.append(m)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Correlation to EXP-1220
# ─────────────────────────────────────────────────────────────────────────────
def exp1220_daily_returns() -> pd.Series:
    """Build a daily-return series from the EXP-1220 trade tape (same engine,
    real IronVault SPY chains). On non-trade days returns are 0; on exit days
    we book trade pnl / 100k."""
    from compass.exp1220_standalone import run_exp1220_trades
    from shared.iron_vault import IronVault
    import yfinance as yf
    hd = IronVault.instance()
    spy = yf.download("SPY", start="2019-06-01", end="2026-07-01", progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).normalize()
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01", progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).normalize()
    trades = run_exp1220_trades(hd, spy, vix)
    s = pd.Series(0.0, index=spy.index)
    for t in trades:
        ed = pd.Timestamp(t["exit_date"]).normalize()
        if ed in s.index:
            s.loc[ed] += t["pnl"] / 100_000
    return s


def monthly_correlation(a: pd.Series, b: pd.Series) -> Dict:
    am = (1 + a).resample("ME").apply(lambda x: x.prod() - 1)
    bm = (1 + b).resample("ME").apply(lambda x: x.prod() - 1)
    common = am.index.intersection(bm.index)
    if len(common) < 6:
        return {"n_months": int(len(common)), "pearson": None, "spearman": None}
    a_c = am.loc[common]; b_c = bm.loc[common]
    return {
        "n_months": int(len(common)),
        "pearson":  round(float(a_c.corr(b_c, method="pearson")), 3),
        "spearman": round(float(a_c.corr(b_c, method="spearman")), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[1/5] downloading FX universe …")
    fx_price, fx_tr = download_panel(FX_UNIVERSE, START, END)
    print(f"      {fx_tr.shape[1]} symbols, {len(fx_tr)} rows")

    print("[2/5] downloading commodity universe …")
    cm_price, cm_tr = download_panel(COMMODITY_UNIVERSE, START, END)
    print(f"      {cm_tr.shape[1]} symbols, {len(cm_tr)} rows")

    print("[3/5] computing carry signals …")
    fx_signal = fx_carry(fx_price, fx_tr)
    cm_signal = commodity_carry(cm_tr)

    fx_w = monthly_rebalance(long_short_weights(fx_signal, k=2))
    cm_w = monthly_rebalance(long_short_weights(cm_signal, k=2))

    fx_ret = sleeve_returns(fx_tr, fx_w)
    cm_ret = sleeve_returns(cm_tr, cm_w)

    # Combined sleeve: 50/50 blend
    common = fx_ret.index.intersection(cm_ret.index)
    combined = 0.5 * fx_ret.reindex(common).fillna(0.0) + 0.5 * cm_ret.reindex(common).fillna(0.0)
    # restrict to live trading window (drop the warmup year)
    combined = combined.loc["2021-01-01":]
    fx_ret   = fx_ret.loc["2021-01-01":]
    cm_ret   = cm_ret.loc["2021-01-01":]

    print("[4/5] correlation to EXP-1220 …")
    try:
        e1220 = exp1220_daily_returns().loc["2021-01-01":]
        corr_combined = monthly_correlation(combined, e1220)
        corr_fx       = monthly_correlation(fx_ret,   e1220)
        corr_cm       = monthly_correlation(cm_ret,   e1220)
    except Exception as e:
        print("      EXP-1220 corr failed:", e)
        corr_combined = corr_fx = corr_cm = {"n_months": 0, "pearson": None, "spearman": None}

    print("[5/5] writing report …")
    payload = {
        "experiment": "EXP-1920",
        "name": "Carry Trade / Interest-Rate-Differential ETF Strategy",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data_source": "Yahoo Finance (auto_adjust=False) — NO synthetic data",
        "universe": {
            "fx":        FX_UNIVERSE,
            "commodity": COMMODITY_UNIVERSE,
        },
        "signals": {
            "fx":        "trailing-12m dividend yield (TR-return − price-return)",
            "commodity": "3m − 12m TR slope (roll-yield proxy, Erb & Harvey 2006 style)",
        },
        "rebalance":     "monthly, long top 2 / short bottom 2 each sleeve, 50/50 blend",
        "live_window":   ["2021-01-01", str(combined.index[-1].date()) if len(combined) else None],
        "fx_sleeve":        metrics(fx_ret,   "fx_sleeve"),
        "commodity_sleeve": metrics(cm_ret,   "commodity_sleeve"),
        "combined":         metrics(combined, "combined_50_50"),
        "walk_forward_combined": walk_forward_yearly(combined),
        "correlation_to_exp1220": {
            "combined":  corr_combined,
            "fx":        corr_fx,
            "commodity": corr_cm,
        },
        "target_sharpe": 2.0,
        "target_met":    metrics(combined, "")["sharpe"] >= 2.0,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("wrote", REPORT_JSON)
    print("wrote", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    rows = "".join(
        f"<tr><td>{k}</td><td>{p[k]['n_days']}</td><td>{p[k]['cagr_pct']:.2f}%</td>"
        f"<td>{p[k]['sharpe']:.2f}</td><td>{p[k]['sortino']:.2f}</td>"
        f"<td>{p[k]['max_dd_pct']:.2f}%</td><td>{p[k]['vol_pct']:.2f}%</td>"
        f"<td>{p[k]['calmar']:.2f}</td></tr>"
        for k in ("fx_sleeve", "commodity_sleeve", "combined")
    )
    rows_w = "".join(
        f"<tr><td>{r['year']}</td><td>{r['cagr_pct']:.2f}%</td><td>{r['sharpe']:.2f}</td>"
        f"<td>{r['max_dd_pct']:.2f}%</td></tr>"
        for r in p["walk_forward_combined"]
    )
    cc = p["correlation_to_exp1220"]["combined"]
    fc = p["correlation_to_exp1220"]["fx"]
    mc = p["correlation_to_exp1220"]["commodity"]
    target_cls = "ok" if p["target_met"] else "warn"
    target_txt = "MET" if p["target_met"] else "NOT MET"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-1920 — Carry Trade ETF Strategy</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .warn{{color:#b86b00;font-weight:600}} .bad{{color:#b80000;font-weight:600}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-1920 — Carry Trade / Rate-Differential ETF Strategy</h1>
<p class='small'>Generated {p['generated']} · Universe: 6 FX + 6 commodity ETFs ·
   Real Yahoo Finance only · Live window: {p['live_window'][0]} → {p['live_window'][1]}</p>

<h2>Sleeve performance</h2>
<table>
<tr><th>Sleeve</th><th>Days</th><th>CAGR</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Vol</th><th>Calmar</th></tr>
{rows}
</table>
<p>Target Sharpe ≥ 2.0: <span class='{target_cls}'>{target_txt}</span></p>

<h2>Walk-forward (combined sleeve, by year)</h2>
<table>
<tr><th>Year</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr>
{rows_w}
</table>

<h2>Correlation to EXP-1220 credit spreads</h2>
<table>
<tr><th>Sleeve</th><th>n months</th><th>Pearson</th><th>Spearman</th></tr>
<tr><td>combined</td><td>{cc['n_months']}</td><td>{cc['pearson']}</td><td>{cc['spearman']}</td></tr>
<tr><td>fx</td><td>{fc['n_months']}</td><td>{fc['pearson']}</td><td>{fc['spearman']}</td></tr>
<tr><td>commodity</td><td>{mc['n_months']}</td><td>{mc['pearson']}</td><td>{mc['spearman']}</td></tr>
</table>

<h2>Method notes</h2>
<ul>
<li>FX carry signal: trailing 12m total-return minus trailing 12m price-return — that residual is the dividend distribution, which for grantor-trust FX ETFs <i>is</i> the interest income paid through from the underlying currency deposit.</li>
<li>Commodity carry signal: 3m TR annualised − 12m TR. Erb & Harvey 2006 style roll-yield proxy.</li>
<li>Monthly rebalance, long top-2 / short bottom-2 in each sleeve, equal weight, 50/50 blend across sleeves. Decision-to-execution lag of one month avoids look-ahead.</li>
<li>EXP-1220 daily-return series built from the same trade engine on real IronVault SPY chains.</li>
<li>All data Yahoo Finance with <code>auto_adjust=False</code>; no synthetic prices anywhere.</li>
</ul>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
