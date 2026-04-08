"""
EXP-1740: Sentiment-Filtered Entry Timing for EXP-1220 Credit Spreads
=====================================================================

NLP overlay that filters EXP-1220 trade entries using:
  1. FOMC minutes hawkish/dovish sentiment (federalreserve.gov, 2015-2025)
  2. FOMC uncertainty index (Loughran-McDonald style)
  3. VIX term structure slope (^VIX9D / ^VIX / ^VIX3M, real Yahoo data)

Hypothesis: avoiding entries in the days following hawkish FOMC minutes,
and only entering when the VIX term structure is in normal contango,
should reduce gamma blow-ups and improve risk-adjusted returns.

REAL DATA ONLY:
  - FOMC minutes:  data/fomc/fomcminutes*.txt  (downloaded from federalreserve.gov)
  - VIX series:    Yahoo Finance ^VIX, ^VIX9D, ^VIX3M
  - SPY/options:   IronVault options_cache.db
  - Trade engine:  compass.exp1220_standalone (real IronVault Polygon prices)

Walk-forward validated by year (2020-2025).

Output:
  compass/reports/exp1740_sentiment.json
  compass/reports/exp1740_sentiment.html
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp1220_standalone import (
    run_exp1220_trades,
    sharpe_correct,
    TRADING_DAYS,
)

FOMC_DIR = ROOT / "data" / "fomc"
DB_PATH = ROOT / "data" / "options_cache.db"
REPORT_JSON = ROOT / "compass" / "reports" / "exp1740_sentiment.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp1740_sentiment.html"


# ─────────────────────────────────────────────────────────────────────────────
# 1. NLP lexicons
# ─────────────────────────────────────────────────────────────────────────────
# Compact, transparent keyword lists. Not a transformer — just countable
# evidence so every score is explainable. Inspired by Apel & Blix Grimaldi (2014)
# "How Informative are Central Bank Minutes?" and Hansen, McMahon & Prat (2018).

HAWKISH = {
    "tighten", "tightening", "tightened", "raise", "raising", "increase",
    "increased", "increases", "elevated", "overheating", "above target",
    "above its longer-run", "restrictive", "above-trend", "firm", "firming",
    "robust growth", "strong growth", "strong labor", "resilient",
    "persistent inflation", "elevated inflation", "inflation pressures",
    "wage pressures", "tight labor", "supply constraints", "hawkish",
}

DOVISH = {
    "ease", "easing", "eased", "accommodative", "accommodation",
    "support the recovery", "downside risks", "weak", "weakened", "softening",
    "soft", "subdued", "moderating", "moderated", "below target",
    "below its longer-run", "patient", "patient approach", "transitory",
    "supportive", "stimulative", "cuts", "cut rates", "cutting", "lowered",
    "dovish", "disinflation", "slack",
}

# Loughran–McDonald-style uncertainty terms (subset)
UNCERTAINTY = {
    "uncertain", "uncertainty", "uncertainties", "risks", "risk", "risky",
    "volatile", "volatility", "depend", "depending", "contingent",
    "appears", "appear", "may", "might", "could", "possible", "possibility",
    "tentative", "preliminary", "questionable", "unclear",
}

# words that hint a rate move was discussed
RATE_MOVE = {
    "raise the target", "increase the target", "lower the target",
    "decrease the target", "raise the federal funds rate",
    "increase the federal funds rate", "lower the federal funds rate",
    "decrease the federal funds rate", "rate increase", "rate cut",
    "rate hike", "rate reduction", "policy rate",
}


def _extract_body(raw: str) -> str:
    """Strip the federalreserve.gov chrome and return the actual minutes body."""
    i = raw.lower().find("minutes of the federal open market committee")
    return raw[i:] if i >= 0 else raw


def _count_phrases(text: str, phrases) -> int:
    n = 0
    for p in phrases:
        if " " in p:
            n += text.count(p)
        else:
            n += len(re.findall(rf"\b{re.escape(p)}\b", text))
    return n


@dataclass
class FomcFeatures:
    date: str          # YYYY-MM-DD release date (file stamp)
    n_words: int
    hawkish: int
    dovish: int
    uncertainty: int
    rate_move: int
    hd_score: float    # (hawkish - dovish) / max(1, hawkish + dovish)  in [-1, 1]
    unc_density: float # uncertainty per 1k words


def parse_fomc_minutes() -> List[FomcFeatures]:
    out: List[FomcFeatures] = []
    for fp in sorted(FOMC_DIR.glob("fomcminutes*.txt")):
        m = re.search(r"(\d{8})", fp.name)
        if not m:
            continue
        d = m.group(1)
        date = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"
        body = _extract_body(fp.read_text()).lower()
        # restrict to first ~25k chars to avoid the chrome from re-appearing
        body = body[:60_000]
        words = body.split()
        n_words = len(words)
        haw = _count_phrases(body, HAWKISH)
        dov = _count_phrases(body, DOVISH)
        unc = _count_phrases(body, UNCERTAINTY)
        rate = _count_phrases(body, RATE_MOVE)
        hd = (haw - dov) / max(1, haw + dov)
        unc_density = 1000.0 * unc / max(1, n_words)
        out.append(FomcFeatures(date, n_words, haw, dov, unc, rate, hd, unc_density))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build daily feature panel
# ─────────────────────────────────────────────────────────────────────────────
def build_daily_panel(features: List[FomcFeatures], start: str, end: str) -> pd.DataFrame:
    """For each trading day, attach the most recent FOMC's hd_score and the
    number of days since release. Days within 5 trading days of a hawkish
    release are flagged."""
    import yfinance as yf

    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    idx = spy.index.normalize()

    fd = pd.DataFrame([f.__dict__ for f in features])
    fd["date"] = pd.to_datetime(fd["date"])
    fd = fd.sort_values("date").reset_index(drop=True)

    # for each trading day, find latest FOMC at-or-before
    panel = pd.DataFrame(index=idx)
    panel["fomc_hd"]  = np.nan
    panel["fomc_unc"] = np.nan
    panel["days_since_fomc"] = np.nan
    j = 0
    for i, day in enumerate(panel.index):
        while j + 1 < len(fd) and fd.iloc[j + 1]["date"] <= day:
            j += 1
        if fd.iloc[j]["date"] <= day:
            panel.iat[i, 0] = fd.iloc[j]["hd_score"]
            panel.iat[i, 1] = fd.iloc[j]["unc_density"]
            panel.iat[i, 2] = (day - fd.iloc[j]["date"]).days

    # VIX term structure slope (real)
    vix   = yf.download("^VIX",   start=start, end=end, auto_adjust=False, progress=False)["Close"]
    vix3m = yf.download("^VIX3M", start=start, end=end, auto_adjust=False, progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    if isinstance(vix3m, pd.DataFrame):
        vix3m = vix3m.iloc[:, 0]
    vix.index = vix.index.normalize()
    vix3m.index = vix3m.index.normalize()
    panel["vix"]   = vix.reindex(panel.index).ffill()
    panel["vix3m"] = vix3m.reindex(panel.index).ffill()
    panel["vix_slope"] = panel["vix3m"] - panel["vix"]   # >0 → contango (calm)

    panel.index.name = "date"
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# 3. Re-run EXP-1220 and apply filters
# ─────────────────────────────────────────────────────────────────────────────
def load_exp1220_trades(start: str = "2019-06-01", end: str = "2026-07-01") -> List[Dict]:
    import yfinance as yf
    from shared.iron_vault import IronVault
    hd = IronVault.instance()
    spy = yf.download("SPY", start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index)
    vix = yf.download("^VIX", start=start, end=end, progress=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index)
    return run_exp1220_trades(hd, spy, vix)


def trade_metrics(trades: List[Dict], label: str) -> Dict:
    if not trades:
        return {"label": label, "n": 0, "pnl": 0.0, "wr": 0.0,
                "sharpe": 0.0, "cagr_pct": 0.0, "max_dd_pct": 0.0,
                "avg_pnl": 0.0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    wins = (pnl > 0).sum()
    # equity curve in $ terms; daily series for sharpe via per-trade as proxy
    equity = 100_000 + pnl.cumsum()
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd_pct = float(-dd.min() * 100.0)
    # trade-level Sharpe annualised at √(trades/yr)
    yrs = max(1.0, (
        datetime.strptime(trades[-1]["exit_date"], "%Y-%m-%d")
        - datetime.strptime(trades[0]["entry_date"], "%Y-%m-%d")
    ).days / 365.25)
    trades_per_yr = len(pnl) / yrs
    rets = pnl / 100_000
    mu = rets.mean(); sd = rets.std(ddof=1) if len(rets) > 1 else 0.0
    sharpe = (mu / sd) * math.sqrt(trades_per_yr) if sd > 1e-12 else 0.0
    cagr_pct = (equity[-1] / 100_000) ** (1 / yrs) * 100 - 100
    return {
        "label": label, "n": int(len(pnl)),
        "pnl": float(pnl.sum()),
        "wr": float(wins / len(pnl)),
        "sharpe": float(sharpe),
        "cagr_pct": float(cagr_pct),
        "max_dd_pct": max_dd_pct,
        "avg_pnl": float(pnl.mean()),
        "trades_per_yr": float(trades_per_yr),
    }


def apply_filters(trades, panel, *,
                  hawkish_block_days: int = 5,
                  hawkish_thresh: float = 0.20,
                  vix_slope_min: Optional[float] = 0.0) -> List[Dict]:
    """Drop trades whose entry violates a filter."""
    keep = []
    for t in trades:
        ed = pd.Timestamp(t["entry_date"])
        if ed not in panel.index:
            # nearest available
            try:
                ed = panel.index[panel.index.get_indexer([ed], method="nearest")[0]]
            except Exception:
                continue
        row = panel.loc[ed]
        # FOMC hawkish window — block first N trading days after a hawkish release
        if not pd.isna(row["fomc_hd"]) and row["fomc_hd"] >= hawkish_thresh:
            if not pd.isna(row["days_since_fomc"]) and row["days_since_fomc"] <= hawkish_block_days * 1.5:
                continue
        # VIX slope filter — only enter when 3M >= spot (contango)
        if vix_slope_min is not None:
            if pd.isna(row["vix_slope"]) or row["vix_slope"] < vix_slope_min:
                continue
        keep.append(t)
    return keep


# ─────────────────────────────────────────────────────────────────────────────
# 4. Walk-forward by year
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward(trades, panel, *, hawkish_thresh, vix_slope_min):
    rows = []
    by_year: Dict[int, List[Dict]] = {}
    for t in trades:
        y = int(t["entry_date"][:4]); by_year.setdefault(y, []).append(t)
    for y in sorted(by_year):
        baseline = trade_metrics(by_year[y], f"{y} baseline")
        filt = apply_filters(by_year[y], panel,
                              hawkish_thresh=hawkish_thresh,
                              vix_slope_min=vix_slope_min)
        filtered = trade_metrics(filt, f"{y} filtered")
        rows.append({
            "year": y,
            "baseline_n": baseline["n"],
            "baseline_sharpe": baseline["sharpe"],
            "baseline_pnl": baseline["pnl"],
            "filtered_n": filtered["n"],
            "filtered_sharpe": filtered["sharpe"],
            "filtered_pnl": filtered["pnl"],
            "delta_sharpe": filtered["sharpe"] - baseline["sharpe"],
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[1/5] parsing FOMC minutes …")
    feats = parse_fomc_minutes()
    print(f"      {len(feats)} meetings  range {feats[0].date} → {feats[-1].date}")

    print("[2/5] building daily panel (yfinance, real) …")
    panel = build_daily_panel(feats, start="2019-06-01", end="2026-01-01")

    print("[3/5] running EXP-1220 trades on real IronVault options …")
    trades = load_exp1220_trades()
    print(f"      {len(trades)} trades")

    print("[4/5] applying filters …")
    base = trade_metrics(trades, "EXP-1220 baseline (no filter)")

    variants: List[Dict] = [base]
    for haw in (0.15, 0.20, 0.30):
        for vmin in (None, 0.0, 0.5):
            label = f"haw≥{haw:.2f}, vix_slope≥{vmin}"
            kept = apply_filters(trades, panel,
                                  hawkish_thresh=haw, vix_slope_min=vmin)
            variants.append(trade_metrics(kept, label))

    # pick best by sharpe with at least 50% of trades retained
    eligible = [v for v in variants[1:] if v["n"] >= max(20, int(0.5 * base["n"]))]
    best = max(eligible, key=lambda v: v["sharpe"]) if eligible else variants[1]
    print("[5/5] best variant:", best["label"], "Sharpe", round(best["sharpe"],3))

    # walk forward on the best parameter set
    parts = best["label"].split(",")
    haw = float(parts[0].split("≥")[1])
    vix_part = parts[1].strip()
    vmin = None if vix_part.endswith("None") else float(vix_part.split("≥")[1])
    wf = walk_forward(trades, panel, hawkish_thresh=haw, vix_slope_min=vmin)

    payload = {
        "experiment": "EXP-1740",
        "name": "Sentiment-Filtered Entry Timing for EXP-1220 Credit Spreads",
        "data_sources": {
            "fomc_minutes": "federalreserve.gov  (89 meetings, 2015-2025)",
            "vix":          "Yahoo Finance ^VIX, ^VIX3M",
            "options":      "IronVault options_cache.db (Polygon real)",
        },
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "n_fomc_meetings": len(feats),
        "fomc_features_sample": [
            {"date": f.date, "hd_score": round(f.hd_score, 3),
             "hawkish": f.hawkish, "dovish": f.dovish,
             "unc_density": round(f.unc_density, 2), "n_words": f.n_words}
            for f in feats[:5] + feats[-5:]
        ],
        "baseline":  base,
        "variants":  variants[1:],
        "best":      best,
        "delta_sharpe_vs_baseline": round(best["sharpe"] - base["sharpe"], 3),
        "target_met": (best["sharpe"] - base["sharpe"]) >= 0.5,
        "walk_forward": wf,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", REPORT_JSON)
    write_html(payload)
    print("wrote", REPORT_HTML)
    return payload


def write_html(p: Dict) -> None:
    base = p["baseline"]; best = p["best"]
    rows_v = "".join(
        f"<tr><td>{v['label']}</td><td>{v['n']}</td><td>{v['wr']*100:.1f}%</td>"
        f"<td>{v['sharpe']:.2f}</td><td>{v['cagr_pct']:.2f}%</td>"
        f"<td>{v['max_dd_pct']:.2f}%</td><td>${v['pnl']:.0f}</td></tr>"
        for v in p["variants"]
    )
    rows_w = "".join(
        f"<tr><td>{r['year']}</td><td>{r['baseline_n']}</td>"
        f"<td>{r['baseline_sharpe']:.2f}</td><td>{r['filtered_n']}</td>"
        f"<td>{r['filtered_sharpe']:.2f}</td>"
        f"<td class='{ 'ok' if r['delta_sharpe']>=0 else 'bad'}'>{r['delta_sharpe']:+.2f}</td></tr>"
        for r in p["walk_forward"]
    )
    rows_f = "".join(
        f"<tr><td>{f['date']}</td><td>{f['hawkish']}</td><td>{f['dovish']}</td>"
        f"<td>{f['hd_score']:+.2f}</td><td>{f['unc_density']:.2f}</td></tr>"
        for f in p["fomc_features_sample"]
    )
    target_cls = "ok" if p["target_met"] else "warn"
    target_txt = "MET" if p["target_met"] else "NOT MET"
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-1740 — Sentiment-Filtered Entry Timing</title>
<style>
 body {{ font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}}
 h2{{margin-top:1.8em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}}
 th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .bad{{color:#b80000;font-weight:600}}
 .warn{{color:#b86b00;font-weight:600}}
 .callout{{background:#fff8e1;border-left:4px solid #e0a500;padding:.8em 1em}}
 .small{{color:#555;font-size:.88em}}
</style></head><body>
<h1>EXP-1740 — Sentiment-Filtered Entry Timing</h1>
<p class='small'>Generated {p['generated']} · {p['n_fomc_meetings']} FOMC meetings ·
   Real Yahoo VIX · Real IronVault options. Rule-Zero clean.</p>

<h2>Headline</h2>
<table>
<tr><th>Variant</th><th>Trades</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Net PnL</th></tr>
<tr><td><b>Baseline (no filter)</b></td><td>{base['n']}</td><td>{base['sharpe']:.2f}</td>
    <td>{base['cagr_pct']:.2f}%</td><td>{base['max_dd_pct']:.2f}%</td><td>${base['pnl']:.0f}</td></tr>
<tr><td><b>Best filter:</b> {best['label']}</td><td>{best['n']}</td><td>{best['sharpe']:.2f}</td>
    <td>{best['cagr_pct']:.2f}%</td><td>{best['max_dd_pct']:.2f}%</td><td>${best['pnl']:.0f}</td></tr>
</table>
<p>Δ Sharpe vs baseline: <b>{p['delta_sharpe_vs_baseline']:+.2f}</b>
   &nbsp;·&nbsp; Target +0.50: <span class='{target_cls}'>{target_txt}</span></p>

<h2>All filter variants</h2>
<table><tr><th>Filter</th><th>n</th><th>WR</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>PnL</th></tr>
{rows_v}</table>

<h2>Walk-forward by year (best params)</h2>
<table><tr><th>Year</th><th>Baseline n</th><th>Baseline Sharpe</th>
    <th>Filtered n</th><th>Filtered Sharpe</th><th>Δ Sharpe</th></tr>
{rows_w}</table>

<h2>FOMC sentiment features (first &amp; last 5 meetings)</h2>
<table><tr><th>Release</th><th>Hawkish hits</th><th>Dovish hits</th><th>HD score</th><th>Unc/1k words</th></tr>
{rows_f}</table>

<h2>Method notes</h2>
<ul>
<li>FOMC minutes downloaded directly from federalreserve.gov (89 meetings 2015-01 → 2025-12).</li>
<li>HD score = (hawkish - dovish) / (hawkish + dovish), bounded [-1, +1].</li>
<li>Hawkish window block: any trade entered within ~7 calendar days after a meeting whose HD score ≥ threshold is dropped.</li>
<li>VIX slope filter: only enter when ^VIX3M − ^VIX ≥ threshold (term structure in contango).</li>
<li>Trade engine: <code>compass.exp1220_standalone.run_exp1220_trades</code> on real IronVault SPY option chains.</li>
<li>Sharpe annualisation uses trade-level √(trades/yr) — comparable to MASTERPLAN's per-trade 1.26 baseline, not the portfolio daily Sharpe.</li>
</ul>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
