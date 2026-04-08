"""
EXP-2650 — AUM Capacity via Multi-Expiry Staggering.

The $50M AUM bottleneck on the SPY credit-spread sleeve comes from
trading ONE expiration per entry. If instead we stagger entries across
2-3 different expirations concurrently (Monday hits one expiry,
Wednesday hits another, Friday a third), each expiry has its own
independent liquidity pool and the effective per-entry capacity
multiplies.

This experiment measures the real multi-expiry capacity of SPY puts
from IronVault using:

  * DAILY VOLUME as the capacity proxy (the canonical market-impact
    denominator). Open interest would be a cleaner measure of
    standing liquidity, BUT IronVault's option_daily.open_interest
    column is 100% NULL for SPY puts (2.4M bars, 0 populated OI).
    Volume is the honest fallback — and is actually the RIGHT metric
    for "how much can I trade today without moving the market",
    which is the question the task asks.

Methodology:

  1. For every Monday (canonical entry day for EXP-1220) in 2020-2025,
     query option_daily for all SPY put expirations with 21-45 DTE on
     that date (matches the EXP-1220 28-DTE target ± 2 weeks).
  2. For each expiration, identify the approximate short strike at
     5% OTM from the SPY spot close, look up its daily volume, and
     record it.
  3. Compute per-date capacity at participation rate p = 10% for:
        single      best single expiry (max over all available)
        top2        sum of the two best
        top3        sum of the top three
        topN        sum across all available (usually 3-5)
  4. Convert contracts-per-entry → AUM capacity assuming the
     EXP-1220 sizing rule (2% per-trade risk × $AUM / $500 max-loss
     per contract = $AUM / $25,000 contracts per trade). Solve for
     max AUM at which contracts_needed ≤ p × sum_volume.
  5. Report at $50M / $100M / $200M book sizes: what fraction of
     trading days is each book size feasible, by strategy variant.

Real data — every volume, strike, and expiration comes from
IronVault option_contracts JOIN option_daily. Rule Zero clean.

Outputs:
  compass/exp2650_multi_expiry_capacity.py            (this file)
  compass/reports/exp2650_multi_expiry_capacity.json
  compass/reports/exp2650_multi_expiry_capacity.html

Tag: EXP-2650
Run: python3 -m compass.exp2650_multi_expiry_capacity
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

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2650_multi_expiry_capacity.json"
REPORT_HTML = REPORT_DIR / "exp2650_multi_expiry_capacity.html"
DB_PATH = ROOT / "data" / "options_cache.db"

START = "2020-01-01"
END = "2025-12-31"

MIN_DTE = 21
MAX_DTE = 45
TARGET_DTE = 28
OTM_PCT = 0.05        # 5% OTM short strike (EXP-1220 canonical)
PARTICIPATION_RATE = 0.10  # 10% of daily volume — market-impact safe
RISK_PER_TRADE = 0.02       # 2% of capital per trade (EXP-1220 canonical)
MAX_LOSS_PER_CONTRACT_USD = 500.0  # $5 spread − $0.50 credit proxy × 100

BOOK_SIZES_USD = [50_000_000, 100_000_000, 200_000_000, 500_000_000,
                  1_000_000_000]


# ── Data helpers ──────────────────────────────────────────────────────


def fetch_spy_daily(con: sqlite3.Connection) -> pd.Series:
    """Real SPY spot from Yahoo."""
    import yfinance as yf
    df = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        raise RuntimeError("SPY empty")
    s = df["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s.dropna()


def oi_coverage_audit(con: sqlite3.Connection) -> Dict[str, int]:
    total = con.execute(
        "SELECT COUNT(*) FROM option_daily d "
        "JOIN option_contracts c USING(contract_symbol) "
        "WHERE c.ticker='SPY' AND c.option_type='P'"
    ).fetchone()[0]
    nn = con.execute(
        "SELECT COUNT(*) FROM option_daily d "
        "JOIN option_contracts c USING(contract_symbol) "
        "WHERE c.ticker='SPY' AND c.option_type='P' "
        "AND d.open_interest IS NOT NULL"
    ).fetchone()[0]
    pos = con.execute(
        "SELECT COUNT(*) FROM option_daily d "
        "JOIN option_contracts c USING(contract_symbol) "
        "WHERE c.ticker='SPY' AND c.option_type='P' "
        "AND d.open_interest > 0"
    ).fetchone()[0]
    return {"total_put_bars": total, "oi_non_null": nn, "oi_positive": pos}


def expirations_on_date(con: sqlite3.Connection, date: str) -> List[Tuple[str, int]]:
    """Return (expiration, DTE) for SPY puts on `date` with DTE in range."""
    rows = con.execute("""
        SELECT DISTINCT c.expiration
        FROM option_daily d
        JOIN option_contracts c ON d.contract_symbol = c.contract_symbol
        WHERE c.ticker='SPY' AND c.option_type='P'
          AND d.date=? AND d.volume > 0
          AND julianday(c.expiration) - julianday(d.date) BETWEEN ? AND ?
        ORDER BY c.expiration
    """, (date, MIN_DTE, MAX_DTE)).fetchall()
    snap = datetime.strptime(date, "%Y-%m-%d")
    out: List[Tuple[str, int]] = []
    for (e,) in rows:
        try:
            dte = (datetime.strptime(e, "%Y-%m-%d") - snap).days
        except ValueError:
            continue
        out.append((e, dte))
    return out


def short_strike_volume(con: sqlite3.Connection, date: str, exp: str,
                        target_strike: float) -> Optional[Tuple[float, int]]:
    """Return (actual strike, volume) for the put closest to target_strike
    on (date, exp) with volume > 0."""
    row = con.execute("""
        SELECT c.strike, d.volume
        FROM option_daily d
        JOIN option_contracts c ON d.contract_symbol = c.contract_symbol
        WHERE c.ticker='SPY' AND c.option_type='P'
          AND c.expiration=? AND d.date=? AND d.volume > 0
        ORDER BY ABS(c.strike - ?) LIMIT 1
    """, (exp, date, target_strike)).fetchone()
    if row is None:
        return None
    return float(row[0]), int(row[1])


# ── Walker ────────────────────────────────────────────────────────────


def mondays_between(start: str, end: str) -> List[str]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s.weekday() != 0:
        s += pd.Timedelta(days=1)
    out: List[str] = []
    while s <= e:
        out.append(s.strftime("%Y-%m-%d"))
        s += pd.Timedelta(days=7)
    return out


def collect_per_date_capacity(con: sqlite3.Connection,
                              spy: pd.Series) -> pd.DataFrame:
    rows: List[Dict] = []
    candidates = mondays_between(START, END)
    print(f"[exp2650] scanning {len(candidates)} Monday entry candidates …",
          flush=True)

    for snap in candidates:
        # SPY spot on that Monday (forward-fill if holiday)
        try:
            spot = float(spy.loc[:snap].iloc[-1])
        except (KeyError, IndexError):
            continue
        target_strike = round(spot * (1 - OTM_PCT))

        exps = expirations_on_date(con, snap)
        if not exps:
            continue

        expiry_details: List[Dict] = []
        for exp, dte in exps:
            sv = short_strike_volume(con, snap, exp, target_strike)
            if sv is None:
                continue
            strike, vol = sv
            expiry_details.append({
                "expiration": exp,
                "dte": dte,
                "strike": strike,
                "volume": vol,
            })
        if not expiry_details:
            continue

        # Sort descending by volume for the top-N calculations
        sorted_by_vol = sorted(expiry_details, key=lambda r: -r["volume"])
        single = sorted_by_vol[0]["volume"]
        top2 = sum(r["volume"] for r in sorted_by_vol[:2])
        top3 = sum(r["volume"] for r in sorted_by_vol[:3])
        top_all = sum(r["volume"] for r in sorted_by_vol)

        rows.append({
            "date": snap,
            "spot": spot,
            "target_strike": target_strike,
            "n_expiries": len(expiry_details),
            "single_vol": single,
            "top2_vol": top2,
            "top3_vol": top3,
            "top_all_vol": top_all,
            "expiries": expiry_details,
        })

    df = pd.DataFrame(rows)
    print(f"[exp2650] collected {len(df)} usable entry dates")
    return df


# ── Capacity → AUM conversion ────────────────────────────────────────


def contracts_needed_for_aum(aum_usd: float,
                             risk_pct: float = RISK_PER_TRADE,
                             max_loss_per_contract: float = MAX_LOSS_PER_CONTRACT_USD
                             ) -> float:
    return (aum_usd * risk_pct) / max_loss_per_contract


def aum_cap_from_volume(total_volume: int,
                        participation: float = PARTICIPATION_RATE,
                        risk_pct: float = RISK_PER_TRADE,
                        max_loss_per_contract: float = MAX_LOSS_PER_CONTRACT_USD
                        ) -> float:
    contracts_ok = total_volume * participation
    return contracts_ok * max_loss_per_contract / risk_pct


def feasibility_pct(df: pd.DataFrame, col: str, aum_usd: float) -> float:
    needed = contracts_needed_for_aum(aum_usd)
    per_trade_cap = df[col] * PARTICIPATION_RATE
    return float((per_trade_cap >= needed).mean() * 100)


def summarise_variant(df: pd.DataFrame, col: str, label: str) -> Dict:
    vols = df[col].values
    aum_caps = np.array([
        aum_cap_from_volume(v) for v in vols
    ])
    return {
        "label": label,
        "median_volume": float(np.median(vols)),
        "p5_volume": float(np.percentile(vols, 5)),
        "p95_volume": float(np.percentile(vols, 95)),
        "median_aum_cap_usd": float(np.median(aum_caps)),
        "p5_aum_cap_usd": float(np.percentile(aum_caps, 5)),
        "p95_aum_cap_usd": float(np.percentile(aum_caps, 95)),
        "feasibility_pct": {
            f"{int(b/1e6)}M": round(feasibility_pct(df, col, b), 2)
            for b in BOOK_SIZES_USD
        },
    }


# ── HTML ───────────────────────────────────────────────────────────────


def _fmt_usd(x: float) -> str:
    if x >= 1e9: return f"${x/1e9:.2f}B"
    if x >= 1e6: return f"${x/1e6:.1f}M"
    if x >= 1e3: return f"${x/1e3:.0f}k"
    return f"${x:.0f}"


def render_html(payload: Dict) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111}
    h1{border-bottom:3px solid #333360}
    h2{margin-top:2em;color:#333360}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#333360;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#333360}
    .pill.ok{background:#0a7d1f}
    .pill.bad{background:#c0392b}
    .pill.warn{background:#c07a1f}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2650 Multi-Expiry Capacity</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2650 — Multi-Expiry Capacity Scaling</h1>",
        "<p class='muted'>Can we multiply the SPY credit-spread capacity "
        "by trading staggered expiries instead of a single expiration? "
        "Uses real IronVault SPY put volume as the liquidity pool.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault volumes</span> "
        "<span class='pill warn'>OI unavailable — using volume as proxy</span></p>",
    ]

    # OI gap
    h.append("<h2>Open-interest gap audit</h2>")
    oi = payload["oi_audit"]
    h.append(
        "<table><tr><th>Metric</th><th>Count</th><th>% of total</th></tr>"
        f"<tr><td class='l'>Total SPY put daily bars</td>"
        f"<td>{oi['total_put_bars']:,}</td><td>100%</td></tr>"
        f"<tr><td class='l'>Bars with non-NULL open_interest</td>"
        f"<td class='neg'>{oi['oi_non_null']:,}</td>"
        f"<td class='neg'>{oi['oi_non_null']/max(oi['total_put_bars'],1)*100:.1f}%</td></tr>"
        f"<tr><td class='l'>Bars with OI &gt; 0</td>"
        f"<td class='neg'>{oi['oi_positive']:,}</td>"
        f"<td class='neg'>{oi['oi_positive']/max(oi['total_put_bars'],1)*100:.1f}%</td></tr>"
        "</table>"
    )
    h.append("<p class='muted'>IronVault's option_daily.open_interest "
             "is entirely unpopulated for SPY puts. The rest of this "
             "report uses DAILY VOLUME as the liquidity proxy, which "
             "is the canonical denominator for the participation-rate "
             "market-impact model anyway.</p>")

    # Config
    h.append("<h2>Scenario config</h2>")
    h.append("<table>"
             f"<tr><td class='l'>DTE band</td><td>[{MIN_DTE}, {MAX_DTE}]</td></tr>"
             f"<tr><td class='l'>Target short strike</td><td>{OTM_PCT*100:.0f}% OTM</td></tr>"
             f"<tr><td class='l'>Participation rate</td><td>{PARTICIPATION_RATE*100:.0f}%</td></tr>"
             f"<tr><td class='l'>Risk per trade</td><td>{RISK_PER_TRADE*100:.0f}% of capital</td></tr>"
             f"<tr><td class='l'>Max loss per contract</td><td>${MAX_LOSS_PER_CONTRACT_USD:.0f}</td></tr>"
             "</table>")

    # Entry-date distribution
    h.append("<h2>Entry-date distribution of available expiries</h2>")
    dist = payload["expiry_count_distribution"]
    h.append("<table><tr><th># concurrent expiries</th><th>Share of Mondays</th></tr>")
    for k, v in sorted(dist.items(), key=lambda kv: int(kv[0])):
        h.append(
            f"<tr><td>{k}</td><td>{v*100:.1f}%</td></tr>"
        )
    h.append("</table>")

    # Variant capacity summary
    h.append("<h2>Capacity by variant (median / p5 / p95 over all Mondays)</h2>")
    h.append("<table><tr><th>Variant</th>"
             "<th>Median volume</th><th>p5 / p95</th>"
             "<th>Median AUM cap</th><th>p5 AUM cap</th><th>p95 AUM cap</th></tr>")
    for variant, s in payload["variants"].items():
        h.append(
            f"<tr><td class='l'><b>{variant}</b></td>"
            f"<td>{s['median_volume']:,.0f}</td>"
            f"<td>{s['p5_volume']:,.0f} / {s['p95_volume']:,.0f}</td>"
            f"<td><b>{_fmt_usd(s['median_aum_cap_usd'])}</b></td>"
            f"<td class='neg'>{_fmt_usd(s['p5_aum_cap_usd'])}</td>"
            f"<td class='pos'>{_fmt_usd(s['p95_aum_cap_usd'])}</td></tr>"
        )
    h.append("</table>")

    # Feasibility per book size
    h.append("<h2>Feasibility rate — % of Mondays where the book size is tradable</h2>")
    h.append("<table><tr><th>Variant</th>" +
             "".join(f"<th>${int(b/1e6)}M</th>" for b in BOOK_SIZES_USD) +
             "</tr>")
    for variant, s in payload["variants"].items():
        h.append(f"<tr><td class='l'><b>{variant}</b></td>")
        for b in BOOK_SIZES_USD:
            key = f"{int(b/1e6)}M"
            val = s["feasibility_pct"].get(key, 0.0)
            cls = "pos" if val > 95 else ("warn" if val > 80 else "neg")
            h.append(f"<td><span class='pill {cls}'>{val:.0f}%</span></td>")
        h.append("</tr>")
    h.append("</table>")
    h.append("<p class='muted'>Feasible = contracts needed to size a "
             "trade at 2% of AUM fit inside 10% of the variant's total "
             "available daily volume. 95%+ is \"always tradable\", "
             "80-95% is \"occasional forced downsize\", &lt; 80% is "
             "\"not a real capacity\".</p>")

    # Key finding
    h.append("<h2>Key finding</h2>")
    h.append(payload["headline_html"])

    # Methodology
    h.append("<h2>Methodology &amp; caveats</h2>")
    h.append("<ul>")
    h.append("<li><b>Volume as a capacity proxy.</b> IronVault's "
             "option_daily.open_interest is 100% NULL for SPY puts. "
             "The participation-rate market-impact model uses daily "
             "volume as its denominator anyway, so this is the right "
             "metric for 'how big a trade can I do today without "
             "moving the book'.</li>")
    h.append(f"<li><b>Entry cadence assumption.</b> Every Monday between "
             f"{START} and {END}, I look up SPY puts with DTE in "
             f"[{MIN_DTE}, {MAX_DTE}] — matches the EXP-1220 "
             f"28-DTE target ± 2 weeks. For each expiry I find the "
             f"strike closest to {OTM_PCT*100:.0f}% OTM of the real "
             f"Yahoo spot and read that strike's daily volume from "
             f"IronVault.</li>")
    h.append("<li><b>Multi-expiry staggering.</b> The 'top2' and "
             "'top3' variants SUM the daily volumes across the 2 or 3 "
             "expiries with the highest-volume short strikes. The "
             "assumption is that each expiry is an independent liquidity "
             "pool — which is correct for different-dated contracts — "
             "and that we can allocate trades proportionally across "
             "them. 'top_all' sums every expiry in the DTE window, "
             "usually 3-5 per Monday.</li>")
    h.append(f"<li><b>AUM conversion.</b> contracts_needed = "
             f"{RISK_PER_TRADE*100:.0f}% × AUM / ${MAX_LOSS_PER_CONTRACT_USD:.0f} "
             f"per contract. A trade is feasible if contracts_needed ≤ "
             f"{PARTICIPATION_RATE*100:.0f}% × sum_volume across the "
             f"chosen expiries.</li>")
    h.append("<li><b>Honest limits of this model.</b> Volume can vary "
             "day-to-day by 5-10×; a single low-volume Monday can "
             "force a trade delay on a capacity-constrained book. The "
             "p5 AUM cap column is the relevant number for "
             "worst-case capacity planning. The median is a fair "
             "number for average-case planning.</li>")
    h.append("<li><b>What this does NOT model.</b> (a) market-impact "
             "non-linearity (10% participation is a rough cap — real "
             "impact cost scales super-linearly above that), (b) "
             "portfolio-margin haircuts that reduce effective AUM, "
             "(c) adverse selection when trading across multiple "
             "expiries simultaneously from a single account, (d) "
             "broker / exchange per-account order-size limits.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))

    try:
        print("[exp2650] OI coverage audit …", flush=True)
        oi_audit = oi_coverage_audit(con)
        print(f"[exp2650] OI audit: {oi_audit}")

        print("[exp2650] loading SPY spot (Yahoo) …", flush=True)
        spy = fetch_spy_daily(con)

        df = collect_per_date_capacity(con, spy)
        if df.empty:
            print("[exp2650] no usable dates — aborting")
            return 1

        # Distribution of concurrent expiry counts
        dist_counts = df["n_expiries"].value_counts(normalize=True).to_dict()
        dist_counts = {str(int(k)): float(v) for k, v in dist_counts.items()}

        variants = {
            "single_expiry": summarise_variant(df, "single_vol", "single expiry (best)"),
            "top2_expiries": summarise_variant(df, "top2_vol", "top-2 expiries"),
            "top3_expiries": summarise_variant(df, "top3_vol", "top-3 expiries"),
            "all_expiries":  summarise_variant(df, "top_all_vol", "all available (3-5)"),
        }

        print("\n[exp2650] variant medians:")
        for name, s in variants.items():
            print(f"  {name:14s}  median vol {s['median_volume']:>8,.0f}  "
                  f"median AUM cap {_fmt_usd(s['median_aum_cap_usd']):>8s}  "
                  f"feasible@100M {s['feasibility_pct']['100M']}%")

        # Headline
        single_median = variants["single_expiry"]["median_aum_cap_usd"]
        top3_median = variants["top3_expiries"]["median_aum_cap_usd"]
        multiplier = top3_median / max(single_median, 1)

        headline = [f"<ul>"]
        headline.append(
            f"<li><b>Single-expiry median AUM cap: "
            f"{_fmt_usd(single_median)}.</b> At the canonical "
            f"EXP-1220 single-expiration Monday entry, the median "
            f"Monday in 2020-2025 supports a book of "
            f"{_fmt_usd(single_median)} at 10% participation.</li>"
        )
        headline.append(
            f"<li><b>Top-3 staggered median AUM cap: "
            f"{_fmt_usd(top3_median)}.</b> Spreading entries across the "
            f"three highest-volume expiries in the 21-45 DTE window "
            f"multiplies median capacity by <b>{multiplier:.1f}×</b>.</li>"
        )
        feas_100 = variants["top3_expiries"]["feasibility_pct"]["100M"]
        feas_200 = variants["top3_expiries"]["feasibility_pct"]["200M"]
        feas_500 = variants["top3_expiries"]["feasibility_pct"]["500M"]
        headline.append(
            f"<li><b>Feasibility with top-3 staggering:</b> "
            f"$100M is tradable on {feas_100:.0f}% of Mondays, "
            f"$200M on {feas_200:.0f}%, $500M on {feas_500:.0f}%. "
            f"A production staggered strategy could comfortably run a "
            f"${int(round(variants['top3_expiries']['p5_aum_cap_usd']/1e6)):,}M "
            f"book with almost no forced-downsize days (the p5-Monday "
            f"capacity is the binding constraint).</li>"
        )
        headline.append(
            f"<li><b>AUM ceiling for 'always-tradable' (95%+ feasibility):</b> "
        )
        for name, s in variants.items():
            # find the largest book size that still has ≥95% feasibility
            best = None
            for b in BOOK_SIZES_USD:
                key = f"{int(b/1e6)}M"
                if s["feasibility_pct"].get(key, 0) >= 95:
                    best = b
            headline[-1] += (f"<b>{name}</b>={_fmt_usd(best) if best else '< $50M'}; ")
        headline[-1] += "</li>"
        headline.append("</ul>")

        payload = {
            "experiment": "EXP-2650",
            "tag": "EXP-2650",
            "description": "Multi-expiry staggering capacity analysis on real SPY put volumes",
            "data_sources": {
                "spot": "Yahoo Finance SPY daily close",
                "option_volumes": "IronVault data/options_cache.db option_daily.volume",
                "oi_status": "NULL for all SPY puts — documented gap",
            },
            "config": {
                "min_dte": MIN_DTE, "max_dte": MAX_DTE, "target_dte": TARGET_DTE,
                "otm_pct": OTM_PCT,
                "participation_rate": PARTICIPATION_RATE,
                "risk_per_trade": RISK_PER_TRADE,
                "max_loss_per_contract_usd": MAX_LOSS_PER_CONTRACT_USD,
                "book_sizes_usd": BOOK_SIZES_USD,
            },
            "oi_audit": oi_audit,
            "n_mondays_scanned": int(len(df)),
            "expiry_count_distribution": dist_counts,
            "variants": variants,
            "headline_html": "".join(headline),
        }

        html = render_html(payload)
        REPORT_HTML.write_text(html)
        print(f"\n[exp2650] wrote {REPORT_HTML}")

        REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[exp2650] wrote {REPORT_JSON}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
