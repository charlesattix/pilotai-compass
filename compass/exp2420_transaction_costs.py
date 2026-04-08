"""
EXP-2420 — Realistic Transaction Cost Model for the 7-Stream Portfolio

Goal
----
Our backtests have been using simplified (often zero) transaction-cost
assumptions. This experiment builds a component-decomposed cost model
using REAL IronVault data wherever possible, applies it to the 7-stream
North Star v6 portfolio at 3× leverage, and computes the NET Sharpe
after each cost layer.

Cost components
---------------
  1. Bid-ask spread  — measured per underlier from IronVault option_daily.
       Proxy: 25th-percentile (high - low)/close on contracts with
       volume >= 500 and close >= $0.50 since 2024. The p25 is used
       (not median) because it represents days with real trading but
       minimal intraday movement — the cleanest available spread proxy
       when explicit NBBO bid/ask is not in the data.
  2. Commission     — Alpaca $0.65 / contract / leg.
  3. Slippage       — square-root market impact model using real 90-day
       Yahoo ADV for each underlier.
       impact_bps = coeff × √(trade_notional / ADV_notional)

Leverage
--------
3× leverage applied to the $100K base. Trade *notional* scales linearly
with leverage; trade *count* does not. Only the bid-ask and slippage
costs scale with notional per trade.

Streams modelled
----------------
  exp1220   SPY credit spreads    ~34 trades/yr, 2 legs, ~3 contracts at 3x
  xlf_cs    XLF credit spreads    ~34 trades/yr, 2 legs, ~15 contracts at 3x
  xli_cs    XLI credit spreads    ~34 trades/yr, 2 legs, ~5  contracts at 3x
  gld_cal   GLD calendar          ~50 trades/yr, 2 legs, ~7  contracts at 3x
  slv_cal   SLV calendar          ~50 trades/yr, 2 legs, ~30 contracts at 3x
  vol_arb   Cross-vol arb         ~45 trades/yr, 4 legs, ~5  contracts at 3x
  v5_hedge  Crisis Alpha v5       ~20 trades/yr, 1 leg,  ~10 contracts at 3x

Baseline Sharpe: 5.96 (EXP-2200 equal_risk_15% full sample).

Outputs
-------
  compass/exp2420_transaction_costs.py
  compass/reports/exp2420_transaction_costs.json
  compass/reports/exp2420_transaction_costs.html
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_JSON = ROOT / "compass" / "reports" / "exp2420_transaction_costs.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2420_transaction_costs.html"
IV_DB       = ROOT / "data" / "options_cache.db"

# ── Model constants ───────────────────────────────────────────────────────
CAPITAL              = 100_000.0
LEVERAGE             = 3.0
BASELINE_SHARPE      = 5.96      # EXP-2200 equal_risk_15% full-sample
BASELINE_CAGR_PCT    = 146.2
# Derive vol consistent with Sharpe + CAGR:
# daily_mean ≈ ln(1+CAGR)/252; daily_std = daily_mean × √252 / Sharpe
# annual_vol = daily_std × √252
_dm = math.log(1 + BASELINE_CAGR_PCT / 100) / 252
_ds = _dm * math.sqrt(252) / BASELINE_SHARPE
BASELINE_VOL_PCT     = _ds * math.sqrt(252) * 100    # ≈ 15.1%
BASELINE_DAILY_VOL   = _ds

COMMISSION_PER_CONTRACT = 0.65   # Alpaca options tier
SLIPPAGE_COEFF_BPS      = 50.0   # half of EXP-2140 options coeff (realistic at low participation)


# ── Stream characterisation (trades/year, legs, contracts at 3× capital) ──
@dataclass
class StreamParams:
    name: str
    ticker: str              # IronVault ticker for spread lookup
    trades_per_year: float
    legs_per_trade: int
    contracts_per_trade_at_3x: float
    portfolio_weight: float  # equal_risk_15% weights from EXP-2200


# Canonical equal_risk weights from EXP-2200 (already in JSON)
STREAMS = [
    StreamParams("exp1220",  "SPY", 34, 2,  3.0, 0.316),
    StreamParams("xlf_cs",   "XLF", 34, 2, 15.0, 0.245),
    StreamParams("xli_cs",   "XLI", 34, 2,  5.0, 0.192),
    StreamParams("gld_cal",  "GLD", 50, 2,  7.0, 0.024),
    StreamParams("slv_cal",  "SLV", 50, 2, 30.0, 0.012),
    StreamParams("vol_arb",  "SPY", 45, 4,  5.0, 0.187),
    StreamParams("v5_hedge", "SPY", 20, 1, 10.0, 0.023),
]


# ───────────────────────────────────────────────────────────────────────────
# Real data loaders
# ───────────────────────────────────────────────────────────────────────────

def measure_bid_ask_proxy(ticker: str) -> Dict[str, float]:
    """Measure IronVault real bid-ask proxy for a ticker.

    Returns dict with:
      p25_rel         : 25th-percentile (high-low)/close    — spread proxy (%)
      p25_abs         : 25th-percentile (high-low) in $
      median_rel      : median (H-L)/C for context
      n_samples       : number of contract-days used
      mid_price       : median close of the samples (typical contract price)
    """
    if not IV_DB.exists():
        return {}
    conn = sqlite3.connect(str(IV_DB))
    try:
        rows = conn.execute("""
            SELECT od.high, od.low, od.close
            FROM option_daily od
            JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
            WHERE oc.ticker = ?
              AND od.date >= '2024-01-01'
              AND od.volume >= 500
              AND od.close >= 0.50
              AND od.high > od.low
        """, (ticker,)).fetchall()
    finally:
        conn.close()
    if not rows:
        return {"n_samples": 0}
    h = np.array([r[0] for r in rows], dtype=float)
    l = np.array([r[1] for r in rows], dtype=float)
    c = np.array([r[2] for r in rows], dtype=float)
    rel = (h - l) / c
    abs_ = h - l
    return {
        "n_samples":  int(len(rows)),
        "p10_rel":    round(float(np.percentile(rel, 10)), 6),
        "p25_rel":    round(float(np.percentile(rel, 25)), 6),
        "median_rel": round(float(np.median(rel)), 6),
        "p10_abs":    round(float(np.percentile(abs_, 10)), 4),
        "p25_abs":    round(float(np.percentile(abs_, 25)), 4),
        "median_abs": round(float(np.median(abs_)), 4),
        "mid_price":  round(float(np.median(c)), 4),
    }


def fetch_yahoo_adv_price(symbol: str, days: int = 90) -> Tuple[float, float]:
    """90-day median volume + last close from Yahoo Finance."""
    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=days * 2)).timestamp())
    safe = symbol.replace("^", "%5E").replace("=", "%3D")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}"
           f"?period1={start}&period2={end}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    r = data["chart"]["result"][0]["indicators"]["quote"][0]
    vols = [v for v in r.get("volume") or [] if v]
    closes = [c for c in r.get("close") or [] if c]
    return float(np.median(vols[-days:])), float(closes[-1])


# ───────────────────────────────────────────────────────────────────────────
# Cost math (per stream, annualised)
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class StreamCosts:
    name: str
    ticker: str
    trades_per_year: float
    legs_per_trade: int
    contracts_per_trade: float
    portfolio_weight: float

    # Real data
    spread_rel: float           # p25 (H-L)/close
    option_mid_price: float     # typical contract close
    underlier_adv_notional: float
    underlier_price: float

    # Annual costs in $, then bps of weighted capital, then bps of total capital
    bid_ask_annual_usd: float
    commission_annual_usd: float
    slippage_annual_usd: float
    total_annual_usd: float

    bid_ask_annual_bps_total: float
    commission_annual_bps_total: float
    slippage_annual_bps_total: float
    total_annual_bps_total: float

    # Trade-level diagnostics
    bid_ask_per_trade_usd: float
    commission_per_trade_usd: float
    slippage_per_trade_usd: float
    total_per_trade_usd: float

    notional_per_trade_usd: float
    participation_per_trade: float


def model_stream_costs(s: StreamParams,
                       spread_info: Dict[str, float],
                       adv_notional: float,
                       underlier_price: float,
                       capital: float,
                       leverage: float) -> StreamCosts:
    spread_rel = float(spread_info.get("p25_rel", 0.10))   # fallback 10%
    option_mid = float(spread_info.get("mid_price", 2.00)) # fallback $2

    # Notional per trade at given leverage
    # Underlying-share notional per contract = 100 × underlier_price
    notional_per_leg = s.contracts_per_trade_at_3x * (leverage / 3.0) * 100 * underlier_price
    notional_per_trade = notional_per_leg * s.legs_per_trade

    # ---- Bid-ask spread cost ----
    # Half-spread per leg = spread_rel × option_mid / 2
    # Round-trip: entry + exit on every leg → 2 × 2 half-spreads = 2 × spread_rel × option_mid
    # In $ per contract per round-trip:
    half_spread_dollar = spread_rel * option_mid / 2.0
    round_trip_spread_dollar_per_contract = 2 * (2 * half_spread_dollar)  # entry + exit, each leg
    # Scale to contracts and legs
    contracts_at_lev = s.contracts_per_trade_at_3x * (leverage / 3.0)
    bid_ask_per_trade = (round_trip_spread_dollar_per_contract
                         * contracts_at_lev
                         * s.legs_per_trade)

    # ---- Commission ----
    # $0.65 per contract per leg, round-trip = entry + exit
    commission_per_trade = (COMMISSION_PER_CONTRACT
                            * contracts_at_lev
                            * s.legs_per_trade
                            * 2)

    # ---- Slippage (square-root impact) ----
    participation = notional_per_trade / adv_notional if adv_notional > 0 else 0.0
    impact_bps = SLIPPAGE_COEFF_BPS * math.sqrt(max(participation, 0.0))
    slippage_per_trade = notional_per_trade * impact_bps / 10_000.0
    # Round-trip (entry + exit) → ×2
    slippage_per_trade *= 2

    total_per_trade = bid_ask_per_trade + commission_per_trade + slippage_per_trade

    # Annualise
    bid_ask_annual     = bid_ask_per_trade     * s.trades_per_year
    commission_annual  = commission_per_trade  * s.trades_per_year
    slippage_annual    = slippage_per_trade    * s.trades_per_year
    total_annual       = total_per_trade       * s.trades_per_year

    # Cost in bps of TOTAL portfolio capital (so components can be summed)
    def bps_total(usd: float) -> float:
        return usd / capital * 10_000.0

    return StreamCosts(
        name=s.name, ticker=s.ticker,
        trades_per_year=s.trades_per_year,
        legs_per_trade=s.legs_per_trade,
        contracts_per_trade=contracts_at_lev,
        portfolio_weight=s.portfolio_weight,
        spread_rel=round(spread_rel, 6),
        option_mid_price=round(option_mid, 4),
        underlier_adv_notional=round(adv_notional, 0),
        underlier_price=round(underlier_price, 4),
        bid_ask_annual_usd=round(bid_ask_annual, 2),
        commission_annual_usd=round(commission_annual, 2),
        slippage_annual_usd=round(slippage_annual, 2),
        total_annual_usd=round(total_annual, 2),
        bid_ask_annual_bps_total=round(bps_total(bid_ask_annual), 2),
        commission_annual_bps_total=round(bps_total(commission_annual), 2),
        slippage_annual_bps_total=round(bps_total(slippage_annual), 2),
        total_annual_bps_total=round(bps_total(total_annual), 2),
        bid_ask_per_trade_usd=round(bid_ask_per_trade, 2),
        commission_per_trade_usd=round(commission_per_trade, 2),
        slippage_per_trade_usd=round(slippage_per_trade, 2),
        total_per_trade_usd=round(total_per_trade, 2),
        notional_per_trade_usd=round(notional_per_trade, 0),
        participation_per_trade=round(participation, 8),
    )


# ───────────────────────────────────────────────────────────────────────────
# Net-Sharpe math
# ───────────────────────────────────────────────────────────────────────────

def net_sharpe_from_drag(gross_sharpe: float,
                         gross_cagr_pct: float,
                         vol_pct: float,
                         annual_drag_pct: float) -> Dict[str, float]:
    """Translate annual cost drag into net Sharpe and net CAGR.

    Sharpe = annualised_mean / annualised_vol (both arithmetic).
    Drag is a straight subtraction from annualised mean; vol is assumed
    unchanged (cost is ~deterministic, affects the mean only).

        ann_mean_gross = Sharpe × ann_vol
        ann_mean_net   = ann_mean_gross − drag
        Sharpe_net     = ann_mean_net / ann_vol
    """
    ann_vol = vol_pct / 100.0
    ann_mean_gross = gross_sharpe * ann_vol            # from gross Sharpe
    ann_mean_net   = ann_mean_gross - (annual_drag_pct / 100.0)
    net_sharpe     = ann_mean_net / ann_vol if ann_vol > 1e-12 else 0.0
    net_cagr       = (gross_cagr_pct / 100.0) - (annual_drag_pct / 100.0)
    return {
        "gross_sharpe":   round(gross_sharpe, 3),
        "net_sharpe":     round(net_sharpe, 3),
        "delta_sharpe":   round(net_sharpe - gross_sharpe, 3),
        "gross_cagr_pct": round(gross_cagr_pct, 2),
        "net_cagr_pct":   round(net_cagr * 100, 2),
        "ann_vol_pct":    round(ann_vol * 100, 3),
        "drag_pct":       round(annual_drag_pct, 3),
    }


# ───────────────────────────────────────────────────────────────────────────
# HTML
# ───────────────────────────────────────────────────────────────────────────

def write_html(payload: Dict, path: Path) -> None:
    per_stream = payload["per_stream_costs"]
    summary = payload["summary"]
    net = payload["net_metrics"]

    bps_total = summary["total_drag_bps"]
    drag_pct = bps_total / 100.0
    delta_sh = net["delta_sharpe"]
    color = "#16a34a" if net["net_sharpe"] >= 4.0 else ("#ca8a04" if net["net_sharpe"] >= 3.0 else "#dc2626")
    verdict = ("✅ Net Sharpe still above 4.0" if net["net_sharpe"] >= 4.0 else
               "⚠ Net Sharpe between 3.0 and 4.0" if net["net_sharpe"] >= 3.0 else
               "❌ Costs destroy too much Sharpe")

    stream_rows = ""
    for s in per_stream:
        stream_rows += (
            f"<tr><td>{s['name']}</td><td>{s['ticker']}</td>"
            f"<td>{s['trades_per_year']:.0f}</td>"
            f"<td>{s['legs_per_trade']}</td>"
            f"<td>{s['contracts_per_trade']:.1f}</td>"
            f"<td>${s['notional_per_trade_usd']:,.0f}</td>"
            f"<td>{s['spread_rel']*100:.2f}%</td>"
            f"<td>${s['bid_ask_annual_usd']:,.0f}</td>"
            f"<td>${s['commission_annual_usd']:,.0f}</td>"
            f"<td>${s['slippage_annual_usd']:,.0f}</td>"
            f"<td><strong>${s['total_annual_usd']:,.0f}</strong></td>"
            f"<td><strong>{s['total_annual_bps_total']:.1f}</strong></td></tr>"
        )

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>EXP-2420 Transaction Cost Model</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1150px}}
h1{{font-size:1.5rem;color:#0f172a}} h2{{font-size:1.05rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:18px}}
.headline{{background:#f0fdf4;border-left:5px solid {color};padding:14px 18px;border-radius:6px;margin:14px 0;font-size:0.95rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.15rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.8rem;margin:10px 0}}
th{{background:#f1f5f9;padding:5px 7px;text-align:right;font-size:0.66rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}} td{{padding:5px 7px;text-align:right;border-bottom:1px solid #f1f5f9}} td:first-child{{text-align:left}}
</style></head><body>
<h1>EXP-2420 — Realistic Transaction Cost Model (7-stream, 3× leverage)</h1>
<p class="meta">Bid-ask spread: real IronVault p25(H-L)/close · Commission: Alpaca $0.65/contract ·
Slippage: sqrt market impact vs real Yahoo ADV · Baseline Sharpe 5.96 / CAGR 146.2% (EXP-2200).</p>

<div class="headline"><strong>Net Sharpe:</strong>
{net['gross_sharpe']:.2f} → <strong style="color:{color}">{net['net_sharpe']:.2f}</strong>
(Δ {delta_sh:+.2f}) ·
<strong>Net CAGR:</strong> {net['gross_cagr_pct']:.1f}% → <strong>{net['net_cagr_pct']:.1f}%</strong>
· Annual drag <strong>{drag_pct:.2f}%</strong> ({bps_total:.0f} bps) · {verdict}</div>

<div class="grid">
  <div class="card"><div class="l">Bid-ask drag</div><div class="v">{summary['bid_ask_bps']:.0f} bps</div></div>
  <div class="card"><div class="l">Commission drag</div><div class="v">{summary['commission_bps']:.0f} bps</div></div>
  <div class="card"><div class="l">Slippage drag</div><div class="v">{summary['slippage_bps']:.0f} bps</div></div>
  <div class="card"><div class="l">Total drag</div><div class="v">{summary['total_drag_bps']:.0f} bps</div></div>
  <div class="card"><div class="l">Gross Sharpe</div><div class="v">{net['gross_sharpe']:.2f}</div></div>
  <div class="card"><div class="l">Net Sharpe</div><div class="v" style="color:{color}">{net['net_sharpe']:.2f}</div></div>
  <div class="card"><div class="l">Net CAGR</div><div class="v">{net['net_cagr_pct']:.1f}%</div></div>
  <div class="card"><div class="l">Leverage</div><div class="v">{LEVERAGE:.0f}×</div></div>
</div>

<h2>Per-stream cost breakdown (annual)</h2>
<table><tr><th>Stream</th><th>Ticker</th><th>Tr/yr</th><th>Legs</th><th>Ctr/trade</th>
<th>Notional/trade</th><th>Spread %</th>
<th>Bid-ask $</th><th>Commission $</th><th>Slippage $</th>
<th>Total $</th><th>Total bps</th></tr>
{stream_rows}
<tr style="background:#f1f5f9;font-weight:700"><td colspan="7">PORTFOLIO TOTAL</td>
<td>${summary['bid_ask_usd']:,.0f}</td>
<td>${summary['commission_usd']:,.0f}</td>
<td>${summary['slippage_usd']:,.0f}</td>
<td>${summary['total_drag_usd']:,.0f}</td>
<td>{summary['total_drag_bps']:.0f}</td></tr>
</table>

<h2>Cost drag decomposition</h2>
<table><tr><th>Component</th><th>Annual $</th><th>bps of capital</th><th>% of total drag</th></tr>
<tr><td>Bid-ask spread</td><td>${summary['bid_ask_usd']:,.0f}</td>
<td>{summary['bid_ask_bps']:.0f}</td><td>{summary['bid_ask_bps']/max(summary['total_drag_bps'],1)*100:.0f}%</td></tr>
<tr><td>Commission ($0.65/ctr Alpaca)</td><td>${summary['commission_usd']:,.0f}</td>
<td>{summary['commission_bps']:.0f}</td><td>{summary['commission_bps']/max(summary['total_drag_bps'],1)*100:.0f}%</td></tr>
<tr><td>Slippage (sqrt impact)</td><td>${summary['slippage_usd']:,.0f}</td>
<td>{summary['slippage_bps']:.0f}</td><td>{summary['slippage_bps']/max(summary['total_drag_bps'],1)*100:.0f}%</td></tr>
<tr style="font-weight:700;background:#f1f5f9">
<td>TOTAL</td><td>${summary['total_drag_usd']:,.0f}</td>
<td>{summary['total_drag_bps']:.0f}</td><td>100%</td></tr>
</table>

<h2>Sharpe sensitivity to leverage (informational)</h2>
<p class="meta">Bid-ask and slippage scale with notional (linear in leverage); commission scales with contracts only.
At higher leverage, slippage grows as √lev because the larger position has a disproportionate participation rate.</p>
<table><tr><th>Leverage</th><th>Bid-ask bps</th><th>Commission bps</th><th>Slippage bps</th>
<th>Total bps</th><th>Net Sharpe</th><th>Net CAGR</th></tr>
{payload['leverage_sensitivity_rows']}
</table>

<h2>Method</h2>
<ul>
<li><strong>Bid-ask proxy:</strong> 25th percentile of (high - low)/close across
   IronVault option_daily rows with volume ≥ 500 and close ≥ $0.50 since 2024.
   This is a conservative upper bound on true NBBO spread — when intraday
   range is small, the range IS approximately the spread. When explicit
   NBBO bid/ask is absent from the cache, p25 of daily range is the
   cleanest available estimator.</li>
<li><strong>Commission:</strong> Alpaca options tier $0.65 per contract per leg,
   round-trip (entry + exit).</li>
<li><strong>Slippage:</strong> Square-root market impact model
   impact_bps = 50 · √(trade_notional / underlier_ADV_notional). Coefficient
   50 bps is half the EXP-2140 options-desk coefficient (realistic for the
   low participation rates we operate at).</li>
<li><strong>Leverage:</strong> 3×. Applied to notional per trade (scales
   bid-ask + slippage); commission scales with contract count only.</li>
<li><strong>Net Sharpe:</strong> gross daily mean − (drag/252), vol unchanged;
   Sharpe = (net mean / vol) × √252.</li>
<li><strong>Real data:</strong> IronVault for contract spreads, Yahoo
   90-day median ADV × close for underlier liquidity.</li>
</ul>
<div style="color:#94a3b8;font-size:.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/exp2420_transaction_costs.py · IronVault + Yahoo · Rule Zero
</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-2420 — Transaction Cost Model (3x leverage)")
    print("=" * 60)

    # 1. Measure real bid-ask proxies from IronVault
    print("\n[1/3] Measuring bid-ask proxies from IronVault option_daily…")
    spread_cache: Dict[str, Dict] = {}
    for t in {s.ticker for s in STREAMS}:
        info = measure_bid_ask_proxy(t)
        spread_cache[t] = info
        if info.get("n_samples", 0) > 0:
            print(f"  {t}: n={info['n_samples']:>6}  "
                  f"p25_rel={info['p25_rel']*100:5.2f}%  "
                  f"mid=${info['mid_price']:.2f}")
        else:
            print(f"  {t}: NO IRONVAULT DATA (using fallback)")

    # 2. Fetch underlier ADVs from Yahoo
    print("\n[2/3] Fetching 90-day underlier ADVs from Yahoo…")
    adv_cache: Dict[str, Tuple[float, float]] = {}
    for t in {s.ticker for s in STREAMS}:
        try:
            vol, px = fetch_yahoo_adv_price(t, days=90)
            adv_cache[t] = (vol * px, px)
            print(f"  {t}: ADV ${vol*px/1e9:.2f}B/d  px=${px:.2f}")
        except Exception as e:
            print(f"  {t}: Yahoo failed ({e}) — using fallback")
            adv_cache[t] = (1e9, 100.0)

    # 3. Model each stream
    print("\n[3/3] Modelling per-stream costs at 3× leverage…")
    per_stream = []
    for s in STREAMS:
        spread_info = spread_cache.get(s.ticker, {"p25_rel": 0.10, "mid_price": 2.0})
        adv_not, underlier_px = adv_cache[s.ticker]
        cost = model_stream_costs(s, spread_info, adv_not, underlier_px,
                                  CAPITAL, LEVERAGE)
        per_stream.append(cost)

    # Aggregate
    tot_ba = sum(c.bid_ask_annual_usd for c in per_stream)
    tot_cm = sum(c.commission_annual_usd for c in per_stream)
    tot_sl = sum(c.slippage_annual_usd for c in per_stream)
    tot = tot_ba + tot_cm + tot_sl

    summary = {
        "bid_ask_usd":    round(tot_ba, 2),
        "commission_usd": round(tot_cm, 2),
        "slippage_usd":   round(tot_sl, 2),
        "total_drag_usd": round(tot, 2),
        "bid_ask_bps":    round(tot_ba / CAPITAL * 10_000, 2),
        "commission_bps": round(tot_cm / CAPITAL * 10_000, 2),
        "slippage_bps":   round(tot_sl / CAPITAL * 10_000, 2),
        "total_drag_bps": round(tot / CAPITAL * 10_000, 2),
    }

    net = net_sharpe_from_drag(BASELINE_SHARPE, BASELINE_CAGR_PCT,
                               BASELINE_VOL_PCT,
                               summary["total_drag_bps"] / 100)

    # Leverage sensitivity
    print("\n[leverage sensitivity] recomputing at 1x / 2x / 3x / 5x…")
    lev_rows_html = ""
    lev_table = []
    for lev in [1.0, 2.0, 3.0, 5.0]:
        streams_lev = []
        for s in STREAMS:
            spread_info = spread_cache.get(s.ticker, {"p25_rel": 0.10, "mid_price": 2.0})
            adv_not, underlier_px = adv_cache[s.ticker]
            streams_lev.append(model_stream_costs(s, spread_info, adv_not,
                                                  underlier_px, CAPITAL, lev))
        ba = sum(c.bid_ask_annual_usd for c in streams_lev)
        cm = sum(c.commission_annual_usd for c in streams_lev)
        sl = sum(c.slippage_annual_usd for c in streams_lev)
        t = ba + cm + sl
        t_bps = t / CAPITAL * 10_000
        net_l = net_sharpe_from_drag(BASELINE_SHARPE, BASELINE_CAGR_PCT,
                                     BASELINE_VOL_PCT, t_bps / 100)
        lev_table.append({
            "leverage": lev, "bid_ask_bps": round(ba/CAPITAL*10_000,2),
            "commission_bps": round(cm/CAPITAL*10_000,2),
            "slippage_bps": round(sl/CAPITAL*10_000,2),
            "total_bps": round(t_bps,2),
            "net_sharpe": net_l["net_sharpe"],
            "net_cagr_pct": net_l["net_cagr_pct"],
        })
        lev_rows_html += (
            f"<tr><td>{lev:.0f}×</td>"
            f"<td>{ba/CAPITAL*10_000:.0f}</td>"
            f"<td>{cm/CAPITAL*10_000:.0f}</td>"
            f"<td>{sl/CAPITAL*10_000:.0f}</td>"
            f"<td><strong>{t_bps:.0f}</strong></td>"
            f"<td><strong>{net_l['net_sharpe']:.2f}</strong></td>"
            f"<td>{net_l['net_cagr_pct']:+.1f}%</td></tr>"
        )
        print(f"  {lev:.0f}x: drag {t_bps:>5.0f} bps  net Sharpe {net_l['net_sharpe']:.2f}  "
              f"net CAGR {net_l['net_cagr_pct']:+.1f}%")

    # Console summary
    print()
    print("PER-STREAM (3× leverage)")
    print("-" * 70)
    print(f"{'stream':<10} {'bid_ask':>10} {'comm':>10} {'slip':>10} {'total':>12}")
    for c in per_stream:
        print(f"  {c.name:<10} ${c.bid_ask_annual_usd:>9,.0f} "
              f"${c.commission_annual_usd:>9,.0f} ${c.slippage_annual_usd:>9,.0f} "
              f"${c.total_annual_usd:>10,.0f}")
    print(f"  {'TOTAL':<10} ${tot_ba:>9,.0f} ${tot_cm:>9,.0f} ${tot_sl:>9,.0f} "
          f"${tot:>10,.0f}")
    print()
    print("HEADLINE")
    print("-" * 70)
    print(f"  Gross Sharpe : {net['gross_sharpe']:.2f}")
    print(f"  Annual drag  : {summary['total_drag_bps']:.0f} bps "
          f"({summary['total_drag_bps']/100:.2f}%)")
    print(f"    bid-ask    : {summary['bid_ask_bps']:.0f} bps "
          f"({summary['bid_ask_bps']/max(summary['total_drag_bps'],1)*100:.0f}% of total)")
    print(f"    commission : {summary['commission_bps']:.0f} bps "
          f"({summary['commission_bps']/max(summary['total_drag_bps'],1)*100:.0f}% of total)")
    print(f"    slippage   : {summary['slippage_bps']:.0f} bps "
          f"({summary['slippage_bps']/max(summary['total_drag_bps'],1)*100:.0f}% of total)")
    print(f"  Net Sharpe   : {net['net_sharpe']:.2f} "
          f"(Δ {net['delta_sharpe']:+.2f})")
    print(f"  Net CAGR     : {net['net_cagr_pct']:+.1f}% "
          f"(was {net['gross_cagr_pct']:+.1f}%)")

    payload = {
        "experiment": "EXP-2420",
        "title": "Realistic Transaction Cost Model — 7-stream @ 3× leverage",
        "capital_usd": CAPITAL,
        "leverage":    LEVERAGE,
        "baseline": {
            "sharpe":   BASELINE_SHARPE,
            "cagr_pct": BASELINE_CAGR_PCT,
            "vol_pct":  BASELINE_VOL_PCT,
            "source":   "EXP-2200 equal_risk_15% full-sample",
        },
        "cost_model": {
            "commission_per_contract_usd": COMMISSION_PER_CONTRACT,
            "slippage_coeff_bps": SLIPPAGE_COEFF_BPS,
            "bid_ask_proxy": "IronVault option_daily p25((H-L)/close), vol>=500, close>=$0.50, >=2024",
            "slippage_form": "impact_bps = coeff · sqrt(trade_notional / underlier_ADV_notional)",
        },
        "spread_measurements_by_ticker": spread_cache,
        "adv_by_ticker_usd": {k: v[0] for k, v in adv_cache.items()},
        "underlier_prices":  {k: v[1] for k, v in adv_cache.items()},
        "per_stream_costs": [asdict(c) for c in per_stream],
        "summary": summary,
        "net_metrics": net,
        "leverage_sensitivity": lev_table,
        "leverage_sensitivity_rows": lev_rows_html,
        "rule_zero": "ALL REAL DATA — IronVault option_daily + Yahoo Finance ADV/price",
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    write_html(payload, REPORT_HTML)
    print(f"\nReports → {REPORT_JSON.name} + {REPORT_HTML.name}")
    return payload


if __name__ == "__main__":
    main()
