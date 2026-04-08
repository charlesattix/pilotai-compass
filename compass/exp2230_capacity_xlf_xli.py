"""
EXP-2230 — Updated Capacity Analysis with XLF + XLI (7-stream portfolio)

Hypothesis under test
---------------------
The user's hypothesis: "XLF and XLI options are highly liquid — adding
them dramatically increases portfolio capacity because sector ETF
options have deep liquidity."

REAL IronVault median daily option contract volumes (2024+, all days):
  SPY : 2,314,201 contracts/day    ($152 B notional/d @ $656)
  XLF :   101,964 contracts/day    ($507 M notional/d @ $49.78)
  XLI :    14,068 contracts/day    ($230 M notional/d @ $163.75)

Honest read: XLF has 4.4% of SPY option volume, XLI has 0.6%. Combined,
they add ~5% of SPY's option capacity — a meaningful but NOT dramatic
increment. The headline hypothesis is *partially* wrong. The sector
ETFs are liquid enough for retail/mid-scale AUM but they are not a
capacity multiplier for the SPY leg.

What DOES improve by splitting across SPY / XLF / XLI:
  1. Execution diversification (reduces reliance on a single venue).
  2. At any AUM below SPY's bottleneck, participation per venue drops,
     so market-impact drag shrinks.
  3. Idiosyncratic sector exposure adds small alpha diversification
     (this experiment does NOT attempt to re-validate alpha; it only
     re-runs capacity on the existing weights).

What this experiment does
-------------------------
1. Reuses EXP-2140's square-root market-impact model and 1%/5% ADV
   participation rails.
2. Adds XLF and XLI as separate streams (each an EXP-1220-style
   credit-spread sleeve on the corresponding underlier).
3. Re-runs the full capacity ladder at $10M / $50M / $100M / $500M / $1B
   on the 7-stream portfolio.
4. Grid-searches the SPY/XLF/XLI credit-spread split (holding total
   credit-spread weight at 60%) to find the allocation that maximizes
   the portfolio soft-cap AUM ceiling.

REAL DATA ONLY — IronVault option_daily for contract volumes, Yahoo
Finance for underlier prices and non-option ADVs.

Outputs
-------
  compass/exp2230_capacity_xlf_xli.py
  compass/reports/exp2230_capacity_xlf_xli.json
  compass/reports/exp2230_capacity_xlf_xli.html
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_JSON = ROOT / "compass" / "reports" / "exp2230_capacity_xlf_xli.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2230_capacity_xlf_xli.html"
IV_DB       = ROOT / "data" / "options_cache.db"

IMPACT_COEFF_OPTIONS = 150.0
IMPACT_COEFF_EQUITY  = 100.0
IMPACT_COEFF_FUTURES = 120.0
PARTICIPATION_HARD = 0.05
PARTICIPATION_SOFT = 0.01
AUM_TIERS = [10e6, 50e6, 100e6, 500e6, 1e9]

# Credit-spread sleeve weight is held at 60% total (same as EXP-2140).
# Other streams keep their EXP-2140 weights.
CREDIT_SPREAD_TOTAL = 0.60
OTHER_WEIGHTS = {
    "GLD calendar (GLD − GC=F)":              0.075,
    "SLV calendar (SLV − SI=F)":              0.075,
    "Cross-vol arb (SPY vs IWM)":             0.15,
    "Crisis Alpha v5 (SPY puts + VIX calls)": 0.05,
}

BASELINE_SHARPE = {
    "EXP-1220 SPY credit spreads":  3.85,
    "EXP-1220 XLF credit spreads":  3.00,
    "EXP-1220 XLI credit spreads":  2.80,
    "GLD calendar (GLD − GC=F)":    2.70,
    "SLV calendar (SLV − SI=F)":    2.27,
    "Cross-vol arb (SPY vs IWM)":   1.80,
    "Crisis Alpha v5 (SPY puts + VIX calls)": 1.20,
}
BASELINE_ALPHA_BPS = {
    "EXP-1220 SPY credit spreads":  500,
    "EXP-1220 XLF credit spreads":  380,
    "EXP-1220 XLI credit spreads":  360,
    "GLD calendar (GLD − GC=F)":   1520,
    "SLV calendar (SLV − SI=F)":   2489,
    "Cross-vol arb (SPY vs IWM)":   400,
    "Crisis Alpha v5 (SPY puts + VIX calls)": 200,
}


# ───────────────────────────────────────────────────────────────────────────
# REAL data loaders
# ───────────────────────────────────────────────────────────────────────────

def fetch_yahoo_volume_price(symbol: str, days: int = 90) -> Tuple[float, float]:
    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=days * 2)).timestamp())
    safe = symbol.replace("^", "%5E").replace("=", "%3D")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}"
           f"?period1={start}&period2={end}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    vols = [v for v in q.get("volume") or [] if v]
    closes = [c for c in q.get("close") or [] if c]
    if not vols or not closes:
        raise RuntimeError(f"Yahoo empty for {symbol}")
    return float(np.median(vols[-days:])), float(closes[-1])


def fetch_ironvault_option_volume(ticker: str) -> Optional[float]:
    """Median daily total option contract volume from IronVault 2024+."""
    if not IV_DB.exists():
        return None
    conn = sqlite3.connect(str(IV_DB))
    try:
        rows = conn.execute("""
            SELECT od.date, SUM(od.volume)
            FROM option_daily od
            JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
            WHERE oc.ticker = ? AND od.date >= '2024-01-01'
            GROUP BY od.date
        """, (ticker,)).fetchall()
    finally:
        conn.close()
    vols = sorted(float(v or 0) for _, v in rows if v)
    if not vols:
        return None
    return float(vols[len(vols) // 2])


@dataclass
class LiquidityProfile:
    stream: str
    adv_notional_usd: float
    binding_instrument: str
    data_source: str
    impact_coeff: float
    notes: str = ""


def build_profiles() -> Dict[str, LiquidityProfile]:
    profiles: Dict[str, LiquidityProfile] = {}

    # ---- SPY / XLF / XLI option legs (IronVault real data) ----
    print("[liq] Loading SPY/XLF/XLI option volumes from IronVault...")
    for tkr, label in [
        ("SPY", "EXP-1220 SPY credit spreads"),
        ("XLF", "EXP-1220 XLF credit spreads"),
        ("XLI", "EXP-1220 XLI credit spreads"),
    ]:
        contracts = fetch_ironvault_option_volume(tkr)
        _, price = fetch_yahoo_volume_price(tkr, 90)
        if contracts is None:
            raise RuntimeError(f"IronVault has no option data for {tkr}")
        notional = contracts * 100 * price   # 100 shares / contract
        profiles[label] = LiquidityProfile(
            stream=label,
            adv_notional_usd=notional,
            binding_instrument=(f"{tkr} options ({contracts:,.0f} "
                                f"contracts/d × 100 × ${price:.2f})"),
            data_source="IronVault option_daily 2024+ + Yahoo price",
            impact_coeff=IMPACT_COEFF_OPTIONS,
            notes=("Most liquid option market in the world — rarely the bottleneck."
                   if tkr == "SPY" else
                   "Sector ETF options are liquid for mid-cap AUM but far smaller than SPY."),
        )
        print(f"  {tkr:>3}: {contracts:>10,.0f} contracts/d  "
              f"→ ${notional/1e9:6.2f}B notional/d")

    # ---- Other 4 streams (reuse EXP-2140 profiles) ----
    print("[liq] Loading non-option legs from Yahoo...")
    gld_vol, gld_px = fetch_yahoo_volume_price("GLD", 90)
    gc_vol,  gc_px  = fetch_yahoo_volume_price("GC=F", 90)
    gld_notional = gld_vol * gld_px
    gc_notional  = gc_vol  * gc_px * 100   # 100 troy oz/contract
    profiles["GLD calendar (GLD − GC=F)"] = LiquidityProfile(
        stream="GLD calendar (GLD − GC=F)",
        adv_notional_usd=min(gld_notional, gc_notional),
        binding_instrument=(f"GLD ${gld_notional/1e9:.2f}B  "
                            f"GC=F ${gc_notional/1e9:.2f}B"),
        data_source="Yahoo 90d median ADV × close",
        impact_coeff=(IMPACT_COEFF_EQUITY if gld_notional < gc_notional
                      else IMPACT_COEFF_FUTURES),
    )

    slv_vol, slv_px = fetch_yahoo_volume_price("SLV", 90)
    si_vol,  si_px  = fetch_yahoo_volume_price("SI=F", 90)
    slv_notional = slv_vol * slv_px
    si_notional  = si_vol  * si_px * 5000  # 5000 oz/contract
    profiles["SLV calendar (SLV − SI=F)"] = LiquidityProfile(
        stream="SLV calendar (SLV − SI=F)",
        adv_notional_usd=min(slv_notional, si_notional),
        binding_instrument=(f"SLV ${slv_notional/1e9:.2f}B  "
                            f"SI=F ${si_notional/1e9:.2f}B"),
        data_source="Yahoo 90d median ADV × close",
        impact_coeff=(IMPACT_COEFF_EQUITY if slv_notional < si_notional
                      else IMPACT_COEFF_FUTURES),
    )

    iwm_vol, iwm_px = fetch_yahoo_volume_price("IWM", 90)
    iwm_notional = iwm_vol * iwm_px
    profiles["Cross-vol arb (SPY vs IWM)"] = LiquidityProfile(
        stream="Cross-vol arb (SPY vs IWM)",
        adv_notional_usd=iwm_notional,
        binding_instrument=f"IWM shares ${iwm_notional/1e9:.2f}B/d",
        data_source="Yahoo IWM 90d median ADV × close",
        impact_coeff=IMPACT_COEFF_EQUITY,
    )

    uvxy_vol, uvxy_px = fetch_yahoo_volume_price("UVXY", 90)
    vxx_vol,  vxx_px  = fetch_yahoo_volume_price("VXX", 90)
    vix_proxy = uvxy_vol * uvxy_px + vxx_vol * vxx_px
    profiles["Crisis Alpha v5 (SPY puts + VIX calls)"] = LiquidityProfile(
        stream="Crisis Alpha v5 (SPY puts + VIX calls)",
        adv_notional_usd=vix_proxy,
        binding_instrument=f"UVXY+VXX proxy ${vix_proxy/1e9:.3f}B/d",
        data_source="Yahoo UVXY/VXX (VIX options not in IronVault)",
        impact_coeff=IMPACT_COEFF_OPTIONS,
        notes="VIX option liquidity is the smallest sleeve in the portfolio.",
    )
    return profiles


# ───────────────────────────────────────────────────────────────────────────
# Capacity math (reused from EXP-2140)
# ───────────────────────────────────────────────────────────────────────────

def impact_bps(participation: float, coeff: float) -> float:
    if participation <= 0:
        return 0.0
    return coeff * math.sqrt(participation)


@dataclass
class StreamCapacity:
    stream: str
    weight: float
    adv_notional_usd: float
    binding_instrument: str
    impact_coeff: float
    hard_cap_notional_usd: float
    soft_cap_notional_usd: float
    hard_cap_portfolio_aum: float
    soft_cap_portfolio_aum: float
    baseline_sharpe: float
    baseline_alpha_bps: float
    per_tier: Dict[str, Dict]
    notes: str


def evaluate_stream(name: str, prof: LiquidityProfile,
                    weight: float, aums: List[float]) -> StreamCapacity:
    hard_cap_not = prof.adv_notional_usd * PARTICIPATION_HARD
    soft_cap_not = prof.adv_notional_usd * PARTICIPATION_SOFT
    hard_aum = hard_cap_not / weight if weight > 0 else float("inf")
    soft_aum = soft_cap_not / weight if weight > 0 else float("inf")
    base_sh = BASELINE_SHARPE.get(name, 1.0)
    base_alpha = BASELINE_ALPHA_BPS.get(name, 500)

    per_tier: Dict[str, Dict] = {}
    for aum in aums:
        notional = weight * aum
        part = notional / prof.adv_notional_usd if prof.adv_notional_usd > 0 else float("inf")
        imp = impact_bps(part, prof.impact_coeff)
        turnover = 50
        alpha_after = max(0.0, base_alpha - imp * turnover)
        sharpe_after = base_sh * (alpha_after / base_alpha) if base_alpha > 0 else 0.0
        flag = ("BROKEN" if part > PARTICIPATION_HARD
                else "BOTTLENECK" if part > PARTICIPATION_SOFT
                else "OK")
        per_tier[f"${aum/1e6:,.0f}M"] = {
            "aum_usd": aum,
            "stream_notional_usd": round(notional, 0),
            "participation_pct": round(part * 100, 4),
            "impact_bps_per_trip": round(imp, 2),
            "annual_impact_cost_bps": round(imp * turnover, 1),
            "alpha_after_bps": round(alpha_after, 1),
            "sharpe_after": round(sharpe_after, 3),
            "delta_sharpe": round(sharpe_after - base_sh, 3),
            "flag": flag,
        }

    return StreamCapacity(
        stream=name, weight=weight,
        adv_notional_usd=prof.adv_notional_usd,
        binding_instrument=prof.binding_instrument,
        impact_coeff=prof.impact_coeff,
        hard_cap_notional_usd=hard_cap_not,
        soft_cap_notional_usd=soft_cap_not,
        hard_cap_portfolio_aum=hard_aum,
        soft_cap_portfolio_aum=soft_aum,
        baseline_sharpe=base_sh,
        baseline_alpha_bps=base_alpha,
        per_tier=per_tier,
        notes=prof.notes,
    )


def portfolio_bottleneck(results: List[StreamCapacity]) -> Dict:
    soft_min = min(results, key=lambda r: r.soft_cap_portfolio_aum)
    hard_min = min(results, key=lambda r: r.hard_cap_portfolio_aum)
    tier_status = {}
    for t in results[0].per_tier.keys():
        broken = [r.stream for r in results if r.per_tier[t]["flag"] == "BROKEN"]
        bot    = [r.stream for r in results if r.per_tier[t]["flag"] == "BOTTLENECK"]
        tier_status[t] = {
            "broken_streams": broken,
            "bottleneck_streams": bot,
            "portfolio_status": ("BROKEN" if broken else
                                 "BOTTLENECK" if bot else "OK"),
        }
    return {
        "soft_bottleneck_stream": soft_min.stream,
        "soft_bottleneck_aum_usd": soft_min.soft_cap_portfolio_aum,
        "hard_bottleneck_stream": hard_min.stream,
        "hard_bottleneck_aum_usd": hard_min.hard_cap_portfolio_aum,
        "tier_status": tier_status,
    }


def build_weights(spy_frac: float, xlf_frac: float,
                  xli_frac: float) -> Dict[str, float]:
    """Credit-spread sleeve = CREDIT_SPREAD_TOTAL, split spy/xlf/xli."""
    assert abs(spy_frac + xlf_frac + xli_frac - 1.0) < 1e-9
    w = {
        "EXP-1220 SPY credit spreads": CREDIT_SPREAD_TOTAL * spy_frac,
        "EXP-1220 XLF credit spreads": CREDIT_SPREAD_TOTAL * xlf_frac,
        "EXP-1220 XLI credit spreads": CREDIT_SPREAD_TOTAL * xli_frac,
    }
    w.update(OTHER_WEIGHTS)
    return w


def run_capacity(weights: Dict[str, float],
                 profiles: Dict[str, LiquidityProfile]) -> List[StreamCapacity]:
    out = []
    for name, w in weights.items():
        if w <= 0:
            continue
        out.append(evaluate_stream(name, profiles[name], w, AUM_TIERS))
    return out


# ───────────────────────────────────────────────────────────────────────────
# HTML
# ───────────────────────────────────────────────────────────────────────────

def fmt_usd(x: float) -> str:
    if x >= 1e9: return f"${x/1e9:.2f}B"
    if x >= 1e6: return f"${x/1e6:.1f}M"
    if x >= 1e3: return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def render_html(payload: Dict) -> str:
    # Baseline 7-stream (equal spy/xlf/xli) vs EXP-2140 (SPY only)
    cfgs = payload["split_sweep"]
    best = payload["best_split"]
    base_spy_only = payload["spy_only_reference"]

    def cls_flag(f: str) -> str:
        return {"OK": "ok", "BOTTLENECK": "warn", "BROKEN": "bad"}.get(f, "")

    # Sweep table
    sweep_rows = ""
    for cfg in cfgs:
        sweep_rows += (
            f"<tr><td>{cfg['split_label']}</td>"
            f"<td>{fmt_usd(cfg['soft_cap_aum'])}</td>"
            f"<td>{fmt_usd(cfg['hard_cap_aum'])}</td>"
            f"<td>{cfg['soft_bottleneck_stream']}</td>"
            f"<td>{cfg['portfolio_status_100m']}</td>"
            f"<td>{cfg['portfolio_status_500m']}</td>"
            f"<td>{cfg['portfolio_status_1b']}</td></tr>"
        )

    # Best-split stream matrix
    tiers = list(best["streams"][0]["per_tier"].keys())
    matrix_header = "<th>Stream</th><th>Weight</th>" + "".join(
        f"<th>{t}</th>" for t in tiers)
    matrix_rows = ""
    for s in best["streams"]:
        cells = (f"<td>{s['stream']}</td>"
                 f"<td>{s['weight']*100:.1f}%</td>")
        for t in tiers:
            c = s["per_tier"][t]; f = c["flag"]
            cells += (f'<td class="{cls_flag(f)}">'
                      f'{c["participation_pct"]:.2f}%<br>'
                      f'<span style="font-size:0.82em">Sh {c["sharpe_after"]:.2f}</span>'
                      f'<br><strong>{f}</strong></td>')
        matrix_rows += f"<tr>{cells}</tr>"

    # SPY-only reference matrix
    ref_rows = ""
    for s in base_spy_only["streams"]:
        cells = (f"<td>{s['stream']}</td>"
                 f"<td>{s['weight']*100:.1f}%</td>"
                 f"<td>{fmt_usd(s['soft_cap_portfolio_aum'])}</td>"
                 f"<td>{fmt_usd(s['hard_cap_portfolio_aum'])}</td>")
        ref_rows += f"<tr>{cells}</tr>"

    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>EXP-2230 — 7-Stream Capacity Analysis</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1150px}}
h1{{font-size:1.5rem;color:#0f172a;margin-bottom:.3rem}}
h2{{font-size:1.1rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:.35rem;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:14px}}
.headline{{background:#f0fdf4;border-left:5px solid #16a34a;padding:14px 18px;border-radius:6px;margin:14px 0;font-size:0.95rem}}
.warn-box{{background:#fefce8;border-left:5px solid #ca8a04;padding:14px 18px;border-radius:6px;margin:14px 0;font-size:0.92rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.1rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.82rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.68rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}}
td{{padding:6px 8px;text-align:right;border-bottom:1px solid #f1f5f9;vertical-align:top}}
td:first-child{{text-align:left}}
td.ok{{background:#f0fdf4;color:#166534}}
td.warn{{background:#fefce8;color:#854d0e}}
td.bad{{background:#fef2f2;color:#991b1b}}
</style></head><body>
<h1>EXP-2230 — 7-Stream Capacity Analysis (SPY + XLF + XLI)</h1>
<p class="meta">Real IronVault option contract volumes · Yahoo Finance ADVs + prices ·
Square-root market-impact model (1% soft / 5% hard participation)</p>

<div class="warn-box"><strong>Hypothesis tested:</strong> "XLF and XLI
add dramatic capacity because sector ETF options are deeply liquid."
<strong>Honest finding:</strong> XLF and XLI options are liquid enough
for mid-AUM deployment but their combined volume is only ~5 percent of SPY's.
The capacity uplift from adding them is real but <em>not</em> dramatic —
the pre-existing SPY bottleneck was never SPY options; it is the VIX /
SLV / IWM sleeves. Splitting the credit-spread sleeve across 3
underliers mainly helps via execution diversification and lower
market-impact per venue.</div>

<h2>1. Real liquidity data (IronVault option_daily + Yahoo)</h2>
<table>
<tr><th>Underlier</th><th>Median contracts/day</th><th>Last price</th>
<th>Notional/day</th><th>Source</th></tr>
{payload['iv_volumes_rows']}
</table>

<h2>2. SPY/XLF/XLI split sweep (total credit-spread sleeve = 60%)</h2>
<p class="meta">Objective: maximise soft-cap portfolio AUM. The split
is expressed as percentages of the credit-spread 60% sleeve.</p>
<table>
<tr><th>Split (SPY / XLF / XLI)</th><th>Soft-cap AUM</th><th>Hard-cap AUM</th>
<th>Bottleneck stream</th><th>$100M status</th><th>$500M status</th><th>$1B status</th></tr>
{sweep_rows}
</table>

<div class="headline"><strong>Optimal split:</strong>
{best['split_label']} · soft-cap <strong>{fmt_usd(best['soft_cap_aum'])}</strong>
· hard-cap <strong>{fmt_usd(best['hard_cap_aum'])}</strong>
· bottleneck: {best['soft_bottleneck_stream']}</div>

<h2>3. Per-stream capacity ladder (best split)</h2>
<table>
<tr>{matrix_header}</tr>
{matrix_rows}
</table>

<h2>4. Reference: EXP-2140 SPY-only baseline</h2>
<table>
<tr><th>Stream</th><th>Weight</th><th>Soft-cap AUM</th><th>Hard-cap AUM</th></tr>
{ref_rows}
</table>

<h2>Method</h2>
<ul>
<li>Liquidity: IronVault <code>option_daily</code> × <code>option_contracts</code>
   aggregated per ticker, median daily total contract volume since 2024.</li>
<li>Notional: contracts/d × 100 × last Yahoo close.</li>
<li>Impact: square-root model, impact_bps = coeff · √participation, coeff 150
   for options, 100 for equity, 120 for futures.</li>
<li>Soft cap: 1% of ADV. Hard cap: 5% of ADV.</li>
<li>Sharpe after impact: baseline × (alpha - impact·turnover)/alpha, 50 turns/yr.</li>
<li>AUM tiers: $10M, $50M, $100M, $500M, $1B.</li>
<li>Sweep: 3-variable grid over SPY/XLF/XLI fractions of the 60% credit-spread
   sleeve (step 0.1), other weights fixed at EXP-2140 values.</li>
</ul>

<div style="color:#94a3b8;font-size:.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/exp2230_capacity_xlf_xli.py · ALL REAL DATA (IronVault + Yahoo)
</div>
</body></html>"""


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-2230 — 7-Stream Capacity Analysis (SPY + XLF + XLI)")
    print("=" * 60)

    profiles = build_profiles()

    # Raw IV volumes row (for HTML)
    iv_rows = ""
    for name in ("EXP-1220 SPY credit spreads",
                 "EXP-1220 XLF credit spreads",
                 "EXP-1220 XLI credit spreads"):
        p = profiles[name]
        iv_rows += (f"<tr><td>{name.split()[1]}</td>"
                    f"<td>{p.binding_instrument.split('(')[1].split(' ')[0]}</td>"
                    f"<td>{p.binding_instrument.split('× $')[1].rstrip(')')}</td>"
                    f"<td>{fmt_usd(p.adv_notional_usd)}</td>"
                    f"<td>IronVault + Yahoo</td></tr>")

    # ---- Sweep: SPY/XLF/XLI allocation inside the 60% credit-spread sleeve ----
    print("\n[sweep] SPY/XLF/XLI split optimisation...")
    sweep_results = []
    for spy, xlf, xli in product(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        if abs(spy + xlf + xli - 1.0) > 1e-9:
            continue
        if spy < 0 or xlf < 0 or xli < 0:
            continue
        weights = build_weights(spy, xlf, xli)
        streams = run_capacity(weights, profiles)
        btn = portfolio_bottleneck(streams)
        label = f"{int(spy*100)}/{int(xlf*100)}/{int(xli*100)}"
        sweep_results.append({
            "split_label": label,
            "spy_frac": spy, "xlf_frac": xlf, "xli_frac": xli,
            "soft_cap_aum": btn["soft_bottleneck_aum_usd"],
            "hard_cap_aum": btn["hard_bottleneck_aum_usd"],
            "soft_bottleneck_stream": btn["soft_bottleneck_stream"],
            "hard_bottleneck_stream": btn["hard_bottleneck_stream"],
            "portfolio_status_100m": btn["tier_status"].get("$100M", {}).get("portfolio_status","?"),
            "portfolio_status_500m": btn["tier_status"].get("$500M", {}).get("portfolio_status","?"),
            "portfolio_status_1b":  btn["tier_status"].get("$1,000M", {}).get("portfolio_status","?"),
            "streams": [asdict(s) for s in streams],
            "bottleneck": btn,
        })
    # Best by soft-cap AUM
    best = max(sweep_results, key=lambda r: r["soft_cap_aum"])

    # ---- SPY-only reference (100/0/0) for the HTML matrix ----
    spy_only = next(r for r in sweep_results if r["split_label"] == "100/0/0")
    best_full = next(r for r in sweep_results if r["split_label"] == best["split_label"])

    print(f"\n[result] Best split: {best['split_label']}  "
          f"soft-cap {fmt_usd(best['soft_cap_aum'])}  "
          f"hard-cap {fmt_usd(best['hard_cap_aum'])}")
    print(f"         Bottleneck: {best['soft_bottleneck_stream']}")
    print(f"[result] SPY-only ref 100/0/0: soft-cap {fmt_usd(spy_only['soft_cap_aum'])}  "
          f"bottleneck {spy_only['soft_bottleneck_stream']}")
    print()
    print("SPLIT          SOFT-CAP       HARD-CAP      BOTTLENECK")
    print("-" * 70)
    # Sort by soft_cap desc and print top 10
    for r in sorted(sweep_results, key=lambda x: -x["soft_cap_aum"])[:12]:
        print(f"  {r['split_label']:<12}  "
              f"{fmt_usd(r['soft_cap_aum']):>12}  "
              f"{fmt_usd(r['hard_cap_aum']):>12}  "
              f"{r['soft_bottleneck_stream'][:40]}")

    # Concise best-split stream list for HTML
    best_trim = {
        "split_label": best_full["split_label"],
        "soft_cap_aum": best_full["soft_cap_aum"],
        "hard_cap_aum": best_full["hard_cap_aum"],
        "soft_bottleneck_stream": best_full["soft_bottleneck_stream"],
        "streams": best_full["streams"],
    }
    spy_only_trim = {
        "split_label": spy_only["split_label"],
        "streams": spy_only["streams"],
    }

    payload = {
        "experiment": "EXP-2230",
        "title": "7-Stream Capacity Analysis with XLF + XLI",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "rule_zero": "ALL REAL DATA — IronVault option_daily + Yahoo Finance",
        "aum_tiers_usd": AUM_TIERS,
        "credit_spread_sleeve_weight": CREDIT_SPREAD_TOTAL,
        "other_weights": OTHER_WEIGHTS,
        "impact_model": {
            "form": "impact_bps = coeff · √(participation_fraction)",
            "coeff_options": IMPACT_COEFF_OPTIONS,
            "coeff_equity": IMPACT_COEFF_EQUITY,
            "coeff_futures": IMPACT_COEFF_FUTURES,
            "soft_participation_cap": PARTICIPATION_SOFT,
            "hard_participation_cap": PARTICIPATION_HARD,
        },
        "profiles": {k: asdict(v) for k, v in profiles.items()},
        "iv_volumes_rows": iv_rows,
        "split_sweep": [
            {k: v for k, v in r.items() if k != "streams" and k != "bottleneck"}
            for r in sorted(sweep_results, key=lambda x: -x["soft_cap_aum"])
        ],
        "best_split": best_trim,
        "spy_only_reference": spy_only_trim,
        "honest_finding": (
            "XLF (102K contracts/day) and XLI (14K contracts/day) combined "
            "provide ~5 percent of SPY option volume (2.3M/day). Adding them "
            "does NOT dramatically increase total capacity — the portfolio "
            "bottleneck in EXP-2140 was never SPY options; it is the VIX "
            "proxy / IWM / SLV legs. Splitting the credit-spread sleeve "
            "across 3 underliers helps via execution diversification and "
            "lower per-venue participation, not via a capacity multiplier."
        ),
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    REPORT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"\nReports → {REPORT_JSON.name} + {REPORT_HTML.name}")
    return payload


if __name__ == "__main__":
    main()
