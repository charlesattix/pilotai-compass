"""
compass/exp2380_futures_calendar_capacity.py — EXP-2380 Futures-based Calendar Capacity.

QUESTION: EXP-2140 identified the GLD/SLV calendar sleeve as the North
Star v6 capacity bottleneck (SLV binds at $82M portfolio AUM). The
binding leg is the ETF OPTION (or the futures leg for the calendar).
Would switching from ETF options to FUTURES-based calendars — trading
GC vs GLD NAV and SI vs SLV NAV directly — solve the capacity problem?

APPROACH (Rule Zero — all real free data):
  1. Pull real 90-day median volume and last price from Yahoo for GLD,
     SLV, GC=F (gold continuous front), SI=F (silver continuous front).
  2. Pull aggregate GLD/SLV option contract volume from IronVault
     option_daily since 2024. Silver has zero IronVault option
     coverage → handled as an honest data gap.
  3. Translate each to $-notional per day using the correct contract
     multipliers:
         GLD shares   — 1 share = 1×price
         SLV shares   — 1 share = 1×price
         GLD/SLV opts — 100 shares × price per contract
         GC=F futures — 100 troy oz × price per contract
         SI=F futures — 5000 troy oz × price per contract
  4. Apply the same square-root market-impact model used in EXP-2140
     (soft cap = 1% ADV, hard cap = 5% ADV, impact coeff 100 bps for
     equities, 120 for futures, 150 for options).
  5. Compute the portfolio AUM that each sleeve supports under both
     the ETF-option path and the futures path.
  6. Compare side-by-side at $100M, $500M, $1B AUM tiers.
  7. Produce a verdict: does futures-based calendar trading lift the
     capacity ceiling above $1B?

OUTPUTS:
  compass/reports/exp2380_futures_calendar_capacity.{json,html}

Run::
    python3 -m compass.exp2380_futures_calendar_capacity
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2380_futures_calendar_capacity.json"
REPORT_HTML = REPORT_DIR / "exp2380_futures_calendar_capacity.html"
IV_DB = ROOT / "data" / "options_cache.db"

# Impact model calibration (same as EXP-2140)
IMPACT_COEFF_EQUITY = 100.0
IMPACT_COEFF_FUTURES = 120.0
IMPACT_COEFF_OPTIONS = 150.0
PARTICIPATION_SOFT = 0.01
PARTICIPATION_HARD = 0.05
TURNOVER_RT_YEAR = 50   # round-trips per year for calendar spreads
DD_CEILING = 0.12        # 12% portfolio DD ceiling

# Contract multipliers
MULT_GC = 100.0          # 100 troy oz
MULT_SI = 5000.0         # 5000 troy oz
MULT_OPT = 100.0         # 100 shares per equity option contract

# Portfolio weights for the calendar sleeves (North Star v6)
SLEEVE_WEIGHTS = {
    "gld_calendar": 0.075,
    "slv_calendar": 0.075,
}

# AUM tiers (portfolio-level)
AUM_TIERS = [10e6, 50e6, 100e6, 500e6, 1e9, 5e9]


# ═══════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════

def fetch_yahoo_vol_px(symbol: str, days: int = 90) -> Tuple[float, float, int]:
    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=days * 2)).timestamp())
    safe = symbol.replace("^", "%5E").replace("=", "%3D")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}"
           f"?period1={start}&period2={end}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    vols = [v for v in (quote.get("volume") or []) if v is not None and v > 0]
    closes = [c for c in (quote.get("close") or []) if c is not None]
    if not vols or not closes:
        raise RuntimeError(f"Yahoo empty for {symbol}")
    vols = vols[-days:]
    return float(np.median(vols)), float(closes[-1]), len(vols)


def fetch_ironvault_option_volume(ticker: str) -> Optional[float]:
    if not IV_DB.exists():
        return None
    conn = sqlite3.connect(str(IV_DB))
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT od.date, SUM(od.volume)
            FROM option_daily od
            JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
            WHERE oc.ticker = ? AND od.date >= '2024-01-01'
            GROUP BY od.date
        """, (ticker,))
        rows = cur.fetchall()
    finally:
        conn.close()
    vols = [float(v or 0) for _, v in rows]
    vols = [v for v in vols if v > 0]
    if not vols:
        return None
    return float(np.median(vols))


# ═══════════════════════════════════════════════════════════════════════════
# Capacity math
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LiquidityLeg:
    name: str
    instrument: str
    daily_volume_units: float
    unit_notional_usd: float
    adv_notional_usd: float
    impact_coeff: float
    data_source: str
    notes: str = ""


def adv_equity(symbol: str) -> LiquidityLeg:
    vol, px, _ = fetch_yahoo_vol_px(symbol)
    return LiquidityLeg(
        name=f"{symbol} shares",
        instrument=f"{symbol} ETF shares",
        daily_volume_units=vol,
        unit_notional_usd=px,
        adv_notional_usd=vol * px,
        impact_coeff=IMPACT_COEFF_EQUITY,
        data_source="Yahoo Finance 90-day median",
    )


def adv_futures(symbol: str, multiplier: float, label: str) -> LiquidityLeg:
    vol, px, _ = fetch_yahoo_vol_px(symbol)
    return LiquidityLeg(
        name=label,
        instrument=f"{symbol} (mult {multiplier:.0f})",
        daily_volume_units=vol,
        unit_notional_usd=px * multiplier,
        adv_notional_usd=vol * px * multiplier,
        impact_coeff=IMPACT_COEFF_FUTURES,
        data_source=f"Yahoo Finance 90-day median · contract mult {multiplier:.0f}",
    )


def adv_etf_options(ticker: str, ref_px: float) -> Optional[LiquidityLeg]:
    contracts = fetch_ironvault_option_volume(ticker)
    if contracts is None:
        return None
    return LiquidityLeg(
        name=f"{ticker} options",
        instrument=f"{ticker} options ({contracts:,.0f} contracts/d)",
        daily_volume_units=contracts,
        unit_notional_usd=ref_px * MULT_OPT,
        adv_notional_usd=contracts * MULT_OPT * ref_px,
        impact_coeff=IMPACT_COEFF_OPTIONS,
        data_source="IronVault option_daily since 2024",
    )


def market_impact_bps(participation: float, coeff: float) -> float:
    if participation <= 0:
        return 0.0
    return coeff * math.sqrt(participation)


def capacity_at_aum(leg: LiquidityLeg, weight: float, aum: float,
                     baseline_alpha_bps: float) -> Dict:
    stream_notional = weight * aum
    participation = (stream_notional / leg.adv_notional_usd
                      if leg.adv_notional_usd > 0 else float("inf"))
    impact_per_trip = market_impact_bps(participation, leg.impact_coeff)
    annual_impact = impact_per_trip * TURNOVER_RT_YEAR
    alpha_after = max(0.0, baseline_alpha_bps - annual_impact)
    decay_pct = (1.0 - alpha_after / baseline_alpha_bps) * 100.0 if baseline_alpha_bps > 0 else 0.0

    if participation > PARTICIPATION_HARD:
        flag = "BROKEN"
    elif participation > PARTICIPATION_SOFT:
        flag = "BOTTLENECK"
    else:
        flag = "OK"

    return {
        "aum_usd": aum,
        "stream_notional_usd": stream_notional,
        "participation_pct": round(participation * 100, 4),
        "impact_bps_per_trip": round(impact_per_trip, 2),
        "annual_impact_bps": round(annual_impact, 1),
        "baseline_alpha_bps": baseline_alpha_bps,
        "alpha_after_bps": round(alpha_after, 1),
        "alpha_decay_pct": round(decay_pct, 2),
        "flag": flag,
    }


def capacity_ceiling(leg: LiquidityLeg, weight: float) -> Dict:
    """Implied portfolio AUM where this sleeve hits each cap."""
    return {
        "soft_cap_notional_usd": leg.adv_notional_usd * PARTICIPATION_SOFT,
        "hard_cap_notional_usd": leg.adv_notional_usd * PARTICIPATION_HARD,
        "soft_cap_portfolio_aum": (leg.adv_notional_usd * PARTICIPATION_SOFT
                                      / weight) if weight > 0 else float("inf"),
        "hard_cap_portfolio_aum": (leg.adv_notional_usd * PARTICIPATION_HARD
                                      / weight) if weight > 0 else float("inf"),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Workflow
# ═══════════════════════════════════════════════════════════════════════════

def build_sleeve_analysis(sleeve_name: str,
                            etf_symbol: str,
                            futures_symbol: str,
                            futures_multiplier: float,
                            baseline_alpha_bps: float) -> Dict:
    weight = SLEEVE_WEIGHTS[sleeve_name]

    # ETF leg (shares) — always exists
    etf_shares = adv_equity(etf_symbol)

    # ETF options leg — may not exist in IronVault
    etf_opts = adv_etf_options(etf_symbol, etf_shares.unit_notional_usd)

    # Futures leg
    fut = adv_futures(futures_symbol, futures_multiplier,
                        f"{futures_symbol} futures")

    # Two paths: (A) ETF-option calendar, (B) futures calendar
    # Path A binding leg = min(ETF_options, ETF_shares) — normally options
    # Path B binding leg = min(futures, ETF_shares) — usually futures

    if etf_opts is not None:
        path_a_binder = (etf_opts if etf_opts.adv_notional_usd < etf_shares.adv_notional_usd
                          else etf_shares)
    else:
        path_a_binder = None  # data gap

    path_b_binder = fut if fut.adv_notional_usd < etf_shares.adv_notional_usd else etf_shares

    # Per-tier capacity for each path
    def tiers_for(leg: Optional[LiquidityLeg]) -> Dict:
        if leg is None:
            return {}
        return {
            f"${int(t/1e6):,}M" if t < 1e9 else f"${t/1e9:.1f}B":
                capacity_at_aum(leg, weight, t, baseline_alpha_bps)
            for t in AUM_TIERS
        }

    return {
        "sleeve": sleeve_name,
        "weight": weight,
        "baseline_alpha_bps": baseline_alpha_bps,
        "legs": {
            "etf_shares":   leg_to_dict(etf_shares),
            "etf_options":  leg_to_dict(etf_opts) if etf_opts else None,
            "futures":      leg_to_dict(fut),
        },
        "path_a_etf_option_calendar": {
            "binding_leg": path_a_binder.name if path_a_binder else None,
            "status": ("DATA_GAP — IronVault has no options for this ticker"
                        if etf_opts is None and "SLV" in etf_symbol
                        else "OK"),
            "ceiling": capacity_ceiling(path_a_binder, weight) if path_a_binder else None,
            "per_tier": tiers_for(path_a_binder),
        },
        "path_b_futures_calendar": {
            "binding_leg": path_b_binder.name,
            "ceiling": capacity_ceiling(path_b_binder, weight),
            "per_tier": tiers_for(path_b_binder),
        },
    }


def leg_to_dict(leg: Optional[LiquidityLeg]) -> Optional[Dict]:
    if leg is None:
        return None
    return {
        "name": leg.name,
        "instrument": leg.instrument,
        "daily_volume_units": leg.daily_volume_units,
        "unit_notional_usd": leg.unit_notional_usd,
        "adv_notional_usd": leg.adv_notional_usd,
        "impact_coeff": leg.impact_coeff,
        "data_source": leg.data_source,
        "notes": leg.notes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def fmt_usd(x: float) -> str:
    if x is None or x == float("inf"):
        return "—"
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    if x >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def render_html(payload: Dict) -> str:
    def cls_flag(f: str) -> str:
        return {"OK": "ok", "BOTTLENECK": "warn", "BROKEN": "bad"}.get(f, "")

    def sleeve_section(s: Dict) -> str:
        legs = s["legs"]
        legs_rows = ""
        for key, leg in legs.items():
            if leg is None:
                legs_rows += f"""<tr>
                    <td><strong>{key}</strong></td>
                    <td colspan="5"><em>DATA_GAP</em></td>
                </tr>"""
                continue
            legs_rows += f"""<tr>
                <td><strong>{key}</strong></td>
                <td>{leg['instrument']}</td>
                <td>{leg['daily_volume_units']:,.0f}</td>
                <td>{fmt_usd(leg['unit_notional_usd'])}</td>
                <td><strong>{fmt_usd(leg['adv_notional_usd'])}</strong></td>
                <td>{leg['impact_coeff']:.0f}</td>
            </tr>"""

        def path_section(label: str, path: Dict) -> str:
            if path is None or not path.get("per_tier"):
                return f"<h4>{label}: <em>DATA_GAP</em></h4>"
            c = path["ceiling"]
            tier_rows = ""
            for tier, t in path["per_tier"].items():
                tier_rows += f"""<tr>
                    <td>{tier}</td>
                    <td>{fmt_usd(t['stream_notional_usd'])}</td>
                    <td>{t['participation_pct']:.3f}%</td>
                    <td>{t['annual_impact_bps']:.1f}</td>
                    <td>{t['alpha_after_bps']:.0f}</td>
                    <td>{t['alpha_decay_pct']:.1f}%</td>
                    <td class="{cls_flag(t['flag'])}"><strong>{t['flag']}</strong></td>
                </tr>"""
            return f"""
                <h4>{label}</h4>
                <p>Binding leg: <strong>{path['binding_leg']}</strong> ·
                   soft cap AUM: {fmt_usd(c['soft_cap_portfolio_aum'])} ·
                   hard cap AUM: {fmt_usd(c['hard_cap_portfolio_aum'])}</p>
                <table>
                    <thead><tr><th>AUM</th><th>Sleeve Notional</th>
                    <th>Participation</th><th>Impact bps/yr</th>
                    <th>Alpha After</th><th>Decay</th><th>Flag</th></tr></thead>
                    <tbody>{tier_rows}</tbody>
                </table>
            """

        return f"""
        <h2>{s['sleeve']} (w={s['weight']*100:.1f}%, baseline alpha {s['baseline_alpha_bps']:.0f} bps/yr)</h2>
        <h3>Real ADV legs</h3>
        <table>
            <thead><tr><th>Leg</th><th>Instrument</th><th>Daily Vol</th>
            <th>Unit Notional</th><th>ADV $</th><th>Impact k</th></tr></thead>
            <tbody>{legs_rows}</tbody>
        </table>
        {path_section("Path A — ETF-option calendar", s['path_a_etf_option_calendar'])}
        {path_section("Path B — Futures calendar", s['path_b_futures_calendar'])}
        """

    sleeve_sections = "".join(sleeve_section(s) for s in payload["sleeves"])

    verdict = payload["verdict"]
    v_cls = "good" if verdict["answer"] == "YES" else "bad"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2380 Futures Calendar Capacity</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1200px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.2em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  h3 {{ color:#475569; margin-top:1.5em; }}
  h4 {{ color:#64748b; margin-top:1.2em; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:150px; }}
  .kpi .value {{ font-size:1.5em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; font-weight:700; }}
  .bad  {{ color:#dc2626; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.74em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  td.ok   {{ background:#dcfce7; }}
  td.warn {{ background:#fef9c3; }}
  td.bad  {{ background:#fee2e2; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
</style></head><body>

<h1>EXP-2380 — Futures-Based Calendar Capacity</h1>
<div class="subtitle">Does switching GLD/SLV calendar from ETF options to futures fix the bottleneck? | {payload['timestamp']}</div>

<div class="note">
    <strong>Model:</strong> square-root market-impact
    <code>impact = k · √participation</code>, k = 100 (equity) / 120
    (futures) / 150 (options). Soft cap = 1% ADV, hard cap = 5% ADV,
    50 round-trips/yr turnover. <strong>Data:</strong> Yahoo Finance
    90-day median for ETFs/futures; IronVault <code>option_daily</code>
    (since 2024) for options. Real data only.
</div>

<h2>Verdict</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value {v_cls}">{verdict['answer']}</div><div class="label">Does futures fix the bottleneck?</div></div>
    <div class="kpi"><div class="value">{verdict['etf_option_path_soft_aum']}</div><div class="label">ETF-option path soft cap</div></div>
    <div class="kpi"><div class="value">{verdict['futures_path_soft_aum']}</div><div class="label">Futures path soft cap</div></div>
    <div class="kpi"><div class="value">{verdict['multiplier']:.1f}×</div><div class="label">Futures/ETF-opt liquidity ratio</div></div>
</div>
<ul>
    {''.join(f'<li>{r}</li>' for r in verdict['reasons'])}
</ul>

{sleeve_sections}

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2380 — compass/exp2380_futures_calendar_capacity.py · Yahoo + IronVault real volumes
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2380 — Futures-Based Calendar Spread Capacity Scaling")
    print("=" * 72)

    print("\n[1/3] Pulling real ADV for ETFs, options, and futures...")

    sleeve_configs = [
        ("gld_calendar", "GLD", "GC=F", MULT_GC, 1520.0),   # 15.2%/yr baseline
        ("slv_calendar", "SLV", "SI=F", MULT_SI, 2489.0),   # 24.9%/yr baseline
    ]

    sleeves: List[Dict] = []
    for name, etf, fut, mult, alpha in sleeve_configs:
        print(f"\n  → {name}")
        s = build_sleeve_analysis(name, etf, fut, mult, alpha)
        # Pretty-print
        legs = s["legs"]
        print(f"    {etf} shares:    {fmt_usd(legs['etf_shares']['adv_notional_usd']):>12s}/day")
        if legs["etf_options"]:
            print(f"    {etf} options:   {fmt_usd(legs['etf_options']['adv_notional_usd']):>12s}/day")
        else:
            print(f"    {etf} options:   DATA_GAP (IronVault)")
        print(f"    {fut:6s} futures: {fmt_usd(legs['futures']['adv_notional_usd']):>12s}/day")

        path_a = s["path_a_etf_option_calendar"]
        path_b = s["path_b_futures_calendar"]
        if path_a.get("ceiling"):
            print(f"    Path A (ETF-option): soft cap {fmt_usd(path_a['ceiling']['soft_cap_portfolio_aum'])}  "
                  f"hard cap {fmt_usd(path_a['ceiling']['hard_cap_portfolio_aum'])}")
        else:
            print(f"    Path A (ETF-option): DATA_GAP")
        print(f"    Path B (futures):    soft cap {fmt_usd(path_b['ceiling']['soft_cap_portfolio_aum'])}  "
              f"hard cap {fmt_usd(path_b['ceiling']['hard_cap_portfolio_aum'])}")
        sleeves.append(s)

    print("\n[2/3] Per-tier capacity comparison...")
    for s in sleeves:
        print(f"\n  {s['sleeve']} (w={s['weight']*100:.1f}%)")
        path_a_tiers = s["path_a_etf_option_calendar"].get("per_tier", {})
        path_b_tiers = s["path_b_futures_calendar"].get("per_tier", {})
        for tier in path_b_tiers:
            a = path_a_tiers.get(tier, {})
            b = path_b_tiers[tier]
            a_flag = a.get("flag", "DATA_GAP")
            a_part = a.get("participation_pct", None)
            a_disp = f"{a_part:.3f}%" if a_part is not None else "N/A"
            print(f"    {tier:>8s}  ETF-opt: part={a_disp:>8s} [{a_flag:>10s}]   "
                  f"Futures: part={b['participation_pct']:>7.3f}% [{b['flag']}]")

    print("\n[3/3] Verdict...")
    # Binding sleeve = the one with the lowest soft-cap AUM on each path
    def soft_min(path_key: str) -> Tuple[Optional[str], Optional[float]]:
        best_name, best_aum = None, None
        for s in sleeves:
            p = s[path_key]
            c = p.get("ceiling")
            if c is None:
                continue
            aum = c["soft_cap_portfolio_aum"]
            if best_aum is None or aum < best_aum:
                best_name, best_aum = s["sleeve"], aum
        return best_name, best_aum

    etf_bind, etf_aum = soft_min("path_a_etf_option_calendar")
    fut_bind, fut_aum = soft_min("path_b_futures_calendar")

    # Liquidity ratio = futures ADV / ETF-option ADV on the bottleneck sleeve
    if etf_aum and fut_aum:
        mult = fut_aum / etf_aum
    else:
        mult = float("inf") if etf_aum is None else 0.0

    answer = "YES" if fut_aum and fut_aum >= 1e9 else ("PARTIAL" if fut_aum and fut_aum >= 100e6 else "NO")

    reasons: List[str] = []
    if etf_aum:
        reasons.append(
            f"ETF-option path binds at {fmt_usd(etf_aum)} portfolio AUM "
            f"(bottleneck sleeve: {etf_bind})."
        )
    else:
        reasons.append(
            "ETF-option path has at least one DATA_GAP sleeve — SLV options "
            "are absent from IronVault, so an apples-to-apples comparison "
            "cannot use SLV options directly. Path A binder falls back to "
            "SLV ETF shares, which is already more liquid than SLV options."
        )
    if fut_aum:
        reasons.append(
            f"Futures path binds at {fmt_usd(fut_aum)} portfolio AUM "
            f"(bottleneck sleeve: {fut_bind})."
        )
        if mult != float("inf"):
            reasons.append(
                f"Futures-path capacity is {mult:.1f}× the ETF-option path."
            )
    if fut_aum and fut_aum >= 1e9:
        reasons.append(
            "Futures calendar supports ≥ $1B portfolio AUM — the bottleneck "
            "is effectively removed at the target scale."
        )
    elif fut_aum and fut_aum >= 100e6:
        reasons.append(
            "Futures calendar lifts the ceiling into the $100M-$1B range "
            "but still binds before $1B. Partial fix — usable for mid-size "
            "AUM but not enough for full billion-dollar scaling."
        )
    else:
        reasons.append(
            "Futures calendar does not provide enough headroom."
        )

    print(f"  ETF-option path binds: {fmt_usd(etf_aum) if etf_aum else 'DATA_GAP'} (sleeve={etf_bind})")
    print(f"  Futures path binds:    {fmt_usd(fut_aum) if fut_aum else 'DATA_GAP'} (sleeve={fut_bind})")
    if mult != float("inf"):
        print(f"  Liquidity multiplier:  {mult:.1f}×")
    print(f"  Answer: {answer}")

    payload = {
        "experiment": "EXP-2380",
        "title": "Futures-Based Calendar Spread Capacity",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": {
            "impact_formula": "impact_bps = k · sqrt(participation)",
            "impact_coeff_equity": IMPACT_COEFF_EQUITY,
            "impact_coeff_futures": IMPACT_COEFF_FUTURES,
            "impact_coeff_options": IMPACT_COEFF_OPTIONS,
            "participation_soft_cap_pct_adv": PARTICIPATION_SOFT * 100,
            "participation_hard_cap_pct_adv": PARTICIPATION_HARD * 100,
            "turnover_round_trips_per_year": TURNOVER_RT_YEAR,
        },
        "aum_tiers_usd": AUM_TIERS,
        "sleeves": sleeves,
        "verdict": {
            "answer": answer,
            "etf_option_bottleneck_sleeve": etf_bind,
            "etf_option_path_soft_aum": fmt_usd(etf_aum) if etf_aum else "DATA_GAP",
            "etf_option_path_soft_aum_usd": etf_aum,
            "futures_bottleneck_sleeve": fut_bind,
            "futures_path_soft_aum": fmt_usd(fut_aum) if fut_aum else "DATA_GAP",
            "futures_path_soft_aum_usd": fut_aum,
            "multiplier": round(mult, 2) if mult != float("inf") else None,
            "reasons": reasons,
        },
        "rule_zero": (
            "Real 90-day median ADV from Yahoo (GLD, SLV, GC=F, SI=F). "
            "IronVault option_daily for GLD options. SLV options are not "
            "in IronVault — reported as DATA_GAP, not simulated. No "
            "synthetic volumes."
        ),
    }

    print("\nWriting reports...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  → {REPORT_JSON}")
    REPORT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"  → {REPORT_HTML}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
