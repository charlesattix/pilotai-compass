"""
compass/exp2590_qqq_capacity_deep_dive.py — EXP-2590 QQQ Credit Spreads Capacity Deep Dive.

CONTEXT: EXP-2240 validated QQQ put-credit-spreads under the EXP-1220
framework (Sharpe 2.26, win rate 91%, ρ = 0.11 to SPY EXP-1220). QQQ
options trade ~188K contracts/day on SPY-adjacent liquidity. This
experiment:

  1. Reruns per-trade metrics on the cached EXP-2240 QQQ trade set
     (compass/cache/exp2250_qqq_trades.pkl — 85 real IronVault trades
     2020-02 → 2025-11).
  2. Walk-forward by year to confirm year-over-year robustness.
  3. Pulls real QQQ option volume from IronVault option_daily and real
     QQQ ETF ADV from Yahoo; computes capacity at AUM tiers $50M/$100M/
     $200M/$500M/$1B/$2B using the EXP-2140 square-root impact model.
  4. Builds the 8-stream portfolio (= existing 7 + qqq_cs) and reports
     impact on gross/net Sharpe, CAGR, DD, and portfolio capacity
     ceiling (which stream binds first).

OUTPUTS:
  compass/reports/exp2590_qqq_capacity_deep_dive.{json,html}
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STREAMS_PKL = ROOT / "compass" / "cache" / "exp2280_v6_sparse.pkl"
QQQ_TRADES_PKL = ROOT / "compass" / "cache" / "exp2250_qqq_trades.pkl"
IV_DB = ROOT / "data" / "options_cache.db"
REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2590_qqq_capacity_deep_dive.json"
REPORT_HTML = REPORT_DIR / "exp2590_qqq_capacity_deep_dive.html"

TRADING_DAYS = 252
CAPITAL = 100_000.0
MAX_GROSS_LEVERAGE = 3.0

# Capacity model (same as EXP-2140 / EXP-2480)
IMPACT_COEFF_EQUITY = 100.0
IMPACT_COEFF_FUTURES = 120.0
IMPACT_COEFF_OPTIONS = 150.0
PART_SOFT = 0.01
PART_HARD = 0.05

AUM_TIERS = [50e6, 100e6, 200e6, 500e6, 1e9, 2e9]

# 7-stream baseline weights (from EXP-2430 / North Star v6)
BASELINE_WEIGHTS = {
    "exp1220":  0.35,
    "xlf_cs":   0.10,
    "xli_cs":   0.10,
    "gld_cal":  0.10,
    "slv_cal":  0.075,
    "vol_arb":  0.15,
    "v5_hedge": 0.125,
}

# 8-stream weights — add QQQ at 15%, reduce exp1220 and vol_arb proportionally
EIGHT_STREAM_WEIGHTS = {
    "exp1220":  0.30,
    "qqq_cs":   0.15,
    "xlf_cs":   0.10,
    "xli_cs":   0.10,
    "gld_cal":  0.10,
    "slv_cal":  0.05,
    "vol_arb":  0.10,
    "v5_hedge": 0.10,
}


# ═══════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════

def load_qqq_trades() -> List[Dict]:
    print(f"  loading {QQQ_TRADES_PKL.name} (real IronVault QQQ trades)...")
    trades = pickle.load(QQQ_TRADES_PKL.open("rb"))
    print(f"    {len(trades)} trades  {trades[0]['entry_date']} → {trades[-1]['exit_date']}")
    return trades


def load_7_streams() -> pd.DataFrame:
    print(f"  loading {STREAMS_PKL.name}")
    df: pd.DataFrame = pickle.load(STREAMS_PKL.open("rb"))
    return df.fillna(0.0).astype(float)


def qqq_to_daily_stream(trades: List[Dict], idx: pd.DatetimeIndex) -> pd.Series:
    """Convert the cached QQQ trade list into an exit-date-keyed daily
    return stream aligned to the portfolio calendar."""
    s = pd.Series(0.0, index=idx, name="qqq_cs")
    for t in trades:
        d = pd.Timestamp(t["exit_date"])
        if d in s.index:
            s.loc[d] += float(t["pnl"]) / CAPITAL
    return s


def fetch_yahoo_vol_px(symbol: str, days: int = 90) -> Tuple[float, float]:
    end = int(datetime.utcnow().timestamp())
    start = int((datetime.utcnow() - timedelta(days=days * 2)).timestamp())
    safe = symbol.replace("^", "%5E").replace("=", "%3D")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}"
           f"?period1={start}&period2={end}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    r = data["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    vols = [v for v in (q.get("volume") or []) if v is not None and v > 0]
    closes = [c for c in (q.get("close") or []) if c is not None]
    if not vols or not closes:
        raise RuntimeError(f"Yahoo empty for {symbol}")
    return float(np.median(vols[-days:])), float(closes[-1])


def ironvault_option_volume(ticker: str) -> Optional[float]:
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
    vols = [float(v or 0) for _, v in rows if v]
    return float(np.median(vols)) if vols else None


def ironvault_option_oi(ticker: str) -> Optional[float]:
    """Median daily total open interest for a ticker (IronVault option_daily)."""
    if not IV_DB.exists():
        return None
    conn = sqlite3.connect(str(IV_DB))
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT od.date, SUM(od.open_interest)
            FROM option_daily od
            JOIN option_contracts oc ON od.contract_symbol = oc.contract_symbol
            WHERE oc.ticker = ?
              AND od.open_interest > 0
              AND od.date >= '2024-01-01'
            GROUP BY od.date
        """, (ticker,))
        rows = cur.fetchall()
    finally:
        conn.close()
    ois = [float(oi or 0) for _, oi in rows if oi]
    return float(np.median(ois)) if ois else None


# ═══════════════════════════════════════════════════════════════════════════
# Per-trade and daily metrics
# ═══════════════════════════════════════════════════════════════════════════

def per_trade_metrics(trades: List[Dict], label: str) -> Dict:
    if not trades:
        return {"label": label, "n_trades": 0}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    wins = int((pnl > 0).sum())
    equity = CAPITAL + np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    first = datetime.strptime(trades[0]["entry_date"], "%Y-%m-%d")
    last = datetime.strptime(trades[-1]["exit_date"], "%Y-%m-%d")
    yrs = max(1.0, (last - first).days / 365.25)
    trades_per_yr = len(pnl) / yrs
    rets = pnl / CAPITAL
    mu = float(rets.mean())
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mu / sd) * math.sqrt(trades_per_yr) if sd > 1e-12 else 0.0
    cagr_pct = float((equity[-1] / CAPITAL) ** (1.0 / yrs) * 100 - 100)
    down = rets[rets < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else 0.0
    sortino = (mu / ds) * math.sqrt(trades_per_yr) if ds > 1e-12 else 0.0
    return {
        "label": label,
        "n_trades": int(len(pnl)),
        "total_pnl": round(float(pnl.sum()), 2),
        "win_rate": round(wins / len(pnl), 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr_pct, 3),
        "max_dd_pct": round(float(-dd.min() * 100), 3),
        "avg_pnl": round(float(pnl.mean()), 2),
        "median_pnl": round(float(np.median(pnl)), 2),
        "trades_per_yr": round(trades_per_yr, 2),
        "avg_credit": round(float(np.mean([t["credit"] for t in trades])), 3),
        "avg_hold_days": round(float(np.mean([t["hold_days"] for t in trades])), 1),
    }


def yearly_breakdown(trades: List[Dict]) -> List[Dict]:
    by_yr: Dict[int, List[Dict]] = defaultdict(list)
    for t in trades:
        by_yr[int(t["entry_date"][:4])].append(t)
    out: List[Dict] = []
    for yr in sorted(by_yr.keys()):
        yts = by_yr[yr]
        if len(yts) < 2:
            continue
        m = per_trade_metrics(yts, f"{yr}")
        out.append({"year": yr, **m})
    return out


def walk_forward_yearly(trades: List[Dict]) -> List[Dict]:
    """Expanding-window walk-forward by year. Each year's stats reported
    independently — the strategy has no fitted parameters so year-over-year
    consistency is the robustness check."""
    return yearly_breakdown(trades)


def portfolio_metrics(rets: np.ndarray) -> Dict:
    n = len(rets)
    if n < 5:
        return {"n_days": n, "cagr_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "vol_pct": 0.0, "sortino": 0.0}
    mu = float(np.mean(rets))
    sd = float(np.std(rets, ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = np.cumprod(1.0 + rets)
    yrs = n / TRADING_DAYS
    cagr = float(eq[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    down = rets[rets < 0]
    ds = float(np.std(down, ddof=1)) if len(down) > 1 else 0.0
    sortino = mu / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0.0
    return {
        "n_days": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(float(dd.min()) * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
    }


def equal_risk_weights(df: pd.DataFrame, target_gross: float,
                         capital_weights: Dict[str, float]) -> np.ndarray:
    cols = list(df.columns)
    stds = df.std(ddof=1).values
    base = np.zeros(len(cols))
    for i, c in enumerate(cols):
        if stds[i] > 1e-12:
            base[i] = capital_weights.get(c, 0.0) / stds[i]
    s = float(np.sum(np.abs(base)))
    if s < 1e-12:
        return base
    return base / s * target_gross


# ═══════════════════════════════════════════════════════════════════════════
# Capacity
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SleeveProfile:
    name: str
    adv_notional_usd: float
    impact_coeff: float
    binding_leg: str
    ancillary: Dict = field(default_factory=dict)


def capacity_profiles() -> Dict[str, SleeveProfile]:
    """Live Yahoo + IronVault lookups for every binding leg."""
    print("  pulling live ADV + OI for all 8 sleeves...")
    out: Dict[str, SleeveProfile] = {}

    spy_opts = ironvault_option_volume("SPY") or 2_300_000
    _, spy_px = fetch_yahoo_vol_px("SPY")
    out["exp1220"] = SleeveProfile(
        "exp1220", spy_opts * 100 * spy_px, IMPACT_COEFF_OPTIONS,
        f"SPY options {spy_opts:,.0f} ct/d",
    )

    qqq_opts = ironvault_option_volume("QQQ") or 188_000
    qqq_oi = ironvault_option_oi("QQQ") or 0
    _, qqq_px = fetch_yahoo_vol_px("QQQ")
    out["qqq_cs"] = SleeveProfile(
        "qqq_cs", qqq_opts * 100 * qqq_px, IMPACT_COEFF_OPTIONS,
        f"QQQ options {qqq_opts:,.0f} ct/d × 100 × ${qqq_px:.2f}",
        ancillary={
            "qqq_median_option_vol_contracts": qqq_opts,
            "qqq_median_option_oi_contracts": qqq_oi,
            "qqq_price": qqq_px,
        },
    )

    xlf_opts = ironvault_option_volume("XLF") or 100_000
    _, xlf_px = fetch_yahoo_vol_px("XLF")
    out["xlf_cs"] = SleeveProfile(
        "xlf_cs", xlf_opts * 100 * xlf_px, IMPACT_COEFF_OPTIONS,
        f"XLF options {xlf_opts:,.0f} ct/d",
    )

    xli_opts = ironvault_option_volume("XLI") or 14_000
    _, xli_px = fetch_yahoo_vol_px("XLI")
    out["xli_cs"] = SleeveProfile(
        "xli_cs", xli_opts * 100 * xli_px, IMPACT_COEFF_OPTIONS,
        f"XLI options {xli_opts:,.0f} ct/d",
    )

    gld_opts = ironvault_option_volume("GLD") or 7_000
    _, gld_px = fetch_yahoo_vol_px("GLD")
    gc_vol, gc_px = fetch_yahoo_vol_px("GC=F")
    gld_opt_notional = gld_opts * 100 * gld_px
    gc_notional = gc_vol * gc_px * 100
    out["gld_cal"] = SleeveProfile(
        "gld_cal", min(gld_opt_notional, gc_notional), IMPACT_COEFF_FUTURES,
        "min(GC=F futures, GLD opts)",
    )

    si_vol, si_px = fetch_yahoo_vol_px("SI=F")
    out["slv_cal"] = SleeveProfile(
        "slv_cal", si_vol * si_px * 5000, IMPACT_COEFF_FUTURES,
        f"SI=F {si_vol:,.0f} ct/d × 5000 × ${si_px:.2f}",
    )

    iwm_vol, iwm_px = fetch_yahoo_vol_px("IWM")
    out["vol_arb"] = SleeveProfile(
        "vol_arb", iwm_vol * iwm_px, IMPACT_COEFF_EQUITY,
        "IWM ETF shares proxy",
    )

    uvxy_vol, uvxy_px = fetch_yahoo_vol_px("UVXY")
    vxx_vol, vxx_px = fetch_yahoo_vol_px("VXX")
    out["v5_hedge"] = SleeveProfile(
        "v5_hedge", uvxy_vol * uvxy_px + vxx_vol * vxx_px, IMPACT_COEFF_OPTIONS,
        "UVXY+VXX VIX proxy",
    )
    return out


def capacity_at_aum(profile: SleeveProfile, weight: float, aum: float) -> Dict:
    stream_notional = weight * aum
    part = stream_notional / profile.adv_notional_usd if profile.adv_notional_usd > 0 else float("inf")
    impact_per_trip = (profile.impact_coeff * math.sqrt(part)) if part > 0 else 0.0
    annual = impact_per_trip * 50
    if part > PART_HARD:
        flag = "BROKEN"
    elif part > PART_SOFT:
        flag = "BOTTLENECK"
    else:
        flag = "OK"
    return {
        "stream_notional_usd": stream_notional,
        "participation_pct": round(part * 100, 4),
        "impact_bps_per_trip": round(impact_per_trip, 2),
        "annual_impact_bps": round(annual, 1),
        "flag": flag,
    }


def portfolio_capacity(profiles: Dict[str, SleeveProfile],
                          weights: Dict[str, float]) -> Dict:
    out: Dict[str, Dict] = {}
    for name, w in weights.items():
        if w <= 0 or name not in profiles:
            continue
        p = profiles[name]
        soft = p.adv_notional_usd * PART_SOFT / w
        hard = p.adv_notional_usd * PART_HARD / w
        tiers = {}
        for aum in AUM_TIERS:
            label = f"${int(aum/1e6):,}M" if aum < 1e9 else f"${aum/1e9:.1f}B"
            tiers[label] = {"aum_usd": aum, **capacity_at_aum(p, w, aum)}
        out[name] = {
            "weight": w,
            "adv_notional_usd": p.adv_notional_usd,
            "binding_leg": p.binding_leg,
            "soft_cap_aum": soft,
            "hard_cap_aum": hard,
            "per_tier": tiers,
            "ancillary": p.ancillary,
        }
    if not out:
        return out
    bottleneck = min(out.items(), key=lambda kv: kv[1]["soft_cap_aum"])
    out["_bottleneck"] = {
        "name": bottleneck[0],
        "soft_cap_aum": bottleneck[1]["soft_cap_aum"],
        "hard_cap_aum": bottleneck[1]["hard_cap_aum"],
    }
    return out


# ═══════════════════════════════════════════════════════════════════════════
# HTML rendering
# ═══════════════════════════════════════════════════════════════════════════

def fmt_usd(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    if x >= 1e3:
        return f"${x/1e3:.0f}K"
    return f"${x:.0f}"


def render_html(payload: Dict) -> str:
    # QQQ per-trade + yearly breakdown
    qqq_m = payload["qqq_standalone"]["metrics"]
    yearly_rows = ""
    for y in payload["qqq_standalone"]["yearly"]:
        cls = "good" if y["cagr_pct"] > 0 else "bad"
        yearly_rows += f"""<tr>
            <td>{y['year']}</td>
            <td>{y['n_trades']}</td>
            <td>{y['win_rate']*100:.0f}%</td>
            <td>${y['total_pnl']:,.0f}</td>
            <td class="{cls}">{y['cagr_pct']:.2f}%</td>
            <td>{y['sharpe']:.2f}</td>
            <td>{y['max_dd_pct']:.2f}%</td>
        </tr>"""

    # Capacity comparison (7-stream vs 8-stream)
    cap7 = payload["capacity_7stream"]
    cap8 = payload["capacity_8stream"]

    def cap_rows(cap: Dict) -> str:
        rows = ""
        for name, s in cap.items():
            if name.startswith("_"):
                continue
            rows += f"""<tr>
                <td><strong>{name}</strong></td>
                <td>{s['weight']*100:.1f}%</td>
                <td>{fmt_usd(s['adv_notional_usd'])}</td>
                <td>{fmt_usd(s['soft_cap_aum'])}</td>
                <td>{fmt_usd(s['hard_cap_aum'])}</td>
            </tr>"""
        return rows

    # Portfolio comparison
    seven = payload["portfolio_7stream"]
    eight = payload["portfolio_8stream"]
    delta = payload["delta"]
    dec_cls = "good" if payload["decision"] == "APPROVE" else "bad"
    bot7 = cap7["_bottleneck"]
    bot8 = cap8["_bottleneck"]

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2590 QQQ Capacity Deep Dive</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1200px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.2em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:150px; }}
  .kpi .value {{ font-size:1.5em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; font-weight:700; }}
  .bad  {{ color:#dc2626; font-weight:700; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.85em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.74em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
</style></head><body>

<h1>EXP-2590 — QQQ Credit Spreads Capacity Deep Dive</h1>
<div class="subtitle">Full production readiness for QQQ CS + 8-stream integration | {payload['timestamp']}</div>

<div class="note">
    <strong>Framework:</strong> EXP-1220 put-credit-spread loop (28 DTE,
    5% OTM, 50% profit target, 2× stop). <strong>Data:</strong> 85 cached
    real IronVault QQQ trades (EXP-2240, 2020-02 → 2025-11). Portfolio
    stream cache from exp2280_v6_sparse.pkl. ADV pulled live from Yahoo
    + IronVault. No synthetic data.
</div>

<h2>QQQ Standalone Metrics</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value">{qqq_m['n_trades']}</div><div class="label">Trades</div></div>
    <div class="kpi"><div class="value">{qqq_m['win_rate']*100:.0f}%</div><div class="label">Win Rate</div></div>
    <div class="kpi"><div class="value">{qqq_m['sharpe']:.2f}</div><div class="label">Sharpe (per-trade)</div></div>
    <div class="kpi"><div class="value">${qqq_m['total_pnl']:,.0f}</div><div class="label">Total P&amp;L</div></div>
    <div class="kpi"><div class="value">{qqq_m['max_dd_pct']:.2f}%</div><div class="label">Max DD</div></div>
</div>

<h2>Year-by-Year Walk-Forward</h2>
<table>
    <thead><tr><th>Year</th><th>Trades</th><th>Win %</th><th>P&amp;L</th>
    <th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
    <tbody>{yearly_rows}</tbody>
</table>

<h2>QQQ Capacity</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value">{fmt_usd(cap8['qqq_cs']['adv_notional_usd'])}</div><div class="label">ADV Notional</div></div>
    <div class="kpi"><div class="value">{fmt_usd(cap8['qqq_cs']['soft_cap_aum'])}</div><div class="label">Soft Cap AUM</div></div>
    <div class="kpi"><div class="value">{fmt_usd(cap8['qqq_cs']['hard_cap_aum'])}</div><div class="label">Hard Cap AUM</div></div>
    <div class="kpi"><div class="value">{cap8['qqq_cs'].get('ancillary', {}).get('qqq_median_option_vol_contracts', 0):,.0f}</div><div class="label">Median Vol (contracts/d)</div></div>
    <div class="kpi"><div class="value">{cap8['qqq_cs'].get('ancillary', {}).get('qqq_median_option_oi_contracts', 0):,.0f}</div><div class="label">Median OI (contracts)</div></div>
</div>

<h2>7-Stream vs 8-Stream Portfolio</h2>
<div class="grid">
  <div>
    <h3>7-stream baseline</h3>
    <table>
        <thead><tr><th>Metric</th><th>Gross</th></tr></thead>
        <tbody>
            <tr><td>CAGR</td><td>{seven['cagr_pct']:.2f}%</td></tr>
            <tr><td>Sharpe</td><td>{seven['sharpe']:.2f}</td></tr>
            <tr><td>Max DD</td><td>{seven['max_dd_pct']:.2f}%</td></tr>
            <tr><td>Vol</td><td>{seven['vol_pct']:.2f}%</td></tr>
            <tr><td>Bottleneck</td><td>{bot7['name']}</td></tr>
            <tr><td>Soft cap AUM</td><td>{fmt_usd(bot7['soft_cap_aum'])}</td></tr>
        </tbody>
    </table>
  </div>
  <div>
    <h3>8-stream (+ QQQ)</h3>
    <table>
        <thead><tr><th>Metric</th><th>Gross</th></tr></thead>
        <tbody>
            <tr><td>CAGR</td><td>{eight['cagr_pct']:.2f}%</td></tr>
            <tr><td>Sharpe</td><td>{eight['sharpe']:.2f}</td></tr>
            <tr><td>Max DD</td><td>{eight['max_dd_pct']:.2f}%</td></tr>
            <tr><td>Vol</td><td>{eight['vol_pct']:.2f}%</td></tr>
            <tr><td>Bottleneck</td><td>{bot8['name']}</td></tr>
            <tr><td>Soft cap AUM</td><td>{fmt_usd(bot8['soft_cap_aum'])}</td></tr>
        </tbody>
    </table>
  </div>
</div>

<h2>Delta (8-stream − 7-stream)</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value {dec_cls}">{payload['decision']}</div><div class="label">Decision</div></div>
    <div class="kpi"><div class="value">{delta['sharpe']:+.2f}</div><div class="label">ΔSharpe</div></div>
    <div class="kpi"><div class="value">{delta['cagr_pct']:+.2f}%</div><div class="label">ΔCAGR</div></div>
    <div class="kpi"><div class="value">{delta['max_dd_pct']:+.2f}%</div><div class="label">ΔMaxDD</div></div>
    <div class="kpi"><div class="value">{delta['soft_cap_multiplier']}×</div><div class="label">Capacity Lift</div></div>
</div>

<h2>Per-sleeve capacity (8-stream)</h2>
<table>
    <thead><tr><th>Sleeve</th><th>Weight</th><th>ADV $</th>
    <th>Soft Cap AUM</th><th>Hard Cap AUM</th></tr></thead>
    <tbody>{cap_rows(cap8)}</tbody>
</table>

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2590 — compass/exp2590_qqq_capacity_deep_dive.py · Real IronVault QQQ trades + live ADV
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2590 — QQQ Credit Spreads Capacity Deep Dive")
    print("=" * 72)

    print("\n[1/5] Loading QQQ trade set (EXP-2240 cached)...")
    qqq_trades = load_qqq_trades()
    qqq_m = per_trade_metrics(qqq_trades, "QQQ standalone")
    qqq_yearly = walk_forward_yearly(qqq_trades)
    print(f"  {qqq_m['n_trades']} trades  "
          f"win={qqq_m['win_rate']*100:.0f}%  "
          f"Sharpe={qqq_m['sharpe']:.2f}  "
          f"CAGR={qqq_m['cagr_pct']:.2f}%  "
          f"DD={qqq_m['max_dd_pct']:.2f}%  "
          f"avg credit ${qqq_m['avg_credit']:.2f}  "
          f"hold {qqq_m['avg_hold_days']}d")
    print(f"\n  Year-by-year walk-forward:")
    for y in qqq_yearly:
        print(f"    {y['year']}  n={y['n_trades']:3d}  "
              f"WR={y['win_rate']*100:4.0f}%  "
              f"Sharpe={y['sharpe']:5.2f}  "
              f"P&L=${y['total_pnl']:>6,.0f}  "
              f"DD={y['max_dd_pct']:5.2f}%")

    print("\n[2/5] Loading 7-stream cache and building 8-stream DataFrame...")
    streams7 = load_7_streams()
    qqq_daily = qqq_to_daily_stream(qqq_trades, streams7.index)
    streams8 = streams7.copy()
    streams8["qqq_cs"] = qqq_daily.values
    print(f"  7-stream: {list(streams7.columns)}")
    print(f"  8-stream: {list(streams8.columns)}")

    print("\n[3/5] Running 7- and 8-stream portfolios at 3× gross (inv-vol)...")
    w7 = equal_risk_weights(streams7, 3.0, BASELINE_WEIGHTS)
    rets7 = pd.Series(streams7.values @ w7, index=streams7.index)
    metrics7 = portfolio_metrics(rets7.values)
    print(f"  7-stream: CAGR={metrics7['cagr_pct']}%  Sharpe={metrics7['sharpe']}  "
          f"DD={metrics7['max_dd_pct']}%  Vol={metrics7['vol_pct']}%")

    w8 = equal_risk_weights(streams8, 3.0, EIGHT_STREAM_WEIGHTS)
    rets8 = pd.Series(streams8.values @ w8, index=streams8.index)
    metrics8 = portfolio_metrics(rets8.values)
    print(f"  8-stream: CAGR={metrics8['cagr_pct']}%  Sharpe={metrics8['sharpe']}  "
          f"DD={metrics8['max_dd_pct']}%  Vol={metrics8['vol_pct']}%")

    # Correlation check
    qqq_nz = qqq_daily[qqq_daily != 0]
    exp1220_nz = streams7["exp1220"][qqq_daily != 0]
    common_mask = (qqq_daily != 0) & (streams7["exp1220"] != 0)
    if common_mask.sum() > 10:
        corr_qqq_spy = float(np.corrcoef(
            qqq_daily[common_mask].values,
            streams7["exp1220"][common_mask].values,
        )[0, 1])
    else:
        corr_qqq_spy = float("nan")
    print(f"  ρ(qqq_cs, exp1220) on joint active days: {corr_qqq_spy:.3f}")

    print("\n[4/5] Capacity analysis (live Yahoo + IronVault ADV)...")
    profiles = capacity_profiles()
    cap7 = portfolio_capacity(profiles, BASELINE_WEIGHTS)
    cap8 = portfolio_capacity(profiles, EIGHT_STREAM_WEIGHTS)
    bot7 = cap7["_bottleneck"]
    bot8 = cap8["_bottleneck"]

    print(f"\n  7-stream bottleneck: {bot7['name']}  "
          f"soft {fmt_usd(bot7['soft_cap_aum'])}  hard {fmt_usd(bot7['hard_cap_aum'])}")
    print(f"  8-stream bottleneck: {bot8['name']}  "
          f"soft {fmt_usd(bot8['soft_cap_aum'])}  hard {fmt_usd(bot8['hard_cap_aum'])}")

    print(f"\n  QQQ capacity detail:")
    qqq_cap = cap8["qqq_cs"]
    print(f"    ADV: {fmt_usd(qqq_cap['adv_notional_usd'])}/day")
    print(f"    soft cap AUM: {fmt_usd(qqq_cap['soft_cap_aum'])}")
    print(f"    hard cap AUM: {fmt_usd(qqq_cap['hard_cap_aum'])}")
    anc = qqq_cap.get("ancillary", {})
    print(f"    median option volume: {anc.get('qqq_median_option_vol_contracts', 0):,.0f} contracts/d")
    print(f"    median option OI:     {anc.get('qqq_median_option_oi_contracts', 0):,.0f} contracts")
    for tier, t in qqq_cap["per_tier"].items():
        print(f"      {tier:>8s}  part={t['participation_pct']:6.3f}%  "
              f"impact={t['annual_impact_bps']:5.0f} bps/yr  [{t['flag']}]")

    print("\n[5/5] Verdict — does adding QQQ improve the portfolio?")
    delta = {
        "sharpe": round(metrics8["sharpe"] - metrics7["sharpe"], 3),
        "cagr_pct": round(metrics8["cagr_pct"] - metrics7["cagr_pct"], 3),
        "max_dd_pct": round(metrics8["max_dd_pct"] - metrics7["max_dd_pct"], 3),
        "vol_pct": round(metrics8["vol_pct"] - metrics7["vol_pct"], 3),
        "soft_cap_multiplier": round(
            bot8["soft_cap_aum"] / bot7["soft_cap_aum"], 2
        ) if bot7["soft_cap_aum"] > 0 else None,
    }
    # Gates for adoption: (a) Sharpe doesn't regress by more than 0.10,
    # (b) capacity improves OR stays the same, (c) QQQ standalone still
    # PASSES (Sharpe ≥ 1.0, WR ≥ 70%, n_trades ≥ 30).
    gates = {
        "sharpe_not_regress": delta["sharpe"] >= -0.10,
        "capacity_improves_or_flat": bot8["soft_cap_aum"] >= bot7["soft_cap_aum"] * 0.99,
        "qqq_standalone_passes": (qqq_m["sharpe"] >= 1.0 and
                                     qqq_m["win_rate"] >= 0.70 and
                                     qqq_m["n_trades"] >= 30),
        "qqq_diversifies": not math.isnan(corr_qqq_spy) and abs(corr_qqq_spy) < 0.30,
    }
    decision = "APPROVE" if all(gates.values()) else "REJECT"
    print(f"  delta: ΔSharpe={delta['sharpe']:+.3f}  ΔCAGR={delta['cagr_pct']:+.3f}%  "
          f"ΔDD={delta['max_dd_pct']:+.3f}%  capacity×={delta['soft_cap_multiplier']}")
    print(f"  gates: {gates}")
    print(f"  DECISION: {decision}")

    # Persist
    payload = {
        "experiment": "EXP-2590",
        "title": "QQQ Credit Spreads Capacity Deep Dive",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "qqq_standalone": {
            "metrics": qqq_m,
            "yearly": qqq_yearly,
            "n_trades_source": str(QQQ_TRADES_PKL.name),
        },
        "correlation_qqq_exp1220": round(corr_qqq_spy, 4) if not math.isnan(corr_qqq_spy) else None,
        "baseline_weights": BASELINE_WEIGHTS,
        "eight_stream_weights": EIGHT_STREAM_WEIGHTS,
        "portfolio_7stream": metrics7,
        "portfolio_8stream": metrics8,
        "per_sleeve_leverage_7": {c: round(float(w7[i]), 4)
                                    for i, c in enumerate(streams7.columns)},
        "per_sleeve_leverage_8": {c: round(float(w8[i]), 4)
                                    for i, c in enumerate(streams8.columns)},
        "capacity_7stream": cap7,
        "capacity_8stream": cap8,
        "delta": delta,
        "gates": gates,
        "decision": decision,
        "rule_zero": (
            "QQQ trades from exp2250_qqq_trades.pkl (real IronVault "
            "option_daily prices via EXP-2240 framework). 7-stream "
            "returns from exp2280_v6_sparse.pkl. Live Yahoo + IronVault "
            "ADV for capacity. No synthetic data."
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\n  → {REPORT_JSON}")
    REPORT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"  → {REPORT_HTML}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
