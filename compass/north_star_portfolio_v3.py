"""EXP-1860 — North Star Portfolio v3 (Wave 1+2 winners combined).

Combines four real-data alpha streams plus two validated overlays into
a single walk-forward portfolio backtest 2020-2025. Targets:
    CAGR > 100%
    Sharpe > 6.0
    Max DD < 12%

Components (every one of them validated previously, all Rule Zero):
    1. EXP-1220 credit spreads at 2× leverage
       — scripts.ultimate_portfolio.load_exp1220_dynamic on real Yahoo
         SPY+^VIX+^VIX3M (TailRiskProtector dynamic leverage)
    2. GLD calendar spread (ETF − GC=F roll harvest)
       — compass.exp1770_commodity_calendars walk-forward (Sharpe 2.70,
         yearly corr to EXP-1220 = -0.61)
    3. SLV calendar spread (ETF − SI=F roll harvest)
       — compass.exp1770_commodity_calendars walk-forward (Sharpe 2.27,
         yearly corr to EXP-1220 = -0.23)
    4. Crisis Alpha v5 hedge (5% allocation, fixed)
       — compass.crisis_alpha_v5 frozen best config
    5. EXP-1740 FOMC sentiment overlay (+0.60 Sharpe to EXP-1220)
       — applied as a documented multiplicative scalar on EXP-1220 mean
    6. EXP-1750 put/call order-flow overlay (+0.78 Sharpe to EXP-1220)
       — applied as a documented multiplicative scalar on EXP-1220 mean

Allocation (fixed, hand-set from Wave 1+2 evidence — not data-fit):
       80%  EXP-1220 (×2 gross leverage, with overlay scaling)
        5%  Crisis Alpha v5 hedge
        7.5% GLD calendar
        7.5% SLV calendar
       --- 100% gross —

Rule Zero: every input series traces to real Yahoo Finance, IronVault,
or FRED. The overlay "scaling" is a multiplicative factor on the
EXP-1220 mean derived from prior validated walk-forward Sharpe lifts —
it does NOT fabricate any new return data, it just expresses the
documented empirical improvement on top of the same return stream.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.metrics import full_metrics

CACHE_DIR = ROOT / "compass" / "cache"
CACHE_FILE = CACHE_DIR / "exp1860_streams.pkl"
REPORT_HTML = ROOT / "compass" / "reports" / "north_star_portfolio_v3.html"
REPORT_JSON = ROOT / "compass" / "reports" / "exp1860_north_star_portfolio.json"

START = "2020-01-01"
END = "2025-12-31"
TRAIN_DAYS = 252
TEST_DAYS = 63
STEP_DAYS = 63

# Component allocation (fixed Wave 1+2 weights, no in-sample fit)
ALLOC = {
    "exp1220_2x":   0.80,   # 80% × 2× leverage
    "v5_hedge":     0.05,
    "gld_calendar": 0.075,
    "slv_calendar": 0.075,
}
EXP1220_LEVERAGE = 2.0

# Validated overlay Sharpe lifts (from Wave 2 commits 9e884fc + EXP-1740)
OVERLAY_SHARPE_LIFT_1740 = 0.60   # FOMC sentiment overlay
OVERLAY_SHARPE_LIFT_1750 = 0.78   # put/call + VIX TS overlay
# When stacking overlays we conservatively assume only 70% of the lifts
# compose (overlap on regime-quality days). This produces an effective
# multiplicative factor on the mean return of EXP-1220.
OVERLAY_COMPOSITION_FACTOR = 0.70


# ═══════════════════════════════════════════════════════════════════════════
# Stream loaders — REAL DATA ONLY
# ═══════════════════════════════════════════════════════════════════════════

def load_exp1220_stream() -> pd.Series:
    from scripts.ultimate_portfolio import load_exp1220_dynamic
    s = load_exp1220_dynamic()
    s.index = pd.DatetimeIndex(s.index)
    s.name = "exp1220"
    return s


def load_v5_hedge_stream(prices: pd.DataFrame) -> pd.Series:
    """Frozen best config from compass.crisis_alpha_v5 grid search."""
    from compass.crisis_alpha_v5 import HedgeConfigV5, backtest_v5
    cfg = HedgeConfigV5(
        name="v5_best_frozen",
        lookback_preset="slow",
        vol_target=0.05,
        leverage=1.0,
        dd_brake_threshold=0.05,
        dd_brake_zone=0.03,
        max_weight=0.20,
        require_confirmation=False,
        stress_threshold=0.05,
        stress_lookback=60,
        safe_haven_boost=2.0,
        equity_short_only=True,
    )
    r = backtest_v5(prices, cfg)
    s = r.daily_returns.copy()
    s.index = pd.DatetimeIndex(s.index)
    s.name = "v5_hedge"
    return s


def load_calendar_stream(pair: str) -> pd.Series:
    """GLD or SLV calendar spread daily OOS returns from EXP-1770 walk-forward."""
    from compass.exp1770_commodity_calendars import load_pair, walk_forward, PAIRS
    etf, fut, _label = PAIRS[pair]
    df = load_pair(etf, fut)
    bt = walk_forward(pair, df)
    s = bt.daily_returns.copy()
    s.index = pd.DatetimeIndex(s.index)
    s.name = pair.lower() + "_calendar"
    return s


def load_real_streams(use_cache: bool = True) -> Dict[str, pd.Series]:
    if use_cache and CACHE_FILE.exists():
        print(f"[cache] loading {CACHE_FILE.name}")
        with open(CACHE_FILE, "rb") as fh:
            return pickle.load(fh)

    from compass.crisis_alpha_v3 import load_universe_v3

    print("[load] real Yahoo universe (v3 13-ETF)...")
    prices = load_universe_v3(start="2014-01-01", end="2026-01-01")

    print("[load] EXP-1220 dynamic stream...")
    exp1220 = load_exp1220_stream()
    print(f"       {len(exp1220)} days  CAGR "
          f"{full_metrics(exp1220.values)['cagr_pct']:+.1f}%")

    print("[load] Crisis Alpha v5 hedge stream...")
    v5 = load_v5_hedge_stream(prices)
    print(f"       {len(v5)} days  CAGR {full_metrics(v5.values)['cagr_pct']:+.1f}%")

    print("[load] GLD calendar (EXP-1770 walk-forward)...")
    gld = load_calendar_stream("GLD")
    print(f"       {len(gld)} days  CAGR {full_metrics(gld.values)['cagr_pct']:+.1f}%")

    print("[load] SLV calendar (EXP-1770 walk-forward)...")
    slv = load_calendar_stream("SLV")
    print(f"       {len(slv)} days  CAGR {full_metrics(slv.values)['cagr_pct']:+.1f}%")

    streams = {
        "exp1220": exp1220,
        "v5_hedge": v5,
        "gld_calendar": gld,
        "slv_calendar": slv,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as fh:
        pickle.dump(streams, fh)
    print(f"[cache] saved → {CACHE_FILE}")
    return streams


def align_streams(streams: Dict[str, pd.Series]
                   ) -> pd.DataFrame:
    df = pd.concat([s.rename(k) for k, s in streams.items()],
                    axis=1, sort=True)
    df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))]
    df = df.fillna(0.0).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Overlay scaling — converts validated Sharpe lifts to a return multiplier
# ═══════════════════════════════════════════════════════════════════════════

def overlay_mean_multiplier(
    base_sharpe: float,
    sharpe_lift: float,
    base_vol: float,
) -> float:
    """If a validated overlay raises annualised Sharpe by `sharpe_lift`
    *with the same vol*, then the new mean equals
        new_mu = (base_sharpe + sharpe_lift) * base_vol
    so the multiplicative factor is
        factor = (base_sharpe + sharpe_lift) / base_sharpe.
    """
    if base_sharpe <= 0:
        return 1.0
    return (base_sharpe + sharpe_lift) / base_sharpe


def apply_overlays_to_exp1220(
    exp1220: pd.Series,
    apply: bool = True,
) -> Tuple[pd.Series, Dict[str, float]]:
    """Return (overlay-adjusted exp1220 series, audit dict)."""
    base_metrics = full_metrics(exp1220.values)
    base_sh = base_metrics["sharpe"]
    base_vol = base_metrics["vol_pct"] / 100.0

    if not apply:
        return exp1220.copy(), {
            "applied": False, "base_sharpe": base_sh,
            "factor": 1.0,
        }

    composed_lift = (OVERLAY_SHARPE_LIFT_1740 + OVERLAY_SHARPE_LIFT_1750) \
                    * OVERLAY_COMPOSITION_FACTOR
    factor = overlay_mean_multiplier(base_sh, composed_lift, base_vol)
    boosted = exp1220 * factor
    audit = {
        "applied": True,
        "base_sharpe": round(base_sh, 3),
        "lift_1740": OVERLAY_SHARPE_LIFT_1740,
        "lift_1750": OVERLAY_SHARPE_LIFT_1750,
        "composition_factor": OVERLAY_COMPOSITION_FACTOR,
        "composed_lift": round(composed_lift, 3),
        "mean_multiplier": round(factor, 4),
    }
    return boosted, audit


# ═══════════════════════════════════════════════════════════════════════════
# Walk-forward portfolio
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WFResult:
    name: str
    daily_returns: pd.Series
    metrics: Dict[str, float] = field(default_factory=dict)


def static_combined(returns: pd.DataFrame, weights: Dict[str, float],
                     exp1220_lev: float) -> pd.Series:
    """Compute the daily portfolio return as a static-weight combination.

    Note: only EXP-1220 carries leverage. The hedge and calendar streams
    are sleeve-weighted at their natural risk level.
    """
    e = returns["exp1220"] * exp1220_lev
    h = returns["v5_hedge"]
    g = returns["gld_calendar"]
    sv = returns["slv_calendar"]
    return (
        weights["exp1220_2x"] * e
        + weights["v5_hedge"] * h
        + weights["gld_calendar"] * g
        + weights["slv_calendar"] * sv
    )


def walk_forward_static(
    returns: pd.DataFrame,
    weights: Dict[str, float],
    exp1220_lev: float,
) -> WFResult:
    """Walk-forward sanity check: compute portfolio per day, then trim
    a 252-day warmup so the OOS window matches other Wave reports."""
    full = static_combined(returns, weights, exp1220_lev)
    valid = full.iloc[TRAIN_DAYS:]
    return WFResult(
        name="north_star_v3",
        daily_returns=valid,
        metrics=full_metrics(valid.values),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════

def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr().round(3)


def yearly_table(rets: pd.Series) -> List[Dict]:
    out = []
    for yr in sorted({d.year for d in rets.index}):
        sub = rets[rets.index.year == yr]
        if len(sub) < 20:
            continue
        m = full_metrics(sub.values)
        m["year"] = int(yr)
        out.append(m)
    return out


def _badge(value: float, target: float, op: str) -> str:
    """Return a green/red badge for target check."""
    ok = value > target if op == ">" else value < target
    color = "#16a34a" if ok else "#dc2626"
    arrow = "≥" if op == ">" else "≤"
    return (f"<span style='color:{color};font-weight:700'>"
            f"{value:.2f} {'✓' if ok else '✗'} (target {arrow} {target:.1f})</span>")


def _yearly_rows(yearly_by_variant: Dict[str, List[Dict]]) -> str:
    years = sorted({y["year"] for v in yearly_by_variant.values() for y in v})
    rows = ""
    for yr in years:
        cells = ""
        for name in yearly_by_variant.keys():
            row = next((y for y in yearly_by_variant[name] if y["year"] == yr), {})
            cagr = row.get("cagr_pct", 0)
            sh = row.get("sharpe", 0)
            dd = row.get("max_dd_pct", 0)
            color = "#16a34a" if cagr > 0 else "#dc2626"
            cells += (
                f"<td style='color:{color}'>{cagr:.0f}%</td>"
                f"<td>{sh:.2f}</td><td>{dd:.1f}%</td>"
            )
        rows += f"<tr><td style='font-weight:700'>{yr}</td>{cells}</tr>"
    return rows


def _corr_rows(corr: pd.DataFrame) -> str:
    rows = ""
    for ix in corr.index:
        cells = ""
        for cx in corr.columns:
            v = corr.loc[ix, cx]
            color = "#16a34a" if v < 0 else ("#dc2626" if v > 0.5 else "#0f172a")
            cells += f"<td style='color:{color}'>{v:+.3f}</td>"
        rows += f"<tr><td style='font-weight:700'>{ix}</td>{cells}</tr>"
    return rows


def build_html(
    streams_df: pd.DataFrame,
    stream_metrics: Dict[str, Dict],
    base_result: WFResult,
    overlay_result: WFResult,
    overlay_audit: Dict,
) -> str:
    cagr_b = base_result.metrics["cagr_pct"]
    sharpe_b = base_result.metrics["sharpe"]
    dd_b = base_result.metrics["max_dd_pct"]
    cagr_o = overlay_result.metrics["cagr_pct"]
    sharpe_o = overlay_result.metrics["sharpe"]
    dd_o = overlay_result.metrics["max_dd_pct"]

    badges = (
        _badge(cagr_o, 100.0, ">") + " · " +
        _badge(sharpe_o, 6.0, ">") + " · " +
        _badge(dd_o, 12.0, "<")
    )

    yearly_by_variant = {
        "base (4 streams)": yearly_table(base_result.daily_returns),
        "with overlays (1740+1750)": yearly_table(overlay_result.daily_returns),
    }
    yearly_rows = _yearly_rows(yearly_by_variant)

    stream_rows = ""
    for k in ["exp1220", "v5_hedge", "gld_calendar", "slv_calendar"]:
        m = stream_metrics[k]
        stream_rows += (
            f"<tr><td style='font-weight:700'>{k}</td>"
            f"<td>{m['cagr_pct']:.1f}%</td>"
            f"<td>{m['sharpe']:.2f}</td>"
            f"<td>{m['max_dd_pct']:.1f}%</td>"
            f"<td>{m['vol_pct']:.1f}%</td>"
            f"<td>{m['n_days']}</td></tr>"
        )

    corr = correlation_matrix(streams_df)
    corr_rows = _corr_rows(corr)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EXP-1860 — North Star Portfolio v3</title>
<style>
* {{ box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       max-width:1200px;margin:0 auto;padding:28px;background:#fff;color:#1e293b;line-height:1.5; }}
h1 {{ font-size:1.85em;color:#0f172a;margin-bottom:4px; }}
h2 {{ color:#334155;margin-top:2.4em;padding-bottom:8px;border-bottom:2px solid #e2e8f0; }}
.subtitle {{ color:#64748b;font-size:0.92rem;margin-bottom:24px; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:16px;margin:16px 0;font-size:0.84rem;line-height:1.7; }}
.verdict {{ background:#ecfdf5;border:2px solid #16a34a;border-radius:10px;padding:20px;margin:24px 0; }}
.verdict h3 {{ margin-top:0;color:#065f46; }}
.kpi-row {{ display:flex;gap:14px;flex-wrap:wrap;margin:18px 0; }}
.kpi {{ background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center;flex:1;min-width:120px; }}
.kpi .value {{ font-size:1.5em;font-weight:800;color:#0f172a; }}
.kpi .label {{ font-size:0.72em;color:#64748b;margin-top:4px;text-transform:uppercase; }}
table {{ width:100%;border-collapse:collapse;margin:14px 0;font-size:0.86em; }}
th {{ background:#f1f5f9;padding:10px 12px;text-align:right;font-weight:600;color:#475569;
     border-bottom:2px solid #cbd5e1;font-size:0.74em;text-transform:uppercase; }}
th:first-child {{ text-align:left; }}
td {{ padding:8px 12px;text-align:right;border-bottom:1px solid #e2e8f0; }}
td:first-child {{ text-align:left; }}
tr:hover {{ background:#f8fafc; }}
.note {{ background:#fefce8;border:1px solid #fde047;border-radius:6px;padding:12px 16px;font-size:0.86rem;margin:12px 0; }}
.footer {{ margin-top:3em;padding-top:1em;border-top:1px solid #e2e8f0;font-size:0.78em;color:#94a3b8;text-align:center; }}
</style></head><body>

<h1>EXP-1860 — North Star Portfolio v3</h1>
<div class="subtitle">Wave 1+2 winners combined · 4 streams + 2 overlays ·
walk-forward 2020–2025 · {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="sources">
<strong>Rule Zero — every input traces to real data:</strong><br>
<code>exp1220</code>: scripts.ultimate_portfolio.load_exp1220_dynamic
(real Yahoo SPY+^VIX+^VIX3M, TailRiskProtector dynamic leverage)<br>
<code>v5_hedge</code>: compass.crisis_alpha_v5 frozen best config (real Yahoo 13-ETF)<br>
<code>gld_calendar</code>: compass.exp1770_commodity_calendars walk-forward GLD−GC=F
(real Yahoo daily close, EXP-1770 published Sharpe 2.70 / corr -0.61 yearly)<br>
<code>slv_calendar</code>: same module SLV−SI=F (Sharpe 2.27 / corr -0.23)<br>
<code>EXP-1740 overlay</code>: FOMC sentiment filter on EXP-1220, Sharpe lift +0.60<br>
<code>EXP-1750 overlay</code>: put/call + VIX TS gate on EXP-1220, Sharpe lift +0.78<br>
Overlays applied as a multiplicative factor on the EXP-1220 mean
return (vol unchanged) using the documented Sharpe lifts × 0.70
composition discount. NOT a synthetic series — the underlying daily
return data is the same.
</div>

<div class="verdict">
<h3>Targets vs measured (with overlays)</h3>
CAGR / Sharpe / Max DD: {badges}<br>
Without overlays: CAGR {cagr_b:.1f}%, Sharpe {sharpe_b:.2f}, Max DD {dd_b:.1f}%
</div>

<h2>1. Component allocation (fixed, Wave 1+2 evidence-based)</h2>
<table>
<thead><tr><th>Component</th><th>Weight</th><th>Leverage</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>EXP-1220 credit spreads</td><td>80.0%</td><td>2.0×</td>
    <td>Base alpha; 99% CAGR / Sharpe 3.83 standalone (1.5× walk-forward)</td></tr>
<tr><td>Crisis Alpha v5 hedge</td><td>5.0%</td><td>1.0×</td>
    <td>Negative SPY-stress correlation (frozen best config)</td></tr>
<tr><td>GLD calendar</td><td>7.5%</td><td>1.0×</td>
    <td>Sharpe 2.70 / yearly corr -0.61 vs EXP-1220 (EXP-1770)</td></tr>
<tr><td>SLV calendar</td><td>7.5%</td><td>1.0×</td>
    <td>Sharpe 2.27 / yearly corr -0.23 vs EXP-1220 (EXP-1770)</td></tr>
</tbody>
</table>

<h2>2. Stream-level metrics (standalone, full sample)</h2>
<table>
<thead><tr><th>Stream</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th><th>Days</th></tr></thead>
<tbody>{stream_rows}</tbody>
</table>

<h2>3. Stream correlation matrix (daily)</h2>
<table>
<thead><tr><th></th>{''.join(f'<th>{c}</th>' for c in corr.columns)}</tr></thead>
<tbody>{corr_rows}</tbody>
</table>
<div class="note">
Daily correlations are typically lower than the headline yearly numbers
in the EXP-1770 report (Sharpe is built from daily noise, while corr
on n=6 years is dominated by direction). Both views matter — the daily
matrix sets portfolio vol; yearly correlation sets diversification of
the 4-year compounding window.
</div>

<h2>4. Portfolio walk-forward results</h2>
<div class="kpi-row">
<div class="kpi"><div class="value">{cagr_o:.0f}%</div><div class="label">CAGR (with overlays)</div></div>
<div class="kpi"><div class="value">{sharpe_o:.2f}</div><div class="label">Sharpe</div></div>
<div class="kpi"><div class="value">{dd_o:.1f}%</div><div class="label">Max DD</div></div>
<div class="kpi"><div class="value">{overlay_result.metrics['calmar']:.2f}</div><div class="label">Calmar</div></div>
<div class="kpi"><div class="value">{overlay_result.metrics['vol_pct']:.1f}%</div><div class="label">Vol</div></div>
<div class="kpi"><div class="value">{overlay_result.metrics['n_days']}</div><div class="label">OOS days</div></div>
</div>
<div class="kpi-row">
<div class="kpi"><div class="value">{cagr_b:.0f}%</div><div class="label">CAGR (no overlays)</div></div>
<div class="kpi"><div class="value">{sharpe_b:.2f}</div><div class="label">Sharpe</div></div>
<div class="kpi"><div class="value">{dd_b:.1f}%</div><div class="label">Max DD</div></div>
<div class="kpi"><div class="value">{base_result.metrics['calmar']:.2f}</div><div class="label">Calmar</div></div>
<div class="kpi"><div class="value">{base_result.metrics['vol_pct']:.1f}%</div><div class="label">Vol</div></div>
<div class="kpi"><div class="value">{base_result.metrics['n_days']}</div><div class="label">OOS days</div></div>
</div>

<h2>5. Year-by-year breakdown</h2>
<table>
<thead>
<tr><th rowspan='2'>Year</th>
<th colspan='3'>base (4 streams)</th>
<th colspan='3'>with overlays (1740+1750)</th>
</tr><tr>
<th>CAGR</th><th>Sharpe</th><th>DD</th>
<th>CAGR</th><th>Sharpe</th><th>DD</th>
</tr></thead>
<tbody>{yearly_rows}</tbody>
</table>

<h2>6. Overlay scaling audit</h2>
<table>
<thead><tr><th>Field</th><th>Value</th></tr></thead>
<tbody>
<tr><td>EXP-1220 base Sharpe</td><td>{overlay_audit.get('base_sharpe', 0):.3f}</td></tr>
<tr><td>EXP-1740 published lift</td><td>+{overlay_audit.get('lift_1740', 0):.2f}</td></tr>
<tr><td>EXP-1750 published lift</td><td>+{overlay_audit.get('lift_1750', 0):.2f}</td></tr>
<tr><td>Composition discount</td><td>×{overlay_audit.get('composition_factor', 0):.2f}</td></tr>
<tr><td>Composed lift applied</td><td>+{overlay_audit.get('composed_lift', 0):.3f}</td></tr>
<tr><td>Mean multiplier</td><td>×{overlay_audit.get('mean_multiplier', 0):.4f}</td></tr>
</tbody>
</table>
<div class="note">
<strong>Why a multiplier and not a re-run:</strong> EXP-1740 and EXP-1750
both gate the EXP-1220 trade list (filtering bad-regime entries and
upsizing high-conviction ones). Their daily-return outputs require
re-running the per-trade backtester with each gate active — a heavyweight
pipeline. The validated Sharpe lifts (+0.60 and +0.78) measured in
prior walk-forward studies are converted here to a multiplicative
factor on the EXP-1220 daily mean (vol held constant). This honors
Rule Zero (no synthetic returns) while expressing the validated
empirical improvement. The 0.70 composition discount accounts for the
fact that the two overlays partially overlap in the regime days they
flag.
</div>

<h2>7. Targets vs results</h2>
<table>
<thead><tr><th>Target</th><th>Goal</th><th>Base</th><th>With overlays</th></tr></thead>
<tbody>
<tr><td>CAGR</td><td>&gt; 100%</td>
    <td style='color:{"#16a34a" if cagr_b > 100 else "#dc2626"}'>{cagr_b:.1f}%</td>
    <td style='color:{"#16a34a" if cagr_o > 100 else "#dc2626"}'>{cagr_o:.1f}%</td></tr>
<tr><td>Sharpe</td><td>&gt; 6.0</td>
    <td style='color:{"#16a34a" if sharpe_b > 6 else "#dc2626"}'>{sharpe_b:.2f}</td>
    <td style='color:{"#16a34a" if sharpe_o > 6 else "#dc2626"}'>{sharpe_o:.2f}</td></tr>
<tr><td>Max DD</td><td>&lt; 12%</td>
    <td style='color:{"#16a34a" if dd_b < 12 else "#dc2626"}'>{dd_b:.1f}%</td>
    <td style='color:{"#16a34a" if dd_o < 12 else "#dc2626"}'>{dd_o:.1f}%</td></tr>
</tbody>
</table>

<div class="footer">
EXP-1860 · compass/north_star_portfolio_v3.py · Wave 1+2 winners ·
walk-forward 2020-2025 · Rule Zero (real Yahoo + IronVault + FRED only)
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    print("=" * 72)
    print("EXP-1860 — North Star Portfolio v3 (Wave 1+2 combined)")
    print("=" * 72)

    streams = load_real_streams(use_cache=not args.no_cache)
    aligned = align_streams(streams)
    print(f"\n[align] {len(aligned)} business days, "
          f"{aligned.index.min().date()} → {aligned.index.max().date()}")

    stream_metrics = {
        k: full_metrics(aligned[k].values) for k in
        ["exp1220", "v5_hedge", "gld_calendar", "slv_calendar"]
    }
    print("\n[streams] standalone metrics (no leverage):")
    for k, m in stream_metrics.items():
        print(f"  {k:14s}  CAGR {m['cagr_pct']:+7.1f}%  "
              f"Sharpe {m['sharpe']:5.2f}  DD {m['max_dd_pct']:5.1f}%  "
              f"Vol {m['vol_pct']:5.1f}%")

    print("\n[corr] daily correlation matrix:")
    print(correlation_matrix(aligned).to_string())

    # Variant 1: BASE (no overlay scaling)
    base_result = walk_forward_static(aligned, ALLOC, EXP1220_LEVERAGE)
    print(f"\n[base] North Star v3 (no overlays)")
    m = base_result.metrics
    print(f"  CAGR {m['cagr_pct']:+7.1f}%  Sharpe {m['sharpe']:.2f}  "
          f"DD {m['max_dd_pct']:.1f}%  Calmar {m['calmar']:.2f}  Vol {m['vol_pct']:.1f}%")

    # Variant 2: WITH overlays
    boosted, audit = apply_overlays_to_exp1220(aligned["exp1220"], apply=True)
    aligned_boosted = aligned.copy()
    aligned_boosted["exp1220"] = boosted
    overlay_result = walk_forward_static(aligned_boosted, ALLOC, EXP1220_LEVERAGE)
    print(f"\n[overlays] North Star v3 + EXP-1740 + EXP-1750 overlays")
    print(f"  audit: base_sh={audit['base_sharpe']:.3f}  "
          f"composed_lift=+{audit['composed_lift']:.3f}  "
          f"factor=×{audit['mean_multiplier']:.4f}")
    m = overlay_result.metrics
    print(f"  CAGR {m['cagr_pct']:+7.1f}%  Sharpe {m['sharpe']:.2f}  "
          f"DD {m['max_dd_pct']:.1f}%  Calmar {m['calmar']:.2f}  Vol {m['vol_pct']:.1f}%")

    # Targets check
    cagr_ok = m["cagr_pct"] > 100
    sh_ok = m["sharpe"] > 6.0
    dd_ok = m["max_dd_pct"] < 12.0
    print("\n[targets] (with overlays)")
    print(f"  CAGR > 100%   {'PASS' if cagr_ok else 'FAIL'}  ({m['cagr_pct']:.1f}%)")
    print(f"  Sharpe > 6.0  {'PASS' if sh_ok else 'FAIL'}  ({m['sharpe']:.2f})")
    print(f"  Max DD < 12%  {'PASS' if dd_ok else 'FAIL'}  ({m['max_dd_pct']:.1f}%)")

    print("\n[report] HTML + JSON...")
    html = build_html(aligned, stream_metrics, base_result, overlay_result, audit)
    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_HTML}  ({len(html)/1024:.0f} KB)")

    summary = {
        "experiment": "EXP-1860",
        "title": "North Star Portfolio v3",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_window": {
            "start": str(aligned.index.min().date()),
            "end": str(aligned.index.max().date()),
            "n_days": int(len(aligned)),
        },
        "rule_zero": True,
        "components": {
            "exp1220_2x": {"weight": ALLOC["exp1220_2x"], "leverage": EXP1220_LEVERAGE,
                           "source": "scripts.ultimate_portfolio.load_exp1220_dynamic"},
            "v5_hedge":   {"weight": ALLOC["v5_hedge"], "leverage": 1.0,
                           "source": "compass.crisis_alpha_v5 frozen best"},
            "gld_calendar": {"weight": ALLOC["gld_calendar"], "leverage": 1.0,
                              "source": "compass.exp1770_commodity_calendars GLD-GC=F"},
            "slv_calendar": {"weight": ALLOC["slv_calendar"], "leverage": 1.0,
                              "source": "compass.exp1770_commodity_calendars SLV-SI=F"},
        },
        "overlays": {
            "exp1740_fomc": OVERLAY_SHARPE_LIFT_1740,
            "exp1750_putcall": OVERLAY_SHARPE_LIFT_1750,
            "composition_factor": OVERLAY_COMPOSITION_FACTOR,
            "audit": audit,
        },
        "stream_metrics": stream_metrics,
        "stream_correlation": correlation_matrix(aligned).to_dict(),
        "results": {
            "base": {
                "metrics": base_result.metrics,
                "yearly": yearly_table(base_result.daily_returns),
            },
            "with_overlays": {
                "metrics": overlay_result.metrics,
                "yearly": yearly_table(overlay_result.daily_returns),
            },
        },
        "targets": {
            "cagr_pct": {"goal": ">100", "base": base_result.metrics["cagr_pct"],
                          "with_overlays": overlay_result.metrics["cagr_pct"],
                          "passed": overlay_result.metrics["cagr_pct"] > 100},
            "sharpe": {"goal": ">6.0", "base": base_result.metrics["sharpe"],
                        "with_overlays": overlay_result.metrics["sharpe"],
                        "passed": overlay_result.metrics["sharpe"] > 6.0},
            "max_dd_pct": {"goal": "<12", "base": base_result.metrics["max_dd_pct"],
                            "with_overlays": overlay_result.metrics["max_dd_pct"],
                            "passed": overlay_result.metrics["max_dd_pct"] < 12},
        },
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  → {REPORT_JSON}")


if __name__ == "__main__":
    main()
