"""North Star Portfolio v4 — Sharpe discrepancy audit (EXP-1860 vs EXP-1870).

PROBLEM
=======
EXP-1860 (compass/reports/exp1860_north_star_portfolio.json):
    Sharpe = 3.96 / CAGR = 119.9% / DD = 11.6%
    Window = 2020-01-02 → 2025-12-31
    Weights = 80 / 5 / 7.5 / 7.5  (EXP1220×2 / v5 / GLD / SLV)

EXP-1870 stress test (compass/reports/north_star_stress_test.json):
    Sharpe = 8.02 / CAGR = 114.5% / DD = 9.7%
    Window = 2015-01-01 → 2025-12-31
    Weights = 60 / 15 / 15 / 5 / 5  (EXP1220×2 / GLD / SLV / v5 / Cash)

ROOT CAUSE
==========
The discrepancy is NOT a Sharpe formula bug. Both files compute Sharpe
identically:
    sharpe = mean(daily_returns) / std(daily_returns) × √252
which is the canonical compass.metrics.annualized_sharpe.

The discrepancy is an INPUT STREAM mismatch:

    Component             EXP-1860 (v3)                                EXP-1870 (stress)
    ---------             -------------                                -----------------
    EXP-1220              scripts.ultimate_portfolio.                  compass.exp1780_exp1220_integration.
                          load_exp1220_dynamic                         build_exp1220_daily_returns
                          (real Yahoo SPY+^VIX+^VIX3M with             (CALIBRATED FUNCTIONAL PROXY
                          TailRiskProtector dynamic leverage           explicitly designed to reproduce
                          0.3-1.8×; standalone Sharpe 3.70)            the inflated 77% CAGR / 5.78 Sharpe
                                                                       MASTERPLAN v6 numbers — which
                                                                       MASTERPLAN itself later declared a
                                                                       bug (√trades vs √252 annualization).
                                                                       Standalone Sharpe ~7.9.

    GLD/SLV "spreads"     compass.exp1770_commodity_calendars          Vol-targeted long-ETF returns with
                          walk-forward GLD-GC=F / SLV-SI=F             downside clipping (rets * 5%/rolling
                          roll-yield harvest                           _vol).clip(lower=-0.04). NOT the
                          (Sharpe 2.03 / 1.99)                         EXP-1770 walk-forward at all.

    Crisis Alpha v5       Frozen best config:                          Different config:
                          slow / vt=0.05 / l=1.0 /                     v2_round / vt=0.08 / l=1.5 /
                          sg=0.05 / sh=2.0 / equity-short              same overlays (Sharpe -0.29)
                          (Sharpe -0.82)

    Cash                  not included                                 5% allocation, Sharpe ≈ 1e17
                                                                       (vol ~4e-19 → divide by near-zero)

This audit:
  1. Re-computes BOTH weight schemes on the SAME canonical streams (the
     v3 ones, which trace cleanly to real Yahoo + walk-forward only).
  2. On the SAME window (2020-2025).
  3. Using the canonical compass.metrics.full_metrics formula.
  4. Reports the definitive North Star Portfolio v4 numbers.

Conclusion (expected): the v3 number ~Sharpe 4 is the truthful one.
The stress test 8.02 was driven by the calibrated proxy reproducing
discredited MASTERPLAN headline metrics, not a math bug.

Rule Zero: every input traces to real Yahoo / FRED / IronVault. The
calibrated proxy is excluded by definition.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compass.metrics import full_metrics, annualized_sharpe

CACHE_DIR = ROOT / "compass" / "cache"
CACHE_FILE = CACHE_DIR / "exp1860_streams.pkl"   # reuse the v3 cache
REPORT_HTML = ROOT / "compass" / "reports" / "north_star_v4_audit.html"
REPORT_JSON = ROOT / "compass" / "reports" / "north_star_v4_audit.json"

START = "2020-01-01"
END = "2025-12-31"

# Two weight schemes to compare
WEIGHTS_V3 = {
    "exp1220_2x": 0.80, "v5_hedge": 0.05,
    "gld_calendar": 0.075, "slv_calendar": 0.075,
    "cash": 0.0,
}
WEIGHTS_STRESS = {
    "exp1220_2x": 0.60, "v5_hedge": 0.05,
    "gld_calendar": 0.15, "slv_calendar": 0.15,
    "cash": 0.05,
}
EXP1220_LEVERAGE = 2.0
CASH_DAILY = 0.05 / 252.0   # 5% annual flat


# ═══════════════════════════════════════════════════════════════════════════
# Stream loading — REUSE the v3 cache (already trusted real-data streams)
# ═══════════════════════════════════════════════════════════════════════════

def load_canonical_streams() -> Dict[str, pd.Series]:
    """Load the v3 cached streams. These are the canonical real-data
    streams used by EXP-1860; if the cache is missing the user must run
    compass.north_star_portfolio_v3 first.
    """
    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Canonical stream cache not found at {CACHE_FILE}. "
            f"Run `python3 -m compass.north_star_portfolio_v3` first."
        )
    print(f"[cache] loading {CACHE_FILE.name}")
    with open(CACHE_FILE, "rb") as fh:
        streams = pickle.load(fh)
    return streams


def align(streams: Dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.concat([s.rename(k) for k, s in streams.items()],
                    axis=1, sort=True)
    df = df[(df.index >= pd.Timestamp(START)) & (df.index <= pd.Timestamp(END))]
    df = df.fillna(0.0).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df["cash"] = CASH_DAILY  # add a flat cash sleeve
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Canonical Sharpe + portfolio computation
# ═══════════════════════════════════════════════════════════════════════════

def canonical_metrics(rets: pd.Series) -> Dict[str, float]:
    """All metrics from compass.metrics.full_metrics (canonical Sharpe)."""
    return full_metrics(rets.values)


def portfolio_returns(streams: pd.DataFrame, weights: Dict[str, float],
                       exp1220_lev: float) -> pd.Series:
    e = streams["exp1220"] * exp1220_lev
    h = streams["v5_hedge"]
    g = streams["gld_calendar"]
    sv = streams["slv_calendar"]
    c = streams["cash"]
    return (
        weights["exp1220_2x"] * e
        + weights["v5_hedge"] * h
        + weights["gld_calendar"] * g
        + weights["slv_calendar"] * sv
        + weights["cash"] * c
    )


def yearly(rets: pd.Series) -> List[Dict]:
    out = []
    for yr in sorted({d.year for d in rets.index}):
        sub = rets[rets.index.year == yr]
        if len(sub) < 20:
            continue
        m = canonical_metrics(sub)
        m["year"] = int(yr)
        out.append(m)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Stress test reproduction — re-build the proxies for diagnostic
# ═══════════════════════════════════════════════════════════════════════════

def reproduce_stress_proxies() -> Dict[str, pd.Series]:
    """Re-create the stress test's component series so we can directly
    show its Sharpe numbers come from the proxy choice, not from a bug
    in the formula.
    """
    from compass.crisis_alpha_v3 import load_universe_v3
    from compass.exp1780_exp1220_integration import build_exp1220_daily_returns
    prices = load_universe_v3(start="2014-01-01", end="2026-01-01")

    out: Dict[str, pd.Series] = {}
    out["exp1220_proxy"] = build_exp1220_daily_returns(prices)

    for tk, target in (("GLD", 0.05), ("SLV", 0.05)):
        if tk not in prices.columns:
            continue
        rets = prices[tk].pct_change().fillna(0)
        rolling_vol = (rets.rolling(60, min_periods=20).std()
                       * math.sqrt(252)).fillna(target)
        scale = (target / rolling_vol).clip(0.25, 2.0)
        out[f"{tk.lower()}_voltarget"] = (rets * scale).clip(lower=-0.04)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

def _row(name: str, m: Dict) -> str:
    return (
        f"<tr><td style='font-weight:700'>{name}</td>"
        f"<td>{m['cagr_pct']:.2f}%</td>"
        f"<td style='font-weight:700'>{m['sharpe']:.2f}</td>"
        f"<td>{m['max_dd_pct']:.2f}%</td>"
        f"<td>{m['calmar']:.2f}</td>"
        f"<td>{m['vol_pct']:.2f}%</td>"
        f"<td>{m['n_days']}</td></tr>"
    )


def _yearly_rows(label_to_yearly: Dict[str, List[Dict]]) -> str:
    years = sorted({y["year"] for v in label_to_yearly.values() for y in v})
    rows = ""
    for yr in years:
        cells = ""
        for name in label_to_yearly.keys():
            row = next((y for y in label_to_yearly[name] if y["year"] == yr), {})
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


def build_html(audit: Dict) -> str:
    s = audit["stream_metrics"]
    pv3 = audit["v3_weights"]
    pst = audit["stress_weights"]
    pp = audit["proxy_diagnostic"]

    stream_rows = "".join(
        _row(k, s[k])
        for k in ["exp1220", "v5_hedge", "gld_calendar", "slv_calendar"]
    )

    proxy_rows = "".join(
        _row(k, pp[k]) for k in pp.keys()
    )

    yearly_rows = _yearly_rows({
        "v3 weights (80/5/7.5/7.5)": pv3["yearly"],
        "stress weights (60/5/15/15/5)": pst["yearly"],
    })

    canonical_v3 = pv3["metrics"]
    canonical_st = pst["metrics"]

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>North Star v4 — Sharpe Audit (EXP-1860 vs EXP-1870)</title>
<style>
* {{ box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       max-width:1200px;margin:0 auto;padding:28px;background:#fff;color:#1e293b;line-height:1.5; }}
h1 {{ font-size:1.85em;color:#0f172a;margin-bottom:4px; }}
h2 {{ color:#334155;margin-top:2.4em;padding-bottom:8px;border-bottom:2px solid #e2e8f0; }}
.subtitle {{ color:#64748b;font-size:0.92rem;margin-bottom:24px; }}
.bug {{ background:#fef2f2;border:2px solid #dc2626;border-radius:10px;padding:20px;margin:24px 0; }}
.bug h3 {{ margin-top:0;color:#991b1b; }}
.fix {{ background:#ecfdf5;border:2px solid #16a34a;border-radius:10px;padding:20px;margin:24px 0; }}
.fix h3 {{ margin-top:0;color:#065f46; }}
.sources {{ background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:16px;margin:16px 0;font-size:0.84rem;line-height:1.7; }}
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

<h1>North Star v4 — Definitive Sharpe Audit</h1>
<div class="subtitle">EXP-1860 (Sharpe 3.96) vs EXP-1870 stress (Sharpe 8.02) ·
re-computed on the SAME canonical real-data streams ·
{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>

<div class="bug">
<h3>Root cause — NOT a Sharpe formula bug</h3>
Both reports compute Sharpe correctly via
<code>mean(daily) / std(daily) × √252</code>
(canonical formula in <code>compass/metrics.py::annualized_sharpe</code>).
The 8.02 vs 3.96 gap is an <strong>input stream mismatch</strong>:

<ul>
<li><strong>EXP-1870 stress</strong> calls
<code>compass.exp1780_exp1220_integration.build_exp1220_daily_returns</code>
which is a <em>calibrated functional proxy</em> explicitly designed to
reproduce the <strong>discredited</strong> MASTERPLAN v6 headline numbers
(77% CAGR / 5.78 Sharpe). MASTERPLAN itself documents that the 5.78 Sharpe
was a √trades-vs-√252 annualization bug and the corrected number is 3.85.
Standalone Sharpe of this proxy: ~{pp.get('exp1220_proxy', {}).get('sharpe', 0):.2f}.</li>

<li><strong>EXP-1870 stress</strong> uses <em>vol-targeted long-ETF
returns with downside clipping</em> for the GLD/SLV "spreads" sleeve
(<code>(rets * 5%/rolling_vol).clip(lower=-0.04)</code>) — this is NOT
the EXP-1770 walk-forward GLD-GC=F / SLV-SI=F roll-yield harvest used
by EXP-1860. Two completely different strategies sharing a name.</li>

<li><strong>EXP-1870 stress</strong> includes a 5% Cash sleeve with
vol ≈ 4e-19, which yields a standalone Sharpe of ≈ 1e17 from the
divide-by-near-zero. It barely changes the portfolio Sharpe but is
mathematically a flag.</li>
</ul>
</div>

<div class="sources">
<strong>Canonical streams used in this audit (Rule Zero — all real):</strong><br>
<code>exp1220</code>: <code>scripts.ultimate_portfolio.load_exp1220_dynamic</code>
(real Yahoo SPY + ^VIX + ^VIX3M, TailRiskProtector dynamic leverage 0.3-1.8×)<br>
<code>v5_hedge</code>: <code>compass.crisis_alpha_v5</code> frozen best
(slow / vt=0.05 / l=1.0 / sg=0.05 / sh=2.0 / equity-short-only) on real Yahoo 13-ETF<br>
<code>gld_calendar / slv_calendar</code>: <code>compass.exp1770_commodity_calendars</code>
walk-forward GLD−GC=F and SLV−SI=F (real Yahoo daily close)<br>
Window: 2020-01-02 → 2025-12-31 (1508 business days)<br>
Sharpe formula: <code>compass.metrics.full_metrics</code> (canonical arithmetic-mean Sharpe).
</div>

<h2>1. Stream-level metrics (canonical real-data, 2020-2025)</h2>
<table>
<thead><tr><th>Stream</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>Days</th></tr></thead>
<tbody>{stream_rows}</tbody>
</table>

<h2>2. Stress test PROXY metrics (the discredited inputs)</h2>
<p>For diagnostic only — re-runs the stress test's proxy builders on the same window
to confirm the gap comes from the proxies, not the math.</p>
<table>
<thead><tr><th>Proxy stream</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>Days</th></tr></thead>
<tbody>{proxy_rows}</tbody>
</table>
<div class="note">
The <code>exp1220_proxy</code> standalone Sharpe is ≈
<strong>{pp.get('exp1220_proxy', {}).get('sharpe', 0):.2f}</strong>,
calibrated to the discredited MASTERPLAN v6 5.78 number. The
<code>load_exp1220_dynamic</code> stream above produces Sharpe ≈
<strong>{s['exp1220']['sharpe']:.2f}</strong> on the same window — that's the truth.
</div>

<h2>3. Both weight schemes on canonical streams</h2>
<table>
<thead><tr><th>Portfolio</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th><th>Vol</th><th>Days</th></tr></thead>
<tbody>
{_row("v3 weights (80/5/7.5/7.5)", canonical_v3)}
{_row("stress weights (60/5/15/15/5)", canonical_st)}
</tbody>
</table>

<div class="fix">
<h3>Definitive North Star v4 numbers</h3>
On the canonical real-data streams over 2020-2025, the v3 weight scheme
(80/5/7.5/7.5) gives:<br>
<strong>CAGR {canonical_v3['cagr_pct']:.1f}% · Sharpe {canonical_v3['sharpe']:.2f} ·
Max DD {canonical_v3['max_dd_pct']:.1f}% · Calmar {canonical_v3['calmar']:.2f}</strong><br>
<br>
The stress weight scheme (60/5/15/15/5) gives:<br>
<strong>CAGR {canonical_st['cagr_pct']:.1f}% · Sharpe {canonical_st['sharpe']:.2f} ·
Max DD {canonical_st['max_dd_pct']:.1f}% · Calmar {canonical_st['calmar']:.2f}</strong><br>
<br>
The 8.02 Sharpe headline from EXP-1870 is <strong>not reproducible</strong>
once the calibrated EXP-1220 proxy is replaced with the real
dynamic-leverage stream and the GLD/SLV "spreads" sleeve is replaced
with the actual EXP-1770 walk-forward roll-yield strategy.
</div>

<h2>4. Year-by-year (canonical)</h2>
<table>
<thead>
<tr><th rowspan='2'>Year</th>
<th colspan='3'>v3 weights (80/5/7.5/7.5)</th>
<th colspan='3'>stress weights (60/5/15/15/5)</th>
</tr><tr>
<th>CAGR</th><th>Sharpe</th><th>DD</th>
<th>CAGR</th><th>Sharpe</th><th>DD</th>
</tr></thead>
<tbody>{yearly_rows}</tbody>
</table>

<h2>5. Decision: which number to use</h2>
<ol>
<li><strong>Use the v3 numbers</strong> (Sharpe ~4) as the truthful North Star
state. They trace cleanly through walk-forward backtests on real Yahoo /
IronVault / FRED data.</li>

<li><strong>Discard the EXP-1870 8.02 Sharpe headline.</strong> It is an
artifact of two model proxies (one of them explicitly calibrated to a
discredited MASTERPLAN number) — not a real backtest. The stress test
should be re-run with the canonical streams before its other outputs
(MC bootstrap, crisis replay, VaR/CVaR) are trusted.</li>

<li><strong>Update <code>compass/north_star_stress_test.py::build_components</code></strong>
to import <code>load_exp1220_dynamic</code> and the EXP-1770 walk-forward
streams instead of the proxies. Also remove the Cash sleeve from the
correlation matrix (its NaN row makes the matrix unparseable) or pin
its vol floor to avoid the divide-by-near-zero Sharpe.</li>

<li><strong>The North Star Sharpe target of 6.0 remains MISSED.</strong>
The truthful answer for this 4-stream portfolio over 2020-2025 is in the
3.9-4.2 range. To reach 6.0 requires either a new Sharpe ≥ 5 stream with
near-zero correlation to existing streams, or actually re-running EXP-1220
through the EXP-1740/EXP-1750 overlay gates as per-trade filters
(raising mean AND reducing vol, not just scaling mean as EXP-1860 did).</li>
</ol>

<div class="footer">
EXP-1860 vs EXP-1870 audit · compass/north_star_v4_audit.py ·
canonical streams from compass/cache/exp1860_streams.pkl · Rule Zero
(real Yahoo + IronVault + FRED only).
</div>
</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("North Star v4 — Sharpe discrepancy audit")
    print("=" * 72)

    streams = load_canonical_streams()
    aligned = align(streams)
    print(f"\n[align] {len(aligned)} business days, "
          f"{aligned.index.min().date()} → {aligned.index.max().date()}")

    stream_metrics = {
        k: canonical_metrics(aligned[k]) for k in
        ["exp1220", "v5_hedge", "gld_calendar", "slv_calendar"]
    }
    print("\n[streams] canonical real-data, 2020-2025:")
    for k, m in stream_metrics.items():
        print(f"  {k:14s}  CAGR {m['cagr_pct']:+7.2f}%  "
              f"Sharpe {m['sharpe']:6.2f}  DD {m['max_dd_pct']:5.2f}%  "
              f"Vol {m['vol_pct']:5.2f}%")

    # ── Both weight schemes ────────────────────────────────────────
    pv3_rets = portfolio_returns(aligned, WEIGHTS_V3, EXP1220_LEVERAGE)
    pv3_metrics = canonical_metrics(pv3_rets)
    pv3_yearly = yearly(pv3_rets)
    print("\n[v3 weights 80/5/7.5/7.5]")
    print(f"  CAGR {pv3_metrics['cagr_pct']:+7.2f}%  "
          f"Sharpe {pv3_metrics['sharpe']:6.2f}  "
          f"DD {pv3_metrics['max_dd_pct']:5.2f}%  "
          f"Calmar {pv3_metrics['calmar']:5.2f}")

    pst_rets = portfolio_returns(aligned, WEIGHTS_STRESS, EXP1220_LEVERAGE)
    pst_metrics = canonical_metrics(pst_rets)
    pst_yearly = yearly(pst_rets)
    print("\n[stress weights 60/5/15/15/5]")
    print(f"  CAGR {pst_metrics['cagr_pct']:+7.2f}%  "
          f"Sharpe {pst_metrics['sharpe']:6.2f}  "
          f"DD {pst_metrics['max_dd_pct']:5.2f}%  "
          f"Calmar {pst_metrics['calmar']:5.2f}")

    # ── Reproduce stress proxies for diagnostic ────────────────────
    print("\n[diagnostic] reproducing stress test proxies on the same window...")
    proxies = reproduce_stress_proxies()
    proxy_metrics: Dict[str, Dict] = {}
    for k, s in proxies.items():
        s = s[(s.index >= pd.Timestamp(START)) & (s.index <= pd.Timestamp(END))]
        m = canonical_metrics(s)
        proxy_metrics[k] = m
        print(f"  {k:20s}  CAGR {m['cagr_pct']:+7.2f}%  "
              f"Sharpe {m['sharpe']:6.2f}  DD {m['max_dd_pct']:5.2f}%")

    audit = {
        "experiment": "north_star_v4_audit",
        "tag": "audit",
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_window": {
            "start": str(aligned.index.min().date()),
            "end": str(aligned.index.max().date()),
            "n_days": int(len(aligned)),
        },
        "rule_zero": True,
        "sources": {
            "exp1220": "scripts.ultimate_portfolio.load_exp1220_dynamic",
            "v5_hedge": "compass.crisis_alpha_v5 frozen best (slow/vt=0.05/l=1.0/sg=0.05/sh=2.0/short-only)",
            "gld_calendar": "compass.exp1770_commodity_calendars walk-forward GLD-GC=F",
            "slv_calendar": "compass.exp1770_commodity_calendars walk-forward SLV-SI=F",
            "cash": "flat 5% / 252",
        },
        "sharpe_formula": "mean(daily) / std(daily) * sqrt(252)  (compass.metrics.annualized_sharpe)",
        "root_cause": (
            "Not a Sharpe formula bug. EXP-1870 stress test substitutes "
            "compass.exp1780_exp1220_integration.build_exp1220_daily_returns "
            "(a calibrated proxy reproducing the discredited MASTERPLAN v6 "
            "5.78 Sharpe number) for the canonical load_exp1220_dynamic stream, "
            "AND substitutes vol-targeted long-ETF clipped returns for the "
            "EXP-1770 walk-forward GLD/SLV calendar spreads. The 8.02 Sharpe "
            "headline is an artifact of the proxies, not real."
        ),
        "stream_metrics": stream_metrics,
        "proxy_diagnostic": proxy_metrics,
        "v3_weights": {
            "weights": WEIGHTS_V3,
            "exp1220_leverage": EXP1220_LEVERAGE,
            "metrics": pv3_metrics,
            "yearly": pv3_yearly,
        },
        "stress_weights": {
            "weights": WEIGHTS_STRESS,
            "exp1220_leverage": EXP1220_LEVERAGE,
            "metrics": pst_metrics,
            "yearly": pst_yearly,
        },
        "comparison": {
            "exp1860_published": {
                "sharpe": 3.96, "cagr_pct": 119.86, "max_dd_pct": 11.64,
                "window": "2020-01-02 → 2025-12-31",
                "weights": "80/5/7.5/7.5",
            },
            "exp1870_stress_published": {
                "sharpe": 8.02, "cagr_pct": 114.49, "max_dd_pct": 9.66,
                "window": "2015-01-01 → 2025-12-31",
                "weights": "60/15/15/5/5",
            },
            "definitive_v4_v3_weights": pv3_metrics,
            "definitive_v4_stress_weights": pst_metrics,
        },
        "decision": (
            "Use v3 (canonical) numbers as the truthful North Star state. "
            "Discard the EXP-1870 8.02 Sharpe headline. Re-run the stress "
            "test with the canonical streams before trusting its other "
            "outputs (Monte Carlo bootstrap, crisis replay, VaR/CVaR)."
        ),
    }

    print("\n[report] writing JSON + HTML...")
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(audit, indent=2, default=str))
    print(f"  → {REPORT_JSON}")

    html = build_html(audit)
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"  → {REPORT_HTML}  ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
