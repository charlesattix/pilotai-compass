"""
compass/exp2190_tail_risk_parity.py — EXP-2190 Portfolio Insurance via Tail-Risk Parity.

HYPOTHESIS:
  A fixed 5% Crisis Alpha allocation is a blunt instrument. A dynamic
  tail-risk-parity overlay that SCALES UP the Crisis Alpha sleeve when
  stress signals fire, and SCALES DOWN when they are quiet, should
  reduce max drawdown without a proportional reduction in mean return —
  i.e., improve Sharpe from the denominator (vol) side.

STRESS TRIGGERS (any one fires → scale up tail hedge):
  (a) VIX term-structure inversion   — ^VIX > ^VIX3M (real Yahoo data)
  (b) Portfolio drawdown > 3%        — peak-to-current on the 5-stream portfolio
  (c) Cross-stream correlation spike — 20-day rolling mean pairwise correlation
                                         across the 4 alpha streams (excluding
                                         hedge) > 0.60

DYNAMIC SIZING:
  base_hedge_weight = 0.05            — 5% floor (always some tail insurance)
  max_hedge_weight  = 0.25            — 25% cap (never blow the budget)
  stress_count      = # triggers firing on a given day ∈ {0,1,2,3}
  target_hedge_w    = 0.05 + stress_count × 0.07     (0→5%, 1→12%, 2→19%, 3→26→25%)

  Hedge sleeve is funded proportionally from the alpha streams, preserving
  their RELATIVE weights. All weights re-normalize to sum to 1.

PORTFOLIO BASE (5 streams at 3× overall leverage):
  exp1220, gld_cal, slv_cal, cross_vol  — alpha sleeves (weighted 60/7.5/7.5/15)
  v5_hedge                              — Crisis Alpha v5 tail hedge (5% baseline)
  Overall 3× leverage applied uniformly.

DATA:
  compass/cache/exp2080_streams.pkl — cached daily return streams from
     EXP-2080, 2020-01-01 → 2025-12-31, all real (EXP-1220 canonical,
     EXP-1770 GLD/SLV calendars walk-forward, cross-vol arb, Crisis
     Alpha v5 hedge). No synthetic data.
  Yahoo ^VIX and ^VIX3M — real daily closes for term-structure trigger.

WALK-FORWARD:
  2020-01 → 2022-12 : in-sample calibration window (used for reference)
  2023-01 → 2025-12 : out-of-sample test window
  The overlay itself has NO trained parameters — thresholds are fixed
  a-priori. The walk-forward serves as a robustness audit: we report
  both full-sample and OOS-only metrics.

OUTPUTS:
  compass/reports/exp2190_tail_risk_parity.{json,html}

Run::
    python3 -m compass.exp2190_tail_risk_parity
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STREAMS_PKL = ROOT / "compass" / "cache" / "exp2080_streams.pkl"
REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2190_tail_risk_parity.json"
REPORT_HTML = REPORT_DIR / "exp2190_tail_risk_parity.html"

TRADING_DAYS = 252

# Portfolio baseline weights (alpha sleeves only; hedge added on top)
ALPHA_WEIGHTS = {
    "exp1220":   0.60,
    "gld_cal":   0.075,
    "slv_cal":   0.075,
    "cross_vol": 0.15,
}
ALPHA_TOTAL = sum(ALPHA_WEIGHTS.values())   # 0.90
BASE_HEDGE_WEIGHT = 0.05                     # the 10% residual is cash buffer
GROSS_LEVERAGE = 3.0

# Dynamic sizing
MAX_HEDGE_WEIGHT = 0.25
HEDGE_STEP = 0.07                            # added per firing trigger
DD_TRIGGER = 0.03                            # 3%
CORR_TRIGGER = 0.60
CORR_WINDOW = 20

# Walk-forward windows
IS_START = "2020-01-01"
IS_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2025-12-31"


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def load_streams() -> pd.DataFrame:
    print(f"  loading {STREAMS_PKL.name} (real cached streams)...")
    with STREAMS_PKL.open("rb") as fh:
        df = pickle.load(fh)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected DataFrame, got {type(df)}")
    cols = ["exp1220", "v5_hedge", "gld_cal", "slv_cal", "cross_vol"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"{c} missing from streams pkl")
    df = df[cols].fillna(0.0).astype(float)
    print(f"    {len(df)} days  {df.index.min().date()} → {df.index.max().date()}")
    return df


def fetch_yahoo_close(symbol: str, start: str, end: str) -> pd.Series:
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    safe = symbol.replace("^", "%5E")
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{safe}"
           f"?period1={start_ts}&period2={end_ts}&interval=1d")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    dates = [datetime.fromtimestamp(t).date() for t in ts]
    s = pd.Series(closes, index=pd.DatetimeIndex(dates), name=symbol).dropna()
    return s[~s.index.duplicated(keep="last")]


def load_vix_term(start: str, end: str) -> pd.DataFrame:
    print(f"  loading ^VIX and ^VIX3M (Yahoo, real)...")
    vix = fetch_yahoo_close("^VIX", start, end).rename("vix")
    vix3m = fetch_yahoo_close("^VIX3M", start, end).rename("vix3m")
    df = pd.concat([vix, vix3m], axis=1).dropna()
    df["inverted"] = (df["vix"] > df["vix3m"]).astype(int)
    print(f"    {len(df)} days, inverted on {int(df['inverted'].sum())} days")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Base (unhedged-dynamic) portfolio
# ═══════════════════════════════════════════════════════════════════════════

def base_portfolio_returns(df: pd.DataFrame) -> pd.Series:
    """Static 5-stream baseline, 3× leverage, fixed 5% hedge."""
    alpha = sum(df[k] * w for k, w in ALPHA_WEIGHTS.items())
    port = alpha + df["v5_hedge"] * BASE_HEDGE_WEIGHT
    return (port * GROSS_LEVERAGE).rename("baseline")


# ═══════════════════════════════════════════════════════════════════════════
# Stress triggers
# ═══════════════════════════════════════════════════════════════════════════

def compute_drawdown(rets: pd.Series) -> pd.Series:
    eq = (1.0 + rets).cumprod()
    peak = eq.cummax()
    return (eq - peak) / peak


def rolling_mean_pairwise_corr(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling mean of the off-diagonal correlation entries across alpha streams."""
    cols = list(ALPHA_WEIGHTS.keys())
    n = len(cols)
    pair_count = n * (n - 1) // 2

    out = pd.Series(np.nan, index=df.index)
    sub = df[cols]
    for i in range(window, len(sub) + 1):
        win = sub.iloc[i - window:i]
        if win.shape[0] < window:
            continue
        try:
            c = win.corr().values
        except Exception:
            continue
        # off-diagonal mean
        mask = ~np.eye(n, dtype=bool)
        off = c[mask]
        off = off[~np.isnan(off)]
        if len(off) > 0:
            out.iloc[i - 1] = float(off.mean())
    return out


def build_trigger_panel(df: pd.DataFrame,
                          vix_term: pd.DataFrame,
                          baseline: pd.Series) -> pd.DataFrame:
    """Per-day trigger state. Each trigger uses only information that is
    available at the CLOSE of day t (so sizing can take effect on day t+1).
    """
    idx = df.index
    panel = pd.DataFrame(index=idx)

    # (a) VIX term inversion
    panel["inverted"] = vix_term["inverted"].reindex(idx, method="ffill").fillna(0).astype(int)

    # (b) Portfolio drawdown trigger — from the BASELINE equity curve
    dd = compute_drawdown(baseline)
    panel["dd"] = dd
    panel["dd_trigger"] = (dd <= -DD_TRIGGER).astype(int)

    # (c) Cross-stream correlation spike
    corr_ts = rolling_mean_pairwise_corr(df, CORR_WINDOW)
    panel["mean_corr"] = corr_ts
    panel["corr_trigger"] = (corr_ts >= CORR_TRIGGER).astype(int)

    panel["stress_count"] = (
        panel["inverted"] + panel["dd_trigger"] + panel["corr_trigger"]
    )
    panel["target_hedge_w"] = np.minimum(
        MAX_HEDGE_WEIGHT,
        BASE_HEDGE_WEIGHT + panel["stress_count"] * HEDGE_STEP,
    )
    return panel


# ═══════════════════════════════════════════════════════════════════════════
# Overlay portfolio
# ═══════════════════════════════════════════════════════════════════════════

def overlay_portfolio_returns(df: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Apply the dynamic hedge weight with 1-day lag (no look-ahead).

    On each day t, target_hedge_w is read from panel.shift(1) so it was
    knowable at t-1 close. Alpha weights are rescaled so the total alpha
    share is (1 - hedge_w), preserving their RELATIVE proportions.
    """
    hedge_w = panel["target_hedge_w"].shift(1).fillna(BASE_HEDGE_WEIGHT)
    alpha_share = 1.0 - hedge_w
    # Each alpha stream's weight scales down proportionally from its
    # baseline share of ALPHA_TOTAL.
    alpha_rets = pd.Series(0.0, index=df.index)
    for k, w in ALPHA_WEIGHTS.items():
        alpha_rets = alpha_rets.add(df[k] * (w / ALPHA_TOTAL), fill_value=0.0)
    port = alpha_share * alpha_rets + hedge_w * df["v5_hedge"]
    return (port * GROSS_LEVERAGE).rename("overlay")


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(rets: pd.Series, label: str) -> Dict:
    r = rets.dropna()
    n = len(r)
    if n < 10:
        return {"label": label, "n_days": n, "cagr_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "vol_pct": 0.0, "hit_rate_pct": 0.0,
                "sortino": 0.0}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = (1.0 + r).cumprod()
    yrs = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    dd = (eq - eq.cummax()) / eq.cummax()
    down = r[r < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else 0.0
    sortino = mu / ds * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0.0
    return {
        "label": label,
        "n_days": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(float(dd.min()) * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
        "hit_rate_pct": round(float((r > 0).mean()) * 100, 1),
        "mean_daily_bps": round(mu * 1e4, 2),
    }


def compare_window(baseline: pd.Series,
                     overlay: pd.Series,
                     start: str, end: str, label: str) -> Dict:
    b = baseline[(baseline.index >= start) & (baseline.index <= end)]
    o = overlay[(overlay.index >= start) & (overlay.index <= end)]
    bm = compute_metrics(b, f"{label}: baseline")
    om = compute_metrics(o, f"{label}: overlay")
    return {
        "window": {"label": label, "start": start, "end": end},
        "baseline": bm,
        "overlay": om,
        "delta": {
            "sharpe": round(om["sharpe"] - bm["sharpe"], 3),
            "cagr_pct": round(om["cagr_pct"] - bm["cagr_pct"], 3),
            "max_dd_pct": round(om["max_dd_pct"] - bm["max_dd_pct"], 3),
            "vol_pct": round(om["vol_pct"] - bm["vol_pct"], 3),
            "sortino": round(om["sortino"] - bm["sortino"], 3),
        },
    }


def trigger_firing_stats(panel: pd.DataFrame) -> Dict:
    n = len(panel)
    def frac(col):
        return round(float(panel[col].mean()) * 100, 2)
    stress_dist = panel["stress_count"].value_counts().sort_index().to_dict()
    return {
        "n_days": n,
        "pct_inverted": frac("inverted"),
        "pct_dd_trigger": frac("dd_trigger"),
        "pct_corr_trigger": frac("corr_trigger"),
        "stress_count_distribution": {str(int(k)): int(v) for k, v in stress_dist.items()},
        "mean_target_hedge_w": round(float(panel["target_hedge_w"].mean()), 4),
        "max_target_hedge_w": round(float(panel["target_hedge_w"].max()), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def render_html(payload: Dict) -> str:
    def row_cmp(cmp: Dict) -> str:
        w = cmp["window"]["label"]
        b = cmp["baseline"]
        o = cmp["overlay"]
        d = cmp["delta"]
        def arrow(v, good_if_positive=True):
            if v > 0:
                return "good" if good_if_positive else "bad"
            if v < 0:
                return "bad" if good_if_positive else "good"
            return ""
        return f"""<tr>
            <td><strong>{w}</strong></td>
            <td>{b['cagr_pct']:.2f}%</td>
            <td>{o['cagr_pct']:.2f}%</td>
            <td class="{arrow(d['cagr_pct'])}">{d['cagr_pct']:+.2f}%</td>
            <td>{b['sharpe']:.2f}</td>
            <td>{o['sharpe']:.2f}</td>
            <td class="{arrow(d['sharpe'])}">{d['sharpe']:+.2f}</td>
            <td>{b['max_dd_pct']:.2f}%</td>
            <td>{o['max_dd_pct']:.2f}%</td>
            <td class="{arrow(d['max_dd_pct'])}">{d['max_dd_pct']:+.2f}%</td>
            <td>{b['vol_pct']:.2f}%</td>
            <td>{o['vol_pct']:.2f}%</td>
        </tr>"""

    rows = "".join(row_cmp(c) for c in payload["windows"])

    stats = payload["trigger_stats_full"]
    verdict = payload["verdict"]
    verdict_cls = "good" if verdict["improved"] else "bad"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2190 Tail-Risk-Parity Overlay</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1200px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.2em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:160px; }}
  .kpi .value {{ font-size:1.5em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .good {{ color:#16a34a; font-weight:700; }}
  .bad {{ color:#dc2626; font-weight:700; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.84em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.72em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
</style></head><body>

<h1>EXP-2190 — Portfolio Insurance via Tail-Risk Parity</h1>
<div class="subtitle">Dynamic Crisis Alpha overlay on 5-stream 3× portfolio | {payload['timestamp']}</div>

<div class="note">
    <strong>Model:</strong> base hedge 5% → scale up by 7% per firing trigger,
    cap at 25%. Triggers: (a) ^VIX &gt; ^VIX3M term inversion, (b) portfolio
    DD &gt; 3%, (c) 20-day mean pairwise correlation across alpha streams
    &gt; 0.60. Alpha-sleeve weights re-normalized to (1 − hedge_w), 1-day
    lag. All streams real: EXP-1220 canonical, EXP-1770 calendars, cross-vol
    arb, Crisis Alpha v5 hedge.
</div>

<h2>Verdict</h2>
<div class="kpi-row">
    <div class="kpi"><div class="value {verdict_cls}">{'IMPROVES' if verdict['improved'] else 'REJECTED'}</div><div class="label">Decision (OOS)</div></div>
    <div class="kpi"><div class="value">{verdict['oos_delta_sharpe']:+.2f}</div><div class="label">ΔSharpe OOS</div></div>
    <div class="kpi"><div class="value">{verdict['oos_delta_dd_pct']:+.2f}%</div><div class="label">ΔMaxDD OOS</div></div>
    <div class="kpi"><div class="value">{verdict['oos_delta_cagr_pct']:+.2f}%</div><div class="label">ΔCAGR OOS</div></div>
</div>

<h2>Window-by-Window Comparison</h2>
<table>
    <thead><tr>
        <th>Window</th>
        <th>Base CAGR</th><th>Ovl CAGR</th><th>Δ</th>
        <th>Base Sh</th><th>Ovl Sh</th><th>Δ</th>
        <th>Base DD</th><th>Ovl DD</th><th>Δ</th>
        <th>Base Vol</th><th>Ovl Vol</th>
    </tr></thead>
    <tbody>{rows}</tbody>
</table>

<h2>Trigger Firing Statistics (full sample)</h2>
<table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
        <tr><td>Days</td><td>{stats['n_days']}</td></tr>
        <tr><td>% VIX-term inverted</td><td>{stats['pct_inverted']:.2f}%</td></tr>
        <tr><td>% DD-trigger (&gt;3%)</td><td>{stats['pct_dd_trigger']:.2f}%</td></tr>
        <tr><td>% Corr-trigger (&gt;0.60)</td><td>{stats['pct_corr_trigger']:.2f}%</td></tr>
        <tr><td>Mean target hedge w</td><td>{stats['mean_target_hedge_w']:.3f}</td></tr>
        <tr><td>Max target hedge w</td><td>{stats['max_target_hedge_w']:.3f}</td></tr>
    </tbody>
</table>

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2190 — compass/exp2190_tail_risk_parity.py · Real streams via EXP-2080 cache + Yahoo VIX term
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2190 — Portfolio Insurance via Tail-Risk Parity")
    print("=" * 72)

    print("\n[1/4] Loading streams and VIX term (real data)...")
    streams = load_streams()
    vix_term = load_vix_term("2019-12-01", "2026-01-01")

    print("\n[2/4] Building baseline portfolio (3× fixed-hedge 5-stream)...")
    baseline = base_portfolio_returns(streams)
    b_full = compute_metrics(baseline, "baseline full")
    print(f"  baseline full: CAGR={b_full['cagr_pct']:.2f}% "
          f"Sharpe={b_full['sharpe']:.2f} DD={b_full['max_dd_pct']:.2f}% "
          f"Vol={b_full['vol_pct']:.2f}%")

    print("\n[3/4] Computing stress triggers and applying dynamic overlay...")
    panel = build_trigger_panel(streams, vix_term, baseline)
    overlay = overlay_portfolio_returns(streams, panel)
    o_full = compute_metrics(overlay, "overlay full")
    print(f"  overlay  full: CAGR={o_full['cagr_pct']:.2f}% "
          f"Sharpe={o_full['sharpe']:.2f} DD={o_full['max_dd_pct']:.2f}% "
          f"Vol={o_full['vol_pct']:.2f}%")
    tstats = trigger_firing_stats(panel)
    print(f"  triggers: inverted={tstats['pct_inverted']}%  "
          f"dd={tstats['pct_dd_trigger']}%  corr={tstats['pct_corr_trigger']}%  "
          f"mean_hedge_w={tstats['mean_target_hedge_w']}")

    print("\n[4/4] Walk-forward window-by-window comparison...")
    cmp_full = compare_window(baseline, overlay,
                                "2020-01-01", "2025-12-31", "full sample")
    cmp_is = compare_window(baseline, overlay, IS_START, IS_END, "in-sample 2020-2022")
    cmp_oos = compare_window(baseline, overlay, OOS_START, OOS_END, "OOS 2023-2025")

    for c in (cmp_full, cmp_is, cmp_oos):
        w = c["window"]["label"]
        b = c["baseline"]
        o = c["overlay"]
        d = c["delta"]
        print(f"  {w:20s}  Sharpe {b['sharpe']:.2f}→{o['sharpe']:.2f} ({d['sharpe']:+.2f})  "
              f"DD {b['max_dd_pct']:.2f}%→{o['max_dd_pct']:.2f}% ({d['max_dd_pct']:+.2f}%)  "
              f"CAGR {b['cagr_pct']:.2f}%→{o['cagr_pct']:.2f}% ({d['cagr_pct']:+.2f}%)")

    verdict = {
        "oos_delta_sharpe": cmp_oos["delta"]["sharpe"],
        "oos_delta_dd_pct": cmp_oos["delta"]["max_dd_pct"],
        "oos_delta_cagr_pct": cmp_oos["delta"]["cagr_pct"],
        # "Improved" = Sharpe up AND DD less negative (better)
        "improved": (cmp_oos["delta"]["sharpe"] > 0
                      and cmp_oos["delta"]["max_dd_pct"] >= 0),
    }
    print(f"\n  VERDICT (OOS 2023-2025): "
          f"{'IMPROVES' if verdict['improved'] else 'REJECTED'}  "
          f"ΔSharpe={verdict['oos_delta_sharpe']:+.2f}  "
          f"ΔDD={verdict['oos_delta_dd_pct']:+.2f}%")

    payload = {
        "experiment": "EXP-2190",
        "title": "Portfolio Insurance via Tail-Risk Parity",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "gross_leverage": GROSS_LEVERAGE,
            "alpha_weights": ALPHA_WEIGHTS,
            "base_hedge_w": BASE_HEDGE_WEIGHT,
            "max_hedge_w": MAX_HEDGE_WEIGHT,
            "hedge_step_per_trigger": HEDGE_STEP,
            "dd_trigger_pct": DD_TRIGGER * 100,
            "corr_trigger": CORR_TRIGGER,
            "corr_window": CORR_WINDOW,
            "is_window": {"start": IS_START, "end": IS_END},
            "oos_window": {"start": OOS_START, "end": OOS_END},
        },
        "windows": [cmp_full, cmp_is, cmp_oos],
        "trigger_stats_full": tstats,
        "verdict": verdict,
        "rule_zero": (
            "Real streams from compass/cache/exp2080_streams.pkl "
            "(EXP-1220 canonical, EXP-1770 walk-forward GLD/SLV, cross-vol "
            "arb, Crisis Alpha v5). Yahoo ^VIX and ^VIX3M for term "
            "inversion. No synthetic data, 1-day lag on overlay (no "
            "look-ahead)."
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
