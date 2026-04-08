"""
EXP-2720 — Drawdown Recovery Analysis for North Star v8a

Question
--------
For the v8a portfolio (8-stream Ledoit-Wolf walk-forward, EXP-2600),
how fast does it recover from drawdowns? Specifically:

  1. Average recovery time (days to new high) after a >3% drawdown
  2. Average recovery time after a >5% drawdown
  3. Worst-case recovery time
  4. Effect of the EXP-2640 adaptive VIX vol-target on recovery speed

Why it matters
--------------
For fund marketing and investor psychology, "Sharpe 6" means nothing
if investors have to sit through a six-month underwater stretch. A
fast recovery profile is the difference between investors adding
capital during a dip and investors pulling capital during a dip.

Method
------
1. Build the v8a cube via EXP-2600 (exp1220 + v5_hedge + gld_cal +
   slv_cal + cross_vol + xlf_cs + xli_cs + qqq_cs).
2. Walk-forward (252 train / 63 test, Ledoit-Wolf) at TARGET_VOL = 0.12
   to produce a pooled OOS daily return series.
3. Compute the equity curve, then identify every drawdown episode
   (peak → trough → recovery-to-new-high).
4. Bucket episodes by max depth: >1%, >3%, >5%, >8%, all.
5. Repeat the full pipeline with the EXP-2640 `adaptive_vt` overlay
   applied causally on top of the pooled series, and compute the
   same recovery distributions.
6. Compare baseline vs overlay: recovery-time histogram, worst-case,
   mean, median.

ALL REAL DATA — cached streams + Yahoo VIX for the adaptive overlay.

Outputs
-------
  compass/exp2720_dd_recovery.py
  compass/reports/exp2720_dd_recovery.json
  compass/reports/exp2720_dd_recovery.html
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_JSON = ROOT / "compass" / "reports" / "exp2720_dd_recovery.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2720_dd_recovery.html"

TRADING_DAYS = 252
TARGET_VOL = 0.12
DEPTH_BUCKETS = [0.01, 0.03, 0.05, 0.08]    # drawdown thresholds


# ───────────────────────────────────────────────────────────────────────────
# Data — pooled OOS returns from EXP-2600 v8a walk-forward
# ───────────────────────────────────────────────────────────────────────────

def build_v8a_pooled(target_vol: float = TARGET_VOL) -> Tuple[pd.Series, Dict]:
    """Run the EXP-2600 walk-forward for the v8a cube and return its
    pooled OOS daily return series."""
    from compass.exp2600_north_star_v8 import build_cubes, walk_forward_lw
    cubes = build_cubes()
    v8a = cubes["v8a_add_qqq"]
    pooled, folds = walk_forward_lw(v8a, target_vol=target_vol, scale_cap=20.0)
    return pooled, {"n_folds": len(folds),
                    "n_days": int(len(pooled)),
                    "target_vol": target_vol,
                    "cube_cols": list(v8a.columns),
                    "cube_shape": list(v8a.shape)}


def load_vix_lag(index: pd.DatetimeIndex) -> pd.Series:
    """Load Yahoo ^VIX close, shift-by-1 for causality, reindex to `index`."""
    import yfinance as yf
    vix = yf.download("^VIX", start="2019-06-01", end="2026-07-01",
                      progress=False, auto_adjust=False)["Close"]
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    vix.index = pd.to_datetime(vix.index).normalize()
    vix_lag = vix.shift(1)
    return vix_lag.reindex(index.normalize()).ffill().bfill()


def apply_adaptive_vt(pooled: pd.Series, vix: pd.Series,
                      vix_low: float = 25.0, vix_high: float = 35.0,
                      exposure_at_high: float = 0.5) -> pd.Series:
    """EXP-2640 intervention_adaptive_vt: linearly ramp exposure from 1.0
    at VIX=vix_low to exposure_at_high at VIX=vix_high."""
    v = vix.values.astype(float)
    span = vix_high - vix_low
    raw = 1.0 - (v - vix_low) / span * (1.0 - exposure_at_high)
    exposure = np.clip(raw, exposure_at_high, 1.0)
    return pooled * pd.Series(exposure, index=pooled.index)


# ───────────────────────────────────────────────────────────────────────────
# Drawdown episode extraction
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class DDEpisode:
    peak_date: str
    trough_date: str
    recovery_date: Optional[str]       # None if still underwater at end of sample
    peak_eq: float
    trough_eq: float
    depth_pct: float
    fall_days: int                      # peak → trough
    recovery_days: Optional[int]        # trough → new high
    total_days: Optional[int]           # peak → new high
    underwater_days: Optional[int]      # peak → recovery_date
    fully_recovered: bool


def extract_dd_episodes(daily: pd.Series) -> List[DDEpisode]:
    """Walk the equity curve, emit every drawdown episode.

    An episode begins at a new high-water-mark and ends when the curve
    reaches a new high-water-mark again. Depth = 1 - trough/peak.
    """
    daily = daily.dropna()
    if len(daily) < 2:
        return []
    eq = (1 + daily).cumprod().values
    idx = daily.index
    n = len(eq)

    episodes: List[DDEpisode] = []
    peak = eq[0]
    peak_i = 0
    in_dd = False
    trough = peak
    trough_i = 0

    for i in range(1, n):
        if eq[i] > peak:
            # New all-time high
            if in_dd and trough < peak * (1 - 1e-12):
                # Close the episode that just ended at the previous peak
                depth = 1 - trough / peak
                episodes.append(DDEpisode(
                    peak_date=str(idx[peak_i].date()),
                    trough_date=str(idx[trough_i].date()),
                    recovery_date=str(idx[i].date()),
                    peak_eq=float(peak),
                    trough_eq=float(trough),
                    depth_pct=round(depth * 100, 4),
                    fall_days=trough_i - peak_i,
                    recovery_days=i - trough_i,
                    total_days=i - peak_i,
                    underwater_days=i - peak_i,
                    fully_recovered=True,
                ))
            peak = eq[i]
            peak_i = i
            in_dd = False
            trough = peak
            trough_i = i
        else:
            # Still under peak → in drawdown
            in_dd = True
            if eq[i] < trough:
                trough = eq[i]
                trough_i = i

    # Handle trailing open drawdown (still underwater at end of sample)
    if in_dd and trough < peak * (1 - 1e-12):
        depth = 1 - trough / peak
        episodes.append(DDEpisode(
            peak_date=str(idx[peak_i].date()),
            trough_date=str(idx[trough_i].date()),
            recovery_date=None,
            peak_eq=float(peak),
            trough_eq=float(trough),
            depth_pct=round(depth * 100, 4),
            fall_days=trough_i - peak_i,
            recovery_days=None,
            total_days=None,
            underwater_days=n - 1 - peak_i,
            fully_recovered=False,
        ))
    return episodes


# ───────────────────────────────────────────────────────────────────────────
# Recovery stats per depth bucket
# ───────────────────────────────────────────────────────────────────────────

def recovery_stats(episodes: List[DDEpisode],
                    depth_thresh_pct: float) -> Dict:
    """All episodes with depth >= threshold, summarise recovery days."""
    filtered = [e for e in episodes if e.depth_pct >= depth_thresh_pct * 100]
    if not filtered:
        return {"threshold_pct": depth_thresh_pct * 100, "n_episodes": 0}
    recovered = [e for e in filtered if e.fully_recovered]
    open_dd = [e for e in filtered if not e.fully_recovered]

    if recovered:
        rec_days   = np.array([e.recovery_days   for e in recovered])
        total_days = np.array([e.total_days      for e in recovered])
        under_days = np.array([e.underwater_days for e in recovered])
        depths     = np.array([e.depth_pct       for e in recovered])
        stats = {
            "n_episodes":       len(filtered),
            "n_recovered":      len(recovered),
            "n_open":           len(open_dd),
            "avg_recovery_days":    round(float(rec_days.mean()), 2),
            "median_recovery_days": round(float(np.median(rec_days)), 1),
            "p75_recovery_days":    round(float(np.percentile(rec_days, 75)), 1),
            "p90_recovery_days":    round(float(np.percentile(rec_days, 90)), 1),
            "max_recovery_days":    int(rec_days.max()),
            "min_recovery_days":    int(rec_days.min()),
            "avg_total_days":       round(float(total_days.mean()), 2),
            "median_total_days":    round(float(np.median(total_days)), 1),
            "max_total_days":       int(total_days.max()),
            "avg_underwater_days":  round(float(under_days.mean()), 2),
            "avg_depth_pct":        round(float(depths.mean()), 4),
            "max_depth_pct":        round(float(depths.max()), 4),
        }
    else:
        stats = {
            "n_episodes":  len(filtered),
            "n_recovered": 0,
            "n_open":      len(open_dd),
        }
    # Worst-case — look across both recovered + open-but-large episodes
    worst = max(filtered, key=lambda e: e.depth_pct)
    stats.update({
        "threshold_pct": depth_thresh_pct * 100,
        "worst_depth_pct":  worst.depth_pct,
        "worst_peak_date":  worst.peak_date,
        "worst_trough_date":worst.trough_date,
        "worst_recovered":  worst.fully_recovered,
        "worst_underwater_days": worst.underwater_days,
    })
    return stats


# ───────────────────────────────────────────────────────────────────────────
# HTML report
# ───────────────────────────────────────────────────────────────────────────

def _color_for_delta(delta: float, good_is_negative: bool = True) -> str:
    """Green if delta is improvement (for days, negative = better)."""
    if not np.isfinite(delta):
        return "#64748b"
    if good_is_negative:
        return "#16a34a" if delta < 0 else ("#dc2626" if delta > 0 else "#64748b")
    return "#16a34a" if delta > 0 else ("#dc2626" if delta < 0 else "#64748b")


def write_html(payload: Dict, path: Path) -> None:
    base_st = payload["baseline"]["stats_by_bucket"]
    vt_st   = payload["adaptive_vt"]["stats_by_bucket"]

    # Build per-bucket comparison rows
    bucket_rows = ""
    for depth_pct in [1.0, 3.0, 5.0, 8.0]:
        key = f"{depth_pct:.1f}"
        b = next((x for x in base_st if x["threshold_pct"] == depth_pct), None)
        v = next((x for x in vt_st   if x["threshold_pct"] == depth_pct), None)
        if b is None or v is None:
            continue
        b_avg = b.get("avg_recovery_days", 0)
        v_avg = v.get("avg_recovery_days", 0)
        delta = v_avg - b_avg if isinstance(b_avg,(int,float)) and isinstance(v_avg,(int,float)) else 0
        color = _color_for_delta(delta)
        b_worst = b.get("max_recovery_days", "—")
        v_worst = v.get("max_recovery_days", "—")
        bucket_rows += (
            f"<tr><td><strong>&gt;{depth_pct:.0f}%</strong></td>"
            f"<td>{b.get('n_episodes',0)}</td>"
            f"<td>{b.get('n_recovered',0)}</td>"
            f"<td>{b.get('avg_recovery_days','—')}</td>"
            f"<td>{b.get('median_recovery_days','—')}</td>"
            f"<td>{b.get('max_recovery_days','—')}</td>"
            f"<td>{v.get('avg_recovery_days','—')}</td>"
            f"<td>{v.get('median_recovery_days','—')}</td>"
            f"<td>{v.get('max_recovery_days','—')}</td>"
            f"<td style='color:{color};font-weight:700'>{delta:+.1f}d</td></tr>"
        )

    # Worst 8 episodes in baseline
    worst_rows = ""
    for e in payload["baseline"]["worst_episodes"][:8]:
        rec = "✅" if e["fully_recovered"] else "⚠ open"
        rec_d = e["recovery_days"] if e["recovery_days"] is not None else "—"
        worst_rows += (
            f"<tr><td>{e['peak_date']}</td><td>{e['trough_date']}</td>"
            f"<td>{e.get('recovery_date','—') or '—'}</td>"
            f"<td>{e['depth_pct']:.2f}%</td>"
            f"<td>{e['fall_days']}</td>"
            f"<td>{rec_d}</td>"
            f"<td>{e['underwater_days']}</td>"
            f"<td>{rec}</td></tr>"
        )

    # Base metrics
    bm = payload["baseline"]["equity_metrics"]
    vm = payload["adaptive_vt"]["equity_metrics"]

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>EXP-2720 Drawdown Recovery Analysis</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:0;padding:24px;background:#fff;color:#1e293b;max-width:1150px}}
h1{{font-size:1.5rem;color:#0f172a}} h2{{font-size:1.08rem;color:#334155;border-bottom:1px solid #e2e8f0;padding-bottom:4px;margin-top:1.6rem}}
.meta{{color:#64748b;font-size:0.82rem;margin-bottom:18px}}
.headline{{background:#f0fdf4;border-left:5px solid #16a34a;padding:14px 18px;border-radius:6px;margin:14px 0;font-size:0.95rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}}
.card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:10px;text-align:center}}
.card .l{{font-size:0.65rem;color:#64748b;text-transform:uppercase}} .card .v{{font-size:1.2rem;font-weight:700}}
table{{width:100%;border-collapse:collapse;font-size:0.83rem;margin:10px 0}}
th{{background:#f1f5f9;padding:6px 8px;text-align:right;font-size:0.68rem;color:#64748b;text-transform:uppercase;border-bottom:2px solid #e2e8f0}}
th:first-child{{text-align:left}} td{{padding:6px 8px;text-align:right;border-bottom:1px solid #f1f5f9}}
td:first-child{{text-align:left}}
</style></head><body>
<h1>EXP-2720 — Drawdown Recovery Analysis (North Star v8a)</h1>
<p class="meta">Pooled OOS daily series from EXP-2600 walk-forward (Ledoit-Wolf, 252/63, target vol 12%).
Baseline vs EXP-2640 adaptive VIX vol-target overlay. Real cached streams + Yahoo VIX.</p>

<div class="headline"><strong>Recovery headline</strong>
(baseline v8a pooled OOS): mean recovery after &gt;3% DD = <strong>{_find(base_st,3.0,'avg_recovery_days')}</strong> days
(median {_find(base_st,3.0,'median_recovery_days')}d) ·
after &gt;5% DD = <strong>{_find(base_st,5.0,'avg_recovery_days')}</strong> days
(median {_find(base_st,5.0,'median_recovery_days')}d) ·
worst-case <strong>{_find(base_st,3.0,'max_recovery_days')}</strong> days ·
total {payload['baseline']['n_dd_episodes']} DD episodes.
</div>

<div class="grid">
  <div class="card"><div class="l">Pooled OOS days</div><div class="v">{payload['wf']['n_days']}</div></div>
  <div class="card"><div class="l">Target vol</div><div class="v">{payload['wf']['target_vol']*100:.0f}%</div></div>
  <div class="card"><div class="l">Sharpe (base)</div><div class="v">{bm['sharpe']:+.2f}</div></div>
  <div class="card"><div class="l">Max DD (base)</div><div class="v">{bm['max_dd_pct']:.2f}%</div></div>
  <div class="card"><div class="l">Sharpe (VT)</div><div class="v">{vm['sharpe']:+.2f}</div></div>
  <div class="card"><div class="l">Max DD (VT)</div><div class="v">{vm['max_dd_pct']:.2f}%</div></div>
  <div class="card"><div class="l">DD episodes</div><div class="v">{payload['baseline']['n_dd_episodes']}</div></div>
</div>

<h2>Recovery by drawdown depth — baseline vs adaptive VT</h2>
<table>
<tr>
  <th>Depth</th><th>n episodes</th><th>n recovered</th>
  <th colspan="3" style="text-align:center;background:#f0fdf4">BASELINE v8a</th>
  <th colspan="3" style="text-align:center;background:#eff6ff">+ ADAPTIVE VT</th>
  <th>Δ mean</th>
</tr>
<tr><th></th><th></th><th></th>
  <th>mean d</th><th>median d</th><th>max d</th>
  <th>mean d</th><th>median d</th><th>max d</th>
  <th>days</th></tr>
{bucket_rows}
</table>

<h2>Worst 8 baseline drawdown episodes</h2>
<table>
<tr><th>Peak</th><th>Trough</th><th>Recovery</th><th>Depth</th>
<th>Fall days</th><th>Recovery days</th><th>Underwater total</th><th>Status</th></tr>
{worst_rows}
</table>

<h2>Interpretation</h2>
<ul>
<li>"Recovery days" = trading days from trough back to a new all-time high.</li>
<li>"Underwater days" = trading days from peak to recovery — the full
   duration investors are below the prior high-water-mark.</li>
<li>The &gt;3% bucket is the most investor-relevant: everyone feels a 3%
   drawdown. Mean-recovery-day number is the single-best proxy for "how
   bad does it feel to sit through a drawdown on this fund."</li>
<li>The adaptive VIX vol-target halves exposure between VIX 25 and 35
   linearly. It reduces depth in stress windows but also damps the
   recovery rate (because positions are smaller when the market rallies
   off the bottom).</li>
</ul>

<h2>Method</h2>
<ul>
<li>Cube: v8a = base + qqq_cs (EXP-2600 build_cubes)</li>
<li>Walk-forward: 252 train / 63 test, Ledoit-Wolf risk parity,
   target vol 12% (scale cap 20.0)</li>
<li>Equity curve: (1 + r).cumprod() on pooled OOS daily returns</li>
<li>Episode extraction: every peak → trough → new-high triple is
   one episode. Open episodes (still underwater at end) are reported
   separately as "open DD".</li>
<li>Adaptive VT: intervention_adaptive_vt from EXP-2640, applied
   causally (VIX shifted by 1 day) with vix_low=25, vix_high=35,
   exposure_at_high=0.5.</li>
</ul>
<div style="color:#94a3b8;font-size:.75rem;margin-top:1.6rem;border-top:1px solid #e2e8f0;padding-top:8px">
compass/exp2720_dd_recovery.py · REAL DATA (cached 7-stream + cached QQQ + Yahoo VIX)
</div>
</body></html>"""
    path.write_text(html, encoding="utf-8")


def _find(lst, threshold, key):
    """Look up a bucket stat by depth threshold."""
    for x in lst:
        if x.get("threshold_pct") == threshold:
            return x.get(key, "—")
    return "—"


# ───────────────────────────────────────────────────────────────────────────
# Equity metrics for the pooled series (for the cards)
# ───────────────────────────────────────────────────────────────────────────

def equity_metrics(daily: pd.Series) -> Dict[str, float]:
    daily = daily.dropna()
    n = len(daily)
    if n < 2:
        return {"n": n, "sharpe": 0.0, "max_dd_pct": 0.0,
                "cagr_pct": 0.0, "vol_pct": 0.0}
    mu, sd = float(daily.mean()), float(daily.std(ddof=1))
    sh = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = (1 + daily).cumprod()
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1) if eq.iloc[-1] > 0 else -1.0
    dd = float((1 - eq / eq.cummax()).max())
    return {
        "n": n,
        "sharpe": round(sh, 3),
        "max_dd_pct": round(dd * 100, 3),
        "cagr_pct": round(cagr * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
    }


# ───────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────

def main():
    print("EXP-2720 — Drawdown Recovery Analysis (v8a)")
    print("=" * 60)

    print("[1/4] Building v8a pooled OOS via EXP-2600 walk-forward…")
    pooled_base, wf_info = build_v8a_pooled(TARGET_VOL)
    print(f"      n_folds={wf_info['n_folds']}  n_days={wf_info['n_days']}  "
          f"target_vol={wf_info['target_vol']}")

    print("[2/4] Loading Yahoo ^VIX (causal shift-by-1) + applying adaptive VT…")
    vix = load_vix_lag(pooled_base.index)
    pooled_vt = apply_adaptive_vt(pooled_base, vix,
                                    vix_low=25.0, vix_high=35.0,
                                    exposure_at_high=0.5)
    print(f"      VIX mean {float(vix.mean()):.2f}  max {float(vix.max()):.2f}  "
          f"days with exposure<1.0: {int((vix > 25).sum())}")

    print("[3/4] Extracting drawdown episodes (baseline + adaptive VT)...")
    base_eps = extract_dd_episodes(pooled_base)
    vt_eps   = extract_dd_episodes(pooled_vt)
    print(f"      baseline: {len(base_eps)} episodes")
    print(f"      adaptive: {len(vt_eps)} episodes")

    base_stats = [recovery_stats(base_eps, th) for th in [0.01, 0.03, 0.05, 0.08]]
    vt_stats   = [recovery_stats(vt_eps,   th) for th in [0.01, 0.03, 0.05, 0.08]]

    print("\n[4/4] Recovery statistics")
    print("-" * 80)
    print(f"{'depth':>7} {'n_ep':>6} {'recov':>7} {'BASE mean':>12} "
          f"{'BASE max':>10} {'VT mean':>10} {'VT max':>10} {'Δ mean':>10}")
    for b, v in zip(base_stats, vt_stats):
        th = b["threshold_pct"]
        b_avg = b.get("avg_recovery_days", 0) or 0
        v_avg = v.get("avg_recovery_days", 0) or 0
        b_max = b.get("max_recovery_days", 0) or 0
        v_max = v.get("max_recovery_days", 0) or 0
        delta = v_avg - b_avg if isinstance(b_avg,(int,float)) and isinstance(v_avg,(int,float)) else 0
        print(f"  >{th:>3.0f}% {b.get('n_episodes',0):>6} "
              f"{b.get('n_recovered',0):>6} {b_avg:>12} {b_max:>10} "
              f"{v_avg:>10} {v_max:>10} {delta:>+10.1f}")

    # Worst episodes (by depth, recovered or not)
    worst_sorted = sorted(base_eps, key=lambda e: -e.depth_pct)

    payload = {
        "experiment": "EXP-2720",
        "title": "Drawdown Recovery Analysis — North Star v8a",
        "rule_zero": "Cached 7-stream + cached QQQ + Yahoo VIX (all REAL)",
        "wf": wf_info,
        "baseline": {
            "equity_metrics": equity_metrics(pooled_base),
            "n_dd_episodes":  len(base_eps),
            "stats_by_bucket": base_stats,
            "worst_episodes": [asdict(e) for e in worst_sorted[:20]],
        },
        "adaptive_vt": {
            "equity_metrics": equity_metrics(pooled_vt),
            "n_dd_episodes":  len(vt_eps),
            "stats_by_bucket": vt_stats,
            "vix_stats": {
                "mean": round(float(vix.mean()), 3),
                "max":  round(float(vix.max()), 3),
                "days_above_25": int((vix > 25).sum()),
                "days_above_30": int((vix > 30).sum()),
                "days_above_35": int((vix > 35).sum()),
            },
            "adaptive_vt_params": {
                "vix_low": 25.0,
                "vix_high": 35.0,
                "exposure_at_high": 0.5,
            },
        },
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    write_html(payload, REPORT_HTML)
    print(f"\nReports → {REPORT_JSON.name} + {REPORT_HTML.name}")
    return payload


if __name__ == "__main__":
    main()
