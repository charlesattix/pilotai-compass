"""
compass/exp2090_calendar_seasonality.py — EXP-2090 Calendar Spread Seasonality Filter.

HYPOTHESIS: Gold and silver have well-documented seasonal demand patterns
(Indian wedding season Oct-Dec, Chinese New Year Jan-Feb, summer
jewelry-demand lull, harvest-driven industrial silver swings). A
monthly sizing overlay that leans INTO historically strong months and
LEANS OUT of weak ones should improve the Sharpe of the EXP-1770
calendar-spread streams without changing the underlying strategy.

PROTOCOL (Rule Zero — all REAL Yahoo data):
  1. Reuse compass.exp1770_commodity_calendars.walk_forward to obtain
     the canonical OOS daily return series for GLD−GC=F and SLV−SI=F.
  2. Split the streams: train window 2015-2019 (5 years in-sample),
     test window 2020-2025 (6 years OOS).
  3. On the TRAIN window only, compute per-calendar-month mean daily
     return → z-score across months → sizing multiplier:
         mult = clip(1 + k*z, MIN, MAX)
     We use k=0.5, MIN=0.25, MAX=1.75 so no month is fully dropped or
     levered beyond 1.75×. These bounds are chosen once and NOT tuned
     on the test set.
  4. Apply the frozen TRAIN multipliers to the TEST window (strict
     walk-forward — no leakage).
  5. Compare unfiltered vs filtered daily-return series on the TEST
     window. The filter "earns a slot" if it lifts Sharpe by ≥ 0.30.
  6. Report CAGR, Sharpe, Max DD, per-month breakdown (count of days,
     mean return, multiplier) in JSON + HTML.

OUTPUTS:
  compass/reports/exp2090_calendar_seasonality.{json,html}

Run::
    python3 -m compass.exp2090_calendar_seasonality
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp1770_commodity_calendars import load_pair, walk_forward

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2090_calendar_seasonality.json"
REPORT_HTML = REPORT_DIR / "exp2090_calendar_seasonality.html"

TRADING_DAYS = 252
SHARPE_LIFT_THRESHOLD = 0.30  # filter "earns a slot" if ≥ this lift

# Sizing-overlay hyperparameters — FIXED, not tuned on the test set.
K = 0.5        # z-score scaling
MIN_MULT = 0.25
MAX_MULT = 1.75

TRAIN_START = "2015-01-01"
TRAIN_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2025-12-31"

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def load_calendar_stream(etf: str, fut: str) -> pd.Series:
    print(f"  loading {etf}/{fut} calendar stream (Yahoo via EXP-1770)...")
    df = load_pair(etf, fut)
    bt = walk_forward(etf, df)
    s = bt.daily_returns.dropna()
    s.name = etf
    m = bt.metrics
    print(f"    {etf}: n_days={m['n_days']} CAGR={m['cagr']*100:.2f}% "
          f"Sharpe={m['sharpe']:.2f} MaxDD={m['max_dd']*100:.2f}%")
    return s


# ═══════════════════════════════════════════════════════════════════════════
# Seasonality overlay
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MonthProfile:
    month: int        # 1..12
    name: str
    n_days: int
    mean_daily_ret: float
    z: float
    multiplier: float


def fit_seasonality(train_rets: pd.Series) -> Dict[int, MonthProfile]:
    """Estimate per-month multipliers on the training window.

    Only NON-ZERO trading days are used so long flat stretches from
    walk-forward "flat" rules do not drag the seasonality estimate.
    """
    df = pd.DataFrame({
        "ret": train_rets.values,
        "month": train_rets.index.month,
    })
    active = df[df["ret"] != 0].copy()
    if active.empty:
        # Degenerate: no signal in training window → flat multipliers.
        return {m: MonthProfile(m, MONTH_NAMES[m - 1], 0, 0.0, 0.0, 1.0)
                for m in range(1, 13)}

    grp = active.groupby("month")["ret"].agg(["count", "mean"]).reindex(range(1, 13))
    counts = grp["count"].fillna(0).astype(int)
    means = grp["mean"].fillna(0.0)
    mu = float(means.mean())
    sd = float(means.std(ddof=0))
    profiles: Dict[int, MonthProfile] = {}
    for m in range(1, 13):
        z = 0.0 if sd < 1e-12 else (float(means.loc[m]) - mu) / sd
        mult = max(MIN_MULT, min(MAX_MULT, 1.0 + K * z))
        profiles[m] = MonthProfile(
            month=m,
            name=MONTH_NAMES[m - 1],
            n_days=int(counts.loc[m]),
            mean_daily_ret=float(means.loc[m]),
            z=z,
            multiplier=mult,
        )
    return profiles


def apply_seasonality(rets: pd.Series,
                        profiles: Dict[int, MonthProfile]) -> pd.Series:
    """Scale each day's return by its month's fitted multiplier."""
    mults = rets.index.month.map(lambda m: profiles[m].multiplier)
    return pd.Series(rets.values * np.asarray(mults, dtype=float),
                      index=rets.index, name=rets.name)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(rets: pd.Series) -> Dict[str, float]:
    r = rets.dropna()
    n = len(r)
    if n < 5:
        return {"n_days": n, "cagr_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "vol_pct": 0.0, "hit_rate_pct": 0.0}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    sharpe = mu / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else 0.0
    eq = (1.0 + r).cumprod()
    yrs = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    pk = eq.cummax()
    dd = (eq - pk) / pk
    return {
        "n_days": n,
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(float(dd.min()) * 100, 3),
        "vol_pct": round(sd * math.sqrt(TRADING_DAYS) * 100, 3),
        "hit_rate_pct": round(float((r != 0).mean()) * 100, 1),
    }


def monthly_breakdown(rets: pd.Series) -> List[Dict]:
    df = pd.DataFrame({"ret": rets.values, "month": rets.index.month})
    grp = df.groupby("month")["ret"].agg(["count", "mean", "std", "sum"])
    out = []
    for m in range(1, 13):
        if m not in grp.index:
            continue
        row = grp.loc[m]
        n = int(row["count"])
        mean = float(row["mean"]) if pd.notna(row["mean"]) else 0.0
        std = float(row["std"]) if pd.notna(row["std"]) else 0.0
        total = float(row["sum"]) if pd.notna(row["sum"]) else 0.0
        sharpe = mean / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0
        out.append({
            "month": m,
            "name": MONTH_NAMES[m - 1],
            "n_days": n,
            "mean_daily_pct": round(mean * 100, 4),
            "total_pct": round(total * 100, 3),
            "sharpe": round(sharpe, 3),
        })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Workflow
# ═══════════════════════════════════════════════════════════════════════════

def run_pair(etf: str, fut: str) -> Dict:
    stream = load_calendar_stream(etf, fut)

    train = stream[(stream.index >= TRAIN_START) & (stream.index <= TRAIN_END)]
    test = stream[(stream.index >= TEST_START) & (stream.index <= TEST_END)]

    profiles = fit_seasonality(train)

    # Unfiltered and filtered TEST windows
    test_unfiltered = test
    test_filtered = apply_seasonality(test, profiles)

    m_train = compute_metrics(train)
    m_test_unf = compute_metrics(test_unfiltered)
    m_test_flt = compute_metrics(test_filtered)

    sharpe_lift = round(m_test_flt["sharpe"] - m_test_unf["sharpe"], 3)
    cagr_lift = round(m_test_flt["cagr_pct"] - m_test_unf["cagr_pct"], 3)

    return {
        "pair": etf,
        "future": fut,
        "train_window": {"start": TRAIN_START, "end": TRAIN_END,
                          "n_days": int(len(train)), "metrics": m_train},
        "test_window": {"start": TEST_START, "end": TEST_END,
                         "n_days": int(len(test))},
        "profiles": {
            MONTH_NAMES[p.month - 1]: {
                "n_train_days": p.n_days,
                "mean_daily_pct": round(p.mean_daily_ret * 100, 4),
                "z": round(p.z, 3),
                "multiplier": round(p.multiplier, 3),
            }
            for p in profiles.values()
        },
        "test_unfiltered": m_test_unf,
        "test_filtered": m_test_flt,
        "sharpe_lift": sharpe_lift,
        "cagr_lift_pct": cagr_lift,
        "earns_slot": sharpe_lift >= SHARPE_LIFT_THRESHOLD,
        "monthly_breakdown_test": {
            "unfiltered": monthly_breakdown(test_unfiltered),
            "filtered": monthly_breakdown(test_filtered),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML
# ═══════════════════════════════════════════════════════════════════════════

def render_html(payload: Dict) -> str:
    def fmt_profile_rows(prof: Dict) -> str:
        rows = ""
        for name in MONTH_NAMES:
            p = prof[name]
            mult = p["multiplier"]
            cls = ("good" if mult > 1.10
                    else "bad" if mult < 0.90 else "")
            rows += f"""<tr><td>{name}</td>
                <td>{p['n_train_days']}</td>
                <td>{p['mean_daily_pct']:+.4f}%</td>
                <td>{p['z']:+.2f}</td>
                <td class="{cls}">{mult:.2f}×</td></tr>"""
        return rows

    def fmt_pair_section(r: Dict) -> str:
        unf = r["test_unfiltered"]
        flt = r["test_filtered"]
        verdict_cls = "pass" if r["earns_slot"] else "fail"
        verdict_text = "EARNS SLOT" if r["earns_slot"] else "REJECTED"
        return f"""
        <h2>{r['pair']} ({r['pair']} − {r['future']})</h2>
        <div class="kpi-row">
            <div class="kpi"><div class="value">{flt['cagr_pct']:.2f}%</div><div class="label">Filtered CAGR</div></div>
            <div class="kpi"><div class="value">{flt['sharpe']:.2f}</div><div class="label">Filtered Sharpe</div></div>
            <div class="kpi"><div class="value">{flt['max_dd_pct']:.2f}%</div><div class="label">Max DD</div></div>
            <div class="kpi"><div class="value {verdict_cls}">{r['sharpe_lift']:+.2f}</div><div class="label">ΔSharpe vs Unfiltered</div></div>
            <div class="kpi"><div class="value {verdict_cls}">{verdict_text}</div><div class="label">Decision</div></div>
        </div>
        <h3>Test-window comparison</h3>
        <table><thead><tr><th>Variant</th><th>CAGR</th><th>Sharpe</th>
            <th>Max DD</th><th>Vol</th><th>Hit %</th></tr></thead>
            <tbody>
            <tr><td>Unfiltered</td><td>{unf['cagr_pct']:.2f}%</td>
                <td>{unf['sharpe']:.2f}</td><td>{unf['max_dd_pct']:.2f}%</td>
                <td>{unf['vol_pct']:.2f}%</td><td>{unf['hit_rate_pct']:.1f}%</td></tr>
            <tr><td>Filtered (seasonality)</td><td>{flt['cagr_pct']:.2f}%</td>
                <td>{flt['sharpe']:.2f}</td><td>{flt['max_dd_pct']:.2f}%</td>
                <td>{flt['vol_pct']:.2f}%</td><td>{flt['hit_rate_pct']:.1f}%</td></tr>
            </tbody></table>
        <h3>Seasonality profile (trained on 2015-2019)</h3>
        <table><thead><tr><th>Month</th><th>Train n</th><th>Mean daily</th>
            <th>z</th><th>Multiplier</th></tr></thead>
            <tbody>{fmt_profile_rows(r['profiles'])}</tbody></table>
        """

    pair_sections = "".join(fmt_pair_section(r) for r in payload["pairs"])

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>EXP-2090 Calendar Seasonality Filter</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         max-width:1100px; margin:0 auto; padding:28px; background:#fff; color:#1e293b; line-height:1.55; }}
  h1 {{ color:#0f172a; }} h2 {{ color:#334155; margin-top:2.2em;
         padding-bottom:8px; border-bottom:2px solid #e2e8f0; }}
  h3 {{ color:#475569; margin-top:1.4em; }}
  .subtitle {{ color:#64748b; font-size:0.9rem; margin-bottom:16px; }}
  .kpi-row {{ display:flex; gap:14px; flex-wrap:wrap; margin:18px 0; }}
  .kpi {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
          padding:18px; text-align:center; flex:1; min-width:140px; }}
  .kpi .value {{ font-size:1.6em; font-weight:800; color:#0f172a; }}
  .kpi .label {{ font-size:0.72em; color:#64748b; margin-top:4px; text-transform:uppercase; }}
  .pass {{ color:#16a34a; }} .fail {{ color:#dc2626; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; font-size:0.86em; }}
  th {{ background:#f1f5f9; padding:9px 12px; text-align:right; font-weight:600;
       color:#475569; border-bottom:2px solid #cbd5e1; font-size:0.78em; text-transform:uppercase; }}
  th:first-child {{ text-align:left; }}
  td {{ padding:8px 12px; text-align:right; border-bottom:1px solid #e2e8f0; }}
  td:first-child {{ text-align:left; }}
  tr:hover {{ background:#f8fafc; }}
  td.good {{ color:#16a34a; font-weight:600; }}
  td.bad  {{ color:#dc2626; font-weight:600; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px;
            padding:14px; margin:14px 0; font-size:0.85rem; }}
</style></head><body>

<h1>EXP-2090 — GLD/SLV Calendar Seasonality Filter</h1>
<div class="subtitle">Walk-forward: train 2015-2019, test 2020-2025 | {payload['timestamp']}</div>

<div class="note">
    <strong>Question:</strong> Do GLD/SLV calendar-spread returns exhibit
    exploitable monthly seasonality? Sizing overlay = clip(1 + 0.5·z,
    0.25, 1.75) where z is computed on 2015-2019 only. Filter earns
    a slot if test-window Sharpe lifts ≥ +{SHARPE_LIFT_THRESHOLD:.2f}.<br>
    <strong>Data:</strong> Real Yahoo GLD/SLV vs GC=F/SI=F, EXP-1770
    walk-forward OOS streams. No synthetic data.
</div>

{pair_sections}

<div style="margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0;
            font-size:0.78em; color:#94a3b8; text-align:center;">
EXP-2090 — compass/exp2090_calendar_seasonality.py · Train 2015-2019 · Test 2020-2025 · Real Yahoo data only
</div>

</body></html>"""


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    print("=" * 72)
    print("EXP-2090 — GLD/SLV Calendar Seasonality Filter")
    print("=" * 72)

    print("\n[1/3] Loading calendar streams (EXP-1770 walk-forward, Yahoo)...")
    pairs_cfg = [("GLD", "GC=F"), ("SLV", "SI=F")]
    results = []
    for etf, fut in pairs_cfg:
        results.append(run_pair(etf, fut))

    print("\n[2/3] Test-window summary (train 2015-2019 → test 2020-2025):")
    for r in results:
        unf = r["test_unfiltered"]
        flt = r["test_filtered"]
        verdict = "EARNS SLOT" if r["earns_slot"] else "REJECTED"
        print(f"  {r['pair']:3s}  unfiltered: CAGR={unf['cagr_pct']:6.2f}% "
              f"Sharpe={unf['sharpe']:5.2f} DD={unf['max_dd_pct']:6.2f}%")
        print(f"       filtered:   CAGR={flt['cagr_pct']:6.2f}% "
              f"Sharpe={flt['sharpe']:5.2f} DD={flt['max_dd_pct']:6.2f}%  "
              f"ΔSharpe={r['sharpe_lift']:+.3f}  {verdict}")

    print("\n[3/3] Writing reports...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "EXP-2090",
        "title": "GLD/SLV Calendar Spread Seasonality Filter",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hypothesis": (
            "Monthly seasonality (Indian wedding season, Chinese New Year, "
            "etc.) creates exploitable patterns in GLD/SLV calendar-spread "
            "returns that can be captured via a pre-frozen sizing overlay."
        ),
        "protocol": {
            "train_window": f"{TRAIN_START} → {TRAIN_END}",
            "test_window": f"{TEST_START} → {TEST_END}",
            "overlay": (f"mult = clip(1 + {K}·z, {MIN_MULT}, {MAX_MULT}) "
                         f"where z is the per-month z-score of mean daily return "
                         f"computed from non-zero days in the train window only"),
            "earns_slot_threshold_delta_sharpe": SHARPE_LIFT_THRESHOLD,
        },
        "pairs": results,
        "decision": {
            r["pair"]: "EARNS_SLOT" if r["earns_slot"] else "REJECTED"
            for r in results
        },
        "rule_zero": (
            "EXP-1770 walk-forward OOS streams from Yahoo GLD/SLV/GC=F/SI=F. "
            "Overlay multipliers frozen on train window, applied as-is on "
            "test window. No test-window leakage, no synthetic data."
        ),
    }
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  → {REPORT_JSON}")
    REPORT_HTML.write_text(render_html(payload), encoding="utf-8")
    print(f"  → {REPORT_HTML}")
    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
