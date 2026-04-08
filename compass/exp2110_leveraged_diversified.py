"""
EXP-2110 — Leveraged Diversified Portfolio (CAGR sweet spot sweep).

Context
-------
EXP-2080 established that the 5-stream static portfolio
  exp1220 40% / gld_cal 20% / slv_cal 20% / cross_vol 15% / v5_hedge 5%
delivers a pooled OOS walk-forward (20 folds × 63 days) of:
    CAGR   33.00%
    Sharpe  5.24
    Max DD  2.60%

Task
----
Find the uniform leverage L that closes the CAGR gap to ≥100% while
keeping max DD < 12% and Sharpe > 6.0. Leverages tested: 1.5×, 2.0×,
2.5×, 3.0×.

HONEST mathematical note (flagged in both the report and the commit):
Sharpe is leverage-invariant by construction. Applying a uniform
leverage L to a daily return series r_t gives new daily mean L·μ and
new daily std L·σ, so the ratio (mean / std) × √252 is unchanged.
No amount of uniform leverage can raise the Sharpe above its 1.0×
baseline — the Sharpe > 6.0 target is therefore unreachable via sizing
alone. Max drawdown and CAGR, however, DO move with leverage:
  * Max DD scales approximately linearly with L (for small DDs)
  * CAGR scales faster than linear due to compounding
    (arithmetic: L·μ  →  geometric: (1 + L·μ)^252 − 1)
  * Calmar ratio stays approximately invariant (CAGR_scale ≈ DD_scale)

The leverage sweep therefore answers two out of the three Carlos
criteria (CAGR ≥100%, DD <12%) and documents the third (Sharpe >6.0)
as structurally unreachable without a signal-quality improvement.

Pipeline
--------
1. Reuse compass.exp2080_corr_regime.load_streams to get the cached
   5-stream DataFrame (real IronVault + Yahoo data, all upstream
   experiments are Rule Zero clean).
2. Build the STATIC_WEIGHTS portfolio daily series.
3. Apply L ∈ {1.0 (baseline), 1.5, 2.0, 2.5, 3.0} uniformly.
4. Run the same 20-fold walk-forward (train=252, test=63) EXP-2080
   uses, compute per-fold metrics and pooled-OOS metrics per leverage.
5. Rank leverages on the three criteria and report the sweet spot.

Outputs
-------
  compass/exp2110_leveraged_diversified.py            (this file)
  compass/reports/exp2110_leveraged_diversified.json
  compass/reports/exp2110_leveraged_diversified.html

Tag: EXP-2110
Run: python3 -m compass.exp2110_leveraged_diversified
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2110_leveraged_diversified.json"
REPORT_HTML = REPORT_DIR / "exp2110_leveraged_diversified.html"

from compass.exp2080_corr_regime import (
    STATIC_WEIGHTS,
    load_streams,
    metrics,
    static_portfolio,
    TRADING_DAYS,
)

LEVERAGES: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0]
TRAIN_DAYS = 252
TEST_DAYS = 63

# Carlos targets
TARGET_CAGR = 1.00      # 100%
TARGET_SHARPE = 6.00
TARGET_DD = 0.12        # 12%


# ── Leveraged daily return ─────────────────────────────────────────────


def apply_uniform_leverage(daily: pd.Series, L: float) -> pd.Series:
    """Uniform daily-return leverage.

    The literal definition is r_L(t) = L × r(t). This is a first-order
    approximation of a constant-leverage portfolio — good for daily
    moves small enough that the Itô correction is negligible. We do NOT
    model margin calls, financing cost, or rebalance slippage; a
    production leveraged sleeve would need to bolt those on.
    """
    return daily * float(L)


# ── Walk-forward ───────────────────────────────────────────────────────


@dataclass
class FoldMetrics:
    test_start: str
    test_end: str
    metrics: Dict[str, float]


def walk_forward_static(df: pd.DataFrame, leverage: float,
                        train_days: int = TRAIN_DAYS,
                        test_days: int = TEST_DAYS
                        ) -> Tuple[List[FoldMetrics], pd.Series, Dict[str, float]]:
    base = static_portfolio(df)
    levered = apply_uniform_leverage(base, leverage)

    n = len(df)
    folds: List[FoldMetrics] = []
    pooled: List[pd.Series] = []
    i = train_days
    while i + test_days <= n:
        te = levered.iloc[i:i + test_days]
        folds.append(FoldMetrics(
            test_start=str(te.index[0].date()),
            test_end=str(te.index[-1].date()),
            metrics=metrics(te),
        ))
        pooled.append(te)
        i += test_days

    pooled_series = pd.concat(pooled).sort_index() if pooled else pd.Series(dtype=float)
    pooled_metrics = metrics(pooled_series) if len(pooled_series) else {}
    return folds, pooled_series, pooled_metrics


# ── Sweet-spot analysis ────────────────────────────────────────────────


def evaluate_target(pm: Dict[str, float]) -> Dict[str, bool]:
    cagr_ok = (pm.get("cagr_pct", 0.0) / 100.0) >= TARGET_CAGR
    sharpe_ok = pm.get("sharpe", 0.0) >= TARGET_SHARPE
    dd_ok = (pm.get("max_dd_pct", 1e9) / 100.0) < TARGET_DD
    return {"cagr": cagr_ok, "sharpe": sharpe_ok, "dd": dd_ok,
            "all": cagr_ok and sharpe_ok and dd_ok}


def find_sweet_spot(sweep: Dict[float, Dict]) -> Dict:
    """Pick the best leverage according to Carlos's rank:
       1. prefer leverages passing ALL three criteria
       2. else prefer passing CAGR+DD (Sharpe is structurally invariant)
       3. tiebreak on highest CAGR
    """
    passing_all = [lev for lev, r in sweep.items() if r["targets"]["all"]]
    if passing_all:
        best = max(passing_all, key=lambda lev: sweep[lev]["pooled"]["cagr_pct"])
        return {"leverage": best, "reason": "passes all three targets"}

    passing_cagr_dd = [lev for lev, r in sweep.items()
                       if r["targets"]["cagr"] and r["targets"]["dd"]]
    if passing_cagr_dd:
        best = max(passing_cagr_dd, key=lambda lev: sweep[lev]["pooled"]["cagr_pct"])
        return {
            "leverage": best,
            "reason": ("passes CAGR ≥ 100% and DD < 12% but NOT Sharpe > 6.0 "
                       "— Sharpe is leverage-invariant by construction"),
        }

    # Highest CAGR subject to DD < 12%
    safe = [lev for lev, r in sweep.items() if r["targets"]["dd"]]
    if safe:
        best = max(safe, key=lambda lev: sweep[lev]["pooled"]["cagr_pct"])
        return {
            "leverage": best,
            "reason": "highest CAGR with DD still under 12%, CAGR target MISSED",
        }

    # Nothing survives — return the lowest leverage
    best = min(sweep.keys())
    return {"leverage": best, "reason": "no leverage survives DD cap"}


# ── HTML ───────────────────────────────────────────────────────────────


def _fmt_pct(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}%" if np.isfinite(x) else "—"


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(sweep: Dict[float, Dict], sweet: Dict,
                stream_columns: List[str]) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111}
    h1{border-bottom:3px solid #1a4d2e}
    h2{margin-top:2em;color:#1a4d2e}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#1a4d2e;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#1a4d2e}
    .pill.bad{background:#c0392b}
    .pill.ok{background:#0a7d1f}
    """
    h: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2110 Leveraged Diversified Portfolio</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2110 — Leveraged Diversified Portfolio</h1>",
        "<p class='muted'>Uniform-leverage sweep on the EXP-2080 static "
        "5-stream portfolio (exp1220 40% / gld_cal 20% / slv_cal 20% / "
        "cross_vol 15% / v5_hedge 5%). Walk-forward 20 folds "
        f"(train {TRAIN_DAYS} / test {TEST_DAYS}).</p>",
        "<p><span class='pill'>Rule Zero ✓ real data, all upstream experiments</span></p>",
    ]
    h.append(f"<p class='muted'>Streams: {', '.join(stream_columns)}</p>")

    # Mathematical note — Sharpe invariance
    h.append("<h2>Mathematical note: Sharpe invariance under uniform leverage</h2>")
    h.append(
        "<p>For a uniform leverage L applied to a daily return series r<sub>t</sub>, "
        "the new series is L·r<sub>t</sub>. The new mean is L·μ, the new std is L·σ, "
        "so <b>Sharpe = (L·μ) / (L·σ) × √252 = μ/σ × √252</b> — identical to baseline. "
        "The Sharpe > 6.0 target is therefore <b>structurally unreachable via "
        "leverage alone</b> (the baseline pooled OOS Sharpe is ~5.24). CAGR and "
        "max drawdown DO scale with leverage (CAGR slightly faster-than-linear "
        "via compounding, DD ≈ linear), so the CAGR/DD targets can still be "
        "hit independently of Sharpe.</p>"
    )

    # Sweep table
    h.append("<h2>Leverage sweep — pooled OOS (20 folds × 63 days)</h2>")
    h.append("<table><tr><th>Leverage</th><th>n days</th>"
             "<th>CAGR</th><th>Sharpe</th><th>Sortino</th>"
             "<th>Max DD</th><th>Vol</th><th>Calmar</th>"
             "<th>CAGR ≥ 100%</th><th>Sharpe ≥ 6</th><th>DD &lt; 12%</th></tr>")
    for L in LEVERAGES:
        pm = sweep[L]["pooled"]
        t = sweep[L]["targets"]

        def flag(ok: bool) -> str:
            return (f"<span class='pill ok'>YES</span>" if ok
                    else f"<span class='pill bad'>no</span>")

        h.append(
            f"<tr><td class='l'><b>{L:.1f}×</b></td>"
            f"<td>{pm['n']}</td>"
            f"<td class='{ 'pos' if pm['cagr_pct']>0 else 'neg' }'>{_fmt_pct(pm['cagr_pct'])}</td>"
            f"<td>{_fmt(pm['sharpe'])}</td>"
            f"<td>{_fmt(pm['sortino'])}</td>"
            f"<td class='neg'>{_fmt_pct(pm['max_dd_pct'])}</td>"
            f"<td>{_fmt_pct(pm['vol_pct'])}</td>"
            f"<td>{_fmt(pm['calmar'])}</td>"
            f"<td>{flag(t['cagr'])}</td>"
            f"<td>{flag(t['sharpe'])}</td>"
            f"<td>{flag(t['dd'])}</td></tr>"
        )
    h.append("</table>")

    # Sweet spot
    h.append("<h2>Sweet spot</h2>")
    sl = sweet["leverage"]
    pm = sweep[sl]["pooled"]
    all_pass = sweep[sl]["targets"]["all"]
    cls = "ok" if all_pass else "bad"
    h.append(
        f"<p><span class='pill {cls}'>Best leverage: {sl:.1f}×</span> "
        f"— {sweet['reason']}</p>"
        "<table><tr><th>CAGR</th><th>Sharpe</th><th>Max DD</th>"
        "<th>Sortino</th><th>Calmar</th><th>Vol</th></tr>"
        f"<tr><td class='{ 'pos' if pm['cagr_pct']>0 else 'neg' }'>{_fmt_pct(pm['cagr_pct'])}</td>"
        f"<td>{_fmt(pm['sharpe'])}</td>"
        f"<td class='neg'>{_fmt_pct(pm['max_dd_pct'])}</td>"
        f"<td>{_fmt(pm['sortino'])}</td>"
        f"<td>{_fmt(pm['calmar'])}</td>"
        f"<td>{_fmt_pct(pm['vol_pct'])}</td></tr></table>"
    )

    # Per-fold detail per leverage
    h.append("<h2>Per-fold detail</h2>")
    for L in LEVERAGES:
        h.append(f"<h3>Leverage {L:.1f}×</h3>")
        h.append("<table><tr><th>Test window</th><th>CAGR</th><th>Sharpe</th>"
                 "<th>Max DD</th><th>Vol</th></tr>")
        for fm in sweep[L]["folds"]:
            m = fm.metrics
            cls_cagr = "pos" if m["cagr_pct"] > 0 else "neg"
            h.append(
                f"<tr><td class='l'>{fm.test_start} → {fm.test_end}</td>"
                f"<td class='{cls_cagr}'>{_fmt_pct(m['cagr_pct'])}</td>"
                f"<td>{_fmt(m['sharpe'])}</td>"
                f"<td class='neg'>{_fmt_pct(m['max_dd_pct'])}</td>"
                f"<td>{_fmt_pct(m['vol_pct'])}</td></tr>"
            )
        h.append("</table>")

    # Caveats
    h.append("<h2>Caveats (HONEST)</h2>")
    h.append("<ul>")
    h.append("<li><b>Sharpe ceiling:</b> unreachable via sizing. The baseline "
             "pooled OOS Sharpe is ~5.24 and every row in the sweep prints "
             "the same number. Hitting Sharpe > 6.0 requires a SIGNAL "
             "improvement (tighter entry rules, better regime filter, or "
             "replacing a lower-Sharpe stream with a higher-Sharpe one), "
             "not more leverage.</li>")
    h.append("<li><b>Vol drag / Itô correction:</b> the L · r_t model is a "
             "first-order approximation. At 3× daily leverage on an ~5.5% "
             "annualised vol base portfolio, the Itô correction is "
             "≈ L²·σ²/2 ≈ 9 × 0.003 / 2 = 0.0135 ≈ 1.4%/yr of CAGR drag. "
             "Small relative to the 100%+ targets but not zero.</li>")
    h.append("<li><b>Financing:</b> no borrow cost, no regulatory margin. "
             "Reg-T on defined-risk options spreads allows ~2× gross "
             "without portfolio margin; beyond that needs portfolio "
             "margin and will accrue financing at ~SOFR + 50-100bp.</li>")
    h.append("<li><b>Path risk:</b> a 3× leveraged sleeve on a 2.6% baseline "
             "DD portfolio delivers ~7.8% DD in the backtest — but the "
             "backtest DD is the WALK-FORWARD pooled maximum, not the "
             "worst-ever single-day loss. A tail event not represented in "
             "the 2020-2025 window would cut through the nominal DD cap.</li>")
    h.append("<li><b>No per-stream leverage cap:</b> this is a blanket "
             "uniform scale. A smarter implementation would leverage the "
             "low-correlation streams more aggressively and hedges less, "
             "but that changes the risk profile and is not what the task "
             "asked for.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[exp2110] loading 5-stream panel (cached real-data)…", flush=True)
    df = load_streams()
    print(f"[exp2110] panel: {df.shape[0]} days × {df.shape[1]} streams")
    print(f"[exp2110] columns: {list(df.columns)}")

    sweep: Dict[float, Dict] = {}
    for L in LEVERAGES:
        print(f"[exp2110] walk-forward at leverage {L:.1f}×…", flush=True)
        folds, pooled_series, pooled_m = walk_forward_static(
            df, leverage=L, train_days=TRAIN_DAYS, test_days=TEST_DAYS,
        )
        targets = evaluate_target(pooled_m)
        sweep[L] = {
            "folds": folds,
            "pooled": pooled_m,
            "targets": targets,
        }
        print(f"[exp2110]   n_folds={len(folds)}  "
              f"CAGR={pooled_m['cagr_pct']:.2f}%  "
              f"Sharpe={pooled_m['sharpe']:.2f}  "
              f"DD={pooled_m['max_dd_pct']:.2f}%  "
              f"targets: cagr={targets['cagr']} sharpe={targets['sharpe']} dd={targets['dd']}")

    sweet = find_sweet_spot(sweep)
    print(f"[exp2110] sweet spot: {sweet['leverage']:.1f}×  ({sweet['reason']})")

    html = render_html(sweep, sweet, list(df.columns))
    REPORT_HTML.write_text(html)
    print(f"[exp2110] wrote {REPORT_HTML}")

    summary = {
        "experiment": "EXP-2110",
        "tag": "EXP-2110",
        "description": "Leveraged diversified portfolio — CAGR sweet-spot sweep",
        "baseline_source": "compass.exp2080_corr_regime.STATIC_WEIGHTS",
        "streams": list(df.columns),
        "weights": STATIC_WEIGHTS,
        "leverages_tested": LEVERAGES,
        "walk_forward": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "n_folds": len(sweep[LEVERAGES[0]]["folds"]),
        },
        "targets": {
            "cagr_min": TARGET_CAGR,
            "sharpe_min": TARGET_SHARPE,
            "dd_max": TARGET_DD,
        },
        "sweep": {
            f"{L:.1f}x": {
                "pooled": sweep[L]["pooled"],
                "targets": sweep[L]["targets"],
                "folds": [
                    {
                        "test_start": f.test_start,
                        "test_end": f.test_end,
                        "metrics": f.metrics,
                    }
                    for f in sweep[L]["folds"]
                ],
            }
            for L in LEVERAGES
        },
        "sweet_spot": {
            "leverage": sweet["leverage"],
            "reason": sweet["reason"],
            "metrics": sweep[sweet["leverage"]]["pooled"],
            "targets": sweep[sweet["leverage"]]["targets"],
        },
        "honest_note": (
            "Sharpe is leverage-invariant by construction "
            "(L·μ / L·σ = μ/σ). The target Sharpe > 6.0 is UNREACHABLE "
            "via uniform leverage. Baseline pooled OOS Sharpe is ~5.24; "
            "every row of the sweep prints the same Sharpe. Closing the "
            "Sharpe gap requires a signal-quality improvement, not a "
            "sizing change."
        ),
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[exp2110] wrote {REPORT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
