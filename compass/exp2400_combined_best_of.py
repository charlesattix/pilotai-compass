"""
EXP-2400 — Combined Best-Of: Ledoit-Wolf + 3% DD Circuit Breaker.

Combines the two winning techniques from the Wave-3 Ledoit-Wolf and
circuit-breaker experiments into a single portfolio configuration and
walk-forward tests it on the same 7-stream cube:

  EXP-2360 showed Ledoit-Wolf covariance + risk-parity weighting
           lifted the pooled-OOS Sharpe from 8.30 (sample covariance)
           to 11.73 on the 7-stream cube with 15% vol target.
  EXP-2370 showed a 3% trailing-drawdown circuit breaker (flatten
           mode, 20-day window) cut the pooled max DD from 24.5% to
           6.77% at CAGR 132% / Sharpe 5.4 on the same stream set but
           with inverse-vol weighting.

EXP-2400 stacks both:

  1. Ledoit-Wolf shrunk covariance estimated on every 252-day
     training window (sklearn.covariance.LedoitWolf).
  2. Equal-risk-contribution (risk-parity) weights solved from that
     Σ via compass.exp2360_robust_cov.risk_parity_weights.
  3. 15% annualised vol target — portfolio scaled per-fold by the
     ratio TARGET_VOL / train-window realised vol (clipped to
     [0.1×, 5×]).
  4. 3% trailing-drawdown circuit breaker from
     compass.exp2370_dd_circuit_breaker.apply_circuit_breaker, run
     causally inside each OOS test window. Flatten mode (0× leverage
     while tripped).
  5. Same 20-fold walk-forward (train 252 / test 63) used by both
     upstream experiments.

The pre-registered question:
  * Can the combined config clear CAGR > 100%, Sharpe > 6.0, and
    Max DD < 12% SIMULTANEOUSLY on pooled OOS data?

REAL DATA — Rule Zero:
  * 7-stream cube via compass.exp2360_robust_cov.build_seven_stream_cube
    (real EXP-1220 tape, EXP-1770 GLD/SLV calendars, EXP-2020 cross-vol
    arb, Crisis Alpha v5, and fresh XLF/XLI credit spreads).

Outputs:
  compass/exp2400_combined_best_of.py            (this file)
  compass/reports/exp2400_combined_best_of.json
  compass/reports/exp2400_combined_best_of.html

Tag: EXP-2400
Run: python3 -m compass.exp2400_combined_best_of
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2400_combined_best_of.json"
REPORT_HTML = REPORT_DIR / "exp2400_combined_best_of.html"

from compass.exp2360_robust_cov import (
    build_seven_stream_cube,
    cov_ledoit_wolf,
    cov_sample,
    risk_parity_weights,
    TRADING_DAYS,
    TRAIN_DAYS,
    TEST_DAYS,
    TARGET_VOL_ANNUAL,
)
from compass.exp2370_dd_circuit_breaker import apply_circuit_breaker

DD_THRESHOLD = 0.03
DD_WINDOW = 20
DD_MODE = "flatten"
VOL_SCALE_CLIP = (0.1, 5.0)

TARGET_CAGR = 1.00
TARGET_SHARPE = 6.0
TARGET_MAX_DD = 0.12


# ── Metrics ────────────────────────────────────────────────────────────


def metrics(daily: pd.Series, label: str = "") -> Dict[str, float]:
    d = daily.dropna()
    n = int(len(d))
    if n < 2:
        return dict(label=label, n=n, cagr_pct=0.0, sharpe=0.0, sortino=0.0,
                    max_dd_pct=0.0, calmar=0.0, vol_pct=0.0)
    eq = (1 + d).cumprod()
    yrs = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / max(yrs, 1e-9)) - 1) if eq.iloc[-1] > 0 else -1.0
    mu = float(d.mean())
    sigma = float(d.std(ddof=1))
    sharpe = (mu / sigma) * math.sqrt(TRADING_DAYS) if sigma > 1e-12 else 0.0
    down = d[d < 0]
    ds = float(down.std(ddof=1)) if len(down) > 1 else sigma
    sortino = (mu / ds) * math.sqrt(TRADING_DAYS) if ds > 1e-12 else 0.0
    dd = float((1 - eq / eq.cummax()).max())
    return dict(
        label=label,
        n=n,
        cagr_pct=round(cagr * 100, 3),
        sharpe=round(sharpe, 3),
        sortino=round(sortino, 3),
        max_dd_pct=round(dd * 100, 3),
        calmar=round(cagr / dd, 3) if dd > 1e-9 else 0.0,
        vol_pct=round(sigma * math.sqrt(TRADING_DAYS) * 100, 3),
    )


# ── Walk-forward with the combined config ────────────────────────────


def walk_forward_combined(df: pd.DataFrame,
                          use_circuit: bool = True,
                          use_ledoit: bool = True
                          ) -> Tuple[List[Dict], pd.Series, pd.Series]:
    """Walk-forward the 7-stream cube with the combined Ledoit-Wolf +
    risk-parity + vol-target + 3% circuit-breaker config.

    Also supports ablation: set use_circuit=False or use_ledoit=False
    to build the component variants for the comparison table.
    """
    cols = list(df.columns)
    n = len(df)
    folds: List[Dict] = []
    pooled_idx: List = []
    pooled_vals: List[float] = []
    pooled_lev: List[pd.Series] = []

    cov_fn = cov_ledoit_wolf if use_ledoit else cov_sample

    i = TRAIN_DAYS
    fold_ix = 0
    while i + TEST_DAYS <= n:
        train = df.iloc[i - TRAIN_DAYS:i]
        test = df.iloc[i:i + TEST_DAYS]

        # Covariance + risk-parity weights
        Sigma = cov_fn(train.values)
        w = risk_parity_weights(Sigma)

        # Vol-target from training window
        train_port = train.values @ w
        train_vol = float(np.std(train_port, ddof=1)) * math.sqrt(TRADING_DAYS)
        if train_vol <= 1e-10:
            scale = 1.0
        else:
            scale = TARGET_VOL_ANNUAL / train_vol
        scale = float(np.clip(scale, VOL_SCALE_CLIP[0], VOL_SCALE_CLIP[1]))

        # Raw OOS scaled portfolio
        raw_oos = pd.Series(test.values @ w * scale, index=test.index)

        # Circuit breaker (per-fold, causal)
        if use_circuit:
            levered, lev_path = apply_circuit_breaker(
                raw_oos, threshold=DD_THRESHOLD, mode=DD_MODE,
                window=DD_WINDOW, base_leverage=1.0,
            )
        else:
            levered = raw_oos
            lev_path = pd.Series(1.0, index=raw_oos.index)

        fold_m = metrics(levered, f"fold_{fold_ix}")
        trip_days = int((lev_path < 1.0 - 1e-9).sum())
        folds.append({
            "fold": fold_ix,
            "test_start": str(test.index[0].date()),
            "test_end": str(test.index[-1].date()),
            "vol_scale": round(scale, 4),
            "trip_days": trip_days,
            "trip_pct": round(trip_days / len(test) * 100, 2),
            "metrics": fold_m,
            "weights": {cols[j]: round(float(w[j]), 4) for j in range(len(cols))},
        })
        pooled_idx.extend(test.index.tolist())
        pooled_vals.extend(levered.tolist())
        pooled_lev.append(lev_path)

        fold_ix += 1
        i += TEST_DAYS

    pooled = pd.Series(pooled_vals, index=pooled_idx, dtype=float)
    lev_series = pd.concat(pooled_lev).sort_index() if pooled_lev else pd.Series(dtype=float)
    return folds, pooled, lev_series


# ── Target check ───────────────────────────────────────────────────────


def check_targets(m: Dict[str, float]) -> Dict[str, bool]:
    cagr_ok = (m["cagr_pct"] / 100.0) >= TARGET_CAGR
    sharpe_ok = m["sharpe"] >= TARGET_SHARPE
    dd_ok = (m["max_dd_pct"] / 100.0) < TARGET_MAX_DD
    return {
        "cagr_ge_100": cagr_ok,
        "sharpe_ge_6": sharpe_ok,
        "max_dd_lt_12": dd_ok,
        "all_three": cagr_ok and sharpe_ok and dd_ok,
    }


# ── HTML ───────────────────────────────────────────────────────────────


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(payload: Dict) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111}
    h1{border-bottom:3px solid #0f4c3a}
    h2{margin-top:2em;color:#0f4c3a}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#0f4c3a;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#0f4c3a}
    .pill.ok{background:#0a7d1f}
    .pill.bad{background:#c0392b}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2400 Combined Best-Of</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2400 — Combined Best-Of (Ledoit-Wolf + 3% DD Circuit Breaker)</h1>",
        "<p class='muted'>Stacks EXP-2360's Ledoit-Wolf covariance + "
        "risk-parity weights on top of EXP-2370's 3% trailing-DD "
        "circuit breaker. 7-stream cube, 15% vol target, walk-forward "
        "20 folds (train 252 / test 63).</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault + Yahoo data only</span></p>",
    ]

    # Pre-registered targets
    t = payload["targets"]
    r = payload["combined"]["pooled"]
    check = payload["combined"]["targets"]
    h.append("<h2>Pre-registered targets</h2>")
    h.append("<table><tr><th>Target</th><th>Required</th>"
             "<th>Combined pooled OOS</th><th>Pass</th></tr>")
    for key, label, req, actual, passed in [
        ("cagr_ge_100", "CAGR", "> 100%",
         f"{r['cagr_pct']:.2f}%", check["cagr_ge_100"]),
        ("sharpe_ge_6", "Sharpe", "> 6.0",
         _fmt(r["sharpe"]), check["sharpe_ge_6"]),
        ("max_dd_lt_12", "Max DD", "< 12%",
         f"{r['max_dd_pct']:.2f}%", check["max_dd_lt_12"]),
    ]:
        cls = "pos" if passed else "neg"
        h.append(
            f"<tr><td class='l'>{label}</td>"
            f"<td>{req}</td>"
            f"<td class='{cls}'>{actual}</td>"
            f"<td><span class='pill {'ok' if passed else 'bad'}'>"
            f"{'YES' if passed else 'NO'}</span></td></tr>"
        )
    h.append("</table>")
    all_three = check["all_three"]
    pill_all = ("<span class='pill ok'>ALL THREE TARGETS MET</span>"
                if all_three
                else "<span class='pill bad'>NOT ALL THREE TARGETS MET</span>")
    h.append(f"<p>{pill_all}</p>")

    # Ablation table
    h.append("<h2>Ablation — which component is doing the work?</h2>")
    h.append("<table><tr><th>Config</th><th>CAGR</th><th>Sharpe</th>"
             "<th>Max DD</th><th>Vol</th><th>Calmar</th>"
             "<th>Circuit trips (%)</th></tr>")
    for name in ("sample_only", "ledoit_only", "circuit_only", "combined"):
        block = payload[name]
        m = block["pooled"]
        trip = block.get("circuit_trip_pct", 0.0)
        trip_str = f"{trip:.2f}%" if trip else "—"
        h.append(
            f"<tr><td class='l'><b>{name}</b></td>"
            f"<td class='{ 'pos' if m['cagr_pct']>0 else 'neg' }'>{m['cagr_pct']:.2f}%</td>"
            f"<td>{_fmt(m['sharpe'])}</td>"
            f"<td class='neg'>{m['max_dd_pct']:.2f}%</td>"
            f"<td>{m['vol_pct']:.2f}%</td>"
            f"<td>{_fmt(m['calmar'])}</td>"
            f"<td>{trip_str}</td></tr>"
        )
    h.append("</table>")
    h.append("<p class='muted'>sample_only = sample covariance, no circuit. "
             "ledoit_only = Ledoit-Wolf covariance, no circuit. "
             "circuit_only = sample covariance + 3% circuit breaker. "
             "combined = Ledoit-Wolf covariance + 3% circuit breaker.</p>")

    # Per-fold table (combined)
    h.append("<h2>Per-fold detail — combined config</h2>")
    h.append("<table><tr><th>Fold</th><th>Test window</th>"
             "<th>Vol scale</th>"
             "<th>CAGR</th><th>Sharpe</th><th>Max DD</th>"
             "<th>Vol</th><th>Trip days (%)</th></tr>")
    for f in payload["combined"]["folds"]:
        m = f["metrics"]
        h.append(
            f"<tr><td>{f['fold']}</td>"
            f"<td class='l'>{f['test_start']} → {f['test_end']}</td>"
            f"<td>{f['vol_scale']}</td>"
            f"<td class='{ 'pos' if m['cagr_pct']>0 else 'neg' }'>{m['cagr_pct']:.2f}%</td>"
            f"<td>{_fmt(m['sharpe'])}</td>"
            f"<td class='neg'>{m['max_dd_pct']:.2f}%</td>"
            f"<td>{m['vol_pct']:.2f}%</td>"
            f"<td>{f['trip_days']} ({f['trip_pct']:.1f}%)</td></tr>"
        )
    h.append("</table>")

    # Methodology + honest caveats
    h.append("<h2>Methodology &amp; honest caveats</h2>")
    h.append("<ul>")
    h.append(f"<li><b>7-stream cube</b> built via "
             "<code>compass.exp2360_robust_cov.build_seven_stream_cube</code> "
             "— same real-data streams used by EXP-2360 and (up to the "
             "vol_arb/cross_vol naming) EXP-2370.</li>")
    h.append(f"<li><b>Covariance:</b> Ledoit-Wolf shrinkage estimator "
             "from sklearn.covariance. Estimated on each 252-day training "
             "window; never peeks at OOS data.</li>")
    h.append(f"<li><b>Weights:</b> equal-risk-contribution (risk parity) "
             "solved from the Ledoit-Wolf Σ via the Chaves-Hsu-Li-Shakernia "
             "fixed-point iteration from EXP-2360.</li>")
    h.append(f"<li><b>Vol target:</b> {TARGET_VOL_ANNUAL*100:.0f}%/yr. Scale "
             f"clipped to [{VOL_SCALE_CLIP[0]}×, {VOL_SCALE_CLIP[1]}×] to "
             "prevent runaway leverage on near-zero-vol training windows.</li>")
    h.append(f"<li><b>Circuit breaker:</b> {DD_THRESHOLD*100:.0f}% trailing "
             f"drawdown over {DD_WINDOW}-day window, {DD_MODE} mode. "
             "Strictly causal — uses only returns through day t-1 to "
             "decide whether to trade on day t. Flattens to 0× leverage "
             "while tripped; resumes 1× when trailing DD falls back "
             "under threshold.</li>")
    h.append("<li><b>Honest caveat — inherited stream-construction bias:</b> "
             "the 7-stream cube contains trade streams whose daily P&amp;L "
             "is attributed on exit date (EXP-1220, XLF/XLI) or smeared "
             "across holding windows. Both conventions produce narrower "
             "daily-return distributions than minute-bar truth, which "
             "inflates daily Sharpe ratios. The EXP-2360 Ledoit-Wolf "
             "pooled Sharpe of 11.73 and the numbers in this report are "
             "therefore proxy-inflated. Treat them as relative rankings "
             "between configs, not absolute risk-adjusted returns.</li>")
    h.append("<li><b>Honest caveat — circuit breaker may not trip:</b> "
             "Ledoit-Wolf + risk parity already produces very low pooled "
             "DD (~2.2% in EXP-2360). A 3% threshold is <i>above</i> that "
             "floor, so the circuit breaker may never fire on this "
             "config. The per-fold trip counts in the detail table "
             "measure how often it actually does.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[exp2400] building 7-stream cube …", flush=True)
    df = build_seven_stream_cube()
    print(f"[exp2400] cube {df.shape}  {df.index[0].date()} → {df.index[-1].date()}")
    print(f"[exp2400] streams: {list(df.columns)}")

    # Four variants for the ablation table:
    variants: Dict[str, Dict] = {}

    for name, use_circuit, use_ledoit in [
        ("sample_only",  False, False),
        ("ledoit_only",  False, True),
        ("circuit_only", True,  False),
        ("combined",     True,  True),
    ]:
        print(f"\n[exp2400] running variant {name} "
              f"(circuit={use_circuit}, ledoit={use_ledoit}) …", flush=True)
        folds, pooled, lev = walk_forward_combined(
            df, use_circuit=use_circuit, use_ledoit=use_ledoit,
        )
        pooled_m = metrics(pooled, label=name)
        trip_pct = float((lev < 1.0 - 1e-9).mean() * 100) if len(lev) > 0 else 0.0
        variants[name] = {
            "pooled": pooled_m,
            "folds": folds,
            "circuit_trip_pct": round(trip_pct, 2),
            "targets": check_targets(pooled_m),
        }
        print(f"[exp2400]   pooled  CAGR={pooled_m['cagr_pct']:.2f}% "
              f"Sharpe={pooled_m['sharpe']:.2f}  DD={pooled_m['max_dd_pct']:.2f}% "
              f"Vol={pooled_m['vol_pct']:.2f}%  trip={trip_pct:.2f}%")

    payload = {
        "experiment": "EXP-2400",
        "tag": "EXP-2400",
        "description": ("Combined best-of: Ledoit-Wolf covariance + "
                        "risk parity + 15% vol target + 3% trailing-DD "
                        "circuit breaker on the 7-stream cube"),
        "data_sources": {
            "seven_stream_cube": (
                "compass.exp2360_robust_cov.build_seven_stream_cube "
                "(real EXP-1220 tape + GLD/SLV calendars + cross-vol arb + "
                "Crisis Alpha v5 + fresh XLF/XLI credit spreads)"
            ),
            "covariance": "sklearn.covariance.LedoitWolf",
            "weights": "equal-risk-contribution (risk parity) via Chaves-Hsu-Li-Shakernia",
            "circuit_breaker": (
                f"{DD_THRESHOLD*100:.0f}% trailing drawdown, "
                f"{DD_WINDOW}-day window, {DD_MODE} mode — "
                "compass.exp2370_dd_circuit_breaker.apply_circuit_breaker"
            ),
        },
        "config": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "target_vol_annual": TARGET_VOL_ANNUAL,
            "vol_scale_clip": list(VOL_SCALE_CLIP),
            "dd_threshold": DD_THRESHOLD,
            "dd_window": DD_WINDOW,
            "dd_mode": DD_MODE,
            "trading_days": TRADING_DAYS,
        },
        "targets": {
            "cagr_min": TARGET_CAGR,
            "sharpe_min": TARGET_SHARPE,
            "max_dd_max": TARGET_MAX_DD,
        },
        "streams": list(df.columns),
        **variants,
    }

    html = render_html(payload)
    REPORT_HTML.write_text(html)
    print(f"\n[exp2400] wrote {REPORT_HTML}")

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2400] wrote {REPORT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
