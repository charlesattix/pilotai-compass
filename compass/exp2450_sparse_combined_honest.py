"""
EXP-2450 — Sparse-Cube Re-run of EXP-2400 Combined Stack (Honest Edition).

Context
-------
EXP-2400 reported a pooled-OOS Sharpe of 11.73 for the combined
Ledoit-Wolf + 3% DD circuit breaker + 15% vol-target config on the
7-stream cube. EXP-2390 subsequently proved that this number was
inflated by the XLF/XLI trade-P&L smearing convention: attributing
each trade's P&L uniformly across its holding window collapses daily
return variance, which drives per-stream Sharpe artificially high.
Rebuilding the same cube with the SPARSE (exit-date) convention drops
the Ledoit-Wolf pooled Sharpe from 11.73 to 6.87 — a 1.71× inflation
factor.

EXP-2450 is the definitive honest answer: re-run EXP-2400's combined
stack (Ledoit-Wolf covariance + risk parity + 15% vol target + 3%
trailing-DD circuit breaker) on the SPARSE cube and apply the EXP-2420
transaction-cost model to get net numbers.

Steps

  1. Rebuild the 7-stream cube with sparse XLF/XLI exit-date P&L
     (reuse EXP-2390's `sparse_xlf_xli` helper — same XLF/XLI trade
     tape, different attribution convention).
  2. Walk-forward the sparse cube with four ablation variants:
        sample_only    sample cov,        no circuit
        ledoit_only    Ledoit-Wolf cov,   no circuit
        circuit_only   sample cov,        3% flatten circuit
        combined       Ledoit-Wolf cov,   3% flatten circuit
  3. Per fold and pooled: compute metrics on the SPARSE daily series.
  4. Apply EXP-2420's annual cost drag (22.205% / yr from that
     report's net_metrics block) via `net_sharpe_from_drag` to derive
     net Sharpe / net CAGR for each variant. The vol stays unchanged
     because costs are ~deterministic.
  5. Check the pre-registered targets (CAGR > 100%, Sharpe > 6.0,
     Max DD < 12%) on both the sparse GROSS and sparse NET numbers.

Rule Zero: every input is real. Sparse cube uses the same
IronVault-backed XLF/XLI trades as EXP-2400 and EXP-2390 — only the
daily-attribution convention changes.

Outputs:
  compass/exp2450_sparse_combined_honest.py            (this file)
  compass/reports/exp2450_sparse_combined_honest.json
  compass/reports/exp2450_sparse_combined_honest.html

Tag: EXP-2450
Run: python3 -m compass.exp2450_sparse_combined_honest
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
REPORT_JSON = REPORT_DIR / "exp2450_sparse_combined_honest.json"
REPORT_HTML = REPORT_DIR / "exp2450_sparse_combined_honest.html"

# Reuse the walk-forward engine from EXP-2400 verbatim so the only
# thing that changes vs the smeared run is the cube the engine operates on.
from compass.exp2400_combined_best_of import (
    walk_forward_combined,
    metrics,
    check_targets,
    TARGET_CAGR,
    TARGET_SHARPE,
    TARGET_MAX_DD,
    DD_THRESHOLD,
    DD_WINDOW,
    DD_MODE,
    VOL_SCALE_CLIP,
)
from compass.exp2360_robust_cov import (
    TARGET_VOL_ANNUAL,
    TRAIN_DAYS,
    TEST_DAYS,
    TRADING_DAYS,
)
from compass.exp2390_robust_cov_audit import sparse_xlf_xli, build_cube
from compass.exp2080_corr_regime import load_streams
from compass.exp2420_transaction_costs import net_sharpe_from_drag

# Annual cost drag from EXP-2420 (see reports/exp2420_transaction_costs.json
# net_metrics.drag_pct — 22.205%). Derived from real IronVault
# bid-ask proxies + real Yahoo ADV + the EXP-2420 slippage model.
EXP2420_ANNUAL_DRAG_PCT = 22.205


# ── Sparse cube build ──────────────────────────────────────────────────


def build_sparse_seven_stream_cube() -> pd.DataFrame:
    print("[exp2450] loading 5-stream cache …", flush=True)
    base = load_streams()
    print(f"[exp2450] base shape: {base.shape}")

    print("[exp2450] building SPARSE XLF/XLI exit-date streams …", flush=True)
    xlf_sp, xli_sp = sparse_xlf_xli(base.index)
    print(f"[exp2450] sparse XLF nonzero days: {(xlf_sp != 0).sum()}, "
          f"annualised vol: {xlf_sp.std() * math.sqrt(TRADING_DAYS) * 100:.2f}%")
    print(f"[exp2450] sparse XLI nonzero days: {(xli_sp != 0).sum()}, "
          f"annualised vol: {xli_sp.std() * math.sqrt(TRADING_DAYS) * 100:.2f}%")

    cube = build_cube(base, xlf_sp, xli_sp)
    print(f"[exp2450] sparse cube shape: {cube.shape}")
    print(f"[exp2450] columns: {list(cube.columns)}")
    return cube


# ── Run the four ablation variants on the sparse cube ────────────────


def run_variants(cube: pd.DataFrame) -> Dict[str, Dict]:
    variants: Dict[str, Dict] = {}
    for name, use_circuit, use_ledoit in [
        ("sample_only",  False, False),
        ("ledoit_only",  False, True),
        ("circuit_only", True,  False),
        ("combined",     True,  True),
    ]:
        print(f"\n[exp2450] running {name} "
              f"(circuit={use_circuit}, ledoit={use_ledoit}) …", flush=True)
        folds, pooled, lev = walk_forward_combined(
            cube, use_circuit=use_circuit, use_ledoit=use_ledoit,
        )
        pooled_m = metrics(pooled, label=name)
        trip_pct = float((lev < 1.0 - 1e-9).mean() * 100) if len(lev) > 0 else 0.0
        variants[name] = {
            "pooled": pooled_m,
            "folds": folds,
            "circuit_trip_pct": round(trip_pct, 2),
            "targets_gross": check_targets(pooled_m),
        }
        print(f"[exp2450]   sparse pooled  CAGR={pooled_m['cagr_pct']:.2f}%  "
              f"Sharpe={pooled_m['sharpe']:.2f}  DD={pooled_m['max_dd_pct']:.2f}%  "
              f"Vol={pooled_m['vol_pct']:.2f}%  trip={trip_pct:.2f}%")
    return variants


# ── Apply EXP-2420 cost drag ──────────────────────────────────────────


def apply_cost_drag(variants: Dict[str, Dict]) -> Dict[str, Dict]:
    for name, v in variants.items():
        m = v["pooled"]
        net = net_sharpe_from_drag(
            gross_sharpe=m["sharpe"],
            gross_cagr_pct=m["cagr_pct"],
            vol_pct=m["vol_pct"],
            annual_drag_pct=EXP2420_ANNUAL_DRAG_PCT,
        )
        net_pooled = {
            "n": m["n"],
            "cagr_pct": net["net_cagr_pct"],
            "sharpe": net["net_sharpe"],
            "max_dd_pct": m["max_dd_pct"],  # DD is unchanged by cost drag assumption
            "vol_pct": m["vol_pct"],
            "sortino": float("nan"),
            "calmar": (
                (net["net_cagr_pct"] / 100.0) / (m["max_dd_pct"] / 100.0)
                if m["max_dd_pct"] > 1e-9 else 0.0
            ),
        }
        v["net"] = net_pooled
        v["net_drag_detail"] = net
        v["targets_net"] = check_targets(net_pooled)
    return variants


# ── HTML ───────────────────────────────────────────────────────────────


def _fmt(x: float, dp: int = 2) -> str:
    return f"{x:.{dp}f}" if np.isfinite(x) else "—"


def render_html(payload: Dict) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111}
    h1{border-bottom:3px solid #5c1f66}
    h2{margin-top:2em;color:#5c1f66}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#5c1f66;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#5c1f66}
    .pill.ok{background:#0a7d1f}
    .pill.bad{background:#c0392b}
    .pill.warn{background:#c07a1f}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2450 Sparse Combined Honest</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2450 — Sparse-Cube Combined Stack, Honest Edition</h1>",
        "<p class='muted'>EXP-2400 combined stack (Ledoit-Wolf + risk "
        "parity + 15% vol target + 3% trailing-DD circuit breaker) "
        "re-run on the SPARSE 7-stream cube (exit-date XLF/XLI "
        "attribution, no P&L smearing). Then NET numbers via EXP-2420 "
        "cost drag.</p>",
        "<p><span class='pill'>Rule Zero ✓ real IronVault + Yahoo data only</span> "
        f"<span class='pill warn'>Smeared-cube result (EXP-2400 Sharpe 11.73) "
        "is retracted as proxy-inflated; this is the definitive number.</span></p>",
    ]

    # Sparse vs smeared reference
    h.append("<h2>Smeared-cube reference → sparse-cube honest baseline</h2>")
    h.append("<table><tr><th>Source</th><th>Cube</th>"
             "<th>Config</th><th>Sharpe</th><th>CAGR</th>"
             "<th>Max DD</th><th>Vol</th></tr>")
    h.append(
        "<tr><td class='l'>EXP-2400 (retracted)</td>"
        "<td>smeared</td>"
        "<td class='l'>Ledoit-Wolf + circuit</td>"
        "<td class='neg'>11.73</td>"
        "<td>103.92%</td>"
        "<td>2.24%</td>"
        "<td>6.10%</td></tr>"
    )
    lw = payload["variants"]["ledoit_only"]["pooled"]
    h.append(
        "<tr><td class='l'>EXP-2390 (audit)</td>"
        "<td>sparse</td>"
        "<td class='l'>Ledoit-Wolf alone</td>"
        "<td>6.87</td>"
        "<td>101.83%</td>"
        "<td>4.21%</td>"
        "<td>10.32%</td></tr>"
    )
    h.append(
        "<tr><td class='l'><b>EXP-2450 (this report)</b></td>"
        "<td><b>sparse</b></td>"
        "<td class='l'><b>combined (L-W + circuit)</b></td>"
        f"<td class='pos'><b>{_fmt(payload['variants']['combined']['pooled']['sharpe'])}</b></td>"
        f"<td><b>{payload['variants']['combined']['pooled']['cagr_pct']:.2f}%</b></td>"
        f"<td><b>{payload['variants']['combined']['pooled']['max_dd_pct']:.2f}%</b></td>"
        f"<td><b>{payload['variants']['combined']['pooled']['vol_pct']:.2f}%</b></td>"
        "</tr>"
    )
    h.append("</table>")

    # Ablation: sparse gross vs sparse net
    h.append("<h2>Sparse cube — gross vs net ablation</h2>")
    h.append("<table><tr><th>Config</th>"
             "<th colspan='4'>Sparse GROSS (no costs)</th>"
             "<th colspan='4'>Sparse NET (22.2%/yr drag)</th></tr>"
             "<tr><th></th>"
             "<th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th>"
             "<th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Calmar</th>"
             "</tr>")
    for name in ("sample_only", "ledoit_only", "circuit_only", "combined"):
        v = payload["variants"][name]
        g = v["pooled"]; n = v["net"]
        h.append(
            f"<tr><td class='l'><b>{name}</b></td>"
            f"<td class='{ 'pos' if g['cagr_pct']>0 else 'neg' }'>{g['cagr_pct']:.2f}%</td>"
            f"<td>{_fmt(g['sharpe'])}</td>"
            f"<td class='neg'>{g['max_dd_pct']:.2f}%</td>"
            f"<td>{g['vol_pct']:.2f}%</td>"
            f"<td class='{ 'pos' if n['cagr_pct']>0 else 'neg' }'>{n['cagr_pct']:.2f}%</td>"
            f"<td>{_fmt(n['sharpe'])}</td>"
            f"<td class='neg'>{n['max_dd_pct']:.2f}%</td>"
            f"<td>{_fmt(n['calmar'])}</td></tr>"
        )
    h.append("</table>")

    # Pre-registered targets check
    h.append("<h2>Pre-registered targets — CAGR &gt; 100%, Sharpe &gt; 6.0, Max DD &lt; 12%</h2>")
    h.append("<table><tr><th>Config</th>"
             "<th colspan='4'>Sparse GROSS targets</th>"
             "<th colspan='4'>Sparse NET targets</th></tr>"
             "<tr><th></th>"
             "<th>CAGR</th><th>Sharpe</th><th>DD</th><th>ALL</th>"
             "<th>CAGR</th><th>Sharpe</th><th>DD</th><th>ALL</th></tr>")
    for name in ("sample_only", "ledoit_only", "circuit_only", "combined"):
        v = payload["variants"][name]
        tg = v["targets_gross"]; tn = v["targets_net"]

        def flag(ok: bool) -> str:
            return (f"<span class='pill ok'>✓</span>" if ok
                    else f"<span class='pill bad'>✗</span>")

        h.append(
            f"<tr><td class='l'><b>{name}</b></td>"
            f"<td>{flag(tg['cagr_ge_100'])}</td>"
            f"<td>{flag(tg['sharpe_ge_6'])}</td>"
            f"<td>{flag(tg['max_dd_lt_12'])}</td>"
            f"<td>{flag(tg['all_three'])}</td>"
            f"<td>{flag(tn['cagr_ge_100'])}</td>"
            f"<td>{flag(tn['sharpe_ge_6'])}</td>"
            f"<td>{flag(tn['max_dd_lt_12'])}</td>"
            f"<td>{flag(tn['all_three'])}</td></tr>"
        )
    h.append("</table>")

    # Per-fold combined sparse
    h.append("<h2>Per-fold detail — combined sparse</h2>")
    h.append("<table><tr><th>Fold</th><th>Test window</th>"
             "<th>Vol scale</th>"
             "<th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Vol</th>"
             "<th>Trip days (%)</th></tr>")
    for f in payload["variants"]["combined"]["folds"]:
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

    # Methodology + caveats
    h.append("<h2>Methodology &amp; honest notes</h2>")
    h.append("<ul>")
    h.append("<li><b>Sparse attribution:</b> each XLF/XLI trade's "
             "pnl_pct_capital lands on its expiration date in the "
             "daily series, not smeared across the holding window. "
             "This is the convention used by EXP-2110 and EXP-2280 "
             "and matches the minute-bar truth better than smearing.</li>")
    h.append(f"<li><b>Engine reuse:</b> the walk-forward engine is "
             "<code>compass.exp2400_combined_best_of.walk_forward_combined</code> "
             "called verbatim. The ONLY thing that changed vs EXP-2400 "
             "is the cube the engine operates on.</li>")
    h.append(f"<li><b>Cost drag:</b> EXP-2420 reported 22.205%/yr total "
             "cost drag (bid-ask + commission + slippage) on the 7-stream "
             "portfolio using real IronVault spread proxies and real "
             "Yahoo ADV. This experiment subtracts that drag from the "
             "annualised mean of each variant — vol and max DD are "
             "assumed unchanged (cost is ~deterministic).</li>")
    h.append("<li><b>Why the combined and ledoit-only diverge on sparse:</b> "
             "the sparse cube has a pooled Ledoit-Wolf DD of ~4.2%, "
             "ABOVE the 3% circuit threshold. Unlike the smeared cube "
             "(2.24% DD, circuit never trips), here the breaker "
             "actually fires. Check the combined per-fold trip_days.</li>")
    h.append("<li><b>Why the combined-sparse Sharpe is lower than the "
             "smeared headline (11.73):</b> the smeared cube collapsed "
             "daily std by attributing each trade uniformly across 20-30 "
             "business days. The sparse cube restores the true "
             "end-of-period variance, which lowers Sharpe but is closer "
             "to what a production strategy would actually experience.</li>")
    h.append("<li><b>Honest caveat:</b> even the sparse Sharpe has some "
             "proxy bias because the non-XLF/XLI streams (EXP-1220 tape, "
             "EXP-1770 calendars, Crisis Alpha v5, EXP-2020 cross-vol arb) "
             "may use their own attribution conventions. A full "
             "minute-bar rebuild would further move the number, probably "
             "down, but by an amount that is hard to estimate without "
             "doing the work.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cube = build_sparse_seven_stream_cube()
    variants = run_variants(cube)
    variants = apply_cost_drag(variants)

    payload = {
        "experiment": "EXP-2450",
        "tag": "EXP-2450",
        "description": ("Sparse-cube re-run of EXP-2400 combined stack, "
                        "with EXP-2420 cost drag applied to derive the "
                        "definitive honest net numbers"),
        "data_sources": {
            "base_cube": "compass.exp2080_corr_regime.load_streams (5-stream cache)",
            "sparse_xlf_xli": (
                "compass.exp2390_robust_cov_audit.sparse_xlf_xli "
                "(exit-date attribution, not smeared)"
            ),
            "walk_forward_engine": (
                "compass.exp2400_combined_best_of.walk_forward_combined"
            ),
            "cost_model": (
                "compass.exp2420_transaction_costs.net_sharpe_from_drag with "
                "annual_drag_pct=22.205 (EXP-2420 net_metrics.drag_pct, "
                "derived from real IronVault bid-ask proxies + real Yahoo ADV)"
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
            "cost_drag_pct": EXP2420_ANNUAL_DRAG_PCT,
        },
        "targets": {
            "cagr_min": TARGET_CAGR,
            "sharpe_min": TARGET_SHARPE,
            "max_dd_max": TARGET_MAX_DD,
        },
        "smeared_reference": {
            "ledoit_wolf_smeared_sharpe": 11.73,
            "ledoit_wolf_smeared_cagr_pct": 103.92,
            "ledoit_wolf_smeared_dd_pct": 2.24,
            "inflation_factor_vs_sparse": 1.71,
            "source": "EXP-2390 walk_forward_reproduction",
        },
        "streams": list(cube.columns),
        "variants": variants,
    }

    html = render_html(payload)
    REPORT_HTML.write_text(html)
    print(f"\n[exp2450] wrote {REPORT_HTML}")

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2450] wrote {REPORT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
