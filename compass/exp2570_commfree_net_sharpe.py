"""
EXP-2570 — Commission-Free Broker Net Sharpe
=============================================

EXP-2510 showed that commission is 37% of total execution drag on the
7-stream portfolio at 3× leverage: $8,273 of $22,205, or 827 bps of
the 2,221 bps annual drag. EXP-2470 showed that stacked execution
optimisation (A+B+C+D — limit-at-mid, patient execution window,
route reallocation, multi-leg combo orders) removes an additional
~503 bps.

This experiment combines both findings and computes the net Sharpe
under each drag configuration on the SAME sparse-cube gross numbers
used throughout the walk-forward audit trail.

Drag scenarios
--------------
  1. GROSS                                0 bps
  2. IBKR baseline                     2,221 bps  (EXP-2510 full cost)
  3. Commfree, same B/A                1,394 bps  (kill 827 bps commission)
  4. Commfree + exec opt A+B+C+D         891 bps  (−503 bps from EXP-2470)   ← TARGET
  5. Commfree, realistic PFOF +30% B/A 1,519 bps  (upper realistic on Alpaca-pilot)
  6. Commfree + exec opt + PFOF +30%   1,016 bps  (conservative target)

Sources
  * Gross sparse cube numbers — EXP-2510 sparse_gross block
        ledoit_only: Sharpe 6.865, CAGR 101.832%, vol 10.321%
        combined:    Sharpe 6.721, CAGR 96.578%,  vol 10.146%
  * net_sharpe_from_drag helper — compass.exp2420_transaction_costs
        (identical formula used by EXP-2510 and EXP-2470)

The walk-forward methodology behind the gross numbers is already
validated by EXP-2280 / EXP-2360 / EXP-2450 on the real 7-stream
cube (5 cached + XLF/XLI from IronVault). This experiment does NOT
re-run the walk-forward — it performs the transformation from gross
to net under each drag scenario and compares them side by side.

Outputs
  compass/reports/exp2570_commfree_net_sharpe.json
  compass/reports/exp2570_commfree_net_sharpe.html
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.exp2420_transaction_costs import net_sharpe_from_drag

REPORT_JSON = ROOT / "compass" / "reports" / "exp2570_commfree_net_sharpe.json"
REPORT_HTML = ROOT / "compass" / "reports" / "exp2570_commfree_net_sharpe.html"


# ─────────────────────────────────────────────────────────────────────────────
# Load the authoritative gross numbers from EXP-2510
# ─────────────────────────────────────────────────────────────────────────────
def load_gross_numbers() -> Dict[str, Dict]:
    p = ROOT / "compass" / "reports" / "exp2510_broker_analysis.json"
    d = json.load(open(p))
    g = d["sparse_gross"]
    return {
        "ledoit_only": {
            "sharpe":   float(g["ledoit_only"]["sharpe"]),
            "cagr_pct": float(g["ledoit_only"]["cagr_pct"]),
            "vol_pct":  float(g["ledoit_only"]["vol_pct"]),
            "max_dd_pct": float(g["ledoit_only"]["max_dd_pct"]),
        },
        "combined": {
            "sharpe":   float(g["combined"]["sharpe"]),
            "cagr_pct": float(g["combined"]["cagr_pct"]),
            "vol_pct":  float(g["combined"]["vol_pct"]),
            "max_dd_pct": float(g["combined"]["max_dd_pct"]),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Drag scenarios — bps of annual capital drag
# ─────────────────────────────────────────────────────────────────────────────
# Source values from EXP-2510 baseline_breakdown.totals_pct
BIDASK_PCT      = 4.175
COMMISSION_PCT  = 8.273
SLIPPAGE_PCT    = 9.756
FULL_DRAG_PCT   = BIDASK_PCT + COMMISSION_PCT + SLIPPAGE_PCT

# EXP-2470 best stack savings (best_scenario: 2220.51 → 1717.77 bps)
EXEC_OPT_SAVINGS_PCT = 5.028

SCENARIOS: List[Dict] = [
    {
        "id": "gross",
        "label": "Gross (no costs)",
        "drag_pct": 0.0,
        "broker": "—",
        "note": "Pre-cost walk-forward ceiling.",
    },
    {
        "id": "ibkr_baseline",
        "label": "IBKR Pro (baseline)",
        "drag_pct": FULL_DRAG_PCT,
        "broker": "IBKR Pro fixed $0.65/contract",
        "note": "EXP-2510 baseline: bid-ask 4.18 + comm 8.27 + slip 9.76.",
    },
    {
        "id": "commfree_same_ba",
        "label": "Commission-free, same B/A",
        "drag_pct": BIDASK_PCT + SLIPPAGE_PCT,
        "broker": "Alpaca commission-free pilot (ideal PFOF)",
        "note": "Kill 8.27% commission, keep bid-ask and slippage constant.",
    },
    {
        "id": "commfree_plus_exec_opt",
        "label": "Commission-free + exec opt (A+B+C+D)",
        "drag_pct": BIDASK_PCT + SLIPPAGE_PCT - EXEC_OPT_SAVINGS_PCT,
        "broker": "Alpaca pilot + EXP-2470 execution stack",
        "note": "Adds -503 bps from limit-at-mid + patient window + route reallocation + multi-leg combos. The TARGET production configuration.",
    },
    {
        "id": "commfree_pfof_30pct",
        "label": "Commission-free, realistic PFOF +30% B/A",
        "drag_pct": BIDASK_PCT * 1.3 + SLIPPAGE_PCT,
        "broker": "Alpaca pilot, realistic PFOF penalty",
        "note": "Model realistic execution quality loss on 2-leg spreads routed to PFOF market makers.",
    },
    {
        "id": "commfree_pfof_30pct_plus_exec_opt",
        "label": "Commission-free + PFOF +30% + exec opt",
        "drag_pct": BIDASK_PCT * 1.3 + SLIPPAGE_PCT - EXEC_OPT_SAVINGS_PCT,
        "broker": "Alpaca pilot + realistic PFOF + EXP-2470 stack",
        "note": "Conservative target — both PFOF penalty AND the EXP-2470 savings applied.",
    },
    {
        "id": "commfree_rh_worst",
        "label": "Commission-free, Robinhood worst case (2× B/A)",
        "drag_pct": BIDASK_PCT * 2.0 + SLIPPAGE_PCT,
        "broker": "Robinhood (documented PFOF penalty + leg-in risk)",
        "note": "Worst-case reference only; RH does not natively support multi-leg combos.",
    },
]


def compute_scenario(gross: Dict[str, Dict], sc: Dict) -> Dict:
    out = {}
    for variant_name, g in gross.items():
        net = net_sharpe_from_drag(
            gross_sharpe=g["sharpe"],
            gross_cagr_pct=g["cagr_pct"],
            vol_pct=g["vol_pct"],
            annual_drag_pct=sc["drag_pct"],
        )
        out[variant_name] = {
            **net,
            "max_dd_pct": g["max_dd_pct"],
            "delta_sharpe_vs_gross": round(net["net_sharpe"] - g["sharpe"], 3),
        }
    return out


def main():
    print("[1/3] loading EXP-2510 gross sparse-cube numbers …")
    gross = load_gross_numbers()
    for k, v in gross.items():
        print(f"      {k:12s}  Sharpe {v['sharpe']:5.2f}  "
              f"CAGR {v['cagr_pct']:6.2f}%  vol {v['vol_pct']:5.2f}%  DD {v['max_dd_pct']:.2f}%")

    print("[2/3] applying drag scenarios …")
    results = []
    for sc in SCENARIOS:
        computed = compute_scenario(gross, sc)
        results.append({
            "id": sc["id"],
            "label": sc["label"],
            "broker": sc["broker"],
            "note":   sc["note"],
            "drag_pct": round(sc["drag_pct"], 3),
            "drag_bps": round(sc["drag_pct"] * 100, 1),
            "variants": computed,
        })
        a = computed["ledoit_only"]; b = computed["combined"]
        print(f"      {sc['id']:35s}  drag {sc['drag_pct']:6.2f}%  "
              f"LW net Sharpe {a['net_sharpe']:5.3f}  "
              f"combined {b['net_sharpe']:5.3f}")

    # Headline — the TARGET scenario
    target = next(r for r in results if r["id"] == "commfree_plus_exec_opt")
    baseline = next(r for r in results if r["id"] == "ibkr_baseline")
    gross_sc = next(r for r in results if r["id"] == "gross")

    headline = {
        "target_drag_bps": target["drag_bps"],
        "target_drag_pct": target["drag_pct"],
        "baseline_drag_bps": baseline["drag_bps"],
        "target_vs_baseline_savings_bps": round(
            baseline["drag_bps"] - target["drag_bps"], 1
        ),
        "ledoit_only": {
            "gross_sharpe":    gross["ledoit_only"]["sharpe"],
            "baseline_net":    baseline["variants"]["ledoit_only"]["net_sharpe"],
            "target_net":      target["variants"]["ledoit_only"]["net_sharpe"],
            "baseline_cagr":   baseline["variants"]["ledoit_only"]["net_cagr_pct"],
            "target_cagr":     target["variants"]["ledoit_only"]["net_cagr_pct"],
            "delta_sharpe_baseline_to_target": round(
                target["variants"]["ledoit_only"]["net_sharpe"]
                - baseline["variants"]["ledoit_only"]["net_sharpe"], 3
            ),
        },
        "combined": {
            "gross_sharpe":    gross["combined"]["sharpe"],
            "baseline_net":    baseline["variants"]["combined"]["net_sharpe"],
            "target_net":      target["variants"]["combined"]["net_sharpe"],
            "baseline_cagr":   baseline["variants"]["combined"]["net_cagr_pct"],
            "target_cagr":     target["variants"]["combined"]["net_cagr_pct"],
            "delta_sharpe_baseline_to_target": round(
                target["variants"]["combined"]["net_sharpe"]
                - baseline["variants"]["combined"]["net_sharpe"], 3
            ),
        },
    }

    payload = {
        "experiment": "EXP-2570",
        "name": "Commission-Free Broker Net Sharpe for the 7-Stream Portfolio",
        "generated": datetime.utcnow().isoformat(timespec="seconds"),
        "data_sources": {
            "gross_cube":     "compass/reports/exp2510_broker_analysis.json (sparse_gross)",
            "cost_breakdown": "compass/reports/exp2510_broker_analysis.json (baseline_breakdown)",
            "exec_opt_savings": "compass/reports/exp2470_execution_optimization.json (best_scenario)",
            "walk_forward":   "EXP-2360/EXP-2450 walk-forward on real 7-stream cube",
        },
        "cost_components_pct": {
            "bid_ask": BIDASK_PCT,
            "commission": COMMISSION_PCT,
            "slippage": SLIPPAGE_PCT,
            "full_drag": FULL_DRAG_PCT,
            "exec_opt_savings": EXEC_OPT_SAVINGS_PCT,
        },
        "gross": gross,
        "scenarios": results,
        "headline": headline,
        "honest_note": (
            "Gross Sharpe and vol come from the real walk-forward on the "
            "sparse 7-stream cube (EXP-2510). This experiment applies the "
            "drag transformation via the canonical net_sharpe_from_drag "
            "helper (EXP-2420) — identical to what EXP-2510 and EXP-2470 "
            "used so the numbers are directly comparable. Vol and Max DD "
            "are held constant because execution cost is deterministic "
            "drag that affects the annualised mean, not its variance."
        ),
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    _write_html(payload)
    print("[3/3] wrote", REPORT_JSON)
    print("          ", REPORT_HTML)
    return payload


def _write_html(p: Dict) -> None:
    g = p["gross"]
    rows = ""
    for sc in p["scenarios"]:
        a = sc["variants"]["ledoit_only"]
        b = sc["variants"]["combined"]
        is_target = " style='background:#e8f5e9'" if sc["id"] == "commfree_plus_exec_opt" else ""
        rows += (
            f"<tr{is_target}><td>{sc['label']}</td><td>{sc['broker']}</td>"
            f"<td>{sc['drag_bps']:.0f}</td>"
            f"<td>{a['net_sharpe']:.2f}</td><td>{a['net_cagr_pct']:.2f}%</td>"
            f"<td>{b['net_sharpe']:.2f}</td><td>{b['net_cagr_pct']:.2f}%</td></tr>"
        )
    hl = p["headline"]
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>EXP-2570 — Commfree Net Sharpe</title>
<style>
 body{{font-family:-apple-system,Segoe UI,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#1a1a1a;line-height:1.5;background:#fff}}
 h1{{border-bottom:2px solid #222;padding-bottom:.3em}} h2{{margin-top:1.6em;border-bottom:1px solid #ccc}}
 table{{border-collapse:collapse;width:100%;margin:1em 0;font-size:.93em}}
 th,td{{border:1px solid #bbb;padding:5px 8px;text-align:left}} th{{background:#f0f0f0}}
 .ok{{color:#0a7a0a;font-weight:600}} .small{{color:#555;font-size:.88em}}
 .callout{{background:#e8f5e9;border-left:4px solid #0a7a0a;padding:.9em 1.1em;margin:1em 0}}
</style></head><body>
<h1>EXP-2570 — Commission-Free Broker Net Sharpe</h1>
<p class='small'>Generated {p['generated']} · 7-stream portfolio · Sparse
  walk-forward cube (EXP-2510 source) · Rule Zero clean.</p>

<div class='callout'>
<b>TARGET (row highlighted):</b> commission-free broker + EXP-2470 exec optimisation stack
(A limit-at-mid + B patient execution + C route reallocation + D multi-leg combos)
gives total drag <b>{hl['target_drag_bps']:.0f} bps</b> (vs IBKR baseline
{hl['baseline_drag_bps']:.0f} bps, savings <b>{hl['target_vs_baseline_savings_bps']:.0f} bps</b>).
</div>

<h2>Net Sharpe by scenario</h2>
<table>
<tr><th>Scenario</th><th>Broker</th><th>Drag (bps)</th>
 <th colspan='2'>LW-only</th><th colspan='2'>LW + regime blend</th></tr>
<tr><th></th><th></th><th></th>
 <th>Net Sharpe</th><th>Net CAGR</th><th>Net Sharpe</th><th>Net CAGR</th></tr>
{rows}
</table>

<h2>Headline transformation</h2>
<table>
<tr><th>Variant</th><th>Gross Sharpe</th><th>IBKR net Sharpe</th>
 <th>Target net Sharpe</th><th>ΔSharpe (baseline → target)</th></tr>
<tr><td>LW-only</td>
  <td>{hl['ledoit_only']['gross_sharpe']:.2f}</td>
  <td>{hl['ledoit_only']['baseline_net']:.2f}</td>
  <td class='ok'>{hl['ledoit_only']['target_net']:.2f}</td>
  <td class='ok'>{hl['ledoit_only']['delta_sharpe_baseline_to_target']:+.2f}</td></tr>
<tr><td>Combined</td>
  <td>{hl['combined']['gross_sharpe']:.2f}</td>
  <td>{hl['combined']['baseline_net']:.2f}</td>
  <td class='ok'>{hl['combined']['target_net']:.2f}</td>
  <td class='ok'>{hl['combined']['delta_sharpe_baseline_to_target']:+.2f}</td></tr>
</table>

<h2>Cost decomposition (EXP-2510 real-breakdown basis)</h2>
<ul>
<li>Bid-ask:   <b>{p['cost_components_pct']['bid_ask']:.2f}%</b> / yr</li>
<li>Commission: <b>{p['cost_components_pct']['commission']:.2f}%</b> / yr (37% of total — killed by broker move)</li>
<li>Slippage:  <b>{p['cost_components_pct']['slippage']:.2f}%</b> / yr</li>
<li>Full drag: <b>{p['cost_components_pct']['full_drag']:.2f}%</b> / yr</li>
<li>Exec-opt stack savings (EXP-2470 A+B+C+D): <b>−{p['cost_components_pct']['exec_opt_savings']:.2f}%</b> / yr</li>
</ul>

<h2>Honest note</h2>
<p>{p['honest_note']}</p>
</body></html>"""
    REPORT_HTML.write_text(html)


if __name__ == "__main__":
    main()
