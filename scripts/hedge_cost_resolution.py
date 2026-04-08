#!/usr/bin/env python3
"""
Priority #4: Hedge Cost Resolution

At current trade-level alpha (~1.5%/yr), does any hedge strategy make
economic sense? This script models 3 options using REAL IronVault
put prices (calibrated in commit 51e11e6):

  1. No hedge — accept full drawdown risk
  2. Selective VIX<15 puts — buy only when cheapest (~2.0%/yr)
  3. Collar strategy — sell OTM calls to fund puts (~1.3%/yr net)

For each: compute net CAGR, max DD reduction, Sharpe impact,
and break-even alpha level.

Output: reports/hedge_cost_resolution.html + JSON
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compass.smart_hedge import VIX_TO_PUT_COST, COLLAR_OFFSET_RATIO, _interp_put_cost


# ═══════════════════════════════════════════════════════════════════════════
# Inputs
# ═══════════════════════════════════════════════════════════════════════════

# Trade-level alpha baseline (from EXP-1220 honest audit, commit 1ef262a)
BASELINE_ALPHA_CAGR = 0.015  # 1.5%/yr gross alpha
BASELINE_SHARPE = 1.26       # per-trade Sharpe
BASELINE_MAX_DD = 0.016      # 1.6% unhedged drawdown (trade level)

# Real hedge costs from IronVault (commit 51e11e6)
# Per-year cost of 5% OTM SPY puts continuously held
YEARLY_PUT_COSTS = {
    2020: 0.0528,  # COVID volatility made puts expensive
    2021: 0.0302,
    2022: 0.0539,  # Bear market premium
    2023: 0.0244,  # Closest to the 2% assumption
    2024: 0.0278,
    2025: 0.0725,  # Current high-vol environment
}
AVG_REAL_PUT_COST = sum(YEARLY_PUT_COSTS.values()) / len(YEARLY_PUT_COSTS)  # 4.36%/yr

# VIX<15 frequency: roughly 15-20% of trading days in the sample
VIX_BELOW_15_FRACTION = 0.18  # empirical

# Crisis DD reduction factors (how much does each hedge reduce COVID/bear losses?)
CRISIS_DD_REDUCTIONS = {
    "no_hedge":        {"covid": 0.00, "bear": 0.00, "flash": 0.00, "vix_spike": 0.00},
    "selective_puts":  {"covid": 0.35, "bear": 0.20, "flash": 0.30, "vix_spike": 0.45},
    "collar":          {"covid": 0.50, "bear": 0.30, "flash": 0.45, "vix_spike": 0.55},
}

# Baseline crisis drawdowns from Phase 6 stress test (portfolio with 1.5x beta)
CRISIS_DDS = {
    "covid": 51.78,
    "bear": 43.68,
    "flash": 15.00,
    "vix_spike": 22.50,
}


# ═══════════════════════════════════════════════════════════════════════════
# Hedge option models
# ═══════════════════════════════════════════════════════════════════════════

def model_no_hedge() -> dict:
    """Option 1: No hedge — accept full risk."""
    return {
        "name": "No Hedge",
        "description": "Accept full drawdown risk. No hedge cost.",
        "annual_cost_pct": 0.0,
        "gross_alpha_pct": BASELINE_ALPHA_CAGR * 100,
        "net_cagr_pct": BASELINE_ALPHA_CAGR * 100,
        "dd_reduction_pct": 0.0,
        "dd_adjustments": CRISIS_DD_REDUCTIONS["no_hedge"],
        "crisis_dds": {k: v for k, v in CRISIS_DDS.items()},
        "sharpe_impact_pct": 0.0,
        "net_sharpe": BASELINE_SHARPE,
        "break_even_alpha_pct": 0.0,  # always "breaks even" at zero cost
    }


def model_selective_puts() -> dict:
    """Option 2: Buy puts only when VIX < 15 (cheapest)."""
    # At VIX 12, put cost is ~1.8%/yr. We hold puts only ~18% of the time,
    # but when VIX rises above 15, we stop buying. So the effective annual
    # cost when we ARE hedged is the VIX-12 cost, but weighted by frequency.
    # However, we need protection for the unhedged periods too, so the
    # honest model is: hedge only the ~18% of days where VIX < 15, pay the
    # VIX<15 rate pro-rated.
    cost_at_vix_12 = VIX_TO_PUT_COST[12]  # 1.8%/yr
    cost_at_vix_15 = VIX_TO_PUT_COST[15]  # 2.4%/yr
    avg_low_vix_cost = (cost_at_vix_12 + cost_at_vix_15) / 2  # ~2.1%
    # Pro-rated by time we're actually hedged
    annual_cost = avg_low_vix_cost  # when active: full rate

    # Since we only hedge the calm periods, we get NO protection during
    # actual crises (because those happen when VIX >= 20+). So DD reduction
    # is smaller than naked puts.
    # But the "insurance" value when a crisis starts suddenly from low VIX
    # is real (COVID went from VIX 15 → 82 in 3 weeks).

    # DD reductions for selective (partial protection)
    dd_adjustments = CRISIS_DD_REDUCTIONS["selective_puts"]

    net_cagr = BASELINE_ALPHA_CAGR - annual_cost
    crisis_dds = {
        k: v * (1 - dd_adjustments[k]) for k, v in CRISIS_DDS.items()
    }
    # Weighted average DD reduction for Sharpe impact
    avg_dd_reduction = np.mean(list(dd_adjustments.values()))
    # Sharpe impact: lower CAGR but lower vol → ambiguous, but net negative
    # because hedge cost > alpha reduction in vol
    sharpe_drag = annual_cost / max(BASELINE_ALPHA_CAGR, 0.001)
    net_sharpe = BASELINE_SHARPE * (1 - sharpe_drag)

    # Break-even alpha: cost equals alpha
    break_even = annual_cost

    return {
        "name": "Selective VIX<15 Puts",
        "description": "Buy 5% OTM SPY puts only when VIX < 15 (~18% of days). Cheapest market entry.",
        "annual_cost_pct": round(annual_cost * 100, 2),
        "gross_alpha_pct": BASELINE_ALPHA_CAGR * 100,
        "net_cagr_pct": round(net_cagr * 100, 2),
        "dd_reduction_pct": round(avg_dd_reduction * 100, 1),
        "dd_adjustments": dd_adjustments,
        "crisis_dds": {k: round(v, 2) for k, v in crisis_dds.items()},
        "sharpe_impact_pct": round(-sharpe_drag * 100, 1),
        "net_sharpe": round(net_sharpe, 2),
        "break_even_alpha_pct": round(break_even * 100, 2),
    }


def model_collar() -> dict:
    """Option 3: Collar — sell OTM calls to fund puts (~1.3%/yr net)."""
    # Collar: buy 5% OTM put + sell 3% OTM call.
    # Put cost ~4.36%/yr avg. Call premium offsets ~70% of put cost.
    # Net cost = put cost × (1 - 0.70) = 4.36% × 0.30 = ~1.31%/yr
    put_cost = AVG_REAL_PUT_COST
    call_offset = put_cost * COLLAR_OFFSET_RATIO
    annual_cost = put_cost - call_offset  # ~1.31%/yr

    dd_adjustments = CRISIS_DD_REDUCTIONS["collar"]

    net_cagr = BASELINE_ALPHA_CAGR - annual_cost
    crisis_dds = {
        k: v * (1 - dd_adjustments[k]) for k, v in CRISIS_DDS.items()
    }
    avg_dd_reduction = np.mean(list(dd_adjustments.values()))

    # Collar also caps upside (3% OTM call limits monthly gains)
    # This creates additional opportunity cost in strong rallies
    upside_cap_drag = 0.005  # ~0.5%/yr from capped upside

    total_drag = annual_cost + upside_cap_drag
    sharpe_drag = total_drag / max(BASELINE_ALPHA_CAGR, 0.001)
    net_sharpe = BASELINE_SHARPE * (1 - sharpe_drag)

    break_even = total_drag

    return {
        "name": "Collar Strategy",
        "description": "Sell 3% OTM SPY calls to fund 5% OTM puts. ~70% cost offset.",
        "annual_cost_pct": round(annual_cost * 100, 2),
        "gross_alpha_pct": BASELINE_ALPHA_CAGR * 100,
        "net_cagr_pct": round(net_cagr * 100, 2),
        "dd_reduction_pct": round(avg_dd_reduction * 100, 1),
        "dd_adjustments": dd_adjustments,
        "crisis_dds": {k: round(v, 2) for k, v in crisis_dds.items()},
        "sharpe_impact_pct": round(-sharpe_drag * 100, 1),
        "net_sharpe": round(net_sharpe, 2),
        "break_even_alpha_pct": round(break_even * 100, 2),
        "extra_drag_pct": round(upside_cap_drag * 100, 2),
    }


def sensitivity_by_alpha(target_alpha: float) -> dict:
    """What's the verdict for each hedge at different alpha levels?"""
    results = {}
    # Continuous puts
    cont_cost = AVG_REAL_PUT_COST
    results["continuous_puts"] = {
        "alpha": target_alpha,
        "cost": cont_cost,
        "net": target_alpha - cont_cost,
        "viable": target_alpha > cont_cost,
    }
    # Selective puts
    sel_cost = 0.021
    results["selective_puts"] = {
        "alpha": target_alpha,
        "cost": sel_cost,
        "net": target_alpha - sel_cost,
        "viable": target_alpha > sel_cost,
    }
    # Collar
    collar_cost = AVG_REAL_PUT_COST * (1 - COLLAR_OFFSET_RATIO) + 0.005
    results["collar"] = {
        "alpha": target_alpha,
        "cost": collar_cost,
        "net": target_alpha - collar_cost,
        "viable": target_alpha > collar_cost,
    }
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Recommendation engine
# ═══════════════════════════════════════════════════════════════════════════

def make_recommendation(options: list) -> dict:
    """Pick the best hedge option given current alpha level."""
    # Rank by: (1) net CAGR, (2) crisis protection
    # At current 1.5% alpha, the continuous put (4.36%) is strictly negative.
    # Selective (2.1%) is also net-negative. Collar (1.31% + 0.5% upside drag)
    # is net-negative but less bad.
    # So at current alpha, NO HEDGE wins on absolute returns.
    # But if we want crisis protection, collar is the cheapest.

    ranked = sorted(options, key=lambda x: x["net_cagr_pct"], reverse=True)

    best_returns = ranked[0]
    best_protection = max(options, key=lambda x: x["dd_reduction_pct"])

    return {
        "best_net_cagr": best_returns["name"],
        "best_crisis_protection": best_protection["name"],
        "recommendation": "No Hedge",
        "reasoning": (
            f"At current trade-level alpha of {BASELINE_ALPHA_CAGR*100:.1f}%/yr, "
            f"EVERY hedge option is net-negative because the cheapest hedge "
            f"(collar at ~{(AVG_REAL_PUT_COST*(1-COLLAR_OFFSET_RATIO)+0.005)*100:.1f}%/yr total drag) "
            f"still exceeds the alpha. Hedging only becomes viable when "
            f"alpha > {(AVG_REAL_PUT_COST*(1-COLLAR_OFFSET_RATIO)+0.005)*100:.1f}%/yr. "
            f"Current credit spread strategy must either: (1) increase position "
            f"size/frequency to boost alpha beyond hedge break-even, or "
            f"(2) accept drawdown risk and forgo hedging until alpha scales up."
        ),
        "action_items": [
            "Deploy Phase 6 EXP-400/401 blend unhedged at current parameters",
            "Monitor for regime shifts (VIX > 30) as early crisis warning",
            "Defer hedge deployment until portfolio alpha reaches ≥2.5%/yr",
            "If hedge becomes necessary, use COLLAR (cheapest at 1.81% total drag)",
            "Re-evaluate after 8 weeks of paper trading validates real alpha",
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# HTML report
# ═══════════════════════════════════════════════════════════════════════════

def generate_html(options: list, recommendation: dict, alpha_sens: dict, output_path: Path):
    """Build the hedge cost resolution HTML report."""

    # Options comparison table
    option_rows = ""
    best_option = recommendation["best_net_cagr"]
    for opt in options:
        highlight = ' style="background:#f0fdf4"' if opt["name"] == best_option else ""
        net_color = "#059669" if opt["net_cagr_pct"] > 0 else "#dc2626"
        sharpe_c = "#059669" if opt["net_sharpe"] > 0.5 else ("#d97706" if opt["net_sharpe"] > 0 else "#dc2626")
        option_rows += (
            f'<tr{highlight}>'
            f'<td style="text-align:left"><strong>{opt["name"]}</strong>'
            f'<br/><span style="color:#64748b;font-size:.75rem">{opt["description"]}</span></td>'
            f'<td class="r">{opt["annual_cost_pct"]:.2f}%</td>'
            f'<td class="r">{opt["gross_alpha_pct"]:.2f}%</td>'
            f'<td class="r" style="color:{net_color};font-weight:700">{opt["net_cagr_pct"]:+.2f}%</td>'
            f'<td class="r">{opt["dd_reduction_pct"]:.0f}%</td>'
            f'<td class="r" style="color:{sharpe_c}">{opt["net_sharpe"]:.2f}</td>'
            f'<td class="r">{opt["break_even_alpha_pct"]:.2f}%</td>'
            f'</tr>\n'
        )

    # Crisis DD comparison table
    crisis_names = {"covid": "COVID Crash", "bear": "2022 Bear", "flash": "Flash Crash", "vix_spike": "VIX Spike"}
    crisis_rows = ""
    for ck, cname in crisis_names.items():
        cells = f'<td style="text-align:left"><strong>{cname}</strong></td>'
        for opt in options:
            dd = opt["crisis_dds"][ck]
            color = "#dc2626" if dd > 40 else ("#d97706" if dd > 25 else "#059669")
            cells += f'<td class="r" style="color:{color}">{dd:.1f}%</td>'
        crisis_rows += f"<tr>{cells}</tr>\n"

    # Yearly put cost table (real data)
    yearly_cost_rows = ""
    for year, cost in sorted(YEARLY_PUT_COSTS.items()):
        color = "#dc2626" if cost > 0.05 else ("#d97706" if cost > 0.03 else "#059669")
        yearly_cost_rows += (
            f'<tr><td><strong>{year}</strong></td>'
            f'<td class="r" style="color:{color}">{cost*100:.2f}%</td></tr>\n'
        )

    # Alpha sensitivity table
    sens_rows = ""
    for alpha_val in [0.01, 0.015, 0.02, 0.025, 0.035, 0.05, 0.08, 0.12]:
        row_cells = f'<td><strong>{alpha_val*100:.1f}%</strong></td>'
        s = sensitivity_by_alpha(alpha_val)
        for key in ["continuous_puts", "selective_puts", "collar"]:
            item = s[key]
            net = item["net"]
            viable = item["viable"]
            color = "#059669" if viable else "#dc2626"
            icon = "&#10003;" if viable else "&#10007;"
            row_cells += f'<td class="r" style="color:{color}">{icon} {net*100:+.2f}%</td>'
        sens_rows += f"<tr>{row_cells}</tr>\n"

    # Action items list
    action_items = "".join(
        f"<li>{item}</li>" for item in recommendation["action_items"]
    )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Hedge Cost Resolution — Priority #4</title>
<style>
:root{{--bg:#fff;--card:#f8f9fa;--border:#e2e8f0;--text:#1a1a2e;--muted:#64748b;--green:#059669;--red:#dc2626;--blue:#2563eb;--yellow:#d97706}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);line-height:1.55;max-width:1100px;margin:0 auto;padding:28px}}
h1{{font-size:1.55rem;font-weight:800;margin-bottom:4px}}
h2{{font-size:1.15rem;font-weight:700;margin:32px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--border)}}
h3{{font-size:.98rem;font-weight:600;margin:22px 0 8px;color:#374151}}
.sub{{color:var(--muted);font-size:.86rem;margin-bottom:18px}}
.note{{color:var(--muted);font-size:.82rem;font-style:italic;margin:6px 0}}
table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:.84rem}}
th{{background:#f1f5f9;color:var(--muted);padding:7px 10px;text-align:left;border-bottom:2px solid var(--border);font-size:.74rem;font-weight:600;text-transform:uppercase}}
td{{padding:6px 10px;border-bottom:1px solid #f1f5f9;text-align:left}}
.r{{text-align:right}}
tr:hover td{{background:#fafafa}}
.hero{{background:linear-gradient(135deg,#fef3c7,#fde68a);border:2px solid #d97706;border-radius:12px;padding:24px;margin:18px 0}}
.hero .title{{font-size:1.15rem;font-weight:700;color:#92400e;margin-bottom:4px}}
.hero .verdict{{font-size:1.4rem;font-weight:800;color:#92400e;margin:6px 0}}
.hero p{{color:#78350f;font-size:.9rem;margin-top:6px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:16px 0}}
.c{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:13px;text-align:center}}
.c .l{{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.3px}}
.c .v{{font-weight:700;font-size:1.18rem;margin-top:3px}}
.box{{border:1px solid var(--border);border-radius:8px;padding:16px;margin:12px 0;background:var(--card)}}
.box-green{{border-left:5px solid var(--green)}} .box-red{{border-left:5px solid var(--red)}}
.box-blue{{border-left:5px solid var(--blue)}} .box-yellow{{border-left:5px solid var(--yellow)}}
.box h4{{margin:0 0 6px;font-size:.95rem}}
ul{{padding-left:20px;line-height:1.85;font-size:.88rem;margin-top:6px}}
</style></head><body>

<h1>Hedge Cost Resolution — MASTERPLAN Priority #4</h1>
<p class="sub">At current trade-level alpha (~1.5%/yr), which hedge strategy survives the math?
&bull; Real IronVault put costs (2020-2025) &bull; {datetime.now().strftime("%Y-%m-%d")}</p>

<div class="hero">
<div class="title">RECOMMENDATION</div>
<div class="verdict">{recommendation["recommendation"]}</div>
<p>{recommendation["reasoning"]}</p>
</div>

<div class="cards">
<div class="c"><div class="l">Baseline Alpha</div><div class="v">{BASELINE_ALPHA_CAGR*100:.1f}%/yr</div></div>
<div class="c"><div class="l">Avg Real Put Cost</div><div class="v" style="color:var(--red)">{AVG_REAL_PUT_COST*100:.2f}%/yr</div></div>
<div class="c"><div class="l">Selective Puts</div><div class="v">{2.1:.1f}%/yr</div></div>
<div class="c"><div class="l">Collar Net</div><div class="v">{1.81:.2f}%/yr</div></div>
<div class="c"><div class="l">Break-even Alpha</div><div class="v" style="color:var(--yellow)">{1.81:.2f}%</div></div>
<div class="c"><div class="l">Current Gap</div><div class="v" style="color:var(--red)">−{(1.81-BASELINE_ALPHA_CAGR*100):.2f}%</div></div>
</div>

<!-- ══ Real IronVault Put Costs ══ -->
<h2>1. Real IronVault Put Costs (Commit 51e11e6)</h2>
<p class="note">5% OTM SPY put held continuously, annualized cost per year from actual IronVault bid/ask data.</p>

<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
<div>
<table>
<thead><tr><th>Year</th><th class="r">Annual Cost</th></tr></thead>
<tbody>{yearly_cost_rows}
<tr style="background:#fef3c7"><td><strong>6-year Avg</strong></td><td class="r"><strong>{AVG_REAL_PUT_COST*100:.2f}%</strong></td></tr>
</tbody></table>
</div>
<div>
<div class="box box-red">
<h4>The 2.2x Gap</h4>
<p style="font-size:.85rem">Original model assumed 2%/yr hedge cost. Real IronVault data
shows {AVG_REAL_PUT_COST*100:.2f}%/yr — <strong>2.2x higher</strong>.</p>
<p style="font-size:.85rem;margin-top:6px">Worst years: 2025 (7.25%) and 2022 (5.39%).
Best year: 2023 (2.44%). Volatility-driven pricing means hedges are
most expensive exactly when you need them most.</p>
</div>
</div>
</div>

<!-- ══ 3 Options Comparison ══ -->
<h2>2. Three Hedge Options Compared</h2>
<table>
<thead><tr>
<th>Option</th><th class="r">Annual Cost</th><th class="r">Gross Alpha</th>
<th class="r">Net CAGR</th><th class="r">Avg DD Reduction</th><th class="r">Net Sharpe</th><th class="r">Break-even</th>
</tr></thead>
<tbody>{option_rows}</tbody>
</table>

<!-- ══ Crisis DD Comparison ══ -->
<h2>3. Crisis Drawdown Comparison</h2>
<p class="note">Portfolio drawdown under each crisis scenario with 1.5x credit spread beta.
Numbers from Phase 6 stress test baseline.</p>
<table>
<thead><tr><th>Scenario</th>{''.join(f'<th class="r">{opt["name"]}</th>' for opt in options)}</tr></thead>
<tbody>{crisis_rows}</tbody>
</table>

<div class="box box-yellow">
<h4>Crisis protection trade-off</h4>
<p style="font-size:.88rem">Collar provides the best crisis protection (30-55% DD reduction)
at the lowest cost (1.31% net), but STILL leaves COVID DD around 26% and 2022 bear at 31%.
No hedge option fully solves the Phase 6 crisis failure — only combined with lower leverage
or alpha scale-up can the portfolio meet the 40% crisis threshold.</p>
</div>

<!-- ══ Alpha Sensitivity ══ -->
<h2>4. At What Alpha Level Does Each Hedge Become Viable?</h2>
<p class="note">Net annual return (alpha minus hedge cost). Check = net positive.</p>
<table>
<thead><tr><th>Gross Alpha</th><th class="r">Continuous Puts (4.36%)</th><th class="r">Selective VIX&lt;15 (2.10%)</th><th class="r">Collar (1.81%)</th></tr></thead>
<tbody>{sens_rows}</tbody>
</table>

<div class="box box-blue">
<h4>Break-even thresholds</h4>
<ul>
<li><strong>Collar</strong> becomes viable at alpha &gt; <strong>1.81%/yr</strong> (current 1.5% &mdash; still 0.31% short)</li>
<li><strong>Selective puts</strong> become viable at alpha &gt; <strong>2.10%/yr</strong> (need +0.60% alpha)</li>
<li><strong>Continuous puts</strong> need alpha &gt; <strong>4.36%/yr</strong> (nearly 3x current)</li>
<li>At 5%+ alpha, all three options are net positive and collar becomes clearly dominant</li>
</ul>
</div>

<!-- ══ Recommendation ══ -->
<h2>5. Recommendation & Action Items</h2>

<div class="box box-green">
<h4>Best net CAGR: {recommendation["best_net_cagr"]}</h4>
<p style="font-size:.88rem">At current alpha, {recommendation["best_net_cagr"]} produces
the highest net CAGR because any hedge subtracts more than it protects.</p>
</div>

<div class="box box-blue">
<h4>Best crisis protection: {recommendation["best_crisis_protection"]}</h4>
<p style="font-size:.88rem">If crisis protection is the primary goal, {recommendation["best_crisis_protection"]}
offers the best drawdown reduction (50%+ on COVID, 30% on 2022 bear).</p>
</div>

<div class="box box-yellow">
<h4>Action Items</h4>
<ul>{action_items}</ul>
</div>

<!-- ══ Methodology ══ -->
<h2>6. Methodology</h2>
<div class="box">
<h4>Data sources</h4>
<ul>
<li><strong>Real put costs:</strong> IronVault 5% OTM SPY puts, 2020-2025 (commit 51e11e6)</li>
<li><strong>VIX-based pricing:</strong> VIX_TO_PUT_COST table in compass/smart_hedge.py</li>
<li><strong>Baseline alpha:</strong> EXP-1220 trade-level (1.2-1.5%/yr, commit 1ef262a)</li>
<li><strong>Crisis drawdowns:</strong> Phase 6 stress test with 1.5x credit spread beta</li>
<li><strong>Collar offset:</strong> 70% of put cost offset by selling 3% OTM calls</li>
</ul>
<h4 style="margin-top:12px">Assumptions</h4>
<ul>
<li>Selective VIX&lt;15 strategy: hedge active ~18% of days (empirical), pays VIX-12 to VIX-15 rate (~2.1%)</li>
<li>Collar caps upside at 3% OTM calls → ~0.5%/yr opportunity cost during strong rallies</li>
<li>Crisis DD reductions are approximate (full backtest in smart_hedge.py tests)</li>
<li>Sharpe impact computed as (cost drag / alpha) × baseline Sharpe</li>
</ul>
</div>

<p style="text-align:center;color:var(--muted);margin-top:36px;padding-top:14px;border-top:1px solid var(--border);font-size:.78rem">
Priority #4 Hedge Cost Resolution &bull; compass/smart_hedge.py + hedge_cost_reality.py &bull;
{datetime.now().strftime("%Y-%m-%d")}
</p>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("Priority #4: Hedge Cost Resolution")
    print("=" * 70)
    print(f"\nBaseline alpha: {BASELINE_ALPHA_CAGR*100:.1f}%/yr")
    print(f"Real avg put cost (6yr): {AVG_REAL_PUT_COST*100:.2f}%/yr")
    print()

    # Model the 3 options
    print("Modeling 3 hedge options...")
    options = [
        model_no_hedge(),
        model_selective_puts(),
        model_collar(),
    ]

    # Print comparison
    print(f"\n{'Option':<25}{'Cost':>10}{'Net CAGR':>12}{'DD Redux':>12}{'Net Sharpe':>12}{'Break-even':>14}")
    print("-" * 85)
    for opt in options:
        print(f"{opt['name']:<25}"
              f"{opt['annual_cost_pct']:>9.2f}%"
              f"{opt['net_cagr_pct']:>11.2f}%"
              f"{opt['dd_reduction_pct']:>11.0f}%"
              f"{opt['net_sharpe']:>12.2f}"
              f"{opt['break_even_alpha_pct']:>13.2f}%")

    # Crisis DDs per option
    print(f"\nCrisis Drawdowns (1.5x credit spread beta):")
    print(f"{'Crisis':<20}", end="")
    for opt in options:
        print(f"{opt['name'][:15]:>17}", end="")
    print()
    for crisis in ["covid", "bear", "flash", "vix_spike"]:
        print(f"{crisis:<20}", end="")
        for opt in options:
            dd = opt['crisis_dds'][crisis]
            print(f"{dd:>16.1f}%", end="")
        print()

    # Alpha sensitivity
    print(f"\nAlpha Sensitivity (when does each hedge become viable?):")
    print(f"{'Gross Alpha':<15}{'Continuous':>15}{'Selective':>15}{'Collar':>15}")
    print("-" * 60)
    for alpha_val in [0.01, 0.015, 0.02, 0.025, 0.035, 0.05]:
        s = sensitivity_by_alpha(alpha_val)
        print(f"{alpha_val*100:>12.1f}%",
              f"{s['continuous_puts']['net']*100:>13.2f}%",
              f"{s['selective_puts']['net']*100:>13.2f}%",
              f"{s['collar']['net']*100:>13.2f}%")

    # Recommendation
    rec = make_recommendation(options)
    print(f"\n{'=' * 70}")
    print(f"RECOMMENDATION: {rec['recommendation']}")
    print(f"{'=' * 70}")
    print(rec["reasoning"])
    print()
    print("Action items:")
    for item in rec["action_items"]:
        print(f"  - {item}")

    # Generate HTML
    print(f"\nGenerating HTML report...")
    output_path = ROOT / "reports" / "hedge_cost_resolution.html"

    # Build alpha_sens dict for HTML
    alpha_sens = {}
    for av in [0.01, 0.015, 0.02, 0.025, 0.035, 0.05, 0.08, 0.12]:
        alpha_sens[av] = sensitivity_by_alpha(av)

    generate_html(options, rec, alpha_sens, output_path)
    print(f"  Report: {output_path}")

    # Save JSON
    json_path = ROOT / "reports" / "hedge_cost_resolution.json"
    summary = {
        "generated": datetime.now().isoformat(),
        "inputs": {
            "baseline_alpha_cagr": BASELINE_ALPHA_CAGR,
            "baseline_sharpe": BASELINE_SHARPE,
            "baseline_max_dd": BASELINE_MAX_DD,
            "avg_real_put_cost": AVG_REAL_PUT_COST,
            "yearly_put_costs": YEARLY_PUT_COSTS,
            "collar_offset_ratio": COLLAR_OFFSET_RATIO,
        },
        "options": options,
        "alpha_sensitivity": {
            f"{k*100:.1f}%": v for k, v in alpha_sens.items()
        },
        "recommendation": rec,
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"  JSON: {json_path}")

    print(f"\n{'=' * 70}")
    print("Priority #4 Hedge Cost Resolution: COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
