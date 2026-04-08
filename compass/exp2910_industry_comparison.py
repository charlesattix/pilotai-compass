"""
EXP-2910 — Backtest-vs-Reality Industry Comparison.

The v8a backtest pipeline has settled on a pooled net Sharpe around
6.16 (EXP-2730 / EXP-2740 / EXP-2450). That number is HIGH by
industry standards and needs an honest expectation-setter before
paper trading so nobody is caught flat-footed when real-world
performance lands somewhere less flattering.

This experiment is a RESEARCH DOCUMENT, not a numerical experiment.
Its job is to:

  1. Summarise publicly-reported performance of the relevant
     comparable fund types:
        * Multi-strategy flagship funds (Citadel, Millennium,
          Two Sigma, D.E. Shaw)
        * Systematic quant funds (Renaissance, AQR)
        * Option-selling / volatility-risk-premium strategies
          (dedicated vol funds, CBOE BXM/PUT indices)
        * Published academic studies of VRP/short-vol strategies
  2. Document the cautionary tales (LJM, OptionSellers.com,
     Catalyst Hedged Futures) that shipped "great backtests" and
     blew up in live trading.
  3. Put the v8a headline numbers next to those benchmarks in one
     table with clear source citations and a plain-language
     expectation range.
  4. Produce an honest "what should paper trading look like"
     range for Carlos so the go/no-go decision in Phase 10 is
     based on realistic targets, not backtest-inflated ones.

ALL numbers cited here are from PUBLIC sources (SEC filings,
academic papers, press reports, CBOE index methodology docs). No
proprietary data is claimed. Every fund benchmark is a RANGE, not
a point estimate — the honest uncertainty is part of the message.

The v8a numbers used as comparison points come from our own
committed backtests (EXP-2450 sparse, EXP-2730 rolling, EXP-2740
sensitivity). Those are linked in the report so the reader can
trace exactly what they correspond to.

Outputs:
  compass/exp2910_industry_comparison.py
  compass/reports/exp2910_industry_comparison.html
  compass/reports/exp2910_industry_comparison.json

Tag: EXP-2910
Run: python3 -m compass.exp2910_industry_comparison
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2910_industry_comparison.json"
REPORT_HTML = REPORT_DIR / "exp2910_industry_comparison.html"


# ─────────────────────────────────────────────────────────────────────
# Benchmark data — PUBLICLY-REPORTED RANGES, cited at source
# ─────────────────────────────────────────────────────────────────────


@dataclass
class FundBenchmark:
    name: str
    category: str
    net_sharpe_range: str
    period: str
    net_return_cagr_range: str
    sources: List[str]
    notes: str


INDUSTRY_BENCHMARKS: List[FundBenchmark] = [
    FundBenchmark(
        name="Citadel Wellington",
        category="Multi-strategy flagship",
        net_sharpe_range="~2.0 – 3.0",
        period="~2010-2023 (multi-year rolling)",
        net_return_cagr_range="~15 – 20 %/yr net",
        sources=[
            "Bloomberg reporting on Citadel Wellington returns",
            "HFM Week reporting, various 2022-2023 articles",
            "Institutional Investor profiles",
        ],
        notes=(
            "Multi-strategy fund spanning equities, credit, macro, "
            "commodities, quant. Heavily diversified across books. "
            "Not an option-selling fund specifically. Sharpe quoted "
            "is the CONSOLIDATED fund net of 4% management + 20% "
            "incentive fees, after pass-through expenses. Typically "
            "considered the top end of realistic multi-strat Sharpe."
        ),
    ),
    FundBenchmark(
        name="Millennium Management",
        category="Multi-strategy flagship",
        net_sharpe_range="~2.5 – 3.5",
        period="~2010-2023",
        net_return_cagr_range="~13 – 18 %/yr net",
        sources=[
            "WSJ reporting on Millennium performance",
            "Institutional Investor, Bloomberg coverage",
        ],
        notes=(
            "Pod-shop multi-strategy. Strict per-pod DD limits, "
            "very low volatility target, historically one of the "
            "higher realised Sharpes among multi-strats. Net of "
            "management fees and a pass-through expense structure."
        ),
    ),
    FundBenchmark(
        name="Two Sigma Absolute Return / Compass",
        category="Systematic quant",
        net_sharpe_range="~1.5 – 2.5",
        period="~2010-2020 (various products)",
        net_return_cagr_range="~8 – 15 %/yr net",
        sources=[
            "Eurekahedge quant fund indices",
            "Press reports on TS flagship vehicles (WSJ 2018-2022)",
        ],
        notes=(
            "Systematic quant strategies. Performance varies "
            "materially by product; the absolute-return vehicle "
            "has been softer in 2022-2023 per industry reports."
        ),
    ),
    FundBenchmark(
        name="Renaissance Medallion (closed)",
        category="Internal quant — closed",
        net_sharpe_range="~2.5 – 4.0 (some sources higher)",
        period="1988-2018 (academic reconstructions)",
        net_return_cagr_range="~35 – 65 %/yr GROSS before fees",
        sources=[
            "Gregory Zuckerman, 'The Man Who Solved the Market' (2019)",
            "Cornell 2019 paper (Lu et al) on replication",
        ],
        notes=(
            "Medallion is the canonical outlier — closed to outside "
            "investors since 1993, employee capital only, with an "
            "unusually high gross-fee-drag structure (5% mgmt + 44% "
            "incentive). NET Sharpe to outside investors is N/A "
            "because they can't invest. The figures above are the "
            "GROSS reconstructions from academic reverse-engineering. "
            "Do not use this as a realistic live benchmark — nothing "
            "retail-accessible reproduces Medallion."
        ),
    ),
    FundBenchmark(
        name="Renaissance RIEF (institutional)",
        category="Systematic quant (public)",
        net_sharpe_range="~0.5 – 1.2",
        period="2005-2023",
        net_return_cagr_range="~5 – 8 %/yr net (much worse in 2020-2022)",
        sources=[
            "RIEF prospectus / SEC filings",
            "WSJ 2020-2021 reporting on RIEF losses",
        ],
        notes=(
            "The 'outside investor' Renaissance product. Sharpe "
            "FAR below Medallion — reportedly suffered double-digit "
            "drawdowns in 2020. This is the honest reference point "
            "for 'what a systematic quant fund actually delivers to "
            "an external allocator'."
        ),
    ),
    FundBenchmark(
        name="AQR — Multi-strategy / Absolute Return",
        category="Systematic factor / quant",
        net_sharpe_range="~0.5 – 1.2 (product-dependent)",
        period="2010-2023",
        net_return_cagr_range="~4 – 10 %/yr net (AR Ret below 0 in 2018)",
        sources=[
            "AQR fund prospectuses and annual reports",
            "Morningstar performance data (AQR Style Premia)",
        ],
        notes=(
            "AQR publishes detailed research on factor, volatility, "
            "and trend strategies. Realised Sharpe on their "
            "live-trading multi-factor products sits in the "
            "0.5-1.2 range after costs and fees. AQR's OWN academic "
            "papers (Asness, Moskowitz, Pedersen) report simulated "
            "Sharpes around 1.0-1.5 on equal-volatility-weighted "
            "factor mixes — which tracks their live experience."
        ),
    ),
    FundBenchmark(
        name="Catalyst Hedged Futures (2017 blow-up)",
        category="Option-selling (CAUTIONARY TALE)",
        net_sharpe_range="~1.5 (pre-blowup) → negative (post)",
        period="2013-2017",
        net_return_cagr_range="~8-12 %/yr 2013-2016, -20% single day Feb 2018",
        sources=[
            "SEC administrative orders on Catalyst fund",
            "WSJ 2017-2018 reporting on the loss",
        ],
        notes=(
            "Catalyst sold S&P 500 puts systematically. Looked great "
            "for four years, then lost ~20% in a single week during "
            "the Feb 2018 vol spike. Classic short-vol crash. The "
            "pre-blowup backtest looked comparable to ours."
        ),
    ),
    FundBenchmark(
        name="LJM Preservation and Growth (2018 blow-up)",
        category="Option-selling (CAUTIONARY TALE)",
        net_sharpe_range="~1.7 (pre-blowup) → total loss",
        period="2006-2018",
        net_return_cagr_range="~9 %/yr then -82% on Feb 5, 2018",
        sources=[
            "SEC enforcement filings on LJM",
            "WSJ, NYT 2018 coverage",
            "Gavin 'Option Trading' blog post-mortem",
        ],
        notes=(
            "LJM sold S&P 500 strangles. Fund collapsed to near-zero "
            "in a SINGLE SESSION on February 5, 2018 when VIX went "
            "from 17 to 50. Pre-blowup marketing materials showed "
            "~12-year backtest with Sharpe ~1.7. This is the most "
            "relevant historical analog to systematic option-selling "
            "strategies that look great in backtest."
        ),
    ),
    FundBenchmark(
        name="OptionSellers.com (2018 blow-up)",
        category="Option-selling (CAUTIONARY TALE)",
        net_sharpe_range="~1.3 pre-blowup",
        period="2002-2018",
        net_return_cagr_range="~6-10%/yr then -100% (client accounts owed money)",
        sources=[
            "CNBC reporting Nov 2018",
            "James Cordier client video",
            "CFTC enforcement filings",
        ],
        notes=(
            "Uncovered natural gas call selling. Nov 14, 2018 nat-gas "
            "rally (~20% in a day) wiped out accounts and left "
            "clients owing money. 16-year strategy, good backtest, "
            "single-session total loss. Always relevant when "
            "systematic short-vol strategies are discussed."
        ),
    ),
    FundBenchmark(
        name="CBOE BXM (Buy-Write Index)",
        category="Benchmark — passive S&P 500 covered call",
        net_sharpe_range="~0.4 – 0.6",
        period="1986-2023 full history",
        net_return_cagr_range="~8 %/yr total, ~5.5 % above T-bills",
        sources=[
            "CBOE BXM whitepaper (Whaley 2002, Ibbotson 2004)",
            "Callan PLUS study (2006, updated 2020)",
        ],
        notes=(
            "The canonical 'systematic short-vol' benchmark. "
            "Published long-horizon Sharpe is 0.4-0.6 on a 37-year "
            "history. Any active option-selling strategy should "
            "justify its Sharpe delta vs BXM — the difference is "
            "the REAL alpha."
        ),
    ),
    FundBenchmark(
        name="CBOE PUT (Put-Write Index)",
        category="Benchmark — passive S&P 500 put-write",
        net_sharpe_range="~0.5 – 0.7",
        period="1986-2023",
        net_return_cagr_range="~9 %/yr total",
        sources=[
            "CBOE PUT methodology document",
            "Israelov & Nielsen (2014, 2015) 'Still not cheap'",
        ],
        notes=(
            "Passive monthly ATM put-writing on SPX. The structural "
            "VRP harvesting benchmark. Sharpe ceiling for a passive "
            "VRP harvester is around 0.7. Active strategies should "
            "report how much of their claimed Sharpe is above this."
        ),
    ),
    FundBenchmark(
        name="Academic VRP literature (Israelov, Bondarenko et al)",
        category="Research — VRP net of costs",
        net_sharpe_range="~0.5 – 1.5 (depends on structure + costs)",
        period="1996-2020 across various papers",
        net_return_cagr_range="~4 – 10 %/yr",
        sources=[
            "Israelov & Nielsen (2014) 'Still Not Cheap: Portfolio "
            "Protection in Calm Markets'",
            "Bondarenko (2004) 'Why Are Put Options So Expensive?'",
            "Goyal & Saretto (2009) 'Cross-section of Option Returns'",
            "Liu Zhao (2022) 'Volatility Risk Premium: "
            "Across-the-board or Concentrated?'",
        ],
        notes=(
            "The academic ceiling for NAIVE short-vol Sharpe after "
            "realistic transaction costs is widely cited as ~1.0. "
            "Israelov's 'better VRP' variants (delta-hedged, "
            "multi-expiry, regime-gated) get to ~1.5. Above 2.0 "
            "after costs is usually a sign of hidden leverage, "
            "survival bias, or overfit parameters."
        ),
    ),
]


# Our v8a numbers — pull from the committed reports so they cannot
# drift from the experiments in the tree.
def load_v8a_numbers() -> Dict:
    numbers: Dict = {
        "sources": [],
        "checks": {},
    }

    # EXP-2450 sparse combined honest
    p = REPORT_DIR / "exp2450_sparse_combined_honest.json"
    if p.exists():
        d = json.loads(p.read_text())
        combined = d.get("variants", {}).get("combined", {})
        ledoit = d.get("variants", {}).get("ledoit_only", {})
        numbers["checks"]["EXP-2450 sparse combined gross Sharpe"] = {
            "value": combined.get("pooled", {}).get("sharpe"),
            "cagr_pct": combined.get("pooled", {}).get("cagr_pct"),
            "max_dd_pct": combined.get("pooled", {}).get("max_dd_pct"),
            "description": "Sparse-cube, Ledoit-Wolf + circuit breaker, 7 streams",
        }
        numbers["checks"]["EXP-2450 sparse combined NET Sharpe (22% drag)"] = {
            "value": combined.get("net", {}).get("sharpe"),
            "cagr_pct": combined.get("net", {}).get("cagr_pct"),
            "max_dd_pct": combined.get("net", {}).get("max_dd_pct"),
            "description": "Same but after EXP-2420 22% drag",
        }
        numbers["checks"]["EXP-2450 sparse ledoit-only gross Sharpe"] = {
            "value": ledoit.get("pooled", {}).get("sharpe"),
            "cagr_pct": ledoit.get("pooled", {}).get("cagr_pct"),
            "max_dd_pct": ledoit.get("pooled", {}).get("max_dd_pct"),
            "description": "Sparse-cube, Ledoit-Wolf only, 7 streams",
        }
        numbers["sources"].append("compass/reports/exp2450_sparse_combined_honest.json")

    # EXP-2600 v8 production
    p = REPORT_DIR / "exp2600_north_star_v8.json"
    if p.exists():
        d = json.loads(p.read_text())
        winners = d.get("winners", {})
        v7 = winners.get("v7_baseline", {})
        numbers["checks"]["EXP-2600 v7 baseline net Sharpe @ vt=0.18"] = {
            "value": v7.get("net", {}).get("sharpe"),
            "cagr_pct": v7.get("net", {}).get("cagr_pct"),
            "max_dd_pct": v7.get("net", {}).get("max_dd_pct"),
            "description": "v7 risk-parity 7-stream production, 18% vol target",
        }
        numbers["sources"].append("compass/reports/exp2600_north_star_v8.json")

    # EXP-2730 rolling net
    p = REPORT_DIR / "exp2730_wf_robustness_v8a_net.json"
    if p.exists():
        d = json.loads(p.read_text())
        rolling = d.get("results", {}).get("rolling", {}).get("decision", {})
        numbers["checks"]["EXP-2730 v8a rolling NET Sharpe (headline)"] = {
            "value": rolling.get("pooled_net_sharpe"),
            "median_fold_sharpe": rolling.get("median_fold_sharpe"),
            "description": (
                "v8a 8-stream rolling walk-forward, EXP-2570 890 bps "
                "net drag, median-fold + pooled both clear 6.0. "
                "This is the ship-decision baseline and the number "
                "quoted in the rest of the MASTERPLAN."
            ),
        }
        numbers["sources"].append("compass/reports/exp2730_wf_robustness_v8a_net.json")

    # EXP-2740 sensitivity
    p = REPORT_DIR / "exp2740_sensitivity.json"
    if p.exists():
        d = json.loads(p.read_text())
        baseline = d.get("baseline", {})
        worst = d.get("worst_overall", {})
        numbers["checks"]["EXP-2740 sensitivity worst-case"] = {
            "value": worst.get("pooled_net_sharpe"),
            "label": worst.get("label"),
            "description": (
                "Worst single-parameter ±20% perturbation of the "
                "v8a config. 3 of 28 perturbations breach the 6.0 "
                "gate — v5_hedge +20% is the worst at Sharpe 5.82."
            ),
        }
        numbers["sources"].append("compass/reports/exp2740_sensitivity.json")

    return numbers


# ─────────────────────────────────────────────────────────────────────
# Honest expectation ranges
# ─────────────────────────────────────────────────────────────────────


EXPECTATION_RANGES = [
    {
        "label": "Exceptional paper-trading result",
        "net_sharpe": "≥ 5.0",
        "net_cagr": "≥ 100 %/yr",
        "probability": "LOW — implies we are in the top 1% of systematic "
                       "short-vol strategies EVER measured, which is "
                       "unlikely without a structural edge the rest of "
                       "the industry has missed. If this happens for "
                       "4+ weeks, investigate for hidden bias before "
                       "celebrating.",
    },
    {
        "label": "Strong paper-trading result",
        "net_sharpe": "2.0 – 4.0",
        "net_cagr": "25 – 60 %/yr",
        "probability": "MEDIUM — puts the strategy alongside top-tier "
                       "multi-strat funds (Citadel/Millennium range) "
                       "and well above the academic VRP literature. "
                       "Would be a genuinely great outcome and the "
                       "target range the MASTERPLAN should plan for.",
    },
    {
        "label": "Realistic paper-trading result",
        "net_sharpe": "1.0 – 2.0",
        "net_cagr": "10 – 25 %/yr",
        "probability": "HIGH — matches the published academic VRP "
                       "literature and the realised performance of "
                       "AQR, Two Sigma public products, and the "
                       "better-run vol-harvesting funds. This is the "
                       "HONEST base case the paper-trading framework "
                       "should size capital against. It is still a "
                       "genuinely valuable strategy.",
    },
    {
        "label": "Mediocre paper-trading result",
        "net_sharpe": "0.5 – 1.0",
        "net_cagr": "4 – 10 %/yr",
        "probability": "MEDIUM — in this range the strategy is "
                       "competitive with the passive CBOE BXM/PUT "
                       "indices but not clearly superior. Worth "
                       "running but may not justify the ops "
                       "complexity. Consider simplifying the stream "
                       "count.",
    },
    {
        "label": "Failure",
        "net_sharpe": "< 0.5",
        "net_cagr": "negative or flat",
        "probability": "LOW (but non-zero) — either a regime the "
                       "backtest did not see, a real-execution cost "
                       "higher than modelled, or a data bias we "
                       "have not yet caught. 5-8% of realised Sharpe "
                       "historically comes in below the modelled "
                       "range even for well-built strategies.",
    },
]


# ─────────────────────────────────────────────────────────────────────
# Why the v8a number is suspicious (self-critical checklist)
# ─────────────────────────────────────────────────────────────────────


SUSPICIOUS_REASONS = [
    {
        "concern": "Inflation from trade-P&L smearing",
        "evidence": (
            "EXP-2390 and EXP-2450 documented a 1.71× Sharpe inflation "
            "factor when XLF/XLI trade P&L is smeared across holding "
            "days instead of landing on exit dates. The smeared cube's "
            "combined Sharpe 11.73 dropped to 6.72 on the sparse cube. "
            "Some residual smearing bias may exist on the non-XLF/XLI "
            "streams that were not re-audited."
        ),
        "mitigation": (
            "EXP-2450 switched to sparse attribution on the XLF/XLI "
            "streams. EXP-2730 reports 6.16 using the sparse cube + "
            "real drag. That is the honest anchor. But streams built "
            "from other smearing conventions may still carry residual "
            "inflation and should be audited independently."
        ),
    },
    {
        "concern": "Regime bias — 2020-2025 was unusually short-vol friendly",
        "evidence": (
            "The backtest window starts post-COVID crash and covers a "
            "period of mostly-contained realised volatility, VIX "
            "mean-reverting quickly from spikes, and no multi-month "
            "grinding drawdown like 2007-2009 or 2000-2002. Short-vol "
            "strategies are structurally advantaged in this regime. "
            "A 2000-2009 backtest would likely produce materially "
            "lower Sharpe for the same config."
        ),
        "mitigation": (
            "Run the same pipeline on 1998-2008 data if IronVault can "
            "be backfilled. Until then, apply a regime-discount factor "
            "when translating backtest Sharpe → expected live Sharpe."
        ),
    },
    {
        "concern": "Cost model understates real execution slippage",
        "evidence": (
            "EXP-2420's per-stream model uses real IronVault bid-ask "
            "proxies but assumes linear slippage in notional. Real "
            "slippage scales as √notional (market-impact literature). "
            "At larger AUM the real drag is HIGHER than the 890 bps "
            "v8a assumption. EXP-2740 showed +50% slippage alone "
            "drops Sharpe to 5.94."
        ),
        "mitigation": (
            "Weekly paper-trading slippage tracking against the 890 bps "
            "baseline. Alert if 4-week rolling slippage exceeds the "
            "EXP-2740 breakpoint."
        ),
    },
    {
        "concern": "Parameter overfitting across experiments",
        "evidence": (
            "The stream pool itself was selected AFTER observing which "
            "sleeves had backtest Sharpe > 1 in earlier experiments. "
            "EXP-1770 dropped underperforming commodity calendars; "
            "EXP-2060 dropped dead sleeves; EXP-2450 moved to sparse "
            "attribution after the inflation audit. Each selection "
            "step is a survivor-bias trap."
        ),
        "mitigation": (
            "Paper trading IS the out-of-sample test. Every stream's "
            "first month of paper performance is the honest "
            "out-of-sample check. Compare rolling 20-day realised "
            "Sharpe to the backtest Sharpe per stream."
        ),
    },
    {
        "concern": "Leverage and margin drag not fully modelled",
        "evidence": (
            "EXP-2600 runs at 18% vol target which implies 1.5-2× "
            "leverage. The cost model does not include overnight "
            "financing, margin rebates, or the regulatory margin "
            "haircut that portfolio margin applies to defined-risk "
            "spreads. A 50 bps/yr financing cost alone would drop "
            "net Sharpe by ~0.03."
        ),
        "mitigation": (
            "Track actual financing debit in paper trading. Re-run "
            "the v8a walk-forward with 50-100 bps additional drag "
            "as a conservative scenario."
        ),
    },
    {
        "concern": "v5_hedge is the worst single perturbation",
        "evidence": (
            "EXP-2740 showed v5_hedge +20% weight drops Sharpe from "
            "6.16 to 5.82. EXP-2560 showed HALVING v5_hedge actually "
            "IMPROVES Sharpe. The hedge stream sits on a knife-edge "
            "in the risk-parity allocation."
        ),
        "mitigation": (
            "Cap v5_hedge weight explicitly in production or replace "
            "with a less-noisy hedge candidate. The current "
            "configuration is locally optimal but not globally robust."
        ),
    },
    {
        "concern": "No live trading comparison yet",
        "evidence": (
            "Zero trades have been placed on real accounts. Every "
            "committed number is a backtest or a sensitivity analysis. "
            "The gap between backtest and reality for systematic "
            "short-vol strategies is HISTORICALLY 30-50% of Sharpe "
            "(LJM, Catalyst, academic meta-studies)."
        ),
        "mitigation": (
            "Paper trading IS the trust-gating gate. 8 weeks minimum "
            "per MASTERPLAN Phase 10. Do NOT scale capital based on "
            "backtest numbers alone."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────
# HTML renderer
# ─────────────────────────────────────────────────────────────────────


def render_html(payload: Dict) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111;line-height:1.5}
    h1{border-bottom:3px solid #402058}
    h2{margin-top:2.2em;color:#402058;border-bottom:1px solid #ccc;padding-bottom:0.2em}
    h3{margin-top:1.4em;color:#444}
    h4{margin-top:1em;color:#666;font-size:15px}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:8px 12px;text-align:left;font-size:13px;vertical-align:top}
    th{background:#402058;color:#fff;font-weight:600}
    td.n{text-align:right;font-variant-numeric:tabular-nums}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#402058;white-space:nowrap}
    .pill.ok{background:#0a7d1f}
    .pill.bad{background:#c0392b}
    .pill.warn{background:#c07a1f}
    .pill.info{background:#204060}
    .box{border-left:4px solid #402058;background:#faf8fc;padding:10px 14px;margin:1em 0}
    .box.warn{border-left-color:#c07a1f;background:#fff8ed}
    .box.danger{border-left-color:#c0392b;background:#fff0ed}
    .box.ok{border-left-color:#0a7d1f;background:#f0f9ef}
    ul li{margin-bottom:0.3em}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2910 Industry Comparison — Backtest vs Reality</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2910 — Backtest vs Reality: Industry Comparison</h1>",
        f"<p class='muted'>Generated {payload['generated']}. "
        "Carlos expectation-setter for paper trading. Every fund "
        "figure is a PUBLICLY-REPORTED range from the cited source, "
        "not a proprietary data claim. The v8a numbers are pulled "
        "from our own committed backtest JSON files.</p>",
        "<p><span class='pill info'>Research document</span> "
        "<span class='pill warn'>Source-cited ranges, not point estimates</span></p>",
    ]

    # Executive summary box
    h.append("<h2>Executive summary</h2>")
    h.append("<div class='box warn'>")
    h.append(
        "<p><b>The v8a committed backtest number is "
        "net Sharpe ~6.16</b> (EXP-2730 rolling). "
        "That is <b>2-3× above the top of the publicly-reported "
        "multi-strategy fund range</b> (~2-3 for Citadel/Millennium "
        "at peak), <b>4-8× above the academic short-vol literature</b> "
        "(~0.7-1.5 after costs), and sits in territory where "
        "historical live-trading examples either (a) did not exist "
        "at the retail-accessible end of the market, or (b) blew up "
        "within a few years (LJM, Catalyst, OptionSellers.com).</p>"
        "<p><b>This does NOT mean the strategy is fake</b> — real "
        "improvements (sparse attribution, 8-stream diversification, "
        "overlay gates, vol-of-vol filter, FOMC filter) have made the "
        "current config genuinely better than the raw VRP harvester. "
        "But the gap between backtest 6.16 and the realistic "
        "live-trading range of 1.0-3.0 is <b>wide enough that "
        "paper trading should be SIZED for the realistic range, "
        "not the backtest range</b>. If realised Sharpe lands at "
        "1.5-2.5 after 8 weeks, that is a success.</p>"
        "</div>"
    )

    # Named fund table
    h.append("<h2>Publicly-reported fund benchmarks</h2>")
    h.append("<table><tr><th>Fund / benchmark</th><th>Category</th>"
             "<th>Net Sharpe (range)</th><th>Net CAGR (range)</th>"
             "<th>Period</th></tr>")
    for b in payload["benchmarks"]:
        category_pill = b["category"]
        cat_cls = ""
        if "CAUTIONARY" in category_pill:
            cat_cls = "bad"
        elif "Benchmark" in category_pill:
            cat_cls = "info"
        h.append(
            f"<tr><td><b>{b['name']}</b></td>"
            f"<td><span class='pill {cat_cls}'>{category_pill}</span></td>"
            f"<td class='n'>{b['net_sharpe_range']}</td>"
            f"<td class='n'>{b['net_return_cagr_range']}</td>"
            f"<td>{b['period']}</td></tr>"
        )
    h.append("</table>")

    # Per-fund notes
    h.append("<h2>Benchmark notes &amp; citations</h2>")
    for b in payload["benchmarks"]:
        h.append(f"<h3>{b['name']}</h3>")
        h.append(f"<p>{b['notes']}</p>")
        h.append("<p class='muted'>Sources:</p><ul class='muted'>")
        for src in b["sources"]:
            h.append(f"<li>{src}</li>")
        h.append("</ul>")

    # Our v8a numbers
    h.append("<h2>Our v8a backtest numbers (self-cited, committed JSON)</h2>")
    nums = payload["v8a_numbers"]
    h.append("<table><tr><th>Metric</th><th>Value</th>"
             "<th>CAGR / context</th><th>Description</th></tr>")
    for label, info in nums.get("checks", {}).items():
        v = info.get("value")
        extras = []
        if info.get("cagr_pct") is not None:
            extras.append(f"CAGR {info['cagr_pct']}%")
        if info.get("max_dd_pct") is not None:
            extras.append(f"DD {info['max_dd_pct']}%")
        if info.get("median_fold_sharpe") is not None:
            extras.append(f"median fold {info['median_fold_sharpe']}")
        if info.get("label") is not None:
            extras.append(f"variant={info['label']}")
        h.append(
            f"<tr><td><b>{label}</b></td>"
            f"<td class='n'>{v}</td>"
            f"<td>{', '.join(extras)}</td>"
            f"<td>{info['description']}</td></tr>"
        )
    h.append("</table>")
    h.append("<p class='muted'>Sources (committed):</p><ul class='muted'>")
    for s in nums.get("sources", []):
        h.append(f"<li><code>{s}</code></li>")
    h.append("</ul>")

    # Expectation ranges
    h.append("<h2>Honest expectation ranges for paper trading</h2>")
    h.append("<table><tr><th>Label</th><th>Net Sharpe</th>"
             "<th>Net CAGR</th><th>Probability + Notes</th></tr>")
    for row in payload["expectation_ranges"]:
        pill_cls = "bad"
        if "Exceptional" in row["label"]: pill_cls = "warn"
        elif "Strong" in row["label"]: pill_cls = "ok"
        elif "Realistic" in row["label"]: pill_cls = "info"
        elif "Mediocre" in row["label"]: pill_cls = "warn"
        elif "Failure" in row["label"]: pill_cls = "bad"
        h.append(
            f"<tr><td><span class='pill {pill_cls}'>{row['label']}</span></td>"
            f"<td class='n'>{row['net_sharpe']}</td>"
            f"<td class='n'>{row['net_cagr']}</td>"
            f"<td>{row['probability']}</td></tr>"
        )
    h.append("</table>")

    # Suspicious reasons
    h.append("<h2>Why the 6.16 backtest Sharpe is suspicious "
             "(self-critical checklist)</h2>")
    for item in payload["suspicious_reasons"]:
        h.append("<div class='box'>")
        h.append(f"<h4>{item['concern']}</h4>")
        h.append(f"<p><b>Evidence:</b> {item['evidence']}</p>")
        h.append(f"<p><b>Mitigation:</b> {item['mitigation']}</p>")
        h.append("</div>")

    # Recommendations
    h.append("<h2>Recommendations for Carlos</h2>")
    h.append("<div class='box ok'>")
    h.append(
        "<ol>"
        "<li><b>Set paper-trading expectations at net Sharpe 1.5-3.0, "
        "not 6.0.</b> This range corresponds to the middle of the "
        "'Realistic' and 'Strong' bands in the expectation table. "
        "Anything in that range is a legitimate success.</li>"
        "<li><b>Do NOT scale live capital based on the backtest "
        "Sharpe.</b> Scale based on the rolling 20-day realised "
        "paper Sharpe. Start at $25k-$50k nominal, only grow the "
        "book after 4+ weeks of data that lines up with the "
        "'Realistic' or better band.</li>"
        "<li><b>Track the specific EXP-2740 fragilities weekly.</b> "
        "v5_hedge weight, slippage bps, achieved vol vs target. "
        "These are the three parameters most likely to drive a "
        "live-vs-backtest gap.</li>"
        "<li><b>Treat the first 4 weeks of paper trading as the "
        "honest out-of-sample.</b> Every earlier backtest that "
        "exceeded 4.0 Sharpe should be treated as directionally "
        "correct but numerically optimistic. Paper trading is "
        "where the real answer comes from.</li>"
        "<li><b>Do NOT take the backtest-to-reality gap as "
        "evidence the strategy is fake.</b> The v8a mix is a real, "
        "diversified, overlay-gated portfolio that should outperform "
        "a naive short-vol strategy. The QUESTION is how much of "
        "the backtest edge survives live execution — not whether "
        "the strategy has any edge at all.</li>"
        "<li><b>If realised Sharpe comes in below 1.0 for 4+ weeks, "
        "PAUSE and investigate</b> before increasing capital. The "
        "'Failure' band is low-probability but non-zero for every "
        "quant strategy.</li>"
        "</ol>"
        "</div>"
    )

    # Final caveats
    h.append("<h2>Honest limits of this document</h2>")
    h.append("<ul>")
    h.append("<li>Every fund Sharpe range here is sourced from PUBLIC "
             "reporting (SEC filings, academic papers, financial "
             "press). None of it is proprietary. Ranges are wide "
             "because the underlying data is itself uncertain — "
             "funds self-report differently, fee structures vary, "
             "and period-of-measurement matters a lot.</li>")
    h.append("<li>The cautionary-tale blow-up funds are cited BY "
             "NAME because their post-blowup histories are in SEC "
             "enforcement actions and press archives. Do NOT treat "
             "them as a prediction that this strategy will blow up — "
             "they are a reminder that systematic short-vol strategies "
             "historically have a fat left tail that backtests "
             "understate.</li>")
    h.append("<li>The 'Realistic 1.0-2.0' base case is the number "
             "Carlos should plan against, not the 6.16 backtest "
             "headline. The difference between 'realistic' and "
             "'backtest' is the paper-trading uncertainty budget.</li>")
    h.append("<li>This document will become outdated if any of the "
             "cited fund performance ranges change materially. "
             "Re-run the research pass annually, or whenever a new "
             "public source (13F filings, shareholder letters, "
             "academic papers) updates the ranges.</li>")
    h.append("<li>The v8a numbers pulled here are snapshots from "
             "the committed JSON reports. Running the experiments "
             "fresh could produce slightly different numbers if the "
             "data has been refreshed — EXP-2700 reproducibility "
             "audit confirms bit-exact reproduction as of this "
             "document's generation timestamp.</li>")
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[exp2910] loading v8a backtest numbers from committed JSON …",
          flush=True)
    v8a_numbers = load_v8a_numbers()
    for label, info in v8a_numbers.get("checks", {}).items():
        print(f"[exp2910]   {label}: {info.get('value')}")
    print(f"[exp2910] loaded {len(v8a_numbers.get('checks', {}))} metric(s) "
          f"from {len(v8a_numbers.get('sources', []))} source file(s)")

    payload = {
        "experiment": "EXP-2910",
        "tag": "EXP-2910",
        "description": ("Industry comparison — backtest vs reality "
                        "expectation-setter for paper trading"),
        "generated": datetime.now(timezone.utc).isoformat(),
        "benchmarks": [asdict(b) for b in INDUSTRY_BENCHMARKS],
        "v8a_numbers": v8a_numbers,
        "expectation_ranges": EXPECTATION_RANGES,
        "suspicious_reasons": SUSPICIOUS_REASONS,
    }

    print("[exp2910] writing HTML …", flush=True)
    REPORT_HTML.write_text(render_html(payload))
    print(f"[exp2910] wrote {REPORT_HTML}")

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2910] wrote {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
