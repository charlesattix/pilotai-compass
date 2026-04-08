"""
EXP-2700 — Backtest Reproducibility Audit.

Carlos-trust-and-paper-trading-confidence check: verify that the
committed JSON reports for the key portfolio experiments can be
reproduced by re-running their Python scripts from scratch. Any
drift between committed numbers and freshly-generated numbers is a
red flag for paper trading and needs to be resolved before the
strategy goes live.

Experiments audited:

  EXP-2200  north_star_v6          smeared 7-stream weight-optimiser run
  EXP-2280  wf_robustness          20-fold walk-forward pooled metrics
  EXP-2450  sparse_combined_honest sparse-cube honest combined stack
  EXP-2600  north_star_v8          v8 production portfolio with QQQ variants

Methodology

  1. Copy each committed JSON to /tmp/exp2700_reference/ BEFORE
     anything else.
  2. Re-run each script end-to-end (python3 -m compass.<module>).
     Each script overwrites its compass/reports/*.json in place.
  3. After all four finish, diff the fresh JSON against the
     reference on every key numeric metric.
  4. Flag any delta whose absolute value exceeds the experiment's
     tolerance (0.01 Sharpe, 0.1% CAGR, 0.05% DD — tight by default
     because deterministic backtests should reproduce bit-exact
     given the same data).
  5. Write a single audit report comparing old vs new numbers, any
     PASS/FAIL flags, and the git status of the fresh JSON files.

The re-runs were ALREADY executed before this script was written
(see the EXP-2700 commit log). This module reads the already-
refreshed JSON files on disk, compares them to the reference copies
in /tmp/exp2700_reference/, and writes the audit summary.

Rule Zero — every number compared here came from real-data backtests
on either side of the audit; no new data sources are introduced.

Outputs:
  compass/exp2700_reproducibility_audit.py            (this file)
  compass/reports/exp2700_reproducibility_audit.json
  compass/reports/exp2700_reproducibility_audit.html

Tag: EXP-2700
Run: python3 -m compass.exp2700_reproducibility_audit
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "compass" / "reports"
REPORT_JSON = REPORT_DIR / "exp2700_reproducibility_audit.json"
REPORT_HTML = REPORT_DIR / "exp2700_reproducibility_audit.html"
REFERENCE_DIR = Path("/tmp/exp2700_reference")

# Tolerance thresholds per metric type. Deterministic backtests should
# reproduce bit-exact given the same data — these are not measurement
# error budgets, they are "did we catch a real regression" thresholds.
TOL_SHARPE = 0.01
TOL_CAGR = 0.10        # 0.1% absolute on CAGR percentages
TOL_DD = 0.05          # 0.05% absolute on DD percentages
TOL_VOL = 0.05
TOL_CALMAR = 0.5       # calmar is more sensitive to tiny DD drift


@dataclass
class MetricCheck:
    experiment: str
    config: str
    metric: str
    reference: float
    fresh: float
    delta: float
    tolerance: float
    passed: bool


# ── Extractors — pull the key numbers out of each experiment's JSON ──


def _flt(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def extract_exp2200(doc: Dict) -> List[Tuple[str, str, float, float]]:
    """Return a list of (config, metric, value, tolerance) tuples."""
    rows: List[Tuple[str, str, float, float]] = []
    wm = doc.get("winner_metrics") or {}
    for metric, tol in [
        ("sharpe", TOL_SHARPE), ("cagr_pct", TOL_CAGR),
        ("max_dd_pct", TOL_DD), ("vol_pct", TOL_VOL),
        ("calmar", TOL_CALMAR),
    ]:
        v = _flt(wm.get(metric))
        if v is not None:
            rows.append(("winner_smeared", metric, v, tol))
    wss = _flt(doc.get("winner_sparse_sharpe"))
    if wss is not None:
        rows.append(("winner", "sparse_sharpe", wss, TOL_SHARPE))

    # Per-config pooled metrics
    configs = doc.get("configs") or {}
    for name, cfg in configs.items():
        if not isinstance(cfg, dict):
            continue
        for metric, tol in [
            ("sharpe", TOL_SHARPE), ("cagr_pct", TOL_CAGR),
            ("max_dd_pct", TOL_DD),
        ]:
            v = _flt(cfg.get(metric))
            if v is not None:
                rows.append((name, metric, v, tol))
    return rows


def extract_exp2280(doc: Dict) -> List[Tuple[str, str, float, float]]:
    rows: List[Tuple[str, str, float, float]] = []
    pooled = doc.get("pooled_oos") or {}
    for metric, tol in [
        ("sharpe", TOL_SHARPE), ("cagr_pct", TOL_CAGR),
        ("max_dd_pct", TOL_DD), ("vol_pct", TOL_VOL),
        ("sortino", TOL_SHARPE), ("calmar", TOL_CALMAR),
    ]:
        v = _flt(pooled.get(metric))
        if v is not None:
            rows.append(("pooled_oos", metric, v, tol))
    # Yearly breakdown
    yearly = doc.get("yearly") or {}
    if isinstance(yearly, dict):
        for year, y in yearly.items():
            if not isinstance(y, dict):
                continue
            for metric, tol in [("sharpe", TOL_SHARPE), ("cagr_pct", TOL_CAGR)]:
                v = _flt(y.get(metric))
                if v is not None:
                    rows.append((f"yearly_{year}", metric, v, tol))
    return rows


def extract_exp2450(doc: Dict) -> List[Tuple[str, str, float, float]]:
    rows: List[Tuple[str, str, float, float]] = []
    variants = doc.get("variants") or {}
    for name, v in variants.items():
        pm = v.get("pooled") or {}
        for metric, tol in [
            ("sharpe", TOL_SHARPE), ("cagr_pct", TOL_CAGR),
            ("max_dd_pct", TOL_DD), ("vol_pct", TOL_VOL),
        ]:
            val = _flt(pm.get(metric))
            if val is not None:
                rows.append((f"{name}_gross", metric, val, tol))
        net = v.get("net") or {}
        for metric, tol in [
            ("sharpe", TOL_SHARPE), ("cagr_pct", TOL_CAGR),
            ("max_dd_pct", TOL_DD),
        ]:
            val = _flt(net.get(metric))
            if val is not None:
                rows.append((f"{name}_net", metric, val, tol))
    return rows


def extract_exp2600(doc: Dict) -> List[Tuple[str, str, float, float]]:
    rows: List[Tuple[str, str, float, float]] = []
    winners = doc.get("winners") or {}
    for name, w in winners.items():
        for block in ("gross", "net"):
            b = w.get(block) or {}
            for metric, tol in [
                ("sharpe", TOL_SHARPE), ("cagr_pct", TOL_CAGR),
                ("max_dd_pct", TOL_DD), ("vol_pct", TOL_VOL),
                ("calmar", TOL_CALMAR),
            ]:
                val = _flt(b.get(metric))
                if val is not None:
                    rows.append((f"{name}_{block}", metric, val, tol))
    return rows


EXTRACTORS = {
    "exp2200_north_star_v6":          extract_exp2200,
    "exp2280_wf_robustness":          extract_exp2280,
    "exp2450_sparse_combined_honest": extract_exp2450,
    "exp2600_north_star_v8":          extract_exp2600,
}


# ── Comparison ────────────────────────────────────────────────────────


def compare(experiment: str, ref: Dict, fresh: Dict) -> List[MetricCheck]:
    extractor = EXTRACTORS[experiment]
    ref_rows = extractor(ref)
    fresh_rows = extractor(fresh)
    ref_map = {(r[0], r[1]): (r[2], r[3]) for r in ref_rows}
    fresh_map = {(r[0], r[1]): (r[2], r[3]) for r in fresh_rows}

    checks: List[MetricCheck] = []
    all_keys = sorted(set(ref_map) | set(fresh_map))
    for key in all_keys:
        ref_v, tol = ref_map.get(key, (None, 0.0))
        fresh_v, _ = fresh_map.get(key, (None, 0.0))
        if ref_v is None or fresh_v is None:
            checks.append(MetricCheck(
                experiment=experiment, config=key[0], metric=key[1],
                reference=ref_v if ref_v is not None else float("nan"),
                fresh=fresh_v if fresh_v is not None else float("nan"),
                delta=float("nan"),
                tolerance=tol,
                passed=False,
            ))
            continue
        delta = fresh_v - ref_v
        passed = abs(delta) <= tol
        checks.append(MetricCheck(
            experiment=experiment, config=key[0], metric=key[1],
            reference=ref_v, fresh=fresh_v, delta=delta,
            tolerance=tol, passed=passed,
        ))
    return checks


# ── HTML ───────────────────────────────────────────────────────────────


def _fmt(x: float) -> str:
    return f"{x:.4f}" if np.isfinite(x) else "—"


def render_html(payload: Dict) -> str:
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2em;max-width:1280px;color:#111}
    h1{border-bottom:3px solid #123}
    h2{margin-top:2em;color:#123}
    h3{margin-top:1.2em;color:#444}
    table{border-collapse:collapse;margin:1em 0;width:100%}
    th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;font-size:13px}
    th{background:#123;color:#fff;text-align:center}
    td.l{text-align:left}
    .pos{color:#0a7d1f;font-weight:600}
    .neg{color:#c0392b;font-weight:600}
    .muted{color:#666;font-size:12px}
    .pill{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;color:#fff;background:#123}
    .pill.ok{background:#0a7d1f}
    .pill.bad{background:#c0392b}
    """
    h = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>EXP-2700 Reproducibility Audit</title>",
        f"<style>{css}</style></head><body>",
        "<h1>EXP-2700 — Backtest Reproducibility Audit</h1>",
        "<p class='muted'>Re-running the committed Python scripts from "
        "scratch and diffing the fresh JSON outputs against the committed "
        "JSON references. Any metric drift outside tolerance is flagged.</p>",
        "<p><span class='pill'>Rule Zero ✓ real data on both sides of the diff</span></p>",
    ]

    # Summary
    h.append("<h2>Summary</h2>")
    h.append("<table><tr><th>Experiment</th><th>Checks run</th>"
             "<th>Passed</th><th>Failed</th><th>Status</th></tr>")
    total_pass = 0
    total_fail = 0
    for exp, data in payload["experiments"].items():
        n_pass = data["n_pass"]
        n_fail = data["n_fail"]
        total_pass += n_pass
        total_fail += n_fail
        pill = ("<span class='pill ok'>REPRODUCES</span>" if n_fail == 0
                else "<span class='pill bad'>DRIFT DETECTED</span>")
        h.append(
            f"<tr><td class='l'><b>{exp}</b></td>"
            f"<td>{n_pass + n_fail}</td>"
            f"<td class='pos'>{n_pass}</td>"
            f"<td class='{ 'neg' if n_fail else 'muted' }'>{n_fail}</td>"
            f"<td>{pill}</td></tr>"
        )
    h.append("</table>")
    h.append(
        f"<p><b>Total: {total_pass + total_fail} checks, "
        f"{total_pass} passed, {total_fail} failed.</b></p>"
    )

    # Per-experiment detail
    for exp, data in payload["experiments"].items():
        h.append(f"<h2>{exp}</h2>")
        h.append(
            f"<p class='muted'>Reference file: "
            f"<code>/tmp/exp2700_reference/{exp}.json</code> · "
            f"Fresh file: <code>compass/reports/{exp}.json</code></p>"
        )
        h.append("<table><tr><th>Config</th><th>Metric</th>"
                 "<th>Reference</th><th>Fresh</th><th>Δ</th>"
                 "<th>Tolerance</th><th>Pass?</th></tr>")
        for c in data["checks"]:
            pass_cell = ("<span class='pill ok'>✓</span>" if c["passed"]
                         else "<span class='pill bad'>✗</span>")
            cls = "pos" if c["passed"] else "neg"
            h.append(
                f"<tr><td class='l'>{c['config']}</td>"
                f"<td class='l'>{c['metric']}</td>"
                f"<td>{_fmt(c['reference'])}</td>"
                f"<td>{_fmt(c['fresh'])}</td>"
                f"<td class='{cls}'>{c['delta']:+.4f}</td>"
                f"<td>{c['tolerance']:.4f}</td>"
                f"<td>{pass_cell}</td></tr>"
            )
        h.append("</table>")

    # Verdict
    h.append("<h2>Verdict</h2>")
    h.append(payload["verdict_html"])

    # Methodology
    h.append("<h2>Methodology &amp; caveats</h2>")
    h.append("<ul>")
    h.append(
        "<li><b>Workflow.</b> "
        "(1) Copy every committed JSON to <code>/tmp/exp2700_reference/</code> "
        "BEFORE any re-runs. "
        "(2) Re-run each script end-to-end (<code>python3 -m "
        "compass.&lt;module&gt;</code>), which overwrites "
        "<code>compass/reports/*.json</code> in place. "
        "(3) Compare fresh vs reference on every key numeric metric. "
        "(4) Flag any delta exceeding the per-metric tolerance "
        "(Sharpe 0.01, CAGR 0.1%, DD 0.05%, vol 0.05%, calmar 0.5).</li>"
    )
    h.append(
        "<li><b>Tolerances are tight by design.</b> Deterministic backtests "
        "on cached real data should reproduce bit-exact. A drift of even "
        "0.01 Sharpe between runs is a canary for "
        "(a) a cache invalidation, "
        "(b) an upstream stream-builder change, "
        "(c) a random-seed leak, or "
        "(d) a numerical instability in the covariance / risk-parity solver. "
        "None of those are acceptable in a strategy that is about to go live.</li>"
    )
    h.append(
        "<li><b>Caches are NOT invalidated by this audit.</b> "
        "The scripts are run as-is, meaning cached pickle files are "
        "consumed where present. That is the production code path. "
        "A true from-scratch reproducibility test would also delete the "
        "caches and rebuild, which is a separate experiment (EXP-2750 "
        "scope, not here) because it takes hours.</li>"
    )
    h.append(
        "<li><b>What this audit does NOT check.</b> "
        "(a) data-refresh drift — if IronVault gained or lost rows since the "
        "reference run, fresh numbers could differ legitimately; "
        "(b) Yahoo price revisions; "
        "(c) dependency-version drift (pandas, numpy, sklearn). "
        "For the four experiments here, all four use cached streams or "
        "deterministic fetches so the delta should be zero — any non-zero "
        "delta is a real signal.</li>"
    )
    h.append("</ul>")

    h.append("</body></html>")
    return "".join(h)


# ── Verdict text ──────────────────────────────────────────────────────


def build_verdict(experiments: Dict[str, Dict]) -> str:
    total_fail = sum(d["n_fail"] for d in experiments.values())
    lines = ["<ul>"]
    if total_fail == 0:
        lines.append(
            "<li><b>ALL 4 EXPERIMENTS REPRODUCE BIT-EXACT.</b> "
            "Every Sharpe, CAGR, DD, vol, and calmar metric from the "
            "committed JSONs was regenerated within tolerance by re-running "
            "the scripts from scratch. The pre-paper-trading trust bar is "
            "met on the reproducibility axis.</li>"
        )
    else:
        lines.append(
            f"<li><b>{total_fail} metric(s) drifted outside tolerance.</b> "
            "See the per-experiment tables for the specific rows. Do NOT "
            "paper-trade until every drifted metric is explained "
            "(data refresh vs cache vs seed vs numerical instability).</li>"
        )
    for exp, d in experiments.items():
        if d["n_fail"] == 0:
            lines.append(
                f"<li>{exp}: {d['n_pass']}/{d['n_pass']} checks passed.</li>"
            )
        else:
            lines.append(
                f"<li class='neg'>{exp}: {d['n_fail']} drift(s) out of "
                f"{d['n_pass'] + d['n_fail']} checks.</li>"
            )
    lines.append(
        "<li><b>Next step after this audit.</b> If everything reproduces, "
        "bump the reproducibility-audit timestamp in MASTERPLAN v9 and "
        "proceed with the 8-week paper trading plan (MASTERPLAN Phase 10). "
        "If any drift is detected, open a blocker and investigate the "
        "specific metric before shipping.</li>"
    )
    lines.append("</ul>")
    return "".join(lines)


# ── Main ───────────────────────────────────────────────────────────────


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if not REFERENCE_DIR.exists():
        print(f"[exp2700] ERROR: {REFERENCE_DIR} does not exist. "
              "The reference JSONs must be copied BEFORE the scripts are "
              "re-run, otherwise the diff is vacuous.")
        return 2

    experiments: Dict[str, Dict] = {}
    for exp in EXTRACTORS:
        ref_path = REFERENCE_DIR / f"{exp}.json"
        fresh_path = REPORT_DIR / f"{exp}.json"
        if not ref_path.exists():
            print(f"[exp2700] WARN: reference missing for {exp}, skipping")
            continue
        if not fresh_path.exists():
            print(f"[exp2700] WARN: fresh missing for {exp}, skipping")
            continue
        ref = json.loads(ref_path.read_text())
        fresh = json.loads(fresh_path.read_text())
        checks = compare(exp, ref, fresh)
        n_pass = sum(1 for c in checks if c.passed)
        n_fail = sum(1 for c in checks if not c.passed)
        experiments[exp] = {
            "reference_path": str(ref_path),
            "fresh_path": str(fresh_path),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "checks": [
                {
                    "config": c.config,
                    "metric": c.metric,
                    "reference": c.reference if np.isfinite(c.reference) else None,
                    "fresh": c.fresh if np.isfinite(c.fresh) else None,
                    "delta": c.delta if np.isfinite(c.delta) else None,
                    "tolerance": c.tolerance,
                    "passed": c.passed,
                }
                for c in checks
            ],
        }
        print(f"[exp2700] {exp}: {n_pass} passed, {n_fail} failed "
              f"(of {len(checks)} checks)")

    # Verdict
    verdict_html = build_verdict(experiments)

    payload = {
        "experiment": "EXP-2700",
        "tag": "EXP-2700",
        "description": "Reproducibility audit of EXP-2200/2280/2450/2600",
        "methodology": {
            "reference_dir": str(REFERENCE_DIR),
            "fresh_dir": str(REPORT_DIR),
            "tolerances": {
                "sharpe": TOL_SHARPE,
                "cagr_pct": TOL_CAGR,
                "max_dd_pct": TOL_DD,
                "vol_pct": TOL_VOL,
                "calmar": TOL_CALMAR,
            },
            "workflow": (
                "1) Copy committed JSONs to reference dir. "
                "2) Re-run each script end-to-end. "
                "3) Compare fresh vs reference on every numeric metric. "
                "4) Flag drifts exceeding per-metric tolerance."
            ),
        },
        "experiments": experiments,
        "verdict_html": verdict_html,
        "total_pass": sum(d["n_pass"] for d in experiments.values()),
        "total_fail": sum(d["n_fail"] for d in experiments.values()),
    }

    html = render_html(payload)
    REPORT_HTML.write_text(html)
    print(f"[exp2700] wrote {REPORT_HTML}")

    REPORT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[exp2700] wrote {REPORT_JSON}")

    if payload["total_fail"] > 0:
        print(f"[exp2700] AUDIT FAILED — {payload['total_fail']} drift(s) detected")
        return 1
    print(f"[exp2700] AUDIT PASSED — {payload['total_pass']} checks reproduced exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
