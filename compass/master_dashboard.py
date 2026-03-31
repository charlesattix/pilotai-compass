"""
compass/master_dashboard.py — Single-page COMPASS project overview.

Aggregates data from:
  - experiments/registry.json         (experiment metadata + status)
  - reports/stress_test_results.json  (MC simulation + crisis hedge impact)
  - experiments/pruned_production_validation.json  (model AUC, features)
  - experiments/pruned_features_benchmark.json     (full vs pruned comparison)
  - compass/crisis_hedge.py           (per-experiment hedge configs)
  - data/models/                      (model files + timestamps)
  - git log                           (phase completion milestones)

Usage::

    python3 -m compass.master_dashboard
    # → reports/master_dashboard.html
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "master_dashboard.html"


# ── Data loaders ─────────────────────────────────────────────────────────


def _load_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def load_registry() -> Dict:
    return _load_json(ROOT / "experiments" / "registry.json") or {}


def load_stress_results() -> Dict:
    return _load_json(ROOT / "reports" / "stress_test_results.json") or {}


def load_pruned_validation() -> Dict:
    return _load_json(ROOT / "experiments" / "pruned_production_validation.json") or {}


def load_pruned_benchmark() -> Dict:
    return _load_json(ROOT / "experiments" / "pruned_features_benchmark.json") or {}


def load_model_files() -> List[Dict]:
    """Scan data/models/ for .joblib files with metadata."""
    model_dir = ROOT / "data" / "models"
    models = []
    if model_dir.exists():
        for f in sorted(model_dir.glob("*.joblib")):
            stat = f.stat()
            models.append({
                "name": f.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    .strftime("%Y-%m-%d %H:%M UTC"),
            })
    return models


def load_git_log(n: int = 25) -> List[Dict]:
    """Get recent git commits as phase milestones."""
    try:
        result = subprocess.run(
            ["git", "log", f"--oneline", f"-{n}", "--format=%h|%s|%ai"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=10,
        )
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append({
                    "hash": parts[0],
                    "message": parts[1],
                    "date": parts[2][:10],
                })
        return commits
    except Exception as exc:
        logger.warning("git log failed: %s", exc)
        return []


def load_hedge_configs() -> Dict[str, Dict]:
    """Load per-experiment hedge config parameters."""
    try:
        from compass.crisis_hedge import (
            EXP400_HEDGE_CONFIG,
            EXP401_HEDGE_CONFIG,
            CrisisHedgeConfig,
        )
        def _cfg_to_dict(cfg: CrisisHedgeConfig) -> Dict:
            return {
                "vix_floor": cfg.vix_scale_floor,
                "vix_ceiling": cfg.vix_scale_ceiling,
                "stop_floor": cfg.vix_stop_floor,
                "stop_ceiling": cfg.vix_stop_ceiling,
                "base_stop": cfg.base_stop_multiplier,
                "min_stop": cfg.min_stop_multiplier,
                "hv_scale": cfg.high_vol_regime_scale,
                "backwardation_penalty": cfg.vix_ts_backwardation_penalty,
            }
        return {
            "EXP-400": _cfg_to_dict(EXP400_HEDGE_CONFIG),
            "EXP-401": _cfg_to_dict(EXP401_HEDGE_CONFIG),
        }
    except Exception as exc:
        logger.warning("Failed to load hedge configs: %s", exc)
        return {}


# ── HTML generation ──────────────────────────────────────────────────────


def _esc(s: str) -> str:
    """Minimal HTML escape."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _status_badge(status: str) -> str:
    colors = {
        "paper_trading": ("#d4edda", "#155724"),
        "in_development": ("#fff3cd", "#856404"),
        "retired": ("#f8d7da", "#721c24"),
    }
    bg, fg = colors.get(status, ("#e2e3e5", "#383d41"))
    label = status.replace("_", " ").title()
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 8px;'
        f'border-radius:3px;font-size:0.8em;font-weight:600">{label}</span>'
    )


def _metric_card(title: str, value: str, subtitle: str = "", color: str = "#1e293b") -> str:
    return (
        f'<div class="card">'
        f'<div class="card-title">{_esc(title)}</div>'
        f'<div class="card-value" style="color:{color}">{_esc(value)}</div>'
        f'{"<div class=card-sub>" + _esc(subtitle) + "</div>" if subtitle else ""}'
        f'</div>'
    )


def _pct(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}%"


def _pass_fail(val: float, threshold: float, lower_is_better: bool = True) -> str:
    if lower_is_better:
        ok = abs(val) <= threshold
    else:
        ok = val >= threshold
    icon = "PASS" if ok else "FAIL"
    color = "#16a34a" if ok else "#dc2626"
    return f'<span style="color:{color};font-weight:700">{icon}</span>'


def generate_html() -> str:
    """Generate the complete master dashboard HTML."""
    registry = load_registry()
    stress = load_stress_results()
    pruned_val = load_pruned_validation()
    pruned_bench = load_pruned_benchmark()
    models = load_model_files()
    git_log = load_git_log()
    hedge_cfgs = load_hedge_configs()

    experiments = registry.get("experiments", {})
    hedge_impact = stress.get("crisis_hedge_impact", [])
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Section 1: Executive summary cards ────────────────────────────
    active_exps = [e for e in experiments.values() if e.get("status") != "retired"]
    paper_trading = [e for e in active_exps if e.get("status") == "paper_trading"]

    # Model metrics
    val_agg = pruned_val.get("xgboost_walk_forward", {}).get("aggregate", {})
    ens_stats = pruned_val.get("ensemble_training_stats", {})
    wf_auc = val_agg.get("auc_mean", 0)
    ens_auc = ens_stats.get("test_auc", ens_stats.get("ensemble_test_auc", 0))
    n_features = pruned_val.get("feature_count", 0)

    # Best hedge metrics
    best_hedge_sharpe = 0
    best_hedge_dd = 100
    for h in hedge_impact:
        if h.get("hedged_sharpe", 0) > best_hedge_sharpe:
            best_hedge_sharpe = h["hedged_sharpe"]
        if h.get("hedged_p5_dd", 100) < best_hedge_dd:
            best_hedge_dd = h["hedged_p5_dd"]

    cards = "".join([
        _metric_card("Active Experiments", str(len(active_exps)),
                     f"{len(paper_trading)} paper trading"),
        _metric_card("Ensemble AUC", f"{ens_auc:.4f}" if ens_auc else "N/A",
                     "Production model (pruned 21 feat)", "#16a34a" if ens_auc > 0.85 else "#1e293b"),
        _metric_card("WF XGBoost AUC", f"{wf_auc:.4f}" if wf_auc else "N/A",
                     f"5-fold walk-forward ({n_features} features)"),
        _metric_card("Best Hedged Sharpe", f"{best_hedge_sharpe:.3f}",
                     "VIX-adaptive crisis hedge"),
        _metric_card("Best Hedged P5 DD", f"{best_hedge_dd:.1f}%",
                     f'{_pass_fail(best_hedge_dd, 30)} target: &le;30%'),
        _metric_card("Feature Count", f"{n_features}",
                     "31 &rarr; 21 (post-ablation)"),
    ])

    # ── Section 2: Experiment registry ────────────────────────────────
    exp_rows = ""
    for eid, exp in sorted(experiments.items()):
        if exp.get("status") == "retired":
            continue
        exp_rows += (
            f'<tr>'
            f'<td><strong>{_esc(eid)}</strong></td>'
            f'<td>{_esc(exp.get("name", ""))}</td>'
            f'<td>{_esc(exp.get("ticker", ""))}</td>'
            f'<td>{_status_badge(exp.get("status", "unknown"))}</td>'
            f'<td>{_esc(exp.get("live_since", "-"))}</td>'
            f'<td style="font-size:0.85em;color:#666;max-width:300px">'
            f'{_esc(exp.get("description", "")[:120])}</td>'
            f'</tr>'
        )

    # ── Section 3: Stress test / hedge results ────────────────────────
    hedge_rows = ""
    for h in hedge_impact:
        name = h.get("name", "")
        hedge_rows += (
            f'<tr>'
            f'<td><strong>{_esc(name)}</strong></td>'
            f'<td>{h.get("unhedged_sharpe", 0):.3f}</td>'
            f'<td style="font-weight:600;color:#16a34a">{h.get("hedged_sharpe", 0):.3f}</td>'
            f'<td>{_pct(h.get("unhedged_p5_dd", 0))}</td>'
            f'<td style="font-weight:600">{_pct(h.get("hedged_p5_dd", 0))} '
            f'{_pass_fail(h.get("hedged_p5_dd", 100), 30)}</td>'
            f'<td>{_pct(h.get("unhedged_crisis_dd", 0))}</td>'
            f'<td>{_pct(h.get("hedged_crisis_dd", 0))}</td>'
            f'</tr>'
        )

    # ── Section 4: Hedge config comparison ────────────────────────────
    hedge_cfg_rows = ""
    for name, cfg in hedge_cfgs.items():
        hedge_cfg_rows += (
            f'<tr>'
            f'<td><strong>{_esc(name)}</strong></td>'
            f'<td>{cfg["vix_floor"]}</td>'
            f'<td>{cfg["vix_ceiling"]}</td>'
            f'<td>{cfg["base_stop"]}x</td>'
            f'<td>{cfg["min_stop"]}x</td>'
            f'<td>{cfg["hv_scale"]}</td>'
            f'<td>{cfg["backwardation_penalty"]}</td>'
            f'</tr>'
        )

    # ── Section 5: Model diagnostics ──────────────────────────────────
    # Pruned benchmark comparison
    bench_meta = pruned_bench.get("metadata", {})
    xgb_full = pruned_bench.get("xgboost_full", {}).get("aggregate", {})
    xgb_pruned = pruned_bench.get("xgboost_pruned", {}).get("aggregate", {})
    ens_full = pruned_bench.get("ensemble_full", {}).get("aggregate", {})
    ens_pruned = pruned_bench.get("ensemble_pruned", {}).get("aggregate", {})

    def _bench_row(label, key, higher_better=True):
        fv = xgb_full.get(key, 0)
        pv = xgb_pruned.get(key, 0)
        efv = ens_full.get(key, 0)
        epv = ens_pruned.get(key, 0)
        d1 = pv - fv
        d2 = epv - efv
        s1 = "+" if d1 > 0 else ""
        s2 = "+" if d2 > 0 else ""
        g1 = "color:#16a34a" if (d1 > 0.001 if higher_better else d1 < -0.001) else ""
        g2 = "color:#16a34a" if (d2 > 0.001 if higher_better else d2 < -0.001) else ""
        return (
            f'<tr><td>{label}</td>'
            f'<td>{fv:.4f}</td><td>{pv:.4f}</td>'
            f'<td style="{g1};font-weight:600">{s1}{d1:.4f}</td>'
            f'<td>{efv:.4f}</td><td>{epv:.4f}</td>'
            f'<td style="{g2};font-weight:600">{s2}{d2:.4f}</td></tr>'
        )

    bench_rows = ""
    if xgb_full and xgb_pruned:
        bench_rows += _bench_row("AUC", "auc_mean")
        bench_rows += _bench_row("Accuracy", "accuracy_mean")
        bench_rows += _bench_row("Precision", "precision_mean")
        bench_rows += _bench_row("Recall", "recall_mean")
        bench_rows += _bench_row("Brier Score", "brier_score_mean", higher_better=False)

    # Walk-forward per-fold
    wf_folds = pruned_val.get("xgboost_walk_forward", {}).get("folds", [])
    fold_rows = ""
    for f in wf_folds:
        fold_rows += (
            f'<tr>'
            f'<td>Fold {f.get("fold", "")}</td>'
            f'<td>{f.get("test_period", "")}</td>'
            f'<td>{f.get("n_train", "")}</td>'
            f'<td>{f.get("n_test", "")}</td>'
            f'<td style="font-weight:600">{f.get("auc", 0):.4f}</td>'
            f'<td>{f.get("accuracy", 0):.4f}</td>'
            f'</tr>'
        )

    # Model files
    model_rows = ""
    for m in models:
        model_rows += (
            f'<tr>'
            f'<td><code>{_esc(m["name"])}</code></td>'
            f'<td>{m["size_kb"]} KB</td>'
            f'<td>{m["modified"]}</td>'
            f'</tr>'
        )

    # Production model stats
    sig_stats = pruned_val.get("signal_model_training_stats", {})
    prod_metrics = ""
    if ens_stats:
        prod_metrics += (
            f'<tr><td>EnsembleSignalModel</td>'
            f'<td>{ens_stats.get("test_auc", ens_stats.get("ensemble_test_auc", 0)):.4f}</td>'
            f'<td>{ens_stats.get("n_features", n_features)}</td>'
            f'<td>{ens_stats.get("n_train", "-")} / {ens_stats.get("n_test", "-")}</td>'
            f'<td>{ens_stats.get("timestamp", "-")[:10] if ens_stats.get("timestamp") else "-"}</td></tr>'
        )
    if sig_stats:
        prod_metrics += (
            f'<tr><td>SignalModel</td>'
            f'<td>{sig_stats.get("test_auc", 0):.4f}</td>'
            f'<td>{sig_stats.get("n_features", n_features)}</td>'
            f'<td>{sig_stats.get("n_train", "-")} / {sig_stats.get("n_test", "-")}</td>'
            f'<td>-</td></tr>'
        )

    # ── Section 6: Features ───────────────────────────────────────────
    try:
        from compass.features import PRUNED_FEATURES, PRUNED_REMOVED
        kept_list = "".join(f"<li><code>{f}</code></li>" for f in PRUNED_FEATURES)
        removed_list = "".join(f"<li><code>{f}</code></li>" for f in PRUNED_REMOVED)
    except ImportError:
        kept_list = "<li>Unable to load</li>"
        removed_list = "<li>Unable to load</li>"

    # ── Section 7: Phase milestones ───────────────────────────────────
    phase_keywords = ["Phase", "phase", "Deploy", "deploy", "Wire", "wire",
                      "Benchmark", "Optimize", "Integrate", "Add"]
    milestone_rows = ""
    for c in git_log:
        is_milestone = any(kw in c["message"] for kw in phase_keywords)
        style = 'font-weight:600' if is_milestone else 'color:#666'
        milestone_rows += (
            f'<tr style="{style}">'
            f'<td><code>{c["hash"]}</code></td>'
            f'<td>{c["date"]}</td>'
            f'<td>{_esc(c["message"][:90])}</td>'
            f'</tr>'
        )

    # ── Assemble HTML ─────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>COMPASS Master Dashboard</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
background:#f8fafc;color:#1e293b;line-height:1.5;padding:24px;max-width:1280px;margin:0 auto}}
h1{{font-size:1.6em;font-weight:700;margin-bottom:4px}}
h2{{font-size:1.15em;font-weight:600;margin:28px 0 12px;padding-bottom:6px;border-bottom:2px solid #e2e8f0}}
.subtitle{{color:#64748b;font-size:0.9em;margin-bottom:20px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px 20px;
min-width:170px;flex:1;box-shadow:0 1px 2px rgba(0,0,0,.04)}}
.card-title{{font-size:0.78em;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px}}
.card-value{{font-size:1.5em;font-weight:700}}
.card-sub{{font-size:0.8em;color:#94a3b8;margin-top:2px}}
table{{border-collapse:collapse;width:100%;font-size:0.88em;margin-bottom:16px}}
th{{background:#f1f5f9;padding:8px 10px;text-align:left;font-weight:600;border-bottom:2px solid #e2e8f0}}
td{{padding:7px 10px;border-bottom:1px solid #e2e8f0}}
tr:hover{{background:#f8fafc}}
code{{font-family:monospace;background:#f1f5f9;padding:1px 5px;border-radius:3px;font-size:0.88em}}
.cols-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.feature-list{{columns:2;font-size:0.85em;list-style:none;padding:0}}
.feature-list li{{padding:2px 0}}
.feature-list li code{{background:#f0fdf4;color:#166534}}
.removed-list li code{{background:#fef2f2;color:#991b1b;text-decoration:line-through}}
@media(max-width:800px){{.cards{{flex-direction:column}}.cols-2{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<h1>COMPASS Master Dashboard</h1>
<p class="subtitle">PilotAI Credit Spreads &mdash; Generated {now}</p>

<div class="cards">{cards}</div>

<h2>Experiment Registry</h2>
<table>
<thead><tr><th>ID</th><th>Name</th><th>Ticker</th><th>Status</th><th>Live Since</th><th>Description</th></tr></thead>
<tbody>{exp_rows}</tbody>
</table>

<h2>Stress Test &amp; Hedge Impact</h2>
<p style="font-size:0.85em;color:#64748b;margin-bottom:8px">
10,000 MC block-bootstrap paths. Crisis hedge: VIX-adaptive position sizing + stop tightening.
Target: hedged MC P5 DD &le; 30%.
</p>
<table>
<thead><tr>
<th>Experiment</th><th>Unhedged Sharpe</th><th>Hedged Sharpe</th>
<th>Unhedged P5 DD</th><th>Hedged P5 DD</th>
<th>Unhedged Crisis DD</th><th>Hedged Crisis DD</th>
</tr></thead>
<tbody>{hedge_rows}</tbody>
</table>

<h2>Hedge Configuration</h2>
<table>
<thead><tr>
<th>Experiment</th><th>VIX Floor</th><th>VIX Ceiling</th>
<th>Base Stop</th><th>Min Stop</th><th>HV Scale</th><th>Backwd. Penalty</th>
</tr></thead>
<tbody>{hedge_cfg_rows}</tbody>
</table>

<h2>Model Diagnostics</h2>

<h3 style="font-size:0.95em;margin:16px 0 8px">Production Models</h3>
<table>
<thead><tr><th>Model</th><th>Test AUC</th><th>Features</th><th>Train / Test Split</th><th>Trained</th></tr></thead>
<tbody>{prod_metrics}</tbody>
</table>

<h3 style="font-size:0.95em;margin:16px 0 8px">Model Files</h3>
<table>
<thead><tr><th>File</th><th>Size</th><th>Last Modified</th></tr></thead>
<tbody>{model_rows}</tbody>
</table>

<h3 style="font-size:0.95em;margin:16px 0 8px">Pruned vs Full Feature Benchmark (5-Fold Walk-Forward)</h3>
<table>
<thead><tr>
<th>Metric</th>
<th>XGB Full</th><th>XGB Pruned</th><th>XGB &Delta;</th>
<th>Ens Full</th><th>Ens Pruned</th><th>Ens &Delta;</th>
</tr></thead>
<tbody>{bench_rows}</tbody>
</table>

<h3 style="font-size:0.95em;margin:16px 0 8px">Walk-Forward Folds (XGBoost, 21 Pruned Features)</h3>
<table>
<thead><tr><th>Fold</th><th>Test Period</th><th>Train N</th><th>Test N</th><th>AUC</th><th>Accuracy</th></tr></thead>
<tbody>{fold_rows}</tbody>
</table>

<h2>Feature Set (21 Pruned)</h2>
<div class="cols-2">
<div>
<h3 style="font-size:0.9em;color:#166534;margin-bottom:6px">Kept (21)</h3>
<ul class="feature-list">{kept_list}</ul>
</div>
<div>
<h3 style="font-size:0.9em;color:#991b1b;margin-bottom:6px">Removed (10)</h3>
<ul class="feature-list removed-list">{removed_list}</ul>
</div>
</div>

<h2>Phase Milestones</h2>
<table>
<thead><tr><th>Commit</th><th>Date</th><th>Message</th></tr></thead>
<tbody>{milestone_rows}</tbody>
</table>

<hr style="margin:28px 0;border:none;border-top:1px solid #e2e8f0">
<p style="font-size:0.75em;color:#94a3b8">
Generated by <code>compass/master_dashboard.py</code> &mdash; {now}
</p>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    html = generate_html()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(html)
    logger.info("Dashboard written to %s (%d bytes)", REPORT_PATH, len(html))


if __name__ == "__main__":
    main()
