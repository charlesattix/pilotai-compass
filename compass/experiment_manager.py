"""
Experiment lifecycle manager — register, track, compare, and promote
experiments through a structured pipeline.

Lifecycle:  PROPOSED → RUNNING → COMPLETED → PROMOTED | KILLED

Integrates with MASTERPLAN North Star targets:
  - 55% avg annual return
  - Sharpe ratio ≥ 6
  - ≤ 30% max drawdown

Components:
  1. Experiment registration    (hypothesis, config, success criteria)
  2. Backtest execution hook    (callable adapter for unified_backtest)
  3. Result comparison          (vs North Star targets)
  4. Status tracking            (state machine with transitions)
  5. A/B testing framework      (variant comparison with stat significance)
  6. Versioning / reproducibility (config + data + code hash)
  7. Leaderboard                (ranked by Sharpe, return, drawdown)
  8. HTML dashboard

All methods are pure computation — no broker connections.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NORTH_STAR = {
    "annual_return": 0.55,
    "sharpe": 6.0,
    "max_drawdown": 0.30,
}


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    COMPLETED = "completed"
    PROMOTED = "promoted"
    KILLED = "killed"


VALID_TRANSITIONS = {
    ExperimentStatus.PROPOSED: {ExperimentStatus.RUNNING, ExperimentStatus.KILLED},
    ExperimentStatus.RUNNING: {ExperimentStatus.COMPLETED, ExperimentStatus.KILLED},
    ExperimentStatus.COMPLETED: {ExperimentStatus.PROMOTED, ExperimentStatus.KILLED},
    ExperimentStatus.PROMOTED: {ExperimentStatus.KILLED},
    ExperimentStatus.KILLED: set(),
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SuccessCriteria:
    """Thresholds an experiment must meet to pass."""
    min_sharpe: float = 1.0
    min_annual_return: float = 0.10
    max_drawdown: float = 0.30
    min_win_rate: float = 0.45
    min_trades: int = 20


@dataclass
class BacktestResult:
    """Standardised backtest output."""
    annual_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    total_pnl: float = 0.0
    daily_returns: Optional[pd.Series] = field(default=None, repr=False)


@dataclass
class ExperimentVersion:
    """Reproducibility snapshot."""
    config_hash: str = ""
    data_hash: str = ""
    code_hash: str = ""
    timestamp: Optional[datetime] = None


@dataclass
class Experiment:
    """Full experiment record."""
    id: str
    name: str
    hypothesis: str
    config: Dict[str, Any]
    criteria: SuccessCriteria
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    version: Optional[ExperimentVersion] = None
    result: Optional[BacktestResult] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class NorthStarComparison:
    """How an experiment compares to North Star targets."""
    experiment_id: str
    annual_return: float
    sharpe: float
    max_drawdown: float
    return_pct: float     # actual / target
    sharpe_pct: float
    drawdown_ok: bool
    meets_north_star: bool


@dataclass
class ABTestResult:
    """Statistical comparison of two experiment variants."""
    control_id: str
    variant_id: str
    control_sharpe: float
    variant_sharpe: float
    sharpe_diff: float
    p_value: float
    is_significant: bool
    winner: str              # control_id | variant_id | "tie"
    n_observations: int


@dataclass
class LeaderboardEntry:
    """Single row on the experiment leaderboard."""
    rank: int
    experiment_id: str
    name: str
    status: ExperimentStatus
    sharpe: float
    annual_return: float
    max_drawdown: float
    n_trades: int
    meets_criteria: bool
    meets_north_star: bool


# ---------------------------------------------------------------------------
# Core manager
# ---------------------------------------------------------------------------

class ExperimentManager:
    """Experiment lifecycle manager.

    Args:
        backtest_fn: Optional callable(config) -> BacktestResult for
                     automated backtest execution.
    """

    def __init__(
        self,
        backtest_fn: Optional[Callable[[Dict[str, Any]], BacktestResult]] = None,
    ) -> None:
        self.backtest_fn = backtest_fn
        self._experiments: Dict[str, Experiment] = {}

    # ------------------------------------------------------------------
    # 1. Registration
    # ------------------------------------------------------------------

    def register(
        self,
        id: str,
        name: str,
        hypothesis: str,
        config: Dict[str, Any],
        criteria: Optional[SuccessCriteria] = None,
        tags: Optional[List[str]] = None,
    ) -> Experiment:
        """Register a new experiment."""
        if id in self._experiments:
            raise ValueError(f"Experiment {id} already registered")

        version = self._compute_version(config)
        exp = Experiment(
            id=id, name=name, hypothesis=hypothesis,
            config=config,
            criteria=criteria or SuccessCriteria(),
            status=ExperimentStatus.PROPOSED,
            version=version,
            created_at=datetime.now(),
            tags=tags or [],
        )
        self._experiments[id] = exp
        logger.info("Registered experiment %s: %s", id, name)
        return exp

    # ------------------------------------------------------------------
    # 2. Status transitions
    # ------------------------------------------------------------------

    def transition(
        self, id: str, new_status: ExperimentStatus,
    ) -> Experiment:
        """Move experiment to a new status (validates transition)."""
        exp = self._get(id)
        valid = VALID_TRANSITIONS.get(exp.status, set())
        if new_status not in valid:
            raise ValueError(
                f"Invalid transition {exp.status.value} → {new_status.value} "
                f"for {id}. Valid: {[s.value for s in valid]}"
            )
        exp.status = new_status
        if new_status == ExperimentStatus.COMPLETED:
            exp.completed_at = datetime.now()
        return exp

    # ------------------------------------------------------------------
    # 3. Backtest execution
    # ------------------------------------------------------------------

    def run_backtest(self, id: str) -> BacktestResult:
        """Execute backtest for an experiment."""
        exp = self._get(id)
        if exp.status == ExperimentStatus.PROPOSED:
            self.transition(id, ExperimentStatus.RUNNING)

        if self.backtest_fn is None:
            raise RuntimeError("No backtest_fn configured")

        result = self.backtest_fn(exp.config)
        exp.result = result
        self.transition(id, ExperimentStatus.COMPLETED)
        return result

    def set_result(self, id: str, result: BacktestResult) -> None:
        """Manually set backtest result (for external execution)."""
        exp = self._get(id)
        exp.result = result
        if exp.status == ExperimentStatus.RUNNING:
            self.transition(id, ExperimentStatus.COMPLETED)

    # ------------------------------------------------------------------
    # 4. North Star comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_north_star(
        experiment_id: str, result: BacktestResult,
    ) -> NorthStarComparison:
        """Compare results against MASTERPLAN North Star targets."""
        ret_pct = result.annual_return / NORTH_STAR["annual_return"] if NORTH_STAR["annual_return"] > 0 else 0
        sharpe_pct = result.sharpe / NORTH_STAR["sharpe"] if NORTH_STAR["sharpe"] > 0 else 0
        dd_ok = result.max_drawdown <= NORTH_STAR["max_drawdown"]
        meets = (
            result.annual_return >= NORTH_STAR["annual_return"]
            and result.sharpe >= NORTH_STAR["sharpe"]
            and dd_ok
        )
        return NorthStarComparison(
            experiment_id=experiment_id,
            annual_return=result.annual_return,
            sharpe=result.sharpe,
            max_drawdown=result.max_drawdown,
            return_pct=ret_pct,
            sharpe_pct=sharpe_pct,
            drawdown_ok=dd_ok,
            meets_north_star=meets,
        )

    def check_criteria(self, id: str) -> bool:
        """Check if experiment meets its success criteria."""
        exp = self._get(id)
        if exp.result is None:
            return False
        r = exp.result
        c = exp.criteria
        return (
            r.sharpe >= c.min_sharpe
            and r.annual_return >= c.min_annual_return
            and r.max_drawdown <= c.max_drawdown
            and r.win_rate >= c.min_win_rate
            and r.n_trades >= c.min_trades
        )

    # ------------------------------------------------------------------
    # 5. A/B testing
    # ------------------------------------------------------------------

    def ab_test(
        self,
        control_id: str,
        variant_id: str,
        confidence: float = 0.95,
    ) -> ABTestResult:
        """Compare two experiments statistically.

        Uses Welch's t-test on daily returns if available,
        falls back to Sharpe comparison.
        """
        control = self._get(control_id)
        variant = self._get(variant_id)
        if control.result is None or variant.result is None:
            raise ValueError("Both experiments must have results")

        cr = control.result
        vr = variant.result

        # Try daily returns t-test
        p_value = 1.0
        n_obs = 0
        if cr.daily_returns is not None and vr.daily_returns is not None:
            c_rets = cr.daily_returns.dropna()
            v_rets = vr.daily_returns.dropna()
            if len(c_rets) >= 5 and len(v_rets) >= 5:
                t_stat, p_value = sp_stats.ttest_ind(v_rets, c_rets, equal_var=False)
                p_value = float(p_value) / 2  # one-sided
                if t_stat < 0:
                    p_value = 1 - p_value
                n_obs = len(c_rets) + len(v_rets)

        is_sig = p_value < (1 - confidence)
        diff = vr.sharpe - cr.sharpe
        if is_sig and diff > 0:
            winner = variant_id
        elif is_sig and diff < 0:
            winner = control_id
        else:
            winner = "tie"

        return ABTestResult(
            control_id=control_id, variant_id=variant_id,
            control_sharpe=cr.sharpe, variant_sharpe=vr.sharpe,
            sharpe_diff=diff, p_value=p_value,
            is_significant=is_sig, winner=winner, n_observations=n_obs,
        )

    # ------------------------------------------------------------------
    # 6. Versioning
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_version(
        config: Dict[str, Any],
        data_hash: str = "",
        code_hash: str = "",
    ) -> ExperimentVersion:
        """Compute reproducibility hashes."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        cfg_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return ExperimentVersion(
            config_hash=cfg_hash,
            data_hash=data_hash or "n/a",
            code_hash=code_hash or "n/a",
            timestamp=datetime.now(),
        )

    def update_version(
        self, id: str, data_hash: str = "", code_hash: str = "",
    ) -> ExperimentVersion:
        """Update version hashes for an experiment."""
        exp = self._get(id)
        exp.version = self._compute_version(exp.config, data_hash, code_hash)
        return exp.version

    # ------------------------------------------------------------------
    # 7. Leaderboard
    # ------------------------------------------------------------------

    def leaderboard(
        self,
        sort_by: str = "sharpe",
        include_killed: bool = False,
    ) -> List[LeaderboardEntry]:
        """Rank all experiments by performance metric."""
        entries: List[LeaderboardEntry] = []
        for exp in self._experiments.values():
            if not include_killed and exp.status == ExperimentStatus.KILLED:
                continue
            r = exp.result or BacktestResult()
            meets_c = self.check_criteria(exp.id) if exp.result else False
            ns = self.compare_north_star(exp.id, r) if exp.result else None
            entries.append(LeaderboardEntry(
                rank=0, experiment_id=exp.id, name=exp.name,
                status=exp.status, sharpe=r.sharpe,
                annual_return=r.annual_return,
                max_drawdown=r.max_drawdown, n_trades=r.n_trades,
                meets_criteria=meets_c,
                meets_north_star=ns.meets_north_star if ns else False,
            ))

        key_map = {
            "sharpe": lambda e: e.sharpe,
            "annual_return": lambda e: e.annual_return,
            "max_drawdown": lambda e: -e.max_drawdown,  # lower is better
            "n_trades": lambda e: e.n_trades,
        }
        entries.sort(key=key_map.get(sort_by, key_map["sharpe"]), reverse=True)
        for i, e in enumerate(entries):
            e.rank = i + 1
        return entries

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, id: str) -> Experiment:
        return self._get(id)

    def _get(self, id: str) -> Experiment:
        if id not in self._experiments:
            raise KeyError(f"Experiment {id} not found")
        return self._experiments[id]

    @property
    def experiments(self) -> Dict[str, Experiment]:
        return dict(self._experiments)

    def by_status(self, status: ExperimentStatus) -> List[Experiment]:
        return [e for e in self._experiments.values() if e.status == status]

    def by_tag(self, tag: str) -> List[Experiment]:
        return [e for e in self._experiments.values() if tag in e.tags]

    # ------------------------------------------------------------------
    # 8. HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_bar(
        labels: List[str], values: List[float], title: str,
        width: int = 650, height: int = 200, color: str = "#2980b9",
    ) -> str:
        if not values:
            return ""
        n = len(values)
        vmax = max(abs(v) for v in values) or 1.0
        pad_l, pad_b = 100, 40
        pw = width - pad_l - 20
        ph = height - 55 - pad_b
        bw = pw / max(n, 1) * 0.7
        gap = pw / max(n, 1)

        p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" style="background:#fff;border:1px solid #ddd;'
             f'border-radius:6px;margin:.5rem 0">']
        p.append(f'<text x="{width // 2}" y="18" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#1a1a2e">{title}</text>')
        for i in range(n):
            x = pad_l + i * gap + (gap - bw) / 2
            bh = abs(values[i]) / vmax * ph
            y = 30 + ph - bh
            c = "#27ae60" if values[i] >= 0 else "#e74c3c"
            if all(v >= 0 for v in values):
                c = color
            p.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="{bw:.0f}" '
                     f'height="{max(bh, 1):.0f}" fill="{c}" rx="3"/>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{height - 8:.0f}" text-anchor="middle" '
                     f'font-size="8" fill="#666">{labels[i]}</text>')
            p.append(f'<text x="{x + bw / 2:.0f}" y="{y - 3:.0f}" text-anchor="middle" '
                     f'font-size="9" fill="#333">{values[i]:.2f}</text>')
        p.append("</svg>")
        return "\n".join(p)

    def generate_report(
        self,
        output_path: str = "reports/experiment_dashboard.html",
    ) -> str:
        """HTML dashboard: leaderboard, status, comparison charts."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lb = self.leaderboard()

        # Status counts
        status_counts = {}
        for exp in self._experiments.values():
            status_counts[exp.status.value] = status_counts.get(exp.status.value, 0) + 1

        # Sharpe comparison chart
        sharpe_svg = ""
        if lb:
            labels = [e.experiment_id for e in lb[:10]]
            vals = [e.sharpe for e in lb[:10]]
            sharpe_svg = self._svg_bar(labels, vals, "Sharpe Ratio Comparison")

        # Leaderboard table
        lb_rows = []
        for e in lb:
            status_cls = "promoted" if e.status == ExperimentStatus.PROMOTED else (
                "killed" if e.status == ExperimentStatus.KILLED else "")
            ns_icon = "Y" if e.meets_north_star else ""
            lb_rows.append(
                f"<tr class='{status_cls}'><td>#{e.rank}</td>"
                f"<td style='text-align:left'>{e.experiment_id}</td>"
                f"<td style='text-align:left'>{e.name}</td>"
                f"<td>{e.status.value}</td>"
                f"<td>{e.sharpe:.2f}</td>"
                f"<td>{e.annual_return:.1%}</td>"
                f"<td>{e.max_drawdown:.1%}</td>"
                f"<td>{e.n_trades}</td>"
                f"<td>{'Y' if e.meets_criteria else ''}</td>"
                f"<td>{ns_icon}</td></tr>"
            )

        # Status summary
        status_html = " | ".join(f"{k}: {v}" for k, v in status_counts.items())

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Experiment Dashboard</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
tr.promoted {{ background: #d4edda; }}
tr.killed {{ color: #999; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
</style></head><body>
<h1>Experiment Dashboard</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Total Experiments:</strong> {len(self._experiments)} | {status_html}</p>
<p><strong>North Star Targets:</strong>
   Return &ge; {NORTH_STAR['annual_return']:.0%} |
   Sharpe &ge; {NORTH_STAR['sharpe']:.1f} |
   Max DD &le; {NORTH_STAR['max_drawdown']:.0%}</p>
</div>

<h2>Sharpe Comparison</h2>
{sharpe_svg}

<h2>Leaderboard</h2>
<table>
<tr><th>#</th><th style='text-align:left'>ID</th>
<th style='text-align:left'>Name</th><th>Status</th>
<th>Sharpe</th><th>Return</th><th>Max DD</th><th>Trades</th>
<th>Criteria</th><th>North Star</th></tr>
{''.join(lb_rows)}
</table>
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Experiment dashboard -> %s", path)
        return str(path)
