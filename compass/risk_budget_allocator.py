"""
Risk budget allocator — distributes portfolio risk across experiments.

Allocation methods:
  - risk_parity:  weights proportional to 1/vol  (equal risk contribution)
  - cvar_budget:  weights proportional to 1/CVaR  (penalise fat tails)

Regime-adaptive limits (uses VolRegime from vol_forecaster):
  LOW      → 120% of base budget
  NORMAL   → 100%
  HIGH     →  70%
  EXTREME  →  40%

Dynamic rebalancing triggers fire when any experiment's budget utilisation
exceeds a configurable threshold (default 90%).

Marginal risk contribution analysis decomposes total portfolio risk into
per-experiment contributions via the covariance-weighted formulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compass.vol_forecaster import VolRegime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REGIME_MULTIPLIERS: Dict[VolRegime, float] = {
    VolRegime.LOW: 1.20,
    VolRegime.NORMAL: 1.00,
    VolRegime.HIGH: 0.70,
    VolRegime.EXTREME: 0.40,
}

DEFAULT_REBALANCE_THRESHOLD = 0.90  # 90% utilisation triggers rebalance


class AllocationMethod(str, Enum):
    RISK_PARITY = "risk_parity"
    CVAR_BUDGET = "cvar_budget"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRiskProfile:
    """Risk characteristics of a single experiment / strategy."""
    name: str
    volatility: float          # annualised vol
    cvar_95: float             # 95% CVaR (positive = loss magnitude)
    current_var: float = 0.0   # VaR currently consumed
    current_position_size: float = 0.0  # fraction of account
    correlation_to_portfolio: float = 0.0
    returns: Optional[pd.Series] = field(default=None, repr=False)
    # per-experiment hard limits
    max_var: float = 0.05
    max_position_size: float = 0.10
    max_correlation_contribution: float = 0.50


@dataclass
class BudgetAllocation:
    """Allocation outcome for one experiment."""
    name: str
    weight: float              # fraction of total risk budget
    allocated_var: float       # effective VaR allowance
    utilisation: float         # current_var / allocated_var
    marginal_risk: float = 0.0 # marginal contribution to portfolio risk
    capped: bool = False
    cap_reason: str = ""


@dataclass
class RebalanceTrigger:
    """Records when a rebalance was triggered."""
    date: datetime
    experiment_name: str
    utilisation: float
    threshold: float
    regime: VolRegime
    effective_budget: float


@dataclass
class BudgetSnapshot:
    """Full allocation state at a point in time."""
    date: datetime
    regime: VolRegime
    regime_multiplier: float
    base_budget: float
    effective_budget: float
    allocations: List[BudgetAllocation] = field(default_factory=list)
    total_utilisation: float = 0.0
    triggers: List[RebalanceTrigger] = field(default_factory=list)


@dataclass
class RegimeAdjustment:
    """Tracks a regime transition and its budget impact."""
    date: datetime
    old_regime: VolRegime
    new_regime: VolRegime
    old_multiplier: float
    new_multiplier: float
    old_budget: float
    new_budget: float


@dataclass
class MarginalRiskContribution:
    """Per-experiment marginal risk contribution."""
    name: str
    marginal_risk: float      # d(portfolio_vol) / d(weight_i) * weight_i
    pct_contribution: float   # fraction of total portfolio vol


# ---------------------------------------------------------------------------
# Core allocator
# ---------------------------------------------------------------------------

class RiskBudgetAllocator:
    """Allocates portfolio risk budget across experiments.

    Args:
        base_budget: Total portfolio risk budget (e.g. 0.15 for 15%).
        method: Allocation method (risk_parity or cvar_budget).
        regime_multipliers: Override regime → multiplier mapping.
        rebalance_threshold: Utilisation level that triggers rebalance.
    """

    def __init__(
        self,
        base_budget: float = 0.15,
        method: AllocationMethod = AllocationMethod.RISK_PARITY,
        regime_multipliers: Optional[Dict[VolRegime, float]] = None,
        rebalance_threshold: float = DEFAULT_REBALANCE_THRESHOLD,
    ) -> None:
        self.base_budget = base_budget
        self.method = method
        self.regime_multipliers = regime_multipliers or dict(DEFAULT_REGIME_MULTIPLIERS)
        self.rebalance_threshold = rebalance_threshold
        self._current_regime: VolRegime = VolRegime.NORMAL
        self._snapshots: List[BudgetSnapshot] = []
        self._regime_history: List[RegimeAdjustment] = []
        self._all_triggers: List[RebalanceTrigger] = []

    # ------------------------------------------------------------------
    # Regime management
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> VolRegime:
        return self._current_regime

    @property
    def effective_budget(self) -> float:
        return self.base_budget * self.regime_multipliers.get(self._current_regime, 1.0)

    def set_regime(
        self, regime: VolRegime, date: Optional[datetime] = None,
    ) -> Optional[RegimeAdjustment]:
        """Update regime; returns adjustment record or None if unchanged."""
        if regime == self._current_regime:
            return None
        old = self._current_regime
        old_m = self.regime_multipliers.get(old, 1.0)
        new_m = self.regime_multipliers.get(regime, 1.0)
        rec = RegimeAdjustment(
            date=date or datetime.now(),
            old_regime=old, new_regime=regime,
            old_multiplier=old_m, new_multiplier=new_m,
            old_budget=self.base_budget * old_m,
            new_budget=self.base_budget * new_m,
        )
        self._current_regime = regime
        self._regime_history.append(rec)
        logger.info("Regime %s->%s: budget %.1f%%->%.1f%%",
                     old.value, regime.value,
                     rec.old_budget * 100, rec.new_budget * 100)
        return rec

    @property
    def regime_history(self) -> List[RegimeAdjustment]:
        return list(self._regime_history)

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _raw_weights(self, experiments: List[ExperimentRiskProfile]) -> np.ndarray:
        """Raw allocation weights before limit enforcement."""
        n = len(experiments)
        if n == 0:
            return np.array([])

        if self.method == AllocationMethod.RISK_PARITY:
            vols = np.array([max(e.volatility, 1e-8) for e in experiments])
            inv = 1.0 / vols
            return inv / inv.sum()

        if self.method == AllocationMethod.CVAR_BUDGET:
            cvars = np.array([max(e.cvar_95, 1e-8) for e in experiments])
            inv = 1.0 / cvars
            return inv / inv.sum()

        return np.ones(n) / n

    # ------------------------------------------------------------------
    # Per-experiment limit enforcement
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_limits(
        alloc: BudgetAllocation,
        profile: ExperimentRiskProfile,
        effective_budget: float,
    ) -> BudgetAllocation:
        """Clamp allocation to per-experiment hard limits."""
        reasons: List[str] = []

        # VaR cap
        if alloc.allocated_var > profile.max_var:
            alloc.allocated_var = profile.max_var
            alloc.weight = profile.max_var / effective_budget if effective_budget > 0 else 0.0
            reasons.append(f"max_var({profile.max_var:.2%})")

        # Position size flag
        if profile.current_position_size > profile.max_position_size:
            reasons.append(f"max_pos({profile.max_position_size:.0%})")

        # Correlation contribution cap
        if abs(profile.correlation_to_portfolio) > profile.max_correlation_contribution:
            scale = profile.max_correlation_contribution / max(
                abs(profile.correlation_to_portfolio), 1e-8)
            alloc.allocated_var *= scale
            alloc.weight *= scale
            reasons.append(f"max_corr({profile.max_correlation_contribution:.2f})")

        if reasons:
            alloc.capped = True
            alloc.cap_reason = "; ".join(reasons)

        # Utilisation
        if alloc.allocated_var > 0:
            alloc.utilisation = min(profile.current_var / alloc.allocated_var, 1.0)
        else:
            alloc.utilisation = 0.0

        return alloc

    # ------------------------------------------------------------------
    # Rebalance trigger detection
    # ------------------------------------------------------------------

    def _check_triggers(
        self, allocations: List[BudgetAllocation], date: datetime,
    ) -> List[RebalanceTrigger]:
        """Return triggers for any experiment exceeding the threshold."""
        triggers: List[RebalanceTrigger] = []
        for a in allocations:
            if a.utilisation >= self.rebalance_threshold:
                t = RebalanceTrigger(
                    date=date,
                    experiment_name=a.name,
                    utilisation=a.utilisation,
                    threshold=self.rebalance_threshold,
                    regime=self._current_regime,
                    effective_budget=self.effective_budget,
                )
                triggers.append(t)
                self._all_triggers.append(t)
                logger.warning(
                    "REBALANCE TRIGGER: %s utilisation %.1f%% >= %.1f%%",
                    a.name, a.utilisation * 100, self.rebalance_threshold * 100,
                )
        return triggers

    @property
    def all_triggers(self) -> List[RebalanceTrigger]:
        return list(self._all_triggers)

    def needs_rebalance(self, experiments: List[ExperimentRiskProfile]) -> bool:
        """Quick check: would any experiment trigger a rebalance?"""
        snap = self.allocate(experiments, record=False)
        return any(
            a.utilisation >= self.rebalance_threshold for a in snap.allocations
        )

    # ------------------------------------------------------------------
    # Marginal risk contribution
    # ------------------------------------------------------------------

    @staticmethod
    def marginal_risk_contributions(
        experiments: List[ExperimentRiskProfile],
        weights: Optional[np.ndarray] = None,
    ) -> List[MarginalRiskContribution]:
        """Compute marginal risk contribution for each experiment.

        Uses the covariance-based decomposition:
            MRC_i = w_i * (Cov @ w)_i / sigma_p

        If experiments carry return series, the covariance matrix is estimated
        from those series.  Otherwise falls back to a diagonal (vol-only)
        approximation using correlation_to_portfolio.
        """
        n = len(experiments)
        if n == 0:
            return []

        if weights is None:
            weights = np.ones(n) / n

        has_returns = all(
            e.returns is not None and len(e.returns) > 1 for e in experiments
        )

        if has_returns:
            # Build covariance from actual return series
            ret_df = pd.DataFrame(
                {e.name: e.returns for e in experiments}
            ).dropna()
            if len(ret_df) > 1:
                cov = ret_df.cov().values
            else:
                has_returns = False

        if not has_returns:
            # Diagonal approximation with cross-correlation hint
            vols = np.array([e.volatility for e in experiments])
            corrs = np.array([e.correlation_to_portfolio for e in experiments])
            # Build pseudo-covariance: diag(vol) @ C @ diag(vol) where C_ij ≈ corr_i * corr_j
            outer_corr = np.outer(corrs, corrs)
            np.fill_diagonal(outer_corr, 1.0)
            cov = np.outer(vols, vols) * outer_corr

        port_var = float(weights @ cov @ weights)
        port_vol = np.sqrt(max(port_var, 1e-16))

        cov_w = cov @ weights
        results: List[MarginalRiskContribution] = []
        for i, exp in enumerate(experiments):
            mrc = float(weights[i] * cov_w[i] / port_vol) if port_vol > 0 else 0.0
            pct = mrc / port_vol if port_vol > 0 else 0.0
            results.append(MarginalRiskContribution(
                name=exp.name, marginal_risk=mrc, pct_contribution=pct,
            ))
        return results

    # ------------------------------------------------------------------
    # Main allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        experiments: List[ExperimentRiskProfile],
        date: Optional[datetime] = None,
        record: bool = True,
    ) -> BudgetSnapshot:
        """Allocate the effective risk budget and check rebalance triggers.

        Args:
            experiments: Risk profiles for each experiment.
            date: Snapshot timestamp.
            record: If True, store snapshot in history.
        """
        dt = date or datetime.now()
        eff = self.effective_budget
        weights = self._raw_weights(experiments)

        # Marginal risk contributions
        mrcs = self.marginal_risk_contributions(experiments, weights)
        mrc_by_name = {m.name: m.marginal_risk for m in mrcs}

        allocations: List[BudgetAllocation] = []
        for i, exp in enumerate(experiments):
            w = float(weights[i]) if i < len(weights) else 0.0
            alloc = BudgetAllocation(
                name=exp.name,
                weight=w,
                allocated_var=w * eff,
                utilisation=0.0,
                marginal_risk=mrc_by_name.get(exp.name, 0.0),
            )
            alloc = self._enforce_limits(alloc, exp, eff)
            allocations.append(alloc)

        # Renormalise if capping pushed total >1
        total_w = sum(a.weight for a in allocations)
        if total_w > 1.0:
            for a in allocations:
                a.weight /= total_w
                a.allocated_var = a.weight * eff

        # Total utilisation
        total_alloc = sum(a.allocated_var for a in allocations)
        total_used = sum(e.current_var for e in experiments)
        total_util = total_used / total_alloc if total_alloc > 0 else 0.0

        # Rebalance triggers
        triggers = self._check_triggers(allocations, dt) if record else []

        snap = BudgetSnapshot(
            date=dt,
            regime=self._current_regime,
            regime_multiplier=self.regime_multipliers.get(self._current_regime, 1.0),
            base_budget=self.base_budget,
            effective_budget=eff,
            allocations=allocations,
            total_utilisation=total_util,
            triggers=triggers,
        )
        if record:
            self._snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def utilisation_by_experiment(
        self, experiments: List[ExperimentRiskProfile],
    ) -> Dict[str, float]:
        """Return {name: utilisation} for each experiment."""
        snap = self.allocate(experiments, record=False)
        return {a.name: a.utilisation for a in snap.allocations}

    def risk_contributions(
        self, experiments: List[ExperimentRiskProfile],
    ) -> Dict[str, float]:
        """Return {name: pct_contribution_to_portfolio_risk}."""
        weights = self._raw_weights(experiments)
        mrcs = self.marginal_risk_contributions(experiments, weights)
        return {m.name: m.pct_contribution for m in mrcs}

    @property
    def snapshots(self) -> List[BudgetSnapshot]:
        return list(self._snapshots)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_pie(
        slices: List[Tuple[str, float, str]],
        width: int = 300, height: int = 300, title: str = "",
    ) -> str:
        """Inline SVG pie chart. slices: [(label, fraction, colour)]."""
        if not slices or all(f <= 0 for _, f, _ in slices):
            return ""
        cx, cy, r = width // 2, height // 2 - 10, min(width, height) // 2 - 40
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" style="background:#fff;border:1px solid #ddd;'
            f'border-radius:6px;margin:.5rem 0">'
        ]
        if title:
            parts.append(
                f'<text x="{cx}" y="18" text-anchor="middle" font-size="13" '
                f'font-weight="bold" fill="#1a1a2e">{title}</text>'
            )
        angle = -90.0
        for label, frac, color in slices:
            if frac <= 0:
                continue
            s_rad = np.radians(angle)
            sweep = frac * 360.0
            e_rad = np.radians(angle + sweep)
            lg = 1 if sweep > 180 else 0
            x1 = cx + r * np.cos(s_rad)
            y1 = cy + r * np.sin(s_rad)
            x2 = cx + r * np.cos(e_rad)
            y2 = cy + r * np.sin(e_rad)
            parts.append(
                f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} '
                f'A{r},{r} 0 {lg} 1 {x2:.1f},{y2:.1f} Z" fill="{color}"/>'
            )
            mid = np.radians(angle + sweep / 2)
            lx = cx + r * 0.6 * np.cos(mid)
            ly = cy + r * 0.6 * np.sin(mid)
            parts.append(
                f'<text x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle" '
                f'font-size="10" fill="#fff" font-weight="bold">{frac:.0%}</text>'
            )
            angle += sweep
        # legend
        lx, ly = 10, height - 16
        for label, frac, color in slices:
            if frac <= 0:
                continue
            parts.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{color}"/>')
            parts.append(
                f'<text x="{lx + 14}" y="{ly + 9}" font-size="10" fill="#333">{label}</text>'
            )
            lx += max(len(label) * 7 + 24, 60)
        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _svg_util_bars(
        allocations: List[BudgetAllocation],
        rebalance_threshold: float = 0.90,
        width: int = 560,
    ) -> str:
        """Horizontal utilisation bars with rebalance threshold line."""
        if not allocations:
            return ""
        bar_h, gap, pad_l = 26, 6, 130
        height = len(allocations) * (bar_h + gap) + 30
        bar_area = width - pad_l - 20
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" style="background:#fff;border:1px solid #ddd;'
            f'border-radius:6px;margin:.5rem 0">'
        ]
        # threshold line
        tx = pad_l + rebalance_threshold * bar_area
        parts.append(
            f'<line x1="{tx:.0f}" y1="0" x2="{tx:.0f}" y2="{height - 20}" '
            f'stroke="#e74c3c" stroke-width="2" stroke-dasharray="4,3"/>'
        )
        parts.append(
            f'<text x="{tx:.0f}" y="{height - 6}" text-anchor="middle" '
            f'font-size="9" fill="#e74c3c">rebal {rebalance_threshold:.0%}</text>'
        )
        for i, a in enumerate(allocations):
            y = 8 + i * (bar_h + gap)
            parts.append(
                f'<text x="{pad_l - 6}" y="{y + bar_h * .7:.0f}" text-anchor="end" '
                f'font-size="11" fill="#333">{a.name}</text>'
            )
            parts.append(
                f'<rect x="{pad_l}" y="{y}" width="{bar_area}" '
                f'height="{bar_h}" fill="#eee" rx="4"/>'
            )
            fw = max(a.utilisation * bar_area, 0)
            c = "#27ae60" if a.utilisation < 0.7 else (
                "#e67e22" if a.utilisation < rebalance_threshold else "#e74c3c")
            parts.append(
                f'<rect x="{pad_l}" y="{y}" width="{fw:.1f}" '
                f'height="{bar_h}" fill="{c}" rx="4"/>'
            )
            parts.append(
                f'<text x="{pad_l + fw + 5:.0f}" y="{y + bar_h * .7:.0f}" '
                f'font-size="11" fill="#333">{a.utilisation:.0%}</text>'
            )
        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _svg_util_timeline(
        snapshots: List[BudgetSnapshot], width: int = 700, height: int = 200,
    ) -> str:
        """Line chart of total utilisation over time."""
        if len(snapshots) < 2:
            return ""
        utils = [s.total_utilisation for s in snapshots]
        y_min, y_max = 0.0, max(max(utils) * 1.15, 0.1)
        pad_l, pad_r, pad_t, pad_b = 50, 15, 25, 30
        pw = width - pad_l - pad_r
        ph = height - pad_t - pad_b
        n = len(utils)

        def tx(i: int) -> float:
            return pad_l + i / max(n - 1, 1) * pw

        def ty(v: float) -> float:
            return pad_t + (1 - (v - y_min) / (y_max - y_min)) * ph

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" style="background:#fff;border:1px solid #ddd;'
            f'border-radius:6px;margin:.5rem 0">'
        ]
        parts.append(
            f'<text x="{width // 2}" y="16" text-anchor="middle" font-size="12" '
            f'font-weight="bold" fill="#1a1a2e">Budget Utilisation Over Time</text>'
        )
        # gridlines
        for frac in [0.25, 0.5, 0.75, 1.0]:
            yy = ty(frac * y_max)
            parts.append(
                f'<line x1="{pad_l}" y1="{yy:.0f}" x2="{width - pad_r}" '
                f'y2="{yy:.0f}" stroke="#eee"/>'
            )
            parts.append(
                f'<text x="{pad_l - 4}" y="{yy + 4:.0f}" text-anchor="end" '
                f'font-size="9" fill="#999">{frac * y_max:.0%}</text>'
            )
        # line
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{tx(i):.1f},{ty(u):.1f}"
            for i, u in enumerate(utils)
        )
        parts.append(f'<path d="{d}" fill="none" stroke="#2980b9" stroke-width="2"/>')
        # trigger dots
        trigger_idxs = set()
        for i, s in enumerate(snapshots):
            if s.triggers:
                trigger_idxs.add(i)
                parts.append(
                    f'<circle cx="{tx(i):.1f}" cy="{ty(utils[i]):.1f}" r="5" '
                    f'fill="#e74c3c" stroke="#fff" stroke-width="1"/>'
                )
        if trigger_idxs:
            parts.append(
                f'<text x="{width - pad_r}" y="{height - 8}" text-anchor="end" '
                f'font-size="9" fill="#e74c3c">red = rebalance trigger</text>'
            )
        parts.append("</svg>")
        return "\n".join(parts)

    def generate_report(
        self,
        experiments: List[ExperimentRiskProfile],
        output_path: str = "reports/risk_budget.html",
    ) -> str:
        """HTML report: pie chart, utilisation bars + timeline, breakdown, regime history."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        snap = self.allocate(experiments)

        # --- Risk contribution pie ---
        palette = ["#2980b9", "#e74c3c", "#27ae60", "#e67e22", "#8e44ad",
                    "#1abc9c", "#d35400", "#2c3e50", "#f39c12", "#c0392b"]
        weights = self._raw_weights(experiments)
        mrcs = self.marginal_risk_contributions(experiments, weights)
        total_mrc = sum(abs(m.marginal_risk) for m in mrcs) or 1.0
        pie_slices = [
            (m.name, abs(m.marginal_risk) / total_mrc, palette[i % len(palette)])
            for i, m in enumerate(mrcs)
        ]
        pie_svg = self._svg_pie(pie_slices, title="Risk Contribution")

        # --- Utilisation bars ---
        util_svg = self._svg_util_bars(
            snap.allocations, self.rebalance_threshold)

        # --- Timeline ---
        timeline_svg = self._svg_util_timeline(self._snapshots)

        # --- Breakdown table ---
        table_rows = []
        for a in snap.allocations:
            cap_td = (f'<td class="capped">{a.cap_reason}</td>'
                      if a.capped else "<td>-</td>")
            table_rows.append(
                f"<tr><td>{a.name}</td>"
                f"<td>{a.weight:.2%}</td>"
                f"<td>{a.allocated_var:.4f}</td>"
                f"<td>{a.utilisation:.1%}</td>"
                f"<td>{a.marginal_risk:.6f}</td>"
                f"{cap_td}</tr>"
            )

        # --- Trigger log ---
        trig_rows = []
        for t in self._all_triggers:
            ds = t.date.strftime("%Y-%m-%d %H:%M") if hasattr(t.date, "strftime") else str(t.date)
            trig_rows.append(
                f"<tr><td>{ds}</td><td>{t.experiment_name}</td>"
                f"<td>{t.utilisation:.1%}</td><td>{t.threshold:.0%}</td>"
                f"<td>{t.regime.value.upper()}</td>"
                f"<td>{t.effective_budget:.2%}</td></tr>"
            )
        trigger_section = ""
        if trig_rows:
            trigger_section = f"""
<h2>Rebalance Triggers</h2>
<table><tr><th>Date</th><th>Experiment</th><th>Util</th><th>Threshold</th>
<th>Regime</th><th>Eff. Budget</th></tr>
{''.join(trig_rows)}</table>"""

        # --- Regime history ---
        regime_rows = []
        for r in self._regime_history:
            ds = r.date.strftime("%Y-%m-%d %H:%M") if hasattr(r.date, "strftime") else str(r.date)
            regime_rows.append(
                f"<tr><td>{ds}</td>"
                f"<td>{r.old_regime.value.upper()}</td>"
                f"<td>{r.new_regime.value.upper()}</td>"
                f"<td>{r.old_multiplier:.2f}</td>"
                f"<td>{r.new_multiplier:.2f}</td>"
                f"<td>{r.old_budget:.2%}</td>"
                f"<td>{r.new_budget:.2%}</td></tr>"
            )
        regime_section = ""
        if regime_rows:
            regime_section = f"""
<h2>Regime Adjustment History</h2>
<table><tr><th>Date</th><th>From</th><th>To</th><th>Old Mult</th>
<th>New Mult</th><th>Old Budget</th><th>New Budget</th></tr>
{''.join(regime_rows)}</table>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Risk Budget Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: .5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; font-weight: 600; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
td.capped {{ color: #e74c3c; font-weight: bold; text-align: left; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
.charts {{ display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: flex-start; }}
</style></head><body>
<h1>Risk Budget Report</h1>
<div class="summary">
<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p><strong>Regime:</strong> {snap.regime.value.upper()}
   (multiplier {snap.regime_multiplier:.2f}x)</p>
<p><strong>Base Budget:</strong> {snap.base_budget:.2%}
   &rarr; <strong>Effective:</strong> {snap.effective_budget:.2%}</p>
<p><strong>Total Utilisation:</strong> {snap.total_utilisation:.1%}</p>
<p><strong>Method:</strong> {self.method.value}</p>
<p><strong>Rebalance Threshold:</strong> {self.rebalance_threshold:.0%}</p>
</div>

<h2>Risk Contribution &amp; Utilisation</h2>
<div class="charts">
{pie_svg}
{util_svg}
</div>

{timeline_svg}

<h2>Per-Experiment Breakdown</h2>
<table>
<tr><th>Experiment</th><th>Weight</th><th>Alloc VaR</th>
<th>Utilisation</th><th>Marginal Risk</th><th>Cap Reason</th></tr>
{''.join(table_rows)}</table>

{trigger_section}
{regime_section}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Risk budget report -> %s", path)
        return str(path)
