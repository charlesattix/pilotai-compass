"""
Risk budget allocator — distributes portfolio risk across experiments.

Allocation methods:
  - risk_parity:   budget proportional to inverse-vol (equal risk contribution)
  - inverse_cvar:  budget proportional to 1/CVaR (less to fat-tailed exps)
  - equal_marginal: equal marginal contribution to portfolio risk

Dynamic regime adjustment (uses VolRegime from vol_forecaster):
  LOW      → 120% of base budget  (lean in)
  NORMAL   → 100%
  HIGH     →  70%
  EXTREME  →  40%  (capital preservation)

Per-experiment limits enforced on every allocation:
  - max VaR  (hard cap)
  - max position size  (as fraction of account)
  - max correlation contribution  (cap on how much one experiment can
    drag the portfolio via correlation)
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
# Regime → budget multiplier
# ---------------------------------------------------------------------------

DEFAULT_REGIME_MULTIPLIERS: Dict[VolRegime, float] = {
    VolRegime.LOW: 1.20,
    VolRegime.NORMAL: 1.00,
    VolRegime.HIGH: 0.70,
    VolRegime.EXTREME: 0.40,
}


class AllocationMethod(str, Enum):
    RISK_PARITY = "risk_parity"
    INVERSE_CVAR = "inverse_cvar"
    EQUAL_MARGINAL = "equal_marginal"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRiskProfile:
    """Risk profile for a single experiment / strategy."""
    name: str
    volatility: float           # annualised vol
    cvar_95: float              # 95% CVaR (positive = loss)
    current_var: float = 0.0    # current VaR consumed
    current_position_size: float = 0.0  # fraction of account
    correlation_to_portfolio: float = 0.0
    # hard limits
    max_var: float = 0.05       # max VaR allowed
    max_position_size: float = 0.10  # max fraction of account
    max_correlation_contribution: float = 0.50  # cap


@dataclass
class BudgetAllocation:
    """Allocation result for one experiment."""
    name: str
    allocated_budget: float     # fraction of total risk budget
    allocated_var: float        # VaR dollars or fraction
    utilisation: float          # current_var / allocated_var
    capped: bool = False        # True if any limit was binding
    cap_reason: str = ""


@dataclass
class BudgetSnapshot:
    """Full allocation snapshot at a point in time."""
    date: datetime
    regime: VolRegime
    regime_multiplier: float
    base_budget: float
    effective_budget: float     # base * multiplier
    allocations: List[BudgetAllocation] = field(default_factory=list)
    total_utilisation: float = 0.0


@dataclass
class RegimeAdjustmentRecord:
    """Tracks a regime change and its budget impact."""
    date: datetime
    old_regime: VolRegime
    new_regime: VolRegime
    old_multiplier: float
    new_multiplier: float
    old_budget: float
    new_budget: float


# ---------------------------------------------------------------------------
# Core allocator
# ---------------------------------------------------------------------------

class RiskBudgetAllocator:
    """Allocates portfolio risk budget across experiments.

    Args:
        base_budget: Total portfolio risk budget (e.g. 0.15 for 15%).
        method: Allocation method.
        regime_multipliers: Override default regime → multiplier map.
    """

    def __init__(
        self,
        base_budget: float = 0.15,
        method: AllocationMethod = AllocationMethod.RISK_PARITY,
        regime_multipliers: Optional[Dict[VolRegime, float]] = None,
    ) -> None:
        self.base_budget = base_budget
        self.method = method
        self.regime_multipliers = regime_multipliers or dict(DEFAULT_REGIME_MULTIPLIERS)
        self._current_regime: VolRegime = VolRegime.NORMAL
        self._snapshots: List[BudgetSnapshot] = []
        self._regime_history: List[RegimeAdjustmentRecord] = []

    # ------------------------------------------------------------------
    # Regime
    # ------------------------------------------------------------------

    @property
    def current_regime(self) -> VolRegime:
        return self._current_regime

    @property
    def effective_budget(self) -> float:
        return self.base_budget * self.regime_multipliers.get(self._current_regime, 1.0)

    def set_regime(self, regime: VolRegime, date: Optional[datetime] = None) -> Optional[RegimeAdjustmentRecord]:
        """Update the current regime and record the adjustment."""
        if regime == self._current_regime:
            return None
        old = self._current_regime
        old_mult = self.regime_multipliers.get(old, 1.0)
        new_mult = self.regime_multipliers.get(regime, 1.0)
        rec = RegimeAdjustmentRecord(
            date=date or datetime.now(),
            old_regime=old,
            new_regime=regime,
            old_multiplier=old_mult,
            new_multiplier=new_mult,
            old_budget=self.base_budget * old_mult,
            new_budget=self.base_budget * new_mult,
        )
        self._current_regime = regime
        self._regime_history.append(rec)
        logger.info(
            "Regime %s → %s: budget %.1f%% → %.1f%%",
            old.value, regime.value,
            rec.old_budget * 100, rec.new_budget * 100,
        )
        return rec

    # ------------------------------------------------------------------
    # Raw weight computation (before limit enforcement)
    # ------------------------------------------------------------------

    def _raw_weights(self, experiments: List[ExperimentRiskProfile]) -> np.ndarray:
        """Compute raw allocation weights using the configured method."""
        n = len(experiments)
        if n == 0:
            return np.array([])

        if self.method == AllocationMethod.RISK_PARITY:
            vols = np.array([max(e.volatility, 1e-8) for e in experiments])
            inv_vol = 1.0 / vols
            return inv_vol / inv_vol.sum()

        if self.method == AllocationMethod.INVERSE_CVAR:
            cvars = np.array([max(e.cvar_95, 1e-8) for e in experiments])
            inv_cvar = 1.0 / cvars
            return inv_cvar / inv_cvar.sum()

        if self.method == AllocationMethod.EQUAL_MARGINAL:
            # Equal marginal contribution ≈ inverse-vol * inverse-correlation
            vols = np.array([max(e.volatility, 1e-8) for e in experiments])
            corrs = np.array([max(abs(e.correlation_to_portfolio), 0.05) for e in experiments])
            inv_mc = 1.0 / (vols * corrs)
            return inv_mc / inv_mc.sum()

        # fallback: equal weight
        return np.ones(n) / n

    # ------------------------------------------------------------------
    # Limit enforcement
    # ------------------------------------------------------------------

    @staticmethod
    def _enforce_limits(
        alloc: BudgetAllocation,
        profile: ExperimentRiskProfile,
        effective_budget: float,
    ) -> BudgetAllocation:
        """Clamp an allocation to per-experiment limits."""
        reasons: List[str] = []

        # max VaR
        if alloc.allocated_var > profile.max_var:
            alloc.allocated_var = profile.max_var
            alloc.allocated_budget = profile.max_var / effective_budget if effective_budget > 0 else 0.0
            reasons.append(f"max_var({profile.max_var:.2%})")

        # max position size
        if profile.current_position_size > profile.max_position_size:
            reasons.append(f"max_pos({profile.max_position_size:.0%})")

        # max correlation contribution
        if abs(profile.correlation_to_portfolio) > profile.max_correlation_contribution:
            scale = profile.max_correlation_contribution / max(abs(profile.correlation_to_portfolio), 1e-8)
            alloc.allocated_var *= scale
            alloc.allocated_budget *= scale
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
    # Main allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        experiments: List[ExperimentRiskProfile],
        date: Optional[datetime] = None,
    ) -> BudgetSnapshot:
        """Allocate the effective risk budget across experiments.

        Returns a BudgetSnapshot with per-experiment allocations.
        """
        eff = self.effective_budget
        weights = self._raw_weights(experiments)

        allocations: List[BudgetAllocation] = []
        for i, exp in enumerate(experiments):
            w = float(weights[i]) if i < len(weights) else 0.0
            raw_var = w * eff
            alloc = BudgetAllocation(
                name=exp.name,
                allocated_budget=w,
                allocated_var=raw_var,
                utilisation=0.0,
            )
            alloc = self._enforce_limits(alloc, exp, eff)
            allocations.append(alloc)

        # Renormalise after capping so budgets sum to ≤1
        total_budget = sum(a.allocated_budget for a in allocations)
        if total_budget > 1.0:
            for a in allocations:
                a.allocated_budget /= total_budget
                a.allocated_var = a.allocated_budget * eff

        total_util = 0.0
        if allocations:
            total_alloc_var = sum(a.allocated_var for a in allocations)
            total_used_var = sum(
                exp.current_var for exp in experiments
            )
            total_util = total_used_var / total_alloc_var if total_alloc_var > 0 else 0.0

        snap = BudgetSnapshot(
            date=date or datetime.now(),
            regime=self._current_regime,
            regime_multiplier=self.regime_multipliers.get(self._current_regime, 1.0),
            base_budget=self.base_budget,
            effective_budget=eff,
            allocations=allocations,
            total_utilisation=total_util,
        )
        self._snapshots.append(snap)
        return snap

    # ------------------------------------------------------------------
    # Utilisation queries
    # ------------------------------------------------------------------

    def utilisation_by_experiment(
        self,
        experiments: List[ExperimentRiskProfile],
    ) -> Dict[str, float]:
        """Return {name: utilisation_fraction} for each experiment."""
        snap = self.allocate(experiments)
        return {a.name: a.utilisation for a in snap.allocations}

    @property
    def snapshots(self) -> List[BudgetSnapshot]:
        return list(self._snapshots)

    @property
    def regime_history(self) -> List[RegimeAdjustmentRecord]:
        return list(self._regime_history)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    @staticmethod
    def _svg_pie_chart(
        slices: List[Tuple[str, float, str]],
        width: int = 320,
        height: int = 320,
        title: str = "",
    ) -> str:
        """Inline SVG pie chart.  slices: [(label, fraction, color), ...]"""
        if not slices:
            return ""
        cx, cy, r = width // 2, height // 2 - 10, min(width, height) // 2 - 40
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                  f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:0.5rem 0">']
        if title:
            parts.append(f'<text x="{cx}" y="18" text-anchor="middle" font-size="13" '
                          f'font-weight="bold" fill="#1a1a2e">{title}</text>')

        angle = -90.0  # start at 12 o'clock
        for label, frac, color in slices:
            if frac <= 0:
                continue
            start_rad = np.radians(angle)
            sweep = frac * 360.0
            end_rad = np.radians(angle + sweep)
            large = 1 if sweep > 180 else 0

            x1 = cx + r * np.cos(start_rad)
            y1 = cy + r * np.sin(start_rad)
            x2 = cx + r * np.cos(end_rad)
            y2 = cy + r * np.sin(end_rad)

            parts.append(
                f'<path d="M{cx},{cy} L{x1:.1f},{y1:.1f} '
                f'A{r},{r} 0 {large} 1 {x2:.1f},{y2:.1f} Z" fill="{color}"/>'
            )

            # label at midpoint
            mid_rad = np.radians(angle + sweep / 2)
            lx = cx + (r * 0.65) * np.cos(mid_rad)
            ly = cy + (r * 0.65) * np.sin(mid_rad)
            parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                          f'font-size="10" fill="#fff" font-weight="bold">{frac:.0%}</text>')
            angle += sweep

        # Legend below
        lx = 10
        ly = height - 16
        for label, frac, color in slices:
            if frac <= 0:
                continue
            parts.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{color}"/>')
            parts.append(f'<text x="{lx + 14}" y="{ly + 9}" font-size="10" fill="#333">{label}</text>')
            lx += max(len(label) * 7 + 24, 60)
        parts.append("</svg>")
        return "\n".join(parts)

    @staticmethod
    def _svg_utilisation_bars(
        allocations: List[BudgetAllocation],
        width: int = 600,
    ) -> str:
        """Horizontal bar chart of utilisation per experiment."""
        if not allocations:
            return ""
        bar_h = 28
        gap = 6
        pad_l = 140
        height = len(allocations) * (bar_h + gap) + 30
        bar_area = width - pad_l - 20

        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                  f'style="background:#fff;border:1px solid #ddd;border-radius:6px;margin:0.5rem 0">']

        for i, a in enumerate(allocations):
            y = 10 + i * (bar_h + gap)
            parts.append(f'<text x="{pad_l - 8}" y="{y + bar_h * 0.7:.0f}" text-anchor="end" '
                          f'font-size="11" fill="#333">{a.name}</text>')
            # bg bar
            parts.append(f'<rect x="{pad_l}" y="{y}" width="{bar_area}" height="{bar_h}" '
                          f'fill="#eee" rx="4"/>')
            # filled bar
            fill_w = max(a.utilisation * bar_area, 0)
            color = "#27ae60" if a.utilisation < 0.7 else "#e67e22" if a.utilisation < 0.9 else "#e74c3c"
            parts.append(f'<rect x="{pad_l}" y="{y}" width="{fill_w:.1f}" height="{bar_h}" '
                          f'fill="{color}" rx="4"/>')
            parts.append(f'<text x="{pad_l + fill_w + 6:.1f}" y="{y + bar_h * 0.7:.0f}" '
                          f'font-size="11" fill="#333">{a.utilisation:.0%}</text>')

        parts.append("</svg>")
        return "\n".join(parts)

    def generate_report(
        self,
        experiments: List[ExperimentRiskProfile],
        output_path: str = "reports/risk_budget.html",
    ) -> str:
        """Write an HTML report for the current risk budget state."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        snap = self.allocate(experiments)

        # --- Pie chart ---
        palette = ["#2980b9", "#e74c3c", "#27ae60", "#e67e22", "#8e44ad",
                    "#1abc9c", "#d35400", "#2c3e50", "#f39c12", "#c0392b"]
        slices = [
            (a.name, a.allocated_budget, palette[i % len(palette)])
            for i, a in enumerate(snap.allocations)
        ]
        pie_svg = self._svg_pie_chart(slices, title="Risk Budget Allocation")

        # --- Utilisation bars ---
        util_svg = self._svg_utilisation_bars(snap.allocations)

        # --- Breakdown table ---
        table_rows = []
        for a in snap.allocations:
            cap_td = f'<td class="capped">{a.cap_reason}</td>' if a.capped else "<td>-</td>"
            table_rows.append(
                f"<tr><td>{a.name}</td>"
                f"<td>{a.allocated_budget:.2%}</td>"
                f"<td>{a.allocated_var:.4f}</td>"
                f"<td>{a.utilisation:.1%}</td>"
                f"{cap_td}</tr>"
            )

        # --- Regime history ---
        regime_rows = []
        for r in self._regime_history:
            dt_str = r.date.strftime("%Y-%m-%d %H:%M") if hasattr(r.date, "strftime") else str(r.date)
            regime_rows.append(
                f"<tr><td>{dt_str}</td>"
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
<table>
<tr><th>Date</th><th>From</th><th>To</th><th>Old Mult</th><th>New Mult</th>
<th>Old Budget</th><th>New Budget</th></tr>
{''.join(regime_rows)}
</table>"""

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Risk Budget Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       margin: 2rem; background: #f5f5f5; color: #1a1a2e; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 0.5rem; }}
h2 {{ color: #16213e; margin-top: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; background: #fff;
         border-radius: 6px; overflow: hidden; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: right; }}
th {{ background: #16213e; color: #fff; font-weight: 600; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
td.capped {{ color: #e74c3c; font-weight: bold; text-align: left; }}
.summary {{ background: #fff; padding: 1.2rem 1.5rem; border-radius: 8px;
            margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
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
</div>

<h2>Budget Allocation &amp; Utilisation</h2>
<div class="charts">
{pie_svg}
{util_svg}
</div>

<h2>Per-Experiment Breakdown</h2>
<table>
<tr><th>Experiment</th><th>Budget %</th><th>Allocated VaR</th>
<th>Utilisation</th><th>Cap Reason</th></tr>
{''.join(table_rows)}
</table>
{regime_section}
</body></html>"""

        path.write_text(html, encoding="utf-8")
        logger.info("Risk budget report written to %s", path)
        return str(path)
