"""
Trade turnover and rebalancing cost optimizer.

Optimal rebalancing frequency analysis (daily/weekly/monthly),
transaction cost modeling (commission + spread + market impact),
turnover decomposition (signal-driven vs drift vs regime change),
net-of-cost performance, tax-lot optimization (FIFO/LIFO/tax-loss
harvesting), and HTML report.

Usage::

    from compass.turnover_optimizer import TurnoverOptimizer
    opt = TurnoverOptimizer(weights_df, returns_df)
    results = opt.analyze()
    opt.generate_report()
"""

from __future__ import annotations

import base64
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "reports" / "turnover_optimizer.html"

FREQUENCIES = ("daily", "weekly", "monthly")


# ── Data classes ────────────────────────────────────────────────────────


@dataclass
class CostModel:
    """Transaction cost parameters."""
    commission_per_contract: float = 0.65
    spread_cost_pct: float = 0.02        # half-spread as % of notional
    market_impact_bps: float = 2.0       # basis points per unit
    tax_rate_short: float = 0.35         # short-term gains
    tax_rate_long: float = 0.15          # long-term gains
    long_term_days: int = 365


@dataclass
class TurnoverSnapshot:
    """Turnover metrics at a single rebalance point."""
    date: str
    turnover: float          # fraction of portfolio traded (0-2)
    cost: float              # dollar cost of rebalance
    signal_driven: float     # fraction from signal changes
    drift_driven: float      # fraction from price drift
    regime_driven: float     # fraction from regime changes
    n_trades: int


@dataclass
class FrequencyResult:
    """Performance of a single rebalancing frequency."""
    frequency: str
    gross_return: float
    total_cost: float
    net_return: float
    net_sharpe: float
    avg_turnover: float
    n_rebalances: int
    cost_drag_bps: float     # annualized cost drag in bps


@dataclass
class TurnoverDecomposition:
    """Aggregate turnover decomposition."""
    total_turnover: float
    signal_pct: float
    drift_pct: float
    regime_pct: float
    avg_per_rebalance: float
    annualized: float


@dataclass
class TaxLot:
    """A single tax lot."""
    asset: str
    entry_date: str
    quantity: float
    cost_basis: float
    current_value: float
    gain_loss: float
    holding_days: int
    is_long_term: bool


@dataclass
class TaxOptResult:
    """Tax-lot optimization result."""
    method: str              # "fifo", "lifo", "tax_loss"
    realized_gains: float
    realized_losses: float
    net_tax: float
    lots_sold: int
    tax_savings_vs_fifo: float


@dataclass
class OptimalFrequency:
    """Recommended rebalancing frequency."""
    frequency: str
    net_sharpe: float
    net_return: float
    cost_drag_bps: float
    reason: str


# ── Cost computation ────────────────────────────────────────────────────


def compute_rebalance_cost(
    turnover: float,
    portfolio_value: float,
    n_positions: int,
    cost_model: CostModel,
) -> float:
    """Compute dollar cost of a rebalance.

    turnover: fraction of portfolio traded (0 = no trades, 2 = full flip)
    """
    notional_traded = turnover * portfolio_value
    if notional_traded <= 0:
        return 0.0

    # Commission: approximate contracts from notional
    avg_contract_value = max(portfolio_value / max(n_positions, 1), 100)
    contracts = notional_traded / avg_contract_value
    commission = contracts * cost_model.commission_per_contract

    # Spread cost
    spread = notional_traded * cost_model.spread_cost_pct / 100

    # Market impact
    impact = notional_traded * cost_model.market_impact_bps / 10_000

    return commission + spread + impact


# ── Optimizer ───────────────────────────────────────────────────────────


class TurnoverOptimizer:
    """Optimize rebalancing frequency and minimize turnover costs."""

    def __init__(
        self,
        target_weights: pd.DataFrame,
        returns: pd.DataFrame,
        regimes: Optional[pd.Series] = None,
        cost_model: Optional[CostModel] = None,
        starting_capital: float = 100_000,
    ) -> None:
        self.target_weights = target_weights.copy()
        self.returns = returns.copy()
        self.assets = list(returns.columns)
        self.regimes = regimes
        self.cost_model = cost_model or CostModel()
        self.starting_capital = starting_capital

        # Align
        common = target_weights.index.intersection(returns.index)
        self.target_weights = self.target_weights.loc[common]
        self.returns = self.returns.loc[common]
        if self.regimes is not None:
            self.regimes = self.regimes.reindex(common, fill_value="neutral")

        # Results
        self.frequency_results: List[FrequencyResult] = []
        self.snapshots: Dict[str, List[TurnoverSnapshot]] = {}
        self.decomposition: Optional[TurnoverDecomposition] = None
        self.tax_results: List[TaxOptResult] = []
        self.optimal: Optional[OptimalFrequency] = None

    @classmethod
    def from_csv(
        cls, weights_path: str, returns_path: str,
        regimes_path: Optional[str] = None, **kwargs: Any,
    ) -> "TurnoverOptimizer":
        w = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        r = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        reg = None
        if regimes_path:
            reg = pd.read_csv(regimes_path, index_col=0, parse_dates=True).iloc[:, 0]
        return cls(w, r, regimes=reg, **kwargs)

    # ── Public API ──────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        for freq in FREQUENCIES:
            snaps = self._simulate_frequency(freq)
            self.snapshots[freq] = snaps
            fr = self._compute_frequency_result(freq, snaps)
            self.frequency_results.append(fr)
        self.decomposition = self._decompose_turnover()
        self.tax_results = self._tax_lot_analysis()
        self.optimal = self._find_optimal()
        return {
            "frequency_results": self.frequency_results,
            "snapshots": self.snapshots,
            "decomposition": self.decomposition,
            "tax_results": self.tax_results,
            "optimal": self.optimal,
        }

    # ── Frequency simulation ────────────────────────────────────────────

    def _simulate_frequency(self, freq: str) -> List[TurnoverSnapshot]:
        """Simulate portfolio with given rebalance frequency."""
        n = len(self.returns)
        if n < 2:
            return []

        # Determine rebalance days
        if freq == "daily":
            rebal_mask = np.ones(n, dtype=bool)
        elif freq == "weekly":
            rebal_mask = np.zeros(n, dtype=bool)
            for i in range(0, n, 5):
                rebal_mask[i] = True
        else:  # monthly
            rebal_mask = np.zeros(n, dtype=bool)
            for i in range(0, n, 21):
                rebal_mask[i] = True
        rebal_mask[0] = True  # Always rebalance on first day

        # Simulate
        current_weights = self.target_weights.iloc[0].values.copy()
        snapshots: List[TurnoverSnapshot] = []

        for i in range(n):
            day_ret = self.returns.iloc[i].values

            if rebal_mask[i] and i > 0:
                target = self.target_weights.iloc[i].values
                turnover = float(np.sum(np.abs(target - current_weights)))

                # Decompose turnover
                sig, drift, regime = self._decompose_single(
                    current_weights, target, i,
                )

                cost = compute_rebalance_cost(
                    turnover, self.starting_capital,
                    len(self.assets), self.cost_model,
                )
                n_trades = int(np.sum(np.abs(target - current_weights) > 0.001))

                snapshots.append(TurnoverSnapshot(
                    date=str(self.returns.index[i]),
                    turnover=turnover, cost=cost,
                    signal_driven=sig, drift_driven=drift,
                    regime_driven=regime, n_trades=n_trades,
                ))
                current_weights = target.copy()
            else:
                # Drift: weights change due to returns
                drifted = current_weights * (1 + day_ret)
                total = drifted.sum()
                if total > 0:
                    current_weights = drifted / total

        return snapshots

    def _decompose_single(
        self, current: np.ndarray, target: np.ndarray, idx: int,
    ) -> Tuple[float, float, float]:
        """Decompose single-rebalance turnover into signal/drift/regime."""
        total = float(np.sum(np.abs(target - current)))
        if total < 1e-10:
            return 0.0, 0.0, 0.0

        # Signal: how much target changed from previous target
        if idx > 0:
            prev_target = self.target_weights.iloc[idx - 1].values
            signal_change = float(np.sum(np.abs(target - prev_target)))
        else:
            signal_change = 0.0

        # Regime: if regime changed, attribute some turnover
        regime_change = 0.0
        if self.regimes is not None and idx > 0:
            cur_regime = self.regimes.iloc[idx]
            prev_regime = self.regimes.iloc[idx - 1]
            if cur_regime != prev_regime:
                regime_change = total * 0.3  # attribute 30% to regime

        # Drift: remainder
        signal_part = min(signal_change, total)
        regime_part = min(regime_change, total - signal_part)
        drift_part = total - signal_part - regime_part

        return (
            signal_part / total,
            drift_part / total,
            regime_part / total,
        )

    # ── Frequency result ────────────────────────────────────────────────

    def _compute_frequency_result(
        self, freq: str, snaps: List[TurnoverSnapshot],
    ) -> FrequencyResult:
        # Gross return (buy-and-hold equivalent of target weights)
        port_ret = (self.returns * self.target_weights).sum(axis=1)
        gross = float(np.prod(1 + port_ret) - 1)

        total_cost = sum(s.cost for s in snaps)
        avg_turn = float(np.mean([s.turnover for s in snaps])) if snaps else 0

        # Net return: approximate by subtracting cost ratio
        cost_ratio = total_cost / self.starting_capital
        net = gross - cost_ratio

        # Net Sharpe
        n = len(port_ret)
        ann_ret = net * 252 / max(n, 1)
        vol = float(port_ret.std() * np.sqrt(252)) if port_ret.std() > 0 else 1
        net_sharpe = ann_ret / vol if vol > 0 else 0

        # Annualized cost drag in bps
        years = max(n / 252, 0.01)
        cost_drag = (total_cost / self.starting_capital) / years * 10_000

        return FrequencyResult(
            frequency=freq, gross_return=gross,
            total_cost=total_cost, net_return=net,
            net_sharpe=net_sharpe, avg_turnover=avg_turn,
            n_rebalances=len(snaps), cost_drag_bps=cost_drag,
        )

    # ── Aggregate decomposition ─────────────────────────────────────────

    def _decompose_turnover(self) -> TurnoverDecomposition:
        """Aggregate turnover decomposition across all frequencies."""
        # Use daily as the reference
        snaps = self.snapshots.get("daily", [])
        if not snaps:
            return TurnoverDecomposition(0, 0, 0, 0, 0, 0)

        total = sum(s.turnover for s in snaps)
        sig = sum(s.signal_driven * s.turnover for s in snaps)
        drift = sum(s.drift_driven * s.turnover for s in snaps)
        regime = sum(s.regime_driven * s.turnover for s in snaps)

        if total > 0:
            sig_pct = sig / total
            drift_pct = drift / total
            regime_pct = regime / total
        else:
            sig_pct = drift_pct = regime_pct = 0

        avg = total / len(snaps)
        n = len(self.returns)
        ann = total * 252 / max(n, 1)

        return TurnoverDecomposition(
            total_turnover=total, signal_pct=sig_pct,
            drift_pct=drift_pct, regime_pct=regime_pct,
            avg_per_rebalance=avg, annualized=ann,
        )

    # ── Tax-lot analysis ────────────────────────────────────────────────

    def _tax_lot_analysis(self) -> List[TaxOptResult]:
        """Simulate FIFO, LIFO, and tax-loss harvesting on synthetic lots."""
        n = len(self.returns)
        if n < 20:
            return []

        # Build synthetic lots from weight changes
        lots: List[TaxLot] = []
        for i in range(0, n, max(n // 10, 1)):
            for j, asset in enumerate(self.assets):
                w = float(self.target_weights.iloc[i].values[j])
                if w < 0.01:
                    continue
                value = self.starting_capital * w
                # Simulate P&L based on forward returns
                fwd = min(i + 30, n)
                fwd_ret = float(self.returns.iloc[i:fwd, j].sum())
                current = value * (1 + fwd_ret)
                days = min(30, n - i)
                lots.append(TaxLot(
                    asset=asset, entry_date=str(self.returns.index[i]),
                    quantity=w * 100, cost_basis=value,
                    current_value=current, gain_loss=current - value,
                    holding_days=days,
                    is_long_term=days >= self.cost_model.long_term_days,
                ))

        if not lots:
            return []

        results: List[TaxOptResult] = []

        for method in ("fifo", "lifo", "tax_loss"):
            if method == "fifo":
                sorted_lots = sorted(lots, key=lambda l: l.entry_date)
            elif method == "lifo":
                sorted_lots = sorted(lots, key=lambda l: l.entry_date, reverse=True)
            else:
                # Tax-loss: sell losers first
                sorted_lots = sorted(lots, key=lambda l: l.gain_loss)

            # Sell top 30% of lots
            n_sell = max(1, len(sorted_lots) // 3)
            sold = sorted_lots[:n_sell]
            gains = sum(l.gain_loss for l in sold if l.gain_loss > 0)
            losses = sum(l.gain_loss for l in sold if l.gain_loss < 0)

            # Tax: short-term rate on gains, losses offset
            net_gain = gains + losses  # losses are negative
            tax = max(net_gain, 0) * self.cost_model.tax_rate_short

            results.append(TaxOptResult(
                method=method, realized_gains=gains,
                realized_losses=losses, net_tax=tax,
                lots_sold=n_sell, tax_savings_vs_fifo=0,
            ))

        # Compute savings vs FIFO
        fifo_tax = results[0].net_tax
        for r in results[1:]:
            r.tax_savings_vs_fifo = fifo_tax - r.net_tax

        return results

    # ── Optimal frequency ───────────────────────────────────────────────

    def _find_optimal(self) -> OptimalFrequency:
        if not self.frequency_results:
            return OptimalFrequency("monthly", 0, 0, 0, "No data")

        best = max(self.frequency_results, key=lambda f: f.net_sharpe)
        return OptimalFrequency(
            frequency=best.frequency,
            net_sharpe=best.net_sharpe,
            net_return=best.net_return,
            cost_drag_bps=best.cost_drag_bps,
            reason=f"Highest net Sharpe ({best.net_sharpe:.2f}) after costs of {best.cost_drag_bps:.0f}bps/yr",
        )

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self, output: str = str(DEFAULT_OUTPUT)) -> str:
        if self.optimal is None:
            self.analyze()
        charts = self._render_charts()
        html = self._build_html(charts)
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return str(out.resolve())

    @staticmethod
    def _fig_to_b64(fig) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")

    def _render_charts(self) -> Dict[str, str]:
        return {
            "cost_curves": self._chart_cost_curves(),
            "turnover_attr": self._chart_turnover_attribution(),
            "frequency_comp": self._chart_frequency_comparison(),
        }

    def _chart_cost_curves(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.frequency_results:
            return ""
        fig, ax = plt.subplots(figsize=(8, 4))
        port_ret = (self.returns * self.target_weights).sum(axis=1)
        gross_equity = self.starting_capital * (1 + port_ret).cumprod()
        ax.plot(range(len(gross_equity)), gross_equity, color="#64748b", lw=0.8, label="Gross")

        colors = {"daily": "#dc2626", "weekly": "#f59e0b", "monthly": "#16a34a"}
        for fr in self.frequency_results:
            cost_ratio = fr.total_cost / self.starting_capital
            daily_drag = cost_ratio / max(len(port_ret), 1)
            adj_ret = port_ret - daily_drag
            eq = self.starting_capital * (1 + adj_ret).cumprod()
            ax.plot(range(len(eq)), eq, color=colors.get(fr.frequency, "#3b82f6"),
                    lw=1.0, label=f"{fr.frequency} (net)")

        ax.set_xlabel("Day"); ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Cost-Adjusted Return Curves", fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_turnover_attribution(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = self.decomposition
        if d is None or d.total_turnover == 0:
            return ""
        labels = ["Signal", "Drift", "Regime"]
        vals = [d.signal_pct, d.drift_pct, d.regime_pct]
        nonzero = [(l, v) for l, v in zip(labels, vals) if v > 0.001]
        if not nonzero:
            return ""
        labels, vals = zip(*nonzero)
        colors = ["#3b82f6", "#f59e0b", "#dc2626"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(vals, labels=labels, colors=colors[:len(vals)], autopct="%1.0f%%",
               startangle=90, textprops={"fontsize": 9})
        ax.set_title("Turnover Attribution", fontsize=11); fig.tight_layout()
        return self._fig_to_b64(fig)

    def _chart_frequency_comparison(self) -> str:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not self.frequency_results:
            return ""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        freqs = [f.frequency for f in self.frequency_results]
        net_sharpes = [f.net_sharpe for f in self.frequency_results]
        cost_drags = [f.cost_drag_bps for f in self.frequency_results]
        colors = ["#16a34a" if f == self.optimal.frequency else "#3b82f6" for f in freqs]

        ax1.bar(freqs, net_sharpes, color=colors, alpha=0.85)
        ax1.set_ylabel("Net Sharpe"); ax1.set_title("Net Sharpe by Frequency", fontsize=10)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2.bar(freqs, cost_drags, color=["#dc2626" if d > 50 else "#f59e0b" for d in cost_drags], alpha=0.85)
        ax2.set_ylabel("Cost Drag (bps/yr)"); ax2.set_title("Annual Cost Drag", fontsize=10)
        ax2.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        return self._fig_to_b64(fig)

    def _build_html(self, charts: Dict[str, str]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        opt = self.optimal or OptimalFrequency("", 0, 0, 0, "")
        dec = self.decomposition or TurnoverDecomposition(0, 0, 0, 0, 0, 0)

        opt_cls = "good"

        freq_rows = ""
        for f in sorted(self.frequency_results, key=lambda x: -x.net_sharpe):
            cls = "good" if f.frequency == opt.frequency else ""
            freq_rows += (
                f'<tr class="{cls}"><td>{f.frequency}</td><td>{f.n_rebalances}</td>'
                f'<td>{f.gross_return:+.2%}</td><td>${f.total_cost:,.0f}</td>'
                f'<td>{f.net_return:+.2%}</td><td>{f.net_sharpe:.2f}</td>'
                f'<td>{f.avg_turnover:.3f}</td><td>{f.cost_drag_bps:.0f}</td></tr>\n'
            )

        tax_rows = ""
        for t in self.tax_results:
            cls = "good" if t.tax_savings_vs_fifo > 0 else ""
            tax_rows += (
                f'<tr><td>{t.method.upper()}</td><td>{t.lots_sold}</td>'
                f'<td>${t.realized_gains:,.0f}</td><td>${t.realized_losses:,.0f}</td>'
                f'<td>${t.net_tax:,.0f}</td>'
                f'<td class="{cls}">${t.tax_savings_vs_fifo:,.0f}</td></tr>\n'
            )
        if not tax_rows:
            tax_rows = '<tr><td colspan="6" style="text-align:center;color:#64748b">Insufficient data</td></tr>'

        def _img(k):
            b = charts.get(k, "")
            return f'<div class="chart"><img src="data:image/png;base64,{b}" alt="{k}"></div>' if b else ""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Turnover Optimizer</title>
<style>
  body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; margin:0; padding:2em 3em; background:#f8fafc; color:#1e293b; }}
  h1 {{ color:#0f172a; border-bottom:2px solid #e2e8f0; padding-bottom:0.4em; }} h2 {{ color:#334155; margin-top:2em; }}
  .meta {{ color:#64748b; font-size:0.9em; margin-bottom:1.5em; }}
  .good {{ color:#16a34a; font-weight:600; }} .bad {{ color:#dc2626; font-weight:600; }}
  .kpi-row {{ display:flex; gap:1.2em; flex-wrap:wrap; margin:1.5em 0; }}
  .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em 1.5em; min-width:120px; flex:1; text-align:center; }}
  .kpi .value {{ font-size:1.5em; font-weight:700; }} .kpi .label {{ font-size:0.75em; color:#64748b; margin-top:0.2em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:0.88em; }}
  th {{ background:#f1f5f9; padding:8px 10px; text-align:left; border-bottom:2px solid #cbd5e1; font-weight:600; }}
  td {{ padding:6px 10px; border-bottom:1px solid #e2e8f0; text-align:right; }} td:first-child {{ text-align:left; }}
  .chart {{ background:#fff; border:1px solid #e2e8f0; border-radius:8px; padding:1em; margin:1.5em 0; text-align:center; }}
  .chart img {{ max-width:100%; height:auto; }}
  footer {{ margin-top:3em; padding-top:1em; border-top:1px solid #e2e8f0; font-size:0.8em; color:#94a3b8; }}
</style></head><body>
<h1>Turnover &amp; Rebalancing Cost Optimizer</h1>
<div class="meta">{len(self.returns)} days &middot; {len(self.assets)} assets &middot; Generated {now}</div>
<div class="kpi-row">
  <div class="kpi"><div class="value {opt_cls}">{opt.frequency}</div><div class="label">Optimal Frequency</div></div>
  <div class="kpi"><div class="value">{opt.net_sharpe:.2f}</div><div class="label">Net Sharpe</div></div>
  <div class="kpi"><div class="value">{opt.cost_drag_bps:.0f}bp</div><div class="label">Cost Drag/yr</div></div>
  <div class="kpi"><div class="value">{dec.annualized:.2f}</div><div class="label">Annual Turnover</div></div>
  <div class="kpi"><div class="value">{dec.signal_pct:.0%}</div><div class="label">Signal-Driven</div></div>
</div>
<h2>1. Cost-Adjusted Return Curves</h2>{_img("cost_curves")}
<h2>2. Frequency Comparison</h2>{_img("frequency_comp")}
<table><thead><tr><th>Frequency</th><th>Rebalances</th><th>Gross</th><th>Cost</th><th>Net</th><th>Net Sharpe</th><th>Avg TO</th><th>Drag (bps)</th></tr></thead>
<tbody>{freq_rows}</tbody></table>
<h2>3. Turnover Attribution</h2>{_img("turnover_attr")}
<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>
<tr><td>Total Turnover</td><td>{dec.total_turnover:.3f}</td></tr>
<tr><td>Signal-Driven</td><td>{dec.signal_pct:.0%}</td></tr>
<tr><td>Drift-Driven</td><td>{dec.drift_pct:.0%}</td></tr>
<tr><td>Regime-Driven</td><td>{dec.regime_pct:.0%}</td></tr>
<tr><td>Avg per Rebalance</td><td>{dec.avg_per_rebalance:.4f}</td></tr>
<tr><td>Annualized</td><td>{dec.annualized:.2f}</td></tr>
</tbody></table>
<h2>4. Tax-Lot Optimization</h2>
<table><thead><tr><th>Method</th><th>Lots Sold</th><th>Gains</th><th>Losses</th><th>Tax</th><th>Savings vs FIFO</th></tr></thead>
<tbody>{tax_rows}</tbody></table>
<h2>5. Recommendation</h2>
<p><strong>{opt.frequency.title()}</strong> rebalancing: {opt.reason}</p>
<footer>Generated by <code>compass/turnover_optimizer.py</code></footer>
</body></html>"""
        return html
